import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from baseline import upscale
from v10.data import V10Dataset, inference_inputs, stratified_indices
from v10.diffusion import Schedule, compose, generate_residual, keyed_noise
from v10.model import ResidualDenoiser, align_features, build_v10, make_bilinear, validate_inputs


class V10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self, size=2):
        low = torch.rand(size, 2, 1, 4, 160)*0.5+0.25
        phase = torch.stack((torch.arange(size)%16, (torch.arange(size)-1)%16), dim=1)
        return low, make_bilinear(low, phase), phase

    def test_all_phase_alignment_and_cfa_colors(self):
        low = torch.rand(16, 1, 4, 160)
        phase = torch.arange(16)
        planes = F.pixel_unshuffle(low, 2)
        actual = F.pixel_shuffle(align_features(planes, phase), 2)
        torch.testing.assert_close(actual, upscale(low, phase))
        for p in range(16):
            # Every observed CFA-plane sample lands exactly at its phase position.
            aligned = align_features(planes[p:p+1], phase[p:p+1])
            torch.testing.assert_close(aligned[0, :, p//4::4, p%4::4], planes[p], atol=1e-5, rtol=1e-5)

    def test_dual_branch_temporal_gradients_and_checkpoint(self):
        model = ResidualDenoiser(channels=8)
        self.assertEqual(model.stem.in_channels, 60)
        low, high, phase = self.inputs()
        low.requires_grad_()
        high = high.detach().requires_grad_()
        with torch.no_grad():
            model.tail.weight.normal_(std=0.01)
        cond = model.prepare_condition(low, high, phase)
        output = model(torch.randn(2, 4, 8, 320), torch.tensor([100, 400]), cond)
        output.square().mean().backward()
        for frame in (0, 1):
            self.assertGreater(low.grad[:, frame].abs().sum().item(), 0)
            self.assertGreater(high.grad[:, frame].abs().sum().item(), 0)
        changed = low.detach().clone()
        changed[:, 1] *= 0.5
        self.assertGreater((model.prepare_condition(changed, high.detach(), phase)-cond.detach()).abs().max().item(), 0)
        config = {"architecture": "v10", "history": 2, "channels": 8, "condition": "both"}
        rebuilt = build_v10(config)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(rebuilt.prepare_condition(low, high, phase), cond)
        with self.assertRaisesRegex(ValueError, "bilinear does not match"):
            validate_inputs(low, high*0.5, phase, check_pair=True)

    def test_condition_masks(self):
        low, high, phase = self.inputs()
        for mode, section in (("decoded", slice(32, 40)), ("bilinear", slice(0, 32))):
            model = ResidualDenoiser(8, mode)
            condition = model.prepare_condition(low, high, phase)
            self.assertEqual(condition[:, section].abs().sum().item(), 0)

    def test_v_prediction_forward_inverse_including_terminal_timestep(self):
        schedule = Schedule()
        clean, noise = torch.randn(4, 4, 8, 320), torch.randn(4, 4, 8, 320)
        t = torch.tensor([0, 300, 900, 999])
        noisy = schedule.add_noise(clean, t, noise)
        velocity = schedule.training_target(clean, t, noise, "v")
        restored, epsilon = schedule.decode_prediction(noisy, t, velocity, "v")
        torch.testing.assert_close(restored, clean, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(epsilon, noise, atol=1e-6, rtol=1e-5)
        legacy = build_v10({"architecture": "v10", "history": 2, "channels": 8})
        self.assertEqual(legacy.prediction_type, "epsilon")
        current = build_v10({"architecture": "v10", "history": 2, "channels": 8, "prediction_type": "v"})
        self.assertEqual(current.prediction_type, "v")

    def test_noise_forward_inverse_and_physical_gamma_zero(self):
        schedule = Schedule()
        clean = torch.randn(3, 4, 8, 320)
        noise = torch.randn_like(clean)
        times = torch.tensor([0, 250, 700])
        noisy = schedule.add_noise(clean, times, noise)
        torch.testing.assert_close(schedule.clean_from_noise(noisy, times, noise), clean, atol=2e-6, rtol=1e-5)
        _, high, _ = self.inputs(3)
        torch.testing.assert_close(compose(high, clean, torch.ones(1, 4, 1, 1), 0), high[:, 0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            compose(high, clean, torch.ones(1, 4, 1, 1), float("nan"))

    def test_keyed_sampling_reproducibility_and_batch_partition(self):
        torch.manual_seed(13)
        model = ResidualDenoiser(8).eval()
        torch.nn.init.normal_(model.tail.weight, std=0.01)
        low, high, phase = self.inputs()
        schedule = Schedule(32)
        keys = [(16, 2), (17, 2)]
        noise = keyed_noise(keys, 0, "video", "cpu")
        torch.testing.assert_close(noise[:1], keyed_noise(keys[:1], 0, "video", "cpu"), rtol=0, atol=0)
        self.assertGreater((noise-keyed_noise(keys, 1, "video", "cpu")).abs().mean().item(), 0)
        fixed = keyed_noise(keys, 0, "video", "cpu", "fixed_screen")
        torch.testing.assert_close(fixed[0], fixed[1], rtol=0, atol=0)
        full = generate_residual(model, schedule, low, high, phase, keys, video_id="video", steps=4)
        individual = torch.cat([generate_residual(model, schedule, low[i:i+1], high[i:i+1], phase[i:i+1],
                               keys[i:i+1], video_id="video", steps=4) for i in range(2)])
        torch.testing.assert_close(full, individual, atol=2e-4, rtol=2e-4)
        self.assertTrue(torch.isfinite(full).all())
        with self.assertRaisesRegex(ValueError, "steps"):
            generate_residual(model, schedule, low, high, phase, keys, steps=0)

    def test_ddim_oracle_recovers_known_residual(self):
        schedule = Schedule(32)
        low, high, phase = self.inputs(1)
        truth = torch.randn(1, 4, 8, 320)*0.1
        def oracle(noisy, t, condition):
            a = schedule.alpha(t)
            return (noisy-a.sqrt()*truth)/(1-a).sqrt()
        fake = SimpleNamespace(training=False, residual_scale=torch.ones(1, 4, 1, 1)*0.1,
                               prepare_condition=lambda *args: None)
        class Oracle:
            training = False
            residual_scale = fake.residual_scale
            prepare_condition = staticmethod(fake.prepare_condition)
            __call__ = staticmethod(oracle)
        actual = generate_residual(Oracle(), schedule, low, high, phase, [(16, 0)], steps=8)
        torch.testing.assert_close(actual, truth, atol=2e-4, rtol=2e-4)

    def test_target_free_inference_and_split_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.sqlite"
            db = sqlite3.connect(path)
            db.executescript("PRAGMA user_version=4; CREATE TABLE samples "
                             "(frame_index,block_index,header_id,phase_id,jpegxs_payload,valid_length,raw_bayer_target);")
            for f in (14, 15, 16):
                db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?)",
                           (f, 0, "header", f%16+1, b"ab", 2, bytes([200])*10240))
            db.commit()
            db.close()
            class Cache:
                def get(self, key):
                    return torch.full((1, 4, 160), key[0]/255)
            dataset = V10Dataset(path, Cache(), frame_start=15)
            self.assertEqual(dataset.keys, [(16, 0)])
            low, phase, target, key = dataset[0]
            self.assertEqual(phase.tolist(), [0, 15])
            self.assertAlmostEqual(low[0].mean().item(), 16/255, places=6)
            self.assertAlmostEqual(target.mean().item(), 200/255, places=6)
            dataset.raw.connection.close()
            def guarded_connect(_):
                connection = sqlite3.connect(path)
                connection.set_authorizer(lambda action, table, column, *_:
                    sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and column == "raw_bayer_target"
                    else sqlite3.SQLITE_OK)
                return connection
            with patch("v10.data.connect", side_effect=guarded_connect):
                actual, high, actual_phase = inference_inputs(path, Cache(), 16, 0, "header")
                torch.testing.assert_close(actual[0], low)
                validate_inputs(actual, high, actual_phase, check_pair=True)

    def test_stratified_validation_fixed_unique_and_bounded(self):
        keys = [(f, b) for f in range(721, 900) for b in range(30)]
        chosen = stratified_indices(keys, 256)
        self.assertEqual(len(chosen), 256)
        self.assertEqual(len(set(chosen)), 256)
        self.assertEqual({keys[i][0]%16 for i in chosen}, set(range(16)))
        self.assertEqual(chosen, stratified_indices(keys, 256))


if __name__ == "__main__":
    unittest.main()
