"""Regression checks for v13 history, batching, quantization and recording."""
import unittest

import torch

from predict_v13 import window
from v13.ensemble import EnsembleGate
from v12 import nearest_bayer_bytes
from v13.base import features
from v13.fusion import (FUSION_VARIANTS, LocalAlignment, consistency_logits, extrapolate_low_frequency, nearest_indices, newest_first,
                       phases_in_age_order, temporal_trend)
from v13.model import ResidualCNN, loss_for, mix_residual_bands, quantize
from v13.pairs import adjacent_groups, pair_loss, previous_indices


class V13Tests(unittest.TestCase):
    def test_fusion_recovers_real_phase_planes_and_keeps_base_at_initialization(self):
        generator = torch.Generator().manual_seed(42)
        decoded = torch.randint(0, 256, (16, 1, 4, 160), dtype=torch.uint8, generator=generator)
        phases = tuple((7 - age) % 16 for age in range(16))
        features_bytes = features(decoded, phases).unsqueeze(0)
        pack = torch.nn.functional.pixel_unshuffle
        measured = pack(features_bytes[:, 1:2].float(), 2)
        ages = features_bytes[:, 2:3, ::2, ::2]
        expected = pack(nearest_bayer_bytes(decoded, torch.tensor(phases)).float(), 2)
        self.assertTrue(torch.equal(newest_first(measured, ages, nearest_indices())[0], expected))
        with torch.inference_mode():
            for variant in FUSION_VARIANTS:
                model = ResidualCNN(variant).to(memory_format=torch.channels_last)
                self.assertTrue(torch.equal(model.predict(features_bytes), features_bytes[:, :1]))
                self.assertTrue(torch.equal(model.predict(features_bytes, 0), features_bytes[:, :1]))
        model = ResidualCNN("R2-frames-guide")
        target = torch.rand((1, 1, 16, 640), generator=generator)
        predicted, guidance = model(features_bytes, fusion_target=target)
        torch.testing.assert_close(predicted, model(features_bytes), rtol=0, atol=0)
        self.assertGreater(guidance.item(), 0)
        guidance.backward()
        self.assertGreater(model.fusion.score.weight.grad.abs().sum().item(), 0)
        clean = ResidualCNN("R2-frames-clean").fusion
        recovered = clean.clean_frames(measured / 255, phases_in_age_order(ages))
        torch.testing.assert_close(recovered[0], expected / 255, rtol=0, atol=1e-7)

    def test_zero_alignment_preserves_history_and_sampling_support(self):
        alignment = LocalAlignment().to(memory_format=torch.channels_last)
        random = torch.Generator().manual_seed(42)
        frames = torch.rand(1, 2, 4, 8, 320, generator=random)
        foot = torch.rand(1, 2, 1, 8, 320, generator=random)
        age = torch.arange(2).reshape(1, 2, 1, 1, 1).float().expand_as(foot)
        aligned, support = alignment(frames, foot, age)
        torch.testing.assert_close(aligned, frames, rtol=0, atol=3e-5)
        torch.testing.assert_close(support, foot, rtol=0, atol=3e-5)

    def test_consistency_suppresses_stale_values_that_disagree(self):
        frames = torch.zeros((1, 3, 4, 8, 10))
        age = torch.tensor([0., 0.25, 1.]).view(1, 3, 1, 1, 1)
        self.assertTrue(torch.count_nonzero(consistency_logits(frames, age)) == 0)
        frames[:, 1:] = 0.2
        weights = consistency_logits(frames, age).softmax(dim=1)
        self.assertTrue(torch.all(weights[:, 0] > weights[:, 1]))
        self.assertTrue(torch.all(weights[:, 1] > weights[:, 2]))

    def test_temporal_trend_recovers_linear_change_without_attenuating_static_regions(self):
        age = torch.linspace(0, 1, 16, dtype=torch.float64).view(1, 16, 1, 1, 1)
        frame = torch.full((1, 16, 4, 8, 10), 0.6, dtype=torch.float64)
        torch.testing.assert_close(temporal_trend(frame, age), torch.zeros_like(frame[:, 0]), rtol=0, atol=1e-12)
        torch.testing.assert_close(temporal_trend(frame - age * 0.3, age), torch.full_like(frame[:, 0], 0.15),
                                   rtol=0, atol=1e-12)
        observed = frame - age * 0.2
        restored = extrapolate_low_frequency(observed, age, temporal_trend(observed, age))
        torch.testing.assert_close(restored, frame, rtol=0, atol=1e-12)

    def test_temporal_groups_exclude_gaps_and_replaced_targets(self):
        manifest = [{"frame": i, "first": i - 15, "block": 0, "header": "a"}
                    for i in (15, 16, 17, 18, 20, 21)]
        pairs, singles = adjacent_groups(manifest, replaced=[3])
        self.assertEqual(pairs.tolist(), [[0, 1], [4, 5]])
        self.assertEqual(singles.tolist(), [2, 3])
        self.assertEqual(previous_indices(manifest, replaced=[3]).tolist(), [-1, 0, 1, -1, -1, 4])
        target = torch.arange(4.).view(4, 1, 1, 1)
        self.assertEqual(pair_loss(target, target, torch.tensor([True, True])).item(), 0)
        self.assertEqual(pair_loss(target * 2, target, torch.tensor([False, True])).item(), 1)

    def test_extended_context_keeps_parameter_count_and_strip_shape(self):
        compact = ResidualCNN("R2")
        extended = ResidualCNN("R2", context="extended")
        self.assertEqual(sum(p.numel() for p in compact.parameters()), sum(p.numel() for p in extended.parameters()))
        self.assertEqual(extended.blocks[2][0].dilation, (1, 3))
        sample = torch.zeros(1, 4, 16, 640, dtype=torch.uint8)
        self.assertTrue(torch.equal(extended.predict(sample), sample[:, :1]))

    def test_band_mixing_preserves_base_and_passes_constant_correction_at_low_strength(self):
        base = torch.rand(2, 1, 16, 640)
        full = base + 0.2
        torch.testing.assert_close(mix_residual_bands(base, full, 0, 0), base, rtol=0, atol=0)
        torch.testing.assert_close(mix_residual_bands(base, full, 1, 0.5), full, rtol=0, atol=2e-7)

    def test_gate_initializes_to_frozen_model_average(self):
        model = EnsembleGate()
        features = torch.zeros(2, 4, 16, 640, dtype=torch.uint8)
        trend = torch.rand(2, 1, 16, 640)
        spatial = torch.rand_like(trend)
        torch.testing.assert_close(model(features, trend, spatial), (trend + spatial) / 2, rtol=0, atol=1e-7)

    def test_history_ends_at_gap_header_or_split(self):
        decoded = torch.zeros((1, 4, 160), dtype=torch.uint8)
        rows = {(frame, 0): ("a", frame % 16, decoded) for frame in range(34)}
        self.assertEqual(window(rows, 20, 0, 0)[1], tuple(t % 16 for t in range(20, 4, -1)))
        self.assertEqual(window(rows, 20, 0, 18)[1], (4, 3, 2))
        rows[19, 0] = ("b", 3, decoded)
        self.assertEqual(window(rows, 20, 0, 0)[1], (4,))
        del rows[19, 0]
        self.assertEqual(window(rows, 20, 0, 0)[1], (4,))

    def test_quantization_bounds_and_ties(self):
        values = torch.tensor([-1., 0.5 / 255, 1.5 / 255, 2.5 / 255, 1.5], dtype=torch.float64)
        self.assertEqual(quantize(values).tolist(), [0, 0, 2, 2, 255])

    def test_partial_microbatch_gradient_and_independent_predictions(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = ResidualCNN("R1")
        small = torch.randint(0, 256, (6, 4, 16, 16), dtype=torch.uint8)
        small[:, 2] %= 16
        small[:, 3] %= 2
        target = torch.rand(6, 1, 16, 16)
        parameters = list(model.parameters())
        a = torch.autograd.grad(loss_for("R1", model(small), target), parameters)
        model.zero_grad()
        for indices in torch.arange(6).split(4):
            (loss_for("R1", model(small[indices]), target[indices]) * (len(indices) / 6)).backward()
        for grad, parameter in zip(a, parameters):
            torch.testing.assert_close(parameter.grad, grad, rtol=1e-5, atol=1e-7)
        with torch.no_grad():
            model.head.weight.fill_(0.002)
            torch.testing.assert_close(model(small)[:1], model(small[:1]), rtol=0, atol=1e-7)
            self.assertTrue(torch.equal(model.predict(small, 0), small[:, :1]))


if __name__ == "__main__":
    unittest.main()
