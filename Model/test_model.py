import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import torch

from data import DecodedDataset, RawDataset
from network import ByteRawModel, ByteRawModelV2, HybridRawModel, ResidualBlock, build_model
from baseline import upscale
from torch.nn import functional as F


class PipelineTests(unittest.TestCase):
    def test_v9_plain8_expansion_gradient_and_checkpoint(self):
        self._check_v9_plain_expansion("plain8", 8, 588)

    def test_v9_plain16_expansion_gradient_and_checkpoint(self):
        self._check_v9_plain_expansion("plain16", 16, 1172)

    def test_v9_plain32_expansion_gradient_and_checkpoint(self):
        self._check_v9_plain_expansion("plain32", 32, 2340)

    def _check_v9_plain_expansion(self, variant, width, parameters):
        config = {"architecture": "v9", "history": 1, "v9_layers": 2, "v9_variant": variant}
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
        self.assertEqual(sum(p.numel()*p.element_size() for p in model.parameters()), parameters*4)
        self.assertEqual([type(c) for c in model.refine], [torch.nn.Conv2d, torch.nn.Conv2d])
        self.assertEqual([(c.in_channels, c.out_channels) for c in model.refine], [(4, width), (width, 4)])
        low = torch.rand(16, 1, 1, 4, 160, requires_grad=True)
        phases = torch.arange(16).reshape(16, 1)
        tokens, lengths = torch.zeros(16, 1, 640, dtype=torch.long), torch.full((16, 1), 100)
        base = upscale(low[:, 0], phases[:, 0])
        torch.testing.assert_close(model(tokens, phases, lengths, low), base)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            F.l1_loss(model(tokens, phases, lengths, low), (base.detach()+0.03).clamp(0, 1)).backward()
            optimizer.step()
        for layer in model.refine:
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertGreater(low.grad.abs().sum().item(), 0)
        result = model(tokens, phases, lengths, low)
        torch.testing.assert_close(result, model(tokens+123, phases, lengths-99, low))
        rebuilt = build_model(config)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(rebuilt(tokens, phases, lengths, low), result)

    def test_v9_relu4_structure_gradient_and_checkpoint(self):
        self._check_v9_relu("relu4", 4, 296)

    def test_v9_relu16_structure_gradient_and_checkpoint(self):
        self._check_v9_relu("relu16", 16, 1172)

    def test_v9_relu16_wide_structure_gradient_and_checkpoint(self):
        self._check_v9_relu("relu16", 16, 3476, kernel_width=9)

    def test_v9_relu16_three_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu16", 16, 3492, layers=3)

    def test_v9_relu16_four_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu16", 16, 5812, layers=4)

    def test_v9_relu16_five_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu16", 16, 8132, layers=5)

    def test_v9_relu24_four_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu24", 24, 12172, layers=4)

    def test_v9_relu32_four_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu32", 32, 20836, layers=4)

    def test_v9_relu64_four_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu64", 64, 78532, layers=4)

    def test_v9_relu128_four_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu128", 128, 304516, layers=4)

    def test_v9_relu256_four_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu256", 256, 1198852, layers=4)

    def test_v9_relu256_five_layers_gradient_and_checkpoint(self):
        self._check_v9_relu("relu256", 256, 1788932, layers=5)

    def test_v9_wide_kernel_reaches_distant_plane_samples(self):
        model = build_model({"architecture": "v9", "history": 1, "v9_layers": 2,
                             "v9_variant": "relu16", "v9_kernel_width": 9})
        for layer in (model.refine[0], model.refine[2]):
            torch.nn.init.ones_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        planes = torch.ones(1, 4, 8, 40, requires_grad=True)
        model.refine(planes)[0, 0, 4, 20].backward()
        support = planes.grad[0].sum((0, 1)).nonzero().flatten().tolist()
        self.assertEqual(support, list(range(12, 29)))
        with self.assertRaisesRegex(ValueError, "kernel width"):
            build_model({"architecture": "v9", "history": 1, "v9_kernel_width": 5})

    def _check_v9_relu(self, variant, width, parameters, kernel_width=3, layers=2):
        torch.manual_seed(42)
        config = {"architecture": "v9", "history": 1, "v9_layers": layers, "v9_variant": variant}
        if kernel_width != 3:
            config["v9_kernel_width"] = kernel_width
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
        self.assertEqual(sum(p.numel()*p.element_size() for p in model.parameters()), parameters*4)
        self.assertEqual([type(c) for c in model.refine],
                         [torch.nn.Conv2d, torch.nn.ReLU]*(layers-1)+[torch.nn.Conv2d])
        self.assertEqual([c.kernel_size for c in model.refine if isinstance(c, torch.nn.Conv2d)],
                         [(3, kernel_width)]*layers)
        self.assertEqual([(c.in_channels, c.out_channels) for c in model.refine
                          if isinstance(c, torch.nn.Conv2d)],
                         [(4, width)]+[(width, width)]*(layers-2)+[(width, 4)])
        low = torch.rand(16, 1, 1, 4, 160, requires_grad=True)
        phases = torch.arange(16).reshape(16, 1)
        tokens, lengths = torch.zeros(16, 1, 640, dtype=torch.long), torch.full((16, 1), 100)
        base = upscale(low[:, 0], phases[:, 0])
        torch.testing.assert_close(model(tokens, phases, lengths, low), base)
        target = (base.detach()+0.03).clamp(0, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            F.l1_loss(model(tokens, phases, lengths, low), target).backward()
            optimizer.step()
        for layer in (c for c in model.refine if isinstance(c, torch.nn.Conv2d)):
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertGreater(low.grad.abs().sum().item(), 0)
        result = model(tokens, phases, lengths, low)
        torch.testing.assert_close(result, model(tokens+123, phases, lengths-99, low))
        rebuilt = build_model(config)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(rebuilt(tokens, phases, lengths, low), result)

    def test_v9_plain4_two_layers_baseline_gradient_and_checkpoint(self):
        config = {"architecture": "v9", "history": 1, "v9_layers": 2, "v9_variant": "plain4"}
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 296)
        self.assertEqual(sum(p.numel()*p.element_size() for p in model.parameters()), 1184)
        self.assertEqual([(c.in_channels, c.out_channels) for c in model.modules()
                          if isinstance(c, torch.nn.Conv2d)], [(4, 4), (4, 4)])
        self.assertFalse(any(isinstance(c, torch.nn.ReLU) for c in model.modules()))
        low = torch.rand(16, 1, 1, 4, 160, requires_grad=True)
        phases = torch.arange(16).reshape(16, 1)
        tokens, lengths = torch.zeros(16, 1, 640, dtype=torch.long), torch.full((16, 1), 100)
        base = upscale(low[:, 0], phases[:, 0])
        torch.testing.assert_close(model(tokens, phases, lengths, low), base)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        target = (base.detach()+0.03).clamp(0, 1)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            F.l1_loss(model(tokens, phases, lengths, low), target).backward()
            optimizer.step()
        for layer in model.refine:
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertGreater(low.grad.abs().sum().item(), 0)
        result = model(tokens, phases, lengths, low)
        torch.testing.assert_close(result, model(tokens+123, phases, lengths-99, low))
        rebuilt = build_model(config)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(rebuilt(tokens, phases, lengths, low), result)
        with self.assertRaisesRegex(ValueError, "unknown v9 variant"):
            build_model({**config, "v9_variant": "invalid"})

    def test_v9_one_layer_baseline_learning_and_encoded_invariance(self):
        config = {"architecture": "v9", "history": 1, "v9_layers": 1}
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 148)
        self.assertEqual([(c.in_channels, c.out_channels) for c in model.modules()
                          if isinstance(c, torch.nn.Conv2d)], [(4, 4)])
        self.assertFalse(any(isinstance(c, torch.nn.ReLU) for c in model.modules()))
        low = torch.rand(16, 1, 1, 4, 160, requires_grad=True)
        phases = torch.arange(16).reshape(16, 1)
        tokens, lengths = torch.zeros(16, 1, 640, dtype=torch.long), torch.full((16, 1), 100)
        base = upscale(low[:, 0], phases[:, 0])
        torch.testing.assert_close(model(tokens, phases, lengths, low), base)
        target = (base.detach()+0.03).clamp(0, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        F.l1_loss(model(tokens, phases, lengths, low), target).backward()
        self.assertGreater(model.refine[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(low.grad.abs().sum().item(), 0)
        optimizer.step()
        output = model(tokens, phases, lengths, low)
        self.assertGreater((output-base).abs().max().item(), 0)
        torch.testing.assert_close(output, model(tokens+123, phases, lengths-99, low))
        restored = build_model(config)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(tokens, phases, lengths, low), output)
        with self.assertRaisesRegex(ValueError, "one to five"):
            build_model({**config, "v9_layers": 6})

    def test_v9_original_two_layer_checkpoint_still_loads(self):
        path = Path(__file__).parent/"runs/simple-v9-q5/best.pt"
        if not path.is_file():
            self.skipTest("original trained checkpoint is not installed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.assertNotIn("v9_layers", saved["config"])
        model = build_model(saved["config"])
        model.load_state_dict(saved["model"])
        self.assertEqual(model.v9_layers, 2)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 588)

    def test_v9_minimal_contract_baseline_and_checkpoint(self):
        config = {"architecture": "v9", "history": 1, "decay": 0.2}
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 588)
        self.assertEqual([(c.in_channels, c.out_channels) for c in model.modules()
                          if isinstance(c, torch.nn.Conv2d)], [(4, 8), (8, 4)])
        self.assertFalse(any(isinstance(c, (ResidualBlock, torch.nn.Embedding, torch.nn.Conv1d))
                             for c in model.modules()))
        low = torch.rand(16, 1, 1, 4, 160)
        phases = torch.arange(16).reshape(16, 1)
        tokens = torch.randint(0, 256, (16, 1, 640))
        lengths = torch.full((16, 1), 100)
        result = model(tokens, phases, lengths, low)
        torch.testing.assert_close(result, upscale(low[:, 0], phases[:, 0]))
        rebuilt = build_model(config)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(rebuilt(tokens, phases, lengths, low), result)
        with self.assertRaisesRegex(ValueError, "history=1"):
            build_model({**config, "history": 16})
        with self.assertRaisesRegex(ValueError, "decoded RAW"):
            model(tokens, phases, lengths, low.expand(-1, 2, -1, -1, -1))
        with self.assertRaisesRegex(ValueError, "floating RAW"):
            model(tokens, phases, lengths, low.byte())

    def test_v9_two_frames_phase_order_previous_dependency_and_checkpoint(self):
        config = {"architecture": "v9", "history": 2, "v9_layers": 4, "v9_variant": "relu128"}
        model = build_model(config)
        self.assertEqual(model.refine[0].in_channels, 8)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 309124)
        low = (torch.rand(16, 2, 1, 4, 160)*0.4+0.3).requires_grad_()
        phases = torch.stack([torch.arange(16), (torch.arange(16)-1)%16], dim=1)
        tokens = torch.zeros(16, 2, 640, dtype=torch.long)
        lengths = torch.full((16, 2), 100)
        captured = []
        hook = model.refine[0].register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
        initial = model(tokens, phases, lengths, low)
        hook.remove()
        current = upscale(low[:, 0], phases[:, 0])
        previous = upscale(low[:, 1], phases[:, 1])
        torch.testing.assert_close(initial, current)
        torch.testing.assert_close(captured[0], torch.cat([F.pixel_unshuffle(current, 2),
                                                        F.pixel_unshuffle(previous, 2)], dim=1))
        with torch.no_grad():
            for layer in model.refine:
                if isinstance(layer, torch.nn.Conv2d):
                    layer.weight.fill_(0.0001)
                    layer.bias.zero_()
        result = model(tokens, phases, lengths, low)
        result.mean().backward()
        self.assertGreater(low.grad[:, 0].abs().sum().item(), 0)
        self.assertGreater(low.grad[:, 1].abs().sum().item(), 0)
        changed = low.detach().clone()
        changed[:, 1] = 0
        self.assertGreater((result-model(tokens, phases, lengths, changed)).abs().max().item(), 0)
        torch.testing.assert_close(result, model(tokens+99, phases, lengths-50, low))
        rebuilt = build_model(config)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(result, rebuilt(tokens, phases, lengths, low))
        with self.assertRaisesRegex(ValueError, "decoded RAW"):
            model(tokens, phases, lengths, low[:, :1])

    def test_v9_upper_strip_mask_gradient_phase_and_checkpoint(self):
        config = {"architecture": "v9", "history": 2, "v9_layers": 4,
                  "v9_variant": "relu128", "v9_upper_strip": True}
        model = build_model(config)
        self.assertEqual(model.refine[0].in_channels, 13)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 314884)
        low = (torch.rand(16, 3, 1, 4, 160)*0.4+0.3).requires_grad_()
        p = torch.arange(16)
        phases = torch.stack([p, (p-1)%16, p], dim=1)
        phases[0, -1] = -1
        tokens, lengths = torch.zeros(16, 3, 640, dtype=torch.long), torch.full((16, 3), 100)
        seen = []
        hook = model.refine[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].detach()))
        initial = model(tokens, phases, lengths, low)
        hook.remove()
        torch.testing.assert_close(initial, upscale(low[:, 0], phases[:, 0]))
        torch.testing.assert_close(seen[0][1:, 8:12], F.pixel_unshuffle(upscale(low[1:, 2], phases[1:, 2]), 2))
        self.assertEqual(seen[0][0, 8:].abs().sum().item(), 0)
        self.assertTrue(torch.all(seen[0][1:, 12] == 1))
        with torch.no_grad():
            for layer in model.refine:
                if isinstance(layer, torch.nn.Conv2d):
                    layer.weight.fill_(0.0001)
                    layer.bias.zero_()
        output = model(tokens, phases, lengths, low)
        output.mean().backward()
        for slot in range(3):
            self.assertGreater(low.grad[1:, slot].abs().sum().item(), 0)
        self.assertEqual(low.grad[0, 2].abs().sum().item(), 0)
        changed = low.detach().clone()
        changed[:, 2] = 0
        altered = model(tokens, phases, lengths, changed)
        torch.testing.assert_close(output[0], altered[0])
        self.assertGreater((output[1:]-altered[1:]).abs().max().item(), 0)
        rebuilt = build_model(config)
        rebuilt.load_state_dict(model.state_dict())
        torch.testing.assert_close(output, rebuilt(tokens, phases, lengths, low))

    def test_v9_learns_and_ignores_encoded_bytes(self):
        torch.manual_seed(42)
        model = build_model({"architecture": "v9", "history": 1})
        low = torch.rand(2, 1, 1, 4, 160, requires_grad=True)
        phases = torch.tensor([[0], [15]])
        tokens, lengths = torch.zeros(2, 1, 640, dtype=torch.long), torch.full((2, 1), 100)
        target = (upscale(low.detach()[:, 0], phases[:, 0])+0.05).clamp(0, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            F.l1_loss(model(tokens, phases, lengths, low), target).backward()
            optimizer.step()
        self.assertGreater(model.refine[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.refine[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(low.grad.abs().sum().item(), 0)
        result = model(tokens, phases, lengths, low)
        torch.testing.assert_close(result, model(tokens+123, phases, lengths-99, low))
        self.assertGreater((result-upscale(low[:, 0], phases[:, 0])).abs().max().item(), 0)

    def test_schema_v4_go_decoded_cache_and_inference(self):
        from baseline import DecodedCache, decode_keys, main as cache_main
        with tempfile.TemporaryDirectory() as folder:
            path, cache_path = Path(folder)/"v4.sqlite", Path(folder)/"v4.npz"
            with sqlite3.connect(path) as db:
                db.executescript("PRAGMA user_version=4; CREATE TABLE shared_header(singleton,definition); "
                                 "INSERT INTO shared_header VALUES(1,'{\"id\":\"fixed\"}'); "
                                 "CREATE TABLE samples(frame_index,block_index,header_id,phase_id,"
                                 "jpegxs_payload,valid_length,raw_bayer_target,decoded);")
                for frame in (0, 1):
                    db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)",
                               (frame, 0, "fixed", frame+1, b"\x00\x7f", 2,
                                bytes([200])*10240, bytes([10+frame])*640))
            db.close()
            with patch("sys.argv", ["baseline.py", "--db", str(path), "--out", str(cache_path)]), redirect_stdout(io.StringIO()):
                cache_main()
            dataset = RawDataset(path, 2)
            cache = DecodedCache(cache_path, path)
            sample = DecodedDataset(dataset, cache)[0]
            torch.testing.assert_close(sample[4][:, 0, 0, 0], torch.tensor([11, 10])/255)
            direct = decode_keys(path, [(1, 0), (0, 0)], Path(folder)/"decoder-must-not-run.exe")
            torch.testing.assert_close(direct, sample[4])
            self.assertAlmostEqual(sample[3].mean().item(), 200/255, places=6)
            dataset.connection.close()
            with sqlite3.connect(path) as db:
                db.execute("UPDATE samples SET decoded=? WHERE frame_index=1", (b"bad",))
            db.close()
            with self.assertRaisesRegex(ValueError, "invalid stored decoded"):
                decode_keys(path, [(1, 0)], Path("missing.exe"))
            with self.assertRaisesRegex(ValueError, "different database"):
                DecodedCache(cache_path, path)

    def test_retired_upper_checkpoint_is_not_loaded_as_original_v7(self):
        with self.assertRaisesRegex(ValueError, "upper-strip checkpoints are retired"):
            build_model({"architecture": "v7", "history": 2, "decay": 0.2, "previous_block": True})

    def test_deeper_models_baseline_gradient_and_checkpoint(self):
        for architecture, depth in (("v5", 5), ("v6", 6), ("v7", 5), ("v8", 5)):
            with self.subTest(architecture=architecture):
                config = {"architecture": architecture, "history": 2, "decay": 0.2}
                model = build_model(config)
                blocks = [layer for layer in model.refine if isinstance(layer, ResidualBlock)]
                self.assertEqual(len(blocks), depth)
                self.assertEqual(sum(isinstance(layer, torch.nn.Conv2d) for layer in model.refine.modules()),
                                 2*depth+2)
                tokens = torch.randint(0, 256, (1, 2, 640))
                phases, lengths = torch.tensor([[0, 15]]), torch.tensor([[120, 100]])
                low = torch.rand(1, 2, 1, 4, 160, requires_grad=True)
                torch.testing.assert_close(model(tokens, phases, lengths, low), upscale(low[:, 0], phases[:, 0]))
                torch.nn.init.normal_(model.refine[-2].weight, std=0.01)
                result = model(tokens, phases, lengths, low)
                result.square().mean().backward()
                for block in blocks:
                    for layer in block.body:
                        if isinstance(layer, torch.nn.Conv2d):
                            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
                self.assertGreater(low.grad[:, 1].abs().sum().item(), 0)
                rebuilt = build_model(config)
                rebuilt.load_state_dict(model.state_dict())
                torch.testing.assert_close(rebuilt(tokens, phases, lengths, low), result)

    def test_four_layer_expansion_contraction(self):
        model = build_model({"architecture": "v8", "history": 2, "decay": 0.2, "residual_layers": 4})
        features = torch.randn(1, 24, 8, 320)
        for block in (b for b in model.refine if isinstance(b, ResidualBlock)):
            convs = [c for c in block.body if isinstance(c, torch.nn.Conv2d)]
            self.assertEqual([(c.in_channels, c.out_channels) for c in convs],
                             [(24, 48), (48, 96), (96, 48), (48, 24)])
            self.assertEqual(block(features).shape, features.shape)
            for parameter in block.parameters():
                torch.nn.init.zeros_(parameter)
            torch.testing.assert_close(block(features), features)

    def test_v8_expands_to_96_and_preserves_legacy_checkpoint(self):
        config = {"architecture": "v8", "history": 2, "decay": 0.2}
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 218748)
        blocks = [b for b in model.refine if isinstance(b, ResidualBlock)]
        self.assertEqual(len(blocks), 5)
        for block in blocks:
            self.assertEqual([(c.in_channels, c.out_channels) for c in block.body
                              if isinstance(c, torch.nn.Conv2d)], [(24, 96), (96, 24)])
        self.assertEqual(model.residual_layers, 2)
        restored = build_model({**config, "residual_layers": 4, "residual_expansion": 2})
        self.assertEqual(restored.residual_layers, 4)
        self.assertEqual(sum(p.numel() for p in restored.parameters()), 530268)
        legacy_path = Path(__file__).parent/"runs/hybrid-v8-history2-expand4layer/best.pt"
        if legacy_path.is_file():
            legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
            restored = build_model(legacy["config"])
            restored.load_state_dict(legacy["model"])
            self.assertEqual(restored.residual_layers, 4)
        with self.assertRaisesRegex(ValueError, "unknown v8 variant"):
            build_model({**config, "v8_variant": "invalid"})

    def test_expanded_residual_contracts_before_shortcut(self):
        model = build_model({"architecture": "v7", "history": 2, "decay": 0.2})
        blocks = [layer for layer in model.refine if isinstance(layer, ResidualBlock)]
        self.assertEqual(len(blocks), 5)
        features = torch.randn(1, 24, 8, 320)
        for block in blocks:
            self.assertEqual(block.body[0].in_channels, 24)
            self.assertEqual(block.body[0].out_channels, 48)
            self.assertEqual(block.body[2].in_channels, 48)
            self.assertEqual(block.body[2].out_channels, 24)
            self.assertEqual(block(features).shape, features.shape)
            for parameter in block.parameters():
                torch.nn.init.zeros_(parameter)
            torch.testing.assert_close(block(features), features)

    def test_four_block_backbone_and_shortcuts(self):
        model = build_model({"architecture": "v4", "history": 1, "decay": 0.2})
        blocks = [layer for layer in model.refine if isinstance(layer, ResidualBlock)]
        self.assertEqual(len(blocks), 4)
        self.assertEqual(sum(isinstance(layer, torch.nn.Conv2d) for layer in model.refine.modules()), 10)
        block = blocks[0]
        features = torch.randn(1, 24, 8, 320)
        for parameter in block.parameters():
            torch.nn.init.zeros_(parameter)
        torch.testing.assert_close(block(features), features)

    def test_v4_output_baseline_backward_and_checkpoint(self):
        for history in (1, 2, 16):
            config = {"architecture": "v4", "history": history, "decay": 0.2}
            model = build_model(config)
            tokens = torch.randint(0, 256, (1, history, 640))
            phases = torch.arange(history).reshape(1, history)
            lengths = torch.full((1, history), 100)
            low = torch.rand(1, history, 1, 4, 160)
            output = model(tokens, phases, lengths, low)
            torch.testing.assert_close(output, upscale(low[:, 0], phases[:, 0]))
            # Once the initially zero output head learns, every residual block
            # must participate in backpropagation.
            torch.nn.init.normal_(model.refine[-2].weight, std=0.01)
            result = model(tokens, phases, lengths, low)
            result.square().mean().backward()
            for block in (layer for layer in model.refine if isinstance(layer, ResidualBlock)):
                self.assertGreater(block.body[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(block.body[2].weight.grad.abs().sum().item(), 0)
            rebuilt = build_model(config)
            rebuilt.load_state_dict(model.state_dict())
            torch.testing.assert_close(rebuilt(tokens, phases, lengths, low), result)

    def test_hybrid_starts_at_real_decoded_baseline(self):
        for history in (1, 2, 16):
            model = HybridRawModel(history)
            tokens = torch.randint(0, 256, (1, history, 640))
            phases = torch.arange(history).reshape(1, history)
            lengths = torch.full((1, history), 100)
            low = torch.rand(1, history, 1, 4, 160)
            output = model(tokens, phases, lengths, low)
            torch.testing.assert_close(output, upscale(low[:, 0], phases[:, 0]))
            changed = model(tokens, phases, lengths, torch.zeros_like(low))
            self.assertGreater((output-changed).abs().mean().item(), 0.1)
            with self.assertRaises(TypeError):
                model(tokens, phases, lengths)
            with self.assertRaises(ValueError):
                model(tokens, phases, lengths, low.byte())

    def test_hybrid_both_input_branches_receive_gradients(self):
        model = HybridRawModel()
        # The zero-initialized last layer intentionally blocks upstream gradients
        # on the first step; emulate a trained head to test branch connectivity.
        torch.nn.init.normal_(model.refine[-2].weight, std=0.02)
        tokens = torch.full((1, 1, 640), 256, dtype=torch.long)
        tokens[:, :, :4] = torch.tensor([0, 2, 0, 127])
        low = torch.rand(1, 1, 1, 4, 160, requires_grad=True)
        phase, length = torch.tensor([[3]]), torch.tensor([[2]])
        output = model(tokens, phase, length, low)
        output.mean().backward()
        self.assertGreater(model.embedding.weight.grad[127].abs().sum().item(), 0)
        self.assertGreater(low.grad.abs().sum().item(), 0)
        changed_tokens = tokens.clone()
        changed_tokens[:, :, 3] = 22
        changed = model(changed_tokens, phase, length, low)
        self.assertGreater((changed-output).abs().max().item(), 0)

    def test_phase_aligned_cfa_baseline_preserves_samples(self):
        low = torch.rand(16, 1, 4, 160)
        phases = torch.arange(16)
        for mode in ("nearest", "bilinear"):
            high = upscale(low, phases, mode)
            source = F.pixel_unshuffle(low, 2)
            enlarged = F.pixel_unshuffle(high, 2)
            for phase in range(16):
                selected = enlarged[phase, :, phase//4::4, phase%4::4]
                torch.testing.assert_close(selected, source[phase], atol=1e-5, rtol=1e-5)
        planes = torch.tensor([0.1, 0.3, 0.6, 0.9]).view(1, 4, 1, 1).expand(16, 4, 2, 80)
        result = F.pixel_unshuffle(upscale(F.pixel_shuffle(planes, 2), phases), 2)
        torch.testing.assert_close(result, planes[:, :, :1, :1].expand(16, 4, 8, 320))

    def test_v2_starts_from_learned_low_and_has_gradient(self):
        for history in (1, 2, 16):
            model = ByteRawModelV2(history)
            tokens = torch.full((1, history, 640), 256, dtype=torch.long)
            tokens[:, :, :4] = torch.tensor([0, 2, 0, 127])
            phase = torch.arange(history).reshape(1, history)
            length = torch.full((1, history), 2)
            output, low = model(tokens, phase, length, return_low=True)
            torch.testing.assert_close(output, upscale(low[:, 0], phase[:, 0]))
            (output.mean()+low.mean()).backward()
            self.assertGreater(model.embedding.weight.grad[127].abs().sum().item(), 0)

    def test_history_boundaries_and_zero_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript("PRAGMA user_version=3; CREATE TABLE samples "
                                 "(frame_index,block_index,header_id,phase_id,jpegxs_payload,valid_length,raw_bayer_target);")
                for frame in range(35):
                    if frame == 17:
                        continue
                    db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?)",
                               (frame, 0, "fixed", frame % 16+1, b"\x00\x7f", 2, bytes([frame])*10240))
            db.close()
            dataset = RawDataset(path, 16, frame_start=1)
            self.assertEqual(dataset.keys, [(16, 0), (33, 0), (34, 0)])
            tokens, phases, lengths, target = dataset[0]
            self.assertEqual(tokens[0, :5].tolist(), [0, 2, 0, 127, 256])
            self.assertEqual(phases.tolist(), [0]+list(range(15, 0, -1)))
            self.assertTrue(torch.all(lengths == 2))
            self.assertAlmostEqual(target.mean().item(), 16/255, places=6)
            dataset.connection.close()

    def test_output_gradient_and_time_order(self):
        torch.set_num_threads(2)
        for history in (1, 2, 16):
            model = ByteRawModel(history)
            self.assertTrue(torch.all(model.time_weights[:-1] > model.time_weights[1:]))
            tokens = torch.full((1, history, 640), 256, dtype=torch.long)
            tokens[:, :, :4] = torch.tensor([0, 2, 0, 127])
            output = model(tokens, torch.zeros((1, history), dtype=torch.long),
                           torch.full((1, history), 2))
            self.assertEqual(output.shape, (1, 1, 16, 640))
            output.mean().backward()
            self.assertGreater(model.embedding.weight.grad[127].abs().sum().item(), 0)
            self.assertEqual(model.embedding.weight.grad[256].abs().sum().item(), 0)

    def test_two_frame_pairing_with_gap_and_phase_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pair.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript("PRAGMA user_version=3; CREATE TABLE samples "
                                 "(frame_index,block_index,header_id,phase_id,jpegxs_payload,valid_length,raw_bayer_target);")
                for frame in (14, 15, 16, 18, 19):
                    for block in (0, 1):
                        db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?)",
                                   (frame, block, "fixed", frame%16+1, bytes([frame, block]), 2,
                                    bytes([frame])*10240))
            db.close()
            dataset = RawDataset(path, 2, frame_start=15)
            self.assertEqual(dataset.keys, [(16, 0), (16, 1), (19, 0), (19, 1)])

            class Cache:
                def get(self, key):
                    frame, block = key
                    return torch.full((1, 4, 160), (frame+block)/255)

            tokens, phases, lengths, target, decoded = DecodedDataset(dataset, Cache())[1]
            self.assertEqual(tokens[:, 2:4].tolist(), [[16, 1], [15, 1]])
            self.assertEqual(phases.tolist(), [0, 15])
            self.assertEqual(lengths.tolist(), [2, 2])
            self.assertEqual(decoded.shape, (2, 1, 4, 160))
            self.assertAlmostEqual(decoded[0].mean().item(), 17/255, places=6)
            self.assertAlmostEqual(decoded[1].mean().item(), 16/255, places=6)
            self.assertAlmostEqual(target.mean().item(), 16/255, places=6)
            dataset.connection.close()

    def test_current_upper_strip_pairing_and_top_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"upper.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript("PRAGMA user_version=4; CREATE TABLE samples "
                                 "(frame_index,block_index,header_id,phase_id,jpegxs_payload,valid_length,raw_bayer_target);")
                for frame in (14, 15, 16, 18, 19):
                    for block in range(3):
                        if (frame, block) == (19, 1):
                            continue
                        db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?)",
                                   (frame, block, "fixed", frame%16+1, bytes([frame, block]), 2,
                                    bytes([frame+block])*10240))
            db.close()
            dataset = RawDataset(path, 2, frame_start=15, v9_upper_strip=True)
            self.assertEqual(dataset.keys, [(16, 0), (16, 1), (16, 2), (19, 0)])
            class Cache:
                def get(self, key):
                    frame, block = key
                    return torch.full((1, 4, 160), (frame+block)/255)
            wrapped = DecodedDataset(dataset, Cache())
            tokens, phases, lengths, target, low = wrapped[1]
            self.assertEqual(tokens[:, 2:4].tolist(), [[16, 1], [15, 1], [16, 0]])
            self.assertEqual(phases.tolist(), [0, 15, 0])
            torch.testing.assert_close(low[:, 0, 0, 0], torch.tensor([17., 16., 16.])/255)
            self.assertAlmostEqual(target.mean().item(), 17/255, places=6)
            tokens, phases, lengths, _, low = wrapped[0]
            self.assertEqual(phases.tolist(), [0, 15, -1])
            self.assertEqual(lengths.tolist(), [2, 2, 0])
            self.assertTrue(torch.all(tokens[-1] == 256))
            self.assertEqual(low[-1].abs().sum().item(), 0)
            dataset.connection.close()

    def test_two_frame_inputs_both_affect_trained_model(self):
        torch.manual_seed(7)
        model = build_model({"architecture": "v4", "history": 2, "decay": 0.2})
        self.assertGreater(model.time_weights[0], model.time_weights[1])
        torch.nn.init.normal_(model.refine[-2].weight, std=0.02)
        tokens = torch.randint(0, 256, (1, 2, 640))
        phase, length = torch.tensor([[0, 15]]), torch.tensor([[120, 100]])
        low = torch.rand(1, 2, 1, 4, 160, requires_grad=True)
        result = model(tokens, phase, length, low)
        result.mean().backward()
        for frame in (0, 1):
            self.assertGreater(low.grad[:, frame].abs().sum().item(), 0)
            altered = tokens.clone()
            altered[:, frame] = (altered[:, frame]+31)%256
            self.assertGreater((result-model(altered, phase, length, low)).abs().max().item(), 0)


if __name__ == "__main__":
    from record_experiment import run_logged
    run_logged("checks", unittest.main, Path(__file__).parent/"runs"/"checks-v8-expand96")
