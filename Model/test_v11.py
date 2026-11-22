import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from train_v11 import split_windows
from v11.data import V11Dataset, build_windows, inference_inputs
from v11.model import FusionNet, phase_shift, warp
from v11.phase import (HEIGHT, LR_HEIGHT, LR_WIDTH, PHASES, TILE, WIDTH, coverage,
                       exact_measurements, phase_offsets, reassemble, scatter_indices)


def make_database(directory, frames=6, blocks=(0, 1)):
    path = Path(directory) / "window.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE samples (frame_index INTEGER, block_index INTEGER, "
                 "phase_id INTEGER, decoded BLOB, raw_bayer_target BLOB)")
    rng = torch.Generator().manual_seed(7)
    for block in blocks:
        for frame in range(frames):
            decoded = torch.randint(0, 256, (LR_HEIGHT * LR_WIDTH,), generator=rng, dtype=torch.uint8)
            target = torch.randint(0, 256, (HEIGHT * WIDTH,), generator=rng, dtype=torch.uint8)
            conn.execute("INSERT INTO samples VALUES (?,?,?,?,?)",
                         (frame, block, frame % 16 + 1, decoded.numpy().tobytes(),
                          target.numpy().tobytes()))
    conn.commit()
    conn.close()
    return str(path)


class V11DataTests(unittest.TestCase):
    def test_windows_are_contiguous_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_database(directory)
            windows = build_windows(path, frames=4)
            self.assertEqual(len(windows), 2 * (6 - 4 + 1))
            for block, frames in windows:
                self.assertEqual(frames[-1] - frames[0], 3)
                self.assertEqual(len(frames), 4)
            self.assertEqual(len(build_windows(path, frames=2)), 2 * (6 - 2 + 1))
            self.assertEqual(len(build_windows(path, frames=4, blocks=(0,))), 3)

    def test_dataset_item_shapes_and_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_database(directory)
            dataset = V11Dataset(path, frames=4)
            decoded, phases, target, key = dataset[0]
            dataset.close()
            self.assertEqual(decoded.shape, (4, 1, LR_HEIGHT, LR_WIDTH))
            self.assertEqual(phases.shape, (4,))
            self.assertEqual(target.shape, (1, HEIGHT, WIDTH))
            self.assertTrue(torch.equal(phases, torch.tensor([3, 2, 1, 0])))
            self.assertEqual(int(key), 3)
            self.assertTrue(0.0 <= float(decoded.min()) and float(decoded.max()) <= 1.0)

    def test_inference_inputs_require_complete_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_database(directory)
            decoded, phases = inference_inputs(path, 5, 0, frames=4)
            self.assertEqual(decoded.shape, (4, 1, LR_HEIGHT, LR_WIDTH))
            # The current frame comes first, as in the input contract.
            self.assertTrue(torch.equal(phases, torch.tensor([5, 4, 3, 2])))
            with self.assertRaisesRegex(ValueError, "incomplete history"):
                inference_inputs(path, 2, 0, frames=4)

    def test_split_and_scene_boundaries_preserve_common_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_database(directory, frames=15)
            train, validation = split_windows(path, 4, (0, 7), (9, 14), [4, 13])
            self.assertEqual({ids[-1] for _, ids in train}, {3, 7})
            self.assertEqual({ids[-1] for _, ids in validation}, {12})
            train_inputs = {frame for _, frames in train for frame in frames}
            val_inputs = {frame for _, frames in validation for frame in frames}
            self.assertTrue(train_inputs.isdisjoint(val_inputs))
            for history in (1, 2):
                shortened = V11Dataset(path, history, windows=validation).windows
                self.assertEqual([(b, f[-1]) for b, f in shortened],
                                 [(b, f[-1]) for b, f in validation])
            with self.assertRaisesRegex(ValueError, "require complete"):
                split_windows(path, 4, (0, 7), (9, 11), [])


class V11ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_phase_shift_uses_unwrapped_coordinate_difference(self):
        reference = torch.tensor([0])
        self.assertEqual(tuple(int(v) for v in phase_shift(reference, torch.tensor([0]))), (0, 0))
        # A frame sampled one lattice step to the right must be sampled one step left.
        self.assertEqual(int(phase_shift(reference, torch.tensor([1]))[0]), -1)
        self.assertEqual(int(phase_shift(reference, torch.tensor([3]))[0]), -3)
        self.assertEqual(int(phase_shift(reference, torch.tensor([4]))[1]), -1)
        self.assertEqual(int(phase_shift(reference, torch.tensor([12]))[1]), -3)
        self.assertEqual(tuple(int(v) for v in phase_shift(torch.tensor([15]), reference)), (3, 3))

    def test_warp_moves_content_by_the_given_offset(self):
        feature = torch.zeros(1, 1, 5, 5)
        feature[0, 0, 2, 2] = 1.0
        dx = torch.zeros(1, 5, 5)
        dy = torch.zeros(1, 5, 5)
        dx[:, :, :] = 1.0
        # grid_sample reads the input at (x + dx), so a positive offset moves content left.
        shifted = warp(feature, dx, dy)
        self.assertGreater(float(shifted[0, 0, 2, 1]), 0.9)
        self.assertLess(float(shifted[0, 0, 2, 2]), 0.1)

    def test_relative_phase_aligns_analytic_ramps_on_valid_overlap(self):
        rows = torch.arange(2).view(2, 1) * 4
        cols = torch.arange(80).view(1, 80) * 4
        for ref_phase, other_phase in ((15, 0), (0, 15)):
            ref, other = torch.tensor([ref_phase]), torch.tensor([other_phase])
            ref_ramp = 2 * (rows + ref_phase // 4) + cols + ref_phase % 4
            other_ramp = 2 * (rows + other_phase // 4) + cols + other_phase % 4
            dx, dy = [axis.float().view(1, 1, 1) / 4 for axis in phase_shift(ref, other)]
            aligned, valid = warp(other_ramp.float()[None, None], dx, dy, return_validity=True)
            self.assertEqual(int(valid.sum()), 79)
            torch.testing.assert_close(aligned[valid], ref_ramp.float()[None, None][valid])

    def test_exact_support_keeps_original_positions_when_motion_is_zero(self):
        decoded = torch.rand(1, 2, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.tensor([[0, 15]])
        for alignment in ("phase", "motion"):
            model = FusionNet(2, 8, 1, alignment=alignment)
            _, geometric, active = model(decoded, phases, return_support=True)
            self.assertTrue(torch.equal(geometric, active))
            self.assertEqual(int(active.sum()), 320)

    def test_phase_mosaic_anchor_starts_at_the_direct_tiling(self):
        model = FusionNet(16, 8, 1, alignment="phase", anchor="phase-mosaic")
        decoded = torch.rand(1, 16, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.arange(16)[None]
        values, _ = reassemble(decoded, phases)
        torch.testing.assert_close(model(decoded, phases), values.sum(1))
        phases[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "distinct phases"):
            model(decoded, phases)

    def test_forward_shape_range_and_gradients(self):
        model = FusionNet(frames=3, channels=8, blocks=1)
        decoded = torch.rand(2, 3, 1, LR_HEIGHT, LR_WIDTH, requires_grad=True)
        phases = torch.tensor([[0, 1, 2], [3, 4, 5]])
        output = model(decoded, phases)
        self.assertEqual(output.shape, (2, 1, HEIGHT, WIDTH))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        # The residual head starts at zero, so the model begins as the bilinear base.
        # Perturb it before checking that gradients reach every frame.
        with torch.no_grad():
            model.tail.weight.normal_(std=0.01)
        output = model(decoded, phases)
        output.square().mean().backward()
        for frame in range(3):
            self.assertGreater(float(decoded.grad[:, frame].abs().sum()), 0.0)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in model.offset.parameters()))

    def test_every_frame_influences_the_output(self):
        model = FusionNet(frames=3, channels=8, blocks=1)
        with torch.no_grad():
            model.tail.weight.normal_(std=0.01)
        decoded = torch.rand(1, 3, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.tensor([[0, 1, 2]])
        base = model(decoded, phases)
        for frame in range(3):
            changed = decoded.clone()
            changed[:, frame] = torch.rand_like(changed[:, frame])
            self.assertGreater(float((model(changed, phases) - base).abs().max()), 1e-6)

    def test_exact_map_marks_the_measured_samples(self):
        model = FusionNet(frames=2, channels=8, blocks=1)
        decoded = torch.rand(1, 2, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.tensor([[0, 1]])
        values, mask = model.exact_map(decoded, phases)
        self.assertEqual(values.shape, (1, 2, 4, 8, 320))
        # Each frame measures 160 samples per CFA plane, a sixteenth of the 8x320 plane.
        self.assertEqual(int(mask[0, 0].sum()), LR_HEIGHT * LR_WIDTH)
        self.assertEqual(int(mask[0, 1].sum()), LR_HEIGHT * LR_WIDTH)
        # Different phases measure disjoint residue classes of the mosaic.
        self.assertEqual(int((mask[0, 0] & mask[0, 1]).sum()), 0)

    def test_invalid_inputs_raise(self):
        model = FusionNet(frames=2, channels=8, blocks=1)
        with self.assertRaisesRegex(ValueError, "expected decoded"):
            model(torch.rand(1, 2, 1, 8, 160), torch.zeros(1, 2, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "expected phases"):
            model(torch.rand(1, 2, 1, 4, 160), torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, "model expects"):
            model(torch.rand(1, 3, 1, 4, 160), torch.zeros(1, 3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "phase must be within"):
            model(torch.rand(1, 2, 1, 4, 160), torch.tensor([[0, 16]]))
        with self.assertRaisesRegex(ValueError, "invalid frames"):
            FusionNet(frames=0)


class V11PhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_phase_offsets_match_pipeline_definition(self):
        # training-data-preparation/internal/dataset/image.go:
        # px, py = ((phase-1)%4)*2, ((phase-1)/4)*2
        for phase in range(16):
            px, py = phase_offsets(torch.tensor(phase))
            self.assertEqual(int(px), ((phase) % 4) * 2)
            self.assertEqual(int(py), ((phase) // 4) * 2)
            self.assertEqual((int(px) % 2, int(py) % 2), (0, 0))

    def test_scatter_indices_place_every_decoded_pixel_once(self):
        for phase in range(PHASES):
            rows, cols = scatter_indices(phase)
            self.assertEqual(len(rows), LR_HEIGHT)
            self.assertEqual(len(cols), LR_WIDTH)
            pairs = set((r, c) for r in rows.tolist() for c in cols.tolist())
            self.assertEqual(len(pairs), LR_HEIGHT * LR_WIDTH)
            for row in rows.tolist():
                self.assertTrue(0 <= row < HEIGHT)
            for col in cols.tolist():
                self.assertTrue(0 <= col < WIDTH)

    def test_reassembly_is_exact_and_masked(self):
        torch.manual_seed(0)
        decoded = torch.rand(2, 3, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.tensor([[0, 1, 2], [3, 4, 5]])
        values, mask = reassemble(decoded, phases)
        self.assertEqual(values.shape, (2, 3, 1, HEIGHT, WIDTH))
        self.assertEqual(mask.shape, (2, 3, 1, HEIGHT, WIDTH))
        for b in range(2):
            for t in range(3):
                rows, cols = scatter_indices(int(phases[b, t]))
                placed = values[b, t, 0][rows[:, None], cols[None, :]]
                torch.testing.assert_close(placed, decoded[b, t, 0])
                self.assertTrue(mask[b, t, 0][rows[:, None], cols[None, :]].all())
                self.assertEqual(int(mask[b, t].sum()), LR_HEIGHT * LR_WIDTH)

    def test_sixteen_phases_tile_the_mosaic_exactly_once(self):
        decoded = torch.ones(1, PHASES, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.arange(PHASES).unsqueeze(0)
        values, mask = reassemble(decoded, phases)
        # The union of the 16 phases covers every high-resolution pixel exactly once.
        self.assertTrue(mask.any(dim=1).all())
        self.assertEqual(mask.sum(1).unique().tolist(), [1])
        # Measured positions hold the decoded value; unmeasured positions stay zero.
        torch.testing.assert_close(values[mask], torch.ones(int(mask.sum())))
        self.assertEqual(int(values.sum()), PHASES * LR_HEIGHT * LR_WIDTH)

    def test_coverage_is_t_over_sixteen_for_distinct_phases(self):
        decoded = torch.zeros(1, 4, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.tensor([[0, 1, 2, 3]])
        _, mask = reassemble(decoded, phases)
        ratio = coverage(mask)
        # Each frame measures 1/16 on its own; the window measures 4/16 in total.
        torch.testing.assert_close(ratio, torch.full((1, 4), 1 / PHASES))
        self.assertAlmostEqual(float(ratio.sum()), 4 / PHASES, places=5)
        self.assertTrue(mask.any(dim=1).sum().item() == 4 * LR_HEIGHT * LR_WIDTH)

    def test_repeated_phases_overlap_instead_of_adding_coverage(self):
        decoded = torch.zeros(1, 2, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.tensor([[5, 5]])
        _, mask = reassemble(decoded, phases)
        ratio = coverage(mask)
        torch.testing.assert_close(ratio, torch.full((1, 2), 1 / PHASES))

    def test_exact_measurements_counts_and_averages(self):
        decoded = torch.rand(1, 2, 1, LR_HEIGHT, LR_WIDTH)
        phases = torch.tensor([[0, 1]])
        values, mask = reassemble(decoded, phases)
        fused, count = exact_measurements(values, mask)
        self.assertEqual(fused.shape, (1, 1, 1, HEIGHT, WIDTH))
        self.assertEqual(count.shape, (1, 1, 1, HEIGHT, WIDTH))
        self.assertEqual(int(count.max()), 1)
        self.assertEqual(int(count.sum()), 2 * LR_HEIGHT * LR_WIDTH)
        # The fused value is the mean over the frames that actually measured the pixel.
        measured = (values * mask).sum(1, keepdim=True) / count.clamp_min(1)
        torch.testing.assert_close(fused, measured)
        # Unmeasured pixels stay zero instead of averaging in the zeros of other frames.
        unmeasured = count[0, 0, 0] == 0
        self.assertTrue(torch.all(fused[0, 0, 0][unmeasured] == 0))

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "phases must be int64"):
            phase_offsets(torch.zeros(1, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "expected decoded"):
            reassemble(torch.rand(1, 2, 1, 8, 160), torch.zeros(1, 2, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "expected phases"):
            reassemble(torch.rand(1, 2, 1, 4, 160), torch.zeros(1, 3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "phase must be within"):
            reassemble(torch.rand(1, 2, 1, 4, 160), torch.tensor([[0, 16]]))
        with self.assertRaisesRegex(ValueError, "phase must be within"):
            reassemble(torch.rand(1, 2, 1, 4, 160), torch.tensor([[0, -1]]))

    def test_reassembly_matches_the_go_pipeline_inverse(self):
        """The Go Downsample copies raw[(8j+py+dy)*640 + 8i+px+dx] into the decoded strip."""
        torch.manual_seed(1)
        raw = torch.rand(HEIGHT, WIDTH)
        for phase in range(PHASES):
            px, py = int(phase_offsets(torch.tensor(phase))[0]), int(phase_offsets(torch.tensor(phase))[1])
            low = torch.empty(LR_HEIGHT, LR_WIDTH)
            for j in range(LR_HEIGHT // 2):
                for i in range(LR_WIDTH // 2):
                    for dy in range(2):
                        for dx in range(2):
                            low[2*j+dy, 2*i+dx] = raw[8*j+py+dy, 8*i+px+dx]
            values, mask = reassemble(low.view(1, 1, 1, LR_HEIGHT, LR_WIDTH),
                                      torch.tensor([[phase]]))
            rows, cols = scatter_indices(phase)
            torch.testing.assert_close(values[0, 0, 0][rows[:, None], cols[None, :]], low)
            self.assertEqual(int(mask.sum()), LR_HEIGHT * LR_WIDTH)


if __name__ == "__main__":
    unittest.main()
