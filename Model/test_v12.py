import unittest

import torch

from v12 import HISTORY, blend_bytes, nearest_bayer_bytes, phase_protected_blend


class V12Tests(unittest.TestCase):
    def test_each_mix_truncates_unsigned_bytes_without_overflow(self):
        current = torch.tensor([255, 255, 0, 1], dtype=torch.uint8)
        older = torch.tensor([255, 254, 255, 0], dtype=torch.uint8)
        blended = torch.tensor([255, 254, 127, 0], dtype=torch.uint8)
        frames = [current, older] + [blended] * (HISTORY - 2)
        torch.testing.assert_close(blend_bytes(frames), blended)

    def test_newest_first_accumulation_gives_last_frame_half_the_weight(self):
        black = torch.tensor([0], dtype=torch.uint8)
        white = torch.tensor([255], dtype=torch.uint8)
        self.assertEqual(blend_bytes([black] * 15 + [white]).item(), 127)
        self.assertEqual(blend_bytes([white] + [black] * 15).item(), 0)

    def test_requires_complete_window(self):
        with self.assertRaisesRegex(ValueError, "fifteen"):
            blend_bytes([torch.tensor([0], dtype=torch.uint8)] * 15)

    def test_nearest_copies_whole_bayer_cells_with_phase_and_border_clamp(self):
        decoded = torch.arange(640).reshape(1, 1, 4, 160).to(torch.uint8).repeat(2, 1, 1, 1)
        actual = nearest_bayer_bytes(decoded, torch.tensor([0, 15]))
        self.assertEqual(actual.shape, (2, 1, 16, 640))
        self.assertEqual(actual.dtype, torch.uint8)
        for batch, offset in enumerate((0, 3)):
            for y, x in ((0, 0), (1, 1), (2, 2), (4, 4), (7, 319), (5, 5)):
                row = min(range(2), key=lambda index: (abs(y - (4 * index + offset)), -index))
                col = min(range(80), key=lambda index: (abs(x - (4 * index + offset)), -index))
                expected = decoded[batch, :, 2*row:2*row+2, 2*col:2*col+2]
                torch.testing.assert_close(actual[batch, :, 2*y:2*y+2, 2*x:2*x+2], expected)

    def test_phase_protection_tiles_original_bytes_by_bayer_cell(self):
        phases = list(reversed(range(HISTORY)))
        decoded = [torch.arange(640).reshape(1, 4, 160).add(phase * 17).to(torch.uint8) for phase in phases]
        enlarged = torch.zeros(HISTORY, 1, 16, 640, dtype=torch.uint8)
        output = phase_protected_blend(enlarged, decoded, phases, decay_points=0)
        expected = torch.empty_like(output)
        for phase, values in zip(phases, decoded):
            y, x = 2 * (phase // 4), 2 * (phase % 4)
            for dy in range(2):
                for dx in range(2):
                    expected[:, y+dy::8, x+dx::8] = values[:, dy::2, dx::2]
        torch.testing.assert_close(output, expected)

    def test_phase_weight_decreases_by_age_and_excludes_owner_from_background(self):
        for age, expected in ((0, 200), (1, 198), (15, 170)):
            enlarged = torch.full((HISTORY, 1, 16, 640), 100, dtype=torch.uint8)
            enlarged[age].fill_(255)
            decoded = torch.full((HISTORY, 1, 4, 160), 200, dtype=torch.uint8)
            output = phase_protected_blend(enlarged, decoded, list(range(HISTORY)))
            y, x = 2 * (age // 4), 2 * (age % 4)
            self.assertEqual(output[0, y, x].item(), expected)

    def test_unowned_positions_keep_the_original_fifty_fifty_fold(self):
        enlarged = torch.arange(HISTORY, dtype=torch.uint8).reshape(HISTORY, 1, 1, 1)
        enlarged = enlarged.expand(-1, -1, 16, 640)
        decoded = torch.zeros((HISTORY, 1, 4, 160), dtype=torch.uint8)
        output = phase_protected_blend(enlarged, decoded, [0] * HISTORY)
        self.assertEqual(output[0, 0, 2].item(), blend_bytes(enlarged)[0, 0, 2].item())

    def test_duplicate_phase_keeps_newest_and_other_positions_use_fixed_blend(self):
        enlarged = torch.full((HISTORY, 1, 16, 640), 70, dtype=torch.uint8)
        newest = torch.full((1, 4, 160), 23, dtype=torch.uint8)
        older = torch.full((1, 4, 160), 91, dtype=torch.uint8)
        output = phase_protected_blend(enlarged, [newest] + [older] * 15, [15] * 16)
        expected = enlarged[0].clone()
        for dy in range(2):
            for dx in range(2):
                expected[:, 6+dy::8, 6+dx::8] = 23
        torch.testing.assert_close(output, expected)
