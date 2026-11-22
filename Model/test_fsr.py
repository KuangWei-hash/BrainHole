import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fsr_reference import fsr_passes, low_rgb, make_atlas, read_atlas, rgb_to_raw


class FsrTests(unittest.TestCase):
    def test_cfa_projection(self):
        rgb = torch.tensor([0.1, 0.4, 0.9]).reshape(1, 3, 1, 1).expand(2, 3, 16, 640)
        raw = rgb_to_raw(rgb)
        for y, x, value in ((0, 0, .1), (0, 1, .4), (1, 0, .4), (1, 1, .9)):
            torch.testing.assert_close(raw[:, 0, y::2, x::2], torch.full((2, 8, 320), value))

    def test_rgb_adapter_all_phases_affine_scene(self):
        # Independent analytic scene: sample each physical CFA location, then
        # check interpolation at the declared output coordinates with border clamp.
        low = torch.empty(16, 1, 4, 160)
        expected = torch.empty(16, 4, 4, 160)
        x, y = torch.arange(160)[None, :]*4+1.5, torch.arange(4)[:, None]*4+1.5
        for phase in range(16):
            px, py = 2*(phase % 4), 2*(phase//4)
            for channel, (cy, cx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                sx, sy = torch.arange(80)[None, :]*8+px+cx, torch.arange(2)[:, None]*8+py+cy
                bias = .1+channel*.1
                low[phase, 0, cy::2, cx::2] = bias+sx*.0002+sy*.001
                expected[phase, channel] = bias+x.clamp(px+cx, 632+px+cx)*.0002+y.clamp(py+cy, 8+py+cy)*.001
        rgb = low_rgb(low, torch.arange(16))
        torch.testing.assert_close(rgb, torch.stack((expected[:, 0], (expected[:, 1]+expected[:, 2])*.5, expected[:, 3]), 1))

    def test_official_cli_constant_and_tile_isolation(self):
        override = os.getenv("BRAINHOLE_TEST_FSR_CLI")
        exe = Path(override) if override else Path(__file__).parent/"tools/fsr1/v1.0.3/FidelityFX_CLI.exe"
        if not exe.exists():
            self.skipTest("AMD official CLI is not installed")
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            rgb = torch.zeros(3, 3, 4, 160)
            rgb[0] = .25
            rgb[1] = torch.arange(160)[None, None, :]/159
            rgb[2, :, :, ::2] = 1
            atlas, solo = folder/"atlas.png", folder/"solo.png"
            make_atlas(rgb, columns=2).save(atlas)
            make_atlas(rgb[1:2], columns=1).save(solo)
            easu, rcas = fsr_passes(exe, [atlas, solo], folder)
            for stage in (easu, rcas):
                batch = read_atlas(stage[0], 3, columns=2)
                single = read_atlas(stage[1], 1, columns=1)
                self.assertTrue(torch.isfinite(batch).all())
                torch.testing.assert_close(batch[0], torch.full_like(batch[0], 64/255), atol=1/255, rtol=0)
                # Atlas neighbours have contradictory content; any seam leakage
                # or hidden context changes the result relative to a solo strip.
                torch.testing.assert_close(batch[1:2], single, atol=1/255, rtol=0)


if __name__ == "__main__":
    unittest.main()
