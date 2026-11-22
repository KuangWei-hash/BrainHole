"""RGB input preparation and isolated tile batching for AMD's official FSR1 CLI.

FSR is not a Bayer scaler. The adapter reconstructs a regular RGB grid from
the four CFA planes, respecting their actual coordinates and frame phase.
Two EASU 2x passes supply this dataset's 4x linear scaling, then optional RCAS.
"""
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def low_rgb(low, phases):
    """Bx1x4x160 CFA -> Bx3x4x160 display-encoded RGB at high-res 4u+1.5.

    Each source CFA plane sample is at (8k+2*phase_x+color_x,
    8j+2*phase_y+color_y) in target coordinates. The half-pixel convention
    below makes ordinary centered 4x RGB scaling land on target pixel centers.
    Border extension is per strip; no other frame/block is accessed.
    """
    if low.shape[1:] != (1, 4, 160) or phases.shape != (len(low),):
        raise ValueError("expected low [B,1,4,160] and phases [B]")
    planes = F.pixel_unshuffle(low, 2)
    x = torch.arange(160, device=low.device, dtype=low.dtype)[None, None, :]*4+1.5
    y = torch.arange(4, device=low.device, dtype=low.dtype)[None, :, None]*4+1.5
    px, py = (phases % 4)[:, None, None]*2, (phases // 4)[:, None, None]*2
    colors = []
    for channel, (cy, cx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        gx = (((x-px-cx)/8)/79*2-1).expand(-1, 4, -1)
        gy = ((y-py-cy)/8*2-1).expand(-1, -1, 160)
        grid = torch.stack((gx, gy), -1)
        colors.append(F.grid_sample(planes[:, channel:channel+1], grid,
                                    padding_mode="border", align_corners=True))
    return torch.cat((colors[0], (colors[1]+colors[2])*0.5, colors[3]), 1).clamp(0, 1)


def rgb_to_raw(rgb):
    """Sample RGB with the target RGGB CFA; not a demosaicing inverse."""
    raw = rgb[:, :1].clone()
    raw[:, 0, ::2, 1::2] = rgb[:, 1, ::2, 1::2]
    raw[:, 0, 1::2, ::2] = rgb[:, 1, 1::2, ::2]
    raw[:, 0, 1::2, 1::2] = rgb[:, 2, 1::2, 1::2]
    return raw


def make_atlas(rgb, columns=16, guard=8):
    """Independent strip tiles, each with replicated borders before filtering."""
    pixels = (rgb.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()*255).round().astype(np.uint8)
    count, height, width, _ = pixels.shape
    rows = (count+columns-1)//columns
    tile_h, tile_w = height+2*guard, width+2*guard
    atlas = np.zeros((rows*tile_h, columns*tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(pixels):
        row, column = divmod(index, columns)
        atlas[row*tile_h:(row+1)*tile_h, column*tile_w:(column+1)*tile_w] = np.pad(
            tile, ((guard, guard), (guard, guard), (0, 0)), mode="edge")
    return Image.fromarray(atlas)


def read_atlas(path, count, columns=16, guard=8, scale=4):
    with Image.open(path) as image:
        atlas = np.asarray(image.convert("RGB"))
    height, width = 4*scale, 160*scale
    tile_h, tile_w, padding = (4+2*guard)*scale, (160+2*guard)*scale, guard*scale
    expected = (((count+columns-1)//columns)*tile_h, columns*tile_w)
    if atlas.shape[:2] != expected:
        raise ValueError(f"FSR output shape {atlas.shape[:2]} != {expected}")
    tiles = []
    for index in range(count):
        row, column = divmod(index, columns)
        y, x = row*tile_h+padding, column*tile_w+padding
        tiles.append(atlas[y:y+height, x:x+width].copy())
    return torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float()/255


def run_cli(executable, pairs, options, log):
    """Official D3D11 shaders; batched file arguments, no shell or GUI window."""
    command = [str(Path(executable).resolve()), *options]
    # Stay well below the Windows 32767 character command-line limit.
    with Path(log).open("ab") as stream:
        for start in range(0, len(pairs), 24):
            args = [item for pair in pairs[start:start+24] for item in map(str, pair)]
            subprocess.run(command+args, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def fsr_passes(executable, inputs, folder, sharpness=0.0):
    """PNG 8-bit boundaries, FP16 textures; preserve all intermediate files."""
    folder = Path(folder)
    second, enlarged, sharpened = [], [], []
    for index in range(len(inputs)):
        second.append(folder/f"easu2-{index:03d}.png")
        enlarged.append(folder/f"easu4-{index:03d}.png")
        sharpened.append(folder/f"rcas-{index:03d}.png")
    for source, destination, options, name in (
        (inputs, second, ["-Mode", "EASU", "-Scale", "2x", "2x", "-FP16"], "easu2"),
        (second, enlarged, ["-Mode", "EASU", "-Scale", "2x", "2x", "-FP16"], "easu4"),
        (enlarged, sharpened, ["-Mode", "RCAS", "-Sharpness", str(sharpness), "-FP16"], "rcas"),
    ):
        run_cli(executable, list(zip(source, destination)), options, folder/f"{name}.log")
        print(f"Official FSR pass {name}: {len(inputs)} atlases", flush=True)
    return enlarged, sharpened
