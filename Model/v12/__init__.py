"""V12 deterministic RAW8 blend of phase-aligned Bayer cells."""
import torch

from v11.phase import HEIGHT, LR_HEIGHT, LR_WIDTH, PHASES, TILE, WIDTH, phase_offsets, scatter_indices


HISTORY = 16
PHASE_DECAY_POINTS = 2


def nearest_bayer_bytes(decoded, phases):
    """Fill each output Bayer cell by copying a whole nearest decoded cell."""
    if decoded.dtype != torch.uint8 or decoded.ndim != 4 or decoded.shape[1:] != (1, LR_HEIGHT, LR_WIDTH):
        raise ValueError("expected decoded RAW8 strips")
    if phases.shape != (len(decoded),) or phases.dtype != torch.long or torch.any((phases < 0) | (phases >= PHASES)):
        raise ValueError("expected one phase in 0..15 per strip")
    px, py = phase_offsets(phases.to(decoded.device))
    scale = TILE // 2
    rows = torch.arange(HEIGHT // 2, device=decoded.device) - py[:, None] // 2
    cols = torch.arange(WIDTH // 2, device=decoded.device) - px[:, None] // 2
    rows = ((rows + scale // 2) // scale).clamp(0, LR_HEIGHT // 2 - 1)
    cols = ((cols + scale // 2) // scale).clamp(0, LR_WIDTH // 2 - 1)
    cells = decoded.reshape(-1, LR_HEIGHT // 2, 2, LR_WIDTH // 2, 2).permute(0, 1, 3, 2, 4)
    batch = torch.arange(len(decoded), device=decoded.device)
    nearest = cells[batch[:, None, None], rows[:, :, None], cols[:, None, :]]
    return nearest.permute(0, 1, 3, 2, 4).reshape(len(decoded), 1, HEIGHT, WIDTH)


def blend_bytes(frames):
    """Fold sixteen RAW8 tensors, newest first, using floor((a + b) / 2)."""
    if len(frames) != HISTORY:
        raise ValueError("provide the current frame and fifteen consecutive frames")
    mixed = frames[0]
    for older in frames[1:]:
        if older.dtype != torch.uint8 or mixed.dtype != torch.uint8 or older.shape != mixed.shape:
            raise ValueError("all inputs must be equally sized RAW8 tensors")
        mixed = ((mixed.to(torch.int16) + older) // 2).to(torch.uint8)
    return mixed


def phase_protected_blend(frames, decoded, phases, decay_points=PHASE_DECAY_POINTS):
    """Give each measured cell a weight of 100 minus its age times decay_points."""
    if len(frames) != HISTORY or len(decoded) != HISTORY or len(phases) != HISTORY:
        raise ValueError("provide the current frame and fifteen consecutive frames")
    if any(b.dtype != torch.uint8 or b.shape != (1, LR_HEIGHT, LR_WIDTH) for b in decoded):
        raise ValueError("expected decoded RAW8 strips")
    if any(not 0 <= phase < PHASES for phase in phases):
        raise ValueError("expected phases in 0..15")
    if not 0 <= decay_points <= 100 // (HISTORY - 1):
        raise ValueError("phase decay would make a weight negative")
    owners = {}
    for age, phase in enumerate(phases):
        owners.setdefault(phase, age)
    shape = (1, HEIGHT, WIDTH)
    if decay_points == 0:
        output = decoded[0].new_empty(shape) if len(owners) == PHASES else blend_bytes(frames)
        for phase, age in owners.items():
            rows, cols = scatter_indices(phase, output.device)
            output[:, rows[:, None], cols[None, :]] = decoded[age]
        return output

    owner_age = torch.full(shape, -1, dtype=torch.int16, device=decoded[0].device)
    measured = decoded[0].new_zeros(shape)
    for phase, age in owners.items():
        rows, cols = scatter_indices(phase, measured.device)
        owner_age[:, rows[:, None], cols[None, :]] = age
        measured[:, rows[:, None], cols[None, :]] = decoded[age]
    # Start with the newest eligible frame and skip the owner in the original fold.
    background = torch.where(owner_age == 0, frames[1], frames[0]).to(torch.int16)
    for age in range(1, HISTORY):
        include = (owner_age != age) & ((owner_age != 0) | (age != 1))
        background = torch.where(include, (background + frames[age]) // 2, background)
    weight = torch.where(owner_age >= 0, 100 - decay_points * owner_age, 0)
    return ((measured * weight + background * (100 - weight)) // 100).to(torch.uint8)
