"""Target-free RAW8 inputs for the fixed v12 base and v13 CNN."""
import torch

from v11.phase import scatter_indices
from v12 import HISTORY, nearest_bayer_bytes, phase_protected_blend


def features(decoded, phases):
    """Return stacked byte planes B, M, A and V for a newest-first window."""
    if not 1 <= len(decoded) <= HISTORY:
        raise ValueError("expected one to sixteen decoded strips")
    nearest = nearest_bayer_bytes(decoded, torch.as_tensor(phases, dtype=torch.long))
    base = nearest[0]
    if len(decoded) == HISTORY:
        base = phase_protected_blend(nearest, decoded, phases, decay_points=2)
    measured = torch.zeros_like(base)
    ages = torch.full_like(base, HISTORY - 1)
    valid = torch.zeros_like(base)
    owners = {}
    for age, phase in enumerate(phases):
        owners.setdefault(phase, age)
    for phase, age in owners.items():
        rows, cols = scatter_indices(phase, decoded.device)
        grid = (slice(None), rows[:, None], cols[None, :])
        measured[grid], ages[grid], valid[grid] = decoded[age], age, 1
    return torch.cat((base, measured, ages, valid))
