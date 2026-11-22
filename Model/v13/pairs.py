"""Pair consecutive training targets from the same strip and source window."""
from collections import defaultdict

import torch


def adjacent_groups(manifest, replaced=()):
    by_block = defaultdict(list)
    for index, entry in enumerate(manifest):
        by_block[entry["block"]].append((index, entry))
    excluded = set(replaced)
    pairs, singles = [], []
    for group in by_block.values():
        pending = None
        for index, row in group:
            if pending is not None:
                prior_index, prior = pending
                if (prior_index not in excluded and index not in excluded
                        and prior["frame"] + 1 == row["frame"] and prior["first"] + 1 == row["first"]
                        and prior["header"] == row["header"]):
                    pairs.append([prior_index, index])
                    pending = None
                    continue
                singles.append(prior_index)
            pending = (index, row)
        if pending is not None:
            singles.append(pending[0])
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2), torch.tensor(singles, dtype=torch.long)


def pair_loss(prediction, target, mask):
    error = prediction - target
    pair_count = len(error) // 2
    changes = (error[1::2] - error[:2 * pair_count:2]).abs().mean((1, 2, 3))
    return (changes * mask).sum() / mask.sum().clamp_min(1)


def previous_indices(manifest, replaced=()):
    excluded = set(replaced)
    history = {}
    result = torch.full((len(manifest),), -1, dtype=torch.long)
    for index, row in enumerate(manifest):
        block, frame, first, header = (row[key] for key in ("block", "frame", "first", "header"))
        prior = history.get(block)
        if prior is not None and index not in excluded:
            other_index, other_frame, other_first, other_header = prior
            if (other_index not in excluded and frame == other_frame + 1
                    and first == other_first + 1 and header == other_header):
                result[index] = other_index
        history[block] = (index, frame, first, header)
    return result
