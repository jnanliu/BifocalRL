"""Exact full-vocabulary statistics, including tensor-parallel vocabularies."""

import math

import torch
import torch.distributed as dist


@torch.no_grad()
def vocabulary_statistics(logits, *, vocab_size=None, group=None, chunk_size=256):
    """Return -mean(log p) and H(p), before sampling temperature/truncation.

    Only a row chunk is materialized in FP32. Padding vocabulary entries are
    excluded. ``group=None`` denotes an unsharded vocabulary, even under DDP.
    """
    world = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    width = logits.shape[-1]
    vocab_size = vocab_size or width * world
    if not 0 < vocab_size <= width * world:
        raise ValueError("Invalid true vocabulary size")
    valid = max(0, min(width, vocab_size - rank * width))
    sc = torch.empty(logits.shape[0], device=logits.device, dtype=torch.float32)
    entropy = torch.empty_like(sc)

    def reduce(x, op=dist.ReduceOp.SUM):
        if world > 1:
            dist.all_reduce(x, op=op, group=group)
        return x

    for start in range(0, len(logits), max(1, chunk_size)):
        end = min(start + max(1, chunk_size), len(logits))
        z = logits[start:end, :valid].float()
        maximum = z.max(-1).values if valid else torch.full_like(sc[start:end], -math.inf)
        reduce(maximum, dist.ReduceOp.MAX)
        shifted = z - maximum[:, None]
        exps = shifted.exp()
        denom = reduce(exps.sum(-1))
        log_z = maximum + denom.log()
        sc[start:end] = log_z - reduce(z.sum(-1)) / vocab_size
        # Shift first to avoid cancellation and inf * zero at padded entries.
        entropy[start:end] = denom.log() - reduce((exps * shifted).sum(-1)) / denom
    return sc, entropy
