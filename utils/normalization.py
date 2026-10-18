"""
Normalization utilities for reward scoring.

Supports:
- Z-score normalization with configurable target mean/std and clipping.
- Unit-interval [0,1] normalization via z-score + sigmoid for capability ratio inputs.
- Min-max normalization to [0,1] along a given dimension.
- GRPO per-group advantages and batch-level z-score (scalar mean/std over all elements).
"""

from __future__ import annotations

from typing import Optional

import torch


def z_score_normalize_tensor(
    scores: torch.Tensor,
    *,
    mu: float = 5.0,
    sigma: float = 1.0,
    clip_threshold: float = 3.0,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Z-score normalize `scores` along `dim`, then map to N(mu, sigma) and clip to [mu-clip, mu+clip].

    This matches the behavior in `old/normalization.py`:
    - Uses population std (ddof=0 / unbiased=False)
    - If std is zero (or invalid), returns a constant tensor filled with `mu`
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if clip_threshold <= 0:
        raise ValueError("clip_threshold must be positive")

    if scores.numel() == 0:
        return scores

    # Handle degenerate case: only one element along dim -> cannot compute meaningful std
    if scores.size(dim) == 1:
        return torch.full_like(scores, float(mu))

    mean = scores.mean(dim=dim, keepdim=True)
    std = scores.std(dim=dim, keepdim=True, unbiased=False)

    # If std is zero/NaN/Inf, fall back to constant mu
    invalid = torch.isnan(std) | torch.isinf(std) | (std.abs() < eps)
    if torch.any(invalid):
        # Expand invalid mask to full shape for a safe where
        # (If std is invalid for any slice, fill that slice with mu)
        # We approximate by filling *all* positions where std is invalid.
        fill = torch.full_like(scores, float(mu))
        z = (scores - mean) / torch.where(invalid, torch.ones_like(std), std)
        mapped = z * float(sigma) + float(mu)
        mapped = torch.where(invalid, fill, mapped)
        return torch.clamp(mapped, float(mu) - float(clip_threshold), float(mu) + float(clip_threshold))

    z = (scores - mean) / std
    mapped = z * float(sigma) + float(mu)
    return torch.clamp(mapped, float(mu) - float(clip_threshold), float(mu) + float(clip_threshold))


def z_score_normalize_tensor_with_bounds(
    scores: torch.Tensor,
    *,
    mu: float,
    sigma: float,
    lower: float,
    upper: float,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Z-score normalize `scores` along `dim`, then map to N(mu, sigma) and clip to [lower, upper].

    Unlike `z_score_normalize_tensor`, this variant takes explicit clipping bounds
    instead of deriving them from a `clip_threshold`.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if lower >= upper:
        raise ValueError("lower must be < upper")

    if scores.numel() == 0:
        return scores

    clipped_mu = min(max(float(mu), float(lower)), float(upper))

    if scores.size(dim) == 1:
        return torch.full_like(scores, clipped_mu)

    mean = scores.mean(dim=dim, keepdim=True)
    std = scores.std(dim=dim, keepdim=True, unbiased=False)

    invalid = torch.isnan(std) | torch.isinf(std) | (std.abs() < eps)
    if torch.any(invalid):
        fill = torch.full_like(scores, clipped_mu)
        z = (scores - mean) / torch.where(invalid, torch.ones_like(std), std)
        mapped = z * float(sigma) + float(mu)
        mapped = torch.where(invalid, fill, mapped)
        return torch.clamp(mapped, float(lower), float(upper))

    z = (scores - mean) / std
    mapped = z * float(sigma) + float(mu)
    return torch.clamp(mapped, float(lower), float(upper))


def normalize_to_unit_interval(
    scores: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Z-score normalize *scores* along *dim*, then map to [0, 1] via sigmoid.

    Steps:
      1. ``z = (x - mean) / std``   (population std, same as ``z_score_normalize_tensor``)
      2. ``out = sigmoid(z)``        (smooth, bounded, monotone)

    Degenerate cases (single element, zero/NaN std) map to 0.5 (sigmoid(0)).
    """
    if scores.numel() == 0:
        return scores

    if scores.size(dim) == 1:
        return torch.full_like(scores, 0.5)

    mean = scores.mean(dim=dim, keepdim=True)
    std = scores.std(dim=dim, keepdim=True, unbiased=False)

    invalid = torch.isnan(std) | torch.isinf(std) | (std.abs() < eps)
    safe_std = torch.where(invalid, torch.ones_like(std), std)
    z = (scores - mean) / safe_std
    z = torch.where(invalid.expand_as(z), torch.zeros_like(z), z)

    return torch.sigmoid(z)


def minmax_normalize(
    scores: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Min-max normalize *scores* to [0, 1] along *dim*.

    ``out = (x - min) / (max - min + eps)``

    Degenerate cases (single element, max == min) map to 0.5.
    """
    if scores.numel() == 0:
        return scores
    if scores.size(dim) <= 1:
        return torch.full_like(scores, 0.5)

    lo = scores.min(dim=dim, keepdim=True).values
    hi = scores.max(dim=dim, keepdim=True).values
    denom = hi - lo
    degenerate = denom.abs() < eps
    safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
    normed = (scores - lo) / (safe_denom + eps)
    return torch.where(degenerate.expand_as(normed), torch.full_like(normed, 0.5), normed)


def grpo_group_advantages(
    rewards: torch.Tensor,
    *,
    dim: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """GRPO group-relative advantages: ``(R - mean) / (std + eps)`` per group.

    Each row (prompt) is normalized independently so that advantages have
    zero mean and approximately unit variance within each prompt group.

    Degenerate cases (single element, zero std) map to 0.

    Args:
        rewards: ``(num_prompts, group_size)`` reward tensor.
        dim: Dimension to normalize over (default 1 = per prompt).
        eps: Numerical stability constant.

    Returns:
        Same-shape tensor of group-relative advantages.
    """
    if rewards.numel() == 0:
        return rewards
    if rewards.size(dim) <= 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean(dim=dim, keepdim=True)
    std = rewards.std(dim=dim, keepdim=True)
    invalid = torch.isnan(std) | torch.isinf(std) | (std.abs() < eps)
    safe_std = torch.where(invalid, torch.ones_like(std), std)
    adv = (rewards - mean) / (safe_std + eps)
    return torch.where(invalid.expand_as(adv), torch.zeros_like(adv), adv)


def batch_zscore_normalize(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Z-score over **all** elements of ``rewards`` (batch-level), not per group.

    ``out = (x - mean(x)) / (std(x) + eps)`` with scalar mean and population std
    (``unbiased=False``), matching typical batch statistics.

    Degenerate cases (empty tensor, single element, invalid or near-zero std)
    return zeros of the same shape.

    Args:
        rewards: Any-shaped tensor (e.g. ``(P, G)`` advantages).
        eps: Numerical stability constant in the denominator.

    Returns:
        Same-shaped tensor of batch-normalized values.
    """
    if rewards.numel() == 0:
        return rewards
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    invalid = torch.isnan(std) | torch.isinf(std) | (std.abs() < eps)
    if bool(invalid.item()):
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


def rowwise_pearson_corrcoef(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = 1,
    keepdim: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Row-wise Pearson correlation between two same-shaped tensors.

    For mixed_reward this is typically used on ``(P, G)`` advantage tensors,
    producing one correlation score per prompt.

    Degenerate slices (single element, zero variance, NaN/Inf denominator) return 0.
    Output is clamped to ``[-1, 1]`` for numerical safety.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    if x.numel() == 0:
        return x

    if x.size(dim) <= 1:
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            del shape[dim]
        return x.new_zeros(shape)

    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    cov = (x_centered * y_centered).mean(dim=dim, keepdim=True)
    x_std = x_centered.pow(2).mean(dim=dim, keepdim=True).sqrt()
    y_std = y_centered.pow(2).mean(dim=dim, keepdim=True).sqrt()

    denom = x_std * y_std
    invalid = torch.isnan(denom) | torch.isinf(denom) | (denom.abs() < eps)
    safe_denom = torch.where(invalid, torch.ones_like(denom), denom)
    corr = cov / safe_denom
    corr = torch.where(invalid, torch.zeros_like(corr), corr)
    corr = torch.clamp(corr, -1.0, 1.0)

    if keepdim:
        return corr
    return corr.squeeze(dim)
