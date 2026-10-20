"""Validation, strategy dispatch and shared advantage normalization."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

from ..normalization import batch_zscore_normalize, grpo_group_advantages
from .config import MixedRewardConfig
from .registry import get_mixed_reward_mode
from .strategies import (
    _capture_mixed_reward_diagnostic,
    _log_mixed_reward_tensor_stats,
)
from .types import MixedRewardContext, MixedRewardResult


def compute_mixed_reward(
    mv_rewards: torch.Tensor,
    intuitor_rewards: torch.Tensor,
    *,
    config: MixedRewardConfig | None = None,
    num_generations_per_model: int | None = None,
    diagnostics: MutableMapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Normalize views, fuse/filter, normalize retained rollouts, apply optional weights.

    This is the config-based API for training code and new experiments.
    Training supplies cached, detached rewards; this API preserves tensor autograd
    behavior for callers that explicitly require gradients.
    """
    config = config if config is not None else MixedRewardConfig()
    eps = config.eps
    alpha_mode = config.alpha_mode
    final_normalize = config.final_normalize
    if mv_rewards.dim() != 2 or intuitor_rewards.dim() != 2:
        raise ValueError(
            "mixed_reward rewards must be 2D tensors shaped (P, G), "
            f"got {tuple(mv_rewards.shape)} and {tuple(intuitor_rewards.shape)}"
        )
    if mv_rewards.shape != intuitor_rewards.shape:
        raise ValueError(
            "mixed_reward reward tensors must have the same shape, "
            f"got {tuple(mv_rewards.shape)} and {tuple(intuitor_rewards.shape)}"
        )
    if mv_rewards.numel() == 0:
        raise ValueError("mixed_reward reward tensors must be non-empty")
    if not bool(torch.isfinite(mv_rewards).all().item()):
        raise ValueError("mixed_reward mv_rewards contains non-finite values")
    if not bool(torch.isfinite(intuitor_rewards).all().item()):
        raise ValueError("mixed_reward intuitor_rewards contains non-finite values")

    adv_mv = grpo_group_advantages(mv_rewards, eps=eps)
    adv_int = grpo_group_advantages(intuitor_rewards, eps=eps)
    _capture_mixed_reward_diagnostic(diagnostics, "mv_rewards", mv_rewards)
    _capture_mixed_reward_diagnostic(diagnostics, "self_certainty_rewards", intuitor_rewards)
    _capture_mixed_reward_diagnostic(diagnostics, "mv_advantages", adv_mv)
    _capture_mixed_reward_diagnostic(diagnostics, "self_certainty_advantages", adv_int)

    mode = str(alpha_mode).strip().lower()
    _capture_mixed_reward_diagnostic(diagnostics, "alpha_mode", mode)
    _capture_mixed_reward_diagnostic(diagnostics, "final_normalize", bool(final_normalize))
    context = MixedRewardContext(
        mv_rewards=mv_rewards,
        intuitor_rewards=intuitor_rewards,
        adv_mv=adv_mv,
        adv_int=adv_int,
        config=config,
        num_generations_per_model=num_generations_per_model,
        diagnostics=diagnostics,
        metadata=metadata,
    )
    result = get_mixed_reward_mode(alpha_mode)(context)
    if not isinstance(result, MixedRewardResult):
        raise TypeError("Mixed-reward strategies must return MixedRewardResult")
    combined = result.advantages
    if combined.shape != mv_rewards.shape:
        raise ValueError("Mixed-reward strategy changed the (queries, responses) shape")
    post_normalize_sample_weight = result.sample_weight
    keep_mask = result.keep_mask
    if keep_mask is not None and (keep_mask.shape != combined.shape or keep_mask.dtype != torch.bool):
        raise ValueError("Mixed-reward keep mask must be boolean with shape (queries, responses)")
    _log_mixed_reward_tensor_stats("combined_before_final_normalize", combined)
    _capture_mixed_reward_diagnostic(diagnostics, "combined_before_final_normalize", combined)

    if final_normalize:
        if keep_mask is None:
            combined = batch_zscore_normalize(combined, eps=eps)
        else:
            normalized = batch_zscore_normalize(combined[keep_mask], eps=eps)
            combined = torch.zeros_like(combined).masked_scatter(keep_mask, normalized)
    elif keep_mask is not None:
        combined = combined.masked_fill(~keep_mask, 0.0)
    _log_mixed_reward_tensor_stats("combined_after_final_normalize", combined)
    _capture_mixed_reward_diagnostic(diagnostics, "combined_after_final_normalize", combined)

    if post_normalize_sample_weight is not None:
        if post_normalize_sample_weight.shape not in (mv_rewards.shape, (mv_rewards.shape[0], 1)):
            raise ValueError("Mixed-reward sample weights must have shape (P, 1) or (P, G)")
        combined = post_normalize_sample_weight * combined
    _log_mixed_reward_tensor_stats("final_advantages", combined)
    _capture_mixed_reward_diagnostic(diagnostics, "final_advantages", combined)

    if not bool(torch.isfinite(combined).all().item()):
        raise ValueError("mixed_reward produced non-finite final advantages")

    return combined
