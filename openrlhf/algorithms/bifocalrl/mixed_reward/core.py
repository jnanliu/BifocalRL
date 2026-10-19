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
    _compute_legacy_corr_sample_weight,
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
    """Normalize each reward view, fuse it, normalize the batch, then apply weights.

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
    if mode != "reliability_sc_corr":
        legacy_sample_weight = _compute_legacy_corr_sample_weight(
            adv_mv,
            adv_int,
            corr_weight_mode=config.corr_weight_mode,
            corr_weight_threshold=config.corr_weight_threshold,
            corr_weight_exponent=config.corr_weight_exponent,
            num_generations_per_model=num_generations_per_model,
        )
        if legacy_sample_weight is not None:
            if post_normalize_sample_weight is not None:
                raise ValueError("Cannot stack strategy sample weights with legacy correlation weights")
            post_normalize_sample_weight = legacy_sample_weight

    _log_mixed_reward_tensor_stats("combined_before_final_normalize", combined)
    _capture_mixed_reward_diagnostic(diagnostics, "combined_before_final_normalize", combined)

    if final_normalize:
        combined = batch_zscore_normalize(combined, eps=eps)
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


def compute_mixed_reward_advantages(
    mv_rewards: torch.Tensor,
    intuitor_rewards: torch.Tensor,
    alpha: float = 0.5,
    alpha_mode: str = "fixed",
    alpha_zscore_mu: float = 0.65,
    alpha_zscore_sigma: float = 0.2,
    alpha_zscore_lower: float = 0.0,
    alpha_zscore_upper: float = 1.0,
    corr_weight_mode: str = "none",
    corr_weight_threshold: float = 0.1,
    corr_weight_exponent: float = 0.8,
    reliability_sc_weight: float = 0.6,
    reliability_corr_weight: float = 0.4,
    reliability_use_sample_weight: bool = True,
    reliability_use_advantage_weight: bool = True,
    reliability_sample_weight_mu: float = 1.0,
    reliability_sample_weight_sigma: float = 0.2,
    reliability_sample_weight_lower: float = 0.5,
    reliability_sample_weight_upper: float = 1.5,
    reliability_advantage_weight_mu: float = 0.5,
    reliability_advantage_weight_sigma: float = 0.2,
    reliability_advantage_weight_lower: float = 0.2,
    reliability_advantage_weight_upper: float = 0.8,
    final_normalize: bool = True,
    eps: float = 1e-6,
    num_generations_per_model: int | None = None,
    diagnostics: MutableMapping[str, Any] | None = None,
) -> torch.Tensor:
    """Backward-compatible keyword API; prefer compute_mixed_reward with a config."""
    config = MixedRewardConfig(
        alpha=alpha,
        alpha_mode=alpha_mode,
        alpha_zscore_mu=alpha_zscore_mu,
        alpha_zscore_sigma=alpha_zscore_sigma,
        alpha_zscore_lower=alpha_zscore_lower,
        alpha_zscore_upper=alpha_zscore_upper,
        corr_weight_mode=corr_weight_mode,
        corr_weight_threshold=corr_weight_threshold,
        corr_weight_exponent=corr_weight_exponent,
        reliability_sc_weight=reliability_sc_weight,
        reliability_corr_weight=reliability_corr_weight,
        reliability_use_sample_weight=reliability_use_sample_weight,
        reliability_use_advantage_weight=reliability_use_advantage_weight,
        reliability_sample_weight_mu=reliability_sample_weight_mu,
        reliability_sample_weight_sigma=reliability_sample_weight_sigma,
        reliability_sample_weight_lower=reliability_sample_weight_lower,
        reliability_sample_weight_upper=reliability_sample_weight_upper,
        reliability_advantage_weight_mu=reliability_advantage_weight_mu,
        reliability_advantage_weight_sigma=reliability_advantage_weight_sigma,
        reliability_advantage_weight_lower=reliability_advantage_weight_lower,
        reliability_advantage_weight_upper=reliability_advantage_weight_upper,
        final_normalize=final_normalize,
        eps=eps,
    )
    return compute_mixed_reward(
        mv_rewards,
        intuitor_rewards,
        config=config,
        num_generations_per_model=num_generations_per_model,
        diagnostics=diagnostics,
    )
