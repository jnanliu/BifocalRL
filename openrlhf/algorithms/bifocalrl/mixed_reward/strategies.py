"""Built-in fusion formulas. Each handler leaves final normalization to core.py."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

import torch

from ..normalization import batch_zscore_normalize, rowwise_pearson_corrcoef, z_score_normalize_tensor_with_bounds
from .registry import register_mixed_reward_mode
from .types import MixedRewardContext, MixedRewardResult

logger = logging.getLogger(__name__)


def _log_mixed_reward_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Log compact population statistics for a mixed-reward tensor."""
    if tensor.numel() == 0:
        logger.info("mixed_reward %s stats: empty", name)
        return
    logger.info(
        "mixed_reward %s stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
        name,
        float(tensor.mean().item()),
        float(tensor.std(unbiased=False).item()),
        float(tensor.min().item()),
        float(tensor.max().item()),
    )


def _capture_mixed_reward_diagnostic(
    diagnostics: MutableMapping[str, Any] | None,
    name: str,
    value: Any,
) -> None:
    """Store an immutable CPU copy of a mixed-reward diagnostic value."""
    if diagnostics is None:
        return
    if isinstance(value, torch.Tensor):
        diagnostics[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()
    else:
        diagnostics[name] = value


def _compute_sc_corr_reliability(
    mv_rewards: torch.Tensor,
    adv_mv: torch.Tensor,
    adv_int: torch.Tensor,
    *,
    sc_weight: float,
    corr_weight: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Build prompt-level reliability from batch-standardized SC and correlation."""
    sc_coef = float(sc_weight)
    corr_coef = float(corr_weight)
    if sc_coef < 0.0 or corr_coef < 0.0:
        raise ValueError(
            "mixed_reward reliability weights must be non-negative, "
            f"got sc_weight={sc_coef}, corr_weight={corr_coef}"
        )
    if sc_coef == 0.0 and corr_coef == 0.0:
        raise ValueError("mixed_reward reliability weights cannot both be zero")

    raw_sc = mv_rewards.mean(dim=1, keepdim=True)
    raw_corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    z_sc = batch_zscore_normalize(raw_sc, eps=eps)
    z_corr = batch_zscore_normalize(raw_corr, eps=eps)
    # Deliberately do not divide by sc_coef + corr_coef.
    reliability = sc_coef * z_sc + corr_coef * z_corr

    result = {
        "raw_sc": raw_sc,
        "raw_corr": raw_corr,
        "z_sc": z_sc,
        "z_corr": z_corr,
        "reliability": reliability,
    }
    for name, tensor in result.items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"mixed_reward reliability produced non-finite {name}")
    return result


def _validate_reliability_mapping(
    name: str,
    *,
    mu: float,
    sigma: float,
    lower: float,
    upper: float,
) -> None:
    if float(sigma) <= 0.0:
        raise ValueError(f"mixed_reward reliability {name} sigma must be positive")
    if float(lower) >= float(upper):
        raise ValueError(f"mixed_reward reliability {name} lower must be < upper")
    if not (float(lower) <= float(mu) <= float(upper)):
        raise ValueError(
            f"mixed_reward reliability {name} mu must be within [lower, upper], "
            f"got mu={mu}, lower={lower}, upper={upper}"
        )


@register_mixed_reward_mode("reliability_sc_corr")
def _mix_reliability_sc_corr(context: MixedRewardContext) -> MixedRewardResult:
    config = context.config
    diagnostics = context.diagnostics
    adv_mv, adv_int = context.adv_mv, context.adv_int
    if context.num_generations_per_model is not None:
        raise ValueError(
            "reliability_sc_corr currently supports single-model mixed_reward only; "
            "co-training / multi-model layouts are not implemented."
        )
    if abs(float(config.alpha) - 0.5) > 1e-12:
        raise ValueError("reliability_sc_corr requires mixed_reward_alpha=0.5, " f"got {config.alpha}")

    use_sample_weight = bool(config.reliability_use_sample_weight)
    use_advantage_weight = bool(config.reliability_use_advantage_weight)
    baseline = not use_sample_weight and not use_advantage_weight
    combined = 0.5 * adv_mv + 0.5 * adv_int
    post_normalize_sample_weight = None

    # Disabled weights are completely irrelevant to the baseline computation.
    # With diagnostics enabled we still record their candidate values.
    if not baseline or diagnostics is not None:
        mapping_options = {}
        for name in ("sample_weight", "advantage_weight"):
            options = {key: getattr(config, f"reliability_{name}_{key}") for key in ("mu", "sigma", "lower", "upper")}
            _validate_reliability_mapping(name, **options)
            mapping_options[name] = {key: float(value) for key, value in options.items()}
        reliability_tensors = _compute_sc_corr_reliability(
            context.mv_rewards,
            adv_mv,
            adv_int,
            sc_weight=config.reliability_sc_weight,
            corr_weight=config.reliability_corr_weight,
            eps=config.eps,
        )
        reliability = reliability_tensors["reliability"]
        sample_weight = z_score_normalize_tensor_with_bounds(
            reliability,
            **mapping_options["sample_weight"],
            dim=0,
        )
        advantage_weight = z_score_normalize_tensor_with_bounds(
            reliability,
            **mapping_options["advantage_weight"],
            dim=0,
        )
        applied_advantage_weight = advantage_weight if use_advantage_weight else torch.full_like(advantage_weight, 0.5)
        if not baseline:
            combined = applied_advantage_weight * adv_mv + (1.0 - applied_advantage_weight) * adv_int
        if use_sample_weight:
            post_normalize_sample_weight = sample_weight

        for name, tensor in reliability_tensors.items():
            _log_mixed_reward_tensor_stats(f"reliability_{name}", tensor)
            _capture_mixed_reward_diagnostic(diagnostics, name, tensor)
        _log_mixed_reward_tensor_stats("reliability_sample_weight", sample_weight)
        _log_mixed_reward_tensor_stats("reliability_advantage_weight", advantage_weight)
        applied_weights = {
            "computed_sample_weight": sample_weight,
            "applied_sample_weight": sample_weight if use_sample_weight else torch.ones_like(sample_weight),
            "computed_advantage_weight": advantage_weight,
            "applied_advantage_weight": applied_advantage_weight,
        }
        for name, tensor in applied_weights.items():
            _capture_mixed_reward_diagnostic(diagnostics, name, tensor)

    _capture_mixed_reward_diagnostic(diagnostics, "diagnostics_only_reliability", baseline)
    _capture_mixed_reward_diagnostic(diagnostics, "use_sample_weight", use_sample_weight)
    _capture_mixed_reward_diagnostic(diagnostics, "use_advantage_weight", use_advantage_weight)
    return MixedRewardResult(combined, post_normalize_sample_weight)


@register_mixed_reward_mode("fixed")
def _mix_fixed(context: MixedRewardContext) -> MixedRewardResult:
    adv_int = context.adv_int
    adv_mv = context.adv_mv
    alpha = context.config.alpha
    post_normalize_sample_weight = None
    a = float(alpha)
    combined = a * adv_mv + (1.0 - a) * adv_int
    return MixedRewardResult(combined, post_normalize_sample_weight)


@register_mixed_reward_mode("bifocalrl")
def _mix_bifocalrl(context: MixedRewardContext) -> MixedRewardResult:
    """Filter original views, then attenuate SC cluster means among retained rollouts."""
    config = context.config
    diagnostics = context.diagnostics
    adv_mv, adv_sc = context.adv_mv, context.adv_int
    mv_weight = torch.full_like(adv_mv[:, :1], 0.5)
    mv_contribution = mv_weight * adv_mv
    sc_contribution = (1.0 - mv_weight) * adv_sc
    combined = mv_contribution + sc_contribution
    snr = combined.abs() / (mv_contribution.abs() + sc_contribution.abs() + config.eps)
    threshold = torch.full_like(mv_weight, config.gd2po_snr_threshold)
    if config.gd2po_filter_mode == "snr":
        keep = snr > threshold
    elif config.gd2po_filter_mode == "hard":
        # Zero contributions are neutral; only strictly opposing signs conflict.
        conflict = ((mv_contribution > 0) & (sc_contribution < 0)) | (
            (mv_contribution < 0) & (sc_contribution > 0)
        )
        keep = ~conflict
    else:
        keep = torch.ones_like(combined, dtype=torch.bool)
    cluster_ids = (context.metadata or {}).get("answer_cluster_ids")
    if cluster_ids is None and config.sc_between_gamma != 1:
        raise ValueError("BifocalRL SC decomposition requires answer_cluster_ids")
    cluster_mean = torch.zeros_like(adv_sc)
    cluster_size = torch.zeros_like(adv_sc)
    if cluster_ids is not None:
        if (
            cluster_ids.shape != adv_sc.shape
            or cluster_ids.dtype != torch.long
            or (cluster_ids < 0).any()
            or (cluster_ids >= adv_sc.shape[1]).any()
        ):
            raise ValueError("answer_cluster_ids must be a (P, G) long tensor with IDs in [0, G)")
        cluster_ids = cluster_ids.to(adv_sc.device)
        counts = torch.zeros_like(adv_sc).scatter_add_(1, cluster_ids, keep.to(adv_sc.dtype))
        sums = torch.zeros_like(adv_sc).scatter_add_(1, cluster_ids, adv_sc.masked_fill(~keep, 0))
        cluster_mean = (sums / counts.clamp_min(1)).gather(1, cluster_ids).masked_fill(~keep, 0)
        cluster_size = counts.gather(1, cluster_ids).masked_fill(~keep, 0)
    sc_within = (adv_sc - cluster_mean).masked_fill(~keep, 0)
    adjusted_sc = adv_sc - (1.0 - config.sc_between_gamma) * cluster_mean
    combined = mv_contribution + (1.0 - mv_weight) * adjusted_sc
    retained_fraction = keep.to(combined.dtype).mean(dim=1, keepdim=True)
    query_weight = retained_fraction if config.gd2po_query_reweight else torch.ones_like(retained_fraction)
    combined = torch.where(keep, query_weight * combined, 0.0)
    for name, value in {
        "mv_weight": mv_weight,
        "snr": snr,
        "snr_threshold": threshold,
        "rollout_keep": keep,
        "retained_fraction": retained_fraction,
        "query_weight": query_weight,
        "sc_between_gamma": torch.full_like(mv_weight, config.sc_between_gamma),
        "sc_cluster_mean": cluster_mean,
        "sc_cluster_size": cluster_size,
        "sc_within": sc_within,
        "sc_adjusted": adjusted_sc.masked_fill(~keep, 0),
    }.items():
        _capture_mixed_reward_diagnostic(diagnostics, name, value)
        _log_mixed_reward_tensor_stats(f"bifocalrl_{name}", value.float())
    return MixedRewardResult(combined, keep_mask=keep)
