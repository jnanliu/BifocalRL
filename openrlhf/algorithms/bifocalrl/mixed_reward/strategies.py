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


def _compute_legacy_corr_sample_weight(
    adv_mv: torch.Tensor,
    adv_int: torch.Tensor,
    *,
    corr_weight_mode: str,
    corr_weight_threshold: float,
    corr_weight_exponent: float,
    num_generations_per_model: int | None,
) -> torch.Tensor | None:
    """Compute the legacy post-normalize correlation weight without applying it."""
    corr_mode = str(corr_weight_mode).strip().lower()
    if corr_mode == "none":
        return None
    if num_generations_per_model is not None:
        raise ValueError(
            "mixed_reward correlation weighting currently supports single-model mixed_reward only; "
            "co-training / multi-model layouts with num_generations_per_model are not implemented."
        )

    threshold = float(corr_weight_threshold)
    exponent = float(corr_weight_exponent)
    if not (0.0 <= threshold < 1.0):
        raise ValueError(f"mixed_reward correlation weighting requires 0 <= threshold < 1, got {threshold}")
    if exponent <= 0.0:
        raise ValueError(f"mixed_reward correlation weighting requires exponent > 0, got {exponent}")
    if corr_mode != "pearson_threshold_power":
        raise ValueError(f"Unknown mixed_reward correlation weighting mode: {corr_weight_mode!r}")

    prompt_adv_corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    prompt_adv_corr = torch.clamp(prompt_adv_corr, -1.0, 1.0)
    prompt_corr_weight = torch.zeros_like(prompt_adv_corr)
    active = prompt_adv_corr >= threshold
    if active.any():
        prompt_corr_weight = torch.where(
            active,
            ((prompt_adv_corr - threshold) / (1.0 - threshold)).pow(exponent),
            torch.zeros_like(prompt_adv_corr),
        )

    _log_mixed_reward_tensor_stats("prompt_adv_corr", prompt_adv_corr)
    _log_mixed_reward_tensor_stats("prompt_corr_weight", prompt_corr_weight)
    return prompt_corr_weight


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


def _validate_single_model_no_final_normalize_mode(
    mode: str,
    *,
    alpha: float,
    corr_weight_mode: str,
    final_normalize: bool,
    num_generations_per_model: int | None,
) -> None:
    """Validate the shared contract for PRISM and eTTRL."""
    if num_generations_per_model is not None:
        raise ValueError(
            f"{mode} currently supports single-model mixed_reward only; "
            "co-training / multi-model layouts are not implemented."
        )
    if abs(float(alpha) - 0.5) > 1e-12:
        raise ValueError(f"{mode} requires mixed_reward_alpha=0.5, got {alpha}")
    if str(corr_weight_mode).strip().lower() != "none":
        raise ValueError(
            f"{mode} requires corr_weight_mode='none'; legacy correlation "
            "weighting cannot be stacked with this mode."
        )
    if bool(final_normalize):
        raise ValueError(f"{mode} requires mixed_reward_final_normalize=false")


def _compute_ettrl_response_weights(
    intuitor_rewards: torch.Tensor,
    *,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Compute eTTRL response weights from raw within-prompt self-certainty."""
    if bool((intuitor_rewards < 0).any().item()):
        raise ValueError("ettrl requires non-negative self-certainty rewards")

    group_mean = intuitor_rewards.mean(dim=1, keepdim=True)
    all_zero_group = (intuitor_rewards == 0).all(dim=1, keepdim=True)
    near_zero_nonempty = (group_mean <= float(eps)) & ~all_zero_group
    if bool(near_zero_nonempty.any().item()):
        raise ValueError(
            "ettrl encountered a non-empty self-certainty group with mean <= eps; "
            "refusing to create unstable response weights"
        )

    safe_mean = torch.where(all_zero_group, torch.ones_like(group_mean), group_mean)
    response_weight = intuitor_rewards / safe_mean
    response_weight = torch.where(
        all_zero_group.expand_as(response_weight),
        torch.ones_like(response_weight),
        response_weight,
    )
    if not bool(torch.isfinite(response_weight).all().item()):
        raise ValueError("ettrl produced non-finite response weights")

    return {
        "self_certainty_group_mean": group_mean,
        "ettrl_all_zero_group": all_zero_group.to(dtype=intuitor_rewards.dtype),
        "mv_advantage_weight": response_weight,
        "self_certainty_advantage_weight": torch.zeros_like(response_weight),
    }


@register_mixed_reward_mode("prism")
@register_mixed_reward_mode("ettrl")
def _mix_prism_ettrl(context: MixedRewardContext) -> MixedRewardResult:
    adv_int = context.adv_int
    adv_mv = context.adv_mv
    diagnostics = context.diagnostics
    intuitor_rewards = context.intuitor_rewards
    num_generations_per_model = context.num_generations_per_model
    alpha = context.config.alpha
    corr_weight_mode = context.config.corr_weight_mode
    eps = context.config.eps
    final_normalize = context.config.final_normalize
    mode = context.mode
    post_normalize_sample_weight = None
    _validate_single_model_no_final_normalize_mode(
        mode,
        alpha=alpha,
        corr_weight_mode=corr_weight_mode,
        final_normalize=final_normalize,
        num_generations_per_model=num_generations_per_model,
    )
    if mode == "prism":
        mv_advantage_weight = torch.ones_like(adv_mv)
        self_certainty_advantage_weight = torch.ones_like(adv_int)
        self_certainty_group_mean = intuitor_rewards.mean(dim=1, keepdim=True)
    else:
        ettrl_tensors = _compute_ettrl_response_weights(
            intuitor_rewards,
            eps=eps,
        )
        mv_advantage_weight = ettrl_tensors["mv_advantage_weight"]
        self_certainty_advantage_weight = ettrl_tensors["self_certainty_advantage_weight"]
        self_certainty_group_mean = ettrl_tensors["self_certainty_group_mean"]
        _capture_mixed_reward_diagnostic(
            diagnostics,
            "ettrl_all_zero_group",
            ettrl_tensors["ettrl_all_zero_group"],
        )

    combined = mv_advantage_weight * adv_mv + self_certainty_advantage_weight * adv_int
    _capture_mixed_reward_diagnostic(diagnostics, "self_certainty_group_mean", self_certainty_group_mean)
    _capture_mixed_reward_diagnostic(diagnostics, "mv_advantage_weight", mv_advantage_weight)
    _capture_mixed_reward_diagnostic(
        diagnostics,
        "self_certainty_advantage_weight",
        self_certainty_advantage_weight,
    )
    _log_mixed_reward_tensor_stats(f"{mode}_mv_advantage_weight", mv_advantage_weight)
    _log_mixed_reward_tensor_stats(
        f"{mode}_self_certainty_advantage_weight",
        self_certainty_advantage_weight,
    )
    return MixedRewardResult(combined, post_normalize_sample_weight)


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
    if str(config.corr_weight_mode).strip().lower() != "none":
        raise ValueError(
            "reliability_sc_corr requires corr_weight_mode='none'; "
            "legacy correlation weighting cannot be stacked with reliability sample weighting."
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


@register_mixed_reward_mode("adaptive_sc")
def _mix_adaptive_sc(context: MixedRewardContext) -> MixedRewardResult:
    adv_int = context.adv_int
    adv_mv = context.adv_mv
    mv_rewards = context.mv_rewards
    num_generations_per_model = context.num_generations_per_model
    post_normalize_sample_weight = None
    P, G = mv_rewards.shape
    if num_generations_per_model is None:
        alpha_per = mv_rewards.mean(dim=1, keepdim=True)  # (P, 1) -> broadcast
    else:
        N = int(num_generations_per_model)
        if G != 2 * N:
            raise ValueError(
                f"adaptive_sc mixed_reward with num_generations_per_model={N} "
                f"expects mv_rewards.shape[1]=={2 * N}, got G={G}"
            )
        sc_a = mv_rewards[:, :N].mean(dim=1, keepdim=True)  # (P, 1)
        sc_b = mv_rewards[:, N:].mean(dim=1, keepdim=True)
        alpha_per = torch.cat(
            [sc_a.expand(-1, N), sc_b.expand(-1, N)],
            dim=1,
        )  # (P, 2N)
    combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
    logger.info(
        "mixed_reward adaptive_sc alpha (SC score, legacy GRPO path): " "mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(alpha_per.mean().item()),
        float(alpha_per.std().item()),
        float(alpha_per.min().item()),
        float(alpha_per.max().item()),
    )
    return MixedRewardResult(combined, post_normalize_sample_weight)


@register_mixed_reward_mode("adaptive_sc_zscore")
def _mix_adaptive_sc_zscore(context: MixedRewardContext) -> MixedRewardResult:
    adv_int = context.adv_int
    adv_mv = context.adv_mv
    mv_rewards = context.mv_rewards
    num_generations_per_model = context.num_generations_per_model
    alpha_zscore_lower = context.config.alpha_zscore_lower
    alpha_zscore_mu = context.config.alpha_zscore_mu
    alpha_zscore_sigma = context.config.alpha_zscore_sigma
    alpha_zscore_upper = context.config.alpha_zscore_upper
    post_normalize_sample_weight = None
    if num_generations_per_model is not None:
        raise ValueError(
            "adaptive_sc_zscore currently supports single-model mixed_reward only; "
            "co-training / multi-model layouts with num_generations_per_model are not implemented."
        )
    raw_sc = mv_rewards.mean(dim=1, keepdim=True)  # (P, 1)
    alpha_per = z_score_normalize_tensor_with_bounds(
        raw_sc,
        mu=float(alpha_zscore_mu),
        sigma=float(alpha_zscore_sigma),
        lower=float(alpha_zscore_lower),
        upper=float(alpha_zscore_upper),
        dim=0,
    )
    combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
    logger.info(
        "mixed_reward adaptive_sc_zscore raw_sc stats: " "mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(raw_sc.mean().item()),
        float(raw_sc.std(unbiased=False).item()),
        float(raw_sc.min().item()),
        float(raw_sc.max().item()),
    )
    logger.info(
        "mixed_reward adaptive_sc_zscore alpha stats: " "mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(alpha_per.mean().item()),
        float(alpha_per.std(unbiased=False).item()),
        float(alpha_per.min().item()),
        float(alpha_per.max().item()),
    )
    return MixedRewardResult(combined, post_normalize_sample_weight)


@register_mixed_reward_mode("adaptive")
def _mix_adaptive(context: MixedRewardContext) -> MixedRewardResult:
    adv_int = context.adv_int
    adv_mv = context.adv_mv
    eps = context.config.eps
    post_normalize_sample_weight = None
    abs_mv = adv_mv.abs()
    abs_int = adv_int.abs()
    denom = abs_mv + abs_int + eps
    alpha_per = abs_mv / denom
    combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
    logger.info(
        "mixed_reward adaptive alpha (magnitude, legacy GRPO path): " "mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(alpha_per.mean().item()),
        float(alpha_per.std().item()),
        float(alpha_per.min().item()),
        float(alpha_per.max().item()),
    )
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
