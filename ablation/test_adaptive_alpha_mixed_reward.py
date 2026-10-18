#!/usr/bin/env python
# coding=utf-8
"""CPU-only checks for mixed_reward alpha modes (no GPU, no models).

Run from repo `co-learning/` root::

    python ablation/test_adaptive_alpha_mixed_reward.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.training.co_intuitor_train import (  # noqa: E402
    _compute_sc_corr_reliability,
    compute_mixed_reward_advantages,
)
from utils.normalization import (  # noqa: E402
    grpo_group_advantages,
    batch_zscore_normalize,
    rowwise_pearson_corrcoef,
    z_score_normalize_tensor_with_bounds,
)


def _expected_adaptive_blend_magnitude(
    mv_rewards: torch.Tensor,
    intuitor_rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Legacy ``adaptive``: per-element blend from advantage magnitudes."""
    adv_mv = grpo_group_advantages(mv_rewards, eps=eps)
    adv_int = grpo_group_advantages(intuitor_rewards, eps=eps)
    abs_mv = adv_mv.abs()
    abs_int = adv_int.abs()
    denom = abs_mv + abs_int + eps
    alpha_per = abs_mv / denom
    return alpha_per * adv_mv + (1.0 - alpha_per) * adv_int


def _expected_adaptive_blend_sc(
    mv_rewards: torch.Tensor,
    intuitor_rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
    num_generations_per_model: Optional[int] = None,
) -> torch.Tensor:
    """``adaptive_sc``: SC score alpha from ``mv_rewards``."""
    adv_mv = grpo_group_advantages(mv_rewards, eps=eps)
    adv_int = grpo_group_advantages(intuitor_rewards, eps=eps)
    if num_generations_per_model is None:
        alpha_per = mv_rewards.mean(dim=1, keepdim=True)
    else:
        P, G = mv_rewards.shape
        N = int(num_generations_per_model)
        assert G == 2 * N
        sc_a = mv_rewards[:, :N].mean(dim=1, keepdim=True)
        sc_b = mv_rewards[:, N:].mean(dim=1, keepdim=True)
        alpha_per = torch.cat(
            [sc_a.expand(-1, N), sc_b.expand(-1, N)], dim=1,
        )
    return alpha_per * adv_mv + (1.0 - alpha_per) * adv_int


def _expected_adaptive_blend_sc_zscore(
    mv_rewards: torch.Tensor,
    intuitor_rewards: torch.Tensor,
    *,
    mu: float = 0.65,
    sigma: float = 0.2,
    lower: float = 0.0,
    upper: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``adaptive_sc_zscore``: z-scored per-prompt SC alpha from ``mv_rewards``."""
    adv_mv = grpo_group_advantages(mv_rewards, eps=eps)
    adv_int = grpo_group_advantages(intuitor_rewards, eps=eps)
    raw_sc = mv_rewards.mean(dim=1, keepdim=True)
    alpha_per = z_score_normalize_tensor_with_bounds(
        raw_sc,
        mu=mu,
        sigma=sigma,
        lower=lower,
        upper=upper,
        dim=0,
    )
    return alpha_per * adv_mv + (1.0 - alpha_per) * adv_int


def _expected_corr_weighted_combined(
    mv_rewards: torch.Tensor,
    intuitor_rewards: torch.Tensor,
    *,
    alpha: float = 0.5,
    alpha_mode: str = "fixed",
    final_normalize: bool = False,
    threshold: float = 0.1,
    exponent: float = 0.8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Expected post-final-normalize prompt correlation weighting."""
    if alpha_mode != "fixed":
        raise ValueError("Test helper currently supports fixed mode only")
    adv_mv = grpo_group_advantages(mv_rewards, eps=eps)
    adv_int = grpo_group_advantages(intuitor_rewards, eps=eps)
    combined = float(alpha) * adv_mv + (1.0 - float(alpha)) * adv_int
    if final_normalize:
        combined = batch_zscore_normalize(combined, eps=eps)
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
    return prompt_corr_weight * combined


def _reliability_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    mv = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    intr = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.1],
            [0.1, 0.9, 0.2, 0.8],
            [0.1, 0.8, 0.7, 0.6],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    return mv, intr


def _expected_reliability_weights(
    mv: torch.Tensor,
    intr: torch.Tensor,
    *,
    sc_weight: float = 0.6,
    corr_weight: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    raw_sc = mv.mean(dim=1, keepdim=True)
    raw_corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    reliability = (
        float(sc_weight) * batch_zscore_normalize(raw_sc)
        + float(corr_weight) * batch_zscore_normalize(raw_corr)
    )
    sample_weight = z_score_normalize_tensor_with_bounds(
        reliability,
        mu=1.0,
        sigma=0.2,
        lower=0.5,
        upper=1.5,
        dim=0,
    )
    advantage_weight = z_score_normalize_tensor_with_bounds(
        reliability,
        mu=0.5,
        sigma=0.2,
        lower=0.2,
        upper=0.8,
        dim=0,
    )
    return reliability, sample_weight, advantage_weight


def _expected_two_phase_sc_blend(
    A_mv: torch.Tensor,
    A_int: torch.Tensor,
    mv_full: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """``adaptive_sc`` branch in ``_compute_mixed_advantages_two_phase``."""
    sc_a = mv_full[:, :N].mean(dim=1, keepdim=True)
    sc_b = mv_full[:, N:].mean(dim=1, keepdim=True)
    alpha_per = torch.cat(
        [sc_a.expand(-1, N), sc_b.expand(-1, N)], dim=1,
    )
    return alpha_per * A_mv + (1.0 - alpha_per) * A_int


def _expected_two_phase_magnitude_blend(
    A_mv: torch.Tensor,
    A_int: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``adaptive`` branch in ``_compute_mixed_advantages_two_phase``."""
    abs_mv = A_mv.abs()
    abs_int = A_int.abs()
    denom = abs_mv + abs_int + eps
    alpha_per = abs_mv / denom
    return alpha_per * A_mv + (1.0 - alpha_per) * A_int


def test_adaptive_matches_manual() -> None:
    torch.manual_seed(0)
    mv = torch.tensor([[1.0, 0.0, 0.5], [0.2, 0.8, 0.3]], dtype=torch.float32)
    intr = torch.tensor([[0.5, 1.0, 0.4], [0.9, 0.1, 0.6]], dtype=torch.float32)
    got = compute_mixed_reward_advantages(
        mv, intr, alpha=0.25, alpha_mode="adaptive", final_normalize=False, eps=1e-6
    )
    want = _expected_adaptive_blend_magnitude(mv, intr, eps=1e-6)
    assert got.shape == want.shape == mv.shape
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_adaptive_sc_matches_manual() -> None:
    torch.manual_seed(0)
    mv = torch.tensor([[1.0, 0.0, 0.5], [0.2, 0.8, 0.3]], dtype=torch.float32)
    intr = torch.tensor([[0.5, 1.0, 0.4], [0.9, 0.1, 0.6]], dtype=torch.float32)
    got = compute_mixed_reward_advantages(
        mv, intr, alpha=0.25, alpha_mode="adaptive_sc", final_normalize=False, eps=1e-6
    )
    want = _expected_adaptive_blend_sc(mv, intr, eps=1e-6)
    assert got.shape == want.shape == mv.shape
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_adaptive_sc_zscore_matches_manual() -> None:
    mv = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    intr = torch.tensor(
        [[0.2, 0.7, 0.4, 0.9], [0.8, 0.1, 0.5, 0.3], [0.4, 0.6, 0.2, 0.8]],
        dtype=torch.float32,
    )
    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.25,
        alpha_mode="adaptive_sc_zscore",
        alpha_zscore_mu=0.65,
        alpha_zscore_sigma=0.2,
        alpha_zscore_lower=0.0,
        alpha_zscore_upper=1.0,
        final_normalize=False,
        eps=1e-6,
    )
    want = _expected_adaptive_blend_sc_zscore(
        mv,
        intr,
        mu=0.65,
        sigma=0.2,
        lower=0.0,
        upper=1.0,
        eps=1e-6,
    )
    assert got.shape == want.shape == mv.shape
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_adaptive_sc_zscore_ignores_alpha() -> None:
    mv = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    intr = torch.tensor(
        [[0.2, 0.7, 0.4, 0.9], [0.8, 0.1, 0.5, 0.3], [0.4, 0.6, 0.2, 0.8]],
        dtype=torch.float32,
    )
    got_a = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.1,
        alpha_mode="adaptive_sc_zscore",
        final_normalize=False,
    )
    got_b = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.9,
        alpha_mode="adaptive_sc_zscore",
        final_normalize=False,
    )
    assert torch.allclose(got_a, got_b, atol=1e-6, rtol=1e-6)


def test_adaptive_sc_zscore_respects_explicit_bounds() -> None:
    mv = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    raw_sc = mv.mean(dim=1, keepdim=True)
    alpha_per = z_score_normalize_tensor_with_bounds(
        raw_sc,
        mu=0.6,
        sigma=0.8,
        lower=0.2,
        upper=0.7,
        dim=0,
    )
    assert torch.all(alpha_per >= 0.2)
    assert torch.all(alpha_per <= 0.7)
    assert torch.isclose(alpha_per.max(), torch.tensor(0.7), atol=1e-6)
    assert torch.isclose(alpha_per.min(), torch.tensor(0.2), atol=1e-6)


def test_adaptive_sc_zscore_degenerate_batch_uses_mu() -> None:
    mv = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    intr = torch.tensor([[0.3, 0.5, 0.7, 0.9]], dtype=torch.float32)
    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.99,
        alpha_mode="adaptive_sc_zscore",
        alpha_zscore_mu=0.8,
        alpha_zscore_sigma=0.2,
        alpha_zscore_lower=0.0,
        alpha_zscore_upper=1.0,
        final_normalize=False,
    )
    want = _expected_adaptive_blend_sc_zscore(
        mv,
        intr,
        mu=0.8,
        sigma=0.2,
        lower=0.0,
        upper=1.0,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_adaptive_sc_zscore_co_layout_rejected() -> None:
    mv = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    intr = torch.randn(2, 4, dtype=torch.float32)
    try:
        compute_mixed_reward_advantages(
            mv,
            intr,
            alpha_mode="adaptive_sc_zscore",
            num_generations_per_model=2,
            final_normalize=False,
        )
    except ValueError as exc:
        msg = str(exc)
        assert "single-model mixed_reward only" in msg
        assert "not implemented" in msg
    else:
        raise AssertionError("adaptive_sc_zscore should reject co-training layout")


def test_rowwise_pearson_corrcoef_basic_cases() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [
            [2.0, 4.0, 6.0, 8.0],
            [8.0, 6.0, 4.0, 2.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    got = rowwise_pearson_corrcoef(x, y, dim=1, keepdim=True)
    assert got.shape == (3, 1)
    assert torch.isclose(got[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(got[1, 0], torch.tensor(-1.0), atol=1e-5)
    assert abs(float(got[2, 0])) < 1e-5


def test_rowwise_pearson_corrcoef_degenerate_cases() -> None:
    single = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    got_single = rowwise_pearson_corrcoef(single, single, dim=1, keepdim=True)
    assert torch.allclose(got_single, torch.zeros_like(got_single))

    zero_var = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    got_zero_var = rowwise_pearson_corrcoef(zero_var, zero_var, dim=1, keepdim=True)
    assert torch.allclose(got_zero_var, torch.zeros_like(got_zero_var))

    try:
        rowwise_pearson_corrcoef(
            torch.ones(2, 3, dtype=torch.float32),
            torch.ones(2, 4, dtype=torch.float32),
            dim=1,
            keepdim=True,
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to raise ValueError")


def test_corr_weighted_mixed_reward_matches_manual() -> None:
    mv = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    intr = torch.tensor(
        [[0.1, 0.8, 0.2, 0.9], [0.9, 0.8, 0.3, 0.2], [0.4, 0.6, 0.3, 0.7]],
        dtype=torch.float32,
    )
    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="fixed",
        corr_weight_mode="pearson_threshold_power",
        corr_weight_threshold=0.1,
        corr_weight_exponent=0.8,
        final_normalize=False,
        eps=1e-6,
    )
    want = _expected_corr_weighted_combined(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="fixed",
        final_normalize=False,
        threshold=0.1,
        exponent=0.8,
        eps=1e-6,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_corr_weight_applies_after_final_normalize() -> None:
    mv = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    intr = torch.tensor(
        [[0.1, 0.8, 0.2, 0.9], [0.9, 0.8, 0.3, 0.2], [0.4, 0.6, 0.3, 0.7]],
        dtype=torch.float32,
    )
    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="fixed",
        corr_weight_mode="pearson_threshold_power",
        corr_weight_threshold=0.1,
        corr_weight_exponent=0.8,
        final_normalize=True,
        eps=1e-6,
    )
    want = _expected_corr_weighted_combined(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="fixed",
        final_normalize=True,
        threshold=0.1,
        exponent=0.8,
        eps=1e-6,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_adaptive_sc_zscore_corr_matches_pre_refactor_formula() -> None:
    mv, intr = _reliability_inputs()
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    alpha_per = z_score_normalize_tensor_with_bounds(
        mv.mean(dim=1, keepdim=True),
        mu=0.65,
        sigma=0.2,
        lower=0.1,
        upper=1.0,
        dim=0,
    )
    combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
    combined = batch_zscore_normalize(combined)

    corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    active = corr >= 0.1
    corr_weight = torch.where(
        active,
        ((corr - 0.1) / 0.9).pow(0.8),
        torch.zeros_like(corr),
    )
    want = corr_weight * combined

    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="adaptive_sc_zscore",
        alpha_zscore_mu=0.65,
        alpha_zscore_sigma=0.2,
        alpha_zscore_lower=0.1,
        alpha_zscore_upper=1.0,
        corr_weight_mode="pearson_threshold_power",
        corr_weight_threshold=0.1,
        corr_weight_exponent=0.8,
        final_normalize=True,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_corr_weight_threshold_gate() -> None:
    mv = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    intr = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    assert torch.isclose(corr[0, 0], torch.tensor(-1.0), atol=1e-5)
    assert torch.isclose(corr[1, 0], torch.tensor(1.0), atol=1e-5)

    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="fixed",
        corr_weight_mode="pearson_threshold_power",
        corr_weight_threshold=0.1,
        corr_weight_exponent=0.8,
        final_normalize=False,
        eps=1e-6,
    )
    assert torch.allclose(got[0], torch.zeros_like(got[0]), atol=1e-6, rtol=1e-6)

    base = compute_mixed_reward_advantages(
        mv[1:].clone(),
        intr[1:].clone(),
        alpha=0.5,
        alpha_mode="fixed",
        corr_weight_mode="none",
        final_normalize=False,
        eps=1e-6,
    )
    assert torch.allclose(got[1:], base, atol=1e-5, rtol=1e-5)


def test_corr_weight_mode_none_matches_legacy() -> None:
    mv = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    intr = torch.tensor([[0.1, 0.8, 0.2, 0.9], [0.9, 0.8, 0.3, 0.2]], dtype=torch.float32)
    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="fixed",
        corr_weight_mode="none",
        final_normalize=True,
        eps=1e-6,
    )
    want = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="fixed",
        final_normalize=True,
        eps=1e-6,
    )
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_corr_weight_single_only_rejected() -> None:
    mv = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    intr = torch.tensor([[0.1, 0.8, 0.2, 0.9], [0.9, 0.2, 0.8, 0.1]], dtype=torch.float32)
    try:
        compute_mixed_reward_advantages(
            mv,
            intr,
            alpha_mode="fixed",
            corr_weight_mode="pearson_threshold_power",
            num_generations_per_model=2,
            final_normalize=False,
        )
    except ValueError as exc:
        msg = str(exc)
        assert "single-model mixed_reward only" in msg
        assert "not implemented" in msg
    else:
        raise AssertionError("Expected correlation weighting to reject co-training layout")


def test_sc_corr_reliability_matches_manual_without_coefficient_normalization() -> None:
    mv, intr = _reliability_inputs()
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    signals = _compute_sc_corr_reliability(
        mv,
        adv_mv,
        adv_int,
        sc_weight=0.9,
        corr_weight=0.4,
        eps=1e-6,
    )

    raw_sc = mv.mean(dim=1, keepdim=True)
    raw_corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    z_sc = batch_zscore_normalize(raw_sc)
    z_corr = batch_zscore_normalize(raw_corr)
    # Coefficients sum to 1.3; this equality catches an unintended division by that sum.
    reliability = 0.9 * z_sc + 0.4 * z_corr

    assert torch.allclose(signals["raw_sc"], raw_sc, atol=1e-6, rtol=1e-6)
    assert torch.allclose(signals["raw_corr"], raw_corr, atol=1e-6, rtol=1e-6)
    assert torch.allclose(signals["z_sc"], z_sc, atol=1e-6, rtol=1e-6)
    assert torch.allclose(signals["z_corr"], z_corr, atol=1e-6, rtol=1e-6)
    assert torch.allclose(signals["reliability"], reliability, atol=1e-6, rtol=1e-6)


def test_reliability_full_matches_manual_and_weights_after_final_normalize() -> None:
    mv, intr = _reliability_inputs()
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    _reliability, sample_weight, advantage_weight = _expected_reliability_weights(mv, intr)
    mixed = advantage_weight * adv_mv + (1.0 - advantage_weight) * adv_int
    want = sample_weight * batch_zscore_normalize(mixed)

    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="reliability_sc_corr",
        corr_weight_mode="none",
        final_normalize=True,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)

    wrong_order = batch_zscore_normalize(sample_weight * mixed)
    assert not torch.allclose(got, wrong_order, atol=1e-5, rtol=1e-5)


def test_reliability_sample_only_matches_manual() -> None:
    mv, intr = _reliability_inputs()
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    _reliability, sample_weight, _advantage_weight = _expected_reliability_weights(mv, intr)
    fixed_mix = 0.5 * adv_mv + 0.5 * adv_int
    want = sample_weight * batch_zscore_normalize(fixed_mix)

    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="reliability_sc_corr",
        reliability_use_sample_weight=True,
        reliability_use_advantage_weight=False,
        final_normalize=True,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_reliability_advantage_only_matches_manual() -> None:
    mv, intr = _reliability_inputs()
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    _reliability, _sample_weight, advantage_weight = _expected_reliability_weights(mv, intr)
    want = batch_zscore_normalize(
        advantage_weight * adv_mv + (1.0 - advantage_weight) * adv_int
    )

    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="reliability_sc_corr",
        reliability_use_sample_weight=False,
        reliability_use_advantage_weight=True,
        final_normalize=True,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got, want)


def test_reliability_six_weighted_variants_match_manual() -> None:
    mv, intr = _reliability_inputs()
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    switch_modes = (
        (True, True),
        (False, True),
        (True, False),
    )
    for sc_weight, corr_weight in ((0.6, 0.4), (0.4, 0.6)):
        _reliability, sample_weight, advantage_weight = _expected_reliability_weights(
            mv,
            intr,
            sc_weight=sc_weight,
            corr_weight=corr_weight,
        )
        for use_sample, use_advantage in switch_modes:
            alpha_per = advantage_weight if use_advantage else torch.full_like(advantage_weight, 0.5)
            want = batch_zscore_normalize(
                alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
            )
            if use_sample:
                want = sample_weight * want

            got = compute_mixed_reward_advantages(
                mv,
                intr,
                alpha=0.5,
                alpha_mode="reliability_sc_corr",
                reliability_sc_weight=sc_weight,
                reliability_corr_weight=corr_weight,
                reliability_use_sample_weight=use_sample,
                reliability_use_advantage_weight=use_advantage,
                final_normalize=True,
            )
            assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (
                sc_weight,
                corr_weight,
                use_sample,
                use_advantage,
                got,
                want,
            )


def test_reliability_baseline_exactly_matches_fixed_alpha050() -> None:
    mv, intr = _reliability_inputs()
    for final_normalize in (False, True):
        got = compute_mixed_reward_advantages(
            mv,
            intr,
            alpha=0.5,
            alpha_mode="reliability_sc_corr",
            reliability_use_sample_weight=False,
            reliability_use_advantage_weight=False,
            reliability_sc_weight=0.0,
            reliability_corr_weight=0.0,
            reliability_sample_weight_sigma=0.0,
            reliability_advantage_weight_sigma=0.0,
            final_normalize=final_normalize,
        )
        want = compute_mixed_reward_advantages(
            mv,
            intr,
            alpha=0.5,
            alpha_mode="fixed",
            corr_weight_mode="none",
            final_normalize=final_normalize,
        )
        assert torch.equal(got, want), (got, want)


def test_reliability_degenerate_batch_uses_center_weights() -> None:
    mv = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    intr = torch.tensor([[0.2, 0.8, 0.3, 0.7]], dtype=torch.float32)
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    signals = _compute_sc_corr_reliability(
        mv,
        adv_mv,
        adv_int,
        sc_weight=0.6,
        corr_weight=0.4,
        eps=1e-6,
    )
    assert torch.equal(signals["z_sc"], torch.zeros_like(signals["z_sc"]))
    assert torch.equal(signals["z_corr"], torch.zeros_like(signals["z_corr"]))
    assert torch.equal(signals["reliability"], torch.zeros_like(signals["reliability"]))

    sample_weight = z_score_normalize_tensor_with_bounds(
        signals["reliability"], mu=1.0, sigma=0.2, lower=0.5, upper=1.5, dim=0
    )
    advantage_weight = z_score_normalize_tensor_with_bounds(
        signals["reliability"], mu=0.5, sigma=0.2, lower=0.2, upper=0.8, dim=0
    )
    assert torch.equal(sample_weight, torch.ones_like(sample_weight))
    assert torch.equal(advantage_weight, torch.full_like(advantage_weight, 0.5))

    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha_mode="reliability_sc_corr",
        final_normalize=True,
    )
    assert bool(torch.isfinite(got).all().item())


def test_reliability_rejects_invalid_settings_and_co_layout() -> None:
    mv, intr = _reliability_inputs()
    cases = [
        ({"reliability_sc_weight": -0.1}, "non-negative"),
        (
            {"reliability_sc_weight": 0.0, "reliability_corr_weight": 0.0},
            "cannot both be zero",
        ),
        ({"alpha": 0.6}, "requires mixed_reward_alpha=0.5"),
        (
            {"corr_weight_mode": "pearson_threshold_power"},
            "requires corr_weight_mode='none'",
        ),
        (
            {"num_generations_per_model": 2},
            "supports single-model mixed_reward only",
        ),
        ({"reliability_sample_weight_sigma": 0.0}, "sigma must be positive"),
        (
            {
                "reliability_advantage_weight_mu": 0.9,
                "reliability_advantage_weight_lower": 0.2,
                "reliability_advantage_weight_upper": 0.8,
            },
            "mu must be within",
        ),
    ]
    for kwargs, expected_message in cases:
        try:
            compute_mixed_reward_advantages(
                mv,
                intr,
                alpha_mode="reliability_sc_corr",
                final_normalize=False,
                **kwargs,
            )
        except ValueError as exc:
            assert expected_message in str(exc), (kwargs, exc)
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")

    try:
        compute_mixed_reward_advantages(
            mv,
            intr[:, :3],
            alpha_mode="reliability_sc_corr",
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched reward shapes to raise ValueError")


def test_fixed_ignores_mode_alpha() -> None:
    mv = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    intr = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    a = 0.25
    got = compute_mixed_reward_advantages(
        mv, intr, alpha=a, alpha_mode="fixed", final_normalize=False
    )
    adv_mv = grpo_group_advantages(mv)
    adv_int = grpo_group_advantages(intr)
    want = a * adv_mv + (1.0 - a) * adv_int
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_adaptive_uniform_halves() -> None:
    """When both raw rewards are constant within each row, GRPO advantages are 0 → blend is 0."""
    mv = torch.ones(2, 4, dtype=torch.float32)
    intr = torch.ones(2, 4, dtype=torch.float32) * 0.5
    got = compute_mixed_reward_advantages(
        mv, intr, alpha=0.5, alpha_mode="adaptive", final_normalize=False
    )
    assert torch.allclose(got, torch.zeros_like(got))


def test_adaptive_sc_co_split_matches_manual() -> None:
    """Co-training layout: alpha block-wise from mv_rewards[:, :N] and [:, N:]."""
    P, N = 2, 3
    mv = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    intr = torch.randn(P, 2 * N, dtype=torch.float32)
    got = compute_mixed_reward_advantages(
        mv,
        intr,
        alpha=0.5,
        alpha_mode="adaptive_sc",
        final_normalize=False,
        num_generations_per_model=N,
    )
    want = _expected_adaptive_blend_sc(
        mv, intr, num_generations_per_model=N,
    )
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_two_phase_sc_blend_formula() -> None:
    torch.manual_seed(1)
    P, N = 3, 4
    A_mv = torch.randn(P, 2 * N, dtype=torch.float32)
    A_int = torch.randn(P, 2 * N, dtype=torch.float32)
    mv_full = torch.zeros(P, 2 * N, dtype=torch.float32)
    mv_full[0, :N] = 1.0
    mv_full[0, N:] = 0.0
    blended = _expected_two_phase_sc_blend(A_mv, A_int, mv_full, N)
    assert blended.shape == A_mv.shape
    alpha_row0_first = 1.0
    alpha_row0_second = 0.0
    i, j = 0, 0
    exp_ij = alpha_row0_first * A_mv[i, j] + (1.0 - alpha_row0_first) * A_int[i, j]
    assert abs(float(blended[i, j] - exp_ij)) < 1e-4
    j2 = N
    exp_i_j2 = alpha_row0_second * A_mv[i, j2] + (1.0 - alpha_row0_second) * A_int[i, j2]
    assert abs(float(blended[i, j2] - exp_i_j2)) < 1e-4


def test_two_phase_magnitude_blend_formula() -> None:
    torch.manual_seed(2)
    A_mv = torch.randn(3, 8, dtype=torch.float32)
    A_int = torch.randn(3, 8, dtype=torch.float32)
    blended = _expected_two_phase_magnitude_blend(A_mv, A_int)
    assert blended.shape == A_mv.shape
    i, j = 0, 0
    A_mv[i, j] = 10.0
    A_int[i, j] = 0.01
    b = _expected_two_phase_magnitude_blend(A_mv, A_int)
    alpha_ij = A_mv[i, j].abs() / (A_mv[i, j].abs() + A_int[i, j].abs() + 1e-6)
    assert alpha_ij > 0.99
    exp_ij = alpha_ij * A_mv[i, j] + (1.0 - alpha_ij) * A_int[i, j]
    assert abs(float(b[i, j] - exp_ij)) < 1e-4


def test_zscore_with_bounds_basic() -> None:
    scores = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)
    got = z_score_normalize_tensor_with_bounds(
        scores,
        mu=0.6,
        sigma=0.2,
        lower=0.0,
        upper=1.0,
        dim=0,
    )
    mean = scores.mean(dim=0, keepdim=True)
    std = scores.std(dim=0, keepdim=True, unbiased=False)
    want = torch.clamp(((scores - mean) / std) * 0.2 + 0.6, 0.0, 1.0)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_zscore_with_bounds_singleton_uses_mu() -> None:
    scores = torch.tensor([[0.3]], dtype=torch.float32)
    got = z_score_normalize_tensor_with_bounds(
        scores,
        mu=0.65,
        sigma=0.2,
        lower=0.0,
        upper=1.0,
        dim=0,
    )
    assert torch.allclose(got, torch.tensor([[0.65]], dtype=torch.float32))


def test_zscore_with_bounds_zero_std_uses_mu() -> None:
    scores = torch.tensor([[0.4], [0.4], [0.4]], dtype=torch.float32)
    got = z_score_normalize_tensor_with_bounds(
        scores,
        mu=0.7,
        sigma=0.2,
        lower=0.1,
        upper=0.9,
        dim=0,
    )
    want = torch.full_like(scores, 0.7)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_zscore_with_bounds_invalid_params() -> None:
    scores = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    try:
        z_score_normalize_tensor_with_bounds(
            scores,
            mu=0.6,
            sigma=0.0,
            lower=0.0,
            upper=1.0,
            dim=0,
        )
    except ValueError as exc:
        assert "sigma must be positive" in str(exc)
    else:
        raise AssertionError("Expected sigma<=0 to raise ValueError")

    try:
        z_score_normalize_tensor_with_bounds(
            scores,
            mu=0.6,
            sigma=0.2,
            lower=0.5,
            upper=0.5,
            dim=0,
        )
    except ValueError as exc:
        assert "lower must be < upper" in str(exc)
    else:
        raise AssertionError("Expected lower>=upper to raise ValueError")


def main() -> None:
    test_adaptive_matches_manual()
    test_adaptive_sc_matches_manual()
    test_adaptive_sc_zscore_matches_manual()
    test_adaptive_sc_zscore_ignores_alpha()
    test_adaptive_sc_zscore_respects_explicit_bounds()
    test_adaptive_sc_zscore_degenerate_batch_uses_mu()
    test_adaptive_sc_zscore_co_layout_rejected()
    test_fixed_ignores_mode_alpha()
    test_adaptive_uniform_halves()
    test_adaptive_sc_co_split_matches_manual()
    test_two_phase_sc_blend_formula()
    test_two_phase_magnitude_blend_formula()
    test_rowwise_pearson_corrcoef_basic_cases()
    test_rowwise_pearson_corrcoef_degenerate_cases()
    test_corr_weighted_mixed_reward_matches_manual()
    test_corr_weight_applies_after_final_normalize()
    test_adaptive_sc_zscore_corr_matches_pre_refactor_formula()
    test_corr_weight_threshold_gate()
    test_corr_weight_mode_none_matches_legacy()
    test_corr_weight_single_only_rejected()
    test_sc_corr_reliability_matches_manual_without_coefficient_normalization()
    test_reliability_full_matches_manual_and_weights_after_final_normalize()
    test_reliability_sample_only_matches_manual()
    test_reliability_advantage_only_matches_manual()
    test_reliability_six_weighted_variants_match_manual()
    test_reliability_baseline_exactly_matches_fixed_alpha050()
    test_reliability_degenerate_batch_uses_center_weights()
    test_reliability_rejects_invalid_settings_and_co_layout()
    test_zscore_with_bounds_basic()
    test_zscore_with_bounds_singleton_uses_mu()
    test_zscore_with_bounds_zero_std_uses_mu()
    test_zscore_with_bounds_invalid_params()
    print("test_adaptive_alpha_mixed_reward: all checks passed.")


if __name__ == "__main__":
    main()
