import pytest
import torch

from src.training.co_intuitor_train import compute_mixed_reward_advantages
from utils.configs import CoIntuitorConfig
from utils.normalization import (
    batch_zscore_normalize,
    grpo_group_advantages,
    rowwise_pearson_corrcoef,
    z_score_normalize_tensor_with_bounds,
)


def _rewards() -> tuple[torch.Tensor, torch.Tensor]:
    mv = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    self_certainty = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [8.0, 4.0, 2.0, 6.0],
        ],
        dtype=torch.float32,
    )
    return mv, self_certainty


def test_prism_is_unit_weight_sum_without_batch_normalization() -> None:
    mv, self_certainty = _rewards()
    diagnostics: dict = {}
    actual = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="prism",
        final_normalize=False,
        diagnostics=diagnostics,
    )
    expected = grpo_group_advantages(mv) + grpo_group_advantages(self_certainty)

    assert torch.equal(actual, expected)
    assert torch.equal(
        diagnostics["mv_advantage_weight"],
        torch.ones_like(mv),
    )
    assert torch.equal(
        diagnostics["self_certainty_advantage_weight"],
        torch.ones_like(self_certainty),
    )
    assert torch.equal(
        diagnostics["combined_before_final_normalize"],
        diagnostics["combined_after_final_normalize"],
    )
    assert torch.equal(diagnostics["final_advantages"], actual)


def test_prism_is_exactly_twice_fixed_alpha050_when_both_skip_batch_norm() -> None:
    mv, self_certainty = _rewards()
    prism = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="prism",
        final_normalize=False,
    )
    fixed = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="fixed",
        final_normalize=False,
    )
    assert torch.equal(prism, 2.0 * fixed)


def test_ettrl_uses_raw_self_certainty_ratio_to_weight_mv_advantage() -> None:
    mv, self_certainty = _rewards()
    diagnostics: dict = {}
    actual = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="ettrl",
        final_normalize=False,
        diagnostics=diagnostics,
    )

    group_mean = self_certainty.mean(dim=1, keepdim=True)
    expected_weight = self_certainty / group_mean
    expected = expected_weight * grpo_group_advantages(mv)
    assert torch.equal(actual, expected)
    assert torch.equal(diagnostics["self_certainty_group_mean"], group_mean)
    assert torch.equal(diagnostics["mv_advantage_weight"], expected_weight)
    assert torch.equal(
        diagnostics["self_certainty_advantage_weight"],
        torch.zeros_like(self_certainty),
    )
    assert torch.allclose(
        diagnostics["mv_advantage_weight"].mean(dim=1),
        torch.ones(mv.size(0)),
    )


def test_ettrl_constant_and_all_zero_groups_reduce_to_mv_advantage() -> None:
    mv, _ = _rewards()
    self_certainty = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0],
            [7.0, 7.0, 7.0, 7.0],
        ],
        dtype=torch.float32,
    )
    diagnostics: dict = {}
    actual = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha_mode="ettrl",
        final_normalize=False,
        diagnostics=diagnostics,
    )

    assert torch.equal(actual, grpo_group_advantages(mv))
    assert torch.equal(
        diagnostics["mv_advantage_weight"],
        torch.ones_like(self_certainty),
    )
    assert torch.equal(
        diagnostics["ettrl_all_zero_group"].reshape(-1),
        torch.tensor([1.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize("mode", ["prism", "ettrl"])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"alpha": 0.4}, "mixed_reward_alpha=0.5"),
        ({"corr_weight_mode": "pearson_threshold_power"}, "corr_weight_mode='none'"),
        ({"final_normalize": True}, "mixed_reward_final_normalize=false"),
        ({"num_generations_per_model": 2}, "single-model"),
    ],
)
def test_new_modes_reject_incompatible_configuration(
    mode: str,
    overrides: dict,
    message: str,
) -> None:
    mv, self_certainty = _rewards()
    kwargs = {
        "alpha": 0.5,
        "alpha_mode": mode,
        "corr_weight_mode": "none",
        "final_normalize": False,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        compute_mixed_reward_advantages(mv, self_certainty, **kwargs)


@pytest.mark.parametrize(
    ("self_certainty", "message"),
    [
        (torch.tensor([[1.0, -1.0, 2.0, 3.0]]), "non-negative"),
        (torch.tensor([[1e-8, 2e-8, 1e-8, 2e-8]]), "mean <= eps"),
    ],
)
def test_ettrl_rejects_invalid_self_certainty(
    self_certainty: torch.Tensor,
    message: str,
) -> None:
    mv = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=message):
        compute_mixed_reward_advantages(
            mv,
            self_certainty,
            alpha_mode="ettrl",
            final_normalize=False,
        )


def test_legacy_default_still_applies_final_batch_zscore() -> None:
    mv, self_certainty = _rewards()
    pre_normalize = 0.5 * grpo_group_advantages(mv) + 0.5 * grpo_group_advantages(
        self_certainty
    )
    actual = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="fixed",
    )
    assert torch.equal(actual, batch_zscore_normalize(pre_normalize))


def test_legacy_adaptive_sc_zscore_corr_keeps_post_normalize_order() -> None:
    mv, self_certainty = _rewards()
    threshold = 0.1
    exponent = 0.8
    adv_mv = grpo_group_advantages(mv)
    adv_self = grpo_group_advantages(self_certainty)
    raw_sc = mv.mean(dim=1, keepdim=True)
    alpha = z_score_normalize_tensor_with_bounds(
        raw_sc,
        mu=0.65,
        sigma=0.2,
        lower=0.0,
        upper=1.0,
        dim=0,
    )
    combined = alpha * adv_mv + (1.0 - alpha) * adv_self
    normalized = batch_zscore_normalize(combined)
    corr = rowwise_pearson_corrcoef(adv_mv, adv_self, dim=1, keepdim=True)
    corr = torch.clamp(corr, -1.0, 1.0)
    corr_weight = torch.where(
        corr >= threshold,
        ((corr - threshold) / (1.0 - threshold)).pow(exponent),
        torch.zeros_like(corr),
    )
    expected = corr_weight * normalized

    actual = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha_mode="adaptive_sc_zscore",
        corr_weight_mode="pearson_threshold_power",
        corr_weight_threshold=threshold,
        corr_weight_exponent=exponent,
    )
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_training_config_preserves_legacy_final_normalize_default(tmp_path) -> None:
    config = CoIntuitorConfig(output_dir=str(tmp_path))
    assert config.mixed_reward_final_normalize is True
