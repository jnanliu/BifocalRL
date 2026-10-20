import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from openrlhf.algorithms.bifocalrl.methods import compute_advantages
from openrlhf.algorithms.bifocalrl.mixed_reward import MixedRewardConfig, compute_mixed_reward
from openrlhf.algorithms.bifocalrl.runtime import DEFAULTS, assign_advantages, configure
from tests.bifocalrl.test_runtime import arguments
from tests.test_experience import Experience

MV = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.], [1., 0., 0., 0.], [1., 1., 0., 0.]])
SC = torch.tensor([[4., 3., 2., 1.], [1., 4., 3., 2.], [1., 2., 3., 4.], [1., 2., 3., 4.]])
CLUSTERS = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 1, 1], [0, 0, 1, 1]])
KEEP = torch.tensor(
    [
        [True, True, False, True], [False, True, False, True], [False, True, False, False],
        [False, False, False, False]
    ]
)


def test_snr_filter_reweight_then_normalize_only_retained_rollouts():
    config = MixedRewardConfig(sc_between_gamma=1)
    diagnostics = {}
    actual = compute_advantages("bifocalrl", MV, SC, mixed=config, diagnostics=diagnostics)
    a_mv = (MV - MV.mean(1, keepdim=True)) / (MV.std(1, keepdim=True) + 1e-6)
    a_sc = (SC - SC.mean(1, keepdim=True)) / (SC.std(1, keepdim=True) + 1e-6)
    fused = 0.5 * a_mv + 0.5 * a_sc
    query_weight = torch.tensor([[0.75], [0.5], [0.25], [0.]])
    retained = (query_weight * fused)[KEEP]
    expected = torch.zeros_like(MV)
    expected[KEEP] = (retained - retained.mean()) / (retained.std(unbiased=False) + 1e-6)
    assert torch.equal(diagnostics["rollout_keep"].bool(), KEEP)
    torch.testing.assert_close(diagnostics["query_weight"], query_weight)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[~KEEP], torch.zeros_like(actual[~KEEP]))
    torch.testing.assert_close(actual[KEEP].mean(), torch.tensor(0.), atol=1e-6, rtol=0)
    torch.testing.assert_close(actual[KEEP].std(unbiased=False), torch.tensor(1.), atol=1e-5, rtol=0)
    # No multiplication after final normalization: query weights are already inside it.
    torch.testing.assert_close(actual, diagnostics["combined_after_final_normalize"])


def test_hard_filter_treats_zero_channel_as_neutral():
    mv = torch.ones(2, 4)
    mv[1] = MV[3]
    sc = SC[3].repeat(2, 1)
    diagnostics = {}
    actual = compute_mixed_reward(
        mv, sc, config=MixedRewardConfig(gd2po_filter_mode="hard", sc_between_gamma=1), diagnostics=diagnostics
    )
    assert diagnostics["rollout_keep"][0].all()
    assert not diagnostics["rollout_keep"][1].any()
    assert not actual[1].any()


def test_filter_decisions_do_not_depend_on_other_queries():
    full = {}
    compute_mixed_reward(MV, SC, metadata={"answer_cluster_ids": CLUSTERS}, diagnostics=full)
    for i in range(len(MV)):
        single = {}
        compute_mixed_reward(
            MV[i:i + 1], SC[i:i + 1],
            metadata={"answer_cluster_ids": CLUSTERS[i:i + 1]}, diagnostics=single
        )
        assert torch.equal(single["rollout_keep"], full["rollout_keep"][i:i + 1])
        assert torch.equal(single["snr_threshold"], torch.tensor([[0.5]]))
        assert torch.equal(single["mv_weight"], torch.tensor([[0.5]]))
        torch.testing.assert_close(single["snr"], full["snr"][i:i + 1])
    assert "reliability" not in full


def test_disabling_filter_restores_fixed_fusion_snapshot():
    data = json.loads(Path(__file__).with_name("parity.json").read_text())
    fixed = next(case for case in data["cases"] if case["config"]["alpha_mode"] == "fixed")
    actual = compute_mixed_reward(
        torch.tensor(data["mv"]),
        torch.tensor(data["sc"]),
        config=MixedRewardConfig(gd2po_filter_mode="none", sc_between_gamma=1)
    )
    torch.testing.assert_close(actual, torch.tensor(fixed["expected"]), atol=0, rtol=0)


def test_query_reweight_can_be_disabled_independently():
    diagnostics = {}
    actual = compute_mixed_reward(
        MV,
        SC,
        config=MixedRewardConfig(gd2po_query_reweight=False),
        metadata={"answer_cluster_ids": CLUSTERS},
        diagnostics=diagnostics
    )
    fused = diagnostics["combined_before_final_normalize"][KEEP]
    expected = (fused - fused.mean()) / (fused.std(unbiased=False) + 1e-6)
    torch.testing.assert_close(actual[KEEP], expected)
    assert torch.equal(diagnostics["query_weight"], torch.ones(4, 1))
    assert torch.equal(diagnostics["retained_fraction"], KEEP.float().mean(1, keepdim=True))


@pytest.mark.parametrize("case", ["all_filtered", "flat_rewards", "one_retained"])
def test_empty_and_degenerate_retained_batches_are_zero(case):
    if case == "all_filtered":
        mv, sc = MV[3:], SC[3:]
    elif case == "flat_rewards":
        mv, sc = torch.ones(1, 4), torch.ones(1, 4)
    else:
        mv, sc = MV[2:3], SC[2:3]
    actual = compute_mixed_reward(mv, sc, metadata={"answer_cluster_ids": torch.tensor([[0, 0, 0, 1]])})
    assert torch.equal(actual, torch.zeros_like(actual))


def test_single_informative_channel_still_trains():
    actual = compute_mixed_reward(
        torch.ones(1, 4), SC[3:], metadata={"answer_cluster_ids": torch.zeros(1, 4, dtype=torch.long)}
    )
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    torch.testing.assert_close(actual.mean(), torch.tensor(0.), atol=1e-6, rtol=0)


def test_filtered_rollouts_have_zero_policy_gradient():
    advantages = compute_mixed_reward(MV, SC, metadata={"answer_cluster_ids": CLUSTERS})
    current = torch.zeros_like(advantages, requires_grad=True)
    ratio = current.exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
    loss.backward()
    assert not current.grad[~KEEP].any()
    assert current.grad[KEEP].abs().sum() > 0


def test_runtime_preserves_query_masks_weights_and_statistics_after_reordering():
    args = arguments()
    args.bifocal.settings = {**DEFAULTS, "mixed": dict(DEFAULTS["mixed"])}
    experiences = []
    for indices in ([15, 0, 6, 8], [5, 11, 2, 12], [1, 9, 7, 13], [10, 3, 14, 4]):
        experiences.append(
            Experience(
                index=list(indices),
                rewards=MV.flatten()[list(indices)],
                action_mask=torch.tensor([[0., 1., 1.]] * 4),
                info={
                    "bifocal_mv": MV.flatten()[list(indices)],
                    "bifocal_sc": SC.flatten()[list(indices)],
                    "bifocal_entropy": torch.ones(4),
                    "bifocal_answer_cluster": CLUSTERS.flatten()[list(indices)],
                },
            )
        )
    assign_advantages(NS(strategy=NS(args=args)), experiences)
    a_mv = (MV - MV.mean(1, keepdim=True)) / (MV.std(1, keepdim=True) + 1e-6)
    a_sc = (SC - SC.mean(1, keepdim=True)) / (SC.std(1, keepdim=True) + 1e-6)
    means = a_sc.masked_fill(~KEEP, 0)
    means[0, :2] = a_sc[0, :2].mean()
    adjusted = a_sc - 0.5 * means
    weighted = KEEP.float().mean(1, keepdim=True) * (0.5 * a_mv + 0.5 * adjusted)
    expected = torch.zeros_like(MV)
    expected[KEEP] = (weighted[KEEP] - weighted[KEEP].mean()) / (weighted[KEEP].std(unbiased=False) + 1e-6)
    for exp in experiences:
        torch.testing.assert_close(exp.advantages[:, 1], expected.flatten()[exp.index])
        assert not exp.advantages[:, 0].any()
        assert torch.equal(exp.info["bifocal_rollout_keep"].bool(), KEEP.flatten()[exp.index])
        torch.testing.assert_close(
            exp.info["bifocal_query_weight"],
            KEEP.float().mean(1).repeat_interleave(4)[exp.index]
        )
        assert torch.equal(exp.info["bifocal_snr_threshold"], torch.full((4, ), 0.5))
        torch.testing.assert_close(exp.info["bifocal_sc_cluster_mean"], means.flatten()[exp.index])
        torch.testing.assert_close(
            exp.info["bifocal_sc_adjusted"], adjusted.masked_fill(~KEEP, 0).flatten()[exp.index]
        )


@pytest.mark.parametrize(
    "setting,value", [
        ("gd2po_filter_mode", "unknown"),
        ("gd2po_snr_threshold", -0.1),
        ("gd2po_snr_threshold", float("nan")),
        ("sc_between_gamma", -0.1),
        ("sc_between_gamma", 1.1),
        ("sc_between_gamma", float("nan")),
        ("alpha", 0.7),
        ("final_normalize", False),
    ]
)
def test_invalid_settings_fail_before_training(tmp_path, setting, value):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"method": "bifocalrl", "mixed": {setting: value}}))
    args = arguments(None)
    args.bifocal.config = str(path)
    with pytest.raises(ValueError):
        configure(args)


@pytest.mark.parametrize("gamma", [0., 0.5, 1.])
def test_sc_decomposition_uses_only_kept_cluster_members(gamma):
    diagnostics = {}
    actual = compute_advantages(
        "bifocalrl", MV, SC, mixed=MixedRewardConfig(sc_between_gamma=gamma),
        metadata={"answer_cluster_ids": CLUSTERS}, diagnostics=diagnostics,
    )
    a_mv = (MV - MV.mean(1, keepdim=True)) / (MV.std(1, keepdim=True) + 1e-6)
    a_sc = (SC - SC.mean(1, keepdim=True)) / (SC.std(1, keepdim=True) + 1e-6)
    # Only the first cluster retains two members; all other retained clusters are singletons.
    means = a_sc.masked_fill(~KEEP, 0)
    means[0, :2] = a_sc[0, :2].mean()
    within = torch.zeros_like(a_sc)
    within[0, :2] = a_sc[0, :2] - means[0, :2]
    adjusted = a_sc - (1 - gamma) * means
    weighted = KEEP.float().mean(1, keepdim=True) * (0.5 * a_mv + 0.5 * adjusted)
    expected = torch.zeros_like(MV)
    retained = weighted[KEEP]
    expected[KEEP] = (retained - retained.mean()) / (retained.std(unbiased=False) + 1e-6)
    assert torch.equal(diagnostics["rollout_keep"].bool(), KEEP)
    torch.testing.assert_close(diagnostics["sc_cluster_mean"], means)
    torch.testing.assert_close(diagnostics["sc_within"], within)
    torch.testing.assert_close(diagnostics["sc_cluster_size"], torch.tensor(
        [[2., 2., 0., 1.], [0., 1., 0., 1.], [0., 1., 0., 0.], [0., 0., 0., 0.]]
    ))
    torch.testing.assert_close(actual, expected)
    if gamma == 1:
        original = compute_mixed_reward(MV, SC, config=MixedRewardConfig(sc_between_gamma=1))
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


def test_gamma_zero_removes_singleton_sc_but_preserves_same_answer_differences():
    mv = torch.ones(2, 4)
    sc = SC[:2]
    diagnostics = {}
    compute_mixed_reward(
        mv, sc, config=MixedRewardConfig(sc_between_gamma=0),
        metadata={"answer_cluster_ids": torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]])},
        diagnostics=diagnostics,
    )
    assert diagnostics["rollout_keep"].all()
    torch.testing.assert_close(diagnostics["sc_adjusted"][0], diagnostics["self_certainty_advantages"][0])
    assert not diagnostics["sc_adjusted"][1].any()


def test_filter_is_not_recomputed_after_sc_decomposition():
    # Original channels agree on response 4, but subtracting its cluster mean reverses SC.
    mv = torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0.]])
    sc = torch.tensor([[1., 1., 1., 1., -0.01, -5., 0.505, 0.505]])
    diagnostics = {}
    compute_mixed_reward(
        mv, sc, config=MixedRewardConfig(sc_between_gamma=0),
        metadata={"answer_cluster_ids": torch.tensor([[0, 0, 0, 0, 1, 1, 2, 3]])},
        diagnostics=diagnostics,
    )
    assert diagnostics["rollout_keep"].all()
    assert diagnostics["self_certainty_advantages"][0, 4] < 0
    assert diagnostics["sc_adjusted"][0, 4] > 0
    assert diagnostics["combined_before_final_normalize"][0, 4] > 0


@pytest.mark.parametrize("ids", [
    None, torch.zeros(4, dtype=torch.long), torch.zeros(4, 4),
    torch.full((4, 4), -1, dtype=torch.long), torch.full((4, 4), 4, dtype=torch.long),
])
def test_decomposition_requires_valid_query_local_cluster_ids(ids):
    with pytest.raises(ValueError, match="answer_cluster_ids"):
        compute_mixed_reward(MV, SC, metadata={"answer_cluster_ids": ids})
