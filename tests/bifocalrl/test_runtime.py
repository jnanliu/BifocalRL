import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from openrlhf.algorithms.bifocalrl.methods import compute_advantages
from openrlhf.algorithms.bifocalrl.runtime import DEFAULTS, METHODS, assign_advantages, configure
from tests.test_experience import Experience


def arguments(method="bifocalrl"):
    return NS(
        bifocal=NS(config=None, method=method, prm_url=None, prm_model=None),
        rollout=NS(n_samples_per_prompt=4, top_p=1),
        vllm=NS(num_engines=1),
        train=NS(async_enable=False, agent_func_path=None, seed=42),
        data=NS(max_images_per_prompt=0),
        ds=NS(tensor_parallel_size=1),
        reward=NS(remote_url=None, model_name_or_path=None, overlong_buffer_len=None, stop_properly_penalty_coef=None),
        critic=NS(model_name_or_path=None),
        actor=NS(policy_loss_type="ppo", dual_clip=None, loss_agg_mode="token-mean"),
        algo=NS(kl=NS(init_coef=0, use_loss=False), advantage=NS(gamma=1, is_correction_level="off")),
    )


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).resolve().parents[2] / "examples/bifocalrl/configs").glob("*.json")),
    ids=lambda p: p.stem,
)
def test_shipped_recipes(path):
    args = arguments(None)
    args.bifocal.config = str(path)
    args.bifocal.prm_url = "http://localhost:31000"
    args.rollout.n_samples_per_prompt = 8
    configure(args)
    assert args.bifocal.settings["method"] in METHODS
    assert args.reward.remote_url == "agent"
    assert args.critic.model_name_or_path is None


def test_native_entry_and_missing_prm():
    args = arguments(None)
    configure(args)
    assert args.bifocal.settings is None and args.reward.remote_url is None
    with pytest.raises(ValueError, match="PRISM requires"):
        configure(arguments("prism"))


@pytest.mark.parametrize(
    "field,value,message", [("async_enable", True, "synchronous"), ("agent_func_path", "custom.py", "synchronous")]
)
def test_incompatible_training_flags(field, value, message):
    args = arguments()
    setattr(args.train, field, value)
    with pytest.raises(ValueError, match=message):
        configure(args)


@pytest.mark.parametrize(
    "case",
    [
        case for case in json.loads(Path(__file__).with_name("parity.json").read_text())["cases"]
        if case["config"]["alpha_mode"] == "reliability_sc_corr"
    ],
    ids=lambda c: c["config"]["alpha_mode"],
)
def test_query_and_batch_statistics_survive_reordered_microbatches(case):
    data = json.loads(Path(__file__).with_name("parity.json").read_text())
    mv, sc = torch.tensor(data["mv"]).flatten(), torch.tensor(data["sc"]).flatten()
    args = arguments()
    method = "reliability_sc_corr" if case["config"]["alpha_mode"] == "reliability_sc_corr" else "bifocalrl"
    cfg = {**DEFAULTS, "method": method, "mixed": case["config"]}
    args.bifocal.settings = cfg
    maker = NS(strategy=NS(args=args))
    permutation = [[9, 2, 11, 6], [0, 8, 3, 10], [7, 4, 1, 5]]
    experiences = []
    for indices in permutation:
        experiences.append(
            Experience(
                index=indices,
                rewards=mv[indices],
                action_mask=torch.tensor([[0.0, 1.0, 1.0]] * 4),
                info={"bifocal_mv": mv[indices], "bifocal_sc": sc[indices], "bifocal_entropy": torch.ones(4)},
            )
        )
    assign_advantages(maker, experiences)
    expected = torch.tensor(case["expected"]).flatten()
    for exp in experiences:
        torch.testing.assert_close(exp.advantages[:, 1], expected[exp.index], atol=0, rtol=0)
        assert torch.equal(exp.advantages[:, 0], torch.zeros(4))
        assert exp.kl is None and "bifocal_gamma" in exp.info


def test_covo_whitening_uses_all_active_tokens():
    args = arguments("covo")
    args.bifocal.settings = {**DEFAULTS, "method": "covo"}
    rewards = [1.0, 2.0, 3.0, 4.0]
    masks = [torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])]
    experiences = [
        Experience(
            index=[0, 2],
            rewards=torch.tensor([1.0, 3.0]),
            action_mask=masks[0],
            info={"bifocal_mv": torch.tensor([1.0, 0.0]), "bifocal_aux": torch.tensor([1.0, 3.0])},
        ),
        Experience(
            index=[1, 3],
            rewards=torch.tensor([2.0, 4.0]),
            action_mask=masks[1],
            info={"bifocal_mv": torch.tensor([0.0, 0.0]), "bifocal_aux": torch.tensor([2.0, 4.0])},
        ),
    ]
    assign_advantages(NS(strategy=NS(args=args)), experiences)
    full = torch.tensor([1.0, 1.0, 3.0, 2.0, 2.0, 2.0, 4.0])
    for exp in experiences:
        unnormalized = torch.tensor([rewards[i] for i in exp.index])[:, None].expand_as(exp.action_mask)
        expected = (unnormalized - full.mean()) / (full.var() + 1e-8).sqrt() * exp.action_mask
        torch.testing.assert_close(exp.advantages, expected)


@pytest.mark.parametrize("step,gamma", [(0, 1.0), (5, 0.25), (10, 0.0)])
def test_prism_quadratic_schedule_uses_restored_step(step, gamma):
    args = arguments("prism")
    args.bifocal.settings = {
        **DEFAULTS,
        "method": "prism",
        "prism_gamma_schedule": "quadratic",
        "prism_gamma_steps": 11,
    }
    scores = torch.tensor([1.0, 2.0, 4.0, 8.0])
    exp = Experience(
        index=list(range(4)),
        rewards=scores,
        action_mask=torch.ones(4, 1),
        info={
            "bifocal_mv": torch.ones(4),
            "bifocal_sc": scores,
            "bifocal_entropy": scores,
            "bifocal_aux": scores.flip(0),
        },
    )
    assign_advantages(NS(strategy=NS(args=args), bifocal_step=step), [exp])
    expected = compute_advantages("prism", torch.ones(1, 4), scores[None, :], prm=scores.flip(0)[None, :], gamma=gamma)
    torch.testing.assert_close(exp.advantages[:, 0], expected.flatten())


def test_incomplete_group_is_rejected():
    args = arguments()
    args.bifocal.settings = dict(DEFAULTS)
    exp = Experience(index=[0], rewards=torch.ones(1), action_mask=torch.ones(1, 1))
    with pytest.raises(ValueError, match="Incomplete"):
        assign_advantages(NS(strategy=NS(args=args)), [exp])


def test_monitor_cluster_populations_and_correctness():
    from openrlhf.algorithms.bifocalrl.runtime import group_monitor_metrics

    ids = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    sc = torch.tensor([[1., 3., 9., 100.], [5., 2., 4., 100.]])
    correct = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.]])
    advantages = torch.tensor([[1., -1., 2., 0.], [-2., 0., 1., 0.]])
    result = group_monitor_metrics(ids, valid, sc, advantages, correct)
    assert result['cluster_count'] == 4
    assert result['cluster_size_mean'] == 1.5
    assert result['singleton_cluster_fraction'] == .5
    assert result['singleton_response_fraction'] == pytest.approx(1 / 3)
    assert result['largest_cluster_fraction'] == pytest.approx(2 / 3)
    assert result['parse_failure_rate'] == .25
    assert result['correct_cluster_sc_mean'] == 2.5
    assert result['incorrect_cluster_sc_mean'] == 7
    assert result['correct_positive_advantage_fraction'] == .5
    assert result['incorrect_positive_advantage_fraction'] == .5
    assert result['correct_positive_cluster_advantage_fraction'] == .5
    assert result['sc_within_rms'] == pytest.approx((4 / 6) ** .5)
    keep = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]])
    retained = group_monitor_metrics(ids, valid, sc, advantages, correct, keep)
    assert retained['cluster_count'] == 2
    assert retained['empty_query_fraction'] == .5
    assert retained['singleton_cluster_fraction'] == 1
    assert retained['sc_within_rms'] == 0


@pytest.mark.parametrize('filtered', [False, True])
def test_monitor_empty_populations_omit_conditional_means(filtered):
    from openrlhf.algorithms.bifocalrl.runtime import group_monitor_metrics

    ids = torch.arange(4)[None, :]
    result = group_monitor_metrics(
        ids, torch.full((1, 4), filtered), sc=torch.ones(1, 4),
        correct=torch.full((1, 4), -1), keep=torch.full((1, 4), not filtered),
    )
    assert result['cluster_count'] == result['labeled_response_count'] == 0
    assert result['empty_query_fraction'] == 1
    assert 'correct_cluster_sc_mean' not in result
    assert 'singleton_cluster_fraction' not in result
    assert 'sc_within_rms' not in result
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())


def test_monitor_uses_global_order_and_never_changes_advantages():
    from copy import deepcopy

    args = arguments()
    args.bifocal.settings = deepcopy(DEFAULTS)
    info = {
        'bifocal_mv': torch.tensor([1., 1., 0., 0., 1., 1., 1., 0.]),
        'bifocal_sc': torch.tensor([1., 2., 3., 4., 6., 4., 5., 1.]),
        'bifocal_entropy': torch.ones(8),
        'bifocal_answer_cluster': torch.tensor([0, 0, 1, 2, 0, 0, 0, 1]),
        'bifocal_answer_valid': torch.ones(8),
        'bifocal_monitor_correct': torch.tensor([1., 1., 0., 0., 0., 0., 0., 1.]),
    }
    whole = Experience(index=list(range(8)), rewards=info['bifocal_mv'].clone(),
                       action_mask=torch.ones(8, 2), info=deepcopy(info))
    original = deepcopy(whole)
    del original.info['bifocal_answer_valid']
    assign_advantages(NS(strategy=NS(args=args)), [original])
    maker = NS(strategy=NS(args=args))
    assign_advantages(maker, [whole])
    torch.testing.assert_close(whole.advantages, original.advantages)
    expected_metrics = maker.bifocal_monitor_metrics.copy()
    split = [Experience(index=indices, rewards=whole.rewards[indices], action_mask=torch.ones(len(indices), 2),
                        info={key: value[indices] for key, value in info.items()})
             for indices in ([7, 3, 1], [6, 0, 5, 4, 2])]
    assign_advantages(maker, split)
    assert maker.bifocal_monitor_metrics == expected_metrics
    for exp in split:
        torch.testing.assert_close(exp.advantages, original.advantages[exp.index])
    altered = deepcopy(whole)
    altered.info['bifocal_monitor_correct'] = 1 - altered.info['bifocal_monitor_correct']
    assign_advantages(maker, [altered])
    torch.testing.assert_close(altered.advantages, original.advantages)
