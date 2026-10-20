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
