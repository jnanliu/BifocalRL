import ast
import asyncio
from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openrlhf.algorithms.bifocalrl.methods import compute_advantages, kl_cov_policy_loss
from openrlhf.algorithms.bifocalrl.rollout import generate_group
from openrlhf.algorithms.bifocalrl.runtime import DEFAULTS, METHODS, assign_advantages, configure
from openrlhf.algorithms.bifocalrl.voting import cluster_voting_rewards
from tests.bifocalrl.test_rollout import Engine, SamplingParams, Tokenizer
from tests.bifocalrl.test_runtime import arguments
from tests.test_experience import Experience


def test_cluster_rewards_include_minority_and_exclude_unparseable_from_denominator():
    assert cluster_voting_rewards(["1", "1.0", "2",
                                   None]) == pytest.approx([2 / 3 + 0.1, 2 / 3 + 0.1, 1 / 3 + 0.1, 0])
    assert cluster_voting_rewards([None, None]) == [0, 0]


def test_rollout_cluster_format_channel_has_no_ground_truth_dependency():
    class Label:
        def __str__(self):
            raise AssertionError("Training read a label")

    cfg = {**DEFAULTS, "method": "circa"}
    engine = Engine([r"\boxed{1}", r"\boxed{1.0}", r"\boxed{2}", "unparseable"])
    records = asyncio.run(
        generate_group(
            "q", Label(), SamplingParams(extra_args={"bifocal": cfg}), 128, Tokenizer(), engine, 4
        )
    )
    assert [r["reward"] for r in records] == pytest.approx([2 / 3 + 0.1, 2 / 3 + 0.1, 1 / 3 + 0.1, 0])
    assert [r["extra_logs"]["bifocal_mv"] for r in records] == [1, 1, 0, 0]


def test_circa_advantages_use_separate_group_norms_without_final_batch_norm():
    cluster = torch.tensor([[0.8, 0.8, 0.2, 0], [0.5, 0.5, 0.5, 0.5]])
    certainty = torch.tensor([[2.0, 1.0, 4.0, 8.0], [1.0, 2.0, 3.0, 4.0]])
    expected = 0.5 * (cluster - cluster.mean(-1, keepdim=True)) / (cluster.std(-1, keepdim=True) + 1e-6)
    expected += 0.5 * (certainty - certainty.mean(-1, keepdim=True)) / \
        (certainty.std(-1, keepdim=True) + 1e-6)
    actual = compute_advantages("circa", torch.zeros_like(cluster), certainty, prm=cluster)
    torch.testing.assert_close(actual, expected)
    # The vocabulary-size constant in stored SCE cancels during group normalization.
    torch.testing.assert_close(
        actual, compute_advantages("circa", cluster, certainty + 10, prm=cluster)
    )
    args = arguments("circa")
    args.bifocal.settings = {**DEFAULTS, "method": "circa"}
    order = [4, 0, 6, 2, 7, 3, 5, 1]
    exp = Experience(
        index=order,
        rewards=cluster.flatten()[order],
        action_mask=torch.tensor([[1, 1, 0]] * 8),
        info={
            "bifocal_mv": torch.zeros(8),
            "bifocal_sc": certainty.flatten()[order],
            "bifocal_entropy": torch.ones(8),
            "bifocal_aux": cluster.flatten()[order],
        }
    )
    assign_advantages(NS(strategy=NS(args=args)), [exp])
    torch.testing.assert_close(exp.advantages[:, 0], expected.flatten()[order])
    assert torch.equal(exp.advantages[:, 2], torch.zeros(8))


def test_kl_cov_replaces_clipping_on_selected_tokens_and_detaches_selection():
    current = torch.tensor([[-0.1, -1.0, -3.0, -4.0, 1000.0]], requires_grad=True)
    old = torch.tensor([[-0.5, -1.1, -3.2, -3.0, 0.0]], requires_grad=True)
    adv = torch.tensor([[3.0, -1.0, 2.0, -2.0, 1e6]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    loss, selected = kl_cov_policy_loss(current, old, adv, mask, fraction=0.5, coefficient=0.05)
    assert selected.tolist() == [[True, False, False, True, False]]
    ratio = (current.detach() - old.detach()).clamp(-20, 20).exp()
    expected = -torch.minimum(ratio * adv.detach(), ratio.clamp(0.8, 1.2) * adv.detach())
    expected[0, [0, 3]] = -ratio[0, [0, 3]] * adv.detach()[0, [0, 3]] + 0.05 * \
        (current.detach() - old.detach()).abs()[0, [0, 3]]
    expected[0, 4] = 0
    torch.testing.assert_close(loss, expected)
    loss.sum().backward()
    assert current.grad[0, 0] != 0 and current.grad[0, 4] == 0
    assert old.grad is None and adv.grad is None
    plain, selected = kl_cov_policy_loss(current, old, adv, mask, fraction=0)
    assert not selected.any()
    torch.testing.assert_close(
        plain[0, :4], -torch.minimum(ratio * adv.detach(),
                                     ratio.clamp(0.8, 1.2) * adv.detach())[0, :4]
    )


def _distributed_covariance_worker(rank, init_file, empty_rank):
    logp = torch.tensor([[-0.1, -1.0, -3.0, -4.0]])
    old = logp - torch.tensor([[0.4, 0.1, 0.2, -1.0]])
    adv = torch.tensor([[3.0, -1.0, 2.0, -2.0]])
    full_loss, full_selected = kl_cov_policy_loss(logp, old, adv, torch.ones_like(logp), fraction=0.5)
    split = 0 if empty_rank else 3
    start, end = (0, split) if rank == 0 else (split, 4)
    local_mask = torch.zeros_like(logp)
    local_mask[:, start:end] = 1
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        loss, selected = kl_cov_policy_loss(logp, old, adv, local_mask, fraction=0.5)
        torch.testing.assert_close(loss, full_loss * local_mask)
        assert torch.equal(selected, full_selected & local_mask.bool())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty_rank", [False, True])
def test_distributed_covariance_matches_unsharded_selection(tmp_path, empty_rank):
    mp.spawn(
        _distributed_covariance_worker, args=(str(tmp_path / "gloo-init"), empty_rank), nprocs=2, join=True
    )


def test_native_training_step_backpropagates_kl_cov_loss():
    path = Path(__file__).resolve().parents[2] / "openrlhf/trainer/ray/ppo_actor.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ActorPPOTrainer"
    )
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "training_step")
    namespace = {
        "torch": torch,
        "fields": fields,
        "Experience": Experience,
        "aggregate_loss": lambda loss, mask, **kw: (loss * mask).sum() / mask.sum(),
        "masked_mean": lambda value, mask, **kw: (value * mask).sum() / mask.sum()
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), str(path), "exec"
        ), namespace
    )

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logp = torch.nn.Parameter(torch.tensor([[-0.1, -1.0, -3.0, -4.0]]))

        def forward(self, *args, **kwargs):
            return self.logp, {}

    actor = Actor()
    args = arguments("circa")
    args.bifocal.settings = {**DEFAULTS, "method": "circa", "kl_cov_fraction": 0.5}
    args.actor.eps_clip_low_high, args.actor.entropy_coef = [0.2, 0.2], None
    args.train.dynamic_batch_enable = False
    strategy = NS(
        ds_device_mesh={"dp": NS(get_group=lambda: None)},
        ring_attn_group=None,
        backward=lambda loss, *a: loss.backward(),
        optimizer_step=lambda *a, **kw: None,
        get_grad_norm=lambda model: model.logp.grad.norm()
    )
    trainer = NS(
        args=args,
        actor=actor,
        strategy=strategy,
        aux_loss=False,
        ema_model=None,
        actor_optim=None,
        actor_scheduler=NS(get_last_lr=lambda: [1e-6]),
        actor_loss_fn=lambda *a, **kw: (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), None, None)
    )
    exp = Experience(
        sequences=torch.ones(1, 5),
        attention_mask=torch.ones(1, 5),
        action_mask=torch.ones(1, 4),
        action_log_probs=torch.tensor([[-0.5, -1.1, -3.2, -3.0]]),
        advantages=torch.tensor([[3., -1., 2., -2.]]),
        info={}
    )
    status = namespace["training_step"](trainer, exp, 0, 0, {})
    reference = actor.logp.detach().clone().requires_grad_()
    expected, _ = kl_cov_policy_loss(
        reference, exp.action_log_probs, exp.advantages, exp.action_mask, fraction=0.5
    )
    expected.mean().backward()
    torch.testing.assert_close(status["metrics"]["policy_loss"], expected.mean())
    torch.testing.assert_close(actor.logp.grad, reference.grad)
    assert exp.info["kl_cov_fraction"] == 0.5


def test_supported_methods_and_recipes_are_exactly_the_requested_set():
    expected = {
        "ttrl", "intuitor", "bifocalrl", "reliability_sc_corr", "ettrl", "covo", "prism", "circa"
    }
    recipes = Path(__file__).resolve().parents[2] / "examples/bifocalrl/configs"
    assert set(METHODS) == expected == {p.stem for p in recipes.glob("*.json")}
    args = arguments("reliability_sc_corr")
    configure(args)
    assert args.bifocal.settings["mixed"]["alpha_mode"] == "reliability_sc_corr"


@pytest.mark.parametrize(
    "method", [
        "mixed", "legacy_prism", "legacy_ettrl", "ettrl_etmr", "ettrl_adv_clip", "ettrl_adv_res",
        "covo_released", "prism_quadratic", "adaptive"
    ]
)
def test_removed_methods_are_rejected(method):
    with pytest.raises(ValueError, match="Unknown BifocalRL method"):
        configure(arguments(method))


@pytest.mark.parametrize(
    "field,value", [("policy_loss_type", "gspo"), ("dual_clip", 3), ("loss_agg_mode", "seq-mean-token-mean")]
)
def test_circa_rejects_incompatible_objectives(field, value):
    args = arguments("circa")
    setattr(args.actor, field, value)
    with pytest.raises(ValueError, match="KL-Cov requires"):
        configure(args)
