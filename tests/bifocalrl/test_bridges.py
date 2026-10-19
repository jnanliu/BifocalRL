import ast
from copy import copy, deepcopy

# ruff: noqa: S102 -- Compile only the production CPU tensor function, without GPU imports.
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
def test_actor_statistics_are_raw_and_response_aligned(temperature):
    path = Path(__file__).resolve().parents[2] / "openrlhf/models/actor.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Actor")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "torch": torch,
        "log_probs_from_logits": lambda z, t: z.log_softmax(-1).gather(-1, t[..., None]).squeeze(-1),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, forward], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    sequences = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    table = torch.arange(25, dtype=torch.float32).reshape(5, 5) / 7
    actor = NS(
        packing_samples=False,
        temperature=temperature,
        bifocal_collect_stats=True,
        model=lambda seq, **kw: {"logits": table[seq].clone()},
    )
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
    with torch.no_grad():
        logps, output = namespace["forward"](actor, sequences, mask, torch.ones_like(sequences), return_output=True)
    logits = table[sequences]
    lp = logits.log_softmax(-1)
    torch.testing.assert_close(output["bifocal_sc"], -lp[:, :-1].mean(-1))
    torch.testing.assert_close(output["bifocal_entropy"], -(lp.exp() * lp).sum(-1)[:, :-1])
    expected = (logits / temperature).log_softmax(-1).gather(-1, torch.roll(sequences, -1, 1)[..., None]).squeeze(-1)[
        :, :-1
    ] * mask
    torch.testing.assert_close(logps, expected)


def test_actor_stats_do_not_run_during_policy_gradient():
    path = Path(__file__).resolve().parents[2] / "openrlhf/models/actor.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Actor")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "torch": torch,
        "log_probs_from_logits": lambda z, t: z.log_softmax(-1).gather(-1, t[..., None]).squeeze(-1),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, forward], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    table = torch.nn.Parameter(torch.randn(5, 5))
    seq = torch.tensor([[0, 1, 2]])
    actor = NS(
        packing_samples=False,
        temperature=1.0,
        bifocal_collect_stats=True,
        model=lambda seq, **kw: {"logits": table[seq]},
    )
    logps, output = namespace["forward"](actor, seq, torch.ones(1, 2), torch.ones_like(seq), return_output=True)
    assert "bifocal_sc" not in output
    logps.sum().backward()
    assert table.grad is not None and table.grad.abs().sum() > 0


def test_eval_can_load_labels_without_requiring_training_labels():
    root = Path(__file__).resolve().parents[2]
    path = root / "openrlhf/trainer/ppo_trainer.py"
    function = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "prepare_datasets"
    )
    seen = []

    class Rows(list):
        def select(self, indices):
            return Rows([self[i] for i in indices])

    class Dataset(list):
        collate_fn = None

        def __init__(self, rows, tokenizer, strategy, **kwargs):
            super().__init__(rows)
            seen.append(strategy.args.data.label_key)

    namespace = {
        "PromptDataset": Dataset,
        "blending_datasets": lambda *a, **kw: Rows([{}, {}]),
        "copy": copy,
        "deepcopy": deepcopy,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    args = NS(
        data=NS(
            prompt_dataset="train",
            prompt_probs=None,
            max_samples=10,
            prompt_split="train",
            label_key=None,
            input_template=None,
            dataloader_num_workers=0,
        ),
        eval=NS(dataset="eval", split="train"),
        train=NS(seed=42, batch_size=8, num_episodes=1, max_epochs=1),
        rollout=NS(n_samples_per_prompt=4),
        bifocal=NS(settings={"eval_label_key": "label"}),
    )
    strategy = NS(args=args, setup_dataloader=lambda dataset, *a, **kw: dataset)
    namespace["prepare_datasets"](strategy, None)
    assert seen == [None, "label"]
    assert args.data.label_key is None
