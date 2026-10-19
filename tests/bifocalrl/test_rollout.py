import ast
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from openrlhf.algorithms.bifocalrl.rollout import generate_group, score_continuation
from openrlhf.algorithms.bifocalrl.runtime import DEFAULTS
from tests.test_experience import Experience


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()


class SamplingParams:
    def __init__(self, **kwargs):
        self.max_tokens = 64
        self.min_tokens = 1
        self.seed = None
        self.extra_args = None
        self.prompt_logprobs = None
        self.__dict__.update(kwargs)


class Engine:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    async def generate(self, prefix, params):
        self.calls.append((prefix[:], params))
        if params.prompt_logprobs is not None:
            lp = -0.1 if prefix[-1] == ord("1") else -2 if prefix[-1] == ord("2") else -0.7
            return NS(prompt_logprobs=[{token: NS(logprob=lp)} for token in prefix])
        tokens = list(next(self.outputs).encode())
        logprobs = [
            {token: NS(logprob=(-9.0 if i == 1 else -8.0 if i == 3 else -0.1))} for i, token in enumerate(tokens)
        ]
        return NS(outputs=[NS(token_ids=tokens, logprobs=logprobs, finish_reason="stop")])


def test_training_voting_never_reads_label_and_preserves_group():
    class UnreadableLabel:
        def __str__(self):
            raise AssertionError("Training tried to read a ground truth label")

    engine = Engine([r"\boxed{1}", r"\boxed{1}", r"\boxed{1}", r"\boxed{2}"])
    params = SamplingParams(extra_args={"bifocal": {**DEFAULTS, "method": "ttrl"}})
    output = asyncio.run(generate_group("q", UnreadableLabel(), params, 128, Tokenizer(), engine, 4))
    assert [r["reward"] for r in output] == [1, 1, 1, 0]
    assert all(call[1].extra_args is None for call in engine.calls)
    for record in output:
        assert len(record["observation_tokens"]) == len(record["rollout_log_probs"])
        assert record["action_ranges"][0][0] == 1


def test_etmr_resamples_fork_and_keeps_prefix_trainable():
    root = "reason\n" + r"\boxed{1}"
    engine = Engine([root] + [r"\boxed{2}"] * 4)
    params = SamplingParams(
        extra_args={"bifocal": {**DEFAULTS, "method": "ettrl", "tree_roots": 1, "tree_forks": 2, "tree_branches": 2}}
    )
    output = asyncio.run(generate_group("q", "unused", params, 128, Tokenizer(), engine, 5))
    assert len(output) == 5
    assert all(set(r["extra_logs"]) == set(output[0]["extra_logs"]) for r in output)
    assert [r["extra_logs"].get("etmr_fork") for r in output[1:]] == [1, 1, 3, 3]
    for branch in output[1:]:
        fork = branch["extra_logs"]["etmr_fork"]
        assert branch["observation_tokens"][1 : fork + 1] == list(root.encode())[:fork]
        assert branch["rollout_log_probs"][1 : fork + 1] == output[0]["rollout_log_probs"][1 : fork + 1]
        assert branch["action_ranges"] == [(1, len(branch["observation_tokens"]))]


def test_dynamic_token_budget_preserves_the_whole_prompt():
    engine = Engine([r"\boxed{1}"])
    params = SamplingParams(max_tokens=None, extra_args={"bifocal": {**DEFAULTS, "method": "ttrl"}})
    output = asyncio.run(generate_group("whole question", "unused", params, 128, Tokenizer(), engine, 1))
    assert engine.calls[0][0] == list(b"whole question")
    assert engine.calls[0][1].max_tokens == 128 - len("whole question")
    assert output[0]["action_ranges"][0][0] == len("whole question")


def test_eval_disables_tree_and_uses_label():
    engine = Engine([r"\boxed{2}"])
    params = SamplingParams(extra_args={"bifocal": {**DEFAULTS, "method": "ettrl"}, "bifocal_evaluation": True})
    output = asyncio.run(generate_group("q", "reasoning\n#### 2", params, 128, Tokenizer(), engine, 1))
    assert output[0]["reward"] == 1
    assert len(engine.calls) == 1


def test_covo_scores_current_engine_prefixes(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", NS(SamplingParams=SamplingParams))
    engine = Engine(["A\n" + r"\boxed{1}", "B\n" + r"\boxed{2}"])
    params = SamplingParams(extra_args={"bifocal": {**DEFAULTS, "method": "covo"}})
    output = asyncio.run(generate_group("q", "unused", params, 128, Tokenizer(), engine, 2))
    assert [r["reward"] for r in output] == pytest.approx([1.7, 0.7])
    assert len([p for _, p in engine.calls if p.prompt_logprobs is not None]) == 6
    assert output[0]["extra_logs"]["covo_consistency"] == 1
    assert output[1]["extra_logs"]["covo_consistency"] == 0


def test_prism_uses_real_provider_probabilities(monkeypatch):
    async def score(cfg, sample):
        assert sample.response and not hasattr(sample, "label")
        return {"step_probabilities": [0.8, 0.4], "completion_probability": 0.8}

    monkeypatch.setitem(sys.modules, "bifocal_test_prm", NS(score=score))
    cfg = {**DEFAULTS, "method": "prism", "prm_function": "bifocal_test_prm.score"}
    engine = Engine([r"\boxed{1}", r"\boxed{2}"])
    output = asyncio.run(
        generate_group("q", "unused", SamplingParams(extra_args={"bifocal": cfg}), 128, Tokenizer(), engine, 2)
    )
    assert [r["reward"] for r in output] == pytest.approx([0.5333333333] * 2)


def test_teacher_forcing_rejects_missing_target(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", NS(SamplingParams=SamplingParams))

    class BrokenEngine:
        async def generate(self, prefix, params):
            return NS(prompt_logprobs=[None] * len(prefix))

    with pytest.raises(ValueError, match="omitted"):
        asyncio.run(score_continuation(BrokenEngine(), [1], [2]))


def test_tree_records_survive_native_experience_padding_and_batching():
    root = "reason\n" + r"\boxed{1}"
    engine = Engine([root] + [r"\boxed{2}"] * 4)
    cfg = {**DEFAULTS, "method": "ettrl", "tree_roots": 1, "tree_forks": 2, "tree_branches": 2}
    records = asyncio.run(
        generate_group("q", "unused", SamplingParams(extra_args={"bifocal": cfg}), 128, Tokenizer(), engine, 5)
    )
    path = Path(__file__).resolve().parents[2] / "openrlhf/trainer/ppo_utils/samples_generator.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "SamplesGenerator"
    )
    function = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_process_response_into_experience"
    )
    namespace = {"torch": torch, "Experience": Experience}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    samples = [namespace[function.name](None, r, max_len=128) for r in records]
    batch = Experience.concat_experiences(samples, 0)
    assert len(batch.sequences) == 5
    assert all(len(value) == 5 for value in batch.info.values())
    assert batch.action_mask[:, 0].tolist() == [1] * 5
