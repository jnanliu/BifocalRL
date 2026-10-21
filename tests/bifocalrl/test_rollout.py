import ast
import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from openrlhf.algorithms.bifocalrl import rollout
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
        logprobs = [{
            token: NS(logprob=(-9.0 if i == 1 else -8.0 if i == 3 else -0.1))
        } for i, token in enumerate(tokens)]
        return NS(outputs=[NS(token_ids=tokens, logprobs=logprobs, finish_reason="stop")])


@pytest.mark.parametrize("method", ["ttrl", "bifocalrl"])
def test_training_voting_never_reads_label_and_preserves_group(method):
    class UnreadableLabel:
        def __str__(self):
            raise AssertionError("Training tried to read a ground truth label")

    engine = Engine([r"\boxed{1}", r"\boxed{1}", r"\boxed{1}", r"\boxed{2}"])
    params = SamplingParams(extra_args={"bifocal": {**DEFAULTS, "method": method}})
    output = asyncio.run(generate_group("q", UnreadableLabel(), params, 128, Tokenizer(), engine, 4))
    assert [r["reward"] for r in output] == [1, 1, 1, 0]
    assert all(call[1].extra_args is None for call in engine.calls)
    for record in output:
        assert len(record["observation_tokens"]) == len(record["rollout_log_probs"])
        assert record["action_ranges"][0][0] == 1


def test_bifocal_answer_clusters_use_math_equivalence_and_separate_invalid_answers():
    engine = Engine([
        r"\boxed{\frac{1}{2}}",
        r"\boxed{0.5}",
        r"\boxed{1/2}",
        "no answer",
        "no answer",
        r"\boxed{2}",
        r"\boxed{2}",
        r"\boxed{3}",
    ])
    params = SamplingParams(extra_args={"bifocal": DEFAULTS})
    records = asyncio.run(generate_group("q", "unused", params, 128, Tokenizer(), engine, 8))
    assert [r["extra_logs"]["bifocal_answer_cluster"] for r in records] == [0, 0, 0, 1, 2, 3, 3, 4]


def test_etmr_resamples_fork_and_keeps_prefix_trainable():
    root = "reason\n" + r"\boxed{1}"
    engine = Engine([root] + [r"\boxed{2}"] * 4)
    params = SamplingParams(
        extra_args={"bifocal": {
            **DEFAULTS, "method": "ettrl",
            "tree_roots": 1,
            "tree_forks": 2,
            "tree_branches": 2
        }})
    output = asyncio.run(generate_group("q", "unused", params, 128, Tokenizer(), engine, 5))
    assert len(output) == 5
    assert all(set(r["extra_logs"]) == set(output[0]["extra_logs"]) for r in output)
    assert [r["extra_logs"].get("etmr_fork") for r in output[1:]] == [1, 1, 3, 3]
    for branch in output[1:]:
        fork = branch["extra_logs"]["etmr_fork"]
        assert branch["observation_tokens"][1:fork + 1] == list(root.encode())[:fork]
        assert branch["rollout_log_probs"][1:fork + 1] == output[0]["rollout_log_probs"][1:fork + 1]
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


@pytest.mark.parametrize("encoded", [False, True])
def test_code_eval_uses_local_scorer_and_requires_all_tests_to_pass(monkeypatch, encoded):
    calls = []
    main_thread = threading.get_ident()
    cases = {"inputs": ["secret", "second"], "outputs": ["1", "2"], "fn_name": "solve"}

    def compute_score(**kwargs):
        assert threading.get_ident() != main_thread
        assert kwargs["test_cases"] == cases
        assert kwargs["sandbox_fusion_url"] == "http://sandbox/run_code"
        assert kwargs["timeout"] == 5 and kwargs["memory_limit_mb"] == 512
        assert not kwargs["continuous"]
        assert isinstance(kwargs["concurrent_semaphore"], threading.BoundedSemaphore)
        calls.append(kwargs)
        return (1. if len(calls) == 1 else 0.5), [{"status": "success"}, {"status": "wrong_answer"}]

    monkeypatch.setattr(rollout, "compute_code_score", compute_score)
    label = {"backend": "sandbox_fusion", "test_cases": json.dumps(cases) if encoded else cases}
    engine = Engine(["```python\nprint(1)\n```", "```python\nprint(2)\n```"])
    cfg = {
        **DEFAULTS, "method": "ettrl",
        "sandbox_fusion_url": "http://sandbox/",
        "sandbox_timeout": 5,
        "sandbox_memory_limit_mb": 512
    }
    params = SamplingParams(extra_args={"bifocal": cfg, "bifocal_evaluation": True})
    records = asyncio.run(
        generate_group("q",
                       json.dumps(label) if encoded else label, params, 128, Tokenizer(), engine, 2))
    assert [r["reward"] for r in records] == [1., 0.]
    assert [r["extra_logs"]["code_test_pass_rate"] for r in records] == [1., 0.5]
    assert all(r["extra_logs"]["code_num_tests"] == 2 for r in records)
    assert all("secret" not in bytes(prefix).decode() for prefix, _ in engine.calls)
    assert len(calls) == 2  # Evaluation bypasses ETMR and the training reward.


@pytest.mark.parametrize("execution", [
    [{
        "status": "api_error",
        "api_request_error": "connection refused"
    }],
    [{
        "status": "sandbox_error"
    }],
    [{
        "status": "unknown_failure"
    }],
    [{
        "error": "Invalid test_cases structure"
    }],
    [],
])
def test_code_eval_service_and_data_errors_do_not_become_incorrect_programs(monkeypatch, execution):
    monkeypatch.setattr(rollout, "compute_code_score", lambda **kwargs: (0., execution))
    cfg = {**DEFAULTS, "sandbox_fusion_url": "http://sandbox/run_code"}
    params = SamplingParams(extra_args={"bifocal": cfg, "bifocal_evaluation": True})
    label = {"backend": "sandbox_fusion", "test_cases": {"inputs": ["1"], "outputs": ["2"]}}
    with pytest.raises(RuntimeError, match="Sandbox Fusion"):
        asyncio.run(generate_group("q", label, params, 128, Tokenizer(), Engine(["```python\npass\n```"]), 1))


@pytest.mark.parametrize("status",
                         ["wrong_answer", "compile_error", "compile_error_skipped", "runtime_error", "timeout"])
def test_code_eval_program_failures_are_zero_scores(monkeypatch, status):
    monkeypatch.setattr(rollout, "compute_code_score", lambda **kwargs: (0., [{"status": status}]))
    cfg = {**DEFAULTS, "sandbox_fusion_url": "http://sandbox"}
    params = SamplingParams(extra_args={"bifocal": cfg, "bifocal_evaluation": True})
    label = {"backend": "sandbox_fusion", "test_cases": {"assert_case": ["assert solve() == 1"]}}
    records = asyncio.run(generate_group("q", label, params, 128, Tokenizer(), Engine(["```python\npass\n```"]), 1))
    assert records[0]["reward"] == 0.


def test_code_eval_missing_code_block_is_an_incorrect_program(monkeypatch):
    monkeypatch.setattr(rollout, "compute_code_score", lambda **kwargs:
                        (0., [{
                            "error": "Invalid completion (missing code block)"
                        }]))
    cfg = {**DEFAULTS, "sandbox_fusion_url": "http://sandbox"}
    params = SamplingParams(extra_args={"bifocal": cfg, "bifocal_evaluation": True})
    label = {"backend": "sandbox_fusion", "test_cases": {"inputs": ["1"], "outputs": ["2"]}}
    records = asyncio.run(generate_group("q", label, params, 128, Tokenizer(), Engine(["no code"]), 1))
    assert records[0]["reward"] == 0.


@pytest.mark.parametrize("cases", [
    {},
    {
        "inputs": [],
        "outputs": []
    },
    {
        "inputs": ["1"],
        "outputs": []
    },
    {
        "inputs": "1",
        "outputs": ["2"]
    },
    {
        "assert_case": "assert True"
    },
])
def test_code_eval_rejects_empty_or_mismatched_test_cases(monkeypatch, cases):
    monkeypatch.setattr(rollout, "compute_code_score",
                        lambda **kwargs: pytest.fail("Invalid cases reached the scorer"))
    cfg = {**DEFAULTS, "sandbox_fusion_url": "http://sandbox"}
    params = SamplingParams(extra_args={"bifocal": cfg, "bifocal_evaluation": True})
    with pytest.raises(ValueError):
        asyncio.run(
            generate_group("q", {
                "backend": "sandbox_fusion",
                "test_cases": cases
            }, params, 128, Tokenizer(), Engine(["```python\npass\n```"]), 1))


def test_code_eval_requires_endpoint_but_code_labels_are_ignored_in_training():
    label = {"backend": "sandbox_fusion", "test_cases": {"inputs": ["1"], "outputs": ["2"]}}
    params = SamplingParams(extra_args={"bifocal": DEFAULTS, "bifocal_evaluation": True})
    with pytest.raises(ValueError, match="sandbox_fusion_url"):
        asyncio.run(generate_group("q", label, params, 128, Tokenizer(), Engine(["code"]), 1))
    params.extra_args["bifocal_evaluation"] = False
    records = asyncio.run(generate_group("q", label, params, 128, Tokenizer(), Engine([r"\boxed{1}"]), 1))
    assert records[0]["reward"] == 1.
    assert "code_test_pass_rate" not in records[0]["extra_logs"]


@pytest.mark.parametrize("completion,cases,rate", [
    ("```python\nprint(int(input()) * 2)\n```", {
        "inputs": ["2\n", "3\n"],
        "outputs": ["4", "7"]
    }, 0.5),
    ("```\nprint('ok')\n```", {
        "inputs": [""],
        "outputs": ["ok\n\n"]
    }, 1.),
    ("```python\nprint(' ok ')\n```", {
        "inputs": [""],
        "outputs": ["ok"]
    }, 0.),
    ("```python\nprint('你好')\n```", {
        "inputs": [""],
        "outputs": ["你好"]
    }, 1.),
    ("```python\ndef add(a, b):\n    return a + b\n```", {
        "inputs": ["1\n2"],
        "outputs": ["3"],
        "fn_name": "add"
    }, 1.),
    ("```python\nclass Solution:\n    def solve(self, xs: List[int]):\n        return list(reversed(xs))\n```", {
        "inputs": ["[1, 2]"],
        "outputs": ["[2, 1]"],
        "fn_name": "solve"
    }, 1.),
    ("```python\ndef solve():\n    return True\n```", {
        "inputs": [""],
        "outputs": ["true"],
        "fn_name": "solve"
    }, 1.),
    ("```python\ndef solve():\n    return None\n```", {
        "inputs": [""],
        "outputs": ["null"],
        "fn_name": "solve"
    }, 1.),
    ("```python\ndef solve():\n    return 1\n```", {
        "assert_case": ["assert solve() == 1", "assert solve() == 2"]
    }, 0.5),
    ("```python\nprint(0)\n```\n```python\nprint(1)\n```", {
        "inputs": [""],
        "outputs": ["1"]
    }, 1.),
    ("```python\nassert False\n```", {
        "inputs": [""],
        "outputs": [""]
    }, 0.),
    ("No code block", {
        "inputs": [""],
        "outputs": ["1"]
    }, 0.),
])
def test_local_code_scoring_over_http_without_verl(monkeypatch, completion, cases, rate):
    import builtins
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    original_import = builtins.__import__

    def import_without_verl(name, *args, **kwargs):
        if name == "verl" or name.startswith("verl."):
            raise AssertionError("Code evaluation imported VERL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_verl)

    async def run():
        requests = []

        async def execute(request):
            payload = await request.json()
            requests.append(payload)
            assert payload["language"] == "python"
            assert payload["compile_timeout"] == payload["run_timeout"] == 2
            assert payload["memory_limit_MB"] == 256
            # This mock service executes only the trusted Python fixtures above, as a separate process.
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                payload["code"],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate((payload["stdin"] or "").encode())
            return web.json_response({
                "status": "Success" if process.returncode == 0 else "Failed",
                "run_result": {
                    "status": "Finished",
                    "return_code": process.returncode,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode()
                },
            })

        app = web.Application()
        app.router.add_post("/run_code", execute)
        async with TestServer(app) as server:
            cfg = {
                **DEFAULTS, "sandbox_fusion_url": str(server.make_url("/")),
                "sandbox_timeout": 2,
                "sandbox_memory_limit_mb": 256
            }
            params = SamplingParams(extra_args={"bifocal": cfg, "bifocal_evaluation": True})
            records = await generate_group("q", {
                "backend": "sandbox_fusion",
                "test_cases": cases
            }, params, 256, Tokenizer(), Engine([completion]), 1)
        assert records[0]["reward"] == float(rate == 1.)
        assert records[0]["extra_logs"]["code_test_pass_rate"] == rate
        assert len(requests) == (0 if completion == "No code block" else len(
            cases.get("inputs", cases.get("assert_case"))))

    asyncio.run(run())


@pytest.mark.parametrize("response,status", [
    ({
        "status": "Failed",
        "compile_result": {
            "status": "Finished",
            "return_code": 1
        }
    }, "compile_error"),
    ({
        "status": "Failed",
        "compile_result": {
            "status": "TimeLimitExceeded"
        }
    }, "compile_timeout"),
    ({
        "status": "Failed",
        "run_result": {
            "status": "TimeLimitExceeded"
        }
    }, "timeout"),
    ({
        "status": "Failed",
        "run_result": {
            "status": "Error"
        }
    }, "runtime_error"),
    ({
        "status": "SandboxError"
    }, "sandbox_error"),
    ({
        "status": "Success"
    }, "unexpected_response"),
    ([], "unexpected_response"),
])
def test_local_code_scorer_classifies_http_responses(response, status):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def run():
        async def execute(request):
            return web.json_response(response)

        app = web.Application()
        app.router.add_post("/run_code", execute)
        async with TestServer(app) as server:
            rate, metadata = await asyncio.to_thread(
                rollout.compute_code_score,
                str(server.make_url("/run_code")),
                threading.BoundedSemaphore(2),
                256,
                "```python\npass\n```",
                {
                    "inputs": ["", ""],
                    "outputs": ["", ""]
                },
            )
        assert rate == 0
        assert metadata[0]["status"] == status
        assert metadata[1]["status"] == status

    asyncio.run(run())


@pytest.mark.parametrize("failure,attempts", [(504, 3), (503, 1), ("invalid_json", 1), ("recover", 2)])
def test_local_code_scorer_retries_gateway_timeouts_and_reports_transport_errors(monkeypatch, failure, attempts):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    delays = []
    monkeypatch.setattr(rollout.time, "sleep", delays.append)

    async def run():
        calls = []

        async def execute(request):
            calls.append(await request.json())
            if failure == "recover":
                if len(calls) == 1:
                    return web.Response(status=504)
                return web.json_response({"status": "Success", "run_result": {"status": "Finished", "stdout": ""}})
            return web.Response(text="invalid json", status=200 if failure == "invalid_json" else failure)

        app = web.Application()
        app.router.add_post("/run_code", execute)
        async with TestServer(app) as server:
            rate, metadata = await asyncio.to_thread(
                rollout.compute_code_score,
                str(server.make_url("/run_code")),
                threading.BoundedSemaphore(2),
                256,
                "```python\npass\n```",
                {
                    "inputs": [""],
                    "outputs": [""]
                },
            )
        assert len(calls) == attempts
        if failure == "recover":
            assert rate == 1 and metadata[0]["status"] == "success"
        else:
            assert rate == 0 and metadata[0]["status"] == "api_error" and metadata[0]["api_request_error"]
        assert delays == ([1, 2] if failure == 504 else [1] if failure == "recover" else [])

    asyncio.run(run())


@pytest.mark.parametrize("last_response,last_status", [
    ({"status": "Success", "run_result": {"status": "Finished", "stdout": "1"}}, "compile_error_skipped"),
    ({"status": "SandboxError"}, "sandbox_error"),
])
def test_local_code_scorer_propagates_compile_errors_without_hiding_service_failures(last_response, last_status):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def run():
        async def execute(request):
            payload = await request.json()
            if payload["stdin"] == "0":
                return web.json_response({"status": "Failed", "compile_result": {"status": "Error"}})
            return web.json_response(last_response)

        app = web.Application()
        app.router.add_post("/run_code", execute)
        async with TestServer(app) as server:
            rate, metadata = await asyncio.to_thread(
                rollout.compute_code_score, str(server.make_url("/run_code")), threading.BoundedSemaphore(2),
                256, "```python\npass\n```", {"inputs": ["0", "1"], "outputs": ["1", "1"]},
            )
        assert rate == 0
        assert [m["status"] for m in metadata] == ["compile_error", last_status]

    asyncio.run(run())


def test_local_code_scoring_shares_concurrency_limit_across_queries():
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def run():
        active, peak = 0, 0

        async def execute(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return web.json_response({
                "status": "Success",
                "run_result": {
                    "status": "Finished",
                    "return_code": 0,
                    "stdout": "1\n",
                }
            })

        app = web.Application()
        app.router.add_post("/run_code", execute)
        async with TestServer(app) as server:
            cfg = {**DEFAULTS, "sandbox_fusion_url": str(server.make_url("/run_code")), "score_concurrency": 2}
            params = SamplingParams(extra_args={"bifocal": cfg, "bifocal_evaluation": True})
            label = {"backend": "sandbox_fusion", "test_cases": {"inputs": [""] * 4, "outputs": ["1"] * 4}}
            groups = await asyncio.gather(
                *(generate_group("q", label, params, 256, Tokenizer(), Engine(["```python\nprint(1)\n```"]), 1)
                  for _ in range(2)))
        assert peak == 2
        assert all(group[0]["reward"] == 1 for group in groups)

    asyncio.run(run())


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
        generate_group("q", "unused", SamplingParams(extra_args={"bifocal": cfg}), 128, Tokenizer(), engine, 2))
    assert [r["reward"] for r in output] == pytest.approx([0.5333333333] * 2)


def test_teacher_forcing_rejects_missing_target(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", NS(SamplingParams=SamplingParams))

    class BrokenEngine:
        async def generate(self, prefix, params):
            return NS(prompt_logprobs=[None] * len(prefix))

    with pytest.raises(ValueError, match="omitted"):
        asyncio.run(score_continuation(BrokenEngine(), [1], [2]))


@pytest.mark.parametrize("method", ["ettrl", "bifocalrl"])
def test_rollout_records_survive_native_experience_padding_and_batching(method):
    root = "reason\n" + r"\boxed{1}"
    engine = Engine([root] + [r"\boxed{2}"] * 4)
    cfg = {**DEFAULTS, "method": method, "tree_roots": 1, "tree_forks": 2, "tree_branches": 2}
    records = asyncio.run(
        generate_group("q", "unused", SamplingParams(extra_args={"bifocal": cfg}), 128, Tokenizer(), engine, 5))
    path = Path(__file__).resolve().parents[2] / "openrlhf/trainer/ppo_utils/samples_generator.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "SamplesGenerator")
    function = next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "_process_response_into_experience")
    namespace = {"torch": torch, "Experience": Experience}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    samples = [namespace[function.name](None, r, max_len=128) for r in records]
    batch = Experience.concat_experiences(samples, 0)
    assert len(batch.sequences) == 5
    assert all(len(value) == 5 for value in batch.info.values())
    assert batch.action_mask[:, 0].tolist() == [1] * 5
    if method == "bifocalrl":
        assert batch.info["bifocal_answer_cluster"].tolist() == [0, 1, 1, 1, 1]
