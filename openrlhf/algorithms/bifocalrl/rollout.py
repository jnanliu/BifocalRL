"""vLLM prompt-group rollout and optional independent GenPRM scoring."""

# SandboxFusion scoring adapted from VERL; Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0

import asyncio
import hashlib
import importlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .methods import covo_curiosity, covo_features, covo_rewards, prism_reward
from .voting import cluster_voting_rewards, extract_answers, is_math_equivalent, weighted_majority_voting

_SANDBOX_SEMAPHORES = {}


def compute_code_score(
    sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, completion, test_cases, continuous=False, timeout=10
):
    """Score Python test cases remotely, retaining VERL's SandboxFusion comparison semantics."""
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        solution = completion.split("```")[1]
        if "\n" in solution:
            first, rest = solution.split("\n", 1)
            if first.strip().isalpha():
                solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    def run_case(index):
        stdin = test_cases["inputs"][index]
        stdin = None if stdin is None else str(stdin)
        expected = test_cases["outputs"][index]
        assertions = test_cases.get("assert_case", [""] * len(test_cases["inputs"]))
        code = solution + "\n\n" + str(assertions[index])
        fn_name = test_cases.get("fn_name")
        if fn_name:
            # Benchmark starter code can omit these common imports. This wrapper runs only in the sandbox.
            modules = (
                "string re datetime collections heapq bisect copy math random statistics "
                "itertools functools operator io sys json builtins typing"
            ).split()
            imports = "".join(f"from {name} import *\n" for name in modules)
            imports += "".join(f"import {name}\n" for name in modules) + "import traceback\n"
            code = imports + code + f'''
_SANDBOX_FN_NAME = {fn_name!r}
if __name__ == '__main__':
    try:
        _raw = sys.stdin.read()
        _args = [json.loads(line) for line in _raw.split('\\n')] if _raw.strip() else []
        _fn = globals().get(_SANDBOX_FN_NAME)
        if _fn is None:
            _fn = getattr(Solution(), _SANDBOX_FN_NAME)
        _value = _fn(*_args)
    except Exception:
        traceback.print_exc()
    else:
        print(json.dumps(_value) if isinstance(_value, (dict, list, tuple, bool)) or _value is None else str(_value))
'''
        payload = {
            "code": code,
            "stdin": stdin,
            "language": "python",
            "compile_timeout": timeout,
            "run_timeout": timeout,
            "memory_limit_MB": memory_limit_mb,
            "files": {},
            "fetch_files": [],
        }
        metadata = {"case_index": index, "input": stdin, "expected_output": expected}
        try:
            with concurrent_semaphore:
                for attempt in range(3):
                    request = Request(
                        sandbox_fusion_url,
                        data=json.dumps(payload).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        },
                    )
                    try:
                        with urlopen(request, timeout=2 * timeout + 10) as response:
                            result = json.load(response)
                        break
                    except HTTPError as exc:
                        exc.close()
                        if exc.code != 504 or attempt == 2:
                            raise
                        logging.getLogger(__name__).warning("Sandbox Fusion HTTP 504; retry %s/2", attempt + 1)
                        time.sleep(attempt + 1)
        except Exception as exc:
            return False, {**metadata, "status": "api_error", "api_request_error": str(exc)}

        if not isinstance(result, dict):
            return False, {**metadata, "status": "unexpected_response", "api_response": result}
        metadata["api_response"] = result
        compile_result = result.get("compile_result") or {}
        run_result = result.get("run_result") or {}
        if result.get("status") == "SandboxError":
            status = "sandbox_error"
        elif result.get("status") == "Failed":
            if compile_result.get("status") in ("Error", "TimeLimitExceeded") or (
                compile_result.get("status") == "Finished" and compile_result.get("return_code") != 0
            ):
                status = "compile_timeout" if compile_result.get("status") == "TimeLimitExceeded" else "compile_error"
            elif run_result.get("status") == "TimeLimitExceeded":
                status = "timeout"
            elif run_result.get("status") == "Error" or (
                run_result.get("status") == "Finished" and run_result.get("return_code") != 0
            ):
                status = "runtime_error"
            else:
                status = "unknown_failure"
        elif result.get("status") == "Success" and run_result.get("status") == "Finished":
            stdout = run_result.get("stdout") or ""
            passed = expected is None or str(stdout).rstrip("\n") == str(expected).rstrip("\n")
            status = "success" if passed else "wrong_answer"
        else:
            status = "unexpected_response"
        return status == "success", {**metadata, "status": status}

    workers = min(len(test_cases["inputs"]), max(32, (os.cpu_count() or 1) * 5))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(run_case, range(len(test_cases["inputs"]))))
    compile_failed = False
    for index, (passed, metadata) in enumerate(outcomes):
        if compile_failed and metadata["status"] in ("success", "wrong_answer", "runtime_error", "timeout"):
            metadata["status"] = "compile_error_skipped"
            outcomes[index] = False, metadata
        compile_failed |= metadata["status"] in ("compile_error", "compile_timeout")
    considered = outcomes[:10] if continuous else outcomes
    return sum(passed for passed, _ in considered) / len(considered), [metadata for _, metadata in outcomes]


async def score_continuation(engine, prefix, continuation):
    from vllm import SamplingParams

    if not prefix or not continuation:
        raise ValueError("CoVo teacher forcing requires nonempty prefix and continuation")
    # vLLM requires a positive generation budget; discard that extra token.
    result = await engine.generate(
        list(prefix) + list(continuation), SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=1))
    rows = result.prompt_logprobs[-len(continuation):]
    if len(rows) != len(continuation):
        raise ValueError("vLLM prompt logprobs do not cover the forced continuation")
    values = []
    for token, row in zip(continuation, rows, strict=True):
        if row is None or token not in row:
            raise ValueError("vLLM omitted a forced prompt token logprob")
        value = row[token].logprob
        if not math.isfinite(value) or value > 1e-5:
            raise ValueError("Invalid teacher-forced logprob")
        values.append(value)
    return values


async def score_covo(cfg, responses, tokenizer, engine):
    answers = extract_answers([r["response_text"] for r in responses])
    unique, answer_ids = [], []
    for answer in answers:
        if answer is None:
            answer_ids.append(None)
            continue
        own = next((i for i, other in enumerate(unique) if is_math_equivalent(answer, other)), None)
        if own is None:
            own = len(unique)
            unique.append(answer)
        answer_ids.append(own)
    candidates = [tokenizer.encode(answer, add_special_tokens=False) for answer in unique]
    semaphore = asyncio.Semaphore(cfg["score_concurrency"])

    async def score(prefix, continuation):
        async with semaphore:
            return await score_continuation(engine, prefix, continuation)

    async def trajectory(response, own):
        if own is None:
            return (0.0, 0.0), 0.0
        prompt_length = response["action_ranges"][0][0]
        prefix = list(response["observation_tokens"][:prompt_length])
        distances, steps = [], []
        for step in reasoning_steps(response["response_text"], cfg["covo_max_steps"]):
            tokens = tokenizer.encode(step, add_special_tokens=False)
            if not tokens:
                continue
            scored = await asyncio.gather(*(score(prefix, answer) for answer in candidates), score(prefix, tokens))
            distances.append([-sum(lp) / len(lp) for lp in scored[:-1]])
            steps.append(scored[-1])
            prefix.extend(tokens)
        feature = covo_features(distances, own) if distances and len(unique) > 1 else (0.0, 0.0)
        return feature, covo_curiosity(steps)

    scored = await asyncio.gather(*(trajectory(r, own) for r, own in zip(responses, answer_ids, strict=True)))
    features, curiosities = zip(*scored, strict=True)
    rewards = covo_rewards(
        features,
        answer_ids,
        curiosities,
        aggregation=cfg["covo_aggregation"],
        curiosity_weight=cfg["covo_curiosity_weight"],
    )
    for response, feature, curiosity in zip(responses, features, curiosities, strict=True):
        response["extra_logs"].update(
            covo_consistency=feature[0], covo_volatility=feature[1], covo_curiosity=curiosity
        )
    return rewards


async def generate_group(prompt, label, sampling_params, max_length, tokenizer, engine, num_samples, images=None):
    metadata = sampling_params.extra_args
    cfg = metadata["bifocal"]
    evaluation = metadata.get("bifocal_evaluation", False)
    if images:
        raise ValueError("BifocalRL currently supports text-only rollout")
    params = deepcopy(sampling_params)
    params.extra_args = None
    params.n = 1
    params.logprobs = 1
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if params.max_tokens is None:
        budget = max_length - len(prompt_tokens)
    else:
        budget = min(params.max_tokens, max_length - 1)
        prompt_tokens = prompt_tokens[-(max_length - budget) :]
    if not prompt_tokens or budget < 1:
        raise ValueError("Prompt must leave room for a nonempty response")
    generation_index = 0

    async def generate(prefix, inherited_tokens=(), inherited_logprobs=(), extra_logs=None):
        nonlocal generation_index
        current = deepcopy(params)
        current.max_tokens = min(budget - len(inherited_tokens), max_length - len(prefix))
        if current.max_tokens < 1:
            raise ValueError("ETMR fork has no continuation budget")
        current.min_tokens = min(current.min_tokens, current.max_tokens)
        if current.seed is not None:
            current.seed += generation_index
        generation_index += 1
        output = (await engine.generate(prefix, current)).outputs[0]
        tokens = list(output.token_ids)
        logprobs = []
        if not tokens or output.logprobs is None or len(output.logprobs) != len(tokens):
            raise ValueError("BifocalRL needs nonempty generation with captured token logprobs")
        for token, row in zip(tokens, output.logprobs, strict=True):
            if token not in row:
                raise ValueError("vLLM omitted the sampled token's logprob")
            logprobs.append(row[token].logprob)
        response_tokens = list(inherited_tokens) + tokens
        response_logprobs = list(inherited_logprobs) + logprobs
        return {
            "prompt": prompt,
            "label": label,
            "images": None,
            "mm_train_inputs": None,
            "observation_tokens": prompt_tokens + response_tokens,
            "action_ranges": [(len(prompt_tokens), len(prompt_tokens) + len(response_tokens))],
            "rollout_log_probs": [0.0] * len(prompt_tokens) + response_logprobs,
            "response_text": tokenizer.decode(response_tokens, skip_special_tokens=False),
            "truncated": output.finish_reason == "length",
            "extra_logs": extra_logs or {},
        }

    if not evaluation and cfg["method"] == "ettrl" and cfg["tree_enabled"]:
        m, n, b = (cfg[k] for k in ("tree_roots", "tree_forks", "tree_branches"))
        if num_samples != m * (1 + n * b):
            raise ValueError("ETMR group size must equal M*(1+N*B)")
        roots = await asyncio.gather(*(generate(prompt_tokens) for _ in range(m)))
        tasks = []
        for root_index, root in enumerate(roots):
            root["extra_logs"].update(etmr_root=root_index, etmr_fork=-1)
            lp = root["rollout_log_probs"][len(prompt_tokens) :]
            if len(lp) < n:
                raise ValueError("ETMR root is shorter than tree_forks")
            forks = sorted(range(len(lp)), key=lambda i: (-lp[i], -i), reverse=True)[:n]
            root_tokens = root["observation_tokens"][len(prompt_tokens) :]
            for fork in forks:
                tasks.extend(
                    generate(
                        prompt_tokens + root_tokens[:fork],
                        root_tokens[:fork],
                        lp[:fork],
                        {"etmr_root": root_index, "etmr_fork": fork},
                    )
                    for _ in range(b)
                )
        responses = list(roots) + list(await asyncio.gather(*tasks))
    else:
        responses = list(await asyncio.gather(*(generate(prompt_tokens) for _ in range(num_samples))))
    answers = extract_answers([r["response_text"] for r in responses])
    if not evaluation and cfg["method"] == "bifocalrl":
        representatives = []
        for response, answer in zip(responses, answers, strict=True):
            cluster = next(
                (i for i, other in enumerate(representatives) if is_math_equivalent(answer, other)), None
            )
            # Unparseable answers remain separate: None is not an equivalent answer.
            if cluster is None:
                cluster = len(representatives)
                representatives.append(answer)
            response["extra_logs"]["bifocal_answer_cluster"] = cluster
    seed = cfg["voting_seed"] + int(hashlib.sha256(prompt.encode()).hexdigest()[:16], 16)
    winner = weighted_majority_voting(answers, [1.0] * len(answers), rng=random.Random(seed)).selected_answer
    votes = [float(is_math_equivalent(answer, winner)) for answer in answers]
    if evaluation:
        from .legacy.math import grade

        specification = label
        if isinstance(label, str):
            try:
                specification = json.loads(label)
            except json.JSONDecodeError:
                pass
        backend = specification.get("backend", "math") if isinstance(specification, dict) else "math"
        if backend == "sandbox_fusion":
            if not cfg["sandbox_fusion_url"]:
                raise ValueError("Code evaluation requires --bifocal.sandbox_fusion_url or sandbox_fusion_url in JSON")
            cases = specification["test_cases"]
            cases = json.loads(cases) if isinstance(cases, str) else deepcopy(cases)
            if not isinstance(cases, dict):
                raise ValueError("Code test_cases must be a dictionary or JSON object")
            if "assert_case" in cases:
                if not isinstance(cases["assert_case"], list):
                    raise ValueError("assert_case must be a list of Python assertions")
                cases.setdefault("inputs", [""] * len(cases["assert_case"]))
                cases.setdefault("outputs", [None] * len(cases["assert_case"]))
            inputs, outputs = cases.get("inputs"), cases.get("outputs")
            if (
                not isinstance(inputs, list)
                or not inputs
                or not isinstance(outputs, list)
                or len(inputs) != len(outputs)
                or ("assert_case" in cases and len(cases["assert_case"]) != len(inputs))
            ):
                raise ValueError("Code evaluation needs nonempty, equally sized inputs/outputs/assert_case lists")
            url = cfg["sandbox_fusion_url"].rstrip("/")
            if not url.endswith("/run_code"):
                url += "/run_code"
            # Shared across prompt groups in this worker; test cases run in threads.
            semaphore = _SANDBOX_SEMAPHORES.setdefault(
                cfg["score_concurrency"], threading.BoundedSemaphore(cfg["score_concurrency"])
            )
            rewards = []
            for response in responses:
                fraction, execution = await asyncio.to_thread(
                    compute_code_score,
                    sandbox_fusion_url=url,
                    concurrent_semaphore=semaphore,
                    memory_limit_mb=cfg["sandbox_memory_limit_mb"],
                    completion=response["response_text"],
                    test_cases=deepcopy(cases),
                    continuous=False,
                    timeout=cfg["sandbox_timeout"],
                )
                if not math.isfinite(fraction) or not 0 <= fraction <= 1 or not execution:
                    raise RuntimeError("Sandbox Fusion returned an invalid score or missing execution metadata")
                for case in execution:
                    # Infrastructure/data errors must not masquerade as an incorrect program.
                    error = case.get("error", "")
                    if (
                        case.get("api_request_error")
                        or case.get("status") in ("api_error", "sandbox_error", "internal_error")
                        or str(case.get("status", "")).startswith(("unknown", "unexpected"))
                        or (error and not error.startswith("Invalid completion"))
                    ):
                        raise RuntimeError(f"Sandbox Fusion evaluation failed: {case}")
                # pass@k requires binary all-tests-passed rewards.
                rewards.append(float(fraction == 1.0))
                response["extra_logs"].update(code_test_pass_rate=float(fraction), code_num_tests=len(inputs))
        elif backend == "math":
            ground_truth = str(label).split("####")[-1].strip()
            ground_truth = extract_answers([ground_truth])[0] or ground_truth
            rewards = [float(answer is not None and grade(answer, ground_truth, fast=True)) for answer in answers]
        else:
            raise ValueError(f"Unknown evaluation backend: {backend}")
    elif cfg["method"] == "covo":
        rewards = await score_covo(cfg, responses, tokenizer, engine)
    elif cfg["method"] == "prism":
        rewards = await asyncio.gather(
            *(
                score_prism(cfg, SimpleNamespace(prompt=prompt, response=r["response_text"], metadata={}))
                for r in responses
            )
        )
    elif cfg["method"] == "circa":
        rewards = cluster_voting_rewards(answers, format_weight=cfg["circa_format_weight"])
    else:
        rewards = votes
    for response, vote, reward in zip(responses, votes, rewards, strict=True):
        response.update(reward=float(reward), scores=float(reward))
        response["extra_logs"].update(bifocal_mv=vote, bifocal_aux=float(reward))
    return responses


@lru_cache(maxsize=8)
def tokenizer_for(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def reasoning_steps(response, max_steps):
    """Newline steps; merge contiguous steps when a trajectory exceeds the cap."""
    # Exclude the final boxed-answer line from intermediate reasoning states.
    boxed = response.rfind("\\boxed")
    end = response.rfind("\n", 0, boxed) if boxed >= 0 else len(response)
    reasoning = response[: max(0, end)] if boxed >= 0 else response
    pieces = [p + "\n" for p in reasoning.splitlines() if p.strip()]
    if len(pieces) > max_steps:
        pieces = [
            "".join(pieces[i * len(pieces) // max_steps : (i + 1) * len(pieces) // max_steps])
            for i in range(max_steps)
        ]
    return pieces


PRM_SYSTEM = (
    "Review the supplied solution to the question. Identify and judge each paragraph in order, "
    "then give a completion judgement for the entire solution. Use one line for each paragraph "
    "in the format 'Paragraph N **Judgement**: $\\boxed{Yes}$' or 'Paragraph N **Judgement**: $\\boxed{No}$'. "
    "End with 'Completion **Judgement**: $\\boxed{Yes}$' or 'Completion **Judgement**: $\\boxed{No}$'. "
    "Judge reasoning correctness and whether the question is fully solved."
)

PRM_PREFIX = "<output>Let's provide judgements for each of the paragraphs.\nParagraph 1 **Judgement**: $\\boxed{"

VERDICT = re.compile(r"(Paragraph\s+\d+|Completion|Overall)[^\n]*?\\boxed\{\s*(Yes|No)", re.IGNORECASE)


def parse_judgements(text):
    matches = list(VERDICT.finditer(text))
    process = [m for m in matches if m.group(1).lower().startswith("paragraph")]
    completion = [m for m in matches if not m.group(1).lower().startswith("paragraph")]
    if not process or len(completion) != 1 or completion[0].start() < process[-1].end():
        raise ValueError(
            "GenPRM must return paragraph judgements followed by exactly one completion judgement; increase prm_max_tokens or provide prm_function"
        )
    return process, completion[0]


async def score_prism(cfg, sample):
    if cfg["prm_function"]:
        module_name, function_name = cfg["prm_function"].rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)

        output = await function(cfg, sample)
        return prism_reward(output["step_probabilities"], output["completion_probability"])
    tokenizer = tokenizer_for(cfg["prm_model"])
    system = Path(cfg["prm_prompt_file"]).read_text() if cfg["prm_prompt_file"] else PRM_SYSTEM
    # Dataset prompts may be rendered chat strings. They are passed as problem
    # text without labels; the PRM has its own tokenizer/chat template.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question:\n{sample.prompt}\n\nSolution:\n{sample.response}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PRM_PREFIX
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    import aiohttp

    url = cfg["prm_url"].rstrip("/")
    if not url.endswith("/generate"):
        url += "/generate"
    semaphore = asyncio.Semaphore(cfg["score_concurrency"])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": cfg["prm_max_tokens"],
                "skip_special_tokens": False,
            },
            "return_logprob": True,
        }
        async with session.post(url, json=payload) as result:
            result.raise_for_status()
            output = await result.json()
        text = PRM_PREFIX + output["text"]
        process, completion = parse_judgements(text)

        async def probability(match):
            # Teacher-force Yes at the actual verdict context, including when
            # GenPRM generated No. The sampled token's probability is not P(Yes).
            prefix = tokenizer.encode(
                prompt + output["text"][: match.start(2) - len(PRM_PREFIX)], add_special_tokens=False
            )
            yes = tokenizer.encode("Yes", add_special_tokens=False)
            payload = {
                "input_ids": prefix + yes,
                "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
                "return_logprob": True,
                "logprob_start_len": len(prefix) - 1,
            }
            async with semaphore, session.post(url, json=payload) as result:
                result.raise_for_status()
                rows = (await result.json())["meta_info"]["input_token_logprobs"][-len(yes) :]
            if len(rows) != len(yes) or [r[1] for r in rows] != yes:
                raise ValueError("PRM logprobs are misaligned with the forced Yes tokens")
            if any(r[0] is None or not math.isfinite(r[0]) or r[0] > 1e-5 for r in rows):
                raise ValueError("Invalid GenPRM token log probability")
            return math.exp(sum(r[0] for r in rows))

        probabilities = await asyncio.gather(*(probability(m) for m in [*process, completion]))
    sample.metadata.update(prism_step_probabilities=probabilities[:-1], prism_completion_probability=probabilities[-1])
    return prism_reward(probabilities[:-1], probabilities[-1])
