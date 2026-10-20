"""vLLM prompt-group rollout and optional independent GenPRM scoring."""

import asyncio
import hashlib
import importlib
import math
import random
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from .methods import covo_curiosity, covo_features, covo_rewards, prism_reward
from .voting import cluster_voting_rewards, extract_answers, is_math_equivalent, weighted_majority_voting


async def score_continuation(engine, prefix, continuation):
    from vllm import SamplingParams

    if not prefix or not continuation:
        raise ValueError("CoVo teacher forcing requires nonempty prefix and continuation")
    # vLLM requires a positive generation budget; discard that extra token.
    result = await engine.generate(
        list(prefix) + list(continuation), SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=1)
    )
    rows = result.prompt_logprobs[-len(continuation) :]
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

        ground_truth = str(label).split("####")[-1].strip()
        ground_truth = extract_answers([ground_truth])[0] or ground_truth
        rewards = [float(answer is not None and grade(answer, ground_truth, fast=True)) for answer in answers]
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
