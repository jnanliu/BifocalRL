# coding=utf-8
"""
Difficulty estimation for curriculum / self-paced / continual learning.

Runs a model through prompts (vLLM rollout), estimates per-prompt difficulty
based on majority-vote (MV) agreement or intuitor self-certainty (SCE), and
partitions prompts into N equally-sized difficulty buckets.

This module provides **interface only** — it is not wired into the training
loop by default.  Use :func:`run_difficulty_estimation` as the end-to-end
entry point, or call the lower-level helpers directly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

BUCKET_LABELS = {0: "easy", 1: "medium", 2: "hard"}


@dataclass(frozen=True)
class DifficultyResult:
    """Per-prompt difficulty info."""
    prompt_index: int
    difficulty_score: float
    bucket_id: int = -1
    bucket_label: str = ""


@dataclass
class DifficultyReport:
    """Full output of a difficulty estimation run."""
    method: str
    n_buckets: int
    results: List[DifficultyResult] = field(default_factory=list)
    bucket_assignments: Dict[int, List[int]] = field(default_factory=dict)
    bucket_labels: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MV-based difficulty
# ---------------------------------------------------------------------------

def estimate_difficulty_mv(
    responses: Sequence[Optional[str]],
    num_generations: int,
) -> List[float]:
    """Estimate per-prompt difficulty from MV agreement rate.

    For each prompt group of *num_generations* responses, we run unweighted
    majority voting and count how many responses match the winning answer.
    Difficulty = 1 - (agreement_ratio).  High MV agreement -> easy.

    Args:
        responses: Flattened responses (num_prompts * num_generations).
        num_generations: Number of responses per prompt.

    Returns:
        List of difficulty scores in [0, 1], one per prompt.
    """
    from utils.mv import extract_answers, weighted_majority_voting, is_math_equivalent

    num_prompts = len(responses) // num_generations
    difficulties: List[float] = []

    for i in range(num_prompts):
        start = i * num_generations
        end = start + num_generations
        group = list(responses[start:end])
        answers = extract_answers(group)
        vote = weighted_majority_voting(answers, [1.0] * len(answers))

        if vote.selected_answer is None:
            difficulties.append(1.0)
            continue

        n_match = sum(
            1 for a in answers
            if a is not None and is_math_equivalent(a, vote.selected_answer)
        )
        agreement = n_match / num_generations
        difficulties.append(1.0 - agreement)

    return difficulties


# ---------------------------------------------------------------------------
# Intuitor (SCE)-based difficulty
# ---------------------------------------------------------------------------

def estimate_difficulty_intuitor(
    sce_scores: Sequence[float],
    num_generations: int,
) -> List[float]:
    """Estimate per-prompt difficulty from mean SCE confidence.

    Higher mean SCE across a prompt's rollouts indicates the model is more
    confident, implying an easier problem.  We invert and normalize to [0, 1]
    using min-max scaling across the batch.

    Args:
        sce_scores: Flattened SCE scores (num_prompts * num_generations).
        num_generations: Number of scores per prompt.

    Returns:
        List of difficulty scores in [0, 1], one per prompt.
    """
    num_prompts = len(sce_scores) // num_generations
    means: List[float] = []
    for i in range(num_prompts):
        start = i * num_generations
        end = start + num_generations
        group = list(sce_scores[start:end])
        means.append(sum(group) / max(len(group), 1))

    if not means:
        return []

    min_v = min(means)
    max_v = max(means)
    spread = max_v - min_v
    if spread < 1e-12:
        return [0.5] * len(means)

    # Higher SCE = easier -> invert: difficulty = 1 - normalized_mean
    return [1.0 - (m - min_v) / spread for m in means]


# ---------------------------------------------------------------------------
# Bucket partitioning
# ---------------------------------------------------------------------------

def partition_into_buckets(
    difficulty_scores: Sequence[float],
    n_buckets: int = 3,
) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Evenly split prompts into *n_buckets* difficulty buckets.

    Prompts are sorted by difficulty score (ascending) and then divided into
    *n_buckets* roughly equal-sized groups.  Bucket 0 is easiest.

    Args:
        difficulty_scores: Per-prompt difficulty (length = num_prompts).
        n_buckets: Number of buckets (default 3: easy / medium / hard).

    Returns:
        (bucket_assignments, bucket_labels):
          - bucket_assignments: ``{bucket_id: [prompt_indices]}``
          - bucket_labels: ``{bucket_id: label_str}``
    """
    n = len(difficulty_scores)
    if n == 0:
        return {}, {}

    sorted_indices = sorted(range(n), key=lambda i: difficulty_scores[i])
    bucket_size = math.ceil(n / n_buckets)

    assignments: Dict[int, List[int]] = {}
    labels: Dict[int, str] = {}
    for b in range(n_buckets):
        start = b * bucket_size
        end = min(start + bucket_size, n)
        indices = sorted_indices[start:end]
        assignments[b] = indices
        if n_buckets == 3 and b in BUCKET_LABELS:
            labels[b] = BUCKET_LABELS[b]
        else:
            labels[b] = f"bucket_{b}"

    return assignments, labels


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

def run_difficulty_estimation(
    model_path: str,
    dataset_key: str,
    *,
    method: str = "mv",
    n_buckets: int = 3,
    num_generations: int = 8,
    max_samples: Optional[int] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_completion_length: int = 2048,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    seed: int = 42,
    max_model_len: Optional[int] = None,
    scoring_batch_size: int = 8,
    max_prompt_length: Optional[int] = 512,
) -> DifficultyReport:
    """End-to-end difficulty estimation: generate rollouts, score, partition.

    Args:
        model_path: HF model checkpoint path.
        dataset_key: Dataset key from ``recipes/datasets.json``.
        method: ``"mv"`` (majority-vote agreement) or ``"intuitor"`` (SCE).
        n_buckets: Number of difficulty buckets (default 3).
        num_generations: Generations per prompt for difficulty probing.
        max_samples: Cap on number of prompts.
        system_prompt: Optional system prompt.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_completion_length: Max tokens per completion.
        gpu_memory_utilization: vLLM GPU memory fraction.
        tensor_parallel_size: vLLM tensor parallelism.
        dtype: Model dtype.
        seed: Random seed.
        max_model_len: Max model context length.
        scoring_batch_size: Batch size for SCE scoring (intuitor method only).
        max_prompt_length: Max prompt length for SCE scoring (intuitor method only).

    Returns:
        :class:`DifficultyReport` with per-prompt results and bucket assignments.
    """
    from transformers import AutoTokenizer

    from utils.dataset import load_co_intuitor_dataset
    from utils.generate import create_offline_llm, generate_sequences, shutdown_offline_llm
    from utils.prompt_utils import render_prompts

    problems = load_co_intuitor_dataset(
        dataset_key=dataset_key,
        max_samples=max_samples,
        system_prompt=system_prompt,
    )
    logger.info("Loaded %d problems from '%s'", len(problems), dataset_key)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = render_prompts(
        problems, tokenizer,
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    effective_max_model_len = max_model_len
    if effective_max_model_len is None:
        effective_max_model_len = (max_prompt_length or 512) + max_completion_length + 64

    llm = create_offline_llm(
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        seed=seed,
        max_model_len=effective_max_model_len,
    )

    try:
        gen_out = generate_sequences(
            model_path=model_path,
            prompts=prompts,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_length,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            max_model_len=effective_max_model_len,
            llm=llm,
        )
        flat_responses = gen_out.response_texts
    finally:
        shutdown_offline_llm(llm)

    if method == "mv":
        difficulty_scores = estimate_difficulty_mv(flat_responses, num_generations)
    elif method == "intuitor":
        from utils.reward import score_responses
        from utils.model_utils import get_model

        model = get_model(model_path, device="cuda")
        flat_prompts = [p for p in prompts for _ in range(num_generations)]
        sce_scores = score_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=flat_prompts,
            responses=flat_responses,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            batch_size=scoring_batch_size,
        )
        import gc, torch as _torch
        del model
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        difficulty_scores = estimate_difficulty_intuitor(sce_scores, num_generations)
    else:
        raise ValueError(f"Unknown difficulty method {method!r}. Must be 'mv' or 'intuitor'.")

    assignments, labels = partition_into_buckets(difficulty_scores, n_buckets)

    # Build per-prompt results with bucket info.
    idx_to_bucket: Dict[int, int] = {}
    for bid, indices in assignments.items():
        for idx in indices:
            idx_to_bucket[idx] = bid

    results: List[DifficultyResult] = []
    for i, score in enumerate(difficulty_scores):
        bid = idx_to_bucket.get(i, -1)
        results.append(DifficultyResult(
            prompt_index=i,
            difficulty_score=score,
            bucket_id=bid,
            bucket_label=labels.get(bid, ""),
        ))

    report = DifficultyReport(
        method=method,
        n_buckets=n_buckets,
        results=results,
        bucket_assignments=assignments,
        bucket_labels=labels,
        metadata={
            "model_path": model_path,
            "dataset_key": dataset_key,
            "num_prompts": len(problems),
            "num_generations": num_generations,
            "temperature": temperature,
            "seed": seed,
        },
    )

    for bid in sorted(assignments.keys()):
        indices = assignments[bid]
        if indices:
            bucket_scores = [difficulty_scores[j] for j in indices]
            logger.info(
                "Bucket %d (%s): %d prompts, difficulty range [%.3f, %.3f]",
                bid, labels.get(bid, "?"), len(indices),
                min(bucket_scores), max(bucket_scores),
            )

    return report
