# coding=utf-8
"""
Ground-truth scoring for completions against reference answers.

Uses HACRL-derived scorers (sympy, Minerva, boxed, QA, etc.) to compare model
completions against dataset ground-truth answers, producing 0/1 correctness rewards.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _extract_gold(raw_gold: str, dataset_hint: Optional[str]) -> str:
    """Pre-extract the actual answer from raw dataset gold text.

    HACRL scorers expect already-extracted gold (e.g. just the number for
    GSM8K, or just the boxed content for MATH).  Raw dataset fields contain
    full solution derivations, so we strip those here.

    TTRL datasets (_ttt suffix) already have clean answer strings, so we
    pass them through directly.
    """
    if dataset_hint and dataset_hint.endswith("_ttt"):
        return str(raw_gold).strip()
    if dataset_hint and "gsm8k" in dataset_hint:
        from eval.common.gsm8k import extract_gsm8k_final_answer
        return extract_gsm8k_final_answer(raw_gold)
    from eval.common.math_equiv import extract_math_final_answer
    return extract_math_final_answer(raw_gold)


def score_completions_against_ground_truth(
    completions: List[str],
    ground_truths: List[str],
    num_generations: int,
    scorer_name: str = "hacrl_prime_math",
    dataset_hint: Optional[str] = None,
) -> List[float]:
    """Score each completion against its corresponding ground-truth answer.

    Args:
        completions: Flattened list of model completions
            (num_prompts * num_generations).
        ground_truths: Per-prompt ground-truth answers (length = num_prompts).
            Each GT is repeated internally for the ``num_generations`` completions
            belonging to that prompt.
        num_generations: Number of completions per prompt.
        scorer_name: HACRL scorer backend (see ``eval.scoring.ScorerName``).
        dataset_hint: Optional dataset identifier passed to ``get_scorer``
            (only relevant when *scorer_name* is ``"default"``).

    Returns:
        List of float scores (same length as *completions*), each 0.0 or 1.0.
    """
    from eval.scorers import get_scorer

    num_prompts = len(ground_truths)
    expected = num_prompts * num_generations
    if len(completions) != expected:
        raise ValueError(
            f"completions length ({len(completions)}) != "
            f"num_prompts ({num_prompts}) * num_generations ({num_generations}) = {expected}"
        )

    extracted_gts = [_extract_gold(gt, dataset_hint) for gt in ground_truths]

    scorer = get_scorer(scorer_name, dataset_hint=dataset_hint)
    scores: List[float] = []

    for i in range(num_prompts):
        gt = extracted_gts[i]
        start = i * num_generations
        end = start + num_generations
        for comp in completions[start:end]:
            try:
                result = scorer.score(comp, gt)
                scores.append(float(result.score))
            except Exception as exc:
                logger.warning("GT scorer raised %s for prompt %d; defaulting to 0.0", exc, i)
                scores.append(0.0)

    n_correct = sum(1 for s in scores if s > 0.5)
    logger.info(
        "GT scoring done: %d/%d correct (%.1f%%), scorer=%s",
        n_correct, len(scores), 100.0 * n_correct / max(len(scores), 1), scorer_name,
    )
    return scores
