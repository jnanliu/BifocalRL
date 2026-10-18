"""TTRL GSM8K scoring (ported from `TTRL/verl/verl/utils/reward_score/gsm8k.py`)."""

from __future__ import annotations

import re
from typing import Any, Dict


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """Extract the final GSM8K answer using TTRL's original logic.

    Important: unlike HACRL's fork, TTRL does not clip the completion to the
    last N characters before applying the regex.
    """

    assert method in ("strict", "flexible")

    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    else:
        answers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answers) != 0:
            invalid_str = ["", "."]
            for final_answer in reversed(answers):
                if final_answer not in invalid_str:
                    break

    return final_answer


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Score a GSM8K response with TTRL's original exact-match rule."""

    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0.0
    if answer == ground_truth:
        return score
    return format_score


def gsm8k_compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> Dict[str, Any]:
    """Structured wrapper used by co-learning scorers."""

    extracted = extract_solution(solution_str, method=method) or ""
    score_val = compute_score(
        solution_str,
        ground_truth,
        method=method,
        format_score=format_score,
        score=score,
    )
    is_correct = score_val > 0.0
    return {
        "score": score_val,
        "format_score": format_score if extracted else 0.0,
        "acc": is_correct,
        "extracted_gt": ground_truth,
        "pred": extracted,
    }
