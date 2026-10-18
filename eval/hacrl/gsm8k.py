"""
HACRL-style GSM8K scoring with strict #### extraction.

Ported from HACRL verl/utils/reward_score/gsm8k.py
"""

from __future__ import annotations

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    assert method in ("strict", "flexible")

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if not solutions:
            return None
        return solutions[-1].replace(",", "").replace("$", "")
    else:
        answers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        if not answers:
            return None
        invalid = {"", "."}
        for final_answer in reversed(answers):
            if final_answer not in invalid:
                return final_answer
        return None


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Score GSM8K style answers.

    Returns *score* if the extracted answer matches ground_truth,
    *format_score* if extraction succeeded but answer differs,
    0 if extraction failed entirely.
    """
    answer = extract_solution(solution_str, method=method)
    if answer is None:
        return 0.0
    return score if answer == ground_truth else format_score
