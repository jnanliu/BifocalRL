"""
QA exact-match / substring-match scoring with <answer> tag extraction.

Ported from HACRL verl/utils/reward_score/search_r1_like_qa_em.py
"""

from __future__ import annotations

import re
import string


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction: str, golden_answers: str | list[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized = normalize_answer(prediction)
    return any(normalize_answer(g) == normalized for g in golden_answers)


def subem_check(prediction: str, golden_answers: str | list[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized = normalize_answer(prediction)
    return any(normalize_answer(g) in normalized for g in golden_answers)


def extract_solution(solution_str: str) -> str | None:
    """Extract the last <answer>...</answer> content."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_score(
    solution_str: str,
    ground_truth: dict | str | list[str],
    method: str = "em",
    score: float = 1.0,
) -> float:
    """Score QA-style answers.

    Args:
        solution_str: model output text.
        ground_truth: either a dict with key ``"target"`` (list of acceptable answers),
            a list of acceptable answers, or a single string answer.
        method: ``"em"`` for exact match, ``"subem"`` for substring match.
        score: reward for a correct answer.
    """
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0

    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", [])
    elif isinstance(ground_truth, list):
        targets = ground_truth
    else:
        targets = [ground_truth]

    checker = em_check if method == "em" else subem_check
    open_count = solution_str.count("<answer>")
    close_count = solution_str.count("</answer>")

    if checker(answer, targets):
        if open_count > 10 or close_count > 10:
            return score / 4
        return score
    return 0.0
