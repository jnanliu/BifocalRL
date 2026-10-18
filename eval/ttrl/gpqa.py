"""GPQA letter-match scorer (ported from TTRL recipe/r1/tasks/gpqa.py)."""

from __future__ import annotations

import re
from typing import Any, Dict

_ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def gpqa_compute_score(solution_str: str, ground_truth: str) -> Dict[str, Any]:
    """Score a GPQA response by extracting Answer: X letter."""
    match = re.search(_ANSWER_PATTERN_MULTICHOICE, solution_str)
    extracted = match.group(1).upper() if match else ""
    gt = str(ground_truth).strip().upper()
    is_correct = extracted == gt and extracted != ""
    return {
        "score": 1.0 if is_correct else 0.0,
        "format_score": 1.0 if extracted else 0.0,
        "acc": is_correct,
        "extracted_gt": gt,
        "pred": extracted,
    }
