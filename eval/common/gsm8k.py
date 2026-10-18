from __future__ import annotations

import re


_FINAL_SEP_RE = re.compile(r"(?:\#{3,4})\s*", re.MULTILINE)


def extract_gsm8k_final_answer(gold_answer: str) -> str:
    """
    GSM8K gold `answer` often contains reasoning + final answer separated by '####' (sometimes '###').
    We return the tail segment (stripped).
    """
    if gold_answer is None:
        return ""
    s = str(gold_answer).strip()

    # Prefer splitting by #### / ### if present.
    parts = _FINAL_SEP_RE.split(s)
    if len(parts) >= 2:
        return parts[-1].strip()

    # Fallback: take last number-ish token.
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[-1] if m else s

