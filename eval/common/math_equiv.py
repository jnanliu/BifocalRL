from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


_NORMALIZATION = NormalizationConfig(
    nits=False,
    malformed_operators=False,
    basic_latex=True,
    equations=True,
    boxed="all",
    units=True,
)

# When parsing a full completion, we usually want to anchor on `\boxed{...}` (like open-r1-intuitor).
_PRED_ANCHORED = LatexExtractionConfig(
    normalization_config=_NORMALIZATION,
    boxed_match_priority=0,
    try_extract_without_anchor=False,
)
# When we already extracted the boxed content (e.g. "\cot x"), there is no anchor left.
_PRED_UNANCHORED = LatexExtractionConfig(
    normalization_config=_NORMALIZATION,
    boxed_match_priority=0,
    try_extract_without_anchor=True,
)

_GOLD_ANCHORED = LatexExtractionConfig(try_extract_without_anchor=False)
_GOLD_UNANCHORED = LatexExtractionConfig(try_extract_without_anchor=True)


@dataclass(frozen=True)
class MathEquivResult:
    is_correct: Optional[bool]
    gold_extracted: str
    pred_extracted: str
    error: Optional[str] = None


def _canon_text(s: str) -> str:
    """
    Canonicalize a short math string for fallback equality checks.
    This is intentionally conservative: only whitespace / wrapper / punctuation cleanup.
    """
    if s is None:
        return ""
    t = str(s).strip()
    # Remove common math wrappers
    for a, b in (("\\(", "\\)"), ("\\[", "\\]")):
        if t.startswith(a) and t.endswith(b):
            t = t[len(a) : -len(b)].strip()
    if t.startswith("$") and t.endswith("$"):
        t = t[1:-1].strip()
    # Remove trailing punctuation often present in solutions
    t = t.rstrip().rstrip(".").rstrip()
    # Collapse whitespace
    t = " ".join(t.split())
    # Remove spaces around common operators
    t = t.replace(" - ", "-").replace(" + ", "+").replace(" * ", "*").replace(" / ", "/")
    return t


def extract_last_boxed(text: str) -> Optional[str]:
    """
    Extract the *content* of the last \\boxed{...} (supports nested braces).
    Returns None if not found.
    """
    if text is None:
        return None
    s = str(text)
    key = r"\boxed{"
    start = s.rfind(key)
    if start == -1:
        # Some datasets escape backslashes in JSON.
        key2 = r"\\boxed{"
        start = s.rfind(key2)
        if start == -1:
            return None
        start = start + len(key2) - 1  # normalize to the "\" of \boxed{
        key = r"\boxed{"

    # Position after the opening brace
    i = start + len(key)
    depth = 1
    out_chars = []
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            out_chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out_chars).strip()
            out_chars.append(ch)
        else:
            out_chars.append(ch)
        i += 1
    return None


def extract_math_final_answer(text: str) -> str:
    """
    Our system_prompt asks the model to put the final answer in \\boxed{...}.
    For evaluation robustness, extract that if present, otherwise fall back to raw string.
    """
    if text is None:
        return ""
    s = str(text).strip()
    boxed = extract_last_boxed(s)
    return boxed if boxed is not None else s


def _safe_parse(text: str, *, is_gold: bool) -> Tuple[str, list]:
    if text is None:
        return "", []
    text = str(text)
    extracted = extract_math_final_answer(text)
    had_boxed = extracted.strip() != text.strip()
    try:
        if is_gold:
            # If we extracted from \boxed{...}, parse the extracted string without requiring anchors.
            cfg = [_GOLD_UNANCHORED] if had_boxed else [_GOLD_ANCHORED, _GOLD_UNANCHORED]
            parsed = parse(extracted, extraction_mode="first_match", extraction_config=cfg)
        else:
            cfg = [_PRED_UNANCHORED] if had_boxed else [_PRED_ANCHORED, _PRED_UNANCHORED]
            parsed = parse(extracted, extraction_mode="first_match", extraction_config=cfg)
        return extracted, parsed
    except Exception:
        return extracted or text, []


def math_equiv(gold: str, pred: str) -> MathEquivResult:
    """
    Compare `pred` and `gold` using math-verify (LaTeX parsing + equivalence).

    Returns:
      - is_correct: True/False if verifiable, None if parsing failed on either side.
    """
    gold_extracted, gold_parsed = _safe_parse(gold, is_gold=True)
    pred_extracted, pred_parsed = _safe_parse(pred, is_gold=False)

    if len(gold_parsed) == 0 or len(pred_parsed) == 0:
        # Fallback: for symbolic/formula answers (e.g. "p - q") math-verify may fail to parse/verify.
        # If the extracted strings match after light canonicalization, treat as correct.
        if _canon_text(gold_extracted) != "" and _canon_text(gold_extracted) == _canon_text(pred_extracted):
            return MathEquivResult(
                is_correct=True,
                gold_extracted=gold_extracted,
                pred_extracted=pred_extracted,
                error="string_match_fallback",
            )
        return MathEquivResult(
            is_correct=None,
            gold_extracted=gold_extracted,
            pred_extracted=pred_extracted,
            error="parse_failed",
        )

    try:
        ok = bool(verify(gold_parsed, pred_parsed))
        return MathEquivResult(
            is_correct=ok,
            gold_extracted=gold_extracted,
            pred_extracted=pred_extracted,
            error=None,
        )
    except Exception as e:
        return MathEquivResult(
            is_correct=None,
            gold_extracted=gold_extracted,
            pred_extracted=pred_extracted,
            error=f"verify_failed: {type(e).__name__}: {e}",
        )

