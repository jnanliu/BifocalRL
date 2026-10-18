"""
Scorer abstraction for pluggable evaluation backends.

Usage::

    scorer = get_scorer("default")          # co-learning's math_verify-based scorer
    scorer = get_scorer("hacrl_prime_math")  # HACRL's sympy-based scorer
    result = scorer.score(prediction, ground_truth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreResult:
    """Unified result returned by every scorer."""

    is_correct: Optional[bool]
    score: float
    pred_extracted: str = ""
    gold_extracted: str = ""
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


class ScorerName(str, Enum):
    """All available scoring backends."""

    DEFAULT = "default"
    HACRL_MATH_DAPO = "hacrl_math_dapo"
    HACRL_MATH_BOXED = "hacrl_math_boxed"
    HACRL_PRIME_MATH = "hacrl_prime_math"
    HACRL_GSM8K = "hacrl_gsm8k"
    HACRL_GSM8K_FLEXIBLE = "hacrl_gsm8k_flexible"
    HACRL_QA_EM = "hacrl_qa_em"
    TTRL_MATH = "ttrl_math"
    TTRL_GSM8K = "ttrl_gsm8k"
    TTRL_GSM8K_FLEXIBLE = "ttrl_gsm8k_flexible"
    TTRL_GPQA = "ttrl_gpqa"

    @classmethod
    def choices(cls) -> list[str]:
        return [e.value for e in cls]


# ---------------------------------------------------------------------------
# Scorer implementations
# ---------------------------------------------------------------------------

class _DefaultMathScorer:
    """co-learning's existing math_verify-based scorer."""

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.common.math_equiv import math_equiv, extract_math_final_answer

        gold_for_check = extract_math_final_answer(ground_truth)
        res = math_equiv(gold_for_check, prediction)
        return ScoreResult(
            is_correct=res.is_correct,
            score=1.0 if res.is_correct else 0.0,
            pred_extracted=res.pred_extracted,
            gold_extracted=res.gold_extracted,
            error=res.error,
        )


class _DefaultGsm8kScorer:
    """co-learning's existing GSM8K scorer."""

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.common.gsm8k import extract_gsm8k_final_answer
        from eval.common.math_equiv import math_equiv

        gold_for_check = extract_gsm8k_final_answer(ground_truth)
        res = math_equiv(gold_for_check, prediction)
        return ScoreResult(
            is_correct=res.is_correct,
            score=1.0 if res.is_correct else 0.0,
            pred_extracted=res.pred_extracted,
            gold_extracted=res.gold_extracted,
            error=res.error,
        )


class _HacrlMathDapoScorer:
    """HACRL DAPO/AIME-style math scorer."""

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.hacrl.math_dapo import compute_score
        from eval.common.math_equiv import extract_math_final_answer

        gold_extracted = extract_math_final_answer(ground_truth)
        res = compute_score(prediction, gold_extracted)
        return ScoreResult(
            is_correct=res["acc"],
            score=res["score"],
            pred_extracted=res.get("pred", ""),
            gold_extracted=gold_extracted,
        )


class _HacrlMathBoxedScorer:
    """HACRL simple boxed + string-normalization scorer."""

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.hacrl.math_boxed import compute_score, last_boxed_only_string, remove_boxed
        from eval.common.math_equiv import extract_math_final_answer

        gold_extracted = extract_math_final_answer(ground_truth)
        score_val = compute_score(prediction, gold_extracted)
        pred_extracted = ""
        try:
            boxed = last_boxed_only_string(prediction)
            if boxed:
                pred_extracted = remove_boxed(boxed)
        except Exception:
            pass
        return ScoreResult(
            is_correct=score_val == 1.0,
            score=score_val,
            pred_extracted=pred_extracted,
            gold_extracted=gold_extracted,
        )


class _HacrlPrimeMathScorer:
    """HACRL comprehensive sympy-based math scorer."""

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.hacrl.prime_math import compute_score
        res = compute_score(prediction, ground_truth)
        return ScoreResult(
            is_correct=res["acc"],
            score=res["score"],
            pred_extracted=res.get("pred", ""),
            gold_extracted=res.get("gold_extracted", ground_truth),
        )


class _HacrlGsm8kScorer:
    """HACRL strict ``#### number`` GSM8K scorer.

    This mirrors HACRL's original ``verl.utils.reward_score.gsm8k`` logic:
    clip to the last 300 chars, extract the final ``#### <number>``, then do
    an exact string match against the gold answer. No ``\\boxed{}`` fallback.
    """

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.hacrl.gsm8k import compute_score, extract_solution
        from eval.common.gsm8k import extract_gsm8k_final_answer

        gold_extracted = extract_gsm8k_final_answer(ground_truth).replace(",", "").replace("$", "")
        score_val = compute_score(prediction, gold_extracted)
        extracted = extract_solution(prediction) or ""

        return ScoreResult(
            is_correct=score_val > 0,
            score=score_val,
            pred_extracted=extracted,
            gold_extracted=gold_extracted,
        )


class _HacrlGsm8kFlexibleScorer:
    """HACRL flexible GSM8K scorer — extracts the last number from output.

    Uses ``method="flexible"`` which finds the last numeric value in the
    response instead of requiring the ``#### <number>`` format.  Useful
    when models output correct numbers without the ``####`` prefix.
    """

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.hacrl.gsm8k import compute_score, extract_solution
        from eval.common.gsm8k import extract_gsm8k_final_answer

        gold_extracted = extract_gsm8k_final_answer(ground_truth).replace(",", "").replace("$", "")
        score_val = compute_score(prediction, gold_extracted, method="flexible")
        extracted = extract_solution(prediction, method="flexible") or ""

        return ScoreResult(
            is_correct=score_val > 0,
            score=score_val,
            pred_extracted=extracted,
            gold_extracted=gold_extracted,
        )


class _HacrlQaEmScorer:
    """HACRL QA exact-match scorer."""

    def score(self, prediction: str, ground_truth: Any) -> ScoreResult:
        from eval.hacrl.qa_em import compute_score, extract_solution
        score_val = compute_score(prediction, ground_truth)
        extracted = extract_solution(prediction) or ""
        if isinstance(ground_truth, dict):
            gold_str = str(ground_truth.get("target", ground_truth))
        else:
            gold_str = str(ground_truth)
        return ScoreResult(
            is_correct=score_val > 0,
            score=score_val,
            pred_extracted=extracted,
            gold_extracted=gold_str,
        )


class _TtrlMathScorer:
    """TTRL three-tier math scorer (mathd -> sympy -> optional math_verify)."""

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.ttrl.math import compute_score
        res = compute_score(prediction, ground_truth)
        return ScoreResult(
            is_correct=res["acc"],
            score=res["score"],
            pred_extracted=res.get("pred", ""),
            gold_extracted=str(res.get("extracted_gt", "")),
        )


class _TtrlGsm8kScorer:
    """TTRL-style GSM8K scorer — pure ``#### number`` extraction.

    Mirrors ``TTRL/verl/verl/utils/reward_score/gsm8k.py``: extracts the last
    ``#### <number>`` from the model response and does an exact string match
    against the gold answer.  No ``\\boxed{}`` fallback.
    """

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.common.gsm8k import extract_gsm8k_final_answer
        from eval.ttrl.gsm8k import compute_score, extract_solution

        gold_extracted = extract_gsm8k_final_answer(ground_truth).replace(",", "").replace("$", "")
        score_val = compute_score(prediction, gold_extracted)
        extracted = extract_solution(prediction) or ""

        return ScoreResult(
            is_correct=score_val > 0,
            score=score_val,
            pred_extracted=extracted,
            gold_extracted=gold_extracted,
        )


class _TtrlGsm8kFlexibleScorer:
    """TTRL-style flexible GSM8K scorer — extracts the last number from output.

    Like ``_TtrlGsm8kScorer`` but uses ``method="flexible"``: finds the last
    numeric value in the response instead of requiring ``#### <number>``.
    No ``\\boxed{}`` fallback.
    """

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.common.gsm8k import extract_gsm8k_final_answer
        from eval.ttrl.gsm8k import compute_score, extract_solution

        gold_extracted = extract_gsm8k_final_answer(ground_truth).replace(",", "").replace("$", "")
        score_val = compute_score(prediction, gold_extracted, method="flexible")
        extracted = extract_solution(prediction, method="flexible") or ""

        return ScoreResult(
            is_correct=score_val > 0,
            score=score_val,
            pred_extracted=extracted,
            gold_extracted=gold_extracted,
        )


class _TtrlGpqaScorer:
    """TTRL GPQA letter-match scorer (Answer: A/B/C/D)."""

    def score(self, prediction: str, ground_truth: str) -> ScoreResult:
        from eval.ttrl.gpqa import gpqa_compute_score
        res = gpqa_compute_score(prediction, ground_truth)
        return ScoreResult(
            is_correct=res["acc"],
            score=res["score"],
            pred_extracted=res.get("pred", ""),
            gold_extracted=str(res.get("extracted_gt", "")),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MATH_SCORERS = {
    ScorerName.DEFAULT: _DefaultMathScorer,
    ScorerName.HACRL_MATH_DAPO: _HacrlMathDapoScorer,
    ScorerName.HACRL_MATH_BOXED: _HacrlMathBoxedScorer,
    ScorerName.HACRL_PRIME_MATH: _HacrlPrimeMathScorer,
    ScorerName.TTRL_MATH: _TtrlMathScorer,
}

_GSM8K_SCORERS = {
    ScorerName.DEFAULT: _DefaultGsm8kScorer,
    ScorerName.HACRL_GSM8K: _HacrlGsm8kScorer,
    ScorerName.HACRL_GSM8K_FLEXIBLE: _HacrlGsm8kFlexibleScorer,
    ScorerName.TTRL_GSM8K: _TtrlGsm8kScorer,
    ScorerName.TTRL_GSM8K_FLEXIBLE: _TtrlGsm8kFlexibleScorer,
}

_TTRL_SCORERS = {
    ScorerName.TTRL_MATH: _TtrlMathScorer,
    ScorerName.TTRL_GSM8K: _TtrlGsm8kScorer,
    ScorerName.TTRL_GSM8K_FLEXIBLE: _TtrlGsm8kFlexibleScorer,
    ScorerName.TTRL_GPQA: _TtrlGpqaScorer,
}

_QA_SCORERS = {
    ScorerName.HACRL_QA_EM: _HacrlQaEmScorer,
}

_GPQA_SCORERS = {
    ScorerName.TTRL_GPQA: _TtrlGpqaScorer,
}

_ALL_SCORERS = {**_MATH_SCORERS, **_GSM8K_SCORERS, **_QA_SCORERS, **_GPQA_SCORERS}


def get_scorer(name: str | ScorerName, dataset_hint: str | None = None):
    """Return a scorer instance.

    When *name* is ``"default"`` the actual scorer depends on *dataset_hint*
    (``"gsm8k"`` → GSM8K extraction, everything else → math_verify).

    For HACRL math scorers evaluating GSM8K, we automatically route to the
    GSM8K-specific ``#### number`` extraction scorer to match the original
    HACRL evaluation behaviour (``verl/utils/reward_score/gsm8k.py``).

    For GPQA datasets, automatically route to ``_TtrlGpqaScorer`` unless an
    explicit GPQA scorer is requested.
    """
    if isinstance(name, str):
        name = ScorerName(name)

    # GPQA routing: letter-match scorer for multiple-choice datasets.
    if dataset_hint and "gpqa" in dataset_hint:
        if name in (ScorerName.TTRL_GPQA,):
            return _TtrlGpqaScorer()
        # Auto-route math scorers to GPQA for gpqa datasets.
        if name in _MATH_SCORERS or name == ScorerName.DEFAULT:
            return _TtrlGpqaScorer()

    if name == ScorerName.DEFAULT:
        if dataset_hint and "gsm8k" in dataset_hint:
            return _DefaultGsm8kScorer()
        return _DefaultMathScorer()

    # Explicit flexible scorers — honour them directly, no re-routing.
    if name in (ScorerName.HACRL_GSM8K_FLEXIBLE, ScorerName.TTRL_GSM8K_FLEXIBLE):
        return _ALL_SCORERS[name]()

    # # TTRL math scorer + gsm8k → strict GSM8K extraction (#### <number>).
    # if dataset_hint and "gsm8k" in dataset_hint and name == ScorerName.TTRL_MATH:
    #     return _TtrlGsm8kScorer()

    # # HACRL dispatches GSM8K to the strict ``####`` scorer regardless of the
    # # math scorer selected elsewhere, so route math scorers accordingly.
    # if dataset_hint and "gsm8k" in dataset_hint and name in _MATH_SCORERS:
    #     return _HacrlGsm8kScorer()

    if name in _ALL_SCORERS:
        return _ALL_SCORERS[name]()

    raise ValueError(
        f"Unknown scorer {name!r}. Available: {ScorerName.choices()}"
    )


def list_scorers() -> list[str]:
    """Return the list of all available scorer names."""
    return ScorerName.choices()
