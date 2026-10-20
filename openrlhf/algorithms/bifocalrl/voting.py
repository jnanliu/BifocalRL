"""Answer extraction, MV/WMV pseudo-labels and grouped voting rewards."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .legacy.math import grade
from .legacy.math_extraction import extract_answer, simplify_expression_string

_logger = logging.getLogger(__name__)


def extract_answers(responses: Sequence[Optional[str]]) -> List[Optional[str]]:
    """Extract final answers from raw responses via \\boxed{} extraction.

    Uses TTRL's extract_answer (last \\boxed{} with robust nested-brace handling).
    Returns None if no \\boxed{} is found or the extracted answer is empty.
    """
    out: List[Optional[str]] = []
    for r in responses:
        if r is None:
            out.append(None)
            continue
        ans = extract_answer(str(r))
        if ans is None or ans.strip() == "":
            out.append(None)
        else:
            out.append(ans.strip())
    return out


def _simplify_answer(answer: Optional[str]) -> Optional[str]:
    """Simplify an extracted answer to a canonical string form for counting."""
    if answer is None:
        return None
    try:
        return simplify_expression_string(answer)
    except Exception:
        return answer


def cluster_voting_rewards(answers, *, format_weight=0.1):
    """Cluster share among parseable boxed answers, plus the format auxiliary."""
    representatives, ids, counts = [], [], []
    for answer in answers:
        if answer is None:
            ids.append(None)
            continue
        cluster = next((i for i, other in enumerate(representatives) if is_math_equivalent(answer, other)), None)
        if cluster is None:
            cluster = len(representatives)
            representatives.append(answer)
            counts.append(0)
        counts[cluster] += 1
        ids.append(cluster)
    parseable = sum(counts)
    return [counts[i] / parseable + format_weight if i is not None else 0.0 for i in ids]


def is_math_equivalent(a: Optional[str], b: Optional[str]) -> bool:
    """Math equivalence check using TTRL's grade function.

    Uses TTRL's three-tier comparison:
    grade_answer_mathd -> grade_answer_sympy -> optional is_latex_equal.
    """
    if a is None or b is None:
        return False
    try:
        return grade(str(a), str(b), fast=True)
    except Exception:
        return str(a).strip() == str(b).strip()


@dataclass(frozen=True)
class WMVResult:
    selected_answer: Optional[str]
    answer_clusters: Dict[str, float]


def weighted_majority_voting(
    answers: Sequence[Optional[str]],
    weights: Sequence[float],
    *,
    rng: Optional[random.Random] = None,
) -> WMVResult:
    """
    Weighted majority voting with simplification-based grouping (TTRL-style).

    - Ignores None answers.
    - Simplifies each answer to a canonical form via sympy/latex2sympy.
    - Groups by simplified string and sums weights per group.
    - Selects the max-weight group; ties broken randomly (seedable via `rng`).
    """
    if len(answers) != len(weights):
        raise ValueError(f"answers length ({len(answers)}) != weights length ({len(weights)})")

    rng = rng or random

    # simplified_key -> (representative original answer, total weight)
    groups: Dict[str, Tuple[str, float]] = {}
    for ans, w in zip(answers, weights):
        if ans is None:
            continue
        ans_s = str(ans)
        key = _simplify_answer(ans_s)
        if key is None:
            key = ans_s
        if key in groups:
            rep, total_w = groups[key]
            groups[key] = (rep, total_w + float(w))
        else:
            groups[key] = (ans_s, float(w))

    if not groups:
        return WMVResult(selected_answer=None, answer_clusters={})

    max_w = max(w for _, w in groups.values())
    eps = 1e-12
    candidates = [rep for key, (rep, w) in groups.items() if abs(w - max_w) < eps]
    selected = rng.choice(candidates) if candidates else None

    answer_clusters = {rep: w for key, (rep, w) in groups.items()}
    return WMVResult(selected_answer=selected, answer_clusters=answer_clusters)


def rewards_from_wmv_pseudolabel(
    responses: Sequence[Optional[str]],
    weights: Sequence[float],
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[float], WMVResult]:
    """
    Run WMV to select a pseudo-label, then assign binary rewards:
    - 1.0 if response answer is math-equivalent to pseudo-label
    - 0.0 otherwise (including when pseudo-label is None)
    """
    answers = extract_answers(responses)
    vote = weighted_majority_voting(answers, weights, rng=rng)

    rewards: List[float] = []
    for ans in answers:
        if vote.selected_answer is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if is_math_equivalent(ans, vote.selected_answer) else 0.0)
    return rewards, vote


def rewards_from_mv_pseudolabel(
    responses: Sequence[Optional[str]],
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[float], WMVResult]:
    """
    Run unweighted majority voting to select a pseudo-label, then assign binary rewards.

    Each non-None answer counts as one vote (weight=1.0). This is equivalent to WMV
    with uniform weights, but the interface does not require callers to supply weights.
    """
    uniform_weights = [1.0] * len(responses)
    return rewards_from_wmv_pseudolabel(responses, uniform_weights, rng=rng)


@dataclass(frozen=True)
class GroupedVoteResult:
    rewards: List[List[float]]
    votes: List[WMVResult]
    extracted_answers: List[List[Optional[str]]]


def vote_response_groups(
    responses: Sequence[Optional[str]],
    num_prompts: int,
    num_generations: int,
    *,
    step: int,
    weights: Optional[Sequence[Sequence[float]]] = None,
    include_answers: bool = False,
) -> GroupedVoteResult:
    """Vote once per query with the historical step/query tie-breaking seed.

    The same grouping and RNG protocol is used by MV, WMV, gated Intuitor
    and mixed reward. Missing answers remain excluded from vote selection.
    """
    if num_generations <= 0 or num_prompts < 0 or len(responses) != num_prompts * num_generations:
        raise ValueError("Responses must contain num_prompts * num_generations entries")
    if weights is not None and len(weights) != num_prompts:
        raise ValueError("Voting weights must contain one row per prompt")
    rewards, votes, answers = [], [], []
    for index in range(num_prompts):
        group = responses[index * num_generations : (index + 1) * num_generations]
        rng = random.Random((int(step) + 1) * 1_000_000 + index)
        if weights is None:
            group_rewards, vote = rewards_from_mv_pseudolabel(group, rng=rng)
        else:
            group_rewards, vote = rewards_from_wmv_pseudolabel(group, weights[index], rng=rng)
        rewards.append(group_rewards)
        votes.append(vote)
        if include_answers:
            answers.append(extract_answers(group))
    return GroupedVoteResult(rewards, votes, answers)
