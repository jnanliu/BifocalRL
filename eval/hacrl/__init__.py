"""
HACRL evaluation backend.

Provides HACRL-derived scorers (sympy, Minerva, boxed, QA, etc.) and the main
evaluation runner for offline/online vLLM-based evaluation of math checkpoints.

Scorers:
  - hacrl_math_dapo:   Minerva-style normalization + boxed extraction (AIME/DAPO)
  - hacrl_math_boxed:  Simple \\boxed{} extraction + string normalization (is_equiv)
  - hacrl_prime_math:  Comprehensive sympy-based equivalence checking
  - hacrl_gsm8k:       Strict #### extraction with format scoring
  - hacrl_qa_em:       Exact-match / substring-match on <answer> tags
"""

from eval.hacrl.runner import EvalConfig, evaluate_cruxeval, evaluate_math_dataset, run_all
from eval.scorers import ScoreResult, ScorerName, get_scorer, list_scorers
