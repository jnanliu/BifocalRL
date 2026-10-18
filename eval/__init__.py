"""
Evaluation package for Co-Intuitor.

Four independent backends, each toggleable via experiment JSON config:

- ``eval.hacrl``     -- HACRL-derived scorers (sympy, Minerva, boxed, QA)
- ``eval.ttrl``      -- TTRL three-tier math scoring + GPQA
- ``eval.lighteval`` -- External LightEval CLI wrapper
- ``eval.zeroeval``  -- ZeroEval / CRUXEval benchmark

Shared scoring primitives live in ``eval.common``.
The unified interface is defined in ``eval.base``.
"""

from eval.base import BaseEvaluator, EvalBackendConfig, EvalResult

