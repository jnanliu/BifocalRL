"""
Shared math answer extraction and simplification helpers.

These functions are used by both evaluation backends (TTRL, HACRL) and
training utilities (majority voting), so they live in the common layer
to avoid circular dependencies.
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.exceptions import SymPyDeprecationWarning

_LATEX2SYMPY_AVAILABLE = False
try:
    from latex2sympy2_extended import latex2sympy
    _LATEX2SYMPY_AVAILABLE = True
except ImportError:
    latex2sympy = None


def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last ``\\boxed{...}`` or ``\\fbox{...}`` substring (including the command)."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx: right_brace_idx + 1]


def remove_boxed(s: str) -> Optional[str]:
    """Strip the ``\\boxed{`` prefix and trailing ``}``."""
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the answer from inside a LaTeX ``\\boxed{}`` command."""
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def extract_answer(passage: str) -> Optional[str]:
    """Extract answer from model output via ``\\boxed{}``."""
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def _looks_like_plain_sympy_expr(expression_string: str) -> bool:
    """Return True when the string is safe for ``parse_expr``.

    Curly braces commonly appear in model answers as LaTeX grouping, but
    ``parse_expr`` interprets them as Python set literals. That can produce
    ``FiniteSet`` bases inside powers and trigger repeated SymPy deprecation
    warnings, so we route those strings to the LaTeX fallback instead.
    """
    return "\\" not in expression_string and "{" not in expression_string and "}" not in expression_string


def _simplify_with_suppressed_pow_warning(sympy_expr) -> str:
    """Simplify while silencing the noisy non-Expr ``Pow`` deprecation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SymPyDeprecationWarning)
        return str(simplify(sympy_expr))


def _timeout_decorator(timeout_seconds: int = 10):
    import os
    import signal

    if os.name == "posix":
        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper
        return decorator
    else:
        def decorator(func):
            return func
        return decorator


@_timeout_decorator(timeout_seconds=10)
def simplify_expression_string(expression_string: str) -> str:
    """Simplify a math expression to a canonical string form for MV counting."""
    expression_string = expression_string.strip()
    if not expression_string:
        return expression_string

    try:
        if _looks_like_plain_sympy_expr(expression_string):
            sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)
            return _simplify_with_suppressed_pow_warning(sympy_expr)
    except TimeoutError:
        return expression_string
    except Exception:
        pass

    if _LATEX2SYMPY_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SymPyDeprecationWarning)
                sympy_expr = latex2sympy(expression_string)
            return _simplify_with_suppressed_pow_warning(sympy_expr)
        except TimeoutError:
            return expression_string
        except Exception:
            return expression_string
    return expression_string
