# Copyright 2024 PRIME team and/or its affiliates
# Adapted from TTRL (https://arxiv.org/abs/2504.16084) evaluation code.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TTRL math answer grading with high recall.

Three-tier grading pipeline:
  1. grade_answer_mathd  – Dan Hendrycks string normalization + exact match
  2. grade_answer_sympy  – full normalization + sympy symbolic comparison
  3. is_latex_equal      – math_verify fallback (optional, fast=False only)

Also provides ``simplify_expression_string`` used by majority voting to
canonicalize answers before counting.
"""

from __future__ import annotations

import os
import re
import signal
from itertools import islice, zip_longest
from math import isclose
from typing import Any, Dict, Optional

import sympy
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.sympy_parser import parse_expr

_LATEX2SYMPY_AVAILABLE = False
try:
    from latex2sympy2_extended import latex2sympy
    _LATEX2SYMPY_AVAILABLE = True
except ImportError:
    latex2sympy = None

_PYLATEXENC_AVAILABLE = False
try:
    from pylatexenc import latex2text
    _PYLATEXENC_AVAILABLE = True
except ImportError:
    latex2text = None

_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    pass

_SYMPY_PARSE_LATEX_AVAILABLE = False
try:
    from sympy.parsing.latex import parse_latex
    _SYMPY_PARSE_LATEX_AVAILABLE = True
except ImportError:
    parse_latex = None


# ---------------------------------------------------------------------------
# Timeout decorator (POSIX only)
# ---------------------------------------------------------------------------

def _timeout_decorator(timeout_seconds: int = 8):
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


class _TimeoutCtx:
    """Context-manager based timeout (POSIX only)."""

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if os.name == "posix":
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.name == "posix":
            signal.alarm(0)


# ---------------------------------------------------------------------------
# Unit text lists (from MathQA / Dan Hendrycks)
# ---------------------------------------------------------------------------

_UNIT_TEXTS = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m",
    "deg", "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd",
    "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab",
    "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .",
    "t", "a", "h", "no change", "men", "soldier", "pie", "bc", "excess",
    "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre",
    "rise", "a . m", "th", "π r 2", "sq", "mark", "l", "toy", "coin",
    "sq . m", "gallon", "° f", "profit", "minw", "yr", "women", "feet",
    "am", "pm", "hr", "cu cm", "square", "v â € ™", "are", "rupee",
    "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day",
    "hour", "minute", "min", "second", "man", "woman", "sec", "cube", "mt",
    "sq inch", "mp", "∏ cm ³", "hectare", "more", "sec", "unit", "cu . m",
    "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm",
    "apple", "liter", "loss", "yard", "pure", "year", "increase", "decrease",
    "d", "less", "Surface", "litre", "pi sq m", "s .", "metre", "meter",
    "inch",
]
_UNIT_TEXTS_FULL = _UNIT_TEXTS + [t + "s" for t in _UNIT_TEXTS]


# ---------------------------------------------------------------------------
# String normalization: Dan Hendrycks / mathd style
# ---------------------------------------------------------------------------

def _mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    def _fix_fracs(s):
        substrs = s.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            for substr in substrs[1:]:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return s
                    a, b = substr[0], substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            new_str += "{" + a + "}{" + b + "}" + substr[2:]
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            new_str += "{" + a + "}" + b + substr[2:]
                        else:
                            new_str += "{" + a + "}" + b
        return new_str

    def _fix_a_slash_b(s):
        if len(s.split("/")) != 2:
            return s
        a_str, b_str = s.split("/")
        try:
            a_int, b_int = int(a_str), int(b_str)
            assert s == "{}/{}".format(a_int, b_int)
            return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
        except Exception:
            return s

    def _remove_right_units(s):
        if "\\text{ " in s:
            splits = s.split("\\text{ ")
            if len(splits) == 2:
                return splits[0]
        return s

    def _fix_sqrt(s):
        if "\\sqrt" not in s:
            return s
        splits = s.split("\\sqrt")
        new_string = splits[0]
        for sp in splits[1:]:
            if sp[0] != "{":
                new_string += "\\sqrt{" + sp[0] + "}" + sp[1:]
            else:
                new_string += "\\sqrt" + sp
        return new_string

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")

    # matrix normalization
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    for _ in range(2):
        for unit_text in _UNIT_TEXTS_FULL:
            _string = re.sub(r"(^|\W)" + re.escape(unit_text) + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


# ---------------------------------------------------------------------------
# Normalization constants (Minerva-style)
# ---------------------------------------------------------------------------

_SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

_REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
    "\\dots",
]


def _normalize_final_answer(final_answer: str) -> str:
    for before, after in _SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in _REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


# ---------------------------------------------------------------------------
# Repetitiveness check (suffix array LCP)
# ---------------------------------------------------------------------------

def _repeatness(s: str):
    def ranks(seq):
        index = {v: i for i, v in enumerate(sorted(set(seq)))}
        return [index[v] for v in seq]

    def suffix_array(seq):
        line = ranks(seq)
        n, k, ans, sa = len(seq), 1, line, [0] * len(seq)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k_val in enumerate(ans):
            sa[k_val] = i
        return ans, sa

    def lcp(arr, sa, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0
        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue
            j = sa[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1
            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1
        return ans

    arr = [ord(c) for c in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffix_array(arr)
    cnt = sum(lcp(arr, sa, c))
    return (cnt * 2 / (n * (n + 1))) > 0.2


# ---------------------------------------------------------------------------
# Helper parsers
# ---------------------------------------------------------------------------

def _latex_eval(latex_str):
    if not _SYMPY_PARSE_LATEX_AVAILABLE:
        raise ImportError("sympy.parsing.latex.parse_latex not available")
    sym = parse_latex(latex_str)
    val = sym.evalf()
    return sym, val


def _numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def _parse_latex_to_text(expr: str) -> str:
    if not _PYLATEXENC_AVAILABLE:
        return expr
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    return int(float(x))


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile(r"([0-9]) +([0-9])")
    return p1.sub(r"\1+\2", step)


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


# ---------------------------------------------------------------------------
# Full normalization (_normalize)
# ---------------------------------------------------------------------------

def _normalize(expr: str) -> Optional[str]:
    if expr is None:
        return None
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex_to_text(expr)
        except Exception:
            pass

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


# ---------------------------------------------------------------------------
# Sympy helpers
# ---------------------------------------------------------------------------

_BAD_SUBSTRINGS = ["^{", "^("]
_BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
_TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _count_unknown_letters(expr: str):
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len(set(x for x in expr if x.isalpha()))


def _should_allow_eval(expr: str):
    if _count_unknown_letters(expr) > 2:
        return False
    for bad in _BAD_SUBSTRINGS:
        if bad in expr:
            return False
    for bad_re in _BAD_REGEXES:
        if re.search(bad_re, expr) is not None:
            return False
    return True


@_timeout_decorator(timeout_seconds=5)
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def _split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in _TUPLE_CHARS
        and expr[-1] in _TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in _TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


# ---------------------------------------------------------------------------
# Boxed extraction
# ---------------------------------------------------------------------------

from eval.common.math_extraction import (  # noqa: E402
    extract_answer,
    extract_boxed_answer,
    last_boxed_only_string,
    remove_boxed,
)


# ---------------------------------------------------------------------------
# Grading functions
# ---------------------------------------------------------------------------

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    """Hendrycks-style normalization + exact match."""
    gt_norm = _mathd_normalize_answer(ground_truth)
    given_norm = _mathd_normalize_answer(given_answer)
    if gt_norm is None or given_norm is None:
        return False
    return gt_norm == given_norm


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """Full normalization + sympy symbolic comparison."""
    gt_norm = _normalize(ground_truth)
    given_norm = _normalize(given_answer)

    if gt_norm is None:
        return False
    if gt_norm == given_norm:
        return True
    if len(given_norm) == 0:
        return False

    gt_elems = _split_tuple(gt_norm)
    given_elems = _split_tuple(given_norm)

    if len(gt_elems) > 1 and (
        gt_norm[0] != given_norm[0] or gt_norm[-1] != given_norm[-1]
    ):
        return False
    if len(gt_elems) != len(given_elems):
        return False

    for gt_elem, given_elem in zip(gt_elems, given_elems):
        if _is_frac(gt_elem) and _is_frac(given_elem):
            is_correct = gt_elem == given_elem
        elif _str_is_int(gt_elem) != _str_is_int(given_elem):
            is_correct = False
        else:
            is_correct = are_equal_under_sympy(gt_elem, given_elem)
        if not is_correct:
            return False
    return True


def is_latex_equal(given_answer: str, ground_truth: str) -> bool:
    """math_verify fallback comparison (requires math_verify + latex2sympy2_extended)."""
    if not _MATH_VERIFY_AVAILABLE:
        return False
    try:
        with _TimeoutCtx(1):
            try:
                if (len(given_answer) > 128 and _repeatness(given_answer)) or (
                    len(ground_truth) > 128 and _repeatness(ground_truth)
                ):
                    return False
                gt_norm = _normalize(ground_truth)
                given_norm = _normalize(given_answer)
                if gt_norm is None:
                    return False
                if gt_norm == given_norm:
                    return True

                ga = given_answer.replace("\n", "")
                gt = ground_truth.replace("\n", "")
                if "$" not in ga:
                    ga = f"${ga}$"
                if "$" not in gt:
                    gt = f"${gt}$"
                return verify(
                    parse(
                        gt,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    parse(
                        ga,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    timeout_seconds=1,
                )
            except Exception:
                return False
    except TimeoutError:
        return False


def grade(model_answer: str, gt_answer: str, fast: bool = True) -> bool:
    """Three-tier grading: mathd -> sympy -> optional math_verify."""
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer) or gt_answer
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast and not correct:
        correct = is_latex_equal(model_answer, gt_answer)
    return correct


# ---------------------------------------------------------------------------
# compute_score: full evaluation pipeline
# ---------------------------------------------------------------------------

def compute_score(
    model_response: str,
    gt_answer,
    fast: bool = False,
) -> Dict[str, Any]:
    """Score a model response against ground truth.

    Returns dict with keys: score, format_score, acc, extracted_gt, pred.
    """
    model_answer = extract_answer(model_response)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }

    if isinstance(gt_answer, (float, int)):
        gt_answer = str(gt_answer)

    is_correct = False
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        for gt in gt_answer:
            if grade(model_answer, gt, fast):
                is_correct = True
                break

    return {
        "score": 1.0 if is_correct else 0.0,
        "format_score": 1.0,
        "acc": is_correct,
        "extracted_gt": gt_answer,
        "pred": model_answer,
    }


# ---------------------------------------------------------------------------
# simplify_expression_string: used by majority voting
# ---------------------------------------------------------------------------

from eval.common.math_extraction import simplify_expression_string  # noqa: E402, F811


