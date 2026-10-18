"""
Sympy-based comprehensive math equivalence checking.

Ported from HACRL verl/utils/reward_score/prime_math/
(combines __init__.py, grader.py, and math_normalize.py).

Dependencies: sympy, pylatexenc
"""

from __future__ import annotations

import contextlib
import math
import re
from math import isclose
from typing import Optional

from .timeout import timeout_limit

# ---------------------------------------------------------------------------
# math_normalize  (from HACRL prime_math/math_normalize.py)
# ---------------------------------------------------------------------------

def _fix_fracs_norm(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b_norm(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        ai, bi = int(a), int(b)
        assert string == f"{ai}/{bi}"
        return f"\\frac{{{ai}}}{{{bi}}}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units_norm(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt_norm(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units_norm(string)
    string = string.replace("\\%", "").replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt_norm(string)
    string = string.replace(" ", "")
    string = _fix_fracs_norm(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b_norm(string)
    return string


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


# ---------------------------------------------------------------------------
# grader  (from HACRL prime_math/grader.py)
# ---------------------------------------------------------------------------

def _is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num
        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def _normalize_grader(answer, pi) -> str:
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer))
        or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")
    answer = _handle_base(answer)
    answer = _handle_pi(answer, pi)
    return answer


def _handle_base(x):
    if isinstance(x, str) and "_" in x:
        x = x.split("_")[0]
        x = float(x)
        return int(x)
    return x


def _handle_pi(string, pi):
    if isinstance(string, str) and "\\pi" in string:
        idx = string.find("\\pi")
        while idx != -1:
            if idx > 0 and string[idx - 1].isdigit():
                string = string[:idx] + f"*{pi}" + string[idx + 3 :]
            else:
                string = string[:idx] + f"1*{pi}" + string[idx + 3 :]
            idx = string.find("\\pi", idx + 1)
        with contextlib.suppress(Exception):
            string = eval(string)  # noqa: S307
    return string


def _format_intervals(prediction: str) -> str:
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }
    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner = match.group(1)
            if key == "Interval(":
                return f"[{inner}]"
            elif key == "Interval.Ropen(":
                return f"[{inner})"
            elif key == "Interval.Lopen(":
                return f"({inner}]"
            elif key == "Interval.open(":
                return f"({inner})"
    return prediction


def _symbolic_equal(a, b, tolerance, timeout=10.0):
    try:
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False

    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                @timeout_limit(seconds=timeout)
                def _do_parse():
                    return f(s)
                return _do_parse()
            except TimeoutError:
                continue
            except Exception:
                continue
        return s

    a_parsed = _parse(a)
    b_parsed = _parse(b)

    try:
        @timeout_limit(seconds=timeout)
        def _simplify():
            return simplify(a_parsed - b_parsed) == 0
        if _simplify():
            return True
    except (TimeoutError, Exception):
        pass

    try:
        @timeout_limit(seconds=timeout)
        def _numerical():
            return isclose(N(a_parsed), N(b_parsed), rel_tol=tolerance)
        if _numerical():
            return True
    except (TimeoutError, Exception):
        pass

    return False


def math_equal(
    prediction,
    reference,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    pi: float = math.pi,
) -> bool:
    prediction = _normalize_grader(prediction, pi)
    reference = _normalize_grader(reference, pi)

    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]

    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:
        if _is_digit(prediction)[0] and _is_digit(reference)[0]:
            pred_val = _is_digit(prediction)[1]
            ref_val = _is_digit(reference)[1]
            gt_results = [ref_val / 100, ref_val, ref_val * 100] if include_percentage else [ref_val]
            for item in gt_results:
                try:
                    if isclose(item, pred_val, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()
    prediction = _format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    if (
        prediction and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(p, r, include_percentage, tolerance)
            for p, r in zip(pred_parts, ref_parts)
        ):
            return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]
        if len(pred_parts) == len(ref_parts):
            if all(math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance) for i in range(len(pred_parts))):
                return True

    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(p, r, include_percentage, tolerance)
            for p, r in zip(pred_parts, ref_parts)
        ):
            return True

    return _symbolic_equal(prediction, reference, tolerance, timeout)


# ---------------------------------------------------------------------------
# prime_math __init__  (from HACRL prime_math/__init__.py)
# ---------------------------------------------------------------------------

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    from sympy.parsing import sympy_parser
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex_text(expr: str) -> str:
    from pylatexenc import latex2text as l2t
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = l2t.LatexNodes2Text().latex_to_text(expr)
    for old, new in [("√", "sqrt"), ("π", "pi"), ("∞", "inf"), ("∪", "U"), ("·", "*"), ("×", "*")]:
        expr = expr.replace(old, new)
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
        return _is_int(float(x))
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    return int(float(x))


def _inject_implicit_mixed_number(step: str) -> str:
    return re.sub(r"([0-9]) +([0-9])", r"\1+\2", step)


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return expr


def _normalize_prime(expr: str) -> str:
    if expr is None:
        return None
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch",
        "yard", "liter",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex_text(expr)

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def _count_unknown_letters(expr: str) -> int:
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len(set(x for x in expr if x.isalpha()))


def _should_allow_eval(expr: str) -> bool:
    if _count_unknown_letters(expr) > 2:
        return False
    for bad in BAD_SUBSTRINGS:
        if bad in expr:
            return False
    return all(re.search(bad, expr) is None for bad in BAD_REGEXES)


@timeout_limit(seconds=10)
def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    import sympy
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            return sympy.simplify(sympy_diff) == 0
    except Exception:
        pass
    return False


def _split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if not expr:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """Grade via normalized string comparison + sympy simplification."""
    if given_answer is None:
        return False

    gt_norm_mathd = normalize_answer(ground_truth)
    ga_norm_mathd = normalize_answer(given_answer)
    if gt_norm_mathd == ga_norm_mathd:
        return True

    gt_norm = _normalize_prime(ground_truth)
    ga_norm = _normalize_prime(given_answer)
    if gt_norm is None:
        return False
    if gt_norm == ga_norm:
        return True
    if not ga_norm:
        return False

    gt_elems = _split_tuple(gt_norm)
    ga_elems = _split_tuple(ga_norm)

    if (
        len(gt_elems) > 1
        and (gt_norm[0] != ga_norm[0] or gt_norm[-1] != ga_norm[-1])
        or len(gt_elems) != len(ga_elems)
    ):
        return False

    for gt_e, ga_e in zip(gt_elems, ga_elems):
        if _is_frac(gt_e) and _is_frac(ga_e):
            is_correct = gt_e == ga_e
        elif _str_is_int(gt_e) != _str_is_int(ga_e):
            is_correct = False
        else:
            try:
                is_correct = _are_equal_under_sympy(gt_e, ga_e)
            except Exception:
                is_correct = False
        if not is_correct:
            return False
    return True


def _last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, left_brace_idx, right_brace_idx, num_open = idx, None, None, 0
    while i < len(string):
        if string[i] == "{":
            num_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_brace_idx = i
                break
        i += 1
    if left_brace_idx is None or right_brace_idx is None:
        return None
    return string[left_brace_idx + 1 : right_brace_idx].strip()


def _match_answer(response: str):
    is_matched = False
    for ans_marker in ["answer:", "answer is", "answers are"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[:ans_idx].strip()
            if response.endswith("\n"):
                response = response[:-2]

    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    if ". " in response:
        dot_idx = response.lower().rfind(". ")
        if dot_idx != -1:
            response = response[:dot_idx].strip()

    for ans_marker in ["be ", "is ", "are ", "=", ": ", "get ", "be\n", "is\n", "are\n", ":\n", "get\n"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    is_matched = is_matched if any(c.isdigit() for c in response) else False
    return is_matched, response


_GSM8K_ANSWER_RE = re.compile(r"####\s*(.+?)$", re.MULTILINE)


def _extract_gold(ground_truth: str) -> str:
    """Extract the actual answer from raw dataset gold text.

    Handles MATH-style ``\\boxed{answer}`` and GSM8K-style ``#### answer``.
    If neither pattern is found, returns the string unchanged (already extracted).
    """
    boxed = _last_boxed_only_string(ground_truth)
    if boxed is not None:
        return boxed

    m = _GSM8K_ANSWER_RE.search(ground_truth)
    if m:
        return m.group(1).strip().replace(",", "")

    return ground_truth


def compute_score(model_output: str, ground_truth: str) -> dict:
    """Score using grade_answer + math_equal (comprehensive sympy-based).

    Returns dict with keys: score (float), acc (bool), pred (str),
    gold_extracted (str).
    """
    model_output = str(model_output)
    ground_truth = str(ground_truth)

    gold_extracted = _extract_gold(ground_truth)

    _, extracted = _match_answer(model_output)

    if grade_answer(extracted, gold_extracted):
        return {"score": 1.0, "acc": True, "pred": extracted, "gold_extracted": gold_extracted}

    try:
        if "\\pi" in extracted or "\\pi" in gold_extracted:
            equivs = [math_equal(extracted, gold_extracted, timeout=True, pi=p) for p in [math.pi, 3.14]]
            is_correct = any(equivs)
        else:
            is_correct = math_equal(extracted, gold_extracted, timeout=True)
    except Exception:
        is_correct = False

    return {"score": 1.0 if is_correct else 0.0, "acc": is_correct, "pred": extracted, "gold_extracted": gold_extracted}
