from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def load_cruxeval_jsonl(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def resolve_datasets_root(project_root: str) -> str:
    """
    Resolve the datasets root directory.

    In this repo, code is often executed from `co-learning/`, but datasets may live at the
    *repo* level (sibling of co-learning), e.g.:
      /home/.../test/
        - co-learning/
        - datasets/
    """
    for k in ("CRUXEVAL_DATASETS_ROOT", "LOCAL_DATASETS_ROOT", "DATASETS_ROOT"):
        v = os.environ.get(k, "").strip()
        if v:
            return v

    # A commonly used path in this project (provided by users / scripts).
    hardcoded = "/home/jizeyu26/jzy/7_co_learning/test/datasets"
    if os.path.isdir(hardcoded):
        return hardcoded

    parent = os.path.abspath(os.path.join(project_root, os.pardir))
    sibling = os.path.join(parent, "datasets")
    if os.path.isdir(sibling):
        return sibling

    local = os.path.join(project_root, "datasets")
    if os.path.isdir(local):
        return local

    # Final fallback (even if it doesn't exist yet).
    return sibling


def default_cruxeval_path(project_root: str) -> str:
    ds_root = resolve_datasets_root(project_root)
    return os.path.join(ds_root, "cruxeval", "test.jsonl")


def construct_cruxeval_prompt(sample: Dict[str, Any]) -> str:
    code = sample["code"]
    input_val = sample["input"]
    return (
        "You are given a Python function and an input. You need to predict the output of the function execution.\n"
        "The output must be a valid Python literal (e.g., 'hello', 5, [1, 2], {'a': 1}).\n"
        "Do not output explanations.\n\n"
        f"[python]\n{code}\n[/python]\n\n"
        f"Input: {input_val}\n"
        "Output:"
    )


def _safe_literal_eval(val_str: str) -> Tuple[bool, Any]:
    try:
        return True, ast.literal_eval(val_str)
    except Exception:
        return False, None


def _smart_extract_literal(text: str) -> str:
    text = text.strip()
    ok, _ = _safe_literal_eval(text)
    if ok:
        return text
    for i in range(len(text), 0, -1):
        sub = text[:i]
        ok, _ = _safe_literal_eval(sub)
        if ok:
            return sub
    return text


def extract_cruxeval_answer(generation: str) -> str:
    """
    Extract a single-line python literal from model output.
    """
    if generation is None:
        return ""
    s = str(generation).strip()

    # Prefer extracting from after an explicit "Output:" marker (may appear mid-text).
    # We take the LAST occurrence to handle cases where the model restates the instruction.
    candidate = None
    matches = list(re.finditer(r"(?i)\bOutput\s*:\s*(.*)", s))
    if matches:
        candidate = matches[-1].group(1).strip()
    else:
        candidate = s

    first_line = candidate.split("\n", 1)[0].strip()

    prefixes = ["[output]", "output:", "result:", "answer:"]
    for p in prefixes:
        if first_line.lower().startswith(p):
            first_line = first_line[len(p) :].strip()

    return _smart_extract_literal(first_line)


@dataclass(frozen=True)
class CruxEvalJudgement:
    is_correct: bool
    predicted: str
    ground_truth: str


def judge_cruxeval(predicted: str, ground_truth: str) -> CruxEvalJudgement:
    p = (predicted or "").strip()
    g = (ground_truth or "").strip()

    if p == g:
        return CruxEvalJudgement(True, p, g)

    ok_p, obj_p = _safe_literal_eval(p)
    ok_g, obj_g = _safe_literal_eval(g)
    if ok_p and ok_g:
        try:
            return CruxEvalJudgement(bool(obj_p == obj_g), p, g)
        except Exception:
            pass

    # Quote-insensitive fallback for strings
    if p.strip("'\"") == g.strip("'\""):
        return CruxEvalJudgement(True, p, g)

    return CruxEvalJudgement(False, p, g)

