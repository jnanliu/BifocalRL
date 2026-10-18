"""
Local dataset script for MATH-500 compatible format.

This script is meant to be loaded by Hugging Face `datasets.load_dataset(path=..., name=...)`.
LightEval will call:
  load_dataset(path=<this_file>, name="default")

We resolve the actual dataset files from:
  ${LOCAL_DATASETS_ROOT}/MATH-500
with some fallbacks on casing.

Supported file formats inside that directory:
- Parquet (*.parquet) (preferred if present)
- JSONL (*.jsonl)
- JSON (*.json) (either list-of-dicts or {"data": [...]})

Expected fields (best-effort mapping):
- problem / question / prompt
- answer / final_answer / target / solution
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import datasets


@dataclass
class _Paths:
    root: Path
    dataset_dir: Path


def _get_paths() -> _Paths:
    root = Path(os.environ.get("LOCAL_DATASETS_ROOT", "/data/wh/czj/test/datasets")).expanduser()
    candidates = [
        root / "MATH-500",
        root / "math_500",
        root / "math-500",
        root / "HuggingFaceH4" / "MATH-500",
        root / "HuggingFaceH4" / "MATH_500",
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            return _Paths(root=root, dataset_dir=d)
    # Default to first candidate for clearer error messages downstream.
    return _Paths(root=root, dataset_dir=candidates[0])


def _iter_parquet_rows(path: Path) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.column_names
    data = table.to_pydict()
    n = len(data[cols[0]]) if cols else 0
    for i in range(n):
        yield {c: data[c][i] for c in cols}


def _iter_json_rows(path: Path) -> Iterator[Dict[str, Any]]:
    s = path.read_text(encoding="utf-8")
    obj = json.loads(s)
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        items = obj["data"]
    elif isinstance(obj, list):
        items = obj
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")
    for it in items:
        if isinstance(it, dict):
            yield it


def _iter_jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _pick_files(dataset_dir: Path) -> Tuple[List[Path], str]:
    # Prefer parquet if present.
    parquet = sorted(dataset_dir.rglob("*.parquet"))
    if parquet:
        return parquet, "parquet"
    jsonl = sorted(dataset_dir.rglob("*.jsonl"))
    if jsonl:
        return jsonl, "jsonl"
    js = sorted(dataset_dir.rglob("*.json"))
    if js:
        return js, "json"
    return [], "none"


def _normalize_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    problem = ex.get("problem", ex.get("question", ex.get("prompt", "")))
    answer = ex.get("answer", ex.get("final_answer", ex.get("target", ex.get("solution", ""))))
    solution = ex.get("solution", ex.get("rationale", ""))
    uid = ex.get("unique_id", ex.get("id", ex.get("uid", "")))
    subject = ex.get("subject", "")
    level = ex.get("level", "")
    return {
        "problem": "" if problem is None else str(problem),
        "answer": "" if answer is None else str(answer),
        "solution": "" if solution is None else str(solution),
        "unique_id": "" if uid is None else str(uid),
        "subject": "" if subject is None else str(subject),
        "level": "" if level is None else str(level),
    }


class LocalMath500Config(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class LocalMath500(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        LocalMath500Config(name="default", description="Local MATH-500 dataset"),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Local MATH-500 dataset (loaded from LOCAL_DATASETS_ROOT).",
            features=datasets.Features(
                {
                    "problem": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "solution": datasets.Value("string"),
                    "unique_id": datasets.Value("string"),
                    "subject": datasets.Value("string"),
                    "level": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        p = _get_paths()
        files, kind = _pick_files(p.dataset_dir)
        if not files:
            raise FileNotFoundError(
                f"Could not find any dataset files under {p.dataset_dir}.\n"
                f"Set LOCAL_DATASETS_ROOT to your datasets root (currently {p.root}).\n"
                "Expected one of: *.parquet, *.jsonl, *.json"
            )

        # MATH-500 is a test-only set in HF; we expose it as `test`.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files": [str(f) for f in files], "kind": kind},
            )
        ]

    def _generate_examples(self, files: List[str], kind: str):
        idx = 0
        for fp in files:
            path = Path(fp)
            if kind == "parquet":
                it = _iter_parquet_rows(path)
            elif kind == "jsonl":
                it = _iter_jsonl_rows(path)
            elif kind == "json":
                it = _iter_json_rows(path)
            else:
                raise ValueError(f"Unsupported kind: {kind}")
            for ex in it:
                yield idx, _normalize_example(ex)
                idx += 1

