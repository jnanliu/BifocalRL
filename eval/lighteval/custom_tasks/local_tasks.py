"""
Local LightEval tasks that load datasets from a local root directory.

This module patches `datasets.load_dataset` to redirect specific hub dataset
ids (e.g. `openai/gsm8k`, `HuggingFaceH4/MATH-500`) to local snapshot folders
under `LOCAL_DATASETS_ROOT` without requiring network access.

It also defines custom task configs (`local_math_500`, `local_gsm8k`) that can
be used as alternatives to LightEval's built-in tasks.  When using the built-in
tasks, pass `eval.lighteval.custom_tasks.offline_patch` instead — it applies the
same dataset redirect without registering any custom tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------
# `datasets` compatibility:
# - ignore trust_remote_code if unsupported
# - optionally redirect hub dataset ids to LOCAL_DATASETS_ROOT snapshots
# ---------------------------------------------------------------------
try:
    import inspect
    import datasets  # type: ignore

    _orig_load_dataset = datasets.load_dataset
    _supports_trust_remote_code = "trust_remote_code" in inspect.signature(_orig_load_dataset).parameters

    def _local_root() -> Path:
        import os

        return Path(os.environ.get("LOCAL_DATASETS_ROOT", "/data/wh/czj/test/datasets")).expanduser()

    def _pick_files(dataset_dir: Path) -> tuple[Optional[str], dict]:
        parquet = sorted(dataset_dir.rglob("*.parquet"))
        if parquet:
            return "parquet", _group_by_split(parquet)
        jsonl = sorted(dataset_dir.rglob("*.jsonl"))
        js = sorted(dataset_dir.rglob("*.json"))
        js = [p for p in js if p.name.lower() not in ("dataset_infos.json", "state.json")]
        if jsonl or js:
            files = jsonl + js
            return "json", _group_by_split(files)
        return None, {}

    def _group_by_split(files: Sequence[Path]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in files:
            n = p.name.lower()
            split = None
            if "train" in n:
                split = "train"
            elif "test" in n:
                split = "test"
            elif "validation" in n or "valid" in n or "dev" in n:
                split = "validation"
            else:
                split = "test"
            out.setdefault(split, []).append(str(p))
        return out

    def _redirect_dataset_if_local(path: str) -> tuple[Optional[str], Optional[str], Optional[dict[str, list[str]]]]:
        root = _local_root()
        roots = [root, root / "datasets"]
        if path == "openai/gsm8k":
            ds_dir = None
            for r in roots:
                for cand in (r / "gsm8k", r / "GSM8K"):
                    if cand.exists():
                        ds_dir = cand
                        break
                if ds_dir is not None:
                    break
            kind, data_files = _pick_files(ds_dir) if ds_dir is not None and ds_dir.exists() else (None, {})
            if kind is None or not data_files:
                return None, None, None
            return kind, None, data_files
        if path == "HuggingFaceH4/MATH-500":
            ds_dir = None
            for r in roots:
                for cand in (r / "MATH-500", r / "MATH_500"):
                    if cand.exists():
                        ds_dir = cand
                        break
                if ds_dir is not None:
                    break
            kind, data_files = _pick_files(ds_dir) if ds_dir is not None and ds_dir.exists() else (None, {})
            if kind is None or not data_files:
                return None, None, None
            if "test" not in data_files:
                data_files = {"test": sum(data_files.values(), [])}
            return kind, None, data_files
        return None, None, None

    def _load_dataset_compat(*args, **kwargs):  # type: ignore[no-untyped-def]
        path = kwargs.get("path", None)
        if path is None and len(args) > 0:
            path = args[0]

        if not _supports_trust_remote_code and "trust_remote_code" in kwargs:
            kwargs.pop("trust_remote_code", None)

        if isinstance(path, str):
            builder, _name, data_files = _redirect_dataset_if_local(path)
            if builder is not None and data_files is not None:
                kwargs.pop("name", None)
                kwargs.pop("data_dir", None)
                kwargs.pop("revision", None)
                kwargs.pop("path", None)
                return _orig_load_dataset(builder, data_files=data_files, **kwargs)

        return _orig_load_dataset(*args, **kwargs)

    datasets.load_dataset = _load_dataset_compat  # type: ignore[assignment]

    try:
        import lighteval.utils.utils as _le_utils  # type: ignore

        if hasattr(_le_utils, "load_dataset"):
            _le_utils.load_dataset = _load_dataset_compat  # type: ignore[assignment]
    except Exception:
        pass
except Exception:
    pass


from lighteval.metrics.metrics import Metrics  # noqa: E402
from lighteval.tasks.lighteval_task import LightevalTaskConfig  # noqa: E402
from lighteval.tasks.requests import Doc  # noqa: E402


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _strip(s: str) -> str:
    return _safe_str(s).strip()


def _prompt_math_500(line: dict, task_name: str) -> Doc:
    problem = _strip(line.get("problem", line.get("question", line.get("prompt", ""))))
    answer = _safe_str(line.get("answer", line.get("final_answer", line.get("target", line.get("solution", "")))))

    query = problem
    if query != "" and not query.rstrip().endswith("Answer:"):
        query = f"{query.rstrip()}\n\nAnswer:"

    try:
        return Doc(task_name=task_name, query=query, choices=[answer], gold_index=0, specific={"gold": answer})
    except TypeError:
        return Doc(task_name=task_name, query=query, choices=[answer], gold_index=0)


def _prompt_gsm8k(line: dict, task_name: str) -> Doc:
    question = _strip(line.get("question", line.get("prompt", "")))
    answer = _safe_str(line.get("answer", line.get("target", "")))

    query = question
    if query != "" and not query.rstrip().endswith("Answer:"):
        query = f"{query.rstrip()}\n\nAnswer:"

    try:
        return Doc(task_name=task_name, query=query, choices=[answer], gold_index=0, specific={"gold": answer})
    except TypeError:
        return Doc(task_name=task_name, query=query, choices=[answer], gold_index=0)


def _make_task_config(
    *,
    name: str,
    prompt_function,
    hf_repo: str,
    hf_subset: str,
    hf_avail_splits,
    evaluation_splits,
    few_shots_split,
    few_shots_select,
    metric,
) -> LightevalTaskConfig:
    import inspect

    sig = inspect.signature(LightevalTaskConfig)
    kwargs = dict(
        name=name,
        prompt_function=prompt_function,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=hf_avail_splits,
        evaluation_splits=evaluation_splits,
        few_shots_split=few_shots_split,
        few_shots_select=few_shots_select,
        generation_size=2048,
        stop_sequence=[],
    )

    if "suite" in sig.parameters:
        kwargs["suite"] = ["custom"]
    if "metric" in sig.parameters:
        kwargs["metric"] = list(metric)
    elif "metrics" in sig.parameters:
        kwargs["metrics"] = list(metric)
    else:
        kwargs["metric"] = list(metric)
    return LightevalTaskConfig(**kwargs)  # type: ignore[arg-type]


TASKS_TABLE = [
    _make_task_config(
        name="local_math_500",
        prompt_function=_prompt_math_500,
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        metric=[Metrics.exact_match],
    ),
    _make_task_config(
        name="local_gsm8k",
        prompt_function=_prompt_gsm8k,
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        metric=[Metrics.exact_match],
    ),
]
