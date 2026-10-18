"""
LightEval offline dataset redirect patch (no task overrides).

Goal:
- Keep evaluation identical to open-r1 (use LightEval built-in tasks/metrics/prompts)
- Only change dataset IO: load the required HF datasets from local snapshots downloaded via `hf download`
  under LOCAL_DATASETS_ROOT (default: /data/wh/czj/test/datasets).

Why this file exists:
- LightEval 0.9 imports `load_dataset` into `lighteval.utils.utils` at import-time, so patching
  `datasets.load_dataset` alone is insufficient.
- Many `datasets` versions no longer support "dataset scripts" (path/to/script.py).
  We instead redirect hub ids to `load_dataset("parquet"/"json", data_files=...)`.

This module is intended to be passed via:
  lighteval ... --custom-tasks eval.lighteval.custom_tasks.offline_patch

It defines an empty TASKS_TABLE to satisfy the registry.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Sequence


TASKS_TABLE = []

_LCB_REPO_IDS = frozenset(
    {
        "livecodebench/code_generation_lite",
        "livecodebench/code_generation",
    }
)
_LCB_RELEASE_RE = re.compile(r"^release_v([1-6])$")
_LCB_RANGE_RE = re.compile(r"^v([1-6])(?:_v([1-6]))?$")


def _local_root() -> Path:
    import os

    return Path(os.environ.get("LOCAL_DATASETS_ROOT", "/data/wh/czj/test/datasets")).expanduser()


def _group_by_split(files: Sequence[Path]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in files:
        n = p.name.lower()
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


def _pick_files(dataset_dir: Path) -> tuple[Optional[str], dict[str, list[str]]]:
    parquet = sorted(dataset_dir.rglob("*.parquet"))
    if parquet:
        return "parquet", _group_by_split(parquet)
    jsonl = sorted(dataset_dir.rglob("*.jsonl"))
    js = sorted(dataset_dir.rglob("*.json"))
    js = [p for p in js if p.name.lower() not in ("dataset_infos.json", "state.json")]
    files = jsonl + js
    if files:
        return "json", _group_by_split(files)
    return None, {}


def _find_dir(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def _find_hf_download_repo_dir(repo_id: str) -> Optional[Path]:
    """
    Find a local directory created by `hf download <repo_id>` under LOCAL_DATASETS_ROOT.

    We support a few common layouts:
    - $ROOT/<owner>/<name>[/main]
    - $ROOT/<name>[/main]
    - Fallback: rglob() for a directory named <name>
    """
    root = _local_root()
    roots = [root, root / "datasets"]
    parts = repo_id.split("/")
    owner = parts[0] if len(parts) > 1 else None
    name = parts[1] if len(parts) > 1 else parts[0]

    candidates: list[Path] = []
    for r in roots:
        if owner is not None:
            candidates.extend(
                [
                    r / owner / name,
                    r / owner / name / "main",
                    r / name,
                    r / name / "main",
                ]
            )
        else:
            candidates.extend([r / name, r / name / "main"])

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    # Fallback: search for a directory named like the dataset.
    for r in roots:
        if not r.exists() or not r.is_dir():
            continue
        try:
            for p in r.rglob(name):
                if p.is_dir():
                    return p
        except Exception:
            continue
    return None


def _lcb_files_for_subset(local_dir: Path, subset: Optional[str]) -> dict[str, list[str]]:
    """Return the exact local JSONL files represented by an LCB subset."""
    subset = subset or "release_latest"
    if subset == "release_latest":
        start, end = 1, 6
    elif match := _LCB_RELEASE_RE.fullmatch(subset):
        start, end = 1, int(match.group(1))
    elif match := _LCB_RANGE_RE.fullmatch(subset):
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        if start > end:
            raise ValueError(f"Invalid LiveCodeBench subset range: {subset!r}")
    else:
        raise ValueError(
            f"Unsupported LiveCodeBench subset: {subset!r}. "
            "Expected release_v1..release_v6, release_latest, v1..v6, or a range such as v4_v5."
        )

    paths = [local_dir / ("test.jsonl" if index == 1 else f"test{index}.jsonl") for index in range(start, end + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LiveCodeBench files for subset {subset!r}: {missing}")
    return {"test": [str(path) for path in paths]}


def _resolve_local_dir(hub_id: str, name: Optional[str] = None) -> Optional[Path]:
    """Try to find a local directory for any Hub dataset ID under LOCAL_DATASETS_ROOT.

    Checks common layouts: $ROOT/<id>, $ROOT/<name>, $ROOT/<id>/<config>, etc.
    Returns the directory path if found, None otherwise.
    """
    root = _local_root()
    roots = [root, root / "datasets"]
    parts = hub_id.split("/")
    short_name = parts[-1]

    candidates: list[Path] = []
    for r in roots:
        candidates.append(r / hub_id)
        if len(parts) > 1:
            candidates.append(r / parts[0] / parts[1])
        candidates.append(r / short_name)
        if name:
            candidates.append(r / short_name / name)
            candidates.append(r / hub_id / name)

    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def _redirect_dataset_if_local(path: str, name: Optional[str]) -> tuple[Optional[str], Optional[dict[str, list[str]]]]:
    root = _local_root()
    roots = [root, root / "datasets"]

    # Note: LightEval built-in tasks may use either `gsm8k` or `openai/gsm8k`.
    if path in ("openai/gsm8k", "gsm8k"):
        # Snapshot layout (from your screenshot):
        #   gsm8k/main/train-*.parquet, gsm8k/main/test-*.parquet
        #   gsm8k/socratic/...
        sub = name or "main"
        ds_dir = None
        for r in roots:
            ds_dir = _find_dir(r / "gsm8k" / sub, r / "GSM8K" / sub, r / "gsm8k", r / "GSM8K")
            if ds_dir is not None:
                break
        if ds_dir is None:
            return None, None
        kind, data_files = _pick_files(ds_dir)
        if kind is None or not data_files:
            return None, None
        return kind, data_files

    # Note: LightEval built-in tasks may use either `HuggingFaceH4/MATH-500` or a short alias.
    if path in ("HuggingFaceH4/MATH-500", "math_500", "MATH-500", "MATH_500"):
        # Snapshot layout:
        #   MATH-500/test.jsonl
        ds_dir = None
        for r in roots:
            ds_dir = _find_dir(r / "MATH-500", r / "MATH_500")
            if ds_dir is not None:
                break
        if ds_dir is None:
            return None, None
        kind, data_files = _pick_files(ds_dir)
        if kind is None or not data_files:
            return None, None
        if "test" not in data_files:
            data_files = {"test": sum(data_files.values(), [])}
        return kind, data_files

    # LiveCodeBench (LCB): try to load from local `hf download` directory if present.
    # LightEval typically calls datasets.load_dataset("livecodebench/code_generation_lite", ...).
    if path in _LCB_REPO_IDS:
        local_dir = _find_hf_download_repo_dir("livecodebench/code_generation_lite")
        if local_dir is None:
            return None, None
        return "json", _lcb_files_for_subset(local_dir, name)

    return None, None


def _install_patch() -> None:
    try:
        import inspect
        import datasets  # type: ignore
        import huggingface_hub  # type: ignore

        _orig = datasets.load_dataset
        supports_trust_remote_code = "trust_remote_code" in inspect.signature(_orig).parameters

        # ---------------------------------------------------------------------
        # Hub snapshot redirect (for script-based datasets like LiveCodeBench).
        #
        # Some datasets (e.g., `livecodebench/code_generation_lite`) ship as dataset scripts and can't be trivially
        # rewritten to parquet/json without a conversion step. For these, we redirect Hugging Face Hub's
        # `snapshot_download` to a local `hf download` directory under LOCAL_DATASETS_ROOT.
        # ---------------------------------------------------------------------
        _orig_snapshot_download = huggingface_hub.snapshot_download

        def _snapshot_download_compat(*args, **kwargs):  # type: ignore[no-untyped-def]
            repo_id = kwargs.get("repo_id", None)
            if repo_id is None and len(args) > 0:
                repo_id = args[0]
            repo_type = kwargs.get("repo_type", None) or "model"

            if isinstance(repo_id, str) and repo_type in ("dataset", "datasets"):
                if repo_id in _LCB_REPO_IDS:
                    local_dir = _find_hf_download_repo_dir("livecodebench/code_generation_lite")
                    if local_dir is not None:
                        return str(local_dir)

            return _orig_snapshot_download(*args, **kwargs)

        huggingface_hub.snapshot_download = _snapshot_download_compat  # type: ignore[assignment]

        def _load_dataset_compat(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Normalize path/name from args/kwargs.
            path = kwargs.get("path", None)
            if path is None and len(args) > 0:
                path = args[0]
            name = kwargs.get("name", None)
            if name is None and len(args) > 1:
                name = args[1]
            if name is None:
                name = kwargs.get("version_tag", None)

            # Drop trust_remote_code if unsupported.
            if not supports_trust_remote_code and "trust_remote_code" in kwargs:
                kwargs.pop("trust_remote_code", None)

            if isinstance(path, str):
                builder, data_files = _redirect_dataset_if_local(path, name)
                if builder is not None and data_files is not None:
                    # Remove hub-specific args.
                    kwargs.pop("name", None)
                    kwargs.pop("data_dir", None)
                    kwargs.pop("revision", None)
                    kwargs.pop("version_tag", None)
                    kwargs.pop("path", None)
                    return _orig(builder, data_files=data_files, **kwargs)

                # Special-case LCB: if local dir exists but we couldn't rewrite to parquet/json,
                # still avoid the Hub by loading directly from the local directory.
                if path in _LCB_REPO_IDS:
                    local_dir = _find_hf_download_repo_dir("livecodebench/code_generation_lite")
                    if local_dir is not None:
                        kwargs2 = dict(kwargs)
                        kwargs2["path"] = str(local_dir)
                        return _orig(**kwargs2)

                # Generic fallback: if a matching local directory exists (arrow cache,
                # dataset_infos.json, etc.), load from the local path directly instead
                # of hitting the Hub.  This covers datasets whose local copy isn't in
                # parquet/json format.
                if not Path(path).is_absolute():
                    local_dir = _resolve_local_dir(path, name)
                    if local_dir is not None:
                        new_kwargs = dict(kwargs)
                        new_kwargs.pop("revision", None)
                        new_kwargs.pop("path", None)
                        new_args = (str(local_dir),) + args[1:]
                        try:
                            return _orig(*new_args, **new_kwargs)
                        except Exception:
                            pass

            return _orig(*args, **kwargs)

        datasets.load_dataset = _load_dataset_compat  # type: ignore[assignment]

        # Patch the already-imported binding inside LightEval 0.9.
        try:
            import lighteval.utils.utils as _le_utils  # type: ignore

            if hasattr(_le_utils, "load_dataset"):
                _le_utils.load_dataset = _load_dataset_compat  # type: ignore[assignment]
        except Exception:
            pass
    except Exception:
        return


_install_patch()
