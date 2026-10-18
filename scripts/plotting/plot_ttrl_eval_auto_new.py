#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


METHOD_LABELS: dict[str, str] = {
    "single-intuitor-gspo": "intuitor gspo",
    "repro-gspo-filter2x": "filter2x",
    "single-intuitor-grpo": "intuitor grpo",
    "single-mixed-reward-adaptive-sc-gspo": "adaptive sc",
    "single-mixed-reward-alpha025-gspo": "alpha=0.25",
    "single-mixed-reward-alpha050-gspo": "alpha=0.50 (reliability baseline)",
    "single-mixed-reward-alpha075-gspo": "alpha=0.75",
    "single-mixed-reward-adaptive-sc-zscore-gspo": "adaptive sc zscore",
    "single-mixed-reward-adaptive-sc-zscore-corr-gspo": "adaptive sc zscore + corr",
    "single-mixed-reward-reliability-06sc-04corr-all-gspo": "reliability 0.6sc/0.4corr all",
    "single-mixed-reward-reliability-05sc-05corr-all-gspo": "reliability 0.5sc/0.5corr all (sigma=0.2)",
    "single-mixed-reward-reliability-05sc-05corr-all-sigma01-gspo": "reliability 0.5sc/0.5corr all (sigma=0.1)",
    "single-mixed-reward-reliability-03sc-07corr-all-sigma01-gspo": "reliability 0.3sc/0.7corr all (sigma=0.1)",
    "single-mixed-reward-reliability-04sc-06corr-all-gspo": "reliability 0.4sc/0.6corr all",
    "single-mixed-reward-reliability-04sc-06corr-all-sigma01-gspo": "reliability 0.4sc/0.6corr all (sigma=0.1)",
    "single-mixed-reward-reliability-06sc-04corr-all-sigma01-gspo": "reliability 0.6sc/0.4corr all (sigma=0.1)",
    "single-mixed-reward-reliability-06sc-04corr-advantage-only-gspo": "reliability 0.6sc/0.4corr advantage only",
    "single-mixed-reward-reliability-05sc-05corr-advantage-only-gspo": "reliability 0.5sc/0.5corr advantage only (sigma=0.2)",
    "single-mixed-reward-reliability-05sc-05corr-advantage-only-sigma01-gspo": "reliability 0.5sc/0.5corr advantage only (sigma=0.1)",
    "single-mixed-reward-reliability-04sc-06corr-advantage-only-gspo": "reliability 0.4sc/0.6corr advantage only",
    "single-mixed-reward-reliability-06sc-04corr-sample-only-gspo": "reliability 0.6sc/0.4corr sample only",
    "single-mixed-reward-reliability-05sc-05corr-sample-only-gspo": "reliability 0.5sc/0.5corr sample only (sigma=0.2)",
    "single-mixed-reward-reliability-05sc-05corr-sample-only-sigma01-gspo": "reliability 0.5sc/0.5corr sample only (sigma=0.1)",
    "single-mixed-reward-reliability-04sc-06corr-sample-only-gspo": "reliability 0.4sc/0.6corr sample only",
    "single-mixed-reward-reliability-baseline-gspo": "reliability baseline (alpha=0.50)",
    "single-mv-gspo": "mv gspo",
    "single-mv-grpo": "mv grpo",
    "repro-gspo": "repro gspo",
    "repro-grpo": "repro grpo",
}

DATASET_CONFIG: dict[str, dict[str, str]] = {
    "math_500": {
        "backend": "ttrl",
        "display_name": "math_500",
        "y_label": "Pass@1",
    },
    "gsm8k": {
        "backend": "ttrl",
        "display_name": "gsm8k",
        "y_label": "Pass@1",
    },
    "aime24": {
        "backend": "ttrl",
        "display_name": "AIME24",
        "y_label": "Pass@1",
    },
    "aime25": {
        "backend": "ttrl",
        "display_name": "AIME25",
        "y_label": "Pass@1",
    },
    "amc23": {
        "backend": "ttrl",
        "display_name": "AMC23",
        "y_label": "Pass@1",
    },
    "lcb_release_v6": {
        "backend": "lighteval",
        "display_name": "LiveCodeBench release_v6",
        "y_label": "Pass@1",
        "task_name": "lcb:codegeneration_release_v6",
        "metric_prefix": "codegen_pass@1:",
    },
    "cruxeval": {
        "backend": "hacrl",
        "display_name": "CRUXEval-O",
        "y_label": "Pass@1",
    },
}

STYLE_POOL: list[dict[str, Any]] = [
    {"color": "#6B7280", "linestyle": "-.", "marker": "P"},
    {"color": "#059669", "linestyle": "-", "marker": "^"},
    {"color": "#2563EB", "linestyle": "-", "marker": "o"},
    {"color": "#DC2626", "linestyle": "-", "marker": "v"},
    {"color": "#7C3AED", "linestyle": "--", "marker": "D"},
    {"color": "#A16207", "linestyle": "-.", "marker": "h"},
    {"color": "#00A6D6", "linestyle": ":", "marker": "X"},
    {"color": "#F59E0B", "linestyle": "--", "marker": "*"},
    {"color": "#111827", "linestyle": "-", "marker": "s"},
    {"color": "#CC6677", "linestyle": "--", "marker": "d"},
    {"color": "#88CCEE", "linestyle": "-.", "marker": "<"},
    {"color": "#332288", "linestyle": ":", "marker": ">"},
    {"color": "#AA4499", "linestyle": "-", "marker": "p"},
    {"color": "#117733", "linestyle": "--", "marker": "H"},
    {"color": "#DDCC77", "linestyle": "-.", "marker": "8"},
    {"color": "#882255", "linestyle": ":", "marker": "D"},
    {"color": "#0077BB", "linestyle": "-", "marker": "1"},
    {"color": "#EE7733", "linestyle": "--", "marker": "2"},
    {"color": "#33BBEE", "linestyle": "-.", "marker": "3"},
    {"color": "#EE3377", "linestyle": ":", "marker": "4"},
]

FIGURE_CONFIG: dict[str, Any] = {
    "size": (12, 6),
    "dpi": 220,
}

Y_AXIS_CONFIG: dict[str, Any] = {
    "label": "Pass@1",
    "limits": None,  # e.g. (0.55, 0.85)
}

TITLE_TEMPLATE = "{model_tag} - {dataset}"
FILENAME_TEMPLATE = "{model_tag}_{dataset}.png"

LOGGER = logging.getLogger("plot_ttrl_eval_auto_new")
BASELINE_GROUP_DECIMALS = 12


def get_co_learning_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = get_co_learning_root() / path
    return path.resolve()


def get_default_eval_root(output_group: str) -> Path:
    return get_co_learning_root() / "outputs" / output_group / "eval"


def get_eval_root(output_group: str, eval_root: str | None) -> Path:
    if eval_root:
        return resolve_repo_path(eval_root)
    return get_default_eval_root(output_group)


def get_output_root(output_dir: str | None, override: str | None) -> Path:
    if override:
        return resolve_repo_path(override)
    if output_dir:
        return resolve_repo_path(output_dir)
    return get_co_learning_root() / "scripts" / "plotting" / "output" / "auto_new"


def get_model_output_dir(model_tag: str, output_dir: str | None, override: str | None) -> Path:
    return get_output_root(output_dir, override) / model_tag


def build_eval_method_root(
    eval_root: Path,
    model_tag: str,
    method_name: str,
    method_dir_suffix: str = "",
) -> Path:
    return eval_root / model_tag / f"{method_name}{method_dir_suffix}"


def build_eval_method_roots(
    eval_root: Path,
    model_tag: str,
    method_name: str,
    method_dir_suffix: str = "",
) -> list[Path]:
    """Return normal results first, followed by compatible eval-only results."""
    suffixes = ["", method_dir_suffix, "-extended-eval"]
    roots: list[Path] = []
    for suffix in suffixes:
        root = build_eval_method_root(eval_root, model_tag, method_name, suffix)
        if root not in roots:
            roots.append(root)
    return roots


def build_summary_path(method_root: Path, step: int) -> Path:
    return method_root / f"a_checkpoint-{step}" / "comparison_summary.json"


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Summary file is not a JSON object: {path}")
    return data


def serialize_source_path(path: Path) -> str:
    """Keep project-local result paths portable in the plotting data JSON."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(get_co_learning_root()).as_posix()
    except ValueError:
        return str(resolved_path)


def validate_figure_config(figure_config: dict[str, Any]) -> None:
    fig_size = figure_config.get("size", (12, 6))
    if not isinstance(fig_size, (list, tuple)) or len(fig_size) != 2:
        raise ValueError("`FIGURE_CONFIG['size']` must be a length-2 list or tuple.")


def validate_y_axis_config(y_axis_config: dict[str, Any]) -> None:
    y_limits = y_axis_config.get("limits")
    if y_limits is None:
        return
    if not isinstance(y_limits, (list, tuple)) or len(y_limits) != 2:
        raise ValueError("`Y_AXIS_CONFIG['limits']` must be None or a length-2 list/tuple.")
    lower = float(y_limits[0])
    upper = float(y_limits[1])
    if lower >= upper:
        raise ValueError("`Y_AXIS_CONFIG['limits']` must satisfy lower < upper.")


def validate_style_pool(selected_method_count: int) -> None:
    if len(STYLE_POOL) < selected_method_count:
        raise ValueError(
            f"`STYLE_POOL` has {len(STYLE_POOL)} styles, but {selected_method_count} methods are enabled."
        )

    for index, style in enumerate(STYLE_POOL):
        if not isinstance(style, dict):
            raise ValueError(f"`STYLE_POOL[{index}]` must be a dict.")
        missing = [key for key in ("color", "linestyle", "marker") if key not in style]
        if missing:
            raise ValueError(f"`STYLE_POOL[{index}]` is missing keys: {', '.join(missing)}.")


def validate_runtime_settings(
    model_tags: list[str],
    selected_methods: list[str],
    datasets: list[str],
    start_step: int,
    end_step: int,
    dataset_method_suffixes: dict[str, str] | None = None,
) -> None:
    if not model_tags:
        raise ValueError("`model_tags` must be a non-empty list.")
    if not selected_methods:
        raise ValueError("`selected_methods` must be a non-empty list.")
    if not datasets:
        raise ValueError("`datasets` must be a non-empty list.")
    if start_step < 0:
        raise ValueError("`start_step` must be >= 0.")
    if end_step < start_step:
        raise ValueError("`end_step` must be >= `start_step`.")

    unknown_methods = [method_name for method_name in selected_methods if method_name not in METHOD_LABELS]
    if unknown_methods:
        joined = ", ".join(unknown_methods)
        raise ValueError(f"`selected_methods` contains methods missing from `METHOD_LABELS`: {joined}")

    unknown_datasets = [dataset for dataset in datasets if dataset not in DATASET_CONFIG]
    if unknown_datasets:
        joined = ", ".join(unknown_datasets)
        raise ValueError(f"`datasets` contains datasets missing from `DATASET_CONFIG`: {joined}")

    if dataset_method_suffixes is not None:
        invalid_suffixes = {
            dataset: suffix
            for dataset, suffix in dataset_method_suffixes.items()
            if not isinstance(suffix, str) or "/" in suffix or "\\" in suffix
        }
        if invalid_suffixes:
            raise ValueError(
                "`dataset_method_suffixes` values must be path-component suffix strings: "
                f"{invalid_suffixes}"
            )

    validate_style_pool(len(selected_methods))
    validate_figure_config(FIGURE_CONFIG)
    validate_y_axis_config(Y_AXIS_CONFIG)


def extract_dataset_value(summary: dict[str, Any], step: int, dataset: str) -> float:
    results = summary.get("results")
    if not isinstance(results, dict):
        raise ValueError("Missing or invalid `results` object.")

    checkpoint_key = f"checkpoint-{step}"
    checkpoint_results = results.get(checkpoint_key)
    if not isinstance(checkpoint_results, dict):
        candidate_items = [(key, value) for key, value in results.items() if isinstance(value, dict)]
        if len(candidate_items) == 1:
            fallback_key, fallback_value = candidate_items[0]
            if dataset in fallback_value:
                checkpoint_results = fallback_value
                LOGGER.info(
                    "Fallback to `results.%s` for step=%d dataset=%s",
                    fallback_key,
                    step,
                    dataset,
                )
            else:
                raise ValueError(
                    f"Missing `results.{checkpoint_key}` and fallback `results.{fallback_key}.{dataset}`."
                )
        else:
            available_keys = ", ".join(sorted(str(key) for key in results.keys()))
            raise ValueError(
                f"Missing `results.{checkpoint_key}`. Available result keys: {available_keys or '<none>'}."
            )

    value = checkpoint_results.get(dataset)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing or non-numeric dataset value for `{dataset}`.")
    return float(value)


def get_dataset_config(dataset: str) -> dict[str, str]:
    try:
        return DATASET_CONFIG[dataset]
    except KeyError as exc:
        raise ValueError(f"Dataset `{dataset}` is missing from `DATASET_CONFIG`.") from exc


def validate_metric_value(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing or non-numeric metric value for {context}.")
    numeric_value = float(value)
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"Metric value for {context} is outside [0, 1]: {numeric_value}")
    return numeric_value


def load_ttrl_dataset_value(method_root: Path, step: int, dataset: str) -> float:
    summary_path = build_summary_path(method_root, step)
    summary = load_summary(summary_path)
    return extract_dataset_value(summary, step, dataset)


def find_lcb_result_path(method_root: Path, step: int) -> Path:
    result_root = method_root / "lighteval" / f"checkpoint-{step}" / "results"
    candidates = sorted(result_root.glob("**/results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No LCB result JSON found under {result_root}")
    if len(candidates) > 1:
        selected = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        warn_once(
            f"Multiple LCB result JSON files found under {result_root}; "
            f"using newest file: {selected}"
        )
        return selected
    return candidates[0]


def extract_lcb_metric(summary: dict[str, Any], dataset: str) -> float:
    dataset_config = get_dataset_config(dataset)
    results = summary.get("results")
    if not isinstance(results, dict):
        raise ValueError("Missing or invalid LCB `results` object.")

    task_name = dataset_config.get("task_name", "")
    metric_prefix = dataset_config.get("metric_prefix", "codegen_pass@1:")
    metric_sources: list[tuple[str, dict[str, Any]]] = []

    for result_name, result_values in results.items():
        if result_name == "all" or not isinstance(result_values, dict):
            continue
        if task_name and task_name in str(result_name):
            metric_sources.append((str(result_name), result_values))

    aggregate = results.get("all")
    if isinstance(aggregate, dict):
        metric_sources.append(("all", aggregate))

    if not metric_sources:
        metric_sources = [
            (str(result_name), result_values)
            for result_name, result_values in results.items()
            if isinstance(result_values, dict)
        ]

    for source_name, source_values in metric_sources:
        metric_names = [
            str(metric_name)
            for metric_name in source_values
            if str(metric_name).startswith(metric_prefix)
            and not str(metric_name).endswith("_stderr")
        ]
        if len(metric_names) == 1:
            metric_name = metric_names[0]
            return validate_metric_value(
                source_values[metric_name],
                context=f"LCB {source_name}.{metric_name}",
            )

    available = {
        str(result_name): sorted(str(metric_name) for metric_name in result_values)
        for result_name, result_values in results.items()
        if isinstance(result_values, dict)
    }
    raise ValueError(
        f"Could not identify one non-stderr `{metric_prefix}*` metric in LCB results: {available}"
    )


def load_lcb_dataset_value(method_root: Path, step: int, dataset: str) -> float:
    result_path = find_lcb_result_path(method_root, step)
    return extract_lcb_metric(load_summary(result_path), dataset)


def load_cruxeval_dataset_value(method_root: Path, step: int) -> float:
    result_path = (
        method_root
        / "hacrl"
        / "cruxeval"
        / f"results_checkpoint-{step}_model-a.json"
    )
    result = load_summary(result_path)
    summary = result.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing or invalid CRUXEval `summary` object: {result_path}")
    return validate_metric_value(
        summary.get("pass_at_k"),
        context=f"CRUXEval checkpoint-{step}.pass_at_k",
    )


def load_dataset_value(method_root: Path, step: int, dataset: str) -> float:
    return float(load_dataset_value_with_source(method_root, step, dataset)["value"])


def load_dataset_value_with_source(
    method_root: Path,
    step: int,
    dataset: str,
) -> dict[str, Any]:
    """Load one metric and retain the exact result file used for it."""
    backend = get_dataset_config(dataset)["backend"]
    if backend == "ttrl":
        source_path = build_summary_path(method_root, step)
        value = load_ttrl_dataset_value(method_root, step, dataset)
    elif backend == "lighteval":
        source_path = find_lcb_result_path(method_root, step)
        value = extract_lcb_metric(load_summary(source_path), dataset)
    elif backend == "hacrl":
        source_path = (
            method_root
            / "hacrl"
            / "cruxeval"
            / f"results_checkpoint-{step}_model-a.json"
        )
        value = load_cruxeval_dataset_value(method_root, step)
    else:
        raise ValueError(f"Unsupported backend `{backend}` for dataset `{dataset}`.")

    return {
        "value": float(value),
        "source": serialize_source_path(source_path),
    }


def warn_once_cache() -> set[str]:
    if not hasattr(warn_once_cache, "_cache"):
        warn_once_cache._cache = set()  # type: ignore[attr-defined]
    return warn_once_cache._cache  # type: ignore[attr-defined]


def warn_once(message: str) -> None:
    cache = warn_once_cache()
    if message in cache:
        return
    cache.add(message)
    LOGGER.warning(message)


def classify_method_root(method_root: Path, method_roots: list[Path]) -> str:
    if method_root == method_roots[0]:
        return "normal"
    if method_root.name.endswith("-extended-eval"):
        return "extended-eval"
    return "dataset-suffix"


def load_series_with_details(
    eval_root: Path,
    model_tag: str,
    method_name: str,
    dataset: str,
    start_step: int,
    end_step: int,
    method_dir_suffix: str = "",
) -> tuple[list[float | None], list[dict[str, Any]]]:
    method_roots = build_eval_method_roots(
        eval_root,
        model_tag,
        method_name,
        method_dir_suffix,
    )
    existing_roots = [root for root in method_roots if root.exists()]
    if not existing_roots:
        searched = ", ".join(str(root) for root in method_roots)
        warn_once(f"Method directories not found, all points will be empty: {searched}")
        points = [
            {
                "step": step,
                "value": None,
                "status": "missing",
                "source": None,
                "source_root": None,
                "source_kind": None,
                "error": f"Method directories not found: {searched}",
            }
            for step in range(start_step, end_step + 1)
        ]
        return [None] * (end_step - start_step + 1), points

    values: list[float | None] = []
    points: list[dict[str, Any]] = []
    for step in range(start_step, end_step + 1):
        errors: list[str] = []
        value: float | None = None
        source: str | None = None
        for method_root in existing_roots:
            try:
                detail = load_dataset_value_with_source(method_root, step, dataset)
                value = float(detail["value"])
                source = str(detail["source"])
                source_root = serialize_source_path(method_root)
                break
            except Exception as exc:
                errors.append(f"{method_root}: {exc}")
        if value is None:
            LOGGER.warning(
                "Failed to extract dataset value for model=%s method=%s step=%d dataset=%s. Searched: %s",
                model_tag,
                method_name,
                step,
                dataset,
                " | ".join(errors),
            )
            values.append(None)
            points.append(
                {
                    "step": step,
                    "value": None,
                    "status": "missing",
                    "source": None,
                    "source_root": None,
                    "source_kind": None,
                    "error": " | ".join(errors),
                }
            )
            continue
        values.append(value)
        points.append(
            {
                "step": step,
                "value": value,
                "status": "ok",
                "source": source,
                "source_root": source_root,
                "source_kind": classify_method_root(method_root, method_roots),
            }
        )
    return values, points


def load_series(
    eval_root: Path,
    model_tag: str,
    method_name: str,
    dataset: str,
    start_step: int,
    end_step: int,
    method_dir_suffix: str = "",
) -> list[float | None]:
    """Compatibility wrapper returning only the values used for plotting."""
    values, _ = load_series_with_details(
        eval_root,
        model_tag,
        method_name,
        dataset,
        start_step,
        end_step,
        method_dir_suffix,
    )
    return values


def load_baseline_detail(
    eval_root: Path,
    model_tag: str,
    method_name: str,
    dataset: str,
    include_step0_baseline: bool,
    method_dir_suffix: str = "",
) -> dict[str, Any]:
    if not include_step0_baseline:
        return {
            "value": None,
            "source": None,
            "source_root": None,
            "source_kind": None,
            "status": "disabled",
        }

    errors: list[str] = []
    for method_root in build_eval_method_roots(
        eval_root,
        model_tag,
        method_name,
        method_dir_suffix,
    ):
        if not method_root.exists():
            continue
        try:
            detail = load_dataset_value_with_source(method_root, 0, dataset)
            return {
                "value": float(detail["value"]),
                "source": detail["source"],
                "source_root": serialize_source_path(method_root),
                "source_kind": classify_method_root(method_root, build_eval_method_roots(
                    eval_root, model_tag, method_name, method_dir_suffix
                )),
                "status": "ok",
            }
        except Exception as exc:
            errors.append(f"{method_root}: {exc}")

    if errors:
        LOGGER.warning(
            "Failed to extract step-0 baseline for model=%s method=%s dataset=%s. Searched: %s",
            model_tag,
            method_name,
            dataset,
            " | ".join(errors),
        )
    return {
        "value": None,
        "source": None,
        "source_root": None,
        "source_kind": None,
        "status": "missing",
        "error": " | ".join(errors),
    }


def load_baseline(
    eval_root: Path,
    model_tag: str,
    method_name: str,
    dataset: str,
    include_step0_baseline: bool,
    method_dir_suffix: str = "",
) -> float | None:
    """Compatibility wrapper returning only the baseline value."""
    return load_baseline_detail(
        eval_root,
        model_tag,
        method_name,
        dataset,
        include_step0_baseline,
        method_dir_suffix,
    ).get("value")


def format_baseline_value(value: float) -> str:
    return f"{value:.{BASELINE_GROUP_DECIMALS}g}"


def resolve_shared_baseline(
    dataset: str,
    baseline_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_candidates = [item for item in baseline_candidates if item.get("value") is not None]
    if not valid_candidates:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    for item in valid_candidates:
        value = float(item["value"])
        group_key = format_baseline_value(value)
        if group_key not in grouped:
            grouped[group_key] = {
                "value": value,
                "count": 0,
                "labels": [],
            }
            ordered_keys.append(group_key)
        grouped[group_key]["count"] += 1
        grouped[group_key]["labels"].append(str(item["label"]))

    if len(grouped) > 1:
        stats = "; ".join(
            f"{grouped[key]['value']:.12g} x{grouped[key]['count']} ({', '.join(grouped[key]['labels'])})"
            for key in ordered_keys
        )
        winner_key = max(ordered_keys, key=lambda key: (int(grouped[key]["count"]), -ordered_keys.index(key)))
        winner = grouped[winner_key]
        LOGGER.warning(
            "Step-0 baseline mismatch for dataset=%s. Using majority baseline %.12g from %d/%d methods. Details: %s",
            dataset,
            winner["value"],
            winner["count"],
            len(valid_candidates),
            stats,
        )
    else:
        winner_key = ordered_keys[0]
        winner = grouped[winner_key]

    return [
        {
            "label": "step0 baseline",
            "color": "#111827",
            "value": float(winner["value"]),
        }
    ]


def get_method_label(method_name: str) -> str:
    return METHOD_LABELS[method_name]


def pick_method_style(index: int) -> dict[str, Any]:
    if index >= len(STYLE_POOL):
        raise ValueError(
            f"`STYLE_POOL` has {len(STYLE_POOL)} styles, cannot assign style for method index {index}."
        )

    style = dict(STYLE_POOL[index])
    style.setdefault("linewidth", 2.2)
    style.setdefault("markersize", 5)
    style.setdefault("alpha", 1.0)
    return style


def prepare_selected_methods(selected_methods: list[str]) -> list[dict[str, Any]]:
    method_entries: list[dict[str, Any]] = []
    for index, method_name in enumerate(selected_methods):
        method_entries.append(
            {
                "name": method_name,
                "label": get_method_label(method_name),
                "style": pick_method_style(index),
            }
        )
    return method_entries


def apply_auto_y_limits(ax: Any, values: list[float]) -> None:
    if not values:
        ax.set_ylim(0.0, 1.0)
        return

    low = min(values)
    high = max(values)
    span = high - low
    pad = max(span * 0.12, 0.015)
    if span <= 1e-12:
        pad = max(pad, 0.04)

    y_min = max(0.0, low - pad)
    y_max = min(1.0, high + pad)
    if y_min >= y_max:
        mid = (low + high) / 2.0
        y_min = max(0.0, mid - 0.04)
        y_max = min(1.0, mid + 0.04)
    ax.set_ylim(y_min, y_max)


def plot_dataset(
    dataset: str,
    series_bundle: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    model_tag: str,
    start_step: int,
    end_step: int,
    output_dir: Path,
) -> Path | None:
    x_values = list(range(start_step, end_step + 1))
    x_labels = [f"ckpt-{step}" for step in x_values]

    numeric_values: list[float] = []
    has_any_data = False

    fig_size_raw = FIGURE_CONFIG.get("size", (12, 6))
    fig_size = (float(fig_size_raw[0]), float(fig_size_raw[1]))
    dpi = int(FIGURE_CONFIG.get("dpi", 220))
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    for item in series_bundle:
        y_values = item["y"]
        plot_y = [float("nan") if v is None else float(v) for v in y_values]
        if any(v is not None for v in y_values):
            has_any_data = True
            numeric_values.extend(float(v) for v in y_values if v is not None)
        ax.plot(x_values, plot_y, label=item["label"], **item["style"])

    for baseline in baselines:
        value = baseline["value"]
        if value is None:
            continue
        has_any_data = True
        numeric_values.append(float(value))
        ax.axhline(
            y=float(value),
            label=baseline["label"],
            color=baseline["color"],
            linestyle=":",
            linewidth=1.7,
            alpha=0.95,
        )

    if not has_any_data:
        LOGGER.warning("Skip model=%s dataset=%s: no valid data points or baselines were found.", model_tag, dataset)
        plt.close(fig)
        return None

    dataset_config = get_dataset_config(dataset)
    display_name = dataset_config.get("display_name", dataset)
    title = TITLE_TEMPLATE.format(model_tag=model_tag, dataset=display_name)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Checkpoint", fontsize=11)

    y_label = dataset_config.get("y_label") or str(Y_AXIS_CONFIG.get("label", "Pass@1"))
    ax.set_ylabel(y_label, fontsize=11)
    y_limits = Y_AXIS_CONFIG.get("limits")
    if isinstance(y_limits, (list, tuple)) and len(y_limits) == 2:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    else:
        apply_auto_y_limits(ax, numeric_values)

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = FILENAME_TEMPLATE.format(model_tag=model_tag, dataset=dataset)
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_plot_data(output_dir: Path, plot_data: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "plot_data.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(plot_data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path


def validate_plot_data(plot_data: dict[str, Any], path: Path) -> None:
    """Validate the persisted schema before using it as plotting input."""
    if plot_data.get("schema_version") != 1:
        raise ValueError(f"Unsupported plot_data schema in {path}: expected version 1.")

    model_tag = plot_data.get("model_tag")
    if not isinstance(model_tag, str) or not model_tag:
        raise ValueError(f"Missing non-empty `model_tag` in {path}.")

    method_order = plot_data.get("method_order")
    dataset_order = plot_data.get("dataset_order")
    if not isinstance(method_order, list) or not method_order:
        raise ValueError(f"`method_order` must be a non-empty list in {path}.")
    if any(not isinstance(method, str) or not method for method in method_order):
        raise ValueError(f"`method_order` contains an invalid method in {path}.")
    if not isinstance(dataset_order, list) or not dataset_order:
        raise ValueError(f"`dataset_order` must be a non-empty list in {path}.")
    if any(not isinstance(dataset, str) or dataset not in DATASET_CONFIG for dataset in dataset_order):
        raise ValueError(f"`dataset_order` contains an unsupported dataset in {path}.")

    step_range = plot_data.get("step_range")
    if not isinstance(step_range, dict):
        raise ValueError(f"Missing `step_range` object in {path}.")
    try:
        start_step = int(step_range["start"])
        end_step = int(step_range["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `step_range` in {path}.") from exc
    if start_step < 0 or end_step < start_step:
        raise ValueError(f"Invalid step range in {path}: {start_step}-{end_step}.")
    expected_steps = list(range(start_step, end_step + 1))

    datasets = plot_data.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Missing `datasets` object in {path}.")

    for dataset in dataset_order:
        dataset_data = datasets.get(dataset)
        if not isinstance(dataset_data, dict):
            raise ValueError(f"Missing dataset data for `{dataset}` in {path}.")
        methods = dataset_data.get("methods")
        if not isinstance(methods, dict):
            raise ValueError(f"Missing methods for dataset `{dataset}` in {path}.")

        for method_name in method_order:
            method_data = methods.get(method_name)
            if not isinstance(method_data, dict):
                raise ValueError(
                    f"Missing method `{method_name}` for dataset `{dataset}` in {path}."
                )
            if not isinstance(method_data.get("label"), str):
                raise ValueError(
                    f"Missing label for `{dataset}/{method_name}` in {path}."
                )
            if not isinstance(method_data.get("style"), dict):
                raise ValueError(
                    f"Missing style for `{dataset}/{method_name}` in {path}."
                )
            points = method_data.get("points")
            if not isinstance(points, list) or len(points) != len(expected_steps):
                raise ValueError(
                    f"Invalid point count for `{dataset}/{method_name}` in {path}: "
                    f"expected {len(expected_steps)}."
                )
            for expected_step, point in zip(expected_steps, points):
                if not isinstance(point, dict) or point.get("step") != expected_step:
                    raise ValueError(
                        f"Invalid step sequence for `{dataset}/{method_name}` in {path}."
                    )
                value = point.get("value")
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                ):
                    raise ValueError(
                        f"Non-numeric value for `{dataset}/{method_name}` in {path}."
                    )

        baseline = dataset_data.get("baseline")
        if not isinstance(baseline, dict):
            raise ValueError(f"Missing baseline for dataset `{dataset}` in {path}.")
        resolved_value = baseline.get("resolved_value")
        if resolved_value is not None and (
            isinstance(resolved_value, bool)
            or not isinstance(resolved_value, (int, float))
        ):
            raise ValueError(f"Invalid baseline for dataset `{dataset}` in {path}.")


def find_plot_data_files(raw_path: str | None) -> list[Path]:
    """Resolve one plot_data.json or all model files under a plot-data root."""
    root = (
        resolve_repo_path(raw_path)
        if raw_path
        else get_output_root(None, None)
    )
    if root.is_file():
        if root.name != "plot_data.json":
            raise ValueError(f"JSON input must be named plot_data.json: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Plot-data path does not exist: {root}")

    files = sorted(root.glob("*/plot_data.json"))
    if not files:
        raise FileNotFoundError(f"No model plot_data.json files found under {root}")
    return files


def plot_from_data_file(data_path: Path, output_root: Path | None) -> None:
    """Render one already-persisted plot_data.json without reading eval outputs."""
    plot_data = load_summary(data_path)
    validate_plot_data(plot_data, data_path)

    model_tag = str(plot_data["model_tag"])
    step_range = plot_data["step_range"]
    start_step = int(step_range["start"])
    end_step = int(step_range["end"])
    datasets = plot_data["datasets"]
    output_dir = (output_root / model_tag) if output_root is not None else data_path.parent

    LOGGER.info("reading persisted plot data: %s", data_path)
    for dataset in plot_data["dataset_order"]:
        dataset_data = datasets[dataset]
        series_bundle = []
        for method_name in plot_data["method_order"]:
            method_data = dataset_data["methods"][method_name]
            series_bundle.append(
                {
                    "label": method_data["label"],
                    "style": dict(method_data["style"]),
                    "y": [point["value"] for point in method_data["points"]],
                }
            )

        resolved_value = dataset_data["baseline"]["resolved_value"]
        baselines = (
            [{"label": "step0 baseline", "color": "#111827", "value": resolved_value}]
            if resolved_value is not None
            else []
        )
        saved = plot_dataset(
            dataset,
            series_bundle,
            baselines,
            model_tag,
            start_step,
            end_step,
            output_dir,
        )
        if saved is not None:
            LOGGER.info("Saved plot: %s", saved)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically plot TTRL, LiveCodeBench, and CRUXEval curves for model checkpoints.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("outputs", "plot-data"),
        default="outputs",
        help="Read raw eval outputs (default) or persisted plot_data.json files.",
    )
    parser.add_argument(
        "--plot-data-root",
        type=str,
        default=None,
        help="One plot_data.json or a root containing <model_tag>/plot_data.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output root override. Images will be written under <output-dir>/<model_tag>/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    if plt is None:
        LOGGER.error(
            "matplotlib is required for plotting but is not installed in this Python environment: %s",
            MATPLOTLIB_IMPORT_ERROR,
        )
        LOGGER.error(
            "Install it first, then rerun. Example: `pip install matplotlib` "
            "or install it into your conda environment."
        )
        return 2

    if args.source_mode == "plot-data":
        output_root = resolve_repo_path(args.output_dir) if args.output_dir else None
        for data_path in find_plot_data_files(args.plot_data_root):
            plot_from_data_file(data_path, output_root)
        LOGGER.info("Done.")
        return 0

    output_group = "mixed_reward"
    eval_root = None  # None -> auto: <co-learning>/outputs/<output_group>/eval

    model_tags = [
        "qwen25_15b_step30",
        "qwen25_15b_lr3em6_step30",

        "smollm3_3b_step30",
        "smollm3_3b_lr3em6_step30",

        "qwen3_4b_base_step30",
        "qwen3_4b_base_lr3em6_step30",

        "deepseek_r1_distill_qwen_15b_step30",
        "deepseek_r1_distill_qwen_15b_lr3em6_step30",

        "qwen25_3b_step30",
        "qwen25_3b_lr3em6_step30",

        "qwen3_17b_base_step30",
        "qwen3_17b_base_lr3em6_step30",

        # Five models, two learning rates, 58 pipeline steps.
        "qwen25_15b_lr1em6_step58",
        "qwen25_15b_lr3em6_step58",
        "qwen3_4b_base_lr1em6_step58",
        "qwen3_4b_base_lr3em6_step58",
        "deepseek_r1_distill_qwen_15b_lr1em6_step58",
        "deepseek_r1_distill_qwen_15b_lr3em6_step58",
        "qwen25_3b_lr1em6_step58",
        "qwen25_3b_lr3em6_step58",
        "qwen3_17b_base_lr1em6_step58",
        "qwen3_17b_base_lr3em6_step58",

        # Rollout-count comparison: rollout=7 and rollout=14.
        "qwen25_15b_rollout7",
        "qwen25_15b_rollout14",
        "qwen3_17b_base_rollout7",
        "qwen3_17b_base_rollout14",
        "qwen3_4b_base_rollout7",
        "qwen3_4b_base_rollout14",
        "deepseek_r1_distill_qwen_15b_rollout7",
        "deepseek_r1_distill_qwen_15b_rollout14",
        "qwen25_3b_rollout7",
        "qwen25_3b_rollout14",

        # Rollout-count comparison with KL beta=0.001.
        "qwen25_15b_rollout7_kl",
        "qwen25_15b_rollout14_kl",
        "qwen3_17b_base_rollout7_kl",
        "qwen3_17b_base_rollout14_kl",
        "qwen3_4b_base_rollout7_kl",
        "qwen3_4b_base_rollout14_kl",
        "deepseek_r1_distill_qwen_15b_rollout7_kl",
        "deepseek_r1_distill_qwen_15b_rollout14_kl",
        "qwen25_3b_rollout7_kl",
        "qwen25_3b_rollout14_kl",

        # Rollout=7, KL beta=0.001, four optimizer updates per pipeline step.
        "qwen25_15b_rollout7_kl_u4_lr1em6",
        "qwen25_15b_rollout7_kl_u4_lr3em6",
        "qwen3_17b_base_rollout7_kl_u4_lr1em6",
        "qwen3_17b_base_rollout7_kl_u4_lr3em6",
        "qwen3_4b_base_rollout7_kl_u4_lr1em6",
        "qwen3_4b_base_rollout7_kl_u4_lr3em6",
        "deepseek_r1_distill_qwen_15b_rollout7_kl_u4_lr1em6",
        "deepseek_r1_distill_qwen_15b_rollout7_kl_u4_lr3em6",
        "qwen25_3b_rollout7_kl_u4_lr1em6",
        "qwen25_3b_rollout7_kl_u4_lr3em6",

        # "qwen25_3b",
        # "deepseek_r1_distill_qwen_15b",


        # "qwen3_17b_base",
        # "qwen3_17b_base_beta2em4_lr3em6",
        
        # "qwen25_15b",
        # "qwen25_15b_beta2em4_lr3em6",
        # "qwen3_4b_base",
        # "qwen25_3b",
        # "deepseek_r1_distill_qwen_15b"

        # "smollm3_3b",
        # "qwen35_2b_base",
        # "qwen35_2b_base_beta2em4_lr3em6",

        # "smollm2_17b_inst",
        # "smollm2_17b_inst_beta2em4_lr3em6",
        # "smollm2_17b_base",
        # "smollm2_17b_base_beta2em4_lr3em6",
        # "llama32_1b_base",
        # "llama32_1b_base_beta2em4_lr3em6",
        # "llama32_1b_inst",
        # "llama32_1b_inst_beta2em4_lr3em6",
        # "llama32_3b_base",
        # "llama32_3b_inst",
        # "phi4_mini_inst",

    ]

    selected_methods = [
        "single-intuitor-gspo",
        "single-mv-gspo",
        "single-mixed-reward-reliability-baseline-gspo",
        # "repro-gspo",

        # "single-mixed-reward-alpha025-gspo",

        # "single-mixed-reward-alpha075-gspo",

        # "single-mixed-reward-adaptive-sc-zscore-gspo",
        # "single-mixed-reward-adaptive-sc-zscore-corr-gspo",

        # "repro-gspo-filter2x",

        # "single-mixed-reward-alpha050-gspo",

        "single-mixed-reward-reliability-05sc-05corr-all-gspo",
        "single-mixed-reward-reliability-05sc-05corr-sample-only-gspo",
        "single-mixed-reward-reliability-05sc-05corr-advantage-only-gspo",

        # "single-mixed-reward-reliability-06sc-04corr-all-gspo",
        # "single-mixed-reward-reliability-04sc-06corr-all-gspo",

        # "single-mixed-reward-reliability-06sc-04corr-sample-only-gspo",
        # "single-mixed-reward-reliability-04sc-06corr-sample-only-gspo",

        # "single-mixed-reward-reliability-06sc-04corr-advantage-only-gspo",
        # "single-mixed-reward-reliability-04sc-06corr-advantage-only-gspo",

        "single-mixed-reward-reliability-05sc-05corr-all-sigma01-gspo",
        "single-mixed-reward-reliability-05sc-05corr-sample-only-sigma01-gspo",
        "single-mixed-reward-reliability-05sc-05corr-advantage-only-sigma01-gspo",
        "single-mixed-reward-reliability-03sc-07corr-all-sigma01-gspo",
        "single-mixed-reward-reliability-04sc-06corr-all-sigma01-gspo",
        "single-mixed-reward-reliability-06sc-04corr-all-sigma01-gspo",
    ]

    datasets = [
        "math_500",
        "gsm8k",
        # "aime24",
        # "aime25",
        # "amc23",
        # "lcb_release_v6",
        # "cruxeval",
    ]
    dataset_method_suffixes = {
        "math_500": "",
        "gsm8k": "",
        "aime24": "",
        "aime25": "",
        "amc23": "",
        "lcb_release_v6": "",
        "cruxeval": "",
    }
    start_step = 11
    start_step = 1
    end_step = 30
    # end_step = 58
    include_step0_baseline = False
    include_step0_baseline = True
    output_dir = None  # None -> auto: <co-learning>/scripts/plotting/output/auto_new

    validate_runtime_settings(
        model_tags,
        selected_methods,
        datasets,
        start_step,
        end_step,
        dataset_method_suffixes,
    )

    eval_root_path = get_eval_root(output_group, eval_root)
    method_entries = prepare_selected_methods(selected_methods)

    LOGGER.info("co-learning root: %s", get_co_learning_root())
    LOGGER.info("eval root: %s", eval_root_path)
    LOGGER.info("output root: %s", get_output_root(output_dir, args.output_dir))
    LOGGER.info("model tags: %s", ", ".join(model_tags))
    LOGGER.info("methods: %s", ", ".join(selected_methods))
    LOGGER.info("datasets: %s", ", ".join(datasets))

    for model_tag in model_tags:
        model_output_dir = get_model_output_dir(model_tag, output_dir, args.output_dir)
        LOGGER.info("plotting model tag: %s", model_tag)
        LOGGER.info("model output dir: %s", model_output_dir)
        plot_data: dict[str, Any] = {
            "schema_version": 1,
            "model_tag": model_tag,
            "eval_root": serialize_source_path(eval_root_path),
            "method_order": list(selected_methods),
            "dataset_order": list(datasets),
            "include_step0_baseline": include_step0_baseline,
            "dataset_method_suffixes": dict(dataset_method_suffixes),
            "step_range": {
                "start": start_step,
                "end": end_step,
            },
            "datasets": {},
        }

        for dataset in datasets:
            method_dir_suffix = dataset_method_suffixes.get(dataset, "")
            baseline_candidates: list[dict[str, Any]] = []
            dataset_data: dict[str, Any] = {
                "backend": get_dataset_config(dataset)["backend"],
                "methods": {},
                "baseline": {
                    "resolved_value": None,
                    "candidates": [],
                },
            }

            for method_entry in method_entries:
                method_name = str(method_entry["name"])
                label = str(method_entry["label"])
                style = dict(method_entry["style"])
                _, points = load_series_with_details(
                    eval_root_path,
                    model_tag,
                    method_name,
                    dataset,
                    start_step,
                    end_step,
                    method_dir_suffix,
                )
                dataset_data["methods"][method_name] = {
                    "label": label,
                    "style": style,
                    "points": points,
                }

                baseline_detail = load_baseline_detail(
                    eval_root_path,
                    model_tag,
                    method_name,
                    dataset,
                    include_step0_baseline,
                    method_dir_suffix,
                )
                baseline_candidates.append(
                    {
                        "method": method_name,
                        "label": label,
                        "color": style["color"],
                        "value": baseline_detail.get("value"),
                        "source": baseline_detail.get("source"),
                        "source_root": baseline_detail.get("source_root"),
                        "source_kind": baseline_detail.get("source_kind"),
                        "status": baseline_detail.get("status"),
                        **(
                            {"error": baseline_detail["error"]}
                            if baseline_detail.get("error")
                            else {}
                        ),
                    }
                )

            baselines = resolve_shared_baseline(dataset, baseline_candidates)
            dataset_data["baseline"] = {
                "resolved_value": baselines[0]["value"] if baselines else None,
                "candidates": baseline_candidates,
            }
            plot_data["datasets"][dataset] = dataset_data

        data_path = write_plot_data(model_output_dir, plot_data)
        LOGGER.info("Saved plot data: %s", data_path)

        # Plot only after the JSON has been written, so the image is generated
        # from the same persisted points that can be audited or reused later.
        persisted_plot_data = load_summary(data_path)
        persisted_step_range = persisted_plot_data["step_range"]
        persisted_start_step = int(persisted_step_range["start"])
        persisted_end_step = int(persisted_step_range["end"])
        for dataset in persisted_plot_data["dataset_order"]:
            dataset_data = persisted_plot_data["datasets"][dataset]
            series_bundle = []
            for method_name, method_data in dataset_data["methods"].items():
                points = method_data["points"]
                series_bundle.append(
                    {
                        "label": method_data["label"],
                        "style": method_data["style"],
                        "y": [point["value"] for point in points],
                    }
                )

            resolved_value = dataset_data["baseline"]["resolved_value"]
            baselines = (
                [
                    {
                        "label": "step0 baseline",
                        "color": "#111827",
                        "value": resolved_value,
                    }
                ]
                if resolved_value is not None
                else []
            )
            saved = plot_dataset(
                dataset,
                series_bundle,
                baselines,
                model_tag,
                persisted_start_step,
                persisted_end_step,
                model_output_dir,
            )
            if saved is not None:
                LOGGER.info("Saved plot: %s", saved)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
