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


# Edit this config directly for your experiment.
CONFIG: dict[str, Any] = {
    "output_group": "mixed_reward",
    "eval_root": None,  # None -> auto: <co-learning>/outputs/<output_group>/eval
    # "model_tag": "qwen3_17b_base",
    # "model_tag": "qwen3_4b_base",
    # "model_tag": "qwen25_3b",
    # "model_tag": "qwen25_math_15b",
    # "model_tag": "llama_3b",
    # "model_tag": "phi4",
    # "model_tag": "qwen25_3b_inst",
    # "model_tag": "qwen25_15b_inst",
    # "model_tag": "qwen25_15b",
    # "model_tag": "deepseek_r1_distill_qwen_15b",
    # "model_tag": "qwen3_4b_inst_2507",
    # "model_tag": "gemma2_2b",
    # "model_tag": "gemma2_2b_it",
    "model_tag": "smollm3_3b_base",
    # "model_tag": "smollm3_3b",
    "methods": [
        {
            "name": "single-intuitor-gspo",
            "label": "intuitor gspo",
            "style": {
                "color": "#6B7280",
                "linestyle": "-.",
                "marker": "P",
            },
        },
        {
            "name": "repro-gspo-filter2x",
            "label": "filter2x",
            "style": {
                "color": "#9CA3AF",
                "linestyle": "-",
                "marker": "P",
            },
        },
        # {
        #     "name": "single-intuitor-grpo",
        #     "label": "intuitor grpo",
        #     "style": {
        #         "color": "#9CA3AF",
        #         "linestyle": "-",
        #         "marker": "P",
        #     },
        # },
        # {
        #     "name": "single-mixed-reward-adaptive-sc-gspo",
        #     "label": "adaptive sc",
        #     "style": {
        #         "color": "#7C3AED",
        #         "linestyle": "--",
        #         "marker": "D",
        #     },
        # },
        {
            "name": "single-mixed-reward-alpha025-gspo",
            "label": "alpha=0.25",
            "style": {
                "color": "#059669",
                "linestyle": "-",
                "marker": "^",
            },
        },
        {
            "name": "single-mixed-reward-alpha050-gspo",
            "label": "alpha=0.50",
            "style": {
                "color": "#2563EB",
                "linestyle": "-",
                "marker": "o",
            },
        },
        {
            "name": "single-mixed-reward-alpha075-gspo",
            "label": "alpha=0.75",
            "style": {
                "color": "#DC2626",
                "linestyle": "-",
                "marker": "v",
            },
        },
        {
            "name": "single-mixed-reward-adaptive-sc-zscore-gspo",
            "label": "adaptive sc zscore",
            "style": {
                "color": "#7C3AED",
                "linestyle": "--",
                "marker": "D",
            },
        },
        {
            "name": "single-mixed-reward-adaptive-sc-zscore-corr-gspo",
            "label": "adaptive sc zscore + corr",
            "style": {
                "color": "#A16207",
                "linestyle": "-.",
                "marker": "h",
            },
        },
        {
            "name": "single-mv-gspo",
            "label": "mv gspo",
            "style": {
                "color": "#EA580C",
                "linestyle": ":",
                "marker": "X",
            },
        },
        # {
        #     "name": "single-mv-grpo",
        #     "label": "mv grpo",
        #     "style": {
        #         "color": "#F59E0B",
        #         "linestyle": "-",
        #         "marker": "X",
        #     },
        # },
        {
            "name": "repro-gspo",
            "label": "repro gspo",
            "style": {
                "color": "#111827",
                "linestyle": "--",
                "marker": "*",
            },
        },
        # {
        #     "name": "repro-grpo",
        #     "label": "repro grpo",
        #     "style": {
        #         "color": "#4B5563",
        #         "linestyle": "-",
        #         "marker": "*",
        #     },
        # },
    ],
    "datasets": ["math_500", "gsm8k"],
    "start_step": 1,
    "end_step": 15,
    "include_step0_baseline": True,
    "output_dir": None,  # None -> auto: <co-learning>/scripts/plotting/output/auto/<model_tag>
    "figure": {
        "size": (12, 6),
        "dpi": 220,
    },
    "y_axis": {
        "label": "Pass@1",
        "limits": None,  # e.g. (0.55, 0.85)
    },
    "title_template": "{model_tag} - {dataset}",
    "filename_template": "{model_tag}_{dataset}.png",
}


LOGGER = logging.getLogger("plot_ttrl_eval_auto")
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]
BASELINE_GROUP_DECIMALS = 12


def get_co_learning_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_eval_root(config: dict[str, Any]) -> Path:
    return get_co_learning_root() / "outputs" / str(config["output_group"]) / "eval"


def get_eval_root(config: dict[str, Any]) -> Path:
    eval_root = config.get("eval_root")
    if eval_root:
        path = Path(eval_root)
        if not path.is_absolute():
            path = get_co_learning_root() / path
        return path
    return get_default_eval_root(config)


def get_output_dir(config: dict[str, Any], override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()

    output_dir = config.get("output_dir")
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = get_co_learning_root() / path
        return path

    return get_co_learning_root() / "scripts" / "plotting" / "output" / "auto" / str(config["model_tag"])


def build_eval_method_root(config: dict[str, Any], method_name: str) -> Path:
    return get_eval_root(config) / str(config["model_tag"]) / method_name


def build_summary_path(method_root: Path, step: int) -> Path:
    return method_root / f"a_checkpoint-{step}" / "comparison_summary.json"


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Summary file is not a JSON object: {path}")
    return data


def validate_config(config: dict[str, Any]) -> None:
    methods = config.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("`CONFIG['methods']` must be a non-empty list.")
    for index, method_cfg in enumerate(methods):
        if not isinstance(method_cfg, dict) or not method_cfg.get("name"):
            raise ValueError(f"`CONFIG['methods'][{index}]` must be a dict with a non-empty `name`.")

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("`CONFIG['datasets']` must be a non-empty list.")

    start_step = int(config["start_step"])
    end_step = int(config["end_step"])
    if start_step < 0:
        raise ValueError("`CONFIG['start_step']` must be >= 0.")
    if end_step < start_step:
        raise ValueError("`CONFIG['end_step']` must be >= `CONFIG['start_step']`.")

    fig_cfg = config.get("figure", {})
    fig_size = fig_cfg.get("size", (12, 6))
    if not isinstance(fig_size, (list, tuple)) or len(fig_size) != 2:
        raise ValueError("`CONFIG['figure']['size']` must be a length-2 list or tuple.")

    y_limits = config.get("y_axis", {}).get("limits")
    if y_limits is not None:
        if not isinstance(y_limits, (list, tuple)) or len(y_limits) != 2:
            raise ValueError("`CONFIG['y_axis']['limits']` must be None or a length-2 list/tuple.")
        lower = float(y_limits[0])
        upper = float(y_limits[1])
        if lower >= upper:
            raise ValueError("`CONFIG['y_axis']['limits']` must satisfy lower < upper.")


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


def load_series(
    config: dict[str, Any],
    method_cfg: dict[str, Any],
    dataset: str,
    start_step: int,
    end_step: int,
) -> list[float | None]:
    method_name = str(method_cfg["name"])
    method_root = build_eval_method_root(config, method_name)
    if not method_root.exists():
        warn_once(f"Method directory not found, all points will be empty: {method_root}")
        return [None] * (end_step - start_step + 1)

    values: list[float | None] = []
    for step in range(start_step, end_step + 1):
        summary_path = build_summary_path(method_root, step)
        if not summary_path.exists():
            LOGGER.warning("Missing summary for %s step=%d dataset=%s", method_name, step, dataset)
            values.append(None)
            continue

        try:
            summary = load_summary(summary_path)
            value = extract_dataset_value(summary, step, dataset)
        except Exception as exc:
            LOGGER.warning(
                "Failed to extract dataset value for %s step=%d dataset=%s: %s",
                method_name,
                step,
                dataset,
                exc,
            )
            values.append(None)
            continue
        values.append(value)
    return values


def load_baseline(config: dict[str, Any], method_cfg: dict[str, Any], dataset: str) -> float | None:
    if not config.get("include_step0_baseline", True):
        return None

    method_name = str(method_cfg["name"])
    method_root = build_eval_method_root(config, method_name)
    summary_path = build_summary_path(method_root, 0)
    if not summary_path.exists():
        return None

    try:
        summary = load_summary(summary_path)
        return extract_dataset_value(summary, 0, dataset)
    except Exception as exc:
        LOGGER.warning(
            "Failed to extract step-0 baseline for %s dataset=%s: %s",
            method_name,
            dataset,
            exc,
        )
        return None


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


def get_method_label(method_cfg: dict[str, Any]) -> str:
    return str(method_cfg.get("label") or method_cfg["name"])


def pick_method_style(method_cfg: dict[str, Any], index: int) -> dict[str, Any]:
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()
    palette = prop_cycle.get("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    raw_style = method_cfg.get("style", {})
    style = dict(raw_style) if isinstance(raw_style, dict) else {}

    style.setdefault("color", palette[index % len(palette)])
    style.setdefault("linestyle", "-")
    style.setdefault("marker", MARKER_CYCLE[index % len(MARKER_CYCLE)])
    style.setdefault("linewidth", 2.2)
    style.setdefault("markersize", 5)
    style.setdefault("alpha", 1.0)
    return style


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
    config: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    start_step = int(config["start_step"])
    end_step = int(config["end_step"])
    x_values = list(range(start_step, end_step + 1))
    x_labels = [f"ckpt-{step}" for step in x_values]

    numeric_values: list[float] = []
    has_any_data = False

    fig_cfg = config.get("figure", {})
    fig_size = fig_cfg.get("size", (12, 6))
    fig_size = (float(fig_size[0]), float(fig_size[1]))
    dpi = int(fig_cfg.get("dpi", 220))
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
        LOGGER.warning("Skip dataset `%s`: no valid data points or baselines were found.", dataset)
        plt.close(fig)
        return None

    title_template = str(config.get("title_template", "{model_tag} - {dataset}"))
    title = title_template.format(model_tag=config["model_tag"], dataset=dataset)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Checkpoint", fontsize=11)

    y_axis_cfg = config.get("y_axis", {})
    ax.set_ylabel(str(y_axis_cfg.get("label", "Pass@1")), fontsize=11)
    y_limits = y_axis_cfg.get("limits")
    if isinstance(y_limits, (list, tuple)) and len(y_limits) == 2:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    else:
        apply_auto_y_limits(ax, numeric_values)

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename_template = str(config.get("filename_template", "{model_tag}_{dataset}.png"))
    filename = filename_template.format(model_tag=config["model_tag"], dataset=dataset)
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically plot TTRL comparison_summary.json curves for one model tag.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Optional model tag override. Example: qwen25_3b",
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

    if args.model_tag:
        CONFIG["model_tag"] = args.model_tag

    validate_config(CONFIG)

    output_dir = get_output_dir(CONFIG, args.output_dir)
    eval_root = get_eval_root(CONFIG)

    LOGGER.info("co-learning root: %s", get_co_learning_root())
    LOGGER.info("eval root: %s", eval_root)
    LOGGER.info("output dir: %s", output_dir)
    LOGGER.info("model tag: %s", CONFIG["model_tag"])

    start_step = int(CONFIG["start_step"])
    end_step = int(CONFIG["end_step"])

    for dataset in CONFIG["datasets"]:
        series_bundle: list[dict[str, Any]] = []
        baseline_candidates: list[dict[str, Any]] = []
        for index, method_cfg in enumerate(CONFIG["methods"]):
            label = get_method_label(method_cfg)
            style = pick_method_style(method_cfg, index)
            y_values = load_series(CONFIG, method_cfg, dataset, start_step, end_step)
            series_bundle.append(
                {
                    "label": label,
                    "style": style,
                    "y": y_values,
                }
            )

            baseline_value = load_baseline(CONFIG, method_cfg, dataset)
            baseline_candidates.append(
                {
                    "label": label,
                    "color": style["color"],
                    "value": baseline_value,
                }
            )

        baselines = resolve_shared_baseline(dataset, baseline_candidates)
        saved = plot_dataset(dataset, series_bundle, baselines, CONFIG, output_dir)
        if saved is not None:
            LOGGER.info("Saved plot: %s", saved)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
