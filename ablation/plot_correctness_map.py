#!/usr/bin/env python
# coding=utf-8
"""
Per-problem correctness heatmap from batch evaluation results.

Parses two evaluation outputs produced by ``run_batch_ttrl_eval.sh``:

1) TTRL eval JSON:
   <results>/<model>/ttrl/.../<dataset>/results_*_model-eval.json

2) LightEval details parquet:
   <results>/<model>/lighteval/**/details_*.parquet

Then generates one heatmap per dataset where each row is a
``(model, method)`` pair and each column is a problem index.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_correctness_map")


def _task_to_dataset(task_name: str) -> str:
    t = str(task_name).strip()
    if "|" in t:
        parts = t.split("|")
        if len(parts) >= 2:
            t = parts[1]
    mapping = {
        "local_math_500": "math_500",
        "local_gsm8k": "gsm8k",
    }
    return mapping.get(t, t)


def _load_ttrl_correctness(filepath: str) -> Optional[Dict[str, Any]]:
    """Load TTRL result JSON and extract per-problem correctness + summary."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", filepath, e)
        return None

    problems = data.get("problems", [])
    summary = data.get("summary", {})
    correctness = [bool(p.get("any_correct", False)) for p in problems]
    return {
        "correctness": correctness,
        "accuracy": summary.get("pass_at_k", 0.0),
        "total": summary.get("total", len(correctness)),
        "correct": summary.get("correct_any", sum(correctness)),
        "scorer": summary.get("scorer", ""),
    }


def _load_parquet_rows(parquet_path: Path) -> Optional[List[dict]]:
    """Best-effort parquet loader (pyarrow first, pandas fallback)."""
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(parquet_path)
        return table.to_pylist()
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(parquet_path)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning("Failed to read parquet %s: %s", parquet_path, e)
        return None


def _iter_numeric_values(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        yield 1.0 if obj else 0.0
    elif isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_numeric_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_numeric_values(v)


def _extract_row_correct_flag(row: dict) -> Optional[bool]:
    """Extract a binary correctness flag from one LightEval detail row."""
    metric_obj = row.get("__metric__")
    if metric_obj is None:
        return None
    nums = list(_iter_numeric_values(metric_obj))
    if not nums:
        return None
    # Most tasks produce a single binary-style metric; if multiple exist,
    # use the max as "any metric judged this sample correct".
    return max(nums) >= 0.5


def _extract_task_from_details_filename(path: Path) -> str:
    stem = path.stem  # details_lighteval|gsm8k|0_2026-...
    if not stem.startswith("details_"):
        return "unknown"
    body = stem[len("details_") :]
    # split timestamp suffix conservatively
    task = body.rsplit("_", 1)[0] if "_" in body else body
    return _task_to_dataset(task)


def _load_lighteval_correctness(parquet_path: Path) -> Optional[Dict[str, Any]]:
    rows = _load_parquet_rows(parquet_path)
    if not rows:
        return None
    correctness: List[bool] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        c = _extract_row_correct_flag(row)
        if c is not None:
            correctness.append(bool(c))
    if not correctness:
        return None
    total = len(correctness)
    correct = int(sum(1 for x in correctness if x))
    return {
        "correctness": correctness,
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "scorer": "lighteval",
    }


def _plot_heatmap(
    rows: List[Dict[str, Any]],
    dataset: str,
    output_path: str,
) -> None:
    """Generate and save a correctness heatmap for one dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        logger.error("matplotlib/numpy not installed — skipping plot for %s", dataset)
        return

    if not rows:
        logger.warning("No data for dataset %s — skipping", dataset)
        return

    n_problems = max(len(r["correctness"]) for r in rows)
    n_rows = len(rows)

    matrix = np.zeros((n_rows, n_problems), dtype=float)
    labels: List[str] = []
    for i, r in enumerate(rows):
        corr = r["correctness"]
        for j, c in enumerate(corr):
            matrix[i, j] = 1.0 if c else 0.0
        acc = r["accuracy"]
        n_correct = r["correct"]
        n_total = r["total"]
        labels.append(f"{r['label']}  ({n_correct}/{n_total} = {acc:.1%})")

    cmap = mcolors.ListedColormap(["#FFCCCC", "#66CC66"])
    bounds = [0, 0.5, 1.0]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_width = max(12, n_problems * 0.04 + 4)
    fig_height = max(3, n_rows * 0.6 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")

    n_xticks = min(n_problems, 20)
    step = max(1, n_problems // n_xticks)
    xtick_positions = list(range(0, n_problems, step))
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(x) for x in xtick_positions], fontsize=7)
    ax.set_xlabel("Problem Index", fontsize=9)

    ax.set_title(f"Per-Problem Correctness — {dataset}", fontsize=11, fontweight="bold")

    per_problem_acc = matrix.mean(axis=0)
    ax2 = ax.twinx()
    ax2.plot(range(n_problems), per_problem_acc, color="#3366AA", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Solve Rate (across rows)", fontsize=8, color="#3366AA")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelsize=7, colors="#3366AA")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_difficulty_ranking(
    rows: List[Dict[str, Any]],
    dataset: str,
    output_path: str,
) -> None:
    """Bar chart of per-problem solve rate, sorted from easiest to hardest."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if not rows:
        return

    n_problems = max(len(r["correctness"]) for r in rows)
    matrix = np.zeros((len(rows), n_problems), dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(r["correctness"]):
            matrix[i, j] = 1.0 if c else 0.0

    solve_rates = matrix.mean(axis=0)
    sorted_indices = np.argsort(solve_rates)[::-1]
    sorted_rates = solve_rates[sorted_indices]

    fig_width = max(10, n_problems * 0.03 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    colors = ["#66CC66" if r >= 0.5 else "#FF9966" if r > 0 else "#FF6666" for r in sorted_rates]
    ax.bar(range(n_problems), sorted_rates, color=colors, width=1.0, edgecolor="none")

    ax.set_xlabel("Problems (sorted by solve rate)", fontsize=9)
    ax.set_ylabel("Solve Rate", fontsize=9)
    ax.set_title(f"Problem Difficulty — {dataset}", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, n_problems - 0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    n_all_solved = int((sorted_rates == 1.0).sum())
    n_none_solved = int((sorted_rates == 0.0).sum())
    ax.text(
        0.98, 0.95,
        f"All correct: {n_all_solved}  |  None correct: {n_none_solved}  |  Total: {n_problems}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-problem correctness heatmaps.")
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Root directory of batch evaluation results.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots. Default: <results_dir>/plots",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")

    root = Path(args.results_dir)
    if not root.exists():
        logger.warning("Results dir not found: %s", args.results_dir)
        return

    # Group by dataset
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    csv_rows: List[Tuple[str, str, str, str, int, int, float, str]] = []

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "plots":
            continue
        model_tag = model_dir.name

        # ---- TTRL JSON ----
        ttrl_dir = model_dir / "ttrl"
        if ttrl_dir.exists():
            for json_path in sorted(ttrl_dir.rglob("results_*_model-eval.json")):
                dataset = json_path.parent.name
                info = _load_ttrl_correctness(str(json_path))
                if info is None:
                    continue
                info["label"] = f"{model_tag} [ttrl]"
                info["model_tag"] = model_tag
                info["method"] = "ttrl"
                by_dataset[dataset].append(info)
                csv_rows.append(
                    (
                        model_tag,
                        "ttrl",
                        dataset,
                        str(info["scorer"]),
                        int(info["correct"]),
                        int(info["total"]),
                        float(info["accuracy"]),
                        str(json_path),
                    )
                )

        # ---- LightEval details parquet ----
        le_dir = model_dir / "lighteval"
        if le_dir.exists():
            for pq_path in sorted(le_dir.rglob("details_*.parquet")):
                dataset = _extract_task_from_details_filename(pq_path)
                info = _load_lighteval_correctness(pq_path)
                if info is None:
                    continue
                info["label"] = f"{model_tag} [lighteval]"
                info["model_tag"] = model_tag
                info["method"] = "lighteval"
                by_dataset[dataset].append(info)
                csv_rows.append(
                    (
                        model_tag,
                        "lighteval",
                        dataset,
                        str(info["scorer"]),
                        int(info["correct"]),
                        int(info["total"]),
                        float(info["accuracy"]),
                        str(pq_path),
                    )
                )

    if not csv_rows:
        logger.warning("No parseable TTRL/LightEval result files under %s", args.results_dir)
        return

    for dataset, rows in sorted(by_dataset.items()):
        rows.sort(key=lambda r: (r["method"], r["model_tag"]))
        logger.info("Dataset '%s': %d rows", dataset, len(rows))

        heatmap_path = os.path.join(output_dir, f"correctness_{dataset}.png")
        _plot_heatmap(rows, dataset, heatmap_path)

        difficulty_path = os.path.join(output_dir, f"difficulty_{dataset}.png")
        _plot_difficulty_ranking(rows, dataset, difficulty_path)

    # Save a combined summary CSV for easy consumption
    csv_path = os.path.join(output_dir, "summary.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,method,dataset,scorer,correct,total,accuracy,source_path\n")
        for model_tag, method, dataset, scorer, correct, total, accuracy, src in csv_rows:
            f.write(
                f"{model_tag},{method},{dataset},{scorer},{correct},{total},{accuracy:.4f},{src}\n"
            )
    logger.info("Summary CSV: %s", csv_path)


if __name__ == "__main__":
    main()
