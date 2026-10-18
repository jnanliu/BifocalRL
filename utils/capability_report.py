"""
Structured capability-ratio reporting.

Writes a per-step CSV and an all-steps JSON summary so that capability ratio
diagnostics are easy to inspect without digging through cache.json.

Usage (called from co_intuitor_train.py):

    from utils.capability_report import CapabilityReporter

    reporter = CapabilityReporter(output_dir)
    reporter.record_step(step, reward_stats, raw_ratios, clipped_ratios,
                         applied_weights, mode, smoothing, stable_perf_applied)
    ...
    reporter.write_summary()
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CSV_FILENAME = "capability_report.csv"
_SUMMARY_FILENAME = "capability_summary.json"

_FIELDNAMES = [
    "step",
    "reward_mean_a", "reward_mean_b",
    "reward_var_a", "reward_var_b",
    "cap_raw_a_to_b", "cap_raw_b_to_a",
    "cap_clipped_a_to_b", "cap_clipped_b_to_a",
    "cap_applied_a_to_b", "cap_applied_b_to_a",
    "cap_delta_a", "cap_delta_b",
    "mode", "smoothing", "stable_perf_applied",
]


class CapabilityReporter:
    """Append-only CSV writer + summary generator for capability ratios."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, _CSV_FILENAME)
        self.summary_path = os.path.join(output_dir, _SUMMARY_FILENAME)
        os.makedirs(output_dir, exist_ok=True)

    def record_step(
        self,
        step: int,
        reward_stats: Dict[str, Dict[str, float]],
        raw_ratios: Dict[str, Dict[str, float]],
        clipped_ratios: Dict[str, Dict[str, float]],
        applied_weights: Dict[str, Dict[str, float]],
        mode: str,
        smoothing: str,
        stable_perf_applied: bool,
    ) -> Dict[str, Any]:
        """Append one row to the CSV and return the row dict."""
        row = {
            "step": int(step),
            "reward_mean_a": reward_stats.get("a", {}).get("mean", 0.0),
            "reward_mean_b": reward_stats.get("b", {}).get("mean", 0.0),
            "reward_var_a": reward_stats.get("a", {}).get("var", 0.0),
            "reward_var_b": reward_stats.get("b", {}).get("var", 0.0),
            "cap_raw_a_to_b": raw_ratios.get("a", {}).get("b", 1.0),
            "cap_raw_b_to_a": raw_ratios.get("b", {}).get("a", 1.0),
            "cap_clipped_a_to_b": clipped_ratios.get("a", {}).get("b", 1.0),
            "cap_clipped_b_to_a": clipped_ratios.get("b", {}).get("a", 1.0),
            "cap_applied_a_to_b": applied_weights.get("a", {}).get("b", 1.0),
            "cap_applied_b_to_a": applied_weights.get("b", {}).get("a", 1.0),
            "cap_delta_a": applied_weights.get("a", {}).get("b", 1.0) - 1.0,
            "cap_delta_b": applied_weights.get("b", {}).get("a", 1.0) - 1.0,
            "mode": mode,
            "smoothing": smoothing,
            "stable_perf_applied": stable_perf_applied,
        }

        write_header = not os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.warning("Failed to write capability report row: %s", e)

        return row

    def write_summary(self) -> Optional[Dict[str, Any]]:
        """Read the CSV and write a JSON summary with per-column mean/std."""
        if not os.path.exists(self.csv_path):
            logger.info("No capability report CSV found, skipping summary.")
            return None

        try:
            rows = self._read_csv()
        except Exception as e:
            logger.warning("Failed to read capability report CSV: %s", e)
            return None

        if not rows:
            return None

        numeric_keys = [k for k in _FIELDNAMES if k not in ("mode", "smoothing", "stable_perf_applied")]
        summary: Dict[str, Any] = {"total_steps": len(rows), "columns": {}}

        for key in numeric_keys:
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
            summary["columns"][key] = {
                "mean": round(mean, 6),
                "std": round(var ** 0.5, 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
                "count": n,
            }

        try:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info("Capability summary written to %s", self.summary_path)
        except Exception as e:
            logger.warning("Failed to write capability summary: %s", e)

        return summary

    def _read_csv(self) -> List[Dict[str, str]]:
        with open(self.csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
