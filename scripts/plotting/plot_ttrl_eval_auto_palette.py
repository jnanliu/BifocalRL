#!/usr/bin/env python3
"""Plot the mixed-reward curves with a fixed, distinct-color palette.

The original ``plot_ttrl_eval_auto_new.py`` remains unchanged by this variant.
This wrapper reuses its data loading and plotting behavior, including the
dynamic ``loc='best'`` legend placement, and only replaces method styles.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plotting import plot_ttrl_eval_auto_new as base  # noqa: E402


METHOD_STYLES: dict[str, dict[str, Any]] = {
    # Baselines: neutral/dark colors, kept separate from the reliability families.
    "single-intuitor-gspo": {
        "color": "#111827",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.5,
        "markersize": 5,
        "markevery": 3,
    },
    "single-mv-gspo": {
        "color": "#C026D3",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.5,
        "markersize": 5,
        "markevery": 3,
    },
    "repro-gspo": {
        "color": "#6C757D",
        "linestyle": "-.",
        "marker": "P",
        "linewidth": 2.5,
        "markersize": 5,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-baseline-gspo": {
        "color": "#BE123C",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 3.0,
        "markersize": 5,
        "markevery": 3,
    },

    # Hue identifies the variant type; shade identifies the SC/Corr ratio.
    # Within each family, 0.6/0.4 is darkest, then 0.5/0.5, then 0.4/0.6.
    "single-mixed-reward-reliability-06sc-04corr-all-gspo": {
        "color": "#0B3C8C",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-06sc-04corr-advantage-only-gspo": {
        "color": "#9A3412",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-06sc-04corr-sample-only-gspo": {
        "color": "#0B5D4B",
        "linestyle": ":",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },

    "single-mixed-reward-reliability-05sc-05corr-all-gspo": {
        "color": "#2774C5",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-05sc-05corr-advantage-only-gspo": {
        "color": "#D97706",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-05sc-05corr-sample-only-gspo": {
        "color": "#238B70",
        "linestyle": ":",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-05sc-05corr-all-sigma01-gspo": {
        "color": "#7B3294",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-06sc-04corr-all-sigma01-gspo": {
        "color": "#4A1486",
        "linestyle": "-",
        "marker": "P",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-04sc-06corr-all-sigma01-gspo": {
        "color": "#A66BB7",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-03sc-07corr-all-sigma01-gspo": {
        "color": "#D8B4E2",
        "linestyle": "-",
        "marker": "X",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-05sc-05corr-advantage-only-sigma01-gspo": {
        "color": "#E7298A",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-05sc-05corr-sample-only-sigma01-gspo": {
        "color": "#66A61E",
        "linestyle": ":",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },

    "single-mixed-reward-reliability-04sc-06corr-all-gspo": {
        "color": "#81B7E8",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-04sc-06corr-advantage-only-gspo": {
        "color": "#FDBA74",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
    "single-mixed-reward-reliability-04sc-06corr-sample-only-gspo": {
        "color": "#78C9A5",
        "linestyle": ":",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 4,
        "markevery": 3,
    },
}


def validate_palette(selected_methods: list[str]) -> None:
    missing = [method for method in selected_methods if method not in METHOD_STYLES]
    if missing:
        raise ValueError("Missing palette styles for methods: " + ", ".join(missing))

    colors = [METHOD_STYLES[method]["color"].strip().lower() for method in selected_methods]
    if len(colors) != len(set(colors)):
        raise ValueError("The selected plotting methods must use distinct colors.")


def pick_method_style(method_name: str) -> dict[str, Any]:
    try:
        style = dict(METHOD_STYLES[method_name])
    except KeyError as exc:
        raise ValueError(f"Missing palette style for method: {method_name}") from exc
    style.setdefault("linewidth", 2.2)
    style.setdefault("markersize", 5)
    style.setdefault("alpha", 1.0)
    return style


def prepare_selected_methods(selected_methods: list[str]) -> list[dict[str, Any]]:
    validate_palette(selected_methods)
    return [
        {
            "name": method_name,
            "label": base.get_method_label(method_name),
            "style": pick_method_style(method_name),
        }
        for method_name in selected_methods
    ]


_base_validate_runtime_settings = base.validate_runtime_settings


def validate_runtime_settings(
    model_tags: list[str],
    selected_methods: list[str],
    datasets: list[str],
    start_step: int,
    end_step: int,
    dataset_method_suffixes: dict[str, str] | None = None,
) -> None:
    _base_validate_runtime_settings(
        model_tags,
        selected_methods,
        datasets,
        start_step,
        end_step,
        dataset_method_suffixes,
    )
    validate_palette(selected_methods)


base.prepare_selected_methods = prepare_selected_methods
base.validate_runtime_settings = validate_runtime_settings


def main() -> int:
    # Keep palette output separate unless the caller explicitly chooses a root.
    if "--output-dir" not in sys.argv[1:]:
        sys.argv.extend(["--output-dir", "scripts/plotting/output/auto_palette"])
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
