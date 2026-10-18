#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")
    return data


def _ensure_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"`{name}` must be a list.")
    return value


def _pick_style(style: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "color" in style:
        out["color"] = style["color"]
    if "linestyle" in style:
        out["linestyle"] = style["linestyle"]
    if "linewidth" in style:
        out["linewidth"] = style["linewidth"]
    if "marker" in style:
        out["marker"] = style["marker"]
    if "markersize" in style:
        out["markersize"] = style["markersize"]
    if "alpha" in style:
        out["alpha"] = style["alpha"]
    if "zorder" in style:
        out["zorder"] = float(style["zorder"])
    return out


def _numeric(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _numeric_list(name: str, value: Any) -> list[float]:
    items = _ensure_list(name, value)
    out: list[float] = []
    for idx, item in enumerate(items):
        num = _numeric(item)
        if num is None:
            raise ValueError(f"`{name}[{idx}]` must be numeric.")
        out.append(num)
    return out


def _count_legend_entries(series_cfg: list[dict[str, Any]], baselines_cfg: list[dict[str, Any]]) -> int:
    """How many legend handles we will add (matches post-plot get_legend_handles_labels count)."""
    n = 0
    for line in series_cfg:
        if not isinstance(line, dict):
            continue
        y_raw = line.get("y", [])
        if not isinstance(y_raw, list) or not y_raw:
            continue
        has_point = False
        for v in y_raw:
            yv = _numeric(v)
            if yv is not None and not math.isnan(yv):
                has_point = True
                break
        if has_point:
            if line.get("include_in_legend", True) is False:
                continue
            n += 1
    for bl in baselines_cfg:
        if isinstance(bl, dict) and _numeric(bl.get("value")) is not None:
            n += 1
    return n


def _apply_y_tight_margin(ax: Any, all_y_points: list[float], y_cfg: dict[str, Any]) -> None:
    """Shrink y-axis to data range plus padding (opt-in via y.autoscale_tight)."""
    if not all_y_points:
        return
    y_lo = min(all_y_points)
    y_hi = max(all_y_points)
    span = y_hi - y_lo
    rel = float(y_cfg.get("autoscale_pad_relative", 0.12))
    abs_pad = float(y_cfg.get("autoscale_pad_absolute", 0.015))
    pad = max(span * rel, abs_pad)
    if span <= 1e-12:
        pad = max(abs_pad, 0.04)
    bot = y_lo - pad
    top = y_hi + pad
    clip = y_cfg.get("clip_01", True)
    if clip is True:
        bot = max(0.0, bot)
        top = min(1.0, top)
    if bot >= top:
        mid = 0.5 * (y_lo + y_hi)
        bot, top = mid - 0.04, mid + 0.04
        if clip is True:
            bot = max(0.0, bot)
            top = min(1.0, top)
    ax.set_ylim(bot, top)


def _effective_ncol_for_below(ncol_raw: int, n_entries: int) -> int:
    """Widen crowded below-axes legends when JSON still says ncol=1."""
    ncol_raw = max(1, ncol_raw)
    if n_entries < 7 or ncol_raw > 1:
        return ncol_raw
    return min(4, max(2, math.ceil(n_entries / 3)))


def _legend_dummy_cell() -> Line2D:
    """Invisible handle to pad the last row so baseline aligns under column 5."""
    return Line2D([], [], linestyle="none", marker="", linewidth=0, color="none", label="\u200b")


def _legend_pipeline_five_columns(
    ax: Any,
    handles: list[Any],
    labels: list[str],
    legend_kwargs: dict[str, Any],
    legend_resolved: dict[str, Any],
) -> Any:
    """5 columns × 3 rows: col i = pipeline i (orig row0, new row1); col4 row2 = baseline."""
    n_expect = 11
    if len(handles) != n_expect or len(labels) != n_expect:
        raise ValueError(
            f"legend layout 'pipeline_columns' needs {n_expect} series+baseline handles, "
            f"got {len(handles)}"
        )
    orig_h = [handles[2 * i] for i in range(5)]
    new_h = [handles[2 * i + 1] for i in range(5)]
    base_h = handles[10]
    orig_l = [labels[2 * i] for i in range(5)]
    new_l = [labels[2 * i + 1] for i in range(5)]
    base_l = labels[10]

    ordered_h = orig_h + new_h + [_legend_dummy_cell() for _ in range(4)] + [base_h]
    ordered_l = orig_l + new_l + ["\u200b"] * 4 + [base_l]

    kw = {**legend_kwargs, "ncol": 5}
    leg = ax.legend(ordered_h, ordered_l, **kw)
    title = legend_resolved.get("title")
    if title:
        leg.set_title(str(title))
        tfs = legend_resolved.get("title_fontsize")
        if tfs is not None:
            leg.get_title().set_fontsize(int(tfs))
    return leg


def _resolve_output_path(config_path: Path, output_raw: str, output_dir_override: Path | None) -> Path:
    if output_dir_override is not None:
        return output_dir_override / output_raw
    out = Path(output_raw)
    if out.is_absolute():
        return out
    return config_path.parent / out


def _resolve_legend_config(global_cfg: dict[str, Any], plot_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge global/plot legend; default is below the axes (flat multi-column), not on the right.

    Opt out of this for a whole file: `"global": {"legend_placement": "default"}`.
    Opt out for one plot: `"legend": {"outside": "default"}`.
    """
    g_leg = global_cfg.get("legend")
    g_leg = dict(g_leg) if isinstance(g_leg, dict) else {}
    p_leg = plot_cfg.get("legend")
    p_leg = dict(p_leg) if isinstance(p_leg, dict) else {}

    if global_cfg.get("legend_placement", "below") == "default":
        return {**g_leg, **p_leg}

    if g_leg.get("outside") == "default" or p_leg.get("outside") == "default":
        out = {**g_leg, **p_leg}
        out.pop("outside", None)
        return out

    repo: dict[str, Any] = {
        "outside": "below",
        "loc": "upper center",
        "ncol": 2,
        "frameon": True,
        "fontsize": 10,
        "framealpha": 0.92,
    }
    merged = {**repo, **g_leg, **p_leg}
    if not merged.get("enabled", True):
        return merged
    if merged.get("outside") != "below":
        return merged

    bty = merged.pop("below_anchor_y", None)
    bty_f = float(bty) if bty is not None else -0.03
    merged["loc"] = str(merged.get("loc", "upper center"))
    merged["bbox_to_anchor"] = [0.5, bty_f]
    merged.pop("borderaxespad", None)
    return merged


def _extra_height_for_below_legend(
    global_cfg: dict[str, Any],
    plot_cfg: dict[str, Any],
    legend_cfg: dict[str, Any],
    n_legend_entries: int,
) -> float:
    base = float(global_cfg.get("figure_extra_height_for_below_legend", 0.24))
    base += float(plot_cfg.get("figure_extra_height_for_below_legend", 0.0))
    ncols = max(1, int(legend_cfg.get("ncol", 2)))
    rows = max(1, (max(1, n_legend_entries) + ncols - 1) // ncols)
    if rows > 2:
        base += 0.12 * float(rows - 2)
    return base


def _plot_one(
    config_path: Path,
    plot_cfg: dict[str, Any],
    output_dir_override: Path | None,
    default_dpi: int,
    global_cfg: dict[str, Any],
) -> Path:
    title = str(plot_cfg.get("title", ""))
    output_raw = plot_cfg.get("output")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ValueError("Each plot must contain a non-empty `output` path.")

    series_cfg = _ensure_list("series", plot_cfg.get("series", []))
    baselines_cfg = _ensure_list("baselines", plot_cfg.get("baselines", []))
    legend_resolved = _resolve_legend_config(global_cfg, plot_cfg)
    outside_below = legend_resolved.get("enabled", True) and legend_resolved.get("outside") == "below"
    n_legend_entries = _count_legend_entries(series_cfg, baselines_cfg)
    layout_str = str(legend_resolved.get("layout", "")).strip().lower()
    legend_ncol_eff: int | None = None
    n_for_legend_height = n_legend_entries
    if outside_below and layout_str == "pipeline_columns":
        n_for_legend_height = 15
    if outside_below:
        leg_h = dict(legend_resolved)
        ncol0 = int(leg_h.get("ncol", 2))
        if layout_str == "pipeline_columns":
            legend_ncol_eff = 5
        else:
            legend_ncol_eff = _effective_ncol_for_below(ncol0, n_legend_entries)
        leg_h["ncol"] = legend_ncol_eff
        fh_extra = _extra_height_for_below_legend(global_cfg, plot_cfg, leg_h, n_for_legend_height)
    else:
        fh_extra = 0.0

    figure_cfg = plot_cfg.get("figure", {})
    if not isinstance(figure_cfg, dict):
        raise ValueError("`figure` must be an object.")
    figsize = figure_cfg.get("size", [12, 6])
    if not (isinstance(figsize, list) and len(figsize) == 2):
        raise ValueError("`figure.size` must be [width, height].")
    fw, fh = float(figsize[0]), float(figsize[1])
    if outside_below:
        fh += fh_extra

    dpi = int(figure_cfg.get("dpi", default_dpi))
    fig, ax = plt.subplots(figsize=(fw, fh))

    x_cfg = plot_cfg.get("x", {})
    if not isinstance(x_cfg, dict):
        raise ValueError("`x` must be an object.")
    x_values = x_cfg.get("values")
    x_labels = x_cfg.get("labels")
    x_label_text = str(x_cfg.get("label", ""))
    x_tick_rotation = float(x_cfg.get("tick_rotation", 45))
    x_tick_fontsize = int(x_cfg.get("tick_fontsize", 11))

    if x_values is not None:
        x_values = _ensure_list("x.values", x_values)
    if x_labels is not None:
        x_labels = _ensure_list("x.labels", x_labels)

    if not series_cfg:
        raise ValueError("Each plot must contain at least one item in `series`.")

    all_x_points: list[float] = []
    all_y_points: list[float] = []

    for idx, line in enumerate(series_cfg):
        if not isinstance(line, dict):
            raise ValueError(f"`series[{idx}]` must be an object.")

        raw_label = line.get("label", f"series-{idx + 1}")
        if line.get("include_in_legend", True) is False:
            plot_label: str = "_nolegend_"
        else:
            plot_label = str(raw_label)
        y_raw = _ensure_list(f"series[{idx}].y", line.get("y", []))
        if not y_raw:
            continue
        y_values = [_numeric(v) for v in y_raw]

        line_x_raw = line.get("x")
        if line_x_raw is not None:
            line_x = _ensure_list(f"series[{idx}].x", line_x_raw)
            if len(line_x) != len(y_values):
                raise ValueError(f"`series[{idx}].x` and `.y` must have same length.")
            x_line = [float(v) for v in line_x]
        elif x_values is not None:
            if len(x_values) != len(y_values):
                raise ValueError(f"`x.values` length must match `series[{idx}].y` length.")
            x_line = [float(v) for v in x_values]
        else:
            x_line = [float(i) for i in range(len(y_values))]

        style = line.get("style", {})
        if not isinstance(style, dict):
            raise ValueError(f"`series[{idx}].style` must be an object.")

        valid_x: list[float] = []
        valid_y: list[float] = []
        for x_v, y_v in zip(x_line, y_values):
            if y_v is None or math.isnan(y_v):
                continue
            valid_x.append(x_v)
            valid_y.append(y_v)

        if not valid_y:
            continue

        plotted = ax.plot(valid_x, valid_y, label=plot_label, **_pick_style(style))
        all_x_points.extend(valid_x)
        all_y_points.extend(valid_y)

        ann_cfg = line.get("annotate", {})
        if isinstance(ann_cfg, dict) and ann_cfg.get("enabled", False):
            ann_fmt = str(ann_cfg.get("fmt", "{:.4f}"))
            ann_fontsize = int(ann_cfg.get("fontsize", 7))
            color = style.get("color", plotted[0].get_color())
            for xv, yv in zip(valid_x, valid_y):
                ax.text(xv, yv, ann_fmt.format(yv), ha="center", va="bottom", fontsize=ann_fontsize, color=color)

    for bi, bl in enumerate(baselines_cfg):
        if not isinstance(bl, dict):
            raise ValueError(f"`baselines[{bi}]` must be an object.")
        val = _numeric(bl.get("value"))
        if val is None:
            continue
        bl_label = str(bl.get("label", f"baseline-{bi + 1}"))
        bl_style = bl.get("style", {})
        if not isinstance(bl_style, dict):
            raise ValueError(f"`baselines[{bi}].style` must be an object.")
        line_style = _pick_style(bl_style)
        if "linestyle" not in line_style:
            line_style["linestyle"] = ":"
        if "linewidth" not in line_style:
            line_style["linewidth"] = 1.5
        ax.axhline(y=val, label=bl_label, **line_style)
        all_y_points.append(val)

    y_cfg = plot_cfg.get("y", {})
    if not isinstance(y_cfg, dict):
        raise ValueError("`y` must be an object.")
    y_tick_fontsize = int(y_cfg.get("tick_fontsize", 11))
    y_label_fontsize = int(y_cfg.get("label_fontsize", y_tick_fontsize))
    if "label" in y_cfg:
        ax.set_ylabel(str(y_cfg["label"]), fontsize=y_label_fontsize)
    y_lim = y_cfg.get("limits")
    if isinstance(y_lim, list) and len(y_lim) == 2:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    elif y_cfg.get("autoscale_tight"):
        _apply_y_tight_margin(ax, all_y_points, y_cfg)

    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    x_label_fontsize = int(x_cfg.get("label_fontsize", x_tick_fontsize))

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    if x_label_text:
        _xlp = x_cfg.get("labelpad")
        if _xlp is not None:
            ax.set_xlabel(x_label_text, fontsize=x_label_fontsize, labelpad=float(_xlp))
        elif outside_below:
            ax.set_xlabel(x_label_text, fontsize=x_label_fontsize, labelpad=14.0)
        else:
            ax.set_xlabel(x_label_text, fontsize=x_label_fontsize)

    if x_labels is not None:
        if x_values is not None and len(x_labels) != len(x_values):
            raise ValueError("`x.labels` length must match `x.values` length.")
        if x_values is not None:
            ax.set_xticks([float(v) for v in x_values])
        else:
            ax.set_xticks([float(i) for i in range(len(x_labels))])
        ax.set_xticklabels(
            [str(v) for v in x_labels],
            rotation=x_tick_rotation,
            ha="right",
            fontsize=x_tick_fontsize,
        )
    elif x_values is not None:
        ax.set_xticks([float(v) for v in x_values])

    ax.tick_params(axis="x", labelsize=x_tick_fontsize)
    ax.tick_params(axis="y", labelsize=y_tick_fontsize)

    grid_cfg = plot_cfg.get("grid", {})
    if not isinstance(grid_cfg, dict):
        raise ValueError("`grid` must be an object.")
    if grid_cfg.get("enabled", True):
        ax.grid(
            True,
            alpha=float(grid_cfg.get("alpha", 0.3)),
            linestyle=str(grid_cfg.get("linestyle", "--")),
        )

    legend_artist = None
    if legend_resolved.get("enabled", True):
        leg_fs = int(legend_resolved.get("fontsize", 9))
        if outside_below:
            leg_fs = max(10, leg_fs)
        legend_kwargs: dict[str, Any] = {
            "fontsize": leg_fs,
            "framealpha": float(legend_resolved.get("framealpha", 0.9)),
        }
        if "frameon" in legend_resolved:
            legend_kwargs["frameon"] = bool(legend_resolved["frameon"])
        if "borderaxespad" in legend_resolved:
            legend_kwargs["borderaxespad"] = float(legend_resolved["borderaxespad"])
        if outside_below and legend_ncol_eff is not None:
            legend_kwargs["ncol"] = legend_ncol_eff
        elif "ncol" in legend_resolved:
            legend_kwargs["ncol"] = int(legend_resolved["ncol"])

        _ncol_leg = int(legend_kwargs.get("ncol", 1))
        if outside_below and _ncol_leg >= 3:
            legend_kwargs.setdefault(
                "columnspacing",
                float(legend_resolved.get("columnspacing", 0.9)),
            )
            legend_kwargs.setdefault(
                "labelspacing",
                float(legend_resolved.get("labelspacing", 0.35)),
            )

        if outside_below:
            legend_kwargs["loc"] = str(legend_resolved.get("loc", "upper center"))
            bbox_to_anchor = legend_resolved.get("bbox_to_anchor")
            if bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = tuple(
                    _numeric_list("legend.bbox_to_anchor", bbox_to_anchor)
                )
            else:
                legend_kwargs["bbox_to_anchor"] = (0.5, -0.03)
        else:
            legend_kwargs["loc"] = str(legend_resolved.get("loc", "best"))
            bbox_to_anchor = legend_resolved.get("bbox_to_anchor")
            if bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = tuple(
                    _numeric_list("legend.bbox_to_anchor", bbox_to_anchor)
                )

        if outside_below and layout_str == "pipeline_columns":
            handles, labels = ax.get_legend_handles_labels()
            legend_artist = _legend_pipeline_five_columns(
                ax, list(handles), list(labels), legend_kwargs, legend_resolved
            )
        else:
            legend_artist = ax.legend(**legend_kwargs)
            if legend_resolved.get("title"):
                legend_artist.set_title(str(legend_resolved["title"]))
                _tfs = legend_resolved.get("title_fontsize")
                if _tfs is not None:
                    legend_artist.get_title().set_fontsize(int(_tfs))

    out_path = _resolve_output_path(config_path, output_raw, output_dir_override)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if outside_below:
        # Keep the axes box tall; reserve only figure margins. Extra vertical space for the
        # legend comes from savefig(..., bbox_extra_artists=...) so the line plot is not
        # vertically squeezed the way tight_layout(rect=[...]) would do.
        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)
        pad_inches = float(
            plot_cfg.get(
                "save_pad_inches",
                global_cfg.get("save_pad_inches_below_legend", 0.06),
            )
        )
        save_kw: dict[str, Any] = {
            "dpi": dpi,
            "bbox_inches": "tight",
            "pad_inches": pad_inches,
        }
        if legend_artist is not None:
            save_kw["bbox_extra_artists"] = (legend_artist,)
        fig.savefig(out_path, **save_kw)
    else:
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch generate plots from one JSON file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "plots_config.json",
        help="Path to one JSON config containing all plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_json(config_path)
    plots = _ensure_list("plots", config.get("plots", []))
    if not plots:
        raise ValueError("Config must contain non-empty `plots`.")

    global_cfg = config.get("global", {})
    if not isinstance(global_cfg, dict):
        raise ValueError("`global` must be an object.")
    default_dpi = int(global_cfg.get("default_dpi", 200))

    print(f"[info] config: {config_path}")
    print(f"[info] plots: {len(plots)}")
    for i, plot_cfg in enumerate(plots, start=1):
        if not isinstance(plot_cfg, dict):
            raise ValueError(f"`plots[{i - 1}]` must be an object.")
        out = _plot_one(
            config_path,
            plot_cfg,
            args.output_dir,
            default_dpi=default_dpi,
            global_cfg=global_cfg,
        )
        print(f"[{i}/{len(plots)}] saved {out}")

    print("[done] all plots generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
