from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_SCRIPT = PROJECT_ROOT / "scripts" / "plotting" / "plot_ttrl_eval_auto_new.py"
EVAL_ONLY_SCRIPT = PROJECT_ROOT / "scripts" / "eval_only_extended_all.sh"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("_plot_ttrl_eval_auto_new_test", PLOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLOT = _load_plot_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_ttrl_reader_preserves_comparison_summary_schema(tmp_path: Path) -> None:
    method_root = tmp_path / "method"
    _write_json(
        method_root / "a_checkpoint-3" / "comparison_summary.json",
        {"results": {"checkpoint-3": {"aime24": 0.375}}},
    )

    assert PLOT.load_dataset_value(method_root, 3, "aime24") == pytest.approx(0.375)


@pytest.mark.parametrize(
    ("task_name", "metric_name"),
    [
        ("custom|lcb:codegeneration_release_v6_single|0", "codegen_pass@1:1"),
        ("extended|lcb:codegeneration_release_v6|0", "codegen_pass@1:16"),
    ],
)
def test_lcb_reader_extracts_metric_and_ignores_stderr(
    tmp_path: Path,
    task_name: str,
    metric_name: str,
) -> None:
    method_root = tmp_path / "method-extended-eval"
    _write_json(
        method_root
        / "lighteval"
        / "checkpoint-4"
        / "results"
        / "model"
        / "results_2026-07-17.json",
        {
            "results": {
                task_name: {
                    metric_name: 0.125,
                    f"{metric_name}_stderr": 0.02,
                },
                "all": {
                    metric_name: 0.125,
                    f"{metric_name}_stderr": 0.02,
                },
            }
        },
    )

    assert PLOT.load_dataset_value(method_root, 4, "lcb_release_v6") == pytest.approx(0.125)


def test_cruxeval_reader_extracts_pass_at_k(tmp_path: Path) -> None:
    method_root = tmp_path / "method-extended-eval"
    _write_json(
        method_root / "hacrl" / "cruxeval" / "results_checkpoint-5_model-a.json",
        {"summary": {"dataset": "cruxeval", "pass_at_k": 0.25}},
    )

    assert PLOT.load_dataset_value(method_root, 5, "cruxeval") == pytest.approx(0.25)


def test_load_series_uses_dataset_method_suffix(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    method_root = eval_root / "model" / "method-extended-eval"
    _write_json(
        method_root / "hacrl" / "cruxeval" / "results_checkpoint-1_model-a.json",
        {"summary": {"pass_at_k": 0.5}},
    )

    values = PLOT.load_series(
        eval_root,
        "model",
        "method",
        "cruxeval",
        1,
        2,
        "-extended-eval",
    )

    assert values == [0.5, None]


def test_load_series_prefers_normal_results_and_falls_back_per_step(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    normal_root = eval_root / "model" / "method"
    extended_root = eval_root / "model" / "method-extended-eval"
    _write_json(
        normal_root / "a_checkpoint-1" / "comparison_summary.json",
        {"results": {"checkpoint-1": {"aime24": 0.75}}},
    )
    _write_json(
        extended_root / "a_checkpoint-1" / "comparison_summary.json",
        {"results": {"checkpoint-1": {"aime24": 0.25}}},
    )
    _write_json(
        extended_root / "a_checkpoint-2" / "comparison_summary.json",
        {"results": {"checkpoint-2": {"aime24": 0.5}}},
    )

    values = PLOT.load_series(
        eval_root,
        "model",
        "method",
        "aime24",
        1,
        2,
    )

    assert values == [0.75, 0.5]


def test_load_baseline_prefers_normal_results(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    normal_root = eval_root / "model" / "method"
    extended_root = eval_root / "model" / "method-extended-eval"
    _write_json(
        normal_root / "a_checkpoint-0" / "comparison_summary.json",
        {"results": {"checkpoint-0": {"aime24": 0.75}}},
    )
    _write_json(
        extended_root / "a_checkpoint-0" / "comparison_summary.json",
        {"results": {"checkpoint-0": {"aime24": 0.25}}},
    )

    value = PLOT.load_baseline(
        eval_root,
        "model",
        "method",
        "aime24",
        True,
    )

    assert value == pytest.approx(0.75)


def test_plot_dataset_uses_code_dataset_metadata(tmp_path: Path) -> None:
    if PLOT.plt is None:
        pytest.skip("matplotlib is not installed")

    output = PLOT.plot_dataset(
        "lcb_release_v6",
        [
            {
                "label": "method",
                "style": {"color": "#2563EB", "linestyle": "-", "marker": "o"},
                "y": [0.1, 0.2],
            }
        ],
        [{"label": "step0 baseline", "color": "#111827", "value": 0.08}],
        "model",
        1,
        2,
        tmp_path,
    )

    assert output == tmp_path / "model_lcb_release_v6.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_eval_only_batch_contains_the_confirmed_eleven_variants() -> None:
    script = EVAL_ONLY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"CONFIG_NAMES=\((.*?)\n\)", script, re.DOTALL)
    assert match is not None
    config_names = re.findall(r'"([^"\n]+\.json)"', match.group(1))

    assert config_names == [
        "single_intuitor.json",
        "single_mv.json",
        "single_repro_gspo.json",
        "single_mixed_reward_alpha050.json",
        "single_mixed_reward_reliability_baseline.json",
        "single_mixed_reward_reliability_06sc_04corr_all.json",
        "single_mixed_reward_reliability_06sc_04corr_advantage_only.json",
        "single_mixed_reward_reliability_06sc_04corr_sample_only.json",
        "single_mixed_reward_reliability_04sc_06corr_all.json",
        "single_mixed_reward_reliability_04sc_06corr_advantage_only.json",
        "single_mixed_reward_reliability_04sc_06corr_sample_only.json",
    ]
