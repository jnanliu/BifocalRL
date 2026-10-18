from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
LCB_ROOT = DATASETS_ROOT / "livecodebench" / "code_generation_lite"
CRUX_PATH = DATASETS_ROOT / "cruxeval" / "test.jsonl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOCAL_DATASETS_ROOT", str(DATASETS_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _load_lcb_codegen_metrics_module():
    spec = importlib.util.find_spec("lighteval")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("lighteval is not installed")
    module_path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "tasks"
        / "extended"
        / "lcb"
        / "codegen_metrics.py"
    )
    module_spec = importlib.util.spec_from_file_location("_lcb_codegen_metrics_cpu_test", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load LightEval LCB metrics from {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("subset", "expected_files"),
    [
        ("release_v1", ["test.jsonl"]),
        ("release_v6", ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]),
        ("release_latest", ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]),
        ("v6", ["test6.jsonl"]),
        ("v4_v5", ["test4.jsonl", "test5.jsonl"]),
    ],
)
def test_lcb_subset_mapping(subset: str, expected_files: list[str]) -> None:
    from eval.lighteval.custom_tasks.offline_patch import _redirect_dataset_if_local

    builder, data_files = _redirect_dataset_if_local("livecodebench/code_generation_lite", subset)
    assert builder == "json"
    assert data_files is not None
    assert [Path(path).name for path in data_files["test"]] == expected_files


def test_lcb_unknown_subset_fails() -> None:
    from eval.lighteval.custom_tasks.offline_patch import _redirect_dataset_if_local

    with pytest.raises(ValueError, match="Unsupported LiveCodeBench subset"):
        _redirect_dataset_if_local("livecodebench/code_generation_lite", "release_v7")


def test_lcb_streaming_load_and_schema() -> None:
    from eval.lighteval.custom_tasks import offline_patch  # noqa: F401
    from datasets import load_dataset

    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        name="release_v6",
        split="test",
        streaming=True,
    )
    row = next(iter(dataset))
    required = {
        "question_content",
        "question_id",
        "starter_code",
        "public_test_cases",
        "private_test_cases",
        "metadata",
    }
    assert required.issubset(row)
    assert json.loads(row["public_test_cases"])


def test_lcb_private_tests_and_controlled_execution() -> None:
    module = _load_lcb_codegen_metrics_module()
    with (LCB_ROOT / "test6.jsonl").open(encoding="utf-8") as handle:
        row = json.loads(next(handle))

    private_tests = module.translate_private_test_cases(row["private_test_cases"])
    assert isinstance(private_tests, list)
    assert private_tests
    assert module.extract_code("text\n```python\nprint(2)\n```") == "print(2)"

    sample = {
        "input_output": json.dumps(
            {
                "inputs": ["1\n"],
                "outputs": ["2\n"],
                "fn_name": None,
            }
        )
    }
    metrics, _ = module.codegen_metrics(
        [sample],
        [["print(int(input()) + 1)"]],
        k_list=[1],
        num_process_evaluate=1,
        timeout=2,
    )
    assert metrics["pass@1"] == pytest.approx(1.0)


def test_lighteval_release_v6_task_registration() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except (LookupError, OSError):
        pytest.skip("Run scripts/setup_lighteval_nltk.sh first")

    from lighteval.tasks.extended.lcb.main import TASKS_TABLE

    assert "lcb:codegeneration_release_v6" in {task.name for task in TASKS_TABLE}


def test_lighteval_release_v6_single_task_requests_one_generation() -> None:
    from eval.lighteval.custom_tasks.offline_lcb_single import TASKS_TABLE
    from lighteval.tasks.lighteval_task import Doc, LightevalTask
    from lighteval.tasks.requests import RequestType

    assert len(TASKS_TABLE) == 1
    config = TASKS_TABLE[0]
    assert config.name == "lcb:codegeneration_release_v6_single"
    assert config.hf_subset == "release_v6"

    task = LightevalTask(f"custom|{config.name}", config)
    assert max(task.num_samples) == 1
    assert [metric.metric_name for metric in task.metrics] == ["codegen_pass@1:1"]

    doc = Doc(task_name=task.name, query="prompt", choices=[""], gold_index=0)
    requests = task.construct_requests(doc, doc.query, "0_7", task.name)
    generation_request = requests[RequestType.GREEDY_UNTIL][0]
    assert generation_request.num_samples == 1
    assert generation_request.do_sample is True


def test_lighteval_legacy_task_sample_limit_is_not_encoded_in_task_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eval.lighteval import runner

    monkeypatch.setattr(runner, "_is_legacy_lighteval", lambda: True)
    task = "extended|lcb:codegeneration_release_v6|0|0"
    assert runner._task_string([task]) == task

    with pytest.raises(ValueError, match="--max_samples"):
        runner._task_string(["extended|lcb:codegeneration_release_v6|0|2"])


def test_gpu_smoke_uses_lighteval_max_samples_option() -> None:
    smoke_script = (PROJECT_ROOT / "scripts" / "smoke_test_extended_eval_gpu.sh").read_text(
        encoding="utf-8"
    )
    assert 'custom|lcb:codegeneration_release_v6_single|0|0' in smoke_script
    assert "--custom_tasks eval.lighteval.custom_tasks.offline_lcb_single" in smoke_script
    assert "--max_samples 2" in smoke_script
    assert "--temperature 0.0" in smoke_script
    assert "--top_p 1.0" in smoke_script


@pytest.mark.parametrize(
    "profile_name",
    [
        "Qwen2.5-3B/single_intuitor.json",
        "Qwen2.5-3B/single_mixed_reward_reliability_04sc_06corr_all_extended_eval.json",
        "DeepSeek-R1-Distill-Qwen-1.5B/single_mixed_reward_reliability_04sc_06corr_all_extended_eval.json",
    ],
)
def test_lcb_profiles_use_single_greedy_task(profile_name: str) -> None:
    profile_path = PROJECT_ROOT / "recipes" / "configs" / "mixed_reward" / "single" / profile_name
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    lighteval = profile["eval"]["lighteval"]

    assert lighteval["tasks"] == "custom|lcb:codegeneration_release_v6_single|0|0"
    assert lighteval["custom_tasks"] == "eval.lighteval.custom_tasks.offline_lcb_single"
    assert lighteval["temperature"] == 0.0
    assert lighteval["top_p"] == 1.0


@pytest.mark.parametrize(
    "profile_name",
    [
        "Qwen2.5-3B/single_mixed_reward_reliability_04sc_06corr_all_extended_eval.json",
        "DeepSeek-R1-Distill-Qwen-1.5B/single_mixed_reward_reliability_04sc_06corr_all_extended_eval.json",
    ],
)
def test_extended_eval_generation_budgets_are_4096(profile_name: str) -> None:
    profile_path = PROJECT_ROOT / "recipes" / "configs" / "mixed_reward" / "single" / profile_name
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    evaluation = profile["eval"]

    assert evaluation["max_new_tokens"] == 4096
    assert evaluation["lighteval"]["max_model_len"] == 8192
    assert evaluation["hacrl"]["max_new_tokens"] == 4096
    assert evaluation["hacrl"]["max_model_len"] >= 8192
    assert evaluation["ttrl"]["max_model_len"] >= 8192

    base_name = profile_name.split("/", 1)[0]
    base_config = PROJECT_ROOT / "recipes" / "configs" / "mixed_reward" / "single" / base_name / (
        "single_mixed_reward_reliability_04sc_06corr_all.json"
    )
    base = json.loads(base_config.read_text(encoding="utf-8"))
    assert base["generation"]["max_completion_length"] == 4096
    assert "max_new_tokens" not in base["eval"]


def test_cruxeval_local_contract() -> None:
    from eval.common.cruxeval import (
        construct_cruxeval_prompt,
        extract_cruxeval_answer,
        judge_cruxeval,
        load_cruxeval_jsonl,
    )

    rows = load_cruxeval_jsonl(str(CRUX_PATH))
    assert len(rows) == 800
    assert len({row["id"] for row in rows}) == 800
    assert all({"id", "code", "input", "output"}.issubset(row) for row in rows)

    sample = rows[0]
    prompt = construct_cruxeval_prompt(sample)
    prediction = extract_cruxeval_answer(f"Output: {sample['output']}\n")
    assert prompt.endswith("Output:")
    assert judge_cruxeval(prediction, sample["output"]).is_correct
    assert not judge_cruxeval("None", sample["output"]).is_correct
