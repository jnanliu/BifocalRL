"""Parse the actual CLI declarations without loading the GPU runtime."""

import argparse
import ast
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from openrlhf.algorithms.bifocalrl.runtime import configure


@pytest.fixture
def eval_cli(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    path = root / "openrlhf/cli/train_ppo_ray.py"
    main = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.If))
    nodes = []
    for node in main.body:
        if isinstance(node, ast.ImportFrom) and node.module == "openrlhf.utils.config":
            break
        nodes.append(node)

    def parse(argv):
        monkeypatch.setattr(sys, "argv", [str(path), *argv])
        namespace = {"argparse": argparse, "datetime": datetime, "json": json, "math": math, "Path": Path}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        return namespace["args"]

    return parse


@pytest.mark.parametrize(
    "recipe,method",
    [
        ("ettrl", None),
        ("bifocalrl", None),
        ("bifocalrl", "BifocalRL"),
        ("reliability_sc_corr", None),
        ("circa", None),
        ("circa", "CIRCA"),
    ],
)
def test_native_cli_installs_recipe_before_critic_and_reward_selection(recipe, method):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("bifocal_cli_config", root / "openrlhf/utils/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = root / "openrlhf/cli/train_ppo_ray.py"
    tree = ast.parse(path.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    declarations = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
            break
        declarations.append(node)
    namespace = {"argparse": argparse, "datetime": datetime}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    parsed = namespace["parser"].parse_args(
        [
            "--bifocal.config",
            str(root / f"examples/bifocalrl/configs/{recipe}.json"),
            "--bifocal.sandbox_fusion_url",
            "http://sandbox/run_code",
            "--eval.n_samples_per_prompt_by_datasource",
            '{"math500":1,"livecodebench":8}',
            "--vllm.num_engines",
            "1",
            "--rollout.n_samples_per_prompt",
            "8",
            "--algo.kl.init_coef",
            "0.005" if recipe == "circa" else "0",
        ]
        + (["--bifocal.method", method] if method else [])
        + (["--algo.kl.use_loss"] if recipe == "circa" else [])
    )
    args = module.hierarchize(parsed)
    configure(args)
    assert args.bifocal.settings["method"] == recipe
    assert args.algo.advantage.estimator == "group_norm"
    assert args.reward.remote_url == "agent"
    assert args.bifocal.settings["sandbox_fusion_url"] == "http://sandbox/run_code"
    assert args.eval.n_samples_per_prompt_by_datasource == '{"math500":1,"livecodebench":8}'
    assert args.critic.model_name_or_path is None
    if recipe == "bifocalrl":
        assert args.bifocal.settings["mixed"]["sc_between_gamma"] == 0.5
    if recipe == "circa":
        assert args.algo.kl.use_loss and args.algo.kl.init_coef == 0.005


@pytest.mark.parametrize("path", sorted(
    (Path(__file__).resolve().parents[2] / "examples/bifocalrl/eval_configs").glob("*.json")
))
def test_eval_json_examples_load_arrays_and_sampling_maps(eval_cli, path):
    expected = json.loads(path.read_text())
    args = eval_cli(["--eval.config", str(path)])
    assert getattr(args, "eval.dataset") == ",".join(expected["dataset"])
    assert getattr(args, "eval.steps") == expected["steps"]
    assert getattr(args, "eval.temperature") == expected["temperature"]
    assert getattr(args, "eval.label_key") == "label"
    assert getattr(args, "eval.before_train") is True
    if "sandbox_fusion_url" in expected:
        assert json.loads(getattr(args, "eval.n_samples_per_prompt_by_datasource")) == expected["n_samples_per_prompt_by_datasource"]
        assert getattr(args, "eval.sandbox_fusion_url") == expected["sandbox_fusion_url"]
        assert getattr(args, "eval.sandbox_concurrency") == 16


def test_eval_cli_overrides_json_even_when_value_equals_parser_default(eval_cli, tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"dataset": ["math.jsonl"], "n_samples_per_prompt": 8,
                                "sandbox_fusion_url": "http://json/run_code"}))
    args = eval_cli(["--eval.config", str(path), "--eval.n_samples_per_prompt", "4",
                     "--eval.dataset", "override.jsonl", "--bifocal.sandbox_fusion_url", "http://cli/run_code"])
    assert getattr(args, "eval.dataset") == "override.jsonl"
    assert getattr(args, "eval.n_samples_per_prompt") == 4
    assert getattr(args, "eval.sandbox_fusion_url") == "http://cli/run_code"


@pytest.mark.parametrize("setting", [True, False])
def test_initial_eval_json_and_explicit_cli_override(eval_cli, tmp_path, setting):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"before_train": setting, "steps": -1}))
    assert getattr(eval_cli(["--eval.config", str(path)]), "eval.before_train") is setting
    flag = "--no-eval.before_train" if setting else "--eval.before_train"
    args = eval_cli(["--eval.config", str(path), flag])
    assert getattr(args, "eval.before_train") is not setting
    assert getattr(args, "eval.steps") == -1


@pytest.mark.parametrize("settings", [
    [], {"unknown": 1}, {"dataset": []}, {"dataset": [1]}, {"n_samples_per_prompt": 1.5},
    {"n_samples_per_prompt": 0}, {"temperature": -1}, {"steps": 0}, {"sandbox_timeout": 0},
    {"n_samples_per_prompt_by_datasource": {"math": True}},
    {"before_train": "false"}, {"before_train": 0}, {"before_train": None},
])
def test_invalid_eval_json_fails_before_training(eval_cli, tmp_path, settings):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(SystemExit) as error:
        eval_cli(["--eval.config", str(path)])
    assert error.value.code == 2


@pytest.mark.parametrize("config_via_env", [True, False])
@pytest.mark.parametrize("recipe,nnodes,gpus", [
    ("bifocalrl", 1, 8), ("circa", 1, 1), ("circa", 1, 2), ("circa", 2, 2),
])
def test_run_script_preserves_json_defaults_and_explicit_environment_overrides(
    eval_cli, tmp_path, config_via_env, recipe, nnodes, gpus
):
    root = Path(__file__).resolve().parents[2]
    capture = tmp_path / "capture-python"
    capture.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    capture.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("EVAL_")}
    env.pop("KL_COEF", None)
    env.pop("BIFOCAL_CONFIG", None)
    env.update(
        PYTHON=str(capture), MODEL_PATH="model with spaces", PROMPT_DATA="train.jsonl",
        NNODE=str(nnodes), GPUS_PER_NODE=str(gpus), VLLM_TP_SIZE="1",
    )
    command = ["bash", str(root / "examples/bifocalrl/run.sh"), recipe]
    config = str(root / "examples/bifocalrl/eval_configs/math.json")
    expected = json.loads(Path(config).read_text())
    if config_via_env:
        env["EVAL_CONFIG"] = config
    else:
        command.extend(["--eval.config", config])
    for override in (False, True):
        if override:
            env.update(EVAL_DATA="override.jsonl", EVAL_N_SAMPLES_PER_PROMPT="2")
        output = subprocess.check_output(command, env=env, text=True)
        printed, captured = output.splitlines()
        argv = json.loads(captured)
        assert printed.startswith("Running command: ")
        assert shlex.split(printed.removeprefix("Running command: ")) == [str(capture), *argv]
        args = eval_cli(argv[2:])
        assert getattr(args, "ref.num_nodes") == getattr(args, "actor.num_nodes") == nnodes
        assert getattr(args, "ref.num_gpus_per_node") == getattr(args, "actor.num_gpus_per_node") == gpus
        assert getattr(args, "algo.kl.init_coef") == (0.005 if recipe == "circa" else 0)
        assert getattr(args, "algo.kl.use_loss") is (recipe == "circa")
        assert getattr(args, "eval.temperature") == expected["temperature"]
        assert getattr(args, "eval.n_samples_per_prompt") == (2 if override else expected["n_samples_per_prompt"])
        assert getattr(args, "eval.dataset") == ("override.jsonl" if override else ",".join(expected["dataset"]))
