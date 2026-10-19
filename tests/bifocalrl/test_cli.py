"""Parse the actual CLI declarations without loading the GPU runtime."""

import argparse
import ast
import importlib.util
from datetime import datetime
from pathlib import Path

from openrlhf.algorithms.bifocalrl.runtime import configure


def test_native_cli_installs_recipe_before_critic_and_reward_selection():
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
            str(root / "examples/bifocalrl/configs/ettrl.json"),
            "--vllm.num_engines",
            "1",
            "--rollout.n_samples_per_prompt",
            "8",
            "--algo.kl.init_coef",
            "0",
        ]
    )
    args = module.hierarchize(parsed)
    configure(args)
    assert args.bifocal.settings["method"] == "ettrl"
    assert args.algo.advantage.estimator == "group_norm"
    assert args.reward.remote_url == "agent"
    assert args.critic.model_name_or_path is None
