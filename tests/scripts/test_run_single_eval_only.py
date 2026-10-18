from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_single.sh"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_model_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"test")


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "EVAL_ONLY",
        "EVAL_PROFILE_FILE",
        "EVAL_ONLY_START_STEP",
        "EVAL_ONLY_END_STEP",
        "EVAL_ONLY_STRICT",
        "EVAL_FORCE",
        "EVAL_ONLY_DRY_RUN",
        "LIGHTEVAL_ENABLED",
        "LIGHTEVAL_MAX_SAMPLES",
        "ZEROEVAL_ENABLED",
        "HACRL_EVAL_ENABLED",
        "TTRL_EVAL_ENABLED",
        "TTRL_EVAL_TASKS",
        "LIGHTEVAL_OUT_PREFIX",
        "ZEROEVAL_OUTPUT_ROOT",
        "HACRL_EVAL_OUTPUT_ROOT",
        "TTRL_EVAL_OUTPUT_ROOT",
        "LATEST_CHECKPOINT_DIR",
    ):
        env.pop(name, None)
    return env


def _base_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    initial_model = tmp_path / "initial-model"
    checkpoint_root = tmp_path / "trained" / "latest_checkpoint"
    model_dir = checkpoint_root / "model"
    _make_model_dir(initial_model)
    model_dir.mkdir(parents=True)

    config_path = tmp_path / "base.json"
    _write_json(
        config_path,
        {
            "output_group": "test",
            "name": "trained-method",
            "models": {
                "policy": str(initial_model),
                "reward": None,
                "model_tag": "tiny",
            },
            "training": {
                "max_steps": 2,
                "start_step": 1,
            },
            "output": {
                "checkpoint_dir": str(checkpoint_root),
                "keep_all_checkpoints": True,
                "ttrl_dir": str(tmp_path / "original-eval"),
            },
            "eval": {
                "ttrl": {
                    "enabled": True,
                    "tasks": "math_500,gsm8k",
                }
            },
        },
    )
    return config_path, initial_model, model_dir


def _eval_profile(tmp_path: Path, *, end_step: int = 2) -> Path:
    profile_path = tmp_path / "extended.json"
    _write_json(
        profile_path,
        {
            "profile": {
                "name": "extended-eval",
                "output_suffix": "extended-eval",
                "output_root": str(tmp_path / "extended-results"),
                "start_step": 0,
                "end_step": end_step,
                "strict": True,
            },
            "eval": {
                "lighteval": {"enabled": False},
                "zeroeval": {"enabled": False},
                "hacrl": {"enabled": False},
                "ttrl": {
                    "enabled": True,
                    "tasks": "aime24,aime25,amc23",
                    "scorer": "ttrl_math",
                    "prompt_mode": "suffix",
                },
            },
        },
    )
    return profile_path


def _run_eval_only(config: Path, profile: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(RUNNER),
            str(config),
            "--eval-only",
            "--eval-profile",
            str(profile),
            "--eval-dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_eval_only_resolves_initial_model_and_existing_checkpoints(tmp_path: Path) -> None:
    config, initial_model, model_dir = _base_config(tmp_path)
    _make_model_dir(model_dir / "checkpoint-1")
    _make_model_dir(model_dir / "checkpoint-2")
    profile = _eval_profile(tmp_path)

    result = _run_eval_only(config, profile)

    assert result.returncode == 0, result.stderr
    assert f"[dry-run] model={initial_model}" in result.stdout
    assert f"[dry-run] model={model_dir / 'checkpoint-1'}" in result.stdout
    assert f"[dry-run] model={model_dir / 'checkpoint-2'}" in result.stdout
    assert "tasks=aime24,aime25,amc23" in result.stdout
    assert "tasks=math_500,gsm8k" not in result.stdout
    assert "Generate" not in result.stdout
    assert "Score and Compute" not in result.stdout
    assert not (tmp_path / "extended-results").exists()


def test_eval_only_fails_before_running_when_checkpoint_is_missing(tmp_path: Path) -> None:
    config, _, model_dir = _base_config(tmp_path)
    _make_model_dir(model_dir / "checkpoint-1")
    profile = _eval_profile(tmp_path)

    result = _run_eval_only(config, profile)

    assert result.returncode != 0
    assert f"step 2: {model_dir / 'checkpoint-2'}" in result.stderr
    assert "--- Step 0: Evaluate ---" not in result.stdout


def test_eval_profile_cannot_override_training_config(tmp_path: Path) -> None:
    config, _, model_dir = _base_config(tmp_path)
    _make_model_dir(model_dir / "checkpoint-1")
    _make_model_dir(model_dir / "checkpoint-2")
    profile = _eval_profile(tmp_path)
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["training"] = {"max_steps": 99}
    _write_json(profile, payload)

    result = _run_eval_only(config, profile)

    assert result.returncode != 0
    assert "forbidden keys: ['training']" in result.stderr


def test_eval_signature_changes_with_task_parameters() -> None:
    command = """
source scripts/_lib.sh
EVAL_COMPLETION_MODE=signature
first=$(_eval_done_marker /tmp/eval ttrl checkpoint-1 aime24)
second=$(_eval_done_marker /tmp/eval ttrl checkpoint-1 aime25)
printf '%s\n%s\n' "$first" "$second"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=PROJECT_ROOT,
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    first, second = result.stdout.strip().splitlines()
    assert first != second
    assert "/.eval_markers/ttrl-" in first
    assert "/.eval_markers/ttrl-" in second
