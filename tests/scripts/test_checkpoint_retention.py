from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIB = PROJECT_ROOT / "scripts" / "_lib.sh"
RUNNER = PROJECT_ROOT / "scripts" / "run_single.sh"
CONFIG = (
    PROJECT_ROOT
    / "recipes"
    / "configs"
    / "mixed_reward"
    / "single"
    / "Qwen2.5-3B"
    / "single_intuitor.json"
)


def _make_checkpoint(
    root: Path,
    step: int,
    *,
    manifest_step: int | None = None,
    include_manifest: bool = True,
    include_state: bool = True,
) -> None:
    checkpoint = root / "model" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    if include_manifest:
        (checkpoint / "runtime_manifest.json").write_text(
            json.dumps(
                {
                    "pipeline_step": step if manifest_step is None else manifest_step,
                    "optimizer_state_persisted": include_state,
                    "saved_training_state_tag": f"checkpoint-{step}" if include_state else None,
                }
            ),
            encoding="utf-8",
        )
    if include_state:
        state = root / "training_state" / f"checkpoint-{step}"
        state.mkdir(parents=True)
        (state / "mp_rank_00_model_states.pt").write_bytes(b"state")


def _run_bash(command: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        cwd=PROJECT_ROOT,
        env={**os.environ, **(env or {})},
        check=True,
        text=True,
        capture_output=True,
    )


def test_checkpoint_completeness_requires_matching_manifest_and_state(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, 4)
    _make_checkpoint(tmp_path, 5, manifest_step=4)
    _make_checkpoint(tmp_path, 6, include_state=False)

    result = _run_bash(
        f"""
        source {LIB!s}
        checkpoint_pair_is_complete {tmp_path / 'model'} {tmp_path / 'training_state'} 4 true
        ! checkpoint_pair_is_complete {tmp_path / 'model'} {tmp_path / 'training_state'} 5 true
        ! checkpoint_pair_is_complete {tmp_path / 'model'} {tmp_path / 'training_state'} 6 true
        latest_complete_checkpoint_step {tmp_path / 'model'} {tmp_path / 'training_state'} true
        """
    )

    assert result.stdout.strip().endswith("4")


def test_prune_checkpoint_pairs_uses_numeric_steps_and_keeps_pairs(tmp_path: Path) -> None:
    for step in (2, 9, 10):
        _make_checkpoint(tmp_path, step)
    _make_checkpoint(tmp_path, 11, include_manifest=False)
    orphan_state = tmp_path / "training_state" / "checkpoint-12"
    orphan_state.mkdir(parents=True)
    (orphan_state / "state.pt").write_bytes(b"orphan")

    _run_bash(
        f"""
        source {LIB!s}
        prune_checkpoint_pairs {tmp_path / 'model'} {tmp_path / 'training_state'} true 2 false
        """
    )

    assert sorted(path.name for path in (tmp_path / "model").glob("checkpoint-*")) == [
        "checkpoint-10",
        "checkpoint-9",
    ]
    assert sorted(path.name for path in (tmp_path / "training_state").glob("checkpoint-*")) == [
        "checkpoint-10",
        "checkpoint-9",
    ]


def test_prune_checkpoint_pairs_can_keep_only_latest(tmp_path: Path) -> None:
    for step in (7, 8):
        _make_checkpoint(tmp_path, step)

    _run_bash(
        f"""
        source {LIB!s}
        prune_checkpoint_pairs {tmp_path / 'model'} {tmp_path / 'training_state'} true 1 false
        """
    )

    assert [path.name for path in (tmp_path / "model").glob("checkpoint-*")] == ["checkpoint-8"]
    assert [path.name for path in (tmp_path / "training_state").glob("checkpoint-*")] == [
        "checkpoint-8"
    ]


def test_run_single_auto_resumes_and_removes_incomplete_tail(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "latest_checkpoint"
    _make_checkpoint(checkpoint_root, 6)
    _make_checkpoint(checkpoint_root, 7, include_manifest=False)

    result = subprocess.run(
        ["bash", str(RUNNER), str(CONFIG)],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "LATEST_CHECKPOINT_DIR": str(checkpoint_root),
            "MAX_STEPS": "6",
            "START_STEP": "1",
            "KEEP_ALL_CHECKPOINTS": "false",
            "CHECKPOINT_KEEP_LIMIT": "2",
            "LIGHTEVAL_ENABLED": "false",
            "ZEROEVAL_ENABLED": "false",
            "HACRL_EVAL_ENABLED": "false",
            "TTRL_EVAL_ENABLED": "false",
            "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        },
        check=True,
        text=True,
        capture_output=True,
    )

    assert "latest complete checkpoint is step 6; continuing at step 7" in result.stdout
    assert (checkpoint_root / "model" / "checkpoint-6").is_dir()
    assert (checkpoint_root / "training_state" / "checkpoint-6").is_dir()
    assert not (checkpoint_root / "model" / "checkpoint-7").exists()
    assert not (checkpoint_root / "training_state" / "checkpoint-7").exists()


def test_run_single_does_not_delete_history_without_a_complete_pair(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "latest_checkpoint"
    _make_checkpoint(checkpoint_root, 2, include_manifest=False)

    result = subprocess.run(
        ["bash", str(RUNNER), str(CONFIG)],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "LATEST_CHECKPOINT_DIR": str(checkpoint_root),
            "MAX_STEPS": "2",
            "START_STEP": "1",
            "KEEP_ALL_CHECKPOINTS": "false",
            "CHECKPOINT_KEEP_LIMIT": "2",
            "LIGHTEVAL_ENABLED": "false",
            "ZEROEVAL_ENABLED": "false",
            "HACRL_EVAL_ENABLED": "false",
            "TTRL_EVAL_ENABLED": "false",
            "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Refusing to delete historical output automatically" in result.stderr
    assert (checkpoint_root / "model" / "checkpoint-2").is_dir()
    assert (checkpoint_root / "training_state" / "checkpoint-2").is_dir()
