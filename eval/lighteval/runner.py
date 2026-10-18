#!/usr/bin/env python
# coding=utf-8

"""
LightEval-based evaluation wrapper (open-r1-intuitor style).

This script calls the `lighteval vllm` CLI if available.

Notes:
- This matches open-r1-intuitor's evaluation approach, but requires `lighteval` to be installed in the runtime env.
- For "local datasets", LightEval will use whatever `datasets` can resolve. If you want fully offline evaluation,
  pre-download/cache datasets and set HF offline env vars (we expose flags for that).
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _require_lighteval() -> str:
    exe = shutil.which("lighteval")
    if not exe:
        raise RuntimeError(
            "Could not find `lighteval` in PATH. Install it (e.g. `pip install lighteval`) or use the custom evaluator."
        )
    return exe


def _is_legacy_lighteval() -> bool:
    """
    LightEval task string formats differ across versions.

    - Legacy (e.g. 0.9.x): expects `suite|task|fewshot|truncate_few_shots`
      (e.g. `lighteval|math_500|0|0`). Sample limits are a separate CLI option.
    - Newer: often accepts `task|fewshot` (e.g. `gsm8k|0`)
    """
    try:
        import lighteval  # type: ignore

        ver = getattr(lighteval, "__version__", "") or ""
        parts = ver.split(".")
        major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        # 0.12+ uses the newer CLI docs/formats.
        return (major, minor) < (0, 12)
    except Exception:
        # Default to legacy to be safe (won't break newer if user passes full strings).
        return True


def _task_string(tasks: List[str]) -> str:
    legacy = _is_legacy_lighteval()
    parts: List[str] = []
    for t in tasks:
        t = t.strip()
        if not t:
            continue
        # Accept open-r1 / LightEval legacy format:
        # - "suite|task|fewshot|truncate_few_shots" (fully qualified) -> keep
        # - "suite|task" or "suite|task|fewshot" -> normalize to 4-field legacy
        if t.count("|") >= 3:
            if legacy:
                fields = t.split("|")
                if len(fields) != 4:
                    raise ValueError(
                        f"Invalid legacy LightEval task {t!r}; expected "
                        "suite|task|fewshot|truncate_few_shots."
                    )
                try:
                    int(fields[2])
                    truncate_few_shots = int(fields[3])
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric fields in LightEval task {t!r}.") from exc
                if truncate_few_shots not in (0, 1):
                    raise ValueError(
                        f"Invalid truncate_few_shots={truncate_few_shots} in LightEval task {t!r}; "
                        "use 0 or 1. Limit evaluated examples with --max_samples instead."
                    )
            parts.append(t)
            continue
        if t.startswith(("lighteval|", "extended|", "custom|")):
            if t.count("|") == 1:
                parts.append(f"{t}|0|0")
                continue
            if t.count("|") == 2:
                parts.append(f"{t}|0")
                continue

        if legacy:
            # Accept "task" or "task|fewshot".
            if t.count("|") == 1:
                task_name, fewshot = t.split("|", 1)
                parts.append(f"lighteval|{task_name}|{fewshot}|0")
            else:
                parts.append(f"lighteval|{t}|0|0")
        else:
            # Newer formats accept "task|fewshot".
            if "|" in t:
                parts.append(t)
            else:
                parts.append(f"{t}|0")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate with LightEval (vLLM backend).")
    parser.add_argument("--model_path", type=str, required=True, help="HF model/checkpoint path")
    parser.add_argument("--tasks", type=str, default="math_500,gsm8k", help="Comma-separated tasks")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LightEval results")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")

    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tokenizer chat template to format prompts (auto-disabled if tokenizer has no chat_template).",
    )
    parser.add_argument(
        "--custom_tasks",
        type=str,
        default=None,
        help="Optional custom tasks module passed to `lighteval ... --custom-tasks` (e.g. for offline dataset redirect).",
    )
    parser.add_argument(
        "--local_datasets_root",
        type=str,
        default=None,
        help="Optional local datasets root used by custom dataset scripts (exported as LOCAL_DATASETS_ROOT).",
    )

    # vLLM settings (kept close to open-r1 defaults but adjustable)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_length", type=int, default=4096)
    # NOTE: With LightEval v0.9.x, setting max_new_tokens==max_model_length (32768) causes
    # the prompt to be truncated to 0 tokens (see lighteval/models/vllm/vllm_model.py).
    # open-r1 recipes typically use 3072 for math-style eval. You can override if needed.
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of examples per task; separate from the task string.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation (vLLM).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument(
        "--save_details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-sample detail files (--save-details / --no-save-details).",
    )

    # Offline mode helpers (for local/cached datasets)
    parser.add_argument("--datasets_offline", action="store_true", default=False)
    parser.add_argument("--hf_home", type=str, default=None)
    parser.add_argument("--datasets_cache", type=str, default=None, help="Override HF_DATASETS_CACHE")
    parser.add_argument("--transformers_cache", type=str, default=None, help="Override TRANSFORMERS_CACHE")

    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be a positive integer when provided")

    _require_lighteval()

    # Best-effort: patch legacy checkpoints missing trust_remote_code python files before LightEval/vLLM loads it.
    try:
        from utils.generate import _ensure_trust_remote_code_files  # type: ignore

        _ensure_trust_remote_code_files(args.model_path)
    except Exception:
        pass

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_str = _task_string(tasks)

    use_chat_template = bool(args.use_chat_template)
    if use_chat_template:
        # LightEval will call `tokenizer.apply_chat_template(...)` when `--use-chat-template` is set.
        # If the tokenizer does not provide a chat_template, transformers raises:
        # "Cannot use chat template functions because tokenizer.chat_template is not set..."
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
            has_template = bool(getattr(tok, "chat_template", None))
            if not has_template:
                use_chat_template = False
                print(
                    "[WARN] Tokenizer has no chat_template; disabling --use-chat-template for LightEval.",
                    flush=True,
                )
        except Exception:
            # Be conservative: if we fail to inspect the tokenizer, keep previous behavior.
            pass

    model_args = (
        f"model_name={args.model_path},"
        f"dtype={args.dtype},"
        f"data_parallel_size={args.data_parallel_size},"
        f"max_model_length={args.max_model_length},"
        f"gpu_memory_utilization={args.gpu_memory_utilization},"
        f"trust_remote_code={str(bool(args.trust_remote_code))},"
        f"generation_parameters={{max_new_tokens:{args.max_new_tokens},temperature:{args.temperature},top_p:{args.top_p},seed:{args.seed}}}"
    )

    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Prevent lighteval from re-downloading NLTK punkt on every run.
    # Uses NLTK_DATA if already set in the environment, otherwise falls back to the
    # standard download location used when punkt was first installed.
    env.setdefault("NLTK_DATA", os.path.join(os.path.expanduser("~"), "nltk_data"))
    if args.datasets_offline:
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if args.hf_home:
        env["HF_HOME"] = args.hf_home
    if args.datasets_cache:
        env["HF_DATASETS_CACHE"] = args.datasets_cache
    if args.transformers_cache:
        env["TRANSFORMERS_CACHE"] = args.transformers_cache
    if args.local_datasets_root:
        env["LOCAL_DATASETS_ROOT"] = args.local_datasets_root
    # Ensure the repo root is importable (for module-based --custom-tasks).
    cwd = os.getcwd()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = cwd if not existing_pp else f"{cwd}{os.pathsep}{existing_pp}"

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "lighteval",
        "vllm",
        model_args,
        task_str,
        "--output-dir",
        args.output_dir,
    ]
    if args.save_details:
        cmd.append("--save-details")
    print(f"[lighteval-wrapper] task_str={task_str}", flush=True)
    if use_chat_template:
        cmd.append("--use-chat-template")
    if args.system_prompt:
        cmd.extend(["--system-prompt", args.system_prompt])
    if args.custom_tasks:
        cmd.extend(["--custom-tasks", args.custom_tasks])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])

    _run_with_idle_watchdog(cmd, env=env, result_dir=args.output_dir)


def _run_with_idle_watchdog(
    cmd: List[str],
    *,
    env: dict | None = None,
    idle_timeout: int | None = None,
    poll_interval: float = 5.0,
    result_dir: str | None = None,
) -> None:
    """Run *cmd* while streaming its stdout, and force-kill it if it hangs.

    vLLM V1 can become silent either during expensive initialization or while
    failing to exit after evaluation. This wrapper bounds that silence and
    kills the process group. A killed run is accepted only when a non-empty
    result file already exists; otherwise it raises an explicit failure.
    """
    if idle_timeout is None:
        idle_timeout = int(os.environ.get("LIGHTEVAL_IDLE_TIMEOUT", "900"))

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    last_output = time.monotonic()
    killed_by_watchdog = False

    try:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], poll_interval)
            if ready:
                data = os.read(proc.stdout.fileno(), 65536)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                last_output = time.monotonic()
            else:
                if proc.poll() is not None:
                    break
                elapsed = time.monotonic() - last_output
                if elapsed > idle_timeout:
                    print(
                        f"\n[lighteval-wrapper] No output for {int(elapsed)}s; "
                        f"idle timeout reached. Force-killing process group (pgid={pgid}).",
                        flush=True,
                    )
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    killed_by_watchdog = True
                    break
    except KeyboardInterrupt:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        raise
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass

    try:
        returncode = proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        returncode = proc.wait(timeout=10)

    if killed_by_watchdog:
        result_files = []
        if result_dir:
            result_root = Path(result_dir)
            result_files = [
                path
                for path in result_root.rglob("*")
                if path.is_file() and "result" in path.name.lower() and path.stat().st_size > 0
            ]
        if not result_files:
            raise RuntimeError(
                f"LightEval produced no result files before the {idle_timeout}s idle timeout. "
                "The process was killed and the evaluation is incomplete."
            )
        print(
            f"[lighteval-wrapper] Results exist under {result_dir}; "
            "treating the forced kill as a cleanup-only termination.",
            flush=True,
        )
    elif returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


if __name__ == "__main__":
    main()
