#!/usr/bin/env python
# coding=utf-8
"""
ZeroEval adapter entrypoint (CRUXEval).

This script integrates WildEval/ZeroEval into this repo's evaluation pipeline:
- Runs ZeroEval's inference for CRUX ("crux") on a local HF checkpoint path
- Writes raw generations under our `output_root`
- Computes accuracy locally (compatible with ZeroEval's string-match CRUX metric)

Expected usage (from pipeline scripts):
  python -m src.evaluate_zeroeval --model_path .../checkpoint-k --model_tag a --checkpoint_tag checkpoint-k

Requirements:
- ZeroEval repo is cloned locally and its deps are installed in the current env.
- Provide its path via --zeroeval_dir or env ZEROEVAL_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from utils.trl_compat import maybe_apply_chat_template

from utils.generate import generate_responses, _ensure_trust_remote_code_files
from eval.common.cruxeval import load_cruxeval_jsonl, default_cruxeval_path


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _project_root() -> Path:
    # src/ -> project root
    return Path(__file__).resolve().parent.parent


def _resolve_local_model_path(model_path: str) -> Path:
    """
    Resolve a local checkpoint path robustly.

    Why: In `zeroeval_repo` mode we run ZeroEval with `cwd=zeroeval_dir`, so any relative
    `--model_name` would be interpreted relative to the ZeroEval repo, not this project.
    That makes vLLM fail to find `config.json` and it then treats the string as a HF repo id.
    """
    p = Path(str(model_path)).expanduser()

    # 1) If caller already gave an absolute path, trust it.
    if p.is_absolute():
        return p.resolve()

    # 2) If relative path exists from current working directory, use it.
    if p.exists():
        return p.resolve()

    # 3) Otherwise interpret it as relative to this repo root.
    cand = (_project_root() / p).resolve()
    if cand.exists():
        return cand

    raise FileNotFoundError(
        "Model checkpoint path not found.\n"
        f"- provided: {model_path}\n"
        f"- tried (cwd-relative): {p}\n"
        f"- tried (repo-relative): {cand}\n"
        "Tip: pass an absolute path or a path relative to the co-learning repo root."
    )


def _resolve_zeroeval_dir(cli: Optional[str]) -> Path:
    if cli:
        p = Path(cli).expanduser().resolve()
        return p
    env = os.environ.get("ZEROEVAL_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (_project_root() / "third_party" / "ZeroEval").resolve()


def _check_zeroeval_layout(zeroeval_dir: Path) -> None:
    required = [
        zeroeval_dir / "src" / "unified_infer.py",
        zeroeval_dir / "src" / "task_configs.py",
        zeroeval_dir / "src" / "unified_utils.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "ZeroEval repo not found or incomplete.\n"
            f"- zeroeval_dir: {zeroeval_dir}\n"
            f"- missing: {missing}\n"
            "Please clone WildEval/ZeroEval and set ZEROEVAL_DIR (or pass --zeroeval_dir)."
        )


def _extract_first_complete_json(s: str) -> Optional[dict]:
    """
    Copy of ZeroEval's brace-stack parser (best-effort).
    """
    if not s:
        return None
    stack: List[int] = []
    first_start: Optional[int] = None
    for i, ch in enumerate(s):
        if ch == "{":
            stack.append(i)
            if first_start is None:
                first_start = i
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and first_start is not None:
                    frag = s[first_start : i + 1]
                    try:
                        return json.loads(frag.replace("\n", ""))
                    except Exception:
                        return None
                    finally:
                        first_start = None
    return None


_ANSWER_QUOTED_RE = re.compile(r'"answer"\s*:\s*"([^"]*?)"')
_ANSWER_UNQUOTED_RE = re.compile(r'"answer"\s*:\s*([^,\s\}]+)')
_ANSWER_NOQUOTEKEY_RE = re.compile(r"\banswer\s*:\s*([^,\s\}]+)")


def _extract_answer_best_effort(text: str) -> Optional[str]:
    """
    ZeroEval asks the model to output JSON. For robustness we:
    - try parsing the first complete JSON object
    - fall back to regex extraction (quoted/unquoted)
    """
    if text is None:
        return None
    s = str(text)
    obj = _extract_first_complete_json(s)
    if isinstance(obj, dict) and "answer" in obj:
        try:
            return str(obj["answer"]).strip().strip("'\"").replace("\n", "\\n")
        except Exception:
            pass

    m = _ANSWER_QUOTED_RE.search(s)
    if m:
        return str(m.group(1)).strip().strip("'\"").replace("\n", "\\n")

    m = _ANSWER_UNQUOTED_RE.search(s)
    if m:
        return str(m.group(1)).strip().strip("'\"").replace("\n", "\\n")

    m = _ANSWER_NOQUOTEKEY_RE.search(s)
    if m:
        return str(m.group(1)).strip().strip("'\"").replace("\n", "\\n")

    return None


@dataclass(frozen=True)
class CruxEvalResult:
    total: int
    correct: int
    no_answer: int

    @property
    def acc(self) -> float:
        return (self.correct / self.total) if self.total else 0.0

    @property
    def no_answer_rate(self) -> float:
        return (self.no_answer / self.total) if self.total else 0.0


def _eval_crux_file(path: Path) -> Tuple[CruxEvalResult, List[Dict[str, Any]]]:
    """
    Evaluate a ZeroEval raw outputs JSON produced by `unified_infer.py` on data_name=crux.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    no_answer = 0
    parsed: List[Dict[str, Any]] = []

    for item in data:
        out_list = item.get("output", [])
        pred_text = out_list[0] if isinstance(out_list, list) and out_list else ""
        pred_ans = _extract_answer_best_effort(pred_text)
        gold = item.get("answer", "")
        gold_s = str(gold).strip().strip("'\"")

        matched = False
        if pred_ans is None:
            no_answer += 1
        else:
            pred_s = str(pred_ans).strip().strip("'\"")
            # ZeroEval uses string match on normalized answers.
            if pred_s == gold_s or (pred_s == "" and gold_s == ""):
                matched = True
                correct += 1

        parsed.append(
            {
                "id": item.get("id") or item.get("session_id"),
                "question": item.get("question"),
                "gold": gold_s,
                "pred": pred_ans,
                "matched": matched,
                "raw_output": pred_text,
            }
        )

    return CruxEvalResult(total=total, correct=correct, no_answer=no_answer), parsed


def _construct_local_crux_prompt(row: Dict[str, Any]) -> str:
    """
    Local CRUXEval prompt (offline) that mimics ZeroEval's "answer" JSON convention.

    Our local dataset rows typically contain:
      - code
      - input
      - output (gold)
    """
    code = row.get("code", "")
    input_val = row.get("input", "")
    return (
        "You are given a Python function and an input. Predict the output of executing the function.\n"
        "Return ONLY a JSON object with a single key \"answer\".\n"
        "The value must be a valid Python literal (e.g., \"hello\", 5, [1,2], {\"a\":1}).\n"
        "Do not output explanations.\n\n"
        f"[python]\n{code}\n[/python]\n\n"
        f"Input: {input_val}\n"
        "Output JSON:"
    )


def _render_chat_prompt(tokenizer, user_prompt: str, system_prompt: Optional[str]) -> str:
    """
    Render a (system,user) message pair through the model tokenizer chat template if available.
    Falls back to raw user prompt if template is missing.
    Uses enable_thinking=False to match HACRL convention.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    messages: list[dict] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if has_template:
        try:
            return str(tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            ))
        except Exception:
            pass
    # No template: return plain text.
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt


def _run_local_offline_cruxeval(
    *,
    model_path: str,
    model_pretty_name: str,
    output_folder: Path,
    dataset_path: Optional[str],
    max_samples: Optional[int],
    system_prompt: Optional[str],
    vllm_host: str,
    vllm_port: int,
    vllm_batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    auto_start_server: bool,
    auto_stop_server: bool,
    seed: Optional[int] = 42,
) -> Path:
    """
    Offline CRUXEval:
    - loads our local jsonl dataset
    - runs vLLM generation via our OpenAI-compatible client
    - writes a ZeroEval-like raw json file: [{id, question, answer, output:[<raw>]}]
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Patch legacy checkpoints missing trust_remote_code files before tokenizer/vLLM loads it.
    try:
        _ensure_trust_remote_code_files(model_path)
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    project_root = str(_project_root())
    ds_path = dataset_path or default_cruxeval_path(project_root)
    rows = load_cruxeval_jsonl(ds_path, max_samples=max_samples)

    prompts: List[str] = []
    payload_rows: List[Dict[str, Any]] = []
    for r in rows:
        user_prompt = _construct_local_crux_prompt(r)
        prompt_text = _render_chat_prompt(tok, user_prompt, system_prompt)
        prompts.append(prompt_text)
        payload_rows.append(
            {
                "id": r.get("id"),
                "question": user_prompt,
                "answer": r.get("output", ""),
            }
        )

    gens = generate_responses(
        model_path=model_path,
        prompts=prompts,
        num_generations=1,
        batch_size=int(vllm_batch_size),
        host=str(vllm_host),
        port=int(vllm_port),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        repetition_penalty=float(repetition_penalty),
        auto_start_server=bool(auto_start_server),
        auto_stop_server=bool(auto_stop_server),
        # Prefer bf16 if the server was started by us; otherwise the existing server's dtype applies.
        dtype=str(os.environ.get("CO_VLLM_DTYPE", "bfloat16")),
        seed=seed,
    )
    if len(gens) != len(payload_rows):
        raise RuntimeError(f"Expected {len(payload_rows)} generations, got {len(gens)}")

    # Save ZeroEval-like raw file.
    for pr, g in zip(payload_rows, gens):
        pr["output"] = [g]

    raw_file = output_folder / f"{model_pretty_name}.json"
    with raw_file.open("w", encoding="utf-8") as f:
        json.dump(payload_rows, f, ensure_ascii=False, indent=2)
    return raw_file


def _run_unified_infer(
    *,
    zeroeval_dir: Path,
    model_path: str,
    model_pretty_name: str,
    output_folder: Path,
    engine: str,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_model_len: int,
    max_samples: Optional[int],
    seed: Optional[int] = 42,
) -> None:
    cmd: List[str] = [
        sys.executable,
        "src/unified_infer.py",
        "--engine",
        str(engine),
        "--data_name",
        "crux",
        "--model_name",
        str(model_path),
        "--model_pretty_name",
        str(model_pretty_name),
        "--output_folder",
        str(output_folder),
        "--tensor_parallel_size",
        str(int(tensor_parallel_size)),
        "--dtype",
        str(dtype),
        "--gpu_memory_utilization",
        str(float(gpu_memory_utilization)),
        "--batch_size",
        str(int(batch_size)),
        "--max_tokens",
        str(int(max_tokens)),
        "--temperature",
        str(float(temperature)),
        "--top_p",
        str(float(top_p)),
        "--repetition_penalty",
        str(float(repetition_penalty)),
        "--use_hf_conv_template",
        "--use_imend_stop",
    ]

    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    if int(max_model_len) != -1:
        cmd += ["--max_model_len", str(int(max_model_len))]

    if max_samples is not None:
        # unified_infer supports sharding via start/end index; we use end_index for quick debug runs.
        cmd += ["--start_index", "0", "--end_index", str(int(max_samples))]

    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # zero_eval_local.sh uses HF_HOME as download_dir; unified_infer expects --download_dir.
    # We keep it consistent: if HF_HOME is set, pass it through as download_dir.
    hf_home = env.get("HF_HOME", None)
    if hf_home:
        cmd += ["--download_dir", str(hf_home)]

    output_folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=str(zeroeval_dir), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRUX via ZeroEval.")
    parser.add_argument("--model_path", type=str, required=True, help="Local HF checkpoint path")
    parser.add_argument("--model_tag", type=str, required=True, help="Model tag (arbitrary string, used in output naming)")
    parser.add_argument("--checkpoint_tag", type=str, required=True, help="Checkpoint tag (e.g. checkpoint-3)")
    parser.add_argument("--output_root", type=str, default="outputs/zeroeval", help="Root folder for outputs")

    parser.add_argument("--zeroeval_dir", type=str, default=None, help="Path to cloned WildEval/ZeroEval repo")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get("ZEROEVAL_MODE", "local"),
        choices=["local", "zeroeval_repo"],
        help="Evaluation mode: local (offline dataset + our vLLM client) or zeroeval_repo (call unified_infer.py).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get("CRUXEVAL_DATASET_PATH", None),
        help="Local CRUXEval jsonl path (offline). Defaults to eval.cruxeval.default_cruxeval_path().",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=os.environ.get("CRUXEVAL_SYSTEM_PROMPT", None),
        help="Optional system prompt (used in local mode with chat template).",
    )

    # Local vLLM server settings (used in local mode).
    parser.add_argument("--vllm_host", type=str, default=os.environ.get("CRUXEVAL_VLLM_HOST", "localhost"))
    parser.add_argument("--vllm_port", type=int, default=int(os.environ.get("CRUXEVAL_VLLM_PORT", "8000")))
    parser.add_argument("--vllm_batch_size", type=int, default=int(os.environ.get("CRUXEVAL_VLLM_BATCH_SIZE", "16")))
    parser.add_argument("--auto_start_server", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto_stop_server", action=argparse.BooleanOptionalAction, default=False)

    # ZeroEval inference knobs (vLLM backend)
    parser.add_argument("--engine", type=str, default=os.environ.get("ZEROEVAL_ENGINE", "vllm"))
    parser.add_argument("--tensor_parallel_size", type=int, default=int(os.environ.get("ZEROEVAL_TP_SIZE", "1")))
    parser.add_argument("--dtype", type=str, default=os.environ.get("ZEROEVAL_DTYPE", "bfloat16"))
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("ZEROEVAL_GPU_MEMORY_UTILIZATION", "0.95")),
    )
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("ZEROEVAL_BATCH_SIZE", "4")))
    parser.add_argument("--max_tokens", type=int, default=int(os.environ.get("ZEROEVAL_MAX_TOKENS", "2048")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("ZEROEVAL_TEMPERATURE", "0")))
    parser.add_argument("--top_p", type=float, default=float(os.environ.get("ZEROEVAL_TOP_P", "1.0")))
    parser.add_argument("--repetition_penalty", type=float, default=float(os.environ.get("ZEROEVAL_RP", "1.0")))
    parser.add_argument("--max_model_len", type=int, default=int(os.environ.get("ZEROEVAL_MAX_MODEL_LEN", "-1")))
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick debug (uses end_index)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation (vLLM).")

    parser.add_argument(
        "--model_pretty_name",
        type=str,
        default=None,
        help="Filename prefix for ZeroEval raw outputs (defaults to co_<tag>_<checkpoint>)",
    )

    args = parser.parse_args()

    model_pretty = args.model_pretty_name or f"co_{args.model_tag}_{args.checkpoint_tag}"
    out_root = Path(args.output_root).expanduser()

    raw_dir = (out_root / "cruxeval" / "raw" / f"model-{args.model_tag}" / args.checkpoint_tag).resolve()
    results_dir = (out_root / "cruxeval").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Produce a raw outputs json.
    if args.mode == "zeroeval_repo":
        zeroeval_dir = _resolve_zeroeval_dir(args.zeroeval_dir)
        _check_zeroeval_layout(zeroeval_dir)
        resolved_model_path = _resolve_local_model_path(args.model_path)
        print(f"[zeroeval] resolved model_path: {resolved_model_path}", flush=True)
        _run_unified_infer(
            zeroeval_dir=zeroeval_dir,
            model_path=str(resolved_model_path),
            model_pretty_name=model_pretty,
            output_folder=raw_dir,
            engine=args.engine,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_model_len=args.max_model_len,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        # unified_infer.py changes output filename when start/end indices are set:
        # - default: <pretty>.json
        # - with start/end: <pretty>.<start>-<end>.json
        #
        # Our wrapper may pass end_index for quick debug (max_samples), so be flexible here and
        # locate the most likely output file.
        candidates: List[Path] = []
        # If we asked for a capped run, unified_infer will typically emit a range-suffixed file.
        if args.max_samples is not None:
            candidates.append(raw_dir / f"{model_pretty}.0-{int(args.max_samples)}.json")
        # Also accept the "full" filename (no range suffix).
        candidates.append(raw_dir / f"{model_pretty}.json")

        raw_file: Optional[Path] = next((p for p in candidates if p.exists()), None)
        if raw_file is None:
            # Fall back to any file with this prefix (helps when end_index was adjusted internally
            # or when the user ran with different indices previously).
            matches = sorted(raw_dir.glob(f"{model_pretty}*.json"))
            if len(matches) == 1:
                raw_file = matches[0]
            else:
                hint = "\n".join([f"- {p}" for p in (candidates + matches)])
                raise FileNotFoundError(
                    "ZeroEval did not produce an expected output file under raw_dir.\n"
                    f"- raw_dir: {raw_dir}\n"
                    f"- model_pretty_name: {model_pretty}\n"
                    f"- looked for:\n{hint if hint else '- (no candidates)'}"
                )
        zeroeval_dir_str = str(zeroeval_dir)
        backend = "zeroeval_repo"
    else:
        raw_file = _run_local_offline_cruxeval(
            model_path=args.model_path,
            model_pretty_name=model_pretty,
            output_folder=raw_dir,
            dataset_path=args.dataset_path,
            max_samples=args.max_samples,
            system_prompt=args.system_prompt,
            vllm_host=args.vllm_host,
            vllm_port=args.vllm_port,
            vllm_batch_size=args.vllm_batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            auto_start_server=args.auto_start_server,
            auto_stop_server=args.auto_stop_server,
            seed=args.seed,
        )
        zeroeval_dir_str = None
        backend = "local_offline"

    # 2) Evaluate locally (ZeroEval-compatible extraction + string match).
    metrics, parsed = _eval_crux_file(raw_file)

    # 3) Save summary + parsed details under our output_root.
    parsed_path = results_dir / f"parsed_{args.checkpoint_tag}_model-{args.model_tag}.json"
    with parsed_path.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    summary_path = results_dir / f"results_{args.checkpoint_tag}_model-{args.model_tag}.json"
    payload = {
        "summary": {
            "dataset": "cruxeval",
            "backend": backend,
            "zeroeval_dir": zeroeval_dir_str,
            "model_path": args.model_path,
            "model_tag": args.model_tag,
            "checkpoint": args.checkpoint_tag,
            "timestamp": _now_iso(),
            "acc": metrics.acc,
            "correct": metrics.correct,
            "total": metrics.total,
            "no_answer_rate": metrics.no_answer_rate,
            "no_answer": metrics.no_answer,
            "raw_file": str(raw_file),
            "parsed_file": str(parsed_path),
            "settings": {
                "mode": args.mode,
                "engine": args.engine if args.mode == "zeroeval_repo" else None,
                "tensor_parallel_size": args.tensor_parallel_size if args.mode == "zeroeval_repo" else None,
                "dtype": args.dtype if args.mode == "zeroeval_repo" else None,
                "gpu_memory_utilization": args.gpu_memory_utilization if args.mode == "zeroeval_repo" else None,
                "batch_size": args.batch_size if args.mode == "zeroeval_repo" else None,
                "vllm_host": args.vllm_host if args.mode == "local" else None,
                "vllm_port": args.vllm_port if args.mode == "local" else None,
                "vllm_batch_size": args.vllm_batch_size if args.mode == "local" else None,
                "dataset_path": args.dataset_path if args.mode == "local" else None,
                "system_prompt": args.system_prompt if args.mode == "local" else None,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_model_len": args.max_model_len,
                "max_samples": args.max_samples,
                "model_pretty_name": model_pretty,
            },
        }
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[zeroeval] cruxeval summary: {summary_path}")
    print(f"[zeroeval] cruxeval raw: {raw_file}")


if __name__ == "__main__":
    main()

