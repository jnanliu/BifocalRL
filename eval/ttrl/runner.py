#!/usr/bin/env python
# coding=utf-8
"""
Serial multi-model evaluation using TTRL-style scoring.

Evaluates one or more models on configurable datasets (MATH-TTT, AIME-TTT,
GPQA-TTT, etc.) and produces a comparison summary table.

Three predefined prompt modes (``--prompt_mode``):
  system  — training-style system prompt, no instruction suffix
  suffix  — no system prompt, TTRL suffix appended to user message (TTRL default)
  bare    — nothing added (raw problem only)

Usage:
    # ── Basic multi-model evaluation ───────────────────────────────────
    python -m eval.ttrl.runner \
        --model_paths /path/to/model1 /path/to/model2 \
        --datasets math_ttt aime_ttt \
        --scorer ttrl_math \
        --output_dir outputs/ablation/ttrl_eval

    # ── With prompt mode (system / suffix / bare) ──────────────────────
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_ttrl_eval \
        --model_paths /path/to/Qwen2.5-Math-1.5B \
        --prompt_mode bare \
        --datasets math_ttt aime_ttt gpqa_ttt \
        --output_dir outputs/ablation/ttrl_eval/bare

    # ── Prompt-mode comparison: run three times with different modes ────
    for MODE in system suffix bare; do
        CUDA_VISIBLE_DEVICES=0 python -m ablation.test_ttrl_eval \
            --model_paths /path/to/model \
            --prompt_mode $MODE \
            --datasets math_ttt aime_ttt \
            --output_dir outputs/ablation/ttrl_eval/$MODE
    done
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_ttrl_eval")

_PRESET_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant, designed to provided well-reasoned and "
    "detailed responses. You FIRST think about the reasoning process step by "
    "step and then provide the user with the answer. Please enclose your final "
    "answer in the box: \\boxed{Your Answer}. Please stop generation immediately "
    "after outputing the box."
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serial multi-model evaluation with TTRL-style scoring.",
    )
    parser.add_argument(
        "--model_paths", type=str, nargs="+", required=True,
        help="Paths to model checkpoints (evaluated serially).",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["math_ttt"],
        help="Dataset keys from recipes/datasets.json. Default: math_ttt.",
    )
    parser.add_argument(
        "--scorer", type=str, default="ttrl_math",
        help="Scorer name. Default: ttrl_math.",
    )
    parser.add_argument(
        "--num_generations", type=int, default=1,
        help="Number of generations per problem.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=None,
        choices=["system", "suffix", "bare"],
        help="Predefined prompt preset. "
        "'system' = training system prompt; "
        "'suffix' = TTRL suffix (TTRL default during train & eval); "
        "'bare' = nothing added (raw problem only). "
        "Overrides --system_prompt, --instruction_suffix, --enable_thinking.",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--instruction_suffix", type=str, default=None,
        help="Instruction suffix appended to user message (overrides dataset default).",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="none",
        choices=["true", "false", "none"],
        help="Control enable_thinking for chat template. "
        "'none' = don't pass (TTRL default).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of problems per dataset (for quick testing).",
    )
    parser.add_argument(
        "--dataset_overrides", type=str, default=None,
        help='JSON string of per-dataset overrides. Supported keys: '
        'num_generations, temperature, top_p, max_new_tokens. Example: '
        '\'{"aime25":{"num_generations":32,"temperature":0.6,'
        '"top_p":0.95,"max_new_tokens":4096}}\'',
    )

    # vLLM settings
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="Max model context length. Auto-computed if not set.",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ablation/ttrl_eval",
    )
    return parser.parse_args()


def _model_tag(model_path: str) -> str:
    """Derive a short tag from the model path for filenames/display."""
    return os.path.basename(model_path.rstrip("/"))


def _parse_dataset_overrides(raw: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Parse ``--dataset_overrides`` JSON string into a dict.

    Returns ``{dataset_key: {param: value, ...}, ...}``.
    Allowed override keys: ``num_generations``, ``temperature``, ``top_p``,
    and ``max_new_tokens``.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse --dataset_overrides JSON: %s", exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("--dataset_overrides must be a JSON object, got %s", type(parsed).__name__)
        return {}
    _ALLOWED = {"num_generations", "temperature", "top_p", "max_new_tokens"}
    out: Dict[str, Dict[str, Any]] = {}
    for ds_key, overrides in parsed.items():
        if not isinstance(overrides, dict):
            continue
        out[ds_key] = {k: v for k, v in overrides.items() if k in _ALLOWED}
    return out


def _metric_label(num_generations: int) -> str:
    """Return a human-readable metric label based on generation count."""
    if num_generations <= 1:
        return "pass@1"
    return f"avg@{num_generations}"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from dataclasses import replace as dc_replace

    from eval.hacrl.runner import EvalConfig, evaluate_math_dataset
    from utils.dataset import get_answer_column, get_ttrl_instruction_suffix
    from utils.generate import create_offline_llm, shutdown_offline_llm

    ds_overrides = _parse_dataset_overrides(args.dataset_overrides)

    # ── Apply prompt-mode preset ─────────────────────────────────────────
    _suffix_is_ttrl = False
    if args.prompt_mode == "system":
        args.system_prompt = _PRESET_SYSTEM_PROMPT
        args.instruction_suffix = ""
        args.enable_thinking = "false"
    elif args.prompt_mode == "suffix":
        args.system_prompt = None
        args.instruction_suffix = None
        args.enable_thinking = "none"
        _suffix_is_ttrl = True
    elif args.prompt_mode == "bare":
        args.system_prompt = None
        args.instruction_suffix = ""
        args.enable_thinking = "none"

    logger.info("Prompt config: mode=%s  system_prompt=%s  suffix=%s  enable_thinking=%s",
                args.prompt_mode or "(custom)",
                "yes" if args.system_prompt else "no",
                "(TTRL per-dataset)" if _suffix_is_ttrl else repr(args.instruction_suffix),
                args.enable_thinking)
    if ds_overrides:
        logger.info("Dataset overrides: %s", json.dumps(ds_overrides, separators=(",", ":")))

    # results[model_tag][dataset] = primary metric value
    all_results: Dict[str, Dict[str, float]] = {}
    all_detail_paths: Dict[str, Dict[str, str]] = {}
    # Track which metric is primary for each dataset
    ds_metric_labels: Dict[str, str] = {}
    model_tags: List[str] = []

    for model_idx, model_path in enumerate(args.model_paths):
        tag = _model_tag(model_path)
        if tag in all_results:
            tag = f"{tag}_{model_idx}"
        model_tags.append(tag)
        all_results[tag] = {}
        all_detail_paths[tag] = {}

        logger.info("")
        logger.info("=" * 70)
        logger.info("Model [%d/%d]: %s", model_idx + 1, len(args.model_paths), tag)
        logger.info("  path: %s", model_path)
        logger.info("=" * 70)

        model_output_dir = os.path.join(args.output_dir, tag)
        os.makedirs(model_output_dir, exist_ok=True)

        base_cfg = EvalConfig(
            model_path=model_path,
            model_tag="eval",
            checkpoint_tag=tag,
            system_prompt=args.system_prompt,
            instruction_suffix=args.instruction_suffix,
            enable_thinking=args.enable_thinking,
            num_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_samples=args.max_samples,
            output_root=model_output_dir,
            seed=args.seed,
            scorer=args.scorer,
            use_offline=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            vllm_dtype=args.dtype,
            vllm_max_model_len=args.max_model_len,
        )

        llm = create_offline_llm(
            model_path=model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            seed=args.seed,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
        )

        try:
            for ds in args.datasets:
                # Start from base config
                cfg = base_cfg

                # Apply TTRL suffix if in suffix mode
                if _suffix_is_ttrl:
                    ds_suffix = get_ttrl_instruction_suffix(ds)
                    cfg = dc_replace(cfg, instruction_suffix=ds_suffix)

                # Apply per-dataset sampling and generation-length overrides.
                overrides = ds_overrides.get(ds, {})
                if overrides:
                    cfg = dc_replace(cfg, **overrides)

                ds_num_gen = cfg.num_generations
                metric_label = _metric_label(ds_num_gen)
                ds_metric_labels[ds] = metric_label

                logger.info("")
                logger.info("-" * 50)
                logger.info("  Dataset: %s  metric=%s  num_gen=%d  temp=%.2f  top_p=%.2f  suffix=%r",
                            ds, metric_label, ds_num_gen, cfg.temperature, cfg.top_p,
                            cfg.instruction_suffix if cfg.instruction_suffix is not None else "(dataset default)")
                logger.info("-" * 50)

                t0 = time.time()
                try:
                    answer_col = get_answer_column(ds)
                    result_path = evaluate_math_dataset(
                        ds, answer_col, cfg, offline_llm=llm,
                    )
                    elapsed = time.time() - t0

                    with open(result_path, "r") as f:
                        result_data = json.load(f)
                    summary = result_data.get("summary", {})
                    pass_at_k = summary.get("pass_at_k", 0.0)
                    mean_score = summary.get("mean_score", 0.0)

                    primary = mean_score if ds_num_gen > 1 else pass_at_k
                    all_results[tag][ds] = primary
                    all_detail_paths[tag][ds] = result_path

                    logger.info(
                        "  -> %s=%.4f  (pass@k=%.4f  mean_score=%.4f  %d/%d correct)  [%.1fs]",
                        metric_label, primary,
                        pass_at_k, mean_score,
                        summary.get("correct_any", 0),
                        summary.get("total", 0),
                        elapsed,
                    )
                except Exception as exc:
                    logger.error("  -> FAILED: %s", exc)
                    all_results[tag][ds] = -1.0
        finally:
            shutdown_offline_llm(llm)

    # ── Comparison summary ────────────────────────────────────────────────
    comparison = {
        "models": model_tags,
        "datasets": args.datasets,
        "scorer": args.scorer,
        "prompt_mode": args.prompt_mode,
        "enable_thinking": args.enable_thinking,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "dataset_overrides": ds_overrides,
        "dataset_metrics": ds_metric_labels,
        "results": all_results,
        "detail_paths": all_detail_paths,
    }

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # ── Print comparison table ────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)

    # Column headers: dataset name + metric label
    ds_headers = [f"{ds}({ds_metric_labels.get(ds, '?')})" for ds in args.datasets]
    ds_col_width = max(len(h) for h in ds_headers) + 2
    model_col_width = max(len(t) for t in model_tags) + 2
    header = f"{'Model':<{model_col_width}}"
    for h in ds_headers:
        header += f"  {h:>{ds_col_width}}"
    logger.info(header)
    logger.info("-" * len(header))

    for tag in model_tags:
        row = f"{tag:<{model_col_width}}"
        for ds in args.datasets:
            val = all_results[tag].get(ds, -1.0)
            if val < 0:
                row += f"  {'FAIL':>{ds_col_width}}"
            else:
                row += f"  {val:>{ds_col_width}.4f}"
        logger.info(row)

    logger.info("")
    logger.info("Results saved to: %s", summary_path)


if __name__ == "__main__":
    main()
