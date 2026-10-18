#!/usr/bin/env python
# coding=utf-8
"""
Test the impact of (max_completion_length, max_prompt_length) on MV answer extraction.

For each configuration group, generates completions using offline vLLM with prompt
truncation, then checks extraction success rate, individual accuracy, **majority-vote
(MV) accuracy**, and truncation rate.

Three predefined prompt modes (``--prompt_mode``):
  system  — training-style system prompt, no instruction suffix
  suffix  — no system prompt, TTRL suffix appended to user message (TTRL default)
  bare    — nothing added (raw problem only)

Usage:
    # ── Prompt-mode comparison on MATH-TTT (MV, n=16) ──────────────────
    # Mode A: system prompt
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_max_tokens_extraction \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --prompt_mode system \
        --configs 3072:1024 4096:1024 \
        --dataset_key math_ttt --scorer ttrl_math \
        --num_generations 16 \
        --output_dir outputs/ablation/max_tokens/system

    # Mode B: instruction suffix only
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_max_tokens_extraction \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --prompt_mode suffix \
        --configs 3072:1024 4096:1024 \
        --dataset_key math_ttt --scorer ttrl_math \
        --num_generations 16 \
        --output_dir outputs/ablation/max_tokens/suffix

    # Mode C: bare (raw problem only)
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_max_tokens_extraction \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --prompt_mode bare \
        --configs 3072:1024 4096:1024 \
        --dataset_key math_ttt --scorer ttrl_math \
        --num_generations 16 \
        --output_dir outputs/ablation/max_tokens/bare
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_max_tokens_extraction")

# ---------------------------------------------------------------------------
# Predefined prompt-mode presets
# ---------------------------------------------------------------------------
_PRESET_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant, designed to provided well-reasoned and "
    "detailed responses. You FIRST think about the reasoning process step by "
    "step and then provide the user with the answer. Please enclose your final "
    "answer in the box: \\boxed{Your Answer}. Please stop generation immediately "
    "after outputing the box."
)

def parse_config_pair(s: str) -> Tuple[int, int]:
    """Parse 'completion:prompt' string into (max_completion_length, max_prompt_length)."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Config must be 'max_completion:max_prompt', got '{s}'"
        )
    return int(parts[0]), int(parts[1])


def truncate_prompts(
    prompts: List[str],
    tokenizer,
    max_prompt_length: int,
) -> Tuple[List[str], int]:
    """Truncate prompts to max_prompt_length tokens. Returns (truncated_prompts, num_truncated)."""
    truncated = []
    num_truncated = 0
    for p in prompts:
        ids = tokenizer.encode(p)
        if len(ids) > max_prompt_length:
            ids = ids[:max_prompt_length]
            num_truncated += 1
            p = tokenizer.decode(ids, skip_special_tokens=False)
        truncated.append(p)
    return truncated, num_truncated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test (max_completion_length, max_prompt_length) impact on MV extraction."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--configs",
        type=parse_config_pair,
        nargs="+",
        default=[(4096, 1024), (2048, 512)],
        metavar="COMPLETION:PROMPT",
        help="Configuration groups as max_completion_length:max_prompt_length pairs. "
        "Default: 4096:1024 2048:512",
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default="math_500",
        help="Dataset key from recipes/datasets.json.",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="hacrl_prime_math",
        help="Scorer name (e.g. hacrl_prime_math, ttrl_math, ttrl_gpqa, hacrl_gsm8k, default).",
    )
    parser.add_argument(
        "--num_problems", type=int, default=100,
        help="Number of problems from the dataset.",
    )
    parser.add_argument(
        "--num_generations", type=int, default=2,
        help="Number of completions per problem.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
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
        "--system_prompt", type=str, default=None, help="Optional system prompt."
    )
    parser.add_argument(
        "--instruction_suffix", type=str, default=None,
        help="Instruction suffix appended to user message (overrides dataset config).",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="none",
        choices=["true", "false", "none"],
        help="Control enable_thinking for chat template rendering. "
        "'none' = don't pass the parameter (TTRL default), "
        "'true' = enable thinking, 'false' = disable thinking.",
    )

    # vLLM
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="Max model context length. Auto-computed if not set.",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/ablation/max_tokens_extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from transformers import AutoTokenizer

    from eval.scorers import get_scorer
    from utils.dataset import get_answer_column, get_ttrl_instruction_suffix, load_co_intuitor_dataset
    from utils.generate import create_offline_llm, shutdown_offline_llm
    from utils.mv import weighted_majority_voting
    from utils.prompt_utils import render_prompts

    # ── Apply prompt-mode preset ─────────────────────────────────────────
    if args.prompt_mode == "system":
        args.system_prompt = _PRESET_SYSTEM_PROMPT
        args.instruction_suffix = ""
        args.enable_thinking = "false"
    elif args.prompt_mode == "suffix":
        args.system_prompt = None
        args.instruction_suffix = get_ttrl_instruction_suffix(args.dataset_key)
        args.enable_thinking = "none"
    elif args.prompt_mode == "bare":
        args.system_prompt = None
        args.instruction_suffix = ""
        args.enable_thinking = "none"

    _et_map = {"true": True, "false": False, "none": None}
    chat_template_kwargs: Dict[str, Any] = {
        "enable_thinking": _et_map[args.enable_thinking],
    }

    logger.info("Prompt config: mode=%s  system_prompt=%s  suffix=%r  enable_thinking=%s",
                args.prompt_mode or "(custom)",
                "yes" if args.system_prompt else "no",
                args.instruction_suffix if args.instruction_suffix is not None else "(dataset default)",
                args.enable_thinking)

    # ── Load dataset ──────────────────────────────────────────────────────
    max_samples = args.num_problems if args.num_problems and args.num_problems > 0 else None
    if max_samples is None:
        logger.info(
            "num_problems=%s interpreted as ALL samples for dataset '%s'",
            args.num_problems,
            args.dataset_key,
        )

    problems_all = load_co_intuitor_dataset(
        dataset_key=args.dataset_key,
        max_samples=max_samples,
        system_prompt=args.system_prompt,
        instruction_suffix_override=args.instruction_suffix,
    )
    logger.info("Loaded %d problems from '%s'", len(problems_all), args.dataset_key)

    # ── Render prompts ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    raw_prompts = render_prompts(
        problems_all, tokenizer, apply_chat_template_kwargs=chat_template_kwargs,
    )
    logger.info("Rendered %d prompts", len(raw_prompts))

    if not raw_prompts:
        logger.error(
            "No prompts rendered for dataset '%s'. Check dataset config/split and num_problems.",
            args.dataset_key,
        )
        summary = {
            "model_path": args.model_path,
            "dataset_key": args.dataset_key,
            "scorer": args.scorer,
            "prompt_mode": args.prompt_mode,
            "system_prompt": args.system_prompt[:80] + "..." if args.system_prompt and len(args.system_prompt) > 80 else args.system_prompt,
            "instruction_suffix": args.instruction_suffix,
            "enable_thinking": args.enable_thinking,
            "num_problems": len(problems_all),
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "configs": [{"max_completion_length": c, "max_prompt_length": p} for c, p in args.configs],
            "results": {},
            "error": "No prompts rendered; nothing to evaluate.",
        }
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Wrote empty summary to: %s", summary_path)
        return

    prompt_token_lens = [len(tokenizer.encode(p)) for p in raw_prompts]
    logger.info(
        "Prompt token lengths: min=%d, max=%d, mean=%.0f",
        min(prompt_token_lens), max(prompt_token_lens),
        sum(prompt_token_lens) / len(prompt_token_lens),
    )

    # ── Answer column & scorer ────────────────────────────────────────────
    answer_col = get_answer_column(args.dataset_key)
    gold_answers = [p.get(answer_col, p.get("solution", "")) for p in problems_all]
    scorer = get_scorer(args.scorer, dataset_hint=args.dataset_key)
    logger.info("Using scorer: %s", args.scorer)

    # ── Create vLLM engine (max_model_len covers the largest config) ──────
    max_completion = max(c for c, _ in args.configs)
    max_prompt = max(p for _, p in args.configs)
    effective_max_model_len = args.max_model_len
    if effective_max_model_len is None:
        effective_max_model_len = max_prompt + max_completion + 64
        logger.info(
            "Auto max_model_len: %d (prompt) + %d (completion) + buffer → %d",
            max_prompt, max_completion, effective_max_model_len,
        )

    llm = create_offline_llm(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        trust_remote_code=True,
        max_model_len=effective_max_model_len,
    )

    # ── Run each config group ─────────────────────────────────────────────
    from vllm import SamplingParams

    all_results: Dict[str, Dict[str, Any]] = {}

    for max_compl, max_prompt_len in args.configs:
        config_tag = f"compl{max_compl}_prompt{max_prompt_len}"
        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "Config: max_completion_length=%d, max_prompt_length=%d", max_compl, max_prompt_len
        )
        logger.info("=" * 70)

        # Truncate prompts
        prompts, n_trunc = truncate_prompts(raw_prompts, tokenizer, max_prompt_len)
        if n_trunc > 0:
            logger.info("Truncated %d / %d prompts to %d tokens", n_trunc, len(prompts), max_prompt_len)
        else:
            logger.info("No prompts exceeded %d tokens", max_prompt_len)

        sp_kwargs: Dict[str, Any] = dict(
            n=args.num_generations,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(max_compl),
            logprobs=1,
            min_tokens=1,
        )
        if args.repetition_penalty != 1.0:
            sp_kwargs["repetition_penalty"] = float(args.repetition_penalty)
        if args.seed is not None:
            sp_kwargs["seed"] = int(args.seed)

        sampling_params = SamplingParams(**sp_kwargs)

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - t0
        logger.info("Generation took %.1fs", gen_time)

        # ── Analyze (individual + majority-vote) ─────────────────────────
        total = 0
        extracted_ok = 0
        correct = 0
        truncated = 0
        no_extract = 0
        details: List[Dict[str, Any]] = []

        mv_problems = 0
        mv_has_answer = 0
        mv_correct_count = 0
        mv_details: List[Dict[str, Any]] = []

        for i, req_output in enumerate(outputs):
            gold = gold_answers[i]
            problem_extracts: List[Optional[str]] = []

            for comp in req_output.outputs:
                total += 1
                text = comp.text
                finish = comp.finish_reason or "stop"
                is_truncated = finish == "length"
                if is_truncated:
                    truncated += 1

                result = scorer.score(text, gold)
                ext = result.pred_extracted.strip() if result.pred_extracted else ""
                has_extract = bool(ext)
                if has_extract:
                    extracted_ok += 1
                else:
                    no_extract += 1
                if result.is_correct:
                    correct += 1

                problem_extracts.append(ext if ext else None)

                details.append({
                    "problem_idx": i,
                    "config": config_tag,
                    "max_completion_length": max_compl,
                    "max_prompt_length": max_prompt_len,
                    "prompt_tokens": len(tokenizer.encode(prompts[i])),
                    "finish_reason": finish,
                    "is_truncated": is_truncated,
                    "pred_extracted": result.pred_extracted,
                    "gold_extracted": result.gold_extracted,
                    "has_extract": has_extract,
                    "is_correct": bool(result.is_correct),
                    "response_len_chars": len(text),
                    "response_len_tokens": len(comp.token_ids),
                })

            # ── Majority vote for this problem ────────────────────────────
            mv_problems += 1
            mv_vote = weighted_majority_voting(
                problem_extracts, [1.0] * len(problem_extracts),
            )
            mv_ans = mv_vote.selected_answer
            mv_is_correct = False
            if mv_ans is not None:
                mv_has_answer += 1
                if "gpqa" in args.dataset_key:
                    fake_resp = f"Answer: {mv_ans}"
                elif "gsm8k" in args.dataset_key:
                    fake_resp = f"#### {mv_ans}"
                else:
                    fake_resp = f"\\boxed{{{mv_ans}}}"
                mv_score = scorer.score(fake_resp, gold)
                mv_is_correct = bool(mv_score.is_correct)
            if mv_is_correct:
                mv_correct_count += 1

            mv_details.append({
                "problem_idx": i,
                "config": config_tag,
                "mv_answer": mv_ans,
                "mv_is_correct": mv_is_correct,
                "num_unique_clusters": len(mv_vote.answer_clusters),
                "answer_clusters": {k: v for k, v in mv_vote.answer_clusters.items()},
            })

        stats = {
            "max_completion_length": max_compl,
            "max_prompt_length": max_prompt_len,
            "total_completions": total,
            "extracted_ok": extracted_ok,
            "no_extract": no_extract,
            "extraction_rate": extracted_ok / total if total else 0,
            "correct": correct,
            "accuracy": correct / total if total else 0,
            "truncated": truncated,
            "truncation_rate": truncated / total if total else 0,
            "prompts_truncated": n_trunc,
            "generation_time_s": round(gen_time, 2),
            "mv_problems": mv_problems,
            "mv_has_answer": mv_has_answer,
            "mv_correct": mv_correct_count,
            "mv_accuracy": mv_correct_count / mv_problems if mv_problems else 0,
        }
        all_results[config_tag] = stats

        logger.info(
            "  extracted=%d/%d (%.1f%%) | ind_acc=%d (%.1f%%) | "
            "mv_acc=%d/%d (%.1f%%) | trunc=%d (%.1f%%) | prompt_trunc=%d",
            extracted_ok, total, stats["extraction_rate"] * 100,
            correct, stats["accuracy"] * 100,
            mv_correct_count, mv_problems, stats["mv_accuracy"] * 100,
            truncated, stats["truncation_rate"] * 100,
            n_trunc,
        )

        detail_path = os.path.join(args.output_dir, f"details_{config_tag}.json")
        with open(detail_path, "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

        mv_detail_path = os.path.join(args.output_dir, f"mv_details_{config_tag}.json")
        with open(mv_detail_path, "w") as f:
            json.dump(mv_details, f, indent=2, ensure_ascii=False)

    # ── Shutdown vLLM ─────────────────────────────────────────────────────
    shutdown_offline_llm(llm)

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "model_path": args.model_path,
        "dataset_key": args.dataset_key,
        "scorer": args.scorer,
        "prompt_mode": args.prompt_mode,
        "system_prompt": args.system_prompt[:80] + "..." if args.system_prompt and len(args.system_prompt) > 80 else args.system_prompt,
        "instruction_suffix": args.instruction_suffix,
        "enable_thinking": args.enable_thinking,
        "num_problems": len(problems_all),
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "configs": [{"max_completion_length": c, "max_prompt_length": p} for c, p in args.configs],
        "results": all_results,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 90)
    logger.info("SUMMARY  (n=%d per problem)", args.num_generations)
    logger.info("=" * 90)
    logger.info(
        "%-12s %-10s %-7s %-11s %-9s %-9s %-9s %-10s",
        "completion", "prompt", "total",
        "extract%", "ind_acc%", "mv_acc%", "trunc%", "p_trunc",
    )
    logger.info("-" * 90)
    for tag in all_results:
        r = all_results[tag]
        logger.info(
            "%-12d %-10d %-7d %-11.1f %-9.1f %-9.1f %-9.1f %-10d",
            r["max_completion_length"], r["max_prompt_length"],
            r["total_completions"],
            r["extraction_rate"] * 100,
            r["accuracy"] * 100,
            r["mv_accuracy"] * 100,
            r["truncation_rate"] * 100,
            r["prompts_truncated"],
        )
    logger.info("")
    logger.info("Results saved to: %s", summary_path)


if __name__ == "__main__":
    main()
