#!/usr/bin/env python
# coding=utf-8
"""
Test the MV-gated intuitor reward: R = R(MV) * sigmoid(R(intuitor)).

MV provides a hard binary gate (1 if response matches pseudo-label, else 0);
sigmoid bounds the intuitor SCE score to (0, 1).  Generates rollouts via
offline vLLM, computes raw MV rewards, raw SCE scores, and the gated product,
then prints a side-by-side comparison table per prompt.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_mv_gated_reward \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --dataset_key math_500 \
        --num_problems 10 \
        --num_generations 4 \
        --output_dir outputs/ablation/mv_gated_reward

    # With TTRL scorer on math_ttt dataset:
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_mv_gated_reward \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --dataset_key math_ttt --scorer ttrl_math \
        --num_problems 10 --num_generations 8 \
        --output_dir outputs/ablation/mv_gated_reward_ttrl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_mv_gated_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test MV-gated intuitor reward: R = R(MV) * sigmoid(R(intuitor))"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_key", type=str, default="math_500")
    parser.add_argument("--scorer", type=str, default="hacrl_prime_math")
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", type=str, default=None)

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--scoring_batch_size", type=int, default=4)

    parser.add_argument("--output_dir", type=str, default="outputs/ablation/mv_gated_reward")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from transformers import AutoTokenizer

    from utils.dataset import load_co_intuitor_dataset, get_answer_column
    from utils.generate import create_offline_llm, generate_sequences, shutdown_offline_llm
    from utils.prompt_utils import render_prompts
    from utils.mv import rewards_from_mv_pseudolabel, extract_answers, weighted_majority_voting
    from utils.reward import score_responses
    from utils.model_utils import get_model
    from eval.scorers import get_scorer

    # ── Load dataset ──────────────────────────────────────────────────
    max_samples = args.num_problems if args.num_problems > 0 else None
    problems = load_co_intuitor_dataset(
        dataset_key=args.dataset_key,
        max_samples=max_samples,
        system_prompt=args.system_prompt,
    )
    logger.info("Loaded %d problems from '%s'", len(problems), args.dataset_key)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = render_prompts(
        problems, tokenizer,
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    # ── Generate rollouts ─────────────────────────────────────────────
    effective_max_model_len = args.max_model_len
    if effective_max_model_len is None:
        effective_max_model_len = (args.max_prompt_length or 512) + args.max_completion_length + 64

    logger.info("Creating offline vLLM engine...")
    llm = create_offline_llm(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=effective_max_model_len,
    )

    t0 = time.time()
    gen_out = generate_sequences(
        model_path=args.model_path,
        prompts=prompts,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=effective_max_model_len,
        llm=llm,
    )
    gen_time = time.time() - t0
    shutdown_offline_llm(llm)
    flat_responses = gen_out.response_texts
    logger.info("Generated %d responses in %.1fs", len(flat_responses), gen_time)

    # ── Compute MV rewards ────────────────────────────────────────────
    num_prompts = len(problems)
    g = args.num_generations
    import random

    mv_all: List[List[float]] = []
    mv_answers: List[Optional[str]] = []
    for i in range(num_prompts):
        start, end = i * g, (i + 1) * g
        rng = random.Random(args.seed + i)
        r_i, vote = rewards_from_mv_pseudolabel(flat_responses[start:end], rng=rng)
        mv_all.append(r_i)
        mv_answers.append(vote.selected_answer)

    mv_t = torch.tensor(mv_all, dtype=torch.float32)  # (P, G)

    # ── Compute SCE (intuitor) scores ─────────────────────────────────
    logger.info("Scoring responses with SCE (intuitor)...")
    model = get_model(args.model_path, device="cuda")
    flat_prompts = [p for p in prompts for _ in range(g)]
    sce_scores = score_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=flat_prompts,
        responses=flat_responses,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        batch_size=args.scoring_batch_size,
    )
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sce_t = torch.tensor(sce_scores, dtype=torch.float32).view(num_prompts, g)

    # ── Compute gated reward: R = R(MV) * sigmoid(R(intuitor)) ──────
    gated_t = mv_t * torch.sigmoid(sce_t)

    # ── Optional: ground-truth scoring ────────────────────────────────
    answer_col = get_answer_column(args.dataset_key)
    gold_answers = [p.get(answer_col, p.get("solution", "")) for p in problems]
    scorer = get_scorer(args.scorer, dataset_hint=args.dataset_key)

    gt_correct: List[List[bool]] = []
    for i in range(num_prompts):
        start, end = i * g, (i + 1) * g
        row = []
        for resp in flat_responses[start:end]:
            result = scorer.score(resp, gold_answers[i])
            row.append(bool(result.is_correct))
        gt_correct.append(row)

    # ── Print comparison table ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 100)
    logger.info(
        "%-6s  %-8s  %-12s  %-12s  %-12s  %-8s  %-10s",
        "Prompt", "Gen", "MV(0/1)", "SCE", "Gated", "GT", "MV_answer",
    )
    logger.info("-" * 100)

    details: List[Dict[str, Any]] = []
    for i in range(num_prompts):
        for j in range(g):
            logger.info(
                "%-6d  %-8d  %-12.4f  %-12.4f  %-12.4f  %-8s  %-10s",
                i, j,
                mv_t[i, j].item(),
                sce_t[i, j].item(),
                gated_t[i, j].item(),
                "Y" if gt_correct[i][j] else "N",
                str(mv_answers[i])[:10] if j == 0 else "",
            )
            details.append({
                "prompt_idx": i,
                "gen_idx": j,
                "mv_reward": mv_t[i, j].item(),
                "sce_score": sce_t[i, j].item(),
                "gated_reward": gated_t[i, j].item(),
                "gt_correct": gt_correct[i][j],
            })

    # ── Summary statistics ────────────────────────────────────────────
    mv_match_rate = (mv_t > 0.5).float().mean().item()
    sce_mean = sce_t.mean().item()
    gated_mean = gated_t.mean().item()
    gt_acc = sum(c for row in gt_correct for c in row) / max(num_prompts * g, 1)

    logger.info("")
    logger.info("=" * 100)
    logger.info("SUMMARY (%d prompts x %d generations)", num_prompts, g)
    logger.info("  MV match rate:    %.1f%%", mv_match_rate * 100)
    logger.info("  Mean SCE:         %.4f", sce_mean)
    logger.info("  Mean gated:       %.4f", gated_mean)
    logger.info("  GT accuracy:      %.1f%%", gt_acc * 100)
    logger.info("  Formula: R = R(MV) * sigmoid(R(intuitor))")
    logger.info("  When MV=0: R=0 (hard gate); When MV=1: R=sigmoid(SCE)")

    # ── Save results ──────────────────────────────────────────────────
    summary = {
        "model_path": args.model_path,
        "dataset_key": args.dataset_key,
        "scorer": args.scorer,
        "num_problems": num_prompts,
        "num_generations": g,
        "temperature": args.temperature,
        "seed": args.seed,
        "mv_match_rate": mv_match_rate,
        "sce_mean": sce_mean,
        "gated_mean": gated_mean,
        "gt_accuracy": gt_acc,
        "generation_time_s": round(gen_time, 2),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    details_path = os.path.join(args.output_dir, "details.json")
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
