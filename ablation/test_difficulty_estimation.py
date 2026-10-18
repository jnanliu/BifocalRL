#!/usr/bin/env python
# coding=utf-8
"""
Test difficulty estimation and bucket partitioning.

Runs a model through a small set of prompts, estimates per-prompt difficulty
using either majority-vote (MV) agreement or intuitor (SCE) confidence,
partitions into N buckets, and prints the distribution.

Usage (single GPU, MV method):
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_difficulty_estimation \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --dataset_key math_500 \
        --method mv \
        --num_problems 20 \
        --num_generations 8 \
        --n_buckets 3 \
        --output_dir outputs/ablation/difficulty_mv

    # Intuitor (SCE) method:
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_difficulty_estimation \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --dataset_key math_500 \
        --method intuitor \
        --num_problems 20 \
        --num_generations 8 \
        --n_buckets 3 \
        --output_dir outputs/ablation/difficulty_intuitor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_difficulty_estimation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test difficulty estimation and bucket partitioning."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_key", type=str, default="math_500")
    parser.add_argument("--method", type=str, default="mv", choices=["mv", "intuitor"])
    parser.add_argument("--n_buckets", type=int, default=3)
    parser.add_argument("--num_problems", type=int, default=20)
    parser.add_argument("--num_generations", type=int, default=8)
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
    parser.add_argument("--scoring_batch_size", type=int, default=8)

    parser.add_argument("--output_dir", type=str, default="outputs/ablation/difficulty")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from utils.difficulty import run_difficulty_estimation

    logger.info(
        "Running difficulty estimation: method=%s, n_buckets=%d, "
        "num_problems=%d, num_generations=%d",
        args.method, args.n_buckets, args.num_problems, args.num_generations,
    )

    report = run_difficulty_estimation(
        model_path=args.model_path,
        dataset_key=args.dataset_key,
        method=args.method,
        n_buckets=args.n_buckets,
        num_generations=args.num_generations,
        max_samples=args.num_problems if args.num_problems > 0 else None,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_length=args.max_completion_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        scoring_batch_size=args.scoring_batch_size,
        max_prompt_length=args.max_prompt_length,
    )

    # ── Print per-prompt results ──────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("%-8s  %-14s  %-10s  %-10s", "Prompt", "Difficulty", "Bucket", "Label")
    logger.info("-" * 80)
    for r in report.results:
        logger.info(
            "%-8d  %-14.4f  %-10d  %-10s",
            r.prompt_index, r.difficulty_score, r.bucket_id, r.bucket_label,
        )

    # ── Print bucket summary ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("BUCKET SUMMARY (method=%s, n_buckets=%d)", report.method, report.n_buckets)
    logger.info("-" * 80)
    for bid in sorted(report.bucket_assignments.keys()):
        indices = report.bucket_assignments[bid]
        scores = [report.results[i].difficulty_score for i in indices]
        label = report.bucket_labels.get(bid, "?")
        logger.info(
            "  Bucket %d (%s): %d prompts, difficulty [%.4f, %.4f], mean=%.4f",
            bid, label, len(indices),
            min(scores) if scores else 0,
            max(scores) if scores else 0,
            sum(scores) / max(len(scores), 1),
        )

    # ── Save results ──────────────────────────────────────────────────
    output: Dict[str, Any] = {
        "method": report.method,
        "n_buckets": report.n_buckets,
        "metadata": report.metadata,
        "results": [
            {
                "prompt_index": r.prompt_index,
                "difficulty_score": r.difficulty_score,
                "bucket_id": r.bucket_id,
                "bucket_label": r.bucket_label,
            }
            for r in report.results
        ],
        "bucket_summary": {
            str(bid): {
                "label": report.bucket_labels.get(bid, ""),
                "count": len(indices),
                "indices": indices,
            }
            for bid, indices in report.bucket_assignments.items()
        },
    }

    out_path = os.path.join(args.output_dir, "difficulty_report.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to: %s", out_path)


if __name__ == "__main__":
    main()
