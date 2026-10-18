#!/usr/bin/env python
# coding=utf-8

import argparse
import os
from typing import List, Optional

from eval.hacrl.runner import EvalConfig, run_all
from eval.scorers import ScorerName


def _infer_checkpoint_tag(model_path: str, fallback: str = "checkpoint-unknown") -> str:
    base = os.path.basename(os.path.normpath(model_path))
    if base.startswith("checkpoint-"):
        return base
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint on multiple datasets.")
    parser.add_argument("--model_path", type=str, required=True, help="HF checkpoint path (e.g., .../checkpoint-3)")
    parser.add_argument("--model_tag", type=str, required=True, choices=["a", "b"], help="Model tag: a or b")
    parser.add_argument("--checkpoint_tag", type=str, default=None, help="Checkpoint tag for filenames (e.g., checkpoint-3)")
    parser.add_argument(
        "--datasets",
        type=str,
        default="math_500,gsm8k",
        help="Comma-separated datasets: math_500,gsm8k,aime24,aime25,amc,cruxeval",
    )
    parser.add_argument("--num_generations", type=int, default=1, help="Generations per problem")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap per dataset")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--output_root", type=str, default="outputs/eval", help="Root folder for evaluation outputs")

    parser.add_argument(
        "--scorer",
        type=str,
        default="default",
        choices=ScorerName.choices(),
        help=(
            "Scoring backend for answer verification. "
            "Choices: %(choices)s. "
            "default=co-learning math_verify, "
            "hacrl_math_dapo=Minerva/AIME, "
            "hacrl_math_boxed=boxed+strip, "
            "hacrl_prime_math=sympy, "
            "hacrl_gsm8k=strict ####, "
            "hacrl_qa_em=QA exact-match."
        ),
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation (vLLM).")

    # Offline vLLM (HACRL-style, default)
    parser.add_argument("--use_offline_vllm", action=argparse.BooleanOptionalAction, default=True,
                        help="Use offline vLLM Python API (HACRL-style). Disable for legacy HTTP server.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--cuda_device", type=int, default=None, help="CUDA device for offline vLLM")

    # Legacy HTTP vLLM (only when --no-use_offline_vllm)
    parser.add_argument("--vllm_port", type=int, default=8010)
    parser.add_argument("--vllm_host", type=str, default="localhost")
    parser.add_argument("--vllm_batch_size", type=int, default=16)
    parser.add_argument("--auto_start_server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto_stop_server", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    datasets: List[str] = [d.strip() for d in args.datasets.split(",") if d.strip()]

    checkpoint_tag = args.checkpoint_tag or _infer_checkpoint_tag(args.model_path)
    system_prompt: Optional[str] = args.system_prompt
    if system_prompt in ("", "null", "None"):
        system_prompt = None

    cfg = EvalConfig(
        model_path=args.model_path,
        model_tag=args.model_tag,
        checkpoint_tag=checkpoint_tag,
        system_prompt=system_prompt,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_samples=args.max_samples,
        output_root=args.output_root,
        seed=args.seed,
        scorer=args.scorer,
        use_offline=args.use_offline_vllm,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        vllm_dtype=args.vllm_dtype,
        vllm_max_model_len=args.vllm_max_model_len,
        cuda_device=args.cuda_device,
        vllm_port=args.vllm_port,
        vllm_host=args.vllm_host,
        vllm_batch_size=args.vllm_batch_size,
        auto_start_server=args.auto_start_server,
        auto_stop_server=args.auto_stop_server,
    )

    mode = "offline (HACRL-style)" if cfg.use_offline else "HTTP server (legacy)"
    print(f"[eval] vLLM mode: {mode} | scorer: {cfg.scorer}")
    out = run_all(cfg, datasets)
    for name, path in out.items():
        print(f"[eval] {name}: {path}")


if __name__ == "__main__":
    main()

