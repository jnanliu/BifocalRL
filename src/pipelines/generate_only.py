# coding=utf-8
"""
Standalone Generation Script (no accelerate).

Generates completions for a single model and saves to cache.
For co-learning, run two instances in parallel (one per model on separate GPUs).
For single-model, run once with --cache_slot a.

This script does NOT use accelerate — vLLM runs as a standalone process,
avoiding v1 sub-process NCCL conflicts.

Usage (co-learning, parallel):
    python src/generate_only.py --model_path $A --tokenizer_path $TOK_A --cache_slot a ... &
    python src/generate_only.py --model_path $B --tokenizer_path $TOK_B --cache_slot b ... &
    wait

Usage (single-model):
    python src/generate_only.py --model_path $POLICY --cache_slot a ...
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import torch

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.cache import CoIntuitorCache
from utils.dataset import load_co_intuitor_dataset
from utils.generate import (
    generate_responses,
    generate_sequences,
    GenerationOutput,
    strip_prompt_echo_from_flattened_responses,
    strip_chat_template_artifacts_from_flattened_responses,
)
from utils.prompt_utils import render_prompts
from utils.model_utils import get_tokenizer
from utils.distributed_utils import (
    sleep_vllm_server_if_possible,
    stop_vllm_server_if_possible,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone generation (no accelerate). "
        "Generates completions for one model and saves to cache."
    )

    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Model path for generation")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer path (defaults to model_path)")
    parser.add_argument("--cache_slot", type=str, choices=["a", "b"], default="a",
                        help="Cache slot to save generations to (a or b)")

    # Dataset
    parser.add_argument("--dataset_key", type=str, required=True, help="Dataset key from recipes/datasets.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cache")
    parser.add_argument("--step", type=int, default=0, help="Step number for cache")
    parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic sampling")
    parser.add_argument("--problems_per_step", type=int, default=128, help="Number of problems per step")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples in dataset")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument(
        "--instruction_suffix",
        type=str,
        default=None,
        help="Instruction suffix override. None=dataset default, ''=no suffix.",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="false",
        choices=["true", "false", "none"],
        help="Chat template enable_thinking control.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open-R1-style shuffled sampling (per-epoch randperm, drop_last).",
    )
    parser.add_argument(
        "--reset_sampling_state",
        action="store_true",
        default=False,
        help="Restart sampling from epoch 0.",
    )

    # Generation
    parser.add_argument("--num_generations", type=int, default=7, help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="vLLM repetition_penalty")
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Max completion length")

    # vLLM settings
    parser.add_argument(
        "--use_offline_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM offline API. If false, falls back to HTTP server API.",
    )
    parser.add_argument("--vllm_gpu", type=int, default=None, help="CUDA device for offline vLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16", help="Model dtype for vLLM")
    parser.add_argument("--vllm_max_model_len", type=int, default=None, help="Max model context length")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None, help="Max concurrent sequences for offline vLLM")
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=None, help="Max total batched tokens for offline vLLM")
    parser.add_argument(
        "--enable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM chunked prefill for offline generation",
    )
    # HTTP server settings (used when --no-use_offline_vllm)
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port (HTTP mode only)")
    parser.add_argument(
        "--auto_start_server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-start vLLM server (HTTP mode only)",
    )
    parser.add_argument(
        "--auto_stop_server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-stop vLLM server (HTTP mode only)",
    )

    args = parser.parse_args()
    logger.setLevel(logging.INFO)

    os.makedirs(args.output_dir, exist_ok=True)
    cache = CoIntuitorCache(os.path.join(args.output_dir, "cache.json"))

    slot = args.cache_slot
    tokenizer_path = args.tokenizer_path or args.model_path

    logger.info(f"Generate-only: model={args.model_path}, slot={slot}")

    # ------------------------------------------------------------------
    # Dataset loading and sampling (deterministic — identical across slots)
    # ------------------------------------------------------------------
    train_problems = load_co_intuitor_dataset(
        dataset_key=args.dataset_key,
        max_samples=args.max_samples,
        system_prompt=args.system_prompt,
        instruction_suffix_override=args.instruction_suffix,
    )

    num_problems = len(train_problems)
    if num_problems <= 0:
        raise ValueError("Dataset is empty (num_problems=0).")
    if int(args.problems_per_step) <= 0:
        raise ValueError("--problems_per_step must be > 0.")

    sampling_strategy = "sequential"
    epoch = None
    pos = None
    epoch_len = None
    perm_seed = None

    if bool(getattr(args, "shuffle_dataset", False)):
        sampling_strategy = "open_r1"
        batch = int(args.problems_per_step)
        epoch_len = (num_problems // batch) * batch
        if epoch_len <= 0:
            raise ValueError(
                f"Dataset too small for open-r1 sampling: num_problems={num_problems} < problems_per_step={batch}."
            )
        steps_per_epoch = epoch_len // batch
        step_i = int(args.step) - 1  # 1-indexed: pipeline step 1 → first batch
        if step_i < 0:
            raise ValueError(f"step must be >= 1 for shuffle sampling, got {args.step}")
        epoch = step_i // steps_per_epoch
        pos = (step_i % steps_per_epoch) * batch

        perm_seed = int(args.seed) + int(epoch)
        g = torch.Generator()
        g.manual_seed(int(perm_seed))
        perm = torch.randperm(num_problems, generator=g).tolist()
        sampled_indices = perm[int(pos): int(pos) + batch]
        if len(sampled_indices) != batch:
            raise RuntimeError(
                f"open-r1 sampler produced wrong batch length: got {len(sampled_indices)}, expected {batch}."
            )
        logger.info(
            f"Step {args.step}: Open-R1 sampling epoch={epoch} pos={pos}/{epoch_len} "
            f"(steps_per_epoch={steps_per_epoch}, perm_seed={perm_seed})"
        )
    else:
        start_index = (int(args.step) - 1) * int(args.problems_per_step)  # 1-indexed
        if start_index >= num_problems:
            start_index = start_index % num_problems
        end_index = min(start_index + int(args.problems_per_step), num_problems)
        sampled_indices = list(range(start_index, end_index))
        if len(sampled_indices) < int(args.problems_per_step):
            remaining = int(args.problems_per_step) - len(sampled_indices)
            sampled_indices.extend(list(range(remaining)))
        pos = int(start_index)

    sampled_raw = [train_problems[i] for i in sampled_indices]

    # Save checkpoint info (idempotent — safe if both slots save the same data).
    cache._load_cache()
    cache.save_checkpoint_info(
        int(args.step),
        int(pos) if pos is not None else 0,
        sampled_indices,
        sampling_strategy=str(sampling_strategy),
        epoch=(int(epoch) if epoch is not None else None),
        pos=(int(pos) if pos is not None else None),
        epoch_len=(int(epoch_len) if epoch_len is not None else None),
        num_problems=int(num_problems),
        problems_per_step=int(args.problems_per_step),
        seed=int(args.seed),
        shuffle_dataset=bool(getattr(args, "shuffle_dataset", False)),
        perm_seed=(int(perm_seed) if perm_seed is not None else None),
    )

    # ------------------------------------------------------------------
    # Render prompts with this model's tokenizer
    # ------------------------------------------------------------------
    tokenizer = get_tokenizer(tokenizer_path)
    _et_map = {"true": True, "false": False, "none": None}
    prompts_text = render_prompts(
        sampled_raw,
        tokenizer,
        apply_chat_template_kwargs={"enable_thinking": _et_map[args.enable_thinking]},
    )

    # ------------------------------------------------------------------
    # Check cache
    # ------------------------------------------------------------------
    use_offline = bool(getattr(args, "use_offline_vllm", True))
    expected = len(prompts_text) * int(args.num_generations)
    finish_reasons: Optional[List[Optional[str]]] = None
    flat_completions: List[str] = []

    cache_check = cache.check_generations_a if slot == "a" else cache.check_generations_b
    cache_load = cache.load_generations_a if slot == "a" else cache.load_generations_b
    cache_save = cache.save_generations_a if slot == "a" else cache.save_generations_b

    if cache_check(args.step, args.num_generations):
        ok, cached_resps, cached_np, cached_prompts, cached_finish = cache_load(
            args.step, args.num_generations
        )
        if (
            ok
            and cached_resps is not None
            and cached_np == len(prompts_text)
            and len(cached_resps) == expected
        ):
            flat_completions = cached_resps
            if cached_prompts is not None:
                prompts_text = cached_prompts
            finish_reasons = cached_finish
            if finish_reasons is not None and len(finish_reasons) != len(flat_completions):
                finish_reasons = [None] * len(flat_completions)
            logger.info(f"Cache hit: generations_{slot} step={args.step} ({len(flat_completions)} completions)")
            if not use_offline and not bool(getattr(args, "auto_start_server", False)):
                sleep_vllm_server_if_possible(int(args.vllm_port), is_main=True)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    if not flat_completions:
        logger.info(
            f"Generating {len(prompts_text)} prompts x {args.num_generations} completions "
            f"(offline={use_offline})..."
        )

        if use_offline:
            gen_out = generate_sequences(
                model_path=args.model_path,
                prompts=prompts_text,
                num_generations=args.num_generations,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_completion_length,
                repetition_penalty=args.repetition_penalty,
                gpu_memory_utilization=float(getattr(args, "gpu_memory_utilization", 0.9)),
                tensor_parallel_size=int(getattr(args, "tensor_parallel_size", 1)),
                dtype=str(getattr(args, "vllm_dtype", "bfloat16")),
                cuda_device=getattr(args, "vllm_gpu", None),
                seed=int(args.seed) if hasattr(args, "seed") else None,
                max_model_len=getattr(args, "vllm_max_model_len", None),
                max_num_seqs=getattr(args, "vllm_max_num_seqs", None),
                max_num_batched_tokens=getattr(args, "vllm_max_num_batched_tokens", None),
                enable_chunked_prefill=getattr(args, "enable_chunked_prefill", None),
            )
            flat_completions = gen_out.response_texts
            finish_reasons = gen_out.finish_reasons
        else:
            gen_out = generate_responses(
                model_path=args.model_path,
                prompts=prompts_text,
                num_generations=args.num_generations,
                batch_size=128,
                port=args.vllm_port,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_completion_length,
                auto_start_server=args.auto_start_server,
                auto_stop_server=args.auto_stop_server,
                return_finish_reasons=True,
            )
            if isinstance(gen_out, tuple) and len(gen_out) == 2:
                flat_completions, finish_reasons = gen_out
            else:
                flat_completions = gen_out
                finish_reasons = None

        if len(flat_completions) != expected:
            raise RuntimeError(f"Expected {expected} completions, got {len(flat_completions)}")
        if finish_reasons is not None and len(finish_reasons) != len(flat_completions):
            finish_reasons = [None] * len(flat_completions)

        # Sanitize completions
        flat_completions, _c1 = strip_prompt_echo_from_flattened_responses(
            prompts_text, flat_completions, num_generations=args.num_generations
        )
        flat_completions, _c2 = strip_chat_template_artifacts_from_flattened_responses(
            prompts_text, flat_completions, num_generations=args.num_generations
        )

        # Reload cache from disk before saving to avoid clobbering the other slot's data
        # when two generate_only processes run in parallel.
        cache._load_cache()
        cache_save(
            args.step, len(prompts_text), flat_completions,
            prompts=prompts_text, finish_reasons=finish_reasons,
        )
        logger.info(f"Saved generations to cache (generations_{slot}).")

    # For HTTP mode: stop server to free GPU memory.
    if not use_offline and bool(getattr(args, "auto_stop_server", False)):
        stopped = stop_vllm_server_if_possible(int(args.vllm_port), is_main=True)
        if not stopped:
            logger.warning(f"vLLM server on port {args.vllm_port} still running after auto-stop.")

    logger.info(f"Generate-only done: slot={slot}, {len(flat_completions)} completions.")


if __name__ == "__main__":
    main()
