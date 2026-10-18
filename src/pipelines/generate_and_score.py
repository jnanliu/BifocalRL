# coding=utf-8
"""
Generate and Score Pipeline for Co-Intuitor.

This script handles:
1. Generation: Generate responses using vLLM (no accelerator needed)
2. Scoring: Score responses using self-certainty (no accelerator needed)

Usage:
    python src/generate_and_score.py \
        --model_a_path /path/to/model_a \
        --model_b_path /path/to/model_b \
        --dataset_key math_500 \
        --output_dir ./cache \
        --step 0
"""

import argparse
import logging
import os
import random
import sys
import json
from typing import List, Optional, Any, Dict, Tuple

import torch
from tqdm import tqdm

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.cache import CoIntuitorCache
from utils.dataset import load_co_intuitor_dataset, get_answer_column
from utils.generate import (
    generate_responses,
    generate_sequences,
    GenerationOutput,
    strip_prompt_echo_from_flattened_responses,
    strip_chat_template_artifacts_from_flattened_responses,
)
from utils.reward import compute_rewards, SCE_METHODS
from utils.reward import get_per_token_self_certainty, get_advantage_from_sce
from utils.logps import get_per_token_logps
from utils.tokenization import build_left_padded_prompt_completion_batch
from utils.env_utils import co_env_flag, co_int_env
from utils.prompt_utils import render_prompt
from utils.model_utils import get_tokenizer, get_model
from utils.distributed_utils import (
    get_accelerator,
    is_main_process,
    wait_for_everyone,
    sleep_vllm_server_if_possible,
    stop_vllm_server_if_possible,
    shard_range,
)
from utils.debug_utils import co_dump_alignment_samples
from utils.invariants import (
    tc_check_generations_block,
    tc_check_logps_matrix,
    tc_check_score_vector,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def generate_responses_for_model(
    model_path: str,
    unique_prompts: List[str],
    num_generations: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    port: int = 8000,
    auto_start_server: bool = True,
    auto_stop_server: bool = True,
    *,
    cuda_device: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    enable_chunked_prefill: Optional[bool] = None,
    seed: Optional[int] = None,
    use_offline: bool = True,
) -> GenerationOutput:
    """Generate responses using vLLM (HACRL-style offline API by default).

    When ``use_offline=True`` (default), uses the vLLM offline Python API
    which returns exact token IDs and per-token log-probabilities computed
    during generation — eliminating text round-trip and the need for a
    separate ``compute_logps`` forward pass.

    When ``use_offline=False``, falls back to the legacy HTTP API (kept for
    backward compatibility with evaluation scripts and pre-started servers).

    Returns:
        :class:`GenerationOutput` containing ``response_texts``,
        ``finish_reasons``, ``response_ids``, and ``old_log_probs``.
    """
    logger.info(
        "Generating %d unique prompts × %d completions (offline=%s) ...",
        len(unique_prompts), num_generations, use_offline,
    )

    if use_offline:
        return generate_sequences(
            model_path=model_path,
            prompts=unique_prompts,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            cuda_device=cuda_device,
            seed=seed,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
        )

    # Legacy HTTP API path
    all_responses, finish_reasons = generate_responses(
        model_path=model_path,
        prompts=unique_prompts,
        num_generations=num_generations,
        batch_size=128,
        port=port,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        auto_start_server=auto_start_server,
        auto_stop_server=auto_stop_server,
        return_finish_reasons=True,
    )
    if len(finish_reasons) != len(all_responses):
        finish_reasons = [None] * len(all_responses)

    return GenerationOutput(
        response_texts=list(all_responses),
        finish_reasons=[fr or "stop" for fr in finish_reasons],
        prompt_ids=[],
        response_ids=[],
        old_log_probs=[],
    )


from utils.scoring_helpers import score_responses_for_model  # noqa: E302


def compute_logps_for_model(
    model_path: str,
    tokenizer_path: str,
    unique_prompt_examples: List[dict],
    all_text_responses_a: List[str],
    all_text_responses_b: List[str],
    all_finish_reasons_a: Optional[List[Optional[str]]] = None,
    all_finish_reasons_b: Optional[List[Optional[str]]] = None,
    *,
    debug_output_dir: Optional[str] = None,
    debug_step: Optional[int] = None,
    debug_model_tag: Optional[str] = None,
    num_generations_per_model: int = 4,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    mask_truncated_completions: bool = False,
    batch_size: int = 8,
    temperature: float = 1.0,
    device: str = "cuda",
    accelerator=None,
    score_only: Optional[str] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute per-token log probabilities for responses using a model.
    
    Args:
        model_path: Path to model
        tokenizer_path: Path to tokenizer
        unique_prompt_examples: List of unique prompt examples (raw dataset dicts; length = num_unique)
        all_text_responses_a: All responses from Model A (flattened: num_unique * num_generations_per_model)
        all_text_responses_b: All responses from Model B (flattened: num_unique * num_generations_per_model)
        num_generations_per_model: Number of generations per model
        max_prompt_length: Maximum prompt length
        batch_size: Batch size for logps computation
        temperature: Temperature for logits scaling
        device: Device to use
        score_only: "a" to compute logps only for Model A's responses, "b" for Model B's only,
                    None to compute for all (both A and B).
    
    Returns:
        Tuple of (logps, completion_lengths):
          - logps: Per-token log probabilities tensor
          - completion_lengths: Per-sample valid token count under this model's tokenizer
    """
    logger.info(f"Computing log probabilities using model: {model_path}")
    
    tokenizer = get_tokenizer(tokenizer_path)
    if accelerator is not None and torch.cuda.is_available():
        torch_device = getattr(accelerator, "device", torch.device(device))
        device_str = str(torch_device)
    else:
        device_str = device
        torch_device = torch.device(device_str)

    model = get_model(model_path, device=device_str)
    model.eval()
    
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if not has_template:
        logger.info("Logps tokenizer has no chat_template — using raw prompt text directly.")
    from utils.prompt_utils import render_prompts
    unique_prompts = render_prompts(unique_prompt_examples, tokenizer, has_template=has_template)

    num_unique_prompts = len(unique_prompts)
    
    # Verify response counts
    expected_responses_a = num_unique_prompts * num_generations_per_model
    expected_responses_b = num_unique_prompts * num_generations_per_model
    if len(all_text_responses_a) != expected_responses_a:
        raise ValueError(f"Expected {expected_responses_a} responses from Model A, got {len(all_text_responses_a)}")
    if len(all_text_responses_b) != expected_responses_b:
        raise ValueError(f"Expected {expected_responses_b} responses from Model B, got {len(all_text_responses_b)}")
    
    # Prepare prompts and responses (batch preparation, no loop)
    prompts_for_logps = []
    responses_for_logps = []
    trunc_for_logps: Optional[List[Optional[bool]]] = (
        [] if (all_finish_reasons_a is not None or all_finish_reasons_b is not None) else None
    )
    
    for i in range(num_unique_prompts):
        prompt = unique_prompts[i]
        
        # Get responses for this unique prompt
        start_a = i * num_generations_per_model
        end_a = (i + 1) * num_generations_per_model
        start_b = i * num_generations_per_model
        end_b = (i + 1) * num_generations_per_model
        
        responses_a = all_text_responses_a[start_a:end_a]
        responses_b = all_text_responses_b[start_b:end_b]

        if score_only == "a":
            combined_responses = list(responses_a)
        elif score_only == "b":
            combined_responses = list(responses_b)
        else:
            combined_responses = list(responses_a) + list(responses_b)

        if trunc_for_logps is not None:
            fr_a = (
                all_finish_reasons_a[start_a:end_a] if all_finish_reasons_a is not None else [None] * len(responses_a)
            )
            fr_b = (
                all_finish_reasons_b[start_b:end_b] if all_finish_reasons_b is not None else [None] * len(responses_b)
            )
            if score_only == "a":
                combined_fr = list(fr_a)
            elif score_only == "b":
                combined_fr = list(fr_b)
            else:
                combined_fr = list(fr_a) + list(fr_b)
            trunc_for_logps.extend([None if x is None else (str(x) == "length") for x in combined_fr])
        
        prompts_for_logps.extend([prompt] * len(combined_responses))
        responses_for_logps.extend(combined_responses)

    # Build a batch by left-padding the *full* prompt+completion sequence once.
    #
    # IMPORTANT: build on CPU first, then move micro-batches to GPU in the logps loop.
    build_device = torch.device("cpu")
    prompt_completion_ids, attention_mask, completion_mask_window, logits_to_keep = (
        build_left_padded_prompt_completion_batch(
            tokenizer,
            prompts_for_logps,
            responses_for_logps,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            mask_truncated_completions=mask_truncated_completions,
            completion_is_truncated=trunc_for_logps,
            device=build_device,
        )
    )

    # Match open-r1 semantics: completion pad positions must have attention_mask=0.
    # build_left_padded_prompt_completion_batch sets attention_mask=1 on ALL non-left-pad
    # positions (including completion pads) for Flash-Attn safety.  This creates a
    # divergence: cached logps on pads ≠ training logps on pads (which use mask=0),
    # causing exp(delta_kl) overflow in the KL term.  Applying completion_mask_window
    # here makes the cached logps consistent with training / open-r1.
    attention_mask[:, -logits_to_keep:] = completion_mask_window

    # Per-sample valid token counts under this model's tokenizer (for cross-tokenizer mask).
    completion_lengths_all: List[int] = completion_mask_window.sum(dim=1).int().tolist()

    # Debug: dump a few alignment samples (open-r1 equivalence check).
    if debug_output_dir is not None and debug_step is not None and debug_model_tag is not None:
        co_dump_alignment_samples(
            stage="logps",
            output_dir=str(debug_output_dir),
            step=int(debug_step),
            model_tag=str(debug_model_tag),
            tokenizer=tokenizer,
            prompts=prompts_for_logps,
            responses=responses_for_logps,
            finish_is_truncated=trunc_for_logps,
            prompt_completion_ids=prompt_completion_ids,
            completion_mask_window=completion_mask_window,
            logits_to_keep=int(logits_to_keep),
            max_items=co_int_env("CO_DEBUG_DATA_N", 3),
            is_main=is_main_process(accelerator),
        )
    if mask_truncated_completions:
        try:
            lengths = completion_mask_window.sum(dim=1)
            num_zero = int((lengths == 0).sum().item())
            logger.info(f"Logps mask_truncated_completions: masked {num_zero}/{int(lengths.numel())} completions (len==0)")
        except Exception:
            pass

    # Compute per-token logps over the suffix window of length `logits_to_keep`.
    all_logps = []
    
    n_total = int(prompt_completion_ids.size(0))
    start, end, per_rank = shard_range(n_total, accelerator)
    is_main = is_main_process(accelerator)

    prompt_completion_ids_local = prompt_completion_ids[start:end]
    attention_mask_local = attention_mask[start:end]

    num_logps_batches = (prompt_completion_ids_local.size(0) + batch_size - 1) // batch_size
    with torch.no_grad():
        iterator = range(0, prompt_completion_ids_local.size(0), batch_size)
        iterator = tqdm(
            iterator,
            desc="Computing log probabilities",
            unit="batch",
            total=num_logps_batches,
            leave=False,
            disable=not is_main,
        )
        for i in iterator:
            end_idx = min(i + batch_size, prompt_completion_ids_local.size(0))
            batch_input_ids = prompt_completion_ids_local[i:end_idx].to(torch_device, non_blocking=True)
            batch_attention_mask = attention_mask_local[i:end_idx].to(torch_device, non_blocking=True)
            
            per_token_logps = get_per_token_logps(
                model, batch_input_ids, batch_attention_mask, logits_to_keep,
                temperature=temperature,
                batch_size=batch_size,
            )
            all_logps.append(per_token_logps.detach().cpu())
    
    if all_logps:
        logps_local = torch.cat(all_logps, dim=0)  # (local_n, logits_to_keep)
    else:
        logps_local = torch.empty((0, logits_to_keep), dtype=torch.float32)

    # Gather with padding on dim0.
    if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
        padded = torch.zeros((per_rank, logits_to_keep), dtype=logps_local.dtype)
        if logps_local.numel() > 0:
            padded[: logps_local.size(0)] = logps_local
        padded = padded.to(torch_device, non_blocking=True)
        gathered = accelerator.gather(padded)  # (world * per_rank, logits_to_keep)
        logps_all = gathered[:n_total].detach().cpu()
    else:
        logps_all = logps_local
    
    logger.info(f"Computed log probabilities: shape {logps_all.shape}")
    
    return logps_all, completion_lengths_all


def main():
    parser = argparse.ArgumentParser(description="Generate and score responses for Co-Intuitor")
    
    # Model paths
    parser.add_argument("--model_a_path", type=str, required=True, help="Path to Model A")
    parser.add_argument("--model_b_path", type=str, required=True, help="Path to Model B")
    parser.add_argument(
        "--tokenizer_a_path",
        type=str,
        default=None,
        help="Tokenizer source-of-truth for Model A (typically the initial/base model dir).",
    )
    parser.add_argument(
        "--tokenizer_b_path",
        type=str,
        default=None,
        help="Tokenizer source-of-truth for Model B (typically the initial/base model dir).",
    )
    
    # Dataset
    parser.add_argument("--dataset_key", type=str, required=True, help="Dataset key from recipes/datasets.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt")
    
    # Generation parameters
    parser.add_argument("--num_generations_per_model", type=int, default=4, help="Number of generations per model")
    # Align defaults with open-r1 (TRL GRPOConfig defaults).
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="vLLM repetition_penalty (open-r1 style)")
    # Project default: cap completions to 2048 tokens to control memory across generate/score/logps/train.
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Max completion length")
    
    # Scoring parameters
    parser.add_argument("--max_prompt_length", type=int, default=None, help="Max prompt length")
    parser.add_argument("--scoring_batch_size", type=int, default=8, help="Batch size for scoring")
    parser.add_argument(
        "--mask_truncated_completions",
        action="store_true",
        default=False,
        help="If set, completions that do not contain an EOS token will be fully masked out in scoring/logps.",
    )

    # Debugging: dump why A scoring has zero advantages.
    parser.add_argument(
        "--debug_scores_a_zeros",
        action="store_true",
        default=False,
        help="If set, dump details for samples with advantage==0 when scoring with Model A.",
    )
    parser.add_argument(
        "--debug_scores_a_zeros_max",
        type=int,
        default=30,
        help="Max number of zero-advantage samples to dump for A scoring.",
    )
    
    # Logps computation parameters
    # Reward method (determines scoring: SCE vs ground-truth)
    parser.add_argument(
        "--reward_method",
        type=str,
        default="sum_scores",
        help="Reward method: truth, self_mv, co_mv, self_scores, co_wmv, sum_scores, mixed_reward, mv_gated_intuitor, co_mv_gated_intuitor.",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="hacrl_prime_math",
        help="HACRL scorer backend for ground-truth reward (e.g. hacrl_prime_math, default, hacrl_gsm8k).",
    )

    parser.add_argument("--compute_logps", action="store_true", default=True, help="Whether to compute old_logps and ref_logps")
    parser.add_argument("--logps_batch_size", type=int, default=8, help="Batch size for logps computation")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient (if > 0, will compute ref_logps)")
    parser.add_argument("--ref_model_a_path", type=str, default=None, help="Path to reference model for Model A (optional)")
    parser.add_argument("--ref_model_b_path", type=str, default=None, help="Path to reference model for Model B (optional)")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of iterations (if > 1, will compute old_logps)")
    parser.add_argument("--force_compute_old_logps", action="store_true", default=False,
                        help="Always compute old_logps regardless of num_iterations (required for GSPO peer-phase IS)")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cache")
    parser.add_argument("--step", type=int, default=0, help="Step number for cache")
    
    # Sampling parameters
    parser.add_argument("--problems_per_step", type=int, default=128, help="Number of problems to sample per step")
    parser.add_argument("--seed", type=int, default=7, help="Seed for sampling and deterministic tie-breaking.")
    parser.add_argument(
        "--shuffle_dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, sample problems in open-r1-intuitor style: per-epoch randperm (no replacement within an epoch), "
            "drop_last to full batches, and use (seed + epoch) for reproducibility. "
            "If false (default), use the legacy sequential slicing sampler."
        ),
    )
    parser.add_argument(
        "--reset_sampling_state",
        action="store_true",
        default=False,
        help="If set, ignore any cached sampling metadata and restart sampling from epoch 0.",
    )
    
    # vLLM settings
    parser.add_argument(
        "--use_offline_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM offline API (HACRL-style, returns token_ids + logprobs). "
             "If false, falls back to HTTP server API.",
    )
    parser.add_argument("--vllm_gpu_a", type=int, default=None, help="CUDA device for Model A offline generation")
    parser.add_argument("--vllm_gpu_b", type=int, default=None, help="CUDA device for Model B offline generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16", help="Model dtype for vLLM")
    parser.add_argument("--vllm_max_model_len", type=int, default=None, help="Max model context length for vLLM")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None, help="Max concurrent sequences for offline vLLM")
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=None, help="Max total batched tokens for offline vLLM")
    parser.add_argument(
        "--enable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM chunked prefill for offline generation",
    )
    # Legacy HTTP server settings (used when --no-use_offline_vllm)
    parser.add_argument("--vllm_port_a", type=int, default=8000, help="vLLM server port for Model A")
    parser.add_argument("--vllm_port_b", type=int, default=8001, help="vLLM server port for Model B")
    parser.add_argument(
        "--auto_start_server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-start vLLM servers (legacy HTTP mode only)",
    )
    parser.add_argument(
        "--auto_stop_server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-stop vLLM servers (legacy HTTP mode only)",
    )
    
    args = parser.parse_args()

    accelerator = get_accelerator()
    is_main = is_main_process(accelerator)
    
    # Setup logging
    log_level = logging.INFO
    logger.setLevel(log_level)
    if not is_main:
        # Keep non-main ranks quiet to avoid log spam under accelerate.
        logger.setLevel(logging.WARNING)
    
    logger.info("=" * 80)
    logger.info("Co-Intuitor Generate and Score Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model A: {args.model_a_path}")
    logger.info(f"Model B: {args.model_b_path}")
    logger.info(f"Dataset: {args.dataset_key}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Step: {args.step}")
    if bool(getattr(args, "mask_truncated_completions", False)) and is_main:
        logger.warning(
            "--mask_truncated_completions is deprecated/ignored in this repo; truncated completions are kept."
        )
    logger.info(f"Problems per step: {args.problems_per_step}")
    logger.info(f"Generations per model: {args.num_generations_per_model}")
    logger.info(f"Expected total responses: {args.problems_per_step * args.num_generations_per_model * 2} (A: {args.problems_per_step * args.num_generations_per_model}, B: {args.problems_per_step * args.num_generations_per_model})")
    
    # Initialize cache early
    os.makedirs(args.output_dir, exist_ok=True)
    cache_file = os.path.join(args.output_dir, "cache.json")
    cache = CoIntuitorCache(cache_file)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_problems = load_co_intuitor_dataset(
        dataset_key=args.dataset_key,
        max_samples=args.max_samples,
        system_prompt=args.system_prompt,
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
        # Open-R1-style sampling:
        # - Each epoch: randperm(dataset), then take only full batches (drop_last)
        # - Step picks the next full batch of size problems_per_step
        sampling_strategy = "open_r1"
        batch = int(args.problems_per_step)
        epoch_len = (num_problems // batch) * batch  # drop_last
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
        sampled_indices = perm[int(pos) : int(pos) + batch]
        if len(sampled_indices) != batch:
            # Should not happen given our epoch_len math, but keep it defensive.
            raise RuntimeError(
                f"open-r1 sampler produced wrong batch length: got {len(sampled_indices)}, expected {batch}."
            )
        logger.info(
            f"Step {args.step}: Open-R1 sampling epoch={epoch} pos={pos}/{epoch_len} "
            f"(steps_per_epoch={steps_per_epoch}, perm_seed={perm_seed})"
        )
    else:
        # Legacy sequential sampling: calculate start_index based on step.
        start_index = (int(args.step) - 1) * int(args.problems_per_step)  # 1-indexed
        if start_index >= num_problems:
            logger.warning(f"Step {args.step} exceeds total problems ({num_problems}), wrapping around")
            start_index = start_index % num_problems
        end_index = min(start_index + int(args.problems_per_step), num_problems)
        sampled_indices = list(range(start_index, end_index))
        if len(sampled_indices) < int(args.problems_per_step):
            remaining = int(args.problems_per_step) - len(sampled_indices)
            sampled_indices.extend(list(range(remaining)))
        logger.info(
            f"Step {args.step}: Sequentially sampled {len(sampled_indices)} problems "
            f"(indices {sampled_indices[0]}-{sampled_indices[-1]})"
        )
        # For sequential mode, keep backward compatible fields.
        epoch = None
        pos = int(start_index)
        epoch_len = None
        perm_seed = None
    
    # Get sampled unique prompts (before applying chat template)
    sampled_unique_prompts_raw = [train_problems[i] for i in sampled_indices]
    
    # Save checkpoint info (which problems were sampled) once.
    if is_main:
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
    
    # Apply chat templates for each model separately.
    #
    # IMPORTANT:
    # Model A and Model B may have different `tokenizer.chat_template` formats (e.g. Qwen vs Phi).
    # If we force both to use tokenizer_a's template, Model B can see malformed prompts and produce
    # garbled outputs / wrong language. We therefore keep two prompt lists:
    # - prompts_a: formatted with tokenizer_a (used for A generation, A scoring/logps, and A training)
    # - prompts_b: formatted with tokenizer_b (used for B generation, B scoring/logps, and B training)
    tokenizer_a = get_tokenizer(args.tokenizer_a_path or args.model_a_path)
    tokenizer_b = get_tokenizer(args.tokenizer_b_path or args.model_b_path)

    has_chat_template_a = getattr(tokenizer_a, "chat_template", None) is not None
    has_chat_template_b = getattr(tokenizer_b, "chat_template", None) is not None
    if not has_chat_template_a:
        logger.info("Model A tokenizer has no chat_template — using raw prompt text directly.")
    if not has_chat_template_b:
        logger.info("Model B tokenizer has no chat_template — using raw prompt text directly.")

    prompts_a = []
    prompts_b = []
    for problem in sampled_unique_prompts_raw:
        prompts_a.append(render_prompt(problem, tokenizer_a, has_chat_template_a))
        prompts_b.append(render_prompt(problem, tokenizer_b, has_chat_template_b))
    
    logger.info(f"Applied chat templates: prompts_a={len(prompts_a)}, prompts_b={len(prompts_b)}")

    # vLLM finish reasons (used to detect true truncation under --mask_truncated_completions).
    all_finish_reasons_a: Optional[List[Optional[str]]] = None
    all_finish_reasons_b: Optional[List[Optional[str]]] = None

    # Keep GenerationOutput references for old_logps extraction (offline mode).
    gen_output_a: Optional[GenerationOutput] = None
    gen_output_b: Optional[GenerationOutput] = None

    use_offline = bool(getattr(args, "use_offline_vllm", True))

    def _gen_kwargs(model_tag: str) -> dict:
        """Build kwargs for generate_responses_for_model based on mode."""
        gpu_attr = f"vllm_gpu_{model_tag}"
        port_attr = f"vllm_port_{model_tag}"
        return dict(
            use_offline=use_offline,
            cuda_device=getattr(args, gpu_attr, None),
            gpu_memory_utilization=float(getattr(args, "gpu_memory_utilization", 0.9)),
            tensor_parallel_size=int(getattr(args, "tensor_parallel_size", 1)),
            dtype=str(getattr(args, "vllm_dtype", "bfloat16")),
            max_model_len=getattr(args, "vllm_max_model_len", None),
            max_num_seqs=getattr(args, "vllm_max_num_seqs", None),
            max_num_batched_tokens=getattr(args, "vllm_max_num_batched_tokens", None),
            enable_chunked_prefill=getattr(args, "enable_chunked_prefill", None),
            seed=int(args.seed) if hasattr(args, "seed") else None,
            port=getattr(args, port_attr, 8000),
            auto_start_server=bool(getattr(args, "auto_start_server", False)),
            auto_stop_server=bool(getattr(args, "auto_stop_server", False)),
        )

    # Step 1: Generate responses for Model A
    logger.info("=" * 80)
    logger.info("Step 1: Generating responses for Model A")
    logger.info("=" * 80)

    if is_main:
        if cache.check_generations_a(args.step, args.num_generations_per_model):
            logger.info(f"Cache hit: Model A generations for step {args.step} already exist")
            is_valid, responses_a, _np_a, cached_prompts_a, cached_finish_a = cache.load_generations_a(
                args.step, args.num_generations_per_model
            )
            if is_valid:
                all_text_responses_a = responses_a
                if cached_prompts_a is not None:
                    prompts_a = cached_prompts_a
                all_finish_reasons_a = cached_finish_a
            else:
                logger.warning("Cache data invalid, regenerating...")
                is_valid = False

        if not cache.check_generations_a(args.step, args.num_generations_per_model) or not is_valid:
            gen_out = generate_responses_for_model(
                model_path=args.model_a_path,
                unique_prompts=prompts_a,
                num_generations=args.num_generations_per_model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_completion_length,
                repetition_penalty=args.repetition_penalty,
                **_gen_kwargs("a"),
            )
            gen_output_a = gen_out
            all_text_responses_a = gen_out.response_texts
            all_finish_reasons_a = gen_out.finish_reasons
            cache.save_generations_a(
                args.step, len(prompts_a), all_text_responses_a,
                prompts=prompts_a, finish_reasons=all_finish_reasons_a,
            )
            logger.info(f"Saved Model A generations to cache: {len(all_text_responses_a)} responses")

    wait_for_everyone(accelerator)
    if not is_main:
        _ok, all_text_responses_a, _np, cached_prompts_a, cached_finish_a = cache.load_generations_a(
            args.step, args.num_generations_per_model
        )
        if cached_prompts_a is not None:
            prompts_a = cached_prompts_a
        all_finish_reasons_a = cached_finish_a
    tc_check_generations_block(
        where="co.generate_and_score.generations_a",
        prompts=prompts_a,
        responses=all_text_responses_a,
        num_generations=int(args.num_generations_per_model),
        finish_reasons=all_finish_reasons_a,
    )

    # Step 2: Generate responses for Model B
    logger.info("=" * 80)
    logger.info("Step 2: Generating responses for Model B")
    logger.info("=" * 80)

    if is_main:
        is_valid = False
        if cache.check_generations_b(args.step, args.num_generations_per_model):
            logger.info(f"Cache hit: Model B generations for step {args.step} already exist")
            is_valid_b, responses_b, _np_b, cached_prompts_b, cached_finish_b = cache.load_generations_b(
                args.step, args.num_generations_per_model
            )
            if is_valid_b:
                all_text_responses_b = responses_b
                if cached_prompts_b is not None:
                    prompts_b = cached_prompts_b
                all_finish_reasons_b = cached_finish_b
                is_valid = True

        if not is_valid:
            gen_out = generate_responses_for_model(
                model_path=args.model_b_path,
                unique_prompts=prompts_b,
                num_generations=args.num_generations_per_model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_completion_length,
                repetition_penalty=args.repetition_penalty,
                **_gen_kwargs("b"),
            )
            gen_output_b = gen_out
            all_text_responses_b = gen_out.response_texts
            all_finish_reasons_b = gen_out.finish_reasons
            cache.save_generations_b(
                args.step, len(prompts_b), all_text_responses_b,
                prompts=prompts_b, finish_reasons=all_finish_reasons_b,
            )
            logger.info(f"Saved Model B generations to cache: {len(all_text_responses_b)} responses")

    wait_for_everyone(accelerator)
    if not is_main:
        _ok, all_text_responses_b, _np, cached_prompts_b, cached_finish_b = cache.load_generations_b(
            args.step, args.num_generations_per_model
        )
        if cached_prompts_b is not None:
            prompts_b = cached_prompts_b
        all_finish_reasons_b = cached_finish_b
    tc_check_generations_block(
        where="co.generate_and_score.generations_b",
        prompts=prompts_b,
        responses=all_text_responses_b,
        num_generations=int(args.num_generations_per_model),
        finish_reasons=all_finish_reasons_b,
    )

    # In legacy HTTP mode, stop servers before scoring/logps to free GPU memory.
    if not use_offline and args.auto_stop_server:
        stop_vllm_server_if_possible(args.vllm_port_a, is_main=is_main)
        stop_vllm_server_if_possible(args.vllm_port_b, is_main=is_main)
    wait_for_everyone(accelerator)
    
    # Verify total responses
    total_responses = len(all_text_responses_a) + len(all_text_responses_b)
    expected_total = args.problems_per_step * args.num_generations_per_model * 2
    logger.info(f"Total responses: {total_responses} (expected: {expected_total})")
    if total_responses != expected_total:
        logger.warning(f"Total responses mismatch! Got {total_responses}, expected {expected_total}")
    
    # Step 3+4: Compute rewards (unified via utils.reward)
    reward_method = getattr(args, "reward_method", "sum_scores")
    needs_cross_model_scores = reward_method in ("sum_scores", "co_wmv", "mixed_reward")
    num_unique_prompts_step = len(prompts_a)
    N = args.num_generations_per_model

    logger.info("=" * 80)
    logger.info(f"Step 3+4: Computing rewards (reward_method={reward_method!r})")
    logger.info("=" * 80)

    # Try loading cached scores first.
    scores_a_cached = scores_b_cached = False
    if cache.check_scores_a(args.step):
        is_valid, scores_a = cache.load_scores_a(args.step)
        if is_valid and scores_a is not None:
            spp = len(scores_a) // num_unique_prompts_step
            advantages_a = torch.tensor(scores_a).view(num_unique_prompts_step, spp)
            scores_a_cached = True
            if is_main:
                logger.info(f"Cache hit: scores_a step={args.step}, shape={advantages_a.shape}")
    if cache.check_scores_b(args.step):
        is_valid, scores_b = cache.load_scores_b(args.step)
        if is_valid and scores_b is not None:
            spp = len(scores_b) // num_unique_prompts_step
            advantages_b = torch.tensor(scores_b).view(num_unique_prompts_step, spp)
            scores_b_cached = True
            if is_main:
                logger.info(f"Cache hit: scores_b step={args.step}, shape={advantages_b.shape}")

    if not (scores_a_cached and scores_b_cached):
        # For SCE-based methods, we still need model inference to get SCE scores.
        sce_scores_a = sce_scores_b = None
        if reward_method in SCE_METHODS:
            score_only_a = "a" if not needs_cross_model_scores else None
            score_only_b = "b" if not needs_cross_model_scores else None

            if not scores_a_cached:
                logger.info(f"Computing SCE scores with Model A (score_only={score_only_a!r})...")
                sce_scores_a = score_responses_for_model(
                    model_path=args.model_a_path,
                    tokenizer_path=(args.tokenizer_a_path or args.model_a_path),
                    unique_prompt_examples=sampled_unique_prompts_raw,
                    all_text_responses_a=all_text_responses_a,
                    all_text_responses_b=all_text_responses_b,
                    all_finish_reasons_a=all_finish_reasons_a,
                    all_finish_reasons_b=all_finish_reasons_b,
                    num_generations_per_model=N,
                    max_prompt_length=args.max_prompt_length,
                    max_completion_length=args.max_completion_length,
                    mask_truncated_completions=False,
                    batch_size=args.scoring_batch_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    accelerator=accelerator,
                    debug_zero_advantages_path=(
                        os.path.join(args.output_dir, f"debug_scores_a_zeros_step{args.step}.jsonl")
                        if bool(args.debug_scores_a_zeros)
                        else None
                    ),
                    debug_zero_advantages_max=int(args.debug_scores_a_zeros_max),
                    score_only=score_only_a,
                    debug_output_dir=args.output_dir,
                    debug_step=int(args.step),
                    debug_model_tag="score_a",
                )

            if not scores_b_cached:
                logger.info(f"Computing SCE scores with Model B (score_only={score_only_b!r})...")
                sce_scores_b = score_responses_for_model(
                    model_path=args.model_b_path,
                    tokenizer_path=(args.tokenizer_b_path or args.model_b_path),
                    unique_prompt_examples=sampled_unique_prompts_raw,
                    all_text_responses_a=all_text_responses_a,
                    all_text_responses_b=all_text_responses_b,
                    all_finish_reasons_a=all_finish_reasons_a,
                    all_finish_reasons_b=all_finish_reasons_b,
                    num_generations_per_model=N,
                    max_prompt_length=args.max_prompt_length,
                    max_completion_length=args.max_completion_length,
                    mask_truncated_completions=False,
                    batch_size=args.scoring_batch_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    accelerator=accelerator,
                    score_only=score_only_b,
                    debug_output_dir=args.output_dir,
                    debug_step=int(args.step),
                    debug_model_tag="score_b",
                )

        # Build ground_truths for truth method.
        ground_truths = None
        if reward_method == "truth":
            answer_col = get_answer_column(args.dataset_key)
            ground_truths = [str(p.get(answer_col, "")) for p in sampled_unique_prompts_raw]
            missing = sum(1 for gt in ground_truths if not gt)
            if missing:
                logger.warning(f"{missing}/{len(ground_truths)} prompts have empty ground-truth")

        advantages_a, advantages_b = compute_rewards(
            reward_method=reward_method,
            responses_a=all_text_responses_a,
            responses_b=all_text_responses_b,
            num_prompts=num_unique_prompts_step,
            num_generations=N,
            ground_truths=ground_truths,
            scorer_name=str(getattr(args, "scorer", "hacrl_prime_math")),
            dataset_hint=str(args.dataset_key),
            sce_scores_a=sce_scores_a,
            sce_scores_b=sce_scores_b,
            step=int(args.step),
        )

        if is_main:
            cache.save_scores_a(args.step, advantages_a.cpu().tolist())
            cache.save_scores_b(args.step, advantages_b.cpu().tolist())
            logger.info(f"Saved rewards: A {advantages_a.shape}, B {advantages_b.shape}")

    wait_for_everyone(accelerator)
    
    expected_scores_per_prompt = (
        int(args.num_generations_per_model)
        if reward_method in ("truth", "co_mv", "self_mv", "self_scores")
        else int(args.num_generations_per_model) * 2
    )
    tc_check_score_vector(
        where="co.generate_and_score.scores_a",
        scores=advantages_a.reshape(-1).tolist(),
        num_prompts=len(prompts_a),
        allowed_per_prompt=[expected_scores_per_prompt],
    )
    tc_check_score_vector(
        where="co.generate_and_score.scores_b",
        scores=advantages_b.reshape(-1).tolist(),
        num_prompts=len(prompts_b),
        allowed_per_prompt=[expected_scores_per_prompt],
    )

    # Step 5+6: Compute and save logps
    #
    # With offline vLLM (use_offline=True), old_logps are extracted directly from
    # generation output — no separate forward pass needed.  Only ref_logps (when
    # beta > 0) require a model forward pass via compute_logps_for_model().
    #
    # With legacy HTTP mode, old_logps still require a separate forward pass.
    if args.compute_logps:
        for model_tag, gen_output_ref, model_path, tok_path, ref_path, responses_own, port_attr in [
            ("a", gen_output_a, args.model_a_path, (args.tokenizer_a_path or args.model_a_path),
             args.ref_model_a_path, all_text_responses_a, "vllm_port_a"),
            ("b", gen_output_b, args.model_b_path, (args.tokenizer_b_path or args.model_b_path),
             args.ref_model_b_path, all_text_responses_b, "vllm_port_b"),
        ]:
            step_label = "5" if model_tag == "a" else "6"
            logger.info("=" * 80)
            logger.info(f"Step {step_label}: Computing logps for Model {model_tag.upper()}")
            logger.info("=" * 80)

            cache_check = cache.check_logps_a if model_tag == "a" else cache.check_logps_b
            cache_save = cache.save_logps_a if model_tag == "a" else cache.save_logps_b

            if cache_check(args.step):
                logger.info(f"Cache hit: Model {model_tag.upper()} logps for step {args.step} already exist")
                continue

            # --- old_logps ---
            old_per_token_logps = None
            compl_lengths: Optional[List[int]] = None

            if args.num_iterations > 1 or args.force_compute_old_logps:
                if use_offline and gen_output_ref is not None and gen_output_ref.old_log_probs:
                    # HACRL-style: use logprobs from vLLM generation directly.
                    logger.info(f"Using old_logps from vLLM generation for Model {model_tag.upper()} (no extra forward pass)")
                    padded_logps, ltk = gen_output_ref.pad_old_logps()
                    old_per_token_logps = torch.tensor(padded_logps, dtype=torch.float32) if padded_logps else None
                    compl_lengths = [len(lp) for lp in gen_output_ref.old_log_probs]
                else:
                    # Legacy: separate forward pass.
                    logger.info(f"Computing old_logps for Model {model_tag.upper()} (forward pass)...")
                    old_per_token_logps, compl_lengths = compute_logps_for_model(
                        model_path=model_path,
                        tokenizer_path=tok_path,
                        unique_prompt_examples=sampled_unique_prompts_raw,
                        all_text_responses_a=all_text_responses_a,
                        all_text_responses_b=all_text_responses_b,
                        all_finish_reasons_a=all_finish_reasons_a,
                        all_finish_reasons_b=all_finish_reasons_b,
                        debug_output_dir=args.output_dir,
                        debug_step=int(args.step),
                        debug_model_tag=f"logps_old_{model_tag}",
                        num_generations_per_model=args.num_generations_per_model,
                        max_prompt_length=args.max_prompt_length,
                        max_completion_length=args.max_completion_length,
                        mask_truncated_completions=False,
                        batch_size=args.logps_batch_size,
                        temperature=args.temperature,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        accelerator=accelerator,
                        score_only=model_tag,
                    )

            # --- ref_logps (only if beta > 0) ---
            ref_per_token_logps = None
            if args.beta > 0.0:
                ref_model = ref_path if ref_path else model_path
                logger.info(f"Computing ref_logps for Model {model_tag.upper()} using {ref_model}...")
                ref_per_token_logps, _cl_ref = compute_logps_for_model(
                    model_path=ref_model,
                    tokenizer_path=tok_path,
                    unique_prompt_examples=sampled_unique_prompts_raw,
                    all_text_responses_a=all_text_responses_a,
                    all_text_responses_b=all_text_responses_b,
                    all_finish_reasons_a=all_finish_reasons_a,
                    all_finish_reasons_b=all_finish_reasons_b,
                    debug_output_dir=args.output_dir,
                    debug_step=int(args.step),
                    debug_model_tag=f"logps_ref_{model_tag}",
                    num_generations_per_model=args.num_generations_per_model,
                    max_prompt_length=args.max_prompt_length,
                    max_completion_length=args.max_completion_length,
                    mask_truncated_completions=False,
                    batch_size=args.logps_batch_size,
                    temperature=args.temperature,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    accelerator=accelerator,
                    score_only=model_tag,
                )
                if compl_lengths is None:
                    compl_lengths = _cl_ref

            # Re-align vLLM offline old_logps to forward-pass ref width.
            # tokens→text→tokens round-trip can change completion lengths,
            # so vLLM logps width may differ from re-tokenized ref width.
            if (
                old_per_token_logps is not None
                and ref_per_token_logps is not None
                and old_per_token_logps.dim() == 2
                and ref_per_token_logps.dim() == 2
                and old_per_token_logps.size(1) != ref_per_token_logps.size(1)
            ):
                _tw = int(ref_per_token_logps.size(1))
                logger.info(
                    f"Re-aligning old_logps width {old_per_token_logps.size(1)} -> {_tw} "
                    f"(ref re-tokenization width) for Model {model_tag.upper()}"
                )
                if old_per_token_logps.size(1) > _tw:
                    old_per_token_logps = old_per_token_logps[:, :_tw]
                else:
                    old_per_token_logps = torch.nn.functional.pad(
                        old_per_token_logps, (0, _tw - old_per_token_logps.size(1))
                    )

            # --- Save to cache ---
            if is_main:
                tc_check_logps_matrix(
                    where=f"co.generate_and_score.logps_{model_tag}.old",
                    logps=old_per_token_logps,
                    expected_rows=len(responses_own),
                )
                tc_check_logps_matrix(
                    where=f"co.generate_and_score.logps_{model_tag}.ref",
                    logps=ref_per_token_logps,
                    expected_rows=len(responses_own),
                )
                old_list = old_per_token_logps.cpu().tolist() if old_per_token_logps is not None else None
                ref_list = ref_per_token_logps.cpu().tolist() if ref_per_token_logps is not None else None
                _ltk = None
                try:
                    src = ref_per_token_logps if ref_per_token_logps is not None else old_per_token_logps
                    _ltk = int(src.size(1)) if src is not None and src.dim() == 2 else None
                except Exception:
                    _ltk = None
                cache_save(
                    args.step, old_list, ref_list,
                    alignment="completion_padded_v1",
                    logits_to_keep=_ltk,
                    completion_lengths=compl_lengths,
                )
                logger.info(
                    f"Saved Model {model_tag.upper()} logps: "
                    f"old={old_list is not None}, ref={ref_list is not None}"
                )

            wait_for_everyone(accelerator)
    
    logger.info("=" * 80)
    logger.info("Generate and Score Pipeline Completed!")
    logger.info("=" * 80)
    logger.info(f"Summary:")
    logger.info(f"  Unique prompts: {len(prompts_a)} (A template), {len(prompts_b)} (B template)")
    logger.info(f"  Model A responses: {len(all_text_responses_a)}")
    logger.info(f"  Model B responses: {len(all_text_responses_b)}")
    logger.info(f"  Total responses: {len(all_text_responses_a) + len(all_text_responses_b)}")
    logger.info(f"  Model A scores shape: {advantages_a.shape}")
    logger.info(f"  Model B scores shape: {advantages_b.shape}")


if __name__ == "__main__":
    main()
