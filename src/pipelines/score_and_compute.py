# coding=utf-8
"""
Score and Compute Pipeline for Co-Intuitor (with accelerate).

Loads cached generations (from generate_only.py), then:
1. Computes rewards (SCE / ground-truth)
2. Computes per-token logps (old + ref)
3. Saves results to cache

This script MUST be launched via accelerate for multi-GPU scoring/logps.

Usage:
    accelerate launch --num_processes $NUM_GPUS \
        src/score_and_compute.py \
        --model_a_path /path/to/a --model_b_path /path/to/b \
        --dataset_key math_500 --output_dir ./cache --step 0
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.cache import CoIntuitorCache
from utils.dataset import load_co_intuitor_dataset, get_answer_column
from utils.reward import compute_rewards, SCE_METHODS
from utils.reward import get_per_token_self_certainty, get_advantage_from_sce
from utils.logps import get_per_token_logps
from utils.tokenization import build_left_padded_prompt_completion_batch
from utils.env_utils import co_env_flag, co_int_env
from utils.prompt_utils import render_prompt, render_prompts
from utils.model_utils import get_tokenizer, get_model
from utils.distributed_utils import (
    get_accelerator,
    is_main_process,
    wait_for_everyone,
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


from utils.scoring_helpers import score_responses_for_model  # noqa: E402


# ---------------------------------------------------------------------------
# Logps helper (same as in generate_and_score.py)
# ---------------------------------------------------------------------------
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
    apply_chat_template_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Compute per-token log probabilities for responses using a model."""
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
    unique_prompts = render_prompts(
        unique_prompt_examples,
        tokenizer,
        has_template=has_template,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )
    num_unique_prompts = len(unique_prompts)

    expected_responses_a = num_unique_prompts * num_generations_per_model
    expected_responses_b = num_unique_prompts * num_generations_per_model
    if len(all_text_responses_a) != expected_responses_a:
        raise ValueError(f"Expected {expected_responses_a} responses from Model A, got {len(all_text_responses_a)}")
    if len(all_text_responses_b) != expected_responses_b:
        raise ValueError(f"Expected {expected_responses_b} responses from Model B, got {len(all_text_responses_b)}")

    prompts_for_logps = []
    responses_for_logps = []
    trunc_for_logps: Optional[List[Optional[bool]]] = (
        [] if (all_finish_reasons_a is not None or all_finish_reasons_b is not None) else None
    )

    for i in range(num_unique_prompts):
        prompt = unique_prompts[i]
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
            fr_a = all_finish_reasons_a[start_a:end_a] if all_finish_reasons_a is not None else [None] * len(responses_a)
            fr_b = all_finish_reasons_b[start_b:end_b] if all_finish_reasons_b is not None else [None] * len(responses_b)
            if score_only == "a":
                combined_fr = list(fr_a)
            elif score_only == "b":
                combined_fr = list(fr_b)
            else:
                combined_fr = list(fr_a) + list(fr_b)
            trunc_for_logps.extend([None if x is None else (str(x) == "length") for x in combined_fr])

        prompts_for_logps.extend([prompt] * len(combined_responses))
        responses_for_logps.extend(combined_responses)

    build_device = torch.device("cpu")
    prompt_completion_ids, attention_mask, completion_mask_window, logits_to_keep = (
        build_left_padded_prompt_completion_batch(
            tokenizer, prompts_for_logps, responses_for_logps,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            mask_truncated_completions=mask_truncated_completions,
            completion_is_truncated=trunc_for_logps,
            device=build_device,
        )
    )
    attention_mask[:, -logits_to_keep:] = completion_mask_window
    completion_lengths_all: List[int] = completion_mask_window.sum(dim=1).int().tolist()

    if debug_output_dir is not None and debug_step is not None and debug_model_tag is not None:
        co_dump_alignment_samples(
            stage="logps", output_dir=str(debug_output_dir), step=int(debug_step),
            model_tag=str(debug_model_tag), tokenizer=tokenizer,
            prompts=prompts_for_logps, responses=responses_for_logps,
            finish_is_truncated=trunc_for_logps,
            prompt_completion_ids=prompt_completion_ids,
            completion_mask_window=completion_mask_window,
            logits_to_keep=int(logits_to_keep),
            max_items=co_int_env("CO_DEBUG_DATA_N", 3),
            is_main=is_main_process(accelerator),
        )

    all_logps = []
    n_total = int(prompt_completion_ids.size(0))
    start, end, per_rank = shard_range(n_total, accelerator)
    is_main = is_main_process(accelerator)

    prompt_completion_ids_local = prompt_completion_ids[start:end]
    attention_mask_local = attention_mask[start:end]

    num_logps_batches = (prompt_completion_ids_local.size(0) + batch_size - 1) // batch_size
    with torch.no_grad():
        iterator = range(0, prompt_completion_ids_local.size(0), batch_size)
        iterator = tqdm(iterator, desc="Computing log probabilities", unit="batch",
                        total=num_logps_batches, leave=False, disable=not is_main)
        for i in iterator:
            end_idx = min(i + batch_size, prompt_completion_ids_local.size(0))
            batch_input_ids = prompt_completion_ids_local[i:end_idx].to(torch_device, non_blocking=True)
            batch_attention_mask = attention_mask_local[i:end_idx].to(torch_device, non_blocking=True)
            per_token_logps = get_per_token_logps(
                model, batch_input_ids, batch_attention_mask, logits_to_keep,
                temperature=temperature, batch_size=batch_size,
            )
            all_logps.append(per_token_logps.detach().cpu())

    if all_logps:
        logps_local = torch.cat(all_logps, dim=0)
    else:
        logps_local = torch.empty((0, logits_to_keep), dtype=torch.float32)

    if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
        padded = torch.zeros((per_rank, logits_to_keep), dtype=logps_local.dtype)
        if logps_local.numel() > 0:
            padded[: logps_local.size(0)] = logps_local
        padded = padded.to(torch_device, non_blocking=True)
        gathered = accelerator.gather(padded)
        logps_all = gathered[:n_total].detach().cpu()
    else:
        logps_all = logps_local

    logger.info(f"Computed log probabilities: shape {logps_all.shape}")
    # Aggressively release model memory before the next model is loaded in this process.
    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logps_all, completion_lengths_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Score and compute logps for Co-Intuitor (accelerate). "
        "Reads cached generations from generate_only.py."
    )

    # Model paths
    parser.add_argument("--model_a_path", type=str, required=True)
    parser.add_argument("--model_b_path", type=str, required=True)
    parser.add_argument("--tokenizer_a_path", type=str, default=None)
    parser.add_argument("--tokenizer_b_path", type=str, default=None)

    # Dataset (needed to re-sample the same problems for scoring tokenizer re-rendering)
    parser.add_argument("--dataset_key", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
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

    # Generation params (for cache loading and logps)
    parser.add_argument("--num_generations_per_model", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=2048)

    # Scoring
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--scoring_batch_size", type=int, default=8)
    parser.add_argument("--mask_truncated_completions", action="store_true", default=False)

    # Reward
    parser.add_argument("--reward_method", type=str, default="sum_scores")
    parser.add_argument("--scorer", type=str, default="hacrl_prime_math")

    # Logps
    parser.add_argument("--compute_logps", action="store_true", default=True)
    parser.add_argument("--logps_batch_size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--ref_model_a_path", type=str, default=None)
    parser.add_argument("--ref_model_b_path", type=str, default=None)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--force_compute_old_logps", action="store_true", default=False)
    parser.add_argument("--compute_cross_old_logps", action="store_true", default=False,
                        help="Compute each model's old logps on the other model's responses (retokenize pipeline).")

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=0)

    # Sampling (must match generate_only.py for deterministic re-sampling)
    parser.add_argument("--problems_per_step", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle_dataset", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    accelerator = get_accelerator()
    is_main = is_main_process(accelerator)

    log_level = logging.INFO
    logger.setLevel(log_level)
    if not is_main:
        logger.setLevel(logging.WARNING)

    logger.info("=" * 80)
    logger.info("Co-Intuitor Score and Compute Pipeline")
    logger.info("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    cache = CoIntuitorCache(os.path.join(args.output_dir, "cache.json"))

    # ------------------------------------------------------------------
    # Re-sample the same problems (deterministic)
    # ------------------------------------------------------------------
    train_problems = load_co_intuitor_dataset(
        dataset_key=args.dataset_key,
        max_samples=args.max_samples,
        system_prompt=args.system_prompt,
        instruction_suffix_override=args.instruction_suffix,
    )
    _et_map = {"true": True, "false": False, "none": None}
    _chat_kwargs = {"enable_thinking": _et_map[args.enable_thinking]}
    num_problems = len(train_problems)

    if bool(getattr(args, "shuffle_dataset", False)):
        batch = int(args.problems_per_step)
        epoch_len = (num_problems // batch) * batch
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
    else:
        start_index = (int(args.step) - 1) * int(args.problems_per_step)  # 1-indexed
        if start_index >= num_problems:
            start_index = start_index % num_problems
        end_index = min(start_index + int(args.problems_per_step), num_problems)
        sampled_indices = list(range(start_index, end_index))
        if len(sampled_indices) < int(args.problems_per_step):
            remaining = int(args.problems_per_step) - len(sampled_indices)
            sampled_indices.extend(list(range(remaining)))

    sampled_unique_prompts_raw = [train_problems[i] for i in sampled_indices]

    # ------------------------------------------------------------------
    # Load cached generations
    # ------------------------------------------------------------------
    ok_a, all_text_responses_a, _np_a, prompts_a, all_finish_reasons_a = cache.load_generations_a(
        args.step, args.num_generations_per_model
    )
    ok_b, all_text_responses_b, _np_b, prompts_b, all_finish_reasons_b = cache.load_generations_b(
        args.step, args.num_generations_per_model
    )

    if not ok_a or all_text_responses_a is None:
        raise RuntimeError(f"No cached generations_a for step {args.step}. Run generate_only.py first.")
    if not ok_b or all_text_responses_b is None:
        raise RuntimeError(f"No cached generations_b for step {args.step}. Run generate_only.py first.")

    if prompts_a is None:
        tokenizer_a = get_tokenizer(args.tokenizer_a_path or args.model_a_path)
        prompts_a = render_prompts(sampled_unique_prompts_raw, tokenizer_a)
    if prompts_b is None:
        tokenizer_b = get_tokenizer(args.tokenizer_b_path or args.model_b_path)
        prompts_b = render_prompts(sampled_unique_prompts_raw, tokenizer_b)

    tc_check_generations_block(
        where="score_and_compute.generations_a",
        prompts=prompts_a, responses=all_text_responses_a,
        num_generations=int(args.num_generations_per_model),
        finish_reasons=all_finish_reasons_a,
    )
    tc_check_generations_block(
        where="score_and_compute.generations_b",
        prompts=prompts_b, responses=all_text_responses_b,
        num_generations=int(args.num_generations_per_model),
        finish_reasons=all_finish_reasons_b,
    )

    total_responses = len(all_text_responses_a) + len(all_text_responses_b)
    logger.info(f"Loaded cached generations: A={len(all_text_responses_a)}, B={len(all_text_responses_b)}, total={total_responses}")

    # ------------------------------------------------------------------
    # Compute rewards
    # ------------------------------------------------------------------
    reward_method = getattr(args, "reward_method", "sum_scores")
    needs_cross_model_scores = reward_method in ("sum_scores", "co_wmv", "co_mv_gated_intuitor", "mixed_reward")
    num_unique_prompts_step = len(prompts_a)
    N = args.num_generations_per_model

    logger.info("=" * 80)
    logger.info(f"Computing rewards (reward_method={reward_method!r})")
    logger.info("=" * 80)

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
                    score_only=score_only_a,
                    debug_output_dir=args.output_dir,
                    debug_step=int(args.step),
                    debug_model_tag="score_a",
                    apply_chat_template_kwargs=_chat_kwargs,
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
                    apply_chat_template_kwargs=_chat_kwargs,
                )

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
        if reward_method in ("truth", "co_mv", "self_mv", "self_scores", "mv_gated_intuitor")
        else int(args.num_generations_per_model) * 2
    )
    tc_check_score_vector(
        where="score_and_compute.scores_a",
        scores=advantages_a.reshape(-1).tolist(),
        num_prompts=len(prompts_a),
        allowed_per_prompt=[expected_scores_per_prompt],
    )
    tc_check_score_vector(
        where="score_and_compute.scores_b",
        scores=advantages_b.reshape(-1).tolist(),
        num_prompts=len(prompts_b),
        allowed_per_prompt=[expected_scores_per_prompt],
    )

    # ------------------------------------------------------------------
    # Compute logps
    # ------------------------------------------------------------------
    if args.compute_logps:
        for model_tag, model_path, tok_path, ref_path, responses_own in [
            ("a", args.model_a_path, (args.tokenizer_a_path or args.model_a_path),
             args.ref_model_a_path, all_text_responses_a),
            ("b", args.model_b_path, (args.tokenizer_b_path or args.model_b_path),
             args.ref_model_b_path, all_text_responses_b),
        ]:
            logger.info("=" * 80)
            logger.info(f"Computing logps for Model {model_tag.upper()}")
            logger.info("=" * 80)

            cache_check = cache.check_logps_a if model_tag == "a" else cache.check_logps_b
            cache_save = cache.save_logps_a if model_tag == "a" else cache.save_logps_b

            if cache_check(args.step):
                logger.info(f"Cache hit: Model {model_tag.upper()} logps for step {args.step}")
                continue

            old_per_token_logps = None
            compl_lengths: Optional[List[int]] = None

            if args.num_iterations > 1 or args.force_compute_old_logps:
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
                    apply_chat_template_kwargs=_chat_kwargs,
                )

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
                    apply_chat_template_kwargs=_chat_kwargs,
                )
                if compl_lengths is None:
                    compl_lengths = _cl_ref

            if is_main:
                tc_check_logps_matrix(
                    where=f"score_and_compute.logps_{model_tag}.old",
                    logps=old_per_token_logps, expected_rows=len(responses_own),
                )
                tc_check_logps_matrix(
                    where=f"score_and_compute.logps_{model_tag}.ref",
                    logps=ref_per_token_logps, expected_rows=len(responses_own),
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
                logger.info(f"Saved Model {model_tag.upper()} logps: old={old_list is not None}, ref={ref_list is not None}")

            wait_for_everyone(accelerator)

    # ------------------------------------------------------------------
    # Compute cross old logps (retokenize pipeline)
    # Model A's old logps on B's responses, and vice versa.
    # ------------------------------------------------------------------
    if getattr(args, "compute_cross_old_logps", False):
        for model_tag, model_path, tok_path, other_tag in [
            ("a", args.model_a_path, (args.tokenizer_a_path or args.model_a_path), "b"),
            ("b", args.model_b_path, (args.tokenizer_b_path or args.model_b_path), "a"),
        ]:
            logger.info("=" * 80)
            logger.info(f"Computing cross old logps: Model {model_tag.upper()} on Model {other_tag.upper()}'s responses")
            logger.info("=" * 80)

            if cache.check_cross_logps(model_tag, args.step):
                logger.info(f"Cache hit: cross_logps_{model_tag} for step {args.step}")
                continue

            cross_logps, cross_compl_lengths = compute_logps_for_model(
                model_path=model_path,
                tokenizer_path=tok_path,
                unique_prompt_examples=sampled_unique_prompts_raw,
                all_text_responses_a=all_text_responses_a,
                all_text_responses_b=all_text_responses_b,
                all_finish_reasons_a=all_finish_reasons_a,
                all_finish_reasons_b=all_finish_reasons_b,
                debug_output_dir=args.output_dir,
                debug_step=int(args.step),
                debug_model_tag=f"cross_logps_{model_tag}_on_{other_tag}",
                num_generations_per_model=args.num_generations_per_model,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                mask_truncated_completions=False,
                batch_size=args.logps_batch_size,
                temperature=args.temperature,
                device="cuda" if torch.cuda.is_available() else "cpu",
                accelerator=accelerator,
                score_only=other_tag,
                apply_chat_template_kwargs=_chat_kwargs,
            )

            if is_main:
                cross_list = cross_logps.cpu().tolist() if cross_logps is not None else None
                _ltk_cross = None
                if cross_logps is not None and cross_logps.dim() == 2:
                    _ltk_cross = int(cross_logps.size(1))
                cache.save_cross_logps(
                    model_tag, args.step, cross_list,
                    alignment="completion_padded_v1",
                    logits_to_keep=_ltk_cross,
                    completion_lengths=cross_compl_lengths,
                )
                logger.info(
                    f"Saved cross old logps: Model {model_tag.upper()} on {other_tag.upper()}'s responses "
                    f"(shape: {cross_logps.shape if cross_logps is not None else None})"
                )

            wait_for_everyone(accelerator)

    logger.info("=" * 80)
    logger.info("Score and Compute Pipeline Completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
