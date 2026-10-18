# coding=utf-8
"""
Score and Compute Pipeline for Single-Model (with accelerate).

Loads cached generations (from generate_only.py --cache_slot a), then:
1. Scores completions (SCE or ground-truth)
2. Computes per-token logps (old + ref)
3. Saves results to cache

This script MUST be launched via accelerate for multi-GPU scoring/logps.

Usage:
    accelerate launch --num_processes $NUM_GPUS \
        src/score_and_compute_single.py \
        --policy_model_path /path/to/model \
        --dataset_key math_500 --output_dir ./cache --step 0
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
from utils.dataset import load_co_intuitor_dataset, get_answer_column
from utils.reward import score_responses
from utils.logps import get_per_token_logps
from utils.tokenization import build_left_padded_prompt_completion_batch
from utils.env_utils import co_int_env
from utils.prompt_utils import render_prompts, flatten_prompts
from utils.model_utils import get_tokenizer, get_model
from utils.distributed_utils import (
    get_accelerator,
    is_main_process,
    wait_for_everyone,
    shard_range,
)
from utils.debug_utils import co_dump_alignment_samples, co_dump_single_logps_samples

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


from utils.logps import compute_per_token_logps_from_prebuilt_batch as _compute_per_token_logps_from_prebuilt_batch


def main():
    parser = argparse.ArgumentParser(
        description="Score and compute logps for single-model pipeline (accelerate). "
        "Reads cached generations from generate_only.py."
    )
    parser.add_argument("--policy_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer path (defaults to policy_model_path)")
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Reward model path for SCE scoring (defaults to policy model)")

    # Dataset
    parser.add_argument("--dataset_key", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=0)

    # Sampling (must match generate_only.py)
    parser.add_argument("--problems_per_step", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument("--shuffle_dataset", action=argparse.BooleanOptionalAction, default=False)

    # Generation params (for cache loading)
    parser.add_argument("--num_generations", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=2048)

    # Scoring
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--scoring_batch_size", type=int, default=8)
    parser.add_argument("--mask_truncated_completions", action="store_true", default=False)

    # Reward
    parser.add_argument("--reward_mode", type=str, default="intuitor",
                        choices=["intuitor", "wmv", "mv", "truth", "trajectory", "mixed_reward"])
    parser.add_argument("--scorer", type=str, default="hacrl_prime_math")

    # Logps
    parser.add_argument("--compute_logps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logps_batch_size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--num_iterations", type=int, default=1)

    # DAPO: overlong reward penalty
    parser.add_argument("--overlong_penalty", type=float, default=None)

    # GSPO: always compute old_logps even when num_iterations==1 so the
    # sequence-level IS ratio is meaningful across optimizer steps.
    parser.add_argument("--force_compute_old_logps", action="store_true", default=False)

    args = parser.parse_args()

    accelerator = get_accelerator()
    is_main = is_main_process(accelerator)
    logger.setLevel(logging.INFO)
    if not is_main:
        logger.setLevel(logging.WARNING)

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

    sampled_raw = [train_problems[i] for i in sampled_indices]

    # ------------------------------------------------------------------
    # Load cached generations
    # ------------------------------------------------------------------
    _et_map = {"true": True, "false": False, "none": None}
    _chat_kwargs = {"enable_thinking": _et_map[args.enable_thinking]}

    ok_g, cached_resps, cached_np, cached_prompts, cached_finish = cache.load_generations_a(
        args.step, args.num_generations
    )
    if not ok_g or cached_resps is None:
        raise RuntimeError(f"No cached generations_a for step {args.step}. Run generate_only.py first.")

    flat_completions = list(cached_resps)
    policy_tokenizer = get_tokenizer(args.tokenizer_path or args.policy_model_path)
    if cached_prompts is not None:
        prompts_text = list(cached_prompts)
    else:
        prompts_text = render_prompts(
            sampled_raw, policy_tokenizer, apply_chat_template_kwargs=_chat_kwargs
        )

    finish_reasons = cached_finish
    if finish_reasons is not None and len(finish_reasons) != len(flat_completions):
        finish_reasons = [None] * len(flat_completions)

    completion_is_truncated: Optional[List[Optional[bool]]] = None
    if finish_reasons is not None and len(finish_reasons) == len(flat_completions):
        completion_is_truncated = [str(x) == "length" for x in finish_reasons]

    logger.info(f"Loaded cached generations: {len(flat_completions)} completions, {len(prompts_text)} prompts")

    # ------------------------------------------------------------------
    # Scoring (SCE or ground-truth)
    # ------------------------------------------------------------------
    reward_mode = getattr(args, "reward_mode", "intuitor")

    if cache.check_scores_a(args.step):
        ok, cached_scores = cache.load_scores_a(args.step)
        if ok and cached_scores is not None and len(cached_scores) == len(flat_completions):
            sce_scores = cached_scores
            logger.info(f"Cache hit: scores_a step={args.step} ({len(sce_scores)} scores)")
        else:
            sce_scores = []
    else:
        sce_scores = []

    if not sce_scores:
        if reward_mode == "truth":
            logger.info("Scoring completions against ground-truth answers (reward_mode=truth)...")
            answer_col = get_answer_column(args.dataset_key)
            ground_truths = [str(p.get(answer_col, "")) for p in sampled_raw]
            missing = sum(1 for gt in ground_truths if not gt)
            if missing:
                logger.warning(f"{missing}/{len(ground_truths)} prompts have empty ground-truth (answer_column={answer_col!r})")

            from eval.common.gt_score import score_completions_against_ground_truth
            scores_all = score_completions_against_ground_truth(
                completions=flat_completions,
                ground_truths=ground_truths,
                num_generations=int(args.num_generations),
                scorer_name=str(getattr(args, "scorer", "hacrl_prime_math")),
                dataset_hint=str(args.dataset_key),
            )

            overlong_penalty = getattr(args, "overlong_penalty", None)
            if overlong_penalty is not None and completion_is_truncated is not None:
                n_penalized = 0
                for idx, is_trunc in enumerate(completion_is_truncated):
                    if is_trunc:
                        scores_all[idx] = float(overlong_penalty)
                        n_penalized += 1
                if n_penalized > 0:
                    logger.info(f"Overlong penalty: {n_penalized}/{len(scores_all)} truncated completions set to {overlong_penalty}")
        else:
            scorer_path = args.reward_model_path or args.policy_model_path
            logger.info("Scoring completions with reward model (SCE)...")

            if accelerator is not None and torch.cuda.is_available():
                device_str = str(getattr(accelerator, "device", torch.device("cuda")))
            else:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
            reward_model = get_model(scorer_path, device=device_str)
            reward_tok = get_tokenizer(scorer_path)
            reward_prompts_text = render_prompts(
                sampled_raw, reward_tok, apply_chat_template_kwargs=_chat_kwargs
            )

            flat_prompts = flatten_prompts(reward_prompts_text, args.num_generations)

            n_total = len(flat_completions)
            start, end, per_rank = shard_range(n_total, accelerator)
            local_scores = score_responses(
                model=reward_model, tokenizer=reward_tok,
                prompts=flat_prompts[start:end],
                responses=flat_completions[start:end],
                accelerator=accelerator,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                mask_truncated_completions=args.mask_truncated_completions,
                completion_is_truncated=(completion_is_truncated[start:end] if completion_is_truncated is not None else None),
                batch_size=args.scoring_batch_size,
                device=None,
            )

            if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
                torch_device = getattr(accelerator, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                local_t = torch.tensor(local_scores, dtype=torch.float32)
                padded = torch.zeros((int(per_rank),), dtype=local_t.dtype)
                if local_t.numel() > 0:
                    padded[: local_t.numel()] = local_t
                padded = padded.to(torch_device, non_blocking=True)
                gathered = accelerator.gather(padded)
                scores_all = gathered[:n_total].detach().cpu().tolist()
            else:
                scores_all = list(local_scores)
            # Aggressively release reward model memory before logps stage loads policy/ref models.
            del reward_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if is_main:
            cache.save_scores_a(args.step, scores_all)
            logger.info("Saved scores to cache (scores_a).")
        wait_for_everyone(accelerator)
        ok_s2, cached_scores2 = cache.load_scores_a(args.step)
        if ok_s2 and cached_scores2 is not None and len(cached_scores2) == len(flat_completions):
            sce_scores = cached_scores2

    # ------------------------------------------------------------------
    # Logps (policy + optional ref)
    # ------------------------------------------------------------------
    if args.compute_logps:
        need_old = (
            int(getattr(args, "num_iterations", 1)) > 1
            or getattr(args, "force_compute_old_logps", False)
        )
        need_ref = True

        ok_lp, old_lp, ref_lp = cache.load_logps_a(args.step)
        cache_has_old = ok_lp and old_lp is not None and len(old_lp) == len(flat_completions)
        cache_has_ref = ok_lp and ref_lp is not None and len(ref_lp) == len(flat_completions)
        cache_ok = ((not need_old) or cache_has_old) and ((not need_ref) or cache_has_ref)

        if cache_ok:
            logger.info(f"Cache hit: logps_a step={args.step}")
        else:
            logger.info(
                f"Computing logps (single): need_old={need_old} (num_iterations={getattr(args,'num_iterations',1)}), "
                f"need_ref={need_ref}"
            )

            ref_path = args.ref_model_path or args.policy_model_path
            ref_logps: Optional[List[List[float]]] = None
            old_logps: Optional[List[List[float]]] = None
            logits_to_keep_meta: Optional[int] = None

            flat_prompts_policy = flatten_prompts(prompts_text, args.num_generations)
            policy_tok = get_tokenizer(args.tokenizer_path or args.policy_model_path)
            pc_ids, pc_mask, cmw, logits_to_keep = build_left_padded_prompt_completion_batch(
                policy_tok, flat_prompts_policy, flat_completions,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                mask_truncated_completions=args.mask_truncated_completions,
                completion_is_truncated=completion_is_truncated,
                device=torch.device("cpu"),
            )
            pc_mask[:, -int(logits_to_keep):] = cmw
            completion_ids_all = pc_ids[:, -int(logits_to_keep):].detach().cpu()
            if logits_to_keep_meta is None:
                logits_to_keep_meta = int(logits_to_keep)

            co_dump_alignment_samples(
                stage="logps", output_dir=str(args.output_dir), step=int(args.step),
                model_tag="policy", tokenizer=policy_tok,
                prompts=flat_prompts_policy, responses=flat_completions,
                finish_is_truncated=completion_is_truncated,
                prompt_completion_ids=pc_ids, completion_mask_window=cmw,
                logits_to_keep=int(logits_to_keep),
                max_items=co_int_env("CO_DEBUG_DATA_N", 3),
                is_main=is_main_process(accelerator),
            )

            if str(ref_path) == str(args.policy_model_path):
                lp_t = _compute_per_token_logps_from_prebuilt_batch(
                    model_path=args.policy_model_path,
                    prompt_completion_ids=pc_ids, attention_mask=pc_mask,
                    logits_to_keep=int(logits_to_keep),
                    temperature=float(args.temperature),
                    batch_size=int(args.logps_batch_size),
                    accelerator=accelerator,
                )
                lp = lp_t.tolist()
                ref_logps = lp
                if need_old and old_logps is None:
                    old_logps = lp
            else:
                if need_old and old_logps is None:
                    old_t = _compute_per_token_logps_from_prebuilt_batch(
                        model_path=args.policy_model_path,
                        prompt_completion_ids=pc_ids, attention_mask=pc_mask,
                        logits_to_keep=int(logits_to_keep),
                        temperature=float(args.temperature),
                        batch_size=int(args.logps_batch_size),
                        accelerator=accelerator,
                    )
                    old_logps = old_t.tolist()

                logger.info(f"Computing ref_logps using reference model: {ref_path} ...")
                ref_tok = get_tokenizer(ref_path)
                ref_prompts_text = render_prompts(
                    sampled_raw, ref_tok, apply_chat_template_kwargs=_chat_kwargs
                )

                flat_prompts_ref = flatten_prompts(ref_prompts_text, args.num_generations)
                pc_ids_ref, pc_mask_ref, cmw_ref, logits_to_keep_ref = build_left_padded_prompt_completion_batch(
                    ref_tok, flat_prompts_ref, flat_completions,
                    max_prompt_length=args.max_prompt_length,
                    max_completion_length=args.max_completion_length,
                    mask_truncated_completions=args.mask_truncated_completions,
                    completion_is_truncated=completion_is_truncated,
                    device=torch.device("cpu"),
                )
                pc_mask_ref[:, -int(logits_to_keep_ref):] = cmw_ref

                co_dump_alignment_samples(
                    stage="logps", output_dir=str(args.output_dir), step=int(args.step),
                    model_tag="ref", tokenizer=ref_tok,
                    prompts=flat_prompts_ref, responses=flat_completions,
                    finish_is_truncated=completion_is_truncated,
                    prompt_completion_ids=pc_ids_ref, completion_mask_window=cmw_ref,
                    logits_to_keep=int(logits_to_keep_ref),
                    max_items=co_int_env("CO_DEBUG_DATA_N", 3),
                    is_main=is_main_process(accelerator),
                )

                ref_t = _compute_per_token_logps_from_prebuilt_batch(
                    model_path=ref_path,
                    prompt_completion_ids=pc_ids_ref, attention_mask=pc_mask_ref,
                    logits_to_keep=int(logits_to_keep_ref),
                    temperature=float(args.temperature),
                    batch_size=int(args.logps_batch_size),
                    accelerator=accelerator,
                )
                ref_logps = ref_t.tolist()
                logits_to_keep_meta = int(logits_to_keep_ref)

            if is_main:
                co_dump_single_logps_samples(
                    base_dir=str(args.output_dir), step=int(args.step),
                    prompts_preview=flatten_prompts(prompts_text, args.num_generations)[: int(os.environ.get("CO_DEBUG_DATA_N", "3"))],
                    responses_preview=flat_completions[: int(os.environ.get("CO_DEBUG_DATA_N", "3"))],
                    finish_preview=finish_reasons,
                    completion_ids=completion_ids_all,
                    old_lp=old_logps, ref_lp=ref_logps,
                    max_items=int(os.environ.get("CO_DEBUG_DATA_N", "3")),
                )

                cache.save_logps_a(
                    args.step, old_logps, ref_logps,
                    alignment="completion_padded_v1",
                    logits_to_keep=logits_to_keep_meta,
                )
                logger.info(
                    f"Saved logps to cache (logps_a): old_logps={old_logps is not None}, "
                    f"ref_logps={ref_logps is not None}, alignment=completion_padded_v1"
                )
            wait_for_everyone(accelerator)

    logger.info("Done.")


if __name__ == "__main__":
    main()
