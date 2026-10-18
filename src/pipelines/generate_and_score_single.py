# coding=utf-8
"""
Generate and Score (Single Policy) Pipeline.

This is a single-model variant of the co-learning pipeline:
- A *policy* model generates `num_generations` completions per prompt via vLLM.
- A fixed *reward* model scores each (prompt, completion) via self-certainty (SCE).
- (Optional) compute old/ref per-token logps for GRPO-style training reuse.

Cache format: reuses `utils.cache.CoIntuitorCache` but only fills the "*_a" slots:
- generations_a: policy generations (+ prompts)
- scores_a: SCE scores (one float per completion)
- logps_a: old/ref per-token logps (optional)
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
from utils.generate import (
    generate_responses,
    generate_sequences,
    GenerationOutput,
    strip_prompt_echo_from_flattened_responses,
    strip_chat_template_artifacts_from_flattened_responses,
)
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
    sleep_vllm_server_if_possible,
    stop_vllm_server_if_possible,
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
    parser = argparse.ArgumentParser(description="Generate and score (single policy) for Intuitor/WMV training")
    parser.add_argument("--policy_model_path", type=str, required=True, help="Policy model path (generates completions)")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer source-of-truth (typically the initial/base model dir). If omitted, uses policy_model_path.",
    )
    # Optional: if not provided, we score with the policy model itself (self-SCE).
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=None,
        help="Reward model path (SCE scorer). If omitted, uses the policy model (self-SCE).",
    )
    parser.add_argument("--dataset_key", type=str, required=True, help="Dataset key from recipes/datasets.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cache")
    parser.add_argument("--step", type=int, default=0, help="Step number for cache")

    # Sampling
    parser.add_argument("--problems_per_step", type=int, default=128, help="Number of problems to sample per step")
    parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic sampling.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples in dataset")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt for dataset loader")
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
        help="If set, ignore any cached sampling metadata and restart sampling from epoch 0 (best-effort).",
    )

    # Generation params (defaults aligned with open-r1 GRPOConfig defaults + recipe demo)
    parser.add_argument("--num_generations", type=int, default=7, help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="vLLM repetition_penalty")
    # Project default: cap completions to 2048 tokens to control memory across generate/score/logps/train.
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Max completion length")

    # Scoring/logps params
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length (truncate left)")
    parser.add_argument("--scoring_batch_size", type=int, default=8, help="Batch size for SCE scoring")
    parser.add_argument(
        "--mask_truncated_completions",
        action="store_true",
        default=False,
        help="If set, completions without EOS are fully masked out in scoring/logps.",
    )

    # Reward mode (determines scoring method: SCE vs ground-truth)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="intuitor",
        choices=["intuitor", "wmv", "mv", "truth"],
        help="Reward mode. 'truth' uses ground-truth answers from the dataset instead of SCE scoring.",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="hacrl_prime_math",
        help="HACRL scorer backend for ground-truth reward mode (e.g. hacrl_prime_math, default, hacrl_gsm8k).",
    )

    parser.add_argument(
        "--compute_logps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute old/ref logps (default: true). Use --no-compute_logps to disable.",
    )
    parser.add_argument("--logps_batch_size", type=int, default=8, help="Batch size for logps computation")
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient; if >0, compute ref_logps")
    parser.add_argument("--ref_model_path", type=str, default=None, help="Reference model path (optional)")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of GRPO iterations. If >1, cache old_logps; ref_logps is always computed for single-model pipeline.",
    )

    # vLLM settings
    parser.add_argument(
        "--use_offline_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM offline API (HACRL-style, returns token_ids + logprobs). "
             "If false, falls back to HTTP server API.",
    )
    parser.add_argument("--vllm_gpu", type=int, default=None, help="CUDA device for offline vLLM generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for offline vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for offline vLLM")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16", help="Model dtype for offline vLLM")
    parser.add_argument("--vllm_max_model_len", type=int, default=None, help="Max model context length for offline vLLM")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None, help="Max concurrent sequences for offline vLLM")
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=None, help="Max total batched tokens for offline vLLM")
    parser.add_argument(
        "--enable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM chunked prefill for offline generation",
    )
    # Legacy HTTP server settings (used when --no-use_offline_vllm)
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

    # DAPO: overlong reward penalty
    parser.add_argument(
        "--overlong_penalty",
        type=float,
        default=None,
        help="If set, responses that hit max_length (finish_reason='length') get this reward score "
             "instead of the normal score. DAPO uses -1.0.",
    )

    args = parser.parse_args()
    logger.setLevel(logging.INFO)

    accelerator = get_accelerator()
    is_main = is_main_process(accelerator)
    if not is_main:
        # Keep non-main ranks quiet to avoid log spam under accelerate.
        logger.setLevel(logging.WARNING)
    else:
        try:
            world = int(getattr(accelerator, "num_processes", 1)) if accelerator is not None else 1
            rank = int(getattr(accelerator, "process_index", 0)) if accelerator is not None else 0
            logger.info(f"Distributed: world_size={world}, rank={rank}")
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)
    cache = CoIntuitorCache(os.path.join(args.output_dir, "cache.json"))

    # Always print one line per rank so we can confirm multi-process is actually active.
    # (Non-main ranks run at WARNING log level, so INFO logs may not show.)
    try:
        world = int(getattr(accelerator, "num_processes", 1)) if accelerator is not None else int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(getattr(accelerator, "process_index", 0)) if accelerator is not None else int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        dev = str(getattr(accelerator, "device", "cuda" if torch.cuda.is_available() else "cpu")) if accelerator is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[dist] world_size={world} rank={rank} local_rank={local_rank} device={dev} CUDA_VISIBLE_DEVICES={cvd}", flush=True)
    except Exception:
        pass

    if bool(getattr(args, "mask_truncated_completions", False)) and is_main:
        logger.warning(
            "--mask_truncated_completions is deprecated/ignored in this repo; truncated completions are kept."
        )

    # Load dataset
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
            raise RuntimeError(
                f"open-r1 sampler produced wrong batch length: got {len(sampled_indices)}, expected {batch}."
            )
        if is_main:
            logger.info(
                f"Step {args.step}: Open-R1 sampling epoch={epoch} pos={pos}/{epoch_len} "
                f"(steps_per_epoch={steps_per_epoch}, perm_seed={perm_seed})"
            )
    else:
        # Legacy sequential sampling by step.
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

    policy_tokenizer = get_tokenizer(args.tokenizer_path or args.policy_model_path)
    prompts_text = render_prompts(sampled_raw, policy_tokenizer)

    # Generation (cache hit check) - in distributed mode, ONLY main rank calls vLLM.
    use_offline = bool(getattr(args, "use_offline_vllm", True))
    expected = len(prompts_text) * int(args.num_generations)
    finish_reasons: Optional[List[Optional[str]]] = None
    flat_completions: List[str] = []
    gen_output_ref: Optional[GenerationOutput] = None

    if is_main:
        if cache.check_generations_a(args.step, args.num_generations):
            ok, cached_resps, cached_nprompts, cached_prompts, cached_finish = cache.load_generations_a(
                args.step, args.num_generations
            )
            if (
                ok
                and cached_resps is not None
                and cached_nprompts == len(prompts_text)
                and len(cached_resps) == expected
            ):
                flat_completions = cached_resps
                if cached_prompts is not None:
                    prompts_text = cached_prompts
                finish_reasons = cached_finish
                if finish_reasons is not None and len(finish_reasons) != len(flat_completions):
                    finish_reasons = [None] * len(flat_completions)
                logger.info(f"Cache hit: generations_a step={args.step} ({len(flat_completions)} completions)")
                if not use_offline and not bool(getattr(args, "auto_start_server", False)):
                    sleep_vllm_server_if_possible(int(args.vllm_port), is_main=True)

        if not flat_completions:
            logger.info(
                f"Generating {len(prompts_text)} prompts x {args.num_generations} completions "
                f"(offline={use_offline})..."
            )

            if use_offline:
                gen_out = generate_sequences(
                    model_path=args.policy_model_path,
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
                gen_output_ref = gen_out
            else:
                # Legacy HTTP API path
                gen_out = generate_responses(
                    model_path=args.policy_model_path,
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
            cache.save_generations_a(
                args.step, len(prompts_text), flat_completions, prompts=prompts_text, finish_reasons=finish_reasons
            )
            logger.info("Saved generations to cache (generations_a).")

    # Ensure all ranks see the same cached generations.
    wait_for_everyone(accelerator)
    ok_g, cached_resps, cached_nprompts, cached_prompts, cached_finish = cache.load_generations_a(
        args.step, args.num_generations
    )
    if not ok_g or cached_resps is None or cached_prompts is None or cached_nprompts != len(prompts_text):
        raise RuntimeError("Failed to load generations_a from cache after generation barrier.")
    flat_completions = list(cached_resps)
    prompts_text = list(cached_prompts)
    finish_reasons = cached_finish
    if finish_reasons is not None and len(finish_reasons) != len(flat_completions):
        finish_reasons = [None] * len(flat_completions)

    # Sanitize completions: strip prompt echo + leading chat markers.
    # Offline API returns clean token IDs (no echo), but cached text may still
    # need sanitization if it was generated by the legacy HTTP path.
    flat_completions, _c1 = strip_prompt_echo_from_flattened_responses(
        prompts_text, flat_completions, num_generations=args.num_generations
    )
    flat_completions, _c2 = strip_chat_template_artifacts_from_flattened_responses(
        prompts_text, flat_completions, num_generations=args.num_generations
    )
    changed = bool(_c1 or _c2)
    if changed and is_main:
        cache.save_generations_a(
            args.step,
            len(prompts_text),
            flat_completions,
            prompts=prompts_text,
            finish_reasons=finish_reasons,
        )

    wait_for_everyone(accelerator)
    ok_g2, cached_resps2, _, cached_prompts2, cached_finish2 = cache.load_generations_a(args.step, args.num_generations)
    if ok_g2 and cached_resps2 is not None and cached_prompts2 is not None:
        flat_completions = list(cached_resps2)
        prompts_text = list(cached_prompts2)
        finish_reasons = cached_finish2

    completion_is_truncated: Optional[List[Optional[bool]]] = None
    if finish_reasons is not None and len(finish_reasons) == len(flat_completions):
        completion_is_truncated = [str(x) == "length" for x in finish_reasons]

    # For HTTP mode: stop vLLM server to free GPU memory before scoring/logps.
    if not use_offline and bool(getattr(args, "auto_stop_server", False)):
        stopped = stop_vllm_server_if_possible(int(args.vllm_port), is_main=is_main)
        if is_main and not stopped:
            raise RuntimeError(
                f"vLLM server on port {int(args.vllm_port)} is still running after auto-stop attempts. "
                "Stop the server (or choose another port) before scoring/logps."
            )
    wait_for_everyone(accelerator)

    # Scoring (SCE or ground-truth)
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
            # Ground-truth scoring: compare completions against dataset answers.
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

            # DAPO: overlong reward penalty — override scores for truncated completions.
            overlong_penalty = getattr(args, "overlong_penalty", None)
            if overlong_penalty is not None and completion_is_truncated is not None:
                n_penalized = 0
                for idx, is_trunc in enumerate(completion_is_truncated):
                    if is_trunc:
                        scores_all[idx] = float(overlong_penalty)
                        n_penalized += 1
                if n_penalized > 0:
                    logger.info(
                        f"Overlong penalty: {n_penalized}/{len(scores_all)} truncated completions set to {overlong_penalty}"
                    )
        else:
            # SCE scoring (original path).
            scorer_path = args.reward_model_path or args.policy_model_path
            if scorer_path == args.policy_model_path and bool(args.auto_start_server) and not bool(args.auto_stop_server):
                raise ValueError(
                    "reward_model_path is omitted (self-SCE), but auto_start_server is enabled and auto_stop_server is disabled. "
                    "Enable --auto_stop_server (or use --no-auto_start_server to connect to an external sleep-mode server)."
                )
            logger.info("Scoring completions with reward model (SCE)...")

            if accelerator is not None and torch.cuda.is_available():
                device_str = str(getattr(accelerator, "device", torch.device("cuda")))
            else:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
            reward_model = get_model(scorer_path, device=device_str)
            reward_tok = get_tokenizer(scorer_path)
            reward_prompts_text = render_prompts(sampled_raw, reward_tok)

            flat_prompts = flatten_prompts(reward_prompts_text, args.num_generations)

            n_total = len(flat_completions)
            start, end, per_rank = shard_range(n_total, accelerator)
            local_scores = score_responses(
                model=reward_model,
                tokenizer=reward_tok,
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
                gathered = accelerator.gather(padded)  # (world * per_rank,)
                scores_all = gathered[:n_total].detach().cpu().tolist()
            else:
                scores_all = list(local_scores)

        if is_main:
            cache.save_scores_a(args.step, scores_all)
            logger.info("Saved scores to cache (scores_a).")
        wait_for_everyone(accelerator)
        ok_s2, cached_scores2 = cache.load_scores_a(args.step)
        if ok_s2 and cached_scores2 is not None and len(cached_scores2) == len(flat_completions):
            sce_scores = cached_scores2

    # Logps (policy + optional ref)
    if args.compute_logps:
        # Single-model semantics:
        # - ref_logps is REQUIRED (even when num_iterations==1)
        # - old_logps is only required when num_iterations > 1
        need_old = int(getattr(args, "num_iterations", 1)) > 1
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
                f"need_ref={need_ref}, offline_logps_available={gen_output_ref is not None}"
            )

            ref_path = args.ref_model_path or args.policy_model_path
            ref_logps: Optional[List[List[float]]] = None
            old_logps: Optional[List[List[float]]] = None
            logits_to_keep_meta: Optional[int] = None

            # --- old_logps: prefer offline generation output (zero-cost) ---
            if need_old and use_offline and gen_output_ref is not None and gen_output_ref.old_log_probs:
                logger.info("Using old_logps from vLLM offline generation (no extra forward pass)")
                padded_logps, ltk = gen_output_ref.pad_old_logps()
                old_logps = padded_logps
                logits_to_keep_meta = ltk

            # Build batch for forward-pass logps (ref_logps, or old_logps fallback).
            flat_prompts_policy = flatten_prompts(prompts_text, args.num_generations)
            policy_tok = get_tokenizer(args.tokenizer_path or args.policy_model_path)
            pc_ids, pc_mask, cmw, logits_to_keep = build_left_padded_prompt_completion_batch(
                policy_tok,
                flat_prompts_policy,
                flat_completions,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                mask_truncated_completions=args.mask_truncated_completions,
                completion_is_truncated=completion_is_truncated,
                device=torch.device("cpu"),
            )
            pc_mask[:, -int(logits_to_keep):] = cmw
            completion_ids_all = pc_ids[:, -int(logits_to_keep) :].detach().cpu()
            if logits_to_keep_meta is None:
                logits_to_keep_meta = int(logits_to_keep)

            # Re-align vLLM offline old_logps to the re-tokenization width.
            # tokens→text→tokens is not identity; vLLM's original completion
            # lengths can differ from re-tokenized lengths.  Training always
            # re-tokenizes from text, so the cached logps width must match the
            # re-tokenization width to avoid a width mismatch in training.
            if old_logps is not None and logits_to_keep_meta != int(logits_to_keep):
                _target = int(logits_to_keep)
                _aligned: List[List[float]] = []
                for _row in old_logps:
                    if len(_row) >= _target:
                        _aligned.append(_row[:_target])
                    else:
                        _aligned.append(_row + [0.0] * (_target - len(_row)))
                logger.info(
                    f"Re-aligned vLLM old_logps width {logits_to_keep_meta} -> {_target} "
                    f"(re-tokenization width from build_left_padded_prompt_completion_batch)"
                )
                old_logps = _aligned
                logits_to_keep_meta = _target

            co_dump_alignment_samples(
                stage="logps",
                output_dir=str(args.output_dir),
                step=int(args.step),
                model_tag="policy",
                tokenizer=policy_tok,
                prompts=flat_prompts_policy,
                responses=flat_completions,
                finish_is_truncated=completion_is_truncated,
                prompt_completion_ids=pc_ids,
                completion_mask_window=cmw,
                logits_to_keep=int(logits_to_keep),
                max_items=co_int_env("CO_DEBUG_DATA_N", 3),
                is_main=is_main_process(accelerator),
            )

            if str(ref_path) == str(args.policy_model_path):
                # ref == policy: compute once and reuse for both ref_logps and old_logps.
                lp_t = _compute_per_token_logps_from_prebuilt_batch(
                    model_path=args.policy_model_path,
                    prompt_completion_ids=pc_ids,
                    attention_mask=pc_mask,
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
                # old_logps via forward pass (only if not already extracted from generation).
                if need_old and old_logps is None:
                    old_t = _compute_per_token_logps_from_prebuilt_batch(
                        model_path=args.policy_model_path,
                        prompt_completion_ids=pc_ids,
                        attention_mask=pc_mask,
                        logits_to_keep=int(logits_to_keep),
                        temperature=float(args.temperature),
                        batch_size=int(args.logps_batch_size),
                        accelerator=accelerator,
                    )
                    old_logps = old_t.tolist()

                logger.info(f"Computing ref_logps using reference model: {ref_path} ...")
                ref_tok = get_tokenizer(ref_path)
                ref_prompts_text = render_prompts(sampled_raw, ref_tok)

                flat_prompts_ref = flatten_prompts(ref_prompts_text, args.num_generations)
                pc_ids_ref, pc_mask_ref, cmw_ref, logits_to_keep_ref = build_left_padded_prompt_completion_batch(
                    ref_tok,
                    flat_prompts_ref,
                    flat_completions,
                    max_prompt_length=args.max_prompt_length,
                    max_completion_length=args.max_completion_length,
                    mask_truncated_completions=args.mask_truncated_completions,
                    completion_is_truncated=completion_is_truncated,
                    device=torch.device("cpu"),
                )
                pc_mask_ref[:, -int(logits_to_keep_ref):] = cmw_ref

                co_dump_alignment_samples(
                    stage="logps",
                    output_dir=str(args.output_dir),
                    step=int(args.step),
                    model_tag="ref",
                    tokenizer=ref_tok,
                    prompts=flat_prompts_ref,
                    responses=flat_completions,
                    finish_is_truncated=completion_is_truncated,
                    prompt_completion_ids=pc_ids_ref,
                    completion_mask_window=cmw_ref,
                    logits_to_keep=int(logits_to_keep_ref),
                    max_items=co_int_env("CO_DEBUG_DATA_N", 3),
                    is_main=is_main_process(accelerator),
                )

                ref_t = _compute_per_token_logps_from_prebuilt_batch(
                    model_path=ref_path,
                    prompt_completion_ids=pc_ids_ref,
                    attention_mask=pc_mask_ref,
                    logits_to_keep=int(logits_to_keep_ref),
                    temperature=float(args.temperature),
                    batch_size=int(args.logps_batch_size),
                    accelerator=accelerator,
                )
                ref_logps = ref_t.tolist()
                logits_to_keep_meta = int(logits_to_keep_ref)

            # Debug dump: verify completion ids vs logps lengths for a few samples.
            if is_main:
                co_dump_single_logps_samples(
                    base_dir=str(args.output_dir),
                    step=int(args.step),
                    prompts_preview=flatten_prompts(prompts_text, args.num_generations)[: int(os.environ.get("CO_DEBUG_DATA_N", "3"))],
                    responses_preview=flat_completions[: int(os.environ.get("CO_DEBUG_DATA_N", "3"))],
                    finish_preview=finish_reasons,
                    completion_ids=completion_ids_all,
                    old_lp=old_logps,
                    ref_lp=ref_logps,
                    max_items=int(os.environ.get("CO_DEBUG_DATA_N", "3")),
                )

                cache.save_logps_a(
                    args.step,
                    old_logps,
                    ref_logps,
                    alignment="completion_padded_v1",
                    logits_to_keep=logits_to_keep_meta,
                )
                logger.info(
                    f"Saved logps to cache (logps_a): old_logps={old_logps is not None}, ref_logps={ref_logps is not None}, alignment=completion_padded_v1"
                )
            wait_for_everyone(accelerator)

    logger.info("Done.")


if __name__ == "__main__":
    main()

