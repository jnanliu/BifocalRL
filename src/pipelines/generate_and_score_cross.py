# coding=utf-8
"""
Cross-model Generate and Score Pipeline.

A fixed *generator* model produces completions via vLLM.
A *trainee* model scores each (prompt, completion) via self-certainty (SCE)
and computes per-token logps for GRPO training.

Two chat templates are used independently:
- Generation prompts use the GENERATOR's chat template.
- Scoring/logps/training prompts use the TRAINEE's chat template.
- Cache stores trainee-template prompts for single_intuitor_train.py compatibility.

Cache format: reuses `utils.cache.CoIntuitorCache` with the "_a" slots (same as single-model).
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Any, Dict

import torch

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.cache import CoIntuitorCache
from utils.dataset import load_co_intuitor_dataset
from utils.generate import (
    generate_responses,
    strip_prompt_echo_from_flattened_responses,
    strip_chat_template_artifacts_from_flattened_responses,
)
from utils.reward import score_responses
from utils.tokenization import build_left_padded_prompt_completion_batch
from utils.env_utils import co_int_env
from utils.prompt_utils import render_prompt, render_prompts, flatten_prompts
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
from utils.logps import compute_per_token_logps_from_prebuilt_batch as _compute_per_token_logps_from_prebuilt_batch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model generate and score: generator produces completions, trainee scores + computes logps"
    )

    # Cross-model paths
    parser.add_argument("--generator_model_path", type=str, required=True,
                        help="Generator model path (served by vLLM for producing completions)")
    parser.add_argument("--trainee_model_path", type=str, required=True,
                        help="Trainee model path (in-process scoring, logps, and training)")
    parser.add_argument("--generator_tokenizer_path", type=str, default=None,
                        help="Tokenizer for generator chat template (defaults to generator_model_path)")
    parser.add_argument("--trainee_tokenizer_path", type=str, default=None,
                        help="Tokenizer for trainee chat template (defaults to trainee_model_path)")

    # Dataset / sampling
    parser.add_argument("--dataset_key", type=str, required=True, help="Dataset key from recipes/datasets.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cache")
    parser.add_argument("--step", type=int, default=0, help="Step number for cache")
    parser.add_argument("--problems_per_step", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--shuffle_dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset_sampling_state", action="store_true", default=False)

    # Generation params
    parser.add_argument("--num_generations", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=2048)

    # Scoring / logps params
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--scoring_batch_size", type=int, default=8)
    parser.add_argument("--mask_truncated_completions", action="store_true", default=False)
    parser.add_argument("--compute_logps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logps_batch_size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient; if >0, compute ref_logps")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Reference model path for KL (defaults to trainee_model_path)")
    parser.add_argument("--num_iterations", type=int, default=1)

    # vLLM server settings (for generator)
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port for generator")
    parser.add_argument("--auto_start_server", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto_stop_server", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    logger.setLevel(logging.INFO)

    accelerator = get_accelerator()
    is_main = is_main_process(accelerator)
    if not is_main:
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

    # Dist info line (all ranks).
    try:
        world = int(getattr(accelerator, "num_processes", 1)) if accelerator is not None else int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(getattr(accelerator, "process_index", 0)) if accelerator is not None else int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        dev = str(getattr(accelerator, "device", "cuda" if torch.cuda.is_available() else "cpu")) if accelerator is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[dist] world_size={world} rank={rank} local_rank={local_rank} device={dev} CUDA_VISIBLE_DEVICES={cvd}", flush=True)
    except Exception:
        pass

    # =========================================================================
    # Load tokenizers for both models
    # =========================================================================
    gen_tok_path = args.generator_tokenizer_path or args.generator_model_path
    train_tok_path = args.trainee_tokenizer_path or args.trainee_model_path
    gen_tokenizer = get_tokenizer(gen_tok_path)
    train_tokenizer = get_tokenizer(train_tok_path)

    has_gen_template = bool(getattr(gen_tokenizer, "chat_template", None))
    has_train_template = bool(getattr(train_tokenizer, "chat_template", None))

    if is_main:
        logger.info(f"Generator: {args.generator_model_path} (template={'yes' if has_gen_template else 'no'})")
        logger.info(f"Trainee:   {args.trainee_model_path} (template={'yes' if has_train_template else 'no'})")

    # =========================================================================
    # Load dataset & sample problems
    # =========================================================================
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
        if is_main:
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

    # =========================================================================
    # Render prompts with BOTH templates
    # =========================================================================
    prompts_gen: List[str] = []    # Generator template (for vLLM generation)
    prompts_train: List[str] = []  # Trainee template (for scoring/logps/cache)
    for problem in sampled_raw:
        prompts_gen.append(render_prompt(problem, gen_tokenizer, has_gen_template))
        prompts_train.append(render_prompt(problem, train_tokenizer, has_train_template))

    if is_main:
        logger.info(f"Rendered prompts: gen={len(prompts_gen)}, train={len(prompts_train)}")
        if prompts_gen:
            logger.info(f"  Generator prompt[0] preview: {prompts_gen[0][:200]!r}")
            logger.info(f"  Trainee  prompt[0] preview: {prompts_train[0][:200]!r}")

        # Sanity-check: prompts must not look like stringified Python lists.
        for tag, plist in [("gen", prompts_gen), ("train", prompts_train)]:
            if plist and plist[0].lstrip().startswith("[{"):
                logger.error(
                    f"Prompt rendering bug: {tag} prompt[0] looks like a raw Python list "
                    f"(maybe_apply_chat_template did not apply the template). Preview: {plist[0][:120]!r}"
                )
                raise RuntimeError(
                    f"Prompt rendering failed for {tag}: output looks like a stringified message list, "
                    f"not a chat-template-rendered string. Check tokenizer chat_template availability."
                )

        # Verify generator vs trainee prompts differ when templates differ.
        if has_gen_template and has_train_template and prompts_gen and prompts_train:
            if gen_tok_path != train_tok_path and prompts_gen[0] == prompts_train[0]:
                logger.warning(
                    "Generator and trainee tokenizer paths differ but rendered prompts are identical. "
                    "This is expected only if both models share the same chat_template."
                )

    # =========================================================================
    # Step 1: Generate completions via generator vLLM
    # =========================================================================
    expected = len(prompts_gen) * int(args.num_generations)
    finish_reasons: Optional[List[Optional[str]]] = None
    flat_completions: List[str] = []

    if is_main:
        logger.info("=" * 80)
        logger.info("Step 1: Generate completions via GENERATOR model")
        logger.info("=" * 80)

        cache_valid = False
        if cache.check_generations_a(args.step, args.num_generations):
            ok, cached_resps, cached_nprompts, cached_prompts, cached_finish = cache.load_generations_a(
                args.step, args.num_generations
            )
            if (
                ok
                and cached_resps is not None
                and cached_nprompts == len(prompts_train)
                and len(cached_resps) == expected
            ):
                # Validate cache model identity to prevent stale data from a previous run.
                cached_gen_model = cache.get_metadata("generator_model_path")
                cached_trainee_model = cache.get_metadata("trainee_model_path")
                if cached_gen_model and str(cached_gen_model) != str(args.generator_model_path):
                    logger.warning(
                        f"Cache model mismatch! Cached generator={cached_gen_model}, "
                        f"current generator={args.generator_model_path}. Invalidating cache."
                    )
                elif cached_trainee_model and str(cached_trainee_model) != str(args.trainee_model_path):
                    logger.warning(
                        f"Cache model mismatch! Cached trainee={cached_trainee_model}, "
                        f"current trainee={args.trainee_model_path}. Invalidating cache."
                    )
                else:
                    flat_completions = cached_resps
                    if cached_prompts is not None:
                        prompts_train = cached_prompts
                    finish_reasons = cached_finish
                    if finish_reasons is not None and len(finish_reasons) != len(flat_completions):
                        finish_reasons = [None] * len(flat_completions)
                    cache_valid = True
                    logger.info(f"Cache hit: generations_a step={args.step} ({len(flat_completions)} completions)")
                    if not bool(getattr(args, "auto_start_server", False)):
                        sleep_vllm_server_if_possible(int(args.vllm_port), is_main=True)

        if not cache_valid:
            flat_completions = []

        if not flat_completions:
            logger.info(f"Generating {len(prompts_gen)} prompts x {args.num_generations} completions (generator)...")
            gen_out = generate_responses(
                model_path=args.generator_model_path,
                prompts=prompts_gen,
                num_generations=args.num_generations,
                batch_size=128,
                port=args.vllm_port,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_completion_length,
                auto_start_server=args.auto_start_server,
                auto_stop_server=False,
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

            # Cache stores TRAINEE-template prompts for training compatibility.
            cache.save_generations_a(
                args.step, len(prompts_train), flat_completions,
                prompts=prompts_train, finish_reasons=finish_reasons,
            )
            # Store model identity metadata so future runs can detect stale caches.
            cache.save_metadata("generator_model_path", str(args.generator_model_path))
            cache.save_metadata("trainee_model_path", str(args.trainee_model_path))
            logger.info("Saved generations to cache (generations_a, with trainee-template prompts).")

            # Sleep the generator vLLM server to free VRAM for scoring/logps.
            sleep_vllm_server_if_possible(int(args.vllm_port), is_main=True)

    # Barrier: all ranks load cached data.
    wait_for_everyone(accelerator)
    ok_g, cached_resps, cached_nprompts, cached_prompts, cached_finish = cache.load_generations_a(
        args.step, args.num_generations
    )
    if not ok_g or cached_resps is None or cached_prompts is None or cached_nprompts != len(prompts_train):
        raise RuntimeError("Failed to load generations_a from cache after generation barrier.")
    flat_completions = list(cached_resps)
    prompts_train = list(cached_prompts)
    finish_reasons = cached_finish
    if finish_reasons is not None and len(finish_reasons) != len(flat_completions):
        finish_reasons = [None] * len(flat_completions)

    # Sanitize completions.
    # Use prompts_gen for stripping because completions were generated from generator prompts.
    # If vLLM echoed the prompt, the echo is in generator format, not trainee format.
    flat_completions, _c1 = strip_prompt_echo_from_flattened_responses(
        prompts_gen, flat_completions, num_generations=args.num_generations
    )
    flat_completions, _c2 = strip_chat_template_artifacts_from_flattened_responses(
        prompts_gen, flat_completions, num_generations=args.num_generations
    )
    changed = bool(_c1 or _c2)
    if changed and is_main:
        cache.save_generations_a(
            args.step, len(prompts_train), flat_completions,
            prompts=prompts_train, finish_reasons=finish_reasons,
        )
    wait_for_everyone(accelerator)
    ok_g2, cached_resps2, _, cached_prompts2, cached_finish2 = cache.load_generations_a(args.step, args.num_generations)
    if ok_g2 and cached_resps2 is not None and cached_prompts2 is not None:
        flat_completions = list(cached_resps2)
        prompts_train = list(cached_prompts2)
        finish_reasons = cached_finish2

    completion_is_truncated: Optional[List[Optional[bool]]] = None
    if finish_reasons is not None and len(finish_reasons) == len(flat_completions):
        completion_is_truncated = [str(x) == "length" for x in finish_reasons]

    # Stop/sleep generator vLLM before in-process scoring/logps.
    if bool(getattr(args, "auto_stop_server", False)):
        stopped = stop_vllm_server_if_possible(int(args.vllm_port), is_main=is_main)
        if is_main and not stopped:
            raise RuntimeError(
                f"vLLM server on port {int(args.vllm_port)} is still running after auto-stop attempts. "
                "Stop the server (or choose another port) before scoring/logps."
            )
    wait_for_everyone(accelerator)

    # =========================================================================
    # Step 2: Score with TRAINEE model (SCE)
    # =========================================================================
    if is_main:
        logger.info("=" * 80)
        logger.info("Step 2: Score completions with TRAINEE model (SCE)")
        logger.info("=" * 80)

    if cache.check_scores_a(args.step):
        ok, cached_scores = cache.load_scores_a(args.step)
        if ok and cached_scores is not None and len(cached_scores) == len(flat_completions):
            sce_scores = cached_scores
            if is_main:
                logger.info(f"Cache hit: scores_a step={args.step} ({len(sce_scores)} scores)")
        else:
            sce_scores = []
    else:
        sce_scores = []

    if not sce_scores:
        if is_main:
            logger.info("Scoring completions with trainee model (SCE)...")

        if accelerator is not None and torch.cuda.is_available():
            device_str = str(getattr(accelerator, "device", torch.device("cuda")))
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        reward_model = get_model(args.trainee_model_path, device=device_str)
        reward_tok = train_tokenizer

        flat_prompts = flatten_prompts(prompts_train, args.num_generations)

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

        # Gather scores across ranks.
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

        if is_main:
            cache.save_scores_a(args.step, scores_all)
            logger.info("Saved scores to cache (scores_a).")
        wait_for_everyone(accelerator)
        ok_s2, cached_scores2 = cache.load_scores_a(args.step)
        if ok_s2 and cached_scores2 is not None and len(cached_scores2) == len(flat_completions):
            sce_scores = cached_scores2

        del reward_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: Compute logps with TRAINEE model
    # =========================================================================
    if args.compute_logps:
        if is_main:
            logger.info("=" * 80)
            logger.info("Step 3: Compute logps with TRAINEE model")
            logger.info("=" * 80)

        need_old = int(getattr(args, "num_iterations", 1)) > 1
        need_ref = True

        ok_lp, old_lp, ref_lp = cache.load_logps_a(args.step)

        cache_has_old = ok_lp and old_lp is not None and len(old_lp) == len(flat_completions)
        cache_has_ref = ok_lp and ref_lp is not None and len(ref_lp) == len(flat_completions)
        cache_ok = ((not need_old) or cache_has_old) and ((not need_ref) or cache_has_ref)

        if cache_ok:
            if is_main:
                logger.info(f"Cache hit: logps_a step={args.step}")
        else:
            if is_main:
                logger.info(
                    f"Computing logps (trainee): need_old={need_old} (num_iterations={getattr(args, 'num_iterations', 1)}), need_ref={need_ref}"
                )

            # Build batch with trainee tokenizer (prompts_train).
            flat_prompts_trainee = flatten_prompts(prompts_train, args.num_generations)
            pc_ids, pc_mask, cmw, logits_to_keep = build_left_padded_prompt_completion_batch(
                train_tokenizer,
                flat_prompts_trainee,
                flat_completions,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                mask_truncated_completions=args.mask_truncated_completions,
                completion_is_truncated=completion_is_truncated,
                device=torch.device("cpu"),
            )
            pc_mask[:, -int(logits_to_keep):] = cmw
            completion_ids_all = pc_ids[:, -int(logits_to_keep):].detach().cpu()

            co_dump_alignment_samples(
                stage="logps",
                output_dir=str(args.output_dir),
                step=int(args.step),
                model_tag="trainee",
                tokenizer=train_tokenizer,
                prompts=flat_prompts_trainee,
                responses=flat_completions,
                finish_is_truncated=completion_is_truncated,
                prompt_completion_ids=pc_ids,
                completion_mask_window=cmw,
                logits_to_keep=int(logits_to_keep),
                max_items=co_int_env("CO_DEBUG_DATA_N", 3),
                is_main=is_main_process(accelerator),
            )

            ref_path = args.ref_model_path or args.trainee_model_path
            ref_logps: Optional[List[List[float]]] = None
            old_logps: Optional[List[List[float]]] = None
            logits_to_keep_meta: Optional[int] = int(logits_to_keep)

            if str(ref_path) == str(args.trainee_model_path):
                # Trainee == ref model: compute once, reuse.
                lp_t = _compute_per_token_logps_from_prebuilt_batch(
                    model_path=args.trainee_model_path,
                    prompt_completion_ids=pc_ids,
                    attention_mask=pc_mask,
                    logits_to_keep=int(logits_to_keep),
                    temperature=float(args.temperature),
                    batch_size=int(args.logps_batch_size),
                    accelerator=accelerator,
                )
                lp = lp_t.tolist()
                ref_logps = lp
                if need_old:
                    old_logps = lp
            else:
                # old_logps from current trainee.
                if need_old:
                    old_t = _compute_per_token_logps_from_prebuilt_batch(
                        model_path=args.trainee_model_path,
                        prompt_completion_ids=pc_ids,
                        attention_mask=pc_mask,
                        logits_to_keep=int(logits_to_keep),
                        temperature=float(args.temperature),
                        batch_size=int(args.logps_batch_size),
                        accelerator=accelerator,
                    )
                    old_logps = old_t.tolist()

                # ref_logps from initial trainee (ref model) with its own template.
                if is_main:
                    logger.info(f"Computing ref_logps using reference model: {ref_path} ...")
                ref_tok = get_tokenizer(ref_path)
                has_ref_template = bool(getattr(ref_tok, "chat_template", None))
                ref_prompts_text = render_prompts(sampled_raw, ref_tok, has_template=has_ref_template)

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

            if is_main:
                co_dump_single_logps_samples(
                    base_dir=str(args.output_dir),
                    step=int(args.step),
                    prompts_preview=flatten_prompts(prompts_train, args.num_generations)[: co_int_env("CO_DEBUG_DATA_N", 3)],
                    responses_preview=flat_completions[: co_int_env("CO_DEBUG_DATA_N", 3)],
                    finish_preview=finish_reasons,
                    completion_ids=completion_ids_all,
                    old_lp=old_logps,
                    ref_lp=ref_logps,
                    max_items=co_int_env("CO_DEBUG_DATA_N", 3),
                )

                cache.save_logps_a(
                    args.step,
                    old_logps,
                    ref_logps,
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
