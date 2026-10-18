# coding=utf-8
"""Shared scoring and log-probability helpers for co-learning pipelines.

Extracted from ``src/generate_and_score.py`` and ``src/score_and_compute.py``
to eliminate duplication.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from utils.distributed_utils import (
    get_accelerator,
    is_main_process,
    shard_range,
)
from utils.env_utils import co_int_env
from utils.model_utils import get_model, get_tokenizer
from utils.logps import get_per_token_logps
from utils.prompt_utils import render_prompts
from utils.reward import get_per_token_self_certainty, get_advantage_from_sce
from utils.tokenization import build_left_padded_prompt_completion_batch

logger = logging.getLogger(__name__)


def score_responses_for_model(
    model_path: str,
    tokenizer_path: str,
    unique_prompt_examples: List[dict],
    all_text_responses_a: List[str],
    all_text_responses_b: List[str],
    all_finish_reasons_a: Optional[List[Optional[str]]] = None,
    all_finish_reasons_b: Optional[List[Optional[str]]] = None,
    num_generations_per_model: int = 4,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    mask_truncated_completions: bool = False,
    batch_size: int = 8,
    device: str = "cuda",
    accelerator=None,
    debug_zero_advantages_path: Optional[str] = None,
    debug_zero_advantages_max: int = 30,
    *,
    score_only: Optional[str] = None,
    debug_output_dir: Optional[str] = None,
    debug_step: Optional[int] = None,
    debug_model_tag: Optional[str] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """Score responses using self-certainty for a model.

    Args:
        model_path: Path to model.
        tokenizer_path: Path to tokenizer.
        unique_prompt_examples: Raw dataset dicts (length = num_unique).
        all_text_responses_a: Flattened responses from Model A.
        all_text_responses_b: Flattened responses from Model B.
        num_generations_per_model: Generations per model per prompt.
        max_prompt_length: Maximum prompt length.
        max_completion_length: Maximum completion length.
        mask_truncated_completions: Mask truncated completions.
        batch_size: Batch size for scoring.
        device: Device to use.
        accelerator: Optional Accelerate accelerator.
        debug_zero_advantages_path: Path to dump zero-advantage debug info.
        debug_zero_advantages_max: Max items in the debug dump.
        score_only: ``"a"`` / ``"b"`` / ``None`` (both).
        debug_output_dir: Debug output directory.
        debug_step: Debug step number.
        debug_model_tag: Debug model tag.
        apply_chat_template_kwargs: Extra kwargs for ``apply_chat_template``.

    Returns:
        Advantages tensor.  Shape ``(num_unique, G)`` or ``(num_unique, 2G)``.
    """
    logger.info("Scoring responses using model: %s", model_path)

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
        logger.info("Scoring tokenizer has no chat_template — using raw prompt text directly.")
    rp_kwargs: dict = dict(has_template=has_template)
    if apply_chat_template_kwargs is not None:
        rp_kwargs["apply_chat_template_kwargs"] = apply_chat_template_kwargs
    unique_prompts = render_prompts(unique_prompt_examples, tokenizer, **rp_kwargs)

    num_unique_prompts = len(unique_prompts)

    expected_a = num_unique_prompts * num_generations_per_model
    expected_b = num_unique_prompts * num_generations_per_model
    if len(all_text_responses_a) != expected_a:
        raise ValueError(f"Expected {expected_a} responses from Model A, got {len(all_text_responses_a)}")
    if len(all_text_responses_b) != expected_b:
        raise ValueError(f"Expected {expected_b} responses from Model B, got {len(all_text_responses_b)}")

    prompts_for_scoring: List[str] = []
    responses_for_scoring: List[str] = []
    trunc_for_scoring: Optional[List[Optional[bool]]] = (
        [] if (all_finish_reasons_a is not None or all_finish_reasons_b is not None) else None
    )

    for i in range(num_unique_prompts):
        prompt = unique_prompts[i]
        sa = i * num_generations_per_model
        ea = (i + 1) * num_generations_per_model

        responses_a = all_text_responses_a[sa:ea]
        responses_b = all_text_responses_b[sa:ea]

        if score_only == "a":
            combined_responses = list(responses_a)
        elif score_only == "b":
            combined_responses = list(responses_b)
        else:
            combined_responses = list(responses_a) + list(responses_b)

        if trunc_for_scoring is not None:
            fr_a = all_finish_reasons_a[sa:ea] if all_finish_reasons_a is not None else [None] * len(responses_a)
            fr_b = all_finish_reasons_b[sa:ea] if all_finish_reasons_b is not None else [None] * len(responses_b)
            if score_only == "a":
                combined_fr = list(fr_a)
            elif score_only == "b":
                combined_fr = list(fr_b)
            else:
                combined_fr = list(fr_a) + list(fr_b)
            trunc_for_scoring.extend([None if x is None else (str(x) == "length") for x in combined_fr])

        prompts_for_scoring.extend([prompt] * len(combined_responses))
        responses_for_scoring.extend(combined_responses)

    build_device = torch.device("cpu")
    prompt_completion_ids, attention_mask, completion_mask_window, logits_to_keep = (
        build_left_padded_prompt_completion_batch(
            tokenizer, prompts_for_scoring, responses_for_scoring,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            mask_truncated_completions=mask_truncated_completions,
            completion_is_truncated=trunc_for_scoring,
            device=build_device,
        )
    )

    if debug_output_dir is not None and debug_step is not None and debug_model_tag is not None:
        from utils.debug_utils import co_dump_alignment_samples
        co_dump_alignment_samples(
            stage="score", output_dir=str(debug_output_dir), step=int(debug_step),
            model_tag=str(debug_model_tag), tokenizer=tokenizer,
            prompts=prompts_for_scoring, responses=responses_for_scoring,
            finish_is_truncated=trunc_for_scoring,
            prompt_completion_ids=prompt_completion_ids,
            completion_mask_window=completion_mask_window,
            logits_to_keep=int(logits_to_keep),
            max_items=co_int_env("CO_DEBUG_DATA_N", 3),
            is_main=is_main_process(accelerator),
        )

    all_advantages = []
    n_total = int(prompt_completion_ids.size(0))
    start, end, per_rank = shard_range(n_total, accelerator)
    is_main = is_main_process(accelerator)

    ids_local = prompt_completion_ids[start:end]
    mask_local = attention_mask[start:end]
    cmask_local = completion_mask_window[start:end]

    num_scoring_batches = (ids_local.size(0) + batch_size - 1) // batch_size
    with torch.no_grad():
        iterator = range(0, ids_local.size(0), batch_size)
        iterator = tqdm(iterator, desc="Scoring responses", unit="batch",
                        total=num_scoring_batches, disable=not is_main)
        for i in iterator:
            end_idx = min(i + batch_size, ids_local.size(0))
            batch_input_ids = ids_local[i:end_idx].to(torch_device, non_blocking=True)
            batch_attention_mask = mask_local[i:end_idx].to(torch_device, non_blocking=True)
            batch_completion_mask = cmask_local[i:end_idx].to(torch_device, non_blocking=True)
            sce_per_token = get_per_token_self_certainty(
                model, batch_input_ids, batch_attention_mask, logits_to_keep, batch_size=batch_size
            )
            advantages = get_advantage_from_sce(sce_per_token, batch_completion_mask)
            all_advantages.append(advantages.detach().cpu())

    if all_advantages:
        advantages_local = torch.cat(all_advantages, dim=0)
    else:
        advantages_local = torch.empty((0,), dtype=torch.float32)

    if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
        padded = torch.zeros((per_rank,), dtype=advantages_local.dtype)
        padded[: advantages_local.numel()] = advantages_local
        padded = padded.to(torch_device, non_blocking=True)
        gathered = accelerator.gather(padded)
        advantages_all = gathered[:n_total].detach().cpu()
    else:
        advantages_all = advantages_local

    gens_per_prompt = num_generations_per_model if score_only else 2 * num_generations_per_model
    advantages_reshaped = advantages_all.view(num_unique_prompts, gens_per_prompt)
    logger.info("Scored %d responses (score_only=%r), reshaped to %s",
                len(advantages_all), score_only, advantages_reshaped.shape)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return advantages_reshaped
