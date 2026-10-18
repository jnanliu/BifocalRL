# coding=utf-8
"""
Runtime invariants / assertions to keep tensor semantics aligned with open-r1-intuitor.

These checks are intentionally lightweight and are guarded by an env var so they can be
enabled during debugging without slowing down normal runs.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch


def enabled() -> bool:
    return os.environ.get("CO_ASSERT_INVARIANTS", "0") not in ("0", "false", "False", "")


def truthcheck_enabled() -> bool:
    return os.environ.get("CO_PIPELINE_TRUTHCHECK", "0") not in ("0", "false", "False", "")


def _tc_fail(*, where: str, msg: str) -> None:
    raise AssertionError(f"[TRUTHCHECK][{where}] {msg}")


def tc_check_generations_block(
    *,
    where: str,
    prompts: Sequence[Any],
    responses: Sequence[Any],
    num_generations: int,
    finish_reasons: Optional[Sequence[Any]] = None,
) -> None:
    if not truthcheck_enabled():
        return
    if int(num_generations) <= 0:
        _tc_fail(where=where, msg=f"num_generations must be > 0, got {num_generations}")
    n_prompts = len(prompts)
    expected = n_prompts * int(num_generations)
    if len(responses) != expected:
        _tc_fail(where=where, msg=f"responses length mismatch: got {len(responses)}, expected {expected}")
    if finish_reasons is not None and len(finish_reasons) != len(responses):
        _tc_fail(
            where=where,
            msg=f"finish_reasons length mismatch: got {len(finish_reasons)}, expected {len(responses)}",
        )


def tc_check_score_vector(
    *,
    where: str,
    scores: Sequence[Any],
    num_prompts: int,
    allowed_per_prompt: Sequence[int],
) -> None:
    if not truthcheck_enabled():
        return
    if int(num_prompts) <= 0:
        _tc_fail(where=where, msg=f"num_prompts must be > 0, got {num_prompts}")
    n = len(scores)
    allowed = [int(x) for x in allowed_per_prompt]
    if n % int(num_prompts) != 0:
        _tc_fail(where=where, msg=f"scores length {n} is not divisible by num_prompts {num_prompts}")
    per_prompt = n // int(num_prompts)
    if per_prompt not in allowed:
        _tc_fail(where=where, msg=f"scores_per_prompt={per_prompt}, allowed={allowed}")


def tc_check_logps_matrix(
    *,
    where: str,
    logps: Optional[Any],
    expected_rows: Optional[int] = None,
) -> None:
    if not truthcheck_enabled() or logps is None:
        return
    rows: Optional[int] = None
    cols: Optional[int] = None
    if isinstance(logps, torch.Tensor):
        if logps.dim() != 2:
            _tc_fail(where=where, msg=f"logps tensor must be 2D, got shape={tuple(logps.shape)}")
        rows, cols = int(logps.size(0)), int(logps.size(1))
    elif isinstance(logps, list):
        rows = len(logps)
        if rows > 0:
            first = logps[0]
            if not isinstance(first, list):
                _tc_fail(where=where, msg="logps list must be List[List[float]]")
            cols = len(first)
            for i, row in enumerate(logps):
                if not isinstance(row, list):
                    _tc_fail(where=where, msg=f"logps row {i} is not a list")
                if len(row) != cols:
                    _tc_fail(where=where, msg=f"logps rows have inconsistent widths at row {i}")
    else:
        _tc_fail(where=where, msg=f"unsupported logps type: {type(logps)}")

    if expected_rows is not None and rows != int(expected_rows):
        _tc_fail(where=where, msg=f"logps rows mismatch: got {rows}, expected {expected_rows}")
    if cols is not None and cols <= 0:
        _tc_fail(where=where, msg=f"logps width must be > 0, got {cols}")


def tc_check_training_batch(
    *,
    where: str,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor] = None,
    ref_per_token_logps: Optional[torch.Tensor] = None,
    require_old_logps: bool = False,
    require_ref_logps: bool = False,
) -> None:
    if not truthcheck_enabled():
        return

    if prompt_ids.dim() != 2 or completion_ids.dim() != 2:
        _tc_fail(where=where, msg=f"prompt/completion ids must be 2D, got {prompt_ids.shape}, {completion_ids.shape}")
    if prompt_mask.shape != prompt_ids.shape:
        _tc_fail(where=where, msg=f"prompt_mask shape mismatch: {prompt_mask.shape} vs {prompt_ids.shape}")
    if completion_mask.shape != completion_ids.shape:
        _tc_fail(where=where, msg=f"completion_mask shape mismatch: {completion_mask.shape} vs {completion_ids.shape}")
    if prompt_ids.size(0) != completion_ids.size(0):
        _tc_fail(where=where, msg=f"batch mismatch: prompt B={prompt_ids.size(0)} vs completion B={completion_ids.size(0)}")
    if advantages.dim() != 1 or int(advantages.size(0)) != int(completion_ids.size(0)):
        _tc_fail(where=where, msg=f"advantages shape mismatch: {advantages.shape}, expected ({completion_ids.size(0)},)")

    if require_old_logps and old_per_token_logps is None:
        _tc_fail(where=where, msg="old_per_token_logps is required but missing")
    if require_ref_logps and ref_per_token_logps is None:
        _tc_fail(where=where, msg="ref_per_token_logps is required but missing")

    for lp_name, lp in (("old_per_token_logps", old_per_token_logps), ("ref_per_token_logps", ref_per_token_logps)):
        if lp is None:
            continue
        if lp.dim() != 2:
            _tc_fail(where=where, msg=f"{lp_name} must be 2D, got {lp.shape}")
        if int(lp.size(0)) != int(completion_ids.size(0)):
            _tc_fail(where=where, msg=f"{lp_name} rows mismatch: {lp.size(0)} vs {completion_ids.size(0)}")


def tc_check_padding_semantics(
    *,
    where: str,
    prompt_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> None:
    if not truthcheck_enabled():
        return
    if prompt_mask.dim() != 2 or completion_mask.dim() != 2:
        _tc_fail(where=where, msg="prompt_mask/completion_mask must be 2D")

    # Prompt uses left-padding: valid segment is a suffix => mask is non-decreasing.
    if prompt_mask.size(1) > 1:
        pm = prompt_mask.to(torch.int64)
        bad_prompt = (pm[:, :-1] > pm[:, 1:]).any()
        if bool(bad_prompt.item()):
            _tc_fail(where=where, msg="prompt_mask is not left-padded (expects 0..01..1)")

    # Completion uses right-padding by EOS window => mask is non-increasing.
    if completion_mask.size(1) > 1:
        cm = completion_mask.to(torch.int64)
        bad_completion = (cm[:, :-1] < cm[:, 1:]).any()
        if bool(bad_completion.item()):
            _tc_fail(where=where, msg="completion_mask is not right-padded (expects 1..10..0)")


def tc_check_hacpo_mixed_batch(
    *,
    where: str,
    model_source: Optional[torch.Tensor],
    advantages: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    capability_weight: float,
    peer_epsilon_low: Optional[float],
    peer_epsilon_high: Optional[float],
) -> None:
    if not truthcheck_enabled() or model_source is None:
        return
    if model_source.dim() != 1:
        _tc_fail(where=where, msg=f"model_source must be 1D, got {model_source.shape}")
    if int(model_source.size(0)) != int(advantages.size(0)):
        _tc_fail(
            where=where,
            msg=f"model_source length mismatch: {model_source.size(0)} vs advantages {advantages.size(0)}",
        )
    vals = set(model_source.detach().to(torch.int64).cpu().tolist())
    if not vals.issubset({0, 1}):
        _tc_fail(where=where, msg=f"model_source has invalid values: {sorted(vals)}")

    has_peer = 1 in vals
    if has_peer:
        if capability_weight <= 0.0:
            _tc_fail(where=where, msg=f"capability_weight must be > 0 for peer samples, got {capability_weight}")
        if old_per_token_logps is None:
            _tc_fail(where=where, msg="peer samples require old_per_token_logps for HACPO-style IS")
        if peer_epsilon_low is None or peer_epsilon_high is None:
            _tc_fail(where=where, msg="peer clip bounds are missing for mixed HACPO batch")

def assert_prompt_completion_shapes(
    *,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    where: str,
) -> None:
    if not enabled():
        return

    if prompt_ids.dim() != 2 or completion_ids.dim() != 2:
        raise AssertionError(f"[{where}] prompt_ids/completion_ids must be 2D (got {prompt_ids.shape}, {completion_ids.shape})")
    if prompt_mask.shape != prompt_ids.shape:
        raise AssertionError(f"[{where}] prompt_mask shape {prompt_mask.shape} != prompt_ids shape {prompt_ids.shape}")
    if completion_mask.shape != completion_ids.shape:
        raise AssertionError(f"[{where}] completion_mask shape {completion_mask.shape} != completion_ids shape {completion_ids.shape}")
    if prompt_ids.size(0) != completion_ids.size(0):
        raise AssertionError(f"[{where}] batch mismatch: prompt_ids B={prompt_ids.size(0)} vs completion_ids B={completion_ids.size(0)}")

    # Masks should be 0/1-ish.
    for name, m in (("prompt_mask", prompt_mask), ("completion_mask", completion_mask)):
        if not torch.is_floating_point(m) and m.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            raise AssertionError(f"[{where}] {name} unexpected dtype {m.dtype}")
        if m.numel() > 0:
            mn = int(m.min().item())
            mx = int(m.max().item())
            if mn < 0 or mx > 1:
                raise AssertionError(f"[{where}] {name} expected in [0,1], got min={mn}, max={mx}")


def assert_attention_mask_matches_concat(
    *,
    prompt_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    where: str,
    allow_completion_padding_ones: bool = False,
) -> None:
    if not enabled():
        return

    if allow_completion_padding_ones:
        # Flash-Attn-safe semantics used by this repo:
        # - prompt_mask is left-padded (0...01...1)
        # - completion attention mask is all-ones (including completion padding tokens),
        #   while `completion_mask` is used to ignore padding in loss/KL aggregation.
        expected = torch.cat([prompt_mask, torch.ones_like(completion_mask)], dim=1)
    else:
        # Legacy semantics: attention_mask equals the concatenation of prompt_mask and completion_mask.
        expected = torch.cat([prompt_mask, completion_mask], dim=1)
    if attention_mask.shape != expected.shape:
        raise AssertionError(f"[{where}] attention_mask shape {attention_mask.shape} != expected {expected.shape}")
    # Avoid heavy allclose; exact equality is expected for integer masks.
    if not torch.equal(attention_mask.cpu(), expected.cpu()):
        raise AssertionError(f"[{where}] attention_mask != cat(prompt_mask, completion_mask)")

