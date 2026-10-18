# coding=utf-8
"""
Tokenization helpers to keep prompt/completion semantics aligned with open-r1-intuitor.

Key point:
- open-r1 uses completion token IDs returned by generation (vLLM / generate), which include EOS when terminated.
- In this repo we often reconstruct completions from *text* with `add_special_tokens=False`, which does NOT append EOS.

These helpers allow us to explicitly inject a single EOS token at the end of each completion sequence so that
`completion_mask` computed from EOS matches open-r1 semantics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def append_eos_1d(ids: torch.Tensor, *, eos_token_id: int) -> torch.Tensor:
    """Append EOS to a 1D token id tensor."""
    if ids.dim() != 1:
        raise ValueError("ids must be 1D")
    return torch.cat([ids, ids.new_tensor([int(eos_token_id)], dtype=ids.dtype)], dim=0)


def pad_sequence_to_length(
    tensors: torch.Tensor,
    max_seq_len: int,
    pad_token_id: int,
    left_pad: bool = False,
) -> torch.Tensor:
    """Pad a 2D tensor in the last dim to ``max_seq_len`` (HACRL-compatible helper)."""
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad: bool = True,
    truncation: str = "error",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process tokenizer outputs to consistent shapes via padding/truncation.

    This is an in-repo adaptation of HACRL's ``verl.utils.torch_functional.postprocess_data``
    so we can reuse the same function stack without cross-project imports.
    """
    if truncation not in ("left", "right", "middle", "error"):
        raise ValueError(f"Unknown truncation method {truncation!r}")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be 2D")

    sequence_length = int(input_ids.shape[-1])
    if sequence_length < int(max_length):
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=int(max_length), pad_token_id=int(pad_token_id), left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=int(max_length), pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > int(max_length):
        if truncation == "left":
            input_ids = input_ids[:, -int(max_length):]
            attention_mask = attention_mask[:, -int(max_length):]
        elif truncation == "right":
            input_ids = input_ids[:, :int(max_length)]
            attention_mask = attention_mask[:, :int(max_length)]
        elif truncation == "middle":
            left_half = int(max_length) // 2
            right_half = int(max_length) - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
        else:
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")

    return input_ids, attention_mask


def tokenize_and_postprocess_data(
    prompt: str,
    tokenizer,
    max_length: int,
    pad_token_id: int,
    left_pad: bool = True,
    truncation: str = "error",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HACRL-style tokenize + postprocess helper (copied/adapted locally)."""
    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    return postprocess_data(input_ids, attention_mask, max_length, pad_token_id, left_pad, truncation)


def get_response_mask(
    response_id: torch.Tensor,
    eos_token: int | List[int],
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """
    HACRL-compatible EOS-inclusive response mask.

    For each sequence, tokens up to and including the first EOS are 1; tokens after EOS are 0.
    If EOS is absent, all positions are 1.
    """
    eos_values = eos_token if isinstance(eos_token, list) else [int(eos_token)]
    eos_t = torch.tensor(eos_values, device=response_id.device)
    eos_mask = torch.isin(response_id, eos_t).int()
    return (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)


def build_left_padded_prompt_completion_batch(
    tokenizer,
    prompts: List[str],
    completions: List[str],
    *,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    mask_truncated_completions: bool = False,
    completion_is_truncated: Optional[List[Optional[bool]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build a **Flash-Attn safe** batched `(input_ids, attention_mask)` for causal LMs by ensuring
    the resulting `attention_mask` is **left-padded only** (i.e., for each row it's `0...01...1`).

    This is important for models (e.g. Qwen2 with Flash Attention) that error on batched inputs
    with right-padding.

    Semantics are aligned with open-r1-intuitor's GRPO pipeline:
    - Prompts and completions are tokenized with `add_special_tokens=False`.
    - `max_prompt_length` truncates prompts on the left (keep last tokens).
    - If `max_completion_length` is provided, we treat `len(completion_ids) >= max_completion_length`
      as "possibly truncated" (no guaranteed EOS), and only append a single EOS when
      `len(completion_ids) < max_completion_length` and it does not already end with EOS.
      This approximates vLLM/open-r1 behavior where completions include EOS when they terminate
      naturally, but may omit EOS when they hit the length cap.
    - We pad each completion to a shared length `logits_to_keep = max(effective_completion_len)` (at least 1),
      then concatenate `prompt_ids + padded_completion_ids`.
      This ensures that the last `logits_to_keep` tokens of the concatenated sequence correspond **exactly**
      to the completion token IDs (plus completion padding), matching open-r1 semantics
      (`logits_to_keep == completion_ids.size(1)` at the batch level).

      IMPORTANT (Flash-Attn safety):
      We set the model `attention_mask` to 1 for completion padding tokens as well, so the final attention mask
      is pure left-padding (`0...01...1`) with no right-padding holes. This is safe for causal LMs because the
      completion padding tokens occur *after* real tokens and therefore do not affect logits of real tokens.
      Loss/scoring uses `completion_mask_window` to ignore padded positions.

    Note on `mask_truncated_completions`:
    - Legacy open-r1 style optionally *drops* truncated completions by setting their completion mask to 0.
    - In this repo we intentionally DO NOT drop truncated completions (to keep training signal),
      so `mask_truncated_completions` is accepted for backward compatibility but has no effect here.

    Returns:
      input_ids: (B, L) left-padded ids
      attention_mask: (B, L) left-padded-only mask (0...01...1)
      completion_mask_window: (B, logits_to_keep) mask over completion positions (left-aligned)
      logits_to_keep: int (suffix window length)
    """
    if len(prompts) != len(completions):
        raise ValueError(f"prompts/completions length mismatch: {len(prompts)} vs {len(completions)}")
    if completion_is_truncated is not None and len(completion_is_truncated) != len(completions):
        raise ValueError(
            f"completion_is_truncated length mismatch: {len(completion_is_truncated)} vs {len(completions)}"
        )

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must have eos_token_id (open-r1 semantics).")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = int(eos_id)

    # Tokenize without padding so we can control the final padding strategy.
    prompt_enc = tokenizer(
        text=prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    comp_enc = tokenizer(
        text=completions,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    prompt_ids_list: List[List[int]] = [list(x) for x in prompt_enc["input_ids"]]
    comp_ids_raw_list: List[List[int]] = [list(x) for x in comp_enc["input_ids"]]

    if max_completion_length is not None:
        mcl = int(max_completion_length)
        if mcl > 0:
            comp_ids_raw_list = [c[:mcl] if len(c) > mcl else c for c in comp_ids_raw_list]

    if max_prompt_length is not None:
        mpl = int(max_prompt_length)
        prompt_ids_list = [p[-mpl:] if len(p) > mpl else p for p in prompt_ids_list]

    effective_comp_ids_list: List[List[int]] = []
    effective_comp_lens: List[int] = []
    for i, ids in enumerate(comp_ids_raw_list):
        ids2 = list(ids)

        # Decide truncation using generation metadata when available.
        #
        # IMPORTANT:
        # When scoring/logps are computed with a tokenizer different from the generation tokenizer
        # (co-learning A scores B generations and vice versa), inferring truncation from token length
        # is unreliable and can incorrectly zero out completion masks. If the caller provides
        # `completion_is_truncated`, we trust it (it should be based on vLLM finish_reason).
        truncated = completion_is_truncated[i] if completion_is_truncated is not None else None

        # Append EOS for non-truncated completions so completion_mask is non-empty and matches open-r1 semantics.
        if len(ids2) > 0 and int(ids2[-1]) == int(eos_id):
            pass
        else:
            if truncated is False:
                ids2.append(int(eos_id))
            elif truncated is True:
                # Do not append EOS; this completion hit the length cap.
                pass
            else:
                # Fallback heuristic when truncation metadata is unavailable:
                # if shorter than the configured cap, assume it terminated and append EOS.
                if max_completion_length is None:
                    ids2.append(int(eos_id))
                else:
                    mcl = int(max_completion_length)
                    if mcl > 0 and len(ids2) < mcl:
                        ids2.append(int(eos_id))

        # Ensure the final completion ids never exceed max_completion_length.
        # This matches open-r1 semantics where completions (including EOS when present) are bounded by max_tokens.
        if max_completion_length is not None:
            mcl2 = int(max_completion_length)
            if mcl2 > 0 and len(ids2) > mcl2:
                ids2 = ids2[:mcl2]

        # Mask to the first EOS (inclusive), if any.
        has_eos = int(eos_id) in ids2
        # Keep truncated completions (do not zero them out).
        if has_eos:
            eos_pos = ids2.index(int(eos_id))
            eff = ids2[: eos_pos + 1]
        else:
            eff = ids2

        effective_comp_ids_list.append(eff)
        effective_comp_lens.append(len(eff))

    logits_to_keep = max(effective_comp_lens, default=0)
    if logits_to_keep <= 0:
        logits_to_keep = 1

    # Pad completions to `logits_to_keep` (right padding) so that:
    # - the last `logits_to_keep` tokens of the full prompt+completion sequence are always the completion IDs (plus pad)
    # - we never mix prompt tokens into the per-token logps window.
    padded_comp_ids_list: List[List[int]] = []
    completion_lengths: List[int] = []
    for i, eff in enumerate(effective_comp_ids_list):
        comp = list(eff)
        if len(comp) > logits_to_keep:
            comp = comp[:logits_to_keep]
        k = min(len(comp), int(logits_to_keep))
        if len(comp) < logits_to_keep:
            comp = comp + [int(pad_id)] * (int(logits_to_keep) - len(comp))
        padded_comp_ids_list.append(comp)
        completion_lengths.append(k)

    comp_tensor = torch.tensor(padded_comp_ids_list, dtype=torch.long)
    eos_mask = get_response_mask(comp_tensor, eos_token=int(eos_id), dtype=torch.long)
    length_mask = torch.zeros_like(eos_mask)
    if len(completion_lengths) > 0:
        idx = torch.arange(int(logits_to_keep), dtype=torch.long).unsqueeze(0).expand(len(completion_lengths), -1)
        lens = torch.tensor(completion_lengths, dtype=torch.long).unsqueeze(1)
        length_mask = (idx < lens).to(torch.long)
    completion_mask_window = eos_mask * length_mask

    full_ids_list: List[List[int]] = [
        list(prompt_ids_list[i]) + list(padded_comp_ids_list[i]) for i in range(len(prompts))
    ]
    max_total_len = max((len(x) for x in full_ids_list), default=0)
    if max_total_len <= 0:
        # Degenerate: all empty. Still return a minimal 1-token batch to keep downstream code stable.
        max_total_len = 1

    B = len(full_ids_list)
    input_ids = torch.full((B, max_total_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((B, max_total_len), dtype=torch.long)

    for i, seq in enumerate(full_ids_list):
        if not seq:
            continue
        start = max_total_len - len(seq)
        input_ids[i, start:] = torch.tensor(seq, dtype=torch.long)
        # Flash-Attn safe: no right-padding holes in attention_mask.
        # (Completion padding positions are masked out via `completion_mask_window`, not attention_mask.)
        attention_mask[i, start:] = 1

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        completion_mask_window = completion_mask_window.to(device)

    return input_ids, attention_mask, completion_mask_window, int(logits_to_keep)

