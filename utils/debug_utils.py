# coding=utf-8
"""Debug / diagnostic helpers shared across pipelines.

All functions are guarded by environment flags (``CO_DEBUG_DATA``, etc.) so they
are no-ops in production runs.
"""

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import torch

from utils.env_utils import co_env_flag


def co_debug_data_dir(base_dir: str) -> str:
    """Resolve the debug-data output directory."""
    return os.environ.get("CO_DEBUG_DATA_DIR", os.path.join(base_dir, "debug_data"))


def co_append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file (best-effort, never raises)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return


def co_write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Atomically write required diagnostic JSON, raising on any failure."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def co_dump_alignment_samples(
    *,
    stage: str,
    output_dir: str,
    step: int,
    model_tag: str,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    finish_is_truncated: Optional[List[Optional[bool]]],
    prompt_completion_ids: torch.Tensor,
    completion_mask_window: torch.Tensor,
    logits_to_keep: int,
    max_items: int = 3,
    is_main: bool = True,
) -> None:
    """Dump a few samples to verify data matches open-r1 semantics.

    Guarded by ``CO_DEBUG_DATA=1``.
    """
    if not is_main or not co_env_flag("CO_DEBUG_DATA", "0"):
        return
    try:
        n = min(int(max_items), int(prompt_completion_ids.size(0)))
        path = os.path.join(
            co_debug_data_dir(output_dir),
            f"alignment_{stage}_{model_tag}_step{int(step)}.jsonl",
        )
        pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None
        eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
        for i in range(n):
            ids_window = prompt_completion_ids[i, -int(logits_to_keep):].detach().to("cpu").tolist()
            mask_window = completion_mask_window[i, :int(logits_to_keep)].detach().to("cpu").tolist()
            cm_sum = int(sum(mask_window))
            fr = None
            if finish_is_truncated is not None and i < len(finish_is_truncated):
                fr = finish_is_truncated[i]
            resp_ids = tokenizer(responses[i], add_special_tokens=False, padding=False, truncation=False)["input_ids"]
            rec = {
                "stage": stage,
                "step": int(step),
                "model_tag": str(model_tag),
                "i": int(i),
                "logits_to_keep": int(logits_to_keep),
                "completion_mask_sum": int(cm_sum),
                "finish_is_truncated": fr,
                "pad_id": pad_id,
                "eos_id": eos_id,
                "resp_len_tokens_raw": int(len(resp_ids)),
                "ids_window_head": ids_window[:min(64, len(ids_window))],
                "mask_window_head": mask_window[:min(64, len(mask_window))],
                "ids_window_tail": ids_window[max(0, len(ids_window) - 16):],
                "mask_window_tail": mask_window[max(0, len(mask_window) - 16):],
                "prompt_preview": str(prompts[i])[:400],
                "response_preview": str(responses[i])[:400],
            }
            co_append_jsonl(path, rec)
    except Exception:
        return


def co_dump_single_logps_samples(
    *,
    base_dir: str,
    step: int,
    prompts_preview: List[str],
    responses_preview: List[str],
    finish_preview: Optional[List[Optional[str]]],
    completion_ids: torch.Tensor,
    old_lp: Optional[List[List[float]]],
    ref_lp: Optional[List[List[float]]],
    max_items: int = 3,
) -> None:
    """Dump a few samples to verify logps/token alignment. Guarded by ``CO_DEBUG_DATA=1``."""
    if not co_env_flag("CO_DEBUG_DATA", "0"):
        return
    try:
        dump_dir = co_debug_data_dir(base_dir)
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"generate_logps_single_step{int(step)}.jsonl")
        n = min(int(max_items), int(completion_ids.size(0)))
        for i in range(n):
            ids = completion_ids[i].detach().to("cpu").tolist()
            rec = {
                "ts": time.time(),
                "step": int(step),
                "i": int(i),
                "completion_len_ids": int(len(ids)),
                "completion_ids_head": ids[:min(64, len(ids))],
                "completion_ids_tail": ids[max(0, len(ids) - 16):],
                "old_logps_len": (len(old_lp[i]) if old_lp is not None and i < len(old_lp) else None),
                "ref_logps_len": (len(ref_lp[i]) if ref_lp is not None and i < len(ref_lp) else None),
                "prompt_preview": str(prompts_preview[i])[:400] if i < len(prompts_preview) else None,
                "response_preview": str(responses_preview[i])[:400] if i < len(responses_preview) else None,
                "finish_reason": (finish_preview[i] if finish_preview is not None and i < len(finish_preview) else None),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return
