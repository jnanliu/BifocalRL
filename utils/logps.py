# coding=utf-8
"""Per-token log probability computation.

Includes both the low-level ``get_per_token_logps`` (used during GRPO / GSPO
training) and the higher-level ``compute_per_token_logps_from_prebuilt_batch``
(distributed-aware wrapper that loads a model, shards data, and gathers
results).
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
from transformers import PreTrainedModel
from utils.trl_compat import selective_log_softmax

from utils.distributed_utils import shard_range, is_main_process
from utils.env_utils import co_env_flag
from utils.model_utils import get_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level: per-token log probabilities
# ---------------------------------------------------------------------------

def get_per_token_logps(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    temperature: float = 1.0,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Get per-token log probabilities for completions.

    Args:
        model: Model to compute logps with
        input_ids: Input token IDs (B, L)
        attention_mask: Attention mask (B, L)
        logits_to_keep: Number of logit positions to keep
        temperature: Temperature for logits scaling
        batch_size: Batch size for processing

    Returns:
        Per-token log probabilities (B, logits_to_keep)
    """
    batch_size = batch_size or input_ids.size(0)
    all_logps = []

    num_batches = (input_ids.size(0) + batch_size - 1) // batch_size

    # Allow callers to keep large tensors on CPU and move micro-batches to GPU.
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = input_ids.device

    if torch.cuda.is_available() and co_env_flag("CO_EMPTY_CACHE_LOGPS", "0"):
        torch.cuda.empty_cache()

    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(
            range(0, input_ids.size(0), batch_size),
            desc="Computing logps",
            unit="batch",
            total=num_batches,
            leave=False,
            position=1,
            dynamic_ncols=True,
        )
    else:
        iterator = range(0, input_ids.size(0), batch_size)

    for i in iterator:
        input_ids_batch = input_ids[i : i + batch_size]
        attention_mask_batch = attention_mask[i : i + batch_size]

        if model_device is not None:
            input_ids_batch = input_ids_batch.to(model_device, non_blocking=True)
            attention_mask_batch = attention_mask_batch.to(model_device, non_blocking=True)

        if torch.cuda.is_available() and i > 0 and co_env_flag("CO_EMPTY_CACHE_LOGPS", "0"):
            torch.cuda.empty_cache()

        logits = model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            logits_to_keep=logits_to_keep + 1,
            use_cache=False,
        ).logits

        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit
        input_ids_batch = input_ids_batch[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]

        # Divide logits by sampling temperature (in-place to save memory).
        if temperature != 1.0:
            logits.div_(logits.new_tensor(temperature))

        logps = selective_log_softmax(logits, input_ids_batch)
        # Do NOT detach — we need gradients for GRPO loss computation.
        all_logps.append(logps)

        del logits
        if torch.cuda.is_available() and co_env_flag("CO_EMPTY_CACHE_LOGPS", "0"):
            torch.cuda.empty_cache()

    return torch.cat(all_logps, dim=0)


# ---------------------------------------------------------------------------
# High-level: distributed-aware wrapper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_token_logps_from_prebuilt_batch(
    *,
    model_path: str,
    prompt_completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    temperature: float,
    batch_size: int,
    accelerator=None,
) -> torch.Tensor:
    """Compute per-token logps over the suffix window of length *logits_to_keep*.

    Returns:
        ``(N, logits_to_keep)`` tensor on CPU, gathered across ranks when
        *accelerator* is distributed.
    """
    if accelerator is not None and torch.cuda.is_available():
        torch_device = getattr(accelerator, "device", torch.device("cuda"))
        device_str = str(torch_device)
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = str(torch_device)

    model = get_model(model_path, device=device_str)
    model.eval()

    n_total = int(prompt_completion_ids.size(0))
    start, end, per_rank = shard_range(n_total, accelerator)
    is_main = is_main_process(accelerator)

    ids_local = prompt_completion_ids[start:end]
    mask_local = attention_mask[start:end]

    num_batches = (ids_local.size(0) + int(batch_size) - 1) // int(batch_size)
    from tqdm import tqdm

    all_logps = []
    iterator = range(0, ids_local.size(0), int(batch_size))
    iterator = tqdm(
        iterator,
        desc="Computing log probabilities",
        unit="batch",
        total=num_batches,
        leave=False,
        disable=not is_main,
    )
    for i in iterator:
        j = min(i + int(batch_size), ids_local.size(0))
        batch_ids = ids_local[i:j].to(torch_device, non_blocking=True)
        batch_mask = mask_local[i:j].to(torch_device, non_blocking=True)
        per_token = get_per_token_logps(
            model,
            batch_ids,
            batch_mask,
            int(logits_to_keep),
            temperature=float(temperature),
            batch_size=int(batch_size),
        )
        all_logps.append(per_token.detach().cpu())

    if all_logps:
        logps_local = torch.cat(all_logps, dim=0)
    else:
        logps_local = torch.empty((0, int(logits_to_keep)), dtype=torch.float32)

    if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
        padded = torch.zeros((int(per_rank), int(logits_to_keep)), dtype=logps_local.dtype)
        if logps_local.numel() > 0:
            padded[: logps_local.size(0)] = logps_local
        padded = padded.to(torch_device, non_blocking=True)
        gathered = accelerator.gather(padded)
        logps_all = gathered[:n_total].detach().cpu()
    else:
        logps_all = logps_local

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return logps_all
