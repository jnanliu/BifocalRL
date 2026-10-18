# coding=utf-8
"""
Co-Intuitor Training Pipeline V2: Two separate runs approach.

This script supports training one model at a time:
- Run 1: Train Model A (policy=Model A, reward=Model B)
- Run 2: Train Model B (policy=Model B, reward=Model A)

Training data is loaded from cache (generated and scored by generate_and_score.py).

Usage:
    accelerate launch src/co_intuitor_train.py --train_model a --output_dir out_model_a --cache_dir ./cache
    accelerate launch src/co_intuitor_train.py --train_model b --output_dir out_model_b --cache_dir ./cache
"""

import logging
import json
import os
import sys
import random
import time
from typing import Optional, List, Dict, Any, Tuple, MutableMapping

import datasets
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from utils.configs import CoIntuitorConfig, CoIntuitorScriptArguments
from utils.single_model_trainer import SingleModelTrainer
from utils.cache import CoIntuitorCache
from utils.generate import strip_prompt_echo_from_flattened_responses, strip_chat_template_artifacts_from_flattened_responses
from utils.mv import rewards_from_wmv_pseudolabel, rewards_from_mv_pseudolabel
from utils.normalization import (
    batch_zscore_normalize,
    grpo_group_advantages,
    z_score_normalize_tensor,
    z_score_normalize_tensor_with_bounds,
    rowwise_pearson_corrcoef,
    normalize_to_unit_interval,
)
from utils.capability_report import CapabilityReporter
from utils.logps import get_per_token_logps
from utils.invariants import (
    assert_attention_mask_matches_concat,
    assert_prompt_completion_shapes,
    tc_check_hacpo_mixed_batch,
    tc_check_generations_block,
    tc_check_logps_matrix,
    truthcheck_enabled,
    tc_check_padding_semantics,
    tc_check_score_vector,
    tc_check_training_batch,
)
from utils.tokenization import append_eos_1d, tokenize_and_postprocess_data, get_response_mask
from utils.trl_compat import ModelConfig, TrlParser, pad
from utils.env_utils import co_env_flag

logger = logging.getLogger(__name__)


def _co_nan_dump_dir() -> str:
    # Keep this easy to disable/remove: driven entirely by env vars.
    return os.environ.get("CO_DEBUG_NAN_DIR", "outputs/debug_nan")


def _co_tensor_stats(t: torch.Tensor, sample_items: int = 64) -> Dict[str, Any]:
    """Lightweight tensor stats safe for json logging."""
    try:
        td = t.detach()
        # Keep stats on CPU to avoid extra GPU pressure.
        if td.device.type != "cpu":
            td = td.to("cpu")
        stats: Dict[str, Any] = {
            "dtype": str(td.dtype),
            "shape": list(td.shape),
            "device": str(t.device),
            "numel": int(td.numel()),
        }
        # isfinite works for ints too (always True).
        finite = torch.isfinite(td)
        stats["finite"] = int(finite.sum().item())
        stats["nonfinite"] = int((~finite).sum().item())
        if td.is_floating_point():
            stats["nan"] = int(torch.isnan(td).sum().item())
            stats["inf"] = int(torch.isinf(td).sum().item())
        if stats["finite"] > 0:
            f = td[finite]
            # Convert to python floats/ints.
            stats["min"] = float(f.min().item())
            stats["max"] = float(f.max().item())
            stats["mean"] = float(f.mean().item())
        # Small sample for quick inspection.
        flat = td.reshape(-1)
        k = min(int(sample_items), int(flat.numel()))
        if k > 0:
            if flat.is_floating_point():
                stats["sample"] = flat[:k].to(torch.float32).tolist()
            else:
                stats["sample"] = flat[:k].tolist()
        return stats
    except Exception as e:
        return {"error": f"stats_failed: {e}"}


def _co_dump_nan_event(tag: str, tensors: Dict[str, Optional[torch.Tensor]], extra: Optional[Dict[str, Any]] = None):
    """Append a JSONL record describing the first non-finite event."""
    if not co_env_flag("CO_DEBUG_NAN", "0"):
        return
    try:
        dump_dir = _co_nan_dump_dir()
        os.makedirs(dump_dir, exist_ok=True)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        pid = os.getpid()
        path = os.path.join(dump_dir, f"nan_dump_rank{rank}.jsonl")
        rec: Dict[str, Any] = {
            "ts": time.time(),
            "tag": tag,
            "rank": str(rank),
            "pid": int(pid),
            "extra": extra or {},
            "tensors": {},
        }
        for k, v in tensors.items():
            if v is None:
                rec["tensors"][k] = None
            else:
                rec["tensors"][k] = _co_tensor_stats(v)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Never crash training due to debug dumping.
        return


def _co_dump_prepare_full_samples(
    *,
    dump_dir: str,
    step: int,
    train_model: str,
    tokenizer,
    samples: List[Dict[str, Any]],
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    max_items: int = 10,
) -> None:
    """
    Dump fully-assembled training samples for debugging:
      - text: prompt (already chat_template-rendered), completion text, and a padded view
      - tokens: ids + token strings for prompt/completion/concat
      - masks: prompt_mask, completion_mask, attention_mask

    Only rank0 writes. Guarded by CO_DEBUG_DATA=1.
    """
    if not co_env_flag("CO_DEBUG_DATA", "0"):
        return
    try:
        rank_s = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        rank = int(rank_s) if str(rank_s).strip().isdigit() else 0
        if rank != 0:
            return
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"prepare_train_full_{str(train_model)}_step{int(step)}_rank{rank}.jsonl")

        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        chat_template = getattr(tokenizer, "chat_template", None)

        n = min(int(max_items), int(completion_ids.size(0)), int(len(samples)))

        def _head_tail(seq: Optional[List[Any]], max_total: int = 50) -> Dict[str, Any]:
            if seq is None:
                return {"len": None, "head": None, "tail": None, "truncated": None}
            n0 = int(len(seq))
            if n0 <= int(max_total):
                return {"len": n0, "head": seq, "tail": [], "truncated": False}
            # Keep head+tail, each half of max_total.
            h = int(max_total) // 2
            t = int(max_total) - h
            return {"len": n0, "head": seq[:h], "tail": seq[-t:], "truncated": True}

        for i in range(n):
            p_ids = prompt_ids[i].detach().to("cpu").tolist()
            p_mask = prompt_mask[i].detach().to("cpu").tolist()
            c_ids = completion_ids[i].detach().to("cpu").tolist()
            c_mask = completion_mask[i].detach().to("cpu").tolist()

            # Concatenated view matches training-time inputs.
            in_ids = p_ids + c_ids
            attn = p_mask + c_mask

            # Token strings (more faithful than decode when padding is involved).
            try:
                p_toks = tokenizer.convert_ids_to_tokens(p_ids)
            except Exception:
                p_toks = None
            try:
                c_toks = tokenizer.convert_ids_to_tokens(c_ids)
            except Exception:
                c_toks = None
            try:
                in_toks = tokenizer.convert_ids_to_tokens(in_ids)
            except Exception:
                in_toks = None

            # Best-effort decoded strings (may drop pad tokens depending on tokenizer).
            try:
                prompt_text_tok = tokenizer.decode(p_ids, skip_special_tokens=False)
            except Exception:
                prompt_text_tok = None
            try:
                completion_text_tok = tokenizer.decode(c_ids, skip_special_tokens=False)
            except Exception:
                completion_text_tok = None
            try:
                full_text_tok = tokenizer.decode(in_ids, skip_special_tokens=False)
            except Exception:
                full_text_tok = None

            rec: Dict[str, Any] = {
                "step": int(step),
                "train_model": str(train_model),
                "i": int(i),
                "tokenizer": {
                    "pad_token_id": (int(pad_id) if pad_id is not None else None),
                    "eos_token_id": (int(eos_id) if eos_id is not None else None),
                    "chat_template": (str(chat_template) if chat_template is not None else None),
                },
                "sample_meta": samples[i],
                "text": {
                    "prompt_rendered": samples[i].get("prompt_text"),
                    "completion_raw": samples[i].get("completion_text"),
                    # Decoded views of padded token sequences (useful to confirm special tokens/padding behavior).
                    "prompt_from_tokens": prompt_text_tok,
                    "completion_from_tokens_padded": completion_text_tok,
                    "prompt_plus_completion_from_tokens_padded": full_text_tok,
                },
                "tokens": {
                    "prompt_ids": _head_tail(p_ids),
                    "prompt_tokens": _head_tail(p_toks),
                    "prompt_mask": _head_tail(p_mask),
                    "completion_ids": _head_tail(c_ids),
                    "completion_tokens": _head_tail(c_toks),
                    "completion_mask": _head_tail(c_mask),
                    "input_ids_concat": _head_tail(in_ids),
                    "input_tokens_concat": _head_tail(in_toks),
                    "attention_mask_concat": _head_tail(attn),
                },
                "lengths": {
                    "prompt_len_mask": int(sum(p_mask)),
                    "completion_len_mask": int(sum(c_mask)),
                    "input_len_ids": int(len(in_ids)),
                },
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return

def _co_shift_mismatch_report(
    ref_lp: torch.Tensor,
    lp: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_shift: int = 16,
) -> Dict[str, Any]:
    """
    Try shifting ref_lp left/right to see if a constant column offset explains large deltas.

    Returns:
      dict with per-shift max_delta on masked tokens and the best shift.
    """
    try:
        if ref_lp.dim() != 2 or lp.dim() != 2 or mask.dim() != 2:
            return {"error": "expected 2D tensors"}
        B, L = ref_lp.shape
        if lp.shape != (B, L) or mask.shape != (B, L):
            return {"error": "shape mismatch"}
        m = mask.to(torch.bool)
        out: Dict[str, Any] = {"max_shift": int(max_shift), "shifts": {}}
        best = None
        for s in range(-int(max_shift), int(max_shift) + 1):
            # Shift ref_lp by s columns (positive s means ref moves right).
            shifted = ref_lp.new_zeros(ref_lp.shape)
            if s == 0:
                shifted = ref_lp
            elif s > 0:
                shifted[:, s:] = ref_lp[:, : L - s]
            else:
                ss = -s
                shifted[:, : L - ss] = ref_lp[:, ss:]
            d = (shifted - lp)
            d = d.masked_fill(~m, 0.0)
            max_d = float(d.max().item()) if d.numel() > 0 else 0.0
            out["shifts"][str(s)] = {"max_delta": max_d}
            if best is None or max_d < best[1]:
                best = (s, max_d)
        if best is not None:
            out["best_shift"] = int(best[0])
            out["best_max_delta"] = float(best[1])
        return out
    except Exception as e:
        return {"error": f"shift_report_failed: {e}"}


def get_tokenizer(model_path: str, training_args: CoIntuitorConfig):
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(model_path: str, training_args: CoIntuitorConfig, model_args: ModelConfig = None):
    """Get the model.
    
    Similar to intuitor_trainer.py model initialization logic.
    Uses model_init_kwargs from training_args or model_args.torch_dtype.
    """
    # Get model_init_kwargs from training_args (like intuitor_trainer.py:378)
    model_init_kwargs = training_args.model_init_kwargs or {}
    
    # Handle torch_dtype (like intuitor_trainer.py:381-391)
    torch_dtype = model_init_kwargs.get("torch_dtype")
    if torch_dtype is None and model_args is not None:
        # Fallback to model_args.torch_dtype if available
        torch_dtype = model_args.torch_dtype if hasattr(model_args, 'torch_dtype') else None
    
    if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
        pass  # torch_dtype is already a torch.dtype or "auto" or None
    elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
        torch_dtype = getattr(torch, torch_dtype)
        model_init_kwargs["torch_dtype"] = torch_dtype
    elif torch_dtype is not None:
        raise ValueError(
            "Invalid `torch_dtype` passed. Expected either 'auto' or a string representing "
            f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
        )
    
    # Set default torch_dtype if not specified
    if "torch_dtype" not in model_init_kwargs:
        model_init_kwargs["torch_dtype"] = torch_dtype

    # Do not pass use_cache into from_pretrained: models like Gemma3ForConditionalGeneration
    # reject it under DeepSpeed ZeRO-3 (kwargs reach __init__). Apply via config after load.
    _use_cache_from_user = model_init_kwargs.pop("use_cache", None)
    use_cache_desired = (
        False
        if training_args.gradient_checkpointing
        else (_use_cache_from_user if _use_cache_from_user is not None else True)
    )

    # Add trust_remote_code if not in model_init_kwargs
    if "trust_remote_code" not in model_init_kwargs:
        model_init_kwargs["trust_remote_code"] = True

    # Prefer flash_attention_2 if available (open-r1 default), otherwise fall back to sdpa.
    from utils.attn import maybe_add_attn_implementation

    maybe_add_attn_implementation(model_init_kwargs)
    attn_impl_requested = model_init_kwargs.get("attn_implementation", None)
    
    # Handle quantization (like model_utils.py)
    # CRITICAL: If DeepSpeed is enabled (especially ZeRO-3), do NOT set device_map or quantization_config
    # DeepSpeed/accelerate will handle device placement and sharding automatically
    # Setting device_map here can cause conflicts with DeepSpeed's parameter sharding
    
    # Check if DeepSpeed is enabled
    # Priority: 1) args.deepspeed (explicit config), 2) accelerate config file
    is_deepspeed_enabled = False
    
    # First check if deepspeed is explicitly set in args
    if hasattr(training_args, "deepspeed") and training_args.deepspeed is not None:
        is_deepspeed_enabled = True
    else:
        # Check accelerate config file (when using accelerate launch --config_file)
        from pathlib import Path
        
        # Try to find accelerate config file
        accelerate_config_file = None
        # Check environment variable (if set by accelerate)
        if "ACCELERATE_CONFIG_FILE" in os.environ:
            accelerate_config_file = os.environ["ACCELERATE_CONFIG_FILE"]
        else:
            # Check default location
            default_config = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
            if default_config.exists():
                accelerate_config_file = str(default_config)
        
        if accelerate_config_file and os.path.exists(accelerate_config_file):
            try:
                import yaml
                with open(accelerate_config_file, "r") as f:
                    accelerate_config = yaml.safe_load(f)
                    # Check if distributed_type is DEEPSPEED
                    if accelerate_config.get("distributed_type") == "DEEPSPEED":
                        is_deepspeed_enabled = True
                        logger.info(f"Detected DeepSpeed from accelerate config: {accelerate_config_file}")
            except Exception as e:
                logger.warning(f"Failed to read accelerate config file {accelerate_config_file}: {e}")
    
    if not is_deepspeed_enabled:
        # Only set device_map and quantization_config if DeepSpeed is NOT enabled
        from utils.trl_compat import get_quantization_config, get_kbit_device_map
        quantization_config = get_quantization_config(model_args if model_args else training_args)
        if quantization_config is not None:
            model_init_kwargs["device_map"] = get_kbit_device_map()
            model_init_kwargs["quantization_config"] = quantization_config
    else:
        # DeepSpeed is enabled - let DeepSpeed/accelerate handle device placement
        logger.info("DeepSpeed is enabled - skipping device_map and quantization_config setup")
    
    # Under DeepSpeed ZeRO-3, embedding weights are partitioned and may have size 0
    # on some ranks. transformers' _init_weights tries to access
    # module.weight.data[module.padding_idx] on zero-sized tensors, causing IndexError.
    # Always patch: the guard is harmless without ZeRO-3 (IndexError simply won't occur).
    import transformers.modeling_utils as _mu
    _original_init_weights = _mu.PreTrainedModel._init_weights

    def _safe_init_weights(self, module):
        try:
            _original_init_weights(self, module)
        except IndexError:
            pass

    _mu.PreTrainedModel._init_weights = _safe_init_weights

    # Load model (be robust to transformers versions/models that don't accept attn_implementation)
    from utils.model_utils import _detect_model_type, _build_mistral3_config
    _mt = _detect_model_type(model_path)

    try:
        if _mt == "mistral3":
            from transformers import Mistral3ForConditionalGeneration
            _cfg = _build_mistral3_config(model_path)
            _full = Mistral3ForConditionalGeneration.from_pretrained(
                model_path, config=_cfg, **model_init_kwargs,
            )
            if hasattr(_full, "language_model") and hasattr(_full.language_model, "lm_head"):
                model = _full.language_model
                del _full
            else:
                model = _full
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_init_kwargs,
            )
    except TypeError:
        if "attn_implementation" in model_init_kwargs:
            model_init_kwargs = dict(model_init_kwargs)
            model_init_kwargs.pop("attn_implementation", None)
            logger.warning("Model init rejected attn_implementation; retrying without it.")
            if _mt == "mistral3":
                from transformers import Mistral3ForConditionalGeneration
                _cfg = _build_mistral3_config(model_path)
                _full = Mistral3ForConditionalGeneration.from_pretrained(
                    model_path, config=_cfg, **model_init_kwargs,
                )
                if hasattr(_full, "language_model") and hasattr(_full.language_model, "lm_head"):
                    model = _full.language_model
                    del _full
                else:
                    model = _full
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_init_kwargs,
                )
        else:
            raise
    finally:
        _mu.PreTrainedModel._init_weights = _original_init_weights

    try:
        if hasattr(model, "config") and model.config is not None:
            model.config.use_cache = use_cache_desired
    except Exception:
        pass

    # Ensure gradient checkpointing actually takes effect (we do not use transformers.Trainer here).
    if getattr(training_args, "gradient_checkpointing", False):
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled on the model.")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing on model: {e}")

    # Log flash-attn status clearly.
    try:
        cfg = getattr(model, "config", None)
        attn_impl_actual = None
        if cfg is not None:
            attn_impl_actual = getattr(cfg, "attn_implementation", None) or getattr(cfg, "_attn_implementation", None)
        if attn_impl_actual is None:
            attn_impl_actual = getattr(model, "_attn_implementation", None)

        if attn_impl_requested == "flash_attention_2":
            try:
                import flash_attn  # type: ignore  # noqa: F401

                if attn_impl_actual == "flash_attention_2":
                    logger.info("Flash Attention 2 is enabled and in use (attn_implementation=flash_attention_2).")
                else:
                    logger.warning(
                        f"Requested flash_attention_2, but model reports attn_implementation={attn_impl_actual}. "
                        "Falling back (this may be expected for some model types/transformers versions)."
                    )
            except Exception as e:
                logger.warning(f"Requested flash_attention_2 but flash_attn import failed: {e}.")
        else:
            logger.info(f"Attention implementation: requested={attn_impl_requested}, actual={attn_impl_actual}")
    except Exception:
        pass
    return model


def _parse_selected_indices(raw: Optional[str], total: int) -> Optional[List[int]]:
    """Parse a user-supplied index specification into a concrete list.

    Accepted formats:
      - None / empty string  ->  None  (caller falls back to default logic)
      - "all"                ->  [0, 1, ..., total-1]
      - "0,1,2,3"           ->  [0, 1, 2, 3]
    Every index must be in [0, total).
    """
    if raw is None or raw.strip() == "":
        return None
    raw = raw.strip()
    if raw.lower() == "all":
        return list(range(total))
    indices = [int(x.strip()) for x in raw.split(",")]
    for idx in indices:
        if idx < 0 or idx >= total:
            raise ValueError(
                f"selected_indices contains {idx}, but valid range is [0, {total}). "
                f"Layout: [A_0..A_{total // 2 - 1}, B_0..B_{total // 2 - 1}]."
            )
    return indices


def _compute_per_prompt_indices(
    learn_cfg: Dict[str, Any],
    train_model: str,
    num_generations_per_model: int,
    scores_combined: torch.Tensor,
    step: int,
) -> List[List[int]]:
    """Compute per-prompt selected indices based on learn_config.

    Data layout per prompt: [A_0 .. A_{N-1}, B_0 .. B_{N-1}].

    Selection modes:
      - "first" (default): take the first N in generation order (static, deterministic).
      - "top_score": rank by combined score (score_a + score_b), pick the highest.
      - "random": random subset (seeded by step + prompt index).

    Args:
        learn_cfg: e.g. {"a": {"own": 4, "other": 3}, "b": ..., "selection": "first"}
        train_model: "a" or "b"
        num_generations_per_model: N (per model)
        scores_combined: (num_prompts, 2N) combined scores (only used for "top_score")
        step: training step (seeds "random" selection)

    Returns:
        List[List[int]]: per-prompt index lists, each with the same length.
    """
    N = num_generations_per_model
    model_cfg = learn_cfg.get(train_model, {})
    own_n = model_cfg.get("own", -1)
    other_n = model_cfg.get("other", -1)
    selection = learn_cfg.get("selection", "first")

    if train_model == "a":
        own_range = list(range(N))
        other_range = list(range(N, 2 * N))
    else:
        own_range = list(range(N, 2 * N))
        other_range = list(range(N))

    if own_n < 0 or own_n > len(own_range):
        own_n = len(own_range)
    if other_n < 0 or other_n > len(other_range):
        other_n = len(other_range)

    total_per = own_n + other_n
    logger.info(
        "learn_config: model %s learns %d own + %d other = %d per prompt (selection=%s)",
        train_model, own_n, other_n, total_per, selection,
    )

    # Fast path: "first" is static — same indices for every prompt.
    if selection == "first":
        own_sel = own_range[:own_n]
        other_sel = other_range[:other_n]
        static_indices = own_sel + other_sel
        num_prompts = scores_combined.size(0)
        return [static_indices] * num_prompts

    num_prompts = scores_combined.size(0)
    result: List[List[int]] = []

    for i in range(num_prompts):
        prompt_scores = scores_combined[i]

        if own_n >= len(own_range):
            own_sel = list(own_range)
        elif selection == "top_score":
            ranked = sorted(own_range, key=lambda j: prompt_scores[j].item(), reverse=True)
            own_sel = sorted(ranked[:own_n])
        else:
            rng = random.Random((step + 1) * 10_000_000 + i * 100)
            own_sel = sorted(rng.sample(own_range, own_n))

        if other_n == 0:
            other_sel: List[int] = []
        elif other_n >= len(other_range):
            other_sel = list(other_range)
        elif selection == "top_score":
            ranked = sorted(other_range, key=lambda j: prompt_scores[j].item(), reverse=True)
            other_sel = sorted(ranked[:other_n])
        else:
            rng = random.Random((step + 1) * 10_000_000 + i * 100 + 1)
            other_sel = sorted(rng.sample(other_range, other_n))

        result.append(own_sel + other_sel)

    return result


def _reconstruct_full_scores(
    scores_a_flat: List[float],
    scores_b_flat: List[float],
    num_unique_prompts: int,
    num_generations_per_model: int,
    reward_method: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct full ``(N, 2G)`` score tensors from cached scores.

    Cached scores may be either per-model ``(N, G)`` (truth/self_scores/co_mv)
    or cross-model ``(N, 2G)`` (sum_scores/co_wmv/mixed_reward).  This helper detects the
    width and pads/concatenates as needed so downstream code always gets
    ``(num_unique_prompts, 2*num_generations_per_model)`` tensors.
    """
    # Defensively flatten nested lists (e.g. from 2D tensor.tolist() saved to cache).
    if scores_a_flat and isinstance(scores_a_flat[0], (list, tuple)):
        scores_a_flat = [x for sub in scores_a_flat for x in sub]
    if scores_b_flat and isinstance(scores_b_flat[0], (list, tuple)):
        scores_b_flat = [x for sub in scores_b_flat for x in sub]

    total_per_prompt = 2 * num_generations_per_model
    G = num_generations_per_model
    scores_per_prompt_a = len(scores_a_flat) // num_unique_prompts
    scores_per_prompt_b = len(scores_b_flat) // num_unique_prompts

    if scores_per_prompt_a == total_per_prompt and scores_per_prompt_b == total_per_prompt:
        # Already full cross-model format.
        scores_a_t = torch.tensor(scores_a_flat, device=device).view(num_unique_prompts, total_per_prompt)
        scores_b_t = torch.tensor(scores_b_flat, device=device).view(num_unique_prompts, total_per_prompt)
        return scores_a_t, scores_b_t

    if scores_per_prompt_a != G or scores_per_prompt_b != G:
        raise ValueError(
            f"Unexpected score widths: scores_a has {scores_per_prompt_a}/prompt, "
            f"scores_b has {scores_per_prompt_b}/prompt. Expected {G} (per-model) or {total_per_prompt} (cross-model)."
        )

    # Per-model scores → reconstruct to (N, 2G).
    own_a = torch.tensor(scores_a_flat, device=device).view(num_unique_prompts, G)
    own_b = torch.tensor(scores_b_flat, device=device).view(num_unique_prompts, G)

    if reward_method in ("truth", "co_mv", "self_mv", "mv_gated_intuitor", "co_mv_gated_intuitor"):
        # truth:   scores_a = A's GT rewards,      scores_b = B's GT rewards.
        # co_mv:   scores_a = A's co-MV rewards,   scores_b = B's co-MV rewards.
        # self_mv: scores_a = A's self-MV rewards,  scores_b = B's self-MV rewards.
        # mv_gated*: scores_a = A's gated rewards,  scores_b = B's gated rewards.
        # All are per-model → concat to get full (N, 2G).
        full = torch.cat([own_a, own_b], dim=1)
        return full, full.clone()

    if reward_method == "self_scores":
        # A scored only own responses → pad other half with zeros.
        full_a = torch.zeros(num_unique_prompts, total_per_prompt, device=device)
        full_a[:, :G] = own_a
        full_b = torch.zeros(num_unique_prompts, total_per_prompt, device=device)
        full_b[:, G:] = own_b
        return full_a, full_b

    raise ValueError(
        f"Per-model scores (width={G}) are not compatible with reward_method={reward_method!r}. "
        f"Re-run generate_and_score with this reward_method to produce cross-model scores "
        f"(e.g. sum_scores / mixed_reward need 2G-wide cached SCE)."
    )


def prepare_training_data_from_cache(
    cache: CoIntuitorCache,
    step: int,
    train_model: str,  # "a" or "b"
    unique_prompts: List[str],
    tokenizer,
    num_generations_per_model: int = 4,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    mask_truncated_completions: bool = False,
    reward_method: str = "sum_scores",
    reward_weight_a: float = 1.0,
    reward_weight_b: float = 1.0,
    mixed_reward_alpha: float = 0.5,
    mixed_reward_alpha_mode: str = "fixed",
    mixed_reward_alpha_zscore_mu: float = 0.65,
    mixed_reward_alpha_zscore_sigma: float = 0.2,
    mixed_reward_alpha_zscore_lower: float = 0.0,
    mixed_reward_alpha_zscore_upper: float = 1.0,
    mixed_reward_corr_weight_mode: str = "none",
    mixed_reward_corr_weight_threshold: float = 0.1,
    mixed_reward_corr_weight_exponent: float = 0.8,
    sum_scores_minmax: bool = False,
    scale_rewards: bool = True,
    device: torch.device = None,
    selected_indices_a_raw: Optional[str] = None,
    selected_indices_b_raw: Optional[str] = None,
    learn_config_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare training data from cache for a specific model.
    
    Args:
        cache: CoIntuitorCache instance
        step: Training step
        train_model: "a" or "b" - which model to train
        unique_prompts: List of unique prompt texts
        tokenizer: Tokenizer for the training model
        num_generations_per_model: Number of generations per model
        max_prompt_length: Maximum prompt length
        mask_truncated_completions: Whether to mask truncated completions
        scale_rewards: Whether to scale rewards
        device: Device to use
        selected_indices_a_raw: Custom comma-separated indices for model A (e.g. "0,1,2,3",
                                "all", or None).
        selected_indices_b_raw: Custom comma-separated indices for model B. Same format.
        learn_config_json: JSON string for score-based per-prompt selection.
                           Takes priority over selected_indices_*. Use other=0 for self-learning.
                           Format: '{"a":{"own":4,"other":3},"b":{"own":4,"other":3},"selection":"top_score"}'
        mixed_reward_alpha: Only used when reward_method=mixed_reward. MV weight=alpha, Intuitor=1-alpha
            (fixed mode only; still used for capability blending in two-phase).
        mixed_reward_alpha_mode: ``fixed``, ``adaptive`` (magnitude), ``adaptive_sc`` (SC score),
            or ``adaptive_sc_zscore`` (single-model only; co path rejected explicitly).
        mixed_reward_corr_weight_mode: Optional post-combine prompt weighting mode for mixed_reward.
        sum_scores_minmax: If True, min-max the weighted SCE sum to [0,1] per prompt (2N) for
            sum_scores and for the Intuitor branch of mixed_reward. Default False = raw SCE sum.

    Returns:
        Dict with training data: prompt_ids, completion_ids, advantages, etc.
    """
    total_per_prompt = num_generations_per_model * 2

    # --- Parse learn_config (highest priority) ---
    learn_cfg: Optional[Dict[str, Any]] = None
    if learn_config_json and learn_config_json.strip():
        learn_cfg = json.loads(learn_config_json)
        logger.info("Using learn_config: %s", learn_cfg)

    # --- Determine index selection mode ---
    # per_prompt_indices will be populated later (after scores are loaded) if learn_cfg is set.
    # For legacy modes, compute static selected_indices now.
    use_dynamic_selection = learn_cfg is not None
    selected_indices: Optional[List[int]] = None

    if not use_dynamic_selection:
        custom_raw = selected_indices_a_raw if train_model == "a" else selected_indices_b_raw
        custom_indices = _parse_selected_indices(custom_raw, total_per_prompt)

        if custom_indices is not None:
            selected_indices = custom_indices
            logger.info(
                "Model %s: using custom selected_indices = %s (from --selected_indices_%s)",
                train_model, selected_indices, train_model,
            )
        else:
            selected_indices_a = [0, 1, 2, 3, 4, 5, 6]
            selected_indices_b = [4, 5, 6, 7, 0, 1, 2]
            if train_model == "a":
                selected_indices = selected_indices_a
            else:
                selected_indices = selected_indices_b
    
    # Load generations and scores from cache
    is_valid_a, responses_a, num_prompts_a, cached_prompts_a, finish_a = cache.load_generations_a(step, num_generations_per_model)
    is_valid_b, responses_b, num_prompts_b, cached_prompts_b, finish_b = cache.load_generations_b(step, num_generations_per_model)
    is_valid_scores_a, scores_a = cache.load_scores_a(step)
    is_valid_scores_b, scores_b = cache.load_scores_b(step)
    
    if not (is_valid_a and is_valid_b and is_valid_scores_a and is_valid_scores_b):
        raise ValueError(f"Cache data not complete for step {step}")
    tc_check_generations_block(
        where="co.train.prepare_cache.generations_a",
        prompts=(cached_prompts_a if cached_prompts_a is not None else unique_prompts),
        responses=responses_a,
        num_generations=int(num_generations_per_model),
        finish_reasons=finish_a,
    )
    tc_check_generations_block(
        where="co.train.prepare_cache.generations_b",
        prompts=(cached_prompts_b if cached_prompts_b is not None else unique_prompts),
        responses=responses_b,
        num_generations=int(num_generations_per_model),
        finish_reasons=finish_b,
    )
    
    # Use cached prompts if available, otherwise use provided unique_prompts.
    #
    # IMPORTANT: prompts can legitimately differ between model A and B if they use different chat templates.
    # We must sanitize A generations using prompts_a, and B generations using prompts_b (matches generate_and_score.py),
    # otherwise we can accidentally strip the wrong prompt prefix and create a mismatch vs cached logps.
    prompts_a_local = cached_prompts_a if cached_prompts_a is not None else list(unique_prompts)
    prompts_b_local = cached_prompts_b if cached_prompts_b is not None else list(unique_prompts)
    if train_model == "a" and cached_prompts_a is not None:
        unique_prompts = cached_prompts_a
    elif train_model == "b" and cached_prompts_b is not None:
        unique_prompts = cached_prompts_b
    elif cached_prompts_a is not None:
        unique_prompts = cached_prompts_a
    elif cached_prompts_b is not None:
        unique_prompts = cached_prompts_b

    # Sanitize cached generations (prompt echo can inflate completion length and break training/logps).
    responses_a, _changed_a = strip_prompt_echo_from_flattened_responses(
        prompts_a_local, responses_a, num_generations=num_generations_per_model
    )
    responses_b, _changed_b = strip_prompt_echo_from_flattened_responses(
        prompts_b_local, responses_b, num_generations=num_generations_per_model
    )
    # Also strip leading chat-template markers that may leak into completions when generated by a different model.
    responses_a, _changed2_a = strip_chat_template_artifacts_from_flattened_responses(
        prompts_a_local, responses_a, num_generations=num_generations_per_model
    )
    responses_b, _changed2_b = strip_chat_template_artifacts_from_flattened_responses(
        prompts_b_local, responses_b, num_generations=num_generations_per_model
    )
    
    if num_prompts_a != num_prompts_b or num_prompts_a != len(unique_prompts):
        raise ValueError(f"Number of prompts mismatch: {num_prompts_a}, {num_prompts_b}, {len(unique_prompts)}")
    
    num_unique_prompts = num_prompts_a
    tc_check_score_vector(
        where="co.train.prepare_cache.scores_a",
        scores=scores_a,
        num_prompts=int(num_unique_prompts),
        allowed_per_prompt=[int(num_generations_per_model), int(num_generations_per_model) * 2],
    )
    tc_check_score_vector(
        where="co.train.prepare_cache.scores_b",
        scores=scores_b,
        num_prompts=int(num_unique_prompts),
        allowed_per_prompt=[int(num_generations_per_model), int(num_generations_per_model) * 2],
    )
    
    if reward_method not in {
        "sum_scores", "self_scores", "co_wmv", "co_mv", "self_mv", "truth",
        "mv_gated_intuitor", "co_mv_gated_intuitor", "mixed_reward",
    }:
        raise ValueError(
            f"Unknown reward_method: {reward_method}. Expected one of: sum_scores, self_scores, co_wmv, "
            f"co_mv, self_mv, truth, mv_gated_intuitor, co_mv_gated_intuitor, mixed_reward."
        )

    # Load raw scores from cache — handles both per-model (N,G) and cross-model (N,2G) formats.
    scores_a_tensor, scores_b_tensor = _reconstruct_full_scores(
        scores_a, scores_b, num_unique_prompts, num_generations_per_model,
        reward_method, device=device,
    )

    # --- Compute per-prompt indices (dynamic or static) ---
    if use_dynamic_selection:
        ranking_scores = reward_weight_a * scores_a_tensor + reward_weight_b * scores_b_tensor
        per_prompt_indices = _compute_per_prompt_indices(
            learn_cfg, train_model, num_generations_per_model,
            ranking_scores, step,
        )
    else:
        per_prompt_indices = [selected_indices] * num_unique_prompts

    num_per_prompt = len(per_prompt_indices[0])

    # Prepare per-response rewards.
    #
    # - sum_scores: continuous reward = weight_a*score_a + weight_b*score_b
    # - self_scores: each model's completions rewarded only by that model's own score
    #   (A's completions → score_a, B's completions → score_b)
    # - co_wmv: normalize A/B scores, combine into WMV weights, vote pseudo-label, assign binary rewards (1/0)
    # - co_mv: unweighted majority voting (each answer = 1 vote), vote pseudo-label, assign binary rewards (1/0)
    # - mixed_reward: per-view GRPO adv, alpha blend, then batch z-score; no extra group norm downstream
    if reward_method in ("truth", "co_mv", "self_mv", "mv_gated_intuitor", "co_mv_gated_intuitor"):
        # Pre-computed rewards from generate_and_score (binary or gated).
        reward_scores_tensor = scores_a_tensor
        combined_weights_tensor = None
    elif reward_method == "self_scores":
        reward_scores_tensor = torch.zeros_like(scores_a_tensor)
        reward_scores_tensor[:, :num_generations_per_model] = scores_a_tensor[:, :num_generations_per_model]
        reward_scores_tensor[:, num_generations_per_model:] = scores_b_tensor[:, num_generations_per_model:]
        combined_weights_tensor = None
    elif reward_method == "sum_scores":
        from utils.normalization import minmax_normalize
        raw = reward_weight_a * scores_a_tensor + reward_weight_b * scores_b_tensor
        reward_scores_tensor = (
            minmax_normalize(raw, dim=1) if sum_scores_minmax else raw
        )
        combined_weights_tensor = None
    elif reward_method == "mixed_reward":
        from utils.normalization import minmax_normalize as _mm_norm
        from utils.reward import compute_rewards as _compute_rewards_dispatch

        raw_intuitor = reward_weight_a * scores_a_tensor + reward_weight_b * scores_b_tensor
        intuitor_full = (
            _mm_norm(raw_intuitor, dim=1) if sum_scores_minmax else raw_intuitor
        )
        mv_a, mv_b = _compute_rewards_dispatch(
            "co_mv",
            responses_a,
            responses_b,
            num_unique_prompts,
            num_generations_per_model,
            step=int(step),
        )
        dev = intuitor_full.device
        dt = torch.float32
        mv_full = torch.cat(
            [mv_a.to(device=dev, dtype=dt), mv_b.to(device=dev, dtype=dt)],
            dim=1,
        )
        _arm = str(mixed_reward_alpha_mode).strip().lower()
        reward_scores_tensor = compute_mixed_reward_advantages(
            mv_full,
            intuitor_full.to(dtype=dt),
            alpha=float(mixed_reward_alpha),
            alpha_mode=str(mixed_reward_alpha_mode),
            alpha_zscore_mu=float(mixed_reward_alpha_zscore_mu),
            alpha_zscore_sigma=float(mixed_reward_alpha_zscore_sigma),
            alpha_zscore_lower=float(mixed_reward_alpha_zscore_lower),
            alpha_zscore_upper=float(mixed_reward_alpha_zscore_upper),
            corr_weight_mode=str(mixed_reward_corr_weight_mode),
            corr_weight_threshold=float(mixed_reward_corr_weight_threshold),
            corr_weight_exponent=float(mixed_reward_corr_weight_exponent),
            final_normalize=True,
            num_generations_per_model=(
                int(num_generations_per_model)
                if _arm in (
                    "adaptive_sc",
                    "adaptive_sc_zscore",
                    "reliability_sc_corr",
                    "prism",
                    "ettrl",
                )
                else None
            ),
        )
        combined_weights_tensor = None
    elif reward_method == "co_wmv":
        norm_a = z_score_normalize_tensor(scores_a_tensor, mu=5.0, sigma=1.0, clip_threshold=3.0, dim=1)
        norm_b = z_score_normalize_tensor(scores_b_tensor, mu=5.0, sigma=1.0, clip_threshold=3.0, dim=1)
        combined_weights_tensor = norm_a + norm_b  # (num_unique_prompts, total_per_prompt)
        reward_scores_tensor = None
    else:
        raise ValueError(f"Unknown reward_method: {reward_method}")
    
    # Prepare data for selected indices
    selected_prompt_ids_list = []
    selected_prompt_mask_list = []
    selected_completion_ids_list = []
    selected_rewards_list = []
    # Debug (rank0): keep aligned text/meta for the first few flattened samples.
    selected_debug_samples: List[Dict[str, Any]] = []
    
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    from tqdm import tqdm
    for i in tqdm(range(num_unique_prompts), desc="Preparing training data", unit="prompt"):
        prompt_text = unique_prompts[i]
        
        # Tokenize prompt (HACRL-style postprocess stack, in-repo adaptation).
        if max_prompt_length is not None and int(max_prompt_length) > 0:
            prompt_id_2d, prompt_mask_2d = tokenize_and_postprocess_data(
                prompt=str(prompt_text),
                tokenizer=tokenizer,
                max_length=int(max_prompt_length),
                pad_token_id=int(pad_token_id),
                left_pad=True,
                truncation="left",
            )
            prompt_id = prompt_id_2d[0]
            prompt_mask = prompt_mask_2d[0]
        else:
            prompt_inputs = tokenizer(
                text=prompt_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_id = prompt_inputs["input_ids"][0]
            prompt_mask = prompt_inputs["attention_mask"][0]
        
        # Get responses for this prompt
        start_a = i * num_generations_per_model
        end_a = (i + 1) * num_generations_per_model
        start_b = i * num_generations_per_model
        end_b = (i + 1) * num_generations_per_model
        
        responses_a_batch = responses_a[start_a:end_a]
        responses_b_batch = responses_b[start_b:end_b]
        finish_a_batch = (
            finish_a[start_a:end_a] if (finish_a is not None and len(finish_a) >= end_a) else [None] * len(responses_a_batch)
        )
        finish_b_batch = (
            finish_b[start_b:end_b] if (finish_b is not None and len(finish_b) >= end_b) else [None] * len(responses_b_batch)
        )
        
        # Combine: [A's 4, B's 4] = 8 total
        combined_responses = responses_a_batch + responses_b_batch
        combined_finish = list(finish_a_batch) + list(finish_b_batch)

        # Compute rewards for this prompt (length = total_per_prompt)
        if reward_method in ("truth", "co_mv", "self_mv", "sum_scores", "self_scores", "mixed_reward", "mv_gated_intuitor", "co_mv_gated_intuitor"):
            rewards_for_prompt: List[float] = reward_scores_tensor[i].tolist()
        elif reward_method == "co_wmv":
            weights = combined_weights_tensor[i].tolist()
            rng = random.Random((int(step) + 1) * 1_000_000 + int(i))
            rewards_for_prompt, _vote = rewards_from_wmv_pseudolabel(combined_responses, weights, rng=rng)
        
        # Select responses based on per-prompt indices
        for j in per_prompt_indices[i]:
            response_text = combined_responses[j]
            finish_reason = combined_finish[j] if j < len(combined_finish) else None
            # vLLM OpenAI-style finish_reason is "length" when max_tokens cap is hit.
            is_truncated = (str(finish_reason) == "length") if finish_reason is not None else None
            
            # Tokenize response
            response_inputs = tokenizer(
                text=response_text,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
            )
            completion_id = response_inputs["input_ids"][0]
            # Cap completion length to match generation-time `max_tokens` and avoid huge sequences.
            if max_completion_length is not None and int(max_completion_length) > 0:
                mcl = int(max_completion_length)
                if completion_id.size(0) > mcl:
                    completion_id = completion_id[:mcl]

            # Text-tokenized completions do not include EOS by default.
            # To approximate vLLM completion_ids, append EOS only when the completion is not truncated.
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("Tokenizer must have eos_token_id to build completion ids (open-r1 semantics).")
            if is_truncated is False:
                completion_id = append_eos_1d(completion_id, eos_token_id=int(eos_id))
            elif is_truncated is True:
                # Do not append EOS; completion hit the length cap.
                pass
            else:
                # Fallback heuristic when finish_reason is unavailable:
                if max_completion_length is None or completion_id.size(0) < int(max_completion_length):
                    completion_id = append_eos_1d(completion_id, eos_token_id=int(eos_id))

            # Ensure completion (including EOS if appended) never exceeds max_completion_length.
            if max_completion_length is not None and int(max_completion_length) > 0:
                mcl2 = int(max_completion_length)
                if completion_id.size(0) > mcl2:
                    completion_id = completion_id[:mcl2]

            # Truncate to the first EOS (inclusive), matching build_left_padded_prompt_completion_batch.
            # Without this, tokens after EOS inflate target_len beyond logits_to_keep_cache,
            # forcing _align_cached_logps to zero-pad and breaking the KL computation.
            eos_id_int = int(eos_id)
            comp_list = completion_id.tolist()
            if eos_id_int in comp_list:
                first_eos = comp_list.index(eos_id_int)
                completion_id = completion_id[: first_eos + 1]

            # Get reward
            reward = torch.tensor(float(rewards_for_prompt[j]), dtype=torch.float32)
            
            selected_prompt_ids_list.append(prompt_id)
            selected_prompt_mask_list.append(prompt_mask)
            selected_completion_ids_list.append(completion_id)
            selected_rewards_list.append(reward)

            # Build debug sample metadata aligned with the flattened training row order.
            # We record it for all rows but will only write the first N later (rank0 only).
            src = "a" if int(j) < int(num_generations_per_model) else "b"
            debug_rec: Dict[str, Any] = {
                "prompt_index": int(i),
                "selected_j": int(j),
                "source_model": str(src),
                "finish_reason": (str(finish_reason) if finish_reason is not None else None),
                "is_truncated": (bool(is_truncated) if is_truncated is not None else None),
                # Text fields requested by the user.
                "prompt_text": str(prompt_text),
                "completion_text": str(response_text),
            }
            selected_debug_samples.append(debug_rec)
    
    # Pad and stack.
    # CRITICAL: prompts must be LEFT-padded to match build_left_padded_prompt_completion_batch
    # (used during logps computation). Right-padding prompts would shift RoPE position_ids,
    # causing training-time logps to diverge from cached old/ref logps.
    selected_prompt_ids = pad(selected_prompt_ids_list, padding_value=pad_token_id, padding_side="left")
    selected_prompt_mask = pad(selected_prompt_mask_list, padding_value=0, padding_side="left")
    selected_completion_ids = pad(selected_completion_ids_list, padding_value=pad_token_id)
    selected_rewards = torch.stack(selected_rewards_list)
    
    # IMPORTANT: Keep tensors on CPU.
    #
    # In the original GRPO/Intuitor trainer flow, the dataset stays on CPU and the dataloader (prepared by accelerate)
    # moves micro-batches to GPU. Moving the entire precomputed training batch to GPU here can easily consume tens of
    # GBs and cause OOM long before we even compute logps/loss.
    #
    # `device` is kept only for backward compatibility; we intentionally do not move data here.
    
    # Create completion mask
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must have eos_token_id to build completion mask (open-r1 semantics).")
    selected_completion_mask = get_response_mask(
        selected_completion_ids,
        eos_token=int(eos_token_id),
        dtype=torch.int64,
    )
    # Keep only true completion tokens; without this, rows without EOS may include right-pad positions.
    completion_lengths = torch.tensor(
        [int(x.size(0)) for x in selected_completion_ids_list],
        dtype=torch.long,
        device=selected_completion_ids.device,
    )
    seq_idx = torch.arange(selected_completion_ids.size(1), device=selected_completion_ids.device).unsqueeze(0)
    selected_completion_mask = selected_completion_mask * (seq_idx < completion_lengths.unsqueeze(1)).to(torch.int64)
    
    # NOTE: We intentionally do NOT drop truncated completions (those without EOS).
    # Truncated completions still carry useful training signal under a length cap.
    # `mask_truncated_completions` is kept for backward compatibility but has no effect here.
    
    # Compute advantages (group-wise normalization), except mixed_reward where
    # batch-level z-score was already applied in compute_mixed_reward_advantages.
    if reward_method == "mixed_reward":
        advantages = selected_rewards
    else:
        num_groups = len(selected_rewards) // num_per_prompt
        rewards_grouped = selected_rewards.view(num_groups, num_per_prompt)
        mean_grouped_rewards = rewards_grouped.mean(dim=1)
        std_grouped_rewards = rewards_grouped.std(dim=1)

        mean_grouped_rewards_expanded = mean_grouped_rewards.repeat_interleave(num_per_prompt, dim=0)
        std_grouped_rewards_expanded = std_grouped_rewards.repeat_interleave(num_per_prompt, dim=0)

        advantages = selected_rewards - mean_grouped_rewards_expanded
        if scale_rewards:
            advantages = advantages / (std_grouped_rewards_expanded + 1e-4)
    
    # Load old_logps from cache if available.
    # Logps are now per-model own responses only: (num_prompts * N, logits_to_keep).
    # Model A's logps cover indices 0..N-1 in the combined space.
    # Model B's logps cover indices N..2N-1 in the combined space.
    old_per_token_logps = None
    ref_per_token_logps = None
    
    if train_model == "a":
        is_valid_logps, old_logps, ref_logps = cache.load_logps_a(step)
    else:
        is_valid_logps, old_logps, ref_logps = cache.load_logps_b(step)

    # Determine the offset for this model's own responses in the combined 2N space.
    own_offset = 0 if train_model == "a" else num_generations_per_model

    # Map selected indices to own-model-only indices.
    # Only indices within [own_offset, own_offset+N) have logps available.
    own_logps_indices = []
    for pp_indices in per_prompt_indices:
        row_indices = []
        for j in pp_indices:
            if own_offset <= j < own_offset + num_generations_per_model:
                row_indices.append(j - own_offset)
            else:
                row_indices.append(-1)
        own_logps_indices.append(row_indices)

    if is_valid_logps and old_logps is not None:
        old_logps_tensor = torch.tensor(old_logps, device=selected_completion_ids.device)
        old_logps_reshaped = old_logps_tensor.view(num_unique_prompts, num_generations_per_model, -1)
        logps_width = old_logps_reshaped.size(-1)
        selected_old_list = []
        for i, row_idx in enumerate(own_logps_indices):
            for j_own in row_idx:
                if j_own >= 0:
                    selected_old_list.append(old_logps_reshaped[i, j_own])
                else:
                    selected_old_list.append(torch.zeros(logps_width, device=selected_completion_ids.device))
        if selected_old_list:
            old_per_token_logps = torch.stack(selected_old_list, dim=0)
    
    if is_valid_logps and ref_logps is not None:
        ref_logps_tensor = torch.tensor(ref_logps, device=selected_completion_ids.device)
        ref_logps_reshaped = ref_logps_tensor.view(num_unique_prompts, num_generations_per_model, -1)
        logps_width = ref_logps_reshaped.size(-1)
        selected_ref_list = []
        for i, row_idx in enumerate(own_logps_indices):
            for j_own in row_idx:
                if j_own >= 0:
                    selected_ref_list.append(ref_logps_reshaped[i, j_own])
                else:
                    selected_ref_list.append(torch.zeros(logps_width, device=selected_completion_ids.device))
        if selected_ref_list:
            ref_per_token_logps = torch.stack(selected_ref_list, dim=0)

    # Align cached logps length to the padded completion length used for training.
    #
    # Align cached logps length to the padded completion length used for training.
    #
    # With consistent EOS truncation (both sides truncate to first EOS), training's
    # target_len should always be <= logits_to_keep_cache (selected is a subset of all).
    # So alignment is a simple prefix truncation — never zero-padding.
    target_len = int(selected_completion_ids.size(1))

    def _align_cached_logps(lp: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if lp is None:
            return None
        if lp.dim() != 2:
            return lp
        if lp.size(1) == target_len:
            return lp
        if lp.size(1) > target_len:
            return lp[:, :target_len]
        # This should not happen with consistent EOS truncation. Log a warning
        # but don't crash — zero-pad as a last resort.
        logger.warning(
            f"Cached logps length ({lp.size(1)}) < training completion length ({target_len}). "
            "This indicates inconsistent tokenization between generate_and_score and training. "
            "Zero-padding to match, but KL values on padded positions may be unreliable. "
            "Consider setting CO_MASK_PAD_BEFORE_EXP=1 or regenerating the cache."
        )
        pad = lp.new_zeros((lp.size(0), target_len - lp.size(1)))
        return torch.cat([lp, pad], dim=1)

    old_per_token_logps = _align_cached_logps(old_per_token_logps)
    ref_per_token_logps = _align_cached_logps(ref_per_token_logps)
    tc_check_logps_matrix(
        where="co.train.prepare_cache.old_logps_selected",
        logps=old_per_token_logps,
        expected_rows=len(selected_rewards_list) if old_per_token_logps is not None else None,
    )
    tc_check_logps_matrix(
        where="co.train.prepare_cache.ref_logps_selected",
        logps=ref_per_token_logps,
        expected_rows=len(selected_rewards_list) if ref_per_token_logps is not None else None,
    )

    # Debug: dump fully-assembled samples (prompt+template+completion+padding) for inspection.
    try:
        dump_dir = os.environ.get("CO_DEBUG_DATA_DIR", os.path.join(os.getcwd(), "outputs", "debug_data"))
        _co_dump_prepare_full_samples(
            dump_dir=str(dump_dir),
            step=int(step),
            train_model=str(train_model),
            tokenizer=tokenizer,
            samples=selected_debug_samples,
            prompt_ids=selected_prompt_ids,
            prompt_mask=selected_prompt_mask,
            completion_ids=selected_completion_ids,
            completion_mask=selected_completion_mask,
            max_items=10,
        )
    except Exception:
        pass

    tc_check_training_batch(
        where="co.train.prepare_cache.output_batch",
        prompt_ids=selected_prompt_ids,
        prompt_mask=selected_prompt_mask,
        completion_ids=selected_completion_ids,
        completion_mask=selected_completion_mask,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
        ref_per_token_logps=ref_per_token_logps,
        require_old_logps=False,
        require_ref_logps=False,
    )
    
    return {
        "prompt_ids": selected_prompt_ids,
        "prompt_mask": selected_prompt_mask,
        "completion_ids": selected_completion_ids,
        "completion_mask": selected_completion_mask,
        "advantages": advantages,
        "old_per_token_logps": old_per_token_logps,
        "ref_per_token_logps": ref_per_token_logps,
    }


def compute_grpo_loss(
    model,
    inputs: Dict[str, Any],
    temperature: float = 1.0,
    epsilon_low: float = 0.1,
    epsilon_high: Optional[float] = None,
    beta: float = 0.0,
    loss_type: str = "grpo",
    max_completion_length: int = 512,
    num_iterations: int = 1,
) -> torch.Tensor:
    """
    Compute GRPO loss.
    
    Args:
        model: Model to compute loss with
        inputs: Training data dict
        temperature: Temperature for logits
        epsilon_low: Lower epsilon for clipping
        epsilon_high: Upper epsilon for clipping
        beta: KL penalty coefficient
        loss_type: Loss type ("grpo", "bnpo", "dr_grpo")
        max_completion_length: Maximum completion length
        num_iterations: Number of iterations
    
    Returns:
        Loss tensor
    """
    if epsilon_high is None:
        epsilon_high = epsilon_low
    
    prompt_ids = inputs["prompt_ids"]
    prompt_mask = inputs["prompt_mask"]
    completion_ids = inputs["completion_ids"]
    completion_mask = inputs["completion_mask"]
    advantages = inputs["advantages"]
    old_per_token_logps = inputs.get("old_per_token_logps")
    ref_per_token_logps = inputs.get("ref_per_token_logps")

    # IMPORTANT: Cap completion length to control memory.
    # Without this, `logits_to_keep` can become very large (e.g., thousands of tokens),
    # which makes the logits tensor (B, L_keep, V) explode in memory and OOM.
    if max_completion_length is not None and completion_ids.size(1) > max_completion_length:
        completion_ids = completion_ids[:, :max_completion_length]
        completion_mask = completion_mask[:, :max_completion_length]

    # Ensure cached logps (if present) match the effective completion length.
    logits_to_keep = int(completion_ids.size(1))

    def _align_logps_window(lp: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if lp is None:
            return None
        if lp.dim() != 2:
            return lp
        if lp.size(1) == logits_to_keep:
            return lp
        if lp.size(1) > logits_to_keep:
            # `prepare_training_data_from_cache` left-aligns cached logps to match right-padded `completion_ids`.
            # So when we further cap completion length, we must keep the *prefix*.
            return lp[:, :logits_to_keep]
        pad = lp.new_zeros((lp.size(0), logits_to_keep - lp.size(1)))
        return torch.cat([lp, pad], dim=1)

    old_per_token_logps = _align_logps_window(old_per_token_logps)
    ref_per_token_logps = _align_logps_window(ref_per_token_logps)
    
    # Concatenate prompt + completion
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    # Match open-r1 GRPO semantics:
    # attention_mask includes completion padding (0s after the first EOS / pad).
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    # Optional invariants to ensure we match open-r1 semantics (enable with CO_ASSERT_INVARIANTS=1).
    assert_prompt_completion_shapes(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        where="src.co_intuitor_train.compute_grpo_loss",
    )
    tc_check_padding_semantics(
        where="co.loss.grpo.padding_semantics",
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
    )
    assert_attention_mask_matches_concat(
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
        attention_mask=attention_mask,
        where="src.co_intuitor_train.compute_grpo_loss",
        allow_completion_padding_ones=False,
    )
    
    # Log input sizes for debugging memory issues
    if torch.cuda.is_available():
        logger.debug(f"Input sizes - batch: {input_ids.size(0)}, seq_len: {input_ids.size(1)}, logits_to_keep: {logits_to_keep}")
        logger.debug(f"GPU memory before logps: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Avoid empty_cache by default (can increase fragmentation under ZeRO-3).
    # Enable explicitly if you need it:
    #   export CO_EMPTY_CACHE_LOGPS=1
    if torch.cuda.is_available():
        v = os.environ.get("CO_EMPTY_CACHE_LOGPS", "0").strip().lower()
        if v not in ("", "0", "false", "no", "off", "none"):
            torch.cuda.empty_cache()
    
    # Compute per-token logps with minimal batch size for DeepSpeed ZeRO-3
    # DeepSpeed ZeRO-3 needs to gather parameters, which uses significant memory
    per_token_logps = get_per_token_logps(
        model, input_ids, attention_mask, logits_to_keep,
        temperature=temperature, batch_size=4
    )

    if co_env_flag("CO_DEBUG_NAN", "0"):
        if not torch.isfinite(per_token_logps).all():
            _co_dump_nan_event(
                "per_token_logps_nonfinite",
                {
                    "prompt_ids": prompt_ids,
                    "prompt_mask": prompt_mask,
                    "completion_ids": completion_ids,
                    "completion_mask": completion_mask,
                    "advantages": advantages,
                    "old_per_token_logps": old_per_token_logps,
                    "ref_per_token_logps": ref_per_token_logps,
                    "per_token_logps": per_token_logps,
                },
                extra={"logits_to_keep": int(logits_to_keep), "beta": float(beta)},
            )
            if co_env_flag("CO_DEBUG_NAN_RAISE", "1"):
                raise RuntimeError("Found non-finite per_token_logps. See CO_DEBUG_NAN_DIR dumps.")
    
    # Compute KL divergence if needed
    per_token_kl = None
    if beta != 0.0 and ref_per_token_logps is not None:
        if co_env_flag("CO_DEBUG_NAN", "0"):
            if not torch.isfinite(ref_per_token_logps).all():
                _co_dump_nan_event(
                    "ref_per_token_logps_nonfinite",
                    {
                        "ref_per_token_logps": ref_per_token_logps,
                        "per_token_logps": per_token_logps,
                    },
                    extra={"beta": float(beta)},
                )
                if co_env_flag("CO_DEBUG_NAN_RAISE", "1"):
                    raise RuntimeError("Found non-finite ref_per_token_logps. See CO_DEBUG_NAN_DIR dumps.")
            try:
                # Align with open-r1 semantics: KL should only be applied on valid completion tokens.
                # Also prevents exp() overflow on padded positions from propagating NaNs via inf*0.
                delta = (ref_per_token_logps - per_token_logps).detach()
                if completion_mask is not None:
                    delta = delta.masked_fill(completion_mask.to(torch.bool) == 0, 0.0)
                if not torch.isfinite(delta).all():
                    _co_dump_nan_event(
                        "kl_delta_nonfinite",
                        {"delta": delta, "ref_per_token_logps": ref_per_token_logps, "per_token_logps": per_token_logps},
                        extra={"beta": float(beta)},
                    )
                    if co_env_flag("CO_DEBUG_NAN_RAISE", "1"):
                        raise RuntimeError("Found non-finite (ref_logps - logps). See CO_DEBUG_NAN_DIR dumps.")
                # Heuristic: exp(x) overflows fp32 around ~88; bf16/fp16 can be even more sensitive.
                max_delta = float(delta.max().item()) if delta.numel() > 0 else 0.0
                if max_delta > 80.0:
                    # Pinpoint where the large delta happens, and check if a constant column shift explains it.
                    try:
                        flat_idx = int(delta.reshape(-1).argmax().item())
                        b = flat_idx // int(delta.size(1))
                        t = flat_idx % int(delta.size(1))
                        tok_id = int(completion_ids[b, t].detach().to("cpu").item())
                        m = int(completion_mask[b, t].detach().to("cpu").item()) if completion_mask is not None else None
                        ref_v = float(ref_per_token_logps[b, t].detach().to("cpu").item())
                        lp_v = float(per_token_logps[b, t].detach().to("cpu").item())
                        d_v = float(delta[b, t].detach().to("cpu").item())
                        # Small token-id window for manual inspection.
                        w0 = max(0, t - 8)
                        w1 = min(int(delta.size(1)), t + 9)
                        tok_window = completion_ids[b, w0:w1].detach().to("cpu").tolist()
                        mask_window = (
                            completion_mask[b, w0:w1].detach().to("cpu").tolist()
                            if completion_mask is not None
                            else None
                        )
                        ref_lp_window = (
                            ref_per_token_logps[b, w0:w1].detach().to("cpu").to(torch.float32).tolist()
                            if ref_per_token_logps is not None
                            else None
                        )
                        lp_window = per_token_logps[b, w0:w1].detach().to("cpu").to(torch.float32).tolist()
                        delta_window = delta[b, w0:w1].detach().to("cpu").to(torch.float32).tolist()
                        shift_report = _co_shift_mismatch_report(
                            ref_per_token_logps.detach().to(torch.float32),
                            per_token_logps.detach().to(torch.float32),
                            completion_mask if completion_mask is not None else torch.ones_like(delta, dtype=torch.long),
                            max_shift=int(os.environ.get("CO_DEBUG_NAN_SHIFT_SCAN", "16")),
                        )
                        delta_extra = {
                            "beta": float(beta),
                            "max_delta": max_delta,
                            "argmax": {"b": int(b), "t": int(t), "flat_idx": int(flat_idx)},
                            "at_argmax": {
                                "completion_token_id": tok_id,
                                "completion_mask": m,
                                "ref_logp": ref_v,
                                "logp": lp_v,
                                "delta": d_v,
                            },
                            "token_window": {
                                "t0": int(w0),
                                "t1": int(w1),
                                "ids": tok_window,
                                "mask": mask_window,
                                "ref_logp": ref_lp_window,
                                "logp": lp_window,
                                "delta": delta_window,
                            },
                            "shift_report": shift_report,
                        }
                    except Exception as _e:
                        delta_extra = {"beta": float(beta), "max_delta": max_delta, "extra_error": str(_e)}
                    _co_dump_nan_event(
                        "kl_delta_large",
                        {"delta": delta},
                        extra=delta_extra,
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
        # Compute KL exactly like open-r1 (no masking before exp, mask at aggregation time).
        #
        # This works because generate_and_score now uses attention_mask=0 on completion
        # pads (matching open-r1 / training semantics), so cached ref_logps and live logps
        # agree on pad positions and delta_kl stays small.
        #
        # Safety net for legacy caches where attention_mask was 1 on pads:
        #   export CO_MASK_PAD_BEFORE_EXP=1
        mask_pad_before_exp = co_env_flag("CO_MASK_PAD_BEFORE_EXP", "0")
        delta_kl = ref_per_token_logps - per_token_logps
        if mask_pad_before_exp:
            delta_kl = delta_kl.masked_fill(completion_mask.to(torch.bool) == 0, 0.0)
        # Match HACRL kl_penalty_forward: clamp input to [-20,20], output to [-10,10].
        delta_kl = torch.clamp(delta_kl, min=-20.0, max=20.0)
        per_token_kl = torch.clamp(torch.exp(delta_kl) - delta_kl - 1, min=-10.0, max=10.0)

        if co_env_flag("CO_DEBUG_NAN", "0"):
            if not torch.isfinite(per_token_kl).all():
                _co_dump_nan_event(
                    "per_token_kl_nonfinite",
                    {
                        "per_token_kl": per_token_kl,
                        "ref_per_token_logps": ref_per_token_logps,
                        "per_token_logps": per_token_logps,
                    },
                    extra={"beta": float(beta)},
                )
                if co_env_flag("CO_DEBUG_NAN_RAISE", "1"):
                    raise RuntimeError("Found non-finite per_token_kl. See CO_DEBUG_NAN_DIR dumps.")
    
    # Compute GRPO loss
    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()
    
    advantages_expanded = advantages.unsqueeze(1)  # (B, 1)

    # open-r1 semantics: compute coef_1 on the full padded window, then mask at aggregation time.
    # Safety net for legacy caches: export CO_MASK_PAD_BEFORE_EXP=1
    delta = (per_token_logps - old_per_token_logps)
    if co_env_flag("CO_MASK_PAD_BEFORE_EXP", "0"):
        delta = delta.masked_fill(completion_mask.to(torch.bool) == 0, 0.0)
    coef_1 = torch.exp(delta)

    if co_env_flag("CO_DEBUG_NAN", "0"):
        if not torch.isfinite(coef_1).all():
            _co_dump_nan_event(
                "coef_1_nonfinite",
                {
                    "coef_1": coef_1,
                    "per_token_logps": per_token_logps,
                    "old_per_token_logps": old_per_token_logps,
                    "advantages": advantages,
                },
                extra={"epsilon_low": float(epsilon_low), "epsilon_high": float(epsilon_high)},
            )
            if co_env_flag("CO_DEBUG_NAN_RAISE", "1"):
                raise RuntimeError("Found non-finite exp(per_token_logps - old_logps). See CO_DEBUG_NAN_DIR dumps.")

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = coef_1 * advantages_expanded
    per_token_loss2 = coef_2 * advantages_expanded
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    
    if beta != 0.0 and per_token_kl is not None:
        per_token_loss = per_token_loss + beta * per_token_kl
    
    # Aggregate loss based on loss_type
    if loss_type == "grpo":
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if co_env_flag("CO_DEBUG_NAN", "0"):
        if not torch.isfinite(loss).all():
            _co_dump_nan_event(
                "loss_nonfinite",
                {
                    "loss": loss,
                    "advantages": advantages,
                    "completion_mask": completion_mask,
                    "per_token_logps": per_token_logps,
                    "old_per_token_logps": old_per_token_logps,
                    "ref_per_token_logps": ref_per_token_logps,
                    "per_token_kl": per_token_kl,
                    "coef_1": coef_1,
                },
                extra={"beta": float(beta), "loss_type": str(loss_type), "logits_to_keep": int(logits_to_keep)},
            )
            if co_env_flag("CO_DEBUG_NAN_RAISE", "1"):
                raise RuntimeError("Found non-finite loss. See CO_DEBUG_NAN_DIR dumps.")
    
    # Verify loss has gradients (for debugging)
    if not loss.requires_grad:
        logger.error(f"Loss does not require grad! This will cause backward() to fail. "
                    f"per_token_logps.requires_grad={per_token_logps.requires_grad}, "
                    f"per_token_loss.requires_grad={per_token_loss.requires_grad}")
        raise RuntimeError("Loss tensor does not require grad. Check that per_token_logps has gradients.")
    
    return loss


# ---------------------------------------------------------------------------
# Sequence-level helpers & GSPO loss
# ---------------------------------------------------------------------------

def _seq_level_logp(
    per_token_logps: torch.Tensor,
    mask: torch.Tensor,
    length_norm: str = "mean",
) -> torch.Tensor:
    """Aggregate per-token log probs into a per-sequence scalar.

    Args:
        per_token_logps: (B, L)
        mask: (B, L) binary, 1 for valid completion tokens
        length_norm: ``'mean'`` → sum/len, ``'sqrt_len'`` → sum/√len, ``'none'`` → sum
    Returns:
        (B,) aggregated log prob
    """
    masked_logps = per_token_logps * mask.to(per_token_logps.dtype)
    seq_sum = masked_logps.sum(dim=-1)
    seq_len = mask.sum(dim=-1).clamp(min=1.0)
    if length_norm == "mean":
        return seq_sum / seq_len
    elif length_norm == "sqrt_len":
        return seq_sum / seq_len.sqrt()
    return seq_sum


def compute_gspo_loss(
    model,
    inputs: Dict[str, Any],
    temperature: float = 1.0,
    epsilon_low: float = 0.1,
    epsilon_high: Optional[float] = None,
    beta: float = 0.0,
    max_completion_length: int = 512,
    length_norm: str = "mean",
    num_iterations: int = 1,
    is_reweight_alpha: float = 0.0,
    peer_epsilon_low: Optional[float] = None,
    peer_epsilon_high: Optional[float] = None,
    capability_weight: float = 1.0,
    mini_batch_idx: int = 0,
    gspo_surrogate_ratio_enabled: bool = True,
    peer_loss_uniform: bool = False,
) -> torch.Tensor:
    """Compute GSPO loss using a sequence-level exponential IS ratio.

    Supports mixed-batch mode (like HACRL): when ``model_source`` is present
    in *inputs*, self samples (model_source==0) and peer samples
    (model_source==1) are handled with different clipping and weighting:

    - Self (J_homo): standard PPO-clip with ``min(ratio*adv, clipped*adv)``
    - Peer (J_hete): direct clipped ratio (no ``min``), with capability weight
      and IS reweighting applied on the loss.

    When ``model_source`` is absent, falls back to single-phase behaviour.
    """
    if epsilon_high is None:
        epsilon_high = epsilon_low

    prompt_ids = inputs["prompt_ids"]
    prompt_mask = inputs["prompt_mask"]
    completion_ids = inputs["completion_ids"]
    completion_mask = inputs["completion_mask"]
    advantages = inputs["advantages"]
    old_per_token_logps = inputs.get("old_per_token_logps")
    ref_per_token_logps = inputs.get("ref_per_token_logps")
    old_seq_mean_logp = inputs.get("old_seq_mean_logp")
    model_source = inputs.get("model_source")
    old_logps_mask = inputs.get("old_logps_mask")

    if max_completion_length is not None and completion_ids.size(1) > max_completion_length:
        completion_ids = completion_ids[:, :max_completion_length]
        completion_mask = completion_mask[:, :max_completion_length]

    logits_to_keep = int(completion_ids.size(1))

    def _align(lp: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if lp is None or lp.dim() != 2:
            return lp
        if lp.size(1) == logits_to_keep:
            return lp
        if lp.size(1) > logits_to_keep:
            return lp[:, :logits_to_keep]
        return torch.cat([lp, lp.new_zeros(lp.size(0), logits_to_keep - lp.size(1))], dim=1)

    old_per_token_logps = _align(old_per_token_logps)
    ref_per_token_logps = _align(ref_per_token_logps)
    old_logps_mask = _align(old_logps_mask)
    assert_prompt_completion_shapes(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        where="src.co_intuitor_train.compute_gspo_loss",
    )
    tc_check_padding_semantics(
        where="co.loss.gspo.padding_semantics",
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
    )

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    per_token_logps = get_per_token_logps(
        model, input_ids, attention_mask, logits_to_keep,
        temperature=temperature, batch_size=4,
    )

    # KL penalty (token-level, low_var_kl / k3 estimator)
    # Matches HACRL kl_penalty_forward: clamp input to [-20,20], output to [-10,10].
    per_token_kl = None
    if beta != 0.0 and ref_per_token_logps is not None:
        delta_kl = ref_per_token_logps - per_token_logps
        if co_env_flag("CO_MASK_PAD_BEFORE_EXP", "0"):
            delta_kl = delta_kl.masked_fill(completion_mask.to(torch.bool) == 0, 0.0)
        delta_kl = torch.clamp(delta_kl, min=-20.0, max=20.0)
        per_token_kl = torch.clamp(torch.exp(delta_kl) - delta_kl - 1, min=-10.0, max=10.0)

    cmask_f = completion_mask.to(per_token_logps.dtype)
    seq_lengths = cmask_f.sum(dim=-1).clamp(min=1)

    if gspo_surrogate_ratio_enabled:
        # --- Per-token GSPO surrogate ratio (HACRL core_algos.compute_policy_loss_mapo_clip) ---
        #
        # s_{i,t}(θ) = sg[s_i(θ)] · π_θ(y_{i,t}) / sg[π_θ(y_{i,t})]
        # log(s_{i,t}) = log_prob_t − sg[log_prob_t] + sg[(1/|y|)(Σ log_prob − Σ old_log_prob)]
        #
        # Per-token surrogate ratios give correct per-token gradients and match the
        # GSPO paper (Eq. 4-5).  Loss is aggregated via seq-mean-token-mean.
        new_mean_logp = (per_token_logps * cmask_f).sum(dim=-1) / seq_lengths

        if old_per_token_logps is not None:
            if old_logps_mask is not None:
                old_mask_f = old_logps_mask.to(per_token_logps.dtype)
                old_seq_lengths = old_mask_f.sum(dim=-1).clamp(min=1)
                old_mean_logp = (old_per_token_logps * old_mask_f).sum(dim=-1) / old_seq_lengths
            else:
                old_mean_logp = (old_per_token_logps * cmask_f).sum(dim=-1) / seq_lengths
        elif old_seq_mean_logp is not None:
            old_mean_logp = old_seq_mean_logp
        else:
            old_mean_logp = new_mean_logp.detach()

        neg_approx_kl_seq = (new_mean_logp - old_mean_logp).detach().unsqueeze(-1)
        log_ratio_per_token = per_token_logps - per_token_logps.detach() + neg_approx_kl_seq
    else:
        # --- Standard per-token ratio (like GRPO): log(π/π_old) per token ---
        if old_per_token_logps is not None:
            log_ratio_per_token = per_token_logps - old_per_token_logps
        else:
            log_ratio_per_token = per_token_logps - per_token_logps.detach()

    log_ratio_per_token = torch.clamp(log_ratio_per_token, max=10.0)
    ratio = torch.exp(log_ratio_per_token)  # (B, L)

    adv_per_token = advantages.unsqueeze(-1) * cmask_f  # (B, L)
    tc_check_hacpo_mixed_batch(
        where="co.loss.gspo.hacpo_contract",
        model_source=model_source,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
        capability_weight=float(capability_weight),
        peer_epsilon_low=peer_epsilon_low,
        peer_epsilon_high=peer_epsilon_high,
    )

    # --- Per-token loss (split self vs peer if mixed batch) ---
    if model_source is not None:
        self_mask = (model_source == 0)
        peer_mask = (model_source == 1)
        pg_losses = torch.zeros_like(ratio)

        # Self (J_homo): standard PPO clip → max(loss1, loss2)
        if self_mask.any():
            s_ratio = ratio[self_mask]
            s_adv = adv_per_token[self_mask]
            s_clipped = torch.clamp(s_ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
            pg_losses[self_mask] = torch.maximum(-s_adv * s_ratio, -s_adv * s_clipped)

        # Peer (J_hete)
        if peer_mask.any():
            p_ratio = ratio[peer_mask]
            p_adv = adv_per_token[peer_mask]
            p_eps_low = peer_epsilon_low if peer_epsilon_low is not None else epsilon_low
            p_eps_high = peer_epsilon_high if peer_epsilon_high is not None else epsilon_high
            p_clipped = torch.clamp(p_ratio, 1.0 - p_eps_low, 1.0 + p_eps_high)
            cw = max(0.1, min(10.0, float(capability_weight)))
            if peer_loss_uniform:
                # PPO-clip (same structure as self), scaled by capability weight
                pg_losses[peer_mask] = torch.maximum(-p_adv * p_ratio, -p_adv * p_clipped) * cw
            else:
                # HACPO: direct clipped, with capability weight & IS reweight
                peer_pg = -p_adv * p_clipped * cw
                if is_reweight_alpha > 0.0:
                    peer_pg = peer_pg * p_clipped.detach().pow(is_reweight_alpha)
                pg_losses[peer_mask] = peer_pg
    else:
        # Single-phase fallback
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
        pg_losses = torch.maximum(-adv_per_token * ratio, -adv_per_token * clipped_ratio)

    # KL penalty at token level (folded before aggregation).
    # HACRL only applies KL to self (main-model) samples; peer samples are excluded.
    if beta != 0.0 and per_token_kl is not None:
        if model_source is not None:
            kl_mask = (model_source == 0).unsqueeze(-1).to(per_token_kl.dtype)
            pg_losses = pg_losses + beta * per_token_kl * kl_mask
        else:
            pg_losses = pg_losses + beta * per_token_kl

    # seq-mean-token-mean aggregation (matches HACRL agg_loss)
    seq_losses = (pg_losses * cmask_f).sum(dim=-1) / seq_lengths
    loss = seq_losses.mean()

    if not loss.requires_grad:
        raise RuntimeError("GSPO loss does not require grad. Check that per_token_logps has gradients.")
    return loss


def compute_policy_loss(
    model,
    inputs: Dict[str, Any],
    *,
    objective_mode: str = "grpo",
    temperature: float = 1.0,
    epsilon_low: float = 0.1,
    epsilon_high: Optional[float] = None,
    beta: float = 0.0,
    loss_type: str = "grpo",
    max_completion_length: int = 512,
    num_iterations: int = 1,
    length_norm: str = "mean",
    is_reweight_alpha: float = 0.0,
    peer_epsilon_low: Optional[float] = None,
    peer_epsilon_high: Optional[float] = None,
    capability_weight: float = 1.0,
    mini_batch_idx: int = 0,
    gspo_surrogate_ratio_enabled: bool = True,
    peer_loss_uniform: bool = False,
) -> torch.Tensor:
    """Unified policy-loss entry dispatching to GRPO or GSPO."""
    if objective_mode == "gspo":
        return compute_gspo_loss(
            model, inputs,
            temperature=temperature, epsilon_low=epsilon_low,
            epsilon_high=epsilon_high, beta=beta,
            max_completion_length=max_completion_length,
            length_norm=length_norm, num_iterations=num_iterations,
            is_reweight_alpha=is_reweight_alpha,
            peer_epsilon_low=peer_epsilon_low,
            peer_epsilon_high=peer_epsilon_high,
            capability_weight=capability_weight,
            mini_batch_idx=mini_batch_idx,
            gspo_surrogate_ratio_enabled=gspo_surrogate_ratio_enabled,
            peer_loss_uniform=peer_loss_uniform,
        )
    return compute_grpo_loss(
        model, inputs,
        temperature=temperature, epsilon_low=epsilon_low,
        epsilon_high=epsilon_high, beta=beta, loss_type=loss_type,
        max_completion_length=max_completion_length,
        num_iterations=num_iterations,
    )


# ---------------------------------------------------------------------------
# Capability weights & GSPO advantages
# ---------------------------------------------------------------------------

def compute_capability_weights(
    mean_rewards: Dict[str, float],
    model_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute the clipped capability ratio matrix ``w[k][j]``."""
    _, _, clipped_ratios = compute_capability_ratio_details(mean_rewards, model_ids)
    return clipped_ratios


def compute_capability_ratio_details(
    mean_rewards: Dict[str, float],
    model_ids: List[str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Compute capability ratio matrix ``w[k][j] = R_k / R_j``.

    Aligned with HACRL stabilization: floor each performance value at 1e-8,
    then clamp each pairwise ratio to [0.1, 10.0].
    """
    floored = {mid: max(float(mean_rewards[mid]), 1e-8) for mid in model_ids}
    raw_ratios: Dict[str, Dict[str, float]] = {}
    clipped_ratios: Dict[str, Dict[str, float]] = {}
    for k in model_ids:
        raw_ratios[k] = {}
        clipped_ratios[k] = {}
        for j in model_ids:
            ratio = floored[k] / floored[j]
            raw_ratios[k][j] = float(ratio)
            clipped_ratios[k][j] = max(0.1, min(10.0, ratio))
    return floored, raw_ratios, clipped_ratios


def compute_gspo_advantages(
    rewards_per_model: Dict[str, torch.Tensor],
    capability_weights: Dict[str, Dict[str, float]],
    target_model: str,
    num_generations: int,
    scale_rewards: bool = True,
    accuracy_based: bool = False,
) -> torch.Tensor:
    """Compute A^(k) with capability-adjusted baseline (Eq. 6–7).

    mu_hat^(k) = (1/(nG)) * sum_j sum_i  w^(k,j) * R(y^(j)_{t,i})
    A^(k)_{t,i} = (R(y^(k)_{t,i}) − mu_hat^(k)) / sigma_{t,joint}

    Fallback: when ``accuracy_based`` (truth rewards in [0,1]), use HACRL's
    original ``mu_hat > 1.0`` threshold.  Otherwise (unsupervised rewards with
    unbounded scale), fall back when ``mu_hat`` exceeds the target model's
    per-prompt max reward.

    Args:
        rewards_per_model: ``{model_id: (num_prompts, G)}``
        capability_weights: nested dict ``weights[k][j]``
        target_model: model id ``k``
        num_generations: ``G`` per model
        scale_rewards: if True, divide by sigma_joint
        accuracy_based: if True, rewards are accuracy in [0,1] and the
            HACRL ``mu_hat > 1.0`` fallback is used
    Returns:
        ``(num_prompts * G,)`` flattened advantage tensor
    """
    model_ids = list(rewards_per_model.keys())
    n = len(model_ids)
    G = num_generations
    k = target_model
    num_prompts = rewards_per_model[k].size(0)
    device = rewards_per_model[k].device

    mu_hat = torch.zeros(num_prompts, device=device)
    for j in model_ids:
        w_kj = capability_weights[k][j]
        mu_hat = mu_hat + w_kj * rewards_per_model[j].sum(dim=1)
    mu_hat = mu_hat / (n * G)

    if accuracy_based:
        fallback_threshold = torch.ones(num_prompts, device=device)
        threshold_label = "1.0"
    else:
        fallback_threshold = rewards_per_model[k].max(dim=1).values
        threshold_label = "own_max"
    fallback_mask = mu_hat > fallback_threshold
    if fallback_mask.any():
        own_mean = rewards_per_model[k].mean(dim=1)
        mu_hat = torch.where(fallback_mask, own_mean, mu_hat)
        logger.info(
            "GSPO advantage fallback: %d/%d prompts had mu_hat > %s, "
            "using target model's own mean instead",
            int(fallback_mask.sum().item()), num_prompts, threshold_label,
        )

    all_rewards = torch.cat([rewards_per_model[j] for j in model_ids], dim=1)
    sigma_joint = all_rewards.std(dim=1)

    R_k = rewards_per_model[k]
    advantages = R_k - mu_hat.unsqueeze(1)
    if scale_rewards:
        advantages = advantages / (sigma_joint.unsqueeze(1) + 1e-6)

    return advantages.view(-1)


def compute_unified_gspo_advantages(
    rewards_per_model: Dict[str, torch.Tensor],
    capability_weights: Dict[str, Dict[str, float]],
    train_model: str,
    num_generations: int,
    scale_rewards: bool = True,
    old_seq_ratio_self: Optional[torch.Tensor] = None,
    old_seq_ratio_peer: Optional[torch.Tensor] = None,
    accuracy_based: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HACRL-aligned advantage: single baseline for ALL samples.

    Matches HACRL ``compute_mapo_outcome_advantage``:
    1. Peer scores scaled by ``cap_weights[train_model][peer_model]``
       (= ``1/performance`` in HACRL, i.e. ``mean_R_train / mean_R_peer``)
    2. Single baseline ``mu_hat`` from ALL scaled scores, weighted by
       ``old_seq_ratio`` (defaults to ones when unavailable)
    3. Std from ALL original (unscaled) scores
    4. Same baseline applied to ALL samples (self AND peer)
    5. Fallback to train model's own mean if ``mu_hat > 1.0``
       (accuracy) or ``mu_hat > own_max`` (unsupervised)

    Args:
        rewards_per_model: ``{model_id: (num_prompts, G)}``
        capability_weights: nested dict ``weights[k][j]``
        train_model: model being trained (``"a"`` or ``"b"``)
        num_generations: ``G`` per model
        scale_rewards: if True, divide by ``sigma_joint``
        accuracy_based: if True, rewards are accuracy in [0,1] and the
            HACRL ``mu_hat > 1.0`` fallback is used
    Returns:
        ``(self_advantages, peer_advantages)`` each ``(num_prompts * G,)``
    """
    model_ids = list(rewards_per_model.keys())
    peer_model = [m for m in model_ids if m != train_model][0]
    G = num_generations
    num_prompts = rewards_per_model[train_model].size(0)
    device = rewards_per_model[train_model].device

    R_self = rewards_per_model[train_model]   # (P, G)
    R_peer = rewards_per_model[peer_model]    # (P, G)

    # Scale peer scores by capability weight (HACRL: 1/performance = main/aux)
    w_scale = capability_weights[train_model][peer_model]  # mean_R_train / mean_R_peer
    R_peer_scaled = R_peer * w_scale

    # HACRL: weighted baseline using old_seq_ratio.
    if old_seq_ratio_self is None:
        old_seq_ratio_self = torch.ones_like(R_self)
    if old_seq_ratio_peer is None:
        old_seq_ratio_peer = torch.ones_like(R_peer)
    old_seq_ratio_self = old_seq_ratio_self.to(device=device, dtype=R_self.dtype)
    old_seq_ratio_peer = old_seq_ratio_peer.to(device=device, dtype=R_peer.dtype)

    all_scaled = torch.cat([R_self, R_peer_scaled], dim=1)  # (P, 2G)
    all_weights = torch.cat([old_seq_ratio_self, old_seq_ratio_peer], dim=1)  # (P, 2G)
    denom = all_weights.sum(dim=1).clamp(min=1e-12)
    mu_hat = (all_scaled * all_weights).sum(dim=1) / denom

    if accuracy_based:
        fallback_threshold = torch.ones(num_prompts, device=device)
        threshold_label = "1.0"
    else:
        fallback_threshold = R_self.max(dim=1).values
        threshold_label = "own_max"
    fallback_mask = mu_hat > fallback_threshold
    if fallback_mask.any():
        own_mean = R_self.mean(dim=1)
        mu_hat = torch.where(fallback_mask, own_mean, mu_hat)
        logger.info(
            "Unified GSPO advantage fallback: %d/%d prompts had mu_hat > %s, "
            "using train model's own mean instead",
            int(fallback_mask.sum().item()), num_prompts, threshold_label,
        )

    # Std from original (unscaled) scores (matches HACRL id2original_score).
    all_original = torch.cat([R_self, R_peer], dim=1)  # (P, 2*G)
    sigma_joint = all_original.std(dim=1)

    # Same baseline for BOTH self and peer (raw scores, NOT scaled)
    A_self = R_self - mu_hat.unsqueeze(1)
    A_peer = R_peer - mu_hat.unsqueeze(1)
    if scale_rewards:
        A_self = A_self / (sigma_joint.unsqueeze(1) + 1e-6)
        A_peer = A_peer / (sigma_joint.unsqueeze(1) + 1e-6)

    return A_self.reshape(-1), A_peer.reshape(-1)


def _log_mixed_reward_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Log compact population statistics for a mixed-reward tensor."""
    if tensor.numel() == 0:
        logger.info("mixed_reward %s stats: empty", name)
        return
    logger.info(
        "mixed_reward %s stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
        name,
        float(tensor.mean().item()),
        float(tensor.std(unbiased=False).item()),
        float(tensor.min().item()),
        float(tensor.max().item()),
    )


def _capture_mixed_reward_diagnostic(
    diagnostics: Optional[MutableMapping[str, Any]],
    name: str,
    value: Any,
) -> None:
    """Store an immutable CPU copy of a mixed-reward diagnostic value."""
    if diagnostics is None:
        return
    if isinstance(value, torch.Tensor):
        diagnostics[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()
    else:
        diagnostics[name] = value


def _compute_sc_corr_reliability(
    mv_rewards: torch.Tensor,
    adv_mv: torch.Tensor,
    adv_int: torch.Tensor,
    *,
    sc_weight: float,
    corr_weight: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """Build prompt-level reliability from batch-standardized SC and correlation."""
    sc_coef = float(sc_weight)
    corr_coef = float(corr_weight)
    if sc_coef < 0.0 or corr_coef < 0.0:
        raise ValueError(
            "mixed_reward reliability weights must be non-negative, "
            f"got sc_weight={sc_coef}, corr_weight={corr_coef}"
        )
    if sc_coef == 0.0 and corr_coef == 0.0:
        raise ValueError("mixed_reward reliability weights cannot both be zero")

    raw_sc = mv_rewards.mean(dim=1, keepdim=True)
    raw_corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    z_sc = batch_zscore_normalize(raw_sc, eps=eps)
    z_corr = batch_zscore_normalize(raw_corr, eps=eps)
    # Deliberately do not divide by sc_coef + corr_coef.
    reliability = sc_coef * z_sc + corr_coef * z_corr

    result = {
        "raw_sc": raw_sc,
        "raw_corr": raw_corr,
        "z_sc": z_sc,
        "z_corr": z_corr,
        "reliability": reliability,
    }
    for name, tensor in result.items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"mixed_reward reliability produced non-finite {name}")
    return result


def _compute_legacy_corr_sample_weight(
    adv_mv: torch.Tensor,
    adv_int: torch.Tensor,
    *,
    corr_weight_mode: str,
    corr_weight_threshold: float,
    corr_weight_exponent: float,
    num_generations_per_model: Optional[int],
) -> Optional[torch.Tensor]:
    """Compute the legacy post-normalize correlation weight without applying it."""
    corr_mode = str(corr_weight_mode).strip().lower()
    if corr_mode == "none":
        return None
    if num_generations_per_model is not None:
        raise ValueError(
            "mixed_reward correlation weighting currently supports single-model mixed_reward only; "
            "co-training / multi-model layouts with num_generations_per_model are not implemented."
        )

    threshold = float(corr_weight_threshold)
    exponent = float(corr_weight_exponent)
    if not (0.0 <= threshold < 1.0):
        raise ValueError(
            f"mixed_reward correlation weighting requires 0 <= threshold < 1, got {threshold}"
        )
    if exponent <= 0.0:
        raise ValueError(
            f"mixed_reward correlation weighting requires exponent > 0, got {exponent}"
        )
    if corr_mode != "pearson_threshold_power":
        raise ValueError(f"Unknown mixed_reward correlation weighting mode: {corr_weight_mode!r}")

    prompt_adv_corr = rowwise_pearson_corrcoef(adv_mv, adv_int, dim=1, keepdim=True)
    prompt_adv_corr = torch.clamp(prompt_adv_corr, -1.0, 1.0)
    prompt_corr_weight = torch.zeros_like(prompt_adv_corr)
    active = prompt_adv_corr >= threshold
    if active.any():
        prompt_corr_weight = torch.where(
            active,
            ((prompt_adv_corr - threshold) / (1.0 - threshold)).pow(exponent),
            torch.zeros_like(prompt_adv_corr),
        )

    _log_mixed_reward_tensor_stats("prompt_adv_corr", prompt_adv_corr)
    _log_mixed_reward_tensor_stats("prompt_corr_weight", prompt_corr_weight)
    return prompt_corr_weight


def _validate_reliability_mapping(
    name: str,
    *,
    mu: float,
    sigma: float,
    lower: float,
    upper: float,
) -> None:
    if float(sigma) <= 0.0:
        raise ValueError(f"mixed_reward reliability {name} sigma must be positive")
    if float(lower) >= float(upper):
        raise ValueError(f"mixed_reward reliability {name} lower must be < upper")
    if not (float(lower) <= float(mu) <= float(upper)):
        raise ValueError(
            f"mixed_reward reliability {name} mu must be within [lower, upper], "
            f"got mu={mu}, lower={lower}, upper={upper}"
        )


def _validate_single_model_no_final_normalize_mode(
    mode: str,
    *,
    alpha: float,
    corr_weight_mode: str,
    final_normalize: bool,
    num_generations_per_model: Optional[int],
) -> None:
    """Validate the shared contract for PRISM and eTTRL."""
    if num_generations_per_model is not None:
        raise ValueError(
            f"{mode} currently supports single-model mixed_reward only; "
            "co-training / multi-model layouts are not implemented."
        )
    if abs(float(alpha) - 0.5) > 1e-12:
        raise ValueError(
            f"{mode} requires mixed_reward_alpha=0.5, got {alpha}"
        )
    if str(corr_weight_mode).strip().lower() != "none":
        raise ValueError(
            f"{mode} requires corr_weight_mode='none'; legacy correlation "
            "weighting cannot be stacked with this mode."
        )
    if bool(final_normalize):
        raise ValueError(
            f"{mode} requires mixed_reward_final_normalize=false"
        )


def _compute_ettrl_response_weights(
    intuitor_rewards: torch.Tensor,
    *,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """Compute eTTRL response weights from raw within-prompt self-certainty."""
    if bool((intuitor_rewards < 0).any().item()):
        raise ValueError("ettrl requires non-negative self-certainty rewards")

    group_mean = intuitor_rewards.mean(dim=1, keepdim=True)
    all_zero_group = (intuitor_rewards == 0).all(dim=1, keepdim=True)
    near_zero_nonempty = (group_mean <= float(eps)) & ~all_zero_group
    if bool(near_zero_nonempty.any().item()):
        raise ValueError(
            "ettrl encountered a non-empty self-certainty group with mean <= eps; "
            "refusing to create unstable response weights"
        )

    safe_mean = torch.where(all_zero_group, torch.ones_like(group_mean), group_mean)
    response_weight = intuitor_rewards / safe_mean
    response_weight = torch.where(
        all_zero_group.expand_as(response_weight),
        torch.ones_like(response_weight),
        response_weight,
    )
    if not bool(torch.isfinite(response_weight).all().item()):
        raise ValueError("ettrl produced non-finite response weights")

    return {
        "self_certainty_group_mean": group_mean,
        "ettrl_all_zero_group": all_zero_group.to(dtype=intuitor_rewards.dtype),
        "mv_advantage_weight": response_weight,
        "self_certainty_advantage_weight": torch.zeros_like(response_weight),
    }


def compute_mixed_reward_advantages(
    mv_rewards: torch.Tensor,
    intuitor_rewards: torch.Tensor,
    alpha: float = 0.5,
    alpha_mode: str = "fixed",
    alpha_zscore_mu: float = 0.65,
    alpha_zscore_sigma: float = 0.2,
    alpha_zscore_lower: float = 0.0,
    alpha_zscore_upper: float = 1.0,
    corr_weight_mode: str = "none",
    corr_weight_threshold: float = 0.1,
    corr_weight_exponent: float = 0.8,
    reliability_sc_weight: float = 0.6,
    reliability_corr_weight: float = 0.4,
    reliability_use_sample_weight: bool = True,
    reliability_use_advantage_weight: bool = True,
    reliability_sample_weight_mu: float = 1.0,
    reliability_sample_weight_sigma: float = 0.2,
    reliability_sample_weight_lower: float = 0.5,
    reliability_sample_weight_upper: float = 1.5,
    reliability_advantage_weight_mu: float = 0.5,
    reliability_advantage_weight_sigma: float = 0.2,
    reliability_advantage_weight_lower: float = 0.2,
    reliability_advantage_weight_upper: float = 0.8,
    final_normalize: bool = True,
    eps: float = 1e-6,
    num_generations_per_model: Optional[int] = None,
    diagnostics: Optional[MutableMapping[str, Any]] = None,
) -> torch.Tensor:
    """GRPO-normalize each reward view, then alpha-combine.

    Pipeline::

        adv_mv  = GRPO_group_norm(mv_rewards)
        adv_int = GRPO_group_norm(intuitor_rewards)
        combined = alpha * adv_mv + (1-alpha) * adv_int   [fixed mode]
        or per-element alpha from |adv_mv|/(|adv_mv|+|adv_int|+eps)   [adaptive]
        or alpha from self-consistency on MV pseudo-label   [adaptive_sc]
        or batch-zscored self-consistency with explicit bounds   [adaptive_sc_zscore]
        reliability_sc_corr mode:
            build prompt reliability from batch-zscored SC and signed Pearson correlation
            map reliability to advantage and sample weights
        prism mode: combined = adv_mv + adv_int
        ettrl mode: combined = (raw_intuitor / prompt_mean_intuitor) * adv_mv
        if final_normalize:
            combined = batch_zscore_normalize(combined)  # over all (P, G) elements
        if post_normalize_sample_weight is not None:
            combined = post_normalize_sample_weight * combined

    In ``adaptive_sc`` mode, ``alpha`` is the consistency score: for each prompt,
    the fraction of that model's completions whose extracted answer matches the
    MV pseudo-label (``mv_rewards`` is binary 0/1). Single-model: mean over all
    ``G`` columns. Co-training: ``num_generations_per_model=N`` with ``G=2N`` —
    first ``N`` columns use model A's SC score, last ``N`` use model B's.

    Args:
        mv_rewards: ``(P, G)`` majority-voting reward tensor.
        intuitor_rewards: ``(P, G)`` Intuitor (SCE-based) reward tensor.
        alpha: MV weight when ``alpha_mode=='fixed'``. Intuitor weight is ``1-alpha``.
        alpha_mode: ``'fixed'``, ``'adaptive'`` (magnitude blend), ``'adaptive_sc'`` (SC score),
            ``'adaptive_sc_zscore'`` (single-model only; batch-normalized SC score), or
            ``'reliability_sc_corr'`` (single-model only; reliability-derived weights),
            ``'prism'`` (single-model only; unit-weight advantage sum), or ``'ettrl'``
            (single-model only; self-certainty-weighted MV advantage).
        alpha_zscore_mu: Target mean for ``adaptive_sc_zscore`` alpha mapping.
        alpha_zscore_sigma: Target std for ``adaptive_sc_zscore`` alpha mapping.
        alpha_zscore_lower: Lower clipping bound for ``adaptive_sc_zscore`` alpha.
        alpha_zscore_upper: Upper clipping bound for ``adaptive_sc_zscore`` alpha.
        corr_weight_mode: Optional post-combine prompt weighting mode.
        corr_weight_threshold: Threshold delta for prompt correlation weighting.
        corr_weight_exponent: Exponent for prompt correlation weighting.
        final_normalize: Apply batch-level z-score to the combined advantages.
        eps: Numerical stability constant (passed to per-view GRPO and adaptive denom).
        num_generations_per_model: Only for ``adaptive_sc`` co-training: ``mv_rewards`` must
            be ``(P, 2N)`` with ``N`` equal to this value (A columns then B columns).
            ``adaptive_sc_zscore`` does not support this layout.

    Returns:
        ``(P, G)`` combined advantage tensor.
    """
    if mv_rewards.dim() != 2 or intuitor_rewards.dim() != 2:
        raise ValueError(
            "mixed_reward rewards must be 2D tensors shaped (P, G), "
            f"got {tuple(mv_rewards.shape)} and {tuple(intuitor_rewards.shape)}"
        )
    if mv_rewards.shape != intuitor_rewards.shape:
        raise ValueError(
            "mixed_reward reward tensors must have the same shape, "
            f"got {tuple(mv_rewards.shape)} and {tuple(intuitor_rewards.shape)}"
        )
    if mv_rewards.numel() == 0:
        raise ValueError("mixed_reward reward tensors must be non-empty")
    if not bool(torch.isfinite(mv_rewards).all().item()):
        raise ValueError("mixed_reward mv_rewards contains non-finite values")
    if not bool(torch.isfinite(intuitor_rewards).all().item()):
        raise ValueError("mixed_reward intuitor_rewards contains non-finite values")

    adv_mv = grpo_group_advantages(mv_rewards, eps=eps)
    adv_int = grpo_group_advantages(intuitor_rewards, eps=eps)
    post_normalize_sample_weight: Optional[torch.Tensor] = None
    _capture_mixed_reward_diagnostic(diagnostics, "mv_rewards", mv_rewards)
    _capture_mixed_reward_diagnostic(
        diagnostics, "self_certainty_rewards", intuitor_rewards
    )
    _capture_mixed_reward_diagnostic(diagnostics, "mv_advantages", adv_mv)
    _capture_mixed_reward_diagnostic(
        diagnostics, "self_certainty_advantages", adv_int
    )

    mode = str(alpha_mode).strip().lower()
    _capture_mixed_reward_diagnostic(diagnostics, "alpha_mode", mode)
    _capture_mixed_reward_diagnostic(
        diagnostics, "final_normalize", bool(final_normalize)
    )
    if mode in ("prism", "ettrl"):
        _validate_single_model_no_final_normalize_mode(
            mode,
            alpha=alpha,
            corr_weight_mode=corr_weight_mode,
            final_normalize=final_normalize,
            num_generations_per_model=num_generations_per_model,
        )
        if mode == "prism":
            mv_advantage_weight = torch.ones_like(adv_mv)
            self_certainty_advantage_weight = torch.ones_like(adv_int)
            self_certainty_group_mean = intuitor_rewards.mean(dim=1, keepdim=True)
        else:
            ettrl_tensors = _compute_ettrl_response_weights(
                intuitor_rewards,
                eps=eps,
            )
            mv_advantage_weight = ettrl_tensors["mv_advantage_weight"]
            self_certainty_advantage_weight = ettrl_tensors[
                "self_certainty_advantage_weight"
            ]
            self_certainty_group_mean = ettrl_tensors[
                "self_certainty_group_mean"
            ]
            _capture_mixed_reward_diagnostic(
                diagnostics,
                "ettrl_all_zero_group",
                ettrl_tensors["ettrl_all_zero_group"],
            )

        combined = (
            mv_advantage_weight * adv_mv
            + self_certainty_advantage_weight * adv_int
        )
        _capture_mixed_reward_diagnostic(
            diagnostics, "self_certainty_group_mean", self_certainty_group_mean
        )
        _capture_mixed_reward_diagnostic(
            diagnostics, "mv_advantage_weight", mv_advantage_weight
        )
        _capture_mixed_reward_diagnostic(
            diagnostics,
            "self_certainty_advantage_weight",
            self_certainty_advantage_weight,
        )
        _log_mixed_reward_tensor_stats(
            f"{mode}_mv_advantage_weight", mv_advantage_weight
        )
        _log_mixed_reward_tensor_stats(
            f"{mode}_self_certainty_advantage_weight",
            self_certainty_advantage_weight,
        )
    elif mode == "reliability_sc_corr":
        if num_generations_per_model is not None:
            raise ValueError(
                "reliability_sc_corr currently supports single-model mixed_reward only; "
                "co-training / multi-model layouts are not implemented."
            )
        if str(corr_weight_mode).strip().lower() != "none":
            raise ValueError(
                "reliability_sc_corr requires corr_weight_mode='none'; "
                "legacy correlation weighting cannot be stacked with reliability sample weighting."
            )
        if abs(float(alpha) - 0.5) > 1e-12:
            raise ValueError(
                "reliability_sc_corr requires mixed_reward_alpha=0.5, "
                f"got {alpha}"
            )

        use_sample_weight = bool(reliability_use_sample_weight)
        use_advantage_weight = bool(reliability_use_advantage_weight)
        reliability_tensors: Optional[Dict[str, torch.Tensor]] = None
        sample_weight: Optional[torch.Tensor] = None
        advantage_weight: Optional[torch.Tensor] = None
        if not use_sample_weight and not use_advantage_weight:
            # Exact alpha050 baseline: Reliability and its parameters are intentionally irrelevant.
            combined = 0.5 * adv_mv + 0.5 * adv_int
            # The baseline still records the candidate Reliability/weights for diagnostics,
            # but those values must never affect the combined advantage.
            if diagnostics is not None:
                _validate_reliability_mapping(
                    "sample_weight",
                    mu=reliability_sample_weight_mu,
                    sigma=reliability_sample_weight_sigma,
                    lower=reliability_sample_weight_lower,
                    upper=reliability_sample_weight_upper,
                )
                _validate_reliability_mapping(
                    "advantage_weight",
                    mu=reliability_advantage_weight_mu,
                    sigma=reliability_advantage_weight_sigma,
                    lower=reliability_advantage_weight_lower,
                    upper=reliability_advantage_weight_upper,
                )
                reliability_tensors = _compute_sc_corr_reliability(
                    mv_rewards,
                    adv_mv,
                    adv_int,
                    sc_weight=reliability_sc_weight,
                    corr_weight=reliability_corr_weight,
                    eps=eps,
                )
                reliability = reliability_tensors["reliability"]
                sample_weight = z_score_normalize_tensor_with_bounds(
                    reliability,
                    mu=float(reliability_sample_weight_mu),
                    sigma=float(reliability_sample_weight_sigma),
                    lower=float(reliability_sample_weight_lower),
                    upper=float(reliability_sample_weight_upper),
                    dim=0,
                )
                advantage_weight = z_score_normalize_tensor_with_bounds(
                    reliability,
                    mu=float(reliability_advantage_weight_mu),
                    sigma=float(reliability_advantage_weight_sigma),
                    lower=float(reliability_advantage_weight_lower),
                    upper=float(reliability_advantage_weight_upper),
                    dim=0,
                )
        else:
            _validate_reliability_mapping(
                "sample_weight",
                mu=reliability_sample_weight_mu,
                sigma=reliability_sample_weight_sigma,
                lower=reliability_sample_weight_lower,
                upper=reliability_sample_weight_upper,
            )
            _validate_reliability_mapping(
                "advantage_weight",
                mu=reliability_advantage_weight_mu,
                sigma=reliability_advantage_weight_sigma,
                lower=reliability_advantage_weight_lower,
                upper=reliability_advantage_weight_upper,
            )
            reliability_tensors = _compute_sc_corr_reliability(
                mv_rewards,
                adv_mv,
                adv_int,
                sc_weight=reliability_sc_weight,
                corr_weight=reliability_corr_weight,
                eps=eps,
            )
            reliability = reliability_tensors["reliability"]
            sample_weight = z_score_normalize_tensor_with_bounds(
                reliability,
                mu=float(reliability_sample_weight_mu),
                sigma=float(reliability_sample_weight_sigma),
                lower=float(reliability_sample_weight_lower),
                upper=float(reliability_sample_weight_upper),
                dim=0,
            )
            advantage_weight = z_score_normalize_tensor_with_bounds(
                reliability,
                mu=float(reliability_advantage_weight_mu),
                sigma=float(reliability_advantage_weight_sigma),
                lower=float(reliability_advantage_weight_lower),
                upper=float(reliability_advantage_weight_upper),
                dim=0,
            )
            alpha_per = (
                advantage_weight
                if use_advantage_weight
                else torch.full_like(advantage_weight, 0.5)
            )
            combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
            if use_sample_weight:
                post_normalize_sample_weight = sample_weight

        if reliability_tensors is not None:
            for name, tensor in reliability_tensors.items():
                _log_mixed_reward_tensor_stats(f"reliability_{name}", tensor)
                _capture_mixed_reward_diagnostic(diagnostics, name, tensor)
        if sample_weight is not None and advantage_weight is not None:
            _log_mixed_reward_tensor_stats("reliability_sample_weight", sample_weight)
            _log_mixed_reward_tensor_stats("reliability_advantage_weight", advantage_weight)
            applied_sample_weight = (
                sample_weight if use_sample_weight else torch.ones_like(sample_weight)
            )
            applied_advantage_weight = (
                advantage_weight
                if use_advantage_weight
                else torch.full_like(advantage_weight, 0.5)
            )
            _capture_mixed_reward_diagnostic(
                diagnostics, "computed_sample_weight", sample_weight
            )
            _capture_mixed_reward_diagnostic(
                diagnostics, "applied_sample_weight", applied_sample_weight
            )
            _capture_mixed_reward_diagnostic(
                diagnostics, "computed_advantage_weight", advantage_weight
            )
            _capture_mixed_reward_diagnostic(
                diagnostics, "applied_advantage_weight", applied_advantage_weight
            )
        _capture_mixed_reward_diagnostic(
            diagnostics,
            "diagnostics_only_reliability",
            not use_sample_weight and not use_advantage_weight,
        )
        _capture_mixed_reward_diagnostic(
            diagnostics, "use_sample_weight", use_sample_weight
        )
        _capture_mixed_reward_diagnostic(
            diagnostics, "use_advantage_weight", use_advantage_weight
        )
    elif mode == "adaptive_sc":
        P, G = mv_rewards.shape
        if num_generations_per_model is None:
            alpha_per = mv_rewards.mean(dim=1, keepdim=True)  # (P, 1) -> broadcast
        else:
            N = int(num_generations_per_model)
            if G != 2 * N:
                raise ValueError(
                    f"adaptive_sc mixed_reward with num_generations_per_model={N} "
                    f"expects mv_rewards.shape[1]=={2 * N}, got G={G}"
                )
            sc_a = mv_rewards[:, :N].mean(dim=1, keepdim=True)  # (P, 1)
            sc_b = mv_rewards[:, N:].mean(dim=1, keepdim=True)
            alpha_per = torch.cat(
                [sc_a.expand(-1, N), sc_b.expand(-1, N)], dim=1,
            )  # (P, 2N)
        combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
        logger.info(
            "mixed_reward adaptive_sc alpha (SC score, legacy GRPO path): "
            "mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(alpha_per.mean().item()),
            float(alpha_per.std().item()),
            float(alpha_per.min().item()),
            float(alpha_per.max().item()),
        )
    elif mode == "adaptive_sc_zscore":
        if num_generations_per_model is not None:
            raise ValueError(
                "adaptive_sc_zscore currently supports single-model mixed_reward only; "
                "co-training / multi-model layouts with num_generations_per_model are not implemented."
            )
        raw_sc = mv_rewards.mean(dim=1, keepdim=True)  # (P, 1)
        alpha_per = z_score_normalize_tensor_with_bounds(
            raw_sc,
            mu=float(alpha_zscore_mu),
            sigma=float(alpha_zscore_sigma),
            lower=float(alpha_zscore_lower),
            upper=float(alpha_zscore_upper),
            dim=0,
        )
        combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
        logger.info(
            "mixed_reward adaptive_sc_zscore raw_sc stats: "
            "mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(raw_sc.mean().item()),
            float(raw_sc.std(unbiased=False).item()),
            float(raw_sc.min().item()),
            float(raw_sc.max().item()),
        )
        logger.info(
            "mixed_reward adaptive_sc_zscore alpha stats: "
            "mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(alpha_per.mean().item()),
            float(alpha_per.std(unbiased=False).item()),
            float(alpha_per.min().item()),
            float(alpha_per.max().item()),
        )
    elif mode == "adaptive":
        abs_mv = adv_mv.abs()
        abs_int = adv_int.abs()
        denom = abs_mv + abs_int + eps
        alpha_per = abs_mv / denom
        combined = alpha_per * adv_mv + (1.0 - alpha_per) * adv_int
        logger.info(
            "mixed_reward adaptive alpha (magnitude, legacy GRPO path): "
            "mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(alpha_per.mean().item()),
            float(alpha_per.std().item()),
            float(alpha_per.min().item()),
            float(alpha_per.max().item()),
        )
    elif mode == "fixed":
        a = float(alpha)
        combined = a * adv_mv + (1.0 - a) * adv_int
    else:
        raise ValueError(f"Unknown mixed_reward alpha mode: {alpha_mode!r}")

    if mode != "reliability_sc_corr":
        post_normalize_sample_weight = _compute_legacy_corr_sample_weight(
            adv_mv,
            adv_int,
            corr_weight_mode=corr_weight_mode,
            corr_weight_threshold=corr_weight_threshold,
            corr_weight_exponent=corr_weight_exponent,
            num_generations_per_model=num_generations_per_model,
        )

    _log_mixed_reward_tensor_stats("combined_before_final_normalize", combined)
    _capture_mixed_reward_diagnostic(
        diagnostics, "combined_before_final_normalize", combined
    )

    if final_normalize:
        combined = batch_zscore_normalize(combined, eps=eps)
    _log_mixed_reward_tensor_stats("combined_after_final_normalize", combined)
    _capture_mixed_reward_diagnostic(
        diagnostics, "combined_after_final_normalize", combined
    )

    if post_normalize_sample_weight is not None:
        combined = post_normalize_sample_weight * combined
    _log_mixed_reward_tensor_stats("final_advantages", combined)
    _capture_mixed_reward_diagnostic(diagnostics, "final_advantages", combined)

    if not bool(torch.isfinite(combined).all().item()):
        raise ValueError("mixed_reward produced non-finite final advantages")

    return combined


def _compute_rewards_for_two_phase(
    cache: "CoIntuitorCache",
    step: int,
    num_unique_prompts: int,
    num_generations_per_model: int,
    reward_method: str,
    reward_weight_a: float,
    reward_weight_b: float,
    responses_a: List[str],
    responses_b: List[str],
    device: Optional[torch.device] = None,
    mixed_reward_alpha: float = 0.5,
    sum_scores_minmax: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute per-model reward tensors ``{model_id: (num_prompts, G)}`` using the existing reward pipeline."""
    total = num_generations_per_model * 2
    N = num_generations_per_model

    _, scores_a_raw = cache.load_scores_a(step)
    _, scores_b_raw = cache.load_scores_b(step)
    scores_a_t, scores_b_t = _reconstruct_full_scores(
        scores_a_raw, scores_b_raw, num_unique_prompts, num_generations_per_model,
        reward_method, device=device,
    )

    if reward_method in ("truth", "co_mv", "self_mv", "mv_gated_intuitor", "co_mv_gated_intuitor"):
        return {"a": scores_a_t[:, :N], "b": scores_a_t[:, N:]}

    if reward_method == "self_scores":
        return {"a": scores_a_t[:, :N], "b": scores_b_t[:, N:]}

    if reward_method == "sum_scores":
        from utils.normalization import minmax_normalize
        raw = reward_weight_a * scores_a_t + reward_weight_b * scores_b_t
        all_rewards = minmax_normalize(raw, dim=1) if sum_scores_minmax else raw
        return {"a": all_rewards[:, :N], "b": all_rewards[:, N:]}

    if reward_method == "mixed_reward":
        from utils.normalization import minmax_normalize
        from utils.reward import compute_rewards as _compute_rewards_dispatch

        sum_intuitor = reward_weight_a * scores_a_t + reward_weight_b * scores_b_t
        intuitor_mm = (
            minmax_normalize(sum_intuitor, dim=1) if sum_scores_minmax else sum_intuitor
        )  # [0,1] over 2N when sum_scores_minmax else raw scale
        mv_a, mv_b = _compute_rewards_dispatch(
            "co_mv",
            responses_a,
            responses_b,
            num_unique_prompts,
            N,
            step=int(step),
        )
        dev = sum_intuitor.device
        dt = torch.float32
        mv_full = torch.cat([mv_a.to(device=dev, dtype=dt), mv_b.to(device=dev, dtype=dt)], dim=1)
        mv_mm = minmax_normalize(mv_full, dim=1)  # [0,1] over 2N
        _alpha = float(mixed_reward_alpha)
        full = _alpha * mv_mm + (1.0 - _alpha) * intuitor_mm.to(dtype=dt)
        return {"a": full[:, :N], "b": full[:, N:]}

    # co_wmv: WMV voting at training time (needs SCE scores as weights).
    rewards_a_list: List[List[float]] = []
    rewards_b_list: List[List[float]] = []
    norm_a = z_score_normalize_tensor(scores_a_t, mu=5.0, sigma=1.0, clip_threshold=3.0, dim=1)
    norm_b = z_score_normalize_tensor(scores_b_t, mu=5.0, sigma=1.0, clip_threshold=3.0, dim=1)
    combined_w = norm_a + norm_b

    for i in range(num_unique_prompts):
        s_a = i * N
        e_a = s_a + N
        s_b = i * N
        e_b = s_b + N
        combined_resp = list(responses_a[s_a:e_a]) + list(responses_b[s_b:e_b])
        rng = random.Random((int(step) + 1) * 1_000_000 + int(i))
        r_all, _ = rewards_from_wmv_pseudolabel(combined_resp, combined_w[i].tolist(), rng=rng)
        rewards_a_list.append(r_all[:N])
        rewards_b_list.append(r_all[N:])

    return {
        "a": torch.tensor(rewards_a_list, dtype=torch.float32, device=device),
        "b": torch.tensor(rewards_b_list, dtype=torch.float32, device=device),
    }


def _compute_mixed_advantages_two_phase(
    cache: "CoIntuitorCache",
    step: int,
    num_unique_prompts: int,
    num_generations_per_model: int,
    reward_weight_a: float,
    reward_weight_b: float,
    mixed_reward_alpha: float,
    responses_a: List[str],
    responses_b: List[str],
    cap_weights_mv: Dict[str, Dict[str, float]],
    cap_weights_int: Dict[str, Dict[str, float]],
    device: Optional[torch.device] = None,
    sum_scores_minmax: bool = False,
    mixed_reward_alpha_mode: str = "fixed",
    mixed_reward_corr_weight_mode: str = "none",
) -> Dict[str, torch.Tensor]:
    """Per-dimension HACPO advantages for mixed_reward in two-phase training.

    For each reward dimension (MV & Intuitor):
      1. Build reward tensor ``(P, 2N)``.
      2. Compute HACPO-style advantage with that dimension's own ``mu_hat``
         using the corresponding per-dimension capability weights.

    Then:
      3. ``A_sum`` = fixed ``alpha * A_mv + (1-alpha) * A_int``, ``adaptive`` magnitude blend,
         or ``adaptive_sc`` per-model self-consistency from ``mv_full``.
      4. Batch-level z-score over all ``P * 2N`` elements of ``A_sum``.

    Returns:
        ``{"a": (P, N), "b": (P, N)}`` advantage tensors ready for training.
    """
    N = num_generations_per_model
    P = num_unique_prompts
    model_ids = ["a", "b"]

    # --- Build per-dimension reward tensors (P, 2N) ---
    _, scores_a_raw = cache.load_scores_a(step)
    _, scores_b_raw = cache.load_scores_b(step)
    scores_a_t, scores_b_t = _reconstruct_full_scores(
        scores_a_raw, scores_b_raw, P, N,
        "mixed_reward", device=device,
    )

    from utils.normalization import minmax_normalize as _mm_norm
    raw_intuitor = (
        float(reward_weight_a) * scores_a_t + float(reward_weight_b) * scores_b_t
    )
    intuitor_full = (
        _mm_norm(raw_intuitor, dim=1) if sum_scores_minmax else raw_intuitor
    )  # (P, 2N): [0,1] if sum_scores_minmax else raw SCE sum

    from utils.reward import compute_rewards as _compute_rewards_dispatch
    mv_a, mv_b = _compute_rewards_dispatch(
        "co_mv", responses_a, responses_b, P, N, step=int(step),
    )
    dev = intuitor_full.device
    dt = torch.float32
    mv_full = torch.cat(
        [mv_a.to(device=dev, dtype=dt), mv_b.to(device=dev, dtype=dt)], dim=1,
    )  # (P, 2N)
    intuitor_full = intuitor_full.to(dtype=dt)

    # --- Per-dimension HACPO advantages ---
    def _hacpo_adv_full(rewards_full: torch.Tensor,
                        cw: Dict[str, Dict[str, float]],
                        dim_name: str) -> torch.Tensor:
        """Compute HACPO advantage (P, 2N) with per-target-model mu_hat.

        For each target model k, mu_hat^(k) uses that dimension's capability
        weights.  The first N columns get mu_hat^(a), the last N get mu_hat^(b).
        sigma is computed jointly over all 2N.
        """
        rewards_pm = {"a": rewards_full[:, :N], "b": rewards_full[:, N:]}
        n_models = len(model_ids)

        sigma_joint = rewards_full.std(dim=1, keepdim=True).clamp(min=1e-6)  # (P, 1)

        adv_parts = []
        for k in model_ids:
            mu_hat_k = torch.zeros(P, device=dev)
            for j in model_ids:
                w_kj = cw[k][j]
                mu_hat_k = mu_hat_k + w_kj * rewards_pm[j].sum(dim=1)
            mu_hat_k = mu_hat_k / (n_models * N)

            fallback_threshold = rewards_pm[k].max(dim=1).values
            fallback_mask = mu_hat_k > fallback_threshold
            if fallback_mask.any():
                own_mean = rewards_pm[k].mean(dim=1)
                mu_hat_k = torch.where(fallback_mask, own_mean, mu_hat_k)
                logger.info(
                    "mixed_reward %s HACPO fallback: %d/%d prompts for model %s",
                    dim_name, int(fallback_mask.sum().item()), P, k,
                )

            A_k = (rewards_pm[k] - mu_hat_k.unsqueeze(1)) / sigma_joint  # (P, N)
            adv_parts.append(A_k)

        return torch.cat(adv_parts, dim=1)  # (P, 2N): [A_a | A_b]

    A_mv = _hacpo_adv_full(mv_full, cap_weights_mv, "MV")
    A_int = _hacpo_adv_full(intuitor_full, cap_weights_int, "Intuitor")

    # --- Alpha-weighted sum + batch z-score on A_sum (legacy co modes only) ---
    _mode = str(mixed_reward_alpha_mode).strip().lower()
    _corr_mode = str(mixed_reward_corr_weight_mode).strip().lower()
    _eps = 1e-6
    if _mode in ("reliability_sc_corr", "prism", "ettrl"):
        raise ValueError(
            f"{_mode} currently supports single-model mixed_reward only; "
            "two-phase co-training is not implemented."
        )
    if _corr_mode != "none":
        raise ValueError(
            "mixed_reward correlation weighting currently supports single-model mixed_reward only; "
            "co-training / two-phase HACPO is not implemented."
        )
    if _mode == "adaptive_sc":
        sc_a = mv_full[:, :N].mean(dim=1, keepdim=True)  # (P, 1)
        sc_b = mv_full[:, N:].mean(dim=1, keepdim=True)
        alpha_per = torch.cat(
            [sc_a.expand(-1, N), sc_b.expand(-1, N)], dim=1,
        )  # (P, 2N)
        A_sum = alpha_per * A_mv + (1.0 - alpha_per) * A_int  # (P, 2N)
        logger.info(
            "mixed_reward adaptive_sc alpha (SC score, two-phase HACPO): "
            "mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(alpha_per.mean().item()),
            float(alpha_per.std().item()),
            float(alpha_per.min().item()),
            float(alpha_per.max().item()),
        )
    elif _mode == "adaptive_sc_zscore":
        raise ValueError(
            "adaptive_sc_zscore currently supports single-model mixed_reward only; "
            "co-training / two-phase HACPO is not implemented."
        )
    elif _mode == "adaptive":
        abs_mv = A_mv.abs()
        abs_int = A_int.abs()
        denom = abs_mv + abs_int + _eps
        alpha_per = abs_mv / denom
        A_sum = alpha_per * A_mv + (1.0 - alpha_per) * A_int  # (P, 2N)
        logger.info(
            "mixed_reward adaptive alpha (magnitude, two-phase HACPO): "
            "mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(alpha_per.mean().item()),
            float(alpha_per.std().item()),
            float(alpha_per.min().item()),
            float(alpha_per.max().item()),
        )
    elif _mode == "fixed":
        alpha = float(mixed_reward_alpha)
        beta = 1.0 - alpha
        A_sum = alpha * A_mv + beta * A_int  # (P, 2N)
    else:
        raise ValueError(f"Unknown mixed_reward alpha mode: {mixed_reward_alpha_mode!r}")

    from utils.normalization import batch_zscore_normalize
    A_final = batch_zscore_normalize(A_sum)  # (P, 2N), batch-level

    logger.info(
        "mixed_reward HACPO advantages: "
        "A_mv mean=%.4f std=%.4f, A_int mean=%.4f std=%.4f, "
        "A_sum mean=%.4f std=%.4f, A_final(batch z) mean=%.4f std=%.4f",
        A_mv.mean().item(), A_mv.std().item(),
        A_int.mean().item(), A_int.std().item(),
        A_sum.mean().item(), A_sum.std().item(),
        A_final.mean().item(), A_final.std().item(),
    )

    return {"a": A_final[:, :N], "b": A_final[:, N:]}


# ---------------------------------------------------------------------------
# Training loop helper (used by both mixed and two-phase modes)
# ---------------------------------------------------------------------------

def _run_training_loop(
    trainer,
    training_data: Dict[str, Any],
    training_args,
    phase_label: str = "",
    clip_ctx: Optional[Dict[str, Any]] = None,
) -> float:
    """Execute one full pass of micro-step training and return average loss.

    When *clip_ctx* is provided and contains stepwise peer-clip config, the
    ``epsilon_low`` entry is updated **per mini-batch** to match HACRL's
    progressive tightening (``batch_idx`` resets to 0 at each update call).
    """
    from torch.utils.data import Dataset, DataLoader
    from transformers.trainer_utils import seed_worker
    from transformers import default_data_collator

    class _DictTensorDataset(Dataset):
        def __init__(self, data_dict: Dict[str, Any]):
            self.data = {k: v for k, v in data_dict.items() if v is not None}
            first = next(iter(self.data.values()))
            self._len = first.size(0)

        def __len__(self):
            return self._len

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    stepwise_cfg = (clip_ctx or {}).get("_stepwise_cfg")

    per_device_bs = int(getattr(training_args, "per_device_train_batch_size", 1))
    train_dataset = _DictTensorDataset(training_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_bs,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        persistent_workers=training_args.dataloader_persistent_workers,
        worker_init_fn=seed_worker,
        drop_last=training_args.dataloader_drop_last,
        prefetch_factor=training_args.dataloader_prefetch_factor,
    )

    if not hasattr(trainer, "_gspo_optim_created"):
        # Co-training still owns its legacy outer-loop scheduling protocol.
        # Keep it explicit while the single-model pipeline uses the exact
        # DeepSpeed-native optimizer-update scheduler.
        trainer.create_optimizer_and_scheduler(
            1,
            train_dataloader=train_dataloader,
            scheduler_backend="legacy",
        )
        trainer._gspo_optim_created = True

    # DeepSpeed path already prepares dataloader inside create_optimizer_and_scheduler.
    # Keep behavior aligned with single_intuitor_train: prefer trainer-managed dataloader.
    effective_dataloader = getattr(trainer, "train_dataloader", train_dataloader)

    tr_loss = torch.tensor(0.0, device=trainer.accelerator.device)
    accum = max(1, int(getattr(training_args, "gradient_accumulation_steps", 1)))

    from tqdm.auto import tqdm

    try:
        total_micro = len(effective_dataloader)
    except TypeError:
        total_micro = None

    desc = f"Training {phase_label}(micro-steps, accum={accum})" if phase_label else f"Training (micro-steps, accum={accum})"
    pbar = None
    if trainer.accelerator.is_main_process:
        pbar = tqdm(total=total_micro, desc=desc, unit="micro", dynamic_ncols=True, leave=True, position=0)

    for _micro, batch in enumerate(effective_dataloader, start=0):
        # HACRL stepwise tightening operates per *optimizer step* (mini-batch),
        # not per micro-batch.  With gradient accumulation, one optimizer step
        # spans ``accum`` micro-batches, so we divide by ``accum`` to match.
        optim_step_idx = _micro // accum
        if clip_ctx is not None:
            clip_ctx["mini_batch_idx"] = optim_step_idx
        if stepwise_cfg is not None and clip_ctx is not None:
            delta = stepwise_cfg["delta"]
            delta_step = stepwise_cfg["delta_step"]
            # HACRL semantics: ratio lower bound = delta + step * delta_step
            # (progressively tighter).  Convert to epsilon:
            #   eps_low = 1.0 - (delta + step * delta_step)
            clip_ctx["peer_epsilon_low"] = max(0.0, (1.0 - delta) - optim_step_idx * delta_step)

        if trainer.is_deepspeed_enabled:
            loss = trainer.training_step(batch)
            tr_loss = tr_loss + loss
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=float(loss.detach().item()))
            trainer.optimizer_step()
        else:
            with trainer.accelerator.accumulate(trainer.model):
                loss = trainer.training_step(batch)
                tr_loss = tr_loss + loss
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=float(loss.detach().item()))
                if trainer.accelerator.sync_gradients:
                    trainer.optimizer_step()

    if pbar is not None:
        pbar.refresh()
        pbar.close()
        print("")

    return tr_loss.item() / max(1, accum)


def main(script_args, training_args, model_args):
    """Main training function."""
    # Validate train_model argument
    if script_args.train_model is None:
        raise ValueError("--train_model argument is required. Use 'a' or 'b'")
    
    if script_args.train_model not in ["a", "b"]:
        raise ValueError(f"Invalid train_model: {script_args.train_model}. Must be 'a' or 'b'")
    
    train_model = script_args.train_model
    
    # CRITICAL: Do NOT call accelerate.utils.set_seed here!
    # accelerate.utils.set_seed() internally uses PartialState() which initializes AcceleratorState
    # This happens BEFORE Accelerator is created in SingleModelTrainer, causing state conflicts
    # Instead, use transformers.set_seed here (doesn't trigger AcceleratorState)
    # Then call accelerate.utils.set_seed AFTER Accelerator is created (in trainer)
    from transformers import set_seed
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Reduce noisy logs from DeepSpeed/Accelerate/NCCL wrappers so our logs remain readable.
    # Keep our module logger at `log_level`, but silence verbose submodules.
    noisy_modules_to_error = [
        "deepspeed",
        "accelerate",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch._dynamo",
        "transformers.generation",
        "transformers.modeling_utils",
        "datasets",
    ]
    for name in noisy_modules_to_error:
        logging.getLogger(name).setLevel(logging.ERROR)

    # DeepSpeed also has its own internal logger.
    try:
        import deepspeed

        if hasattr(deepspeed, "utils") and hasattr(deepspeed.utils, "logging"):
            ds_logger = getattr(deepspeed.utils.logging, "logger", None)
            if ds_logger is not None:
                ds_logger.setLevel(logging.ERROR)
    except Exception:
        pass
    
    logger.info(f"Training model: {train_model}")
    logger.info(
        "Reward method: %s (SCE weight_a=%s, weight_b=%s; mixed w_mv=%s, mode=%s)",
        getattr(training_args, "reward_method", "sum_scores"),
        getattr(training_args, "reward_weight_a", 1.0),
        getattr(training_args, "reward_weight_b", 1.0),
        getattr(training_args, "mixed_reward_alpha", 0.5),
        getattr(training_args, "mixed_reward_alpha_mode", "fixed"),
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    # Print TRL GRPO clip settings explicitly (avoid guessing defaults).
    try:
        logger.info(
            f"GRPO clip config: epsilon={getattr(training_args, 'epsilon', None)} "
            f"epsilon_high={getattr(training_args, 'epsilon_high', None)} "
            f"self_update_epsilon_low={getattr(training_args, 'self_update_epsilon_low', None)} "
            f"self_update_epsilon_high={getattr(training_args, 'self_update_epsilon_high', None)} "
            f"peer_asym={getattr(training_args, 'peer_special_clip_asymmetric_enabled', False)} "
            f"peer_stepwise={getattr(training_args, 'peer_special_clip_stepwise_enabled', False)} "
            f"peer_delta={getattr(training_args, 'peer_special_clip_delta', 0.8)} "
            f"peer_delta_step={getattr(training_args, 'peer_special_clip_delta_step', 0.4)}"
        )
    except Exception:
        pass
    
    # Determine model paths
    # Policy model: use latest checkpoint if available, otherwise use initial model
    if train_model == "a":
        policy_model_path = script_args.model_a_path if script_args.model_a_path else model_args.model_name_or_path
        # Check for latest checkpoint in output_dir
        if os.path.exists(training_args.output_dir):
            latest_checkpoint = get_last_checkpoint(training_args.output_dir)
            if latest_checkpoint is not None:
                policy_model_path = latest_checkpoint
                logger.info(f"Found latest checkpoint for Model A: {policy_model_path}")
    else:  # b
        policy_model_path = script_args.model_b_path if script_args.model_b_path else model_args.model_name_or_path
        # Check for latest checkpoint in output_dir
        if os.path.exists(training_args.output_dir):
            latest_checkpoint = get_last_checkpoint(training_args.output_dir)
            if latest_checkpoint is not None:
                policy_model_path = latest_checkpoint
                logger.info(f"Found latest checkpoint for Model B: {policy_model_path}")
    
    logger.info(f"Policy model path: {policy_model_path}")
    
    # Ref model: use ref_model_path if provided, otherwise use initial model
    # For Model A: ref_model_a_path or initial_model_a_path
    # For Model B: ref_model_b_path or initial_model_b_path
    ref_model_path = None
    if hasattr(script_args, 'ref_model_a_path') and train_model == "a" and script_args.ref_model_a_path:
        ref_model_path = script_args.ref_model_a_path
    elif hasattr(script_args, 'ref_model_b_path') and train_model == "b" and script_args.ref_model_b_path:
        ref_model_path = script_args.ref_model_b_path
    elif train_model == "a" and script_args.model_a_path:
        # Fallback: use initial model as ref
        ref_model_path = script_args.model_a_path
    elif train_model == "b" and script_args.model_b_path:
        # Fallback: use initial model as ref
        ref_model_path = script_args.model_b_path
    
    if ref_model_path:
        logger.info(f"Reference model path: {ref_model_path}")
    
    # Initialize cache
    cache_dir = script_args.cache_dir if script_args.cache_dir else training_args.output_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cache.json")
    cache = CoIntuitorCache(cache_file)
    cap_reporter = CapabilityReporter(cache_dir)
    logger.info(f"Using cache directory: {cache_dir}")
    
    # Load tokenizer (source-of-truth) and model for policy.
    #
    # IMPORTANT: Tokenizer/chat_template should remain stable across steps. Checkpoints saved via
    # `save_pretrained()` can drift (e.g., tokenizer_config.json losing chat_template). We therefore
    # allow passing `--tokenizer_a_path/--tokenizer_b_path` to force tokenizer loading from the
    # initial/base model folders.
    tokenizer_path = policy_model_path
    if train_model == "a" and getattr(script_args, "tokenizer_a_path", None):
        tokenizer_path = str(script_args.tokenizer_a_path)
    if train_model == "b" and getattr(script_args, "tokenizer_b_path", None):
        tokenizer_path = str(script_args.tokenizer_b_path)
    tokenizer = get_tokenizer(tokenizer_path, training_args)
    model = get_model(policy_model_path, training_args, model_args)
    
    # Create trainer
    objective_mode = getattr(training_args, "objective_mode", "grpo")
    is_length_norm = getattr(training_args, "is_length_norm", "mean")

    # Resolve self-update epsilon: prefer dedicated fields, fall back to global epsilon.
    _global_eps = float(getattr(training_args, 'epsilon', 0.1))
    _global_eps_high = getattr(training_args, 'epsilon_high', None)
    _self_eps_low = getattr(training_args, 'self_update_epsilon_low', None)
    _self_eps_high = getattr(training_args, 'self_update_epsilon_high', None)
    _effective_self_eps_low = float(_self_eps_low) if _self_eps_low is not None else _global_eps
    _effective_self_eps_high = float(_self_eps_high) if _self_eps_high is not None else _global_eps_high

    # Mutable clip context: the loss function reads from this dict so we can
    # swap epsilon/clip bounds between self-phase and peer-phase without
    # re-creating the trainer.
    _clip_ctx: Dict[str, Any] = {
        "epsilon_low": _effective_self_eps_low,
        "epsilon_high": _effective_self_eps_high,
        "is_reweight_alpha": 0.0,
        "peer_epsilon_low": None,
        "peer_epsilon_high": None,
        "capability_weight": 1.0,
        "mini_batch_idx": 0,
    }

    def compute_loss_fn(model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("compute_loss_fn does not support returning outputs")
        return compute_policy_loss(
            model=model,
            inputs=inputs,
            objective_mode=objective_mode,
            temperature=getattr(training_args, 'temperature', 1.0),
            epsilon_low=_clip_ctx["epsilon_low"],
            epsilon_high=_clip_ctx["epsilon_high"],
            beta=getattr(training_args, 'beta', 0.0),
            loss_type=getattr(training_args, 'loss_type', 'grpo'),
            max_completion_length=getattr(training_args, 'max_completion_length', 512),
            num_iterations=getattr(training_args, 'num_iterations', 1),
            length_norm=is_length_norm,
            is_reweight_alpha=_clip_ctx["is_reweight_alpha"],
            peer_epsilon_low=_clip_ctx.get("peer_epsilon_low"),
            peer_epsilon_high=_clip_ctx.get("peer_epsilon_high"),
            capability_weight=_clip_ctx.get("capability_weight", 1.0),
            mini_batch_idx=_clip_ctx.get("mini_batch_idx", 0),
            gspo_surrogate_ratio_enabled=bool(getattr(training_args, 'gspo_surrogate_ratio_enabled', True)),
            peer_loss_uniform=bool(getattr(training_args, 'peer_loss_uniform', False)),
        )
    
    trainer = SingleModelTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_loss=compute_loss_fn,
        metadata_source_dir=tokenizer_path,
    )
    
    # Set seed with accelerate.utils.set_seed AFTER Accelerator is created
    # This ensures AcceleratorState is properly initialized before set_seed accesses it
    # Reference: intuitor_trainer.py line 622 - set_seed is called after super().__init__()
    from accelerate.utils import set_seed as accelerate_set_seed
    accelerate_set_seed(training_args.seed, device_specific=True)
    
    num_generations_per_model = getattr(training_args, 'num_generations_per_model', 4)
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    
    # Training loop: train on latest step with complete cache data.
    #
    # IMPORTANT: Do NOT use `training_args.max_steps` to bound the cache scan.
    # In this project, we run `--max_steps 1` to mean "do one optimizer update", while the cache step is the outer
    # pipeline iteration index (0, 1, 2, ...). If we bound by `max_steps`, step=1+ would never be considered.
    current_step = cache.get_latest_complete_step(num_generations_per_model)
    
    if current_step < 0:
        error_msg = f"No complete cache data found for training. Please run generate_and_score.py first."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Training on step {current_step} (latest step with complete cache data)")
    
    # Training loop (single step)
    logger.info("*** Starting training step ***")
    
    step = current_step
    
    # Load prompts directly from cache (generate_and_score.py always saves rendered prompts).
    if train_model == "a":
        is_valid_p, _responses_p, num_prompts_step, cached_prompts, _finish_p = cache.load_generations_a(step, num_generations_per_model)
    else:
        is_valid_p, _responses_p, num_prompts_step, cached_prompts, _finish_p = cache.load_generations_b(step, num_generations_per_model)
    if not is_valid_p:
        raise ValueError(f"Cache data not complete for step {step}")
    if cached_prompts is None:
        raise ValueError(
            f"Prompts not found in cache for step {step}. "
            "Ensure generate_and_score.py saved prompts to cache."
        )
    step_prompts = cached_prompts
    
    co_update_mode = getattr(training_args, "co_update_mode", "mixed")
    model_ids: List[str] = list(getattr(training_args, "model_ids", ["a", "b"]))
    capability_weight_mode = getattr(training_args, "capability_weight_mode", "sc_ratio")
    capability_ratio_smoothing = getattr(training_args, "capability_ratio_smoothing", "latest")
    capability_ratio_k = int(getattr(training_args, "capability_ratio_k", 5))
    capability_normalize_sce = bool(getattr(training_args, "capability_normalize_sce", False))
    strong_model_peer_penalty = float(getattr(training_args, "strong_model_peer_penalty", 0.0))
    is_strict_old_for_peer = bool(getattr(training_args, "is_strict_old_for_peer", False))
    reward_method = getattr(training_args, "reward_method", "sum_scores")
    reward_weight_a = getattr(training_args, "reward_weight_a", 1.0)
    reward_weight_b = getattr(training_args, "reward_weight_b", 1.0)
    mixed_reward_alpha = getattr(training_args, "mixed_reward_alpha", 0.5)
    mixed_reward_alpha_mode = getattr(training_args, "mixed_reward_alpha_mode", "fixed")
    mixed_reward_alpha_zscore_mu = getattr(training_args, "mixed_reward_alpha_zscore_mu", 0.65)
    mixed_reward_alpha_zscore_sigma = getattr(training_args, "mixed_reward_alpha_zscore_sigma", 0.2)
    mixed_reward_alpha_zscore_lower = getattr(training_args, "mixed_reward_alpha_zscore_lower", 0.0)
    mixed_reward_alpha_zscore_upper = getattr(training_args, "mixed_reward_alpha_zscore_upper", 1.0)
    mixed_reward_corr_weight_mode = getattr(training_args, "mixed_reward_corr_weight_mode", "none")
    mixed_reward_corr_weight_threshold = getattr(training_args, "mixed_reward_corr_weight_threshold", 0.1)
    mixed_reward_corr_weight_exponent = getattr(training_args, "mixed_reward_corr_weight_exponent", 0.8)
    scale_rewards = getattr(training_args, "scale_rewards", True)
    N = num_generations_per_model

    # Load both models' generations for two-phase (needed regardless of branch to size prompts)
    is_valid_a, responses_a, _, _, _ = cache.load_generations_a(step, N)
    is_valid_b, responses_b, _, _, _ = cache.load_generations_b(step, N)

    if co_update_mode == "two_phase" and objective_mode == "gspo":
        # ---------------------------------------------------------------
        # Two-phase GSPO co-training: self-phase then peer-phase
        # Supports learn_config to control per-phase sample counts.
        # ---------------------------------------------------------------
        logger.info("Using two-phase GSPO co-update mode")

        if not (is_valid_a and is_valid_b):
            raise ValueError(f"Both models must have complete cache data for two-phase mode (step {step})")

        num_unique_prompts = len(step_prompts)
        peer_model = "b" if train_model == "a" else "a"

        # --- Parse learn_config to determine per-phase sample counts ---
        learn_config_raw = getattr(training_args, "learn_config", None)
        learn_cfg = None
        if learn_config_raw is not None:
            if isinstance(learn_config_raw, dict):
                learn_cfg = learn_config_raw
            elif isinstance(learn_config_raw, str) and learn_config_raw.strip().lower() not in ("", "none", "null"):
                learn_cfg = json.loads(learn_config_raw)

        own_count = N
        other_count = N
        if learn_cfg:
            model_cfg = learn_cfg.get(train_model, {})
            own_count = model_cfg.get("own", -1)
            other_count = model_cfg.get("other", -1)
            if own_count < 0:
                own_count = N
            if other_count < 0:
                other_count = N
            own_count = min(own_count, N)
            other_count = min(other_count, N)

        has_self_phase = own_count > 0
        has_peer_phase = other_count > 0

        if not has_self_phase and not has_peer_phase:
            raise ValueError(
                f"learn_config results in own=0 and other=0 for model '{train_model}' — nothing to train on"
            )

        logger.info(f"Two-phase config: own_count={own_count}/{N}, other_count={other_count}/{N}, "
                     f"has_self={has_self_phase}, has_peer={has_peer_phase}")

        # --- Compute rewards per model (ALL responses participate) ---
        sum_scores_minmax = bool(getattr(training_args, "sum_scores_minmax", False))
        rewards_per_model = _compute_rewards_for_two_phase(
            cache=cache, step=step,
            num_unique_prompts=num_unique_prompts,
            num_generations_per_model=N,
            reward_method=reward_method,
            reward_weight_a=reward_weight_a,
            reward_weight_b=reward_weight_b,
            responses_a=responses_a if responses_a else [],
            responses_b=responses_b if responses_b else [],
            mixed_reward_alpha=mixed_reward_alpha,

            sum_scores_minmax=sum_scores_minmax,
        )

        reward_stats = {
            mid: {
                "mean": float(rewards_per_model[mid].mean().item()),
                "var": float(rewards_per_model[mid].var(unbiased=False).item()),
            }
            for mid in model_ids
        }
        mean_rewards = {mid: reward_stats[mid]["mean"] for mid in model_ids}
        logger.info(f"Reward stats per model (step {step}): {reward_stats}")

        cache.save_reward_stats(step, reward_stats)
        cache.save_mean_rewards(step, mean_rewards)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # --- Determine base values for capability ratio ---
        stable_perf = int(getattr(training_args, "stable_perf", 0))
        _force_uniform_cap = stable_perf > 0 and step < stable_perf

        # Per-dimension capability weights (populated only for mixed_reward)
        cap_weights_mv: Optional[Dict[str, Dict[str, float]]] = None
        cap_weights_int: Optional[Dict[str, Dict[str, float]]] = None
        sum_capability_ratio: Optional[Dict[str, Dict[str, float]]] = None

        if reward_method == "mixed_reward":
            # ----------------------------------------------------------
            # mixed_reward: compute per-dimension capability ratios
            # ----------------------------------------------------------
            dev = rewards_per_model[model_ids[0]].device

            # Build per-dimension reward tensors to extract per-model means
            _, scores_a_raw = cache.load_scores_a(step)
            _, scores_b_raw = cache.load_scores_b(step)
            scores_a_t, scores_b_t = _reconstruct_full_scores(
                scores_a_raw, scores_b_raw, num_unique_prompts, N,
                reward_method, device=dev,
            )
            from utils.normalization import minmax_normalize as _mm_norm
            raw_intuitor = float(reward_weight_a) * scores_a_t + float(reward_weight_b) * scores_b_t
            intuitor_full = (
                _mm_norm(raw_intuitor, dim=1) if sum_scores_minmax else raw_intuitor
            )  # (P, 2N)

            from utils.reward import compute_rewards as _compute_rewards_dispatch
            mv_a, mv_b = _compute_rewards_dispatch(
                "co_mv", responses_a if responses_a else [],
                responses_b if responses_b else [],
                num_unique_prompts, N, step=int(step),
            )
            mv_full = torch.cat(
                [mv_a.to(device=dev, dtype=torch.float32),
                 mv_b.to(device=dev, dtype=torch.float32)], dim=1,
            )

            mean_mv = {
                "a": float(mv_full[:, :N].mean().item()),
                "b": float(mv_full[:, N:].mean().item()),
            }
            mean_int = {
                "a": float(intuitor_full[:, :N].mean().item()),
                "b": float(intuitor_full[:, N:].mean().item()),
            }
            cache.save_mixed_reward_dim_means(step, mean_mv, mean_int)
            logger.info(
                "mixed_reward per-dim means: MV=%s, Intuitor=%s",
                mean_mv, mean_int,
            )

            # Sliding-window smoothing for each dimension independently
            mean_mv_for_cap = mean_mv
            mean_int_for_cap = mean_int
            if capability_ratio_smoothing == "last_k":
                dim_history = cache.load_mixed_reward_dim_means_history(step, capability_ratio_k)
                if len(dim_history) > 0:
                    n_hist = len(dim_history)
                    mean_mv_for_cap = {
                        mid: sum(h["mv"][mid] for h in dim_history) / n_hist
                        for mid in model_ids
                    }
                    mean_int_for_cap = {
                        mid: sum(h["int"][mid] for h in dim_history) / n_hist
                        for mid in model_ids
                    }
                    logger.info(
                        "mixed_reward cap smoothing: last_k (K=%d, avail=%d), "
                        "smoothed MV=%s, smoothed Int=%s",
                        capability_ratio_k, n_hist,
                        mean_mv_for_cap, mean_int_for_cap,
                    )
                else:
                    logger.info("No dim history for last_k smoothing, using current step")

            _, cap_raw_mv, cap_clipped_mv = compute_capability_ratio_details(mean_mv_for_cap, model_ids)
            _, cap_raw_int, cap_clipped_int = compute_capability_ratio_details(mean_int_for_cap, model_ids)

            if _force_uniform_cap:
                _uniform = {k: {j: 1.0 for j in model_ids} for k in model_ids}
                cap_weights_mv = _uniform
                cap_weights_int = _uniform
                logger.info(
                    f"stable_perf={stable_perf}: step {step} < {stable_perf}, "
                    f"forcing all per-dimension capability weights to 1.0"
                )
            else:
                cap_weights_mv = cap_clipped_mv
                cap_weights_int = cap_clipped_int

            # sum_capability_ratio = alpha * cap_mv + (1-alpha) * cap_int
            alpha = float(mixed_reward_alpha)
            beta = 1.0 - alpha
            sum_capability_ratio = {}
            for k in model_ids:
                sum_capability_ratio[k] = {}
                for j in model_ids:
                    sum_capability_ratio[k][j] = (
                        alpha * cap_weights_mv[k][j] + beta * cap_weights_int[k][j]
                    )

            # Also compute the composite cap_weights for logging / saving
            # (use the combined reward means as before for backward compat)
            mean_cap_base = mean_rewards
            cap_weights = sum_capability_ratio

            cap_base_values, cap_raw_ratios, cap_clipped_ratios = compute_capability_ratio_details(
                mean_rewards, model_ids,
            )

            cache.save_capability_ratios(
                step,
                {
                    "mode": "mixed_reward_per_dim",
                    "smoothing": str(capability_ratio_smoothing),
                    "k": int(capability_ratio_k),
                    "mv_means_current": mean_mv,
                    "int_means_current": mean_int,
                    "mv_means_used": mean_mv_for_cap,
                    "int_means_used": mean_int_for_cap,
                    "cap_raw_mv": cap_raw_mv,
                    "cap_clipped_mv": cap_clipped_mv,
                    "cap_raw_int": cap_raw_int,
                    "cap_clipped_int": cap_clipped_int,
                    "cap_weights_mv": cap_weights_mv,
                    "cap_weights_int": cap_weights_int,
                    "sum_capability_ratio": sum_capability_ratio,
                    "combined_raw_ratios": cap_raw_ratios,
                    "combined_clipped_ratios": cap_clipped_ratios,
                    "stable_perf_applied": bool(_force_uniform_cap),
                },
            )
            logger.info(
                "mixed_reward per-dim cap ratios (step %s): "
                "MV raw=%s clipped=%s, Int raw=%s clipped=%s, "
                "sum_cap_ratio=%s",
                step, cap_raw_mv, cap_clipped_mv,
                cap_raw_int, cap_clipped_int,
                sum_capability_ratio,
            )
            logger.info(
                "mixed_reward: per-dim HACPO advantages, then alpha-weighted + GRPO final. "
                "(alpha_mv=%.3f, alpha_int=%.3f, w_a=%.3f, w_b=%.3f)",
                alpha, beta, reward_weight_a, reward_weight_b,
            )
        else:
            # ----------------------------------------------------------
            # Non-mixed_reward: existing single-dimension capability logic
            # ----------------------------------------------------------
            if capability_weight_mode == "sc_ratio":
                _, scores_a_raw = cache.load_scores_a(step)
                _, scores_b_raw = cache.load_scores_b(step)
                sc_a_t, sc_b_t = _reconstruct_full_scores(
                    scores_a_raw, scores_b_raw, num_unique_prompts, N,
                    reward_method, device=rewards_per_model[model_ids[0]].device,
                )
                if capability_normalize_sce:
                    sc_a_t = normalize_to_unit_interval(sc_a_t, dim=1)
                    sc_b_t = normalize_to_unit_interval(sc_b_t, dim=1)
                    logger.info("capability_normalize_sce: SCE scores normalized to [0,1] for capability ratio")
                mean_cap_base = {
                    "a": float(sc_a_t[:, :N].mean().item()),
                    "b": float(sc_b_t[:, N:].mean().item()),
                }
                cache.save_mean_sc_scores(step, mean_cap_base)
                logger.info(f"sc_ratio: mean SCE scores for cap ratio = {mean_cap_base}")
            else:
                if capability_normalize_sce and reward_method in ("sum_scores", "self_scores"):
                    norm_rewards = {
                        mid: normalize_to_unit_interval(rewards_per_model[mid], dim=1)
                        for mid in model_ids
                    }
                    mean_cap_base = {
                        mid: float(norm_rewards[mid].mean().item()) for mid in model_ids
                    }
                    logger.info(
                        "capability_normalize_sce: reward_ratio mode with SCE rewards — "
                        "normalized to [0,1] for capability ratio: %s", mean_cap_base,
                    )
                else:
                    mean_cap_base = mean_rewards

            if capability_ratio_smoothing == "last_k":
                if capability_weight_mode == "sc_ratio":
                    history = cache.load_mean_sc_scores_history(step, capability_ratio_k)
                else:
                    history = cache.load_mean_rewards_history(step, capability_ratio_k)
                if len(history) > 0:
                    smoothed = {}
                    for mid in model_ids:
                        smoothed[mid] = sum(h[mid] for h in history) / len(history)
                    logger.info(
                        f"Capability ratio smoothing: last_k (K={capability_ratio_k}, "
                        f"available={len(history)} steps), "
                        f"smoothed values: {smoothed}"
                    )
                    mean_rewards_for_cap = smoothed
                else:
                    logger.info("No history available for last_k smoothing, using current step only")
                    mean_rewards_for_cap = mean_cap_base
            else:
                mean_rewards_for_cap = mean_cap_base

            if reward_method == "sum_scores":
                if sum_scores_minmax:
                    logger.info(
                        "sum_scores mode: R = minmax(w_a*scores_a + w_b*scores_b) "
                        "over 2N per prompt → [0,1] (w_a=%s, w_b=%s)",
                        reward_weight_a, reward_weight_b,
                    )
                else:
                    logger.info(
                        "sum_scores mode: R = raw w_a*scores_a + w_b*scores_b "
                        "(no intuitor min-max; w_a=%s, w_b=%s)",
                        reward_weight_a, reward_weight_b,
                    )
            cap_base_values, cap_raw_ratios, cap_clipped_ratios = compute_capability_ratio_details(
                mean_rewards_for_cap, model_ids
            )
            if _force_uniform_cap:
                cap_weights = {k: {j: 1.0 for j in model_ids} for k in model_ids}
                logger.info(
                    f"stable_perf={stable_perf}: step {step} < {stable_perf}, "
                    f"forcing all capability weights to 1.0"
                )
            else:
                cap_weights = cap_clipped_ratios
            cache.save_capability_ratios(
                step,
                {
                    "mode": str(capability_weight_mode),
                    "smoothing": str(capability_ratio_smoothing),
                    "k": int(capability_ratio_k),
                    "current_values": {mid: float(mean_cap_base[mid]) for mid in model_ids},
                    "values_used": {mid: float(mean_rewards_for_cap[mid]) for mid in model_ids},
                    "floored_values": {mid: float(cap_base_values[mid]) for mid in model_ids},
                    "raw_ratios": cap_raw_ratios,
                    "clipped_ratios": cap_clipped_ratios,
                    "applied_weights": cap_weights,
                    "stable_perf_applied": bool(_force_uniform_cap),
                },
            )
            logger.info(
                "Capability ratio details (step %s): raw=%s clipped=%s applied=%s",
                step, cap_raw_ratios, cap_clipped_ratios, cap_weights,
            )
        logger.info(f"Capability weights: w({train_model},{peer_model})={cap_weights[train_model][peer_model]:.4f}, "
                     f"w({peer_model},{train_model})={cap_weights[peer_model][train_model]:.4f}")

        cap_reporter.record_step(
            step=step,
            reward_stats=reward_stats,
            raw_ratios=cap_raw_ratios if reward_method != "mixed_reward" else cap_raw_mv,
            clipped_ratios=cap_clipped_ratios if reward_method != "mixed_reward" else cap_clipped_mv,
            applied_weights=cap_weights,
            mode=str(capability_weight_mode) if reward_method != "mixed_reward" else "mixed_reward_per_dim",
            smoothing=str(capability_ratio_smoothing),
            stable_perf_applied=bool(_force_uniform_cap),
        )

        # --- Compute advantages on ALL completions ---
        accuracy_based = reward_method in ("truth", "co_mv", "self_mv", "mv_gated_intuitor", "co_mv_gated_intuitor")
        w_jk = 1.0

        if reward_method == "mixed_reward":
            # Per-dimension HACPO advantages → alpha-weighted sum → GRPO final.
            mixed_adv_dict = _compute_mixed_advantages_two_phase(
                cache=cache, step=step,
                num_unique_prompts=num_unique_prompts,
                num_generations_per_model=N,
                reward_weight_a=reward_weight_a,
                reward_weight_b=reward_weight_b,
                mixed_reward_alpha=mixed_reward_alpha,
                responses_a=responses_a if responses_a else [],
                responses_b=responses_b if responses_b else [],
                cap_weights_mv=cap_weights_mv,
                cap_weights_int=cap_weights_int,
                device=rewards_per_model[model_ids[0]].device,
                sum_scores_minmax=sum_scores_minmax,
                mixed_reward_alpha_mode=mixed_reward_alpha_mode,
                mixed_reward_corr_weight_mode=mixed_reward_corr_weight_mode,
            )
            self_advantages_all = mixed_adv_dict[train_model].reshape(-1)
            if has_peer_phase:
                peer_advantages_all = mixed_adv_dict[peer_model].reshape(-1)
                w_jk = sum_capability_ratio[peer_model][train_model]
                if (
                    strong_model_peer_penalty > 0.0
                    and not _force_uniform_cap
                    and mean_rewards.get(train_model, 0.0) > mean_rewards.get(peer_model, 0.0)
                ):
                    w_jk_before = w_jk
                    w_jk = max(0.1, w_jk - strong_model_peer_penalty)
                    logger.info(
                        "strong_model_peer_penalty: train_model=%s is stronger, "
                        "w_jk %.4f -> %.4f (penalty=%.4f)",
                        train_model, w_jk_before, w_jk, strong_model_peer_penalty,
                    )
                logger.info(
                    f"Mixed-reward HACPO advantages: "
                    f"self mean={self_advantages_all.mean():.4f} std={self_advantages_all.std():.4f}, "
                    f"peer mean={peer_advantages_all.mean():.4f} std={peer_advantages_all.std():.4f}, "
                    f"sum_cap_ratio(peer→train)={w_jk:.4f}"
                )
            else:
                logger.info(
                    f"Mixed-reward HACPO advantages: "
                    f"self mean={self_advantages_all.mean():.4f} std={self_advantages_all.std():.4f} "
                    f"(peer phase skipped)"
                )
        elif has_peer_phase:
            # HACRL-aligned GSPO: single baseline for self AND peer.
            self_advantages_all, peer_advantages_all = compute_unified_gspo_advantages(
                rewards_per_model, cap_weights, train_model, N, scale_rewards,
                accuracy_based=accuracy_based,
            )
            w_jk = cap_weights[peer_model][train_model]
            if (
                strong_model_peer_penalty > 0.0
                and not _force_uniform_cap
                and mean_rewards.get(train_model, 0.0) > mean_rewards.get(peer_model, 0.0)
            ):
                w_jk_before = w_jk
                w_jk = max(0.1, w_jk - strong_model_peer_penalty)
                logger.info(
                    "strong_model_peer_penalty: train_model=%s is stronger, "
                    "w_jk %.4f -> %.4f (penalty=%.4f)",
                    train_model, w_jk_before, w_jk, strong_model_peer_penalty,
                )
            logger.info(f"Unified advantage stats: self mean={self_advantages_all.mean():.4f} std={self_advantages_all.std():.4f}, "
                         f"peer mean={peer_advantages_all.mean():.4f} std={peer_advantages_all.std():.4f}, "
                         f"capability_weight(peer→train)={w_jk:.4f}")
        else:
            self_advantages_all = compute_gspo_advantages(
                rewards_per_model, cap_weights, train_model, N, scale_rewards,
                accuracy_based=accuracy_based,
            )
            logger.info(f"Advantage stats: self mean={self_advantages_all.mean():.4f} std={self_advantages_all.std():.4f} "
                         f"(peer phase skipped)")

        # --- Prepare training data for each phase ---
        self_data = None
        peer_data = None

        if has_self_phase:
            if train_model == "a":
                own_global = list(range(own_count))
            else:
                own_global = list(range(N, N + own_count))
            own_indices_str = ",".join(str(i) for i in own_global)

            self_data = prepare_training_data_from_cache(
                cache=cache, step=step, train_model=train_model,
                unique_prompts=step_prompts, tokenizer=tokenizer,
                num_generations_per_model=N,
                max_prompt_length=getattr(training_args, 'max_prompt_length', None),
                max_completion_length=getattr(training_args, 'max_completion_length', None),
                mask_truncated_completions=getattr(training_args, 'mask_truncated_completions', False),
                reward_method=reward_method, reward_weight_a=reward_weight_a,
                reward_weight_b=reward_weight_b,
                mixed_reward_alpha=mixed_reward_alpha,
                mixed_reward_alpha_mode=mixed_reward_alpha_mode,
                mixed_reward_alpha_zscore_mu=mixed_reward_alpha_zscore_mu,
                mixed_reward_alpha_zscore_sigma=mixed_reward_alpha_zscore_sigma,
                mixed_reward_alpha_zscore_lower=mixed_reward_alpha_zscore_lower,
                mixed_reward_alpha_zscore_upper=mixed_reward_alpha_zscore_upper,
                mixed_reward_corr_weight_mode=mixed_reward_corr_weight_mode,
                mixed_reward_corr_weight_threshold=mixed_reward_corr_weight_threshold,
                mixed_reward_corr_weight_exponent=mixed_reward_corr_weight_exponent,
                sum_scores_minmax=sum_scores_minmax,
                scale_rewards=scale_rewards,
                device=None,
                selected_indices_a_raw=own_indices_str if train_model == "a" else None,
                selected_indices_b_raw=own_indices_str if train_model == "b" else None,
                learn_config_json=None,
            )
            if own_count < N:
                self_adv = self_advantages_all.view(num_unique_prompts, N)[:, :own_count].reshape(-1)
            else:
                self_adv = self_advantages_all
            self_data["advantages"] = self_adv
            if truthcheck_enabled():
                expected_self = int(num_unique_prompts * own_count)
                got_self = int(self_data["advantages"].size(0))
                if got_self != expected_self:
                    raise AssertionError(
                        f"[TRUTHCHECK][co.train.two_phase.self_phase] "
                        f"self advantages rows mismatch: got {got_self}, expected {expected_self}"
                    )

        if has_peer_phase:
            if train_model == "a":
                peer_global = list(range(N, N + other_count))
            else:
                peer_global = list(range(other_count))
            peer_indices_str = ",".join(str(i) for i in peer_global)

            peer_data = prepare_training_data_from_cache(
                cache=cache, step=step, train_model=train_model,
                unique_prompts=step_prompts, tokenizer=tokenizer,
                num_generations_per_model=N,
                max_prompt_length=getattr(training_args, 'max_prompt_length', None),
                max_completion_length=getattr(training_args, 'max_completion_length', None),
                mask_truncated_completions=getattr(training_args, 'mask_truncated_completions', False),
                reward_method=reward_method, reward_weight_a=reward_weight_a,
                reward_weight_b=reward_weight_b,
                mixed_reward_alpha=mixed_reward_alpha,
                mixed_reward_alpha_mode=mixed_reward_alpha_mode,
                mixed_reward_alpha_zscore_mu=mixed_reward_alpha_zscore_mu,
                mixed_reward_alpha_zscore_sigma=mixed_reward_alpha_zscore_sigma,
                mixed_reward_alpha_zscore_lower=mixed_reward_alpha_zscore_lower,
                mixed_reward_alpha_zscore_upper=mixed_reward_alpha_zscore_upper,
                mixed_reward_corr_weight_mode=mixed_reward_corr_weight_mode,
                mixed_reward_corr_weight_threshold=mixed_reward_corr_weight_threshold,
                mixed_reward_corr_weight_exponent=mixed_reward_corr_weight_exponent,
                sum_scores_minmax=sum_scores_minmax,
                scale_rewards=scale_rewards,
                device=None,
                selected_indices_a_raw=peer_indices_str if train_model == "a" else None,
                selected_indices_b_raw=peer_indices_str if train_model == "b" else None,
                learn_config_json=None,
            )
            if other_count < N:
                peer_adv = peer_advantages_all.view(num_unique_prompts, N)[:, :other_count].reshape(-1)
            else:
                peer_adv = peer_advantages_all
            peer_data["advantages"] = peer_adv
            if truthcheck_enabled():
                expected_peer = int(num_unique_prompts * other_count)
                got_peer = int(peer_data["advantages"].size(0))
                if got_peer != expected_peer:
                    raise AssertionError(
                        f"[TRUTHCHECK][co.train.two_phase.peer_phase] "
                        f"peer advantages rows mismatch: got {got_peer}, expected {expected_peer}"
                    )

            # Load old_logps for peer phase IS ratio.
            peer_retokenize = bool(getattr(training_args, "peer_retokenize_old_logps", False))
            target_len = peer_data["completion_ids"].size(1)

            if peer_retokenize:
                # Retokenize mode: use train model's old logps on peer's responses
                # (cross old logps — same tokenizer, no old_logps_mask needed).
                _cross_ok, cross_old_logps_raw = cache.load_cross_logps(train_model, step)
                if _cross_ok and cross_old_logps_raw is not None:
                    cross_old_t = torch.tensor(cross_old_logps_raw, dtype=torch.float32)
                    cross_old_reshaped = cross_old_t.view(num_unique_prompts, N, -1)
                    cross_own_old = cross_old_reshaped[:, :other_count, :]
                    cross_own_old_flat = cross_own_old.reshape(num_unique_prompts * other_count, -1)

                    if cross_own_old_flat.size(1) > target_len:
                        cross_own_old_flat = cross_own_old_flat[:, :target_len]
                    elif cross_own_old_flat.size(1) < target_len:
                        pad_w = target_len - cross_own_old_flat.size(1)
                        cross_own_old_flat = torch.cat(
                            [cross_own_old_flat, cross_own_old_flat.new_zeros(cross_own_old_flat.size(0), pad_w)], dim=1,
                        )
                    peer_data["old_per_token_logps"] = cross_own_old_flat
                    logger.info(
                        f"Loaded cross old logps: {train_model}_old on {peer_model}_resp "
                        f"(shape: {cross_own_old_flat.shape})"
                    )
                elif is_strict_old_for_peer:
                    raise ValueError(
                        f"peer_retokenize_old_logps=True but cross old logps for "
                        f"model '{train_model}' not found in cache for step {step}"
                    )
                else:
                    logger.warning(
                        f"Cross old logps for '{train_model}' not found; "
                        f"peer phase will use on-policy ratio (no IS)"
                    )
            else:
                # Default HACPO mode: use peer model's own old logps (B_old).
                if peer_model == "a":
                    _, peer_old_logps_raw, _ = cache.load_logps_a(step)
                else:
                    _, peer_old_logps_raw, _ = cache.load_logps_b(step)

                if peer_old_logps_raw is not None:
                    peer_old_t = torch.tensor(peer_old_logps_raw, dtype=torch.float32)
                    peer_old_reshaped = peer_old_t.view(num_unique_prompts, N, -1)
                    peer_own_old = peer_old_reshaped[:, :other_count, :]
                    peer_own_old_flat = peer_own_old.reshape(num_unique_prompts * other_count, -1)

                    # Build old_logps_mask from the peer's cached completion_lengths
                    # (HACRL's old_log_prob_mask — reflects the generating model's tokenizer).
                    peer_compl_lengths = cache.load_completion_lengths(peer_model, step)
                    if peer_compl_lengths is not None:
                        pcl_t = torch.tensor(peer_compl_lengths, dtype=torch.long)
                        pcl_reshaped = pcl_t.view(num_unique_prompts, N)
                        pcl_own = pcl_reshaped[:, :other_count].reshape(-1)
                        old_logps_mask = torch.arange(target_len).unsqueeze(0) < pcl_own.unsqueeze(1)
                        peer_data["old_logps_mask"] = old_logps_mask.float()
                        logger.info(
                            f"Built old_logps_mask for peer ({peer_model}): "
                            f"peer_tokenizer avg_len={pcl_own.float().mean():.1f}, "
                            f"train_tokenizer target_len={target_len}"
                        )

                    if peer_own_old_flat.size(1) > target_len:
                        peer_own_old_flat = peer_own_old_flat[:, :target_len]
                    elif peer_own_old_flat.size(1) < target_len:
                        pad_w = target_len - peer_own_old_flat.size(1)
                        peer_own_old_flat = torch.cat(
                            [peer_own_old_flat, peer_own_old_flat.new_zeros(peer_own_old_flat.size(0), pad_w)], dim=1,
                        )
                    peer_data["old_per_token_logps"] = peer_own_old_flat
                    logger.info(f"Loaded peer ({peer_model}) per-token old logps for IS (shape: {peer_own_old_flat.shape})")
                elif is_strict_old_for_peer:
                    raise ValueError(
                        f"is_strict_old_for_peer=True but peer model '{peer_model}' old logps not found in cache for step {step}"
                    )
                else:
                    logger.warning(f"Peer model '{peer_model}' old logps not found; peer phase will use on-policy ratio (no IS)")

        # --- Peer special clip / reweight config ---
        peer_asym = bool(getattr(training_args, "peer_special_clip_asymmetric_enabled", False))
        peer_stepwise = bool(getattr(training_args, "peer_special_clip_stepwise_enabled", False))
        peer_delta = float(getattr(training_args, "peer_special_clip_delta", 0.8))
        peer_delta_step = float(getattr(training_args, "peer_special_clip_delta_step", 0.4))
        peer_is_reweight_enabled = bool(getattr(training_args, "peer_is_reweight_enabled", False))
        peer_is_reweight_alpha = float(getattr(training_args, "peer_is_reweight_alpha", 0.5)) if peer_is_reweight_enabled else 0.0

        # --- Merge self + peer into mixed batch (HACRL-style) ---
        # HACRL semantics: peer_delta is a *direct lower bound* for the ratio,
        # e.g. clamp(ratio, min=0.8, max=1.0).  Our loss uses epsilon form:
        #   clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
        # so we must convert: eps_low = 1.0 - delta.
        peer_eps_low = _effective_self_eps_low
        peer_eps_high = _effective_self_eps_high

        if peer_asym:
            peer_eps_high = 0.0
        if peer_stepwise:
            peer_eps_low = 1.0 - peer_delta

        if has_self_phase and has_peer_phase:
            # Merge into a single mixed-batch with model_source flags
            n_self = self_data["advantages"].size(0)
            n_peer = peer_data["advantages"].size(0)

            # For cross-tokenizer support: if peer has old_logps_mask, give self
            # samples a mask equal to their completion_mask (same tokenizer → same mask).
            if "old_logps_mask" in peer_data and "old_logps_mask" not in self_data:
                self_data["old_logps_mask"] = self_data["completion_mask"].clone().float()

            def _pad_and_cat(key):
                t_self = self_data.get(key)
                t_peer = peer_data.get(key)
                if t_self is None and t_peer is None:
                    return None
                if t_self is None:
                    t_self = torch.zeros_like(t_peer[:n_self]) if n_self > 0 else None
                if t_peer is None:
                    t_peer = torch.zeros_like(t_self[:n_peer]) if n_peer > 0 else None
                if t_self is None or t_peer is None:
                    return t_self if t_self is not None else t_peer
                if t_self.dim() > 1 and t_peer.dim() > 1 and t_self.size(1) != t_peer.size(1):
                    max_len = max(t_self.size(1), t_peer.size(1))
                    if t_self.size(1) < max_len:
                        pad = t_self.new_zeros(t_self.size(0), max_len - t_self.size(1))
                        t_self = torch.cat([t_self, pad], dim=1)
                    if t_peer.size(1) < max_len:
                        pad = t_peer.new_zeros(t_peer.size(0), max_len - t_peer.size(1))
                        t_peer = torch.cat([t_peer, pad], dim=1)
                return torch.cat([t_self, t_peer], dim=0)

            mixed_data: Dict[str, Any] = {}
            for key in ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
                         "advantages", "old_per_token_logps", "ref_per_token_logps",
                         "old_logps_mask"]:
                mixed_data[key] = _pad_and_cat(key)

            # model_source: 0=self, 1=peer
            mixed_data["model_source"] = torch.cat([
                torch.zeros(n_self, dtype=torch.long),
                torch.ones(n_peer, dtype=torch.long),
            ], dim=0)
            if truthcheck_enabled():
                ms = mixed_data["model_source"]
                n_self_chk = int((ms == 0).sum().item())
                n_peer_chk = int((ms == 1).sum().item())
                if n_self_chk != int(n_self) or n_peer_chk != int(n_peer):
                    raise AssertionError(
                        f"[TRUTHCHECK][co.train.two_phase.model_source] "
                        f"model_source count mismatch: self={n_self_chk}/{n_self}, peer={n_peer_chk}/{n_peer}"
                    )

            logger.info(f"Mixed batch: {n_self} self + {n_peer} peer = {n_self + n_peer} total")
            training_data = mixed_data
        elif has_self_phase:
            training_data = self_data
        else:
            # peer only: add model_source=1
            n_peer = peer_data["advantages"].size(0)
            peer_data["model_source"] = torch.ones(n_peer, dtype=torch.long)
            training_data = peer_data

        if torch.cuda.is_available():
            v = os.environ.get("CO_EMPTY_CACHE_AFTER_PREPARE", "0").strip().lower()
            if v not in ("", "0", "false", "no", "off", "none"):
                torch.cuda.empty_cache()

        # --- Configure clip context for mixed batch ---
        _clip_ctx["epsilon_low"] = _effective_self_eps_low
        _clip_ctx["epsilon_high"] = _effective_self_eps_high
        _clip_ctx["peer_epsilon_low"] = peer_eps_low
        _clip_ctx["peer_epsilon_high"] = peer_eps_high
        _clip_ctx["is_reweight_alpha"] = peer_is_reweight_alpha
        _clip_ctx["capability_weight"] = w_jk

        if peer_stepwise:
            _clip_ctx["_stepwise_cfg"] = {
                "delta": peer_delta,
                "delta_step": peer_delta_step,
            }

        logger.info(
            f"Mixed-batch clip: self=[{_clip_ctx['epsilon_low']}, {_clip_ctx['epsilon_high']}], "
            f"peer=[{peer_eps_low}, {peer_eps_high}], "
            f"is_reweight_alpha={peer_is_reweight_alpha}, capability_weight={w_jk:.4f}, "
            f"stepwise={peer_stepwise}"
        )
        tc_check_hacpo_mixed_batch(
            where="co.train.two_phase.mixed_batch_before_loop",
            model_source=training_data.get("model_source"),
            advantages=training_data["advantages"],
            old_per_token_logps=training_data.get("old_per_token_logps"),
            capability_weight=float(_clip_ctx["capability_weight"]),
            peer_epsilon_low=float(_clip_ctx["peer_epsilon_low"]) if _clip_ctx.get("peer_epsilon_low") is not None else None,
            peer_epsilon_high=float(_clip_ctx["peer_epsilon_high"]) if _clip_ctx.get("peer_epsilon_high") is not None else None,
        )

        avg_loss = _run_training_loop(
            trainer, training_data, training_args, phase_label="mixed ",
            clip_ctx=_clip_ctx,
        )
        _clip_ctx.pop("_stepwise_cfg", None)
        logger.info(f"Step {step}, Mixed-batch loss: {avg_loss:.4f}")

    else:
        # ---------------------------------------------------------------
        # Legacy mixed mode (original behaviour)
        # ---------------------------------------------------------------
        try:
            training_data = prepare_training_data_from_cache(
                cache=cache, step=step, train_model=train_model,
                unique_prompts=step_prompts, tokenizer=tokenizer,
                num_generations_per_model=N,
                max_prompt_length=getattr(training_args, 'max_prompt_length', None),
                max_completion_length=getattr(training_args, 'max_completion_length', None),
                mask_truncated_completions=getattr(training_args, 'mask_truncated_completions', False),
                reward_method=reward_method, reward_weight_a=reward_weight_a,
                reward_weight_b=reward_weight_b,
                mixed_reward_alpha=mixed_reward_alpha,
                mixed_reward_alpha_mode=mixed_reward_alpha_mode,
                mixed_reward_alpha_zscore_mu=mixed_reward_alpha_zscore_mu,
                mixed_reward_alpha_zscore_sigma=mixed_reward_alpha_zscore_sigma,
                mixed_reward_alpha_zscore_lower=mixed_reward_alpha_zscore_lower,
                mixed_reward_alpha_zscore_upper=mixed_reward_alpha_zscore_upper,
                mixed_reward_corr_weight_mode=mixed_reward_corr_weight_mode,
                mixed_reward_corr_weight_threshold=mixed_reward_corr_weight_threshold,
                mixed_reward_corr_weight_exponent=mixed_reward_corr_weight_exponent,
                sum_scores_minmax=bool(getattr(training_args, "sum_scores_minmax", False)),
                scale_rewards=scale_rewards,
                device=None,
                selected_indices_a_raw=getattr(training_args, "selected_indices_a", None),
                selected_indices_b_raw=getattr(training_args, "selected_indices_b", None),
                learn_config_json=getattr(training_args, "learn_config", None),
            )
        except Exception as e:
            logger.error(f"Failed to prepare training data for step {step}: {e}")
            raise

        if torch.cuda.is_available():
            v = os.environ.get("CO_EMPTY_CACHE_AFTER_PREPARE", "0").strip().lower()
            if v not in ("", "0", "false", "no", "off", "none"):
                torch.cuda.empty_cache()

        avg_loss = _run_training_loop(trainer, training_data, training_args)
        logger.info(f"Step {step}, Loss: {avg_loss:.4f}")
    
    # Save checkpoint (always save after training step)
    # Use step number from cache (this matches the step from generate_and_score.py)
    # Save as checkpoint-{step} format (like transformers.Trainer)
    checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if trainer.accelerator.is_main_process:
        logger.info("Starting checkpoint save (this may take a while under ZeRO-3)...")
    trainer.save_model(checkpoint_dir)
    logger.info(f"Checkpoint saved to {checkpoint_dir}")

    cap_reporter.write_summary()

    # Note: We don't save a "final_model" anymore - only checkpoint-k directories
    # The latest checkpoint can be found using get_last_checkpoint() or get_latest_checkpoint() in shell script


if __name__ == "__main__":
    parser = TrlParser((CoIntuitorScriptArguments, CoIntuitorConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
