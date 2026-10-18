# coding=utf-8
"""
Drop-in replacements for trl utilities used by co-learning.

This module removes the hard dependency on the `trl` package by providing
standalone implementations of the specific functions and classes that
co-learning actually uses. This allows freely updating `transformers`,
`vllm`, `lighteval`, etc. without trl version-pinning conflicts.

Replaced items:
- trl.trainer.utils.selective_log_softmax
- trl.trainer.utils.pad
- trl.data_utils.maybe_apply_chat_template
- trl.models.unwrap_model_for_generation
- trl.ModelConfig
- trl.ScriptArguments
- trl.TrlParser
- trl.get_quantization_config / trl.get_kbit_device_map
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from accelerate import Accelerator
    from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# selective_log_softmax  (was: trl.trainer.utils.selective_log_softmax)
# ---------------------------------------------------------------------------

def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Memory-efficient ``log_softmax → gather`` in one pass.

    Equivalent to ``torch.gather(logits.log_softmax(-1), -1, index.unsqueeze(-1)).squeeze(-1)``.
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in (torch.float32, torch.float64):
        selected_logits = torch.gather(logits, dim=-1, index=index)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)
    return per_token_logps


# ---------------------------------------------------------------------------
# pad  (was: trl.trainer.utils.pad)
# ---------------------------------------------------------------------------

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    """Pad a list of tensors to the same shape along the first dimension."""
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    output = torch.full(
        (len(tensors), *output_shape),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


# ---------------------------------------------------------------------------
# maybe_apply_chat_template  (was: trl.data_utils.maybe_apply_chat_template)
# ---------------------------------------------------------------------------

def maybe_apply_chat_template(example: dict, tokenizer, **kwargs) -> dict:
    """Apply the tokenizer's chat template to ``example["prompt"]`` if available.

    Mirrors trl.data_utils.maybe_apply_chat_template behaviour: if the prompt
    is a list of message dicts *and* the tokenizer has a chat_template, render
    it into a string. Otherwise return the example unchanged.
    """
    if tokenizer is None:
        return example

    prompt = example.get("prompt")
    if prompt is None:
        return example

    if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
        if getattr(tokenizer, "chat_template", None) is not None:
            rendered = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            example = dict(example)
            example["prompt"] = rendered
    return example


# ---------------------------------------------------------------------------
# unwrap_model_for_generation  (was: trl.models.unwrap_model_for_generation)
# ---------------------------------------------------------------------------

@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
    generation_kwargs: dict | None = None,
):
    """Context manager to unwrap distributed models for generation.

    Handles DeepSpeed ZeRO-3 parameter gathering and gradient-checkpoint toggling.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    is_gc = getattr(unwrapped_model, "is_gradient_checkpointing", False)
    if is_gc:
        unwrapped_model.gradient_checkpointing_disable()

    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    is_zero3 = ds_plugin is not None and ds_plugin.zero_stage == 3

    if is_zero3 and gather_deepspeed3_params:
        import deepspeed

        with deepspeed.zero.GatheredParameters(model.parameters()):
            # Remove/re-add optimizer hooks to avoid ZeRO bookkeeping errors
            _remove_hooks(model)
            try:
                yield accelerator.unwrap_model(model)
            finally:
                _add_hooks(model)
    else:
        yield unwrapped_model

    if is_gc:
        unwrapped_model.gradient_checkpointing_enable()


def _remove_hooks(model) -> None:
    """Remove DeepSpeed optimizer hooks (ZeRO-3)."""
    if not hasattr(model, "optimizer") or model.optimizer is None:
        return
    offload = getattr(model.optimizer, "parameter_offload", model.optimizer)
    import itertools

    for param in itertools.chain(
        *(
            [p for _, p in sub.named_parameters(recurse=False)]
            + list(getattr(sub, "ds_external_parameters", lambda: [])())
            for sub in offload.module.modules()
        )
    ):
        if hasattr(param, "ds_active_sub_modules"):
            param.ds_active_sub_modules.clear()
    for hook in getattr(offload, "forward_hooks", []):
        hook.remove()
    for hook in getattr(offload, "backward_hooks", []):
        hook.remove()
    offload.forward_hooks = []
    offload.backward_hooks = []


def _add_hooks(model) -> None:
    """Re-register DeepSpeed optimizer hooks (ZeRO-3)."""
    if not hasattr(model, "optimizer") or model.optimizer is None:
        return
    offload = getattr(model.optimizer, "parameter_offload", model.optimizer)
    try:
        import deepspeed
        from packaging.version import Version

        if Version(deepspeed.__version__) >= Version("0.16.4"):
            offload._register_deepspeed_module(offload.module)
        else:
            offload._register_hooks_recursively(offload.module)
    except Exception:
        try:
            offload._register_hooks_recursively(offload.module)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ModelConfig  (was: trl.ModelConfig)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Minimal model configuration matching the subset of trl.ModelConfig used by co-learning."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use (branch, tag, or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Override default dtype. Options: bfloat16, float16, float32, auto."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code from model repos."},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation to use. Options: eager, sdpa, flash_attention_2."},
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA R value."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha value."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout value."})
    lora_target_modules: Optional[list[str]] = field(
        default=None, metadata={"help": "LoRA target modules."}
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None, metadata={"help": "Model layers to unfreeze & train."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Use 8-bit quantization."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Use 4-bit quantization."})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type for 4-bit."})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "Use nested quantization."})


# ---------------------------------------------------------------------------
# ScriptArguments  (was: trl.ScriptArguments)
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    """Minimal script arguments matching the subset of trl.ScriptArguments used by co-learning."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the dataset from the HuggingFace Hub."},
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "Configuration name of the dataset."}
    )
    dataset_train_split: str = field(
        default="train", metadata={"help": "Training split name."}
    )
    dataset_test_split: Optional[str] = field(
        default=None, metadata={"help": "Test split name."}
    )


# ---------------------------------------------------------------------------
# TrlParser  (was: trl.TrlParser)
# ---------------------------------------------------------------------------

class TrlParser:
    """Lightweight replacement for trl.TrlParser using HfArgumentParser.

    Supports the same ``TrlParser(dataclass_types)`` + ``.parse_args_and_config()`` API,
    including ``--config config.yaml`` for loading parameters from YAML files
    (with command-line arguments taking priority as overrides).
    """

    def __init__(self, dataclass_types):
        from transformers import HfArgumentParser

        if not isinstance(dataclass_types, (list, tuple)):
            dataclass_types = [dataclass_types]
        self._dataclass_types = dataclass_types
        self._parser = HfArgumentParser(dataclass_types)

    def parse_args_and_config(self):
        # Check for --config in sys.argv and handle YAML loading (matching trl.TrlParser).
        config_file = None
        argv = list(sys.argv[1:])
        if "--config" in argv:
            idx = argv.index("--config")
            if idx + 1 < len(argv):
                config_file = argv[idx + 1]
                # Remove --config and its value so HfArgumentParser doesn't choke on them.
                argv = argv[:idx] + argv[idx + 2:]

        if config_file is not None:
            import yaml

            with open(config_file, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

            # Handle env vars section (trl convention).
            import os
            env_vars = yaml_data.pop("env", None)
            if isinstance(env_vars, dict):
                for k, v in env_vars.items():
                    os.environ.setdefault(str(k), str(v))

            # Build default args from YAML, then let CLI args override.
            yaml_args = []
            for k, v in yaml_data.items():
                cli_key = f"--{k}"
                # Only add YAML value if not already in CLI argv (CLI takes priority).
                if cli_key not in argv:
                    if isinstance(v, bool):
                        if v:
                            yaml_args.append(cli_key)
                        # For False booleans, we need explicit --no_<key> or skip
                        # HfArgumentParser uses store_true, so not adding is equivalent to False.
                    elif isinstance(v, list):
                        yaml_args.append(cli_key)
                        yaml_args.extend(str(item) for item in v)
                    else:
                        yaml_args.append(cli_key)
                        yaml_args.append(str(v))

            combined_argv = yaml_args + argv
            return self._parser.parse_args_into_dataclasses(args=combined_argv)

        return self._parser.parse_args_into_dataclasses()

    # Alias used by some call sites
    parse = parse_args_and_config


# ---------------------------------------------------------------------------
# get_quantization_config / get_kbit_device_map  (was: trl.get_quantization_config / trl.get_kbit_device_map)
# ---------------------------------------------------------------------------

def _resolve_torch_dtype(model_args) -> torch.dtype:
    """Resolve the torch dtype from model_args, defaulting to bfloat16."""
    raw = getattr(model_args, "torch_dtype", None)
    if raw is None:
        return torch.bfloat16
    if isinstance(raw, torch.dtype):
        return raw
    # Handle string values like "bfloat16", "float16", "float32", "auto"
    dtype_str = str(raw).strip().lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def get_quantization_config(model_args) -> Optional[Any]:
    """Build a BitsAndBytesConfig from model_args if quantization is requested."""
    load_in_4bit = getattr(model_args, "load_in_4bit", False)
    load_in_8bit = getattr(model_args, "load_in_8bit", False)
    if not load_in_4bit and not load_in_8bit:
        return None
    try:
        from transformers import BitsAndBytesConfig

        if load_in_4bit:
            compute_dtype = _resolve_torch_dtype(model_args)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=getattr(model_args, "bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=getattr(model_args, "use_bnb_nested_quant", False),
                bnb_4bit_compute_dtype=compute_dtype,
            )
        return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        return None


def get_kbit_device_map() -> Optional[dict]:
    """Return a device_map suitable for quantized models.

    Uses accelerate's PartialState to determine the correct local process index
    for multi-GPU setups. Falls back to ``{"":  0}`` when accelerate is not
    available or CUDA/XPU is not present.
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return None
    try:
        from accelerate import PartialState
        return {"": PartialState().local_process_index}
    except Exception:
        # Fallback for environments without accelerate or without GPU.
        if torch.cuda.is_available():
            return {"": 0}
        return None
