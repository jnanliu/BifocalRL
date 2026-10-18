"""vLLM compatibility classes for text-only Qwen3.5 checkpoints."""

from __future__ import annotations

import torch

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM as _Qwen3_5MoeForCausalLM,
)


class _Qwen3_5TextHybridMixin:
    # vLLM 0.23 omits the hybrid marker and state helpers from its text-only
    # Qwen3.5 classes even though their decoder contains both attention and GDN.
    is_hybrid = True
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Build M-RoPE positions for a text-only Qwen3.5 request."""
        if mm_features:
            raise ValueError(
                "Qwen3.5 text-only checkpoints do not accept multimodal features"
            )
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()


class Qwen3_5ForCausalLM(_Qwen3_5TextHybridMixin, _Qwen3_5ForCausalLM):
    pass


class Qwen3_5MoeForCausalLM(_Qwen3_5TextHybridMixin, _Qwen3_5MoeForCausalLM):
    pass
