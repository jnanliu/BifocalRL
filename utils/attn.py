from __future__ import annotations

import os
from typing import Any, Dict, Optional


def resolve_attn_implementation() -> Optional[str]:
    """
    Resolve the attention implementation to request from Transformers models.

    Priority:
    - CO_ATTN_IMPLEMENTATION env var if set (supports "none"/"null" to disable)
    - If flash-attn is importable -> "flash_attention_2"
    - Else -> "sdpa"
    """
    v = os.environ.get("CO_ATTN_IMPLEMENTATION", "").strip()
    if v:
        if v.lower() in {"none", "null", "disable", "disabled"}:
            return None
        return v

    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def maybe_add_attn_implementation(model_init_kwargs: Dict[str, Any]) -> None:
    """
    Mutate `model_init_kwargs` in-place to include attn_implementation if not present.
    """
    if "attn_implementation" in model_init_kwargs:
        return
    attn_impl = resolve_attn_implementation()
    if attn_impl is None:
        return
    model_init_kwargs["attn_implementation"] = attn_impl

