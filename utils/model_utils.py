# coding=utf-8
"""Model and tokenizer loading for inference pipelines (generate, score, logps, eval).

Training-specific model loading (which needs ``CoIntuitorConfig``, ``ModelConfig``,
gradient checkpointing, etc.) lives in ``src/co_intuitor_train.py``.
"""

import gc
import json
import logging
import os
from typing import Optional, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.attn import maybe_add_attn_implementation
from utils.generate import _ensure_trust_remote_code_files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mistral3 / Ministral3 compatibility
# ---------------------------------------------------------------------------

def _fixup_ministral3_text_config(tc_obj):
    """Ensure a ``ministral3`` text-backbone config has the fields vLLM needs.

    1. ``architectures`` must be a list (vLLM does ``tuple(cfg.architectures)``
       which crashes on ``None``).
    2. ``model_type`` must be ``"mistral"`` so vLLM finds the right model class.
    """
    if getattr(tc_obj, "architectures", None) is None:
        tc_obj.architectures = ["MistralForCausalLM"]
    if getattr(tc_obj, "model_type", None) == "ministral3":
        tc_obj.model_type = "mistral"


_ministral3_patched = False


def patch_ministral3_config():
    """Monkey-patch ``Mistral3Config.__init__`` so that a ``text_config`` dict
    with ``model_type == "ministral3"`` is pre-instantiated before the
    ``CONFIG_MAPPING["ministral3"]`` lookup (missing in transformers < 5.0).

    Covers ALL callers including vLLM's internal ``AutoConfig.from_pretrained``.
    Safe to call multiple times (idempotent).
    """
    global _ministral3_patched
    if _ministral3_patched:
        return
    _ministral3_patched = True

    try:
        from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    except ImportError:
        return

    _orig_init = Mistral3Config.__init__

    def _safe_init(self, *args, **kwargs):
        tc = kwargs.get("text_config")
        if isinstance(tc, dict) and tc.get("model_type") == "ministral3":
            from transformers import MistralConfig
            try:
                from transformers.models.ministral3.configuration_ministral3 import (
                    Ministral3Config as _Cls,
                )
            except ImportError:
                _Cls = MistralConfig
            tc_obj = _Cls(**tc)
            _fixup_ministral3_text_config(tc_obj)
            kwargs["text_config"] = tc_obj
        _orig_init(self, *args, **kwargs)

    Mistral3Config.__init__ = _safe_init


patch_ministral3_config()


# ---------------------------------------------------------------------------
# Phi-3 / Phi-4 hub modeling (trust_remote_code)
# ---------------------------------------------------------------------------

_phi_hub_losskwargs_patched = False


def patch_phi_hub_losskwargs_import() -> None:
    """Hub ``modeling_phi3.py`` does ``from transformers.utils import LossKwargs``.

    In transformers 4.55+, that type was renamed to ``TransformersKwargs`` and is
    no longer re-exported as ``LossKwargs`` from ``transformers.utils``, so dynamic
    modules fail to import.  Alias it for backward compatibility (idempotent).
    """
    global _phi_hub_losskwargs_patched
    if _phi_hub_losskwargs_patched:
        return
    _phi_hub_losskwargs_patched = True

    import transformers.utils as tu

    if getattr(tu, "LossKwargs", None) is not None:
        return
    try:
        from transformers.utils.generic import TransformersKwargs
    except ImportError:
        return
    tu.LossKwargs = TransformersKwargs  # type: ignore[attr-defined]


patch_phi_hub_losskwargs_import()


_qwen35_vllm_text_registry_patched = False

_QWEN35_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def patch_vllm_qwen35_text_registry() -> None:
    """Register Qwen3.5 text-only architectures in vLLM when omitted upstream.

    Some vLLM releases ship the implementation classes
    ``Qwen3_5ForCausalLM`` / ``Qwen3_5MoeForCausalLM`` but only register the
    multimodal ``*ForConditionalGeneration`` variants. Text-only checkpoints
    saved by this pipeline then miss the normal text-generation registry path
    and can be misrouted into multimodal processing during eval.

    This patch is idempotent and only replaces a text-only registry entry when
    it is absent or lacks vLLM's hybrid attention/Mamba interface.
    """
    global _qwen35_vllm_text_registry_patched
    if _qwen35_vllm_text_registry_patched:
        return
    _qwen35_vllm_text_registry_patched = True

    try:
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            Qwen3_5ForConditionalGenerationConfig,
        )
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception:
        return

    registrations = {
        "Qwen3_5ForCausalLM": "utils.vllm_qwen35_compat:Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM": "utils.vllm_qwen35_compat:Qwen3_5MoeForCausalLM",
    }

    for arch, target in registrations.items():
        registered = ModelRegistry.models.get(arch)
        needs_compat = registered is None
        if registered is not None:
            try:
                needs_compat = not registered.inspect_model_cls().is_hybrid
            except Exception:
                needs_compat = True
        if needs_compat:
            ModelRegistry.register_model(arch, target)
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)


def is_qwen35_model_type(model_type: Optional[str]) -> bool:
    """Return whether *model_type* belongs to the Qwen3.5 model family."""
    return model_type in _QWEN35_MODEL_TYPES


def is_qwen35_text_model_type(model_type: Optional[str]) -> bool:
    """Return whether *model_type* already denotes a text-only Qwen3.5 model."""
    return model_type in {"qwen3_5_text", "qwen3_5_moe_text"}


def has_qwen35_container_state_dict_keys(state_dict: Optional[Mapping[str, torch.Tensor]]) -> bool:
    """Detect the multimodal-container namespace in a Qwen3.5 state dict.

    Transformers can expose a Qwen3.5 text backbone through a model whose
    config still reports ``qwen3_5``.  The key prefix is therefore a stronger
    signal than ``config.model_type`` at checkpoint-save time.
    """
    if not state_dict:
        return False
    return any(
        key.startswith(("model.language_model.", "language_model."))
        for key in state_dict
    )


def normalize_qwen35_text_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite Qwen3.5 text-only checkpoint keys to the vLLM/HF text layout.

    The training pipeline currently serializes text-only Qwen3.5 checkpoints
    with an extra ``model.language_model.`` namespace because the underlying
    base model is still the multimodal container. vLLM's text-only loader, and
    the conventional HuggingFace text checkpoint layout, expect ``model.*``.

    This helper rewrites the namespace without changing tensor contents.
    """
    rewritten: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith("model.language_model."):
            new_key = key.replace("model.language_model.", "model.", 1)
        elif key.startswith("language_model."):
            new_key = key.replace("language_model.", "model.", 1)
        elif key.startswith(
            (
                "visual.",
                "vision.",
                "video.",
                "image.",
                "model.visual.",
                "model.vision.",
                "model.video.",
                "model.image.",
                "model.visual_merger.",
            )
        ):
            continue
        else:
            new_key = key
        if new_key in rewritten:
            raise ValueError(
                "Qwen3.5 state-dict normalization produced a duplicate key: "
                f"{new_key}"
            )
        rewritten[new_key] = tensor
    return rewritten


def _detect_model_type(model_path: str) -> Optional[str]:
    """Read model_type from config.json without instantiating the full config."""
    config_file = os.path.join(model_path, "config.json")
    if os.path.isfile(config_file):
        with open(config_file) as f:
            return json.load(f).get("model_type")
    return None


def _build_mistral3_config(model_path: str):
    """Build a ``Mistral3Config`` from disk, pre-instantiating ``text_config``
    so that the ``CONFIG_MAPPING["ministral3"]`` lookup (missing in
    transformers < 5.0) is never hit.
    """
    from transformers import MistralConfig
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    config_file = os.path.join(model_path, "config.json")
    with open(config_file) as f:
        raw = json.load(f)

    text_cfg = raw.get("text_config")
    if isinstance(text_cfg, dict) and text_cfg.get("model_type") == "ministral3":
        try:
            from transformers.models.ministral3.configuration_ministral3 import (
                Ministral3Config as _TextCls,
            )
        except ImportError:
            _TextCls = MistralConfig
        tc_obj = _TextCls(**text_cfg)
        _fixup_ministral3_text_config(tc_obj)
        raw["text_config"] = tc_obj

    return Mistral3Config(**raw)


def _load_mistral3_language_model(model_path: str, model_init_kwargs: dict):
    """Load a ``mistral3`` multimodal checkpoint and return a model whose
    ``forward()`` produces ``.logits``.

    If the inner ``language_model`` is a full CausalLM (has ``lm_head``), we
    extract it to free the vision encoder.  Otherwise the backbone is a bare
    transformer whose output has no ``.logits``; in that case we return the
    complete ``Mistral3ForConditionalGeneration`` whose ``forward()`` *does*
    produce ``.logits`` for text-only inputs.
    """
    from transformers import Mistral3ForConditionalGeneration

    config = _build_mistral3_config(model_path)

    logger.info("Loading Mistral3 via Mistral3ForConditionalGeneration...")
    full_model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path, config=config, **model_init_kwargs,
    )

    if hasattr(full_model, "language_model"):
        lm = full_model.language_model
        if hasattr(lm, "lm_head"):
            logger.info("Extracted language_model (CausalLM with lm_head).")
            del full_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return lm
        logger.info(
            "language_model has no lm_head; keeping full "
            "Mistral3ForConditionalGeneration for .logits output."
        )

    return full_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tokenizer(model_path: str):
    """Load a tokenizer for inference (left-padded, trust_remote_code=True)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except OSError:
        if _ensure_trust_remote_code_files(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            raise
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(model_path: str, device: str = "cuda"):
    """Load a causal-LM for inference (scoring / logps).

    In multi-process mode (accelerate) each process loads the model on its own GPU.
    No model sharding—data parallelism is used instead.
    """
    model_init_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        if device in ("auto", "balanced", "balanced_low_0", "sequential"):
            model_init_kwargs["device_map"] = device
        elif isinstance(device, str) and device.startswith("cuda"):
            model_init_kwargs["device_map"] = {"": device}
        elif device == "cuda":
            model_init_kwargs["device_map"] = {"": "cuda:0"}
        else:
            model_init_kwargs["device_map"] = None

    maybe_add_attn_implementation(model_init_kwargs)

    model_type = _detect_model_type(model_path)

    # For mistral3 multimodal checkpoints, AutoModelForCausalLM has no mapping
    # and CONFIG_MAPPING lacks "ministral3" in transformers < 5.0.
    # Load via the dedicated class (with a manually-built config) and extract
    # the language_model backbone.
    if model_type == "mistral3":
        try:
            return _load_mistral3_language_model(model_path, model_init_kwargs)
        except TypeError:
            if "attn_implementation" in model_init_kwargs:
                kw = dict(model_init_kwargs)
                kw.pop("attn_implementation", None)
                logger.warning("Mistral3 rejected attn_implementation; retrying without it.")
                return _load_mistral3_language_model(model_path, kw)
            raise

    # All other models (including ministral3 text-only after patching).
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
    except OSError:
        if _ensure_trust_remote_code_files(model_path):
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        else:
            raise
    except TypeError:
        if "attn_implementation" in model_init_kwargs:
            model_init_kwargs = dict(model_init_kwargs)
            model_init_kwargs.pop("attn_implementation", None)
            logger.warning("Model init rejected attn_implementation; retrying without it.")
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        else:
            raise
    return model
