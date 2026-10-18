from pathlib import Path
import json
import sys
from types import SimpleNamespace

import torch
import pytest
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.model_utils import (  # noqa: E402
    has_qwen35_container_state_dict_keys,
    is_qwen35_model_type,
    normalize_qwen35_text_state_dict,
)
from utils.single_model_trainer import SingleModelTrainer  # noqa: E402


def test_qwen35_state_dict_normalization_removes_container_namespace():
    state = {
        "model.language_model.layers.0.weight": torch.tensor([1.0]),
        "model.language_model.norm.weight": torch.tensor([2.0]),
        "model.visual.patch_embed.weight": torch.tensor([3.0]),
        "lm_head.weight": torch.tensor([4.0]),
    }

    assert has_qwen35_container_state_dict_keys(state)
    normalized = normalize_qwen35_text_state_dict(state)

    assert set(normalized) == {"model.layers.0.weight", "model.norm.weight", "lm_head.weight"}
    assert not has_qwen35_container_state_dict_keys(normalized)
    assert torch.equal(normalized["model.layers.0.weight"], state["model.language_model.layers.0.weight"])


def test_qwen35_model_type_detection_covers_source_and_text_configs():
    assert is_qwen35_model_type("qwen3_5")
    assert is_qwen35_model_type("qwen3_5_text")
    assert is_qwen35_model_type("qwen3_5_moe")
    assert is_qwen35_model_type("qwen3_5_moe_text")
    assert not is_qwen35_model_type("qwen2")


def test_qwen35_vllm_text_compat_exposes_hybrid_state_interfaces():
    pytest.importorskip("vllm")
    try:
        from vllm.model_executor.models.interfaces import supports_mrope
        from utils.vllm_qwen35_compat import (
            Qwen3_5ForCausalLM,
            Qwen3_5MoeForCausalLM,
        )
    except ImportError:
        pytest.skip("Installed vLLM does not include Qwen3.5 text model classes")

    for model_class in (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM):
        assert model_class.is_hybrid is True
        assert callable(model_class.get_mamba_state_dtype_from_config)
        assert callable(model_class.get_mamba_state_shape_from_config)
        assert callable(model_class.get_mamba_state_copy_func)
        assert model_class.supports_mrope is True

        model = model_class.__new__(model_class)
        assert supports_mrope(model)
        positions, delta = model.get_mrope_input_positions([10, 11, 12, 13], [])
        assert positions.tolist() == [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
        assert delta == 0

        with pytest.raises(ValueError, match="do not accept multimodal features"):
            model.get_mrope_input_positions([10], [object()])


def test_qwen35_state_dict_normalization_rejects_key_collisions():
    state = {
        "model.language_model.norm.weight": torch.tensor([1.0]),
        "model.norm.weight": torch.tensor([2.0]),
    }

    with pytest.raises(ValueError, match="duplicate key"):
        normalize_qwen35_text_state_dict(state)


def test_ensure_qwen35_text_config_converts_multimodal_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "transformers_version": "5.12.1",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 2048,
                },
            }
        ),
        encoding="utf-8",
    )

    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer._ensure_qwen35_text_config(str(tmp_path))
    result = json.loads(config_path.read_text(encoding="utf-8"))

    assert result["model_type"] == "qwen3_5_text"
    assert result["architectures"] == ["Qwen3_5ForCausalLM"]
    assert result["hidden_size"] == 2048
    assert result["transformers_version"] == "5.12.1"


def test_ensure_qwen35_text_config_repairs_wrong_architecture(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "hidden_size": 2048,
            }
        ),
        encoding="utf-8",
    )

    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer._ensure_qwen35_text_config(str(tmp_path))
    result = json.loads(config_path.read_text(encoding="utf-8"))

    assert result["model_type"] == "qwen3_5_text"
    assert result["architectures"] == ["Qwen3_5ForCausalLM"]
    assert result["hidden_size"] == 2048


def test_save_normalizes_qwen35_source_model_state_dict(tmp_path):
    class DummyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self):
            config = PretrainedConfig()
            config.model_type = "qwen3_5"
            super().__init__(config)

        def save_pretrained(self, save_directory, **kwargs):
            assert kwargs["save_original_format"] is False
            from safetensors.torch import save_file

            save_file(
                kwargs["state_dict"],
                Path(save_directory) / "model.safetensors",
                metadata={"format": "pt"},
            )

    class DummyAccelerator:
        @staticmethod
        def unwrap_model(model):
            return model

    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.model = DummyModel()
    trainer.accelerator = DummyAccelerator()
    trainer.args = SimpleNamespace(save_safetensors=True)
    trainer.tokenizer = None
    trainer._ensure_qwen35_text_config = lambda output_dir: (  # type: ignore[method-assign]
        Path(output_dir, "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_text",
                    "architectures": ["Qwen3_5ForCausalLM"],
                }
            ),
            encoding="utf-8",
        )
    )
    trainer._maybe_restore_non_weight_files_from_source = (  # type: ignore[method-assign]
        lambda *args, **kwargs: None
    )
    trainer._maybe_include_remote_code_files = (  # type: ignore[method-assign]
        lambda *args, **kwargs: None
    )

    trainer._save(
        str(tmp_path),
        state_dict={
            "model.language_model.layers.0.weight": torch.tensor([1.0]),
            "model.language_model.norm.weight": torch.tensor([2.0]),
            "model.visual.patch_embed.weight": torch.tensor([3.0]),
        },
    )

    with safe_open(tmp_path / "model.safetensors", framework="pt", device="cpu") as f:
        assert set(f.keys()) == {"model.layers.0.weight", "model.norm.weight"}


def test_save_detects_qwen35_through_deepspeed_style_wrapper(tmp_path):
    class DummyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self):
            config = PretrainedConfig()
            config.model_type = "qwen3_5_text"
            super().__init__(config)

        def save_pretrained(self, save_directory, **kwargs):
            assert kwargs["save_original_format"] is False
            from safetensors.torch import save_file

            save_file(
                kwargs["state_dict"],
                Path(save_directory) / "model.safetensors",
                metadata={"format": "pt"},
            )

    class DeepSpeedStyleWrapper:
        pass

    unwrapped_model = DummyModel()

    class DummyAccelerator:
        @staticmethod
        def unwrap_model(model):
            assert isinstance(model, DeepSpeedStyleWrapper)
            return unwrapped_model

    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.model = DeepSpeedStyleWrapper()
    trainer.accelerator = DummyAccelerator()
    trainer.args = SimpleNamespace(save_safetensors=True)
    trainer.tokenizer = None
    trainer._ensure_qwen35_text_config = lambda output_dir: (  # type: ignore[method-assign]
        Path(output_dir, "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_text",
                    "architectures": ["Qwen3_5ForCausalLM"],
                }
            ),
            encoding="utf-8",
        )
    )
    trainer._maybe_restore_non_weight_files_from_source = (  # type: ignore[method-assign]
        lambda *args, **kwargs: None
    )
    trainer._maybe_include_remote_code_files = (  # type: ignore[method-assign]
        lambda *args, **kwargs: None
    )

    # This matches accelerator.get_state_dict(DeepSpeedEngine): the gathered
    # training keys already use the internal text-model namespace.
    trainer._save(
        str(tmp_path),
        state_dict={
            "model.layers.0.weight": torch.tensor([1.0]),
            "model.norm.weight": torch.tensor([2.0]),
        },
    )

    with safe_open(tmp_path / "model.safetensors", framework="pt", device="cpu") as f:
        assert set(f.keys()) == {"model.layers.0.weight", "model.norm.weight"}


def test_validate_qwen35_text_checkpoint_rejects_container_keys(tmp_path):
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "architectures": ["Qwen3_5ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {"model.language_model.norm.weight": torch.tensor([1.0])},
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )

    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.args = SimpleNamespace(save_safetensors=True)
    with pytest.raises(RuntimeError, match="multimodal-container weight keys"):
        trainer._validate_qwen35_text_checkpoint_layout(str(tmp_path))
