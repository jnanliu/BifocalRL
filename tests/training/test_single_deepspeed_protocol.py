from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainingArguments
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
from transformers.optimization import get_scheduler

from utils.single_model_trainer import SingleModelTrainer, compute_pipeline_update_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeAccelerator:
    is_main_process = False

    def __init__(self) -> None:
        self.backward_calls = 0
        self.barrier_calls = 0

    def backward(self, _loss) -> None:
        self.backward_calls += 1

    def wait_for_everyone(self) -> None:
        self.barrier_calls += 1


class _FakeDeepSpeedEngine:
    def __init__(self, gas: int = 4) -> None:
        self.gas = gas
        self.micro_steps = 0
        self.global_steps = 0
        self.backward_calls = 0
        self.loaded = None
        self.saved = None

    def backward(self, _loss) -> None:
        self.backward_calls += 1

    def is_gradient_accumulation_boundary(self) -> bool:
        return (self.micro_steps + 1) % self.gas == 0

    def gradient_accumulation_steps(self) -> int:
        return self.gas

    def train_micro_batch_size_per_gpu(self) -> int:
        return 2

    def train_batch_size(self) -> int:
        return 16

    def step(self) -> None:
        if self.is_gradient_accumulation_boundary():
            self.global_steps += 1
        self.micro_steps += 1

    def load_checkpoint(self, state_dir, **kwargs):
        self.loaded = (state_dir, kwargs)
        self.global_steps = 4
        return str(Path(state_dir) / kwargs["tag"]), {"completed_optimizer_updates": 4}

    def save_checkpoint(self, state_dir, **kwargs):
        self.saved = (state_dir, kwargs)
        return True


def _fake_trainer(*, deepspeed: bool = True) -> SingleModelTrainer:
    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.is_deepspeed_enabled = deepspeed
    trainer.model = _FakeDeepSpeedEngine()
    trainer.accelerator = _FakeAccelerator()
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
    )
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer._micro_step = 0
    trainer._optimizer_step_count = 0
    return trainer


def test_main_experiment_update_plan() -> None:
    plan = compute_pipeline_update_plan(
        num_samples=1024,
        per_device_batch_size=2,
        world_size=2,
        gradient_accumulation_steps=64,
        total_pipeline_steps=15,
        current_pipeline_step=7,
    )

    assert plan["global_micro_batch_size"] == 4
    assert plan["micro_steps_per_rank"] == 256
    assert plan["updates_per_pipeline_step"] == 4
    assert plan["total_optimizer_updates"] == 60
    assert plan["completed_optimizer_updates"] == 24


@pytest.mark.parametrize("num_samples", [1023, 1028])
def test_update_plan_rejects_incomplete_accumulation(num_samples: int) -> None:
    with pytest.raises(ValueError, match="divisible"):
        compute_pipeline_update_plan(
            num_samples=num_samples,
            per_device_batch_size=2,
            world_size=2,
            gradient_accumulation_steps=64,
            total_pipeline_steps=15,
            current_pipeline_step=1,
        )


def test_deepspeed_backward_bypasses_accelerate_wrapper() -> None:
    trainer = _fake_trainer()

    trainer.backward(object())

    assert trainer.model.backward_calls == 1
    assert trainer.accelerator.backward_calls == 0


def test_deepspeed_step_counts_only_real_boundaries() -> None:
    trainer = _fake_trainer()
    updates = []

    for micro_step in range(1, 9):
        trainer._micro_step = micro_step
        updates.append(trainer.optimizer_step())

    assert updates == [False, False, False, True, False, False, False, True]
    assert trainer.model.micro_steps == 8
    assert trainer.model.global_steps == 2
    assert trainer._optimizer_step_count == 2


def test_deepspeed_training_state_round_trip_contract(tmp_path: Path) -> None:
    trainer = _fake_trainer()

    loaded = trainer.load_training_state(str(tmp_path), "checkpoint-1")
    saved_path = trainer.save_training_state(
        str(tmp_path),
        "checkpoint-2",
        {"completed_optimizer_updates": 8},
    )

    assert loaded["completed_optimizer_updates"] == 4
    assert trainer._optimizer_step_count == 4
    assert trainer.model.loaded[1]["load_optimizer_states"] is True
    assert trainer.model.loaded[1]["load_lr_scheduler_states"] is True
    assert trainer.model.saved[1]["client_state"]["completed_optimizer_updates"] == 8
    assert saved_path == str(tmp_path / "checkpoint-2")


def test_deepspeed_resume_restores_pipeline_optimizer_counter(tmp_path: Path) -> None:
    trainer = _fake_trainer()

    def load_checkpoint(_state_dir, **_kwargs):
        trainer.model.global_steps = 4  # DeepSpeed engine-local counter.
        return str(tmp_path / "checkpoint-15"), {
            "completed_optimizer_updates": 60,
            "scheduler_total_optimizer_updates": 60,
            "scheduler_warmup_optimizer_updates": 6,
        }

    trainer.model.load_checkpoint = load_checkpoint
    trainer._scheduler_backend = "deepspeed_native"
    trainer._scheduler_total_steps = 120
    trainer._scheduler_warmup_steps = 12
    trainer._scheduler_step_count = 0
    trainer._rebase_loaded_native_scheduler = lambda **kwargs: None
    trainer.scheduler_runtime_state = lambda: {
        "backend": "deepspeed_native",
        "scheduler_completed_optimizer_updates": 60,
    }

    trainer.load_training_state(str(tmp_path), "checkpoint-15")

    assert trainer._optimizer_step_count == 60


def test_adamw_and_cosine_scheduler_resume_without_resetting_moments() -> None:
    def build():
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=6,
            num_training_steps=60,
        )
        return parameter, optimizer, scheduler

    parameter, optimizer, scheduler = build()
    for _ in range(4):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    resumed_parameter, resumed_optimizer, resumed_scheduler = build()
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)

    adam_state = next(iter(resumed_optimizer.state.values()))
    assert int(adam_state["step"].item()) == 4
    assert resumed_scheduler.last_epoch == 4
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()

    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    resumed_optimizer.step()
    resumed_scheduler.step()
    assert int(adam_state["step"].item()) == 5
    assert resumed_scheduler.last_epoch == 5


def test_zero3_configs_delegate_clipping_to_training_arguments() -> None:
    for path in sorted((PROJECT_ROOT / "recipes" / "accelerate_configs").glob("zero3*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "gradient_clipping: auto" in text, path
        assert "gradient_clipping: 0.0" not in text, path


def test_transformers_resolves_deepspeed_clipping_to_one(tmp_path: Path) -> None:
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        max_grad_norm=1.0,
    )
    config = HfTrainerDeepSpeedConfig(
        {
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "train_batch_size": "auto",
            "gradient_clipping": "auto",
            "zero_optimization": {"stage": 3},
        }
    )

    config.trainer_config_process(args, auto_find_batch_size=False)

    assert config.config["gradient_clipping"] == pytest.approx(1.0)
