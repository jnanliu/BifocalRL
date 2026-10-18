from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.single_model_trainer import SingleModelTrainer, compute_pipeline_update_plan


def test_pipeline_update_plan_counts_real_optimizer_updates():
    plan = compute_pipeline_update_plan(
        num_samples=1024,
        per_device_batch_size=2,
        world_size=2,
        gradient_accumulation_steps=64,
        total_pipeline_steps=15,
        current_pipeline_step=9,
    )

    assert plan["global_micro_batch_size"] == 4
    assert plan["micro_steps_per_rank"] == 256
    assert plan["updates_per_pipeline_step"] == 4
    assert plan["total_optimizer_updates"] == 60
    assert plan["completed_optimizer_updates"] == 32


def test_native_cosine_scheduler_config_uses_exact_update_horizon():
    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.args = SimpleNamespace(lr_scheduler_type="cosine", learning_rate=1e-6)
    config = {}

    trainer._configure_native_deepspeed_scheduler(
        config=config,
        total_steps=60,
        warmup_steps=6,
        completed_steps=32,
    )

    assert config["scheduler"] == {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": 60,
            "warmup_min_ratio": 0.0,
            "warmup_num_steps": 6,
            "cos_min_ratio": 0.0,
            "warmup_type": "linear",
            "last_batch_iteration": 31,
        },
    }


@pytest.mark.parametrize(
    ("backend", "iteration", "expected"),
    [
        ("deepspeed_native", 3, 4),
        ("legacy", 4, 4),
    ],
)
def test_scheduler_runtime_state_uses_backend_specific_iteration_semantics(
    backend, iteration, expected
):
    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.is_deepspeed_enabled = backend == "deepspeed_native"
    trainer._scheduler_backend = backend
    trainer._scheduler_total_steps = 60
    trainer._scheduler_warmup_steps = 6
    trainer._scheduler_step_count = expected
    trainer._boundary_call_count = 4
    trainer._optimizer_step_count = expected
    trainer._skipped_optimizer_updates = 0
    scheduler = SimpleNamespace(
        last_batch_iteration=iteration if backend == "deepspeed_native" else None,
        last_epoch=iteration if backend == "legacy" else None,
        get_last_lr=lambda: [5e-7],
    )
    trainer.model = SimpleNamespace(lr_scheduler=scheduler)
    trainer.lr_scheduler = scheduler

    state = trainer.scheduler_runtime_state()

    assert state["scheduler_completed_optimizer_updates"] == expected
    assert state["current_learning_rate"] == 5e-7
    assert state["effective_optimizer_updates"] == expected


def test_deepspeed_cosine_scheduler_round_trip_is_nonzero_after_four_updates():
    try:
        from deepspeed.runtime.lr_schedules import WarmupCosineLR
    except Exception as exc:  # pragma: no cover - dependency-specific skip
        pytest.skip(f"DeepSpeed scheduler unavailable: {exc}")

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scheduler = WarmupCosineLR(
        optimizer,
        total_num_steps=60,
        warmup_min_ratio=0.0,
        warmup_num_steps=6,
        cos_min_ratio=0.0,
        warmup_type="linear",
        last_batch_iteration=-1,
    )
    assert optimizer.param_groups[0]["lr"] == 0.0

    for _ in range(4):
        optimizer.step()
        scheduler.step()

    assert scheduler.last_batch_iteration == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-7)

    restored_optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-6)
    restored = WarmupCosineLR(
        restored_optimizer,
        total_num_steps=60,
        warmup_min_ratio=0.0,
        warmup_num_steps=6,
        cos_min_ratio=0.0,
        warmup_type="linear",
        last_batch_iteration=-1,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.last_batch_iteration == 3


def test_native_optimizer_step_uses_engine_boundaries_for_scheduler():
    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.is_deepspeed_enabled = True
    trainer._scheduler_backend = "deepspeed_native"
    trainer._scheduler_step_count = 0
    trainer._optimizer_step_count = 0
    trainer._skipped_optimizer_updates = 0
    trainer._boundary_call_count = 0
    trainer._micro_step = 0
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
    )
    trainer.accelerator = SimpleNamespace(is_main_process=False)

    class _Engine:
        def __init__(self):
            self.micro_steps = 0
            self.global_steps = 0
            self.lr_scheduler = SimpleNamespace(last_batch_iteration=-1)

        def is_gradient_accumulation_boundary(self):
            return (self.micro_steps + 1) % 4 == 0

        def step(self):
            boundary = self.is_gradient_accumulation_boundary()
            if boundary:
                self.global_steps += 1
                self.lr_scheduler.last_batch_iteration += 1
            self.micro_steps += 1

    trainer.model = _Engine()
    updates = []
    for micro_step in range(1, 9):
        trainer._micro_step = micro_step
        updates.append(trainer.optimizer_step())

    assert updates == [False, False, False, True, False, False, False, True]
    assert trainer._optimizer_step_count == 2
    assert trainer._scheduler_step_count == 2
    assert trainer.model.lr_scheduler.last_batch_iteration == 1


def test_native_resume_can_extend_scheduler_horizon_without_resetting_state():
    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.is_deepspeed_enabled = True
    trainer._scheduler_backend = "deepspeed_native"
    trainer._scheduler_total_steps = 120
    trainer._scheduler_warmup_steps = 12
    trainer._scheduler_step_count = 0
    trainer._optimizer_step_count = 0
    trainer._skipped_optimizer_updates = 0
    trainer._boundary_call_count = 0

    class _Scheduler:
        total_num_steps = 120
        warmup_num_steps = 12
        inverse_log_warm_up = 1.0
        last_batch_iteration = -1

        def step(self, last_batch_iteration=None):
            if last_batch_iteration is not None:
                self.last_batch_iteration = last_batch_iteration
            else:
                self.last_batch_iteration += 1

        def get_last_lr(self):
            return [7e-7]

    class _Engine:
        def __init__(self):
            self.global_steps = 60
            self.lr_scheduler = _Scheduler()

        def load_checkpoint(self, state_dir, **kwargs):
            return str(Path(state_dir) / kwargs["tag"]), {
                "completed_optimizer_updates": 60,
                "scheduler_total_optimizer_updates": 60,
                "scheduler_warmup_optimizer_updates": 6,
            }

    trainer.model = _Engine()
    trainer.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)

    loaded = trainer.load_training_state("/tmp/state", "checkpoint-15")

    assert loaded["completed_optimizer_updates"] == 60
    assert trainer.model.lr_scheduler.total_num_steps == 120
    assert trainer.model.lr_scheduler.warmup_num_steps == 6
    assert trainer.model.lr_scheduler.last_batch_iteration == 59
    assert trainer._scheduler_warmup_steps == 6


def test_stateless_native_resume_synchronizes_optimizer_learning_rate():
    try:
        from deepspeed.runtime.lr_schedules import WarmupCosineLR
    except Exception as exc:  # pragma: no cover - dependency-specific skip
        pytest.skip(f"DeepSpeed scheduler unavailable: {exc}")

    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-6)
    scheduler = WarmupCosineLR(
        optimizer,
        total_num_steps=120,
        warmup_min_ratio=0.0,
        warmup_num_steps=12,
        cos_min_ratio=0.0,
        warmup_type="linear",
        last_batch_iteration=59,
    )
    assert optimizer.param_groups[0]["lr"] == 1e-6

    trainer = SingleModelTrainer.__new__(SingleModelTrainer)
    trainer.is_deepspeed_enabled = True
    trainer.model = SimpleNamespace(lr_scheduler=scheduler)
    trainer._scheduler_total_steps = 120
    trainer._scheduler_warmup_steps = 12
    trainer._rebase_loaded_native_scheduler(completed_steps=60, saved_warmup_steps=12)

    assert scheduler.last_batch_iteration == 59
    assert optimizer.param_groups[0]["lr"] == pytest.approx(scheduler.get_last_lr()[0])
    assert 0.0 < optimizer.param_groups[0]["lr"] < 1e-6
