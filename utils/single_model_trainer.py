# coding=utf-8
"""
Single Model Trainer: Train one model at a time.

This trainer is designed for the "two separate runs" approach:
- Run 1: Train Qwen (policy=Qwen, reward=Llama)
- Run 2: Train Llama (policy=Llama, reward=Qwen)

Each trainer instance handles only ONE model, avoiding DeepSpeed+Accelerate multi-model issues.
"""

import os
import logging
import json
import shutil
import math
import tempfile
import torch
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, TrainingArguments, PreTrainedTokenizerBase
from transformers.utils import is_accelerate_available
from packaging import version

from utils.model_utils import (
    has_qwen35_container_state_dict_keys,
    is_qwen35_model_type,
    is_qwen35_text_model_type,
    normalize_qwen35_text_state_dict,
)

logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate import __version__ as accelerate_version

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def compute_pipeline_update_plan(
    *,
    num_samples: int,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    total_pipeline_steps: int,
    current_pipeline_step: int,
) -> Dict[str, int]:
    """Compute exact distributed micro-step and optimizer-update counts."""
    values = {
        "num_samples": int(num_samples),
        "per_device_batch_size": int(per_device_batch_size),
        "world_size": int(world_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "total_pipeline_steps": int(total_pipeline_steps),
        "current_pipeline_step": int(current_pipeline_step),
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if values["current_pipeline_step"] > values["total_pipeline_steps"]:
        raise ValueError(
            "current_pipeline_step must not exceed total_pipeline_steps: "
            f"{values['current_pipeline_step']} > {values['total_pipeline_steps']}"
        )

    global_micro_batch_size = values["per_device_batch_size"] * values["world_size"]
    if values["num_samples"] % global_micro_batch_size != 0:
        raise ValueError(
            "Rollout sample count must be divisible by the distributed micro-batch size; "
            f"got samples={values['num_samples']}, per_device_batch_size={values['per_device_batch_size']}, "
            f"world_size={values['world_size']}, global_micro_batch_size={global_micro_batch_size}."
        )

    micro_steps_per_rank = values["num_samples"] // global_micro_batch_size
    gas = values["gradient_accumulation_steps"]
    if micro_steps_per_rank % gas != 0:
        raise ValueError(
            "Micro-steps per rank must be divisible by gradient_accumulation_steps so every rollout ends on a "
            f"real optimizer boundary; got micro_steps_per_rank={micro_steps_per_rank}, gas={gas}."
        )

    updates_per_pipeline_step = micro_steps_per_rank // gas
    return {
        **values,
        "global_micro_batch_size": global_micro_batch_size,
        "micro_steps_per_rank": micro_steps_per_rank,
        "updates_per_pipeline_step": updates_per_pipeline_step,
        "total_optimizer_updates": values["total_pipeline_steps"] * updates_per_pipeline_step,
        "completed_optimizer_updates": (values["current_pipeline_step"] - 1) * updates_per_pipeline_step,
    }


class SingleModelTrainer:
    """
    Trainer for a single model.
    
    This trainer handles:
    1. One model with one accelerator
    2. One training step (not a full training loop)
    3. Training data is passed in (from cache)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        compute_loss: Optional[Any] = None,
        metadata_source_dir: Optional[str] = None,
    ):
        """
        Initialize SingleModelTrainer.
        
        Args:
            model: Model to train
            args: TrainingArguments
            tokenizer: Tokenizer for the model
            compute_loss: Loss computation function
        """
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.compute_loss = compute_loss
        # Optional: a stable source directory to copy non-weight files from after saving.
        # Use this to prevent tokenizer/config drift across checkpoints.
        self.metadata_source_dir = metadata_source_dir
        # Micro-step counter for debugging accumulation/boundaries.
        self._micro_step = 0
        # Optimizer-step counter: number of actual parameter updates performed.
        # Used by stepwise clip to determine the clip lower bound.
        self._optimizer_step_count = 0
        self._boundary_call_count = 0
        self._skipped_optimizer_updates = 0
        self._scheduler_step_count = 0
        self._scheduler_backend = "uninitialized"
        self._scheduler_total_steps = None
        self._scheduler_warmup_steps = None
        
        # Initialize accelerator
        self._init_accelerator()
        
        # Don't prepare model here - will be prepared together with optimizer/scheduler
        # This is necessary for DeepSpeed which requires model + optimizer + scheduler together
        
        # Optimizer and scheduler will be created when training data is available
        self.optimizer = None
        self.lr_scheduler = None
        
        logger.info("SingleModelTrainer initialized")

        # Optional one-line debug to verify accumulation settings at runtime.
        if self._debug_boundary() and self.accelerator.is_main_process:
            acc_gas = getattr(self.accelerator, "gradient_accumulation_steps", None)
            print(
                f"[debug] accelerator gas={acc_gas} | args gas={getattr(self.args, 'gradient_accumulation_steps', None)}"
                f" | deepspeed={bool(self.is_deepspeed_enabled)}",
                flush=True,
            )
    
    @staticmethod
    def _debug_boundary() -> bool:
        v = os.environ.get("CO_DEBUG_TRAIN_BOUNDARY", "0")
        return str(v).strip().lower() not in ("", "0", "false", "no", "off", "none")

    @staticmethod
    def _debug_ds_steps() -> bool:
        v = os.environ.get("CO_DEBUG_DS_STEPS", "0")
        return str(v).strip().lower() not in ("", "0", "false", "no", "off", "none")

    def _init_accelerator(self):
        """Initialize accelerator for distributed training.
        
        This method follows the same pattern as transformers.Trainer.create_accelerator_and_postprocess():
        - Uses accelerator_config from TrainingArguments
        - Handles gradient_accumulation_kwargs
        - Extracts dataloader_config
        - Creates Accelerator with proper configuration
        
        Reference:
        - transformers/trainer.py: create_accelerator_and_postprocess()
        - old/trainer.py: create_accelerator_and_postprocess()
        """
        # This is a minimal, close copy of `transformers.Trainer.create_accelerator_and_postprocess()`,
        # restricted to accelerator initialization and flags (no Trainer-specific post-processing).
        grad_acc_kwargs = {}
        if (
            is_accelerate_available("0.28.0")
            and getattr(self.args, "accelerator_config", None) is not None
            and self.args.accelerator_config.gradient_accumulation_kwargs is not None
        ):
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                # raise because we do not know which setting is intended.
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        if getattr(self.args, "accelerator_config", None) is not None:
            accelerator_config = self.args.accelerator_config.to_dict()
        else:
            accelerator_config = {}

        dataloader_config = None
        if is_accelerate_available("0.28.0") and accelerator_config:
            dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
            dataloader_config = DataLoaderConfiguration(
                **{param: accelerator_config.pop(param) for param in dataloader_params if param in accelerator_config}
            )
            if is_accelerate_available("1.1.0") and dataloader_config is not None:
                dataloader_config.data_seed = getattr(self.args, "data_seed", None)

        non_blocking = accelerator_config.pop("non_blocking", False)
        if not is_accelerate_available("0.30.0"):
            if non_blocking:
                raise ImportError(
                    "`non_blocking` is only supported in accelerate v0.30.0 and above. Please upgrade accelerate to use this feature."
                )
        else:
            if non_blocking and not self.args.dataloader_pin_memory:
                logger.warning(
                    "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
                )
            if dataloader_config is not None:
                dataloader_config.non_blocking = non_blocking

        # this would have been updated above, no need for it anymore
        accelerator_config.pop("gradient_accumulation_kwargs", None)

        args = {"deepspeed_plugin": getattr(self.args, "deepspeed_plugin", None)}
        if is_accelerate_available("0.28.0") and dataloader_config is not None:
            args["dataloader_config"] = dataloader_config
        else:
            args.update(accelerator_config)
        
        # tp is initialized at Accelerator init phase so
        # args should be prepared here
        if hasattr(self.model, "tp_size") and self.model.tp_size is not None and self.model.tp_size > 1:
            self.is_tp_enabled = True
            if version.parse(accelerate_version) > version.parse("1.3.0"):
                from accelerate.utils import TorchTensorParallelPlugin
                args["torch_tp_plugin"] = TorchTensorParallelPlugin(tp_size=self.model.tp_size)
            else:
                raise ValueError("Requires accelerate>1.3.0 to use Tensor Parallelism.")
        else:
            self.is_tp_enabled = False

        # create accelerator object
        # IMPORTANT: make Accelerator aware of gradient accumulation.
        # Without this, Accelerate/DeepSpeed may use a default or inferred value that can differ
        # from TrainingArguments.gradient_accumulation_steps, causing optimizer updates to happen
        # at unexpected micro-steps (e.g., 224 vs 448).
        args["gradient_accumulation_steps"] = int(getattr(self.args, "gradient_accumulation_steps", 1))
        self.accelerator = Accelerator(**args)
        
        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None

        # Mirror Trainer's behavior: propagate TrainingArguments into deepspeed plugin if needed.
        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self._propagate_args_to_deepspeed()

        # Optional: dump deepspeed config knobs relevant to accumulation/micro-batch.
        if self._debug_boundary() and self.accelerator.is_main_process and self.is_deepspeed_enabled:
            ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
            ds_cfg = None
            try:
                if ds_plugin is not None and hasattr(ds_plugin, "hf_ds_config") and ds_plugin.hf_ds_config is not None:
                    ds_cfg = getattr(ds_plugin.hf_ds_config, "config", None)
            except Exception:
                ds_cfg = None
            if isinstance(ds_cfg, dict):
                keys = ["train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps", "zero_optimization", "optimizer"]
                slim = {k: ds_cfg.get(k) for k in keys if k in ds_cfg}
                print(f"[debug] deepspeed_config (slim)={json.dumps(slim, ensure_ascii=False)}", flush=True)

    def _propagate_args_to_deepspeed(self):
        """Propagate TrainingArguments into the accelerate DeepSpeed plugin config.

        This mirrors the minimal portion of `transformers.Trainer.propagate_args_to_deepspeed()` needed for
        accelerate+deepspeed setups where `args.deepspeed` may be None but a plugin is active.
        """
        try:
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
        except Exception:
            return

        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is None:
            return
        hf_ds_config = getattr(ds_plugin, "hf_ds_config", None)
        if hf_ds_config is None or not hasattr(hf_ds_config, "config"):
            return

        ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(hf_ds_config.config)
        ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
        # Keep auto_find_batch_size=False (we don't use it here).
        ds_plugin.hf_ds_config.trainer_config_process(self.args, False)
        configured_clip = ds_plugin.deepspeed_config.get("gradient_clipping")
        expected_clip = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
        try:
            actual_clip = float(configured_clip)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"DeepSpeed gradient_clipping was not resolved to a number: {configured_clip!r}"
            ) from exc
        if not math.isclose(actual_clip, expected_clip, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "DeepSpeed gradient clipping does not match TrainingArguments.max_grad_norm: "
                f"deepspeed={actual_clip}, training_args={expected_clip}"
            )

    def _get_deepspeed_config(self) -> Optional[Dict[str, Any]]:
        """Return the live DeepSpeed config dictionary used by Accelerate."""
        if not self.is_deepspeed_enabled:
            return None
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            return None
        hf_config = getattr(plugin, "hf_ds_config", None)
        config = getattr(hf_config, "config", None) if hf_config is not None else None
        if isinstance(config, dict):
            return config
        config = getattr(plugin, "deepspeed_config", None)
        if isinstance(config, dict):
            return config
        return None

    @staticmethod
    def _scheduler_name(value: Any) -> str:
        name = getattr(value, "value", value)
        return str(name).strip().lower()

    def _configure_native_deepspeed_scheduler(
        self,
        *,
        config: Dict[str, Any],
        total_steps: int,
        warmup_steps: int,
        completed_steps: int,
    ) -> None:
        """Add an exact-update scheduler to the live DeepSpeed config.

        The scheduler is deliberately configured at runtime because the total
        number of optimizer updates depends on rollout size and distributed
        batch geometry. Keeping it out of the static accelerate YAML prevents
        stale 15-step values from being reused for a 60-update run.
        """
        scheduler_name = self._scheduler_name(getattr(self.args, "lr_scheduler_type", "cosine"))
        last_batch_iteration = max(-1, int(completed_steps) - 1)
        if scheduler_name == "cosine":
            scheduler = {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": int(total_steps),
                    "warmup_min_ratio": 0.0,
                    "warmup_num_steps": int(warmup_steps),
                    "cos_min_ratio": 0.0,
                    "warmup_type": "linear",
                    "last_batch_iteration": last_batch_iteration,
                },
            }
        elif scheduler_name == "linear":
            scheduler = {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": int(total_steps),
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": float(self.args.learning_rate),
                    "warmup_num_steps": int(warmup_steps),
                    "warmup_type": "linear",
                    "last_batch_iteration": last_batch_iteration,
                },
            }
        elif scheduler_name in {"constant", "constant_with_warmup"}:
            scheduler = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": float(self.args.learning_rate),
                    "warmup_num_steps": int(warmup_steps),
                    "warmup_type": "linear",
                    "last_batch_iteration": last_batch_iteration,
                },
            }
        else:
            raise ValueError(
                "DeepSpeed native scheduler does not support lr_scheduler_type="
                f"{scheduler_name!r}; supported values are cosine, linear, constant, "
                "and constant_with_warmup."
            )
        config["scheduler"] = scheduler

    def _get_engine_scheduler(self) -> Any:
        if not self.is_deepspeed_enabled:
            return None
        scheduler = getattr(self.model, "lr_scheduler", None)
        if scheduler is None:
            raise RuntimeError(
                "DeepSpeed native scheduler was requested, but the engine has no lr_scheduler. "
                "Do not pass an external Hugging Face scheduler to this DeepSpeed path."
            )
        return scheduler

    def _rebase_loaded_native_scheduler(
        self,
        *,
        completed_steps: int,
        saved_warmup_steps: Optional[int],
    ) -> None:
        """Recompute the loaded native scheduler at the restored update position.

        DeepSpeed scheduler state only stores ``last_batch_iteration``.  The
        optimizer checkpoint restores the old learning rate, while changing
        ``max_steps`` creates a scheduler with a new horizon.  Reapplying the
        restored position makes the scheduler and optimizer agree again and
        lets a run extend its horizon without discarding AdamW state.
        """
        scheduler = self._get_engine_scheduler()
        if saved_warmup_steps is not None and hasattr(scheduler, "warmup_num_steps"):
            warmup_steps = max(2, int(saved_warmup_steps))
            scheduler.warmup_num_steps = warmup_steps
            if hasattr(scheduler, "inverse_log_warm_up"):
                scheduler.inverse_log_warm_up = 1.0 / math.log(warmup_steps)
            self._scheduler_warmup_steps = int(saved_warmup_steps)

        if hasattr(scheduler, "total_num_steps") and self._scheduler_total_steps is not None:
            scheduler.total_num_steps = int(self._scheduler_total_steps)

        last_batch_iteration = max(-1, int(completed_steps) - 1)
        scheduler.step(last_batch_iteration=last_batch_iteration)

    def scheduler_runtime_state(self) -> Dict[str, Any]:
        """Return small, serializable scheduler/update diagnostics."""
        scheduler = getattr(self.model, "lr_scheduler", None) if self.is_deepspeed_enabled else self.lr_scheduler
        current_lr = None
        scheduler_iteration = None
        if scheduler is not None:
            try:
                current_lr = float(scheduler.get_last_lr()[0])
            except Exception:
                pass
            raw_scheduler = getattr(scheduler, "scheduler", scheduler)
            last_iteration = getattr(raw_scheduler, "last_batch_iteration", None)
            if last_iteration is None:
                last_iteration = getattr(raw_scheduler, "last_epoch", None)
            if last_iteration is not None:
                scheduler_iteration = int(last_iteration)
        if current_lr is None:
            try:
                current_lr = float(self.model.optimizer.param_groups[0]["lr"])
            except Exception:
                try:
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                except Exception:
                    pass
        if scheduler_iteration is None:
            scheduler_completed_updates = None
        elif self._scheduler_backend == "deepspeed_native":
            # DeepSpeed starts at last_batch_iteration=-1 and increments
            # after each effective optimizer update.
            scheduler_completed_updates = max(0, scheduler_iteration + 1)
        else:
            # torch/Transformers schedulers initialize last_epoch=0 during
            # construction, so their value already equals completed steps.
            scheduler_completed_updates = max(0, scheduler_iteration)
        return {
            "backend": str(self._scheduler_backend),
            "total_optimizer_updates": self._scheduler_total_steps,
            "warmup_optimizer_updates": self._scheduler_warmup_steps,
            "scheduler_iteration": scheduler_iteration,
            "scheduler_completed_optimizer_updates": scheduler_completed_updates,
            "scheduler_step_count": int(self._scheduler_step_count),
            "current_learning_rate": current_lr,
            "boundary_calls": int(self._boundary_call_count),
            "effective_optimizer_updates": int(self._optimizer_step_count),
            "skipped_optimizer_updates": int(self._skipped_optimizer_updates),
        }
    
    def create_optimizer_and_scheduler(
        self,
        num_training_steps: int,
        train_dataloader=None,
        *,
        scheduler_total_steps: Optional[int] = None,
        scheduler_completed_steps: Optional[int] = None,
        scheduler_backend: str = "legacy",
    ):
        """Create optimizer and scheduler.

        If DeepSpeed is enabled, it's best to call `accelerator.prepare()` with a real dataloader so that accelerate
        can infer `train_micro_batch_size_per_gpu` (or validate the DeepSpeed config). This matches the GRPO/Intuitor
        flow more closely than using a dummy dataloader.
        """
        from accelerate.utils import DummyOptim, DummyScheduler

        scheduler_backend = str(scheduler_backend).strip().lower()
        if scheduler_backend not in {"legacy", "deepspeed_native"}:
            raise ValueError(f"Unknown scheduler_backend={scheduler_backend!r}")
        if scheduler_backend == "deepspeed_native" and not self.is_deepspeed_enabled:
            raise ValueError("scheduler_backend='deepspeed_native' requires DeepSpeed")
        self._scheduler_backend = scheduler_backend
        
        model_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        # Get DeepSpeed config if enabled
        # When using accelerate config file, args.deepspeed may be None,
        # but deepspeed config is available via accelerator.state.deepspeed_plugin
        ds_config = self._get_deepspeed_config()
        if self.is_deepspeed_enabled and ds_config is None:
            raise RuntimeError("DeepSpeed is enabled but its live config could not be inspected")
        
        # Create optimizer
        # Check if ds_config is a dict and contains optimizer key
        has_optimizer_in_config = (
            ds_config is not None 
            and isinstance(ds_config, dict) 
            and "optimizer" in ds_config
        )
        if has_optimizer_in_config:
            optimizer = DummyOptim(params=model_parameters)
        else:
            from torch.optim import AdamW
            # Torch AdamW defaults to foreach/multi-tensor when available, which can create large
            # temporary buffers during optimizer step (often triggering OOM under high memory pressure,
            # especially with ZeRO-3).
            #
            # Default behavior here:
            # - If CO_ADAMW_FOREACH is explicitly set, respect it (0/1/true/false).
            # - Otherwise, disable foreach when DeepSpeed is enabled to reduce peak memory.
            foreach_env = os.environ.get("CO_ADAMW_FOREACH", None)
            if foreach_env is None:
                foreach = False if self.is_deepspeed_enabled else True
            else:
                foreach = str(foreach_env).strip().lower() not in ("", "0", "false", "no", "off", "none")
            optimizer = AdamW(
                [{"params": model_parameters, "weight_decay": self.args.weight_decay}],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                foreach=bool(foreach),
            )
        
        if scheduler_total_steps is None:
            # Preserve legacy behavior for shared callers that have not yet
            # adopted exact optimizer-update accounting (for example co-training).
            total_pipeline_steps = int(getattr(self.args, "total_pipeline_steps", 0) or 0)
            current_pipeline_step = int(getattr(self.args, "current_pipeline_step", 0) or 0)
            effective_num_steps = total_pipeline_steps or int(num_training_steps)
            completed_scheduler_steps = max(0, current_pipeline_step - 1) if total_pipeline_steps else 0
        else:
            effective_num_steps = int(scheduler_total_steps)
            completed_scheduler_steps = int(scheduler_completed_steps or 0)
        if effective_num_steps <= 0:
            raise ValueError(f"scheduler_total_steps must be positive, got {effective_num_steps}")
        if completed_scheduler_steps < 0 or completed_scheduler_steps > effective_num_steps:
            raise ValueError(
                "scheduler_completed_steps must be in [0, scheduler_total_steps], got "
                f"{completed_scheduler_steps} and {effective_num_steps}"
            )

        # Compute warmup: warmup_ratio takes priority when warmup_steps == 0.
        warmup_steps = int(getattr(self.args, "warmup_steps", 0) or 0)
        if warmup_steps == 0:
            warmup_ratio = float(getattr(self.args, "warmup_ratio", 0.0) or 0.0)
            if warmup_ratio > 0:
                warmup_steps = math.ceil(effective_num_steps * warmup_ratio)

        self._scheduler_total_steps = int(effective_num_steps)
        self._scheduler_warmup_steps = int(warmup_steps)
        self._scheduler_step_count = 0

        if scheduler_backend == "deepspeed_native":
            assert ds_config is not None
            self._configure_native_deepspeed_scheduler(
                config=ds_config,
                total_steps=effective_num_steps,
                warmup_steps=warmup_steps,
                completed_steps=completed_scheduler_steps,
            )
            lr_scheduler = None
        else:
            has_scheduler_in_config = (
                ds_config is not None
                and isinstance(ds_config, dict)
                and "scheduler" in ds_config
            )
            if has_scheduler_in_config:
                lr_scheduler = DummyScheduler(optimizer)
            else:
                from transformers.optimization import get_scheduler
                lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=effective_num_steps,
                )

            # Legacy/stateless callers still reconstruct their external scheduler.
            if completed_scheduler_steps > 0:
                if self.accelerator.is_main_process:
                    logger.info(
                        f"Legacy scheduler: advancing {completed_scheduler_steps} optimizer updates "
                        f"(warmup={warmup_steps}, scheduler_total={effective_num_steps})"
                    )
                for _ in range(completed_scheduler_steps):
                    lr_scheduler.step()
        
        # Prepare with accelerator
        if self.is_deepspeed_enabled:
            if train_dataloader is None:
                raise ValueError(
                    "DeepSpeed is enabled but no train_dataloader was provided. "
                    "Pass a real dataloader to create_optimizer_and_scheduler(..., train_dataloader=...) so "
                    "accelerate/deepspeed can validate or infer micro-batch size."
                )
            # Debug: log how many micro-batches each rank should see.
            if self._debug_boundary() and self.accelerator.is_main_process:
                try:
                    dataset_len = None
                    try:
                        dataset_len = len(getattr(train_dataloader, "dataset", []))
                    except Exception:
                        dataset_len = None
                    bs = getattr(train_dataloader, "batch_size", None)
                    num_proc = int(getattr(self.accelerator, "num_processes", 1) or 1)
                    local_batches = None
                    per_rank_batches_est = None
                    if isinstance(dataset_len, int) and isinstance(bs, int) and bs > 0:
                        local_batches = int(math.ceil(dataset_len / bs))
                        per_rank_batches_est = int(math.ceil(local_batches / num_proc))
                    print(
                        "[debug] dataloader(pre-prepare): "
                        f"dataset_len={dataset_len} batch_size={bs} local_batches={local_batches} per_rank_batches_est={per_rank_batches_est}"
                        f" | accel(num_proc={num_proc})"
                        f" | args(per_device_bs={getattr(self.args, 'per_device_train_batch_size', None)}"
                        f" gas={getattr(self.args, 'gradient_accumulation_steps', None)})",
                        flush=True,
                    )
                except Exception:
                    pass
            if scheduler_backend == "deepspeed_native":
                # The scheduler lives inside DeepSpeed. Passing a Hugging Face
                # LambdaLR here would make Accelerate wrap it externally, while
                # engine.step() would never advance it.
                self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
                    self.model, optimizer, train_dataloader
                )
                self.lr_scheduler = self._get_engine_scheduler()
                if completed_scheduler_steps > 0:
                    self._rebase_loaded_native_scheduler(
                        completed_steps=completed_scheduler_steps,
                        saved_warmup_steps=warmup_steps,
                    )
                self._scheduler_step_count = int(completed_scheduler_steps)
                self._optimizer_step_count = int(completed_scheduler_steps)
            else:
                self.model, self.optimizer, self.lr_scheduler, train_dataloader = self.accelerator.prepare(
                    self.model, optimizer, lr_scheduler, train_dataloader
                )
            self.train_dataloader = train_dataloader

            if self._debug_boundary() and self.accelerator.is_main_process:
                try:
                    post_len = None
                    try:
                        post_len = len(self.train_dataloader)
                    except Exception:
                        post_len = None
                    print(
                        "[debug] dataloader(post-prepare): "
                        f"reported_len={post_len} | accel(process_index={getattr(self.accelerator, 'process_index', None)}/{getattr(self.accelerator, 'num_processes', None)})",
                        flush=True,
                    )
                except Exception:
                    pass
        else:
            # For non-DeepSpeed, prepare model + optimizer + scheduler + dataloader together
            # so that Accelerate handles device placement and distributed sharding uniformly.
            if not hasattr(self.model, '_hf_accelerator_prepared'):
                if train_dataloader is not None:
                    self.model, optimizer, lr_scheduler, train_dataloader = self.accelerator.prepare(
                        self.model, optimizer, lr_scheduler, train_dataloader
                    )
                    self.train_dataloader = train_dataloader
                else:
                    self.model, optimizer, lr_scheduler = self.accelerator.prepare(
                        self.model, optimizer, lr_scheduler
                    )
            else:
                if train_dataloader is not None:
                    optimizer, lr_scheduler, train_dataloader = self.accelerator.prepare(
                        optimizer, lr_scheduler, train_dataloader
                    )
                    self.train_dataloader = train_dataloader
                else:
                    optimizer, lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        
        return self.optimizer, self.lr_scheduler
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor values in *inputs* to ``self.accelerator.device``.

        Mirrors ``transformers.Trainer._prepare_inputs`` so that loss functions
        can assume every tensor is already on the correct device without
        scattered ``.to(device)`` calls.
        """
        device = self.accelerator.device
        prepared: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(device)
            else:
                prepared[k] = v
        return prepared

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward without depending on Accelerate's DeepSpeed wrapper semantics."""
        if self.is_deepspeed_enabled:
            self.model.backward(loss)
        else:
            self.accelerator.backward(loss)

    def training_step(
        self,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            inputs: Training data dict (prompt_ids, completion_ids, advantages, etc.)
            num_items_in_batch: Number of items in batch (for loss scaling)
        
        Returns:
            Loss tensor
        """
        self._micro_step += 1
        # DeepSpeed step-counter debug: track how DS micro/global steps evolve across backward/step.
        if self.is_deepspeed_enabled and self._debug_ds_steps() and self.accelerator.is_main_process:
            try:
                ms = getattr(self.model, "micro_steps", None)
                gs = getattr(self.model, "global_steps", None)
                print(
                    f"[ds_steps] pre-backward trainer_micro={int(self._micro_step)} ds_micro_steps={ms} ds_global_steps={gs}",
                    flush=True,
                )
            except Exception:
                pass
        # NOTE:
        # Do NOT call `torch.cuda.empty_cache()` every micro-step by default.
        # It forces allocator cache flushes (DeepSpeed even warns about it) and can *increase*
        # fragmentation/peak memory during ZeRO-3 param gathers and optimizer step.
        #
        # If you really want this behavior for debugging, enable it explicitly:
        #   export CO_EMPTY_CACHE_EVERY_STEP=1
        if torch.cuda.is_available():
            v = os.environ.get("CO_EMPTY_CACHE_EVERY_STEP", "0").strip().lower()
            if v not in ("", "0", "false", "no", "off", "none"):
                torch.cuda.empty_cache()
        
        self.model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss
        if self.compute_loss is not None:
            loss = self.compute_loss(self.model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            # Default loss computation
            outputs = self.model(**inputs)
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                raise ValueError("Model output must contain 'loss' key or attribute")
        
        # Normalize loss for gradient accumulation.
        #
        # DeepSpeed's engine.backward() already divides by gradient_accumulation_steps
        # (via _scale_loss_by_gas with scale_wrt_gas=True by default).
        # Dividing here as well would double-scale the loss, making the effective
        # learning rate GAS times smaller than configured.
        # Only scale manually for the non-DeepSpeed path (where Accelerate's
        # accumulate() context does NOT scale the loss).
        if not self.is_deepspeed_enabled and getattr(self.args, "gradient_accumulation_steps", 1) > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Keep DeepSpeed's own GAS counter authoritative across Accelerate
        # versions. Accelerate 1.14 otherwise overrides the boundary from
        # sync_gradients, which is never advanced by this custom DS loop.
        self.backward(loss)

        if self.is_deepspeed_enabled and self._debug_ds_steps() and self.accelerator.is_main_process:
            try:
                ms = getattr(self.model, "micro_steps", None)
                gs = getattr(self.model, "global_steps", None)
                ib = None
                try:
                    if hasattr(self.model, "is_gradient_accumulation_boundary") and callable(
                        getattr(self.model, "is_gradient_accumulation_boundary")
                    ):
                        ib = bool(self.model.is_gradient_accumulation_boundary())
                except Exception:
                    ib = None
                print(
                    f"[ds_steps] post-backward trainer_micro={int(self._micro_step)} ds_micro_steps={ms} ds_global_steps={gs} is_boundary={ib}",
                    flush=True,
                )
            except Exception:
                pass
        
        return loss.detach()
    
    def optimizer_step(self) -> bool:
        """Advance one micro-step and report whether parameters were updated."""
        if self.is_deepspeed_enabled:
            scheduler_backend = str(getattr(self, "_scheduler_backend", "legacy"))
            is_boundary = bool(self.model.is_gradient_accumulation_boundary())
            if is_boundary:
                self._boundary_call_count = int(getattr(self, "_boundary_call_count", 0)) + 1
            if scheduler_backend == "deepspeed_native":
                self._get_engine_scheduler()
            if self._debug_boundary() and self.accelerator.is_main_process and is_boundary:
                ds_gas = int(self.model.gradient_accumulation_steps())
                ds_mbs = int(self.model.train_micro_batch_size_per_gpu())
                ds_tbs = int(self.model.train_batch_size())
                print(
                    "[boundary] "
                    f"kind=deepspeed micro_step={int(self._micro_step)}"
                    f" | args(per_device_bs={getattr(self.args, 'per_device_train_batch_size', None)} "
                    f"gas={getattr(self.args, 'gradient_accumulation_steps', None)})"
                    f" | ds(mbs={ds_mbs} tbs={ds_tbs} gas={ds_gas} "
                    f"micro_steps={getattr(self.model, 'micro_steps', None)})",
                    flush=True,
                )

            if (
                is_boundary
                and torch.cuda.is_available()
                and str(os.environ.get("CO_EMPTY_CACHE_ON_BOUNDARY", "0")).strip().lower()
                not in ("", "0", "false", "no", "off", "none")
            ):
                self.accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
                self.accelerator.wait_for_everyone()

            before_global_steps = int(getattr(self.model, "global_steps", 0) or 0)
            before_skipped_steps = int(getattr(self.model, "skipped_steps", 0) or 0)
            before_scheduler_iteration = None
            if scheduler_backend == "deepspeed_native":
                raw_scheduler = getattr(self.model, "lr_scheduler", None)
                before_scheduler_iteration = getattr(raw_scheduler, "last_batch_iteration", None)
            # DeepSpeed performs native ZeRO-aware clipping, scheduler stepping,
            # optimizer stepping, and zero_grad only at a real GAS boundary.
            self.model.step()
            after_global_steps = int(getattr(self.model, "global_steps", 0) or 0)
            after_skipped_steps = int(getattr(self.model, "skipped_steps", 0) or 0)
            skipped_delta = max(0, after_skipped_steps - before_skipped_steps)
            updates_applied = max(0, after_global_steps - before_global_steps - skipped_delta)
            self._skipped_optimizer_updates = int(getattr(self, "_skipped_optimizer_updates", 0)) + skipped_delta
            if scheduler_backend == "deepspeed_native" and updates_applied > 0:
                raw_scheduler = getattr(self.model, "lr_scheduler", None)
                after_scheduler_iteration = getattr(raw_scheduler, "last_batch_iteration", None)
                if (
                    before_scheduler_iteration is not None
                    and after_scheduler_iteration is not None
                    and int(after_scheduler_iteration) != int(before_scheduler_iteration) + 1
                ):
                    raise RuntimeError(
                        "DeepSpeed scheduler did not advance exactly once with the optimizer update: "
                        f"before={before_scheduler_iteration}, after={after_scheduler_iteration}"
                    )
                self._scheduler_step_count = int(getattr(self, "_scheduler_step_count", 0)) + 1
            self._optimizer_step_count = int(getattr(self, "_optimizer_step_count", 0)) + updates_applied

            if self._debug_ds_steps() and self.accelerator.is_main_process:
                print(
                    f"[ds_steps] post-step trainer_micro={int(self._micro_step)} "
                    f"ds_micro_steps={getattr(self.model, 'micro_steps', None)} "
                    f"ds_global_steps={after_global_steps} updated={bool(updates_applied)}",
                    flush=True,
                )
            return updates_applied > 0

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self._scheduler_step_count += 1
        self.optimizer.zero_grad()
        self._optimizer_step_count += 1
        return True

    def load_training_state(self, state_dir: str, tag: str) -> Dict[str, Any]:
        """Restore model, AdamW, scheduler, and DeepSpeed counters on all ranks."""
        if not self.is_deepspeed_enabled:
            raise RuntimeError("Persistent training state currently requires DeepSpeed.")
        self.accelerator.wait_for_everyone()
        load_path, client_state = self.model.load_checkpoint(
            state_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise FileNotFoundError(f"Unable to load DeepSpeed training state: {state_dir}/{tag}")
        self.accelerator.wait_for_everyone()
        self._optimizer_step_count = int(getattr(self.model, "global_steps", 0) or 0)
        if str(getattr(self, "_scheduler_backend", "legacy")) == "deepspeed_native":
            saved_updates = None
            saved_scheduler_total = None
            saved_scheduler_warmup = None
            if isinstance(client_state, dict):
                saved_updates = client_state.get("completed_optimizer_updates")
                saved_scheduler_total = client_state.get("scheduler_total_optimizer_updates")
                saved_scheduler_warmup = client_state.get("scheduler_warmup_optimizer_updates")
            if saved_scheduler_total is None:
                raise RuntimeError(
                    "DeepSpeed training state does not contain scheduler_total_optimizer_updates; "
                    "it was not written by the resumable single-model protocol."
                )
            saved_scheduler_total = int(saved_scheduler_total)
            if self._scheduler_total_steps is None or int(self._scheduler_total_steps) < saved_scheduler_total:
                raise ValueError(
                    "The resumed run cannot shorten its optimizer-update horizon: "
                    f"saved={saved_scheduler_total}, current={self._scheduler_total_steps}"
                )
            if saved_updates is not None:
                # DeepSpeed's engine.global_steps is local to the loaded engine
                # state and is not the pipeline-wide optimizer-update counter.
                # The resumable protocol stores the authoritative cumulative
                # count in client_state; restore it before reporting runtime
                # scheduler state or applying the next update.
                saved_updates = int(saved_updates)
                self._optimizer_step_count = saved_updates
                self._rebase_loaded_native_scheduler(
                    completed_steps=saved_updates,
                    saved_warmup_steps=(
                        int(saved_scheduler_warmup) if saved_scheduler_warmup is not None else None
                    ),
                )
                scheduler_state = self.scheduler_runtime_state()
                if scheduler_state["backend"] != "deepspeed_native":
                    raise RuntimeError("Loaded DeepSpeed state with an unexpected scheduler backend")
                scheduler_updates = scheduler_state.get("scheduler_completed_optimizer_updates")
                if scheduler_updates is None or int(scheduler_updates) != saved_updates:
                    raise RuntimeError(
                        "Restored DeepSpeed scheduler position does not match optimizer updates: "
                        f"scheduler={scheduler_updates}, saved_updates={saved_updates}"
                    )
                self._scheduler_step_count = int(saved_updates)
            if self._scheduler_total_steps > saved_scheduler_total:
                logger.warning(
                    "Extending the scheduler horizon while resuming DeepSpeed state: saved=%d, current=%d. "
                    "AdamW and completed optimizer updates are preserved; future LR decay follows the new horizon.",
                    saved_scheduler_total,
                    self._scheduler_total_steps,
                )
        logger.info("Loaded DeepSpeed training state from %s", load_path)
        return client_state if isinstance(client_state, dict) else {}

    def save_training_state(self, state_dir: str, tag: str, client_state: Dict[str, Any]) -> str:
        """Save a resumable ZeRO checkpoint; every distributed rank participates."""
        if not self.is_deepspeed_enabled:
            raise RuntimeError("Persistent training state currently requires DeepSpeed.")
        os.makedirs(state_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        ok = self.model.save_checkpoint(state_dir, tag=tag, client_state=client_state)
        self.accelerator.wait_for_everyone()
        if ok is False:
            raise RuntimeError(f"DeepSpeed failed to save training state: {state_dir}/{tag}")
        return os.path.join(state_dir, tag)
    
    def save_model(self, output_dir: str):
        """
        Save the model.
        
        This implementation matches transformers.Trainer.save_model() for DeepSpeed ZeRO-3 compatibility.
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.accelerator.is_main_process:
            logger.info(f"Saving model checkpoint to {output_dir}")

        if self.is_deepspeed_enabled:
            # IMPORTANT:
            # - `accelerator.get_state_dict(self.model)` can use collectives (e.g., all_gather) under ZeRO-3.
            # - If only rank0 calls it while other ranks return early, NCCL will hang and eventually timeout.
            #
            # Therefore, ALL ranks must participate in the gather. Only the main process writes files.
            self.accelerator.wait_for_everyone()
            try:
                if self.accelerator.is_main_process:
                    logger.info("Gathering ZeRO-3 weights for HF checkpoint save...")
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.accelerator.is_main_process:
                    logger.info("Weights gathered. Writing HF checkpoint to disk...")
                if self.accelerator.is_main_process:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError as e:
                # If ValueError is raised, it means stage3_gather_16bit_weights_on_model_save=false.
                # In that case, use DeepSpeed's sharded checkpoint saving (must be called on all ranks).
                if self.accelerator.is_main_process:
                    logger.warning(
                        "stage3_gather_16bit_weights_on_model_save=false. Saving a full DeepSpeed checkpoint instead; "
                        "use zero_to_fp32.py to recover weights. "
                        f"Error: {e}"
                    )

                # Remove any partially written files from rank0-only paths (if any).
                if self.accelerator.is_main_process:
                    from transformers.modeling_utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME

                    weights_file = os.path.join(output_dir, WEIGHTS_NAME)
                    safe_weights_file = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
                    if os.path.exists(weights_file):
                        os.remove(weights_file)
                    if os.path.exists(safe_weights_file):
                        os.remove(safe_weights_file)

                if hasattr(self.model, "save_checkpoint"):
                    self.model.save_checkpoint(output_dir)
                else:
                    if self.accelerator.is_main_process:
                        logger.error("Model does not have save_checkpoint method")
            finally:
                self.accelerator.wait_for_everyone()
        else:
            # Standard save (non-DeepSpeed): only main process writes.
            if not self.accelerator.is_main_process:
                return
            self._save(output_dir)
    
    def _save(self, output_dir: str, state_dict=None):
        """
        Internal save method matching transformers.Trainer._save().
        
        Args:
            output_dir: Output directory
            state_dict: Optional state dict (for DeepSpeed)
        """
        # NOTE: `TRAINING_ARGS_NAME` moved across transformers versions.
        # - Older versions exposed it from `transformers.modeling_utils`
        # - Newer versions may not; fall back to the canonical filename.
        from transformers.modeling_utils import PreTrainedModel, WEIGHTS_NAME, SAFE_WEIGHTS_NAME
        try:
            from transformers.trainer import TRAINING_ARGS_NAME  # type: ignore
        except Exception:
            TRAINING_ARGS_NAME = "training_args.bin"
        from transformers.utils import is_peft_available

        # During DeepSpeed training ``self.model`` is a DeepSpeedEngine, which
        # does not expose the wrapped Hugging Face config.  Detect model-specific
        # save behavior from the unwrapped model so the distributed and direct
        # save paths produce the same checkpoint layout.
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model_type = getattr(getattr(unwrapped_model, "config", None), "model_type", None)

        # The Qwen3.5 source model is a multimodal container
        # (``model_type=qwen3_5``), while AutoModelForCausalLM exposes its text
        # backbone for training.  Do not rely on the config type alone: the
        # actual state-dict namespace is the authoritative signal here.
        qwen35_checkpoint = is_qwen35_model_type(model_type)
        if state_dict is None and qwen35_checkpoint:
            state_dict = self.model.state_dict()
        qwen35_checkpoint = qwen35_checkpoint or has_qwen35_container_state_dict_keys(state_dict)
        if qwen35_checkpoint:
            if state_dict is None:
                state_dict = self.model.state_dict()
            state_dict = normalize_qwen35_text_state_dict(state_dict)
            if has_qwen35_container_state_dict_keys(state_dict):
                raise RuntimeError(
                    "Qwen3.5 checkpoint normalization left model.language_model.* keys behind"
                )
            if not any(key.startswith("model.") for key in state_dict):
                raise RuntimeError(
                    "Qwen3.5 checkpoint normalization produced no text model.* weights"
                )
        
        # Determine supported classes
        if is_peft_available():
            from peft import PeftModel
            supported_classes = (PreTrainedModel, PeftModel)
        else:
            supported_classes = (PreTrainedModel,)
        
        # Save model
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrapped_model, supported_classes):
                unwrapped_model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=getattr(self.args, 'save_safetensors', False),
                    **({"save_original_format": False} if qwen35_checkpoint else {}),
                )
            else:
                logger.info("Model is not a `PreTrainedModel`, only saving its state dict.")
                if getattr(self.args, 'save_safetensors', False):
                    try:
                        import safetensors.torch
                        safetensors.torch.save_file(
                            state_dict,
                            os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                            metadata={"format": "pt"}
                        )
                    except ImportError:
                        logger.warning("safetensors not available, falling back to torch.save")
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=getattr(self.args, 'save_safetensors', False),
                **({"save_original_format": False} if qwen35_checkpoint else {}),
            )

        if qwen35_checkpoint:
            self._ensure_qwen35_text_config(output_dir)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Preserve *non-weight* files from the source model directory.
        #
        # Motivation:
        # - `save_pretrained()` re-serializes `config.json` and tokenizer files from in-memory objects.
        # - In practice, this can drop fields like `tokenizer_config.json:chat_template` and cause a drift in
        #   `config.json` / other metadata files across steps.
        # - For this project, we want checkpoints to differ primarily by weights; everything else should be
        #   copied from the source model folder we loaded from.
        #
        # We do this *after* saving weights/tokenizer/training_args, then overwrite metadata files from source,
        # while carefully avoiding copying any weight shards/indexes or training artifacts.
        self._maybe_restore_non_weight_files_from_source(output_dir, training_args_name=str(TRAINING_ARGS_NAME))

        # Some models (e.g. Phi-3) rely on `trust_remote_code=True` and require custom python files
        # like `modeling_phi3.py` to be present *inside* the checkpoint folder. When the base model
        # comes from HF Hub, those files live in the transformers dynamic module cache and are NOT
        # automatically included by `save_pretrained()`. Downstream `from_pretrained(checkpoint_dir)`
        # will then fail with "does not appear to have a file named modeling_xxx.py".
        self._maybe_include_remote_code_files(output_dir)

        if qwen35_checkpoint:
            self._validate_qwen35_text_checkpoint_layout(output_dir)

    def _validate_qwen35_text_checkpoint_layout(self, output_dir: str) -> None:
        """Reject a Qwen3.5 text checkpoint whose config and weight keys disagree."""
        config_path = os.path.join(output_dir, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3.5 checkpoint is missing a readable config.json: {config_path}"
            ) from exc

        model_type = config.get("model_type")
        if not is_qwen35_text_model_type(model_type):
            raise RuntimeError(
                "Qwen3.5 text checkpoint has an incompatible final config: "
                f"model_type={model_type!r}, path={config_path}"
            )

        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("weight_map is empty")
                weight_files = sorted(
                    {os.path.join(output_dir, str(name)) for name in weight_map.values()}
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Qwen3.5 checkpoint has an unreadable safetensors index: {index_path}"
                ) from exc
        else:
            weight_path = os.path.join(output_dir, "model.safetensors")
            weight_files = [weight_path] if os.path.isfile(weight_path) else []

        # The normal pipeline uses safe serialization. For an explicitly
        # configured legacy .bin save, save_original_format=False above is still
        # applied, but avoid loading the entire checkpoint back into host RAM.
        if not weight_files:
            if getattr(self.args, "save_safetensors", False):
                raise RuntimeError(
                    f"Qwen3.5 checkpoint did not produce safetensors weights: {output_dir}"
                )
            return

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to validate a Qwen3.5 checkpoint layout"
            ) from exc

        has_text_weights = False
        nested_keys = []
        for weight_file in weight_files:
            if not os.path.isfile(weight_file):
                raise RuntimeError(
                    f"Qwen3.5 checkpoint index references a missing shard: {weight_file}"
                )
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(("model.language_model.", "language_model.")):
                        nested_keys.append(key)
                    if key.startswith("model.") and not key.startswith("model.language_model."):
                        has_text_weights = True

        if nested_keys:
            preview = ", ".join(nested_keys[:3])
            raise RuntimeError(
                "Qwen3.5 text checkpoint contains multimodal-container weight keys; "
                f"examples=[{preview}], path={output_dir}"
            )
        if not has_text_weights:
            raise RuntimeError(
                f"Qwen3.5 text checkpoint contains no model.* weights: {output_dir}"
            )

    def _ensure_qwen35_text_config(self, output_dir: str) -> None:
        """Ensure a normalized Qwen3.5 checkpoint has a text-only config.

        ``save_pretrained`` normally emits this config when Transformers has
        exposed the inner causal LM.  The fallback below covers versions that
        instead serialize the original multimodal container config.
        """
        config_path = os.path.join(output_dir, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3.5 checkpoint is missing a readable config.json: {config_path}"
            ) from exc

        model_type = config.get("model_type")
        architectures = config.get("architectures")
        is_moe = "moe" in str(model_type).lower() or any(
            "Moe" in str(arch) for arch in (architectures or [])
        )
        expected_model_type = "qwen3_5_moe_text" if is_moe else "qwen3_5_text"
        expected_architectures = [
            "Qwen3_5MoeForCausalLM" if is_moe else "Qwen3_5ForCausalLM"
        ]
        if model_type == expected_model_type and architectures == expected_architectures:
            return
        if model_type == expected_model_type:
            normalized = dict(config)
        else:
            text_config = config.get("text_config")
            if not isinstance(text_config, dict):
                raise RuntimeError(
                    "Qwen3.5 checkpoint did not produce a text_config, so its text-only "
                    f"config cannot be reconstructed: {config_path}"
                )
            normalized = dict(text_config)
        normalized["model_type"] = expected_model_type
        normalized["architectures"] = expected_architectures
        for key in ("transformers_version", "dtype"):
            if key in config and key not in normalized:
                normalized[key] = config[key]

        fd, tmp_path = tempfile.mkstemp(
            prefix=".config.json.", suffix=".tmp", dir=output_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, config_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _maybe_restore_non_weight_files_from_source(self, output_dir: str, *, training_args_name: str) -> None:
        """
        Best-effort: copy non-weight files from the source model directory into `output_dir`.

        This keeps metadata stable across checkpoints (e.g., tokenizer chat_template, config.json, etc.).
        """
        if not self.accelerator.is_main_process:
            return

        # Resolve the "source" directory (where to copy metadata from).
        #
        # Priority:
        # 1) explicit `metadata_source_dir` (recommended: initial/base model directory)
        # 2) tokenizer/model name_or_path if it's a local folder
        src_root = None
        if isinstance(getattr(self, "metadata_source_dir", None), str) and os.path.isdir(str(self.metadata_source_dir)):
            src_root = str(self.metadata_source_dir)
        candidates = []
        try:
            if self.tokenizer is not None:
                candidates.append(getattr(self.tokenizer, "name_or_path", None))
        except Exception:
            pass
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            candidates.append(getattr(unwrapped, "name_or_path", None))
            candidates.append(getattr(getattr(unwrapped, "config", None), "_name_or_path", None))
        except Exception:
            candidates.append(getattr(getattr(self.model, "config", None), "_name_or_path", None))

        for c in candidates:
            if isinstance(c, str) and c and os.path.isdir(c):
                if src_root is None:
                    src_root = c
                    break

        if not src_root:
            return
        if os.path.abspath(src_root) == os.path.abspath(output_dir):
            # In-place save; nothing to restore.
            return

        def _read_model_type(root: str | None) -> str | None:
            if not root:
                return None
            cfg_path = os.path.join(root, "config.json")
            if not os.path.isfile(cfg_path):
                return None
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                return None
            model_type = cfg.get("model_type")
            return model_type if isinstance(model_type, str) else None

        # `save_pretrained()` already wrote a checkpoint-specific config.json.
        # For Qwen3.5 text checkpoints, the source model metadata is multimodal
        # (`model_type=qwen3_5`) and would overwrite the saved text-only config
        # (`model_type=qwen3_5_text`), reintroducing the vLLM/HF load failure.
        saved_model_type = _read_model_type(output_dir)
        source_model_type = _read_model_type(src_root)
        preserve_saved_config = (
            is_qwen35_text_model_type(saved_model_type)
            and is_qwen35_model_type(source_model_type)
        )

        # Exclude training/runtime artifacts and anything that could overwrite weights.
        excluded_names = {
            str(training_args_name),
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "random_states.pkl",
            "scaler.pt",
        }
        if preserve_saved_config:
            excluded_names.add("config.json")
            logger.info(
                "Preserving checkpoint-generated config.json for Qwen3.5 text checkpoint "
                "instead of restoring the source multimodal config."
            )
        excluded_dirs = {".git", ".dvc", ".idea", ".vscode", "__pycache__", "wandb", "runs"}

        def _is_probably_weight_file(name: str) -> bool:
            n = str(name)
            # Safetensors are always weights in HF checkpoints.
            if n.endswith(".safetensors") or n.endswith(".safetensors.index.json"):
                return True
            # Common HF weight shard patterns.
            if n.startswith(("pytorch_model", "model", "adapter_model")) and n.endswith(".bin"):
                return True
            if n.startswith(("pytorch_model", "model", "adapter_model")) and n.endswith(".bin.index.json"):
                return True
            # Some stacks may use .pt/.pth for consolidated weights.
            if n.endswith((".pt", ".pth")) and ("model" in n or "weights" in n or "consolidated" in n):
                return True
            return False

        max_copy_bytes = 200 * 1024 * 1024  # safety guard
        copied = 0
        skipped_large = 0
        try:
            for root, dirs, files in os.walk(src_root):
                # Prune excluded dirs early.
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                for fname in files:
                    if fname in excluded_names:
                        continue
                    if _is_probably_weight_file(fname):
                        continue
                    src_path = os.path.join(root, fname)
                    try:
                        st = os.stat(src_path)
                        if int(st.st_size) > max_copy_bytes:
                            skipped_large += 1
                            continue
                    except Exception:
                        # If we can't stat, skip silently.
                        continue

                    rel = os.path.relpath(src_path, src_root)
                    dst_path = os.path.join(output_dir, rel)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied += 1
                    except Exception:
                        # Best-effort; skip problematic files.
                        continue
        finally:
            if copied > 0:
                logger.info(
                    f"Restored {copied} non-weight files from source model dir into checkpoint (src={src_root})."
                )
            if skipped_large > 0:
                logger.info(
                    f"Skipped {skipped_large} large files while restoring metadata from source (>{max_copy_bytes} bytes)."
                )

    def _maybe_include_remote_code_files(self, output_dir: str) -> None:
        """
        Best-effort: copy `trust_remote_code` python files from the source model repo into `output_dir`.

        This makes the saved checkpoint self-contained for offline evaluation.
        """
        if not self.accelerator.is_main_process:
            return

        cfg_path = os.path.join(output_dir, "config.json")
        if not os.path.exists(cfg_path):
            return

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read config.json for remote-code copy: {e}")
            return

        auto_map = cfg.get("auto_map", None)
        if not isinstance(auto_map, dict) or not auto_map:
            return

        # Extract module file paths from auto_map values.
        # Examples:
        # - "modeling_phi3.Phi3ForCausalLM" -> "modeling_phi3.py"
        # - "phi3.modeling_phi3.Phi3ForCausalLM" -> "phi3/modeling_phi3.py"
        needed_rel_paths = set()
        for _k, v in auto_map.items():
            values = v if isinstance(v, (list, tuple)) else [v]
            for item in values:
                if not isinstance(item, str):
                    continue
                module_path = item.rsplit(".", 1)[0] if "." in item else item
                rel_py = module_path.replace(".", "/") + ".py"
                needed_rel_paths.add(rel_py)

        # Determine where the base model code lives (local dir or HF cache snapshot).
        source = None
        try:
            # Prefer model's recorded name/path (set by from_pretrained)
            unwrapped = self.accelerator.unwrap_model(self.model)
            source = getattr(unwrapped, "name_or_path", None) or getattr(getattr(unwrapped, "config", None), "_name_or_path", None)
        except Exception:
            source = getattr(getattr(self.model, "config", None), "_name_or_path", None)

        if not source:
            return

        src_root = None
        if isinstance(source, str) and os.path.isdir(source):
            src_root = source
        else:
            # HF Hub repo id or unknown: try to resolve from local cache only (no network).
            try:
                from huggingface_hub import snapshot_download

                src_root = snapshot_download(
                    repo_id=str(source),
                    local_files_only=True,
                )
            except Exception as e:
                logger.warning(
                    f"Could not resolve remote-code source repo '{source}' from local HF cache; "
                    f"checkpoint may miss custom modeling files. Error: {e}"
                )
                return

        copied_any = False
        for rel_path in sorted(needed_rel_paths):
            dst_path = os.path.join(output_dir, rel_path)
            if os.path.exists(dst_path):
                continue

            # Ensure intermediate packages are importable if the module lives in subfolders.
            # We prefer copying existing __init__.py from source; otherwise create empty ones.
            rel_dir = os.path.dirname(rel_path)
            if rel_dir:
                parts = rel_dir.split("/")
                for i in range(1, len(parts) + 1):
                    pkg_rel = "/".join(parts[:i])
                    init_rel = os.path.join(pkg_rel, "__init__.py")
                    init_dst = os.path.join(output_dir, init_rel)
                    if os.path.exists(init_dst):
                        continue
                    init_src = os.path.join(src_root, init_rel)
                    os.makedirs(os.path.dirname(init_dst), exist_ok=True)
                    try:
                        if os.path.exists(init_src):
                            shutil.copy2(init_src, init_dst)
                        else:
                            with open(init_dst, "w", encoding="utf-8") as f:
                                f.write("")
                    except Exception:
                        # Non-fatal; continue copying the main module.
                        pass

            src_path = os.path.join(src_root, rel_path)
            if not os.path.exists(src_path):
                # Some repos place files at top-level even if module path includes a package.
                # Fall back to the basename match at repo root.
                alt_src_path = os.path.join(src_root, os.path.basename(rel_path))
                if os.path.exists(alt_src_path):
                    src_path = alt_src_path
                else:
                    logger.warning(f"Missing remote-code file in source repo: {rel_path} (looked in {src_root})")
                    continue

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy2(src_path, dst_path)
                copied_any = True
            except Exception as e:
                logger.warning(f"Failed to copy remote-code file {src_path} -> {dst_path}: {e}")

        if copied_any:
            logger.info(f"Included trust_remote_code python files in checkpoint: {output_dir}")
