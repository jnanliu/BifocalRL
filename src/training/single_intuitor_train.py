# coding=utf-8
"""
Single-model Intuitor/WMV training from cache.

This script trains ONE policy model using data produced by `src/generate_and_score_single.py`.

Two reward modes:
- intuitor: use SCE scores directly as rewards (continuous), then group-normalize to advantages.
- wmv: use SCE scores as weights, run weighted majority voting to get a pseudo-label, assign 0/1 rewards, then
       group-normalize to advantages.

The loss is the same GRPO-style objective as `src/co_intuitor_train.py` (`compute_grpo_loss`).
"""

from __future__ import annotations

import logging
import os
import sys
import random
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from utils.trl_compat import ModelConfig, TrlParser, pad

from utils.cache import CoIntuitorCache
from utils.mv import extract_answers, rewards_from_mv_pseudolabel, rewards_from_wmv_pseudolabel
from utils.normalization import z_score_normalize_tensor
from utils.single_model_trainer import SingleModelTrainer, compute_pipeline_update_plan
from utils.configs import CoIntuitorConfig
from utils.invariants import (
    assert_prompt_completion_shapes,
    tc_check_generations_block,
    tc_check_logps_matrix,
    tc_check_score_vector,
    tc_check_training_batch,
)
from utils.generate import strip_prompt_echo_from_flattened_responses, strip_chat_template_artifacts_from_flattened_responses
from utils.tokenization import append_eos_1d

# Reuse loss implementations from the co pipeline.
from src.training.co_intuitor_train import compute_grpo_loss, compute_policy_loss, get_model, get_tokenizer  # type: ignore

from utils.env_utils import co_env_flag
from utils.debug_utils import co_debug_data_dir, co_write_json_atomic

logger = logging.getLogger(__name__)


_REQUIRED_MIXED_REWARD_DIAGNOSTIC_MODES = frozenset(
    {"reliability_sc_corr", "prism", "ettrl"}
)


def _requires_mixed_reward_diagnostics(reward_mode: str, alpha_mode: str) -> bool:
    return (
        str(reward_mode).strip().lower() == "mixed_reward"
        and str(alpha_mode).strip().lower()
        in _REQUIRED_MIXED_REWARD_DIAGNOSTIC_MODES
    )


def _requires_reliability_diagnostics(reward_mode: str, alpha_mode: str) -> bool:
    """Compatibility helper retained for existing diagnostics callers/tests."""
    return (
        str(reward_mode).strip().lower() == "mixed_reward"
        and str(alpha_mode).strip().lower() == "reliability_sc_corr"
    )


def _diagnostic_tensor(
    diagnostics: MutableMapping[str, Any],
    name: str,
    expected_shape: Tuple[int, ...],
) -> torch.Tensor:
    value = diagnostics.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Missing required mixed-reward diagnostic tensor: {name}")
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tuple(tensor.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"Mixed-reward diagnostic {name} has shape {tuple(tensor.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"Mixed-reward diagnostic {name} contains NaN/Inf")
    return tensor


def _diagnostic_stats(tensor: torch.Tensor) -> Dict[str, float]:
    flat = tensor.reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _rows_as_floats(tensor: torch.Tensor) -> List[List[float]]:
    return [[float(value) for value in row] for row in tensor.tolist()]


def _vector_as_floats(tensor: torch.Tensor) -> List[float]:
    return [float(value) for value in tensor.reshape(-1).tolist()]


def _build_reliability_diagnostics_payload(
    *,
    step: int,
    prompts: List[str],
    sampled_indices: List[int],
    num_generations: int,
    diagnostics: MutableMapping[str, Any],
    training_advantages: torch.Tensor,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Build and validate the required per-step Reliability diagnostic artifact."""
    num_prompts = len(prompts)
    num_generations = int(num_generations)
    if len(sampled_indices) != num_prompts:
        raise RuntimeError(
            "Reliability diagnostics sampled-index count does not match prompts: "
            f"indices={len(sampled_indices)}, prompts={num_prompts}"
        )

    pg_shape = (num_prompts, num_generations)
    p1_shape = (num_prompts, 1)
    mv_rewards = _diagnostic_tensor(diagnostics, "mv_rewards", pg_shape)
    self_rewards = _diagnostic_tensor(diagnostics, "self_certainty_rewards", pg_shape)
    mv_advantages = _diagnostic_tensor(diagnostics, "mv_advantages", pg_shape)
    self_advantages = _diagnostic_tensor(
        diagnostics, "self_certainty_advantages", pg_shape
    )
    raw_sc = _diagnostic_tensor(diagnostics, "raw_sc", p1_shape)
    raw_corr = _diagnostic_tensor(diagnostics, "raw_corr", p1_shape)
    z_sc = _diagnostic_tensor(diagnostics, "z_sc", p1_shape)
    z_corr = _diagnostic_tensor(diagnostics, "z_corr", p1_shape)
    reliability = _diagnostic_tensor(diagnostics, "reliability", p1_shape)
    computed_sample = _diagnostic_tensor(
        diagnostics, "computed_sample_weight", p1_shape
    )
    applied_sample = _diagnostic_tensor(
        diagnostics, "applied_sample_weight", p1_shape
    )
    computed_advantage = _diagnostic_tensor(
        diagnostics, "computed_advantage_weight", p1_shape
    )
    applied_advantage = _diagnostic_tensor(
        diagnostics, "applied_advantage_weight", p1_shape
    )
    combined_before = _diagnostic_tensor(
        diagnostics, "combined_before_final_normalize", pg_shape
    )
    combined_after = _diagnostic_tensor(
        diagnostics, "combined_after_final_normalize", pg_shape
    )
    final_advantages = _diagnostic_tensor(diagnostics, "final_advantages", pg_shape)

    actual_training_advantages = training_advantages.detach().to(
        device="cpu", dtype=torch.float32
    ).view(pg_shape)
    if not torch.equal(final_advantages, actual_training_advantages):
        max_diff = float((final_advantages - actual_training_advantages).abs().max().item())
        raise RuntimeError(
            "Saved mixed-reward final advantages do not match the training batch: "
            f"max_abs_diff={max_diff}"
        )

    sample_lower = float(parameters["sample_weight"]["lower"])
    sample_upper = float(parameters["sample_weight"]["upper"])
    advantage_lower = float(parameters["advantage_weight"]["lower"])
    advantage_upper = float(parameters["advantage_weight"]["upper"])
    tolerance = 1e-6
    if float(computed_sample.min()) < sample_lower - tolerance or float(
        computed_sample.max()
    ) > sample_upper + tolerance:
        raise RuntimeError("Computed sample weights fall outside configured bounds")
    if float(computed_advantage.min()) < advantage_lower - tolerance or float(
        computed_advantage.max()
    ) > advantage_upper + tolerance:
        raise RuntimeError("Computed advantage weights fall outside configured bounds")

    use_sample = bool(diagnostics.get("use_sample_weight"))
    use_advantage = bool(diagnostics.get("use_advantage_weight"))
    if use_sample != bool(parameters.get("use_sample_weight")):
        raise RuntimeError("Diagnostic sample-weight switch does not match parameters")
    if use_advantage != bool(parameters.get("use_advantage_weight")):
        raise RuntimeError("Diagnostic advantage-weight switch does not match parameters")
    diagnostics_only = bool(diagnostics.get("diagnostics_only_reliability", False))
    if diagnostics_only != (not use_sample and not use_advantage):
        raise RuntimeError("diagnostics_only_reliability is inconsistent with weight switches")

    expected_applied_sample = (
        computed_sample if use_sample else torch.ones_like(computed_sample)
    )
    expected_applied_advantage = (
        computed_advantage
        if use_advantage
        else torch.full_like(computed_advantage, 0.5)
    )
    if not torch.equal(applied_sample, expected_applied_sample):
        raise RuntimeError("Applied sample weights do not match the configured switch")
    if not torch.equal(applied_advantage, expected_applied_advantage):
        raise RuntimeError("Applied advantage weights do not match the configured switch")

    if not torch.allclose(raw_sc, mv_rewards.mean(dim=1, keepdim=True), atol=1e-7, rtol=0.0):
        raise RuntimeError("raw_sc does not equal the per-prompt mean MV reward")
    expected_reliability = (
        float(parameters["sc_weight"]) * z_sc
        + float(parameters["corr_weight"]) * z_corr
    )
    if not torch.allclose(reliability, expected_reliability, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Reliability does not match the configured SC/Corr weighted sum")
    expected_combined_before = (
        applied_advantage * mv_advantages
        + (1.0 - applied_advantage) * self_advantages
    )
    if not torch.allclose(
        combined_before, expected_combined_before, atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Pre-normalize combined advantage does not match applied weights")
    expected_final = applied_sample * combined_after
    if not torch.allclose(final_advantages, expected_final, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Final advantages do not match applied sample weights")

    mv_votes = diagnostics.get("mv_votes")
    extracted_answers = diagnostics.get("mv_extracted_answers")
    if not isinstance(mv_votes, list) or len(mv_votes) != num_prompts:
        raise RuntimeError("Reliability diagnostics are missing aligned MV vote records")
    if not isinstance(extracted_answers, list) or len(extracted_answers) != num_prompts:
        raise RuntimeError("Reliability diagnostics are missing aligned extracted answers")
    if any(not isinstance(row, list) or len(row) != num_generations for row in extracted_answers):
        raise RuntimeError("Extracted-answer rows do not match num_generations")

    prompt_indices = list(range(num_prompts))
    reliability_values = _vector_as_floats(reliability)
    dataset_indices = [int(value) for value in sampled_indices]
    ranking_desc_indices = sorted(
        prompt_indices, key=lambda index: (-reliability_values[index], index)
    )
    ranking_asc_indices = sorted(
        prompt_indices, key=lambda index: (reliability_values[index], index)
    )

    def _ranking(indices: List[int]) -> List[Dict[str, Any]]:
        return [
            {
                "rank": rank,
                "prompt_index": prompt_index,
                "dataset_index": dataset_indices[prompt_index],
                "reliability": reliability_values[prompt_index],
            }
            for rank, prompt_index in enumerate(indices, start=1)
        ]

    mv_reward_rows = _rows_as_floats(mv_rewards)
    self_reward_rows = _rows_as_floats(self_rewards)
    mv_advantage_rows = _rows_as_floats(mv_advantages)
    self_advantage_rows = _rows_as_floats(self_advantages)
    combined_before_rows = _rows_as_floats(combined_before)
    combined_after_rows = _rows_as_floats(combined_after)
    final_rows = _rows_as_floats(final_advantages)
    computed_sample_values = _vector_as_floats(computed_sample)
    applied_sample_values = _vector_as_floats(applied_sample)
    computed_advantage_values = _vector_as_floats(computed_advantage)
    applied_advantage_values = _vector_as_floats(applied_advantage)

    records: List[Dict[str, Any]] = []
    for prompt_index in prompt_indices:
        vote = mv_votes[prompt_index]
        if not isinstance(vote, dict):
            raise RuntimeError(f"Invalid MV vote record at prompt {prompt_index}")
        records.append(
            {
                "prompt_index": prompt_index,
                "dataset_index": dataset_indices[prompt_index],
                "prompt": str(prompts[prompt_index]),
                "mv_vote": {
                    "selected_answer": vote.get("selected_answer"),
                    "answer_clusters": vote.get("answer_clusters", {}),
                    "extracted_answers": extracted_answers[prompt_index],
                },
                "reliability": {
                    "raw_sc": float(raw_sc[prompt_index, 0]),
                    "raw_corr": float(raw_corr[prompt_index, 0]),
                    "z_sc": float(z_sc[prompt_index, 0]),
                    "z_corr": float(z_corr[prompt_index, 0]),
                    "value": reliability_values[prompt_index],
                },
                "weights": {
                    "computed_sample": computed_sample_values[prompt_index],
                    "applied_sample": applied_sample_values[prompt_index],
                    "computed_advantage": computed_advantage_values[prompt_index],
                    "applied_advantage": applied_advantage_values[prompt_index],
                },
                "mv": {
                    "rewards": mv_reward_rows[prompt_index],
                    "advantages": mv_advantage_rows[prompt_index],
                },
                "self_certainty": {
                    "rewards": self_reward_rows[prompt_index],
                    "advantages": self_advantage_rows[prompt_index],
                },
                "mixed": {
                    "before_final_normalize": combined_before_rows[prompt_index],
                    "after_final_normalize": combined_after_rows[prompt_index],
                    "final_advantages": final_rows[prompt_index],
                },
            }
        )

    return {
        "schema_version": 1,
        "step": int(step),
        "num_prompts": num_prompts,
        "num_generations": num_generations,
        "num_responses": num_prompts * num_generations,
        "diagnostics_only_reliability": diagnostics_only,
        "parameters": parameters,
        "tensor_shapes": {
            "prompt_response": list(pg_shape),
            "prompt_scalar": list(p1_shape),
        },
        "prompt_indices": prompt_indices,
        "dataset_indices": dataset_indices,
        "batch_reliability": reliability_values,
        "batch_reliability_components": {
            "raw_sc": _vector_as_floats(raw_sc),
            "raw_corr": _vector_as_floats(raw_corr),
            "z_sc": _vector_as_floats(z_sc),
            "z_corr": _vector_as_floats(z_corr),
        },
        "batch_weights": {
            "computed_sample": computed_sample_values,
            "applied_sample": applied_sample_values,
            "computed_advantage": computed_advantage_values,
            "applied_advantage": applied_advantage_values,
        },
        "reliability_ranking_desc": _ranking(ranking_desc_indices),
        "reliability_ranking_asc": _ranking(ranking_asc_indices),
        "batch_statistics": {
            "raw_sc": _diagnostic_stats(raw_sc),
            "raw_corr": _diagnostic_stats(raw_corr),
            "z_sc": _diagnostic_stats(z_sc),
            "z_corr": _diagnostic_stats(z_corr),
            "reliability": _diagnostic_stats(reliability),
            "computed_sample_weight": _diagnostic_stats(computed_sample),
            "applied_sample_weight": _diagnostic_stats(applied_sample),
            "computed_advantage_weight": _diagnostic_stats(computed_advantage),
            "applied_advantage_weight": _diagnostic_stats(applied_advantage),
        },
        "batch_prompt_summaries": {
            "mv_reward_mean": _vector_as_floats(mv_rewards.mean(dim=1)),
            "self_certainty_reward_mean": _vector_as_floats(self_rewards.mean(dim=1)),
            "mv_advantage_abs_mean": _vector_as_floats(mv_advantages.abs().mean(dim=1)),
            "self_certainty_advantage_abs_mean": _vector_as_floats(
                self_advantages.abs().mean(dim=1)
            ),
            "final_advantage_abs_mean": _vector_as_floats(
                final_advantages.abs().mean(dim=1)
            ),
        },
        "records": records,
    }


def _write_reliability_diagnostics(
    *,
    base_dir: str,
    step: int,
    payload: Dict[str, Any],
) -> str:
    dump_dir = co_debug_data_dir(str(base_dir))
    path = os.path.join(
        dump_dir, f"mixed_reward_reliability_step{int(step)}.json"
    )
    co_write_json_atomic(path, payload)
    return path


def _write_reliability_diagnostics_if_main(
    *,
    is_main_process: bool,
    base_dir: str,
    step: int,
    payload: Dict[str, Any],
) -> Optional[str]:
    if not is_main_process:
        return None
    return _write_reliability_diagnostics(
        base_dir=base_dir,
        step=step,
        payload=payload,
    )


def _build_prism_ettrl_diagnostics_payload(
    *,
    step: int,
    prompts: List[str],
    sampled_indices: List[int],
    num_generations: int,
    diagnostics: MutableMapping[str, Any],
    training_advantages: torch.Tensor,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the required per-step diagnostic artifact for PRISM/eTTRL."""
    mode = str(parameters.get("alpha_mode", "")).strip().lower()
    if mode not in {"prism", "ettrl"}:
        raise RuntimeError(f"Unsupported PRISM/eTTRL diagnostic mode: {mode!r}")
    if bool(parameters.get("final_normalize", True)):
        raise RuntimeError(f"{mode} diagnostics require final_normalize=false")

    num_prompts = len(prompts)
    num_generations = int(num_generations)
    if len(sampled_indices) != num_prompts:
        raise RuntimeError(
            f"{mode} diagnostics sampled-index count does not match prompts: "
            f"indices={len(sampled_indices)}, prompts={num_prompts}"
        )

    pg_shape = (num_prompts, num_generations)
    p1_shape = (num_prompts, 1)
    mv_rewards = _diagnostic_tensor(diagnostics, "mv_rewards", pg_shape)
    self_rewards = _diagnostic_tensor(
        diagnostics, "self_certainty_rewards", pg_shape
    )
    mv_advantages = _diagnostic_tensor(diagnostics, "mv_advantages", pg_shape)
    self_advantages = _diagnostic_tensor(
        diagnostics, "self_certainty_advantages", pg_shape
    )
    self_group_mean = _diagnostic_tensor(
        diagnostics, "self_certainty_group_mean", p1_shape
    )
    mv_weight = _diagnostic_tensor(
        diagnostics, "mv_advantage_weight", pg_shape
    )
    self_weight = _diagnostic_tensor(
        diagnostics, "self_certainty_advantage_weight", pg_shape
    )
    combined_before = _diagnostic_tensor(
        diagnostics, "combined_before_final_normalize", pg_shape
    )
    combined_after = _diagnostic_tensor(
        diagnostics, "combined_after_final_normalize", pg_shape
    )
    final_advantages = _diagnostic_tensor(
        diagnostics, "final_advantages", pg_shape
    )

    actual_training_advantages = training_advantages.detach().to(
        device="cpu", dtype=torch.float32
    ).view(pg_shape)
    if not torch.equal(final_advantages, actual_training_advantages):
        max_diff = float(
            (final_advantages - actual_training_advantages).abs().max().item()
        )
        raise RuntimeError(
            f"Saved {mode} final advantages do not match the training batch: "
            f"max_abs_diff={max_diff}"
        )

    expected_group_mean = self_rewards.mean(dim=1, keepdim=True)
    if not torch.equal(self_group_mean, expected_group_mean):
        raise RuntimeError(
            f"{mode} self-certainty group means do not match raw rewards"
        )

    if mode == "prism":
        ettrl_all_zero_group = None
        if not torch.equal(mv_weight, torch.ones_like(mv_weight)):
            raise RuntimeError("PRISM MV advantage weights must all equal 1")
        if not torch.equal(self_weight, torch.ones_like(self_weight)):
            raise RuntimeError("PRISM self-certainty advantage weights must all equal 1")
    else:
        if bool((self_rewards < 0).any().item()):
            raise RuntimeError("eTTRL diagnostics found negative self-certainty rewards")
        all_zero_group = (self_rewards == 0).all(dim=1, keepdim=True)
        ettrl_all_zero_group = _diagnostic_tensor(
            diagnostics, "ettrl_all_zero_group", p1_shape
        )
        if not torch.equal(
            ettrl_all_zero_group, all_zero_group.to(dtype=torch.float32)
        ):
            raise RuntimeError("eTTRL all-zero-group fallback flags are inconsistent")
        safe_mean = torch.where(
            all_zero_group, torch.ones_like(self_group_mean), self_group_mean
        )
        expected_mv_weight = self_rewards / safe_mean
        expected_mv_weight = torch.where(
            all_zero_group.expand_as(expected_mv_weight),
            torch.ones_like(expected_mv_weight),
            expected_mv_weight,
        )
        if not torch.allclose(mv_weight, expected_mv_weight, atol=1e-7, rtol=1e-7):
            raise RuntimeError("eTTRL MV weights do not match raw SCE / group mean")
        if not torch.equal(self_weight, torch.zeros_like(self_weight)):
            raise RuntimeError("eTTRL self-certainty advantage weights must all equal 0")
        nonzero_groups = ~all_zero_group.reshape(-1)
        if bool(nonzero_groups.any().item()):
            weight_means = mv_weight.mean(dim=1)
            if not torch.allclose(
                weight_means[nonzero_groups],
                torch.ones_like(weight_means[nonzero_groups]),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise RuntimeError("eTTRL response weights must average to 1 per group")

    expected_combined = mv_weight * mv_advantages + self_weight * self_advantages
    if not torch.allclose(
        combined_before, expected_combined, atol=1e-7, rtol=1e-7
    ):
        raise RuntimeError(f"{mode} combined advantages do not match applied weights")
    if not torch.equal(combined_after, combined_before):
        raise RuntimeError(f"{mode} unexpectedly changed advantages during final normalize")
    if not torch.equal(final_advantages, combined_after):
        raise RuntimeError(f"{mode} unexpectedly applied a post-normalize weight")

    mv_votes = diagnostics.get("mv_votes")
    extracted_answers = diagnostics.get("mv_extracted_answers")
    if not isinstance(mv_votes, list) or len(mv_votes) != num_prompts:
        raise RuntimeError(f"{mode} diagnostics are missing aligned MV vote records")
    if not isinstance(extracted_answers, list) or len(extracted_answers) != num_prompts:
        raise RuntimeError(f"{mode} diagnostics are missing aligned extracted answers")
    if any(
        not isinstance(row, list) or len(row) != num_generations
        for row in extracted_answers
    ):
        raise RuntimeError("Extracted-answer rows do not match num_generations")

    prompt_indices = list(range(num_prompts))
    dataset_indices = [int(value) for value in sampled_indices]
    mv_reward_rows = _rows_as_floats(mv_rewards)
    self_reward_rows = _rows_as_floats(self_rewards)
    mv_advantage_rows = _rows_as_floats(mv_advantages)
    self_advantage_rows = _rows_as_floats(self_advantages)
    mv_weight_rows = _rows_as_floats(mv_weight)
    self_weight_rows = _rows_as_floats(self_weight)
    combined_before_rows = _rows_as_floats(combined_before)
    combined_after_rows = _rows_as_floats(combined_after)
    final_rows = _rows_as_floats(final_advantages)
    self_group_means = _vector_as_floats(self_group_mean)

    records: List[Dict[str, Any]] = []
    for prompt_index in prompt_indices:
        vote = mv_votes[prompt_index]
        if not isinstance(vote, dict):
            raise RuntimeError(f"Invalid MV vote record at prompt {prompt_index}")
        record = {
            "prompt_index": prompt_index,
            "dataset_index": dataset_indices[prompt_index],
            "prompt": str(prompts[prompt_index]),
            "mv_vote": {
                "selected_answer": vote.get("selected_answer"),
                "answer_clusters": vote.get("answer_clusters", {}),
                "extracted_answers": extracted_answers[prompt_index],
            },
            "weights": {
                "mv_advantage": mv_weight_rows[prompt_index],
                "self_certainty_advantage": self_weight_rows[prompt_index],
            },
            "mv": {
                "rewards": mv_reward_rows[prompt_index],
                "advantages": mv_advantage_rows[prompt_index],
            },
            "self_certainty": {
                "rewards": self_reward_rows[prompt_index],
                "group_mean": self_group_means[prompt_index],
                "advantages": self_advantage_rows[prompt_index],
            },
            "mixed": {
                "before_final_normalize": combined_before_rows[prompt_index],
                "after_final_normalize": combined_after_rows[prompt_index],
                "final_advantages": final_rows[prompt_index],
            },
        }
        if ettrl_all_zero_group is not None:
            record["self_certainty"]["all_zero_group"] = bool(
                ettrl_all_zero_group[prompt_index, 0].item()
            )
        records.append(record)

    payload = {
        "schema_version": 1,
        "step": int(step),
        "num_prompts": num_prompts,
        "num_generations": num_generations,
        "num_responses": num_prompts * num_generations,
        "alpha_mode": mode,
        "parameters": parameters,
        "tensor_shapes": {
            "prompt_response": list(pg_shape),
            "prompt_scalar": list(p1_shape),
        },
        "prompt_indices": prompt_indices,
        "dataset_indices": dataset_indices,
        "batch_self_certainty_group_mean": self_group_means,
        "batch_weights": {
            "mv_advantage": mv_weight_rows,
            "self_certainty_advantage": self_weight_rows,
        },
        "batch_statistics": {
            "mv_rewards": _diagnostic_stats(mv_rewards),
            "self_certainty_rewards": _diagnostic_stats(self_rewards),
            "mv_advantages": _diagnostic_stats(mv_advantages),
            "self_certainty_advantages": _diagnostic_stats(self_advantages),
            "self_certainty_group_mean": _diagnostic_stats(self_group_mean),
            "mv_advantage_weight": _diagnostic_stats(mv_weight),
            "self_certainty_advantage_weight": _diagnostic_stats(self_weight),
            "combined_before_final_normalize": _diagnostic_stats(combined_before),
            "final_advantages": _diagnostic_stats(final_advantages),
        },
        "batch_prompt_summaries": {
            "mv_reward_mean": _vector_as_floats(mv_rewards.mean(dim=1)),
            "self_certainty_reward_mean": self_group_means,
            "mv_advantage_abs_mean": _vector_as_floats(
                mv_advantages.abs().mean(dim=1)
            ),
            "self_certainty_advantage_abs_mean": _vector_as_floats(
                self_advantages.abs().mean(dim=1)
            ),
            "mv_advantage_weight_mean": _vector_as_floats(mv_weight.mean(dim=1)),
            "self_certainty_advantage_weight_mean": _vector_as_floats(
                self_weight.mean(dim=1)
            ),
            "final_advantage_abs_mean": _vector_as_floats(
                final_advantages.abs().mean(dim=1)
            ),
        },
        "records": records,
    }
    if ettrl_all_zero_group is not None:
        payload["batch_ettrl_all_zero_group"] = [
            bool(value) for value in ettrl_all_zero_group.reshape(-1).tolist()
        ]
    return payload


def _write_prism_ettrl_diagnostics_if_main(
    *,
    is_main_process: bool,
    base_dir: str,
    step: int,
    alpha_mode: str,
    payload: Dict[str, Any],
) -> Optional[str]:
    if not is_main_process:
        return None
    mode = str(alpha_mode).strip().lower()
    if mode not in {"prism", "ettrl"}:
        raise ValueError(f"Unsupported PRISM/eTTRL diagnostic mode: {alpha_mode!r}")
    dump_dir = co_debug_data_dir(str(base_dir))
    path = os.path.join(dump_dir, f"mixed_reward_{mode}_step{int(step)}.json")
    co_write_json_atomic(path, payload)
    return path


def _co_dump_single_prepare_full_samples(
    *,
    base_dir: str,
    step: int,
    reward_mode: str,
    tokenizer,
    prompts: List[str],
    completions: List[str],
    num_generations: int,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    max_items: int = 10,
) -> None:
    """
    Dump fully-assembled training samples for the single-model pipeline:
      - text: prompt (already chat_template-rendered), completion text, and token-decoded padded views
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
        dump_dir = co_debug_data_dir(str(base_dir))
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"prepare_train_full_single_step{int(step)}_rank{rank}.jsonl")

        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        chat_template = getattr(tokenizer, "chat_template", None)

        def _head_tail(seq: Optional[List[Any]], max_total: int = 50) -> Dict[str, Any]:
            if seq is None:
                return {"len": None, "head": None, "tail": None, "truncated": None}
            n0 = int(len(seq))
            if n0 <= int(max_total):
                return {"len": n0, "head": seq, "tail": [], "truncated": False}
            h = int(max_total) // 2
            t = int(max_total) - h
            return {"len": n0, "head": seq[:h], "tail": seq[-t:], "truncated": True}

        n = min(int(max_items), int(completion_ids.size(0)))
        for i in range(n):
            # Flattened sample mapping: prompt_idx = i // g
            pi = int(i) // int(num_generations) if int(num_generations) > 0 else 0
            prompt_text = str(prompts[pi]) if pi < len(prompts) else ""
            completion_text = str(completions[i]) if i < len(completions) else ""

            p_ids = prompt_ids[i].detach().to("cpu").tolist()
            p_mask = prompt_mask[i].detach().to("cpu").tolist()
            c_ids = completion_ids[i].detach().to("cpu").tolist()
            c_mask = completion_mask[i].detach().to("cpu").tolist()
            in_ids = p_ids + c_ids
            attn = p_mask + c_mask

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
                "ts": time.time(),
                "step": int(step),
                "reward_mode": str(reward_mode),
                "i": int(i),
                "tokenizer": {
                    "pad_token_id": (int(pad_id) if pad_id is not None else None),
                    "eos_token_id": (int(eos_id) if eos_id is not None else None),
                    "chat_template": (str(chat_template) if chat_template is not None else None),
                },
                "text": {
                    "prompt_rendered": prompt_text,
                    "completion_raw": completion_text,
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


@dataclass
class SingleTrainScriptArguments:
    policy_model_path: str = field(metadata={"help": "Policy model path to train."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing cache.json / logps_cache.json. If None, uses output_dir."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional tokenizer source-of-truth directory. If set, we load tokenizer/chat_template from this path "
                "(typically the initial/base model) instead of from the evolving checkpoint directory."
            )
        },
    )
    step: int = field(default=-1, metadata={"help": "Which cache step to train on. If -1, use latest complete."})
    reward_mode: str = field(
        default="intuitor",
        metadata={
            "help": (
                "Reward signal source. 'intuitor' = self-certainty (SCE), 'trajectory' = logprob, "
                "'truth' = ground-truth 0/1, 'wmv' = weighted majority voting, 'mv' = unweighted majority voting, "
                "'mv_gated_intuitor' = R(MV)*sigmoid(R(intuitor)), "
                "'mixed_reward' = GRPO-normalize MV and Intuitor separately then combine (GDPO). "
                "For 'intuitor' and 'trajectory', the denominator is controlled by --reward_length_norm."
            )
        },
    )
    reward_length_norm: str = field(
        default="response",
        metadata={
            "help": (
                "Length normalization denominator for 'intuitor' and 'trajectory' reward modes. "
                "'response' = divide by individual response length; "
                "'rollout' = divide by sum of all response lengths in the rollout group."
            )
        },
    )
    filter_groups: bool = field(
        default=False,
        metadata={
            "help": (
                "DAPO dynamic sampling: filter out prompt groups where all scores are identical "
                "(all correct or all wrong, i.e., std == 0). Only groups with mixed results are kept for training. "
                "Set problems_per_step > train effective batch to over-generate, then filter."
            )
        },
    )
    training_state_dir: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed checkpoint root used to persist AdamW and LR scheduler state."},
    )
    resume_training_state_tag: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed checkpoint tag to restore before this pipeline step."},
    )
    training_state_tag: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed checkpoint tag to save after this pipeline step."},
    )
    keep_all_training_states: bool = field(
        default=False,
        metadata={"help": "Keep all ZeRO training-state tags instead of retaining only the latest tag."},
    )


def _load_single_cache(
    cache: CoIntuitorCache, step: int, num_generations: Optional[int] = None
) -> Tuple[
    List[str],
    List[str],
    List[float],
    Optional[List[List[float]]],
    Optional[List[List[float]]],
    Optional[List[Optional[str]]],
    int,
]:
    """
    Load (prompts, completions, scores, old_logps, ref_logps, finish_reasons, num_generations) from the "*_a" cache slots.
    """
    # We intentionally peek into the cache structure to infer num_generations robustly.
    # `load_generations_a(step, g)` needs the correct g to validate expected_count.
    cache._load_cache()  # noqa: SLF001 (internal; safe for our usage)
    gen_data = (cache._cache_data or {}).get("generations_a")  # noqa: SLF001
    if not isinstance(gen_data, dict) or int(gen_data.get("step", -999999)) != int(step):
        raise ValueError(f"Missing generations_a in cache for step={step}. Run generate_and_score_single.py.")
    completions = gen_data.get("text_responses", None)
    prompts = gen_data.get("prompts", None)
    finish = gen_data.get("finish_reasons", None)
    n_prompts = int(gen_data.get("total_unique_prompts", 0))
    if not isinstance(completions, list) or not isinstance(prompts, list):
        raise ValueError(f"Missing generations_a/prompts in cache for step={step}. Run generate_and_score_single.py.")

    if n_prompts <= 0:
        raise ValueError("Invalid total_unique_prompts in cache.")
    if len(completions) % n_prompts != 0:
        raise ValueError(f"completions length {len(completions)} is not divisible by num_prompts {n_prompts}.")
    inferred_g = len(completions) // n_prompts
    if num_generations is not None and int(num_generations) != int(inferred_g):
        raise ValueError(f"Requested num_generations={num_generations}, but cache implies g={inferred_g}.")

    # scores_a
    ok_s, scores = cache.load_scores_a(step)
    if not ok_s or scores is None or len(scores) != len(completions):
        raise ValueError(f"Missing/invalid scores_a in cache for step={step}.")
    tc_check_generations_block(
        where="single.train.load_cache.generations_a",
        prompts=prompts,
        responses=completions,
        num_generations=int(inferred_g),
        finish_reasons=finish if isinstance(finish, list) else None,
    )
    tc_check_score_vector(
        where="single.train.load_cache.scores_a",
        scores=scores,
        num_prompts=int(n_prompts),
        allowed_per_prompt=[int(inferred_g)],
    )

    # logps_a (optional)
    ok_lp, old_lp, ref_lp = cache.load_logps_a(step)
    if not ok_lp:
        old_lp, ref_lp = None, None
    tc_check_logps_matrix(
        where="single.train.load_cache.old_logps",
        logps=old_lp,
        expected_rows=len(completions) if old_lp is not None else None,
    )
    tc_check_logps_matrix(
        where="single.train.load_cache.ref_logps",
        logps=ref_lp,
        expected_rows=len(completions) if ref_lp is not None else None,
    )

    finish_reasons: Optional[List[Optional[str]]] = None
    if isinstance(finish, list) and len(finish) == len(completions):
        finish_reasons = [None if x is None else str(x) for x in finish]

    return prompts, completions, scores, old_lp, ref_lp, finish_reasons, inferred_g


def _build_training_tensors(
    *,
    tokenizer,
    prompts: List[str],
    completions: List[str],
    scores: List[float],
    num_generations: int,
    reward_mode: str,
    reward_length_norm: str = "response",
    step: int,
    max_prompt_length: Optional[int],
    max_completion_length: Optional[int],
    mask_truncated_completions: bool,
    scale_rewards: bool,
    mixed_reward_alpha: float = 0.5,
    mixed_reward_alpha_mode: str = "fixed",
    mixed_reward_final_normalize: bool = True,
    mixed_reward_alpha_zscore_mu: float = 0.65,
    mixed_reward_alpha_zscore_sigma: float = 0.2,
    mixed_reward_alpha_zscore_lower: float = 0.0,
    mixed_reward_alpha_zscore_upper: float = 1.0,
    mixed_reward_corr_weight_mode: str = "none",
    mixed_reward_corr_weight_threshold: float = 0.1,
    mixed_reward_corr_weight_exponent: float = 0.8,
    mixed_reward_reliability_sc_weight: float = 0.6,
    mixed_reward_reliability_corr_weight: float = 0.4,
    mixed_reward_reliability_use_sample_weight: bool = True,
    mixed_reward_reliability_use_advantage_weight: bool = True,
    mixed_reward_reliability_sample_weight_mu: float = 1.0,
    mixed_reward_reliability_sample_weight_sigma: float = 0.2,
    mixed_reward_reliability_sample_weight_lower: float = 0.5,
    mixed_reward_reliability_sample_weight_upper: float = 1.5,
    mixed_reward_reliability_advantage_weight_mu: float = 0.5,
    mixed_reward_reliability_advantage_weight_sigma: float = 0.2,
    mixed_reward_reliability_advantage_weight_lower: float = 0.2,
    mixed_reward_reliability_advantage_weight_upper: float = 0.8,
    finish_reasons: Optional[List[Optional[str]]],
    old_logps: Optional[List[List[float]]],
    ref_logps: Optional[List[List[float]]],
    mixed_reward_diagnostics: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the dict-of-tensors dataset on CPU, matching the shape expectations of `compute_grpo_loss`.
    """
    num_prompts = len(prompts)
    num_samples = len(completions)
    if num_samples != num_prompts * num_generations:
        raise ValueError(
            f"Expected completions length {num_prompts*num_generations}, got {num_samples} "
            f"(num_prompts={num_prompts}, g={num_generations})."
        )

    # Align behavior with `co_intuitor_train.prepare_training_data_from_cache`:
    # - sanitize completions (prompt echo + leaked chat markers)
    # - append EOS only when not truncated (finish_reason != "length")
    # - cap completion length to max_completion_length
    completions, _changed1 = strip_prompt_echo_from_flattened_responses(
        prompts, completions, num_generations=num_generations
    )
    completions, _changed2 = strip_chat_template_artifacts_from_flattened_responses(
        prompts, completions, num_generations=num_generations
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have pad_token_id or eos_token_id.")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must have eos_token_id to build completion ids/mask (open-r1 semantics).")

    prompt_ids_list: List[torch.Tensor] = []
    prompt_mask_list: List[torch.Tensor] = []
    completion_ids_list: List[torch.Tensor] = []

    # Tokenize prompt once per unique prompt, then replicate for flattened completions.
    prompt_tok = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    prompt_ids_mat = prompt_tok["input_ids"]
    prompt_mask_mat = prompt_tok["attention_mask"]
    if max_prompt_length is not None:
        prompt_ids_mat = prompt_ids_mat[:, -int(max_prompt_length) :]
        prompt_mask_mat = prompt_mask_mat[:, -int(max_prompt_length) :]

    # Completion: tokenize one-by-one to apply EOS append policy based on finish_reason (vLLM semantics).
    for i in range(num_prompts):
        pid = prompt_ids_mat[i]
        pmask = prompt_mask_mat[i]
        for j in range(int(num_generations)):
            idx = i * int(num_generations) + j
            response_text = completions[idx]
            finish_reason = finish_reasons[idx] if (finish_reasons is not None and idx < len(finish_reasons)) else None
            is_truncated = (str(finish_reason) == "length") if finish_reason is not None else None

            resp_inputs = tokenizer(
                text=response_text,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
            )
            cid = resp_inputs["input_ids"][0]
            # Cap completion length to match generation-time max_tokens.
            if max_prompt_length is not None:
                # max_prompt_length already applied to prompt; keep here only for parity with co args signature.
                pass
            if getattr(tokenizer, "model_max_length", None) is not None:
                # no-op; we explicitly cap by max_completion_length when provided
                pass
            if max_completion_length is not None and int(max_completion_length) > 0:
                mcl = int(max_completion_length)
                if cid.size(0) > mcl:
                    cid = cid[:mcl]

            # Append EOS only when completion is not truncated.
            if is_truncated is False:
                cid = append_eos_1d(cid, eos_token_id=int(eos_id))
            elif is_truncated is True:
                pass
            else:
                # Fallback heuristic when finish_reason is unavailable.
                if max_completion_length is None or cid.size(0) < int(max_completion_length):
                    cid = append_eos_1d(cid, eos_token_id=int(eos_id))

            # Ensure completion never exceeds max_completion_length after EOS append.
            if max_completion_length is not None and int(max_completion_length) > 0:
                mcl2 = int(max_completion_length)
                if cid.size(0) > mcl2:
                    cid = cid[:mcl2]

            # Truncate to the first EOS (inclusive), matching co_intuitor_train and
            # build_left_padded_prompt_completion_batch.  Without this, tokens after
            # EOS inflate target_len beyond logits_to_keep_cache, forcing
            # _align_cached_logps to zero-pad and potentially breaking KL computation.
            eos_id_int = int(eos_id)
            comp_list = cid.tolist()
            if eos_id_int in comp_list:
                first_eos = comp_list.index(eos_id_int)
                cid = cid[: first_eos + 1]

            prompt_ids_list.append(pid)
            prompt_mask_list.append(pmask)
            completion_ids_list.append(cid)

    # CRITICAL: prompts must be LEFT-padded to match build_left_padded_prompt_completion_batch
    # (used during logps computation). Right-padding prompts would shift RoPE position_ids,
    # causing training-time logps to diverge from cached old/ref logps.
    prompt_ids_rep = pad(prompt_ids_list, padding_value=int(pad_token_id), padding_side="left")
    prompt_mask_rep = pad(prompt_mask_list, padding_value=0, padding_side="left")
    completion_ids = pad(completion_ids_list, padding_value=int(pad_token_id))

    # Create completion mask based on EOS position.
    is_eos = completion_ids == int(eos_id)
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    seq_idx = torch.arange(is_eos.size(1), device=eos_idx.device).expand(is_eos.size(0), -1)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

    # NOTE: match co-intuitor behavior: do NOT drop/mask truncated completions.
    # Keep the flag for CLI compatibility only.
    _ = bool(mask_truncated_completions)

    # Old/ref logps (optional, keep on CPU).
    # Prepared before reward computation because trajectory reward modes need aligned old_logps.
    old_per_token_logps: Optional[torch.Tensor] = None
    ref_per_token_logps: Optional[torch.Tensor] = None
    if old_logps is not None:
        old_per_token_logps = torch.tensor(old_logps, dtype=torch.float32)
    if ref_logps is not None:
        ref_per_token_logps = torch.tensor(ref_logps, dtype=torch.float32)

    target_len = int(completion_ids.size(1))

    def _align_cached_logps(lp: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if lp is None:
            return None
        if lp.dim() != 2:
            return lp
        if lp.size(1) == target_len:
            return lp
        if lp.size(1) > target_len:
            return lp[:, :target_len]
        pad2 = lp.new_zeros((lp.size(0), target_len - lp.size(1)))
        return torch.cat([lp, pad2], dim=1)

    old_per_token_logps = _align_cached_logps(old_per_token_logps)
    ref_per_token_logps = _align_cached_logps(ref_per_token_logps)

    # Pre-compute per-response and per-rollout lengths for reward normalization.
    completion_mask_f = completion_mask.float()
    response_lengths = completion_mask_f.sum(dim=1)  # (num_samples,)
    response_lengths_grouped = response_lengths.view(num_prompts, num_generations)  # (P, G)
    rollout_total_lengths = response_lengths_grouped.sum(dim=1, keepdim=True)  # (P, 1)

    # Rewards per prompt group.
    scores_t = torch.tensor(scores, dtype=torch.float32).view(num_prompts, num_generations)

    if reward_mode in ("intuitor", "trajectory"):
        # Unified certainty-based reward: compute raw (unnormalized) signal,
        # then divide by the denominator selected via reward_length_norm.
        if reward_mode == "intuitor":
            # Cached SCE scores are pre-normalized (SCE_sum / response_len).
            # Recover raw sums so we can re-normalize uniformly.
            raw_signal = scores_t * response_lengths_grouped
        else:
            if old_per_token_logps is None:
                raise ValueError(
                    "trajectory reward requires old_logps. "
                    "Ensure objective_mode='gspo' so that force_compute_old_logps is set."
                )
            raw_signal = (old_per_token_logps * completion_mask_f).sum(dim=1).view(num_prompts, num_generations)

        if reward_length_norm == "rollout":
            rewards_t = raw_signal / (rollout_total_lengths + 1e-8)
        else:
            rewards_t = raw_signal / (response_lengths_grouped + 1e-8)

    elif reward_mode == "truth":
        rewards_t = scores_t

    elif reward_mode == "mv_gated_intuitor":
        # R = R(MV) * sigmoid(R(intuitor))
        # MV is a hard binary gate; sigmoid bounds the SCE score to (0, 1).
        mv_rows: List[List[float]] = []
        for i in range(num_prompts):
            rng = random.Random((int(step) + 1) * 1_000_000 + int(i))
            start = i * num_generations
            end = (i + 1) * num_generations
            resp_i = completions[start:end]
            r_i, _vote = rewards_from_mv_pseudolabel(resp_i, rng=rng)
            mv_rows.append(r_i)
        mv_t = torch.tensor(mv_rows, dtype=torch.float32)  # (P, G) binary 0/1
        sig_sce = torch.sigmoid(scores_t)
        rewards_t = mv_t * sig_sce

    elif reward_mode in ("wmv", "mv"):
        rewards_rows: List[List[float]] = []
        if reward_mode == "wmv":
            weights_t = z_score_normalize_tensor(scores_t, mu=5.0, sigma=1.0, clip_threshold=3.0, dim=1)
        for i in range(num_prompts):
            rng = random.Random((int(step) + 1) * 1_000_000 + int(i))
            start = i * num_generations
            end = (i + 1) * num_generations
            resp_i = completions[start:end]
            if reward_mode == "wmv":
                w_i = weights_t[i].tolist()
                r_i, _vote = rewards_from_wmv_pseudolabel(resp_i, w_i, rng=rng)
            else:
                r_i, _vote = rewards_from_mv_pseudolabel(resp_i, rng=rng)
            rewards_rows.append(r_i)
        rewards_t = torch.tensor(rewards_rows, dtype=torch.float32)

    elif reward_mode == "mixed_reward":
        # MV rewards: binary 0/1 from majority voting pseudo-labels.
        mv_rows: List[List[float]] = []
        mv_votes: List[Dict[str, Any]] = []
        mv_extracted_answers: List[List[Optional[str]]] = []
        for i in range(num_prompts):
            rng = random.Random((int(step) + 1) * 1_000_000 + int(i))
            start = i * num_generations
            end = (i + 1) * num_generations
            resp_i = completions[start:end]
            r_i, vote = rewards_from_mv_pseudolabel(resp_i, rng=rng)
            mv_rows.append(r_i)
            if mixed_reward_diagnostics is not None:
                mv_votes.append(
                    {
                        "selected_answer": vote.selected_answer,
                        "answer_clusters": {
                            str(answer): float(weight)
                            for answer, weight in vote.answer_clusters.items()
                        },
                    }
                )
                mv_extracted_answers.append(extract_answers(resp_i))
        mv_rewards = torch.tensor(mv_rows, dtype=torch.float32)  # (P, G)

        # Intuitor rewards: SCE scores (already per-response normalized).
        intuitor_rewards = scores_t  # (P, G)

        from src.training.co_intuitor_train import compute_mixed_reward_advantages

        if mixed_reward_diagnostics is not None:
            mixed_reward_diagnostics["mv_votes"] = mv_votes
            mixed_reward_diagnostics["mv_extracted_answers"] = mv_extracted_answers

        advantages = compute_mixed_reward_advantages(
            mv_rewards,
            intuitor_rewards,
            alpha=mixed_reward_alpha,
            alpha_mode=mixed_reward_alpha_mode,
            alpha_zscore_mu=mixed_reward_alpha_zscore_mu,
            alpha_zscore_sigma=mixed_reward_alpha_zscore_sigma,
            alpha_zscore_lower=mixed_reward_alpha_zscore_lower,
            alpha_zscore_upper=mixed_reward_alpha_zscore_upper,
            corr_weight_mode=mixed_reward_corr_weight_mode,
            corr_weight_threshold=mixed_reward_corr_weight_threshold,
            corr_weight_exponent=mixed_reward_corr_weight_exponent,
            reliability_sc_weight=mixed_reward_reliability_sc_weight,
            reliability_corr_weight=mixed_reward_reliability_corr_weight,
            reliability_use_sample_weight=mixed_reward_reliability_use_sample_weight,
            reliability_use_advantage_weight=mixed_reward_reliability_use_advantage_weight,
            reliability_sample_weight_mu=mixed_reward_reliability_sample_weight_mu,
            reliability_sample_weight_sigma=mixed_reward_reliability_sample_weight_sigma,
            reliability_sample_weight_lower=mixed_reward_reliability_sample_weight_lower,
            reliability_sample_weight_upper=mixed_reward_reliability_sample_weight_upper,
            reliability_advantage_weight_mu=mixed_reward_reliability_advantage_weight_mu,
            reliability_advantage_weight_sigma=mixed_reward_reliability_advantage_weight_sigma,
            reliability_advantage_weight_lower=mixed_reward_reliability_advantage_weight_lower,
            reliability_advantage_weight_upper=mixed_reward_reliability_advantage_weight_upper,
            final_normalize=mixed_reward_final_normalize,
            diagnostics=mixed_reward_diagnostics,
        ).view(-1)

    else:
        raise ValueError(
            f"Unknown reward_mode={reward_mode!r}. "
            "Must be 'intuitor', 'trajectory', 'truth', 'wmv', 'mv', "
            "'mv_gated_intuitor', or 'mixed_reward'."
        )

    if reward_mode != "mixed_reward":
        # Flatten rewards and compute group-wise advantages.
        rewards_flat = rewards_t.view(-1)
        rewards_grouped = rewards_flat.view(num_prompts, num_generations)
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        # Match open-r1-intuitor: use torch default std (unbiased=True) for group-wise normalization.
        std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = (rewards_grouped - mean).view(-1)
        if scale_rewards:
            advantages = advantages / (std.view(-1).repeat_interleave(num_generations) + 1e-6)

    out = {
        "prompt_ids": prompt_ids_rep,
        "prompt_mask": prompt_mask_rep,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "old_per_token_logps": old_per_token_logps,
        "ref_per_token_logps": ref_per_token_logps,
    }

    # Optional invariants (enable with CO_ASSERT_INVARIANTS=1).
    assert_prompt_completion_shapes(
        prompt_ids=out["prompt_ids"],
        prompt_mask=out["prompt_mask"],
        completion_ids=out["completion_ids"],
        completion_mask=out["completion_mask"],
        where="src.single_intuitor_train._build_training_tensors",
    )
    return out


def main(script_args: SingleTrainScriptArguments, training_args: CoIntuitorConfig, model_args: ModelConfig):
    # CRITICAL: Do NOT call accelerate.utils.set_seed here (see co_intuitor_train).
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Reduce noisy logs from DeepSpeed/Accelerate/NCCL wrappers so our logs remain readable.
    noisy_modules_to_error = [
        "deepspeed",
        "accelerate",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch._dynamo",
        "transformers.generation",
        "transformers.modeling_utils",
    ]
    for name in noisy_modules_to_error:
        logging.getLogger(name).setLevel(logging.ERROR)

    # DeepSpeed also has its own internal logger. Silence it to avoid config spam like:
    #   [INFO] [config.py:...:print] ...
    # This matches `src/co_intuitor_train.py`.
    try:
        import deepspeed

        if hasattr(deepspeed, "utils") and hasattr(deepspeed.utils, "logging"):
            ds_logger = getattr(deepspeed.utils.logging, "logger", None)
            if ds_logger is not None:
                ds_logger.setLevel(logging.ERROR)
    except Exception:
        pass

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    # Print TRL GRPO clip settings explicitly (avoid guessing defaults).
    try:
        logger.info(
            f"GRPO clip config: epsilon={getattr(training_args, 'epsilon', None)} "
            f"epsilon_high={getattr(training_args, 'epsilon_high', None)}"
        )
    except Exception:
        pass

    # Resolve cache_dir and policy model path.
    cache_dir = str(script_args.cache_dir) if script_args.cache_dir else str(training_args.output_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cache = CoIntuitorCache(os.path.join(cache_dir, "cache.json"))

    policy_model_path = str(script_args.policy_model_path)
    if os.path.exists(training_args.output_dir):
        latest_checkpoint = get_last_checkpoint(training_args.output_dir)
        if latest_checkpoint is not None:
            policy_model_path = latest_checkpoint
            logger.info(f"Found latest checkpoint for policy model: {policy_model_path}")
    logger.info(f"Policy model path: {policy_model_path}")

    # Tokenizer/chat_template should ideally remain stable across steps. For parity with co_intuitor_train:
    # - default: load tokenizer from the same path as the policy model we train (often a checkpoint dir)
    # - override: allow --tokenizer_path to point to the initial/base model directory
    tokenizer_path = str(script_args.tokenizer_path) if script_args.tokenizer_path else str(policy_model_path)

    # Select step (default: latest complete for single cache).
    step = int(script_args.step)
    if step < 0:
        cache._load_cache()  # noqa: SLF001
        gen = (cache._cache_data or {}).get("generations_a")  # noqa: SLF001
        sc = (cache._cache_data or {}).get("scores_a")  # noqa: SLF001
        if not isinstance(gen, dict) or not isinstance(sc, dict):
            raise ValueError("No cache data found for single-model training. Run generate_and_score_single.py first.")
        try:
            gen_step = int(gen.get("step"))
            sc_step = int(sc.get("step"))
        except Exception as e:
            raise ValueError(f"Invalid cache step metadata: {e}")
        if gen_step != sc_step:
            raise ValueError(f"Cache incomplete: generations_a.step={gen_step} != scores_a.step={sc_step}")
        step = gen_step

    prompts, completions, scores, old_lp, ref_lp, finish, g = _load_single_cache(cache, step, num_generations=None)
    logger.info(
        f"Loaded cache step={step}: prompts={len(prompts)}, completions={len(completions)}, g={g}, "
        f"finish_reasons={'yes' if finish is not None else 'no'}"
    )

    alpha_mode = str(
        getattr(training_args, "mixed_reward_alpha_mode", "fixed")
    ).strip().lower()
    require_mixed_reward_diagnostics = _requires_mixed_reward_diagnostics(
        script_args.reward_mode,
        alpha_mode,
    )
    sampled_indices: List[int] = []
    if require_mixed_reward_diagnostics:
        checkpoint_info = cache.load_checkpoint_info(step)
        raw_sampled_indices = (
            checkpoint_info.get("sampled_indices")
            if isinstance(checkpoint_info, dict)
            else None
        )
        if not isinstance(raw_sampled_indices, list) or len(raw_sampled_indices) != len(prompts):
            raise RuntimeError(
                "Reliability diagnostics require checkpoint_info.sampled_indices "
                f"aligned with prompts for step {step}: "
                f"indices={len(raw_sampled_indices) if isinstance(raw_sampled_indices, list) else None}, "
                f"prompts={len(prompts)}"
            )
        sampled_indices = [int(value) for value in raw_sampled_indices]

    # DAPO dynamic sampling: filter out prompt groups where all scores are identical.
    if script_args.filter_groups:
        import numpy as np

        num_prompts_before = len(prompts)
        scores_arr = np.array(scores).reshape(num_prompts_before, g)
        group_std = scores_arr.std(axis=1)
        keep_mask = group_std > 0.0  # keep groups with mixed scores

        kept_indices = [i for i, keep in enumerate(keep_mask) if keep]
        n_kept = len(kept_indices)
        n_filtered = num_prompts_before - n_kept
        logger.info(
            f"Dynamic sampling (filter_groups): {num_prompts_before} groups -> "
            f"{n_kept} kept, {n_filtered} filtered (all-same-score)"
        )

        if n_kept == 0:
            raise RuntimeError(
                "filter_groups: ALL prompt groups were filtered out (all groups have uniform scores). "
                "Increase problems_per_step or check dataset difficulty."
            )

        def _select_groups(flat_list, indices, group_size):
            out = []
            for idx in indices:
                start = idx * group_size
                out.extend(flat_list[start : start + group_size])
            return out

        prompts = [prompts[i] for i in kept_indices]
        if require_mixed_reward_diagnostics:
            sampled_indices = [sampled_indices[i] for i in kept_indices]
        completions = _select_groups(completions, kept_indices, g)
        scores = _select_groups(scores, kept_indices, g)
        if old_lp is not None:
            old_lp = _select_groups(old_lp, kept_indices, g)
        if ref_lp is not None:
            ref_lp = _select_groups(ref_lp, kept_indices, g)
        if finish is not None:
            finish = _select_groups(finish, kept_indices, g)

    # Load tokenizer/model (policy)
    tokenizer = get_tokenizer(tokenizer_path, training_args)
    model = get_model(policy_model_path, training_args, model_args)

    objective_mode = getattr(training_args, "objective_mode", "grpo")
    is_length_norm = getattr(training_args, "is_length_norm", "mean")

    # Resolve self-update epsilon: prefer dedicated fields, fall back to global epsilon.
    _global_eps = float(getattr(training_args, "epsilon", 0.1))
    _global_eps_high = getattr(training_args, "epsilon_high", None)
    _self_eps_low = getattr(training_args, "self_update_epsilon_low", None)
    _self_eps_high = getattr(training_args, "self_update_epsilon_high", None)
    _effective_self_eps_low = float(_self_eps_low) if _self_eps_low is not None else _global_eps
    _effective_self_eps_high = float(_self_eps_high) if _self_eps_high is not None else _global_eps_high

    def compute_loss_fn(model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("compute_loss_fn does not support returning outputs")
        return compute_policy_loss(
            model=model,
            inputs=inputs,
            objective_mode=objective_mode,
            temperature=getattr(training_args, "temperature", 1.0),
            epsilon_low=_effective_self_eps_low,
            epsilon_high=_effective_self_eps_high,
            beta=getattr(training_args, "beta", 0.0),
            loss_type=getattr(training_args, "loss_type", "grpo"),
            max_completion_length=getattr(training_args, "max_completion_length", 512),
            num_iterations=getattr(training_args, "num_iterations", 1),
            length_norm=is_length_norm,
        )

    trainer = SingleModelTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_loss=compute_loss_fn,
        metadata_source_dir=tokenizer_path,
    )

    # Seed AFTER accelerator is created (matches intuitor style).
    from accelerate.utils import set_seed as accelerate_set_seed

    accelerate_set_seed(training_args.seed, device_specific=True)

    # Build training tensors on CPU.
    mixed_reward_diagnostics: Optional[MutableMapping[str, Any]] = (
        {} if require_mixed_reward_diagnostics else None
    )
    training_data = _build_training_tensors(
        tokenizer=tokenizer,
        prompts=prompts,
        completions=completions,
        scores=scores,
        num_generations=g,
        reward_mode=script_args.reward_mode,
        reward_length_norm=getattr(script_args, "reward_length_norm", "response"),
        step=step,
        max_prompt_length=getattr(training_args, "max_prompt_length", None),
        max_completion_length=getattr(training_args, "max_completion_length", None),
        mask_truncated_completions=getattr(training_args, "mask_truncated_completions", False),
        scale_rewards=getattr(training_args, "scale_rewards", True),
        finish_reasons=finish,
        old_logps=old_lp,
        ref_logps=ref_lp,
        mixed_reward_alpha=getattr(training_args, "mixed_reward_alpha", 0.5),
        mixed_reward_alpha_mode=getattr(training_args, "mixed_reward_alpha_mode", "fixed"),
        mixed_reward_final_normalize=getattr(
            training_args, "mixed_reward_final_normalize", True
        ),
        mixed_reward_alpha_zscore_mu=getattr(training_args, "mixed_reward_alpha_zscore_mu", 0.65),
        mixed_reward_alpha_zscore_sigma=getattr(training_args, "mixed_reward_alpha_zscore_sigma", 0.2),
        mixed_reward_alpha_zscore_lower=getattr(training_args, "mixed_reward_alpha_zscore_lower", 0.0),
        mixed_reward_alpha_zscore_upper=getattr(training_args, "mixed_reward_alpha_zscore_upper", 1.0),
        mixed_reward_corr_weight_mode=getattr(training_args, "mixed_reward_corr_weight_mode", "none"),
        mixed_reward_corr_weight_threshold=getattr(training_args, "mixed_reward_corr_weight_threshold", 0.1),
        mixed_reward_corr_weight_exponent=getattr(training_args, "mixed_reward_corr_weight_exponent", 0.8),
        mixed_reward_reliability_sc_weight=getattr(
            training_args, "mixed_reward_reliability_sc_weight", 0.6
        ),
        mixed_reward_reliability_corr_weight=getattr(
            training_args, "mixed_reward_reliability_corr_weight", 0.4
        ),
        mixed_reward_reliability_use_sample_weight=getattr(
            training_args, "mixed_reward_reliability_use_sample_weight", True
        ),
        mixed_reward_reliability_use_advantage_weight=getattr(
            training_args, "mixed_reward_reliability_use_advantage_weight", True
        ),
        mixed_reward_reliability_sample_weight_mu=getattr(
            training_args, "mixed_reward_reliability_sample_weight_mu", 1.0
        ),
        mixed_reward_reliability_sample_weight_sigma=getattr(
            training_args, "mixed_reward_reliability_sample_weight_sigma", 0.2
        ),
        mixed_reward_reliability_sample_weight_lower=getattr(
            training_args, "mixed_reward_reliability_sample_weight_lower", 0.5
        ),
        mixed_reward_reliability_sample_weight_upper=getattr(
            training_args, "mixed_reward_reliability_sample_weight_upper", 1.5
        ),
        mixed_reward_reliability_advantage_weight_mu=getattr(
            training_args, "mixed_reward_reliability_advantage_weight_mu", 0.5
        ),
        mixed_reward_reliability_advantage_weight_sigma=getattr(
            training_args, "mixed_reward_reliability_advantage_weight_sigma", 0.2
        ),
        mixed_reward_reliability_advantage_weight_lower=getattr(
            training_args, "mixed_reward_reliability_advantage_weight_lower", 0.2
        ),
        mixed_reward_reliability_advantage_weight_upper=getattr(
            training_args, "mixed_reward_reliability_advantage_weight_upper", 0.8
        ),
        mixed_reward_diagnostics=mixed_reward_diagnostics,
    )
    require_old_lp = bool(objective_mode == "gspo")
    require_ref_lp = bool(objective_mode == "gspo" and float(getattr(training_args, "beta", 0.0)) != 0.0)
    tc_check_training_batch(
        where="single.train.training_batch",
        prompt_ids=training_data["prompt_ids"],
        prompt_mask=training_data["prompt_mask"],
        completion_ids=training_data["completion_ids"],
        completion_mask=training_data["completion_mask"],
        advantages=training_data["advantages"],
        old_per_token_logps=training_data.get("old_per_token_logps"),
        ref_per_token_logps=training_data.get("ref_per_token_logps"),
        require_old_logps=require_old_lp,
        require_ref_logps=require_ref_lp,
    )

    if mixed_reward_diagnostics is not None and alpha_mode in {"prism", "ettrl"}:
        mode_parameters = {
            "reward_mode": "mixed_reward",
            "alpha_mode": alpha_mode,
            "alpha": float(getattr(training_args, "mixed_reward_alpha", 0.5)),
            "corr_weight_mode": str(
                getattr(training_args, "mixed_reward_corr_weight_mode", "none")
            ),
            "final_normalize": bool(
                getattr(training_args, "mixed_reward_final_normalize", True)
            ),
            "ettrl_all_zero_group_fallback": "unit_weight",
        }
        diagnostics_payload = _build_prism_ettrl_diagnostics_payload(
            step=step,
            prompts=list(prompts),
            sampled_indices=sampled_indices,
            num_generations=g,
            diagnostics=mixed_reward_diagnostics,
            training_advantages=training_data["advantages"],
            parameters=mode_parameters,
        )
        diagnostics_path = _write_prism_ettrl_diagnostics_if_main(
            is_main_process=bool(trainer.accelerator.is_main_process),
            base_dir=cache_dir,
            step=step,
            alpha_mode=alpha_mode,
            payload=diagnostics_payload,
        )
        if diagnostics_path is not None:
            logger.info("Saved required %s diagnostics to %s", alpha_mode, diagnostics_path)
        trainer.accelerator.wait_for_everyone()

    if mixed_reward_diagnostics is not None and alpha_mode == "reliability_sc_corr":
        reliability_parameters = {
            "reward_mode": "mixed_reward",
            "alpha_mode": "reliability_sc_corr",
            "alpha": float(getattr(training_args, "mixed_reward_alpha", 0.5)),
            "sc_weight": float(
                getattr(training_args, "mixed_reward_reliability_sc_weight", 0.6)
            ),
            "corr_weight": float(
                getattr(training_args, "mixed_reward_reliability_corr_weight", 0.4)
            ),
            "use_sample_weight": bool(
                getattr(
                    training_args,
                    "mixed_reward_reliability_use_sample_weight",
                    True,
                )
            ),
            "use_advantage_weight": bool(
                getattr(
                    training_args,
                    "mixed_reward_reliability_use_advantage_weight",
                    True,
                )
            ),
            "sample_weight": {
                "mu": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_sample_weight_mu",
                        1.0,
                    )
                ),
                "sigma": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_sample_weight_sigma",
                        0.2,
                    )
                ),
                "lower": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_sample_weight_lower",
                        0.5,
                    )
                ),
                "upper": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_sample_weight_upper",
                        1.5,
                    )
                ),
            },
            "advantage_weight": {
                "mu": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_advantage_weight_mu",
                        0.5,
                    )
                ),
                "sigma": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_advantage_weight_sigma",
                        0.2,
                    )
                ),
                "lower": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_advantage_weight_lower",
                        0.2,
                    )
                ),
                "upper": float(
                    getattr(
                        training_args,
                        "mixed_reward_reliability_advantage_weight_upper",
                        0.8,
                    )
                ),
            },
            "final_normalize": bool(
                getattr(training_args, "mixed_reward_final_normalize", True)
            ),
        }
        diagnostics_payload = _build_reliability_diagnostics_payload(
            step=step,
            prompts=list(prompts),
            sampled_indices=sampled_indices,
            num_generations=g,
            diagnostics=mixed_reward_diagnostics,
            training_advantages=training_data["advantages"],
            parameters=reliability_parameters,
        )
        diagnostics_path = _write_reliability_diagnostics_if_main(
            is_main_process=bool(trainer.accelerator.is_main_process),
            base_dir=cache_dir,
            step=step,
            payload=diagnostics_payload,
        )
        if diagnostics_path is not None:
            logger.info("Saved required Reliability diagnostics to %s", diagnostics_path)
        trainer.accelerator.wait_for_everyone()

    # Debug dump: fully-assembled samples (prompt+template+completion+padding).
    _co_dump_single_prepare_full_samples(
        base_dir=str(cache_dir),
        step=int(step),
        reward_mode=str(script_args.reward_mode),
        tokenizer=tokenizer,
        prompts=list(prompts),
        completions=list(completions),
        num_generations=int(g),
        prompt_ids=training_data["prompt_ids"],
        prompt_mask=training_data["prompt_mask"],
        completion_ids=training_data["completion_ids"],
        completion_mask=training_data["completion_mask"],
        max_items=10,
    )

    # Avoid empty_cache by default (can increase fragmentation under ZeRO-3).
    # Enable explicitly if you need it:
    #   export CO_EMPTY_CACHE_AFTER_BUILD_TENSORS=1
    if torch.cuda.is_available():
        v = os.environ.get("CO_EMPTY_CACHE_AFTER_BUILD_TENSORS", "0").strip().lower()
        if v not in ("", "0", "false", "no", "off", "none"):
            torch.cuda.empty_cache()

    # Build dataloader (same as co_intuitor_train).
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

    total_pipeline_steps = int(getattr(training_args, "total_pipeline_steps", 0) or 0)
    current_pipeline_step = int(getattr(training_args, "current_pipeline_step", 0) or 0)
    update_plan = compute_pipeline_update_plan(
        num_samples=len(train_dataset),
        per_device_batch_size=per_device_bs,
        world_size=int(trainer.accelerator.num_processes),
        gradient_accumulation_steps=int(training_args.gradient_accumulation_steps),
        total_pipeline_steps=total_pipeline_steps,
        current_pipeline_step=current_pipeline_step,
    )
    resume_training_state = bool(script_args.resume_training_state_tag)
    scheduler_completed_steps = 0 if resume_training_state else update_plan["completed_optimizer_updates"]
    trainer.create_optimizer_and_scheduler(
        update_plan["updates_per_pipeline_step"],
        train_dataloader=train_dataloader,
        scheduler_total_steps=update_plan["total_optimizer_updates"],
        scheduler_completed_steps=scheduler_completed_steps,
        scheduler_backend="deepspeed_native" if trainer.is_deepspeed_enabled else "legacy",
    )

    loaded_client_state: Dict[str, Any] = {}
    if resume_training_state:
        if not script_args.training_state_dir:
            raise ValueError("--resume_training_state_tag requires --training_state_dir")
        loaded_client_state = trainer.load_training_state(
            str(script_args.training_state_dir),
            str(script_args.resume_training_state_tag),
        )
        saved_updates = int(loaded_client_state.get("completed_optimizer_updates", -1))
        expected_updates = update_plan["completed_optimizer_updates"]
        if saved_updates != expected_updates:
            raise ValueError(
                "DeepSpeed training-state update count does not match this pipeline step: "
                f"saved={saved_updates}, expected={expected_updates}, "
                f"tag={script_args.resume_training_state_tag}"
            )
        saved_pipeline_step = int(loaded_client_state.get("pipeline_step", -1))
        if saved_pipeline_step != current_pipeline_step - 1:
            raise ValueError(
                "DeepSpeed training state belongs to the wrong pipeline step: "
                f"saved={saved_pipeline_step}, expected={current_pipeline_step - 1}"
            )
        saved_plan = loaded_client_state.get("update_plan")
        for key in ("global_micro_batch_size", "gradient_accumulation_steps", "updates_per_pipeline_step"):
            if not isinstance(saved_plan, dict) or int(saved_plan.get(key, -1)) != update_plan[key]:
                raise ValueError(
                    "DeepSpeed training-state protocol does not match the current run: "
                    f"field={key}, saved={saved_plan.get(key) if isinstance(saved_plan, dict) else None}, "
                    f"expected={update_plan[key]}"
                )
        if not isinstance(saved_plan, dict):
            raise ValueError("DeepSpeed training-state protocol is missing its update_plan metadata")
        saved_total_updates = int(saved_plan.get("total_optimizer_updates", -1))
        if saved_total_updates > update_plan["total_optimizer_updates"]:
            raise ValueError(
                "The resumed run cannot shorten its optimizer-update horizon: "
                f"saved={saved_total_updates}, current={update_plan['total_optimizer_updates']}"
            )
        if saved_total_updates < update_plan["total_optimizer_updates"]:
            logger.warning(
                "Extending total pipeline steps during resume: saved optimizer horizon=%d, current=%d. "
                "The restored AdamW/scheduler position will be preserved.",
                saved_total_updates,
                update_plan["total_optimizer_updates"],
            )
        scheduler_state = trainer.scheduler_runtime_state()
        saved_scheduler_type = loaded_client_state.get("scheduler_type")
        current_scheduler_type = trainer._scheduler_name(
            getattr(training_args, "lr_scheduler_type", "cosine")
        )
        if saved_scheduler_type is not None and saved_scheduler_type != current_scheduler_type:
            raise ValueError(
                "DeepSpeed training-state scheduler type does not match the current run: "
                f"saved={saved_scheduler_type}, current={current_scheduler_type}"
            )
        saved_scheduler_backend = loaded_client_state.get("scheduler_backend")
        if saved_scheduler_backend is not None and saved_scheduler_backend != scheduler_state["backend"]:
            raise ValueError(
                "DeepSpeed training-state scheduler backend does not match the current run: "
                f"saved={saved_scheduler_backend}, current={scheduler_state['backend']}"
            )
        scheduler_updates = scheduler_state.get("scheduler_completed_optimizer_updates")
        if scheduler_updates != update_plan["completed_optimizer_updates"]:
            raise ValueError(
                "DeepSpeed scheduler position does not match the current pipeline step: "
                f"scheduler_updates={scheduler_updates}, "
                f"expected={update_plan['completed_optimizer_updates']}"
            )
    elif current_pipeline_step > 1:
        logger.warning(
            "No DeepSpeed training state was supplied for pipeline step %d. AdamW moments are reset; "
            "the LR scheduler was reconstructed at optimizer update %d.",
            current_pipeline_step,
            update_plan["completed_optimizer_updates"],
        )

    # Train exactly one pass over this cached batch (micro-step loop).
    tr_loss = torch.tensor(0.0, device=trainer.accelerator.device)
    accum = max(1, int(getattr(training_args, "gradient_accumulation_steps", 1)))

    from tqdm.auto import tqdm

    effective_dataloader = getattr(trainer, "train_dataloader", train_dataloader)
    try:
        total_micro = len(effective_dataloader)
    except TypeError:
        total_micro = None
    if total_micro != update_plan["micro_steps_per_rank"]:
        raise RuntimeError(
            "Prepared dataloader length does not match the computed distributed micro-step plan: "
            f"actual={total_micro}, expected={update_plan['micro_steps_per_rank']}, plan={update_plan}"
        )

    logger.info(
        "Training update plan: samples=%d, global_micro_batch=%d, micro_steps_per_rank=%d, "
        "GAS=%d, updates_this_step=%d, completed_updates=%d, total_updates=%d",
        update_plan["num_samples"],
        update_plan["global_micro_batch_size"],
        update_plan["micro_steps_per_rank"],
        update_plan["gradient_accumulation_steps"],
        update_plan["updates_per_pipeline_step"],
        update_plan["completed_optimizer_updates"],
        update_plan["total_optimizer_updates"],
    )

    pbar = None
    if trainer.accelerator.is_main_process:
        pbar = tqdm(
            total=total_micro,
            desc=f"Training (micro-steps, accum={accum})",
            unit="micro",
            dynamic_ncols=True,
            leave=True,
            position=0,
        )

    initial_ds_global_steps = (
        int(getattr(trainer.model, "global_steps", 0) or 0) if trainer.is_deepspeed_enabled else None
    )
    actual_optimizer_updates = 0
    for _micro_step, batch in enumerate(effective_dataloader, start=1):
        if trainer.is_deepspeed_enabled:
            loss = trainer.training_step(batch)
            tr_loss = tr_loss + loss
            actual_optimizer_updates += int(trainer.optimizer_step())
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=float(loss.detach().item()))
        else:
            with trainer.accelerator.accumulate(trainer.model):
                loss = trainer.training_step(batch)
                tr_loss = tr_loss + loss
                if trainer.accelerator.sync_gradients:
                    actual_optimizer_updates += int(trainer.optimizer_step())
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=float(loss.detach().item()))

    if pbar is not None:
        pbar.refresh()
        pbar.close()
        print("")

    expected_optimizer_updates = update_plan["updates_per_pipeline_step"]
    if actual_optimizer_updates != expected_optimizer_updates:
        raise RuntimeError(
            "Optimizer update count mismatch: "
            f"actual={actual_optimizer_updates}, expected={expected_optimizer_updates}, plan={update_plan}"
        )
    if trainer.is_deepspeed_enabled:
        final_ds_global_steps = int(getattr(trainer.model, "global_steps", 0) or 0)
        if final_ds_global_steps - int(initial_ds_global_steps or 0) != expected_optimizer_updates:
            raise RuntimeError(
                "DeepSpeed global_steps mismatch: "
                f"before={initial_ds_global_steps}, after={final_ds_global_steps}, "
                f"expected_delta={expected_optimizer_updates}"
            )

    avg_loss = tr_loss.item() / max(1, int(total_micro or 0))
    logger.info(f"Step {step}, Loss: {avg_loss:.6f}")

    completed_optimizer_updates = (
        update_plan["completed_optimizer_updates"] + update_plan["updates_per_pipeline_step"]
    )
    scheduler_state = trainer.scheduler_runtime_state()
    if scheduler_state.get("effective_optimizer_updates") != completed_optimizer_updates:
        raise RuntimeError(
            "Runtime optimizer update count does not match the completed pipeline plan: "
            f"runtime={scheduler_state.get('effective_optimizer_updates')}, "
            f"expected={completed_optimizer_updates}"
        )
    if scheduler_state.get("scheduler_completed_optimizer_updates") != completed_optimizer_updates:
        raise RuntimeError(
            "Runtime scheduler update count does not match the completed pipeline plan: "
            f"scheduler={scheduler_state.get('scheduler_completed_optimizer_updates')}, "
            f"expected={completed_optimizer_updates}"
        )

    # Do not leave a model directory that looks complete when runtime-state
    # validation has already failed. The pipeline uses runtime_manifest.json
    # and the DeepSpeed state tag as its completion contract.
    checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if trainer.accelerator.is_main_process:
        logger.info("Saving checkpoint...")
    trainer.save_model(checkpoint_dir)

    if script_args.training_state_dir:
        if not trainer.is_deepspeed_enabled:
            raise RuntimeError("--training_state_dir is enabled, but this run is not using DeepSpeed.")
        state_tag = str(script_args.training_state_tag or f"checkpoint-{step}")
        client_state = {
            "protocol_version": 1,
            "pipeline_step": int(step),
            "completed_optimizer_updates": completed_optimizer_updates,
            "update_plan": update_plan,
            "scheduler_backend": scheduler_state["backend"],
            "scheduler_type": trainer._scheduler_name(getattr(training_args, "lr_scheduler_type", "cosine")),
            "scheduler_total_optimizer_updates": scheduler_state["total_optimizer_updates"],
            "scheduler_warmup_optimizer_updates": scheduler_state["warmup_optimizer_updates"],
            "scheduler_completed_optimizer_updates": scheduler_state[
                "scheduler_completed_optimizer_updates"
            ],
        }
        trainer.save_training_state(str(script_args.training_state_dir), state_tag, client_state)
        if not script_args.keep_all_training_states:
            trainer.accelerator.wait_for_everyone()
            if trainer.accelerator.is_main_process:
                import shutil

                state_root = str(script_args.training_state_dir)
                for name in os.listdir(state_root):
                    path = os.path.join(state_root, name)
                    if name.startswith("checkpoint-") and name != state_tag and os.path.isdir(path):
                        shutil.rmtree(path)
            trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        import accelerate
        import deepspeed

        runtime_manifest = {
            "protocol_version": 1,
            "pipeline_step": int(step),
            "update_plan": update_plan,
            "actual_optimizer_updates": actual_optimizer_updates,
            "completed_optimizer_updates": completed_optimizer_updates,
            "deepspeed_enabled": bool(trainer.is_deepspeed_enabled),
            "gradient_clipping": float(getattr(training_args, "max_grad_norm", 0.0) or 0.0),
            "optimizer_state_persisted": bool(script_args.training_state_dir),
            "scheduler": scheduler_state,
            "resumed_training_state_tag": script_args.resume_training_state_tag,
            "saved_training_state_tag": (
                script_args.training_state_tag or f"checkpoint-{step}"
                if script_args.training_state_dir
                else None
            ),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
                "deepspeed": deepspeed.__version__,
            },
        }
        manifest_path = os.path.join(checkpoint_dir, "runtime_manifest.json")
        manifest_tmp_path = f"{manifest_path}.tmp"
        try:
            with open(manifest_tmp_path, "w", encoding="utf-8") as handle:
                json.dump(runtime_manifest, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(manifest_tmp_path, manifest_path)
        except Exception as exc:
            try:
                os.remove(manifest_tmp_path)
            except FileNotFoundError:
                pass
            raise RuntimeError(f"Failed to write runtime manifest: {manifest_path}") from exc
    logger.info(f"Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    parser = TrlParser((SingleTrainScriptArguments, CoIntuitorConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
