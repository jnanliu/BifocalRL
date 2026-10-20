"""Inputs and outputs of a mixed-reward strategy."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch

from .config import MixedRewardConfig


@dataclass(frozen=True)
class MixedRewardContext:
    """A full cached rollout batch; rows are queries, columns are responses.

    metadata can carry answer IDs or other future method-specific information.
    Strategies return unnormalized advantages; core.py owns final normalization.
    """

    mv_rewards: torch.Tensor
    intuitor_rewards: torch.Tensor
    adv_mv: torch.Tensor
    adv_int: torch.Tensor
    config: MixedRewardConfig
    num_generations_per_model: int | None = None
    diagnostics: MutableMapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def mode(self) -> str:
        return str(self.config.alpha_mode).strip().lower()


@dataclass(frozen=True)
class MixedRewardResult:
    """Fused advantages, retained rollouts, and optional post-normalization weights.

    sample_weight may be query-level (P, 1) or response-level (P, G).
    keep_mask excludes filtered rollouts from batch statistics and final updates.
    KL is handled separately by the policy objective.
    """

    advantages: torch.Tensor
    sample_weight: torch.Tensor | None = None
    keep_mask: torch.Tensor | None = None
