"""Numerical reward-fusion settings, independent of CLI and training frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MixedRewardConfig:
    """Existing mixed_reward_* CLI/JSON parameters without the shared prefix."""

    alpha: float = 0.5
    alpha_mode: str = "fixed"
    alpha_zscore_mu: float = 0.65
    alpha_zscore_sigma: float = 0.2
    alpha_zscore_lower: float = 0.0
    alpha_zscore_upper: float = 1.0
    corr_weight_mode: str = "none"
    corr_weight_threshold: float = 0.1
    corr_weight_exponent: float = 0.8
    reliability_sc_weight: float = 0.6
    reliability_corr_weight: float = 0.4
    reliability_use_sample_weight: bool = True
    reliability_use_advantage_weight: bool = True
    reliability_sample_weight_mu: float = 1.0
    reliability_sample_weight_sigma: float = 0.2
    reliability_sample_weight_lower: float = 0.5
    reliability_sample_weight_upper: float = 1.5
    reliability_advantage_weight_mu: float = 0.5
    reliability_advantage_weight_sigma: float = 0.2
    reliability_advantage_weight_lower: float = 0.2
    reliability_advantage_weight_upper: float = 0.8
    final_normalize: bool = True
    eps: float = 1e-06

    @classmethod
    def from_training_args(cls, args: Any) -> MixedRewardConfig:
        """Read existing TrainingArguments; missing values retain historical defaults."""
        return cls(**{field.name: getattr(args, f"mixed_reward_{field.name}", field.default) for field in fields(cls)})

    def as_kwargs(self) -> dict[str, Any]:
        """Arguments accepted by the legacy compute_mixed_reward_advantages API."""
        return asdict(self)
