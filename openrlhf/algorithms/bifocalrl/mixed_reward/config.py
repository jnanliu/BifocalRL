"""Numerical reward-fusion settings, independent of CLI and training frameworks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MixedRewardConfig:
    """Existing mixed_reward_* CLI/JSON parameters without the shared prefix."""

    alpha: float = 0.5
    alpha_mode: str = "bifocalrl"
    gd2po_filter_mode: str = "snr"
    gd2po_snr_threshold: float = 0.5
    gd2po_query_reweight: bool = True
    sc_between_gamma: float = 0.5
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

    def __post_init__(self):
        if self.gd2po_filter_mode not in ("none", "hard", "snr"):
            raise ValueError("gd2po_filter_mode must be none, hard, or snr")
        if not math.isfinite(self.gd2po_snr_threshold) or not 0 <= self.gd2po_snr_threshold <= 1:
            raise ValueError("gd2po_snr_threshold must be in [0, 1]")
        if self.alpha_mode == "bifocalrl":
            if not math.isfinite(self.sc_between_gamma) or not 0 <= self.sc_between_gamma <= 1:
                raise ValueError("sc_between_gamma must be in [0, 1]")
            if not self.final_normalize:
                raise ValueError("BifocalRL requires final_normalize=true")
            if not math.isfinite(self.alpha) or self.alpha != 0.5:
                raise ValueError("BifocalRL requires fixed alpha=0.5")

    @classmethod
    def from_training_args(cls, args: Any) -> MixedRewardConfig:
        """Read existing TrainingArguments; missing values retain historical defaults."""
        return cls(**{field.name: getattr(args, f"mixed_reward_{field.name}", field.default) for field in fields(cls)})

    def as_kwargs(self) -> dict[str, Any]:
        """Serialize the fusion settings as a plain dictionary."""
        return asdict(self)
