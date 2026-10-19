"""Public mixed-reward API and extension points.

Importing this package requires PyTorch, not Transformers, Accelerate or datasets.
"""

from .config import MixedRewardConfig
from .core import compute_mixed_reward, compute_mixed_reward_advantages
from .registry import available_mixed_reward_modes, register_mixed_reward_mode
from .types import MixedRewardContext, MixedRewardResult

__all__ = [
    "MixedRewardConfig",
    "MixedRewardContext",
    "MixedRewardResult",
    "compute_mixed_reward",
    "compute_mixed_reward_advantages",
    "available_mixed_reward_modes",
    "register_mixed_reward_mode",
]
