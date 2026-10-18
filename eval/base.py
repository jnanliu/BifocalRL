"""
Unified evaluation interface.

All evaluation backends (HACRL, TTRL, LightEval, ZeroEval) implement the
same abstract interface so callers use a consistent pattern. Each backend
has its own ``enabled`` flag and parameters in the experiment JSON config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvalBackendConfig:
    """Base configuration shared across all evaluation backends."""

    model_path: str
    output_root: str
    datasets: List[str] = field(default_factory=list)
    num_generations: int = 1
    max_samples: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 4096
    seed: int = 7


@dataclass(frozen=True)
class EvalResult:
    """Unified result returned by every evaluation backend."""

    dataset: str
    accuracy: float
    num_correct: int
    num_total: int
    output_path: str
    details: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation backends."""

    @abstractmethod
    def evaluate(self, config: EvalBackendConfig) -> List[EvalResult]:
        """Run evaluation, return per-dataset results."""
