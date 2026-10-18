# coding=utf-8
"""
Experience replay buffer for continual / curriculum learning.

Maintains a persistent buffer of "breakthrough" trajectories — correct
solutions to problems that were initially classified as Hard (difficulty
score indicating p < 0.2 solve rate).  During training, a configurable
fraction (default 10%) of each batch can be replaced with buffer samples
to stabilize acquired reasoning capabilities.

This module provides **interface only** — it is not wired into the training
loop by default.
"""

from __future__ import annotations

import json
import logging
import os
import random as _random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """A single trajectory stored in the replay buffer."""
    prompt: str
    completion: str
    prompt_id: str = ""
    difficulty_score: float = 0.0
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """JSON-backed persistent experience replay buffer.

    Args:
        save_path: Path to the JSON file for persistence.
        max_size: Maximum number of entries to retain (FIFO eviction).
        hard_threshold: Difficulty score threshold — only entries with
            ``difficulty_score >= hard_threshold`` are eligible for addition.
            Since difficulty = 1 - accuracy, ``hard_threshold=0.8`` corresponds
            to problems with p < 0.2 solve rate.
    """

    def __init__(
        self,
        save_path: str,
        max_size: int = 10000,
        hard_threshold: float = 0.8,
    ) -> None:
        self.save_path = save_path
        self.max_size = max_size
        self.hard_threshold = hard_threshold
        self._entries: List[BufferEntry] = []
        if os.path.exists(save_path):
            self.load()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def should_add(self, difficulty_score: float) -> bool:
        """Return True if the sample qualifies as a Hard breakthrough."""
        return difficulty_score >= self.hard_threshold

    def add(
        self,
        prompt: str,
        completion: str,
        prompt_id: str = "",
        difficulty_score: float = 0.0,
        reward: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a breakthrough trajectory to the buffer."""
        entry = BufferEntry(
            prompt=prompt,
            completion=completion,
            prompt_id=prompt_id,
            difficulty_score=difficulty_score,
            reward=reward,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size:]

    def sample(
        self,
        n: int,
        rng: Optional[_random.Random] = None,
    ) -> List[BufferEntry]:
        """Randomly sample *n* entries (with replacement if n > len)."""
        if not self._entries:
            return []
        rng = rng or _random.Random()
        if n >= len(self._entries):
            return list(self._entries)
        return rng.sample(self._entries, n)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the buffer to disk as JSON."""
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        data = {
            "max_size": self.max_size,
            "hard_threshold": self.hard_threshold,
            "entries": [asdict(e) for e in self._entries],
        }
        tmp = self.save_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.save_path)
        logger.debug("Replay buffer saved: %d entries -> %s", len(self._entries), self.save_path)

    def load(self) -> None:
        """Load the buffer from disk."""
        if not os.path.exists(self.save_path):
            return
        with open(self.save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.max_size = data.get("max_size", self.max_size)
        self.hard_threshold = data.get("hard_threshold", self.hard_threshold)
        raw_entries = data.get("entries", [])
        self._entries = []
        for raw in raw_entries:
            self._entries.append(BufferEntry(
                prompt=raw.get("prompt", ""),
                completion=raw.get("completion", ""),
                prompt_id=raw.get("prompt_id", ""),
                difficulty_score=raw.get("difficulty_score", 0.0),
                reward=raw.get("reward", 0.0),
                metadata=raw.get("metadata", {}),
            ))
        logger.debug("Replay buffer loaded: %d entries from %s", len(self._entries), self.save_path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(save_path={self.save_path!r}, "
            f"size={len(self._entries)}, max_size={self.max_size})"
        )

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()


# ---------------------------------------------------------------------------
# Batch mixing helper
# ---------------------------------------------------------------------------

def mix_replay_into_batch(
    batch_prompts: List[str],
    batch_completions: List[str],
    buffer: ReplayBuffer,
    replay_ratio: float = 0.1,
    rng: Optional[_random.Random] = None,
) -> Tuple[List[str], List[str], int]:
    """Replace a fraction of the batch with replay buffer samples.

    Randomly selects ``ceil(batch_size * replay_ratio)`` positions in the batch
    and overwrites them with samples drawn from *buffer*.  If the buffer is
    empty, the batch is returned unchanged.

    Args:
        batch_prompts: Prompts in the current training batch.
        batch_completions: Completions in the current training batch.
        buffer: The :class:`ReplayBuffer` to draw from.
        replay_ratio: Fraction of the batch to replace (default 0.1 = 10%).
        rng: Optional seeded RNG for reproducibility.

    Returns:
        ``(mixed_prompts, mixed_completions, n_replaced)``
    """
    if len(buffer) == 0 or replay_ratio <= 0:
        return batch_prompts, batch_completions, 0

    rng = rng or _random.Random()
    batch_size = len(batch_prompts)
    n_replace = max(1, int(_import_math().ceil(batch_size * replay_ratio)))
    n_replace = min(n_replace, batch_size, len(buffer))

    replay_samples = buffer.sample(n_replace, rng=rng)
    replace_indices = rng.sample(range(batch_size), n_replace)

    mixed_p = list(batch_prompts)
    mixed_c = list(batch_completions)
    for idx, entry in zip(replace_indices, replay_samples):
        mixed_p[idx] = entry.prompt
        mixed_c[idx] = entry.completion

    return mixed_p, mixed_c, n_replace


def _import_math():
    import math
    return math
