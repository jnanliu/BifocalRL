"""Explicit registration for built-in and experimental reward-fusion modes."""

from __future__ import annotations

from collections.abc import Callable

from .types import MixedRewardContext, MixedRewardResult

MixedRewardStrategy = Callable[[MixedRewardContext], MixedRewardResult]
_STRATEGIES: dict[str, MixedRewardStrategy] = {}


def register_mixed_reward_mode(
    name: str,
    strategy: MixedRewardStrategy | None = None,
):
    """Register a strategy directly or as a decorator; reject duplicate names."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Mixed-reward mode name cannot be empty")

    def register(handler: MixedRewardStrategy) -> MixedRewardStrategy:
        if key in _STRATEGIES:
            raise ValueError(f"Mixed-reward mode already registered: {key}")
        if not callable(handler):
            raise TypeError("Mixed-reward strategy must be callable")
        _STRATEGIES[key] = handler
        return handler

    return register if strategy is None else register(strategy)


def get_mixed_reward_mode(name: str) -> MixedRewardStrategy:
    try:
        return _STRATEGIES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown mixed_reward alpha mode: {name!r}") from None


def available_mixed_reward_modes() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))
