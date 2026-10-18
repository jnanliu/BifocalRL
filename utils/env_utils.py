# coding=utf-8
"""Environment variable helpers."""

import os


def co_env_flag(name: str, default: str = "0") -> bool:
    """Parse env var as a boolean flag (0/1/false/true/yes/no/on/off/none)."""
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in ("", "0", "false", "no", "off", "none")


def co_int_env(name: str, default: int) -> int:
    """Parse env var as an integer, returning *default* on any failure."""
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)
