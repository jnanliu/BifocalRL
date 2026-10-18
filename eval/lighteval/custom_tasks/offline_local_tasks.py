"""
Offline + local tasks bundle for LightEval.

Why this file exists:
- `offline_patch.py` only patches dataset IO for offline/local snapshots; it defines no tasks.
- `local_tasks.py` defines custom tasks (e.g. `local_math_500`) that default to single-sample generation.

LightEval only accepts one `--custom-tasks` module path, so we expose a single module that:
1) installs the offline dataset redirect patch on import
2) exports TASKS_TABLE with our custom tasks

Use with:
  --custom-tasks eval.lighteval.custom_tasks.offline_local_tasks
and tasks like:
  custom|local_math_500|0|0,custom|local_gsm8k|0|0
"""

from __future__ import annotations

# Install offline dataset redirect patch (runs on import).
from . import offline_patch as _offline_patch  # noqa: F401

# Export custom tasks table.
from .local_tasks import TASKS_TABLE  # noqa: F401

