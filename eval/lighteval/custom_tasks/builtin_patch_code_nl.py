"""
Monkey-patch LightEval 内置任务 (gsm8k, math_500)，注入 code/nl 准确率指标。

当使用 LightEval 默认任务（不用 local_tasks）时，用此模块作为 --custom-tasks，
可在不替换任务定义的情况下，给内置任务追加 code/nl 指标。

用法:
  export CO_LIGHTEVAL_CODE_NL_METRICS=1
  python -m src.evaluate_lighteval \
    --model_path ... \
    --tasks math_500,gsm8k \
    --custom_tasks eval.lighteval.custom_tasks.builtin_patch_code_nl \
    --no-map_builtin_math_500_to_custom \
    --no-map_builtin_gsm8k_to_custom \
    ...

注意: 本模块会先 import offline_patch 做数据集本地重定向（若需要），再注入指标。
"""

from __future__ import annotations

import os
import sys

# 先应用 offline patch（与 offline_patch 相同，用于本地数据集）
from . import offline_patch as _  # noqa: F401

# 必须设置才注入
if os.environ.get("CO_LIGHTEVAL_CODE_NL_METRICS", "0") != "1":
    TASKS_TABLE = []
else:
    # 注入 code/nl 指标到内置任务
    from .local_tasks import (
        _GSM8K_CODE_ACCURACY_METRIC,
        _GSM8K_NL_ACCURACY_METRIC,
        _MATH_CODE_ACCURACY_METRIC,
        _MATH_NL_ACCURACY_METRIC,
    )

    def _patch_module(module_path: str, task_name: str, extra_metrics: list) -> None:
        if module_path not in sys.modules:
            return
        mod = sys.modules[module_path]
        table = getattr(mod, "TASKS_TABLE", None)
        if not table:
            return
        for config in table:
            if getattr(config, "name", None) == task_name:
                metrics = list(getattr(config, "metrics", []) or [])
                for m in extra_metrics:
                    if m not in metrics:
                        metrics.append(m)
                config.metrics = metrics
                break

    _patch_module("lighteval.tasks.tasks.gsm8k", "gsm8k", [_GSM8K_CODE_ACCURACY_METRIC, _GSM8K_NL_ACCURACY_METRIC])
    _patch_module("lighteval.tasks.tasks.math_500", "math_500", [_MATH_CODE_ACCURACY_METRIC, _MATH_NL_ACCURACY_METRIC])

    TASKS_TABLE = []  # 不添加新任务，只 patch 内置
