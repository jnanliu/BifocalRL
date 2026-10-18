"""Offline LiveCodeBench release_v6 task with one completion per problem."""

from __future__ import annotations

import numpy as np

# Install the repository's local-dataset redirects before LightEval loads LCB.
from . import offline_patch as _offline_patch  # noqa: F401

from lighteval.metrics.metrics import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.extended.lcb.main import codegen_metric, lcb_codegeneration_prompt_fn
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LCB_CODEGEN_PASS_AT_1_SINGLE = SampleLevelMetric(
    metric_name="codegen_pass@1:1",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=codegen_metric,
    corpus_level_fn=np.mean,
)


TASKS_TABLE = [
    LightevalTaskConfig(
        name="lcb:codegeneration_release_v6_single",
        suite=["custom"],
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="livecodebench/code_generation_lite",
        hf_subset="release_v6",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=32768,
        metric=[LCB_CODEGEN_PASS_AT_1_SINGLE],
        stop_sequence=[],
        trust_dataset=True,
        version=0,
    )
]
