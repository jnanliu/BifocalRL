import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def test_terminal_oversampling_buffer_is_drained_before_next_episode(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", MagicMock())
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=object))
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.experience", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.vllm_engine", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", MagicMock())

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_samples_generator_test",
        root / "openrlhf" / "trainer" / "ppo_utils" / "samples_generator.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=2, n_samples_per_prompt=1, vllm_generate_batch_size=4),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        vllm=SimpleNamespace(enable_sleep=False),
    )
    loader = MagicMock()
    loader.__iter__.side_effect = lambda: iter([0, 1, 2])
    generator = module.SamplesGenerator(SimpleNamespace(args=args), loader, None, None, [])

    def generate(dataloader_iter, num_prompts, dynamic_filtering, **kwargs):
        del dynamic_filtering, kwargs
        generated = []
        for _ in range(num_prompts):
            try:
                generated.append(next(dataloader_iter))
            except StopIteration:
                break
        return generated, len(generated), len(generated) < num_prompts

    generator._generate_vllm = generate

    first, _, first_prompts, first_exhausted = generator.generate_samples()
    first_buffer = list(generator._sample_buffer)
    second, _, second_prompts, second_exhausted = generator.generate_samples()
    third, _, third_prompts, third_exhausted = generator.generate_samples()

    assert (first, first_prompts, first_exhausted) == ([0, 1], 3, False)
    assert first_buffer == [2]
    assert (second, second_prompts, second_exhausted) == ([2], 0, True)
    assert (third, third_prompts, third_exhausted) == ([0, 1], 3, False)
    assert loader.__iter__.call_count == 2


class _Pbar:
    def __init__(self, *_args, **_kwargs):
        pass

    def set_postfix(self, *_args, **_kwargs):
        pass

    def update(self, *_args, **_kwargs):
        pass


@pytest.fixture
def samples_generator_module(monkeypatch):
    fake_ray = types.ModuleType("ray")
    fake_ray.cancelled = []
    fake_ray.wait = lambda refs, **_: (refs[:1], refs[1:])
    fake_ray.get = lambda ref: ref
    fake_ray.cancel = fake_ray.cancelled.append
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=_Pbar))

    experience_module = types.ModuleType("openrlhf.trainer.ppo_utils.experience")
    experience_module.Experience = object
    monkeypatch.setitem(sys.modules, experience_module.__name__, experience_module)

    engine_module = types.ModuleType("openrlhf.trainer.ray.vllm_engine")
    engine_module.batch_vllm_engine_call = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)

    logging_module = types.ModuleType("openrlhf.utils.logging_utils")
    logging_module.init_logger = lambda *_args, **_kwargs: SimpleNamespace(info=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, logging_module.__name__, logging_module)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_samples_generator_test",
        root / "openrlhf" / "trainer" / "ppo_utils" / "samples_generator.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, fake_ray


def test_dynamic_filtering_preserves_terminal_accepted_rollouts(samples_generator_module):
    module, fake_ray = samples_generator_module
    generator = module.SamplesGenerator.__new__(module.SamplesGenerator)
    generator.args = SimpleNamespace(algo=SimpleNamespace(dynamic_filtering_range=(0.0, 1.0)))

    def group(prefix, scores):
        return [
            SimpleNamespace(name=f"{prefix}-{index}", scores=torch.tensor([score]))
            for index, score in enumerate(scores)
        ]

    refs = [group("accepted", [0.0, 1.0]), group("rejected", [0.0, 0.0]), group("pending", [0.0, 1.0])]
    generator._dispatch_prompts_to_vllm = lambda *_args, **_kwargs: refs
    generator._process_response_into_experience = lambda response, **_kwargs: response

    dataloader = iter((["default"], [f"prompt-{index}"], [None], [None]) for index in range(3))
    experiences, prompts_consumed, exhausted = generator._generate_vllm(
        dataloader, num_prompts=3, dynamic_filtering=True
    )

    assert [experience.name for experience in experiences] == [
        "accepted-0",
        "accepted-1",
        "pending-0",
        "pending-1",
    ]
    assert prompts_consumed == 3
    assert exhausted is True
    assert fake_ray.cancelled == []


def test_eval_uses_datasource_sample_counts_and_tags_out_of_order_groups(samples_generator_module):
    module, fake_ray = samples_generator_module
    module.SamplingParams = SimpleNamespace
    calls, results = [], {}

    def generate_responses(**kwargs):
        calls.append(kwargs)
        ref = object()
        results[ref] = [
            SimpleNamespace(prompts=[kwargs["prompt"]], info={}, scores=None)
            for _ in range(kwargs["num_samples"])
        ]
        return ref

    engine = SimpleNamespace(
        generate_responses=SimpleNamespace(remote=generate_responses),
        get_num_unfinished_requests=SimpleNamespace(remote=lambda: 0),
    )
    fake_ray.get = lambda ref: ref if isinstance(ref, list) else results[ref]
    fake_ray.wait = lambda refs, **_: (refs[-1:], refs[:-1])
    args = SimpleNamespace(
        eval=SimpleNamespace(
            n_samples_per_prompt=2,
            n_samples_per_prompt_by_datasource='{"math500":1,"livecodebench":3}',
            sandbox_concurrency=3, sandbox_timeout=5, sandbox_memory_limit_mb=512,
        ),
        rollout=SimpleNamespace(batch_size=2, n_samples_per_prompt=7),
        algo=SimpleNamespace(advantage=SimpleNamespace(is_correction_level="off")),
        vllm=SimpleNamespace(enable_sleep=False),
        bifocal=SimpleNamespace(settings={"score_concurrency": 16, "sandbox_timeout": 10, "sandbox_memory_limit_mb": 1024}),
    )
    loader = [
        (["math500"], ["same prompt"], ["math label"], [None]),
        (["livecodebench"], ["same prompt"], ["code tests"], [None]),
        (["other"], ["last prompt"], ["last label"], [None]),
    ]
    generator = module.SamplesGenerator(SimpleNamespace(args=args), None, loader, None, [engine])
    generator._process_response_into_experience = lambda response, **kwargs: response
    for _ in range(2):
        samples = generator.generate_eval_samples()
        assert [s.info["eval_prompt_index"].item() for s in samples] == [1, 1, 1, 0, 2, 2]
        assert [s.info["eval_n_samples_per_prompt"].item() for s in samples] == [3, 3, 3, 1, 2, 2]
        assert generator._eval_dataloader_iter is None
    assert [call["num_samples"] for call in calls] == [1, 3, 2, 1, 3, 2]
    eval_cfg = calls[0]["sampling_params"].extra_args["bifocal"]
    assert (eval_cfg["score_concurrency"], eval_cfg["sandbox_timeout"], eval_cfg["sandbox_memory_limit_mb"]) == (3, 5, 512)
    # Training keeps its own group size even when its datasource has an eval override.
    generator._dispatch_prompts_to_vllm(
        ["train"], ["unused"], datasources=["livecodebench"], n_samples_per_prompt=7
    )
    assert calls[-1]["num_samples"] == 7
    assert calls[-1]["sampling_params"].extra_args["bifocal"]["score_concurrency"] == 16
    assert args.bifocal.settings["sandbox_timeout"] == 10


@pytest.mark.parametrize("overrides", ["[]", '{"math":0}', '{"math":-1}', '{"math":1.5}', '{"math":true}'])
def test_eval_sample_count_overrides_require_positive_integers(samples_generator_module, overrides):
    module, _ = samples_generator_module
    args = SimpleNamespace(eval=SimpleNamespace(n_samples_per_prompt=1, n_samples_per_prompt_by_datasource=overrides))
    with pytest.raises(ValueError, match="positive integers"):
        module.SamplesGenerator(SimpleNamespace(args=args), None, None, None, [])
