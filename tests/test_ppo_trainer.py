import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


class _Pbar:
    def __init__(self, iterable, **_):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, _):
        pass


@pytest.fixture
def ppo_module(monkeypatch):
    fake_ray = MagicMock()
    fake_ray.remote.side_effect = lambda obj=None, **_: (lambda cls: cls) if obj is None else obj
    fake_ray.get.side_effect = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=_Pbar))

    # Keep this control-flow test importable without Ray, vLLM, DeepSpeed, or datasets.
    for name in (
        "openrlhf.datasets",
        "openrlhf.datasets.utils",
        "openrlhf.trainer.ppo_utils.experience",
        "openrlhf.trainer.ppo_utils.experience_maker",
        "openrlhf.trainer.ppo_utils.kl_controller",
        "openrlhf.trainer.ppo_utils.samples_generator",
        "openrlhf.trainer.ray.launcher",
        "openrlhf.trainer.ray.vllm_engine",
        "openrlhf.utils.deepspeed",
        "openrlhf.utils.logging_utils",
        "openrlhf.utils.utils",
    ):
        monkeypatch.setitem(sys.modules, name, MagicMock())

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_ppo_trainer_test",
        root / "openrlhf" / "trainer" / "ppo_trainer.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.tqdm = _Pbar
    return module


class _Loader:
    def __len__(self):
        return 129

    def state_dict(self):
        return {"done": True}


@pytest.mark.parametrize("n", [1, 2])
def test_eval_reports_binary_program_passes_separately_from_test_pass_rates(ppo_module, n):
    loader = [(["math", "code"], ["math question", "code question"], ["1", "tests"], [None, None])]
    samples = []
    for prompt, rewards, fractions in (
        ("math question", [1., 0.], [None, None]),
        ("code question", [0., 1.], [0.5, 1.]),
    ):
        for reward, fraction in zip(rewards[:n], fractions[:n]):
            samples.append(SimpleNamespace(
                prompts=[prompt], rewards=torch.tensor([reward]), response_length=torch.tensor([10]),
                truncated=torch.tensor([False]), info={} if fraction is None else {
                    "code_test_pass_rate": torch.tensor([fraction])
                },
            ))
    logs = ppo_module.compute_eval_metrics(loader, samples, n)
    assert logs["eval_math_pass1"] == (1. if n == 1 else 0.5)
    assert logs["eval_code_pass1"] == (0. if n == 1 else 0.5)
    assert logs[f"eval_code_pass{n}"] == (0. if n == 1 else 1.)
    assert logs["eval_code_test_pass_rate"] == (0.5 if n == 1 else 0.75)
    assert "eval_math_test_pass_rate" not in logs


@pytest.mark.parametrize("incomplete", [False, True])
def test_eval_variable_groups_use_prompt_ids_even_when_text_is_duplicated(ppo_module, incomplete):
    loader = [(["math500", "livecodebench"], ["same prompt", "same prompt"], ["1", "tests"], [None, None])]
    samples = []
    for index, count, rewards in ((0, 1, [0.]), (1, 3, [1., 0., 0.])):
        for reward in rewards:
            samples.append(SimpleNamespace(
                prompts=["same prompt"], rewards=torch.tensor([reward]),
                response_length=torch.tensor([10]), truncated=torch.tensor([False]),
                info={"eval_prompt_index": torch.tensor([index]), "eval_n_samples_per_prompt": torch.tensor([count])},
            ))
    samples = [samples[i] for i in [1, 0, 3, 2]]
    if incomplete:
        with pytest.raises(ValueError, match="Incomplete evaluation group"):
            ppo_module.compute_eval_metrics(loader, samples[:-1], 1)
        return
    logs = ppo_module.compute_eval_metrics(loader, samples, 1)
    assert logs["eval_math500_pass1"] == 0.
    assert logs["eval_livecodebench_pass1"] == pytest.approx(1 / 3)
    assert logs["eval_livecodebench_pass3"] == 1.
    assert logs["eval_math500_n_samples_per_prompt"] == 1.
    assert logs["eval_livecodebench_n_samples_per_prompt"] == 3.
    assert logs["eval_num_samples"] == 4.


class _Generator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def generate_samples(self, **_):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("The terminal result must end the episode")
        return self.result


def _trainer(module, result):
    trainer = module.PPOTrainer.__new__(module.PPOTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(num_episodes=1),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        eval=SimpleNamespace(steps=float("inf"), temperature=1.0, n_samples_per_prompt=1),
    )
    trainer.prompts_dataloader = _Loader()
    trainer.eval_dataloader = None
    trainer.samples_generator = _Generator(result)
    trainer.generate_kwargs = {}
    trainer.init_checkpoint_states = lambda: {
        "episode": 0,
        "global_step": 0,
        "total_consumed_prompts": 0,
        "data_loader_state_dict": {},
    }
    trainer.restore_best_checkpoint_state = lambda _: None
    trainer.save_logs_and_checkpoints = lambda *_: None
    trainer.wandb_logger = trainer.tensorboard_logger = None
    return trainer


def test_sync_fit_processes_nonempty_exhausted_rollout(ppo_module):
    tail = [f"rollout-{i}" for i in range(129)]
    trainer = _trainer(ppo_module, (tail, None, 129, True))
    calls = []
    trainer.train_step = lambda samples, step: (calls.append(samples) or ({}, step + 1))

    trainer.fit()

    assert calls == [tail]
    assert trainer.samples_generator.calls == 1


def test_sync_fit_does_not_train_empty_exhausted_result(ppo_module):
    trainer = _trainer(ppo_module, ([], None, 0, True))
    trainer.train_step = lambda *_: pytest.fail("empty terminal result must not be trained")

    trainer.fit()

    assert trainer.samples_generator.calls == 1


@pytest.mark.parametrize("before_train,has_data,resumed", [
    (True, True, False), (False, True, False), (True, False, False), (True, True, True),
])
def test_sync_initial_eval_precedes_rollout_and_does_not_repeat_on_resume(ppo_module, before_train, has_data, resumed):
    trainer = _trainer(ppo_module, (["rollout"], None, 1, True))
    trainer.args.eval.before_train = before_train
    trainer.args.eval.temperature = 0.6
    trainer.args.eval.n_samples_per_prompt = 8
    trainer.eval_dataloader = ["eval"] if has_data else None
    state = trainer.init_checkpoint_states()
    state["global_step"] = 5 if resumed else 0
    trainer.init_checkpoint_states = lambda: state
    trainer.broadcast_to_vllm = lambda: None
    trainer.generate_kwargs = {"temperature": 1.0, "n_samples_per_prompt": 16}
    events = []

    def evaluate(step, **kwargs):
        assert trainer.samples_generator.calls == 0
        assert kwargs == {"temperature": 0.6, "n_samples_per_prompt": 8}
        events.append(("eval", step))
        return {"eval_math_pass1": 0.5}

    trainer.evaluate = evaluate
    trainer.save_eval_logs_and_checkpoint = lambda step, logs, states: events.append(("save", step, states.copy()))
    trainer.train_step = lambda samples, step: (events.append(("train", step)) or ({}, step + 1))
    trainer.fit()

    if before_train and has_data and not resumed:
        assert events == [("eval", 0), ("save", 0, state), ("train", 0)]
    else:
        assert events == [("train", state["global_step"])]
    assert trainer.samples_generator.calls == 1


def test_sync_periodic_eval_continues_after_initial_eval(ppo_module):
    trainer = _trainer(ppo_module, (["rollout"], None, 1, True))
    trainer.eval_dataloader = ["eval"]
    trainer.args.eval.steps = 1
    events = []
    trainer.evaluate = lambda step, **kwargs: events.append(("eval", step)) or {"eval_math_pass1": 0.5}
    trainer.save_eval_logs_and_checkpoint = lambda step, logs, states: events.append(("save", step))
    trainer.train_step = lambda samples, step: (events.append(("train", step)) or ({}, step + 1))

    trainer.fit()

    assert events == [("eval", 0), ("save", 0), ("train", 0), ("eval", 1), ("save", 1)]


@pytest.fixture
def async_ppo_module(ppo_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_trainer", ppo_module)
    monkeypatch.setitem(sys.modules, "ray.util.queue", MagicMock())
    path = Path(__file__).resolve().parents[1] / "openrlhf/trainer/ppo_trainer_async.py"
    spec = importlib.util.spec_from_file_location("_openrlhf_async_ppo_trainer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("before_train", [True, False])
@pytest.mark.parametrize("resumed", [True, False])
@pytest.mark.parametrize("has_data", [True, False])
def test_async_initial_eval_finishes_before_launching_training(async_ppo_module, before_train, resumed, has_data):
    events = []
    state = {"episode": 0, "global_step": 5 if resumed else 0,
             "total_consumed_prompts": 3 if resumed else 0, "data_loader_state_dict": {}}
    controller = async_ppo_module.PPOTrainerAsync.__new__(async_ppo_module.PPOTrainerAsync)
    controller.eval_before_train = before_train
    controller.generator_actor = MagicMock()
    controller.trainer_actor = MagicMock()
    controller.trainer_actor.init_checkpoint_states.remote.return_value = state
    controller.generator_actor._run_eval.remote.side_effect = lambda: events.append("eval") or (
        {"eval_math_pass1": 0.5} if has_data else {}
    )
    controller.trainer_actor.save_eval_logs_and_checkpoint.remote.side_effect = lambda *args: events.append("save")
    controller.generator_actor.fit.remote.side_effect = lambda **kwargs: events.append("rollout")
    controller.trainer_actor.fit.remote.side_effect = lambda **kwargs: events.append("train")

    controller.fit()

    expected = (["eval", "save"] if has_data else ["eval"]) if before_train and not resumed else []
    assert events == expected + ["rollout", "train"]
    if before_train and not resumed and has_data:
        controller.trainer_actor.save_eval_logs_and_checkpoint.remote.assert_called_once_with(
            0, {"eval_math_pass1": 0.5}, state
        )
    assert controller.trainer_actor.broadcast_to_vllm.remote.call_count == int(resumed)


def test_async_initial_eval_skips_generation_without_dataset(async_ppo_module):
    generator = async_ppo_module.GenerateSamplesActor.__new__(async_ppo_module.GenerateSamplesActor)
    generator.eval_dataloader = None
    generator.samples_generator = MagicMock()
    assert generator._run_eval() == {}
    generator.samples_generator.generate_eval_samples.assert_not_called()


def _checkpoint_trainer(module):
    trainer = module.BasePPOTrainer.__new__(module.BasePPOTrainer)
    trainer.args = SimpleNamespace(logger=SimpleNamespace(logging_steps=1), ckpt=SimpleNamespace(save_steps=1))
    trainer.actor_model_group = MagicMock()
    trainer.actor_model_group.async_run_method.return_value = []
    trainer.critic_model_group = None
    trainer.wandb_logger = trainer.tensorboard_logger = None
    trainer.best_eval_metric_key = ""
    trainer.best_eval_metric_value = float("-inf")
    trainer._latest_eval_metric_value = None
    return trainer


def test_initial_eval_logs_and_saves_resumable_best_checkpoint(ppo_module):
    trainer = _checkpoint_trainer(ppo_module)
    trainer.wandb_logger = MagicMock()
    trainer.tensorboard_logger = MagicMock()
    state = {"episode": 0, "global_step": 0, "total_consumed_prompts": 0, "data_loader_state_dict": {}}
    metrics = {"eval_math_pass1": 0.5}

    trainer.save_eval_logs_and_checkpoint(0, metrics, state)

    trainer.wandb_logger.log_eval.assert_called_once_with(0, metrics)
    trainer.tensorboard_logger.log_eval.assert_called_once_with(0, metrics)
    saved = trainer.actor_model_group.async_run_method.call_args.kwargs
    assert saved["tag"] == "best_global_step0"
    assert saved["client_states"]["global_step"] == 0
    assert saved["client_states"]["total_consumed_prompts"] == 0
    assert saved["client_states"]["data_loader_state_dict"] == {}
    assert trainer.best_eval_metric_value == 0.5


@pytest.mark.parametrize("best,latest", [(0.8, 0.5), (0.0, -0.2), (-0.2, -0.5)])
def test_regular_checkpoint_restores_best_and_latest_metrics(ppo_module, tmp_path, best, latest):
    import torch

    trainer = _checkpoint_trainer(ppo_module)
    trainer.save_best_checkpoint({"eval_math_pass1": best}, 1)
    trainer.save_best_checkpoint({"eval_math_pass1": latest}, 2)
    trainer.save_logs_and_checkpoints(3, client_states={"global_step": 3})
    saved = trainer.actor_model_group.async_run_method.call_args.kwargs["client_states"]
    path = tmp_path / "client_state.pt"
    torch.save(saved, path)

    resumed = _checkpoint_trainer(ppo_module)
    resumed.restore_best_checkpoint_state(torch.load(path, weights_only=True))
    assert resumed.best_eval_metric_key == "eval_math_pass1"
    assert resumed.best_eval_metric_value == best
    assert resumed._latest_eval_metric_value == latest
    resumed.save_best_checkpoint({"eval_math_pass1": (best + latest) / 2}, 4)
    resumed.actor_model_group.async_run_method.assert_not_called()
    resumed.save_best_checkpoint({"eval_math_pass1": best + 0.1}, 5)
    assert resumed.actor_model_group.async_run_method.call_args.kwargs["tag"] == "best_global_step5"


def test_best_checkpoint_overwrites_previous_latest_metric(ppo_module):
    trainer = _checkpoint_trainer(ppo_module)
    trainer.save_best_checkpoint({"eval_math_pass1": 0.8}, 1, {"latest_eval_metric_value": 0.2})
    saved = trainer.actor_model_group.async_run_method.call_args.kwargs["client_states"]
    resumed = _checkpoint_trainer(ppo_module)
    resumed.restore_best_checkpoint_state(saved)
    assert resumed.best_eval_metric_value == resumed._latest_eval_metric_value == 0.8


@pytest.mark.parametrize(
    "state,best,latest",
    [
        ({}, float("-inf"), None),
        ({"best_eval_metric_key": "eval_math_pass1", "best_eval_metric_value": 0.0}, 0.0, 0.0),
        ({"best_eval_metric_value": 0.8, "latest_eval_metric_value": None}, 0.8, None),
    ],
)
def test_restore_checkpoint_metric_compatibility(ppo_module, state, best, latest):
    trainer = _checkpoint_trainer(ppo_module)
    trainer.restore_best_checkpoint_state(state)
    assert trainer.best_eval_metric_value == best
    assert trainer._latest_eval_metric_value == latest


@pytest.mark.parametrize('enabled', [True, False])
def test_eval_cluster_monitor_preserves_metrics_and_training_state(ppo_module, enabled):
    from copy import deepcopy
    from openrlhf.algorithms.bifocalrl.runtime import DEFAULTS, assign_advantages

    calls = []

    class Maker:
        def make_experience_batch(self, samples):
            calls.append((self.args.rollout.n_samples_per_prompt, len(samples)))
            exp = SimpleNamespace(
                index=list(range(len(samples))), rewards=torch.cat([s.rewards for s in samples]),
                action_mask=torch.ones(len(samples), 1),
                info={key: torch.cat([s.info[key] for s in samples]) for key in samples[0].info},
            )
            exp.info['bifocal_sc'] = torch.arange(len(samples)).float()
            exp.info['bifocal_entropy'] = torch.ones(len(samples))
            assign_advantages(self, [exp])

    args = SimpleNamespace(
        bifocal=SimpleNamespace(settings={**deepcopy(DEFAULTS), 'monitor_eval': enabled}),
        rollout=SimpleNamespace(n_samples_per_prompt=16),
    )
    strategy = SimpleNamespace(args=args)
    maker = Maker()
    maker.args, maker.strategy = args, strategy
    maker.bifocal_monitor_metrics = {'training_metric': 123}
    maker.initial_model_group, maker.critic_model_group = 'reference', 'critic'
    loader = [(['math2', 'math4', 'code'], ['q0', 'q1', 'q2'], ['1', '1', '{}'], [None] * 3)]
    samples = []
    for index, n in enumerate([2, 4, 2]):
        for j in range(n):
            correct = float(j == 0)
            samples.append(SimpleNamespace(
                prompts=[f'q{index}'], rewards=torch.tensor([correct]), response_length=None, truncated=None,
                info={
                    'eval_prompt_index': torch.tensor([index]), 'eval_n_samples_per_prompt': torch.tensor([n]),
                    'bifocal_answer_cluster': torch.tensor([j % 2]), 'bifocal_answer_valid': torch.tensor([1.]),
                    'bifocal_monitor_math': torch.tensor([float(index != 2)]),
                    'bifocal_monitor_correct': torch.tensor([correct]), 'bifocal_mv': torch.tensor([float(j % 2 == 0)]),
                },
            ))
    # Tagged prompt groups may arrive out of order.
    samples = samples[::-1]
    trainer = object.__new__(ppo_module.PPOTrainer)
    trainer.args, trainer.strategy, trainer.experience_maker = args, strategy, maker
    trainer.eval_dataloader = loader
    trainer.samples_generator = SimpleNamespace(generate_eval_samples=lambda **_: samples)
    expected = ppo_module.compute_eval_metrics(loader, samples, 2)
    logs = trainer.evaluate(0, n_samples_per_prompt=2)
    for key, value in expected.items():
        assert logs[key] == pytest.approx(value)
    assert logs['eval_math4_monitor/before/cluster_count'] == 2
    assert not any(key.startswith('eval_code_monitor') for key in logs)
    assert ('eval_math2_monitor/retained/correct_positive_advantage_fraction' in logs) == enabled
    assert sorted(calls) == ([(2, 2), (4, 4)] if enabled else [])
    assert maker.args.rollout.n_samples_per_prompt == 16
    assert maker.initial_model_group == 'reference' and maker.critic_model_group == 'critic'
    assert maker.bifocal_monitor_metrics == {'training_metric': 123}
    assert all('bifocal_sc' not in s.info for s in samples)
    assert all(s.rewards.item() == s.info['bifocal_monitor_correct'].item() for s in samples)
