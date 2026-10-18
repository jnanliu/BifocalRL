import json
from pathlib import Path

import pytest
import torch

from src.training.co_intuitor_train import compute_mixed_reward_advantages
from src.training.single_intuitor_train import (
    _build_prism_ettrl_diagnostics_payload,
    _build_training_tensors,
    _build_reliability_diagnostics_payload,
    _requires_mixed_reward_diagnostics,
    _requires_reliability_diagnostics,
    _write_prism_ettrl_diagnostics_if_main,
    _write_reliability_diagnostics_if_main,
)
from utils.cache import CoIntuitorCache


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    model_max_length = 128

    def __call__(
        self,
        *,
        text,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False,
    ):
        del return_tensors, padding, add_special_tokens
        texts = text if isinstance(text, list) else [text]
        rows = [torch.tensor([3 + (ord(ch) % 41) for ch in value], dtype=torch.long) for value in texts]
        width = max(row.numel() for row in rows)
        padded = []
        masks = []
        for row in rows:
            missing = width - row.numel()
            pad = torch.zeros(missing, dtype=torch.long)
            mask_pad = torch.zeros(missing, dtype=torch.long)
            mask = torch.ones(row.numel(), dtype=torch.long)
            if padding_side == "left":
                padded.append(torch.cat([pad, row]))
                masks.append(torch.cat([mask_pad, mask]))
            else:
                padded.append(torch.cat([row, pad]))
                masks.append(torch.cat([mask, mask_pad]))
        return {"input_ids": torch.stack(padded), "attention_mask": torch.stack(masks)}


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    mv = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    self_certainty = torch.tensor(
        [
            [0.9, 0.7, 0.5, 0.1],
            [0.2, 0.8, 0.4, 0.6],
            [0.8, 0.1, 0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    return mv, self_certainty


def _parameters(*, use_sample: bool, use_advantage: bool) -> dict:
    return {
        "reward_mode": "mixed_reward",
        "alpha_mode": "reliability_sc_corr",
        "alpha": 0.5,
        "sc_weight": 0.6,
        "corr_weight": 0.4,
        "use_sample_weight": use_sample,
        "use_advantage_weight": use_advantage,
        "sample_weight": {
            "mu": 1.0,
            "sigma": 0.2,
            "lower": 0.5,
            "upper": 1.5,
        },
        "advantage_weight": {
            "mu": 0.5,
            "sigma": 0.2,
            "lower": 0.2,
            "upper": 0.8,
        },
        "final_normalize": True,
    }


def _diagnostics(*, use_sample: bool, use_advantage: bool):
    mv, self_certainty = _inputs()
    diagnostics: dict = {}
    final = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="reliability_sc_corr",
        reliability_use_sample_weight=use_sample,
        reliability_use_advantage_weight=use_advantage,
        final_normalize=True,
        diagnostics=diagnostics,
    )
    diagnostics["mv_votes"] = [
        {"selected_answer": str(index), "answer_clusters": {str(index): 3.0}}
        for index in range(mv.size(0))
    ]
    diagnostics["mv_extracted_answers"] = [
        [str(index), str(index), None, str(index)] for index in range(mv.size(0))
    ]
    return final, diagnostics


@pytest.mark.parametrize(
    ("use_sample", "use_advantage"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_reliability_diagnostics_capture_computed_and_applied_weights(
    use_sample: bool,
    use_advantage: bool,
) -> None:
    final, diagnostics = _diagnostics(
        use_sample=use_sample, use_advantage=use_advantage
    )

    expected_keys = {
        "mv_rewards",
        "self_certainty_rewards",
        "mv_advantages",
        "self_certainty_advantages",
        "raw_sc",
        "raw_corr",
        "z_sc",
        "z_corr",
        "reliability",
        "computed_sample_weight",
        "applied_sample_weight",
        "computed_advantage_weight",
        "applied_advantage_weight",
        "combined_before_final_normalize",
        "combined_after_final_normalize",
        "final_advantages",
    }
    assert expected_keys <= diagnostics.keys()
    for key in expected_keys:
        assert diagnostics[key].device.type == "cpu"
        assert diagnostics[key].dtype == torch.float32
        assert torch.isfinite(diagnostics[key]).all()

    computed_sample = diagnostics["computed_sample_weight"]
    computed_advantage = diagnostics["computed_advantage_weight"]
    expected_sample = computed_sample if use_sample else torch.ones_like(computed_sample)
    expected_advantage = (
        computed_advantage
        if use_advantage
        else torch.full_like(computed_advantage, 0.5)
    )
    assert torch.equal(diagnostics["applied_sample_weight"], expected_sample)
    assert torch.equal(diagnostics["applied_advantage_weight"], expected_advantage)
    assert torch.equal(diagnostics["final_advantages"], final)
    assert diagnostics["diagnostics_only_reliability"] is (
        not use_sample and not use_advantage
    )


def test_single_training_tensor_builder_captures_real_mv_diagnostics() -> None:
    prompts = ["first problem", "second problem"]
    completions = [
        "answer \\boxed{1}",
        "again \\boxed{1}",
        "different \\boxed{2}",
        "missing box",
        "answer \\boxed{3}",
        "again \\boxed{3}",
        "third \\boxed{3}",
        "different \\boxed{4}",
    ]
    scores = [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.5, 0.2]
    diagnostics: dict = {}
    training_data = _build_training_tensors(
        tokenizer=_TinyTokenizer(),
        prompts=prompts,
        completions=completions,
        scores=scores,
        num_generations=4,
        reward_mode="mixed_reward",
        reward_length_norm="response",
        step=2,
        max_prompt_length=64,
        max_completion_length=64,
        mask_truncated_completions=False,
        scale_rewards=True,
        mixed_reward_alpha_mode="reliability_sc_corr",
        mixed_reward_reliability_use_sample_weight=True,
        mixed_reward_reliability_use_advantage_weight=False,
        finish_reasons=["stop"] * 8,
        old_logps=None,
        ref_logps=None,
        mixed_reward_diagnostics=diagnostics,
    )

    assert tuple(training_data["advantages"].shape) == (8,)
    assert len(diagnostics["mv_votes"]) == 2
    assert len(diagnostics["mv_extracted_answers"]) == 2
    assert diagnostics["mv_votes"][0]["selected_answer"] == "1"
    assert diagnostics["mv_extracted_answers"][0] == ["1", "1", "2", None]
    assert tuple(diagnostics["mv_rewards"].shape) == (2, 4)
    assert torch.equal(
        diagnostics["final_advantages"].reshape(-1),
        training_data["advantages"],
    )


def test_diagnostics_baseline_remains_exact_alpha050() -> None:
    mv, self_certainty = _inputs()
    diagnostics: dict = {}
    baseline = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="reliability_sc_corr",
        reliability_use_sample_weight=False,
        reliability_use_advantage_weight=False,
        diagnostics=diagnostics,
    )
    fixed = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha=0.5,
        alpha_mode="fixed",
    )
    assert torch.equal(baseline, fixed)
    assert diagnostics["diagnostics_only_reliability"] is True
    assert torch.equal(
        diagnostics["applied_sample_weight"],
        torch.ones_like(diagnostics["applied_sample_weight"]),
    )
    assert torch.equal(
        diagnostics["applied_advantage_weight"],
        torch.full_like(diagnostics["applied_advantage_weight"], 0.5),
    )


def test_payload_batch_lists_rankings_and_records_are_aligned() -> None:
    final, diagnostics = _diagnostics(use_sample=True, use_advantage=False)
    prompts = ["prompt zero", "prompt one", "prompt two"]
    sampled_indices = [91, 17, 44]
    payload = _build_reliability_diagnostics_payload(
        step=5,
        prompts=prompts,
        sampled_indices=sampled_indices,
        num_generations=4,
        diagnostics=diagnostics,
        training_advantages=final.reshape(-1),
        parameters=_parameters(use_sample=True, use_advantage=False),
    )

    assert list(payload)[:8] == [
        "schema_version",
        "step",
        "num_prompts",
        "num_generations",
        "num_responses",
        "diagnostics_only_reliability",
        "parameters",
        "tensor_shapes",
    ]
    assert payload["num_prompts"] == 3
    assert payload["num_responses"] == 12
    assert payload["prompt_indices"] == [0, 1, 2]
    assert payload["dataset_indices"] == sampled_indices
    assert len(payload["batch_reliability"]) == 3
    assert len(payload["records"]) == 3

    for index, record in enumerate(payload["records"]):
        assert record["prompt_index"] == index
        assert record["dataset_index"] == sampled_indices[index]
        assert record["prompt"] == prompts[index]
        assert record["reliability"]["value"] == payload["batch_reliability"][index]
        assert record["weights"]["computed_sample"] == payload["batch_weights"][
            "computed_sample"
        ][index]
        assert len(record["mv"]["rewards"]) == 4
        assert len(record["mv"]["advantages"]) == 4
        assert len(record["self_certainty"]["rewards"]) == 4
        assert len(record["self_certainty"]["advantages"]) == 4
        assert len(record["mixed"]["final_advantages"]) == 4

    values = payload["batch_reliability"]
    expected_desc = sorted(range(3), key=lambda index: (-values[index], index))
    expected_asc = sorted(range(3), key=lambda index: (values[index], index))
    assert [row["prompt_index"] for row in payload["reliability_ranking_desc"]] == expected_desc
    assert [row["prompt_index"] for row in payload["reliability_ranking_asc"]] == expected_asc


def test_payload_rejects_misaligned_indices_and_training_advantages() -> None:
    final, diagnostics = _diagnostics(use_sample=True, use_advantage=True)
    kwargs = {
        "step": 1,
        "prompts": ["a", "b", "c"],
        "sampled_indices": [1, 2, 3],
        "num_generations": 4,
        "diagnostics": diagnostics,
        "training_advantages": final.reshape(-1),
        "parameters": _parameters(use_sample=True, use_advantage=True),
    }
    with pytest.raises(RuntimeError, match="sampled-index count"):
        _build_reliability_diagnostics_payload(
            **{**kwargs, "sampled_indices": [1, 2]}
        )

    wrong = final.clone()
    wrong[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="do not match the training batch"):
        _build_reliability_diagnostics_payload(
            **{**kwargs, "training_advantages": wrong.reshape(-1)}
        )


def test_required_writer_is_atomic_overwrites_and_skips_non_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CO_DEBUG_DATA", "0")
    payload = {"schema_version": 1, "step": 7, "value": 1}
    skipped = _write_reliability_diagnostics_if_main(
        is_main_process=False,
        base_dir=str(tmp_path),
        step=7,
        payload=payload,
    )
    assert skipped is None
    assert not list(tmp_path.rglob("mixed_reward_reliability_step7.json"))

    path = _write_reliability_diagnostics_if_main(
        is_main_process=True,
        base_dir=str(tmp_path),
        step=7,
        payload=payload,
    )
    assert path is not None
    artifact = Path(path)
    assert json.loads(artifact.read_text()) == payload

    replacement = {"schema_version": 1, "step": 7, "value": 2}
    _write_reliability_diagnostics_if_main(
        is_main_process=True,
        base_dir=str(tmp_path),
        step=7,
        payload=replacement,
    )
    assert json.loads(artifact.read_text()) == replacement
    assert not list(artifact.parent.glob(f".{artifact.name}.*.tmp"))


def test_full_128_prompt_artifact_has_complete_aligned_batch_lists(tmp_path: Path) -> None:
    torch.manual_seed(7)
    num_prompts, num_generations = 128, 8
    mv = (torch.rand(num_prompts, num_generations) > 0.45).float()
    self_certainty = torch.randn(num_prompts, num_generations)
    diagnostics: dict = {}
    final = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha_mode="reliability_sc_corr",
        reliability_use_sample_weight=True,
        reliability_use_advantage_weight=True,
        diagnostics=diagnostics,
    )
    diagnostics["mv_votes"] = [
        {"selected_answer": str(index), "answer_clusters": {str(index): 1.0}}
        for index in range(num_prompts)
    ]
    diagnostics["mv_extracted_answers"] = [
        [str(index)] * num_generations for index in range(num_prompts)
    ]
    payload = _build_reliability_diagnostics_payload(
        step=9,
        prompts=[f"prompt {index}" for index in range(num_prompts)],
        sampled_indices=list(range(1000, 1000 + num_prompts)),
        num_generations=num_generations,
        diagnostics=diagnostics,
        training_advantages=final.reshape(-1),
        parameters=_parameters(use_sample=True, use_advantage=True),
    )
    path = _write_reliability_diagnostics_if_main(
        is_main_process=True,
        base_dir=str(tmp_path),
        step=9,
        payload=payload,
    )
    loaded = json.loads(Path(path).read_text())
    assert len(loaded["batch_reliability"]) == num_prompts
    assert len(loaded["batch_reliability_components"]["raw_sc"]) == num_prompts
    assert len(loaded["batch_weights"]["applied_sample"]) == num_prompts
    assert len(loaded["reliability_ranking_desc"]) == num_prompts
    assert len(loaded["reliability_ranking_asc"]) == num_prompts
    assert len(loaded["records"]) == num_prompts
    assert {row["prompt_index"] for row in loaded["reliability_ranking_desc"]} == set(
        range(num_prompts)
    )


def test_reliability_diagnostics_are_required_without_debug_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CO_DEBUG_DATA", "0")
    assert _requires_reliability_diagnostics("mixed_reward", "reliability_sc_corr")
    assert not _requires_reliability_diagnostics("mixed_reward", "fixed")
    assert not _requires_reliability_diagnostics("mv", "reliability_sc_corr")


@pytest.mark.parametrize("mode", ["reliability_sc_corr", "prism", "ettrl"])
def test_required_mixed_reward_diagnostics_include_new_modes(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CO_DEBUG_DATA", "0")
    assert _requires_mixed_reward_diagnostics("mixed_reward", mode)
    assert not _requires_mixed_reward_diagnostics("mv", mode)


@pytest.mark.parametrize("mode", ["prism", "ettrl"])
def test_prism_ettrl_payload_and_writer_are_complete(
    mode: str,
    tmp_path: Path,
) -> None:
    mv, self_certainty = _inputs()
    diagnostics: dict = {}
    final = compute_mixed_reward_advantages(
        mv,
        self_certainty,
        alpha_mode=mode,
        final_normalize=False,
        diagnostics=diagnostics,
    )
    diagnostics["mv_votes"] = [
        {"selected_answer": str(index), "answer_clusters": {str(index): 1.0}}
        for index in range(mv.size(0))
    ]
    diagnostics["mv_extracted_answers"] = [
        [str(index)] * mv.size(1) for index in range(mv.size(0))
    ]
    parameters = {
        "reward_mode": "mixed_reward",
        "alpha_mode": mode,
        "alpha": 0.5,
        "corr_weight_mode": "none",
        "final_normalize": False,
        "ettrl_all_zero_group_fallback": "unit_weight",
    }
    payload = _build_prism_ettrl_diagnostics_payload(
        step=4,
        prompts=["p0", "p1", "p2"],
        sampled_indices=[10, 20, 30],
        num_generations=4,
        diagnostics=diagnostics,
        training_advantages=final.reshape(-1),
        parameters=parameters,
    )

    assert payload["alpha_mode"] == mode
    assert len(payload["batch_self_certainty_group_mean"]) == 3
    assert len(payload["batch_weights"]["mv_advantage"]) == 3
    assert len(payload["batch_weights"]["mv_advantage"][0]) == 4
    assert len(payload["records"]) == 3
    assert payload["records"][0]["mixed"]["final_advantages"] == payload[
        "records"
    ][0]["mixed"]["before_final_normalize"]
    if mode == "ettrl":
        assert payload["batch_ettrl_all_zero_group"] == [False, False, False]
        assert payload["records"][0]["self_certainty"]["all_zero_group"] is False
    else:
        assert "batch_ettrl_all_zero_group" not in payload

    path = _write_prism_ettrl_diagnostics_if_main(
        is_main_process=True,
        base_dir=str(tmp_path),
        step=4,
        alpha_mode=mode,
        payload=payload,
    )
    assert path is not None
    artifact = Path(path)
    assert artifact.name == f"mixed_reward_{mode}_step4.json"
    assert json.loads(artifact.read_text())["alpha_mode"] == mode


@pytest.mark.parametrize("mode", ["prism", "ettrl"])
def test_single_training_tensor_builder_threads_new_mode_and_final_norm(
    mode: str,
) -> None:
    prompts = ["first problem", "second problem"]
    completions = [
        "answer \\boxed{1}",
        "again \\boxed{1}",
        "different \\boxed{2}",
        "missing box",
        "answer \\boxed{3}",
        "again \\boxed{3}",
        "third \\boxed{3}",
        "different \\boxed{4}",
    ]
    scores = [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]
    diagnostics: dict = {}
    training_data = _build_training_tensors(
        tokenizer=_TinyTokenizer(),
        prompts=prompts,
        completions=completions,
        scores=scores,
        num_generations=4,
        reward_mode="mixed_reward",
        reward_length_norm="response",
        step=2,
        max_prompt_length=64,
        max_completion_length=64,
        mask_truncated_completions=False,
        scale_rewards=True,
        mixed_reward_alpha=0.5,
        mixed_reward_alpha_mode=mode,
        mixed_reward_final_normalize=False,
        mixed_reward_corr_weight_mode="none",
        finish_reasons=["stop"] * 8,
        old_logps=None,
        ref_logps=None,
        mixed_reward_diagnostics=diagnostics,
    )

    assert diagnostics["alpha_mode"] == mode
    assert diagnostics["final_normalize"] is False
    assert torch.equal(
        diagnostics["final_advantages"].reshape(-1),
        training_data["advantages"],
    )
    assert torch.equal(
        diagnostics["combined_before_final_normalize"],
        diagnostics["combined_after_final_normalize"],
    )


def test_checkpoint_info_loader_is_step_checked_and_defensive(tmp_path: Path) -> None:
    cache = CoIntuitorCache(str(tmp_path / "cache.json"))
    cache.save_checkpoint_info(3, 20, [7, 8, 9], sampling_strategy="open_r1")
    loaded = cache.load_checkpoint_info(3)
    assert loaded is not None
    assert loaded["sampled_indices"] == [7, 8, 9]
    loaded["sampled_indices"][0] = 999
    assert cache.load_checkpoint_info(3)["sampled_indices"] == [7, 8, 9]
    assert cache.load_checkpoint_info(4) is None
