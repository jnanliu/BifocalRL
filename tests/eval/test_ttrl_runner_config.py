from __future__ import annotations

from eval.ttrl.runner import _parse_dataset_overrides


def test_dataset_overrides_support_per_dataset_generation_length() -> None:
    overrides = _parse_dataset_overrides(
        '{"aime24":{"max_new_tokens":4096,"temperature":0.6},'
        '"gsm8k":{"max_new_tokens":2048,"unknown":true}}'
    )

    assert overrides == {
        "aime24": {"max_new_tokens": 4096, "temperature": 0.6},
        "gsm8k": {"max_new_tokens": 2048},
    }
