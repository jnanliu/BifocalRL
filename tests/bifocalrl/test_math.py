import json
from pathlib import Path

import pytest
import torch

from openrlhf.algorithms.bifocalrl.methods import (
    compute_advantages,
    covo_curiosity,
    covo_features,
    covo_rewards,
    ettrl_advantages,
    prism_reward,
)
from openrlhf.algorithms.bifocalrl.mixed_reward import MixedRewardConfig, compute_mixed_reward
from openrlhf.algorithms.bifocalrl.rollout import parse_judgements, reasoning_steps
from openrlhf.algorithms.bifocalrl.stats import vocabulary_statistics


@pytest.mark.parametrize("chunk", [1, 3, 256])
@pytest.mark.parametrize("vocab", [1, 7, 11])
def test_full_vocabulary_statistics(chunk, vocab):
    logits = torch.randn(8, 11) * 30
    logits[:, vocab:] = 1e10  # padding must not influence either statistic
    sc, entropy = vocabulary_statistics(logits, vocab_size=vocab, chunk_size=chunk)
    lp = logits[:, :vocab].log_softmax(-1)
    torch.testing.assert_close(sc, -lp.mean(-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy, -(lp.exp() * lp).sum(-1), atol=1e-5, rtol=1e-5)


def test_migration_matches_original_fusion_snapshots():
    # Snapshots were computed with the parent BifocalRL core before migration.
    data = json.loads(Path(__file__).with_name("parity.json").read_text())
    mv, sc = torch.tensor(data["mv"]), torch.tensor(data["sc"])
    for case in data["cases"]:
        actual = compute_mixed_reward(mv, sc, config=MixedRewardConfig(**case["config"]))
        torch.testing.assert_close(actual, torch.tensor(case["expected"]), rtol=0, atol=0)


def test_ettrl_shaping_is_entropy_and_not_legacy_sce_weighting():
    mv = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    h = torch.tensor([[1.0, 2.0, 3.0, 2.0]])
    base = (mv - mv.mean()) / (mv.std(unbiased=False) + 1e-6)
    torch.testing.assert_close(ettrl_advantages(mv, h, beta=1), base.clamp(-1, 1) * (2 - h / h.mean()))
    torch.testing.assert_close(ettrl_advantages(mv, torch.zeros_like(h), mode="res"), base)
    torch.testing.assert_close(ettrl_advantages(torch.ones_like(mv), h), torch.zeros_like(mv))


def test_prism_min_harmonic_and_separate_normalization():
    assert prism_reward([0.9, 0.3, 0.7], 0.6) == pytest.approx(0.4)
    assert prism_reward([0], 0) == 0
    with pytest.raises(ValueError):
        prism_reward([], 0.5)
    sc = torch.tensor([[3.0, 4.0, 6.0]])
    prm = torch.tensor([[0.1, 0.8, 0.5]])
    expected = 0.3 * (sc - sc.mean()) / (sc.std() + 1e-6) + (prm - prm.mean()) / (prm.std() + 1e-6)
    actual = compute_advantages("prism", torch.zeros_like(sc), sc, prm=prm, gamma=0.3)
    torch.testing.assert_close(actual, expected)


def test_covo_consistency_volatility_and_cluster_denominator():
    distances = [[1, 3], [3, 1], [1, 2], [1, 4]]
    assert covo_features(distances, 0) == (0.75, 0.25)
    assert covo_features(distances, 0, released=True) == (0.75, 0.5)
    assert covo_features([[1, 1]], 0) == (1.0, 0.0)
    assert covo_features([[1, 1]], 0, released=True) == (0.0, 0.0)
    rewards = covo_rewards([(1, 0), (0.5, 0), (0, 0)], [0, 0, 1], [0, 0, 0])
    assert rewards == [0.75, 0.75, 0.0]
    assert covo_curiosity([[-2.0, -2.0]]) == pytest.approx(2)
    released = covo_rewards([(0.6, 0), (0.2, 0)], [0, 1], [3, 1], released=True)
    assert released == [1.4, 0.0]


def test_prm_parsing_rejects_missing_completion():
    text = r"Paragraph 1 **Judgement**: $\boxed{Yes}$" + "\n" + r"Completion **Judgement**: $\boxed{No}$"
    paragraphs, completion = parse_judgements(text)
    assert len(paragraphs) == 1 and completion.group(2) == "No"
    with pytest.raises(ValueError):
        parse_judgements(text.split("\n")[0])
    assert len(reasoning_steps("A\nB\nC\nD\n" + r"\boxed{1}", 2)) == 2
