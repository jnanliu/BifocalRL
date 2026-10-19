"""Pure tensor/scalar algorithms; no Ray, SGLang or Megatron dependencies."""

import math
from collections import defaultdict

import torch

from .mixed_reward import MixedRewardConfig, compute_mixed_reward
from .normalization import grpo_group_advantages


def group_standardize(x, *, population=False, eps=1e-6):
    if not population:
        return grpo_group_advantages(x, eps=eps)
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, unbiased=False, keepdim=True) + eps)


def ettrl_advantages(mv, entropy, *, mode="clip_res", beta=2.0, res_clip=None, eps=1e-6):
    """ETTRL Eqs. 2, 12--15: AdvClip followed by entropy AdvRes."""
    if beta <= 0 or not torch.isfinite(entropy).all() or (entropy < 0).any():
        raise ValueError("ETTRL needs beta > 0 and finite nonnegative Shannon entropy")
    advantage = group_standardize(mv, population=True, eps=eps)
    if mode in ("clip", "clip_res"):
        advantage = advantage.clamp(-beta, beta)
    if mode in ("res", "clip_res"):
        mean = entropy.mean(-1, keepdim=True)
        adjustment = torch.where(mean > eps, (mean - entropy) / mean.clamp_min(eps), torch.zeros_like(entropy))
        if res_clip is not None:
            adjustment = adjustment.clamp(-res_clip, res_clip)
        advantage = advantage * (1 + adjustment)
    if mode not in ("none", "clip", "res", "clip_res"):
        raise ValueError(f"Unknown EAR mode: {mode}")
    return advantage


def prism_reward(step_probabilities, completion_probability):
    """Min process aggregation, harmonically combined with completion P(Yes)."""
    values = list(step_probabilities) + [completion_probability]
    if len(values) < 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("PRISM requires at least one process probability and a completion probability in [0,1]")
    process = min(step_probabilities)
    return (
        2 * process * completion_probability / (process + completion_probability)
        if process + completion_probability
        else 0.0
    )


def covo_features(distances, own, *, released=False):
    d = torch.as_tensor(distances, dtype=torch.float64)
    if d.ndim != 2 or not d.shape[0] or not 0 <= own < d.shape[1] or not torch.isfinite(d).all():
        raise ValueError("CoVo requires a finite [states, candidate answers] distance matrix")
    best = d[:, own] == d.min(-1).values
    if released:
        # The public implementation uses strict preference for Con and a
        # one-based last deviation for Vol; the paper uses Eqs. 4 and 5.
        others = torch.cat([d[:, :own], d[:, own + 1 :]], -1)
        if not others.shape[1]:
            return 0.0, 0.0
        con = float((d[:, own] < others.min(-1).values).double().mean())
    else:
        con = float(best.double().mean())
    deviations = (~best).nonzero().flatten()
    vol = float(deviations[-1] + int(released)) / len(d) if len(deviations) else 0.0
    return con, vol


def covo_curiosity(step_logprobs, *, released=False):
    """Eq. 10; normalize observed token probabilities across a reasoning step.

    The released code instead normalizes negative log probabilities, uses
    perplexity and a cube-root KL penalty. Both conventions are explicit.
    """
    scores = []
    for values in step_logprobs:
        lp = torch.as_tensor(values, dtype=torch.float64)
        if not lp.numel() or not torch.isfinite(lp).all() or (lp > 1e-6).any():
            raise ValueError("CoVo step log probabilities must be finite and nonpositive")
        p = (-lp / (-lp).sum()) if released and (-lp).sum() > 0 else lp.softmax(0)
        kl = (p * (p.clamp_min(1e-300).log() + math.log(len(p)))).sum().clamp_min(0).item()
        surprise = -lp.mean().item()
        scores.append(
            math.exp(min(surprise, 700)) - math.log1p(kl ** (1 / 3)) if released else surprise - math.log1p(kl)
        )
    return sum(scores) / len(scores) if scores else 0.0


def covo_rewards(features, answer_ids, curiosities, *, aggregation="vector", released=False, curiosity_weight=1.0):
    clusters = defaultdict(list)
    for i, answer in enumerate(answer_ids):
        if answer is not None:
            clusters[answer].append(i)
    rewards = [0.0] * len(features)
    for indices in clusters.values():
        cs, vs = zip(*(features[i] for i in indices), strict=True)
        if aggregation == "vector":
            intrinsic = math.hypot(
                sum(c * math.cos(v) for c, v in zip(cs, vs, strict=True)),
                sum(c * math.sin(v) for c, v in zip(cs, vs, strict=True)),
            ) / len(indices)
        elif aggregation == "linear":
            intrinsic = sum(c - v for c, v in zip(cs, vs, strict=True)) / len(indices)
        else:
            raise ValueError(f"Unknown CoVo aggregation: {aggregation}")
        if released:
            intrinsic = float(intrinsic >= (0.5 if aggregation == "vector" else 0.05))
        for i in indices:
            rewards[i] = intrinsic
    for i, curiosity in enumerate(curiosities):
        rewards[i] += curiosity_weight * (0.4 * float(curiosity > 2) if released else curiosity)
    return rewards


def compute_advantages(
    method,
    mv,
    sc,
    *,
    mixed=None,
    entropy=None,
    prm=None,
    gamma=1.0,
    ear_mode="clip_res",
    ear_beta=2.0,
    ear_res_clip=None,
):
    if method == "ttrl":
        return group_standardize(mv)
    if method == "intuitor":
        return group_standardize(sc)
    if method in ("mixed", "legacy_prism", "legacy_ettrl"):
        config = mixed or MixedRewardConfig()
        if method.startswith("legacy_"):
            from dataclasses import replace

            config = replace(config, alpha_mode=method.removeprefix("legacy_"), final_normalize=False)
        return compute_mixed_reward(mv, sc, config=config)
    if method == "ettrl":
        return ettrl_advantages(mv, entropy, mode=ear_mode, beta=ear_beta, res_clip=ear_res_clip)
    if method == "prism":
        if prm is None:
            raise ValueError("PRISM requires real PRM scores")
        return gamma * group_standardize(sc) + group_standardize(prm)
    if method == "covo":
        # Reinforce++ (gamma=1, zero reward KL): outcome return, whitened
        # globally over trainable tokens in the framework adapter.
        return prm
    raise ValueError(f"Unknown BifocalRL method: {method}")
