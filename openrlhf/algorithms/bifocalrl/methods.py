"""Pure tensor/scalar algorithms; no Ray, SGLang or Megatron dependencies."""

import math
from collections import defaultdict
from dataclasses import replace

import torch
import torch.distributed as dist

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


def covo_features(distances, own):
    d = torch.as_tensor(distances, dtype=torch.float64)
    if d.ndim != 2 or not d.shape[0] or not 0 <= own < d.shape[1] or not torch.isfinite(d).all():
        raise ValueError("CoVo requires a finite [states, candidate answers] distance matrix")
    best = d[:, own] == d.min(-1).values
    con = float(best.double().mean())
    deviations = (~best).nonzero().flatten()
    vol = float(deviations[-1]) / len(d) if len(deviations) else 0.0
    return con, vol


def covo_curiosity(step_logprobs):
    """Eq. 10; normalize observed token probabilities across a reasoning step."""
    scores = []
    for values in step_logprobs:
        lp = torch.as_tensor(values, dtype=torch.float64)
        if not lp.numel() or not torch.isfinite(lp).all() or (lp > 1e-6).any():
            raise ValueError("CoVo step log probabilities must be finite and nonpositive")
        p = lp.softmax(0)
        kl = (p * (p.clamp_min(1e-300).log() + math.log(len(p)))).sum().clamp_min(0).item()
        surprise = -lp.mean().item()
        scores.append(surprise - math.log1p(kl))
    return sum(scores) / len(scores) if scores else 0.0


def covo_rewards(features, answer_ids, curiosities, *, aggregation="vector", curiosity_weight=1.0):
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
        for i in indices:
            rewards[i] = intrinsic
    for i, curiosity in enumerate(curiosities):
        rewards[i] += curiosity_weight * curiosity
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
    cluster_weight=0.5,
    certainty_weight=0.5,
    diagnostics=None,
    metadata=None,
):
    if method == "ttrl":
        return group_standardize(mv)
    if method == "intuitor":
        return group_standardize(sc)
    if method in ("bifocalrl", "reliability_sc_corr"):
        config = mixed or MixedRewardConfig()
        config = replace(config, alpha_mode=method)
        return compute_mixed_reward(mv, sc, config=config, diagnostics=diagnostics, metadata=metadata)
    if method == "circa":
        if prm is None:
            raise ValueError("CIRCA requires cluster/format scores")
        return cluster_weight * group_standardize(prm) + certainty_weight * group_standardize(sc)
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


def kl_cov_policy_loss(
    log_probs,
    old_log_probs,
    advantages,
    action_mask,
    *,
    clip_low=0.2,
    clip_high=0.2,
    fraction=0.02,
    coefficient=0.05,
    group=None,
):
    """Eqs. 6--8: rank valid tokens across the DP micro-batch, then replace clipping.

    Selection is detached and deterministic under ties. Gradient accumulation
    accumulates these losses; it does not enlarge the covariance-selection batch.
    """
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Invalid KL-Cov fraction or coefficient")
    mask = action_mask.bool()
    if any(t.shape != log_probs.shape for t in (old_log_probs, advantages, mask)):
        raise ValueError("KL-Cov inputs must have the same token shape")
    values = torch.stack((log_probs.detach()[mask].float(), advantages.detach()[mask].float()), dim=-1)
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    sizes = [len(values)]
    pooled = values
    if world > 1:
        size = torch.tensor([len(values)], dtype=torch.long, device=values.device)
        gathered_sizes = [torch.empty_like(size) for _ in range(world)]
        dist.all_gather(gathered_sizes, size, group=group)
        sizes = [int(x.item()) for x in gathered_sizes]
        padded = values.new_zeros((max(1, max(sizes)), 2))
        padded[: len(values)] = values
        gathered = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(gathered, padded, group=group)
        pooled = torch.cat([x[:n] for x, n in zip(gathered, sizes, strict=True)])
    if not torch.isfinite(pooled).all():
        raise ValueError("Nonfinite KL-Cov log probability or advantage")
    selected = torch.zeros(len(pooled), dtype=torch.bool, device=values.device)
    if len(pooled) and fraction > 0:
        centered = pooled - pooled.mean(0)
        covariance = centered[:, 0] * centered[:, 1]
        count = max(1, int(fraction * len(pooled)))
        selected[covariance.argsort(descending=True, stable=True)[:count]] = True
    local_selected = torch.zeros_like(mask)
    start = sum(sizes[:rank])
    local_selected[mask] = selected[start : start + len(values)]
    log_ratio = torch.where(mask, log_probs.float() - old_log_probs.detach().float(), 0.0)
    ratio = log_ratio.clamp(-20, 20).exp()
    adv = torch.where(mask, advantages.detach().float(), 0.0)
    surrogate = ratio * adv
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high) * adv
    loss = torch.where(local_selected, -surrogate + coefficient * log_ratio.abs(), -torch.minimum(surrogate, clipped))
    return torch.where(mask, loss, 0.0), local_selected
