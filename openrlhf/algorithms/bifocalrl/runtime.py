"""OpenRLHF configuration and global rollout advantage construction."""

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .methods import compute_advantages
from .mixed_reward import MixedRewardConfig

logger = logging.getLogger(__name__)
METHODS = ("ttrl", "intuitor", "bifocalrl", "reliability_sc_corr", "ettrl", "covo", "prism", "circa")
DEFAULTS = {
    "method": "bifocalrl",
    "mixed": asdict(MixedRewardConfig()),
    "ear_mode": "clip_res",
    "ear_beta": 2.0,
    "ear_res_clip": None,
    "tree_enabled": True,
    "tree_roots": 2,
    "tree_forks": 3,
    "tree_branches": 1,
    "prism_gamma": 1.0,
    "prism_gamma_schedule": "constant",
    "prism_gamma_steps": None,
    "prm_url": None,
    "prm_model": "GenPRM/GenPRM-7B",
    "prm_function": None,
    "prm_max_tokens": 2048,
    "prm_prompt_file": None,
    "covo_aggregation": "vector",
    "covo_curiosity_weight": 1.0,
    "covo_max_steps": 15,
    "circa_cluster_weight": 0.5,
    "circa_certainty_weight": 0.5,
    "circa_format_weight": 0.1,
    "kl_cov_fraction": 0.02,
    "kl_cov_coef": 0.05,
    "score_concurrency": 16,
    "voting_seed": 42,
    "eval_label_key": "label",
    "monitor_eval": True,
    "monitor_train_labels": False,
    "sandbox_fusion_url": None,
    "sandbox_timeout": 10,
    "sandbox_memory_limit_mb": 1024,
}


def add_arguments(parser):
    group = parser.add_argument_group("BifocalRL unsupervised algorithms")
    group.add_argument("--bifocal.config", help="JSON recipe; omitted means native OpenRLHF")
    group.add_argument("--bifocal.method", type=str.lower, choices=METHODS)
    group.add_argument("--bifocal.prm_url", help="Independent GenPRM SGLang endpoint")
    group.add_argument("--bifocal.prm_model", help="PRM tokenizer path matching the endpoint")
    group.add_argument(
        "--bifocal.sandbox_fusion_url", "--eval.sandbox_fusion_url", dest="eval.sandbox_fusion_url",
        help="Sandbox Fusion /run_code endpoint for code evaluation",
    )


def configure(args):
    if not (args.bifocal.config or args.bifocal.method):
        args.bifocal.settings = None
        return
    cfg = dict(DEFAULTS)
    data = json.loads(Path(args.bifocal.config).read_text()) if args.bifocal.config else {}
    if data.keys() - cfg.keys():
        raise ValueError(f"Unknown BifocalRL settings: {sorted(data.keys() - cfg.keys())}")
    cfg.update(data)
    cfg["mixed"] = {**DEFAULTS["mixed"], **data.get("mixed", {})}
    for key in ("method", "prm_url", "prm_model", "sandbox_fusion_url"):
        if getattr(args.bifocal, key, None) is not None:
            cfg[key] = getattr(args.bifocal, key)
    eval_url = getattr(getattr(args, "eval", None), "sandbox_fusion_url", None)
    if eval_url is not None:
        cfg["sandbox_fusion_url"] = eval_url
    cfg["method"] = cfg["method"].lower()
    if cfg["method"] not in METHODS:
        raise ValueError(f"Unknown BifocalRL method: {cfg['method']}")
    if cfg["method"] in ("bifocalrl", "reliability_sc_corr"):
        mode = cfg["method"]
        if data.get("mixed", {}).get("alpha_mode", mode) != mode:
            raise ValueError(f"Method {cfg['method']} requires mixed.alpha_mode={mode!r}")
        cfg["mixed"]["alpha_mode"] = mode
    MixedRewardConfig(**cfg["mixed"])
    for key in ("monitor_eval", "monitor_train_labels"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if not math.isfinite(cfg["sandbox_timeout"]) or cfg["sandbox_timeout"] <= 0:
        raise ValueError("sandbox_timeout must be finite and positive")
    if not isinstance(cfg["sandbox_memory_limit_mb"], int) or cfg["sandbox_memory_limit_mb"] < 1:
        raise ValueError("sandbox_memory_limit_mb must be a positive integer")
    if args.rollout.n_samples_per_prompt < 2 or args.vllm.num_engines < 1:
        raise ValueError("BifocalRL requires vLLM and at least two responses per prompt")
    if args.train.async_enable or args.train.agent_func_path or args.data.max_images_per_prompt:
        raise ValueError("BifocalRL currently supports synchronous text-only, single-turn training")
    if args.reward.remote_url or args.reward.model_name_or_path or args.critic.model_name_or_path:
        raise ValueError("BifocalRL owns the intrinsic reward; omit reward/critic model overrides")
    if args.algo.kl.init_coef and not args.algo.kl.use_loss:
        raise ValueError("Use --algo.kl.use_loss for reference KL, or set --algo.kl.init_coef 0")
    if args.algo.advantage.gamma != 1:
        raise ValueError("BifocalRL outcome advantages require gamma=1")
    if args.reward.overlong_buffer_len or args.reward.stop_properly_penalty_coef:
        raise ValueError("Length reward penalties change these baselines; omit them")
    if args.ds.tensor_parallel_size > 1 and cfg["method"] not in ("ttrl", "covo"):
        raise ValueError("Full-vocabulary actor statistics currently require DeepSpeed tensor_parallel_size=1")
    if cfg["method"] == "prism" and not (cfg["prm_url"] or cfg["prm_function"]):
        raise ValueError("PRISM requires a real prm_url or prm_function")
    if cfg["prism_gamma_schedule"] not in ("constant", "quadratic"):
        raise ValueError("Invalid PRISM gamma schedule")
    if cfg["prism_gamma_steps"] is not None and cfg["prism_gamma_steps"] < 2:
        raise ValueError("prism_gamma_steps must be at least 2")
    for key in ("prism_gamma", "covo_curiosity_weight", "ear_beta"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["ear_mode"] not in ("none", "clip", "res", "clip_res") or cfg["ear_beta"] <= 0:
        raise ValueError("Invalid EAR clipping settings")
    if cfg["ear_res_clip"] is not None and cfg["ear_res_clip"] <= 0:
        raise ValueError("ear_res_clip must be positive or null")
    if cfg["covo_aggregation"] not in ("vector", "linear"):
        raise ValueError("Invalid CoVo aggregation")
    for key in (
        "circa_cluster_weight",
        "circa_certainty_weight",
        "circa_format_weight",
        "kl_cov_coef",
    ):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["circa_cluster_weight"] + cfg["circa_certainty_weight"] == 0:
        raise ValueError("CIRCA channel weights cannot both be zero")
    if not math.isfinite(cfg["kl_cov_fraction"]) or not 0 <= cfg["kl_cov_fraction"] <= 1:
        raise ValueError("kl_cov_fraction must be in [0, 1]")
    if cfg["method"] == "circa" and (
        args.actor.policy_loss_type != "ppo"
        or args.actor.dual_clip is not None
        or args.actor.loss_agg_mode != "token-mean"
        or args.algo.advantage.is_correction_level != "off"
    ):
        raise ValueError("CIRCA KL-Cov requires token-mean PPO without dual clipping or IS correction")
    if cfg["covo_max_steps"] < 1 or cfg["score_concurrency"] < 1:
        raise ValueError("Scoring limits must be positive")
    if cfg["method"] == "ettrl" and cfg["tree_enabled"]:
        m, n, b = (cfg[k] for k in ("tree_roots", "tree_forks", "tree_branches"))
        if any(not isinstance(x, int) or x < 1 for x in (m, n, b)) or args.rollout.n_samples_per_prompt != m * (
            1 + n * b
        ):
            raise ValueError("ETMR requires n_samples_per_prompt = M*(1+N*B)")
        if args.rollout.top_p != 1:
            raise ValueError("ETMR currently requires rollout.top_p=1")
    cfg["voting_seed"] = data.get("voting_seed", args.train.seed)
    args.bifocal.settings = cfg
    args.algo.advantage.estimator = "reinforce" if cfg["method"] == "covo" else "group_norm"
    args.reward.remote_url = "agent"
    print("BifocalRL configuration: " + json.dumps(cfg, sort_keys=True))


@torch.no_grad()
def group_monitor_metrics(cluster_ids, valid, sc=None, advantages=None, correct=None, keep=None):
    """Aggregate prompt-local answer clusters; conditional means omit empty populations."""
    selected = torch.ones_like(valid, dtype=torch.bool) if keep is None else keep.bool()
    valid = valid.bool() & selected
    counts = torch.zeros_like(cluster_ids, dtype=torch.float32)
    counts.scatter_add_(1, cluster_ids.long(), valid.float())
    occupied = counts > 0
    sizes = counts[occupied]
    totals = counts.sum(-1)
    active = totals > 0
    metrics = {
        "query_count": float(len(counts)),
        "selected_response_count": selected.sum().item(),
        "valid_response_count": valid.sum().item(),
        "cluster_count": occupied.sum().item(),
        "clusters_per_query": occupied.sum(-1).float().mean().item(),
        "empty_query_fraction": (~active).float().mean().item(),
    }
    if selected.any():
        metrics["parse_failure_rate"] = (~valid & selected).sum().item() / selected.sum().item()
    if active.any():
        probabilities = counts[active] / totals[active, None]
        metrics["largest_cluster_fraction"] = probabilities.max(-1).values.mean().item()
        metrics["answer_entropy"] = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean().item()
        metrics["cluster_size_mean"] = sizes.mean().item()
        metrics["singleton_cluster_fraction"] = (sizes == 1).float().mean().item()
        metrics["singleton_response_fraction"] = (sizes == 1).sum().item() / valid.sum().item()
        for lower, upper, name in ((2, 2, "2"), (3, 4, "3_4"), (5, 8, "5_8"), (9, float("inf"), "9_plus")):
            metrics[f"cluster_size_fraction_{name}"] = ((sizes >= lower) & (sizes <= upper)).float().mean().item()
    if sc is not None and valid.any():
        sums = torch.zeros_like(counts).scatter_add_(1, cluster_ids.long(), sc.masked_fill(~valid, 0))
        means = sums / counts.clamp_min(1)
        within = (sc - means.gather(1, cluster_ids.long()))[valid]
        metrics["sc_within_rms"] = within.square().mean().sqrt().item()
        metrics["sc_within_abs_mean"] = within.abs().mean().item()
    if correct is not None:
        labeled = valid & (correct >= 0)
        metrics["labeled_response_count"] = labeled.sum().item()
        for value, name in ((1, "correct"), (0, "incorrect")):
            mask = labeled & (correct == value)
            class_counts = torch.zeros_like(counts).scatter_add_(1, cluster_ids.long(), mask.float())
            # A cluster is classifiable only when all its selected members share a known label.
            clusters = occupied & (class_counts == counts)
            metrics[f"{name}_response_count"] = mask.sum().item()
            metrics[f"{name}_cluster_count"] = clusters.sum().item()
            if sc is not None and mask.any():
                metrics[f"{name}_sc_mean"] = sc[mask].mean().item()
                if clusters.any():
                    metrics[f"{name}_cluster_sc_mean"] = means[clusters].mean().item()
            if advantages is not None and mask.any():
                metrics[f"{name}_positive_advantage_fraction"] = (advantages[mask] > 0).float().mean().item()
                if clusters.any():
                    adv_sums = torch.zeros_like(counts).scatter_add_(1, cluster_ids.long(), advantages.masked_fill(~valid, 0))
                    metrics[f"{name}_positive_cluster_advantage_fraction"] = (adv_sums[clusters] > 0).float().mean().item()
    return metrics


@torch.no_grad()
def assign_advantages(maker, experiences):
    """Restore original rollout order before query and batch normalization."""
    args = maker.strategy.args
    cfg = args.bifocal.settings
    sizes = [len(exp.index) for exp in experiences]
    indices = torch.tensor([i for exp in experiences for i in exp.index], dtype=torch.long)
    if sorted(indices.tolist()) != list(range(len(indices))):
        raise ValueError("BifocalRL needs each rollout sample exactly once before advantage computation")
    group_size = args.rollout.n_samples_per_prompt
    if len(indices) % group_size:
        raise ValueError("Incomplete BifocalRL prompt group")
    scores = {}
    maker.bifocal_monitor_metrics = {}
    score_names = ("mv", "sc", "entropy", "aux")
    monitor = all("bifocal_answer_valid" in exp.info for exp in experiences)
    if cfg["method"] == "bifocalrl" or monitor:
        score_names += ("answer_cluster",)
    if monitor:
        score_names += ("answer_valid", "monitor_correct")
    for name in score_names:
        key = f"bifocal_{name}"
        required = (
            name in ("mv", "answer_cluster", "answer_valid", "monitor_correct")
            or (name in ("sc", "entropy") and cfg["method"] not in ("ttrl", "covo"))
            or (name == "aux" and cfg["method"] in ("prism", "covo", "circa"))
        )
        parts = []
        for exp in experiences:
            if required and key not in exp.info:
                raise ValueError(f"Missing actor/rollout statistic {key}")
            parts.append(exp.info.get(key, torch.zeros(len(exp.index))).float())
        ordered = torch.empty(len(indices), dtype=torch.float32)
        ordered[indices] = torch.cat(parts)
        if not torch.isfinite(ordered).all():
            raise ValueError(f"Nonfinite {key}")
        if name == "answer_cluster":
            if (ordered != ordered.trunc()).any():
                raise ValueError("bifocal_answer_cluster must contain integer IDs")
            ordered = ordered.long()
        scores[name] = ordered.reshape(-1, group_size)
    gamma = cfg["prism_gamma"]
    if cfg["prism_gamma_schedule"] == "quadratic":
        steps = cfg["prism_gamma_steps"] or maker.bifocal_total_steps
        gamma *= max(0.0, 1 - maker.bifocal_step / max(1, steps - 1)) ** 2
    diagnostics = {} if cfg["method"] == "bifocalrl" else None
    advantages = (
        compute_advantages(
            cfg["method"],
            scores["mv"],
            scores["sc"],
            mixed=MixedRewardConfig(**cfg["mixed"]),
            entropy=scores["entropy"],
            prm=scores["aux"],
            gamma=gamma,
            ear_mode=cfg["ear_mode"],
            ear_beta=cfg["ear_beta"],
            ear_res_clip=cfg["ear_res_clip"],
            cluster_weight=cfg["circa_cluster_weight"],
            certainty_weight=cfg["circa_certainty_weight"],
            diagnostics=diagnostics,
            metadata={"answer_cluster_ids": scores.get("answer_cluster")},
        )
        .flatten()[indices]
        .split(sizes)
    )
    for exp, value in zip(experiences, advantages, strict=True):
        mask = exp.action_mask.float()
        if (mask.sum(-1) == 0).any():
            raise ValueError("BifocalRL cannot train an empty/masked response")
        exp.advantages = value[:, None].expand_as(mask).clone() * mask
        exp.returns = exp.advantages.clone()
        exp.kl = None
        if cfg["method"] == "intuitor":
            exp.rewards = scores["sc"].flatten()[exp.index]
            exp.info["reward"] = exp.rewards.clone()
        exp.info["return"] = exp.rewards.float()
        exp.info["group_reward_std"] = scores["mv"].std(-1).repeat_interleave(group_size)[exp.index]
        exp.info["bifocal_gamma"] = torch.full_like(value, gamma)
        if diagnostics is not None:
            for name in (
                "mv_weight", "snr", "snr_threshold",
                "rollout_keep", "retained_fraction", "query_weight",
                "sc_between_gamma", "sc_cluster_mean", "sc_cluster_size", "sc_within", "sc_adjusted",
            ):
                ordered = diagnostics[name].expand_as(scores["mv"]).flatten()
                exp.info[f"bifocal_{name}"] = ordered[exp.index]
    if cfg["method"] == "covo":
        # Match the slime adapter's Reinforce++ convention, including Bessel correction.
        count = sum(exp.action_mask.sum() for exp in experiences).float()
        mean = sum((exp.advantages * exp.action_mask).sum() for exp in experiences) / count
        variance = sum(((exp.advantages - mean).square() * exp.action_mask).sum() for exp in experiences) / (
            count - 1
        ).clamp_min(1)
        for exp in experiences:
            exp.advantages = (exp.advantages - mean) * (variance + 1e-8).rsqrt() * exp.action_mask
            exp.returns = exp.advantages.clone()
    if monitor:
        final_advantages = torch.empty(len(indices))
        for exp in experiences:
            final_advantages[exp.index] = (exp.advantages * exp.action_mask).sum(-1) / exp.action_mask.sum(-1)
        monitor_args = dict(
            cluster_ids=scores["answer_cluster"], valid=scores["answer_valid"],
            sc=scores["sc"] if cfg["method"] not in ("ttrl", "covo") else None,
            advantages=final_advantages.reshape(-1, group_size), correct=scores["monitor_correct"],
        )
        before = group_monitor_metrics(**monitor_args)
        after = group_monitor_metrics(**monitor_args, keep=diagnostics["rollout_keep"] if diagnostics else None)
        maker.bifocal_monitor_metrics = {
            **{f"bifocal_monitor/before/{key}": value for key, value in before.items()},
            **{f"bifocal_monitor/retained/{key}": value for key, value in after.items()},
        }
        if diagnostics:
            kept = diagnostics["rollout_keep"].bool() & scores["answer_valid"].bool()
            if kept.any():
                maker.bifocal_monitor_metrics["bifocal_monitor/retained/sc_adv_within_rms"] = (
                    diagnostics["sc_within"][kept].square().mean().sqrt().item()
                )
    logger.info(
        "BifocalRL %s: prompts=%d samples=%d MV=%.4f SCE=%.4f entropy=%.4f aux=%.4f gamma=%.4f",
        cfg["method"],
        len(indices) // group_size,
        len(indices),
        scores["mv"].mean(),
        scores["sc"].mean(),
        scores["entropy"].mean(),
        scores["aux"].mean(),
        gamma,
    )
    return experiences
