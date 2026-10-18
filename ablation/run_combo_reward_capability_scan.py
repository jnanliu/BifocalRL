#!/usr/bin/env python3
# coding=utf-8
"""
Quick ablation: for each co-learning recipe combo folder, load intuitor / MV /
truthcheck configs, generate with ref vs checkpoint-k on **one GPU**, then report
mean rewards and capability ratios.

Pipeline (matches training flow, without the train step):
  1. **Sample** — only vLLM: each side (A then B) runs in a **fresh subprocess**
     so the OS reclaims GPU memory after exit (vLLM v1 + CUDA graphs often do not
     fully free in-process).  Rollouts are written under ``--rollout_cache_dir``.
  2. **Compute** — only HF: truth / co_mv / self_scores (SCE), plus **sum_scores**
     and **mixed_reward** (cross-model SCE → weighted sum → per-model means and
     capability ratios; mixed adds co_mv term like training).  One scoring model
     at a time (``score_responses_for_model`` already unloads).

This avoids holding two vLLM engines (or vLLM + large HF) on the same GPU.

Designed for the six combo dirs:
  co-q1.5q1.7, co-q1.7q3, co-q2.5phi4, co-q2.5q1.7, co-q2.5q3, co-q3phi4

Phases: ``--phase all`` (default), ``sample``, or ``compute``.  After ``sample``,
run ``--phase compute`` with the same ``--rollout_cache_dir`` to score only.

Usage (from ``co-learning/``):

    CUDA_VISIBLE_DEVICES=0 python -m ablation.run_combo_reward_capability_scan \\
        --project_root .. \\
        --num_problems 32

    # Match training recipe weights for sum_scores / mixed_reward capability rows:
    #   --reward_weight_a 1.0 --reward_weight_b 1.0 \\
    #   --mixed_reward_mv_weight 0.5 --mixed_reward_intuitor_weight 0.5

Optional path fixes when recipe JSON still points at another machine:

    MODEL_PATH_FALLBACK_ROOT=/path/to/models python -m ablation ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("combo_reward_cap_scan")

_CO_LEARNING_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMBOS = (
    "co-q1.5q1.7",
    "co-q1.7q3",
    "co-q2.5phi4",
    "co-q2.5q1.7",
    "co-q2.5q3",
    "co-q3phi4",
)

RECIPE_FILES = {
    "intuitor": "co_intuitor_hacpo.json",
    "mv": "co_mv_hacpo.json",
    "truthcheck": "repro_hacpo_truthcheck.json",
}

# Report rows / skip placeholders (must stay in sync).
REPORT_REWARD_VIEWS = (
    "truth",
    "co_mv",
    "intuitor_self_scores",
    "sum_scores",
    "mixed_reward",
)


def _compute_capability_ratio_details(
    mean_rewards: Dict[str, float],
    model_ids: List[str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Same semantics as ``co_intuitor_train.compute_capability_ratio_details``."""
    floored = {mid: max(float(mean_rewards[mid]), 1e-8) for mid in model_ids}
    raw_ratios: Dict[str, Dict[str, float]] = {}
    clipped_ratios: Dict[str, Dict[str, float]] = {}
    for k in model_ids:
        raw_ratios[k] = {}
        clipped_ratios[k] = {}
        for j in model_ids:
            ratio = floored[k] / floored[j]
            raw_ratios[k][j] = float(ratio)
            clipped_ratios[k][j] = max(0.1, min(10.0, ratio))
    return floored, raw_ratios, clipped_ratios


def _resolve_local_path(path_str: str, project_root: Path, fallback_root: Optional[Path]) -> str:
    p = Path(path_str).expanduser()
    if p.exists():
        return str(p.resolve())
    if path_str.startswith("Qwen/") or path_str.startswith("meta-llama/") or "/" not in path_str:
        return path_str
    if fallback_root is not None:
        alt = fallback_root / p.name
        if alt.exists():
            return str(alt.resolve())
    trial = project_root / path_str
    if trial.exists():
        return str(trial.resolve())
    return path_str


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _latest_checkpoint_step(base: Path) -> Optional[int]:
    """Find the highest checkpoint-N step for which both model_a and model_b exist."""
    dir_a = base / "model_a"
    dir_b = base / "model_b"
    if not dir_a.is_dir() or not dir_b.is_dir():
        return None
    import re as _re
    steps_a = set()
    for d in dir_a.iterdir():
        m = _re.match(r"checkpoint-(\d+)$", d.name)
        if m and d.is_dir():
            steps_a.add(int(m.group(1)))
    best: Optional[int] = None
    for s in sorted(steps_a, reverse=True):
        if (dir_b / f"checkpoint-{s}").is_dir():
            best = s
            break
    return best


def _checkpoint_pair(
    recipe: dict,
    project_root: Path,
    step: Optional[int],
) -> Optional[Tuple[str, str]]:
    rel = (recipe.get("output") or {}).get("checkpoint_dir")
    if not rel:
        return None
    base = (project_root / rel).resolve()
    if step is None or step < 0:
        step = _latest_checkpoint_step(base)
        if step is None:
            return None
        logger.info("Auto-detected latest checkpoint step: %d (in %s)", step, base)
    ca = base / "model_a" / f"checkpoint-{step}"
    cb = base / "model_b" / f"checkpoint-{step}"
    if ca.is_dir() and cb.is_dir():
        return str(ca), str(cb)
    return None


def _enable_thinking_kw(mode: str) -> dict:
    _map = {"true": True, "false": False, "none": None}
    m = str(mode or "none").lower()
    return {"enable_thinking": _map.get(m, None)}


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _vllm_release_hard(llm: Any) -> None:
    """Tear down vLLM and reclaim GPU before loading another engine (best-effort)."""
    import gc

    from utils.generate import shutdown_offline_llm

    shutdown_offline_llm(llm)
    try:
        del llm
    except Exception:
        pass
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


@dataclass
class RolloutBundle:
    responses_a: List[str]
    responses_b: List[str]
    finish_a: Optional[List[Optional[str]]]
    finish_b: Optional[List[Optional[str]]]


def _vllm_subprocess_env() -> dict:
    e = os.environ.copy()
    e.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    e.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    return e


def _vllm_generate_one_side_in_process(
    model_path: str,
    prompts: List[str],
    *,
    num_generations: int,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    max_prompt_length: int,
    gpu_memory_utilization: float,
    dtype: str,
    cuda_device: Optional[int],
    max_model_len: Optional[int],
    vllm_max_num_seqs: Optional[int],
    vllm_max_batched_tokens: Optional[int],
    vllm_chunked_prefill: bool,
    seed: int,
) -> Tuple[List[str], Optional[List[Optional[str]]]]:
    from utils.generate import create_offline_llm, generate_sequences

    eff_max_len = max_model_len
    if eff_max_len is None:
        eff_max_len = max_prompt_length + max_completion_tokens + 64

    llm = create_offline_llm(
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        dtype=dtype,
        cuda_device=cuda_device,
        seed=seed,
        max_model_len=eff_max_len,
        max_num_seqs=vllm_max_num_seqs,
        max_num_batched_tokens=vllm_max_batched_tokens,
        enable_chunked_prefill=vllm_chunked_prefill,
    )
    try:
        out = generate_sequences(
            model_path=model_path,
            prompts=prompts,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_tokens,
            repetition_penalty=1.0,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            cuda_device=cuda_device,
            seed=seed,
            max_model_len=eff_max_len,
            max_num_seqs=vllm_max_num_seqs,
            max_num_batched_tokens=vllm_max_batched_tokens,
            enable_chunked_prefill=vllm_chunked_prefill,
            llm=llm,
        )
    finally:
        _vllm_release_hard(llm)

    fr = [str(x) if x is not None else None for x in out.finish_reasons]
    finish = fr if len(fr) == len(out.response_texts) else None
    return out.response_texts, finish


def _vllm_generate_worker_main(in_path: str, out_path: str) -> None:
    """Child entry: load JSON config, run one vLLM side, write JSON results."""
    co_root = Path(__file__).resolve().parents[1]
    if str(co_root) not in sys.path:
        sys.path.insert(0, str(co_root))
    with open(in_path, encoding="utf-8") as f:
        cfg = json.load(f)
    texts, finish = _vllm_generate_one_side_in_process(**cfg)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"response_texts": texts, "finish_reasons": finish}, f, ensure_ascii=False)


def _vllm_generate_one_side(
    model_path: str,
    prompts: List[str],
    *,
    num_generations: int,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    max_prompt_length: int,
    gpu_memory_utilization: float,
    dtype: str,
    cuda_device: Optional[int],
    max_model_len: Optional[int],
    vllm_max_num_seqs: Optional[int],
    vllm_max_batched_tokens: Optional[int],
    vllm_chunked_prefill: bool,
    seed: int,
) -> Tuple[List[str], Optional[List[Optional[str]]]]:
    """Run vLLM in a subprocess so GPU memory is released when the child exits."""
    cfg = {
        "model_path": model_path,
        "prompts": prompts,
        "num_generations": num_generations,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
        "max_prompt_length": max_prompt_length,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "cuda_device": cuda_device,
        "max_model_len": max_model_len,
        "vllm_max_num_seqs": vllm_max_num_seqs,
        "vllm_max_batched_tokens": vllm_max_batched_tokens,
        "vllm_chunked_prefill": vllm_chunked_prefill,
        "seed": seed,
    }
    fd, in_path = tempfile.mkstemp(prefix="combo_vllm_in_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as wf:
            json.dump(cfg, wf, ensure_ascii=False)
        out_path = in_path + ".out.json"
        script = str(Path(__file__).resolve())
        subprocess.run(
            [sys.executable, script, "--vllm-one-shot-worker", in_path, out_path],
            check=True,
            env=_vllm_subprocess_env(),
        )
        with open(out_path, encoding="utf-8") as rf:
            payload = json.load(rf)
        texts = list(payload["response_texts"])
        fr_raw = payload.get("finish_reasons")
        finish: Optional[List[Optional[str]]] = None
        if isinstance(fr_raw, list) and len(fr_raw) == len(texts):
            finish = [str(x) if x is not None else None for x in fr_raw]
        return texts, finish
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass
        try:
            os.unlink(in_path + ".out.json")
        except OSError:
            pass


def _sample_rollout_to_bundle(
    path_a: str,
    path_b: str,
    problems: List[dict],
    *,
    tok_a,
    tok_b,
    chat_kwargs: dict,
    num_generations: int,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    max_prompt_length: int,
    gpu_memory_utilization: float,
    dtype: str,
    cuda_device: Optional[int],
    max_model_len: Optional[int],
    vllm_max_num_seqs: Optional[int],
    vllm_max_batched_tokens: Optional[int],
    vllm_chunked_prefill: bool,
    seed: int,
) -> RolloutBundle:
    from utils.prompt_utils import render_prompts

    has_a = getattr(tok_a, "chat_template", None) is not None
    has_b = getattr(tok_b, "chat_template", None) is not None
    prompts_a = render_prompts(problems, tok_a, has_template=has_a, apply_chat_template_kwargs=chat_kwargs)
    prompts_b = render_prompts(problems, tok_b, has_template=has_b, apply_chat_template_kwargs=chat_kwargs)

    ra, fa = _vllm_generate_one_side(
        path_a,
        prompts_a,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        max_prompt_length=max_prompt_length,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        cuda_device=cuda_device,
        max_model_len=max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_batched_tokens=vllm_max_batched_tokens,
        vllm_chunked_prefill=vllm_chunked_prefill,
        seed=seed,
    )
    rb, fb = _vllm_generate_one_side(
        path_b,
        prompts_b,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        max_prompt_length=max_prompt_length,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        cuda_device=cuda_device,
        max_model_len=max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_batched_tokens=vllm_max_batched_tokens,
        vllm_chunked_prefill=vllm_chunked_prefill,
        seed=seed + 1,
    )
    return RolloutBundle(responses_a=ra, responses_b=rb, finish_a=fa, finish_b=fb)


def _reward_metrics(
    reward_label: str,
    rewards_a: torch.Tensor,
    rewards_b: torch.Tensor,
) -> Dict[str, Any]:
    ma = float(rewards_a.mean().item())
    mb = float(rewards_b.mean().item())
    _, raw, clipped = _compute_capability_ratio_details({"a": ma, "b": mb}, ["a", "b"])
    return {
        "reward_view": reward_label,
        "mean_reward_a": ma,
        "mean_reward_b": mb,
        "cap_raw_a_to_b": raw["a"]["b"],
        "cap_raw_b_to_a": raw["b"]["a"],
        "cap_clipped_a_to_b": clipped["a"]["b"],
        "cap_clipped_b_to_a": clipped["b"]["a"],
    }


def _reward_metrics_combined_full(
    reward_label: str,
    combined_2g: torch.Tensor,
    num_generations: int,
) -> Dict[str, Any]:
    """Per-model means from a ``(P, 2G)`` tensor: first G cols = A, last G = B."""
    g = int(num_generations)
    return _reward_metrics(
        reward_label,
        combined_2g[:, :g],
        combined_2g[:, g:],
    )


def _all_reward_views(
    bundle: RolloutBundle,
    *,
    num_prompts: int,
    num_generations: int,
    ground_truths: List[str],
    dataset_key: str,
    truth_scorer: str,
    problems: List[dict],
    path_a: str,
    path_b: str,
    max_prompt_length: int,
    max_completion_length: int,
    scoring_batch_size: int,
    step: int,
    enable_thinking_mode: str,
    reward_weight_a: float = 1.0,
    reward_weight_b: float = 1.0,
    mixed_reward_mv_weight: float = 0.5,
    mixed_reward_intuitor_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    from utils.normalization import normalize_to_unit_interval
    from utils.reward import compute_rewards
    from utils.scoring_helpers import score_responses_for_model

    ra, rb = bundle.responses_a, bundle.responses_b

    t_a, t_b = compute_rewards(
        "truth",
        ra,
        rb,
        num_prompts,
        num_generations,
        ground_truths=ground_truths,
        scorer_name=truth_scorer,
        dataset_hint=dataset_key,
        step=step,
    )
    mv_a, mv_b = compute_rewards(
        "co_mv", ra, rb, num_prompts, num_generations, step=step,
    )

    _chat_kw = _enable_thinking_kw(enable_thinking_mode)
    sce_a = score_responses_for_model(
        path_a,
        path_a,
        problems,
        ra,
        rb,
        all_finish_reasons_a=bundle.finish_a,
        all_finish_reasons_b=bundle.finish_b,
        num_generations_per_model=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        batch_size=scoring_batch_size,
        score_only="a",
        accelerator=None,
        apply_chat_template_kwargs=_chat_kw,
    )
    sce_b = score_responses_for_model(
        path_b,
        path_b,
        problems,
        ra,
        rb,
        all_finish_reasons_a=bundle.finish_a,
        all_finish_reasons_b=bundle.finish_b,
        num_generations_per_model=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        batch_size=scoring_batch_size,
        score_only="b",
        accelerator=None,
        apply_chat_template_kwargs=_chat_kw,
    )
    intu_a, intu_b = compute_rewards(
        "self_scores",
        ra,
        rb,
        num_prompts,
        num_generations,
        sce_scores_a=sce_a,
        sce_scores_b=sce_b,
        step=step,
    )

    cross_sce_a = score_responses_for_model(
        path_a,
        path_a,
        problems,
        ra,
        rb,
        all_finish_reasons_a=bundle.finish_a,
        all_finish_reasons_b=bundle.finish_b,
        num_generations_per_model=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        batch_size=scoring_batch_size,
        score_only=None,
        accelerator=None,
        apply_chat_template_kwargs=_chat_kw,
    )
    cross_sce_b = score_responses_for_model(
        path_b,
        path_b,
        problems,
        ra,
        rb,
        all_finish_reasons_a=bundle.finish_a,
        all_finish_reasons_b=bundle.finish_b,
        num_generations_per_model=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        batch_size=scoring_batch_size,
        score_only=None,
        accelerator=None,
        apply_chat_template_kwargs=_chat_kw,
    )

    wa, wb = float(reward_weight_a), float(reward_weight_b)
    sum_combined = wa * cross_sce_a.float() + wb * cross_sce_b.float()
    sum_row = _reward_metrics_combined_full("sum_scores", sum_combined, num_generations)

    w_mv = float(mixed_reward_mv_weight)
    w_int = float(mixed_reward_intuitor_weight)
    intuitor_norm = normalize_to_unit_interval(sum_combined, dim=1)
    mv_full = torch.cat([mv_a.float(), mv_b.float()], dim=1)
    mixed_combined = w_mv * mv_full + w_int * intuitor_norm.float()
    mixed_row = _reward_metrics_combined_full("mixed_reward", mixed_combined, num_generations)

    return [
        _reward_metrics("truth", t_a, t_b),
        _reward_metrics("co_mv", mv_a, mv_b),
        _reward_metrics("intuitor_self_scores", intu_a, intu_b),
        sum_row,
        mixed_row,
    ]


MANIFEST_NAME = "manifest.json"
_CACHE_VERSION = 2  # v2: optional reward_weight_* / mixed_reward_* for sum_scores & mixed_reward rows


def _rollout_cache_basename(combo: str, slot: str, variant: str) -> str:
    safe = lambda s: str(s).replace("/", "_").replace(" ", "_")
    return f"{safe(combo)}__{safe(slot)}__{safe(variant)}.json"


def _skip_report_rows(combo: str, slot: str, variant: str, status: str) -> List[Dict[str, Any]]:
    rows = []
    for view in REPORT_REWARD_VIEWS:
        rows.append({
            "combo": combo,
            "recipe_slot": slot,
            "variant": variant,
            "reward_view": view,
            "mean_reward_a": None,
            "mean_reward_b": None,
            "cap_raw_a_to_b": None,
            "cap_raw_b_to_a": None,
            "cap_clipped_a_to_b": None,
            "cap_clipped_b_to_a": None,
            "status": status,
        })
    return rows


def _failed_job_rows(record_base: Dict[str, Any], status: str) -> List[Dict[str, Any]]:
    rows = []
    for view in REPORT_REWARD_VIEWS:
        rows.append({
            **record_base,
            "reward_view": view,
            "mean_reward_a": None,
            "mean_reward_b": None,
            "cap_raw_a_to_b": None,
            "cap_raw_b_to_a": None,
            "cap_clipped_a_to_b": None,
            "cap_clipped_b_to_a": None,
            "status": status,
        })
    return rows


def _write_rollout_cache(
    path: Path,
    *,
    record_base: Dict[str, Any],
    num_prompts: int,
    num_generations: int,
    ground_truths: List[str],
    problems: List[dict],
    dataset_key: str,
    truth_scorer: str,
    max_prompt_length: int,
    max_completion_length: int,
    scoring_batch_size: int,
    enable_thinking_mode: str,
    bundle: RolloutBundle,
    reward_weight_a: float = 1.0,
    reward_weight_b: float = 1.0,
    mixed_reward_mv_weight: float = 0.5,
    mixed_reward_intuitor_weight: float = 0.5,
) -> None:
    payload = {
        "version": _CACHE_VERSION,
        "record_base": record_base,
        "num_prompts": num_prompts,
        "num_generations": num_generations,
        "ground_truths": ground_truths,
        "problems": _to_jsonable(problems),
        "dataset_key": dataset_key,
        "truth_scorer": truth_scorer,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "scoring_batch_size": scoring_batch_size,
        "enable_thinking_mode": enable_thinking_mode,
        "reward_weight_a": float(reward_weight_a),
        "reward_weight_b": float(reward_weight_b),
        "mixed_reward_mv_weight": float(mixed_reward_mv_weight),
        "mixed_reward_intuitor_weight": float(mixed_reward_intuitor_weight),
        "responses_a": bundle.responses_a,
        "responses_b": bundle.responses_b,
        "finish_a": bundle.finish_a,
        "finish_b": bundle.finish_b,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _compute_from_rollout_cache(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rb = payload["record_base"]
    bundle = RolloutBundle(
        responses_a=payload["responses_a"],
        responses_b=payload["responses_b"],
        finish_a=payload.get("finish_a"),
        finish_b=payload.get("finish_b"),
    )
    problems = payload["problems"]
    views = _all_reward_views(
        bundle,
        num_prompts=int(payload["num_prompts"]),
        num_generations=int(payload["num_generations"]),
        ground_truths=list(payload["ground_truths"]),
        dataset_key=str(payload["dataset_key"]),
        truth_scorer=str(payload["truth_scorer"]),
        problems=problems,
        path_a=str(rb["model_a"]),
        path_b=str(rb["model_b"]),
        max_prompt_length=int(payload["max_prompt_length"]),
        max_completion_length=int(payload["max_completion_length"]),
        scoring_batch_size=int(payload["scoring_batch_size"]),
        step=0,
        enable_thinking_mode=str(payload.get("enable_thinking_mode", "none")),
        reward_weight_a=float(payload.get("reward_weight_a", 1.0)),
        reward_weight_b=float(payload.get("reward_weight_b", 1.0)),
        mixed_reward_mv_weight=float(payload.get("mixed_reward_mv_weight", 0.5)),
        mixed_reward_intuitor_weight=float(payload.get("mixed_reward_intuitor_weight", 0.5)),
    )
    flat: List[Dict[str, Any]] = []
    for vm in views:
        flat.append({**rb, **vm, "status": "ok"})
    return flat


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    keys = [
        "combo",
        "recipe_slot",
        "variant",
        "reward_view",
        "mean_reward_a",
        "mean_reward_b",
        "cap_raw_a_to_b",
        "cap_raw_b_to_a",
        "cap_clipped_a_to_b",
        "cap_clipped_b_to_a",
        "status",
    ]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combo reward + capability ratio scan (1 GPU).")
    p.add_argument(
        "--project_root",
        type=str,
        default=str(_CO_LEARNING_ROOT.parent),
        help="Repo root (contains ``outputs/`` and recipe-relative paths).",
    )
    p.add_argument(
        "--configs_root",
        type=str,
        default=str(_CO_LEARNING_ROOT / "recipes" / "configs"),
        help="Parent of ``co-q*`` combo folders.",
    )
    p.add_argument(
        "--config_subdir",
        type=str,
        default="origin",
        help="``origin`` or ``new`` under each combo folder.",
    )
    p.add_argument(
        "--combos",
        nargs="*",
        default=list(DEFAULT_COMBOS),
        help="Combo folder names under configs_root.",
    )
    p.add_argument("--checkpoint_step", type=int, default=-1, help="Use ``checkpoint-{k}``; -1 = auto-detect latest.")
    p.add_argument("--num_problems", type=int, default=32)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda_device", type=int, default=None, help="Physical GPU id (also respects CUDA_VISIBLE_DEVICES).")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--scoring_batch_size", type=int, default=2)
    p.add_argument(
        "--model_path_fallback_root",
        type=str,
        default=os.environ.get("MODEL_PATH_FALLBACK_ROOT", ""),
        help="If a local model path is missing, try ``fallback_root / basename``.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(_CO_LEARNING_ROOT / "ablation" / "outputs" / "combo_reward_capability_scan"),
    )
    p.add_argument(
        "--phase",
        type=str,
        choices=("all", "sample", "compute"),
        default="all",
        help="``sample``: vLLM rollouts only; ``compute``: score cached rollouts; ``all``: both.",
    )
    p.add_argument(
        "--rollout_cache_dir",
        type=str,
        default="",
        help="Where rollout JSON files live. Default: ``<output_dir>/rollouts``.",
    )
    p.add_argument(
        "--reward_weight_a",
        type=float,
        default=1.0,
        help="SCE weight for model A in sum_scores / mixed_reward intuitor term.",
    )
    p.add_argument(
        "--reward_weight_b",
        type=float,
        default=1.0,
        help="SCE weight for model B in sum_scores / mixed_reward intuitor term.",
    )
    p.add_argument(
        "--mixed_reward_mv_weight",
        type=float,
        default=0.5,
        help="Coefficient for co_mv in mixed_reward capability view.",
    )
    p.add_argument(
        "--mixed_reward_intuitor_weight",
        type=float,
        default=0.5,
        help="Coefficient for normalized intuitor sum in mixed_reward capability view.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if str(_CO_LEARNING_ROOT) not in sys.path:
        sys.path.insert(0, str(_CO_LEARNING_ROOT))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.rollout_cache_dir) if args.rollout_cache_dir else (out_dir / "rollouts")
    cache_dir = cache_dir.resolve()

    if args.phase in ("all", "sample"):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    project_root = Path(args.project_root).resolve()
    configs_root = Path(args.configs_root).resolve()
    fallback = Path(args.model_path_fallback_root).resolve() if args.model_path_fallback_root else None

    from transformers import AutoTokenizer

    from utils.dataset import load_co_intuitor_dataset, get_answer_column

    flat_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    if args.phase in ("all", "sample"):
        cache_dir.mkdir(parents=True, exist_ok=True)

        for combo in args.combos:
            combo_dir = configs_root / combo / args.config_subdir
            if not combo_dir.is_dir():
                logger.warning("Skip combo %s: missing %s", combo, combo_dir)
                continue

            base_path = combo_dir / RECIPE_FILES["intuitor"]
            if not base_path.exists():
                logger.warning("Skip combo %s: missing %s", combo, base_path)
                continue
            base_recipe = _load_json(base_path)
            models = base_recipe.get("models") or {}
            ref_a = _resolve_local_path(str(models.get("model_a", "")), project_root, fallback)
            ref_b = _resolve_local_path(str(models.get("model_b", "")), project_root, fallback)

            ds = base_recipe.get("dataset") or {}
            dataset_key = str(ds.get("key", "math_lighteval"))
            instr_suffix = ds.get("instruction_suffix")
            system_prompt = ds.get("system_prompt")
            enable_mode = str(ds.get("enable_thinking", "none"))

            gen = base_recipe.get("generation") or {}
            temperature = float(gen.get("temperature", 1.0))
            top_p = float(gen.get("top_p", 1.0))
            max_completion = int(gen.get("max_completion_length", 2048))
            max_prompt_len = int(gen.get("max_prompt_length", 1024))

            vllm = base_recipe.get("vllm") or {}
            max_model_len = vllm.get("max_model_len")
            if max_model_len is not None:
                max_model_len = int(max_model_len)
            vllm_max_seqs = vllm.get("max_num_seqs")
            if vllm_max_seqs is not None:
                vllm_max_seqs = int(vllm_max_seqs)
            vllm_batched = vllm.get("max_num_batched_tokens")
            if vllm_batched is not None:
                vllm_batched = int(vllm_batched)
            vllm_chunked = bool(vllm.get("chunked_prefill", True))

            problems = load_co_intuitor_dataset(
                dataset_key=dataset_key,
                max_samples=args.num_problems if args.num_problems > 0 else None,
                system_prompt=system_prompt,
                instruction_suffix_override=instr_suffix,
            )
            answer_col = get_answer_column(dataset_key)
            ground_truths = [str(pr[answer_col]) for pr in problems]
            num_prompts = len(problems)
            chat_kwargs = _enable_thinking_kw(enable_mode)

            for slot, fname in RECIPE_FILES.items():
                recipe_path = combo_dir / fname
                if not recipe_path.exists():
                    logger.warning("[%s] missing %s", combo, recipe_path)
                    continue
                recipe = _load_json(recipe_path)
                truth_scorer = str((recipe.get("reward") or {}).get("scorer", "hacrl_prime_math"))

                ck_step = args.checkpoint_step if args.checkpoint_step >= 0 else None
                ck = _checkpoint_pair(recipe, project_root, ck_step)

                if ck:
                    detected = _latest_checkpoint_step(
                        (project_root / ((recipe.get("output") or {}).get("checkpoint_dir", ""))).resolve()
                    ) if ck_step is None else ck_step
                    ck_label = "checkpoint_%d" % (detected or 0)
                else:
                    ck_label = "checkpoint_latest"

                variant_paths: List[Tuple[str, Optional[str], Optional[str]]] = [
                    ("ref", ref_a, ref_b),
                ]
                if ck:
                    variant_paths.append((ck_label, ck[0], ck[1]))
                else:
                    variant_paths.append((ck_label, None, None))

                for variant, pa, pb in variant_paths:
                    if not pa or not pb:
                        reason = "skipped_no_checkpoint" if variant != "ref" else "skipped_bad_ref_paths"
                        manifest.append({
                            "type": "skip",
                            "rows": _skip_report_rows(combo, slot, variant, reason),
                        })
                        continue

                    record_base = {
                        "combo": combo,
                        "recipe_slot": slot,
                        "variant": variant,
                        "model_a": pa,
                        "model_b": pb,
                        "dataset_key": dataset_key,
                        "num_prompts": num_prompts,
                        "num_generations": args.num_generations,
                        "truth_scorer": truth_scorer,
                    }

                    cfile = cache_dir / _rollout_cache_basename(combo, slot, variant)
                    if cfile.is_file():
                        logger.info(
                            "[%s %s %s] rollout cache exists, skipping sample: %s",
                            combo, slot, variant, cfile,
                        )
                        manifest.append({"type": "rollout", "file": cfile.name})
                        continue

                    # For "ref" variant, reuse rollout from another slot if available
                    # (same models + same generation params → identical outputs).
                    if variant == "ref":
                        donor = None
                        for other_slot in RECIPE_FILES:
                            if other_slot == slot:
                                continue
                            other_cfile = cache_dir / _rollout_cache_basename(combo, other_slot, "ref")
                            if other_cfile.is_file():
                                donor = other_cfile
                                break
                        if donor is not None:
                            import shutil
                            logger.info(
                                "[%s %s ref] reusing rollout from %s",
                                combo, slot, donor.name,
                            )
                            shutil.copy2(str(donor), str(cfile))
                            # Patch record_base in the copy so report columns are correct
                            with open(cfile, encoding="utf-8") as _f:
                                _payload = json.load(_f)
                            _payload["record_base"]["recipe_slot"] = slot
                            _payload["record_base"]["truth_scorer"] = truth_scorer
                            with open(cfile, "w", encoding="utf-8") as _f:
                                json.dump(_payload, _f, indent=2, ensure_ascii=False)
                            manifest.append({"type": "rollout", "file": cfile.name})
                            continue

                    try:
                        tok_a = AutoTokenizer.from_pretrained(pa, trust_remote_code=True)
                        tok_b = AutoTokenizer.from_pretrained(pb, trust_remote_code=True)
                        tok_a.padding_side = "left"
                        tok_b.padding_side = "left"
                        if tok_a.pad_token_id is None:
                            tok_a.pad_token = tok_a.eos_token
                        if tok_b.pad_token_id is None:
                            tok_b.pad_token = tok_b.eos_token

                        bundle = _sample_rollout_to_bundle(
                            pa,
                            pb,
                            problems,
                            tok_a=tok_a,
                            tok_b=tok_b,
                            chat_kwargs=chat_kwargs,
                            num_generations=args.num_generations,
                            temperature=temperature,
                            top_p=top_p,
                            max_completion_tokens=max_completion,
                            max_prompt_length=max_prompt_len,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            dtype=args.dtype,
                            cuda_device=args.cuda_device,
                            max_model_len=max_model_len,
                            vllm_max_num_seqs=vllm_max_seqs,
                            vllm_max_batched_tokens=vllm_batched,
                            vllm_chunked_prefill=vllm_chunked,
                            seed=args.seed,
                        )
                        _write_rollout_cache(
                            cfile,
                            record_base=record_base,
                            num_prompts=num_prompts,
                            num_generations=args.num_generations,
                            ground_truths=ground_truths,
                            problems=problems,
                            dataset_key=dataset_key,
                            truth_scorer=truth_scorer,
                            max_prompt_length=max_prompt_len,
                            max_completion_length=max_completion,
                            scoring_batch_size=args.scoring_batch_size,
                            enable_thinking_mode=enable_mode,
                            bundle=bundle,
                            reward_weight_a=args.reward_weight_a,
                            reward_weight_b=args.reward_weight_b,
                            mixed_reward_mv_weight=args.mixed_reward_mv_weight,
                            mixed_reward_intuitor_weight=args.mixed_reward_intuitor_weight,
                        )
                        manifest.append({"type": "rollout", "file": cfile.name})
                        logger.info(
                            "[%s %s %s] sample phase: wrote %s",
                            combo, slot, variant, cfile,
                        )
                    except Exception as exc:
                        logger.exception("[%s %s %s] sample failed: %s", combo, slot, variant, exc)
                        manifest.append({
                            "type": "skip",
                            "rows": _failed_job_rows(record_base, f"error:{exc}"),
                        })

        with open(cache_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "items": manifest}, f, indent=2, ensure_ascii=False)
        logger.info("Wrote manifest %s (%d items)", cache_dir / MANIFEST_NAME, len(manifest))

    if args.phase in ("all", "compute"):
        compute_rows: List[Dict[str, Any]] = []
        mpath = cache_dir / MANIFEST_NAME
        if mpath.is_file():
            with open(mpath, encoding="utf-8") as f:
                mdoc = json.load(f)
            for it in mdoc.get("items", []):
                if it.get("type") == "skip":
                    compute_rows.extend(it.get("rows", []))
                elif it.get("type") == "rollout":
                    cpath = cache_dir / str(it["file"])
                    if not cpath.is_file():
                        logger.warning("Missing rollout cache %s (skip scoring)", cpath)
                        continue
                    try:
                        compute_rows.extend(_compute_from_rollout_cache(cpath))
                    except Exception as exc:
                        logger.exception("compute failed for %s: %s", cpath, exc)
        else:
            logger.warning(
                "No %s in %s — globbing *.json (excluding manifest)",
                MANIFEST_NAME,
                cache_dir,
            )
            for p in sorted(cache_dir.glob("*.json")):
                if p.name == MANIFEST_NAME:
                    continue
                try:
                    compute_rows.extend(_compute_from_rollout_cache(p))
                except Exception as exc:
                    logger.exception("compute failed for %s: %s", p, exc)

        flat_rows = compute_rows

    if args.phase == "sample":
        logger.info("Sample-only: run again with --phase compute to build report.json")
        return

    json_path = out_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flat_rows, f, indent=2, ensure_ascii=False)

    md_path = out_dir / "report.md"
    _write_markdown(flat_rows, md_path)

    logger.info("Wrote %s and %s (%d rows)", json_path, md_path, len(flat_rows))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--vllm-one-shot-worker":
        _vllm_generate_worker_main(sys.argv[2], sys.argv[3])
        sys.exit(0)
    main()
