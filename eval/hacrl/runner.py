from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoTokenizer

from utils.dataset import (
    get_answer_column,
    load_co_intuitor_dataset,
)
from utils.generate import (
    GenerationOutput,
    check_server_health,
    create_offline_llm,
    generate_responses,
    generate_sequences,
    is_sleeping,
    is_port_in_use,
    pick_free_port,
    start_vllm_server,
    stop_vllm_server,
    shutdown_offline_llm,
    wait_for_vllm_server,
    wake_server,
    _ensure_trust_remote_code_files,
)
from utils.prompt_utils import render_prompts

from eval.common.cruxeval import (
    construct_cruxeval_prompt,
    default_cruxeval_path,
    extract_cruxeval_answer,
    judge_cruxeval,
    load_cruxeval_jsonl,
)
from eval.common.gsm8k import extract_gsm8k_final_answer
from eval.common.math_equiv import extract_math_final_answer, math_equiv
from eval.scorers import ScoreResult, ScorerName, get_scorer

_HACRL_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


@dataclass(frozen=True)
class EvalConfig:
    model_path: str
    model_tag: str  # "a" or "b"
    checkpoint_tag: str  # e.g. "checkpoint-3"
    system_prompt: Optional[str] = None
    num_generations: int = 1
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    max_samples: Optional[int] = None
    output_root: str = "outputs/eval"
    seed: Optional[int] = 42
    scorer: str = "default"
    # Prompt control — instruction_suffix overrides the per-dataset
    # _DATASET_INSTRUCTIONS when set (empty string → no suffix).
    instruction_suffix: Optional[str] = None
    # "true"/"false"/"none"; "none" = don't pass enable_thinking (TTRL style).
    enable_thinking: str = "false"
    # Offline vLLM (HACRL-style, default)
    use_offline: bool = True
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    vllm_dtype: str = "bfloat16"
    vllm_max_model_len: Optional[int] = None
    cuda_device: Optional[int] = None
    # Legacy HTTP vLLM (only when use_offline=False)
    vllm_port: int = 8010
    vllm_host: str = "localhost"
    vllm_batch_size: int = 16
    auto_start_server: bool = True
    auto_stop_server: bool = True


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


_ENABLE_THINKING_MAP = {"true": True, "false": False, "none": None}


def _build_chat_prompts(
    tokenizer,
    problems: List[Dict[str, Any]],
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    kwargs = chat_template_kwargs if chat_template_kwargs is not None else _HACRL_CHAT_TEMPLATE_KWARGS
    return render_prompts(
        problems,
        tokenizer,
        apply_chat_template_kwargs=kwargs,
    )


def _flatten_generations(gens: List[str], num_generations: int) -> List[List[str]]:
    """
    `generate_responses` returns flattened list: [p0_g0..p0_g{k-1}, p1_g0..].
    Convert to per-problem list of generations.
    """
    if num_generations <= 1:
        return [[g] for g in gens]
    grouped: List[List[str]] = []
    for i in range(0, len(gens), num_generations):
        grouped.append(gens[i : i + num_generations])
    return grouped


def _generate_texts(
    prompts: List[str],
    cfg: EvalConfig,
    max_tokens_override: Optional[int] = None,
    offline_llm: Optional[Any] = None,
) -> List[List[str]]:
    """Generate text completions, returning grouped per-prompt lists.

    Uses the offline vLLM API (HACRL-style) when ``cfg.use_offline`` is True,
    otherwise falls back to the legacy HTTP server API.
    """
    max_tokens = max_tokens_override if max_tokens_override is not None else cfg.max_new_tokens

    if cfg.use_offline:
        gen_out: GenerationOutput = generate_sequences(
            model_path=cfg.model_path,
            prompts=prompts,
            num_generations=cfg.num_generations,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=max_tokens,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.vllm_dtype,
            cuda_device=cfg.cuda_device,
            seed=cfg.seed,
            max_model_len=cfg.vllm_max_model_len,
            llm=offline_llm,
        )
        return _flatten_generations(gen_out.response_texts, cfg.num_generations)

    generations = generate_responses(
        model_path=cfg.model_path,
        prompts=prompts,
        num_generations=cfg.num_generations,
        batch_size=cfg.vllm_batch_size,
        port=cfg.vllm_port,
        host=cfg.vllm_host,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_tokens,
        auto_start_server=cfg.auto_start_server,
        auto_stop_server=cfg.auto_stop_server,
        seed=cfg.seed,
    )
    return _flatten_generations(generations, cfg.num_generations)


def _save_results(dataset_name: str, cfg: EvalConfig, payload: Dict[str, Any]) -> str:
    out_dir = os.path.join(cfg.output_root, dataset_name)
    _ensure_dir(out_dir)
    filename = f"results_{cfg.checkpoint_tag}_model-{cfg.model_tag}.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def evaluate_math_dataset(
    dataset_key: str,
    answer_column: str,
    cfg: EvalConfig,
    offline_llm: Optional[Any] = None,
) -> str:
    _ensure_trust_remote_code_files(cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    problems = load_co_intuitor_dataset(
        dataset_key=dataset_key,
        max_samples=cfg.max_samples,
        system_prompt=cfg.system_prompt,
        instruction_suffix_override=cfg.instruction_suffix,
    )

    et_val = _ENABLE_THINKING_MAP.get(cfg.enable_thinking, False)
    chat_kwargs = {"enable_thinking": et_val}
    prompts = _build_chat_prompts(tokenizer, problems, chat_template_kwargs=chat_kwargs)
    grouped = _generate_texts(prompts, cfg, offline_llm=offline_llm)

    scorer = get_scorer(cfg.scorer, dataset_hint=dataset_key)
    use_default = cfg.scorer == "default" and not dataset_key.endswith("_ttt")

    total = len(problems)
    correct_any = 0
    sum_per_problem_mean_score = 0.0
    verifiable = 0
    verifiable_correct_any = 0
    parse_failed = 0
    string_fallback = 0
    verify_failed = 0
    per_problem: List[Dict[str, Any]] = []

    for ex, gens, prompt_rendered in zip(problems, grouped, prompts):
        gold = ex.get(answer_column, "")
        per_problem_score_sum = 0.0
        per_problem_num_gens = 0

        if dataset_key == "gpqa_ttt":
            gold_for_check = str(gold).strip()
        elif dataset_key == "gsm8k":
            gold_for_check = extract_gsm8k_final_answer(gold)
        elif dataset_key.endswith("_ttt"):
            gold_for_check = str(gold).strip()
        else:
            gold_for_check = extract_math_final_answer(gold)

        responses: List[Dict[str, Any]] = []
        any_ok = False
        any_verifiable = False
        any_verifiable_ok = False
        for g in gens:
            # HACRL parity: score raw decoded rollout text without extra completion cleanup.
            g_clean = str(g)

            if use_default:
                res = math_equiv(gold_for_check, g_clean)
                sr = ScoreResult(
                    is_correct=res.is_correct,
                    score=1.0 if res.is_correct else 0.0,
                    pred_extracted=res.pred_extracted,
                    gold_extracted=res.gold_extracted,
                    error=res.error,
                )
            else:
                sr = scorer.score(g_clean, gold_for_check)
            per_problem_score_sum += float(sr.score)
            per_problem_num_gens += 1

            if sr.error == "parse_failed":
                parse_failed += 1
            elif sr.error == "string_match_fallback":
                string_fallback += 1
            elif sr.error is not None and str(sr.error).startswith("verify_failed"):
                verify_failed += 1

            ok = sr.is_correct is True
            if ok:
                any_ok = True
            if sr.is_correct is not None:
                any_verifiable = True
                if ok:
                    any_verifiable_ok = True
            responses.append(
                {
                    "text": g_clean,
                    "is_correct": sr.is_correct,
                    "error": sr.error,
                    "gold_used": gold_for_check,
                    "gold_extracted": sr.gold_extracted,
                    "pred_extracted": sr.pred_extracted,
                }
            )

        if any_ok:
            correct_any += 1
        per_problem_mean_score = (
            per_problem_score_sum / per_problem_num_gens if per_problem_num_gens > 0 else 0.0
        )
        sum_per_problem_mean_score += per_problem_mean_score
        if any_verifiable:
            verifiable += 1
            if any_verifiable_ok:
                verifiable_correct_any += 1

        per_problem.append(
            {
                "prompt": prompt_rendered,
                "gold": gold,
                "gold_used": gold_for_check,
                "responses": responses,
                "any_correct": any_ok,
                "mean_score": per_problem_mean_score,
            }
        )

    mean_score = sum_per_problem_mean_score / total if total else 0.0
    payload = {
        "summary": {
            "dataset": dataset_key,
            "scorer": cfg.scorer,
            "model_path": cfg.model_path,
            "model_tag": cfg.model_tag,
            "checkpoint": cfg.checkpoint_tag,
            "num_generations": cfg.num_generations,
            "timestamp": _now_iso(),
            "pass_at_k": correct_any / total if total else 0.0,
            "mean_score": mean_score,
            "correct_any": correct_any,
            "total": total,
            "verifiable_problems": verifiable,
            "verifiable_pass_at_k": (verifiable_correct_any / verifiable) if verifiable else None,
            "parse_failed_responses": parse_failed,
            "string_match_fallback_responses": string_fallback,
            "verify_failed_responses": verify_failed,
        },
        "problems": per_problem,
    }
    return _save_results(dataset_key, cfg, payload)


def evaluate_cruxeval(
    cfg: EvalConfig,
    dataset_path: Optional[str] = None,
    offline_llm: Optional[Any] = None,
) -> str:
    _ensure_trust_remote_code_files(cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = dataset_path or default_cruxeval_path(project_root)
    data = load_cruxeval_jsonl(dataset_path, max_samples=cfg.max_samples)

    # Wrap each dataset prompt as a user message under the same system_prompt used in training.
    problems: List[Dict[str, Any]] = []
    for row in data:
        user_prompt = construct_cruxeval_prompt(row)
        messages: List[Dict[str, str]] = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        problems.append(
            {
                "prompt": messages,
                "id": row.get("id"),
                "gold": row.get("output", ""),
            }
        )

    prompts = _build_chat_prompts(tokenizer, problems)
    grouped = _generate_texts(
        prompts,
        cfg,
        max_tokens_override=cfg.max_new_tokens,
        offline_llm=offline_llm,
    )

    total = len(problems)
    correct_any = 0
    per_problem: List[Dict[str, Any]] = []
    for ex, gens, prompt_rendered in zip(problems, grouped, prompts):
        gold = ex.get("gold", "")
        responses: List[Dict[str, Any]] = []
        any_ok = False
        for g in gens:
            # HACRL parity: score raw decoded rollout text without extra completion cleanup.
            g_clean = str(g)
            pred = extract_cruxeval_answer(g_clean)
            j = judge_cruxeval(pred, gold)
            if j.is_correct:
                any_ok = True
            responses.append(
                {
                    "text": g_clean,
                    "predicted": j.predicted,
                    "ground_truth": j.ground_truth,
                    "is_correct": j.is_correct,
                }
            )
        if any_ok:
            correct_any += 1
        per_problem.append(
            {
                "id": ex.get("id"),
                "prompt": prompt_rendered,
                "gold": gold,
                "responses": responses,
                "any_correct": any_ok,
            }
        )

    payload = {
        "summary": {
            "dataset": "cruxeval",
            "dataset_path": dataset_path,
            "model_path": cfg.model_path,
            "model_tag": cfg.model_tag,
            "checkpoint": cfg.checkpoint_tag,
            "num_generations": cfg.num_generations,
            "timestamp": _now_iso(),
            "pass_at_k": correct_any / total if total else 0.0,
            "correct_any": correct_any,
            "total": total,
        },
        "problems": per_problem,
    }
    return _save_results("cruxeval", cfg, payload)


def run_all(cfg: EvalConfig, datasets: Sequence[str]) -> Dict[str, str]:
    """
    Returns mapping dataset_name -> output_json_path.

    Scorer selection: set ``cfg.scorer`` to one of :pydata:`ScorerName` values.
    Available scorers::

        default            – co-learning math_verify / gsm8k (original)
        hacrl_math_dapo    – Minerva normalization (AIME / DAPO)
        hacrl_math_boxed   – Simple \\boxed{} + string normalization
        hacrl_prime_math   – Comprehensive sympy-based equivalence
        hacrl_gsm8k        – Strict #### extraction
        hacrl_qa_em        – QA exact match with <answer> tags
    """
    outputs: Dict[str, str] = {}

    _CRUX_ALIASES = {"cruxeval", "crux"}
    _MATH_DATASETS = {
        "math_500", "gsm8k", "math_lighteval", "aime24", "aime25", "amc", "amc23",
        "math_ttt", "aime_ttt", "amc_ttt", "gpqa_ttt",
        "math_l1_ttt", "math_l2_ttt", "math_l3_ttt", "math_l4_ttt", "math_l5_ttt",
    }

    # Offline vLLM: create one reusable engine for all datasets, then
    # shutdown once at the end to avoid multi-dataset memory overlap.
    if cfg.use_offline:
        offline_llm = create_offline_llm(
            model_path=cfg.model_path,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.vllm_dtype,
            cuda_device=cfg.cuda_device,
            seed=cfg.seed,
            max_model_len=cfg.vllm_max_model_len,
        )
        try:
            for d in datasets:
                if d in _CRUX_ALIASES:
                    outputs["cruxeval"] = evaluate_cruxeval(cfg, offline_llm=offline_llm)
                elif d in _MATH_DATASETS:
                    answer_col = get_answer_column(d)
                    outputs[d] = evaluate_math_dataset(d, answer_col, cfg, offline_llm=offline_llm)
                else:
                    raise ValueError(
                        f"Unknown dataset: {d!r}. "
                        f"Available: {sorted(_MATH_DATASETS | _CRUX_ALIASES)}"
                    )
            return outputs
        finally:
            shutdown_offline_llm(offline_llm)

    # Legacy HTTP server path
    server_proc = None
    cfg_inner = cfg
    try:
        if cfg.auto_start_server:
            port = int(cfg.vllm_port)
            if is_port_in_use(port):
                new_port = pick_free_port(port + 1)
                port = int(new_port)
                cfg_inner = replace(cfg_inner, vllm_port=port)

            server_proc = start_vllm_server(
                model_path=cfg.model_path,
                port=cfg_inner.vllm_port,
                host="0.0.0.0",
                dtype="bfloat16",
                cuda_device=0,
                log_file=f"vllm_eval_{cfg.model_tag}_{cfg.checkpoint_tag}_{cfg_inner.vllm_port}.log",
            )
            ok = wait_for_vllm_server(f"http://localhost:{cfg_inner.vllm_port}/health", timeout=600)
            if not ok:
                raise RuntimeError("Failed to start vLLM server for evaluation")
        else:
            if not check_server_health(cfg.vllm_host, cfg.vllm_port):
                raise RuntimeError(f"vLLM server at {cfg.vllm_host}:{cfg.vllm_port} is not reachable")
            sleeping = is_sleeping(cfg.vllm_host, cfg.vllm_port)
            if sleeping is True:
                if not wake_server(cfg.vllm_host, cfg.vllm_port):
                    raise RuntimeError("Failed to wake up sleeping vLLM server")

        cfg_inner = replace(cfg_inner, auto_start_server=False, auto_stop_server=False)

        for d in datasets:
            if d in _CRUX_ALIASES:
                outputs["cruxeval"] = evaluate_cruxeval(cfg_inner)
            elif d in _MATH_DATASETS:
                answer_col = get_answer_column(d)
                outputs[d] = evaluate_math_dataset(d, answer_col, cfg_inner)
            else:
                raise ValueError(
                    f"Unknown dataset: {d!r}. "
                    f"Available: {sorted(_MATH_DATASETS | _CRUX_ALIASES)}"
                )
        return outputs
    finally:
        if server_proc is not None and cfg.auto_stop_server:
            stop_vllm_server(server_proc)
