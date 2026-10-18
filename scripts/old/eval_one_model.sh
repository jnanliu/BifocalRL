#!/bin/bash

# Evaluate a single model checkpoint (no training / no generation+scoring).
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint ./scripts/eval_one_model.sh
#
# Or:
#   ./scripts/eval_one_model.sh /path/to/checkpoint
#
# Backends:
#   - EVAL_BACKEND=custom    -> uses our local vLLM-based evaluator (supports cruxeval)
#   - EVAL_BACKEND=lighteval -> uses LightEval vLLM backend (supports local_* tasks via --custom-tasks)
#
# Scorers (custom backend only):
#   SCORER=default            -> co-learning math_verify (original)
#   SCORER=hacrl_math_dapo    -> Minerva normalization (AIME/DAPO)
#   SCORER=hacrl_math_boxed   -> Simple \boxed{} + string normalization
#   SCORER=hacrl_prime_math   -> Comprehensive sympy-based equivalence
#   SCORER=hacrl_gsm8k        -> Strict #### extraction
#   SCORER=hacrl_qa_em        -> QA exact match with <answer> tags
#
# Examples:
#   EVAL_BACKEND=custom MODEL_PATH=outputs/.../checkpoint-3 ./scripts/eval_one_model.sh
#   EVAL_BACKEND=custom SCORER=hacrl_prime_math MODEL_PATH=outputs/.../checkpoint-3 ./scripts/eval_one_model.sh
#   EVAL_BACKEND=lighteval MODEL_PATH=outputs/.../checkpoint-3 \
#     LIGHTEVAL_TASKS="lighteval|math_500|0|0,lighteval|gsm8k|0|0" ./scripts/eval_one_model.sh

set -euo pipefail

# vLLM multiprocessing: prefer spawn (open-r1 default).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Deterministic scheduling for reproducible evaluation (vLLM V1).
# See https://docs.vllm.ai/en/latest/usage/reproducibility.html
export VLLM_ENABLE_V1_MULTIPROCESSING=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
  echo "[ERROR] $*" >&2
  exit 1
}

infer_checkpoint_tag() {
  local p="$1"
  local base
  base="$(basename "$(realpath "$p")")"
  if [[ "$base" == checkpoint-* ]]; then
    echo "$base"
  else
    echo "checkpoint-unknown"
  fi
}

# -----------------------------------------------------------------------------
# Configuration (override via env vars)
# -----------------------------------------------------------------------------

# Required: model path
MODEL_PATH="${MODEL_PATH:-${1:-/home/jizeyu26/jzy/7_co_learning/test/models/Phi-4-mini-instruct}}"
if [[ -z "$MODEL_PATH" ]]; then
  error "MODEL_PATH is required. Provide env MODEL_PATH=... or pass as first arg."
fi

# Logging / verbosity
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_DATASETS_VERBOSITY="${HF_DATASETS_VERBOSITY:-error}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Evaluation backend
EVAL_BACKEND="${EVAL_BACKEND:-lighteval}" # custom | lighteval

# Common eval params
SYSTEM_PROMPT="You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}. Please stop generation immediately after outputing the box."
SEED="${SEED:-7}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-400}"
EVAL_NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval_single}"

# Custom evaluator (vLLM server)
VLLM_PORT="${VLLM_PORT:-8010}"
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-400}"
CUSTOM_DATASETS="${CUSTOM_DATASETS:-math_500,gsm8k}"
SCORER="${SCORER:-default}"

# ZeroEval (CRUXEval)
ZEROEVAL_DIR="${ZEROEVAL_DIR:-third_party/ZeroEval}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-outputs/zeroeval_single}"
ZEROEVAL_TP_SIZE="${ZEROEVAL_TP_SIZE:-1}"
ZEROEVAL_BATCH_SIZE="${ZEROEVAL_BATCH_SIZE:-4}"
ZEROEVAL_MAX_TOKENS="${ZEROEVAL_MAX_TOKENS:-2048}"
ZEROEVAL_GPU_MEMORY_UTILIZATION="${ZEROEVAL_GPU_MEMORY_UTILIZATION:-0.95}"
ZEROEVAL_MAX_SAMPLES="${ZEROEVAL_MAX_SAMPLES:-800}"

# LightEval (aligned with open-r1)
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-math_500,gsm8k}"
# Patch dataset IO so built-in tasks load from LOCAL_DATASETS_ROOT snapshots.
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-eval.lighteval.custom_tasks.offline_patch}"
LIGHTEVAL_LOCAL_DATASETS_ROOT="${LIGHTEVAL_LOCAL_DATASETS_ROOT:-${LOCAL_DATASETS_ROOT:-/home/jizeyu26/jzy/7_co_learning/test/datasets}}"
LIGHTEVAL_OUTPUT_DIR="${LIGHTEVAL_OUTPUT_DIR:-outputs/lighteval/single/$(infer_checkpoint_tag "$MODEL_PATH")}"

# Where LightEval (datasets/transformers) will read/write caches.
LIGHTEVAL_HF_HOME="${LIGHTEVAL_HF_HOME:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/hf_home}"
LIGHTEVAL_DATASETS_CACHE="${LIGHTEVAL_DATASETS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT}"
LIGHTEVAL_TRANSFORMERS_CACHE="${LIGHTEVAL_TRANSFORMERS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/transformers_cache}"
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-1}"
# NOTE: With LightEval v0.9.x, setting max_new_tokens==max_model_length (32768) causes
# the prompt to be truncated to 0 tokens (see lighteval/models/vllm/vllm_model.py).
# open-r1 recipes typically use 3072 for math-style eval. You can override if needed.
LIGHTEVAL_MAX_NEW_TOKENS="${LIGHTEVAL_MAX_NEW_TOKENS:-2048}"
# Sampling overrides (make math-500 + LCB fast by default).



main() {
  log "=================================================================================="
  log "Single-model evaluation"
  log "=================================================================================="
  log "MODEL_PATH: $MODEL_PATH"
  log "EVAL_BACKEND: $EVAL_BACKEND"

  if [[ "$EVAL_BACKEND" == "lighteval" ]]; then
    log "Tasks (LightEval): $LIGHTEVAL_TASKS"
    log "Custom tasks file: $LIGHTEVAL_CUSTOM_TASKS"
    log "Local datasets root: $LIGHTEVAL_LOCAL_DATASETS_ROOT"
    log "Output dir: $LIGHTEVAL_OUTPUT_DIR"
    mkdir -p "$LIGHTEVAL_OUTPUT_DIR"

    # Force offline by default (your network is unreliable). Override by exporting
    # HF_HUB_OFFLINE=0 / HF_DATASETS_OFFLINE=0 / TRANSFORMERS_OFFLINE=0 if desired.
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

    python -m eval.lighteval.runner \
      --model_path "$MODEL_PATH" \
      --tasks "$LIGHTEVAL_TASKS" \
      --output_dir "$LIGHTEVAL_OUTPUT_DIR" \
      --system_prompt "$SYSTEM_PROMPT" \
      --max_new_tokens "$LIGHTEVAL_MAX_NEW_TOKENS" \
      --temperature 0.0 \
      --top_p 1.0 \
      --seed "$SEED" \
      --data_parallel_size "$LIGHTEVAL_DATA_PARALLEL_SIZE" \
      --custom_tasks "$LIGHTEVAL_CUSTOM_TASKS" \
      --local_datasets_root "$LIGHTEVAL_LOCAL_DATASETS_ROOT" \
      --hf_home "$LIGHTEVAL_HF_HOME" \
      --datasets_cache "$LIGHTEVAL_DATASETS_CACHE" \
      --transformers_cache "$LIGHTEVAL_TRANSFORMERS_CACHE"

    log "Evaluate (ZeroEval CRUX): $MODEL_PATH"
    ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
    ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
    ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
    ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
    python -m eval.zeroeval.runner \
      --model_path "$MODEL_PATH" \
      --model_tag a \
      --checkpoint_tag "$(infer_checkpoint_tag "$MODEL_PATH")" \
      --output_root "$ZEROEVAL_OUTPUT_ROOT" \
      --zeroeval_dir "$ZEROEVAL_DIR" \
      --mode zeroeval_repo \
      --engine vllm \
      --tensor_parallel_size "$ZEROEVAL_TP_SIZE" \
      --dtype bfloat16 \
      --gpu_memory_utilization "$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
      --batch_size "$ZEROEVAL_BATCH_SIZE" \
      --max_tokens "$ZEROEVAL_MAX_TOKENS" \
      --temperature 0 \
      --top_p 1.0 \
      --seed "$SEED" \
      --max_samples "$ZEROEVAL_MAX_SAMPLES"

    log "Done. LightEval outputs at: $LIGHTEVAL_OUTPUT_DIR"
    return 0
  fi

  if [[ "$EVAL_BACKEND" != "custom" ]]; then
    error "Unknown EVAL_BACKEND=$EVAL_BACKEND (expected custom|lighteval)"
  fi

  local checkpoint_tag
  checkpoint_tag="$(infer_checkpoint_tag "$MODEL_PATH")"
  log "Datasets (custom): $CUSTOM_DATASETS"
  log "Scorer: $SCORER"
  log "Output root: $OUTPUT_ROOT"
  log "Checkpoint tag: $checkpoint_tag"

  python -m src.evaluate_checkpoints \
    --model_path "$MODEL_PATH" \
    --model_tag a \
    --checkpoint_tag "$checkpoint_tag" \
    --datasets "$CUSTOM_DATASETS" \
    --num_generations "$EVAL_NUM_GENERATIONS" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --system_prompt "$SYSTEM_PROMPT" \
    --output_root "$OUTPUT_ROOT" \
    --vllm_port "$VLLM_PORT" \
    --vllm_host "$VLLM_HOST" \
    --vllm_batch_size "$VLLM_BATCH_SIZE" \
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --scorer "$SCORER"

  log "Evaluate (ZeroEval CRUX): $MODEL_PATH"
  ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
  ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
  ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
  ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
  python -m eval.zeroeval.runner \
    --model_path "$MODEL_PATH" \
    --model_tag a \
    --checkpoint_tag "$checkpoint_tag" \
    --output_root "$ZEROEVAL_OUTPUT_ROOT" \
    --zeroeval_dir "$ZEROEVAL_DIR" \
    --mode zeroeval_repo \
    --engine vllm \
    --tensor_parallel_size "$ZEROEVAL_TP_SIZE" \
    --dtype bfloat16 \
    --gpu_memory_utilization "$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
    --batch_size "$ZEROEVAL_BATCH_SIZE" \
    --max_tokens "$ZEROEVAL_MAX_TOKENS" \
    --temperature 0 \
    --top_p 1.0 \
    --seed "$SEED" \
    --max_samples "$ZEROEVAL_MAX_SAMPLES"

  log "Done. Custom eval outputs under: $OUTPUT_ROOT"
}

main "$@"

