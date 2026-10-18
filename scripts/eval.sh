#!/usr/bin/env bash
set -euo pipefail

# Config-driven standalone evaluation script.
#
# Runs LightEval (math_500, gsm8k, etc.) and/or ZeroEval (CRUXEval) on a
# single model checkpoint, reading all parameters from a JSON config.
#
# Usage:
#   ./scripts/eval.sh recipes/configs/eval.json --model /path/to/checkpoint
#   MODEL_PATH=/path/to/checkpoint ./scripts/eval.sh recipes/configs/eval.json
#
# The --model flag (or MODEL_PATH env var) is the only required override;
# everything else comes from the config with sensible defaults.
#
# Environment variables always take precedence over JSON values.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"
cd "$SCRIPT_DIR/.." || exit 1

# ============================================================================
# Parse arguments
# ============================================================================
CONFIG_FILE=""
EXTRA_MODEL_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model_path)
      EXTRA_MODEL_PATH="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 <config.json> [--model /path/to/checkpoint]"
      echo ""
      echo "All parameters come from the JSON config. Override any value via env vars."
      echo "MODEL_PATH can also be set as an env var."
      exit 0 ;;
    *)
      if [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="$1"
      else
        error "Unknown argument: $1. Use --help for usage."
      fi
      shift ;;
  esac
done

[[ -z "$CONFIG_FILE" ]] && error "Usage: $0 <config.json> [--model /path/to/checkpoint]"
_load_config "$CONFIG_FILE"

# ============================================================================
# Read config (env vars override JSON values)
# ============================================================================

# Model
MODEL_PATH="${MODEL_PATH:-${EXTRA_MODEL_PATH:-$(cfg model.path)}}"
[[ -z "$MODEL_PATH" ]] && error "model.path is required. Set MODEL_PATH env var or pass --model."
MODEL_TAG="${MODEL_TAG:-$(cfg model.tag "a")}"

# Dataset
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$(cfg dataset.system_prompt)}"
LOCAL_DATASETS_ROOT="${LOCAL_DATASETS_ROOT:-$(cfg dataset.local_datasets_root)}"

# Generation
TEMPERATURE="${EVAL_TEMPERATURE:-$(cfg generation.temperature "0.0")}"
TOP_P="${EVAL_TOP_P:-$(cfg generation.top_p "1.0")}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-$(cfg generation.max_new_tokens "2048")}"
SEED="${SEED:-$(cfg generation.seed "7")}"

# LightEval
LIGHTEVAL_ENABLED="${LIGHTEVAL_ENABLED:-$(cfg lighteval.enabled "true")}"
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-$(cfg lighteval.tasks "math_500,gsm8k")}"
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-$(cfg lighteval.custom_tasks "eval.lighteval.custom_tasks.offline_patch")}"
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-$(cfg lighteval.data_parallel_size "1")}"
LIGHTEVAL_PROMPT_MODE="${LIGHTEVAL_PROMPT_MODE:-$(cfg lighteval.prompt_mode "system")}"
LIGHTEVAL_SYSTEM_PROMPT="${LIGHTEVAL_SYSTEM_PROMPT:-$(cfg lighteval.system_prompt)}"
if [[ -z "$LIGHTEVAL_SYSTEM_PROMPT" ]]; then
  LIGHTEVAL_SYSTEM_PROMPT="$SYSTEM_PROMPT"
fi
LIGHTEVAL_GPU_MEM="${LIGHTEVAL_GPU_MEMORY_UTILIZATION:-$(cfg lighteval.gpu_memory_utilization "0.9")}"
LIGHTEVAL_MAX_MODEL_LENGTH="${LIGHTEVAL_MAX_MODEL_LENGTH:-$(cfg lighteval.max_model_length "4096")}"
LIGHTEVAL_DTYPE="${LIGHTEVAL_DTYPE:-$(cfg lighteval.dtype "bfloat16")}"

# ZeroEval
ZEROEVAL_ENABLED="${ZEROEVAL_ENABLED:-$(cfg zeroeval.enabled "true")}"
ZEROEVAL_DIR="${ZEROEVAL_DIR:-$(cfg zeroeval.repo_dir "third_party/ZeroEval")}"
ZEROEVAL_TP_SIZE="${ZEROEVAL_TP_SIZE:-$(cfg zeroeval.tp_size "1")}"
ZEROEVAL_BATCH_SIZE="${ZEROEVAL_BATCH_SIZE:-$(cfg zeroeval.batch_size "4")}"
ZEROEVAL_MAX_TOKENS="${ZEROEVAL_MAX_TOKENS:-$(cfg zeroeval.max_tokens "2048")}"
ZEROEVAL_GPU_MEM="${ZEROEVAL_GPU_MEMORY_UTILIZATION:-$(cfg zeroeval.gpu_memory_utilization "0.95")}"
ZEROEVAL_MAX_SAMPLES="${ZEROEVAL_MAX_SAMPLES:-$(cfg zeroeval.max_samples "800")}"

# HACRL native eval (uses evaluate_checkpoints.py with pluggable scorer)
HACRL_EVAL_ENABLED="${HACRL_EVAL_ENABLED:-$(cfg hacrl.enabled "false")}"
HACRL_EVAL_TASKS="${HACRL_EVAL_TASKS:-$(cfg hacrl.tasks "math_500,gsm8k")}"
HACRL_EVAL_SCORER="${HACRL_EVAL_SCORER:-$(cfg hacrl.scorer "default")}"
HACRL_EVAL_NUM_GENERATIONS="${HACRL_EVAL_NUM_GENERATIONS:-$(cfg hacrl.num_generations "1")}"
HACRL_EVAL_MAX_SAMPLES="${HACRL_EVAL_MAX_SAMPLES:-$(cfg hacrl.max_samples "")}"
HACRL_EVAL_USE_OFFLINE="${HACRL_EVAL_USE_OFFLINE:-$(cfg hacrl.use_offline "true")}"
HACRL_EVAL_GPU_MEM="${HACRL_EVAL_GPU_MEM:-$(cfg hacrl.gpu_memory_utilization "0.9")}"
HACRL_EVAL_TP_SIZE="${HACRL_EVAL_TP_SIZE:-$(cfg hacrl.tensor_parallel_size "1")}"
HACRL_EVAL_DTYPE="${HACRL_EVAL_DTYPE:-$(cfg hacrl.dtype "bfloat16")}"
HACRL_EVAL_MAX_MODEL_LEN="${HACRL_EVAL_MAX_MODEL_LEN:-$(cfg hacrl.max_model_len "")}"
HACRL_EVAL_CUDA_DEVICE="${HACRL_EVAL_CUDA_DEVICE:-$(cfg hacrl.cuda_device "")}"
HACRL_EVAL_VLLM_BATCH_SIZE="${HACRL_EVAL_VLLM_BATCH_SIZE:-$(cfg hacrl.vllm_batch_size "16")}"

# TTRL eval (uses eval.ttrl.runner with TTRL-style scoring)
TTRL_EVAL_ENABLED="${TTRL_EVAL_ENABLED:-$(cfg ttrl.enabled "false")}"
TTRL_EVAL_TASKS="${TTRL_EVAL_TASKS:-$(cfg ttrl.tasks "math_ttt,aime_ttt")}"
TTRL_EVAL_SCORER="${TTRL_EVAL_SCORER:-$(cfg ttrl.scorer "ttrl_math")}"
TTRL_EVAL_NUM_GENERATIONS="${TTRL_EVAL_NUM_GENERATIONS:-$(cfg ttrl.num_generations "1")}"
TTRL_EVAL_MAX_SAMPLES="${TTRL_EVAL_MAX_SAMPLES:-$(cfg ttrl.max_samples "")}"
TTRL_EVAL_PROMPT_MODE="${TTRL_EVAL_PROMPT_MODE:-$(cfg ttrl.prompt_mode "bare")}"
TTRL_EVAL_TEMPERATURE="${TTRL_EVAL_TEMPERATURE:-$(cfg ttrl.temperature "0.0")}"
TTRL_EVAL_TOP_P="${TTRL_EVAL_TOP_P:-$(cfg ttrl.top_p "1.0")}"
TTRL_EVAL_GPU_MEM="${TTRL_EVAL_GPU_MEM:-$(cfg ttrl.gpu_memory_utilization "0.9")}"
TTRL_EVAL_TP_SIZE="${TTRL_EVAL_TP_SIZE:-$(cfg ttrl.tensor_parallel_size "1")}"
TTRL_EVAL_DTYPE="${TTRL_EVAL_DTYPE:-$(cfg ttrl.dtype "bfloat16")}"
TTRL_EVAL_MAX_MODEL_LEN="${TTRL_EVAL_MAX_MODEL_LEN:-$(cfg ttrl.max_model_len "$(cfg vllm.max_model_len "4096")")}"
TTRL_EVAL_DATASET_OVERRIDES="${TTRL_EVAL_DATASET_OVERRIDES:-$(cfg_json ttrl.dataset_overrides)}"

# Output
LIGHTEVAL_OUTPUT_DIR="${LIGHTEVAL_OUTPUT_DIR:-$(cfg output.lighteval_dir "outputs/lighteval/eval/{checkpoint}")}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-$(cfg output.zeroeval_root "outputs/zeroeval/eval")}"
HACRL_EVAL_OUTPUT_ROOT="${HACRL_EVAL_OUTPUT_ROOT:-$(cfg output.hacrl_dir "outputs/hacrl_eval/eval")}"
TTRL_EVAL_OUTPUT_ROOT="${TTRL_EVAL_OUTPUT_ROOT:-$(cfg output.ttrl_dir "outputs/ttrl_eval/eval")}"

# ============================================================================
# Derived values
# ============================================================================

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_DATASETS_VERBOSITY="${HF_DATASETS_VERBOSITY:-error}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0

infer_checkpoint_tag() {
  local base
  base="$(basename "$(realpath "$1")")"
  if [[ "$base" == checkpoint-* ]]; then
    echo "$base"
  else
    echo "$(basename "$1")"
  fi
}

CHECKPOINT_TAG="$(infer_checkpoint_tag "$MODEL_PATH")"

# Expand {checkpoint} placeholder in output paths.
LIGHTEVAL_OUTPUT_DIR="${LIGHTEVAL_OUTPUT_DIR//\{checkpoint\}/$CHECKPOINT_TAG}"

# Offline mode: default on, override via env vars.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ============================================================================
# Summary
# ============================================================================
log "================================================================"
log "Evaluation (config-driven)"
log "  config:       $CONFIG_FILE"
log "  name:         $(cfg name '<unnamed>')"
log "  model:        $MODEL_PATH"
log "  model_tag:    $MODEL_TAG"
log "  checkpoint:   $CHECKPOINT_TAG"
log "  temperature:  $TEMPERATURE"
log "  top_p:        $TOP_P"
log "  max_tokens:   $MAX_NEW_TOKENS"
log "  seed:         $SEED"
log "  lighteval:    $LIGHTEVAL_ENABLED (tasks: $LIGHTEVAL_TASKS)"
log "  zeroeval:     $ZEROEVAL_ENABLED"
log "  hacrl:        $HACRL_EVAL_ENABLED (tasks: $HACRL_EVAL_TASKS, scorer: $HACRL_EVAL_SCORER)"
log "  ttrl:         $TTRL_EVAL_ENABLED (tasks: $TTRL_EVAL_TASKS, scorer: $TTRL_EVAL_SCORER, mode: $TTRL_EVAL_PROMPT_MODE)"
log "================================================================"

# ============================================================================
# LightEval
# ============================================================================
if [[ "$LIGHTEVAL_ENABLED" == "true" ]]; then
  LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT="$LIGHTEVAL_SYSTEM_PROMPT"
  case "$LIGHTEVAL_PROMPT_MODE" in
    bare|suffix)
      LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT=""
      ;;
    system|"")
      ;;
    *)
      log "WARNING: unknown lighteval.prompt_mode='$LIGHTEVAL_PROMPT_MODE', fallback to 'system'"
      ;;
  esac

  LIGHTEVAL_DONE_MARKER="$LIGHTEVAL_OUTPUT_DIR/.eval_done"
  if [[ -f "$LIGHTEVAL_DONE_MARKER" ]]; then
    log "LightEval already done (marker: $LIGHTEVAL_DONE_MARKER). Skipping."
  else
    log "--- LightEval: $LIGHTEVAL_TASKS ---"
    mkdir -p "$LIGHTEVAL_OUTPUT_DIR"

    LIGHTEVAL_CMD=(
      python -m eval.lighteval.runner
        --model_path "$MODEL_PATH"
        --tasks "$LIGHTEVAL_TASKS"
        --output_dir "$LIGHTEVAL_OUTPUT_DIR"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --seed "$SEED"
        --dtype "$LIGHTEVAL_DTYPE"
        --max_model_length "$LIGHTEVAL_MAX_MODEL_LENGTH"
        --gpu_memory_utilization "$LIGHTEVAL_GPU_MEM"
        --data_parallel_size "$LIGHTEVAL_DATA_PARALLEL_SIZE"
    )

    [[ -n "$LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT" ]] && LIGHTEVAL_CMD+=(--system_prompt "$LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT")
    [[ -n "$LIGHTEVAL_CUSTOM_TASKS" ]] && LIGHTEVAL_CMD+=(--custom_tasks "$LIGHTEVAL_CUSTOM_TASKS")

    if [[ -n "$LOCAL_DATASETS_ROOT" ]]; then
      LIGHTEVAL_CMD+=(
        --local_datasets_root "$LOCAL_DATASETS_ROOT"
        --hf_home "$LOCAL_DATASETS_ROOT/hf_home"
        --datasets_cache "$LOCAL_DATASETS_ROOT"
        --transformers_cache "$LOCAL_DATASETS_ROOT/transformers_cache"
      )
    fi

    "${LIGHTEVAL_CMD[@]}"
    date -Iseconds > "$LIGHTEVAL_DONE_MARKER"
    log "LightEval done. Results: $LIGHTEVAL_OUTPUT_DIR"
  fi
fi

# ============================================================================
# ZeroEval (CRUXEval)
# ============================================================================
if [[ "$ZEROEVAL_ENABLED" == "true" ]]; then
  ZEROEVAL_DONE_MARKER="$ZEROEVAL_OUTPUT_ROOT/cruxeval/.eval_done_${MODEL_TAG}_${CHECKPOINT_TAG}"
  if [[ -f "$ZEROEVAL_DONE_MARKER" ]]; then
    log "ZeroEval already done (marker: $ZEROEVAL_DONE_MARKER). Skipping."
  else
    log "--- ZeroEval (CRUXEval) ---"

    ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
    ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
    ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
    ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEM" \
    python -m eval.zeroeval.runner \
      --model_path "$MODEL_PATH" \
      --model_tag "$MODEL_TAG" \
      --checkpoint_tag "$CHECKPOINT_TAG" \
      --output_root "$ZEROEVAL_OUTPUT_ROOT" \
      --zeroeval_dir "$ZEROEVAL_DIR" \
      --mode zeroeval_repo \
      --engine vllm \
      --tensor_parallel_size "$ZEROEVAL_TP_SIZE" \
      --dtype bfloat16 \
      --gpu_memory_utilization "$ZEROEVAL_GPU_MEM" \
      --batch_size "$ZEROEVAL_BATCH_SIZE" \
      --max_tokens "$ZEROEVAL_MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --seed "$SEED" \
      --max_samples "$ZEROEVAL_MAX_SAMPLES"

    mkdir -p "$(dirname "$ZEROEVAL_DONE_MARKER")"
    date -Iseconds > "$ZEROEVAL_DONE_MARKER"
    log "ZeroEval done. Results: $ZEROEVAL_OUTPUT_ROOT"
  fi
fi

# ============================================================================
# HACRL Native Eval (evaluate_checkpoints.py with pluggable scorer)
# ============================================================================
if [[ "$HACRL_EVAL_ENABLED" == "true" ]]; then
  HACRL_DONE_MARKER="$HACRL_EVAL_OUTPUT_ROOT/.hacrl_done_${MODEL_TAG}_${CHECKPOINT_TAG}"
  if [[ -f "$HACRL_DONE_MARKER" ]]; then
    log "HACRL eval already done (marker: $HACRL_DONE_MARKER). Skipping."
  else
    log "--- HACRL Eval: tasks=$HACRL_EVAL_TASKS scorer=$HACRL_EVAL_SCORER offline=$HACRL_EVAL_USE_OFFLINE ---"
    run_hacrl_eval "$MODEL_PATH" "$MODEL_TAG" "$CHECKPOINT_TAG" "$HACRL_EVAL_OUTPUT_ROOT" \
      "$HACRL_EVAL_TASKS" "$HACRL_EVAL_SCORER" "$SYSTEM_PROMPT" "$SEED" \
      "$MAX_NEW_TOKENS" "$HACRL_EVAL_VLLM_BATCH_SIZE" \
      "$HACRL_EVAL_NUM_GENERATIONS" "$HACRL_EVAL_MAX_SAMPLES" \
      "$TEMPERATURE" "$TOP_P" \
      "$HACRL_EVAL_USE_OFFLINE" "$HACRL_EVAL_GPU_MEM" "$HACRL_EVAL_TP_SIZE" \
      "$HACRL_EVAL_DTYPE" "$HACRL_EVAL_MAX_MODEL_LEN" "$HACRL_EVAL_CUDA_DEVICE"
    log "HACRL eval done. Results: $HACRL_EVAL_OUTPUT_ROOT"
  fi
fi

log "================================================================"

# ============================================================================
# TTRL Eval (TTRL-style scoring on TTT datasets)
# ============================================================================
if [[ "$TTRL_EVAL_ENABLED" == "true" ]]; then
  log "--- TTRL Eval: tasks=$TTRL_EVAL_TASKS scorer=$TTRL_EVAL_SCORER mode=$TTRL_EVAL_PROMPT_MODE ---"
  run_ttrl_eval "$MODEL_PATH" "$MODEL_TAG" "$CHECKPOINT_TAG" "$TTRL_EVAL_OUTPUT_ROOT" \
    "$TTRL_EVAL_TASKS" "$TTRL_EVAL_SCORER" "$SEED" \
    "$MAX_NEW_TOKENS" "$TTRL_EVAL_NUM_GENERATIONS" "$TTRL_EVAL_MAX_SAMPLES" \
    "$TTRL_EVAL_TEMPERATURE" "$TTRL_EVAL_TOP_P" \
    "$TTRL_EVAL_PROMPT_MODE" \
    "$TTRL_EVAL_GPU_MEM" "$TTRL_EVAL_TP_SIZE" "$TTRL_EVAL_DTYPE" \
    "$TTRL_EVAL_MAX_MODEL_LEN" \
    "$TTRL_EVAL_DATASET_OVERRIDES"
  log "TTRL eval done. Results: $TTRL_EVAL_OUTPUT_ROOT"
fi

log "================================================================"
log "Evaluation complete."
[[ "$LIGHTEVAL_ENABLED" == "true" ]] && log "  LightEval: $LIGHTEVAL_OUTPUT_DIR"
[[ "$ZEROEVAL_ENABLED" == "true" ]] && log "  ZeroEval:  $ZEROEVAL_OUTPUT_ROOT"
[[ "$HACRL_EVAL_ENABLED" == "true" ]] && log "  HACRL:     $HACRL_EVAL_OUTPUT_ROOT"
[[ "$TTRL_EVAL_ENABLED" == "true" ]] && log "  TTRL:      $TTRL_EVAL_OUTPUT_ROOT"
log "================================================================"
