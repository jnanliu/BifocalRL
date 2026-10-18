#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Batch TTRL evaluation for AIME/AMC style datasets.
#
# Default behavior matches recipes/configs/eval_aime.json:
#   - prompt_mode: suffix (no system prompt, use per-dataset TTRL suffix)
#   - scorer: ttrl_math
#   - datasets: amc23 aime24 aime25
#   - dataset_overrides:
#       aime24/aime25/amc23 -> num_generations=32, temperature=0.6, top_p=0.95
#
# Prompt rendering uses tokenizer's own chat_template when available
# (handled inside ablation.test_ttrl_eval -> eval.runner -> prompt_utils).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash ablation/run_batch_ttrl_eval.sh
#
#   # Override defaults via env vars:
#   DATASETS="amc23 aime24 aime25" \
#   CUDA_VISIBLE_DEVICES=0 bash ablation/run_batch_ttrl_eval.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT" || exit 1

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EDIT HERE: list all model paths to evaluate                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MODELS=(
    /home/jizeyu26/jzy/7_co_learning/test/models/Qwen2.5-3B
    /home/jizeyu26/jzy/7_co_learning/test/models/Llama-3.2-3B-Instruct
    /home/jizeyu26/jzy/7_co_learning/test/models/Qwen3-4B-Base
    /home/jizeyu26/jzy/7_co_learning/test/models/Phi-4-mini-instruct
    # /path/to/another-model
)

# ---- Configurable via env or defaults ----
DATASETS="${DATASETS:-amc23 aime24 aime25}"
NUM_GENERATIONS="${NUM_GENERATIONS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SEED="${SEED:-7}"
GPU_MEM="${GPU_MEM:-0.9}"
TP="${TP:-1}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablation/batch_eval}"

# TTRL eval settings
TTRL_SCORER="${TTRL_SCORER:-ttrl_math}"
TTRL_MODE="${TTRL_MODE:-suffix}"
TTRL_DATASET_OVERRIDES="${TTRL_DATASET_OVERRIDES:-{\"aime25\":{\"num_generations\":32,\"temperature\":0.6,\"top_p\":0.95},\"aime24\":{\"num_generations\":32,\"temperature\":0.6,\"top_p\":0.95},\"amc23\":{\"num_generations\":32,\"temperature\":0.6,\"top_p\":0.95}}}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "================================================================"
log "Batch dual-method evaluation"
log "  models:          ${#MODELS[@]}"
log "  datasets:        $DATASETS"
log "  TTRL:            mode=$TTRL_MODE  scorer=$TTRL_SCORER"
log "  ds_overrides:    $TTRL_DATASET_OVERRIDES"
log "  num_generations: $NUM_GENERATIONS"
log "  temperature:     $TEMPERATURE"
log "  output:          $OUTPUT_ROOT"
log "================================================================"
for m in "${MODELS[@]}"; do
  log "  - $m"
done
log "================================================================"

OVERALL=()

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_TAG="$(basename "$MODEL_PATH")"

  # ── TTRL eval (suffix + per-dataset overrides) ────────────────────────
  TTRL_OUT="${OUTPUT_ROOT}/${MODEL_TAG}/ttrl_suffix_aime_amc"
  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Model: ${MODEL_TAG}  |  TTRL eval (${TTRL_MODE}, ${TTRL_SCORER})"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  set +e
  python -m ablation.test_ttrl_eval \
    --model_paths "$MODEL_PATH" \
    --prompt_mode "$TTRL_MODE" \
    --datasets $DATASETS \
    --scorer "$TTRL_SCORER" \
    --num_generations "$NUM_GENERATIONS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --dataset_overrides "$TTRL_DATASET_OVERRIDES" \
    --gpu_memory_utilization "$GPU_MEM" \
    --tensor_parallel_size "$TP" \
    --dtype "$DTYPE" \
    --output_dir "$TTRL_OUT" \
    2>&1 | tee "${TTRL_OUT}.log"
  RC_TTRL=${PIPESTATUS[0]}
  set -e

  if [[ $RC_TTRL -ne 0 ]]; then
    log "WARNING: ${MODEL_TAG} TTRL eval exited with code ${RC_TTRL}"
    OVERALL+=("${MODEL_TAG}/ttrl_suffix_aime_amc: FAILED (rc=${RC_TTRL})")
  else
    OVERALL+=("${MODEL_TAG}/ttrl_suffix_aime_amc: OK → ${TTRL_OUT}")
  fi
done

# ── Generate correctness heatmaps ──────────────────────────────────────
log ""
log "━━━  Generating per-problem correctness maps  ━━━"
python -m ablation.plot_correctness_map \
  --results_dir "$OUTPUT_ROOT" \
  --output_dir "${OUTPUT_ROOT}/plots" \
  2>&1 || log "WARNING: plot_correctness_map failed"

# ── Summary ────────────────────────────────────────────────────────────
log ""
log "================================================================"
log "All batch evaluations complete."
log "================================================================"
for entry in "${OVERALL[@]}"; do
  log "  $entry"
done
log ""
log "Results directory: $OUTPUT_ROOT"
log "Plots:            ${OUTPUT_ROOT}/plots/"
