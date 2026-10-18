#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Prompt-mode ablation: compare suffix / bare on TTRL evaluation.
#
# Part 1: test_ttrl_eval.py — evaluate on math_500 & gsm8k (suffix / bare).
# Part 2: test_max_tokens_extraction.py — MV on math_lighteval (bare only).
#
# Usage:
#   # Minimal (uses defaults):
#   CUDA_VISIBLE_DEVICES=0 bash ablation/run_prompt_mode_ablation.sh \
#       --model /path/to/Qwen2.5-Math-1.5B
#
#   # Full customisation:
#   CUDA_VISIBLE_DEVICES=0 bash ablation/run_prompt_mode_ablation.sh \
#       --model /path/to/Qwen2.5-Math-1.5B \
#       --datasets "math_500 gsm8k" \
#       --scorer ttrl_math \
#       --num_generations 16 \
#       --configs "3072:1024 4096:1024" \
#       --output outputs/ablation/prompt_mode
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT" || exit 1

# ---- Defaults ----
MODEL_PATH=""
DATASETS="math_500 gsm8k"
SCORER="ttrl_math"
NUM_GENERATIONS=16
TEMPERATURE=0.6
TOP_P=0.95
MAX_NEW_TOKENS=4096
CONFIGS="3072:1024 4096:1024"
NUM_PROBLEMS=0                # 0 = all
OUTPUT_ROOT="outputs/ablation/prompt_mode"
GPU_MEM=0.9
TP=1
DTYPE="bfloat16"
SEED=7

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)            MODEL_PATH="$2";       shift 2 ;;
    --datasets)         DATASETS="$2";         shift 2 ;;
    --scorer)           SCORER="$2";           shift 2 ;;
    --num_generations)  NUM_GENERATIONS="$2";  shift 2 ;;
    --temperature)      TEMPERATURE="$2";      shift 2 ;;
    --top_p)            TOP_P="$2";            shift 2 ;;
    --max_new_tokens)   MAX_NEW_TOKENS="$2";   shift 2 ;;
    --configs)          CONFIGS="$2";          shift 2 ;;
    --num_problems)     NUM_PROBLEMS="$2";     shift 2 ;;
    --output)           OUTPUT_ROOT="$2";      shift 2 ;;
    --gpu_mem)          GPU_MEM="$2";          shift 2 ;;
    --tp)               TP="$2";               shift 2 ;;
    --seed)             SEED="$2";             shift 2 ;;
    --help|-h)
      sed -n '2,/^# =====/p' "$0" | head -n -1; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[[ -z "$MODEL_PATH" ]] && { echo "ERROR: --model is required."; exit 1; }

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "================================================================"
log "Prompt-mode ablation"
log "  model:           $MODEL_PATH"
log "  datasets:        $DATASETS"
log "  scorer:          $SCORER"
log "  num_generations: $NUM_GENERATIONS"
log "  temperature:     $TEMPERATURE"
log "  configs (MV):    $CONFIGS"
log "  output:          $OUTPUT_ROOT"
log "================================================================"

# ---- Part 1: test_ttrl_eval with suffix and bare modes ----
for MODE in suffix bare; do
  OUT="${OUTPUT_ROOT}/ttrl_eval/${MODE}"
  log ""
  log "━━━  test_ttrl_eval  prompt_mode=${MODE}  ━━━"

  python -m ablation.test_ttrl_eval \
    --model_paths "$MODEL_PATH" \
    --prompt_mode "$MODE" \
    --datasets $DATASETS \
    --scorer "$SCORER" \
    --num_generations 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --gpu_memory_utilization "$GPU_MEM" \
    --tensor_parallel_size "$TP" \
    --dtype "$DTYPE" \
    --output_dir "$OUT" \
    2>&1 | tee "${OUT}.log" || log "WARNING: test_ttrl_eval (${MODE}) exited non-zero"
done

# ---- Part 2: test_max_tokens_extraction with MV (bare mode, math_lighteval) ----
MV_MODE="bare"
MV_DATASET="math_lighteval"

OUT="${OUTPUT_ROOT}/max_tokens_mv/${MV_MODE}"
log ""
log "━━━  test_max_tokens_extraction  prompt_mode=${MV_MODE}  dataset=${MV_DATASET}  ━━━"

python -m ablation.test_max_tokens_extraction \
  --model_path "$MODEL_PATH" \
  --prompt_mode "$MV_MODE" \
  --configs $CONFIGS \
  --dataset_key "$MV_DATASET" \
  --scorer "$SCORER" \
  --num_problems "$NUM_PROBLEMS" \
  --num_generations "$NUM_GENERATIONS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --seed "$SEED" \
  --gpu_memory_utilization "$GPU_MEM" \
  --tensor_parallel_size "$TP" \
  --dtype "$DTYPE" \
  --output_dir "$OUT" \
  2>&1 | tee "${OUT}.log" || log "WARNING: test_max_tokens (${MV_MODE}) exited non-zero"

log ""
log "================================================================"
log "All prompt-mode ablation runs complete."
log "Results: $OUTPUT_ROOT"
log "================================================================"
