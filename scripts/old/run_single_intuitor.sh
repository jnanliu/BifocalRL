#!/usr/bin/env bash
set -euo pipefail

# Single-model Intuitor training runner.
#
# One policy model is trained iteratively using cached generations:
#   (1) policy generates via vLLM
#   (2) reward model scores via SCE
#   (3) policy trains one step from cache (GRPO loss)
#
# Defaults are aligned with open-r1 intuitor demo where applicable.

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_LOG_LEVEL=error
# Keep library logs quiet (match co-intuitor).
export DS_VERBOSE=0
export DS_LOG_LEVEL=WARNING
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
export HF_DATASETS_VERBOSITY=error
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_P2P_DISABLE=1
# vLLM multiprocessing: prefer spawn (open-r1 default).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Triton cache dir guard (avoid noisy missing-dir logs).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune" || true
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

export CO_ASSERT_INVARIANTS="${CO_ASSERT_INVARIANTS:-1}"

export CO_DEBUG_TRAIN_BOUNDARY="${CO_DEBUG_TRAIN_BOUNDARY:-1}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# Ensure we run from repo root (so `python -m src...` works).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# -----------------------------------------------------------------------------
# Required
# -----------------------------------------------------------------------------
POLICY_INIT_PATH="${POLICY_INIT_PATH:-/home/jizeyu26/jzy/7_co_learning/test/models/Llama-3.2-3B-Instruct}"
#home/chenzj/czj/INTUITOR/beifen/test/models/Qwen2.5-3B
#/home/jizeyu26/jzy/7_co_learning/test/models/Phi-4-mini-instruct
# Optional: if empty, score with policy itself (self-SCE).
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-}"

# -----------------------------------------------------------------------------
# Output/cache
# -----------------------------------------------------------------------------
LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR:-outputs/single-intuitor/llama/latest_checkpoint}"
MODEL_DIR="$LATEST_CHECKPOINT_DIR/model"

# ============================================================================
# Debug (enabled by default; comment out exports to disable)
# ============================================================================
export CO_DEBUG_NAN="${CO_DEBUG_NAN:-1}"
export CO_DEBUG_NAN_DIR="${CO_DEBUG_NAN_DIR:-$LATEST_CHECKPOINT_DIR/debug_nan}"
export CO_DEBUG_NAN_RAISE="${CO_DEBUG_NAN_RAISE:-1}"

export CO_DEBUG_DATA="${CO_DEBUG_DATA:-1}"
export CO_DEBUG_DATA_DIR="${CO_DEBUG_DATA_DIR:-$LATEST_CHECKPOINT_DIR/debug_data}"
export CO_DEBUG_DATA_N="${CO_DEBUG_DATA_N:-3}"

# -----------------------------------------------------------------------------
# Dataset / sampling
# -----------------------------------------------------------------------------
DATASET_KEY="${DATASET_KEY:-math_lighteval}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SYSTEM_PROMPT="You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}. Please stop generation immediately after outputing the box."
PROBLEMS_PER_STEP="${PROBLEMS_PER_STEP:-128}"

# -----------------------------------------------------------------------------
# Generation (open-r1 defaults)
# -----------------------------------------------------------------------------
NUM_GENERATIONS="${NUM_GENERATIONS:-7}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"

# vLLM
VLLM_PORT="${VLLM_PORT:-7900}"
AUTO_START_SERVER="${AUTO_START_SERVER:-false}"
AUTO_STOP_SERVER="${AUTO_STOP_SERVER:-true}"

# Optional: start/stop a per-step vLLM server (like co-intuitor).
USE_STEP_VLLM_SERVER="${USE_STEP_VLLM_SERVER:-true}"
VLLM_GPU="${VLLM_GPU:-0}"
VLLM_SERVER_START_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-900}"
VLLM_SERVER_GPU_MEMORY_UTILIZATION="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_SERVER_DTYPE="${VLLM_SERVER_DTYPE:-bfloat16}"
VLLM_SERVER_MAX_MODEL_LEN="${VLLM_SERVER_MAX_MODEL_LEN:-32768}"
VLLM_SERVER_MAX_NUM_SEQS="${VLLM_SERVER_MAX_NUM_SEQS:-128}"
VLLM_SERVER_MAX_NUM_BATCHED_TOKENS="${VLLM_SERVER_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_SERVER_CHUNKED_PREFILL="${VLLM_SERVER_CHUNKED_PREFILL:-true}" # true|false

# -----------------------------------------------------------------------------
# Training (open-r1-ish defaults, override as needed)
# -----------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-2}"
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-zero3}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/${ACCELERATOR_CONFIG}.yaml}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-224}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_STEPS="${MAX_STEPS:-20}"
SEED="${SEED:-7}"
BF16="${BF16:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"

# Sampling mode (match open-r1-intuitor style when true)
# Default true: you want to test randomized sampling.
SHUFFLE_DATASET="${SHUFFLE_DATASET:-true}"

# KL (beta>0 enables ref_logps computation in generate step)
BETA="${BETA:-0.0005}"
REF_MODEL_PATH="${REF_MODEL_PATH:-$POLICY_INIT_PATH}"
LOGPS_BATCH_SIZE="${LOGPS_BATCH_SIZE:-16}"
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-4}"
# NOTE: Default should be "unset/false". Only enable when explicitly requested.
MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-false}"

# -----------------------------------------------------------------------------
# Evaluation (each step)
# -----------------------------------------------------------------------------
EVAL_BACKEND="${EVAL_BACKEND:-lighteval}" # lighteval | none
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"

# LightEval (built-in tasks + offline/local datasets redirect)
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-math_500,gsm8k}"
# Patch dataset IO (defines no tasks; safe with built-ins).
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-eval.lighteval.custom_tasks.offline_patch}"

# Local datasets root (used by offline_patch via LOCAL_DATASETS_ROOT env var).
LIGHTEVAL_LOCAL_DATASETS_ROOT="${LIGHTEVAL_LOCAL_DATASETS_ROOT:-${LOCAL_DATASETS_ROOT:-/home/jizeyu26/jzy/7_co_learning/test/datasets}}"

# Where LightEval (datasets/transformers) will read/write caches.
LIGHTEVAL_HF_HOME="${LIGHTEVAL_HF_HOME:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/hf_home}"
LIGHTEVAL_DATASETS_CACHE="${LIGHTEVAL_DATASETS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT}"
LIGHTEVAL_TRANSFORMERS_CACHE="${LIGHTEVAL_TRANSFORMERS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/transformers_cache}"
# Default to 1 to avoid Ray; override to use multi-GPU eval.
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-1}"
# Sampling overrides (make math-500 + LCB fast by default).


# ZeroEval (CRUXEval) parameters
ZEROEVAL_DIR="${ZEROEVAL_DIR:-third_party/ZeroEval}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-outputs/zeroeval/single-intuitor}"
ZEROEVAL_TP_SIZE="${ZEROEVAL_TP_SIZE:-2}"
ZEROEVAL_BATCH_SIZE="${ZEROEVAL_BATCH_SIZE:-100}"
ZEROEVAL_MAX_TOKENS="${ZEROEVAL_MAX_TOKENS:-2048}"
ZEROEVAL_GPU_MEMORY_UTILIZATION="${ZEROEVAL_GPU_MEMORY_UTILIZATION:-0.95}"
ZEROEVAL_MAX_SAMPLES="${ZEROEVAL_MAX_SAMPLES:-800}"

check_required() {
  if [ "$POLICY_INIT_PATH" = "/path/to/policy" ]; then error "Set POLICY_INIT_PATH"; fi
}

init_dirs() {
  mkdir -p "$LATEST_CHECKPOINT_DIR" "$MODEL_DIR"
}

prune_old_checkpoints() {
  local model_dir="$1"
  if [ -z "$model_dir" ] || [ ! -d "$model_dir" ]; then
    return 0
  fi
  local latest
  latest=$(ls -td "$model_dir"/checkpoint-* 2>/dev/null | head -n 1 || true)
  if [ -z "$latest" ]; then
    return 0
  fi
  for ckpt in "$model_dir"/checkpoint-*; do
    if [ ! -d "$ckpt" ]; then
      continue
    fi
    if [ "$ckpt" != "$latest" ]; then
      log "Pruning old checkpoint: $ckpt"
      rm -rf "$ckpt" || true
    fi
  done
}

_vllm_port_pids() { lsof -Pi ":${1}" -sTCP:LISTEN -t 2>/dev/null || true; }
_vllm_kill_port() {
  local port="$1"
  local pids
  pids="$(_vllm_port_pids "$port")"
  if [ -z "$pids" ]; then return 0; fi
  log "Stopping vLLM on port ${port} (pids: ${pids})"
  kill -TERM $pids >/dev/null 2>&1 || true
  sleep 2
  pids="$(_vllm_port_pids "$port")"
  if [ -n "$pids" ]; then kill -KILL $pids >/dev/null 2>&1 || true; fi
}
_vllm_wait_health() {
  local port="$1"; local timeout_s="${2:-600}"
  local start_ts; start_ts="$(date +%s)"
  while true; do
    if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then return 0; fi
    local now_ts; now_ts="$(date +%s)"
    if [ $((now_ts - start_ts)) -ge "$timeout_s" ]; then return 1; fi
    sleep 2
  done
}
start_step_vllm_server() {
  local model_path="$1"
  if [ "$USE_STEP_VLLM_SERVER" != "true" ]; then
    return 0
  fi
  log "Starting per-step vLLM server..."
  log "  port=${VLLM_PORT}, gpu=${VLLM_GPU}, model=${model_path}"
  _vllm_kill_port "$VLLM_PORT"
  ( CUDA_VISIBLE_DEVICES="$VLLM_GPU" VLLM_SERVER_DEV_MODE=1 \
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --host 0.0.0.0 \
      --port "$VLLM_PORT" \
      --enable-sleep-mode \
      --gpu-memory-utilization "$VLLM_SERVER_GPU_MEMORY_UTILIZATION" \
      --dtype "$VLLM_SERVER_DTYPE" \
      --trust-remote-code \
      --max-model-len "$VLLM_SERVER_MAX_MODEL_LEN" \
      --max-num-seqs "$VLLM_SERVER_MAX_NUM_SEQS" \
      --max-num-batched-tokens "$VLLM_SERVER_MAX_NUM_BATCHED_TOKENS" \
      $( [[ "$VLLM_SERVER_CHUNKED_PREFILL" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
  ) >"vllm_server_${VLLM_PORT}.log" 2>&1 &
  if ! _vllm_wait_health "$VLLM_PORT" "$VLLM_SERVER_START_TIMEOUT"; then
    _vllm_kill_port "$VLLM_PORT"
    error "Timed out waiting for vLLM server on port $VLLM_PORT"
  fi
  log "Per-step vLLM server is healthy."
}
stop_step_vllm_server() {
  if [ "$USE_STEP_VLLM_SERVER" != "true" ]; then
    return 0
  fi
  _vllm_kill_port "$VLLM_PORT"
  log "Per-step vLLM server stopped."
}

get_latest_checkpoint() {
  if [ -d "$MODEL_DIR" ]; then
    local last
    last=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$last" ]; then
      echo "$last"
      return
    fi
  fi
  echo "$POLICY_INIT_PATH"
}

check_required
init_dirs
prune_old_checkpoints "$MODEL_DIR"

log "Single Intuitor run"
log "  policy_init: $POLICY_INIT_PATH"
log "  reward_model: ${REWARD_MODEL_PATH:-<policy (self-SCE)>}"
log "  out: $LATEST_CHECKPOINT_DIR"
log "  dataset: $DATASET_KEY"
log "  steps: $MAX_STEPS"

for step in $(seq 3 $((MAX_STEPS - 1))); do
  log "=================================================================================="
  log "Step $step: Generate+Score -> Train (intuitor)"
  log "=================================================================================="

  POLICY_PATH=$(get_latest_checkpoint)
  log "Policy: $POLICY_PATH"

  # Run generate+score under the same visible GPU set.
  GEN_GPU_LIST=""
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -z "$GEN_GPU_LIST" ]; then GEN_GPU_LIST="$i"; else GEN_GPU_LIST="$GEN_GPU_LIST,$i"; fi
  done
  export CUDA_VISIBLE_DEVICES="$GEN_GPU_LIST"

  GEN_ARGS=(
    --policy_model_path "$POLICY_PATH"
    --dataset_key "$DATASET_KEY"
    --output_dir "$LATEST_CHECKPOINT_DIR"
    --step "$step"
    --seed "$SEED"
    --problems_per_step "$PROBLEMS_PER_STEP"
    --num_generations "$NUM_GENERATIONS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --repetition_penalty "$REPETITION_PENALTY"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --scoring_batch_size "$SCORING_BATCH_SIZE"
    --logps_batch_size "$LOGPS_BATCH_SIZE"
    --vllm_port "$VLLM_PORT"
    --beta "$BETA"
    --ref_model_path "$REF_MODEL_PATH"
    --num_iterations "$NUM_ITERATIONS"
  )
  if [ -n "$REWARD_MODEL_PATH" ]; then GEN_ARGS+=(--reward_model_path "$REWARD_MODEL_PATH"); fi

  if [ -n "$MAX_SAMPLES" ]; then GEN_ARGS+=(--max_samples "$MAX_SAMPLES"); fi
  if [ -n "$SYSTEM_PROMPT" ]; then GEN_ARGS+=(--system_prompt "$SYSTEM_PROMPT"); fi
  if [ "$MASK_TRUNCATED_COMPLETIONS" = "true" ]; then GEN_ARGS+=(--mask_truncated_completions); fi
  if [ "$AUTO_START_SERVER" = "true" ]; then GEN_ARGS+=(--auto_start_server); else GEN_ARGS+=(--no-auto_start_server); fi
  if [ "$AUTO_STOP_SERVER" = "true" ]; then GEN_ARGS+=(--auto_stop_server); else GEN_ARGS+=(--no-auto_stop_server); fi
  if [ "$SHUFFLE_DATASET" = "true" ]; then GEN_ARGS+=(--shuffle_dataset); else GEN_ARGS+=(--no-shuffle_dataset); fi

  start_step_vllm_server "$POLICY_PATH"
  set +e
  accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "${CO_MAIN_PROCESS_PORT_GEN:-26100}" \
    src/pipelines/generate_and_score_single.py "${GEN_ARGS[@]}"
  GEN_RC=$?
  set -e
  stop_step_vllm_server
  if [ "$GEN_RC" -ne 0 ]; then
    error "Generation/scoring/logps failed for step $step"
  fi

  # Prepare CUDA_VISIBLE_DEVICES for training (so accelerate reliably sees NUM_GPUS GPUs).
  TRAIN_GPU_LIST=""
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -z "$TRAIN_GPU_LIST" ]; then
      TRAIN_GPU_LIST="$i"
    else
      TRAIN_GPU_LIST="$TRAIN_GPU_LIST,$i"
    fi
  done
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_LIST"

  # Skip training if this step's checkpoint already exists.
  CKPT_DIR="$MODEL_DIR/checkpoint-$step"
  if [ -d "$CKPT_DIR" ]; then
    log "Checkpoint already exists at step $step: $CKPT_DIR"
    log "Skipping training for step $step."
  else
  TRAIN_ARGS=(
    --policy_model_path "$POLICY_PATH"
    --cache_dir "$LATEST_CHECKPOINT_DIR"
    --step "$step"
    --reward_mode "intuitor"
    --output_dir "$MODEL_DIR"
    --max_steps 1
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --weight_decay "$WEIGHT_DECAY"
    --seed "$SEED"
    --beta "$BETA"
    --epsilon 0.2
    --num_iterations "$NUM_ITERATIONS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --logging_steps 1
    --logging_strategy steps
    --logging_first_step
    --save_steps 1
    --save_strategy steps
    --save_total_limit 1
    --overwrite_output_dir
    --torch_dtype "$TORCH_DTYPE"
    --trust_remote_code
  )
  if [ -n "$SYSTEM_PROMPT" ]; then TRAIN_ARGS+=(--system_prompt "$SYSTEM_PROMPT"); fi
  if [ "$MASK_TRUNCATED_COMPLETIONS" = "true" ]; then TRAIN_ARGS+=(--mask_truncated_completions); fi

  # Keep flags style consistent with co-intuitor.
  if [ "$BF16" = "true" ]; then TRAIN_ARGS+=(--bf16); else TRAIN_ARGS+=(--no_bf16); fi
  if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then TRAIN_ARGS+=(--gradient_checkpointing); else TRAIN_ARGS+=(--no_gradient_checkpointing); fi

  # Avoid "port 29500 already in use" when multiple jobs run on the same node.
  # Set CO_MAIN_PROCESS_PORT_TRAIN to a fixed port if you need determinism.
  # IMPORTANT: do NOT default to 0 for DeepSpeed/NCCL rendezvous; use a real free port to avoid hangs/retries.
  if [ -f "$ACCELERATE_CONFIG" ]; then
    accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT_TRAIN:-25950}" src/training/single_intuitor_train.py "${TRAIN_ARGS[@]}"
  else
    accelerate launch --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT_TRAIN:-25950}" src/training/single_intuitor_train.py "${TRAIN_ARGS[@]}"
  fi
  fi

  prune_old_checkpoints "$MODEL_DIR"

  # Step eval
  if [ "$EVAL_BACKEND" = "lighteval" ]; then
    CKPT_PATH="$MODEL_DIR/checkpoint-$step"
    OUT_DIR="outputs/lighteval/single-intuitor/llama/checkpoint-$step"
    DONE_MARKER="$OUT_DIR/.eval_done"
    mkdir -p "$OUT_DIR"
    log "--- Step $step: Evaluate (LightEval) ---"
    log "  model: $CKPT_PATH"
    log "  tasks: $LIGHTEVAL_TASKS"

    # Prefer offline-by-default (your network is unreliable).
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

    if [ -f "$DONE_MARKER" ]; then
      log "Eval already done for step $step. Skipping. Marker: $DONE_MARKER"
    else
      python -m eval.lighteval.runner \
        --model_path "$CKPT_PATH" \
        --tasks "$LIGHTEVAL_TASKS" \
        --output_dir "$OUT_DIR" \
        --system_prompt "$SYSTEM_PROMPT" \
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
        --temperature 0.0 \
        --top_p 1.0 \
        --seed "$SEED" \
        --data_parallel_size "$LIGHTEVAL_DATA_PARALLEL_SIZE" \
        --custom_tasks "$LIGHTEVAL_CUSTOM_TASKS" \
        --local_datasets_root "$LIGHTEVAL_LOCAL_DATASETS_ROOT" \
        --hf_home "$LIGHTEVAL_HF_HOME" \
        --datasets_cache "$LIGHTEVAL_DATASETS_CACHE" \
        --transformers_cache "$LIGHTEVAL_TRANSFORMERS_CACHE"
      date -Iseconds > "$DONE_MARKER"
    fi

    log "--- Step $step: Evaluate (ZeroEval CRUX) ---"
    ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
    ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
    ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
    ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
    python -m eval.zeroeval.runner \
      --model_path "$CKPT_PATH" \
      --model_tag a \
      --checkpoint_tag "checkpoint-$step" \
      --output_root "$ZEROEVAL_OUTPUT_ROOT" \
      --mode zeroeval_repo \
      --zeroeval_dir "$ZEROEVAL_DIR" \
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
  fi
done

log "Done. Latest checkpoints under: $MODEL_DIR"

