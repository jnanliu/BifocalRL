#!/bin/bash

# Co-Intuitor Training Script
# Iterative training approach:
# For each step:
#   1. Generate and score responses (using vLLM, no accelerator)
#   2. Train Model A (policy=latest_checkpoint/model_a, ref=MODEL_A_PATH)
#   3. Train Model B (policy=latest_checkpoint/model_b, ref=MODEL_B_PATH)
#
# Usage:
#   ./scripts/run_co_intuitor.sh

set -e  # Exit on error

# CRITICAL: Set environment variables EARLY to disable verbose logging before any imports
export DS_VERBOSE=0
export DS_LOG_LEVEL=WARNING
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
# Silence common library logs (they can drown out our own logs).
export TRANSFORMERS_VERBOSITY=error
export HF_DATASETS_VERBOSITY=error
export ACCELERATE_LOG_LEVEL=error
export TORCH_DISTRIBUTED_DEBUG=OFF
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
# vLLM multiprocessing: prefer spawn (open-r1 default).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Triton (used by torch/flash-attn kernels) may run `df` on its autotune cache dir.
# Ensure the directory exists to avoid noisy "No such file or directory" logs.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune" || true
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_CUDA_ARCH_LIST="8.0"
# Get script directory for resolving relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ============================================================================
# Configuration
# ============================================================================

# Initial model paths (ref models - never change)
INITIAL_MODEL_A_PATH="${INITIAL_MODEL_A_PATH:-/home/jizeyu26/jzy/7_co_learning/test/models/Qwen2.5-3B}"
INITIAL_MODEL_B_PATH="${INITIAL_MODEL_B_PATH:-/home/jizeyu26/jzy/7_co_learning/test/models/Llama-3.1-8B-Instruct}"

# Latest checkpoint directory (contains cache, model_a, model_b)
LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR:-outputs/co-intuitor-7B/latest_checkpoint}"

# Dataset
DATASET_KEY="${DATASET_KEY:-math_lighteval}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SYSTEM_PROMPT="You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}. Please stop generation immediately after outputing the box."

# Generation parameters
NUM_GENERATIONS_PER_MODEL="${NUM_GENERATIONS_PER_MODEL:-4}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"

# Scoring parameters
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-4}"
LOGPS_BATCH_SIZE="${LOGPS_BATCH_SIZE:-16}"

# vLLM server settings
VLLM_PORT_A="${VLLM_PORT_A:-8030}"
VLLM_PORT_B="${VLLM_PORT_B:-8031}"
AUTO_START_SERVER="${AUTO_START_SERVER:-false}"
AUTO_STOP_SERVER="${AUTO_STOP_SERVER:-true}"
# Per-step external vLLM servers (start before generate_and_score, stop after)
# This is useful when auto-start inside accelerate conflicts/hangs.
USE_STEP_VLLM_SERVERS="${USE_STEP_VLLM_SERVERS:-true}"
VLLM_GPU_A="${VLLM_GPU_A:-0}"
VLLM_GPU_B="${VLLM_GPU_B:-1}"
VLLM_SERVER_START_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-900}"
VLLM_SERVER_GPU_MEMORY_UTILIZATION="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_SERVER_DTYPE="${VLLM_SERVER_DTYPE:-bfloat16}"
# Performance knobs (override via env). These are important when the model advertises very large max_seq_len
# (e.g. 131072), which can cause KV cache to hit 100% and slow the server dramatically mid-run.
VLLM_SERVER_MAX_MODEL_LEN="${VLLM_SERVER_MAX_MODEL_LEN:-4096}"
VLLM_SERVER_MAX_NUM_SEQS="${VLLM_SERVER_MAX_NUM_SEQS:-32}"
VLLM_SERVER_MAX_NUM_BATCHED_TOKENS="${VLLM_SERVER_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_SERVER_CHUNKED_PREFILL="${VLLM_SERVER_CHUNKED_PREFILL:-true}" # true|false

# Training parameters
MAX_STEPS="${MAX_STEPS:-20}"
LEARNING_RATE="${LEARNING_RATE:-3.0e-6}"
BETA="${BETA:-0.0005}"
SEED="${SEED:-7}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
PROBLEMS_PER_STEP="${PROBLEMS_PER_STEP:-128}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
# Sampling mode (match open-r1-intuitor style when true)
# Default true: you want to test randomized sampling.
SHUFFLE_DATASET="${SHUFFLE_DATASET:-true}"

# IMPORTANT:
# `run_co_intuitor.sh` historically hard-coded different micro-batch / accumulation settings for A vs B.
# Expose them as env-configurable knobs (matching `run_co_wmv.sh`) so experiments are reproducible and tunable.
GRADIENT_ACCUMULATION_STEPS_A="${GRADIENT_ACCUMULATION_STEPS_A:-112}"
GRADIENT_ACCUMULATION_STEPS_B="${GRADIENT_ACCUMULATION_STEPS_B:-112}"
PER_DEVICE_BATCH_SIZE_A="${PER_DEVICE_BATCH_SIZE_A:-2}"
PER_DEVICE_BATCH_SIZE_B="${PER_DEVICE_BATCH_SIZE_B:-2}"

# Reward method used inside `src/training/co_intuitor_train.py` when building rewards from cached scores/logps.
# - sum_scores: continuous reward = weight_a*score_a + weight_b*score_b (default; previous behavior)
# - co_wmv: WMV pseudo-label reward (binary) (only makes sense if you want co-WMV)
REWARD_METHOD="${REWARD_METHOD:-sum_scores}" # sum_scores | co_wmv
REWARD_WEIGHT_A="${REWARD_WEIGHT_A:-1.0}"
REWARD_WEIGHT_B="${REWARD_WEIGHT_B:-1.0}"

# ============================================================================
# Debug: NaN定位dump（好去除：删掉/注释掉下面3行export即可）
# - CO_DEBUG_NAN=1: 遇到非有限值时dump关键信息到jsonl
# - CO_DEBUG_NAN_DIR: dump目录（默认放在latest_checkpoint下）
# - CO_DEBUG_NAN_RAISE=1: dump后直接raise中断（建议开启，避免继续训练导致崩塌）
# ============================================================================
export CO_DEBUG_NAN="${CO_DEBUG_NAN:-1}"
export CO_DEBUG_NAN_DIR="${CO_DEBUG_NAN_DIR:-$LATEST_CHECKPOINT_DIR/debug_nan}"
export CO_DEBUG_NAN_RAISE="${CO_DEBUG_NAN_RAISE:-1}"

# ============================================================================
# Debug: 数据/对齐检查（generate_and_score + train prepare），好去除：删掉/注释掉下面3行export即可
# - CO_DEBUG_DATA=1: dump少量样本的token window/mask，用于核对是否与open-r1语义一致
# - CO_DEBUG_DATA_DIR: dump目录（建议放在latest_checkpoint下，便于清理）
# - CO_DEBUG_DATA_N: 每个阶段dump多少条样本（默认3）
# ============================================================================
export CO_DEBUG_DATA="${CO_DEBUG_DATA:-1}"
export CO_DEBUG_DATA_DIR="${CO_DEBUG_DATA_DIR:-$LATEST_CHECKPOINT_DIR/debug_data}"
export CO_DEBUG_DATA_N="${CO_DEBUG_DATA_N:-3}"

# Debug: print DeepSpeed accumulation boundaries + CUDA memory at update steps.
# Set to 0 to disable.
export CO_DEBUG_TRAIN_BOUNDARY="${CO_DEBUG_TRAIN_BOUNDARY:-1}"

# Evaluation parameters
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-400}"
EVAL_NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-1}"
EVAL_VLLM_BATCH_SIZE="${EVAL_VLLM_BATCH_SIZE:-400}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
EVAL_BACKEND="${EVAL_BACKEND:-lighteval}"  # custom | lighteval
# Use LightEval built-in tasks (no custom tasks).
# IMPORTANT: We still pass `--custom-tasks eval.lighteval.custom_tasks.offline_patch` to:
# - patch dataset IO so `math_500` / `gsm8k` load from LOCAL_DATASETS_ROOT local snapshots
# - keep prompts/metrics identical to LightEval built-ins
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-math_500,gsm8k}" # ,extended|lcb:codegeneration|0|0

# Offline/local datasets redirect patch (defines no tasks; safe to use with built-ins).
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-eval.lighteval.custom_tasks.offline_patch}"

# Local datasets root (used by offline_patch via LOCAL_DATASETS_ROOT env var).
LIGHTEVAL_LOCAL_DATASETS_ROOT="${LIGHTEVAL_LOCAL_DATASETS_ROOT:-${LOCAL_DATASETS_ROOT:-/home/jizeyu26/jzy/7_co_learning/test/datasets}}"

# Where LightEval (datasets/transformers) will read/write caches.
LIGHTEVAL_HF_HOME="${LIGHTEVAL_HF_HOME:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/hf_home}"
LIGHTEVAL_DATASETS_CACHE="${LIGHTEVAL_DATASETS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT}"
LIGHTEVAL_TRANSFORMERS_CACHE="${LIGHTEVAL_TRANSFORMERS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/transformers_cache}"
# LightEval data parallel size.
# NOTE: Using >1 will start Ray; eval_one_model.sh defaults to 1.
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-1}"
# Sampling overrides (make math-500 + LCB fast by default).


# ZeroEval (CRUXEval) parameters
ZEROEVAL_DIR="${ZEROEVAL_DIR:-third_party/ZeroEval}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-outputs/zeroeval/co-intuitor-7B}"
ZEROEVAL_TP_SIZE="${ZEROEVAL_TP_SIZE:-2}"
ZEROEVAL_BATCH_SIZE="${ZEROEVAL_BATCH_SIZE:-100}"
ZEROEVAL_MAX_TOKENS="${ZEROEVAL_MAX_TOKENS:-2048}"
ZEROEVAL_GPU_MEMORY_UTILIZATION="${ZEROEVAL_GPU_MEMORY_UTILIZATION:-0.95}"
ZEROEVAL_MAX_SAMPLES="${ZEROEVAL_MAX_SAMPLES:-800}"

# GPU configuration
NUM_GPUS="${NUM_GPUS:-4}"

# Accelerate config
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-zero3}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/${ACCELERATOR_CONFIG}.yaml}"

# Wandb configuration
# NOTE: Do NOT hardcode secrets in scripts. If you want wandb, export WANDB_API_KEY in your shell env.
WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -n "$WANDB_API_KEY" ]; then export WANDB_API_KEY; fi

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_required_vars() {
    if [ -z "$INITIAL_MODEL_A_PATH" ] || [ "$INITIAL_MODEL_A_PATH" = "/path/to/model_a" ]; then
        error "INITIAL_MODEL_A_PATH must be set"
    fi
    if [ -z "$INITIAL_MODEL_B_PATH" ] || [ "$INITIAL_MODEL_B_PATH" = "/path/to/model_b" ]; then
        error "INITIAL_MODEL_B_PATH must be set"
    fi
    if [ -z "$DATASET_KEY" ]; then
        error "DATASET_KEY must be set"
    fi
}

# Initialize latest_checkpoint directory structure
init_checkpoint_dir() {
    log "Initializing latest_checkpoint directory: $LATEST_CHECKPOINT_DIR"
    mkdir -p "$LATEST_CHECKPOINT_DIR"
    mkdir -p "$LATEST_CHECKPOINT_DIR/model_a"
    mkdir -p "$LATEST_CHECKPOINT_DIR/model_b"
    
    # Note: We don't copy initial models anymore. Ref models always use INITIAL_MODEL_A_PATH/B_PATH directly.
    # Trained models will be saved as checkpoint-k in model_a/model_b directories.
    log "Ref models will use INITIAL_MODEL paths directly (no copying)"

    # Disk guard: ensure we don't start with multiple checkpoints already on disk.
    prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_a"
    prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_b"
}

# Keep at most ONE checkpoint-* directory in `model_dir` (the newest by mtime).
# This prevents disk blow-up from accumulating checkpoints across steps.
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

# ============================================================================
# vLLM Step Server Helpers (optional)
# ============================================================================

_vllm_port_pids() {
    local port="$1"
    lsof -Pi ":${port}" -sTCP:LISTEN -t 2>/dev/null || true
}

_vllm_kill_port() {
    local port="$1"
    local pids
    pids="$(_vllm_port_pids "$port")"
    if [ -z "$pids" ]; then
        return 0
    fi
    log "Stopping vLLM on port ${port} (pids: ${pids})"
    kill -TERM $pids >/dev/null 2>&1 || true
    sleep 2
    pids="$(_vllm_port_pids "$port")"
    if [ -n "$pids" ]; then
        kill -KILL $pids >/dev/null 2>&1 || true
    fi
}

_vllm_wait_health() {
    local port="$1"
    local timeout_s="${2:-600}"
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        local now_ts
        now_ts="$(date +%s)"
        if [ $((now_ts - start_ts)) -ge "$timeout_s" ]; then
            return 1
        fi
        sleep 2
    done
}

start_step_vllm_servers() {
    local model_a_path="$1"
    local model_b_path="$2"

    if [ "$USE_STEP_VLLM_SERVERS" != "true" ]; then
        return 0
    fi

    log "Starting per-step vLLM servers..."
    log "  A: port=${VLLM_PORT_A}, gpu=${VLLM_GPU_A}, model=${model_a_path}"
    log "  B: port=${VLLM_PORT_B}, gpu=${VLLM_GPU_B}, model=${model_b_path}"

    # Ensure ports are free to avoid run_vllm_server.sh interactive prompt.
    _vllm_kill_port "$VLLM_PORT_A"
    _vllm_kill_port "$VLLM_PORT_B"

    # Start server A and B in parallel, then wait for both to become healthy.
    #
    # IMPORTANT: We intentionally bypass `scripts/run_vllm_server.sh` here.
    # In some environments, prefixing `bash scripts/run_vllm_server.sh` with inline env vars
    # can behave unexpectedly (falling back to defaults, causing both servers to bind the same port).
    #
    # We start vLLM directly so (port, GPU) are unambiguous.
    ( CUDA_VISIBLE_DEVICES="$VLLM_GPU_A" VLLM_SERVER_DEV_MODE=1 \
      python -m vllm.entrypoints.openai.api_server \
        --model "$model_a_path" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT_A" \
        --enable-sleep-mode \
        --gpu-memory-utilization "$VLLM_SERVER_GPU_MEMORY_UTILIZATION" \
        --dtype "$VLLM_SERVER_DTYPE" \
        --trust-remote-code \
        --max-model-len "$VLLM_SERVER_MAX_MODEL_LEN" \
        --max-num-seqs "$VLLM_SERVER_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$VLLM_SERVER_MAX_NUM_BATCHED_TOKENS" \
        $( [[ "$VLLM_SERVER_CHUNKED_PREFILL" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
    ) >"vllm_server_${VLLM_PORT_A}.log" 2>&1 &
    VLLM_STEP_PID_A=$!

    ( CUDA_VISIBLE_DEVICES="$VLLM_GPU_B" VLLM_SERVER_DEV_MODE=1 \
      python -m vllm.entrypoints.openai.api_server \
        --model "$model_b_path" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT_B" \
        --enable-sleep-mode \
        --gpu-memory-utilization "$VLLM_SERVER_GPU_MEMORY_UTILIZATION" \
        --dtype "$VLLM_SERVER_DTYPE" \
        --trust-remote-code \
        --max-model-len "$VLLM_SERVER_MAX_MODEL_LEN" \
        --max-num-seqs "$VLLM_SERVER_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$VLLM_SERVER_MAX_NUM_BATCHED_TOKENS" \
        $( [[ "$VLLM_SERVER_CHUNKED_PREFILL" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
    ) >"vllm_server_${VLLM_PORT_B}.log" 2>&1 &
    VLLM_STEP_PID_B=$!

    # Wait in parallel.
    _vllm_wait_health "$VLLM_PORT_A" "$VLLM_SERVER_START_TIMEOUT" &
    WAIT_PID_A=$!
    _vllm_wait_health "$VLLM_PORT_B" "$VLLM_SERVER_START_TIMEOUT" &
    WAIT_PID_B=$!

    wait "$WAIT_PID_A"
    OK_A=$?
    wait "$WAIT_PID_B"
    OK_B=$?

    if [ "$OK_A" -ne 0 ] || [ "$OK_B" -ne 0 ]; then
        _vllm_kill_port "$VLLM_PORT_A"
        _vllm_kill_port "$VLLM_PORT_B"
        if [ "$OK_A" -ne 0 ] && [ "$OK_B" -ne 0 ]; then
            error "Timed out waiting for vLLM servers A($VLLM_PORT_A) and B($VLLM_PORT_B)"
        elif [ "$OK_A" -ne 0 ]; then
            error "Timed out waiting for vLLM server A on port $VLLM_PORT_A"
        else
            error "Timed out waiting for vLLM server B on port $VLLM_PORT_B"
        fi
    fi

    log "Per-step vLLM servers are healthy."
}

stop_step_vllm_servers() {
    if [ "$USE_STEP_VLLM_SERVERS" != "true" ]; then
        return 0
    fi
    _vllm_kill_port "$VLLM_PORT_A"
    _vllm_kill_port "$VLLM_PORT_B"
    log "Per-step vLLM servers stopped."
}

# Get latest checkpoint path for a model
get_latest_checkpoint() {
    local model_name=$1  # "a" or "b"
    local checkpoint_dir="$LATEST_CHECKPOINT_DIR/model_$model_name"
    
    # Check if checkpoint directory exists and has checkpoints
    if [ -d "$checkpoint_dir" ]; then
        # Find the latest checkpoint
        local latest_checkpoint=$(find "$checkpoint_dir" -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -n "$latest_checkpoint" ]; then
            echo "$latest_checkpoint"
            return 0
        fi
    fi
    
    # Fallback to initial model path
    if [ "$model_name" = "a" ]; then
        echo "$INITIAL_MODEL_A_PATH"
    else
        echo "$INITIAL_MODEL_B_PATH"
    fi
}

# ============================================================================
# Main Script
# ============================================================================

main() {
    log "=================================================================================="
    log "Co-Intuitor Training Pipeline (Iterative)"
    log "=================================================================================="
    
    check_required_vars
    
    log "Configuration:"
    log "  INITIAL_MODEL_A_PATH (ref): $INITIAL_MODEL_A_PATH"
    log "  INITIAL_MODEL_B_PATH (ref): $INITIAL_MODEL_B_PATH"
    log "  LATEST_CHECKPOINT_DIR: $LATEST_CHECKPOINT_DIR"
    log "  DATASET_KEY: $DATASET_KEY"
    log "  MAX_STEPS: $MAX_STEPS"
    log ""
    
    # Initialize checkpoint directory
    init_checkpoint_dir
    
    # Iterative training loop
    for step in $(seq 0 $((MAX_STEPS - 1))); do
        log "=================================================================================="
        log "Step $step: Generate -> Train Model A -> Train Model B"
        log "=================================================================================="
        
        # Get current model paths (old models)
        CURRENT_MODEL_A_PATH=$(get_latest_checkpoint "a")
        CURRENT_MODEL_B_PATH=$(get_latest_checkpoint "b")
        
        log "Current models (old):"
        log "  Model A: $CURRENT_MODEL_A_PATH"
        log "  Model B: $CURRENT_MODEL_B_PATH"
        log ""
        
        # Step 1: Generate and Score
        log "--- Step $step.1: Generate and Score Responses ---"

        # Make sure generate/score/logps see the same GPU set as training.
        # (When using accelerate launch, each process will bind to its own GPU.)
        GEN_GPU_LIST=""
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            if [ -z "$GEN_GPU_LIST" ]; then
                GEN_GPU_LIST="$i"
            else
                GEN_GPU_LIST="$GEN_GPU_LIST,$i"
            fi
        done
        export CUDA_VISIBLE_DEVICES=$GEN_GPU_LIST
        
        GENERATE_ARGS=(
            --model_a_path "$CURRENT_MODEL_A_PATH"
            --model_b_path "$CURRENT_MODEL_B_PATH"
            --tokenizer_a_path "$INITIAL_MODEL_A_PATH"
            --tokenizer_b_path "$INITIAL_MODEL_B_PATH"
            --dataset_key "$DATASET_KEY"
            --output_dir "$LATEST_CHECKPOINT_DIR"
            --step "$step"
            --seed "$SEED"
            --num_generations_per_model "$NUM_GENERATIONS_PER_MODEL"
            --temperature "$TEMPERATURE"
            --top_p "$TOP_P"
            --repetition_penalty "$REPETITION_PENALTY"
            --max_completion_length "$MAX_COMPLETION_LENGTH"
            --vllm_port_a "$VLLM_PORT_A"
            --vllm_port_b "$VLLM_PORT_B"
            --scoring_batch_size "$SCORING_BATCH_SIZE"
            --logps_batch_size "$LOGPS_BATCH_SIZE"
            --problems_per_step "$PROBLEMS_PER_STEP"
            --beta "$BETA"
            --num_iterations "$NUM_ITERATIONS"
            --ref_model_a_path "$INITIAL_MODEL_A_PATH"
            --ref_model_b_path "$INITIAL_MODEL_B_PATH"
        )
        
        if [ -n "$MAX_SAMPLES" ]; then
            GENERATE_ARGS+=(--max_samples "$MAX_SAMPLES")
        fi
        
        if [ -n "$SYSTEM_PROMPT" ]; then
            GENERATE_ARGS+=(--system_prompt "$SYSTEM_PROMPT")
        fi
        
        if [ -n "$MAX_PROMPT_LENGTH" ]; then
            GENERATE_ARGS+=(--max_prompt_length "$MAX_PROMPT_LENGTH")
        fi
        
        if [ "$AUTO_START_SERVER" = "true" ]; then
            GENERATE_ARGS+=(--auto_start_server)
        else
            GENERATE_ARGS+=(--no-auto_start_server)
        fi
        
        if [ "$AUTO_STOP_SERVER" = "true" ]; then
            GENERATE_ARGS+=(--auto_stop_server)
        else
            GENERATE_ARGS+=(--no-auto_stop_server)
        fi

        # Sampling strategy (default: sequential; when enabled: open-r1-style shuffle-per-epoch + drop_last).
        if [ "$SHUFFLE_DATASET" = "true" ]; then
            GENERATE_ARGS+=(--shuffle_dataset)
        else
            GENERATE_ARGS+=(--no-shuffle_dataset)
        fi
        
        # Use data-parallel inference for scoring/logps across GPUs.
        # Generation is guarded inside the script to run only on the main process.
        start_step_vllm_servers "$CURRENT_MODEL_A_PATH" "$CURRENT_MODEL_B_PATH"

        set +e
        accelerate launch \
            --num_processes "$NUM_GPUS" \
            --main_process_port "${CO_MAIN_PROCESS_PORT:-25900}" \
            src/pipelines/generate_and_score.py "${GENERATE_ARGS[@]}"
        GEN_RC=$?
        set -e

        stop_step_vllm_servers

        if [ "$GEN_RC" -ne 0 ]; then
            error "Generation and scoring failed for step $step"
        fi
        
        log "Step $step.1: Generation and scoring completed!"
        log ""

        # Prepare CUDA_VISIBLE_DEVICES once for training.
        # IMPORTANT: If Model A training is skipped, we still need CUDA_VISIBLE_DEVICES set for Model B.
        TRAIN_GPU_LIST=""
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            if [ -z "$TRAIN_GPU_LIST" ]; then
                TRAIN_GPU_LIST="$i"
            else
                TRAIN_GPU_LIST="$TRAIN_GPU_LIST,$i"
            fi
        done
        export CUDA_VISIBLE_DEVICES=$TRAIN_GPU_LIST
        
        # Step 2: Train Model A
        log "--- Step $step.2: Train Model A ---"
        
        # Skip training if this step's checkpoint already exists
        MODEL_A_STEP_CKPT="$LATEST_CHECKPOINT_DIR/model_a/checkpoint-$step"
        if [ -d "$MODEL_A_STEP_CKPT" ]; then
            log "Checkpoint already exists for Model A at step $step: $MODEL_A_STEP_CKPT"
            log "Skipping Model A training for step $step."
            log ""
        else
        
        TRAIN_ARGS_A=(
            --train_model a
            --model_name_or_path "$CURRENT_MODEL_A_PATH"
            --model_a_path "$CURRENT_MODEL_A_PATH"
            --model_b_path "$CURRENT_MODEL_B_PATH"
            --tokenizer_a_path "$INITIAL_MODEL_A_PATH"
            --tokenizer_b_path "$INITIAL_MODEL_B_PATH"
            --dataset_key "$DATASET_KEY"
            --output_dir "$LATEST_CHECKPOINT_DIR/model_a"
            --cache_dir "$LATEST_CHECKPOINT_DIR"
            --max_steps 1
            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS_A"
            --learning_rate "$LEARNING_RATE"
            --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE_A"
            --num_generations_per_model "$NUM_GENERATIONS_PER_MODEL"
            --max_completion_length "$MAX_COMPLETION_LENGTH"
            --max_prompt_length "$MAX_PROMPT_LENGTH"
            --temperature "$TEMPERATURE"
            --top_p "$TOP_P"
            --beta "$BETA"
            --epsilon 0.2
            --seed "$SEED"
            --warmup_ratio "$WARMUP_RATIO"
            --lr_scheduler_type cosine
            --gradient_checkpointing
            --bf16
            --system_prompt "$SYSTEM_PROMPT"
            --logging_steps 1
            --logging_strategy steps
            --logging_first_step
            --save_steps 1
            --save_strategy steps
            --save_total_limit 1
            --overwrite_output_dir
            --torch_dtype bfloat16
            --trust_remote_code
            --reward_method "$REWARD_METHOD"
            --reward_weight_a "$REWARD_WEIGHT_A"
            --reward_weight_b "$REWARD_WEIGHT_B"
        )
        
        if [ -n "$MAX_SAMPLES" ]; then
            TRAIN_ARGS_A+=(--max_samples "$MAX_SAMPLES")
        fi
        
        if [ -n "$DEEPSPEED_CONFIG" ]; then
            TRAIN_ARGS_A+=(--deepspeed "$DEEPSPEED_CONFIG")
        fi
        
        # Avoid "port 29500 already in use" when multiple jobs run on the same node.
        # Set CO_MAIN_PROCESS_PORT to a fixed port if you need determinism; default 0 auto-picks a free port.
        if [ -f "$ACCELERATE_CONFIG" ]; then
            accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-25901}" src/training/co_intuitor_train.py "${TRAIN_ARGS_A[@]}"
        else
            accelerate launch --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-25901}" src/training/co_intuitor_train.py "${TRAIN_ARGS_A[@]}"
        fi
        
        if [ $? -ne 0 ]; then
            error "Training Model A failed for step $step"
        fi
        
        log "Step $step.2: Model A training completed!"
        log ""
        fi
        # Hard limit: keep only the newest checkpoint on disk.
        prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_a"

        # Step 2.5: Evaluate Model A on multiple datasets
        log "--- Step $step.2.5: Evaluate Model A ---"
        EVAL_MODEL_A_PATH=$(get_latest_checkpoint "a")
        if [ "$EVAL_BACKEND" = "lighteval" ]; then
            EVAL_OUT_DIR_A="outputs/lighteval/co-intuitor-7B/model-a/checkpoint-$step"
            EVAL_DONE_A="$EVAL_OUT_DIR_A/.eval_done"
            if [ -f "$EVAL_DONE_A" ]; then
                log "Eval already done for Model A (step=$step, backend=lighteval). Skipping. Marker: $EVAL_DONE_A"
            else
            # Match eval_one_model.sh: prefer offline-by-default (your network is unreliable).
            # Override by exporting HF_HUB_OFFLINE=0 / HF_DATASETS_OFFLINE=0 / TRANSFORMERS_OFFLINE=0 if desired.
            export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
            export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
            export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
            mkdir -p "$EVAL_OUT_DIR_A"
            python -m eval.lighteval.runner \
                --model_path "$EVAL_MODEL_A_PATH" \
                --tasks "$LIGHTEVAL_TASKS" \
                --output_dir "$EVAL_OUT_DIR_A" \
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
            # Mark eval complete to avoid re-running on restarts.
            date -Iseconds > "$EVAL_DONE_A"
            fi
            # CRUXEval via ZeroEval (mode=zeroeval_repo)
            ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
            ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
            ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
            ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
            python -m eval.zeroeval.runner \
                --model_path "$EVAL_MODEL_A_PATH" \
                --model_tag a \
                --checkpoint_tag "checkpoint-$step" \
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
        else
            EVAL_OUT_ROOT_A="outputs/eval/co-intuitor-7B"
            EVAL_DONE_A="$EVAL_OUT_ROOT_A/.eval_done_model-a_checkpoint-$step"
            if [ -f "$EVAL_DONE_A" ]; then
                log "Eval already done for Model A (step=$step, backend=evaluate_checkpoints). Skipping. Marker: $EVAL_DONE_A"
            else
            mkdir -p "$EVAL_OUT_ROOT_A"
            python -m src.evaluate_checkpoints \
                --model_path "$EVAL_MODEL_A_PATH" \
                --model_tag a \
                --checkpoint_tag "checkpoint-$step" \
                --datasets "math_500,gsm8k" \
                --num_generations "$EVAL_NUM_GENERATIONS" \
                --max_samples "$EVAL_MAX_SAMPLES" \
                --system_prompt "$SYSTEM_PROMPT" \
                --output_root "$EVAL_OUT_ROOT_A" \
                --vllm_port 8010 \
                --vllm_batch_size "$EVAL_VLLM_BATCH_SIZE" \
                --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
                --seed "$SEED"
            # Mark eval complete to avoid re-running on restarts.
            date -Iseconds > "$EVAL_DONE_A"
            fi

            ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
            ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
            ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
            ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
            python -m eval.zeroeval.runner \
                --model_path "$EVAL_MODEL_A_PATH" \
                --model_tag a \
                --checkpoint_tag "checkpoint-$step" \
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
        fi
        log "Step $step.2.5: Model A evaluation completed!"
        log ""
        
        # Step 3: Train Model B
        log "--- Step $step.3: Train Model B ---"
        
        # Update current model paths after Model A training
        CURRENT_MODEL_A_PATH=$(get_latest_checkpoint "a")

        # Skip training if this step's checkpoint already exists
        MODEL_B_STEP_CKPT="$LATEST_CHECKPOINT_DIR/model_b/checkpoint-$step"
        if [ -d "$MODEL_B_STEP_CKPT" ]; then
            log "Checkpoint already exists for Model B at step $step: $MODEL_B_STEP_CKPT"
            log "Skipping Model B training for step $step."
            log ""
        else
        
        TRAIN_ARGS_B=(
            --train_model b
            --model_name_or_path "$CURRENT_MODEL_B_PATH"
            --model_a_path "$CURRENT_MODEL_A_PATH"
            --model_b_path "$CURRENT_MODEL_B_PATH"
            --tokenizer_a_path "$INITIAL_MODEL_A_PATH"
            --tokenizer_b_path "$INITIAL_MODEL_B_PATH"
            --dataset_key "$DATASET_KEY"
            --output_dir "$LATEST_CHECKPOINT_DIR/model_b"
            --cache_dir "$LATEST_CHECKPOINT_DIR"
            --max_steps 1
            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS_B"
            --learning_rate "$LEARNING_RATE"
            --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE_B"
            --num_generations_per_model "$NUM_GENERATIONS_PER_MODEL"
            --max_completion_length "$MAX_COMPLETION_LENGTH"
            --max_prompt_length "$MAX_PROMPT_LENGTH"
            --temperature "$TEMPERATURE"
            --top_p "$TOP_P"
            --beta "$BETA"
            --epsilon 0.2
            --seed "$SEED"
            --warmup_ratio "$WARMUP_RATIO"
            --lr_scheduler_type cosine
            --gradient_checkpointing
            --bf16
            --system_prompt "$SYSTEM_PROMPT"
            --logging_steps 1
            --logging_strategy steps
            --logging_first_step
            --save_steps 1
            --save_strategy steps
            --save_total_limit 1
            --overwrite_output_dir
            --torch_dtype bfloat16
            --trust_remote_code
            --reward_method "$REWARD_METHOD"
            --reward_weight_a "$REWARD_WEIGHT_A"
            --reward_weight_b "$REWARD_WEIGHT_B"
        )
        
        if [ -n "$MAX_SAMPLES" ]; then
            TRAIN_ARGS_B+=(--max_samples "$MAX_SAMPLES")
        fi
        
        if [ -n "$DEEPSPEED_CONFIG" ]; then
            TRAIN_ARGS_B+=(--deepspeed "$DEEPSPEED_CONFIG")
        fi
        
        if [ -f "$ACCELERATE_CONFIG" ]; then
            accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-25930}" src/training/co_intuitor_train.py "${TRAIN_ARGS_B[@]}"
        else
            accelerate launch --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-25930}" src/training/co_intuitor_train.py "${TRAIN_ARGS_B[@]}"
        fi
        
        if [ $? -ne 0 ]; then
            error "Training Model B failed for step $step"
        fi
        
        log "Step $step.3: Model B training completed!"
        log ""
        fi
        # Hard limit: keep only the newest checkpoint on disk.
        prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_b"

        # Step 3.5: Evaluate Model B on multiple datasets
        log "--- Step $step.3.5: Evaluate Model B ---"
        EVAL_MODEL_B_PATH=$(get_latest_checkpoint "b")
        if [ "$EVAL_BACKEND" = "lighteval" ]; then
            EVAL_OUT_DIR_B="outputs/lighteval/co-intuitor-7B/model-b/checkpoint-$step"
            EVAL_DONE_B="$EVAL_OUT_DIR_B/.eval_done"
            if [ -f "$EVAL_DONE_B" ]; then
                log "Eval already done for Model B (step=$step, backend=lighteval). Skipping. Marker: $EVAL_DONE_B"
            else
            # Match eval_one_model.sh: prefer offline-by-default (your network is unreliable).
            export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
            export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
            export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
            mkdir -p "$EVAL_OUT_DIR_B"
            python -m eval.lighteval.runner \
                --model_path "$EVAL_MODEL_B_PATH" \
                --tasks "$LIGHTEVAL_TASKS" \
                --output_dir "$EVAL_OUT_DIR_B" \
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
            # Mark eval complete to avoid re-running on restarts.
            date -Iseconds > "$EVAL_DONE_B"
            fi
            ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
            ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
            ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
            ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
            python -m eval.zeroeval.runner \
                --model_path "$EVAL_MODEL_B_PATH" \
                --model_tag b \
                --checkpoint_tag "checkpoint-$step" \
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
        else
            EVAL_OUT_ROOT_B="outputs/eval/co-intuitor-7B"
            EVAL_DONE_B="$EVAL_OUT_ROOT_B/.eval_done_model-b_checkpoint-$step"
            if [ -f "$EVAL_DONE_B" ]; then
                log "Eval already done for Model B (step=$step, backend=evaluate_checkpoints). Skipping. Marker: $EVAL_DONE_B"
            else
            mkdir -p "$EVAL_OUT_ROOT_B"
            python -m src.evaluate_checkpoints \
                --model_path "$EVAL_MODEL_B_PATH" \
                --model_tag b \
                --checkpoint_tag "checkpoint-$step" \
                --datasets "math_500,gsm8k" \
                --num_generations "$EVAL_NUM_GENERATIONS" \
                --max_samples "$EVAL_MAX_SAMPLES" \
                --system_prompt "$SYSTEM_PROMPT" \
                --output_root "$EVAL_OUT_ROOT_B" \
                --vllm_port 8011 \
                --vllm_batch_size "$EVAL_VLLM_BATCH_SIZE" \
                --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
                --seed "$SEED"
            # Mark eval complete to avoid re-running on restarts.
            date -Iseconds > "$EVAL_DONE_B"
            fi

            ZEROEVAL_TP_SIZE="$ZEROEVAL_TP_SIZE" \
            ZEROEVAL_BATCH_SIZE="$ZEROEVAL_BATCH_SIZE" \
            ZEROEVAL_MAX_TOKENS="$ZEROEVAL_MAX_TOKENS" \
            ZEROEVAL_GPU_MEMORY_UTILIZATION="$ZEROEVAL_GPU_MEMORY_UTILIZATION" \
            python -m eval.zeroeval.runner \
                --model_path "$EVAL_MODEL_B_PATH" \
                --model_tag b \
                --checkpoint_tag "checkpoint-$step" \
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
        fi
        log "Step $step.3.5: Model B evaluation completed!"
        log ""
        
        log "Step $step completed successfully!"
        log ""
    done
    
    log "=================================================================================="
    log "Co-Intuitor Training Pipeline Completed!"
    log "=================================================================================="
    log ""
    log "Final models:"
    log "  Model A: $LATEST_CHECKPOINT_DIR/model_a"
    log "  Model B: $LATEST_CHECKPOINT_DIR/model_b"
    log "  Cache: $LATEST_CHECKPOINT_DIR"
}

# Run main function (guarded so this file can be sourced by other scripts)
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
