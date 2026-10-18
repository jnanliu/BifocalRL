#!/bin/bash

# Co-WMV Self-Learning (SL) Training Script
# Each model trains ONLY on its own generated responses, but WMV pseudo-labels
# are computed from both models' responses combined.
# Default: each model generates 7 responses per prompt (14 total).
#
# Difference vs co-wmv:
#   - learn_config with other=0: each model trains only on its own responses
#   - NUM_GENERATIONS_PER_MODEL=7 by default (each model generates 7)
#
# Usage:
#   ./scripts/run_co_wmv_sl.sh
#
# Override most settings via environment variables, e.g.:
#   INITIAL_MODEL_A_PATH=/path/to/A INITIAL_MODEL_B_PATH=/path/to/B ./scripts/run_co_wmv_sl.sh

set -e  # Exit on error

# CRITICAL: Set environment variables EARLY to disable verbose logging before any imports
export DS_VERBOSE=0
export DS_LOG_LEVEL=WARNING
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
export HF_DATASETS_VERBOSITY=error
export ACCELERATE_LOG_LEVEL=error
export TORCH_DISTRIBUTED_DEBUG=OFF
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune" || true
export TORCH_CUDA_ARCH_LIST="8.0"

# Get script directory for resolving relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ============================================================================
# Configuration
# ============================================================================

# Initial model paths (ref models - never change)
INITIAL_MODEL_A_PATH="${INITIAL_MODEL_A_PATH:-/home/jizeyu26/jzy/7_co_learning/test/models/Qwen2.5-3B}"
INITIAL_MODEL_B_PATH="${INITIAL_MODEL_B_PATH:-/home/jizeyu26/jzy/7_co_learning/test/models/Llama-3.2-3B-Instruct}"

# Latest checkpoint directory (contains cache, model_a, model_b)
LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR:-outputs/co-wmv-sl/latest_checkpoint}"

# Dataset
DATASET_KEY="${DATASET_KEY:-math_lighteval}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SYSTEM_PROMPT="You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}. Please stop generation immediately after outputing the box."

# Generation parameters (SL default: 7 per model)
NUM_GENERATIONS_PER_MODEL="${NUM_GENERATIONS_PER_MODEL:-7}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"

# Scoring parameters
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-4}"
LOGPS_BATCH_SIZE="${LOGPS_BATCH_SIZE:-16}"

# vLLM server settings
VLLM_PORT_A="${VLLM_PORT_A:-8061}"
VLLM_PORT_B="${VLLM_PORT_B:-8062}"
AUTO_START_SERVER="${AUTO_START_SERVER:-false}"
AUTO_STOP_SERVER="${AUTO_STOP_SERVER:-true}"
USE_STEP_VLLM_SERVERS="${USE_STEP_VLLM_SERVERS:-true}"
VLLM_GPU_A="${VLLM_GPU_A:-0}"
VLLM_GPU_B="${VLLM_GPU_B:-1}"
VLLM_SERVER_START_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-900}"
VLLM_SERVER_GPU_MEMORY_UTILIZATION="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_SERVER_DTYPE="${VLLM_SERVER_DTYPE:-bfloat16}"
VLLM_SERVER_MAX_MODEL_LEN="${VLLM_SERVER_MAX_MODEL_LEN:-32768}"
VLLM_SERVER_MAX_NUM_SEQS="${VLLM_SERVER_MAX_NUM_SEQS:-128}"
VLLM_SERVER_MAX_NUM_BATCHED_TOKENS="${VLLM_SERVER_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_SERVER_CHUNKED_PREFILL="${VLLM_SERVER_CHUNKED_PREFILL:-true}"

# Training parameters
MAX_STEPS="${MAX_STEPS:-20}"
LEARNING_RATE="${LEARNING_RATE:-3.0e-6}"
BETA="${BETA:-0.0005}"
SEED="${SEED:-7}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
PROBLEMS_PER_STEP="${PROBLEMS_PER_STEP:-128}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
SHUFFLE_DATASET="${SHUFFLE_DATASET:-true}"
GRADIENT_ACCUMULATION_STEPS_A="${GRADIENT_ACCUMULATION_STEPS_A:-224}"
GRADIENT_ACCUMULATION_STEPS_B="${GRADIENT_ACCUMULATION_STEPS_B:-224}"
PER_DEVICE_BATCH_SIZE_A="${PER_DEVICE_BATCH_SIZE_A:-2}"
PER_DEVICE_BATCH_SIZE_B="${PER_DEVICE_BATCH_SIZE_B:-2}"

# Reward method: IMPORTANT for co-WMV
# co_wmv: weighted majority voting (self-certainty scores as weights)
# co_mv:  unweighted majority voting (each answer = 1 vote)
# sum_scores: continuous reward = weight_a*score_a + weight_b*score_b
REWARD_METHOD="${REWARD_METHOD:-co_mv}" # co_wmv | co_mv | sum_scores
REWARD_WEIGHT_A="${REWARD_WEIGHT_A:-1.0}"
REWARD_WEIGHT_B="${REWARD_WEIGHT_B:-1.0}"

# ============================================================================
# Debug flags
# ============================================================================
export CO_DEBUG_NAN="${CO_DEBUG_NAN:-1}"
export CO_DEBUG_NAN_DIR="${CO_DEBUG_NAN_DIR:-$LATEST_CHECKPOINT_DIR/debug_nan}"
export CO_DEBUG_NAN_RAISE="${CO_DEBUG_NAN_RAISE:-1}"

export CO_DEBUG_DATA="${CO_DEBUG_DATA:-1}"
export CO_DEBUG_DATA_DIR="${CO_DEBUG_DATA_DIR:-$LATEST_CHECKPOINT_DIR/debug_data}"
export CO_DEBUG_DATA_N="${CO_DEBUG_DATA_N:-3}"

export CO_DEBUG_TRAIN_BOUNDARY="${CO_DEBUG_TRAIN_BOUNDARY:-1}"
export CO_ASSERT_INVARIANTS="${CO_ASSERT_INVARIANTS:-1}"

# Evaluation parameters
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-400}"
EVAL_NUM_GENERATIONS="${EVAL_NUM_GENERATIONS:-1}"
EVAL_VLLM_BATCH_SIZE="${EVAL_VLLM_BATCH_SIZE:-400}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
EVAL_BACKEND="${EVAL_BACKEND:-lighteval}"
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-math_500,gsm8k}"
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-eval.lighteval.custom_tasks.offline_patch}"
LIGHTEVAL_LOCAL_DATASETS_ROOT="${LIGHTEVAL_LOCAL_DATASETS_ROOT:-${LOCAL_DATASETS_ROOT:-/home/jizeyu26/jzy/7_co_learning/test/datasets}}"
LIGHTEVAL_HF_HOME="${LIGHTEVAL_HF_HOME:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/hf_home}"
LIGHTEVAL_DATASETS_CACHE="${LIGHTEVAL_DATASETS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT}"
LIGHTEVAL_TRANSFORMERS_CACHE="${LIGHTEVAL_TRANSFORMERS_CACHE:-$LIGHTEVAL_LOCAL_DATASETS_ROOT/transformers_cache}"
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-1}"


ZEROEVAL_DIR="${ZEROEVAL_DIR:-third_party/ZeroEval}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-outputs/zeroeval/co-wmv-sl}"
ZEROEVAL_TP_SIZE="${ZEROEVAL_TP_SIZE:-2}"
ZEROEVAL_BATCH_SIZE="${ZEROEVAL_BATCH_SIZE:-100}"
ZEROEVAL_MAX_TOKENS="${ZEROEVAL_MAX_TOKENS:-2048}"
ZEROEVAL_GPU_MEMORY_UTILIZATION="${ZEROEVAL_GPU_MEMORY_UTILIZATION:-0.95}"
ZEROEVAL_MAX_SAMPLES="${ZEROEVAL_MAX_SAMPLES:-800}"

# GPU configuration
NUM_GPUS="${NUM_GPUS:-2}"

# Accelerate config
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-zero3}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/${ACCELERATOR_CONFIG}.yaml}"

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
    if [ -z "$REWARD_METHOD" ]; then
        error "REWARD_METHOD must be set (co_wmv|co_mv|sum_scores)"
    fi
}

init_checkpoint_dir() {
    log "Initializing latest_checkpoint directory: $LATEST_CHECKPOINT_DIR"
    mkdir -p "$LATEST_CHECKPOINT_DIR"
    mkdir -p "$LATEST_CHECKPOINT_DIR/model_a"
    mkdir -p "$LATEST_CHECKPOINT_DIR/model_b"
    log "Ref models will use INITIAL_MODEL paths directly (no copying)"
    prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_a"
    prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_b"
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
        if [ ! -d "$ckpt" ]; then continue; fi
        if [ "$ckpt" != "$latest" ]; then
            log "Pruning old checkpoint: $ckpt"
            rm -rf "$ckpt" || true
        fi
    done
}

_vllm_port_pids() {
    local port="$1"
    lsof -Pi ":${port}" -sTCP:LISTEN -t 2>/dev/null || true
}

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
    local port="$1"
    local timeout_s="${2:-600}"
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then return 0; fi
        local now_ts
        now_ts="$(date +%s)"
        if [ $((now_ts - start_ts)) -ge "$timeout_s" ]; then return 1; fi
        sleep 2
    done
}

start_step_vllm_servers() {
    local model_a_path="$1"
    local model_b_path="$2"
    if [ "$USE_STEP_VLLM_SERVERS" != "true" ]; then return 0; fi
    log "Starting per-step vLLM servers..."
    log "  A: port=${VLLM_PORT_A}, gpu=${VLLM_GPU_A}, model=${model_a_path}"
    log "  B: port=${VLLM_PORT_B}, gpu=${VLLM_GPU_B}, model=${model_b_path}"
    _vllm_kill_port "$VLLM_PORT_A"
    _vllm_kill_port "$VLLM_PORT_B"
    ( CUDA_VISIBLE_DEVICES="$VLLM_GPU_A" VLLM_SERVER_DEV_MODE=1 \
      python -m vllm.entrypoints.openai.api_server \
        --model "$model_a_path" --host 0.0.0.0 --port "$VLLM_PORT_A" \
        --enable-sleep-mode --gpu-memory-utilization "$VLLM_SERVER_GPU_MEMORY_UTILIZATION" \
        --dtype "$VLLM_SERVER_DTYPE" --trust-remote-code \
        --max-model-len "$VLLM_SERVER_MAX_MODEL_LEN" --max-num-seqs "$VLLM_SERVER_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$VLLM_SERVER_MAX_NUM_BATCHED_TOKENS" \
        $( [[ "$VLLM_SERVER_CHUNKED_PREFILL" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
    ) >"vllm_server_${VLLM_PORT_A}.log" 2>&1 &
    ( CUDA_VISIBLE_DEVICES="$VLLM_GPU_B" VLLM_SERVER_DEV_MODE=1 \
      python -m vllm.entrypoints.openai.api_server \
        --model "$model_b_path" --host 0.0.0.0 --port "$VLLM_PORT_B" \
        --enable-sleep-mode --gpu-memory-utilization "$VLLM_SERVER_GPU_MEMORY_UTILIZATION" \
        --dtype "$VLLM_SERVER_DTYPE" --trust-remote-code \
        --max-model-len "$VLLM_SERVER_MAX_MODEL_LEN" --max-num-seqs "$VLLM_SERVER_MAX_NUM_SEQS" \
        --max-num-batched-tokens "$VLLM_SERVER_MAX_NUM_BATCHED_TOKENS" \
        $( [[ "$VLLM_SERVER_CHUNKED_PREFILL" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
    ) >"vllm_server_${VLLM_PORT_B}.log" 2>&1 &
    _vllm_wait_health "$VLLM_PORT_A" "$VLLM_SERVER_START_TIMEOUT" &
    WAIT_PID_A=$!
    _vllm_wait_health "$VLLM_PORT_B" "$VLLM_SERVER_START_TIMEOUT" &
    WAIT_PID_B=$!
    wait "$WAIT_PID_A"; OK_A=$?
    wait "$WAIT_PID_B"; OK_B=$?
    if [ "$OK_A" -ne 0 ] || [ "$OK_B" -ne 0 ]; then
        _vllm_kill_port "$VLLM_PORT_A"; _vllm_kill_port "$VLLM_PORT_B"
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
    if [ "$USE_STEP_VLLM_SERVERS" != "true" ]; then return 0; fi
    _vllm_kill_port "$VLLM_PORT_A"
    _vllm_kill_port "$VLLM_PORT_B"
    log "Per-step vLLM servers stopped."
}

get_latest_checkpoint() {
    local model_name=$1
    local checkpoint_dir="$LATEST_CHECKPOINT_DIR/model_$model_name"
    if [ -d "$checkpoint_dir" ]; then
        local latest_checkpoint
        latest_checkpoint=$(find "$checkpoint_dir" -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -n "$latest_checkpoint" ]; then
            echo "$latest_checkpoint"
            return 0
        fi
    fi
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
    log "Co-WMV Self-Learning (SL) Training Pipeline (Iterative)"
    log "=================================================================================="

    check_required_vars

    log "Configuration:"
    log "  INITIAL_MODEL_A_PATH (ref): $INITIAL_MODEL_A_PATH"
    log "  INITIAL_MODEL_B_PATH (ref): $INITIAL_MODEL_B_PATH"
    log "  LATEST_CHECKPOINT_DIR: $LATEST_CHECKPOINT_DIR"
    log "  DATASET_KEY: $DATASET_KEY"
    log "  REWARD_METHOD: $REWARD_METHOD"
    log "  NUM_GENERATIONS_PER_MODEL: $NUM_GENERATIONS_PER_MODEL"
    log "  LEARN_CONFIG (SL): each model learns own only (other=0)"
    log "  MAX_STEPS: $MAX_STEPS"
    log ""

    init_checkpoint_dir

    for step in $(seq 19 $((MAX_STEPS - 1))); do
        log "=================================================================================="
        log "Step $step: Generate -> Train Model A (SL) -> Train Model B (SL)"
        log "=================================================================================="

        CURRENT_MODEL_A_PATH=$(get_latest_checkpoint "a")
        CURRENT_MODEL_B_PATH=$(get_latest_checkpoint "b")

        log "Current models (old):"
        log "  Model A: $CURRENT_MODEL_A_PATH"
        log "  Model B: $CURRENT_MODEL_B_PATH"
        log ""

        # Step 1: Generate and Score
        log "--- Step $step.1: Generate and Score Responses ---"

        GEN_GPU_LIST=""
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            if [ -z "$GEN_GPU_LIST" ]; then GEN_GPU_LIST="$i"; else GEN_GPU_LIST="$GEN_GPU_LIST,$i"; fi
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

        if [ -n "$MAX_SAMPLES" ]; then GENERATE_ARGS+=(--max_samples "$MAX_SAMPLES"); fi
        if [ -n "$SYSTEM_PROMPT" ]; then GENERATE_ARGS+=(--system_prompt "$SYSTEM_PROMPT"); fi
        if [ -n "$MAX_PROMPT_LENGTH" ]; then GENERATE_ARGS+=(--max_prompt_length "$MAX_PROMPT_LENGTH"); fi
        if [ "$AUTO_START_SERVER" = "true" ]; then GENERATE_ARGS+=(--auto_start_server); else GENERATE_ARGS+=(--no-auto_start_server); fi
        if [ "$AUTO_STOP_SERVER" = "true" ]; then GENERATE_ARGS+=(--auto_stop_server); else GENERATE_ARGS+=(--no-auto_stop_server); fi
        if [ "$SHUFFLE_DATASET" = "true" ]; then GENERATE_ARGS+=(--shuffle_dataset); else GENERATE_ARGS+=(--no-shuffle_dataset); fi

        start_step_vllm_servers "$CURRENT_MODEL_A_PATH" "$CURRENT_MODEL_B_PATH"

        set +e
        accelerate launch \
            --num_processes "$NUM_GPUS" \
            --main_process_port "${CO_MAIN_PROCESS_PORT:-26000}" \
            src/pipelines/generate_and_score.py "${GENERATE_ARGS[@]}"
        GEN_RC=$?
        set -e

        stop_step_vllm_servers

        if [ "$GEN_RC" -ne 0 ]; then
            error "Generation and scoring failed for step $step"
        fi

        log "Step $step.1: Generation and scoring completed!"
        log ""

        TRAIN_GPU_LIST=""
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            if [ -z "$TRAIN_GPU_LIST" ]; then TRAIN_GPU_LIST="$i"; else TRAIN_GPU_LIST="$TRAIN_GPU_LIST,$i"; fi
        done
        export CUDA_VISIBLE_DEVICES=$TRAIN_GPU_LIST

        # Step 2: Train Model A (self-learning)
        log "--- Step $step.2: Train Model A (SL) ---"

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
                --max_grad_norm 0.0
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
                --learn_config "{\"a\":{\"own\":${NUM_GENERATIONS_PER_MODEL},\"other\":0},\"b\":{\"own\":${NUM_GENERATIONS_PER_MODEL},\"other\":0},\"selection\":\"first\"}"
            )

            if [ -n "$MAX_SAMPLES" ]; then TRAIN_ARGS_A+=(--max_samples "$MAX_SAMPLES"); fi

            if [ -f "$ACCELERATE_CONFIG" ]; then
                accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-26001}" src/training/co_intuitor_train.py "${TRAIN_ARGS_A[@]}"
            else
                accelerate launch --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-26001}" src/training/co_intuitor_train.py "${TRAIN_ARGS_A[@]}"
            fi

            if [ $? -ne 0 ]; then error "Training Model A failed for step $step"; fi

            log "Step $step.2: Model A training completed!"
            log ""
        fi
        prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_a"

        # Step 2.5: Evaluate Model A
        log "--- Step $step.2.5: Evaluate Model A ---"
        EVAL_MODEL_A_PATH=$(get_latest_checkpoint "a")
        if [ "$EVAL_BACKEND" = "lighteval" ]; then
            EVAL_OUT_DIR_A="outputs/lighteval/co-wmv-sl/model-a/checkpoint-$step"
            EVAL_DONE_A="$EVAL_OUT_DIR_A/.eval_done"
            if [ -f "$EVAL_DONE_A" ]; then
                log "Eval already done for Model A (step=$step). Skipping."
            else
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

        # Step 3: Train Model B (self-learning)
        log "--- Step $step.3: Train Model B (SL) ---"

        CURRENT_MODEL_A_PATH=$(get_latest_checkpoint "a")

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
                --max_grad_norm 0.0
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
                --learn_config "{\"a\":{\"own\":${NUM_GENERATIONS_PER_MODEL},\"other\":0},\"b\":{\"own\":${NUM_GENERATIONS_PER_MODEL},\"other\":0},\"selection\":\"first\"}"
            )

            if [ -n "$MAX_SAMPLES" ]; then TRAIN_ARGS_B+=(--max_samples "$MAX_SAMPLES"); fi

            if [ -f "$ACCELERATE_CONFIG" ]; then
                accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-26030}" src/training/co_intuitor_train.py "${TRAIN_ARGS_B[@]}"
            else
                accelerate launch --num_processes "$NUM_GPUS" --main_process_port "${CO_MAIN_PROCESS_PORT:-26030}" src/training/co_intuitor_train.py "${TRAIN_ARGS_B[@]}"
            fi

            if [ $? -ne 0 ]; then error "Training Model B failed for step $step"; fi

            log "Step $step.3: Model B training completed!"
            log ""
        fi
        prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_b"

        # Step 3.5: Evaluate Model B
        log "--- Step $step.3.5: Evaluate Model B ---"
        EVAL_MODEL_B_PATH=$(get_latest_checkpoint "b")
        if [ "$EVAL_BACKEND" = "lighteval" ]; then
            EVAL_OUT_DIR_B="outputs/lighteval/co-wmv-sl/model-b/checkpoint-$step"
            EVAL_DONE_B="$EVAL_OUT_DIR_B/.eval_done"
            if [ -f "$EVAL_DONE_B" ]; then
                log "Eval already done for Model B (step=$step). Skipping."
            else
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

        log "Step $step completed successfully!"
        log ""
    done

    log "=================================================================================="
    log "Co-WMV SL Training Pipeline Completed!"
    log "=================================================================================="
    log ""
    log "Final models:"
    log "  Model A: $LATEST_CHECKPOINT_DIR/model_a"
    log "  Model B: $LATEST_CHECKPOINT_DIR/model_b"
    log "  Cache: $LATEST_CHECKPOINT_DIR"
}

main "$@"
