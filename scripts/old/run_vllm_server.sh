#!/bin/bash

# vLLM Server Script
# Usage: ./run_vllm_server.sh
# 
# This script starts a vLLM server that can be used by run_co_intuitor.sh
# Run this in a separate terminal before starting training

# Set default values
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/test/models/Qwen2.5-3B}"
VLLM_PORT="${VLLM_PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
# Kill existing process on port if requested (non-interactive).
KILL_EXISTING_ON_PORT="${KILL_EXISTING_ON_PORT:-false}"
# GPU memory utilization (default: 0.9 for dedicated GPU, 0.5 for shared GPU)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
# Data type: "bfloat16" (for A100/H100, compute capability >= 8.0), "float16" or "half" (for V100, compute capability < 8.0)
# Default changed to float16 for better compatibility with V100
DTYPE="${DTYPE:-bfloat16}"

# -----------------------------------------------------------------------------
# Performance / stability knobs (override via env)
#
# In your log, vLLM picked max_seq_len=131072 from the model config. That can dramatically
# increase KV cache pressure and make the server "slow in the middle" when KV cache hits 100%.
# Capping max context and concurrency keeps throughput stable.
# -----------------------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"  # true|false

if [[ "$ENABLE_CHUNKED_PREFILL" == "true" ]]; then
    CHUNKED_PREFILL_FLAG="--enable-chunked-prefill"
else
    CHUNKED_PREFILL_FLAG="--disable-chunked-prefill"
fi

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Port: $VLLM_PORT"
echo "CUDA device: $CUDA_DEVICE (CUDA_VISIBLE_DEVICES will be set to only show this GPU)"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "Data type: $DTYPE"
echo "Max model len: $MAX_MODEL_LEN"
echo "Max num seqs: $MAX_NUM_SEQS"
echo "Max num batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "Chunked prefill: $ENABLE_CHUNKED_PREFILL"
echo "=========================================="
echo ""
echo "Note: Keep this terminal open while training."
echo "      The server will be used by run_co_intuitor.sh"
echo "      Press Ctrl+C to stop the server"
echo ""
echo "IMPORTANT: This server will ONLY use GPU $CUDA_DEVICE"
echo "           Other GPUs will not be visible to this process"
echo ""

# Check if port is already in use
if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    if [ "$KILL_EXISTING_ON_PORT" = "true" ]; then
        echo "Port $VLLM_PORT is in use; killing existing listener(s)..."
        PIDS=$(lsof -Pi ":${VLLM_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            kill -TERM $PIDS >/dev/null 2>&1 || true
            sleep 2
            PIDS=$(lsof -Pi ":${VLLM_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
            if [ -n "$PIDS" ]; then
                kill -KILL $PIDS >/dev/null 2>&1 || true
            fi
        fi
    else
    echo "WARNING: Port $VLLM_PORT is already in use!"
    echo "If vLLM server is already running, you can proceed with training."
    echo "If not, please stop the process using port $VLLM_PORT first."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    fi
fi

# Start vLLM server
echo "Starting vLLM server..."
export VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_DATASETS_VERBOSITY=error
export ACCELERATE_LOG_LEVEL=error
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune" || true

# Use native vLLM entrypoint instead of trl wrapper to allow --enable-sleep-mode
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --enable-sleep-mode \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    $CHUNKED_PREFILL_FLAG \
    2>&1 | tee vllm_server_${VLLM_PORT}.log
