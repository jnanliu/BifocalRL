#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

MODEL_PATH="${1:-$REPO_ROOT/models/DeepSeek-R1-Distill-Qwen-1.5B}"
OUTPUT_ROOT="${2:-$PROJECT_ROOT/outputs/eval_smoke/extended}"
DATASETS_ROOT="${LOCAL_DATASETS_ROOT:-$REPO_ROOT/datasets}"
CACHE_ROOT="${LIGHTEVAL_CACHE_ROOT:-$DATASETS_ROOT/.cache/eval}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_DATASETS_ROOT="$DATASETS_ROOT"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT/hf_home" "$CACHE_ROOT/datasets" "$CACHE_ROOT/transformers"
cd "$PROJECT_ROOT"

python - <<'PY'
import nltk

for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    print(f"nltk_ok: {resource} -> {nltk.data.find(resource)}")
PY

echo "[1/3] LiveCodeBench release_v6: 2 problems"
LIGHTEVAL_IDLE_TIMEOUT="${LIGHTEVAL_IDLE_TIMEOUT:-900}" \
env -u PYTORCH_CUDA_ALLOC_CONF python -m eval.lighteval.runner \
  --model_path "$MODEL_PATH" \
  --tasks "custom|lcb:codegeneration_release_v6_single|0|0" \
  --output_dir "$OUTPUT_ROOT/lcb" \
  --custom_tasks eval.lighteval.custom_tasks.offline_lcb_single \
  --local_datasets_root "$DATASETS_ROOT" \
  --datasets_offline \
  --hf_home "$CACHE_ROOT/hf_home" \
  --datasets_cache "$CACHE_ROOT/datasets" \
  --transformers_cache "$CACHE_ROOT/transformers" \
  --max_model_length 8192 \
  --max_new_tokens 4096 \
  --max_samples 2 \
  --gpu_memory_utilization 0.9 \
  --data_parallel_size 1 \
  --dtype bfloat16 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 7

echo "[2/3] CRUXEval-O: 2 problems"
env -u PYTORCH_CUDA_ALLOC_CONF python -m src.evaluate_checkpoints \
  --model_path "$MODEL_PATH" \
  --model_tag a \
  --checkpoint_tag smoke \
  --datasets cruxeval \
  --scorer default \
  --max_samples 2 \
  --output_root "$OUTPUT_ROOT/cruxeval" \
  --max_new_tokens 4096 \
  --num_generations 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 7 \
  --use_offline_vllm \
  --gpu_memory_utilization 0.9 \
  --tensor_parallel_size 1 \
  --vllm_dtype bfloat16 \
  --vllm_max_model_len 8192

echo "[3/3] TTRL math: 2 problems per dataset"
env -u PYTORCH_CUDA_ALLOC_CONF python -m eval.ttrl.runner \
  --model_paths "$MODEL_PATH" \
  --datasets math_500 gsm8k aime24 aime25 amc23 \
  --scorer ttrl_math \
  --num_generations 1 \
  --max_samples 2 \
  --prompt_mode suffix \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 4096 \
  --seed 7 \
  --gpu_memory_utilization 0.9 \
  --tensor_parallel_size 1 \
  --dtype bfloat16 \
  --max_model_len 8192 \
  --output_dir "$OUTPUT_ROOT/ttrl"

echo "Smoke outputs: $OUTPUT_ROOT"
