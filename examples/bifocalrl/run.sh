#!/usr/bin/env bash
set -euo pipefail

OPENRLHF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$OPENRLHF_ROOT"

RECIPE_NAME="${1:-mixed}"
if (( $# )); then
  shift
fi
RECIPE="${BIFOCAL_CONFIG:-examples/bifocalrl/configs/${RECIPE_NAME}.json}"
test -f "$RECIPE" || {
  echo "Recipe not found: $RECIPE" >&2
  exit 1
}

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the actor Hugging Face model or directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to training JSONL, save_to_disk directory, or HF dataset ID}"

GPUS="${NUM_GPUS:-8}"
VLLM_TP="${VLLM_TP_SIZE:-1}"

GROUP_SIZE="${GROUP_SIZE:-8}"
PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-output/bifocalrl-${RECIPE_NAME}}"

if (( GPUS % VLLM_TP )); then
  echo "NUM_GPUS must be divisible by VLLM_TP_SIZE" >&2
  exit 1
fi

export PYTHONPATH="$OPENRLHF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXTRA=()
if [[ "${APPLY_CHAT_TEMPLATE:-1}" == 1 ]]; then
  EXTRA+=(--data.apply_chat_template)
fi
if [[ -n "${PRM_URL:-}" ]]; then
  EXTRA+=(--bifocal.prm_url "$PRM_URL")
fi
if [[ -n "${PRM_MODEL:-}" ]]; then
  EXTRA+=(--bifocal.prm_model "$PRM_MODEL")
fi
if [[ -n "${EVAL_DATA:-}" ]]; then
  EXTRA+=(
    --eval.dataset "$EVAL_DATA"
    --eval.split "${EVAL_SPLIT:-train}"
    --eval.steps "${EVAL_STEPS:-10}"
    --eval.temperature 0
    --eval.n_samples_per_prompt 1
  )
fi

exec "${PYTHON:-python}" -m openrlhf.cli.train_ppo_ray \
  --bifocal.config "$RECIPE" \
  --actor.model_name_or_path "$HF_CHECKPOINT" \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$GPUS" \
  --train.colocate_all \
  --vllm.num_engines "$((GPUS / VLLM_TP))" \
  --vllm.tensor_parallel_size "$VLLM_TP" \
  --vllm.enable_sleep \
  --vllm.enable_prefix_caching \
  --vllm.gpu_memory_utilization "${VLLM_MEMORY_FRACTION:-0.5}" \
  --data.prompt_dataset "$PROMPT_DATA" \
  --data.prompt_split "${PROMPT_SPLIT:-train}" \
  --data.input_key prompt \
  --data.max_len "${MAX_LENGTH:-4096}" \
  --data.max_samples "${MAX_SAMPLES:-100000000}" \
  --rollout.max_new_tokens "${MAX_RESPONSE_LEN:-3072}" \
  --rollout.temperature "${TEMPERATURE:-1.0}" \
  --rollout.top_p 1 \
  --rollout.batch_size "$PROMPTS_PER_ROLLOUT" \
  --rollout.n_samples_per_prompt "$GROUP_SIZE" \
  --rollout.micro_batch_size "${MICRO_ROLLOUT_BATCH_SIZE:-1}" \
  --train.batch_size "$((PROMPTS_PER_ROLLOUT * GROUP_SIZE))" \
  --train.micro_batch_size "${MICRO_TRAIN_BATCH_SIZE:-1}" \
  --train.max_epochs 1 \
  --train.num_episodes "${NUM_EPISODES:-1}" \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --ds.enable_sleep \
  --actor.gradient_checkpointing_enable \
  --actor.adam.lr "${LR:-1e-6}" \
  --algo.kl.init_coef 0 \
  --ckpt.output_dir "$OUTPUT_DIR/final" \
  --ckpt.path "$OUTPUT_DIR/checkpoints" \
  --ckpt.save_steps 20 \
  "${EXTRA[@]}" "$@"
