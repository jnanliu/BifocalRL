#!/usr/bin/env bash
set -euo pipefail

OPENRLHF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$OPENRLHF_ROOT"

RECIPE_NAME="${1:-bifocalrl}"
RECIPE_NAME="${RECIPE_NAME,,}"
if (( $# )); then
  shift
fi
RECIPE="${BIFOCAL_CONFIG:-examples/bifocalrl/configs/${RECIPE_NAME}.json}"
test -f "$RECIPE" || {
  echo "Recipe not found: $RECIPE" >&2
  exit 1
}

: "${MODEL_PATH:?Set MODEL_PATH to the actor Hugging Face model or directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to training JSONL, save_to_disk directory, or HF dataset ID}"

NNODE="${NNODE:-1}"
GPUS="${GPUS_PER_NODE:-8}"
VLLM_TP="${VLLM_TP_SIZE:-1}"

GROUP_SIZE="${GROUP_SIZE:-16}"
PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-output/bifocalrl-${RECIPE_NAME}}"

if (( GPUS % VLLM_TP )); then
  echo "NUM_GPUS must be divisible by VLLM_TP_SIZE" >&2
  exit 1
fi

export PYTHONPATH="$OPENRLHF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXTRA=()
if [[ "$RECIPE_NAME" == circa ]]; then
  KL_COEF="${KL_COEF:-0.005}"
  EXTRA+=(--algo.kl.use_loss)
fi
if [[ "${APPLY_CHAT_TEMPLATE:-1}" == 1 ]]; then
  EXTRA+=(--data.apply_chat_template)
fi
if [[ -n "${PRM_URL:-}" ]]; then
  EXTRA+=(--bifocal.prm_url "$PRM_URL")
fi
if [[ -n "${PRM_MODEL:-}" ]]; then
  EXTRA+=(--bifocal.prm_model "$PRM_MODEL")
fi
if [[ -n "${SANDBOX_FUSION_URL:-}" ]]; then
  EXTRA+=(--bifocal.sandbox_fusion_url "$SANDBOX_FUSION_URL")
fi
eval_config_given=false
for argument in "$@"; do
  if [[ "$argument" == --eval.config || "$argument" == --eval.config=* ]]; then
    eval_config_given=true
  fi
done
if [[ -n "${EVAL_CONFIG:-}" ]]; then
  EXTRA+=(--eval.config "$EVAL_CONFIG")
  eval_config_given=true
fi
if [[ "$eval_config_given" == false && -n "${EVAL_DATA:-}" ]]; then
  EVAL_SPLIT="${EVAL_SPLIT:-train}"
  EVAL_STEPS="${EVAL_STEPS:-10}"
  EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
  EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
fi
if [[ -n "${EVAL_DATA:-}" ]]; then
  EXTRA+=(--eval.dataset "$EVAL_DATA")
fi
if [[ -n "${EVAL_SPLIT:-}" ]]; then
  EXTRA+=(--eval.split "$EVAL_SPLIT")
fi
if [[ -n "${EVAL_STEPS:-}" ]]; then
  EXTRA+=(--eval.steps "$EVAL_STEPS")
fi
if [[ -n "${EVAL_TEMPERATURE:-}" ]]; then
  EXTRA+=(--eval.temperature "$EVAL_TEMPERATURE")
fi
if [[ -n "${EVAL_N_SAMPLES_PER_PROMPT:-}" ]]; then
  EXTRA+=(--eval.n_samples_per_prompt "$EVAL_N_SAMPLES_PER_PROMPT")
fi
if [[ -n "${EVAL_N_SAMPLES_PER_PROMPT_BY_DATASOURCE:-}" ]]; then
  EXTRA+=(--eval.n_samples_per_prompt_by_datasource "$EVAL_N_SAMPLES_PER_PROMPT_BY_DATASOURCE")
fi

exec "${PYTHON:-python}" -m openrlhf.cli.train_ppo_ray \
  --bifocal.config "$RECIPE" \
  --actor.model_name_or_path "$MODEL_PATH" \
  --actor.num_nodes "$NNODE" \
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
  --data.max_len "${MAX_LENGTH:-32768}" \
  --data.max_samples "${MAX_SAMPLES:-100000000}" \
  --data.apply_chat_template \
  --rollout.max_new_tokens "${MAX_RESPONSE_LEN:-3072}" \
  --rollout.temperature "${TEMPERATURE:-1.0}" \
  --rollout.top_p 1 \
  --rollout.batch_size "$PROMPTS_PER_ROLLOUT" \
  --rollout.n_samples_per_prompt "$GROUP_SIZE" \
  --train.batch_size "$((PROMPTS_PER_ROLLOUT * GROUP_SIZE))" \
  --train.micro_batch_size "$((PROMPTS_PER_ROLLOUT * GROUP_SIZE))" \
  --train.max_epochs 1 \
  --train.num_episodes "${NUM_EPISODES:-1}" \
  --train.dynamic_batch_enable \
  --train.max_tokens_per_gpu 32768 \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --ds.enable_sleep \
  --ds.packing_samples \
  --actor.gradient_checkpointing_enable \
  --actor.adam.lr "${LR:-1e-6}" \
  --algo.kl.init_coef "${KL_COEF:-0}" \
  --ckpt.output_dir "$OUTPUT_DIR/final" \
  --ckpt.path "$OUTPUT_DIR/checkpoints" \
  --ckpt.save_steps 20 \
  --ckpt.save_hf \
  --ckpt.disable_ds \
  --logger.tensorboard_dir "$OUTPUT_DIR/logs" \
  --logger.logging_steps 1 \
  "${EXTRA[@]}" "$@"
