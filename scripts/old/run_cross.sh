#!/usr/bin/env bash
set -euo pipefail

# Config-driven cross-model pipeline runner.
#
# A fixed GENERATOR model produces completions via vLLM.
# A TRAINEE model scores (SCE), computes logps, and trains iteratively.
# Generation uses the generator's chat template; scoring/training uses the trainee's.
#
# Usage:
#   ./scripts/run_cross.sh recipes/configs/cross_intuitor.json
#   GENERATOR_PATH=/a TRAINEE_INIT_PATH=/b ./scripts/run_cross.sh recipes/configs/cross_intuitor.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"
cd "$SCRIPT_DIR/.." || exit 1

CONFIG_FILE="${1:?Usage: $0 <config.json>}"
_load_config "$CONFIG_FILE"

# ============================================================================
# Read config (env vars override JSON)
# ============================================================================

# Models
GENERATOR_PATH="${GENERATOR_PATH:-$(cfg models.generator)}"
TRAINEE_INIT_PATH="${TRAINEE_INIT_PATH:-$(cfg models.trainee)}"
GENERATOR_TOKENIZER_PATH="${GENERATOR_TOKENIZER_PATH:-${GENERATOR_PATH}}"
TRAINEE_TOKENIZER_PATH="${TRAINEE_TOKENIZER_PATH:-${TRAINEE_INIT_PATH}}"

# Output
LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR:-$(cfg output.checkpoint_dir "outputs/cross/latest_checkpoint")}"
MODEL_DIR="$LATEST_CHECKPOINT_DIR/model"
LIGHTEVAL_OUT_PREFIX="${LIGHTEVAL_OUT_PREFIX:-$(cfg output.lighteval_prefix "outputs/lighteval/cross")}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-$(cfg output.zeroeval_root "outputs/zeroeval/cross")}"

# Dataset
DATASET_KEY="${DATASET_KEY:-$(cfg dataset.key "math_lighteval")}"
MAX_SAMPLES="${MAX_SAMPLES:-$(cfg dataset.max_samples)}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$(cfg dataset.system_prompt)}"
PROBLEMS_PER_STEP="${PROBLEMS_PER_STEP:-$(cfg dataset.problems_per_step "128")}"
SHUFFLE_DATASET="${SHUFFLE_DATASET:-$(cfg dataset.shuffle "true")}"

# Generation
NUM_GENERATIONS="${NUM_GENERATIONS:-$(cfg generation.num_generations "7")}"
TEMPERATURE="${TEMPERATURE:-$(cfg generation.temperature "0.9")}"
TOP_P="${TOP_P:-$(cfg generation.top_p "1.0")}"
REPETITION_PENALTY="${REPETITION_PENALTY:-$(cfg generation.repetition_penalty "1.0")}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-$(cfg generation.max_completion_length "2048")}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$(cfg generation.max_prompt_length "512")}"

# Scoring
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-$(cfg scoring.scoring_batch_size "4")}"
LOGPS_BATCH_SIZE="${LOGPS_BATCH_SIZE:-$(cfg scoring.logps_batch_size "16")}"
MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-$(cfg scoring.mask_truncated_completions "false")}"

# Reward
REWARD_MODE="${REWARD_MODE:-$(cfg reward.mode "intuitor")}"
BETA="${BETA:-$(cfg reward.beta "0.0005")}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${TRAINEE_INIT_PATH}}"

# Training
NUM_GPUS="${NUM_GPUS:-$(cfg training.num_gpus "2")}"
MAX_STEPS="${MAX_STEPS:-$(cfg training.max_steps "20")}"
START_STEP="${START_STEP:-$(cfg training.start_step "0")}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$(cfg training.per_device_batch_size "2")}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(cfg training.gradient_accumulation_steps "224")}"
LEARNING_RATE="${LEARNING_RATE:-$(cfg training.learning_rate "3e-6")}"
WARMUP_RATIO="${WARMUP_RATIO:-$(cfg training.warmup_ratio "0.1")}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-$(cfg training.lr_scheduler "cosine")}"
WEIGHT_DECAY="${WEIGHT_DECAY:-$(cfg training.weight_decay "0.0")}"
TORCH_DTYPE="${TORCH_DTYPE:-$(cfg training.torch_dtype "bfloat16")}"
SEED="${SEED:-$(cfg training.seed "7")}"
BF16="${BF16:-$(cfg training.bf16 "true")}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-$(cfg training.gradient_checkpointing "true")}"
NUM_ITERATIONS="${NUM_ITERATIONS:-$(cfg training.num_iterations "1")}"
EPSILON="${EPSILON:-$(cfg training.epsilon "0.2")}"
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-$(cfg training.accelerator_config "zero3")}"
ACCELERATE_CONFIG="recipes/accelerate_configs/${ACCELERATOR_CONFIG}.yaml"

# vLLM (generator only; trainee is in-process)
VLLM_PORT="${VLLM_PORT:-$(cfg vllm.port "7900")}"
VLLM_GPU="${VLLM_GPU:-$(cfg vllm.gpu "0")}"
VLLM_GPU_MEM="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-$(cfg vllm.gpu_memory_utilization "0.9")}"
VLLM_DTYPE="${VLLM_SERVER_DTYPE:-$(cfg vllm.dtype "bfloat16")}"
VLLM_MAX_MODEL_LEN="${VLLM_SERVER_MAX_MODEL_LEN:-$(cfg vllm.max_model_len "4096")}"
VLLM_MAX_NUM_SEQS="${VLLM_SERVER_MAX_NUM_SEQS:-$(cfg vllm.max_num_seqs "128")}"
VLLM_MAX_BATCHED="${VLLM_SERVER_MAX_NUM_BATCHED_TOKENS:-$(cfg vllm.max_num_batched_tokens "8192")}"
VLLM_CHUNKED="${VLLM_SERVER_CHUNKED_PREFILL:-$(cfg vllm.chunked_prefill "true")}"
VLLM_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-$(cfg vllm.start_timeout "900")}"

# Eval – shared
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-$(cfg eval.max_new_tokens "2048")}"
EVAL_LOCAL_DATASETS_ROOT="${EVAL_LOCAL_DATASETS_ROOT:-$(cfg eval.local_datasets_root "/home/jizeyu26/jzy/7_co_learning/test/datasets")}"

# Eval – lighteval
LIGHTEVAL_ENABLED="${LIGHTEVAL_ENABLED:-$(cfg eval.lighteval.enabled "false")}"
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-$(cfg eval.lighteval.tasks "math_500,gsm8k")}"
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-$(cfg eval.lighteval.custom_tasks "eval.lighteval.custom_tasks.offline_patch")}"
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-$(cfg eval.lighteval.data_parallel_size "1")}"

# Eval – zeroeval
ZEROEVAL_ENABLED="${ZEROEVAL_ENABLED:-$(cfg eval.zeroeval.enabled "false")}"
ZEROEVAL_TASKS="${ZEROEVAL_TASKS:-$(cfg eval.zeroeval.tasks "cruxeval")}"
ZEROEVAL_DIR="${ZEROEVAL_DIR:-$(cfg eval.zeroeval.repo_dir "third_party/ZeroEval")}"
ZEROEVAL_TP_SIZE="${ZEROEVAL_TP_SIZE:-$(cfg eval.zeroeval.tp_size "2")}"
ZEROEVAL_BATCH_SIZE="${ZEROEVAL_BATCH_SIZE:-$(cfg eval.zeroeval.batch_size "100")}"
ZEROEVAL_MAX_TOKENS="${ZEROEVAL_MAX_TOKENS:-$(cfg eval.zeroeval.max_tokens "2048")}"
ZEROEVAL_GPU_MEM="${ZEROEVAL_GPU_MEMORY_UTILIZATION:-$(cfg eval.zeroeval.gpu_memory_utilization "0.95")}"
ZEROEVAL_MAX_SAMPLES="${ZEROEVAL_MAX_SAMPLES:-$(cfg eval.zeroeval.max_samples "800")}"

# Eval – hacrl
HACRL_EVAL_ENABLED="${HACRL_EVAL_ENABLED:-$(cfg eval.hacrl.enabled "false")}"
HACRL_EVAL_TASKS="${HACRL_EVAL_TASKS:-$(cfg eval.hacrl.tasks "math_500,gsm8k")}"
HACRL_EVAL_SCORER="${HACRL_EVAL_SCORER:-$(cfg eval.hacrl.scorer "hacrl_prime_math")}"
HACRL_EVAL_NUM_GENERATIONS="${HACRL_EVAL_NUM_GENERATIONS:-$(cfg eval.hacrl.num_generations "1")}"
HACRL_EVAL_MAX_SAMPLES="${HACRL_EVAL_MAX_SAMPLES:-$(cfg eval.hacrl.max_samples "")}"
HACRL_EVAL_VLLM_BATCH_SIZE="${HACRL_EVAL_VLLM_BATCH_SIZE:-$(cfg eval.hacrl.vllm_batch_size "16")}"
HACRL_EVAL_TEMPERATURE="${HACRL_EVAL_TEMPERATURE:-$(cfg eval.hacrl.temperature "0.0")}"
HACRL_EVAL_TOP_P="${HACRL_EVAL_TOP_P:-$(cfg eval.hacrl.top_p "1.0")}"

WANDB_API_KEY="${WANDB_API_KEY:-}"
[ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY

# Backward compat for old flat eval config
_eval_compat

# ============================================================================
# Validate & Setup
# ============================================================================
[ -z "$GENERATOR_PATH" ] && error "models.generator is required in config"
[ -z "$TRAINEE_INIT_PATH" ] && error "models.trainee is required in config"
setup_debug "$LATEST_CHECKPOINT_DIR"
mkdir -p "$LATEST_CHECKPOINT_DIR" "$MODEL_DIR"
prune_old_checkpoints "$MODEL_DIR"

EFF_BATCH=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
log "================================================================"
log "Cross-model pipeline (config-driven)"
log "  config:       $CONFIG_FILE"
log "  name:         $(cfg name '<unnamed>')"
log "  generator:    $GENERATOR_PATH (fixed)"
log "  trainee:      $TRAINEE_INIT_PATH"
log "  reward_mode:  $REWARD_MODE"
log "  num_gen:      $NUM_GENERATIONS"
log "  steps:        $START_STEP .. $((MAX_STEPS - 1))"
log "  eff_batch:    ${NUM_GPUS} × ${PER_DEVICE_TRAIN_BATCH_SIZE} × ${GRADIENT_ACCUMULATION_STEPS} = ${EFF_BATCH}"
log "================================================================"

# ============================================================================
# Helpers
# ============================================================================
get_latest_trainee_checkpoint() {
  if [ -d "$MODEL_DIR" ]; then
    local last
    last=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$last" ]; then echo "$last"; return; fi
  fi
  echo "$TRAINEE_INIT_PATH"
}

# ============================================================================
# Start generator vLLM ONCE (fixed model, sleep/wake between steps)
# ============================================================================
_start_vllm_server "$GENERATOR_PATH" "$VLLM_PORT" "$VLLM_GPU" \
  "$VLLM_GPU_MEM" "$VLLM_DTYPE" "$VLLM_MAX_MODEL_LEN" \
  "$VLLM_MAX_NUM_SEQS" "$VLLM_MAX_BATCHED" "$VLLM_CHUNKED" "$VLLM_TIMEOUT"

trap '_vllm_kill_port "$VLLM_PORT"' EXIT

# ============================================================================
# Main Loop
# ============================================================================
for step in $(seq "$START_STEP" $((MAX_STEPS - 1))); do
  log "======================== Step $step ========================"
  TRAINEE_PATH=$(get_latest_trainee_checkpoint)
  log "Generator: $GENERATOR_PATH (fixed)"
  log "Trainee:   $TRAINEE_PATH"

  export CUDA_VISIBLE_DEVICES="$(_gpu_list "$NUM_GPUS")"

  # ---- Generate (generator) + Score/Logps (trainee) ----
  GEN_ARGS=(
    --generator_model_path "$GENERATOR_PATH"
    --trainee_model_path "$TRAINEE_PATH"
    --generator_tokenizer_path "$GENERATOR_TOKENIZER_PATH"
    --trainee_tokenizer_path "$TRAINEE_TOKENIZER_PATH"
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
    --no-auto_start_server --no-auto_stop_server
  )
  [ -n "$MAX_SAMPLES" ] && GEN_ARGS+=(--max_samples "$MAX_SAMPLES")
  [ -n "$SYSTEM_PROMPT" ] && GEN_ARGS+=(--system_prompt "$SYSTEM_PROMPT")
  [ "$MASK_TRUNCATED_COMPLETIONS" = "true" ] && GEN_ARGS+=(--mask_truncated_completions)
  [ "$SHUFFLE_DATASET" = "true" ] && GEN_ARGS+=(--shuffle_dataset) || GEN_ARGS+=(--no-shuffle_dataset)

  set +e
  accelerate launch --num_processes "$NUM_GPUS" \
    --main_process_port "${CO_MAIN_PROCESS_PORT_GEN:-26200}" \
    src/pipelines/generate_and_score_cross.py "${GEN_ARGS[@]}"
  GEN_RC=$?
  set -e
  [ "$GEN_RC" -ne 0 ] && error "Cross-model generation/scoring failed at step $step"

  # ---- Train (trainee only) ----
  CKPT_DIR="$MODEL_DIR/checkpoint-$step"
  if [ -d "$CKPT_DIR" ]; then
    log "Checkpoint exists at step $step, skipping training."
  else
    TRAIN_ARGS=(
      --policy_model_path "$TRAINEE_PATH"
      --tokenizer_path "$TRAINEE_TOKENIZER_PATH"
      --cache_dir "$LATEST_CHECKPOINT_DIR"
      --step "$step"
      --reward_mode "$REWARD_MODE"
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
      --epsilon "$EPSILON"
      --num_iterations "$NUM_ITERATIONS"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --max_prompt_length "$MAX_PROMPT_LENGTH"
      --max_completion_length "$MAX_COMPLETION_LENGTH"
      --logging_steps 1 --logging_strategy steps --logging_first_step
      --save_steps 1 --save_strategy steps --save_total_limit 1
      --overwrite_output_dir
      --torch_dtype "$TORCH_DTYPE"
      --trust_remote_code
    )
    [ -n "$SYSTEM_PROMPT" ] && TRAIN_ARGS+=(--system_prompt "$SYSTEM_PROMPT")
    [ "$MASK_TRUNCATED_COMPLETIONS" = "true" ] && TRAIN_ARGS+=(--mask_truncated_completions)
    [ "$BF16" = "true" ] && TRAIN_ARGS+=(--bf16) || TRAIN_ARGS+=(--no_bf16)
    [ "$GRADIENT_CHECKPOINTING" = "true" ] && TRAIN_ARGS+=(--gradient_checkpointing) || TRAIN_ARGS+=(--no_gradient_checkpointing)

    if [ -f "$ACCELERATE_CONFIG" ]; then
      accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" \
        --main_process_port "${CO_MAIN_PROCESS_PORT_TRAIN:-25960}" \
        src/training/single_intuitor_train.py "${TRAIN_ARGS[@]}"
    else
      accelerate launch --num_processes "$NUM_GPUS" \
        --main_process_port "${CO_MAIN_PROCESS_PORT_TRAIN:-25960}" \
        src/training/single_intuitor_train.py "${TRAIN_ARGS[@]}"
    fi
  fi

  prune_old_checkpoints "$MODEL_DIR"

  # ---- Evaluate (trainee) ----
  CKPT_PATH="$MODEL_DIR/checkpoint-$step"
  log "--- Step $step: Evaluate ---"

  if [ "$LIGHTEVAL_ENABLED" = "true" ]; then
    EVAL_OUT_DIR="${LIGHTEVAL_OUT_PREFIX}/checkpoint-$step"
    log "  [lighteval] tasks=$LIGHTEVAL_TASKS"
    run_lighteval "$CKPT_PATH" "$EVAL_OUT_DIR" "$LIGHTEVAL_TASKS" "$SYSTEM_PROMPT" \
      "$EVAL_MAX_NEW_TOKENS" "$SEED" "$LIGHTEVAL_CUSTOM_TASKS" "$EVAL_LOCAL_DATASETS_ROOT" \
      "$LIGHTEVAL_DATA_PARALLEL_SIZE"
  fi

  if [ "$ZEROEVAL_ENABLED" = "true" ]; then
    log "  [zeroeval] tasks=$ZEROEVAL_TASKS"
    run_zeroeval "$CKPT_PATH" "trainee" "checkpoint-$step" "$ZEROEVAL_OUTPUT_ROOT" \
      "$ZEROEVAL_DIR" "$ZEROEVAL_TP_SIZE" "$ZEROEVAL_BATCH_SIZE" "$ZEROEVAL_MAX_TOKENS" \
      "$ZEROEVAL_GPU_MEM" "$SEED" "$ZEROEVAL_MAX_SAMPLES"
  fi

  if [ "$HACRL_EVAL_ENABLED" = "true" ]; then
    local hacrl_out="${HACRL_EVAL_OUTPUT_ROOT:-outputs/hacrl_eval/$(cfg name 'cross')}"
    log "  [hacrl] scorer=$HACRL_EVAL_SCORER tasks=$HACRL_EVAL_TASKS"
    run_hacrl_eval "$CKPT_PATH" "trainee" "checkpoint-$step" "$hacrl_out" \
      "$HACRL_EVAL_TASKS" "$HACRL_EVAL_SCORER" "$SYSTEM_PROMPT" "$SEED" \
      "$EVAL_MAX_NEW_TOKENS" "$HACRL_EVAL_VLLM_BATCH_SIZE" \
      "$HACRL_EVAL_NUM_GENERATIONS" "$HACRL_EVAL_MAX_SAMPLES" \
      "$HACRL_EVAL_TEMPERATURE" "$HACRL_EVAL_TOP_P"
  fi
done

log "Done. Latest trainee checkpoints: $MODEL_DIR"
