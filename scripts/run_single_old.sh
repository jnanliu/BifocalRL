#!/usr/bin/env bash
set -euo pipefail

# Config-driven single-model pipeline runner.
#
# One policy model is trained iteratively:
#   (1) generate via vLLM  (2) score+logps  (3) train one step (GRPO)
#
# Usage:
#   ./scripts/run_single.sh recipes/configs/single_intuitor.json
#   POLICY_INIT_PATH=/path/to/model ./scripts/run_single.sh recipes/configs/single_intuitor.json
#
# All parameters come from the JSON config file.
# Environment variables override JSON values.
#
# Typical single JSON defaults (see recipes/configs/single/*):
#   ~3B  → training.accelerator_config zero3,     per_device_batch_size 2
#   ~1.5B → zero3,     per_device_batch_size 4
#   ~4B  → zero3_offload, per_device_batch_size 16  (e.g. Qwen3-4B, Phi-4-mini)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"
cd "$SCRIPT_DIR/.." || exit 1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
CONFIG_FILE="${1:?Usage: $0 <config.json>}"
_load_config "$CONFIG_FILE"

# ============================================================================
# Read config (env vars override JSON)
# ============================================================================

# Models
POLICY_INIT_PATH="${POLICY_INIT_PATH:-$(cfg models.policy)}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$(cfg models.reward)}"

# Output
LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR:-$(cfg output.checkpoint_dir "outputs/single/latest_checkpoint")}"
MODEL_DIR="$LATEST_CHECKPOINT_DIR/model"
LIGHTEVAL_OUT_PREFIX="${LIGHTEVAL_OUT_PREFIX:-$(cfg output.lighteval_prefix "outputs/lighteval/single")}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-$(cfg output.zeroeval_root "outputs/zeroeval/single")}"
TTRL_EVAL_OUTPUT_ROOT="${TTRL_EVAL_OUTPUT_ROOT:-$(cfg output.ttrl_dir "")}"
KEEP_ALL_CHECKPOINTS="${KEEP_ALL_CHECKPOINTS:-$(cfg output.keep_all_checkpoints "false")}"

# Dataset
DATASET_KEY="${DATASET_KEY:-$(cfg dataset.key "math_lighteval")}"
MAX_SAMPLES="${MAX_SAMPLES:-$(cfg dataset.max_samples)}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$(cfg dataset.system_prompt)}"
DATASET_PROMPT_MODE="${DATASET_PROMPT_MODE:-$(cfg dataset.prompt_mode)}"
DATASET_INSTRUCTION_SUFFIX="${DATASET_INSTRUCTION_SUFFIX:-$(cfg dataset.instruction_suffix)}"
DATASET_ENABLE_THINKING="${DATASET_ENABLE_THINKING:-$(cfg dataset.enable_thinking "false")}"
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
REWARD_LENGTH_NORM="${REWARD_LENGTH_NORM:-$(cfg reward.length_norm "response")}"
REWARD_SCORER="${REWARD_SCORER:-$(cfg reward.scorer "hacrl_prime_math")}"
BETA="${BETA:-$(cfg reward.beta "0.0005")}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${POLICY_INIT_PATH}}"
MIXED_REWARD_ALPHA="${MIXED_REWARD_ALPHA:-$(cfg reward.mixed_reward_alpha "0.5")}"
MIXED_REWARD_ALPHA_MODE="${MIXED_REWARD_ALPHA_MODE:-$(cfg reward.mixed_reward_alpha_mode "fixed")}"
MIXED_REWARD_ALPHA_ZSCORE_MU="${MIXED_REWARD_ALPHA_ZSCORE_MU:-$(cfg reward.mixed_reward_alpha_zscore_mu "0.65")}"
MIXED_REWARD_ALPHA_ZSCORE_SIGMA="${MIXED_REWARD_ALPHA_ZSCORE_SIGMA:-$(cfg reward.mixed_reward_alpha_zscore_sigma "0.2")}"
MIXED_REWARD_ALPHA_ZSCORE_LOWER="${MIXED_REWARD_ALPHA_ZSCORE_LOWER:-$(cfg reward.mixed_reward_alpha_zscore_lower "0.0")}"
MIXED_REWARD_ALPHA_ZSCORE_UPPER="${MIXED_REWARD_ALPHA_ZSCORE_UPPER:-$(cfg reward.mixed_reward_alpha_zscore_upper "1.0")}"
MIXED_REWARD_CORR_WEIGHT_MODE="${MIXED_REWARD_CORR_WEIGHT_MODE:-$(cfg reward.mixed_reward_corr_weight_mode "none")}"
MIXED_REWARD_CORR_WEIGHT_THRESHOLD="${MIXED_REWARD_CORR_WEIGHT_THRESHOLD:-$(cfg reward.mixed_reward_corr_weight_threshold "0.1")}"
MIXED_REWARD_CORR_WEIGHT_EXPONENT="${MIXED_REWARD_CORR_WEIGHT_EXPONENT:-$(cfg reward.mixed_reward_corr_weight_exponent "0.8")}"
# mixed_reward_alpha = MV weight; Intuitor/SCE = 1 - alpha (co_intuitor_train.py / single_intuitor_train.py)
MIXED_REWARD_MV_WEIGHT="${MIXED_REWARD_MV_WEIGHT:-$MIXED_REWARD_ALPHA}"
MIXED_REWARD_INTUITOR_WEIGHT="${MIXED_REWARD_INTUITOR_WEIGHT:-$(awk -v a="$MIXED_REWARD_ALPHA" 'BEGIN { printf "%.4f", 1.0 - a }')}"

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
# Allocator tuning for accelerate-launched stages only (score/train).
# vLLM/generate/eval paths always strip this via `env -u`, so expandable_segments is safe here.
ACCELERATE_CUDA_ALLOC_CONF="${ACCELERATE_CUDA_ALLOC_CONF:-$(cfg training.accelerate_cuda_alloc_conf "expandable_segments:True")}"
ACCELERATE_EMPTY_CACHE_ON_BOUNDARY="${ACCELERATE_EMPTY_CACHE_ON_BOUNDARY:-$(cfg training.empty_cache_on_boundary "1")}"
ACCELERATE_EMPTY_CACHE_EVERY_STEP="${ACCELERATE_EMPTY_CACHE_EVERY_STEP:-$(cfg training.empty_cache_every_step "")}"

# GSPO settings (single pipeline supports grpo/gspo dual mode)
OBJECTIVE_MODE="${OBJECTIVE_MODE:-$(cfg training.objective_mode "grpo")}"
IS_LENGTH_NORM="${IS_LENGTH_NORM:-$(cfg training.is_length_norm "mean")}"

# Self-update GSPO epsilon overrides
SELF_UPDATE_EPSILON_LOW="${SELF_UPDATE_EPSILON_LOW:-$(cfg training.self_update_epsilon_low)}"
SELF_UPDATE_EPSILON_HIGH="${SELF_UPDATE_EPSILON_HIGH:-$(cfg training.self_update_epsilon_high)}"
# [TRUTHCHECK] Pipeline-level format/logic checks toggle.
PIPELINE_TRUTHCHECK="${PIPELINE_TRUTHCHECK:-$(cfg debug.pipeline_truthcheck "false")}"
export CO_PIPELINE_TRUTHCHECK="$PIPELINE_TRUTHCHECK"

# vLLM
USE_OFFLINE_VLLM="${USE_OFFLINE_VLLM:-$(cfg vllm.use_offline "true")}"
VLLM_PORT="${VLLM_PORT:-$(cfg vllm.port "7900")}"
VLLM_GPU="${VLLM_GPU:-$(cfg vllm.gpu "0")}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(cfg vllm.tensor_parallel_size "1")}"
VLLM_GPU_MEM="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-$(cfg vllm.gpu_memory_utilization "0.9")}"
VLLM_DTYPE="${VLLM_SERVER_DTYPE:-$(cfg vllm.dtype "bfloat16")}"
VLLM_MAX_MODEL_LEN="${VLLM_SERVER_MAX_MODEL_LEN:-$(cfg vllm.max_model_len "4096")}"
VLLM_MAX_NUM_SEQS="${VLLM_SERVER_MAX_NUM_SEQS:-$(cfg vllm.max_num_seqs "128")}"
VLLM_MAX_BATCHED="${VLLM_SERVER_MAX_NUM_BATCHED_TOKENS:-$(cfg vllm.max_num_batched_tokens "8192")}"
VLLM_CHUNKED="${VLLM_SERVER_CHUNKED_PREFILL:-$(cfg vllm.chunked_prefill "true")}"
VLLM_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-$(cfg vllm.start_timeout "900")}"
AUTO_START_SERVER="${AUTO_START_SERVER:-$(cfg vllm.auto_start "false")}"
AUTO_STOP_SERVER="${AUTO_STOP_SERVER:-$(cfg vllm.auto_stop "true")}"

# DAPO settings
OVERLONG_PENALTY="${OVERLONG_PENALTY:-$(cfg generation.overlong_penalty)}"
FILTER_GROUPS="${FILTER_GROUPS:-$(cfg training.filter_groups "false")}"

# Eval – shared
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-$(cfg eval.max_new_tokens "2048")}"
EVAL_LOCAL_DATASETS_ROOT="${EVAL_LOCAL_DATASETS_ROOT:-$(cfg eval.local_datasets_root "/home/jizeyu26/jzy/7_co_learning/test/datasets")}"

# Eval – lighteval
LIGHTEVAL_ENABLED="${LIGHTEVAL_ENABLED:-$(cfg eval.lighteval.enabled "false")}"
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-$(cfg eval.lighteval.tasks "math_500,gsm8k")}"
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-$(cfg eval.lighteval.custom_tasks "eval.lighteval.custom_tasks.offline_patch")}"
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-$(cfg eval.lighteval.data_parallel_size "1")}"
LIGHTEVAL_PROMPT_MODE="${LIGHTEVAL_PROMPT_MODE:-$(cfg eval.lighteval.prompt_mode "system")}"
LIGHTEVAL_SYSTEM_PROMPT="${LIGHTEVAL_SYSTEM_PROMPT:-$(cfg eval.lighteval.system_prompt)}"
if [ -z "$LIGHTEVAL_SYSTEM_PROMPT" ]; then
  LIGHTEVAL_SYSTEM_PROMPT="$SYSTEM_PROMPT"
fi
LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT="$LIGHTEVAL_SYSTEM_PROMPT"
case "$LIGHTEVAL_PROMPT_MODE" in
  bare|suffix) LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT="" ;;
  system|"") ;;
  *) log "WARNING: unknown eval.lighteval.prompt_mode='$LIGHTEVAL_PROMPT_MODE', fallback to system" ;;
esac

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
HACRL_EVAL_USE_OFFLINE="${HACRL_EVAL_USE_OFFLINE:-$(cfg eval.hacrl.use_offline "true")}"
HACRL_EVAL_GPU_MEM="${HACRL_EVAL_GPU_MEM:-$(cfg eval.hacrl.gpu_memory_utilization "$VLLM_GPU_MEM")}"
HACRL_EVAL_TP_SIZE="${HACRL_EVAL_TP_SIZE:-$(cfg eval.hacrl.tensor_parallel_size "$VLLM_TP_SIZE")}"
HACRL_EVAL_DTYPE="${HACRL_EVAL_DTYPE:-$(cfg eval.hacrl.dtype "$VLLM_DTYPE")}"
HACRL_EVAL_MAX_MODEL_LEN="${HACRL_EVAL_MAX_MODEL_LEN:-$(cfg eval.hacrl.max_model_len "$VLLM_MAX_MODEL_LEN")}"
HACRL_EVAL_CUDA_DEVICE="${HACRL_EVAL_CUDA_DEVICE:-$(cfg eval.hacrl.cuda_device "")}"

# Eval – ttrl
TTRL_EVAL_ENABLED="${TTRL_EVAL_ENABLED:-$(cfg eval.ttrl.enabled "false")}"
TTRL_EVAL_TASKS="${TTRL_EVAL_TASKS:-$(cfg eval.ttrl.tasks "math_ttt,aime_ttt")}"
TTRL_EVAL_SCORER="${TTRL_EVAL_SCORER:-$(cfg eval.ttrl.scorer "ttrl_math")}"
TTRL_EVAL_NUM_GENERATIONS="${TTRL_EVAL_NUM_GENERATIONS:-$(cfg eval.ttrl.num_generations "1")}"
TTRL_EVAL_MAX_SAMPLES="${TTRL_EVAL_MAX_SAMPLES:-$(cfg eval.ttrl.max_samples "")}"
TTRL_EVAL_PROMPT_MODE="${TTRL_EVAL_PROMPT_MODE:-$(cfg eval.ttrl.prompt_mode "bare")}"
TTRL_EVAL_TEMPERATURE="${TTRL_EVAL_TEMPERATURE:-$(cfg eval.ttrl.temperature "0.0")}"
TTRL_EVAL_TOP_P="${TTRL_EVAL_TOP_P:-$(cfg eval.ttrl.top_p "1.0")}"
TTRL_EVAL_GPU_MEM="${TTRL_EVAL_GPU_MEM:-$(cfg eval.ttrl.gpu_memory_utilization "0.9")}"
TTRL_EVAL_TP_SIZE="${TTRL_EVAL_TP_SIZE:-$(cfg eval.ttrl.tensor_parallel_size "1")}"
TTRL_EVAL_DTYPE="${TTRL_EVAL_DTYPE:-$(cfg eval.ttrl.dtype "bfloat16")}"
TTRL_EVAL_MAX_MODEL_LEN="${TTRL_EVAL_MAX_MODEL_LEN:-$(cfg eval.ttrl.max_model_len "$VLLM_MAX_MODEL_LEN")}"
TTRL_EVAL_DATASET_OVERRIDES="${TTRL_EVAL_DATASET_OVERRIDES:-$(cfg_json eval.ttrl.dataset_overrides)}"

# Wandb
WANDB_API_KEY="${WANDB_API_KEY:-}"
[ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY

# Backward compat for old flat eval config
_eval_compat

# ============================================================================
# Validate & Setup
# ============================================================================
[ -z "$POLICY_INIT_PATH" ] && error "models.policy is required in config"
setup_debug "$LATEST_CHECKPOINT_DIR"
mkdir -p "$LATEST_CHECKPOINT_DIR" "$MODEL_DIR"
if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
  prune_old_checkpoints "$MODEL_DIR"
fi

EFF_BATCH=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
log "================================================================"
log "Single-model pipeline (config-driven)"
log "  config:       $CONFIG_FILE"
log "  name:         $(cfg name '<unnamed>')"
log "  policy:       $POLICY_INIT_PATH"
log "  reward_model: ${REWARD_MODEL_PATH:-<self>}"
log "  reward_mode:  $REWARD_MODE"
if [ "$REWARD_MODE" = "mixed_reward" ]; then
  log "  mixed_alpha:  $MIXED_REWARD_ALPHA (MV=α, Intuitor=1-α)"
  log "  mixed_mv_w:   $MIXED_REWARD_MV_WEIGHT"
  log "  mixed_int_w:  $MIXED_REWARD_INTUITOR_WEIGHT"
fi
log "  reward_lnorm: $REWARD_LENGTH_NORM"
log "  num_gen:      $NUM_GENERATIONS"
log "  steps:        $START_STEP .. $((MAX_STEPS))"
log "  eff_batch:    ${NUM_GPUS} × ${PER_DEVICE_TRAIN_BATCH_SIZE} × ${GRADIENT_ACCUMULATION_STEPS} = ${EFF_BATCH}"
log "  keep_ckpts:   $KEEP_ALL_CHECKPOINTS"
log "  truthcheck:   $CO_PIPELINE_TRUTHCHECK"
log "================================================================"

# ============================================================================
# Helpers
# ============================================================================
_run_accelerate_launch() {
  local mode="$1"; shift
  local main_port="$1"; shift
  local -a env_vars=(env)
  [ -n "$ACCELERATE_CUDA_ALLOC_CONF" ] && env_vars+=(PYTORCH_CUDA_ALLOC_CONF="$ACCELERATE_CUDA_ALLOC_CONF")
  [ -n "$ACCELERATE_EMPTY_CACHE_ON_BOUNDARY" ] && env_vars+=(CO_EMPTY_CACHE_ON_BOUNDARY="$ACCELERATE_EMPTY_CACHE_ON_BOUNDARY")
  [ -n "$ACCELERATE_EMPTY_CACHE_EVERY_STEP" ] && env_vars+=(CO_EMPTY_CACHE_EVERY_STEP="$ACCELERATE_EMPTY_CACHE_EVERY_STEP")

  if [ "$mode" = "train" ] && [ -f "$ACCELERATE_CONFIG" ]; then
    "${env_vars[@]}" accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_GPUS" \
      --main_process_port "$main_port" "$@"
  else
    "${env_vars[@]}" accelerate launch --num_processes "$NUM_GPUS" \
      --main_process_port "$main_port" "$@"
  fi
}

get_latest_checkpoint() {
  if [ -d "$MODEL_DIR" ]; then
    local last
    last=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$last" ]; then echo "$last"; return; fi
  fi
  echo "$POLICY_INIT_PATH"
}

resolve_training_prompt_controls() {
  TRAIN_SYSTEM_PROMPT="$SYSTEM_PROMPT"
  TRAIN_INSTRUCTION_SUFFIX_MODE="default"  # default|custom|empty
  TRAIN_INSTRUCTION_SUFFIX_VALUE=""
  TRAIN_ENABLE_THINKING="$DATASET_ENABLE_THINKING"

  case "${DATASET_PROMPT_MODE:-}" in
    system)
      TRAIN_INSTRUCTION_SUFFIX_MODE="empty"
      TRAIN_ENABLE_THINKING="none"
      ;;
    suffix)
      TRAIN_SYSTEM_PROMPT=""
      if [ -n "$DATASET_INSTRUCTION_SUFFIX" ]; then
        TRAIN_INSTRUCTION_SUFFIX_MODE="custom"
        TRAIN_INSTRUCTION_SUFFIX_VALUE="$DATASET_INSTRUCTION_SUFFIX"
      else
        TRAIN_INSTRUCTION_SUFFIX_MODE="default"
      fi
      TRAIN_ENABLE_THINKING="none"
      ;;
    bare)
      TRAIN_SYSTEM_PROMPT=""
      TRAIN_INSTRUCTION_SUFFIX_MODE="empty"
      TRAIN_ENABLE_THINKING="none"
      ;;
    *)
      if [ -n "$DATASET_INSTRUCTION_SUFFIX" ]; then
        TRAIN_INSTRUCTION_SUFFIX_MODE="custom"
        TRAIN_INSTRUCTION_SUFFIX_VALUE="$DATASET_INSTRUCTION_SUFFIX"
      fi
      ;;
  esac
}

resolve_training_prompt_controls

# ============================================================================
# Main Loop
# ============================================================================

# ---- Evaluate ----
if [ "$START_STEP" = "1" ]; then
  log "--- Step 0: Evaluate ---"

  if [ "$TTRL_EVAL_ENABLED" = "true" ]; then
    ttrl_out="${TTRL_EVAL_OUTPUT_ROOT:-outputs/ttrl_eval/$(cfg name 'single')}"
    log "  [ttrl] scorer=$TTRL_EVAL_SCORER tasks=$TTRL_EVAL_TASKS mode=$TTRL_EVAL_PROMPT_MODE"
    run_ttrl_eval "$POLICY_INIT_PATH" "a" "checkpoint-0" "$ttrl_out" \
      "$TTRL_EVAL_TASKS" "$TTRL_EVAL_SCORER" "$SEED" \
      "$EVAL_MAX_NEW_TOKENS" "$TTRL_EVAL_NUM_GENERATIONS" "$TTRL_EVAL_MAX_SAMPLES" \
      "$TTRL_EVAL_TEMPERATURE" "$TTRL_EVAL_TOP_P" \
      "$TTRL_EVAL_PROMPT_MODE" \
      "$TTRL_EVAL_GPU_MEM" "$TTRL_EVAL_TP_SIZE" "$TTRL_EVAL_DTYPE" \
      "$TTRL_EVAL_MAX_MODEL_LEN" \
      "$TTRL_EVAL_DATASET_OVERRIDES"
  fi
fi


for step in $(seq "$START_STEP" $((MAX_STEPS))); do
  log "======================== Step $step ========================"
  POLICY_PATH=$(get_latest_checkpoint)
  log "Policy: $POLICY_PATH"

  export CUDA_VISIBLE_DEVICES="$(_gpu_list "$NUM_GPUS")"

  # ---- Generate (no accelerate — avoids vLLM v1 NCCL conflicts) ----
  log "--- Step $step: Generate ---"
  GEN_ARGS=(
    --model_path "$POLICY_PATH"
    --tokenizer_path "$POLICY_INIT_PATH"
    --cache_slot a
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
    --enable_thinking "$TRAIN_ENABLE_THINKING"
  )
  [ -n "$MAX_SAMPLES" ] && GEN_ARGS+=(--max_samples "$MAX_SAMPLES")
  [ -n "$TRAIN_SYSTEM_PROMPT" ] && GEN_ARGS+=(--system_prompt "$TRAIN_SYSTEM_PROMPT")
  if [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "custom" ]; then
    GEN_ARGS+=(--instruction_suffix "$TRAIN_INSTRUCTION_SUFFIX_VALUE")
  elif [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "empty" ]; then
    GEN_ARGS+=(--instruction_suffix "")
  fi
  [ "$SHUFFLE_DATASET" = "true" ] && GEN_ARGS+=(--shuffle_dataset) || GEN_ARGS+=(--no-shuffle_dataset)

  if [ "$USE_OFFLINE_VLLM" = "true" ]; then
    GEN_ARGS+=(--use_offline_vllm)
    GEN_ARGS+=(--vllm_gpu "$VLLM_GPU")
    GEN_ARGS+=(--tensor_parallel_size "$VLLM_TP_SIZE")
    GEN_ARGS+=(--gpu_memory_utilization "$VLLM_GPU_MEM")
    GEN_ARGS+=(--vllm_dtype "$VLLM_DTYPE")
    GEN_ARGS+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
    GEN_ARGS+=(--vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS")
    GEN_ARGS+=(--vllm_max_num_batched_tokens "$VLLM_MAX_BATCHED")
    [ "$VLLM_CHUNKED" = "true" ] && GEN_ARGS+=(--enable_chunked_prefill) || GEN_ARGS+=(--no-enable_chunked_prefill)
  else
    GEN_ARGS+=(--no-use_offline_vllm)
    GEN_ARGS+=(--vllm_port "$VLLM_PORT")
    [ "$AUTO_START_SERVER" = "true" ] && GEN_ARGS+=(--auto_start_server) || GEN_ARGS+=(--no-auto_start_server)
    [ "$AUTO_STOP_SERVER" = "true" ] && GEN_ARGS+=(--auto_stop_server) || GEN_ARGS+=(--no-auto_stop_server)
    _start_vllm_server "$POLICY_PATH" "$VLLM_PORT" "$VLLM_GPU" \
      "$VLLM_GPU_MEM" "$VLLM_DTYPE" "$VLLM_MAX_MODEL_LEN" \
      "$VLLM_MAX_NUM_SEQS" "$VLLM_MAX_BATCHED" "$VLLM_CHUNKED" "$VLLM_TIMEOUT"
  fi

  set +e
  env -u PYTORCH_CUDA_ALLOC_CONF python src/pipelines/generate_only.py "${GEN_ARGS[@]}"
  GEN_RC=$?
  set -e

  if [ "$USE_OFFLINE_VLLM" != "true" ]; then
    _vllm_kill_port "$VLLM_PORT"
  fi
  [ "$GEN_RC" -ne 0 ] && error "Generation failed at step $step"

  # ---- Score + Compute logps (with accelerate) ----
  log "--- Step $step: Score and Compute ---"
  SCORE_ARGS=(
    --policy_model_path "$POLICY_PATH"
    --tokenizer_path "$POLICY_INIT_PATH"
    --dataset_key "$DATASET_KEY"
    --output_dir "$LATEST_CHECKPOINT_DIR"
    --step "$step"
    --seed "$SEED"
    --problems_per_step "$PROBLEMS_PER_STEP"
    --num_generations "$NUM_GENERATIONS"
    --temperature "$TEMPERATURE"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --scoring_batch_size "$SCORING_BATCH_SIZE"
    --logps_batch_size "$LOGPS_BATCH_SIZE"
    --beta "$BETA"
    --ref_model_path "$REF_MODEL_PATH"
    --num_iterations "$NUM_ITERATIONS"
    --reward_mode "$REWARD_MODE"
    --scorer "$REWARD_SCORER"
    --enable_thinking "$TRAIN_ENABLE_THINKING"
  )
  [ -n "$REWARD_MODEL_PATH" ] && SCORE_ARGS+=(--reward_model_path "$REWARD_MODEL_PATH")
  [ -n "$MAX_SAMPLES" ] && SCORE_ARGS+=(--max_samples "$MAX_SAMPLES")
  [ -n "$TRAIN_SYSTEM_PROMPT" ] && SCORE_ARGS+=(--system_prompt "$TRAIN_SYSTEM_PROMPT")
  if [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "custom" ]; then
    SCORE_ARGS+=(--instruction_suffix "$TRAIN_INSTRUCTION_SUFFIX_VALUE")
  elif [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "empty" ]; then
    SCORE_ARGS+=(--instruction_suffix "")
  fi
  [ "$MASK_TRUNCATED_COMPLETIONS" = "true" ] && SCORE_ARGS+=(--mask_truncated_completions)
  [ "$SHUFFLE_DATASET" = "true" ] && SCORE_ARGS+=(--shuffle_dataset) || SCORE_ARGS+=(--no-shuffle_dataset)
  [ -n "$OVERLONG_PENALTY" ] && SCORE_ARGS+=(--overlong_penalty "$OVERLONG_PENALTY")
  if [ "$OBJECTIVE_MODE" = "gspo" ]; then
    SCORE_ARGS+=(--force_compute_old_logps)
  fi

  set +e
  _run_accelerate_launch "score" "${CO_MAIN_PROCESS_PORT_GEN:-$(_find_free_port)}" \
    src/pipelines/score_and_compute_single.py "${SCORE_ARGS[@]}"
  SCORE_RC=$?
  set -e
  [ "$SCORE_RC" -ne 0 ] && error "Scoring/compute failed at step $step"

  # ---- Train ----
  CKPT_DIR="$MODEL_DIR/checkpoint-$step"
  if [ -d "$CKPT_DIR" ]; then
    log "Checkpoint exists at step $step, skipping training."
  else
    TRAIN_ARGS=(
      --policy_model_path "$POLICY_PATH"
      --tokenizer_path "$POLICY_INIT_PATH"
      --cache_dir "$LATEST_CHECKPOINT_DIR"
      --step "$step"
      --reward_mode "$REWARD_MODE"
      --reward_length_norm "$REWARD_LENGTH_NORM"
      --output_dir "$MODEL_DIR"
      --max_steps 1
      --total_pipeline_steps "$MAX_STEPS"
      --current_pipeline_step "$step"
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
      --save_steps 1 --save_strategy steps
      --overwrite_output_dir
      --torch_dtype "$TORCH_DTYPE"
      --trust_remote_code
      --objective_mode "$OBJECTIVE_MODE"
      --is_length_norm "$IS_LENGTH_NORM"
    )
    [ -n "$TRAIN_SYSTEM_PROMPT" ] && TRAIN_ARGS+=(--system_prompt "$TRAIN_SYSTEM_PROMPT")
    [ "$MASK_TRUNCATED_COMPLETIONS" = "true" ] && TRAIN_ARGS+=(--mask_truncated_completions)
    [ "$BF16" = "true" ] && TRAIN_ARGS+=(--bf16) || TRAIN_ARGS+=(--no_bf16)
    [ "$GRADIENT_CHECKPOINTING" = "true" ] && TRAIN_ARGS+=(--gradient_checkpointing) || TRAIN_ARGS+=(--no_gradient_checkpointing)
    if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
      TRAIN_ARGS+=(--save_total_limit 1)
    fi

    # mixed_reward alpha
    if [ "$REWARD_MODE" = "mixed_reward" ]; then
      TRAIN_ARGS+=(--mixed_reward_alpha "$MIXED_REWARD_ALPHA")
      TRAIN_ARGS+=(--mixed_reward_alpha_mode "$MIXED_REWARD_ALPHA_MODE")
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_mu "$MIXED_REWARD_ALPHA_ZSCORE_MU")
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_sigma "$MIXED_REWARD_ALPHA_ZSCORE_SIGMA")
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_lower "$MIXED_REWARD_ALPHA_ZSCORE_LOWER")
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_upper "$MIXED_REWARD_ALPHA_ZSCORE_UPPER")
      TRAIN_ARGS+=(--mixed_reward_corr_weight_mode "$MIXED_REWARD_CORR_WEIGHT_MODE")
      TRAIN_ARGS+=(--mixed_reward_corr_weight_threshold "$MIXED_REWARD_CORR_WEIGHT_THRESHOLD")
      TRAIN_ARGS+=(--mixed_reward_corr_weight_exponent "$MIXED_REWARD_CORR_WEIGHT_EXPONENT")
    fi

    # Self-update epsilon overrides
    [ -n "$SELF_UPDATE_EPSILON_LOW" ] && TRAIN_ARGS+=(--self_update_epsilon_low "$SELF_UPDATE_EPSILON_LOW")
    [ -n "$SELF_UPDATE_EPSILON_HIGH" ] && TRAIN_ARGS+=(--self_update_epsilon_high "$SELF_UPDATE_EPSILON_HIGH")

    # DAPO: dynamic sampling (filter all-correct/all-wrong groups)
    [ "$FILTER_GROUPS" = "true" ] && TRAIN_ARGS+=(--filter_groups)

    _run_accelerate_launch "train" "${CO_MAIN_PROCESS_PORT_TRAIN:-$(_find_free_port)}" \
      src/training/single_intuitor_train.py "${TRAIN_ARGS[@]}"
  fi

  if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
    prune_old_checkpoints "$MODEL_DIR"
  fi

  # ---- Evaluate ----
  CKPT_PATH="$MODEL_DIR/checkpoint-$step"
  log "--- Step $step: Evaluate ---"

  if [ "$LIGHTEVAL_ENABLED" = "true" ]; then
    EVAL_OUT_DIR="${LIGHTEVAL_OUT_PREFIX}/checkpoint-$step"
    log "  [lighteval] tasks=$LIGHTEVAL_TASKS"
    run_lighteval "$CKPT_PATH" "$EVAL_OUT_DIR" "$LIGHTEVAL_TASKS" "$LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT" \
      "$EVAL_MAX_NEW_TOKENS" "$SEED" "$LIGHTEVAL_CUSTOM_TASKS" "$EVAL_LOCAL_DATASETS_ROOT" \
      "$LIGHTEVAL_DATA_PARALLEL_SIZE"
  fi

  if [ "$ZEROEVAL_ENABLED" = "true" ]; then
    log "  [zeroeval] tasks=$ZEROEVAL_TASKS"
    run_zeroeval "$CKPT_PATH" "a" "checkpoint-$step" "$ZEROEVAL_OUTPUT_ROOT" \
      "$ZEROEVAL_DIR" "$ZEROEVAL_TP_SIZE" "$ZEROEVAL_BATCH_SIZE" "$ZEROEVAL_MAX_TOKENS" \
      "$ZEROEVAL_GPU_MEM" "$SEED" "$ZEROEVAL_MAX_SAMPLES"
  fi

  if [ "$HACRL_EVAL_ENABLED" = "true" ]; then
    hacrl_out="${HACRL_EVAL_OUTPUT_ROOT:-outputs/hacrl_eval/$(cfg name 'single')}"
    log "  [hacrl] scorer=$HACRL_EVAL_SCORER tasks=$HACRL_EVAL_TASKS"
    run_hacrl_eval "$CKPT_PATH" "a" "checkpoint-$step" "$hacrl_out" \
      "$HACRL_EVAL_TASKS" "$HACRL_EVAL_SCORER" "$SYSTEM_PROMPT" "$SEED" \
      "$EVAL_MAX_NEW_TOKENS" "$HACRL_EVAL_VLLM_BATCH_SIZE" \
      "$HACRL_EVAL_NUM_GENERATIONS" "$HACRL_EVAL_MAX_SAMPLES" \
      "$HACRL_EVAL_TEMPERATURE" "$HACRL_EVAL_TOP_P" \
      "$HACRL_EVAL_USE_OFFLINE" "$HACRL_EVAL_GPU_MEM" "$HACRL_EVAL_TP_SIZE" \
      "$HACRL_EVAL_DTYPE" "$HACRL_EVAL_MAX_MODEL_LEN" "$HACRL_EVAL_CUDA_DEVICE"
  fi

  if [ "$TTRL_EVAL_ENABLED" = "true" ]; then
    ttrl_out="${TTRL_EVAL_OUTPUT_ROOT:-outputs/ttrl_eval/$(cfg name 'single')}"
    log "  [ttrl] scorer=$TTRL_EVAL_SCORER tasks=$TTRL_EVAL_TASKS mode=$TTRL_EVAL_PROMPT_MODE"
    run_ttrl_eval "$CKPT_PATH" "a" "checkpoint-$step" "$ttrl_out" \
      "$TTRL_EVAL_TASKS" "$TTRL_EVAL_SCORER" "$SEED" \
      "$EVAL_MAX_NEW_TOKENS" "$TTRL_EVAL_NUM_GENERATIONS" "$TTRL_EVAL_MAX_SAMPLES" \
      "$TTRL_EVAL_TEMPERATURE" "$TTRL_EVAL_TOP_P" \
      "$TTRL_EVAL_PROMPT_MODE" \
      "$TTRL_EVAL_GPU_MEM" "$TTRL_EVAL_TP_SIZE" "$TTRL_EVAL_DTYPE" \
      "$TTRL_EVAL_MAX_MODEL_LEN" \
      "$TTRL_EVAL_DATASET_OVERRIDES"
  fi
done

log "Done. Latest checkpoints: $MODEL_DIR"
