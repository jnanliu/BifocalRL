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
#   MODEL_ROOT=/path/to/models ./scripts/run_single.sh recipes/configs/single_intuitor.json
#   ./scripts/run_single.sh recipes/configs/single_intuitor.json \
#     --eval-only --eval-profile recipes/configs/eval/extended_eval.json
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

usage() {
  cat <<EOF
Usage:
  $0 <config.json>
  $0 <config.json> --eval-only --eval-profile <profile.json> [options]

Eval-only options:
  --eval-start-step N  First checkpoint to evaluate (0 = initial model)
  --eval-end-step N    Last checkpoint to evaluate
  --eval-force         Ignore exact-signature completion markers
  --eval-dry-run       Validate and print work without launching evaluators
EOF
}

CONFIG_FILE=""
EVAL_ONLY="${EVAL_ONLY:-false}"
EVAL_PROFILE_FILE="${EVAL_PROFILE_FILE:-}"
EVAL_ONLY_START_STEP="${EVAL_ONLY_START_STEP:-}"
EVAL_ONLY_END_STEP="${EVAL_ONLY_END_STEP:-}"
EVAL_FORCE="${EVAL_FORCE:-false}"
EVAL_ONLY_DRY_RUN="${EVAL_ONLY_DRY_RUN:-false}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --eval-only)
      EVAL_ONLY="true"
      shift
      ;;
    --eval-profile)
      [ "$#" -ge 2 ] || error "--eval-profile requires a JSON path"
      EVAL_PROFILE_FILE="$2"
      shift 2
      ;;
    --eval-start-step)
      [ "$#" -ge 2 ] || error "--eval-start-step requires an integer"
      EVAL_ONLY_START_STEP="$2"
      shift 2
      ;;
    --eval-end-step)
      [ "$#" -ge 2 ] || error "--eval-end-step requires an integer"
      EVAL_ONLY_END_STEP="$2"
      shift 2
      ;;
    --eval-force)
      EVAL_FORCE="true"
      shift
      ;;
    --eval-dry-run)
      EVAL_ONLY_DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      error "Unknown option: $1"
      ;;
    *)
      [ -z "$CONFIG_FILE" ] || error "Only one training config may be provided"
      CONFIG_FILE="$1"
      shift
      ;;
  esac
done

[ -n "$CONFIG_FILE" ] || { usage >&2; error "A training config is required"; }
[ "$EVAL_ONLY" = "true" ] || [ -z "$EVAL_PROFILE_FILE" ] || error "--eval-profile requires --eval-only"
[ "$EVAL_ONLY" = "true" ] || [ "$EVAL_FORCE" = "false" ] || error "--eval-force requires --eval-only"
[ "$EVAL_ONLY" = "true" ] || [ "$EVAL_ONLY_DRY_RUN" = "false" ] || error "--eval-dry-run requires --eval-only"
[ "$EVAL_ONLY" != "true" ] || [ -n "$EVAL_PROFILE_FILE" ] || error "--eval-only requires --eval-profile"

_load_config "$CONFIG_FILE"
[ -z "$EVAL_PROFILE_FILE" ] || _load_eval_profile "$EVAL_PROFILE_FILE"

# ============================================================================
# Read config (env vars override JSON)
# ============================================================================

# Models
CONFIG_POLICY_PATH="$(cfg models.policy)"
if [ -n "${POLICY_INIT_PATH:-}" ]; then
  : # An explicit model path has highest priority.
elif [ -n "${MODEL_ROOT:-}" ]; then
  [ -n "$CONFIG_POLICY_PATH" ] || error "models.policy is required when MODEL_ROOT is set"
  POLICY_INIT_PATH="${MODEL_ROOT%/}/${CONFIG_POLICY_PATH##*/}"
else
  POLICY_INIT_PATH="$CONFIG_POLICY_PATH"
fi
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$(cfg models.reward)}"

# Output
OUTPUT_GROUP="$(cfg output_group "single")"
EXPERIMENT_NAME="$(cfg name "unnamed")"
MODEL_TAG="$(cfg models.model_tag "model")"
LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR:-$(cfg output.checkpoint_dir "outputs/single/latest_checkpoint")}"
MODEL_DIR="$LATEST_CHECKPOINT_DIR/model"
TRAINING_STATE_DIR="${TRAINING_STATE_DIR:-$(cfg output.training_state_dir "$LATEST_CHECKPOINT_DIR/training_state")}"
if [ "$EVAL_ONLY" = "true" ]; then
  EVAL_PROFILE_OUTPUT_SUFFIX="$(eval_profile_cfg profile.output_suffix "extended-eval")"
  EVAL_PROFILE_OUTPUT_ROOT="$(eval_profile_cfg profile.output_root "")"
  if [ -z "$EVAL_PROFILE_OUTPUT_ROOT" ]; then
    EVAL_PROFILE_OUTPUT_ROOT="outputs/${OUTPUT_GROUP}/eval/${MODEL_TAG}/${EXPERIMENT_NAME}-${EVAL_PROFILE_OUTPUT_SUFFIX}"
  fi
  LIGHTEVAL_OUT_PREFIX="${LIGHTEVAL_OUT_PREFIX:-$EVAL_PROFILE_OUTPUT_ROOT/lighteval}"
  ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-$EVAL_PROFILE_OUTPUT_ROOT/zeroeval}"
  HACRL_EVAL_OUTPUT_ROOT="${HACRL_EVAL_OUTPUT_ROOT:-$EVAL_PROFILE_OUTPUT_ROOT/hacrl}"
  TTRL_EVAL_OUTPUT_ROOT="${TTRL_EVAL_OUTPUT_ROOT:-$EVAL_PROFILE_OUTPUT_ROOT}"
else
  LIGHTEVAL_OUT_PREFIX="${LIGHTEVAL_OUT_PREFIX:-$(cfg output.lighteval_prefix "outputs/lighteval/single")}"
  ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-$(cfg output.zeroeval_root "outputs/zeroeval/single")}"
  TTRL_EVAL_OUTPUT_ROOT="${TTRL_EVAL_OUTPUT_ROOT:-$(cfg output.ttrl_dir "")}"
  HACRL_EVAL_OUTPUT_ROOT="${HACRL_EVAL_OUTPUT_ROOT:-$(cfg output.hacrl_root "")}"
fi
KEEP_ALL_CHECKPOINTS="${KEEP_ALL_CHECKPOINTS:-$(cfg output.keep_all_checkpoints "false")}"
CHECKPOINT_KEEP_LIMIT="${CHECKPOINT_KEEP_LIMIT:-$(cfg output.checkpoint_keep_limit "1")}"

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
MIXED_REWARD_FINAL_NORMALIZE="${MIXED_REWARD_FINAL_NORMALIZE:-$(cfg reward.mixed_reward_final_normalize "true")}"
MIXED_REWARD_ALPHA_ZSCORE_MU="${MIXED_REWARD_ALPHA_ZSCORE_MU:-$(cfg reward.mixed_reward_alpha_zscore_mu "0.65")}"
MIXED_REWARD_ALPHA_ZSCORE_SIGMA="${MIXED_REWARD_ALPHA_ZSCORE_SIGMA:-$(cfg reward.mixed_reward_alpha_zscore_sigma "0.2")}"
MIXED_REWARD_ALPHA_ZSCORE_LOWER="${MIXED_REWARD_ALPHA_ZSCORE_LOWER:-$(cfg reward.mixed_reward_alpha_zscore_lower "0.0")}"
MIXED_REWARD_ALPHA_ZSCORE_UPPER="${MIXED_REWARD_ALPHA_ZSCORE_UPPER:-$(cfg reward.mixed_reward_alpha_zscore_upper "1.0")}"
MIXED_REWARD_CORR_WEIGHT_MODE="${MIXED_REWARD_CORR_WEIGHT_MODE:-$(cfg reward.mixed_reward_corr_weight_mode "none")}"
MIXED_REWARD_CORR_WEIGHT_THRESHOLD="${MIXED_REWARD_CORR_WEIGHT_THRESHOLD:-$(cfg reward.mixed_reward_corr_weight_threshold "0.1")}"
MIXED_REWARD_CORR_WEIGHT_EXPONENT="${MIXED_REWARD_CORR_WEIGHT_EXPONENT:-$(cfg reward.mixed_reward_corr_weight_exponent "0.8")}"
MIXED_REWARD_RELIABILITY_SC_WEIGHT="${MIXED_REWARD_RELIABILITY_SC_WEIGHT:-$(cfg reward.mixed_reward_reliability_sc_weight "0.6")}"
MIXED_REWARD_RELIABILITY_CORR_WEIGHT="${MIXED_REWARD_RELIABILITY_CORR_WEIGHT:-$(cfg reward.mixed_reward_reliability_corr_weight "0.4")}"
MIXED_REWARD_RELIABILITY_USE_SAMPLE_WEIGHT="${MIXED_REWARD_RELIABILITY_USE_SAMPLE_WEIGHT:-$(cfg reward.mixed_reward_reliability_use_sample_weight "true")}"
MIXED_REWARD_RELIABILITY_USE_ADVANTAGE_WEIGHT="${MIXED_REWARD_RELIABILITY_USE_ADVANTAGE_WEIGHT:-$(cfg reward.mixed_reward_reliability_use_advantage_weight "true")}"
MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_MU="${MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_MU:-$(cfg reward.mixed_reward_reliability_sample_weight_mu "1.0")}"
MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_SIGMA="${MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_SIGMA:-$(cfg reward.mixed_reward_reliability_sample_weight_sigma "0.2")}"
MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_LOWER="${MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_LOWER:-$(cfg reward.mixed_reward_reliability_sample_weight_lower "0.5")}"
MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_UPPER="${MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_UPPER:-$(cfg reward.mixed_reward_reliability_sample_weight_upper "1.5")}"
MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_MU="${MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_MU:-$(cfg reward.mixed_reward_reliability_advantage_weight_mu "0.5")}"
MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_SIGMA="${MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_SIGMA:-$(cfg reward.mixed_reward_reliability_advantage_weight_sigma "0.2")}"
MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_LOWER="${MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_LOWER:-$(cfg reward.mixed_reward_reliability_advantage_weight_lower "0.2")}"
MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_UPPER="${MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_UPPER:-$(cfg reward.mixed_reward_reliability_advantage_weight_upper "0.8")}"
case "$MIXED_REWARD_FINAL_NORMALIZE" in
  true|false) ;;
  *) error "reward.mixed_reward_final_normalize must be true or false" ;;
esac
# mixed_reward_alpha = MV weight; Intuitor/SCE = 1 - alpha (co_intuitor_train.py / single_intuitor_train.py)
MIXED_REWARD_MV_WEIGHT="${MIXED_REWARD_MV_WEIGHT:-$MIXED_REWARD_ALPHA}"
MIXED_REWARD_INTUITOR_WEIGHT="${MIXED_REWARD_INTUITOR_WEIGHT:-$(awk -v a="$MIXED_REWARD_ALPHA" 'BEGIN { printf "%.4f", 1.0 - a }')}"

# Training
NUM_GPUS="${NUM_GPUS:-$(cfg training.num_gpus "2")}"
TRAIN_GPU_LIST="${TRAIN_GPU_LIST:-${CUDA_VISIBLE_DEVICES:-$(_gpu_list "$NUM_GPUS")}}"
if [ "$EVAL_ONLY" != "true" ]; then
  IFS=',' read -r -a _TRAIN_GPU_IDS <<< "$TRAIN_GPU_LIST"
  TRAIN_GPU_COUNT="${#_TRAIN_GPU_IDS[@]}"
  [ "$TRAIN_GPU_COUNT" -eq "$NUM_GPUS" ] \
    || error "TRAIN_GPU_LIST must contain exactly NUM_GPUS entries (got $TRAIN_GPU_COUNT, expected $NUM_GPUS): $TRAIN_GPU_LIST"
fi
MAX_STEPS="${MAX_STEPS:-$(cfg training.max_steps "20")}"
START_STEP="${START_STEP:-$(cfg training.start_step "0")}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$(cfg training.per_device_batch_size "2")}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(cfg training.gradient_accumulation_steps "224")}"
LEARNING_RATE="${LEARNING_RATE:-$(cfg training.learning_rate "3e-6")}"
WARMUP_RATIO="${WARMUP_RATIO:-$(cfg training.warmup_ratio "0.1")}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-$(cfg training.lr_scheduler "cosine")}"
WEIGHT_DECAY="${WEIGHT_DECAY:-$(cfg training.weight_decay "0.0")}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-$(cfg training.max_grad_norm "1.0")}"
TORCH_DTYPE="${TORCH_DTYPE:-$(cfg training.torch_dtype "bfloat16")}"
SEED="${SEED:-$(cfg training.seed "7")}"
BF16="${BF16:-$(cfg training.bf16 "true")}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-$(cfg training.gradient_checkpointing "true")}"
NUM_ITERATIONS="${NUM_ITERATIONS:-$(cfg training.num_iterations "1")}"
EPSILON="${EPSILON:-$(cfg training.epsilon "0.2")}"
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-$(cfg training.accelerator_config "zero3")}"
ACCELERATE_CONFIG="recipes/accelerate_configs/${ACCELERATOR_CONFIG}.yaml"
PERSIST_OPTIMIZER_STATE="${PERSIST_OPTIMIZER_STATE:-$(cfg training.persist_optimizer_state "true")}"
REQUIRE_OPTIMIZER_STATE="${REQUIRE_OPTIMIZER_STATE:-$(cfg training.require_optimizer_state "true")}"
KEEP_ALL_TRAINING_STATES="${KEEP_ALL_TRAINING_STATES:-$(cfg output.keep_all_training_states "false")}"
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
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-$(eval_cfg eval.max_new_tokens "2048")}"
EVAL_LOCAL_DATASETS_ROOT="${EVAL_LOCAL_DATASETS_ROOT:-$(eval_cfg eval.local_datasets_root "../datasets")}"

# Eval – lighteval
LIGHTEVAL_ENABLED="${LIGHTEVAL_ENABLED:-$(eval_cfg eval.lighteval.enabled "false")}"
LIGHTEVAL_TASKS="${LIGHTEVAL_TASKS:-$(eval_cfg eval.lighteval.tasks "math_500,gsm8k")}"
LIGHTEVAL_CUSTOM_TASKS="${LIGHTEVAL_CUSTOM_TASKS:-$(eval_cfg eval.lighteval.custom_tasks "eval.lighteval.custom_tasks.offline_patch")}"
LIGHTEVAL_DATA_PARALLEL_SIZE="${LIGHTEVAL_DATA_PARALLEL_SIZE:-$(eval_cfg eval.lighteval.data_parallel_size "1")}"
LIGHTEVAL_MAX_NEW_TOKENS="${LIGHTEVAL_MAX_NEW_TOKENS:-$(eval_cfg eval.lighteval.max_new_tokens "$EVAL_MAX_NEW_TOKENS")}"
LIGHTEVAL_TEMPERATURE="${LIGHTEVAL_TEMPERATURE:-$(eval_cfg eval.lighteval.temperature "0.0")}"
LIGHTEVAL_TOP_P="${LIGHTEVAL_TOP_P:-$(eval_cfg eval.lighteval.top_p "1.0")}"
LIGHTEVAL_MAX_MODEL_LEN="${LIGHTEVAL_MAX_MODEL_LEN:-$(eval_cfg eval.lighteval.max_model_len "$VLLM_MAX_MODEL_LEN")}"
LIGHTEVAL_GPU_MEM="${LIGHTEVAL_GPU_MEM:-$(eval_cfg eval.lighteval.gpu_memory_utilization "$VLLM_GPU_MEM")}"
LIGHTEVAL_DTYPE="${LIGHTEVAL_DTYPE:-$(eval_cfg eval.lighteval.dtype "$VLLM_DTYPE")}"
LIGHTEVAL_MAX_SAMPLES="${LIGHTEVAL_MAX_SAMPLES:-$(eval_cfg eval.lighteval.max_samples "")}"
LIGHTEVAL_PROMPT_MODE="${LIGHTEVAL_PROMPT_MODE:-$(eval_cfg eval.lighteval.prompt_mode "system")}"
LIGHTEVAL_SYSTEM_PROMPT="${LIGHTEVAL_SYSTEM_PROMPT:-$(eval_cfg eval.lighteval.system_prompt)}"
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
ZEROEVAL_ENABLED="${ZEROEVAL_ENABLED:-$(eval_cfg eval.zeroeval.enabled "false")}"
ZEROEVAL_TASKS="${ZEROEVAL_TASKS:-$(eval_cfg eval.zeroeval.tasks "cruxeval")}"
ZEROEVAL_DIR="${ZEROEVAL_DIR:-$(eval_cfg eval.zeroeval.repo_dir "third_party/ZeroEval")}"
ZEROEVAL_TP_SIZE="${ZEROEVAL_TP_SIZE:-$(eval_cfg eval.zeroeval.tp_size "2")}"
ZEROEVAL_BATCH_SIZE="${ZEROEVAL_BATCH_SIZE:-$(eval_cfg eval.zeroeval.batch_size "100")}"
ZEROEVAL_MAX_TOKENS="${ZEROEVAL_MAX_TOKENS:-$(eval_cfg eval.zeroeval.max_tokens "2048")}"
ZEROEVAL_GPU_MEM="${ZEROEVAL_GPU_MEMORY_UTILIZATION:-$(eval_cfg eval.zeroeval.gpu_memory_utilization "0.95")}"
ZEROEVAL_MAX_SAMPLES="${ZEROEVAL_MAX_SAMPLES:-$(eval_cfg eval.zeroeval.max_samples "800")}"

# Eval – hacrl
HACRL_EVAL_ENABLED="${HACRL_EVAL_ENABLED:-$(eval_cfg eval.hacrl.enabled "false")}"
HACRL_EVAL_TASKS="${HACRL_EVAL_TASKS:-$(eval_cfg eval.hacrl.tasks "math_500,gsm8k")}"
HACRL_EVAL_SCORER="${HACRL_EVAL_SCORER:-$(eval_cfg eval.hacrl.scorer "hacrl_prime_math")}"
HACRL_EVAL_NUM_GENERATIONS="${HACRL_EVAL_NUM_GENERATIONS:-$(eval_cfg eval.hacrl.num_generations "1")}"
HACRL_EVAL_MAX_SAMPLES="${HACRL_EVAL_MAX_SAMPLES:-$(eval_cfg eval.hacrl.max_samples "")}"
HACRL_EVAL_MAX_NEW_TOKENS="${HACRL_EVAL_MAX_NEW_TOKENS:-$(eval_cfg eval.hacrl.max_new_tokens "$EVAL_MAX_NEW_TOKENS")}"
HACRL_EVAL_VLLM_BATCH_SIZE="${HACRL_EVAL_VLLM_BATCH_SIZE:-$(eval_cfg eval.hacrl.vllm_batch_size "16")}"
HACRL_EVAL_TEMPERATURE="${HACRL_EVAL_TEMPERATURE:-$(eval_cfg eval.hacrl.temperature "0.0")}"
HACRL_EVAL_TOP_P="${HACRL_EVAL_TOP_P:-$(eval_cfg eval.hacrl.top_p "1.0")}"
HACRL_EVAL_USE_OFFLINE="${HACRL_EVAL_USE_OFFLINE:-$(eval_cfg eval.hacrl.use_offline "true")}"
HACRL_EVAL_GPU_MEM="${HACRL_EVAL_GPU_MEM:-$(eval_cfg eval.hacrl.gpu_memory_utilization "$VLLM_GPU_MEM")}"
HACRL_EVAL_TP_SIZE="${HACRL_EVAL_TP_SIZE:-$(eval_cfg eval.hacrl.tensor_parallel_size "$VLLM_TP_SIZE")}"
HACRL_EVAL_DTYPE="${HACRL_EVAL_DTYPE:-$(eval_cfg eval.hacrl.dtype "$VLLM_DTYPE")}"
HACRL_EVAL_MAX_MODEL_LEN="${HACRL_EVAL_MAX_MODEL_LEN:-$(eval_cfg eval.hacrl.max_model_len "$VLLM_MAX_MODEL_LEN")}"
HACRL_EVAL_CUDA_DEVICE="${HACRL_EVAL_CUDA_DEVICE:-$(eval_cfg eval.hacrl.cuda_device "")}"

# Eval – ttrl
TTRL_EVAL_ENABLED="${TTRL_EVAL_ENABLED:-$(eval_cfg eval.ttrl.enabled "false")}"
TTRL_EVAL_TASKS="${TTRL_EVAL_TASKS:-$(eval_cfg eval.ttrl.tasks "math_ttt,aime_ttt")}"
TTRL_EVAL_SCORER="${TTRL_EVAL_SCORER:-$(eval_cfg eval.ttrl.scorer "ttrl_math")}"
TTRL_EVAL_NUM_GENERATIONS="${TTRL_EVAL_NUM_GENERATIONS:-$(eval_cfg eval.ttrl.num_generations "1")}"
TTRL_EVAL_MAX_SAMPLES="${TTRL_EVAL_MAX_SAMPLES:-$(eval_cfg eval.ttrl.max_samples "")}"
TTRL_EVAL_PROMPT_MODE="${TTRL_EVAL_PROMPT_MODE:-$(eval_cfg eval.ttrl.prompt_mode "bare")}"
TTRL_EVAL_TEMPERATURE="${TTRL_EVAL_TEMPERATURE:-$(eval_cfg eval.ttrl.temperature "0.0")}"
TTRL_EVAL_TOP_P="${TTRL_EVAL_TOP_P:-$(eval_cfg eval.ttrl.top_p "1.0")}"
TTRL_EVAL_GPU_MEM="${TTRL_EVAL_GPU_MEM:-$(eval_cfg eval.ttrl.gpu_memory_utilization "0.9")}"
TTRL_EVAL_TP_SIZE="${TTRL_EVAL_TP_SIZE:-$(eval_cfg eval.ttrl.tensor_parallel_size "1")}"
TTRL_EVAL_DTYPE="${TTRL_EVAL_DTYPE:-$(eval_cfg eval.ttrl.dtype "bfloat16")}"
TTRL_EVAL_MAX_MODEL_LEN="${TTRL_EVAL_MAX_MODEL_LEN:-$(eval_cfg eval.ttrl.max_model_len "$VLLM_MAX_MODEL_LEN")}"
TTRL_EVAL_DATASET_OVERRIDES="${TTRL_EVAL_DATASET_OVERRIDES:-$(eval_cfg_json eval.ttrl.dataset_overrides)}"

# Wandb
WANDB_API_KEY="${WANDB_API_KEY:-}"
[ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY

# Backward compat for old flat eval config
_eval_compat

if [ "$EVAL_ONLY" = "true" ]; then
  EVAL_ONLY_START_STEP="${EVAL_ONLY_START_STEP:-$(eval_profile_cfg profile.start_step "0")}"
  EVAL_ONLY_END_STEP="${EVAL_ONLY_END_STEP:-$(eval_profile_cfg profile.end_step "$MAX_STEPS")}"
  EVAL_ONLY_STRICT="${EVAL_ONLY_STRICT:-$(eval_profile_cfg profile.strict "true")}"
  EVAL_COMPLETION_MODE="signature"
else
  EVAL_COMPLETION_MODE="${EVAL_COMPLETION_MODE:-legacy}"
fi
export EVAL_COMPLETION_MODE EVAL_FORCE

# ============================================================================
# Validate & Setup
# ============================================================================
[ -z "$POLICY_INIT_PATH" ] && error "models.policy is required in config"
if [ "$EVAL_ONLY" != "true" ]; then
  if [ "$PERSIST_OPTIMIZER_STATE" = "true" ] \
    && ! grep -qE '^distributed_type:[[:space:]]*DEEPSPEED' "$ACCELERATE_CONFIG"; then
    error "training.persist_optimizer_state=true currently requires a DeepSpeed accelerate config: $ACCELERATE_CONFIG"
  fi
  setup_debug "$LATEST_CHECKPOINT_DIR"
  mkdir -p "$LATEST_CHECKPOINT_DIR" "$MODEL_DIR"
  [ "$PERSIST_OPTIMIZER_STATE" = "true" ] && mkdir -p "$TRAINING_STATE_DIR"
  if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
    [[ "$CHECKPOINT_KEEP_LIMIT" =~ ^[1-9][0-9]*$ ]] \
      || error "output.checkpoint_keep_limit must be a positive integer: $CHECKPOINT_KEEP_LIMIT"
  fi
fi

EFF_BATCH=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
log "================================================================"
log "Single-model pipeline (config-driven)"
log "  config:       $CONFIG_FILE"
log "  name:         $(cfg name '<unnamed>')"
log "  policy:       $POLICY_INIT_PATH"
if [ "$EVAL_ONLY" = "true" ]; then
  log "  mode:         eval-only"
  log "  eval_profile: $EVAL_PROFILE_FILE"
  log "  checkpoints:  $MODEL_DIR"
  log "  eval_steps:   $EVAL_ONLY_START_STEP .. $EVAL_ONLY_END_STEP"
  log "  eval_output:  $EVAL_PROFILE_OUTPUT_ROOT"
  log "  eval_force:   $EVAL_FORCE"
  log "  eval_dry_run: $EVAL_ONLY_DRY_RUN"
else
  log "  mode:         train"
fi
log "  reward_model: ${REWARD_MODEL_PATH:-<self>}"
log "  reward_mode:  $REWARD_MODE"
if [ "$REWARD_MODE" = "mixed_reward" ]; then
  log "  mixed_mode:   $MIXED_REWARD_ALPHA_MODE"
  log "  final_norm:   $MIXED_REWARD_FINAL_NORMALIZE"
  case "$MIXED_REWARD_ALPHA_MODE" in
    prism)
      log "  mixed_formula: A_mv + A_int"
      ;;
    ettrl)
      log "  mixed_formula: (R_int / group_mean(R_int)) * A_mv"
      ;;
    *)
      log "  mixed_alpha:  $MIXED_REWARD_ALPHA (MV=α, Intuitor=1-α)"
      log "  mixed_mv_w:   $MIXED_REWARD_MV_WEIGHT"
      log "  mixed_int_w:  $MIXED_REWARD_INTUITOR_WEIGHT"
      ;;
  esac
  if [ "$MIXED_REWARD_ALPHA_MODE" = "reliability_sc_corr" ]; then
    if [ "$MIXED_REWARD_RELIABILITY_USE_SAMPLE_WEIGHT" = "false" ] \
      && [ "$MIXED_REWARD_RELIABILITY_USE_ADVANTAGE_WEIGHT" = "false" ]; then
      log "  reliability:  disabled (fixed alpha050 baseline)"
    else
      log "  reliability:  SC=$MIXED_REWARD_RELIABILITY_SC_WEIGHT Corr=$MIXED_REWARD_RELIABILITY_CORR_WEIGHT"
      log "  rel_weights:  sample=$MIXED_REWARD_RELIABILITY_USE_SAMPLE_WEIGHT advantage=$MIXED_REWARD_RELIABILITY_USE_ADVANTAGE_WEIGHT"
      log "  sample_map:   mu=$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_MU sigma=$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_SIGMA bounds=[$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_LOWER,$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_UPPER]"
      log "  adv_map:      mu=$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_MU sigma=$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_SIGMA bounds=[$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_LOWER,$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_UPPER]"
    fi
  fi
fi
log "  reward_lnorm: $REWARD_LENGTH_NORM"
log "  num_gen:      $NUM_GENERATIONS"
if [ "$EVAL_ONLY" != "true" ]; then
  log "  steps:        $START_STEP .. $((MAX_STEPS))"
  log "  train_gpus:   $NUM_GPUS (CUDA_VISIBLE_DEVICES=$TRAIN_GPU_LIST)"
  log "  grad_clip:    $MAX_GRAD_NORM (DeepSpeed native)"
  log "  optim_state:  persist=$PERSIST_OPTIMIZER_STATE require=$REQUIRE_OPTIMIZER_STATE"
  [ "$PERSIST_OPTIMIZER_STATE" = "true" ] && log "  state_dir:    $TRAINING_STATE_DIR"
fi
log "  eff_batch:    ${NUM_GPUS} × ${PER_DEVICE_TRAIN_BATCH_SIZE} × ${GRADIENT_ACCUMULATION_STEPS} = ${EFF_BATCH}"
log "  keep_ckpts:   $KEEP_ALL_CHECKPOINTS"
[ "$KEEP_ALL_CHECKPOINTS" != "true" ] && log "  ckpt_limit:   $CHECKPOINT_KEEP_LIMIT"
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

get_policy_for_step() {
  local current_step="$1" previous_checkpoint
  if [ "$current_step" -eq 1 ]; then
    echo "$POLICY_INIT_PATH"
    return
  fi

  previous_checkpoint="$MODEL_DIR/checkpoint-$((current_step - 1))"
  [ -d "$previous_checkpoint" ] \
    || error "Missing policy checkpoint for step $current_step: $previous_checkpoint"
  [ -f "$previous_checkpoint/config.json" ] \
    || error "Incomplete policy checkpoint for step $current_step: missing $previous_checkpoint/config.json"
  [ -n "$(find "$previous_checkpoint" -maxdepth 1 -type f -size +0c \
    \( -name '*.safetensors' -o -name '*.safetensors.index.json' \
       -o -name 'pytorch_model*.bin' -o -name 'pytorch_model*.bin.index.json' \) \
    -print -quit)" ] \
    || error "Incomplete policy checkpoint for step $current_step: no model weights in $previous_checkpoint"
  echo "$previous_checkpoint"
}

validate_existing_training_checkpoint() {
  local step="$1" checkpoint_dir="$MODEL_DIR/checkpoint-$1"
  [ -d "$checkpoint_dir" ] || return 0
  checkpoint_pair_is_complete \
    "$MODEL_DIR" "$TRAINING_STATE_DIR" "$step" "$PERSIST_OPTIMIZER_STATE" \
    || error "Incomplete checkpoint at step $step: model weights, runtime manifest, and DeepSpeed state must all belong to the same step."
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

resolve_eval_model_path() {
  local step="$1"
  if [ "$step" -eq 0 ]; then
    echo "$POLICY_INIT_PATH"
  else
    echo "$MODEL_DIR/checkpoint-$step"
  fi
}

is_eval_model_path_ready() {
  local model_path="$1"
  [ -d "$model_path" ] || return 1
  [ -f "$model_path/config.json" ] || return 1
  [ -n "$(find "$model_path" -maxdepth 1 -type f -size +0c \
    \( -name '*.safetensors' -o -name '*.safetensors.index.json' \
       -o -name 'pytorch_model*.bin' -o -name 'pytorch_model*.bin.index.json' \) \
    -print -quit)" ]
}

validate_eval_only_run() {
  [[ "$EVAL_ONLY_START_STEP" =~ ^[0-9]+$ ]] \
    || error "eval-only start step must be a non-negative integer: $EVAL_ONLY_START_STEP"
  [[ "$EVAL_ONLY_END_STEP" =~ ^[0-9]+$ ]] \
    || error "eval-only end step must be a non-negative integer: $EVAL_ONLY_END_STEP"
  [ "$EVAL_ONLY_START_STEP" -le "$EVAL_ONLY_END_STEP" ] \
    || error "eval-only start step must be <= end step"
  [[ "$EVAL_PROFILE_OUTPUT_SUFFIX" =~ ^[A-Za-z0-9._-]+$ ]] \
    || error "profile.output_suffix may only contain letters, numbers, dot, underscore, and hyphen"
  case "$EVAL_ONLY_STRICT" in true|false) ;; *) error "profile.strict must be true or false" ;; esac
  case "$EVAL_ONLY_DRY_RUN" in true|false) ;; *) error "EVAL_ONLY_DRY_RUN must be true or false" ;; esac
  case "$EVAL_FORCE" in true|false) ;; *) error "EVAL_FORCE must be true or false" ;; esac

  if [ "$LIGHTEVAL_ENABLED" != "true" ] \
    && [ "$ZEROEVAL_ENABLED" != "true" ] \
    && [ "$HACRL_EVAL_ENABLED" != "true" ] \
    && [ "$TTRL_EVAL_ENABLED" != "true" ]; then
    error "Eval-only requires at least one enabled evaluation backend"
  fi

  local original_ttrl_root missing=() step model_path
  original_ttrl_root="$(cfg output.ttrl_dir "")"
  if [ -n "$original_ttrl_root" ] && [ "$EVAL_PROFILE_OUTPUT_ROOT" = "$original_ttrl_root" ]; then
    error "Eval-only output root must differ from the original experiment eval root"
  fi

  for step in $(seq "$EVAL_ONLY_START_STEP" "$EVAL_ONLY_END_STEP"); do
    model_path="$(resolve_eval_model_path "$step")"
    is_eval_model_path_ready "$model_path" || missing+=("step $step: $model_path")
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    if [ "$EVAL_ONLY_STRICT" = "true" ]; then
      printf '[ERROR] Missing eval-only model paths:\n' >&2
      printf '  %s\n' "${missing[@]}" >&2
      exit 1
    fi
    log "WARNING: missing eval-only model paths will be skipped:"
    printf '  %s\n' "${missing[@]}" >&2
  fi
}

run_enabled_evaluations() {
  local model_path="$1" step="$2"
  local checkpoint_tag="checkpoint-$step"

  log "--- Step $step: Evaluate ---"

  if [ "$EVAL_ONLY" = "true" ] && [ "$EVAL_ONLY_DRY_RUN" = "true" ]; then
    log "  [dry-run] model=$model_path"
    [ "$LIGHTEVAL_ENABLED" = "true" ] \
      && log "  [dry-run:lighteval] tasks=$LIGHTEVAL_TASKS output=$LIGHTEVAL_OUT_PREFIX/checkpoint-$step"
    [ "$ZEROEVAL_ENABLED" = "true" ] \
      && log "  [dry-run:zeroeval] tasks=$ZEROEVAL_TASKS output=$ZEROEVAL_OUTPUT_ROOT"
    [ "$HACRL_EVAL_ENABLED" = "true" ] \
      && log "  [dry-run:hacrl] tasks=$HACRL_EVAL_TASKS output=$HACRL_EVAL_OUTPUT_ROOT"
    [ "$TTRL_EVAL_ENABLED" = "true" ] \
      && log "  [dry-run:ttrl] tasks=$TTRL_EVAL_TASKS output=$TTRL_EVAL_OUTPUT_ROOT/a_$checkpoint_tag"
    return 0
  fi

  if [ "$LIGHTEVAL_ENABLED" = "true" ]; then
    local lighteval_out="${LIGHTEVAL_OUT_PREFIX}/checkpoint-$step"
    log "  [lighteval] tasks=$LIGHTEVAL_TASKS"
    run_lighteval "$model_path" "$lighteval_out" "$LIGHTEVAL_TASKS" "$LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT" \
      "$LIGHTEVAL_MAX_NEW_TOKENS" "$SEED" "$LIGHTEVAL_CUSTOM_TASKS" "$EVAL_LOCAL_DATASETS_ROOT" \
      "$LIGHTEVAL_DATA_PARALLEL_SIZE" "$LIGHTEVAL_TEMPERATURE" "$LIGHTEVAL_TOP_P" \
      "$LIGHTEVAL_MAX_MODEL_LEN" "$LIGHTEVAL_GPU_MEM" "$LIGHTEVAL_DTYPE" \
      "$LIGHTEVAL_MAX_SAMPLES"
  fi

  if [ "$ZEROEVAL_ENABLED" = "true" ]; then
    log "  [zeroeval] tasks=$ZEROEVAL_TASKS"
    run_zeroeval "$model_path" "a" "$checkpoint_tag" "$ZEROEVAL_OUTPUT_ROOT" \
      "$ZEROEVAL_DIR" "$ZEROEVAL_TP_SIZE" "$ZEROEVAL_BATCH_SIZE" "$ZEROEVAL_MAX_TOKENS" \
      "$ZEROEVAL_GPU_MEM" "$SEED" "$ZEROEVAL_MAX_SAMPLES"
  fi

  if [ "$HACRL_EVAL_ENABLED" = "true" ]; then
    local hacrl_out="${HACRL_EVAL_OUTPUT_ROOT:-outputs/hacrl_eval/$(cfg name 'single')}"
    log "  [hacrl] scorer=$HACRL_EVAL_SCORER tasks=$HACRL_EVAL_TASKS"
    run_hacrl_eval "$model_path" "a" "$checkpoint_tag" "$hacrl_out" \
      "$HACRL_EVAL_TASKS" "$HACRL_EVAL_SCORER" "$SYSTEM_PROMPT" "$SEED" \
      "$HACRL_EVAL_MAX_NEW_TOKENS" "$HACRL_EVAL_VLLM_BATCH_SIZE" \
      "$HACRL_EVAL_NUM_GENERATIONS" "$HACRL_EVAL_MAX_SAMPLES" \
      "$HACRL_EVAL_TEMPERATURE" "$HACRL_EVAL_TOP_P" \
      "$HACRL_EVAL_USE_OFFLINE" "$HACRL_EVAL_GPU_MEM" "$HACRL_EVAL_TP_SIZE" \
      "$HACRL_EVAL_DTYPE" "$HACRL_EVAL_MAX_MODEL_LEN" "$HACRL_EVAL_CUDA_DEVICE"
  fi

  if [ "$TTRL_EVAL_ENABLED" = "true" ]; then
    local ttrl_out="${TTRL_EVAL_OUTPUT_ROOT:-outputs/ttrl_eval/$(cfg name 'single')}"
    log "  [ttrl] scorer=$TTRL_EVAL_SCORER tasks=$TTRL_EVAL_TASKS mode=$TTRL_EVAL_PROMPT_MODE"
    run_ttrl_eval "$model_path" "a" "$checkpoint_tag" "$ttrl_out" \
      "$TTRL_EVAL_TASKS" "$TTRL_EVAL_SCORER" "$SEED" \
      "$EVAL_MAX_NEW_TOKENS" "$TTRL_EVAL_NUM_GENERATIONS" "$TTRL_EVAL_MAX_SAMPLES" \
      "$TTRL_EVAL_TEMPERATURE" "$TTRL_EVAL_TOP_P" \
      "$TTRL_EVAL_PROMPT_MODE" \
      "$TTRL_EVAL_GPU_MEM" "$TTRL_EVAL_TP_SIZE" "$TTRL_EVAL_DTYPE" \
      "$TTRL_EVAL_MAX_MODEL_LEN" \
      "$TTRL_EVAL_DATASET_OVERRIDES"
  fi
}

# ============================================================================
# Main Loop
# ============================================================================

if [ "$EVAL_ONLY" = "true" ]; then
  validate_eval_only_run
  for step in $(seq "$EVAL_ONLY_START_STEP" "$EVAL_ONLY_END_STEP"); do
    EVAL_MODEL_PATH="$(resolve_eval_model_path "$step")"
    if ! is_eval_model_path_ready "$EVAL_MODEL_PATH"; then
      log "WARNING: skipping missing or incomplete step $step: $EVAL_MODEL_PATH"
      continue
    fi
    run_enabled_evaluations "$EVAL_MODEL_PATH" "$step"
  done
  log "Eval-only done. Results: $EVAL_PROFILE_OUTPUT_ROOT"
  exit 0
fi

# ---- Resume discovery and initial evaluation ----
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_LIST"
LATEST_COMPLETE_STEP="$(latest_complete_checkpoint_step \
  "$MODEL_DIR" "$TRAINING_STATE_DIR" "$PERSIST_OPTIMIZER_STATE")"

if [ "$LATEST_COMPLETE_STEP" -gt 0 ] && [ "$LATEST_COMPLETE_STEP" -ge "$START_STEP" ]; then
  START_STEP=$((LATEST_COMPLETE_STEP + 1))
  log "Auto-resume: latest complete checkpoint is step $LATEST_COMPLETE_STEP; continuing at step $START_STEP."
fi

if [ "$LATEST_COMPLETE_STEP" -eq 0 ] \
  && [ "$START_STEP" -le 1 ] \
  && [ "$REQUIRE_OPTIMIZER_STATE" != "false" ]; then
  HIGHEST_EXISTING_STEP=0
  for EXISTING_PATH in "$MODEL_DIR"/checkpoint-*; do
    [ -d "$EXISTING_PATH" ] || continue
    EXISTING_NAME="${EXISTING_PATH##*/}"
    [[ "$EXISTING_NAME" =~ ^checkpoint-([0-9]+)$ ]] || continue
    EXISTING_STEP="${BASH_REMATCH[1]}"
    [ "$EXISTING_STEP" -gt "$HIGHEST_EXISTING_STEP" ] && HIGHEST_EXISTING_STEP="$EXISTING_STEP"
  done
  if [ "$HIGHEST_EXISTING_STEP" -gt 1 ]; then
    error "Found checkpoint directories through step $HIGHEST_EXISTING_STEP, but none is a complete model/DeepSpeed-state pair. Refusing to delete historical output automatically."
  fi
fi

RESUME_EVAL_STEP=0
if [ "$START_STEP" -gt 1 ]; then
  PREVIOUS_STEP=$((START_STEP - 1))
  if checkpoint_pair_is_complete \
    "$MODEL_DIR" "$TRAINING_STATE_DIR" "$PREVIOUS_STEP" "$PERSIST_OPTIMIZER_STATE"; then
    RESUME_EVAL_STEP="$PREVIOUS_STEP"
  elif [ "$REQUIRE_OPTIMIZER_STATE" != "false" ]; then
    error "Cannot start at step $START_STEP: checkpoint-$PREVIOUS_STEP is not a complete model/DeepSpeed-state pair."
  else
    log "WARNING: checkpoint-$PREVIOUS_STEP is not a complete pair; allowing the configured inexact warm start."
  fi
fi

# A failed save may leave model/state directories for the step that must be retried.
# Remove only the incomplete tail; the last complete pair remains available.
for PARTIAL_PATH in "$MODEL_DIR"/checkpoint-* "$TRAINING_STATE_DIR"/checkpoint-*; do
  [ -d "$PARTIAL_PATH" ] || continue
  PARTIAL_NAME="${PARTIAL_PATH##*/}"
  [[ "$PARTIAL_NAME" =~ ^checkpoint-([0-9]+)$ ]] || continue
  PARTIAL_STEP="${BASH_REMATCH[1]}"
  [ "$PARTIAL_STEP" -ge "$START_STEP" ] || continue
  if ! checkpoint_pair_is_complete \
    "$MODEL_DIR" "$TRAINING_STATE_DIR" "$PARTIAL_STEP" "$PERSIST_OPTIMIZER_STATE"; then
    log "Removing incomplete checkpoint tail at step $PARTIAL_STEP before retry."
    rm -rf "$MODEL_DIR/checkpoint-$PARTIAL_STEP" "$TRAINING_STATE_DIR/checkpoint-$PARTIAL_STEP"
  fi
done

if [ "$START_STEP" = "1" ]; then
  run_enabled_evaluations "$POLICY_INIT_PATH" "0"
elif [ "$RESUME_EVAL_STEP" -gt 0 ]; then
  run_enabled_evaluations "$MODEL_DIR/checkpoint-$RESUME_EVAL_STEP" "$RESUME_EVAL_STEP"
  if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
    prune_checkpoint_pairs \
      "$MODEL_DIR" "$TRAINING_STATE_DIR" "$PERSIST_OPTIMIZER_STATE" \
      "$CHECKPOINT_KEEP_LIMIT" "$KEEP_ALL_TRAINING_STATES"
  fi
fi


for step in $(seq "$START_STEP" $((MAX_STEPS))); do
  log "======================== Step $step ========================"
  CKPT_DIR="$MODEL_DIR/checkpoint-$step"
  validate_existing_training_checkpoint "$step"
  POLICY_PATH=$(get_policy_for_step "$step")
  log "Policy: $POLICY_PATH"

  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_LIST"

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
  if [ -d "$CKPT_DIR" ]; then
    log "Complete checkpoint exists at step $step, skipping training."
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
      --max_grad_norm "$MAX_GRAD_NORM"
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
      --torch_dtype "$TORCH_DTYPE"
      --trust_remote_code
      --objective_mode "$OBJECTIVE_MODE"
      --is_length_norm "$IS_LENGTH_NORM"
    )
    if [ "$PERSIST_OPTIMIZER_STATE" = "true" ]; then
      CURRENT_STATE_TAG="checkpoint-$step"
      TRAIN_ARGS+=(--training_state_dir "$TRAINING_STATE_DIR")
      TRAIN_ARGS+=(--training_state_tag "$CURRENT_STATE_TAG")
      # The shell prunes only after validating the new model/state/manifest pair.
      TRAIN_ARGS+=(--keep_all_training_states)

      if [ "$step" -gt 1 ]; then
        PREVIOUS_STATE_TAG="checkpoint-$((step - 1))"
        if [ -d "$TRAINING_STATE_DIR/$PREVIOUS_STATE_TAG" ]; then
          TRAIN_ARGS+=(--resume_training_state_tag "$PREVIOUS_STATE_TAG")
        elif [ "$REQUIRE_OPTIMIZER_STATE" = "false" ] && [ "$step" -eq "$START_STEP" ]; then
          log "WARNING: optimizer state $PREVIOUS_STATE_TAG is missing; AdamW will restart and scheduler position will be reconstructed."
        else
          error "Missing required DeepSpeed optimizer state: $TRAINING_STATE_DIR/$PREVIOUS_STATE_TAG. training.require_optimizer_state=false only permits one inexact warm start at START_STEP."
        fi
      fi
    fi
    [ -n "$TRAIN_SYSTEM_PROMPT" ] && TRAIN_ARGS+=(--system_prompt "$TRAIN_SYSTEM_PROMPT")
    [ "$MASK_TRUNCATED_COMPLETIONS" = "true" ] && TRAIN_ARGS+=(--mask_truncated_completions)
    [ "$BF16" = "true" ] && TRAIN_ARGS+=(--bf16) || TRAIN_ARGS+=(--no_bf16)
    [ "$GRADIENT_CHECKPOINTING" = "true" ] && TRAIN_ARGS+=(--gradient_checkpointing) || TRAIN_ARGS+=(--no_gradient_checkpointing)
    # mixed_reward alpha
    if [ "$REWARD_MODE" = "mixed_reward" ]; then
      TRAIN_ARGS+=(--mixed_reward_alpha "$MIXED_REWARD_ALPHA")
      TRAIN_ARGS+=(--mixed_reward_alpha_mode "$MIXED_REWARD_ALPHA_MODE")
      [ "$MIXED_REWARD_FINAL_NORMALIZE" = "true" ] \
        && TRAIN_ARGS+=(--mixed_reward_final_normalize) \
        || TRAIN_ARGS+=(--no_mixed_reward_final_normalize)
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_mu "$MIXED_REWARD_ALPHA_ZSCORE_MU")
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_sigma "$MIXED_REWARD_ALPHA_ZSCORE_SIGMA")
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_lower "$MIXED_REWARD_ALPHA_ZSCORE_LOWER")
      TRAIN_ARGS+=(--mixed_reward_alpha_zscore_upper "$MIXED_REWARD_ALPHA_ZSCORE_UPPER")
      TRAIN_ARGS+=(--mixed_reward_corr_weight_mode "$MIXED_REWARD_CORR_WEIGHT_MODE")
      TRAIN_ARGS+=(--mixed_reward_corr_weight_threshold "$MIXED_REWARD_CORR_WEIGHT_THRESHOLD")
      TRAIN_ARGS+=(--mixed_reward_corr_weight_exponent "$MIXED_REWARD_CORR_WEIGHT_EXPONENT")
      TRAIN_ARGS+=(--mixed_reward_reliability_sc_weight "$MIXED_REWARD_RELIABILITY_SC_WEIGHT")
      TRAIN_ARGS+=(--mixed_reward_reliability_corr_weight "$MIXED_REWARD_RELIABILITY_CORR_WEIGHT")
      [ "$MIXED_REWARD_RELIABILITY_USE_SAMPLE_WEIGHT" = "true" ] \
        && TRAIN_ARGS+=(--mixed_reward_reliability_use_sample_weight) \
        || TRAIN_ARGS+=(--no_mixed_reward_reliability_use_sample_weight)
      [ "$MIXED_REWARD_RELIABILITY_USE_ADVANTAGE_WEIGHT" = "true" ] \
        && TRAIN_ARGS+=(--mixed_reward_reliability_use_advantage_weight) \
        || TRAIN_ARGS+=(--no_mixed_reward_reliability_use_advantage_weight)
      TRAIN_ARGS+=(--mixed_reward_reliability_sample_weight_mu "$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_MU")
      TRAIN_ARGS+=(--mixed_reward_reliability_sample_weight_sigma "$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_SIGMA")
      TRAIN_ARGS+=(--mixed_reward_reliability_sample_weight_lower "$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_LOWER")
      TRAIN_ARGS+=(--mixed_reward_reliability_sample_weight_upper "$MIXED_REWARD_RELIABILITY_SAMPLE_WEIGHT_UPPER")
      TRAIN_ARGS+=(--mixed_reward_reliability_advantage_weight_mu "$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_MU")
      TRAIN_ARGS+=(--mixed_reward_reliability_advantage_weight_sigma "$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_SIGMA")
      TRAIN_ARGS+=(--mixed_reward_reliability_advantage_weight_lower "$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_LOWER")
      TRAIN_ARGS+=(--mixed_reward_reliability_advantage_weight_upper "$MIXED_REWARD_RELIABILITY_ADVANTAGE_WEIGHT_UPPER")
    fi

    # Self-update epsilon overrides
    [ -n "$SELF_UPDATE_EPSILON_LOW" ] && TRAIN_ARGS+=(--self_update_epsilon_low "$SELF_UPDATE_EPSILON_LOW")
    [ -n "$SELF_UPDATE_EPSILON_HIGH" ] && TRAIN_ARGS+=(--self_update_epsilon_high "$SELF_UPDATE_EPSILON_HIGH")

    # DAPO: dynamic sampling (filter all-correct/all-wrong groups)
    [ "$FILTER_GROUPS" = "true" ] && TRAIN_ARGS+=(--filter_groups)

    _run_accelerate_launch "train" "${CO_MAIN_PROCESS_PORT_TRAIN:-$(_find_free_port)}" \
      src/training/single_intuitor_train.py "${TRAIN_ARGS[@]}"
  fi

  CKPT_PATH="$MODEL_DIR/checkpoint-$step"
  checkpoint_pair_is_complete \
    "$MODEL_DIR" "$TRAINING_STATE_DIR" "$step" "$PERSIST_OPTIMIZER_STATE" \
    || error "Training returned without a complete checkpoint pair at step $step."
  if [ "$PERSIST_OPTIMIZER_STATE" = "true" ] \
    && [ "$KEEP_ALL_CHECKPOINTS" = "true" ] \
    && [ "$KEEP_ALL_TRAINING_STATES" != "true" ]; then
    for STATE_PATH in "$TRAINING_STATE_DIR"/checkpoint-*; do
      [ -d "$STATE_PATH" ] || continue
      [ "${STATE_PATH##*/}" = "checkpoint-$step" ] && continue
      log "Pruning old training state: $STATE_PATH"
      rm -rf "$STATE_PATH"
    done
  fi
  run_enabled_evaluations "$CKPT_PATH" "$step"
  if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
    prune_checkpoint_pairs \
      "$MODEL_DIR" "$TRAINING_STATE_DIR" "$PERSIST_OPTIMIZER_STATE" \
      "$CHECKPOINT_KEEP_LIMIT" "$KEEP_ALL_TRAINING_STATES"
  fi
done

log "Done. Latest checkpoints: $MODEL_DIR"
