#!/usr/bin/env bash
set -euo pipefail

# Config-driven co-learning pipeline runner.
#
# Two models (A and B) are trained iteratively:
#   (1) Both generate via vLLM (parallel)
#   (2) Cross-score + logps
#   (3) Train Model A   (4) Train Model B
#   (5) Eval A + B (parallel, separate GPUs)
#
# Supports:
#   - reward_method: sum_scores / co_wmv / co_mv
#   - learn_config: per-model data selection (use other=0 for self-learning)
#   - per-model batch size / gradient accumulation
#
# Usage:
#   ./scripts/run_co.sh recipes/configs/co_intuitor.json
#   NUM_GPUS=2 ./scripts/run_co.sh recipes/configs/co_intuitor_sl.json

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
INITIAL_MODEL_A_PATH="${INITIAL_MODEL_A_PATH:-$(cfg models.model_a)}"
INITIAL_MODEL_B_PATH="${INITIAL_MODEL_B_PATH:-$(cfg models.model_b)}"

# Output
LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR:-$(cfg output.checkpoint_dir "outputs/co/latest_checkpoint")}"
LIGHTEVAL_OUT_PREFIX="${LIGHTEVAL_OUT_PREFIX:-$(cfg output.lighteval_prefix "outputs/lighteval/co")}"
ZEROEVAL_OUTPUT_ROOT="${ZEROEVAL_OUTPUT_ROOT:-$(cfg output.zeroeval_root "outputs/zeroeval/co")}"
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
NUM_GENERATIONS_PER_MODEL="${NUM_GENERATIONS_PER_MODEL:-$(cfg generation.num_generations_per_model "4")}"
TEMPERATURE="${TEMPERATURE:-$(cfg generation.temperature "0.9")}"
TOP_P="${TOP_P:-$(cfg generation.top_p "1.0")}"
REPETITION_PENALTY="${REPETITION_PENALTY:-$(cfg generation.repetition_penalty "1.0")}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-$(cfg generation.max_completion_length "2048")}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$(cfg generation.max_prompt_length "512")}"

# Scoring
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-$(cfg scoring.scoring_batch_size "4")}"
LOGPS_BATCH_SIZE="${LOGPS_BATCH_SIZE:-$(cfg scoring.logps_batch_size "16")}"

# Reward
REWARD_METHOD="${REWARD_METHOD:-$(cfg reward.method "sum_scores")}"
REWARD_SCORER="${REWARD_SCORER:-$(cfg reward.scorer "hacrl_prime_math")}"
REWARD_WEIGHT_A="${REWARD_WEIGHT_A:-$(cfg reward.weight_a "1.0")}"
REWARD_WEIGHT_B="${REWARD_WEIGHT_B:-$(cfg reward.weight_b "1.0")}"
MIXED_REWARD_ALPHA="${MIXED_REWARD_ALPHA:-$(cfg reward.mixed_reward_alpha "0.5")}"
MIXED_REWARD_ALPHA_MODE="${MIXED_REWARD_ALPHA_MODE:-$(cfg reward.mixed_reward_alpha_mode "fixed")}"
MIXED_REWARD_MV_WEIGHT="${MIXED_REWARD_MV_WEIGHT:-$MIXED_REWARD_ALPHA}"
MIXED_REWARD_INTUITOR_WEIGHT="${MIXED_REWARD_INTUITOR_WEIGHT:-$(awk -v a="$MIXED_REWARD_ALPHA" 'BEGIN { printf "%.4f", 1.0 - a }')}"
SUM_SCORES_MINMAX="${SUM_SCORES_MINMAX:-$(cfg reward.sum_scores_minmax "false")}"
BETA="${BETA:-$(cfg reward.beta "0.0005")}"

# Training
NUM_GPUS="${NUM_GPUS:-$(cfg training.num_gpus "4")}"
MAX_STEPS="${MAX_STEPS:-$(cfg training.max_steps "20")}"
START_STEP="${START_STEP:-$(cfg training.start_step "0")}"
LEARNING_RATE="${LEARNING_RATE:-$(cfg training.learning_rate "3e-6")}"
WARMUP_RATIO="${WARMUP_RATIO:-$(cfg training.warmup_ratio "0.1")}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-$(cfg training.lr_scheduler "cosine")}"
WEIGHT_DECAY="${WEIGHT_DECAY:-$(cfg training.weight_decay "0.0")}"
SEED="${SEED:-$(cfg training.seed "7")}"
NUM_ITERATIONS="${NUM_ITERATIONS:-$(cfg training.num_iterations "1")}"
EPSILON="${EPSILON:-$(cfg training.epsilon "0.2")}"
ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-$(cfg training.accelerator_config "zero3")}"
ACCELERATOR_CONFIG_A="${ACCELERATOR_CONFIG_A:-$(cfg training.accelerator_config_a "$ACCELERATOR_CONFIG")}"
ACCELERATOR_CONFIG_B="${ACCELERATOR_CONFIG_B:-$(cfg training.accelerator_config_b "$ACCELERATOR_CONFIG")}"
ACCELERATE_CONFIG_A="recipes/accelerate_configs/${ACCELERATOR_CONFIG_A}.yaml"
ACCELERATE_CONFIG_B="recipes/accelerate_configs/${ACCELERATOR_CONFIG_B}.yaml"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-$(cfg training.max_grad_norm "0.0")}"
# Allocator tuning for accelerate-launched stages only (score/train).
# vLLM/generate/eval paths always strip this via `env -u`, so expandable_segments is safe here.
ACCELERATE_CUDA_ALLOC_CONF="${ACCELERATE_CUDA_ALLOC_CONF:-$(cfg training.accelerate_cuda_alloc_conf "expandable_segments:True")}"
ACCELERATE_EMPTY_CACHE_ON_BOUNDARY="${ACCELERATE_EMPTY_CACHE_ON_BOUNDARY:-$(cfg training.empty_cache_on_boundary "1")}"
ACCELERATE_EMPTY_CACHE_EVERY_STEP="${ACCELERATE_EMPTY_CACHE_EVERY_STEP:-$(cfg training.empty_cache_every_step "")}"

# GSPO / two-phase settings
OBJECTIVE_MODE="${OBJECTIVE_MODE:-$(cfg training.objective_mode "grpo")}"
CO_UPDATE_MODE="${CO_UPDATE_MODE:-$(cfg training.co_update_mode "two_phase")}"
IS_LENGTH_NORM="${IS_LENGTH_NORM:-$(cfg training.is_length_norm "mean")}"
IS_STRICT_OLD_FOR_PEER="${IS_STRICT_OLD_FOR_PEER:-$(cfg training.is_strict_old_for_peer "false")}"
CAPABILITY_WEIGHT_MODE="${CAPABILITY_WEIGHT_MODE:-$(cfg training.capability_weight_mode "sc_ratio")}"
CAPABILITY_RATIO_SMOOTHING="${CAPABILITY_RATIO_SMOOTHING:-$(cfg training.capability_ratio_smoothing "latest")}"
CAPABILITY_RATIO_K="${CAPABILITY_RATIO_K:-$(cfg training.capability_ratio_k "5")}"
CAPABILITY_NORMALIZE_SCE="${CAPABILITY_NORMALIZE_SCE:-$(cfg training.capability_normalize_sce "false")}"
STRONG_MODEL_PEER_PENALTY="${STRONG_MODEL_PEER_PENALTY:-$(cfg training.strong_model_peer_penalty "0.0")}"

# Self-update GSPO epsilon overrides
SELF_UPDATE_EPSILON_LOW="${SELF_UPDATE_EPSILON_LOW:-$(cfg training.self_update_epsilon_low)}"
SELF_UPDATE_EPSILON_HIGH="${SELF_UPDATE_EPSILON_HIGH:-$(cfg training.self_update_epsilon_high)}"

# Peer-update special clipping
PEER_CLIP_ASYMMETRIC="${PEER_CLIP_ASYMMETRIC:-$(cfg training.peer_special_clip_asymmetric_enabled "false")}"
PEER_CLIP_STEPWISE="${PEER_CLIP_STEPWISE:-$(cfg training.peer_special_clip_stepwise_enabled "false")}"
PEER_CLIP_DELTA="${PEER_CLIP_DELTA:-$(cfg training.peer_special_clip_delta "0.8")}"
PEER_CLIP_DELTA_STEP="${PEER_CLIP_DELTA_STEP:-$(cfg training.peer_special_clip_delta_step "0.4")}"
PEER_IS_REWEIGHT_ENABLED="${PEER_IS_REWEIGHT_ENABLED:-$(cfg training.peer_is_reweight_enabled "false")}"
PEER_IS_REWEIGHT_ALPHA="${PEER_IS_REWEIGHT_ALPHA:-$(cfg training.peer_is_reweight_alpha "0.5")}"
STABLE_PERF="${STABLE_PERF:-$(cfg training.stable_perf "0")}"

# Retokenize pipeline switches
GSPO_SURROGATE_RATIO_ENABLED="${GSPO_SURROGATE_RATIO_ENABLED:-$(cfg training.gspo_surrogate_ratio_enabled "true")}"
PEER_RETOKENIZE_OLD_LOGPS="${PEER_RETOKENIZE_OLD_LOGPS:-$(cfg training.peer_retokenize_old_logps "false")}"
PEER_LOSS_UNIFORM="${PEER_LOSS_UNIFORM:-$(cfg training.peer_loss_uniform "false")}"

# Per-model training params (fall back to shared value, then default)
_SHARED_ACCUM="$(cfg training.gradient_accumulation_steps "112")"
_SHARED_BATCH="$(cfg training.per_device_batch_size "2")"
GRADIENT_ACCUMULATION_STEPS_A="${GRADIENT_ACCUMULATION_STEPS_A:-$(cfg training.gradient_accumulation_steps_a "$_SHARED_ACCUM")}"
GRADIENT_ACCUMULATION_STEPS_B="${GRADIENT_ACCUMULATION_STEPS_B:-$(cfg training.gradient_accumulation_steps_b "$_SHARED_ACCUM")}"
PER_DEVICE_BATCH_SIZE_A="${PER_DEVICE_BATCH_SIZE_A:-$(cfg training.per_device_batch_size_a "$_SHARED_BATCH")}"
PER_DEVICE_BATCH_SIZE_B="${PER_DEVICE_BATCH_SIZE_B:-$(cfg training.per_device_batch_size_b "$_SHARED_BATCH")}"

# Learn config (score-based per-prompt data selection; use other=0 for self-learning)
LEARN_CONFIG="${LEARN_CONFIG:-$(cfg_json training.learn_config)}"
# [TRUTHCHECK] Pipeline-level format/logic checks toggle.
PIPELINE_TRUTHCHECK="${PIPELINE_TRUTHCHECK:-$(cfg debug.pipeline_truthcheck "false")}"
export CO_PIPELINE_TRUTHCHECK="$PIPELINE_TRUTHCHECK"

# vLLM
USE_OFFLINE_VLLM="${USE_OFFLINE_VLLM:-$(cfg vllm.use_offline "true")}"
VLLM_PORT_A="${VLLM_PORT_A:-$(cfg vllm.port_a "8004")}"
VLLM_PORT_B="${VLLM_PORT_B:-$(cfg vllm.port_b "8005")}"
VLLM_GPU_A="${VLLM_GPU_A:-$(cfg vllm.gpu_a "0")}"
VLLM_GPU_B="${VLLM_GPU_B:-$(cfg vllm.gpu_b "1")}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(cfg vllm.tensor_parallel_size "1")}"
VLLM_GPU_MEM="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-$(cfg vllm.gpu_memory_utilization "0.9")}"
VLLM_DTYPE="${VLLM_SERVER_DTYPE:-$(cfg vllm.dtype "bfloat16")}"
VLLM_MAX_MODEL_LEN="${VLLM_SERVER_MAX_MODEL_LEN:-$(cfg vllm.max_model_len "4096")}"
VLLM_MAX_NUM_SEQS="${VLLM_SERVER_MAX_NUM_SEQS:-$(cfg vllm.max_num_seqs "128")}"
VLLM_MAX_BATCHED="${VLLM_SERVER_MAX_NUM_BATCHED_TOKENS:-$(cfg vllm.max_num_batched_tokens "8192")}"
VLLM_CHUNKED="${VLLM_SERVER_CHUNKED_PREFILL:-$(cfg vllm.chunked_prefill "true")}"
VLLM_TIMEOUT="${VLLM_SERVER_START_TIMEOUT:-$(cfg vllm.start_timeout "1800")}"
AUTO_START_SERVER="${AUTO_START_SERVER:-$(cfg vllm.auto_start "false")}"
AUTO_STOP_SERVER="${AUTO_STOP_SERVER:-$(cfg vllm.auto_stop "true")}"

# Eval – parallel GPU assignment (defaults to generation GPUs)
EVAL_GPU_A="${EVAL_GPU_A:-$(cfg eval.gpu_a "$VLLM_GPU_A")}"
EVAL_GPU_B="${EVAL_GPU_B:-$(cfg eval.gpu_b "$VLLM_GPU_B")}"

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

WANDB_API_KEY="${WANDB_API_KEY:-}"
[ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY

# Backward compat for old flat eval config
_eval_compat

# ============================================================================
# Validate & Setup
# ============================================================================
[ -z "$INITIAL_MODEL_A_PATH" ] && error "models.model_a is required in config"
[ -z "$INITIAL_MODEL_B_PATH" ] && error "models.model_b is required in config"
setup_debug "$LATEST_CHECKPOINT_DIR"
mkdir -p "$LATEST_CHECKPOINT_DIR/model_a" "$LATEST_CHECKPOINT_DIR/model_b"
if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
  if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
    prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_a"
  fi
  if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
    prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_b"
  fi
fi

EFF_A=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE_A * GRADIENT_ACCUMULATION_STEPS_A))
EFF_B=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE_B * GRADIENT_ACCUMULATION_STEPS_B))
log "================================================================"
log "Co-learning pipeline (config-driven)"
log "  config:        $CONFIG_FILE"
log "  name:          $(cfg name '<unnamed>')"
log "  model_a:       $INITIAL_MODEL_A_PATH"
log "  model_b:       $INITIAL_MODEL_B_PATH"
log "  reward_method: $REWARD_METHOD (SCE w_a=$REWARD_WEIGHT_A, w_b=$REWARD_WEIGHT_B; mixed w_mv=$MIXED_REWARD_MV_WEIGHT, w_int=$MIXED_REWARD_INTUITOR_WEIGHT)"
[ -n "$LEARN_CONFIG" ] && log "  learn_config:  $LEARN_CONFIG"
log "  gen/model:     $NUM_GENERATIONS_PER_MODEL"
log "  steps:         $START_STEP .. $((MAX_STEPS))"
log "  eff_batch A:   ${NUM_GPUS} × ${PER_DEVICE_BATCH_SIZE_A} × ${GRADIENT_ACCUMULATION_STEPS_A} = ${EFF_A}"
log "  eff_batch B:   ${NUM_GPUS} × ${PER_DEVICE_BATCH_SIZE_B} × ${GRADIENT_ACCUMULATION_STEPS_B} = ${EFF_B}"
log "  accel_cfg A:   $ACCELERATOR_CONFIG_A ($ACCELERATE_CONFIG_A)"
log "  accel_cfg B:   $ACCELERATOR_CONFIG_B ($ACCELERATE_CONFIG_B)"
log "  objective:     $OBJECTIVE_MODE  co_update: $CO_UPDATE_MODE"
log "  is_length_norm:$IS_LENGTH_NORM  cap_weight: $CAPABILITY_WEIGHT_MODE"
log "  eval_gpu A:    $EVAL_GPU_A"
log "  eval_gpu B:    $EVAL_GPU_B"
log "  keep_ckpts:    $KEEP_ALL_CHECKPOINTS"
log "  truthcheck:    $CO_PIPELINE_TRUTHCHECK"
log "================================================================"

# ============================================================================
# Helpers
# ============================================================================
_run_accelerate_launch() {
  local mode="$1"; shift
  local main_port="$1"; shift
  local accel_config="${1:-}"; shift
  local -a env_vars=(env)
  [ -n "$ACCELERATE_CUDA_ALLOC_CONF" ] && env_vars+=(PYTORCH_CUDA_ALLOC_CONF="$ACCELERATE_CUDA_ALLOC_CONF")
  [ -n "$ACCELERATE_EMPTY_CACHE_ON_BOUNDARY" ] && env_vars+=(CO_EMPTY_CACHE_ON_BOUNDARY="$ACCELERATE_EMPTY_CACHE_ON_BOUNDARY")
  [ -n "$ACCELERATE_EMPTY_CACHE_EVERY_STEP" ] && env_vars+=(CO_EMPTY_CACHE_EVERY_STEP="$ACCELERATE_EMPTY_CACHE_EVERY_STEP")

  if [ "$mode" = "train" ] && [ -n "$accel_config" ] && [ -f "$accel_config" ]; then
    "${env_vars[@]}" accelerate launch --config_file "$accel_config" --num_processes "$NUM_GPUS" \
      --main_process_port "$main_port" "$@"
  else
    "${env_vars[@]}" accelerate launch --num_processes "$NUM_GPUS" \
      --main_process_port "$main_port" "$@"
  fi
}

get_latest_checkpoint() {
  local model_name="$1"
  local ckpt_dir="$LATEST_CHECKPOINT_DIR/model_$model_name"
  if [ -d "$ckpt_dir" ]; then
    local latest
    latest=$(find "$ckpt_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
    if [ -n "$latest" ]; then echo "$latest"; return 0; fi
  fi
  [ "$model_name" = "a" ] && echo "$INITIAL_MODEL_A_PATH" || echo "$INITIAL_MODEL_B_PATH"
}

_start_both_vllm() {
  local model_a_path="$1" model_b_path="$2"
  log "Starting vLLM servers for A and B..."
  _vllm_kill_port "$VLLM_PORT_A"
  _vllm_kill_port "$VLLM_PORT_B"
  # Wait until both ports are fully released before starting new servers.
  if ! _vllm_wait_port_free "$VLLM_PORT_A" 30; then
    error "Cannot start vLLM A: port $VLLM_PORT_A still occupied"
  fi
  if ! _vllm_wait_port_free "$VLLM_PORT_B" 30; then
    error "Cannot start vLLM B: port $VLLM_PORT_B still occupied"
  fi

  ( env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES="$VLLM_GPU_A" VLLM_SERVER_DEV_MODE=1 \
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_a_path" --host 0.0.0.0 --port "$VLLM_PORT_A" \
      --enable-sleep-mode --gpu-memory-utilization "$VLLM_GPU_MEM" \
      --dtype "$VLLM_DTYPE" --trust-remote-code \
      --max-model-len "$VLLM_MAX_MODEL_LEN" --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --max-num-batched-tokens "$VLLM_MAX_BATCHED" \
      $( [[ "$VLLM_CHUNKED" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
  ) >"vllm_server_${VLLM_PORT_A}.log" 2>&1 &

  ( env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES="$VLLM_GPU_B" VLLM_SERVER_DEV_MODE=1 \
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_b_path" --host 0.0.0.0 --port "$VLLM_PORT_B" \
      --enable-sleep-mode --gpu-memory-utilization "$VLLM_GPU_MEM" \
      --dtype "$VLLM_DTYPE" --trust-remote-code \
      --max-model-len "$VLLM_MAX_MODEL_LEN" --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --max-num-batched-tokens "$VLLM_MAX_BATCHED" \
      $( [[ "$VLLM_CHUNKED" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
  ) >"vllm_server_${VLLM_PORT_B}.log" 2>&1 &

  _vllm_wait_health "$VLLM_PORT_A" "$VLLM_TIMEOUT" &
  local WAIT_A=$!
  _vllm_wait_health "$VLLM_PORT_B" "$VLLM_TIMEOUT" &
  local WAIT_B=$!
  wait "$WAIT_A"; local OK_A=$?
  wait "$WAIT_B"; local OK_B=$?

  if [ "$OK_A" -ne 0 ] || [ "$OK_B" -ne 0 ]; then
    _vllm_kill_port "$VLLM_PORT_A"; _vllm_kill_port "$VLLM_PORT_B"
    [ "$OK_A" -ne 0 ] && [ "$OK_B" -ne 0 ] && error "Timed out: both vLLM servers"
    [ "$OK_A" -ne 0 ] && error "Timed out: vLLM A (port $VLLM_PORT_A)"
    error "Timed out: vLLM B (port $VLLM_PORT_B)"
  fi
  log "Both vLLM servers are healthy."
}

_stop_both_vllm() {
  _vllm_kill_port "$VLLM_PORT_A"
  _vllm_kill_port "$VLLM_PORT_B"
  log "vLLM servers stopped."
}

_train_model() {
  local which="$1"
  local model_path="$2"
  local current_a="$3"
  local current_b="$4"
  local out_dir="$LATEST_CHECKPOINT_DIR/model_$which"
  local accum_steps batch_size main_port

  local accel_config
  if [ "$which" = "a" ]; then
    accum_steps="$GRADIENT_ACCUMULATION_STEPS_A"
    batch_size="$PER_DEVICE_BATCH_SIZE_A"
    main_port="${CO_MAIN_PROCESS_PORT_A:-$(_find_free_port)}"
    accel_config="$ACCELERATE_CONFIG_A"
  else
    accum_steps="$GRADIENT_ACCUMULATION_STEPS_B"
    batch_size="$PER_DEVICE_BATCH_SIZE_B"
    main_port="${CO_MAIN_PROCESS_PORT_B:-$(_find_free_port)}"
    accel_config="$ACCELERATE_CONFIG_B"
  fi

  TRAIN_ARGS=(
    --train_model "$which"
    --model_name_or_path "$model_path"
    --model_a_path "$current_a"
    --model_b_path "$current_b"
    --tokenizer_a_path "$INITIAL_MODEL_A_PATH"
    --tokenizer_b_path "$INITIAL_MODEL_B_PATH"
    --dataset_key "$DATASET_KEY"
    --output_dir "$out_dir"
    --cache_dir "$LATEST_CHECKPOINT_DIR"
    --max_steps 1
    --total_pipeline_steps "$MAX_STEPS"
    --current_pipeline_step "$step"
    --gradient_accumulation_steps "$accum_steps"
    --learning_rate "$LEARNING_RATE"
    --max_grad_norm "$MAX_GRAD_NORM"
    --per_device_train_batch_size "$batch_size"
    --num_generations_per_model "$NUM_GENERATIONS_PER_MODEL"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --beta "$BETA"
    --epsilon "$EPSILON"
    --seed "$SEED"
    --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --weight_decay "$WEIGHT_DECAY"
    --gradient_checkpointing --bf16
    --logging_steps 1 --logging_strategy steps --logging_first_step
    --save_steps 1 --save_strategy steps
    --overwrite_output_dir
    --torch_dtype bfloat16 --trust_remote_code
    --reward_method "$REWARD_METHOD"
    --reward_weight_a "$REWARD_WEIGHT_A"
    --reward_weight_b "$REWARD_WEIGHT_B"
    --mixed_reward_alpha "$MIXED_REWARD_ALPHA"
    --mixed_reward_alpha_mode "$MIXED_REWARD_ALPHA_MODE"
    --objective_mode "$OBJECTIVE_MODE"
  )
  [ "$SUM_SCORES_MINMAX" = "true" ] && TRAIN_ARGS+=(--sum_scores_minmax)
  TRAIN_ARGS+=(
    --co_update_mode "$CO_UPDATE_MODE"
    --is_length_norm "$IS_LENGTH_NORM"
    --capability_weight_mode "$CAPABILITY_WEIGHT_MODE"
    --capability_ratio_smoothing "$CAPABILITY_RATIO_SMOOTHING"
    --capability_ratio_k "$CAPABILITY_RATIO_K"
    --strong_model_peer_penalty "$STRONG_MODEL_PEER_PENALTY"
  )
  [ -n "$TRAIN_SYSTEM_PROMPT" ] && TRAIN_ARGS+=(--system_prompt "$TRAIN_SYSTEM_PROMPT")
  [ "$IS_STRICT_OLD_FOR_PEER" = "true" ] && TRAIN_ARGS+=(--is_strict_old_for_peer)
  [ "$CAPABILITY_NORMALIZE_SCE" = "true" ] && TRAIN_ARGS+=(--capability_normalize_sce)
  [ -n "$MAX_SAMPLES" ] && TRAIN_ARGS+=(--max_samples "$MAX_SAMPLES")
  [ -n "$LEARN_CONFIG" ] && TRAIN_ARGS+=(--learn_config "$LEARN_CONFIG")

  # Self-update epsilon overrides
  [ -n "$SELF_UPDATE_EPSILON_LOW" ] && TRAIN_ARGS+=(--self_update_epsilon_low "$SELF_UPDATE_EPSILON_LOW")
  [ -n "$SELF_UPDATE_EPSILON_HIGH" ] && TRAIN_ARGS+=(--self_update_epsilon_high "$SELF_UPDATE_EPSILON_HIGH")

  # Peer special clipping
  [ "$PEER_CLIP_ASYMMETRIC" = "true" ] && TRAIN_ARGS+=(--peer_special_clip_asymmetric_enabled)
  [ "$PEER_CLIP_STEPWISE" = "true" ] && TRAIN_ARGS+=(--peer_special_clip_stepwise_enabled)
  TRAIN_ARGS+=(--peer_special_clip_delta "$PEER_CLIP_DELTA")
  TRAIN_ARGS+=(--peer_special_clip_delta_step "$PEER_CLIP_DELTA_STEP")
  [ "$PEER_IS_REWEIGHT_ENABLED" = "true" ] && TRAIN_ARGS+=(--peer_is_reweight_enabled)
  TRAIN_ARGS+=(--peer_is_reweight_alpha "$PEER_IS_REWEIGHT_ALPHA")
  TRAIN_ARGS+=(--stable_perf "$STABLE_PERF")

  # Retokenize pipeline switches
  if [ "$GSPO_SURROGATE_RATIO_ENABLED" = "false" ]; then
    TRAIN_ARGS+=(--no_gspo_surrogate_ratio_enabled)
  fi
  [ "$PEER_RETOKENIZE_OLD_LOGPS" = "true" ] && TRAIN_ARGS+=(--peer_retokenize_old_logps)
  [ "$PEER_LOSS_UNIFORM" = "true" ] && TRAIN_ARGS+=(--peer_loss_uniform)
  if [ "$KEEP_ALL_CHECKPOINTS" != "true" ]; then
    TRAIN_ARGS+=(--save_total_limit 1)
  fi

  _run_accelerate_launch "train" "$main_port" "$accel_config" src/training/co_intuitor_train.py "${TRAIN_ARGS[@]}"
}

resolve_training_prompt_controls() {
  TRAIN_SYSTEM_PROMPT="$SYSTEM_PROMPT"
  TRAIN_INSTRUCTION_SUFFIX_MODE="default"  # default|custom|empty
  TRAIN_INSTRUCTION_SUFFIX_VALUE=""
  TRAIN_ENABLE_THINKING="$DATASET_ENABLE_THINKING"

  case "${DATASET_PROMPT_MODE:-}" in
    system)
      TRAIN_INSTRUCTION_SUFFIX_MODE="empty"
      TRAIN_ENABLE_THINKING="false"
      ;;
    suffix)
      TRAIN_SYSTEM_PROMPT=""
      if [ -n "$DATASET_INSTRUCTION_SUFFIX" ]; then
        TRAIN_INSTRUCTION_SUFFIX_MODE="custom"
        TRAIN_INSTRUCTION_SUFFIX_VALUE="$DATASET_INSTRUCTION_SUFFIX"
      else
        TRAIN_INSTRUCTION_SUFFIX_MODE="default"
      fi
      TRAIN_ENABLE_THINKING="none" #TRAIN_ENABLE_THINKING="$DATASET_ENABLE_THINKING" are both ok
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

_eval_model() {
  local model_tag="$1" ckpt_path="$2" step="$3"
  log "--- Evaluate Model ${model_tag} (step $step) ---"

  # ---- LightEval ----
  if [ "$LIGHTEVAL_ENABLED" = "true" ]; then
    local eval_out="${LIGHTEVAL_OUT_PREFIX}/model-${model_tag}/checkpoint-$step"
    log "  [lighteval] tasks=$LIGHTEVAL_TASKS"
    run_lighteval "$ckpt_path" "$eval_out" "$LIGHTEVAL_TASKS" "$LIGHTEVAL_EFFECTIVE_SYSTEM_PROMPT" \
      "$EVAL_MAX_NEW_TOKENS" "$SEED" "$LIGHTEVAL_CUSTOM_TASKS" "$EVAL_LOCAL_DATASETS_ROOT" \
      "$LIGHTEVAL_DATA_PARALLEL_SIZE"
  fi

  # ---- ZeroEval ----
  if [ "$ZEROEVAL_ENABLED" = "true" ]; then
    log "  [zeroeval] tasks=$ZEROEVAL_TASKS"
    run_zeroeval "$ckpt_path" "$model_tag" "checkpoint-$step" "$ZEROEVAL_OUTPUT_ROOT" \
      "$ZEROEVAL_DIR" "$ZEROEVAL_TP_SIZE" "$ZEROEVAL_BATCH_SIZE" "$ZEROEVAL_MAX_TOKENS" \
      "$ZEROEVAL_GPU_MEM" "$SEED" "$ZEROEVAL_MAX_SAMPLES"
  fi

  # ---- HACRL ----
  if [ "$HACRL_EVAL_ENABLED" = "true" ]; then
    local hacrl_out="${HACRL_EVAL_OUTPUT_ROOT:-outputs/hacrl_eval/$(cfg name 'co')}"
    log "  [hacrl] scorer=$HACRL_EVAL_SCORER tasks=$HACRL_EVAL_TASKS"
    run_hacrl_eval "$ckpt_path" "$model_tag" "checkpoint-$step" "$hacrl_out" \
      "$HACRL_EVAL_TASKS" "$HACRL_EVAL_SCORER" "$SYSTEM_PROMPT" "$SEED" \
      "$EVAL_MAX_NEW_TOKENS" "$HACRL_EVAL_VLLM_BATCH_SIZE" \
      "$HACRL_EVAL_NUM_GENERATIONS" "$HACRL_EVAL_MAX_SAMPLES" \
      "$HACRL_EVAL_TEMPERATURE" "$HACRL_EVAL_TOP_P" \
      "$HACRL_EVAL_USE_OFFLINE" "$HACRL_EVAL_GPU_MEM" "$HACRL_EVAL_TP_SIZE" \
      "$HACRL_EVAL_DTYPE" "$HACRL_EVAL_MAX_MODEL_LEN" "$HACRL_EVAL_CUDA_DEVICE"
  fi

  # ---- TTRL ----
  if [ "$TTRL_EVAL_ENABLED" = "true" ]; then
    local ttrl_out="${TTRL_EVAL_OUTPUT_ROOT:-outputs/ttrl_eval/$(cfg name 'co')}"
    log "  [ttrl] scorer=$TTRL_EVAL_SCORER tasks=$TTRL_EVAL_TASKS mode=$TTRL_EVAL_PROMPT_MODE"
    run_ttrl_eval "$ckpt_path" "$model_tag" "checkpoint-$step" "$ttrl_out" \
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
for step in $(seq "$START_STEP" $((MAX_STEPS))); do
  log "=================================================================================="
  log "Step $step: Generate -> Score -> Train A -> Train B -> Eval A+B (parallel)"
  log "=================================================================================="

  CURRENT_A=$(get_latest_checkpoint "a")
  CURRENT_B=$(get_latest_checkpoint "b")
  log "Model A: $CURRENT_A"
  log "Model B: $CURRENT_B"

  export CUDA_VISIBLE_DEVICES="$(_gpu_list "$NUM_GPUS")"

  # ---- Generate (no accelerate, parallel for A and B) ----
  log "--- Step $step.1a: Generate Model A and B (parallel) ---"

  GEN_COMMON_ARGS=(
    --dataset_key "$DATASET_KEY"
    --output_dir "$LATEST_CHECKPOINT_DIR"
    --step "$step"
    --seed "$SEED"
    --num_generations "$NUM_GENERATIONS_PER_MODEL"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --repetition_penalty "$REPETITION_PENALTY"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --problems_per_step "$PROBLEMS_PER_STEP"
    --enable_thinking "$TRAIN_ENABLE_THINKING"
  )
  [ -n "$MAX_SAMPLES" ] && GEN_COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
  [ -n "$TRAIN_SYSTEM_PROMPT" ] && GEN_COMMON_ARGS+=(--system_prompt "$TRAIN_SYSTEM_PROMPT")
  if [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "custom" ]; then
    GEN_COMMON_ARGS+=(--instruction_suffix "$TRAIN_INSTRUCTION_SUFFIX_VALUE")
  elif [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "empty" ]; then
    GEN_COMMON_ARGS+=(--instruction_suffix "")
  fi
  [ "$SHUFFLE_DATASET" = "true" ] && GEN_COMMON_ARGS+=(--shuffle_dataset) || GEN_COMMON_ARGS+=(--no-shuffle_dataset)

  GEN_VLLM_ARGS_A=()
  GEN_VLLM_ARGS_B=()
  if [ "$USE_OFFLINE_VLLM" = "true" ]; then
    GEN_VLLM_ARGS_A=(--use_offline_vllm --vllm_gpu "$VLLM_GPU_A" --tensor_parallel_size "$VLLM_TP_SIZE" --gpu_memory_utilization "$VLLM_GPU_MEM" --vllm_dtype "$VLLM_DTYPE" --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" --vllm_max_num_batched_tokens "$VLLM_MAX_BATCHED")
    GEN_VLLM_ARGS_B=(--use_offline_vllm --vllm_gpu "$VLLM_GPU_B" --tensor_parallel_size "$VLLM_TP_SIZE" --gpu_memory_utilization "$VLLM_GPU_MEM" --vllm_dtype "$VLLM_DTYPE" --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" --vllm_max_num_batched_tokens "$VLLM_MAX_BATCHED")
    [ "$VLLM_CHUNKED" = "true" ] && { GEN_VLLM_ARGS_A+=(--enable_chunked_prefill); GEN_VLLM_ARGS_B+=(--enable_chunked_prefill); } || { GEN_VLLM_ARGS_A+=(--no-enable_chunked_prefill); GEN_VLLM_ARGS_B+=(--no-enable_chunked_prefill); }
  else
    GEN_VLLM_ARGS_A=(--no-use_offline_vllm --vllm_port "$VLLM_PORT_A")
    GEN_VLLM_ARGS_B=(--no-use_offline_vllm --vllm_port "$VLLM_PORT_B")
    [ "$AUTO_START_SERVER" = "true" ] && { GEN_VLLM_ARGS_A+=(--auto_start_server); GEN_VLLM_ARGS_B+=(--auto_start_server); } || { GEN_VLLM_ARGS_A+=(--no-auto_start_server); GEN_VLLM_ARGS_B+=(--no-auto_start_server); }
    [ "$AUTO_STOP_SERVER" = "true" ] && { GEN_VLLM_ARGS_A+=(--auto_stop_server); GEN_VLLM_ARGS_B+=(--auto_stop_server); } || { GEN_VLLM_ARGS_A+=(--no-auto_stop_server); GEN_VLLM_ARGS_B+=(--no-auto_stop_server); }
    _start_both_vllm "$CURRENT_A" "$CURRENT_B"
  fi

  set +e
  env -u PYTORCH_CUDA_ALLOC_CONF python src/pipelines/generate_only.py \
    --model_path "$CURRENT_A" --tokenizer_path "$INITIAL_MODEL_A_PATH" \
    --cache_slot a "${GEN_COMMON_ARGS[@]}" "${GEN_VLLM_ARGS_A[@]}" &
  GEN_PID_A=$!

  env -u PYTORCH_CUDA_ALLOC_CONF python src/pipelines/generate_only.py \
    --model_path "$CURRENT_B" --tokenizer_path "$INITIAL_MODEL_B_PATH" \
    --cache_slot b "${GEN_COMMON_ARGS[@]}" "${GEN_VLLM_ARGS_B[@]}" &
  GEN_PID_B=$!

  wait "$GEN_PID_A"; GEN_RC_A=$?
  wait "$GEN_PID_B"; GEN_RC_B=$?
  set -e

  if [ "$USE_OFFLINE_VLLM" != "true" ]; then
    _stop_both_vllm
  fi
  [ "$GEN_RC_A" -ne 0 ] && error "Generation Model A failed at step $step"
  [ "$GEN_RC_B" -ne 0 ] && error "Generation Model B failed at step $step"
  log "Step $step.1a: Generation completed!"

  # ---- Score and Compute (with accelerate) ----
  log "--- Step $step.1b: Score and Compute ---"
  SCORE_ARGS=(
    --model_a_path "$CURRENT_A"
    --model_b_path "$CURRENT_B"
    --tokenizer_a_path "$INITIAL_MODEL_A_PATH"
    --tokenizer_b_path "$INITIAL_MODEL_B_PATH"
    --dataset_key "$DATASET_KEY"
    --output_dir "$LATEST_CHECKPOINT_DIR"
    --step "$step"
    --seed "$SEED"
    --num_generations_per_model "$NUM_GENERATIONS_PER_MODEL"
    --temperature "$TEMPERATURE"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --scoring_batch_size "$SCORING_BATCH_SIZE"
    --logps_batch_size "$LOGPS_BATCH_SIZE"
    --problems_per_step "$PROBLEMS_PER_STEP"
    --beta "$BETA"
    --num_iterations "$NUM_ITERATIONS"
    --ref_model_a_path "$INITIAL_MODEL_A_PATH"
    --ref_model_b_path "$INITIAL_MODEL_B_PATH"
    --reward_method "$REWARD_METHOD"
    --scorer "$REWARD_SCORER"
    --enable_thinking "$TRAIN_ENABLE_THINKING"
  )
  [ -n "$MAX_SAMPLES" ] && SCORE_ARGS+=(--max_samples "$MAX_SAMPLES")
  [ -n "$TRAIN_SYSTEM_PROMPT" ] && SCORE_ARGS+=(--system_prompt "$TRAIN_SYSTEM_PROMPT")
  if [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "custom" ]; then
    SCORE_ARGS+=(--instruction_suffix "$TRAIN_INSTRUCTION_SUFFIX_VALUE")
  elif [ "$TRAIN_INSTRUCTION_SUFFIX_MODE" = "empty" ]; then
    SCORE_ARGS+=(--instruction_suffix "")
  fi
  [ -n "$MAX_PROMPT_LENGTH" ] && SCORE_ARGS+=(--max_prompt_length "$MAX_PROMPT_LENGTH")
  [ "$SHUFFLE_DATASET" = "true" ] && SCORE_ARGS+=(--shuffle_dataset) || SCORE_ARGS+=(--no-shuffle_dataset)
  if [ "$CO_UPDATE_MODE" = "two_phase" ] && [ "$OBJECTIVE_MODE" = "gspo" ]; then
    SCORE_ARGS+=(--force_compute_old_logps)
  fi
  [ "$PEER_RETOKENIZE_OLD_LOGPS" = "true" ] && SCORE_ARGS+=(--compute_cross_old_logps)

  set +e
  _run_accelerate_launch "score" "${CO_MAIN_PROCESS_PORT:-$(_find_free_port)}" "" \
    src/pipelines/score_and_compute.py "${SCORE_ARGS[@]}"
  GEN_RC=$?
  set -e
  [ "$GEN_RC" -ne 0 ] && error "Scoring/compute failed at step $step"
  log "Step $step.1b: Score and compute completed!"

  export CUDA_VISIBLE_DEVICES="$(_gpu_list "$NUM_GPUS")"

  # ---- Train Model A ----
  log "--- Step $step.2: Train Model A ---"
  MODEL_A_CKPT="$LATEST_CHECKPOINT_DIR/model_a/checkpoint-$step"
  if [ -d "$MODEL_A_CKPT" ]; then
    log "Checkpoint exists for A at step $step, skipping."
  else
    _train_model "a" "$CURRENT_A" "$CURRENT_A" "$CURRENT_B"
    [ $? -ne 0 ] && error "Training Model A failed at step $step"
    log "Step $step.2: Model A training completed!"
  fi
  prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_a"

  # ---- Train Model B (with updated A) ----
  log "--- Step $step.3: Train Model B ---"
  CURRENT_A=$(get_latest_checkpoint "a")
  MODEL_B_CKPT="$LATEST_CHECKPOINT_DIR/model_b/checkpoint-$step"
  if [ -d "$MODEL_B_CKPT" ]; then
    log "Checkpoint exists for B at step $step, skipping."
  else
    _train_model "b" "$CURRENT_B" "$CURRENT_A" "$CURRENT_B"
    [ $? -ne 0 ] && error "Training Model B failed at step $step"
    log "Step $step.3: Model B training completed!"
  fi
  prune_old_checkpoints "$LATEST_CHECKPOINT_DIR/model_b"

  # ---- Evaluate Model A & B (parallel, separate GPUs) ----
  log "--- Step $step.4: Evaluate Model A and B (parallel) ---"
  EVAL_A_PATH=$(get_latest_checkpoint "a")
  EVAL_B_PATH=$(get_latest_checkpoint "b")
  log "Eval A: $EVAL_A_PATH  (GPU $EVAL_GPU_A)"
  log "Eval B: $EVAL_B_PATH  (GPU $EVAL_GPU_B)"

  set +e
  ( set -e; export CUDA_VISIBLE_DEVICES="$EVAL_GPU_A"; _eval_model "a" "$EVAL_A_PATH" "$step" ) &
  EVAL_PID_A=$!
  ( set -e; export CUDA_VISIBLE_DEVICES="$EVAL_GPU_B"; _eval_model "b" "$EVAL_B_PATH" "$step" ) &
  EVAL_PID_B=$!

  wait "$EVAL_PID_A"; EVAL_RC_A=$?
  wait "$EVAL_PID_B"; EVAL_RC_B=$?
  set -e

  if [ "$EVAL_RC_A" -ne 0 ] || [ "$EVAL_RC_B" -ne 0 ]; then
    [ "$EVAL_RC_A" -ne 0 ] && log "ERROR: Evaluation Model A failed at step $step"
    [ "$EVAL_RC_B" -ne 0 ] && log "ERROR: Evaluation Model B failed at step $step"
    error "Evaluation failed at step $step (A=$EVAL_RC_A, B=$EVAL_RC_B)"
  fi
  log "Step $step.4: Evaluation of both models completed!"

  log "Step $step completed successfully!"
done

log "================================================================"
log "Co-learning pipeline completed!"
log "  Model A: $LATEST_CHECKPOINT_DIR/model_a"
log "  Model B: $LATEST_CHECKPOINT_DIR/model_b"
log "================================================================"
