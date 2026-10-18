#!/usr/bin/env bash
# Shared library for config-driven pipeline scripts.
# Source this file from run_single.sh / run_cross.sh / run_co.sh.
# Do NOT execute directly.

if ((BASH_VERSINFO[0] < 4)); then
  echo "[ERROR] Bash 4+ required (have: ${BASH_VERSION})" >&2
  exit 1
fi

# ============================================================================
# Environment (silence noisy libraries)
# ============================================================================
export DS_VERBOSE=0
export DS_LOG_LEVEL=WARNING
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export HF_DATASETS_VERBOSITY=error
export ACCELERATE_LOG_LEVEL=error
export TORCH_DISTRIBUTED_DEBUG=OFF
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune" 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
# Prefer the active conda toolchain/runtime libs so DeepSpeed JIT ops
# (e.g. cpu_adam.so) don't accidentally bind to an older system libstdc++.
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Set PYTHONPATH to include co-learning directory for utils imports
CO_LEARNING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${CO_LEARNING_ROOT}:${PYTHONPATH:-}"

export CO_ASSERT_INVARIANTS="${CO_ASSERT_INVARIANTS:-1}"
export CO_DEBUG_TRAIN_BOUNDARY="${CO_DEBUG_TRAIN_BOUNDARY:-1}"
# [TRUTHCHECK] Unified pipeline data-format checks (easy to disable/remove).
export CO_PIPELINE_TRUTHCHECK="${CO_PIPELINE_TRUTHCHECK:-0}"

# ============================================================================
# Logging
# ============================================================================
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ============================================================================
# JSON Config Loading
#
# Uses a bash associative array populated by a single Python call.
# Access values with:  cfg "section.key" "default_value"
# Override any value by exporting the corresponding env var before sourcing.
#
# Placeholder syntax inside JSON string values:
#   ${name}             -> another config path
#   ${vars.output_root} -> another nested config path
#   ${HOME}             -> environment variable (fallback if no config path exists)
#   ${env:HOME}         -> explicit environment variable
# ============================================================================
declare -A _CFG=()
_CFG_FILE=""
declare -A _EVAL_PROFILE_CFG=()
_EVAL_PROFILE_FILE=""

_load_config() {
  local config_file="$1"
  if [ ! -f "$config_file" ]; then
    error "Config file not found: $config_file"
  fi
  _CFG_FILE="$config_file"
  log "Loading config: $config_file"
  local line key value
  while IFS= read -r line; do
    key="${line%%=*}"
    value="${line#*=}"
    _CFG["$key"]="$value"
  done < <(python3 "$CO_LEARNING_ROOT/scripts/config_resolver.py" flatten "$config_file")
}

cfg() {
  local val="${_CFG[$1]:-}"
  if [ -n "$val" ]; then
    echo "$val"
  else
    echo "${2:-}"
  fi
}

# Read a nested object from config as a JSON string (not flattened).
cfg_json() {
  python3 "$CO_LEARNING_ROOT/scripts/config_resolver.py" get "$_CFG_FILE" "$1"
}

_load_eval_profile() {
  local profile_file="$1"
  if [ ! -f "$profile_file" ]; then
    error "Eval profile not found: $profile_file"
  fi

  python3 - "$profile_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    data = json.load(handle)

if not isinstance(data, dict):
    raise SystemExit("[ERROR] Eval profile must be a JSON object.")

allowed = {"profile", "eval"}
unknown = sorted(set(data) - allowed)
if unknown:
    raise SystemExit(
        "[ERROR] Eval profile may only contain top-level 'profile' and 'eval' objects; "
        f"found forbidden keys: {unknown}"
    )
if not isinstance(data.get("profile"), dict):
    raise SystemExit("[ERROR] Eval profile requires a 'profile' object.")
if not isinstance(data.get("eval"), dict):
    raise SystemExit("[ERROR] Eval profile requires an 'eval' object.")
PY

  _EVAL_PROFILE_FILE="$profile_file"
  _EVAL_PROFILE_CFG=()
  local line key value
  while IFS= read -r line; do
    key="${line%%=*}"
    value="${line#*=}"
    _EVAL_PROFILE_CFG["$key"]="$value"
  done < <(python3 "$CO_LEARNING_ROOT/scripts/config_resolver.py" flatten "$profile_file")
}

eval_profile_cfg() {
  local val="${_EVAL_PROFILE_CFG[$1]:-}"
  if [ -n "$val" ]; then
    echo "$val"
  else
    echo "${2:-}"
  fi
}

# Profiles override eval.* only. The original config remains the sole owner of
# model and training paths, so eval-only checkpoint discovery cannot drift.
eval_cfg() {
  local val="${_EVAL_PROFILE_CFG[$1]:-}"
  if [ -n "$val" ]; then
    echo "$val"
  else
    cfg "$1" "${2:-}"
  fi
}

eval_cfg_json() {
  local val=""
  if [ -n "$_EVAL_PROFILE_FILE" ]; then
    val="$(python3 "$CO_LEARNING_ROOT/scripts/config_resolver.py" get "$_EVAL_PROFILE_FILE" "$1")"
  fi
  if [ -n "$val" ]; then
    echo "$val"
  else
    cfg_json "$1"
  fi
}

# ============================================================================
# Debug Setup
# ============================================================================
setup_debug() {
  local checkpoint_dir="$1"
  export CO_DEBUG_NAN="${CO_DEBUG_NAN:-1}"
  export CO_DEBUG_NAN_DIR="${CO_DEBUG_NAN_DIR:-$checkpoint_dir/debug_nan}"
  export CO_DEBUG_NAN_RAISE="${CO_DEBUG_NAN_RAISE:-1}"
  export CO_DEBUG_DATA="${CO_DEBUG_DATA:-1}"
  export CO_DEBUG_DATA_DIR="${CO_DEBUG_DATA_DIR:-$checkpoint_dir/debug_data}"
  export CO_DEBUG_DATA_N="${CO_DEBUG_DATA_N:-3}"
}

# ============================================================================
# GPU Helpers
# ============================================================================
_gpu_list() {
  local n="$1" list="" i
  for i in $(seq 0 $((n - 1))); do
    [ -z "$list" ] && list="$i" || list="$list,$i"
  done
  echo "$list"
}

# ============================================================================
# Port Helpers
# ============================================================================
# Find a free TCP port.  Uses Python to bind to port 0, read the
# kernel-assigned port, then immediately close the socket.
# This avoids hardcoded defaults that clash when multiple pipelines
# run concurrently on the same node.
_find_free_port() {
  python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

# ============================================================================
# vLLM Helpers
# ============================================================================
_vllm_port_pids() { lsof -Pi ":${1}" -sTCP:LISTEN -t 2>/dev/null || true; }

_vllm_kill_port() {
  local port="$1" pids
  pids="$(_vllm_port_pids "$port")"
  if [ -z "$pids" ]; then return 0; fi
  log "Stopping vLLM on port ${port} (pids: ${pids})"
  # Kill the entire process group for each pid so worker children are also terminated.
  for pid in $pids; do
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" || true
    if [ -n "$pgid" ] && [ "$pgid" != "0" ]; then
      kill -TERM -- -"$pgid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 3
  pids="$(_vllm_port_pids "$port")"
  if [ -n "$pids" ]; then
    log "Port ${port} still occupied after SIGTERM, escalating to SIGKILL (pids: ${pids})"
    for pid in $pids; do
      local pgid
      pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" || true
      if [ -n "$pgid" ] && [ "$pgid" != "0" ]; then
        kill -KILL -- -"$pgid" 2>/dev/null || true
      else
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
    sleep 3
  fi
  # Verify port is actually freed.
  pids="$(_vllm_port_pids "$port")"
  if [ -n "$pids" ]; then
    log "WARNING: port ${port} still has listeners (pids: ${pids}) after SIGKILL. Last resort kill..."
    kill -9 $pids 2>/dev/null || true
    sleep 2
  fi
}

_vllm_wait_health() {
  local port="$1" timeout_s="${2:-600}" start_ts now_ts
  start_ts="$(date +%s)"
  while true; do
    if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then return 0; fi
    now_ts="$(date +%s)"
    if [ $((now_ts - start_ts)) -ge "$timeout_s" ]; then return 1; fi
    sleep 2
  done
}

_vllm_wait_port_free() {
  local port="$1" timeout_s="${2:-30}" pids start_ts now_ts
  start_ts="$(date +%s)"
  while true; do
    pids="$(_vllm_port_pids "$port")"
    if [ -z "$pids" ]; then return 0; fi
    now_ts="$(date +%s)"
    if [ $((now_ts - start_ts)) -ge "$timeout_s" ]; then
      log "WARNING: port $port still occupied after ${timeout_s}s (pids: $pids)"
      return 1
    fi
    sleep 1
  done
}

_start_vllm_server() {
  local model_path="$1" port="$2" gpu="$3"
  local gpu_mem_util="$4" dtype="$5" max_model_len="$6"
  local max_num_seqs="$7" max_num_batched="$8" chunked="$9" timeout="${10}"
  log "Starting vLLM: port=$port gpu=$gpu model=$model_path"
  _vllm_kill_port "$port"
  # Wait until the port is fully released before starting a new server.
  if ! _vllm_wait_port_free "$port" 30; then
    error "Cannot start vLLM: port $port is still occupied after killing previous server"
  fi
  ( env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES="$gpu" VLLM_SERVER_DEV_MODE=1 \
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" --host 0.0.0.0 --port "$port" \
      --enable-sleep-mode \
      --gpu-memory-utilization "$gpu_mem_util" \
      --dtype "$dtype" --trust-remote-code \
      --max-model-len "$max_model_len" \
      --max-num-seqs "$max_num_seqs" \
      --max-num-batched-tokens "$max_num_batched" \
      $( [[ "$chunked" == "true" ]] && echo "--enable-chunked-prefill" || echo "--disable-chunked-prefill" ) \
  ) >"vllm_server_${port}.log" 2>&1 &
  if ! _vllm_wait_health "$port" "$timeout"; then
    _vllm_kill_port "$port"
    error "Timed out waiting for vLLM on port $port"
  fi
  # /health returns 200 as soon as the HTTP server starts, BEFORE model weights are loaded.
  # With --enable-sleep-mode the model loading is async; poll /v1/models until the model
  # is registered, which guarantees it is ready to serve generation requests.
  local _ts _now
  _ts="$(date +%s)"
  log "Waiting for vLLM model to finish loading on port $port..."
  while true; do
    if curl -fsS "http://localhost:${port}/v1/models" 2>/dev/null | grep -q '"id"'; then
      break
    fi
    _now="$(date +%s)"
    if [ $((_now - _ts)) -ge "$timeout" ]; then
      _vllm_kill_port "$port"
      error "Timed out waiting for vLLM model to load on port $port"
    fi
    sleep 3
  done
  log "vLLM on port $port is healthy."
}

# ============================================================================
# Checkpoint Helpers
# ============================================================================
prune_old_checkpoints() {
  local model_dir="$1"
  if [ -z "$model_dir" ] || [ ! -d "$model_dir" ]; then return 0; fi
  local latest
  latest=$(ls -td "$model_dir"/checkpoint-* 2>/dev/null | head -n 1 || true)
  if [ -z "$latest" ]; then return 0; fi
  for ckpt in "$model_dir"/checkpoint-*; do
    [ ! -d "$ckpt" ] && continue
    if [ "$ckpt" != "$latest" ]; then
      log "Pruning old checkpoint: $ckpt"
      rm -rf "$ckpt" || true
    fi
  done
}

checkpoint_pair_is_complete() {
  local model_dir="$1" state_dir="$2" step="$3" require_state="$4"
  local checkpoint_dir="$model_dir/checkpoint-$step"
  local manifest="$checkpoint_dir/runtime_manifest.json"

  [ -d "$checkpoint_dir" ] || return 1
  [ -f "$checkpoint_dir/config.json" ] || return 1
  [ -f "$manifest" ] || return 1
  [ -n "$(find "$checkpoint_dir" -maxdepth 1 -type f -size +0c \
    \( -name '*.safetensors' -o -name '*.safetensors.index.json' \
       -o -name 'pytorch_model*.bin' -o -name 'pytorch_model*.bin.index.json' \) \
    -print -quit)" ] || return 1

  python3 - "$manifest" "$step" "$require_state" <<'PY' >/dev/null 2>&1 || return 1
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
step = int(sys.argv[2])
if int(manifest.get("pipeline_step", -1)) != step:
    raise SystemExit(1)
if sys.argv[3] == "true":
    if manifest.get("optimizer_state_persisted") is not True:
        raise SystemExit(1)
    if manifest.get("saved_training_state_tag") != f"checkpoint-{step}":
        raise SystemExit(1)
PY

  if [ "$require_state" = "true" ]; then
    local state_path="$state_dir/checkpoint-$step"
    [ -d "$state_path" ] || return 1
    [ -n "$(find "$state_path" -type f -size +0c -print -quit)" ] || return 1
  fi
}

latest_complete_checkpoint_step() {
  local model_dir="$1" state_dir="$2" require_state="$3"
  local checkpoint name step latest=0

  for checkpoint in "$model_dir"/checkpoint-*; do
    [ -d "$checkpoint" ] || continue
    name="${checkpoint##*/}"
    [[ "$name" =~ ^checkpoint-([0-9]+)$ ]] || continue
    step="${BASH_REMATCH[1]}"
    if [ "$step" -gt "$latest" ] \
      && checkpoint_pair_is_complete "$model_dir" "$state_dir" "$step" "$require_state"; then
      latest="$step"
    fi
  done
  printf '%s\n' "$latest"
}

prune_checkpoint_pairs() {
  local model_dir="$1" state_dir="$2" require_state="$3" keep_limit="$4" keep_all_states="$5"
  local checkpoint name step index
  local -a complete_steps=()
  local -A keep_steps=()

  [[ "$keep_limit" =~ ^[1-9][0-9]*$ ]] \
    || error "checkpoint keep limit must be a positive integer: $keep_limit"

  while IFS= read -r step; do
    [ -n "$step" ] && complete_steps+=("$step")
  done < <(
    for checkpoint in "$model_dir"/checkpoint-*; do
      [ -d "$checkpoint" ] || continue
      name="${checkpoint##*/}"
      [[ "$name" =~ ^checkpoint-([0-9]+)$ ]] || continue
      step="${BASH_REMATCH[1]}"
      checkpoint_pair_is_complete "$model_dir" "$state_dir" "$step" "$require_state" \
        && printf '%s\n' "$step"
    done | sort -nr
  )

  [ "${#complete_steps[@]}" -gt 0 ] || return 0
  for ((index = 0; index < ${#complete_steps[@]} && index < keep_limit; index++)); do
    keep_steps["${complete_steps[$index]}"]=1
  done

  for checkpoint in "$model_dir"/checkpoint-*; do
    [ -d "$checkpoint" ] || continue
    name="${checkpoint##*/}"
    [[ "$name" =~ ^checkpoint-([0-9]+)$ ]] || continue
    step="${BASH_REMATCH[1]}"
    if [ -z "${keep_steps[$step]:-}" ]; then
      log "Pruning model checkpoint: $checkpoint"
      rm -rf "$checkpoint"
    fi
  done

  if [ "$require_state" = "true" ] && [ "$keep_all_states" != "true" ]; then
    for checkpoint in "$state_dir"/checkpoint-*; do
      [ -d "$checkpoint" ] || continue
      name="${checkpoint##*/}"
      [[ "$name" =~ ^checkpoint-([0-9]+)$ ]] || continue
      step="${BASH_REMATCH[1]}"
      if [ -z "${keep_steps[$step]:-}" ]; then
        log "Pruning training state: $checkpoint"
        rm -rf "$checkpoint"
      fi
    done
  fi
}

# ============================================================================
# Evaluation Config Helpers
# ============================================================================

# Backward compatibility: old configs use flat eval.backend / eval.tasks.
# New configs use eval.lighteval.enabled / eval.zeroeval.enabled / eval.hacrl.enabled.
# Call this AFTER reading eval config vars to patch old-style configs.
_eval_compat() {
  local old_backend
  old_backend="$(cfg eval.backend "")"
  if [ -z "$old_backend" ]; then return 0; fi

  # Old config detected – map to new-style enabled flags if not explicitly set.
  if [ "$(cfg eval.lighteval.enabled "")" = "" ]; then
    if [ "$old_backend" = "lighteval" ] || [ "$old_backend" = "both" ]; then
      LIGHTEVAL_ENABLED="true"
    fi
    # Old configs put lighteval tasks in eval.tasks (flat).
    local old_tasks
    old_tasks="$(cfg eval.tasks "")"
    [ -n "$old_tasks" ] && LIGHTEVAL_TASKS="$old_tasks"
    # Old flat keys for lighteval.
    local v
    v="$(cfg eval.custom_tasks "")";           [ -n "$v" ] && LIGHTEVAL_CUSTOM_TASKS="$v"
    v="$(cfg eval.lighteval_data_parallel_size "")"; [ -n "$v" ] && LIGHTEVAL_DATA_PARALLEL_SIZE="$v"
    v="$(cfg eval.local_datasets_root "")";    [ -n "$v" ] && EVAL_LOCAL_DATASETS_ROOT="$v"
  fi

  if [ "$(cfg eval.zeroeval.enabled "")" = "" ]; then
    if [ "$old_backend" = "zeroeval" ] || [ "$old_backend" = "both" ]; then
      ZEROEVAL_ENABLED="true"
    fi
    local v
    v="$(cfg eval.zeroeval_repo_dir "")";              [ -n "$v" ] && ZEROEVAL_DIR="$v"
    v="$(cfg eval.zeroeval_tp_size "")";                [ -n "$v" ] && ZEROEVAL_TP_SIZE="$v"
    v="$(cfg eval.zeroeval_batch_size "")";             [ -n "$v" ] && ZEROEVAL_BATCH_SIZE="$v"
    v="$(cfg eval.zeroeval_max_tokens "")";             [ -n "$v" ] && ZEROEVAL_MAX_TOKENS="$v"
    v="$(cfg eval.zeroeval_gpu_memory_utilization "")"; [ -n "$v" ] && ZEROEVAL_GPU_MEM="$v"
    v="$(cfg eval.zeroeval_max_samples "")";            [ -n "$v" ] && ZEROEVAL_MAX_SAMPLES="$v"
  fi
}

# ============================================================================
# Evaluation Helpers
# ============================================================================
_eval_signature() {
  printf '%s\n' "$@" | sha256sum | awk '{print substr($1, 1, 16)}'
}

_eval_done_marker() {
  local output_root="$1" backend="$2"
  shift 2
  if [ "${EVAL_COMPLETION_MODE:-legacy}" != "signature" ]; then
    echo ""
    return 0
  fi
  local signature
  signature="$(_eval_signature "$backend" "$@")"
  echo "$output_root/.eval_markers/${backend}-${signature}.done"
}

_eval_should_skip() {
  local marker="$1" description="$2"
  if [ "${EVAL_FORCE:-false}" = "true" ]; then
    return 1
  fi
  if [ -n "$marker" ] && [ -f "$marker" ]; then
    log "$description already done for this exact eval signature (marker: $marker). Skipping."
    return 0
  fi
  return 1
}

_mark_eval_done() {
  local marker="$1" backend="$2"
  [ -z "$marker" ] && return 0
  mkdir -p "$(dirname "$marker")"
  printf 'backend=%s\ncompleted_at=%s\n' "$backend" "$(date -Iseconds)" > "$marker"
}

run_lighteval() {
  local model_path="$1" out_dir="$2" tasks="$3" system_prompt="$4"
  local max_new_tokens="$5" seed="$6" custom_tasks="$7"
  local local_datasets_root="$8" data_parallel_size="${9:-1}"
  local eval_temperature="${10:-0.0}" eval_top_p="${11:-1.0}"
  local max_model_length="${12:-4096}" gpu_memory_utilization="${13:-0.8}"
  local dtype="${14:-bfloat16}" max_samples="${15:-}"
  local done_marker
  if [ "${EVAL_COMPLETION_MODE:-legacy}" = "signature" ]; then
    done_marker="$(_eval_done_marker "$out_dir" "lighteval" \
      "$model_path" "$tasks" "$system_prompt" "$max_new_tokens" "$seed" \
      "$custom_tasks" "$local_datasets_root" "$data_parallel_size" \
      "$eval_temperature" "$eval_top_p" "$max_model_length" "$dtype" \
      "$max_samples")"
  else
    done_marker="$out_dir/.eval_done"
  fi
  _eval_should_skip "$done_marker" "LightEval" && return 0
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  local eval_cache_root="${LIGHTEVAL_CACHE_ROOT:-$local_datasets_root/.cache/eval}"
  mkdir -p "$out_dir" "$eval_cache_root/hf_home" "$eval_cache_root/datasets" "$eval_cache_root/transformers"
  local -a LE_ARGS=(
    -m eval.lighteval.runner
    --model_path "$model_path"
    --tasks "$tasks"
    --output_dir "$out_dir"
    --max_new_tokens "$max_new_tokens"
    --max_model_length "$max_model_length"
    --gpu_memory_utilization "$gpu_memory_utilization"
    --dtype "$dtype"
    --temperature "$eval_temperature" --top_p "$eval_top_p" --seed "$seed"
    --data_parallel_size "$data_parallel_size"
    --custom_tasks "$custom_tasks"
    --local_datasets_root "$local_datasets_root"
    --hf_home "$eval_cache_root/hf_home"
    --datasets_cache "$eval_cache_root/datasets"
    --transformers_cache "$eval_cache_root/transformers"
  )
  [ -n "$system_prompt" ] && LE_ARGS+=(--system_prompt "$system_prompt")
  [ -n "$max_samples" ] && LE_ARGS+=(--max_samples "$max_samples")
  env -u PYTORCH_CUDA_ALLOC_CONF python "${LE_ARGS[@]}"
  if [ "${EVAL_COMPLETION_MODE:-legacy}" = "signature" ]; then
    _mark_eval_done "$done_marker" "lighteval"
  else
    date -Iseconds > "$done_marker"
  fi
}

run_zeroeval() {
  local model_path="$1" model_tag="$2" checkpoint_tag="$3" output_root="$4"
  local zeroeval_dir="$5" tp_size="$6" batch_size="$7" max_tokens="$8"
  local gpu_mem="$9" seed="${10}" max_samples="${11}"
  local done_marker
  done_marker="$(_eval_done_marker "$output_root" "zeroeval" \
    "$model_path" "$model_tag" "$checkpoint_tag" "$zeroeval_dir" "$tp_size" \
    "$batch_size" "$max_tokens" "$gpu_mem" "$seed" "$max_samples")"
  _eval_should_skip "$done_marker" "ZeroEval" && return 0

  env -u PYTORCH_CUDA_ALLOC_CONF \
  ZEROEVAL_TP_SIZE="$tp_size" \
  ZEROEVAL_BATCH_SIZE="$batch_size" \
  ZEROEVAL_MAX_TOKENS="$max_tokens" \
  ZEROEVAL_GPU_MEMORY_UTILIZATION="$gpu_mem" \
  python -m eval.zeroeval.runner \
    --model_path "$model_path" \
    --model_tag "$model_tag" \
    --checkpoint_tag "$checkpoint_tag" \
    --output_root "$output_root" \
    --zeroeval_dir "$zeroeval_dir" \
    --mode zeroeval_repo \
    --engine vllm \
    --tensor_parallel_size "$tp_size" \
    --dtype bfloat16 \
    --gpu_memory_utilization "$gpu_mem" \
    --batch_size "$batch_size" \
    --max_tokens "$max_tokens" \
    --temperature 0 --top_p 1.0 \
    --seed "$seed" \
    --max_samples "$max_samples"
  _mark_eval_done "$done_marker" "zeroeval"
}

run_hacrl_eval() {
  local model_path="$1" model_tag="$2" checkpoint_tag="$3" output_root="$4"
  local datasets="$5" scorer="$6" system_prompt="$7" seed="$8"
  local max_new_tokens="${9:-4096}" vllm_batch_size="${10:-16}"
  local num_generations="${11:-1}" max_samples="${12:-}"
  local temperature="${13:-0.0}" top_p="${14:-1.0}"
  local use_offline="${15:-true}"
  local gpu_mem="${16:-0.9}" tp_size="${17:-1}" dtype="${18:-bfloat16}"
  local max_model_len="${19:-}" cuda_device="${20:-}"
  local done_marker
  if [ "${EVAL_COMPLETION_MODE:-legacy}" = "signature" ]; then
    done_marker="$(_eval_done_marker "$output_root" "hacrl" \
      "$model_path" "$model_tag" "$checkpoint_tag" "$datasets" "$scorer" \
      "$system_prompt" "$seed" "$max_new_tokens" "$num_generations" \
      "$max_samples" "$temperature" "$top_p" "$use_offline" "$dtype" \
      "$max_model_len")"
  else
    done_marker="$output_root/.hacrl_done_${model_tag}_${checkpoint_tag}"
  fi
  _eval_should_skip "$done_marker" "HACRL eval" && return 0
  mkdir -p "$output_root"
  local HACRL_ARGS=(
    --model_path "$model_path"
    --model_tag "$model_tag"
    --checkpoint_tag "$checkpoint_tag"
    --datasets "$datasets"
    --scorer "$scorer"
    --output_root "$output_root"
    --seed "$seed"
    --max_new_tokens "$max_new_tokens"
    --num_generations "$num_generations"
    --temperature "$temperature"
    --top_p "$top_p"
  )
  if [[ "$use_offline" == "true" ]]; then
    HACRL_ARGS+=(--use_offline_vllm)
    HACRL_ARGS+=(--gpu_memory_utilization "$gpu_mem")
    HACRL_ARGS+=(--tensor_parallel_size "$tp_size")
    HACRL_ARGS+=(--vllm_dtype "$dtype")
    [ -n "$max_model_len" ] && HACRL_ARGS+=(--vllm_max_model_len "$max_model_len")
    [ -n "$cuda_device" ] && HACRL_ARGS+=(--cuda_device "$cuda_device")
  else
    HACRL_ARGS+=(--no-use_offline_vllm)
    HACRL_ARGS+=(--vllm_batch_size "$vllm_batch_size")
  fi
  if [ -n "$system_prompt" ] && [ "$system_prompt" != "null" ]; then
    HACRL_ARGS+=(--system_prompt "$system_prompt")
  fi
  [ -n "$max_samples" ] && HACRL_ARGS+=(--max_samples "$max_samples")
  env -u PYTORCH_CUDA_ALLOC_CONF python -m src.evaluate_checkpoints "${HACRL_ARGS[@]}"
  if [ "${EVAL_COMPLETION_MODE:-legacy}" = "signature" ]; then
    _mark_eval_done "$done_marker" "hacrl"
  else
    date -Iseconds > "$done_marker"
  fi
}

run_ttrl_eval() {
  local model_path="$1" model_tag="$2" checkpoint_tag="$3" output_root="$4"
  local datasets="$5" scorer="$6" seed="$7"
  local max_new_tokens="${8:-4096}" num_generations="${9:-1}" max_samples="${10:-}"
  local temperature="${11:-0.0}" top_p="${12:-1.0}"
  local prompt_mode="${13:-bare}"
  local gpu_mem="${14:-0.9}" tp_size="${15:-1}" dtype="${16:-bfloat16}"
  local max_model_len="${17:-}" dataset_overrides="${18:-}"
  local done_marker
  if [ "${EVAL_COMPLETION_MODE:-legacy}" = "signature" ]; then
    done_marker="$(_eval_done_marker "$output_root" "ttrl" \
      "$model_path" "$model_tag" "$checkpoint_tag" "$datasets" "$scorer" \
      "$seed" "$max_new_tokens" "$num_generations" "$max_samples" \
      "$temperature" "$top_p" "$prompt_mode" "$dtype" "$max_model_len" \
      "$dataset_overrides")"
  else
    done_marker="$output_root/.ttrl_done_${model_tag}_${checkpoint_tag}"
  fi
  _eval_should_skip "$done_marker" "TTRL eval" && return 0
  mkdir -p "$output_root"

  local -a ds_arr
  IFS=',' read -ra ds_arr <<< "$datasets"
  local TTRL_ARGS=(
    --model_paths "$model_path"
    --datasets "${ds_arr[@]}"
    --scorer "$scorer"
    --num_generations "$num_generations"
    --temperature "$temperature"
    --top_p "$top_p"
    --max_new_tokens "$max_new_tokens"
    --seed "$seed"
    --gpu_memory_utilization "$gpu_mem"
    --tensor_parallel_size "$tp_size"
    --dtype "$dtype"
    --output_dir "$output_root/${model_tag}_${checkpoint_tag}"
  )
  [ -n "$prompt_mode" ] && TTRL_ARGS+=(--prompt_mode "$prompt_mode")
  [ -n "$max_samples" ] && TTRL_ARGS+=(--max_samples "$max_samples")
  [ -n "$max_model_len" ] && TTRL_ARGS+=(--max_model_len "$max_model_len")
  [ -n "$dataset_overrides" ] && TTRL_ARGS+=(--dataset_overrides "$dataset_overrides")

  env -u PYTORCH_CUDA_ALLOC_CONF python -m eval.ttrl.runner "${TTRL_ARGS[@]}"
  if [ "${EVAL_COMPLETION_MODE:-legacy}" = "signature" ]; then
    _mark_eval_done "$done_marker" "ttrl"
  else
    date -Iseconds > "$done_marker"
  fi
}
