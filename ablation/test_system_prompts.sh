#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Test different system prompts on gsm8k & math_500 using lighteval.
#
# Calls `python -m src.evaluate_lighteval` (same as the training pipeline)
# to ensure offline dataset loading works consistently.
#
# Usage:
#   ./ablation/test_system_prompts.sh \
#       --model /home/jizeyu26/jzy/7_co_learning/test/models/Qwen2.5-3B \
#       --prompts ablation/example_system_prompts.json
#
#   # With custom options:
#   ./ablation/test_system_prompts.sh \
#       --model /home/jizeyu26/jzy/7_co_learning/test/models/Llama-3.2-3B-Instruct \
#       --prompts ablation/example_system_prompts.json \
#       --no-save-details
#
#   # Or pass system prompts inline (use @@ as separator):
#   ./ablation/test_system_prompts.sh \
#       --model /home/jizeyu26/jzy/7_co_learning/test/models/Qwen2.5-3B \
#       --inline "You are a math expert.@@Think step by step."
#
# prompts.json format:
#   {
#     "no_prompt": null,
#     "math_expert": "You are a math expert. Solve step by step and give final answer in \\boxed{}.",
#     "cot_simple": "Think step by step."
#   }
#
# Each key becomes a subdirectory name under the output directory.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT" || exit 1

# ---- Defaults ----
MODEL_PATH=""
PROMPTS_FILE=""
INLINE_PROMPTS=""
OUTPUT_ROOT="outputs/ablation/system_prompts"
SAVE_DETAILS=true   # default: save details (use --no-save-details to turn off)
TASKS="math_500,gsm8k"
CUSTOM_TASKS="eval.lighteval.custom_tasks.offline_patch"
LOCAL_DATASETS_ROOT="/home/jizeyu26/jzy/7_co_learning/test/datasets"
MAX_NEW_TOKENS=4096
TEMPERATURE=0.0
TOP_P=1.0
SEED=7
DTYPE="bfloat16"
MAX_MODEL_LENGTH=4096
GPU_MEMORY_UTILIZATION=0.9
DATA_PARALLEL_SIZE=1

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model_path)     MODEL_PATH="$2"; shift 2 ;;
    --prompts)                PROMPTS_FILE="$2"; shift 2 ;;
    --inline)                 INLINE_PROMPTS="$2"; shift 2 ;;
    --output)                 OUTPUT_ROOT="$2"; shift 2 ;;
    --save-details)           SAVE_DETAILS=true; shift ;;
    --no-save-details)        SAVE_DETAILS=false; shift ;;
    --tasks)                  TASKS="$2"; shift 2 ;;
    --custom-tasks)           CUSTOM_TASKS="$2"; shift 2 ;;
    --local-datasets-root)    LOCAL_DATASETS_ROOT="$2"; shift 2 ;;
    --max-new-tokens)         MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature)            TEMPERATURE="$2"; shift 2 ;;
    --top-p)                  TOP_P="$2"; shift 2 ;;
    --seed)                   SEED="$2"; shift 2 ;;
    --dtype)                  DTYPE="$2"; shift 2 ;;
    --max-model-length)       MAX_MODEL_LENGTH="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --data-parallel-size)     DATA_PARALLEL_SIZE="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 --model MODEL_PATH --prompts prompts.json [options]"
      echo ""
      echo "Options:"
      echo "  --model PATH               Model checkpoint path (required)"
      echo "  --prompts FILE              JSON file mapping name->system_prompt (null = no prompt)"
      echo "  --inline PROMPTS            Inline prompts separated by @@ (names auto-generated)"
      echo "  --output DIR                Output root (default: $OUTPUT_ROOT)"
      echo "  --save-details              Save lighteval detail files (default: on)"
      echo "  --no-save-details           Do not save detail files"
      echo "  --tasks TASKS               Comma-separated tasks (default: $TASKS)"
      echo "  --custom-tasks MODULE       Custom tasks module (default: $CUSTOM_TASKS)"
      echo "  --local-datasets-root PATH  Local datasets root for offline eval"
      echo "  --max-new-tokens N          Max new tokens (default: $MAX_NEW_TOKENS)"
      echo "  --temperature T             Sampling temperature (default: $TEMPERATURE)"
      echo "  --seed N                    Random seed (default: $SEED)"
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[[ -z "$MODEL_PATH" ]] && { echo "ERROR: --model is required."; exit 1; }
[[ -z "$PROMPTS_FILE" && -z "$INLINE_PROMPTS" ]] && {
  echo "ERROR: specify --prompts FILE or --inline PROMPTS"; exit 1
}

# ---- Environment ----
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_DATASETS_VERBOSITY="${HF_DATASETS_VERBOSITY:-error}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- Parse system prompts into parallel arrays ----
declare -a PROMPT_NAMES=()
declare -a PROMPT_VALUES=()

if [[ -n "$PROMPTS_FILE" ]]; then
  while IFS=$'\t' read -r name value; do
    PROMPT_NAMES+=("$name")
    PROMPT_VALUES+=("$value")
  done < <(python3 -c "
import json, sys
with open('$PROMPTS_FILE') as f:
    data = json.load(f)
for k, v in data.items():
    val = '' if v is None else str(v)
    print(f'{k}\t{val}')
")
elif [[ -n "$INLINE_PROMPTS" ]]; then
  idx=0
  IFS='@@' read -ra _parts <<< "$INLINE_PROMPTS"
  for part in "${_parts[@]}"; do
    part="$(echo "$part" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$part" ]] && continue
    PROMPT_NAMES+=("prompt_${idx}")
    PROMPT_VALUES+=("$part")
    ((idx++))
  done
fi

if [[ ${#PROMPT_NAMES[@]} -eq 0 ]]; then
  echo "ERROR: no system prompts found."; exit 1
fi

# ---- Summary ----
log "================================================================"
log "System Prompt Ablation"
log "  model:        $MODEL_PATH"
log "  tasks:        $TASKS"
log "  prompts:      ${#PROMPT_NAMES[@]}"
log "  save_details: $SAVE_DETAILS"
log "  output:       $OUTPUT_ROOT"
log "  datasets:     $LOCAL_DATASETS_ROOT"
log "================================================================"
for i in "${!PROMPT_NAMES[@]}"; do
  val="${PROMPT_VALUES[$i]}"
  [[ -z "$val" ]] && val="(none)"
  log "  [${PROMPT_NAMES[$i]}] $val"
done
log "================================================================"

# ---- Run lighteval for each system prompt ----
OVERALL_RESULTS=()

for i in "${!PROMPT_NAMES[@]}"; do
  name="${PROMPT_NAMES[$i]}"
  sp="${PROMPT_VALUES[$i]}"
  out_dir="${OUTPUT_ROOT}/${name}"

  # Skip if already completed
  done_marker="$out_dir/.eval_done"
  if [[ -f "$done_marker" ]]; then
    log "Already done: ${name} (marker: $done_marker). Skipping."
    OVERALL_RESULTS+=("${name}: SKIPPED (already done)")
    continue
  fi

  mkdir -p "$out_dir"

  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Running: ${name}"
  if [[ -n "$sp" ]]; then
    log "  system_prompt: ${sp}"
  else
    log "  system_prompt: (none)"
  fi
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Build command using src.evaluate_lighteval (same as pipeline)
  CMD=(
    env -u PYTORCH_CUDA_ALLOC_CONF
    python -m src.evaluate_lighteval
      --model_path "$MODEL_PATH"
      --tasks "$TASKS"
      --output_dir "$out_dir"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --seed "$SEED"
      --dtype "$DTYPE"
      --max_model_length "$MAX_MODEL_LENGTH"
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
      --data_parallel_size "$DATA_PARALLEL_SIZE"
  )

  if [[ -n "$sp" ]]; then
    CMD+=(--system_prompt "$sp")
  fi

  if [[ "$SAVE_DETAILS" == "true" ]]; then
    CMD+=(--save_details)
  else
    CMD+=(--no-save_details)
  fi

  if [[ -n "$CUSTOM_TASKS" ]]; then
    CMD+=(--custom_tasks "$CUSTOM_TASKS")
  fi

  if [[ -n "$LOCAL_DATASETS_ROOT" ]]; then
    CMD+=(
      --local_datasets_root "$LOCAL_DATASETS_ROOT"
      --hf_home "${LOCAL_DATASETS_ROOT}/hf_home"
      --datasets_cache "$LOCAL_DATASETS_ROOT"
      --transformers_cache "${LOCAL_DATASETS_ROOT}/transformers_cache"
    )
  fi

  log "CMD: ${CMD[*]}"

  set +e
  "${CMD[@]}" 2>&1 | tee "${out_dir}/eval.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    log "WARNING: lighteval failed for '${name}' (exit code $rc). Continuing..."
    OVERALL_RESULTS+=("${name}: FAILED (rc=$rc)")
  else
    date -Iseconds > "$done_marker"
    log "Done: ${name} → ${out_dir}"
    OVERALL_RESULTS+=("${name}: OK → ${out_dir}")
  fi
done

# ---- Final summary ----
log ""
log "================================================================"
log "All evaluations complete."
log "================================================================"
for entry in "${OVERALL_RESULTS[@]}"; do
  log "  $entry"
done
log ""
log "Results directory: $OUTPUT_ROOT"

# Collect results.json files into a combined summary
python3 -c "
import json, os, glob, sys

out_root = '$OUTPUT_ROOT'
combined = {}
for name_dir in sorted(os.listdir(out_root)):
    full = os.path.join(out_root, name_dir)
    if not os.path.isdir(full):
        continue
    for rfile in glob.glob(os.path.join(full, '**', 'results*.json'), recursive=True):
        try:
            with open(rfile) as f:
                data = json.load(f)
            combined[name_dir] = {'results_file': rfile, 'data': data}
            break
        except Exception:
            pass

if combined:
    summary_path = os.path.join(out_root, 'combined_results.json')
    with open(summary_path, 'w') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f'Combined results saved to: {summary_path}')
else:
    print('No results.json files found to combine.', file=sys.stderr)
" || true

log "Done."
