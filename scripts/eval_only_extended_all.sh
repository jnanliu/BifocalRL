#!/usr/bin/env bash
set -euo pipefail

# Run the extended evaluation profile against already-saved checkpoints.
# Positional arguments are model configuration directories under
# recipes/configs/mixed_reward/single/, for example:
#   bash scripts/eval_only_extended_all.sh Qwen3-4B-Base DeepSeek-R1-Distill-Qwen-1.5B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_ROOT="$PROJECT_ROOT/recipes/configs/mixed_reward/single"

CONDA_ENV="${COLEARNING_CONDA_ENV:-colearning-self}"
PROFILE_FILE="${EVAL_PROFILE_FILE:-$CONFIG_ROOT/Qwen2.5-3B/single_mixed_reward_reliability_04sc_06corr_all_extended_eval.json}"
START_STEP="${EVAL_START_STEP:-0}"
END_STEP="${EVAL_END_STEP:-15}"
STRICT_MODE="${EVAL_ONLY_STRICT:-false}"
FORCE_EVAL="false"
DRY_RUN="false"
CODE_ONLY="false"

# The three standard variants, the original fixed-alpha baseline, and the
# seven reliability variants. Their order is intentionally stable for logs.
CONFIG_NAMES=(
  "single_intuitor.json"
  "single_mv.json"
  "single_repro_gspo.json"

  "single_mixed_reward_alpha050.json"

  "single_mixed_reward_reliability_baseline.json"

  "single_mixed_reward_reliability_06sc_04corr_all.json"
  "single_mixed_reward_reliability_06sc_04corr_advantage_only.json"
  "single_mixed_reward_reliability_06sc_04corr_sample_only.json"

  "single_mixed_reward_reliability_04sc_06corr_all.json"
  "single_mixed_reward_reliability_04sc_06corr_advantage_only.json"
  "single_mixed_reward_reliability_04sc_06corr_sample_only.json"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_only_extended_all.sh [options] [MODEL_CONFIG_DIR ...]

Examples:
  scripts/eval_only_extended_all.sh Qwen3-4B-Base
  scripts/eval_only_extended_all.sh Qwen3-4B-Base DeepSeek-R1-Distill-Qwen-1.5B

MODEL_CONFIG_DIR may be a directory name under
recipes/configs/mixed_reward/single/ or an absolute path.

Options:
  --profile PATH       Eval-only profile JSON
  --start-step N       First checkpoint to evaluate (default: 0)
  --end-step N         Last checkpoint to evaluate (default: 15)
  --strict             Fail if any checkpoint in the range is missing
  --non-strict         Skip missing checkpoints (default)
  --force              Re-run evaluations even when signature markers exist
  --dry-run            Validate and print work without launching evaluators
  --code-only          Evaluate only LCB and CRUXEval (disable TTRL math tasks)
  -h, --help           Show this help

Environment overrides:
  COLEARNING_CONDA_ENV  Conda environment (default: colearning-self)
  CUDA_VISIBLE_DEVICES  GPU(s) used by the sequential evaluations (default: 0)
  EVAL_PROFILE_FILE     Default profile path
  EVAL_START_STEP       Default start step
  EVAL_END_STEP         Default end step
  EVAL_ONLY_STRICT      true or false
EOF
}

MODEL_DIRS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      [ "$#" -ge 2 ] || { echo "[ERROR] --profile requires a path" >&2; exit 1; }
      PROFILE_FILE="$2"
      shift 2
      ;;
    --start-step)
      [ "$#" -ge 2 ] || { echo "[ERROR] --start-step requires an integer" >&2; exit 1; }
      START_STEP="$2"
      shift 2
      ;;
    --end-step)
      [ "$#" -ge 2 ] || { echo "[ERROR] --end-step requires an integer" >&2; exit 1; }
      END_STEP="$2"
      shift 2
      ;;
    --strict)
      STRICT_MODE="true"
      shift
      ;;
    --non-strict)
      STRICT_MODE="false"
      shift
      ;;
    --force)
      FORCE_EVAL="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --code-only)
      CODE_ONLY="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      MODEL_DIRS+=("$@")
      break
      ;;
    -*)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      MODEL_DIRS+=("$1")
      shift
      ;;
  esac
done

if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
  MODEL_DIRS=("Qwen3-4B-Base")
fi

[[ "$START_STEP" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid start step: $START_STEP" >&2; exit 1; }
[[ "$END_STEP" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid end step: $END_STEP" >&2; exit 1; }
[ "$START_STEP" -le "$END_STEP" ] || { echo "[ERROR] start step must be <= end step" >&2; exit 1; }
case "$STRICT_MODE" in
  true|false) ;;
  *) echo "[ERROR] strict mode must be true or false: $STRICT_MODE" >&2; exit 1 ;;
esac
[ -f "$PROFILE_FILE" ] || { echo "[ERROR] Eval profile not found: $PROFILE_FILE" >&2; exit 1; }

# Activate the requested environment without requiring a shell-specific rc file.
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available in PATH; activate conda before running this script." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Some environments have activation hooks that reference optional variables
# without defaults. Keep the runner strict, but let conda initialize first.
set +u
conda activate "$CONDA_ENV"
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOCAL_DATASETS_ROOT="${LOCAL_DATASETS_ROOT:-$PROJECT_ROOT/../datasets}"
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"

resolve_model_config_dir() {
  local requested="$1"
  if [ -d "$requested" ]; then
    printf '%s\n' "$(cd "$requested" && pwd)"
  elif [ -d "$CONFIG_ROOT/$requested" ]; then
    printf '%s\n' "$(cd "$CONFIG_ROOT/$requested" && pwd)"
  else
    echo "[ERROR] Model config directory not found: $requested" >&2
    exit 1
  fi
}

for requested_model_dir in "${MODEL_DIRS[@]}"; do
  model_config_dir="$(resolve_model_config_dir "$requested_model_dir")"
  echo
  echo "=============================================================="
  echo "Extended eval-only model: $model_config_dir"
  echo "Steps: $START_STEP..$END_STEP"
  echo "GPU(s): $CUDA_VISIBLE_DEVICES"
  echo "Strict: $STRICT_MODE"
  echo "Scope: $([ "$CODE_ONLY" = "true" ] && echo 'code-only (LCB + CRUXEval)' || echo 'all extended tasks')"
  echo "=============================================================="

  for config_name in "${CONFIG_NAMES[@]}"; do
    config_path="$model_config_dir/$config_name"
    [ -f "$config_path" ] || {
      if [ "$STRICT_MODE" = "true" ]; then
        echo "[ERROR] Missing variant config: $config_path" >&2
        exit 1
      fi
      echo "[WARNING] Missing variant config, skipping: $config_path" >&2
      continue
    }

    echo
    echo "--------------------------------------------------------------"
    echo "Variant: $config_name"
    echo "--------------------------------------------------------------"

    eval_args=(
      "$config_path"
      --eval-only
      --eval-profile "$PROFILE_FILE"
      --eval-start-step "$START_STEP"
      --eval-end-step "$END_STEP"
    )
    [ "$FORCE_EVAL" = "true" ] && eval_args+=(--eval-force)
    [ "$DRY_RUN" = "true" ] && eval_args+=(--eval-dry-run)

    if [ "$CODE_ONLY" = "true" ]; then
      EVAL_ONLY_STRICT="$STRICT_MODE" TTRL_EVAL_ENABLED=false \
        bash "$PROJECT_ROOT/scripts/run_single.sh" "${eval_args[@]}"
    else
      EVAL_ONLY_STRICT="$STRICT_MODE" \
        bash "$PROJECT_ROOT/scripts/run_single.sh" "${eval_args[@]}"
    fi
  done
done

echo
echo "All requested extended eval-only jobs finished."
