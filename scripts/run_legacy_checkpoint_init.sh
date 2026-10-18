#!/usr/bin/env bash
set -euo pipefail

# Start a new experiment from a model-only checkpoint produced by the legacy
# pipeline. The new run gets its own cache/checkpoint/state directories.
#
# Usage:
#   bash scripts/run_legacy_checkpoint_init.sh \
#     <config.json> <legacy_checkpoint_dir> <new_checkpoint_root> reset [steps]
#
#   bash scripts/run_legacy_checkpoint_init.sh \
#     <config.json> <legacy_checkpoint_dir> <new_checkpoint_root> labeled [end_step]
#
# reset:   train the initialized model as a new run, steps 1..N (default N=15)
# labeled: keep the old experiment's labels, steps 16..N (default N=30).
#          Step 16 is intentionally an inexact warm start; steps 17 onward
#          use the new run's persisted optimizer/scheduler state.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  sed -n '1,24p' "$0"
}

[ "$#" -ge 4 ] || { usage >&2; exit 2; }

CONFIG_FILE="$1"
LEGACY_CHECKPOINT="$2"
NEW_CHECKPOINT_ROOT="$3"
MODE="$4"
END_STEP="${5:-}"

case "$CONFIG_FILE" in
  /*) ;;
  *) CONFIG_FILE="$PWD/$CONFIG_FILE" ;;
esac
case "$LEGACY_CHECKPOINT" in
  /*) ;;
  *) LEGACY_CHECKPOINT="$PWD/$LEGACY_CHECKPOINT" ;;
esac
case "$NEW_CHECKPOINT_ROOT" in
  /*) ;;
  *) NEW_CHECKPOINT_ROOT="$PWD/$NEW_CHECKPOINT_ROOT" ;;
esac

[ -f "$CONFIG_FILE" ] || { echo "Missing config: $CONFIG_FILE" >&2; exit 1; }
[ -d "$LEGACY_CHECKPOINT" ] || { echo "Missing legacy checkpoint: $LEGACY_CHECKPOINT" >&2; exit 1; }
[ -f "$LEGACY_CHECKPOINT/config.json" ] || {
  echo "Legacy checkpoint is not an HF model directory: $LEGACY_CHECKPOINT" >&2
  exit 1
}

case "$MODE" in
  reset)
    START_STEP=1
    END_STEP="${END_STEP:-15}"
    REQUIRE_OPTIMIZER_STATE=true
    ;;
  labeled)
    START_STEP=16
    END_STEP="${END_STEP:-30}"
    REQUIRE_OPTIMIZER_STATE=false
    ;;
  *)
    echo "Mode must be 'reset' or 'labeled', got: $MODE" >&2
    exit 2
    ;;
esac

[[ "$END_STEP" =~ ^[0-9]+$ ]] || { echo "Invalid end step: $END_STEP" >&2; exit 2; }
[ "$END_STEP" -ge "$START_STEP" ] || { echo "End step is before start step" >&2; exit 2; }

cd "$REPO_ROOT"

export POLICY_INIT_PATH="$LEGACY_CHECKPOINT"
export LATEST_CHECKPOINT_DIR="$NEW_CHECKPOINT_ROOT"
export TRAINING_STATE_DIR="$NEW_CHECKPOINT_ROOT/training_state"
export START_STEP
export MAX_STEPS="$END_STEP"
export PERSIST_OPTIMIZER_STATE=true
export REQUIRE_OPTIMIZER_STATE
export KEEP_ALL_CHECKPOINTS=true
export KEEP_ALL_TRAINING_STATES=true

# Keep every output of this initialized run separate from the legacy run.
export TTRL_EVAL_OUTPUT_ROOT="$NEW_CHECKPOINT_ROOT/eval/ttrl"
export LIGHTEVAL_OUT_PREFIX="$NEW_CHECKPOINT_ROOT/eval/lighteval"
export ZEROEVAL_OUTPUT_ROOT="$NEW_CHECKPOINT_ROOT/eval/zeroeval"
export HACRL_EVAL_OUTPUT_ROOT="$NEW_CHECKPOINT_ROOT/eval/hacrl"

exec bash "$REPO_ROOT/scripts/run_single.sh" "$CONFIG_FILE"
