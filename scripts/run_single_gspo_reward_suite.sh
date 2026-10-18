#!/usr/bin/env bash
set -euo pipefail

# Run four single-model GSPO reward experiments sequentially:
#   1) Intuitor  + response norm  (SCE / response length)        — baseline
#   2) Intuitor  + rollout norm   (SCE / rollout total length)
#   3) Trajectory + rollout norm  (logprob / rollout total length)
#   4) Trajectory + response norm (logprob / response length)
#
# Usage:
#   ./scripts/run_single_gspo_reward_suite.sh
#
# Override individual configs via positional args:
#   ./scripts/run_single_gspo_reward_suite.sh <config1> <config2> <config3> <config4>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_1="${1:-recipes/configs/single_intuitor_gspo.json}"
CONFIG_2="${2:-recipes/configs/single_intuitor_rollout_gspo.json}"
CONFIG_3="${3:-recipes/configs/single_trajectory_rollout_gspo.json}"
CONFIG_4="${4:-recipes/configs/single_trajectory_response_gspo.json}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

log "============================================================"
log "Single-Model GSPO Reward Suite (4 experiments)"
log "  1) $CONFIG_1"
log "  2) $CONFIG_2"
log "  3) $CONFIG_3"
log "  4) $CONFIG_4"
log "============================================================"

log ">>> Starting Experiment 1/4: Intuitor + response norm"
"$SCRIPT_DIR/run_single.sh" "$CONFIG_1"
log "<<< Experiment 1/4 finished."

log ">>> Starting Experiment 2/4: Intuitor + rollout norm"
"$SCRIPT_DIR/run_single.sh" "$CONFIG_2"
log "<<< Experiment 2/4 finished."

log ">>> Starting Experiment 3/4: Trajectory + rollout norm"
"$SCRIPT_DIR/run_single.sh" "$CONFIG_3"
log "<<< Experiment 3/4 finished."

log ">>> Starting Experiment 4/4: Trajectory + response norm"
"$SCRIPT_DIR/run_single.sh" "$CONFIG_4"
log "<<< Experiment 4/4 finished."

log "============================================================"
log "All 4 experiments completed successfully."
log "============================================================"
