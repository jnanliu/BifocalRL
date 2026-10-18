#!/usr/bin/env bash
# Single-GPU sweep: ref vs checkpoint-4 for all six co-q* combos × three recipe slots.
# Run from anywhere; delegates to co-learning on PYTHONPATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CO_LEARNING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CO_LEARNING_DIR}/.." && pwd)"

export PYTHONPATH="${CO_LEARNING_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m ablation.run_combo_reward_capability_scan \
  --project_root "${PROJECT_ROOT}" \
  "$@"
