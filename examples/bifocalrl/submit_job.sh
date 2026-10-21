#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=260G
#SBATCH --partition=A100
#SBATCH --qos=gpua100
#SBATCH --output=./logs/slurm-%x-%j.out
#SBATCH --error=./logs/slurm-%x-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd -- "${SUBMIT_DIR}" && pwd)"
SCRIPT_DIR="${REPO_ROOT}/examples/bifocalrl"

if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Virtual environment not found: ${REPO_ROOT}/.venv/bin/activate" >&2
    exit 1
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

# Slurm may export this AMD/ROCm variable even on CUDA nodes. Ray captures its
# environment when the cluster starts, so it must be removed before starting
# either the head or worker processes. Otherwise veRL workers see both
# ROCR_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES and abort during initialization.
unset ROCR_VISIBLE_DEVICES

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NNODE="${NNODE:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export EVAL_CONFIG="${EVAL_CONFIG:-examples/bifocalrl/eval_configs/full.json}"
export RECIPE_NAME="${RECIPE_NAME:-bifocalrl}"
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://127.0.0.1:8080/run_code}"

mapfile -t ray_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#ray_nodes[@]}" -ne "${NNODE}" ]]; then
    echo "Expected ${NNODE} nodes, but Slurm allocated ${#ray_nodes[@]}." >&2
    exit 1
fi

head_ip="$(hostname -I | awk '{print $1}')"
ray_port="${RAY_PORT:-6379}"
ray_address="${head_ip}:${ray_port}"

ray_started=0
worker_steps=()
cleanup() {
    for worker_step in "${worker_steps[@]}"; do
        kill "${worker_step}" >/dev/null 2>&1 || true
    done
    if [[ "${ray_started}" -eq 1 ]]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

env -u ROCR_VISIBLE_DEVICES ray start --head --node-ip-address="${head_ip}" --port="${ray_port}" \
    --num-gpus="${GPUS_PER_NODE}" --disable-usage-stats
ray_started=1

for worker_node in "${ray_nodes[@]:1}"; do
    srun --nodes=1 --ntasks=1 --nodelist="${worker_node}" --exclusive \
        env -u ROCR_VISIBLE_DEVICES \
        ray start --address="${ray_address}" --num-gpus="${GPUS_PER_NODE}" \
        --disable-usage-stats --block &
    worker_steps+=("$!")
done

# Give worker nodes time to register before Trinity validates cluster resources.
sleep 5

"${SCRIPT_DIR}/run.sh" "${RECIPE_NAME}" "$@"
