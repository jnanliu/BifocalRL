#!/usr/bin/env bash
# Co-Learning: conda Python 3.11 → pip install -r requirements.txt → (Linux) conda cuda-toolkit 12.4 → cuda-nvcc 12.4
#
# CUDA 子步骤顺序：先 cuda-toolkit，再单独 pin cuda-nvcc=12.4，避免只装 toolkit 时 solver 拉到 13.x
# 而 nvcc 必须是 12.4。
#
# 用法:
#   bash co-learning/scripts/setup_env.sh
#   CO_LEARNING_ENV=my-env bash co-learning/scripts/setup_env.sh
#
# 环境变量:
#   CO_LEARNING_ENV   conda 环境名，默认 co-learning
#   SKIP_CUDA_CONDA=1  跳过 conda 的 cuda-toolkit / cuda-nvcc（仅 Python + pip）
#   FORCE_CUDA_CONDA=1 非 Linux 也尝试安装 conda CUDA（一般仅用于调试，易失败）
#
set -euo pipefail

CO_LEARNING_ENV="${CO_LEARNING_ENV:-co-learning}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CO_LEARNING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQ_FILE="${CO_LEARNING_ROOT}/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: requirements not found: ${REQ_FILE}" >&2
  exit 1
fi

OS="$(uname -s)"
SKIP_CUDA="${SKIP_CUDA_CONDA:-0}"
if [[ "${OS}" != "Linux" ]]; then
  if [[ "${FORCE_CUDA_CONDA:-0}" == "1" ]]; then
    echo "[warn] FORCE_CUDA_CONDA=1 on ${OS}: conda CUDA 可能不可用或架构不匹配。"
  else
    if [[ "${SKIP_CUDA}" != "1" ]]; then
      echo "[info] 非 Linux（${OS}）：默认跳过 conda 的 cuda-toolkit / nvcc。"
      echo "       若只需 Python 依赖，保持默认即可；训练机请在 Linux 上执行本脚本。"
      echo "       强制尝试 CUDA: FORCE_CUDA_CONDA=1 bash ..."
      SKIP_CUDA=1
    fi
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda 未在 PATH 中。请先安装 Miniconda/Anaconda。" >&2
  exit 1
fi

echo "============================================"
echo "  Co-Learning 环境: ${CO_LEARNING_ENV}"
echo "  Python: 3.11 | requirements: ${REQ_FILE}"
echo "  Skip conda CUDA: ${SKIP_CUDA}"
echo "============================================"

if conda run -n "${CO_LEARNING_ENV}" true 2>/dev/null; then
  echo ""
  echo "[conda] 环境已存在: ${CO_LEARNING_ENV}"
else
  echo ""
  echo "[conda] 创建环境 ${CO_LEARNING_ENV} (python=3.11) ..."
  conda create -y -n "${CO_LEARNING_ENV}" "python=3.11"
fi

echo ""
echo "[pip] 安装 co-learning/requirements.txt ..."
conda run -n "${CO_LEARNING_ENV}" python -m pip install --upgrade pip
conda run -n "${CO_LEARNING_ENV}" python -m pip install -r "${REQ_FILE}"

if [[ "${SKIP_CUDA}" != "1" ]]; then
  echo ""
  echo "[conda] (1/2) 安装 cuda-toolkit 12.4（nvidia channel；先于 nvcc）..."
  conda install -y -n "${CO_LEARNING_ENV}" -c nvidia "cuda-toolkit=12.4"

  echo ""
  echo "[conda] (2/2) 单独 pin cuda-nvcc=12.4，确保 nvcc 为 12.4 ..."
  conda install -y -n "${CO_LEARNING_ENV}" -c nvidia "cuda-nvcc=12.4"

  echo ""
  echo "[verify] nvcc --version（须含 release 12.4）"
  if ! conda run -n "${CO_LEARNING_ENV}" nvcc --version 2>/dev/null | grep -q "release 12.4"; then
    echo "ERROR: nvcc 不是 12.4。请检查 conda list cuda-nvcc / cuda-toolkit，必要时手动指定 build。" >&2
    conda run -n "${CO_LEARNING_ENV}" nvcc --version || true
    exit 1
  fi
  conda run -n "${CO_LEARNING_ENV}" nvcc --version
fi

echo ""
echo "============================================"
echo "  完成。激活环境:"
echo "    conda activate ${CO_LEARNING_ENV}"
echo "============================================"
