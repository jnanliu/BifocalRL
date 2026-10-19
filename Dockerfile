# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:26.03-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    HF_HUB_CACHE=/project/huggingface \
    HF_HUB_OFFLINE=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    TRINITY_HOME=/opt/trinity-venv \
    PATH=/opt/trinity-venv/bin:${PATH}

# Install only utilities not guaranteed by the NGC vLLM image. apt cache
# mounts speed up retries without becoming part of the resulting image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates \
        build-essential \
        cuda-nvcc-13-0 \
        ffmpeg \
        git \
        git-lfs \
        mesa-utils \
        openjdk-17-jdk-headless \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        tmux \
        vim \
        xvfb \
        zsh \
        > /dev/null \
    && chmod u-s,g-s /usr/lib/*-linux-gnu/utempter/utempter

WORKDIR /home/jliu0514/projects/BifocalRL

COPY . .

RUN pip install -e .[vllm]

CMD ["sleep", "infinity"]
