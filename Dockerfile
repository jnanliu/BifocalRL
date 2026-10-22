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
    HF_HUB_OFFLINE=1

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates \
        build-essential \
        ffmpeg \
        git \
        git-lfs \
        mesa-utils \
        openjdk-17-jdk-headless \
        python3-venv \
        tmux \
        vim \
        xvfb \
        zsh \
        > /dev/null \
    && chmod u-s,g-s /usr/lib/*-linux-gnu/utempter/utempter

ENV SANDBOXFUSION_HOME=/opt/SandboxFusion \
    SANDBOXFUSION_RUNTIME_PATH=/opt/sandboxfusion-runtime \
    SANDBOXFUSION_PYTHON=/opt/sandboxfusion-runtime/bin/python \
    SANDBOXFUSION_WORK_DIR=/mnt/sandboxfusion

COPY examples/bifocalrl/sandboxfusion-runtime-requirements.txt \
    /tmp/sandboxfusion-runtime-requirements.txt
COPY examples/bifocalrl/sandboxfusion-uv-runtime.patch \
    /tmp/sandboxfusion-uv-runtime.patch

RUN git clone --depth=1 https://github.com/bytedance/SandboxFusion.git \
        "$SANDBOXFUSION_HOME" \
    && git -C "$SANDBOXFUSION_HOME" apply \
        /tmp/sandboxfusion-uv-runtime.patch \
    && rm /tmp/sandboxfusion-uv-runtime.patch \
    && mkdir -p \
        "$SANDBOXFUSION_HOME/docs/build" \
        "$SANDBOXFUSION_WORK_DIR" \
    && chmod -R a+rX "$SANDBOXFUSION_HOME" \
    && chmod 1777 "$SANDBOXFUSION_WORK_DIR"

# Keep SandboxFusion's dependencies separate from the training environment.
# NVIDIA's pip constraints apply to the training environment only.
RUN unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME \
    && python -m venv "$SANDBOXFUSION_RUNTIME_PATH" \
    && PIP_CONSTRAINT= PIP_BUILD_CONSTRAINT= "$SANDBOXFUSION_PYTHON" -m pip install \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && PIP_CONSTRAINT= PIP_BUILD_CONSTRAINT= "$SANDBOXFUSION_PYTHON" -m pip install \
        -r /tmp/sandboxfusion-runtime-requirements.txt \
        --editable "$SANDBOXFUSION_HOME" \
    && "$SANDBOXFUSION_PYTHON" -m pip check \
    && "$SANDBOXFUSION_PYTHON" -c \
        'from importlib.metadata import version; from packaging.version import Version; assert Version(version("starlette")) >= Version("1.0.1"), "CVE-2026-48710: vulnerable Starlette"; print("SandboxFusion web dependencies:", {p: version(p) for p in ("fastapi", "starlette", "pydantic")})' \
    && rm /tmp/sandboxfusion-runtime-requirements.txt \
    && chmod -R a+rX "$SANDBOXFUSION_HOME"

# NGC puts its training libtorch first in LD_LIBRARY_PATH, even inside a venv.
RUN unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME \
    && "$SANDBOXFUSION_PYTHON" -c \
        'import nltk; nltk.download("punkt"); nltk.download("stopwords")' \
    && "$SANDBOXFUSION_PYTHON" -c \
        'import numpy, pandas, sandbox, scipy, sklearn, torch, uvicorn; assert torch.version.cuda is None; assert (torch.tensor([1, 2]) + 1).tolist() == [2, 3]; print("SandboxFusion isolated runtime OK", torch.__version__, torch._C.__file__)' \
    && "$SANDBOXFUSION_PYTHON" -c \
        'from sandbox.server.server import app; assert app.openapi()["paths"]["/run_code"]; print("SandboxFusion API import OK")' \
    && printf '%s\n' \
        '#!/bin/sh' \
        'unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME' \
        'export TMPDIR="${SANDBOXFUSION_WORK_DIR:-/mnt/sandboxfusion}"' \
        'cd "$SANDBOXFUSION_HOME" || exit 1' \
        'exec "$SANDBOXFUSION_PYTHON" -m uvicorn sandbox.server.server:app --host "${HOST:-127.0.0.1}" --port "${_BYTEFAAS_RUNTIME_PORT:-8080}" "$@"' \
        > /usr/local/bin/sandboxfusion-server \
    && chmod 755 /usr/local/bin/sandboxfusion-server

WORKDIR /home/jliu0514/projects/BifocalRL

COPY requirements.txt /tmp/openrlhf-requirements.txt

# Constraints only pin versions; install extras through the requirements file.
RUN sed -E 's/\[[^]]+\]//g' /tmp/openrlhf-requirements.txt > /tmp/openrlhf-constraints.txt

# Install vLLM's PyTorch before compiling extensions, then keep that version.
RUN python -m pip install --no-build-isolation \
        -c /tmp/openrlhf-constraints.txt 'vllm==0.29.0' \
    && python -c 'from importlib.metadata import version; print("torch==" + version("torch"))' \
        > /tmp/openrlhf-torch-constraints.txt

RUN python -m pip install -c /tmp/openrlhf-torch-constraints.txt 'ray[default]==2.55.0'
RUN python -m pip install --no-build-isolation \
        -c /tmp/openrlhf-torch-constraints.txt 'deepspeed==0.19.6'
COPY examples/scripts/install_flash_attn.py /tmp/install_flash_attn.py
RUN python -m pip install packaging typer \
    && python /tmp/install_flash_attn.py

RUN python -m pip install --no-build-isolation \
        -c /tmp/openrlhf-torch-constraints.txt -r /tmp/openrlhf-requirements.txt

COPY . .

RUN python -m pip install --no-build-isolation \
        -c /tmp/openrlhf-torch-constraints.txt -e '.[vllm,liger]' \
    && python -c \
        'from importlib.metadata import version; from packaging.version import Version; assert Version(version("starlette")) >= Version("1.0.1"), "CVE-2026-48710: vulnerable Starlette"; print("Training Starlette:", version("starlette"))'

# Enable repository syncing to Maveric user space through S3 and Git LFS.
RUN python -m pip install --no-cache-dir git-remote-s3

RUN git lfs install --system \
    && git config --system protocol.s3.allow always \
    && git config --system user.name "git-remote-s3 tests" \
    && git config --system user.email "test@example.com"

# PEFT probes NGC's optional Transformer Engine during import.
# Remove its incompatible CUDA build after installing training dependencies.
RUN python -m pip uninstall -y \
        transformer-engine transformer-engine-torch \
        transformer-engine-cu12 transformer-engine-cu13 \
    && python -c \
        'from peft import LoraConfig; from openrlhf.models import Actor; print("PEFT and OpenRLHF Actor import OK")'

CMD ["sleep", "infinity"]
