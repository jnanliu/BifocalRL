#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NLTK_DATA_DIR="${NLTK_DATA:-$HOME/nltk_data}"
mkdir -p "$NLTK_DATA_DIR"

echo "Installing NLTK resources into: $NLTK_DATA_DIR"
python -m nltk.downloader -d "$NLTK_DATA_DIR" punkt punkt_tab

NLTK_DATA="$NLTK_DATA_DIR" python - <<'PY'
import nltk

for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    path = nltk.data.find(resource)
    print(f"nltk_ok: {resource} -> {path}")
PY

NLTK_DATA="$NLTK_DATA_DIR" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python - <<'PY'
from eval.lighteval.custom_tasks.offline_lcb_single import TASKS_TABLE

task_name = "lcb:codegeneration_release_v6_single"
registered = {task.name for task in TASKS_TABLE}
if task_name not in registered:
    raise RuntimeError(f"LightEval task is not registered: {task_name}")
metrics = [metric.metric_name for task in TASKS_TABLE for metric in task.metric]
if metrics != ["codegen_pass@1:1"]:
    raise RuntimeError(f"Unexpected LCB metrics: {metrics}")
print(f"lighteval_ok: custom|{task_name}, metric={metrics[0]}")
PY
