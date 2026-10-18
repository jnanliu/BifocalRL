#!/usr/bin/env bash
#
# Start/stop two vLLM OpenAI-compatible servers with sleep-mode enabled.
#
# Why:
# - When `src/generate_and_score.py` becomes distributed, it should NOT be responsible for
#   starting long-lived vLLM servers.
# - We manually start two servers:
#     - port 8000 for model A
#     - port 8001 for model B
# - The pipeline connects to them with `--auto_start_server=false --auto_stop_server=false`,
#   waking them up for generation and putting them to sleep afterwards.
#
# Usage:
#   MODEL_A_PATH=/path/to/a MODEL_B_PATH=/path/to/b ./scripts/vllm_dual_server.sh start
#   ./scripts/vllm_dual_server.sh status
#   ./scripts/vllm_dual_server.sh sleep
#   ./scripts/vllm_dual_server.sh wake
#   ./scripts/vllm_dual_server.sh stop
#
# Optional overrides:
#   PORT_A=8000 PORT_B=8001 HOST=0.0.0.0 \
#   CUDA_A=0 CUDA_B=1 DTYPE=bfloat16 GPU_MEM_UTIL=0.90 \
#   LOG_DIR=outputs/vllm \
#   ./scripts/vllm_dual_server.sh start
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CMD="${1:-}"
if [[ -z "$CMD" ]]; then
  echo "Usage: $0 {start|stop|status|wake|sleep}" >&2
  exit 2
fi

MODEL_A_PATH="${MODEL_A_PATH:-}"
MODEL_B_PATH="${MODEL_B_PATH:-}"
PORT_A="${PORT_A:-8000}"
PORT_B="${PORT_B:-8001}"
HOST="${HOST:-0.0.0.0}"
CUDA_A="${CUDA_A:-0}"
CUDA_B="${CUDA_B:-1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
LOG_DIR="${LOG_DIR:-outputs/vllm}"

mkdir -p "$LOG_DIR"
PID_A_FILE="$LOG_DIR/vllm_${PORT_A}.pid"
PID_B_FILE="$LOG_DIR/vllm_${PORT_B}.pid"
LOG_A="$LOG_DIR/vllm_${PORT_A}.log"
LOG_B="$LOG_DIR/vllm_${PORT_B}.log"

health_url() {
  local port="$1"
  echo "http://localhost:${port}/health"
}

is_running_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then return 1; fi
  kill -0 "$pid" >/dev/null 2>&1
}

pid_from_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    tr -d ' \n\t' <"$f" || true
  fi
}

wait_health() {
  local port="$1"
  local timeout="${2:-600}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "$(health_url "$port")" >/dev/null 2>&1; then
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start > timeout )); then
      return 1
    fi
    sleep 2
  done
}

start_one() {
  local model_path="$1"
  local port="$2"
  local cuda="$3"
  local pid_file="$4"
  local log_file="$5"

  if [[ -z "$model_path" ]]; then
    echo "[ERROR] MODEL path missing for port=$port" >&2
    exit 1
  fi

  # If pid file exists and process is alive, do not start another.
  local pid
  pid="$(pid_from_file "$pid_file")"
  if [[ -n "$pid" ]] && is_running_pid "$pid"; then
    echo "[INFO] vLLM already running on port=$port (pid=$pid)"
    return 0
  fi

  echo "[INFO] Starting vLLM server on port=$port (CUDA_VISIBLE_DEVICES=$cuda, dtype=$DTYPE)"
  echo "[INFO]   model: $model_path"
  echo "[INFO]   log:   $log_file"

  # Run in its own process group so `stop` can kill children.
  (
    export CUDA_VISIBLE_DEVICES="$cuda"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --host "$HOST" \
      --port "$port" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --dtype "$DTYPE" \
      --enable-sleep-mode \
      --trust-remote-code \
      --no-enable-chunked-prefill
  ) >"$log_file" 2>&1 &

  local new_pid="$!"
  echo "$new_pid" >"$pid_file"

  if ! wait_health "$port" 600; then
    echo "[ERROR] vLLM failed to become healthy on port=$port (pid=$new_pid). Check $log_file" >&2
    exit 1
  fi

  echo "[INFO] vLLM healthy on port=$port (pid=$new_pid)"
}

stop_one() {
  local port="$1"
  local pid_file="$2"

  local pid
  pid="$(pid_from_file "$pid_file")"
  if [[ -z "$pid" ]]; then
    echo "[INFO] No pid file for port=$port ($pid_file)"
    return 0
  fi
  if ! is_running_pid "$pid"; then
    echo "[INFO] Stale pid file for port=$port (pid=$pid). Removing."
    rm -f "$pid_file"
    return 0
  fi

  echo "[INFO] Stopping vLLM on port=$port (pid=$pid)"
  # Best-effort kill process group, then pid.
  kill -TERM "-$pid" >/dev/null 2>&1 || true
  kill -TERM "$pid" >/dev/null 2>&1 || true
  sleep 2
  if is_running_pid "$pid"; then
    kill -KILL "-$pid" >/dev/null 2>&1 || true
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$pid_file"
}

post_json() {
  local port="$1"
  local path="$2"
  curl -fsS -X POST "http://localhost:${port}${path}" >/dev/null 2>&1 || true
}

post_sleep() {
  local port="$1"
  local level="${2:-1}"
  curl -fsS -X POST "http://localhost:${port}/sleep?level=${level}" >/dev/null 2>&1 || true
}

case "$CMD" in
  start)
    start_one "$MODEL_A_PATH" "$PORT_A" "$CUDA_A" "$PID_A_FILE" "$LOG_A"
    start_one "$MODEL_B_PATH" "$PORT_B" "$CUDA_B" "$PID_B_FILE" "$LOG_B"
    ;;
  stop)
    stop_one "$PORT_A" "$PID_A_FILE"
    stop_one "$PORT_B" "$PID_B_FILE"
    ;;
  status)
    for port in "$PORT_A" "$PORT_B"; do
      if curl -fsS "$(health_url "$port")" >/dev/null 2>&1; then
        echo "[INFO] port=$port healthy"
      else
        echo "[INFO] port=$port not healthy/unreachable"
      fi
    done
    ;;
  wake)
    post_json "$PORT_A" "/wake_up"
    post_json "$PORT_B" "/wake_up"
    ;;
  sleep)
    post_sleep "$PORT_A" 1
    post_sleep "$PORT_B" 1
    ;;
  *)
    echo "Usage: $0 {start|stop|status|wake|sleep}" >&2
    exit 2
    ;;
esac

