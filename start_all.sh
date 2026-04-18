#!/usr/bin/env bash
# Start the full serving stack: three vLLM servers + webapp in screen.
#
# Idempotent — any component already healthy is left alone.
#
# Layout (per README co-hosting plan):
#   :8000  glm47-flash      coder      gpu_mem_util=0.70
#   :8001  gemma3-4b        reranker   gpu_mem_util=0.20
#   :8002  jina-embed-v3    embedder   gpu_mem_util=0.05 (trust_remote_code)
#   :7860  webapp           UI         inside screen session "vllm"
#
# Usage:
#   ./start_all.sh
#   SCREEN_NAME=mysession ./start_all.sh
set -euo pipefail

TOOLS_DIR=/home/igogo/vllm-tools
CLI="$TOOLS_DIR/vllm_models.py"
SCREEN_NAME="${SCREEN_NAME:-vllm}"

health() {
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health")" = "200" ]
}

wait_health() {
  local port=$1 name=$2 timeout=${3:-240}
  for i in $(seq 1 "$timeout"); do
    if health "$port"; then echo "  $name ready on :$port (${i}s)"; return 0; fi
    sleep 1
  done
  echo "  TIMEOUT waiting for $name on :$port — see ~/.config/vllm-models/server-*.log" >&2
  return 1
}

ensure_server() {
  local name=$1 port=$2 mem=$3
  if health "$port"; then
    echo "✓ $name already healthy on :$port"
    return 0
  fi
  echo "→ starting $name on :$port (gpu_mem=$mem)"
  if ! "$CLI" serve "$name" --port "$port" --gpu-memory-utilization "$mem"; then
    echo "  serve command failed — is the model already running on another port?" >&2
    return 1
  fi
  wait_health "$port" "$name"
}

ensure_server glm47-flash   8000 0.70
ensure_server gemma3-4b     8001 0.20
ensure_server jina-embed-v3 8002 0.05

# Webapp in screen
if health_code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/) && [ "$health_code" = "200" ]; then
  echo "✓ webapp already up on :7860"
else
  if ! screen -ls | grep -qE "[0-9]+\.${SCREEN_NAME}\b"; then
    echo "→ creating screen session '$SCREEN_NAME'"
    screen -dmS "$SCREEN_NAME"
    sleep 1
  fi
  echo "→ starting webapp in screen '$SCREEN_NAME'"
  screen -S "$SCREEN_NAME" -X stuff "cd $TOOLS_DIR && ./.venv/bin/python run_webapp.py$(printf '\r')"
  for i in $(seq 1 30); do
    if [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = "200" ]; then
      echo "✓ webapp ready on :7860 (${i}s)"; break
    fi
    sleep 1
  done
fi

echo
echo "=== state ==="
"$CLI" status
echo
echo "attach to webapp screen: screen -r $SCREEN_NAME"
