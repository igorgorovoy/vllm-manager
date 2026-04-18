#!/usr/bin/env bash
# Benchmark every model profile in models.json sequentially.
#
# For each model: start server via vllm_models.py, wait for /health,
# run `vllm bench serve` with a fixed synthetic workload, stop server.
# Results: per-run JSON + appended row in benchmarks/summary.csv.
#
# Defaults (override via env):
#   NUM_PROMPTS=100  INPUT_LEN=512  OUTPUT_LEN=256
#   REQUEST_RATE=inf (all at once — stress test)
#   DATASET=random   (fixed synthetic tokens — apples-to-apples)
#   PORT=8000        ONLY=name1,name2 (restrict set)
#   BENCH_KEEP=20    (retain last N per-run JSONs)
#
# Flags:
#   --if-idle   Skip run (exit 0) if any vLLM server is currently active.
#               Used by the systemd timer so scheduled runs don't kill
#               a server someone is using.
#
# Usage:
#   ./bench_models.sh
#   ONLY=glm47-flash,gemma4-31b-it ./bench_models.sh
#   NUM_PROMPTS=50 REQUEST_RATE=4 ./bench_models.sh
#   ./bench_models.sh --if-idle
set -euo pipefail

IF_IDLE=0
for arg in "$@"; do
  case "$arg" in
    --if-idle) IF_IDLE=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
NUM_PROMPTS=${NUM_PROMPTS:-100}
REQUEST_RATE=${REQUEST_RATE:-inf}
DATASET=${DATASET:-random}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-900}   # seconds to wait for /health
ONLY=${ONLY:-}
BENCH_KEEP=${BENCH_KEEP:-20}
NUM_WARMUPS=${NUM_WARMUPS:-5}
NUM_REPEATS=${NUM_REPEATS:-3}

TOOLS_DIR=/home/igogo/vllm-tools
VENV_BIN="$TOOLS_DIR/.venv/bin"
CLI="$TOOLS_DIR/vllm_models.py"
CFG="$HOME/.config/vllm-models/models.json"
OUT_DIR="$HOME/.config/vllm-models/benchmarks"
CSV="$OUT_DIR/summary.csv"

mkdir -p "$OUT_DIR"

if [[ ! -s "$CSV" ]]; then
  echo "timestamp,model,repo,num_prompts,input_len,output_len,request_rate,request_throughput_rps,output_throughput_tps,total_token_throughput_tps,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,result_file" > "$CSV"
fi

wait_health() {
  local end=$((SECONDS + HEALTH_TIMEOUT))
  while (( SECONDS < end )); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

stop_all() {
  "$CLI" stop >/dev/null 2>&1 || true
  # give GPU memory time to release
  sleep 8
}

# Return 0 if any vLLM server/engine is active, 1 otherwise.
vllm_active() {
  if pgrep -u "$(id -u)" -f 'vllm.entrypoints.openai.api_server|VLLM::EngineCore' >/dev/null 2>&1; then
    return 0
  fi
  local servers_file="$HOME/.config/vllm-models/servers.json"
  if [[ -s "$servers_file" ]] && [[ "$(jq '.servers | length' "$servers_file" 2>/dev/null || echo 0)" -gt 0 ]]; then
    return 0
  fi
  return 1
}

if [[ "$IF_IDLE" -eq 1 ]] && vllm_active; then
  echo "[skip] vLLM server detected — --if-idle run aborted cleanly"
  exit 0
fi

echo "== Cleaning up any running vLLM servers =="
stop_all

if [[ -n "$ONLY" ]]; then
  models=$(echo "$ONLY" | tr ',' '\n')
else
  models=$(jq -r '.models | keys[]' "$CFG")
fi

for name in $models; do
  echo
  echo "===== $name ====="
  repo=$(jq -r --arg n "$name" '.models[$n].repo // empty' "$CFG")
  if [[ -z "$repo" ]]; then
    echo "[$name] not found in models.json, skipping" >&2
    continue
  fi

  ts=$(date -u +%Y%m%dT%H%M%SZ)

  echo "[$name] starting server ($repo)"
  if ! "$CLI" serve "$name" --port "$PORT"; then
    echo "[$name] serve failed, skipping" >&2
    continue
  fi

  echo "[$name] waiting for /health (up to ${HEALTH_TIMEOUT}s)..."
  if ! wait_health; then
    echo "[$name] health check timed out, skipping" >&2
    stop_all
    continue
  fi
  echo "[$name] ready, running benchmark ($NUM_REPEATS repeats)"

  run_files=()
  for (( r=1; r<=NUM_REPEATS; r++ )); do
    if [[ "$NUM_REPEATS" -gt 1 ]]; then
      result_file="bench_${name}_${ts}_run${r}.json"
      echo "[$name] --- repeat $r/$NUM_REPEATS ---"
    else
      result_file="bench_${name}_${ts}.json"
    fi

    set +e
    "$VENV_BIN/vllm" bench serve \
      --host "$HOST" --port "$PORT" \
      --model "$repo" \
      --dataset-name "$DATASET" \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$NUM_PROMPTS" \
      --num-warmups "$NUM_WARMUPS" \
      --request-rate "$REQUEST_RATE" \
      --percentile-metrics "ttft,tpot,itl,e2el" \
      --save-result \
      --result-dir "$OUT_DIR" \
      --result-filename "$result_file"
    bench_rc=$?
    set -e

    if [[ $bench_rc -ne 0 ]]; then
      echo "[$name] run $r exited with $bench_rc" >&2
    fi

    if [[ -f "$OUT_DIR/$result_file" ]]; then
      run_files+=("$OUT_DIR/$result_file")
    fi
  done

  if (( ${#run_files[@]} > 0 )); then
    if [[ "$NUM_REPEATS" -gt 1 ]]; then
      rf_csv="bench_${name}_${ts}_run*.json (median of ${#run_files[@]})"
    else
      rf_csv="$(basename "${run_files[0]}")"
    fi
    jq -r -s \
      --arg ts "$ts" --arg name "$name" --arg repo "$repo" \
      --arg np "$NUM_PROMPTS" --arg il "$INPUT_LEN" --arg ol "$OUTPUT_LEN" --arg rr "$REQUEST_RATE" \
      --arg rf "$rf_csv" \
      'def med(f): map(f // 0) | sort | .[length/2|floor];
       [$ts,$name,$repo,$np,$il,$ol,$rr,
        (med(.request_throughput) | tostring),
        (med(.output_throughput) | tostring),
        (med(.total_token_throughput) | tostring),
        (med(.mean_ttft_ms) | tostring),
        (med(.median_ttft_ms) | tostring),
        (med(.p99_ttft_ms) | tostring),
        (med(.mean_tpot_ms) | tostring),
        (med(.median_tpot_ms) | tostring),
        (med(.p99_tpot_ms) | tostring),
        (med(.mean_itl_ms) | tostring),
        $rf] | @csv' \
      "${run_files[@]}" >> "$CSV"
    echo "[$name] appended median row to $CSV"
  else
    echo "[$name] no result JSONs produced" >&2
  fi

  echo "[$name] stopping server"
  stop_all
done

echo
echo "== Summary =="
column -s, -t "$CSV"
echo
echo "Raw JSONs: $OUT_DIR"
echo "CSV:       $CSV"

# Retention: keep only the newest $BENCH_KEEP per-run JSONs.
if [[ "$BENCH_KEEP" -gt 0 ]]; then
  mapfile -t old_jsons < <(ls -1t "$OUT_DIR"/bench_*.json 2>/dev/null | tail -n +$((BENCH_KEEP + 1)))
  if (( ${#old_jsons[@]} > 0 )); then
    echo "Retention: removing ${#old_jsons[@]} old JSON(s) beyond last $BENCH_KEEP"
    rm -f "${old_jsons[@]}"
  fi
fi
