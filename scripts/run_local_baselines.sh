#!/usr/bin/env bash
# Baselines on local open models via an MLX OpenAI-compatible server (Apple silicon).
# usage: scripts/run_local_baselines.sh [n_per_split]
set -uo pipefail
cd "$(dirname "$0")/.."
N="${1:-175}"
PORT=8080
OUT=results/runs
mkdir -p "$OUT" results/logs
MODELS=(
  "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
  "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
)
ANCHOR_MODELS=("mlx-community/Llama-3.2-3B-Instruct-4bit")

run_split () {  # model split n
  local slug; slug=$(echo "$1" | tr '/' '_')
  LOCAL_API_KEY=x uv run eval @ configs/eval_local.toml -m "$1" -n "$3" \
    --env.taskset.split "$2" --env.taskset.num-tasks 400 \
    --client.base-url "http://127.0.0.1:$PORT/v1" --client.api-key-var LOCAL_API_KEY \
    --max-concurrent 4 --no-rich -o "$OUT" --run.name "${slug}--$2" --clean \
    > "results/logs/${slug}--$2.log" 2>&1
  echo "$(date +%T) done $1 $2 (exit $?)"
}

for m in "${MODELS[@]}"; do
  echo "$(date +%T) starting server for $m"
  uvx --python 3.12 --from mlx-lm mlx_lm.server --model "$m" --port $PORT > results/logs/server.log 2>&1 &
  SERVER=$!
  for _ in $(seq 1 900); do curl -s -m 1 "localhost:$PORT/v1/models" >/dev/null && break; sleep 1; done
  run_split "$m" train "$N"
  run_split "$m" heldout "$N"
  if [[ " ${ANCHOR_MODELS[*]} " == *" $m "* ]]; then run_split "$m" anchor 400; fi
  kill $SERVER; wait $SERVER 2>/dev/null
done
echo "$(date +%T) all done"
