#!/usr/bin/env bash
# Run one cell's slice of the primary grid.
#
#   CELL=neuron PRECISION=bf16 MODEL=meta-llama/Llama-3.2-3B-Instruct \
#   ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
#   HOURLY_USD=0.758 INSTANCE=inf2.xlarge REPS=3 \
#   tools/run_grid.sh
#
# Primary grid per cell: 5 concurrency x 4 length profiles x REPS.
# (Length profiles: fixed 128 / 512 / 2048 + mixed at mean 512.)
# Across 4 cells x 2 models x 2 precision arms that is ~320 points x REPS.
set -euo pipefail

HARNESS="${HARNESS:-harness/build/harness}"
: "${ENDPOINT:?set ENDPOINT to the cell's chat-completions URL}"
: "${CELL:?set CELL}"; : "${MODEL:?set MODEL}"
PRECISION="${PRECISION:-bf16}"
HOURLY_USD="${HOURLY_USD:-0}"
INSTANCE="${INSTANCE:-unknown}"
REPS="${REPS:-3}"
REQUESTS="${REQUESTS:-60}"
WARMUP="${WARMUP:-6}"
OUT_TOKENS="${OUT_TOKENS:-128}"
mkdir -p results

run_point() {
  local conc="$1" dist="$2" plen="$3" rep="$4"
  local out="results/${CELL}_${PRECISION}_c${conc}_${dist}${plen}_r${rep}.json"
  echo "== $out"
  "$HARNESS" --endpoint "$ENDPOINT" --model "$MODEL" \
    --cell "$CELL" --precision "$PRECISION" --instance "$INSTANCE" \
    --hourly-usd "$HOURLY_USD" \
    --concurrency "$conc" --requests "$REQUESTS" --warmup "$WARMUP" \
    --prompt-tokens "$plen" --distribution "$dist" \
    --output-tokens "$OUT_TOKENS" --seed 42 \
    --out "$out"
}

for rep in $(seq 1 "$REPS"); do
  for conc in 1 4 8 16 32; do
    for spec in "fixed 128" "fixed 512" "fixed 2048" "mixed 512"; do
      run_point "$conc" $spec "$rep"
    done
  done
done

python3 analysis/aggregate.py results/
echo "Grid slice complete. Raw records in results/, tables in analysis/."
