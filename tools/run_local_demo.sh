#!/usr/bin/env bash
# Local validation demo — no GPU, no cloud, no spend.
# Builds nothing; expects harness/build/harness to exist (see README).
# Starts the mock server, sweeps concurrency, aggregates, and cleans up.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tools/mock_server.py --port 8901 --decode-ms 5 \
  --prefill-ms-per-token 0.05 > /tmp/inf2l4_mock.log 2>&1 &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT
sleep 1

for conc in 1 8 32; do
  ./harness/build/harness \
    --endpoint http://127.0.0.1:8901/v1/chat/completions \
    --cell local-mock --model mock-3b --instance local --hourly-usd 0 \
    --concurrency "$conc" --requests 24 --warmup 4 \
    --prompt-tokens 512 --distribution fixed --output-tokens 64 \
    --out "results/demo_c${conc}.json"
done

python3 analysis/aggregate.py results/
echo "Demo done — see analysis/summary.md (mock numbers, methodology only)."
