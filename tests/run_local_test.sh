#!/usr/bin/env bash
# End-to-end smoke test: harness vs mock server, then field-level asserts
# on the results record. Requires harness/build/harness (see README).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tools/mock_server.py --port 8902 --decode-ms 2 \
  --prefill-ms-per-token 0.02 > /tmp/inf2l4_test_mock.log 2>&1 &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT
sleep 1

./harness/build/harness \
  --endpoint http://127.0.0.1:8902/v1/chat/completions \
  --cell test --model mock --concurrency 4 --requests 10 --warmup 2 \
  --prompt-tokens 256 --distribution mixed --output-tokens 16 \
  --hourly-usd 0.8048 \
  --out /tmp/inf2l4_test_result.json

python3 - <<'EOF'
import json
r = json.load(open("/tmp/inf2l4_test_result.json"))
cfg, agg = r["config"], r["aggregates"]
assert cfg["cell"] == "test" and cfg["distribution"] == "mixed"
assert agg["requests"] == 10 and agg["requests_ok"] == 10, agg
assert agg["ttft_ms_p50"] > 0 and agg["tpot_ms_p50"] > 0
assert agg["throughput_tokens_per_s"] > 0
assert agg["slo_goodput"] == 1.0
assert agg["cost_per_1m_output_tokens_usd"] > 0
assert len(r["requests"]) == 10
assert all(x["output_tokens"] == 16 for x in r["requests"])
# mixed distribution at mean 256 must actually vary prompt lengths
lengths = {x["prompt_tokens"] for x in r["requests"]}
assert len(lengths) > 1, lengths
print("SMOKE TEST PASSED:", {k: round(v, 2) if isinstance(v, float) else v
                             for k, v in agg.items()})
EOF
