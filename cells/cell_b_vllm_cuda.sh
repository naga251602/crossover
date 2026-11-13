#!/usr/bin/env bash
# Cell B — vLLM-CUDA on an NVIDIA L4. Run on: g6.xlarge.
# Precision arm is selected with PRECISION=bf16|fp8|awq (quant flag differs).
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
PRECISION="${PRECISION:-bf16}"

pip install --quiet vllm

QUANT_ARGS=()
case "$PRECISION" in
  bf16) ;;
  fp8)  QUANT_ARGS=(--quantization fp8) ;;
  awq)  QUANT_ARGS=(--quantization awq) ;;
  *) echo "unknown PRECISION=$PRECISION"; exit 2 ;;
esac

echo "Serving $MODEL ($PRECISION) on :8000"
nohup vllm serve "$MODEL" --port 8000 "${QUANT_ARGS[@]}" \
  > /tmp/cell_b_vllm.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 10; done
echo "Cell B ready: http://127.0.0.1:8000/v1/chat/completions"
echo "Cross-check reminder: run 'vllm bench serve' on the same workload and"
echo "compare against the harness before trusting any grid numbers."
