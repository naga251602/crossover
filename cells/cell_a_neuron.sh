#!/usr/bin/env bash
# Cell A — vLLM on AWS Inferentia2 (Neuron), SDK PINNED at 2.28.
# Run on: inf2.xlarge (us-east-1 or us-west-2). Requires the Inferentia
# quota increase (Service Quotas code L-1945791B) to have been approved.
#
# Neuron 2.29+ drops Inf2 from NxD Inference, so this cell runs inside the
# pinned AWS Neuron container. AOT compilation of batch/sequence buckets
# happens on first serve and IS BILLED — watch the clock, log compile times.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
# The official vLLM-Neuron image for SDK 2.28. If AWS republishes tags,
# keep the 2.28 pin — that pin is a methodological requirement, not a detail.
IMAGE="${NEURON_IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:2.28.0}"

echo "Pulling pinned Neuron image: $IMAGE"
docker pull "$IMAGE"

echo "Serving $MODEL on :8000 (compile buckets on first start — timed)"
date +%s > /tmp/cell_a_compile_start
docker run --rm -d --name cell-a \
  --device=/dev/neuron0 \
  -p 8000:8000 \
  -e NEURON_RT_NUM_CORES="${NEURON_CORES:-1}" \
  "$IMAGE" \
  vllm serve "$MODEL" --port 8000

echo "Waiting for the endpoint (compilation can take 10-45 min)..."
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 30; done
echo "compile+load wall time: $(( $(date +%s) - $(cat /tmp/cell_a_compile_start) ))s — log this for RQ4"
echo "Cell A ready: http://127.0.0.1:8000/v1/chat/completions"
