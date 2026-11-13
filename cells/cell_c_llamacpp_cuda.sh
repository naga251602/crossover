#!/usr/bin/env bash
# Cell C — llama.cpp server, CUDA backend, on the same L4 (g6.xlarge).
# GGUF quant is chosen by the model file you point MODEL_GGUF at
# (e.g. Q8_0 for the bf16-comparable arm, Q4_K_M for the quant arm).
set -euo pipefail

MODEL_GGUF="${MODEL_GGUF:?set MODEL_GGUF to the .gguf file for this arm}"

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j --target llama-server

echo "Serving $MODEL_GGUF (CUDA) on :8080"
nohup ./llama.cpp/build/bin/llama-server -m "$MODEL_GGUF" \
  --port 8080 -ngl 99 > /tmp/cell_c_llamacpp.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 5; done
echo "Cell C ready: http://127.0.0.1:8080/v1/chat/completions"
