#!/usr/bin/env bash
# Cell D — llama.cpp server, Vulkan backend, on the SAME L4 and the SAME
# GGUF files as cell C, so C vs D is a pure backend comparison.
# Requires Vulkan headers/loader on the instance (libvulkan-dev, mesa or
# NVIDIA Vulkan ICD) — install before building if the build cannot find Vulkan.
set -euo pipefail

MODEL_GGUF="${MODEL_GGUF:?set MODEL_GGUF to the .gguf file for this arm}"

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
cmake -S llama.cpp -B llama.cpp/build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-vulkan -j --target llama-server

echo "Serving $MODEL_GGUF (Vulkan) on :8080"
nohup ./llama.cpp/build-vulkan/bin/llama-server -m "$MODEL_GGUF" \
  --port 8080 -ngl 99 > /tmp/cell_d_llamacpp.log 2>&1 &
until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 5; done
echo "Cell D ready: http://127.0.0.1:8080/v1/chat/completions"
