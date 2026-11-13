# Test cells

Four cells, one harness. Every cell exposes an OpenAI-compatible streaming
endpoint on loopback; the harness runs on the same instance (§4 of the
implementation plan), so the network is not part of the measurement.

| Cell | Instance | Script | Endpoint |
| --- | --- | --- | --- |
| A — Neuron / vLLM | inf2.xlarge ($0.758/h) | `cell_a_neuron.sh` | :8000 |
| B — vLLM-CUDA | g6.xlarge ($0.8048/h) | `cell_b_vllm_cuda.sh` | :8000 |
| C — llama.cpp CUDA | g6.xlarge | `cell_c_llamacpp_cuda.sh` | :8080 |
| D — llama.cpp Vulkan | g6.xlarge | `cell_d_llamacpp_vulkan.sh` | :8080 |

Prices: us-east-1 on-demand, verified 2026-11-12 (inf2.xlarge $0.758/h,
g6.xlarge $0.8048/h — ~6.2% apart, the price match the study stands on).

## Rules that protect the study

1. **Pin everything.** Neuron SDK **2.28** for Inf2 (2.29+ drops Inf2 from
   NxD Inference) — cell A runs in the pinned container for exactly this
   reason. Record versions in the run notes.
2. **Compiles are billed.** Neuron AOT compiles take ~10–45 min per bucket
   configuration. Log start/stop for every compile — that log is RQ4's data.
3. **Stop instances between sessions.** Never leave an inf2/g6 instance
   running overnight; the $250–550 budget assumes stopped-between-sessions.
4. **Week-4 kill-gate.** If Llama-3.2-3B is not serving on Inf2 with the
   pinned stack by end of week 4, drop cell A and pivot to the llama.cpp
   CUDA-vs-Vulkan study (cells C vs D) with an "attempted port" section.
   Record every failed configuration on the way — failures are data.
5. **Failed configs are results.** A precision arm a stack does not support
   goes in the grid as a support failure, not a hole.
