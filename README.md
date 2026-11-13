# Inferentia2 vs L4 — LLM Serving Characterization

**When does a purpose-built inference chip actually win?** A price-matched
characterization of small-LLM serving (Llama-3.2-3B / 3.1-8B) on AWS
Inferentia2 (Neuron/vLLM, SDK pinned 2.28) vs NVIDIA L4 (vLLM-CUDA, plus
llama.cpp CUDA and Vulkan on the same card) — measured by one hand-written
C++20 harness across all four stacks.

Price match (us-east-1, verified 2026-11-12): inf2.xlarge **$0.758/h** vs
g6.xlarge **$0.8048/h** (~6.2% apart).

Full design: `inferentia2-vs-l4-llm-serving-characterization_implementation_plan.md`
(the implementation plan this repo implements) and `paper/notes.md`.

## Status — read this first

- ✅ **Implemented and validated locally:** the C++20 harness compiles and
  runs; it was verified end-to-end against the bundled mock server
  (`tests/run_local_test.sh` passes; `tools/run_local_demo.sh` produces
  aggregated tables). The numbers in `results/demo_*.json` are **mock
  numbers — methodology validation only, not measurements of any real
  stack.**
- ⏳ **Not yet run:** the AWS experiment grid. That needs the professor's
  topic approval (proposal due Nov 19, 2026), the Inferentia/G quota
  increases, and real instance spend ($250–550 budget). The cell scripts,
  grid runner, and analysis pipeline are ready for that phase, planned
  mid-Jan → Jun 2027, with the **week-4 kill-gate**: if 3B isn't serving
  on Inf2 by end of week 4, drop the Neuron arm and pivot to the
  llama.cpp CUDA-vs-Vulkan study (cells C vs D — work already carries over).

## The harness

One client for every stack, so differences belong to the platform:

- **Streaming load generator** (Boost.Asio, C++20 coroutines) — speaks the
  OpenAI-compatible SSE streaming API that vLLM and llama.cpp share;
  timestamps start / first token / completion per request.
- **Workload generator** — seeded, deterministic prompt streams. `fixed`
  lengths (128/512/2048) and a `mixed` distribution holding the *mean*
  constant while variability changes (RQ2's whole point).
- **Config** (Boost.Program_options) — every run fully described by its
  command line; SLO thresholds are inputs, fixed before the run.
- **Results** (Boost.JSON) — one record per run: full config, per-request
  TTFT/TPOT/latency, aggregates (p50/p99, throughput, SLO goodput, cost
  per 1M output tokens at the instance's hourly price). Raw records are
  the artifact; analysis never edits them.

## Build & local validation

```bash
# Dependencies (Ubuntu/Debian): Boost >= 1.80 (program_options, json, asio)
sudo apt install libboost-all-dev cmake g++

cmake -S harness -B harness/build -DCMAKE_BUILD_TYPE=Release
cmake --build harness/build -j
bash tests/run_local_test.sh     # harness vs mock server + field asserts
bash tools/run_local_demo.sh     # concurrency sweep -> analysis/summary.md
```

## Running a real cell (AWS phase)

```bash
# On the instance, after starting a cell (see cells/README.md):
bash cells/cell_b_vllm_cuda.sh                    # or cell_a/c/d
CELL=vllm-cuda PRECISION=bf16 MODEL=meta-llama/Llama-3.2-3B-Instruct \
ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
HOURLY_USD=0.8048 INSTANCE=g6.xlarge REPS=3 \
  tools/run_grid.sh
python3 analysis/aggregate.py results/            # -> summary.csv / .md
python3 analysis/perplexity_guardrail.py ...      # per quantized variant
```

Before trusting grid numbers on cell B, cross-check the harness against
`vllm bench serve` on the same workload (plan §4: agree within ~5%).

## Research questions

- **RQ1** — steady-state price performance vs the vendor claim
- **RQ2** — how fast the Inf2 advantage erodes as prompt lengths depart
  from compiled buckets (fixed → mixed at constant mean)
- **RQ3** — per-stack quantization: speed bought vs perplexity paid
  (`analysis/perplexity_guardrail.py` is the guardrail)
- **RQ4** — porting cost: AOT compile wall-time, cold start, failed
  configurations (log them — failures are data)

## Layout

```
harness/    C++20 harness (Asio loadgen, workload, config, results)
cells/      per-cell setup scripts + rules (pins, kill-gate, billing)
tools/      mock server, grid runner, local demo, workload freezer
analysis/   aggregate.py (records -> tables), perplexity_guardrail.py
workloads/  workload docs + frozen sample stream (seed 42)
results/    raw JSON run records (demo/mock records included, labeled)
paper/      notes: claim, hypothesis, gap, figures, venue reality
tests/      end-to-end smoke test against the mock server
```
