# Inferentia2 vs L4 LLM Serving Characterization — Implementation Plan

**Project 9, extracted from `projects_context.md`.**
Status: proposed topic — not yet confirmed by Gaurav or the professor. The proposal
email draft is owed by **Thu Nov 19, 2026** (professor name/email still blank).
Planned start: **mid-January 2027** (EAD start date Jan 14, 2027).

---

## 1. The project, as defined in projects_context.md

**Problem Statement**

AWS markets Inferentia2 as offering up to 40% better price performance for inference, but no 2024–26 peer-reviewed study tests that claim against a price-matched modern GPU under concurrent, variable-length serving load, with one client across Neuron, CUDA, and Vulkan, and counting the porting cost — ahead-of-time compile time, static batch and sequence buckets, and failed models. Because Neuron compiles static buckets ahead of time, any Inferentia2 advantage should appear at steady, uniform load and erode as the workload departs from those buckets, while CUDA's continuous batching degrades more smoothly.

**Our Solution**

A price-matched characterization of small-LLM serving — Llama-3.2-3B and Llama-3.1-8B — on AWS Inferentia2 (Neuron / vLLM, SDK pinned at 2.28) versus an NVIDIA L4 (vLLM-CUDA, plus llama.cpp CUDA and Vulkan on the same card), measured by one hand-written C++20 harness using Boost.Asio, Boost.Program_options, and Boost.JSON. The study varies concurrency, prompt-length variability, and per-stack quantization across four test cells, and measures time to first token, time per output token, throughput, SLO goodput, cost per 1M output tokens, and compile wall-time, with a perplexity guardrail to catch fast-but-broken quantized configurations. A week-4 kill-gate applies: if the 3B model is not serving on Inferentia2 with the pinned stack, the Neuron arm is dropped and the work pivots to the llama.cpp CUDA-versus-Vulkan serving study.

**Tech Stack**

C++20, Boost.Asio, Boost.Program_options, Boost.JSON, AWS Inferentia2 (Neuron / vLLM, SDK 2.28), NVIDIA L4, vLLM-CUDA, llama.cpp (CUDA and Vulkan), Llama-3.2-3B, Llama-3.1-8B

**Price match (verified 2026-11-12, us-east-1 on-demand):** inf2.xlarge **$0.758/h** vs g6.xlarge (1× L4) **$0.8048/h** — ~6.2% apart, close enough to call price-matched and say so flat-out.

---

## 2. Research questions

- **RQ1 — Steady-state price performance.** At uniform prompt length and steady concurrency, does Inferentia2 actually beat the price-matched L4 on throughput and cost per 1M output tokens, and by how much?
- **RQ2 — Workload-shape sensitivity.** As prompt-length variability departs from Neuron's compiled batch/sequence buckets, how fast does any Inf2 advantage erode, and where is the crossover against CUDA's continuous batching?
- **RQ3 — Quantization per stack.** What does each stack's quantization path (Neuron INT8/FP8 as supported by SDK 2.28, CUDA FP8/AWQ, GGUF quant levels) buy in latency/throughput, and what does it cost in perplexity?
- **RQ4 — Porting cost.** What does it cost to get there at all: AOT compile wall-time per bucket configuration, time-to-first-token-served from a cold start, and model-support failures encountered?

**Pre-registered publishable outcomes** (any one is a paper): (1) a crossover map of where each platform wins; (2) a clean negative result against the vendor price-performance claim; (3) a portability-cost result — the serving numbers are competitive but the compile/bucket tax dominates; (4) the pivot study (CUDA vs Vulkan serving) if the kill-gate fires.

---

## 3. Scope

**In scope**
- 2 models: Llama-3.2-3B, Llama-3.1-8B (both on the Neuron NxD supported list).
- 4 test cells (see §5), one measurement client for all of them.
- Serving load only: concurrent, streaming, variable-length prompts.

**Out of scope (the drop list — say it in the proposal, it builds credibility)**
- OpenGL compute (spec frozen at 4.6; no LLM runtime ships it — GLSL skill transfers to Vulkan/SPIR-V).
- Boost.Compute and Boost.Interprocess (Boost is credible here only as Asio / Program_options / JSON in the harness).
- Trainium as a primary platform (cost and Capacity-Blocks-only availability).
- Hand-written CUDA/Vulkan GEMM kernels as the main bet.
- A100/H100 baselines (different price class; breaks the price match).
- Training, fine-tuning, and offline fixed-batch throughput benchmarks.

---

## 4. The measurement harness (the hand-written C++20 core)

One client measures every stack, so cross-stack differences are the platform's, not the tooling's. Written by hand — this is the niche craft artifact of the project.

**Modules**
1. **Streaming load generator** (Boost.Asio) — speaks the OpenAI-compatible streaming HTTP API (SSE) that vLLM and llama.cpp both expose; maintains N concurrent in-flight requests; timestamps request start, first token, every token, and completion per request.
2. **Workload generator** — deterministic, seeded prompt streams: fixed lengths (128 / 512 / 2048 input tokens) and a **mixed-length distribution at the same mean**, so RQ2 isolates variability, not length. Output length fixed at 128 tokens for the primary grid.
3. **CLI + config** (Boost.Program_options) — cell, model, concurrency, length profile, repetitions, warmup; every run fully described by its command line.
4. **Results writer** (Boost.JSON) — one JSON record per run: full config, environment (instance type, SDK/runtime versions, driver), per-request latencies, aggregates. Raw records are the artifact; analysis never edits them.
5. **Analysis scripts** (Python, offline, not part of the measured path) — aggregate JSON records into the paper's tables and figures; cost per 1M output tokens computed from the verified hourly prices above.

**Validation before any real run**
- Cross-check the harness against `vllm bench serve` on the CUDA cell: same workload, throughput and TTFT must agree within a pre-set tolerance (suggested 5%); investigate, don't shrug, if they don't.
- Warmup runs excluded from measurement in every cell; 3 repetitions per grid point; report spread, not just the mean.
- Server and client on the same instance (loopback) for the primary grid, so network variance is not part of the measurement; note this choice in the paper.

**Repository layout**
```
harness/     C++20 sources (loadgen, workload, config, results) + CMake
cells/       per-cell setup: Dockerfiles / setup scripts, pinned versions
workloads/   prompt-stream generators and frozen workload files
results/     raw JSON run records (never hand-edited)
analysis/    aggregation + figure scripts
paper/       notes, related-work table, draft sections
```

---

## 5. Systems under test — the 4 cells

| Cell | Hardware | Stack | Precision arms |
| --- | --- | --- | --- |
| A — Neuron | inf2.xlarge | vLLM on Neuron, **SDK pinned 2.28**, containerised | BF16; quantized (INT8 vs FP8 — whichever the pinned stack supports per model; confirm at bring-up, §8) |
| B — vLLM-CUDA | g6.xlarge (L4) | vLLM-CUDA | BF16; FP8 or AWQ |
| C — llama.cpp CUDA | g6.xlarge (L4) | llama.cpp server, CUDA backend | GGUF quant levels (e.g. Q8_0, Q4_K_M) |
| D — llama.cpp Vulkan | g6.xlarge (L4) | llama.cpp server, Vulkan backend | Same GGUF quants as C, so C↔D is a pure backend comparison |

Cell notes:
- **Inf2 is legacy**: NxD Inference from Neuron 2.29 supports Trn2+ only — pin **2.28** for Inf2 and containerise the pinned stack during bring-up so it can't drift.
- Neuron compiles static batch/sequence buckets ahead of time; compiles take ~10–45 min and **are billed** — log compile wall-time per configuration (that log is RQ4's data).
- Cells C and D together are also the pivot study (§7): work spent on them is never wasted.

---

## 6. Experiment design

**Variables**
- Concurrency: 1 / 4 / 8 / 16 / 32 in-flight requests.
- Prompt length: 128 / 512 / 2048 tokens fixed, plus the mixed-length distribution at the same mean as the 512 arm.
- Output length: 128 tokens (primary grid).
- Precision: per cell, §5.

**Primary grid:** 4 cells × 2 models × 5 concurrency levels × 4 length profiles × 2 precision arms ≈ **320 measured points × 3 repetitions**. Sub-grids that don't apply to a cell (e.g. a quant the pinned stack doesn't support) are recorded as support failures — that is RQ4 data, not a hole.

**Metrics per point**
- TTFT and TPOT: p50 and p99.
- Throughput (output tokens/s) and request completion rate.
- SLO goodput: fraction of requests meeting pre-set TTFT/TPOT SLO thresholds (fix the thresholds before the first full run; write them into the run config).
- Memory footprint where the stack reports it.
- Cost per 1M output tokens = hourly price ÷ measured output tokens per hour.
- Porting cost (per cell/model, once): compile wall-time, time-to-first-token-served from cold start, failed configurations.

**Quality guardrail**
- WikiText perplexity for every quantized variant vs its BF16 reference on the same stack. A quantized config that wins on speed but blows the perplexity budget is reported as fast-but-broken, never as a win. Set the tolerance before the grid run.

---

## 7. Timeline (part-time, mid-Jan → Jun 2027)

**Phase 0 — Before the start (Nov–Dec 2026, no instances)**
- [ ] Proposal sent (due Nov 19) and topic confirmed with the professor.
- [ ] AWS quota requests filed: Inf `L-1945791B` → 32 vCPUs in us-east-1 + us-west-2; G/VT `L-DB2E81BA` → 16 vCPUs in us-east-1 (default Inf quota is 0; approvals take days). Decision from the Nov 13 quota message still open: file this week or wait for professor confirmation.
- [ ] January: file the AWS Cloud Credit for Research application once the topic is approved (grad students up to $5,000; review can take 90–120 days — the pilot must not depend on it).

**Weeks 1–2 — Harness skeleton + CUDA cell first**
- [ ] Harness modules 1–4 compiling and running against a local mock server.
- [ ] Cell B (vLLM-CUDA) up on g6.xlarge; harness validated against `vllm bench serve`.
- [ ] Cells C and D built (llama.cpp CUDA + Vulkan) — they share the L4 box.
- Done when: one full pilot point (3B, concurrency 8, 512 fixed) measured end-to-end on B, C, D with JSON records written.

**Weeks 3–4 — Neuron bring-up (the risk window)**
- [ ] Cell A: containerised Neuron 2.28 stack on inf2.xlarge; Llama-3.2-3B compiled and serving.
- [ ] Confirm which quantization the pinned 2.28 stack supports for 3B and 8B; log every failed configuration.
- [ ] **KILL-GATE (end of week 4):** if 3B is not serving on Inf2 with the pinned stack, drop the Neuron arm and pivot to Backup 1 — the llama.cpp CUDA-vs-Vulkan serving characterization on the L4 — with an "attempted port" section in the paper. Cells B–D work already done carries over in full.

**Month 2 — Pilot grid**
- [ ] Full grid on 3B only, all surviving cells; sanity-check the crossover shape against the hypothesis.
- [ ] Fix SLO thresholds and the perplexity tolerance in writing before the full run.
- Done when: pilot tables exist and no metric definition is still moving.

**Months 2–4 — Full grid + RQ2/RQ3 depth**
- [ ] 8B grid; mixed-length distribution experiments (RQ2); quantization arms + perplexity guardrails (RQ3).
- [ ] Compile/porting-cost log completed for every cell and model (RQ4).
- [ ] Instances stopped between sessions; spend tracked against the $250–550 budget.

**Month 5 — Replication + artifact + writing**
- [ ] Replication rerun of the headline points; artifact packaged (harness source, workload files, raw JSON records, analysis scripts).
- [ ] Paper drafted: crossover map figure, per-cell tables, porting-cost table, related-work gap table.

**Months 5–6 — Submission**
- [ ] Submit to the best-fit venue with a posted CFP (est. from 2026 cycles: IISWC 2027 ~May 21, SYSTOR 2027 short paper ~May 21, APSys 2027 May–Jun — all 2027 CFPs unposted when this plan was written; re-verify before targeting).
- [ ] arXiv preprint on submission. Optional early milestone: ICPE 2027 Posters/Demos (verified deadline Feb 8, 2027) with a pilot/harness poster only.
- Not targets: MLSys 2027 main (deadline was Oct 30, 2026) and ICPE 2027 research track (Nov 16, 2026) — both close before this project starts.

---

## 8. Risks and open questions

| Risk / unknown | Handling |
| --- | --- |
| Inf2 capacity: documented multi-AZ failures to launch inf2.xlarge | Request quota in us-east-1 **and** us-west-2; try both regions at bring-up; log failures (also RQ4 colour) |
| Default Inferentia quota is 0 | Phase 0 quota requests; nothing launches before approval |
| Compile time is billed (10–45 min per bucket config) | Batch compile decisions; never leave instances running between sessions; budget line $250–550 total, ~$100 pilot out-of-pocket |
| Which quantization SDK 2.28 supports for 3B/8B is unconfirmed | Week 3 bring-up is the confirmation point; the grid records unsupported arms as support failures |
| Neuron stack drifts (2.29 drops Inf2) | Pin 2.28 and containerise in week 3 |
| Research credits arrive late (90–120 day review) | File in January; pilot budgeted out-of-pocket; credits fund the full grid, not the start |
| Professor has not yet approved the topic | Everything before Phase 0's first checkbox is proposal-stage only; no spend until approval |

---

## 9. Definition of done

- [ ] Harness validated against `vllm bench serve` on the CUDA cell.
- [ ] Kill-gate decision made and recorded at end of week 4.
- [ ] ~320-point grid measured ×3 repetitions, raw JSON records preserved.
- [ ] Perplexity guardrail reported for every quantized variant.
- [ ] Porting-cost table complete (compile time, cold-start, failures).
- [ ] Artifact packaged so a reader can re-run a headline point.
- [ ] Paper submitted + arXiv preprint posted.
