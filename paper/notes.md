# Paper notes — Inferentia2 vs L4 LLM serving characterization

**Working title:** *When does a purpose-built inference chip actually win?*

## Claim under test
AWS markets Inferentia2 at "up to 40% better price performance." No
peer-reviewed 2024–26 study tests that against a price-matched modern GPU
under concurrent, variable-length serving load, with one client across
Neuron, CUDA and Vulkan, counting porting cost.

## Hypothesis (mechanistic)
Neuron AOT-compiles static batch/sequence buckets, so any Inf2 advantage
appears at steady, uniform load and erodes as the workload departs from the
compiled buckets; CUDA continuous batching degrades smoothly. Either a
crossover map or a clean negative result is publishable.

## Research questions
- RQ1 steady-state price performance (throughput, cost / 1M output tokens)
- RQ2 workload-shape sensitivity (fixed -> mixed length at constant mean)
- RQ3 per-stack quantization (speed bought vs perplexity paid)
- RQ4 porting cost (compile wall-time, cold start, support failures)

## Gap vs prior work (from the 2026-11-12 report packet)
- LLM-Inference-Bench (SC'24): A100/H100/MI250/Gaudi2 — no Neuron, no
  Vulkan, no price matching.
- Closest prior: an unreviewed thesis comparing T4 vs inf2 (2018 GPU
  comparator, no quantization or length-variance axis).
- Vendor papers showcase optimizations; they do not price-match or
  stress bucket departure.

## Figures to produce (from analysis/)
1. Crossover map: winner by (concurrency x length-variability), per model.
2. TTFT/TPOT p50/p99 vs concurrency, per cell.
3. Cost per 1M output tokens vs concurrency.
4. Porting-cost table: compile time, cold start, failed configs per cell.
5. Perplexity deltas for every quantized arm vs its BF16 reference.

## Venue reality (re-verify before targeting)
Primary fits, 2027 CFPs unposted at plan time, dates estimated from 2026
cycles: IISWC 2027 (~May 21), SYSTOR 2027 short paper (~May 21),
APSys 2027 (May–Jun). Optional early milestone: ICPE 2027 Posters/Demos
(verified Feb 8, 2027). Not available to this timeline: MLSys 2027 main
(Oct 30, 2026) and ICPE 2027 research (Nov 16, 2026) — both close before
the January 2027 start.
