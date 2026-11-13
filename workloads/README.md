# Workloads

The harness generates its prompt stream internally from `--seed`
(deterministic; identical across cells for the same seed — that is what
makes the comparison fair). Length profiles:

- `fixed` — every prompt exactly `--prompt-tokens` long.
- `mixed` — lengths drawn from {mean/2, mean, 3·mean/2} with equal
  probability, so the mean stays at `--prompt-tokens` while variability
  changes (RQ2 isolates variability, not length).

Token counts are nominal (1 word ≈ 1 token for the filler vocabulary) and
are recorded per request in every results file. To inspect or replay a
stream outside the harness (e.g. for a `vllm bench serve` cross-check),
freeze it with:

```bash
python3 tools/gen_workloads.py --seed 42 --count 40 --prompt-tokens 512 \
  --distribution mixed --out workloads/mixed_512.jsonl
```

A frozen sample (`mixed_512.jsonl`, seed 42) is included. For the
perplexity guardrail you also need a real text file (one passage per
line, e.g. a WikiText sample) — supply your own at
`workloads/wikitext_sample.txt`; it is not redistributed here.
