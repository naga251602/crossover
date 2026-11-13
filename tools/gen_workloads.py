#!/usr/bin/env python3
"""Freeze a workload file (JSONL) for inspection / reuse.

The C++ harness generates the identical stream internally from --seed; this
script exists so a reviewer can see exactly what a run sent, and so other
tools (e.g. `vllm bench serve` cross-checks) can replay the same prompts.

    python3 tools/gen_workloads.py --seed 42 --count 40 \
        --prompt-tokens 512 --distribution mixed --out workloads/mixed_512.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

VOCAB = ("storage kernel latency throughput buffer cache scheduler request "
         "response network memory token vector index query server client "
         "stream block page device driver queue worker thread lock batch "
         "model inference decode prefill attention").split()


def make_prompt(rng: random.Random, tokens: int) -> str:
    words = [rng.choice(VOCAB) for _ in range(tokens)]
    return "Summarize the following system notes in one paragraph: " + " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--distribution", choices=["fixed", "mixed"], default="fixed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.out, "w") as fh:
        for _ in range(args.count):
            length = args.prompt_tokens
            if args.distribution == "mixed":
                length = rng.choice([args.prompt_tokens // 2,
                                     args.prompt_tokens,
                                     3 * args.prompt_tokens // 2])
            fh.write(json.dumps({"prompt": make_prompt(rng, length),
                                 "prompt_tokens_nominal": length}) + "\n")
    print(f"wrote {args.count} prompts -> {args.out}")


if __name__ == "__main__":
    main()
