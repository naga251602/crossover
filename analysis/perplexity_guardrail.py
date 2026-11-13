#!/usr/bin/env python3
"""Perplexity guardrail for quantized variants (RQ3).

Scores a fixed text file (default: a WikiText sample you supply, one passage
per line) through an OpenAI-compatible /v1/completions endpoint with
echo + logprobs, and reports mean perplexity. Run it for the BF16 reference
and for every quantized variant on the same stack: a variant whose perplexity
blows past the tolerance you set is fast-but-broken, never a win.

    python3 analysis/perplexity_guardrail.py \
        --base-url http://127.0.0.1:8000/v1 --model <served-model> \
        --text workloads/wikitext_sample.txt --label q4_k_m

Requires a server that returns logprobs for echoed prompts (vLLM does).
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request


def score(base_url: str, model: str, text: str, api_key: str) -> float | None:
    body = json.dumps({
        "model": model, "prompt": text, "max_tokens": 0,
        "echo": True, "logprobs": 1, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key or 'not-needed'}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    logprobs = data["choices"][0].get("logprobs", {})
    token_logprobs = [lp for lp in logprobs.get("token_logprobs", [])
                      if lp is not None]
    if not token_logprobs:
        return None
    return math.exp(-sum(token_logprobs) / len(token_logprobs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, help="one passage per line")
    ap.add_argument("--label", default="variant")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    passages = [ln.strip() for ln in open(args.text) if ln.strip()]
    scores = []
    for passage in passages:
        ppl = score(args.base_url, args.model, passage, args.api_key)
        if ppl is not None:
            scores.append(ppl)
    if not scores:
        print("no logprobs returned by the server — guardrail cannot run")
        raise SystemExit(1)
    result = {"label": args.label, "model": args.model,
              "passages": len(scores),
              "perplexity_mean": round(sum(scores) / len(scores), 3)}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
