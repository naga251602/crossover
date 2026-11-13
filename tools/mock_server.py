#!/usr/bin/env python3
"""Mock OpenAI-compatible streaming server for local harness validation.

NOT a model. It exists so the C++ harness can be validated end-to-end with no
GPU and no cloud spend: it speaks SSE chat completions, charges a simulated
prefill cost proportional to prompt length, then streams tokens at a fixed
decode interval. Concurrency is real (threaded), so queueing effects show up.

    python3 tools/mock_server.py --port 8901 --decode-ms 5 --prefill-ms-per-token 0.05
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    decode_ms = 5.0
    prefill_ms_per_token = 0.05


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep stdout clean
        pass

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok","server":"mock"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        max_tokens = int(payload.get("max_tokens", 128))
        prompt = ""
        for msg in payload.get("messages", []):
            prompt += msg.get("content", "")
        prompt_words = max(1, len(prompt.split()))

        time.sleep(State.prefill_ms_per_token * prompt_words / 1000.0)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(content: str, finish: str | None = None) -> bytes:
            obj = {"choices": [{"delta": {"content": content},
                                "finish_reason": finish}]}
            return f"data: {json.dumps(obj)}\n\n".encode()

        try:
            for i in range(max_tokens):
                self.wfile.write(chunk("tok "))
                self.wfile.flush()
                if State.decode_ms:
                    time.sleep(State.decode_ms / 1000.0)
            self.wfile.write(chunk("", "stop"))
            usage = {"choices": [], "usage": {"completion_tokens": max_tokens}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--decode-ms", type=float, default=5.0)
    ap.add_argument("--prefill-ms-per-token", type=float, default=0.05)
    args = ap.parse_args()
    State.decode_ms = args.decode_ms
    State.prefill_ms_per_token = args.prefill_ms_per_token
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock server on 127.0.0.1:{args.port} "
          f"(decode {args.decode_ms} ms/tok, prefill {args.prefill_ms_per_token} ms/tok)")
    server.serve_forever()


if __name__ == "__main__":
    main()
