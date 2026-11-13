#pragma once
#include <boost/asio.hpp>
#include <string>

#include "metrics.hpp"

namespace net = boost::asio;

// One streaming chat-completions request against an OpenAI-compatible
// endpoint (vLLM and llama.cpp both speak this protocol over SSE).
//
// Timestamps, all from a steady clock:
//   t0        request bytes sent
//   first     first SSE chunk carrying content -> TTFT = first - t0
//   each      every content chunk is one observed output token
//   done      stream closed -> latency = done - t0,
//             TPOT = (done - first) / (tokens - 1)
net::awaitable<RequestRecord> run_request(net::io_context& ioc,
                                          const std::string& host,
                                          const std::string& port,
                                          const std::string& path,
                                          const std::string& api_key,
                                          const std::string& model,
                                          const std::string& prompt,
                                          int prompt_tokens,
                                          int max_tokens);
