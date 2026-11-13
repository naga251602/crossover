#pragma once
#include <string>

struct Config {
    // Target server (any OpenAI-compatible streaming endpoint: vLLM, llama.cpp)
    std::string endpoint = "http://127.0.0.1:8000/v1/chat/completions";
    std::string model = "meta-llama/Llama-3.2-3B-Instruct";
    std::string api_key = "";  // sent as Bearer token when non-empty

    // Experiment identity — written verbatim into the results record.
    std::string cell = "local-mock";        // e.g. neuron, vllm-cuda, llamacpp-cuda, llamacpp-vulkan
    std::string precision = "bf16";         // bf16 | fp8 | awq | q8_0 | q4_k_m | int8
    std::string instance = "local";         // e.g. inf2.xlarge, g6.xlarge
    double hourly_usd = 0.0;                // instance price for cost/1M tokens

    // Workload shape
    int concurrency = 8;
    int requests = 40;                      // measured requests
    int warmup = 4;                         // warmup requests (excluded)
    int prompt_tokens = 512;                // fixed length, or mean for "mixed"
    std::string distribution = "fixed";     // fixed | mixed
    int output_tokens = 128;
    unsigned seed = 42;

    // SLO thresholds for goodput (fixed before the run, recorded in results)
    double slo_ttft_ms = 2000.0;
    double slo_tpot_ms = 100.0;

    std::string out = "results/run.json";
};

// Parses argc/argv with Boost.Program_options. Exits the process on --help
// or on a malformed option, like any well-behaved CLI.
Config parse_config(int argc, char** argv);
