#include "config.hpp"

#include <boost/program_options.hpp>
#include <cstdlib>
#include <iostream>

namespace po = boost::program_options;

Config parse_config(int argc, char** argv) {
    Config c;
    po::options_description desc("Inferentia2 vs L4 serving characterization harness");
    desc.add_options()
        ("help", "show this help")
        ("endpoint", po::value(&c.endpoint), "OpenAI-compatible chat completions URL")
        ("model", po::value(&c.model), "model name served by the endpoint")
        ("api-key", po::value(&c.api_key), "bearer token (optional)")
        ("cell", po::value(&c.cell), "cell label: neuron|vllm-cuda|llamacpp-cuda|llamacpp-vulkan")
        ("precision", po::value(&c.precision), "precision arm label")
        ("instance", po::value(&c.instance), "instance type label")
        ("hourly-usd", po::value(&c.hourly_usd), "instance hourly price in USD")
        ("concurrency", po::value(&c.concurrency), "in-flight requests")
        ("requests", po::value(&c.requests), "measured requests")
        ("warmup", po::value(&c.warmup), "warmup requests (excluded from results)")
        ("prompt-tokens", po::value(&c.prompt_tokens), "fixed prompt length, or mean for mixed")
        ("distribution", po::value(&c.distribution), "fixed | mixed")
        ("output-tokens", po::value(&c.output_tokens), "requested output tokens")
        ("seed", po::value(&c.seed), "workload RNG seed")
        ("slo-ttft-ms", po::value(&c.slo_ttft_ms), "TTFT SLO threshold for goodput")
        ("slo-tpot-ms", po::value(&c.slo_tpot_ms), "TPOT SLO threshold for goodput")
        ("out", po::value(&c.out), "results JSON output path");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        std::exit(0);
    }
    if (c.distribution != "fixed" && c.distribution != "mixed") {
        std::cerr << "error: --distribution must be 'fixed' or 'mixed'\n";
        std::exit(2);
    }
    if (c.concurrency < 1 || c.requests < 1) {
        std::cerr << "error: --concurrency and --requests must be >= 1\n";
        std::exit(2);
    }
    return c;
}
