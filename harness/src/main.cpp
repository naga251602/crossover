// Inferentia2 vs L4 LLM serving characterization — measurement harness.
//
// One hand-written C++20 client measures every stack (Neuron/vLLM,
// vLLM-CUDA, llama.cpp CUDA, llama.cpp Vulkan) through the same
// OpenAI-compatible streaming API, so cross-stack differences belong to the
// platform, not the tooling. Every run writes one Boost.JSON record: full
// config, per-request timings, and aggregates. Raw records are the artifact;
// analysis never edits them.
#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>

#include "client.hpp"
#include "config.hpp"
#include "metrics.hpp"
#include "workload.hpp"

namespace net = boost::asio;
namespace json = boost::json;

struct Endpoint {
    std::string host, port, path;
};

static Endpoint parse_endpoint(const std::string& url) {
    // http://host[:port]/path — loopback or same-instance endpoints only.
    static const std::regex re(R"(^https?://([^/:]+)(?::(\d+))?(/.*)?$)");
    std::smatch m;
    if (!std::regex_match(url, m, re)) {
        std::cerr << "error: cannot parse endpoint URL: " << url << "\n";
        std::exit(2);
    }
    Endpoint ep;
    ep.host = m[1].str();
    ep.port = m[2].matched ? m[2].str() : "80";
    ep.path = m[3].matched ? m[3].str() : "/";
    return ep;
}

static net::awaitable<void> worker(net::io_context& ioc, const Config& cfg,
                                   const Endpoint& ep, Workload& workload,
                                   std::atomic<int>& remaining,
                                   std::vector<RequestRecord>& records,
                                   std::mutex& records_mu, bool warmup) {
    while (remaining.fetch_sub(1) > 0) {
        auto [prompt, nominal] = workload.next();
        auto rec = co_await run_request(ioc, ep.host, ep.port, ep.path,
                                        cfg.api_key, cfg.model, prompt,
                                        nominal, cfg.output_tokens);
        if (!warmup) {
            std::lock_guard<std::mutex> lock(records_mu);
            records.push_back(rec);
        }
    }
}

static json::object summarize(const std::vector<RequestRecord>& records,
                              const Config& cfg, double wall_seconds) {
    std::vector<double> ttft, tpot, latency;
    long total_output_tokens = 0;
    int ok = 0, slo_ok = 0;
    for (const auto& r : records) {
        if (!r.ok) continue;
        ++ok;
        ttft.push_back(r.ttft_ms);
        tpot.push_back(r.tpot_ms);
        latency.push_back(r.latency_ms);
        total_output_tokens += r.output_tokens;
        if (r.ttft_ms <= cfg.slo_ttft_ms && r.tpot_ms <= cfg.slo_tpot_ms)
            ++slo_ok;
    }
    json::object agg;
    agg["requests"] = static_cast<int>(records.size());
    agg["requests_ok"] = ok;
    agg["ttft_ms_p50"] = percentile(ttft, 50);
    agg["ttft_ms_p99"] = percentile(ttft, 99);
    agg["tpot_ms_p50"] = percentile(tpot, 50);
    agg["tpot_ms_p99"] = percentile(tpot, 99);
    agg["latency_ms_p50"] = percentile(latency, 50);
    agg["latency_ms_p99"] = percentile(latency, 99);
    agg["output_tokens"] = total_output_tokens;
    agg["wall_seconds"] = wall_seconds;
    double tps = wall_seconds > 0 ? total_output_tokens / wall_seconds : 0.0;
    agg["throughput_tokens_per_s"] = tps;
    agg["slo_goodput"] = ok ? static_cast<double>(slo_ok) / ok : 0.0;
    if (cfg.hourly_usd > 0 && tps > 0) {
        // cost per 1M output tokens = hourly price / tokens per hour * 1e6
        agg["cost_per_1m_output_tokens_usd"] = cfg.hourly_usd / (tps * 3600.0) * 1e6;
    }
    return agg;
}

int main(int argc, char** argv) {
    Config cfg = parse_config(argc, argv);
    Endpoint ep = parse_endpoint(cfg.endpoint);

    net::io_context ioc;
    Workload workload(cfg.seed, cfg.prompt_tokens, cfg.distribution);
    std::vector<RequestRecord> records;
    std::mutex records_mu;

    auto run_phase = [&](int count, bool warmup) {
        if (count <= 0) return 0.0;
        std::atomic<int> remaining{count};
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < cfg.concurrency; ++i) {
            net::co_spawn(ioc,
                          worker(ioc, cfg, ep, workload, remaining, records,
                                 records_mu, warmup),
                          net::detached);
        }
        ioc.restart();
        ioc.run();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
            .count();
    };

    run_phase(cfg.warmup, /*warmup=*/true);
    double wall = run_phase(cfg.requests, /*warmup=*/false);

    json::object root;
    json::object config_json;
    config_json["cell"] = cfg.cell;
    config_json["precision"] = cfg.precision;
    config_json["instance"] = cfg.instance;
    config_json["hourly_usd"] = cfg.hourly_usd;
    config_json["endpoint"] = cfg.endpoint;
    config_json["model"] = cfg.model;
    config_json["concurrency"] = cfg.concurrency;
    config_json["requests"] = cfg.requests;
    config_json["warmup"] = cfg.warmup;
    config_json["prompt_tokens"] = cfg.prompt_tokens;
    config_json["distribution"] = cfg.distribution;
    config_json["output_tokens"] = cfg.output_tokens;
    config_json["seed"] = cfg.seed;
    config_json["slo_ttft_ms"] = cfg.slo_ttft_ms;
    config_json["slo_tpot_ms"] = cfg.slo_tpot_ms;
    root["config"] = config_json;
    root["aggregates"] = summarize(records, cfg, wall);

    json::array per_request;
    for (const auto& r : records) {
        json::object o;
        o["prompt_tokens"] = r.prompt_tokens;
        o["output_tokens"] = r.output_tokens;
        o["ttft_ms"] = r.ttft_ms;
        o["tpot_ms"] = r.tpot_ms;
        o["latency_ms"] = r.latency_ms;
        o["ok"] = r.ok;
        per_request.push_back(o);
    }
    root["requests"] = per_request;

    std::ofstream out(cfg.out);
    if (!out) {
        std::cerr << "error: cannot write " << cfg.out << "\n";
        return 1;
    }
    out << json::serialize(root) << "\n";

    auto agg = root["aggregates"].as_object();
    std::cout << "cell=" << cfg.cell << " precision=" << cfg.precision
              << " concurrency=" << cfg.concurrency
              << " dist=" << cfg.distribution
              << " ok=" << agg["requests_ok"].as_int64() << "/"
              << agg["requests"].as_int64()
              << " ttft_p50=" << agg["ttft_ms_p50"].as_double()
              << "ms tpot_p50=" << agg["tpot_ms_p50"].as_double()
              << "ms thr=" << agg["throughput_tokens_per_s"].as_double()
              << " tok/s goodput=" << agg["slo_goodput"].as_double()
              << " -> " << cfg.out << "\n";
    return agg["requests_ok"].as_int64() > 0 ? 0 : 1;
}
