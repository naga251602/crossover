#include "workload.hpp"

#include <sstream>

Workload::Workload(unsigned seed, int prompt_tokens, std::string distribution)
    : rng_(seed), mean_tokens_(prompt_tokens), mixed_(distribution == "mixed") {
    // Neutral filler vocabulary: no cell gets a content advantage, and the
    // stream is identical across cells for the same seed.
    vocab_ = {"storage", "kernel", "latency", "throughput", "buffer", "cache",
              "scheduler", "request", "response", "network", "memory", "token",
              "vector", "index", "query", "server", "client", "stream", "block",
              "page", "device", "driver", "queue", "worker", "thread", "lock",
              "batch", "model", "inference", "decode", "prefill", "attention"};
}

std::string Workload::make_prompt(int tokens) {
    // A light instruction prefix keeps the prompt shaped like a real request;
    // the filler body sets the length.
    std::ostringstream oss;
    oss << "Summarize the following system notes in one paragraph: ";
    std::uniform_int_distribution<size_t> pick(0, vocab_.size() - 1);
    for (int i = 0; i < tokens; ++i) {
        oss << vocab_[pick(rng_)] << ' ';
    }
    return oss.str();
}

std::pair<std::string, int> Workload::next() {
    int length = mean_tokens_;
    if (mixed_) {
        // {mean/2, mean, 3*mean/2} with equal probability -> mean is preserved.
        std::uniform_int_distribution<int> which(0, 2);
        const int options[3] = {mean_tokens_ / 2, mean_tokens_,
                                (3 * mean_tokens_) / 2};
        length = options[which(rng_)];
    }
    ++counter_;
    return {make_prompt(length), length};
}
