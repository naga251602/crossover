#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct RequestRecord {
    int prompt_tokens = 0;
    int output_tokens = 0;      // tokens actually observed in the stream
    double ttft_ms = 0.0;       // request start -> first token
    double tpot_ms = 0.0;       // mean time per output token after the first
    double latency_ms = 0.0;    // request start -> completion
    bool ok = false;
};

inline double percentile(std::vector<double> values, double pct) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    double rank = (pct / 100.0) * (values.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = rank - lo;
    return values[lo] * (1 - frac) + values[hi] * frac;
}

inline double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}
