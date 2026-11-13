#pragma once
#include <random>
#include <string>
#include <vector>

// Deterministic, seeded prompt streams.
//
// Token counting is approximate (1 word ~= 1 token for English prose); the
// harness records the nominal length of every request, and every cell sees
// the identical stream for a given seed, which is what makes the comparison
// fair. The "mixed" distribution holds the MEAN at prompt_tokens while
// spreading lengths over {mean/2, mean, 3*mean/2} with equal probability —
// RQ2 isolates variability at constant mean, not length itself.
class Workload {
public:
    Workload(unsigned seed, int prompt_tokens, std::string distribution);

    // The next request: a prompt and its nominal prompt length in tokens.
    std::pair<std::string, int> next();

private:
    std::mt19937 rng_;
    int mean_tokens_;
    bool mixed_;
    std::vector<std::string> vocab_;
    int counter_ = 0;

    std::string make_prompt(int tokens);
};
