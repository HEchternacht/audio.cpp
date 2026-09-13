#include "engine/models/yue2/vram_budget.h"

namespace engine::models::yue2 {

int64_t yue2_cache_bytes_per_token(const Yue2CacheDims & dims) {
    // One K and one V per layer, each kv_heads * head_dim elements.
    return 2 * dims.layers * dims.kv_heads * dims.head_dim * dims.bytes_per_element;
}

int64_t yue2_effective_headroom(int64_t free_bytes, int64_t requested_headroom_bytes) {
    if (free_bytes <= 0) {
        return requested_headroom_bytes;
    }
    const int64_t half = free_bytes / 2;
    return requested_headroom_bytes < half ? requested_headroom_bytes : half;
}

int64_t yue2_max_tokens_for(int64_t free_bytes, int64_t headroom_bytes,
                            int64_t bytes_per_token, int64_t sequences,
                            int64_t cost_weight, double scale) {
    if (bytes_per_token <= 0 || sequences <= 0 || cost_weight <= 0 || scale <= 0.0) {
        return 0;
    }
    const int64_t usable = free_bytes - headroom_bytes;
    if (usable <= 0) {
        return 0;
    }
    const int64_t tokens = usable / (bytes_per_token * sequences * cost_weight);
    return static_cast<int64_t>(static_cast<double>(tokens) * scale);
}

int64_t yue2_generated_tokens_for(int64_t cache_tokens, int64_t prefix_tokens,
                                  int64_t decode_caches, int64_t downstream_weight) {
    if (cache_tokens <= 0 || decode_caches <= 0 || downstream_weight < 0 || prefix_tokens < 0) {
        return 0;
    }
    const int64_t left = cache_tokens - prefix_tokens * decode_caches;
    if (left <= 0) {
        return 0;
    }
    return left / (decode_caches + downstream_weight);
}

Yue2StageWindow yue2_stage_window(const std::string & name, int64_t requested, int64_t cap) {
    Yue2StageWindow out;
    out.tokens = requested;
    if (cap <= 0 || requested <= cap) {
        return out;
    }
    out.tokens = cap;
    out.clamped = true;
    out.note = name + " reduced from " + std::to_string(requested) + " to " + std::to_string(cap) +
               " so its cache fits the free device memory";
    return out;
}

}   // namespace engine::models::yue2

#ifdef AUDIOCPP_SELFTEST_YUE2_VRAM_BUDGET
#include <cstdio>

namespace {

int g_pass = 0;
int g_fail = 0;

void record(const char * unit, const std::string & in, const std::string & out, const std::string & exp) {
    std::printf("[%s] IN : %s\n[%s] OUT: %s\n[%s] EXP: %s\n", unit, in.c_str(), unit, out.c_str(), unit,
                exp.c_str());
    if (out == exp) {
        std::printf("[%s] OK\n", unit);
        ++g_pass;
    } else {
        std::printf("[%s] FAIL\n", unit);
        ++g_fail;
    }
}

}   // namespace

int main() {
    using namespace engine::models::yue2;

    const Yue2CacheDims dims{};   // the 3B: 28 layers, 8 kv heads, 128 head dim, f32
    const int64_t per_token = yue2_cache_bytes_per_token(dims);
    record("yue2_cache_bytes_per_token", "28 layers, 8 kv heads, 128 head dim, f32",
           std::to_string(per_token), "229376");
    record("yue2_cache_bytes_per_token", "zero layers",
           std::to_string(yue2_cache_bytes_per_token(Yue2CacheDims{0, 8, 128, 4})), "0");

    // 1 GiB free, 256 MiB held back: 768 MiB / 224 KiB per token.
    record("yue2_max_tokens_for", "1 GiB free, 256 MiB headroom, one sequence",
           std::to_string(yue2_max_tokens_for(1LL << 30, 256LL << 20, per_token, 1)), "3510");
    record("yue2_max_tokens_for", "same, two sequences (guidance)",
           std::to_string(yue2_max_tokens_for(1LL << 30, 256LL << 20, per_token, 2)), "1755");
    record("yue2_max_tokens_for", "headroom exceeds free",
           std::to_string(yue2_max_tokens_for(128LL << 20, 256LL << 20, per_token, 1)), "0");
    record("yue2_max_tokens_for", "zero sequences",
           std::to_string(yue2_max_tokens_for(1LL << 30, 0, per_token, 0)), "0");
    record("yue2_max_tokens_for", "unmeasurable free memory",
           std::to_string(yue2_max_tokens_for(0, 0, per_token, 1)), "0");

    record("yue2_effective_headroom", "640 MiB asked of 2 GiB free",
           std::to_string(yue2_effective_headroom(2LL << 30, 640LL << 20)), "671088640");
    record("yue2_effective_headroom", "2560 MiB asked of 978 MiB free",
           std::to_string(yue2_effective_headroom(978LL << 20, 2560LL << 20)), "512753664");
    record("yue2_effective_headroom", "free unmeasurable, request kept",
           std::to_string(yue2_effective_headroom(0, 640LL << 20)), "671088640");
    record("yue2_max_tokens_for", "prefix-state weight, three caches a token",
           std::to_string(yue2_max_tokens_for(1LL << 30, 256LL << 20, per_token, 1,
                                              kYue2PrefixStateCostWeight)),
           "1170");
    record("yue2_max_tokens_for", "half scale after a retry",
           std::to_string(yue2_max_tokens_for(1LL << 30, 256LL << 20, per_token, 1,
                                              kYue2DecodeCostWeight, 0.5)),
           "1755");
    record("yue2_max_tokens_for", "scale of zero is not a budget",
           std::to_string(yue2_max_tokens_for(1LL << 30, 256LL << 20, per_token, 1, 1, 0.0)), "0");
    record("yue2_max_tokens_for", "headroom capped at half, so a budget remains",
           std::to_string(yue2_max_tokens_for(978LL << 20, yue2_effective_headroom(978LL << 20, 2560LL << 20),
                                              per_token, 1)),
           "2235");

    {
        const auto window = yue2_stage_window("semantic_max_tokens", 9000, 4772);
        // 10000 cache tokens, a 2600-token plan+lyrics prefix, no guidance, the
    // acoustic stage charging 3: (10000 - 2600) / 4.
    record("yue2_generated_tokens_for", "10000 cache tokens, 2600 prefix, 1 decode cache, 3 downstream",
           std::to_string(yue2_generated_tokens_for(10000, 2600, 1, 3)), "1850");
    // The same budget under guidance: the prefix is held twice and every
    // generated token costs two caches plus the acoustic three.
    record("yue2_generated_tokens_for", "same, guidance (2 decode caches)",
           std::to_string(yue2_generated_tokens_for(10000, 2600, 2, 3)), "960");
    record("yue2_generated_tokens_for", "no prefix",
           std::to_string(yue2_generated_tokens_for(10000, 0, 1, 3)), "2500");
    record("yue2_generated_tokens_for", "downstream free",
           std::to_string(yue2_generated_tokens_for(10000, 2600, 1, 0)), "7400");
    record("yue2_generated_tokens_for", "prefix alone exhausts the budget",
           std::to_string(yue2_generated_tokens_for(2000, 2600, 1, 3)), "0");
    record("yue2_generated_tokens_for", "unmeasurable budget",
           std::to_string(yue2_generated_tokens_for(0, 100, 1, 3)), "0");
    record("yue2_generated_tokens_for", "zero decode caches",
           std::to_string(yue2_generated_tokens_for(10000, 0, 0, 3)), "0");

    record("yue2_stage_window", "9000 requested, 4772 fits",
               std::to_string(window.tokens) + " clamped=" + (window.clamped ? "1" : "0"),
               "4772 clamped=1");
        record("yue2_stage_window", "note text", window.note,
               "semantic_max_tokens reduced from 9000 to 4772 so its cache fits the free device memory");
    }
    {
        const auto window = yue2_stage_window("abc_max_tokens", 4096, 9545);
        record("yue2_stage_window", "4096 requested, 9545 fits",
               std::to_string(window.tokens) + " clamped=" + (window.clamped ? "1" : "0") +
                   " note=" + (window.note.empty() ? "none" : window.note),
               "4096 clamped=0 note=none");
    }
    {
        // Cap of 0 means "could not measure": honour the request rather than
        // refusing a run that might have fit.
        const auto window = yue2_stage_window("semantic_max_tokens", 9000, 0);
        record("yue2_stage_window", "cap unknown",
               std::to_string(window.tokens) + " clamped=" + (window.clamped ? "1" : "0"),
               "9000 clamped=0");
    }
    {
        const auto window = yue2_stage_window("nar_context", 4096, 4096);
        record("yue2_stage_window", "exactly fits",
               std::to_string(window.tokens) + " clamped=" + (window.clamped ? "1" : "0"),
               "4096 clamped=0");
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif
