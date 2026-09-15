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
                                  const Yue2StageCost * stages, int64_t stage_count) {
    if (cache_tokens <= 0 || prefix_tokens < 0 || stages == nullptr || stage_count <= 0) {
        return 0;
    }
    int64_t answer = -1;
    for (int64_t i = 0; i < stage_count; ++i) {
        const Yue2StageCost & stage = stages[i];
        if (stage.generated_cost <= 0) {
            return 0;
        }
        const int64_t budget = (cache_tokens - stage.fixed_tokens) * kYue2CostScale -
                               prefix_tokens * stage.prefix_cost;
        const int64_t tokens = budget <= 0 ? 0 : budget / stage.generated_cost;
        if (answer < 0 || tokens < answer) {
            answer = tokens;
        }
    }
    return answer < 0 ? 0 : answer;
}

int64_t yue2_acoustic_chunk_for(int64_t cache_tokens, int64_t prefix_tokens,
                                int64_t overlap_tokens) {
    if (cache_tokens <= 0 || prefix_tokens < 0 || overlap_tokens < 0) {
        return 0;
    }
    const int64_t left = (cache_tokens - kYue2AcousticFixedTokens) * kYue2CostScale -
                         (prefix_tokens + overlap_tokens) * kYue2AcousticPrefixCost;
    if (left <= 0) {
        return 0;
    }
    return left / kYue2AcousticGeneratedCost;
}

int64_t yue2_score_tokens_for(int64_t cache_tokens, int64_t base_prefix_tokens) {
    if (cache_tokens <= 0 || base_prefix_tokens < 0) {
        return 0;
    }
    const int64_t acoustic_budget =
        cache_tokens - kYue2SemanticRetentionTokens - kYue2AcousticFixedTokens;
    const int64_t spendable =
        acoustic_budget * kYue2CostScale - kYue2MinAcousticChunkTokens * kYue2AcousticGeneratedCost;
    if (spendable <= 0) {
        return 0;
    }
    const int64_t prefix = spendable / kYue2AcousticPrefixCost;
    const int64_t score = prefix - base_prefix_tokens - kYue2AcousticOverlapTokens;
    return score > 0 ? score : 0;
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

    // Every budget below is what the card actually had at that stage, less the
    // 256 MiB of headroom: free MiB * 1024 / 112.
    const int64_t budget_3492 = 29586;   // cot=off, semantic stage
    const int64_t budget_3434 = 29057;   // cot=off, the run that was measured twice
    const int64_t budget_3228 = 27173;   // cot=full, after the planner stage

    // Each cost exists to cover what its stage was *measured* to spend, so that
    // is what is asserted, in the same thousandths the constants are written in.
    record("stage_costs", "semantic covers its measured peak",
           std::to_string(kYue2SemanticTokenCost >= 1950), "1");
    record("stage_costs", "acoustic prefix covers its measured rate",
           std::to_string(kYue2AcousticPrefixCost >= 3650), "1");
    record("stage_costs", "acoustic generated covers its measured rate",
           std::to_string(kYue2AcousticGeneratedCost >= 4210), "1");
    // And are not wastefully above it. A cost a whole cache clear of the
    // measurement is half a minute of song this card could have sung, which is
    // what these numbers were wrong in for a long time.
    record("stage_costs", "semantic is not wasteful",
           std::to_string(kYue2SemanticTokenCost <= 2950), "1");
    record("stage_costs", "acoustic prefix is not wasteful",
           std::to_string(kYue2AcousticPrefixCost <= 4650), "1");
    record("stage_costs", "acoustic generated is not wasteful",
           std::to_string(kYue2AcousticGeneratedCost <= 5210), "1");

    const Yue2StageCost measured_stages[] = {
        Yue2StageCost{kYue2SemanticTokenCost, kYue2SemanticTokenCost, 0},
        Yue2StageCost{kYue2AcousticPrefixCost, kYue2AcousticGeneratedCost, kYue2AcousticFixedTokens},
    };
    constexpr int64_t kMeasuredStageCount = 2;

    // What the acoustic model predicts, against four runs it was fitted to only
    // two of. Peaks in MiB, one cache being 112 KiB.
    auto acoustic_mib = [](int64_t prefix, int64_t generated) {
        const int64_t tokens =
            kYue2AcousticFixedTokens +
            (prefix * kYue2AcousticPrefixCost + generated * kYue2AcousticGeneratedCost) /
                kYue2CostScale;
        return tokens * 112 / 1024;
    };
    auto within = [](int64_t predicted, int64_t measured) {
        const int64_t slack = measured / 20;   // five per cent either way
        return std::to_string(predicted >= measured - slack && predicted <= measured + slack);
    };
    record("stage_costs", "predicts the 2774 MiB peak of 800 prefix, 3000 generated",
           within(acoustic_mib(800, 3000), 2774), "1");
    record("stage_costs", "predicts the 3198 MiB peak of 4748 prefix, 500 generated",
           within(acoustic_mib(4748, 500), 3198), "1");

    // And -- the point of the whole exercise -- it admits the takes that ran at
    // full speed and refuses the two that filled the card and were paged to
    // system RAM. Those two are why the fixed cost is nearly a gigabyte: both
    // overshot by about 330 MiB, which is what the unmeasured runtime setup
    // came to once it was finally measured.
    record("yue2_generated_tokens_for", "admits the 3000-token take that ran at 0.83 realtime",
           std::to_string(yue2_generated_tokens_for(budget_3434, 800, measured_stages,
                                                    kMeasuredStageCount) >= 3000),
           "1");
    record("yue2_generated_tokens_for", "refuses the 5152-token take that spilled",
           std::to_string(yue2_generated_tokens_for(budget_3492, 800, measured_stages,
                                                    kMeasuredStageCount) < 5152),
           "1");
    record("yue2_generated_tokens_for", "refuses the 4079-token cot=full take that spilled",
           std::to_string(yue2_generated_tokens_for(budget_3228, 1394, measured_stages,
                                                    kMeasuredStageCount) < 4079),
           "1");
    // The stages run one after another, so the answer is the one that binds and
    // never their sum. Adding them is what reserved the semantic peak through a
    // stage that had already freed it, and cost a third of every take.
    record("yue2_generated_tokens_for", "the binding stage decides, not the sum",
           std::to_string(yue2_generated_tokens_for(budget_3492, 800, measured_stages,
                                                    kMeasuredStageCount)),
           "3904");
    {
        // A semantic stage made expensive enough to bind instead.
        const Yue2StageCost lopsided[] = {
            Yue2StageCost{20000, 20000, 0},
            Yue2StageCost{kYue2AcousticPrefixCost, kYue2AcousticGeneratedCost, kYue2AcousticFixedTokens},
        };
        record("yue2_generated_tokens_for", "and it is whichever stage is dearest",
               std::to_string(yue2_generated_tokens_for(20000, 0, lopsided, 2)), "1000");
    }

    // A capped score must leave a song behind it. Uncapped, the 3948 tokens this
    // card writes by default cost the acoustic stage 2.1 GiB on their own.
    {
        const int64_t score = yue2_score_tokens_for(budget_3492, 810);
        record("yue2_score_tokens_for", "the score a 20-second chunk floor allows",
               std::to_string(score), "649");
        // The cap exists to keep the chunk workable once the semantic stage has
        // taken its share, so that is what is checked: against the budget the
        // acoustic stage will actually see, the chunk is still at its floor.
        const int64_t acoustic_budget = budget_3492 - kYue2SemanticRetentionTokens;
        record("yue2_score_tokens_for", "at the cap the chunk is still at its floor",
               std::to_string(yue2_acoustic_chunk_for(acoustic_budget, 810 + score,
                                                      kYue2AcousticOverlapTokens) >=
                              kYue2MinAcousticChunkTokens),
               "1");
        record("yue2_score_tokens_for", "and one token past it is not",
               std::to_string(yue2_acoustic_chunk_for(acoustic_budget, 810 + score + 1,
                                                      kYue2AcousticOverlapTokens) <
                              kYue2MinAcousticChunkTokens),
               "1");
        // The 3083-token score that sizing against the planner stage let through
        // could not be prefilled by the acoustic stage at all.
        record("yue2_score_tokens_for", "the score sized against the planner stage fit nothing",
               std::to_string(yue2_acoustic_chunk_for(acoustic_budget, 810 + 3083,
                                                      kYue2AcousticOverlapTokens)),
               "0");
        record("yue2_score_tokens_for", "unmeasurable budget",
               std::to_string(yue2_score_tokens_for(0, 810)), "0");
    }

    // Chunking is what takes the take's own length out of the acoustic budget.
    // What is left over after the fixed cost and the prefix decides how long one
    // piece may be, and the piece is the same length whether the song runs two
    // minutes or ten.
    {
        const int64_t chunk = yue2_acoustic_chunk_for(budget_3492, 810, kYue2AcousticOverlapTokens);
        record("yue2_acoustic_chunk_for", "cot=off, a 256-frame overlap",
               std::to_string(chunk), "3675");
        // Zero means the prefix alone does not fit, which is not the same as
        // "no limit" -- and the caller that read it that way put a 7105-frame
        // take back into one piece and paged most of it to system RAM.
        record("yue2_acoustic_chunk_for", "zero is a refusal, not an absence of limit",
               std::to_string(yue2_acoustic_chunk_for(budget_3492 - kYue2SemanticRetentionTokens,
                                                      3893, kYue2AcousticOverlapTokens)),
               "0");
        // The two takes that spilled both fit in one such chunk plus a second,
        // and neither needs shortening any more.
        record("yue2_acoustic_chunk_for", "the 5152-token take now needs two chunks, not a shorter song",
               std::to_string((5152 + chunk - 1) / chunk), "2");
        record("yue2_acoustic_chunk_for", "a chunk costs what the whole take used to",
               within(acoustic_mib(810 + kYue2AcousticOverlapTokens, chunk), 3236), "1");
        // A score is prefix at this stage too, so it shortens the chunk rather
        // than the song -- the same trade, at a tenth the cost.
        record("yue2_acoustic_chunk_for", "a 3948-token score shortens the chunk instead",
               std::to_string(yue2_acoustic_chunk_for(budget_3492, 810 + 3948,
                                                      kYue2AcousticOverlapTokens) > 0),
               "1");
        record("yue2_acoustic_chunk_for", "and a prefix that cannot fit at all admits no chunk",
               std::to_string(yue2_acoustic_chunk_for(12000, 60000, 256)), "0");
        record("yue2_acoustic_chunk_for", "unmeasurable budget",
               std::to_string(yue2_acoustic_chunk_for(0, 810, 256)), "0");
    }

    // With the acoustic stage chunked, only the semantic cache still grows with
    // the take, and that is what the length is budgeted against now.
    {
        const Yue2StageCost semantic_only[] = {
            Yue2StageCost{kYue2SemanticTokenCost + kYue2SemanticGuidanceCost,
                          kYue2SemanticTokenCost + kYue2SemanticGuidanceCost, 0},
        };
        const int64_t cap = yue2_generated_tokens_for(budget_3492, 810, semantic_only, 1);
        record("yue2_generated_tokens_for", "guidance on, the semantic stage alone",
               std::to_string(cap), "9052");
        // 25 tokens is one second, so this is the three minutes the card could
        // not hold while the take had to fit the acoustic stage whole.
        record("yue2_generated_tokens_for", "which is over five minutes of audio",
               std::to_string(cap / 25 >= 300), "1");
    }

    // Zero is returned for two unrelated reasons, and `yue2_stage_window` reads
    // a zero cap as "no cap". A caller that cannot tell them apart leaves the
    // window at its full size and generates tokens the run cannot pay for --
    // which is exactly what failed an acoustic stage twenty minutes in. The
    // contract is pinned here so the trap stays visible.
    record("yue2_stage_window", "a zero cap is not a cap",
           std::to_string(yue2_stage_window("semantic_max_tokens", 9000, 0).tokens), "9000");
    record("yue2_stage_window", "and is not reported as clamped either",
           std::to_string(yue2_stage_window("semantic_max_tokens", 9000, 0).clamped), "0");

    {
        const Yue2StageCost one[] = {Yue2StageCost{5000, 6000, 0}};
        record("yue2_generated_tokens_for", "no prefix",
               std::to_string(yue2_generated_tokens_for(12000, 0, one, 1)), "2000");
        record("yue2_generated_tokens_for", "prefix alone exhausts the budget",
               std::to_string(yue2_generated_tokens_for(2000, 2600, one, 1)), "0");
        record("yue2_generated_tokens_for", "unmeasurable budget",
               std::to_string(yue2_generated_tokens_for(0, 100, one, 1)), "0");
        const Yue2StageCost free_stage[] = {Yue2StageCost{5000, 0, 0}};
        record("yue2_generated_tokens_for", "zero generated cost",
               std::to_string(yue2_generated_tokens_for(10000, 0, free_stage, 1)), "0");
        record("yue2_generated_tokens_for", "no stages at all",
               std::to_string(yue2_generated_tokens_for(10000, 0, nullptr, 0)), "0");
    }

    {
        const auto window = yue2_stage_window("semantic_max_tokens", 9000, 4772);
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
