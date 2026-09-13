#pragma once

#include <cstdint>
#include <string>

namespace engine::models::yue2 {

// One concern: how long a yue2 stage may get before its KV cache stops fitting
// the device. Nothing here touches ggml or the model -- the caller measures the
// free memory, asks, and applies the answer.
//
// Every stage of a run is bounded by the same thing: a decode cache holding one
// f32 K and one f32 V per layer per token, allocated up front for the whole
// window the stage was asked for. Ask for a window the card cannot hold and the
// allocation fails partway through a run that has already cost minutes, which
// is the failure this exists to prevent.
//
// The caller is expected to ask once per stage, immediately before it starts,
// so the measurement already accounts for whatever earlier stages still hold.

struct Yue2CacheDims {
    int64_t layers = 28;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    // K and V live in the decode cache as f32.
    int64_t bytes_per_element = 4;
};

// Bytes of KV cache one token of one sequence costs.
int64_t yue2_cache_bytes_per_token(const Yue2CacheDims & dims);

// The headroom a stage can actually hold back. A caller raising the headroom to
// force smaller windows -- the retry after an out-of-memory does exactly that --
// can easily name more than the device has left, and a headroom above the free
// memory would leave no budget at all and be read as "could not measure". Never
// hold back more than half of what is free, so a budget always exists and always
// shrinks when the caller asks for more headroom.
int64_t yue2_effective_headroom(int64_t free_bytes, int64_t requested_headroom_bytes);

// What a stage costs per token, as a multiple of one plain KV cache entry.
// Token decoding pays exactly one cache. The acoustic stage's prefix-state
// graph pays far more: it materialises the whole prefix as f32, rounds it to
// bf16, casts to f16 and copies that into the cache, so several representations
// of the same tokens are live at once. Measured at about 2.5x on the 3B; 3 is
// the number used, because being wrong here costs a whole run.
constexpr int64_t kYue2DecodeCostWeight = 1;
constexpr int64_t kYue2PrefixStateCostWeight = 3;

// The largest window that fits, for `sequences` concurrent caches -- guidance
// runs a positive and a negative one -- at `cost_weight` caches per token.
// `scale` shrinks the answer further; a retry after an out-of-memory halves it,
// which is the one escalation that cannot be defeated by a tight device.
// Returns 0 when not even one token fits.
int64_t yue2_max_tokens_for(int64_t free_bytes, int64_t headroom_bytes,
                            int64_t bytes_per_token, int64_t sequences,
                            int64_t cost_weight = 1, double scale = 1.0);

// How many tokens a stage may *generate*, given that a prefix already sits in
// the same cache and a later stage will pay again for every token generated.
//
// The two costs are not the same and must not be charged the same. A prefix
// token costs one cache per decoding sequence and nothing afterwards: the
// acoustic stage never sees the planner's score or the lyrics. A generated
// token costs those same caches *and* its share of the acoustic prefix-state
// graph. Charging the prefix at the generated rate is what makes a run with a
// long plan collapse to nothing -- `cot=full` writes a score of thousands of
// tokens, and at four caches a token that plan alone can exhaust a budget that
// would comfortably have held it at one.
//
// `cache_tokens` is the budget in plain single-cache token units, i.e.
// yue2_max_tokens_for(..., cost_weight = 1). Returns 0 when the prefix alone
// does not fit, or when the budget could not be measured.
int64_t yue2_generated_tokens_for(int64_t cache_tokens, int64_t prefix_tokens,
                                  int64_t decode_caches, int64_t downstream_weight);

// The window a stage should actually use, and why. `cap` of 0 means the free
// memory could not be measured, in which case `requested` is honoured: a run
// that might have fit is better than one refused on a guess.
struct Yue2StageWindow {
    int64_t tokens = 0;
    bool clamped = false;
    std::string note;   // empty unless clamped
};

Yue2StageWindow yue2_stage_window(const std::string & name, int64_t requested, int64_t cap);

}   // namespace engine::models::yue2
