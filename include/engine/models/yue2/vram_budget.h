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

// The largest window that fits, for `sequences` concurrent caches -- guidance
// runs a positive and a negative one. Returns 0 when not even one token fits.
int64_t yue2_max_tokens_for(int64_t free_bytes, int64_t headroom_bytes,
                            int64_t bytes_per_token, int64_t sequences);

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
