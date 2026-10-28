#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::yue2 {

constexpr int32_t kEodToken = 151643;
constexpr int32_t kAbcStartToken = 151847;
constexpr int32_t kAbcEndToken = 151848;
constexpr int32_t kMusicStartToken = 151851;
constexpr int32_t kMusicEndToken = 151852;
constexpr int32_t kCodecOffset = 151853;
constexpr int32_t kCodecSize = 32768;
constexpr int32_t kVocabSize = 184704;
constexpr int64_t kContextTokens = 24576;

struct Yue2ModelConfig {
    int64_t hidden_size = 2048;
    int64_t layers = 28;
    int64_t attention_heads = 16;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t intermediate_size = 6144;
    int64_t vocab_size = kVocabSize;
    int64_t max_position_embeddings = kContextTokens;
    int64_t latent_dim = 64;
    int64_t max_latent_frames = kContextTokens;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    float timestep_shift = 1.0F;
};

struct Yue2VaeConfig {
    int sample_rate = 48000;
    int64_t channels = 2;
    int64_t latent_dim = 64;
    int64_t encoder_latent_dim = 128;
    int64_t downsampling_ratio = 1920;
    int64_t decode_core_frames = 1024;
    int64_t decode_halo_frames = 16;
};

struct Yue2SamplingConfig {
    float temperature = 1.0F;
    float top_p = 0.95F;
    int64_t top_k = 100;
    float repetition_penalty = 1.2F;
    int64_t penalty_window = 50;
    int64_t min_tokens = 200;
    // 25 tokens is one second of audio. 8800 is what a 6 GB card sustains
    // without spilling to system RAM -- measured, not chosen -- and the budget
    // lowers it whenever it measures less free memory than that.
    int64_t max_tokens = 8800;
};

struct Yue2GenerationConfig {
    Yue2SamplingConfig abc;
    Yue2SamplingConfig semantic;
    int64_t ode_steps = 32;
    int64_t context = kContextTokens;
    // Device memory held back from the per-stage KV budget, for whatever the
    // per-stage costs do not name and the allocator's own rounding. It used to
    // be 640 because the largest unnamed cost -- the near-gigabyte the acoustic
    // stage spends building its runtimes -- was not modelled at all, and even
    // 640 did not cover it. Now that it is charged explicitly, holding back
    // another 640 would charge it twice and cost half a minute of song.
    // Raising this is still the lever a caller has when a run spills anyway --
    // and at 100 that lever is the one to reach for first. This buys the longest
    // score and the longest song the card will give, by holding back almost
    // nothing for what the model does not price; a cot=full take that spills
    // says so in its notes, and the answer is to raise this rather than to
    // shorten anything.
    int64_t vram_headroom_mb = 100;
    // Shrinks every window the budget allows, multiplicatively. 1.0 trusts the
    // estimate; a run that ran out of memory anyway is retried at 0.5, then
    // 0.25, which no amount of unaccounted overhead can defeat.
    double vram_budget_scale = 1.0;
    // Solve the acoustic stage as one window, whatever the memory budget says.
    //
    // The budget sizes the chunk against free memory and splits a take that
    // does not fit, carrying an overlap so the seams are not audible. That is
    // the right default for a card that cannot hold a take whole -- but an
    // overlap narrows a seam rather than removing it, and a caller who would
    // rather pay for one window in speed than hear one has no way to say so.
    // This is that way: the whole take in a single window, spilling to system
    // RAM if it must, with the overflow named in the run's notes.
    bool acoustic_single_window = false;
};

struct Yue2Config {
    Yue2ModelConfig model;
    Yue2VaeConfig vae;
    Yue2GenerationConfig generation;
};

enum class Yue2CotMode {
    Off,
    Melody,
    Full,
};

// What a supplied score is for.
//
// Strict closes the ABC block after the score, so the planner never runs and the
// model sings exactly what it was given. Seed leaves the block open: the score
// becomes the opening of the planner's own context, and the model carries on
// writing it -- same key, same meter, same feel, because it is reading its own
// prefix -- then sings the whole thing. That is the difference between handing
// over an arrangement and handing over an idea.
enum class Yue2AbcMode {
    Strict,
    Seed,
};

struct Yue2Request {
    std::string style;
    std::string lyrics;
    Yue2CotMode cot = Yue2CotMode::Full;
    std::string abc;
    Yue2AbcMode abc_mode = Yue2AbcMode::Strict;
    std::vector<int32_t> semantic_codes;
    std::vector<float> nar_noise;
    uint64_t seed = 831001;
    float cfg_scale = -1.0F;
    Yue2GenerationConfig generation;
};

struct Yue2Plan {
    Yue2CotMode cot = Yue2CotMode::Full;
    std::string abc;
    std::vector<int32_t> abc_ids;
    std::vector<int32_t> prefix;
    bool truncated = false;
};

struct Yue2SemanticResult {
    Yue2Plan plan;
    std::vector<int32_t> tokens;
    bool truncated = false;
};

// Everything one run produced, not just the audio. The score is what makes the
// model's documented editing workflow reachable -- read it, change a bar, send
// it back as `abc` -- and the semantic codes are what make a continuation
// possible, since they can be replayed as a prefix for the next run.
struct Yue2RunOutput {
    runtime::AudioBuffer audio;
    std::string abc;
    // Codec space, i.e. with kCodecOffset already removed, so these round-trip
    // straight back into a request.
    std::vector<int32_t> semantic_codes;
    bool abc_truncated = false;
    bool semantic_truncated = false;
    // Windows the run had to shrink to fit the device, one line each. Empty on
    // a card with room, so a client can show it only when it means something.
    std::vector<std::string> budget_notes;
};

const char * cot_mode_name(Yue2CotMode mode) noexcept;
Yue2CotMode parse_cot_mode(const std::string & value);
Yue2AbcMode parse_abc_mode(const std::string & value);
const char * cot_instruction(Yue2CotMode mode) noexcept;
float request_guidance_scale(const Yue2Request & request) noexcept;

}  // namespace engine::models::yue2
