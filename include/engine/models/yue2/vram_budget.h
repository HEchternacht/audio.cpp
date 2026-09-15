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

// What a run costs, per token, as a multiple of one plain KV cache entry.
//
// Decoding one token into an empty cache costs exactly one entry, and that is
// all `kYue2DecodeCostWeight` claims. A whole run costs far more than its
// caches, and the extra is not one number but one per stage, because the stages
// do not run together and do not hold the same things.
//
// A run reaches its peak twice, never at once. The semantic stage holds one
// decode cache per token it carries -- prefix and generated alike, doubled when
// guidance keeps a negative sequence -- and then
// `Yue2ArRuntime::release_runtime_graphs` frees that cache before the acoustic
// stage builds anything. The acoustic stage then prefills the AR over prefix
// plus take on a device that no longer holds the semantic cache, and solves the
// flow-matching chunk in an arena sized to the take.
//
// So the two costs must be compared, not added. Adding them -- which is what
// this budget did until the measurements below -- reserves the semantic peak
// through a stage where it has already been freed, and a `cot=full` score pays
// that surcharge twice: once as prefix in a cache that is gone by the time the
// acoustic stage runs, and once more in a sum that pretends it is still there.
// That is what turned a four-thousand-token score into ninety seconds of song.
//
// Both numbers are measured, not chosen -- by `report_stage`, which samples the
// peak from a second thread while a stage is still running, since free memory
// after a stage tells you only what it kept.
//
// On an RTX 4050, one cache entry being 112 KiB, guidance off:
//
//   semantic  576 MiB over 2800 tokens -> 1.88 caches/token
//   semantic  686 MiB over 3800 tokens -> 1.65
//   acoustic 1344 MiB with  800 prefix, 2000 generated
//   acoustic 1804 MiB with  800 prefix, 3000 generated
//
// The acoustic pair differ only in the take, so the slope between them is what
// a generated token costs there: 460 MiB per 1000 tokens, 4.21 caches each.
// What is left at 2000 tokens -- 424 MiB -- is what the stage spends before it
// has generated anything, and `kYue2AcousticFixedTokens` carries it.
//
// Guidance adds one more cache per token to the semantic stage and nothing to
// the acoustic one, which never runs a negative sequence.
//
// Being wrong here costs a run that has already spent minutes, so where the data
// allows a range, take the conservative end: each rate is its measurement
// rounded up.
// Costs are carried in thousandths of a cache. Rounding a measured 4.21 up to
// an integer 5 is not caution, it is nineteen per cent of the take -- half a
// minute of song on this card -- and the driver's system-memory fallback now
// catches a miss that would once have been a crash, so the honest number is
// also the safe one.
constexpr int64_t kYue2CostScale = 1000;
constexpr int64_t kYue2DecodeCostWeight = 1;

// Per token the semantic cache holds, prefix and generated alike. Measured
// 1.51, 1.65, 1.88 and 1.95 across four takes; 2.0 is the conservative end.
constexpr int64_t kYue2SemanticTokenCost = 2000;
// Guidance keeps a negative sequence alive beside the positive one, so every
// token held costs one more cache. Nothing of it reaches the acoustic stage,
// which never runs a negative sequence.
constexpr int64_t kYue2SemanticGuidanceCost = 1000;

// The acoustic stage, fitted to the measurements above: it prefills the AR over
// prefix plus take, and solves a flow-matching chunk sized to the take. A score
// token is therefore not free here either -- it is prefilled again, having
// bought the singing nothing at this stage.
//
// The fixed term is nearly a gigabyte and almost all of it used to be invisible.
// `ensure_nar` loads the NAR weights and `ensure_ar` rebuilds what the semantic
// stage released, both after the budget has been decided and -- until the watch
// was moved above them -- outside anything that measured. The same 800-prefix,
// 3000-token run that peaked at 1804 MiB measured from below them peaks at 2774
// MiB measured from above: 970 MiB spent every run and reserved by nothing.
//
// That omission is the whole of the recent overcommitment. With it counted, the
// model predicts every take that fit and every take that spilled, to within a
// few per cent and in the right direction -- including the two that filled the
// card to nothing and were paged to system RAM at six times realtime.
constexpr int64_t kYue2AcousticPrefixCost = 3700;
constexpr int64_t kYue2AcousticGeneratedCost = 4300;
constexpr int64_t kYue2AcousticFixedTokens = 9837;

// How much of the take before a chunk is re-solved with it, purely as context.
// Chunking the acoustic stage is what stops the take's length from deciding
// whether it fits the card at all -- but a chunk solved against nothing but the
// style prefix has no idea what the music was doing a moment ago, and the seam
// arrives as a lurch. Carrying the previous frames through the solver and then
// discarding their latents is what makes the join continuous. Ten seconds of
// audio, costing one prefill's worth of prefix and about an eighth of the
// solver's work.
constexpr int64_t kYue2AcousticOverlapTokens = 256;

// The longest chunk the acoustic stage can solve in one piece: what is left of
// the budget once its fixed cost and the prefix it re-prefills -- style, lyrics
// and any score, plus the overlap carried from the chunk before -- are paid
// for. Returns 0 when not even that prefix fits, which is the one case where
// the take itself still has to be shortened.
int64_t yue2_acoustic_chunk_for(int64_t cache_tokens, int64_t prefix_tokens,
                                int64_t overlap_tokens);

// The shortest acoustic chunk worth solving. A score is prefix at that stage
// and is re-prefilled for every chunk, so a long one no longer shortens the
// song -- it shortens the piece. Left alone that trade runs away: the
// 3948-token score this card writes by default would leave chunks of six
// seconds, each carrying a longer overlap than it keeps. Twenty seconds is the
// floor, and it is a floor rather than a target: a prefix too large for even
// this is chunked at it anyway and allowed to spill, because the alternative --
// falling back to one chunk for the whole take -- spills very much harder.
constexpr int64_t kYue2MinAcousticChunkTokens = 512;

// What the semantic stage is still holding when the acoustic stage starts, over
// and above what the acoustic stage then allocates for itself. Releasing the
// decode caches does not give all of it back: the runtimes, their graphs and
// the backend's own cached allocations stay, and measured across four takes
// that came to between 968 and 1222 MiB. The conservative end, in cache-tokens.
//
// It matters at exactly one point. The planner writes its score before the
// semantic stage runs, so the only free memory anyone can measure then is the
// larger figure -- and sizing a score against that is what let a 3083-token
// score through, which the acoustic stage could not then prefill at all. The
// next lever on length is to stop retaining this, not to budget around it.
constexpr int64_t kYue2SemanticRetentionTokens = 11200;

// The longest planner score that still leaves a workable chunk behind it. Takes
// the budget as measured at the planner stage and discounts what the semantic
// stage will still be holding by the time the score has to be prefilled again.
// Returns 0 when the budget could not be measured, or when no score fits.
int64_t yue2_score_tokens_for(int64_t cache_tokens, int64_t base_prefix_tokens);

// The largest window that fits, for `sequences` concurrent caches -- guidance
// runs a positive and a negative one -- at `cost_weight` caches per token.
// `scale` shrinks the answer further; a retry after an out-of-memory halves it,
// which is the one escalation that cannot be defeated by a tight device.
// Returns 0 when not even one token fits.
int64_t yue2_max_tokens_for(int64_t free_bytes, int64_t headroom_bytes,
                            int64_t bytes_per_token, int64_t sequences,
                            int64_t cost_weight = 1, double scale = 1.0);

// What one stage costs, in thousandths of a plain single-cache token: per token
// of the prefix it carries, per token it generates, and -- in whole tokens --
// what it spends regardless of either.
struct Yue2StageCost {
    int64_t prefix_cost = 0;
    int64_t generated_cost = kYue2CostScale;
    int64_t fixed_tokens = 0;
};

// How many tokens a run may generate, given a prefix it must carry and the
// per-stage costs above: the smallest window every stage can still hold. The
// stages run one after another and each must fit on its own, so the answer is
// their minimum and never their sum.
//
// `cache_tokens` is the budget in plain single-cache token units, i.e.
// yue2_max_tokens_for(..., cost_weight = 1). Returns 0 when some stage cannot
// hold the prefix at all, or when the budget could not be measured.
int64_t yue2_generated_tokens_for(int64_t cache_tokens, int64_t prefix_tokens,
                                  const Yue2StageCost * stages, int64_t stage_count);

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
