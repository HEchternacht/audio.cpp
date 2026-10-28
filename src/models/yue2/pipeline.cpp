#include "engine/models/yue2/pipeline.h"

#include "engine/framework/debug/profiler.h"
#include "engine/models/yue2/ar_runtime.h"
#include "engine/models/yue2/nar_runtime.h"
#include "engine/models/yue2/vram_budget.h"

#include <ggml-backend.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <stdexcept>
#include <utility>

namespace engine::models::yue2 {
namespace {

using Clock = std::chrono::steady_clock;

std::string request_text(const Yue2Request & request) {
    std::string text;
    text += cot_instruction(request.cot);
    text += "\n[Tags]\n";
    text += request.style;
    text += "\n[Lyrics]\n";
    text += request.lyrics;
    text += "\n";
    return text;
}

std::vector<int32_t> token_prefixes(
    const Yue2Request & request,
    const Yue2TextTokenizer & tokenizer,
    const std::vector<int32_t> & abc_ids) {
    std::vector<int32_t> out;
    out.push_back(kEodToken);
    auto text_ids = tokenizer.encode(request_text(request));
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    out.push_back(kAbcStartToken);
    if (request.cot == Yue2CotMode::Off) {
        out.push_back(kAbcEndToken);
        out.push_back(kMusicStartToken);
        return out;
    }
    out.insert(out.end(), abc_ids.begin(), abc_ids.end());
    // Closing the block is what makes a score binding: after kAbcEndToken there
    // is nothing left for the planner to add. Seed mode leaves it open, so the
    // supplied bars read as the beginning of a score the model is still writing
    // rather than the whole of one. The planner closes it once it has finished.
    if (!request.abc.empty() && request.abc_mode == Yue2AbcMode::Strict) {
        out.push_back(kAbcEndToken);
        out.push_back(kMusicStartToken);
    }
    return out;
}

std::vector<int32_t> negative_prefix(
    const Yue2Request & request,
    const Yue2TextTokenizer & tokenizer,
    const std::vector<int32_t> & abc_ids) {
    std::vector<int32_t> out;
    out.push_back(kEodToken);
    const auto text_ids = tokenizer.encode(cot_instruction(request.cot));
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    if (request.cot == Yue2CotMode::Off) {
        out.push_back(kMusicStartToken);
        return out;
    }
    out.push_back(kAbcStartToken);
    out.insert(out.end(), abc_ids.begin(), abc_ids.end());
    out.push_back(kAbcEndToken);
    out.push_back(kMusicStartToken);
    return out;
}

std::vector<int32_t> codec_from_semantic_tokens(const std::vector<int32_t> & tokens) {
    std::vector<int32_t> out;
    out.reserve(tokens.size());
    for (const int32_t token : tokens) {
        if (token >= kCodecOffset && token < kCodecOffset + kCodecSize) {
            out.push_back(token - kCodecOffset);
        }
    }
    return out;
}

Yue2ArSamplingWindow abc_window(const Yue2GenerationConfig & generation) {
    return Yue2ArSamplingWindow{
        0,
        kEodToken,
        kAbcEndToken,
        generation.abc.min_tokens,
        generation.abc.max_tokens,
        generation.abc,
    };
}

Yue2ArSamplingWindow semantic_window(const Yue2GenerationConfig & generation) {
    return Yue2ArSamplingWindow{
        kCodecOffset,
        kCodecOffset + kCodecSize,
        kMusicEndToken,
        generation.semantic.min_tokens,
        generation.semantic.max_tokens,
        generation.semantic,
    };
}

}  // namespace

class Yue2PipelineRuntime::Impl {
public:
    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType model_weight_type,
        assets::TensorStorageType vae_weight_type,
        size_t model_weight_context_bytes,
        size_t vae_weight_context_bytes,
        size_t ar_prefill_graph_arena_bytes,
        size_t ar_decode_graph_arena_bytes,
        size_t nar_graph_arena_bytes,
        size_t vae_graph_arena_bytes)
        : execution(&execution),
          assets(std::move(assets)),
          tokenizer(this->assets->tiktoken_path),
          model_weight_type(model_weight_type),
          vae_weight_type(vae_weight_type),
          model_weight_context_bytes(model_weight_context_bytes),
          vae_weight_context_bytes(vae_weight_context_bytes),
          ar_prefill_graph_arena_bytes(ar_prefill_graph_arena_bytes),
          ar_decode_graph_arena_bytes(ar_decode_graph_arena_bytes),
          nar_graph_arena_bytes(nar_graph_arena_bytes),
          vae_graph_arena_bytes(vae_graph_arena_bytes) {
        if (!this->assets) {
            throw std::runtime_error("Yue2 pipeline requires assets");
        }
        (void) this->nar_graph_arena_bytes;
    }

    Yue2Plan plan(const Yue2Request & request) {
        Yue2Plan out;
        out.cot = request.cot;
        if (!request.abc.empty()) {
            out.abc = request.abc;
            out.abc_ids = tokenizer.encode(request.abc);
        }
        out.prefix = token_prefixes(request, tokenizer, out.abc_ids);
        engine::debug::timing_log_scalar("yue2.plan.prefix_tokens", out.prefix.size());
        engine::debug::timing_log_scalar("yue2.plan.abc_tokens", out.abc_ids.size());
        return out;
    }

    // Free bytes on the device the model runs on, or 0 when that cannot be
    // answered -- a CPU backend, or a device that does not report. Measured per
    // stage rather than once, so each answer already accounts for what the
    // stages before it still hold.
    int64_t free_device_bytes() const {
        auto * backend = execution->backend();
        if (backend == nullptr) {
            return 0;
        }
        auto * device = ggml_backend_get_device(backend);
        if (device == nullptr) {
            return 0;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        return static_cast<int64_t>(free_bytes);
    }

    // Headroom for everything that is not the KV cache: the graph arenas, the
    // logits readback, the allocator's own rounding. Measured peaks on a 6 GB
    // card sit about this far above the cache alone.
    int64_t vram_headroom_bytes = 100LL << 20;
    double vram_budget_scale = 1.0;

    int64_t cache_bytes_per_token() const {
        return yue2_cache_bytes_per_token(Yue2CacheDims{
            assets->config.model.layers,
            assets->config.model.kv_heads,
            assets->config.model.head_dim,
            yue2_ar_cache_bytes_per_element(execution->backend_type())});
    }

    // Free memory at a named moment of the run. The score cap is sized against
    // a prediction of what the semantic stage will still be holding when the
    // acoustic stage starts, and until now nothing printed the quantity being
    // predicted -- so the constant could only ever be checked by inference.
    void report_free(const char * moment) const {
        const int64_t free_bytes = free_device_bytes();
        if (free_bytes <= 0) {
            return;
        }
        const int64_t per_token = std::max<int64_t>(1, cache_bytes_per_token());
        std::fprintf(stderr, "[yue2] free at %s: %lld MiB (%lld cache-tokens)\n", moment,
                     static_cast<long long>(free_bytes >> 20),
                     static_cast<long long>(free_bytes / per_token));
        engine::debug::timing_log_scalar(std::string("yue2.free.") + moment, free_bytes);
    }

    int64_t stage_token_cap(int64_t sequences, int64_t cost_weight) const {
        const Yue2CacheDims dims{
            assets->config.model.layers,
            assets->config.model.kv_heads,
            assets->config.model.head_dim,
            yue2_ar_cache_bytes_per_element(execution->backend_type())};
        const int64_t free_bytes = free_device_bytes();
        return yue2_max_tokens_for(free_bytes,
                                   yue2_effective_headroom(free_bytes, vram_headroom_bytes),
                                   yue2_cache_bytes_per_token(dims), sequences, cost_weight,
                                   vram_budget_scale);
    }

    // What a stage cost at its worst moment, against what the budget charged
    // it. Free memory after a stage tells you only what it kept; what decides
    // whether a run survives is the peak, and the peak is visible only while
    // the stage is still running. Sampling from a second thread catches it --
    // every allocation involved is held for seconds, so 50 ms is ample -- and
    // is the only way these weights stop being numbers I chose by hand.
    void report_stage(const char * name, int64_t before, int64_t low, int64_t tokens) {
        if (before <= 0 || tokens <= 0) {
            return;
        }
        // A stage that drove the device to nothing used to report a peak of
        // zero and say nothing else, which read as "this stage was free" -- the
        // opposite of what happened. Floor the low-water mark instead, so the
        // peak is at least everything that was there.
        low = low < 0 ? 0 : low;
        const int64_t peak = before - low;
        const int64_t per_token = peak / tokens;
        std::fprintf(stderr, "[yue2] %s peak %lld MiB over %lld tokens (%lld KiB/token, %.2f caches)\n",
                     name, static_cast<long long>(peak >> 20), static_cast<long long>(tokens),
                     static_cast<long long>(per_token >> 10),
                     static_cast<double>(per_token) / static_cast<double>(cache_bytes_per_token()));
        engine::debug::timing_log_scalar(std::string("yue2.peak.") + name, peak);
        engine::debug::timing_log_scalar(std::string("yue2.peak_per_token.") + name, per_token);
        // A stage that ran the device down to nothing did not necessarily fail:
        // with the driver's system-memory fallback on, the allocation that no
        // longer fits is served from host RAM instead, and the only symptom is
        // wall time -- six times realtime where the same stage a thousand
        // tokens shorter runs under one. The budget predicts the spills it
        // knows about; this catches the ones it got wrong.
        if (low < kSpillFloorBytes) {
            report_spilling(name, low);
        }
    }

    // Free device memory below which the driver is assumed to be serving
    // allocations from host RAM. Nothing reports the fallback directly, so the
    // floor it leaves behind is the only evidence there is.
    static constexpr int64_t kSpillFloorBytes = 128LL << 20;

    void report_spilling(const char * name, int64_t low_bytes) {
        const std::string line =
            std::string(name) + ": USING CPU/RAM -- device memory fell to " +
            std::to_string(low_bytes >> 20) +
            " MiB during this stage, so the driver served the rest from system RAM; that is why "
            "this run is several times slower than its length alone would explain";
        std::fprintf(stderr, "[yue2] %s\n", line.c_str());
        if (active_control != nullptr) {
            active_control->log(line);
        }
        budget_notes.push_back(line);
    }

    class StageWatch {
    public:
        StageWatch(Impl & owner, const char * name) : owner_(owner), name_(name) {
            before_ = owner_.free_device_bytes();
            low_.store(before_);
            if (before_ <= 0) {
                return;
            }
            sampler_ = std::thread([this] {
                while (!stop_.load()) {
                    const int64_t now = owner_.free_device_bytes();
                    if (now > 0 && now < low_.load()) {
                        low_.store(now);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
        }

        // The token count a stage is charged against is often known only once it
        // has finished, so it arrives late rather than at construction.
        void charge(int64_t tokens) { tokens_ = tokens; }

        ~StageWatch() {
            stop_.store(true);
            if (sampler_.joinable()) {
                sampler_.join();
            }
            owner_.report_stage(name_, before_, low_.load(), tokens_);
        }

        StageWatch(const StageWatch &) = delete;
        StageWatch & operator=(const StageWatch &) = delete;

    private:
        Impl & owner_;
        const char * name_;
        int64_t before_ = 0;
        int64_t tokens_ = 0;
        std::atomic<int64_t> low_{0};
        std::atomic<bool> stop_{false};
        std::thread sampler_;
    };

    // Applies a cap in place and records why, so a short take is explained
    // rather than mysterious. One line per stage, on stderr rather than the
    // timing log, because the number that decides whether a run survives should
    // be in the server log without anyone having to turn tracing on.
    void apply_cap(const char * name, int64_t & window, int64_t cap, const std::string & detail) {
        const auto fitted = yue2_stage_window(name, window, cap);
        std::fprintf(stderr, "[yue2] %s: %s, window %lld -> %lld\n", name, detail.c_str(),
                     static_cast<long long>(window), static_cast<long long>(fitted.tokens));
        // The same line to whoever is waiting on the run. The server log has had
        // it all along; the person waiting for a song has not, and this is the
        // one number that explains a take being shorter than it was asked for.
        if (active_control != nullptr) {
            active_control->log(std::string(name) + ": " + detail + ", window " +
                                std::to_string(window) + " -> " + std::to_string(fitted.tokens));
        }
        if (!fitted.clamped) {
            return;
        }
        window = fitted.tokens;
        budget_notes.push_back(fitted.note);
        engine::debug::timing_log_scalar(std::string("yue2.budget.") + name, fitted.tokens);
    }

    // A window that survived the cap is not always a window that fits: a
    // sampling floor outranks the budget, and so does a caller who set one. The
    // driver's system-memory fallback means such a run no longer dies -- it
    // crawls instead, the difference paged over PCIe, several times slower for
    // the whole stage. That is worth saying out loud while it happens rather
    // than leaving someone to guess why a take is taking twenty minutes.
    void warn_if_spilling(const char * name, int64_t window, int64_t cap) {
        if (cap <= 0 || window <= cap) {
            return;
        }
        const std::string line =
            std::string(name) + ": USING CPU/RAM -- " + std::to_string(window) +
            " tokens is above the " + std::to_string(cap) +
            " that fit in device memory, so the driver spills the difference to system RAM; "
            "expect this run to take several times longer";
        std::fprintf(stderr, "[yue2] %s\n", line.c_str());
        if (active_control != nullptr) {
            active_control->log(line);
        }
        budget_notes.push_back(line);
    }

    void fit_stage(const char * name, int64_t & window, int64_t sequences, int64_t cost_weight) {
        const int64_t free_bytes = free_device_bytes();
        apply_cap(name, window, stage_token_cap(sequences, cost_weight),
                  std::to_string(free_bytes >> 20) + " free MiB, " + std::to_string(sequences) +
                      " sequences");
    }

    Yue2SemanticResult generate_semantic(const Yue2Request & request, Yue2Plan plan,
                                         runtime::RunControl * control = nullptr) {
        const auto total_start = Clock::now();
        Yue2SemanticResult out;
        out.plan = std::move(plan);
        if (!request.semantic_codes.empty()) {
            out.tokens.reserve(request.semantic_codes.size());
            for (const int32_t code : request.semantic_codes) {
                out.tokens.push_back(code + kCodecOffset);
            }
            engine::debug::timing_log_scalar("yue2.semantic.input_codes", request.semantic_codes.size());
            engine::debug::timing_log_scalar("yue2.semantic.tokens", out.tokens.size());
            engine::debug::timing_log_scalar("yue2.semantic.total_inner_ms", engine::debug::elapsed_ms(total_start));
            return out;
        }
        // Seed mode runs the planner too: the supplied score is its opening, not
        // its replacement.
        if (request.cot != Yue2CotMode::Off &&
            (request.abc.empty() || request.abc_mode == Yue2AbcMode::Seed)) {
            report_free("plan");
            ensure_ar();
            report_free("plan_ar_loaded");
            const auto abc_start = Clock::now();
            auto abc = abc_window(request.generation);
            fit_stage("abc_max_tokens", abc.max_tokens, 1, kYue2DecodeCostWeight);
            // A score is not paid for once. Every token of it is prefix for the
            // singing that follows, and then prefix again for every acoustic
            // chunk, which re-prefills the AR over score and chunk together.
            // Since the acoustic stage was chunked that cost no longer falls on
            // the song's length -- it falls on the chunk, and a score long
            // enough drives the chunk down to a few seconds, where the overlap
            // costs more than the music it carries.
            const int64_t plan_budget = stage_token_cap(1, kYue2DecodeCostWeight);
            if (plan_budget > 0) {
                apply_cap("abc_max_tokens", abc.max_tokens,
                          yue2_score_tokens_for(plan_budget,
                                                static_cast<int64_t>(out.plan.prefix.size())),
                          "the longest score that still leaves a 20-second acoustic chunk");
            }
            // What the model adds, which in seed mode is the continuation rather
            // than the whole. The prefix already carries the seed -- appending
            // the seed again here is what would turn "carry on from these bars"
            // into "play these bars twice".
            const auto written = ar->generate(out.plan.prefix, abc, request.seed, control);
            engine::debug::timing_log_scalar("yue2.semantic.abc_generate_ms", engine::debug::elapsed_ms(abc_start));
            engine::debug::timing_log_scalar("yue2.semantic.abc_seed_tokens", out.plan.abc_ids.size());
            out.plan.abc_ids.insert(out.plan.abc_ids.end(), written.begin(), written.end());
            // Detokenise the plan the model just wrote. Until now these ids were
            // consumed as a prefix and discarded, so the score the model composed
            // was never visible -- which is what made the documented
            // edit-the-score-and-re-render workflow impossible to reach. In seed
            // mode this is seed and continuation together: the score that was
            // actually sung, which is the one worth handing back.
            out.plan.abc = tokenizer.decode(out.plan.abc_ids);
            // Against the window actually used, not the one requested. The
            // budget caps `abc.max_tokens` well below `request.generation`'s
            // copy, so comparing with the request said "not truncated" about
            // every score this card has ever cut -- which is why a score
            // stopping mid-bar went unreported for as long as it did.
            out.plan.truncated = static_cast<int64_t>(written.size()) >= abc.max_tokens;
            out.plan.prefix.insert(out.plan.prefix.end(), written.begin(), written.end());
            out.plan.prefix.push_back(kAbcEndToken);
            out.plan.prefix.push_back(kMusicStartToken);
            engine::debug::timing_log_scalar("yue2.semantic.abc_generated_tokens", written.size());
            engine::debug::timing_log_scalar("yue2.semantic.abc_truncated", out.plan.truncated);
            if (out.plan.truncated) {
                // A cut score is a fact about the take, not a diagnostic: the
                // song stops where the score does, and the ABC handed back for
                // editing ends mid-bar. It travels with the other budget notes
                // so the run says it out loud.
                const std::string line =
                    "abc_max_tokens: the score was cut at " + std::to_string(abc.max_tokens) +
                    " tokens, so it ends mid-phrase and the take is shorter than the lyrics ask for";
                std::fprintf(stderr, "[yue2] %s\n", line.c_str());
                if (active_control != nullptr) {
                    active_control->log(line);
                }
                budget_notes.push_back(line);
            }
            ar->release_runtime_graphs();
            report_free("abc_released");
        }
        ensure_ar();
        const auto negative_start = Clock::now();
        const auto neg = negative_prefix(request, tokenizer, out.plan.abc_ids);
        engine::debug::timing_log_scalar("yue2.semantic.negative_prefix_ms", engine::debug::elapsed_ms(negative_start));
        engine::debug::timing_log_scalar("yue2.semantic.positive_prefix_tokens", out.plan.prefix.size());
        engine::debug::timing_log_scalar("yue2.semantic.negative_prefix_tokens", neg.size());
        const auto music_start = Clock::now();
        auto semantic = semantic_window(request.generation);
        // Every token decoded here is paid for twice: once in this stage's own
        // cache -- twice over when guidance keeps a negative one -- and again in
        // the acoustic stage, which prefills the AR over the whole take while
        // these caches are still resident and costs about three caches a token
        // doing it. Budgeting both together here is what lets the acoustic stage
        // run in one piece, which is the only way it sounds right.
        // Guidance keeps a second sequence alive through the whole decode, so
        // every token -- prefix and generated alike -- costs one more cache.
        const int64_t guidance_extra = request_guidance_scale(request) == 1.0F ? 0 : 1;
        // The prefix is in this cache too, and under cot=full it is not small:
        // the planner's score is thousands of tokens long and every one of them
        // is prefilled before the first musical token is sampled. It costs a
        // decode cache each and nothing downstream -- the acoustic stage is
        // given the music, never the score -- so charging it at the generated
        // rate is what turned cot=full into eight seconds of audio while cot=off
        // ran for minutes on the same card.
        const int64_t prefix_tokens =
            std::max<int64_t>(static_cast<int64_t>(out.plan.prefix.size()),
                              static_cast<int64_t>(neg.size()));
        const int64_t free_bytes = free_device_bytes();
        const int64_t budget = stage_token_cap(1, kYue2DecodeCostWeight);
        // Only this stage is charged against the length any more. The acoustic
        // stage that follows is solved in chunks sized to the card, so what it
        // costs stops depending on how long the take is -- and the two never
        // hold the device at the same time, since the semantic cache is freed
        // before the acoustic stage builds anything. Adding the two, which is
        // what this did, reserved a peak through a stage that had already
        // released it and then shortened the take to pay for it twice.
        const int64_t semantic_cost =
            kYue2SemanticTokenCost + guidance_extra * kYue2SemanticGuidanceCost;
        const Yue2StageCost stages[] = {
            Yue2StageCost{semantic_cost, semantic_cost, 0},
        };
        const int64_t cap = yue2_generated_tokens_for(budget, prefix_tokens, stages,
                                                      static_cast<int64_t>(sizeof(stages) / sizeof(stages[0])));
        const std::string detail =
            std::to_string(free_bytes >> 20) + " free MiB, " + std::to_string(prefix_tokens) +
            " prefix tokens at " + std::to_string(semantic_cost) + " milli-caches each";
        if (budget > 0 && cap <= 0) {
            // Zero means two different things -- "the budget could not be
            // measured" and "the prefix alone does not fit" -- and a zero cap is
            // read downstream as no cap at all. Conflating them is what let a
            // long score run the full 9000 tokens and then die in the acoustic
            // stage, twenty minutes in, having produced nothing. Sing the floor
            // and say why instead.
            apply_cap("semantic_max_tokens", semantic.max_tokens, semantic.min_tokens,
                      detail + " -- the prefix alone does not fit, so only the floor is left");
        } else {
            apply_cap("semantic_max_tokens", semantic.max_tokens, cap, detail);
        }
        semantic.max_tokens = std::max<int64_t>(semantic.min_tokens, semantic.max_tokens);
        warn_if_spilling("semantic_max_tokens", prefix_tokens + semantic.max_tokens,
                         cap > 0 ? prefix_tokens + cap : 0);
        {
            StageWatch watch(*this, "semantic");
            out.tokens = ar->generate_cfg(
                out.plan.prefix,
                neg,
                semantic,
                request_guidance_scale(request),
                request.seed,
                control);
            // Charged against prefix + generated: that is what the cache holds.
            watch.charge(prefix_tokens + static_cast<int64_t>(out.tokens.size()));
        }
        engine::debug::timing_log_scalar("yue2.semantic.music_generate_ms", engine::debug::elapsed_ms(music_start));
        // Against the window used, not the one asked for -- the same correction
        // the score needed, and for the same reason: the budget lowers this one
        // too, so comparing with the request called every capped take
        // "finished".
        out.truncated = static_cast<int64_t>(out.tokens.size()) >= semantic.max_tokens;
        if (out.truncated) {
            // The song did not end, it ran out. Worth saying plainly: the take
            // stops mid-phrase and the only fix is a larger window -- less
            // headroom, or a shorter score sharing the budget.
            const std::string line =
                "semantic_max_tokens: the take reached its ceiling of " +
                std::to_string(semantic.max_tokens) + " tokens (" +
                std::to_string(semantic.max_tokens / 25) +
                " seconds) and stops there rather than ending; lower vram_headroom_mb or shorten "
                "the score to buy a longer one";
            std::fprintf(stderr, "[yue2] %s\n", line.c_str());
            if (active_control != nullptr) {
                active_control->log(line);
            }
            budget_notes.push_back(line);
        }
        engine::debug::timing_log_scalar("yue2.semantic.tokens", out.tokens.size());
        engine::debug::timing_log_scalar("yue2.semantic.truncated", out.truncated);
        engine::debug::timing_log_scalar("yue2.semantic.total_inner_ms", engine::debug::elapsed_ms(total_start));
        ar->release_runtime_graphs();
        report_free("semantic_released");
        return out;
    }

    std::vector<float> synthesize_latents(
        const Yue2SemanticResult & semantic,
        const Yue2GenerationConfig & generation,
        const std::vector<float> & noise,
        uint64_t seed,
        runtime::RunControl * control = nullptr) {
        const auto codec_start = Clock::now();
        const auto codec = codec_from_semantic_tokens(semantic.tokens);
        engine::debug::timing_log_scalar("yue2.nar.codec_extract_ms", engine::debug::elapsed_ms(codec_start));
        engine::debug::timing_log_scalar("yue2.nar.codec_tokens", codec.size());
        if (codec.empty()) {
            throw std::runtime_error("Yue2 semantic generation produced no codec tokens");
        }
        // The watch opens before the runtimes are built, not after: loading the
        // NAR weights and rebuilding the AR the semantic stage released are
        // both spent out of the same budget, and measuring from after them
        // hides a fixed cost the budget then fails to reserve. On this card
        // that omission was most of the margin the run was supposed to keep.
        StageWatch watch(*this, "acoustic");
        // Measured before the runtimes are built, for the same reason the watch
        // opens here: `kYue2AcousticFixedTokens` exists precisely to reserve the
        // NAR load and the AR rebuild, so a budget read after them has already
        // paid that cost once and the fixed term then charges it again. On this
        // card the double charge was 964 MiB -- 8800 cache-tokens of chunk,
        // which is most of a chunk and, through the score cap that is sized off
        // the same arithmetic, most of the score.
        const int64_t chunk_budget = stage_token_cap(1, kYue2DecodeCostWeight);
        ensure_nar();
        ensure_ar();
        report_free("acoustic_ready");
        // Chunked, and sized here rather than by the model's context window,
        // which on this card is some five times what the memory allows. This
        // used to be left unbudgeted on the grounds that splitting a take is
        // audible -- each chunk was solved against the style prefix and nothing
        // of the chunk before it, so the seams arrived as a lurch. Carrying an
        // overlap through the solver and discarding its latents removes that
        // reason, and with it the need to shorten every take until the whole of
        // it fits the card at once.
        int64_t context = generation.context;
        const int64_t prefix_tokens = static_cast<int64_t>(semantic.plan.prefix.size());
        const int64_t fitted_chunk = yue2_acoustic_chunk_for(chunk_budget, prefix_tokens,
                                                             kYue2AcousticOverlapTokens);
        // A chunk of zero means the prefix alone does not fit, and the runtime
        // below reads zero as "no limit" -- the same conflation that once let a
        // score run to nine thousand tokens. Chunk at the floor instead and let
        // the stage say it is spilling: a take split into pieces that each
        // overflow a little is far cheaper than one piece that overflows
        // entirely.
        int64_t max_chunk = fitted_chunk > 0 ? fitted_chunk : kYue2MinAcousticChunkTokens;
        if (fitted_chunk <= 0) {
            warn_if_spilling("acoustic", prefix_tokens + max_chunk, prefix_tokens);
        }
        const int64_t take = static_cast<int64_t>(codec.size());
        // Asked for outright: one window, whatever the budget worked out. The
        // model's own window caps a chunk at (context - prefix - 3) / 2 inside
        // `chunk_ranges`, so widening the budget alone would not do it -- the
        // context has to be widened to match or the take splits anyway, which is
        // the failure mode where a checkbox silently does nothing.
        if (generation.acoustic_single_window && take > 0) {
            context = std::max(context, 2 * take + prefix_tokens + 3);
            if (take > max_chunk) {
                const std::string line =
                    "acoustic_single_window: solving all " + std::to_string(take) +
                    " frames in one window, though only " + std::to_string(max_chunk) +
                    " fit device memory -- the seam is gone and the driver serves the "
                    "difference from system RAM, so expect this stage to take several times "
                    "longer; clear the option to let it chunk again";
                std::fprintf(stderr, "[yue2] %s\n", line.c_str());
                if (active_control != nullptr) {
                    active_control->log(line);
                }
                budget_notes.push_back(line);
            }
            max_chunk = take;
        }
        if (take > max_chunk) {
            const int64_t chunks = (take + max_chunk - 1) / max_chunk;
            const std::string line =
                "acoustic: solving " + std::to_string(take) + " frames in " +
                std::to_string(chunks) + " chunks of at most " + std::to_string(max_chunk) +
                ", each carrying " + std::to_string(kYue2AcousticOverlapTokens) +
                " frames of the one before it, so the take fits device memory whole";
            std::fprintf(stderr, "[yue2] %s\n", line.c_str());
            if (active_control != nullptr) {
                active_control->log(line);
            }
        }
        // Charged per generated token, the unit the semantic budget reserves on
        // its behalf. This is the stage that actually runs out of memory, and
        // until now it was the only one with no measurement at all.
        watch.charge(static_cast<int64_t>(semantic.tokens.size()));
        return nar->synthesize(
            semantic.plan.prefix,
            codec,
            [this](const std::vector<int32_t> & tokens) {
                return ar->prefill_device_state(tokens);
            },
            noise,
            seed,
            generation.ode_steps,
            context,
            max_chunk,
            kYue2AcousticOverlapTokens,
            control);
    }

    runtime::AudioBuffer decode_audio(const std::vector<float> & latents, int64_t frames,
                                      runtime::RunControl * control = nullptr) {
        ensure_vae();
        const int64_t channels = assets->config.model.latent_dim;
        if (frames <= 0 || static_cast<int64_t>(latents.size()) != frames * channels) {
            throw std::runtime_error("Yue2 VAE latent shape mismatch");
        }
        const int64_t core_frames = assets->config.vae.decode_core_frames;
        const int64_t halo_frames = assets->config.vae.decode_halo_frames;
        const int64_t ratio = assets->config.vae.downsampling_ratio;
        if (core_frames <= 0 || halo_frames < 0 || ratio <= 0) {
            throw std::runtime_error("Yue2 VAE tile configuration is invalid");
        }
        engine::debug::timing_log_scalar("yue2.vae_decode.latent_frames", frames);
        engine::debug::timing_log_scalar("yue2.vae_decode.channels", channels);
        double planar_pack_ms = 0.0;

        auto make_planar_tile = [&](int64_t begin, int64_t end) {
            const auto start = Clock::now();
            const int64_t tile_frames = end - begin;
            std::vector<float> planar(static_cast<size_t>(channels * tile_frames), 0.0F);
            for (int64_t t = 0; t < tile_frames; ++t) {
                for (int64_t c = 0; c < channels; ++c) {
                    planar[static_cast<size_t>(c * tile_frames + t)] =
                        latents[static_cast<size_t>((begin + t) * channels + c)];
                }
            }
            planar_pack_ms += engine::debug::elapsed_ms(start);
            return planar;
        };

        if (frames <= core_frames) {
            const auto decode_start = Clock::now();
            auto audio = vae->decode(make_planar_tile(0, frames), 1, frames).front();
            engine::debug::timing_log_scalar("yue2.vae_decode.tiles", 1);
            engine::debug::timing_log_scalar("yue2.vae_decode.planar_pack_ms", planar_pack_ms);
            engine::debug::timing_log_scalar("yue2.vae_decode.tile_decode_ms", engine::debug::elapsed_ms(decode_start));
            engine::debug::timing_log_scalar("yue2.vae_decode.tile_copy_ms", 0.0);
            engine::debug::timing_log_scalar("yue2.vae_decode.output_frames", static_cast<int64_t>(audio.samples.size()) / audio.channels);
            return audio;
        }

        const int64_t total_output_frames = frames * ratio - 64;
        if (total_output_frames <= 0) {
            throw std::runtime_error("Yue2 VAE output frame count is invalid");
        }
        runtime::AudioBuffer audio;
        audio.sample_rate = assets->config.vae.sample_rate;
        audio.channels = static_cast<int>(assets->config.vae.channels);
        audio.samples.assign(static_cast<size_t>(total_output_frames * audio.channels), 0.0F);
        int64_t tiles = 0;
        double tile_decode_ms = 0.0;
        double tile_copy_ms = 0.0;
        const int64_t tile_total = core_frames > 0 ? (frames + core_frames - 1) / core_frames : 0;
        for (int64_t start = 0; start < frames; start += core_frames) {
            // A tile is roughly forty seconds of audio through the decoder, so
            // this is the coarsest of the four cancel points -- but it is also
            // the last stage, where a cancel is least likely to be wanted.
            if (control != nullptr) {
                control->throw_if_cancelled();
                control->progress(runtime::ProgressEvent{"decode", tiles, tile_total, ""});
            }
            const int64_t end = std::min(frames, start + core_frames);
            const int64_t left = std::max<int64_t>(0, start - halo_frames);
            const int64_t right = std::min(frames, end + halo_frames);
            const auto tile_decode_start = Clock::now();
            auto tile_audio = vae->decode(make_planar_tile(left, right), 1, right - left).front();
            tile_decode_ms += engine::debug::elapsed_ms(tile_decode_start);
            const int64_t out_start = start * ratio;
            const int64_t out_end = std::min(end * ratio, total_output_frames);
            const int64_t copy_frames = out_end - out_start;
            const int64_t crop_start = (start - left) * ratio;
            if (copy_frames <= 0 ||
                crop_start < 0 ||
                crop_start + copy_frames > static_cast<int64_t>(tile_audio.samples.size()) / tile_audio.channels) {
                throw std::runtime_error("Yue2 VAE tile did not cover output core");
            }
            const auto copy_start = Clock::now();
            for (int64_t t = 0; t < copy_frames; ++t) {
                for (int64_t c = 0; c < audio.channels; ++c) {
                    audio.samples[static_cast<size_t>((out_start + t) * audio.channels + c)] =
                        tile_audio.samples[static_cast<size_t>((crop_start + t) * tile_audio.channels + c)];
                }
            }
            tile_copy_ms += engine::debug::elapsed_ms(copy_start);
            ++tiles;
        }
        engine::debug::timing_log_scalar("yue2.vae_decode.tiles", tiles);
        engine::debug::timing_log_scalar("yue2.vae_decode.planar_pack_ms", planar_pack_ms);
        engine::debug::timing_log_scalar("yue2.vae_decode.tile_decode_ms", tile_decode_ms);
        engine::debug::timing_log_scalar("yue2.vae_decode.tile_copy_ms", tile_copy_ms);
        engine::debug::timing_log_scalar("yue2.vae_decode.output_frames", total_output_frames);
        return audio;
    }

    Yue2RunOutput run_detailed(const Yue2Request & request, runtime::RunControl * control = nullptr) {
        // Held for the duration so the budget can speak without every helper
        // below growing a parameter it would only pass along.
        active_control = control;
        struct Clear {
            runtime::RunControl ** slot;
            ~Clear() { *slot = nullptr; }
        } clear{&active_control};
        budget_notes.clear();
        vram_headroom_bytes = request.generation.vram_headroom_mb << 20;
        vram_budget_scale = request.generation.vram_budget_scale;
        const auto plan_start = Clock::now();
        auto planned = plan(request);
        engine::debug::timing_log_scalar("yue2.plan_ms", engine::debug::elapsed_ms(plan_start, Clock::now()));
        const auto semantic_start = Clock::now();
        auto semantic = generate_semantic(request, std::move(planned), control);
        engine::debug::timing_log_scalar("yue2.semantic_ms", engine::debug::elapsed_ms(semantic_start, Clock::now()));
        const auto nar_start = Clock::now();
        auto latents = synthesize_latents(semantic, request.generation, request.nar_noise, request.seed, control);
        engine::debug::timing_log_scalar("yue2.nar_ms", engine::debug::elapsed_ms(nar_start, Clock::now()));
        const int64_t frames = static_cast<int64_t>(latents.size()) / assets->config.model.latent_dim;
        ar.reset();
        nar.reset();
        const auto vae_start = Clock::now();
        auto audio = decode_audio(latents, frames, control);
        engine::debug::timing_log_scalar("yue2.vae_decode_ms", engine::debug::elapsed_ms(vae_start, Clock::now()));
        if (vae) {
            vae->release_runtime_graphs();
        }

        Yue2RunOutput out;
        out.audio = std::move(audio);
        out.abc = semantic.plan.abc;
        // Offset already stripped, so these feed straight back into a request.
        out.semantic_codes = codec_from_semantic_tokens(semantic.tokens);
        out.abc_truncated = semantic.plan.truncated;
        out.semantic_truncated = semantic.truncated;
        out.budget_notes = budget_notes;
        return out;
    }

    runtime::AudioBuffer run(const Yue2Request & request, runtime::RunControl * control = nullptr) {
        return run_detailed(request, control).audio;
    }

    void release_runtime_graphs() {
        if (vae) {
            vae->release_runtime_graphs();
        }
        if (ar) {
            ar->release_runtime_graphs();
        }
        if (nar) {
            nar->release_runtime_graphs();
        }
    }

private:
    void ensure_ar() {
        if (ar) {
            return;
        }
        const auto start = Clock::now();
        ar = std::make_unique<Yue2ArRuntime>(
            *execution,
            assets,
            model_weight_type,
            model_weight_context_bytes,
            ar_prefill_graph_arena_bytes,
            ar_decode_graph_arena_bytes);
        engine::debug::timing_log_scalar("yue2.ar.init_ms", engine::debug::elapsed_ms(start));
    }

    void ensure_vae() {
        if (vae) {
            return;
        }
        codecs::OobleckAudioVaeConfig config;
        config.sample_rate = assets->config.vae.sample_rate;
        config.audio_channels = assets->config.vae.channels;
        config.encoder_latent_dim = assets->config.vae.encoder_latent_dim;
        config.decoder_latent_dim = assets->config.vae.latent_dim;
        config.encoder_prefix = "encoder";
        config.decoder_prefix = "decoder";
        codecs::OobleckAudioVaeRuntimeOptions options;
        options.weight_context_bytes = vae_weight_context_bytes;
        options.graph_arena_bytes = vae_graph_arena_bytes;
        options.weight_storage_type = vae_weight_type;
        const auto start = Clock::now();
        vae = std::make_unique<codecs::OobleckAudioVaeRuntime>(
            assets->vae_weights,
            *execution,
            std::move(config),
            options);
        engine::debug::timing_log_scalar("yue2.vae.init_ms", engine::debug::elapsed_ms(start));
    }

    void ensure_nar() {
        if (nar) {
            return;
        }
        const auto start = Clock::now();
        nar = std::make_unique<Yue2NarRuntime>(
            *execution,
            assets,
            model_weight_type,
            model_weight_context_bytes,
            nar_graph_arena_bytes);
        engine::debug::timing_log_scalar("yue2.nar.init_ms", engine::debug::elapsed_ms(start));
    }

    core::ExecutionContext * execution = nullptr;
    std::vector<std::string> budget_notes;
    runtime::RunControl * active_control = nullptr;
    std::shared_ptr<const Yue2Assets> assets;
    Yue2TextTokenizer tokenizer;
    assets::TensorStorageType model_weight_type = assets::TensorStorageType::Native;
    assets::TensorStorageType vae_weight_type = assets::TensorStorageType::Native;
    size_t model_weight_context_bytes = 0;
    size_t vae_weight_context_bytes = 0;
    size_t ar_prefill_graph_arena_bytes = 0;
    size_t ar_decode_graph_arena_bytes = 0;
    size_t nar_graph_arena_bytes = 0;
    size_t vae_graph_arena_bytes = 0;
    std::unique_ptr<codecs::OobleckAudioVaeRuntime> vae;
    std::unique_ptr<Yue2ArRuntime> ar;
    std::unique_ptr<Yue2NarRuntime> nar;
};

Yue2PipelineRuntime::Yue2PipelineRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const Yue2Assets> assets,
    assets::TensorStorageType model_weight_type,
    assets::TensorStorageType vae_weight_type,
    size_t model_weight_context_bytes,
    size_t vae_weight_context_bytes,
    size_t ar_prefill_graph_arena_bytes,
    size_t ar_decode_graph_arena_bytes,
    size_t nar_graph_arena_bytes,
    size_t vae_graph_arena_bytes)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          model_weight_type,
          vae_weight_type,
          model_weight_context_bytes,
          vae_weight_context_bytes,
          ar_prefill_graph_arena_bytes,
          ar_decode_graph_arena_bytes,
          nar_graph_arena_bytes,
          vae_graph_arena_bytes)) {}

Yue2PipelineRuntime::~Yue2PipelineRuntime() = default;

Yue2Plan Yue2PipelineRuntime::plan(const Yue2Request & request) {
    return impl_->plan(request);
}

Yue2SemanticResult Yue2PipelineRuntime::generate_semantic(const Yue2Request & request,
                                                          Yue2Plan plan,
                                                          runtime::RunControl * control) {
    return impl_->generate_semantic(request, std::move(plan), control);
}

std::vector<float> Yue2PipelineRuntime::synthesize_latents(
    const Yue2SemanticResult & semantic,
    const Yue2GenerationConfig & generation,
    const std::vector<float> & noise,
    uint64_t seed,
    runtime::RunControl * control) {
    // `noise` used to be dropped here: the wrapper passed a static empty vector,
    // so anything driving the pipeline stage by stage silently lost the
    // teacher-forced noise that nar_noise_file exists to supply.
    return impl_->synthesize_latents(semantic, generation, noise, seed, control);
}

runtime::AudioBuffer Yue2PipelineRuntime::decode_audio(const std::vector<float> & latents,
                                                      int64_t frames,
                                                      runtime::RunControl * control) {
    return impl_->decode_audio(latents, frames, control);
}

Yue2RunOutput Yue2PipelineRuntime::run_detailed(const Yue2Request & request, runtime::RunControl * control) {
    return impl_->run_detailed(request, control);
}

runtime::AudioBuffer Yue2PipelineRuntime::run(const Yue2Request & request, runtime::RunControl * control) {
    return impl_->run(request, control);
}

void Yue2PipelineRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::yue2
