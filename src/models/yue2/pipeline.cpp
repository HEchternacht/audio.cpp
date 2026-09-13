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
    if (!request.abc.empty()) {
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
    int64_t vram_headroom_bytes = 640LL << 20;
    double vram_budget_scale = 1.0;

    int64_t cache_bytes_per_token() const {
        return yue2_cache_bytes_per_token(Yue2CacheDims{
            assets->config.model.layers,
            assets->config.model.kv_heads,
            assets->config.model.head_dim,
            yue2_ar_cache_bytes_per_element(execution->backend_type())});
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
    void report_stage(const char * name, int64_t before, int64_t low, int64_t tokens) const {
        if (before <= 0 || low <= 0 || tokens <= 0) {
            return;
        }
        const int64_t peak = before - low;
        const int64_t per_token = peak / tokens;
        std::fprintf(stderr, "[yue2] %s peak %lld MiB over %lld tokens (%lld KiB/token, %.2f caches)\n",
                     name, static_cast<long long>(peak >> 20), static_cast<long long>(tokens),
                     static_cast<long long>(per_token >> 10),
                     static_cast<double>(per_token) / static_cast<double>(cache_bytes_per_token()));
        engine::debug::timing_log_scalar(std::string("yue2.peak.") + name, peak);
        engine::debug::timing_log_scalar(std::string("yue2.peak_per_token.") + name, per_token);
    }

    class StageWatch {
    public:
        StageWatch(const Impl & owner, const char * name) : owner_(owner), name_(name) {
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
        const Impl & owner_;
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
        if (!fitted.clamped) {
            return;
        }
        window = fitted.tokens;
        budget_notes.push_back(fitted.note);
        engine::debug::timing_log_scalar(std::string("yue2.budget.") + name, fitted.tokens);
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
        if (request.cot != Yue2CotMode::Off && request.abc.empty()) {
            ensure_ar();
            const auto abc_start = Clock::now();
            auto abc = abc_window(request.generation);
            fit_stage("abc_max_tokens", abc.max_tokens, 1, kYue2DecodeCostWeight);
            out.plan.abc_ids = ar->generate(out.plan.prefix, abc, request.seed, control);
            engine::debug::timing_log_scalar("yue2.semantic.abc_generate_ms", engine::debug::elapsed_ms(abc_start));
            // Detokenise the plan the model just wrote. Until now these ids were
            // consumed as a prefix and discarded, so the score the model composed
            // was never visible -- which is what made the documented
            // edit-the-score-and-re-render workflow impossible to reach.
            out.plan.abc = tokenizer.decode(out.plan.abc_ids);
            out.plan.truncated = static_cast<int64_t>(out.plan.abc_ids.size()) >= request.generation.abc.max_tokens;
            out.plan.prefix.insert(out.plan.prefix.end(), out.plan.abc_ids.begin(), out.plan.abc_ids.end());
            out.plan.prefix.push_back(kAbcEndToken);
            out.plan.prefix.push_back(kMusicStartToken);
            engine::debug::timing_log_scalar("yue2.semantic.abc_generated_tokens", out.plan.abc_ids.size());
            engine::debug::timing_log_scalar("yue2.semantic.abc_truncated", out.plan.truncated);
            ar->release_runtime_graphs();
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
        const int64_t cap = yue2_generated_tokens_for(stage_token_cap(1, kYue2DecodeCostWeight),
                                                      prefix_tokens,
                                                      kYue2PrefixCostWeight + guidance_extra,
                                                      kYue2GeneratedCostWeight + guidance_extra);
        apply_cap("semantic_max_tokens", semantic.max_tokens, cap,
                  std::to_string(free_bytes >> 20) + " free MiB, " + std::to_string(prefix_tokens) +
                      " prefix tokens at " + std::to_string(kYue2PrefixCostWeight + guidance_extra) +
                      " caches, " + std::to_string(kYue2GeneratedCostWeight + guidance_extra) +
                      " per generated");
        semantic.max_tokens = std::max<int64_t>(semantic.min_tokens, semantic.max_tokens);
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
        out.truncated = static_cast<int64_t>(out.tokens.size()) >= request.generation.semantic.max_tokens;
        engine::debug::timing_log_scalar("yue2.semantic.tokens", out.tokens.size());
        engine::debug::timing_log_scalar("yue2.semantic.truncated", out.truncated);
        engine::debug::timing_log_scalar("yue2.semantic.total_inner_ms", engine::debug::elapsed_ms(total_start));
        ar->release_runtime_graphs();
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
        ensure_nar();
        ensure_ar();
        // Deliberately unbudgeted. Splitting the take into chunks is the one
        // economy that is audible: each chunk is solved against its own window
        // only, so the seams arrive as a lurch every few seconds. A take that is
        // shorter but whole is worth more than a long one that is stitched, so
        // the length is budgeted instead -- at the semantic stage, which is
        // where the decision can still be made -- and this window is left at
        // whatever the caller asked for.
        const int64_t context = generation.context;
        // Charged per generated token, the unit the semantic budget reserves on
        // its behalf. This is the stage that actually runs out of memory, and
        // until now it was the only one with no measurement at all.
        StageWatch watch(*this, "acoustic");
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
