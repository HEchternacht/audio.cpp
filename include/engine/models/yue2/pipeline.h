#pragma once

#include "engine/framework/runtime/run_control.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/codecs/oobleck_audio_vae_runtime.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/request.h"
#include "engine/models/yue2/tokenizer_text.h"

#include <memory>
#include <vector>

namespace engine::models::yue2 {

class Yue2PipelineRuntime {
public:
    Yue2PipelineRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType model_weight_type,
        assets::TensorStorageType vae_weight_type,
        size_t model_weight_context_bytes,
        size_t vae_weight_context_bytes,
        size_t ar_prefill_graph_arena_bytes,
        size_t ar_decode_graph_arena_bytes,
        size_t nar_graph_arena_bytes,
        size_t vae_graph_arena_bytes);
    ~Yue2PipelineRuntime();

    Yue2Plan plan(const Yue2Request & request);
    Yue2SemanticResult generate_semantic(const Yue2Request & request,
                                         Yue2Plan plan,
                                         runtime::RunControl * control = nullptr);
    std::vector<float> synthesize_latents(
        const Yue2SemanticResult & semantic,
        const Yue2GenerationConfig & generation,
        const std::vector<float> & noise,
        uint64_t seed,
        runtime::RunControl * control = nullptr);
    runtime::AudioBuffer decode_audio(const std::vector<float> & latents,
                                      int64_t frames,
                                      runtime::RunControl * control = nullptr);

    // Everything the run produced: audio, the score the model wrote, and the
    // semantic codes behind it. run() keeps returning just the audio so existing
    // callers are untouched.
    Yue2RunOutput run_detailed(const Yue2Request & request, runtime::RunControl * control = nullptr);
    runtime::AudioBuffer run(const Yue2Request & request, runtime::RunControl * control = nullptr);
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::yue2
