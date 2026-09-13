#pragma once

#include "engine/framework/runtime/run_control.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/types.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace engine::models::yue2 {

struct Yue2ArSamplingWindow {
    int32_t begin = 0;
    int32_t end = 0;
    int32_t stop_token = 0;
    int64_t min_tokens = 0;
    int64_t max_tokens = 0;
    Yue2SamplingConfig sampling;
};

struct Yue2ArDevicePrefixState {
    int64_t current_end = 0;
    std::vector<core::TensorValue> keys;
    std::vector<core::TensorValue> values;
};

class Yue2ArRuntime {
public:
    Yue2ArRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType weight_type,
        size_t weight_context_bytes,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes);
    ~Yue2ArRuntime();

    // `control` is optional and defaulted, so every existing caller -- the CLI,
    // the parity probes, the benchmarks -- is unchanged. When present it is
    // polled once per decoded token: that is the only granularity available,
    // since ggml has no CUDA abort callback and a graph already in flight cannot
    // be interrupted.
    std::vector<int32_t> generate(
        const std::vector<int32_t> & prefix,
        const Yue2ArSamplingWindow & window,
        uint64_t seed,
        runtime::RunControl * control = nullptr);

    std::vector<int32_t> generate_cfg(
        const std::vector<int32_t> & positive_prefix,
        const std::vector<int32_t> & negative_prefix,
        const Yue2ArSamplingWindow & window,
        float guidance_scale,
        uint64_t seed,
        runtime::RunControl * control = nullptr);

    runtime::TransformerKVState prefill_state(const std::vector<int32_t> & tokens);
    Yue2ArDevicePrefixState prefill_device_state(const std::vector<int32_t> & tokens);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::yue2
