#pragma once

#include "engine/framework/runtime/run_control.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/yue2/ar_runtime.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace engine::models::yue2 {

class Yue2NarRuntime {
public:
    Yue2NarRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType weight_type,
        size_t weight_context_bytes,
        size_t graph_arena_bytes);
    ~Yue2NarRuntime();

    std::vector<float> synthesize(
        const std::vector<int32_t> & prefix,
        const std::vector<int32_t> & codec,
        const std::function<Yue2ArDevicePrefixState(const std::vector<int32_t> &)> & prefill_state,
        const std::vector<float> & noise,
        uint64_t seed,
        int64_t ode_steps,
        int64_t context,
        // The largest span solved in one piece, and how much of the span before
        // it is re-solved as context and then discarded. Zero for either leaves
        // the model's own context window as the only limit -- which is what the
        // take length used to have to be budgeted against instead.
        int64_t max_chunk_frames,
        int64_t overlap_frames,
        runtime::RunControl * control = nullptr);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::yue2
