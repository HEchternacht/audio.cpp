#pragma once

// One concern: which models are resident, and running one task against one.
//
// This is the API server's equivalent of the residency machinery buried in
// ServerState, minus everything the API does not have: no voice presets, no
// browser uploads, no package installer. Those are most of what entangles the
// original, so writing this focused version is less code than de-tangling it --
// and it leaves the existing audiocpp_server untouched, which is the point of
// keeping it buildable.
//
// Serialisation of two jobs against one model is NOT this file's job: the queue's
// model-claim set does that before a worker ever gets here. BusyGuard stays as a
// safety net so a manual load can never race a running job.

#include "busy_guard.h"
#include "config.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/run_control.h"
#include "engine/framework/runtime/session.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace audiocpp::api {

// A model as configured, plus its session once loaded. One session per model,
// which is exactly why two jobs for one model must never overlap: the session
// holds mutable graph arenas and KV state.
struct PooledModel {
    minitts::server::ServerModelConfig config;
    engine::runtime::TaskSpec task;
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
    engine::runtime::IOfflineVoiceTaskSession * offline = nullptr;
    engine::runtime::IStreamingVoiceTaskSession * streaming = nullptr;
    std::atomic<bool> loaded{false};
    std::atomic<int64_t> last_used_ms{0};
    minitts::server::BusyGuard busy;

    void unload();
};

struct ModelSummary {
    std::string id;
    std::string family;
    std::string task;
    std::string mode;
    std::string path;
    bool loaded = false;
    bool lazy = false;
    int64_t last_used_ms = 0;
};

class ModelPool {
public:
    explicit ModelPool(const minitts::server::ServerConfig & config);
    ~ModelPool();

    // Registers every configured model. Eager entries load now; lazy ones on
    // first use. Throws on a duplicate id.
    void register_models();

    bool has(const std::string & id) const;
    std::vector<ModelSummary> summaries() const;
    std::optional<ModelSummary> summary(const std::string & id) const;
    // The family a model is configured as, without loading it. Needed at submit
    // time to pick the right operation table.
    std::string family_of(const std::string & id) const;

    // Loads if needed and runs. `control` may be null. Throws
    // engine::runtime::RunCancelled when the run observed a cancel, ServerBusyError
    // or InsufficientMemoryError when the model could not be made ready.
    engine::runtime::TaskResult run(const std::string & id,
                                    const engine::runtime::TaskRequest & request,
                                    const std::shared_ptr<engine::runtime::RunControl> & control);

    // Explicit load/unload behind the same guards the run path uses.
    void load(const std::string & id);
    bool unload(const std::string & id);
    int unload_all_idle();

    int loaded_count() const;

private:
    PooledModel * find(const std::string & id) const;
    void ensure_loaded(PooledModel & model);
    void evict_for_limit(const PooledModel & loading);
    void ensure_fits_memory(const minitts::server::ServerModelConfig & config);

    minitts::server::ServerConfig config_;
    mutable std::mutex mutex_;
    std::mutex load_mutex_;
    std::vector<std::unique_ptr<PooledModel>> models_;
};

// Pure helpers, split out so they can be tested without an engine or a GPU.

// Chooses which resident model to evict to make room, given each candidate's id
// and last-used time. Least recently used wins. Ids already in `busy` are never
// chosen -- a model mid-inference must not be pulled out from under a worker.
struct EvictionCandidate {
    std::string id;
    int64_t last_used_ms = 0;
    bool loaded = false;
    bool busy = false;
};
std::optional<std::string> choose_eviction_victim(const std::vector<EvictionCandidate> & candidates,
                                                  const std::string & loading_id);

// True when a config asks for more concurrency than its residency limit can
// serve. Two workers with room for one model turn eviction into random failures,
// so the server refuses to start instead.
bool concurrency_exceeds_residency(int workers, int max_loaded_models);
std::string concurrency_conflict_message(int workers, int max_loaded_models);

}  // namespace audiocpp::api
