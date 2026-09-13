#include "model_pool.h"

#include "model_memory.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/host_memory.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>

namespace audiocpp::api {
namespace {

using minitts::server::BusyGuard;
using minitts::server::InsufficientMemoryError;
using minitts::server::ServerModelConfig;

std::string format_bytes(size_t bytes) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << gib << " GiB";
    return out.str();
}

int64_t steady_now_ms() {
    return minitts::server::steady_now_ms();
}

}  // namespace

std::optional<std::string> choose_eviction_victim(const std::vector<EvictionCandidate> & candidates,
                                                  const std::string & loading_id) {
    const EvictionCandidate * best = nullptr;
    for (const auto & candidate : candidates) {
        if (!candidate.loaded || candidate.busy || candidate.id == loading_id) {
            continue;
        }
        if (best == nullptr || candidate.last_used_ms < best->last_used_ms) {
            best = &candidate;
        }
    }
    if (best == nullptr) {
        return std::nullopt;
    }
    return best->id;
}

bool concurrency_exceeds_residency(int workers, int max_loaded_models) {
    return workers > 1 && max_loaded_models > 0 && workers > max_loaded_models;
}

std::string concurrency_conflict_message(int workers, int max_loaded_models) {
    return "api.workers is " + std::to_string(workers) + " but max_loaded_models is " +
           std::to_string(max_loaded_models) +
           ": a worker would have to evict a model another worker is running, which fails the job "
           "rather than waiting. Raise max_loaded_models to at least " + std::to_string(workers) +
           ", or lower api.workers. On a 6 GB card running yue2, workers=1 is the only value that fits.";
}

void PooledModel::unload() {
    offline = nullptr;
    streaming = nullptr;
    session.reset();
    model.reset();
    loaded.store(false);
}

ModelPool::ModelPool(const minitts::server::ServerConfig & config) : config_(config) {}

ModelPool::~ModelPool() = default;

void ModelPool::register_models() {
    for (const auto & entry : config_.models) {
        if (has(entry.id)) {
            throw std::runtime_error("duplicate model id in config: " + entry.id);
        }
        auto pooled = std::make_unique<PooledModel>();
        pooled->config = entry;
        pooled->task = engine::runtime::TaskSpec{
            engine::runtime::parse_voice_task_kind(entry.task),
            engine::runtime::parse_run_mode(entry.mode),
        };
        std::lock_guard<std::mutex> lock(mutex_);
        models_.push_back(std::move(pooled));
    }

    int eager = 0;
    for (auto & pooled : models_) {
        if (pooled->config.lazy) {
            continue;
        }
        if (config_.max_loaded_models > 0 && eager >= config_.max_loaded_models) {
            // Loading now would immediately evict an earlier entry, so an
            // over-limit config would churn through doomed loads at startup.
            std::cerr << "[api] model '" << pooled->config.id
                      << "' registered but not loaded (max_loaded_models="
                      << config_.max_loaded_models << " reached); it loads on first use\n";
            continue;
        }
        ensure_loaded(*pooled);
        ++eager;
    }
}

PooledModel * ModelPool::find(const std::string & id) const {
    for (const auto & pooled : models_) {
        if (pooled->config.id == id) {
            return pooled.get();
        }
    }
    return nullptr;
}

bool ModelPool::has(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return find(id) != nullptr;
}

std::string ModelPool::family_of(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto * pooled = find(id);
    return pooled == nullptr ? std::string() : pooled->config.family;
}

std::vector<ModelSummary> ModelPool::summaries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ModelSummary> out;
    out.reserve(models_.size());
    for (const auto & pooled : models_) {
        ModelSummary summary;
        summary.id = pooled->config.id;
        summary.family = pooled->config.family;
        summary.task = pooled->config.task;
        summary.mode = pooled->config.mode;
        summary.path = pooled->config.path.string();
        summary.loaded = pooled->loaded.load();
        summary.lazy = pooled->config.lazy;
        summary.last_used_ms = pooled->last_used_ms.load();
        out.push_back(std::move(summary));
    }
    return out;
}

std::optional<ModelSummary> ModelPool::summary(const std::string & id) const {
    for (auto & entry : summaries()) {
        if (entry.id == id) {
            return entry;
        }
    }
    return std::nullopt;
}

int ModelPool::loaded_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (const auto & pooled : models_) {
        if (pooled->loaded.load()) {
            ++count;
        }
    }
    return count;
}

void ModelPool::evict_for_limit(const PooledModel & loading) {
    for (;;) {
        std::vector<EvictionCandidate> candidates;
        int resident = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto & pooled : models_) {
                if (!pooled->loaded.load()) {
                    continue;
                }
                ++resident;
                // try_acquire is the probe that keeps a running model safe: if we
                // cannot take its lock, it is mid-inference and not a candidate.
                auto probe = pooled->busy.try_acquire();
                candidates.push_back(EvictionCandidate{pooled->config.id,
                                                       pooled->last_used_ms.load(),
                                                       true,
                                                       !probe.has_value()});
            }
        }
        if (resident < config_.max_loaded_models) {
            return;
        }
        const auto victim = choose_eviction_victim(candidates, loading.config.id);
        if (!victim.has_value()) {
            throw minitts::server::ServerBusyError(
                "cannot load model '" + loading.config.id + "': the residency limit of " +
                std::to_string(config_.max_loaded_models) +
                " is reached and every resident model is busy");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto * target = find(*victim);
        if (target == nullptr) {
            return;
        }
        auto lock_held = target->busy.try_acquire();
        if (!lock_held.has_value()) {
            continue;   // it became busy between the probe and here; re-evaluate
        }
        std::cerr << "[api] unloading '" << target->config.id << "' to make room for '"
                  << loading.config.id << "'\n";
        target->unload();
    }
}

void ModelPool::ensure_fits_memory(const ServerModelConfig & model) {
    if (config_.min_free_memory_mb <= 0) {
        return;
    }
    const auto estimate = minitts::server::estimate_model_memory_bytes(model);
    if (!estimate.has_value()) {
        std::cerr << "[api] memory guard skipped for model '" << model.id
                  << "': indeterminate footprint (ambiguous model directory)\n";
        return;
    }
    const size_t headroom = static_cast<size_t>(config_.min_free_memory_mb) * 1024ull * 1024ull;

    const size_t host_available = engine::core::available_host_memory_bytes();
    if (host_available > 0 && *estimate + headroom > host_available) {
        throw InsufficientMemoryError(
            "cannot load model '" + model.id + "': estimated " + format_bytes(*estimate) + " + " +
            std::to_string(config_.min_free_memory_mb) + " MiB headroom exceeds available host memory (" +
            format_bytes(host_available) + ")");
    }

    engine::core::ensure_backends_loaded();
    const engine::core::BackendMemorySnapshot device = engine::core::query_backend_memory(
        engine::core::BackendConfig{config_.backend, config_.device, config_.threads});
    if (device.available && *estimate + headroom > static_cast<size_t>(device.free_bytes)) {
        throw InsufficientMemoryError(
            "cannot load model '" + model.id + "': estimated " + format_bytes(*estimate) + " + " +
            std::to_string(config_.min_free_memory_mb) + " MiB headroom exceeds available device memory (" +
            format_bytes(static_cast<size_t>(device.free_bytes)) + ")");
    }
}

void ModelPool::ensure_loaded(PooledModel & model) {
    model.last_used_ms.store(steady_now_ms());
    if (model.session != nullptr) {
        return;
    }
    // Serialise the evict -> memory-check -> load sequence only when a guard
    // actually needs it, so unrelated first loads keep their concurrency.
    const bool serialize = config_.max_loaded_models > 0 || config_.min_free_memory_mb > 0;
    std::unique_lock<std::mutex> load_lock(load_mutex_, std::defer_lock);
    if (serialize) {
        load_lock.lock();
    }
    if (model.session != nullptr) {
        return;   // another thread won the race while we waited
    }
    if (config_.max_loaded_models > 0) {
        evict_for_limit(model);
    }
    ensure_fits_memory(model.config);

    auto registry = engine::runtime::make_default_registry();

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model.config.path;
    load_request.model_spec_override = model.config.model_spec_override.has_value()
                                           ? model.config.model_spec_override
                                           : config_.model_spec_override;
    load_request.family_hint = model.config.family;
    load_request.config_id = model.config.config_id;
    load_request.weight_id = model.config.weight_id;
    load_request.options = model.config.load_options;

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = config_.backend;
    session_options.backend.device = config_.device;
    session_options.backend.threads = config_.threads;
    session_options.options = model.config.session_options;

    auto loaded_model = registry.load(load_request);
    auto session = loaded_model->create_task_session(model.task, session_options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session.get());
    if (model.task.mode == engine::runtime::RunMode::Offline && offline == nullptr) {
        throw std::runtime_error("configured model does not provide offline execution: " + model.config.id);
    }
    if (model.task.mode == engine::runtime::RunMode::Streaming && streaming == nullptr) {
        throw std::runtime_error("configured model does not provide streaming execution: " + model.config.id);
    }
    model.model = std::move(loaded_model);
    model.session = std::move(session);
    model.offline = offline;
    model.streaming = streaming;
    model.loaded.store(true);
}

void ModelPool::load(const std::string & id) {
    PooledModel * model = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        model = find(id);
    }
    if (model == nullptr) {
        throw std::runtime_error("unknown model id: " + id);
    }
    // Holding the guard means a load can never land while a job is running.
    auto held = model->busy.acquire(config_.busy_timeout_ms, model->config.id);
    ensure_loaded(*model);
}

engine::runtime::TaskResult ModelPool::run(const std::string & id,
                                           const engine::runtime::TaskRequest & request,
                                           const std::shared_ptr<engine::runtime::RunControl> & control) {
    PooledModel * model = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        model = find(id);
    }
    if (model == nullptr) {
        throw std::runtime_error("unknown model id: " + id);
    }

    // The queue's claim set means this never actually contends; it is here so a
    // manual load or unload cannot slip in beside a running job.
    auto held = model->busy.acquire(config_.busy_timeout_ms, model->config.id);
    ensure_loaded(*model);
    model->last_used_ms.store(steady_now_ms());

    if (model->offline == nullptr) {
        throw std::runtime_error("model does not support offline execution: " + id);
    }

    auto with_control = request;
    with_control.control = control;
    // Families gate run() on having been prepared, so this is not optional --
    // it is where a session learns the shape of the work it is about to do.
    model->session->prepare(engine::runtime::build_preparation_request(with_control));
    auto result = model->offline->run(with_control);
    model->last_used_ms.store(steady_now_ms());
    return result;
}

bool ModelPool::unload(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto * model = find(id);
    if (model == nullptr || !model->loaded.load()) {
        return false;
    }
    auto held = model->busy.acquire(0, model->config.id);
    model->unload();
    return true;
}

int ModelPool::unload_all_idle() {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (auto & pooled : models_) {
        if (!pooled->loaded.load()) {
            continue;
        }
        auto held = pooled->busy.try_acquire();
        if (!held.has_value()) {
            continue;   // busy; leave it alone
        }
        pooled->unload();
        ++count;
    }
    return count;
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_MODEL_POOL
// Direct-execution self-test. Only the decision logic is covered here: loading a
// real model needs weights and a GPU, so that path belongs to the end-to-end
// checks. What can be tested without either is exactly what tends to be wrong --
// which model gets evicted, and whether a config is self-defeating.
#include <cstdio>
#include <functional>

namespace {

int g_pass = 0;
int g_fail = 0;

void record(const char * unit, const std::string & in, const std::string & out, const std::string & exp) {
    std::printf("[%s] IN : %s\n[%s] OUT: %s\n[%s] EXP: %s\n",
                unit, in.c_str(), unit, out.c_str(), unit, exp.c_str());
    if (out == exp) {
        std::printf("[%s] OK\n", unit);
        ++g_pass;
    } else {
        std::printf("[%s] FAIL\n", unit);
        ++g_fail;
    }
}

void guarded(const char * unit, const std::function<void()> & body) {
    try {
        body();
    } catch (const std::exception & ex) {
        std::printf("[%s] FAIL unexpected throw: %s\n", unit, ex.what());
        ++g_fail;
    }
}

}  // namespace

int main() {
    using namespace audiocpp::api;

    guarded("choose_eviction_victim/lru", [] {
        const std::vector<EvictionCandidate> candidates{
            {"a", 300, true, false},
            {"b", 100, true, false},
            {"c", 200, true, false},
        };
        record("choose_eviction_victim/lru", "last used 300/100/200",
               choose_eviction_victim(candidates, "d").value_or("<none>"), "b");
    });

    guarded("choose_eviction_victim/skips_busy", [] {
        // The invariant that keeps a running job alive: a busy model is never a
        // victim, even when it is the least recently used.
        const std::vector<EvictionCandidate> candidates{
            {"a", 100, true, true},    // oldest, but running
            {"b", 200, true, false},
        };
        record("choose_eviction_victim/skips_busy", "oldest is busy",
               choose_eviction_victim(candidates, "d").value_or("<none>"), "b");
    });

    guarded("choose_eviction_victim/skips_unloaded", [] {
        const std::vector<EvictionCandidate> candidates{
            {"a", 100, false, false},   // not resident, nothing to free
            {"b", 200, true, false},
        };
        record("choose_eviction_victim/skips_unloaded", "oldest is not loaded",
               choose_eviction_victim(candidates, "d").value_or("<none>"), "b");
    });

    guarded("choose_eviction_victim/skips_self", [] {
        // Evicting the model we are trying to load would loop forever.
        const std::vector<EvictionCandidate> candidates{
            {"a", 100, true, false},
        };
        record("choose_eviction_victim/skips_self", "only candidate is the loader",
               choose_eviction_victim(candidates, "a").has_value() ? "chose one" : "none", "none");
    });

    guarded("choose_eviction_victim/none_available", [] {
        const std::vector<EvictionCandidate> candidates{
            {"a", 100, true, true},
            {"b", 200, true, true},
        };
        record("choose_eviction_victim/all_busy", "every resident model is busy",
               choose_eviction_victim(candidates, "c").has_value() ? "chose one" : "none", "none");
        record("choose_eviction_victim/empty", "no candidates",
               choose_eviction_victim({}, "c").has_value() ? "chose one" : "none", "none");
    });

    guarded("concurrency_exceeds_residency", [] {
        record("concurrency/one_worker", "workers=1, max_loaded=1",
               concurrency_exceeds_residency(1, 1) ? "conflict" : "ok", "ok");
        record("concurrency/unbounded_residency", "workers=4, max_loaded=0",
               concurrency_exceeds_residency(4, 0) ? "conflict" : "ok", "ok");
        record("concurrency/matched", "workers=2, max_loaded=2",
               concurrency_exceeds_residency(2, 2) ? "conflict" : "ok", "ok");
        record("concurrency/room_to_spare", "workers=2, max_loaded=4",
               concurrency_exceeds_residency(2, 4) ? "conflict" : "ok", "ok");
        // The configuration that turns concurrency into random 503s.
        record("concurrency/conflict", "workers=2, max_loaded=1",
               concurrency_exceeds_residency(2, 1) ? "conflict" : "ok", "conflict");
    });

    guarded("concurrency_conflict_message", [] {
        const auto message = concurrency_conflict_message(2, 1);
        record("concurrency_conflict_message/names_both", "workers=2, max_loaded=1",
               message.find("api.workers is 2") != std::string::npos &&
                       message.find("max_loaded_models is 1") != std::string::npos
                   ? "names both"
                   : message,
               "names both");
        record("concurrency_conflict_message/suggests_fix", "actionable",
               message.find("Raise max_loaded_models to at least 2") != std::string::npos ? "suggests" : message,
               "suggests");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_MODEL_POOL
