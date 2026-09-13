#include "api_config.h"

#include "model_pool.h"

#include <stdexcept>

namespace audiocpp::api {
namespace {

using engine::io::json::Value;

int64_t optional_i64_at(const Value & object, const char * key, int64_t fallback) {
    return engine::io::json::optional_i64(object, key, fallback);
}

}  // namespace

std::filesystem::path default_data_dir(const std::filesystem::path & config_path) {
    auto base = config_path.parent_path();
    if (base.empty()) {
        base = std::filesystem::current_path();
    }
    return base / ".audiocpp-api";
}

ApiConfig apply_api_overrides(ApiConfig defaults, const Value & document) {
    const auto * api = document.find("api");
    if (api == nullptr || !api->is_object()) {
        return defaults;
    }
    ApiConfig config = std::move(defaults);
    config.workers = static_cast<int>(optional_i64_at(*api, "workers", config.workers));
    config.max_queued = static_cast<size_t>(
        optional_i64_at(*api, "max_queued", static_cast<int64_t>(config.max_queued)));
    config.progress_interval_ms =
        static_cast<int>(optional_i64_at(*api, "progress_interval_ms", config.progress_interval_ms));
    config.gc_interval_minutes =
        static_cast<int>(optional_i64_at(*api, "gc_interval_minutes", config.gc_interval_minutes));
    config.shutdown_grace_ms =
        static_cast<int>(optional_i64_at(*api, "shutdown_grace_ms", config.shutdown_grace_ms));
    config.retention_days = static_cast<int>(optional_i64_at(*api, "retention_days", config.retention_days));
    config.max_jobs_retained = optional_i64_at(*api, "max_jobs_retained", config.max_jobs_retained);
    config.max_bytes_retained = optional_i64_at(*api, "max_bytes_retained", config.max_bytes_retained);
    config.requeue_max_age_hours =
        static_cast<int>(optional_i64_at(*api, "requeue_max_age_hours", config.requeue_max_age_hours));
    config.requeue_on_start = engine::io::json::optional_bool(*api, "requeue_on_start", config.requeue_on_start);
    config.unload_after_cancel =
        engine::io::json::optional_bool(*api, "unload_after_cancel", config.unload_after_cancel);
    config.unload_after_error =
        engine::io::json::optional_bool(*api, "unload_after_error", config.unload_after_error);
    config.unload_after_job =
        engine::io::json::optional_bool(*api, "unload_after_job", config.unload_after_job);
    config.oom_retries = static_cast<int>(optional_i64_at(*api, "oom_retries", config.oom_retries));
    config.max_event_subscribers = static_cast<size_t>(optional_i64_at(
        *api, "max_event_subscribers", static_cast<int64_t>(config.max_event_subscribers)));

    const auto data_dir = engine::io::json::optional_string(*api, "data_dir", "");
    if (!data_dir.empty()) {
        config.data_dir = std::filesystem::path(data_dir);
    }
    return config;
}

void validate_api_config(const ApiConfig & config) {
    if (config.workers < 1) {
        throw std::runtime_error("api.workers must be at least 1");
    }
    if (config.progress_interval_ms < 0) {
        throw std::runtime_error("api.progress_interval_ms must not be negative");
    }
    // Refusing at startup is the whole point: with more workers than resident
    // model slots, a worker has to evict a model another worker is running, which
    // fails that job instead of waiting. A clear error now beats intermittent
    // 503s under load.
    if (concurrency_exceeds_residency(config.workers, config.base.max_loaded_models)) {
        throw std::runtime_error(
            concurrency_conflict_message(config.workers, config.base.max_loaded_models));
    }
    if (config.data_dir.empty()) {
        throw std::runtime_error("api.data_dir resolved to an empty path");
    }
}

ApiConfig load_api_config(const std::filesystem::path & config_path) {
    ApiConfig config;
    config.base = minitts::server::load_server_config(config_path);
    config.data_dir = default_data_dir(config_path);
    config = apply_api_overrides(std::move(config), engine::io::json::parse_jsonc_file(config_path));
    validate_api_config(config);
    return config;
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_API_CONFIG
// Direct-execution self-test over the pure layering and validation logic; the
// file-reading wrapper is covered by starting the server.
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

void expect_throws(const char * unit, const std::string & in, const std::function<void()> & body) {
    bool threw = false;
    try {
        body();
    } catch (const std::exception &) {
        threw = true;
    }
    record(unit, in, threw ? "throws" : "returns", "throws");
}

}  // namespace

int main() {
    using namespace audiocpp::api;

    guarded("apply_api_overrides/defaults", [] {
        ApiConfig defaults;
        defaults.data_dir = "C:/store";
        const auto config = apply_api_overrides(defaults, engine::io::json::parse("{}"));
        record("apply/no_api_block", "{}", std::to_string(config.workers), "1");
        record("apply/default_retention", "{}", std::to_string(config.retention_days), "30");
        record("apply/default_max_queued", "{}", std::to_string(config.max_queued), "256");
        record("apply/default_unload_after_cancel", "{}",
               config.unload_after_cancel ? "true" : "false", "true");
        record("apply/default_unload_after_error", "{}",
               config.unload_after_error ? "true" : "false", "true");
        record("apply/default_unload_after_job", "{}",
               config.unload_after_job ? "true" : "false", "false");
        record("apply/default_oom_retries", "{}", std::to_string(config.oom_retries), "2");
    });

    guarded("apply_api_overrides/each_field", [] {
        ApiConfig defaults;
        defaults.data_dir = "C:/store";
        const auto document = engine::io::json::parse(R"({"api":{
            "workers": 3,
            "max_queued": 12,
            "progress_interval_ms": 100,
            "gc_interval_minutes": 5,
            "shutdown_grace_ms": 2000,
            "retention_days": 7,
            "max_jobs_retained": 42,
            "max_bytes_retained": 1024,
            "requeue_max_age_hours": 6,
            "requeue_on_start": false,
            "unload_after_cancel": false,
            "unload_after_error": false,
            "unload_after_job": true,
            "oom_retries": 1,
            "max_event_subscribers": 8,
            "data_dir": "C:/elsewhere"
        }})");
        const auto config = apply_api_overrides(defaults, document);
        record("apply/workers", "3", std::to_string(config.workers), "3");
        record("apply/max_queued", "12", std::to_string(config.max_queued), "12");
        record("apply/progress_interval_ms", "100", std::to_string(config.progress_interval_ms), "100");
        record("apply/gc_interval_minutes", "5", std::to_string(config.gc_interval_minutes), "5");
        record("apply/shutdown_grace_ms", "2000", std::to_string(config.shutdown_grace_ms), "2000");
        record("apply/retention_days", "7", std::to_string(config.retention_days), "7");
        record("apply/max_jobs_retained", "42", std::to_string(config.max_jobs_retained), "42");
        record("apply/max_bytes_retained", "1024", std::to_string(config.max_bytes_retained), "1024");
        record("apply/requeue_max_age_hours", "6", std::to_string(config.requeue_max_age_hours), "6");
        record("apply/requeue_on_start", "false", config.requeue_on_start ? "true" : "false", "false");
        record("apply/unload_after_cancel", "false", config.unload_after_cancel ? "true" : "false", "false");
        record("apply/unload_after_error", "false", config.unload_after_error ? "true" : "false", "false");
        record("apply/unload_after_job", "true", config.unload_after_job ? "true" : "false", "true");
        record("apply/oom_retries", "1", std::to_string(config.oom_retries), "1");
        record("apply/max_event_subscribers", "8", std::to_string(config.max_event_subscribers), "8");
        record("apply/data_dir", "C:/elsewhere", config.data_dir.string(), "C:/elsewhere");
    });

    guarded("apply_api_overrides/partial", [] {
        // An "api" block that names one field must not reset the others.
        ApiConfig defaults;
        defaults.data_dir = "C:/store";
        const auto config =
            apply_api_overrides(defaults, engine::io::json::parse(R"({"api":{"workers":2}})"));
        record("apply/partial_sets", "workers only", std::to_string(config.workers), "2");
        record("apply/partial_keeps_rest", "retention untouched", std::to_string(config.retention_days), "30");
        record("apply/partial_keeps_data_dir", "data_dir untouched", config.data_dir.string(), "C:/store");
    });

    guarded("apply_api_overrides/ignores_other_keys", [] {
        // A config written for audiocpp_server has no "api" block at all and must
        // still load.
        ApiConfig defaults;
        defaults.data_dir = "C:/store";
        const auto document = engine::io::json::parse(R"({"host":"127.0.0.1","port":8770,"models":[]})");
        const auto config = apply_api_overrides(defaults, document);
        record("apply/old_config_shape", "no api block", std::to_string(config.workers), "1");
    });

    guarded("validate_api_config/accepts", [] {
        ApiConfig config;
        config.data_dir = "C:/store";
        config.workers = 1;
        config.base.max_loaded_models = 0;
        bool ok = true;
        try {
            validate_api_config(config);
        } catch (const std::exception &) {
            ok = false;
        }
        record("validate/default", "workers=1, unbounded residency", ok ? "accepted" : "rejected", "accepted");

        config.workers = 4;
        config.base.max_loaded_models = 4;
        ok = true;
        try {
            validate_api_config(config);
        } catch (const std::exception &) {
            ok = false;
        }
        record("validate/matched", "workers=4, max_loaded=4", ok ? "accepted" : "rejected", "accepted");
    });

    guarded("validate_api_config/rejects", [] {
        ApiConfig base;
        base.data_dir = "C:/store";

        auto too_few_workers = base;
        too_few_workers.workers = 0;
        expect_throws("validate/workers_zero", "workers=0", [&] { validate_api_config(too_few_workers); });

        auto conflict = base;
        conflict.workers = 2;
        conflict.base.max_loaded_models = 1;
        expect_throws("validate/concurrency_conflict", "workers=2, max_loaded=1",
                      [&] { validate_api_config(conflict); });

        auto no_dir = base;
        no_dir.data_dir.clear();
        expect_throws("validate/empty_data_dir", "data_dir=\"\"", [&] { validate_api_config(no_dir); });

        auto negative_progress = base;
        negative_progress.progress_interval_ms = -1;
        expect_throws("validate/negative_progress", "progress_interval_ms=-1",
                      [&] { validate_api_config(negative_progress); });
    });

    guarded("validate_api_config/conflict_explains", [] {
        ApiConfig conflict;
        conflict.data_dir = "C:/store";
        conflict.workers = 2;
        conflict.base.max_loaded_models = 1;
        std::string message;
        try {
            validate_api_config(conflict);
        } catch (const std::exception & ex) {
            message = ex.what();
        }
        record("validate/conflict_message", "workers=2, max_loaded=1",
               message.find("Raise max_loaded_models") != std::string::npos ? "actionable" : message,
               "actionable");
    });

    guarded("default_data_dir", [] {
        record("default_data_dir/beside_config", "C:/models/api.json",
               default_data_dir("C:/models/api.json").string(), "C:/models\\.audiocpp-api");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_API_CONFIG
