#pragma once

// One concern: the API server's own settings, layered on top of the server
// config the rest of the codebase already understands.
//
// Everything model- and backend-shaped is reused verbatim from
// minitts::server::ServerConfig, so a config file written for audiocpp_server
// works here unchanged; the extra settings live under an "api" object that the
// old server ignores.

#include "config.h"

#include "engine/framework/io/json.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace audiocpp::api {

struct ApiConfig {
    minitts::server::ServerConfig base;

    // Default 1, and on a 6 GB card running yue2 it is the only value that fits:
    // one session per model plus multi-gigabyte graph arenas leave no room for a
    // second resident generation.
    int workers = 1;
    std::filesystem::path data_dir;
    size_t max_queued = 256;
    // The largest file a client may upload as a job input, in MiB. A recording
    // to transcribe is a few tens of megabytes; anything far past that is a
    // mistake or an attempt to fill the disk.
    int max_upload_mb = 128;
    int progress_interval_ms = 250;
    int gc_interval_minutes = 60;
    int shutdown_grace_ms = 10000;

    // Retention. Each is disabled by 0, the convention used elsewhere here.
    int retention_days = 30;
    int64_t max_jobs_retained = 5000;
    int64_t max_bytes_retained = 50LL * 1024 * 1024 * 1024;

    // A job that was queued when the server died is re-enqueued on restart, but
    // only if it is still recent: nobody wants a week-old queue replaying after a
    // reboot.
    int requeue_max_age_hours = 24;
    bool requeue_on_start = true;

    // After a cancelled run, unload the model rather than reusing a session whose
    // generation loop was unwound mid-flight. Costs one reload; buys certainty
    // that a partially advanced KV cache never contaminates the next job.
    bool unload_after_cancel = true;

    // A run that threw leaves its session holding the graph arenas it had
    // already allocated, so on a card with little headroom every later job on
    // that model fails the same way -- most visibly as a CUDA OOM reported as
    // "failed to allocate ... graph". Dropping the session is the only reliable
    // way to hand the next job a clean device.
    bool unload_after_error = true;

    // Unload after *every* job, not just the unhappy ones. Off by default,
    // because it costs a reload per job. Worth turning on where peak VRAM is
    // close to the card: a completed run leaves a few hundred MiB resident, and
    // a few of those in a row are the difference between fitting and not.
    bool unload_after_job = false;

    // How many times a job that ran out of device memory is retried with more
    // memory held back from the model's own window sizing. Each retry doubles
    // it, so two attempts quarter the budget. 0 disables the retry.
    int oom_retries = 2;

    size_t max_event_subscribers = 64;
};

// Parses the "api" block out of an already-loaded config document and applies
// the defaults above. Throws with an explanatory message when the settings
// contradict each other.
ApiConfig load_api_config(const std::filesystem::path & config_path);

// Split out so it can be tested without a file: applies the "api" object in
// `document` over `defaults`.
ApiConfig apply_api_overrides(ApiConfig defaults, const engine::io::json::Value & document);

// Throws when the configuration is self-defeating. Called after overrides.
void validate_api_config(const ApiConfig & config);

// Where the store lives when the config does not say: beside the config file, so
// a server started with `--config C:/models/api.json` keeps its history with the
// models it generated from.
std::filesystem::path default_data_dir(const std::filesystem::path & config_path);

}  // namespace audiocpp::api
