// audiocpp_apiserver -- headless music generation and editing API.
//
// No UI, no embedded assets, no browser routes. Every request that touches a
// model becomes a durable job: submitted, queued, run, and recorded on disk so
// it survives a crash, a kill, or a restart.

#include "api_config.h"
#include "api_handler.h"

#include "../common/build_info.h"
#include "http.h"

#include "engine/framework/core/backend.h"

#include <csignal>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

volatile std::sig_atomic_t g_shutdown_requested = 0;

void request_shutdown(int) {
    g_shutdown_requested = 1;
}

bool shutdown_requested() {
    return g_shutdown_requested != 0;
}

std::optional<std::string> arg_value(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

bool has_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

void print_usage() {
    std::cout << R"(audiocpp_apiserver -- headless music generation and editing API

Usage:
  audiocpp_apiserver --config <path> [options]

Required:
  --config <path>          Server config JSON (the same shape audiocpp_server takes,
                           plus an optional "api" block)

Server:
  --host <addr>            Listen address (overrides the config)
  --port <n>               Listen port (overrides the config)
  --cors-origins <list>    Allowed origins; the UI is a separate origin now, so a
                           dev server must be listed explicitly

API:
  --workers <n>            Concurrent jobs (default 1). Two jobs for the same model
                           always serialise whatever this is set to.
  --data-dir <path>        Where jobs and their outputs are kept
                           (default: <config dir>/.audiocpp-api)

Other:
  --version                Print version and exit
  --help                   Print this help and exit

The "api" config block, with its defaults:

  "api": {
    "workers": 1,
    "data_dir": "<config dir>/.audiocpp-api",
    "max_queued": 256,
    "retention_days": 30,
    "max_jobs_retained": 5000,
    "max_bytes_retained": 53687091200,
    "requeue_on_start": true,
    "requeue_max_age_hours": 24,
    "unload_after_cancel": true,
    "unload_after_error": true,
    "unload_after_job": false,
    "oom_retries": 2,
    "progress_interval_ms": 250,
    "max_event_subscribers": 64
  }
)";
}

}  // namespace

int main(int argc, char ** argv) {
    if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) {
        print_usage();
        return 0;
    }
    if (has_arg(argc, argv, "--version")) {
        minitts::app::print_build_info(std::cout);
        return 0;
    }

    const auto config_arg = arg_value(argc, argv, "--config");
    if (!config_arg.has_value()) {
        std::cerr << "error: --config is required\n\n";
        print_usage();
        return 1;
    }

    try {
        const std::filesystem::path config_path = std::filesystem::absolute(*config_arg);
        if (!std::filesystem::exists(config_path)) {
            std::cerr << "error: config not found: " << config_path.string() << "\n";
            return 1;
        }

        auto config = audiocpp::api::load_api_config(config_path);

        if (const auto host = arg_value(argc, argv, "--host")) {
            config.base.host = *host;
        }
        if (const auto port = arg_value(argc, argv, "--port")) {
            config.base.port = std::stoi(*port);
        }
        if (const auto origins = arg_value(argc, argv, "--cors-origins")) {
            config.base.cors_origins = *origins;
        }
        if (const auto workers = arg_value(argc, argv, "--workers")) {
            config.workers = std::stoi(*workers);
        }
        if (const auto data_dir = arg_value(argc, argv, "--data-dir")) {
            config.data_dir = std::filesystem::path(*data_dir);
        }
        // Re-validate: a flag can contradict the file just as easily as the file
        // can contradict itself.
        audiocpp::api::validate_api_config(config);

        std::signal(SIGINT, request_shutdown);
        std::signal(SIGTERM, request_shutdown);

        audiocpp::api::ApiHandler handler(config, config_path.parent_path());
        handler.start();

        std::cerr << "[api] listening on http://" << config.base.host << ":" << config.base.port
                  << " with " << config.workers << " worker" << (config.workers == 1 ? "" : "s") << "\n";
        std::cerr << "[api] store: " << config.data_dir.string() << "\n";

        minitts::server::serve_http(config.base.host,
                                    config.base.port,
                                    handler,
                                    shutdown_requested,
                                    config.base.max_request_body_bytes);

        // Graceful: in-flight jobs are cancelled and recorded as such, rather
        // than being found half-written and marked interrupted on the next boot.
        std::cerr << "[api] shutting down\n";
        handler.stop();
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
