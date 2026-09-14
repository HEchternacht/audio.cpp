#pragma once

// One concern: wiring. The route table, and the glue that turns an HTTP request
// into a queued job and a finished job back into a response.
//
// Every piece it coordinates -- the store, the queue, the pool, the workers, the
// event bus -- is testable on its own; this file is the part that only makes
// sense assembled, so it deliberately holds as little logic as possible.

#include "api_config.h"
#include "event_bus.h"
#include "http.h"
#include "job_queue.h"
#include "job_store.h"
#include "model_pool.h"
#include "router.h"
#include "worker_pool.h"

#include <atomic>
#include <memory>
#include <string>

namespace audiocpp::api {

class ApiHandler final : public minitts::server::IHttpHandler {
public:
    ApiHandler(ApiConfig config, std::filesystem::path config_dir);
    ~ApiHandler() override;

    // Opens the store, reconciles what the last process left behind, registers
    // models and starts the workers. Separate from the constructor so a failure
    // here reports properly instead of throwing out of a member initialiser.
    void start();
    void stop();

    minitts::server::HttpResponse handle(const minitts::server::HttpRequest & request) override;

    const ApiConfig & config() const noexcept { return config_; }

private:
    void register_routes();
    void run_job(Job & job, const std::shared_ptr<engine::runtime::RunControl> & control);
    void publish_status(const Job & job);
    void publish_progress(const Job & job, const engine::runtime::ProgressEvent & event);
    void publish_log(const std::string & job_id, const std::string & line);

    minitts::server::HttpResponse handle_health(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_capabilities(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_models(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_model_load(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_model_unload(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_submit_task(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_list_jobs(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_get_job(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_job_result(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_job_output(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_job_cancel(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_job_delete(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_job_patch(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_job_events(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_queue(const minitts::server::HttpRequest &, const PathParams &);
    minitts::server::HttpResponse handle_openapi(const minitts::server::HttpRequest &, const PathParams &);

    Job require_job(const std::string & id) const;

    ApiConfig config_;
    std::filesystem::path config_dir_;
    std::string server_run_id_;
    int64_t started_ms_ = 0;

    JobStore store_;
    JobQueue queue_;
    ModelPool pool_;
    EventBus events_;
    std::unique_ptr<WorkerPool> workers_;
    Router router_;
    std::atomic<bool> running_{false};
};

}  // namespace audiocpp::api
