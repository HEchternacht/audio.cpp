#pragma once

// One concern: running queued jobs on N threads and recording what happened.
//
// The pool knows nothing about models or HTTP. The work itself arrives as an
// injected callable, which is what lets this file be tested with a fake that
// sleeps and watches its cancel flag instead of loading a 3B model.
//
// The lifecycle it guarantees, in order: claim, check for a cancel that landed
// while queued, mark running (durably), run, mark terminal (durably), release
// the claim. The first terminal outcome wins, so a late failure can never
// overwrite a cancel the user already saw.

#include "engine/framework/runtime/run_control.h"
#include "job.h"
#include "job_queue.h"
#include "job_store.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace audiocpp::api {

// Runs one job. Throws engine::runtime::RunCancelled if it observes the control
// being cancelled; any other exception fails the job with that message.
// Implementations mutate `job` to attach outputs and metrics.
using RunFn = std::function<void(Job & job, const std::shared_ptr<engine::runtime::RunControl> & control)>;

// Notified as a job moves through its lifecycle, so events reach subscribers
// without the pool depending on the event bus.
struct WorkerHooks {
    std::function<void(const Job &)> on_status;
    std::function<void(const Job &, const engine::runtime::ProgressEvent &)> on_progress;
};

struct WorkerStats {
    int64_t started = 0;
    int64_t succeeded = 0;
    int64_t failed = 0;
    int64_t cancelled = 0;
    int64_t requeued = 0;
};

class WorkerPool {
public:
    WorkerPool(size_t workers,
               JobQueue & queue,
               JobStore & store,
               RunFn run,
               WorkerHooks hooks = {},
               int progress_interval_ms = 250);
    ~WorkerPool();

    WorkerPool(const WorkerPool &) = delete;
    WorkerPool & operator=(const WorkerPool &) = delete;

    void start();
    // Asks every worker to finish its current job and exit. When `cancel_running`
    // is set, in-flight jobs are cancelled first -- that is the graceful shutdown
    // path, where a running job ends up `cancelled` rather than `interrupted`.
    void stop(bool cancel_running);

    // Cancels a job that is currently running. Returns false when it is not.
    bool cancel_running(const std::string & job_id);
    bool is_running(const std::string & job_id) const;
    std::vector<std::string> running_ids() const;

    WorkerStats stats() const;

private:
    void worker_main();
    void finish(Job & job, JobStatus status, const std::string & error_type, const std::string & message);

    size_t workers_;
    JobQueue & queue_;
    JobStore & store_;
    RunFn run_;
    WorkerHooks hooks_;
    int progress_interval_ms_;

    std::vector<std::thread> threads_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<engine::runtime::RunControl>> controls_;
    WorkerStats stats_;
    std::atomic<bool> stopping_{false};
};

}  // namespace audiocpp::api
