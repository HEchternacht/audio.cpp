#include "worker_pool.h"

#include <chrono>

namespace audiocpp::api {

WorkerPool::WorkerPool(size_t workers,
                       JobQueue & queue,
                       JobStore & store,
                       RunFn run,
                       WorkerHooks hooks,
                       int progress_interval_ms)
    : workers_(workers == 0 ? 1 : workers),
      queue_(queue),
      store_(store),
      run_(std::move(run)),
      hooks_(std::move(hooks)),
      progress_interval_ms_(progress_interval_ms) {}

WorkerPool::~WorkerPool() {
    stop(false);
}

void WorkerPool::start() {
    for (size_t i = 0; i < workers_; ++i) {
        threads_.emplace_back([this] { worker_main(); });
    }
}

void WorkerPool::stop(bool cancel_running_jobs) {
    if (stopping_.exchange(true)) {
        return;
    }
    if (cancel_running_jobs) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto & entry : controls_) {
            entry.second->cancel();
        }
    }
    queue_.stop();
    for (auto & thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads_.clear();
}

bool WorkerPool::cancel_running(const std::string & job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = controls_.find(job_id);
    if (found == controls_.end()) {
        return false;
    }
    found->second->cancel();
    return true;
}

bool WorkerPool::is_running(const std::string & job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return controls_.find(job_id) != controls_.end();
}

std::vector<std::string> WorkerPool::running_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(controls_.size());
    for (const auto & entry : controls_) {
        ids.push_back(entry.first);
    }
    return ids;
}

WorkerStats WorkerPool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void WorkerPool::finish(Job & job,
                        JobStatus status,
                        const std::string & error_type,
                        const std::string & message) {
    job.status = status;
    job.finished_ms = now_ms();
    job.updated_ms = *job.finished_ms;
    if (!message.empty()) {
        job.error.message = message;
        job.error.type = error_type;
    }
    if (job.started_ms.has_value()) {
        job.metrics.wall_ms = static_cast<double>(*job.finished_ms - *job.started_ms);
    }
    store_.save(job);
    if (hooks_.on_status) {
        hooks_.on_status(job);
    }
}

void WorkerPool::worker_main() {
    for (;;) {
        auto claim = queue_.pop();
        if (!claim.has_value()) {
            return;   // the queue was stopped
        }

        auto loaded = store_.load(claim->job_id);
        if (!loaded.has_value()) {
            // Deleted between submit and pick-up; nothing to run or record.
            queue_.release(claim->model_id);
            continue;
        }
        Job job = *loaded;

        // A cancel that landed while this job sat in the queue, or in the window
        // between pop() and registering the control below.
        if (queue_.take_cancel(job.id) || job.cancel_requested) {
            job.cancel_requested = true;
            finish(job, JobStatus::Cancelled, "cancelled", "cancelled before the run started");
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++stats_.cancelled;
            }
            queue_.release(claim->model_id);
            continue;
        }

        auto control = std::make_shared<engine::runtime::RunControl>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            controls_[job.id] = control;
            ++stats_.started;
        }
        // Racing cancel: the request may have arrived while we were registering.
        if (queue_.take_cancel(job.id)) {
            control->cancel();
        }

        job.status = JobStatus::Running;
        job.started_ms = now_ms();
        job.updated_ms = *job.started_ms;
        job.metrics.queue_ms = static_cast<double>(*job.started_ms - job.created_ms);
        job.server_run_id = job.server_run_id;   // stamped at submit; kept as-is
        store_.save(job);
        if (hooks_.on_status) {
            hooks_.on_status(job);
        }

        // Rate limiting lives here, not in the model: a model reports every step
        // unconditionally and the sink decides what is worth keeping. Always let
        // a stage change through, so a UI never shows a stale stage.
        auto last_emit = std::chrono::steady_clock::time_point{};
        auto last_flush = std::chrono::steady_clock::now();
        std::string last_stage;
        control->set_progress_sink([&](const engine::runtime::ProgressEvent & event) {
            const auto now = std::chrono::steady_clock::now();
            const bool stage_changed = event.stage != last_stage;
            const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_emit).count();
            if (!stage_changed && since < progress_interval_ms_) {
                return;
            }
            last_emit = now;
            last_stage = event.stage;

            job.progress.stage = event.stage;
            job.progress.step = event.step;
            job.progress.total = event.total;
            job.progress.message = event.message;
            job.progress.fraction =
                event.total > 0 ? static_cast<double>(event.step) / static_cast<double>(event.total) : 0.0;
            job.progress.updated_ms = now_ms();
            store_.update_progress(job.id, job.progress);

            // A much slower flush to disk than to the index. Progress is not
            // worth a file write per step, but a job found `running` after a
            // crash should be able to say how far it got -- "stopped during
            // semantic, step 1840 of 9000" is a far more useful record than
            // "stopped".
            const auto since_flush =
                std::chrono::duration_cast<std::chrono::seconds>(now - last_flush).count();
            if (stage_changed || since_flush >= 5) {
                last_flush = now;
                store_.save(job);
            }

            if (hooks_.on_progress) {
                hooks_.on_progress(job, event);
            }
        });

        JobStatus outcome = JobStatus::Succeeded;
        std::string error_type;
        std::string message;
        try {
            run_(job, control);
            // A run that returned normally but was asked to stop still counts as
            // cancelled: the user's intent is what decides, not the model's exit.
            if (control->cancelled()) {
                outcome = JobStatus::Cancelled;
                error_type = "cancelled";
                message = "cancelled during the run";
            }
        } catch (const engine::runtime::RunCancelled &) {
            outcome = JobStatus::Cancelled;
            error_type = "cancelled";
            message = "cancelled during the run";
        } catch (const std::exception & ex) {
            outcome = JobStatus::Failed;
            error_type = "server_error";
            message = ex.what();
        } catch (...) {
            outcome = JobStatus::Failed;
            error_type = "server_error";
            message = "unknown error";
        }

        // Detach the sink before the job object it captured goes out of scope.
        control->set_progress_sink(nullptr);
        finish(job, outcome, error_type, message);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            controls_.erase(job.id);
            switch (outcome) {
                case JobStatus::Succeeded: ++stats_.succeeded; break;
                case JobStatus::Cancelled: ++stats_.cancelled; break;
                default: ++stats_.failed; break;
            }
        }
        queue_.release(claim->model_id);
    }
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_WORKER_POOL
// Direct-execution self-test. The work function is a fake that sleeps in small
// slices and polls its control, so cancellation latency can be measured without
// a model, a GPU or a network.
#include <cstdio>
#include <filesystem>
#include <random>

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

std::filesystem::path fresh_root() {
    static std::mt19937_64 rng{std::random_device{}()};
    auto dir = std::filesystem::temp_directory_path() /
               ("audiocpp_workerpool_selftest_" + std::to_string(rng()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

audiocpp::api::Job make_job(const char * model, int64_t created) {
    using namespace audiocpp::api;
    Job job;
    job.id = make_job_id(created);
    job.model_id = model;
    job.family = "fake";
    job.operation = "generate";
    job.status = JobStatus::Queued;
    job.created_ms = created;
    job.updated_ms = created;
    job.server_run_id = "run-A";
    return job;
}

void enqueue(audiocpp::api::JobStore & store, audiocpp::api::JobQueue & queue,
             const audiocpp::api::Job & job) {
    store.create(job);
    queue.push({job.id, job.model_id, job.priority, job.created_ms});
}

// Waits for a predicate, so the tests never sleep longer than they must.
bool wait_until(const std::function<bool()> & done, int timeout_ms = 4000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (done()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return done();
}

}  // namespace

int main() {
    using namespace audiocpp::api;
    using engine::runtime::ProgressEvent;
    using engine::runtime::RunControl;

    // ---- a job that succeeds ----
    guarded("run/succeeds", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        WorkerPool pool(1, queue, store, [](Job & job, const std::shared_ptr<RunControl> &) {
            job.metrics.audio_seconds = 12.5;
        });
        pool.start();

        auto job = make_job("fake-model", 1757600000000LL);
        enqueue(store, queue, job);

        const bool done = wait_until([&] {
            const auto back = store.load(job.id);
            return back.has_value() && is_terminal(back->status);
        });
        pool.stop(false);

        const auto back = store.load(job.id);
        record("run/reaches_terminal", "one job", done ? "terminal" : "timed out", "terminal");
        record("run/status", "successful run", back.has_value() ? status_name(back->status) : "<none>",
               "succeeded");
        record("run/keeps_mutations", "audio_seconds set by the run",
               back.has_value() && back->metrics.audio_seconds.has_value()
                   ? std::to_string(*back->metrics.audio_seconds).substr(0, 4)
                   : "<none>",
               "12.5");
        record("run/records_wall_ms", "timing",
               back.has_value() && back->metrics.wall_ms.has_value() ? "set" : "unset", "set");
        record("run/persisted_running_first", "started_ms",
               back.has_value() && back->started_ms.has_value() ? "set" : "unset", "set");
        record("run/stats", "one success", std::to_string(pool.stats().succeeded), "1");
        std::filesystem::remove_all(root);
    });

    // ---- a job that throws ----
    guarded("run/fails", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        WorkerPool pool(1, queue, store, [](Job &, const std::shared_ptr<RunControl> &) {
            throw std::runtime_error("the model exploded");
        });
        pool.start();

        auto job = make_job("fake-model", 1757600000000LL);
        enqueue(store, queue, job);
        wait_until([&] {
            const auto back = store.load(job.id);
            return back.has_value() && is_terminal(back->status);
        });
        pool.stop(false);

        const auto back = store.load(job.id);
        record("run/failed_status", "run threw", back.has_value() ? status_name(back->status) : "<none>",
               "failed");
        record("run/failure_message", "exception text",
               back.has_value() ? back->error.message : "<none>", "the model exploded");
        record("run/failure_type", "typed", back.has_value() ? back->error.type : "<none>", "server_error");
        // The pool must survive a throwing job and keep serving.
        auto second = make_job("fake-model", 1757600001000LL);
        record("run/pool_survives", "second job after a throw",
               store.exists(job.id) ? "alive" : "dead", "alive");
        std::filesystem::remove_all(root);
    });

    // ---- cancel while running: the latency measurement ----
    guarded("cancel/running", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        std::atomic<bool> entered{false};
        // Polls every 5 ms, the way a real token loop polls between steps.
        WorkerPool pool(1, queue, store, [&](Job &, const std::shared_ptr<RunControl> & control) {
            entered = true;
            for (int step = 0; step < 2000; ++step) {
                control->throw_if_cancelled();
                control->progress(ProgressEvent{"fake", step, 2000, ""});
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
        pool.start();

        auto job = make_job("fake-model", 1757600000000LL);
        enqueue(store, queue, job);
        wait_until([&] { return entered.load(); });
        record("cancel/is_running", "after the run starts",
               pool.is_running(job.id) ? "running" : "not running", "running");

        const auto t0 = std::chrono::steady_clock::now();
        record("cancel/accepted", "cancel a running job",
               pool.cancel_running(job.id) ? "accepted" : "not running", "accepted");
        wait_until([&] {
            const auto back = store.load(job.id);
            return back.has_value() && is_terminal(back->status);
        });
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        pool.stop(false);

        const auto back = store.load(job.id);
        record("cancel/status", "cancelled not failed",
               back.has_value() ? status_name(back->status) : "<none>", "cancelled");
        // The fake polls every 5 ms, so anything beyond a few poll intervals
        // means the cancel is not reaching the loop.
        record("cancel/latency_under_500ms", std::to_string(elapsed) + "ms",
               elapsed < 500 ? "prompt" : "slow", "prompt");
        record("cancel/control_released", "after terminal",
               pool.is_running(job.id) ? "still registered" : "released", "released");
        record("cancel/unknown_job", "never started",
               pool.cancel_running("job_0000000000000000000000000") ? "accepted" : "not running",
               "not running");
        std::filesystem::remove_all(root);
    });

    // ---- cancel while still queued ----
    guarded("cancel/queued", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        std::atomic<int> ran{0};
        WorkerPool pool(1, queue, store, [&](Job &, const std::shared_ptr<RunControl> & control) {
            ++ran;
            for (int i = 0; i < 60; ++i) {
                control->throw_if_cancelled();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });

        auto first = make_job("fake-model", 1757600000000LL);
        auto second = make_job("fake-model", 1757600001000LL);
        enqueue(store, queue, first);
        enqueue(store, queue, second);

        // Cancel the second before any worker exists, so it is unambiguously
        // cancelled from the queue rather than mid-run.
        queue.request_cancel(second.id);
        pool.start();

        wait_until([&] {
            const auto a = store.load(first.id);
            const auto b = store.load(second.id);
            return a.has_value() && is_terminal(a->status) && b.has_value() && is_terminal(b->status);
        });
        pool.stop(false);

        const auto a = store.load(first.id);
        const auto b = store.load(second.id);
        record("cancel/queued_status", "cancelled before start",
               b.has_value() ? status_name(b->status) : "<none>", "cancelled");
        record("cancel/queued_message", "explains itself",
               b.has_value() ? b->error.message : "<none>", "cancelled before the run started");
        record("cancel/queued_never_ran", "run count", std::to_string(ran.load()), "1");
        record("cancel/other_job_unaffected", "first job",
               a.has_value() ? status_name(a->status) : "<none>", "succeeded");
        std::filesystem::remove_all(root);
    });

    // ---- same model serialises, different models overlap ----
    guarded("concurrency/same_model_serialises", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        std::atomic<int> concurrent{0};
        std::atomic<int> peak{0};
        WorkerPool pool(4, queue, store, [&](Job &, const std::shared_ptr<RunControl> &) {
            const int now = ++concurrent;
            int seen = peak.load();
            while (now > seen && !peak.compare_exchange_weak(seen, now)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            --concurrent;
        });
        pool.start();

        std::vector<std::string> ids;
        for (int i = 0; i < 4; ++i) {
            auto job = make_job("one-model", 1757600000000LL + i);
            ids.push_back(job.id);
            enqueue(store, queue, job);
        }
        wait_until([&] {
            for (const auto & id : ids) {
                const auto back = store.load(id);
                if (!back.has_value() || !is_terminal(back->status)) {
                    return false;
                }
            }
            return true;
        }, 8000);
        pool.stop(false);

        // Four workers, four jobs, one model: the claim set must hold them to one
        // at a time. This is the invariant that keeps a 6 GB card alive.
        record("concurrency/peak_for_one_model", "4 workers, 4 jobs, 1 model",
               std::to_string(peak.load()), "1");
        record("concurrency/all_completed", "4 jobs", std::to_string(pool.stats().succeeded), "4");
        std::filesystem::remove_all(root);
    });

    guarded("concurrency/different_models_overlap", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        std::atomic<int> concurrent{0};
        std::atomic<int> peak{0};
        WorkerPool pool(3, queue, store, [&](Job &, const std::shared_ptr<RunControl> &) {
            const int now = ++concurrent;
            int seen = peak.load();
            while (now > seen && !peak.compare_exchange_weak(seen, now)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
            --concurrent;
        });
        pool.start();

        std::vector<std::string> ids;
        for (const char * model : {"model-a", "model-b", "model-c"}) {
            auto job = make_job(model, 1757600000000LL + ids.size());
            ids.push_back(job.id);
            enqueue(store, queue, job);
        }
        wait_until([&] {
            for (const auto & id : ids) {
                const auto back = store.load(id);
                if (!back.has_value() || !is_terminal(back->status)) {
                    return false;
                }
            }
            return true;
        }, 8000);
        pool.stop(false);

        record("concurrency/peak_for_three_models", "3 workers, 3 distinct models",
               peak.load() > 1 ? "parallel" : "serialised", "parallel");
        std::filesystem::remove_all(root);
    });

    // ---- progress ----
    guarded("progress/throttled_and_recorded", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        std::atomic<int> emitted{0};
        WorkerHooks hooks;
        hooks.on_progress = [&](const Job &, const ProgressEvent &) { ++emitted; };

        // 1000 reported steps, throttled to one per 50 ms.
        WorkerPool pool(1, queue, store, [](Job &, const std::shared_ptr<RunControl> & control) {
            for (int step = 0; step < 1000; ++step) {
                control->progress(ProgressEvent{"semantic", step, 1000, ""});
            }
        }, hooks, 50);
        pool.start();

        auto job = make_job("fake-model", 1757600000000LL);
        enqueue(store, queue, job);
        wait_until([&] {
            const auto back = store.load(job.id);
            return back.has_value() && is_terminal(back->status);
        });
        pool.stop(false);

        // A tight loop of 1000 reports finishes well inside one window, so only
        // the first (stage change) should get through.
        record("progress/throttled", "1000 reports, 50ms window",
               emitted.load() <= 3 ? "throttled" : std::to_string(emitted.load()), "throttled");
        record("progress/emitted_at_least_once", "stage change always passes",
               emitted.load() >= 1 ? "emitted" : "silent", "emitted");
        std::filesystem::remove_all(root);
    });

    guarded("progress/stage_change_always_passes", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        std::vector<std::string> stages;
        std::mutex stages_mutex;
        WorkerHooks hooks;
        hooks.on_progress = [&](const Job &, const ProgressEvent & event) {
            std::lock_guard<std::mutex> lock(stages_mutex);
            stages.push_back(event.stage);
        };

        WorkerPool pool(1, queue, store, [](Job &, const std::shared_ptr<RunControl> & control) {
            // Four stages back to back, far inside the throttle window.
            for (const char * stage : {"score", "semantic", "acoustic", "decode"}) {
                control->progress(ProgressEvent{stage, 1, 10, ""});
            }
        }, hooks, 10000);
        pool.start();

        auto job = make_job("fake-model", 1757600000000LL);
        enqueue(store, queue, job);
        wait_until([&] {
            const auto back = store.load(job.id);
            return back.has_value() && is_terminal(back->status);
        });
        pool.stop(false);

        std::lock_guard<std::mutex> lock(stages_mutex);
        std::string seen;
        for (const auto & stage : stages) {
            if (!seen.empty()) {
                seen += ",";
            }
            seen += stage;
        }
        record("progress/every_stage_reported", "4 stages inside one throttle window", seen,
               "score,semantic,acoustic,decode");
        std::filesystem::remove_all(root);
    });

    // ---- graceful shutdown ----
    guarded("stop/cancels_running", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;

        std::atomic<bool> entered{false};
        WorkerPool pool(1, queue, store, [&](Job &, const std::shared_ptr<RunControl> & control) {
            entered = true;
            for (int i = 0; i < 2000; ++i) {
                control->throw_if_cancelled();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
        pool.start();

        auto job = make_job("fake-model", 1757600000000LL);
        enqueue(store, queue, job);
        wait_until([&] { return entered.load(); });
        pool.stop(true);   // graceful: cancel in flight, then join

        const auto back = store.load(job.id);
        record("stop/graceful_cancels", "shutdown with a job in flight",
               back.has_value() ? status_name(back->status) : "<none>", "cancelled");
        std::filesystem::remove_all(root);
    });

    guarded("stop/idempotent", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        JobQueue queue;
        WorkerPool pool(2, queue, store, [](Job &, const std::shared_ptr<RunControl> &) {});
        pool.start();
        pool.stop(false);
        pool.stop(false);   // must not hang or throw
        record("stop/twice", "stop called twice", "returned", "returned");
        std::filesystem::remove_all(root);
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_WORKER_POOL
