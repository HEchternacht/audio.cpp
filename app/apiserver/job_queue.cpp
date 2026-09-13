#include "job_queue.h"

#include <algorithm>

namespace audiocpp::api {

bool queue_orders_before(const QueueEntry & a, const QueueEntry & b) {
    if (a.priority != b.priority) {
        return a.priority > b.priority;   // higher priority first
    }
    if (a.created_ms != b.created_ms) {
        return a.created_ms < b.created_ms;   // then oldest first
    }
    return a.job_id < b.job_id;   // deterministic tiebreak
}

bool JobQueue::orders_before(const QueueEntry & a, const QueueEntry & b) {
    return queue_orders_before(a, b);
}

PushResult JobQueue::push(QueueEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queued_ids_.find(entry.job_id) != queued_ids_.end()) {
        return PushResult::Duplicate;
    }
    if (capacity_ > 0 && entries_.size() >= capacity_) {
        return PushResult::QueueFull;
    }
    const auto at = std::upper_bound(entries_.begin(), entries_.end(), entry, orders_before);
    entries_.insert(at, entry);
    queued_ids_[entry.job_id] = true;
    cv_.notify_one();
    return PushResult::Accepted;
}

void JobQueue::push_front(QueueEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queued_ids_.find(entry.job_id) != queued_ids_.end()) {
        return;
    }
    // Ahead of everything it would otherwise tie with, but still behind any
    // higher priority band.
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), entry, orders_before);
    entries_.insert(at, entry);
    queued_ids_[entry.job_id] = true;
    cv_.notify_one();
}

std::optional<Claim> JobQueue::pop(std::vector<std::string> * skipped_for_model) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        if (stopped_) {
            return std::nullopt;
        }
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (claimed_models_.find(it->model_id) != claimed_models_.end()) {
                if (skipped_for_model != nullptr) {
                    skipped_for_model->push_back(it->job_id);
                }
                continue;
            }
            Claim claim{it->job_id, it->model_id};
            claimed_models_.insert(it->model_id);
            queued_ids_.erase(it->job_id);
            entries_.erase(it);
            return claim;
        }
        // Everything queued is blocked behind a claim, or there is nothing at
        // all. Either way the next state change is what unblocks us.
        cv_.wait(lock);
    }
}

void JobQueue::release(const std::string & model_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        claimed_models_.erase(model_id);
    }
    // A released model may unblock several waiting workers.
    cv_.notify_all();
}

bool JobQueue::try_claim(const std::string & model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return claimed_models_.insert(model_id).second;
}

bool JobQueue::is_claimed(const std::string & model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return claimed_models_.find(model_id) != claimed_models_.end();
}

bool JobQueue::cancel_queued(const std::string & job_id) {
    bool removed = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto at = std::find_if(entries_.begin(), entries_.end(),
                                     [&](const QueueEntry & e) { return e.job_id == job_id; });
        if (at != entries_.end()) {
            entries_.erase(at);
            queued_ids_.erase(job_id);
            removed = true;
        }
    }
    if (removed) {
        cv_.notify_all();
    }
    return removed;
}

void JobQueue::request_cancel(const std::string & job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    cancel_requested_.insert(job_id);
}

bool JobQueue::take_cancel(const std::string & job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancel_requested_.erase(job_id) > 0;
}

size_t JobQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

bool JobQueue::contains(const std::string & job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queued_ids_.find(job_id) != queued_ids_.end();
}

int JobQueue::position(const std::string & job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int index = 0;
    for (const auto & entry : entries_) {
        if (entry.job_id == job_id) {
            return index;
        }
        ++index;
    }
    return -1;
}

std::vector<QueueEntry> JobQueue::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {entries_.begin(), entries_.end()};
}

void JobQueue::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    cv_.notify_all();
}

bool JobQueue::stopped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopped_;
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_JOB_QUEUE
// Direct-execution self-test. Never fires on import: this block only compiles
// when this file is built as its own target with -DAUDIOCPP_SELFTEST_JOB_QUEUE,
// so a sibling linked in as a dependency never contributes a second main().
#include <chrono>
#include <cstdio>
#include <functional>
#include <thread>

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

audiocpp::api::QueueEntry entry(const char * id, const char * model, int priority, int64_t created) {
    return audiocpp::api::QueueEntry{id, model, priority, created};
}

std::string ids_of(const std::vector<audiocpp::api::QueueEntry> & entries) {
    std::string out;
    for (const auto & e : entries) {
        if (!out.empty()) {
            out += ",";
        }
        out += e.job_id;
    }
    return out;
}

}  // namespace

int main() {
    using namespace audiocpp::api;

    // ---- ordering ----
    guarded("queue_orders_before", [] {
        record("queue_orders_before/priority", "p=5 vs p=0",
               queue_orders_before(entry("a", "m", 5, 100), entry("b", "m", 0, 1)) ? "a first" : "b first",
               "a first");
        record("queue_orders_before/age", "same priority, older first",
               queue_orders_before(entry("a", "m", 0, 100), entry("b", "m", 0, 200)) ? "a first" : "b first",
               "a first");
        record("queue_orders_before/tiebreak", "identical priority and time",
               queue_orders_before(entry("a", "m", 0, 100), entry("b", "m", 0, 100)) ? "a first" : "b first",
               "a first");
    });

    guarded("push/order", [] {
        JobQueue q;
        q.push(entry("j1", "m", 0, 100));
        q.push(entry("j2", "m", 0, 200));
        q.push(entry("j3", "m", 0, 300));
        record("push/fifo", "three at priority 0", ids_of(q.snapshot()), "j1,j2,j3");

        q.push(entry("j4", "m", 5, 400));
        record("push/priority_jumps", "j4 at priority 5", ids_of(q.snapshot()), "j4,j1,j2,j3");

        q.push(entry("j5", "m", 5, 500));
        record("push/priority_band_is_fifo", "j5 also priority 5",
               ids_of(q.snapshot()), "j4,j5,j1,j2,j3");
    });

    guarded("push_front", [] {
        JobQueue q;
        q.push(entry("j1", "m", 0, 100));
        q.push(entry("j2", "m", 0, 200));
        // A requeued job keeps its place rather than going to the back.
        q.push_front(entry("j3", "m", 0, 150));
        record("push_front/keeps_place", "requeue j3", ids_of(q.snapshot()), "j1,j3,j2");
    });

    guarded("push/duplicate", [] {
        JobQueue q;
        record("push/first", "j1", q.push(entry("j1", "m", 0, 1)) == PushResult::Accepted ? "accepted" : "rejected",
               "accepted");
        record("push/duplicate", "j1 again",
               q.push(entry("j1", "m", 0, 1)) == PushResult::Duplicate ? "duplicate" : "accepted", "duplicate");
        record("push/size", "after duplicate", std::to_string(q.size()), "1");
    });

    guarded("push/capacity", [] {
        JobQueue q(2);
        q.push(entry("j1", "m", 0, 1));
        q.push(entry("j2", "m", 0, 2));
        record("push/full", "capacity 2, third push",
               q.push(entry("j3", "m", 0, 3)) == PushResult::QueueFull ? "full" : "accepted", "full");
        record("push/capacity_size", "after rejection", std::to_string(q.size()), "2");
    });

    // ---- the model-claim rule ----
    guarded("pop/claims_model", [] {
        JobQueue q;
        q.push(entry("j1", "yue2", 0, 100));
        const auto first = q.pop();
        record("pop/returns_job", "one queued job", first.has_value() ? first->job_id : "<none>", "j1");
        record("pop/claims", "yue2 after pop", q.is_claimed("yue2") ? "claimed" : "free", "claimed");
        q.release("yue2");
        record("release", "after release", q.is_claimed("yue2") ? "claimed" : "free", "free");
    });

    guarded("pop/skips_claimed_model", [] {
        // The rule that makes concurrency safe: two jobs for one model never run
        // together, but a job for a different model is not blocked behind them.
        JobQueue q;
        q.push(entry("j1", "yue2", 0, 100));
        q.push(entry("j2", "yue2", 0, 200));
        q.push(entry("j3", "ace", 0, 300));

        const auto a = q.pop();
        record("pop/first", "head of queue", a.has_value() ? a->job_id : "<none>", "j1");
        const auto b = q.pop();
        record("pop/skips_to_other_model", "j2 blocked, j3 free",
               b.has_value() ? b->job_id : "<none>", "j3");
        record("pop/j2_still_queued", "j2", q.contains("j2") ? "queued" : "gone", "queued");

        q.release("yue2");
        const auto c = q.pop();
        record("pop/after_release", "yue2 free again", c.has_value() ? c->job_id : "<none>", "j2");
    });

    guarded("try_claim", [] {
        // A manual model load takes the same claim, so it cannot race a worker.
        JobQueue q;
        record("try_claim/free", "yue2", q.try_claim("yue2") ? "claimed" : "busy", "claimed");
        record("try_claim/held", "yue2 again", q.try_claim("yue2") ? "claimed" : "busy", "busy");
        q.release("yue2");
        record("try_claim/after_release", "yue2", q.try_claim("yue2") ? "claimed" : "busy", "claimed");
    });

    // ---- cancellation ----
    guarded("cancel_queued", [] {
        JobQueue q;
        q.push(entry("j1", "m", 0, 100));
        q.push(entry("j2", "m", 0, 200));
        q.push(entry("j3", "m", 0, 300));
        record("cancel_queued/hit", "j2", q.cancel_queued("j2") ? "removed" : "absent", "removed");
        record("cancel_queued/order", "after removing j2", ids_of(q.snapshot()), "j1,j3");
        record("cancel_queued/miss", "j2 again", q.cancel_queued("j2") ? "removed" : "absent", "absent");
        record("cancel_queued/unknown", "never queued",
               q.cancel_queued("nope") ? "removed" : "absent", "absent");
        // A cancelled job must be pushable again -- the id bookkeeping has to be
        // released along with the entry.
        record("cancel_queued/frees_id", "re-push j2",
               q.push(entry("j2", "m", 0, 200)) == PushResult::Accepted ? "accepted" : "rejected",
               "accepted");
    });

    guarded("request_cancel/take_cancel", [] {
        // Closes the race where cancel lands between pop() and the worker
        // registering its control object.
        JobQueue q;
        q.request_cancel("j1");
        record("take_cancel/set", "j1", q.take_cancel("j1") ? "cancelled" : "not cancelled", "cancelled");
        record("take_cancel/consumed", "j1 again", q.take_cancel("j1") ? "cancelled" : "not cancelled",
               "not cancelled");
        record("take_cancel/unset", "j2", q.take_cancel("j2") ? "cancelled" : "not cancelled",
               "not cancelled");
    });

    // ---- position reporting ----
    guarded("position", [] {
        JobQueue q;
        q.push(entry("j1", "m", 0, 100));
        q.push(entry("j2", "m", 0, 200));
        record("position/head", "j1", std::to_string(q.position("j1")), "0");
        record("position/second", "j2", std::to_string(q.position("j2")), "1");
        record("position/absent", "j9", std::to_string(q.position("j9")), "-1");
    });

    // ---- blocking behaviour ----
    guarded("pop/blocks_then_wakes", [] {
        JobQueue q;
        std::optional<Claim> got;
        std::thread worker([&] { got = q.pop(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        const bool still_waiting = !got.has_value();
        q.push(entry("j1", "m", 0, 100));
        worker.join();
        record("pop/waited", "empty queue", still_waiting ? "blocked" : "returned early", "blocked");
        record("pop/woke_on_push", "after push", got.has_value() ? got->job_id : "<none>", "j1");
    });

    guarded("pop/stop_unblocks", [] {
        JobQueue q;
        std::optional<Claim> got;
        bool returned = false;
        std::thread worker([&] {
            got = q.pop();
            returned = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        q.stop();
        worker.join();
        record("pop/stop_returns", "stop() while blocked", returned ? "returned" : "hung", "returned");
        record("pop/stop_is_nullopt", "stopped queue", got.has_value() ? "job" : "nullopt", "nullopt");
        record("stopped", "after stop()", q.stopped() ? "stopped" : "running", "stopped");
    });

    guarded("pop/release_wakes_blocked_worker", [] {
        // Everything queued is behind a claim; releasing it must wake the waiter.
        JobQueue q;
        q.push(entry("j1", "yue2", 0, 100));
        const auto held = q.pop();
        q.push(entry("j2", "yue2", 0, 200));

        std::optional<Claim> got;
        std::thread worker([&] { got = q.pop(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        const bool blocked = !got.has_value();
        q.release(held->model_id);
        worker.join();
        record("pop/blocked_on_claim", "only a claimed model queued",
               blocked ? "blocked" : "returned early", "blocked");
        record("pop/woke_on_release", "after release", got.has_value() ? got->job_id : "<none>", "j2");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_JOB_QUEUE
