#pragma once

// One concern: deciding which job runs next.
//
// This is a scheduling structure and nothing more -- it does not know what a
// model is, only that jobs name one and that two jobs naming the same one must
// not run at once. That rule is the model-claim set, and it is why BusyGuard
// underneath never actually contends: a worker is never handed a job whose model
// another worker is already holding.
//
// Ordering is (priority desc, created_ms asc). Priority exists because the shape
// of the API should not have to change to gain it later; nothing sets it above 0
// today.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace audiocpp::api {

struct QueueEntry {
    std::string job_id;
    std::string model_id;
    int priority = 0;
    int64_t created_ms = 0;
};

// What a worker was handed. `model_id` is the claim it now holds and must
// release when done.
struct Claim {
    std::string job_id;
    std::string model_id;
};

enum class PushResult {
    Accepted,
    QueueFull,
    Duplicate,
};

class JobQueue {
public:
    explicit JobQueue(size_t capacity = 0) : capacity_(capacity) {}

    PushResult push(QueueEntry entry);
    // Puts an entry back at the head of its priority band. Used when a worker
    // could not start after all, so the job keeps its place rather than going to
    // the back of a queue it already waited through.
    void push_front(QueueEntry entry);

    // Blocks until a runnable job exists, the queue is stopped, or the caller is
    // woken by a state change. Returns nullopt only on stop. A job whose model is
    // claimed is skipped, not waited on -- the next different-model job runs.
    std::optional<Claim> pop(std::vector<std::string> * skipped_for_model = nullptr);

    // Releases a model claim taken by pop().
    void release(const std::string & model_id);

    // Claim a model outside the worker path, so a manual load or unload cannot
    // pull a model out from under a running job. Returns false if already held.
    bool try_claim(const std::string & model_id);
    bool is_claimed(const std::string & model_id) const;

    // Removes a still-queued job. Returns true if it was found and removed,
    // which is what lets a cancel of a queued job be answered synchronously.
    bool cancel_queued(const std::string & job_id);
    // Records a cancel for a job that may already have been popped. A worker
    // checks this immediately after claiming, closing the race between the HTTP
    // thread and the worker.
    void request_cancel(const std::string & job_id);
    bool take_cancel(const std::string & job_id);

    size_t size() const;
    bool contains(const std::string & job_id) const;
    // 0-based position in the queue, or -1 when not queued.
    int position(const std::string & job_id) const;
    std::vector<QueueEntry> snapshot() const;

    void stop();
    bool stopped() const;

private:
    // Ordering predicate, exposed for the self-test.
    static bool orders_before(const QueueEntry & a, const QueueEntry & b);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<QueueEntry> entries_;
    std::unordered_set<std::string> claimed_models_;
    std::unordered_set<std::string> cancel_requested_;
    std::unordered_map<std::string, bool> queued_ids_;
    size_t capacity_ = 0;
    bool stopped_ = false;

    friend bool queue_orders_before(const QueueEntry & a, const QueueEntry & b);
};

bool queue_orders_before(const QueueEntry & a, const QueueEntry & b);

}  // namespace audiocpp::api
