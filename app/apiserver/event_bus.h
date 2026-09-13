#pragma once

// One concern: delivering job events to whoever is watching, and remembering
// enough of them that a client which reconnects does not miss the middle.
//
// Each job gets a bounded ring of recent events plus a sequence number. A
// subscriber names the sequence it last saw; anything still in the ring is
// replayed, and if it fell too far behind it is told so rather than silently
// handed a gap.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace audiocpp::api {

struct JobEvent {
    int64_t seq = 0;
    int64_t ts_ms = 0;
    std::string job_id;
    std::string type;    // "status" | "progress" | "stage" | "log" | "output" | "done"
    std::string json;    // the serialised payload, ready to write after "data: "
};

// What a waiting subscriber got back.
struct EventBatch {
    std::vector<JobEvent> events;
    bool closed = false;      // the job reached a terminal state; the stream ends
    int64_t dropped = 0;      // events aged out of the ring before we were asked
};

class EventBus {
public:
    explicit EventBus(size_t ring_capacity = 512, size_t max_subscribers = 64)
        : ring_capacity_(ring_capacity), max_subscribers_(max_subscribers) {}

    // Assigns the sequence number and wakes every waiter on that job.
    int64_t publish(const std::string & job_id, const std::string & type, std::string json, int64_t ts_ms);

    // Marks a job finished: waiters get whatever is left plus closed=true, and
    // future subscribers are told immediately rather than hanging.
    void close(const std::string & job_id);

    // Blocks up to `timeout_ms` for events after `after_seq`. Returns an empty,
    // open batch on timeout, which the caller turns into an SSE heartbeat.
    EventBatch wait(const std::string & job_id, int64_t after_seq, int timeout_ms);

    // Everything still in the ring after `after_seq`, without blocking.
    EventBatch replay(const std::string & job_id, int64_t after_seq);

    bool try_subscribe();
    void unsubscribe();
    size_t subscribers() const;

    // Frees a job's ring once nothing is watching it.
    void forget(const std::string & job_id);
    void shutdown();

private:
    struct Stream {
        std::deque<JobEvent> ring;
        int64_t next_seq = 1;
        int64_t first_seq = 1;   // lowest sequence still in the ring
        bool closed = false;
    };

    Stream & stream_locked(const std::string & job_id);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, Stream> streams_;
    size_t ring_capacity_;
    size_t max_subscribers_;
    size_t subscribers_ = 0;
    bool shutdown_ = false;
};

// Builds the SSE frame for one event. Split out because getting the framing
// wrong (a payload containing a newline, a missing blank line) breaks clients in
// ways that are hard to see from the server side.
std::string sse_frame(const JobEvent & event);
std::string sse_comment(const std::string & text);

}  // namespace audiocpp::api
