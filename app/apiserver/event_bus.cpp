#include "event_bus.h"

#include <chrono>

namespace audiocpp::api {

std::string sse_frame(const JobEvent & event) {
    std::string out;
    out += "event: " + event.type + "\n";
    out += "id: " + std::to_string(event.seq) + "\n";
    // A payload containing a newline must become several data: lines, or every
    // line after the first is read as a new field.
    size_t start = 0;
    for (;;) {
        const auto end = event.json.find('\n', start);
        if (end == std::string::npos) {
            out += "data: " + event.json.substr(start) + "\n";
            break;
        }
        out += "data: " + event.json.substr(start, end - start) + "\n";
        start = end + 1;
    }
    out += "\n";
    return out;
}

std::string sse_comment(const std::string & text) {
    return ": " + text + "\n\n";
}

EventBus::Stream & EventBus::stream_locked(const std::string & job_id) {
    return streams_[job_id];
}

int64_t EventBus::publish(const std::string & job_id,
                          const std::string & type,
                          std::string json,
                          int64_t ts_ms) {
    int64_t seq = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto & stream = stream_locked(job_id);
        JobEvent event;
        event.seq = stream.next_seq++;
        event.ts_ms = ts_ms;
        event.job_id = job_id;
        event.type = type;
        event.json = std::move(json);
        seq = event.seq;
        stream.ring.push_back(std::move(event));
        while (stream.ring.size() > ring_capacity_) {
            stream.first_seq = stream.ring.front().seq + 1;
            stream.ring.pop_front();
        }
    }
    cv_.notify_all();
    return seq;
}

void EventBus::close(const std::string & job_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_locked(job_id).closed = true;
    }
    cv_.notify_all();
}

EventBatch EventBus::replay(const std::string & job_id, int64_t after_seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    EventBatch batch;
    const auto found = streams_.find(job_id);
    if (found == streams_.end()) {
        return batch;
    }
    const auto & stream = found->second;
    if (after_seq + 1 < stream.first_seq) {
        // The caller is further behind than the ring goes. Say how much was lost
        // rather than handing over a sequence with a hole in it.
        batch.dropped = stream.first_seq - (after_seq + 1);
    }
    for (const auto & event : stream.ring) {
        if (event.seq > after_seq) {
            batch.events.push_back(event);
        }
    }
    batch.closed = stream.closed;
    return batch;
}

EventBatch EventBus::wait(const std::string & job_id, int64_t after_seq, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    for (;;) {
        auto & stream = stream_locked(job_id);
        EventBatch batch;
        if (after_seq + 1 < stream.first_seq) {
            batch.dropped = stream.first_seq - (after_seq + 1);
        }
        for (const auto & event : stream.ring) {
            if (event.seq > after_seq) {
                batch.events.push_back(event);
            }
        }
        batch.closed = stream.closed;
        if (!batch.events.empty() || batch.closed || shutdown_) {
            return batch;
        }
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            return EventBatch{};   // caller sends a heartbeat and comes back
        }
    }
}

bool EventBus::try_subscribe() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Each subscriber costs a connection thread, so this is a real resource
    // bound rather than a formality.
    if (subscribers_ >= max_subscribers_) {
        return false;
    }
    ++subscribers_;
    return true;
}

void EventBus::unsubscribe() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (subscribers_ > 0) {
        --subscribers_;
    }
}

size_t EventBus::subscribers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return subscribers_;
}

void EventBus::forget(const std::string & job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    streams_.erase(job_id);
}

void EventBus::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_EVENT_BUS
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

std::string seqs_of(const audiocpp::api::EventBatch & batch) {
    std::string out;
    for (const auto & event : batch.events) {
        if (!out.empty()) {
            out += ",";
        }
        out += std::to_string(event.seq);
    }
    return out;
}

}  // namespace

int main() {
    using namespace audiocpp::api;

    // ---- SSE framing ----
    guarded("sse_frame", [] {
        JobEvent event;
        event.seq = 7;
        event.type = "progress";
        event.json = "{\"step\":1}";
        record("sse_frame/shape", "one event", sse_frame(event),
               "event: progress\nid: 7\ndata: {\"step\":1}\n\n");
    });

    guarded("sse_frame/multiline", [] {
        // The case that silently breaks a client: a payload with a newline must
        // become two data: lines, not one broken field.
        JobEvent event;
        event.seq = 1;
        event.type = "log";
        event.json = "line one\nline two";
        record("sse_frame/splits_newlines", "payload with a newline", sse_frame(event),
               "event: log\nid: 1\ndata: line one\ndata: line two\n\n");
    });

    guarded("sse_comment", [] {
        record("sse_comment", "keepalive", sse_comment("keepalive"), ": keepalive\n\n");
    });

    // ---- sequencing ----
    guarded("publish/sequences", [] {
        EventBus bus;
        record("publish/first_seq", "first event", std::to_string(bus.publish("j1", "status", "{}", 0)), "1");
        record("publish/second_seq", "second event", std::to_string(bus.publish("j1", "status", "{}", 0)), "2");
        // Sequences are per job, not global.
        record("publish/per_job", "first event on another job",
               std::to_string(bus.publish("j2", "status", "{}", 0)), "1");
    });

    guarded("replay", [] {
        EventBus bus;
        for (int i = 0; i < 5; ++i) {
            bus.publish("j1", "progress", "{}", 0);
        }
        record("replay/from_zero", "5 events", seqs_of(bus.replay("j1", 0)), "1,2,3,4,5");
        record("replay/from_three", "after seq 3", seqs_of(bus.replay("j1", 3)), "4,5");
        record("replay/from_end", "after seq 5", seqs_of(bus.replay("j1", 5)), "");
        record("replay/unknown_job", "never published", seqs_of(bus.replay("nope", 0)), "");
        record("replay/not_closed", "job still running", bus.replay("j1", 0).closed ? "closed" : "open",
               "open");
    });

    guarded("replay/ring_eviction", [] {
        // A small ring, deliberately overrun: the caller must be told how much it
        // missed rather than handed a silent gap.
        EventBus bus(3);
        for (int i = 0; i < 6; ++i) {
            bus.publish("j1", "progress", "{}", 0);
        }
        const auto batch = bus.replay("j1", 0);
        record("replay/ring_bounded", "6 published, ring of 3", seqs_of(batch), "4,5,6");
        record("replay/reports_dropped", "3 aged out", std::to_string(batch.dropped), "3");
        record("replay/no_drop_when_current", "asking from seq 4",
               std::to_string(bus.replay("j1", 4).dropped), "0");
    });

    guarded("close", [] {
        EventBus bus;
        bus.publish("j1", "status", "{}", 0);
        bus.close("j1");
        const auto batch = bus.replay("j1", 0);
        record("close/marks_closed", "closed job", batch.closed ? "closed" : "open", "closed");
        record("close/keeps_events", "closed job", seqs_of(batch), "1");
        // A subscriber arriving after the end must not hang.
        const auto after = bus.wait("j1", 1, 5000);
        record("close/wait_returns_immediately", "subscribe after close",
               after.closed ? "closed" : "hung", "closed");
    });

    // ---- waiting ----
    guarded("wait/returns_pending", [] {
        EventBus bus;
        bus.publish("j1", "status", "{}", 0);
        record("wait/already_available", "one pending event", seqs_of(bus.wait("j1", 0, 1000)), "1");
    });

    guarded("wait/blocks_then_wakes", [] {
        EventBus bus;
        EventBatch got;
        std::thread waiter([&] { got = bus.wait("j1", 0, 5000); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        const bool blocked = got.events.empty();
        bus.publish("j1", "progress", "{\"step\":1}", 0);
        waiter.join();
        record("wait/blocked", "nothing published yet", blocked ? "blocked" : "returned early", "blocked");
        record("wait/woke", "after publish", seqs_of(got), "1");
    });

    guarded("wait/times_out", [] {
        EventBus bus;
        const auto start = std::chrono::steady_clock::now();
        const auto batch = bus.wait("quiet", 0, 60);
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                .count();
        record("wait/timeout_empty", "60ms, nothing published", batch.events.empty() ? "empty" : "events",
               "empty");
        record("wait/timeout_open", "60ms, nothing published", batch.closed ? "closed" : "open", "open");
        record("wait/timeout_waited", std::to_string(elapsed) + "ms", elapsed >= 50 ? "waited" : "returned early",
               "waited");
    });

    guarded("wait/wakes_on_close", [] {
        EventBus bus;
        EventBatch got;
        std::thread waiter([&] { got = bus.wait("j1", 0, 5000); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        bus.close("j1");
        waiter.join();
        record("wait/close_wakes", "close while waiting", got.closed ? "closed" : "still open", "closed");
    });

    guarded("shutdown/unblocks", [] {
        EventBus bus;
        bool returned = false;
        std::thread waiter([&] {
            bus.wait("j1", 0, 10000);
            returned = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        bus.shutdown();
        waiter.join();
        record("shutdown/unblocks_waiters", "shutdown while waiting", returned ? "returned" : "hung",
               "returned");
    });

    // ---- subscriber accounting ----
    guarded("subscribers", [] {
        EventBus bus(512, 2);
        record("try_subscribe/first", "limit 2", bus.try_subscribe() ? "accepted" : "refused", "accepted");
        record("try_subscribe/second", "limit 2", bus.try_subscribe() ? "accepted" : "refused", "accepted");
        // Each subscriber holds a connection thread, so the cap has to bite.
        record("try_subscribe/over_limit", "limit 2", bus.try_subscribe() ? "accepted" : "refused",
               "refused");
        record("subscribers/count", "two held", std::to_string(bus.subscribers()), "2");
        bus.unsubscribe();
        record("unsubscribe/frees_slot", "after one leaves", bus.try_subscribe() ? "accepted" : "refused",
               "accepted");
    });

    guarded("forget", [] {
        EventBus bus;
        bus.publish("j1", "status", "{}", 0);
        bus.forget("j1");
        record("forget/clears_ring", "after forget", seqs_of(bus.replay("j1", 0)), "");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_EVENT_BUS
