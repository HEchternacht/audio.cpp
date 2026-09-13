#pragma once

// One concern: letting a caller stop a running inference and watch it progress.
//
// ggml's abort_callback is CPU/Metal only -- there is no CUDA equivalent -- so
// cancellation cannot be delivered into a graph that is already executing. It has
// to be observed *between* work units instead: between decoded tokens, between
// ODE steps, between decoder tiles. A model therefore polls this object at the
// top of each such loop, and cancellation takes effect at the next boundary
// rather than instantly. What that costs depends on the loop: a token is tens of
// milliseconds, one flow-matching step over a whole sequence can be seconds.
//
// The object is created by whoever runs the task and observed by the model.
// cancel() is called from another thread while the model is inside run(), so the
// flag is atomic and the progress sink is installed once, before the run starts,
// and only read afterwards.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace engine::runtime {

// Where a run currently is. `total` is 0 when the loop's length is not known up
// front. Stage names are per-family; a family advertises its own set so a client
// can render them without hardcoding.
struct ProgressEvent {
    std::string stage;
    int64_t step = 0;
    int64_t total = 0;
    std::string message;
};

// Thrown out of a model's loop when cancellation is observed. A distinct type so
// a caller can tell "the user asked to stop" from "the model failed", which are
// different outcomes to report and different things for a user to do next.
class RunCancelled : public std::runtime_error {
public:
    RunCancelled() : std::runtime_error("run cancelled") {}
};

class RunControl {
public:
    // Callable from any thread, at any time, including before the run starts.
    void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
    bool cancelled() const noexcept { return cancelled_.load(std::memory_order_acquire); }

    void throw_if_cancelled() const {
        if (cancelled()) {
            throw RunCancelled();
        }
    }

    // Install before handing the control to a run; not thread-safe against the
    // run itself. Rate limiting belongs in the sink, not in the model: a model
    // reports every step unconditionally and the sink decides what to keep.
    void set_progress_sink(std::function<void(const ProgressEvent &)> sink) {
        sink_ = std::move(sink);
    }

    void progress(const ProgressEvent & event) const {
        if (sink_) {
            sink_(event);
        }
    }

private:
    std::atomic<bool> cancelled_{false};
    std::function<void(const ProgressEvent &)> sink_;
};

// Null-safe helpers, so a model's hot loop carries no branch of its own and a
// caller that does not care about cancellation passes nothing.
inline void run_checkpoint(const std::shared_ptr<RunControl> & control) {
    if (control) {
        control->throw_if_cancelled();
    }
}

inline void run_checkpoint(const std::shared_ptr<RunControl> & control, const ProgressEvent & event) {
    if (!control) {
        return;
    }
    control->throw_if_cancelled();
    control->progress(event);
}

}  // namespace engine::runtime
