#pragma once

// One concern: what a job *is*. The record, its statuses, its id format, and the
// translation to and from the JSON that lands on disk.
//
// Nothing here touches the filesystem, the queue, or a model. That separation is
// deliberate: this file defines the on-disk format, so it is the one that has to
// be right, and it is far easier to get right when it can be tested with no I/O
// at all.

#include "engine/framework/io/json.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace audiocpp::api {

// `Interrupted` is deliberately distinct from `Failed`: it means the server died
// while this job was running. That is an operational fact worth reporting
// honestly rather than blaming the model for, and it is the only terminal state
// for which offering "resubmit" makes sense.
enum class JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
};

const char * status_name(JobStatus status) noexcept;
std::optional<JobStatus> parse_status(std::string_view name) noexcept;
bool is_terminal(JobStatus status) noexcept;

// One produced file. `asset_id` is what makes chaining work: every output is
// registered as an asset, so editing a previous result is a reference rather
// than a re-upload.
struct JobOutput {
    std::string name;          // "audio" | "score" | "semantic_codes" | "result"
    std::string kind;          // "audio" | "score" | "codes" | "text" | "midi"
    std::string content_type;
    std::string filename;      // relative to the job directory
    uint64_t bytes = 0;
    std::string asset_id;
    std::optional<int> sample_rate;
    std::optional<int> channels;
    std::optional<double> duration_seconds;
    std::optional<int64_t> count;   // for code blobs
    bool missing = false;           // on disk at submit, gone by the time we looked
};

struct JobProgress {
    std::string stage;
    int64_t step = 0;
    int64_t total = 0;
    double fraction = 0.0;      // within the stage
    double overall = 0.0;       // across stages, using the family's stage weights
    std::string message;
    int64_t updated_ms = 0;
};

struct JobError {
    std::string message;
    std::string type;           // mirrors the HTTP error convention
    bool empty() const noexcept { return message.empty() && type.empty(); }
};

struct JobMetrics {
    std::optional<double> wall_ms;
    std::optional<double> queue_ms;
    std::optional<double> audio_seconds;
    std::optional<double> rtf;
};

struct Job {
    std::string id;
    std::string model_id;
    std::string family;
    std::string operation;
    JobStatus status = JobStatus::Queued;
    int priority = 0;

    int64_t created_ms = 0;
    int64_t updated_ms = 0;
    std::optional<int64_t> started_ms;
    std::optional<int64_t> finished_ms;

    // The client's body, verbatim. This is the point of the whole store: a
    // generation's lyrics, style, score and seed must be recoverable months
    // later, whatever happened to the run.
    engine::io::json::Value request;
    // What actually reached the model, after defaults and validation. Kept
    // separately because it answers a different question than `request` does.
    std::unordered_map<std::string, std::string> resolved_options;

    JobProgress progress;
    JobError error;
    std::vector<JobOutput> outputs;
    JobMetrics metrics;

    bool cancel_requested = false;
    bool pinned = false;
    std::optional<std::string> idempotency_key;
    engine::io::json::Value metadata;   // opaque to the server
    int64_t last_event_seq = 0;

    // Which server process owned this job. Drives startup reconciliation: a
    // `Running` record stamped with a foreign run id means we died mid-flight.
    std::string server_run_id;
    uint32_t schema_version = 1;
};

// ULID: 48-bit millisecond timestamp then 80 bits of randomness, Crockford
// base32. Sorts lexicographically by creation time, which is why a directory
// listing is already chronological and history pagination is a plain cursor
// comparison with no counter to persist and no offset scan.
std::string make_ulid(int64_t now_ms);
std::string make_job_id(int64_t now_ms);
std::string make_asset_id(int64_t now_ms);

// Accepts only ids this server could have minted. Every path built from a
// client-supplied id goes through here first, so a traversal attempt never
// reaches the filesystem.
bool is_valid_job_id(std::string_view id) noexcept;
bool is_valid_asset_id(std::string_view id) noexcept;

engine::io::json::Value job_to_json(const Job & job);
Job job_from_json(const engine::io::json::Value & value);

int64_t now_ms() noexcept;

}  // namespace audiocpp::api
