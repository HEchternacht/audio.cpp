#pragma once

// One concern: keeping jobs on disk so they outlive the process.
//
// This is the file that answers the complaint the whole server exists for --
// "after generation history is not saved at all". Everything is written before
// the model runs, so a crash, a kill or a power cut leaves a record of what was
// asked for rather than a hole. The client's request is stored verbatim and
// never rewritten, so lyrics and style survive even if the run did not.
//
// Layout under the store root:
//
//   version                       refuse to start on an unknown one
//   jobs/<id>/job.json            authoritative metadata, atomically replaced
//   jobs/<id>/request.json        the client body, written once, never touched
//   jobs/<id>/events.ndjson       append-only; a torn tail is dropped on read
//   jobs/<id>/outputs/...         audio.wav, score.abc, semantic_codes.i32
//   jobs/.corrupt/<id>/           quarantine; one bad record must not stop boot
//   jobs/.trash/<id>/             rename-then-delete staging for the collector
//
// The root path arrives as a constructor argument and nothing happens at import
// time, so a self-test can point the whole store at a temp directory.

#include "job.h"

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace audiocpp::api {

struct RetentionPolicy {
    // Each bound is disabled by 0, the convention the rest of this codebase
    // already uses for limits (see LiveIngestLimits, busy_timeout_ms).
    int64_t max_age_ms = 0;
    int64_t max_jobs = 0;
    int64_t max_bytes = 0;
};

struct ReconcileReport {
    std::vector<std::string> requeued;     // were queued when we died; intent intact
    int interrupted = 0;                   // were running when we died
    int quarantined = 0;                   // unparseable, moved aside
    int removed = 0;                       // empty or half-created directories
    int loaded = 0;
};

struct JobQuery {
    std::vector<JobStatus> statuses;
    std::string model;
    std::string family;
    std::string operation;
    std::string text;                      // substring over metadata + request text
    std::optional<bool> pinned;
    std::optional<int64_t> created_after_ms;
    std::optional<int64_t> created_before_ms;
    std::string cursor;                    // an id; results continue strictly past it
    size_t limit = 50;
    bool newest_first = true;
};

struct JobPage {
    std::vector<Job> jobs;
    std::string next_cursor;
    size_t total_matched = 0;
};

class JobStore {
public:
    // `server_run_id` stamps every job this process owns, which is what lets
    // reconciliation tell "running right now" from "running when we died".
    JobStore(std::filesystem::path root, std::string server_run_id);

    const std::filesystem::path & root() const noexcept { return root_; }

    // Scans the store, repairs what the last process left behind, and fills the
    // in-memory index. Call once, before serving.
    ReconcileReport open();

    // Durable before it returns: a job that has been created is a job the client
    // can be told about.
    void create(const Job & job);
    void save(const Job & job);

    std::optional<Job> load(const std::string & id) const;
    bool exists(const std::string & id) const;
    JobPage list(const JobQuery & query) const;
    size_t count() const;

    // Writes one output file and records it on the job. Returns the stored
    // descriptor, including the byte count actually written.
    JobOutput write_output(Job & job, JobOutput output, const void * data, size_t bytes);
    std::filesystem::path output_path(const Job & job, const JobOutput & output) const;
    std::optional<JobOutput> find_output(const Job & job, const std::string & name) const;

    // Updates the in-memory index only. Progress arrives thousands of times per
    // run, and rewriting job.json for each would mean thousands of file writes
    // for information a crash makes moot anyway -- the record is reconciled as
    // interrupted either way. Polling still sees it because it reads the index.
    void update_progress(const std::string & id, const JobProgress & progress);

    void append_event(const std::string & id, const std::string & json_line);
    std::vector<std::string> read_events(const std::string & id, int64_t from_seq) const;

    bool remove(const std::string & id);
    // Applies the policy oldest-terminal-first. Never touches queued, running or
    // pinned jobs. Returns how many it collected.
    int collect(const RetentionPolicy & policy, int64_t now);

    std::filesystem::path job_dir(const std::string & id) const;

private:
    void write_json_atomically(const std::filesystem::path & path, const std::string & text) const;
    void index_locked(const Job & job);

    std::filesystem::path root_;
    std::string server_run_id_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Job> index_;
    std::vector<std::string> order_;   // ids, ascending; ULIDs make this chronological
};

// Exposed for testing: the substring match backing JobQuery::text.
bool job_matches_text(const Job & job, const std::string & needle);
bool job_matches(const Job & job, const JobQuery & query);

}  // namespace audiocpp::api
