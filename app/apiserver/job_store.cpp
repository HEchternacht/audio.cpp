#include "job_store.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace audiocpp::api {
namespace {

constexpr const char * kStoreVersion = "1";
constexpr const char * kJobsDir = "jobs";
constexpr const char * kCorruptDir = ".corrupt";
constexpr const char * kTrashDir = ".trash";
constexpr const char * kJobFile = "job.json";
constexpr const char * kRequestFile = "request.json";
constexpr const char * kEventsFile = "events.ndjson";
constexpr const char * kOutputsDir = "outputs";

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

// Walks a JSON value collecting every string it contains, so a text search can
// cover lyrics and style without the store knowing what those fields are called.
void collect_strings(const engine::io::json::Value & value, std::string & out) {
    switch (value.kind()) {
        case engine::io::json::Value::Kind::String:
            out += value.as_string();
            out += '\n';
            break;
        case engine::io::json::Value::Kind::Array:
            for (const auto & item : value.as_array()) {
                collect_strings(item, out);
            }
            break;
        case engine::io::json::Value::Kind::Object:
            for (const auto & entry : value.as_object()) {
                collect_strings(entry.second, out);
            }
            break;
        default:
            break;
    }
}

uint64_t directory_bytes(const std::filesystem::path & dir) {
    uint64_t total = 0;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
        if (ec) {
            break;
        }
        if (it->is_regular_file(ec)) {
            total += it->file_size(ec);
        }
    }
    return total;
}

}  // namespace

bool job_matches_text(const Job & job, const std::string & needle) {
    if (needle.empty()) {
        return true;
    }
    std::string haystack;
    collect_strings(job.request, haystack);
    collect_strings(job.metadata, haystack);
    haystack += job.model_id;
    haystack += '\n';
    haystack += job.operation;
    return lowercase(haystack).find(lowercase(needle)) != std::string::npos;
}

bool job_matches(const Job & job, const JobQuery & query) {
    if (!query.statuses.empty() &&
        std::find(query.statuses.begin(), query.statuses.end(), job.status) == query.statuses.end()) {
        return false;
    }
    if (!query.model.empty() && job.model_id != query.model) {
        return false;
    }
    if (!query.family.empty() && job.family != query.family) {
        return false;
    }
    if (!query.operation.empty() && job.operation != query.operation) {
        return false;
    }
    if (query.pinned.has_value() && job.pinned != *query.pinned) {
        return false;
    }
    if (query.created_after_ms.has_value() && job.created_ms < *query.created_after_ms) {
        return false;
    }
    if (query.created_before_ms.has_value() && job.created_ms > *query.created_before_ms) {
        return false;
    }
    return job_matches_text(job, query.text);
}

JobStore::JobStore(std::filesystem::path root, std::string server_run_id)
    : root_(std::move(root)), server_run_id_(std::move(server_run_id)) {}

std::filesystem::path JobStore::job_dir(const std::string & id) const {
    // Never build a path from an id we did not mint: this is the only guard
    // between a URL path segment and the filesystem.
    if (!is_valid_job_id(id)) {
        throw std::runtime_error("refusing to build a path for a malformed job id: " + id);
    }
    return root_ / kJobsDir / id;
}

void JobStore::write_json_atomically(const std::filesystem::path & path, const std::string & text) const {
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("cannot open for write: " + tmp);
        }
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.flush();
        if (!out) {
            throw std::runtime_error("write failed: " + tmp);
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        // A virus scanner or an indexer can hold the target open briefly on
        // Windows; remove-then-rename is the documented fallback.
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            throw std::runtime_error("cannot replace " + path.string() + ": " + ec.message());
        }
    }
}

ReconcileReport JobStore::open() {
    std::lock_guard<std::mutex> lock(mutex_);
    ReconcileReport report;

    std::filesystem::create_directories(root_ / kJobsDir);

    const auto version_path = root_ / "version";
    if (std::filesystem::exists(version_path)) {
        std::ifstream in(version_path, std::ios::binary);
        std::string found;
        std::getline(in, found);
        while (!found.empty() && (found.back() == '\r' || found.back() == '\n')) {
            found.pop_back();
        }
        if (found != kStoreVersion) {
            throw std::runtime_error("job store at " + root_.string() + " has version '" + found +
                                     "', this build understands '" + kStoreVersion + "'");
        }
    } else {
        std::ofstream out(version_path, std::ios::binary | std::ios::trunc);
        out << kStoreVersion << "\n";
    }

    const auto jobs_root = root_ / kJobsDir;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(jobs_root, ec)) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name == kCorruptDir || name == kTrashDir) {
            continue;
        }

        const auto record = entry.path() / kJobFile;
        if (!std::filesystem::exists(record)) {
            // Crashed between mkdir and the first write; there is nothing to
            // recover and leaving it would confuse the next scan.
            std::filesystem::remove_all(entry.path(), ec);
            ++report.removed;
            continue;
        }

        // Stale temp files from a write that never completed.
        std::filesystem::remove(record.string() + ".tmp", ec);

        Job job;
        try {
            job = job_from_json(engine::io::json::parse_file(record));
        } catch (const std::exception &) {
            // One unreadable record must never stop the server from starting --
            // that would turn durability into an availability bug.
            const auto quarantine = jobs_root / kCorruptDir;
            std::filesystem::create_directories(quarantine, ec);
            std::filesystem::rename(entry.path(), quarantine / name, ec);
            if (ec) {
                std::filesystem::remove_all(entry.path(), ec);
            }
            ++report.quarantined;
            continue;
        }

        if (job.status == JobStatus::Running && job.server_run_id != server_run_id_) {
            job.status = JobStatus::Interrupted;
            job.finished_ms = now_ms();
            job.updated_ms = *job.finished_ms;
            std::string where = job.progress.stage.empty()
                                    ? std::string()
                                    : " (stage " + job.progress.stage + ", step " +
                                          std::to_string(job.progress.step) + " of " +
                                          std::to_string(job.progress.total) + ")";
            job.error.message = "the server stopped while this job was running" + where;
            job.error.type = "interrupted";
            write_json_atomically(record, engine::io::json::stringify(job_to_json(job)));
            ++report.interrupted;
        } else if (job.status == JobStatus::Queued && job.server_run_id != server_run_id_) {
            // Never started, so the client's intent is intact and the request is
            // on disk. Hand it back to the caller to re-enqueue.
            report.requeued.push_back(job.id);
        }

        // Report outputs that vanished rather than failing the job: history
        // should still show what was generated.
        for (auto & output : job.outputs) {
            output.missing = !std::filesystem::exists(output_path(job, output));
        }

        index_locked(job);
        ++report.loaded;
    }

    std::sort(order_.begin(), order_.end());
    // Requeue oldest first, so a restart resumes in the order the user asked.
    std::sort(report.requeued.begin(), report.requeued.end());
    return report;
}

void JobStore::index_locked(const Job & job) {
    if (index_.find(job.id) == index_.end()) {
        order_.push_back(job.id);
    }
    index_[job.id] = job;
}

void JobStore::create(const Job & job) {
    const auto dir = job_dir(job.id);
    std::filesystem::create_directories(dir / kOutputsDir);
    // request.json is written once and never rewritten. It is deliberate
    // redundancy with the `request` field inside job.json: if that record is
    // ever damaged, the user's lyrics are still recoverable from here.
    write_json_atomically(dir / kRequestFile, engine::io::json::stringify(job.request));
    write_json_atomically(dir / kJobFile, engine::io::json::stringify(job_to_json(job)));
    std::lock_guard<std::mutex> lock(mutex_);
    index_locked(job);
}

void JobStore::save(const Job & job) {
    write_json_atomically(job_dir(job.id) / kJobFile, engine::io::json::stringify(job_to_json(job)));
    std::lock_guard<std::mutex> lock(mutex_);
    index_locked(job);
}

std::optional<Job> JobStore::load(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = index_.find(id);
    if (found == index_.end()) {
        return std::nullopt;
    }
    return found->second;
}

bool JobStore::exists(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.find(id) != index_.end();
}

size_t JobStore::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.size();
}

JobPage JobStore::list(const JobQuery & query) const {
    std::lock_guard<std::mutex> lock(mutex_);
    JobPage page;

    std::vector<const Job *> matched;
    matched.reserve(index_.size());
    for (const auto & id : order_) {
        const auto found = index_.find(id);
        if (found != index_.end() && job_matches(found->second, query)) {
            matched.push_back(&found->second);
        }
    }
    page.total_matched = matched.size();

    if (query.newest_first) {
        std::reverse(matched.begin(), matched.end());
    }

    // ULIDs sort chronologically, so the cursor is a plain comparison -- no
    // offset scan, and no skew if jobs are added while a client pages.
    size_t start = 0;
    if (!query.cursor.empty()) {
        while (start < matched.size() && matched[start]->id != query.cursor) {
            ++start;
        }
        if (start < matched.size()) {
            ++start;
        }
    }

    const size_t limit = query.limit == 0 ? matched.size() : query.limit;
    for (size_t i = start; i < matched.size() && page.jobs.size() < limit; ++i) {
        page.jobs.push_back(*matched[i]);
    }
    if (start + page.jobs.size() < matched.size() && !page.jobs.empty()) {
        page.next_cursor = page.jobs.back().id;
    }
    return page;
}

std::filesystem::path JobStore::output_path(const Job & job, const JobOutput & output) const {
    return job_dir(job.id) / kOutputsDir / output.filename;
}

std::optional<JobOutput> JobStore::find_output(const Job & job, const std::string & name) const {
    for (const auto & output : job.outputs) {
        if (output.name == name) {
            return output;
        }
    }
    return std::nullopt;
}

JobOutput JobStore::write_output(Job & job, JobOutput output, const void * data, size_t bytes) {
    // The filename is server-chosen, never client-supplied, so there is nothing
    // to sanitise here -- but assert it anyway, because that invariant is what
    // makes the download route safe.
    if (output.filename.find('/') != std::string::npos ||
        output.filename.find('\\') != std::string::npos ||
        output.filename.find("..") != std::string::npos) {
        throw std::runtime_error("output filename must be a plain name: " + output.filename);
    }
    const auto dir = job_dir(job.id) / kOutputsDir;
    std::filesystem::create_directories(dir);
    const auto path = dir / output.filename;
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("cannot write output: " + path.string());
        }
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        out.flush();
        if (!out) {
            throw std::runtime_error("output write failed: " + path.string());
        }
    }
    output.bytes = bytes;
    output.missing = false;
    if (output.asset_id.empty()) {
        output.asset_id = make_asset_id(now_ms());
    }

    auto existing = std::find_if(job.outputs.begin(), job.outputs.end(),
                                 [&](const JobOutput & o) { return o.name == output.name; });
    if (existing != job.outputs.end()) {
        *existing = output;
    } else {
        job.outputs.push_back(output);
    }
    return output;
}

void JobStore::update_progress(const std::string & id, const JobProgress & progress) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = index_.find(id);
    if (found != index_.end()) {
        found->second.progress = progress;
    }
}

void JobStore::append_event(const std::string & id, const std::string & json_line) {
    const auto path = job_dir(id) / kEventsFile;
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) {
        return;   // events are best-effort; losing one must not fail a run
    }
    out << json_line << "\n";
}

std::vector<std::string> JobStore::read_events(const std::string & id, int64_t from_seq) const {
    std::vector<std::string> lines;
    std::ifstream in(job_dir(id) / kEventsFile, std::ios::binary);
    if (!in) {
        return lines;
    }
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        // A crash can leave a partial final line; it will not parse, so drop it
        // rather than handing a client malformed JSON.
        try {
            const auto value = engine::io::json::parse(line);
            if (engine::io::json::optional_i64(value, "seq", 0) < from_seq) {
                continue;
            }
        } catch (const std::exception &) {
            continue;
        }
        lines.push_back(line);
    }
    return lines;
}

bool JobStore::remove(const std::string & id) {
    const auto dir = job_dir(id);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = index_.find(id);
        if (found == index_.end()) {
            return false;
        }
        index_.erase(found);
        order_.erase(std::remove(order_.begin(), order_.end(), id), order_.end());
    }
    // Rename into .trash first: a directory caught half-deleted by a crash must
    // never look like a live job on the next boot.
    std::error_code ec;
    const auto trash = root_ / kJobsDir / kTrashDir;
    std::filesystem::create_directories(trash, ec);
    const auto staged = trash / id;
    std::filesystem::rename(dir, staged, ec);
    std::filesystem::remove_all(ec ? dir : staged, ec);
    return true;
}

int JobStore::collect(const RetentionPolicy & policy, int64_t now) {
    std::vector<std::string> candidates;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Clear any trash a previous run left behind.
        std::error_code ec;
        std::filesystem::remove_all(root_ / kJobsDir / kTrashDir, ec);

        for (const auto & id : order_) {
            const auto found = index_.find(id);
            if (found == index_.end()) {
                continue;
            }
            const auto & job = found->second;
            if (!is_terminal(job.status) || job.pinned) {
                continue;
            }
            candidates.push_back(id);
        }
    }
    // `order_` is ascending, so candidates are already oldest-first.

    std::vector<std::string> doomed;
    if (policy.max_age_ms > 0) {
        for (const auto & id : candidates) {
            const auto job = load(id);
            if (job.has_value() && now - job->created_ms > policy.max_age_ms) {
                doomed.push_back(id);
            }
        }
    }

    auto not_yet_doomed = [&](const std::string & id) {
        return std::find(doomed.begin(), doomed.end(), id) == doomed.end();
    };

    if (policy.max_jobs > 0) {
        const auto total = static_cast<int64_t>(count());
        int64_t over = total - static_cast<int64_t>(doomed.size()) - policy.max_jobs;
        for (const auto & id : candidates) {
            if (over <= 0) {
                break;
            }
            if (not_yet_doomed(id)) {
                doomed.push_back(id);
                --over;
            }
        }
    }

    if (policy.max_bytes > 0) {
        int64_t total = 0;
        for (const auto & id : candidates) {
            total += static_cast<int64_t>(directory_bytes(job_dir(id)));
        }
        for (const auto & id : candidates) {
            if (total <= policy.max_bytes) {
                break;
            }
            if (not_yet_doomed(id)) {
                total -= static_cast<int64_t>(directory_bytes(job_dir(id)));
                doomed.push_back(id);
            }
        }
    }

    int collected = 0;
    for (const auto & id : doomed) {
        if (remove(id)) {
            ++collected;
        }
    }

    // Drop the staging directory itself once everything in it is gone, so a
    // scan of jobs/ shows only jobs.
    std::error_code ec;
    std::filesystem::remove(root_ / kJobsDir / kTrashDir, ec);
    return collected;
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_JOB_STORE
// Direct-execution self-test. Uses a temp directory it creates and removes; no
// fixtures, no network, no model.
#include <cstdio>
#include <functional>
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

void expect_throws(const char * unit, const std::string & in, const std::function<void()> & body) {
    bool threw = false;
    try {
        body();
    } catch (const std::exception &) {
        threw = true;
    }
    record(unit, in, threw ? "throws" : "returns", "throws");
}

std::filesystem::path fresh_root() {
    static std::mt19937_64 rng{std::random_device{}()};
    auto dir = std::filesystem::temp_directory_path() /
               ("audiocpp_jobstore_selftest_" + std::to_string(rng()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

audiocpp::api::Job sample_job(audiocpp::api::JobStatus status, int64_t created, const char * run_id) {
    using namespace audiocpp::api;
    using engine::io::json::Value;
    Job job;
    job.id = make_job_id(created);
    job.model_id = "yue2-3b-q8";
    job.family = "yue2";
    job.operation = "generate";
    job.status = status;
    job.created_ms = created;
    job.updated_ms = created;
    job.server_run_id = run_id;
    Value::Object request;
    request["style"] = Value::make_string("dreamy synth pop");
    request["lyrics"] = Value::make_string("[verse]\nneon rain on the causeway\n");
    job.request = Value::make_object(std::move(request));
    return job;
}

}  // namespace

int main() {
    using namespace audiocpp::api;
    using engine::io::json::Value;

    // ---- round trip through disk ----
    guarded("create/load", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        auto job = sample_job(JobStatus::Queued, 1757600000000LL, "run-A");
        store.create(job);

        record("create/exists_on_disk", "job.json",
               std::filesystem::exists(root / "jobs" / job.id / "job.json") ? "present" : "absent",
               "present");
        record("create/request_sidecar", "request.json",
               std::filesystem::exists(root / "jobs" / job.id / "request.json") ? "present" : "absent",
               "present");

        // A second store over the same root must see it -- that is the whole point.
        JobStore reopened(root, "run-B");
        const auto report = reopened.open();
        const auto back = reopened.load(job.id);
        record("load/after_reopen", job.id, back.has_value() ? back->id : "<missing>", job.id);
        record("load/lyrics_survive", "verbatim lyrics",
               back.has_value() ? back->request.require("lyrics").as_string()
                                : std::string("<missing>"),
               "[verse]\nneon rain on the causeway\n");
        record("open/requeued_queued_job", "queued under a foreign run id",
               std::to_string(report.requeued.size()), "1");
        std::filesystem::remove_all(root);
    });

    // ---- atomic replace ----
    guarded("save/atomic", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        auto job = sample_job(JobStatus::Queued, 1757600000000LL, "run-A");
        store.create(job);
        job.status = JobStatus::Running;
        job.progress.stage = "semantic";
        store.save(job);
        job.status = JobStatus::Succeeded;
        store.save(job);

        record("save/no_tmp_residue", "two writes",
               std::filesystem::exists(root / "jobs" / job.id / "job.json.tmp") ? "residue" : "clean",
               "clean");
        JobStore reopened(root, "run-B");
        reopened.open();
        const auto back = reopened.load(job.id);
        record("save/last_write_wins", "succeeded",
               back.has_value() ? status_name(back->status) : "<missing>", "succeeded");
        std::filesystem::remove_all(root);
    });

    // ---- crash reconciliation: the restart test, in miniature ----
    guarded("open/reconcile", [] {
        const auto root = fresh_root();
        {
            JobStore store(root, "run-A");
            store.open();
            auto running = sample_job(JobStatus::Running, 1757600000000LL, "run-A");
            running.progress.stage = "semantic";
            running.progress.step = 1840;
            running.progress.total = 9000;
            store.create(running);
            store.create(sample_job(JobStatus::Queued, 1757600001000LL, "run-A"));
            store.create(sample_job(JobStatus::Succeeded, 1757600002000LL, "run-A"));
        }
        // A different process id: everything above was owned by a server that died.
        JobStore restarted(root, "run-B");
        const auto report = restarted.open();
        record("reconcile/loaded", "3 jobs", std::to_string(report.loaded), "3");
        record("reconcile/interrupted", "1 was running", std::to_string(report.interrupted), "1");
        record("reconcile/requeued", "1 was queued", std::to_string(report.requeued.size()), "1");

        JobQuery all;
        all.limit = 0;
        bool saw_interrupted = false;
        std::string message;
        for (const auto & job : restarted.list(all).jobs) {
            if (job.status == JobStatus::Interrupted) {
                saw_interrupted = true;
                message = job.error.message;
            }
        }
        record("reconcile/marks_interrupted", "running job", saw_interrupted ? "interrupted" : "not marked",
               "interrupted");
        record("reconcile/error_names_stage", "stage semantic step 1840",
               message.find("stage semantic, step 1840 of 9000") != std::string::npos ? "named" : message,
               "named");
        // A succeeded job must be left exactly alone.
        bool succeeded_intact = false;
        for (const auto & job : restarted.list(all).jobs) {
            if (job.status == JobStatus::Succeeded) {
                succeeded_intact = job.error.empty();
            }
        }
        record("reconcile/leaves_terminal_alone", "succeeded job",
               succeeded_intact ? "untouched" : "modified", "untouched");
        std::filesystem::remove_all(root);
    });

    // ---- a corrupt record must not stop the store from opening ----
    guarded("open/quarantine", [] {
        const auto root = fresh_root();
        std::string good_id;
        {
            JobStore store(root, "run-A");
            store.open();
            auto good = sample_job(JobStatus::Succeeded, 1757600000000LL, "run-A");
            good_id = good.id;
            store.create(good);
            auto bad = sample_job(JobStatus::Succeeded, 1757600001000LL, "run-A");
            store.create(bad);
            std::ofstream out(root / "jobs" / bad.id / "job.json", std::ios::binary | std::ios::trunc);
            out << "{\"id\": truncated";
        }
        JobStore reopened(root, "run-B");
        const auto report = reopened.open();
        record("quarantine/count", "1 truncated record", std::to_string(report.quarantined), "1");
        record("quarantine/good_still_loads", good_id,
               reopened.load(good_id).has_value() ? "loaded" : "lost", "loaded");
        record("quarantine/moved_aside", ".corrupt",
               std::filesystem::exists(root / "jobs" / ".corrupt") ? "present" : "absent", "present");
        std::filesystem::remove_all(root);
    });

    // ---- a directory with no record is debris from a crashed submit ----
    guarded("open/removes_empty", [] {
        const auto root = fresh_root();
        const auto orphan = root / "jobs" / make_job_id(1757600000000LL);
        std::filesystem::create_directories(orphan);
        JobStore store(root, "run-A");
        const auto report = store.open();
        record("open/removed_empty_dir", "dir with no job.json", std::to_string(report.removed), "1");
        record("open/dir_gone", orphan.filename().string(),
               std::filesystem::exists(orphan) ? "present" : "removed", "removed");
        std::filesystem::remove_all(root);
    });

    // ---- outputs ----
    guarded("write_output", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        auto job = sample_job(JobStatus::Running, 1757600000000LL, "run-A");
        store.create(job);

        const std::string payload = "RIFFxxxxWAVE";
        JobOutput out;
        out.name = "audio";
        out.kind = "audio";
        out.content_type = "audio/wav";
        out.filename = "audio.wav";
        const auto written = store.write_output(job, out, payload.data(), payload.size());
        store.save(job);

        record("write_output/bytes", "12 bytes", std::to_string(written.bytes), "12");
        record("write_output/asset_id", "auto-assigned",
               is_valid_asset_id(written.asset_id) ? "valid" : written.asset_id, "valid");
        record("write_output/on_disk", "outputs/audio.wav",
               std::filesystem::exists(root / "jobs" / job.id / "outputs" / "audio.wav") ? "present" : "absent",
               "present");
        record("find_output/hit", "audio",
               store.find_output(job, "audio").has_value() ? "found" : "missing", "found");
        record("find_output/miss", "score",
               store.find_output(job, "score").has_value() ? "found" : "missing", "missing");

        // Rewriting the same name replaces rather than duplicating.
        store.write_output(job, out, payload.data(), payload.size());
        record("write_output/replaces", "same name twice", std::to_string(job.outputs.size()), "1");
        std::filesystem::remove_all(root);
    });

    guarded("write_output/rejects_path", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        auto job = sample_job(JobStatus::Running, 1757600000000LL, "run-A");
        store.create(job);
        for (const char * name : {"../escape.wav", "sub/dir.wav", "back\\slash.wav"}) {
            JobOutput bad;
            bad.name = "audio";
            bad.filename = name;
            expect_throws("write_output/rejects_path", name,
                          [&] { store.write_output(job, bad, "x", 1); });
        }
        std::filesystem::remove_all(root);
    });

    // ---- path guard ----
    guarded("job_dir/guard", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        for (const char * bad : {"../../etc", "job_x", "", "job_0000000000000000000000000/.."}) {
            expect_throws("job_dir/guard", std::string("\"") + bad + "\"",
                          [&] { store.job_dir(bad); });
        }
        std::filesystem::remove_all(root);
    });

    // ---- events ----
    guarded("events", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        auto job = sample_job(JobStatus::Running, 1757600000000LL, "run-A");
        store.create(job);
        store.append_event(job.id, "{\"seq\":1,\"type\":\"status\"}");
        store.append_event(job.id, "{\"seq\":2,\"type\":\"progress\"}");
        store.append_event(job.id, "{\"seq\":3,\"type\":\"done\"}");
        record("read_events/all", "from_seq=0", std::to_string(store.read_events(job.id, 0).size()), "3");
        record("read_events/from_seq", "from_seq=3", std::to_string(store.read_events(job.id, 3).size()), "1");

        // A torn final line, as a crash would leave it.
        {
            std::ofstream out(root / "jobs" / job.id / "events.ndjson", std::ios::binary | std::ios::app);
            out << "{\"seq\":4,\"ty";
        }
        record("read_events/drops_torn_tail", "partial last line",
               std::to_string(store.read_events(job.id, 0).size()), "3");
        std::filesystem::remove_all(root);
    });

    // ---- querying ----
    guarded("list/filter", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        for (int i = 0; i < 5; ++i) {
            auto job = sample_job(i % 2 == 0 ? JobStatus::Succeeded : JobStatus::Failed,
                                  1757600000000LL + i * 1000, "run-A");
            if (i == 3) {
                job.model_id = "ace-step";
                job.family = "ace_step";
                Value::Object request;
                request["text"] = Value::make_string("a lonely trumpet in a warehouse");
                job.request = Value::make_object(std::move(request));
            }
            store.create(job);
        }

        JobQuery all;
        all.limit = 0;
        record("list/all", "5 jobs", std::to_string(store.list(all).total_matched), "5");

        JobQuery succeeded;
        succeeded.statuses = {JobStatus::Succeeded};
        succeeded.limit = 0;
        record("list/by_status", "succeeded", std::to_string(store.list(succeeded).total_matched), "3");

        JobQuery by_model;
        by_model.model = "ace-step";
        by_model.limit = 0;
        record("list/by_model", "ace-step", std::to_string(store.list(by_model).total_matched), "1");

        JobQuery by_text;
        by_text.text = "TRUMPET";     // case-insensitive, and inside the request body
        by_text.limit = 0;
        record("list/by_text", "\"TRUMPET\"", std::to_string(store.list(by_text).total_matched), "1");

        JobQuery none;
        none.text = "nothing matches this";
        none.limit = 0;
        record("list/no_match", "absent text", std::to_string(store.list(none).total_matched), "0");

        // Newest first by default, because a history view wants the last take on top.
        JobQuery ordered;
        ordered.limit = 0;
        const auto page = store.list(ordered);
        record("list/newest_first", "5 jobs",
               page.jobs.front().created_ms > page.jobs.back().created_ms ? "descending" : "ascending",
               "descending");
        std::filesystem::remove_all(root);
    });

    guarded("list/paginate", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        for (int i = 0; i < 7; ++i) {
            store.create(sample_job(JobStatus::Succeeded, 1757600000000LL + i * 1000, "run-A"));
        }
        JobQuery page1;
        page1.limit = 3;
        const auto first = store.list(page1);
        record("list/page_size", "limit 3", std::to_string(first.jobs.size()), "3");
        record("list/has_cursor", "more remain", first.next_cursor.empty() ? "none" : "present", "present");

        JobQuery page2 = page1;
        page2.cursor = first.next_cursor;
        const auto second = store.list(page2);
        record("list/page2_size", "limit 3 after cursor", std::to_string(second.jobs.size()), "3");
        record("list/no_overlap", "page1 vs page2",
               first.jobs.back().id != second.jobs.front().id ? "disjoint" : "overlaps", "disjoint");

        JobQuery page3 = page1;
        page3.cursor = second.next_cursor;
        const auto third = store.list(page3);
        record("list/last_page", "remainder", std::to_string(third.jobs.size()), "1");
        record("list/end_cursor", "nothing left", third.next_cursor.empty() ? "none" : "present", "none");
        std::filesystem::remove_all(root);
    });

    // ---- deletion and retention ----
    guarded("remove", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        auto job = sample_job(JobStatus::Succeeded, 1757600000000LL, "run-A");
        store.create(job);
        record("remove/hit", job.id, store.remove(job.id) ? "removed" : "absent", "removed");
        record("remove/dir_gone", "job dir",
               std::filesystem::exists(root / "jobs" / job.id) ? "present" : "gone", "gone");
        record("remove/miss", "second call", store.remove(job.id) ? "removed" : "absent", "absent");
        std::filesystem::remove_all(root);
    });

    guarded("collect/by_age", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        const int64_t now = 1757600000000LL;
        store.create(sample_job(JobStatus::Succeeded, now - 40LL * 86400000LL, "run-A"));  // 40 days
        store.create(sample_job(JobStatus::Succeeded, now - 1000, "run-A"));               // fresh
        RetentionPolicy policy;
        policy.max_age_ms = 30LL * 86400000LL;
        record("collect/by_age", "30 day limit, one 40 day old job",
               std::to_string(store.collect(policy, now)), "1");
        record("collect/by_age_remaining", "after collection", std::to_string(store.count()), "1");
        std::filesystem::remove_all(root);
    });

    guarded("collect/by_count", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        for (int i = 0; i < 6; ++i) {
            store.create(sample_job(JobStatus::Succeeded, 1757600000000LL + i * 1000, "run-A"));
        }
        RetentionPolicy policy;
        policy.max_jobs = 3;
        record("collect/by_count", "6 jobs, keep 3", std::to_string(store.collect(policy, now_ms())), "3");
        record("collect/by_count_remaining", "after collection", std::to_string(store.count()), "3");
        std::filesystem::remove_all(root);
    });

    guarded("collect/protects", [] {
        const auto root = fresh_root();
        JobStore store(root, "run-A");
        store.open();
        auto pinned = sample_job(JobStatus::Succeeded, 1757600000000LL, "run-A");
        pinned.pinned = true;
        store.create(pinned);
        store.create(sample_job(JobStatus::Queued, 1757600001000LL, "run-A"));
        store.create(sample_job(JobStatus::Running, 1757600002000LL, "run-A"));
        auto doomed = sample_job(JobStatus::Failed, 1757600003000LL, "run-A");
        store.create(doomed);

        RetentionPolicy policy;
        policy.max_jobs = 1;   // aggressive: everything collectable should go
        const int collected = store.collect(policy, now_ms());
        record("collect/only_terminal_unpinned", "pinned + queued + running + failed",
               std::to_string(collected), "1");
        record("collect/pinned_survives", pinned.id,
               store.load(pinned.id).has_value() ? "kept" : "collected", "kept");
        record("collect/failed_collected", doomed.id,
               store.load(doomed.id).has_value() ? "kept" : "collected", "collected");
        record("collect/no_trash_residue", ".trash",
               std::filesystem::exists(root / "jobs" / ".trash") ? "residue" : "clean", "clean");
        std::filesystem::remove_all(root);
    });

    // ---- store version ----
    guarded("open/version_guard", [] {
        const auto root = fresh_root();
        {
            JobStore store(root, "run-A");
            store.open();
        }
        {
            std::ofstream out(root / "version", std::ios::binary | std::ios::trunc);
            out << "99\n";
        }
        expect_throws("open/version_guard", "version 99 on disk", [&] {
            JobStore store(root, "run-B");
            store.open();
        });
        std::filesystem::remove_all(root);
    });

    // ---- pure predicates ----
    guarded("job_matches_text", [] {
        auto job = sample_job(JobStatus::Succeeded, 1757600000000LL, "run-A");
        record("job_matches_text/empty", "\"\"", job_matches_text(job, "") ? "match" : "no match", "match");
        record("job_matches_text/lyrics", "\"causeway\"",
               job_matches_text(job, "causeway") ? "match" : "no match", "match");
        record("job_matches_text/case", "\"NEON\"",
               job_matches_text(job, "NEON") ? "match" : "no match", "match");
        record("job_matches_text/model", "\"yue2-3b\"",
               job_matches_text(job, "yue2-3b") ? "match" : "no match", "match");
        record("job_matches_text/absent", "\"bagpipes\"",
               job_matches_text(job, "bagpipes") ? "match" : "no match", "no match");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_JOB_STORE
