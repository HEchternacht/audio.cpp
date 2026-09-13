#include "job.h"

#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>

namespace audiocpp::api {
namespace {

using engine::io::json::Value;

// Crockford base32: no I, L, O or U, so an id cannot be misread aloud or
// mistyped into a different id that happens to be valid.
constexpr char kCrockford[] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
constexpr size_t kUlidLength = 26;
constexpr const char * kJobPrefix = "job_";
constexpr const char * kAssetPrefix = "as_";

bool is_crockford(char c) noexcept {
    return c != '\0' && std::strchr(kCrockford, c) != nullptr;
}

// One generator for the process. The mutex matters because ids are minted from
// HTTP threads and workers alike.
uint64_t random_bits() {
    static std::mutex mutex;
    static std::mt19937_64 rng{std::random_device{}()};
    std::lock_guard<std::mutex> lock(mutex);
    return rng();
}

bool valid_id(std::string_view id, const char * prefix) noexcept {
    const size_t prefix_len = std::strlen(prefix);
    if (id.size() != prefix_len + kUlidLength) {
        return false;
    }
    if (id.compare(0, prefix_len, prefix) != 0) {
        return false;
    }
    for (size_t i = prefix_len; i < id.size(); ++i) {
        if (!is_crockford(id[i])) {
            return false;
        }
    }
    return true;
}

void put_optional_i64(Value::Object & object, const std::string & key, const std::optional<int64_t> & value) {
    if (value.has_value()) {
        object[key] = Value::make_number(static_cast<double>(*value));
    }
}

void put_optional_double(Value::Object & object, const std::string & key, const std::optional<double> & value) {
    if (value.has_value()) {
        object[key] = Value::make_number(*value);
    }
}

void put_optional_int(Value::Object & object, const std::string & key, const std::optional<int> & value) {
    if (value.has_value()) {
        object[key] = Value::make_number(static_cast<double>(*value));
    }
}

std::optional<int64_t> read_optional_i64(const Value & object, const std::string & key) {
    const auto * found = object.find(key);
    if (found == nullptr || !found->is_number()) {
        return std::nullopt;
    }
    return found->as_i64();
}

std::optional<double> read_optional_double(const Value & object, const std::string & key) {
    const auto * found = object.find(key);
    if (found == nullptr || !found->is_number()) {
        return std::nullopt;
    }
    return found->as_number();
}

std::optional<int> read_optional_int(const Value & object, const std::string & key) {
    const auto value = read_optional_i64(object, key);
    if (!value.has_value()) {
        return std::nullopt;
    }
    return static_cast<int>(*value);
}

}  // namespace

const char * status_name(JobStatus status) noexcept {
    switch (status) {
        case JobStatus::Queued: return "queued";
        case JobStatus::Running: return "running";
        case JobStatus::Succeeded: return "succeeded";
        case JobStatus::Failed: return "failed";
        case JobStatus::Cancelled: return "cancelled";
        case JobStatus::Interrupted: return "interrupted";
    }
    return "unknown";
}

std::optional<JobStatus> parse_status(std::string_view name) noexcept {
    if (name == "queued") return JobStatus::Queued;
    if (name == "running") return JobStatus::Running;
    if (name == "succeeded") return JobStatus::Succeeded;
    if (name == "failed") return JobStatus::Failed;
    if (name == "cancelled") return JobStatus::Cancelled;
    if (name == "interrupted") return JobStatus::Interrupted;
    return std::nullopt;
}

bool is_terminal(JobStatus status) noexcept {
    return status == JobStatus::Succeeded || status == JobStatus::Failed ||
           status == JobStatus::Cancelled || status == JobStatus::Interrupted;
}

std::string make_ulid(int64_t now_ms) {
    // 10 characters of timestamp (48 bits) then 16 of randomness (80 bits).
    std::array<char, kUlidLength> out{};
    auto stamp = static_cast<uint64_t>(now_ms) & 0xFFFFFFFFFFFFULL;
    for (int i = 9; i >= 0; --i) {
        out[static_cast<size_t>(i)] = kCrockford[stamp & 0x1F];
        stamp >>= 5;
    }
    uint64_t high = random_bits();
    uint64_t low = random_bits();
    for (int i = 25; i >= 18; --i) {
        out[static_cast<size_t>(i)] = kCrockford[low & 0x1F];
        low >>= 5;
    }
    for (int i = 17; i >= 10; --i) {
        out[static_cast<size_t>(i)] = kCrockford[high & 0x1F];
        high >>= 5;
    }
    return std::string(out.data(), kUlidLength);
}

std::string make_job_id(int64_t now_ms) { return std::string(kJobPrefix) + make_ulid(now_ms); }

std::string make_asset_id(int64_t now_ms) { return std::string(kAssetPrefix) + make_ulid(now_ms); }

bool is_valid_job_id(std::string_view id) noexcept { return valid_id(id, kJobPrefix); }

bool is_valid_asset_id(std::string_view id) noexcept { return valid_id(id, kAssetPrefix); }

int64_t now_ms() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

Value job_to_json(const Job & job) {
    Value::Object root;
    root["schema_version"] = Value::make_number(job.schema_version);
    root["id"] = Value::make_string(job.id);
    root["model"] = Value::make_string(job.model_id);
    root["family"] = Value::make_string(job.family);
    root["operation"] = Value::make_string(job.operation);
    root["status"] = Value::make_string(status_name(job.status));
    root["priority"] = Value::make_number(job.priority);
    root["created_ms"] = Value::make_number(static_cast<double>(job.created_ms));
    root["updated_ms"] = Value::make_number(static_cast<double>(job.updated_ms));
    put_optional_i64(root, "started_ms", job.started_ms);
    put_optional_i64(root, "finished_ms", job.finished_ms);
    root["request"] = job.request;
    root["metadata"] = job.metadata;
    root["cancel_requested"] = Value::make_bool(job.cancel_requested);
    root["pinned"] = Value::make_bool(job.pinned);
    root["last_event_seq"] = Value::make_number(static_cast<double>(job.last_event_seq));
    root["server_run_id"] = Value::make_string(job.server_run_id);
    if (job.idempotency_key.has_value()) {
        root["idempotency_key"] = Value::make_string(*job.idempotency_key);
    }

    Value::Object options;
    for (const auto & entry : job.resolved_options) {
        options[entry.first] = Value::make_string(entry.second);
    }
    root["resolved_options"] = Value::make_object(std::move(options));

    Value::Object progress;
    progress["stage"] = Value::make_string(job.progress.stage);
    progress["step"] = Value::make_number(static_cast<double>(job.progress.step));
    progress["total"] = Value::make_number(static_cast<double>(job.progress.total));
    progress["fraction"] = Value::make_number(job.progress.fraction);
    progress["overall"] = Value::make_number(job.progress.overall);
    progress["message"] = Value::make_string(job.progress.message);
    progress["updated_ms"] = Value::make_number(static_cast<double>(job.progress.updated_ms));
    root["progress"] = Value::make_object(std::move(progress));

    if (job.error.empty()) {
        root["error"] = Value::make_null();
    } else {
        Value::Object error;
        error["message"] = Value::make_string(job.error.message);
        error["type"] = Value::make_string(job.error.type);
        root["error"] = Value::make_object(std::move(error));
    }

    Value::Object metrics;
    put_optional_double(metrics, "wall_ms", job.metrics.wall_ms);
    put_optional_double(metrics, "queue_ms", job.metrics.queue_ms);
    put_optional_double(metrics, "audio_seconds", job.metrics.audio_seconds);
    put_optional_double(metrics, "rtf", job.metrics.rtf);
    root["metrics"] = Value::make_object(std::move(metrics));

    Value::Array outputs;
    outputs.reserve(job.outputs.size());
    for (const auto & output : job.outputs) {
        Value::Object entry;
        entry["name"] = Value::make_string(output.name);
        entry["kind"] = Value::make_string(output.kind);
        entry["content_type"] = Value::make_string(output.content_type);
        entry["filename"] = Value::make_string(output.filename);
        entry["bytes"] = Value::make_number(static_cast<double>(output.bytes));
        entry["asset_id"] = Value::make_string(output.asset_id);
        entry["missing"] = Value::make_bool(output.missing);
        put_optional_int(entry, "sample_rate", output.sample_rate);
        put_optional_int(entry, "channels", output.channels);
        put_optional_double(entry, "duration_seconds", output.duration_seconds);
        put_optional_i64(entry, "count", output.count);
        outputs.push_back(Value::make_object(std::move(entry)));
    }
    root["outputs"] = Value::make_array(std::move(outputs));

    return Value::make_object(std::move(root));
}

Job job_from_json(const Value & value) {
    if (!value.is_object()) {
        throw std::runtime_error("job record must be a JSON object");
    }
    Job job;
    job.schema_version = static_cast<uint32_t>(engine::io::json::optional_i64(value, "schema_version", 1));
    job.id = engine::io::json::require_string(value, "id");
    if (!is_valid_job_id(job.id)) {
        throw std::runtime_error("job record has a malformed id: " + job.id);
    }
    job.model_id = engine::io::json::optional_string(value, "model", "");
    job.family = engine::io::json::optional_string(value, "family", "");
    job.operation = engine::io::json::optional_string(value, "operation", "");

    const auto status_text = engine::io::json::optional_string(value, "status", "");
    const auto status = parse_status(status_text);
    if (!status.has_value()) {
        throw std::runtime_error("job record has an unknown status: " + status_text);
    }
    job.status = *status;

    job.priority = static_cast<int>(engine::io::json::optional_i64(value, "priority", 0));
    job.created_ms = engine::io::json::optional_i64(value, "created_ms", 0);
    job.updated_ms = engine::io::json::optional_i64(value, "updated_ms", 0);
    job.started_ms = read_optional_i64(value, "started_ms");
    job.finished_ms = read_optional_i64(value, "finished_ms");

    if (const auto * request = value.find("request")) {
        job.request = *request;
    }
    if (const auto * metadata = value.find("metadata")) {
        job.metadata = *metadata;
    }
    job.cancel_requested = engine::io::json::optional_bool(value, "cancel_requested", false);
    job.pinned = engine::io::json::optional_bool(value, "pinned", false);
    job.last_event_seq = engine::io::json::optional_i64(value, "last_event_seq", 0);
    job.server_run_id = engine::io::json::optional_string(value, "server_run_id", "");
    const auto * key = value.find("idempotency_key");
    if (key != nullptr && key->is_string()) {
        job.idempotency_key = key->as_string();
    }

    const auto * options = value.find("resolved_options");
    if (options != nullptr && options->is_object()) {
        for (const auto & entry : options->as_object()) {
            if (entry.second.is_string()) {
                job.resolved_options[entry.first] = entry.second.as_string();
            }
        }
    }

    const auto * progress = value.find("progress");
    if (progress != nullptr && progress->is_object()) {
        job.progress.stage = engine::io::json::optional_string(*progress, "stage", "");
        job.progress.step = engine::io::json::optional_i64(*progress, "step", 0);
        job.progress.total = engine::io::json::optional_i64(*progress, "total", 0);
        job.progress.fraction = read_optional_double(*progress, "fraction").value_or(0.0);
        job.progress.overall = read_optional_double(*progress, "overall").value_or(0.0);
        job.progress.message = engine::io::json::optional_string(*progress, "message", "");
        job.progress.updated_ms = engine::io::json::optional_i64(*progress, "updated_ms", 0);
    }

    const auto * error = value.find("error");
    if (error != nullptr && error->is_object()) {
        job.error.message = engine::io::json::optional_string(*error, "message", "");
        job.error.type = engine::io::json::optional_string(*error, "type", "");
    }

    const auto * metrics = value.find("metrics");
    if (metrics != nullptr && metrics->is_object()) {
        job.metrics.wall_ms = read_optional_double(*metrics, "wall_ms");
        job.metrics.queue_ms = read_optional_double(*metrics, "queue_ms");
        job.metrics.audio_seconds = read_optional_double(*metrics, "audio_seconds");
        job.metrics.rtf = read_optional_double(*metrics, "rtf");
    }

    const auto * outputs = value.find("outputs");
    if (outputs != nullptr && outputs->is_array()) {
        for (const auto & entry : outputs->as_array()) {
            if (!entry.is_object()) {
                continue;
            }
            JobOutput output;
            output.name = engine::io::json::optional_string(entry, "name", "");
            output.kind = engine::io::json::optional_string(entry, "kind", "");
            output.content_type = engine::io::json::optional_string(entry, "content_type", "");
            output.filename = engine::io::json::optional_string(entry, "filename", "");
            output.bytes = static_cast<uint64_t>(engine::io::json::optional_i64(entry, "bytes", 0));
            output.asset_id = engine::io::json::optional_string(entry, "asset_id", "");
            output.missing = engine::io::json::optional_bool(entry, "missing", false);
            output.sample_rate = read_optional_int(entry, "sample_rate");
            output.channels = read_optional_int(entry, "channels");
            output.duration_seconds = read_optional_double(entry, "duration_seconds");
            output.count = read_optional_i64(entry, "count");
            job.outputs.push_back(std::move(output));
        }
    }

    return job;
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_JOB
// Direct-execution self-test. Never fires on import: this block only compiles
// when this file is built as its own target with -DAUDIOCPP_SELFTEST_JOB, so a
// sibling linked in as a dependency never contributes a second main().
#include <cstdio>
#include <functional>
#include <set>

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

// Runs a case, recording an unexpected throw as a failure rather than letting it
// abort the remaining cases (protocol rule 10).
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

}  // namespace

int main() {
    using namespace audiocpp::api;
    using engine::io::json::Value;

    // ---- status vocabulary ----
    guarded("status_name", [] {
        const std::string names =
            std::string(status_name(JobStatus::Queued)) + "," + status_name(JobStatus::Running) + "," +
            status_name(JobStatus::Succeeded) + "," + status_name(JobStatus::Failed) + "," +
            status_name(JobStatus::Cancelled) + "," + status_name(JobStatus::Interrupted);
        record("status_name", "all six", names,
               "queued,running,succeeded,failed,cancelled,interrupted");
    });

    guarded("parse_status", [] {
        bool ok = true;
        for (auto status : {JobStatus::Queued, JobStatus::Running, JobStatus::Succeeded,
                            JobStatus::Failed, JobStatus::Cancelled, JobStatus::Interrupted}) {
            const auto parsed = parse_status(status_name(status));
            ok = ok && parsed.has_value() && *parsed == status;
        }
        record("parse_status", "round-trip all six", ok ? "round-trips" : "lost a value", "round-trips");
        record("parse_status/unknown", "\"bogus\"",
               parse_status("bogus").has_value() ? "accepted" : "rejected", "rejected");
    });

    guarded("is_terminal", [] {
        const std::string got =
            std::string(is_terminal(JobStatus::Queued) ? "T" : "F") +
            (is_terminal(JobStatus::Running) ? "T" : "F") +
            (is_terminal(JobStatus::Succeeded) ? "T" : "F") +
            (is_terminal(JobStatus::Failed) ? "T" : "F") +
            (is_terminal(JobStatus::Cancelled) ? "T" : "F") +
            (is_terminal(JobStatus::Interrupted) ? "T" : "F");
        record("is_terminal", "queued,running,succeeded,failed,cancelled,interrupted", got, "FFTTTT");
    });

    // ---- id generation ----
    guarded("make_ulid", [] {
        const auto id = make_ulid(1757600000000LL);
        record("make_ulid/length", "1757600000000", std::to_string(id.size()), "26");
        bool crockford = true;
        for (char c : id) {
            crockford = crockford && std::strchr("0123456789ABCDEFGHJKMNPQRSTVWXYZ", c) != nullptr;
        }
        record("make_ulid/alphabet", id, crockford ? "crockford" : "out of alphabet", "crockford");
    });

    guarded("make_ulid/sortable", [] {
        // The whole reason for ULIDs: later timestamp must sort strictly after.
        const auto early = make_ulid(1000000000000LL);
        const auto late = make_ulid(1000000001000LL);
        record("make_ulid/sortable", "t=1000000000000 vs t=1000000001000",
               early < late ? "ordered" : "unordered", "ordered");
    });

    guarded("make_job_id/unique", [] {
        std::set<std::string> seen;
        for (int i = 0; i < 1000; ++i) {
            seen.insert(make_job_id(1757600000000LL));
        }
        record("make_job_id/unique", "1000 ids at one timestamp", std::to_string(seen.size()), "1000");
    });

    guarded("is_valid_job_id", [] {
        const auto good = make_job_id(now_ms());
        record("is_valid_job_id/minted", good, is_valid_job_id(good) ? "valid" : "invalid", "valid");
        record("is_valid_job_id/asset", "as_...",
               is_valid_job_id(make_asset_id(now_ms())) ? "valid" : "invalid", "invalid");
        // Traversal and shape attacks must all be rejected before any path is built.
        for (const char * bad : {"", "job_", "job_../../etc/passwd", "../job_0000000000000000000000000",
                                 "job_0000000000000000000000000/", "job_IIIIIIIIIIIIIIIIIIIIIIIIII",
                                 "job_000000000000000000000000000", "JOB_0000000000000000000000000"}) {
            record("is_valid_job_id/reject", std::string("\"") + bad + "\"",
                   is_valid_job_id(bad) ? "valid" : "invalid", "invalid");
        }
    });

    guarded("is_valid_asset_id", [] {
        const auto good = make_asset_id(now_ms());
        record("is_valid_asset_id/minted", good, is_valid_asset_id(good) ? "valid" : "invalid", "valid");
    });

    // ---- JSON round-trip ----
    guarded("job_to_json/job_from_json", [] {
        Job job;
        job.id = make_job_id(1757600000000LL);
        job.model_id = "yue2-3b-q8";
        job.family = "yue2";
        job.operation = "generate";
        job.status = JobStatus::Succeeded;
        job.priority = 3;
        job.created_ms = 1757600000000LL;
        job.updated_ms = 1757600184000LL;
        job.started_ms = 1757600000180LL;
        job.finished_ms = 1757600184000LL;
        job.cancel_requested = true;
        job.pinned = true;
        job.idempotency_key = "webui-8f2a1c";
        job.last_event_seq = 42;
        job.server_run_id = "run-1";
        job.resolved_options["style"] = "dreamy synth pop";
        job.resolved_options["seed"] = "831001";
        job.progress = {"semantic", 1840, 9000, 0.2044, 0.193, "half way", 1757600100000LL};
        job.error = {"out of memory", "insufficient_memory"};
        job.metrics.wall_ms = 184320.0;
        job.metrics.queue_ms = 180.0;
        job.metrics.audio_seconds = 48.0;
        job.metrics.rtf = 3.84;

        JobOutput audio;
        audio.name = "audio";
        audio.kind = "audio";
        audio.content_type = "audio/wav";
        audio.filename = "audio.wav";
        audio.bytes = 9216044;
        audio.asset_id = make_asset_id(1757600000000LL);
        audio.sample_rate = 48000;
        audio.channels = 2;
        audio.duration_seconds = 48.0;
        job.outputs.push_back(audio);

        JobOutput codes;
        codes.name = "semantic_codes";
        codes.kind = "codes";
        codes.filename = "semantic_codes.i32";
        codes.count = 6421;
        job.outputs.push_back(codes);

        // Lyrics with a quote, a newline and non-ASCII: the exact content most
        // likely to be mangled by a hand-rolled serializer.
        Value::Object request;
        request["lyrics"] = Value::make_string("[verse]\n\"hold on\" \xC3\xA9 \xE2\x80\x94 caf\xC3\xA9\n");
        request["style"] = Value::make_string("dreamy synth pop");
        job.request = Value::make_object(std::move(request));

        Value::Object metadata;
        metadata["title"] = Value::make_string("take 3");
        job.metadata = Value::make_object(std::move(metadata));

        const auto text = engine::io::json::stringify(job_to_json(job));
        const auto back = job_from_json(engine::io::json::parse(text));

        record("job/id", job.id, back.id, job.id);
        record("job/model", job.model_id, back.model_id, job.model_id);
        record("job/family", job.family, back.family, job.family);
        record("job/operation", job.operation, back.operation, job.operation);
        record("job/status", "succeeded", status_name(back.status), "succeeded");
        record("job/priority", "3", std::to_string(back.priority), "3");
        record("job/created_ms", "1757600000000", std::to_string(back.created_ms), "1757600000000");
        record("job/started_ms", "1757600000180",
               std::to_string(back.started_ms.value_or(-1)), "1757600000180");
        record("job/cancel_requested", "true", back.cancel_requested ? "true" : "false", "true");
        record("job/pinned", "true", back.pinned ? "true" : "false", "true");
        record("job/idempotency_key", "webui-8f2a1c",
               back.idempotency_key.value_or("<none>"), "webui-8f2a1c");
        record("job/last_event_seq", "42", std::to_string(back.last_event_seq), "42");
        record("job/server_run_id", "run-1", back.server_run_id, "run-1");
        record("job/resolved_options", "2 entries",
               std::to_string(back.resolved_options.size()) + ":" + back.resolved_options.at("style"),
               "2:dreamy synth pop");
        record("job/progress.stage", "semantic", back.progress.stage, "semantic");
        record("job/progress.step", "1840", std::to_string(back.progress.step), "1840");
        record("job/error.type", "insufficient_memory", back.error.type, "insufficient_memory");
        record("job/metrics.rtf", "3.84", std::to_string(back.metrics.rtf.value_or(0.0)).substr(0, 4), "3.84");
        record("job/outputs.size", "2", std::to_string(back.outputs.size()), "2");
        record("job/outputs[0].bytes", "9216044", std::to_string(back.outputs[0].bytes), "9216044");
        record("job/outputs[0].sample_rate", "48000",
               std::to_string(back.outputs[0].sample_rate.value_or(0)), "48000");
        record("job/outputs[1].count", "6421", std::to_string(back.outputs[1].count.value_or(0)), "6421");
        record("job/outputs[1].sample_rate", "unset",
               back.outputs[1].sample_rate.has_value() ? "set" : "unset", "unset");

        // The one that actually matters for the user's complaint: the lyrics
        // must survive verbatim, quotes, newlines, accents and all.
        const auto & lyrics = back.request.require("lyrics").as_string();
        record("job/request.lyrics", "quotes + newline + non-ASCII",
               lyrics == "[verse]\n\"hold on\" \xC3\xA9 \xE2\x80\x94 caf\xC3\xA9\n" ? "verbatim" : lyrics,
               "verbatim");
        record("job/metadata.title", "take 3", back.metadata.require("title").as_string(), "take 3");
    });

    guarded("job_to_json/empty_error", [] {
        Job job;
        job.id = make_job_id(now_ms());
        job.status = JobStatus::Queued;
        const auto back = job_from_json(engine::io::json::parse(engine::io::json::stringify(job_to_json(job))));
        record("job/empty_error", "no error set", back.error.empty() ? "empty" : "populated", "empty");
        record("job/empty_metrics", "no metrics set",
               back.metrics.wall_ms.has_value() ? "set" : "unset", "unset");
        record("job/no_outputs", "no outputs", std::to_string(back.outputs.size()), "0");
    });

    // ---- rejection cases ----
    expect_throws("job_from_json/not_object", "42",
                  [] { job_from_json(engine::io::json::parse("42")); });
    expect_throws("job_from_json/bad_id", "id=\"../etc\"",
                  [] { job_from_json(engine::io::json::parse("{\"id\":\"../etc\",\"status\":\"queued\"}")); });
    expect_throws("job_from_json/unknown_status", "status=\"melting\"", [] {
        const auto id = make_job_id(now_ms());
        job_from_json(engine::io::json::parse("{\"id\":\"" + id + "\",\"status\":\"melting\"}"));
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_JOB
