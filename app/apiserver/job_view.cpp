#include "job_view.h"

#include "api_error.h"

#include <algorithm>

namespace audiocpp::api {
namespace {

using engine::io::json::Value;

Value::Object output_object(const Job & job, const JobOutput & output) {
    Value::Object entry;
    entry["name"] = Value::make_string(output.name);
    entry["kind"] = Value::make_string(output.kind);
    entry["content_type"] = Value::make_string(content_type_for(output));
    entry["bytes"] = Value::make_number(static_cast<double>(output.bytes));
    entry["url"] = Value::make_string("/v1/jobs/" + job.id + "/outputs/" + output.name);
    if (!output.asset_id.empty()) {
        entry["asset_id"] = Value::make_string(output.asset_id);
    }
    if (output.sample_rate.has_value()) {
        entry["sample_rate"] = Value::make_number(*output.sample_rate);
    }
    if (output.channels.has_value()) {
        entry["channels"] = Value::make_number(*output.channels);
    }
    if (output.duration_seconds.has_value()) {
        entry["duration_seconds"] = Value::make_number(*output.duration_seconds);
    }
    if (output.count.has_value()) {
        entry["count"] = Value::make_number(static_cast<double>(*output.count));
    }
    if (output.missing) {
        // The record survived but the bytes did not. Saying so is better than a
        // download that 404s with no explanation.
        entry["missing"] = Value::make_bool(true);
    }
    return entry;
}

int64_t parse_i64_param(const std::unordered_map<std::string, std::string> & params,
                        const char * key,
                        int64_t fallback) {
    const auto found = params.find(key);
    if (found == params.end() || found->second.empty()) {
        return fallback;
    }
    try {
        return std::stoll(found->second);
    } catch (const std::exception &) {
        throw bad_request(std::string("query parameter '") + key + "' must be an integer, got '" +
                          found->second + "'");
    }
}

std::string param_or(const std::unordered_map<std::string, std::string> & params,
                     const char * key,
                     const char * fallback = "") {
    const auto found = params.find(key);
    return found == params.end() ? std::string(fallback) : found->second;
}

}  // namespace

std::string content_type_for(const JobOutput & output) {
    if (!output.content_type.empty()) {
        return output.content_type;
    }
    if (output.kind == "audio") {
        return "audio/wav";
    }
    if (output.kind == "score") {
        return "text/vnd.abc; charset=utf-8";
    }
    if (output.kind == "text") {
        return "text/plain; charset=utf-8";
    }
    if (output.kind == "json") {
        return "application/json";
    }
    if (output.kind == "midi") {
        return "audio/midi";
    }
    return "application/octet-stream";
}

std::string download_filename(const Job & job, const JobOutput & output) {
    return job.id + "_" + output.filename;
}

Value job_envelope(const Job & job, const JobViewOptions & options) {
    Value::Object root;
    root["id"] = Value::make_string(job.id);
    root["status"] = Value::make_string(status_name(job.status));
    root["model"] = Value::make_string(job.model_id);
    root["family"] = Value::make_string(job.family);
    root["operation"] = Value::make_string(job.operation);
    root["priority"] = Value::make_number(job.priority);
    root["created_ms"] = Value::make_number(static_cast<double>(job.created_ms));
    root["updated_ms"] = Value::make_number(static_cast<double>(job.updated_ms));
    root["started_ms"] = job.started_ms.has_value()
                             ? Value::make_number(static_cast<double>(*job.started_ms))
                             : Value::make_null();
    root["finished_ms"] = job.finished_ms.has_value()
                              ? Value::make_number(static_cast<double>(*job.finished_ms))
                              : Value::make_null();
    root["queue_position"] = options.queue_position >= 0
                                 ? Value::make_number(options.queue_position)
                                 : Value::make_null();
    root["pinned"] = Value::make_bool(job.pinned);
    root["cancel_requested"] = Value::make_bool(job.cancel_requested);
    root["terminal"] = Value::make_bool(is_terminal(job.status));

    Value::Object progress;
    progress["stage"] = Value::make_string(job.progress.stage);
    progress["step"] = Value::make_number(static_cast<double>(job.progress.step));
    progress["total"] = Value::make_number(static_cast<double>(job.progress.total));
    progress["fraction"] = Value::make_number(job.progress.fraction);
    if (!job.progress.message.empty()) {
        progress["message"] = Value::make_string(job.progress.message);
    }
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
    if (job.metrics.wall_ms.has_value()) {
        metrics["wall_ms"] = Value::make_number(*job.metrics.wall_ms);
    }
    if (job.metrics.queue_ms.has_value()) {
        metrics["queue_ms"] = Value::make_number(*job.metrics.queue_ms);
    }
    if (job.metrics.audio_seconds.has_value()) {
        metrics["audio_seconds"] = Value::make_number(*job.metrics.audio_seconds);
    }
    if (job.metrics.rtf.has_value()) {
        metrics["rtf"] = Value::make_number(*job.metrics.rtf);
    }
    root["metrics"] = Value::make_object(std::move(metrics));

    Value::Array outputs;
    outputs.reserve(job.outputs.size());
    for (const auto & output : job.outputs) {
        outputs.push_back(Value::make_object(output_object(job, output)));
    }
    root["outputs"] = Value::make_array(std::move(outputs));

    root["metadata"] = job.metadata;
    if (options.include_request) {
        root["request"] = job.request;
    }

    Value::Object links;
    links["self"] = Value::make_string("/v1/jobs/" + job.id);
    links["events"] = Value::make_string("/v1/jobs/" + job.id + "/events");
    links["result"] = Value::make_string("/v1/jobs/" + job.id + "/result");
    links["cancel"] = Value::make_string("/v1/jobs/" + job.id + "/cancel");
    root["links"] = Value::make_object(std::move(links));

    return Value::make_object(std::move(root));
}

Value job_list_envelope(const JobPage & page, bool include_request) {
    JobViewOptions options;
    options.include_request = include_request;

    Value::Array jobs;
    jobs.reserve(page.jobs.size());
    for (const auto & job : page.jobs) {
        jobs.push_back(job_envelope(job, options));
    }

    Value::Object root;
    root["object"] = Value::make_string("list");
    root["data"] = Value::make_array(std::move(jobs));
    root["total_matched"] = Value::make_number(static_cast<double>(page.total_matched));
    root["next_cursor"] =
        page.next_cursor.empty() ? Value::make_null() : Value::make_string(page.next_cursor);
    return Value::make_object(std::move(root));
}

Value job_result_envelope(const Job & job) {
    auto envelope = job_envelope(job, JobViewOptions{});
    return envelope;
}

JobQuery parse_job_query(const std::unordered_map<std::string, std::string> & params) {
    JobQuery query;

    const auto statuses = param_or(params, "status");
    if (!statuses.empty()) {
        size_t start = 0;
        while (start <= statuses.size()) {
            auto end = statuses.find(',', start);
            if (end == std::string::npos) {
                end = statuses.size();
            }
            const auto name = statuses.substr(start, end - start);
            if (!name.empty()) {
                const auto parsed = parse_status(name);
                if (!parsed.has_value()) {
                    throw bad_request("unknown status '" + name +
                                      "'; expected queued, running, succeeded, failed, cancelled or "
                                      "interrupted");
                }
                query.statuses.push_back(*parsed);
            }
            if (end == statuses.size()) {
                break;
            }
            start = end + 1;
        }
    }

    query.model = param_or(params, "model");
    query.family = param_or(params, "family");
    query.operation = param_or(params, "operation");
    query.text = param_or(params, "q");
    query.cursor = param_or(params, "cursor");

    const auto pinned = param_or(params, "pinned");
    if (!pinned.empty()) {
        query.pinned = pinned == "1" || pinned == "true";
    }

    if (params.count("created_after")) {
        query.created_after_ms = parse_i64_param(params, "created_after", 0);
    }
    if (params.count("created_before")) {
        query.created_before_ms = parse_i64_param(params, "created_before", 0);
    }

    const auto limit = parse_i64_param(params, "limit", 50);
    if (limit < 0) {
        throw bad_request("query parameter 'limit' must not be negative");
    }
    // A cap, not a suggestion: an unbounded list over a large store would build a
    // response big enough to matter.
    query.limit = static_cast<size_t>(std::min<int64_t>(limit, 200));

    const auto order = param_or(params, "order", "desc");
    if (order != "desc" && order != "asc") {
        throw bad_request("query parameter 'order' must be 'asc' or 'desc', got '" + order + "'");
    }
    query.newest_first = order == "desc";

    return query;
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_JOB_VIEW
#include <cstdio>
#include <functional>

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

audiocpp::api::Job sample() {
    using namespace audiocpp::api;
    using engine::io::json::Value;
    Job job;
    job.id = "job_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    job.model_id = "yue2-3b-q8";
    job.family = "yue2";
    job.operation = "generate";
    job.status = JobStatus::Succeeded;
    job.created_ms = 1757600000000LL;
    job.updated_ms = 1757600184000LL;
    job.started_ms = 1757600000180LL;
    job.finished_ms = 1757600184000LL;
    job.metrics.wall_ms = 184320.0;
    job.metrics.audio_seconds = 48.0;

    JobOutput audio;
    audio.name = "audio";
    audio.kind = "audio";
    audio.filename = "audio.wav";
    audio.bytes = 9216044;
    audio.asset_id = "as_01ARZ3NDEKTSV4RRFFQ69G5FAW";
    audio.sample_rate = 48000;
    audio.channels = 2;
    audio.duration_seconds = 48.0;
    job.outputs.push_back(audio);

    Value::Object request;
    request["style"] = Value::make_string("dreamy synth pop");
    job.request = Value::make_object(std::move(request));
    return job;
}

}  // namespace

int main() {
    using namespace audiocpp::api;
    using engine::io::json::Value;

    guarded("job_envelope", [] {
        const auto job = sample();
        const auto envelope = job_envelope(job);
        record("envelope/id", "sample job", envelope.require("id").as_string(),
               "job_01ARZ3NDEKTSV4RRFFQ69G5FAV");
        record("envelope/status", "succeeded", envelope.require("status").as_string(), "succeeded");
        record("envelope/terminal", "succeeded", envelope.require("terminal").as_bool() ? "true" : "false",
               "true");
        record("envelope/queue_position_null", "not queued",
               envelope.require("queue_position").is_null() ? "null" : "set", "null");
        record("envelope/error_null", "no error",
               envelope.require("error").is_null() ? "null" : "set", "null");
        record("envelope/includes_request", "default options",
               envelope.find("request") != nullptr ? "present" : "absent", "present");
        record("envelope/links_self", "sample job",
               envelope.require("links").require("self").as_string(),
               "/v1/jobs/job_01ARZ3NDEKTSV4RRFFQ69G5FAV");
        record("envelope/links_events", "sample job",
               envelope.require("links").require("events").as_string(),
               "/v1/jobs/job_01ARZ3NDEKTSV4RRFFQ69G5FAV/events");
    });

    guarded("job_envelope/outputs", [] {
        const auto job = sample();
        const auto envelope = job_envelope(job);
        const auto & outputs = envelope.require("outputs").as_array();
        record("envelope/output_count", "one output", std::to_string(outputs.size()), "1");
        const auto & audio = outputs.front();
        record("envelope/output_url", "audio", audio.require("url").as_string(),
               "/v1/jobs/job_01ARZ3NDEKTSV4RRFFQ69G5FAV/outputs/audio");
        record("envelope/output_content_type", "audio kind", audio.require("content_type").as_string(),
               "audio/wav");
        record("envelope/output_sample_rate", "48000",
               std::to_string(audio.require("sample_rate").as_i64()), "48000");
        // The size that must never be inlined.
        record("envelope/no_inline_audio", "9 MB of WAV",
               audio.find("data") == nullptr && audio.find("audio_base64") == nullptr ? "absent" : "inlined",
               "absent");
    });

    guarded("job_envelope/omits_request", [] {
        JobViewOptions options;
        options.include_request = false;
        const auto envelope = job_envelope(sample(), options);
        record("envelope/omits_request", "include_request=false",
               envelope.find("request") == nullptr ? "absent" : "present", "absent");
    });

    guarded("job_envelope/queue_position", [] {
        JobViewOptions options;
        options.queue_position = 3;
        const auto envelope = job_envelope(sample(), options);
        record("envelope/queue_position", "position 3",
               std::to_string(envelope.require("queue_position").as_i64()), "3");
    });

    guarded("job_envelope/error", [] {
        auto job = sample();
        job.status = JobStatus::Failed;
        job.error = {"out of memory", "insufficient_memory"};
        const auto envelope = job_envelope(job);
        record("envelope/error_message", "failed job",
               envelope.require("error").require("message").as_string(), "out of memory");
        record("envelope/error_type", "failed job",
               envelope.require("error").require("type").as_string(), "insufficient_memory");
    });

    guarded("job_envelope/missing_output", [] {
        auto job = sample();
        job.outputs.front().missing = true;
        const auto envelope = job_envelope(job);
        record("envelope/missing_flagged", "output gone from disk",
               envelope.require("outputs").as_array().front().require("missing").as_bool() ? "flagged"
                                                                                           : "silent",
               "flagged");
    });

    guarded("job_list_envelope", [] {
        JobPage page;
        page.jobs = {sample(), sample()};
        page.total_matched = 7;
        page.next_cursor = "job_01ARZ3NDEKTSV4RRFFQ69G5FAV";
        const auto envelope = job_list_envelope(page, false);
        record("list/object", "page", envelope.require("object").as_string(), "list");
        record("list/count", "two jobs", std::to_string(envelope.require("data").as_array().size()), "2");
        record("list/total_matched", "7", std::to_string(envelope.require("total_matched").as_i64()), "7");
        record("list/next_cursor", "more remain", envelope.require("next_cursor").as_string(),
               "job_01ARZ3NDEKTSV4RRFFQ69G5FAV");
        record("list/omits_request", "list view",
               envelope.require("data").as_array().front().find("request") == nullptr ? "absent" : "present",
               "absent");

        JobPage last;
        last.jobs = {sample()};
        record("list/null_cursor_at_end", "no more pages",
               job_list_envelope(last, false).require("next_cursor").is_null() ? "null" : "set", "null");
    });

    guarded("content_type_for", [] {
        JobOutput output;
        output.kind = "audio";
        record("content_type/audio", "kind=audio", content_type_for(output), "audio/wav");
        output.kind = "score";
        record("content_type/score", "kind=score", content_type_for(output),
               "text/vnd.abc; charset=utf-8");
        output.kind = "codes";
        record("content_type/codes", "kind=codes", content_type_for(output), "application/octet-stream");
        output.kind = "midi";
        record("content_type/midi", "kind=midi", content_type_for(output), "audio/midi");
        output.content_type = "audio/mpeg";
        record("content_type/explicit_wins", "explicit content_type", content_type_for(output), "audio/mpeg");
    });

    guarded("download_filename", [] {
        const auto job = sample();
        record("download_filename", "audio.wav on a job", download_filename(job, job.outputs.front()),
               "job_01ARZ3NDEKTSV4RRFFQ69G5FAV_audio.wav");
    });

    // ---- query parsing ----
    guarded("parse_job_query/defaults", [] {
        const auto query = parse_job_query({});
        record("query/default_limit", "no params", std::to_string(query.limit), "50");
        record("query/default_order", "no params", query.newest_first ? "desc" : "asc", "desc");
        record("query/no_statuses", "no params", std::to_string(query.statuses.size()), "0");
    });

    guarded("parse_job_query/fields", [] {
        const auto query = parse_job_query({{"status", "succeeded,failed"},
                                            {"model", "yue2-3b-q8"},
                                            {"family", "yue2"},
                                            {"operation", "generate"},
                                            {"q", "neon"},
                                            {"cursor", "job_01X"},
                                            {"pinned", "true"},
                                            {"limit", "10"},
                                            {"order", "asc"},
                                            {"created_after", "1757600000000"}});
        record("query/statuses", "succeeded,failed", std::to_string(query.statuses.size()), "2");
        record("query/model", "yue2-3b-q8", query.model, "yue2-3b-q8");
        record("query/text", "neon", query.text, "neon");
        record("query/cursor", "job_01X", query.cursor, "job_01X");
        record("query/pinned", "true", query.pinned.value_or(false) ? "true" : "false", "true");
        record("query/limit", "10", std::to_string(query.limit), "10");
        record("query/order", "asc", query.newest_first ? "desc" : "asc", "asc");
        record("query/created_after", "1757600000000",
               std::to_string(query.created_after_ms.value_or(0)), "1757600000000");
    });

    guarded("parse_job_query/limit_cap", [] {
        record("query/limit_capped", "limit=9999",
               std::to_string(parse_job_query({{"limit", "9999"}}).limit), "200");
        record("query/limit_zero", "limit=0", std::to_string(parse_job_query({{"limit", "0"}}).limit), "0");
    });

    guarded("parse_job_query/rejects", [] {
        expect_throws("query/bad_status", "status=melting",
                      [] { parse_job_query({{"status", "melting"}}); });
        expect_throws("query/bad_limit", "limit=abc", [] { parse_job_query({{"limit", "abc"}}); });
        expect_throws("query/negative_limit", "limit=-1", [] { parse_job_query({{"limit", "-1"}}); });
        expect_throws("query/bad_order", "order=sideways", [] { parse_job_query({{"order", "sideways"}}); });
    });

    guarded("parse_job_query/error_names_field", [] {
        std::string message;
        try {
            parse_job_query({{"status", "melting"}});
        } catch (const std::exception & ex) {
            message = ex.what();
        }
        record("query/error_names_value", "status=melting",
               message.find("melting") != std::string::npos ? "names it" : message, "names it");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_JOB_VIEW
