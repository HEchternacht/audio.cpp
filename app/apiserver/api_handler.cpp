#include "api_handler.h"

#include "api_error.h"
#include "job_view.h"
#include "result_json.h"
#include "wav.h"

#include "request.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace audiocpp::api {
namespace {

using engine::io::json::Value;
using minitts::server::HttpRequest;
using minitts::server::HttpResponse;

std::string json_body(const Value & value, int status = 200) {
    return engine::io::json::stringify(value);
}

HttpResponse json_ok(const Value & value, int status = 200) {
    return minitts::server::json_response(engine::io::json::stringify(value), status);
}

Value parse_body(const std::string & text) {
    if (text.empty()) {
        throw bad_request("a JSON request body is required");
    }
    try {
        return engine::io::json::parse(text);
    } catch (const std::exception & ex) {
        throw bad_request(std::string("request body is not valid JSON: ") + ex.what());
    }
}

std::string read_file_bytes(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw not_found("output file is missing from the store: " + path.filename().string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

}  // namespace

ApiHandler::ApiHandler(ApiConfig config, std::filesystem::path config_dir)
    : config_(std::move(config)),
      config_dir_(std::move(config_dir)),
      server_run_id_(make_ulid(now_ms())),
      started_ms_(now_ms()),
      store_(config_.data_dir, server_run_id_),
      queue_(config_.max_queued),
      pool_(config_.base),
      events_(512, config_.max_event_subscribers) {
    register_routes();
}

ApiHandler::~ApiHandler() {
    stop();
}

void ApiHandler::start() {
    const auto report = store_.open();
    std::cerr << "[api] store " << config_.data_dir.string() << ": " << report.loaded << " jobs loaded";
    if (report.interrupted > 0) {
        std::cerr << ", " << report.interrupted << " marked interrupted";
    }
    if (report.quarantined > 0) {
        std::cerr << ", " << report.quarantined << " quarantined";
    }
    if (report.removed > 0) {
        std::cerr << ", " << report.removed << " empty removed";
    }
    std::cerr << "\n";

    pool_.register_models();

    workers_ = std::make_unique<WorkerPool>(
        static_cast<size_t>(config_.workers),
        queue_,
        store_,
        [this](Job & job, const std::shared_ptr<engine::runtime::RunControl> & control) {
            run_job(job, control);
        },
        WorkerHooks{
            [this](const Job & job) { publish_status(job); },
            [this](const Job & job, const engine::runtime::ProgressEvent & event) {
                publish_progress(job, event);
            },
        },
        config_.progress_interval_ms);
    workers_->start();
    running_.store(true);

    // Jobs that were queued when the last process died: the request is intact on
    // disk, so honour it rather than making the user resubmit.
    if (config_.requeue_on_start && !report.requeued.empty()) {
        const int64_t cutoff =
            config_.requeue_max_age_hours > 0
                ? now_ms() - static_cast<int64_t>(config_.requeue_max_age_hours) * 3600000LL
                : 0;
        int restored = 0;
        int expired = 0;
        for (const auto & id : report.requeued) {
            auto job = store_.load(id);
            if (!job.has_value()) {
                continue;
            }
            if (cutoff > 0 && job->created_ms < cutoff) {
                // Nobody wants a week-old queue replaying after a reboot.
                job->status = JobStatus::Interrupted;
                job->finished_ms = now_ms();
                job->error.message = "queued before the server restarted, and too old to resume";
                job->error.type = "interrupted";
                store_.save(*job);
                ++expired;
                continue;
            }
            job->server_run_id = server_run_id_;
            store_.save(*job);
            queue_.push(QueueEntry{job->id, job->model_id, job->priority, job->created_ms});
            ++restored;
        }
        std::cerr << "[api] restored " << restored << " queued jobs";
        if (expired > 0) {
            std::cerr << ", expired " << expired << " older than " << config_.requeue_max_age_hours << "h";
        }
        std::cerr << "\n";
    }

    if (config_.retention_days > 0 || config_.max_jobs_retained > 0 || config_.max_bytes_retained > 0) {
        RetentionPolicy policy;
        policy.max_age_ms = static_cast<int64_t>(config_.retention_days) * 86400000LL;
        policy.max_jobs = config_.max_jobs_retained;
        policy.max_bytes = config_.max_bytes_retained;
        const int collected = store_.collect(policy, now_ms());
        if (collected > 0) {
            std::cerr << "[api] gc: removed " << collected << " jobs, " << store_.count() << " retained\n";
        }
    }
}

void ApiHandler::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    events_.shutdown();
    if (workers_) {
        // Graceful: cancel what is in flight so those jobs end up `cancelled`
        // rather than being found `interrupted` on the next boot.
        workers_->stop(true);
    }
}

void ApiHandler::publish_status(const Job & job) {
    Value::Object payload;
    payload["job"] = Value::make_string(job.id);
    payload["type"] = Value::make_string("status");
    payload["status"] = Value::make_string(status_name(job.status));
    if (is_terminal(job.status)) {
        payload["terminal"] = Value::make_bool(true);
    }
    const auto text = engine::io::json::stringify(Value::make_object(std::move(payload)));
    events_.publish(job.id, "status", text, now_ms());
    store_.append_event(job.id, text);
    if (is_terminal(job.status)) {
        Value::Object done;
        done["job"] = Value::make_string(job.id);
        done["type"] = Value::make_string("done");
        done["status"] = Value::make_string(status_name(job.status));
        done["result_url"] = Value::make_string("/v1/jobs/" + job.id + "/result");
        const auto done_text = engine::io::json::stringify(Value::make_object(std::move(done)));
        events_.publish(job.id, "done", done_text, now_ms());
        store_.append_event(job.id, done_text);
        events_.close(job.id);
    }
}

void ApiHandler::publish_progress(const Job & job, const engine::runtime::ProgressEvent & event) {
    Value::Object payload;
    payload["job"] = Value::make_string(job.id);
    payload["type"] = Value::make_string("progress");
    payload["stage"] = Value::make_string(event.stage);
    payload["step"] = Value::make_number(static_cast<double>(event.step));
    payload["total"] = Value::make_number(static_cast<double>(event.total));
    payload["fraction"] = Value::make_number(job.progress.fraction);
    if (!event.message.empty()) {
        payload["message"] = Value::make_string(event.message);
    }
    events_.publish(job.id, "progress", engine::io::json::stringify(Value::make_object(std::move(payload))),
                    now_ms());
}

void ApiHandler::run_job(Job & job, const std::shared_ptr<engine::runtime::RunControl> & control) {
    // The request body is stored verbatim; rebuild the engine request from it so
    // a replay produces exactly what the original run did.
    const auto * request_json = job.request.find("request");
    const auto & effective = request_json != nullptr ? *request_json : job.request;

    // Client strings never reach resolve_case_path: the base directory is the
    // job's own input folder, which nothing outside the server can write to.
    const auto staging = store_.job_dir(job.id) / "input";
    auto task_request = minitts::cli::build_request_from_json(effective, staging);

    for (const auto & entry : task_request.options) {
        job.resolved_options[entry.first] = entry.second;
    }

    const auto result = pool_.run(job.model_id, task_request, control);

    if (result.audio_output.has_value()) {
        const auto wav = minitts::server::encode_pcm16_wav(*result.audio_output);
        JobOutput output;
        output.name = "audio";
        output.kind = "audio";
        output.content_type = "audio/wav";
        output.filename = "audio.wav";
        output.sample_rate = result.audio_output->sample_rate;
        output.channels = result.audio_output->channels;
        if (result.audio_output->sample_rate > 0 && result.audio_output->channels > 0) {
            output.duration_seconds =
                static_cast<double>(result.audio_output->samples.size()) /
                (static_cast<double>(result.audio_output->sample_rate) * result.audio_output->channels);
            job.metrics.audio_seconds = output.duration_seconds;
        }
        store_.write_output(job, output, wav.data(), wav.size());
    }

    // yue2 returns its generated ABC score here, which is what makes the
    // edit-and-re-render round trip possible.
    if (result.text_output.has_value() && !result.text_output->text.empty()) {
        const auto & text = result.text_output->text;
        const bool looks_like_score = text.rfind("X:", 0) == 0;
        JobOutput output;
        output.name = looks_like_score ? "score" : "text";
        output.kind = looks_like_score ? "score" : "text";
        output.filename = looks_like_score ? "score.abc" : "text.txt";
        store_.write_output(job, output, text.data(), text.size());
    }

    for (const auto & artifact : result.output_artifacts) {
        JobOutput output;
        output.name = artifact.id.empty() ? "artifact" : artifact.id;
        output.kind = "codes";
        output.filename = output.name + ".bin";
        const auto it = artifact.meta.find("count");
        if (it != artifact.meta.end()) {
            try {
                output.count = std::stoll(it->second);
            } catch (const std::exception &) {
                // A non-numeric count is metadata we simply do not surface.
            }
        }
        store_.write_output(job, output, artifact.payload.data(), artifact.payload.size());
    }

    // A JSON view of everything else the model returned (segments, word
    // timings, speaker turns), so nothing is lost when the API has no
    // first-class field for it. The audio is deliberately dropped first:
    // task_result_json base64s it, which for a three-minute track is tens of
    // megabytes of duplicate stored beside the .wav we just wrote.
    auto detail = result;
    detail.audio_output.reset();
    detail.named_audio_outputs.clear();
    const auto summary = minitts::server::task_result_json(detail, 0.0);
    JobOutput result_output;
    result_output.name = "result";
    result_output.kind = "json";
    result_output.content_type = "application/json";
    result_output.filename = "result.json";
    store_.write_output(job, result_output, summary.data(), summary.size());

    if (job.metrics.audio_seconds.has_value() && job.started_ms.has_value()) {
        const auto wall = static_cast<double>(now_ms() - *job.started_ms);
        if (*job.metrics.audio_seconds > 0.0) {
            job.metrics.rtf = wall / (*job.metrics.audio_seconds * 1000.0);
        }
    }

    // A cancelled run may leave a session with a partially advanced KV cache and
    // graph arenas still resident, because the unwind skips the cleanup the
    // success path does. Unloading costs one reload and buys certainty.
    if (control && control->cancelled() && config_.unload_after_cancel) {
        pool_.unload(job.model_id);
    }
}

Job ApiHandler::require_job(const std::string & id) const {
    if (!is_valid_job_id(id)) {
        throw not_found("no such job: " + id);
    }
    auto job = store_.load(id);
    if (!job.has_value()) {
        throw not_found("no such job: " + id);
    }
    return *job;
}

// ---------------------------------------------------------------- routes ----

HttpResponse ApiHandler::handle_health(const HttpRequest &, const PathParams &) {
    Value::Object root;
    root["status"] = Value::make_string("ok");
    root["version"] = Value::make_string("dev");
    root["backend"] = Value::make_string(config_.base.backend == engine::core::BackendType::Cuda ? "cuda" : "cpu");
    root["device"] = Value::make_number(config_.base.device);
    root["models"] = Value::make_number(static_cast<double>(pool_.summaries().size()));
    root["models_loaded"] = Value::make_number(pool_.loaded_count());
    root["workers"] = Value::make_number(config_.workers);
    root["max_loaded_models"] = Value::make_number(config_.base.max_loaded_models);
    root["uptime_ms"] = Value::make_number(static_cast<double>(now_ms() - started_ms_));
    root["data_dir"] = Value::make_string(config_.data_dir.string());

    Value::Object jobs;
    jobs["queued"] = Value::make_number(static_cast<double>(queue_.size()));
    jobs["running"] = Value::make_number(
        static_cast<double>(workers_ ? workers_->running_ids().size() : 0));
    jobs["retained"] = Value::make_number(static_cast<double>(store_.count()));
    root["jobs"] = Value::make_object(std::move(jobs));

    return json_ok(Value::make_object(std::move(root)));
}

HttpResponse ApiHandler::handle_capabilities(const HttpRequest &, const PathParams &) {
    Value::Object root;
    root["api_version"] = Value::make_number(1);
    root["sse"] = Value::make_bool(true);

    Value::Object server;
    Value::Object concurrency;
    concurrency["workers"] = Value::make_number(config_.workers);
    concurrency["max_loaded_models"] = Value::make_number(config_.base.max_loaded_models);
    concurrency["max_queued"] = Value::make_number(static_cast<double>(config_.max_queued));
    // Two jobs for one model always serialise, whatever the worker count: a model
    // has one session, and that session holds mutable graph state.
    concurrency["note"] = Value::make_string(
        "Jobs for the same model always run one at a time; extra workers only help across "
        "different models, and are bounded by GPU memory.");
    server["concurrency"] = Value::make_object(std::move(concurrency));

    Value::Object cancellation;
    cancellation["supported"] = Value::make_bool(true);
    cancellation["granularity"] = Value::make_string("model_step");
    cancellation["note"] = Value::make_string(
        "A cancel takes effect at the next step boundary. During token decoding that is tens of "
        "milliseconds; during flow-matching or decoding it can be one full step.");
    server["cancellation"] = Value::make_object(std::move(cancellation));

    Value::Object persistence;
    persistence["enabled"] = Value::make_bool(true);
    persistence["retention_days"] = Value::make_number(config_.retention_days);
    persistence["max_jobs_retained"] = Value::make_number(static_cast<double>(config_.max_jobs_retained));
    server["persistence"] = Value::make_object(std::move(persistence));
    root["server"] = Value::make_object(std::move(server));

    Value::Array models;
    for (const auto & summary : pool_.summaries()) {
        Value::Object entry;
        entry["id"] = Value::make_string(summary.id);
        entry["family"] = Value::make_string(summary.family);
        entry["task"] = Value::make_string(summary.task);
        entry["mode"] = Value::make_string(summary.mode);
        entry["loaded"] = Value::make_bool(summary.loaded);

        // Per-family truths, so a client never assumes a capability one family
        // has is available on another.
        if (summary.family == "yue2") {
            entry["accepts_audio_input"] = Value::make_bool(false);
            entry["notes"] = Value::make_string(
                "yue2 has no audio-conditioning path. Covers and edits go through an ABC score, not "
                "audio. Region repaint that rejoins the surrounding audio is not achievable.");
            Value::Array stages;
            for (const char * stage : {"score", "semantic", "acoustic", "decode"}) {
                stages.push_back(Value::make_string(stage));
            }
            entry["progress_stages"] = Value::make_array(std::move(stages));
        } else if (summary.family == "ace_step" || summary.family == "stable_audio") {
            entry["accepts_audio_input"] = Value::make_bool(true);
        }
        models.push_back(Value::make_object(std::move(entry)));
    }
    root["models"] = Value::make_array(std::move(models));
    return json_ok(Value::make_object(std::move(root)));
}

HttpResponse ApiHandler::handle_models(const HttpRequest &, const PathParams &) {
    Value::Array data;
    for (const auto & summary : pool_.summaries()) {
        Value::Object entry;
        entry["id"] = Value::make_string(summary.id);
        entry["object"] = Value::make_string("model");
        entry["family"] = Value::make_string(summary.family);
        entry["task"] = Value::make_string(summary.task);
        entry["mode"] = Value::make_string(summary.mode);
        entry["path"] = Value::make_string(summary.path);
        entry["loaded"] = Value::make_bool(summary.loaded);
        entry["lazy"] = Value::make_bool(summary.lazy);
        data.push_back(Value::make_object(std::move(entry)));
    }
    Value::Object root;
    root["object"] = Value::make_string("list");
    root["data"] = Value::make_array(std::move(data));
    return json_ok(Value::make_object(std::move(root)));
}

HttpResponse ApiHandler::handle_model_load(const HttpRequest & request, const PathParams &) {
    const auto body = parse_body(request.body);
    const auto id = engine::io::json::optional_string(body, "model", "");
    if (id.empty()) {
        throw bad_request("request requires a 'model' id");
    }
    if (!pool_.has(id)) {
        throw not_found("unknown model id: " + id);
    }
    // A load can force an eviction, so it must not land while a job is running.
    if (!queue_.try_claim(id)) {
        throw conflict("model '" + id + "' is in use by a running job");
    }
    const auto started = now_ms();
    try {
        pool_.load(id);
    } catch (...) {
        queue_.release(id);
        throw;
    }
    queue_.release(id);

    Value::Object root;
    root["model"] = Value::make_string(id);
    root["loaded"] = Value::make_bool(true);
    root["load_ms"] = Value::make_number(static_cast<double>(now_ms() - started));
    return json_ok(Value::make_object(std::move(root)));
}

HttpResponse ApiHandler::handle_model_unload(const HttpRequest & request, const PathParams &) {
    const auto body = parse_body(request.body);
    if (engine::io::json::optional_bool(body, "all", false)) {
        const int count = pool_.unload_all_idle();
        Value::Object root;
        root["unloaded"] = Value::make_number(count);
        return json_ok(Value::make_object(std::move(root)));
    }
    const auto id = engine::io::json::optional_string(body, "model", "");
    if (id.empty()) {
        throw bad_request("request requires a 'model' id, or {\"all\":true}");
    }
    if (!pool_.has(id)) {
        throw not_found("unknown model id: " + id);
    }
    if (!queue_.try_claim(id)) {
        throw conflict("model '" + id + "' is in use by a running job");
    }
    const bool unloaded = pool_.unload(id);
    queue_.release(id);

    Value::Object root;
    root["model"] = Value::make_string(id);
    root["loaded"] = Value::make_bool(!unloaded && false);
    root["unloaded"] = Value::make_bool(unloaded);
    return json_ok(Value::make_object(std::move(root)));
}

HttpResponse ApiHandler::handle_submit_task(const HttpRequest & request, const PathParams &) {
    const auto body = parse_body(request.body);
    const auto model_id = engine::io::json::optional_string(body, "model", "");
    if (model_id.empty()) {
        throw bad_request("request requires a 'model' id");
    }
    if (!pool_.has(model_id)) {
        throw not_found("unknown model id: " + model_id);
    }

    Job job;
    job.id = make_job_id(now_ms());
    job.model_id = model_id;
    job.family = pool_.family_of(model_id);
    job.operation = engine::io::json::optional_string(body, "operation", "task");
    job.status = JobStatus::Queued;
    job.priority = static_cast<int>(engine::io::json::optional_i64(body, "priority", 0));
    job.created_ms = now_ms();
    job.updated_ms = job.created_ms;
    job.request = body;
    job.server_run_id = server_run_id_;
    if (const auto * metadata = body.find("metadata")) {
        job.metadata = *metadata;
    }
    const auto idempotency = engine::io::json::optional_string(body, "idempotency_key", "");
    if (!idempotency.empty()) {
        job.idempotency_key = idempotency;
    }

    // Durable before the client is told it exists, so a crash between the two
    // leaves a record that reconciliation can pick up rather than a lost job.
    store_.create(job);

    const auto pushed = queue_.push(QueueEntry{job.id, job.model_id, job.priority, job.created_ms});
    if (pushed == PushResult::QueueFull) {
        job.status = JobStatus::Failed;
        job.error.message = "the queue is full (" + std::to_string(config_.max_queued) + " waiting)";
        job.error.type = "server_busy";
        job.finished_ms = now_ms();
        store_.save(job);
        throw busy(job.error.message);
    }

    JobViewOptions options;
    options.queue_position = queue_.position(job.id);
    auto response = json_ok(job_envelope(job, options), 202);
    response.headers["Location"] = "/v1/jobs/" + job.id;
    return response;
}

HttpResponse ApiHandler::handle_list_jobs(const HttpRequest & request, const PathParams &) {
    const auto params = parse_query(request.query);
    const auto query = parse_job_query(params);
    const auto page = store_.list(query);
    const bool include_request = params.count("include_request") &&
                                 (params.at("include_request") == "1" ||
                                  params.at("include_request") == "true");
    return json_ok(job_list_envelope(page, include_request));
}

HttpResponse ApiHandler::handle_get_job(const HttpRequest &, const PathParams & params) {
    const auto job = require_job(params.at("id"));
    JobViewOptions options;
    options.queue_position = queue_.position(job.id);
    return json_ok(job_envelope(job, options));
}

HttpResponse ApiHandler::handle_job_result(const HttpRequest &, const PathParams & params) {
    const auto job = require_job(params.at("id"));
    if (!is_terminal(job.status)) {
        throw conflict("job '" + job.id + "' is still " + status_name(job.status) +
                       "; poll /v1/jobs/" + job.id + " or stream /v1/jobs/" + job.id + "/events");
    }
    return json_ok(job_result_envelope(job));
}

HttpResponse ApiHandler::handle_job_output(const HttpRequest &, const PathParams & params) {
    const auto job = require_job(params.at("id"));
    const auto name = params.at("name");
    // The name is matched against the job's own descriptors, never used to build
    // a path: a client cannot name a file the job does not own.
    const auto output = store_.find_output(job, name);
    if (!output.has_value()) {
        throw not_found("job '" + job.id + "' has no output named '" + name + "'");
    }
    const auto path = store_.output_path(job, *output);
    auto bytes = read_file_bytes(path);

    HttpResponse response;
    response.status = 200;
    response.content_type = content_type_for(*output);
    response.headers["Content-Disposition"] =
        "attachment; filename=\"" + download_filename(job, *output) + "\"";
    response.body = std::move(bytes);
    return response;
}

HttpResponse ApiHandler::handle_job_cancel(const HttpRequest &, const PathParams & params) {
    auto job = require_job(params.at("id"));
    if (is_terminal(job.status)) {
        throw conflict("job '" + job.id + "' is already " + status_name(job.status));
    }

    job.cancel_requested = true;
    queue_.request_cancel(job.id);

    // Still queued: nothing has started, so this is decided here and now.
    if (queue_.cancel_queued(job.id)) {
        queue_.take_cancel(job.id);
        job.status = JobStatus::Cancelled;
        job.finished_ms = now_ms();
        job.updated_ms = *job.finished_ms;
        job.error.message = "cancelled while queued";
        job.error.type = "cancelled";
        store_.save(job);
        publish_status(job);
        return json_ok(job_envelope(job), 200);
    }

    store_.save(job);
    if (workers_ && workers_->cancel_running(job.id)) {
        // Accepted, not done: it lands at the model's next step boundary.
        return json_ok(job_envelope(job), 202);
    }
    return json_ok(job_envelope(job), 202);
}

HttpResponse ApiHandler::handle_job_delete(const HttpRequest &, const PathParams & params) {
    const auto job = require_job(params.at("id"));
    if (!is_terminal(job.status)) {
        throw conflict("job '" + job.id + "' is " + status_name(job.status) + "; cancel it first");
    }
    store_.remove(job.id);
    events_.forget(job.id);
    Value::Object root;
    root["id"] = Value::make_string(job.id);
    root["deleted"] = Value::make_bool(true);
    return json_ok(Value::make_object(std::move(root)));
}

HttpResponse ApiHandler::handle_job_patch(const HttpRequest & request, const PathParams & params) {
    auto job = require_job(params.at("id"));
    const auto body = parse_body(request.body);
    bool changed = false;
    if (const auto * pinned = body.find("pinned"); pinned != nullptr && pinned->is_bool()) {
        job.pinned = pinned->as_bool();
        changed = true;
    }
    if (const auto * metadata = body.find("metadata")) {
        job.metadata = *metadata;
        changed = true;
    }
    if (!changed) {
        throw bad_request("only 'pinned' and 'metadata' can be changed");
    }
    job.updated_ms = now_ms();
    store_.save(job);
    return json_ok(job_envelope(job));
}

HttpResponse ApiHandler::handle_job_events(const HttpRequest & request, const PathParams & params) {
    const auto job = require_job(params.at("id"));
    if (!events_.try_subscribe()) {
        throw busy("too many event subscribers (" + std::to_string(config_.max_event_subscribers) + ")");
    }

    const auto query = parse_query(request.query);
    int64_t after_seq = 0;
    const auto since = query.find("since");
    if (since != query.end() && !since->second.empty()) {
        try {
            after_seq = std::stoll(since->second);
        } catch (const std::exception &) {
            events_.unsubscribe();
            throw bad_request("query parameter 'since' must be an integer");
        }
    }
    const auto last_event_id = request.headers.find("last-event-id");
    if (last_event_id != request.headers.end()) {
        try {
            after_seq = std::stoll(last_event_id->second);
        } catch (const std::exception &) {
            // A malformed reconnect header just means "start from the beginning".
        }
    }

    const bool already_done = is_terminal(job.status);
    const std::string job_id = job.id;

    HttpResponse response;
    response.status = 200;
    response.content_type = "text/event-stream";
    response.headers["Cache-Control"] = "no-cache";
    response.headers["X-Accel-Buffering"] = "no";
    response.stream_body = [this, job_id, after_seq, already_done](
                               minitts::server::HttpStreamWriter & writer) {
        int64_t cursor = after_seq;
        struct Unsubscriber {
            EventBus & bus;
            ~Unsubscriber() { bus.unsubscribe(); }
        } unsubscriber{events_};

        // Replay whatever the ring still holds, so a client attaching mid-run
        // sees the whole story rather than only what happens next.
        const auto backlog = events_.replay(job_id, cursor);
        if (backlog.dropped > 0) {
            writer.write(sse_comment("dropped " + std::to_string(backlog.dropped) +
                                     " events before this point"));
        }
        for (const auto & event : backlog.events) {
            writer.write(sse_frame(event));
            cursor = event.seq;
        }
        if (backlog.closed || already_done) {
            return;
        }

        for (;;) {
            const auto batch = events_.wait(job_id, cursor, 15000);
            for (const auto & event : batch.events) {
                writer.write(sse_frame(event));
                cursor = event.seq;
            }
            if (batch.closed) {
                return;
            }
            if (batch.events.empty()) {
                // Keeps proxies and socket send timeouts from killing a stream
                // that is simply waiting through a long stage.
                writer.write(sse_comment("keepalive"));
            }
        }
    };
    return response;
}

HttpResponse ApiHandler::handle_queue(const HttpRequest &, const PathParams &) {
    Value::Array queued;
    int position = 0;
    for (const auto & entry : queue_.snapshot()) {
        Value::Object item;
        item["id"] = Value::make_string(entry.job_id);
        item["model"] = Value::make_string(entry.model_id);
        item["priority"] = Value::make_number(entry.priority);
        item["position"] = Value::make_number(position++);
        queued.push_back(Value::make_object(std::move(item)));
    }

    Value::Array running;
    if (workers_) {
        for (const auto & id : workers_->running_ids()) {
            Value::Object item;
            item["id"] = Value::make_string(id);
            const auto job = store_.load(id);
            if (job.has_value()) {
                item["model"] = Value::make_string(job->model_id);
                item["stage"] = Value::make_string(job->progress.stage);
            }
            running.push_back(Value::make_object(std::move(item)));
        }
    }

    Value::Object capacity;
    capacity["workers"] = Value::make_number(config_.workers);
    capacity["max_queued"] = Value::make_number(static_cast<double>(config_.max_queued));

    Value::Object root;
    root["queued"] = Value::make_array(std::move(queued));
    root["running"] = Value::make_array(std::move(running));
    root["capacity"] = Value::make_object(std::move(capacity));
    return json_ok(Value::make_object(std::move(root)));
}

HttpResponse ApiHandler::handle_openapi(const HttpRequest &, const PathParams &) {
    // Served from beside the binary so the spec a client generates against is
    // always the one this build shipped with.
    const std::filesystem::path candidates[] = {
        config_dir_ / "openapi.yaml",
        std::filesystem::path("docs") / "openapi.yaml",
    };
    for (const auto & path : candidates) {
        if (std::filesystem::exists(path)) {
            HttpResponse response;
            response.status = 200;
            response.content_type = "application/yaml";
            response.body = read_file_bytes(path);
            return response;
        }
    }
    throw not_found("openapi.yaml is not installed beside this server");
}

void ApiHandler::register_routes() {
    auto bind = [this](auto method) {
        return [this, method](const HttpRequest & request, const PathParams & params) {
            return (this->*method)(request, params);
        };
    };

    router_.add("GET", "/health", bind(&ApiHandler::handle_health));
    router_.add("GET", "/openapi.yaml", bind(&ApiHandler::handle_openapi));
    router_.add("GET", "/v1/capabilities", bind(&ApiHandler::handle_capabilities));
    router_.add("GET", "/v1/models", bind(&ApiHandler::handle_models));
    router_.add("POST", "/v1/models/load", bind(&ApiHandler::handle_model_load));
    router_.add("POST", "/v1/models/unload", bind(&ApiHandler::handle_model_unload));
    router_.add("GET", "/v1/queue", bind(&ApiHandler::handle_queue));

    // Literal routes go in before the capture route that would also match them.
    router_.add("POST", "/v1/tasks", bind(&ApiHandler::handle_submit_task));
    router_.add("GET", "/v1/jobs", bind(&ApiHandler::handle_list_jobs));
    router_.add("POST", "/v1/jobs", bind(&ApiHandler::handle_submit_task));
    router_.add("GET", "/v1/jobs/{id}", bind(&ApiHandler::handle_get_job));
    router_.add("DELETE", "/v1/jobs/{id}", bind(&ApiHandler::handle_job_delete));
    router_.add("PATCH", "/v1/jobs/{id}", bind(&ApiHandler::handle_job_patch));
    router_.add("GET", "/v1/jobs/{id}/result", bind(&ApiHandler::handle_job_result));
    router_.add("GET", "/v1/jobs/{id}/events", bind(&ApiHandler::handle_job_events));
    router_.add("POST", "/v1/jobs/{id}/cancel", bind(&ApiHandler::handle_job_cancel));
    router_.add("GET", "/v1/jobs/{id}/outputs/{name}", bind(&ApiHandler::handle_job_output));
}

HttpResponse ApiHandler::handle(const HttpRequest & request) {
    if (request.method == "OPTIONS") {
        HttpResponse response;
        response.status = 204;
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS";
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Last-Event-ID";
        if (!config_.base.cors_origins.empty()) {
            response.headers["Access-Control-Allow-Origin"] = config_.base.cors_origins;
        }
        return response;
    }

    auto response = guard([&] { return router_.dispatch(request); });
    if (!config_.base.cors_origins.empty()) {
        response.headers["Access-Control-Allow-Origin"] = config_.base.cors_origins;
    }
    return response;
}

}  // namespace audiocpp::api
