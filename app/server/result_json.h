#pragma once

// One concern: rendering an engine TaskResult / StreamEvent as the JSON the HTTP
// API returns, plus the timing envelopes that wrap it. Extracted from
// runtime.cpp's anonymous namespace so the queued API server serialises results
// through exactly the same code as the synchronous one, instead of growing a
// second dialect of the same schema.

#include "engine/framework/runtime/session.h"

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace minitts::server {

std::string json_quote(std::string_view value);

double audio_duration_ms(const engine::runtime::AudioBuffer & audio);
double audio_rtf(double wall_ms, double duration_ms);
std::string timing_json(double wall_ms);
std::string timing_json(double wall_ms, const engine::runtime::AudioBuffer & audio);
std::string ttft_timing_json(double ttft_ms);
std::string live_speech_timing_json(double request_start_to_first_audio_ms, double input_end_ms);

bool stream_event_has_output(const engine::runtime::StreamEvent & event);
bool task_result_has_output(const engine::runtime::TaskResult & result);
const char * artifact_kind_name(engine::runtime::ArtifactKind kind);

double require_ttft_ms(const std::optional<double> & ttft_ms);
double require_input_end_ms(const std::optional<double> & input_end_ms);
std::unordered_map<std::string, std::string> timing_headers(
    double wall_ms,
    const engine::runtime::AudioBuffer & audio);

std::string task_result_json_with_timing(
    const engine::runtime::TaskResult & result,
    const std::string & timing);
std::string task_result_json(const engine::runtime::TaskResult & result, double wall_ms);
std::string alignment_result_json(
    const engine::runtime::TaskResult & result,
    const engine::runtime::AudioBuffer & audio,
    double wall_ms);
std::string streaming_task_result_json(
    const engine::runtime::TaskResult & result,
    const std::optional<double> & ttft_ms);
std::string stream_event_json(const engine::runtime::StreamEvent & event);

// A template, so it stays in the header: callers pass their own `field` lambda,
// which tracks comma placement for the object being built.
template <typename FieldFn>
void write_transcript_detail_fields(
    std::ostringstream & out,
    const engine::runtime::TaskResult & result,
    FieldFn field) {
    if (!result.speech_segments.empty()) {
        field("segments");
        out << "[";
        for (size_t i = 0; i < result.speech_segments.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & segment = result.speech_segments[i];
            out << "{\"start_sample\":" << segment.span.start_sample
                << ",\"end_sample\":" << segment.span.end_sample
                << ",\"confidence\":" << segment.confidence;
            if (!segment.text.empty()) {
                out << ",\"text\":" << json_quote(segment.text);
            }
            out << "}";
        }
        out << "]";
    }
    if (!result.speaker_turns.empty()) {
        field("speaker_turns");
        out << "[";
        for (size_t i = 0; i < result.speaker_turns.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & turn = result.speaker_turns[i];
            out << "{\"start_sample\":" << turn.span.start_sample
                << ",\"end_sample\":" << turn.span.end_sample
                << ",\"speaker_id\":" << json_quote(turn.speaker_id)
                << ",\"confidence\":" << turn.confidence;
            if (!turn.text.empty()) {
                out << ",\"text\":" << json_quote(turn.text);
            }
            out << "}";
        }
        out << "]";
    }
    if (!result.word_timestamps.empty()) {
        field("words");
        out << "[";
        for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & word = result.word_timestamps[i];
            out << "{\"word\":" << json_quote(word.word)
                << ",\"start_sample\":" << word.span.start_sample
                << ",\"end_sample\":" << word.span.end_sample
                << ",\"confidence\":" << word.confidence << "}";
        }
        out << "]";
    }
}

}  // namespace minitts::server
