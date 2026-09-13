#include "result_json.h"

#include "base64.h"
#include "wav.h"

#include "engine/framework/io/json.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minitts::server {

std::string json_quote(std::string_view value) {
    return engine::io::json::stringify_string(value);
}

double audio_duration_ms(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        return 0.0;
    }
    return 1000.0 * static_cast<double>(audio.samples.size()) /
        static_cast<double>(audio.sample_rate * audio.channels);
}

double audio_rtf(double wall_ms, double duration_ms) {
    return duration_ms > 0.0 ? wall_ms / duration_ms : 0.0;
}

std::string timing_json(double wall_ms) {
    std::ostringstream out;
    out << "{\"wall_ms\":" << wall_ms << "}";
    return out.str();
}

std::string timing_json(double wall_ms, const engine::runtime::AudioBuffer & audio) {
    const double duration_ms = audio_duration_ms(audio);
    std::ostringstream out;
    out << "{\"wall_ms\":" << wall_ms
        << ",\"audio_duration_ms\":" << duration_ms
        << ",\"rtf\":" << audio_rtf(wall_ms, duration_ms) << "}";
    return out.str();
}

std::string ttft_timing_json(double ttft_ms) {
    std::ostringstream out;
    out << "{\"ttft_ms\":" << ttft_ms << "}";
    return out.str();
}

std::string live_speech_timing_json(double request_start_to_first_audio_ms, double input_end_ms) {
    const double input_end_to_first_audio_ms = request_start_to_first_audio_ms - input_end_ms;
    const bool first_audio_before_input_end = input_end_to_first_audio_ms < 0.0;
    const double overlap_ms = first_audio_before_input_end ? -input_end_to_first_audio_ms : 0.0;
    std::ostringstream out;
    out << "{\"ttft_ms\":";
    if (first_audio_before_input_end) {
        out << "null";
    } else {
        out << input_end_to_first_audio_ms;
    }
    out << ",\"first_audio_before_input_end\":" << (first_audio_before_input_end ? "true" : "false")
        << ",\"request_start_to_first_audio_ms\":" << request_start_to_first_audio_ms
        << ",\"input_end_ms\":" << input_end_ms
        << ",\"overlap_ms\":" << overlap_ms << "}";
    return out.str();
}

bool stream_event_has_output(const engine::runtime::StreamEvent & event) {
    return (event.partial_text.has_value() && !event.partial_text->text.empty()) ||
        event.audio_output.has_value() ||
        !event.named_audio_outputs.empty();
}

bool task_result_has_output(const engine::runtime::TaskResult & result) {
    return result.text_output.has_value() ||
        result.audio_output.has_value() ||
        !result.named_audio_outputs.empty() ||
        result.artifact_output.has_value() ||
        !result.output_artifacts.empty();
}

const char * artifact_kind_name(engine::runtime::ArtifactKind kind) {
    using engine::runtime::ArtifactKind;
    switch (kind) {
    case ArtifactKind::SpeakerEmbedding: return "speaker_embedding";
    case ArtifactKind::StyleEmbedding: return "style_embedding";
    case ArtifactKind::PromptEmbedding: return "prompt_embedding";
    case ArtifactKind::AcousticTokens: return "acoustic_tokens";
    case ArtifactKind::Midi: return "midi";
    case ArtifactKind::TranscriptAlignment: return "transcript_alignment";
    case ArtifactKind::DiarizationState: return "diarization_state";
    case ArtifactKind::VadState: return "vad_state";
    case ArtifactKind::Custom: return "custom";
    }
    return "custom";
}

double require_ttft_ms(const std::optional<double> & ttft_ms) {
    if (!ttft_ms.has_value()) {
        throw std::runtime_error("streaming response produced no TTFT event");
    }
    return *ttft_ms;
}

double require_input_end_ms(const std::optional<double> & input_end_ms) {
    if (!input_end_ms.has_value()) {
        throw std::runtime_error("live streaming response did not observe input end");
    }
    return *input_end_ms;
}

std::unordered_map<std::string, std::string> timing_headers(
    double wall_ms,
    const engine::runtime::AudioBuffer & audio) {
    const double duration_ms = audio_duration_ms(audio);
    return {
        {"X-AudioCPP-Wall-Ms", std::to_string(wall_ms)},
        {"X-AudioCPP-Audio-Duration-Ms", std::to_string(duration_ms)},
        {"X-AudioCPP-RTF", std::to_string(audio_rtf(wall_ms, duration_ms))},
    };
}

// Transcript detail arrays shared by /v1/tasks/run and /v1/audio/transcriptions.
// ASR models that produce timestamps populate only these fields, so a route that
// omits them silently discards work the model already did.
std::string task_result_json_with_timing(
    const engine::runtime::TaskResult & result,
    const std::string & timing) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const std::string & name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << json_quote(name) << ":";
    };

    if (result.text_output.has_value()) {
        field("text");
        out << json_quote(result.text_output->text);
        if (!result.text_output->language.empty()) {
            field("language");
            out << json_quote(result.text_output->language);
        }
    }
    if (!result.notes.empty()) {
        field("notes");
        out << "[";
        for (size_t i = 0; i < result.notes.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << json_quote(result.notes[i]);
        }
        out << "]";
    }
    if (result.audio_output.has_value()) {
        const auto wav = encode_pcm16_wav(*result.audio_output);
        field("audio");
        out << json_quote(base64_encode(wav));
        field("sample_rate");
        out << result.audio_output->sample_rate;
        field("channels");
        out << result.audio_output->channels;
    }
    if (!result.named_audio_outputs.empty()) {
        field("named_audio_outputs");
        out << "[";
        for (size_t i = 0; i < result.named_audio_outputs.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto wav = encode_pcm16_wav(result.named_audio_outputs[i].audio);
            out << "{\"id\":" << json_quote(result.named_audio_outputs[i].id)
                << ",\"audio\":" << json_quote(base64_encode(wav))
                << ",\"sample_rate\":" << result.named_audio_outputs[i].audio.sample_rate
                << ",\"channels\":" << result.named_audio_outputs[i].audio.channels
                << "}";
        }
        out << "]";
    }
    if (result.artifact_output.has_value() || !result.output_artifacts.empty()) {
        field("artifacts");
        out << "[";
        bool first_artifact = true;
        auto write_artifact = [&](const engine::runtime::VoiceArtifact & artifact) {
            if (!first_artifact) out << ",";
            first_artifact = false;
            out << "{\"id\":" << json_quote(artifact.id)
                << ",\"kind\":" << json_quote(artifact_kind_name(artifact.kind))
                << ",\"payload\":" << json_quote(base64_encode(artifact.payload))
                << ",\"meta\":{";
            bool first_meta = true;
            for (const auto & [key, value] : artifact.meta) {
                if (!first_meta) out << ",";
                first_meta = false;
                out << json_quote(key) << ":" << json_quote(value);
            }
            out << "}}";
        };
        if (result.artifact_output.has_value()) write_artifact(*result.artifact_output);
        for (const auto & artifact : result.output_artifacts) write_artifact(artifact);
        out << "]";
    }
    write_transcript_detail_fields(out, result, field);
    field("timing");
    out << timing;
    out << "}";
    return out.str();
}

std::string task_result_json(const engine::runtime::TaskResult & result, double wall_ms) {
    if (result.audio_output.has_value()) {
        return task_result_json_with_timing(result, timing_json(wall_ms, *result.audio_output));
    }
    if (result.named_audio_outputs.size() == 1) {
        return task_result_json_with_timing(result, timing_json(wall_ms, result.named_audio_outputs.front().audio));
    }
    return task_result_json_with_timing(result, timing_json(wall_ms));
}

std::string alignment_result_json(
    const engine::runtime::TaskResult & result,
    const engine::runtime::AudioBuffer & audio,
    double wall_ms) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("alignment timing requires a positive input sample rate");
    }

    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const std::string & name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << json_quote(name) << ":";
    };
    if (result.text_output.has_value()) {
        field("text");
        out << json_quote(result.text_output->text);
        if (!result.text_output->language.empty()) {
            field("language");
            out << json_quote(result.text_output->language);
        }
    }
    field("words");
    out << "[";
    for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & word = result.word_timestamps[i];
        out << "{\"word\":" << json_quote(word.word)
            << ",\"start\":" << (static_cast<double>(word.span.start_sample) / audio.sample_rate)
            << ",\"end\":" << (static_cast<double>(word.span.end_sample) / audio.sample_rate)
            << ",\"start_sample\":" << word.span.start_sample
            << ",\"end_sample\":" << word.span.end_sample
            << ",\"confidence\":" << word.confidence << "}";
    }
    out << "]";
    field("timing");
    out << timing_json(wall_ms, audio);
    out << "}";
    return out.str();
}

std::string streaming_task_result_json(
    const engine::runtime::TaskResult & result,
    const std::optional<double> & ttft_ms) {
    return task_result_json_with_timing(result, ttft_timing_json(require_ttft_ms(ttft_ms)));
}

std::string stream_event_json(const engine::runtime::StreamEvent & event) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const char * name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << name << "\":";
    };
    if (event.partial_text.has_value()) {
        field("partial_text");
        out << "{\"text\":" << json_quote(event.partial_text->text)
            << ",\"language\":" << json_quote(event.partial_text->language)
            << "}";
    }
    if (event.audio_output.has_value()) {
        const auto wav = encode_pcm16_wav(*event.audio_output);
        field("audio");
        out << json_quote(base64_encode(wav));
    }
    if (!event.named_audio_outputs.empty()) {
        field("named_audio_outputs");
        out << "[";
        for (size_t i = 0; i < event.named_audio_outputs.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto wav = encode_pcm16_wav(event.named_audio_outputs[i].audio);
            out << "{\"id\":" << json_quote(event.named_audio_outputs[i].id)
                << ",\"audio\":" << json_quote(base64_encode(wav))
                << ",\"format\":\"wav\"}";
        }
        out << "]";
    }
    if (!event.word_timestamps.empty()) {
        field("word_timestamps");
        out << "[";
        for (size_t i = 0; i < event.word_timestamps.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << "{\"start_sample\":" << event.word_timestamps[i].span.start_sample
                << ",\"end_sample\":" << event.word_timestamps[i].span.end_sample
                << ",\"word\":" << json_quote(event.word_timestamps[i].word)
                << ",\"confidence\":" << event.word_timestamps[i].confidence
                << "}";
        }
        out << "]";
    }
    field("is_final");
    out << (event.is_final ? "true" : "false");
    out << "}";
    return out.str();
}

}  // namespace minitts::server
