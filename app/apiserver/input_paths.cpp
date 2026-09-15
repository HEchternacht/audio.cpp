#include "input_paths.h"

#include "api_error.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace audiocpp::api {

namespace {

// The request fields the CLI builder turns into paths, plus the option suffixes
// families use for the same thing. Listed rather than guessed: a field that
// reads a file and is not here is a hole, which is why the self-test walks the
// same list the builder does.
const char * const kPathFields[] = {
    "audio", "voice_ref", "source_audio", "target_voice", "prosody_ref", "style_ref",
};

bool looks_like_path_option(const std::string & key) {
    const auto ends_with = [&key](const char * suffix) {
        const std::string tail(suffix);
        return key.size() >= tail.size() && key.compare(key.size() - tail.size(), tail.size(), tail) == 0;
    };
    return ends_with("_file") || ends_with("_path") || key == "audio" || key == "source_audio";
}

void require_confined(const std::string & field, const std::string & value) {
    if (value.empty() || is_confined_name(value)) return;
    throw bad_request(
        field + " must name a file inside this job's own inputs: upload it first and use the name it "
                "comes back with. Absolute paths and .. are refused, because over HTTP they would read "
                "whatever the server can reach.");
}

}  // namespace

bool is_confined_name(const std::string & value) {
    if (value.empty()) return false;
    // "-" means stdin to the CLI, and nothing at all to a server.
    if (value == "-") return false;
    const std::filesystem::path path(value);
    if (path.is_absolute() || !path.root_name().empty() || !path.root_directory().empty()) return false;
    for (const auto & part : path) {
        if (part == "..") return false;
    }
    return true;
}

std::string safe_upload_name(const std::string & requested) {
    const std::filesystem::path path(requested);
    std::string stem = path.stem().string();
    std::string extension = path.extension().string();

    const auto scrub = [](std::string & text, size_t limit) {
        std::string out;
        for (const char character : text) {
            const bool keep = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9') || character == '.' ||
                              character == '-' || character == '_';
            out.push_back(keep ? character : '_');
            if (out.size() >= limit) break;
        }
        text = out;
    };
    scrub(stem, 64);
    scrub(extension, 12);

    // A name made entirely of what was scrubbed away is no name at all.
    if (stem.empty() || stem.find_first_not_of("._-") == std::string::npos) stem = "upload";
    if (extension.empty() || extension == ".") extension = ".bin";
    if (extension.front() != '.') extension.insert(extension.begin(), '.');
    return stem + extension;
}

void confine_request_paths(const engine::io::json::Value & request) {
    for (const char * field : kPathFields) {
        const auto * value = request.find(field);
        if (value != nullptr && value->is_string()) require_confined(field, value->as_string());
    }
    const auto * voice = request.find("voice");
    if (voice != nullptr && voice->is_object()) {
        for (const char * field : {"ref", "audio"}) {
            const auto * value = voice->find(field);
            if (value != nullptr && value->is_string()) {
                require_confined(std::string("voice.") + field, value->as_string());
            }
        }
    }
    const auto * options = request.find("options");
    if (options != nullptr && options->is_object()) {
        for (const auto & entry : options->as_object()) {
            if (!entry.second.is_string() || !looks_like_path_option(entry.first)) continue;
            require_confined("options." + entry.first, entry.second.as_string());
        }
    }
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_INPUT_PATHS
// Direct-execution self-test. Never fires on import: this block only compiles
// when this file is built as its own target with -DAUDIOCPP_SELFTEST_INPUT_PATHS,
// so a sibling linked in as a dependency never contributes a second main().
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

std::string refused(const std::string & body) {
    try {
        audiocpp::api::confine_request_paths(engine::io::json::parse(body));
        return "accepted";
    } catch (const audiocpp::api::ApiError &) {
        return "refused";
    }
}

}  // namespace

int main() {
    using audiocpp::api::is_confined_name;
    using audiocpp::api::safe_upload_name;

    for (const char * name : {"song.wav", "take.flac", "nested/take.wav"}) {
        record("is_confined_name/plain", name, is_confined_name(name) ? "ok" : "refused", "ok");
    }
    // The three shapes that walk out of the job: absolute, drive-relative, and
    // the one that looks harmless until it is joined to a base directory.
    for (const char * name : {"C:/Users/henri/.ssh/id_rsa", "/etc/passwd", "C:song.wav",
                              "../../secrets.wav", "nested/../../secrets.wav", "-", ""}) {
        record("is_confined_name/escape", name, is_confined_name(name) ? "ok" : "refused", "refused");
    }

    record("safe_upload_name/keeps_extension", "My Song (final).WAV",
           safe_upload_name("My Song (final).WAV"), "My_Song__final_.WAV");
    record("safe_upload_name/strips_directories", "../../etc/passwd",
           safe_upload_name("../../etc/passwd"), "passwd.bin");
    record("safe_upload_name/names_the_nameless", "...", safe_upload_name("..."), "upload.bin");

    record("confine_request_paths/plain", R"({"audio":"song.wav"})",
           refused(R"({"audio":"song.wav"})"), "accepted");
    record("confine_request_paths/absolute", R"({"audio":"C:/Users/henri/.ssh/id_rsa"})",
           refused(R"({"audio":"C:/Users/henri/.ssh/id_rsa"})"), "refused");
    record("confine_request_paths/traversal", R"({"source_audio":"../../x.wav"})",
           refused(R"({"source_audio":"../../x.wav"})"), "refused");
    record("confine_request_paths/voice_ref", R"({"voice":{"ref":"/etc/passwd"})",
           refused(R"({"voice":{"ref":"/etc/passwd"}})"), "refused");
    // Options reach the families directly, and several of them open a file.
    record("confine_request_paths/option_file", R"({"options":{"abc_file":"/etc/passwd"}})",
           refused(R"({"options":{"abc_file":"/etc/passwd"}})"), "refused");
    record("confine_request_paths/option_text", R"({"options":{"style":"C:/not/a/path"}})",
           refused(R"({"options":{"style":"C:/not/a/path"}})"), "accepted");
    record("confine_request_paths/empty", "{}", refused("{}"), "accepted");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_INPUT_PATHS
