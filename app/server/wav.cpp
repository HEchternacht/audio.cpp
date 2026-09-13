#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace minitts::server {

std::vector<uint8_t> encode_pcm16_wav(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    const uint16_t channels = static_cast<uint16_t>(audio.channels);
    const uint16_t bits_per_sample = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;
    const uint32_t byte_rate = static_cast<uint32_t>(audio.sample_rate) * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;

    std::vector<uint8_t> out;
    out.reserve(44 + data_bytes);
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    auto append_u16 = [&](uint16_t value) { append_bytes(&value, sizeof(value)); };
    auto append_u32 = [&](uint32_t value) { append_bytes(&value, sizeof(value)); };

    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    append_u32(riff_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_u32(16);
    append_u16(1);
    append_u16(channels);
    append_u32(static_cast<uint32_t>(audio.sample_rate));
    append_u32(byte_rate);
    append_u16(block_align);
    append_u16(bits_per_sample);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    append_u32(data_bytes);
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

std::vector<uint8_t> encode_pcm16_samples(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    std::vector<uint8_t> out;
    out.reserve(audio.samples.size() * sizeof(int16_t));
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

}  // namespace minitts::server

#ifdef AUDIOCPP_SELFTEST_WAV
// Direct-execution self-test. Never fires on import: this block only compiles
// when this file is built as its own target with -DAUDIOCPP_SELFTEST_WAV, so a
// sibling linked in as a dependency never contributes a second main().
#include <cstdio>
#include <string>

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

// Reads a little-endian scalar out of the encoded byte stream.
template <typename T>
T read_at(const std::vector<uint8_t> & bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

std::string ascii_at(const std::vector<uint8_t> & bytes, size_t offset, size_t count) {
    return std::string(reinterpret_cast<const char *>(bytes.data()) + offset, count);
}

engine::runtime::AudioBuffer make_audio(std::vector<float> samples, int rate, int channels) {
    engine::runtime::AudioBuffer audio;
    audio.samples = std::move(samples);
    audio.sample_rate = rate;
    audio.channels = channels;
    return audio;
}

// Exercises a call expected to throw; records rather than aborting either way.
void expect_throws(const char * unit, const std::string & in, const std::function<void()> & fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    record(unit, in, threw ? "throws" : "returns", "throws");
}

}  // namespace

int main() {
    using minitts::server::encode_pcm16_samples;
    using minitts::server::encode_pcm16_wav;

    // A 4-sample mono buffer: full scale both ways, silence, and a half value.
    const auto mono = make_audio({1.0F, -1.0F, 0.0F, 0.5F}, 48000, 1);

    try {
        const auto wav = encode_pcm16_wav(mono);
        record("encode_pcm16_wav/size", "4 mono samples", std::to_string(wav.size()), std::to_string(44 + 8));
        record("encode_pcm16_wav/riff", "4 mono samples", ascii_at(wav, 0, 4), "RIFF");
        record("encode_pcm16_wav/wave", "4 mono samples", ascii_at(wav, 8, 4), "WAVE");
        record("encode_pcm16_wav/data_tag", "4 mono samples", ascii_at(wav, 36, 4), "data");
        record("encode_pcm16_wav/riff_size", "4 mono samples",
               std::to_string(read_at<uint32_t>(wav, 4)), std::to_string(36 + 8));
        record("encode_pcm16_wav/rate", "48000", std::to_string(read_at<uint32_t>(wav, 24)), "48000");
        record("encode_pcm16_wav/channels", "1", std::to_string(read_at<uint16_t>(wav, 22)), "1");
        record("encode_pcm16_wav/bits", "pcm16", std::to_string(read_at<uint16_t>(wav, 34)), "16");
        record("encode_pcm16_wav/block_align", "1ch pcm16", std::to_string(read_at<uint16_t>(wav, 32)), "2");
        record("encode_pcm16_wav/byte_rate", "48000 1ch pcm16",
               std::to_string(read_at<uint32_t>(wav, 28)), std::to_string(48000 * 2));
        record("encode_pcm16_wav/data_size", "4 mono samples",
               std::to_string(read_at<uint32_t>(wav, 40)), "8");
        // Clamping and rounding: +1 -> 32767, -1 -> -32767, 0 -> 0, 0.5 -> 16384 (lrint).
        record("encode_pcm16_wav/sample0", "1.0", std::to_string(read_at<int16_t>(wav, 44)), "32767");
        record("encode_pcm16_wav/sample1", "-1.0", std::to_string(read_at<int16_t>(wav, 46)), "-32767");
        record("encode_pcm16_wav/sample2", "0.0", std::to_string(read_at<int16_t>(wav, 48)), "0");
        record("encode_pcm16_wav/sample3", "0.5", std::to_string(read_at<int16_t>(wav, 50)), "16384");
    } catch (const std::exception & ex) {
        std::printf("[encode_pcm16_wav] FAIL unexpected throw: %s\n", ex.what());
        ++g_fail;
    }

    // Out-of-range input must be clamped, not wrapped.
    try {
        const auto loud = make_audio({4.0F, -4.0F}, 8000, 1);
        const auto wav = encode_pcm16_wav(loud);
        record("encode_pcm16_wav/clamp_high", "4.0", std::to_string(read_at<int16_t>(wav, 44)), "32767");
        record("encode_pcm16_wav/clamp_low", "-4.0", std::to_string(read_at<int16_t>(wav, 46)), "-32767");
    } catch (const std::exception & ex) {
        std::printf("[encode_pcm16_wav/clamp] FAIL unexpected throw: %s\n", ex.what());
        ++g_fail;
    }

    // Stereo: block align and byte rate must double, and the frame count halves.
    try {
        const auto stereo = make_audio({0.0F, 0.0F, 0.0F, 0.0F}, 44100, 2);
        const auto wav = encode_pcm16_wav(stereo);
        record("encode_pcm16_wav/stereo_align", "2ch", std::to_string(read_at<uint16_t>(wav, 32)), "4");
        record("encode_pcm16_wav/stereo_byte_rate", "44100 2ch",
               std::to_string(read_at<uint32_t>(wav, 28)), std::to_string(44100 * 4));
    } catch (const std::exception & ex) {
        std::printf("[encode_pcm16_wav/stereo] FAIL unexpected throw: %s\n", ex.what());
        ++g_fail;
    }

    try {
        const auto raw = encode_pcm16_samples(mono);
        record("encode_pcm16_samples/size", "4 mono samples", std::to_string(raw.size()), "8");
        record("encode_pcm16_samples/sample0", "1.0", std::to_string(read_at<int16_t>(raw, 0)), "32767");
        record("encode_pcm16_samples/sample3", "0.5", std::to_string(read_at<int16_t>(raw, 6)), "16384");
        // The raw stream must be byte-identical to the WAV payload after the header.
        const auto wav = encode_pcm16_wav(mono);
        const bool same = std::equal(raw.begin(), raw.end(), wav.begin() + 44);
        record("encode_pcm16_samples/matches_wav_payload", "4 mono samples",
               same ? "identical" : "differs", "identical");
    } catch (const std::exception & ex) {
        std::printf("[encode_pcm16_samples] FAIL unexpected throw: %s\n", ex.what());
        ++g_fail;
    }

    expect_throws("encode_pcm16_wav/bad_rate", "sample_rate=0",
                  [] { encode_pcm16_wav(make_audio({0.0F}, 0, 1)); });
    expect_throws("encode_pcm16_wav/bad_channels", "channels=0",
                  [] { encode_pcm16_wav(make_audio({0.0F}, 48000, 0)); });
    expect_throws("encode_pcm16_wav/ragged", "3 samples over 2 channels",
                  [] { encode_pcm16_wav(make_audio({0.0F, 0.0F, 0.0F}, 48000, 2)); });
    expect_throws("encode_pcm16_samples/bad_rate", "sample_rate=-1",
                  [] { encode_pcm16_samples(make_audio({0.0F}, -1, 1)); });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_WAV
