#pragma once

// One concern: turning an engine AudioBuffer into PCM16 bytes, either framed as
// a RIFF/WAVE file or raw. Extracted from runtime.cpp's anonymous namespace so
// the API server can reuse the exact same framing rather than growing a second,
// subtly different copy.

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace minitts::server {

// Little-endian 16-bit PCM in a 44-byte canonical RIFF/WAVE container.
// Samples are clamped to [-1, 1] before scaling.
std::vector<uint8_t> encode_pcm16_wav(const engine::runtime::AudioBuffer & audio);

// The same sample conversion with no container, for streaming transports that
// send the header once and then raw frames.
std::vector<uint8_t> encode_pcm16_samples(const engine::runtime::AudioBuffer & audio);

}  // namespace minitts::server
