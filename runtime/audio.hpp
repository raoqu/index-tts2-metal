#pragma once

#include <cstdint>
#include <string>

namespace mtts_audio {

struct WavInfo {
    bool valid = false;
    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    uint64_t frames = 0;
    double duration_seconds = 0.0;
    bool pcm16_mono = false;
};

WavInfo inspect_wav_file(const std::string& path, std::string* error = nullptr);

bool convert_wav_to_pcm16_mono(const std::string& input_wav,
                               const std::string& output_wav,
                               WavInfo* output_info = nullptr,
                               std::string* error = nullptr);

}  // namespace mtts_audio
