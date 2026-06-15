#include "audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace mtts_audio {
namespace {

constexpr uint16_t kWavFormatPcm = 1;
constexpr uint16_t kWavFormatIeeeFloat = 3;
constexpr uint16_t kWavFormatExtensible = 0xfffe;

uint16_t read_u16_le(const std::vector<char>& bytes, size_t offset) {
    if (offset + 2 > bytes.size()) {
        throw std::runtime_error("truncated u16");
    }
    return static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset])) |
           static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset + 1]) << 8);
}

uint32_t read_u32_le(const std::vector<char>& bytes, size_t offset) {
    if (offset + 4 > bytes.size()) {
        throw std::runtime_error("truncated u32");
    }
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3])) << 24);
}

uint64_t read_u64_le(const std::vector<char>& bytes, size_t offset) {
    if (offset + 8 > bytes.size()) {
        throw std::runtime_error("truncated u64");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[offset + i])) << (i * 8u);
    }
    return value;
}

void write_u16_le(std::ostream& out, uint16_t value) {
    char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void write_u32_le(std::ostream& out, uint32_t value) {
    char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

std::vector<char> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open wav input file: " + path);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

struct ParsedWav {
    WavInfo info;
    std::vector<char> data;
};

ParsedWav parse_wav(const std::string& path) {
    std::vector<char> bytes = read_file_bytes(path);
    if (bytes.size() < 44 || std::string(bytes.data(), bytes.data() + 4) != "RIFF" ||
        std::string(bytes.data() + 8, bytes.data() + 12) != "WAVE") {
        throw std::runtime_error("expected RIFF/WAVE file: " + path);
    }

    bool saw_fmt = false;
    bool saw_data = false;
    uint16_t raw_format = 0;
    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> audio_data;

    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const std::string chunk_id(bytes.data() + pos, bytes.data() + pos + 4);
        const uint32_t chunk_size = read_u32_le(bytes, pos + 4);
        const size_t data_pos = pos + 8;
        if (data_pos + chunk_size > bytes.size()) {
            throw std::runtime_error("truncated wav chunk in: " + path);
        }
        if (chunk_id == "fmt ") {
            if (chunk_size < 16) {
                throw std::runtime_error("invalid wav fmt chunk in: " + path);
            }
            raw_format = read_u16_le(bytes, data_pos);
            channels = read_u16_le(bytes, data_pos + 2);
            sample_rate = read_u32_le(bytes, data_pos + 4);
            bits_per_sample = read_u16_le(bytes, data_pos + 14);
            format = raw_format;
            if (raw_format == kWavFormatExtensible && chunk_size >= 40) {
                format = read_u16_le(bytes, data_pos + 24);
            }
            saw_fmt = true;
        } else if (chunk_id == "data" && !saw_data) {
            audio_data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_pos),
                              bytes.begin() + static_cast<std::ptrdiff_t>(data_pos + chunk_size));
            saw_data = true;
        }
        pos = data_pos + chunk_size + (chunk_size & 1u);
    }

    if (!saw_fmt || !saw_data || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
        throw std::runtime_error("invalid wav: " + path);
    }

    const size_t bytes_per_sample = bits_per_sample / 8u;
    const size_t block_align = static_cast<size_t>(channels) * bytes_per_sample;
    if (bytes_per_sample == 0 || block_align == 0 || (audio_data.size() % block_align) != 0) {
        throw std::runtime_error("invalid wav data alignment: " + path);
    }

    WavInfo info;
    info.valid = true;
    info.format = format;
    info.channels = channels;
    info.sample_rate = sample_rate;
    info.bits_per_sample = bits_per_sample;
    info.frames = static_cast<uint64_t>(audio_data.size() / block_align);
    info.duration_seconds = static_cast<double>(info.frames) / static_cast<double>(sample_rate);
    info.pcm16_mono = format == kWavFormatPcm && channels == 1 && bits_per_sample == 16;
    (void)raw_format;
    return ParsedWav{info, std::move(audio_data)};
}

float sample_to_f32(const std::vector<char>& data, size_t offset, uint16_t format, uint16_t bits_per_sample) {
    if (format == kWavFormatPcm) {
        if (bits_per_sample == 8) {
            return (static_cast<int32_t>(static_cast<unsigned char>(data[offset])) - 128) / 128.0f;
        }
        if (bits_per_sample == 16) {
            const int16_t sample = static_cast<int16_t>(read_u16_le(data, offset));
            return static_cast<float>(sample) / 32768.0f;
        }
        if (bits_per_sample == 24) {
            int32_t sample = static_cast<int32_t>(static_cast<unsigned char>(data[offset])) |
                             (static_cast<int32_t>(static_cast<unsigned char>(data[offset + 1])) << 8) |
                             (static_cast<int32_t>(static_cast<unsigned char>(data[offset + 2])) << 16);
            if (sample & 0x00800000) sample |= static_cast<int32_t>(0xff000000);
            return static_cast<float>(sample) / 8388608.0f;
        }
        if (bits_per_sample == 32) {
            const int32_t sample = static_cast<int32_t>(read_u32_le(data, offset));
            return static_cast<float>(sample) / 2147483648.0f;
        }
    } else if (format == kWavFormatIeeeFloat) {
        if (bits_per_sample == 32) {
            const uint32_t raw = read_u32_le(data, offset);
            float sample = 0.0f;
            std::memcpy(&sample, &raw, sizeof(sample));
            return std::isfinite(sample) ? sample : 0.0f;
        }
        if (bits_per_sample == 64) {
            const uint64_t raw = read_u64_le(data, offset);
            double sample = 0.0;
            std::memcpy(&sample, &raw, sizeof(sample));
            return std::isfinite(sample) ? static_cast<float>(sample) : 0.0f;
        }
    }
    throw std::runtime_error("unsupported wav encoding");
}

void write_pcm16_mono_wav(const std::string& path,
                          uint32_t sample_rate,
                          const std::vector<char>& frames) {
    if ((frames.size() % 2) != 0) {
        throw std::runtime_error("PCM16 frame byte count must be even");
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open wav output file: " + path);
    }
    constexpr uint16_t channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    constexpr uint16_t bytes_per_sample = bits_per_sample / 8;
    const uint32_t data_bytes = static_cast<uint32_t>(frames.size());

    out.write("RIFF", 4);
    write_u32_le(out, 36u + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32_le(out, 16);
    write_u16_le(out, kWavFormatPcm);
    write_u16_le(out, channels);
    write_u32_le(out, sample_rate);
    write_u32_le(out, sample_rate * channels * bytes_per_sample);
    write_u16_le(out, channels * bytes_per_sample);
    write_u16_le(out, bits_per_sample);
    out.write("data", 4);
    write_u32_le(out, data_bytes);
    out.write(frames.data(), static_cast<std::streamsize>(frames.size()));
    if (!out) {
        throw std::runtime_error("failed to write wav output file: " + path);
    }
}

}  // namespace

WavInfo inspect_wav_file(const std::string& path, std::string* error) {
    try {
        return parse_wav(path).info;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return {};
    }
}

bool convert_wav_to_pcm16_mono(const std::string& input_wav,
                               const std::string& output_wav,
                               WavInfo* output_info,
                               std::string* error) {
    try {
        const ParsedWav parsed = parse_wav(input_wav);
        const bool supported_pcm = parsed.info.format == kWavFormatPcm &&
            (parsed.info.bits_per_sample == 8 || parsed.info.bits_per_sample == 16 ||
             parsed.info.bits_per_sample == 24 || parsed.info.bits_per_sample == 32);
        const bool supported_float = parsed.info.format == kWavFormatIeeeFloat &&
            (parsed.info.bits_per_sample == 32 || parsed.info.bits_per_sample == 64);
        if (!supported_pcm && !supported_float) {
            throw std::runtime_error("unsupported wav encoding: " + input_wav);
        }

        const size_t bytes_per_sample = parsed.info.bits_per_sample / 8u;
        const size_t block_align = static_cast<size_t>(parsed.info.channels) * bytes_per_sample;
        std::vector<char> frames;
        frames.reserve(static_cast<size_t>(parsed.info.frames) * 2u);
        for (uint64_t frame = 0; frame < parsed.info.frames; ++frame) {
            double mono = 0.0;
            const size_t frame_offset = static_cast<size_t>(frame) * block_align;
            for (uint16_t ch = 0; ch < parsed.info.channels; ++ch) {
                mono += static_cast<double>(sample_to_f32(parsed.data,
                                                          frame_offset + static_cast<size_t>(ch) * bytes_per_sample,
                                                          parsed.info.format,
                                                          parsed.info.bits_per_sample));
            }
            mono /= static_cast<double>(parsed.info.channels);
            const double clipped = std::max(-1.0, std::min(1.0, mono));
            const int32_t quantized = static_cast<int32_t>(std::lrint(clipped * 32767.0));
            const int16_t sample = static_cast<int16_t>(std::max(-32768, std::min(32767, quantized)));
            frames.push_back(static_cast<char>(static_cast<uint16_t>(sample) & 0xffu));
            frames.push_back(static_cast<char>((static_cast<uint16_t>(sample) >> 8) & 0xffu));
        }

        write_pcm16_mono_wav(output_wav, parsed.info.sample_rate, frames);
        if (output_info) {
            output_info->valid = true;
            output_info->format = kWavFormatPcm;
            output_info->channels = 1;
            output_info->sample_rate = parsed.info.sample_rate;
            output_info->bits_per_sample = 16;
            output_info->frames = parsed.info.frames;
            output_info->duration_seconds = parsed.info.duration_seconds;
            output_info->pcm16_mono = true;
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

}  // namespace mtts_audio
