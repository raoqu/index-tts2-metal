#include "mit2/allocator.hpp"
#include "mit2/text_frontend.hpp"
#include "audio.hpp"
#include <set>
#include <cstdlib>
#include <memory>
#include <functional>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include "mit2/bundle.hpp"
#include "mit2/metal_context.hpp"

#include <iomanip>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>

// Set by main() when --verbose is passed; long-lived modes (e.g. --server)
// read it to decide whether stage JSON streams to the console.
static bool g_cli_verbose = false;
#include <utility>
#include <vector>

#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>

#include "impl/server.hpp"

namespace {

using Clock = std::chrono::steady_clock;
// Must match the threadgroup scores[] capacity in mit2_dit_attention_qkv_rope*_f32.
constexpr uint32_t kFusedDitAttentionMaxTokens = 4096;
constexpr uint32_t kCfmSteps = 25;           // golden tests (goldens were generated at 25)
// Production synthesis steps. 16 euler steps ≈ 1.55x faster acoustic with
// near-identical quality. Priority: --cfm_steps CLI > MIT2_CFM_STEPS env > 16.
uint32_t g_cfm_steps_override = 0;
// 0 = use the voice bundle's full s2mel prompt (629 tokens for qin). Smaller
// values truncate the CFM voice prompt — large acoustic savings for short
// segments, at the cost of weaker timbre conditioning (quality knob).
uint32_t g_prompt_tokens_override = 0;
// LRU capacity for the per-voice conditioning cache (conformer/perceiver
// frontend output keyed by voice bundle dir). Repeated requests for a cached
// voice skip ~0.25s of frontend per segment. 0 disables caching.
// Override: --lrucache N CLI > MIT2_LRU_CACHE env > 3.
uint32_t g_voice_cond_cache_size = 3;
inline uint32_t cfm_synthesis_steps() {
    if (g_cfm_steps_override != 0) {
        return g_cfm_steps_override;
    }
    static const uint32_t steps = []() -> uint32_t {
        if (const char* v = std::getenv("MIT2_CFM_STEPS")) {
            const long n = std::strtol(v, nullptr, 10);
            if (n >= 1 && n <= 200) {
                return static_cast<uint32_t>(n);
            }
        }
        return 16;
    }();
    return steps;
}
constexpr uint32_t kMaxCodesPerTextToken = 15;

std::string json_escape(const std::string& value);

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void print_nullable_double_json_field(const std::string& name,
                                      double value,
                                      bool available,
                                      bool trailing_comma = true) {
    std::cout << "  \"" << name << "\": ";
    if (available) {
        std::cout << value;
    } else {
        std::cout << "null";
    }
    if (trailing_comma) {
        std::cout << ",\n";
    } else {
        std::cout << "\n";
    }
}

void print_gpt_decode_rate_json(uint32_t raw_codes,
                                size_t output_codes,
                                double elapsed_seconds,
                                bool trailing_comma = true) {
    const bool raw_available = elapsed_seconds > 0.0 && raw_codes > 0;
    const bool output_available = elapsed_seconds > 0.0 && output_codes > 0;
    print_nullable_double_json_field(
        "raw_codes_per_second",
        raw_available ? static_cast<double>(raw_codes) / elapsed_seconds : 0.0,
        raw_available);
    print_nullable_double_json_field(
        "seconds_per_raw_code",
        raw_available ? elapsed_seconds / static_cast<double>(raw_codes) : 0.0,
        raw_available);
    print_nullable_double_json_field(
        "codes_per_second",
        output_available ? static_cast<double>(output_codes) / elapsed_seconds : 0.0,
        output_available);
    print_nullable_double_json_field(
        "seconds_per_code",
        output_available ? elapsed_seconds / static_cast<double>(output_codes) : 0.0,
        output_available,
        trailing_comma);
}

void print_metal_resource_stats_json(const std::string& prefix,
                                     const mit2::MetalResourceStats& stats,
                                     bool trailing_comma = true) {
    std::cout << "  \"" << prefix << "command_buffers_submitted\": " << stats.command_buffers_submitted << ",\n";
    std::cout << "  \"" << prefix << "buffer_allocations\": " << stats.buffer_allocations << ",\n";
    std::cout << "  \"" << prefix << "buffer_bytes_allocated\": " << stats.buffer_bytes_allocated << ",\n";
    std::cout << "  \"" << prefix << "gpu_elapsed_seconds\": " << stats.gpu_elapsed_seconds;
    if (trailing_comma) {
        std::cout << ",\n";
    } else {
        std::cout << "\n";
    }
}

void usage(const char* argv0) {
#ifdef MIT2_TTS_LAUNCHER
    std::cerr << "usage: " << argv0 << " MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT OUTPUT_WAV [PRESET]\n"
              << "       (--tts/--clone are quiet by default; add --verbose for stage-by-stage JSON reports)\n"
              << "       " << argv0 << " --capabilities\n"
              << "       " << argv0 << " --readiness MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR\n"
              << "       " << argv0 << " --preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT\n"
              << "       " << argv0 << " --clone AUDIO_WAV OUTPUT_VOICE_BUNDLE\n"
              << "       " << argv0 << " --clone MODEL_BUNDLE_DIR AUDIO_WAV OUTPUT_VOICE_BUNDLE\n"
              << "       " << argv0 << " --web [--webkey KEY] [--host HOST] [--port PORT] [--model_bundle DIR] [--voice_store DIR] [--lrucache N]\n"
              << "       (--lrucache N: voice-conditioning LRU capacity, default 3, 0 disables; env MIT2_LRU_CACHE)\n"
              << "       " << argv0 << " --clone-real MODEL_BUNDLE_DIR FEATURE_MANIFEST W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 OUTPUT_VOICE_BUNDLE\n"
              << "       " << argv0 << " --clone-preflight AUDIO_WAV\n"
              << "       " << argv0 << " --clone-preprocess AUDIO_WAV OUTPUT_F32\n"
              << "       " << argv0 << " --clone-readiness PREPROCESS_MANIFEST\n"
              << "       " << argv0 << " --clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32\n"
              << "       " << argv0 << " --clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32\n"
              << "       " << argv0 << " --clone-prepare-features AUDIO_WAV OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-feature-readiness FEATURE_MANIFEST\n"
              << "       " << argv0 << " --clone-encoder-model-readiness MODEL_BUNDLE_DIR\n"
              << "       " << argv0 << " --clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST S2MEL_STYLE_F32\n"
              << "       " << argv0 << " --clone-campplus-style-from-features MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32\n"
              << "       " << argv0 << " --clone-campplus-head-golden MODEL_BUNDLE_DIR FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR\n"
              << "       " << argv0 << " --clone-w2v-feature-project MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32\n"
              << "       " << argv0 << " --clone-w2v-encoder MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_SPK_COND_F3\n"
              << "       " << argv0 << " --clone-w2v-extract-features PREPROCESS_MANIFEST OUTPUT_FEATURES_F32 OUTPUT_MASK_U32\n"
              << "       " << argv0 << " --clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32 W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-qkv MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32 W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32 W2V_TOKENS OUTPUT_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-ffn1-activate W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_ACTIVATED_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32 W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-qkv MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer1-attention W2V_LAYER1_Q_F32 W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER1_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-attention-residual W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER1_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_INTERMEDIATE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-ffn1-activate W2V_LAYER2_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_ACTIVATED_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_OUTPUT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-ffn1-residual W2V_LAYER1_F32 W2V_LAYER2_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-qkv MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer2-attention W2V_LAYER2_Q_F32 W2V_LAYER2_K_F32 W2V_LAYER2_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER2_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-attention-residual W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER2_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-conv-residual MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_LAYER2_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer2-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER2_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_INTERMEDIATE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-ffn1-activate W2V_LAYER3_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_ACTIVATED_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_OUTPUT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-ffn1-residual W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_LAYER3_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-qkv MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer3-attention W2V_LAYER3_Q_F32 W2V_LAYER3_K_F32 W2V_LAYER3_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER3_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-attention-residual W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-ffn1-activate W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_ACTIVATED_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32 W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-qkv MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32 W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER4_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-attention-residual W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER4_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-conv-residual MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_LAYER4_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer4-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER4_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER4_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-qkv MODEL_BUNDLE_DIR W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer5-attention W2V_LAYER5_Q_F32 W2V_LAYER5_K_F32 W2V_LAYER5_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER5_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-attention-project MODEL_BUNDLE_DIR W2V_LAYER5_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-attention-residual W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_LAYER5_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-attention-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-conv-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-conv-glu MODEL_BUNDLE_DIR W2V_LAYER5_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER5_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-conv-residual MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_LAYER5_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer5-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER5_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER5_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-qkv MODEL_BUNDLE_DIR W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer6-attention W2V_LAYER6_Q_F32 W2V_LAYER6_K_F32 W2V_LAYER6_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER6_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-attention-project MODEL_BUNDLE_DIR W2V_LAYER6_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-attention-residual W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_LAYER6_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-attention-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-conv-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-conv-glu MODEL_BUNDLE_DIR W2V_LAYER6_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER6_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-conv-residual MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_LAYER6_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer6-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER6_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER6_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-qkv MODEL_BUNDLE_DIR W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer7-attention W2V_LAYER7_Q_F32 W2V_LAYER7_K_F32 W2V_LAYER7_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER7_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-attention-project MODEL_BUNDLE_DIR W2V_LAYER7_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-attention-residual W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_LAYER7_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-attention-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-conv-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-conv-glu MODEL_BUNDLE_DIR W2V_LAYER7_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER7_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-conv-residual MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_LAYER7_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer7-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER7_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER7_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-qkv MODEL_BUNDLE_DIR W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer8-attention W2V_LAYER8_Q_F32 W2V_LAYER8_K_F32 W2V_LAYER8_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER8_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-attention-project MODEL_BUNDLE_DIR W2V_LAYER8_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-attention-residual W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_LAYER8_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-attention-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-conv-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-conv-glu MODEL_BUNDLE_DIR W2V_LAYER8_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER8_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-conv-residual MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_LAYER8_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer8-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER8_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER8_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-qkv MODEL_BUNDLE_DIR W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer9-attention W2V_LAYER9_Q_F32 W2V_LAYER9_K_F32 W2V_LAYER9_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER9_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-attention-project MODEL_BUNDLE_DIR W2V_LAYER9_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-attention-residual W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_LAYER9_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-attention-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-conv-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-conv-glu MODEL_BUNDLE_DIR W2V_LAYER9_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER9_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-conv-residual MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_LAYER9_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer9-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER9_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER9_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-qkv MODEL_BUNDLE_DIR W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer10-attention W2V_LAYER10_Q_F32 W2V_LAYER10_K_F32 W2V_LAYER10_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER10_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-attention-project MODEL_BUNDLE_DIR W2V_LAYER10_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-attention-residual W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_LAYER10_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-attention-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-conv-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-conv-glu MODEL_BUNDLE_DIR W2V_LAYER10_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER10_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-conv-residual MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_LAYER10_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer10-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER10_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER10_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-qkv MODEL_BUNDLE_DIR W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer11-attention W2V_LAYER11_Q_F32 W2V_LAYER11_K_F32 W2V_LAYER11_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER11_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-attention-project MODEL_BUNDLE_DIR W2V_LAYER11_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-attention-residual W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_LAYER11_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-attention-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-conv-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-conv-glu MODEL_BUNDLE_DIR W2V_LAYER11_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER11_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-conv-residual MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_LAYER11_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer11-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER11_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER11_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-qkv MODEL_BUNDLE_DIR W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer12-attention W2V_LAYER12_Q_F32 W2V_LAYER12_K_F32 W2V_LAYER12_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER12_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-attention-project MODEL_BUNDLE_DIR W2V_LAYER12_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-attention-residual W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_LAYER12_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-attention-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-conv-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-conv-glu MODEL_BUNDLE_DIR W2V_LAYER12_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER12_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-conv-residual MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_LAYER12_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer12-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER12_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER12_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-qkv MODEL_BUNDLE_DIR W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer13-attention W2V_LAYER13_Q_F32 W2V_LAYER13_K_F32 W2V_LAYER13_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER13_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-attention-project MODEL_BUNDLE_DIR W2V_LAYER13_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-attention-residual W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_LAYER13_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-attention-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-conv-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-conv-glu MODEL_BUNDLE_DIR W2V_LAYER13_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER13_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-conv-residual MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_LAYER13_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer13-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER13_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN2_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER13_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN1_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER14_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN2_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer14-qkv MODEL_BUNDLE_DIR W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-w2v-layer14-attention W2V_LAYER14_Q_F32 W2V_LAYER14_K_F32 W2V_LAYER14_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER14_CONTEXT_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-attention-project MODEL_BUNDLE_DIR W2V_LAYER14_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-attention-residual W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_LAYER14_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-attention-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-conv-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_NORM_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-conv-glu MODEL_BUNDLE_DIR W2V_LAYER14_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_GLU_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER14_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_DEPTHWISE_F32\n"
              << "       " << argv0 << " --clone-w2v-layer14-conv-residual MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_LAYER14_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_RESIDUAL_F32\n"
              << "       " << argv0 << " --clone-w2v-layer15-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER14_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN1_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-qkv MODEL_BUNDLE_DIR W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DI\n"
              << "       " << argv0 << " --clone-w2v-layer15-attention W2V_LAYER15_Q_F32 W2V_LAYER15_K_F32 W2V_LAYER15_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER15_CONTEXT_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-attention-project MODEL_BUNDLE_DIR W2V_LAYER15_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-attention-residual W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_LAYER15_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-attention-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_NORM_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-conv-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_NORM_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-conv-glu MODEL_BUNDLE_DIR W2V_LAYER15_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_GLU_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER15_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_DEPTHWISE_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-conv-residual MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_LAYER15_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer15-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER15_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN2_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER15_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN1_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-qkv MODEL_BUNDLE_DIR W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DI\n"
              << "       " << argv0 << " --clone-w2v-layer16-attention W2V_LAYER16_Q_F32 W2V_LAYER16_K_F32 W2V_LAYER16_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER16_CONTEXT_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-attention-project MODEL_BUNDLE_DIR W2V_LAYER16_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-attention-residual W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_LAYER16_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-attention-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_NORM_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-conv-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_NORM_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-conv-glu MODEL_BUNDLE_DIR W2V_LAYER16_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_GLU_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER16_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_DEPTHWISE_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-conv-residual MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_LAYER16_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer16-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER16_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN2_RESIDUAL_F3\n"
              << "       " << argv0 << " --clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32 W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32\n"
              << "       " << argv0 << " --clone-w2v-normalize MODEL_BUNDLE_DIR W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32\n"
              << "       " << argv0 << " --clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32 SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32\n"
              << "       " << argv0 << " --clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS OUTPUT_DIR\n"
              << "       " << argv0 << " --clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32\n"
              << "       " << argv0 << " --clone-encoder-readiness FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32\n"
              << "       " << argv0 << " --clone-write-voice-bundle PREPROCESS_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS MEL_F32 OUTPUT_VOICE_BUNDLE\n"
              << "       " << argv0 << " --clone-write-voice-bundle-from-features FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS OUTPUT_VOICE_BUNDLE\n"
              << "       " << argv0 << " --text-readiness MODEL_BUNDLE_DIR TEXT\n\n"
              << "PRESET is one of: smoke, short, standard. The default is standard.\n"
              << "This launcher currently targets cached-voice native CJK TTS. General text\n"
              << "normalization and fresh native voice cloning remain explicit readiness gaps.\n";
#else
    std::cerr << "usage: " << argv0
              << " [--inspect-bundle DIR] [--inspect-model-bundle DIR] [--inspect-voice-bundle DIR] [--test-gpt-layer DIR] [--test-vq2emb DIR]"
              << " [--test-gpt-prepare-inputs DIR] [--test-gpt-layer0-qkv DIR] [--test-gpt-layer0-attn DIR]"
              << " [--test-gpt-layer0-kv-attn DIR] [--test-gpt-layer0-block DIR]"
              << " [--test-gpt-transformer-stack DIR] [--test-gpt-logits DIR]"
              << " [--test-gpt-kv-decode DIR] [--test-gpt-greedy DIR] [--test-gpt-kv-greedy DIR]"
              << " [--test-gpt-kv-greedy-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-sampling-processors]"
              << " [--export-text-ids-cjk TOKENIZER_DIR TEXT OUTPUT_TEXT_IDS_U32]"
              << " [--export-text-ids-cjk-segments TOKENIZER_DIR TEXT MAX_TOKENS OUTPUT_DIR]"
              << " [--tokenize-cjk-smoke TOKENIZER_DIR TEXT OUTPUT_TEXT_IDS_U32]"
              << " [--test-gpt-subsampling-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-subsampling-metal-linear-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--export-gpt-subsampling BUNDLE_DIR SPK_COND_EMB_F32 OUTPUT_STACK_F32 OUTPUT_POS_EMB_F32 OUTPUT_MASK_U32]"
              << " [--test-gpt-emovec-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-emovec-metal-linear-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--export-gpt-emovec BUNDLE_DIR SPK_COND_EMB_F32 OUTPUT_EMOVEC_F32]"
              << " [--test-gpt-conformer-ff-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-ff-metal-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-attn-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-attn-metal-proj-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-conv-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-conv-metal-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-block-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-block-metal-ff-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-block-metal-attn-ff-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-block-metal-attn-conv-ff-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-stack-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-stack-metal-ff-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-stack-metal-attn-ff-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-gpt-conformer-stack-metal-attn-conv-ff-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--export-gpt-conformer-stack BUNDLE_DIR STACK_INPUT_F32 POS_EMB_F32 MASK_U32 OUTPUT_CONTEXT_F32]"
              << " [--test-gpt-perceiver-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--export-gpt-perceiver BUNDLE_DIR CONTEXT_F32 MASK_U32 OUTPUT_F32]"
              << " [--export-gpt-frontend-tail BUNDLE_DIR SPEECH_COND_F32 EMOVEC_F32 TEXT_IDS_U32 OUTPUT_CONDS_F32 OUTPUT_FAKE_U32 OUTPUT_INPUTS_F32 OUTPUT_MASK_U32]"
              << " [--export-gpt-kv-codes-golden BUNDLE_DIR GOLDEN_DIR OUTPUT_CODES_U32]"
              << " [--export-gpt-kv-codes-inputs BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES OUTPUT_CODES_U32]"
              << " [--test-gpt-kv-codes-inputs BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES EXPECTED_CODES_U32]"
              << " [--export-gpt-kv-codes-inputs-sampled BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES SEED TEMPERATURE TOP_K TOP_P REPETITION_PENALTY OUTPUT_CODES_U32]"
              << " [--test-gpt-sampled-inputs-determinism BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES]"
              << " [--test-gpt-latent-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--trace-gpt-latent-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-length-regulator-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--export-length-regulator-golden BUNDLE_DIR GOLDEN_DIR OUTPUT_F32]"
              << " [--export-length-regulator-stages-golden BUNDLE_DIR GOLDEN_DIR OUTPUT_DIR]"
              << " [--test-length-regulator-front DIR] [--test-length-regulator-full DIR]"
              << " [--test-timestep-embedder DIR] [--test-dit-input-merge DIR] [--test-dit-attention-proj DIR]"
              << " [--test-dit-feed-forward DIR] [--test-dit-adaptive-norm DIR] [--test-dit-attention-core DIR]"
              << " [--test-dit-transformer-block0 DIR] [--test-dit-transformer-stack DIR]"
              << " [--test-dit-post-transformer-proj DIR] [--test-dit-final-layer DIR]"
              << " [--test-wavenet-layer0-gate DIR] [--test-wavenet-stack DIR]"
              << " [--test-dit-estimator-step DIR] [--test-dit-estimator-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-cfm-euler DIR] [--test-cfm-euler-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-cfm-euler-cfg DIR] [--test-cfm-euler-cfg-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-s2mel-full-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-s2mel-full-inputs BUNDLE_DIR NOISE_F32 PROMPT_MEL_F32 CONDITION_F32 STYLE_F32 STEPS CFG_RATE EXPECTED_FULL_F32]"
              << " [--trace-s2mel-cfm-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--trace-s2mel-cfm-error-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--test-hot-tts-golden BUNDLE_DIR S2MEL_GOLDEN_DIR WAVE_GOLDEN_DIR]"
              << " [--test-hot-tts-from-gpt-golden BUNDLE_DIR VOICE_BUNDLE_DIR GPT_GOLDEN_DIR S2MEL_GOLDEN_DIR WAVE_GOLDEN_DIR]"
              << " [--test-hot-tts-from-codes-golden BUNDLE_DIR VOICE_BUNDLE_DIR GPT_GOLDEN_DIR S2MEL_GOLDEN_DIR WAVE_GOLDEN_DIR]"
              << " [--synthesize-hot-gpt-golden BUNDLE_DIR VOICE_BUNDLE_DIR GPT_GOLDEN_DIR S2MEL_GOLDEN_DIR OUTPUT_WAV]"
              << " [--export-hot-codes-condition-golden BUNDLE_DIR VOICE_BUNDLE_DIR GPT_GOLDEN_DIR S2MEL_GOLDEN_DIR OUTPUT_CONDITION_F32]"
              << " [--export-hot-codes-condition-input BUNDLE_DIR VOICE_BUNDLE_DIR GPT_GOLDEN_DIR S2MEL_GOLDEN_DIR CODES_U32 OUTPUT_CONDITION_F32]"
              << " [--export-hot-codes-condition-inputs BUNDLE_DIR VOICE_BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 CODES_U32 PROMPT_TOKENS OUTPUT_CONDITION_F32]"
              << " [--test-hot-tts-condition-golden BUNDLE_DIR VOICE_BUNDLE_DIR S2MEL_GOLDEN_DIR WAVE_GOLDEN_DIR CONDITION_F32]"
              << " [--synthesize-hot-condition-golden BUNDLE_DIR VOICE_BUNDLE_DIR S2MEL_GOLDEN_DIR CONDITION_F32 OUTPUT_WAV]"
              << " [--synthesize-hot-condition-inputs BUNDLE_DIR VOICE_BUNDLE_DIR CONDITION_F32 NOISE_F32 PROMPT_TOKENS STEPS CFG_RATE OUTPUT_WAV]"
              << " [--cfm_steps N(12-25)]"
              << " [--synthesize-hot-inputs BUNDLE_DIR VOICE_BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES NOISE_F32 PROMPT_TOKENS STEPS CFG_RATE OUTPUT_WAV]"
              << " [--synthesize-hot-inputs-seeded BUNDLE_DIR VOICE_BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES SEED TEMPERATURE PROMPT_TOKENS STEPS CFG_RATE OUTPUT_WAV]"
              << " [--synthesize-hot-text-cjk-seeded BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_CODES SEED TEMPERATURE PROMPT_TOKENS_OR_0 STEPS CFG_RATE OUTPUT_WAV]"
              << " [--synthesize-hot-text-cjk-sampled-seeded BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_CODES GPT_SEED GPT_TEMPERATURE GPT_TOP_K GPT_TOP_P GPT_REPETITION_PENALTY NOISE_SEED NOISE_TEMPERATURE PROMPT_TOKENS_OR_0 STEPS CFG_RATE OUTPUT_WAV]"
              << " [--synthesize-hot-text-cjk-segments-seeded BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_TEXT_TOKENS_PER_SEGMENT MAX_CODES SEED TEMPERATURE PROMPT_TOKENS_OR_0 STEPS CFG_RATE INTERVAL_SILENCE_MS OUTPUT_WAV]"
              << " [--synthesize-hot-text-cjk-segments-sampled-seeded BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_TEXT_TOKENS_PER_SEGMENT MAX_CODES GPT_SEED GPT_TEMPERATURE GPT_TOP_K GPT_TOP_P GPT_REPETITION_PENALTY NOISE_SEED NOISE_TEMPERATURE PROMPT_TOKENS_OR_0 STEPS CFG_RATE INTERVAL_SILENCE_MS OUTPUT_WAV]"
              << " [--tts-validate-bundles]"
              << " [--tts-cjk BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_CODES STEPS OUTPUT_WAV]"
              << " [--tts-cjk-segments BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_TEXT_TOKENS_PER_SEGMENT MAX_CODES STEPS INTERVAL_SILENCE_MS OUTPUT_WAV]"
              << " [--tts-cjk-sampled BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_CODES STEPS OUTPUT_WAV]"
              << " [--tts-cjk-segments-sampled BUNDLE_DIR VOICE_BUNDLE_DIR TEXT MAX_TEXT_TOKENS_PER_SEGMENT MAX_CODES STEPS INTERVAL_SILENCE_MS OUTPUT_WAV]"
              << " [--tts-cjk-preset BUNDLE_DIR VOICE_BUNDLE_DIR TEXT PRESET OUTPUT_WAV]"
              << " [--tts-cjk-segments-preset BUNDLE_DIR VOICE_BUNDLE_DIR TEXT PRESET OUTPUT_WAV]"
              << " [--tts-cjk-auto-preset BUNDLE_DIR VOICE_BUNDLE_DIR TEXT PRESET OUTPUT_WAV]"
              << " [--clone AUDIO_WAV OUTPUT_VOICE_BUNDLE]"
              << " [--clone MODEL_BUNDLE_DIR AUDIO_WAV OUTPUT_VOICE_BUNDLE]"
              << " [--tts BUNDLE_DIR VOICE_BUNDLE_DIR TEXT OUTPUT_WAV]"
              << " [--server --model_bundle DIR --voice_store DIR --queue_size N --tts_concurrency N --clone_concurrency N]"
              << " [--web --webkey KEY]"
              << " [--tts-cjk-text-readiness BUNDLE_DIR TEXT]"
              << " [--tts-product-readiness BUNDLE_DIR VOICE_BUNDLE_DIR]"
              << " [--synthesize-hot-inputs-sampled-seeded BUNDLE_DIR VOICE_BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES GPT_SEED GPT_TEMPERATURE GPT_TOP_K GPT_TOP_P GPT_REPETITION_PENALTY NOISE_SEED NOISE_TEMPERATURE PROMPT_TOKENS STEPS CFG_RATE OUTPUT_WAV]"
              << " [--synthesize-hot-inputs-seeded-shared BUNDLE_DIR VOICE_BUNDLE_DIR CONDS_F32 TEXT_IDS_U32 MAX_CODES SEED TEMPERATURE PROMPT_TOKENS STEPS CFG_RATE OUTPUT_WAV]"
              << " [--synthesize-hot-native-golden BUNDLE_DIR VOICE_BUNDLE_DIR GPT_GOLDEN_DIR S2MEL_GOLDEN_DIR OUTPUT_WAV]"
              << " [--test-bigvgan-conv-pre DIR] [--test-bigvgan-upsample0 DIR] [--test-bigvgan-upsampler-stack DIR]"
              << " [--test-bigvgan-front DIR] [--test-bigvgan-activation-post DIR] [--test-bigvgan-activation-rb0 DIR]"
              << " [--test-bigvgan-resblock0-pair0 DIR] [--test-bigvgan-resblock0 DIR]"
              << " [--test-bigvgan-resblock-group0 DIR] [--test-bigvgan-body DIR]"
              << " [--test-bigvgan-post DIR] [--test-bigvgan-vocoder DIR]"
              << " [--test-bigvgan-vocoder-golden BUNDLE_DIR GOLDEN_DIR]"
              << " [--plan-hot-scratch MAX_PREFIX_TOKENS MAX_CODES PROMPT_TOKENS]"
              << " [--plan-hot-scratch-inputs CONDS_F32 TEXT_IDS_U32 MAX_CODES PROMPT_TOKENS]"
              << " [--test-text-ids-cjk-version-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-slash-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-temperature-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-plus-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-operator-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-measure-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-date-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-time-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-currency-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-phone-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-ratio-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-fraction-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-quote-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-ellipsis-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-percent-tokenizer TOKENIZER_DIR]"
              << " [--test-text-ids-cjk-no-tokenizer TOKENIZER_DIR]"
              << " [--test-scratch-allocator] [--test-metal-scratch-arena] [--smoke-copy] [--test-primitives] [--diagnostics]\n";
#endif
}

size_t checked_mul_size(size_t a, size_t b, const std::string& name) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(name + " size overflow");
    }
    return a * b;
}

size_t checked_add_size(size_t a, size_t b, const std::string& name) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error(name + " size overflow");
    }
    return a + b;
}

size_t tensor_bytes(size_t elements, size_t element_size, const std::string& name) {
    return checked_mul_size(elements, element_size, name);
}

struct ScratchItem {
    std::string name;
    size_t bytes = 0;
};

struct HotScratchPlan {
    bool ok = false;
    size_t alignment = 0;
    uint32_t max_prefix_tokens = 0;
    uint32_t max_codes = 0;
    uint32_t prompt_tokens = 0;
    std::string source;
    uint32_t cond_tokens = 0;
    uint32_t text_tokens = 0;
    uint32_t generated_tokens = 0;
    uint32_t total_mel_tokens = 0;
    size_t gpt_kv_cache_bytes = 0;
    size_t gpt_phase_peak_bytes = 0;
    size_t condition_tensor_bytes = 0;
    size_t condition_phase_peak_bytes = 0;
    size_t acoustic_mel_tensor_bytes = 0;
    size_t acoustic_phase_peak_bytes = 0;
    size_t unshared_phase_peak_total_bytes = 0;
    size_t planned_scratch_capacity_bytes = 0;
    size_t scratch_reuse_saves_bytes = 0;
};

std::vector<float> read_raw_f32(const std::string& path);
std::vector<uint32_t> read_raw_u32(const std::string& path);

size_t plan_scratch_phase(const std::vector<ScratchItem>& items, size_t alignment) {
    size_t capacity = 0;
    for (const auto& item : items) {
        capacity = checked_add_size(capacity, item.bytes, item.name);
        capacity = checked_add_size(capacity, alignment, item.name + "_alignment");
    }
    mit2::ScratchAllocator scratch(capacity, alignment);
    for (const auto& item : items) {
        scratch.allocate(item.name, item.bytes);
    }
    return scratch.peak();
}

HotScratchPlan compute_hot_scratch_plan(uint32_t max_prefix_tokens,
                                        uint32_t max_codes,
                                        uint32_t prompt_tokens,
                                        const std::string& source = "manual",
                                        uint32_t cond_tokens = 0,
                                        uint32_t text_tokens = 0) {
    if (max_prefix_tokens == 0 || max_codes == 0 || prompt_tokens == 0) {
        throw std::invalid_argument("hot scratch plan requires positive token counts");
    }
    constexpr size_t alignment = 256;
    constexpr size_t f32 = sizeof(float);
    constexpr size_t u32 = sizeof(uint32_t);
    constexpr size_t gpt_layers = 24;
    constexpr size_t gpt_hidden = 1280;
    constexpr size_t gpt_vocab = 8194;
    constexpr size_t s2mel_width = 512;
    constexpr size_t mel_bins = 80;
    constexpr size_t dit_merge_width = 864;
    constexpr size_t max_bigvgan_channels = 512;

    const uint32_t generated_tokens = static_cast<uint32_t>(std::floor(static_cast<float>(max_codes) * 1.72f));
    const uint32_t total_mel_tokens = prompt_tokens + generated_tokens;
    const size_t gpt_cache_tokens = static_cast<size_t>(max_prefix_tokens) + max_codes;

    const size_t gpt_kv_cache_bytes = tensor_bytes(
        checked_mul_size(checked_mul_size(gpt_layers, 2, "gpt_kv_layers"), checked_mul_size(gpt_cache_tokens, gpt_hidden, "gpt_kv_tokens"), "gpt_kv"),
        f32,
        "gpt_kv_cache");
    const size_t gpt_logits_bytes = tensor_bytes(gpt_vocab, f32, "gpt_logits");
    const size_t gpt_codes_bytes = tensor_bytes(max_codes, u32, "gpt_codes");
    const std::vector<ScratchItem> gpt_items{
        {"gpt_kv_cache_f32", gpt_kv_cache_bytes},
        {"gpt_logits_f32", gpt_logits_bytes},
        {"gpt_codes_u32", gpt_codes_bytes},
    };

    const size_t gpt_latent_bytes = tensor_bytes(checked_mul_size(max_codes, gpt_hidden, "gpt_latent"), f32, "gpt_latent");
    const size_t s2mel_token_bytes = tensor_bytes(checked_mul_size(max_codes, s2mel_width, "s2mel_tokens"), f32, "s2mel_tokens");
    const size_t lr_generated_bytes = tensor_bytes(checked_mul_size(generated_tokens, s2mel_width, "length_regulator"), f32, "length_regulator");
    const size_t condition_bytes = tensor_bytes(checked_mul_size(total_mel_tokens, s2mel_width, "condition"), f32, "condition");
    const std::vector<ScratchItem> condition_items{
        {"gpt_latent_f32", gpt_latent_bytes},
        {"gpt_layer_f32", s2mel_token_bytes},
        {"vq2emb_f32", s2mel_token_bytes},
        {"s_infer_f32", s2mel_token_bytes},
        {"length_regulator_f32", lr_generated_bytes},
        {"condition_f32", condition_bytes},
    };

    const size_t mel_tensor_bytes = tensor_bytes(checked_mul_size(total_mel_tokens, mel_bins, "mel_tokens"), f32, "mel_tokens");
    const size_t dit_merge_bytes = tensor_bytes(checked_mul_size(total_mel_tokens, dit_merge_width, "dit_merge"), f32, "dit_merge");
    const size_t dit_hidden_bytes = tensor_bytes(checked_mul_size(total_mel_tokens, s2mel_width, "dit_hidden"), f32, "dit_hidden");
    const size_t dit_cfg_hidden_bytes = checked_mul_size(2, dit_hidden_bytes, "dit_cfg_hidden");
    const size_t bigvgan_feature_bytes = tensor_bytes(
        checked_mul_size(checked_mul_size(generated_tokens, 256, "bigvgan_samples"), max_bigvgan_channels, "bigvgan_feature"),
        f32,
        "bigvgan_feature");
    const std::vector<ScratchItem> acoustic_items{
        {"noise_f32", mel_tensor_bytes},
        {"full_mel_f32", mel_tensor_bytes},
        {"dphi_f32", mel_tensor_bytes},
        {"cfg_dphi_f32", mel_tensor_bytes},
        {"dit_input_merge_f32", dit_merge_bytes},
        {"dit_cfg_hidden_f32", dit_cfg_hidden_bytes},
        {"wavenet_skip_f32", dit_hidden_bytes},
        {"bigvgan_feature_f32", bigvgan_feature_bytes},
    };

    const size_t gpt_peak = plan_scratch_phase(gpt_items, alignment);
    const size_t condition_peak = plan_scratch_phase(condition_items, alignment);
    const size_t acoustic_peak = plan_scratch_phase(acoustic_items, alignment);
    const size_t planned_capacity = std::max(gpt_peak, std::max(condition_peak, acoustic_peak));
    const size_t unshared_total = checked_add_size(checked_add_size(gpt_peak, condition_peak, "phase_total"), acoustic_peak, "phase_total");
    const size_t reuse_saves = unshared_total - planned_capacity;
    const bool ok = planned_capacity > 0 && reuse_saves > 0 && total_mel_tokens > prompt_tokens;

    HotScratchPlan plan;
    plan.ok = ok;
    plan.alignment = alignment;
    plan.max_prefix_tokens = max_prefix_tokens;
    plan.max_codes = max_codes;
    plan.prompt_tokens = prompt_tokens;
    plan.source = source;
    plan.cond_tokens = cond_tokens;
    plan.text_tokens = text_tokens;
    plan.generated_tokens = generated_tokens;
    plan.total_mel_tokens = total_mel_tokens;
    plan.gpt_kv_cache_bytes = gpt_kv_cache_bytes;
    plan.gpt_phase_peak_bytes = gpt_peak;
    plan.condition_tensor_bytes = condition_bytes;
    plan.condition_phase_peak_bytes = condition_peak;
    plan.acoustic_mel_tensor_bytes = mel_tensor_bytes;
    plan.acoustic_phase_peak_bytes = acoustic_peak;
    plan.unshared_phase_peak_total_bytes = unshared_total;
    plan.planned_scratch_capacity_bytes = planned_capacity;
    plan.scratch_reuse_saves_bytes = reuse_saves;
    return plan;
}

void print_hot_scratch_plan_fields_json(const HotScratchPlan& plan,
                                        const std::string& prefix,
                                        bool include_source_tokens,
                                        bool trailing_comma = true) {
    std::cout << "  \"" << prefix << "ok\": " << (plan.ok ? "true" : "false") << ",\n";
    std::cout << "  \"" << prefix << "alignment\": " << plan.alignment << ",\n";
    std::cout << "  \"" << prefix << "max_prefix_tokens\": " << plan.max_prefix_tokens << ",\n";
    std::cout << "  \"" << prefix << "max_codes\": " << plan.max_codes << ",\n";
    std::cout << "  \"" << prefix << "prompt_tokens\": " << plan.prompt_tokens << ",\n";
    std::cout << "  \"" << prefix << "source\": \"" << plan.source << "\",\n";
    if (include_source_tokens || plan.cond_tokens > 0 || plan.text_tokens > 0) {
        std::cout << "  \"" << prefix << "cond_tokens\": " << plan.cond_tokens << ",\n";
        std::cout << "  \"" << prefix << "text_tokens\": " << plan.text_tokens << ",\n";
    }
    std::cout << "  \"" << prefix << "generated_tokens\": " << plan.generated_tokens << ",\n";
    std::cout << "  \"" << prefix << "total_mel_tokens\": " << plan.total_mel_tokens << ",\n";
    std::cout << "  \"" << prefix << "gpt_kv_cache_bytes\": " << plan.gpt_kv_cache_bytes << ",\n";
    std::cout << "  \"" << prefix << "gpt_phase_peak_bytes\": " << plan.gpt_phase_peak_bytes << ",\n";
    std::cout << "  \"" << prefix << "condition_tensor_bytes\": " << plan.condition_tensor_bytes << ",\n";
    std::cout << "  \"" << prefix << "condition_phase_peak_bytes\": " << plan.condition_phase_peak_bytes << ",\n";
    std::cout << "  \"" << prefix << "acoustic_mel_tensor_bytes\": " << plan.acoustic_mel_tensor_bytes << ",\n";
    std::cout << "  \"" << prefix << "acoustic_phase_peak_bytes\": " << plan.acoustic_phase_peak_bytes << ",\n";
    const std::string capacity_key = prefix.empty() ? "planned_scratch_capacity_bytes" : prefix + "capacity_bytes";
    const std::string reuse_key = prefix.empty() ? "scratch_reuse_saves_bytes" : prefix + "reuse_saves_bytes";
    std::cout << "  \"" << prefix << "unshared_phase_peak_total_bytes\": " << plan.unshared_phase_peak_total_bytes << ",\n";
    std::cout << "  \"" << capacity_key << "\": " << plan.planned_scratch_capacity_bytes << ",\n";
    std::cout << "  \"" << reuse_key << "\": " << plan.scratch_reuse_saves_bytes;
    if (trailing_comma) {
        std::cout << ",\n";
    } else {
        std::cout << "\n";
    }
}

void print_hot_scratch_actual_fields_json(const HotScratchPlan& plan,
                                          uint32_t actual_codes,
                                          uint32_t actual_total_mel_tokens,
                                          const std::string& prefix,
                                          bool trailing_comma = true) {
    if (actual_total_mel_tokens < plan.prompt_tokens) {
        throw std::runtime_error("hot scratch actual token count is below prompt token count");
    }
    const uint32_t actual_generated_tokens = actual_total_mel_tokens - plan.prompt_tokens;
    const bool covers_actual = actual_codes <= plan.max_codes &&
                               actual_generated_tokens <= plan.generated_tokens &&
                               actual_total_mel_tokens <= plan.total_mel_tokens;
    const uint32_t code_slack = actual_codes <= plan.max_codes ? plan.max_codes - actual_codes : 0;
    const uint32_t generated_slack = actual_generated_tokens <= plan.generated_tokens ? plan.generated_tokens - actual_generated_tokens : 0;
    const uint32_t total_slack = actual_total_mel_tokens <= plan.total_mel_tokens ? plan.total_mel_tokens - actual_total_mel_tokens : 0;
    std::cout << "  \"" << prefix << "actual_codes\": " << actual_codes << ",\n";
    std::cout << "  \"" << prefix << "actual_generated_tokens\": " << actual_generated_tokens << ",\n";
    std::cout << "  \"" << prefix << "actual_total_mel_tokens\": " << actual_total_mel_tokens << ",\n";
    std::cout << "  \"" << prefix << "code_slack\": " << code_slack << ",\n";
    std::cout << "  \"" << prefix << "generated_token_slack\": " << generated_slack << ",\n";
    std::cout << "  \"" << prefix << "total_mel_token_slack\": " << total_slack << ",\n";
    std::cout << "  \"" << prefix << "covers_actual\": " << (covers_actual ? "true" : "false");
    if (trailing_comma) {
        std::cout << ",\n";
    } else {
        std::cout << "\n";
    }
}

void print_hot_scratch_plan_object_json(const HotScratchPlan& plan, const std::string& indent) {
    std::cout << indent << "{\n";
    std::cout << indent << "  \"ok\": " << (plan.ok ? "true" : "false") << ",\n";
    std::cout << indent << "  \"source\": \"" << json_escape(plan.source) << "\",\n";
    std::cout << indent << "  \"max_prefix_tokens\": " << plan.max_prefix_tokens << ",\n";
    std::cout << indent << "  \"cond_tokens\": " << plan.cond_tokens << ",\n";
    std::cout << indent << "  \"text_tokens\": " << plan.text_tokens << ",\n";
    std::cout << indent << "  \"max_codes\": " << plan.max_codes << ",\n";
    std::cout << indent << "  \"prompt_tokens\": " << plan.prompt_tokens << ",\n";
    std::cout << indent << "  \"generated_tokens\": " << plan.generated_tokens << ",\n";
    std::cout << indent << "  \"total_mel_tokens\": " << plan.total_mel_tokens << ",\n";
    std::cout << indent << "  \"gpt_kv_cache_bytes\": " << plan.gpt_kv_cache_bytes << ",\n";
    std::cout << indent << "  \"gpt_phase_peak_bytes\": " << plan.gpt_phase_peak_bytes << ",\n";
    std::cout << indent << "  \"condition_tensor_bytes\": " << plan.condition_tensor_bytes << ",\n";
    std::cout << indent << "  \"condition_phase_peak_bytes\": " << plan.condition_phase_peak_bytes << ",\n";
    std::cout << indent << "  \"acoustic_mel_tensor_bytes\": " << plan.acoustic_mel_tensor_bytes << ",\n";
    std::cout << indent << "  \"acoustic_phase_peak_bytes\": " << plan.acoustic_phase_peak_bytes << ",\n";
    std::cout << indent << "  \"unshared_phase_peak_total_bytes\": " << plan.unshared_phase_peak_total_bytes << ",\n";
    std::cout << indent << "  \"planned_scratch_capacity_bytes\": " << plan.planned_scratch_capacity_bytes << ",\n";
    std::cout << indent << "  \"scratch_reuse_saves_bytes\": " << plan.scratch_reuse_saves_bytes << "\n";
    std::cout << indent << "}";
}

HotScratchPlan compute_hot_scratch_plan_from_inputs(const std::vector<float>& conds,
                                                    const std::vector<uint32_t>& text_ids,
                                                    uint32_t max_codes,
                                                    uint32_t prompt_tokens) {
    constexpr uint32_t width = 1280;
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("hot scratch input plan conds must have shape [tokens,1280]");
    }
    if (text_ids.empty()) {
        throw std::runtime_error("hot scratch input plan text ids must be non-empty");
    }
    const size_t cond_token_count = conds.size() / width;
    if (cond_token_count > std::numeric_limits<uint32_t>::max() ||
        text_ids.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("hot scratch input token count overflow");
    }
    const uint32_t cond_tokens = static_cast<uint32_t>(cond_token_count);
    const uint32_t text_tokens = static_cast<uint32_t>(text_ids.size());
    const size_t prefix = checked_add_size(checked_add_size(cond_tokens, text_tokens, "hot_scratch_prefix"), 2, "hot_scratch_prefix");
    if (prefix > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("hot scratch input prefix token count overflow");
    }
    return compute_hot_scratch_plan(static_cast<uint32_t>(prefix),
                                    max_codes,
                                    prompt_tokens,
                                    "inputs",
                                    cond_tokens,
                                    text_tokens);
}

HotScratchPlan compute_hot_scratch_plan_from_input_files(const std::string& conds_path,
                                                         const std::string& text_ids_path,
                                                         uint32_t max_codes,
                                                         uint32_t prompt_tokens) {
    return compute_hot_scratch_plan_from_inputs(read_raw_f32(conds_path), read_raw_u32(text_ids_path), max_codes, prompt_tokens);
}

bool run_hot_scratch_plan(uint32_t max_prefix_tokens,
                          uint32_t max_codes,
                          uint32_t prompt_tokens,
                          const std::string& source = "manual",
                          uint32_t cond_tokens = 0,
                          uint32_t text_tokens = 0) {
    const auto plan = compute_hot_scratch_plan(max_prefix_tokens, max_codes, prompt_tokens, source, cond_tokens, text_tokens);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_scratch_plan\",\n";
    print_hot_scratch_plan_fields_json(plan, "", false, false);
    std::cout << "}\n";
    return plan.ok;
}

bool run_hot_scratch_plan_from_inputs(const std::string& conds_path,
                                      const std::string& text_ids_path,
                                      uint32_t max_codes,
                                      uint32_t prompt_tokens) {
    const auto plan = compute_hot_scratch_plan_from_input_files(conds_path, text_ids_path, max_codes, prompt_tokens);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_scratch_plan\",\n";
    print_hot_scratch_plan_fields_json(plan, "", true, false);
    std::cout << "}\n";
    return plan.ok;
}

bool run_scratch_allocator_test() {
    bool invalid_alignment_rejected = false;
    try {
        mit2::ScratchAllocator bad(1024, 0);
    } catch (const std::invalid_argument&) {
        invalid_alignment_rejected = true;
    }

    bool overflow_rejected = false;
    mit2::ScratchAllocator scratch(1024, 256);
    const auto logits = scratch.allocate("logits", 10);
    const auto codes = scratch.allocate("codes", 257);
    const size_t mark = scratch.checkpoint();
    const auto temp = scratch.allocate("temp", 1);
    scratch.rewind(mark);
    const auto mel = scratch.allocate("mel", 128);
    const size_t peak_before_reset = scratch.peak();
    scratch.reset();
    const bool reset_preserves_peak = scratch.used() == 0 && scratch.allocations().empty() && scratch.peak() == peak_before_reset;
    const auto after_reset = scratch.allocate("after_reset", 512);
    try {
        scratch.allocate("too_large", 2048);
    } catch (const std::runtime_error&) {
        overflow_rejected = true;
    }

    const bool ok =
        invalid_alignment_rejected &&
        overflow_rejected &&
        scratch.capacity() == 1024 &&
        scratch.alignment() == 256 &&
        logits.offset == 0 &&
        logits.size == 10 &&
        codes.offset == 256 &&
        codes.size == 257 &&
        mark == 513 &&
        temp.offset == 768 &&
        mel.offset == 768 &&
        mel.size == 130 &&
        peak_before_reset == 896 &&
        reset_preserves_peak &&
        after_reset.offset == 0 &&
        after_reset.size == 512 &&
        scratch.used() == 512 &&
        scratch.peak() == 896;

    std::cout << "{\n";
    std::cout << "  \"stage\": \"scratch_allocator\",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"capacity\": " << scratch.capacity() << ",\n";
    std::cout << "  \"alignment\": " << scratch.alignment() << ",\n";
    std::cout << "  \"logits_offset\": " << logits.offset << ",\n";
    std::cout << "  \"codes_offset\": " << codes.offset << ",\n";
    std::cout << "  \"checkpoint\": " << mark << ",\n";
    std::cout << "  \"temp_offset\": " << temp.offset << ",\n";
    std::cout << "  \"mel_offset\": " << mel.offset << ",\n";
    std::cout << "  \"peak_before_reset\": " << peak_before_reset << ",\n";
    std::cout << "  \"used_after_reset_allocation\": " << scratch.used() << ",\n";
    std::cout << "  \"peak_after_reset_allocation\": " << scratch.peak() << ",\n";
    std::cout << "  \"invalid_alignment_rejected\": " << (invalid_alignment_rejected ? "true" : "false") << ",\n";
    std::cout << "  \"overflow_rejected\": " << (overflow_rejected ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_metal_scratch_arena_test() {
    constexpr size_t capacity = 4096;
    mit2::ScratchAllocator scratch(capacity, 256);
    const auto source = scratch.allocate("source", 6 * sizeof(float));
    const auto destination = scratch.allocate("destination", 6 * sizeof(float));
    std::vector<float> values{0.0f, 1.0f, 2.0f, 3.5f, -4.0f, 8.0f};
    mit2::MetalContext metal;
    const bool copied = metal.smoke_scratch_arena(scratch.capacity(), source.offset, destination.offset, values);
    const auto stats = metal.resource_stats();
    const bool ok =
        copied &&
        scratch.capacity() == capacity &&
        scratch.alignment() == 256 &&
        source.offset == 0 &&
        destination.offset == 256 &&
        scratch.peak() == destination.offset + destination.size &&
        stats.command_buffers_submitted == 1 &&
        stats.buffer_allocations == 2 &&
        stats.buffer_bytes_allocated == capacity + sizeof(uint32_t);

    std::cout << "{\n";
    std::cout << "  \"stage\": \"metal_scratch_arena\",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"capacity\": " << scratch.capacity() << ",\n";
    std::cout << "  \"alignment\": " << scratch.alignment() << ",\n";
    std::cout << "  \"source_offset\": " << source.offset << ",\n";
    std::cout << "  \"destination_offset\": " << destination.offset << ",\n";
    std::cout << "  \"source_size\": " << source.size << ",\n";
    std::cout << "  \"destination_size\": " << destination.size << ",\n";
    std::cout << "  \"peak\": " << scratch.peak() << ",\n";
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"copied\": " << (copied ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("max_abs_error size mismatch");
    }
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<float>::infinity();
        }
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

uint32_t count_nonfinite(const std::vector<float>& values) {
    uint32_t count = 0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            ++count;
        }
    }
    return count;
}

float finite_max_abs_value(const std::vector<float>& values) {
    float max_value = 0.0f;
    for (float value : values) {
        if (std::isfinite(value)) {
            max_value = std::max(max_value, std::abs(value));
        }
    }
    return max_value;
}

int64_t first_nonfinite_index(const std::vector<float>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

void print_tensor_stats_json(const char* name, const std::vector<float>& values, uint32_t width, bool trailing_comma) {
    const int64_t first = first_nonfinite_index(values);
    std::cout << "    {\"name\": \"" << name
              << "\", \"count\": " << values.size()
              << ", \"width\": " << width
              << ", \"nonfinite\": " << count_nonfinite(values)
              << ", \"finite_max_abs\": " << finite_max_abs_value(values)
              << ", \"first_nonfinite_index\": " << first
              << ", \"first_nonfinite_row\": " << (first >= 0 && width > 0 ? first / static_cast<int64_t>(width) : -1)
              << ", \"first_nonfinite_col\": " << (first >= 0 && width > 0 ? first % static_cast<int64_t>(width) : -1)
              << "}" << (trailing_comma ? ",\n" : "\n");
}

struct ProcessMemoryInfo {
    uint64_t resident_bytes = 0;
    uint64_t resident_peak_bytes = 0;
};

ProcessMemoryInfo process_memory_info() {
    ProcessMemoryInfo info;
    mach_task_basic_info_data_t basic{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&basic), &count);
    if (kr == KERN_SUCCESS) {
        info.resident_bytes = static_cast<uint64_t>(basic.resident_size);
        info.resident_peak_bytes = static_cast<uint64_t>(basic.resident_size_max);
    }
    return info;
}

std::vector<float> read_raw_f32(const std::string& path) {
    std::ifstream fp(path, std::ios::binary | std::ios::ate);
    if (!fp) {
        throw std::runtime_error("failed to open raw f32 file: " + path);
    }
    const std::streamsize bytes = fp.tellg();
    if (bytes < 0 || (bytes % static_cast<std::streamsize>(sizeof(float))) != 0) {
        throw std::runtime_error("raw f32 file has invalid byte length: " + path);
    }
    fp.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    if (!values.empty() && !fp.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("failed to read raw f32 file: " + path);
    }
    return values;
}

std::vector<uint32_t> read_raw_u32(const std::string& path) {
    std::ifstream fp(path, std::ios::binary | std::ios::ate);
    if (!fp) {
        throw std::runtime_error("failed to open raw u32 file: " + path);
    }
    const std::streamsize bytes = fp.tellg();
    if (bytes < 0 || (bytes % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) {
        throw std::runtime_error("raw u32 file has invalid byte length: " + path);
    }
    fp.seekg(0, std::ios::beg);
    std::vector<uint32_t> values(static_cast<size_t>(bytes) / sizeof(uint32_t));
    if (!values.empty() && !fp.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("failed to read raw u32 file: " + path);
    }
    return values;
}

std::string read_text_file(const std::string& path) {
    std::ifstream fp(path);
    if (!fp) {
        throw std::runtime_error("failed to open text file: " + path);
    }
    std::ostringstream buf;
    buf << fp.rdbuf();
    if (!fp.good() && !fp.eof()) {
        throw std::runtime_error("failed to read text file: " + path);
    }
    return buf.str();
}

void write_raw_f32(const std::string& path, const std::vector<float>& values) {
    std::ofstream fp(path, std::ios::binary);
    if (!fp) {
        throw std::runtime_error("failed to open raw f32 output file: " + path);
    }
    if (!values.empty()) {
        fp.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    if (!fp) {
        throw std::runtime_error("failed to write raw f32 output file: " + path);
    }
}

void write_raw_u32(const std::string& path, const std::vector<uint32_t>& values) {
    std::ofstream fp(path, std::ios::binary);
    if (!fp) {
        throw std::runtime_error("failed to open raw u32 output file: " + path);
    }
    if (!values.empty()) {
        fp.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(uint32_t)));
    }
    if (!fp) {
        throw std::runtime_error("failed to write raw u32 output file: " + path);
    }
}

void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream fp(path);
    if (!fp) {
        throw std::runtime_error("failed to open text output file: " + path);
    }
    fp << text;
    if (!fp) {
        throw std::runtime_error("failed to write text output file: " + path);
    }
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(hex[(ch >> 4) & 0x0f]);
                    out.push_back(hex[ch & 0x0f]);
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return out;
}


// ---------------------------------------------------------------------------
// Implementation modules.
//
// These legacy modules still share a single translation unit because they have
// many implicit ordering dependencies. The server implementation is compiled
// separately so web/API edits no longer rebuild the full model pipeline.
// ---------------------------------------------------------------------------
#include "impl/tokenizer.cpp"       // CJK tokenizer (TokenizerPieceInfo, tokenize_cjk_text, ...)
#include "impl/tokenizer_tests.cpp" // CJK tokenizer test harnesses
#include "impl/gpt.cpp"             // GPT model implementation and unit tests
#include "impl/dit_cfm_bigvgan.cpp" // DiT transformer, CFM Euler solver, BigVGAN vocoder
#include "impl/tts_synthesis.cpp"   // TTS synthesis pipeline, golden tests, export utilities

static int run_cli(int argc, char** argv) {
    try {
#ifdef MIT2_TTS_LAUNCHER
        if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) {
            usage(argv[0]);
            return argc == 1 ? 2 : 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--capabilities") {
            print_tts_launcher_capabilities_json();
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--readiness") {
            return inspect_tts_product_readiness(argv[2], argv[3]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--preflight") {
            return inspect_tts_product_preflight(argv[2], argv[3], argv[4]) ? 0 : 1;
        }
        if (argc == 4 && std::string(argv[1]) == "--clone") {
            return run_tts_clone_fast(argv[2], argv[3]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone") {
            return run_tts_clone_native(argv[2], argv[3], argv[4]) ? 0 : 1;
        }
        if (argc == 4 && std::string(argv[1]) == "--clone-fast") {
            return run_tts_clone_fast(argv[2], argv[3]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-real") {
            return run_tts_clone_real(argv[2], argv[3], argv[4], argv[5], argv[6]) ? 0 : 1;
        }
        if (argc == 3 && std::string(argv[1]) == "--clone-preflight") {
            return inspect_tts_clone_audio_preflight(argv[2]) ? 0 : 1;
        }
        if (argc == 4 && std::string(argv[1]) == "--clone-preprocess") {
            return run_tts_clone_audio_preprocess(argv[2], argv[3]) ? 0 : 1;
        }
        if (argc == 3 && std::string(argv[1]) == "--clone-readiness") {
            return inspect_tts_clone_readiness(argv[2]) ? 0 : 1;
        }
        if (argc == 4 && std::string(argv[1]) == "--clone-extract-mel") {
            return run_tts_clone_extract_mel(argv[2], argv[3]) ? 0 : 1;
        }
        if (argc == 4 && std::string(argv[1]) == "--clone-extract-fbank") {
            return run_tts_clone_extract_fbank(argv[2], argv[3]) ? 0 : 1;
        }
        if (argc == 4 && std::string(argv[1]) == "--clone-prepare-features") {
            return run_tts_clone_prepare_features(argv[2], argv[3]) ? 0 : 1;
        }
        if (argc == 3 && std::string(argv[1]) == "--clone-feature-readiness") {
            return inspect_tts_clone_feature_readiness(argv[2]) ? 0 : 1;
        }
        if (argc == 3 && std::string(argv[1]) == "--clone-encoder-model-readiness") {
            return inspect_tts_clone_encoder_model_readiness(argv[2]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-campplus-style-readiness") {
            return inspect_tts_clone_campplus_style_readiness(argv[2], argv[3], argv[4]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-campplus-style-from-features") {
            return run_tts_clone_campplus_style_from_features(argv[2], argv[3], argv[4]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-campplus-head-golden") {
            return run_tts_clone_campplus_head_golden(argv[2], argv[3], argv[4]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-feature-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_feature_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-encoder") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_encoder(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-w2v-extract-features") {
            return run_tts_clone_w2v_extract_features(argv[2], argv[3], argv[4]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-ffn1-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_ffn1_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-ffn1-intermediate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_ffn1_intermediate(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-w2v-layer0-ffn1-activate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[3], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_ffn1_activate(argv[2], w2v_tokens, argv[4]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-ffn1-output") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_ffn1_output(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer0-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_attention(argv[2],
                                                      argv[3],
                                                      argv[4],
                                                      argv[5],
                                                      w2v_tokens,
                                                      argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer0-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer0-final-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer0_final_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-ffn1-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_ffn1_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-ffn1-intermediate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_ffn1_intermediate(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-w2v-layer1-ffn1-activate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[3], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_ffn1_activate(argv[2], w2v_tokens, argv[4]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-ffn1-output") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_ffn1_output(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer1-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_attention(argv[2],
                                                      argv[3],
                                                      argv[4],
                                                      argv[5],
                                                      w2v_tokens,
                                                      argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer1-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer1-final-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer1_final_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-ffn1-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_ffn1_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-ffn1-intermediate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_ffn1_intermediate(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-w2v-layer2-ffn1-activate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[3], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_ffn1_activate(argv[2], w2v_tokens, argv[4]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-ffn1-output") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_ffn1_output(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer2-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_attention(argv[2],
                                                      argv[3],
                                                      argv[4],
                                                      argv[5],
                                                      w2v_tokens,
                                                      argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer2-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer2-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer2_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-ffn1-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_ffn1_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-ffn1-intermediate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_ffn1_intermediate(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-w2v-layer3-ffn1-activate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[3], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_ffn1_activate(argv[2], w2v_tokens, argv[4]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-ffn1-output") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_ffn1_output(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer3-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_attention(argv[2],
                                                      argv[3],
                                                      argv[4],
                                                      argv[5],
                                                      w2v_tokens,
                                                      argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer3-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer3-final-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer3_final_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-ffn1-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_ffn1_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-ffn1-intermediate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_ffn1_intermediate(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--clone-w2v-layer4-ffn1-activate") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[3], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_ffn1_activate(argv[2], w2v_tokens, argv[4]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-ffn1-output") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_ffn1_output(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer4-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_attention(argv[2],
                                                      argv[3],
                                                      argv[4],
                                                      argv[5],
                                                      w2v_tokens,
                                                      argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer4-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer4-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer4_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer5-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer5-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer5-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer5_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer6-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer6-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer6-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer6_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer7-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer7-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer7-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer7_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer8-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer8-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer8-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer8_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer9-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer9-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer9-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer9_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer10-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer10-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer10-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer10_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer11-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_attention(argv[2],
                                                       argv[3],
                                                       argv[4],
                                                       argv[5],
                                                       w2v_tokens,
                                                       argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer11-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer11-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer11_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer12-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_attention(argv[2],
                                                       argv[3],
                                                       argv[4],
                                                       argv[5],
                                                       w2v_tokens,
                                                       argv[7], {})
                ? 0
                : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer12-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer12-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer12_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer13-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_attention(argv[2],
                                                       argv[3],
                                                       argv[4],
                                                       argv[5],
                                                       w2v_tokens,
                                                       argv[7], {})
                ? 0
                : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer13-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer13-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer13_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer14-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer14-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer14-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer14_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer15-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer15-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer15-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer15_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-ffn1-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_ffn1_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-qkv") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_qkv(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "--clone-w2v-layer16-attention") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[6], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_attention(argv[2], argv[3], argv[4], argv[5], w2v_tokens, argv[7], {}) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-attention-project") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_attention_project(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-attention-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_attention_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-attention-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_attention_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-conv-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_conv_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-conv-glu") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_conv_glu(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-conv-depthwise") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_conv_depthwise(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-w2v-layer16-conv-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[5], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_conv_residual(argv[2], argv[3], argv[4], w2v_tokens, argv[6]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer16-ffn2-residual") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer16_ffn2_residual(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-layer17-final-norm") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_layer17_final_norm(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 6 && std::string(argv[1]) == "--clone-w2v-normalize") {
            const uint32_t w2v_tokens = parse_positive_u32_arg(argv[4], "W2V_TOKENS");
            return run_tts_clone_w2v_normalize(argv[2], argv[3], w2v_tokens, argv[5]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-semantic-quantize") {
            const uint32_t spk_tokens = parse_positive_u32_arg(argv[4], "SPK_TOKENS");
            return run_tts_clone_semantic_quantize(argv[2],
                                                   argv[3],
                                                   spk_tokens,
                                                   argv[5],
                                                   argv[6]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-semantic-prompt-from-spk-cond") {
            const uint32_t spk_tokens = parse_positive_u32_arg(argv[5], "SPK_TOKENS");
            return run_tts_clone_semantic_prompt_from_spk_cond(argv[2],
                                                               argv[3],
                                                               argv[4],
                                                               spk_tokens,
                                                               argv[6]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-s2mel-prompt-from-sref") {
            const uint32_t sref_tokens = parse_positive_u32_arg(argv[5], "S_REF_TOKENS");
            return run_tts_clone_s2mel_prompt_from_sref(argv[2],
                                                        argv[3],
                                                        argv[4],
                                                        sref_tokens,
                                                        argv[6]) ? 0 : 1;
        }
        if (argc == 7 && std::string(argv[1]) == "--clone-encoder-readiness") {
            const uint32_t spk_tokens = parse_positive_u32_arg(argv[4], "SPK_TOKENS");
            return inspect_tts_clone_encoder_readiness(argv[2],
                                                       argv[3],
                                                       spk_tokens,
                                                       argv[5],
                                                       argv[6]) ? 0 : 1;
        }
        if (argc == 10 && std::string(argv[1]) == "--clone-write-voice-bundle") {
            const uint32_t spk_tokens = parse_positive_u32_arg(argv[4], "SPK_TOKENS");
            const uint32_t prompt_tokens = parse_positive_u32_arg(argv[7], "PROMPT_TOKENS");
            return run_tts_clone_write_voice_bundle(argv[2],
                                                    argv[3],
                                                    spk_tokens,
                                                    argv[5],
                                                    argv[6],
                                                    prompt_tokens,
                                                    argv[8],
                                                    argv[9]) ? 0 : 1;
        }
        if (argc == 9 && std::string(argv[1]) == "--clone-write-voice-bundle-from-features") {
            const uint32_t spk_tokens = parse_positive_u32_arg(argv[4], "SPK_TOKENS");
            const uint32_t prompt_tokens = parse_positive_u32_arg(argv[7], "PROMPT_TOKENS");
            return run_tts_clone_write_voice_bundle_from_features(argv[2],
                                                                  argv[3],
                                                                  spk_tokens,
                                                                  argv[5],
                                                                  argv[6],
                                                                  prompt_tokens,
                                                                  argv[8]) ? 0 : 1;
        }
        if (argc == 4 && std::string(argv[1]) == "--text-readiness") {
            return inspect_tts_cjk_text_readiness(argv[2], argv[3]) ? 0 : 1;
        }
        if ((argc == 5 || argc == 6) && std::string(argv[1]).rfind("-", 0) != 0) {
            const std::string preset = argc == 6 ? std::string(argv[5]) : "standard";
            return run_tts_product_entry("mit2_tts", argv[1], argv[2], argv[3], argv[4], preset) ? 0 : 1;
        }
        if (std::string(argv[1]).rfind("-", 0) != 0) {
            usage(argv[0]);
            return 2;
        }
#endif
        bool diagnostics = false;
        bool smoke = false;
        bool test_primitives = false;
        bool test_scratch_allocator = false;
        bool test_metal_scratch_arena = false;
        bool plan_hot_scratch = false;
        uint32_t plan_hot_scratch_max_prefix_tokens = 0;
        uint32_t plan_hot_scratch_max_codes = 0;
        uint32_t plan_hot_scratch_prompt_tokens = 0;
        bool plan_hot_scratch_inputs = false;
        std::string plan_hot_scratch_inputs_conds_path;
        std::string plan_hot_scratch_inputs_text_ids_path;
        uint32_t plan_hot_scratch_inputs_max_codes = 0;
        uint32_t plan_hot_scratch_inputs_prompt_tokens = 0;
        std::string bundle_dir;
        std::string model_bundle_contract_dir;
        std::string voice_bundle_contract_dir;
        bool tts_validate_bundles = false;
        std::string gpt_layer_bundle_dir;
        std::string gpt_prepare_inputs_bundle_dir;
        std::string gpt_layer0_qkv_bundle_dir;
        std::string gpt_layer0_attention_bundle_dir;
        std::string gpt_layer0_kv_attention_bundle_dir;
        std::string gpt_layer0_block_bundle_dir;
        std::string gpt_transformer_stack_bundle_dir;
        std::string gpt_logits_bundle_dir;
        std::string gpt_kv_decode_bundle_dir;
        std::string gpt_greedy_bundle_dir;
        std::string gpt_kv_greedy_bundle_dir;
        std::string gpt_kv_greedy_golden_bundle_dir;
        std::string gpt_kv_greedy_golden_dir;
        bool gpt_sampling_processors = false;
        std::string export_text_ids_cjk_tokenizer_dir;
        std::string export_text_ids_cjk_text;
        std::string export_text_ids_cjk_output_path;
        std::string export_text_ids_cjk_segments_tokenizer_dir;
        std::string export_text_ids_cjk_segments_text;
        uint32_t export_text_ids_cjk_segments_max_tokens = 0;
        std::string export_text_ids_cjk_segments_output_dir;
        std::string test_text_ids_cjk_version_tokenizer_dir;
        std::string test_text_ids_cjk_slash_tokenizer_dir;
        std::string test_text_ids_cjk_temperature_tokenizer_dir;
        std::string test_text_ids_cjk_plus_tokenizer_dir;
        std::string test_text_ids_cjk_operator_tokenizer_dir;
        std::string test_text_ids_cjk_measure_tokenizer_dir;
        std::string test_text_ids_cjk_date_tokenizer_dir;
        std::string test_text_ids_cjk_time_tokenizer_dir;
        std::string test_text_ids_cjk_currency_tokenizer_dir;
        std::string test_text_ids_cjk_phone_tokenizer_dir;
        std::string test_text_ids_cjk_ratio_tokenizer_dir;
        std::string test_text_ids_cjk_fraction_tokenizer_dir;
        std::string test_text_ids_cjk_quote_tokenizer_dir;
        std::string test_text_ids_cjk_ellipsis_tokenizer_dir;
        std::string test_text_ids_cjk_percent_tokenizer_dir;
        std::string test_text_ids_cjk_no_tokenizer_dir;
        std::string tokenize_cjk_smoke_tokenizer_dir;
        std::string tokenize_cjk_smoke_text;
        std::string tokenize_cjk_smoke_output_path;
        std::string gpt_subsampling_golden_bundle_dir;
        std::string gpt_subsampling_golden_dir;
        std::string gpt_subsampling_metal_linear_golden_bundle_dir;
        std::string gpt_subsampling_metal_linear_golden_dir;
        std::string export_gpt_subsampling_bundle_dir;
        std::string export_gpt_subsampling_input_path;
        std::string export_gpt_subsampling_output_stack_path;
        std::string export_gpt_subsampling_output_pos_emb_path;
        std::string export_gpt_subsampling_output_mask_path;
        std::string gpt_emovec_golden_bundle_dir;
        std::string gpt_emovec_golden_dir;
        std::string gpt_emovec_metal_linear_golden_bundle_dir;
        std::string gpt_emovec_metal_linear_golden_dir;
        std::string export_gpt_emovec_bundle_dir;
        std::string export_gpt_emovec_input_path;
        std::string export_gpt_emovec_output_path;
        std::string gpt_conformer_ff_golden_bundle_dir;
        std::string gpt_conformer_ff_golden_dir;
        std::string gpt_conformer_ff_metal_golden_bundle_dir;
        std::string gpt_conformer_ff_metal_golden_dir;
        std::string gpt_conformer_attn_golden_bundle_dir;
        std::string gpt_conformer_attn_golden_dir;
        std::string gpt_conformer_attn_metal_proj_golden_bundle_dir;
        std::string gpt_conformer_attn_metal_proj_golden_dir;
        std::string gpt_conformer_conv_golden_bundle_dir;
        std::string gpt_conformer_conv_golden_dir;
        std::string gpt_conformer_conv_metal_golden_bundle_dir;
        std::string gpt_conformer_conv_metal_golden_dir;
        std::string gpt_conformer_block_golden_bundle_dir;
        std::string gpt_conformer_block_golden_dir;
        std::string gpt_conformer_block_metal_ff_golden_bundle_dir;
        std::string gpt_conformer_block_metal_ff_golden_dir;
        std::string gpt_conformer_block_metal_attn_ff_golden_bundle_dir;
        std::string gpt_conformer_block_metal_attn_ff_golden_dir;
        std::string gpt_conformer_block_metal_attn_conv_ff_golden_bundle_dir;
        std::string gpt_conformer_block_metal_attn_conv_ff_golden_dir;
        std::string gpt_conformer_stack_golden_bundle_dir;
        std::string gpt_conformer_stack_golden_dir;
        std::string gpt_conformer_stack_metal_ff_golden_bundle_dir;
        std::string gpt_conformer_stack_metal_ff_golden_dir;
        std::string gpt_conformer_stack_metal_attn_ff_golden_bundle_dir;
        std::string gpt_conformer_stack_metal_attn_ff_golden_dir;
        std::string gpt_conformer_stack_metal_attn_conv_ff_golden_bundle_dir;
        std::string gpt_conformer_stack_metal_attn_conv_ff_golden_dir;
        std::string export_gpt_conformer_stack_bundle_dir;
        std::string export_gpt_conformer_stack_input_path;
        std::string export_gpt_conformer_stack_pos_emb_path;
        std::string export_gpt_conformer_stack_mask_path;
        std::string export_gpt_conformer_stack_output_path;
        std::string gpt_perceiver_golden_bundle_dir;
        std::string gpt_perceiver_golden_dir;
        std::string export_gpt_perceiver_bundle_dir;
        std::string export_gpt_perceiver_context_path;
        std::string export_gpt_perceiver_mask_path;
        std::string export_gpt_perceiver_output_path;
        std::string export_gpt_frontend_tail_bundle_dir;
        std::string export_gpt_frontend_tail_speech_cond_path;
        std::string export_gpt_frontend_tail_emovec_path;
        std::string export_gpt_frontend_tail_text_ids_path;
        std::string export_gpt_frontend_tail_output_conds_path;
        std::string export_gpt_frontend_tail_output_fake_path;
        std::string export_gpt_frontend_tail_output_inputs_path;
        std::string export_gpt_frontend_tail_output_mask_path;
        std::string export_gpt_kv_codes_bundle_dir;
        std::string export_gpt_kv_codes_golden_dir;
        std::string export_gpt_kv_codes_output_path;
        std::string export_gpt_kv_codes_inputs_bundle_dir;
        std::string export_gpt_kv_codes_inputs_conds_path;
        std::string export_gpt_kv_codes_inputs_text_ids_path;
        uint32_t export_gpt_kv_codes_inputs_max_codes = 0;
        std::string export_gpt_kv_codes_inputs_output_path;
        std::string test_gpt_kv_codes_inputs_bundle_dir;
        std::string test_gpt_kv_codes_inputs_conds_path;
        std::string test_gpt_kv_codes_inputs_text_ids_path;
        uint32_t test_gpt_kv_codes_inputs_max_codes = 0;
        std::string test_gpt_kv_codes_inputs_expected_path;
        std::string export_gpt_kv_codes_inputs_sampled_bundle_dir;
        std::string export_gpt_kv_codes_inputs_sampled_conds_path;
        std::string export_gpt_kv_codes_inputs_sampled_text_ids_path;
        uint32_t export_gpt_kv_codes_inputs_sampled_max_codes = 0;
        GptSamplingConfig export_gpt_kv_codes_inputs_sampled_config;
        std::string export_gpt_kv_codes_inputs_sampled_output_path;
        std::string test_gpt_sampled_inputs_determinism_bundle_dir;
        std::string test_gpt_sampled_inputs_determinism_conds_path;
        std::string test_gpt_sampled_inputs_determinism_text_ids_path;
        uint32_t test_gpt_sampled_inputs_determinism_max_codes = 0;
        std::string gpt_latent_golden_bundle_dir;
        std::string gpt_latent_golden_dir;
        std::string trace_gpt_latent_bundle_dir;
        std::string trace_gpt_latent_golden_dir;
        std::string vq2emb_bundle_dir;
        std::string length_regulator_front_bundle_dir;
        std::string length_regulator_full_bundle_dir;
        std::string length_regulator_golden_bundle_dir;
        std::string length_regulator_golden_dir;
        std::string export_length_regulator_bundle_dir;
        std::string export_length_regulator_golden_dir;
        std::string export_length_regulator_output_path;
        std::string export_length_regulator_stages_bundle_dir;
        std::string export_length_regulator_stages_golden_dir;
        std::string export_length_regulator_stages_output_dir;
        std::string timestep_embedder_bundle_dir;
        std::string dit_input_merge_bundle_dir;
        std::string dit_attention_projection_bundle_dir;
        std::string dit_feed_forward_bundle_dir;
        std::string dit_adaptive_norm_bundle_dir;
        std::string dit_attention_core_bundle_dir;
        std::string dit_transformer_block0_bundle_dir;
        std::string dit_transformer_stack_bundle_dir;
        std::string dit_post_transformer_proj_bundle_dir;
        std::string dit_final_layer_bundle_dir;
        std::string wavenet_layer0_gate_bundle_dir;
        std::string wavenet_stack_bundle_dir;
        std::string dit_estimator_step_bundle_dir;
        std::string dit_estimator_golden_bundle_dir;
        std::string dit_estimator_golden_dir;
        std::string cfm_euler_bundle_dir;
        std::string cfm_euler_golden_bundle_dir;
        std::string cfm_euler_golden_dir;
        std::string cfm_euler_cfg_bundle_dir;
        std::string cfm_euler_cfg_large_bundle_dir;
        std::string cfm_euler_cfg_golden_bundle_dir;
        std::string cfm_euler_cfg_golden_dir;
        std::string s2mel_full_golden_bundle_dir;
        std::string s2mel_full_golden_dir;
        std::string s2mel_full_inputs_bundle_dir;
        std::string s2mel_full_inputs_noise_path;
        std::string s2mel_full_inputs_prompt_path;
        std::string s2mel_full_inputs_condition_path;
        std::string s2mel_full_inputs_style_path;
        uint32_t s2mel_full_inputs_steps = 0;
        float s2mel_full_inputs_cfg_rate = 0.0f;
        std::string s2mel_full_inputs_expected_path;
        std::string trace_s2mel_cfm_bundle_dir;
        std::string trace_s2mel_cfm_golden_dir;
        std::string trace_s2mel_cfm_error_bundle_dir;
        std::string trace_s2mel_cfm_error_golden_dir;
        std::string hot_tts_golden_bundle_dir;
        std::string hot_tts_s2mel_golden_dir;
        std::string hot_tts_wave_golden_dir;
        std::string hot_tts_from_gpt_bundle_dir;
        std::string hot_tts_from_gpt_voice_bundle_dir;
        std::string hot_tts_from_gpt_golden_dir;
        std::string hot_tts_from_gpt_s2mel_golden_dir;
        std::string hot_tts_from_gpt_wave_golden_dir;
        std::string hot_tts_from_codes_bundle_dir;
        std::string hot_tts_from_codes_voice_bundle_dir;
        std::string hot_tts_from_codes_golden_dir;
        std::string hot_tts_from_codes_s2mel_golden_dir;
        std::string hot_tts_from_codes_wave_golden_dir;
        std::string synthesize_hot_gpt_bundle_dir;
        std::string synthesize_hot_gpt_voice_bundle_dir;
        std::string synthesize_hot_gpt_golden_dir;
        std::string synthesize_hot_gpt_s2mel_golden_dir;
        std::string synthesize_hot_gpt_output_wav;
        std::string export_hot_codes_bundle_dir;
        std::string export_hot_codes_voice_bundle_dir;
        std::string export_hot_codes_golden_dir;
        std::string export_hot_codes_s2mel_golden_dir;
        std::string export_hot_codes_output_condition;
        std::string export_hot_codes_input_bundle_dir;
        std::string export_hot_codes_input_voice_bundle_dir;
        std::string export_hot_codes_input_golden_dir;
        std::string export_hot_codes_input_s2mel_golden_dir;
        std::string export_hot_codes_input_codes_path;
        std::string export_hot_codes_input_output_condition;
        std::string export_hot_codes_inputs_bundle_dir;
        std::string export_hot_codes_inputs_voice_bundle_dir;
        std::string export_hot_codes_inputs_conds_path;
        std::string export_hot_codes_inputs_text_ids_path;
        std::string export_hot_codes_inputs_codes_path;
        uint32_t export_hot_codes_inputs_prompt_tokens = 0;
        std::string export_hot_codes_inputs_output_condition;
        std::string hot_tts_condition_bundle_dir;
        std::string hot_tts_condition_voice_bundle_dir;
        std::string hot_tts_condition_s2mel_golden_dir;
        std::string hot_tts_condition_wave_golden_dir;
        std::string hot_tts_condition_path;
        std::string synthesize_hot_condition_bundle_dir;
        std::string synthesize_hot_condition_voice_bundle_dir;
        std::string synthesize_hot_condition_s2mel_golden_dir;
        std::string synthesize_hot_condition_path;
        std::string synthesize_hot_condition_output_wav;
        std::string synthesize_hot_condition_inputs_bundle_dir;
        std::string synthesize_hot_condition_inputs_voice_bundle_dir;
        std::string synthesize_hot_condition_inputs_condition_path;
        std::string synthesize_hot_condition_inputs_noise_path;
        uint32_t synthesize_hot_condition_inputs_prompt_tokens = 0;
        uint32_t synthesize_hot_condition_inputs_steps = 0;
        float synthesize_hot_condition_inputs_cfg_rate = 0.0f;
        std::string synthesize_hot_condition_inputs_output_wav;
        std::string synthesize_hot_inputs_bundle_dir;
        std::string synthesize_hot_inputs_voice_bundle_dir;
        std::string synthesize_hot_inputs_conds_path;
        std::string synthesize_hot_inputs_text_ids_path;
        uint32_t synthesize_hot_inputs_max_codes = 0;
        std::string synthesize_hot_inputs_noise_path;
        uint32_t synthesize_hot_inputs_prompt_tokens = 0;
        uint32_t synthesize_hot_inputs_steps = 0;
        float synthesize_hot_inputs_cfg_rate = 0.0f;
        std::string synthesize_hot_inputs_output_wav;
        std::string synthesize_hot_inputs_seeded_bundle_dir;
        std::string synthesize_hot_inputs_seeded_voice_bundle_dir;
        std::string synthesize_hot_inputs_seeded_conds_path;
        std::string synthesize_hot_inputs_seeded_text_ids_path;
        uint32_t synthesize_hot_inputs_seeded_max_codes = 0;
        uint64_t synthesize_hot_inputs_seeded_seed = 0;
        float synthesize_hot_inputs_seeded_temperature = 0.0f;
        uint32_t synthesize_hot_inputs_seeded_prompt_tokens = 0;
        uint32_t synthesize_hot_inputs_seeded_steps = 0;
        float synthesize_hot_inputs_seeded_cfg_rate = 0.0f;
        std::string synthesize_hot_inputs_seeded_output_wav;
        std::string synthesize_hot_text_cjk_seeded_bundle_dir;
        std::string synthesize_hot_text_cjk_seeded_voice_bundle_dir;
        std::string synthesize_hot_text_cjk_seeded_text;
        uint64_t synthesize_hot_text_cjk_seeded_seed = 0;
        float synthesize_hot_text_cjk_seeded_temperature = 0.0f;
        uint32_t synthesize_hot_text_cjk_seeded_prompt_tokens = 0;
        uint32_t synthesize_hot_text_cjk_seeded_steps = 0;
        float synthesize_hot_text_cjk_seeded_cfg_rate = 0.0f;
        std::string synthesize_hot_text_cjk_seeded_output_wav;
        std::string synthesize_hot_text_cjk_sampled_seeded_bundle_dir;
        std::string synthesize_hot_text_cjk_sampled_seeded_voice_bundle_dir;
        std::string synthesize_hot_text_cjk_sampled_seeded_text;
        GptSamplingConfig synthesize_hot_text_cjk_sampled_seeded_gpt_config;
        uint64_t synthesize_hot_text_cjk_sampled_seeded_noise_seed = 0;
        float synthesize_hot_text_cjk_sampled_seeded_noise_temperature = 0.0f;
        uint32_t synthesize_hot_text_cjk_sampled_seeded_prompt_tokens = 0;
        uint32_t synthesize_hot_text_cjk_sampled_seeded_steps = 0;
        float synthesize_hot_text_cjk_sampled_seeded_cfg_rate = 0.0f;
        std::string synthesize_hot_text_cjk_sampled_seeded_output_wav;
        std::string synthesize_hot_text_cjk_segments_seeded_bundle_dir;
        std::string synthesize_hot_text_cjk_segments_seeded_voice_bundle_dir;
        std::string synthesize_hot_text_cjk_segments_seeded_text;
        uint32_t synthesize_hot_text_cjk_segments_seeded_max_text_tokens = 0;
        uint64_t synthesize_hot_text_cjk_segments_seeded_seed = 0;
        float synthesize_hot_text_cjk_segments_seeded_temperature = 0.0f;
        uint32_t synthesize_hot_text_cjk_segments_seeded_prompt_tokens = 0;
        uint32_t synthesize_hot_text_cjk_segments_seeded_steps = 0;
        float synthesize_hot_text_cjk_segments_seeded_cfg_rate = 0.0f;
        uint32_t synthesize_hot_text_cjk_segments_seeded_interval_silence_ms = 0;
        std::string synthesize_hot_text_cjk_segments_seeded_output_wav;
        std::string synthesize_hot_text_cjk_segments_sampled_seeded_bundle_dir;
        std::string synthesize_hot_text_cjk_segments_sampled_seeded_voice_bundle_dir;
        std::string synthesize_hot_text_cjk_segments_sampled_seeded_text;
        uint32_t synthesize_hot_text_cjk_segments_sampled_seeded_max_text_tokens = 0;
        GptSamplingConfig synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config;
        uint64_t synthesize_hot_text_cjk_segments_sampled_seeded_noise_seed = 0;
        float synthesize_hot_text_cjk_segments_sampled_seeded_noise_temperature = 0.0f;
        uint32_t synthesize_hot_text_cjk_segments_sampled_seeded_prompt_tokens = 0;
        uint32_t synthesize_hot_text_cjk_segments_sampled_seeded_steps = 0;
        float synthesize_hot_text_cjk_segments_sampled_seeded_cfg_rate = 0.0f;
        uint32_t synthesize_hot_text_cjk_segments_sampled_seeded_interval_silence_ms = 0;
        std::string synthesize_hot_text_cjk_segments_sampled_seeded_output_wav;
        std::string tts_cjk_bundle_dir;
        std::string tts_cjk_voice_bundle_dir;
        std::string tts_cjk_text;
        uint32_t tts_cjk_steps = 0;
        std::string tts_cjk_output_wav;
        std::string tts_cjk_segments_bundle_dir;
        std::string tts_cjk_segments_voice_bundle_dir;
        std::string tts_cjk_segments_text;
        uint32_t tts_cjk_segments_max_text_tokens = 0;
        uint32_t tts_cjk_segments_steps = 0;
        uint32_t tts_cjk_segments_interval_silence_ms = 0;
        std::string tts_cjk_segments_output_wav;
        std::string tts_cjk_sampled_bundle_dir;
        std::string tts_cjk_sampled_voice_bundle_dir;
        std::string tts_cjk_sampled_text;
        uint32_t tts_cjk_sampled_steps = 0;
        std::string tts_cjk_sampled_output_wav;
        std::string tts_cjk_segments_sampled_bundle_dir;
        std::string tts_cjk_segments_sampled_voice_bundle_dir;
        std::string tts_cjk_segments_sampled_text;
        uint32_t tts_cjk_segments_sampled_max_text_tokens = 0;
        uint32_t tts_cjk_segments_sampled_steps = 0;
        uint32_t tts_cjk_segments_sampled_interval_silence_ms = 0;
        std::string tts_cjk_segments_sampled_output_wav;
        std::string tts_cjk_preset_bundle_dir;
        std::string tts_cjk_preset_voice_bundle_dir;
        std::string tts_cjk_preset_text;
        std::string tts_cjk_preset_name;
        std::string tts_cjk_preset_output_wav;
        std::string tts_cjk_segments_preset_bundle_dir;
        std::string tts_cjk_segments_preset_voice_bundle_dir;
        std::string tts_cjk_segments_preset_text;
        std::string tts_cjk_segments_preset_name;
        std::string tts_cjk_segments_preset_output_wav;
        std::string tts_cjk_auto_preset_bundle_dir;
        std::string tts_cjk_auto_preset_voice_bundle_dir;
        std::string tts_cjk_auto_preset_text;
        std::string tts_cjk_auto_preset_name;
        std::string tts_cjk_auto_preset_output_wav;
        std::string serve_bundle_dir;
        std::string serve_voice_bundle_dir;
        std::string http_host = "127.0.0.1";
        uint16_t http_port = 3456;
        bool http_server_requested = false;
        bool http_web_requested = false;
        std::string http_model_bundle_dir;
        std::string http_voice_store_dir;
        std::string http_web_key;
        uint32_t http_queue_size = 0;
        uint32_t http_tts_concurrency = 0;
        uint32_t http_clone_concurrency = 0;
        std::string tts_product_bundle_dir;
        std::string tts_product_voice_bundle_dir;
        std::string tts_product_text;
        std::string tts_product_output_wav;
        std::string tts_cjk_text_readiness_bundle_dir;
        std::string tts_cjk_text_readiness_text;
        std::string tts_product_readiness_bundle_dir;
        std::string tts_product_readiness_voice_bundle_dir;
        std::string synthesize_hot_inputs_sampled_seeded_bundle_dir;
        std::string synthesize_hot_inputs_sampled_seeded_voice_bundle_dir;
        std::string synthesize_hot_inputs_sampled_seeded_conds_path;
        std::string synthesize_hot_inputs_sampled_seeded_text_ids_path;
        uint32_t synthesize_hot_inputs_sampled_seeded_max_codes = 0;
        GptSamplingConfig synthesize_hot_inputs_sampled_seeded_gpt_config;
        uint64_t synthesize_hot_inputs_sampled_seeded_noise_seed = 0;
        float synthesize_hot_inputs_sampled_seeded_noise_temperature = 0.0f;
        uint32_t synthesize_hot_inputs_sampled_seeded_prompt_tokens = 0;
        uint32_t synthesize_hot_inputs_sampled_seeded_steps = 0;
        float synthesize_hot_inputs_sampled_seeded_cfg_rate = 0.0f;
        std::string synthesize_hot_inputs_sampled_seeded_output_wav;
        std::string synthesize_hot_inputs_seeded_shared_bundle_dir;
        std::string synthesize_hot_inputs_seeded_shared_voice_bundle_dir;
        std::string synthesize_hot_inputs_seeded_shared_conds_path;
        std::string synthesize_hot_inputs_seeded_shared_text_ids_path;
        uint32_t synthesize_hot_inputs_seeded_shared_max_codes = 0;
        uint64_t synthesize_hot_inputs_seeded_shared_seed = 0;
        float synthesize_hot_inputs_seeded_shared_temperature = 0.0f;
        uint32_t synthesize_hot_inputs_seeded_shared_prompt_tokens = 0;
        uint32_t synthesize_hot_inputs_seeded_shared_steps = 0;
        float synthesize_hot_inputs_seeded_shared_cfg_rate = 0.0f;
        std::string synthesize_hot_inputs_seeded_shared_output_wav;
        std::string synthesize_hot_native_bundle_dir;
        std::string synthesize_hot_native_voice_bundle_dir;
        std::string synthesize_hot_native_gpt_golden_dir;
        std::string synthesize_hot_native_s2mel_golden_dir;
        std::string synthesize_hot_native_output_wav;
        std::string bigvgan_conv_pre_bundle_dir;
        std::string bigvgan_upsample0_bundle_dir;
        std::string bigvgan_upsampler_stack_bundle_dir;
        std::string bigvgan_front_bundle_dir;
        std::string bigvgan_activation_post_bundle_dir;
        std::string bigvgan_activation_rb0_bundle_dir;
        std::string bigvgan_resblock0_pair0_bundle_dir;
        std::string bigvgan_resblock0_bundle_dir;
        std::string bigvgan_resblock_group0_bundle_dir;
        std::string bigvgan_body_bundle_dir;
        std::string bigvgan_post_bundle_dir;
        std::string bigvgan_vocoder_bundle_dir;
        std::string bigvgan_vocoder_golden_bundle_dir;
        std::string bigvgan_vocoder_golden_dir;

        // Env fallback for the voice conditioning LRU capacity; a later
        // --lrucache flag (parsed below) takes precedence.
        if (const char* v = std::getenv("MIT2_LRU_CACHE")) {
            const long n = std::strtol(v, nullptr, 10);
            if (n >= 0 && n <= 64) {
                g_voice_cond_cache_size = static_cast<uint32_t>(n);
            }
        }

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--diagnostics") {
                diagnostics = true;
            } else if (arg == "--smoke-copy") {
                smoke = true;
            } else if (arg == "--test-primitives") {
                test_primitives = true;
            } else if (arg == "--test-scratch-allocator") {
                test_scratch_allocator = true;
            } else if (arg == "--test-metal-scratch-arena") {
                test_metal_scratch_arena = true;
            } else if (arg == "--plan-hot-scratch" && i + 3 < argc) {
                plan_hot_scratch = true;
                plan_hot_scratch_max_prefix_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                plan_hot_scratch_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                plan_hot_scratch_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--plan-hot-scratch-inputs" && i + 4 < argc) {
                plan_hot_scratch_inputs = true;
                plan_hot_scratch_inputs_conds_path = argv[++i];
                plan_hot_scratch_inputs_text_ids_path = argv[++i];
                plan_hot_scratch_inputs_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                plan_hot_scratch_inputs_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--inspect-bundle" && i + 1 < argc) {
                bundle_dir = argv[++i];
            } else if (arg == "--inspect-model-bundle" && i + 1 < argc) {
                model_bundle_contract_dir = argv[++i];
            } else if (arg == "--inspect-voice-bundle" && i + 1 < argc) {
                voice_bundle_contract_dir = argv[++i];
            } else if (arg == "--tts-validate-bundles") {
                tts_validate_bundles = true;
            } else if (arg == "--test-gpt-layer" && i + 1 < argc) {
                gpt_layer_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-prepare-inputs" && i + 1 < argc) {
                gpt_prepare_inputs_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-layer0-qkv" && i + 1 < argc) {
                gpt_layer0_qkv_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-layer0-attn" && i + 1 < argc) {
                gpt_layer0_attention_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-layer0-kv-attn" && i + 1 < argc) {
                gpt_layer0_kv_attention_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-layer0-block" && i + 1 < argc) {
                gpt_layer0_block_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-transformer-stack" && i + 1 < argc) {
                gpt_transformer_stack_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-logits" && i + 1 < argc) {
                gpt_logits_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-kv-decode" && i + 1 < argc) {
                gpt_kv_decode_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-greedy" && i + 1 < argc) {
                gpt_greedy_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-kv-greedy" && i + 1 < argc) {
                gpt_kv_greedy_bundle_dir = argv[++i];
            } else if (arg == "--test-gpt-kv-greedy-golden" && i + 2 < argc) {
                gpt_kv_greedy_golden_bundle_dir = argv[++i];
                gpt_kv_greedy_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-sampling-processors") {
                gpt_sampling_processors = true;
            } else if (arg == "--export-text-ids-cjk" && i + 3 < argc) {
                export_text_ids_cjk_tokenizer_dir = argv[++i];
                export_text_ids_cjk_text = argv[++i];
                export_text_ids_cjk_output_path = argv[++i];
            } else if (arg == "--export-text-ids-cjk-segments" && i + 4 < argc) {
                export_text_ids_cjk_segments_tokenizer_dir = argv[++i];
                export_text_ids_cjk_segments_text = argv[++i];
                export_text_ids_cjk_segments_max_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                export_text_ids_cjk_segments_output_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-version-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_version_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-slash-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_slash_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-temperature-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_temperature_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-plus-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_plus_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-operator-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_operator_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-measure-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_measure_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-date-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_date_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-time-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_time_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-currency-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_currency_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-phone-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_phone_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-ratio-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_ratio_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-fraction-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_fraction_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-quote-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_quote_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-ellipsis-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_ellipsis_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-percent-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_percent_tokenizer_dir = argv[++i];
            } else if (arg == "--test-text-ids-cjk-no-tokenizer" && i + 1 < argc) {
                test_text_ids_cjk_no_tokenizer_dir = argv[++i];
            } else if (arg == "--tokenize-cjk-smoke" && i + 3 < argc) {
                tokenize_cjk_smoke_tokenizer_dir = argv[++i];
                tokenize_cjk_smoke_text = argv[++i];
                tokenize_cjk_smoke_output_path = argv[++i];
            } else if (arg == "--test-gpt-subsampling-golden" && i + 2 < argc) {
                gpt_subsampling_golden_bundle_dir = argv[++i];
                gpt_subsampling_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-subsampling-metal-linear-golden" && i + 2 < argc) {
                gpt_subsampling_metal_linear_golden_bundle_dir = argv[++i];
                gpt_subsampling_metal_linear_golden_dir = argv[++i];
            } else if (arg == "--export-gpt-subsampling" && i + 5 < argc) {
                export_gpt_subsampling_bundle_dir = argv[++i];
                export_gpt_subsampling_input_path = argv[++i];
                export_gpt_subsampling_output_stack_path = argv[++i];
                export_gpt_subsampling_output_pos_emb_path = argv[++i];
                export_gpt_subsampling_output_mask_path = argv[++i];
            } else if (arg == "--test-gpt-emovec-golden" && i + 2 < argc) {
                gpt_emovec_golden_bundle_dir = argv[++i];
                gpt_emovec_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-emovec-metal-linear-golden" && i + 2 < argc) {
                gpt_emovec_metal_linear_golden_bundle_dir = argv[++i];
                gpt_emovec_metal_linear_golden_dir = argv[++i];
            } else if (arg == "--export-gpt-emovec" && i + 3 < argc) {
                export_gpt_emovec_bundle_dir = argv[++i];
                export_gpt_emovec_input_path = argv[++i];
                export_gpt_emovec_output_path = argv[++i];
            } else if (arg == "--test-gpt-conformer-ff-golden" && i + 2 < argc) {
                gpt_conformer_ff_golden_bundle_dir = argv[++i];
                gpt_conformer_ff_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-ff-metal-golden" && i + 2 < argc) {
                gpt_conformer_ff_metal_golden_bundle_dir = argv[++i];
                gpt_conformer_ff_metal_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-attn-golden" && i + 2 < argc) {
                gpt_conformer_attn_golden_bundle_dir = argv[++i];
                gpt_conformer_attn_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-attn-metal-proj-golden" && i + 2 < argc) {
                gpt_conformer_attn_metal_proj_golden_bundle_dir = argv[++i];
                gpt_conformer_attn_metal_proj_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-conv-golden" && i + 2 < argc) {
                gpt_conformer_conv_golden_bundle_dir = argv[++i];
                gpt_conformer_conv_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-conv-metal-golden" && i + 2 < argc) {
                gpt_conformer_conv_metal_golden_bundle_dir = argv[++i];
                gpt_conformer_conv_metal_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-block-golden" && i + 2 < argc) {
                gpt_conformer_block_golden_bundle_dir = argv[++i];
                gpt_conformer_block_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-block-metal-ff-golden" && i + 2 < argc) {
                gpt_conformer_block_metal_ff_golden_bundle_dir = argv[++i];
                gpt_conformer_block_metal_ff_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-block-metal-attn-ff-golden" && i + 2 < argc) {
                gpt_conformer_block_metal_attn_ff_golden_bundle_dir = argv[++i];
                gpt_conformer_block_metal_attn_ff_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-block-metal-attn-conv-ff-golden" && i + 2 < argc) {
                gpt_conformer_block_metal_attn_conv_ff_golden_bundle_dir = argv[++i];
                gpt_conformer_block_metal_attn_conv_ff_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-stack-golden" && i + 2 < argc) {
                gpt_conformer_stack_golden_bundle_dir = argv[++i];
                gpt_conformer_stack_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-stack-metal-ff-golden" && i + 2 < argc) {
                gpt_conformer_stack_metal_ff_golden_bundle_dir = argv[++i];
                gpt_conformer_stack_metal_ff_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-stack-metal-attn-ff-golden" && i + 2 < argc) {
                gpt_conformer_stack_metal_attn_ff_golden_bundle_dir = argv[++i];
                gpt_conformer_stack_metal_attn_ff_golden_dir = argv[++i];
            } else if (arg == "--test-gpt-conformer-stack-metal-attn-conv-ff-golden" && i + 2 < argc) {
                gpt_conformer_stack_metal_attn_conv_ff_golden_bundle_dir = argv[++i];
                gpt_conformer_stack_metal_attn_conv_ff_golden_dir = argv[++i];
            } else if (arg == "--export-gpt-conformer-stack" && i + 5 < argc) {
                export_gpt_conformer_stack_bundle_dir = argv[++i];
                export_gpt_conformer_stack_input_path = argv[++i];
                export_gpt_conformer_stack_pos_emb_path = argv[++i];
                export_gpt_conformer_stack_mask_path = argv[++i];
                export_gpt_conformer_stack_output_path = argv[++i];
            } else if (arg == "--test-gpt-perceiver-golden" && i + 2 < argc) {
                gpt_perceiver_golden_bundle_dir = argv[++i];
                gpt_perceiver_golden_dir = argv[++i];
            } else if (arg == "--export-gpt-perceiver" && i + 4 < argc) {
                export_gpt_perceiver_bundle_dir = argv[++i];
                export_gpt_perceiver_context_path = argv[++i];
                export_gpt_perceiver_mask_path = argv[++i];
                export_gpt_perceiver_output_path = argv[++i];
            } else if (arg == "--export-gpt-frontend-tail" && i + 8 < argc) {
                export_gpt_frontend_tail_bundle_dir = argv[++i];
                export_gpt_frontend_tail_speech_cond_path = argv[++i];
                export_gpt_frontend_tail_emovec_path = argv[++i];
                export_gpt_frontend_tail_text_ids_path = argv[++i];
                export_gpt_frontend_tail_output_conds_path = argv[++i];
                export_gpt_frontend_tail_output_fake_path = argv[++i];
                export_gpt_frontend_tail_output_inputs_path = argv[++i];
                export_gpt_frontend_tail_output_mask_path = argv[++i];
            } else if (arg == "--export-gpt-kv-codes-golden" && i + 3 < argc) {
                export_gpt_kv_codes_bundle_dir = argv[++i];
                export_gpt_kv_codes_golden_dir = argv[++i];
                export_gpt_kv_codes_output_path = argv[++i];
            } else if (arg == "--export-gpt-kv-codes-inputs" && i + 5 < argc) {
                export_gpt_kv_codes_inputs_bundle_dir = argv[++i];
                export_gpt_kv_codes_inputs_conds_path = argv[++i];
                export_gpt_kv_codes_inputs_text_ids_path = argv[++i];
                export_gpt_kv_codes_inputs_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                export_gpt_kv_codes_inputs_output_path = argv[++i];
            } else if (arg == "--test-gpt-kv-codes-inputs" && i + 5 < argc) {
                test_gpt_kv_codes_inputs_bundle_dir = argv[++i];
                test_gpt_kv_codes_inputs_conds_path = argv[++i];
                test_gpt_kv_codes_inputs_text_ids_path = argv[++i];
                test_gpt_kv_codes_inputs_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                test_gpt_kv_codes_inputs_expected_path = argv[++i];
            } else if (arg == "--export-gpt-kv-codes-inputs-sampled" && i + 10 < argc) {
                export_gpt_kv_codes_inputs_sampled_bundle_dir = argv[++i];
                export_gpt_kv_codes_inputs_sampled_conds_path = argv[++i];
                export_gpt_kv_codes_inputs_sampled_text_ids_path = argv[++i];
                export_gpt_kv_codes_inputs_sampled_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                export_gpt_kv_codes_inputs_sampled_config.do_sample = true;
                export_gpt_kv_codes_inputs_sampled_config.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                export_gpt_kv_codes_inputs_sampled_config.temperature = std::stof(argv[++i]);
                export_gpt_kv_codes_inputs_sampled_config.top_k = static_cast<uint32_t>(std::stoul(argv[++i]));
                export_gpt_kv_codes_inputs_sampled_config.top_p = std::stof(argv[++i]);
                export_gpt_kv_codes_inputs_sampled_config.repetition_penalty = std::stof(argv[++i]);
                export_gpt_kv_codes_inputs_sampled_output_path = argv[++i];
            } else if (arg == "--test-gpt-sampled-inputs-determinism" && i + 4 < argc) {
                test_gpt_sampled_inputs_determinism_bundle_dir = argv[++i];
                test_gpt_sampled_inputs_determinism_conds_path = argv[++i];
                test_gpt_sampled_inputs_determinism_text_ids_path = argv[++i];
                test_gpt_sampled_inputs_determinism_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--test-gpt-latent-golden" && i + 2 < argc) {
                gpt_latent_golden_bundle_dir = argv[++i];
                gpt_latent_golden_dir = argv[++i];
            } else if (arg == "--trace-gpt-latent-golden" && i + 2 < argc) {
                trace_gpt_latent_bundle_dir = argv[++i];
                trace_gpt_latent_golden_dir = argv[++i];
            } else if (arg == "--test-vq2emb" && i + 1 < argc) {
                vq2emb_bundle_dir = argv[++i];
            } else if (arg == "--test-length-regulator-front" && i + 1 < argc) {
                length_regulator_front_bundle_dir = argv[++i];
            } else if (arg == "--test-length-regulator-full" && i + 1 < argc) {
                length_regulator_full_bundle_dir = argv[++i];
            } else if (arg == "--test-length-regulator-golden" && i + 2 < argc) {
                length_regulator_golden_bundle_dir = argv[++i];
                length_regulator_golden_dir = argv[++i];
            } else if (arg == "--export-length-regulator-golden" && i + 3 < argc) {
                export_length_regulator_bundle_dir = argv[++i];
                export_length_regulator_golden_dir = argv[++i];
                export_length_regulator_output_path = argv[++i];
            } else if (arg == "--export-length-regulator-stages-golden" && i + 3 < argc) {
                export_length_regulator_stages_bundle_dir = argv[++i];
                export_length_regulator_stages_golden_dir = argv[++i];
                export_length_regulator_stages_output_dir = argv[++i];
            } else if (arg == "--test-timestep-embedder" && i + 1 < argc) {
                timestep_embedder_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-input-merge" && i + 1 < argc) {
                dit_input_merge_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-attention-proj" && i + 1 < argc) {
                dit_attention_projection_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-feed-forward" && i + 1 < argc) {
                dit_feed_forward_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-adaptive-norm" && i + 1 < argc) {
                dit_adaptive_norm_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-attention-core" && i + 1 < argc) {
                dit_attention_core_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-attention-tokens" && i + 2 < argc) {
                const std::string bdir = argv[++i];
                const uint32_t ntok = static_cast<uint32_t>(std::stoul(argv[++i]));
                return run_dit_attention_core_tokens_test(bdir, ntok) ? 0 : 1;
            } else if (arg == "--bench-cfm" && i + 4 < argc) {
                const std::string bdir = argv[++i];
                const uint32_t ntok = static_cast<uint32_t>(std::stoul(argv[++i]));
                const uint32_t nsteps = static_cast<uint32_t>(std::stoul(argv[++i]));
                const uint32_t niters = static_cast<uint32_t>(std::stoul(argv[++i]));
                return run_bench_cfm(bdir, ntok, nsteps, niters) ? 0 : 1;
            } else if (arg == "--bench-bigvgan" && i + 3 < argc) {
                const std::string bdir = argv[++i];
                const uint32_t ntok = static_cast<uint32_t>(std::stoul(argv[++i]));
                const uint32_t niters = static_cast<uint32_t>(std::stoul(argv[++i]));
                return run_bench_bigvgan(bdir, ntok, niters) ? 0 : 1;
            } else if (arg == "--bench-bigvgan-breakdown" && i + 3 < argc) {
                const std::string bdir = argv[++i];
                const uint32_t ntok = static_cast<uint32_t>(std::stoul(argv[++i]));
                const uint32_t niters = static_cast<uint32_t>(std::stoul(argv[++i]));
                return run_bench_bigvgan_breakdown(bdir, ntok, niters) ? 0 : 1;
            } else if (arg == "--test-dit-estimator-pass-tokens" && i + 2 < argc) {
                const std::string bdir = argv[++i];
                const uint32_t ntok = static_cast<uint32_t>(std::stoul(argv[++i]));
                return run_dit_estimator_pass_tokens_test(bdir, ntok) ? 0 : 1;
            } else if (arg == "--test-dit-transformer-block0" && i + 1 < argc) {
                dit_transformer_block0_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-transformer-stack" && i + 1 < argc) {
                dit_transformer_stack_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-post-transformer-proj" && i + 1 < argc) {
                dit_post_transformer_proj_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-final-layer" && i + 1 < argc) {
                dit_final_layer_bundle_dir = argv[++i];
            } else if (arg == "--test-wavenet-layer0-gate" && i + 1 < argc) {
                wavenet_layer0_gate_bundle_dir = argv[++i];
            } else if (arg == "--test-wavenet-stack" && i + 1 < argc) {
                wavenet_stack_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-estimator-step" && i + 1 < argc) {
                dit_estimator_step_bundle_dir = argv[++i];
            } else if (arg == "--test-dit-estimator-golden" && i + 2 < argc) {
                dit_estimator_golden_bundle_dir = argv[++i];
                dit_estimator_golden_dir = argv[++i];
            } else if (arg == "--test-cfm-euler" && i + 1 < argc) {
                cfm_euler_bundle_dir = argv[++i];
            } else if (arg == "--test-cfm-euler-golden" && i + 2 < argc) {
                cfm_euler_golden_bundle_dir = argv[++i];
                cfm_euler_golden_dir = argv[++i];
            } else if (arg == "--test-cfm-euler-cfg" && i + 1 < argc) {
                cfm_euler_cfg_bundle_dir = argv[++i];
            } else if (arg == "--test-cfm-euler-cfg-large" && i + 1 < argc) {
                cfm_euler_cfg_large_bundle_dir = argv[++i];
            } else if (arg == "--test-cfm-euler-cfg-golden" && i + 2 < argc) {
                cfm_euler_cfg_golden_bundle_dir = argv[++i];
                cfm_euler_cfg_golden_dir = argv[++i];
            } else if (arg == "--test-s2mel-full-golden" && i + 2 < argc) {
                s2mel_full_golden_bundle_dir = argv[++i];
                s2mel_full_golden_dir = argv[++i];
            } else if (arg == "--test-s2mel-full-inputs" && i + 8 < argc) {
                s2mel_full_inputs_bundle_dir = argv[++i];
                s2mel_full_inputs_noise_path = argv[++i];
                s2mel_full_inputs_prompt_path = argv[++i];
                s2mel_full_inputs_condition_path = argv[++i];
                s2mel_full_inputs_style_path = argv[++i];
                s2mel_full_inputs_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                s2mel_full_inputs_cfg_rate = std::stof(argv[++i]);
                s2mel_full_inputs_expected_path = argv[++i];
            } else if (arg == "--trace-s2mel-cfm-golden" && i + 2 < argc) {
                trace_s2mel_cfm_bundle_dir = argv[++i];
                trace_s2mel_cfm_golden_dir = argv[++i];
            } else if (arg == "--trace-s2mel-cfm-error-golden" && i + 2 < argc) {
                trace_s2mel_cfm_error_bundle_dir = argv[++i];
                trace_s2mel_cfm_error_golden_dir = argv[++i];
            } else if (arg == "--test-hot-tts-golden" && i + 3 < argc) {
                hot_tts_golden_bundle_dir = argv[++i];
                hot_tts_s2mel_golden_dir = argv[++i];
                hot_tts_wave_golden_dir = argv[++i];
            } else if (arg == "--test-hot-tts-from-gpt-golden" && i + 5 < argc) {
                hot_tts_from_gpt_bundle_dir = argv[++i];
                hot_tts_from_gpt_voice_bundle_dir = argv[++i];
                hot_tts_from_gpt_golden_dir = argv[++i];
                hot_tts_from_gpt_s2mel_golden_dir = argv[++i];
                hot_tts_from_gpt_wave_golden_dir = argv[++i];
            } else if (arg == "--test-hot-tts-from-codes-golden" && i + 5 < argc) {
                hot_tts_from_codes_bundle_dir = argv[++i];
                hot_tts_from_codes_voice_bundle_dir = argv[++i];
                hot_tts_from_codes_golden_dir = argv[++i];
                hot_tts_from_codes_s2mel_golden_dir = argv[++i];
                hot_tts_from_codes_wave_golden_dir = argv[++i];
            } else if (arg == "--synthesize-hot-gpt-golden" && i + 5 < argc) {
                synthesize_hot_gpt_bundle_dir = argv[++i];
                synthesize_hot_gpt_voice_bundle_dir = argv[++i];
                synthesize_hot_gpt_golden_dir = argv[++i];
                synthesize_hot_gpt_s2mel_golden_dir = argv[++i];
                synthesize_hot_gpt_output_wav = argv[++i];
            } else if (arg == "--export-hot-codes-condition-golden" && i + 5 < argc) {
                export_hot_codes_bundle_dir = argv[++i];
                export_hot_codes_voice_bundle_dir = argv[++i];
                export_hot_codes_golden_dir = argv[++i];
                export_hot_codes_s2mel_golden_dir = argv[++i];
                export_hot_codes_output_condition = argv[++i];
            } else if (arg == "--export-hot-codes-condition-input" && i + 6 < argc) {
                export_hot_codes_input_bundle_dir = argv[++i];
                export_hot_codes_input_voice_bundle_dir = argv[++i];
                export_hot_codes_input_golden_dir = argv[++i];
                export_hot_codes_input_s2mel_golden_dir = argv[++i];
                export_hot_codes_input_codes_path = argv[++i];
                export_hot_codes_input_output_condition = argv[++i];
            } else if (arg == "--export-hot-codes-condition-inputs" && i + 7 < argc) {
                export_hot_codes_inputs_bundle_dir = argv[++i];
                export_hot_codes_inputs_voice_bundle_dir = argv[++i];
                export_hot_codes_inputs_conds_path = argv[++i];
                export_hot_codes_inputs_text_ids_path = argv[++i];
                export_hot_codes_inputs_codes_path = argv[++i];
                export_hot_codes_inputs_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                export_hot_codes_inputs_output_condition = argv[++i];
            } else if (arg == "--test-hot-tts-condition-golden" && i + 5 < argc) {
                hot_tts_condition_bundle_dir = argv[++i];
                hot_tts_condition_voice_bundle_dir = argv[++i];
                hot_tts_condition_s2mel_golden_dir = argv[++i];
                hot_tts_condition_wave_golden_dir = argv[++i];
                hot_tts_condition_path = argv[++i];
            } else if (arg == "--synthesize-hot-condition-golden" && i + 5 < argc) {
                synthesize_hot_condition_bundle_dir = argv[++i];
                synthesize_hot_condition_voice_bundle_dir = argv[++i];
                synthesize_hot_condition_s2mel_golden_dir = argv[++i];
                synthesize_hot_condition_path = argv[++i];
                synthesize_hot_condition_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-condition-inputs" && i + 8 < argc) {
                synthesize_hot_condition_inputs_bundle_dir = argv[++i];
                synthesize_hot_condition_inputs_voice_bundle_dir = argv[++i];
                synthesize_hot_condition_inputs_condition_path = argv[++i];
                synthesize_hot_condition_inputs_noise_path = argv[++i];
                synthesize_hot_condition_inputs_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_condition_inputs_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_condition_inputs_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_condition_inputs_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-inputs" && i + 10 < argc) {
                synthesize_hot_inputs_bundle_dir = argv[++i];
                synthesize_hot_inputs_voice_bundle_dir = argv[++i];
                synthesize_hot_inputs_conds_path = argv[++i];
                synthesize_hot_inputs_text_ids_path = argv[++i];
                synthesize_hot_inputs_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_noise_path = argv[++i];
                synthesize_hot_inputs_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_inputs_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-inputs-seeded" && i + 11 < argc) {
                synthesize_hot_inputs_seeded_bundle_dir = argv[++i];
                synthesize_hot_inputs_seeded_voice_bundle_dir = argv[++i];
                synthesize_hot_inputs_seeded_conds_path = argv[++i];
                synthesize_hot_inputs_seeded_text_ids_path = argv[++i];
                synthesize_hot_inputs_seeded_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_seeded_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_inputs_seeded_temperature = std::stof(argv[++i]);
                synthesize_hot_inputs_seeded_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_seeded_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_seeded_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_inputs_seeded_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-text-cjk-seeded" && i + 9 < argc) {
                synthesize_hot_text_cjk_seeded_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_seeded_voice_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_seeded_text = argv[++i];
                synthesize_hot_text_cjk_seeded_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_text_cjk_seeded_temperature = std::stof(argv[++i]);
                synthesize_hot_text_cjk_seeded_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_seeded_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_seeded_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_text_cjk_seeded_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-text-cjk-sampled-seeded" && i + 14 < argc) {
                synthesize_hot_text_cjk_sampled_seeded_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_sampled_seeded_voice_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_sampled_seeded_text = argv[++i];
                synthesize_hot_text_cjk_sampled_seeded_gpt_config.do_sample = true;
                synthesize_hot_text_cjk_sampled_seeded_gpt_config.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_text_cjk_sampled_seeded_gpt_config.temperature = std::stof(argv[++i]);
                synthesize_hot_text_cjk_sampled_seeded_gpt_config.top_k = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_sampled_seeded_gpt_config.top_p = std::stof(argv[++i]);
                synthesize_hot_text_cjk_sampled_seeded_gpt_config.repetition_penalty = std::stof(argv[++i]);
                synthesize_hot_text_cjk_sampled_seeded_noise_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_text_cjk_sampled_seeded_noise_temperature = std::stof(argv[++i]);
                synthesize_hot_text_cjk_sampled_seeded_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_sampled_seeded_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_sampled_seeded_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_text_cjk_sampled_seeded_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-text-cjk-segments-seeded" && i + 11 < argc) {
                synthesize_hot_text_cjk_segments_seeded_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_segments_seeded_voice_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_segments_seeded_text = argv[++i];
                synthesize_hot_text_cjk_segments_seeded_max_text_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_seeded_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_text_cjk_segments_seeded_temperature = std::stof(argv[++i]);
                synthesize_hot_text_cjk_segments_seeded_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_seeded_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_seeded_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_text_cjk_segments_seeded_interval_silence_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_seeded_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-text-cjk-segments-sampled-seeded" && i + 16 < argc) {
                synthesize_hot_text_cjk_segments_sampled_seeded_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_segments_sampled_seeded_voice_bundle_dir = argv[++i];
                synthesize_hot_text_cjk_segments_sampled_seeded_text = argv[++i];
                synthesize_hot_text_cjk_segments_sampled_seeded_max_text_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config.do_sample = true;
                synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config.temperature = std::stof(argv[++i]);
                synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config.top_k = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config.top_p = std::stof(argv[++i]);
                synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config.repetition_penalty = std::stof(argv[++i]);
                synthesize_hot_text_cjk_segments_sampled_seeded_noise_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_text_cjk_segments_sampled_seeded_noise_temperature = std::stof(argv[++i]);
                synthesize_hot_text_cjk_segments_sampled_seeded_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_sampled_seeded_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_sampled_seeded_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_text_cjk_segments_sampled_seeded_interval_silence_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_text_cjk_segments_sampled_seeded_output_wav = argv[++i];
            } else if (arg == "--tts-cjk" && i + 5 < argc) {
                tts_cjk_bundle_dir = argv[++i];
                tts_cjk_voice_bundle_dir = argv[++i];
                tts_cjk_text = argv[++i];
                tts_cjk_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_output_wav = argv[++i];
            } else if (arg == "--tts-cjk-segments" && i + 7 < argc) {
                tts_cjk_segments_bundle_dir = argv[++i];
                tts_cjk_segments_voice_bundle_dir = argv[++i];
                tts_cjk_segments_text = argv[++i];
                tts_cjk_segments_max_text_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_segments_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_segments_interval_silence_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_segments_output_wav = argv[++i];
            } else if (arg == "--tts-cjk-sampled" && i + 5 < argc) {
                tts_cjk_sampled_bundle_dir = argv[++i];
                tts_cjk_sampled_voice_bundle_dir = argv[++i];
                tts_cjk_sampled_text = argv[++i];
                tts_cjk_sampled_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_sampled_output_wav = argv[++i];
            } else if (arg == "--tts-cjk-segments-sampled" && i + 7 < argc) {
                tts_cjk_segments_sampled_bundle_dir = argv[++i];
                tts_cjk_segments_sampled_voice_bundle_dir = argv[++i];
                tts_cjk_segments_sampled_text = argv[++i];
                tts_cjk_segments_sampled_max_text_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_segments_sampled_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_segments_sampled_interval_silence_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
                tts_cjk_segments_sampled_output_wav = argv[++i];
            } else if (arg == "--tts-cjk-preset" && i + 5 < argc) {
                tts_cjk_preset_bundle_dir = argv[++i];
                tts_cjk_preset_voice_bundle_dir = argv[++i];
                tts_cjk_preset_text = argv[++i];
                tts_cjk_preset_name = argv[++i];
                tts_cjk_preset_output_wav = argv[++i];
            } else if (arg == "--tts-cjk-segments-preset" && i + 5 < argc) {
                tts_cjk_segments_preset_bundle_dir = argv[++i];
                tts_cjk_segments_preset_voice_bundle_dir = argv[++i];
                tts_cjk_segments_preset_text = argv[++i];
                tts_cjk_segments_preset_name = argv[++i];
                tts_cjk_segments_preset_output_wav = argv[++i];
            } else if (arg == "--tts-cjk-auto-preset" && i + 5 < argc) {
                tts_cjk_auto_preset_bundle_dir = argv[++i];
                tts_cjk_auto_preset_voice_bundle_dir = argv[++i];
                tts_cjk_auto_preset_text = argv[++i];
                tts_cjk_auto_preset_name = argv[++i];
                tts_cjk_auto_preset_output_wav = argv[++i];
            } else if (arg == "--cfm_steps" && i + 1 < argc) {
                const long n = std::strtol(argv[++i], nullptr, 10);
                if (n < 12 || n > 25) {
                    std::cerr << "error: --cfm_steps must be between 12 and 25 (got " << n << ")" << std::endl;
                    return 1;
                }
                g_cfm_steps_override = static_cast<uint32_t>(n);
            } else if (arg == "--tts" && i + 4 < argc) {
                tts_product_bundle_dir = argv[++i];
                tts_product_voice_bundle_dir = argv[++i];
                tts_product_text = argv[++i];
                tts_product_output_wav = argv[++i];
            } else if (arg == "--prompt_tokens" && i + 1 < argc) {
                const long n = std::strtol(argv[++i], nullptr, 10);
                if (n < 0 || n > 2048) {
                    std::cerr << "error: --prompt_tokens must be between 0 (full) and 2048 (got " << n << ")" << std::endl;
                    return 1;
                }
                g_prompt_tokens_override = static_cast<uint32_t>(n);
            } else if (arg == "--lrucache" && i + 1 < argc) {
                const long n = std::strtol(argv[++i], nullptr, 10);
                if (n < 0 || n > 64) {
                    std::cerr << "error: --lrucache must be between 0 (disabled) and 64 (got " << n << ")" << std::endl;
                    return 1;
                }
                g_voice_cond_cache_size = static_cast<uint32_t>(n);
            } else if (arg == "--serve" && i + 2 < argc) {
                serve_bundle_dir = argv[++i];
                serve_voice_bundle_dir = argv[++i];
            } else if (arg == "--server") {
                http_server_requested = true;
            } else if (arg == "--web") {
                http_server_requested = true;
                http_web_requested = true;
            } else if (arg == "--webkey" && i + 1 < argc) {
                http_web_key = argv[++i];
            } else if (arg == "--host" && i + 1 < argc) {
                http_host = argv[++i];
            } else if (arg == "--port" && i + 1 < argc) {
                const long n = std::strtol(argv[++i], nullptr, 10);
                if (n < 1 || n > 65535) {
                    std::cerr << "error: --port must be 1-65535 (got " << n << ")" << std::endl;
                    return 1;
                }
                http_port = static_cast<uint16_t>(n);
            } else if (arg == "--model_bundle" && i + 1 < argc) {
                http_model_bundle_dir = argv[++i];
            } else if (arg == "--voice_store" && i + 1 < argc) {
                http_voice_store_dir = argv[++i];
            } else if (arg == "--queue_size" && i + 1 < argc) {
                const long n = std::strtol(argv[++i], nullptr, 10);
                if (n < 0 || n > 10000) {
                    std::cerr << "error: --queue_size must be 0-10000 (got " << n << ")" << std::endl;
                    return 1;
                }
                http_queue_size = static_cast<uint32_t>(n);
            } else if (arg == "--tts_concurrency" && i + 1 < argc) {
                const long n = std::strtol(argv[++i], nullptr, 10);
                if (n < 1 || n > 64) {
                    std::cerr << "error: --tts_concurrency must be 1-64 (got " << n << ")" << std::endl;
                    return 1;
                }
                http_tts_concurrency = static_cast<uint32_t>(n);
            } else if (arg == "--clone_concurrency" && i + 1 < argc) {
                const long n = std::strtol(argv[++i], nullptr, 10);
                if (n < 1 || n > 64) {
                    std::cerr << "error: --clone_concurrency must be 1-64 (got " << n << ")" << std::endl;
                    return 1;
                }
                http_clone_concurrency = static_cast<uint32_t>(n);
            } else if (arg == "--tts-cjk-text-readiness" && i + 2 < argc) {
                tts_cjk_text_readiness_bundle_dir = argv[++i];
                tts_cjk_text_readiness_text = argv[++i];
            } else if (arg == "--tts-product-readiness" && i + 2 < argc) {
                tts_product_readiness_bundle_dir = argv[++i];
                tts_product_readiness_voice_bundle_dir = argv[++i];
            } else if (arg == "--synthesize-hot-inputs-sampled-seeded" && i + 16 < argc) {
                synthesize_hot_inputs_sampled_seeded_bundle_dir = argv[++i];
                synthesize_hot_inputs_sampled_seeded_voice_bundle_dir = argv[++i];
                synthesize_hot_inputs_sampled_seeded_conds_path = argv[++i];
                synthesize_hot_inputs_sampled_seeded_text_ids_path = argv[++i];
                synthesize_hot_inputs_sampled_seeded_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_sampled_seeded_gpt_config.do_sample = true;
                synthesize_hot_inputs_sampled_seeded_gpt_config.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_inputs_sampled_seeded_gpt_config.temperature = std::stof(argv[++i]);
                synthesize_hot_inputs_sampled_seeded_gpt_config.top_k = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_sampled_seeded_gpt_config.top_p = std::stof(argv[++i]);
                synthesize_hot_inputs_sampled_seeded_gpt_config.repetition_penalty = std::stof(argv[++i]);
                synthesize_hot_inputs_sampled_seeded_noise_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_inputs_sampled_seeded_noise_temperature = std::stof(argv[++i]);
                synthesize_hot_inputs_sampled_seeded_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_sampled_seeded_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_sampled_seeded_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_inputs_sampled_seeded_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-inputs-seeded-shared" && i + 11 < argc) {
                synthesize_hot_inputs_seeded_shared_bundle_dir = argv[++i];
                synthesize_hot_inputs_seeded_shared_voice_bundle_dir = argv[++i];
                synthesize_hot_inputs_seeded_shared_conds_path = argv[++i];
                synthesize_hot_inputs_seeded_shared_text_ids_path = argv[++i];
                synthesize_hot_inputs_seeded_shared_max_codes = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_seeded_shared_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
                synthesize_hot_inputs_seeded_shared_temperature = std::stof(argv[++i]);
                synthesize_hot_inputs_seeded_shared_prompt_tokens = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_seeded_shared_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
                synthesize_hot_inputs_seeded_shared_cfg_rate = std::stof(argv[++i]);
                synthesize_hot_inputs_seeded_shared_output_wav = argv[++i];
            } else if (arg == "--synthesize-hot-native-golden" && i + 5 < argc) {
                synthesize_hot_native_bundle_dir = argv[++i];
                synthesize_hot_native_voice_bundle_dir = argv[++i];
                synthesize_hot_native_gpt_golden_dir = argv[++i];
                synthesize_hot_native_s2mel_golden_dir = argv[++i];
                synthesize_hot_native_output_wav = argv[++i];
            } else if (arg == "--test-bigvgan-conv-pre" && i + 1 < argc) {
                bigvgan_conv_pre_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-upsample0" && i + 1 < argc) {
                bigvgan_upsample0_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-upsampler-stack" && i + 1 < argc) {
                bigvgan_upsampler_stack_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-front" && i + 1 < argc) {
                bigvgan_front_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-activation-post" && i + 1 < argc) {
                bigvgan_activation_post_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-activation-rb0" && i + 1 < argc) {
                bigvgan_activation_rb0_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-resblock0-pair0" && i + 1 < argc) {
                bigvgan_resblock0_pair0_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-resblock0" && i + 1 < argc) {
                bigvgan_resblock0_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-resblock-group0" && i + 1 < argc) {
                bigvgan_resblock_group0_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-body" && i + 1 < argc) {
                bigvgan_body_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-post" && i + 1 < argc) {
                bigvgan_post_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-vocoder" && i + 1 < argc) {
                bigvgan_vocoder_bundle_dir = argv[++i];
            } else if (arg == "--test-bigvgan-vocoder-golden" && i + 2 < argc) {
                bigvgan_vocoder_golden_bundle_dir = argv[++i];
                bigvgan_vocoder_golden_dir = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                usage(argv[0]);
                return 2;
            }
        }

        if (test_scratch_allocator && !run_scratch_allocator_test()) {
            return 1;
        }

        if (plan_hot_scratch &&
            !run_hot_scratch_plan(plan_hot_scratch_max_prefix_tokens,
                                  plan_hot_scratch_max_codes,
                                  plan_hot_scratch_prompt_tokens)) {
            return 1;
        }

        if (plan_hot_scratch_inputs &&
            !run_hot_scratch_plan_from_inputs(plan_hot_scratch_inputs_conds_path,
                                              plan_hot_scratch_inputs_text_ids_path,
                                              plan_hot_scratch_inputs_max_codes,
                                              plan_hot_scratch_inputs_prompt_tokens)) {
            return 1;
        }

        if (test_metal_scratch_arena && !run_metal_scratch_arena_test()) {
            return 1;
        }

        if (diagnostics || smoke || test_primitives) {
            mit2::MetalContext metal;
            const auto d = metal.diagnostics();
            if (diagnostics || smoke) {
                const auto memory = process_memory_info();
                std::cout << "metal.device=" << d.device_name << "\n";
                std::cout << "metal.unified_memory=" << (d.unified_memory ? "true" : "false") << "\n";
                std::cout << "metal.recommended_max_working_set=" << d.recommended_max_working_set << "\n";
                const auto stats = metal.resource_stats();
                std::cout << "metal.command_buffers_submitted=" << stats.command_buffers_submitted << "\n";
                std::cout << "metal.buffer_allocations=" << stats.buffer_allocations << "\n";
                std::cout << "metal.buffer_bytes_allocated=" << stats.buffer_bytes_allocated << "\n";
                std::cout << "metal.gpu_elapsed_seconds=" << stats.gpu_elapsed_seconds << "\n";
                std::cout << "process.resident_bytes=" << memory.resident_bytes << "\n";
                std::cout << "process.resident_peak_bytes=" << memory.resident_peak_bytes << "\n";
            }
            if (smoke) {
                std::vector<float> values{0.0f, 1.0f, 2.0f, 3.5f, -4.0f, 8.0f};
                if (!metal.smoke_copy(values)) {
                    throw std::runtime_error("Metal copy smoke test failed");
                }
                std::cout << "smoke.copy_f32=";
                for (size_t i = 0; i < values.size(); ++i) {
                    if (i) {
                        std::cout << ",";
                    }
                    std::cout << std::setprecision(3) << values[i];
                }
                std::cout << "\n";
                const auto stats = metal.resource_stats();
                std::cout << "smoke.command_buffers_submitted=" << stats.command_buffers_submitted << "\n";
                std::cout << "smoke.buffer_allocations=" << stats.buffer_allocations << "\n";
                std::cout << "smoke.buffer_bytes_allocated=" << stats.buffer_bytes_allocated << "\n";
                std::cout << "smoke.gpu_elapsed_seconds=" << stats.gpu_elapsed_seconds << "\n";
            }
            if (test_primitives && !run_primitive_tests(metal)) {
                return 1;
            }
        }

        if (!bundle_dir.empty()) {
            mit2::Bundle bundle(bundle_dir);
            std::cout << "bundle.root=" << bundle.root() << "\n";
            std::cout << "bundle.tensor_count=" << bundle.tensor_count() << "\n";
            std::cout << "bundle.tensor_bytes=" << bundle.total_tensor_bytes() << "\n";
            for (const auto& t : bundle.tensors()) {
                std::cout << "tensor " << t.name << " dtype=" << t.dtype << " bytes=" << t.nbytes << " shape=[";
                for (size_t i = 0; i < t.shape.size(); ++i) {
                    if (i) {
                        std::cout << ",";
                    }
                    std::cout << t.shape[i];
                }
                std::cout << "] component=" << t.component << "\n";
            }

            mit2::ScratchAllocator scratch(1024 * 1024);
            scratch.allocate("logits", 8194 * sizeof(float));
            scratch.allocate("codes", 1815 * sizeof(int32_t));
            std::cout << "scratch.peak=" << scratch.peak() << "\n";
        }

        if (!model_bundle_contract_dir.empty() && !inspect_model_bundle_contract(model_bundle_contract_dir)) {
            return 1;
        }

        if (!voice_bundle_contract_dir.empty() && !inspect_voice_bundle_contract(voice_bundle_contract_dir)) {
            return 1;
        }

        if (!gpt_layer_bundle_dir.empty() && !run_gpt_layer_test(gpt_layer_bundle_dir)) {
            return 1;
        }

        if (!gpt_prepare_inputs_bundle_dir.empty() && !run_gpt_prepare_inputs_test(gpt_prepare_inputs_bundle_dir)) {
            return 1;
        }

        if (!gpt_layer0_qkv_bundle_dir.empty() && !run_gpt_layer0_qkv_test(gpt_layer0_qkv_bundle_dir)) {
            return 1;
        }

        if (!gpt_layer0_attention_bundle_dir.empty() && !run_gpt_layer0_attention_test(gpt_layer0_attention_bundle_dir)) {
            return 1;
        }

        if (!gpt_layer0_kv_attention_bundle_dir.empty() && !run_gpt_layer0_kv_attention_test(gpt_layer0_kv_attention_bundle_dir)) {
            return 1;
        }

        if (!gpt_layer0_block_bundle_dir.empty() && !run_gpt_layer0_block_test(gpt_layer0_block_bundle_dir)) {
            return 1;
        }

        if (!gpt_transformer_stack_bundle_dir.empty() && !run_gpt_transformer_stack_test(gpt_transformer_stack_bundle_dir)) {
            return 1;
        }

        if (!gpt_logits_bundle_dir.empty() && !run_gpt_logits_test(gpt_logits_bundle_dir)) {
            return 1;
        }

        if (!gpt_kv_decode_bundle_dir.empty() && !run_gpt_kv_decode_test(gpt_kv_decode_bundle_dir)) {
            return 1;
        }

        if (!gpt_greedy_bundle_dir.empty() && !run_gpt_greedy_test(gpt_greedy_bundle_dir)) {
            return 1;
        }

        if (!gpt_kv_greedy_bundle_dir.empty() && !run_gpt_kv_greedy_test(gpt_kv_greedy_bundle_dir)) {
            return 1;
        }

        if (!gpt_kv_greedy_golden_bundle_dir.empty() && !run_gpt_kv_greedy_golden_test(gpt_kv_greedy_golden_bundle_dir, gpt_kv_greedy_golden_dir)) {
            return 1;
        }

        if (gpt_sampling_processors && !run_gpt_sampling_processors_test()) {
            return 1;
        }

        if (!export_text_ids_cjk_tokenizer_dir.empty() &&
            !run_export_text_ids_cjk(export_text_ids_cjk_tokenizer_dir,
                                     export_text_ids_cjk_text,
                                     export_text_ids_cjk_output_path,
                                     "mit2-text-ids-cjk")) {
            return 1;
        }

        if (!export_text_ids_cjk_segments_tokenizer_dir.empty() &&
            !run_export_text_ids_cjk_segments(export_text_ids_cjk_segments_tokenizer_dir,
                                             export_text_ids_cjk_segments_text,
                                             export_text_ids_cjk_segments_max_tokens,
                                             export_text_ids_cjk_segments_output_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_version_tokenizer_dir.empty() &&
            !run_text_ids_cjk_version_tokenizer_test(test_text_ids_cjk_version_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_slash_tokenizer_dir.empty() &&
            !run_text_ids_cjk_slash_tokenizer_test(test_text_ids_cjk_slash_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_temperature_tokenizer_dir.empty() &&
            !run_text_ids_cjk_temperature_tokenizer_test(test_text_ids_cjk_temperature_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_plus_tokenizer_dir.empty() &&
            !run_text_ids_cjk_plus_tokenizer_test(test_text_ids_cjk_plus_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_operator_tokenizer_dir.empty() &&
            !run_text_ids_cjk_operator_tokenizer_test(test_text_ids_cjk_operator_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_measure_tokenizer_dir.empty() &&
            !run_text_ids_cjk_measure_tokenizer_test(test_text_ids_cjk_measure_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_date_tokenizer_dir.empty() &&
            !run_text_ids_cjk_date_tokenizer_test(test_text_ids_cjk_date_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_time_tokenizer_dir.empty() &&
            !run_text_ids_cjk_time_tokenizer_test(test_text_ids_cjk_time_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_currency_tokenizer_dir.empty() &&
            !run_text_ids_cjk_currency_tokenizer_test(test_text_ids_cjk_currency_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_phone_tokenizer_dir.empty() &&
            !run_text_ids_cjk_phone_tokenizer_test(test_text_ids_cjk_phone_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_ratio_tokenizer_dir.empty() &&
            !run_text_ids_cjk_ratio_tokenizer_test(test_text_ids_cjk_ratio_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_fraction_tokenizer_dir.empty() &&
            !run_text_ids_cjk_fraction_tokenizer_test(test_text_ids_cjk_fraction_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_quote_tokenizer_dir.empty() &&
            !run_text_ids_cjk_quote_tokenizer_test(test_text_ids_cjk_quote_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_ellipsis_tokenizer_dir.empty() &&
            !run_text_ids_cjk_ellipsis_tokenizer_test(test_text_ids_cjk_ellipsis_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_percent_tokenizer_dir.empty() &&
            !run_text_ids_cjk_percent_tokenizer_test(test_text_ids_cjk_percent_tokenizer_dir)) {
            return 1;
        }

        if (!test_text_ids_cjk_no_tokenizer_dir.empty() &&
            !run_text_ids_cjk_no_tokenizer_test(test_text_ids_cjk_no_tokenizer_dir)) {
            return 1;
        }

        if (!tokenize_cjk_smoke_tokenizer_dir.empty() &&
            !run_tokenize_cjk_smoke(tokenize_cjk_smoke_tokenizer_dir,
                                    tokenize_cjk_smoke_text,
                                    tokenize_cjk_smoke_output_path)) {
            return 1;
        }

        if (!gpt_subsampling_golden_bundle_dir.empty() &&
            !run_gpt_subsampling_golden_test(gpt_subsampling_golden_bundle_dir, gpt_subsampling_golden_dir)) {
            return 1;
        }

        if (!gpt_subsampling_metal_linear_golden_bundle_dir.empty() &&
            !run_gpt_subsampling_metal_linear_golden_test(gpt_subsampling_metal_linear_golden_bundle_dir, gpt_subsampling_metal_linear_golden_dir)) {
            return 1;
        }

        if (!export_gpt_subsampling_bundle_dir.empty() &&
            !export_gpt_subsampling(export_gpt_subsampling_bundle_dir,
                                    export_gpt_subsampling_input_path,
                                    export_gpt_subsampling_output_stack_path,
                                    export_gpt_subsampling_output_pos_emb_path,
                                    export_gpt_subsampling_output_mask_path)) {
            return 1;
        }

        if (!gpt_emovec_golden_bundle_dir.empty() &&
            !run_gpt_emovec_golden_test(gpt_emovec_golden_bundle_dir, gpt_emovec_golden_dir)) {
            return 1;
        }

        if (!gpt_emovec_metal_linear_golden_bundle_dir.empty() &&
            !run_gpt_emovec_metal_linear_golden_test(gpt_emovec_metal_linear_golden_bundle_dir, gpt_emovec_metal_linear_golden_dir)) {
            return 1;
        }

        if (!export_gpt_emovec_bundle_dir.empty() &&
            !export_gpt_emovec(export_gpt_emovec_bundle_dir,
                               export_gpt_emovec_input_path,
                               export_gpt_emovec_output_path)) {
            return 1;
        }

        if (!gpt_conformer_ff_golden_bundle_dir.empty() &&
            !run_gpt_conformer_ff_golden_test(gpt_conformer_ff_golden_bundle_dir, gpt_conformer_ff_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_ff_metal_golden_bundle_dir.empty() &&
            !run_gpt_conformer_ff_metal_golden_test(gpt_conformer_ff_metal_golden_bundle_dir, gpt_conformer_ff_metal_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_attn_golden_bundle_dir.empty() &&
            !run_gpt_conformer_attn_golden_test(gpt_conformer_attn_golden_bundle_dir, gpt_conformer_attn_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_attn_metal_proj_golden_bundle_dir.empty() &&
            !run_gpt_conformer_attn_metal_proj_golden_test(gpt_conformer_attn_metal_proj_golden_bundle_dir, gpt_conformer_attn_metal_proj_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_conv_golden_bundle_dir.empty() &&
            !run_gpt_conformer_conv_golden_test(gpt_conformer_conv_golden_bundle_dir, gpt_conformer_conv_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_conv_metal_golden_bundle_dir.empty() &&
            !run_gpt_conformer_conv_metal_golden_test(gpt_conformer_conv_metal_golden_bundle_dir, gpt_conformer_conv_metal_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_block_golden_bundle_dir.empty() &&
            !run_gpt_conformer_block_golden_test(gpt_conformer_block_golden_bundle_dir, gpt_conformer_block_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_block_metal_ff_golden_bundle_dir.empty() &&
            !run_gpt_conformer_block_metal_ff_golden_test(gpt_conformer_block_metal_ff_golden_bundle_dir, gpt_conformer_block_metal_ff_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_block_metal_attn_ff_golden_bundle_dir.empty() &&
            !run_gpt_conformer_block_metal_attn_ff_golden_test(gpt_conformer_block_metal_attn_ff_golden_bundle_dir, gpt_conformer_block_metal_attn_ff_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_block_metal_attn_conv_ff_golden_bundle_dir.empty() &&
            !run_gpt_conformer_block_metal_attn_conv_ff_golden_test(gpt_conformer_block_metal_attn_conv_ff_golden_bundle_dir, gpt_conformer_block_metal_attn_conv_ff_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_stack_golden_bundle_dir.empty() &&
            !run_gpt_conformer_stack_golden_test(gpt_conformer_stack_golden_bundle_dir, gpt_conformer_stack_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_stack_metal_ff_golden_bundle_dir.empty() &&
            !run_gpt_conformer_stack_metal_ff_golden_test(gpt_conformer_stack_metal_ff_golden_bundle_dir, gpt_conformer_stack_metal_ff_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_stack_metal_attn_ff_golden_bundle_dir.empty() &&
            !run_gpt_conformer_stack_metal_attn_ff_golden_test(gpt_conformer_stack_metal_attn_ff_golden_bundle_dir, gpt_conformer_stack_metal_attn_ff_golden_dir)) {
            return 1;
        }

        if (!gpt_conformer_stack_metal_attn_conv_ff_golden_bundle_dir.empty() &&
            !run_gpt_conformer_stack_metal_attn_conv_ff_golden_test(gpt_conformer_stack_metal_attn_conv_ff_golden_bundle_dir, gpt_conformer_stack_metal_attn_conv_ff_golden_dir)) {
            return 1;
        }

        if (!export_gpt_conformer_stack_bundle_dir.empty() &&
            !export_gpt_conformer_stack(export_gpt_conformer_stack_bundle_dir,
                                        export_gpt_conformer_stack_input_path,
                                        export_gpt_conformer_stack_pos_emb_path,
                                        export_gpt_conformer_stack_mask_path,
                                        export_gpt_conformer_stack_output_path)) {
            return 1;
        }

        if (!gpt_perceiver_golden_bundle_dir.empty() &&
            !run_gpt_perceiver_golden_test(gpt_perceiver_golden_bundle_dir, gpt_perceiver_golden_dir)) {
            return 1;
        }

        if (!export_gpt_perceiver_bundle_dir.empty() &&
            !export_gpt_perceiver(export_gpt_perceiver_bundle_dir,
                                  export_gpt_perceiver_context_path,
                                  export_gpt_perceiver_mask_path,
                                  export_gpt_perceiver_output_path)) {
            return 1;
        }

        if (!export_gpt_frontend_tail_bundle_dir.empty() &&
            !export_gpt_frontend_tail(export_gpt_frontend_tail_bundle_dir,
                                      export_gpt_frontend_tail_speech_cond_path,
                                      export_gpt_frontend_tail_emovec_path,
                                      export_gpt_frontend_tail_text_ids_path,
                                      export_gpt_frontend_tail_output_conds_path,
                                      export_gpt_frontend_tail_output_fake_path,
                                      export_gpt_frontend_tail_output_inputs_path,
                                      export_gpt_frontend_tail_output_mask_path)) {
            return 1;
        }

        if (!export_gpt_kv_codes_bundle_dir.empty() &&
            !export_gpt_kv_codes_golden(export_gpt_kv_codes_bundle_dir,
                                        export_gpt_kv_codes_golden_dir,
                                        export_gpt_kv_codes_output_path)) {
            return 1;
        }

        if (!export_gpt_kv_codes_inputs_bundle_dir.empty() &&
            !export_gpt_kv_codes_inputs(export_gpt_kv_codes_inputs_bundle_dir,
                                        export_gpt_kv_codes_inputs_conds_path,
                                        export_gpt_kv_codes_inputs_text_ids_path,
                                        export_gpt_kv_codes_inputs_max_codes,
                                        export_gpt_kv_codes_inputs_output_path)) {
            return 1;
        }

        if (!test_gpt_kv_codes_inputs_bundle_dir.empty() &&
            !run_gpt_kv_codes_inputs_test(test_gpt_kv_codes_inputs_bundle_dir,
                                         test_gpt_kv_codes_inputs_conds_path,
                                         test_gpt_kv_codes_inputs_text_ids_path,
                                         test_gpt_kv_codes_inputs_max_codes,
                                         test_gpt_kv_codes_inputs_expected_path)) {
            return 1;
        }

        if (!export_gpt_kv_codes_inputs_sampled_bundle_dir.empty() &&
            !export_gpt_kv_codes_inputs_sampled(export_gpt_kv_codes_inputs_sampled_bundle_dir,
                                                export_gpt_kv_codes_inputs_sampled_conds_path,
                                                export_gpt_kv_codes_inputs_sampled_text_ids_path,
                                                export_gpt_kv_codes_inputs_sampled_max_codes,
                                                export_gpt_kv_codes_inputs_sampled_config,
                                                export_gpt_kv_codes_inputs_sampled_output_path)) {
            return 1;
        }

        if (!test_gpt_sampled_inputs_determinism_bundle_dir.empty() &&
            !run_gpt_sampled_inputs_determinism_test(test_gpt_sampled_inputs_determinism_bundle_dir,
                                                     test_gpt_sampled_inputs_determinism_conds_path,
                                                     test_gpt_sampled_inputs_determinism_text_ids_path,
                                                     test_gpt_sampled_inputs_determinism_max_codes)) {
            return 1;
        }

        if (!gpt_latent_golden_bundle_dir.empty() && !run_gpt_latent_golden_test(gpt_latent_golden_bundle_dir, gpt_latent_golden_dir)) {
            return 1;
        }

        if (!trace_gpt_latent_bundle_dir.empty() && !trace_gpt_latent_golden(trace_gpt_latent_bundle_dir, trace_gpt_latent_golden_dir)) {
            return 1;
        }

        if (!vq2emb_bundle_dir.empty() && !run_vq2emb_test(vq2emb_bundle_dir)) {
            return 1;
        }

        if (!length_regulator_front_bundle_dir.empty() && !run_length_regulator_front_test(length_regulator_front_bundle_dir)) {
            return 1;
        }

        if (!length_regulator_full_bundle_dir.empty() && !run_length_regulator_full_test(length_regulator_full_bundle_dir)) {
            return 1;
        }

        if (!length_regulator_golden_bundle_dir.empty() && !run_length_regulator_golden_test(length_regulator_golden_bundle_dir, length_regulator_golden_dir)) {
            return 1;
        }

        if (!export_length_regulator_bundle_dir.empty() &&
            !export_length_regulator_golden(export_length_regulator_bundle_dir, export_length_regulator_golden_dir, export_length_regulator_output_path)) {
            return 1;
        }

        if (!export_length_regulator_stages_bundle_dir.empty() &&
            !export_length_regulator_stages_golden(export_length_regulator_stages_bundle_dir,
                                                   export_length_regulator_stages_golden_dir,
                                                   export_length_regulator_stages_output_dir)) {
            return 1;
        }

        if (!timestep_embedder_bundle_dir.empty() && !run_timestep_embedder_test(timestep_embedder_bundle_dir)) {
            return 1;
        }

        if (!dit_input_merge_bundle_dir.empty() && !run_dit_input_merge_test(dit_input_merge_bundle_dir)) {
            return 1;
        }

        if (!dit_attention_projection_bundle_dir.empty() && !run_dit_attention_projection_test(dit_attention_projection_bundle_dir)) {
            return 1;
        }

        if (!dit_feed_forward_bundle_dir.empty() && !run_dit_feed_forward_test(dit_feed_forward_bundle_dir)) {
            return 1;
        }

        if (!dit_adaptive_norm_bundle_dir.empty() && !run_dit_adaptive_norm_test(dit_adaptive_norm_bundle_dir)) {
            return 1;
        }

        if (!dit_attention_core_bundle_dir.empty() && !run_dit_attention_core_test(dit_attention_core_bundle_dir)) {
            return 1;
        }

        if (!dit_transformer_block0_bundle_dir.empty() && !run_dit_transformer_block0_test(dit_transformer_block0_bundle_dir)) {
            return 1;
        }

        if (!dit_transformer_stack_bundle_dir.empty() && !run_dit_transformer_stack_test(dit_transformer_stack_bundle_dir)) {
            return 1;
        }

        if (!dit_post_transformer_proj_bundle_dir.empty() && !run_dit_post_transformer_proj_test(dit_post_transformer_proj_bundle_dir)) {
            return 1;
        }

        if (!dit_final_layer_bundle_dir.empty() && !run_dit_final_layer_test(dit_final_layer_bundle_dir)) {
            return 1;
        }

        if (!wavenet_layer0_gate_bundle_dir.empty() && !run_wavenet_layer0_gate_test(wavenet_layer0_gate_bundle_dir)) {
            return 1;
        }

        if (!wavenet_stack_bundle_dir.empty() && !run_wavenet_stack_test(wavenet_stack_bundle_dir)) {
            return 1;
        }

        if (!dit_estimator_step_bundle_dir.empty() && !run_dit_estimator_step_test(dit_estimator_step_bundle_dir)) {
            return 1;
        }

        if (!dit_estimator_golden_bundle_dir.empty() && !run_dit_estimator_golden_test(dit_estimator_golden_bundle_dir, dit_estimator_golden_dir)) {
            return 1;
        }

        if (!cfm_euler_bundle_dir.empty() && !run_cfm_euler_test(cfm_euler_bundle_dir)) {
            return 1;
        }

        if (!cfm_euler_golden_bundle_dir.empty() && !run_cfm_euler_golden_test(cfm_euler_golden_bundle_dir, cfm_euler_golden_dir)) {
            return 1;
        }

        if (!cfm_euler_cfg_bundle_dir.empty() && !run_cfm_euler_cfg_test(cfm_euler_cfg_bundle_dir)) {
            return 1;
        }

        if (!cfm_euler_cfg_large_bundle_dir.empty() && !run_cfm_euler_cfg_large_test(cfm_euler_cfg_large_bundle_dir)) {
            return 1;
        }

        if (!cfm_euler_cfg_golden_bundle_dir.empty() &&
            !run_cfm_euler_cfg_golden_test(cfm_euler_cfg_golden_bundle_dir, cfm_euler_cfg_golden_dir)) {
            return 1;
        }

        if (!s2mel_full_golden_bundle_dir.empty() && !run_s2mel_full_golden_test(s2mel_full_golden_bundle_dir, s2mel_full_golden_dir)) {
            return 1;
        }

        if (!s2mel_full_inputs_bundle_dir.empty() &&
            !run_s2mel_full_inputs_test(s2mel_full_inputs_bundle_dir,
                                        s2mel_full_inputs_noise_path,
                                        s2mel_full_inputs_prompt_path,
                                        s2mel_full_inputs_condition_path,
                                        s2mel_full_inputs_style_path,
                                        s2mel_full_inputs_steps,
                                        s2mel_full_inputs_cfg_rate,
                                        s2mel_full_inputs_expected_path)) {
            return 1;
        }

        if (!trace_s2mel_cfm_bundle_dir.empty() && !trace_s2mel_cfm_golden(trace_s2mel_cfm_bundle_dir, trace_s2mel_cfm_golden_dir)) {
            return 1;
        }

        if (!trace_s2mel_cfm_error_bundle_dir.empty() && !trace_s2mel_cfm_error_golden(trace_s2mel_cfm_error_bundle_dir, trace_s2mel_cfm_error_golden_dir)) {
            return 1;
        }

        if (!hot_tts_golden_bundle_dir.empty() &&
            !run_hot_tts_golden_test(hot_tts_golden_bundle_dir, hot_tts_s2mel_golden_dir, hot_tts_wave_golden_dir)) {
            return 1;
        }

        if (!hot_tts_from_gpt_bundle_dir.empty() &&
            !run_hot_tts_from_gpt_golden_test(hot_tts_from_gpt_bundle_dir,
                                              hot_tts_from_gpt_voice_bundle_dir,
                                              hot_tts_from_gpt_golden_dir,
                                              hot_tts_from_gpt_s2mel_golden_dir,
                                              hot_tts_from_gpt_wave_golden_dir)) {
            return 1;
        }

        if (!hot_tts_from_codes_bundle_dir.empty() &&
            !run_hot_tts_from_codes_golden_test(hot_tts_from_codes_bundle_dir,
                                                hot_tts_from_codes_voice_bundle_dir,
                                                hot_tts_from_codes_golden_dir,
                                                hot_tts_from_codes_s2mel_golden_dir,
                                                hot_tts_from_codes_wave_golden_dir)) {
            return 1;
        }

        if (!synthesize_hot_gpt_bundle_dir.empty() &&
            !synthesize_hot_gpt_golden_wav(synthesize_hot_gpt_bundle_dir,
                                           synthesize_hot_gpt_voice_bundle_dir,
                                           synthesize_hot_gpt_golden_dir,
                                           synthesize_hot_gpt_s2mel_golden_dir,
                                           synthesize_hot_gpt_output_wav)) {
            return 1;
        }

        if (!export_hot_codes_bundle_dir.empty() &&
            !run_hot_tts_from_codes_golden_test(export_hot_codes_bundle_dir,
                                                export_hot_codes_voice_bundle_dir,
                                                export_hot_codes_golden_dir,
                                                export_hot_codes_s2mel_golden_dir,
                                                "",
                                                export_hot_codes_output_condition)) {
            return 1;
        }

        if (!export_hot_codes_input_bundle_dir.empty() &&
            !run_hot_tts_from_codes_golden_test(export_hot_codes_input_bundle_dir,
                                                export_hot_codes_input_voice_bundle_dir,
                                                export_hot_codes_input_golden_dir,
                                                export_hot_codes_input_s2mel_golden_dir,
                                                "",
                                                export_hot_codes_input_output_condition,
                                                export_hot_codes_input_codes_path)) {
            return 1;
        }

        if (!export_hot_codes_inputs_bundle_dir.empty() &&
            !export_hot_codes_condition_inputs(export_hot_codes_inputs_bundle_dir,
                                               export_hot_codes_inputs_voice_bundle_dir,
                                               export_hot_codes_inputs_conds_path,
                                               export_hot_codes_inputs_text_ids_path,
                                               export_hot_codes_inputs_codes_path,
                                               export_hot_codes_inputs_prompt_tokens,
                                               export_hot_codes_inputs_output_condition)) {
            return 1;
        }

        if (!hot_tts_condition_bundle_dir.empty() &&
            !run_hot_tts_condition_golden_test(hot_tts_condition_bundle_dir,
                                               hot_tts_condition_voice_bundle_dir,
                                               hot_tts_condition_s2mel_golden_dir,
                                               hot_tts_condition_wave_golden_dir,
                                               hot_tts_condition_path)) {
            return 1;
        }

        if (!synthesize_hot_condition_bundle_dir.empty() &&
            !synthesize_hot_condition_golden_wav(synthesize_hot_condition_bundle_dir,
                                                 synthesize_hot_condition_voice_bundle_dir,
                                                 synthesize_hot_condition_s2mel_golden_dir,
                                                 synthesize_hot_condition_path,
                                                 synthesize_hot_condition_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_condition_inputs_bundle_dir.empty() &&
            !synthesize_hot_condition_inputs_wav(synthesize_hot_condition_inputs_bundle_dir,
                                                synthesize_hot_condition_inputs_voice_bundle_dir,
                                                synthesize_hot_condition_inputs_condition_path,
                                                synthesize_hot_condition_inputs_noise_path,
                                                synthesize_hot_condition_inputs_prompt_tokens,
                                                synthesize_hot_condition_inputs_steps,
                                                synthesize_hot_condition_inputs_cfg_rate,
                                                synthesize_hot_condition_inputs_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_inputs_bundle_dir.empty() &&
            !synthesize_hot_inputs_wav(synthesize_hot_inputs_bundle_dir,
                                      synthesize_hot_inputs_voice_bundle_dir,
                                      synthesize_hot_inputs_conds_path,
                                      synthesize_hot_inputs_text_ids_path,
                                      synthesize_hot_inputs_max_codes,
                                      synthesize_hot_inputs_noise_path,
                                      synthesize_hot_inputs_prompt_tokens,
                                      synthesize_hot_inputs_steps,
                                      synthesize_hot_inputs_cfg_rate,
                                      synthesize_hot_inputs_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_inputs_seeded_bundle_dir.empty() &&
            !synthesize_hot_inputs_seeded_wav(synthesize_hot_inputs_seeded_bundle_dir,
                                             synthesize_hot_inputs_seeded_voice_bundle_dir,
                                             synthesize_hot_inputs_seeded_conds_path,
                                             synthesize_hot_inputs_seeded_text_ids_path,
                                             synthesize_hot_inputs_seeded_max_codes,
                                             synthesize_hot_inputs_seeded_seed,
                                             synthesize_hot_inputs_seeded_temperature,
                                             synthesize_hot_inputs_seeded_prompt_tokens,
                                             synthesize_hot_inputs_seeded_steps,
                                             synthesize_hot_inputs_seeded_cfg_rate,
                                             synthesize_hot_inputs_seeded_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_text_cjk_seeded_bundle_dir.empty() &&
            !synthesize_hot_text_cjk_seeded_wav(synthesize_hot_text_cjk_seeded_bundle_dir,
                                               synthesize_hot_text_cjk_seeded_voice_bundle_dir,
                                               synthesize_hot_text_cjk_seeded_text,
                                               synthesize_hot_text_cjk_seeded_seed,
                                               synthesize_hot_text_cjk_seeded_temperature,
                                               synthesize_hot_text_cjk_seeded_prompt_tokens,
                                               synthesize_hot_text_cjk_seeded_steps,
                                               synthesize_hot_text_cjk_seeded_cfg_rate,
                                               synthesize_hot_text_cjk_seeded_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_text_cjk_sampled_seeded_bundle_dir.empty() &&
            !synthesize_hot_text_cjk_sampled_seeded_wav(synthesize_hot_text_cjk_sampled_seeded_bundle_dir,
                                                       synthesize_hot_text_cjk_sampled_seeded_voice_bundle_dir,
                                                       synthesize_hot_text_cjk_sampled_seeded_text,
                                                       synthesize_hot_text_cjk_sampled_seeded_gpt_config,
                                                       synthesize_hot_text_cjk_sampled_seeded_noise_seed,
                                                       synthesize_hot_text_cjk_sampled_seeded_noise_temperature,
                                                       synthesize_hot_text_cjk_sampled_seeded_prompt_tokens,
                                                       synthesize_hot_text_cjk_sampled_seeded_steps,
                                                       synthesize_hot_text_cjk_sampled_seeded_cfg_rate,
                                                       synthesize_hot_text_cjk_sampled_seeded_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_text_cjk_segments_seeded_bundle_dir.empty() &&
            !synthesize_hot_text_cjk_segments_seeded_wav(synthesize_hot_text_cjk_segments_seeded_bundle_dir,
                                                        synthesize_hot_text_cjk_segments_seeded_voice_bundle_dir,
                                                        synthesize_hot_text_cjk_segments_seeded_text,
                                                        synthesize_hot_text_cjk_segments_seeded_max_text_tokens,
                                                        synthesize_hot_text_cjk_segments_seeded_seed,
                                                        synthesize_hot_text_cjk_segments_seeded_temperature,
                                                        synthesize_hot_text_cjk_segments_seeded_prompt_tokens,
                                                        synthesize_hot_text_cjk_segments_seeded_steps,
                                                        synthesize_hot_text_cjk_segments_seeded_cfg_rate,
                                                        synthesize_hot_text_cjk_segments_seeded_interval_silence_ms,
                                                        synthesize_hot_text_cjk_segments_seeded_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_text_cjk_segments_sampled_seeded_bundle_dir.empty() &&
            !synthesize_hot_text_cjk_segments_sampled_seeded_wav(synthesize_hot_text_cjk_segments_sampled_seeded_bundle_dir,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_voice_bundle_dir,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_text,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_max_text_tokens,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_gpt_config,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_noise_seed,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_noise_temperature,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_prompt_tokens,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_steps,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_cfg_rate,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_interval_silence_ms,
                                                                synthesize_hot_text_cjk_segments_sampled_seeded_output_wav)) {
            return 1;
        }

        if (!tts_cjk_bundle_dir.empty() &&
            ((tts_validate_bundles &&
              (!inspect_model_bundle_contract(tts_cjk_bundle_dir) ||
               !inspect_voice_bundle_contract(tts_cjk_voice_bundle_dir))) ||
             !synthesize_hot_text_cjk_seeded_wav(tts_cjk_bundle_dir,
                                                tts_cjk_voice_bundle_dir,
                                                tts_cjk_text,
                                                20240605,
                                                1.0f,
                                                0,
                                                tts_cjk_steps,
                                                0.7f,
                                                tts_cjk_output_wav))) {
                return 1;
        }
        if (!tts_cjk_bundle_dir.empty()) {
            print_tts_product_summary_json("tts-cjk",
                                           tts_cjk_bundle_dir,
                                           tts_cjk_voice_bundle_dir,
                                           tts_cjk_text,
                                           tts_cjk_output_wav,
                                           tts_cjk_steps,
                                           false,
                                           false,
                                           tts_validate_bundles);
        }

        if (!tts_cjk_segments_bundle_dir.empty() &&
            ((tts_validate_bundles &&
              (!inspect_model_bundle_contract(tts_cjk_segments_bundle_dir) ||
               !inspect_voice_bundle_contract(tts_cjk_segments_voice_bundle_dir))) ||
             !synthesize_hot_text_cjk_segments_seeded_wav(tts_cjk_segments_bundle_dir,
                                                         tts_cjk_segments_voice_bundle_dir,
                                                         tts_cjk_segments_text,
                                                         tts_cjk_segments_max_text_tokens,
                                                         20240605,
                                                         1.0f,
                                                         0,
                                                         tts_cjk_segments_steps,
                                                         0.7f,
                                                         tts_cjk_segments_interval_silence_ms,
                                                         tts_cjk_segments_output_wav))) {
                return 1;
        }
        if (!tts_cjk_segments_bundle_dir.empty()) {
            print_tts_product_summary_json("tts-cjk-segments",
                                           tts_cjk_segments_bundle_dir,
                                           tts_cjk_segments_voice_bundle_dir,
                                           tts_cjk_segments_text,
                                           tts_cjk_segments_output_wav,
                                           tts_cjk_segments_steps,
                                           false,
                                           true,
                                           tts_validate_bundles,
                                           tts_cjk_segments_max_text_tokens,
                                           tts_cjk_segments_interval_silence_ms);
        }

        if (!tts_cjk_sampled_bundle_dir.empty()) {
            GptSamplingConfig sampling;
            sampling.do_sample = true;
            sampling.seed = 20240605;
            sampling.temperature = 0.8f;
            sampling.top_k = 30;
            sampling.top_p = 0.8f;
            sampling.repetition_penalty = 10.0f;
            if (tts_validate_bundles &&
                (!inspect_model_bundle_contract(tts_cjk_sampled_bundle_dir) ||
                 !inspect_voice_bundle_contract(tts_cjk_sampled_voice_bundle_dir))) {
                return 1;
            }
            if (!synthesize_hot_text_cjk_sampled_seeded_wav(tts_cjk_sampled_bundle_dir,
                                                            tts_cjk_sampled_voice_bundle_dir,
                                                            tts_cjk_sampled_text,
                                                            sampling,
                                                            20240605,
                                                            1.0f,
                                                            0,
                                                            tts_cjk_sampled_steps,
                                                            0.7f,
                                                            tts_cjk_sampled_output_wav)) {
                return 1;
            }
            print_tts_product_summary_json("tts-cjk-sampled",
                                           tts_cjk_sampled_bundle_dir,
                                           tts_cjk_sampled_voice_bundle_dir,
                                           tts_cjk_sampled_text,
                                           tts_cjk_sampled_output_wav,
                                           tts_cjk_sampled_steps,
                                           true,
                                           false,
                                           tts_validate_bundles);
        }

        if (!tts_cjk_segments_sampled_bundle_dir.empty()) {
            GptSamplingConfig sampling;
            sampling.do_sample = true;
            sampling.seed = 20240605;
            sampling.temperature = 0.8f;
            sampling.top_k = 30;
            sampling.top_p = 0.8f;
            sampling.repetition_penalty = 10.0f;
            if (tts_validate_bundles &&
                (!inspect_model_bundle_contract(tts_cjk_segments_sampled_bundle_dir) ||
                 !inspect_voice_bundle_contract(tts_cjk_segments_sampled_voice_bundle_dir))) {
                return 1;
            }
            if (!synthesize_hot_text_cjk_segments_sampled_seeded_wav(tts_cjk_segments_sampled_bundle_dir,
                                                                     tts_cjk_segments_sampled_voice_bundle_dir,
                                                                     tts_cjk_segments_sampled_text,
                                                                     tts_cjk_segments_sampled_max_text_tokens,
                                                                     sampling,
                                                                     20240605,
                                                                     1.0f,
                                                                     0,
                                                                     tts_cjk_segments_sampled_steps,
                                                                     0.7f,
                                                                     tts_cjk_segments_sampled_interval_silence_ms,
                                                                     tts_cjk_segments_sampled_output_wav)) {
                return 1;
            }
            print_tts_product_summary_json("tts-cjk-segments-sampled",
                                           tts_cjk_segments_sampled_bundle_dir,
                                           tts_cjk_segments_sampled_voice_bundle_dir,
                                           tts_cjk_segments_sampled_text,
                                           tts_cjk_segments_sampled_output_wav,
                                           tts_cjk_segments_sampled_steps,
                                           true,
                                           true,
                                           tts_validate_bundles,
                                           tts_cjk_segments_sampled_max_text_tokens,
                                           tts_cjk_segments_sampled_interval_silence_ms);
        }

        if (!tts_cjk_preset_bundle_dir.empty()) {
            const TtsCjkPreset preset = resolve_tts_cjk_preset(tts_cjk_preset_name);
            if (tts_validate_bundles &&
                (!inspect_model_bundle_contract(tts_cjk_preset_bundle_dir) ||
                 !inspect_voice_bundle_contract(tts_cjk_preset_voice_bundle_dir))) {
                return 1;
            }
            if (!synthesize_hot_text_cjk_seeded_wav(tts_cjk_preset_bundle_dir,
                                                   tts_cjk_preset_voice_bundle_dir,
                                                   tts_cjk_preset_text,
                                                   20240605,
                                                   1.0f,
                                                   0,
                                                   preset.steps,
                                                   0.7f,
                                                   tts_cjk_preset_output_wav)) {
                return 1;
            }
            print_tts_product_summary_json("tts-cjk-preset",
                                           tts_cjk_preset_bundle_dir,
                                           tts_cjk_preset_voice_bundle_dir,
                                           tts_cjk_preset_text,
                                           tts_cjk_preset_output_wav,
                                           preset.steps,
                                           false,
                                           false,
                                           tts_validate_bundles,
                                           0,
                                           0,
                                           preset.name);
        }

        if (!tts_cjk_segments_preset_bundle_dir.empty()) {
            const TtsCjkPreset preset = resolve_tts_cjk_preset(tts_cjk_segments_preset_name);
            if (tts_validate_bundles &&
                (!inspect_model_bundle_contract(tts_cjk_segments_preset_bundle_dir) ||
                 !inspect_voice_bundle_contract(tts_cjk_segments_preset_voice_bundle_dir))) {
                return 1;
            }
            if (!synthesize_hot_text_cjk_segments_seeded_wav(tts_cjk_segments_preset_bundle_dir,
                                                            tts_cjk_segments_preset_voice_bundle_dir,
                                                            tts_cjk_segments_preset_text,
                                                            preset.max_text_tokens_per_segment,
                                                            20240605,
                                                            1.0f,
                                                            0,
                                                            preset.steps,
                                                            0.7f,
                                                            preset.interval_silence_ms,
                                                            tts_cjk_segments_preset_output_wav)) {
                return 1;
            }
            print_tts_product_summary_json("tts-cjk-segments-preset",
                                           tts_cjk_segments_preset_bundle_dir,
                                           tts_cjk_segments_preset_voice_bundle_dir,
                                           tts_cjk_segments_preset_text,
                                           tts_cjk_segments_preset_output_wav,
                                           preset.steps,
                                           false,
                                           true,
                                           tts_validate_bundles,
                                           preset.max_text_tokens_per_segment,
                                           preset.interval_silence_ms,
                                           preset.name);
        }

        if (!tts_cjk_auto_preset_bundle_dir.empty()) {
            const TtsCjkPreset preset = resolve_tts_cjk_preset(tts_cjk_auto_preset_name);
            if (tts_validate_bundles &&
                (!inspect_model_bundle_contract(tts_cjk_auto_preset_bundle_dir) ||
                 !inspect_voice_bundle_contract(tts_cjk_auto_preset_voice_bundle_dir))) {
                return 1;
            }
            const bool segmented = tts_cjk_preset_needs_segments(tts_cjk_auto_preset_bundle_dir,
                                                                 tts_cjk_auto_preset_text,
                                                                 preset.max_text_tokens_per_segment);
            if (segmented) {
                if (!synthesize_hot_text_cjk_segments_seeded_wav(tts_cjk_auto_preset_bundle_dir,
                                                                 tts_cjk_auto_preset_voice_bundle_dir,
                                                                 tts_cjk_auto_preset_text,
                                                                 preset.max_text_tokens_per_segment,
                                                                 20240605,
                                                                 1.0f,
                                                                 0,
                                                                 preset.steps,
                                                                 0.7f,
                                                                 preset.interval_silence_ms,
                                                                 tts_cjk_auto_preset_output_wav)) {
                    return 1;
                }
            } else {
                if (!synthesize_hot_text_cjk_seeded_wav(tts_cjk_auto_preset_bundle_dir,
                                                       tts_cjk_auto_preset_voice_bundle_dir,
                                                       tts_cjk_auto_preset_text,
                                                       20240605,
                                                       1.0f,
                                                       0,
                                                       preset.steps,
                                                       0.7f,
                                                       tts_cjk_auto_preset_output_wav)) {
                    return 1;
                }
            }
            print_tts_product_summary_json("tts-cjk-auto-preset",
                                           tts_cjk_auto_preset_bundle_dir,
                                           tts_cjk_auto_preset_voice_bundle_dir,
                                           tts_cjk_auto_preset_text,
                                           tts_cjk_auto_preset_output_wav,
                                           preset.steps,
                                           false,
                                           segmented,
                                           tts_validate_bundles,
                                           segmented ? preset.max_text_tokens_per_segment : 0,
                                           segmented ? preset.interval_silence_ms : 0,
                                           preset.name,
                                           true);
        }

        if (!tts_product_bundle_dir.empty()) {
            if (!run_tts_product_entry("tts",
                                       tts_product_bundle_dir,
                                       tts_product_voice_bundle_dir,
                                       tts_product_text,
                                       tts_product_output_wav,
                                       "standard")) {
                return 1;
            }
        }

        if (http_server_requested) {
            // Production layout: ./bin holds the MIT2 model bundle
            // (manifest.json + weights.bin + tokenizer/).
            static constexpr const char* kDefaultModelBundleDir = "bin";

            std::string model_dir = http_model_bundle_dir;
            if (model_dir.empty()) {
                if (const char* env = std::getenv("MODEL_BUNDLE")) {
                    model_dir = env;
                } else {
                    model_dir = kDefaultModelBundleDir;
                }
            }
            if (!std::filesystem::exists(std::filesystem::path(model_dir) / "manifest.json")) {
                std::cerr << "error: model bundle not found: " << model_dir
                          << " (pass --model_bundle DIR or set MODEL_BUNDLE)" << std::endl;
                return 1;
            }
            return mit2_server::run_server(http_host,
                                           http_port,
                                           model_dir,
                                           http_voice_store_dir,
                                           http_queue_size,
                                           http_tts_concurrency,
                                           http_clone_concurrency,
                                           http_web_requested,
                                           http_web_key,
                                           g_cli_verbose);
        }

        if (!serve_bundle_dir.empty()) {
            // Resident service mode: one process, all caches (weights, ICB,
            // t-tables, residents) warm after the first request — every
            // subsequent request runs at steady-state RTF. Reads requests from
            // stdin, one per line: "OUTPUT_WAV<TAB>TEXT". Per-request summary
            // goes to stderr; stage JSON stays on stdout for logging.
            std::cerr << "{\"serve\": \"ready\", \"voice\": \"" << serve_voice_bundle_dir << "\"}" << std::endl;
            std::string line;
            uint32_t request_index = 0;
            while (std::getline(std::cin, line)) {
                if (line.empty()) {
                    continue;
                }
                const size_t tab = line.find('\t');
                std::string out_wav;
                std::string text;
                if (tab == std::string::npos) {
                    out_wav = "serve_out_" + std::to_string(request_index) + ".wav";
                    text = line;
                } else {
                    out_wav = line.substr(0, tab);
                    text = line.substr(tab + 1);
                }
                const auto t0 = std::chrono::steady_clock::now();
                bool ok = false;
                try {
                    ok = run_tts_product_entry("tts", serve_bundle_dir, serve_voice_bundle_dir, text, out_wav, "standard");
                } catch (const std::exception& e) {
                    std::cerr << "{\"serve_request\": " << request_index << ", \"ok\": false, \"error\": \"" << e.what() << "\"}" << std::endl;
                    ++request_index;
                    continue;
                }
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                double audio_seconds = 0.0;
                {
                    std::ifstream f(out_wav, std::ios::binary);
                    if (f) {
                        f.seekg(0, std::ios::end);
                        const auto bytes = static_cast<long long>(f.tellg());
                        if (bytes > 44) {
                            audio_seconds = static_cast<double>(bytes - 44) / 2.0 / 22050.0;
                        }
                    }
                }
                std::cerr << "{\"serve_request\": " << request_index
                          << ", \"ok\": " << (ok ? "true" : "false")
                          << ", \"output\": \"" << out_wav << "\""
                          << ", \"audio_seconds\": " << audio_seconds
                          << ", \"wall_seconds\": " << wall
                          << ", \"rtf\": " << (audio_seconds > 0 ? wall / audio_seconds : 0.0)
                          << "}" << std::endl;
                ++request_index;
            }
            return 0;
        }

        if (!tts_product_readiness_bundle_dir.empty() &&
            !inspect_tts_product_readiness(tts_product_readiness_bundle_dir,
                                           tts_product_readiness_voice_bundle_dir)) {
            return 1;
        }

        if (!tts_cjk_text_readiness_bundle_dir.empty() &&
            !inspect_tts_cjk_text_readiness(tts_cjk_text_readiness_bundle_dir,
                                            tts_cjk_text_readiness_text)) {
            return 1;
        }

        if (!synthesize_hot_inputs_sampled_seeded_bundle_dir.empty() &&
            !synthesize_hot_inputs_sampled_seeded_wav(synthesize_hot_inputs_sampled_seeded_bundle_dir,
                                                     synthesize_hot_inputs_sampled_seeded_voice_bundle_dir,
                                                     synthesize_hot_inputs_sampled_seeded_conds_path,
                                                     synthesize_hot_inputs_sampled_seeded_text_ids_path,
                                                     synthesize_hot_inputs_sampled_seeded_max_codes,
                                                     synthesize_hot_inputs_sampled_seeded_gpt_config,
                                                     synthesize_hot_inputs_sampled_seeded_noise_seed,
                                                     synthesize_hot_inputs_sampled_seeded_noise_temperature,
                                                     synthesize_hot_inputs_sampled_seeded_prompt_tokens,
                                                     synthesize_hot_inputs_sampled_seeded_steps,
                                                     synthesize_hot_inputs_sampled_seeded_cfg_rate,
                                                     synthesize_hot_inputs_sampled_seeded_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_inputs_seeded_shared_bundle_dir.empty() &&
            !synthesize_hot_inputs_seeded_shared_wav(synthesize_hot_inputs_seeded_shared_bundle_dir,
                                                    synthesize_hot_inputs_seeded_shared_voice_bundle_dir,
                                                    synthesize_hot_inputs_seeded_shared_conds_path,
                                                    synthesize_hot_inputs_seeded_shared_text_ids_path,
                                                    synthesize_hot_inputs_seeded_shared_max_codes,
                                                    synthesize_hot_inputs_seeded_shared_seed,
                                                    synthesize_hot_inputs_seeded_shared_temperature,
                                                    synthesize_hot_inputs_seeded_shared_prompt_tokens,
                                                    synthesize_hot_inputs_seeded_shared_steps,
                                                    synthesize_hot_inputs_seeded_shared_cfg_rate,
                                                    synthesize_hot_inputs_seeded_shared_output_wav)) {
            return 1;
        }

        if (!synthesize_hot_native_bundle_dir.empty() &&
            !synthesize_hot_native_golden_wav(synthesize_hot_native_bundle_dir,
                                              synthesize_hot_native_voice_bundle_dir,
                                              synthesize_hot_native_gpt_golden_dir,
                                              synthesize_hot_native_s2mel_golden_dir,
                                              synthesize_hot_native_output_wav)) {
            return 1;
        }

        if (!bigvgan_conv_pre_bundle_dir.empty() && !run_bigvgan_conv_pre_test(bigvgan_conv_pre_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_upsample0_bundle_dir.empty() && !run_bigvgan_upsample0_test(bigvgan_upsample0_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_upsampler_stack_bundle_dir.empty() && !run_bigvgan_upsampler_stack_test(bigvgan_upsampler_stack_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_front_bundle_dir.empty() && !run_bigvgan_front_test(bigvgan_front_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_activation_post_bundle_dir.empty() && !run_bigvgan_activation_post_test(bigvgan_activation_post_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_activation_rb0_bundle_dir.empty() && !run_bigvgan_activation_rb0_test(bigvgan_activation_rb0_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_resblock0_pair0_bundle_dir.empty() && !run_bigvgan_resblock0_pair0_test(bigvgan_resblock0_pair0_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_resblock0_bundle_dir.empty() && !run_bigvgan_resblock0_test(bigvgan_resblock0_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_resblock_group0_bundle_dir.empty() && !run_bigvgan_resblock_group0_test(bigvgan_resblock_group0_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_body_bundle_dir.empty() && !run_bigvgan_body_test(bigvgan_body_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_post_bundle_dir.empty() && !run_bigvgan_post_test(bigvgan_post_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_vocoder_bundle_dir.empty() && !run_bigvgan_vocoder_test(bigvgan_vocoder_bundle_dir)) {
            return 1;
        }

        if (!bigvgan_vocoder_golden_bundle_dir.empty() &&
            !run_bigvgan_vocoder_golden_test(bigvgan_vocoder_golden_bundle_dir, bigvgan_vocoder_golden_dir)) {
            return 1;
        }

        if (!diagnostics && !smoke && !test_primitives && !test_scratch_allocator && !test_metal_scratch_arena && !plan_hot_scratch && !plan_hot_scratch_inputs && bundle_dir.empty() && gpt_layer_bundle_dir.empty() &&
            gpt_prepare_inputs_bundle_dir.empty() &&
            gpt_layer0_qkv_bundle_dir.empty() &&
            gpt_layer0_attention_bundle_dir.empty() &&
            gpt_layer0_kv_attention_bundle_dir.empty() &&
            gpt_layer0_block_bundle_dir.empty() &&
            gpt_transformer_stack_bundle_dir.empty() &&
            gpt_logits_bundle_dir.empty() &&
            gpt_kv_decode_bundle_dir.empty() &&
            gpt_greedy_bundle_dir.empty() &&
            gpt_kv_greedy_bundle_dir.empty() &&
            gpt_kv_greedy_golden_bundle_dir.empty() &&
            model_bundle_contract_dir.empty() &&
            voice_bundle_contract_dir.empty() &&
            !gpt_sampling_processors &&
            export_text_ids_cjk_tokenizer_dir.empty() &&
            export_text_ids_cjk_segments_tokenizer_dir.empty() &&
            test_text_ids_cjk_version_tokenizer_dir.empty() &&
            test_text_ids_cjk_slash_tokenizer_dir.empty() &&
            test_text_ids_cjk_temperature_tokenizer_dir.empty() &&
            test_text_ids_cjk_plus_tokenizer_dir.empty() &&
            test_text_ids_cjk_operator_tokenizer_dir.empty() &&
            test_text_ids_cjk_measure_tokenizer_dir.empty() &&
            test_text_ids_cjk_date_tokenizer_dir.empty() &&
            test_text_ids_cjk_time_tokenizer_dir.empty() &&
            test_text_ids_cjk_currency_tokenizer_dir.empty() &&
            test_text_ids_cjk_phone_tokenizer_dir.empty() &&
            test_text_ids_cjk_ratio_tokenizer_dir.empty() &&
            test_text_ids_cjk_fraction_tokenizer_dir.empty() &&
            test_text_ids_cjk_quote_tokenizer_dir.empty() &&
            test_text_ids_cjk_ellipsis_tokenizer_dir.empty() &&
            test_text_ids_cjk_percent_tokenizer_dir.empty() &&
            test_text_ids_cjk_no_tokenizer_dir.empty() &&
            tokenize_cjk_smoke_tokenizer_dir.empty() &&
            gpt_subsampling_golden_bundle_dir.empty() &&
            gpt_subsampling_metal_linear_golden_bundle_dir.empty() &&
            export_gpt_subsampling_bundle_dir.empty() &&
            gpt_emovec_golden_bundle_dir.empty() &&
            gpt_emovec_metal_linear_golden_bundle_dir.empty() &&
            export_gpt_emovec_bundle_dir.empty() &&
            gpt_conformer_ff_golden_bundle_dir.empty() &&
            gpt_conformer_ff_metal_golden_bundle_dir.empty() &&
            gpt_conformer_attn_golden_bundle_dir.empty() &&
            gpt_conformer_attn_metal_proj_golden_bundle_dir.empty() &&
            gpt_conformer_conv_golden_bundle_dir.empty() &&
            gpt_conformer_conv_metal_golden_bundle_dir.empty() &&
            gpt_conformer_block_golden_bundle_dir.empty() &&
            gpt_conformer_block_metal_ff_golden_bundle_dir.empty() &&
            gpt_conformer_block_metal_attn_ff_golden_bundle_dir.empty() &&
            gpt_conformer_block_metal_attn_conv_ff_golden_bundle_dir.empty() &&
            gpt_conformer_stack_golden_bundle_dir.empty() &&
            gpt_conformer_stack_metal_ff_golden_bundle_dir.empty() &&
            gpt_conformer_stack_metal_attn_ff_golden_bundle_dir.empty() &&
            gpt_conformer_stack_metal_attn_conv_ff_golden_bundle_dir.empty() &&
            export_gpt_conformer_stack_bundle_dir.empty() &&
            gpt_perceiver_golden_bundle_dir.empty() &&
            export_gpt_perceiver_bundle_dir.empty() &&
            export_gpt_frontend_tail_bundle_dir.empty() &&
            export_gpt_kv_codes_bundle_dir.empty() &&
            export_gpt_kv_codes_inputs_bundle_dir.empty() &&
            test_gpt_kv_codes_inputs_bundle_dir.empty() &&
            export_gpt_kv_codes_inputs_sampled_bundle_dir.empty() &&
            test_gpt_sampled_inputs_determinism_bundle_dir.empty() &&
            gpt_latent_golden_bundle_dir.empty() &&
            trace_gpt_latent_bundle_dir.empty() &&
            vq2emb_bundle_dir.empty() && length_regulator_front_bundle_dir.empty() && length_regulator_full_bundle_dir.empty() &&
            length_regulator_golden_bundle_dir.empty() &&
            export_length_regulator_bundle_dir.empty() &&
            export_length_regulator_stages_bundle_dir.empty() &&
            timestep_embedder_bundle_dir.empty() && dit_input_merge_bundle_dir.empty() && dit_attention_projection_bundle_dir.empty() &&
            dit_feed_forward_bundle_dir.empty() && dit_adaptive_norm_bundle_dir.empty() && dit_attention_core_bundle_dir.empty() &&
            dit_transformer_block0_bundle_dir.empty() && dit_transformer_stack_bundle_dir.empty() &&
            dit_post_transformer_proj_bundle_dir.empty() && dit_final_layer_bundle_dir.empty() &&
            wavenet_layer0_gate_bundle_dir.empty() && wavenet_stack_bundle_dir.empty() && dit_estimator_step_bundle_dir.empty() &&
            dit_estimator_golden_bundle_dir.empty() &&
            cfm_euler_bundle_dir.empty() && cfm_euler_golden_bundle_dir.empty() && cfm_euler_cfg_bundle_dir.empty() &&
            cfm_euler_cfg_golden_bundle_dir.empty() && s2mel_full_golden_bundle_dir.empty() &&
            s2mel_full_inputs_bundle_dir.empty() &&
            trace_s2mel_cfm_bundle_dir.empty() &&
            trace_s2mel_cfm_error_bundle_dir.empty() &&
            hot_tts_golden_bundle_dir.empty() && hot_tts_from_gpt_bundle_dir.empty() && hot_tts_from_codes_bundle_dir.empty() &&
            synthesize_hot_gpt_bundle_dir.empty() &&
            export_hot_codes_bundle_dir.empty() &&
            export_hot_codes_input_bundle_dir.empty() &&
            export_hot_codes_inputs_bundle_dir.empty() &&
            hot_tts_condition_bundle_dir.empty() &&
            synthesize_hot_condition_bundle_dir.empty() &&
            synthesize_hot_condition_inputs_bundle_dir.empty() &&
            synthesize_hot_inputs_bundle_dir.empty() &&
            synthesize_hot_inputs_seeded_bundle_dir.empty() &&
            synthesize_hot_text_cjk_seeded_bundle_dir.empty() &&
            synthesize_hot_text_cjk_sampled_seeded_bundle_dir.empty() &&
            synthesize_hot_text_cjk_segments_seeded_bundle_dir.empty() &&
            synthesize_hot_text_cjk_segments_sampled_seeded_bundle_dir.empty() &&
            tts_cjk_bundle_dir.empty() &&
            tts_cjk_segments_bundle_dir.empty() &&
            tts_cjk_sampled_bundle_dir.empty() &&
            tts_cjk_segments_sampled_bundle_dir.empty() &&
            tts_cjk_preset_bundle_dir.empty() &&
            tts_cjk_segments_preset_bundle_dir.empty() &&
            tts_cjk_auto_preset_bundle_dir.empty() &&
            tts_product_bundle_dir.empty() &&
            tts_cjk_text_readiness_bundle_dir.empty() &&
            tts_product_readiness_bundle_dir.empty() &&
            synthesize_hot_inputs_sampled_seeded_bundle_dir.empty() &&
            synthesize_hot_inputs_seeded_shared_bundle_dir.empty() &&
            synthesize_hot_native_bundle_dir.empty() &&
            bigvgan_conv_pre_bundle_dir.empty() &&
            bigvgan_upsample0_bundle_dir.empty() && bigvgan_upsampler_stack_bundle_dir.empty() && bigvgan_front_bundle_dir.empty() &&
            bigvgan_activation_post_bundle_dir.empty() && bigvgan_activation_rb0_bundle_dir.empty() &&
            bigvgan_resblock0_pair0_bundle_dir.empty() && bigvgan_resblock0_bundle_dir.empty() &&
            bigvgan_resblock_group0_bundle_dir.empty() && bigvgan_body_bundle_dir.empty() &&
            bigvgan_post_bundle_dir.empty() && bigvgan_vocoder_bundle_dir.empty() &&
            bigvgan_vocoder_golden_bundle_dir.empty()) {
            usage(argv[0]);
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

std::string mtts_json_escape(const std::string& value) {
    return json_escape(value);
}

void server_reset_tts_stage_acc() {
    g_tts_stage_acc.reset();
}

bool server_run_tts_clone_native(const std::string& model_bundle_dir,
                                 const std::string& audio_wav,
                                 const std::string& output_voice_bundle) {
    return run_tts_clone_native(model_bundle_dir, audio_wav, output_voice_bundle);
}

bool server_run_tts_clone_fast(const std::string& audio_wav, const std::string& output_voice_bundle) {
    return run_tts_clone_fast(audio_wav, output_voice_bundle);
}

bool server_run_tts_product_entry(const std::string& command,
                                  const std::string& bundle_dir,
                                  const std::string& voice_bundle_dir,
                                  const std::string& text,
                                  const std::string& output_wav,
                                  const std::string& preset) {
    return run_tts_product_entry(command, bundle_dir, voice_bundle_dir, text, output_wav, preset);
}

// For the product commands (--tts, --clone) the stage-by-stage JSON reports on
// stdout are a debugging surface, not user output. By default they are captured
// and only replayed (to stderr) when the command fails; --verbose streams them
// through as before. Inspection commands (--capabilities, --readiness, ...)
// keep printing JSON: that output is their machine-readable contract.
int main(int argc, char** argv) {
    bool verbose = false;
    bool product_command = false;
    std::vector<char*> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--verbose") {
            verbose = true;
            g_cli_verbose = true;
            continue;
        }
        if (arg == "--tts" || arg == "--clone" || arg == "--clone-fast") {
            product_command = true;
        }
        args.push_back(argv[i]);
    }
    // Positional product form: mit2_tts MODEL_BUNDLE VOICE_BUNDLE TEXT OUT [PRESET]
    if (args.size() >= 2 && args[1][0] != '-') {
        product_command = true;
    }
    if (verbose || !product_command) {
        return run_cli(static_cast<int>(args.size()), args.data());
    }

    std::ostringstream captured;
    std::streambuf* previous = std::cout.rdbuf(captured.rdbuf());
    int rc = 1;
    try {
        rc = run_cli(static_cast<int>(args.size()), args.data());
    } catch (...) {
        std::cout.rdbuf(previous);
        std::cerr << captured.str();
        throw;
    }
    std::cout.rdbuf(previous);
    if (rc != 0) {
        std::cerr << captured.str();
    }
    return rc;
}
