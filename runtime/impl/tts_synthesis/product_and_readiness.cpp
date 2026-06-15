// TTS product command entry + readiness: segment planning, CJK text readiness, product summary JSON, run_tts_product_entry, and product readiness inspection.
// Part of the tts_synthesis translation unit (see ../tts_synthesis.cpp);
// these files are #included in order into a single TU, so definition order
// across files is significant.

void print_json_cjk_segments(const std::vector<CjkTokenizedSegment>& segments) {
    std::cout << "[\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        std::cout << "      {\n";
        std::cout << "        \"index\": " << i << ",\n";
        std::cout << "        \"pieces\": ";
        print_json_string_array(segments[i].pieces);
        std::cout << ",\n";
        std::cout << "        \"ids\": ";
        print_json_u32_array(segments[i].ids);
        std::cout << "\n";
        std::cout << "      }" << (i + 1 == segments.size() ? "\n" : ",\n");
    }
    std::cout << "    ]";
}

bool inspect_tts_cjk_text_readiness(const std::string& bundle_dir, const std::string& text) {
    const std::string tokenizer_dir = bundle_dir + "/tokenizer";
    try {
        const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
        const auto tokenized = tokenize_cjk_text(piece_to_id, text);
        const std::vector<TtsCjkPreset> presets{
            resolve_tts_cjk_preset("smoke"),
            resolve_tts_cjk_preset("short"),
            resolve_tts_cjk_preset("standard"),
        };
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_cjk_text_readiness\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"ready_native_cjk_text\": true,\n";
        std::cout << "  \"native_text_surface\": \"focused_cjk_limited_ascii\",\n";
        std::cout << "  \"ready_cached_voice_tts_cjk_text\": true,\n";
        std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
        std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
        std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
        std::cout << "  \"token_count\": " << tokenized.ids.size() << ",\n";
        std::cout << "  \"tokens\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "  \"token_ids\": ";
        print_json_u32_array(tokenized.ids);
        std::cout << ",\n";
        std::cout << "  \"presets\": [\n";
        for (size_t i = 0; i < presets.size(); ++i) {
            const auto segments = split_cjk_tokenized_text(tokenized, presets[i].max_text_tokens_per_segment);
            std::cout << "    {\n";
            std::cout << "      \"name\": \"" << json_escape(presets[i].name) << "\",\n";
            std::cout << "      \"max_codes_per_text_token\": " << kMaxCodesPerTextToken << ",\n";
            std::cout << "      \"steps\": " << presets[i].steps << ",\n";
            std::cout << "      \"max_text_tokens_per_segment\": " << presets[i].max_text_tokens_per_segment << ",\n";
            std::cout << "      \"interval_silence_ms\": " << presets[i].interval_silence_ms << ",\n";
            std::cout << "      \"segment_count\": " << segments.size() << ",\n";
            std::cout << "      \"auto_segmented\": " << (segments.size() > 1 ? "true" : "false") << ",\n";
            std::cout << "      \"segments\": ";
            print_json_cjk_segments(segments);
            std::cout << "\n";
            std::cout << "    }" << (i + 1 == presets.size() ? "\n" : ",\n");
        }
        std::cout << "  ],\n";
        std::cout << "  \"python_boundary_if_false\": \"full TextNormalizer/SentencePiece for general text\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_cjk_text_readiness\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"ready_native_cjk_text\": false,\n";
        std::cout << "  \"native_text_surface\": \"focused_cjk_limited_ascii\",\n";
        std::cout << "  \"ready_cached_voice_tts_cjk_text\": false,\n";
        std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
        std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
        std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\",\n";
        std::cout << "  \"python_boundary\": \"full TextNormalizer/SentencePiece for general text\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<std::vector<uint32_t>> read_segment_code_sidecars(const std::string& output_wav) {
    std::vector<std::vector<uint32_t>> segments;
    const auto segment_dir = std::filesystem::path(output_wav + ".segments");
    for (size_t i = 0;; ++i) {
        std::ostringstream name;
        name << "segment_" << std::setw(3) << std::setfill('0') << i << ".wav.codes.u32";
        const auto path = segment_dir / name.str();
        if (!std::filesystem::exists(path)) {
            break;
        }
        segments.push_back(read_raw_u32(path.string()));
    }
    return segments;
}

void print_json_u32_nested_array(const std::vector<std::vector<uint32_t>>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        print_json_u32_array(values[i]);
    }
    std::cout << "]";
}

void print_tts_product_summary_json(const std::string& command,
                                    const std::string& bundle_dir,
                                    const std::string& voice_bundle_dir,
                                    const std::string& text,
                                    const std::string& output_wav,
                                    uint32_t steps,
                                    bool sampled,
                                    bool segmented,
                                    bool validated_bundles,
                                    uint32_t max_text_tokens_per_segment = 0,
                                    uint32_t interval_silence_ms = 0,
                                    const std::string& preset = "",
                                    bool auto_selected = false) {
    const auto wav = read_wav_pcm16_mono_bytes(output_wav);
    const auto memory = process_memory_info();
    const auto predicted_codes = segmented ? std::vector<uint32_t>{} : read_raw_u32(output_wav + ".codes.u32");
    const auto segment_predicted_codes = segmented ? read_segment_code_sidecars(output_wav) : std::vector<std::vector<uint32_t>>{};
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_cjk_product\",\n";
    std::cout << "  \"command\": \"" << json_escape(command) << "\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"output_wav\": \"" << json_escape(output_wav) << "\",\n";
    std::cout << "  \"output_wav_sha256\": \"" << file_sha256_hex(output_wav) << "\",\n";
    std::cout << "  \"sample_rate\": " << wav.sample_rate << ",\n";
    std::cout << "  \"samples\": " << (wav.frames.size() / 2) << ",\n";
    std::cout << "  \"max_codes_per_text_token\": " << kMaxCodesPerTextToken << ",\n";
    if (segmented) {
        std::cout << "  \"max_text_tokens_per_segment\": " << max_text_tokens_per_segment << ",\n";
        std::cout << "  \"interval_silence_ms\": " << interval_silence_ms << ",\n";
    }
    if (!preset.empty()) {
        std::cout << "  \"preset\": \"" << json_escape(preset) << "\",\n";
    }
    if (auto_selected) {
        std::cout << "  \"auto_segmented\": " << (segmented ? "true" : "false") << ",\n";
    }
    if (segmented) {
        std::cout << "  \"segment_predicted_codes\": ";
        print_json_u32_nested_array(segment_predicted_codes);
        std::cout << ",\n";
    } else {
        std::cout << "  \"predicted_codes\": ";
        print_json_u32_array(predicted_codes);
        std::cout << ",\n";
    }
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"sampled\": " << (sampled ? "true" : "false") << ",\n";
    std::cout << "  \"segmented\": " << (segmented ? "true" : "false") << ",\n";
    std::cout << "  \"validated_bundles\": " << (validated_bundles ? "true" : "false") << ",\n";
    std::cout << "  \"prompt_tokens_source\": \"voice_bundle\",\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << "\n";
    std::cout << "}\n";
}

bool run_tts_product_entry(const std::string& command,
                           const std::string& bundle_dir,
                           const std::string& voice_bundle_dir,
                           const std::string& text,
                           const std::string& output_wav,
                           const std::string& preset_name) {
    const TtsCjkPreset preset = resolve_tts_cjk_preset(preset_name);
    // Contract validation (full sha256 sweep of the 7.7GB bundle, ~3-4s) is
    // per-directory work — cache it so resident-service requests only pay once.
    {
        static std::set<std::string> validated_dirs;
        if (validated_dirs.find(bundle_dir) == validated_dirs.end()) {
            if (!inspect_model_bundle_contract(bundle_dir)) {
                return false;
            }
            validated_dirs.insert(bundle_dir);
        }
        if (validated_dirs.find(voice_bundle_dir) == validated_dirs.end()) {
            if (!inspect_voice_bundle_contract(voice_bundle_dir)) {
                return false;
            }
            validated_dirs.insert(voice_bundle_dir);
        }
    }
    const bool segmented = tts_cjk_preset_needs_segments(bundle_dir,
                                                         text,
                                                         preset.max_text_tokens_per_segment);
    const GptSamplingConfig sampling = reference_default_gpt_sampling(20240605);
    if (segmented) {
        if (!synthesize_hot_text_cjk_segments_sampled_seeded_wav(bundle_dir,
                                                                 voice_bundle_dir,
                                                                 text,
                                                                 preset.max_text_tokens_per_segment,
                                                                 sampling,
                                                                 20240605,
                                                                 1.0f,
                                                                 g_prompt_tokens_override,
                                                                 preset.steps,
                                                                 0.7f,
                                                                 preset.interval_silence_ms,
                                                                 output_wav)) {
            return false;
        }
    } else {
        if (!synthesize_hot_text_cjk_sampled_seeded_wav(bundle_dir,
                                                       voice_bundle_dir,
                                                       text,
                                                       sampling,
                                                       20240605,
                                                       1.0f,
                                                       g_prompt_tokens_override,
                                                       preset.steps,
                                                       0.7f,
                                                       output_wav)) {
            return false;
        }
    }
    print_tts_product_summary_json(command,
                                   bundle_dir,
                                   voice_bundle_dir,
                                   text,
                                   output_wav,
                                   preset.steps,
                                   true,
                                   segmented,
                                   true,
                                   segmented ? preset.max_text_tokens_per_segment : 0,
                                   segmented ? preset.interval_silence_ms : 0,
                                   preset.name,
                                   true);
    return true;
}

bool inspect_tts_product_readiness(const std::string& bundle_dir, const std::string& voice_bundle_dir) {
    const bool model_ok = inspect_model_bundle_contract(bundle_dir);
    const bool voice_ok = inspect_voice_bundle_contract(voice_bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    const uint32_t prompt_tokens = infer_voice_prompt_tokens(voice);
    const auto memory = process_memory_info();
    const bool ok = model_ok && voice_ok && prompt_tokens > 0;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_product_readiness\",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"native_text_surface\": \"focused_cjk_limited_ascii\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"model_contract_ok\": " << (model_ok ? "true" : "false") << ",\n";
    std::cout << "  \"voice_contract_ok\": " << (voice_ok ? "true" : "false") << ",\n";
    std::cout << "  \"voice_prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"ready_cached_voice_tts_cjk\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"ready_cached_voice_tts_general_text\": false,\n";
    std::cout << "  \"ready_native_voice_clone\": false,\n";
    std::cout << "  \"supported_product_surfaces\": [\n";
    std::cout << "    \"cached_voice_cjk_text_to_wav\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"unsupported_product_surfaces\": [\n";
    std::cout << "    \"cached_voice_general_text_to_wav\",\n";
    std::cout << "    \"native_clone_audio_to_voice_bundle\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"python_boundaries\": [\n";
    std::cout << "    \"full TextNormalizer/SentencePiece for general text\",\n";
    std::cout << "    \"clone-time semantic/acoustic speech encoders for voice tensor creation\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"native_product_commands\": [\n";
    std::cout << "    \"--tts-cjk\",\n";
    std::cout << "    \"--tts-cjk-segments\",\n";
    std::cout << "    \"--tts-cjk-sampled\",\n";
    std::cout << "    \"--tts-cjk-segments-sampled\",\n";
    std::cout << "    \"--tts-cjk-preset\",\n";
    std::cout << "    \"--tts-cjk-segments-preset\",\n";
    std::cout << "    \"--tts-cjk-auto-preset\",\n";
    std::cout << "    \"--tts\",\n";
    std::cout << "    \"mit2_tts\",\n";
    std::cout << "    \"--preflight\",\n";
    std::cout << "    \"--clone-preflight\",\n";
    std::cout << "    \"--clone-preprocess\",\n";
    std::cout << "    \"--clone-readiness\",\n";
    std::cout << "    \"--clone-extract-mel\",\n";
    std::cout << "    \"--clone-extract-fbank\",\n";
    std::cout << "    \"--clone-prepare-features\",\n";
    std::cout << "    \"--clone-feature-readiness\",\n";
    std::cout << "    \"--clone-encoder-model-readiness\",\n";
    std::cout << "    \"--clone-campplus-style-readiness\",\n";
    std::cout << "    \"--clone-campplus-style-from-features\",\n";
    std::cout << "    \"--clone-campplus-head-golden\",\n";
    std::cout << "    \"--clone-w2v-feature-project\",\n";
    std::cout << "    \"--clone-w2v-encoder\",\n";
    std::cout << "    \"--clone-w2v-extract-features\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-norm\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-intermediate\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-activate\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-output\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer0-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer0-final-norm\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-norm\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-intermediate\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-activate\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-output\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer1-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer1-final-norm\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-norm\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-intermediate\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-activate\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-output\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer2-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-norm\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-intermediate\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-activate\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-output\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer3-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer3-final-norm\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-norm\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-intermediate\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-activate\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-output\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer4-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer5-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer5-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer5-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer6-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer6-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer6-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer7-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer7-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer7-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer8-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer8-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer8-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer9-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer9-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer9-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer10-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer10-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer10-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer11-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer11-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer11-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer12-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer12-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer12-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer13-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer13-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer13-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer14-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer15-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer14-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention\",\n";
    std::cout << "    \"--clone-w2v-layer14-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer17-final-norm\",\n";
    std::cout << "    \"--clone-w2v-normalize\",\n";
    std::cout << "    \"--clone-semantic-quantize\",\n";
    std::cout << "    \"--clone-semantic-prompt-from-spk-cond\",\n";
    std::cout << "    \"--clone-s2mel-prompt-from-sref\",\n";
    std::cout << "    \"--clone-encoder-readiness\",\n";
    std::cout << "    \"--clone-write-voice-bundle\",\n";
    std::cout << "    \"--clone-write-voice-bundle-from-features\",\n";
    std::cout << "    \"--clone\",\n";
    std::cout << "    \"--tts-cjk-text-readiness\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"start_sh_replacement_audit\": {\n";
    std::cout << "    \"can_replace_start_sh_full_clone_tts\": false,\n";
    std::cout << "    \"can_replace_start_sh_cached_voice_cjk\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "    \"covered_surfaces\": [\n";
    std::cout << "      \"native model bundle + native voice bundle + focused CJK/limited ASCII text to PCM16 WAV\",\n";
    std::cout << "      \"native bundle contract validation before synthesis\",\n";
    std::cout << "      \"native per-request text readiness and preset segmentation planning\"\n";
    std::cout << "    ],\n";
    std::cout << "    \"missing_surfaces\": [\n";
    std::cout << "      {\n";
    std::cout << "        \"surface\": \"cached_voice_general_text_to_wav\",\n";
    std::cout << "        \"current_boundary\": \"python_owned\",\n";
    std::cout << "        \"required_work\": \"full native TextNormalizer/SentencePiece and general text segmentation\"\n";
    std::cout << "      },\n";
    std::cout << "      {\n";
    std::cout << "        \"surface\": \"native_clone_audio_to_voice_bundle\",\n";
    std::cout << "        \"current_boundary\": \"python_owned\",\n";
    std::cout << "        \"required_work\": \"native semantic/acoustic speech encoders that feed the native mel extractor and voice bundle writer\"\n";
    std::cout << "      },\n";
    std::cout << "      {\n";
    std::cout << "        \"surface\": \"production_length_resource_planning\",\n";
    std::cout << "        \"current_boundary\": \"incomplete\",\n";
    std::cout << "        \"required_work\": \"full-length acoustic validation, memory planning, and production error handling\"\n";
    std::cout << "      }\n";
    std::cout << "    ],\n";
    std::cout << "    \"recommended_preflight_commands\": [\n";
    std::cout << "      \"--preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT\",\n";
    std::cout << "      \"--clone-preflight AUDIO_WAV\",\n";
    std::cout << "      \"--clone-preprocess AUDIO_WAV OUTPUT_F32\",\n";
    std::cout << "      \"--clone-readiness PREPROCESS_MANIFEST\",\n";
    std::cout << "      \"--clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32\",\n";
    std::cout << "      \"--clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32\",\n";
    std::cout << "      \"--clone-prepare-features AUDIO_WAV OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-feature-readiness FEATURE_MANIFEST\",\n";
    std::cout << "      \"--clone-encoder-model-readiness MODEL_BUNDLE_DIR\",\n";
    std::cout << "      \"--clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST S2MEL_STYLE_F32\",\n";
    std::cout << "      \"--clone-campplus-style-from-features MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32\",\n";
    std::cout << "      \"--clone-campplus-head-golden MODEL_BUNDLE_DIR FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR\",\n";
    std::cout << "      \"--clone-w2v-feature-project MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32 W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-qkv MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32 W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32 W2V_TOKENS OUTPUT_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-activate W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32 W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-qkv MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention W2V_LAYER1_Q_F32 W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER1_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention-residual W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER1_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-activate W2V_LAYER2_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-residual W2V_LAYER1_F32 W2V_LAYER2_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-qkv MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention W2V_LAYER2_Q_F32 W2V_LAYER2_K_F32 W2V_LAYER2_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER2_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention-residual W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER2_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-residual MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_LAYER2_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER2_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-activate W2V_LAYER3_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-residual W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_LAYER3_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-qkv MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention W2V_LAYER3_Q_F32 W2V_LAYER3_K_F32 W2V_LAYER3_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER3_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention-residual W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-activate W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32 W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-qkv MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32 W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER4_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention-residual W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER4_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-residual MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_LAYER4_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER4_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER4_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-qkv MODEL_BUNDLE_DIR W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention W2V_LAYER5_Q_F32 W2V_LAYER5_K_F32 W2V_LAYER5_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER5_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention-project MODEL_BUNDLE_DIR W2V_LAYER5_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention-residual W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_LAYER5_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-glu MODEL_BUNDLE_DIR W2V_LAYER5_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER5_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-residual MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_LAYER5_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER5_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER5_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-qkv MODEL_BUNDLE_DIR W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer6-attention W2V_LAYER6_Q_F32 W2V_LAYER6_K_F32 W2V_LAYER6_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER6_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-attention-project MODEL_BUNDLE_DIR W2V_LAYER6_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-attention-residual W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_LAYER6_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-glu MODEL_BUNDLE_DIR W2V_LAYER8_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER8_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-residual MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_LAYER8_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER8_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER8_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-qkv MODEL_BUNDLE_DIR W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention W2V_LAYER9_Q_F32 W2V_LAYER9_K_F32 W2V_LAYER9_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER9_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention-project MODEL_BUNDLE_DIR W2V_LAYER9_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention-residual W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_LAYER9_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-glu MODEL_BUNDLE_DIR W2V_LAYER9_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER9_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-residual MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_LAYER9_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER9_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER9_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-qkv MODEL_BUNDLE_DIR W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention W2V_LAYER10_Q_F32 W2V_LAYER10_K_F32 W2V_LAYER10_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER10_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention-project MODEL_BUNDLE_DIR W2V_LAYER10_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention-residual W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_LAYER10_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-glu MODEL_BUNDLE_DIR W2V_LAYER10_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER10_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-residual MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_LAYER10_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention W2V_LAYER13_Q_F32 W2V_LAYER13_K_F32 W2V_LAYER13_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER13_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention-project MODEL_BUNDLE_DIR W2V_LAYER13_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention-residual W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_LAYER13_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-glu MODEL_BUNDLE_DIR W2V_LAYER13_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER13_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-residual MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_LAYER13_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER13_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER13_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32 W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32\",\n";
    std::cout << "      \"--clone-w2v-normalize MODEL_BUNDLE_DIR W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32\",\n";
    std::cout << "      \"--clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32 SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32\",\n";
    std::cout << "      \"--clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32\",\n";
    std::cout << "      \"--clone-encoder-readiness FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32\",\n";
    std::cout << "      \"--clone-write-voice-bundle PREPROCESS_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS MEL_F32 OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "      \"--clone-write-voice-bundle-from-features FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "      \"--readiness MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR\",\n";
    std::cout << "      \"--text-readiness MODEL_BUNDLE_DIR TEXT\"\n";
    std::cout << "    ]\n";
    std::cout << "  },\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << "\n";
    std::cout << "}\n";
    return ok;
}

[[maybe_unused]] bool inspect_tts_product_preflight(const std::string& bundle_dir,
                                                    const std::string& voice_bundle_dir,
                                                    const std::string& text) {
    const bool model_ok = inspect_model_bundle_contract(bundle_dir);
    const bool voice_ok = inspect_voice_bundle_contract(voice_bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    const uint32_t prompt_tokens = infer_voice_prompt_tokens(voice);
    const std::string tokenizer_dir = bundle_dir + "/tokenizer";
    const std::vector<TtsCjkPreset> presets{
        resolve_tts_cjk_preset("smoke"),
        resolve_tts_cjk_preset("short"),
        resolve_tts_cjk_preset("standard"),
    };

    bool text_ok = false;
    std::string text_error;
    CjkTokenizedText tokenized;
    try {
        const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
        tokenized = tokenize_cjk_text(piece_to_id, text);
        text_ok = true;
    } catch (const std::exception& e) {
        text_error = e.what();
    }

    const auto memory = process_memory_info();
    const bool ok = model_ok && voice_ok && prompt_tokens > 0 && text_ok;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_product_preflight\",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"native_text_surface\": \"focused_cjk_limited_ascii\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"model_contract_ok\": " << (model_ok ? "true" : "false") << ",\n";
    std::cout << "  \"voice_contract_ok\": " << (voice_ok ? "true" : "false") << ",\n";
    std::cout << "  \"voice_prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"ready_native_cjk_text\": " << (text_ok ? "true" : "false") << ",\n";
    std::cout << "  \"ready_cached_voice_tts_cjk_text\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"ready_to_synthesize\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"ready_cached_voice_tts_general_text\": false,\n";
    std::cout << "  \"ready_native_voice_clone\": false,\n";
    if (text_ok) {
        std::cout << "  \"token_count\": " << tokenized.ids.size() << ",\n";
        std::cout << "  \"tokens\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "  \"token_ids\": ";
        print_json_u32_array(tokenized.ids);
        std::cout << ",\n";
        std::cout << "  \"presets\": [\n";
        for (size_t i = 0; i < presets.size(); ++i) {
            const auto segments = split_cjk_tokenized_text(tokenized, presets[i].max_text_tokens_per_segment);
            constexpr uint32_t frontend_cond_tokens = 34;
            std::vector<HotScratchPlan> segment_plans;
            segment_plans.reserve(segments.size());
            HotScratchPlan aggregate_plan;
            uint32_t max_segment_text_tokens = 0;
            for (const auto& segment : segments) {
                const uint32_t segment_text_tokens = static_cast<uint32_t>(segment.ids.size());
                const uint32_t segment_max_codes = segment_text_tokens * kMaxCodesPerTextToken;
                const uint32_t max_prefix_tokens = frontend_cond_tokens + segment_text_tokens + 2;
                auto plan = compute_hot_scratch_plan(max_prefix_tokens,
                                                     segment_max_codes,
                                                     prompt_tokens,
                                                     "preflight",
                                                     frontend_cond_tokens,
                                                     segment_text_tokens);
                if (segment_plans.empty() ||
                    plan.planned_scratch_capacity_bytes > aggregate_plan.planned_scratch_capacity_bytes) {
                    aggregate_plan = plan;
                }
                max_segment_text_tokens = std::max(max_segment_text_tokens, segment_text_tokens);
                segment_plans.push_back(std::move(plan));
            }
            std::cout << "    {\n";
            std::cout << "      \"name\": \"" << json_escape(presets[i].name) << "\",\n";
            std::cout << "      \"max_codes_per_text_token\": " << kMaxCodesPerTextToken << ",\n";
            std::cout << "      \"steps\": " << presets[i].steps << ",\n";
            std::cout << "      \"max_text_tokens_per_segment\": " << presets[i].max_text_tokens_per_segment << ",\n";
            std::cout << "      \"interval_silence_ms\": " << presets[i].interval_silence_ms << ",\n";
            std::cout << "      \"segment_count\": " << segments.size() << ",\n";
            std::cout << "      \"auto_segmented\": " << (segments.size() > 1 ? "true" : "false") << ",\n";
            std::cout << "      \"resource_plan\": {\n";
            std::cout << "        \"planner\": \"hot_scratch_phase_reuse\",\n";
            std::cout << "        \"segment_count\": " << segments.size() << ",\n";
            std::cout << "        \"max_segment_text_tokens\": " << max_segment_text_tokens << ",\n";
            std::cout << "        \"aggregate_plan_source\": \"max_segment_planned_scratch_capacity\",\n";
            std::cout << "        \"aggregate_segment_plan\": ";
            print_hot_scratch_plan_object_json(aggregate_plan, "        ");
            std::cout << ",\n";
            std::cout << "        \"segment_plans\": [\n";
            for (size_t segment_index = 0; segment_index < segment_plans.size(); ++segment_index) {
                std::cout << "          {\n";
                std::cout << "            \"index\": " << segment_index << ",\n";
                std::cout << "            \"text_tokens\": " << segment_plans[segment_index].text_tokens << ",\n";
                std::cout << "            \"plan\": ";
                print_hot_scratch_plan_object_json(segment_plans[segment_index], "            ");
                std::cout << "\n";
                std::cout << "          }" << (segment_index + 1 == segment_plans.size() ? "\n" : ",\n");
            }
            std::cout << "        ]\n";
            std::cout << "      }\n";
            std::cout << "    }" << (i + 1 == presets.size() ? "\n" : ",\n");
        }
        std::cout << "  ],\n";
    } else {
        std::cout << "  \"text_error\": \"" << json_escape(text_error) << "\",\n";
        std::cout << "  \"python_boundary\": \"full TextNormalizer/SentencePiece for general text\",\n";
    }
    std::cout << "  \"start_sh_replacement_audit\": {\n";
    std::cout << "    \"can_replace_start_sh_full_clone_tts\": false,\n";
    std::cout << "    \"can_replace_start_sh_cached_voice_cjk\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "    \"covered_surfaces\": [\n";
    std::cout << "      \"native model bundle + native voice bundle + focused CJK/limited ASCII text to PCM16 WAV\",\n";
    std::cout << "      \"native bundle contract validation before synthesis\",\n";
    std::cout << "      \"native per-request text readiness and preset segmentation planning\"\n";
    std::cout << "    ],\n";
    std::cout << "    \"missing_surfaces\": [\n";
    std::cout << "      {\n";
    std::cout << "        \"surface\": \"cached_voice_general_text_to_wav\",\n";
    std::cout << "        \"current_boundary\": \"python_owned\",\n";
    std::cout << "        \"required_work\": \"full native TextNormalizer/SentencePiece and general text segmentation\"\n";
    std::cout << "      },\n";
    std::cout << "      {\n";
    std::cout << "        \"surface\": \"native_clone_audio_to_voice_bundle\",\n";
    std::cout << "        \"current_boundary\": \"python_owned\",\n";
    std::cout << "        \"required_work\": \"native semantic/acoustic speech encoders that feed the native mel extractor and voice bundle writer\"\n";
    std::cout << "      },\n";
    std::cout << "      {\n";
    std::cout << "        \"surface\": \"production_length_resource_planning\",\n";
    std::cout << "        \"current_boundary\": \"incomplete\",\n";
    std::cout << "        \"required_work\": \"full-length acoustic validation, memory planning, and production error handling\"\n";
    std::cout << "      }\n";
    std::cout << "    ],\n";
    std::cout << "    \"recommended_preflight_commands\": [\n";
    std::cout << "      \"--preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT\",\n";
    std::cout << "      \"--clone-preflight AUDIO_WAV\",\n";
    std::cout << "      \"--clone-preprocess AUDIO_WAV OUTPUT_F32\",\n";
    std::cout << "      \"--clone-readiness PREPROCESS_MANIFEST\",\n";
    std::cout << "      \"--clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32\",\n";
    std::cout << "      \"--clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32\",\n";
    std::cout << "      \"--clone-prepare-features AUDIO_WAV OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-feature-readiness FEATURE_MANIFEST\",\n";
    std::cout << "      \"--clone-encoder-model-readiness MODEL_BUNDLE_DIR\",\n";
    std::cout << "      \"--clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST S2MEL_STYLE_F32\",\n";
    std::cout << "      \"--clone-campplus-style-from-features MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32\",\n";
    std::cout << "      \"--clone-campplus-head-golden MODEL_BUNDLE_DIR FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR\",\n";
    std::cout << "      \"--clone-w2v-feature-project MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32 W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-qkv MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32 W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32 W2V_TOKENS OUTPUT_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-activate W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32 W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-qkv MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention W2V_LAYER1_Q_F32 W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER1_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention-residual W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER1_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-activate W2V_LAYER2_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn1-residual W2V_LAYER1_F32 W2V_LAYER2_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-qkv MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention W2V_LAYER2_Q_F32 W2V_LAYER2_K_F32 W2V_LAYER2_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER2_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention-residual W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER2_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-conv-residual MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_LAYER2_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer2-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER2_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-activate W2V_LAYER3_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn1-residual W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_LAYER3_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-qkv MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention W2V_LAYER3_Q_F32 W2V_LAYER3_K_F32 W2V_LAYER3_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER3_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention-residual W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-activate W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_ACTIVATED_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32 W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-qkv MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32 W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER4_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention-residual W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER4_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-conv-residual MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_LAYER4_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer4-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER4_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER4_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-qkv MODEL_BUNDLE_DIR W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention W2V_LAYER5_Q_F32 W2V_LAYER5_K_F32 W2V_LAYER5_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER5_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention-project MODEL_BUNDLE_DIR W2V_LAYER5_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention-residual W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_LAYER5_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-attention-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-glu MODEL_BUNDLE_DIR W2V_LAYER5_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER5_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-conv-residual MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_LAYER5_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer5-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER5_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER5_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-qkv MODEL_BUNDLE_DIR W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer6-attention W2V_LAYER6_Q_F32 W2V_LAYER6_K_F32 W2V_LAYER6_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER6_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-attention-project MODEL_BUNDLE_DIR W2V_LAYER6_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-attention-residual W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_LAYER6_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-glu MODEL_BUNDLE_DIR W2V_LAYER8_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER8_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-conv-residual MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_LAYER8_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER8_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER8_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-qkv MODEL_BUNDLE_DIR W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention W2V_LAYER9_Q_F32 W2V_LAYER9_K_F32 W2V_LAYER9_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER9_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention-project MODEL_BUNDLE_DIR W2V_LAYER9_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention-residual W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_LAYER9_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-attention-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-glu MODEL_BUNDLE_DIR W2V_LAYER9_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER9_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-conv-residual MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_LAYER9_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer9-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER9_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER9_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-qkv MODEL_BUNDLE_DIR W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention W2V_LAYER10_Q_F32 W2V_LAYER10_K_F32 W2V_LAYER10_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER10_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention-project MODEL_BUNDLE_DIR W2V_LAYER10_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention-residual W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_LAYER10_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-attention-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-glu MODEL_BUNDLE_DIR W2V_LAYER10_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER10_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-conv-residual MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_LAYER10_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer10-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER10_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER10_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-qkv MODEL_BUNDLE_DIR W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer11-attention W2V_LAYER11_Q_F32 W2V_LAYER11_K_F32 W2V_LAYER11_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER11_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-attention-project MODEL_BUNDLE_DIR W2V_LAYER11_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-attention-residual W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_LAYER11_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-attention-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-conv-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-conv-glu MODEL_BUNDLE_DIR W2V_LAYER11_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER11_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-conv-residual MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_LAYER11_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer11-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER11_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER11_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-qkv MODEL_BUNDLE_DIR W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer12-attention W2V_LAYER12_Q_F32 W2V_LAYER12_K_F32 W2V_LAYER12_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER12_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-attention-project MODEL_BUNDLE_DIR W2V_LAYER12_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-attention-residual W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_LAYER12_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-attention-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-conv-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-conv-glu MODEL_BUNDLE_DIR W2V_LAYER12_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER12_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-conv-residual MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_LAYER12_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer12-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER12_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER12_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-qkv MODEL_BUNDLE_DIR W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention W2V_LAYER13_Q_F32 W2V_LAYER13_K_F32 W2V_LAYER13_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER13_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention-project MODEL_BUNDLE_DIR W2V_LAYER13_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention-residual W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_LAYER13_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-attention-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-glu MODEL_BUNDLE_DIR W2V_LAYER13_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER13_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-conv-residual MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_LAYER13_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer13-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER13_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER13_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32 W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32\",\n";
    std::cout << "      \"--clone-w2v-normalize MODEL_BUNDLE_DIR W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32\",\n";
    std::cout << "      \"--clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32 SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32\",\n";
    std::cout << "      \"--clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32\",\n";
    std::cout << "      \"--clone-encoder-readiness FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32\",\n";
    std::cout << "      \"--clone-write-voice-bundle PREPROCESS_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS MEL_F32 OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "      \"--clone-write-voice-bundle-from-features FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "      \"--readiness MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR\",\n";
    std::cout << "      \"--text-readiness MODEL_BUNDLE_DIR TEXT\"\n";
    std::cout << "    ]\n";
    std::cout << "  },\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << "\n";
    std::cout << "}\n";
    return ok;
}

[[maybe_unused]] bool inspect_tts_clone_audio_preflight(const std::string& audio_wav) {
    try {
        const auto wav = read_wav_pcm16_mono_bytes(audio_wav);
        const auto quality = analyze_clone_audio_quality(wav);
        const bool quality_ok = quality.issues.empty();
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_audio_preflight\",\n";
        std::cout << "  \"ok\": " << (quality_ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"audio_format\": \"pcm16_mono_wav\",\n";
        std::cout << "  \"audio_sha256\": \"" << file_sha256_hex(audio_wav) << "\",\n";
        std::cout << "  \"sample_rate\": " << wav.sample_rate << ",\n";
        std::cout << "  \"samples\": " << quality.samples << ",\n";
        std::cout << "  \"duration_seconds\": " << quality.duration_seconds << ",\n";
        std::cout << "  \"frame_bytes\": " << wav.frames.size() << ",\n";
        std::cout << "  \"audio_peak_abs_i16\": " << quality.peak_abs << ",\n";
        std::cout << "  \"audio_peak_normalized\": " << quality.peak << ",\n";
        std::cout << "  \"audio_rms_normalized\": " << quality.rms << ",\n";
        std::cout << "  \"audio_mean_normalized\": " << quality.mean << ",\n";
        std::cout << "  \"nonzero_sample_ratio\": " << quality.nonzero_ratio << ",\n";
        std::cout << "  \"near_silence_sample_ratio\": " << quality.near_silence_ratio << ",\n";
        std::cout << "  \"clipping_sample_ratio\": " << quality.clipping_ratio << ",\n";
        std::cout << "  \"recommended_resample_rate\": 16000,\n";
        std::cout << "  \"ready_native_clone_audio_quality\": " << (quality_ok ? "true" : "false") << ",\n";
        std::cout << "  \"quality_issues\": ";
        print_json_string_array(quality.issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_clone_audio_input\": " << (quality_ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"python_boundary_after_input\": \"clone-time mel extraction and speech encoders for voice tensor creation\",\n";
        std::cout << "  \"required_next_native_work\": [\n";
        std::cout << "    \"native audio resampling and normalization\",\n";
        std::cout << "    \"native semantic and acoustic clone-time feature extraction\",\n";
        std::cout << "    \"native speech encoders for voice profile tensors\",\n";
        std::cout << "    \"native voice bundle writing from encoder tensors\"\n";
        std::cout << "  ]\n";
        std::cout << "}\n";
        return quality_ok;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_audio_preflight\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"ready_native_clone_audio_input\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\",\n";
        std::cout << "  \"expected_audio_format\": \"pcm16_mono_wav\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_audio_preprocess(const std::string& audio_wav,
                                                     const std::string& output_f32) {
    try {
        constexpr uint32_t target_rate = 16000;
        const auto wav = read_wav_pcm16_mono_bytes(audio_wav);
        const auto quality = analyze_clone_audio_quality(wav);
        const bool quality_ok = quality.issues.empty();
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_audio_preprocess\",\n";
        std::cout << "  \"ok\": " << (quality_ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"output_f32\": \"" << json_escape(output_f32) << "\",\n";
        std::cout << "  \"audio_format\": \"pcm16_mono_wav\",\n";
        std::cout << "  \"audio_sha256\": \"" << file_sha256_hex(audio_wav) << "\",\n";
        std::cout << "  \"source_sample_rate\": " << wav.sample_rate << ",\n";
        std::cout << "  \"target_sample_rate\": " << target_rate << ",\n";
        std::cout << "  \"source_samples\": " << quality.samples << ",\n";
        std::cout << "  \"source_duration_seconds\": " << quality.duration_seconds << ",\n";
        std::cout << "  \"audio_peak_normalized\": " << quality.peak << ",\n";
        std::cout << "  \"audio_rms_normalized\": " << quality.rms << ",\n";
        std::cout << "  \"near_silence_sample_ratio\": " << quality.near_silence_ratio << ",\n";
        std::cout << "  \"clipping_sample_ratio\": " << quality.clipping_ratio << ",\n";
        std::cout << "  \"ready_native_clone_audio_quality\": " << (quality_ok ? "true" : "false") << ",\n";
        std::cout << "  \"quality_issues\": ";
        print_json_string_array(quality.issues);
        std::cout << ",\n";
        const std::string manifest_path = output_f32 + ".manifest.json";
        if (quality_ok) {
            const auto normalized = pcm16_mono_wav_to_f32(wav);
            const auto resampled = resample_linear_f32(normalized, wav.sample_rate, target_rate);
            write_raw_f32(output_f32, resampled);
            const std::string output_sha = file_sha256_hex(output_f32);
            std::ostringstream manifest;
            manifest << "{\n";
            manifest << "  \"format\": \"mit2-clone-audio-preprocess\",\n";
            manifest << "  \"version\": 1,\n";
            manifest << "  \"source_audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
            manifest << "  \"source_audio_sha256\": \"" << file_sha256_hex(audio_wav) << "\",\n";
            manifest << "  \"output_f32\": \"" << json_escape(output_f32) << "\",\n";
            manifest << "  \"output_sha256\": \"" << output_sha << "\",\n";
            manifest << "  \"audio_format\": \"f32_mono_raw\",\n";
            manifest << "  \"source_sample_rate\": " << wav.sample_rate << ",\n";
            manifest << "  \"target_sample_rate\": " << target_rate << ",\n";
            manifest << "  \"source_samples\": " << quality.samples << ",\n";
            manifest << "  \"preprocessed_samples\": " << resampled.size() << ",\n";
            manifest << "  \"preprocessed_duration_seconds\": "
                     << (static_cast<double>(resampled.size()) / static_cast<double>(target_rate)) << ",\n";
            manifest << "  \"audio_peak_normalized\": " << quality.peak << ",\n";
            manifest << "  \"audio_rms_normalized\": " << quality.rms << ",\n";
            manifest << "  \"near_silence_sample_ratio\": " << quality.near_silence_ratio << ",\n";
            manifest << "  \"clipping_sample_ratio\": " << quality.clipping_ratio << ",\n";
            manifest << "  \"ready_native_clone_audio_preprocess\": true,\n";
            manifest << "  \"ready_native_voice_clone\": false,\n";
            manifest << "  \"next_native_boundary\": \"clone-time mel extraction and semantic/acoustic speech encoders for voice tensor creation\"\n";
            manifest << "}\n";
            write_text_file(manifest_path, manifest.str());
            std::cout << "  \"preprocessed_audio_format\": \"f32_mono_raw\",\n";
            std::cout << "  \"preprocessed_sample_rate\": " << target_rate << ",\n";
            std::cout << "  \"preprocessed_samples\": " << resampled.size() << ",\n";
            std::cout << "  \"preprocessed_duration_seconds\": "
                      << (static_cast<double>(resampled.size()) / static_cast<double>(target_rate)) << ",\n";
            std::cout << "  \"preprocessed_sha256\": \"" << output_sha << "\",\n";
            std::cout << "  \"preprocess_manifest\": \"" << json_escape(manifest_path) << "\",\n";
            std::cout << "  \"preprocess_manifest_sha256\": \"" << file_sha256_hex(manifest_path) << "\",\n";
            std::cout << "  \"ready_native_clone_audio_preprocess\": true,\n";
        } else {
            std::cout << "  \"preprocessed_audio_format\": \"f32_mono_raw\",\n";
            std::cout << "  \"preprocessed_sample_rate\": " << target_rate << ",\n";
            std::cout << "  \"preprocessed_samples\": 0,\n";
            std::cout << "  \"preprocess_manifest\": \"" << json_escape(manifest_path) << "\",\n";
            std::cout << "  \"ready_native_clone_audio_preprocess\": false,\n";
        }
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"python_boundary_after_preprocess\": \"clone-time mel extraction and semantic/acoustic speech encoders for voice tensor creation\"\n";
        std::cout << "}\n";
        return quality_ok;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_audio_preprocess\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"output_f32\": \"" << json_escape(output_f32) << "\",\n";
        std::cout << "  \"ready_native_clone_audio_preprocess\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\",\n";
        std::cout << "  \"expected_audio_format\": \"pcm16_mono_wav\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool inspect_tts_clone_readiness(const std::string& preprocess_manifest) {
    std::vector<std::string> issues;
    ClonePreprocessManifest manifest;
    bool parsed = false;
    try {
        manifest = parse_clone_preprocess_manifest(preprocess_manifest);
        parsed = true;
        if (manifest.format != "mit2-clone-audio-preprocess") {
            issues.push_back("unexpected_manifest_format");
        }
        if (manifest.version != 1) {
            issues.push_back("unsupported_manifest_version");
        }
        if (manifest.audio_format != "f32_mono_raw") {
            issues.push_back("unsupported_preprocessed_audio_format");
        }
        if (manifest.target_sample_rate != 16000) {
            issues.push_back("unexpected_preprocessed_sample_rate");
        }
        if (manifest.preprocessed_samples == 0) {
            issues.push_back("empty_preprocessed_audio");
        }
        if (!manifest.ready_native_clone_audio_preprocess) {
            issues.push_back("manifest_preprocess_not_ready");
        }
        if (manifest.ready_native_voice_clone) {
            issues.push_back("manifest_must_not_claim_voice_clone_ready");
        }
        if (manifest.output_f32.empty()) {
            issues.push_back("missing_output_f32_path");
        } else if (!std::filesystem::exists(manifest.output_f32)) {
            issues.push_back("preprocessed_output_f32_missing");
        } else {
            const uint64_t expected_bytes = manifest.preprocessed_samples * static_cast<uint64_t>(sizeof(float));
            const uint64_t actual_bytes = static_cast<uint64_t>(std::filesystem::file_size(manifest.output_f32));
            if (actual_bytes != expected_bytes) {
                issues.push_back("preprocessed_output_f32_size_mismatch");
            }
            if (!manifest.output_sha256.empty()) {
                const std::string actual_sha = file_sha256_hex(manifest.output_f32);
                if (actual_sha != manifest.output_sha256) {
                    issues.push_back("preprocessed_output_f32_sha256_mismatch");
                }
            } else {
                issues.push_back("missing_output_sha256");
            }
        }
        if (!manifest.source_audio_wav.empty() && std::filesystem::exists(manifest.source_audio_wav) &&
            !manifest.source_audio_sha256.empty()) {
            const std::string actual_source_sha = file_sha256_hex(manifest.source_audio_wav);
            if (actual_source_sha != manifest.source_audio_sha256) {
                issues.push_back("source_audio_sha256_mismatch");
            }
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("manifest_parse_error: ") + e.what());
    }

    const bool preprocess_ready = parsed && issues.empty();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_clone_readiness\",\n";
    std::cout << "  \"ok\": " << (preprocess_ready ? "true" : "false") << ",\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
    std::cout << "  \"manifest_parsed\": " << (parsed ? "true" : "false") << ",\n";
    if (parsed) {
        std::cout << "  \"manifest_format\": \"" << json_escape(manifest.format) << "\",\n";
        std::cout << "  \"manifest_version\": " << manifest.version << ",\n";
        std::cout << "  \"preprocessed_audio_format\": \"" << json_escape(manifest.audio_format) << "\",\n";
        std::cout << "  \"preprocessed_output_f32\": \"" << json_escape(manifest.output_f32) << "\",\n";
        std::cout << "  \"preprocessed_sha256\": \"" << json_escape(manifest.output_sha256) << "\",\n";
        std::cout << "  \"source_sample_rate\": " << manifest.source_sample_rate << ",\n";
        std::cout << "  \"preprocessed_sample_rate\": " << manifest.target_sample_rate << ",\n";
        std::cout << "  \"source_samples\": " << manifest.source_samples << ",\n";
        std::cout << "  \"preprocessed_samples\": " << manifest.preprocessed_samples << ",\n";
    }
    std::cout << "  \"clone_readiness_issues\": ";
    print_json_string_array(issues);
    std::cout << ",\n";
    std::cout << "  \"ready_native_clone_audio_preprocess\": " << (preprocess_ready ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_voice_bundle_creation\": false,\n";
    std::cout << "  \"ready_native_voice_clone\": false,\n";
    std::cout << "  \"required_voice_bundle_tensors\": [\n";
    std::cout << "    {\"name\": \"spk_cond_emb\", \"dtype\": \"f32\", \"shape\": \"[1,tokens>0,1024]\", \"source\": \"native semantic clone encoder\"},\n";
    std::cout << "    {\"name\": \"s2mel_style\", \"dtype\": \"f32\", \"shape\": \"[1,192]\", \"source\": \"native acoustic style encoder\"},\n";
    std::cout << "    {\"name\": \"s2mel_prompt\", \"dtype\": \"f32\", \"shape\": \"[1,frames>0,512]\", \"source\": \"native S2Mel length regulator from MaskGCT S_ref\"},\n";
    std::cout << "    {\"name\": \"mel\", \"dtype\": \"f32\", \"shape\": \"[1,80,frames>0]\", \"source\": \"native mel extraction\"}\n";
    std::cout << "  ],\n";
    std::cout << "  \"required_next_native_work\": [\n";
    std::cout << "    \"native mel extraction from 16 kHz preprocessed audio through --clone-extract-mel\",\n";
    std::cout << "    \"native W2V-BERT/semantic feature extraction and MaskGCT quantize for spk_cond_emb\",\n";
    std::cout << "    \"native CAMPPlus/acoustic style path for s2mel_style\",\n";
    std::cout << "    \"native S2Mel prompt condition generation\",\n";
    std::cout << "    \"native MIT2 voice bundle writer consumes spk_cond_emb, s2mel_style, s2mel_prompt, and mel\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"python_boundary_after_preprocess\": \"clone-time mel extraction and semantic/acoustic speech encoders for voice tensor creation\"\n";
    std::cout << "}\n";
    return preprocess_ready;
}

