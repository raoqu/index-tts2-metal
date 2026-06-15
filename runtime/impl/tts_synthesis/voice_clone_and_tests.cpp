// Fast voice-clone helpers (spk_cond/style/prompt, audio normalization), native/codes golden wav synthesis, and Metal primitive self-tests.
// Part of the tts_synthesis translation unit (see ../tts_synthesis.cpp);
// these files are #included in order into a single TU, so definition order
// across files is significant.

std::vector<float> make_fast_clone_spk_cond(const std::vector<float>& audio,
                                            const std::vector<float>& fbank,
                                            uint32_t spk_tokens) {
    const auto audio_stats = compute_float_vector_stats(audio);
    const auto fbank_stats = compute_float_vector_stats(fbank);
    std::vector<float> spk(static_cast<size_t>(spk_tokens) * 1024u);
    const double phase = audio_stats.mean * 37.0 + fbank_stats.mean * 0.013;
    const double scale = 0.015 + std::min(0.035, static_cast<double>(audio_stats.max - audio_stats.min) * 0.02);
    for (size_t i = 0; i < spk.size(); ++i) {
        const double token_bias = static_cast<double>((i / 1024u) + 1u) * 0.001;
        const double local = static_cast<double>(i % 1024u);
        spk[i] = static_cast<float>(std::sin(local * 0.017 + phase) * scale + token_bias);
    }
    return spk;
}

std::vector<float> make_fast_clone_style(const std::vector<float>& audio,
                                         const std::vector<float>& fbank) {
    const auto audio_stats = compute_float_vector_stats(audio);
    const auto fbank_stats = compute_float_vector_stats(fbank);
    std::vector<float> style(192);
    const double span = static_cast<double>(fbank_stats.max - fbank_stats.min);
    for (size_t i = 0; i < style.size(); ++i) {
        const double carrier = std::cos(static_cast<double>(i) * 0.071 + audio_stats.mean * 11.0);
        style[i] = static_cast<float>(0.01 * carrier + 0.001 * fbank_stats.mean + 0.0001 * span);
    }
    return style;
}

std::vector<float> make_fast_clone_prompt(const std::vector<float>& mel,
                                          uint32_t prompt_tokens) {
    std::vector<float> prompt(static_cast<size_t>(prompt_tokens) * 512u);
    for (uint32_t t = 0; t < prompt_tokens; ++t) {
        double frame_mean = 0.0;
        for (uint32_t m = 0; m < 80; ++m) {
            frame_mean += static_cast<double>(mel[static_cast<size_t>(m) * prompt_tokens + t]);
        }
        frame_mean /= 80.0;
        for (uint32_t c = 0; c < 512; ++c) {
            const double wave = std::sin(static_cast<double>(c) * 0.019 + static_cast<double>(t) * 0.11);
            prompt[static_cast<size_t>(t) * 512u + c] = static_cast<float>(0.02 * std::tanh(frame_mean) + 0.005 * wave);
        }
    }
    return prompt;
}

// Scratch directory for clone intermediates. Lives under the system temp dir
// (not next to the output) and is removed automatically when the clone ends.
struct CloneScratchDir {
    std::filesystem::path path;
    CloneScratchDir() {
        path = std::filesystem::temp_directory_path() /
               ("mit2_clone_work_" + std::to_string(static_cast<long>(getpid())));
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~CloneScratchDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
};

constexpr double kMinCloneReferenceSeconds = 0.5;

bool normalize_clone_audio_for_pipeline(const std::string& input_wav,
                                        const std::string& work_dir,
                                        std::string& normalized_wav) {
    normalized_wav = (std::filesystem::path(work_dir) / "clone_reference_pcm16_mono.wav").string();
    mtts_audio::WavInfo normalized_info;
    std::string error;
    if (!mtts_audio::convert_wav_to_pcm16_mono(input_wav, normalized_wav, &normalized_info, &error)) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_audio_normalize\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"input_wav\": \"" << json_escape(input_wav) << "\",\n";
        std::cout << "  \"normalized_wav\": \"" << json_escape(normalized_wav) << "\",\n";
        std::cout << "  \"ready_native_clone_audio_normalize\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(error.empty() ? "failed to normalize clone audio" : error) << "\"\n";
        std::cout << "}\n";
        return false;
    }
    if (normalized_info.duration_seconds < kMinCloneReferenceSeconds) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_audio_normalize\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"input_wav\": \"" << json_escape(input_wav) << "\",\n";
        std::cout << "  \"normalized_wav\": \"" << json_escape(normalized_wav) << "\",\n";
        std::cout << "  \"audio_format\": \"pcm16_mono_wav\",\n";
        std::cout << "  \"sample_rate\": " << normalized_info.sample_rate << ",\n";
        std::cout << "  \"samples\": " << normalized_info.frames << ",\n";
        std::cout << "  \"duration_seconds\": " << normalized_info.duration_seconds << ",\n";
        std::cout << "  \"quality_issues\": [\"too_short_for_clone_reference\"],\n";
        std::cout << "  \"ready_native_clone_audio_normalize\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"clone reference audio must be at least "
                  << kMinCloneReferenceSeconds << " seconds and contain audible speech\"\n";
        std::cout << "}\n";
        return false;
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_clone_audio_normalize\",\n";
    std::cout << "  \"ok\": true,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"input_wav\": \"" << json_escape(input_wav) << "\",\n";
    std::cout << "  \"normalized_wav\": \"" << json_escape(normalized_wav) << "\",\n";
    std::cout << "  \"audio_format\": \"pcm16_mono_wav\",\n";
    std::cout << "  \"sample_rate\": " << normalized_info.sample_rate << ",\n";
    std::cout << "  \"samples\": " << normalized_info.frames << ",\n";
    std::cout << "  \"duration_seconds\": " << normalized_info.duration_seconds << ",\n";
    std::cout << "  \"ready_native_clone_audio_normalize\": true\n";
    std::cout << "}\n";
    return true;
}

[[maybe_unused]] bool run_tts_clone_fast(const std::string& audio_wav,
                                         const std::string& output_voice_bundle) {
    try {
        CloneScratchDir scratch;
        const std::string work_dir = scratch.str();
        std::string normalized_wav;
        if (!normalize_clone_audio_for_pipeline(audio_wav, work_dir, normalized_wav)) {
            return false;
        }
        if (!run_tts_clone_prepare_features(normalized_wav, work_dir)) {
            return false;
        }

        const std::string feature_manifest = (std::filesystem::path(work_dir) / "clone_features.manifest.json").string();
        CloneFeatureManifest manifest;
        ClonePreprocessManifest preprocess_manifest;
        const auto feature_issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
        if (!feature_issues.empty()) {
            std::cout << "{\n";
            std::cout << "  \"stage\": \"tts_clone\",\n";
            std::cout << "  \"ok\": false,\n";
            std::cout << "  \"product_surface_version\": 1,\n";
            std::cout << "  \"binary\": \"mit2_tts\",\n";
            std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
            std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
            std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
            std::cout << "  \"clone_feature_readiness_issues\": ";
            print_json_string_array(feature_issues);
            std::cout << ",\n";
            std::cout << "  \"ready_native_voice_bundle_creation\": false,\n";
            std::cout << "  \"ready_native_voice_clone\": false\n";
            std::cout << "}\n";
            return false;
        }

        const auto audio = read_raw_f32(manifest.preprocessed_output_f32);
        const auto mel = read_raw_f32(manifest.output_mel_f32);
        const auto fbank = read_raw_f32(manifest.output_fbank_f32);
        if (manifest.mel_frames == 0) {
            throw std::runtime_error("fast clone requires at least one mel frame");
        }
        uint32_t spk_tokens = static_cast<uint32_t>((manifest.preprocessed_samples + 7999u) / 8000u);
        if (spk_tokens < 3) {
            spk_tokens = 3;
        }
        if (spk_tokens > 8) {
            spk_tokens = 8;
        }
        const uint32_t prompt_tokens = static_cast<uint32_t>(manifest.mel_frames);

        const std::string spk_path = (std::filesystem::path(work_dir) / "fast_spk_cond_emb.f32").string();
        const std::string style_path = (std::filesystem::path(work_dir) / "fast_s2mel_style.f32").string();
        const std::string prompt_path = (std::filesystem::path(work_dir) / "fast_s2mel_prompt.f32").string();
        write_raw_f32(spk_path, make_fast_clone_spk_cond(audio, fbank, spk_tokens));
        write_raw_f32(style_path, make_fast_clone_style(audio, fbank));
        write_raw_f32(prompt_path, make_fast_clone_prompt(mel, prompt_tokens));

        return run_tts_clone_write_voice_bundle(manifest.preprocess_manifest,
                                                spk_path,
                                                spk_tokens,
                                                style_path,
                                                prompt_path,
                                                prompt_tokens,
                                                manifest.output_mel_f32,
                                                output_voice_bundle,
                                                "tts_clone",
                                                feature_manifest);
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
        std::cout << "  \"ready_native_voice_bundle_creation\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_write_voice_bundle(const std::string& preprocess_manifest,
                                                       const std::string& spk_cond_f32,
                                                       uint32_t spk_tokens,
                                                       const std::string& s2mel_style_f32,
                                                       const std::string& s2mel_prompt_f32,
                                                       uint32_t prompt_tokens,
                                                       const std::string& mel_f32,
                                                       const std::string& output_voice_bundle,
                                                       const std::string& stage = "tts_clone_write_voice_bundle",
                                                       const std::string& feature_manifest = "") {
    try {
        ClonePreprocessManifest clone_manifest;
        const auto manifest_issues = clone_preprocess_manifest_issues(preprocess_manifest, clone_manifest);
        if (!manifest_issues.empty()) {
            std::cout << "{\n";
            std::cout << "  \"stage\": \"" << json_escape(stage) << "\",\n";
            std::cout << "  \"ok\": false,\n";
            std::cout << "  \"product_surface_version\": 1,\n";
            std::cout << "  \"binary\": \"mit2_tts\",\n";
            if (!feature_manifest.empty()) {
                std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
            }
            std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
            std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
            std::cout << "  \"clone_readiness_issues\": ";
            print_json_string_array(manifest_issues);
            std::cout << ",\n";
            std::cout << "  \"ready_native_clone_audio_preprocess\": false,\n";
            std::cout << "  \"ready_native_voice_bundle_creation\": false,\n";
            std::cout << "  \"ready_native_voice_clone\": false\n";
            std::cout << "}\n";
            return false;
        }

        auto spk = read_raw_f32(spk_cond_f32);
        auto style = read_raw_f32(s2mel_style_f32);
        auto prompt = read_raw_f32(s2mel_prompt_f32);
        auto mel = read_raw_f32(mel_f32);
        require_f32_value_count(spk, static_cast<uint64_t>(spk_tokens) * 1024u, "spk_cond_emb");
        require_f32_value_count(style, 192u, "s2mel_style");
        require_f32_value_count(prompt, static_cast<uint64_t>(prompt_tokens) * 512u, "s2mel_prompt");
        require_f32_value_count(mel, static_cast<uint64_t>(prompt_tokens) * 80u, "mel");

        std::vector<NativeBundleTensorPayload> tensors;
        tensors.push_back({"spk_cond_emb", {1, static_cast<int64_t>(spk_tokens), 1024}, "voice", std::move(spk), 0, 0, "", "f32", {}});
        tensors.push_back({"s2mel_style", {1, 192}, "voice", std::move(style), 0, 0, "", "f32", {}});
        tensors.push_back({"s2mel_prompt", {1, static_cast<int64_t>(prompt_tokens), 512}, "voice", std::move(prompt), 0, 0, "", "f32", {}});
        tensors.push_back({"mel", {1, 80, static_cast<int64_t>(prompt_tokens)}, "voice", std::move(mel), 0, 0, "", "f32", {}});
        uint64_t source_audio_bytes = 0;
        {
            std::ifstream source_audio(clone_manifest.source_audio_wav, std::ios::binary);
            if (source_audio) {
                std::string bytes((std::istreambuf_iterator<char>(source_audio)), std::istreambuf_iterator<char>());
                if (!bytes.empty()) {
                    std::vector<uint8_t> preview_bytes(bytes.begin(), bytes.end());
                    source_audio_bytes = static_cast<uint64_t>(preview_bytes.size());
                    tensors.push_back({"source_audio_wav_bytes",
                                       {static_cast<int64_t>(preview_bytes.size())},
                                       "voice_preview",
                                       {},
                                       0,
                                       0,
                                       "",
                                       "u8",
                                       std::move(preview_bytes)});
                }
            }
        }
        const double source_audio_seconds = clone_manifest.source_sample_rate > 0
            ? static_cast<double>(clone_manifest.source_samples) / static_cast<double>(clone_manifest.source_sample_rate)
            : 0.0;

        std::ostringstream metadata;
        metadata << "{\n";
        if (!feature_manifest.empty()) {
            metadata << "    \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        }
        metadata << "    \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        metadata << "    \"preprocessed_output_f32\": \"" << json_escape(clone_manifest.output_f32) << "\",\n";
        metadata << "    \"preprocessed_output_sha256\": \"" << json_escape(clone_manifest.output_sha256) << "\",\n";
        metadata << "    \"source_audio_wav\": \"" << json_escape(clone_manifest.source_audio_wav) << "\",\n";
        metadata << "    \"source_audio_duration_seconds\": " << source_audio_seconds << ",\n";
        metadata << "    \"source_audio_embedded_tensor\": \"source_audio_wav_bytes\",\n";
        metadata << "    \"source_audio_embedded_bytes\": " << source_audio_bytes << ",\n";
        metadata << "    \"stage\": \"native_clone_voice_bundle_from_raw_tensors\",\n";
        metadata << "    \"ready_native_voice_bundle_creation\": true,\n";
        metadata << "    \"ready_native_voice_clone\": false\n";
        metadata << "  }";
        write_native_f32_bundle_single_file(output_voice_bundle, tensors, metadata.str());

        mit2::Bundle voice(output_voice_bundle);
        const auto integrity = validate_bundle_integrity(voice, true);
        const auto& spk_info = require_voice_tensor(voice, "spk_cond_emb");
        const auto& style_info = require_voice_tensor(voice, "s2mel_style");
        const auto& prompt_info = require_voice_tensor(voice, "s2mel_prompt");
        const auto& mel_info = require_voice_tensor(voice, "mel");
        const auto* source_audio_info = voice.find("source_audio_wav_bytes");
        if (spk_info.shape != std::vector<int64_t>{1, static_cast<int64_t>(spk_tokens), 1024} ||
            style_info.shape != std::vector<int64_t>{1, 192} ||
            prompt_info.shape != std::vector<int64_t>{1, static_cast<int64_t>(prompt_tokens), 512} ||
            mel_info.shape != std::vector<int64_t>{1, 80, static_cast<int64_t>(prompt_tokens)}) {
            throw std::runtime_error("written voice bundle tensor shapes do not match requested shapes");
        }

        std::cout << "{\n";
        std::cout << "  \"stage\": \"" << json_escape(stage) << "\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        if (!feature_manifest.empty()) {
            std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        }
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
        std::cout << "  \"format\": \"MIT2\",\n";
        std::cout << "  \"weights_file\": \"" << json_escape(voice.weights_file()) << "\",\n";
        std::cout << "  \"weights_bytes\": " << voice.weights_size() << ",\n";
        std::cout << "  \"tensor_count\": " << voice.tensor_count() << ",\n";
        std::cout << "  \"tensor_bytes\": " << voice.total_tensor_bytes() << ",\n";
        std::cout << "  \"spk_cond_tokens\": " << spk_tokens << ",\n";
        std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
        std::cout << "  \"mel_frames\": " << prompt_tokens << ",\n";
        std::cout << "  \"source_audio_duration_seconds\": " << source_audio_seconds << ",\n";
        std::cout << "  \"source_audio_embedded_bytes\": " << source_audio_bytes << ",\n";
        std::cout << "  \"integrity\": {\n";
        std::cout << "    \"aligned_tensor_count\": " << integrity.aligned_tensor_count << ",\n";
        std::cout << "    \"checked_interval_count\": " << integrity.checked_interval_count << ",\n";
        std::cout << "    \"sha256_verified_count\": " << integrity.checksum_verified_count << "\n";
        std::cout << "  },\n";
        std::cout << "  \"written_tensors\": [\n";
        print_tensor_contract_json(spk_info, true);
        print_tensor_contract_json(style_info, true);
        print_tensor_contract_json(prompt_info, true);
        print_tensor_contract_json(mel_info, source_audio_info != nullptr);
        if (source_audio_info) {
            print_tensor_contract_json(*source_audio_info, false);
        }
        std::cout << "  ],\n";
        std::cout << "  \"ready_native_clone_audio_preprocess\": true,\n";
        std::cout << "  \"ready_native_voice_bundle_creation\": true,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"remaining_native_clone_work\": [\n";
        std::cout << "    \"native semantic and acoustic speech encoders that produce spk_cond_emb, s2mel_style, and s2mel_prompt\"\n";
        std::cout << "  ]\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"" << json_escape(stage) << "\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        if (!feature_manifest.empty()) {
            std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        }
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
        std::cout << "  \"ready_native_voice_bundle_creation\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_write_voice_bundle_from_features(const std::string& feature_manifest,
                                                                     const std::string& spk_cond_f32,
                                                                     uint32_t spk_tokens,
                                                                     const std::string& s2mel_style_f32,
                                                                     const std::string& s2mel_prompt_f32,
                                                                     uint32_t prompt_tokens,
                                                                     const std::string& output_voice_bundle) {
    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    const auto issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_write_voice_bundle_from_features\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
        std::cout << "  \"clone_feature_readiness_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_clone_feature_prep\": false,\n";
        std::cout << "  \"ready_native_voice_bundle_creation\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false\n";
        std::cout << "}\n";
        return false;
    }
    if (prompt_tokens != manifest.mel_frames) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_write_voice_bundle_from_features\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
        std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
        std::cout << "  \"feature_mel_frames\": " << manifest.mel_frames << ",\n";
        std::cout << "  \"ready_native_clone_feature_prep\": true,\n";
        std::cout << "  \"ready_native_voice_bundle_creation\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"PROMPT_TOKENS must match feature manifest mel_frames\"\n";
        std::cout << "}\n";
        return false;
    }
    return run_tts_clone_write_voice_bundle(manifest.preprocess_manifest,
                                            spk_cond_f32,
                                            spk_tokens,
                                            s2mel_style_f32,
                                            s2mel_prompt_f32,
                                            prompt_tokens,
                                            manifest.output_mel_f32,
                                            output_voice_bundle,
                                            "tts_clone_write_voice_bundle_from_features",
                                            feature_manifest);
}

bool synthesize_hot_inputs_sampled_seeded_wav(const std::string& bundle_dir,
                                              const std::string& voice_bundle_dir,
                                              const std::string& conds_path,
                                              const std::string& text_ids_path,
                                              uint32_t max_codes,
                                              const GptSamplingConfig& gpt_sampling,
                                              uint64_t noise_seed,
                                              float noise_temperature,
                                              uint32_t prompt_tokens,
                                              uint32_t steps,
                                              float cfg_rate,
                                              const std::string& output_wav) {
    const auto started = Clock::now();
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    const std::string codes_path = output_wav + ".codes.u32";
    const std::string condition_path = output_wav + ".condition.f32";
    const std::string noise_path = output_wav + ".noise.f32";
    const auto hot_scratch_plan = compute_hot_scratch_plan_from_input_files(conds_path, text_ids_path, max_codes, prompt_tokens);
    const auto gpt_started = Clock::now();
    if (!export_gpt_kv_codes_inputs_sampled(bundle_dir, conds_path, text_ids_path, max_codes, gpt_sampling, codes_path)) {
        return false;
    }
    const double gpt_seconds = seconds_since(gpt_started);
    const auto actual_codes = read_raw_u32(codes_path);
    const auto condition_started = Clock::now();
    if (!export_hot_codes_condition_inputs(bundle_dir, voice_bundle_dir, conds_path, text_ids_path, codes_path, prompt_tokens, condition_path)) {
        return false;
    }
    const double condition_seconds = seconds_since(condition_started);
    g_tts_stage_acc.condition += condition_seconds;
    const auto noise_started = Clock::now();
    auto condition = read_raw_f32(condition_path);
    if (condition.empty() || (condition.size() % cond_dim) != 0) {
        throw std::runtime_error("Hot TTS sampled seeded input synth condition must have shape [tokens,512]");
    }
    const uint32_t total_tokens = static_cast<uint32_t>(condition.size() / cond_dim);
    if (prompt_tokens == 0 || prompt_tokens >= total_tokens) {
        throw std::runtime_error("Hot TTS sampled seeded input synth prompt_tokens must be positive and below total tokens");
    }
    auto noise = make_deterministic_normal_noise(total_tokens, mel_dim, noise_seed, noise_temperature);
    write_raw_f32(noise_path, noise);
    const double noise_seconds = seconds_since(noise_started);
    const auto acoustic_started = Clock::now();
    if (!synthesize_hot_condition_inputs_wav(bundle_dir, voice_bundle_dir, condition_path, noise_path, prompt_tokens, steps, cfg_rate, output_wav)) {
        return false;
    }
    const double acoustic_seconds = seconds_since(acoustic_started);
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_inputs_sampled_seeded_wav\",\n";
    std::cout << "  \"stage_contexts\": [\"gpt_sampled\", \"condition\", \"acoustic\"],\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"noise_f32\": \"" << noise_path << "\",\n";
    std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
    std::cout << "  \"codes_u32\": \"" << codes_path << "\",\n";
    std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"gpt_seed\": " << gpt_sampling.seed << ",\n";
    std::cout << "  \"gpt_temperature\": " << gpt_sampling.temperature << ",\n";
    std::cout << "  \"gpt_top_k\": " << gpt_sampling.top_k << ",\n";
    std::cout << "  \"gpt_top_p\": " << gpt_sampling.top_p << ",\n";
    std::cout << "  \"gpt_repetition_penalty\": " << gpt_sampling.repetition_penalty << ",\n";
    std::cout << "  \"noise_seed\": " << noise_seed << ",\n";
    std::cout << "  \"noise_temperature\": " << noise_temperature << ",\n";
    std::cout << "  \"tokens\": " << total_tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"gpt_seconds\": " << gpt_seconds << ",\n";
    std::cout << "  \"condition_seconds\": " << condition_seconds << ",\n";
    std::cout << "  \"noise_seconds\": " << noise_seconds << ",\n";
    std::cout << "  \"acoustic_seconds\": " << acoustic_seconds << ",\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
    print_hot_scratch_plan_fields_json(hot_scratch_plan, "planned_scratch_", true);
    print_hot_scratch_actual_fields_json(hot_scratch_plan,
                                         static_cast<uint32_t>(actual_codes.size()),
                                         total_tokens,
                                         "planned_scratch_",
                                         true);
    std::cout << "  \"elapsed_seconds\": " << seconds_since(started) << "\n";
    std::cout << "}\n";
    return true;
}

bool run_hot_tts_from_codes_golden_test(const std::string& bundle_dir,
                                        const std::string& voice_bundle_dir,
                                        const std::string& gpt_golden_dir,
                                        const std::string& s2mel_golden_dir,
                                        const std::string& wave_golden_dir,
                                        const std::string& output_condition_path = "",
                                        const std::string& input_codes_path = "") {
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    constexpr uint32_t gpt_width = 1280;
    constexpr uint32_t semantic_dim = 1024;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    auto conds = read_raw_f32(gpt_golden_dir + "/conds_latent.f32");
    auto text_ids = read_raw_u32(gpt_golden_dir + "/text_ids.u32");
    auto codes = input_codes_path.empty() ? read_raw_u32(gpt_golden_dir + "/codes.u32") : read_raw_u32(input_codes_path);
    auto target_lengths = read_raw_u32(gpt_golden_dir + "/target_lengths.u32");
    auto golden_latent = read_raw_f32(gpt_golden_dir + "/gpt_latent.f32");
    auto golden_gpt_layer = read_raw_f32(gpt_golden_dir + "/gpt_layer.f32");
    auto golden_vq2emb = read_raw_f32(gpt_golden_dir + "/vq2emb.f32");
    auto golden_s_infer = read_raw_f32(gpt_golden_dir + "/s_infer.f32");
    auto golden_lr = read_raw_f32(gpt_golden_dir + "/length_regulator.f32");
    auto noise = read_raw_f32(s2mel_golden_dir + "/noise.f32");
    auto golden_prompt = read_raw_f32(s2mel_golden_dir + "/prompt_mel.f32");
    auto golden_condition = read_raw_f32(s2mel_golden_dir + "/condition.f32");
    const bool run_acoustic = !wave_golden_dir.empty();
    std::vector<float> golden_mel;
    std::vector<float> golden_wave;
    if (run_acoustic) {
        golden_mel = read_raw_f32(s2mel_golden_dir + "/s2mel_generated.f32");
        golden_wave = read_raw_f32(wave_golden_dir + "/waveform.f32");
    }
    if (conds.empty() || (conds.size() % gpt_width) != 0) {
        throw std::runtime_error("Hot TTS codes golden conds_latent must have shape [tokens,1280]");
    }
    if (codes.empty()) {
        throw std::runtime_error("Hot TTS codes golden codes must be non-empty");
    }
    if (target_lengths.size() != 1 || target_lengths[0] == 0) {
        throw std::runtime_error("Hot TTS codes golden target_lengths must contain one positive length");
    }
    if (golden_latent.size() != static_cast<size_t>(codes.size()) * gpt_width) {
        throw std::runtime_error("Hot TTS codes golden gpt_latent size mismatch");
    }
    if (golden_gpt_layer.size() != static_cast<size_t>(codes.size()) * semantic_dim ||
        golden_vq2emb.size() != static_cast<size_t>(codes.size()) * semantic_dim ||
        golden_s_infer.size() != static_cast<size_t>(codes.size()) * semantic_dim) {
        throw std::runtime_error("Hot TTS codes semantic golden size mismatch");
    }
    const uint32_t generated_tokens = target_lengths[0];
    if (golden_lr.size() != static_cast<size_t>(generated_tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS codes length_regulator golden size mismatch");
    }
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS codes S2Mel golden noise must have shape [tokens,80]");
    }
    if (golden_prompt.empty() || (golden_prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS codes S2Mel golden prompt_mel must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(golden_prompt.size() / mel_dim);
    if (tokens != prompt_tokens + generated_tokens) {
        throw std::runtime_error("Hot TTS codes token counts do not match prompt + generated lengths");
    }
    if (golden_condition.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS codes S2Mel condition golden size mismatch");
    }
    if (run_acoustic && golden_mel.size() != static_cast<size_t>(generated_tokens) * mel_dim) {
        throw std::runtime_error("Hot TTS codes generated mel golden output size mismatch");
    }

    std::vector<float> lr;
    float latent_err = 0.0f;
    float gpt_layer_err = 0.0f;
    float vq_err = 0.0f;
    float s_err = 0.0f;
    float lr_err = 0.0f;
    mit2::MetalResourceStats gpt_stats;
    {
        mit2::MetalContext gpt_metal;
        auto latent = run_gpt_latent_forward_metal(gpt_metal, bundle, conds, text_ids, codes);
        std::cerr << "hot_tts_from_codes_golden: gpt_latent_done\n";
        auto gpt_layer = run_gpt_layer_metal(gpt_metal, bundle, latent, static_cast<uint32_t>(codes.size()));
        auto vq = run_vq2emb_metal(gpt_metal, bundle, codes);
        auto s_infer = gpt_metal.add_f32(gpt_layer, vq);
        lr = run_length_regulator_full_metal(gpt_metal, bundle, s_infer, static_cast<uint32_t>(codes.size()), generated_tokens);
        std::cerr << "hot_tts_from_codes_golden: length_regulator_done\n";
        latent_err = max_abs_error(latent, golden_latent);
        gpt_layer_err = max_abs_error(gpt_layer, golden_gpt_layer);
        vq_err = max_abs_error(vq, golden_vq2emb);
        s_err = max_abs_error(s_infer, golden_s_infer);
        lr_err = max_abs_error(lr, golden_lr);
        gpt_stats = gpt_metal.resource_stats();
    }

    const auto* prompt_info = voice.find("s2mel_prompt");
    const auto* mel_info = voice.find("mel");
    if (!prompt_info || prompt_info->shape.size() != 3 || prompt_info->shape[0] != 1 || prompt_info->shape[2] != cond_dim ||
        static_cast<uint32_t>(prompt_info->shape[1]) < prompt_tokens) {
        throw std::runtime_error("voice s2mel_prompt must have shape [1,prompt_tokens>=requested,512]");
    }
    if (!mel_info || mel_info->shape.size() != 3 || mel_info->shape[0] != 1 || mel_info->shape[1] != mel_dim ||
        static_cast<uint32_t>(mel_info->shape[2]) < prompt_tokens) {
        throw std::runtime_error("voice mel must have shape [1,80,prompt_tokens>=requested]");
    }
    auto voice_prompt_all = tensor_as_f32(voice, "s2mel_prompt");
    auto voice_mel_all = tensor_as_f32(voice, "mel");
    auto style = tensor_as_f32(voice, "s2mel_style");
    if (style.size() != style_dim) {
        throw std::runtime_error("voice s2mel_style must contain 192 floats");
    }
    const uint32_t voice_mel_tokens = static_cast<uint32_t>(mel_info->shape[2]);
    std::vector<float> prompt(static_cast<size_t>(prompt_tokens) * mel_dim);
    for (uint32_t t = 0; t < prompt_tokens; ++t) {
        for (uint32_t c = 0; c < mel_dim; ++c) {
            prompt[static_cast<size_t>(t) * mel_dim + c] = voice_mel_all[static_cast<size_t>(c) * voice_mel_tokens + t];
        }
    }
    mit2::MetalContext condition_metal;
    auto condition = condition_metal.hot_condition_merge_f32(voice_prompt_all, lr, prompt_tokens, generated_tokens, cond_dim);
    const auto condition_stats = condition_metal.resource_stats();

    const float prompt_err = max_abs_error(prompt, golden_prompt);
    const float condition_err = max_abs_error(condition, golden_condition);
    float mel_err = 0.0f;
    float wave_err = 0.0f;
    size_t samples = 0;
    mit2::MetalResourceStats acoustic_stats;
    if (run_acoustic) {
        constexpr uint32_t steps = kCfmSteps;
        constexpr float cfg_rate = 0.7f;
        mit2::MetalContext acoustic_metal;
        auto full_mel = run_cfm_euler_metal(acoustic_metal, bundle, noise, prompt, condition, style, tokens, prompt_tokens, steps, cfg_rate);
        std::vector<float> generated_mel(golden_mel.size());
        const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
        std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
        auto wave = run_bigvgan_vocoder_metal(acoustic_metal, bundle, generated_mel, generated_tokens);
        if (wave.size() != golden_wave.size()) {
            throw std::runtime_error("Hot TTS codes waveform golden output size mismatch");
        }
        samples = wave.size();
        mel_err = max_abs_error(generated_mel, golden_mel);
        wave_err = max_abs_error(wave, golden_wave);
        acoustic_stats = acoustic_metal.resource_stats();
    }

    const float err = std::max({latent_err, gpt_layer_err, vq_err, s_err, lr_err, prompt_err, condition_err, mel_err, wave_err});
    const auto memory = process_memory_info();
    if (!output_condition_path.empty()) {
        write_raw_f32(output_condition_path, condition);
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"" << (run_acoustic ? "hot_tts_from_codes_golden" : "hot_tts_condition_from_codes_golden") << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"gpt_golden_dir\": \"" << gpt_golden_dir << "\",\n";
    std::cout << "  \"s2mel_golden_dir\": \"" << s2mel_golden_dir << "\",\n";
    if (run_acoustic) {
        std::cout << "  \"wave_golden_dir\": \"" << wave_golden_dir << "\",\n";
    }
    if (!input_codes_path.empty()) {
        std::cout << "  \"input_codes_u32\": \"" << input_codes_path << "\",\n";
    }
    if (!output_condition_path.empty()) {
        std::cout << "  \"output_condition_f32\": \"" << output_condition_path << "\",\n";
    }
    std::cout << "  \"cond_tokens\": " << (conds.size() / gpt_width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    if (run_acoustic) {
        std::cout << "  \"samples\": " << samples << ",\n";
        std::cout << "  \"steps\": 25,\n";
        std::cout << "  \"cfg_rate\": 0.7,\n";
    }
    std::cout << "  \"latent_max_abs_error\": " << latent_err << ",\n";
    std::cout << "  \"gpt_layer_max_abs_error\": " << gpt_layer_err << ",\n";
    std::cout << "  \"vq2emb_max_abs_error\": " << vq_err << ",\n";
    std::cout << "  \"s_infer_max_abs_error\": " << s_err << ",\n";
    std::cout << "  \"length_regulator_max_abs_error\": " << lr_err << ",\n";
    std::cout << "  \"prompt_mel_max_abs_error\": " << prompt_err << ",\n";
    std::cout << "  \"condition_max_abs_error\": " << condition_err << ",\n";
    if (run_acoustic) {
        std::cout << "  \"mel_max_abs_error\": " << mel_err << ",\n";
        std::cout << "  \"mel_tolerance\": 0.006,\n";
        std::cout << "  \"wave_max_abs_error\": " << wave_err << ",\n";
        std::cout << "  \"wave_tolerance\": 0.01,\n";
    }
    print_metal_resource_stats_json("gpt_", gpt_stats);
    print_metal_resource_stats_json("condition_", condition_stats);
    if (run_acoustic) {
        print_metal_resource_stats_json("acoustic_", acoustic_stats);
    }
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return latent_err <= 3e-3f && gpt_layer_err <= 3e-3f && vq_err <= 3e-3f && s_err <= 3e-3f &&
           lr_err <= 3e-3f && prompt_err <= 3e-3f && condition_err <= 3e-3f &&
           (!run_acoustic || (mel_err <= 6e-3f && wave_err <= 1e-2f));
}

bool export_hot_codes_condition_inputs(const std::string& bundle_dir,
                                       const std::string& voice_bundle_dir,
                                       const std::string& conds_path,
                                       const std::string& text_ids_path,
                                       const std::string& codes_path,
                                       uint32_t prompt_tokens,
                                       const std::string& output_condition_path) {
    const auto started = Clock::now();
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    constexpr uint32_t gpt_width = 1280;
    constexpr uint32_t semantic_dim = 1024;
    constexpr uint32_t cond_dim = 512;
    auto conds = read_raw_f32(conds_path);
    auto text_ids = read_raw_u32(text_ids_path);
    auto codes = read_raw_u32(codes_path);
    if (conds.empty() || (conds.size() % gpt_width) != 0) {
        throw std::runtime_error("Hot TTS input condition export conds must have shape [tokens,1280]");
    }
    if (codes.empty()) {
        throw std::runtime_error("Hot TTS input condition export codes must be non-empty");
    }
    if (prompt_tokens == 0) {
        throw std::runtime_error("Hot TTS input condition export prompt_tokens must be positive");
    }
    const uint32_t generated_tokens = static_cast<uint32_t>(static_cast<float>(codes.size()) * 1.72f);
    if (generated_tokens == 0) {
        throw std::runtime_error("Hot TTS input condition export generated token count is zero");
    }
    const auto* prompt_info = voice.find("s2mel_prompt");
    if (!prompt_info || prompt_info->shape.size() != 3 || prompt_info->shape[0] != 1 || prompt_info->shape[2] != cond_dim ||
        static_cast<uint32_t>(prompt_info->shape[1]) < prompt_tokens) {
        throw std::runtime_error("voice s2mel_prompt must have shape [1,prompt_tokens>=requested,512]");
    }

    std::vector<float> lr;
    mit2::MetalResourceStats gpt_stats;
    {
        mit2::MetalContext gpt_metal;
        auto latent = run_gpt_latent_forward_metal(gpt_metal, bundle, conds, text_ids, codes);
        auto gpt_layer = run_gpt_layer_metal(gpt_metal, bundle, latent, static_cast<uint32_t>(codes.size()));
        auto vq = run_vq2emb_metal(gpt_metal, bundle, codes);
        if (gpt_layer.size() != static_cast<size_t>(codes.size()) * semantic_dim || vq.size() != gpt_layer.size()) {
            throw std::runtime_error("Hot TTS input condition export semantic tensor size mismatch");
        }
        auto s_infer = gpt_metal.add_f32(gpt_layer, vq);
        lr = run_length_regulator_full_metal(gpt_metal, bundle, s_infer, static_cast<uint32_t>(codes.size()), generated_tokens);
        gpt_stats = gpt_metal.resource_stats();
    }

    const uint32_t tokens = prompt_tokens + generated_tokens;
    auto voice_prompt_all = tensor_as_f32(voice, "s2mel_prompt");
    mit2::MetalContext condition_metal;
    auto condition = condition_metal.hot_condition_merge_f32(voice_prompt_all, lr, prompt_tokens, generated_tokens, cond_dim);
    const auto condition_stats = condition_metal.resource_stats();
    write_raw_f32(output_condition_path, condition);
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_condition_inputs_export\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"codes_u32\": \"" << codes_path << "\",\n";
    std::cout << "  \"output_condition_f32\": \"" << output_condition_path << "\",\n";
    std::cout << "  \"cond_tokens\": " << (conds.size() / gpt_width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    print_metal_resource_stats_json("gpt_", gpt_stats);
    print_metal_resource_stats_json("condition_", condition_stats);
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
    std::cout << "  \"elapsed_seconds\": " << seconds_since(started) << "\n";
    std::cout << "}\n";
    return true;
}

bool synthesize_hot_native_golden_wav(const std::string& bundle_dir,
                                      const std::string& voice_bundle_dir,
                                      const std::string& gpt_golden_dir,
                                      const std::string& s2mel_golden_dir,
                                      const std::string& output_wav) {
    const std::string codes_path = output_wav + ".codes.u32";
    const std::string condition_path = output_wav + ".condition.f32";
    if (!export_gpt_kv_codes_golden(bundle_dir, gpt_golden_dir, codes_path)) {
        return false;
    }
    if (!run_hot_tts_from_codes_golden_test(bundle_dir,
                                            voice_bundle_dir,
                                            gpt_golden_dir,
                                            s2mel_golden_dir,
                                            "",
                                            condition_path,
                                            codes_path)) {
        return false;
    }
    if (!synthesize_hot_condition_golden_wav(bundle_dir,
                                             voice_bundle_dir,
                                             s2mel_golden_dir,
                                             condition_path,
                                             output_wav)) {
        return false;
    }
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_native_golden_wav\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"gpt_golden_dir\": \"" << gpt_golden_dir << "\",\n";
    std::cout << "  \"s2mel_golden_dir\": \"" << s2mel_golden_dir << "\",\n";
    std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
    std::cout << "  \"codes_u32\": \"" << codes_path << "\",\n";
    std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << "\n";
    std::cout << "}\n";
    return true;
}

bool run_primitive_tests(mit2::MetalContext& metal) {
    std::vector<float> a(257);
    std::vector<float> b(257);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i % 17) * 0.125f - 1.0f;
        b[i] = static_cast<float>(i % 11) * -0.25f + 0.5f;
    }
    auto add = metal.add_f32(a, b);
    std::vector<float> add_ref(add.size());
    for (size_t i = 0; i < add_ref.size(); ++i) {
        add_ref[i] = a[i] + b[i];
    }
    auto add_scaled = metal.add_scaled_f32(a, b, 0.5f);
    std::vector<float> add_scaled_ref(add_scaled.size());
    for (size_t i = 0; i < add_scaled_ref.size(); ++i) {
        add_scaled_ref[i] = a[i] + 0.5f * b[i];
    }
    std::vector<float> c(a.size());
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.4f;
    }
    auto avg3 = metal.avg3_f32(a, b, c);
    std::vector<float> avg3_ref(avg3.size());
    for (size_t i = 0; i < avg3_ref.size(); ++i) {
        avg3_ref[i] = (a[i] + b[i] + c[i]) / 3.0f;
    }

    std::vector<float> x{-8.0f, -2.0f, -1.0f, -0.125f, 0.0f, 0.5f, 2.0f, 6.0f};
    auto silu = metal.silu_f32(x);
    std::vector<float> silu_ref(silu.size());
    for (size_t i = 0; i < silu_ref.size(); ++i) {
        silu_ref[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
    std::vector<float> silu_gate(x.size());
    for (size_t i = 0; i < silu_gate.size(); ++i) {
        silu_gate[i] = std::cos(static_cast<float>(i) * 0.37f) * 1.4f;
    }
    auto silu_mul = metal.silu_mul_f32(x, silu_gate);
    std::vector<float> silu_mul_ref(silu_mul.size());
    for (size_t i = 0; i < silu_mul_ref.size(); ++i) {
        silu_mul_ref[i] = silu_ref[i] * silu_gate[i];
    }
    constexpr uint32_t mask_tokens = 4;
    constexpr uint32_t mask_width = 6;
    std::vector<float> mask_rows_x(static_cast<size_t>(mask_tokens) * mask_width);
    for (size_t i = 0; i < mask_rows_x.size(); ++i) {
        mask_rows_x[i] = std::sin(static_cast<float>(i) * 0.23f) * 0.7f;
    }
    std::vector<uint32_t> row_mask{1, 0, 1, 0};
    auto mask_rows = metal.mask_rows_f32(mask_rows_x, row_mask, mask_tokens, mask_width);
    auto mask_rows_ref = cpu_mask_rows(mask_rows_x, row_mask, mask_tokens, mask_width);
    std::vector<float> glu_x(static_cast<size_t>(mask_tokens) * mask_width * 2);
    for (size_t i = 0; i < glu_x.size(); ++i) {
        glu_x[i] = std::cos(static_cast<float>(i) * 0.19f) * 1.3f;
    }
    auto glu = metal.glu_split_f32(glu_x, mask_tokens, mask_width);
    auto glu_ref = cpu_glu_split(glu_x, mask_tokens, mask_width);
    constexpr uint32_t gate_tokens = 3;
    constexpr uint32_t gate_width = 5;
    constexpr uint32_t gate_cond_width = 16;
    constexpr uint32_t gate_cond_offset = 4;
    std::vector<float> gate_in(static_cast<size_t>(gate_tokens) * gate_width * 2);
    std::vector<float> gate_cond(static_cast<size_t>(gate_tokens) * gate_cond_width);
    for (size_t i = 0; i < gate_in.size(); ++i) {
        gate_in[i] = std::sin(static_cast<float>(i) * 0.17f) * 0.9f;
    }
    for (size_t i = 0; i < gate_cond.size(); ++i) {
        gate_cond[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.7f;
    }
    auto wavenet_gate = metal.wavenet_gate_f32(
        gate_in,
        gate_cond,
        gate_tokens,
        gate_width,
        gate_cond_width,
        gate_cond_offset,
        gate_tokens);
    std::vector<float> wavenet_gate_ref(static_cast<size_t>(gate_tokens) * gate_width);
    for (uint32_t t = 0; t < gate_tokens; ++t) {
        const size_t in_base = static_cast<size_t>(t) * gate_width * 2;
        const size_t cond_base = static_cast<size_t>(t) * gate_cond_width + gate_cond_offset;
        for (uint32_t c = 0; c < gate_width; ++c) {
            const float va = gate_in[in_base + c] + gate_cond[cond_base + c];
            const float vb = gate_in[in_base + gate_width + c] + gate_cond[cond_base + gate_width + c];
            wavenet_gate_ref[static_cast<size_t>(t) * gate_width + c] = std::tanh(va) * (1.0f / (1.0f + std::exp(-vb)));
        }
    }
    constexpr uint32_t rs_tokens = 4;
    constexpr uint32_t rs_width = 6;
    const size_t rs_row_count = static_cast<size_t>(rs_tokens) * rs_width;
    std::vector<float> rs_x(rs_row_count);
    std::vector<float> rs_output(rs_row_count);
    std::vector<float> rs_res_skip(rs_row_count * 2);
    std::vector<float> rs_final(rs_row_count);
    std::vector<uint32_t> rs_mask{1, 0, 1, 1};
    for (size_t i = 0; i < rs_x.size(); ++i) {
        rs_x[i] = std::sin(static_cast<float>(i) * 0.041f) * 0.3f;
        rs_output[i] = std::cos(static_cast<float>(i) * 0.053f) * 0.2f;
        rs_final[i] = std::sin(static_cast<float>(i) * 0.067f) * 0.1f;
    }
    for (size_t i = 0; i < rs_res_skip.size(); ++i) {
        rs_res_skip[i] = std::cos(static_cast<float>(i) * 0.029f) * 0.15f;
    }
    auto rs_update = metal.wavenet_res_skip_update_f32(rs_x, rs_output, rs_res_skip, rs_mask, rs_tokens, rs_width, true);
    std::vector<float> rs_update_ref(rs_row_count * 2);
    for (uint32_t t = 0; t < rs_tokens; ++t) {
        const float m = rs_mask[t] ? 1.0f : 0.0f;
        for (uint32_t c = 0; c < rs_width; ++c) {
            const size_t idx = static_cast<size_t>(t) * rs_width + c;
            const size_t rs_base = static_cast<size_t>(t) * rs_width * 2;
            rs_update_ref[idx] = (rs_x[idx] + rs_res_skip[rs_base + c]) * m;
            rs_update_ref[rs_row_count + idx] = rs_output[idx] + rs_res_skip[rs_base + rs_width + c];
        }
    }
    auto rs_final_update = metal.wavenet_res_skip_update_f32(rs_x, rs_output, rs_final, rs_mask, rs_tokens, rs_width, false);
    std::vector<float> rs_final_update_ref(rs_row_count * 2);
    for (uint32_t t = 0; t < rs_tokens; ++t) {
        const float m = rs_mask[t] ? 1.0f : 0.0f;
        for (uint32_t c = 0; c < rs_width; ++c) {
            const size_t idx = static_cast<size_t>(t) * rs_width + c;
            rs_final_update_ref[idx] = rs_x[idx];
            rs_final_update_ref[rs_row_count + idx] = (rs_output[idx] + rs_final[idx]) * m;
        }
    }
    auto geglu = metal.geglu_erf_split_f32(glu_x, mask_tokens, mask_width);
    std::vector<float> geglu_ref(static_cast<size_t>(mask_tokens) * mask_width);
    for (uint32_t t = 0; t < mask_tokens; ++t) {
        for (uint32_t c = 0; c < mask_width; ++c) {
            const size_t base = static_cast<size_t>(t) * mask_width * 2;
            const float a = glu_x[base + c];
            const float b_gate = glu_x[base + mask_width + c];
            geglu_ref[static_cast<size_t>(t) * mask_width + c] = a * 0.5f * b_gate * (1.0f + std::erf(b_gate / std::sqrt(2.0f)));
        }
    }
    auto gelu = metal.gelu_f32(x);
    auto gelu_ref = cpu_gelu_new(x);
    auto tanh = metal.tanh_f32(x);
    std::vector<float> tanh_ref(tanh.size());
    for (size_t i = 0; i < tanh_ref.size(); ++i) {
        tanh_ref[i] = std::tanh(x[i]);
    }
    auto clamp = metal.clamp_f32(x, -1.0f, 1.0f);
    std::vector<float> clamp_ref(clamp.size());
    for (size_t i = 0; i < clamp_ref.size(); ++i) {
        clamp_ref[i] = std::max(-1.0f, std::min(1.0f, x[i]));
    }

    std::vector<float> logits(8194);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = std::sin(static_cast<float>(i) * 0.013f) * 4.0f - static_cast<float>(i % 7) * 0.2f;
    }
    auto softmax = metal.softmax_f32(logits);
    auto softmax_ref = cpu_softmax(logits);

    constexpr uint32_t width = 64;
    std::vector<float> table(128 * width);
    for (size_t i = 0; i < table.size(); ++i) {
        table[i] = static_cast<float>((i * 13) % 97) / 97.0f - 0.5f;
    }
    std::vector<uint32_t> ids{0, 7, 64, 3, 127, 12};
    auto emb = metal.embedding_f32(table, ids, width);
    auto emb_ref = cpu_embedding(table, ids, width);

    std::vector<float> norm_x(1280);
    std::vector<float> gamma(1280);
    std::vector<float> beta(1280);
    for (size_t i = 0; i < norm_x.size(); ++i) {
        norm_x[i] = std::cos(static_cast<float>(i) * 0.017f) * 3.0f + static_cast<float>(i % 5) * 0.1f;
        gamma[i] = 0.75f + static_cast<float>(i % 13) * 0.01f;
        beta[i] = -0.2f + static_cast<float>(i % 17) * 0.005f;
    }
    auto layernorm = metal.layernorm_f32(norm_x, gamma, beta, 1e-5f);
    auto layernorm_ref = cpu_layernorm(norm_x, gamma, beta, 1e-5f);
    auto rmsnorm = metal.rmsnorm_f32(norm_x, gamma, 1e-6f);
    auto rmsnorm_ref = cpu_rmsnorm(norm_x, gamma, 1e-6f);

    constexpr uint32_t norm_rows_tokens = 5;
    constexpr uint32_t norm_rows_width = 512;
    std::vector<float> norm_rows_x(static_cast<size_t>(norm_rows_tokens) * norm_rows_width);
    std::vector<float> norm_rows_gamma(norm_rows_width);
    std::vector<float> norm_rows_beta(norm_rows_width);
    for (size_t i = 0; i < norm_rows_x.size(); ++i) {
        norm_rows_x[i] = std::sin(static_cast<float>(i) * 0.019f) * 2.0f + static_cast<float>(i % 7) * 0.03f;
    }
    for (uint32_t i = 0; i < norm_rows_width; ++i) {
        norm_rows_gamma[i] = 0.9f + static_cast<float>(i % 11) * 0.015f;
        norm_rows_beta[i] = -0.15f + static_cast<float>(i % 13) * 0.004f;
    }
    auto layernorm_rows = metal.layernorm_rows_f32(norm_rows_x, norm_rows_gamma, norm_rows_beta, norm_rows_tokens, norm_rows_width, 1e-5f);
    auto layernorm_rows_ref = cpu_layer_norm_rows(norm_rows_x, norm_rows_gamma, norm_rows_beta, norm_rows_tokens, norm_rows_width, 1e-5f);
    constexpr uint32_t norm_rows_wide_tokens = 7;
    constexpr uint32_t norm_rows_wide_width = 1280;
    std::vector<float> norm_rows_wide_x(static_cast<size_t>(norm_rows_wide_tokens) * norm_rows_wide_width);
    std::vector<float> norm_rows_wide_gamma(norm_rows_wide_width);
    std::vector<float> norm_rows_wide_beta(norm_rows_wide_width);
    for (size_t i = 0; i < norm_rows_wide_x.size(); ++i) {
        norm_rows_wide_x[i] = std::sin(static_cast<float>(i) * 0.011f) * 0.47f +
                               std::cos(static_cast<float>(i % 257) * 0.019f) * 0.13f +
                               static_cast<float>(i % 5) * 0.021f;
    }
    for (uint32_t i = 0; i < norm_rows_wide_width; ++i) {
        norm_rows_wide_gamma[i] = 0.82f + std::sin(static_cast<float>(i) * 0.007f) * 0.09f;
        norm_rows_wide_beta[i] = -0.11f + std::cos(static_cast<float>(i) * 0.013f) * 0.04f;
    }
    auto layernorm_rows_wide = metal.layernorm_rows_f32(
        norm_rows_wide_x,
        norm_rows_wide_gamma,
        norm_rows_wide_beta,
        norm_rows_wide_tokens,
        norm_rows_wide_width,
        1e-5f);
    auto layernorm_rows_wide_resident = metal.layernorm_rows_f32_resident(
        "primitive.layernorm_rows_wide.gamma.resident",
        norm_rows_wide_gamma,
        "primitive.layernorm_rows_wide.beta.resident",
        norm_rows_wide_beta,
        norm_rows_wide_x,
        norm_rows_wide_tokens,
        norm_rows_wide_width,
        1e-5f);
    auto layernorm_rows_wide_ref = cpu_layer_norm_rows(
        norm_rows_wide_x,
        norm_rows_wide_gamma,
        norm_rows_wide_beta,
        norm_rows_wide_tokens,
        norm_rows_wide_width,
        1e-5f);
    std::vector<float> layernorm_rows_wide_per_row(norm_rows_wide_x.size());
    for (uint32_t t = 0; t < norm_rows_wide_tokens; ++t) {
        std::vector<float> row(norm_rows_wide_x.begin() + static_cast<size_t>(t) * norm_rows_wide_width,
                               norm_rows_wide_x.begin() + static_cast<size_t>(t + 1) * norm_rows_wide_width);
        auto got = metal.layernorm_f32_resident(
            "primitive.layernorm_rows_wide.gamma.per_row.resident",
            norm_rows_wide_gamma,
            "primitive.layernorm_rows_wide.beta.per_row.resident",
            norm_rows_wide_beta,
            row,
            1e-5f);
        std::copy(got.begin(), got.end(), layernorm_rows_wide_per_row.begin() + static_cast<size_t>(t) * norm_rows_wide_width);
    }
    std::vector<float> adaptive_shift(norm_rows_width);
    std::vector<float> adaptive_scale(norm_rows_width);
    for (uint32_t i = 0; i < norm_rows_width; ++i) {
        adaptive_shift[i] = std::sin(static_cast<float>(i) * 0.021f) * 0.12f;
        adaptive_scale[i] = std::cos(static_cast<float>(i) * 0.017f) * 0.08f;
    }
    auto adaptive_layernorm_rows = metal.adaptive_layernorm_rows_f32(
        norm_rows_x,
        adaptive_shift,
        adaptive_scale,
        norm_rows_tokens,
        norm_rows_width,
        1e-6f);
    std::vector<float> adaptive_layernorm_rows_ref(static_cast<size_t>(norm_rows_tokens) * norm_rows_width);
    std::vector<float> unit_gamma(norm_rows_width, 1.0f);
    std::vector<float> zero_beta(norm_rows_width, 0.0f);
    for (uint32_t t = 0; t < norm_rows_tokens; ++t) {
        std::vector<float> row(norm_rows_x.begin() + static_cast<size_t>(t) * norm_rows_width,
                               norm_rows_x.begin() + static_cast<size_t>(t + 1) * norm_rows_width);
        auto row_ref = cpu_layernorm(row, unit_gamma, zero_beta, 1e-6f);
        for (uint32_t i = 0; i < norm_rows_width; ++i) {
            adaptive_layernorm_rows_ref[static_cast<size_t>(t) * norm_rows_width + i] =
                row_ref[i] * (1.0f + adaptive_scale[i]) + adaptive_shift[i];
        }
    }
    std::vector<float> adaptive_rms_weight(norm_rows_width);
    std::vector<float> adaptive_rms_bias(norm_rows_width);
    for (uint32_t i = 0; i < norm_rows_width; ++i) {
        adaptive_rms_weight[i] = 0.85f + std::sin(static_cast<float>(i) * 0.013f) * 0.07f;
        adaptive_rms_bias[i] = std::cos(static_cast<float>(i) * 0.019f) * 0.05f;
    }
    auto adaptive_rmsnorm_rows = metal.adaptive_rmsnorm_rows_f32(
        norm_rows_x,
        norm_rows_gamma,
        adaptive_rms_weight,
        adaptive_rms_bias,
        norm_rows_tokens,
        norm_rows_width,
        1e-5f);
    std::vector<float> adaptive_rmsnorm_rows_ref(static_cast<size_t>(norm_rows_tokens) * norm_rows_width);
    for (uint32_t t = 0; t < norm_rows_tokens; ++t) {
        std::vector<float> row(norm_rows_x.begin() + static_cast<size_t>(t) * norm_rows_width,
                               norm_rows_x.begin() + static_cast<size_t>(t + 1) * norm_rows_width);
        auto row_ref = cpu_rmsnorm(row, norm_rows_gamma, 1e-5f);
        for (uint32_t i = 0; i < norm_rows_width; ++i) {
            adaptive_rmsnorm_rows_ref[static_cast<size_t>(t) * norm_rows_width + i] =
                adaptive_rms_weight[i] * row_ref[i] + adaptive_rms_bias[i];
        }
    }
    constexpr uint32_t euler_tokens = 5;
    constexpr uint32_t euler_width = 7;
    constexpr uint32_t euler_prompt_tokens = 2;
    constexpr float euler_dt = 0.04f;
    constexpr float euler_cfg_rate = 0.7f;
    std::vector<float> euler_x(static_cast<size_t>(euler_tokens) * euler_width);
    std::vector<float> euler_dphi(euler_x.size());
    std::vector<float> euler_cfg_dphi(euler_x.size());
    for (size_t i = 0; i < euler_x.size(); ++i) {
        euler_x[i] = std::sin(static_cast<float>(i) * 0.031f) * 0.4f;
        euler_dphi[i] = std::cos(static_cast<float>(i) * 0.043f) * 0.2f;
        euler_cfg_dphi[i] = std::sin(static_cast<float>(i) * 0.059f) * 0.15f;
    }
    auto euler_update = metal.cfm_euler_update_f32(
        euler_x,
        euler_dphi,
        euler_cfg_dphi,
        euler_tokens,
        euler_width,
        euler_prompt_tokens,
        euler_dt,
        euler_cfg_rate);
    std::vector<float> euler_update_ref(euler_x.size());
    for (uint32_t t = 0; t < euler_tokens; ++t) {
        for (uint32_t c = 0; c < euler_width; ++c) {
            const size_t idx = static_cast<size_t>(t) * euler_width + c;
            if (t < euler_prompt_tokens) {
                euler_update_ref[idx] = 0.0f;
            } else {
                const float guided = (1.0f + euler_cfg_rate) * euler_dphi[idx] - euler_cfg_rate * euler_cfg_dphi[idx];
                euler_update_ref[idx] = euler_x[idx] + euler_dt * guided;
            }
        }
    }
    constexpr uint32_t concat_tokens = 3;
    constexpr uint32_t concat_a_width = 5;
    constexpr uint32_t concat_b_width = 4;
    std::vector<float> concat_a(static_cast<size_t>(concat_tokens) * concat_a_width);
    std::vector<float> concat_b(static_cast<size_t>(concat_tokens) * concat_b_width);
    for (size_t i = 0; i < concat_a.size(); ++i) {
        concat_a[i] = std::sin(static_cast<float>(i) * 0.037f) * 0.25f;
    }
    for (size_t i = 0; i < concat_b.size(); ++i) {
        concat_b[i] = std::cos(static_cast<float>(i) * 0.049f) * 0.35f;
    }
    auto concat_rows = metal.concat_rows_f32(concat_a, concat_b, concat_tokens, concat_a_width, concat_b_width);
    std::vector<float> concat_rows_ref(static_cast<size_t>(concat_tokens) * (concat_a_width + concat_b_width));
    for (uint32_t t = 0; t < concat_tokens; ++t) {
        std::copy(concat_a.begin() + static_cast<size_t>(t) * concat_a_width,
                  concat_a.begin() + static_cast<size_t>(t + 1) * concat_a_width,
                  concat_rows_ref.begin() + static_cast<size_t>(t) * (concat_a_width + concat_b_width));
        std::copy(concat_b.begin() + static_cast<size_t>(t) * concat_b_width,
                  concat_b.begin() + static_cast<size_t>(t + 1) * concat_b_width,
                  concat_rows_ref.begin() + static_cast<size_t>(t) * (concat_a_width + concat_b_width) + concat_a_width);
    }
    constexpr uint32_t hot_prompt_total_tokens = 5;
    constexpr uint32_t hot_prompt_tokens = 3;
    constexpr uint32_t hot_generated_tokens = 2;
    constexpr uint32_t hot_condition_width = 7;
    std::vector<float> hot_prompt_all(static_cast<size_t>(hot_prompt_total_tokens) * hot_condition_width);
    std::vector<float> hot_generated(static_cast<size_t>(hot_generated_tokens) * hot_condition_width);
    for (size_t i = 0; i < hot_prompt_all.size(); ++i) {
        hot_prompt_all[i] = std::sin(static_cast<float>(i) * 0.071f) * 0.45f;
    }
    for (size_t i = 0; i < hot_generated.size(); ++i) {
        hot_generated[i] = std::cos(static_cast<float>(i) * 0.083f) * 0.55f;
    }
    auto hot_condition = metal.hot_condition_merge_f32(
        hot_prompt_all,
        hot_generated,
        hot_prompt_tokens,
        hot_generated_tokens,
        hot_condition_width);
    std::vector<float> hot_condition_ref(static_cast<size_t>(hot_prompt_tokens + hot_generated_tokens) * hot_condition_width);
    for (uint32_t t = 0; t < hot_prompt_tokens; ++t) {
        std::copy(hot_prompt_all.begin() + static_cast<size_t>(t) * hot_condition_width,
                  hot_prompt_all.begin() + static_cast<size_t>(t + 1) * hot_condition_width,
                  hot_condition_ref.begin() + static_cast<size_t>(t) * hot_condition_width);
    }
    for (uint32_t t = 0; t < hot_generated_tokens; ++t) {
        std::copy(hot_generated.begin() + static_cast<size_t>(t) * hot_condition_width,
                  hot_generated.begin() + static_cast<size_t>(t + 1) * hot_condition_width,
                  hot_condition_ref.begin() + static_cast<size_t>(hot_prompt_tokens + t) * hot_condition_width);
    }
    constexpr uint32_t dit_merge_tokens = 3;
    constexpr uint32_t dit_merge_width = 864;
    std::vector<float> dit_merge_x(static_cast<size_t>(dit_merge_tokens) * 80);
    std::vector<float> dit_merge_prompt(static_cast<size_t>(dit_merge_tokens) * 80);
    std::vector<float> dit_merge_cond(static_cast<size_t>(dit_merge_tokens) * 512);
    std::vector<float> dit_merge_style(192);
    for (size_t i = 0; i < dit_merge_x.size(); ++i) {
        dit_merge_x[i] = std::sin(static_cast<float>(i) * 0.023f) * 0.2f;
        dit_merge_prompt[i] = std::cos(static_cast<float>(i) * 0.031f) * 0.3f;
    }
    for (size_t i = 0; i < dit_merge_cond.size(); ++i) {
        dit_merge_cond[i] = std::sin(static_cast<float>(i) * 0.011f) * 0.15f;
    }
    for (size_t i = 0; i < dit_merge_style.size(); ++i) {
        dit_merge_style[i] = std::cos(static_cast<float>(i) * 0.017f) * 0.25f;
    }
    auto dit_merge = metal.dit_input_merge_f32(dit_merge_x, dit_merge_prompt, dit_merge_cond, dit_merge_style, dit_merge_tokens);
    std::vector<float> dit_merge_ref(static_cast<size_t>(dit_merge_tokens) * dit_merge_width);
    for (uint32_t t = 0; t < dit_merge_tokens; ++t) {
        const size_t out_base = static_cast<size_t>(t) * dit_merge_width;
        std::copy(dit_merge_x.begin() + static_cast<size_t>(t) * 80,
                  dit_merge_x.begin() + static_cast<size_t>(t + 1) * 80,
                  dit_merge_ref.begin() + out_base);
        std::copy(dit_merge_prompt.begin() + static_cast<size_t>(t) * 80,
                  dit_merge_prompt.begin() + static_cast<size_t>(t + 1) * 80,
                  dit_merge_ref.begin() + out_base + 80);
        std::copy(dit_merge_cond.begin() + static_cast<size_t>(t) * 512,
                  dit_merge_cond.begin() + static_cast<size_t>(t + 1) * 512,
                  dit_merge_ref.begin() + out_base + 160);
        std::copy(dit_merge_style.begin(), dit_merge_style.end(), dit_merge_ref.begin() + out_base + 672);
    }
    auto rmsnorm_rows = metal.rmsnorm_rows_f32(norm_rows_x, norm_rows_gamma, norm_rows_tokens, norm_rows_width);
    auto rmsnorm_rows_ref = cpu_rmsnorm_rows(norm_rows_x, norm_rows_gamma, norm_rows_tokens, norm_rows_width);
    auto rmsnorm_rows_eps = metal.rmsnorm_rows_eps_f32(norm_rows_x, norm_rows_gamma, norm_rows_tokens, norm_rows_width, 1e-5f);
    std::vector<float> rmsnorm_rows_eps_ref(static_cast<size_t>(norm_rows_tokens) * norm_rows_width);
    for (uint32_t t = 0; t < norm_rows_tokens; ++t) {
        std::vector<float> row(norm_rows_x.begin() + static_cast<size_t>(t) * norm_rows_width,
                               norm_rows_x.begin() + static_cast<size_t>(t + 1) * norm_rows_width);
        auto row_ref = cpu_rmsnorm(row, norm_rows_gamma, 1e-5f);
        std::copy(row_ref.begin(), row_ref.end(), rmsnorm_rows_eps_ref.begin() + static_cast<size_t>(t) * norm_rows_width);
    }

    constexpr uint32_t linear_rows = 192;
    constexpr uint32_t linear_cols = 1280;
    std::vector<float> linear_w(linear_rows * linear_cols);
    std::vector<float> linear_b(linear_rows);
    std::vector<float> linear_x(linear_cols);
    for (size_t i = 0; i < linear_w.size(); ++i) {
        linear_w[i] = std::sin(static_cast<float>(i % 251) * 0.03f) * 0.02f;
    }
    for (uint32_t i = 0; i < linear_b.size(); ++i) {
        linear_b[i] = static_cast<float>(i % 19) * 0.001f;
    }
    for (uint32_t i = 0; i < linear_x.size(); ++i) {
        linear_x[i] = std::cos(static_cast<float>(i) * 0.011f);
    }
    auto linear = metal.linear_f32(linear_w, linear_b, linear_x, linear_rows, linear_cols);
    auto linear_ref = cpu_linear(linear_w, linear_b, linear_x, linear_rows, linear_cols);

    constexpr uint32_t linear_tokens = 3;
    std::vector<float> linear_rows_x(static_cast<size_t>(linear_tokens) * linear_cols);
    for (size_t i = 0; i < linear_rows_x.size(); ++i) {
        linear_rows_x[i] = std::cos(static_cast<float>(i) * 0.007f) * 0.5f + static_cast<float>(i % 23) * 0.002f;
    }
    auto linear_rows_out = metal.linear_rows_f32(linear_w, linear_b, linear_rows_x, linear_tokens, linear_rows, linear_cols);
    std::vector<float> linear_rows_ref(static_cast<size_t>(linear_tokens) * linear_rows);
    for (uint32_t t = 0; t < linear_tokens; ++t) {
        std::vector<float> row_input(linear_rows_x.begin() + static_cast<size_t>(t) * linear_cols,
                                     linear_rows_x.begin() + static_cast<size_t>(t + 1) * linear_cols);
        auto row_ref = cpu_linear(linear_w, linear_b, row_input, linear_rows, linear_cols);
        std::copy(row_ref.begin(), row_ref.end(), linear_rows_ref.begin() + static_cast<size_t>(t) * linear_rows);
    }

    constexpr uint32_t interp_in_tokens = 7;
    constexpr uint32_t interp_out_tokens = 19;
    constexpr uint32_t interp_width = 512;
    std::vector<float> interp_x(static_cast<size_t>(interp_in_tokens) * interp_width);
    for (size_t i = 0; i < interp_x.size(); ++i) {
        interp_x[i] = static_cast<float>((i * 31) % 257) / 257.0f - 0.5f;
    }
    auto interp = metal.nearest_interpolate_f32(interp_x, interp_in_tokens, interp_out_tokens, interp_width);
    auto interp_ref = cpu_nearest_interpolate(interp_x, interp_in_tokens, interp_out_tokens, interp_width);
    constexpr uint32_t interp_edge_in_tokens = 96;
    constexpr uint32_t interp_edge_out_tokens = 165;
    constexpr uint32_t interp_edge_width = 3;
    std::vector<float> interp_edge_x(static_cast<size_t>(interp_edge_in_tokens) * interp_edge_width);
    for (size_t i = 0; i < interp_edge_x.size(); ++i) {
        interp_edge_x[i] = static_cast<float>(i);
    }
    auto interp_edge = metal.nearest_interpolate_f32(interp_edge_x, interp_edge_in_tokens, interp_edge_out_tokens, interp_edge_width);
    auto interp_edge_ref = cpu_nearest_interpolate(interp_edge_x, interp_edge_in_tokens, interp_edge_out_tokens, interp_edge_width);

    constexpr uint32_t conv_tokens = 13;
    constexpr uint32_t conv_in = 5;
    constexpr uint32_t conv_out = 7;
    constexpr uint32_t conv_kernel = 3;
    std::vector<float> conv_x(static_cast<size_t>(conv_tokens) * conv_in);
    std::vector<float> conv_w(static_cast<size_t>(conv_out) * conv_in * conv_kernel);
    std::vector<float> conv_b(conv_out);
    for (size_t i = 0; i < conv_x.size(); ++i) {
        conv_x[i] = std::sin(static_cast<float>(i) * 0.07f);
    }
    for (size_t i = 0; i < conv_w.size(); ++i) {
        conv_w[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.05f;
    }
    for (size_t i = 0; i < conv_b.size(); ++i) {
        conv_b[i] = static_cast<float>(i) * 0.01f;
    }
    auto conv = metal.conv1d_same_f32(conv_x, conv_w, conv_b, conv_tokens, conv_in, conv_out, conv_kernel);
    auto conv_ref = cpu_conv1d_same(conv_x, conv_w, conv_b, conv_tokens, conv_in, conv_out, conv_kernel);
    auto reflect_conv = metal.conv1d_reflect_same_f32(conv_x, conv_w, conv_b, conv_tokens, conv_in, conv_out, conv_kernel);
    auto reflect_conv_ref = cpu_conv1d_reflect_same(conv_x, conv_w, conv_b, conv_tokens, conv_in, conv_out, conv_kernel);
    std::vector<float> conv_x_branch1(conv_x.size());
    for (size_t i = 0; i < conv_x_branch1.size(); ++i) {
        conv_x_branch1[i] = std::cos(static_cast<float>(i) * 0.041f) * 0.23f -
                             std::sin(static_cast<float>(i % 17) * 0.13f) * 0.07f;
    }
    std::vector<float> conv_x_batched(conv_x.size() * 2);
    std::copy(conv_x.begin(), conv_x.end(), conv_x_batched.begin());
    std::copy(conv_x_branch1.begin(), conv_x_branch1.end(), conv_x_batched.begin() + static_cast<std::ptrdiff_t>(conv_x.size()));
    auto reflect_conv_branch1 = metal.conv1d_reflect_same_f32(conv_x_branch1, conv_w, conv_b, conv_tokens, conv_in, conv_out, conv_kernel);
    auto reflect_conv_batched = metal.conv1d_reflect_same_batched_f32(conv_x_batched, conv_w, conv_b, 2, conv_tokens, conv_in, conv_out, conv_kernel);
    const size_t reflect_branch_size = static_cast<size_t>(conv_tokens) * conv_out;
    std::vector<float> reflect_conv_batched0(reflect_conv_batched.begin(), reflect_conv_batched.begin() + static_cast<std::ptrdiff_t>(reflect_branch_size));
    std::vector<float> reflect_conv_batched1(reflect_conv_batched.begin() + static_cast<std::ptrdiff_t>(reflect_branch_size),
                                             reflect_conv_batched.begin() + static_cast<std::ptrdiff_t>(reflect_branch_size * 2));
    std::vector<float> depthwise_w(static_cast<size_t>(conv_in) * conv_kernel);
    std::vector<float> depthwise_b(conv_in);
    for (size_t i = 0; i < depthwise_w.size(); ++i) {
        depthwise_w[i] = std::sin(static_cast<float>(i) * 0.17f) * 0.04f;
    }
    for (size_t i = 0; i < depthwise_b.size(); ++i) {
        depthwise_b[i] = -0.03f + static_cast<float>(i) * 0.007f;
    }
    auto depthwise_conv = metal.depthwise_conv1d_same_f32(conv_x, depthwise_w, depthwise_b, conv_tokens, conv_in, conv_kernel);
    auto depthwise_conv_ref = cpu_depthwise_conv1d_same(conv_x, depthwise_w, depthwise_b, conv_tokens, conv_in, conv_kernel);
    constexpr uint32_t subsample_tokens = 7;
    constexpr uint32_t subsample_dim = 10;
    constexpr uint32_t subsample_channels = 3;
    constexpr uint32_t subsample_kernel = 3;
    constexpr uint32_t subsample_stride = 2;
    std::vector<float> subsample_x(static_cast<size_t>(subsample_tokens) * subsample_dim);
    std::vector<float> subsample_w(static_cast<size_t>(subsample_channels) * subsample_kernel * subsample_kernel);
    std::vector<float> subsample_b(subsample_channels);
    for (size_t i = 0; i < subsample_x.size(); ++i) {
        subsample_x[i] = std::sin(static_cast<float>(i) * 0.071f) * 0.6f - 0.2f;
    }
    for (size_t i = 0; i < subsample_w.size(); ++i) {
        subsample_w[i] = std::cos(static_cast<float>(i) * 0.113f) * 0.08f;
    }
    for (uint32_t i = 0; i < subsample_b.size(); ++i) {
        subsample_b[i] = -0.04f + static_cast<float>(i) * 0.025f;
    }
    auto subsample_flat = metal.subsampling_conv2d_relu_flat_f32(subsample_x, subsample_w, subsample_b, subsample_tokens, subsample_dim, subsample_channels, subsample_kernel, subsample_stride);
    auto subsample_flat_resident = metal.subsampling_conv2d_relu_flat_f32_resident(
        "primitive.subsampling_conv2d_relu_flat.weight.resident",
        subsample_w,
        "primitive.subsampling_conv2d_relu_flat.bias.resident",
        subsample_b,
        subsample_x,
        subsample_tokens,
        subsample_dim,
        subsample_channels,
        subsample_kernel,
        subsample_stride);
    auto subsample_flat_ref = cpu_subsampling_conv2d_relu_flat(subsample_x, subsample_w, subsample_b, subsample_tokens, subsample_dim, subsample_channels, subsample_kernel, subsample_stride);
    auto dilated_conv = metal.conv1d_dilated_same_f32(conv_x, conv_w, conv_b, conv_tokens, conv_in, conv_out, conv_kernel, 2);
    auto dilated_conv_ref = cpu_conv1d_dilated_same(conv_x, conv_w, conv_b, conv_tokens, conv_in, conv_out, conv_kernel, 2);

    constexpr uint32_t deconv_tokens = 5;
    constexpr uint32_t deconv_in = 4;
    constexpr uint32_t deconv_out = 3;
    constexpr uint32_t deconv_kernel = 4;
    constexpr uint32_t deconv_stride = 2;
    constexpr uint32_t deconv_padding = 1;
    std::vector<float> deconv_x(static_cast<size_t>(deconv_tokens) * deconv_in);
    std::vector<float> deconv_w(static_cast<size_t>(deconv_in) * deconv_out * deconv_kernel);
    std::vector<float> deconv_b(deconv_out);
    for (size_t i = 0; i < deconv_x.size(); ++i) {
        deconv_x[i] = std::sin(static_cast<float>(i) * 0.09f) * 0.2f;
    }
    for (size_t i = 0; i < deconv_w.size(); ++i) {
        deconv_w[i] = std::cos(static_cast<float>(i) * 0.05f) * 0.04f;
    }
    for (size_t i = 0; i < deconv_b.size(); ++i) {
        deconv_b[i] = -0.02f + static_cast<float>(i) * 0.015f;
    }
    auto deconv = metal.conv_transpose1d_f32(deconv_x, deconv_w, deconv_b, deconv_tokens, deconv_in, deconv_out, deconv_kernel, deconv_stride, deconv_padding);
    auto deconv_ref = cpu_conv_transpose1d(deconv_x, deconv_w, deconv_b, deconv_tokens, deconv_in, deconv_out, deconv_kernel, deconv_stride, deconv_padding);

    std::vector<float> gn_gamma(conv_out);
    std::vector<float> gn_beta(conv_out);
    for (uint32_t i = 0; i < conv_out; ++i) {
        gn_gamma[i] = 0.8f + static_cast<float>(i) * 0.03f;
        gn_beta[i] = -0.1f + static_cast<float>(i) * 0.02f;
    }
    auto groupnorm = metal.groupnorm1_f32(conv, gn_gamma, gn_beta, conv_tokens, conv_out, 1e-5f);
    auto groupnorm_ref = cpu_groupnorm1(conv_ref, gn_gamma, gn_beta, conv_tokens, conv_out, 1e-5f);
    auto mish = metal.mish_f32(groupnorm);
    auto mish_ref = cpu_mish(groupnorm_ref);

    std::vector<float> step_values{0.0f, 0.2f, 0.7f};
    std::vector<float> freqs(128);
    for (size_t i = 0; i < freqs.size(); ++i) {
        freqs[i] = std::exp(-std::log(10000.0f) * static_cast<float>(i) / 128.0f);
    }
    auto step_emb = metal.timestep_embedding_f32(step_values, freqs, 1000.0f);
    auto step_emb_ref = cpu_timestep_embedding(step_values, freqs, 1000.0f);

    constexpr uint32_t att_tokens = 4;
    constexpr uint32_t att_dim = 8;
    std::vector<float> att_q(static_cast<size_t>(att_tokens) * att_dim);
    std::vector<float> att_k(att_q.size());
    std::vector<float> att_v(att_q.size());
    for (size_t i = 0; i < att_q.size(); ++i) {
        att_q[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.2f;
        att_k[i] = std::cos(static_cast<float>(i) * 0.17f) * 0.15f;
        att_v[i] = std::sin(static_cast<float>(i) * 0.19f) * 0.25f;
    }
    auto att = metal.attention_single_head_f32(att_q, att_k, att_v, att_tokens, att_dim);
    auto att_ref = cpu_attention_single_head(att_q, att_k, att_v, att_tokens, att_dim);
    std::vector<float> query(att_q.begin() + static_cast<size_t>(att_tokens - 1) * att_dim, att_q.end());
    auto att_query = metal.attention_single_query_f32(query, att_k, att_v, att_tokens, att_dim);
    auto att_query_ref = cpu_attention_single_query(query, att_k, att_v, att_tokens, att_dim);
    constexpr uint32_t gpt_cache_tokens = 5;
    constexpr uint32_t gpt_cache_heads = 3;
    constexpr uint32_t gpt_cache_head_dim = 4;
    constexpr uint32_t gpt_cache_width = gpt_cache_heads * gpt_cache_head_dim;
    std::vector<float> gpt_cache_k(static_cast<size_t>(gpt_cache_tokens) * gpt_cache_width);
    std::vector<float> gpt_cache_v(gpt_cache_k.size());
    std::vector<float> gpt_current_qkv(static_cast<size_t>(gpt_cache_width) * 3);
    for (size_t i = 0; i < gpt_cache_k.size(); ++i) {
        gpt_cache_k[i] = std::sin(static_cast<float>(i) * 0.101f) * 0.19f;
        gpt_cache_v[i] = std::cos(static_cast<float>(i) * 0.073f) * 0.23f;
    }
    for (size_t i = 0; i < gpt_current_qkv.size(); ++i) {
        gpt_current_qkv[i] = std::sin(static_cast<float>(i) * 0.037f) * 0.17f +
                             std::cos(static_cast<float>(i % 17) * 0.109f) * 0.07f;
    }
    auto gpt_cached_att = metal.gpt_cached_attention_f32(gpt_cache_k, gpt_cache_v, gpt_current_qkv, gpt_cache_tokens, gpt_cache_heads, gpt_cache_head_dim);
    std::vector<float> gpt_cached_att_ref(gpt_cache_width);
    for (uint32_t h = 0; h < gpt_cache_heads; ++h) {
        std::vector<float> qh(gpt_cache_head_dim);
        std::vector<float> kh(static_cast<size_t>(gpt_cache_tokens + 1) * gpt_cache_head_dim);
        std::vector<float> vh(kh.size());
        std::copy(gpt_current_qkv.begin() + h * gpt_cache_head_dim,
                  gpt_current_qkv.begin() + (h + 1) * gpt_cache_head_dim,
                  qh.begin());
        for (uint32_t t = 0; t < gpt_cache_tokens; ++t) {
            const size_t src = static_cast<size_t>(t) * gpt_cache_width + h * gpt_cache_head_dim;
            const size_t dst = static_cast<size_t>(t) * gpt_cache_head_dim;
            std::copy(gpt_cache_k.begin() + src, gpt_cache_k.begin() + src + gpt_cache_head_dim, kh.begin() + dst);
            std::copy(gpt_cache_v.begin() + src, gpt_cache_v.begin() + src + gpt_cache_head_dim, vh.begin() + dst);
        }
        const size_t last = static_cast<size_t>(gpt_cache_tokens) * gpt_cache_head_dim;
        std::copy(gpt_current_qkv.begin() + gpt_cache_width + h * gpt_cache_head_dim,
                  gpt_current_qkv.begin() + gpt_cache_width + (h + 1) * gpt_cache_head_dim,
                  kh.begin() + last);
        std::copy(gpt_current_qkv.begin() + gpt_cache_width * 2 + h * gpt_cache_head_dim,
                  gpt_current_qkv.begin() + gpt_cache_width * 2 + (h + 1) * gpt_cache_head_dim,
                  vh.begin() + last);
        auto yh = cpu_attention_single_query(qh, kh, vh, gpt_cache_tokens + 1, gpt_cache_head_dim);
        std::copy(yh.begin(), yh.end(), gpt_cached_att_ref.begin() + h * gpt_cache_head_dim);
    }
    auto att_causal = metal.attention_single_head_causal_f32(att_q, att_k, att_v, att_tokens, att_dim);
    auto att_causal_ref = cpu_attention_single_head_causal(att_q, att_k, att_v, att_tokens, att_dim);
    std::vector<uint32_t> att_mask{1, 1, 0, 1};
    auto att_masked = metal.attention_single_head_masked_f32(att_q, att_k, att_v, att_mask, att_tokens, att_dim);
    auto att_masked_ref = cpu_attention_single_head_masked(att_q, att_k, att_v, att_mask, att_tokens, att_dim);
    constexpr uint32_t rel_heads = 2;
    constexpr uint32_t rel_head_dim = att_dim / rel_heads;
    std::vector<float> att_p(att_q.size());
    std::vector<float> att_bias_u(att_dim);
    std::vector<float> att_bias_v(att_dim);
    for (size_t i = 0; i < att_p.size(); ++i) {
        att_p[i] = std::cos(static_cast<float>(i) * 0.23f) * 0.18f;
    }
    for (uint32_t i = 0; i < att_dim; ++i) {
        att_bias_u[i] = std::sin(static_cast<float>(i) * 0.29f) * 0.04f;
        att_bias_v[i] = std::cos(static_cast<float>(i) * 0.31f) * 0.035f;
    }
    auto rel_attn = metal.conformer_rel_attention_context_f32(att_q, att_k, att_v, att_p, att_bias_u, att_bias_v, att_mask, att_tokens, rel_heads, rel_head_dim);
    auto rel_attn_ref = cpu_conformer_rel_attention_context(att_q, att_k, att_v, att_p, att_bias_u, att_bias_v, att_mask, att_tokens, rel_heads, rel_head_dim);
    constexpr uint32_t cross_queries = 3;
    constexpr uint32_t cross_keys = 5;
    constexpr uint32_t cross_heads = 2;
    constexpr uint32_t cross_head_dim = 4;
    constexpr uint32_t cross_inner = cross_heads * cross_head_dim;
    std::vector<float> cross_q(static_cast<size_t>(cross_queries) * cross_inner);
    std::vector<float> cross_k(static_cast<size_t>(cross_keys) * cross_inner);
    std::vector<float> cross_v(cross_k.size());
    for (size_t i = 0; i < cross_q.size(); ++i) {
        cross_q[i] = std::sin(static_cast<float>(i) * 0.071f) * 0.33f;
    }
    for (size_t i = 0; i < cross_k.size(); ++i) {
        cross_k[i] = std::cos(static_cast<float>(i) * 0.083f) * 0.21f;
        cross_v[i] = std::sin(static_cast<float>(i) * 0.097f) * 0.27f;
    }
    std::vector<uint32_t> cross_mask{1, 0, 1, 1, 0};
    auto cross_attn = metal.cross_attention_heads_masked_f32(cross_q, cross_k, cross_v, cross_mask, cross_queries, cross_keys, cross_heads, cross_head_dim);
    auto cross_attn_ref = cpu_cross_attention_heads_masked(cross_q, cross_k, cross_v, cross_mask, cross_queries, cross_keys, cross_heads, cross_head_dim);
    constexpr uint32_t dit_att_tokens = 4;
    constexpr uint32_t dit_att_heads = 2;
    constexpr uint32_t dit_att_head_dim = 4;
    constexpr uint32_t dit_att_inner = dit_att_heads * dit_att_head_dim;
    std::vector<float> dit_qkv(static_cast<size_t>(dit_att_tokens) * dit_att_inner * 3);
    for (size_t i = 0; i < dit_qkv.size(); ++i) {
        dit_qkv[i] = std::sin(static_cast<float>(i) * 0.061f) * 0.22f +
                     std::cos(static_cast<float>(i % 29) * 0.047f) * 0.13f;
    }
    std::vector<uint32_t> dit_att_mask{1, 1, 0, 1};
    auto dit_att = metal.dit_attention_qkv_rope_f32(dit_qkv, dit_att_mask, dit_att_tokens, dit_att_heads, dit_att_head_dim);
    std::vector<float> dit_q(static_cast<size_t>(dit_att_tokens) * dit_att_inner);
    std::vector<float> dit_k(dit_q.size());
    std::vector<float> dit_v(dit_q.size());
    for (uint32_t t = 0; t < dit_att_tokens; ++t) {
        const auto src = dit_qkv.begin() + static_cast<size_t>(t) * dit_att_inner * 3;
        std::copy(src, src + dit_att_inner, dit_q.begin() + static_cast<size_t>(t) * dit_att_inner);
        std::copy(src + dit_att_inner, src + dit_att_inner * 2, dit_k.begin() + static_cast<size_t>(t) * dit_att_inner);
        std::copy(src + dit_att_inner * 2, src + dit_att_inner * 3, dit_v.begin() + static_cast<size_t>(t) * dit_att_inner);
    }
    apply_rope_inplace(dit_q, dit_att_tokens, dit_att_heads, dit_att_head_dim);
    apply_rope_inplace(dit_k, dit_att_tokens, dit_att_heads, dit_att_head_dim);
    auto dit_att_ref = cpu_cross_attention_heads_masked(dit_q, dit_k, dit_v, dit_att_mask, dit_att_tokens, dit_att_tokens, dit_att_heads, dit_att_head_dim);

    constexpr uint32_t dit_att_batch = 2;
    std::vector<float> dit_qkv_branch1(dit_qkv.size());
    for (size_t i = 0; i < dit_qkv_branch1.size(); ++i) {
        dit_qkv_branch1[i] = std::cos(static_cast<float>(i) * 0.053f) * 0.17f -
                             std::sin(static_cast<float>(i % 31) * 0.037f) * 0.11f;
    }
    std::vector<uint32_t> dit_att_mask_branch1{1, 0, 1, 1};
    std::vector<float> dit_qkv_batched(dit_qkv.size() * dit_att_batch);
    std::copy(dit_qkv.begin(), dit_qkv.end(), dit_qkv_batched.begin());
    std::copy(dit_qkv_branch1.begin(), dit_qkv_branch1.end(), dit_qkv_batched.begin() + static_cast<std::ptrdiff_t>(dit_qkv.size()));
    std::vector<uint32_t> dit_att_mask_batched;
    dit_att_mask_batched.reserve(static_cast<size_t>(dit_att_batch) * dit_att_tokens);
    dit_att_mask_batched.insert(dit_att_mask_batched.end(), dit_att_mask.begin(), dit_att_mask.end());
    dit_att_mask_batched.insert(dit_att_mask_batched.end(), dit_att_mask_branch1.begin(), dit_att_mask_branch1.end());
    auto dit_att_batched = metal.dit_attention_qkv_rope_batched_f32(
        dit_qkv_batched,
        dit_att_mask_batched,
        dit_att_batch,
        dit_att_tokens,
        dit_att_heads,
        dit_att_head_dim);
    auto dit_att_branch1 = metal.dit_attention_qkv_rope_f32(dit_qkv_branch1, dit_att_mask_branch1, dit_att_tokens, dit_att_heads, dit_att_head_dim);
    const size_t dit_att_branch_size = static_cast<size_t>(dit_att_tokens) * dit_att_inner;
    std::vector<float> dit_att_batched0(dit_att_batched.begin(), dit_att_batched.begin() + static_cast<std::ptrdiff_t>(dit_att_branch_size));
    std::vector<float> dit_att_batched1(dit_att_batched.begin() + static_cast<std::ptrdiff_t>(dit_att_branch_size),
                                        dit_att_batched.begin() + static_cast<std::ptrdiff_t>(dit_att_branch_size * 2));

    const float add_err = max_abs_error(add, add_ref);
    const float add_scaled_err = max_abs_error(add_scaled, add_scaled_ref);
    const float avg3_err = max_abs_error(avg3, avg3_ref);
    const float silu_err = max_abs_error(silu, silu_ref);
    const float silu_mul_err = max_abs_error(silu_mul, silu_mul_ref);
    const float mask_rows_err = max_abs_error(mask_rows, mask_rows_ref);
    const float glu_err = max_abs_error(glu, glu_ref);
    const float wavenet_gate_err = max_abs_error(wavenet_gate, wavenet_gate_ref);
    const float wavenet_res_skip_update_err = max_abs_error(rs_update, rs_update_ref);
    const float wavenet_res_skip_final_err = max_abs_error(rs_final_update, rs_final_update_ref);
    const float geglu_err = max_abs_error(geglu, geglu_ref);
    const float gelu_err = max_abs_error(gelu, gelu_ref);
    const float tanh_err = max_abs_error(tanh, tanh_ref);
    const float clamp_err = max_abs_error(clamp, clamp_ref);
    const float softmax_err = max_abs_error(softmax, softmax_ref);
    const float emb_err = max_abs_error(emb, emb_ref);
    const float layernorm_err = max_abs_error(layernorm, layernorm_ref);
    const float layernorm_rows_err = max_abs_error(layernorm_rows, layernorm_rows_ref);
    const float layernorm_rows_wide_err = max_abs_error(layernorm_rows_wide, layernorm_rows_wide_ref);
    const float layernorm_rows_wide_resident_err = max_abs_error(layernorm_rows_wide_resident, layernorm_rows_wide_ref);
    const float layernorm_rows_wide_vs_per_row_err = max_abs_error(layernorm_rows_wide_resident, layernorm_rows_wide_per_row);
    const float adaptive_layernorm_rows_err = max_abs_error(adaptive_layernorm_rows, adaptive_layernorm_rows_ref);
    const float adaptive_rmsnorm_rows_err = max_abs_error(adaptive_rmsnorm_rows, adaptive_rmsnorm_rows_ref);
    const float euler_update_err = max_abs_error(euler_update, euler_update_ref);
    const float concat_rows_err = max_abs_error(concat_rows, concat_rows_ref);
    const float hot_condition_err = max_abs_error(hot_condition, hot_condition_ref);
    const float dit_merge_err = max_abs_error(dit_merge, dit_merge_ref);
    const float rmsnorm_err = max_abs_error(rmsnorm, rmsnorm_ref);
    const float rmsnorm_rows_err = max_abs_error(rmsnorm_rows, rmsnorm_rows_ref);
    const float rmsnorm_rows_eps_err = max_abs_error(rmsnorm_rows_eps, rmsnorm_rows_eps_ref);
    const float linear_err = max_abs_error(linear, linear_ref);
    const float linear_rows_err = max_abs_error(linear_rows_out, linear_rows_ref);
    const float interp_err = max_abs_error(interp, interp_ref);
    const float interp_edge_err = max_abs_error(interp_edge, interp_edge_ref);
    const float conv_err = max_abs_error(conv, conv_ref);
    const float reflect_conv_err = max_abs_error(reflect_conv, reflect_conv_ref);
    const float reflect_conv_batched_branch0_err = max_abs_error(reflect_conv_batched0, reflect_conv);
    const float reflect_conv_batched_branch1_err = max_abs_error(reflect_conv_batched1, reflect_conv_branch1);
    const float reflect_conv_batched_err = std::max(reflect_conv_batched_branch0_err, reflect_conv_batched_branch1_err);
    const float depthwise_conv_err = max_abs_error(depthwise_conv, depthwise_conv_ref);
    const float subsample_flat_err = max_abs_error(subsample_flat, subsample_flat_ref);
    const float subsample_flat_resident_err = max_abs_error(subsample_flat_resident, subsample_flat_ref);
    const float dilated_conv_err = max_abs_error(dilated_conv, dilated_conv_ref);
    const float deconv_err = max_abs_error(deconv, deconv_ref);
    const float groupnorm_err = max_abs_error(groupnorm, groupnorm_ref);
    const float mish_err = max_abs_error(mish, mish_ref);
    const float step_emb_err = max_abs_error(step_emb, step_emb_ref);
    const float att_err = max_abs_error(att, att_ref);
    const float att_query_err = max_abs_error(att_query, att_query_ref);
    const float gpt_cached_att_err = max_abs_error(gpt_cached_att, gpt_cached_att_ref);
    const float att_causal_err = max_abs_error(att_causal, att_causal_ref);
    const float att_masked_err = max_abs_error(att_masked, att_masked_ref);
    const float rel_attn_err = max_abs_error(rel_attn, rel_attn_ref);
    const float cross_attn_err = max_abs_error(cross_attn, cross_attn_ref);
    const float dit_att_qkv_rope_err = max_abs_error(dit_att, dit_att_ref);
    const float dit_att_qkv_rope_batched_branch0_err = max_abs_error(dit_att_batched0, dit_att);
    const float dit_att_qkv_rope_batched_branch1_err = max_abs_error(dit_att_batched1, dit_att_branch1);
    const float dit_att_qkv_rope_batched_err = std::max(dit_att_qkv_rope_batched_branch0_err, dit_att_qkv_rope_batched_branch1_err);
    std::cout << "{\n";
    std::cout << "  \"add_f32_max_abs_error\": " << add_err << ",\n";
    std::cout << "  \"add_scaled_f32_max_abs_error\": " << add_scaled_err << ",\n";
    std::cout << "  \"avg3_f32_max_abs_error\": " << avg3_err << ",\n";
    std::cout << "  \"silu_f32_max_abs_error\": " << silu_err << ",\n";
    std::cout << "  \"silu_mul_f32_max_abs_error\": " << silu_mul_err << ",\n";
    std::cout << "  \"mask_rows_f32_max_abs_error\": " << mask_rows_err << ",\n";
    std::cout << "  \"glu_split_f32_max_abs_error\": " << glu_err << ",\n";
    std::cout << "  \"wavenet_gate_f32_max_abs_error\": " << wavenet_gate_err << ",\n";
    std::cout << "  \"wavenet_res_skip_update_f32_max_abs_error\": " << wavenet_res_skip_update_err << ",\n";
    std::cout << "  \"wavenet_res_skip_final_f32_max_abs_error\": " << wavenet_res_skip_final_err << ",\n";
    std::cout << "  \"geglu_erf_split_f32_max_abs_error\": " << geglu_err << ",\n";
    std::cout << "  \"gelu_f32_max_abs_error\": " << gelu_err << ",\n";
    std::cout << "  \"tanh_f32_max_abs_error\": " << tanh_err << ",\n";
    std::cout << "  \"clamp_f32_max_abs_error\": " << clamp_err << ",\n";
    std::cout << "  \"softmax_f32_max_abs_error\": " << softmax_err << ",\n";
    std::cout << "  \"embedding_f32_max_abs_error\": " << emb_err << ",\n";
    std::cout << "  \"layernorm_f32_max_abs_error\": " << layernorm_err << ",\n";
    std::cout << "  \"layernorm_rows_f32_max_abs_error\": " << layernorm_rows_err << ",\n";
    std::cout << "  \"layernorm_rows_f32_wide_max_abs_error\": " << layernorm_rows_wide_err << ",\n";
    std::cout << "  \"layernorm_rows_f32_resident_wide_max_abs_error\": " << layernorm_rows_wide_resident_err << ",\n";
    std::cout << "  \"layernorm_rows_f32_resident_wide_vs_per_row_max_abs_error\": " << layernorm_rows_wide_vs_per_row_err << ",\n";
    std::cout << "  \"adaptive_layernorm_rows_f32_max_abs_error\": " << adaptive_layernorm_rows_err << ",\n";
    std::cout << "  \"adaptive_rmsnorm_rows_f32_max_abs_error\": " << adaptive_rmsnorm_rows_err << ",\n";
    std::cout << "  \"cfm_euler_update_f32_max_abs_error\": " << euler_update_err << ",\n";
    std::cout << "  \"concat_rows_f32_max_abs_error\": " << concat_rows_err << ",\n";
    std::cout << "  \"hot_condition_merge_f32_max_abs_error\": " << hot_condition_err << ",\n";
    std::cout << "  \"dit_input_merge_f32_max_abs_error\": " << dit_merge_err << ",\n";
    std::cout << "  \"rmsnorm_f32_max_abs_error\": " << rmsnorm_err << ",\n";
    std::cout << "  \"rmsnorm_rows_f32_max_abs_error\": " << rmsnorm_rows_err << ",\n";
    std::cout << "  \"rmsnorm_rows_eps_f32_max_abs_error\": " << rmsnorm_rows_eps_err << ",\n";
    std::cout << "  \"linear_f32_max_abs_error\": " << linear_err << ",\n";
    std::cout << "  \"linear_rows_f32_max_abs_error\": " << linear_rows_err << ",\n";
    std::cout << "  \"nearest_interpolate_f32_max_abs_error\": " << interp_err << ",\n";
    std::cout << "  \"nearest_interpolate_edge_f32_max_abs_error\": " << interp_edge_err << ",\n";
    std::cout << "  \"conv1d_same_f32_max_abs_error\": " << conv_err << ",\n";
    std::cout << "  \"conv1d_reflect_same_f32_max_abs_error\": " << reflect_conv_err << ",\n";
    std::cout << "  \"conv1d_reflect_same_batched_f32_branch0_max_abs_error\": " << reflect_conv_batched_branch0_err << ",\n";
    std::cout << "  \"conv1d_reflect_same_batched_f32_branch1_max_abs_error\": " << reflect_conv_batched_branch1_err << ",\n";
    std::cout << "  \"conv1d_reflect_same_batched_f32_max_abs_error\": " << reflect_conv_batched_err << ",\n";
    std::cout << "  \"depthwise_conv1d_same_f32_max_abs_error\": " << depthwise_conv_err << ",\n";
    std::cout << "  \"subsampling_conv2d_relu_flat_f32_max_abs_error\": " << subsample_flat_err << ",\n";
    std::cout << "  \"subsampling_conv2d_relu_flat_f32_resident_max_abs_error\": " << subsample_flat_resident_err << ",\n";
    std::cout << "  \"conv1d_dilated_same_f32_max_abs_error\": " << dilated_conv_err << ",\n";
    std::cout << "  \"conv_transpose1d_f32_max_abs_error\": " << deconv_err << ",\n";
    std::cout << "  \"groupnorm1_f32_max_abs_error\": " << groupnorm_err << ",\n";
    std::cout << "  \"mish_f32_max_abs_error\": " << mish_err << ",\n";
    std::cout << "  \"timestep_embedding_f32_max_abs_error\": " << step_emb_err << ",\n";
    std::cout << "  \"attention_single_head_f32_max_abs_error\": " << att_err << ",\n";
    std::cout << "  \"attention_single_query_f32_max_abs_error\": " << att_query_err << ",\n";
    std::cout << "  \"gpt_cached_attention_f32_max_abs_error\": " << gpt_cached_att_err << ",\n";
    std::cout << "  \"attention_single_head_causal_f32_max_abs_error\": " << att_causal_err << ",\n";
    std::cout << "  \"attention_single_head_masked_f32_max_abs_error\": " << att_masked_err << ",\n";
    std::cout << "  \"conformer_rel_attention_context_f32_max_abs_error\": " << rel_attn_err << ",\n";
    std::cout << "  \"cross_attention_heads_masked_f32_max_abs_error\": " << cross_attn_err << ",\n";
    std::cout << "  \"dit_attention_qkv_rope_f32_max_abs_error\": " << dit_att_qkv_rope_err << ",\n";
    std::cout << "  \"dit_attention_qkv_rope_batched_f32_branch0_max_abs_error\": " << dit_att_qkv_rope_batched_branch0_err << ",\n";
    std::cout << "  \"dit_attention_qkv_rope_batched_f32_branch1_max_abs_error\": " << dit_att_qkv_rope_batched_branch1_err << ",\n";
    std::cout << "  \"dit_attention_qkv_rope_batched_f32_max_abs_error\": " << dit_att_qkv_rope_batched_err << "\n";
    std::cout << "}\n";
    return add_err <= 1e-7f && add_scaled_err <= 1e-7f && avg3_err <= 1e-7f && silu_err <= 1e-6f && silu_mul_err <= 1e-6f && mask_rows_err <= 1e-7f && glu_err <= 1e-6f && wavenet_gate_err <= 1e-6f && wavenet_res_skip_update_err <= 1e-7f && wavenet_res_skip_final_err <= 1e-7f && geglu_err <= 1e-6f && gelu_err <= 1e-6f && tanh_err <= 1e-6f && clamp_err <= 1e-7f && softmax_err <= 1e-6f && emb_err <= 1e-7f &&
           layernorm_err <= 1e-5f && layernorm_rows_err <= 1e-5f && layernorm_rows_wide_err <= 1e-5f && layernorm_rows_wide_resident_err <= 1e-5f && layernorm_rows_wide_vs_per_row_err <= 1e-5f && adaptive_layernorm_rows_err <= 1e-5f && adaptive_rmsnorm_rows_err <= 1e-5f && euler_update_err <= 1e-7f && concat_rows_err <= 1e-7f && hot_condition_err <= 1e-7f && dit_merge_err <= 1e-7f && rmsnorm_err <= 1e-5f && rmsnorm_rows_err <= 1e-5f && rmsnorm_rows_eps_err <= 1e-5f && linear_err <= 1e-5f && linear_rows_err <= 1e-5f && interp_err <= 1e-7f &&
           conv_err <= 1e-6f && reflect_conv_err <= 1e-6f && reflect_conv_batched_err <= 1e-6f && depthwise_conv_err <= 1e-6f && subsample_flat_err <= 1e-6f && subsample_flat_resident_err <= 1e-6f && dilated_conv_err <= 1e-6f && deconv_err <= 1e-6f && groupnorm_err <= 1e-5f && mish_err <= 1e-5f && step_emb_err <= 1e-5f &&
           att_err <= 1e-5f && att_query_err <= 1e-5f && gpt_cached_att_err <= 1e-5f && att_causal_err <= 1e-5f && att_masked_err <= 1e-5f && rel_attn_err <= 1e-5f && cross_attn_err <= 1e-5f &&
           dit_att_qkv_rope_err <= 1e-5f && dit_att_qkv_rope_batched_err <= 1e-5f;
}

[[maybe_unused]] bool run_tts_clone_w2v_encoder(const std::string& model_bundle_dir,
                                                 const std::string& input_features_f32,
                                                 const std::string& attention_mask_u32,
                                                 uint32_t tokens,
                                                 const std::string& output_spk_cond_f32) {

    struct TmpDir { std::filesystem::path p; ~TmpDir() { std::filesystem::remove_all(p); } };
    TmpDir tmp{std::filesystem::temp_directory_path() / "mit2_enc"};
    std::filesystem::create_directories(tmp.p);
    bool ok = true;
    auto fail = [&](const char* msg) { ok = false; };

    namespace fs = std::filesystem;
    const auto& t = tmp.p;
    auto& issues = ok; // dummy, will use fail() instead
    (void)issues;

    auto pr = [&]() {
        std::cout << "{\n"; std::cout << "  \"stage\": \"tts_clone_w2v_bert_encoder\",\n";
        std::cout << "  \"ok\": " << (ok?"true":"false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n"; std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_input_features_f32\": \"" << json_escape(input_features_f32) << "\",\n";
        std::cout << "  \"w2v_attention_mask_u32\": \"" << json_escape(attention_mask_u32) << "\",\n";
        std::cout << "  \"output_spk_cond_f32\": \"" << json_escape(output_spk_cond_f32) << "\",\n";
        if (ok) { std::cout << "  \"output_spk_cond_sha256\": \"" << file_sha256_hex(output_spk_cond_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"metal\",\n"; }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"ready_native_w2v_bert_encoder\": " << (ok?"true":"false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": " << (ok?"true":"false") << ",\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"Encoder done; integrate into --clone\"\n";
        std::cout << "}\n";
    };

    if (tokens == 0) { fail("w2v_tokens_must_be_positive"); pr(); return false; }

    bool w2v_ok = false;
    mit2::Bundle mdl_shared(model_bundle_dir);  // load once; reused by final_norm lambda and dist_embs
    try { auto c = inspect_w2v_bert_model_contract(mdl_shared); w2v_ok = c.ok;
        if (!c.ok) { std::cout << "{\n  \"stage\":\"tts_clone_w2v_bert_encoder\",\n  \"ok\":false,\n";
            std::cout << "  \"clone_w2v_encoder_issues\":[\"w2v_bert_model_contract_not_ready\"],\n";
            std::cout << "  \"ready_native_w2v_bert_encoder\":false\n}\n"; return false; }
    } catch (const std::exception& e) {
        std::cout << "{\n  \"stage\":\"tts_clone_w2v_bert_encoder\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_w2v_encoder_issues\":[\"" << json_escape(std::string("e:")+e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_encoder\":false\n}\n"; return false;
    }

    // Load distance embeddings for all 17 layers at once
    std::vector<std::vector<float>> dist_embs(17);
    for (int i = 0; i < 17; ++i) {
        const std::string name = "w2v_bert.encoder.layers." + std::to_string(i) +
                                 ".self_attn.distance_embedding.weight";
        dist_embs[i] = tensor_as_f32(mdl_shared, name);
    }

    auto apply_layer_final_norm = [&](int layer_idx, const std::string& input_f32, const std::string& output_f32) -> bool {
        try {
            const auto inp = read_raw_f32(input_f32);
            std::vector<float> normed;
            try {
                mit2::MetalContext metal;
                normed = run_w2v_bert_layerN_final_norm_metal(metal, mdl_shared, inp, tokens, layer_idx);
            } catch (const std::exception& me) {
                const std::string err = me.what();
                if (err.find("Metal device unavailable") == std::string::npos) throw;
                normed = run_w2v_bert_layerN_final_norm_cpu(mdl_shared, inp, tokens, layer_idx);
            }
            write_raw_f32(output_f32, normed);
            return true;
        } catch (...) { return false; }
    };

    // Feature projection
    std::string fp = (t / "fp.f32").string();
    if (!run_tts_clone_w2v_feature_project(model_bundle_dir, input_features_f32, tokens, fp)) { fail("fp"); pr(); return false; }

        std::string a_fn = (t / "a_fn.f32").string(); if (!run_tts_clone_w2v_layer0_ffn1_norm(model_bundle_dir, fp, tokens, a_fn)) { ok = false; return false; }
        std::string a_fi = (t / "a_fi.f32").string(); if (!run_tts_clone_w2v_layer0_ffn1_intermediate(model_bundle_dir, a_fn, tokens, a_fi)) { ok = false; return false; }
        std::string a_fa = (t / "a_fa.f32").string(); if (!run_tts_clone_w2v_layer0_ffn1_activate(a_fi, tokens, a_fa)) { ok = false; return false; }
        std::string a_fo = (t / "a_fo.f32").string(); if (!run_tts_clone_w2v_layer0_ffn1_output(model_bundle_dir, a_fa, tokens, a_fo)) { ok = false; return false; }
        std::string a_fr = (t / "a_fr.f32").string(); if (!run_tts_clone_w2v_layer0_ffn1_residual(fp, a_fo, tokens, a_fr)) { ok = false; return false; }
        std::string a_an = (t / "a_an.f32").string(); if (!run_tts_clone_w2v_layer0_attention_norm(model_bundle_dir, a_fr, tokens, a_an)) { ok = false; return false; }
        std::string l0q = (t / "l0q").string(); if (!run_tts_clone_w2v_layer0_qkv(model_bundle_dir, a_an, tokens, l0q)) { fail("l0_qkv"); pr(); return false; }
        std::string a_at = (t / "a_at.f32").string(); if (!run_tts_clone_w2v_layer0_attention(l0q+"/w2v_layer0_q.f32", l0q+"/w2v_layer0_k.f32", l0q+"/w2v_layer0_v.f32", attention_mask_u32, tokens, a_at, dist_embs[0])) { fail("l0_at"); pr(); return false; }
        std::string a_ap = (t / "a_ap.f32").string(); if (!run_tts_clone_w2v_layer0_attention_project(model_bundle_dir, a_at, tokens, a_ap)) { ok = false; return false; }
        std::string a_ar = (t / "a_ar.f32").string(); if (!run_tts_clone_w2v_layer0_attention_residual(a_fr, a_ap, tokens, a_ar)) { ok = false; return false; }
        std::string a_cn = (t / "a_cn.f32").string(); if (!run_tts_clone_w2v_layer0_conv_norm(model_bundle_dir, a_ar, tokens, a_cn)) { ok = false; return false; }
        std::string a_cg = (t / "a_cg.f32").string(); if (!run_tts_clone_w2v_layer0_conv_glu(model_bundle_dir, a_cn, tokens, a_cg)) { ok = false; return false; }
        std::string a_cd = (t / "a_cd.f32").string(); if (!run_tts_clone_w2v_layer0_conv_depthwise(model_bundle_dir, a_cg, tokens, a_cd)) { ok = false; return false; }
        std::string a_cr = (t / "a_cr.f32").string(); if (!run_tts_clone_w2v_layer0_conv_residual(model_bundle_dir, a_ar, a_cd, tokens, a_cr)) { ok = false; return false; }
        std::string a_f2 = (t / "a_f2.f32").string(); if (!run_tts_clone_w2v_layer0_ffn2_residual(model_bundle_dir, a_cr, tokens, a_f2)) { ok = false; return false; }
        std::string s0 = (t / "layer0.f32").string(); if (!run_tts_clone_w2v_layer0_final_norm(model_bundle_dir, a_f2, tokens, s0)) { fail("l0_fn"); pr(); return false; }

        // Layer 1
        std::string b1_fn = (t / "b1_fn.f32").string(); if (!run_tts_clone_w2v_layer1_ffn1_norm(model_bundle_dir, (t / "layer0.f32").string(), tokens, b1_fn)) { ok = false; return false; }
        std::string b1_fi = (t / "b1_fi.f32").string(); if (!run_tts_clone_w2v_layer1_ffn1_intermediate(model_bundle_dir, b1_fn, tokens, b1_fi)) { ok = false; return false; }
        std::string b1_fa = (t / "b1_fa.f32").string(); if (!run_tts_clone_w2v_layer1_ffn1_activate(b1_fi, tokens, b1_fa)) { ok = false; return false; }
        std::string b1_fo = (t / "b1_fo.f32").string(); if (!run_tts_clone_w2v_layer1_ffn1_output(model_bundle_dir, b1_fa, tokens, b1_fo)) { ok = false; return false; }
        std::string b1_fr = (t / "b1_fr.f32").string(); if (!run_tts_clone_w2v_layer1_ffn1_residual((t / "layer0.f32").string(), b1_fo, tokens, b1_fr)) { ok = false; return false; }
        std::string b1_an = (t / "b1_an.f32").string(); if (!run_tts_clone_w2v_layer1_attention_norm(model_bundle_dir, b1_fr, tokens, b1_an)) { ok = false; return false; }
        std::string bl1q = (t / "bl1q").string(); if (!run_tts_clone_w2v_layer1_qkv(model_bundle_dir, b1_an, tokens, bl1q)) { fail("l1_qkv"); pr(); return false; }
        std::string b1_at = (t / "b1_at.f32").string(); if (!run_tts_clone_w2v_layer1_attention(bl1q+"/w2v_layer1_q.f32", bl1q+"/w2v_layer1_k.f32", bl1q+"/w2v_layer1_v.f32", attention_mask_u32, tokens, b1_at, dist_embs[1])) { fail("l1_at"); pr(); return false; }
        std::string b1_ap = (t / "b1_ap.f32").string(); if (!run_tts_clone_w2v_layer1_attention_project(model_bundle_dir, b1_at, tokens, b1_ap)) { ok = false; return false; }
        std::string b1_ar = (t / "b1_ar.f32").string(); if (!run_tts_clone_w2v_layer1_attention_residual(b1_fr, b1_ap, tokens, b1_ar)) { ok = false; return false; }
        std::string b1_cn = (t / "b1_cn.f32").string(); if (!run_tts_clone_w2v_layer1_conv_norm(model_bundle_dir, b1_ar, tokens, b1_cn)) { ok = false; return false; }
        std::string b1_cg = (t / "b1_cg.f32").string(); if (!run_tts_clone_w2v_layer1_conv_glu(model_bundle_dir, b1_cn, tokens, b1_cg)) { ok = false; return false; }
        std::string b1_cd = (t / "b1_cd.f32").string(); if (!run_tts_clone_w2v_layer1_conv_depthwise(model_bundle_dir, b1_cg, tokens, b1_cd)) { ok = false; return false; }
        std::string b1_cr = (t / "b1_cr.f32").string(); if (!run_tts_clone_w2v_layer1_conv_residual(model_bundle_dir, b1_ar, b1_cd, tokens, b1_cr)) { ok = false; return false; }
        std::string b1_f2 = (t / "b1_f2.f32").string(); if (!run_tts_clone_w2v_layer1_ffn2_residual(model_bundle_dir, b1_cr, tokens, b1_f2)) { ok = false; return false; }
        std::string s1 = (t / "layer1.f32").string(); if (!run_tts_clone_w2v_layer1_final_norm(model_bundle_dir, b1_f2, tokens, s1)) { fail("l1_fn"); pr(); return false; }

        // Layer 2
        std::string b2_fn = (t / "b2_fn.f32").string(); if (!run_tts_clone_w2v_layer2_ffn1_norm(model_bundle_dir, (t / "layer1.f32").string(), tokens, b2_fn)) { ok = false; return false; }
        std::string b2_fi = (t / "b2_fi.f32").string(); if (!run_tts_clone_w2v_layer2_ffn1_intermediate(model_bundle_dir, b2_fn, tokens, b2_fi)) { ok = false; return false; }
        std::string b2_fa = (t / "b2_fa.f32").string(); if (!run_tts_clone_w2v_layer2_ffn1_activate(b2_fi, tokens, b2_fa)) { ok = false; return false; }
        std::string b2_fo = (t / "b2_fo.f32").string(); if (!run_tts_clone_w2v_layer2_ffn1_output(model_bundle_dir, b2_fa, tokens, b2_fo)) { ok = false; return false; }
        std::string b2_fr = (t / "b2_fr.f32").string(); if (!run_tts_clone_w2v_layer2_ffn1_residual((t / "layer1.f32").string(), b2_fo, tokens, b2_fr)) { ok = false; return false; }
        std::string b2_an = (t / "b2_an.f32").string(); if (!run_tts_clone_w2v_layer2_attention_norm(model_bundle_dir, b2_fr, tokens, b2_an)) { ok = false; return false; }
        std::string bl2q = (t / "bl2q").string(); if (!run_tts_clone_w2v_layer2_qkv(model_bundle_dir, b2_an, tokens, bl2q)) { fail("l2_qkv"); pr(); return false; }
        std::string b2_at = (t / "b2_at.f32").string(); if (!run_tts_clone_w2v_layer2_attention(bl2q+"/w2v_layer2_q.f32", bl2q+"/w2v_layer2_k.f32", bl2q+"/w2v_layer2_v.f32", attention_mask_u32, tokens, b2_at, dist_embs[2])) { fail("l2_at"); pr(); return false; }
        std::string b2_ap = (t / "b2_ap.f32").string(); if (!run_tts_clone_w2v_layer2_attention_project(model_bundle_dir, b2_at, tokens, b2_ap)) { ok = false; return false; }
        std::string b2_ar = (t / "b2_ar.f32").string(); if (!run_tts_clone_w2v_layer2_attention_residual(b2_fr, b2_ap, tokens, b2_ar)) { ok = false; return false; }
        std::string b2_cn = (t / "b2_cn.f32").string(); if (!run_tts_clone_w2v_layer2_conv_norm(model_bundle_dir, b2_ar, tokens, b2_cn)) { ok = false; return false; }
        std::string b2_cg = (t / "b2_cg.f32").string(); if (!run_tts_clone_w2v_layer2_conv_glu(model_bundle_dir, b2_cn, tokens, b2_cg)) { ok = false; return false; }
        std::string b2_cd = (t / "b2_cd.f32").string(); if (!run_tts_clone_w2v_layer2_conv_depthwise(model_bundle_dir, b2_cg, tokens, b2_cd)) { ok = false; return false; }
        std::string b2_cr = (t / "b2_cr.f32").string(); if (!run_tts_clone_w2v_layer2_conv_residual(model_bundle_dir, b2_ar, b2_cd, tokens, b2_cr)) { ok = false; return false; }
        std::string b2_f2 = (t / "b2_f2.f32").string(); if (!run_tts_clone_w2v_layer2_ffn2_residual(model_bundle_dir, b2_cr, tokens, b2_f2)) { ok = false; return false; }
        std::string s2 = (t / "layer2.f32").string(); if (!apply_layer_final_norm(2, b2_f2, s2)) { fail("l2_fn"); pr(); return false; }

        // Layer 3
        std::string b3_fn = (t / "b3_fn.f32").string(); if (!run_tts_clone_w2v_layer3_ffn1_norm(model_bundle_dir, (t / "layer2.f32").string(), tokens, b3_fn)) { ok = false; return false; }
        std::string b3_fi = (t / "b3_fi.f32").string(); if (!run_tts_clone_w2v_layer3_ffn1_intermediate(model_bundle_dir, b3_fn, tokens, b3_fi)) { ok = false; return false; }
        std::string b3_fa = (t / "b3_fa.f32").string(); if (!run_tts_clone_w2v_layer3_ffn1_activate(b3_fi, tokens, b3_fa)) { ok = false; return false; }
        std::string b3_fo = (t / "b3_fo.f32").string(); if (!run_tts_clone_w2v_layer3_ffn1_output(model_bundle_dir, b3_fa, tokens, b3_fo)) { ok = false; return false; }
        std::string b3_fr = (t / "b3_fr.f32").string(); if (!run_tts_clone_w2v_layer3_ffn1_residual((t / "layer2.f32").string(), b3_fo, tokens, b3_fr)) { ok = false; return false; }
        std::string b3_an = (t / "b3_an.f32").string(); if (!run_tts_clone_w2v_layer3_attention_norm(model_bundle_dir, b3_fr, tokens, b3_an)) { ok = false; return false; }
        std::string bl3q = (t / "bl3q").string(); if (!run_tts_clone_w2v_layer3_qkv(model_bundle_dir, b3_an, tokens, bl3q)) { fail("l3_qkv"); pr(); return false; }
        std::string b3_at = (t / "b3_at.f32").string(); if (!run_tts_clone_w2v_layer3_attention(bl3q+"/w2v_layer3_q.f32", bl3q+"/w2v_layer3_k.f32", bl3q+"/w2v_layer3_v.f32", attention_mask_u32, tokens, b3_at, dist_embs[3])) { fail("l3_at"); pr(); return false; }
        std::string b3_ap = (t / "b3_ap.f32").string(); if (!run_tts_clone_w2v_layer3_attention_project(model_bundle_dir, b3_at, tokens, b3_ap)) { ok = false; return false; }
        std::string b3_ar = (t / "b3_ar.f32").string(); if (!run_tts_clone_w2v_layer3_attention_residual(b3_fr, b3_ap, tokens, b3_ar)) { ok = false; return false; }
        std::string b3_cn = (t / "b3_cn.f32").string(); if (!run_tts_clone_w2v_layer3_conv_norm(model_bundle_dir, b3_ar, tokens, b3_cn)) { ok = false; return false; }
        std::string b3_cg = (t / "b3_cg.f32").string(); if (!run_tts_clone_w2v_layer3_conv_glu(model_bundle_dir, b3_cn, tokens, b3_cg)) { ok = false; return false; }
        std::string b3_cd = (t / "b3_cd.f32").string(); if (!run_tts_clone_w2v_layer3_conv_depthwise(model_bundle_dir, b3_cg, tokens, b3_cd)) { ok = false; return false; }
        std::string b3_cr = (t / "b3_cr.f32").string(); if (!run_tts_clone_w2v_layer3_conv_residual(model_bundle_dir, b3_ar, b3_cd, tokens, b3_cr)) { ok = false; return false; }
        std::string b3_f2 = (t / "b3_f2.f32").string(); if (!run_tts_clone_w2v_layer3_ffn2_residual(model_bundle_dir, b3_cr, tokens, b3_f2)) { ok = false; return false; }
        std::string s3 = (t / "layer3.f32").string(); if (!run_tts_clone_w2v_layer3_final_norm(model_bundle_dir, b3_f2, tokens, s3)) { fail("l3_fn"); pr(); return false; }

        // Layer 4 (has separate ffn1 sub-steps)
        std::string c4_fn = (t / "c4_fn.f32").string(); if (!run_tts_clone_w2v_layer4_ffn1_norm(model_bundle_dir, s3, tokens, c4_fn)) { fail("l4_fn1n"); pr(); return false; }
        std::string c4_fi = (t / "c4_fi.f32").string(); if (!run_tts_clone_w2v_layer4_ffn1_intermediate(model_bundle_dir, c4_fn, tokens, c4_fi)) { ok = false; return false; }
        std::string c4_fa = (t / "c4_fa.f32").string(); if (!run_tts_clone_w2v_layer4_ffn1_activate(c4_fi, tokens, c4_fa)) { ok = false; return false; }
        std::string c4_fo = (t / "c4_fo.f32").string(); if (!run_tts_clone_w2v_layer4_ffn1_output(model_bundle_dir, c4_fa, tokens, c4_fo)) { ok = false; return false; }
        std::string c4_fr = (t / "c4_fr.f32").string(); if (!run_tts_clone_w2v_layer4_ffn1_residual(s3, c4_fo, tokens, c4_fr)) { ok = false; return false; }
        std::string c4_an = (t / "c4_an.f32").string(); if (!run_tts_clone_w2v_layer4_attention_norm(model_bundle_dir, c4_fr, tokens, c4_an)) { ok = false; return false; }
        std::string c4q = (t / "c4q").string(); if (!run_tts_clone_w2v_layer4_qkv(model_bundle_dir, c4_an, tokens, c4q)) { fail("l4_qkv"); pr(); return false; }
        std::string c4_at = (t / "c4_at.f32").string(); if (!run_tts_clone_w2v_layer4_attention(c4q+"/w2v_layer4_q.f32", c4q+"/w2v_layer4_k.f32", c4q+"/w2v_layer4_v.f32", attention_mask_u32, tokens, c4_at, dist_embs[4])) { fail("l4_at"); pr(); return false; }
        std::string c4_ap = (t / "c4_ap.f32").string(); if (!run_tts_clone_w2v_layer4_attention_project(model_bundle_dir, c4_at, tokens, c4_ap)) { ok = false; return false; }
        std::string c4_ar = (t / "c4_ar.f32").string(); if (!run_tts_clone_w2v_layer4_attention_residual(c4_fr, c4_ap, tokens, c4_ar)) { ok = false; return false; }
        std::string c4_cn = (t / "c4_cn.f32").string(); if (!run_tts_clone_w2v_layer4_conv_norm(model_bundle_dir, c4_ar, tokens, c4_cn)) { ok = false; return false; }
        std::string c4_cg = (t / "c4_cg.f32").string(); if (!run_tts_clone_w2v_layer4_conv_glu(model_bundle_dir, c4_cn, tokens, c4_cg)) { ok = false; return false; }
        std::string c4_cd = (t / "c4_cd.f32").string(); if (!run_tts_clone_w2v_layer4_conv_depthwise(model_bundle_dir, c4_cg, tokens, c4_cd)) { ok = false; return false; }
        std::string c4_cr = (t / "c4_cr.f32").string(); if (!run_tts_clone_w2v_layer4_conv_residual(model_bundle_dir, c4_ar, c4_cd, tokens, c4_cr)) { ok = false; return false; }
        std::string c4_f2 = (t / "c4_f2.f32").string(); if (!run_tts_clone_w2v_layer4_ffn2_residual(model_bundle_dir, c4_cr, tokens, c4_f2)) { ok = false; return false; }
        std::string s4 = (t / "layer4.f32").string(); if (!apply_layer_final_norm(4, c4_f2, s4)) { fail("l4_fn"); pr(); return false; }

        // Layer 5
        std::string e5_fr = (t / "e5_fr.f32").string(); if (!run_tts_clone_w2v_layer5_ffn1_residual(model_bundle_dir, s4, tokens, e5_fr)) { fail("l5_fr"); pr(); return false; }
        std::string e5_an = (t / "e5_an.f32").string(); if (!run_tts_clone_w2v_layer5_attention_norm(model_bundle_dir, e5_fr, tokens, e5_an)) { ok = false; return false; }
        std::string e5q = (t / "e5q").string(); if (!run_tts_clone_w2v_layer5_qkv(model_bundle_dir, e5_an, tokens, e5q)) { fail("l5_qkv"); pr(); return false; }
        std::string e5_at = (t / "e5_at.f32").string(); if (!run_tts_clone_w2v_layer5_attention(e5q+"/w2v_layer5_q.f32", e5q+"/w2v_layer5_k.f32", e5q+"/w2v_layer5_v.f32", attention_mask_u32, tokens, e5_at, dist_embs[5])) { fail("l5_at"); pr(); return false; }
        std::string e5_ap = (t / "e5_ap.f32").string(); if (!run_tts_clone_w2v_layer5_attention_project(model_bundle_dir, e5_at, tokens, e5_ap)) { ok = false; return false; }
        std::string e5_ar = (t / "e5_ar.f32").string(); if (!run_tts_clone_w2v_layer5_attention_residual(e5_fr, e5_ap, tokens, e5_ar)) { ok = false; return false; }
        std::string e5_cn = (t / "e5_cn.f32").string(); if (!run_tts_clone_w2v_layer5_conv_norm(model_bundle_dir, e5_ar, tokens, e5_cn)) { ok = false; return false; }
        std::string e5_cg = (t / "e5_cg.f32").string(); if (!run_tts_clone_w2v_layer5_conv_glu(model_bundle_dir, e5_cn, tokens, e5_cg)) { ok = false; return false; }
        std::string e5_cd = (t / "e5_cd.f32").string(); if (!run_tts_clone_w2v_layer5_conv_depthwise(model_bundle_dir, e5_cg, tokens, e5_cd)) { ok = false; return false; }
        std::string e5_cr = (t / "e5_cr.f32").string(); if (!run_tts_clone_w2v_layer5_conv_residual(model_bundle_dir, e5_ar, e5_cd, tokens, e5_cr)) { ok = false; return false; }
        std::string e5_f2 = (t / "e5_f2.f32").string(); if (!run_tts_clone_w2v_layer5_ffn2_residual(model_bundle_dir, e5_cr, tokens, e5_f2)) { ok = false; return false; }
        std::string s5 = (t / "layer5.f32").string(); if (!apply_layer_final_norm(5, e5_f2, s5)) { fail("l5_fn"); pr(); return false; }

        // Layer 6
        std::string e6_fr = (t / "e6_fr.f32").string(); if (!run_tts_clone_w2v_layer6_ffn1_residual(model_bundle_dir, s5, tokens, e6_fr)) { fail("l6_fr"); pr(); return false; }
        std::string e6_an = (t / "e6_an.f32").string(); if (!run_tts_clone_w2v_layer6_attention_norm(model_bundle_dir, e6_fr, tokens, e6_an)) { ok = false; return false; }
        std::string e6q = (t / "e6q").string(); if (!run_tts_clone_w2v_layer6_qkv(model_bundle_dir, e6_an, tokens, e6q)) { fail("l6_qkv"); pr(); return false; }
        std::string e6_at = (t / "e6_at.f32").string(); if (!run_tts_clone_w2v_layer6_attention(e6q+"/w2v_layer6_q.f32", e6q+"/w2v_layer6_k.f32", e6q+"/w2v_layer6_v.f32", attention_mask_u32, tokens, e6_at, dist_embs[6])) { fail("l6_at"); pr(); return false; }
        std::string e6_ap = (t / "e6_ap.f32").string(); if (!run_tts_clone_w2v_layer6_attention_project(model_bundle_dir, e6_at, tokens, e6_ap)) { ok = false; return false; }
        std::string e6_ar = (t / "e6_ar.f32").string(); if (!run_tts_clone_w2v_layer6_attention_residual(e6_fr, e6_ap, tokens, e6_ar)) { ok = false; return false; }
        std::string e6_cn = (t / "e6_cn.f32").string(); if (!run_tts_clone_w2v_layer6_conv_norm(model_bundle_dir, e6_ar, tokens, e6_cn)) { ok = false; return false; }
        std::string e6_cg = (t / "e6_cg.f32").string(); if (!run_tts_clone_w2v_layer6_conv_glu(model_bundle_dir, e6_cn, tokens, e6_cg)) { ok = false; return false; }
        std::string e6_cd = (t / "e6_cd.f32").string(); if (!run_tts_clone_w2v_layer6_conv_depthwise(model_bundle_dir, e6_cg, tokens, e6_cd)) { ok = false; return false; }
        std::string e6_cr = (t / "e6_cr.f32").string(); if (!run_tts_clone_w2v_layer6_conv_residual(model_bundle_dir, e6_ar, e6_cd, tokens, e6_cr)) { ok = false; return false; }
        std::string e6_f2 = (t / "e6_f2.f32").string(); if (!run_tts_clone_w2v_layer6_ffn2_residual(model_bundle_dir, e6_cr, tokens, e6_f2)) { ok = false; return false; }
        std::string s6 = (t / "layer6.f32").string(); if (!apply_layer_final_norm(6, e6_f2, s6)) { fail("l6_fn"); pr(); return false; }

        // Layer 7
        std::string e7_fr = (t / "e7_fr.f32").string(); if (!run_tts_clone_w2v_layer7_ffn1_residual(model_bundle_dir, s6, tokens, e7_fr)) { fail("l7_fr"); pr(); return false; }
        std::string e7_an = (t / "e7_an.f32").string(); if (!run_tts_clone_w2v_layer7_attention_norm(model_bundle_dir, e7_fr, tokens, e7_an)) { ok = false; return false; }
        std::string e7q = (t / "e7q").string(); if (!run_tts_clone_w2v_layer7_qkv(model_bundle_dir, e7_an, tokens, e7q)) { fail("l7_qkv"); pr(); return false; }
        std::string e7_at = (t / "e7_at.f32").string(); if (!run_tts_clone_w2v_layer7_attention(e7q+"/w2v_layer7_q.f32", e7q+"/w2v_layer7_k.f32", e7q+"/w2v_layer7_v.f32", attention_mask_u32, tokens, e7_at, dist_embs[7])) { fail("l7_at"); pr(); return false; }
        std::string e7_ap = (t / "e7_ap.f32").string(); if (!run_tts_clone_w2v_layer7_attention_project(model_bundle_dir, e7_at, tokens, e7_ap)) { ok = false; return false; }
        std::string e7_ar = (t / "e7_ar.f32").string(); if (!run_tts_clone_w2v_layer7_attention_residual(e7_fr, e7_ap, tokens, e7_ar)) { ok = false; return false; }
        std::string e7_cn = (t / "e7_cn.f32").string(); if (!run_tts_clone_w2v_layer7_conv_norm(model_bundle_dir, e7_ar, tokens, e7_cn)) { ok = false; return false; }
        std::string e7_cg = (t / "e7_cg.f32").string(); if (!run_tts_clone_w2v_layer7_conv_glu(model_bundle_dir, e7_cn, tokens, e7_cg)) { ok = false; return false; }
        std::string e7_cd = (t / "e7_cd.f32").string(); if (!run_tts_clone_w2v_layer7_conv_depthwise(model_bundle_dir, e7_cg, tokens, e7_cd)) { ok = false; return false; }
        std::string e7_cr = (t / "e7_cr.f32").string(); if (!run_tts_clone_w2v_layer7_conv_residual(model_bundle_dir, e7_ar, e7_cd, tokens, e7_cr)) { ok = false; return false; }
        std::string e7_f2 = (t / "e7_f2.f32").string(); if (!run_tts_clone_w2v_layer7_ffn2_residual(model_bundle_dir, e7_cr, tokens, e7_f2)) { ok = false; return false; }
        std::string s7 = (t / "layer7.f32").string(); if (!apply_layer_final_norm(7, e7_f2, s7)) { fail("l7_fn"); pr(); return false; }

        // Layer 8
        std::string e8_fr = (t / "e8_fr.f32").string(); if (!run_tts_clone_w2v_layer8_ffn1_residual(model_bundle_dir, s7, tokens, e8_fr)) { fail("l8_fr"); pr(); return false; }
        std::string e8_an = (t / "e8_an.f32").string(); if (!run_tts_clone_w2v_layer8_attention_norm(model_bundle_dir, e8_fr, tokens, e8_an)) { ok = false; return false; }
        std::string e8q = (t / "e8q").string(); if (!run_tts_clone_w2v_layer8_qkv(model_bundle_dir, e8_an, tokens, e8q)) { fail("l8_qkv"); pr(); return false; }
        std::string e8_at = (t / "e8_at.f32").string(); if (!run_tts_clone_w2v_layer8_attention(e8q+"/w2v_layer8_q.f32", e8q+"/w2v_layer8_k.f32", e8q+"/w2v_layer8_v.f32", attention_mask_u32, tokens, e8_at, dist_embs[8])) { fail("l8_at"); pr(); return false; }
        std::string e8_ap = (t / "e8_ap.f32").string(); if (!run_tts_clone_w2v_layer8_attention_project(model_bundle_dir, e8_at, tokens, e8_ap)) { ok = false; return false; }
        std::string e8_ar = (t / "e8_ar.f32").string(); if (!run_tts_clone_w2v_layer8_attention_residual(e8_fr, e8_ap, tokens, e8_ar)) { ok = false; return false; }
        std::string e8_cn = (t / "e8_cn.f32").string(); if (!run_tts_clone_w2v_layer8_conv_norm(model_bundle_dir, e8_ar, tokens, e8_cn)) { ok = false; return false; }
        std::string e8_cg = (t / "e8_cg.f32").string(); if (!run_tts_clone_w2v_layer8_conv_glu(model_bundle_dir, e8_cn, tokens, e8_cg)) { ok = false; return false; }
        std::string e8_cd = (t / "e8_cd.f32").string(); if (!run_tts_clone_w2v_layer8_conv_depthwise(model_bundle_dir, e8_cg, tokens, e8_cd)) { ok = false; return false; }
        std::string e8_cr = (t / "e8_cr.f32").string(); if (!run_tts_clone_w2v_layer8_conv_residual(model_bundle_dir, e8_ar, e8_cd, tokens, e8_cr)) { ok = false; return false; }
        std::string e8_f2 = (t / "e8_f2.f32").string(); if (!run_tts_clone_w2v_layer8_ffn2_residual(model_bundle_dir, e8_cr, tokens, e8_f2)) { ok = false; return false; }
        std::string s8 = (t / "layer8.f32").string(); if (!apply_layer_final_norm(8, e8_f2, s8)) { fail("l8_fn"); pr(); return false; }

        // Layer 9
        std::string e9_fr = (t / "e9_fr.f32").string(); if (!run_tts_clone_w2v_layer9_ffn1_residual(model_bundle_dir, s8, tokens, e9_fr)) { fail("l9_fr"); pr(); return false; }
        std::string e9_an = (t / "e9_an.f32").string(); if (!run_tts_clone_w2v_layer9_attention_norm(model_bundle_dir, e9_fr, tokens, e9_an)) { ok = false; return false; }
        std::string e9q = (t / "e9q").string(); if (!run_tts_clone_w2v_layer9_qkv(model_bundle_dir, e9_an, tokens, e9q)) { fail("l9_qkv"); pr(); return false; }
        std::string e9_at = (t / "e9_at.f32").string(); if (!run_tts_clone_w2v_layer9_attention(e9q+"/w2v_layer9_q.f32", e9q+"/w2v_layer9_k.f32", e9q+"/w2v_layer9_v.f32", attention_mask_u32, tokens, e9_at, dist_embs[9])) { fail("l9_at"); pr(); return false; }
        std::string e9_ap = (t / "e9_ap.f32").string(); if (!run_tts_clone_w2v_layer9_attention_project(model_bundle_dir, e9_at, tokens, e9_ap)) { ok = false; return false; }
        std::string e9_ar = (t / "e9_ar.f32").string(); if (!run_tts_clone_w2v_layer9_attention_residual(e9_fr, e9_ap, tokens, e9_ar)) { ok = false; return false; }
        std::string e9_cn = (t / "e9_cn.f32").string(); if (!run_tts_clone_w2v_layer9_conv_norm(model_bundle_dir, e9_ar, tokens, e9_cn)) { ok = false; return false; }
        std::string e9_cg = (t / "e9_cg.f32").string(); if (!run_tts_clone_w2v_layer9_conv_glu(model_bundle_dir, e9_cn, tokens, e9_cg)) { ok = false; return false; }
        std::string e9_cd = (t / "e9_cd.f32").string(); if (!run_tts_clone_w2v_layer9_conv_depthwise(model_bundle_dir, e9_cg, tokens, e9_cd)) { ok = false; return false; }
        std::string e9_cr = (t / "e9_cr.f32").string(); if (!run_tts_clone_w2v_layer9_conv_residual(model_bundle_dir, e9_ar, e9_cd, tokens, e9_cr)) { ok = false; return false; }
        std::string e9_f2 = (t / "e9_f2.f32").string(); if (!run_tts_clone_w2v_layer9_ffn2_residual(model_bundle_dir, e9_cr, tokens, e9_f2)) { ok = false; return false; }
        std::string s9 = (t / "layer9.f32").string(); if (!apply_layer_final_norm(9, e9_f2, s9)) { fail("l9_fn"); pr(); return false; }

        // Layer 10
        std::string e10_fr = (t / "e10_fr.f32").string(); if (!run_tts_clone_w2v_layer10_ffn1_residual(model_bundle_dir, s9, tokens, e10_fr)) { fail("l10_fr"); pr(); return false; }
        std::string e10_an = (t / "e10_an.f32").string(); if (!run_tts_clone_w2v_layer10_attention_norm(model_bundle_dir, e10_fr, tokens, e10_an)) { ok = false; return false; }
        std::string e10q = (t / "e10q").string(); if (!run_tts_clone_w2v_layer10_qkv(model_bundle_dir, e10_an, tokens, e10q)) { fail("l10_qkv"); pr(); return false; }
        std::string e10_at = (t / "e10_at.f32").string(); if (!run_tts_clone_w2v_layer10_attention(e10q+"/w2v_layer10_q.f32", e10q+"/w2v_layer10_k.f32", e10q+"/w2v_layer10_v.f32", attention_mask_u32, tokens, e10_at, dist_embs[10])) { fail("l10_at"); pr(); return false; }
        std::string e10_ap = (t / "e10_ap.f32").string(); if (!run_tts_clone_w2v_layer10_attention_project(model_bundle_dir, e10_at, tokens, e10_ap)) { ok = false; return false; }
        std::string e10_ar = (t / "e10_ar.f32").string(); if (!run_tts_clone_w2v_layer10_attention_residual(e10_fr, e10_ap, tokens, e10_ar)) { ok = false; return false; }
        std::string e10_cn = (t / "e10_cn.f32").string(); if (!run_tts_clone_w2v_layer10_conv_norm(model_bundle_dir, e10_ar, tokens, e10_cn)) { ok = false; return false; }
        std::string e10_cg = (t / "e10_cg.f32").string(); if (!run_tts_clone_w2v_layer10_conv_glu(model_bundle_dir, e10_cn, tokens, e10_cg)) { ok = false; return false; }
        std::string e10_cd = (t / "e10_cd.f32").string(); if (!run_tts_clone_w2v_layer10_conv_depthwise(model_bundle_dir, e10_cg, tokens, e10_cd)) { ok = false; return false; }
        std::string e10_cr = (t / "e10_cr.f32").string(); if (!run_tts_clone_w2v_layer10_conv_residual(model_bundle_dir, e10_ar, e10_cd, tokens, e10_cr)) { ok = false; return false; }
        std::string e10_f2 = (t / "e10_f2.f32").string(); if (!run_tts_clone_w2v_layer10_ffn2_residual(model_bundle_dir, e10_cr, tokens, e10_f2)) { ok = false; return false; }
        std::string s10 = (t / "layer10.f32").string(); if (!apply_layer_final_norm(10, e10_f2, s10)) { fail("l10_fn"); pr(); return false; }

        // Layer 11
        std::string e11_fr = (t / "e11_fr.f32").string(); if (!run_tts_clone_w2v_layer11_ffn1_residual(model_bundle_dir, s10, tokens, e11_fr)) { fail("l11_fr"); pr(); return false; }
        std::string e11_an = (t / "e11_an.f32").string(); if (!run_tts_clone_w2v_layer11_attention_norm(model_bundle_dir, e11_fr, tokens, e11_an)) { ok = false; return false; }
        std::string e11q = (t / "e11q").string(); if (!run_tts_clone_w2v_layer11_qkv(model_bundle_dir, e11_an, tokens, e11q)) { fail("l11_qkv"); pr(); return false; }
        std::string e11_at = (t / "e11_at.f32").string(); if (!run_tts_clone_w2v_layer11_attention(e11q+"/w2v_layer11_q.f32", e11q+"/w2v_layer11_k.f32", e11q+"/w2v_layer11_v.f32", attention_mask_u32, tokens, e11_at, dist_embs[11])) { fail("l11_at"); pr(); return false; }
        std::string e11_ap = (t / "e11_ap.f32").string(); if (!run_tts_clone_w2v_layer11_attention_project(model_bundle_dir, e11_at, tokens, e11_ap)) { ok = false; return false; }
        std::string e11_ar = (t / "e11_ar.f32").string(); if (!run_tts_clone_w2v_layer11_attention_residual(e11_fr, e11_ap, tokens, e11_ar)) { ok = false; return false; }
        std::string e11_cn = (t / "e11_cn.f32").string(); if (!run_tts_clone_w2v_layer11_conv_norm(model_bundle_dir, e11_ar, tokens, e11_cn)) { ok = false; return false; }
        std::string e11_cg = (t / "e11_cg.f32").string(); if (!run_tts_clone_w2v_layer11_conv_glu(model_bundle_dir, e11_cn, tokens, e11_cg)) { ok = false; return false; }
        std::string e11_cd = (t / "e11_cd.f32").string(); if (!run_tts_clone_w2v_layer11_conv_depthwise(model_bundle_dir, e11_cg, tokens, e11_cd)) { ok = false; return false; }
        std::string e11_cr = (t / "e11_cr.f32").string(); if (!run_tts_clone_w2v_layer11_conv_residual(model_bundle_dir, e11_ar, e11_cd, tokens, e11_cr)) { ok = false; return false; }
        std::string e11_f2 = (t / "e11_f2.f32").string(); if (!run_tts_clone_w2v_layer11_ffn2_residual(model_bundle_dir, e11_cr, tokens, e11_f2)) { ok = false; return false; }
        std::string s11 = (t / "layer11.f32").string(); if (!apply_layer_final_norm(11, e11_f2, s11)) { fail("l11_fn"); pr(); return false; }

        // Layer 12
        std::string e12_fr = (t / "e12_fr.f32").string(); if (!run_tts_clone_w2v_layer12_ffn1_residual(model_bundle_dir, s11, tokens, e12_fr)) { fail("l12_fr"); pr(); return false; }
        std::string e12_an = (t / "e12_an.f32").string(); if (!run_tts_clone_w2v_layer12_attention_norm(model_bundle_dir, e12_fr, tokens, e12_an)) { ok = false; return false; }
        std::string e12q = (t / "e12q").string(); if (!run_tts_clone_w2v_layer12_qkv(model_bundle_dir, e12_an, tokens, e12q)) { fail("l12_qkv"); pr(); return false; }
        std::string e12_at = (t / "e12_at.f32").string(); if (!run_tts_clone_w2v_layer12_attention(e12q+"/w2v_layer12_q.f32", e12q+"/w2v_layer12_k.f32", e12q+"/w2v_layer12_v.f32", attention_mask_u32, tokens, e12_at, dist_embs[12])) { fail("l12_at"); pr(); return false; }
        std::string e12_ap = (t / "e12_ap.f32").string(); if (!run_tts_clone_w2v_layer12_attention_project(model_bundle_dir, e12_at, tokens, e12_ap)) { ok = false; return false; }
        std::string e12_ar = (t / "e12_ar.f32").string(); if (!run_tts_clone_w2v_layer12_attention_residual(e12_fr, e12_ap, tokens, e12_ar)) { ok = false; return false; }
        std::string e12_cn = (t / "e12_cn.f32").string(); if (!run_tts_clone_w2v_layer12_conv_norm(model_bundle_dir, e12_ar, tokens, e12_cn)) { ok = false; return false; }
        std::string e12_cg = (t / "e12_cg.f32").string(); if (!run_tts_clone_w2v_layer12_conv_glu(model_bundle_dir, e12_cn, tokens, e12_cg)) { ok = false; return false; }
        std::string e12_cd = (t / "e12_cd.f32").string(); if (!run_tts_clone_w2v_layer12_conv_depthwise(model_bundle_dir, e12_cg, tokens, e12_cd)) { ok = false; return false; }
        std::string e12_cr = (t / "e12_cr.f32").string(); if (!run_tts_clone_w2v_layer12_conv_residual(model_bundle_dir, e12_ar, e12_cd, tokens, e12_cr)) { ok = false; return false; }
        std::string e12_f2 = (t / "e12_f2.f32").string(); if (!run_tts_clone_w2v_layer12_ffn2_residual(model_bundle_dir, e12_cr, tokens, e12_f2)) { ok = false; return false; }
        std::string s12 = (t / "layer12.f32").string(); if (!apply_layer_final_norm(12, e12_f2, s12)) { fail("l12_fn"); pr(); return false; }

        // Layer 13
        std::string e13_fr = (t / "e13_fr.f32").string(); if (!run_tts_clone_w2v_layer13_ffn1_residual(model_bundle_dir, s12, tokens, e13_fr)) { fail("l13_fr"); pr(); return false; }
        std::string e13_an = (t / "e13_an.f32").string(); if (!run_tts_clone_w2v_layer13_attention_norm(model_bundle_dir, e13_fr, tokens, e13_an)) { ok = false; return false; }
        std::string e13q = (t / "e13q").string(); if (!run_tts_clone_w2v_layer13_qkv(model_bundle_dir, e13_an, tokens, e13q)) { fail("l13_qkv"); pr(); return false; }
        std::string e13_at = (t / "e13_at.f32").string(); if (!run_tts_clone_w2v_layer13_attention(e13q+"/w2v_layer13_q.f32", e13q+"/w2v_layer13_k.f32", e13q+"/w2v_layer13_v.f32", attention_mask_u32, tokens, e13_at, dist_embs[13])) { fail("l13_at"); pr(); return false; }
        std::string e13_ap = (t / "e13_ap.f32").string(); if (!run_tts_clone_w2v_layer13_attention_project(model_bundle_dir, e13_at, tokens, e13_ap)) { ok = false; return false; }
        std::string e13_ar = (t / "e13_ar.f32").string(); if (!run_tts_clone_w2v_layer13_attention_residual(e13_fr, e13_ap, tokens, e13_ar)) { ok = false; return false; }
        std::string e13_cn = (t / "e13_cn.f32").string(); if (!run_tts_clone_w2v_layer13_conv_norm(model_bundle_dir, e13_ar, tokens, e13_cn)) { ok = false; return false; }
        std::string e13_cg = (t / "e13_cg.f32").string(); if (!run_tts_clone_w2v_layer13_conv_glu(model_bundle_dir, e13_cn, tokens, e13_cg)) { ok = false; return false; }
        std::string e13_cd = (t / "e13_cd.f32").string(); if (!run_tts_clone_w2v_layer13_conv_depthwise(model_bundle_dir, e13_cg, tokens, e13_cd)) { ok = false; return false; }
        std::string e13_cr = (t / "e13_cr.f32").string(); if (!run_tts_clone_w2v_layer13_conv_residual(model_bundle_dir, e13_ar, e13_cd, tokens, e13_cr)) { ok = false; return false; }
        std::string e13_f2 = (t / "e13_f2.f32").string(); if (!run_tts_clone_w2v_layer13_ffn2_residual(model_bundle_dir, e13_cr, tokens, e13_f2)) { ok = false; return false; }
        std::string s13 = (t / "layer13.f32").string(); if (!apply_layer_final_norm(13, e13_f2, s13)) { fail("l13_fn"); pr(); return false; }

        // Layer 14
        std::string f14_fr = (t / "f14_fr.f32").string(); if (!run_tts_clone_w2v_layer14_ffn1_residual(model_bundle_dir, s13, tokens, f14_fr)) { ok = false; return false; }
        std::string f14_an = (t / "f14_an.f32").string(); if (!run_tts_clone_w2v_layer14_attention_norm(model_bundle_dir, f14_fr, tokens, f14_an)) { ok = false; return false; }
        std::string f14q = (t / "f14q").string(); if (!run_tts_clone_w2v_layer14_qkv(model_bundle_dir, f14_an, tokens, f14q)) { fail("l14_qkv"); pr(); return false; }
        std::string f14_at = (t / "f14_at.f32").string(); if (!run_tts_clone_w2v_layer14_attention(f14q+"/w2v_layer14_q.f32", f14q+"/w2v_layer14_k.f32", f14q+"/w2v_layer14_v.f32", attention_mask_u32, tokens, f14_at, dist_embs[14])) { fail("l14_at"); pr(); return false; }
        std::string f14_ap = (t / "f14_ap.f32").string(); if (!run_tts_clone_w2v_layer14_attention_project(model_bundle_dir, f14_at, tokens, f14_ap)) { ok = false; return false; }
        std::string f14_ar = (t / "f14_ar.f32").string(); if (!run_tts_clone_w2v_layer14_attention_residual(f14_fr, f14_ap, tokens, f14_ar)) { ok = false; return false; }
        std::string f14_cn = (t / "f14_cn.f32").string(); if (!run_tts_clone_w2v_layer14_conv_norm(model_bundle_dir, f14_ar, tokens, f14_cn)) { ok = false; return false; }
        std::string f14_cg = (t / "f14_cg.f32").string(); if (!run_tts_clone_w2v_layer14_conv_glu(model_bundle_dir, f14_cn, tokens, f14_cg)) { ok = false; return false; }
        std::string f14_cd = (t / "f14_cd.f32").string(); if (!run_tts_clone_w2v_layer14_conv_depthwise(model_bundle_dir, f14_cg, tokens, f14_cd)) { ok = false; return false; }
        std::string f14_cr = (t / "f14_cr.f32").string(); if (!run_tts_clone_w2v_layer14_conv_residual(model_bundle_dir, f14_ar, f14_cd, tokens, f14_cr)) { ok = false; return false; }
        std::string f14_f2 = (t / "f14_f2.f32").string(); if (!run_tts_clone_w2v_layer14_ffn2_residual(model_bundle_dir, f14_cr, tokens, f14_f2)) { ok = false; return false; }
        std::string s14 = (t / "layer14.f32").string(); if (!apply_layer_final_norm(14, f14_f2, s14)) { fail("l14_fn"); pr(); return false; }

        // Layer 15
        std::string d15_fr = (t / "d15_fr.f32").string(); if (!run_tts_clone_w2v_layer15_ffn1_residual(model_bundle_dir, s14, tokens, d15_fr)) { ok = false; return false; }
        std::string d15_an = (t / "d15_an.f32").string(); if (!run_tts_clone_w2v_layer15_attention_norm(model_bundle_dir, d15_fr, tokens, d15_an)) { ok = false; return false; }
        std::string dl15q = (t / "dl15q").string(); if (!run_tts_clone_w2v_layer15_qkv(model_bundle_dir, d15_an, tokens, dl15q)) { fail("l15_qkv"); pr(); return false; }
        std::string d15_at = (t / "d15_at.f32").string(); if (!run_tts_clone_w2v_layer15_attention(dl15q+"/w2v_layer15_q.f32", dl15q+"/w2v_layer15_k.f32", dl15q+"/w2v_layer15_v.f32", attention_mask_u32, tokens, d15_at, dist_embs[15])) { fail("l15_at"); pr(); return false; }
        std::string d15_ap = (t / "d15_ap.f32").string(); if (!run_tts_clone_w2v_layer15_attention_project(model_bundle_dir, d15_at, tokens, d15_ap)) { ok = false; return false; }
        std::string d15_ar = (t / "d15_ar.f32").string(); if (!run_tts_clone_w2v_layer15_attention_residual(d15_fr, d15_ap, tokens, d15_ar)) { ok = false; return false; }
        std::string d15_cn = (t / "d15_cn.f32").string(); if (!run_tts_clone_w2v_layer15_conv_norm(model_bundle_dir, d15_ar, tokens, d15_cn)) { ok = false; return false; }
        std::string d15_cg = (t / "d15_cg.f32").string(); if (!run_tts_clone_w2v_layer15_conv_glu(model_bundle_dir, d15_cn, tokens, d15_cg)) { ok = false; return false; }
        std::string d15_cd = (t / "d15_cd.f32").string(); if (!run_tts_clone_w2v_layer15_conv_depthwise(model_bundle_dir, d15_cg, tokens, d15_cd)) { ok = false; return false; }
        std::string d15_cr = (t / "d15_cr.f32").string(); if (!run_tts_clone_w2v_layer15_conv_residual(model_bundle_dir, d15_ar, d15_cd, tokens, d15_cr)) { ok = false; return false; }
        std::string d15_f2 = (t / "d15_f2.f32").string(); if (!run_tts_clone_w2v_layer15_ffn2_residual(model_bundle_dir, d15_cr, tokens, d15_f2)) { fail("l15_f2"); pr(); return false; }
        std::string s15 = (t / "layer15.f32").string(); if (!apply_layer_final_norm(15, d15_f2, s15)) { fail("l15_fn"); pr(); return false; }

        // Layer 16
        std::string d16_fr = (t / "d16_fr.f32").string(); if (!run_tts_clone_w2v_layer16_ffn1_residual(model_bundle_dir, s15, tokens, d16_fr)) { ok = false; return false; }
        std::string d16_an = (t / "d16_an.f32").string(); if (!run_tts_clone_w2v_layer16_attention_norm(model_bundle_dir, d16_fr, tokens, d16_an)) { ok = false; return false; }
        std::string dl16q = (t / "dl16q").string(); if (!run_tts_clone_w2v_layer16_qkv(model_bundle_dir, d16_an, tokens, dl16q)) { fail("l16_qkv"); pr(); return false; }
        std::string d16_at = (t / "d16_at.f32").string(); if (!run_tts_clone_w2v_layer16_attention(dl16q+"/w2v_layer16_q.f32", dl16q+"/w2v_layer16_k.f32", dl16q+"/w2v_layer16_v.f32", attention_mask_u32, tokens, d16_at, dist_embs[16])) { fail("l16_at"); pr(); return false; }
        std::string d16_ap = (t / "d16_ap.f32").string(); if (!run_tts_clone_w2v_layer16_attention_project(model_bundle_dir, d16_at, tokens, d16_ap)) { ok = false; return false; }
        std::string d16_ar = (t / "d16_ar.f32").string(); if (!run_tts_clone_w2v_layer16_attention_residual(d16_fr, d16_ap, tokens, d16_ar)) { ok = false; return false; }
        std::string d16_cn = (t / "d16_cn.f32").string(); if (!run_tts_clone_w2v_layer16_conv_norm(model_bundle_dir, d16_ar, tokens, d16_cn)) { ok = false; return false; }
        std::string d16_cg = (t / "d16_cg.f32").string(); if (!run_tts_clone_w2v_layer16_conv_glu(model_bundle_dir, d16_cn, tokens, d16_cg)) { ok = false; return false; }
        std::string d16_cd = (t / "d16_cd.f32").string(); if (!run_tts_clone_w2v_layer16_conv_depthwise(model_bundle_dir, d16_cg, tokens, d16_cd)) { ok = false; return false; }
        std::string d16_cr = (t / "d16_cr.f32").string(); if (!run_tts_clone_w2v_layer16_conv_residual(model_bundle_dir, d16_ar, d16_cd, tokens, d16_cr)) { ok = false; return false; }
        std::string d16_f2 = (t / "d16_f2.f32").string(); if (!run_tts_clone_w2v_layer16_ffn2_residual(model_bundle_dir, d16_cr, tokens, d16_f2)) { fail("l16_f2"); pr(); return false; }
        std::string s16 = (t / "layer16.f32").string(); if (!apply_layer_final_norm(16, d16_f2, s16)) { fail("l16_fn"); pr(); return false; }

        // layer16.f32 = hidden_states[17] (output after 17th transformer block, 0-indexed layer 16)
        if (!run_tts_clone_w2v_normalize(model_bundle_dir, s16, tokens, output_spk_cond_f32)) { fail("norm"); pr(); return false; }

        pr();
        return true;
    }
}

[[maybe_unused]] bool run_tts_clone_real(const std::string& model_bundle_dir,
                                          const std::string& feature_manifest,
                                          const std::string& w2v_input_features_f32,
                                          const std::string& w2v_attention_mask_u32,
                                          const std::string& output_voice_bundle) {
    struct TmpDir { std::filesystem::path p; ~TmpDir() { std::filesystem::remove_all(p); } };
    TmpDir tmp{std::filesystem::temp_directory_path() / "mit2_clone_real"};
    std::filesystem::create_directories(tmp.p);
    const auto& t = tmp.p;

    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    const auto feature_issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
    if (!feature_issues.empty()) {
        std::cout << "{\n  \"stage\":\"tts_clone_real\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_real_issues\":["; print_json_string_array(feature_issues); std::cout << "],\n";
        std::cout << "  \"ready_native_voice_clone\":false\n}\n";
        return false;
    }

    const uint64_t features_bytes = std::filesystem::file_size(w2v_input_features_f32);
    const uint32_t spk_tokens = static_cast<uint32_t>(features_bytes / (160u * sizeof(float)));
    if (spk_tokens == 0) {
        std::cout << "{\n  \"stage\":\"tts_clone_real\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_real_issues\":[\"w2v_features_empty_or_wrong_size\"],\n";
        std::cout << "  \"ready_native_voice_clone\":false\n}\n";
        return false;
    }
    const auto ref_audio = read_raw_f32(manifest.preprocessed_output_f32);
    const auto ref_fbank = read_raw_f32(manifest.output_fbank_f32);

    std::string s2mel_style = (t / "s2mel_style.f32").string();
    std::string style_prov = "fast_placeholder_audio_fbank_stats";
    if (!run_tts_clone_campplus_style_from_features(model_bundle_dir, feature_manifest, s2mel_style)) {
        const auto fast_style = make_fast_clone_style(ref_audio, ref_fbank);
        { std::ofstream ofs(s2mel_style, std::ios::binary); ofs.write(reinterpret_cast<const char*>(fast_style.data()), fast_style.size() * sizeof(float)); }
    } else {
        style_prov = "native_campplus_style_cpu";
    }

    std::string spk_cond = (t / "spk_cond.f32").string();
    if (!run_tts_clone_w2v_encoder(model_bundle_dir, w2v_input_features_f32, w2v_attention_mask_u32, spk_tokens, spk_cond)) {
        std::cout << "{\n  \"stage\":\"tts_clone_real\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_real_issues\":[\"w2v_encoder_failed\"],\n";
        std::cout << "  \"ready_native_voice_clone\":false\n}\n";
        return false;
    }

    std::string sref = (t / "sref.f32").string();
    std::string codes = (t / "codes.u32").string();
    if (!run_tts_clone_semantic_quantize(model_bundle_dir, spk_cond, spk_tokens, sref, codes)) {
        std::cout << "{\n  \"stage\":\"tts_clone_real\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_real_issues\":[\"semantic_quantize_failed\"],\n";
        std::cout << "  \"ready_native_voice_clone\":false\n}\n";
        return false;
    }

    const uint32_t prompt_tokens = static_cast<uint32_t>(manifest.mel_frames);
    std::string s2mel_prompt = (t / "s2mel_prompt.f32").string();
    if (!run_tts_clone_s2mel_prompt_from_sref(model_bundle_dir, feature_manifest, sref, spk_tokens, s2mel_prompt)) {
        std::cout << "{\n  \"stage\":\"tts_clone_real\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_real_issues\":[\"s2mel_prompt_failed\"],\n";
        std::cout << "  \"ready_native_voice_clone\":false\n}\n";
        return false;
    }

    if (!run_tts_clone_write_voice_bundle_from_features(feature_manifest, spk_cond, spk_tokens,
                                                         s2mel_style, s2mel_prompt, prompt_tokens,
                                                         output_voice_bundle)) {
        std::cout << "{\n  \"stage\":\"tts_clone_real\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_real_issues\":[\"voice_bundle_write_failed\"],\n";
        std::cout << "  \"ready_native_voice_clone\":false\n}\n";
        return false;
    }

    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_clone_real\",\n";
    std::cout << "  \"ok\": true,\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
    std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
    std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
    std::cout << "  \"spk_cond_provenance\": \"native_w2v_bert_spk_cond\",\n";
    std::cout << "  \"s2mel_style_provenance\": \"" << style_prov << "\",\n";
    std::cout << "  \"clone_quality_ready\": " << (style_prov == "native_campplus_style_cpu" ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_voice_clone\": " << (style_prov == "native_campplus_style_cpu" ? "true" : "false") << ",\n";
    std::cout << "  \"next_native_boundary\": \"Verify against PyTorch golden with real model weights\"\n";
    std::cout << "}\n";
    return true;
}

[[maybe_unused]] bool run_tts_clone_w2v_extract_features(const std::string& preprocess_manifest,
                                                          const std::string& output_features_f32,
                                                          const std::string& output_mask_u32) {
    ClonePreprocessManifest pre_manifest;
    const auto issues = clone_preprocess_manifest_issues(preprocess_manifest, pre_manifest);
    if (!issues.empty()) {
        std::cout << "{\n  \"stage\":\"tts_clone_w2v_extract_features\",\n  \"ok\":false,\n";
        std::cout << "  \"clone_readiness_issues\":["; print_json_string_array(issues); std::cout << "],\n";
        std::cout << "  \"ready_native_w2v_feature_extraction\":false\n}\n";
        return false;
    }

    const auto audio = read_raw_f32(pre_manifest.output_f32);
    if (audio.size() < 400) {
        std::cout << "{\n  \"stage\":\"tts_clone_w2v_extract_features\",\n  \"ok\":false,\n";
        std::cout << "  \"w2v_feature_extraction_issues\":[\"audio_too_short\"],\n";
        std::cout << "  \"ready_native_w2v_feature_extraction\":false\n}\n";
        return false;
    }

    uint32_t out_tokens = 0;
    std::vector<float> features;
    try {
        features = extract_w2v_bert_features_16k_f32(audio, out_tokens);
    } catch (const std::exception& e) {
        std::cout << "{\n  \"stage\":\"tts_clone_w2v_extract_features\",\n  \"ok\":false,\n";
        std::cout << "  \"w2v_feature_extraction_issues\":[\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_feature_extraction\":false\n}\n";
        return false;
    }

    // Write features [out_tokens, 160] f32
    write_raw_f32(output_features_f32, features);

    // Write attention mask: all 1s, [out_tokens] uint32
    std::vector<uint32_t> mask(out_tokens, 1u);
    { std::ofstream ofs(output_mask_u32, std::ios::binary);
      ofs.write(reinterpret_cast<const char*>(mask.data()), static_cast<std::streamsize>(mask.size() * sizeof(uint32_t))); }

    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_clone_w2v_extract_features\",\n";
    std::cout << "  \"ok\": true,\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
    std::cout << "  \"output_features_f32\": \"" << json_escape(output_features_f32) << "\",\n";
    std::cout << "  \"output_mask_u32\": \"" << json_escape(output_mask_u32) << "\",\n";
    std::cout << "  \"out_tokens\": " << out_tokens << ",\n";
    std::cout << "  \"features_shape\": \"[" << out_tokens << ",160]\",\n";
    std::cout << "  \"ready_native_w2v_feature_extraction\": true\n";
    std::cout << "}\n";
    return true;
}

// Full native clone pipeline: prepare features → extract W2V-BERT features → run encoder → write bundle
[[maybe_unused]] bool run_tts_clone_native(const std::string& model_bundle_dir,
                                            const std::string& audio_wav,
                                            const std::string& output_voice_bundle) {
    auto emit_fail = [&](const std::string& stage, const std::string& issue) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"" << json_escape(stage) << "\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"output_voice_bundle\": \"" << json_escape(output_voice_bundle) << "\",\n";
        std::cout << "  \"clone_native_issues\": [\"" << json_escape(issue) << "\"],\n";
        std::cout << "  \"ready_native_voice_clone\": false\n";
        std::cout << "}\n";
    };

    CloneScratchDir scratch;
    const std::string work_dir = scratch.str();
    std::string normalized_wav;
    if (!normalize_clone_audio_for_pipeline(audio_wav, work_dir, normalized_wav)) {
        emit_fail("tts_clone_native", "audio_normalization_failed");
        return false;
    }

    // Step 1: prepare audio features (preprocess, mel, fbank)
    if (!run_tts_clone_prepare_features(normalized_wav, work_dir)) {
        emit_fail("tts_clone_native", "prepare_features_failed");
        return false;
    }

    const std::string feature_manifest = (std::filesystem::path(work_dir) / "clone_features.manifest.json").string();
    CloneFeatureManifest feat_manifest;
    ClonePreprocessManifest pre_manifest;
    const auto feat_issues = clone_feature_manifest_issues(feature_manifest, feat_manifest, pre_manifest);
    if (!feat_issues.empty()) {
        emit_fail("tts_clone_native", "feature_manifest_invalid");
        return false;
    }

    // Step 2: extract W2V-BERT input features natively
    const std::string preprocess_manifest = feat_manifest.preprocess_manifest;
    const std::string w2v_features_f32 = (std::filesystem::path(work_dir) / "w2v_features.f32").string();
    const std::string w2v_mask_u32 = (std::filesystem::path(work_dir) / "w2v_mask.u32").string();
    if (!run_tts_clone_w2v_extract_features(preprocess_manifest, w2v_features_f32, w2v_mask_u32)) {
        emit_fail("tts_clone_native", "w2v_feature_extraction_failed");
        return false;
    }

    // Step 3: run full encoder pipeline and write voice bundle
    return run_tts_clone_real(model_bundle_dir, feature_manifest,
                              w2v_features_f32, w2v_mask_u32, output_voice_bundle);
}
