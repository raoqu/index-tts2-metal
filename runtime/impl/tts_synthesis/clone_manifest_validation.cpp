// Voice-clone manifest validation: preprocess/feature manifest issue checks and raw f32/u32 count validators.
// Part of the tts_synthesis translation unit (see ../tts_synthesis.cpp);
// these files are #included in order into a single TU, so definition order
// across files is significant.

std::vector<std::string> clone_preprocess_manifest_issues(const std::string& preprocess_manifest,
                                                          ClonePreprocessManifest& manifest) {
    std::vector<std::string> issues;
    try {
        manifest = parse_clone_preprocess_manifest(preprocess_manifest);
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
            if (manifest.output_sha256.empty()) {
                issues.push_back("missing_output_sha256");
            } else if (file_sha256_hex(manifest.output_f32) != manifest.output_sha256) {
                issues.push_back("preprocessed_output_f32_sha256_mismatch");
            }
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("manifest_parse_error: ") + e.what());
    }
    return issues;
}

void append_file_sha_size_issues(std::vector<std::string>& issues,
                                 const std::string& path,
                                 const std::string& expected_sha256,
                                 uint64_t expected_bytes,
                                 const std::string& missing_issue,
                                 const std::string& size_issue,
                                 const std::string& missing_sha_issue,
                                 const std::string& sha_issue) {
    if (path.empty()) {
        issues.push_back(missing_issue);
        return;
    }
    if (!std::filesystem::exists(path)) {
        issues.push_back(missing_issue);
        return;
    }
    const uint64_t actual_bytes = static_cast<uint64_t>(std::filesystem::file_size(path));
    if (actual_bytes != expected_bytes) {
        issues.push_back(size_issue);
    }
    if (expected_sha256.empty()) {
        issues.push_back(missing_sha_issue);
    } else if (file_sha256_hex(path) != expected_sha256) {
        issues.push_back(sha_issue);
    }
}

std::vector<std::string> clone_feature_manifest_issues(const std::string& feature_manifest,
                                                       CloneFeatureManifest& manifest,
                                                       ClonePreprocessManifest& preprocess_manifest) {
    std::vector<std::string> issues;
    try {
        manifest = parse_clone_feature_manifest(feature_manifest);
        if (manifest.format != "mit2-clone-feature-prep") {
            issues.push_back("unexpected_feature_manifest_format");
        }
        if (manifest.version != 1) {
            issues.push_back("unsupported_feature_manifest_version");
        }
        if (manifest.preprocessed_sample_rate != 16000) {
            issues.push_back("unexpected_feature_preprocessed_sample_rate");
        }
        if (manifest.preprocessed_samples == 0) {
            issues.push_back("empty_feature_preprocessed_audio");
        }
        if (manifest.mel_frames == 0) {
            issues.push_back("empty_feature_mel");
        }
        if (manifest.fbank_frames == 0) {
            issues.push_back("empty_feature_fbank");
        }
        if (!manifest.ready_native_clone_audio_preprocess) {
            issues.push_back("feature_manifest_audio_preprocess_not_ready");
        }
        if (!manifest.ready_native_clone_mel_extraction) {
            issues.push_back("feature_manifest_mel_extraction_not_ready");
        }
        if (!manifest.ready_native_clone_fbank_extraction) {
            issues.push_back("feature_manifest_fbank_extraction_not_ready");
        }
        if (!manifest.ready_native_clone_feature_prep) {
            issues.push_back("feature_manifest_not_ready");
        }
        if (manifest.ready_native_voice_clone) {
            issues.push_back("feature_manifest_must_not_claim_voice_clone_ready");
        }

        const auto preprocess_issues = clone_preprocess_manifest_issues(manifest.preprocess_manifest, preprocess_manifest);
        for (const auto& issue : preprocess_issues) {
            issues.push_back("preprocess_" + issue);
        }
        if (preprocess_issues.empty()) {
            if (preprocess_manifest.output_f32 != manifest.preprocessed_output_f32) {
                issues.push_back("preprocessed_output_f32_path_mismatch");
            }
            if (preprocess_manifest.output_sha256 != manifest.preprocessed_output_sha256) {
                issues.push_back("preprocessed_output_sha256_mismatch");
            }
            if (preprocess_manifest.preprocessed_samples != manifest.preprocessed_samples) {
                issues.push_back("preprocessed_samples_mismatch");
            }
        }

        if (!manifest.source_audio_wav.empty() && std::filesystem::exists(manifest.source_audio_wav)) {
            if (manifest.source_audio_sha256.empty()) {
                issues.push_back("missing_feature_source_audio_sha256");
            } else if (file_sha256_hex(manifest.source_audio_wav) != manifest.source_audio_sha256) {
                issues.push_back("feature_source_audio_sha256_mismatch");
            }
        }

        append_file_sha_size_issues(issues,
                                    manifest.preprocessed_output_f32,
                                    manifest.preprocessed_output_sha256,
                                    manifest.preprocessed_samples * static_cast<uint64_t>(sizeof(float)),
                                    "feature_preprocessed_output_f32_missing",
                                    "feature_preprocessed_output_f32_size_mismatch",
                                    "missing_feature_preprocessed_output_sha256",
                                    "feature_preprocessed_output_f32_sha256_mismatch");
        append_file_sha_size_issues(issues,
                                    manifest.output_mel_f32,
                                    manifest.output_mel_sha256,
                                    manifest.mel_frames * 80u * static_cast<uint64_t>(sizeof(float)),
                                    "feature_output_mel_f32_missing",
                                    "feature_output_mel_f32_size_mismatch",
                                    "missing_feature_output_mel_sha256",
                                    "feature_output_mel_f32_sha256_mismatch");
        append_file_sha_size_issues(issues,
                                    manifest.output_fbank_f32,
                                    manifest.output_fbank_sha256,
                                    manifest.fbank_frames * 80u * static_cast<uint64_t>(sizeof(float)),
                                    "feature_output_fbank_f32_missing",
                                    "feature_output_fbank_f32_size_mismatch",
                                    "missing_feature_output_fbank_sha256",
                                    "feature_output_fbank_f32_sha256_mismatch");

        try {
            const std::string mel_text = read_text_file(manifest.mel_manifest);
            if (parse_json_string_field(mel_text, "format") != "mit2-clone-mel-extract") {
                issues.push_back("unexpected_mel_manifest_format");
            }
            if (!parse_json_bool_field(mel_text, "ready_native_clone_mel_extraction")) {
                issues.push_back("mel_manifest_not_ready");
            }
            if (parse_json_string_field(mel_text, "output_mel_f32") != manifest.output_mel_f32) {
                issues.push_back("mel_manifest_output_path_mismatch");
            }
            if (parse_json_string_field(mel_text, "output_mel_sha256") != manifest.output_mel_sha256) {
                issues.push_back("mel_manifest_sha256_mismatch");
            }
            if (parse_json_u64_field(mel_text, "mel_frames") != manifest.mel_frames) {
                issues.push_back("mel_manifest_frames_mismatch");
            }
        } catch (const std::exception& e) {
            issues.push_back(std::string("mel_manifest_parse_error: ") + e.what());
        }
        try {
            const std::string fbank_text = read_text_file(manifest.fbank_manifest);
            if (parse_json_string_field(fbank_text, "format") != "mit2-clone-fbank-extract") {
                issues.push_back("unexpected_fbank_manifest_format");
            }
            if (!parse_json_bool_field(fbank_text, "ready_native_clone_fbank_extraction")) {
                issues.push_back("fbank_manifest_not_ready");
            }
            if (parse_json_string_field(fbank_text, "output_fbank_f32") != manifest.output_fbank_f32) {
                issues.push_back("fbank_manifest_output_path_mismatch");
            }
            if (parse_json_string_field(fbank_text, "output_fbank_sha256") != manifest.output_fbank_sha256) {
                issues.push_back("fbank_manifest_sha256_mismatch");
            }
            if (parse_json_u64_field(fbank_text, "fbank_frames") != manifest.fbank_frames) {
                issues.push_back("fbank_manifest_frames_mismatch");
            }
        } catch (const std::exception& e) {
            issues.push_back(std::string("fbank_manifest_parse_error: ") + e.what());
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("feature_manifest_parse_error: ") + e.what());
    }
    return issues;
}

[[maybe_unused]] bool run_tts_clone_extract_mel(const std::string& preprocess_manifest,
                                                const std::string& output_mel_f32) {
    try {
        ClonePreprocessManifest clone_manifest;
        const auto manifest_issues = clone_preprocess_manifest_issues(preprocess_manifest, clone_manifest);
        if (!manifest_issues.empty()) {
            std::cout << "{\n";
            std::cout << "  \"stage\": \"tts_clone_extract_mel\",\n";
            std::cout << "  \"ok\": false,\n";
            std::cout << "  \"product_surface_version\": 1,\n";
            std::cout << "  \"binary\": \"mit2_tts\",\n";
            std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
            std::cout << "  \"output_mel_f32\": \"" << json_escape(output_mel_f32) << "\",\n";
            std::cout << "  \"clone_readiness_issues\": ";
            print_json_string_array(manifest_issues);
            std::cout << ",\n";
            std::cout << "  \"ready_native_clone_audio_preprocess\": false,\n";
            std::cout << "  \"ready_native_clone_mel_extraction\": false,\n";
            std::cout << "  \"ready_native_voice_clone\": false\n";
            std::cout << "}\n";
            return false;
        }
        const auto audio_16k = read_raw_f32(clone_manifest.output_f32);
        const auto audio_22k = resample_linear_f32(audio_16k, 16000, 22050);
        uint32_t mel_frames = 0;
        const auto mel = extract_clone_mel_22k_f32(audio_22k, mel_frames);
        write_raw_f32(output_mel_f32, mel);
        const std::string output_sha = file_sha256_hex(output_mel_f32);
        const std::string manifest_path = output_mel_f32 + ".manifest.json";
        float min_value = mel.empty() ? 0.0f : mel[0];
        float max_value = mel.empty() ? 0.0f : mel[0];
        long double sum = 0.0L;
        for (float value : mel) {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            sum += value;
        }
        const double mean = mel.empty() ? 0.0 : static_cast<double>(sum / static_cast<long double>(mel.size()));

        std::ostringstream manifest;
        manifest << "{\n";
        manifest << "  \"format\": \"mit2-clone-mel-extract\",\n";
        manifest << "  \"version\": 1,\n";
        manifest << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        manifest << "  \"preprocessed_output_f32\": \"" << json_escape(clone_manifest.output_f32) << "\",\n";
        manifest << "  \"preprocessed_output_sha256\": \"" << json_escape(clone_manifest.output_sha256) << "\",\n";
        manifest << "  \"output_mel_f32\": \"" << json_escape(output_mel_f32) << "\",\n";
        manifest << "  \"output_mel_sha256\": \"" << output_sha << "\",\n";
        manifest << "  \"mel_layout\": \"[1,80,frames]_row_major_flat\",\n";
        manifest << "  \"source_sample_rate\": 16000,\n";
        manifest << "  \"mel_sample_rate\": 22050,\n";
        manifest << "  \"n_fft\": 1024,\n";
        manifest << "  \"win_length\": 1024,\n";
        manifest << "  \"hop_length\": 256,\n";
        manifest << "  \"n_mels\": 80,\n";
        manifest << "  \"mel_frames\": " << mel_frames << ",\n";
        manifest << "  \"mel_values\": " << mel.size() << ",\n";
        manifest << "  \"mel_min\": " << min_value << ",\n";
        manifest << "  \"mel_max\": " << max_value << ",\n";
        manifest << "  \"mel_mean\": " << mean << ",\n";
        manifest << "  \"normalization\": \"indextts_s2mel_log_clamp_1e-5\",\n";
        manifest << "  \"mel_filterbank\": \"librosa_slaney_default\",\n";
        manifest << "  \"ready_native_clone_mel_extraction\": true,\n";
        manifest << "  \"ready_native_voice_clone\": false,\n";
        manifest << "  \"next_native_boundary\": \"clone-time semantic and acoustic speech encoders for voice tensor creation\"\n";
        manifest << "}\n";
        write_text_file(manifest_path, manifest.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_extract_mel\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        std::cout << "  \"preprocessed_output_f32\": \"" << json_escape(clone_manifest.output_f32) << "\",\n";
        std::cout << "  \"output_mel_f32\": \"" << json_escape(output_mel_f32) << "\",\n";
        std::cout << "  \"output_mel_sha256\": \"" << output_sha << "\",\n";
        std::cout << "  \"mel_manifest\": \"" << json_escape(manifest_path) << "\",\n";
        std::cout << "  \"mel_manifest_sha256\": \"" << file_sha256_hex(manifest_path) << "\",\n";
        std::cout << "  \"mel_layout\": \"[1,80,frames]_row_major_flat\",\n";
        std::cout << "  \"source_sample_rate\": 16000,\n";
        std::cout << "  \"mel_sample_rate\": 22050,\n";
        std::cout << "  \"n_fft\": 1024,\n";
        std::cout << "  \"win_length\": 1024,\n";
        std::cout << "  \"hop_length\": 256,\n";
        std::cout << "  \"n_mels\": 80,\n";
        std::cout << "  \"mel_filterbank\": \"librosa_slaney_default\",\n";
        std::cout << "  \"mel_frames\": " << mel_frames << ",\n";
        std::cout << "  \"mel_values\": " << mel.size() << ",\n";
        std::cout << "  \"mel_min\": " << min_value << ",\n";
        std::cout << "  \"mel_max\": " << max_value << ",\n";
        std::cout << "  \"mel_mean\": " << mean << ",\n";
        std::cout << "  \"ready_native_clone_audio_preprocess\": true,\n";
        std::cout << "  \"ready_native_clone_mel_extraction\": true,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"remaining_native_clone_work\": [\n";
        std::cout << "    \"native semantic speech encoder and MaskGCT quantize for spk_cond_emb\",\n";
        std::cout << "    \"native CAMPPlus style forward for s2mel_style and MaskGCT S_ref prompt condition for s2mel_prompt\"\n";
        std::cout << "  ]\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_extract_mel\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        std::cout << "  \"output_mel_f32\": \"" << json_escape(output_mel_f32) << "\",\n";
        std::cout << "  \"ready_native_clone_mel_extraction\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_extract_fbank(const std::string& preprocess_manifest,
                                                  const std::string& output_fbank_f32) {
    try {
        ClonePreprocessManifest clone_manifest;
        const auto manifest_issues = clone_preprocess_manifest_issues(preprocess_manifest, clone_manifest);
        if (!manifest_issues.empty()) {
            std::cout << "{\n";
            std::cout << "  \"stage\": \"tts_clone_extract_fbank\",\n";
            std::cout << "  \"ok\": false,\n";
            std::cout << "  \"product_surface_version\": 1,\n";
            std::cout << "  \"binary\": \"mit2_tts\",\n";
            std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
            std::cout << "  \"output_fbank_f32\": \"" << json_escape(output_fbank_f32) << "\",\n";
            std::cout << "  \"clone_readiness_issues\": ";
            print_json_string_array(manifest_issues);
            std::cout << ",\n";
            std::cout << "  \"ready_native_clone_audio_preprocess\": false,\n";
            std::cout << "  \"ready_native_clone_fbank_extraction\": false,\n";
            std::cout << "  \"ready_native_voice_clone\": false\n";
            std::cout << "}\n";
            return false;
        }
        const auto audio_16k = read_raw_f32(clone_manifest.output_f32);
        uint32_t fbank_frames = 0;
        const auto fbank = extract_clone_fbank_16k_f32(audio_16k, fbank_frames);
        write_raw_f32(output_fbank_f32, fbank);
        const std::string output_sha = file_sha256_hex(output_fbank_f32);
        const std::string manifest_path = output_fbank_f32 + ".manifest.json";
        float min_value = fbank.empty() ? 0.0f : fbank[0];
        float max_value = fbank.empty() ? 0.0f : fbank[0];
        long double sum = 0.0L;
        for (float value : fbank) {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            sum += value;
        }
        const double mean = fbank.empty() ? 0.0 : static_cast<double>(sum / static_cast<long double>(fbank.size()));

        std::ostringstream manifest;
        manifest << "{\n";
        manifest << "  \"format\": \"mit2-clone-fbank-extract\",\n";
        manifest << "  \"version\": 1,\n";
        manifest << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        manifest << "  \"preprocessed_output_f32\": \"" << json_escape(clone_manifest.output_f32) << "\",\n";
        manifest << "  \"preprocessed_output_sha256\": \"" << json_escape(clone_manifest.output_sha256) << "\",\n";
        manifest << "  \"output_fbank_f32\": \"" << json_escape(output_fbank_f32) << "\",\n";
        manifest << "  \"output_fbank_sha256\": \"" << output_sha << "\",\n";
        manifest << "  \"fbank_layout\": \"[frames,80]_row_major_flat\",\n";
        manifest << "  \"sample_rate\": 16000,\n";
        manifest << "  \"frame_length_samples\": 400,\n";
        manifest << "  \"frame_shift_samples\": 160,\n";
        manifest << "  \"frame_length_ms\": 25,\n";
        manifest << "  \"frame_shift_ms\": 10,\n";
        manifest << "  \"n_fft\": 512,\n";
        manifest << "  \"num_mel_bins\": 80,\n";
        manifest << "  \"dither\": 0,\n";
        manifest << "  \"mean_normalized\": true,\n";
        manifest << "  \"fbank_frames\": " << fbank_frames << ",\n";
        manifest << "  \"fbank_values\": " << fbank.size() << ",\n";
        manifest << "  \"fbank_min\": " << min_value << ",\n";
        manifest << "  \"fbank_max\": " << max_value << ",\n";
        manifest << "  \"fbank_mean\": " << mean << ",\n";
        manifest << "  \"feature_contract\": \"kaldi_style_fbank_for_campplus\",\n";
        manifest << "  \"ready_native_clone_fbank_extraction\": true,\n";
        manifest << "  \"ready_native_voice_clone\": false,\n";
        manifest << "  \"next_native_boundary\": \"native CAMPPlus acoustic style encoder for s2mel_style\"\n";
        manifest << "}\n";
        write_text_file(manifest_path, manifest.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_extract_fbank\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        std::cout << "  \"preprocessed_output_f32\": \"" << json_escape(clone_manifest.output_f32) << "\",\n";
        std::cout << "  \"output_fbank_f32\": \"" << json_escape(output_fbank_f32) << "\",\n";
        std::cout << "  \"output_fbank_sha256\": \"" << output_sha << "\",\n";
        std::cout << "  \"fbank_manifest\": \"" << json_escape(manifest_path) << "\",\n";
        std::cout << "  \"fbank_manifest_sha256\": \"" << file_sha256_hex(manifest_path) << "\",\n";
        std::cout << "  \"fbank_layout\": \"[frames,80]_row_major_flat\",\n";
        std::cout << "  \"sample_rate\": 16000,\n";
        std::cout << "  \"frame_length_samples\": 400,\n";
        std::cout << "  \"frame_shift_samples\": 160,\n";
        std::cout << "  \"frame_length_ms\": 25,\n";
        std::cout << "  \"frame_shift_ms\": 10,\n";
        std::cout << "  \"n_fft\": 512,\n";
        std::cout << "  \"num_mel_bins\": 80,\n";
        std::cout << "  \"dither\": 0,\n";
        std::cout << "  \"mean_normalized\": true,\n";
        std::cout << "  \"fbank_frames\": " << fbank_frames << ",\n";
        std::cout << "  \"fbank_values\": " << fbank.size() << ",\n";
        std::cout << "  \"fbank_min\": " << min_value << ",\n";
        std::cout << "  \"fbank_max\": " << max_value << ",\n";
        std::cout << "  \"fbank_mean\": " << mean << ",\n";
        std::cout << "  \"ready_native_clone_audio_preprocess\": true,\n";
        std::cout << "  \"ready_native_clone_fbank_extraction\": true,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"remaining_native_clone_work\": [\n";
        std::cout << "    \"native CAMPPlus acoustic style encoder forward for s2mel_style\",\n";
        std::cout << "    \"native semantic speech encoder and MaskGCT quantize for spk_cond_emb\",\n";
        std::cout << "    \"native MaskGCT S_ref prompt condition for s2mel_prompt\"\n";
        std::cout << "  ]\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_extract_fbank\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest) << "\",\n";
        std::cout << "  \"output_fbank_f32\": \"" << json_escape(output_fbank_f32) << "\",\n";
        std::cout << "  \"ready_native_clone_fbank_extraction\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_prepare_features(const std::string& audio_wav,
                                                     const std::string& output_dir) {
    try {
        constexpr uint32_t target_rate = 16000;
        const auto wav = read_wav_pcm16_mono_bytes(audio_wav);
        const auto quality = analyze_clone_audio_quality(wav);
        const bool quality_ok = quality.issues.empty();
        if (!quality_ok) {
            std::cout << "{\n";
            std::cout << "  \"stage\": \"tts_clone_prepare_features\",\n";
            std::cout << "  \"ok\": false,\n";
            std::cout << "  \"product_surface_version\": 1,\n";
            std::cout << "  \"binary\": \"mit2_tts\",\n";
            std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
            std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
            std::cout << "  \"ready_native_clone_audio_quality\": false,\n";
            std::cout << "  \"quality_issues\": ";
            print_json_string_array(quality.issues);
            std::cout << ",\n";
            std::cout << "  \"ready_native_clone_feature_prep\": false,\n";
            std::cout << "  \"ready_native_voice_clone\": false\n";
            std::cout << "}\n";
            return false;
        }

        std::filesystem::create_directories(output_dir);
        const auto dir = std::filesystem::path(output_dir);
        const std::string audio_f32_path = (dir / "audio_16k.f32").string();
        const std::string preprocess_manifest_path = audio_f32_path + ".manifest.json";
        const std::string mel_path = (dir / "mel.f32").string();
        const std::string mel_manifest_path = mel_path + ".manifest.json";
        const std::string fbank_path = (dir / "fbank.f32").string();
        const std::string fbank_manifest_path = fbank_path + ".manifest.json";
        const std::string features_manifest_path = (dir / "clone_features.manifest.json").string();

        const std::string source_sha = file_sha256_hex(audio_wav);
        const auto normalized = pcm16_mono_wav_to_f32(wav);
        const auto audio_16k = resample_linear_f32(normalized, wav.sample_rate, target_rate);
        write_raw_f32(audio_f32_path, audio_16k);
        const std::string audio_sha = file_sha256_hex(audio_f32_path);

        std::ostringstream preprocess_manifest;
        preprocess_manifest << "{\n";
        preprocess_manifest << "  \"format\": \"mit2-clone-audio-preprocess\",\n";
        preprocess_manifest << "  \"version\": 1,\n";
        preprocess_manifest << "  \"source_audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        preprocess_manifest << "  \"source_audio_sha256\": \"" << source_sha << "\",\n";
        preprocess_manifest << "  \"output_f32\": \"" << json_escape(audio_f32_path) << "\",\n";
        preprocess_manifest << "  \"output_sha256\": \"" << audio_sha << "\",\n";
        preprocess_manifest << "  \"audio_format\": \"f32_mono_raw\",\n";
        preprocess_manifest << "  \"source_sample_rate\": " << wav.sample_rate << ",\n";
        preprocess_manifest << "  \"target_sample_rate\": " << target_rate << ",\n";
        preprocess_manifest << "  \"source_samples\": " << quality.samples << ",\n";
        preprocess_manifest << "  \"preprocessed_samples\": " << audio_16k.size() << ",\n";
        preprocess_manifest << "  \"preprocessed_duration_seconds\": "
                            << (static_cast<double>(audio_16k.size()) / static_cast<double>(target_rate)) << ",\n";
        preprocess_manifest << "  \"audio_peak_normalized\": " << quality.peak << ",\n";
        preprocess_manifest << "  \"audio_rms_normalized\": " << quality.rms << ",\n";
        preprocess_manifest << "  \"near_silence_sample_ratio\": " << quality.near_silence_ratio << ",\n";
        preprocess_manifest << "  \"clipping_sample_ratio\": " << quality.clipping_ratio << ",\n";
        preprocess_manifest << "  \"ready_native_clone_audio_preprocess\": true,\n";
        preprocess_manifest << "  \"ready_native_voice_clone\": false,\n";
        preprocess_manifest << "  \"next_native_boundary\": \"clone-time semantic/acoustic speech encoders for voice tensor creation\"\n";
        preprocess_manifest << "}\n";
        write_text_file(preprocess_manifest_path, preprocess_manifest.str());

        const auto audio_22k = resample_linear_f32(audio_16k, target_rate, 22050);
        uint32_t mel_frames = 0;
        const auto mel = extract_clone_mel_22k_f32(audio_22k, mel_frames);
        write_raw_f32(mel_path, mel);
        const std::string mel_sha = file_sha256_hex(mel_path);
        const auto mel_stats = compute_float_vector_stats(mel);
        std::ostringstream mel_manifest;
        mel_manifest << "{\n";
        mel_manifest << "  \"format\": \"mit2-clone-mel-extract\",\n";
        mel_manifest << "  \"version\": 1,\n";
        mel_manifest << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest_path) << "\",\n";
        mel_manifest << "  \"preprocessed_output_f32\": \"" << json_escape(audio_f32_path) << "\",\n";
        mel_manifest << "  \"preprocessed_output_sha256\": \"" << audio_sha << "\",\n";
        mel_manifest << "  \"output_mel_f32\": \"" << json_escape(mel_path) << "\",\n";
        mel_manifest << "  \"output_mel_sha256\": \"" << mel_sha << "\",\n";
        mel_manifest << "  \"mel_layout\": \"[1,80,frames]_row_major_flat\",\n";
        mel_manifest << "  \"source_sample_rate\": 16000,\n";
        mel_manifest << "  \"mel_sample_rate\": 22050,\n";
        mel_manifest << "  \"n_fft\": 1024,\n";
        mel_manifest << "  \"win_length\": 1024,\n";
        mel_manifest << "  \"hop_length\": 256,\n";
        mel_manifest << "  \"n_mels\": 80,\n";
        mel_manifest << "  \"mel_frames\": " << mel_frames << ",\n";
        mel_manifest << "  \"mel_values\": " << mel.size() << ",\n";
        mel_manifest << "  \"mel_min\": " << mel_stats.min << ",\n";
        mel_manifest << "  \"mel_max\": " << mel_stats.max << ",\n";
        mel_manifest << "  \"mel_mean\": " << mel_stats.mean << ",\n";
        mel_manifest << "  \"normalization\": \"indextts_s2mel_log_clamp_1e-5\",\n";
        mel_manifest << "  \"mel_filterbank\": \"librosa_slaney_default\",\n";
        mel_manifest << "  \"ready_native_clone_mel_extraction\": true,\n";
        mel_manifest << "  \"ready_native_voice_clone\": false,\n";
        mel_manifest << "  \"next_native_boundary\": \"clone-time semantic and acoustic speech encoders for voice tensor creation\"\n";
        mel_manifest << "}\n";
        write_text_file(mel_manifest_path, mel_manifest.str());

        uint32_t fbank_frames = 0;
        const auto fbank = extract_clone_fbank_16k_f32(audio_16k, fbank_frames);
        write_raw_f32(fbank_path, fbank);
        const std::string fbank_sha = file_sha256_hex(fbank_path);
        const auto fbank_stats = compute_float_vector_stats(fbank);
        std::ostringstream fbank_manifest;
        fbank_manifest << "{\n";
        fbank_manifest << "  \"format\": \"mit2-clone-fbank-extract\",\n";
        fbank_manifest << "  \"version\": 1,\n";
        fbank_manifest << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest_path) << "\",\n";
        fbank_manifest << "  \"preprocessed_output_f32\": \"" << json_escape(audio_f32_path) << "\",\n";
        fbank_manifest << "  \"preprocessed_output_sha256\": \"" << audio_sha << "\",\n";
        fbank_manifest << "  \"output_fbank_f32\": \"" << json_escape(fbank_path) << "\",\n";
        fbank_manifest << "  \"output_fbank_sha256\": \"" << fbank_sha << "\",\n";
        fbank_manifest << "  \"fbank_layout\": \"[frames,80]_row_major_flat\",\n";
        fbank_manifest << "  \"sample_rate\": 16000,\n";
        fbank_manifest << "  \"frame_length_samples\": 400,\n";
        fbank_manifest << "  \"frame_shift_samples\": 160,\n";
        fbank_manifest << "  \"frame_length_ms\": 25,\n";
        fbank_manifest << "  \"frame_shift_ms\": 10,\n";
        fbank_manifest << "  \"n_fft\": 512,\n";
        fbank_manifest << "  \"num_mel_bins\": 80,\n";
        fbank_manifest << "  \"dither\": 0,\n";
        fbank_manifest << "  \"mean_normalized\": true,\n";
        fbank_manifest << "  \"fbank_frames\": " << fbank_frames << ",\n";
        fbank_manifest << "  \"fbank_values\": " << fbank.size() << ",\n";
        fbank_manifest << "  \"fbank_min\": " << fbank_stats.min << ",\n";
        fbank_manifest << "  \"fbank_max\": " << fbank_stats.max << ",\n";
        fbank_manifest << "  \"fbank_mean\": " << fbank_stats.mean << ",\n";
        fbank_manifest << "  \"feature_contract\": \"kaldi_style_fbank_for_campplus\",\n";
        fbank_manifest << "  \"ready_native_clone_fbank_extraction\": true,\n";
        fbank_manifest << "  \"ready_native_voice_clone\": false,\n";
        fbank_manifest << "  \"next_native_boundary\": \"native CAMPPlus acoustic style encoder for s2mel_style\"\n";
        fbank_manifest << "}\n";
        write_text_file(fbank_manifest_path, fbank_manifest.str());

        std::ostringstream features_manifest;
        features_manifest << "{\n";
        features_manifest << "  \"format\": \"mit2-clone-feature-prep\",\n";
        features_manifest << "  \"version\": 1,\n";
        features_manifest << "  \"source_audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        features_manifest << "  \"source_audio_sha256\": \"" << source_sha << "\",\n";
        features_manifest << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        features_manifest << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest_path) << "\",\n";
        features_manifest << "  \"preprocessed_output_f32\": \"" << json_escape(audio_f32_path) << "\",\n";
        features_manifest << "  \"preprocessed_output_sha256\": \"" << audio_sha << "\",\n";
        features_manifest << "  \"mel_manifest\": \"" << json_escape(mel_manifest_path) << "\",\n";
        features_manifest << "  \"output_mel_f32\": \"" << json_escape(mel_path) << "\",\n";
        features_manifest << "  \"output_mel_sha256\": \"" << mel_sha << "\",\n";
        features_manifest << "  \"fbank_manifest\": \"" << json_escape(fbank_manifest_path) << "\",\n";
        features_manifest << "  \"output_fbank_f32\": \"" << json_escape(fbank_path) << "\",\n";
        features_manifest << "  \"output_fbank_sha256\": \"" << fbank_sha << "\",\n";
        features_manifest << "  \"preprocessed_sample_rate\": 16000,\n";
        features_manifest << "  \"preprocessed_samples\": " << audio_16k.size() << ",\n";
        features_manifest << "  \"mel_frames\": " << mel_frames << ",\n";
        features_manifest << "  \"fbank_frames\": " << fbank_frames << ",\n";
        features_manifest << "  \"ready_native_clone_audio_preprocess\": true,\n";
        features_manifest << "  \"ready_native_clone_mel_extraction\": true,\n";
        features_manifest << "  \"ready_native_clone_fbank_extraction\": true,\n";
        features_manifest << "  \"ready_native_clone_feature_prep\": true,\n";
        features_manifest << "  \"ready_native_voice_clone\": false,\n";
        features_manifest << "  \"remaining_native_clone_work\": [\n";
        features_manifest << "    \"native semantic speech encoder and MaskGCT quantize for spk_cond_emb\",\n";
        features_manifest << "    \"native CAMPPlus acoustic style encoder forward for s2mel_style\",\n";
        features_manifest << "    \"native MaskGCT S_ref prompt condition for s2mel_prompt\"\n";
        features_manifest << "  ]\n";
        features_manifest << "}\n";
        write_text_file(features_manifest_path, features_manifest.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_prepare_features\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"features_manifest\": \"" << json_escape(features_manifest_path) << "\",\n";
        std::cout << "  \"features_manifest_sha256\": \"" << file_sha256_hex(features_manifest_path) << "\",\n";
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(preprocess_manifest_path) << "\",\n";
        std::cout << "  \"preprocessed_output_f32\": \"" << json_escape(audio_f32_path) << "\",\n";
        std::cout << "  \"preprocessed_output_sha256\": \"" << audio_sha << "\",\n";
        std::cout << "  \"mel_manifest\": \"" << json_escape(mel_manifest_path) << "\",\n";
        std::cout << "  \"output_mel_f32\": \"" << json_escape(mel_path) << "\",\n";
        std::cout << "  \"output_mel_sha256\": \"" << mel_sha << "\",\n";
        std::cout << "  \"fbank_manifest\": \"" << json_escape(fbank_manifest_path) << "\",\n";
        std::cout << "  \"output_fbank_f32\": \"" << json_escape(fbank_path) << "\",\n";
        std::cout << "  \"output_fbank_sha256\": \"" << fbank_sha << "\",\n";
        std::cout << "  \"preprocessed_sample_rate\": 16000,\n";
        std::cout << "  \"preprocessed_samples\": " << audio_16k.size() << ",\n";
        std::cout << "  \"mel_frames\": " << mel_frames << ",\n";
        std::cout << "  \"mel_values\": " << mel.size() << ",\n";
        std::cout << "  \"fbank_frames\": " << fbank_frames << ",\n";
        std::cout << "  \"fbank_values\": " << fbank.size() << ",\n";
        std::cout << "  \"ready_native_clone_audio_preprocess\": true,\n";
        std::cout << "  \"ready_native_clone_mel_extraction\": true,\n";
        std::cout << "  \"ready_native_clone_fbank_extraction\": true,\n";
        std::cout << "  \"ready_native_clone_feature_prep\": true,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"remaining_native_clone_work\": [\n";
        std::cout << "    \"native semantic speech encoder and MaskGCT quantize for spk_cond_emb\",\n";
        std::cout << "    \"native CAMPPlus acoustic style encoder forward for s2mel_style\",\n";
        std::cout << "    \"native MaskGCT S_ref prompt condition for s2mel_prompt\"\n";
        std::cout << "  ]\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_prepare_features\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"audio_wav\": \"" << json_escape(audio_wav) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"ready_native_clone_feature_prep\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool inspect_tts_clone_feature_readiness(const std::string& feature_manifest) {
    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    const auto issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
    const bool ok = issues.empty();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_clone_feature_readiness\",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
    std::cout << "  \"manifest_parsed\": " << (manifest.format.empty() ? "false" : "true") << ",\n";
    if (!manifest.format.empty()) {
        std::cout << "  \"manifest_format\": \"" << json_escape(manifest.format) << "\",\n";
        std::cout << "  \"manifest_version\": " << manifest.version << ",\n";
        std::cout << "  \"source_audio_wav\": \"" << json_escape(manifest.source_audio_wav) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(manifest.output_dir) << "\",\n";
        std::cout << "  \"preprocess_manifest\": \"" << json_escape(manifest.preprocess_manifest) << "\",\n";
        std::cout << "  \"preprocessed_output_f32\": \"" << json_escape(manifest.preprocessed_output_f32) << "\",\n";
        std::cout << "  \"preprocessed_output_sha256\": \"" << json_escape(manifest.preprocessed_output_sha256) << "\",\n";
        std::cout << "  \"mel_manifest\": \"" << json_escape(manifest.mel_manifest) << "\",\n";
        std::cout << "  \"output_mel_f32\": \"" << json_escape(manifest.output_mel_f32) << "\",\n";
        std::cout << "  \"output_mel_sha256\": \"" << json_escape(manifest.output_mel_sha256) << "\",\n";
        std::cout << "  \"fbank_manifest\": \"" << json_escape(manifest.fbank_manifest) << "\",\n";
        std::cout << "  \"output_fbank_f32\": \"" << json_escape(manifest.output_fbank_f32) << "\",\n";
        std::cout << "  \"output_fbank_sha256\": \"" << json_escape(manifest.output_fbank_sha256) << "\",\n";
        std::cout << "  \"preprocessed_sample_rate\": " << manifest.preprocessed_sample_rate << ",\n";
        std::cout << "  \"preprocessed_samples\": " << manifest.preprocessed_samples << ",\n";
        std::cout << "  \"mel_frames\": " << manifest.mel_frames << ",\n";
        std::cout << "  \"fbank_frames\": " << manifest.fbank_frames << ",\n";
    }
    std::cout << "  \"clone_feature_readiness_issues\": ";
    print_json_string_array(issues);
    std::cout << ",\n";
    std::cout << "  \"ready_native_clone_audio_preprocess\": " << (ok && manifest.ready_native_clone_audio_preprocess ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_clone_mel_extraction\": " << (ok && manifest.ready_native_clone_mel_extraction ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_clone_fbank_extraction\": " << (ok && manifest.ready_native_clone_fbank_extraction ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_clone_feature_prep\": " << (ok && manifest.ready_native_clone_feature_prep ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_voice_clone\": false,\n";
    std::cout << "  \"required_voice_bundle_tensors_remaining\": [\n";
    std::cout << "    {\"name\": \"spk_cond_emb\", \"dtype\": \"f32\", \"shape\": \"[1,tokens>0,1024]\", \"source\": \"native semantic speech encoder and MaskGCT quantize\"},\n";
    std::cout << "    {\"name\": \"s2mel_style\", \"dtype\": \"f32\", \"shape\": \"[1,192]\", \"source\": \"native CAMPPlus style encoder\"},\n";
    std::cout << "    {\"name\": \"s2mel_prompt\", \"dtype\": \"f32\", \"shape\": \"[1,frames>0,512]\", \"source\": \"native S2Mel length regulator from MaskGCT S_ref\"}\n";
    std::cout << "  ],\n";
    std::cout << "  \"next_native_boundary\": \"clone-time semantic/acoustic speech encoder tensor creation\"\n";
    std::cout << "}\n";
    return ok;
}

void append_raw_f32_count_issue(std::vector<std::string>& issues,
                                const std::string& path,
                                uint64_t expected_values,
                                const std::string& name) {
    if (path.empty()) {
        issues.push_back(name + "_path_missing");
        return;
    }
    if (!std::filesystem::exists(path)) {
        issues.push_back(name + "_missing");
        return;
    }
    const uint64_t actual_bytes = static_cast<uint64_t>(std::filesystem::file_size(path));
    const uint64_t expected_bytes = expected_values * static_cast<uint64_t>(sizeof(float));
    if (actual_bytes != expected_bytes) {
        std::ostringstream issue;
        issue << name << "_size_mismatch: expected " << expected_values
              << " f32 values but file has " << (actual_bytes / sizeof(float));
        issues.push_back(issue.str());
    }
}

void append_raw_u32_count_issue(std::vector<std::string>& issues,
                                const std::string& path,
                                uint64_t expected_values,
                                const std::string& name) {
    if (path.empty()) {
        issues.push_back(name + "_path_missing");
        return;
    }
    if (!std::filesystem::exists(path)) {
        issues.push_back(name + "_missing");
        return;
    }
    const uint64_t actual_bytes = static_cast<uint64_t>(std::filesystem::file_size(path));
    const uint64_t expected_bytes = expected_values * static_cast<uint64_t>(sizeof(uint32_t));
    if (actual_bytes != expected_bytes) {
        std::ostringstream issue;
        issue << name << "_size_mismatch: expected " << expected_values
              << " u32 values but file has " << (actual_bytes / sizeof(uint32_t));
        issues.push_back(issue.str());
    }
}

[[maybe_unused]] bool inspect_tts_clone_encoder_readiness(const std::string& feature_manifest,
                                                          const std::string& spk_cond_f32,
                                                          uint32_t spk_tokens,
                                                          const std::string& s2mel_style_f32,
                                                          const std::string& s2mel_prompt_f32) {
    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    auto issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
    const uint64_t prompt_tokens = manifest.mel_frames;
    if (spk_tokens == 0) {
        issues.push_back("spk_tokens_must_be_positive");
    }
    if (prompt_tokens == 0) {
        issues.push_back("feature_manifest_mel_frames_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               spk_cond_f32,
                               static_cast<uint64_t>(spk_tokens) * 1024u,
                               "spk_cond_emb");
    append_raw_f32_count_issue(issues,
                               s2mel_style_f32,
                               192u,
                               "s2mel_style");
    append_raw_f32_count_issue(issues,
                               s2mel_prompt_f32,
                               prompt_tokens * 512u,
                               "s2mel_prompt");
    const bool ok = issues.empty();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_clone_encoder_readiness\",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
    std::cout << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
    std::cout << "  \"s2mel_style_f32\": \"" << json_escape(s2mel_style_f32) << "\",\n";
    std::cout << "  \"s2mel_prompt_f32\": \"" << json_escape(s2mel_prompt_f32) << "\",\n";
    std::cout << "  \"spk_tokens\": " << spk_tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"feature_mel_frames\": " << manifest.mel_frames << ",\n";
    std::cout << "  \"expected_tensors\": [\n";
    std::cout << "    {\"name\": \"spk_cond_emb\", \"dtype\": \"f32\", \"shape\": \"[1," << spk_tokens << ",1024]\", \"values\": " << (static_cast<uint64_t>(spk_tokens) * 1024u) << "},\n";
    std::cout << "    {\"name\": \"s2mel_style\", \"dtype\": \"f32\", \"shape\": \"[1,192]\", \"values\": 192},\n";
    std::cout << "    {\"name\": \"s2mel_prompt\", \"dtype\": \"f32\", \"shape\": \"[1," << prompt_tokens << ",512]\", \"values\": " << (prompt_tokens * 512u) << "}\n";
    std::cout << "  ],\n";
    std::cout << "  \"clone_encoder_readiness_issues\": ";
    print_json_string_array(issues);
    std::cout << ",\n";
    std::cout << "  \"ready_native_clone_feature_prep\": " << (ok && manifest.ready_native_clone_feature_prep ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_clone_encoder_outputs\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_voice_bundle_creation\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_voice_clone\": false,\n";
    std::cout << "  \"next_native_boundary\": \"write native MIT2 voice bundle from feature-prep plus encoder outputs\"\n";
    std::cout << "}\n";
    return ok;
}

[[maybe_unused]] bool inspect_tts_clone_campplus_style_readiness(const std::string& model_bundle_dir,
                                                                 const std::string& feature_manifest,
                                                                 const std::string& s2mel_style_f32) {
    std::vector<std::string> issues;
    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    auto feature_issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
    issues.insert(issues.end(), feature_issues.begin(), feature_issues.end());

    bool campplus_ok = false;
    size_t campplus_required = 0;
    size_t campplus_present = 0;
    std::vector<std::string> campplus_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto campplus = inspect_campplus_model_contract(model);
        campplus_ok = campplus.ok;
        campplus_required = campplus.specs.size();
        campplus_present = campplus.present_tensors.size();
        campplus_issues = campplus.issues;
        if (!campplus.ok) {
            issues.push_back("campplus_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        campplus_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    append_raw_f32_count_issue(issues,
                               s2mel_style_f32,
                               192u,
                               "s2mel_style");
    const bool ok = issues.empty();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_clone_campplus_style_readiness\",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
    std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
    std::cout << "  \"s2mel_style_f32\": \"" << json_escape(s2mel_style_f32) << "\",\n";
    std::cout << "  \"output_fbank_f32\": \"" << json_escape(manifest.output_fbank_f32) << "\",\n";
    std::cout << "  \"fbank_frames\": " << manifest.fbank_frames << ",\n";
    std::cout << "  \"fbank_values\": " << (manifest.fbank_frames * 80u) << ",\n";
    std::cout << "  \"expected_style\": {\"name\": \"s2mel_style\", \"dtype\": \"f32\", \"shape\": \"[1,192]\", \"values\": 192},\n";
    std::cout << "  \"has_campplus_model_contract\": " << (campplus_ok ? "true" : "false") << ",\n";
    std::cout << "  \"campplus_required_tensor_count\": " << campplus_required << ",\n";
    std::cout << "  \"campplus_required_tensors_present\": " << campplus_present << ",\n";
    std::cout << "  \"campplus_contract_issues\": ";
    print_json_string_array(campplus_issues);
    std::cout << ",\n";
    std::cout << "  \"clone_campplus_style_readiness_issues\": ";
    print_json_string_array(issues);
    std::cout << ",\n";
    std::cout << "  \"ready_native_clone_feature_prep\": " << (ok && manifest.ready_native_clone_feature_prep ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_clone_fbank_extraction\": " << (ok && manifest.ready_native_clone_fbank_extraction ? "true" : "false") << ",\n";
    std::cout << "  \"ready_reference_campplus_style\": " << (ok ? "true" : "false") << ",\n";
    std::cout << "  \"ready_native_campplus_style_forward\": false,\n";
    std::cout << "  \"ready_native_voice_clone\": false,\n";
    std::cout << "  \"next_native_boundary\": \"run --clone-campplus-style-from-features for native s2mel_style generation, then native W2V-BERT semantic features for spk_cond_emb\"\n";
    std::cout << "}\n";
    return ok;
}

