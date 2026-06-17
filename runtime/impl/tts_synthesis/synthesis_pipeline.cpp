// Hot-path TTS synthesis orchestration: shared MetalContext, GPT code decode, condition/acoustic vector synthesis, CJK text -> wav (seeded/sampled/segmented), presets, and golden tests.
// Part of the tts_synthesis translation unit (see ../tts_synthesis.cpp);
// these files are #included in order into a single TU, so definition order
// across files is significant.

// Per-request stage accumulators for the HTTP server's console summary.
struct TtsStageAccumulators {
    double frontend = 0.0;
    double gpt = 0.0;
    double condition = 0.0;
    double acoustic = 0.0;
    void reset() { frontend = gpt = condition = acoustic = 0.0; }
};
TtsStageAccumulators g_tts_stage_acc;

// Process-wide shared MetalContext for the hot TTS path: avoids recompiling the
// Metal library and re-uploading resident weights for every segment x stage.
mit2::MetalContext& hot_shared_metal() {
    static mit2::MetalContext ctx;
    return ctx;
}

static mit2::MetalResourceStats metal_stats_delta(const mit2::MetalResourceStats& a,
                                                  const mit2::MetalResourceStats& b) {
    mit2::MetalResourceStats d;
    d.command_buffers_submitted = b.command_buffers_submitted - a.command_buffers_submitted;
    d.buffer_allocations = b.buffer_allocations - a.buffer_allocations;
    d.buffer_bytes_allocated = b.buffer_bytes_allocated - a.buffer_bytes_allocated;
    d.gpu_elapsed_seconds = b.gpu_elapsed_seconds - a.gpu_elapsed_seconds;
    return d;
}

bool run_hot_tts_golden_test(const std::string& bundle_dir, const std::string& s2mel_golden_dir, const std::string& wave_golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    auto noise = read_raw_f32(s2mel_golden_dir + "/noise.f32");
    auto prompt = read_raw_f32(s2mel_golden_dir + "/prompt_mel.f32");
    auto cond = read_raw_f32(s2mel_golden_dir + "/condition.f32");
    auto style = read_raw_f32(s2mel_golden_dir + "/style.f32");
    auto golden_mel = read_raw_f32(s2mel_golden_dir + "/s2mel_generated.f32");
    auto golden_wave = read_raw_f32(wave_golden_dir + "/waveform.f32");
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS golden noise must have shape [tokens,80]");
    }
    if (prompt.empty() || (prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS golden prompt_mel must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(prompt.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("Hot TTS prompt_tokens exceeds total tokens");
    }
    const uint32_t generated_tokens = tokens - prompt_tokens;
    if (cond.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS condition must have shape [tokens,512]");
    }
    if (style.size() != style_dim) {
        throw std::runtime_error("Hot TTS style must have shape [192]");
    }
    if (golden_mel.size() != static_cast<size_t>(generated_tokens) * mel_dim) {
        throw std::runtime_error("Hot TTS generated mel golden output size mismatch");
    }
    auto full_mel = run_cfm_euler_metal(metal, bundle, noise, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> generated_mel(golden_mel.size());
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
    auto wave = run_bigvgan_vocoder_metal(metal, bundle, generated_mel, generated_tokens);
    if (wave.size() != golden_wave.size()) {
        throw std::runtime_error("Hot TTS waveform golden output size mismatch");
    }
    const float mel_err = max_abs_error(generated_mel, golden_mel);
    const float wave_err = max_abs_error(wave, golden_wave);
    const float err = std::max(mel_err, wave_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_golden\",\n";
    std::cout << "  \"s2mel_golden_dir\": \"" << s2mel_golden_dir << "\",\n";
    std::cout << "  \"wave_golden_dir\": \"" << wave_golden_dir << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"samples\": " << wave.size() << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"mel_max_abs_error\": " << mel_err << ",\n";
    std::cout << "  \"wave_max_abs_error\": " << wave_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return mel_err <= 3e-3f && wave_err <= 1e-3f;
}

bool run_hot_tts_from_gpt_golden_test(const std::string& bundle_dir,
                                      const std::string& voice_bundle_dir,
                                      const std::string& gpt_golden_dir,
                                      const std::string& s2mel_golden_dir,
                                      const std::string& wave_golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t in_dim = 1024;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    auto s_infer = read_raw_f32(gpt_golden_dir + "/s_infer.f32");
    auto target_lengths = read_raw_u32(gpt_golden_dir + "/target_lengths.u32");
    auto golden_lr = read_raw_f32(gpt_golden_dir + "/length_regulator.f32");
    auto noise = read_raw_f32(s2mel_golden_dir + "/noise.f32");
    auto golden_prompt = read_raw_f32(s2mel_golden_dir + "/prompt_mel.f32");
    auto golden_condition = read_raw_f32(s2mel_golden_dir + "/condition.f32");
    auto golden_mel = read_raw_f32(s2mel_golden_dir + "/s2mel_generated.f32");
    auto golden_wave = read_raw_f32(wave_golden_dir + "/waveform.f32");
    if (s_infer.empty() || (s_infer.size() % in_dim) != 0) {
        throw std::runtime_error("Hot TTS GPT golden s_infer must have shape [tokens,1024]");
    }
    if (target_lengths.size() != 1 || target_lengths[0] == 0) {
        throw std::runtime_error("Hot TTS GPT golden target_lengths must contain one positive length");
    }
    const uint32_t gpt_tokens = static_cast<uint32_t>(s_infer.size() / in_dim);
    const uint32_t generated_tokens = target_lengths[0];
    if (golden_lr.size() != static_cast<size_t>(generated_tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS GPT golden length_regulator output size mismatch");
    }
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS S2Mel golden noise must have shape [tokens,80]");
    }
    if (golden_prompt.empty() || (golden_prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS S2Mel golden prompt_mel must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(golden_prompt.size() / mel_dim);
    if (tokens != prompt_tokens + generated_tokens) {
        throw std::runtime_error("Hot TTS token counts do not match prompt + generated lengths");
    }
    if (golden_condition.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS S2Mel golden condition must have shape [tokens,512]");
    }
    if (golden_mel.size() != static_cast<size_t>(generated_tokens) * mel_dim) {
        throw std::runtime_error("Hot TTS generated mel golden output size mismatch");
    }

    auto lr = run_length_regulator_full_metal(metal, bundle, s_infer, gpt_tokens, generated_tokens);

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
    auto condition = metal.hot_condition_merge_f32(voice_prompt_all, lr, prompt_tokens, generated_tokens, cond_dim);

    auto full_mel = run_cfm_euler_metal(metal, bundle, noise, prompt, condition, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> generated_mel(golden_mel.size());
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
    auto wave = run_bigvgan_vocoder_metal(metal, bundle, generated_mel, generated_tokens);
    if (wave.size() != golden_wave.size()) {
        throw std::runtime_error("Hot TTS waveform golden output size mismatch");
    }
    const auto stats = metal.resource_stats();

    const float lr_err = max_abs_error(lr, golden_lr);
    const float prompt_err = max_abs_error(prompt, golden_prompt);
    const float condition_err = max_abs_error(condition, golden_condition);
    const float mel_err = max_abs_error(generated_mel, golden_mel);
    const float wave_err = max_abs_error(wave, golden_wave);
    const float err = std::max({lr_err, prompt_err, condition_err, mel_err, wave_err});
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_from_gpt_golden\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"gpt_golden_dir\": \"" << gpt_golden_dir << "\",\n";
    std::cout << "  \"s2mel_golden_dir\": \"" << s2mel_golden_dir << "\",\n";
    std::cout << "  \"wave_golden_dir\": \"" << wave_golden_dir << "\",\n";
    std::cout << "  \"gpt_tokens\": " << gpt_tokens << ",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"samples\": " << wave.size() << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"length_regulator_max_abs_error\": " << lr_err << ",\n";
    std::cout << "  \"prompt_mel_max_abs_error\": " << prompt_err << ",\n";
    std::cout << "  \"condition_max_abs_error\": " << condition_err << ",\n";
    std::cout << "  \"mel_max_abs_error\": " << mel_err << ",\n";
    std::cout << "  \"mel_tolerance\": 0.006,\n";
    std::cout << "  \"wave_max_abs_error\": " << wave_err << ",\n";
    std::cout << "  \"wave_tolerance\": 0.01,\n";
    print_metal_resource_stats_json("metal_", stats);
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return lr_err <= 3e-3f && prompt_err <= 3e-3f && condition_err <= 3e-3f && mel_err <= 6e-3f && wave_err <= 1e-2f;
}

bool synthesize_hot_gpt_golden_wav(const std::string& bundle_dir,
                                   const std::string& voice_bundle_dir,
                                   const std::string& gpt_golden_dir,
                                   const std::string& s2mel_golden_dir,
                                   const std::string& output_wav) {
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t in_dim = 1024;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    constexpr uint32_t sample_rate = 22050;
    auto s_infer = read_raw_f32(gpt_golden_dir + "/s_infer.f32");
    auto target_lengths = read_raw_u32(gpt_golden_dir + "/target_lengths.u32");
    auto noise = read_raw_f32(s2mel_golden_dir + "/noise.f32");
    auto golden_prompt = read_raw_f32(s2mel_golden_dir + "/prompt_mel.f32");
    if (s_infer.empty() || (s_infer.size() % in_dim) != 0) {
        throw std::runtime_error("Hot GPT synth s_infer must have shape [tokens,1024]");
    }
    if (target_lengths.size() != 1 || target_lengths[0] == 0) {
        throw std::runtime_error("Hot GPT synth target_lengths must contain one positive length");
    }
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot GPT synth noise must have shape [tokens,80]");
    }
    if (golden_prompt.empty() || (golden_prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot GPT synth prompt_mel must have shape [prompt_tokens,80]");
    }

    const uint32_t gpt_tokens = static_cast<uint32_t>(s_infer.size() / in_dim);
    const uint32_t generated_tokens = target_lengths[0];
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(golden_prompt.size() / mel_dim);
    if (tokens != prompt_tokens + generated_tokens) {
        throw std::runtime_error("Hot GPT synth token counts do not match prompt + generated lengths");
    }

    auto lr = run_length_regulator_full_metal(metal, bundle, s_infer, gpt_tokens, generated_tokens);

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
    auto condition = metal.hot_condition_merge_f32(voice_prompt_all, lr, prompt_tokens, generated_tokens, cond_dim);

    auto full_mel = run_cfm_euler_metal(metal, bundle, noise, prompt, condition, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> generated_mel(static_cast<size_t>(generated_tokens) * mel_dim);
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
    auto wave = run_bigvgan_vocoder_metal(metal, bundle, generated_mel, generated_tokens);
    const auto stats = metal.resource_stats();
    write_wav_pcm16(output_wav, wave, sample_rate);

    float peak_abs = 0.0f;
    for (float sample : wave) {
        peak_abs = std::max(peak_abs, std::abs(sample));
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_from_gpt_wav\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"gpt_golden_dir\": \"" << gpt_golden_dir << "\",\n";
    std::cout << "  \"s2mel_golden_dir\": \"" << s2mel_golden_dir << "\",\n";
    std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
    std::cout << "  \"gpt_tokens\": " << gpt_tokens << ",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"samples\": " << wave.size() << ",\n";
    std::cout << "  \"sample_rate\": " << sample_rate << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"peak_abs\": " << peak_abs << ",\n";
    print_metal_resource_stats_json("", stats, false);
    std::cout << "}\n";
    return true;
}

bool run_hot_tts_condition_golden_test(const std::string& bundle_dir,
                                       const std::string& voice_bundle_dir,
                                       const std::string& s2mel_golden_dir,
                                       const std::string& wave_golden_dir,
                                       const std::string& condition_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    auto noise = read_raw_f32(s2mel_golden_dir + "/noise.f32");
    auto golden_prompt = read_raw_f32(s2mel_golden_dir + "/prompt_mel.f32");
    auto golden_condition = read_raw_f32(s2mel_golden_dir + "/condition.f32");
    auto condition = read_raw_f32(condition_path);
    auto golden_mel = read_raw_f32(s2mel_golden_dir + "/s2mel_generated.f32");
    auto golden_wave = read_raw_f32(wave_golden_dir + "/waveform.f32");
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS condition noise must have shape [tokens,80]");
    }
    if (golden_prompt.empty() || (golden_prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS condition prompt_mel must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(golden_prompt.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("Hot TTS condition prompt_tokens exceeds total tokens");
    }
    const uint32_t generated_tokens = tokens - prompt_tokens;
    if (condition.size() != static_cast<size_t>(tokens) * cond_dim ||
        golden_condition.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS condition tensors must have shape [tokens,512]");
    }
    if (golden_mel.size() != static_cast<size_t>(generated_tokens) * mel_dim) {
        throw std::runtime_error("Hot TTS condition generated mel golden output size mismatch");
    }

    const auto* mel_info = voice.find("mel");
    if (!mel_info || mel_info->shape.size() != 3 || mel_info->shape[0] != 1 || mel_info->shape[1] != mel_dim ||
        static_cast<uint32_t>(mel_info->shape[2]) < prompt_tokens) {
        throw std::runtime_error("voice mel must have shape [1,80,prompt_tokens>=requested]");
    }
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

    auto full_mel = run_cfm_euler_metal(metal, bundle, noise, prompt, condition, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> generated_mel(golden_mel.size());
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
    auto wave = run_bigvgan_vocoder_metal(metal, bundle, generated_mel, generated_tokens);
    if (wave.size() != golden_wave.size()) {
        throw std::runtime_error("Hot TTS condition waveform golden output size mismatch");
    }
    const float condition_err = max_abs_error(condition, golden_condition);
    const float prompt_err = max_abs_error(prompt, golden_prompt);
    const float mel_err = max_abs_error(generated_mel, golden_mel);
    const float wave_err = max_abs_error(wave, golden_wave);
    const float err = std::max({condition_err, prompt_err, mel_err, wave_err});
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_condition_golden\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"s2mel_golden_dir\": \"" << s2mel_golden_dir << "\",\n";
    std::cout << "  \"wave_golden_dir\": \"" << wave_golden_dir << "\",\n";
    std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"samples\": " << wave.size() << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"condition_max_abs_error\": " << condition_err << ",\n";
    std::cout << "  \"prompt_mel_max_abs_error\": " << prompt_err << ",\n";
    std::cout << "  \"mel_max_abs_error\": " << mel_err << ",\n";
    std::cout << "  \"wave_max_abs_error\": " << wave_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return condition_err <= 3e-3f && prompt_err <= 3e-3f && mel_err <= 3e-3f && wave_err <= 1e-3f;
}

bool synthesize_hot_condition_golden_wav(const std::string& bundle_dir,
                                         const std::string& voice_bundle_dir,
                                         const std::string& s2mel_golden_dir,
                                         const std::string& condition_path,
                                         const std::string& output_wav) {
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    constexpr uint32_t sample_rate = 22050;
    auto noise = read_raw_f32(s2mel_golden_dir + "/noise.f32");
    auto golden_prompt = read_raw_f32(s2mel_golden_dir + "/prompt_mel.f32");
    auto condition = read_raw_f32(condition_path);
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS condition synth noise must have shape [tokens,80]");
    }
    if (golden_prompt.empty() || (golden_prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS condition synth prompt_mel must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(golden_prompt.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("Hot TTS condition synth prompt_tokens exceeds total tokens");
    }
    const uint32_t generated_tokens = tokens - prompt_tokens;
    if (condition.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS condition synth tensor must have shape [tokens,512]");
    }

    const auto* mel_info = voice.find("mel");
    if (!mel_info || mel_info->shape.size() != 3 || mel_info->shape[0] != 1 || mel_info->shape[1] != mel_dim ||
        static_cast<uint32_t>(mel_info->shape[2]) < prompt_tokens) {
        throw std::runtime_error("voice mel must have shape [1,80,prompt_tokens>=requested]");
    }
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

    auto full_mel = run_cfm_euler_metal(metal, bundle, noise, prompt, condition, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> generated_mel(static_cast<size_t>(generated_tokens) * mel_dim);
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
    auto wave = run_bigvgan_vocoder_metal(metal, bundle, generated_mel, generated_tokens);
    const auto stats = metal.resource_stats();
    write_wav_pcm16(output_wav, wave, sample_rate);
    float peak_abs = 0.0f;
    for (float sample : wave) {
        peak_abs = std::max(peak_abs, std::abs(sample));
    }
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_condition_wav\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"s2mel_golden_dir\": \"" << s2mel_golden_dir << "\",\n";
    std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"samples\": " << wave.size() << ",\n";
    std::cout << "  \"sample_rate\": " << sample_rate << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"peak_abs\": " << peak_abs << ",\n";
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << "\n";
    std::cout << "}\n";
    return true;
}

bool synthesize_hot_condition_inputs_wav(const std::string& bundle_dir,
                                         const std::string& voice_bundle_dir,
                                         const std::string& condition_path,
                                         const std::string& noise_path,
                                         uint32_t prompt_tokens,
                                         uint32_t steps,
                                         float cfg_rate,
                                         const std::string& output_wav) {
    const auto started = Clock::now();
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t sample_rate = 22050;
    auto noise = read_raw_f32(noise_path);
    auto condition = read_raw_f32(condition_path);
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS condition input synth noise must have shape [tokens,80]");
    }
    if (prompt_tokens == 0) {
        throw std::runtime_error("Hot TTS condition input synth prompt_tokens must be positive");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("Hot TTS condition input synth prompt_tokens exceeds total tokens");
    }
    if (steps == 0) {
        throw std::runtime_error("Hot TTS condition input synth steps must be positive");
    }
    const uint32_t generated_tokens = tokens - prompt_tokens;
    if (condition.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS condition input synth condition must have shape [tokens,512]");
    }

    const auto* mel_info = voice.find("mel");
    if (!mel_info || mel_info->shape.size() != 3 || mel_info->shape[0] != 1 || mel_info->shape[1] != mel_dim ||
        static_cast<uint32_t>(mel_info->shape[2]) < prompt_tokens) {
        throw std::runtime_error("voice mel must have shape [1,80,prompt_tokens>=requested]");
    }
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

    auto full_mel = run_cfm_euler_metal(metal, bundle, noise, prompt, condition, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> generated_mel(static_cast<size_t>(generated_tokens) * mel_dim);
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
    auto wave = run_bigvgan_vocoder_metal(metal, bundle, generated_mel, generated_tokens);
    const auto stats = metal.resource_stats();
    write_wav_pcm16(output_wav, wave, sample_rate);
    float peak_abs = 0.0f;
    for (float sample : wave) {
        peak_abs = std::max(peak_abs, std::abs(sample));
    }
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_condition_inputs_wav\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"noise_f32\": \"" << noise_path << "\",\n";
    std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"samples\": " << wave.size() << ",\n";
    std::cout << "  \"sample_rate\": " << sample_rate << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"peak_abs\": " << peak_abs << ",\n";
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
    std::cout << "  \"elapsed_seconds\": " << seconds_since(started) << "\n";
    std::cout << "}\n";
    return true;
}

bool synthesize_hot_inputs_wav(const std::string& bundle_dir,
                               const std::string& voice_bundle_dir,
                               const std::string& conds_path,
                               const std::string& text_ids_path,
                               uint32_t max_codes,
                               const std::string& noise_path,
                               uint32_t prompt_tokens,
                               uint32_t steps,
                               float cfg_rate,
                               const std::string& output_wav) {
    const std::string codes_path = output_wav + ".codes.u32";
    const std::string condition_path = output_wav + ".condition.f32";
    if (!export_gpt_kv_codes_inputs(bundle_dir, conds_path, text_ids_path, max_codes, codes_path)) {
        return false;
    }
    if (!export_hot_codes_condition_inputs(bundle_dir, voice_bundle_dir, conds_path, text_ids_path, codes_path, prompt_tokens, condition_path)) {
        return false;
    }
    if (!synthesize_hot_condition_inputs_wav(bundle_dir, voice_bundle_dir, condition_path, noise_path, prompt_tokens, steps, cfg_rate, output_wav)) {
        return false;
    }
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_inputs_wav\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"noise_f32\": \"" << noise_path << "\",\n";
    std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
    std::cout << "  \"codes_u32\": \"" << codes_path << "\",\n";
    std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << "\n";
    std::cout << "}\n";
    return true;
}

struct HotConditionInputs {
    std::vector<float> condition;
    uint32_t generated_tokens = 0;
    uint32_t tokens = 0;
};

struct HotWavMetrics {
    uint32_t generated_tokens = 0;
    uint32_t tokens = 0;
    size_t samples = 0;
    uint32_t sample_rate = 22050;
    float peak_abs = 0.0f;
};

// Trims a trailing repeating cycle from GPT-generated codes when no EOS was found.
// Greedy decoding without repetition_penalty tends to loop after real content ends.
// Returns the trimmed size; leaves the vector unchanged if no substantial repeat is found.
static size_t trim_code_repeating_tail(const std::vector<uint32_t>& codes) {
    const size_t n = codes.size();
    for (uint32_t period = 1; period <= 8; ++period) {
        if (n < static_cast<size_t>(period) * 8) continue;
        size_t repeat_start = n;
        while (repeat_start > period &&
               codes[repeat_start - 1] == codes[repeat_start - 1 - period]) {
            --repeat_start;
        }
        const size_t repeat_len = n - repeat_start;
        // Require at least 6 full periods to avoid false positives
        if (repeat_len >= static_cast<size_t>(period) * 6) {
            return repeat_start + period;
        }
    }
    return n;
}

std::vector<uint32_t> run_gpt_kv_codes_inputs_shared(mit2::MetalContext& metal,
                                                     const mit2::Bundle& bundle,
                                                     const std::vector<float>& conds,
                                                     const std::vector<uint32_t>& text_ids,
                                                     uint32_t max_codes,
                                                     uint32_t* raw_codes,
                                                     int* first_stop_step,
                                                     uint32_t* prefix_tokens) {
    constexpr uint32_t width = 1280;
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("GPT input code export conds must have shape [tokens,1280]");
    }
    if (max_codes == 0) {
        throw std::runtime_error("GPT input code export max_codes must be positive");
    }
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_slots = static_cast<uint32_t>(text_ids.size());
    auto prefix = run_gpt_prepare_inputs_metal(metal, bundle, conds, text_ids, cond_tokens, text_slots);
    const uint32_t local_prefix_tokens = static_cast<uint32_t>(prefix.inputs_embeds.size() / width);
    auto generated = run_gpt_kv_greedy_metal(metal, bundle, prefix.inputs_embeds, local_prefix_tokens, max_codes, true);
    std::vector<uint32_t> codes = generated.predicted_tokens;
    if (generated.first_stop_step >= 0) {
        codes.resize(static_cast<size_t>(generated.first_stop_step));
    } else {
        codes.resize(trim_code_repeating_tail(codes));
    }
    if (raw_codes) {
        *raw_codes = static_cast<uint32_t>(generated.predicted_tokens.size());
    }
    if (first_stop_step) {
        *first_stop_step = generated.first_stop_step;
    }
    if (prefix_tokens) {
        *prefix_tokens = local_prefix_tokens;
    }
    return codes;
}

bool run_gpt_kv_codes_inputs_test(const std::string& bundle_dir,
                                  const std::string& conds_path,
                                  const std::string& text_ids_path,
                                  uint32_t max_codes,
                                  const std::string& expected_codes_path) {
    const auto started = Clock::now();
    constexpr uint32_t gpt_width = 1280;
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    auto conds = read_raw_f32(conds_path);
    auto text_ids = read_raw_u32(text_ids_path);
    auto expected = read_raw_u32(expected_codes_path);
    if (conds.empty() || (conds.size() % gpt_width) != 0) {
        throw std::runtime_error("GPT input code test conds must have shape [tokens,1280]");
    }
    if (expected.empty()) {
        throw std::runtime_error("GPT input code test expected codes must be non-empty");
    }
    if (max_codes == 0) {
        throw std::runtime_error("GPT input code test max_codes must be positive");
    }

    uint32_t raw_codes = 0;
    uint32_t prefix_tokens = 0;
    int first_stop_step = -1;
    const auto codes = run_gpt_kv_codes_inputs_shared(metal,
                                                      bundle,
                                                      conds,
                                                      text_ids,
                                                      max_codes,
                                                      &raw_codes,
                                                      &first_stop_step,
                                                      &prefix_tokens);
    const bool codes_match = codes == expected;
    size_t first_mismatch = std::numeric_limits<size_t>::max();
    const size_t compare = std::min(codes.size(), expected.size());
    for (size_t i = 0; i < compare; ++i) {
        if (codes[i] != expected[i]) {
            first_mismatch = i;
            break;
        }
    }
    if (first_mismatch == std::numeric_limits<size_t>::max() && codes.size() != expected.size()) {
        first_mismatch = compare;
    }
    const auto stats = metal.resource_stats();
    const double elapsed = seconds_since(started);

    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_codes_inputs_test\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"conds_f32\": \"" << json_escape(conds_path) << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << json_escape(text_ids_path) << "\",\n";
    std::cout << "  \"expected_codes_u32\": \"" << json_escape(expected_codes_path) << "\",\n";
    std::cout << "  \"cond_tokens\": " << (conds.size() / gpt_width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"raw_codes\": " << raw_codes << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"expected_codes_count\": " << expected.size() << ",\n";
    std::cout << "  \"first_stop_step\": " << first_stop_step << ",\n";
    if (first_mismatch == std::numeric_limits<size_t>::max()) {
        std::cout << "  \"first_mismatch\": null,\n";
    } else {
        std::cout << "  \"first_mismatch\": " << first_mismatch << ",\n";
    }
    std::cout << "  \"predicted_codes\": ";
    print_json_u32_array(codes);
    std::cout << ",\n";
    std::cout << "  \"expected_codes\": ";
    print_json_u32_array(expected);
    std::cout << ",\n";
    std::cout << "  \"codes_match_expected\": " << (codes_match ? "true" : "false") << ",\n";
    print_gpt_decode_rate_json(raw_codes, codes.size(), elapsed);
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"elapsed_seconds\": " << elapsed << "\n";
    std::cout << "}\n";
    return codes_match;
}

std::vector<uint32_t> run_gpt_kv_codes_inputs_sampled_shared(mit2::MetalContext& metal,
                                                             const mit2::Bundle& bundle,
                                                             const std::vector<float>& conds,
                                                             const std::vector<uint32_t>& text_ids,
                                                             uint32_t max_codes,
                                                             const GptSamplingConfig& sampling,
                                                             uint32_t* raw_codes,
                                                             int* first_stop_step,
                                                             uint32_t* prefix_tokens) {
    constexpr uint32_t width = 1280;
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("GPT sampled input code export conds must have shape [tokens,1280]");
    }
    if (max_codes == 0) {
        throw std::runtime_error("GPT sampled input code export max_codes must be positive");
    }
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_slots = static_cast<uint32_t>(text_ids.size());
    auto prefix = run_gpt_prepare_inputs_metal(metal, bundle, conds, text_ids, cond_tokens, text_slots);
    const uint32_t local_prefix_tokens = static_cast<uint32_t>(prefix.inputs_embeds.size() / width);
    auto generated = run_gpt_kv_greedy_metal(metal,
                                             bundle,
                                             prefix.inputs_embeds,
                                             local_prefix_tokens,
                                             max_codes,
                                             true,
                                             &sampling,
                                             &prefix.fake_inputs);
    std::vector<uint32_t> codes = generated.predicted_tokens;
    if (generated.first_stop_step >= 0) {
        codes.resize(static_cast<size_t>(generated.first_stop_step));
    }
    if (raw_codes) {
        *raw_codes = static_cast<uint32_t>(generated.predicted_tokens.size());
    }
    if (first_stop_step) {
        *first_stop_step = generated.first_stop_step;
    }
    if (prefix_tokens) {
        *prefix_tokens = local_prefix_tokens;
    }
    return codes;
}

bool export_gpt_kv_codes_inputs_sampled(const std::string& bundle_dir,
                                        const std::string& conds_path,
                                        const std::string& text_ids_path,
                                        uint32_t max_codes,
                                        const GptSamplingConfig& sampling,
                                        const std::string& output_codes_path) {
    const auto started = Clock::now();
    constexpr uint32_t gpt_width = 1280;
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    auto conds = read_raw_f32(conds_path);
    auto text_ids = read_raw_u32(text_ids_path);
    if (conds.empty() || (conds.size() % gpt_width) != 0) {
        throw std::runtime_error("GPT sampled input code export conds must have shape [tokens,1280]");
    }
    uint32_t raw_codes = 0;
    uint32_t prefix_tokens = 0;
    int first_stop_step = -1;
    auto codes = run_gpt_kv_codes_inputs_sampled_shared(metal,
                                                        bundle,
                                                        conds,
                                                        text_ids,
                                                        max_codes,
                                                        sampling,
                                                        &raw_codes,
                                                        &first_stop_step,
                                                        &prefix_tokens);
    write_raw_u32(output_codes_path, codes);
    const auto stats = metal.resource_stats();
    const double elapsed = seconds_since(started);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_codes_inputs_sampled_export\",\n";
    std::cout << "  \"bundle_dir\": \"" << bundle_dir << "\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"output_codes_u32\": \"" << output_codes_path << "\",\n";
    std::cout << "  \"cond_tokens\": " << (conds.size() / gpt_width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"raw_codes\": " << raw_codes << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"first_stop_step\": " << first_stop_step << ",\n";
    std::cout << "  \"seed\": " << sampling.seed << ",\n";
    std::cout << "  \"temperature\": " << sampling.temperature << ",\n";
    std::cout << "  \"top_k\": " << sampling.top_k << ",\n";
    std::cout << "  \"top_p\": " << sampling.top_p << ",\n";
    std::cout << "  \"repetition_penalty\": " << sampling.repetition_penalty << ",\n";
    std::cout << "  \"predicted_codes\": [";
    for (size_t i = 0; i < codes.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << codes[i];
    }
    std::cout << "],\n";
    print_gpt_decode_rate_json(raw_codes, codes.size(), elapsed);
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"elapsed_seconds\": " << elapsed << "\n";
    std::cout << "}\n";
    return !codes.empty() || first_stop_step == 0;
}

bool run_gpt_sampled_inputs_determinism_test(const std::string& bundle_dir,
                                             const std::string& conds_path,
                                             const std::string& text_ids_path,
                                             uint32_t max_codes) {
    const auto started = Clock::now();
    constexpr uint32_t gpt_width = 1280;
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    auto conds = read_raw_f32(conds_path);
    auto text_ids = read_raw_u32(text_ids_path);
    if (conds.empty() || (conds.size() % gpt_width) != 0) {
        throw std::runtime_error("GPT sampled determinism test conds must have shape [tokens,1280]");
    }
    if (max_codes == 0) {
        throw std::runtime_error("GPT sampled determinism test max_codes must be positive");
    }

    GptSamplingConfig sampling;
    sampling.do_sample = true;
    sampling.seed = 20240605;
    sampling.temperature = 0.8f;
    sampling.top_k = 30;
    sampling.top_p = 0.8f;
    sampling.repetition_penalty = 10.0f;

    uint32_t raw_a = 0;
    uint32_t raw_b = 0;
    uint32_t raw_c = 0;
    uint32_t prefix_a = 0;
    uint32_t prefix_b = 0;
    uint32_t prefix_c = 0;
    int first_stop_a = -1;
    int first_stop_b = -1;
    int first_stop_c = -1;

    const auto codes_a = run_gpt_kv_codes_inputs_sampled_shared(metal,
                                                                 bundle,
                                                                 conds,
                                                                 text_ids,
                                                                 max_codes,
                                                                 sampling,
                                                                 &raw_a,
                                                                 &first_stop_a,
                                                                 &prefix_a);
    const auto codes_b = run_gpt_kv_codes_inputs_sampled_shared(metal,
                                                                 bundle,
                                                                 conds,
                                                                 text_ids,
                                                                 max_codes,
                                                                 sampling,
                                                                 &raw_b,
                                                                 &first_stop_b,
                                                                 &prefix_b);
    GptSamplingConfig other_sampling = sampling;
    other_sampling.seed = 20240606;
    const auto codes_c = run_gpt_kv_codes_inputs_sampled_shared(metal,
                                                                 bundle,
                                                                 conds,
                                                                 text_ids,
                                                                 max_codes,
                                                                 other_sampling,
                                                                 &raw_c,
                                                                 &first_stop_c,
                                                                 &prefix_c);

    const bool same_seed_ok = codes_a == codes_b && raw_a == raw_b &&
                              first_stop_a == first_stop_b && prefix_a == prefix_b;
    const bool nonempty_ok = !codes_a.empty() || first_stop_a == 0;
    const bool prefix_ok = prefix_a == prefix_b && prefix_a == prefix_c;
    const bool different_seed_changed = codes_a != codes_c;
    const bool ok = same_seed_ok && nonempty_ok && prefix_ok;
    const auto stats = metal.resource_stats();
    const double elapsed = seconds_since(started);
    const uint32_t total_raw_codes = raw_a + raw_b + raw_c;
    const size_t total_output_codes = codes_a.size() + codes_b.size() + codes_c.size();

    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_sampled_inputs_determinism\",\n";
    std::cout << "  \"bundle_dir\": \"" << bundle_dir << "\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"cond_tokens\": " << (conds.size() / gpt_width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_a << ",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"seed\": " << sampling.seed << ",\n";
    std::cout << "  \"other_seed\": " << other_sampling.seed << ",\n";
    std::cout << "  \"temperature\": " << sampling.temperature << ",\n";
    std::cout << "  \"top_k\": " << sampling.top_k << ",\n";
    std::cout << "  \"top_p\": " << sampling.top_p << ",\n";
    std::cout << "  \"repetition_penalty\": " << sampling.repetition_penalty << ",\n";
    std::cout << "  \"raw_codes\": " << raw_a << ",\n";
    std::cout << "  \"first_stop_step\": " << first_stop_a << ",\n";
    std::cout << "  \"codes_seed_a\": ";
    print_json_u32_array(codes_a);
    std::cout << ",\n";
    std::cout << "  \"codes_seed_a_repeat\": ";
    print_json_u32_array(codes_b);
    std::cout << ",\n";
    std::cout << "  \"codes_seed_b\": ";
    print_json_u32_array(codes_c);
    std::cout << ",\n";
    std::cout << "  \"same_seed_ok\": " << (same_seed_ok ? "true" : "false") << ",\n";
    std::cout << "  \"different_seed_changed\": " << (different_seed_changed ? "true" : "false") << ",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    print_gpt_decode_rate_json(total_raw_codes, total_output_codes, elapsed);
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"elapsed_seconds\": " << elapsed << "\n";
    std::cout << "}\n";
    return ok;
}

HotConditionInputs build_hot_codes_condition_inputs_shared(mit2::MetalContext& metal,
                                                           const mit2::Bundle& bundle,
                                                           const mit2::Bundle& voice,
                                                           const std::vector<float>& conds,
                                                           const std::vector<uint32_t>& text_ids,
                                                           const std::vector<uint32_t>& codes,
                                                           uint32_t prompt_tokens) {
    constexpr uint32_t gpt_width = 1280;
    constexpr uint32_t semantic_dim = 1024;
    constexpr uint32_t cond_dim = 512;
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

    auto latent = run_gpt_latent_forward_metal(metal, bundle, conds, text_ids, codes);
    auto gpt_layer = run_gpt_layer_metal(metal, bundle, latent, static_cast<uint32_t>(codes.size()));
    auto vq = run_vq2emb_metal(metal, bundle, codes);
    if (gpt_layer.size() != static_cast<size_t>(codes.size()) * semantic_dim || vq.size() != gpt_layer.size()) {
        throw std::runtime_error("Hot TTS input condition export semantic tensor size mismatch");
    }
    auto s_infer = metal.add_f32(gpt_layer, vq);
    auto lr = run_length_regulator_full_metal(metal, bundle, s_infer, static_cast<uint32_t>(codes.size()), generated_tokens);

    const uint32_t tokens = prompt_tokens + generated_tokens;
    auto voice_prompt_all = tensor_as_f32(voice, "s2mel_prompt");
    auto condition = metal.hot_condition_merge_f32(voice_prompt_all, lr, prompt_tokens, generated_tokens, cond_dim);
    return HotConditionInputs{std::move(condition), generated_tokens, tokens};
}

HotWavMetrics synthesize_hot_condition_vectors_wav_shared(mit2::MetalContext& metal,
                                                          const mit2::Bundle& bundle,
                                                          const mit2::Bundle& voice,
                                                          const std::vector<float>& condition,
                                                          const std::vector<float>& noise,
                                                          uint32_t prompt_tokens,
                                                          uint32_t steps,
                                                          float cfg_rate,
                                                          const std::string& output_wav) {
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t sample_rate = 22050;
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("Hot TTS condition input synth noise must have shape [tokens,80]");
    }
    if (prompt_tokens == 0) {
        throw std::runtime_error("Hot TTS condition input synth prompt_tokens must be positive");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("Hot TTS condition input synth prompt_tokens exceeds total tokens");
    }
    if (steps == 0) {
        throw std::runtime_error("Hot TTS condition input synth steps must be positive");
    }
    const uint32_t generated_tokens = tokens - prompt_tokens;
    if (condition.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("Hot TTS condition input synth condition must have shape [tokens,512]");
    }
    const auto* mel_info = voice.find("mel");
    if (!mel_info || mel_info->shape.size() != 3 || mel_info->shape[0] != 1 || mel_info->shape[1] != mel_dim ||
        static_cast<uint32_t>(mel_info->shape[2]) < prompt_tokens) {
        throw std::runtime_error("voice mel must have shape [1,80,prompt_tokens>=requested]");
    }
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
    auto full_mel = run_cfm_euler_metal(metal, bundle, noise, prompt, condition, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> generated_mel(static_cast<size_t>(generated_tokens) * mel_dim);
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(full_mel.begin() + generated_offset, full_mel.end(), generated_mel.begin());
    auto wave = run_bigvgan_vocoder_metal(metal, bundle, generated_mel, generated_tokens);
    write_wav_pcm16(output_wav, wave, sample_rate);
    float peak_abs = 0.0f;
    for (float sample : wave) {
        peak_abs = std::max(peak_abs, std::abs(sample));
    }
    return HotWavMetrics{generated_tokens, tokens, wave.size(), sample_rate, peak_abs};
}

// ---------------------------------------------------------------------------
// Acoustic pipeline worker: runs the acoustic stage (CFM + BigVGAN) of segment
// N on its own thread + MetalContext while the main thread computes
// frontend/GPT/condition of segment N+1. The worker never prints — the main
// thread joins (flush) and prints the deferred JSON blocks, keeping stdout
// single-threaded. Enabled only inside the multi-segment loop.
// ---------------------------------------------------------------------------
struct AcousticJob {
    // inputs
    std::string bundle_dir;
    std::string voice_bundle_dir;
    std::string output_wav;
    std::vector<float> condition;
    std::vector<float> noise;
    uint32_t prompt_tokens = 0;
    uint32_t steps = 0;
    float cfg_rate = 0.0f;
    uint32_t expected_tokens = 0;
    // outputs
    HotWavMetrics metrics;
    mit2::MetalResourceStats stats;
    double seconds = 0.0;
    decltype(process_memory_info()) mem{};
    bool ok = false;
    std::string error;
    std::function<void(const AcousticJob&)> print_fn;
};

class AcousticWorker {
public:
    static AcousticWorker& instance() {
        static AcousticWorker w;
        return w;
    }
    void submit(std::shared_ptr<AcousticJob> job) {
        ensure_thread();
        {
            std::lock_guard<std::mutex> g(mu_);
            queue_.push_back(std::move(job));
        }
        cv_.notify_one();
    }
    // Join all outstanding jobs and print their deferred JSON (main thread only).
    bool flush_and_print() {
        std::vector<std::shared_ptr<AcousticJob>> done;
        {
            std::unique_lock<std::mutex> g(mu_);
            done_cv_.wait(g, [&] { return queue_.empty() && !running_; });
            done.swap(completed_);
        }
        bool all_ok = true;
        for (const auto& j : done) {
            if (j->ok) {
                j->print_fn(*j);
            } else {
                std::cerr << "acoustic worker failed for " << j->output_wav << ": " << j->error << std::endl;
                all_ok = false;
            }
        }
        return all_ok;
    }
private:
    void ensure_thread() {
        std::call_once(once_, [this] {
            std::thread([this] { loop(); }).detach();
        });
    }
    void loop() {
        mit2::MetalContext metal;
        std::unique_ptr<mit2::Bundle> bundle;
        std::unique_ptr<mit2::Bundle> voice;
        std::string bundle_dir, voice_dir;
        for (;;) {
            std::shared_ptr<AcousticJob> job;
            {
                std::unique_lock<std::mutex> g(mu_);
                cv_.wait(g, [&] { return !queue_.empty(); });
                job = queue_.front();
                queue_.pop_front();
                running_ = true;
            }
            try {
                // Per-job autorelease drain: this worker thread runs forever,
                // so Metal/MPS objects autoreleased during acoustic synthesis
                // would otherwise accumulate for the life of the process.
                mit2::AutoreleasePool pool;
                if (!bundle || bundle_dir != job->bundle_dir) {
                    bundle = std::make_unique<mit2::Bundle>(job->bundle_dir);
                    bundle_dir = job->bundle_dir;
                }
                if (!voice || voice_dir != job->voice_bundle_dir) {
                    voice = std::make_unique<mit2::Bundle>(job->voice_bundle_dir);
                    voice_dir = job->voice_bundle_dir;
                }
                const auto t0 = Clock::now();
                const auto s0 = metal.resource_stats();
                job->metrics = synthesize_hot_condition_vectors_wav_shared(
                    metal, *bundle, *voice, job->condition, job->noise,
                    job->prompt_tokens, job->steps, job->cfg_rate, job->output_wav);
                job->stats = metal_stats_delta(s0, metal.resource_stats());
                job->seconds = seconds_since(t0);
                job->mem = process_memory_info();
                if (job->metrics.tokens != job->expected_tokens) {
                    job->error = "Hot TTS seeded input synth token count mismatch";
                } else {
                    job->ok = true;
                }
            } catch (const std::exception& e) {
                job->error = e.what();
            }
            {
                std::lock_guard<std::mutex> g(mu_);
                completed_.push_back(std::move(job));
                running_ = false;
            }
            done_cv_.notify_all();
        }
    }
    std::once_flag once_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::deque<std::shared_ptr<AcousticJob>> queue_;
    std::vector<std::shared_ptr<AcousticJob>> completed_;
    bool running_ = false;
};

bool g_acoustic_pipeline_async = false;

bool synthesize_hot_inputs_seeded_shared_wav(const std::string& bundle_dir,
                                             const std::string& voice_bundle_dir,
                                             const std::string& conds_path,
                                             const std::string& text_ids_path,
                                             uint32_t max_codes,
                                             uint64_t seed,
                                             float temperature,
                                             uint32_t prompt_tokens,
                                             uint32_t steps,
                                             float cfg_rate,
                                             const std::string& output_wav,
                                             const GptSamplingConfig* gpt_sampling = nullptr) {
    const auto started = Clock::now();
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t gpt_width = 1280;
    const std::string codes_path = output_wav + ".codes.u32";
    const std::string condition_path = output_wav + ".condition.f32";
    const std::string noise_path = output_wav + ".noise.f32";
    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    auto conds = read_raw_f32(conds_path);
    auto text_ids = read_raw_u32(text_ids_path);
    if (conds.empty() || (conds.size() % gpt_width) != 0) {
        throw std::runtime_error("GPT input code export conds must have shape [tokens,1280]");
    }
    const auto hot_scratch_plan = compute_hot_scratch_plan_from_inputs(conds, text_ids, max_codes, prompt_tokens);
    uint32_t raw_codes = 0;
    uint32_t prefix_tokens = 0;
    int first_stop_step = -1;
    std::vector<uint32_t> codes;
    mit2::MetalResourceStats gpt_stats;
    const auto gpt_started = Clock::now();
    {
        mit2::MetalContext& gpt_metal = hot_shared_metal();
        const auto stats0 = gpt_metal.resource_stats();
        // Sampled vs greedy GPT decode, both on the shared resident context.
        if (gpt_sampling) {
            codes = run_gpt_kv_codes_inputs_sampled_shared(gpt_metal, bundle, conds, text_ids, max_codes, *gpt_sampling, &raw_codes, &first_stop_step, &prefix_tokens);
        } else {
            codes = run_gpt_kv_codes_inputs_shared(gpt_metal, bundle, conds, text_ids, max_codes, &raw_codes, &first_stop_step, &prefix_tokens);
        }
        gpt_stats = metal_stats_delta(stats0, gpt_metal.resource_stats());
    }
    const double gpt_seconds = seconds_since(gpt_started);
    g_tts_stage_acc.gpt += gpt_seconds;
    write_raw_u32(codes_path, codes);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_codes_inputs_export\",\n";
    std::cout << "  \"shared_bundle\": true,\n";
    std::cout << "  \"stage_context\": \"gpt\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"output_codes_u32\": \"" << codes_path << "\",\n";
    std::cout << "  \"cond_tokens\": " << (conds.size() / gpt_width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"raw_codes\": " << raw_codes << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"first_stop_step\": " << first_stop_step << ",\n";
    std::cout << "  \"predicted_codes\": [";
    for (size_t i = 0; i < codes.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << codes[i];
    }
    std::cout << "],\n";
    print_gpt_decode_rate_json(raw_codes, codes.size(), gpt_seconds);
    print_metal_resource_stats_json("", gpt_stats, false);
    std::cout << "}\n";

    HotConditionInputs hot_condition;
    mit2::MetalResourceStats condition_stats;
    const auto condition_started = Clock::now();
    {
        mit2::MetalContext& condition_metal = hot_shared_metal();
        const auto stats0 = condition_metal.resource_stats();
        hot_condition = build_hot_codes_condition_inputs_shared(condition_metal, bundle, voice, conds, text_ids, codes, prompt_tokens);
        condition_stats = metal_stats_delta(stats0, condition_metal.resource_stats());
    }
    const double condition_seconds = seconds_since(condition_started);
    g_tts_stage_acc.condition += condition_seconds;
    write_raw_f32(condition_path, hot_condition.condition);
    const auto condition_memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_condition_inputs_export\",\n";
    std::cout << "  \"shared_bundle\": true,\n";
    std::cout << "  \"stage_context\": \"condition\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"codes_u32\": \"" << codes_path << "\",\n";
    std::cout << "  \"output_condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"cond_tokens\": " << (conds.size() / gpt_width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << hot_condition.generated_tokens << ",\n";
    std::cout << "  \"tokens\": " << hot_condition.tokens << ",\n";
    print_metal_resource_stats_json("", condition_stats);
    std::cout << "  \"resident_bytes\": " << condition_memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << condition_memory.resident_peak_bytes << "\n";
    std::cout << "}\n";

    const auto noise_started = Clock::now();
    auto noise = make_deterministic_normal_noise(hot_condition.tokens, mel_dim, seed, temperature);
    write_raw_f32(noise_path, noise);
    const double noise_seconds = seconds_since(noise_started);
    HotWavMetrics wav_metrics;
    mit2::MetalResourceStats acoustic_stats;
    // Deferred-print closure: emits the acoustic + summary JSON blocks from the
    // job outputs. Used by both the inline (sync) and pipelined (async) paths.
    const uint32_t expected_tokens = hot_condition.tokens;
    const size_t codes_count = codes.size();
    auto print_acoustic_blocks = [=](const AcousticJob& job) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"hot_tts_condition_inputs_wav\",\n";
        std::cout << "  \"shared_bundle\": true,\n";
        std::cout << "  \"stage_context\": \"acoustic\",\n";
        std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
        std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
        std::cout << "  \"noise_f32\": \"" << noise_path << "\",\n";
        std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
        std::cout << "  \"tokens\": " << job.metrics.tokens << ",\n";
        std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
        std::cout << "  \"generated_tokens\": " << job.metrics.generated_tokens << ",\n";
        std::cout << "  \"samples\": " << job.metrics.samples << ",\n";
        std::cout << "  \"sample_rate\": " << job.metrics.sample_rate << ",\n";
        std::cout << "  \"steps\": " << steps << ",\n";
        std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
        std::cout << "  \"peak_abs\": " << job.metrics.peak_abs << ",\n";
        print_metal_resource_stats_json("", job.stats);
        std::cout << "  \"resident_bytes\": " << job.mem.resident_bytes << ",\n";
        std::cout << "  \"resident_peak_bytes\": " << job.mem.resident_peak_bytes << "\n";
        std::cout << "}\n";
        const auto memory = process_memory_info();
        std::cout << "{\n";
        std::cout << "  \"stage\": \"hot_tts_inputs_seeded_wav\",\n";
        std::cout << "  \"shared_bundle\": true,\n";
        std::cout << "  \"stage_contexts\": [\"gpt\", \"condition\", \"acoustic\"],\n";
        std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
        std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
        std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
        std::cout << "  \"noise_f32\": \"" << noise_path << "\",\n";
        std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
        std::cout << "  \"codes_u32\": \"" << codes_path << "\",\n";
        std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
        std::cout << "  \"max_codes\": " << max_codes << ",\n";
        std::cout << "  \"seed\": " << seed << ",\n";
        std::cout << "  \"temperature\": " << temperature << ",\n";
        std::cout << "  \"tokens\": " << expected_tokens << ",\n";
        std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
        std::cout << "  \"steps\": " << steps << ",\n";
        std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
        std::cout << "  \"gpt_seconds\": " << gpt_seconds << ",\n";
        std::cout << "  \"condition_seconds\": " << condition_seconds << ",\n";
        std::cout << "  \"noise_seconds\": " << noise_seconds << ",\n";
        std::cout << "  \"acoustic_seconds\": " << job.seconds << ",\n";
        std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
        std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
        print_hot_scratch_plan_fields_json(hot_scratch_plan, "planned_scratch_", true);
        print_hot_scratch_actual_fields_json(hot_scratch_plan,
                                             static_cast<uint32_t>(codes_count),
                                             expected_tokens,
                                             "planned_scratch_",
                                             true);
        std::cout << "  \"elapsed_seconds\": " << seconds_since(started) << "\n";
        std::cout << "}\n";
    };

    if (g_acoustic_pipeline_async) {
        // Join + print the previous segment's acoustic before queueing ours.
        if (!AcousticWorker::instance().flush_and_print()) {
            return false;
        }
        auto job = std::make_shared<AcousticJob>();
        job->bundle_dir = bundle_dir;
        job->voice_bundle_dir = voice_bundle_dir;
        job->output_wav = output_wav;
        job->condition = std::move(hot_condition.condition);
        job->noise = std::move(noise);
        job->prompt_tokens = prompt_tokens;
        job->steps = steps;
        job->cfg_rate = cfg_rate;
        job->expected_tokens = expected_tokens;
        job->print_fn = print_acoustic_blocks;
        AcousticWorker::instance().submit(std::move(job));
        return true;
    }

    const auto acoustic_started = Clock::now();
    AcousticJob inline_job;
    {
        mit2::MetalContext& acoustic_metal = hot_shared_metal();
        const auto stats0 = acoustic_metal.resource_stats();
        inline_job.metrics = synthesize_hot_condition_vectors_wav_shared(acoustic_metal, bundle, voice, hot_condition.condition, noise, prompt_tokens, steps, cfg_rate, output_wav);
        inline_job.stats = metal_stats_delta(stats0, acoustic_metal.resource_stats());
    }
    inline_job.seconds = seconds_since(acoustic_started);
    g_tts_stage_acc.acoustic += inline_job.seconds;
    inline_job.mem = process_memory_info();
    if (inline_job.metrics.tokens != expected_tokens) {
        throw std::runtime_error("Hot TTS seeded input synth token count mismatch");
    }
    print_acoustic_blocks(inline_job);
    return true;

}

bool synthesize_hot_inputs_seeded_wav(const std::string& bundle_dir,
                                      const std::string& voice_bundle_dir,
                                      const std::string& conds_path,
                                      const std::string& text_ids_path,
                                      uint32_t max_codes,
                                      uint64_t seed,
                                      float temperature,
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
    if (!export_gpt_kv_codes_inputs(bundle_dir, conds_path, text_ids_path, max_codes, codes_path)) {
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
        throw std::runtime_error("Hot TTS seeded input synth condition must have shape [tokens,512]");
    }
    const uint32_t total_tokens = static_cast<uint32_t>(condition.size() / cond_dim);
    if (prompt_tokens == 0 || prompt_tokens >= total_tokens) {
        throw std::runtime_error("Hot TTS seeded input synth prompt_tokens must be positive and below total tokens");
    }
    auto noise = make_deterministic_normal_noise(total_tokens, mel_dim, seed, temperature);
    write_raw_f32(noise_path, noise);
    const double noise_seconds = seconds_since(noise_started);
    const auto acoustic_started = Clock::now();
    if (!synthesize_hot_condition_inputs_wav(bundle_dir, voice_bundle_dir, condition_path, noise_path, prompt_tokens, steps, cfg_rate, output_wav)) {
        return false;
    }
    const double acoustic_seconds = seconds_since(acoustic_started);
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_inputs_seeded_wav\",\n";
    std::cout << "  \"stage_contexts\": [\"gpt\", \"condition\", \"acoustic\"],\n";
    std::cout << "  \"voice_bundle_dir\": \"" << voice_bundle_dir << "\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"noise_f32\": \"" << noise_path << "\",\n";
    std::cout << "  \"output_wav\": \"" << output_wav << "\",\n";
    std::cout << "  \"codes_u32\": \"" << codes_path << "\",\n";
    std::cout << "  \"condition_f32\": \"" << condition_path << "\",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"seed\": " << seed << ",\n";
    std::cout << "  \"temperature\": " << temperature << ",\n";
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
                                              const std::string& output_wav);

bool synthesize_hot_text_cjk_tokenized_seeded_wav(const std::string& bundle_dir,
                                                  const std::string& voice_bundle_dir,
                                                  const std::string& text,
                                                  const CjkTokenizedText& tokenized,
                                                  uint64_t seed,
                                                  float temperature,
                                                  uint32_t prompt_tokens,
                                                  uint32_t steps,
                                                  float cfg_rate,
                                                  const std::string& output_wav,
                                                  const GptSamplingConfig* gpt_sampling = nullptr,
                                                  uint64_t sampled_noise_seed = 0,
                                                  float sampled_noise_temperature = 0.0f) {
    const auto started = Clock::now();
    const uint32_t max_codes = static_cast<uint32_t>(tokenized.ids.size()) * kMaxCodesPerTextToken;
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t gpt_width = 1280;
    constexpr uint32_t cond_num = 32;
    const std::string tokenizer_dir = bundle_dir + "/tokenizer";
    const std::string text_ids_path = output_wav + ".text_ids.u32";
    const std::string speech_path = output_wav + ".speech_conditioning.f32";
    const std::string emovec_path = output_wav + ".emovec.f32";
    const std::string conds_path = output_wav + ".conds.f32";
    const std::string fake_path = output_wav + ".fake_inputs.u32";
    const std::string inputs_path = output_wav + ".inputs_embeds.f32";
    const std::string mask_path = output_wav + ".attention_mask.u32";

    mit2::Bundle bundle(bundle_dir);
    mit2::Bundle voice(voice_bundle_dir);
    const uint32_t requested_prompt_tokens = prompt_tokens;
    const uint32_t resolved_prompt_tokens = resolve_voice_prompt_tokens(voice, requested_prompt_tokens);
    write_raw_u32(text_ids_path, tokenized.ids);

    const auto* spk_info = voice.find("spk_cond_emb");
    if (!spk_info || spk_info->shape.size() != 3 || spk_info->shape[0] != 1 || spk_info->shape[2] != input_dim ||
        spk_info->shape[1] <= 0) {
        throw std::runtime_error("voice spk_cond_emb must have shape [1,tokens>0,1024]");
    }
    const uint32_t input_tokens = static_cast<uint32_t>(spk_info->shape[1]);
    auto spk_cond_emb = tensor_as_f32(voice, "spk_cond_emb");
    if (spk_cond_emb.size() != static_cast<size_t>(input_tokens) * input_dim) {
        throw std::runtime_error("voice spk_cond_emb tensor size mismatch");
    }

    const auto frontend_started = Clock::now();
    mit2::MetalResourceStats frontend_stats;
    std::vector<float> speech;
    std::vector<float> emovec;
    std::vector<float> conds;
    GptPrepareInputsOutputs prefix;
    uint32_t conditioning_tokens = 0;
    {
        mit2::MetalContext& metal = hot_shared_metal();
        const auto frontend_stats0 = metal.resource_stats();
        // The conformer/perceiver voice conditioning depends only on the voice
        // bundle — cache it (LRU, capacity g_voice_cond_cache_size, --lrucache)
        // so repeated requests with the same voice (the common case: dialog
        // turns alternate between a few voices) skip ~0.25s of frontend per
        // segment. Capacity 0 disables the cache.
        struct VoiceConditioning {
            std::string voice_dir;
            std::vector<float> speech;
            std::vector<float> emovec;
            uint32_t conditioning_tokens = 0;
        };
        static std::deque<VoiceConditioning> voice_cond_cache;
        const size_t cache_capacity = g_voice_cond_cache_size;
        bool cache_hit = false;
        if (cache_capacity > 0) {
            for (auto it = voice_cond_cache.begin(); it != voice_cond_cache.end(); ++it) {
                if (it->voice_dir == voice_bundle_dir) {
                    speech = it->speech;
                    emovec = it->emovec;
                    conditioning_tokens = it->conditioning_tokens;
                    // Move to front (most recently used).
                    VoiceConditioning hit = std::move(*it);
                    voice_cond_cache.erase(it);
                    voice_cond_cache.push_front(std::move(hit));
                    cache_hit = true;
                    break;
                }
            }
        }
        if (!cache_hit) {
            emovec = run_gpt_emovec_metal_linear(metal, bundle, spk_cond_emb, input_tokens);
            auto sub = run_gpt_conditioning_subsampling_metal_linear(metal, bundle, spk_cond_emb, input_tokens);
            conditioning_tokens = sub.output_tokens;
            auto context = run_gpt_conformer_stack_metal_attn_conv_ff(metal,
                                                                      bundle,
                                                                      sub.subsampling,
                                                                      sub.pos_emb,
                                                                      sub.mask,
                                                                      sub.output_tokens,
                                                                      6);
            std::vector<uint32_t> perceiver_mask(cond_num, 1);
            perceiver_mask.insert(perceiver_mask.end(), sub.mask.begin(), sub.mask.end());
            speech = run_gpt_perceiver_metal(metal, bundle, context, perceiver_mask, sub.output_tokens);
            if (cache_capacity > 0) {
                voice_cond_cache.push_front(VoiceConditioning{voice_bundle_dir, speech, emovec, conditioning_tokens});
                while (voice_cond_cache.size() > cache_capacity) {
                    voice_cond_cache.pop_back();
                }
            }
        }
        auto speed = tensor_as_f32(bundle, "gpt.speed_emb.weight");
        if (speech.size() != static_cast<size_t>(cond_num) * gpt_width || emovec.size() != gpt_width ||
            speed.size() != static_cast<size_t>(2) * gpt_width) {
            throw std::runtime_error("native text CJK frontend tensor shape mismatch");
        }
        conds.assign(static_cast<size_t>(cond_num + 2) * gpt_width, 0.0f);
        for (uint32_t t = 0; t < cond_num; ++t) {
            for (uint32_t c = 0; c < gpt_width; ++c) {
                conds[static_cast<size_t>(t) * gpt_width + c] = speech[static_cast<size_t>(t) * gpt_width + c] + emovec[c];
            }
        }
        std::copy(speed.begin() + gpt_width,
                  speed.begin() + static_cast<size_t>(2) * gpt_width,
                  conds.begin() + static_cast<size_t>(cond_num) * gpt_width);
        std::copy(speed.begin(), speed.begin() + gpt_width, conds.begin() + static_cast<size_t>(cond_num + 1) * gpt_width);
        prefix = run_gpt_prepare_inputs_metal(metal, bundle, conds, tokenized.ids, cond_num + 2, static_cast<uint32_t>(tokenized.ids.size()));
        frontend_stats = metal_stats_delta(frontend_stats0, metal.resource_stats());
    }
    const double frontend_seconds = seconds_since(frontend_started);
    g_tts_stage_acc.frontend += frontend_seconds;
    write_raw_f32(speech_path, speech);
    write_raw_f32(emovec_path, emovec);
    write_raw_f32(conds_path, conds);
    write_raw_u32(fake_path, prefix.fake_inputs);
    write_raw_f32(inputs_path, prefix.inputs_embeds);
    write_raw_u32(mask_path, prefix.attention_mask);

    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_frontend_text_cjk_export\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << json_escape(text_ids_path) << "\",\n";
    std::cout << "  \"speech_conditioning_f32\": \"" << json_escape(speech_path) << "\",\n";
    std::cout << "  \"emovec_f32\": \"" << json_escape(emovec_path) << "\",\n";
    std::cout << "  \"conds_f32\": \"" << json_escape(conds_path) << "\",\n";
    std::cout << "  \"fake_inputs_u32\": \"" << json_escape(fake_path) << "\",\n";
    std::cout << "  \"inputs_embeds_f32\": \"" << json_escape(inputs_path) << "\",\n";
    std::cout << "  \"attention_mask_u32\": \"" << json_escape(mask_path) << "\",\n";
    std::cout << "  \"pieces\": ";
    print_json_string_array(tokenized.pieces);
    std::cout << ",\n";
    std::cout << "  \"ids\": ";
    print_json_u32_array(tokenized.ids);
    std::cout << ",\n";
    std::cout << "  \"voice_conditioning_tokens\": " << input_tokens << ",\n";
    std::cout << "  \"subsampled_conditioning_tokens\": " << conditioning_tokens << ",\n";
    std::cout << "  \"cond_tokens\": " << (cond_num + 2) << ",\n";
    std::cout << "  \"text_tokens\": " << tokenized.ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << (prefix.inputs_embeds.size() / gpt_width) << ",\n";
    std::cout << "  \"frontend_tail_source\": \"native\",\n";
    std::cout << "  \"text_ids_source\": \"native_cjk_inline\",\n";
    std::cout << "  \"emovec_source\": \"native_full_metal_subsampling_conformer_perceiver_linear\",\n";
    std::cout << "  \"subsampling_source\": \"native_metal_resident_conv_linear\",\n";
    std::cout << "  \"conformer_source\": \"native_stack_metal_resident_attn_core_conv_ff\",\n";
    std::cout << "  \"perceiver_source\": \"native_metal_resident_linear_cross_attn_geglu_rmsnorm\",\n";
    std::cout << "  \"nonfinite_speech_conditioning\": " << count_nonfinite(speech) << ",\n";
    std::cout << "  \"nonfinite_emovec\": " << count_nonfinite(emovec) << ",\n";
    std::cout << "  \"nonfinite_conds\": " << count_nonfinite(conds) << ",\n";
    std::cout << "  \"nonfinite_inputs\": " << count_nonfinite(prefix.inputs_embeds) << ",\n";
    std::cout << "  \"frontend_seconds\": " << frontend_seconds << ",\n";
    print_metal_resource_stats_json("", frontend_stats, false);
    std::cout << "}\n";
    if (count_nonfinite(speech) != 0 || count_nonfinite(emovec) != 0 || count_nonfinite(conds) != 0 ||
        count_nonfinite(prefix.inputs_embeds) != 0) {
        return false;
    }
    // Both branches use the resident shared MetalContext (hot_shared_metal) and
    // keep tensors in memory. The sampled branch passes the GPT sampling config;
    // noise uses the sampled noise seed/temperature. (Previously the sampled
    // branch went through synthesize_hot_inputs_sampled_seeded_wav, which spun up
    // a fresh MetalContext per stage and round-tripped tensors through disk every
    // request — re-uploading weights and recompiling pipelines each call.)
    if (gpt_sampling) {
        if (!synthesize_hot_inputs_seeded_shared_wav(bundle_dir,
                                                     voice_bundle_dir,
                                                     conds_path,
                                                     text_ids_path,
                                                     max_codes,
                                                     sampled_noise_seed,
                                                     sampled_noise_temperature,
                                                     resolved_prompt_tokens,
                                                     steps,
                                                     cfg_rate,
                                                     output_wav,
                                                     gpt_sampling)) {
            return false;
        }
    } else {
        if (!synthesize_hot_inputs_seeded_shared_wav(bundle_dir,
                                                     voice_bundle_dir,
                                                     conds_path,
                                                     text_ids_path,
                                                     max_codes,
                                                     seed,
                                                     temperature,
                                                     resolved_prompt_tokens,
                                                     steps,
                                                     cfg_rate,
                                                     output_wav)) {
            return false;
        }
    }
    const auto memory = process_memory_info();
    std::cout << "{\n";
    std::cout << "  \"stage\": \"" << (gpt_sampling ? "hot_tts_text_cjk_sampled_seeded_wav" : "hot_tts_text_cjk_seeded_wav") << "\",\n";
    std::cout << "  \"stage_contexts\": [\"frontend\", \"gpt\", \"condition\", \"acoustic\"],\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"output_wav\": \"" << json_escape(output_wav) << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << json_escape(text_ids_path) << "\",\n";
    std::cout << "  \"conds_f32\": \"" << json_escape(conds_path) << "\",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"seed\": " << seed << ",\n";
    std::cout << "  \"temperature\": " << temperature << ",\n";
    if (gpt_sampling) {
        std::cout << "  \"gpt_seed\": " << gpt_sampling->seed << ",\n";
        std::cout << "  \"gpt_temperature\": " << gpt_sampling->temperature << ",\n";
        std::cout << "  \"gpt_top_k\": " << gpt_sampling->top_k << ",\n";
        std::cout << "  \"gpt_top_p\": " << gpt_sampling->top_p << ",\n";
        std::cout << "  \"gpt_repetition_penalty\": " << gpt_sampling->repetition_penalty << ",\n";
        std::cout << "  \"noise_seed\": " << sampled_noise_seed << ",\n";
        std::cout << "  \"noise_temperature\": " << sampled_noise_temperature << ",\n";
    }
    std::cout << "  \"requested_prompt_tokens\": " << requested_prompt_tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << resolved_prompt_tokens << ",\n";
    std::cout << "  \"prompt_tokens_source\": \"" << (requested_prompt_tokens == 0 ? "voice_bundle" : "argument") << "\",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"frontend_seconds\": " << frontend_seconds << ",\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
    std::cout << "  \"elapsed_seconds\": " << seconds_since(started) << "\n";
    std::cout << "}\n";
    return true;
}

bool synthesize_hot_text_cjk_seeded_wav(const std::string& bundle_dir,
                                        const std::string& voice_bundle_dir,
                                        const std::string& text,
                                        uint64_t seed,
                                        float temperature,
                                        uint32_t prompt_tokens,
                                        uint32_t steps,
                                        float cfg_rate,
                                        const std::string& output_wav) {
    const std::string tokenizer_dir = bundle_dir + "/tokenizer";
    const auto tokenized = tokenize_text_full(tokenizer_dir, text);
    return synthesize_hot_text_cjk_tokenized_seeded_wav(bundle_dir,
                                                        voice_bundle_dir,
                                                        text,
                                                        tokenized,
                                                        seed,
                                                        temperature,
                                                        prompt_tokens,
                                                        steps,
                                                        cfg_rate,
                                                        output_wav);
}

bool synthesize_hot_text_cjk_sampled_seeded_wav(const std::string& bundle_dir,
                                                const std::string& voice_bundle_dir,
                                                const std::string& text,
                                                const GptSamplingConfig& gpt_sampling,
                                                uint64_t noise_seed,
                                                float noise_temperature,
                                                uint32_t prompt_tokens,
                                                uint32_t steps,
                                                float cfg_rate,
                                                const std::string& output_wav) {
    const std::string tokenizer_dir = bundle_dir + "/tokenizer";
    const auto tokenized = tokenize_text_full(tokenizer_dir, text);
    return synthesize_hot_text_cjk_tokenized_seeded_wav(bundle_dir,
                                                        voice_bundle_dir,
                                                        text,
                                                        tokenized,
                                                        noise_seed,
                                                        noise_temperature,
                                                        prompt_tokens,
                                                        steps,
                                                        cfg_rate,
                                                        output_wav,
                                                        &gpt_sampling,
                                                        noise_seed,
                                                        noise_temperature);
}

struct SegmentConcatResult {
    uint32_t sample_rate = 0;
    uint32_t samples = 0;
    double concat_seconds = 0.0;
    std::vector<uint32_t> segment_samples;
};

SegmentConcatResult concat_segment_wavs_pcm16(const std::vector<std::string>& segment_wavs,
                                              uint32_t interval_silence_ms,
                                              const std::string& output_wav) {
    if (segment_wavs.empty()) {
        throw std::runtime_error("no segment wavs to concatenate");
    }
    const auto started = Clock::now();
    uint32_t sample_rate = 0;
    std::vector<char> joined;
    std::vector<uint32_t> segment_samples;
    segment_samples.reserve(segment_wavs.size());
    for (size_t i = 0; i < segment_wavs.size(); ++i) {
        auto wav = read_wav_pcm16_mono_bytes(segment_wavs[i]);
        if (sample_rate == 0) {
            sample_rate = wav.sample_rate;
        } else if (wav.sample_rate != sample_rate) {
            throw std::runtime_error("segment wav sample rates do not match");
        }
        segment_samples.push_back(static_cast<uint32_t>(wav.frames.size() / 2));
        joined.insert(joined.end(), wav.frames.begin(), wav.frames.end());
        if (i + 1 < segment_wavs.size() && interval_silence_ms > 0) {
            const uint32_t silence_frames = static_cast<uint32_t>(
                (static_cast<uint64_t>(sample_rate) * interval_silence_ms) / 1000u);
            joined.insert(joined.end(), static_cast<size_t>(silence_frames) * 2u, 0);
        }
    }
    write_wav_pcm16_mono_bytes(output_wav, sample_rate, joined);
    return SegmentConcatResult{
        sample_rate,
        static_cast<uint32_t>(joined.size() / 2),
        seconds_since(started),
        std::move(segment_samples),
    };
}

bool synthesize_hot_text_cjk_segments_seeded_wav(const std::string& bundle_dir,
                                                 const std::string& voice_bundle_dir,
                                                 const std::string& text,
                                                 uint32_t max_text_tokens_per_segment,
                                                 uint64_t seed,
                                                 float temperature,
                                                 uint32_t prompt_tokens,
                                                 uint32_t steps,
                                                 float cfg_rate,
                                                 uint32_t interval_silence_ms,
                                                 const std::string& output_wav) {
    const auto started = Clock::now();
    const std::string tokenizer_dir = bundle_dir + "/tokenizer";
    const auto tokenized = tokenize_text_full(tokenizer_dir, text);
    const auto segments = split_cjk_tokenized_text(tokenized, max_text_tokens_per_segment);
    mit2::Bundle voice(voice_bundle_dir);
    const uint32_t requested_prompt_tokens = prompt_tokens;
    const uint32_t resolved_prompt_tokens = resolve_voice_prompt_tokens(voice, requested_prompt_tokens);
    const auto segment_dir = std::filesystem::path(output_wav + ".segments");
    std::filesystem::create_directories(segment_dir);

    std::vector<std::string> segment_wavs;
    segment_wavs.reserve(segments.size());
    // Pipeline: segment N's acoustic runs on the worker thread while segment
    // N+1's frontend/GPT/condition run here. Flushed before concat.
    // Measured net-negative on a single GPU once stages became GPU-bound (the
    // second MetalContext costs library compile + weight re-upload, and the GPU
    // just time-slices) — so opt-in via MIT2_SEGMENT_PIPELINE=1.
    static const bool pipeline_enabled = []() {
        const char* v = std::getenv("MIT2_SEGMENT_PIPELINE");
        return v && v[0] == '1';
    }();
    g_acoustic_pipeline_async = pipeline_enabled && segments.size() > 1;
    bool segments_ok = true;
    for (size_t i = 0; i < segments.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "segment_%03zu.wav", i);
        const std::string segment_wav = (segment_dir / name).string();
        CjkTokenizedText segment_text{segments[i].pieces, segments[i].ids};
        const std::string segment_label = text + "#segment_" + std::to_string(i);
        if (!synthesize_hot_text_cjk_tokenized_seeded_wav(bundle_dir,
                                                          voice_bundle_dir,
                                                          segment_label,
                                                          segment_text,
                                                          seed + static_cast<uint64_t>(i),
                                                          temperature,
                                                          resolved_prompt_tokens,
                                                          steps,
                                                          cfg_rate,
                                                          segment_wav)) {
            segments_ok = false;
            break;
        }
        segment_wavs.push_back(segment_wav);
    }
    const bool flush_ok = AcousticWorker::instance().flush_and_print();
    g_acoustic_pipeline_async = false;
    if (!segments_ok || !flush_ok) {
        return false;
    }

    const auto concat = concat_segment_wavs_pcm16(segment_wavs, interval_silence_ms, output_wav);
    const auto memory = process_memory_info();

    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_text_cjk_segments_seeded_wav\",\n";
    std::cout << "  \"stage_contexts\": [\"split\", \"segments\", \"concat\"],\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"output_wav\": \"" << json_escape(output_wav) << "\",\n";
    std::cout << "  \"segment_dir\": \"" << json_escape(segment_dir.string()) << "\",\n";
    std::cout << "  \"max_text_tokens_per_segment\": " << max_text_tokens_per_segment << ",\n";
    std::cout << "  \"max_codes_per_text_token\": " << kMaxCodesPerTextToken << ",\n";
    std::cout << "  \"seed\": " << seed << ",\n";
    std::cout << "  \"temperature\": " << temperature << ",\n";
    std::cout << "  \"requested_prompt_tokens\": " << requested_prompt_tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << resolved_prompt_tokens << ",\n";
    std::cout << "  \"prompt_tokens_source\": \"" << (requested_prompt_tokens == 0 ? "voice_bundle" : "argument") << "\",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"interval_silence_ms\": " << interval_silence_ms << ",\n";
    std::cout << "  \"sample_rate\": " << concat.sample_rate << ",\n";
    std::cout << "  \"samples\": " << concat.samples << ",\n";
    std::cout << "  \"tokens\": ";
    print_json_string_array(tokenized.pieces);
    std::cout << ",\n";
    std::cout << "  \"token_ids\": ";
    print_json_u32_array(tokenized.ids);
    std::cout << ",\n";
    std::cout << "  \"segments\": [\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        std::cout << "    {\n";
        std::cout << "      \"index\": " << i << ",\n";
        std::cout << "      \"seed\": " << (seed + static_cast<uint64_t>(i)) << ",\n";
        std::cout << "      \"output_wav\": \"" << json_escape(segment_wavs[i]) << "\",\n";
        std::cout << "      \"samples\": " << concat.segment_samples[i] << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(segments[i].pieces);
        std::cout << ",\n";
        std::cout << "      \"ids\": ";
        print_json_u32_array(segments[i].ids);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == segments.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"concat_seconds\": " << concat.concat_seconds << ",\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
    std::cout << "  \"elapsed_seconds\": " << seconds_since(started) << "\n";
    std::cout << "}\n";
    return true;
}

bool synthesize_hot_text_cjk_segments_sampled_seeded_wav(const std::string& bundle_dir,
                                                         const std::string& voice_bundle_dir,
                                                         const std::string& text,
                                                         uint32_t max_text_tokens_per_segment,
                                                         const GptSamplingConfig& gpt_sampling,
                                                         uint64_t noise_seed,
                                                         float noise_temperature,
                                                         uint32_t prompt_tokens,
                                                         uint32_t steps,
                                                         float cfg_rate,
                                                         uint32_t interval_silence_ms,
                                                         const std::string& output_wav) {
    const auto started = Clock::now();
    const std::string tokenizer_dir = bundle_dir + "/tokenizer";
    const auto tokenized = tokenize_text_full(tokenizer_dir, text);
    const auto segments = split_cjk_tokenized_text(tokenized, max_text_tokens_per_segment);
    mit2::Bundle voice(voice_bundle_dir);
    const uint32_t requested_prompt_tokens = prompt_tokens;
    const uint32_t resolved_prompt_tokens = resolve_voice_prompt_tokens(voice, requested_prompt_tokens);
    const auto segment_dir = std::filesystem::path(output_wav + ".segments");
    std::filesystem::create_directories(segment_dir);

    std::vector<std::string> segment_wavs;
    std::vector<uint64_t> segment_gpt_seeds;
    std::vector<uint64_t> segment_noise_seeds;
    segment_wavs.reserve(segments.size());
    segment_gpt_seeds.reserve(segments.size());
    segment_noise_seeds.reserve(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "segment_%03zu.wav", i);
        const std::string segment_wav = (segment_dir / name).string();
        CjkTokenizedText segment_text{segments[i].pieces, segments[i].ids};
        const std::string segment_label = text + "#segment_" + std::to_string(i);
        GptSamplingConfig segment_sampling = gpt_sampling;
        segment_sampling.seed = gpt_sampling.seed + static_cast<uint64_t>(i);
        const uint64_t segment_noise_seed = noise_seed + static_cast<uint64_t>(i);
        if (!synthesize_hot_text_cjk_tokenized_seeded_wav(bundle_dir,
                                                          voice_bundle_dir,
                                                          segment_label,
                                                          segment_text,
                                                          segment_noise_seed,
                                                          noise_temperature,
                                                          resolved_prompt_tokens,
                                                          steps,
                                                          cfg_rate,
                                                          segment_wav,
                                                          &segment_sampling,
                                                          segment_noise_seed,
                                                          noise_temperature)) {
            return false;
        }
        segment_wavs.push_back(segment_wav);
        segment_gpt_seeds.push_back(segment_sampling.seed);
        segment_noise_seeds.push_back(segment_noise_seed);
    }

    const auto concat = concat_segment_wavs_pcm16(segment_wavs, interval_silence_ms, output_wav);
    const auto memory = process_memory_info();

    std::cout << "{\n";
    std::cout << "  \"stage\": \"hot_tts_text_cjk_segments_sampled_seeded_wav\",\n";
    std::cout << "  \"stage_contexts\": [\"split\", \"sampled_segments\", \"concat\"],\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"output_wav\": \"" << json_escape(output_wav) << "\",\n";
    std::cout << "  \"segment_dir\": \"" << json_escape(segment_dir.string()) << "\",\n";
    std::cout << "  \"max_text_tokens_per_segment\": " << max_text_tokens_per_segment << ",\n";
    std::cout << "  \"max_codes_per_text_token\": " << kMaxCodesPerTextToken << ",\n";
    std::cout << "  \"gpt_seed\": " << gpt_sampling.seed << ",\n";
    std::cout << "  \"gpt_temperature\": " << gpt_sampling.temperature << ",\n";
    std::cout << "  \"gpt_top_k\": " << gpt_sampling.top_k << ",\n";
    std::cout << "  \"gpt_top_p\": " << gpt_sampling.top_p << ",\n";
    std::cout << "  \"gpt_repetition_penalty\": " << gpt_sampling.repetition_penalty << ",\n";
    std::cout << "  \"noise_seed\": " << noise_seed << ",\n";
    std::cout << "  \"noise_temperature\": " << noise_temperature << ",\n";
    std::cout << "  \"requested_prompt_tokens\": " << requested_prompt_tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << resolved_prompt_tokens << ",\n";
    std::cout << "  \"prompt_tokens_source\": \"" << (requested_prompt_tokens == 0 ? "voice_bundle" : "argument") << "\",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"interval_silence_ms\": " << interval_silence_ms << ",\n";
    std::cout << "  \"sample_rate\": " << concat.sample_rate << ",\n";
    std::cout << "  \"samples\": " << concat.samples << ",\n";
    std::cout << "  \"tokens\": ";
    print_json_string_array(tokenized.pieces);
    std::cout << ",\n";
    std::cout << "  \"token_ids\": ";
    print_json_u32_array(tokenized.ids);
    std::cout << ",\n";
    std::cout << "  \"segments\": [\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        std::cout << "    {\n";
        std::cout << "      \"index\": " << i << ",\n";
        std::cout << "      \"gpt_seed\": " << segment_gpt_seeds[i] << ",\n";
        std::cout << "      \"noise_seed\": " << segment_noise_seeds[i] << ",\n";
        std::cout << "      \"output_wav\": \"" << json_escape(segment_wavs[i]) << "\",\n";
        std::cout << "      \"samples\": " << concat.segment_samples[i] << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(segments[i].pieces);
        std::cout << ",\n";
        std::cout << "      \"ids\": ";
        print_json_u32_array(segments[i].ids);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == segments.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"concat_seconds\": " << concat.concat_seconds << ",\n";
    std::cout << "  \"resident_bytes\": " << memory.resident_bytes << ",\n";
    std::cout << "  \"resident_peak_bytes\": " << memory.resident_peak_bytes << ",\n";
    std::cout << "  \"elapsed_seconds\": " << seconds_since(started) << "\n";
    std::cout << "}\n";
    return true;
}

struct TtsCjkPreset {
    std::string name;
    uint32_t steps = 0;
    uint32_t max_text_tokens_per_segment = 0;
    uint32_t interval_silence_ms = 0;
};

TtsCjkPreset resolve_tts_cjk_preset(const std::string& preset) {
    if (preset == "smoke") {
        return {preset, 1, 4, 200};
    }
    if (preset == "short") {
        return {preset, cfm_synthesis_steps(), 80, 200};
    }
    if (preset == "standard") {
        // 64 text tokens ≈ 500-700 codes ≈ 10-12s audio per segment: the sweet
        // spot between voice-prompt amortization (favors long) and GPT KV /
        // CFM O(T^2) growth (favors short). Measured seg-RTF 0.87-0.89 here vs
        // 1.28+ for 120-token mega-segments.
        return {preset, cfm_synthesis_steps(), 64, 200};
    }
    throw std::runtime_error("unknown --tts-cjk-preset value: " + preset + " (expected smoke, short, or standard)");
}

GptSamplingConfig reference_default_gpt_sampling(uint64_t seed) {
    GptSamplingConfig sampling;
    sampling.do_sample = true;
    sampling.seed = seed;
    sampling.temperature = 1.0f;
    sampling.top_k = 30;
    sampling.top_p = 0.8f;
    sampling.repetition_penalty = 10.0f;
    return sampling;
}

bool tts_cjk_preset_needs_segments(const std::string& bundle_dir,
                                   const std::string& text,
                                   uint32_t max_text_tokens_per_segment) {
    const auto tokenized = tokenize_text_full(bundle_dir + "/tokenizer", text);
    const auto segments = split_cjk_tokenized_text(tokenized, max_text_tokens_per_segment);
    return segments.size() > 1;
}

[[maybe_unused]] void print_tts_launcher_capabilities_json() {
    const std::vector<TtsCjkPreset> presets{
        resolve_tts_cjk_preset("smoke"),
        resolve_tts_cjk_preset("short"),
        resolve_tts_cjk_preset("standard"),
    };
    std::cout << "{\n";
    std::cout << "  \"stage\": \"tts_launcher_capabilities\",\n";
    std::cout << "  \"ok\": true,\n";
    std::cout << "  \"binary\": \"mit2_tts\",\n";
    std::cout << "  \"product_surface_version\": 1,\n";
    std::cout << "  \"default_preset\": \"standard\",\n";
    std::cout << "  \"supports_cached_voice_tts_cjk\": true,\n";
    std::cout << "  \"supports_cached_voice_tts_general_text\": false,\n";
    std::cout << "  \"supports_native_voice_clone\": false,\n";
    std::cout << "  \"requires_model_bundle\": true,\n";
    std::cout << "  \"requires_voice_bundle\": true,\n";
    std::cout << "  \"native_text_surface\": \"focused_cjk_limited_ascii\",\n";
    std::cout << "  \"supported_product_surfaces\": [\n";
    std::cout << "    \"cached_voice_cjk_text_to_wav\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"unsupported_product_surfaces\": [\n";
    std::cout << "    \"cached_voice_general_text_to_wav\",\n";
    std::cout << "    \"native_clone_audio_to_voice_bundle\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"native_product_commands\": [\n";
    std::cout << "    \"mit2_tts MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT OUTPUT_WAV [PRESET]\",\n";
    std::cout << "    \"--readiness MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR\",\n";
    std::cout << "    \"--preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT\",\n";
    std::cout << "    \"--clone-preflight AUDIO_WAV\",\n";
    std::cout << "    \"--clone-preprocess AUDIO_WAV OUTPUT_F32\",\n";
    std::cout << "    \"--clone-readiness PREPROCESS_MANIFEST\",\n";
    std::cout << "    \"--clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32\",\n";
    std::cout << "    \"--clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32\",\n";
    std::cout << "    \"--clone-prepare-features AUDIO_WAV OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-feature-readiness FEATURE_MANIFEST\",\n";
    std::cout << "    \"--clone-encoder-model-readiness MODEL_BUNDLE_DIR\",\n";
    std::cout << "    \"--clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST S2MEL_STYLE_F32\",\n";
    std::cout << "    \"--clone-campplus-style-from-features MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32\",\n";
    std::cout << "    \"--clone-campplus-head-golden MODEL_BUNDLE_DIR FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR\",\n";
    std::cout << "    \"--clone-w2v-feature-project MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32\",\n";
    std::cout << "    \"--clone-w2v-encoder MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_SPK_COND_F32\",\n";
    std::cout << "    \"--clone-w2v-extract-features PREPROCESS_MANIFEST OUTPUT_FEATURES_F32 OUTPUT_MASK_U32\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32 W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-qkv MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32 W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32 W2V_TOKENS OUTPUT_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-activate W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_ACTIVATED_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32 W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-qkv MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention W2V_LAYER1_Q_F32 W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER1_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention-residual W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER1_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-activate W2V_LAYER2_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_ACTIVATED_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_OUTPUT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn1-residual W2V_LAYER1_F32 W2V_LAYER2_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-qkv MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention W2V_LAYER2_Q_F32 W2V_LAYER2_K_F32 W2V_LAYER2_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER2_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention-residual W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER2_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-conv-residual MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_LAYER2_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer2-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER2_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-activate W2V_LAYER3_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_ACTIVATED_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_OUTPUT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn1-residual W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_LAYER3_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-qkv MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention W2V_LAYER3_Q_F32 W2V_LAYER3_K_F32 W2V_LAYER3_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER3_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention-residual W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-activate W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_ACTIVATED_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32 W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-qkv MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32 W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER4_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention-residual W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER4_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-conv-residual MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_LAYER4_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer4-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER4_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER4_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-qkv MODEL_BUNDLE_DIR W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention W2V_LAYER5_Q_F32 W2V_LAYER5_K_F32 W2V_LAYER5_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER5_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention-project MODEL_BUNDLE_DIR W2V_LAYER5_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention-residual W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_LAYER5_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-attention-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-glu MODEL_BUNDLE_DIR W2V_LAYER5_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER5_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-conv-residual MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_LAYER5_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer5-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER5_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER5_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-qkv MODEL_BUNDLE_DIR W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention W2V_LAYER6_Q_F32 W2V_LAYER6_K_F32 W2V_LAYER6_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER6_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention-project MODEL_BUNDLE_DIR W2V_LAYER6_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention-residual W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_LAYER6_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-attention-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-glu MODEL_BUNDLE_DIR W2V_LAYER6_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER6_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-conv-residual MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_LAYER6_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer6-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER6_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER6_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-qkv MODEL_BUNDLE_DIR W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention W2V_LAYER7_Q_F32 W2V_LAYER7_K_F32 W2V_LAYER7_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER7_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention-project MODEL_BUNDLE_DIR W2V_LAYER7_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention-residual W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_LAYER7_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-attention-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-glu MODEL_BUNDLE_DIR W2V_LAYER7_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER7_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-conv-residual MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_LAYER7_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer7-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER7_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER7_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-qkv MODEL_BUNDLE_DIR W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention W2V_LAYER8_Q_F32 W2V_LAYER8_K_F32 W2V_LAYER8_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER8_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention-project MODEL_BUNDLE_DIR W2V_LAYER8_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention-residual W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_LAYER8_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-attention-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-glu MODEL_BUNDLE_DIR W2V_LAYER8_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER8_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-conv-residual MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_LAYER8_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer8-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER8_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER8_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-qkv MODEL_BUNDLE_DIR W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention W2V_LAYER9_Q_F32 W2V_LAYER9_K_F32 W2V_LAYER9_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER9_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention-project MODEL_BUNDLE_DIR W2V_LAYER9_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention-residual W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_LAYER9_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-attention-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-glu MODEL_BUNDLE_DIR W2V_LAYER9_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER9_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-conv-residual MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_LAYER9_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer9-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER9_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER9_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-qkv MODEL_BUNDLE_DIR W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention W2V_LAYER10_Q_F32 W2V_LAYER10_K_F32 W2V_LAYER10_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER10_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention-project MODEL_BUNDLE_DIR W2V_LAYER10_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention-residual W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_LAYER10_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-attention-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-glu MODEL_BUNDLE_DIR W2V_LAYER10_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER10_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-conv-residual MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_LAYER10_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer10-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER10_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER10_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-qkv MODEL_BUNDLE_DIR W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention W2V_LAYER11_Q_F32 W2V_LAYER11_K_F32 W2V_LAYER11_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER11_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention-project MODEL_BUNDLE_DIR W2V_LAYER11_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention-residual W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_LAYER11_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-attention-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-glu MODEL_BUNDLE_DIR W2V_LAYER11_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER11_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-conv-residual MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_LAYER11_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer11-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER11_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER11_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-qkv MODEL_BUNDLE_DIR W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention W2V_LAYER12_Q_F32 W2V_LAYER12_K_F32 W2V_LAYER12_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER12_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention-project MODEL_BUNDLE_DIR W2V_LAYER12_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention-residual W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_LAYER12_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-attention-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-glu MODEL_BUNDLE_DIR W2V_LAYER12_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER12_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-conv-residual MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_LAYER12_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer12-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER12_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER12_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-qkv MODEL_BUNDLE_DIR W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention W2V_LAYER13_Q_F32 W2V_LAYER13_K_F32 W2V_LAYER13_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER13_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention-project MODEL_BUNDLE_DIR W2V_LAYER13_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention-residual W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_LAYER13_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-attention-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-glu MODEL_BUNDLE_DIR W2V_LAYER13_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER13_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-conv-residual MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_LAYER13_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer13-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER13_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER13_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER14_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-residual MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_LAYER14_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER14_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-glu MODEL_BUNDLE_DIR W2V_LAYER14_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-conv-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention-residual W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_LAYER14_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention-project MODEL_BUNDLE_DIR W2V_LAYER14_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-attention W2V_LAYER14_Q_F32 W2V_LAYER14_K_F32 W2V_LAYER14_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER14_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer14-qkv MODEL_BUNDLE_DIR W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer15-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER14_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-qkv MODEL_BUNDLE_DIR W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention W2V_LAYER15_Q_F32 W2V_LAYER15_K_F32 W2V_LAYER15_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER15_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-ffn1-residual\",\n";
    std::cout << "    \"--clone-w2v-layer16-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention\",\n";
    std::cout << "    \"--clone-w2v-layer16-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer15-ffn2-residual\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-residual\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-depthwise\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-glu\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-norm\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention-norm\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention-residual\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention-project\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention\",\n";
    std::cout << "    \"--clone-w2v-layer15-qkv\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention-project MODEL_BUNDLE_DIR W2V_LAYER15_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention-residual W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_LAYER15_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-attention-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-glu MODEL_BUNDLE_DIR W2V_LAYER15_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER15_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-conv-residual MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_LAYER15_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer15-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER15_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER15_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN1_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-qkv MODEL_BUNDLE_DIR W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention W2V_LAYER16_Q_F32 W2V_LAYER16_K_F32 W2V_LAYER16_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER16_CONTEXT_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention-project MODEL_BUNDLE_DIR W2V_LAYER16_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention-residual W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_LAYER16_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-attention-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_NORM_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-glu MODEL_BUNDLE_DIR W2V_LAYER16_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_GLU_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER16_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_DEPTHWISE_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-conv-residual MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_LAYER16_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer16-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER16_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN2_RESIDUAL_F32\",\n";
    std::cout << "    \"--clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32 W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32\",\n";
    std::cout << "    \"--clone-w2v-normalize MODEL_BUNDLE_DIR W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32\",\n";
    std::cout << "    \"--clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32 SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32\",\n";
    std::cout << "    \"--clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS OUTPUT_DIR\",\n";
    std::cout << "    \"--clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32\",\n";
    std::cout << "    \"--clone-encoder-readiness FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32\",\n";
    std::cout << "    \"--clone-write-voice-bundle PREPROCESS_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS MEL_F32 OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "    \"--clone-write-voice-bundle-from-features FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "    \"--clone AUDIO_WAV OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "    \"--clone-real MODEL_BUNDLE_DIR FEATURE_MANIFEST W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "    \"--text-readiness MODEL_BUNDLE_DIR TEXT\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"text_readiness_command\": \"--text-readiness MODEL_BUNDLE_DIR TEXT\",\n";
    std::cout << "  \"synthesis_preflight_command\": \"--preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT\",\n";
    std::cout << "  \"clone_audio_preflight_command\": \"--clone-preflight AUDIO_WAV\",\n";
    std::cout << "  \"clone_audio_preprocess_command\": \"--clone-preprocess AUDIO_WAV OUTPUT_F32\",\n";
    std::cout << "  \"clone_readiness_command\": \"--clone-readiness PREPROCESS_MANIFEST\",\n";
    std::cout << "  \"clone_mel_extraction_command\": \"--clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32\",\n";
    std::cout << "  \"clone_fbank_extraction_command\": \"--clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32\",\n";
    std::cout << "  \"clone_feature_prepare_command\": \"--clone-prepare-features AUDIO_WAV OUTPUT_DIR\",\n";
    std::cout << "  \"clone_feature_readiness_command\": \"--clone-feature-readiness FEATURE_MANIFEST\",\n";
    std::cout << "  \"clone_command\": \"--clone AUDIO_WAV OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "  \"clone_real_command\": \"--clone-real MODEL_BUNDLE_DIR FEATURE_MANIFEST W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "  \"clone_encoder_model_readiness_command\": \"--clone-encoder-model-readiness MODEL_BUNDLE_DIR\",\n";
    std::cout << "  \"clone_campplus_style_readiness_command\": \"--clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST S2MEL_STYLE_F32\",\n";
    std::cout << "  \"clone_campplus_style_from_features_command\": \"--clone-campplus-style-from-features MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32\",\n";
    std::cout << "  \"clone_campplus_head_golden_command\": \"--clone-campplus-head-golden MODEL_BUNDLE_DIR FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR\",\n";
    std::cout << "  \"clone_w2v_feature_project_command\": \"--clone-w2v-feature-project MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32\",\n";
    std::cout << "  \"clone_w2v_encoder_command\": \"--clone-w2v-encoder MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_SPK_COND_F32\",\n";
    std::cout << "  \"clone_w2v_extract_features_command\": \"--clone-w2v-extract-features PREPROCESS_MANIFEST OUTPUT_FEATURES_F32 OUTPUT_MASK_U32\",\n";
    std::cout << "  \"clone_w2v_layer0_ffn1_norm_command\": \"--clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_ffn1_intermediate_command\": \"--clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_ffn1_activate_command\": \"--clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_ffn1_output_command\": \"--clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_ffn1_residual_command\": \"--clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32 W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_qkv_command\": \"--clone-w2v-layer0-qkv MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer0_attention_command\": \"--clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_attention_project_command\": \"--clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_attention_residual_command\": \"--clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32 W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_attention_norm_command\": \"--clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_conv_norm_command\": \"--clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_conv_glu_command\": \"--clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32 W2V_TOKENS OUTPUT_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_conv_depthwise_command\": \"--clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_conv_residual_command\": \"--clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_ffn2_residual_command\": \"--clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer0_final_norm_command\": \"--clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_ffn1_norm_command\": \"--clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_ffn1_intermediate_command\": \"--clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_ffn1_activate_command\": \"--clone-w2v-layer1-ffn1-activate W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_ACTIVATED_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_ffn1_output_command\": \"--clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_ffn1_residual_command\": \"--clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32 W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_qkv_command\": \"--clone-w2v-layer1-qkv MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer1_attention_command\": \"--clone-w2v-layer1-attention W2V_LAYER1_Q_F32 W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER1_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_attention_project_command\": \"--clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_attention_residual_command\": \"--clone-w2v-layer1-attention-residual W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_attention_norm_command\": \"--clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_conv_norm_command\": \"--clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_conv_glu_command\": \"--clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_conv_depthwise_command\": \"--clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_conv_residual_command\": \"--clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_ffn2_residual_command\": \"--clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer1_final_norm_command\": \"--clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_ffn1_norm_command\": \"--clone-w2v-layer2-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER1_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_ffn1_intermediate_command\": \"--clone-w2v-layer2-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_ffn1_activate_command\": \"--clone-w2v-layer2-ffn1-activate W2V_LAYER2_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_ACTIVATED_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_ffn1_output_command\": \"--clone-w2v-layer2-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_OUTPUT_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_ffn1_residual_command\": \"--clone-w2v-layer2-ffn1-residual W2V_LAYER1_F32 W2V_LAYER2_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_qkv_command\": \"--clone-w2v-layer2-qkv MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer2_attention_command\": \"--clone-w2v-layer2-attention W2V_LAYER2_Q_F32 W2V_LAYER2_K_F32 W2V_LAYER2_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER2_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_attention_project_command\": \"--clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_attention_residual_command\": \"--clone-w2v-layer2-attention-residual W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_attention_norm_command\": \"--clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_conv_norm_command\": \"--clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_conv_glu_command\": \"--clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_conv_depthwise_command\": \"--clone-w2v-layer2-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER2_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_conv_residual_command\": \"--clone-w2v-layer2-conv-residual MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_LAYER2_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer2_ffn2_residual_command\": \"--clone-w2v-layer2-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER2_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_ffn1_norm_command\": \"--clone-w2v-layer3-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_ffn1_intermediate_command\": \"--clone-w2v-layer3-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_ffn1_activate_command\": \"--clone-w2v-layer3-ffn1-activate W2V_LAYER3_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_ACTIVATED_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_ffn1_output_command\": \"--clone-w2v-layer3-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_OUTPUT_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_ffn1_residual_command\": \"--clone-w2v-layer3-ffn1-residual W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_LAYER3_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_qkv_command\": \"--clone-w2v-layer3-qkv MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer3_attention_command\": \"--clone-w2v-layer3-attention W2V_LAYER3_Q_F32 W2V_LAYER3_K_F32 W2V_LAYER3_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER3_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_attention_project_command\": \"--clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_attention_residual_command\": \"--clone-w2v-layer3-attention-residual W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_attention_norm_command\": \"--clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_conv_norm_command\": \"--clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_conv_glu_command\": \"--clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_conv_depthwise_command\": \"--clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_conv_residual_command\": \"--clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_ffn2_residual_command\": \"--clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer3_final_norm_command\": \"--clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_ffn1_norm_command\": \"--clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_ffn1_intermediate_command\": \"--clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_ffn1_activate_command\": \"--clone-w2v-layer4-ffn1-activate W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_ACTIVATED_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_ffn1_output_command\": \"--clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_ffn1_residual_command\": \"--clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32 W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_qkv_command\": \"--clone-w2v-layer4-qkv MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer4_attention_command\": \"--clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32 W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER4_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_attention_project_command\": \"--clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_attention_residual_command\": \"--clone-w2v-layer4-attention-residual W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_attention_norm_command\": \"--clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_conv_norm_command\": \"--clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_conv_glu_command\": \"--clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_conv_depthwise_command\": \"--clone-w2v-layer4-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER4_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_conv_residual_command\": \"--clone-w2v-layer4-conv-residual MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_LAYER4_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer4_ffn2_residual_command\": \"--clone-w2v-layer4-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER4_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_ffn1_residual_command\": \"--clone-w2v-layer5-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER4_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_qkv_command\": \"--clone-w2v-layer5-qkv MODEL_BUNDLE_DIR W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer5_attention_command\": \"--clone-w2v-layer5-attention W2V_LAYER5_Q_F32 W2V_LAYER5_K_F32 W2V_LAYER5_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER5_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_attention_project_command\": \"--clone-w2v-layer5-attention-project MODEL_BUNDLE_DIR W2V_LAYER5_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_attention_residual_command\": \"--clone-w2v-layer5-attention-residual W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_LAYER5_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_attention_norm_command\": \"--clone-w2v-layer5-attention-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_conv_norm_command\": \"--clone-w2v-layer5-conv-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_conv_glu_command\": \"--clone-w2v-layer5-conv-glu MODEL_BUNDLE_DIR W2V_LAYER5_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_conv_depthwise_command\": \"--clone-w2v-layer5-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER5_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_conv_residual_command\": \"--clone-w2v-layer5-conv-residual MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_LAYER5_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer5_ffn2_residual_command\": \"--clone-w2v-layer5-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER5_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_ffn1_residual_command\": \"--clone-w2v-layer6-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER5_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_qkv_command\": \"--clone-w2v-layer6-qkv MODEL_BUNDLE_DIR W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer6_attention_command\": \"--clone-w2v-layer6-attention W2V_LAYER6_Q_F32 W2V_LAYER6_K_F32 W2V_LAYER6_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER6_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_attention_project_command\": \"--clone-w2v-layer6-attention-project MODEL_BUNDLE_DIR W2V_LAYER6_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_attention_residual_command\": \"--clone-w2v-layer6-attention-residual W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_LAYER6_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_attention_norm_command\": \"--clone-w2v-layer6-attention-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_conv_norm_command\": \"--clone-w2v-layer6-conv-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_conv_glu_command\": \"--clone-w2v-layer6-conv-glu MODEL_BUNDLE_DIR W2V_LAYER6_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_conv_depthwise_command\": \"--clone-w2v-layer6-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER6_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_conv_residual_command\": \"--clone-w2v-layer6-conv-residual MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_LAYER6_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer6_ffn2_residual_command\": \"--clone-w2v-layer6-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER6_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_ffn1_residual_command\": \"--clone-w2v-layer7-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER6_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_qkv_command\": \"--clone-w2v-layer7-qkv MODEL_BUNDLE_DIR W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer7_attention_command\": \"--clone-w2v-layer7-attention W2V_LAYER7_Q_F32 W2V_LAYER7_K_F32 W2V_LAYER7_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER7_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_attention_project_command\": \"--clone-w2v-layer7-attention-project MODEL_BUNDLE_DIR W2V_LAYER7_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_attention_residual_command\": \"--clone-w2v-layer7-attention-residual W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_LAYER7_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_attention_norm_command\": \"--clone-w2v-layer7-attention-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_conv_norm_command\": \"--clone-w2v-layer7-conv-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_conv_glu_command\": \"--clone-w2v-layer7-conv-glu MODEL_BUNDLE_DIR W2V_LAYER7_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_conv_depthwise_command\": \"--clone-w2v-layer7-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER7_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_conv_residual_command\": \"--clone-w2v-layer7-conv-residual MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_LAYER7_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer7_ffn2_residual_command\": \"--clone-w2v-layer7-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER7_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_ffn1_residual_command\": \"--clone-w2v-layer8-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER7_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_qkv_command\": \"--clone-w2v-layer8-qkv MODEL_BUNDLE_DIR W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer8_attention_command\": \"--clone-w2v-layer8-attention W2V_LAYER8_Q_F32 W2V_LAYER8_K_F32 W2V_LAYER8_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER8_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_attention_project_command\": \"--clone-w2v-layer8-attention-project MODEL_BUNDLE_DIR W2V_LAYER8_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_attention_residual_command\": \"--clone-w2v-layer8-attention-residual W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_LAYER8_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_attention_norm_command\": \"--clone-w2v-layer8-attention-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_conv_norm_command\": \"--clone-w2v-layer8-conv-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_conv_glu_command\": \"--clone-w2v-layer8-conv-glu MODEL_BUNDLE_DIR W2V_LAYER8_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_conv_depthwise_command\": \"--clone-w2v-layer8-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER8_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_conv_residual_command\": \"--clone-w2v-layer8-conv-residual MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_LAYER8_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer8_ffn2_residual_command\": \"--clone-w2v-layer8-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER8_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_ffn1_residual_command\": \"--clone-w2v-layer9-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER8_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_qkv_command\": \"--clone-w2v-layer9-qkv MODEL_BUNDLE_DIR W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer9_attention_command\": \"--clone-w2v-layer9-attention W2V_LAYER9_Q_F32 W2V_LAYER9_K_F32 W2V_LAYER9_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER9_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_attention_project_command\": \"--clone-w2v-layer9-attention-project MODEL_BUNDLE_DIR W2V_LAYER9_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_attention_residual_command\": \"--clone-w2v-layer9-attention-residual W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_LAYER9_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_attention_norm_command\": \"--clone-w2v-layer9-attention-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_conv_norm_command\": \"--clone-w2v-layer9-conv-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_conv_glu_command\": \"--clone-w2v-layer9-conv-glu MODEL_BUNDLE_DIR W2V_LAYER9_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_conv_depthwise_command\": \"--clone-w2v-layer9-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER9_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_conv_residual_command\": \"--clone-w2v-layer9-conv-residual MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_LAYER9_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer9_ffn2_residual_command\": \"--clone-w2v-layer9-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER9_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_ffn1_residual_command\": \"--clone-w2v-layer10-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER9_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_qkv_command\": \"--clone-w2v-layer10-qkv MODEL_BUNDLE_DIR W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer10_attention_command\": \"--clone-w2v-layer10-attention W2V_LAYER10_Q_F32 W2V_LAYER10_K_F32 W2V_LAYER10_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER10_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_attention_project_command\": \"--clone-w2v-layer10-attention-project MODEL_BUNDLE_DIR W2V_LAYER10_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_attention_residual_command\": \"--clone-w2v-layer10-attention-residual W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_LAYER10_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_attention_norm_command\": \"--clone-w2v-layer10-attention-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_conv_norm_command\": \"--clone-w2v-layer10-conv-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_conv_glu_command\": \"--clone-w2v-layer10-conv-glu MODEL_BUNDLE_DIR W2V_LAYER10_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_conv_depthwise_command\": \"--clone-w2v-layer10-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER10_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_conv_residual_command\": \"--clone-w2v-layer10-conv-residual MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_LAYER10_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer10_ffn2_residual_command\": \"--clone-w2v-layer10-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER10_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_ffn1_residual_command\": \"--clone-w2v-layer11-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER10_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_qkv_command\": \"--clone-w2v-layer11-qkv MODEL_BUNDLE_DIR W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer11_attention_command\": \"--clone-w2v-layer11-attention W2V_LAYER11_Q_F32 W2V_LAYER11_K_F32 W2V_LAYER11_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER11_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_attention_project_command\": \"--clone-w2v-layer11-attention-project MODEL_BUNDLE_DIR W2V_LAYER11_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_attention_residual_command\": \"--clone-w2v-layer11-attention-residual W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_LAYER11_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_attention_norm_command\": \"--clone-w2v-layer11-attention-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_conv_norm_command\": \"--clone-w2v-layer11-conv-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_conv_glu_command\": \"--clone-w2v-layer11-conv-glu MODEL_BUNDLE_DIR W2V_LAYER11_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_conv_depthwise_command\": \"--clone-w2v-layer11-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER11_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_conv_residual_command\": \"--clone-w2v-layer11-conv-residual MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_LAYER11_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer11_ffn2_residual_command\": \"--clone-w2v-layer11-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER11_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_ffn1_residual_command\": \"--clone-w2v-layer12-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER11_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_qkv_command\": \"--clone-w2v-layer12-qkv MODEL_BUNDLE_DIR W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer12_attention_command\": \"--clone-w2v-layer12-attention W2V_LAYER12_Q_F32 W2V_LAYER12_K_F32 W2V_LAYER12_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER12_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_attention_project_command\": \"--clone-w2v-layer12-attention-project MODEL_BUNDLE_DIR W2V_LAYER12_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_attention_residual_command\": \"--clone-w2v-layer12-attention-residual W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_LAYER12_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_attention_norm_command\": \"--clone-w2v-layer12-attention-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_conv_norm_command\": \"--clone-w2v-layer12-conv-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_conv_glu_command\": \"--clone-w2v-layer12-conv-glu MODEL_BUNDLE_DIR W2V_LAYER12_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_conv_depthwise_command\": \"--clone-w2v-layer12-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER12_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_conv_residual_command\": \"--clone-w2v-layer12-conv-residual MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_LAYER12_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer12_ffn2_residual_command\": \"--clone-w2v-layer12-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER12_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_ffn1_residual_command\": \"--clone-w2v-layer13-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER12_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_qkv_command\": \"--clone-w2v-layer13-qkv MODEL_BUNDLE_DIR W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer13_attention_command\": \"--clone-w2v-layer13-attention W2V_LAYER13_Q_F32 W2V_LAYER13_K_F32 W2V_LAYER13_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER13_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_attention_project_command\": \"--clone-w2v-layer13-attention-project MODEL_BUNDLE_DIR W2V_LAYER13_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_attention_residual_command\": \"--clone-w2v-layer13-attention-residual W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_LAYER13_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_attention_norm_command\": \"--clone-w2v-layer13-attention-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_conv_norm_command\": \"--clone-w2v-layer13-conv-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_conv_glu_command\": \"--clone-w2v-layer13-conv-glu MODEL_BUNDLE_DIR W2V_LAYER13_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_conv_depthwise_command\": \"--clone-w2v-layer13-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER13_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_conv_residual_command\": \"--clone-w2v-layer13-conv-residual MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_LAYER13_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer13_ffn2_residual_command\": \"--clone-w2v-layer13-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER13_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_ffn1_residual_command\": \"--clone-w2v-layer14-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER13_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_ffn2_residual_command\": \"--clone-w2v-layer14-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER14_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_conv_residual_command\": \"--clone-w2v-layer14-conv-residual MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_LAYER14_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_conv_depthwise_command\": \"--clone-w2v-layer14-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER14_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_conv_glu_command\": \"--clone-w2v-layer14-conv-glu MODEL_BUNDLE_DIR W2V_LAYER14_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_conv_norm_command\": \"--clone-w2v-layer14-conv-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_attention_norm_command\": \"--clone-w2v-layer14-attention-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_attention_residual_command\": \"--clone-w2v-layer14-attention-residual W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_LAYER14_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_attention_project_command\": \"--clone-w2v-layer14-attention-project MODEL_BUNDLE_DIR W2V_LAYER14_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_attention_command\": \"--clone-w2v-layer14-attention W2V_LAYER14_Q_F32 W2V_LAYER14_K_F32 W2V_LAYER14_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER14_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer14_qkv_command\": \"--clone-w2v-layer14-qkv MODEL_BUNDLE_DIR W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer15_ffn1_residual_command\": \"--clone-w2v-layer15-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER14_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_qkv_command\": \"--clone-w2v-layer15-qkv MODEL_BUNDLE_DIR W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer15_attention_command\": \"--clone-w2v-layer15-attention W2V_LAYER15_Q_F32 W2V_LAYER15_K_F32 W2V_LAYER15_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER15_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_attention_project_command\": \"--clone-w2v-layer15-attention-project MODEL_BUNDLE_DIR W2V_LAYER15_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_attention_residual_command\": \"--clone-w2v-layer15-attention-residual W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_LAYER15_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_attention_norm_command\": \"--clone-w2v-layer15-attention-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_conv_norm_command\": \"--clone-w2v-layer15-conv-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_conv_glu_command\": \"--clone-w2v-layer15-conv-glu MODEL_BUNDLE_DIR W2V_LAYER15_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_conv_depthwise_command\": \"--clone-w2v-layer15-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER15_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_conv_residual_command\": \"--clone-w2v-layer15-conv-residual MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_LAYER15_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer15_ffn2_residual_command\": \"--clone-w2v-layer15-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER15_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_ffn1_residual_command\": \"--clone-w2v-layer16-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER15_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN1_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_qkv_command\": \"--clone-w2v-layer16-qkv MODEL_BUNDLE_DIR W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_w2v_layer16_attention_command\": \"--clone-w2v-layer16-attention W2V_LAYER16_Q_F32 W2V_LAYER16_K_F32 W2V_LAYER16_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER16_CONTEXT_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_attention_project_command\": \"--clone-w2v-layer16-attention-project MODEL_BUNDLE_DIR W2V_LAYER16_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_attention_residual_command\": \"--clone-w2v-layer16-attention-residual W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_LAYER16_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_attention_norm_command\": \"--clone-w2v-layer16-attention-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_conv_norm_command\": \"--clone-w2v-layer16-conv-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_NORM_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_conv_glu_command\": \"--clone-w2v-layer16-conv-glu MODEL_BUNDLE_DIR W2V_LAYER16_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_GLU_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_conv_depthwise_command\": \"--clone-w2v-layer16-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER16_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_DEPTHWISE_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_conv_residual_command\": \"--clone-w2v-layer16-conv-residual MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_LAYER16_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer16_ffn2_residual_command\": \"--clone-w2v-layer16-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER16_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN2_RESIDUAL_F32\",\n";
    std::cout << "  \"clone_w2v_layer17_final_norm_command\": \"--clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32 W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32\",\n";
    std::cout << "  \"clone_w2v_normalize_command\": \"--clone-w2v-normalize MODEL_BUNDLE_DIR W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32\",\n";
    std::cout << "  \"clone_semantic_quantize_command\": \"--clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32 SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32\",\n";
    std::cout << "  \"clone_semantic_prompt_from_spk_cond_command\": \"--clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS OUTPUT_DIR\",\n";
    std::cout << "  \"clone_s2mel_prompt_from_sref_command\": \"--clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32\",\n";
    std::cout << "  \"clone_encoder_readiness_command\": \"--clone-encoder-readiness FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32\",\n";
    std::cout << "  \"clone_voice_bundle_writer_command\": \"--clone-write-voice-bundle PREPROCESS_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS MEL_F32 OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "  \"clone_voice_bundle_writer_from_features_command\": \"--clone-write-voice-bundle-from-features FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS OUTPUT_VOICE_BUNDLE\",\n";
    std::cout << "  \"bundle_readiness_command\": \"--readiness MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR\",\n";
    std::cout << "  \"synthesis_command\": \"MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT OUTPUT_WAV [PRESET]\",\n";
    std::cout << "  \"python_boundaries\": [\n";
    std::cout << "    \"full TextNormalizer/SentencePiece for general text\",\n";
    std::cout << "    \"clone-time semantic/acoustic speech encoders for voice tensor creation\"\n";
    std::cout << "  ],\n";
    std::cout << "  \"start_sh_replacement_audit\": {\n";
    std::cout << "    \"can_replace_start_sh_full_clone_tts\": false,\n";
    std::cout << "    \"can_replace_start_sh_cached_voice_cjk\": true,\n";
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
    std::cout << "      \"--clone-w2v-encoder MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_SPK_COND_F32\",\n";
    std::cout << "      \"--clone-w2v-extract-features PREPROCESS_MANIFEST OUTPUT_FEATURES_F32 OUTPUT_MASK_U32\",\n";
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
    std::cout << "      \"--clone-w2v-layer6-attention-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-conv-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-conv-glu MODEL_BUNDLE_DIR W2V_LAYER6_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER6_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-conv-residual MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_LAYER6_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer6-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER6_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER6_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-qkv MODEL_BUNDLE_DIR W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer7-attention W2V_LAYER7_Q_F32 W2V_LAYER7_K_F32 W2V_LAYER7_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER7_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-attention-project MODEL_BUNDLE_DIR W2V_LAYER7_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-attention-residual W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_LAYER7_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-attention-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-conv-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-conv-glu MODEL_BUNDLE_DIR W2V_LAYER7_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER7_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-conv-residual MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_LAYER7_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer7-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER7_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER7_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-qkv MODEL_BUNDLE_DIR W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer8-attention W2V_LAYER8_Q_F32 W2V_LAYER8_K_F32 W2V_LAYER8_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER8_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-attention-project MODEL_BUNDLE_DIR W2V_LAYER8_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-attention-residual W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_LAYER8_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer8-attention-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_NORM_F32\",\n";
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
    std::cout << "      \"--clone-w2v-layer14-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER14_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-conv-residual MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_LAYER14_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER14_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-conv-glu MODEL_BUNDLE_DIR W2V_LAYER14_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-conv-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER14_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-attention-norm MODEL_BUNDLE_DIR W2V_LAYER14_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-attention-residual W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_LAYER14_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-attention-project MODEL_BUNDLE_DIR W2V_LAYER14_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER14_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-attention W2V_LAYER14_Q_F32 W2V_LAYER14_K_F32 W2V_LAYER14_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER14_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer14-qkv MODEL_BUNDLE_DIR W2V_LAYER14_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer15-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER14_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-qkv MODEL_BUNDLE_DIR W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer15-attention W2V_LAYER15_Q_F32 W2V_LAYER15_K_F32 W2V_LAYER15_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER15_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-attention-project MODEL_BUNDLE_DIR W2V_LAYER15_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-attention-residual W2V_LAYER15_FFN1_RESIDUAL_F32 W2V_LAYER15_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-attention-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-conv-norm MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-conv-glu MODEL_BUNDLE_DIR W2V_LAYER15_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER15_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-conv-residual MODEL_BUNDLE_DIR W2V_LAYER15_ATTENTION_NORM_F32 W2V_LAYER15_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER15_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer15-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER15_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER15_FFN2_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER15_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN1_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-qkv MODEL_BUNDLE_DIR W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR\",\n";
    std::cout << "      \"--clone-w2v-layer16-attention W2V_LAYER16_Q_F32 W2V_LAYER16_K_F32 W2V_LAYER16_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER16_CONTEXT_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-attention-project MODEL_BUNDLE_DIR W2V_LAYER16_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-attention-residual W2V_LAYER16_FFN1_RESIDUAL_F32 W2V_LAYER16_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-attention-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_ATTENTION_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-conv-norm MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_NORM_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-conv-glu MODEL_BUNDLE_DIR W2V_LAYER16_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_GLU_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER16_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_DEPTHWISE_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-conv-residual MODEL_BUNDLE_DIR W2V_LAYER16_ATTENTION_NORM_F32 W2V_LAYER16_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER16_CONV_RESIDUAL_F32\",\n";
    std::cout << "      \"--clone-w2v-layer16-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER16_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER16_FFN2_RESIDUAL_F32\",\n";
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
    std::cout << "  \"presets\": [\n";
    for (size_t i = 0; i < presets.size(); ++i) {
        std::cout << "    {\n";
        std::cout << "      \"name\": \"" << json_escape(presets[i].name) << "\",\n";
        std::cout << "      \"max_codes_per_text_token\": " << kMaxCodesPerTextToken << ",\n";
        std::cout << "      \"steps\": " << presets[i].steps << ",\n";
        std::cout << "      \"max_text_tokens_per_segment\": " << presets[i].max_text_tokens_per_segment << ",\n";
        std::cout << "      \"interval_silence_ms\": " << presets[i].interval_silence_ms << "\n";
        std::cout << "    }" << (i + 1 == presets.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

