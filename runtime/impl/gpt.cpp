bool run_gpt_layer_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    std::vector<float> input(tokens * 1280);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.007f) * 0.5f + std::cos(static_cast<float>(i % 97) * 0.03f);
    }
    auto ref = run_gpt_layer_cpu(bundle, input, tokens);
    auto got = run_gpt_layer_metal(metal, bundle, input, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_gpt_layer\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"input_dim\": 1280,\n";
    std::cout << "  \"output_dim\": 1024,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

struct GptPrepareInputsOutputs {
    std::vector<uint32_t> fake_inputs;
    std::vector<float> inputs_embeds;
    std::vector<uint32_t> attention_mask;
    uint32_t valid_text_tokens = 0;
};

GptPrepareInputsOutputs run_gpt_prepare_inputs_cpu(const mit2::Bundle& bundle, const std::vector<float>& conditional_latents, const std::vector<uint32_t>& text_inputs, uint32_t cond_tokens, uint32_t text_slots) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t start_text_token = 0;
    constexpr uint32_t stop_text_token = 1;
    constexpr uint32_t start_mel_token = 8192;
    auto text_embedding = tensor_as_f32(bundle, "gpt.text_embedding.weight");
    auto text_pos_embedding = tensor_as_f32(bundle, "gpt.text_pos_embedding.emb.weight");
    std::vector<uint32_t> filtered;
    for (uint32_t id : text_inputs) {
        if (id != start_text_token && id != stop_text_token) {
            filtered.push_back(id);
        }
    }
    std::vector<uint32_t> prepared_text;
    prepared_text.reserve(filtered.size() + 2);
    prepared_text.push_back(start_text_token);
    prepared_text.insert(prepared_text.end(), filtered.begin(), filtered.end());
    prepared_text.push_back(stop_text_token);
    const uint32_t text_tokens = static_cast<uint32_t>(prepared_text.size());
    const uint32_t target_len = cond_tokens + text_slots + 2;
    const uint32_t padding = text_slots + 2 - text_tokens;
    GptPrepareInputsOutputs out;
    out.valid_text_tokens = static_cast<uint32_t>(filtered.size());
    out.inputs_embeds.assign(static_cast<size_t>(target_len) * width, 0.0f);
    out.attention_mask.assign(static_cast<size_t>(target_len) + 1, 1);
    for (uint32_t i = 0; i < padding; ++i) {
        out.attention_mask[i] = 0;
    }
    std::copy(conditional_latents.begin(), conditional_latents.end(), out.inputs_embeds.begin() + static_cast<size_t>(padding) * width);
    const size_t text_offset = static_cast<size_t>(padding + cond_tokens) * width;
    for (uint32_t t = 0; t < text_tokens; ++t) {
        const uint32_t id = prepared_text[t];
        for (uint32_t c = 0; c < width; ++c) {
            out.inputs_embeds[text_offset + static_cast<size_t>(t) * width + c] =
                text_embedding[static_cast<size_t>(id) * width + c] + text_pos_embedding[static_cast<size_t>(t) * width + c];
        }
    }
    out.fake_inputs.assign(static_cast<size_t>(target_len) + 1, 1);
    out.fake_inputs.back() = start_mel_token;
    return out;
}

GptPrepareInputsOutputs run_gpt_prepare_inputs_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& conditional_latents, const std::vector<uint32_t>& text_inputs, uint32_t cond_tokens, uint32_t text_slots) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t start_text_token = 0;
    constexpr uint32_t stop_text_token = 1;
    constexpr uint32_t start_mel_token = 8192;
    auto text_embedding = tensor_as_f32(bundle, "gpt.text_embedding.weight");
    auto text_pos_embedding = tensor_as_f32(bundle, "gpt.text_pos_embedding.emb.weight");
    std::vector<uint32_t> filtered;
    for (uint32_t id : text_inputs) {
        if (id != start_text_token && id != stop_text_token) {
            filtered.push_back(id);
        }
    }
    std::vector<uint32_t> prepared_text;
    prepared_text.reserve(filtered.size() + 2);
    prepared_text.push_back(start_text_token);
    prepared_text.insert(prepared_text.end(), filtered.begin(), filtered.end());
    prepared_text.push_back(stop_text_token);
    const uint32_t text_tokens = static_cast<uint32_t>(prepared_text.size());
    const uint32_t target_len = cond_tokens + text_slots + 2;
    const uint32_t padding = text_slots + 2 - text_tokens;
    auto text_emb = metal.embedding_f32_resident("gpt.text_embedding.weight.resident", text_embedding, prepared_text, width);
    std::vector<uint32_t> pos_ids(text_tokens);
    for (uint32_t i = 0; i < text_tokens; ++i) {
        pos_ids[i] = i;
    }
    auto pos_emb = metal.embedding_f32_resident("gpt.text_pos_embedding.emb.weight.resident", text_pos_embedding, pos_ids, width);
    GptPrepareInputsOutputs out;
    out.valid_text_tokens = static_cast<uint32_t>(filtered.size());
    out.inputs_embeds.assign(static_cast<size_t>(target_len) * width, 0.0f);
    out.attention_mask.assign(static_cast<size_t>(target_len) + 1, 1);
    for (uint32_t i = 0; i < padding; ++i) {
        out.attention_mask[i] = 0;
    }
    std::copy(conditional_latents.begin(), conditional_latents.end(), out.inputs_embeds.begin() + static_cast<size_t>(padding) * width);
    const size_t text_offset = static_cast<size_t>(padding + cond_tokens) * width;
    for (size_t i = 0; i < text_emb.size(); ++i) {
        out.inputs_embeds[text_offset + i] = text_emb[i] + pos_emb[i];
    }
    out.fake_inputs.assign(static_cast<size_t>(target_len) + 1, 1);
    out.fake_inputs.back() = start_mel_token;
    return out;
}

bool run_gpt_prepare_inputs_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t cond_tokens = 34;
    constexpr uint32_t text_slots = 8;
    constexpr uint32_t width = 1280;
    std::vector<float> cond(static_cast<size_t>(cond_tokens) * width);
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.007f) * 0.045f + std::cos(static_cast<float>(i % 251) * 0.011f) * 0.018f;
    }
    std::vector<uint32_t> text_inputs{0, 22, 37, 1, 49, 1, 0, 55};
    auto ref = run_gpt_prepare_inputs_cpu(bundle, cond, text_inputs, cond_tokens, text_slots);
    auto got = run_gpt_prepare_inputs_metal(metal, bundle, cond, text_inputs, cond_tokens, text_slots);
    const float embed_err = max_abs_error(got.inputs_embeds, ref.inputs_embeds);
    const bool fake_match = got.fake_inputs == ref.fake_inputs;
    const bool mask_match = got.attention_mask == ref.attention_mask;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_prepare_inputs\",\n";
    std::cout << "  \"cond_tokens\": " << cond_tokens << ",\n";
    std::cout << "  \"text_slots\": " << text_slots << ",\n";
    std::cout << "  \"valid_text_tokens\": " << got.valid_text_tokens << ",\n";
    std::cout << "  \"sequence_tokens\": " << (got.fake_inputs.size() - 1) << ",\n";
    std::cout << "  \"fake_inputs_match\": " << (fake_match ? "true" : "false") << ",\n";
    std::cout << "  \"attention_mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"inputs_embeds_max_abs_error\": " << embed_err << "\n";
    std::cout << "}\n";
    return fake_match && mask_match && embed_err <= 1e-6f;
}

bool export_gpt_frontend_tail(const std::string& bundle_dir,
                              const std::string& speech_cond_path,
                              const std::string& emovec_path,
                              const std::string& text_ids_path,
                              const std::string& output_conds_path,
                              const std::string& output_fake_path,
                              const std::string& output_inputs_path,
                              const std::string& output_mask_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t cond_num = 32;
    constexpr uint32_t width = 1280;
    auto speech = read_raw_f32(speech_cond_path);
    auto emovec = read_raw_f32(emovec_path);
    auto text_ids = read_raw_u32(text_ids_path);
    if (speech.size() != static_cast<size_t>(cond_num) * width) {
        throw std::runtime_error("GPT frontend tail speech conditioning must have shape [32,1280]");
    }
    if (emovec.size() != width) {
        throw std::runtime_error("GPT frontend tail emovec must have shape [1280]");
    }
    if (text_ids.empty()) {
        throw std::runtime_error("GPT frontend tail text ids must not be empty");
    }
    auto speed = tensor_as_f32(bundle, "gpt.speed_emb.weight");
    if (speed.size() != static_cast<size_t>(2) * width) {
        throw std::runtime_error("gpt.speed_emb.weight shape mismatch");
    }
    std::vector<float> conds(static_cast<size_t>(cond_num + 2) * width);
    for (uint32_t t = 0; t < cond_num; ++t) {
        for (uint32_t c = 0; c < width; ++c) {
            conds[static_cast<size_t>(t) * width + c] = speech[static_cast<size_t>(t) * width + c] + emovec[c];
        }
    }
    std::copy(speed.begin() + width,
              speed.begin() + static_cast<size_t>(2) * width,
              conds.begin() + static_cast<size_t>(cond_num) * width);
    std::copy(speed.begin(), speed.begin() + width, conds.begin() + static_cast<size_t>(cond_num + 1) * width);

    auto prefix = run_gpt_prepare_inputs_metal(metal, bundle, conds, text_ids, cond_num + 2, static_cast<uint32_t>(text_ids.size()));
    write_raw_f32(output_conds_path, conds);
    write_raw_u32(output_fake_path, prefix.fake_inputs);
    write_raw_f32(output_inputs_path, prefix.inputs_embeds);
    write_raw_u32(output_mask_path, prefix.attention_mask);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_frontend_tail_export\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"speech_cond_f32\": \"" << json_escape(speech_cond_path) << "\",\n";
    std::cout << "  \"emovec_f32\": \"" << json_escape(emovec_path) << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << json_escape(text_ids_path) << "\",\n";
    std::cout << "  \"output_conds_f32\": \"" << json_escape(output_conds_path) << "\",\n";
    std::cout << "  \"output_fake_u32\": \"" << json_escape(output_fake_path) << "\",\n";
    std::cout << "  \"output_inputs_f32\": \"" << json_escape(output_inputs_path) << "\",\n";
    std::cout << "  \"output_mask_u32\": \"" << json_escape(output_mask_path) << "\",\n";
    std::cout << "  \"cond_tokens\": " << (cond_num + 2) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << (prefix.inputs_embeds.size() / width) << ",\n";
    std::cout << "  \"fake_tokens\": " << prefix.fake_inputs.size() << ",\n";
    std::cout << "  \"nonfinite_conds\": " << count_nonfinite(conds) << ",\n";
    std::cout << "  \"nonfinite_inputs\": " << count_nonfinite(prefix.inputs_embeds) << "\n";
    std::cout << "}\n";
    return count_nonfinite(conds) == 0 && count_nonfinite(prefix.inputs_embeds) == 0;
}

std::vector<float> transpose_gpt_conv1d_weight(const std::vector<float>& weight_in_out, uint32_t in_dim, uint32_t out_dim) {
    std::vector<float> out(static_cast<size_t>(out_dim) * in_dim);
    for (uint32_t i = 0; i < in_dim; ++i) {
        for (uint32_t o = 0; o < out_dim; ++o) {
            out[static_cast<size_t>(o) * in_dim + i] = weight_in_out[static_cast<size_t>(i) * out_dim + o];
        }
    }
    return out;
}

struct GptTransformerLayerWeights {
    std::string key_prefix;
    std::vector<float> ln1_weight;
    std::vector<float> ln1_bias;
    std::vector<float> c_attn_weight;
    std::vector<float> c_attn_bias;
    std::vector<float> attn_proj_weight;
    std::vector<float> attn_proj_bias;
    std::vector<float> ln2_weight;
    std::vector<float> ln2_bias;
    std::vector<float> c_fc_weight;
    std::vector<float> c_fc_bias;
    std::vector<float> mlp_proj_weight;
    std::vector<float> mlp_proj_bias;
};

struct GptDecodeWeights {
    std::vector<GptTransformerLayerWeights> layers;
    std::vector<float> ln_f_weight;
    std::vector<float> ln_f_bias;
    std::vector<float> final_norm_weight;
    std::vector<float> final_norm_bias;
    std::vector<float> head_weight;
    std::vector<float> head_bias;
    std::vector<float> mel_embedding;
    std::vector<float> mel_pos_embedding;
};

GptTransformerLayerWeights load_gpt_transformer_layer_weights(const mit2::Bundle& bundle, uint32_t layer) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    const std::string base = "gpt.gpt.h." + std::to_string(layer);
    GptTransformerLayerWeights weights;
    weights.key_prefix = base;
    weights.ln1_weight = tensor_as_f32(bundle, base + ".ln_1.weight");
    weights.ln1_bias = tensor_as_f32(bundle, base + ".ln_1.bias");
    weights.c_attn_weight = transpose_gpt_conv1d_weight(tensor_as_f32(bundle, base + ".attn.c_attn.weight"), width, qkv_width);
    weights.c_attn_bias = tensor_as_f32(bundle, base + ".attn.c_attn.bias");
    weights.attn_proj_weight = transpose_gpt_conv1d_weight(tensor_as_f32(bundle, base + ".attn.c_proj.weight"), width, width);
    weights.attn_proj_bias = tensor_as_f32(bundle, base + ".attn.c_proj.bias");
    weights.ln2_weight = tensor_as_f32(bundle, base + ".ln_2.weight");
    weights.ln2_bias = tensor_as_f32(bundle, base + ".ln_2.bias");
    weights.c_fc_weight = transpose_gpt_conv1d_weight(tensor_as_f32(bundle, base + ".mlp.c_fc.weight"), width, mlp_width);
    weights.c_fc_bias = tensor_as_f32(bundle, base + ".mlp.c_fc.bias");
    weights.mlp_proj_weight = transpose_gpt_conv1d_weight(tensor_as_f32(bundle, base + ".mlp.c_proj.weight"), mlp_width, width);
    weights.mlp_proj_bias = tensor_as_f32(bundle, base + ".mlp.c_proj.bias");
    return weights;
}

GptDecodeWeights load_gpt_decode_weights(const mit2::Bundle& bundle) {
    GptDecodeWeights weights;
    weights.layers.reserve(24);
    for (uint32_t layer = 0; layer < 24; ++layer) {
        weights.layers.push_back(load_gpt_transformer_layer_weights(bundle, layer));
    }
    weights.ln_f_weight = tensor_as_f32(bundle, "gpt.gpt.ln_f.weight");
    weights.ln_f_bias = tensor_as_f32(bundle, "gpt.gpt.ln_f.bias");
    weights.final_norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    weights.final_norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    weights.head_weight = tensor_as_f32(bundle, "gpt.mel_head.weight");
    weights.head_bias = tensor_as_f32(bundle, "gpt.mel_head.bias");
    weights.mel_embedding = tensor_as_f32(bundle, "gpt.mel_embedding.weight");
    weights.mel_pos_embedding = tensor_as_f32(bundle, "gpt.mel_pos_embedding.emb.weight");
    return weights;
}

std::string resident_weight_key(const GptTransformerLayerWeights& weights, const std::string& suffix) {
    return weights.key_prefix + "." + suffix + ".weight.resident";
}

std::string resident_bias_key(const GptTransformerLayerWeights& weights, const std::string& suffix) {
    return weights.key_prefix + "." + suffix + ".bias.resident";
}

struct GptLayer0QkvOutputs {
    std::vector<float> ln1;
    std::vector<float> qkv;
};

GptLayer0QkvOutputs run_gpt_layer0_qkv_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    auto ln_weight = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_1.weight");
    auto ln_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_1.bias");
    auto c_attn_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_attn.weight");
    auto c_attn_weight = transpose_gpt_conv1d_weight(c_attn_weight_io, width, qkv_width);
    auto c_attn_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_attn.bias");
    GptLayer0QkvOutputs out;
    out.ln1.resize(static_cast<size_t>(tokens) * width);
    out.qkv.resize(static_cast<size_t>(tokens) * qkv_width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + static_cast<size_t>(t) * width, input.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = cpu_layernorm(row, ln_weight, ln_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), out.ln1.begin() + static_cast<size_t>(t) * width);
        auto qkv = cpu_linear(c_attn_weight, c_attn_bias, normed, qkv_width, width);
        std::copy(qkv.begin(), qkv.end(), out.qkv.begin() + static_cast<size_t>(t) * qkv_width);
    }
    return out;
}

GptLayer0QkvOutputs run_gpt_layer0_qkv_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    auto ln_weight = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_1.weight");
    auto ln_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_1.bias");
    auto c_attn_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_attn.weight");
    auto c_attn_weight = transpose_gpt_conv1d_weight(c_attn_weight_io, width, qkv_width);
    auto c_attn_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_attn.bias");
    GptLayer0QkvOutputs out;
    out.ln1.resize(static_cast<size_t>(tokens) * width);
    out.qkv.resize(static_cast<size_t>(tokens) * qkv_width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + static_cast<size_t>(t) * width, input.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = metal.layernorm_f32(row, ln_weight, ln_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), out.ln1.begin() + static_cast<size_t>(t) * width);
        auto qkv = metal.linear_f32(c_attn_weight, c_attn_bias, normed, qkv_width, width);
        std::copy(qkv.begin(), qkv.end(), out.qkv.begin() + static_cast<size_t>(t) * qkv_width);
    }
    return out;
}

bool run_gpt_layer0_qkv_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 4;
    constexpr uint32_t width = 1280;
    std::vector<float> input(static_cast<size_t>(tokens) * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.065f + std::cos(static_cast<float>(i % 307) * 0.013f) * 0.021f;
    }
    auto ref = run_gpt_layer0_qkv_cpu(bundle, input, tokens);
    auto got = run_gpt_layer0_qkv_metal(metal, bundle, input, tokens);
    const float ln_err = max_abs_error(got.ln1, ref.ln1);
    const float qkv_err = max_abs_error(got.qkv, ref.qkv);
    const float err = std::max(ln_err, qkv_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_layer0_qkv\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"hidden_dim\": " << width << ",\n";
    std::cout << "  \"qkv_dim\": 3840,\n";
    std::cout << "  \"ln1_max_abs_error\": " << ln_err << ",\n";
    std::cout << "  \"qkv_max_abs_error\": " << qkv_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

struct GptLayer0AttentionOutputs {
    std::vector<float> attention;
    std::vector<float> projected;
};

std::vector<float> run_gpt_layer0_attention_core_cpu(const std::vector<float>& qkv, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    std::vector<float> concat(static_cast<size_t>(tokens) * width);
    for (uint32_t h = 0; h < heads; ++h) {
        std::vector<float> qh(static_cast<size_t>(tokens) * head_dim);
        std::vector<float> kh(qh.size());
        std::vector<float> vh(qh.size());
        for (uint32_t t = 0; t < tokens; ++t) {
            const size_t base = static_cast<size_t>(t) * qkv_width + h * head_dim;
            std::copy(qkv.begin() + base, qkv.begin() + base + head_dim, qh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(qkv.begin() + base + width, qkv.begin() + base + width + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(qkv.begin() + base + width * 2, qkv.begin() + base + width * 2 + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
        }
        auto yh = cpu_attention_single_head_causal(qh, kh, vh, tokens, head_dim);
        for (uint32_t t = 0; t < tokens; ++t) {
            std::copy(yh.begin() + static_cast<size_t>(t) * head_dim,
                      yh.begin() + static_cast<size_t>(t + 1) * head_dim,
                      concat.begin() + static_cast<size_t>(t) * width + h * head_dim);
        }
    }
    return concat;
}

std::vector<float> run_gpt_layer0_attention_core_metal(mit2::MetalContext& metal, const std::vector<float>& qkv, uint32_t tokens) {
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    // Single fused dispatch over all heads (was 20 command buffers + CPU reshuffles).
    return metal.gpt_causal_attention_f32(qkv, tokens, heads, head_dim);
}

GptLayer0AttentionOutputs run_gpt_layer0_attention_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    auto qkv = run_gpt_layer0_qkv_cpu(bundle, input, tokens).qkv;
    auto c_proj_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_proj.weight");
    auto c_proj_weight = transpose_gpt_conv1d_weight(c_proj_weight_io, width, width);
    auto c_proj_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_proj.bias");
    GptLayer0AttentionOutputs out;
    out.attention = run_gpt_layer0_attention_core_cpu(qkv, tokens);
    out.projected.resize(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(out.attention.begin() + static_cast<size_t>(t) * width, out.attention.begin() + static_cast<size_t>(t + 1) * width);
        auto y = cpu_linear(c_proj_weight, c_proj_bias, row, width, width);
        std::copy(y.begin(), y.end(), out.projected.begin() + static_cast<size_t>(t) * width);
    }
    return out;
}

GptLayer0AttentionOutputs run_gpt_layer0_attention_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    auto qkv = run_gpt_layer0_qkv_metal(metal, bundle, input, tokens).qkv;
    auto c_proj_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_proj.weight");
    auto c_proj_weight = transpose_gpt_conv1d_weight(c_proj_weight_io, width, width);
    auto c_proj_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.attn.c_proj.bias");
    GptLayer0AttentionOutputs out;
    out.attention = run_gpt_layer0_attention_core_metal(metal, qkv, tokens);
    out.projected.resize(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(out.attention.begin() + static_cast<size_t>(t) * width, out.attention.begin() + static_cast<size_t>(t + 1) * width);
        auto y = metal.linear_f32(c_proj_weight, c_proj_bias, row, width, width);
        std::copy(y.begin(), y.end(), out.projected.begin() + static_cast<size_t>(t) * width);
    }
    return out;
}

bool run_gpt_layer0_attention_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 4;
    constexpr uint32_t width = 1280;
    std::vector<float> input(static_cast<size_t>(tokens) * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.065f + std::cos(static_cast<float>(i % 307) * 0.013f) * 0.021f;
    }
    auto ref = run_gpt_layer0_attention_cpu(bundle, input, tokens);
    auto got = run_gpt_layer0_attention_metal(metal, bundle, input, tokens);
    const float attn_err = max_abs_error(got.attention, ref.attention);
    const float proj_err = max_abs_error(got.projected, ref.projected);
    const float err = std::max(attn_err, proj_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_layer0_attention\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"heads\": 20,\n";
    std::cout << "  \"head_dim\": 64,\n";
    std::cout << "  \"attention_max_abs_error\": " << attn_err << ",\n";
    std::cout << "  \"c_proj_max_abs_error\": " << proj_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_gpt_layer0_attention_core_cached_cpu(const std::vector<float>& qkv, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    std::vector<float> concat(static_cast<size_t>(tokens) * width);
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t step = 0; step < tokens; ++step) {
            std::vector<float> qh(head_dim);
            std::vector<float> kh(static_cast<size_t>(step + 1) * head_dim);
            std::vector<float> vh(kh.size());
            const size_t q_base = static_cast<size_t>(step) * qkv_width + h * head_dim;
            std::copy(qkv.begin() + q_base, qkv.begin() + q_base + head_dim, qh.begin());
            for (uint32_t t = 0; t <= step; ++t) {
                const size_t base = static_cast<size_t>(t) * qkv_width + h * head_dim;
                std::copy(qkv.begin() + base + width, qkv.begin() + base + width + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
                std::copy(qkv.begin() + base + width * 2, qkv.begin() + base + width * 2 + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
            }
            auto yh = cpu_attention_single_query(qh, kh, vh, step + 1, head_dim);
            std::copy(yh.begin(), yh.end(), concat.begin() + static_cast<size_t>(step) * width + h * head_dim);
        }
    }
    return concat;
}

std::vector<float> run_gpt_layer0_attention_core_cached_metal(mit2::MetalContext& metal, const std::vector<float>& qkv, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    std::vector<float> concat(static_cast<size_t>(tokens) * width);
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t step = 0; step < tokens; ++step) {
            std::vector<float> qh(head_dim);
            std::vector<float> kh(static_cast<size_t>(step + 1) * head_dim);
            std::vector<float> vh(kh.size());
            const size_t q_base = static_cast<size_t>(step) * qkv_width + h * head_dim;
            std::copy(qkv.begin() + q_base, qkv.begin() + q_base + head_dim, qh.begin());
            for (uint32_t t = 0; t <= step; ++t) {
                const size_t base = static_cast<size_t>(t) * qkv_width + h * head_dim;
                std::copy(qkv.begin() + base + width, qkv.begin() + base + width + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
                std::copy(qkv.begin() + base + width * 2, qkv.begin() + base + width * 2 + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
            }
            auto yh = metal.attention_single_query_f32(qh, kh, vh, step + 1, head_dim);
            std::copy(yh.begin(), yh.end(), concat.begin() + static_cast<size_t>(step) * width + h * head_dim);
        }
    }
    return concat;
}

bool run_gpt_layer0_kv_attention_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    constexpr uint32_t width = 1280;
    std::vector<float> input(static_cast<size_t>(tokens) * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.065f + std::cos(static_cast<float>(i % 307) * 0.013f) * 0.021f;
    }
    auto qkv_ref = run_gpt_layer0_qkv_cpu(bundle, input, tokens).qkv;
    auto qkv_got = run_gpt_layer0_qkv_metal(metal, bundle, input, tokens).qkv;
    auto full_ref = run_gpt_layer0_attention_core_cpu(qkv_ref, tokens);
    auto cached_ref = run_gpt_layer0_attention_core_cached_cpu(qkv_ref, tokens);
    auto cached_got = run_gpt_layer0_attention_core_cached_metal(metal, qkv_got, tokens);
    const float full_cached_err = max_abs_error(cached_ref, full_ref);
    const float metal_cached_err = max_abs_error(cached_got, cached_ref);
    const float err = std::max(full_cached_err, metal_cached_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_layer0_kv_attention\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"heads\": 20,\n";
    std::cout << "  \"head_dim\": 64,\n";
    std::cout << "  \"full_vs_cached_cpu_max_abs_error\": " << full_cached_err << ",\n";
    std::cout << "  \"cached_metal_max_abs_error\": " << metal_cached_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

struct GptLayer0BlockOutputs {
    std::vector<float> attention_residual;
    std::vector<float> ln2;
    std::vector<float> mlp_hidden;
    std::vector<float> mlp_projected;
    std::vector<float> block_output;
};

GptLayer0BlockOutputs run_gpt_layer0_block_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t mlp_width = 5120;
    GptLayer0BlockOutputs out;
    auto attn = run_gpt_layer0_attention_cpu(bundle, input, tokens);
    out.attention_residual.resize(static_cast<size_t>(tokens) * width);
    for (size_t i = 0; i < out.attention_residual.size(); ++i) {
        out.attention_residual[i] = input[i] + attn.projected[i];
    }

    auto ln_weight = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_2.weight");
    auto ln_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_2.bias");
    auto c_fc_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_fc.weight");
    auto c_fc_weight = transpose_gpt_conv1d_weight(c_fc_weight_io, width, mlp_width);
    auto c_fc_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_fc.bias");
    auto c_proj_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_proj.weight");
    auto c_proj_weight = transpose_gpt_conv1d_weight(c_proj_weight_io, mlp_width, width);
    auto c_proj_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_proj.bias");

    out.ln2.resize(static_cast<size_t>(tokens) * width);
    out.mlp_hidden.resize(static_cast<size_t>(tokens) * mlp_width);
    out.mlp_projected.resize(static_cast<size_t>(tokens) * width);
    out.block_output.resize(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(out.attention_residual.begin() + static_cast<size_t>(t) * width,
                               out.attention_residual.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = cpu_layernorm(row, ln_weight, ln_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), out.ln2.begin() + static_cast<size_t>(t) * width);
        auto hidden = cpu_linear(c_fc_weight, c_fc_bias, normed, mlp_width, width);
        hidden = cpu_gelu_new(hidden);
        std::copy(hidden.begin(), hidden.end(), out.mlp_hidden.begin() + static_cast<size_t>(t) * mlp_width);
        auto projected = cpu_linear(c_proj_weight, c_proj_bias, hidden, width, mlp_width);
        std::copy(projected.begin(), projected.end(), out.mlp_projected.begin() + static_cast<size_t>(t) * width);
        for (uint32_t c = 0; c < width; ++c) {
            const size_t idx = static_cast<size_t>(t) * width + c;
            out.block_output[idx] = out.attention_residual[idx] + projected[c];
        }
    }
    return out;
}

GptLayer0BlockOutputs run_gpt_layer0_block_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t mlp_width = 5120;
    GptLayer0BlockOutputs out;
    auto attn = run_gpt_layer0_attention_metal(metal, bundle, input, tokens);
    out.attention_residual = metal.add_f32(input, attn.projected);

    auto ln_weight = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_2.weight");
    auto ln_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.ln_2.bias");
    auto c_fc_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_fc.weight");
    auto c_fc_weight = transpose_gpt_conv1d_weight(c_fc_weight_io, width, mlp_width);
    auto c_fc_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_fc.bias");
    auto c_proj_weight_io = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_proj.weight");
    auto c_proj_weight = transpose_gpt_conv1d_weight(c_proj_weight_io, mlp_width, width);
    auto c_proj_bias = tensor_as_f32(bundle, "gpt.gpt.h.0.mlp.c_proj.bias");

    out.ln2.resize(static_cast<size_t>(tokens) * width);
    out.mlp_hidden.resize(static_cast<size_t>(tokens) * mlp_width);
    out.mlp_projected.resize(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(out.attention_residual.begin() + static_cast<size_t>(t) * width,
                               out.attention_residual.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = metal.layernorm_f32(row, ln_weight, ln_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), out.ln2.begin() + static_cast<size_t>(t) * width);
        auto hidden = metal.linear_f32(c_fc_weight, c_fc_bias, normed, mlp_width, width);
        hidden = metal.gelu_f32(hidden);
        std::copy(hidden.begin(), hidden.end(), out.mlp_hidden.begin() + static_cast<size_t>(t) * mlp_width);
        auto projected = metal.linear_f32(c_proj_weight, c_proj_bias, hidden, width, mlp_width);
        std::copy(projected.begin(), projected.end(), out.mlp_projected.begin() + static_cast<size_t>(t) * width);
    }
    out.block_output = metal.add_f32(out.attention_residual, out.mlp_projected);
    return out;
}

bool run_gpt_layer0_block_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 4;
    constexpr uint32_t width = 1280;
    std::vector<float> input(static_cast<size_t>(tokens) * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.065f + std::cos(static_cast<float>(i % 307) * 0.013f) * 0.021f;
    }
    auto ref = run_gpt_layer0_block_cpu(bundle, input, tokens);
    auto got = run_gpt_layer0_block_metal(metal, bundle, input, tokens);
    const float attn_res_err = max_abs_error(got.attention_residual, ref.attention_residual);
    const float ln2_err = max_abs_error(got.ln2, ref.ln2);
    const float mlp_hidden_err = max_abs_error(got.mlp_hidden, ref.mlp_hidden);
    const float mlp_proj_err = max_abs_error(got.mlp_projected, ref.mlp_projected);
    const float out_err = max_abs_error(got.block_output, ref.block_output);
    const float err = std::max({attn_res_err, ln2_err, mlp_hidden_err, mlp_proj_err, out_err});
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_layer0_block\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"hidden_dim\": " << width << ",\n";
    std::cout << "  \"mlp_dim\": 5120,\n";
    std::cout << "  \"attention_residual_max_abs_error\": " << attn_res_err << ",\n";
    std::cout << "  \"ln2_max_abs_error\": " << ln2_err << ",\n";
    std::cout << "  \"mlp_hidden_max_abs_error\": " << mlp_hidden_err << ",\n";
    std::cout << "  \"mlp_projected_max_abs_error\": " << mlp_proj_err << ",\n";
    std::cout << "  \"block_output_max_abs_error\": " << out_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_gpt_transformer_block_cpu_layer(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layer) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    const std::string base = "gpt.gpt.h." + std::to_string(layer);
    auto ln1_weight = tensor_as_f32(bundle, base + ".ln_1.weight");
    auto ln1_bias = tensor_as_f32(bundle, base + ".ln_1.bias");
    auto c_attn_weight_io = tensor_as_f32(bundle, base + ".attn.c_attn.weight");
    auto c_attn_weight = transpose_gpt_conv1d_weight(c_attn_weight_io, width, qkv_width);
    auto c_attn_bias = tensor_as_f32(bundle, base + ".attn.c_attn.bias");
    auto attn_proj_weight_io = tensor_as_f32(bundle, base + ".attn.c_proj.weight");
    auto attn_proj_weight = transpose_gpt_conv1d_weight(attn_proj_weight_io, width, width);
    auto attn_proj_bias = tensor_as_f32(bundle, base + ".attn.c_proj.bias");
    std::vector<float> qkv(static_cast<size_t>(tokens) * qkv_width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + static_cast<size_t>(t) * width, input.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = cpu_layernorm(row, ln1_weight, ln1_bias, 1e-5f);
        auto projected = cpu_linear(c_attn_weight, c_attn_bias, normed, qkv_width, width);
        std::copy(projected.begin(), projected.end(), qkv.begin() + static_cast<size_t>(t) * qkv_width);
    }
    auto attention = run_gpt_layer0_attention_core_cpu(qkv, tokens);
    std::vector<float> attention_residual(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(attention.begin() + static_cast<size_t>(t) * width, attention.begin() + static_cast<size_t>(t + 1) * width);
        auto projected = cpu_linear(attn_proj_weight, attn_proj_bias, row, width, width);
        for (uint32_t c = 0; c < width; ++c) {
            const size_t idx = static_cast<size_t>(t) * width + c;
            attention_residual[idx] = input[idx] + projected[c];
        }
    }

    auto ln2_weight = tensor_as_f32(bundle, base + ".ln_2.weight");
    auto ln2_bias = tensor_as_f32(bundle, base + ".ln_2.bias");
    auto c_fc_weight_io = tensor_as_f32(bundle, base + ".mlp.c_fc.weight");
    auto c_fc_weight = transpose_gpt_conv1d_weight(c_fc_weight_io, width, mlp_width);
    auto c_fc_bias = tensor_as_f32(bundle, base + ".mlp.c_fc.bias");
    auto mlp_proj_weight_io = tensor_as_f32(bundle, base + ".mlp.c_proj.weight");
    auto mlp_proj_weight = transpose_gpt_conv1d_weight(mlp_proj_weight_io, mlp_width, width);
    auto mlp_proj_bias = tensor_as_f32(bundle, base + ".mlp.c_proj.bias");
    std::vector<float> out(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(attention_residual.begin() + static_cast<size_t>(t) * width,
                               attention_residual.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = cpu_layernorm(row, ln2_weight, ln2_bias, 1e-5f);
        auto hidden = cpu_linear(c_fc_weight, c_fc_bias, normed, mlp_width, width);
        hidden = cpu_gelu_new(hidden);
        auto projected = cpu_linear(mlp_proj_weight, mlp_proj_bias, hidden, width, mlp_width);
        for (uint32_t c = 0; c < width; ++c) {
            const size_t idx = static_cast<size_t>(t) * width + c;
            out[idx] = attention_residual[idx] + projected[c];
        }
    }
    return out;
}

std::vector<float> run_gpt_transformer_block_metal_layer(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layer) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    const std::string base = "gpt.gpt.h." + std::to_string(layer);
    auto ln1_weight = tensor_as_f32(bundle, base + ".ln_1.weight");
    auto ln1_bias = tensor_as_f32(bundle, base + ".ln_1.bias");
    auto c_attn_weight_io = tensor_as_f32(bundle, base + ".attn.c_attn.weight");
    auto c_attn_weight = transpose_gpt_conv1d_weight(c_attn_weight_io, width, qkv_width);
    auto c_attn_bias = tensor_as_f32(bundle, base + ".attn.c_attn.bias");
    auto attn_proj_weight_io = tensor_as_f32(bundle, base + ".attn.c_proj.weight");
    auto attn_proj_weight = transpose_gpt_conv1d_weight(attn_proj_weight_io, width, width);
    auto attn_proj_bias = tensor_as_f32(bundle, base + ".attn.c_proj.bias");
    std::vector<float> qkv(static_cast<size_t>(tokens) * qkv_width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + static_cast<size_t>(t) * width, input.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = metal.layernorm_f32(row, ln1_weight, ln1_bias, 1e-5f);
        auto projected = metal.linear_f32(c_attn_weight, c_attn_bias, normed, qkv_width, width);
        std::copy(projected.begin(), projected.end(), qkv.begin() + static_cast<size_t>(t) * qkv_width);
    }
    auto attention = run_gpt_layer0_attention_core_metal(metal, qkv, tokens);
    std::vector<float> attention_projected(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(attention.begin() + static_cast<size_t>(t) * width, attention.begin() + static_cast<size_t>(t + 1) * width);
        auto projected = metal.linear_f32(attn_proj_weight, attn_proj_bias, row, width, width);
        std::copy(projected.begin(), projected.end(), attention_projected.begin() + static_cast<size_t>(t) * width);
    }
    auto attention_residual = metal.add_f32(input, attention_projected);

    auto ln2_weight = tensor_as_f32(bundle, base + ".ln_2.weight");
    auto ln2_bias = tensor_as_f32(bundle, base + ".ln_2.bias");
    auto c_fc_weight_io = tensor_as_f32(bundle, base + ".mlp.c_fc.weight");
    auto c_fc_weight = transpose_gpt_conv1d_weight(c_fc_weight_io, width, mlp_width);
    auto c_fc_bias = tensor_as_f32(bundle, base + ".mlp.c_fc.bias");
    auto mlp_proj_weight_io = tensor_as_f32(bundle, base + ".mlp.c_proj.weight");
    auto mlp_proj_weight = transpose_gpt_conv1d_weight(mlp_proj_weight_io, mlp_width, width);
    auto mlp_proj_bias = tensor_as_f32(bundle, base + ".mlp.c_proj.bias");
    std::vector<float> mlp_projected(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(attention_residual.begin() + static_cast<size_t>(t) * width,
                               attention_residual.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = metal.layernorm_f32(row, ln2_weight, ln2_bias, 1e-5f);
        auto hidden = metal.linear_f32(c_fc_weight, c_fc_bias, normed, mlp_width, width);
        hidden = metal.gelu_f32(hidden);
        auto projected = metal.linear_f32(mlp_proj_weight, mlp_proj_bias, hidden, width, mlp_width);
        std::copy(projected.begin(), projected.end(), mlp_projected.begin() + static_cast<size_t>(t) * width);
    }
    return metal.add_f32(attention_residual, mlp_projected);
}

std::vector<float> run_gpt_transformer_block_metal_layer(mit2::MetalContext& metal, const GptTransformerLayerWeights& weights, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    auto normed1 = metal.layernorm_rows_f32_resident(
        resident_weight_key(weights, "ln_1"),
        weights.ln1_weight,
        resident_bias_key(weights, "ln_1"),
        weights.ln1_bias,
        input,
        tokens,
        width,
        1e-5f);
    auto qkv = metal.linear_rows_f32_resident(
        resident_weight_key(weights, "attn.c_attn"),
        weights.c_attn_weight,
        resident_bias_key(weights, "attn.c_attn"),
        weights.c_attn_bias,
        normed1,
        tokens,
        qkv_width,
        width);
    auto attention = run_gpt_layer0_attention_core_metal(metal, qkv, tokens);
    auto attention_projected = metal.linear_rows_f32_resident(
        resident_weight_key(weights, "attn.c_proj"),
        weights.attn_proj_weight,
        resident_bias_key(weights, "attn.c_proj"),
        weights.attn_proj_bias,
        attention,
        tokens,
        width,
        width);
    auto attention_residual = metal.add_f32(input, attention_projected);

    auto normed2 = metal.layernorm_rows_f32_resident(
        resident_weight_key(weights, "ln_2"),
        weights.ln2_weight,
        resident_bias_key(weights, "ln_2"),
        weights.ln2_bias,
        attention_residual,
        tokens,
        width,
        1e-5f);
    auto hidden = metal.linear_rows_f32_resident(
        resident_weight_key(weights, "mlp.c_fc"),
        weights.c_fc_weight,
        resident_bias_key(weights, "mlp.c_fc"),
        weights.c_fc_bias,
        normed2,
        tokens,
        mlp_width,
        width);
    hidden = metal.gelu_f32(hidden);
    auto mlp_projected = metal.linear_rows_f32_resident(
        resident_weight_key(weights, "mlp.c_proj"),
        weights.mlp_proj_weight,
        resident_bias_key(weights, "mlp.c_proj"),
        weights.mlp_proj_bias,
        hidden,
        tokens,
        width,
        mlp_width);
    return metal.add_f32(attention_residual, mlp_projected);
}

struct GptTransformerStackOutputs {
    std::vector<float> hidden;
    std::vector<float> ln_f;
};

GptTransformerStackOutputs run_gpt_transformer_stack_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layers) {
    constexpr uint32_t width = 1280;
    GptTransformerStackOutputs out;
    out.hidden = input;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        out.hidden = run_gpt_transformer_block_cpu_layer(bundle, out.hidden, tokens, layer);
    }
    auto ln_weight = tensor_as_f32(bundle, "gpt.gpt.ln_f.weight");
    auto ln_bias = tensor_as_f32(bundle, "gpt.gpt.ln_f.bias");
    out.ln_f.resize(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(out.hidden.begin() + static_cast<size_t>(t) * width, out.hidden.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = cpu_layernorm(row, ln_weight, ln_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), out.ln_f.begin() + static_cast<size_t>(t) * width);
    }
    return out;
}

GptTransformerStackOutputs run_gpt_transformer_stack_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layers) {
    constexpr uint32_t width = 1280;
    GptTransformerStackOutputs out;
    out.hidden = input;
    // Weights are immutable per bundle: cache the CPU-side structs across calls.
    static const mit2::Bundle* stack_cached_for = nullptr;
    static uint32_t stack_cached_layers = 0;
    static std::vector<GptTransformerLayerWeights> weights_cache;
    if (stack_cached_for != &bundle || stack_cached_layers < layers) {
        weights_cache.clear();
        weights_cache.reserve(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            weights_cache.push_back(load_gpt_transformer_layer_weights(bundle, layer));
        }
        stack_cached_for = &bundle;
        stack_cached_layers = layers;
    }
    const auto& weights = weights_cache;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        out.hidden = run_gpt_transformer_block_metal_layer(metal, weights[layer], out.hidden, tokens);
    }
    auto ln_weight = tensor_as_f32(bundle, "gpt.gpt.ln_f.weight");
    auto ln_bias = tensor_as_f32(bundle, "gpt.gpt.ln_f.bias");
    out.ln_f = metal.layernorm_rows_f32_resident(
        "gpt.gpt.ln_f.weight.resident",
        ln_weight,
        "gpt.gpt.ln_f.bias.resident",
        ln_bias,
        out.hidden,
        tokens,
        width,
        1e-5f);
    return out;
}

// Whole transformer stack in a single command buffer (pass mode): one sync
// instead of ~240. Returns hidden after ln_f. Uses the same cached layer
// weights as the non-pass stack.
std::vector<float> run_gpt_transformer_stack_lnf_metal_pass(mit2::MetalContext& metal,
                                                            const mit2::Bundle& bundle,
                                                            const std::vector<float>& input,
                                                            uint32_t tokens,
                                                            uint32_t layers) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    static const mit2::Bundle* cached_for = nullptr;
    static uint32_t cached_layers = 0;
    static std::vector<GptTransformerLayerWeights> weights_cache;
    if (cached_for != &bundle || cached_layers < layers) {
        weights_cache.clear();
        weights_cache.reserve(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            weights_cache.push_back(load_gpt_transformer_layer_weights(bundle, layer));
        }
        cached_for = &bundle;
        cached_layers = layers;
    }
    auto ln_f_weight = tensor_as_f32(bundle, "gpt.gpt.ln_f.weight");
    auto ln_f_bias = tensor_as_f32(bundle, "gpt.gpt.ln_f.bias");

    // Workspace: persistent hidden ping/pong + per-layer scratch (reset per layer).
    // Scratch per layer: ln(w) + qkv(3w) + attn(w) + proj(w) + residual(w) + ln2(w)
    //                  + fc(4w) + gelu(4w) + mlp_proj(w) + next hidden(w) = 18w
    const size_t per_token_floats = 18u * width;
    const size_t ws = (static_cast<size_t>(tokens) * (2u * width + per_token_floats) +
                       static_cast<size_t>(tokens) * width /*ln_f out*/ +
                       16u * 64u + 4096u) * sizeof(float);
    metal.beginPass(ws);
    // Persistent: uploaded input + hidden ping/pong + ln_f output.
    auto hidden = metal.passUploadAlloc(input);
    auto hidden_a = metal.passAlloc(tokens * width);
    auto hidden_b = metal.passAlloc(tokens * width);
    metal.passSetScratchBase();

    for (uint32_t layer = 0; layer < layers; ++layer) {
        const auto& lw = weights_cache[layer];
        const mit2::PassSlot out_state = (layer % 2 == 0) ? hidden_a : hidden_b;
        auto normed1 = metal.layernorm_rows_f32_pass(
            resident_weight_key(lw, "ln_1"), lw.ln1_weight,
            resident_bias_key(lw, "ln_1"), lw.ln1_bias,
            hidden, tokens, width, 1e-5f);
        auto qkv = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "attn.c_attn"), lw.c_attn_weight,
            resident_bias_key(lw, "attn.c_attn"), lw.c_attn_bias,
            normed1, tokens, qkv_width, width);
        auto attention = metal.gpt_causal_attention_f32_pass(qkv, tokens, heads, head_dim);
        auto attn_proj = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "attn.c_proj"), lw.attn_proj_weight,
            resident_bias_key(lw, "attn.c_proj"), lw.attn_proj_bias,
            attention, tokens, width, width);
        auto attn_residual = metal.add_f32_pass(hidden, attn_proj);
        auto normed2 = metal.layernorm_rows_f32_pass(
            resident_weight_key(lw, "ln_2"), lw.ln2_weight,
            resident_bias_key(lw, "ln_2"), lw.ln2_bias,
            attn_residual, tokens, width, 1e-5f);
        auto fc = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "mlp.c_fc"), lw.c_fc_weight,
            resident_bias_key(lw, "mlp.c_fc"), lw.c_fc_bias,
            normed2, tokens, mlp_width, width);
        auto gelu_out = metal.gelu_f32_pass(fc, tokens * mlp_width);
        auto mlp_proj = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "mlp.c_proj"), lw.mlp_proj_weight,
            resident_bias_key(lw, "mlp.c_proj"), lw.mlp_proj_bias,
            gelu_out, tokens, width, mlp_width);
        metal.add_f32_pass_into(attn_residual, mlp_proj, out_state);
        hidden = out_state;
        metal.passResetScratch();
    }

    // ln_f output lives in scratch; nothing overwrites it before endPass, and
    // passRead is valid on any workspace slot after the pass completes.
    auto ln_f_out = metal.layernorm_rows_f32_pass(
        "gpt.gpt.ln_f.weight.resident", ln_f_weight,
        "gpt.gpt.ln_f.bias.resident", ln_f_bias,
        hidden, tokens, width, 1e-5f);
    metal.endPass();
    return metal.passRead(ln_f_out);
}

// Prefill in a single command buffer: runs the 24-layer stack over the prefix
// and stores each layer's K/V directly into the resident GPU KV cache
// (no per-layer readbacks, no 170MB CPU->GPU cache upload afterwards).
void run_gpt_kv_prefill_metal_pass(mit2::MetalContext& metal,
                                   const GptDecodeWeights& weights,
                                   const std::vector<float>& prefix,
                                   uint32_t prefix_tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    const uint32_t layers = static_cast<uint32_t>(weights.layers.size());
    const size_t per_token_floats = 18u * width;
    const size_t ws = (static_cast<size_t>(prefix_tokens) * (2u * width + per_token_floats) +
                       16u * 64u + 4096u) * sizeof(float);
    metal.beginPass(ws);
    auto hidden = metal.passUploadAlloc(prefix);
    auto hidden_a = metal.passAlloc(prefix_tokens * width);
    auto hidden_b = metal.passAlloc(prefix_tokens * width);
    metal.passSetScratchBase();

    for (uint32_t layer = 0; layer < layers; ++layer) {
        const auto& lw = weights.layers[layer];
        const mit2::PassSlot out_state = (layer % 2 == 0) ? hidden_a : hidden_b;
        auto normed1 = metal.layernorm_rows_f32_pass(
            resident_weight_key(lw, "ln_1"), lw.ln1_weight,
            resident_bias_key(lw, "ln_1"), lw.ln1_bias,
            hidden, prefix_tokens, width, 1e-5f);
        auto qkv = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "attn.c_attn"), lw.c_attn_weight,
            resident_bias_key(lw, "attn.c_attn"), lw.c_attn_bias,
            normed1, prefix_tokens, qkv_width, width);
        metal.gptKvStoreFromQkv_pass(layer, qkv, prefix_tokens);
        auto attention = metal.gpt_causal_attention_f32_pass(qkv, prefix_tokens, heads, head_dim);
        auto attn_proj = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "attn.c_proj"), lw.attn_proj_weight,
            resident_bias_key(lw, "attn.c_proj"), lw.attn_proj_bias,
            attention, prefix_tokens, width, width);
        auto attn_residual = metal.add_f32_pass(hidden, attn_proj);
        auto normed2 = metal.layernorm_rows_f32_pass(
            resident_weight_key(lw, "ln_2"), lw.ln2_weight,
            resident_bias_key(lw, "ln_2"), lw.ln2_bias,
            attn_residual, prefix_tokens, width, 1e-5f);
        auto fc = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "mlp.c_fc"), lw.c_fc_weight,
            resident_bias_key(lw, "mlp.c_fc"), lw.c_fc_bias,
            normed2, prefix_tokens, mlp_width, width);
        auto gelu_out = metal.gelu_f32_pass(fc, prefix_tokens * mlp_width);
        auto mlp_proj = metal.linear_rows_f32_pass(
            resident_weight_key(lw, "mlp.c_proj"), lw.mlp_proj_weight,
            resident_bias_key(lw, "mlp.c_proj"), lw.mlp_proj_bias,
            gelu_out, prefix_tokens, width, mlp_width);
        metal.add_f32_pass_into(attn_residual, mlp_proj, out_state);
        hidden = out_state;
        metal.passResetScratch();
    }
    metal.endPass();
}

bool run_gpt_transformer_stack_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 2;
    constexpr uint32_t width = 1280;
    constexpr uint32_t layers = 24;
    std::vector<float> input(static_cast<size_t>(tokens) * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.065f + std::cos(static_cast<float>(i % 307) * 0.013f) * 0.021f;
    }
    auto ref = run_gpt_transformer_stack_cpu(bundle, input, tokens, layers);
    auto got = run_gpt_transformer_stack_metal(metal, bundle, input, tokens, layers);
    const float hidden_err = max_abs_error(got.hidden, ref.hidden);
    const float lnf_err = max_abs_error(got.ln_f, ref.ln_f);
    const float err = std::max(hidden_err, lnf_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_transformer_stack\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"layers\": " << layers << ",\n";
    std::cout << "  \"hidden_dim\": " << width << ",\n";
    std::cout << "  \"hidden_max_abs_error\": " << hidden_err << ",\n";
    std::cout << "  \"ln_f_max_abs_error\": " << lnf_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-3f;
}

std::vector<float> run_gpt_layer_qkv_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layer) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    const std::string base = "gpt.gpt.h." + std::to_string(layer);
    auto ln_weight = tensor_as_f32(bundle, base + ".ln_1.weight");
    auto ln_bias = tensor_as_f32(bundle, base + ".ln_1.bias");
    auto c_attn_weight_io = tensor_as_f32(bundle, base + ".attn.c_attn.weight");
    auto c_attn_weight = transpose_gpt_conv1d_weight(c_attn_weight_io, width, qkv_width);
    auto c_attn_bias = tensor_as_f32(bundle, base + ".attn.c_attn.bias");
    std::vector<float> qkv(static_cast<size_t>(tokens) * qkv_width);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + static_cast<size_t>(t) * width, input.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = cpu_layernorm(row, ln_weight, ln_bias, 1e-5f);
        auto projected = cpu_linear(c_attn_weight, c_attn_bias, normed, qkv_width, width);
        std::copy(projected.begin(), projected.end(), qkv.begin() + static_cast<size_t>(t) * qkv_width);
    }
    return qkv;
}

std::vector<float> run_gpt_layer_qkv_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layer) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    const std::string base = "gpt.gpt.h." + std::to_string(layer);
    auto ln_weight = tensor_as_f32(bundle, base + ".ln_1.weight");
    auto ln_bias = tensor_as_f32(bundle, base + ".ln_1.bias");
    auto c_attn_weight_io = tensor_as_f32(bundle, base + ".attn.c_attn.weight");
    auto c_attn_weight = transpose_gpt_conv1d_weight(c_attn_weight_io, width, qkv_width);
    auto c_attn_bias = tensor_as_f32(bundle, base + ".attn.c_attn.bias");
    auto normed = metal.layernorm_rows_f32(input, ln_weight, ln_bias, tokens, width, 1e-5f);
    return metal.linear_rows_f32(c_attn_weight, c_attn_bias, normed, tokens, qkv_width, width);
}

std::vector<float> run_gpt_layer_qkv_metal(mit2::MetalContext& metal, const GptTransformerLayerWeights& weights, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    auto normed = metal.layernorm_rows_f32_resident(
        resident_weight_key(weights, "ln_1"),
        weights.ln1_weight,
        resident_bias_key(weights, "ln_1"),
        weights.ln1_bias,
        input,
        tokens,
        width,
        1e-5f);
    return metal.linear_rows_f32_resident(
        resident_weight_key(weights, "attn.c_attn"),
        weights.c_attn_weight,
        resident_bias_key(weights, "attn.c_attn"),
        weights.c_attn_bias,
        normed,
        tokens,
        qkv_width,
        width);
}

struct GptLayerKvCache {
    std::vector<float> k;
    std::vector<float> v;
    uint32_t tokens = 0;
};

GptLayerKvCache extract_gpt_kv_cache(const std::vector<float>& qkv, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    GptLayerKvCache cache;
    cache.tokens = tokens;
    cache.k.resize(static_cast<size_t>(tokens) * width);
    cache.v.resize(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        const size_t src = static_cast<size_t>(t) * qkv_width;
        const size_t dst = static_cast<size_t>(t) * width;
        std::copy(qkv.begin() + src + width, qkv.begin() + src + width * 2, cache.k.begin() + dst);
        std::copy(qkv.begin() + src + width * 2, qkv.begin() + src + width * 3, cache.v.begin() + dst);
    }
    return cache;
}

void append_gpt_current_kv(GptLayerKvCache& cache, const std::vector<float>& current_qkv) {
    constexpr uint32_t width = 1280;
    cache.k.resize(static_cast<size_t>(cache.tokens + 1) * width);
    cache.v.resize(static_cast<size_t>(cache.tokens + 1) * width);
    const size_t dst = static_cast<size_t>(cache.tokens) * width;
    std::copy(current_qkv.begin() + width, current_qkv.begin() + width * 2, cache.k.begin() + dst);
    std::copy(current_qkv.begin() + width * 2, current_qkv.begin() + width * 3, cache.v.begin() + dst);
    ++cache.tokens;
}

std::vector<float> run_gpt_cached_attention_current_cpu(const GptLayerKvCache& cache, const std::vector<float>& current_qkv) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    std::vector<float> concat(width);
    for (uint32_t h = 0; h < heads; ++h) {
        std::vector<float> qh(head_dim);
        std::vector<float> kh(static_cast<size_t>(cache.tokens + 1) * head_dim);
        std::vector<float> vh(kh.size());
        std::copy(current_qkv.begin() + h * head_dim, current_qkv.begin() + (h + 1) * head_dim, qh.begin());
        for (uint32_t t = 0; t < cache.tokens; ++t) {
            const size_t src = static_cast<size_t>(t) * width + h * head_dim;
            const size_t dst = static_cast<size_t>(t) * head_dim;
            std::copy(cache.k.begin() + src, cache.k.begin() + src + head_dim, kh.begin() + dst);
            std::copy(cache.v.begin() + src, cache.v.begin() + src + head_dim, vh.begin() + dst);
        }
        const size_t last = static_cast<size_t>(cache.tokens) * head_dim;
        std::copy(current_qkv.begin() + width + h * head_dim, current_qkv.begin() + width + (h + 1) * head_dim, kh.begin() + last);
        std::copy(current_qkv.begin() + width * 2 + h * head_dim, current_qkv.begin() + width * 2 + (h + 1) * head_dim, vh.begin() + last);
        auto yh = cpu_attention_single_query(qh, kh, vh, cache.tokens + 1, head_dim);
        std::copy(yh.begin(), yh.end(), concat.begin() + h * head_dim);
    }
    return concat;
}

std::vector<float> run_gpt_cached_attention_current_metal(mit2::MetalContext& metal, const GptLayerKvCache& cache, const std::vector<float>& current_qkv) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    (void)width;
    return metal.gpt_cached_attention_f32(cache.k, cache.v, current_qkv, cache.tokens, heads, head_dim);
}

std::vector<float> run_gpt_cached_block_current_cpu(const mit2::Bundle& bundle, const GptLayerKvCache& cache, const std::vector<float>& current, const std::vector<float>& current_qkv, uint32_t layer) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t mlp_width = 5120;
    const std::string base = "gpt.gpt.h." + std::to_string(layer);
    auto attn = run_gpt_cached_attention_current_cpu(cache, current_qkv);
    auto attn_proj_weight_io = tensor_as_f32(bundle, base + ".attn.c_proj.weight");
    auto attn_proj_weight = transpose_gpt_conv1d_weight(attn_proj_weight_io, width, width);
    auto attn_proj_bias = tensor_as_f32(bundle, base + ".attn.c_proj.bias");
    auto attn_projected = cpu_linear(attn_proj_weight, attn_proj_bias, attn, width, width);
    std::vector<float> attention_residual(width);
    for (uint32_t i = 0; i < width; ++i) {
        attention_residual[i] = current[i] + attn_projected[i];
    }
    auto ln2_weight = tensor_as_f32(bundle, base + ".ln_2.weight");
    auto ln2_bias = tensor_as_f32(bundle, base + ".ln_2.bias");
    auto c_fc_weight_io = tensor_as_f32(bundle, base + ".mlp.c_fc.weight");
    auto c_fc_weight = transpose_gpt_conv1d_weight(c_fc_weight_io, width, mlp_width);
    auto c_fc_bias = tensor_as_f32(bundle, base + ".mlp.c_fc.bias");
    auto mlp_proj_weight_io = tensor_as_f32(bundle, base + ".mlp.c_proj.weight");
    auto mlp_proj_weight = transpose_gpt_conv1d_weight(mlp_proj_weight_io, mlp_width, width);
    auto mlp_proj_bias = tensor_as_f32(bundle, base + ".mlp.c_proj.bias");
    auto normed = cpu_layernorm(attention_residual, ln2_weight, ln2_bias, 1e-5f);
    auto hidden = cpu_linear(c_fc_weight, c_fc_bias, normed, mlp_width, width);
    hidden = cpu_gelu_new(hidden);
    auto projected = cpu_linear(mlp_proj_weight, mlp_proj_bias, hidden, width, mlp_width);
    std::vector<float> out(width);
    for (uint32_t i = 0; i < width; ++i) {
        out[i] = attention_residual[i] + projected[i];
    }
    return out;
}

[[maybe_unused]] std::vector<float> run_gpt_cached_block_current_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const GptLayerKvCache& cache, const std::vector<float>& current, const std::vector<float>& current_qkv, uint32_t layer) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t mlp_width = 5120;
    const std::string base = "gpt.gpt.h." + std::to_string(layer);
    auto attn = run_gpt_cached_attention_current_metal(metal, cache, current_qkv);
    auto attn_proj_weight_io = tensor_as_f32(bundle, base + ".attn.c_proj.weight");
    auto attn_proj_weight = transpose_gpt_conv1d_weight(attn_proj_weight_io, width, width);
    auto attn_proj_bias = tensor_as_f32(bundle, base + ".attn.c_proj.bias");
    auto attn_projected = metal.linear_f32(attn_proj_weight, attn_proj_bias, attn, width, width);
    std::vector<float> attention_residual(width);
    for (uint32_t i = 0; i < width; ++i) {
        attention_residual[i] = current[i] + attn_projected[i];
    }
    auto ln2_weight = tensor_as_f32(bundle, base + ".ln_2.weight");
    auto ln2_bias = tensor_as_f32(bundle, base + ".ln_2.bias");
    auto c_fc_weight_io = tensor_as_f32(bundle, base + ".mlp.c_fc.weight");
    auto c_fc_weight = transpose_gpt_conv1d_weight(c_fc_weight_io, width, mlp_width);
    auto c_fc_bias = tensor_as_f32(bundle, base + ".mlp.c_fc.bias");
    auto mlp_proj_weight_io = tensor_as_f32(bundle, base + ".mlp.c_proj.weight");
    auto mlp_proj_weight = transpose_gpt_conv1d_weight(mlp_proj_weight_io, mlp_width, width);
    auto mlp_proj_bias = tensor_as_f32(bundle, base + ".mlp.c_proj.bias");
    auto normed = metal.layernorm_f32(attention_residual, ln2_weight, ln2_bias, 1e-5f);
    auto hidden = metal.linear_f32(c_fc_weight, c_fc_bias, normed, mlp_width, width);
    hidden = metal.gelu_f32(hidden);
    auto projected = metal.linear_f32(mlp_proj_weight, mlp_proj_bias, hidden, width, mlp_width);
    std::vector<float> out(width);
    for (uint32_t i = 0; i < width; ++i) {
        out[i] = attention_residual[i] + projected[i];
    }
    return out;
}

std::vector<float> run_gpt_cached_block_current_metal(mit2::MetalContext& metal, const GptTransformerLayerWeights& weights, const GptLayerKvCache& cache, const std::vector<float>& current, const std::vector<float>& current_qkv) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t mlp_width = 5120;
    auto attn = run_gpt_cached_attention_current_metal(metal, cache, current_qkv);
    auto attn_projected = metal.linear_f32_resident(
        resident_weight_key(weights, "attn.c_proj"),
        weights.attn_proj_weight,
        resident_bias_key(weights, "attn.c_proj"),
        weights.attn_proj_bias,
        attn,
        width,
        width);
    std::vector<float> attention_residual(width);
    for (uint32_t i = 0; i < width; ++i) {
        attention_residual[i] = current[i] + attn_projected[i];
    }
    auto normed = metal.layernorm_f32_resident(
        resident_weight_key(weights, "ln_2"),
        weights.ln2_weight,
        resident_bias_key(weights, "ln_2"),
        weights.ln2_bias,
        attention_residual,
        1e-5f);
    auto hidden = metal.linear_f32_resident(
        resident_weight_key(weights, "mlp.c_fc"),
        weights.c_fc_weight,
        resident_bias_key(weights, "mlp.c_fc"),
        weights.c_fc_bias,
        normed,
        mlp_width,
        width);
    hidden = metal.gelu_f32(hidden);
    auto projected = metal.linear_f32_resident(
        resident_weight_key(weights, "mlp.c_proj"),
        weights.mlp_proj_weight,
        resident_bias_key(weights, "mlp.c_proj"),
        weights.mlp_proj_bias,
        hidden,
        width,
        mlp_width);
    std::vector<float> out(width);
    for (uint32_t i = 0; i < width; ++i) {
        out[i] = attention_residual[i] + projected[i];
    }
    return out;
}

struct GptLogitsOutputs {
    std::vector<float> final_norm;
    std::vector<float> logits;
    uint32_t last_argmax = 0;
};

uint32_t argmax_row(const std::vector<float>& values, size_t offset, uint32_t count) {
    uint32_t best = 0;
    float best_value = values[offset];
    for (uint32_t i = 1; i < count; ++i) {
        const float value = values[offset + i];
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return best;
}

GptLogitsOutputs run_gpt_logits_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t vocab = 8194;
    auto stack = run_gpt_transformer_stack_cpu(bundle, input, tokens, 24);
    auto norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    auto norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    auto head_weight = tensor_as_f32(bundle, "gpt.mel_head.weight");
    auto head_bias = tensor_as_f32(bundle, "gpt.mel_head.bias");
    GptLogitsOutputs out;
    out.final_norm.resize(static_cast<size_t>(tokens) * width);
    out.logits.resize(static_cast<size_t>(tokens) * vocab);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(stack.ln_f.begin() + static_cast<size_t>(t) * width,
                               stack.ln_f.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = cpu_layernorm(row, norm_weight, norm_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), out.final_norm.begin() + static_cast<size_t>(t) * width);
        auto logits = cpu_linear(head_weight, head_bias, normed, vocab, width);
        std::copy(logits.begin(), logits.end(), out.logits.begin() + static_cast<size_t>(t) * vocab);
    }
    out.last_argmax = argmax_row(out.logits, static_cast<size_t>(tokens - 1) * vocab, vocab);
    return out;
}

GptLogitsOutputs run_gpt_logits_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t vocab = 8194;
    auto stack = run_gpt_transformer_stack_metal(metal, bundle, input, tokens, 24);
    auto norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    auto norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    auto head_weight = tensor_as_f32(bundle, "gpt.mel_head.weight");
    auto head_bias = tensor_as_f32(bundle, "gpt.mel_head.bias");
    GptLogitsOutputs out;
    out.final_norm = metal.layernorm_rows_f32_resident(
        "gpt.final_norm.weight.resident",
        norm_weight,
        "gpt.final_norm.bias.resident",
        norm_bias,
        stack.ln_f,
        tokens,
        width,
        1e-5f);
    out.logits = metal.linear_rows_f32_resident(
        "gpt.mel_head.weight.resident",
        head_weight,
        "gpt.mel_head.bias.resident",
        head_bias,
        out.final_norm,
        tokens,
        vocab,
        width);
    out.last_argmax = argmax_row(out.logits, static_cast<size_t>(tokens - 1) * vocab, vocab);
    return out;
}

bool run_gpt_logits_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 2;
    constexpr uint32_t width = 1280;
    std::vector<float> input(static_cast<size_t>(tokens) * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.065f + std::cos(static_cast<float>(i % 307) * 0.013f) * 0.021f;
    }
    auto ref = run_gpt_logits_cpu(bundle, input, tokens);
    auto got = run_gpt_logits_metal(metal, bundle, input, tokens);
    const float norm_err = max_abs_error(got.final_norm, ref.final_norm);
    const float logits_err = max_abs_error(got.logits, ref.logits);
    const bool argmax_match = got.last_argmax == ref.last_argmax;
    const float err = std::max(norm_err, logits_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_logits\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"vocab\": 8194,\n";
    std::cout << "  \"final_norm_max_abs_error\": " << norm_err << ",\n";
    std::cout << "  \"logits_max_abs_error\": " << logits_err << ",\n";
    std::cout << "  \"last_argmax_ref\": " << ref.last_argmax << ",\n";
    std::cout << "  \"last_argmax_got\": " << got.last_argmax << ",\n";
    std::cout << "  \"last_argmax_match\": " << (argmax_match ? "true" : "false") << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-3f && argmax_match;
}

struct GptCachedDecodeOutputs {
    std::vector<float> ln_f;
    std::vector<float> final_norm;
    std::vector<float> logits;
    uint32_t argmax = 0;
};

GptCachedDecodeOutputs run_gpt_cached_decode_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t prefix_tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t vocab = 8194;
    std::vector<float> prefix(input.begin(), input.begin() + static_cast<size_t>(prefix_tokens) * width);
    std::vector<float> current(input.begin() + static_cast<size_t>(prefix_tokens) * width,
                               input.begin() + static_cast<size_t>(prefix_tokens + 1) * width);
    for (uint32_t layer = 0; layer < 24; ++layer) {
        auto qkv_prefix = run_gpt_layer_qkv_cpu(bundle, prefix, prefix_tokens, layer);
        auto cache = extract_gpt_kv_cache(qkv_prefix, prefix_tokens);
        prefix = run_gpt_transformer_block_cpu_layer(bundle, prefix, prefix_tokens, layer);
        auto current_qkv = run_gpt_layer_qkv_cpu(bundle, current, 1, layer);
        current = run_gpt_cached_block_current_cpu(bundle, cache, current, current_qkv, layer);
    }
    auto ln_f_weight = tensor_as_f32(bundle, "gpt.gpt.ln_f.weight");
    auto ln_f_bias = tensor_as_f32(bundle, "gpt.gpt.ln_f.bias");
    auto final_norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    auto final_norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    auto head_weight = tensor_as_f32(bundle, "gpt.mel_head.weight");
    auto head_bias = tensor_as_f32(bundle, "gpt.mel_head.bias");
    GptCachedDecodeOutputs out;
    out.ln_f = cpu_layernorm(current, ln_f_weight, ln_f_bias, 1e-5f);
    out.final_norm = cpu_layernorm(out.ln_f, final_norm_weight, final_norm_bias, 1e-5f);
    out.logits = cpu_linear(head_weight, head_bias, out.final_norm, vocab, width);
    out.argmax = argmax_row(out.logits, 0, vocab);
    return out;
}

GptCachedDecodeOutputs run_gpt_cached_decode_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t prefix_tokens) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t vocab = 8194;
    const auto weights = load_gpt_decode_weights(bundle);
    std::vector<float> prefix(input.begin(), input.begin() + static_cast<size_t>(prefix_tokens) * width);
    std::vector<float> current(input.begin() + static_cast<size_t>(prefix_tokens) * width,
                               input.begin() + static_cast<size_t>(prefix_tokens + 1) * width);
    for (uint32_t layer = 0; layer < weights.layers.size(); ++layer) {
        const auto& layer_weights = weights.layers[layer];
        auto qkv_prefix = run_gpt_layer_qkv_metal(metal, layer_weights, prefix, prefix_tokens);
        auto cache = extract_gpt_kv_cache(qkv_prefix, prefix_tokens);
        prefix = run_gpt_transformer_block_metal_layer(metal, layer_weights, prefix, prefix_tokens);
        auto current_qkv = run_gpt_layer_qkv_metal(metal, layer_weights, current, 1);
        current = run_gpt_cached_block_current_metal(metal, layer_weights, cache, current, current_qkv);
    }
    GptCachedDecodeOutputs out;
    out.ln_f = metal.layernorm_f32_resident(
        "gpt.gpt.ln_f.weight.resident",
        weights.ln_f_weight,
        "gpt.gpt.ln_f.bias.resident",
        weights.ln_f_bias,
        current,
        1e-5f);
    out.final_norm = metal.layernorm_f32_resident(
        "gpt.final_norm.weight.resident",
        weights.final_norm_weight,
        "gpt.final_norm.bias.resident",
        weights.final_norm_bias,
        out.ln_f,
        1e-5f);
    out.logits = metal.linear_f32_resident(
        "gpt.mel_head.weight.resident",
        weights.head_weight,
        "gpt.mel_head.bias.resident",
        weights.head_bias,
        out.final_norm,
        vocab,
        width);
    out.argmax = argmax_row(out.logits, 0, vocab);
    return out;
}

bool run_gpt_kv_decode_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t prefix_tokens = 3;
    constexpr uint32_t total_tokens = prefix_tokens + 1;
    constexpr uint32_t width = 1280;
    constexpr uint32_t vocab = 8194;
    std::vector<float> input(static_cast<size_t>(total_tokens) * width);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.065f + std::cos(static_cast<float>(i % 307) * 0.013f) * 0.021f;
    }
    auto full_ref = run_gpt_logits_cpu(bundle, input, total_tokens);
    std::vector<float> full_last_logits(full_ref.logits.end() - vocab, full_ref.logits.end());
    std::vector<float> full_last_norm(full_ref.final_norm.end() - width, full_ref.final_norm.end());
    auto cached_ref = run_gpt_cached_decode_cpu(bundle, input, prefix_tokens);
    auto cached_got = run_gpt_cached_decode_metal(metal, bundle, input, prefix_tokens);
    const float cpu_logits_err = max_abs_error(cached_ref.logits, full_last_logits);
    const float cpu_norm_err = max_abs_error(cached_ref.final_norm, full_last_norm);
    const float metal_logits_err = max_abs_error(cached_got.logits, cached_ref.logits);
    const float metal_norm_err = max_abs_error(cached_got.final_norm, cached_ref.final_norm);
    const bool cpu_argmax_match = cached_ref.argmax == full_ref.last_argmax;
    const bool metal_argmax_match = cached_got.argmax == cached_ref.argmax;
    const float err = std::max({cpu_logits_err, cpu_norm_err, metal_logits_err, metal_norm_err});
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_decode\",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"decode_tokens\": 1,\n";
    std::cout << "  \"layers\": 24,\n";
    std::cout << "  \"cpu_cached_vs_full_norm_max_abs_error\": " << cpu_norm_err << ",\n";
    std::cout << "  \"cpu_cached_vs_full_logits_max_abs_error\": " << cpu_logits_err << ",\n";
    std::cout << "  \"metal_cached_norm_max_abs_error\": " << metal_norm_err << ",\n";
    std::cout << "  \"metal_cached_logits_max_abs_error\": " << metal_logits_err << ",\n";
    std::cout << "  \"full_argmax\": " << full_ref.last_argmax << ",\n";
    std::cout << "  \"cached_ref_argmax\": " << cached_ref.argmax << ",\n";
    std::cout << "  \"cached_got_argmax\": " << cached_got.argmax << ",\n";
    std::cout << "  \"cpu_argmax_match\": " << (cpu_argmax_match ? "true" : "false") << ",\n";
    std::cout << "  \"metal_argmax_match\": " << (metal_argmax_match ? "true" : "false") << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-3f && cpu_argmax_match && metal_argmax_match;
}

std::vector<float> build_gpt_greedy_sequence_cpu(const mit2::Bundle& bundle, const std::vector<float>& prefix, uint32_t prefix_tokens, const std::vector<uint32_t>& mel_tokens) {
    constexpr uint32_t width = 1280;
    auto mel_embedding = tensor_as_f32(bundle, "gpt.mel_embedding.weight");
    auto mel_pos_embedding = tensor_as_f32(bundle, "gpt.mel_pos_embedding.emb.weight");
    std::vector<float> out(static_cast<size_t>(prefix_tokens + mel_tokens.size()) * width);
    std::copy(prefix.begin(), prefix.end(), out.begin());
    for (uint32_t t = 0; t < mel_tokens.size(); ++t) {
        const uint32_t token = mel_tokens[t];
        for (uint32_t c = 0; c < width; ++c) {
            out[static_cast<size_t>(prefix_tokens + t) * width + c] =
                mel_embedding[static_cast<size_t>(token) * width + c] + mel_pos_embedding[static_cast<size_t>(t) * width + c];
        }
    }
    return out;
}

std::vector<float> build_gpt_greedy_sequence_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& prefix, uint32_t prefix_tokens, const std::vector<uint32_t>& mel_tokens) {
    constexpr uint32_t width = 1280;
    auto mel_embedding = tensor_as_f32(bundle, "gpt.mel_embedding.weight");
    auto mel_pos_embedding = tensor_as_f32(bundle, "gpt.mel_pos_embedding.emb.weight");
    auto token_emb = metal.embedding_f32_resident("gpt.mel_embedding.weight.resident", mel_embedding, mel_tokens, width);
    std::vector<uint32_t> pos_ids(mel_tokens.size());
    for (uint32_t i = 0; i < pos_ids.size(); ++i) {
        pos_ids[i] = i;
    }
    auto pos_emb = metal.embedding_f32_resident("gpt.mel_pos_embedding.emb.weight.resident", mel_pos_embedding, pos_ids, width);
    std::vector<float> out(static_cast<size_t>(prefix_tokens + mel_tokens.size()) * width);
    std::copy(prefix.begin(), prefix.end(), out.begin());
    for (size_t i = 0; i < token_emb.size(); ++i) {
        out[static_cast<size_t>(prefix_tokens) * width + i] = token_emb[i] + pos_emb[i];
    }
    return out;
}

std::vector<float> build_gpt_greedy_current_cpu(const mit2::Bundle& bundle, uint32_t token, uint32_t position) {
    constexpr uint32_t width = 1280;
    auto mel_embedding = tensor_as_f32(bundle, "gpt.mel_embedding.weight");
    auto mel_pos_embedding = tensor_as_f32(bundle, "gpt.mel_pos_embedding.emb.weight");
    std::vector<float> out(width);
    for (uint32_t c = 0; c < width; ++c) {
        out[c] = mel_embedding[static_cast<size_t>(token) * width + c] + mel_pos_embedding[static_cast<size_t>(position) * width + c];
    }
    return out;
}

[[maybe_unused]] std::vector<float> build_gpt_greedy_current_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, uint32_t token, uint32_t position) {
    constexpr uint32_t width = 1280;
    auto mel_embedding = tensor_as_f32(bundle, "gpt.mel_embedding.weight");
    auto mel_pos_embedding = tensor_as_f32(bundle, "gpt.mel_pos_embedding.emb.weight");
    auto token_emb = metal.embedding_f32_resident("gpt.mel_embedding.weight.resident", mel_embedding, std::vector<uint32_t>{token}, width);
    auto pos_emb = metal.embedding_f32_resident("gpt.mel_pos_embedding.emb.weight.resident", mel_pos_embedding, std::vector<uint32_t>{position}, width);
    std::vector<float> out(width);
    for (uint32_t c = 0; c < width; ++c) {
        out[c] = token_emb[c] + pos_emb[c];
    }
    return out;
}

std::vector<float> build_gpt_greedy_current_cached(const GptDecodeWeights& weights, uint32_t token, uint32_t position) {
    constexpr uint32_t width = 1280;
    std::vector<float> out(width);
    const size_t token_offset = static_cast<size_t>(token) * width;
    const size_t pos_offset = static_cast<size_t>(position) * width;
    if (token_offset + width > weights.mel_embedding.size() || pos_offset + width > weights.mel_pos_embedding.size()) {
        throw std::runtime_error("GPT cached current embedding index out of range");
    }
    for (uint32_t c = 0; c < width; ++c) {
        out[c] = weights.mel_embedding[token_offset + c] + weights.mel_pos_embedding[pos_offset + c];
    }
    return out;
}

struct GptGreedyOutputs {
    std::vector<uint32_t> predicted_tokens;
    std::vector<uint32_t> input_tokens;
    std::vector<float> last_logits;
    std::vector<float> last_processed_logits;
    float max_step_logits_error = 0.0f;
    int32_t first_stop_step = -1;
};

struct GptSamplingConfig {
    bool do_sample = false;
    uint64_t seed = 0;
    float temperature = 1.0f;
    uint32_t top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
};

struct SplitMix64 {
    uint64_t state;

    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next_u64() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    double next_unit() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
};

std::vector<float> apply_gpt_sampling_processors(const std::vector<float>& logits,
                                                 const std::vector<uint32_t>& history,
                                                 const GptSamplingConfig& config) {
    std::vector<float> scores = logits;
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    if (config.repetition_penalty != 1.0f) {
        if (config.repetition_penalty <= 0.0f) {
            throw std::runtime_error("GPT repetition penalty must be positive");
        }
        std::vector<uint8_t> seen(scores.size(), 0);
        for (uint32_t token : history) {
            if (token >= scores.size() || seen[token]) {
                continue;
            }
            seen[token] = 1;
            if (scores[token] < 0.0f) {
                scores[token] *= config.repetition_penalty;
            } else {
                scores[token] /= config.repetition_penalty;
            }
        }
    }
    if (config.temperature <= 0.0f) {
        throw std::runtime_error("GPT sampling temperature must be positive");
    }
    if (config.temperature != 1.0f) {
        for (float& score : scores) {
            score /= config.temperature;
        }
    }
    if (config.top_k > 0 && config.top_k < scores.size()) {
        std::vector<float> sorted = scores;
        const size_t kth = static_cast<size_t>(config.top_k - 1);
        std::nth_element(sorted.begin(), sorted.begin() + kth, sorted.end(), std::greater<float>());
        const float pivot = sorted[kth];
        for (float& score : scores) {
            if (score < pivot) {
                score = neg_inf;
            }
        }
    }
    if (config.top_p < 1.0f) {
        if (config.top_p <= 0.0f) {
            throw std::runtime_error("GPT top_p must be in (0, 1]");
        }
        std::vector<size_t> order(scores.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (scores[a] == scores[b]) {
                return a < b;
            }
            return scores[a] < scores[b];
        });
        float max_score = neg_inf;
        for (float score : scores) {
            if (std::isfinite(score)) {
                max_score = std::max(max_score, score);
            }
        }
        if (!std::isfinite(max_score)) {
            throw std::runtime_error("GPT top_p filtering removed all logits");
        }
        double total = 0.0;
        std::vector<double> probs(scores.size(), 0.0);
        for (size_t i = 0; i < scores.size(); ++i) {
            if (std::isfinite(scores[i])) {
                probs[i] = std::exp(static_cast<double>(scores[i] - max_score));
                total += probs[i];
            }
        }
        if (total <= 0.0) {
            throw std::runtime_error("GPT sampling probability mass is zero");
        }
        double cumulative = 0.0;
        std::vector<uint8_t> remove(scores.size(), 0);
        const double low_tail_threshold = 1.0 - static_cast<double>(config.top_p);
        size_t finite_seen = 0;
        for (size_t idx : order) {
            if (!std::isfinite(scores[idx])) {
                remove[idx] = 1;
                continue;
            }
            ++finite_seen;
            cumulative += probs[idx] / total;
            if (cumulative <= low_tail_threshold) {
                remove[idx] = 1;
            }
        }
        if (finite_seen > 0) {
            remove[order.back()] = 0;
        }
        for (size_t i = 0; i < scores.size(); ++i) {
            if (remove[i]) {
                scores[i] = neg_inf;
            }
        }
    }
    return scores;
}

uint32_t sample_gpt_token_from_logits(const std::vector<float>& processed_logits, SplitMix64& rng) {
    float max_score = -std::numeric_limits<float>::infinity();
    for (float score : processed_logits) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("GPT sampling received no finite logits");
    }
    double total = 0.0;
    std::vector<double> probs(processed_logits.size(), 0.0);
    for (size_t i = 0; i < processed_logits.size(); ++i) {
        if (std::isfinite(processed_logits[i])) {
            probs[i] = std::exp(static_cast<double>(processed_logits[i] - max_score));
            total += probs[i];
        }
    }
    if (total <= 0.0) {
        throw std::runtime_error("GPT sampling probability mass is zero");
    }
    const double target = rng.next_unit() * total;
    double cumulative = 0.0;
    uint32_t last_finite = 0;
    for (uint32_t i = 0; i < probs.size(); ++i) {
        if (probs[i] <= 0.0) {
            continue;
        }
        last_finite = i;
        cumulative += probs[i];
        if (target < cumulative) {
            return i;
        }
    }
    return last_finite;
}

bool run_gpt_sampling_processors_test() {
    std::vector<float> logits{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<uint32_t> history{2, 7, 3, 7};
    GptSamplingConfig config;
    config.do_sample = true;
    config.seed = 20240605;
    config.temperature = 0.5f;
    config.top_k = 4;
    config.top_p = 0.95f;
    config.repetition_penalty = 2.0f;
    auto processed = apply_gpt_sampling_processors(logits, history, config);
    SplitMix64 rng(config.seed);
    const uint32_t sampled = sample_gpt_token_from_logits(processed, rng);
    const bool filtered_ok = !std::isfinite(processed[0]) &&
                             !std::isfinite(processed[5]) &&
                             !std::isfinite(processed[6]) &&
                             std::isfinite(processed[8]) &&
                             std::isfinite(processed[9]);
    const bool values_ok = std::abs(processed[8] - 12.0f) <= 1e-6f &&
                           std::abs(processed[9] - 14.0f) <= 1e-6f;
    const bool sampled_ok = sampled == 9;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_sampling_processors\",\n";
    std::cout << "  \"temperature\": " << config.temperature << ",\n";
    std::cout << "  \"top_k\": " << config.top_k << ",\n";
    std::cout << "  \"top_p\": " << config.top_p << ",\n";
    std::cout << "  \"repetition_penalty\": " << config.repetition_penalty << ",\n";
    std::cout << "  \"finite_token_8\": " << (std::isfinite(processed[8]) ? "true" : "false") << ",\n";
    std::cout << "  \"finite_token_9\": " << (std::isfinite(processed[9]) ? "true" : "false") << ",\n";
    std::cout << "  \"sampled\": " << sampled << ",\n";
    std::cout << "  \"passed\": " << ((filtered_ok && values_ok && sampled_ok) ? "true" : "false") << "\n";
    std::cout << "}\n";
    return filtered_ok && values_ok && sampled_ok;
}

struct GptKvPrefillState {
    std::vector<GptLayerKvCache> caches;
};

GptKvPrefillState run_gpt_kv_prefill_cpu(const mit2::Bundle& bundle, const std::vector<float>& prefix, uint32_t prefix_tokens) {
    GptKvPrefillState state;
    state.caches.reserve(24);
    std::vector<float> hidden = prefix;
    for (uint32_t layer = 0; layer < 24; ++layer) {
        auto qkv = run_gpt_layer_qkv_cpu(bundle, hidden, prefix_tokens, layer);
        state.caches.push_back(extract_gpt_kv_cache(qkv, prefix_tokens));
        hidden = run_gpt_transformer_block_cpu_layer(bundle, hidden, prefix_tokens, layer);
    }
    return state;
}

[[maybe_unused]] GptKvPrefillState run_gpt_kv_prefill_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& prefix, uint32_t prefix_tokens) {
    GptKvPrefillState state;
    state.caches.reserve(24);
    std::vector<float> hidden = prefix;
    for (uint32_t layer = 0; layer < 24; ++layer) {
        auto qkv = run_gpt_layer_qkv_metal(metal, bundle, hidden, prefix_tokens, layer);
        state.caches.push_back(extract_gpt_kv_cache(qkv, prefix_tokens));
        hidden = run_gpt_transformer_block_metal_layer(metal, bundle, hidden, prefix_tokens, layer);
    }
    return state;
}

GptKvPrefillState run_gpt_kv_prefill_metal(mit2::MetalContext& metal, const GptDecodeWeights& weights, const std::vector<float>& prefix, uint32_t prefix_tokens) {
    GptKvPrefillState state;
    state.caches.reserve(weights.layers.size());
    std::vector<float> hidden = prefix;
    for (uint32_t layer = 0; layer < weights.layers.size(); ++layer) {
        const auto& layer_weights = weights.layers[layer];
        auto qkv = run_gpt_layer_qkv_metal(metal, layer_weights, hidden, prefix_tokens);
        state.caches.push_back(extract_gpt_kv_cache(qkv, prefix_tokens));
        hidden = run_gpt_transformer_block_metal_layer(metal, layer_weights, hidden, prefix_tokens);
    }
    return state;
}

GptGreedyOutputs run_gpt_greedy_cpu(const mit2::Bundle& bundle, const std::vector<float>& prefix, uint32_t prefix_tokens, uint32_t steps) {
    constexpr uint32_t vocab = 8194;
    constexpr uint32_t start_mel_token = 8192;
    constexpr uint32_t stop_mel_token = 8193;
    GptGreedyOutputs out;
    out.input_tokens.push_back(start_mel_token);
    for (uint32_t step = 0; step < steps; ++step) {
        auto sequence = build_gpt_greedy_sequence_cpu(bundle, prefix, prefix_tokens, out.input_tokens);
        auto logits = run_gpt_logits_cpu(bundle, sequence, prefix_tokens + static_cast<uint32_t>(out.input_tokens.size()));
        out.last_logits.assign(logits.logits.end() - vocab, logits.logits.end());
        const uint32_t next = logits.last_argmax;
        out.predicted_tokens.push_back(next);
        if (next == stop_mel_token && out.first_stop_step < 0) {
            out.first_stop_step = static_cast<int32_t>(step);
        }
        out.input_tokens.push_back(next);
    }
    return out;
}

GptGreedyOutputs run_gpt_greedy_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& prefix, uint32_t prefix_tokens, uint32_t steps) {
    constexpr uint32_t vocab = 8194;
    constexpr uint32_t start_mel_token = 8192;
    constexpr uint32_t stop_mel_token = 8193;
    GptGreedyOutputs out;
    out.input_tokens.push_back(start_mel_token);
    for (uint32_t step = 0; step < steps; ++step) {
        auto sequence = build_gpt_greedy_sequence_metal(metal, bundle, prefix, prefix_tokens, out.input_tokens);
        auto logits = run_gpt_logits_metal(metal, bundle, sequence, prefix_tokens + static_cast<uint32_t>(out.input_tokens.size()));
        out.last_logits.assign(logits.logits.end() - vocab, logits.logits.end());
        const uint32_t next = logits.last_argmax;
        out.predicted_tokens.push_back(next);
        if (next == stop_mel_token && out.first_stop_step < 0) {
            out.first_stop_step = static_cast<int32_t>(step);
        }
        out.input_tokens.push_back(next);
    }
    return out;
}

uint32_t gpt_cached_generate_position(uint32_t step, bool hf_generate_positions) {
    return hf_generate_positions && step > 0 ? step + 1 : step;
}

GptGreedyOutputs run_gpt_kv_greedy_cpu(const mit2::Bundle& bundle, const std::vector<float>& prefix, uint32_t prefix_tokens, uint32_t steps, bool hf_generate_positions = false) {
    constexpr uint32_t vocab = 8194;
    constexpr uint32_t start_mel_token = 8192;
    constexpr uint32_t stop_mel_token = 8193;
    GptGreedyOutputs out;
    out.input_tokens.push_back(start_mel_token);
    auto state = run_gpt_kv_prefill_cpu(bundle, prefix, prefix_tokens);
    auto ln_f_weight = tensor_as_f32(bundle, "gpt.gpt.ln_f.weight");
    auto ln_f_bias = tensor_as_f32(bundle, "gpt.gpt.ln_f.bias");
    auto final_norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    auto final_norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    auto head_weight = tensor_as_f32(bundle, "gpt.mel_head.weight");
    auto head_bias = tensor_as_f32(bundle, "gpt.mel_head.bias");
    for (uint32_t step = 0; step < steps; ++step) {
        std::vector<float> current = build_gpt_greedy_current_cpu(bundle, out.input_tokens.back(), gpt_cached_generate_position(step, hf_generate_positions));
        for (uint32_t layer = 0; layer < 24; ++layer) {
            auto current_qkv = run_gpt_layer_qkv_cpu(bundle, current, 1, layer);
            current = run_gpt_cached_block_current_cpu(bundle, state.caches[layer], current, current_qkv, layer);
            append_gpt_current_kv(state.caches[layer], current_qkv);
        }
        auto ln_f = cpu_layernorm(current, ln_f_weight, ln_f_bias, 1e-5f);
        auto final_norm = cpu_layernorm(ln_f, final_norm_weight, final_norm_bias, 1e-5f);
        out.last_logits = cpu_linear(head_weight, head_bias, final_norm, vocab, 1280);
        const uint32_t next = argmax_row(out.last_logits, 0, vocab);
        out.predicted_tokens.push_back(next);
        if (next == stop_mel_token && out.first_stop_step < 0) {
            out.first_stop_step = static_cast<int32_t>(step);
            break;
        }
        out.input_tokens.push_back(next);
    }
    return out;
}

GptGreedyOutputs run_gpt_kv_greedy_metal(mit2::MetalContext& metal,
                                         const mit2::Bundle& bundle,
                                         const std::vector<float>& prefix,
                                         uint32_t prefix_tokens,
                                         uint32_t steps,
                                         bool hf_generate_positions = false,
                                         const GptSamplingConfig* sampling = nullptr,
                                         const std::vector<uint32_t>* repetition_history_prefix = nullptr) {
    constexpr uint32_t vocab = 8194;
    constexpr uint32_t width = 1280;
    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    constexpr uint32_t heads = 20;
    constexpr uint32_t head_dim = 64;
    constexpr uint32_t n_layers = 24;
    constexpr uint32_t start_mel_token = 8192;
    constexpr uint32_t stop_mel_token = 8193;
    GptGreedyOutputs out;
    out.input_tokens.push_back(start_mel_token);
    // Weights are immutable per bundle: cache the CPU-side struct across segments
    // (~1.9GB of tensor copies otherwise repeated per segment).
    static const mit2::Bundle* weights_cached_for = nullptr;
    static GptDecodeWeights weights_cache;
    if (weights_cached_for != &bundle) {
        weights_cache = load_gpt_decode_weights(bundle);
        weights_cached_for = &bundle;
    }
    const auto& weights = weights_cache;
    SplitMix64 rng(sampling ? sampling->seed : 0);

    // Resident KV cache: prefill writes it on-GPU in a single pass; per-token
    // appends happen inside gpt_cached_attention_resident_pass.
    const uint32_t max_kv_tokens = prefix_tokens + steps + 1;
    metal.gptKvCacheCreate(n_layers, max_kv_tokens, width);
    run_gpt_kv_prefill_metal_pass(metal, weights, prefix, prefix_tokens);
    uint32_t kv_tokens_counter = prefix_tokens;

    // Per layer: ln1(width) + qkv(qkv_width) + attn_out(width) + attn_proj(width)
    //          + attn_residual(width) + ln2(width) + c_fc(mlp_width) + gelu(mlp_width)
    //          + mlp_proj(width). Plus current/ln_f/final_norm/logits.
    // Per layer 10 slots: ln1(w) + qkv(3w) + attn_out(w) + attn_proj(w) + attn_residual(w)
    //                    + ln2(w) + c_fc(4w) + gelu(4w) + mlp_proj(w) + next_current(w).
    constexpr uint32_t per_layer_static_floats = qkv_width + 7 * width + 2 * mlp_width;  // 23040
    // +64 floats per allocation for the 256-byte slot alignment (10 allocs/layer + tail).
    const size_t workspace_bytes =
        (static_cast<size_t>(width) +
         static_cast<size_t>(n_layers) * (per_layer_static_floats + 10 * 64) +
         2 * width + vocab + 8 * 64 + 1024) * sizeof(float);

    static const bool gpt_icb_enabled = []() {
        const char* v = std::getenv("MIT2_GPT_ICB");
        return !(v && v[0] == '0');
    }();
    if (!(sampling && sampling->do_sample) && gpt_icb_enabled) {
        // ---------------------------------------------------------------
        // ICB greedy decode: the 127-dispatch per-token graph is recorded
        // ONCE into an indirect command buffer over a dedicated workspace;
        // each chunk just executes it K times (near-zero CPU encode).
        // kv_tokens/position/step live in a GPU state buffer advanced by a
        // recorded kernel; token ids land in a GPU history array.
        // ---------------------------------------------------------------
        if (!metal.gptIcbAvailable()) {
            const size_t icb_ws_bytes =
                (static_cast<size_t>(1 + width) +
                 static_cast<size_t>(n_layers) * (qkv_width + width + width + mlp_width + width) +
                 2 * width + vocab + 140 * 64) * sizeof(float);
            metal.gptIcbBeginRecord(132, icb_ws_bytes, 4096);
            auto token_slot = metal.gptIcbAlloc(1);
            auto current = metal.gptIcb_build_current(
                token_slot,
                "gpt.mel_embedding.weight.resident", weights.mel_embedding,
                "gpt.mel_pos_embedding.emb.weight.resident", weights.mel_pos_embedding,
                width);
            const mit2::PassSlot no_res{};
            for (uint32_t l = 0; l < n_layers; ++l) {
                const auto& lw = weights.layers[l];
                auto qkv_slot = metal.gptIcb_fused_gemv_f16w(
                    resident_weight_key(lw, "attn.c_attn"), lw.c_attn_weight,
                    resident_bias_key(lw, "attn.c_attn"), lw.c_attn_bias,
                    current, qkv_width, width,
                    true, resident_weight_key(lw, "ln_1"), lw.ln1_weight,
                    resident_bias_key(lw, "ln_1"), lw.ln1_bias,
                    false, false, no_res, 1e-5f);
                auto attn_out = metal.gptIcb_attention_resident(l, qkv_slot, heads, head_dim);
                auto attn_residual = metal.gptIcb_fused_gemv_f16w(
                    resident_weight_key(lw, "attn.c_proj"), lw.attn_proj_weight,
                    resident_bias_key(lw, "attn.c_proj"), lw.attn_proj_bias,
                    attn_out, width, width,
                    false, "", {}, "", {},
                    false, true, current, 1e-5f);
                auto gelu_out = metal.gptIcb_fused_gemv_f16w(
                    resident_weight_key(lw, "mlp.c_fc"), lw.c_fc_weight,
                    resident_bias_key(lw, "mlp.c_fc"), lw.c_fc_bias,
                    attn_residual, mlp_width, width,
                    true, resident_weight_key(lw, "ln_2"), lw.ln2_weight,
                    resident_bias_key(lw, "ln_2"), lw.ln2_bias,
                    true, false, no_res, 1e-5f);
                current = metal.gptIcb_fused_gemv_f16w(
                    resident_weight_key(lw, "mlp.c_proj"), lw.mlp_proj_weight,
                    resident_bias_key(lw, "mlp.c_proj"), lw.mlp_proj_bias,
                    gelu_out, width, mlp_width,
                    false, "", {}, "", {},
                    false, true, attn_residual, 1e-5f);
            }
            auto ln_f_out = metal.gptIcb_layernorm(
                "gpt.gpt.ln_f.weight.resident", weights.ln_f_weight,
                "gpt.gpt.ln_f.bias.resident", weights.ln_f_bias,
                current, width, 1e-5f);
            auto final_norm_out = metal.gptIcb_layernorm(
                "gpt.final_norm.weight.resident", weights.final_norm_weight,
                "gpt.final_norm.bias.resident", weights.final_norm_bias,
                ln_f_out, width, 1e-5f);
            auto logits_slot = metal.gptIcb_fused_gemv_f16w(
                "gpt.mel_head.weight.resident", weights.head_weight,
                "gpt.mel_head.bias.resident", weights.head_bias,
                final_norm_out, vocab, width,
                false, "", {}, "", {},
                false, false, no_res, 1e-5f);
            metal.gptIcb_argmax_into(logits_slot, vocab, token_slot);
            metal.gptIcb_record_token(token_slot);
            metal.gptIcb_advance_state();
            metal.gptIcbEndRecord(token_slot, logits_slot);
        }

        constexpr uint32_t kIcbChunk = 8;
        uint32_t step = 0;
        bool stopped = false;
        while (step < steps && !stopped) {
            const uint32_t chunk = std::min(kIcbChunk, steps - step);
            auto res = metal.gptIcbExecute(chunk, out.input_tokens.back(), kv_tokens_counter, step, vocab);
            out.last_logits = res.last_logits;
            out.last_processed_logits = out.last_logits;
            for (uint32_t j = 0; j < chunk; ++j) {
                const uint32_t next = res.tokens[j];
                ++kv_tokens_counter;
                out.predicted_tokens.push_back(next);
                if (next == stop_mel_token && out.first_stop_step < 0) {
                    out.first_stop_step = static_cast<int32_t>(step + j);
                    stopped = true;
                    kv_tokens_counter -= 1;
                    break;
                }
                out.input_tokens.push_back(next);
            }
            step += chunk;
        }
        return out;
    }

    if (!(sampling && sampling->do_sample)) {
        // ---------------------------------------------------------------
        // Greedy chunked decode: K tokens per command buffer. GPU-side
        // argmax + next-embedding build chain the tokens inside one pass,
        // amortizing the command-buffer round-trip latency K-fold.
        // ---------------------------------------------------------------
        constexpr uint32_t kChunk = 8;
        const mit2::PassSlot no_residual{};
        uint32_t step = 0;
        bool stopped = false;
        while (step < steps && !stopped) {
            const uint32_t chunk = std::min(kChunk, steps - step);
            metal.beginPass(workspace_bytes * chunk + 4096);

            // Seed token slot with the previous token id.
            const uint32_t seed_token = out.input_tokens.back();
            auto token_slot = metal.passUploadAllocU32(std::vector<uint32_t>{seed_token});
            std::vector<mit2::PassSlot> token_slots(chunk);
            mit2::PassSlot logits_slot{};

            for (uint32_t j = 0; j < chunk; ++j) {
                const uint32_t pos = gpt_cached_generate_position(step + j, hf_generate_positions);
                auto current_slot = metal.gpt_build_current_pass(
                    token_slot,
                    "gpt.mel_embedding.weight.resident", weights.mel_embedding,
                    "gpt.mel_pos_embedding.emb.weight.resident", weights.mel_pos_embedding,
                    width, pos);
                for (uint32_t l = 0; l < n_layers; ++l) {
                    const auto& lw = weights.layers[l];
                    auto qkv_slot = metal.gpt_fused_gemv_f16w_pass(
                        resident_weight_key(lw, "attn.c_attn"), lw.c_attn_weight,
                        resident_bias_key(lw, "attn.c_attn"), lw.c_attn_bias,
                        current_slot, qkv_width, width,
                        true, resident_weight_key(lw, "ln_1"), lw.ln1_weight,
                        resident_bias_key(lw, "ln_1"), lw.ln1_bias,
                        false, false, no_residual, 1e-5f);
                    auto attn_out = metal.gpt_cached_attention_resident_pass(
                        l, qkv_slot, kv_tokens_counter + j, heads, head_dim);
                    auto attn_residual = metal.gpt_fused_gemv_f16w_pass(
                        resident_weight_key(lw, "attn.c_proj"), lw.attn_proj_weight,
                        resident_bias_key(lw, "attn.c_proj"), lw.attn_proj_bias,
                        attn_out, width, width,
                        false, "", {}, "", {},
                        false, true, current_slot, 1e-5f);
                    auto gelu_out = metal.gpt_fused_gemv_f16w_pass(
                        resident_weight_key(lw, "mlp.c_fc"), lw.c_fc_weight,
                        resident_bias_key(lw, "mlp.c_fc"), lw.c_fc_bias,
                        attn_residual, mlp_width, width,
                        true, resident_weight_key(lw, "ln_2"), lw.ln2_weight,
                        resident_bias_key(lw, "ln_2"), lw.ln2_bias,
                        true, false, no_residual, 1e-5f);
                    current_slot = metal.gpt_fused_gemv_f16w_pass(
                        resident_weight_key(lw, "mlp.c_proj"), lw.mlp_proj_weight,
                        resident_bias_key(lw, "mlp.c_proj"), lw.mlp_proj_bias,
                        gelu_out, width, mlp_width,
                        false, "", {}, "", {},
                        false, true, attn_residual, 1e-5f);
                }
                auto ln_f_out = metal.layernorm_f32_pass(
                    "gpt.gpt.ln_f.weight.resident", weights.ln_f_weight,
                    "gpt.gpt.ln_f.bias.resident", weights.ln_f_bias,
                    current_slot, width, 1e-5f);
                auto final_norm_out = metal.layernorm_f32_pass(
                    "gpt.final_norm.weight.resident", weights.final_norm_weight,
                    "gpt.final_norm.bias.resident", weights.final_norm_bias,
                    ln_f_out, width, 1e-5f);
                logits_slot = metal.linear_f32_pass(
                    "gpt.mel_head.weight.resident", weights.head_weight,
                    "gpt.mel_head.bias.resident", weights.head_bias,
                    final_norm_out, vocab, width);
                token_slot = metal.gpt_argmax_pass(logits_slot, vocab);
                token_slots[j] = token_slot;
            }

            metal.endPass();

            out.last_logits = metal.passRead(logits_slot);
            out.last_processed_logits = out.last_logits;
            for (uint32_t j = 0; j < chunk; ++j) {
                const uint32_t next = metal.passReadU32(token_slots[j])[0];
                ++kv_tokens_counter;
                out.predicted_tokens.push_back(next);
                if (next == stop_mel_token && out.first_stop_step < 0) {
                    out.first_stop_step = static_cast<int32_t>(step + j);
                    stopped = true;
                    // K/V already appended for speculatively decoded tokens past the
                    // stop; they are never attended to (counter rolls back below).
                    kv_tokens_counter -= 1;
                    out.predicted_tokens.resize(out.predicted_tokens.size());
                    break;
                }
                out.input_tokens.push_back(next);
            }
            step += chunk;
        }
        return out;
    }

    for (uint32_t step = 0; step < steps; ++step) {
        const uint32_t kv_tokens = kv_tokens_counter;

        const std::vector<float> current_in = build_gpt_greedy_current_cached(
            weights, out.input_tokens.back(), gpt_cached_generate_position(step, hf_generate_positions));

        metal.beginPass(workspace_bytes);
        auto current_slot = metal.passUploadAlloc(current_in);

        const mit2::PassSlot no_residual{};
        for (uint32_t l = 0; l < n_layers; ++l) {
            const auto& lw = weights.layers[l];

            // 5 fused dispatches per layer (was 10): ln1+qkv, attention,
            // proj+residual, ln2+fc+gelu, mlp_proj+residual.
            auto qkv_slot = metal.gpt_fused_gemv_f16w_pass(
                resident_weight_key(lw, "attn.c_attn"), lw.c_attn_weight,
                resident_bias_key(lw, "attn.c_attn"), lw.c_attn_bias,
                current_slot, qkv_width, width,
                /*has_ln=*/true, resident_weight_key(lw, "ln_1"), lw.ln1_weight,
                resident_bias_key(lw, "ln_1"), lw.ln1_bias,
                /*gelu=*/false, /*residual=*/false, no_residual, 1e-5f);
            auto attn_out = metal.gpt_cached_attention_resident_pass(
                l, qkv_slot, kv_tokens, heads, head_dim);
            auto attn_residual = metal.gpt_fused_gemv_f16w_pass(
                resident_weight_key(lw, "attn.c_proj"), lw.attn_proj_weight,
                resident_bias_key(lw, "attn.c_proj"), lw.attn_proj_bias,
                attn_out, width, width,
                /*has_ln=*/false, "", {}, "", {},
                /*gelu=*/false, /*residual=*/true, current_slot, 1e-5f);
            auto gelu_out = metal.gpt_fused_gemv_f16w_pass(
                resident_weight_key(lw, "mlp.c_fc"), lw.c_fc_weight,
                resident_bias_key(lw, "mlp.c_fc"), lw.c_fc_bias,
                attn_residual, mlp_width, width,
                /*has_ln=*/true, resident_weight_key(lw, "ln_2"), lw.ln2_weight,
                resident_bias_key(lw, "ln_2"), lw.ln2_bias,
                /*gelu=*/true, /*residual=*/false, no_residual, 1e-5f);
            current_slot = metal.gpt_fused_gemv_f16w_pass(
                resident_weight_key(lw, "mlp.c_proj"), lw.mlp_proj_weight,
                resident_bias_key(lw, "mlp.c_proj"), lw.mlp_proj_bias,
                gelu_out, width, mlp_width,
                /*has_ln=*/false, "", {}, "", {},
                /*gelu=*/false, /*residual=*/true, attn_residual, 1e-5f);
        }

        auto ln_f_out = metal.layernorm_f32_pass(
            "gpt.gpt.ln_f.weight.resident", weights.ln_f_weight,
            "gpt.gpt.ln_f.bias.resident", weights.ln_f_bias,
            current_slot, width, 1e-5f);
        auto final_norm_out = metal.layernorm_f32_pass(
            "gpt.final_norm.weight.resident", weights.final_norm_weight,
            "gpt.final_norm.bias.resident", weights.final_norm_bias,
            ln_f_out, width, 1e-5f);
        auto logits_slot = metal.linear_f32_pass(
            "gpt.mel_head.weight.resident", weights.head_weight,
            "gpt.mel_head.bias.resident", weights.head_bias,
            final_norm_out, vocab, width);

        metal.endPass();

        out.last_logits = metal.passRead(logits_slot);
        ++kv_tokens_counter;  // K/V appended on-GPU by the resident attention op

        uint32_t next = 0;
        if (sampling && sampling->do_sample) {
            std::vector<uint32_t> history;
            if (repetition_history_prefix) {
                history.insert(history.end(), repetition_history_prefix->begin(), repetition_history_prefix->end());
            } else {
                history.push_back(start_mel_token);
            }
            history.insert(history.end(), out.predicted_tokens.begin(), out.predicted_tokens.end());
            out.last_processed_logits = apply_gpt_sampling_processors(out.last_logits, history, *sampling);
            next = sample_gpt_token_from_logits(out.last_processed_logits, rng);
        } else {
            out.last_processed_logits = out.last_logits;
            next = argmax_row(out.last_logits, 0, vocab);
        }
        out.predicted_tokens.push_back(next);
        if (next == stop_mel_token && out.first_stop_step < 0) {
            out.first_stop_step = static_cast<int32_t>(step);
            break;
        }
        out.input_tokens.push_back(next);
    }
    return out;
}

bool run_gpt_greedy_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t cond_tokens = 8;
    constexpr uint32_t text_slots = 0;
    constexpr uint32_t width = 1280;
    constexpr uint32_t steps = 3;
    std::vector<float> cond(static_cast<size_t>(cond_tokens) * width);
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.007f) * 0.045f + std::cos(static_cast<float>(i % 251) * 0.011f) * 0.018f;
    }
    std::vector<uint32_t> text_inputs{0, 1};
    auto prefix_ref = run_gpt_prepare_inputs_cpu(bundle, cond, text_inputs, cond_tokens, text_slots);
    auto prefix_got = run_gpt_prepare_inputs_metal(metal, bundle, cond, text_inputs, cond_tokens, text_slots);
    const uint32_t prefix_tokens = static_cast<uint32_t>(prefix_ref.inputs_embeds.size() / width);
    std::vector<uint32_t> start_tokens{8192};
    auto first_sequence_ref = build_gpt_greedy_sequence_cpu(bundle, prefix_ref.inputs_embeds, prefix_tokens, start_tokens);
    auto first_sequence_got = build_gpt_greedy_sequence_metal(metal, bundle, prefix_got.inputs_embeds, prefix_tokens, start_tokens);
    auto ref = run_gpt_greedy_cpu(bundle, prefix_ref.inputs_embeds, prefix_tokens, steps);
    auto got = run_gpt_greedy_metal(metal, bundle, prefix_got.inputs_embeds, prefix_tokens, steps);
    const bool tokens_match = got.predicted_tokens == ref.predicted_tokens;
    const bool inputs_match = got.input_tokens == ref.input_tokens;
    const float prefix_err = max_abs_error(prefix_got.inputs_embeds, prefix_ref.inputs_embeds);
    const float first_sequence_err = max_abs_error(first_sequence_got, first_sequence_ref);
    const float logits_err = max_abs_error(got.last_logits, ref.last_logits);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_greedy_loop\",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"prefix_max_abs_error\": " << prefix_err << ",\n";
    std::cout << "  \"first_sequence_max_abs_error\": " << first_sequence_err << ",\n";
    std::cout << "  \"last_logits_max_abs_error\": " << logits_err << ",\n";
    std::cout << "  \"last_logits_nonfinite_ref\": " << count_nonfinite(ref.last_logits) << ",\n";
    std::cout << "  \"last_logits_nonfinite_got\": " << count_nonfinite(got.last_logits) << ",\n";
    std::cout << "  \"tokens_match\": " << (tokens_match ? "true" : "false") << ",\n";
    std::cout << "  \"inputs_match\": " << (inputs_match ? "true" : "false") << ",\n";
    std::cout << "  \"first_stop_step_ref\": " << ref.first_stop_step << ",\n";
    std::cout << "  \"first_stop_step_got\": " << got.first_stop_step << ",\n";
    std::cout << "  \"predicted_tokens_ref\": [";
    for (size_t i = 0; i < ref.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << ref.predicted_tokens[i];
    }
    std::cout << "],\n";
    std::cout << "  \"predicted_tokens_got\": [";
    for (size_t i = 0; i < got.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << got.predicted_tokens[i];
    }
    std::cout << "],\n";
    std::cout << "  \"max_abs_error\": " << std::max({prefix_err, first_sequence_err, logits_err}) << "\n";
    std::cout << "}\n";
    return prefix_err <= 1e-6f && first_sequence_err <= 1e-6f && logits_err <= 3e-3f && tokens_match && inputs_match && ref.first_stop_step == got.first_stop_step;
}

bool run_gpt_kv_greedy_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t cond_tokens = 8;
    constexpr uint32_t text_slots = 0;
    constexpr uint32_t width = 1280;
    constexpr uint32_t steps = 3;
    std::vector<float> cond(static_cast<size_t>(cond_tokens) * width);
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.007f) * 0.045f + std::cos(static_cast<float>(i % 251) * 0.011f) * 0.018f;
    }
    std::vector<uint32_t> text_inputs{0, 1};
    auto prefix_ref = run_gpt_prepare_inputs_cpu(bundle, cond, text_inputs, cond_tokens, text_slots);
    auto prefix_got = run_gpt_prepare_inputs_metal(metal, bundle, cond, text_inputs, cond_tokens, text_slots);
    const uint32_t prefix_tokens = static_cast<uint32_t>(prefix_ref.inputs_embeds.size() / width);
    auto full_ref = run_gpt_greedy_cpu(bundle, prefix_ref.inputs_embeds, prefix_tokens, steps);
    auto cached_ref = run_gpt_kv_greedy_cpu(bundle, prefix_ref.inputs_embeds, prefix_tokens, steps);
    auto cached_got = run_gpt_kv_greedy_metal(metal, bundle, prefix_got.inputs_embeds, prefix_tokens, steps);
    const bool cpu_tokens_match = cached_ref.predicted_tokens == full_ref.predicted_tokens;
    const bool metal_tokens_match = cached_got.predicted_tokens == cached_ref.predicted_tokens;
    const bool cpu_inputs_match = cached_ref.input_tokens == full_ref.input_tokens;
    const bool metal_inputs_match = cached_got.input_tokens == cached_ref.input_tokens;
    const float prefix_err = max_abs_error(prefix_got.inputs_embeds, prefix_ref.inputs_embeds);
    const float cpu_logits_err = max_abs_error(cached_ref.last_logits, full_ref.last_logits);
    const float metal_logits_err = max_abs_error(cached_got.last_logits, cached_ref.last_logits);
    const float err = std::max({prefix_err, cpu_logits_err, metal_logits_err});
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_greedy_loop\",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"layers\": 24,\n";
    std::cout << "  \"prefix_max_abs_error\": " << prefix_err << ",\n";
    std::cout << "  \"cpu_cached_vs_full_last_logits_max_abs_error\": " << cpu_logits_err << ",\n";
    std::cout << "  \"metal_cached_last_logits_max_abs_error\": " << metal_logits_err << ",\n";
    std::cout << "  \"last_logits_nonfinite_full_ref\": " << count_nonfinite(full_ref.last_logits) << ",\n";
    std::cout << "  \"last_logits_nonfinite_cached_ref\": " << count_nonfinite(cached_ref.last_logits) << ",\n";
    std::cout << "  \"last_logits_nonfinite_cached_got\": " << count_nonfinite(cached_got.last_logits) << ",\n";
    std::cout << "  \"cpu_tokens_match\": " << (cpu_tokens_match ? "true" : "false") << ",\n";
    std::cout << "  \"metal_tokens_match\": " << (metal_tokens_match ? "true" : "false") << ",\n";
    std::cout << "  \"cpu_inputs_match\": " << (cpu_inputs_match ? "true" : "false") << ",\n";
    std::cout << "  \"metal_inputs_match\": " << (metal_inputs_match ? "true" : "false") << ",\n";
    std::cout << "  \"first_stop_step_full_ref\": " << full_ref.first_stop_step << ",\n";
    std::cout << "  \"first_stop_step_cached_ref\": " << cached_ref.first_stop_step << ",\n";
    std::cout << "  \"first_stop_step_cached_got\": " << cached_got.first_stop_step << ",\n";
    std::cout << "  \"predicted_tokens_full_ref\": [";
    for (size_t i = 0; i < full_ref.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << full_ref.predicted_tokens[i];
    }
    std::cout << "],\n";
    std::cout << "  \"predicted_tokens_cached_ref\": [";
    for (size_t i = 0; i < cached_ref.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << cached_ref.predicted_tokens[i];
    }
    std::cout << "],\n";
    std::cout << "  \"predicted_tokens_cached_got\": [";
    for (size_t i = 0; i < cached_got.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << cached_got.predicted_tokens[i];
    }
    std::cout << "],\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return prefix_err <= 1e-6f && cpu_logits_err <= 3e-3f && metal_logits_err <= 3e-3f &&
           cpu_tokens_match && metal_tokens_match && cpu_inputs_match && metal_inputs_match &&
           full_ref.first_stop_step == cached_ref.first_stop_step && cached_ref.first_stop_step == cached_got.first_stop_step;
}

std::vector<float> run_gpt_perceiver_cpu(const mit2::Bundle& bundle,
                                         const std::vector<float>& context,
                                         const std::vector<uint32_t>& mask,
                                         uint32_t context_tokens,
                                         const std::string& perceiver_prefix = "gpt.perceiver_encoder",
                                         uint32_t context_dim = 512,
                                         uint32_t dim = 1280,
                                         uint32_t latents = 32,
                                         uint32_t heads = 8) {
    constexpr uint32_t head_dim = 64;
    const uint32_t inner = heads * head_dim;
    if (context.size() != static_cast<size_t>(context_tokens) * context_dim) {
        throw std::runtime_error("GPT perceiver context shape mismatch");
    }
    if (mask.size() != static_cast<size_t>(latents) + context_tokens) {
        throw std::runtime_error("GPT perceiver mask shape mismatch");
    }

    const std::vector<float> zero_inner_bias(inner, 0.0f);
    const std::vector<float> zero_kv_bias(inner * 2, 0.0f);
    const std::vector<float> zero_dim_bias(dim, 0.0f);

    auto proj_w = tensor_as_f32(bundle, perceiver_prefix + ".proj_context.weight");
    auto proj_b = tensor_as_f32(bundle, perceiver_prefix + ".proj_context.bias");
    auto projected_context = cpu_linear_rows(proj_w, proj_b, context, context_tokens, dim, context_dim);

    auto latents_tensor = tensor_as_f32(bundle, perceiver_prefix + ".latents");
    if (latents_tensor.size() != static_cast<size_t>(latents) * dim) {
        throw std::runtime_error("gpt.perceiver_encoder.latents shape mismatch");
    }
    std::vector<float> latent_state = latents_tensor;

    for (uint32_t layer = 0; layer < 2; ++layer) {
        const std::string base = perceiver_prefix + ".layers." + std::to_string(layer);
        auto to_q = tensor_as_f32(bundle, base + ".0.to_q.weight");
        auto to_kv = tensor_as_f32(bundle, base + ".0.to_kv.weight");
        auto to_out = tensor_as_f32(bundle, base + ".0.to_out.weight");
        auto ff0_w = tensor_as_f32(bundle, base + ".1.0.weight");
        auto ff0_b = tensor_as_f32(bundle, base + ".1.0.bias");
        auto ff2_w = tensor_as_f32(bundle, base + ".1.2.weight");
        auto ff2_b = tensor_as_f32(bundle, base + ".1.2.bias");
        if ((ff0_b.size() % 2) != 0) {
            throw std::runtime_error("GPT perceiver GEGLU bias shape mismatch");
        }
        const uint32_t ff_inner = static_cast<uint32_t>(ff0_b.size() / 2);

        const uint32_t key_tokens = latents + context_tokens;
        std::vector<float> combined_context(static_cast<size_t>(key_tokens) * dim);
        std::copy(latent_state.begin(), latent_state.end(), combined_context.begin());
        std::copy(projected_context.begin(), projected_context.end(), combined_context.begin() + static_cast<size_t>(latents) * dim);

        auto q = cpu_linear_rows(to_q, zero_inner_bias, latent_state, latents, inner, dim);
        auto kv = cpu_linear_rows(to_kv, zero_kv_bias, combined_context, key_tokens, inner * 2, dim);
        std::vector<float> k(static_cast<size_t>(key_tokens) * inner);
        std::vector<float> v(static_cast<size_t>(key_tokens) * inner);
        for (uint32_t t = 0; t < key_tokens; ++t) {
            std::copy(kv.begin() + static_cast<size_t>(t) * inner * 2,
                      kv.begin() + static_cast<size_t>(t) * inner * 2 + inner,
                      k.begin() + static_cast<size_t>(t) * inner);
            std::copy(kv.begin() + static_cast<size_t>(t) * inner * 2 + inner,
                      kv.begin() + static_cast<size_t>(t + 1) * inner * 2,
                      v.begin() + static_cast<size_t>(t) * inner);
        }

        auto attn = cpu_cross_attention_heads_masked(q, k, v, mask, latents, key_tokens, heads, head_dim);
        auto attn_projected = cpu_linear_rows(to_out, zero_dim_bias, attn, latents, dim, inner);
        for (size_t i = 0; i < latent_state.size(); ++i) {
            latent_state[i] += attn_projected[i];
        }

        auto ff0 = cpu_linear_rows(ff0_w, ff0_b, latent_state, latents, ff_inner * 2, dim);
        std::vector<float> ff_gate(static_cast<size_t>(latents) * ff_inner);
        for (uint32_t t = 0; t < latents; ++t) {
            std::copy(ff0.begin() + static_cast<size_t>(t) * ff_inner * 2 + ff_inner,
                      ff0.begin() + static_cast<size_t>(t + 1) * ff_inner * 2,
                      ff_gate.begin() + static_cast<size_t>(t) * ff_inner);
        }
        ff_gate = cpu_gelu_erf(ff_gate);
        std::vector<float> ff_mid(static_cast<size_t>(latents) * ff_inner);
        for (uint32_t t = 0; t < latents; ++t) {
            for (uint32_t d = 0; d < ff_inner; ++d) {
                const size_t src = static_cast<size_t>(t) * ff_inner * 2 + d;
                const size_t dst = static_cast<size_t>(t) * ff_inner + d;
                ff_mid[dst] = ff0[src] * ff_gate[dst];
            }
        }
        auto ff_out = cpu_linear_rows(ff2_w, ff2_b, ff_mid, latents, dim, ff_inner);
        for (size_t i = 0; i < latent_state.size(); ++i) {
            latent_state[i] += ff_out[i];
        }
    }

    auto gamma = tensor_as_f32(bundle, perceiver_prefix + ".norm.gamma");
    return cpu_rmsnorm_rows(latent_state, gamma, latents, dim);
}

std::vector<float> run_gpt_perceiver_metal(mit2::MetalContext& metal,
                                           const mit2::Bundle& bundle,
                                           const std::vector<float>& context,
                                           const std::vector<uint32_t>& mask,
                                           uint32_t context_tokens,
                                           const std::string& perceiver_prefix = "gpt.perceiver_encoder",
                                           uint32_t context_dim = 512,
                                           uint32_t dim = 1280,
                                           uint32_t latents = 32,
                                           uint32_t heads = 8) {
    constexpr uint32_t head_dim = 64;
    const uint32_t inner = heads * head_dim;
    if (context.size() != static_cast<size_t>(context_tokens) * context_dim) {
        throw std::runtime_error("GPT perceiver context shape mismatch");
    }
    if (mask.size() != static_cast<size_t>(latents) + context_tokens) {
        throw std::runtime_error("GPT perceiver mask shape mismatch");
    }

    const std::vector<float> zero_inner_bias(inner, 0.0f);
    const std::vector<float> zero_kv_bias(inner * 2, 0.0f);
    const std::vector<float> zero_dim_bias(dim, 0.0f);

    auto proj_w = tensor_as_f32(bundle, perceiver_prefix + ".proj_context.weight");
    auto proj_b = tensor_as_f32(bundle, perceiver_prefix + ".proj_context.bias");
    auto projected_context = metal.linear_rows_f32_resident(
        perceiver_prefix + ".proj_context.weight.resident",
        proj_w,
        perceiver_prefix + ".proj_context.bias.resident",
        proj_b,
        context,
        context_tokens,
        dim,
        context_dim);

    auto latents_tensor = tensor_as_f32(bundle, perceiver_prefix + ".latents");
    if (latents_tensor.size() != static_cast<size_t>(latents) * dim) {
        throw std::runtime_error("gpt.perceiver_encoder.latents shape mismatch");
    }
    std::vector<float> latent_state = latents_tensor;

    for (uint32_t layer = 0; layer < 2; ++layer) {
        const std::string base = perceiver_prefix + ".layers." + std::to_string(layer);
        auto to_q = tensor_as_f32(bundle, base + ".0.to_q.weight");
        auto to_kv = tensor_as_f32(bundle, base + ".0.to_kv.weight");
        auto to_out = tensor_as_f32(bundle, base + ".0.to_out.weight");
        auto ff0_w = tensor_as_f32(bundle, base + ".1.0.weight");
        auto ff0_b = tensor_as_f32(bundle, base + ".1.0.bias");
        auto ff2_w = tensor_as_f32(bundle, base + ".1.2.weight");
        auto ff2_b = tensor_as_f32(bundle, base + ".1.2.bias");
        if ((ff0_b.size() % 2) != 0) {
            throw std::runtime_error("GPT perceiver GEGLU bias shape mismatch");
        }
        const uint32_t ff_inner = static_cast<uint32_t>(ff0_b.size() / 2);

        const uint32_t key_tokens = latents + context_tokens;
        std::vector<float> combined_context(static_cast<size_t>(key_tokens) * dim);
        std::copy(latent_state.begin(), latent_state.end(), combined_context.begin());
        std::copy(projected_context.begin(), projected_context.end(), combined_context.begin() + static_cast<size_t>(latents) * dim);

        auto q = metal.linear_rows_f32_resident(
            base + ".0.to_q.weight.resident",
            to_q,
            perceiver_prefix + ".zero_inner_bias.resident",
            zero_inner_bias,
            latent_state,
            latents,
            inner,
            dim);
        auto kv = metal.linear_rows_f32_resident(
            base + ".0.to_kv.weight.resident",
            to_kv,
            perceiver_prefix + ".zero_kv_bias.resident",
            zero_kv_bias,
            combined_context,
            key_tokens,
            inner * 2,
            dim);
        std::vector<float> k(static_cast<size_t>(key_tokens) * inner);
        std::vector<float> v(static_cast<size_t>(key_tokens) * inner);
        for (uint32_t t = 0; t < key_tokens; ++t) {
            std::copy(kv.begin() + static_cast<size_t>(t) * inner * 2,
                      kv.begin() + static_cast<size_t>(t) * inner * 2 + inner,
                      k.begin() + static_cast<size_t>(t) * inner);
            std::copy(kv.begin() + static_cast<size_t>(t) * inner * 2 + inner,
                      kv.begin() + static_cast<size_t>(t + 1) * inner * 2,
                      v.begin() + static_cast<size_t>(t) * inner);
        }

        auto attn = metal.cross_attention_heads_masked_f32(q, k, v, mask, latents, key_tokens, heads, head_dim);
        auto attn_projected = metal.linear_rows_f32_resident(
            base + ".0.to_out.weight.resident",
            to_out,
            perceiver_prefix + ".zero_dim_bias.resident",
            zero_dim_bias,
            attn,
            latents,
            dim,
            inner);
        latent_state = metal.add_f32(latent_state, attn_projected);

        auto ff0 = metal.linear_rows_f32_resident(
            base + ".1.0.weight.resident",
            ff0_w,
            base + ".1.0.bias.resident",
            ff0_b,
            latent_state,
            latents,
            ff_inner * 2,
            dim);
        auto ff_mid = metal.geglu_erf_split_f32(ff0, latents, ff_inner);
        auto ff_out = metal.linear_rows_f32_resident(
            base + ".1.2.weight.resident",
            ff2_w,
            base + ".1.2.bias.resident",
            ff2_b,
            ff_mid,
            latents,
            dim,
            ff_inner);
        latent_state = metal.add_f32(latent_state, ff_out);
    }

    auto gamma = tensor_as_f32(bundle, perceiver_prefix + ".norm.gamma");
    return metal.rmsnorm_rows_f32_resident(
        perceiver_prefix + ".norm.gamma.resident",
        gamma,
        latent_state,
        latents,
        dim);
}

bool run_gpt_perceiver_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t context_dim = 512;
    constexpr uint32_t dim = 1280;
    auto context = read_raw_f32(golden_dir + "/conditioning_context.f32");
    auto mask = read_raw_u32(golden_dir + "/perceiver_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/perceiver.f32");
    if (context.empty() || (context.size() % context_dim) != 0) {
        throw std::runtime_error("GPT perceiver golden context must have shape [tokens,512]");
    }
    if (golden.size() != static_cast<size_t>(32) * dim) {
        throw std::runtime_error("GPT perceiver golden output must have shape [32,1280]");
    }
    const uint32_t context_tokens = static_cast<uint32_t>(context.size() / context_dim);
    auto got = run_gpt_perceiver_metal(metal, bundle, context, mask, context_tokens);
    const float err = max_abs_error(got, golden);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_perceiver_golden\",\n";
    std::cout << "  \"projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"attention_source\": \"metal_cross_attention_heads_masked_f32\",\n";
    std::cout << "  \"ff_source\": \"metal_geglu_erf_split_linear_rows_resident_add\",\n";
    std::cout << "  \"norm_source\": \"metal_rmsnorm_rows_f32_resident\",\n";
    std::cout << "  \"context_tokens\": " << context_tokens << ",\n";
    std::cout << "  \"context_dim\": " << context_dim << ",\n";
    std::cout << "  \"latents\": 32,\n";
    std::cout << "  \"latent_dim\": " << dim << ",\n";
    std::cout << "  \"max_abs_error\": " << err << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << "\n";
    std::cout << "}\n";
    return err <= 2e-3f && count_nonfinite(got) == 0;
}

bool export_gpt_perceiver(const std::string& bundle_dir,
                          const std::string& context_path,
                          const std::string& mask_path,
                          const std::string& output_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t context_dim = 512;
    constexpr uint32_t dim = 1280;
    auto context = read_raw_f32(context_path);
    auto mask = read_raw_u32(mask_path);
    if (context.empty() || (context.size() % context_dim) != 0) {
        throw std::runtime_error("GPT perceiver export context must have shape [tokens,512]");
    }
    const uint32_t context_tokens = static_cast<uint32_t>(context.size() / context_dim);
    auto out = run_gpt_perceiver_metal(metal, bundle, context, mask, context_tokens);
    write_raw_f32(output_path, out);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_perceiver_export\",\n";
    std::cout << "  \"projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"attention_source\": \"metal_cross_attention_heads_masked_f32\",\n";
    std::cout << "  \"ff_source\": \"metal_geglu_erf_split_linear_rows_resident_add\",\n";
    std::cout << "  \"norm_source\": \"metal_rmsnorm_rows_f32_resident\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"context_f32\": \"" << json_escape(context_path) << "\",\n";
    std::cout << "  \"mask_u32\": \"" << json_escape(mask_path) << "\",\n";
    std::cout << "  \"output_f32\": \"" << json_escape(output_path) << "\",\n";
    std::cout << "  \"context_tokens\": " << context_tokens << ",\n";
    std::cout << "  \"context_dim\": " << context_dim << ",\n";
    std::cout << "  \"latents\": 32,\n";
    std::cout << "  \"latent_dim\": " << dim << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(out) << "\n";
    std::cout << "}\n";
    return count_nonfinite(out) == 0;
}

struct GptSubsamplingOutput {
    std::vector<float> subsampling;
    std::vector<float> pos_emb;
    std::vector<uint32_t> mask;
    uint32_t output_tokens = 0;
};

GptSubsamplingOutput run_gpt_conditioning_subsampling_cpu(const mit2::Bundle& bundle,
                                                          const std::vector<float>& input,
                                                          uint32_t input_tokens,
                                                          const std::string& encoder_prefix = "gpt.conditioning_encoder") {
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t channels = 512;
    constexpr uint32_t conv_kernel = 3;
    constexpr uint32_t conv_stride = 2;
    constexpr uint32_t conv_freq = 511;
    constexpr uint32_t flat_dim = channels * conv_freq;
    constexpr uint32_t output_dim = 512;
    if (input_tokens < conv_kernel) {
        throw std::runtime_error("GPT subsampling input_tokens must be at least 3");
    }
    if (input.size() != static_cast<size_t>(input_tokens) * input_dim) {
        throw std::runtime_error("GPT subsampling input must have shape [tokens,1024]");
    }
    const uint32_t output_tokens = ((input_tokens - conv_kernel) / conv_stride) + 1;

    auto conv_w = tensor_as_f32(bundle, encoder_prefix + ".embed.conv.0.weight");
    auto conv_b = tensor_as_f32(bundle, encoder_prefix + ".embed.conv.0.bias");
    auto linear_w = tensor_as_f32(bundle, encoder_prefix + ".embed.out.0.weight");
    auto linear_b = tensor_as_f32(bundle, encoder_prefix + ".embed.out.0.bias");
    auto pe = tensor_as_f32(bundle, encoder_prefix + ".embed.pos_enc.pe");
    if (conv_w.size() != static_cast<size_t>(channels) * conv_kernel * conv_kernel || conv_b.size() != channels) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.conv.0 shape mismatch");
    }
    if (linear_w.size() != static_cast<size_t>(output_dim) * flat_dim || linear_b.size() != output_dim) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.out.0 shape mismatch");
    }
    if (pe.size() < static_cast<size_t>(output_tokens) * output_dim) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.pos_enc.pe shape mismatch");
    }

    auto flat = cpu_subsampling_conv2d_relu_flat(input, conv_w, conv_b, input_tokens, input_dim, channels, conv_kernel, conv_stride);

    std::vector<float> subsampling(static_cast<size_t>(output_tokens) * output_dim);
    const float positional_scale = std::sqrt(static_cast<float>(output_dim));
    for (uint32_t t = 0; t < output_tokens; ++t) {
        const float* flat_row = flat.data() + static_cast<size_t>(t) * flat_dim;
        for (uint32_t row = 0; row < output_dim; ++row) {
            float acc = linear_b[row];
            const float* weight_row = linear_w.data() + static_cast<size_t>(row) * flat_dim;
            for (uint32_t col = 0; col < flat_dim; ++col) {
                acc += weight_row[col] * flat_row[col];
            }
            subsampling[static_cast<size_t>(t) * output_dim + row] = acc * positional_scale;
        }
    }

    std::vector<float> pos_emb(static_cast<size_t>(output_tokens) * output_dim);
    std::copy(pe.begin(), pe.begin() + pos_emb.size(), pos_emb.begin());
    std::vector<uint32_t> mask(output_tokens, 1);
    return GptSubsamplingOutput{std::move(subsampling), std::move(pos_emb), std::move(mask), output_tokens};
}

GptSubsamplingOutput run_gpt_conditioning_subsampling_metal_linear(mit2::MetalContext& metal,
                                                                   const mit2::Bundle& bundle,
                                                                   const std::vector<float>& input,
                                                                   uint32_t input_tokens,
                                                                   const std::string& encoder_prefix = "gpt.conditioning_encoder") {
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t channels = 512;
    constexpr uint32_t conv_kernel = 3;
    constexpr uint32_t conv_stride = 2;
    constexpr uint32_t conv_freq = 511;
    constexpr uint32_t flat_dim = channels * conv_freq;
    constexpr uint32_t output_dim = 512;
    if (input_tokens < conv_kernel) {
        throw std::runtime_error("GPT subsampling input_tokens must be at least 3");
    }
    if (input.size() != static_cast<size_t>(input_tokens) * input_dim) {
        throw std::runtime_error("GPT subsampling input must have shape [tokens,1024]");
    }
    const uint32_t output_tokens = ((input_tokens - conv_kernel) / conv_stride) + 1;

    auto conv_w = tensor_as_f32(bundle, encoder_prefix + ".embed.conv.0.weight");
    auto conv_b = tensor_as_f32(bundle, encoder_prefix + ".embed.conv.0.bias");
    auto linear_w = tensor_as_f32(bundle, encoder_prefix + ".embed.out.0.weight");
    auto linear_b = tensor_as_f32(bundle, encoder_prefix + ".embed.out.0.bias");
    auto pe = tensor_as_f32(bundle, encoder_prefix + ".embed.pos_enc.pe");
    if (conv_w.size() != static_cast<size_t>(channels) * conv_kernel * conv_kernel || conv_b.size() != channels) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.conv.0 shape mismatch");
    }
    if (linear_w.size() != static_cast<size_t>(output_dim) * flat_dim || linear_b.size() != output_dim) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.out.0 shape mismatch");
    }
    if (pe.size() < static_cast<size_t>(output_tokens) * output_dim) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.pos_enc.pe shape mismatch");
    }

    auto flat = metal.subsampling_conv2d_relu_flat_f32_resident(
        encoder_prefix + ".embed.conv.0.weight.resident",
        conv_w,
        encoder_prefix + ".embed.conv.0.bias.resident",
        conv_b,
        input,
        input_tokens,
        input_dim,
        channels,
        conv_kernel,
        conv_stride);

    auto subsampling = metal.linear_rows_f32_resident(
        encoder_prefix + ".embed.out.0.weight.resident",
        linear_w,
        encoder_prefix + ".embed.out.0.bias.resident",
        linear_b,
        flat,
        output_tokens,
        output_dim,
        flat_dim);
    const float positional_scale = std::sqrt(static_cast<float>(output_dim));
    for (float& v : subsampling) {
        v *= positional_scale;
    }
    std::vector<float> pos_emb(static_cast<size_t>(output_tokens) * output_dim);
    std::copy(pe.begin(), pe.begin() + pos_emb.size(), pos_emb.begin());
    std::vector<uint32_t> mask(output_tokens, 1);
    return GptSubsamplingOutput{std::move(subsampling), std::move(pos_emb), std::move(mask), output_tokens};
}

bool run_gpt_subsampling_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t output_dim = 512;
    auto input = read_raw_f32(golden_dir + "/spk_cond_emb.f32");
    auto golden = read_raw_f32(golden_dir + "/subsampling.f32");
    auto golden_pos = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/subsampling_mask.u32");
    if (input.empty() || (input.size() % input_dim) != 0) {
        throw std::runtime_error("GPT subsampling golden input must have shape [tokens,1024]");
    }
    if (golden.empty() || (golden.size() % output_dim) != 0) {
        throw std::runtime_error("GPT subsampling golden output must have shape [tokens,512]");
    }
    const uint32_t input_tokens = static_cast<uint32_t>(input.size() / input_dim);
    const uint32_t output_tokens = static_cast<uint32_t>(golden.size() / output_dim);
    auto got = run_gpt_conditioning_subsampling_cpu(bundle, input, input_tokens);
    if (got.output_tokens != output_tokens || golden_pos.size() != got.pos_emb.size() || golden_mask.size() != got.mask.size()) {
        throw std::runtime_error("GPT subsampling golden shape mismatch");
    }
    const float subsampling_err = max_abs_error(got.subsampling, golden);
    const float pos_err = max_abs_error(got.pos_emb, golden_pos);
    bool mask_match = true;
    for (size_t i = 0; i < got.mask.size(); ++i) {
        mask_match = mask_match && got.mask[i] == golden_mask[i];
    }
    const float err = std::max(subsampling_err, pos_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_subsampling_golden\",\n";
    std::cout << "  \"input_tokens\": " << input_tokens << ",\n";
    std::cout << "  \"input_dim\": " << input_dim << ",\n";
    std::cout << "  \"output_tokens\": " << output_tokens << ",\n";
    std::cout << "  \"output_dim\": " << output_dim << ",\n";
    std::cout << "  \"subsampling_max_abs_error\": " << subsampling_err << ",\n";
    std::cout << "  \"pos_emb_max_abs_error\": " << pos_err << ",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got.subsampling) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return subsampling_err <= 2e-2f && pos_err <= 1e-6f && mask_match && count_nonfinite(got.subsampling) == 0;
}

bool run_gpt_subsampling_metal_linear_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t output_dim = 512;
    auto input = read_raw_f32(golden_dir + "/spk_cond_emb.f32");
    auto golden = read_raw_f32(golden_dir + "/subsampling.f32");
    auto golden_pos = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/subsampling_mask.u32");
    if (input.empty() || (input.size() % input_dim) != 0) {
        throw std::runtime_error("GPT subsampling golden input must have shape [tokens,1024]");
    }
    if (golden.empty() || (golden.size() % output_dim) != 0) {
        throw std::runtime_error("GPT subsampling golden output must have shape [tokens,512]");
    }
    const uint32_t input_tokens = static_cast<uint32_t>(input.size() / input_dim);
    const uint32_t output_tokens = static_cast<uint32_t>(golden.size() / output_dim);
    auto got = run_gpt_conditioning_subsampling_metal_linear(metal, bundle, input, input_tokens);
    if (got.output_tokens != output_tokens || golden_pos.size() != got.pos_emb.size() || golden_mask.size() != got.mask.size()) {
        throw std::runtime_error("GPT subsampling metal-linear golden shape mismatch");
    }
    const float subsampling_err = max_abs_error(got.subsampling, golden);
    const float pos_err = max_abs_error(got.pos_emb, golden_pos);
    bool mask_match = true;
    for (size_t i = 0; i < got.mask.size(); ++i) {
        mask_match = mask_match && got.mask[i] == golden_mask[i];
    }
    const float err = std::max(subsampling_err, pos_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_subsampling_metal_linear_golden\",\n";
    std::cout << "  \"input_tokens\": " << input_tokens << ",\n";
    std::cout << "  \"input_dim\": " << input_dim << ",\n";
    std::cout << "  \"output_tokens\": " << output_tokens << ",\n";
    std::cout << "  \"output_dim\": " << output_dim << ",\n";
    std::cout << "  \"conv_source\": \"metal_subsampling_conv2d_relu_flat_f32_resident\",\n";
    std::cout << "  \"projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"subsampling_max_abs_error\": " << subsampling_err << ",\n";
    std::cout << "  \"pos_emb_max_abs_error\": " << pos_err << ",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got.subsampling) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return subsampling_err <= 2e-2f && pos_err <= 1e-6f && mask_match && count_nonfinite(got.subsampling) == 0;
}

bool export_gpt_subsampling(const std::string& bundle_dir,
                            const std::string& input_path,
                            const std::string& output_stack_path,
                            const std::string& output_pos_emb_path,
                            const std::string& output_mask_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t output_dim = 512;
    auto input = read_raw_f32(input_path);
    if (input.empty() || (input.size() % input_dim) != 0) {
        throw std::runtime_error("GPT subsampling export input must have shape [tokens,1024]");
    }
    const uint32_t input_tokens = static_cast<uint32_t>(input.size() / input_dim);
    auto out = run_gpt_conditioning_subsampling_metal_linear(metal, bundle, input, input_tokens);
    write_raw_f32(output_stack_path, out.subsampling);
    write_raw_f32(output_pos_emb_path, out.pos_emb);
    write_raw_u32(output_mask_path, out.mask);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_subsampling_export\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"input_f32\": \"" << json_escape(input_path) << "\",\n";
    std::cout << "  \"output_stack_f32\": \"" << json_escape(output_stack_path) << "\",\n";
    std::cout << "  \"output_pos_emb_f32\": \"" << json_escape(output_pos_emb_path) << "\",\n";
    std::cout << "  \"output_mask_u32\": \"" << json_escape(output_mask_path) << "\",\n";
    std::cout << "  \"input_tokens\": " << input_tokens << ",\n";
    std::cout << "  \"output_tokens\": " << out.output_tokens << ",\n";
    std::cout << "  \"output_dim\": " << output_dim << ",\n";
    std::cout << "  \"conv_source\": \"metal_subsampling_conv2d_relu_flat_f32_resident\",\n";
    std::cout << "  \"projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(out.subsampling) << "\n";
    std::cout << "}\n";
    return count_nonfinite(out.subsampling) == 0;
}

std::vector<float> run_gpt_conformer_stack_cpu(const mit2::Bundle& bundle,
                                               const std::vector<float>& input,
                                               const std::vector<float>& pos_emb,
                                               const std::vector<uint32_t>& mask,
                                               uint32_t tokens,
                                               uint32_t layers,
                                               const std::string& encoder_prefix,
                                               uint32_t heads);

std::vector<float> run_gpt_conformer_stack_metal_attn_conv_ff(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layers,
                                                              const std::string& encoder_prefix,
                                                              uint32_t heads);

std::vector<float> finish_gpt_emovec_from_subsampling_cpu(const mit2::Bundle& bundle,
                                                          const GptSubsamplingOutput& sub) {
    constexpr uint32_t conformer_dim = 512;
    constexpr uint32_t emo_dim = 1024;
    constexpr uint32_t model_dim = 1280;
    auto context = run_gpt_conformer_stack_cpu(bundle,
                                               sub.subsampling,
                                               sub.pos_emb,
                                               sub.mask,
                                               sub.output_tokens,
                                               4,
                                               "gpt.emo_conditioning_encoder",
                                               4);
    std::vector<uint32_t> perceiver_mask;
    perceiver_mask.reserve(static_cast<size_t>(sub.output_tokens) + 1);
    perceiver_mask.push_back(1);
    perceiver_mask.insert(perceiver_mask.end(), sub.mask.begin(), sub.mask.end());
    auto emo_perceiver = run_gpt_perceiver_cpu(bundle,
                                               context,
                                               perceiver_mask,
                                               sub.output_tokens,
                                               "gpt.emo_perceiver_encoder",
                                               conformer_dim,
                                               emo_dim,
                                               1,
                                               4);
    auto emovec_w = tensor_as_f32(bundle, "gpt.emovec_layer.weight");
    auto emovec_b = tensor_as_f32(bundle, "gpt.emovec_layer.bias");
    auto emo_w = tensor_as_f32(bundle, "gpt.emo_layer.weight");
    auto emo_b = tensor_as_f32(bundle, "gpt.emo_layer.bias");
    auto projected = cpu_linear_rows(emovec_w, emovec_b, emo_perceiver, 1, model_dim, emo_dim);
    return cpu_linear_rows(emo_w, emo_b, projected, 1, model_dim, model_dim);
}

std::vector<float> finish_gpt_emovec_from_subsampling_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& bundle,
                                                            const GptSubsamplingOutput& sub) {
    constexpr uint32_t conformer_dim = 512;
    constexpr uint32_t emo_dim = 1024;
    constexpr uint32_t model_dim = 1280;
    auto context = run_gpt_conformer_stack_metal_attn_conv_ff(metal,
                                                              bundle,
                                                              sub.subsampling,
                                                              sub.pos_emb,
                                                              sub.mask,
                                                              sub.output_tokens,
                                                              4,
                                                              "gpt.emo_conditioning_encoder",
                                                              4);
    std::vector<uint32_t> perceiver_mask;
    perceiver_mask.reserve(static_cast<size_t>(sub.output_tokens) + 1);
    perceiver_mask.push_back(1);
    perceiver_mask.insert(perceiver_mask.end(), sub.mask.begin(), sub.mask.end());
    auto emo_perceiver = run_gpt_perceiver_metal(metal,
                                                 bundle,
                                                 context,
                                                 perceiver_mask,
                                                 sub.output_tokens,
                                                 "gpt.emo_perceiver_encoder",
                                                 conformer_dim,
                                                 emo_dim,
                                                 1,
                                                 4);
    auto emovec_w = tensor_as_f32(bundle, "gpt.emovec_layer.weight");
    auto emovec_b = tensor_as_f32(bundle, "gpt.emovec_layer.bias");
    auto emo_w = tensor_as_f32(bundle, "gpt.emo_layer.weight");
    auto emo_b = tensor_as_f32(bundle, "gpt.emo_layer.bias");
    auto projected = metal.linear_rows_f32_resident(
        "gpt.emovec_layer.weight.resident",
        emovec_w,
        "gpt.emovec_layer.bias.resident",
        emovec_b,
        emo_perceiver,
        1,
        model_dim,
        emo_dim);
    return metal.linear_rows_f32_resident(
        "gpt.emo_layer.weight.resident",
        emo_w,
        "gpt.emo_layer.bias.resident",
        emo_b,
        projected,
        1,
        model_dim,
        model_dim);
}

std::vector<float> run_gpt_emovec_cpu(const mit2::Bundle& bundle,
                                      const std::vector<float>& spk_cond_emb,
                                      uint32_t input_tokens) {
    constexpr uint32_t input_dim = 1024;
    if (spk_cond_emb.size() != static_cast<size_t>(input_tokens) * input_dim) {
        throw std::runtime_error("GPT emovec input must have shape [tokens,1024]");
    }
    auto sub = run_gpt_conditioning_subsampling_cpu(bundle, spk_cond_emb, input_tokens, "gpt.emo_conditioning_encoder");
    return finish_gpt_emovec_from_subsampling_cpu(bundle, sub);
}

std::vector<float> run_gpt_emovec_metal_linear(mit2::MetalContext& metal,
                                               const mit2::Bundle& bundle,
                                               const std::vector<float>& spk_cond_emb,
                                               uint32_t input_tokens) {
    constexpr uint32_t input_dim = 1024;
    if (spk_cond_emb.size() != static_cast<size_t>(input_tokens) * input_dim) {
        throw std::runtime_error("GPT emovec input must have shape [tokens,1024]");
    }
    auto sub = run_gpt_conditioning_subsampling_metal_linear(metal, bundle, spk_cond_emb, input_tokens, "gpt.emo_conditioning_encoder");
    return finish_gpt_emovec_from_subsampling_metal(metal, bundle, sub);
}

bool run_gpt_emovec_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t model_dim = 1280;
    auto input = read_raw_f32(golden_dir + "/spk_cond_emb.f32");
    auto golden = read_raw_f32(golden_dir + "/emovec.f32");
    if (input.empty() || (input.size() % input_dim) != 0) {
        throw std::runtime_error("GPT emovec golden input must have shape [tokens,1024]");
    }
    if (golden.size() != model_dim) {
        throw std::runtime_error("GPT emovec golden output must have shape [1280]");
    }
    const uint32_t input_tokens = static_cast<uint32_t>(input.size() / input_dim);
    auto got = run_gpt_emovec_cpu(bundle, input, input_tokens);
    const float err = max_abs_error(got, golden);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_emovec_golden\",\n";
    std::cout << "  \"input_tokens\": " << input_tokens << ",\n";
    std::cout << "  \"input_dim\": " << input_dim << ",\n";
    std::cout << "  \"output_dim\": " << model_dim << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-2f && count_nonfinite(got) == 0;
}

bool run_gpt_emovec_metal_linear_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t model_dim = 1280;
    auto input = read_raw_f32(golden_dir + "/spk_cond_emb.f32");
    auto golden = read_raw_f32(golden_dir + "/emovec.f32");
    if (input.empty() || (input.size() % input_dim) != 0) {
        throw std::runtime_error("GPT emovec golden input must have shape [tokens,1024]");
    }
    if (golden.size() != model_dim) {
        throw std::runtime_error("GPT emovec golden output must have shape [1280]");
    }
    const uint32_t input_tokens = static_cast<uint32_t>(input.size() / input_dim);
    auto got = run_gpt_emovec_metal_linear(metal, bundle, input, input_tokens);
    const float err = max_abs_error(got, golden);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_emovec_metal_linear_golden\",\n";
    std::cout << "  \"input_tokens\": " << input_tokens << ",\n";
    std::cout << "  \"input_dim\": " << input_dim << ",\n";
    std::cout << "  \"output_dim\": " << model_dim << ",\n";
    std::cout << "  \"subsampling_conv_source\": \"metal_subsampling_conv2d_relu_flat_f32_resident\",\n";
    std::cout << "  \"subsampling_projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"conformer_source\": \"metal_resident_attn_core_conv_ff\",\n";
    std::cout << "  \"perceiver_source\": \"metal_resident_linear_cross_attn_geglu_rmsnorm\",\n";
    std::cout << "  \"projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-2f && count_nonfinite(got) == 0;
}

bool export_gpt_emovec(const std::string& bundle_dir,
                       const std::string& input_path,
                       const std::string& output_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t input_dim = 1024;
    auto input = read_raw_f32(input_path);
    if (input.empty() || (input.size() % input_dim) != 0) {
        throw std::runtime_error("GPT emovec export input must have shape [tokens,1024]");
    }
    const uint32_t input_tokens = static_cast<uint32_t>(input.size() / input_dim);
    auto out = run_gpt_emovec_metal_linear(metal, bundle, input, input_tokens);
    write_raw_f32(output_path, out);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_emovec_export\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"input_f32\": \"" << json_escape(input_path) << "\",\n";
    std::cout << "  \"output_emovec_f32\": \"" << json_escape(output_path) << "\",\n";
    std::cout << "  \"input_tokens\": " << input_tokens << ",\n";
    std::cout << "  \"output_dim\": 1280,\n";
    std::cout << "  \"subsampling_conv_source\": \"metal_subsampling_conv2d_relu_flat_f32_resident\",\n";
    std::cout << "  \"subsampling_projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"conformer_source\": \"metal_resident_attn_core_conv_ff\",\n";
    std::cout << "  \"perceiver_source\": \"metal_resident_linear_cross_attn_geglu_rmsnorm\",\n";
    std::cout << "  \"projection_source\": \"metal_linear_rows_f32_resident\",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(out) << "\n";
    std::cout << "}\n";
    return count_nonfinite(out) == 0;
}

struct GptConformerFfOutput {
    std::vector<float> normed;
    std::vector<float> raw;
    std::vector<float> tail;
};

GptConformerFfOutput run_gpt_conformer_ff_tail_cpu(const mit2::Bundle& bundle,
                                                   const std::vector<float>& input,
                                                   uint32_t tokens,
                                                   uint32_t layer_index = 0,
                                                   const std::string& encoder_prefix = "gpt.conditioning_encoder") {
    constexpr uint32_t dim = 512;
    constexpr float ff_scale = 1.0f;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer FF input must have shape [tokens,512]");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    auto norm_w = tensor_as_f32(bundle, base + ".norm_ff.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_ff.bias");
    auto w1 = tensor_as_f32(bundle, base + ".feed_forward.w_1.weight");
    auto b1 = tensor_as_f32(bundle, base + ".feed_forward.w_1.bias");
    auto w2 = tensor_as_f32(bundle, base + ".feed_forward.w_2.weight");
    auto b2 = tensor_as_f32(bundle, base + ".feed_forward.w_2.bias");
    auto final_norm_w = tensor_as_f32(bundle, base + ".norm_final.weight");
    auto final_norm_b = tensor_as_f32(bundle, base + ".norm_final.bias");
    const uint32_t hidden = static_cast<uint32_t>(b1.size());
    if (hidden == 0 ||
        w1.size() != static_cast<size_t>(hidden) * dim ||
        w2.size() != static_cast<size_t>(dim) * hidden ||
        b2.size() != dim) {
        throw std::runtime_error("GPT conformer FF weight shape mismatch");
    }

    auto normed = cpu_layer_norm_rows(input, norm_w, norm_b, tokens, dim);
    auto hidden_pre = cpu_linear_rows(w1, b1, normed, tokens, hidden, dim);
    auto hidden_act = cpu_silu(hidden_pre);
    auto raw = cpu_linear_rows(w2, b2, hidden_act, tokens, dim, hidden);
    std::vector<float> residual(raw.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = input[i] + ff_scale * raw[i];
    }
    auto tail = cpu_layer_norm_rows(residual, final_norm_w, final_norm_b, tokens, dim);
    return GptConformerFfOutput{std::move(normed), std::move(raw), std::move(tail)};
}

GptConformerFfOutput run_gpt_conformer_ff_tail_metal(mit2::MetalContext& metal,
                                                     const mit2::Bundle& bundle,
                                                     const std::vector<float>& input,
                                                     uint32_t tokens,
                                                     uint32_t layer_index = 0,
                                                     const std::string& encoder_prefix = "gpt.conditioning_encoder") {
    constexpr uint32_t dim = 512;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer FF input must have shape [tokens,512]");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    auto norm_w = tensor_as_f32(bundle, base + ".norm_ff.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_ff.bias");
    auto w1 = tensor_as_f32(bundle, base + ".feed_forward.w_1.weight");
    auto b1 = tensor_as_f32(bundle, base + ".feed_forward.w_1.bias");
    auto w2 = tensor_as_f32(bundle, base + ".feed_forward.w_2.weight");
    auto b2 = tensor_as_f32(bundle, base + ".feed_forward.w_2.bias");
    auto final_norm_w = tensor_as_f32(bundle, base + ".norm_final.weight");
    auto final_norm_b = tensor_as_f32(bundle, base + ".norm_final.bias");
    const uint32_t hidden = static_cast<uint32_t>(b1.size());
    if (hidden == 0 ||
        w1.size() != static_cast<size_t>(hidden) * dim ||
        w2.size() != static_cast<size_t>(dim) * hidden ||
        b2.size() != dim) {
        throw std::runtime_error("GPT conformer FF weight shape mismatch");
    }

    auto normed = metal.layernorm_rows_f32_resident(
        base + ".norm_ff.weight.resident",
        norm_w,
        base + ".norm_ff.bias.resident",
        norm_b,
        input,
        tokens,
        dim,
        1e-5f);
    auto hidden_pre = metal.linear_rows_f32_resident(
        base + ".feed_forward.w_1.weight.resident",
        w1,
        base + ".feed_forward.w_1.bias.resident",
        b1,
        normed,
        tokens,
        hidden,
        dim);
    auto hidden_act = metal.silu_f32(hidden_pre);
    auto raw = metal.linear_rows_f32_resident(
        base + ".feed_forward.w_2.weight.resident",
        w2,
        base + ".feed_forward.w_2.bias.resident",
        b2,
        hidden_act,
        tokens,
        dim,
        hidden);
    auto residual = metal.add_f32(input, raw);
    auto tail = metal.layernorm_rows_f32_resident(
        base + ".norm_final.weight.resident",
        final_norm_w,
        base + ".norm_final.bias.resident",
        final_norm_b,
        residual,
        tokens,
        dim,
        1e-5f);
    return GptConformerFfOutput{std::move(normed), std::move(raw), std::move(tail)};
}

bool run_gpt_conformer_ff_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/ff_input.f32");
    auto golden_normed = read_raw_f32(golden_dir + "/ff_normed.f32");
    auto golden_raw = read_raw_f32(golden_dir + "/ff_raw.f32");
    auto golden_tail = read_raw_f32(golden_dir + "/ff_tail.f32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer FF golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    auto got = run_gpt_conformer_ff_tail_cpu(bundle, input, tokens);
    if (golden_normed.size() != got.normed.size() || golden_raw.size() != got.raw.size() || golden_tail.size() != got.tail.size()) {
        throw std::runtime_error("GPT conformer FF golden shape mismatch");
    }
    const float normed_err = max_abs_error(got.normed, golden_normed);
    const float raw_err = max_abs_error(got.raw, golden_raw);
    const float tail_err = max_abs_error(got.tail, golden_tail);
    const float err = std::max({normed_err, raw_err, tail_err});
    const size_t nonfinite = count_nonfinite(got.normed) + count_nonfinite(got.raw) + count_nonfinite(got.tail);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_ff_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"hidden_units\": 2048,\n";
    std::cout << "  \"ff_scale\": 1,\n";
    std::cout << "  \"normed_max_abs_error\": " << normed_err << ",\n";
    std::cout << "  \"raw_max_abs_error\": " << raw_err << ",\n";
    std::cout << "  \"tail_max_abs_error\": " << tail_err << ",\n";
    std::cout << "  \"nonfinite\": " << nonfinite << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return normed_err <= 1e-3f && raw_err <= 2e-2f && tail_err <= 1e-3f && nonfinite == 0;
}

bool run_gpt_conformer_ff_metal_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/ff_input.f32");
    auto golden_normed = read_raw_f32(golden_dir + "/ff_normed.f32");
    auto golden_raw = read_raw_f32(golden_dir + "/ff_raw.f32");
    auto golden_tail = read_raw_f32(golden_dir + "/ff_tail.f32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer FF golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    auto got = run_gpt_conformer_ff_tail_metal(metal, bundle, input, tokens);
    if (golden_normed.size() != got.normed.size() || golden_raw.size() != got.raw.size() || golden_tail.size() != got.tail.size()) {
        throw std::runtime_error("GPT conformer FF metal golden shape mismatch");
    }
    const float normed_err = max_abs_error(got.normed, golden_normed);
    const float raw_err = max_abs_error(got.raw, golden_raw);
    const float tail_err = max_abs_error(got.tail, golden_tail);
    const float err = std::max({normed_err, raw_err, tail_err});
    const size_t nonfinite = count_nonfinite(got.normed) + count_nonfinite(got.raw) + count_nonfinite(got.tail);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_ff_metal_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"kernel_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"normed_max_abs_error\": " << normed_err << ",\n";
    std::cout << "  \"raw_max_abs_error\": " << raw_err << ",\n";
    std::cout << "  \"tail_max_abs_error\": " << tail_err << ",\n";
    std::cout << "  \"nonfinite\": " << nonfinite << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return normed_err <= 1e-3f && raw_err <= 2e-2f && tail_err <= 1e-3f && nonfinite == 0;
}

struct GptConformerAttnOutput {
    std::vector<float> normed;
    std::vector<float> attn;
    std::vector<float> residual;
};

GptConformerAttnOutput run_gpt_conformer_attention_cpu(const mit2::Bundle& bundle,
                                                       const std::vector<float>& input,
                                                       const std::vector<float>& pos_emb,
                                                       const std::vector<uint32_t>& mask,
                                                       uint32_t tokens,
                                                       uint32_t layer_index = 0,
                                                       const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                       uint32_t heads = 8) {
    constexpr uint32_t dim = 512;
    const uint32_t head_dim = dim / heads;
    if (heads == 0 || dim % heads != 0) {
        throw std::runtime_error("GPT conformer attention heads must divide dim");
    }
    if (input.size() != static_cast<size_t>(tokens) * dim ||
        pos_emb.size() != static_cast<size_t>(tokens) * dim ||
        mask.size() != tokens) {
        throw std::runtime_error("GPT conformer attention input shape mismatch");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    auto norm_w = tensor_as_f32(bundle, base + ".norm_mha.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_mha.bias");
    auto q_w = tensor_as_f32(bundle, base + ".self_attn.linear_q.weight");
    auto q_b = tensor_as_f32(bundle, base + ".self_attn.linear_q.bias");
    auto k_w = tensor_as_f32(bundle, base + ".self_attn.linear_k.weight");
    auto k_b = tensor_as_f32(bundle, base + ".self_attn.linear_k.bias");
    auto v_w = tensor_as_f32(bundle, base + ".self_attn.linear_v.weight");
    auto v_b = tensor_as_f32(bundle, base + ".self_attn.linear_v.bias");
    auto pos_w = tensor_as_f32(bundle, base + ".self_attn.linear_pos.weight");
    auto out_w = tensor_as_f32(bundle, base + ".self_attn.linear_out.weight");
    auto out_b = tensor_as_f32(bundle, base + ".self_attn.linear_out.bias");
    auto bias_u = tensor_as_f32(bundle, base + ".self_attn.pos_bias_u");
    auto bias_v = tensor_as_f32(bundle, base + ".self_attn.pos_bias_v");
    if (bias_u.size() != static_cast<size_t>(heads) * head_dim ||
        bias_v.size() != static_cast<size_t>(heads) * head_dim) {
        throw std::runtime_error("GPT conformer attention bias shape mismatch");
    }

    auto normed = cpu_layer_norm_rows(input, norm_w, norm_b, tokens, dim);
    auto q_linear = cpu_linear_rows(q_w, q_b, normed, tokens, dim, dim);
    auto k_linear = cpu_linear_rows(k_w, k_b, normed, tokens, dim, dim);
    auto v_linear = cpu_linear_rows(v_w, v_b, normed, tokens, dim, dim);
    std::vector<float> zero_dim_bias(dim, 0.0f);
    auto p_linear = cpu_linear_rows(pos_w, zero_dim_bias, pos_emb, tokens, dim, dim);

    auto attn_context = cpu_conformer_rel_attention_context(q_linear, k_linear, v_linear, p_linear, bias_u, bias_v, mask, tokens, heads, head_dim);
    auto attn = cpu_linear_rows(out_w, out_b, attn_context, tokens, dim, dim);
    std::vector<float> residual(attn.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = input[i] + attn[i];
    }
    return GptConformerAttnOutput{std::move(normed), std::move(attn), std::move(residual)};
}

GptConformerAttnOutput run_gpt_conformer_attention_metal_proj(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layer_index = 0,
                                                              const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                              uint32_t heads = 8) {
    constexpr uint32_t dim = 512;
    const uint32_t head_dim = dim / heads;
    if (heads == 0 || dim % heads != 0) {
        throw std::runtime_error("GPT conformer attention heads must divide dim");
    }
    if (input.size() != static_cast<size_t>(tokens) * dim ||
        pos_emb.size() != static_cast<size_t>(tokens) * dim ||
        mask.size() != tokens) {
        throw std::runtime_error("GPT conformer attention input shape mismatch");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    auto norm_w = tensor_as_f32(bundle, base + ".norm_mha.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_mha.bias");
    auto q_w = tensor_as_f32(bundle, base + ".self_attn.linear_q.weight");
    auto q_b = tensor_as_f32(bundle, base + ".self_attn.linear_q.bias");
    auto k_w = tensor_as_f32(bundle, base + ".self_attn.linear_k.weight");
    auto k_b = tensor_as_f32(bundle, base + ".self_attn.linear_k.bias");
    auto v_w = tensor_as_f32(bundle, base + ".self_attn.linear_v.weight");
    auto v_b = tensor_as_f32(bundle, base + ".self_attn.linear_v.bias");
    auto pos_w = tensor_as_f32(bundle, base + ".self_attn.linear_pos.weight");
    auto out_w = tensor_as_f32(bundle, base + ".self_attn.linear_out.weight");
    auto out_b = tensor_as_f32(bundle, base + ".self_attn.linear_out.bias");
    auto bias_u = tensor_as_f32(bundle, base + ".self_attn.pos_bias_u");
    auto bias_v = tensor_as_f32(bundle, base + ".self_attn.pos_bias_v");
    if (bias_u.size() != static_cast<size_t>(heads) * head_dim ||
        bias_v.size() != static_cast<size_t>(heads) * head_dim) {
        throw std::runtime_error("GPT conformer attention bias shape mismatch");
    }

    auto normed = metal.layernorm_rows_f32_resident(
        base + ".norm_mha.weight.resident",
        norm_w,
        base + ".norm_mha.bias.resident",
        norm_b,
        input,
        tokens,
        dim,
        1e-5f);
    auto q_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_q.weight.resident",
        q_w,
        base + ".self_attn.linear_q.bias.resident",
        q_b,
        normed,
        tokens,
        dim,
        dim);
    auto k_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_k.weight.resident",
        k_w,
        base + ".self_attn.linear_k.bias.resident",
        k_b,
        normed,
        tokens,
        dim,
        dim);
    auto v_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_v.weight.resident",
        v_w,
        base + ".self_attn.linear_v.bias.resident",
        v_b,
        normed,
        tokens,
        dim,
        dim);
    std::vector<float> zero_dim_bias(dim, 0.0f);
    auto p_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_pos.weight.resident",
        pos_w,
        base + ".self_attn.linear_pos.zero_bias.resident",
        zero_dim_bias,
        pos_emb,
        tokens,
        dim,
        dim);

    auto attn_context = metal.conformer_rel_attention_context_f32_resident(
        base + ".self_attn.pos_bias_u.resident",
        bias_u,
        base + ".self_attn.pos_bias_v.resident",
        bias_v,
        q_linear,
        k_linear,
        v_linear,
        p_linear,
        mask,
        tokens,
        heads,
        head_dim);
    auto attn = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_out.weight.resident",
        out_w,
        base + ".self_attn.linear_out.bias.resident",
        out_b,
        attn_context,
        tokens,
        dim,
        dim);
    auto residual = metal.add_f32(input, attn);
    return GptConformerAttnOutput{std::move(normed), std::move(attn), std::move(residual)};
}

bool run_gpt_conformer_attn_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/attn_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/attn_mask.u32");
    auto golden_normed = read_raw_f32(golden_dir + "/attn_normed.f32");
    auto golden_attn = read_raw_f32(golden_dir + "/attn_out.f32");
    auto golden_residual = read_raw_f32(golden_dir + "/attn_residual.f32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer attention golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    auto got = run_gpt_conformer_attention_cpu(bundle, input, pos_emb, mask, tokens);
    if (golden_normed.size() != got.normed.size() ||
        golden_attn.size() != got.attn.size() ||
        golden_residual.size() != got.residual.size()) {
        throw std::runtime_error("GPT conformer attention golden shape mismatch");
    }
    const float normed_err = max_abs_error(got.normed, golden_normed);
    const float attn_err = max_abs_error(got.attn, golden_attn);
    const float residual_err = max_abs_error(got.residual, golden_residual);
    const float err = std::max({normed_err, attn_err, residual_err});
    const size_t nonfinite = count_nonfinite(got.normed) + count_nonfinite(got.attn) + count_nonfinite(got.residual);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_attn_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"heads\": 8,\n";
    std::cout << "  \"head_dim\": 64,\n";
    std::cout << "  \"normed_max_abs_error\": " << normed_err << ",\n";
    std::cout << "  \"attn_max_abs_error\": " << attn_err << ",\n";
    std::cout << "  \"residual_max_abs_error\": " << residual_err << ",\n";
    std::cout << "  \"nonfinite\": " << nonfinite << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return normed_err <= 1e-3f && attn_err <= 2e-3f && residual_err <= 2e-3f && nonfinite == 0;
}

bool run_gpt_conformer_attn_metal_proj_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/attn_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/attn_mask.u32");
    auto golden_normed = read_raw_f32(golden_dir + "/attn_normed.f32");
    auto golden_attn = read_raw_f32(golden_dir + "/attn_out.f32");
    auto golden_residual = read_raw_f32(golden_dir + "/attn_residual.f32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer attention golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    auto got = run_gpt_conformer_attention_metal_proj(metal, bundle, input, pos_emb, mask, tokens);
    if (golden_normed.size() != got.normed.size() ||
        golden_attn.size() != got.attn.size() ||
        golden_residual.size() != got.residual.size()) {
        throw std::runtime_error("GPT conformer attention metal-proj golden shape mismatch");
    }
    const float normed_err = max_abs_error(got.normed, golden_normed);
    const float attn_err = max_abs_error(got.attn, golden_attn);
    const float residual_err = max_abs_error(got.residual, golden_residual);
    const float err = std::max({normed_err, attn_err, residual_err});
    const size_t nonfinite = count_nonfinite(got.normed) + count_nonfinite(got.attn) + count_nonfinite(got.residual);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_attn_metal_proj_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"heads\": 8,\n";
    std::cout << "  \"head_dim\": 64,\n";
    std::cout << "  \"projection_source\": \"metal_resident_layernorm_rows_linear_rows_add\",\n";
    std::cout << "  \"attention_core_source\": \"metal_resident_bias_relative_attention_softmax\",\n";
    std::cout << "  \"normed_max_abs_error\": " << normed_err << ",\n";
    std::cout << "  \"attn_max_abs_error\": " << attn_err << ",\n";
    std::cout << "  \"residual_max_abs_error\": " << residual_err << ",\n";
    std::cout << "  \"nonfinite\": " << nonfinite << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return normed_err <= 1e-3f && attn_err <= 2e-3f && residual_err <= 2e-3f && nonfinite == 0;
}

struct GptConformerConvOutput {
    std::vector<float> normed;
    std::vector<float> raw;
    std::vector<float> residual;
};

GptConformerConvOutput run_gpt_conformer_conv_cpu(const mit2::Bundle& bundle,
                                                  const std::vector<float>& input,
                                                  const std::vector<uint32_t>& mask,
                                                  uint32_t tokens,
                                                  uint32_t layer_index = 0,
                                                  const std::string& encoder_prefix = "gpt.conditioning_encoder") {
    constexpr uint32_t dim = 512;
    constexpr uint32_t doubled = 1024;
    constexpr uint32_t kernel = 15;
    constexpr uint32_t pad = 7;
    if (input.size() != static_cast<size_t>(tokens) * dim || mask.size() != tokens) {
        throw std::runtime_error("GPT conformer conv input shape mismatch");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    const std::string conv = base + ".conv_module";
    auto norm_w = tensor_as_f32(bundle, base + ".norm_conv.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_conv.bias");
    auto pw1_w = tensor_as_f32(bundle, conv + ".pointwise_conv1.weight");
    auto pw1_b = tensor_as_f32(bundle, conv + ".pointwise_conv1.bias");
    auto depth_w = tensor_as_f32(bundle, conv + ".depthwise_conv.weight");
    auto depth_b = tensor_as_f32(bundle, conv + ".depthwise_conv.bias");
    auto conv_norm_w = tensor_as_f32(bundle, conv + ".norm.weight");
    auto conv_norm_b = tensor_as_f32(bundle, conv + ".norm.bias");
    auto pw2_w = tensor_as_f32(bundle, conv + ".pointwise_conv2.weight");
    auto pw2_b = tensor_as_f32(bundle, conv + ".pointwise_conv2.bias");
    if (pw1_w.size() != static_cast<size_t>(doubled) * dim ||
        pw2_w.size() != static_cast<size_t>(dim) * dim ||
        depth_w.size() != static_cast<size_t>(dim) * kernel ||
        depth_b.size() != dim) {
        throw std::runtime_error("GPT conformer conv weight shape mismatch");
    }

    auto normed = cpu_layer_norm_rows(input, norm_w, norm_b, tokens, dim);
    std::vector<float> masked_normed = normed;
    for (uint32_t t = 0; t < tokens; ++t) {
        if (mask[t] == 0) {
            std::fill(masked_normed.begin() + static_cast<size_t>(t) * dim,
                      masked_normed.begin() + static_cast<size_t>(t + 1) * dim,
                      0.0f);
        }
    }

    auto pw1 = cpu_linear_rows(pw1_w, pw1_b, masked_normed, tokens, doubled, dim);
    std::vector<float> glu(static_cast<size_t>(tokens) * dim);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < dim; ++c) {
            const size_t base_idx = static_cast<size_t>(t) * doubled;
            const float a = pw1[base_idx + c];
            const float b = pw1[base_idx + dim + c];
            glu[static_cast<size_t>(t) * dim + c] = a / (1.0f + std::exp(-b));
        }
    }

    std::vector<float> depth(static_cast<size_t>(tokens) * dim);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < dim; ++c) {
            float acc = depth_b[c];
            for (uint32_t k = 0; k < kernel; ++k) {
                const int src_t = static_cast<int>(t) + static_cast<int>(k) - static_cast<int>(pad);
                if (src_t < 0 || src_t >= static_cast<int>(tokens)) {
                    continue;
                }
                acc += glu[static_cast<size_t>(src_t) * dim + c] * depth_w[static_cast<size_t>(c) * kernel + k];
            }
            depth[static_cast<size_t>(t) * dim + c] = acc;
        }
    }

    auto conv_normed = cpu_layer_norm_rows(depth, conv_norm_w, conv_norm_b, tokens, dim);
    auto conv_act = cpu_silu(conv_normed);
    auto raw = cpu_linear_rows(pw2_w, pw2_b, conv_act, tokens, dim, dim);
    for (uint32_t t = 0; t < tokens; ++t) {
        if (mask[t] == 0) {
            std::fill(raw.begin() + static_cast<size_t>(t) * dim,
                      raw.begin() + static_cast<size_t>(t + 1) * dim,
                      0.0f);
        }
    }
    std::vector<float> residual(raw.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = input[i] + raw[i];
    }
    return GptConformerConvOutput{std::move(normed), std::move(raw), std::move(residual)};
}

GptConformerConvOutput run_gpt_conformer_conv_metal(mit2::MetalContext& metal,
                                                    const mit2::Bundle& bundle,
                                                    const std::vector<float>& input,
                                                    const std::vector<uint32_t>& mask,
                                                    uint32_t tokens,
                                                    uint32_t layer_index = 0,
                                                    const std::string& encoder_prefix = "gpt.conditioning_encoder") {
    constexpr uint32_t dim = 512;
    constexpr uint32_t doubled = 1024;
    constexpr uint32_t kernel = 15;
    if (input.size() != static_cast<size_t>(tokens) * dim || mask.size() != tokens) {
        throw std::runtime_error("GPT conformer conv input shape mismatch");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    const std::string conv = base + ".conv_module";
    auto norm_w = tensor_as_f32(bundle, base + ".norm_conv.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_conv.bias");
    auto pw1_w = tensor_as_f32(bundle, conv + ".pointwise_conv1.weight");
    auto pw1_b = tensor_as_f32(bundle, conv + ".pointwise_conv1.bias");
    auto depth_w = tensor_as_f32(bundle, conv + ".depthwise_conv.weight");
    auto depth_b = tensor_as_f32(bundle, conv + ".depthwise_conv.bias");
    auto conv_norm_w = tensor_as_f32(bundle, conv + ".norm.weight");
    auto conv_norm_b = tensor_as_f32(bundle, conv + ".norm.bias");
    auto pw2_w = tensor_as_f32(bundle, conv + ".pointwise_conv2.weight");
    auto pw2_b = tensor_as_f32(bundle, conv + ".pointwise_conv2.bias");
    if (pw1_w.size() != static_cast<size_t>(doubled) * dim ||
        pw2_w.size() != static_cast<size_t>(dim) * dim ||
        depth_w.size() != static_cast<size_t>(dim) * kernel ||
        depth_b.size() != dim) {
        throw std::runtime_error("GPT conformer conv weight shape mismatch");
    }

    auto normed = metal.layernorm_rows_f32_resident(
        base + ".norm_conv.weight.resident",
        norm_w,
        base + ".norm_conv.bias.resident",
        norm_b,
        input,
        tokens,
        dim,
        1e-5f);
    auto masked_normed = metal.mask_rows_f32(normed, mask, tokens, dim);
    auto pw1 = metal.linear_rows_f32_resident(
        conv + ".pointwise_conv1.weight.resident",
        pw1_w,
        conv + ".pointwise_conv1.bias.resident",
        pw1_b,
        masked_normed,
        tokens,
        doubled,
        dim);
    auto glu = metal.glu_split_f32(pw1, tokens, dim);
    auto depth = metal.depthwise_conv1d_same_f32_resident(
        conv + ".depthwise_conv.weight.resident",
        depth_w,
        conv + ".depthwise_conv.bias.resident",
        depth_b,
        glu,
        tokens,
        dim,
        kernel);
    auto conv_normed = metal.layernorm_rows_f32_resident(
        conv + ".norm.weight.resident",
        conv_norm_w,
        conv + ".norm.bias.resident",
        conv_norm_b,
        depth,
        tokens,
        dim,
        1e-5f);
    auto conv_act = metal.silu_f32(conv_normed);
    auto raw_unmasked = metal.linear_rows_f32_resident(
        conv + ".pointwise_conv2.weight.resident",
        pw2_w,
        conv + ".pointwise_conv2.bias.resident",
        pw2_b,
        conv_act,
        tokens,
        dim,
        dim);
    auto raw = metal.mask_rows_f32(raw_unmasked, mask, tokens, dim);
    auto residual = metal.add_f32(input, raw);
    return GptConformerConvOutput{std::move(normed), std::move(raw), std::move(residual)};
}

bool run_gpt_conformer_conv_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/conv_input.f32");
    auto mask = read_raw_u32(golden_dir + "/conv_mask.u32");
    auto golden_normed = read_raw_f32(golden_dir + "/conv_normed.f32");
    auto golden_raw = read_raw_f32(golden_dir + "/conv_raw.f32");
    auto golden_residual = read_raw_f32(golden_dir + "/conv_residual.f32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer conv golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    auto got = run_gpt_conformer_conv_cpu(bundle, input, mask, tokens);
    if (golden_normed.size() != got.normed.size() ||
        golden_raw.size() != got.raw.size() ||
        golden_residual.size() != got.residual.size()) {
        throw std::runtime_error("GPT conformer conv golden shape mismatch");
    }
    const float normed_err = max_abs_error(got.normed, golden_normed);
    const float raw_err = max_abs_error(got.raw, golden_raw);
    const float residual_err = max_abs_error(got.residual, golden_residual);
    const float err = std::max({normed_err, raw_err, residual_err});
    const size_t nonfinite = count_nonfinite(got.normed) + count_nonfinite(got.raw) + count_nonfinite(got.residual);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_conv_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"kernel_size\": 15,\n";
    std::cout << "  \"normed_max_abs_error\": " << normed_err << ",\n";
    std::cout << "  \"raw_max_abs_error\": " << raw_err << ",\n";
    std::cout << "  \"residual_max_abs_error\": " << residual_err << ",\n";
    std::cout << "  \"nonfinite\": " << nonfinite << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return normed_err <= 1e-3f && raw_err <= 2e-3f && residual_err <= 2e-3f && nonfinite == 0;
}

bool run_gpt_conformer_conv_metal_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/conv_input.f32");
    auto mask = read_raw_u32(golden_dir + "/conv_mask.u32");
    auto golden_normed = read_raw_f32(golden_dir + "/conv_normed.f32");
    auto golden_raw = read_raw_f32(golden_dir + "/conv_raw.f32");
    auto golden_residual = read_raw_f32(golden_dir + "/conv_residual.f32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer conv golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    auto got = run_gpt_conformer_conv_metal(metal, bundle, input, mask, tokens);
    if (golden_normed.size() != got.normed.size() ||
        golden_raw.size() != got.raw.size() ||
        golden_residual.size() != got.residual.size()) {
        throw std::runtime_error("GPT conformer conv metal golden shape mismatch");
    }
    const float normed_err = max_abs_error(got.normed, golden_normed);
    const float raw_err = max_abs_error(got.raw, golden_raw);
    const float residual_err = max_abs_error(got.residual, golden_residual);
    const float err = std::max({normed_err, raw_err, residual_err});
    const size_t nonfinite = count_nonfinite(got.normed) + count_nonfinite(got.raw) + count_nonfinite(got.residual);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_conv_metal_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"kernel_size\": 15,\n";
    std::cout << "  \"conv_source\": \"metal_resident_layernorm_rows_mask_linear_rows_glu_depthwise_conv_silu_add\",\n";
    std::cout << "  \"mask_glu_source\": \"metal_mask_rows_glu_split\",\n";
    std::cout << "  \"normed_max_abs_error\": " << normed_err << ",\n";
    std::cout << "  \"raw_max_abs_error\": " << raw_err << ",\n";
    std::cout << "  \"residual_max_abs_error\": " << residual_err << ",\n";
    std::cout << "  \"nonfinite\": " << nonfinite << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return normed_err <= 1e-3f && raw_err <= 2e-3f && residual_err <= 2e-3f && nonfinite == 0;
}

std::vector<float> run_gpt_conformer_block_cpu(const mit2::Bundle& bundle,
                                               const std::vector<float>& input,
                                               const std::vector<float>& pos_emb,
                                               const std::vector<uint32_t>& mask,
                                               uint32_t tokens,
                                               uint32_t layer_index = 0,
                                               const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                               uint32_t heads = 8) {
    auto attn = run_gpt_conformer_attention_cpu(bundle, input, pos_emb, mask, tokens, layer_index, encoder_prefix, heads);
    auto conv = run_gpt_conformer_conv_cpu(bundle, attn.residual, mask, tokens, layer_index, encoder_prefix);
    auto ff = run_gpt_conformer_ff_tail_cpu(bundle, conv.residual, tokens, layer_index, encoder_prefix);
    return ff.tail;
}

std::vector<float> run_gpt_conformer_block_metal_ff(mit2::MetalContext& metal,
                                                    const mit2::Bundle& bundle,
                                                    const std::vector<float>& input,
                                                    const std::vector<float>& pos_emb,
                                                    const std::vector<uint32_t>& mask,
                                                    uint32_t tokens,
                                                    uint32_t layer_index = 0,
                                                    const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                    uint32_t heads = 8) {
    auto attn = run_gpt_conformer_attention_cpu(bundle, input, pos_emb, mask, tokens, layer_index, encoder_prefix, heads);
    auto conv = run_gpt_conformer_conv_cpu(bundle, attn.residual, mask, tokens, layer_index, encoder_prefix);
    auto ff = run_gpt_conformer_ff_tail_metal(metal, bundle, conv.residual, tokens, layer_index, encoder_prefix);
    return ff.tail;
}

std::vector<float> run_gpt_conformer_block_metal_attn_ff(mit2::MetalContext& metal,
                                                         const mit2::Bundle& bundle,
                                                         const std::vector<float>& input,
                                                         const std::vector<float>& pos_emb,
                                                         const std::vector<uint32_t>& mask,
                                                         uint32_t tokens,
                                                         uint32_t layer_index = 0,
                                                         const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                         uint32_t heads = 8) {
    auto attn = run_gpt_conformer_attention_metal_proj(metal, bundle, input, pos_emb, mask, tokens, layer_index, encoder_prefix, heads);
    auto conv = run_gpt_conformer_conv_cpu(bundle, attn.residual, mask, tokens, layer_index, encoder_prefix);
    auto ff = run_gpt_conformer_ff_tail_metal(metal, bundle, conv.residual, tokens, layer_index, encoder_prefix);
    return ff.tail;
}

std::vector<float> run_gpt_conformer_block_metal_attn_conv_ff(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layer_index = 0,
                                                              const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                              uint32_t heads = 8) {
    auto attn = run_gpt_conformer_attention_metal_proj(metal, bundle, input, pos_emb, mask, tokens, layer_index, encoder_prefix, heads);
    auto conv = run_gpt_conformer_conv_metal(metal, bundle, attn.residual, mask, tokens, layer_index, encoder_prefix);
    auto ff = run_gpt_conformer_ff_tail_metal(metal, bundle, conv.residual, tokens, layer_index, encoder_prefix);
    return ff.tail;
}

bool run_gpt_conformer_block_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/block_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/block_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/block_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/block_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer block golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer block golden shape mismatch");
    }
    auto got = run_gpt_conformer_block_cpu(bundle, input, pos_emb, mask, tokens, 0);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_block_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-3f && mask_match && count_nonfinite(got) == 0;
}

bool run_gpt_conformer_block_metal_ff_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/block_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/block_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/block_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/block_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer block golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer block metal-ff golden shape mismatch");
    }
    auto got = run_gpt_conformer_block_metal_ff(metal, bundle, input, pos_emb, mask, tokens, 0);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_block_metal_ff_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"ff_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-3f && mask_match && count_nonfinite(got) == 0;
}

bool run_gpt_conformer_block_metal_attn_ff_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/block_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/block_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/block_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/block_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer block golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer block metal-attn-ff golden shape mismatch");
    }
    auto got = run_gpt_conformer_block_metal_attn_ff(metal, bundle, input, pos_emb, mask, tokens, 0);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_block_metal_attn_ff_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"attn_projection_source\": \"metal_resident_layernorm_rows_linear_rows_add\",\n";
    std::cout << "  \"attention_core_source\": \"metal_resident_bias_relative_attention_softmax\",\n";
    std::cout << "  \"ff_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-3f && mask_match && count_nonfinite(got) == 0;
}

bool run_gpt_conformer_block_metal_attn_conv_ff_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/block_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/block_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/block_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/block_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer block golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer block metal-attn-conv-ff golden shape mismatch");
    }
    auto got = run_gpt_conformer_block_metal_attn_conv_ff(metal, bundle, input, pos_emb, mask, tokens, 0);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_block_metal_attn_conv_ff_golden\",\n";
    std::cout << "  \"layer_index\": 0,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"attn_projection_source\": \"metal_resident_layernorm_rows_linear_rows_add\",\n";
    std::cout << "  \"attention_core_source\": \"metal_resident_bias_relative_attention_softmax\",\n";
    std::cout << "  \"conv_source\": \"metal_resident_layernorm_rows_mask_linear_rows_glu_depthwise_conv_silu_add\",\n";
    std::cout << "  \"conv_mask_glu_source\": \"metal_mask_rows_glu_split\",\n";
    std::cout << "  \"ff_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 2e-3f && mask_match && count_nonfinite(got) == 0;
}

std::vector<float> run_gpt_conformer_stack_cpu(const mit2::Bundle& bundle,
                                               const std::vector<float>& input,
                                               const std::vector<float>& pos_emb,
                                               const std::vector<uint32_t>& mask,
                                               uint32_t tokens,
                                               uint32_t layers = 6,
                                               const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                               uint32_t heads = 8) {
    constexpr uint32_t dim = 512;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer stack input must have shape [tokens,512]");
    }
    std::vector<float> state = input;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        state = run_gpt_conformer_block_cpu(bundle, state, pos_emb, mask, tokens, layer, encoder_prefix, heads);
    }
    auto after_w = tensor_as_f32(bundle, encoder_prefix + ".after_norm.weight");
    auto after_b = tensor_as_f32(bundle, encoder_prefix + ".after_norm.bias");
    return cpu_layer_norm_rows(state, after_w, after_b, tokens, dim);
}

std::vector<float> run_gpt_conformer_stack_metal_ff(mit2::MetalContext& metal,
                                                    const mit2::Bundle& bundle,
                                                    const std::vector<float>& input,
                                                    const std::vector<float>& pos_emb,
                                                    const std::vector<uint32_t>& mask,
                                                    uint32_t tokens,
                                                    uint32_t layers = 6,
                                                    const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                    uint32_t heads = 8) {
    constexpr uint32_t dim = 512;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer stack input must have shape [tokens,512]");
    }
    std::vector<float> state = input;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        state = run_gpt_conformer_block_metal_ff(metal, bundle, state, pos_emb, mask, tokens, layer, encoder_prefix, heads);
    }
    auto after_w = tensor_as_f32(bundle, encoder_prefix + ".after_norm.weight");
    auto after_b = tensor_as_f32(bundle, encoder_prefix + ".after_norm.bias");
    return metal.layernorm_rows_f32_resident(
        encoder_prefix + ".after_norm.weight.resident",
        after_w,
        encoder_prefix + ".after_norm.bias.resident",
        after_b,
        state,
        tokens,
        dim,
        1e-5f);
}

std::vector<float> run_gpt_conformer_stack_metal_attn_ff(mit2::MetalContext& metal,
                                                         const mit2::Bundle& bundle,
                                                         const std::vector<float>& input,
                                                         const std::vector<float>& pos_emb,
                                                         const std::vector<uint32_t>& mask,
                                                         uint32_t tokens,
                                                         uint32_t layers = 6,
                                                         const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                         uint32_t heads = 8) {
    constexpr uint32_t dim = 512;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer stack input must have shape [tokens,512]");
    }
    std::vector<float> state = input;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        state = run_gpt_conformer_block_metal_attn_ff(metal, bundle, state, pos_emb, mask, tokens, layer, encoder_prefix, heads);
    }
    auto after_w = tensor_as_f32(bundle, encoder_prefix + ".after_norm.weight");
    auto after_b = tensor_as_f32(bundle, encoder_prefix + ".after_norm.bias");
    return metal.layernorm_rows_f32_resident(
        encoder_prefix + ".after_norm.weight.resident",
        after_w,
        encoder_prefix + ".after_norm.bias.resident",
        after_b,
        state,
        tokens,
        dim,
        1e-5f);
}

std::vector<float> run_gpt_conformer_stack_metal_attn_conv_ff(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layers = 6,
                                                              const std::string& encoder_prefix = "gpt.conditioning_encoder",
                                                              uint32_t heads = 8) {
    constexpr uint32_t dim = 512;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer stack input must have shape [tokens,512]");
    }
    std::vector<float> state = input;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        state = run_gpt_conformer_block_metal_attn_conv_ff(metal, bundle, state, pos_emb, mask, tokens, layer, encoder_prefix, heads);
    }
    auto after_w = tensor_as_f32(bundle, encoder_prefix + ".after_norm.weight");
    auto after_b = tensor_as_f32(bundle, encoder_prefix + ".after_norm.bias");
    return metal.layernorm_rows_f32_resident(
        encoder_prefix + ".after_norm.weight.resident",
        after_w,
        encoder_prefix + ".after_norm.bias.resident",
        after_b,
        state,
        tokens,
        dim,
        1e-5f);
}

bool run_gpt_conformer_stack_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/stack_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/stack_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/stack_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/stack_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer stack golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer stack golden shape mismatch");
    }
    auto got = run_gpt_conformer_stack_cpu(bundle, input, pos_emb, mask, tokens, 6);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_stack_golden\",\n";
    std::cout << "  \"layers\": 6,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 3e-3f && mask_match && count_nonfinite(got) == 0;
}

bool run_gpt_conformer_stack_metal_ff_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/stack_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/stack_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/stack_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/stack_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer stack golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer stack metal-ff golden shape mismatch");
    }
    auto got = run_gpt_conformer_stack_metal_ff(metal, bundle, input, pos_emb, mask, tokens, 6);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_stack_metal_ff_golden\",\n";
    std::cout << "  \"layers\": 6,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"ff_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"after_norm_source\": \"metal_layernorm_rows_f32_resident\",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 3e-3f && mask_match && count_nonfinite(got) == 0;
}

bool run_gpt_conformer_stack_metal_attn_ff_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/stack_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/stack_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/stack_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/stack_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer stack golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer stack metal-attn-ff golden shape mismatch");
    }
    auto got = run_gpt_conformer_stack_metal_attn_ff(metal, bundle, input, pos_emb, mask, tokens, 6);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_stack_metal_attn_ff_golden\",\n";
    std::cout << "  \"layers\": 6,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"attn_projection_source\": \"metal_resident_layernorm_rows_linear_rows_add\",\n";
    std::cout << "  \"attention_core_source\": \"metal_resident_bias_relative_attention_softmax\",\n";
    std::cout << "  \"ff_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"after_norm_source\": \"metal_layernorm_rows_f32_resident\",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 3e-3f && mask_match && count_nonfinite(got) == 0;
}

bool run_gpt_conformer_stack_metal_attn_conv_ff_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(golden_dir + "/stack_input.f32");
    auto pos_emb = read_raw_f32(golden_dir + "/pos_emb.f32");
    auto mask = read_raw_u32(golden_dir + "/stack_mask.u32");
    auto golden = read_raw_f32(golden_dir + "/stack_output.f32");
    auto golden_mask = read_raw_u32(golden_dir + "/stack_output_mask.u32");
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer stack golden input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (golden.size() != input.size() || golden_mask.size() != mask.size()) {
        throw std::runtime_error("GPT conformer stack metal-attn-conv-ff golden shape mismatch");
    }
    auto got = run_gpt_conformer_stack_metal_attn_conv_ff(metal, bundle, input, pos_emb, mask, tokens, 6);
    const float err = max_abs_error(got, golden);
    bool mask_match = true;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask_match = mask_match && mask[i] == golden_mask[i];
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_stack_metal_attn_conv_ff_golden\",\n";
    std::cout << "  \"layers\": 6,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"attn_projection_source\": \"metal_resident_layernorm_rows_linear_rows_add\",\n";
    std::cout << "  \"attention_core_source\": \"metal_resident_bias_relative_attention_softmax\",\n";
    std::cout << "  \"conv_source\": \"metal_resident_layernorm_rows_mask_linear_rows_glu_depthwise_conv_silu_add\",\n";
    std::cout << "  \"conv_mask_glu_source\": \"metal_mask_rows_glu_split\",\n";
    std::cout << "  \"ff_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"after_norm_source\": \"metal_layernorm_rows_f32_resident\",\n";
    std::cout << "  \"mask_match\": " << (mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(got) << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 3e-3f && mask_match && count_nonfinite(got) == 0;
}

bool export_gpt_conformer_stack(const std::string& bundle_dir,
                                const std::string& input_path,
                                const std::string& pos_emb_path,
                                const std::string& mask_path,
                                const std::string& output_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t dim = 512;
    auto input = read_raw_f32(input_path);
    auto pos_emb = read_raw_f32(pos_emb_path);
    auto mask = read_raw_u32(mask_path);
    if (input.empty() || (input.size() % dim) != 0) {
        throw std::runtime_error("GPT conformer stack export input must have shape [tokens,512]");
    }
    const uint32_t tokens = static_cast<uint32_t>(input.size() / dim);
    if (pos_emb.size() != input.size() || mask.size() != tokens) {
        throw std::runtime_error("GPT conformer stack export shape mismatch");
    }
    auto out = run_gpt_conformer_stack_metal_attn_conv_ff(metal, bundle, input, pos_emb, mask, tokens, 6);
    write_raw_f32(output_path, out);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_conformer_stack_export\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"stack_input_f32\": \"" << json_escape(input_path) << "\",\n";
    std::cout << "  \"pos_emb_f32\": \"" << json_escape(pos_emb_path) << "\",\n";
    std::cout << "  \"mask_u32\": \"" << json_escape(mask_path) << "\",\n";
    std::cout << "  \"output_context_f32\": \"" << json_escape(output_path) << "\",\n";
    std::cout << "  \"layers\": 6,\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"dim\": " << dim << ",\n";
    std::cout << "  \"attn_projection_source\": \"metal_resident_layernorm_rows_linear_rows_add\",\n";
    std::cout << "  \"attention_core_source\": \"metal_resident_bias_relative_attention_softmax\",\n";
    std::cout << "  \"conv_source\": \"metal_resident_layernorm_rows_mask_linear_rows_glu_depthwise_conv_silu_add\",\n";
    std::cout << "  \"conv_mask_glu_source\": \"metal_mask_rows_glu_split\",\n";
    std::cout << "  \"ff_source\": \"metal_resident_layernorm_rows_linear_rows_silu_add\",\n";
    std::cout << "  \"after_norm_source\": \"metal_layernorm_rows_f32_resident\",\n";
    std::cout << "  \"nonfinite\": " << count_nonfinite(out) << "\n";
    std::cout << "}\n";
    return count_nonfinite(out) == 0;
}

bool run_gpt_kv_greedy_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t width = 1280;
    auto conds = read_raw_f32(golden_dir + "/conds_latent.f32");
    auto text_ids = read_raw_u32(golden_dir + "/text_ids.u32");
    auto expected_codes = read_raw_u32(golden_dir + "/codes.u32");
    auto expected_inputs_embeds = read_raw_f32(golden_dir + "/inputs_embeds.f32");
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("GPT golden conds_latent must have shape [tokens,1280]");
    }
    if (expected_codes.empty()) {
        throw std::runtime_error("GPT golden codes must contain at least one token");
    }
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_slots = static_cast<uint32_t>(text_ids.size());
    auto prefix_ref = run_gpt_prepare_inputs_cpu(bundle, conds, text_ids, cond_tokens, text_slots);
    auto prefix_got = run_gpt_prepare_inputs_metal(metal, bundle, conds, text_ids, cond_tokens, text_slots);
    if (prefix_ref.inputs_embeds.size() != expected_inputs_embeds.size()) {
        throw std::runtime_error("GPT golden inputs_embeds size does not match native prepared prefix");
    }
    const uint32_t prefix_tokens = static_cast<uint32_t>(prefix_ref.inputs_embeds.size() / width);
    auto cached_ref = run_gpt_kv_greedy_cpu(bundle, prefix_ref.inputs_embeds, prefix_tokens, static_cast<uint32_t>(expected_codes.size()), true);
    auto cached_got = run_gpt_kv_greedy_metal(metal, bundle, prefix_got.inputs_embeds, prefix_tokens, static_cast<uint32_t>(expected_codes.size()), true);
    const bool fake_inputs_match = prefix_ref.fake_inputs == prefix_got.fake_inputs;
    const bool attention_mask_match = prefix_ref.attention_mask == prefix_got.attention_mask;
    const bool cpu_codes_match = cached_ref.predicted_tokens == expected_codes;
    const bool metal_codes_match = cached_got.predicted_tokens == expected_codes;
    const bool metal_inputs_match = cached_got.input_tokens == cached_ref.input_tokens;
    const float prefix_cpu_vs_golden_err = max_abs_error(prefix_ref.inputs_embeds, expected_inputs_embeds);
    const float prefix_metal_err = max_abs_error(prefix_got.inputs_embeds, prefix_ref.inputs_embeds);
    const float metal_logits_err = max_abs_error(cached_got.last_logits, cached_ref.last_logits);
    const float err = std::max({prefix_cpu_vs_golden_err, prefix_metal_err, metal_logits_err});
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_greedy_golden\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"cond_tokens\": " << cond_tokens << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"steps\": " << expected_codes.size() << ",\n";
    std::cout << "  \"prefix_cpu_vs_golden_max_abs_error\": " << prefix_cpu_vs_golden_err << ",\n";
    std::cout << "  \"prefix_metal_max_abs_error\": " << prefix_metal_err << ",\n";
    std::cout << "  \"metal_cached_last_logits_max_abs_error\": " << metal_logits_err << ",\n";
    std::cout << "  \"fake_inputs_match\": " << (fake_inputs_match ? "true" : "false") << ",\n";
    std::cout << "  \"attention_mask_match\": " << (attention_mask_match ? "true" : "false") << ",\n";
    std::cout << "  \"cpu_codes_match_golden\": " << (cpu_codes_match ? "true" : "false") << ",\n";
    std::cout << "  \"metal_codes_match_golden\": " << (metal_codes_match ? "true" : "false") << ",\n";
    std::cout << "  \"metal_inputs_match_cpu\": " << (metal_inputs_match ? "true" : "false") << ",\n";
    std::cout << "  \"expected_codes\": [";
    for (size_t i = 0; i < expected_codes.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << expected_codes[i];
    }
    std::cout << "],\n";
    std::cout << "  \"predicted_codes_cpu\": [";
    for (size_t i = 0; i < cached_ref.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << cached_ref.predicted_tokens[i];
    }
    std::cout << "],\n";
    std::cout << "  \"predicted_codes_metal\": [";
    for (size_t i = 0; i < cached_got.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << cached_got.predicted_tokens[i];
    }
    std::cout << "],\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return prefix_cpu_vs_golden_err <= 1e-6f && prefix_metal_err <= 1e-6f && metal_logits_err <= 3e-3f &&
           fake_inputs_match && attention_mask_match && cpu_codes_match && metal_codes_match && metal_inputs_match;
}

bool export_gpt_kv_codes_golden(const std::string& bundle_dir, const std::string& golden_dir, const std::string& output_codes_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t width = 1280;
    auto conds = read_raw_f32(golden_dir + "/conds_latent.f32");
    auto text_ids = read_raw_u32(golden_dir + "/text_ids.u32");
    auto expected_codes = read_raw_u32(golden_dir + "/codes.u32");
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("GPT code export conds_latent must have shape [tokens,1280]");
    }
    if (expected_codes.empty()) {
        throw std::runtime_error("GPT code export golden codes must contain at least one token");
    }
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_slots = static_cast<uint32_t>(text_ids.size());
    auto prefix = run_gpt_prepare_inputs_metal(metal, bundle, conds, text_ids, cond_tokens, text_slots);
    const uint32_t prefix_tokens = static_cast<uint32_t>(prefix.inputs_embeds.size() / width);
    auto generated = run_gpt_kv_greedy_metal(metal, bundle, prefix.inputs_embeds, prefix_tokens, static_cast<uint32_t>(expected_codes.size()), true);
    write_raw_u32(output_codes_path, generated.predicted_tokens);
    const bool codes_match = generated.predicted_tokens == expected_codes;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_codes_export\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"output_codes_u32\": \"" << output_codes_path << "\",\n";
    std::cout << "  \"cond_tokens\": " << cond_tokens << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"codes\": " << generated.predicted_tokens.size() << ",\n";
    std::cout << "  \"codes_match_golden\": " << (codes_match ? "true" : "false") << ",\n";
    std::cout << "  \"predicted_codes\": [";
    for (size_t i = 0; i < generated.predicted_tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << generated.predicted_tokens[i];
    }
    std::cout << "]\n";
    std::cout << "}\n";
    return codes_match;
}

bool export_gpt_kv_codes_inputs(const std::string& bundle_dir,
                                const std::string& conds_path,
                                const std::string& text_ids_path,
                                uint32_t max_codes,
                                const std::string& output_codes_path) {
    const auto started = Clock::now();
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t width = 1280;
    auto conds = read_raw_f32(conds_path);
    auto text_ids = read_raw_u32(text_ids_path);
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("GPT input code export conds must have shape [tokens,1280]");
    }
    if (max_codes == 0) {
        throw std::runtime_error("GPT input code export max_codes must be positive");
    }
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_slots = static_cast<uint32_t>(text_ids.size());
    auto prefix = run_gpt_prepare_inputs_metal(metal, bundle, conds, text_ids, cond_tokens, text_slots);
    const uint32_t prefix_tokens = static_cast<uint32_t>(prefix.inputs_embeds.size() / width);
    auto generated = run_gpt_kv_greedy_metal(metal, bundle, prefix.inputs_embeds, prefix_tokens, max_codes, true);
    std::vector<uint32_t> codes = generated.predicted_tokens;
    if (generated.first_stop_step >= 0) {
        codes.resize(static_cast<size_t>(generated.first_stop_step));
    }
    const auto stats = metal.resource_stats();
    const double elapsed = seconds_since(started);
    write_raw_u32(output_codes_path, codes);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_kv_codes_inputs_export\",\n";
    std::cout << "  \"conds_f32\": \"" << conds_path << "\",\n";
    std::cout << "  \"text_ids_u32\": \"" << text_ids_path << "\",\n";
    std::cout << "  \"output_codes_u32\": \"" << output_codes_path << "\",\n";
    std::cout << "  \"cond_tokens\": " << cond_tokens << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"prefix_tokens\": " << prefix_tokens << ",\n";
    std::cout << "  \"max_codes\": " << max_codes << ",\n";
    std::cout << "  \"raw_codes\": " << generated.predicted_tokens.size() << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"first_stop_step\": " << generated.first_stop_step << ",\n";
    std::cout << "  \"predicted_codes\": [";
    for (size_t i = 0; i < codes.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << codes[i];
    }
    std::cout << "],\n";
    print_gpt_decode_rate_json(static_cast<uint32_t>(generated.predicted_tokens.size()), codes.size(), elapsed);
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"elapsed_seconds\": " << elapsed << "\n";
    std::cout << "}\n";
    return true;
}

std::vector<float> run_vq2emb_cpu(const mit2::Bundle& bundle, const std::vector<uint32_t>& codes);
std::vector<float> run_vq2emb_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<uint32_t>& codes);
std::vector<float> run_length_regulator_full_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens);
std::vector<float> run_length_regulator_full_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens);
bool export_hot_codes_condition_inputs(const std::string& bundle_dir,
                                       const std::string& voice_bundle_dir,
                                       const std::string& conds_path,
                                       const std::string& text_ids_path,
                                       const std::string& codes_path,
                                       uint32_t prompt_tokens,
                                       const std::string& output_condition_path);

std::vector<float> build_gpt_latent_forward_input_cpu(const mit2::Bundle& bundle, const std::vector<float>& conds, const std::vector<uint32_t>& text_ids, const std::vector<uint32_t>& codes) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t start_text_token = 0;
    constexpr uint32_t stop_text_token = 1;
    constexpr uint32_t start_mel_token = 8192;
    constexpr uint32_t stop_mel_token = 8193;
    auto text_embedding = tensor_as_f32(bundle, "gpt.text_embedding.weight");
    auto text_pos_embedding = tensor_as_f32(bundle, "gpt.text_pos_embedding.emb.weight");
    auto mel_embedding = tensor_as_f32(bundle, "gpt.mel_embedding.weight");
    auto mel_pos_embedding = tensor_as_f32(bundle, "gpt.mel_pos_embedding.emb.weight");
    std::vector<uint32_t> text_inputs;
    text_inputs.reserve(text_ids.size() + 2);
    text_inputs.push_back(start_text_token);
    text_inputs.insert(text_inputs.end(), text_ids.begin(), text_ids.end());
    text_inputs.push_back(stop_text_token);
    std::vector<uint32_t> mel_inputs;
    mel_inputs.reserve(codes.size() + 2);
    mel_inputs.push_back(start_mel_token);
    mel_inputs.insert(mel_inputs.end(), codes.begin(), codes.end());
    mel_inputs.push_back(stop_mel_token);
    std::vector<float> out(conds.size() + static_cast<size_t>(text_inputs.size() + mel_inputs.size()) * width);
    std::copy(conds.begin(), conds.end(), out.begin());
    size_t dst = conds.size();
    for (uint32_t t = 0; t < text_inputs.size(); ++t) {
        const uint32_t token = text_inputs[t];
        for (uint32_t c = 0; c < width; ++c) {
            out[dst + static_cast<size_t>(t) * width + c] =
                text_embedding[static_cast<size_t>(token) * width + c] + text_pos_embedding[static_cast<size_t>(t) * width + c];
        }
    }
    dst += static_cast<size_t>(text_inputs.size()) * width;
    for (uint32_t t = 0; t < mel_inputs.size(); ++t) {
        const uint32_t token = mel_inputs[t];
        for (uint32_t c = 0; c < width; ++c) {
            out[dst + static_cast<size_t>(t) * width + c] =
                mel_embedding[static_cast<size_t>(token) * width + c] + mel_pos_embedding[static_cast<size_t>(t) * width + c];
        }
    }
    return out;
}

std::vector<float> build_gpt_latent_forward_input_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& conds, const std::vector<uint32_t>& text_ids, const std::vector<uint32_t>& codes) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t start_text_token = 0;
    constexpr uint32_t stop_text_token = 1;
    constexpr uint32_t start_mel_token = 8192;
    constexpr uint32_t stop_mel_token = 8193;
    auto skip_if_resident = [&](const char* name) {
        return metal.residentExists(std::string(name) + ".resident")
                   ? std::vector<float>{}
                   : tensor_as_f32(bundle, name);
    };
    auto text_embedding = skip_if_resident("gpt.text_embedding.weight");
    auto text_pos_embedding = skip_if_resident("gpt.text_pos_embedding.emb.weight");
    auto mel_embedding = skip_if_resident("gpt.mel_embedding.weight");
    auto mel_pos_embedding = skip_if_resident("gpt.mel_pos_embedding.emb.weight");
    std::vector<uint32_t> text_inputs;
    text_inputs.reserve(text_ids.size() + 2);
    text_inputs.push_back(start_text_token);
    text_inputs.insert(text_inputs.end(), text_ids.begin(), text_ids.end());
    text_inputs.push_back(stop_text_token);
    std::vector<uint32_t> mel_inputs;
    mel_inputs.reserve(codes.size() + 2);
    mel_inputs.push_back(start_mel_token);
    mel_inputs.insert(mel_inputs.end(), codes.begin(), codes.end());
    mel_inputs.push_back(stop_mel_token);
    auto text_emb = metal.embedding_f32_resident("gpt.text_embedding.weight.resident", text_embedding, text_inputs, width);
    std::vector<uint32_t> text_pos(text_inputs.size());
    for (uint32_t i = 0; i < text_pos.size(); ++i) {
        text_pos[i] = i;
    }
    auto text_pos_emb = metal.embedding_f32_resident("gpt.text_pos_embedding.emb.weight.resident", text_pos_embedding, text_pos, width);
    auto mel_emb = metal.embedding_f32_resident("gpt.mel_embedding.weight.resident", mel_embedding, mel_inputs, width);
    std::vector<uint32_t> mel_pos(mel_inputs.size());
    for (uint32_t i = 0; i < mel_pos.size(); ++i) {
        mel_pos[i] = i;
    }
    auto mel_pos_emb = metal.embedding_f32_resident("gpt.mel_pos_embedding.emb.weight.resident", mel_pos_embedding, mel_pos, width);
    std::vector<float> out(conds.size() + static_cast<size_t>(text_inputs.size() + mel_inputs.size()) * width);
    std::copy(conds.begin(), conds.end(), out.begin());
    size_t dst = conds.size();
    for (size_t i = 0; i < text_emb.size(); ++i) {
        out[dst + i] = text_emb[i] + text_pos_emb[i];
    }
    dst += text_emb.size();
    for (size_t i = 0; i < mel_emb.size(); ++i) {
        out[dst + i] = mel_emb[i] + mel_pos_emb[i];
    }
    return out;
}

std::vector<float> run_gpt_latent_forward_cpu(const mit2::Bundle& bundle, const std::vector<float>& conds, const std::vector<uint32_t>& text_ids, const std::vector<uint32_t>& codes) {
    constexpr uint32_t width = 1280;
    auto input = build_gpt_latent_forward_input_cpu(bundle, conds, text_ids, codes);
    const uint32_t total_tokens = static_cast<uint32_t>(input.size() / width);
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_tokens = static_cast<uint32_t>(text_ids.size() + 2);
    auto stack = run_gpt_transformer_stack_cpu(bundle, input, total_tokens, 24);
    auto norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    auto norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    std::vector<float> out(static_cast<size_t>(codes.size()) * width);
    const uint32_t mel_offset = cond_tokens + text_tokens;
    for (uint32_t t = 0; t < codes.size(); ++t) {
        std::vector<float> row(stack.ln_f.begin() + static_cast<size_t>(mel_offset + t) * width,
                               stack.ln_f.begin() + static_cast<size_t>(mel_offset + t + 1) * width);
        auto normed = cpu_layernorm(row, norm_weight, norm_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), out.begin() + static_cast<size_t>(t) * width);
    }
    return out;
}

std::vector<float> run_gpt_latent_forward_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& conds, const std::vector<uint32_t>& text_ids, const std::vector<uint32_t>& codes) {
    constexpr uint32_t width = 1280;
    auto input = build_gpt_latent_forward_input_metal(metal, bundle, conds, text_ids, codes);
    const uint32_t total_tokens = static_cast<uint32_t>(input.size() / width);
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_tokens = static_cast<uint32_t>(text_ids.size() + 2);
    auto ln_f = run_gpt_transformer_stack_lnf_metal_pass(metal, bundle, input, total_tokens, 24);
    auto norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    auto norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    if (codes.empty()) {
        return {};
    }
    std::vector<float> latent_input(static_cast<size_t>(codes.size()) * width);
    const uint32_t mel_offset = cond_tokens + text_tokens;
    for (uint32_t t = 0; t < codes.size(); ++t) {
        std::copy(ln_f.begin() + static_cast<size_t>(mel_offset + t) * width,
                  ln_f.begin() + static_cast<size_t>(mel_offset + t + 1) * width,
                  latent_input.begin() + static_cast<size_t>(t) * width);
    }
    return metal.layernorm_rows_f32_resident(
        "gpt.final_norm.weight.resident",
        norm_weight,
        "gpt.final_norm.bias.resident",
        norm_bias,
        latent_input,
        static_cast<uint32_t>(codes.size()),
        width,
        1e-5f);
}

bool run_gpt_latent_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t width = 1280;
    auto conds = read_raw_f32(golden_dir + "/conds_latent.f32");
    auto text_ids = read_raw_u32(golden_dir + "/text_ids.u32");
    auto codes = read_raw_u32(golden_dir + "/codes.u32");
    auto golden_latent = read_raw_f32(golden_dir + "/gpt_latent.f32");
    auto golden_gpt_layer = read_raw_f32(golden_dir + "/gpt_layer.f32");
    auto golden_vq2emb = read_raw_f32(golden_dir + "/vq2emb.f32");
    auto golden_s_infer = read_raw_f32(golden_dir + "/s_infer.f32");
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("GPT latent golden conds_latent must have shape [tokens,1280]");
    }
    if (golden_latent.size() != static_cast<size_t>(codes.size()) * width) {
        throw std::runtime_error("GPT latent golden gpt_latent size mismatch");
    }
    auto latent_ref = run_gpt_latent_forward_cpu(bundle, conds, text_ids, codes);
    auto latent_got = run_gpt_latent_forward_metal(metal, bundle, conds, text_ids, codes);
    auto gpt_layer_ref = run_gpt_layer_cpu(bundle, latent_ref, static_cast<uint32_t>(codes.size()));
    auto gpt_layer_got = run_gpt_layer_metal(metal, bundle, latent_got, static_cast<uint32_t>(codes.size()));
    auto vq_ref = run_vq2emb_cpu(bundle, codes);
    auto vq_got = run_vq2emb_metal(metal, bundle, codes);
    std::vector<float> s_ref(gpt_layer_ref.size());
    for (size_t i = 0; i < s_ref.size(); ++i) {
        s_ref[i] = gpt_layer_ref[i] + vq_ref[i];
    }
    auto s_got = metal.add_f32(gpt_layer_got, vq_got);
    const float latent_cpu_golden_err = max_abs_error(latent_ref, golden_latent);
    const float latent_metal_err = max_abs_error(latent_got, latent_ref);
    const float gpt_layer_cpu_golden_err = max_abs_error(gpt_layer_ref, golden_gpt_layer);
    const float gpt_layer_metal_err = max_abs_error(gpt_layer_got, gpt_layer_ref);
    const float vq_cpu_golden_err = max_abs_error(vq_ref, golden_vq2emb);
    const float vq_metal_err = max_abs_error(vq_got, vq_ref);
    const float s_cpu_golden_err = max_abs_error(s_ref, golden_s_infer);
    const float s_metal_err = max_abs_error(s_got, s_ref);
    const float err = std::max({latent_cpu_golden_err, latent_metal_err, gpt_layer_cpu_golden_err, gpt_layer_metal_err, vq_cpu_golden_err, vq_metal_err, s_cpu_golden_err, s_metal_err});
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_latent_golden\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"cond_tokens\": " << (conds.size() / width) << ",\n";
    std::cout << "  \"text_tokens\": " << text_ids.size() << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"latent_cpu_vs_golden_max_abs_error\": " << latent_cpu_golden_err << ",\n";
    std::cout << "  \"latent_metal_max_abs_error\": " << latent_metal_err << ",\n";
    std::cout << "  \"gpt_layer_cpu_vs_golden_max_abs_error\": " << gpt_layer_cpu_golden_err << ",\n";
    std::cout << "  \"gpt_layer_metal_max_abs_error\": " << gpt_layer_metal_err << ",\n";
    std::cout << "  \"vq2emb_cpu_vs_golden_max_abs_error\": " << vq_cpu_golden_err << ",\n";
    std::cout << "  \"vq2emb_metal_max_abs_error\": " << vq_metal_err << ",\n";
    std::cout << "  \"s_infer_cpu_vs_golden_max_abs_error\": " << s_cpu_golden_err << ",\n";
    std::cout << "  \"s_infer_metal_max_abs_error\": " << s_metal_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return latent_cpu_golden_err <= 3e-3f && latent_metal_err <= 3e-3f &&
           gpt_layer_cpu_golden_err <= 3e-3f && gpt_layer_metal_err <= 3e-3f &&
           vq_cpu_golden_err <= 3e-3f && vq_metal_err <= 3e-3f &&
           s_cpu_golden_err <= 3e-3f && s_metal_err <= 3e-3f;
}

bool trace_gpt_latent_golden(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t width = 1280;
    auto conds = read_raw_f32(golden_dir + "/conds_latent.f32");
    auto text_ids = read_raw_u32(golden_dir + "/text_ids.u32");
    auto codes = read_raw_u32(golden_dir + "/codes.u32");
    auto golden_latent = read_raw_f32(golden_dir + "/gpt_latent.f32");
    if (conds.empty() || (conds.size() % width) != 0) {
        throw std::runtime_error("GPT latent trace conds_latent must have shape [tokens,1280]");
    }
    if (golden_latent.size() != static_cast<size_t>(codes.size()) * width) {
        throw std::runtime_error("GPT latent trace gpt_latent size mismatch");
    }

    auto hidden_ref = build_gpt_latent_forward_input_cpu(bundle, conds, text_ids, codes);
    auto hidden_got = build_gpt_latent_forward_input_metal(metal, bundle, conds, text_ids, codes);
    const uint32_t total_tokens = static_cast<uint32_t>(hidden_ref.size() / width);
    const uint32_t cond_tokens = static_cast<uint32_t>(conds.size() / width);
    const uint32_t text_tokens = static_cast<uint32_t>(text_ids.size() + 2);
    const uint32_t mel_offset = cond_tokens + text_tokens;
    if (hidden_got.size() != hidden_ref.size()) {
        throw std::runtime_error("GPT latent trace input size mismatch");
    }

    float max_err = max_abs_error(hidden_got, hidden_ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"gpt_latent_trace\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"cond_tokens\": " << cond_tokens << ",\n";
    std::cout << "  \"text_token_ids\": " << text_ids.size() << ",\n";
    std::cout << "  \"text_tokens_with_boundaries\": " << text_tokens << ",\n";
    std::cout << "  \"codes\": " << codes.size() << ",\n";
    std::cout << "  \"total_tokens\": " << total_tokens << ",\n";
    std::cout << "  \"input_max_abs_error\": " << max_err << ",\n";
    std::cout << "  \"input_nonfinite\": " << (count_nonfinite(hidden_ref) + count_nonfinite(hidden_got)) << ",\n";
    auto hidden_resident = hidden_got;
    std::vector<GptTransformerLayerWeights> resident_weights;
    resident_weights.reserve(24);
    for (uint32_t layer = 0; layer < 24; ++layer) {
        resident_weights.push_back(load_gpt_transformer_layer_weights(bundle, layer));
    }

    constexpr uint32_t qkv_width = 3840;
    constexpr uint32_t mlp_width = 5120;
    const auto& trace_layer0 = resident_weights.front();
    std::vector<float> trace_ln1_row(static_cast<size_t>(total_tokens) * width);
    std::vector<float> trace_qkv_row(static_cast<size_t>(total_tokens) * qkv_width);
    for (uint32_t t = 0; t < total_tokens; ++t) {
        std::vector<float> row(hidden_got.begin() + static_cast<size_t>(t) * width,
                               hidden_got.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = metal.layernorm_f32(row, trace_layer0.ln1_weight, trace_layer0.ln1_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), trace_ln1_row.begin() + static_cast<size_t>(t) * width);
        auto qkv = metal.linear_f32(trace_layer0.c_attn_weight, trace_layer0.c_attn_bias, normed, qkv_width, width);
        std::copy(qkv.begin(), qkv.end(), trace_qkv_row.begin() + static_cast<size_t>(t) * qkv_width);
    }
    auto trace_ln1_resident = metal.layernorm_rows_f32_resident(
        "trace.gpt.gpt.h.0.ln_1.weight.resident",
        trace_layer0.ln1_weight,
        "trace.gpt.gpt.h.0.ln_1.bias.resident",
        trace_layer0.ln1_bias,
        hidden_got,
        total_tokens,
        width,
        1e-5f);
    auto trace_qkv_resident = metal.linear_rows_f32_resident(
        "trace.gpt.gpt.h.0.attn.c_attn.weight.resident",
        trace_layer0.c_attn_weight,
        "trace.gpt.gpt.h.0.attn.c_attn.bias.resident",
        trace_layer0.c_attn_bias,
        trace_ln1_resident,
        total_tokens,
        qkv_width,
        width);
    auto trace_attention_row = run_gpt_layer0_attention_core_metal(metal, trace_qkv_row, total_tokens);
    auto trace_attention_resident = run_gpt_layer0_attention_core_metal(metal, trace_qkv_resident, total_tokens);
    std::vector<float> trace_attn_proj_row(static_cast<size_t>(total_tokens) * width);
    for (uint32_t t = 0; t < total_tokens; ++t) {
        std::vector<float> row(trace_attention_row.begin() + static_cast<size_t>(t) * width,
                               trace_attention_row.begin() + static_cast<size_t>(t + 1) * width);
        auto projected = metal.linear_f32(trace_layer0.attn_proj_weight, trace_layer0.attn_proj_bias, row, width, width);
        std::copy(projected.begin(), projected.end(), trace_attn_proj_row.begin() + static_cast<size_t>(t) * width);
    }
    auto trace_attn_proj_resident = metal.linear_rows_f32_resident(
        "trace.gpt.gpt.h.0.attn.c_proj.weight.resident",
        trace_layer0.attn_proj_weight,
        "trace.gpt.gpt.h.0.attn.c_proj.bias.resident",
        trace_layer0.attn_proj_bias,
        trace_attention_resident,
        total_tokens,
        width,
        width);
    auto trace_attn_residual_row = metal.add_f32(hidden_got, trace_attn_proj_row);
    auto trace_attn_residual_resident = metal.add_f32(hidden_got, trace_attn_proj_resident);
    std::vector<float> trace_ln2_row(static_cast<size_t>(total_tokens) * width);
    std::vector<float> trace_fc_row(static_cast<size_t>(total_tokens) * mlp_width);
    for (uint32_t t = 0; t < total_tokens; ++t) {
        std::vector<float> row(trace_attn_residual_row.begin() + static_cast<size_t>(t) * width,
                               trace_attn_residual_row.begin() + static_cast<size_t>(t + 1) * width);
        auto normed = metal.layernorm_f32(row, trace_layer0.ln2_weight, trace_layer0.ln2_bias, 1e-5f);
        std::copy(normed.begin(), normed.end(), trace_ln2_row.begin() + static_cast<size_t>(t) * width);
        auto hidden = metal.linear_f32(trace_layer0.c_fc_weight, trace_layer0.c_fc_bias, normed, mlp_width, width);
        hidden = metal.gelu_f32(hidden);
        std::copy(hidden.begin(), hidden.end(), trace_fc_row.begin() + static_cast<size_t>(t) * mlp_width);
    }
    auto trace_ln2_resident = metal.layernorm_rows_f32_resident(
        "trace.gpt.gpt.h.0.ln_2.weight.resident",
        trace_layer0.ln2_weight,
        "trace.gpt.gpt.h.0.ln_2.bias.resident",
        trace_layer0.ln2_bias,
        trace_attn_residual_resident,
        total_tokens,
        width,
        1e-5f);
    auto trace_fc_resident = metal.linear_rows_f32_resident(
        "trace.gpt.gpt.h.0.mlp.c_fc.weight.resident",
        trace_layer0.c_fc_weight,
        "trace.gpt.gpt.h.0.mlp.c_fc.bias.resident",
        trace_layer0.c_fc_bias,
        trace_ln2_resident,
        total_tokens,
        mlp_width,
        width);
    trace_fc_resident = metal.gelu_f32(trace_fc_resident);
    std::vector<float> trace_mlp_proj_row(static_cast<size_t>(total_tokens) * width);
    for (uint32_t t = 0; t < total_tokens; ++t) {
        std::vector<float> row(trace_fc_row.begin() + static_cast<size_t>(t) * mlp_width,
                               trace_fc_row.begin() + static_cast<size_t>(t + 1) * mlp_width);
        auto projected = metal.linear_f32(trace_layer0.mlp_proj_weight, trace_layer0.mlp_proj_bias, row, width, mlp_width);
        std::copy(projected.begin(), projected.end(), trace_mlp_proj_row.begin() + static_cast<size_t>(t) * width);
    }
    auto trace_mlp_proj_resident = metal.linear_rows_f32_resident(
        "trace.gpt.gpt.h.0.mlp.c_proj.weight.resident",
        trace_layer0.mlp_proj_weight,
        "trace.gpt.gpt.h.0.mlp.c_proj.bias.resident",
        trace_layer0.mlp_proj_bias,
        trace_fc_resident,
        total_tokens,
        width,
        mlp_width);
    auto trace_block_row = metal.add_f32(trace_attn_residual_row, trace_mlp_proj_row);
    auto trace_block_resident = metal.add_f32(trace_attn_residual_resident, trace_mlp_proj_resident);

    std::cout << "  \"resident_layer0_detail\": {\n";
    std::cout << "    \"ln1_vs_row_metal_max_abs_error\": " << max_abs_error(trace_ln1_resident, trace_ln1_row) << ",\n";
    std::cout << "    \"qkv_vs_row_metal_max_abs_error\": " << max_abs_error(trace_qkv_resident, trace_qkv_row) << ",\n";
    std::cout << "    \"attention_vs_row_metal_max_abs_error\": " << max_abs_error(trace_attention_resident, trace_attention_row) << ",\n";
    std::cout << "    \"attn_proj_vs_row_metal_max_abs_error\": " << max_abs_error(trace_attn_proj_resident, trace_attn_proj_row) << ",\n";
    std::cout << "    \"attn_residual_vs_row_metal_max_abs_error\": " << max_abs_error(trace_attn_residual_resident, trace_attn_residual_row) << ",\n";
    std::cout << "    \"ln2_vs_row_metal_max_abs_error\": " << max_abs_error(trace_ln2_resident, trace_ln2_row) << ",\n";
    std::cout << "    \"fc_gelu_vs_row_metal_max_abs_error\": " << max_abs_error(trace_fc_resident, trace_fc_row) << ",\n";
    std::cout << "    \"mlp_proj_vs_row_metal_max_abs_error\": " << max_abs_error(trace_mlp_proj_resident, trace_mlp_proj_row) << ",\n";
    std::cout << "    \"block_vs_row_metal_max_abs_error\": " << max_abs_error(trace_block_resident, trace_block_row) << "\n";
    std::cout << "  },\n";
    std::cout << "  \"layers\": [\n";
    std::cout.flush();

    for (uint32_t layer = 0; layer < 24; ++layer) {
        hidden_ref = run_gpt_transformer_block_cpu_layer(bundle, hidden_ref, total_tokens, layer);
        hidden_got = run_gpt_transformer_block_metal_layer(metal, bundle, hidden_got, total_tokens, layer);
        hidden_resident = run_gpt_transformer_block_metal_layer(metal, resident_weights[layer], hidden_resident, total_tokens);
        const float layer_err = max_abs_error(hidden_got, hidden_ref);
        const float resident_layer_err = max_abs_error(hidden_resident, hidden_ref);
        const float resident_vs_row_err = max_abs_error(hidden_resident, hidden_got);
        max_err = std::max({max_err, layer_err, resident_layer_err});
        std::cout << "    {\"layer\": " << layer
                  << ", \"hidden_max_abs_error\": " << layer_err
                  << ", \"resident_hidden_max_abs_error\": " << resident_layer_err
                  << ", \"resident_vs_row_metal_max_abs_error\": " << resident_vs_row_err
                  << ", \"ref_nonfinite\": " << count_nonfinite(hidden_ref)
                  << ", \"metal_nonfinite\": " << count_nonfinite(hidden_got)
                  << ", \"resident_nonfinite\": " << count_nonfinite(hidden_resident)
                  << "}" << (layer + 1 < 24 ? "," : "") << "\n";
        std::cout.flush();
    }
    std::cout << "  ],\n";

    auto ln_weight = tensor_as_f32(bundle, "gpt.gpt.ln_f.weight");
    auto ln_bias = tensor_as_f32(bundle, "gpt.gpt.ln_f.bias");
    std::vector<float> ln_ref(static_cast<size_t>(total_tokens) * width);
    std::vector<float> ln_got(static_cast<size_t>(total_tokens) * width);
    for (uint32_t t = 0; t < total_tokens; ++t) {
        std::vector<float> row_ref(hidden_ref.begin() + static_cast<size_t>(t) * width,
                                   hidden_ref.begin() + static_cast<size_t>(t + 1) * width);
        auto normed_ref = cpu_layernorm(row_ref, ln_weight, ln_bias, 1e-5f);
        std::copy(normed_ref.begin(), normed_ref.end(), ln_ref.begin() + static_cast<size_t>(t) * width);

        std::vector<float> row_got(hidden_got.begin() + static_cast<size_t>(t) * width,
                                   hidden_got.begin() + static_cast<size_t>(t + 1) * width);
        auto normed_got = metal.layernorm_f32(row_got, ln_weight, ln_bias, 1e-5f);
        std::copy(normed_got.begin(), normed_got.end(), ln_got.begin() + static_cast<size_t>(t) * width);
    }
    const float ln_err = max_abs_error(ln_got, ln_ref);
    max_err = std::max(max_err, ln_err);

    auto norm_weight = tensor_as_f32(bundle, "gpt.final_norm.weight");
    auto norm_bias = tensor_as_f32(bundle, "gpt.final_norm.bias");
    std::vector<float> latent_ref(static_cast<size_t>(codes.size()) * width);
    std::vector<float> latent_got(static_cast<size_t>(codes.size()) * width);
    for (uint32_t t = 0; t < codes.size(); ++t) {
        std::vector<float> row_ref(ln_ref.begin() + static_cast<size_t>(mel_offset + t) * width,
                                   ln_ref.begin() + static_cast<size_t>(mel_offset + t + 1) * width);
        auto normed_ref = cpu_layernorm(row_ref, norm_weight, norm_bias, 1e-5f);
        std::copy(normed_ref.begin(), normed_ref.end(), latent_ref.begin() + static_cast<size_t>(t) * width);

        std::vector<float> row_got(ln_got.begin() + static_cast<size_t>(mel_offset + t) * width,
                                   ln_got.begin() + static_cast<size_t>(mel_offset + t + 1) * width);
        auto normed_got = metal.layernorm_f32(row_got, norm_weight, norm_bias, 1e-5f);
        std::copy(normed_got.begin(), normed_got.end(), latent_got.begin() + static_cast<size_t>(t) * width);
    }
    const float latent_err = max_abs_error(latent_got, latent_ref);
    const float cpu_golden_err = max_abs_error(latent_ref, golden_latent);
    max_err = std::max(max_err, latent_err);
    auto latent_full_got = run_gpt_latent_forward_metal(metal, bundle, conds, text_ids, codes);
    const float latent_full_err = max_abs_error(latent_full_got, latent_ref);
    const float latent_full_vs_interleaved_err = max_abs_error(latent_full_got, latent_got);
    max_err = std::max(max_err, latent_full_err);
    std::cout << "  \"ln_f_max_abs_error\": " << ln_err << ",\n";
    std::cout << "  \"latent_interleaved_metal_max_abs_error\": " << latent_err << ",\n";
    std::cout << "  \"latent_full_metal_max_abs_error\": " << latent_full_err << ",\n";
    std::cout << "  \"latent_full_vs_interleaved_max_abs_error\": " << latent_full_vs_interleaved_err << ",\n";
    std::cout << "  \"latent_cpu_vs_golden_max_abs_error\": " << cpu_golden_err << ",\n";
    std::cout << "  \"max_abs_error\": " << max_err << "\n";
    std::cout << "}\n";
    return count_nonfinite(hidden_ref) == 0 && count_nonfinite(hidden_got) == 0 &&
           count_nonfinite(ln_ref) == 0 && count_nonfinite(ln_got) == 0 &&
           count_nonfinite(latent_ref) == 0 && count_nonfinite(latent_got) == 0;
}

bool run_length_regulator_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t in_dim = 1024;
    constexpr uint32_t out_dim = 512;
    auto s_infer = read_raw_f32(golden_dir + "/s_infer.f32");
    auto target_lengths = read_raw_u32(golden_dir + "/target_lengths.u32");
    auto golden = read_raw_f32(golden_dir + "/length_regulator.f32");
    if (s_infer.empty() || (s_infer.size() % in_dim) != 0) {
        throw std::runtime_error("length regulator golden s_infer must have shape [tokens,1024]");
    }
    if (target_lengths.size() != 1 || target_lengths[0] == 0) {
        throw std::runtime_error("length regulator golden target_lengths must contain one positive length");
    }
    if (golden.size() != static_cast<size_t>(target_lengths[0]) * out_dim) {
        throw std::runtime_error("length regulator golden output size mismatch");
    }
    const uint32_t in_tokens = static_cast<uint32_t>(s_infer.size() / in_dim);
    const uint32_t out_tokens = target_lengths[0];
    auto ref = run_length_regulator_full_cpu(bundle, s_infer, in_tokens, out_tokens);
    auto got = run_length_regulator_full_metal(metal, bundle, s_infer, in_tokens, out_tokens);
    const float cpu_golden_err = max_abs_error(ref, golden);
    const float metal_err = max_abs_error(got, ref);
    const float err = std::max(cpu_golden_err, metal_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_length_regulator_golden\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"input_tokens\": " << in_tokens << ",\n";
    std::cout << "  \"output_tokens\": " << out_tokens << ",\n";
    std::cout << "  \"input_dim\": " << in_dim << ",\n";
    std::cout << "  \"output_dim\": " << out_dim << ",\n";
    std::cout << "  \"cpu_vs_golden_max_abs_error\": " << cpu_golden_err << ",\n";
    std::cout << "  \"metal_max_abs_error\": " << metal_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return cpu_golden_err <= 3e-3f && metal_err <= 3e-3f;
}

bool export_length_regulator_golden(const std::string& bundle_dir, const std::string& golden_dir, const std::string& output_path) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t in_dim = 1024;
    constexpr uint32_t out_dim = 512;
    auto s_infer = read_raw_f32(golden_dir + "/s_infer.f32");
    auto target_lengths = read_raw_u32(golden_dir + "/target_lengths.u32");
    auto golden = read_raw_f32(golden_dir + "/length_regulator.f32");
    if (s_infer.empty() || (s_infer.size() % in_dim) != 0) {
        throw std::runtime_error("length regulator export s_infer must have shape [tokens,1024]");
    }
    if (target_lengths.size() != 1 || target_lengths[0] == 0) {
        throw std::runtime_error("length regulator export target_lengths must contain one positive length");
    }
    const uint32_t in_tokens = static_cast<uint32_t>(s_infer.size() / in_dim);
    const uint32_t out_tokens = target_lengths[0];
    auto got = run_length_regulator_full_metal(metal, bundle, s_infer, in_tokens, out_tokens);
    if (got.size() != static_cast<size_t>(out_tokens) * out_dim) {
        throw std::runtime_error("length regulator export output size mismatch");
    }
    write_raw_f32(output_path, got);
    float golden_err = 0.0f;
    if (golden.size() == got.size()) {
        golden_err = max_abs_error(got, golden);
    }
    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_length_regulator_export\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"output_f32\": \"" << output_path << "\",\n";
    std::cout << "  \"input_tokens\": " << in_tokens << ",\n";
    std::cout << "  \"output_tokens\": " << out_tokens << ",\n";
    std::cout << "  \"golden_max_abs_error\": " << golden_err << "\n";
    std::cout << "}\n";
    return true;
}

bool export_length_regulator_stages_golden(const std::string& bundle_dir, const std::string& golden_dir, const std::string& output_dir) {
    mit2::Bundle bundle(bundle_dir);
    constexpr uint32_t in_dim = 1024;
    constexpr uint32_t out_dim = 512;
    auto s_infer = read_raw_f32(golden_dir + "/s_infer.f32");
    auto target_lengths = read_raw_u32(golden_dir + "/target_lengths.u32");
    if (s_infer.empty() || (s_infer.size() % in_dim) != 0) {
        throw std::runtime_error("length regulator stage export s_infer must have shape [tokens,1024]");
    }
    if (target_lengths.size() != 1 || target_lengths[0] == 0) {
        throw std::runtime_error("length regulator stage export target_lengths must contain one positive length");
    }
    const uint32_t in_tokens = static_cast<uint32_t>(s_infer.size() / in_dim);
    const uint32_t out_tokens = target_lengths[0];
    std::filesystem::create_directories(output_dir);
    auto projected = [&]() {
        auto weight = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.weight");
        auto bias = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.bias");
        std::vector<float> out(static_cast<size_t>(in_tokens) * out_dim);
        for (uint32_t t = 0; t < in_tokens; ++t) {
            std::vector<float> row(s_infer.begin() + static_cast<size_t>(t) * in_dim,
                                   s_infer.begin() + static_cast<size_t>(t + 1) * in_dim);
            auto y = cpu_linear(weight, bias, row, out_dim, in_dim);
            std::copy(y.begin(), y.end(), out.begin() + static_cast<size_t>(t) * out_dim);
        }
        return out;
    }();
    write_raw_f32(output_dir + "/00_projected.f32", projected);
    auto x = cpu_nearest_interpolate(projected, in_tokens, out_tokens, out_dim);
    write_raw_f32(output_dir + "/01_interpolated.f32", x);
    const int conv_indices[] = {0, 3, 6, 9};
    const int norm_indices[] = {1, 4, 7, 10};
    for (int block = 0; block < 4; ++block) {
        const std::string conv = "s2mel.net.length_regulator.model." + std::to_string(conv_indices[block]);
        const std::string norm = "s2mel.net.length_regulator.model." + std::to_string(norm_indices[block]);
        x = cpu_conv1d_same(x, tensor_as_f32(bundle, conv + ".weight"), tensor_as_f32(bundle, conv + ".bias"), out_tokens, out_dim, out_dim, 3);
        write_raw_f32(output_dir + "/" + std::to_string(block + 2) + "0_conv.f32", x);
        x = cpu_groupnorm1(x, tensor_as_f32(bundle, norm + ".weight"), tensor_as_f32(bundle, norm + ".bias"), out_tokens, out_dim, 1e-5f);
        write_raw_f32(output_dir + "/" + std::to_string(block + 2) + "1_groupnorm.f32", x);
        x = cpu_mish(x);
        write_raw_f32(output_dir + "/" + std::to_string(block + 2) + "2_mish.f32", x);
    }
    x = cpu_conv1d_same(
        x,
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.weight"),
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.bias"),
        out_tokens,
        out_dim,
        out_dim,
        1);
    write_raw_f32(output_dir + "/60_final.f32", x);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_length_regulator_stage_export\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"output_dir\": \"" << output_dir << "\",\n";
    std::cout << "  \"input_tokens\": " << in_tokens << ",\n";
    std::cout << "  \"output_tokens\": " << out_tokens << "\n";
    std::cout << "}\n";
    return true;
}

std::vector<float> semantic_out_project_weight(const mit2::Bundle& bundle) {
    auto g = tensor_as_f32(bundle, "semantic_codec.quantizer.quantizers.0.out_project.weight_g");
    auto v = tensor_as_f32(bundle, "semantic_codec.quantizer.quantizers.0.out_project.weight_v");
    std::vector<float> weight(1024 * 8);
    for (uint32_t row = 0; row < 1024; ++row) {
        float norm = 0.0f;
        for (uint32_t col = 0; col < 8; ++col) {
            const float value = v[row * 8 + col];
            norm += value * value;
        }
        norm = std::sqrt(norm);
        const float scale = g[row] / norm;
        for (uint32_t col = 0; col < 8; ++col) {
            weight[row * 8 + col] = v[row * 8 + col] * scale;
        }
    }
    return weight;
}

std::vector<float> weight_norm_rowmajor(const std::vector<float>& g, const std::vector<float>& v, uint32_t rows, uint32_t cols) {
    std::vector<float> weight(static_cast<size_t>(rows) * cols);
    for (uint32_t row = 0; row < rows; ++row) {
        float norm = 0.0f;
        for (uint32_t col = 0; col < cols; ++col) {
            const float value = v[static_cast<size_t>(row) * cols + col];
            norm += value * value;
        }
        norm = std::sqrt(norm);
        const float scale = g[row] / norm;
        for (uint32_t col = 0; col < cols; ++col) {
            weight[static_cast<size_t>(row) * cols + col] = v[static_cast<size_t>(row) * cols + col] * scale;
        }
    }
    return weight;
}

std::vector<float> weight_norm_conv_weight(const mit2::Bundle& bundle, const std::string& prefix, uint32_t out_channels, uint32_t in_channels, uint32_t kernel) {
    auto g = tensor_as_f32(bundle, prefix + ".weight_g");
    auto v = tensor_as_f32(bundle, prefix + ".weight_v");
    return weight_norm_rowmajor(g, v, out_channels, in_channels * kernel);
}

std::vector<float> weight_norm_conv_transpose_weight(const mit2::Bundle& bundle, const std::string& prefix, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    auto g = tensor_as_f32(bundle, prefix + ".weight_g");
    auto v = tensor_as_f32(bundle, prefix + ".weight_v");
    return weight_norm_rowmajor(g, v, in_channels, out_channels * kernel);
}

std::vector<float> run_vq2emb_cpu(const mit2::Bundle& bundle, const std::vector<uint32_t>& codes) {
    auto codebook = tensor_as_f32(bundle, "semantic_codec.quantizer.quantizers.0.codebook.weight");
    auto weight = semantic_out_project_weight(bundle);
    auto bias = tensor_as_f32(bundle, "semantic_codec.quantizer.quantizers.0.out_project.bias");
    std::vector<float> out(codes.size() * 1024);
    for (size_t t = 0; t < codes.size(); ++t) {
        std::vector<float> emb(8);
        for (uint32_t col = 0; col < 8; ++col) {
            emb[col] = codebook[codes[t] * 8 + col];
        }
        auto projected = cpu_linear(weight, bias, emb, 1024, 8);
        std::copy(projected.begin(), projected.end(), out.begin() + t * 1024);
    }
    return out;
}

std::vector<float> run_vq2emb_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<uint32_t>& codes) {
    auto codebook = tensor_as_f32(bundle, "semantic_codec.quantizer.quantizers.0.codebook.weight");
    auto weight = semantic_out_project_weight(bundle);
    auto bias = tensor_as_f32(bundle, "semantic_codec.quantizer.quantizers.0.out_project.bias");
    auto embeddings = metal.embedding_f32_resident(
        "semantic_codec.quantizer.quantizers.0.codebook.weight.resident",
        codebook,
        codes,
        8);
    return metal.linear_rows_f32_resident(
        "semantic_codec.quantizer.quantizers.0.out_project.weight.resident",
        weight,
        "semantic_codec.quantizer.quantizers.0.out_project.bias.resident",
        bias,
        embeddings,
        static_cast<uint32_t>(codes.size()),
        1024,
        8);
}

struct SemanticQuantizeResult {
    std::vector<float> sref;
    std::vector<uint32_t> codes;
};

SemanticQuantizeResult run_semantic_codec_quantize_cpu(const mit2::Bundle& bundle,
                                                       const std::vector<float>& spk_cond,
                                                       uint32_t spk_tokens) {
    const std::string prefix = "semantic_codec.quantizer.quantizers.0";
    const auto in_weight = weight_norm_conv_weight(bundle, prefix + ".in_project", 8, 1024, 1);
    const auto in_bias = tensor_as_f32(bundle, prefix + ".in_project.bias");
    const auto codebook = tensor_as_f32(bundle, prefix + ".codebook.weight");
    const auto out_weight = semantic_out_project_weight(bundle);
    const auto out_bias = tensor_as_f32(bundle, prefix + ".out_project.bias");

    SemanticQuantizeResult result;
    result.sref.resize(static_cast<size_t>(spk_tokens) * 1024u);
    result.codes.resize(spk_tokens);
    std::vector<float> latent(8);
    std::vector<float> quantized(8);
    for (uint32_t t = 0; t < spk_tokens; ++t) {
        const float* row = spk_cond.data() + static_cast<size_t>(t) * 1024u;
        for (uint32_t out = 0; out < 8; ++out) {
            float acc = in_bias[out];
            for (uint32_t in = 0; in < 1024; ++in) {
                acc += in_weight[static_cast<size_t>(out) * 1024u + in] * row[in];
            }
            latent[out] = acc;
        }

        uint32_t best_code = 0;
        float best_dist = std::numeric_limits<float>::infinity();
        for (uint32_t code = 0; code < 8192; ++code) {
            float dist = 0.0f;
            for (uint32_t dim = 0; dim < 8; ++dim) {
                const float delta = latent[dim] - codebook[static_cast<size_t>(code) * 8u + dim];
                dist += delta * delta;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_code = code;
            }
        }
        result.codes[t] = best_code;
        for (uint32_t dim = 0; dim < 8; ++dim) {
            quantized[dim] = codebook[static_cast<size_t>(best_code) * 8u + dim];
        }
        const auto projected = cpu_linear(out_weight, out_bias, quantized, 1024, 8);
        std::copy(projected.begin(), projected.end(), result.sref.begin() + static_cast<size_t>(t) * 1024u);
    }
    return result;
}

SemanticQuantizeResult run_semantic_codec_quantize_metal(mit2::MetalContext& metal,
                                                         const mit2::Bundle& bundle,
                                                         const std::vector<float>& spk_cond,
                                                         uint32_t spk_tokens) {
    const std::string prefix = "semantic_codec.quantizer.quantizers.0";
    const auto in_weight = weight_norm_conv_weight(bundle, prefix + ".in_project", 8, 1024, 1);
    const auto in_bias = tensor_as_f32(bundle, prefix + ".in_project.bias");
    const auto codebook = tensor_as_f32(bundle, prefix + ".codebook.weight");
    const auto out_weight = semantic_out_project_weight(bundle);
    const auto out_bias = tensor_as_f32(bundle, prefix + ".out_project.bias");
    SemanticQuantizeResult result;
    result.sref = metal.semantic_quantize_f32_resident(prefix + ".in_project.weight.resident",
                                                       in_weight,
                                                       prefix + ".in_project.bias.resident",
                                                       in_bias,
                                                       prefix + ".codebook.weight.resident",
                                                       codebook,
                                                       prefix + ".out_project.weight.resident",
                                                       out_weight,
                                                       prefix + ".out_project.bias.resident",
                                                       out_bias,
                                                       spk_cond,
                                                       spk_tokens,
                                                       result.codes);
    return result;
}

bool run_vq2emb_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    std::vector<uint32_t> codes{0, 52, 8191, 17, 2048, 4096, 777};
    auto ref = run_vq2emb_cpu(bundle, codes);
    auto got = run_vq2emb_metal(metal, bundle, codes);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"semantic_codec_vq2emb\",\n";
    std::cout << "  \"tokens\": " << codes.size() << ",\n";
    std::cout << "  \"codebook_dim\": 8,\n";
    std::cout << "  \"output_dim\": 1024,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-5f;
}

std::vector<float> run_length_regulator_front_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens) {
    auto weight = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.weight");
    auto bias = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.bias");
    std::vector<float> projected(static_cast<size_t>(in_tokens) * 512);
    for (uint32_t t = 0; t < in_tokens; ++t) {
        std::vector<float> row(input.begin() + t * 1024, input.begin() + (t + 1) * 1024);
        auto out = cpu_linear(weight, bias, row, 512, 1024);
        std::copy(out.begin(), out.end(), projected.begin() + t * 512);
    }
    return cpu_nearest_interpolate(projected, in_tokens, out_tokens, 512);
}

std::vector<float> run_length_regulator_front_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens) {
    auto weight = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.weight");
    auto bias = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.bias");
    auto projected = metal.linear_rows_f32_resident(
        "s2mel.net.length_regulator.content_in_proj.weight.resident",
        weight,
        "s2mel.net.length_regulator.content_in_proj.bias.resident",
        bias,
        input,
        in_tokens,
        512,
        1024);
    return metal.nearest_interpolate_f32(projected, in_tokens, out_tokens, 512);
}

std::vector<float> run_length_regulator_full_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens) {
    auto x = run_length_regulator_front_cpu(bundle, input, in_tokens, out_tokens);
    const int conv_indices[] = {0, 3, 6, 9};
    const int norm_indices[] = {1, 4, 7, 10};
    for (int block = 0; block < 4; ++block) {
        const std::string conv = "s2mel.net.length_regulator.model." + std::to_string(conv_indices[block]);
        const std::string norm = "s2mel.net.length_regulator.model." + std::to_string(norm_indices[block]);
        x = cpu_conv1d_same(x, tensor_as_f32(bundle, conv + ".weight"), tensor_as_f32(bundle, conv + ".bias"), out_tokens, 512, 512, 3);
        x = cpu_groupnorm1(x, tensor_as_f32(bundle, norm + ".weight"), tensor_as_f32(bundle, norm + ".bias"), out_tokens, 512, 1e-5f);
        x = cpu_mish(x);
    }
    return cpu_conv1d_same(
        x,
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.weight"),
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.bias"),
        out_tokens,
        512,
        512,
        1);
}

std::vector<float> run_length_regulator_full_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens) {
    auto x = run_length_regulator_front_metal(metal, bundle, input, in_tokens, out_tokens);
    const int conv_indices[] = {0, 3, 6, 9};
    const int norm_indices[] = {1, 4, 7, 10};
    for (int block = 0; block < 4; ++block) {
        const std::string conv = "s2mel.net.length_regulator.model." + std::to_string(conv_indices[block]);
        const std::string norm = "s2mel.net.length_regulator.model." + std::to_string(norm_indices[block]);
        x = metal.conv1d_same_f32_resident(
            conv + ".weight.resident",
            tensor_as_f32(bundle, conv + ".weight"),
            conv + ".bias.resident",
            tensor_as_f32(bundle, conv + ".bias"),
            x,
            out_tokens,
            512,
            512,
            3);
        x = metal.groupnorm1_f32_resident(
            norm + ".weight.resident",
            tensor_as_f32(bundle, norm + ".weight"),
            norm + ".bias.resident",
            tensor_as_f32(bundle, norm + ".bias"),
            x,
            out_tokens,
            512,
            1e-5f);
        x = metal.mish_f32(x);
    }
    return metal.conv1d_same_f32_resident(
        "s2mel.net.length_regulator.model.12.weight.resident",
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.weight"),
        "s2mel.net.length_regulator.model.12.bias.resident",
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.bias"),
        x,
        out_tokens,
        512,
        512,
        1);
}

bool run_length_regulator_front_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t in_tokens = 9;
    constexpr uint32_t out_tokens = 16;
    std::vector<float> input(static_cast<size_t>(in_tokens) * 1024);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.019f) * 0.25f + std::cos(static_cast<float>(i % 113) * 0.021f) * 0.5f;
    }
    auto ref = run_length_regulator_front_cpu(bundle, input, in_tokens, out_tokens);
    auto got = run_length_regulator_front_metal(metal, bundle, input, in_tokens, out_tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_length_regulator_front\",\n";
    std::cout << "  \"input_tokens\": " << in_tokens << ",\n";
    std::cout << "  \"output_tokens\": " << out_tokens << ",\n";
    std::cout << "  \"input_dim\": 1024,\n";
    std::cout << "  \"output_dim\": 512,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-5f;
}

bool run_length_regulator_full_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t in_tokens = 7;
    constexpr uint32_t out_tokens = 11;
    std::vector<float> input(static_cast<size_t>(in_tokens) * 1024);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.1f + std::cos(static_cast<float>(i % 131) * 0.013f) * 0.2f;
    }
    auto ref = run_length_regulator_full_cpu(bundle, input, in_tokens, out_tokens);
    auto got = run_length_regulator_full_metal(metal, bundle, input, in_tokens, out_tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_length_regulator_full\",\n";
    std::cout << "  \"input_tokens\": " << in_tokens << ",\n";
    std::cout << "  \"output_tokens\": " << out_tokens << ",\n";
    std::cout << "  \"channels\": 512,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_timestep_embedder_cpu(const mit2::Bundle& bundle, const std::vector<float>& timesteps, const std::string& prefix) {
    auto freqs = tensor_as_f32(bundle, prefix + ".freqs");
    auto w0 = tensor_as_f32(bundle, prefix + ".mlp.0.weight");
    auto b0 = tensor_as_f32(bundle, prefix + ".mlp.0.bias");
    auto w2 = tensor_as_f32(bundle, prefix + ".mlp.2.weight");
    auto b2 = tensor_as_f32(bundle, prefix + ".mlp.2.bias");
    auto emb = cpu_timestep_embedding(timesteps, freqs, 1000.0f);
    std::vector<float> out(timesteps.size() * 512);
    for (size_t b = 0; b < timesteps.size(); ++b) {
        std::vector<float> row(emb.begin() + b * 256, emb.begin() + (b + 1) * 256);
        auto h = cpu_linear(w0, b0, row, 512, 256);
        for (float& v : h) {
            v = v / (1.0f + std::exp(-v));
        }
        auto y = cpu_linear(w2, b2, h, 512, 512);
        std::copy(y.begin(), y.end(), out.begin() + b * 512);
    }
    return out;
}

std::vector<float> run_timestep_embedder_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& timesteps, const std::string& prefix) {
    auto freqs = tensor_as_f32(bundle, prefix + ".freqs");
    auto w0 = tensor_as_f32(bundle, prefix + ".mlp.0.weight");
    auto b0 = tensor_as_f32(bundle, prefix + ".mlp.0.bias");
    auto w2 = tensor_as_f32(bundle, prefix + ".mlp.2.weight");
    auto b2 = tensor_as_f32(bundle, prefix + ".mlp.2.bias");
    auto emb = metal.timestep_embedding_f32(timesteps, freqs, 1000.0f);
    auto h = metal.linear_rows_f32_resident(
        prefix + ".mlp.0.weight.resident",
        w0,
        prefix + ".mlp.0.bias.resident",
        b0,
        emb,
        static_cast<uint32_t>(timesteps.size()),
        512,
        256);
    h = metal.silu_f32(h);
    return metal.linear_rows_f32_resident(
        prefix + ".mlp.2.weight.resident",
        w2,
        prefix + ".mlp.2.bias.resident",
        b2,
        h,
        static_cast<uint32_t>(timesteps.size()),
        512,
        512);
}

bool run_timestep_embedder_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    std::vector<float> timesteps{0.0f, 0.125f, 0.5f, 0.875f, 1.0f};
    const std::string prefix1 = "s2mel.net.cfm.estimator.t_embedder";
    const std::string prefix2 = "s2mel.net.cfm.estimator.t_embedder2";
    auto ref1 = run_timestep_embedder_cpu(bundle, timesteps, prefix1);
    auto got1 = run_timestep_embedder_metal(metal, bundle, timesteps, prefix1);
    auto ref2 = run_timestep_embedder_cpu(bundle, timesteps, prefix2);
    auto got2 = run_timestep_embedder_metal(metal, bundle, timesteps, prefix2);
    const float err1 = max_abs_error(got1, ref1);
    const float err2 = max_abs_error(got2, ref2);
    const float err = std::max(err1, err2);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_timestep_embedders\",\n";
    std::cout << "  \"batch\": " << timesteps.size() << ",\n";
    std::cout << "  \"embedding_dim\": 512,\n";
    std::cout << "  \"t_embedder_max_abs_error\": " << err1 << ",\n";
    std::cout << "  \"t_embedder2_max_abs_error\": " << err2 << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

