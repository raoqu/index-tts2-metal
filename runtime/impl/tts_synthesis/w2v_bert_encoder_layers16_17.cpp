// w2v-BERT conformer encoder: layers 16-17 and final norm (CPU + Metal), plus the remaining w2v clone CLI entry points.
// Part of the tts_synthesis translation unit (see ../tts_synthesis.cpp);
// these files are #included in order into a single TU, so definition order
// across files is significant.

std::vector<float> run_w2v_bert_layer16_ffn1_residual_cpu(const mit2::Bundle& model,
                                                         const std::vector<float>& layer15_residual,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.output_dense.bias");
    if (tokens == 0 || layer15_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_ffn1_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(layer15_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = layer15_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer16_ffn1_residual_metal(mit2::MetalContext& metal,
                                                           const mit2::Bundle& model,
                                                           const std::vector<float>& layer15_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn1.output_dense.bias");
    if (tokens == 0 || layer15_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_ffn1_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.16.ffn1_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.16.ffn1_layer_norm.bias.resident",
        norm_b,
        layer15_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.16.ffn1.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.16.ffn1.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.16.ffn1.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.16.ffn1.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(layer15_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_ffn1_residual(const std::string& model_bundle_dir,
                                                             const std::string& layer15_residual_f32,
                                                             uint32_t tokens,
                                                             const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer15_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer15_ffn2_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_ffn1_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer15_ffn2_residual_f32\": \"" << json_escape(layer15_residual_f32) << "\",\n";
        std::cout << "  \"output_layer16_ffn1_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_ffn1_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer15_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer15_ffn2_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer16_ffn1_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_ffn1_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_ffn1_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_ffn1_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 self-attention/convolution/ffn2 and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer15_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer16_ffn1_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer16_ffn1_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

W2VLayerQkv run_w2v_bert_layer16_qkv_cpu(const mit2::Bundle& model,
                                        const std::vector<float>& layer16_ffn1_residual,
                                        uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_v.weight");
    if (tokens == 0 || layer16_ffn1_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        q_w.size() != static_cast<size_t>(1024) * 1024 ||
        k_w.size() != static_cast<size_t>(1024) * 1024 ||
        v_w.size() != static_cast<size_t>(1024) * 1024) {
        throw std::invalid_argument("w2v_bert_layer16_qkv invalid input sizes");
    }
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_v.bias");
    return {
        cpu_linear_rows(q_w, q_b, layer16_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(k_w, k_b, layer16_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(v_w, v_b, layer16_ffn1_residual, tokens, 1024, 1024),
    };
}

W2VLayerQkv run_w2v_bert_layer16_qkv_metal(mit2::MetalContext& metal,
                                          const mit2::Bundle& model,
                                          const std::vector<float>& layer16_ffn1_residual,
                                          uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_v.weight");
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_v.bias");
    return {
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.16.self_attn.linear_q.weight.resident",
                                       q_w,
                                       "w2v_bert.encoder.layers.16.self_attn.linear_q.bias.resident",
                                       q_b,
                                       layer16_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.16.self_attn.linear_k.weight.resident",
                                       k_w,
                                       "w2v_bert.encoder.layers.16.self_attn.linear_k.bias.resident",
                                       k_b,
                                       layer16_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.16.self_attn.linear_v.weight.resident",
                                       v_w,
                                       "w2v_bert.encoder.layers.16.self_attn.linear_v.bias.resident",
                                       v_b,
                                       layer16_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
    };
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_qkv(const std::string& model_bundle_dir,
                                                  const std::string& layer16_ffn1_residual_f32,
                                                  uint32_t tokens,
                                                  const std::string& output_dir) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer16_ffn1_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_ffn1_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    const auto dir = std::filesystem::path(output_dir);
    const auto output_q = (dir / "w2v_layer16_q.f32").string();
    const auto output_k = (dir / "w2v_layer16_k.f32").string();
    const auto output_v = (dir / "w2v_layer16_v.f32").string();
    const auto output_manifest = (dir / "w2v_layer16_qkv.manifest.json").string();

    auto print_failure = [&](const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_qkv\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_ffn1_residual_f32\": \"" << json_escape(layer16_ffn1_residual_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_qkv_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_qkv\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_qkv\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 attention scores/context, convolution, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_failure(issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer16_ffn1_residual_f32);
        std::string execution_backend = "metal";
        W2VLayerQkv qkv;
        try {
            mit2::MetalContext metal;
            qkv = run_w2v_bert_layer16_qkv_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            qkv = run_w2v_bert_layer16_qkv_cpu(model, input, tokens);
        }

        std::filesystem::create_directories(dir);
        write_raw_f32(output_q, qkv.q);
        write_raw_f32(output_k, qkv.k);
        write_raw_f32(output_v, qkv.v);
        std::ostringstream manifest_json;
        manifest_json << "{\n";
        manifest_json << "  \"format\": \"mit2-w2v-layer16-qkv-sidecars\",\n";
        manifest_json << "  \"version\": 1,\n";
        manifest_json << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        manifest_json << "  \"w2v_layer16_ffn1_residual_f32\": \"" << json_escape(layer16_ffn1_residual_f32) << "\",\n";
        manifest_json << "  \"w2v_tokens\": " << tokens << ",\n";
        manifest_json << "  \"output_q_f32\": \"" << json_escape(output_q) << "\",\n";
        manifest_json << "  \"output_q_sha256\": \"" << file_sha256_hex(output_q) << "\",\n";
        manifest_json << "  \"output_k_f32\": \"" << json_escape(output_k) << "\",\n";
        manifest_json << "  \"output_k_sha256\": \"" << file_sha256_hex(output_k) << "\",\n";
        manifest_json << "  \"output_v_f32\": \"" << json_escape(output_v) << "\",\n";
        manifest_json << "  \"output_v_sha256\": \"" << file_sha256_hex(output_v) << "\",\n";
        manifest_json << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        manifest_json << "  \"ready_native_w2v_bert_layer16_qkv\": true,\n";
        manifest_json << "  \"ready_native_w2v_bert_semantic_features\": false\n";
        manifest_json << "}\n";
        write_text_file(output_manifest, manifest_json.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_qkv\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_ffn1_residual_f32\": \"" << json_escape(layer16_ffn1_residual_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"output_manifest\": \"" << json_escape(output_manifest) << "\",\n";
        std::cout << "  \"output_q_f32\": \"" << json_escape(output_q) << "\",\n";
        std::cout << "  \"output_q_sha256\": \"" << file_sha256_hex(output_q) << "\",\n";
        std::cout << "  \"output_k_f32\": \"" << json_escape(output_k) << "\",\n";
        std::cout << "  \"output_k_sha256\": \"" << file_sha256_hex(output_k) << "\",\n";
        std::cout << "  \"output_v_f32\": \"" << json_escape(output_v) << "\",\n";
        std::cout << "  \"output_v_sha256\": \"" << file_sha256_hex(output_v) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_ffn1_residual_values\": " << input.size() << ",\n";
        std::cout << "  \"q_values\": " << qkv.q.size() << ",\n";
        std::cout << "  \"k_values\": " << qkv.k.size() << ",\n";
        std::cout << "  \"v_values\": " << qkv.v.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_qkv_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_qkv\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_qkv\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 attention scores/context, convolution, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        print_failure({e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_attention_context_cpu(const std::vector<float>& q,
                                                             const std::vector<float>& k,
                                                             const std::vector<float>& v,
                                                             const std::vector<uint32_t>& key_mask,
                                                             uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    if (tokens == 0 || q.size() != static_cast<size_t>(tokens) * heads * head_dim ||
        k.size() != q.size() || v.size() != q.size() || key_mask.size() != tokens) {
        throw std::invalid_argument("w2v_bert_layer16_attention_context invalid input sizes");
    }
    return cpu_cross_attention_heads_masked(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

std::vector<float> run_w2v_bert_layer16_attention_context_metal(mit2::MetalContext& metal,
                                                               const std::vector<float>& q,
                                                               const std::vector<float>& k,
                                                               const std::vector<float>& v,
                                                               const std::vector<uint32_t>& key_mask,
                                                               uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    return metal.cross_attention_heads_masked_f32(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_attention(const std::string& q_f32,
                                                         const std::string& k_f32,
                                                         const std::string& v_f32,
                                                         const std::string& attention_mask_u32,
                                                         uint32_t tokens,
                                                         const std::string& output_context_f32,
                                                         const std::vector<float>& dist_emb) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    const uint64_t qkv_values = static_cast<uint64_t>(tokens) * 1024u;
    append_raw_f32_count_issue(issues, q_f32, qkv_values, "w2v_layer16_q");
    append_raw_f32_count_issue(issues, k_f32, qkv_values, "w2v_layer16_k");
    append_raw_f32_count_issue(issues, v_f32, qkv_values, "w2v_layer16_v");
    append_raw_u32_count_issue(issues, attention_mask_u32, tokens, "w2v_attention_mask");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* q,
                            const std::vector<float>* k,
                            const std::vector<float>* v,
                            const std::vector<uint32_t>* mask,
                            const std::vector<float>* context,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_attention\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer16_q_f32\": \"" << json_escape(q_f32) << "\",\n";
        std::cout << "  \"w2v_layer16_k_f32\": \"" << json_escape(k_f32) << "\",\n";
        std::cout << "  \"w2v_layer16_v_f32\": \"" << json_escape(v_f32) << "\",\n";
        std::cout << "  \"w2v_attention_mask_u32\": \"" << json_escape(attention_mask_u32) << "\",\n";
        std::cout << "  \"output_layer16_context_f32\": \"" << json_escape(output_context_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_context_sha256\": \"" << file_sha256_hex(output_context_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"heads\": 16,\n";
        std::cout << "  \"head_dim\": 64,\n";
        if (q && k && v && mask && context) {
            std::cout << "  \"q_values\": " << q->size() << ",\n";
            std::cout << "  \"k_values\": " << k->size() << ",\n";
            std::cout << "  \"v_values\": " << v->size() << ",\n";
            std::cout << "  \"mask_values\": " << mask->size() << ",\n";
        }
        std::cout << "  \"layer16_context_shape\": \"[1," << tokens << ",1024]\",\n";
        if (context) {
            std::cout << "  \"layer16_context_values\": " << context->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer16_attention_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_attention_context\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_attention_context\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 attention output projection/residual/norm, convolution, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, nullptr, nullptr, nullptr, issues);
        return false;
    }

    try {
        const auto q = read_raw_f32(q_f32);
        const auto k = read_raw_f32(k_f32);
        const auto v = read_raw_f32(v_f32);
        const auto mask = read_raw_u32(attention_mask_u32);
        std::string backend = "cpu_distance_emb";
        const std::vector<float> context = cpu_w2v_bert_cross_attention_with_distance(q, k, v, mask, tokens, dist_emb, 64, 8);
        write_raw_f32(output_context_f32, context);
        print_report(true, backend, &q, &k, &v, &mask, &context, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_attention_project_cpu(const mit2::Bundle& model,
                                                             const std::vector<float>& context,
                                                             uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.16.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_out.bias");
    }
    if (tokens == 0 || context.size() != static_cast<size_t>(tokens) * 1024u ||
        out_w.size() != static_cast<size_t>(1024) * 1024 || out_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_attention_project invalid input sizes");
    }
    return cpu_linear_rows(out_w, out_b, context, tokens, 1024, 1024);
}

std::vector<float> run_w2v_bert_layer16_attention_project_metal(mit2::MetalContext& metal,
                                                               const mit2::Bundle& model,
                                                               const std::vector<float>& context,
                                                               uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.16.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn.linear_out.bias");
    }
    return metal.linear_rows_f32_resident("w2v_bert.encoder.layers.16.self_attn.linear_out.weight.resident",
                                          out_w,
                                          "w2v_bert.encoder.layers.16.self_attn.linear_out.bias.resident_or_zero",
                                          out_b,
                                          context,
                                          tokens,
                                          1024,
                                          1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_attention_project(const std::string& model_bundle_dir,
                                                                 const std::string& context_f32,
                                                                 uint32_t tokens,
                                                                 const std::string& output_attention_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               context_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_attention_context");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_attention_project\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_layer16_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_attention_sha256\": \"" << file_sha256_hex(output_attention_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer16_context_values\": " << input->size() << ",\n";
            std::cout << "  \"layer16_attention_projection_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_attention_project_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_attention_projection\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_attention_projection\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 attention residual, attention norm, convolution, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto context = read_raw_f32(context_f32);
        std::string backend = "metal";
        std::vector<float> projected;
        try {
            mit2::MetalContext metal;
            projected = run_w2v_bert_layer16_attention_project_metal(metal, model, context, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            projected = run_w2v_bert_layer16_attention_project_cpu(model, context, tokens);
        }
        write_raw_f32(output_attention_f32, projected);
        print_report(true, backend, &context, &projected, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_attention_residual_cpu(const std::vector<float>& ffn1_residual,
                                                              const std::vector<float>& attention_projection,
                                                              uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer16_attention_residual invalid input sizes");
    }
    std::vector<float> residual(expected);
    for (size_t i = 0; i < expected; ++i) {
        residual[i] = ffn1_residual[i] + attention_projection[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer16_attention_residual_metal(mit2::MetalContext& metal,
                                                                const std::vector<float>& ffn1_residual,
                                                                const std::vector<float>& attention_projection,
                                                                uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer16_attention_residual invalid input sizes");
    }
    return metal.add_f32(ffn1_residual, attention_projection);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_attention_residual(const std::string& ffn1_residual_f32,
                                                                  const std::string& attention_f32,
                                                                  uint32_t tokens,
                                                                  const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               ffn1_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_ffn1_residual");
    append_raw_f32_count_issue(issues,
                               attention_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_attention_projection");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* ffn1_residual,
                            const std::vector<float>* attention_projection,
                            const std::vector<float>* residual,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_attention_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer16_ffn1_residual_f32\": \"" << json_escape(ffn1_residual_f32) << "\",\n";
        std::cout << "  \"w2v_layer16_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_layer16_attention_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_attention_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (ffn1_residual && attention_projection && residual) {
            std::cout << "  \"layer16_ffn1_residual_values\": " << ffn1_residual->size() << ",\n";
            std::cout << "  \"layer16_attention_projection_values\": " << attention_projection->size() << ",\n";
            std::cout << "  \"layer16_attention_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer16_attention_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_attention_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_attention_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 attention norm, convolution, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, nullptr, issues);
        return false;
    }

    try {
        const auto ffn1_residual = read_raw_f32(ffn1_residual_f32);
        const auto attention_projection = read_raw_f32(attention_f32);
        std::string backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer16_attention_residual_metal(metal,
                                                                    ffn1_residual,
                                                                    attention_projection,
                                                                    tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer16_attention_residual_cpu(ffn1_residual, attention_projection, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &ffn1_residual, &attention_projection, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_attention_norm_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& attention_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn_layer_norm.bias");
    if (tokens == 0 || attention_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_attention_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer16_attention_norm_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& attention_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.self_attn_layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.16.self_attn_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.16.self_attn_layer_norm.bias.resident",
                                             norm_b,
                                             attention_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_attention_norm(const std::string& model_bundle_dir,
                                                              const std::string& attention_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_norm_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_attention_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_attention_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_layer16_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_attention_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer16_attention_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer16_attention_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_attention_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_attention_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_attention_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 convolution, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_residual = read_raw_f32(attention_residual_f32);
        std::string backend = "metal";
        std::vector<float> normed;
        try {
            mit2::MetalContext metal;
            normed = run_w2v_bert_layer16_attention_norm_metal(metal, model, attention_residual, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer16_attention_norm_cpu(model, attention_residual, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_residual, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_conv_norm_cpu(const mit2::Bundle& model,
                                                     const std::vector<float>& attention_norm,
                                                     uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.layer_norm.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_conv_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_norm, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer16_conv_norm_metal(mit2::MetalContext& metal,
                                                       const mit2::Bundle& model,
                                                       const std::vector<float>& attention_norm,
                                                       uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.16.conv_module.layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.16.conv_module.layer_norm.bias.resident",
                                             norm_b,
                                             attention_norm,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_conv_norm(const std::string& model_bundle_dir,
                                                         const std::string& attention_norm_f32,
                                                         uint32_t tokens,
                                                         const std::string& output_norm_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_attention_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_conv_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer16_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_conv_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer16_attention_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer16_conv_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_conv_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_conv_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_conv_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 convolution GLU, depthwise/residual, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        std::string backend = "metal";
        std::vector<float> normed;
        try {
            mit2::MetalContext metal;
            normed = run_w2v_bert_layer16_conv_norm_metal(metal, model, attention_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer16_conv_norm_cpu(model, attention_norm, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_norm, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_conv_glu_cpu(const mit2::Bundle& model,
                                                    const std::vector<float>& conv_norm,
                                                    uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer16_conv_glu invalid input sizes");
    }
    const auto projected = cpu_linear_rows(pw1_w, pw1_b, conv_norm, tokens, 2048, 1024);
    return cpu_glu_split(projected, tokens, 1024);
}

std::vector<float> run_w2v_bert_layer16_conv_glu_metal(mit2::MetalContext& metal,
                                                      const mit2::Bundle& model,
                                                      const std::vector<float>& conv_norm,
                                                      uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer16_conv_glu invalid input sizes");
    }
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.16.conv_module.pointwise_conv1.weight.resident",
        pw1_w,
        "w2v_bert.encoder.layers.16.conv_module.pointwise_conv1.bias.resident",
        pw1_b,
        conv_norm,
        tokens,
        2048,
        1024);
    return metal.glu_split_f32(projected, tokens, 1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_conv_glu(const std::string& model_bundle_dir,
                                                        const std::string& conv_norm_f32,
                                                        uint32_t tokens,
                                                        const std::string& output_glu_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_conv_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_conv_glu\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer16_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_conv_glu_sha256\": \"" << file_sha256_hex(output_glu_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer16_conv_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer16_conv_glu_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_conv_glu_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_conv_glu\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_conv_glu\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 depthwise convolution, convolution residual, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_norm = read_raw_f32(conv_norm_f32);
        std::string backend = "metal";
        std::vector<float> glu;
        try {
            mit2::MetalContext metal;
            glu = run_w2v_bert_layer16_conv_glu_metal(metal, model, conv_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            glu = run_w2v_bert_layer16_conv_glu_cpu(model, conv_norm, tokens);
        }
        write_raw_f32(output_glu_f32, glu);
        print_report(true, backend, &conv_norm, &glu, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_conv_depthwise_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& conv_glu,
                                                          uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_conv.bias");
    if (tokens == 0 || conv_glu.size() != static_cast<size_t>(tokens) * 1024u ||
        weight.size() != static_cast<size_t>(1024) * 31 || bias.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_conv_depthwise invalid input sizes");
    }
    return cpu_depthwise_conv1d_causal(conv_glu, weight, bias, tokens, 1024, 31);
}

std::vector<float> run_w2v_bert_layer16_conv_depthwise_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& conv_glu,
                                                            uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_conv.bias");
    return metal.depthwise_conv1d_causal_f32_resident(
        "w2v_bert.encoder.layers.16.conv_module.depthwise_conv.weight.resident",
        weight,
        "w2v_bert.encoder.layers.16.conv_module.depthwise_conv.bias.resident",
        bias,
        conv_glu,
        tokens,
        1024,
        31);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_conv_depthwise(const std::string& model_bundle_dir,
                                                              const std::string& conv_glu_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_depthwise_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_glu_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_conv_glu");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_conv_depthwise\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer16_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_conv_depthwise_sha256\": \"" << file_sha256_hex(output_depthwise_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer16_conv_glu_values\": " << input->size() << ",\n";
            std::cout << "  \"layer16_conv_depthwise_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_conv_depthwise_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_conv_depthwise\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_conv_depthwise\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 convolution residual, ffn2, and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_glu = read_raw_f32(conv_glu_f32);
        std::string backend = "metal";
        std::vector<float> depthwise;
        try {
            mit2::MetalContext metal;
            depthwise = run_w2v_bert_layer16_conv_depthwise_metal(metal, model, conv_glu, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            depthwise = run_w2v_bert_layer16_conv_depthwise_cpu(model, conv_glu, tokens);
        }
        write_raw_f32(output_depthwise_f32, depthwise);
        print_report(true, backend, &conv_glu, &depthwise, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_conv_residual_cpu(const mit2::Bundle& model,
                                                         const std::vector<float>& attention_norm,
                                                         const std::vector<float>& conv_depthwise,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_conv_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_depthwise, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto activated = cpu_silu(normed);
    const auto projected = cpu_linear_rows(pw2_w, pw2_b, activated, tokens, 1024, 1024);
    std::vector<float> residual(projected.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = attention_norm[i] + projected[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer16_conv_residual_metal(mit2::MetalContext& metal,
                                                           const mit2::Bundle& model,
                                                           const std::vector<float>& attention_norm,
                                                           const std::vector<float>& conv_depthwise,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_conv_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.16.conv_module.depthwise_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.16.conv_module.depthwise_layer_norm.bias.resident",
        norm_b,
        conv_depthwise,
        tokens,
        1024,
        1e-5f);
    const auto activated = metal.silu_f32(normed);
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.16.conv_module.pointwise_conv2.weight.resident",
        pw2_w,
        "w2v_bert.encoder.layers.16.conv_module.pointwise_conv2.bias.resident",
        pw2_b,
        activated,
        tokens,
        1024,
        1024);
    return metal.add_f32(attention_norm, projected);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_conv_residual(const std::string& model_bundle_dir,
                                                             const std::string& attention_norm_f32,
                                                             const std::string& conv_depthwise_f32,
                                                             uint32_t tokens,
                                                             const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_attention_norm");
    append_raw_f32_count_issue(issues,
                               conv_depthwise_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_conv_depthwise");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* attention_norm,
                            const std::vector<float>* conv_depthwise,
                            const std::vector<float>* residual,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_conv_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer16_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer16_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_conv_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (attention_norm && conv_depthwise && residual) {
            std::cout << "  \"layer16_attention_norm_values\": " << attention_norm->size() << ",\n";
            std::cout << "  \"layer16_conv_depthwise_values\": " << conv_depthwise->size() << ",\n";
            std::cout << "  \"layer16_conv_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_conv_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_conv_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_conv_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-16 ffn2 and encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        const auto conv_depthwise = read_raw_f32(conv_depthwise_f32);
        std::string backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer16_conv_residual_metal(metal, model, attention_norm, conv_depthwise, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer16_conv_residual_cpu(model, attention_norm, conv_depthwise, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &attention_norm, &conv_depthwise, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer16_ffn2_residual_cpu(const mit2::Bundle& model,
                                                         const std::vector<float>& conv_residual,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_ffn2_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = conv_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer16_ffn2_residual_metal(mit2::MetalContext& metal,
                                                           const mit2::Bundle& model,
                                                           const std::vector<float>& conv_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.16.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer16_ffn2_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.16.ffn2_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.16.ffn2_layer_norm.bias.resident",
        norm_b,
        conv_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.16.ffn2.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.16.ffn2.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.16.ffn2.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.16.ffn2.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(conv_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer16_ffn2_residual(const std::string& model_bundle_dir,
                                                             const std::string& conv_residual_f32,
                                                             uint32_t tokens,
                                                             const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer16_conv_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer16_ffn2_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_layer16_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer16_ffn2_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer16_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer16_conv_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer16_ffn2_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer16_ffn2_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer16_ffn2_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer16_ffn2_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(conv_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer16_ffn2_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer16_ffn2_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_attention_project_cpu(const mit2::Bundle& model,
                                                              const std::vector<float>& context,
                                                              uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.13.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_out.bias");
    }
    if (tokens == 0 || context.size() != static_cast<size_t>(tokens) * 1024u ||
        out_w.size() != static_cast<size_t>(1024) * 1024 || out_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_attention_project invalid input sizes");
    }
    return cpu_linear_rows(out_w, out_b, context, tokens, 1024, 1024);
}

std::vector<float> run_w2v_bert_layer13_attention_project_metal(mit2::MetalContext& metal,
                                                                const mit2::Bundle& model,
                                                                const std::vector<float>& context,
                                                                uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.13.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_out.bias");
    }
    return metal.linear_rows_f32_resident("w2v_bert.encoder.layers.13.self_attn.linear_out.weight.resident",
                                          out_w,
                                          "w2v_bert.encoder.layers.13.self_attn.linear_out.bias.resident_or_zero",
                                          out_b,
                                          context,
                                          tokens,
                                          1024,
                                          1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_attention_project(const std::string& model_bundle_dir,
                                                                  const std::string& context_f32,
                                                                  uint32_t tokens,
                                                                  const std::string& output_attention_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               context_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_attention_context");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_attention_project\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_layer13_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_attention_sha256\": \"" << file_sha256_hex(output_attention_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer13_context_values\": " << input->size() << ",\n";
            std::cout << "  \"layer13_attention_projection_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_attention_project_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_attention_projection\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_attention_projection\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 attention residual/norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto context = read_raw_f32(context_f32);
        std::string backend = "metal";
        std::vector<float> projected;
        try {
            mit2::MetalContext metal;
            projected = run_w2v_bert_layer13_attention_project_metal(metal, model, context, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            projected = run_w2v_bert_layer13_attention_project_cpu(model, context, tokens);
        }
        write_raw_f32(output_attention_f32, projected);
        print_report(true, backend, &context, &projected, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_attention_residual_cpu(const std::vector<float>& ffn1_residual,
                                                               const std::vector<float>& attention_projection,
                                                               uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer13_attention_residual invalid input sizes");
    }
    std::vector<float> residual(expected);
    for (size_t i = 0; i < expected; ++i) {
        residual[i] = ffn1_residual[i] + attention_projection[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer13_attention_residual_metal(mit2::MetalContext& metal,
                                                                 const std::vector<float>& ffn1_residual,
                                                                 const std::vector<float>& attention_projection,
                                                                 uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer13_attention_residual invalid input sizes");
    }
    return metal.add_f32(ffn1_residual, attention_projection);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_attention_residual(const std::string& ffn1_residual_f32,
                                                                   const std::string& attention_f32,
                                                                   uint32_t tokens,
                                                                   const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               ffn1_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_ffn1_residual");
    append_raw_f32_count_issue(issues,
                               attention_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_attention_projection");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* ffn1_residual,
                            const std::vector<float>* attention_projection,
                            const std::vector<float>* residual,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_attention_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer13_ffn1_residual_f32\": \"" << json_escape(ffn1_residual_f32) << "\",\n";
        std::cout << "  \"w2v_layer13_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_layer13_attention_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_attention_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (ffn1_residual && attention_projection && residual) {
            std::cout << "  \"layer13_ffn1_residual_values\": " << ffn1_residual->size() << ",\n";
            std::cout << "  \"layer13_attention_projection_values\": " << attention_projection->size() << ",\n";
            std::cout << "  \"layer13_attention_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer13_attention_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_attention_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_attention_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 attention norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, nullptr, issues);
        return false;
    }

    try {
        const auto ffn1_residual = read_raw_f32(ffn1_residual_f32);
        const auto attention_projection = read_raw_f32(attention_f32);
        std::string backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer13_attention_residual_metal(metal,
                                                                     ffn1_residual,
                                                                     attention_projection,
                                                                     tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer13_attention_residual_cpu(ffn1_residual, attention_projection, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &ffn1_residual, &attention_projection, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_attention_norm_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& attention_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn_layer_norm.bias");
    if (tokens == 0 || attention_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_attention_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer13_attention_norm_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& attention_residual,
                                                             uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn_layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.13.self_attn_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.13.self_attn_layer_norm.bias.resident",
                                             norm_b,
                                             attention_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_attention_norm(const std::string& model_bundle_dir,
                                                               const std::string& attention_residual_f32,
                                                               uint32_t tokens,
                                                               const std::string& output_norm_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_attention_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_attention_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_layer13_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_attention_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer13_attention_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer13_attention_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_attention_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_attention_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_attention_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_residual = read_raw_f32(attention_residual_f32);
        std::string backend = "metal";
        std::vector<float> normed;
        try {
            mit2::MetalContext metal;
            normed = run_w2v_bert_layer13_attention_norm_metal(metal, model, attention_residual, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer13_attention_norm_cpu(model, attention_residual, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_residual, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_conv_norm_cpu(const mit2::Bundle& model,
                                                      const std::vector<float>& attention_norm,
                                                      uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.layer_norm.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_conv_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_norm, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer13_conv_norm_metal(mit2::MetalContext& metal,
                                                        const mit2::Bundle& model,
                                                        const std::vector<float>& attention_norm,
                                                        uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.13.conv_module.layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.13.conv_module.layer_norm.bias.resident",
                                             norm_b,
                                             attention_norm,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_conv_norm(const std::string& model_bundle_dir,
                                                         const std::string& attention_norm_f32,
                                                         uint32_t tokens,
                                                         const std::string& output_norm_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_attention_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_conv_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer13_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_conv_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer13_attention_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer13_conv_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_conv_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_conv_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_conv_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 convolution GLU, depthwise/residual, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        std::string backend = "metal";
        std::vector<float> normed;
        try {
            mit2::MetalContext metal;
            normed = run_w2v_bert_layer13_conv_norm_metal(metal, model, attention_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer13_conv_norm_cpu(model, attention_norm, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_norm, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer1_conv_norm_cpu(const mit2::Bundle& model,
                                                     const std::vector<float>& attention_norm,
                                                     uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.layer_norm.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_conv_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_norm, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer1_conv_norm_metal(mit2::MetalContext& metal,
                                                       const mit2::Bundle& model,
                                                       const std::vector<float>& attention_norm,
                                                       uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.1.conv_module.layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.1.conv_module.layer_norm.bias.resident",
                                             norm_b,
                                             attention_norm,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer1_conv_norm(const std::string& model_bundle_dir,
                                                         const std::string& attention_norm_f32,
                                                         uint32_t tokens,
                                                         const std::string& output_norm_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer1_attention_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_norm_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution GLU/depthwise/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        std::string execution_backend = "metal";
        std::vector<float> normed;
        try {
            mit2::MetalContext metal;
            normed = run_w2v_bert_layer1_conv_norm_metal(metal, model, attention_norm, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            normed = run_w2v_bert_layer1_conv_norm_cpu(model, attention_norm, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_norm\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_attention_norm_values\": " << attention_norm.size() << ",\n";
        std::cout << "  \"layer1_conv_norm_values\": " << normed.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_norm_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_norm\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_norm\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution GLU/depthwise/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer1_conv_norm_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution GLU/depthwise/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer1_conv_glu_cpu(const mit2::Bundle& model,
                                                    const std::vector<float>& conv_norm,
                                                    uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer1_conv_glu invalid input sizes");
    }
    const auto projected = cpu_linear_rows(pw1_w, pw1_b, conv_norm, tokens, 2048, 1024);
    return cpu_glu_split(projected, tokens, 1024);
}

std::vector<float> run_w2v_bert_layer1_conv_glu_metal(mit2::MetalContext& metal,
                                                      const mit2::Bundle& model,
                                                      const std::vector<float>& conv_norm,
                                                      uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer1_conv_glu invalid input sizes");
    }
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.weight.resident",
        pw1_w,
        "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.bias.resident",
        pw1_b,
        conv_norm,
        tokens,
        2048,
        1024);
    return metal.glu_split_f32(projected, tokens, 1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer1_conv_glu(const std::string& model_bundle_dir,
                                                        const std::string& conv_norm_f32,
                                                        uint32_t tokens,
                                                        const std::string& output_glu_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer1_conv_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_glu\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_glu_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_glu\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_glu\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution depthwise/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_norm = read_raw_f32(conv_norm_f32);
        std::string execution_backend = "metal";
        std::vector<float> glu;
        try {
            mit2::MetalContext metal;
            glu = run_w2v_bert_layer1_conv_glu_metal(metal, model, conv_norm, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            glu = run_w2v_bert_layer1_conv_glu_cpu(model, conv_norm, tokens);
        }
        write_raw_f32(output_glu_f32, glu);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_glu\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_glu_sha256\": \"" << file_sha256_hex(output_glu_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_norm_values\": " << conv_norm.size() << ",\n";
        std::cout << "  \"layer1_conv_glu_values\": " << glu.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_glu_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_glu\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_glu\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution depthwise/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_glu\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer1_conv_glu_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_glu\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_glu\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution depthwise/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer1_conv_depthwise_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& conv_glu,
                                                          uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.bias");
    if (tokens == 0 || conv_glu.size() != static_cast<size_t>(tokens) * 1024u ||
        weight.size() != static_cast<size_t>(1024) * 31 || bias.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_conv_depthwise invalid input sizes");
    }
    return cpu_depthwise_conv1d_causal(conv_glu, weight, bias, tokens, 1024, 31);
}

std::vector<float> run_w2v_bert_layer1_conv_depthwise_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& conv_glu,
                                                            uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.bias");
    return metal.depthwise_conv1d_causal_f32_resident(
        "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.weight.resident",
        weight,
        "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.bias.resident",
        bias,
        conv_glu,
        tokens,
        1024,
        31);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer1_conv_depthwise(const std::string& model_bundle_dir,
                                                              const std::string& conv_glu_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_depthwise_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_glu_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer1_conv_glu");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_depthwise\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_depthwise_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_depthwise\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_depthwise\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution activation/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_glu = read_raw_f32(conv_glu_f32);
        std::string execution_backend = "metal";
        std::vector<float> depthwise;
        try {
            mit2::MetalContext metal;
            depthwise = run_w2v_bert_layer1_conv_depthwise_metal(metal, model, conv_glu, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            depthwise = run_w2v_bert_layer1_conv_depthwise_cpu(model, conv_glu, tokens);
        }
        write_raw_f32(output_depthwise_f32, depthwise);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_depthwise\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_depthwise_sha256\": \"" << file_sha256_hex(output_depthwise_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_glu_values\": " << conv_glu.size() << ",\n";
        std::cout << "  \"layer1_conv_depthwise_values\": " << depthwise.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_depthwise_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_depthwise\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_depthwise\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution activation/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_depthwise\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer1_conv_depthwise_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_depthwise\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_depthwise\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 convolution activation/projection and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer1_conv_residual_cpu(const mit2::Bundle& model,
                                                         const std::vector<float>& attention_norm,
                                                         const std::vector<float>& conv_depthwise,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_conv_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_depthwise, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto activated = cpu_silu(normed);
    const auto projected = cpu_linear_rows(pw2_w, pw2_b, activated, tokens, 1024, 1024);
    std::vector<float> residual(projected.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = attention_norm[i] + projected[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer1_conv_residual_metal(mit2::MetalContext& metal,
                                                           const mit2::Bundle& model,
                                                           const std::vector<float>& attention_norm,
                                                           const std::vector<float>& conv_depthwise,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_conv_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.bias.resident",
        norm_b,
        conv_depthwise,
        tokens,
        1024,
        1e-5f);
    const auto activated = metal.silu_f32(normed);
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.weight.resident",
        pw2_w,
        "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.bias.resident",
        pw2_b,
        activated,
        tokens,
        1024,
        1024);
    return metal.add_f32(attention_norm, projected);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer1_conv_residual(const std::string& model_bundle_dir,
                                                             const std::string& attention_norm_f32,
                                                             const std::string& conv_depthwise_f32,
                                                             uint32_t tokens,
                                                             const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer1_attention_norm");
    append_raw_f32_count_issue(issues,
                               conv_depthwise_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer1_conv_depthwise");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_residual_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 ffn2/final norm and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        const auto conv_depthwise = read_raw_f32(conv_depthwise_f32);
        std::string execution_backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer1_conv_residual_metal(metal, model, attention_norm, conv_depthwise, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            residual = run_w2v_bert_layer1_conv_residual_cpu(model, attention_norm, conv_depthwise, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_residual\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_attention_norm_values\": " << attention_norm.size() << ",\n";
        std::cout << "  \"layer1_conv_depthwise_values\": " << conv_depthwise.size() << ",\n";
        std::cout << "  \"layer1_conv_residual_values\": " << residual.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_conv_residual_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_residual\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_residual\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 ffn2/final norm and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_conv_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer1_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer1_conv_residual_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_conv_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_conv_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 ffn2/final norm and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_attention_project_cpu(const mit2::Bundle& model,
                                                             const std::vector<float>& context,
                                                             uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.0.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn.linear_out.bias");
    }
    if (tokens == 0 || context.size() != static_cast<size_t>(tokens) * 1024u ||
        out_w.size() != static_cast<size_t>(1024) * 1024 || out_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_attention_project invalid input sizes");
    }
    return cpu_linear_rows(out_w, out_b, context, tokens, 1024, 1024);
}

std::vector<float> run_w2v_bert_layer0_attention_project_metal(mit2::MetalContext& metal,
                                                               const mit2::Bundle& model,
                                                               const std::vector<float>& context,
                                                               uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.0.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn.linear_out.bias");
    }
    return metal.linear_rows_f32_resident("w2v_bert.encoder.layers.0.self_attn.linear_out.weight.resident",
                                          out_w,
                                          "w2v_bert.encoder.layers.0.self_attn.linear_out.bias.resident_or_zero",
                                          out_b,
                                          context,
                                          tokens,
                                          1024,
                                          1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_attention_project(const std::string& model_bundle_dir,
                                                                 const std::string& context_f32,
                                                                 uint32_t tokens,
                                                                 const std::string& output_attention_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               context_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_attention_context");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_project\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_attention_project_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_projection\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_projection\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 norm/feed-forward/convolution and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto context = read_raw_f32(context_f32);
        std::string execution_backend = "metal";
        std::vector<float> projected;
        try {
            mit2::MetalContext metal;
            projected = run_w2v_bert_layer0_attention_project_metal(metal, model, context, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            projected = run_w2v_bert_layer0_attention_project_cpu(model, context, tokens);
        }
        write_raw_f32(output_attention_f32, projected);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_project\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        std::cout << "  \"output_attention_sha256\": \"" << file_sha256_hex(output_attention_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"context_values\": " << context.size() << ",\n";
        std::cout << "  \"attention_projection_values\": " << projected.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_attention_project_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_projection\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_projection\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 norm/feed-forward/convolution and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_project\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_attention_project_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_projection\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_projection\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 norm/feed-forward/convolution and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_attention_residual_cpu(const std::vector<float>& feature_projection,
                                                              const std::vector<float>& attention_projection,
                                                              uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || feature_projection.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer0_attention_residual invalid input sizes");
    }
    std::vector<float> residual(expected);
    for (size_t i = 0; i < expected; ++i) {
        residual[i] = feature_projection[i] + attention_projection[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer0_attention_residual_metal(mit2::MetalContext& metal,
                                                                const std::vector<float>& feature_projection,
                                                                const std::vector<float>& attention_projection,
                                                                uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || feature_projection.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer0_attention_residual invalid input sizes");
    }
    return metal.add_f32(feature_projection, attention_projection);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_attention_residual(const std::string& feature_projection_f32,
                                                                  const std::string& attention_f32,
                                                                  uint32_t tokens,
                                                                  const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               feature_projection_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_feature_projection");
    append_raw_f32_count_issue(issues,
                               attention_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_attention_projection");

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_feature_projection_f32\": \"" << json_escape(feature_projection_f32) << "\",\n";
        std::cout << "  \"w2v_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"feature_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"clone_w2v_layer0_attention_residual_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 norm/feed-forward/convolution and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        const auto feature_projection = read_raw_f32(feature_projection_f32);
        const auto attention_projection = read_raw_f32(attention_f32);
        std::string execution_backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer0_attention_residual_metal(metal,
                                                                    feature_projection,
                                                                    attention_projection,
                                                                    tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            residual = run_w2v_bert_layer0_attention_residual_cpu(feature_projection, attention_projection, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_residual\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_feature_projection_f32\": \"" << json_escape(feature_projection_f32) << "\",\n";
        std::cout << "  \"w2v_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"output_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"feature_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"feature_projection_values\": " << feature_projection.size() << ",\n";
        std::cout << "  \"attention_projection_values\": " << attention_projection.size() << ",\n";
        std::cout << "  \"residual_values\": " << residual.size() << ",\n";
        std::cout << "  \"clone_w2v_layer0_attention_residual_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_residual\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_residual\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 norm/feed-forward/convolution and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_feature_projection_f32\": \"" << json_escape(feature_projection_f32) << "\",\n";
        std::cout << "  \"w2v_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_attention_residual_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 norm/feed-forward/convolution and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_attention_norm_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& attention_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn_layer_norm.bias");
    if (tokens == 0 || attention_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_attention_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer0_attention_norm_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& attention_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.self_attn_layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.0.self_attn_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.0.self_attn_layer_norm.bias.resident",
                                             norm_b,
                                             attention_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_attention_norm(const std::string& model_bundle_dir,
                                                              const std::string& attention_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_norm_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_attention_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_attention_norm_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution module and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_residual = read_raw_f32(attention_residual_f32);
        std::string execution_backend = "metal";
        std::vector<float> normed;
        try {
            mit2::MetalContext metal;
            normed = run_w2v_bert_layer0_attention_norm_metal(metal, model, attention_residual, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            normed = run_w2v_bert_layer0_attention_norm_cpu(model, attention_residual, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_norm\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"output_attention_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_residual_values\": " << attention_residual.size() << ",\n";
        std::cout << "  \"attention_norm_values\": " << normed.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_attention_norm_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_norm\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_norm\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution module and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_attention_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_attention_norm_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_attention_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_attention_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution module and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_conv_norm_cpu(const mit2::Bundle& model,
                                                     const std::vector<float>& attention_norm,
                                                     uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.layer_norm.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_conv_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_norm, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer0_conv_norm_metal(mit2::MetalContext& metal,
                                                       const mit2::Bundle& model,
                                                       const std::vector<float>& attention_norm,
                                                       uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.0.conv_module.layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.0.conv_module.layer_norm.bias.resident",
                                             norm_b,
                                             attention_norm,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_conv_norm(const std::string& model_bundle_dir,
                                                         const std::string& attention_norm_f32,
                                                         uint32_t tokens,
                                                         const std::string& output_norm_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_attention_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_norm_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution GLU/depthwise/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        std::string execution_backend = "metal";
        std::vector<float> normed;
        try {
            mit2::MetalContext metal;
            normed = run_w2v_bert_layer0_conv_norm_metal(metal, model, attention_norm, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            normed = run_w2v_bert_layer0_conv_norm_cpu(model, attention_norm, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_norm\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"output_conv_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_norm_values\": " << attention_norm.size() << ",\n";
        std::cout << "  \"conv_norm_values\": " << normed.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_norm_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_norm\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_norm\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution GLU/depthwise/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_conv_norm_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution GLU/depthwise/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_conv_glu_cpu(const mit2::Bundle& model,
                                                    const std::vector<float>& conv_norm,
                                                    uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer0_conv_glu invalid input sizes");
    }
    const auto projected = cpu_linear_rows(pw1_w, pw1_b, conv_norm, tokens, 2048, 1024);
    return cpu_glu_split(projected, tokens, 1024);
}

std::vector<float> run_w2v_bert_layer0_conv_glu_metal(mit2::MetalContext& metal,
                                                      const mit2::Bundle& model,
                                                      const std::vector<float>& conv_norm,
                                                      uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer0_conv_glu invalid input sizes");
    }
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.weight.resident",
        pw1_w,
        "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.bias.resident",
        pw1_b,
        conv_norm,
        tokens,
        2048,
        1024);
    return metal.glu_split_f32(projected, tokens, 1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_conv_glu(const std::string& model_bundle_dir,
                                                        const std::string& conv_norm_f32,
                                                        uint32_t tokens,
                                                        const std::string& output_glu_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_conv_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_glu\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_glu_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_glu\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_glu\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution depthwise/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_norm = read_raw_f32(conv_norm_f32);
        std::string execution_backend = "metal";
        std::vector<float> glu;
        try {
            mit2::MetalContext metal;
            glu = run_w2v_bert_layer0_conv_glu_metal(metal, model, conv_norm, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            glu = run_w2v_bert_layer0_conv_glu_cpu(model, conv_norm, tokens);
        }
        write_raw_f32(output_glu_f32, glu);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_glu\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        std::cout << "  \"output_conv_glu_sha256\": \"" << file_sha256_hex(output_glu_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_norm_values\": " << conv_norm.size() << ",\n";
        std::cout << "  \"conv_glu_values\": " << glu.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_glu_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_glu\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_glu\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution depthwise/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_glu\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_conv_glu_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_glu\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_glu\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution depthwise/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_conv_depthwise_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& conv_glu,
                                                          uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.bias");
    if (tokens == 0 || conv_glu.size() != static_cast<size_t>(tokens) * 1024u ||
        weight.size() != static_cast<size_t>(1024) * 31 || bias.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_conv_depthwise invalid input sizes");
    }
    return cpu_depthwise_conv1d_causal(conv_glu, weight, bias, tokens, 1024, 31);
}

std::vector<float> run_w2v_bert_layer0_conv_depthwise_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& conv_glu,
                                                            uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.bias");
    return metal.depthwise_conv1d_causal_f32_resident(
        "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.weight.resident",
        weight,
        "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.bias.resident",
        bias,
        conv_glu,
        tokens,
        1024,
        31);
}

std::vector<float> run_w2v_bert_layer0_conv_residual_cpu(const mit2::Bundle& model,
                                                         const std::vector<float>& attention_norm,
                                                         const std::vector<float>& conv_depthwise,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_conv_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_depthwise, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto activated = cpu_silu(normed);
    const auto projected = cpu_linear_rows(pw2_w, pw2_b, activated, tokens, 1024, 1024);
    std::vector<float> residual(projected.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = attention_norm[i] + projected[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer0_conv_residual_metal(mit2::MetalContext& metal,
                                                           const mit2::Bundle& model,
                                                           const std::vector<float>& attention_norm,
                                                           const std::vector<float>& conv_depthwise,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_conv_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.bias.resident",
        norm_b,
        conv_depthwise,
        tokens,
        1024,
        1e-5f);
    const auto activated = metal.silu_f32(normed);
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.weight.resident",
        pw2_w,
        "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.bias.resident",
        pw2_b,
        activated,
        tokens,
        1024,
        1024);
    return metal.add_f32(attention_norm, projected);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_conv_depthwise(const std::string& model_bundle_dir,
                                                              const std::string& conv_glu_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_depthwise_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_glu_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_conv_glu");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_depthwise\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_depthwise_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_depthwise\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_depthwise\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution activation/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_glu = read_raw_f32(conv_glu_f32);
        std::string execution_backend = "metal";
        std::vector<float> depthwise;
        try {
            mit2::MetalContext metal;
            depthwise = run_w2v_bert_layer0_conv_depthwise_metal(metal, model, conv_glu, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            depthwise = run_w2v_bert_layer0_conv_depthwise_cpu(model, conv_glu, tokens);
        }
        write_raw_f32(output_depthwise_f32, depthwise);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_depthwise\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        std::cout << "  \"output_conv_depthwise_sha256\": \"" << file_sha256_hex(output_depthwise_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_glu_values\": " << conv_glu.size() << ",\n";
        std::cout << "  \"conv_depthwise_values\": " << depthwise.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_depthwise_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_depthwise\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_depthwise\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution activation/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_depthwise\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_conv_depthwise_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_depthwise\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_depthwise\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 convolution activation/projection and encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_conv_residual(const std::string& model_bundle_dir,
                                                             const std::string& attention_norm_f32,
                                                             const std::string& conv_depthwise_f32,
                                                             uint32_t tokens,
                                                             const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_attention_norm");
    append_raw_f32_count_issue(issues,
                               conv_depthwise_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_conv_depthwise");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_residual_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 ffn2 and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        const auto conv_depthwise = read_raw_f32(conv_depthwise_f32);
        std::string execution_backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer0_conv_residual_metal(metal, model, attention_norm, conv_depthwise, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            residual = run_w2v_bert_layer0_conv_residual_cpu(model, attention_norm, conv_depthwise, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_residual\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"output_conv_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_norm_values\": " << attention_norm.size() << ",\n";
        std::cout << "  \"conv_depthwise_values\": " << conv_depthwise.size() << ",\n";
        std::cout << "  \"conv_residual_values\": " << residual.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_conv_residual_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_residual\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_residual\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 ffn2 and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_conv_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_conv_residual_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_conv_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_conv_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 ffn2 and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_ffn2_residual_cpu(const mit2::Bundle& model,
                                                         const std::vector<float>& conv_residual,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_ffn2_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = conv_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer0_ffn2_residual_metal(mit2::MetalContext& metal,
                                                           const mit2::Bundle& model,
                                                           const std::vector<float>& conv_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_ffn2_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.0.ffn2_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.0.ffn2_layer_norm.bias.resident",
        norm_b,
        conv_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.0.ffn2.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.0.ffn2.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(conv_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_ffn2_residual(const std::string& model_bundle_dir,
                                                            const std::string& conv_residual_f32,
                                                            uint32_t tokens,
                                                            const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_conv_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_ffn2_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_ffn2_residual_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_ffn2_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_ffn2_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 final LayerNorm and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_residual = read_raw_f32(conv_residual_f32);
        std::string execution_backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer0_ffn2_residual_metal(metal, model, conv_residual, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            residual = run_w2v_bert_layer0_ffn2_residual_cpu(model, conv_residual, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_ffn2_residual\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"output_ffn2_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"conv_residual_values\": " << conv_residual.size() << ",\n";
        std::cout << "  \"ffn2_residual_values\": " << residual.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_ffn2_residual_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_ffn2_residual\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_ffn2_residual\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 final LayerNorm and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_ffn2_residual\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_ffn2_residual_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_ffn2_residual\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_ffn2_residual\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-0 final LayerNorm and remaining encoder stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer0_final_norm_cpu(const mit2::Bundle& model,
                                                      const std::vector<float>& ffn2_residual,
                                                      uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.final_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.final_layer_norm.bias");
    if (tokens == 0 || ffn2_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_final_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(ffn2_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer0_final_norm_metal(mit2::MetalContext& metal,
                                                        const mit2::Bundle& model,
                                                        const std::vector<float>& ffn2_residual,
                                                        uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.0.final_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.0.final_layer_norm.bias");
    if (tokens == 0 || ffn2_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer0_final_norm invalid input sizes");
    }
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.0.final_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.0.final_layer_norm.bias.resident",
                                             norm_b,
                                             ffn2_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer0_final_norm(const std::string& model_bundle_dir,
                                                          const std::string& ffn2_residual_f32,
                                                          uint32_t tokens,
                                                          const std::string& output_layer0_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               ffn2_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer0_ffn2_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_final_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_ffn2_residual_f32\": \"" << json_escape(ffn2_residual_f32) << "\",\n";
        std::cout << "  \"output_layer0_f32\": \"" << json_escape(output_layer0_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer0_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_final_norm_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_final_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_final_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layer 1 and remaining stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto ffn2_residual = read_raw_f32(ffn2_residual_f32);
        std::string execution_backend = "metal";
        std::vector<float> layer0;
        try {
            mit2::MetalContext metal;
            layer0 = run_w2v_bert_layer0_final_norm_metal(metal, model, ffn2_residual, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            layer0 = run_w2v_bert_layer0_final_norm_cpu(model, ffn2_residual, tokens);
        }
        write_raw_f32(output_layer0_f32, layer0);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_final_norm\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_ffn2_residual_f32\": \"" << json_escape(ffn2_residual_f32) << "\",\n";
        std::cout << "  \"output_layer0_f32\": \"" << json_escape(output_layer0_f32) << "\",\n";
        std::cout << "  \"output_layer0_sha256\": \"" << file_sha256_hex(output_layer0_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer0_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"ffn2_residual_values\": " << ffn2_residual.size() << ",\n";
        std::cout << "  \"layer0_values\": " << layer0.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer0_final_norm_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_final_norm\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_final_norm\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layer 1 and remaining stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer0_final_norm\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_ffn2_residual_f32\": \"" << json_escape(ffn2_residual_f32) << "\",\n";
        std::cout << "  \"output_layer0_f32\": \"" << json_escape(output_layer0_f32) << "\",\n";
        std::cout << "  \"clone_w2v_layer0_final_norm_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_layer0_final_norm\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer0_final_norm\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layer 1 and remaining stack to hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

std::vector<float> run_w2v_bert_layer1_ffn2_residual_cpu(const mit2::Bundle& model,
                                                         const std::vector<float>& conv_residual,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_ffn2_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = conv_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer1_ffn2_residual_metal(mit2::MetalContext& metal,
                                                           const mit2::Bundle& model,
                                                           const std::vector<float>& conv_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_ffn2_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.1.ffn2_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.1.ffn2_layer_norm.bias.resident",
        norm_b,
        conv_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.1.ffn2.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.1.ffn2.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(conv_residual, output, 0.5f);
}

std::vector<float> run_w2v_bert_layer1_final_norm_cpu(const mit2::Bundle& model,
                                                      const std::vector<float>& ffn2_residual,
                                                      uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.final_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.final_layer_norm.bias");
    if (tokens == 0 || ffn2_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_final_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(ffn2_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer1_final_norm_metal(mit2::MetalContext& metal,
                                                        const mit2::Bundle& model,
                                                        const std::vector<float>& ffn2_residual,
                                                        uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.1.final_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.1.final_layer_norm.bias");
    if (tokens == 0 || ffn2_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer1_final_norm invalid input sizes");
    }
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.1.final_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.1.final_layer_norm.bias.resident",
                                             norm_b,
                                             ffn2_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer1_ffn2_residual(const std::string& model_bundle_dir,
                                                             const std::string& conv_residual_f32,
                                                             uint32_t tokens,
                                                             const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues, conv_residual_f32, static_cast<uint64_t>(tokens) * 1024u, "w2v_layer1_conv_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_ffn2_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_layer1_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer1_ffn2_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer1_conv_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer1_ffn2_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_ffn2_residual_issues\": ";
        print_json_string_array(ok ? std::vector<std::string>{} : issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_ffn2_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_ffn2_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-1 final LayerNorm and encoder layers 2-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr);
        return false;
    }
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(conv_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer1_ffn2_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer1_ffn2_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output);
        return true;
    } catch (const std::exception& e) {
        issues = {e.what()};
        print_report(false, "", nullptr, nullptr);
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_w2v_layer1_final_norm(const std::string& model_bundle_dir,
                                                          const std::string& ffn2_residual_f32,
                                                          uint32_t tokens,
                                                          const std::string& output_layer1_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues, ffn2_residual_f32, static_cast<uint64_t>(tokens) * 1024u, "w2v_layer1_ffn2_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer1_final_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer1_ffn2_residual_f32\": \"" << json_escape(ffn2_residual_f32) << "\",\n";
        std::cout << "  \"output_layer1_f32\": \"" << json_escape(output_layer1_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer1_sha256\": \"" << file_sha256_hex(output_layer1_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer1_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer1_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer1_ffn2_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer1_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer1_final_norm_issues\": ";
        print_json_string_array(ok ? std::vector<std::string>{} : issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer1_final_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer1_final_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layers 2-16 and layer-17 final norm to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr);
        return false;
    }
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(ffn2_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer1_final_norm_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer1_final_norm_cpu(model, input, tokens);
        }
        write_raw_f32(output_layer1_f32, output);
        print_report(true, backend, &input, &output);
        return true;
    } catch (const std::exception& e) {
        issues = {e.what()};
        print_report(false, "", nullptr, nullptr);
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_conv_glu_cpu(const mit2::Bundle& model,
                                                     const std::vector<float>& conv_norm,
                                                     uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer13_conv_glu invalid input sizes");
    }
    const auto projected = cpu_linear_rows(pw1_w, pw1_b, conv_norm, tokens, 2048, 1024);
    return cpu_glu_split(projected, tokens, 1024);
}

std::vector<float> run_w2v_bert_layer13_conv_glu_metal(mit2::MetalContext& metal,
                                                       const mit2::Bundle& model,
                                                       const std::vector<float>& conv_norm,
                                                       uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer13_conv_glu invalid input sizes");
    }
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.weight.resident",
        pw1_w,
        "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.bias.resident",
        pw1_b,
        conv_norm,
        tokens,
        2048,
        1024);
    return metal.glu_split_f32(projected, tokens, 1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_conv_glu(const std::string& model_bundle_dir,
                                                         const std::string& conv_norm_f32,
                                                         uint32_t tokens,
                                                         const std::string& output_glu_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_conv_norm");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_conv_glu\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer13_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_conv_glu_sha256\": \"" << file_sha256_hex(output_glu_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer13_conv_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer13_conv_glu_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_conv_glu_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_conv_glu\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_conv_glu\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 depthwise convolution, convolution residual, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_norm = read_raw_f32(conv_norm_f32);
        std::string backend = "metal";
        std::vector<float> glu;
        try {
            mit2::MetalContext metal;
            glu = run_w2v_bert_layer13_conv_glu_metal(metal, model, conv_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            glu = run_w2v_bert_layer13_conv_glu_cpu(model, conv_norm, tokens);
        }
        write_raw_f32(output_glu_f32, glu);
        print_report(true, backend, &conv_norm, &glu, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_conv_depthwise_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& conv_glu,
                                                           uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.bias");
    if (tokens == 0 || conv_glu.size() != static_cast<size_t>(tokens) * 1024u ||
        weight.size() != static_cast<size_t>(1024) * 31 || bias.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_conv_depthwise invalid input sizes");
    }
    return cpu_depthwise_conv1d_causal(conv_glu, weight, bias, tokens, 1024, 31);
}

std::vector<float> run_w2v_bert_layer13_conv_depthwise_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& conv_glu,
                                                             uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.bias");
    return metal.depthwise_conv1d_causal_f32_resident(
        "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.weight.resident",
        weight,
        "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.bias.resident",
        bias,
        conv_glu,
        tokens,
        1024,
        31);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_conv_depthwise(const std::string& model_bundle_dir,
                                                               const std::string& conv_glu_f32,
                                                               uint32_t tokens,
                                                               const std::string& output_depthwise_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_glu_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_conv_glu");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_conv_depthwise\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer13_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_conv_depthwise_sha256\": \"" << file_sha256_hex(output_depthwise_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer13_conv_glu_values\": " << input->size() << ",\n";
            std::cout << "  \"layer13_conv_depthwise_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_conv_depthwise_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_conv_depthwise\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_conv_depthwise\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 convolution residual, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto conv_glu = read_raw_f32(conv_glu_f32);
        std::string backend = "metal";
        std::vector<float> depthwise;
        try {
            mit2::MetalContext metal;
            depthwise = run_w2v_bert_layer13_conv_depthwise_metal(metal, model, conv_glu, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            depthwise = run_w2v_bert_layer13_conv_depthwise_cpu(model, conv_glu, tokens);
        }
        write_raw_f32(output_depthwise_f32, depthwise);
        print_report(true, backend, &conv_glu, &depthwise, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_conv_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& attention_norm,
                                                          const std::vector<float>& conv_depthwise,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_conv_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_depthwise, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto activated = cpu_silu(normed);
    const auto projected = cpu_linear_rows(pw2_w, pw2_b, activated, tokens, 1024, 1024);
    std::vector<float> residual(projected.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = attention_norm[i] + projected[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer13_conv_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& attention_norm,
                                                            const std::vector<float>& conv_depthwise,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_conv_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.bias.resident",
        norm_b,
        conv_depthwise,
        tokens,
        1024,
        1e-5f);
    const auto activated = metal.silu_f32(normed);
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.weight.resident",
        pw2_w,
        "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.bias.resident",
        pw2_b,
        activated,
        tokens,
        1024,
        1024);
    return metal.add_f32(attention_norm, projected);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_conv_residual(const std::string& model_bundle_dir,
                                                              const std::string& attention_norm_f32,
                                                              const std::string& conv_depthwise_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               attention_norm_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_attention_norm");
    append_raw_f32_count_issue(issues,
                               conv_depthwise_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_conv_depthwise");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* attention_norm,
                            const std::vector<float>* conv_depthwise,
                            const std::vector<float>* residual,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_conv_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer13_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer13_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_conv_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (attention_norm && conv_depthwise && residual) {
            std::cout << "  \"layer13_attention_norm_values\": " << attention_norm->size() << ",\n";
            std::cout << "  \"layer13_conv_depthwise_values\": " << conv_depthwise->size() << ",\n";
            std::cout << "  \"layer13_conv_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_conv_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_conv_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_conv_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 ffn2 and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto attention_norm = read_raw_f32(attention_norm_f32);
        const auto conv_depthwise = read_raw_f32(conv_depthwise_f32);
        std::string backend = "metal";
        std::vector<float> residual;
        try {
            mit2::MetalContext metal;
            residual = run_w2v_bert_layer13_conv_residual_metal(metal, model, attention_norm, conv_depthwise, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer13_conv_residual_cpu(model, attention_norm, conv_depthwise, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &attention_norm, &conv_depthwise, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_ffn2_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& conv_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_ffn2_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(conv_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = conv_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer13_ffn2_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& conv_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_ffn2_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.13.ffn2_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.13.ffn2_layer_norm.bias.resident",
        norm_b,
        conv_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.13.ffn2.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.13.ffn2.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(conv_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_ffn2_residual(const std::string& model_bundle_dir,
                                                              const std::string& conv_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               conv_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_conv_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_ffn2_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_layer13_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_ffn2_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer13_conv_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer13_ffn2_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_ffn2_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_ffn2_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_ffn2_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(conv_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer13_ffn2_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer13_ffn2_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer14_ffn1_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& layer13_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.output_dense.bias");
    if (tokens == 0 || layer13_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer14_ffn1_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(layer13_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = layer13_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer14_ffn1_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& layer13_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.14.ffn1.output_dense.bias");
    if (tokens == 0 || layer13_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer14_ffn1_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.14.ffn1_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.14.ffn1_layer_norm.bias.resident",
        norm_b,
        layer13_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.14.ffn1.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.14.ffn1.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(layer13_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer14_ffn1_residual(const std::string& model_bundle_dir,
                                                              const std::string& layer13_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer13_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_ffn2_residual");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer14_ffn1_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_ffn2_residual_f32\": \"" << json_escape(layer13_residual_f32) << "\",\n";
        std::cout << "  \"output_layer14_ffn1_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer14_ffn1_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer14_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer13_ffn2_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer14_ffn1_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer14_ffn1_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer14_ffn1_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer14_ffn1_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-14 self-attention, convolution, ffn2, and encoder layers 15-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer13_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer14_ffn1_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer14_ffn1_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

static std::vector<float> run_w2v_bert_layerN_final_norm_cpu(const mit2::Bundle& model,
                                                              const std::vector<float>& input,
                                                              uint32_t tokens,
                                                              int layer_idx) {
    const std::string key = "w2v_bert.encoder.layers." + std::to_string(layer_idx) + ".final_layer_norm";
    const auto norm_w = tensor_as_f32(model, key + ".weight");
    const auto norm_b = tensor_as_f32(model, key + ".bias");
    if (tokens == 0 || input.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layerN_final_norm invalid sizes for layer " + std::to_string(layer_idx));
    }
    return cpu_layer_norm_rows(input, norm_w, norm_b, tokens, 1024, 1e-5f);
}

static std::vector<float> run_w2v_bert_layerN_final_norm_metal(mit2::MetalContext& metal,
                                                                const mit2::Bundle& model,
                                                                const std::vector<float>& input,
                                                                uint32_t tokens,
                                                                int layer_idx) {
    const std::string key = "w2v_bert.encoder.layers." + std::to_string(layer_idx) + ".final_layer_norm";
    const std::string wk = key + ".weight";
    const std::string bk = key + ".bias";
    const auto norm_w = tensor_as_f32(model, wk);
    const auto norm_b = tensor_as_f32(model, bk);
    if (tokens == 0 || input.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layerN_final_norm invalid sizes for layer " + std::to_string(layer_idx));
    }
    return metal.layernorm_rows_f32_resident(wk + ".resident", norm_w, bk + ".resident", norm_b,
                                             input, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer17_final_norm_cpu(const mit2::Bundle& model,
                                                       const std::vector<float>& layer16,
                                                       uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.17.final_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.17.final_layer_norm.bias");
    if (tokens == 0 || layer16.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer17_final_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(layer16, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer17_final_norm_metal(mit2::MetalContext& metal,
                                                         const mit2::Bundle& model,
                                                         const std::vector<float>& layer16,
                                                         uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.17.final_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.17.final_layer_norm.bias");
    if (tokens == 0 || layer16.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer17_final_norm invalid input sizes");
    }
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.17.final_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.17.final_layer_norm.bias.resident",
                                             norm_b,
                                             layer16,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer17_final_norm(const std::string& model_bundle_dir,
                                                           const std::string& layer16_f32,
                                                           uint32_t tokens,
                                                           const std::string& output_hidden_state_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues, layer16_f32, static_cast<uint64_t>(tokens) * 1024u, "w2v_layer16");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* input,
                            const std::vector<float>* output) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer17_final_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer16_f32\": \"" << json_escape(layer16_f32) << "\",\n";
        std::cout << "  \"output_w2v_hidden_state_17_f32\": \"" << json_escape(output_hidden_state_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_w2v_hidden_state_17_sha256\": \"" << file_sha256_hex(output_hidden_state_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer16_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"w2v_hidden_state_17_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer16_values\": " << input->size() << ",\n";
            std::cout << "  \"w2v_hidden_state_17_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer17_final_norm_issues\": ";
        print_json_string_array(ok ? std::vector<std::string>{} : issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer17_final_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer17_final_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layers 2-16 to feed layer-17 final norm from audio\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr);
        return false;
    }
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer16_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer17_final_norm_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer17_final_norm_cpu(model, input, tokens);
        }
        write_raw_f32(output_hidden_state_f32, output);
        print_report(true, backend, &input, &output);
        return true;
    } catch (const std::exception& e) {
        issues = {e.what()};
        print_report(false, "", nullptr, nullptr);
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_w2v_normalize(const std::string& model_bundle_dir,
                                                  const std::string& hidden_state_f32,
                                                  uint32_t tokens,
                                                  const std::string& output_spk_cond_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               hidden_state_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_hidden_state_17");

    bool w2v_ok = false;
    size_t w2v_required = 0;
    size_t w2v_present = 0;
    std::vector<std::string> w2v_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto contract = inspect_w2v_bert_model_contract(model);
        w2v_ok = contract.ok;
        w2v_required = contract.specs.size();
        w2v_present = contract.present_tensors.size();
        w2v_issues = contract.issues;
        if (!contract.ok) {
            issues.push_back("w2v_bert_model_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        w2v_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_normalize\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_hidden_state_17_f32\": \"" << json_escape(hidden_state_f32) << "\",\n";
        std::cout << "  \"output_spk_cond_f32\": \"" << json_escape(output_spk_cond_f32) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"spk_cond_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_normalize_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_stats_normalize\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_stats_normalize\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT feature extractor/encoder forward to produce hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto hidden = read_raw_f32(hidden_state_f32);
        std::string execution_backend = "metal";
        std::vector<float> spk_cond;
        try {
            mit2::MetalContext metal;
            spk_cond = run_w2v_bert_normalize_metal(metal, model, hidden, tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            spk_cond = run_w2v_bert_normalize_cpu(model, hidden, tokens);
        }
        write_raw_f32(output_spk_cond_f32, spk_cond);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_normalize\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_hidden_state_17_f32\": \"" << json_escape(hidden_state_f32) << "\",\n";
        std::cout << "  \"output_spk_cond_f32\": \"" << json_escape(output_spk_cond_f32) << "\",\n";
        std::cout << "  \"output_spk_cond_sha256\": \"" << file_sha256_hex(output_spk_cond_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"w2v_hidden_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"spk_cond_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"w2v_hidden_values\": " << hidden.size() << ",\n";
        std::cout << "  \"spk_cond_values\": " << spk_cond.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_normalize_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_stats_normalize\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_stats_normalize\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT feature extractor/encoder forward to produce hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_normalize\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_hidden_state_17_f32\": \"" << json_escape(hidden_state_f32) << "\",\n";
        std::cout << "  \"output_spk_cond_f32\": \"" << json_escape(output_spk_cond_f32) << "\",\n";
        std::cout << "  \"clone_w2v_normalize_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_w2v_bert_stats_normalize\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_stats_normalize\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT feature extractor/encoder forward to produce hidden_state_17\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_semantic_quantize(const std::string& model_bundle_dir,
                                                      const std::string& spk_cond_f32,
                                                      uint32_t spk_tokens,
                                                      const std::string& output_sref_f32,
                                                      const std::string& output_codes_u32) {
    std::vector<std::string> issues;
    if (spk_tokens == 0) {
        issues.push_back("spk_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               spk_cond_f32,
                               static_cast<uint64_t>(spk_tokens) * 1024u,
                               "spk_cond_emb");

    bool semantic_quantize_ok = false;
    size_t semantic_quantize_required = 0;
    size_t semantic_quantize_present = 0;
    std::vector<std::string> semantic_quantize_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto semantic_quantize = inspect_semantic_codec_quantize_contract(model);
        semantic_quantize_ok = semantic_quantize.ok;
        semantic_quantize_required = semantic_quantize.specs.size();
        semantic_quantize_present = semantic_quantize.present_tensors.size();
        semantic_quantize_issues = semantic_quantize.issues;
        if (!semantic_quantize.ok) {
            issues.push_back("semantic_codec_quantize_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        semantic_quantize_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_semantic_quantize\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
        std::cout << "  \"output_sref_f32\": \"" << json_escape(output_sref_f32) << "\",\n";
        std::cout << "  \"output_codes_u32\": \"" << json_escape(output_codes_u32) << "\",\n";
        std::cout << "  \"spk_tokens\": " << spk_tokens << ",\n";
        std::cout << "  \"sref_values\": " << (static_cast<uint64_t>(spk_tokens) * 1024u) << ",\n";
        std::cout << "  \"has_semantic_codec_quantize_contract\": " << (semantic_quantize_ok ? "true" : "false") << ",\n";
        std::cout << "  \"semantic_codec_quantize_required_tensor_count\": " << semantic_quantize_required << ",\n";
        std::cout << "  \"semantic_codec_quantize_required_tensors_present\": " << semantic_quantize_present << ",\n";
        std::cout << "  \"semantic_codec_quantize_contract_issues\": ";
        print_json_string_array(semantic_quantize_issues);
        std::cout << ",\n";
        std::cout << "  \"semantic_codec_quantize_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": false,\n";
        std::cout << "  \"ready_metal_semantic_codec_quantize_from_spk_cond\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto spk_cond = read_raw_f32(spk_cond_f32);
        std::string execution_backend = "metal";
        SemanticQuantizeResult quantized;
        try {
            mit2::MetalContext metal;
            quantized = run_semantic_codec_quantize_metal(metal, model, spk_cond, spk_tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            quantized = run_semantic_codec_quantize_cpu(model, spk_cond, spk_tokens);
        }
        write_raw_f32(output_sref_f32, quantized.sref);
        write_raw_u32(output_codes_u32, quantized.codes);

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_semantic_quantize\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
        std::cout << "  \"output_sref_f32\": \"" << json_escape(output_sref_f32) << "\",\n";
        std::cout << "  \"output_sref_sha256\": \"" << file_sha256_hex(output_sref_f32) << "\",\n";
        std::cout << "  \"output_codes_u32\": \"" << json_escape(output_codes_u32) << "\",\n";
        std::cout << "  \"output_codes_sha256\": \"" << file_sha256_hex(output_codes_u32) << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"spk_tokens\": " << spk_tokens << ",\n";
        std::cout << "  \"spk_cond_shape\": \"[1," << spk_tokens << ",1024]\",\n";
        std::cout << "  \"sref_shape\": \"[1," << spk_tokens << ",1024]\",\n";
        std::cout << "  \"semantic_codes_shape\": \"[1," << spk_tokens << "]\",\n";
        std::cout << "  \"spk_cond_values\": " << spk_cond.size() << ",\n";
        std::cout << "  \"sref_values\": " << quantized.sref.size() << ",\n";
        std::cout << "  \"semantic_code_count\": " << quantized.codes.size() << ",\n";
        std::cout << "  \"codebook_size\": 8192,\n";
        std::cout << "  \"codebook_dim\": 8,\n";
        std::cout << "  \"has_semantic_codec_quantize_contract\": true,\n";
        std::cout << "  \"semantic_codec_quantize_required_tensor_count\": " << semantic_quantize_required << ",\n";
        std::cout << "  \"semantic_codec_quantize_required_tensors_present\": " << semantic_quantize_present << ",\n";
        std::cout << "  \"semantic_codec_quantize_contract_issues\": ";
        print_json_string_array(semantic_quantize_issues);
        std::cout << ",\n";
        std::cout << "  \"semantic_codec_quantize_issues\": [],\n";
        std::cout << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": true,\n";
        std::cout << "  \"ready_metal_semantic_codec_quantize_from_spk_cond\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_semantic_quantize\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
        std::cout << "  \"output_sref_f32\": \"" << json_escape(output_sref_f32) << "\",\n";
        std::cout << "  \"output_codes_u32\": \"" << json_escape(output_codes_u32) << "\",\n";
        std::cout << "  \"semantic_codec_quantize_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": false,\n";
        std::cout << "  \"ready_metal_semantic_codec_quantize_from_spk_cond\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_s2mel_prompt_from_sref(const std::string& model_bundle_dir,
                                                           const std::string& feature_manifest,
                                                           const std::string& sref_f32,
                                                           uint32_t sref_tokens,
                                                           const std::string& output_s2mel_prompt_f32) {
    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    std::vector<std::string> issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
    if (sref_tokens == 0) {
        issues.push_back("sref_tokens_must_be_positive");
    }
    if (manifest.mel_frames == 0) {
        issues.push_back("feature_manifest_mel_frames_must_be_positive");
    }
    if (manifest.mel_frames > std::numeric_limits<uint32_t>::max()) {
        issues.push_back("feature_manifest_mel_frames_exceeds_native_length_regulator_limit");
    }
    append_raw_f32_count_issue(issues,
                               sref_f32,
                               static_cast<uint64_t>(sref_tokens) * 1024u,
                               "sref");

    bool prompt_condition_ok = false;
    size_t prompt_condition_required = 0;
    size_t prompt_condition_present = 0;
    std::vector<std::string> prompt_condition_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto prompt_condition = inspect_s2mel_prompt_condition_contract(model);
        prompt_condition_ok = prompt_condition.ok;
        prompt_condition_required = prompt_condition.specs.size();
        prompt_condition_present = prompt_condition.present_tensors.size();
        prompt_condition_issues = prompt_condition.issues;
        if (!prompt_condition.ok) {
            issues.push_back("s2mel_prompt_condition_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        prompt_condition_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_s2mel_prompt_from_sref\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"sref_f32\": \"" << json_escape(sref_f32) << "\",\n";
        std::cout << "  \"output_s2mel_prompt_f32\": \"" << json_escape(output_s2mel_prompt_f32) << "\",\n";
        std::cout << "  \"sref_tokens\": " << sref_tokens << ",\n";
        std::cout << "  \"prompt_tokens\": " << manifest.mel_frames << ",\n";
        std::cout << "  \"s2mel_prompt_values\": " << (manifest.mel_frames * 512u) << ",\n";
        std::cout << "  \"has_s2mel_prompt_condition_contract\": " << (prompt_condition_ok ? "true" : "false") << ",\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensor_count\": " << prompt_condition_required << ",\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensors_present\": " << prompt_condition_present << ",\n";
        std::cout << "  \"s2mel_prompt_condition_contract_issues\": ";
        print_json_string_array(prompt_condition_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_s2mel_prompt_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_s2mel_prompt_condition_from_sref\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const std::vector<float> sref = read_raw_f32(sref_f32);
        const uint32_t prompt_tokens = static_cast<uint32_t>(manifest.mel_frames);
        std::string execution_backend = "metal";
        std::vector<float> prompt;
        try {
            mit2::MetalContext metal;
            prompt = run_length_regulator_full_metal(metal, model, sref, sref_tokens, prompt_tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            prompt = run_length_regulator_full_cpu(model, sref, sref_tokens, prompt_tokens);
        }
        write_raw_f32(output_s2mel_prompt_f32, prompt);
        const std::string output_sha = file_sha256_hex(output_s2mel_prompt_f32);

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_s2mel_prompt_from_sref\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"sref_f32\": \"" << json_escape(sref_f32) << "\",\n";
        std::cout << "  \"output_s2mel_prompt_f32\": \"" << json_escape(output_s2mel_prompt_f32) << "\",\n";
        std::cout << "  \"output_s2mel_prompt_sha256\": \"" << output_sha << "\",\n";
        std::cout << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        std::cout << "  \"sref_tokens\": " << sref_tokens << ",\n";
        std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
        std::cout << "  \"feature_mel_frames\": " << manifest.mel_frames << ",\n";
        std::cout << "  \"sref_values\": " << sref.size() << ",\n";
        std::cout << "  \"s2mel_prompt_values\": " << prompt.size() << ",\n";
        std::cout << "  \"sref_shape\": \"[1," << sref_tokens << ",1024]\",\n";
        std::cout << "  \"s2mel_prompt_shape\": \"[1," << prompt_tokens << ",512]\",\n";
        std::cout << "  \"has_s2mel_prompt_condition_contract\": true,\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensor_count\": " << prompt_condition_required << ",\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensors_present\": " << prompt_condition_present << ",\n";
        std::cout << "  \"s2mel_prompt_condition_contract_issues\": ";
        print_json_string_array(prompt_condition_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_s2mel_prompt_issues\": [],\n";
        std::cout << "  \"ready_native_s2mel_prompt_condition_from_sref\": true,\n";
        std::cout << "  \"ready_metal_s2mel_prompt_condition_from_sref\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_s2mel_prompt_from_sref\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"sref_f32\": \"" << json_escape(sref_f32) << "\",\n";
        std::cout << "  \"output_s2mel_prompt_f32\": \"" << json_escape(output_s2mel_prompt_f32) << "\",\n";
        std::cout << "  \"clone_s2mel_prompt_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_s2mel_prompt_condition_from_sref\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_semantic_prompt_from_spk_cond(const std::string& model_bundle_dir,
                                                                  const std::string& feature_manifest,
                                                                  const std::string& spk_cond_f32,
                                                                  uint32_t spk_tokens,
                                                                  const std::string& output_dir) {
    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    std::vector<std::string> issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);
    if (spk_tokens == 0) {
        issues.push_back("spk_tokens_must_be_positive");
    }
    if (manifest.mel_frames == 0) {
        issues.push_back("feature_manifest_mel_frames_must_be_positive");
    }
    if (manifest.mel_frames > std::numeric_limits<uint32_t>::max()) {
        issues.push_back("feature_manifest_mel_frames_exceeds_native_length_regulator_limit");
    }
    append_raw_f32_count_issue(issues,
                               spk_cond_f32,
                               static_cast<uint64_t>(spk_tokens) * 1024u,
                               "spk_cond_emb");

    bool semantic_quantize_ok = false;
    size_t semantic_quantize_required = 0;
    size_t semantic_quantize_present = 0;
    std::vector<std::string> semantic_quantize_issues;
    bool prompt_condition_ok = false;
    size_t prompt_condition_required = 0;
    size_t prompt_condition_present = 0;
    std::vector<std::string> prompt_condition_issues;
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto semantic_quantize = inspect_semantic_codec_quantize_contract(model);
        semantic_quantize_ok = semantic_quantize.ok;
        semantic_quantize_required = semantic_quantize.specs.size();
        semantic_quantize_present = semantic_quantize.present_tensors.size();
        semantic_quantize_issues = semantic_quantize.issues;
        if (!semantic_quantize.ok) {
            issues.push_back("semantic_codec_quantize_contract_not_ready");
        }
        const auto prompt_condition = inspect_s2mel_prompt_condition_contract(model);
        prompt_condition_ok = prompt_condition.ok;
        prompt_condition_required = prompt_condition.specs.size();
        prompt_condition_present = prompt_condition.present_tensors.size();
        prompt_condition_issues = prompt_condition.issues;
        if (!prompt_condition.ok) {
            issues.push_back("s2mel_prompt_condition_contract_not_ready");
        }
    } catch (const std::exception& e) {
        issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        semantic_quantize_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
        prompt_condition_issues.push_back(std::string("model_bundle_parse_error: ") + e.what());
    }

    const auto dir = std::filesystem::path(output_dir);
    const auto output_sref_f32 = (dir / "s_ref.f32").string();
    const auto output_codes_u32 = (dir / "semantic_codes.u32").string();
    const auto output_prompt_f32 = (dir / "s2mel_prompt.f32").string();
    const auto output_manifest = (dir / "semantic_prompt_sidecars.manifest.json").string();
    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_semantic_prompt_from_spk_cond\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"output_sref_f32\": \"" << json_escape(output_sref_f32) << "\",\n";
        std::cout << "  \"output_codes_u32\": \"" << json_escape(output_codes_u32) << "\",\n";
        std::cout << "  \"output_s2mel_prompt_f32\": \"" << json_escape(output_prompt_f32) << "\",\n";
        std::cout << "  \"spk_tokens\": " << spk_tokens << ",\n";
        std::cout << "  \"prompt_tokens\": " << manifest.mel_frames << ",\n";
        std::cout << "  \"has_semantic_codec_quantize_contract\": " << (semantic_quantize_ok ? "true" : "false") << ",\n";
        std::cout << "  \"semantic_codec_quantize_required_tensor_count\": " << semantic_quantize_required << ",\n";
        std::cout << "  \"semantic_codec_quantize_required_tensors_present\": " << semantic_quantize_present << ",\n";
        std::cout << "  \"semantic_codec_quantize_contract_issues\": ";
        print_json_string_array(semantic_quantize_issues);
        std::cout << ",\n";
        std::cout << "  \"has_s2mel_prompt_condition_contract\": " << (prompt_condition_ok ? "true" : "false") << ",\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensor_count\": " << prompt_condition_required << ",\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensors_present\": " << prompt_condition_present << ",\n";
        std::cout << "  \"s2mel_prompt_condition_contract_issues\": ";
        print_json_string_array(prompt_condition_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_semantic_prompt_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": false,\n";
        std::cout << "  \"ready_native_s2mel_prompt_condition_from_sref\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        std::filesystem::create_directories(dir);
        mit2::Bundle model(model_bundle_dir);
        const auto spk_cond = read_raw_f32(spk_cond_f32);

        std::string semantic_backend = "metal";
        SemanticQuantizeResult quantized;
        try {
            mit2::MetalContext metal;
            quantized = run_semantic_codec_quantize_metal(metal, model, spk_cond, spk_tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            semantic_backend = "cpu_fallback";
            quantized = run_semantic_codec_quantize_cpu(model, spk_cond, spk_tokens);
        }
        write_raw_f32(output_sref_f32, quantized.sref);
        write_raw_u32(output_codes_u32, quantized.codes);

        const uint32_t prompt_tokens = static_cast<uint32_t>(manifest.mel_frames);
        std::string prompt_backend = "metal";
        std::vector<float> prompt;
        try {
            mit2::MetalContext metal;
            prompt = run_length_regulator_full_metal(metal, model, quantized.sref, spk_tokens, prompt_tokens);
        } catch (const std::exception& e) {
            const std::string error = e.what();
            if (error.find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            prompt_backend = "cpu_fallback";
            prompt = run_length_regulator_full_cpu(model, quantized.sref, spk_tokens, prompt_tokens);
        }
        write_raw_f32(output_prompt_f32, prompt);

        std::ostringstream manifest_json;
        manifest_json << "{\n";
        manifest_json << "  \"format\": \"mit2-clone-semantic-prompt-sidecars\",\n";
        manifest_json << "  \"version\": 1,\n";
        manifest_json << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        manifest_json << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
        manifest_json << "  \"spk_tokens\": " << spk_tokens << ",\n";
        manifest_json << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
        manifest_json << "  \"output_sref_f32\": \"" << json_escape(output_sref_f32) << "\",\n";
        manifest_json << "  \"output_sref_sha256\": \"" << file_sha256_hex(output_sref_f32) << "\",\n";
        manifest_json << "  \"output_codes_u32\": \"" << json_escape(output_codes_u32) << "\",\n";
        manifest_json << "  \"output_codes_sha256\": \"" << file_sha256_hex(output_codes_u32) << "\",\n";
        manifest_json << "  \"output_s2mel_prompt_f32\": \"" << json_escape(output_prompt_f32) << "\",\n";
        manifest_json << "  \"output_s2mel_prompt_sha256\": \"" << file_sha256_hex(output_prompt_f32) << "\",\n";
        manifest_json << "  \"semantic_execution_backend\": \"" << semantic_backend << "\",\n";
        manifest_json << "  \"prompt_execution_backend\": \"" << prompt_backend << "\",\n";
        manifest_json << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": true,\n";
        manifest_json << "  \"ready_native_s2mel_prompt_condition_from_sref\": true,\n";
        manifest_json << "  \"ready_native_voice_clone\": false\n";
        manifest_json << "}\n";
        write_text_file(output_manifest, manifest_json.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_semantic_prompt_from_spk_cond\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"output_manifest\": \"" << json_escape(output_manifest) << "\",\n";
        std::cout << "  \"output_sref_f32\": \"" << json_escape(output_sref_f32) << "\",\n";
        std::cout << "  \"output_sref_sha256\": \"" << file_sha256_hex(output_sref_f32) << "\",\n";
        std::cout << "  \"output_codes_u32\": \"" << json_escape(output_codes_u32) << "\",\n";
        std::cout << "  \"output_codes_sha256\": \"" << file_sha256_hex(output_codes_u32) << "\",\n";
        std::cout << "  \"output_s2mel_prompt_f32\": \"" << json_escape(output_prompt_f32) << "\",\n";
        std::cout << "  \"output_s2mel_prompt_sha256\": \"" << file_sha256_hex(output_prompt_f32) << "\",\n";
        std::cout << "  \"semantic_execution_backend\": \"" << semantic_backend << "\",\n";
        std::cout << "  \"prompt_execution_backend\": \"" << prompt_backend << "\",\n";
        std::cout << "  \"spk_tokens\": " << spk_tokens << ",\n";
        std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
        std::cout << "  \"spk_cond_shape\": \"[1," << spk_tokens << ",1024]\",\n";
        std::cout << "  \"sref_shape\": \"[1," << spk_tokens << ",1024]\",\n";
        std::cout << "  \"semantic_codes_shape\": \"[1," << spk_tokens << "]\",\n";
        std::cout << "  \"s2mel_prompt_shape\": \"[1," << prompt_tokens << ",512]\",\n";
        std::cout << "  \"spk_cond_values\": " << spk_cond.size() << ",\n";
        std::cout << "  \"sref_values\": " << quantized.sref.size() << ",\n";
        std::cout << "  \"semantic_code_count\": " << quantized.codes.size() << ",\n";
        std::cout << "  \"s2mel_prompt_values\": " << prompt.size() << ",\n";
        std::cout << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": true,\n";
        std::cout << "  \"ready_metal_semantic_codec_quantize_from_spk_cond\": " << (semantic_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_s2mel_prompt_condition_from_sref\": true,\n";
        std::cout << "  \"ready_metal_s2mel_prompt_condition_from_sref\": " << (prompt_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_semantic_prompt_from_spk_cond\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"spk_cond_f32\": \"" << json_escape(spk_cond_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"clone_semantic_prompt_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": false,\n";
        std::cout << "  \"ready_native_s2mel_prompt_condition_from_sref\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }
}

[[maybe_unused]] bool run_tts_clone_campplus_head_golden(const std::string& model_bundle_dir,
                                                         const std::string& feature_manifest,
                                                         const std::string& campplus_golden_dir) {
    try {
        CloneFeatureManifest manifest;
        ClonePreprocessManifest preprocess_manifest;
        auto issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);

        mit2::Bundle model(model_bundle_dir);
        const auto campplus = inspect_campplus_model_contract(model);
        if (!campplus.ok) {
            issues.push_back("campplus_model_contract_not_ready");
        }

        const std::vector<float> fbank = read_raw_f32(manifest.output_fbank_f32);
        const std::vector<float> expected = read_raw_f32(campplus_golden_dir + "/campplus_head_conv1_bn_relu.f32");
        const std::vector<float> expected_layer1 = read_raw_f32(campplus_golden_dir + "/campplus_head_layer1.f32");
        const std::vector<float> expected_layer2 = read_raw_f32(campplus_golden_dir + "/campplus_head_layer2.f32");
        const std::vector<float> expected_conv2 = read_raw_f32(campplus_golden_dir + "/campplus_head_conv2_bn_relu.f32");
        const std::vector<float> expected_output = read_raw_f32(campplus_golden_dir + "/campplus_head_output.f32");
        const std::vector<float> expected_tdnn = read_raw_f32(campplus_golden_dir + "/campplus_xvector_tdnn.f32");
        const std::vector<float> expected_block1_tdnnd1 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd1.f32");
        const std::vector<float> expected_block1_after_tdnnd1 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd1.f32");
        const std::vector<float> expected_block1_tdnnd2 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd2.f32");
        const std::vector<float> expected_block1_after_tdnnd2 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd2.f32");
        const std::vector<float> expected_block1_tdnnd3 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd3.f32");
        const std::vector<float> expected_block1_after_tdnnd3 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd3.f32");
        const std::vector<float> expected_block1_tdnnd4 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd4.f32");
        const std::vector<float> expected_block1_after_tdnnd4 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd4.f32");
        const std::vector<float> expected_block1_tdnnd5 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd5.f32");
        const std::vector<float> expected_block1_after_tdnnd5 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd5.f32");
        const std::vector<float> expected_block1_tdnnd6 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd6.f32");
        const std::vector<float> expected_block1_after_tdnnd6 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd6.f32");
        const std::vector<float> expected_block1_tdnnd7 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd7.f32");
        const std::vector<float> expected_block1_after_tdnnd7 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd7.f32");
        const std::vector<float> expected_block1_tdnnd8 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd8.f32");
        const std::vector<float> expected_block1_after_tdnnd8 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd8.f32");
        const std::vector<float> expected_block1_tdnnd9 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd9.f32");
        const std::vector<float> expected_block1_after_tdnnd9 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd9.f32");
        const std::vector<float> expected_block1_tdnnd10 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd10.f32");
        const std::vector<float> expected_block1_after_tdnnd10 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd10.f32");
        const std::vector<float> expected_block1_tdnnd11 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd11.f32");
        const std::vector<float> expected_block1_after_tdnnd11 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd11.f32");
        const std::vector<float> expected_block1_tdnnd12 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_tdnnd12.f32");
        const std::vector<float> expected_block1_after_tdnnd12 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block1_after_tdnnd12.f32");
        const std::vector<float> expected_transit1 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_transit1.f32");
        const std::vector<float> expected_block2_tdnnd1 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd1.f32");
        const std::vector<float> expected_block2_after_tdnnd1 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd1.f32");
        const std::vector<float> expected_block2_tdnnd2 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd2.f32");
        const std::vector<float> expected_block2_after_tdnnd2 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd2.f32");
        const std::vector<float> expected_block2_tdnnd3 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd3.f32");
        const std::vector<float> expected_block2_after_tdnnd3 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd3.f32");
        const std::vector<float> expected_block2_tdnnd4 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd4.f32");
        const std::vector<float> expected_block2_after_tdnnd4 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd4.f32");
        const std::vector<float> expected_block2_tdnnd5 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd5.f32");
        const std::vector<float> expected_block2_after_tdnnd5 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd5.f32");
        const std::vector<float> expected_block2_tdnnd6 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd6.f32");
        const std::vector<float> expected_block2_after_tdnnd6 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd6.f32");
        const std::vector<float> expected_block2_tdnnd7 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd7.f32");
        const std::vector<float> expected_block2_after_tdnnd7 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd7.f32");
        const std::vector<float> expected_block2_tdnnd8 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd8.f32");
        const std::vector<float> expected_block2_after_tdnnd8 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd8.f32");
        const std::vector<float> expected_block2_tdnnd9 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd9.f32");
        const std::vector<float> expected_block2_after_tdnnd9 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd9.f32");
        const std::vector<float> expected_block2_tdnnd10 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd10.f32");
        const std::vector<float> expected_block2_after_tdnnd10 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd10.f32");
        const std::vector<float> expected_block2_tdnnd11 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd11.f32");
        const std::vector<float> expected_block2_after_tdnnd11 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd11.f32");
        const std::vector<float> expected_block2_tdnnd12 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd12.f32");
        const std::vector<float> expected_block2_after_tdnnd12 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd12.f32");
        const std::vector<float> expected_block2_tdnnd13 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd13.f32");
        const std::vector<float> expected_block2_after_tdnnd13 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd13.f32");
        const std::vector<float> expected_block2_tdnnd14 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd14.f32");
        const std::vector<float> expected_block2_after_tdnnd14 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd14.f32");
        const std::vector<float> expected_block2_tdnnd15 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd15.f32");
        const std::vector<float> expected_block2_after_tdnnd15 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd15.f32");
        const std::vector<float> expected_block2_tdnnd16 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd16.f32");
        const std::vector<float> expected_block2_after_tdnnd16 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd16.f32");
        const std::vector<float> expected_block2_tdnnd17 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd17.f32");
        const std::vector<float> expected_block2_after_tdnnd17 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd17.f32");
        const std::vector<float> expected_block2_tdnnd18 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd18.f32");
        const std::vector<float> expected_block2_after_tdnnd18 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd18.f32");
        const std::vector<float> expected_block2_tdnnd19 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd19.f32");
        const std::vector<float> expected_block2_after_tdnnd19 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd19.f32");
        const std::vector<float> expected_block2_tdnnd20 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd20.f32");
        const std::vector<float> expected_block2_after_tdnnd20 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd20.f32");
        const std::vector<float> expected_block2_tdnnd21 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd21.f32");
        const std::vector<float> expected_block2_after_tdnnd21 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd21.f32");
        const std::vector<float> expected_block2_tdnnd22 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd22.f32");
        const std::vector<float> expected_block2_after_tdnnd22 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd22.f32");
        const std::vector<float> expected_block2_tdnnd23 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd23.f32");
        const std::vector<float> expected_block2_after_tdnnd23 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd23.f32");
        const std::vector<float> expected_block2_tdnnd24 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_tdnnd24.f32");
        const std::vector<float> expected_block2_after_tdnnd24 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block2_after_tdnnd24.f32");
        const std::vector<float> expected_transit2 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_transit2.f32");
        const std::vector<float> expected_block3_tdnnd1 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd1.f32");
        const std::vector<float> expected_block3_after_tdnnd1 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd1.f32");
        const std::vector<float> expected_block3_tdnnd2 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd2.f32");
        const std::vector<float> expected_block3_after_tdnnd2 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd2.f32");
        const std::vector<float> expected_block3_tdnnd3 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd3.f32");
        const std::vector<float> expected_block3_after_tdnnd3 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd3.f32");
        const std::vector<float> expected_block3_tdnnd4 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd4.f32");
        const std::vector<float> expected_block3_after_tdnnd4 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd4.f32");
        const std::vector<float> expected_block3_tdnnd5 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd5.f32");
        const std::vector<float> expected_block3_after_tdnnd5 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd5.f32");
        const std::vector<float> expected_block3_tdnnd6 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd6.f32");
        const std::vector<float> expected_block3_after_tdnnd6 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd6.f32");
        const std::vector<float> expected_block3_tdnnd7 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd7.f32");
        const std::vector<float> expected_block3_after_tdnnd7 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd7.f32");
        const std::vector<float> expected_block3_tdnnd8 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd8.f32");
        const std::vector<float> expected_block3_after_tdnnd8 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd8.f32");
        const std::vector<float> expected_block3_tdnnd9 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd9.f32");
        const std::vector<float> expected_block3_after_tdnnd9 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd9.f32");
        const std::vector<float> expected_block3_tdnnd10 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd10.f32");
        const std::vector<float> expected_block3_after_tdnnd10 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd10.f32");
        const std::vector<float> expected_block3_tdnnd11 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd11.f32");
        const std::vector<float> expected_block3_after_tdnnd11 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd11.f32");
        const std::vector<float> expected_block3_tdnnd12 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd12.f32");
        const std::vector<float> expected_block3_after_tdnnd12 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd12.f32");
        const std::vector<float> expected_block3_tdnnd13 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd13.f32");
        const std::vector<float> expected_block3_after_tdnnd13 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd13.f32");
        const std::vector<float> expected_block3_tdnnd14 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd14.f32");
        const std::vector<float> expected_block3_after_tdnnd14 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd14.f32");
        const std::vector<float> expected_block3_tdnnd15 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd15.f32");
        const std::vector<float> expected_block3_after_tdnnd15 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd15.f32");
        const std::vector<float> expected_block3_tdnnd16 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_tdnnd16.f32");
        const std::vector<float> expected_block3_after_tdnnd16 =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_block3_after_tdnnd16.f32");
        const std::vector<float> expected_transit3 = read_raw_f32(campplus_golden_dir + "/campplus_xvector_transit3.f32");
        const std::vector<float> expected_out_nonlinear =
            read_raw_f32(campplus_golden_dir + "/campplus_xvector_out_nonlinear.f32");
        const std::vector<float> expected_stats = read_raw_f32(campplus_golden_dir + "/campplus_xvector_stats.f32");
        const std::vector<float> expected_dense = read_raw_f32(campplus_golden_dir + "/campplus_xvector_dense.f32");
        const uint64_t expected_values = static_cast<uint64_t>(manifest.fbank_frames) * 80u * 32u;
        const uint32_t layer1_freq = 40u;
        const uint32_t layer2_freq = 20u;
        const uint32_t final_freq = 10u;
        const uint64_t expected_layer1_values = static_cast<uint64_t>(manifest.fbank_frames) * layer1_freq * 32u;
        const uint64_t expected_layer2_values = static_cast<uint64_t>(manifest.fbank_frames) * layer2_freq * 32u;
        const uint64_t expected_conv2_values = static_cast<uint64_t>(manifest.fbank_frames) * final_freq * 32u;
        const uint64_t expected_output_values = static_cast<uint64_t>(manifest.fbank_frames) * 320u;
        const uint32_t expected_tdnn_frames = ((static_cast<uint32_t>(manifest.fbank_frames) + 4u - 5u) / 2u) + 1u;
        const uint64_t expected_tdnn_values = static_cast<uint64_t>(expected_tdnn_frames) * 128u;
        const uint64_t expected_block1_tdnnd1_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd1_values = static_cast<uint64_t>(expected_tdnn_frames) * 160u;
        const uint64_t expected_block1_tdnnd2_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd2_values = static_cast<uint64_t>(expected_tdnn_frames) * 192u;
        const uint64_t expected_block1_tdnnd3_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd3_values = static_cast<uint64_t>(expected_tdnn_frames) * 224u;
        const uint64_t expected_block1_tdnnd4_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd4_values = static_cast<uint64_t>(expected_tdnn_frames) * 256u;
        const uint64_t expected_block1_tdnnd5_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd5_values = static_cast<uint64_t>(expected_tdnn_frames) * 288u;
        const uint64_t expected_block1_tdnnd6_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd6_values = static_cast<uint64_t>(expected_tdnn_frames) * 320u;
        const uint64_t expected_block1_tdnnd7_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd7_values = static_cast<uint64_t>(expected_tdnn_frames) * 352u;
        const uint64_t expected_block1_tdnnd8_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd8_values = static_cast<uint64_t>(expected_tdnn_frames) * 384u;
        const uint64_t expected_block1_tdnnd9_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd9_values = static_cast<uint64_t>(expected_tdnn_frames) * 416u;
        const uint64_t expected_block1_tdnnd10_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd10_values = static_cast<uint64_t>(expected_tdnn_frames) * 448u;
        const uint64_t expected_block1_tdnnd11_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd11_values = static_cast<uint64_t>(expected_tdnn_frames) * 480u;
        const uint64_t expected_block1_tdnnd12_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block1_after_tdnnd12_values = static_cast<uint64_t>(expected_tdnn_frames) * 512u;
        const uint64_t expected_transit1_values = static_cast<uint64_t>(expected_tdnn_frames) * 256u;
        const uint64_t expected_block2_tdnnd1_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd1_values = static_cast<uint64_t>(expected_tdnn_frames) * 288u;
        const uint64_t expected_block2_tdnnd2_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd2_values = static_cast<uint64_t>(expected_tdnn_frames) * 320u;
        const uint64_t expected_block2_tdnnd3_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd3_values = static_cast<uint64_t>(expected_tdnn_frames) * 352u;
        const uint64_t expected_block2_tdnnd4_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd4_values = static_cast<uint64_t>(expected_tdnn_frames) * 384u;
        const uint64_t expected_block2_tdnnd5_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd5_values = static_cast<uint64_t>(expected_tdnn_frames) * 416u;
        const uint64_t expected_block2_tdnnd6_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd6_values = static_cast<uint64_t>(expected_tdnn_frames) * 448u;
        const uint64_t expected_block2_tdnnd7_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd7_values = static_cast<uint64_t>(expected_tdnn_frames) * 480u;
        const uint64_t expected_block2_tdnnd8_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd8_values = static_cast<uint64_t>(expected_tdnn_frames) * 512u;
        const uint64_t expected_block2_tdnnd9_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd9_values = static_cast<uint64_t>(expected_tdnn_frames) * 544u;
        const uint64_t expected_block2_tdnnd10_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd10_values = static_cast<uint64_t>(expected_tdnn_frames) * 576u;
        const uint64_t expected_block2_tdnnd11_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd11_values = static_cast<uint64_t>(expected_tdnn_frames) * 608u;
        const uint64_t expected_block2_tdnnd12_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd12_values = static_cast<uint64_t>(expected_tdnn_frames) * 640u;
        const uint64_t expected_block2_tdnnd13_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd13_values = static_cast<uint64_t>(expected_tdnn_frames) * 672u;
        const uint64_t expected_block2_tdnnd14_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd14_values = static_cast<uint64_t>(expected_tdnn_frames) * 704u;
        const uint64_t expected_block2_tdnnd15_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd15_values = static_cast<uint64_t>(expected_tdnn_frames) * 736u;
        const uint64_t expected_block2_tdnnd16_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd16_values = static_cast<uint64_t>(expected_tdnn_frames) * 768u;
        const uint64_t expected_block2_tdnnd17_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd17_values = static_cast<uint64_t>(expected_tdnn_frames) * 800u;
        const uint64_t expected_block2_tdnnd18_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd18_values = static_cast<uint64_t>(expected_tdnn_frames) * 832u;
        const uint64_t expected_block2_tdnnd19_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd19_values = static_cast<uint64_t>(expected_tdnn_frames) * 864u;
        const uint64_t expected_block2_tdnnd20_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd20_values = static_cast<uint64_t>(expected_tdnn_frames) * 896u;
        const uint64_t expected_block2_tdnnd21_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd21_values = static_cast<uint64_t>(expected_tdnn_frames) * 928u;
        const uint64_t expected_block2_tdnnd22_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd22_values = static_cast<uint64_t>(expected_tdnn_frames) * 960u;
        const uint64_t expected_block2_tdnnd23_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd23_values = static_cast<uint64_t>(expected_tdnn_frames) * 992u;
        const uint64_t expected_block2_tdnnd24_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block2_after_tdnnd24_values = static_cast<uint64_t>(expected_tdnn_frames) * 1024u;
        const uint64_t expected_transit2_values = static_cast<uint64_t>(expected_tdnn_frames) * 512u;
        const uint64_t expected_block3_tdnnd1_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd1_values = static_cast<uint64_t>(expected_tdnn_frames) * 544u;
        const uint64_t expected_block3_tdnnd2_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd2_values = static_cast<uint64_t>(expected_tdnn_frames) * 576u;
        const uint64_t expected_block3_tdnnd3_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd3_values = static_cast<uint64_t>(expected_tdnn_frames) * 608u;
        const uint64_t expected_block3_tdnnd4_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd4_values = static_cast<uint64_t>(expected_tdnn_frames) * 640u;
        const uint64_t expected_block3_tdnnd5_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd5_values = static_cast<uint64_t>(expected_tdnn_frames) * 672u;
        const uint64_t expected_block3_tdnnd6_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd6_values = static_cast<uint64_t>(expected_tdnn_frames) * 704u;
        const uint64_t expected_block3_tdnnd7_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd7_values = static_cast<uint64_t>(expected_tdnn_frames) * 736u;
        const uint64_t expected_block3_tdnnd8_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd8_values = static_cast<uint64_t>(expected_tdnn_frames) * 768u;
        const uint64_t expected_block3_tdnnd9_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd9_values = static_cast<uint64_t>(expected_tdnn_frames) * 800u;
        const uint64_t expected_block3_tdnnd10_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd10_values = static_cast<uint64_t>(expected_tdnn_frames) * 832u;
        const uint64_t expected_block3_tdnnd11_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd11_values = static_cast<uint64_t>(expected_tdnn_frames) * 864u;
        const uint64_t expected_block3_tdnnd12_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd12_values = static_cast<uint64_t>(expected_tdnn_frames) * 896u;
        const uint64_t expected_block3_tdnnd13_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd13_values = static_cast<uint64_t>(expected_tdnn_frames) * 928u;
        const uint64_t expected_block3_tdnnd14_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd14_values = static_cast<uint64_t>(expected_tdnn_frames) * 960u;
        const uint64_t expected_block3_tdnnd15_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd15_values = static_cast<uint64_t>(expected_tdnn_frames) * 992u;
        const uint64_t expected_block3_tdnnd16_values = static_cast<uint64_t>(expected_tdnn_frames) * 32u;
        const uint64_t expected_block3_after_tdnnd16_values = static_cast<uint64_t>(expected_tdnn_frames) * 1024u;
        const uint64_t expected_transit3_values = static_cast<uint64_t>(expected_tdnn_frames) * 512u;
        const uint64_t expected_out_nonlinear_values = static_cast<uint64_t>(expected_tdnn_frames) * 512u;
        const uint64_t expected_stats_values = 1024u;
        const uint64_t expected_dense_values = 192u;
        if (fbank.size() != static_cast<size_t>(manifest.fbank_frames) * 80u) {
            issues.push_back("fbank_shape_mismatch");
        }
        if (expected.size() != expected_values) {
            issues.push_back("campplus_head_golden_shape_mismatch");
        }
        if (expected_layer1.size() != expected_layer1_values) {
            issues.push_back("campplus_head_layer1_golden_shape_mismatch");
        }
        if (expected_layer2.size() != expected_layer2_values) {
            issues.push_back("campplus_head_layer2_golden_shape_mismatch");
        }
        if (expected_conv2.size() != expected_conv2_values) {
            issues.push_back("campplus_head_conv2_golden_shape_mismatch");
        }
        if (expected_output.size() != expected_output_values) {
            issues.push_back("campplus_head_output_golden_shape_mismatch");
        }
        if (expected_tdnn.size() != expected_tdnn_values) {
            issues.push_back("campplus_xvector_tdnn_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd1.size() != expected_block1_tdnnd1_values) {
            issues.push_back("campplus_xvector_block1_tdnnd1_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd1.size() != expected_block1_after_tdnnd1_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd1_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd2.size() != expected_block1_tdnnd2_values) {
            issues.push_back("campplus_xvector_block1_tdnnd2_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd2.size() != expected_block1_after_tdnnd2_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd2_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd3.size() != expected_block1_tdnnd3_values) {
            issues.push_back("campplus_xvector_block1_tdnnd3_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd3.size() != expected_block1_after_tdnnd3_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd3_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd4.size() != expected_block1_tdnnd4_values) {
            issues.push_back("campplus_xvector_block1_tdnnd4_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd4.size() != expected_block1_after_tdnnd4_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd4_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd5.size() != expected_block1_tdnnd5_values) {
            issues.push_back("campplus_xvector_block1_tdnnd5_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd5.size() != expected_block1_after_tdnnd5_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd5_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd6.size() != expected_block1_tdnnd6_values) {
            issues.push_back("campplus_xvector_block1_tdnnd6_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd6.size() != expected_block1_after_tdnnd6_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd6_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd7.size() != expected_block1_tdnnd7_values) {
            issues.push_back("campplus_xvector_block1_tdnnd7_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd7.size() != expected_block1_after_tdnnd7_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd7_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd8.size() != expected_block1_tdnnd8_values) {
            issues.push_back("campplus_xvector_block1_tdnnd8_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd8.size() != expected_block1_after_tdnnd8_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd8_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd9.size() != expected_block1_tdnnd9_values) {
            issues.push_back("campplus_xvector_block1_tdnnd9_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd9.size() != expected_block1_after_tdnnd9_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd9_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd10.size() != expected_block1_tdnnd10_values) {
            issues.push_back("campplus_xvector_block1_tdnnd10_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd10.size() != expected_block1_after_tdnnd10_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd10_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd11.size() != expected_block1_tdnnd11_values) {
            issues.push_back("campplus_xvector_block1_tdnnd11_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd11.size() != expected_block1_after_tdnnd11_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd11_golden_shape_mismatch");
        }
        if (expected_block1_tdnnd12.size() != expected_block1_tdnnd12_values) {
            issues.push_back("campplus_xvector_block1_tdnnd12_golden_shape_mismatch");
        }
        if (expected_block1_after_tdnnd12.size() != expected_block1_after_tdnnd12_values) {
            issues.push_back("campplus_xvector_block1_after_tdnnd12_golden_shape_mismatch");
        }
        if (expected_transit1.size() != expected_transit1_values) {
            issues.push_back("campplus_xvector_transit1_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd1.size() != expected_block2_tdnnd1_values) {
            issues.push_back("campplus_xvector_block2_tdnnd1_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd1.size() != expected_block2_after_tdnnd1_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd1_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd2.size() != expected_block2_tdnnd2_values) {
            issues.push_back("campplus_xvector_block2_tdnnd2_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd2.size() != expected_block2_after_tdnnd2_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd2_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd3.size() != expected_block2_tdnnd3_values) {
            issues.push_back("campplus_xvector_block2_tdnnd3_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd3.size() != expected_block2_after_tdnnd3_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd3_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd4.size() != expected_block2_tdnnd4_values) {
            issues.push_back("campplus_xvector_block2_tdnnd4_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd4.size() != expected_block2_after_tdnnd4_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd4_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd5.size() != expected_block2_tdnnd5_values) {
            issues.push_back("campplus_xvector_block2_tdnnd5_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd5.size() != expected_block2_after_tdnnd5_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd5_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd6.size() != expected_block2_tdnnd6_values) {
            issues.push_back("campplus_xvector_block2_tdnnd6_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd6.size() != expected_block2_after_tdnnd6_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd6_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd7.size() != expected_block2_tdnnd7_values) {
            issues.push_back("campplus_xvector_block2_tdnnd7_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd7.size() != expected_block2_after_tdnnd7_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd7_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd8.size() != expected_block2_tdnnd8_values) {
            issues.push_back("campplus_xvector_block2_tdnnd8_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd8.size() != expected_block2_after_tdnnd8_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd8_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd9.size() != expected_block2_tdnnd9_values) {
            issues.push_back("campplus_xvector_block2_tdnnd9_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd9.size() != expected_block2_after_tdnnd9_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd9_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd10.size() != expected_block2_tdnnd10_values) {
            issues.push_back("campplus_xvector_block2_tdnnd10_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd10.size() != expected_block2_after_tdnnd10_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd10_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd11.size() != expected_block2_tdnnd11_values) {
            issues.push_back("campplus_xvector_block2_tdnnd11_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd11.size() != expected_block2_after_tdnnd11_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd11_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd12.size() != expected_block2_tdnnd12_values) {
            issues.push_back("campplus_xvector_block2_tdnnd12_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd12.size() != expected_block2_after_tdnnd12_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd12_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd13.size() != expected_block2_tdnnd13_values) {
            issues.push_back("campplus_xvector_block2_tdnnd13_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd13.size() != expected_block2_after_tdnnd13_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd13_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd14.size() != expected_block2_tdnnd14_values) {
            issues.push_back("campplus_xvector_block2_tdnnd14_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd14.size() != expected_block2_after_tdnnd14_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd14_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd15.size() != expected_block2_tdnnd15_values) {
            issues.push_back("campplus_xvector_block2_tdnnd15_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd15.size() != expected_block2_after_tdnnd15_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd15_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd16.size() != expected_block2_tdnnd16_values) {
            issues.push_back("campplus_xvector_block2_tdnnd16_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd16.size() != expected_block2_after_tdnnd16_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd16_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd17.size() != expected_block2_tdnnd17_values) {
            issues.push_back("campplus_xvector_block2_tdnnd17_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd17.size() != expected_block2_after_tdnnd17_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd17_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd18.size() != expected_block2_tdnnd18_values) {
            issues.push_back("campplus_xvector_block2_tdnnd18_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd18.size() != expected_block2_after_tdnnd18_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd18_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd19.size() != expected_block2_tdnnd19_values) {
            issues.push_back("campplus_xvector_block2_tdnnd19_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd19.size() != expected_block2_after_tdnnd19_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd19_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd20.size() != expected_block2_tdnnd20_values) {
            issues.push_back("campplus_xvector_block2_tdnnd20_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd20.size() != expected_block2_after_tdnnd20_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd20_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd21.size() != expected_block2_tdnnd21_values) {
            issues.push_back("campplus_xvector_block2_tdnnd21_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd21.size() != expected_block2_after_tdnnd21_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd21_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd22.size() != expected_block2_tdnnd22_values) {
            issues.push_back("campplus_xvector_block2_tdnnd22_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd22.size() != expected_block2_after_tdnnd22_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd22_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd23.size() != expected_block2_tdnnd23_values) {
            issues.push_back("campplus_xvector_block2_tdnnd23_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd23.size() != expected_block2_after_tdnnd23_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd23_golden_shape_mismatch");
        }
        if (expected_block2_tdnnd24.size() != expected_block2_tdnnd24_values) {
            issues.push_back("campplus_xvector_block2_tdnnd24_golden_shape_mismatch");
        }
        if (expected_block2_after_tdnnd24.size() != expected_block2_after_tdnnd24_values) {
            issues.push_back("campplus_xvector_block2_after_tdnnd24_golden_shape_mismatch");
        }
        if (expected_transit2.size() != expected_transit2_values) {
            issues.push_back("campplus_xvector_transit2_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd1.size() != expected_block3_tdnnd1_values) {
            issues.push_back("campplus_xvector_block3_tdnnd1_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd1.size() != expected_block3_after_tdnnd1_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd1_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd2.size() != expected_block3_tdnnd2_values) {
            issues.push_back("campplus_xvector_block3_tdnnd2_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd2.size() != expected_block3_after_tdnnd2_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd2_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd3.size() != expected_block3_tdnnd3_values) {
            issues.push_back("campplus_xvector_block3_tdnnd3_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd3.size() != expected_block3_after_tdnnd3_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd3_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd4.size() != expected_block3_tdnnd4_values) {
            issues.push_back("campplus_xvector_block3_tdnnd4_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd4.size() != expected_block3_after_tdnnd4_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd4_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd5.size() != expected_block3_tdnnd5_values) {
            issues.push_back("campplus_xvector_block3_tdnnd5_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd5.size() != expected_block3_after_tdnnd5_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd5_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd6.size() != expected_block3_tdnnd6_values) {
            issues.push_back("campplus_xvector_block3_tdnnd6_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd6.size() != expected_block3_after_tdnnd6_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd6_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd7.size() != expected_block3_tdnnd7_values) {
            issues.push_back("campplus_xvector_block3_tdnnd7_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd7.size() != expected_block3_after_tdnnd7_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd7_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd8.size() != expected_block3_tdnnd8_values) {
            issues.push_back("campplus_xvector_block3_tdnnd8_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd8.size() != expected_block3_after_tdnnd8_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd8_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd9.size() != expected_block3_tdnnd9_values) {
            issues.push_back("campplus_xvector_block3_tdnnd9_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd9.size() != expected_block3_after_tdnnd9_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd9_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd10.size() != expected_block3_tdnnd10_values) {
            issues.push_back("campplus_xvector_block3_tdnnd10_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd10.size() != expected_block3_after_tdnnd10_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd10_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd11.size() != expected_block3_tdnnd11_values) {
            issues.push_back("campplus_xvector_block3_tdnnd11_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd11.size() != expected_block3_after_tdnnd11_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd11_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd12.size() != expected_block3_tdnnd12_values) {
            issues.push_back("campplus_xvector_block3_tdnnd12_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd12.size() != expected_block3_after_tdnnd12_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd12_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd13.size() != expected_block3_tdnnd13_values) {
            issues.push_back("campplus_xvector_block3_tdnnd13_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd13.size() != expected_block3_after_tdnnd13_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd13_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd14.size() != expected_block3_tdnnd14_values) {
            issues.push_back("campplus_xvector_block3_tdnnd14_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd14.size() != expected_block3_after_tdnnd14_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd14_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd15.size() != expected_block3_tdnnd15_values) {
            issues.push_back("campplus_xvector_block3_tdnnd15_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd15.size() != expected_block3_after_tdnnd15_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd15_golden_shape_mismatch");
        }
        if (expected_block3_tdnnd16.size() != expected_block3_tdnnd16_values) {
            issues.push_back("campplus_xvector_block3_tdnnd16_golden_shape_mismatch");
        }
        if (expected_block3_after_tdnnd16.size() != expected_block3_after_tdnnd16_values) {
            issues.push_back("campplus_xvector_block3_after_tdnnd16_golden_shape_mismatch");
        }
        if (expected_transit3.size() != expected_transit3_values) {
            issues.push_back("campplus_xvector_transit3_golden_shape_mismatch");
        }
        if (expected_out_nonlinear.size() != expected_out_nonlinear_values) {
            issues.push_back("campplus_xvector_out_nonlinear_golden_shape_mismatch");
        }
        if (expected_stats.size() != expected_stats_values) {
            issues.push_back("campplus_xvector_stats_golden_shape_mismatch");
        }
        if (expected_dense.size() != expected_dense_values) {
            issues.push_back("campplus_xvector_dense_golden_shape_mismatch");
        }

        float err = std::numeric_limits<float>::infinity();
        float layer1_err = std::numeric_limits<float>::infinity();
        float layer2_err = std::numeric_limits<float>::infinity();
        float conv2_err = std::numeric_limits<float>::infinity();
        float output_err = std::numeric_limits<float>::infinity();
        float tdnn_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd1_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd1_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd2_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd2_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd3_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd3_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd4_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd4_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd5_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd5_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd6_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd6_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd7_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd7_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd8_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd8_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd9_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd9_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd10_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd10_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd11_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd11_err = std::numeric_limits<float>::infinity();
        float block1_tdnnd12_err = std::numeric_limits<float>::infinity();
        float block1_after_tdnnd12_err = std::numeric_limits<float>::infinity();
        float transit1_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd1_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd1_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd2_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd2_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd3_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd3_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd4_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd4_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd5_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd5_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd6_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd6_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd7_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd7_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd8_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd8_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd9_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd9_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd10_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd10_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd11_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd11_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd12_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd12_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd13_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd13_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd14_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd14_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd15_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd15_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd16_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd16_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd17_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd17_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd18_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd18_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd19_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd19_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd20_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd20_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd21_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd21_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd22_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd22_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd23_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd23_err = std::numeric_limits<float>::infinity();
        float block2_tdnnd24_err = std::numeric_limits<float>::infinity();
        float block2_after_tdnnd24_err = std::numeric_limits<float>::infinity();
        float transit2_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd1_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd1_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd2_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd2_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd3_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd3_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd4_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd4_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd5_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd5_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd6_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd6_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd7_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd7_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd8_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd8_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd9_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd9_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd10_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd10_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd11_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd11_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd12_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd12_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd13_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd13_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd14_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd14_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd15_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd15_err = std::numeric_limits<float>::infinity();
        float block3_tdnnd16_err = std::numeric_limits<float>::infinity();
        float block3_after_tdnnd16_err = std::numeric_limits<float>::infinity();
        float transit3_err = std::numeric_limits<float>::infinity();
        float out_nonlinear_err = std::numeric_limits<float>::infinity();
        float stats_err = std::numeric_limits<float>::infinity();
        float dense_err = std::numeric_limits<float>::infinity();
        if (issues.empty()) {
            const auto actual = cpu_campplus_head_conv1_bn_relu(
                fbank,
                tensor_as_f32(model, "campplus.head.conv1.weight"),
                tensor_as_f32(model, "campplus.head.bn1.weight"),
                tensor_as_f32(model, "campplus.head.bn1.bias"),
                tensor_as_f32(model, "campplus.head.bn1.running_mean"),
                tensor_as_f32(model, "campplus.head.bn1.running_var"),
                manifest.fbank_frames);
            err = max_abs_error(actual, expected);
            NchwShape layer0_shape{32u, 80u, static_cast<uint32_t>(manifest.fbank_frames)};
            NchwShape block0_shape;
            auto layer1 = cpu_campplus_head_layer1_block0(
                actual,
                layer0_shape,
                tensor_as_f32(model, "campplus.head.layer1.0.conv1.weight"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn1.weight"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn1.bias"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn1.running_mean"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn1.running_var"),
                tensor_as_f32(model, "campplus.head.layer1.0.conv2.weight"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn2.weight"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn2.bias"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn2.running_mean"),
                tensor_as_f32(model, "campplus.head.layer1.0.bn2.running_var"),
                tensor_as_f32(model, "campplus.head.layer1.0.shortcut.0.weight"),
                tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.weight"),
                tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.bias"),
                tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.running_mean"),
                tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.running_var"),
                block0_shape);
            NchwShape block1_shape;
            layer1 = cpu_campplus_head_layer1_block1(
                layer1,
                block0_shape,
                tensor_as_f32(model, "campplus.head.layer1.1.conv1.weight"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn1.weight"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn1.bias"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn1.running_mean"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn1.running_var"),
                tensor_as_f32(model, "campplus.head.layer1.1.conv2.weight"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn2.weight"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn2.bias"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn2.running_mean"),
                tensor_as_f32(model, "campplus.head.layer1.1.bn2.running_var"),
                block1_shape);
            layer1_err = max_abs_error(layer1, expected_layer1);
            NchwShape layer2_block0_shape;
            auto layer2 = cpu_campplus_head_layer1_block0(
                layer1,
                block1_shape,
                tensor_as_f32(model, "campplus.head.layer2.0.conv1.weight"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn1.weight"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn1.bias"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn1.running_mean"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn1.running_var"),
                tensor_as_f32(model, "campplus.head.layer2.0.conv2.weight"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn2.weight"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn2.bias"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn2.running_mean"),
                tensor_as_f32(model, "campplus.head.layer2.0.bn2.running_var"),
                tensor_as_f32(model, "campplus.head.layer2.0.shortcut.0.weight"),
                tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.weight"),
                tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.bias"),
                tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.running_mean"),
                tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.running_var"),
                layer2_block0_shape);
            NchwShape layer2_block1_shape;
            layer2 = cpu_campplus_head_layer1_block1(
                layer2,
                layer2_block0_shape,
                tensor_as_f32(model, "campplus.head.layer2.1.conv1.weight"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn1.weight"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn1.bias"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn1.running_mean"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn1.running_var"),
                tensor_as_f32(model, "campplus.head.layer2.1.conv2.weight"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn2.weight"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn2.bias"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn2.running_mean"),
                tensor_as_f32(model, "campplus.head.layer2.1.bn2.running_var"),
                layer2_block1_shape);
            layer2_err = max_abs_error(layer2, expected_layer2);
            NchwShape conv2_shape;
            auto conv2 = cpu_conv2d_nchw_no_bias(
                layer2,
                layer2_block1_shape,
                tensor_as_f32(model, "campplus.head.conv2.weight"),
                32,
                3,
                3,
                2,
                1,
                1,
                1,
                conv2_shape);
            cpu_batchnorm2d_inplace(conv2,
                                    conv2_shape,
                                    tensor_as_f32(model, "campplus.head.bn2.weight"),
                                    tensor_as_f32(model, "campplus.head.bn2.bias"),
                                    tensor_as_f32(model, "campplus.head.bn2.running_mean"),
                                    tensor_as_f32(model, "campplus.head.bn2.running_var"),
                                    true);
            conv2_err = max_abs_error(conv2, expected_conv2);
            output_err = max_abs_error(conv2, expected_output);
            uint32_t tdnn_frames = 0;
            auto tdnn = cpu_conv1d_ncw_no_bias(conv2,
                                               tensor_as_f32(model, "campplus.xvector.tdnn.linear.weight"),
                                               320u,
                                               static_cast<uint32_t>(manifest.fbank_frames),
                                               128u,
                                               5u,
                                               2u,
                                               2u,
                                               1u,
                                               tdnn_frames);
            cpu_batchnorm1d_inplace(tdnn,
                                    128u,
                                    tdnn_frames,
                                    tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.weight"),
                                    tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.bias"),
                                    tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.running_mean"),
                                    tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.running_var"),
                                    true);
            tdnn_err = max_abs_error(tdnn, expected_tdnn);
            auto block1_tdnnd1 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd1",
                tdnn,
                128u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd1.cam_layer.linear2.bias"));
            block1_tdnnd1_err = max_abs_error(block1_tdnnd1, expected_block1_tdnnd1);
            const auto block1_after_tdnnd1 = concat_channels_ncw(tdnn, 128u, block1_tdnnd1, 32u, tdnn_frames);
            block1_after_tdnnd1_err = max_abs_error(block1_after_tdnnd1, expected_block1_after_tdnnd1);
            auto block1_tdnnd2 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd2",
                block1_after_tdnnd1,
                160u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd2.cam_layer.linear2.bias"));
            block1_tdnnd2_err = max_abs_error(block1_tdnnd2, expected_block1_tdnnd2);
            const auto block1_after_tdnnd2 = concat_channels_ncw(block1_after_tdnnd1, 160u, block1_tdnnd2, 32u, tdnn_frames);
            block1_after_tdnnd2_err = max_abs_error(block1_after_tdnnd2, expected_block1_after_tdnnd2);
            auto block1_tdnnd3 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd3",
                block1_after_tdnnd2,
                192u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd3.cam_layer.linear2.bias"));
            block1_tdnnd3_err = max_abs_error(block1_tdnnd3, expected_block1_tdnnd3);
            const auto block1_after_tdnnd3 = concat_channels_ncw(block1_after_tdnnd2, 192u, block1_tdnnd3, 32u, tdnn_frames);
            block1_after_tdnnd3_err = max_abs_error(block1_after_tdnnd3, expected_block1_after_tdnnd3);
            auto block1_tdnnd4 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd4",
                block1_after_tdnnd3,
                224u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd4.cam_layer.linear2.bias"));
            block1_tdnnd4_err = max_abs_error(block1_tdnnd4, expected_block1_tdnnd4);
            const auto block1_after_tdnnd4 = concat_channels_ncw(block1_after_tdnnd3, 224u, block1_tdnnd4, 32u, tdnn_frames);
            block1_after_tdnnd4_err = max_abs_error(block1_after_tdnnd4, expected_block1_after_tdnnd4);
            auto block1_tdnnd5 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd5",
                block1_after_tdnnd4,
                256u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd5.cam_layer.linear2.bias"));
            block1_tdnnd5_err = max_abs_error(block1_tdnnd5, expected_block1_tdnnd5);
            const auto block1_after_tdnnd5 = concat_channels_ncw(block1_after_tdnnd4, 256u, block1_tdnnd5, 32u, tdnn_frames);
            block1_after_tdnnd5_err = max_abs_error(block1_after_tdnnd5, expected_block1_after_tdnnd5);
            auto block1_tdnnd6 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd6",
                block1_after_tdnnd5,
                288u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd6.cam_layer.linear2.bias"));
            block1_tdnnd6_err = max_abs_error(block1_tdnnd6, expected_block1_tdnnd6);
            const auto block1_after_tdnnd6 = concat_channels_ncw(block1_after_tdnnd5, 288u, block1_tdnnd6, 32u, tdnn_frames);
            block1_after_tdnnd6_err = max_abs_error(block1_after_tdnnd6, expected_block1_after_tdnnd6);
            auto block1_tdnnd7 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd7",
                block1_after_tdnnd6,
                320u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd7.cam_layer.linear2.bias"));
            block1_tdnnd7_err = max_abs_error(block1_tdnnd7, expected_block1_tdnnd7);
            const auto block1_after_tdnnd7 = concat_channels_ncw(block1_after_tdnnd6, 320u, block1_tdnnd7, 32u, tdnn_frames);
            block1_after_tdnnd7_err = max_abs_error(block1_after_tdnnd7, expected_block1_after_tdnnd7);
            auto block1_tdnnd8 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd8",
                block1_after_tdnnd7,
                352u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd8.cam_layer.linear2.bias"));
            block1_tdnnd8_err = max_abs_error(block1_tdnnd8, expected_block1_tdnnd8);
            const auto block1_after_tdnnd8 = concat_channels_ncw(block1_after_tdnnd7, 352u, block1_tdnnd8, 32u, tdnn_frames);
            block1_after_tdnnd8_err = max_abs_error(block1_after_tdnnd8, expected_block1_after_tdnnd8);
            auto block1_tdnnd9 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd9",
                block1_after_tdnnd8,
                384u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd9.cam_layer.linear2.bias"));
            block1_tdnnd9_err = max_abs_error(block1_tdnnd9, expected_block1_tdnnd9);
            const auto block1_after_tdnnd9 = concat_channels_ncw(block1_after_tdnnd8, 384u, block1_tdnnd9, 32u, tdnn_frames);
            block1_after_tdnnd9_err = max_abs_error(block1_after_tdnnd9, expected_block1_after_tdnnd9);
            auto block1_tdnnd10 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd10",
                block1_after_tdnnd9,
                416u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd10.cam_layer.linear2.bias"));
            block1_tdnnd10_err = max_abs_error(block1_tdnnd10, expected_block1_tdnnd10);
            const auto block1_after_tdnnd10 = concat_channels_ncw(block1_after_tdnnd9, 416u, block1_tdnnd10, 32u, tdnn_frames);
            block1_after_tdnnd10_err = max_abs_error(block1_after_tdnnd10, expected_block1_after_tdnnd10);
            auto block1_tdnnd11 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd11",
                block1_after_tdnnd10,
                448u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd11.cam_layer.linear2.bias"));
            block1_tdnnd11_err = max_abs_error(block1_tdnnd11, expected_block1_tdnnd11);
            const auto block1_after_tdnnd11 = concat_channels_ncw(block1_after_tdnnd10, 448u, block1_tdnnd11, 32u, tdnn_frames);
            block1_after_tdnnd11_err = max_abs_error(block1_after_tdnnd11, expected_block1_after_tdnnd11);
            auto block1_tdnnd12 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block1_tdnnd12",
                block1_after_tdnnd11,
                480u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block1.tdnnd12.cam_layer.linear2.bias"));
            block1_tdnnd12_err = max_abs_error(block1_tdnnd12, expected_block1_tdnnd12);
            const auto block1_after_tdnnd12 = concat_channels_ncw(block1_after_tdnnd11, 480u, block1_tdnnd12, 32u, tdnn_frames);
            block1_after_tdnnd12_err = max_abs_error(block1_after_tdnnd12, expected_block1_after_tdnnd12);
            auto transit1_input = block1_after_tdnnd12;
            cpu_batchnorm1d_inplace(transit1_input,
                                    512u,
                                    tdnn_frames,
                                    tensor_as_f32(model, "campplus.xvector.transit1.nonlinear.batchnorm.weight"),
                                    tensor_as_f32(model, "campplus.xvector.transit1.nonlinear.batchnorm.bias"),
                                    tensor_as_f32(model, "campplus.xvector.transit1.nonlinear.batchnorm.running_mean"),
                                    tensor_as_f32(model, "campplus.xvector.transit1.nonlinear.batchnorm.running_var"),
                                    true);
            uint32_t transit1_width = 0;
            const auto transit1 = cpu_conv1d_ncw_no_bias(transit1_input,
                                                         tensor_as_f32(model, "campplus.xvector.transit1.linear.weight"),
                                                         512u,
                                                         tdnn_frames,
                                                         256u,
                                                         1u,
                                                         1u,
                                                         0u,
                                                         1u,
                                                         transit1_width);
            if (transit1_width != tdnn_frames) {
                issues.push_back("campplus_xvector_transit1_width_mismatch");
            }
            transit1_err = max_abs_error(transit1, expected_transit1);
            auto block2_tdnnd1 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd1",
                transit1,
                256u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd1.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd1_err = max_abs_error(block2_tdnnd1, expected_block2_tdnnd1);
            const auto block2_after_tdnnd1 = concat_channels_ncw(transit1, 256u, block2_tdnnd1, 32u, tdnn_frames);
            block2_after_tdnnd1_err = max_abs_error(block2_after_tdnnd1, expected_block2_after_tdnnd1);
            auto block2_tdnnd2 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd2",
                block2_after_tdnnd1,
                288u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd2.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd2_err = max_abs_error(block2_tdnnd2, expected_block2_tdnnd2);
            const auto block2_after_tdnnd2 = concat_channels_ncw(block2_after_tdnnd1, 288u, block2_tdnnd2, 32u, tdnn_frames);
            block2_after_tdnnd2_err = max_abs_error(block2_after_tdnnd2, expected_block2_after_tdnnd2);
            auto block2_tdnnd3 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd3",
                block2_after_tdnnd2,
                320u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd3.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd3_err = max_abs_error(block2_tdnnd3, expected_block2_tdnnd3);
            const auto block2_after_tdnnd3 = concat_channels_ncw(block2_after_tdnnd2, 320u, block2_tdnnd3, 32u, tdnn_frames);
            block2_after_tdnnd3_err = max_abs_error(block2_after_tdnnd3, expected_block2_after_tdnnd3);
            auto block2_tdnnd4 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd4",
                block2_after_tdnnd3,
                352u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd4.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd4_err = max_abs_error(block2_tdnnd4, expected_block2_tdnnd4);
            const auto block2_after_tdnnd4 = concat_channels_ncw(block2_after_tdnnd3, 352u, block2_tdnnd4, 32u, tdnn_frames);
            block2_after_tdnnd4_err = max_abs_error(block2_after_tdnnd4, expected_block2_after_tdnnd4);
            auto block2_tdnnd5 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd5",
                block2_after_tdnnd4,
                384u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd5.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd5_err = max_abs_error(block2_tdnnd5, expected_block2_tdnnd5);
            const auto block2_after_tdnnd5 = concat_channels_ncw(block2_after_tdnnd4, 384u, block2_tdnnd5, 32u, tdnn_frames);
            block2_after_tdnnd5_err = max_abs_error(block2_after_tdnnd5, expected_block2_after_tdnnd5);
            auto block2_tdnnd6 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd6",
                block2_after_tdnnd5,
                416u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd6.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd6_err = max_abs_error(block2_tdnnd6, expected_block2_tdnnd6);
            const auto block2_after_tdnnd6 = concat_channels_ncw(block2_after_tdnnd5, 416u, block2_tdnnd6, 32u, tdnn_frames);
            block2_after_tdnnd6_err = max_abs_error(block2_after_tdnnd6, expected_block2_after_tdnnd6);
            auto block2_tdnnd7 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd7",
                block2_after_tdnnd6,
                448u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd7.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd7_err = max_abs_error(block2_tdnnd7, expected_block2_tdnnd7);
            const auto block2_after_tdnnd7 = concat_channels_ncw(block2_after_tdnnd6, 448u, block2_tdnnd7, 32u, tdnn_frames);
            block2_after_tdnnd7_err = max_abs_error(block2_after_tdnnd7, expected_block2_after_tdnnd7);
            auto block2_tdnnd8 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd8",
                block2_after_tdnnd7,
                480u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd8.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd8_err = max_abs_error(block2_tdnnd8, expected_block2_tdnnd8);
            const auto block2_after_tdnnd8 = concat_channels_ncw(block2_after_tdnnd7, 480u, block2_tdnnd8, 32u, tdnn_frames);
            block2_after_tdnnd8_err = max_abs_error(block2_after_tdnnd8, expected_block2_after_tdnnd8);
            auto block2_tdnnd9 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd9",
                block2_after_tdnnd8,
                512u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd9.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd9_err = max_abs_error(block2_tdnnd9, expected_block2_tdnnd9);
            const auto block2_after_tdnnd9 = concat_channels_ncw(block2_after_tdnnd8, 512u, block2_tdnnd9, 32u, tdnn_frames);
            block2_after_tdnnd9_err = max_abs_error(block2_after_tdnnd9, expected_block2_after_tdnnd9);
            auto block2_tdnnd10 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd10",
                block2_after_tdnnd9,
                544u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd10.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd10_err = max_abs_error(block2_tdnnd10, expected_block2_tdnnd10);
            const auto block2_after_tdnnd10 =
                concat_channels_ncw(block2_after_tdnnd9, 544u, block2_tdnnd10, 32u, tdnn_frames);
            block2_after_tdnnd10_err = max_abs_error(block2_after_tdnnd10, expected_block2_after_tdnnd10);
            auto block2_tdnnd11 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd11",
                block2_after_tdnnd10,
                576u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd11.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd11_err = max_abs_error(block2_tdnnd11, expected_block2_tdnnd11);
            const auto block2_after_tdnnd11 =
                concat_channels_ncw(block2_after_tdnnd10, 576u, block2_tdnnd11, 32u, tdnn_frames);
            block2_after_tdnnd11_err = max_abs_error(block2_after_tdnnd11, expected_block2_after_tdnnd11);
            auto block2_tdnnd12 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd12",
                block2_after_tdnnd11,
                608u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd12.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd12_err = max_abs_error(block2_tdnnd12, expected_block2_tdnnd12);
            const auto block2_after_tdnnd12 =
                concat_channels_ncw(block2_after_tdnnd11, 608u, block2_tdnnd12, 32u, tdnn_frames);
            block2_after_tdnnd12_err = max_abs_error(block2_after_tdnnd12, expected_block2_after_tdnnd12);
            auto block2_tdnnd13 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd13",
                block2_after_tdnnd12,
                640u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd13.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd13_err = max_abs_error(block2_tdnnd13, expected_block2_tdnnd13);
            const auto block2_after_tdnnd13 =
                concat_channels_ncw(block2_after_tdnnd12, 640u, block2_tdnnd13, 32u, tdnn_frames);
            block2_after_tdnnd13_err = max_abs_error(block2_after_tdnnd13, expected_block2_after_tdnnd13);
            auto block2_tdnnd14 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd14",
                block2_after_tdnnd13,
                672u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd14.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd14_err = max_abs_error(block2_tdnnd14, expected_block2_tdnnd14);
            const auto block2_after_tdnnd14 =
                concat_channels_ncw(block2_after_tdnnd13, 672u, block2_tdnnd14, 32u, tdnn_frames);
            block2_after_tdnnd14_err = max_abs_error(block2_after_tdnnd14, expected_block2_after_tdnnd14);
            auto block2_tdnnd15 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd15",
                block2_after_tdnnd14,
                704u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd15.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd15_err = max_abs_error(block2_tdnnd15, expected_block2_tdnnd15);
            const auto block2_after_tdnnd15 =
                concat_channels_ncw(block2_after_tdnnd14, 704u, block2_tdnnd15, 32u, tdnn_frames);
            block2_after_tdnnd15_err = max_abs_error(block2_after_tdnnd15, expected_block2_after_tdnnd15);
            auto block2_tdnnd16 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd16",
                block2_after_tdnnd15,
                736u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd16.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd16_err = max_abs_error(block2_tdnnd16, expected_block2_tdnnd16);
            const auto block2_after_tdnnd16 =
                concat_channels_ncw(block2_after_tdnnd15, 736u, block2_tdnnd16, 32u, tdnn_frames);
            block2_after_tdnnd16_err = max_abs_error(block2_after_tdnnd16, expected_block2_after_tdnnd16);
            auto block2_tdnnd17 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd17",
                block2_after_tdnnd16,
                768u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd17.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd17_err = max_abs_error(block2_tdnnd17, expected_block2_tdnnd17);
            const auto block2_after_tdnnd17 =
                concat_channels_ncw(block2_after_tdnnd16, 768u, block2_tdnnd17, 32u, tdnn_frames);
            block2_after_tdnnd17_err = max_abs_error(block2_after_tdnnd17, expected_block2_after_tdnnd17);
            auto block2_tdnnd18 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd18",
                block2_after_tdnnd17,
                800u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd18.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd18_err = max_abs_error(block2_tdnnd18, expected_block2_tdnnd18);
            const auto block2_after_tdnnd18 =
                concat_channels_ncw(block2_after_tdnnd17, 800u, block2_tdnnd18, 32u, tdnn_frames);
            block2_after_tdnnd18_err = max_abs_error(block2_after_tdnnd18, expected_block2_after_tdnnd18);
            auto block2_tdnnd19 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd19",
                block2_after_tdnnd18,
                832u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd19.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd19_err = max_abs_error(block2_tdnnd19, expected_block2_tdnnd19);
            const auto block2_after_tdnnd19 =
                concat_channels_ncw(block2_after_tdnnd18, 832u, block2_tdnnd19, 32u, tdnn_frames);
            block2_after_tdnnd19_err = max_abs_error(block2_after_tdnnd19, expected_block2_after_tdnnd19);
            auto block2_tdnnd20 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd20",
                block2_after_tdnnd19,
                864u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd20.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd20_err = max_abs_error(block2_tdnnd20, expected_block2_tdnnd20);
            const auto block2_after_tdnnd20 =
                concat_channels_ncw(block2_after_tdnnd19, 864u, block2_tdnnd20, 32u, tdnn_frames);
            block2_after_tdnnd20_err = max_abs_error(block2_after_tdnnd20, expected_block2_after_tdnnd20);
            auto block2_tdnnd21 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd21",
                block2_after_tdnnd20,
                896u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd21.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd21_err = max_abs_error(block2_tdnnd21, expected_block2_tdnnd21);
            const auto block2_after_tdnnd21 =
                concat_channels_ncw(block2_after_tdnnd20, 896u, block2_tdnnd21, 32u, tdnn_frames);
            block2_after_tdnnd21_err = max_abs_error(block2_after_tdnnd21, expected_block2_after_tdnnd21);
            auto block2_tdnnd22 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd22",
                block2_after_tdnnd21,
                928u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd22.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd22_err = max_abs_error(block2_tdnnd22, expected_block2_tdnnd22);
            const auto block2_after_tdnnd22 =
                concat_channels_ncw(block2_after_tdnnd21, 928u, block2_tdnnd22, 32u, tdnn_frames);
            block2_after_tdnnd22_err = max_abs_error(block2_after_tdnnd22, expected_block2_after_tdnnd22);
            auto block2_tdnnd23 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd23",
                block2_after_tdnnd22,
                960u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd23.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd23_err = max_abs_error(block2_tdnnd23, expected_block2_tdnnd23);
            const auto block2_after_tdnnd23 =
                concat_channels_ncw(block2_after_tdnnd22, 960u, block2_tdnnd23, 32u, tdnn_frames);
            block2_after_tdnnd23_err = max_abs_error(block2_after_tdnnd23, expected_block2_after_tdnnd23);
            auto block2_tdnnd24 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block2_tdnnd24",
                block2_after_tdnnd23,
                992u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block2.tdnnd24.cam_layer.linear2.bias"),
                2u,
                2u);
            block2_tdnnd24_err = max_abs_error(block2_tdnnd24, expected_block2_tdnnd24);
            const auto block2_after_tdnnd24 =
                concat_channels_ncw(block2_after_tdnnd23, 992u, block2_tdnnd24, 32u, tdnn_frames);
            block2_after_tdnnd24_err = max_abs_error(block2_after_tdnnd24, expected_block2_after_tdnnd24);
            auto transit2_input = block2_after_tdnnd24;
            cpu_batchnorm1d_inplace(transit2_input,
                                    1024u,
                                    tdnn_frames,
                                    tensor_as_f32(model, "campplus.xvector.transit2.nonlinear.batchnorm.weight"),
                                    tensor_as_f32(model, "campplus.xvector.transit2.nonlinear.batchnorm.bias"),
                                    tensor_as_f32(model, "campplus.xvector.transit2.nonlinear.batchnorm.running_mean"),
                                    tensor_as_f32(model, "campplus.xvector.transit2.nonlinear.batchnorm.running_var"),
                                    true);
            uint32_t transit2_width = 0;
            const auto transit2 = cpu_conv1d_ncw_no_bias(transit2_input,
                                                         tensor_as_f32(model, "campplus.xvector.transit2.linear.weight"),
                                                         1024u,
                                                         tdnn_frames,
                                                         512u,
                                                         1u,
                                                         1u,
                                                         0u,
                                                         1u,
                                                         transit2_width);
            if (transit2_width != tdnn_frames) {
                issues.push_back("campplus_xvector_transit2_width_mismatch");
            }
            transit2_err = max_abs_error(transit2, expected_transit2);
            auto block3_tdnnd1 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd1",
                transit2,
                512u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd1.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd1_err = max_abs_error(block3_tdnnd1, expected_block3_tdnnd1);
            const auto block3_after_tdnnd1 = concat_channels_ncw(transit2, 512u, block3_tdnnd1, 32u, tdnn_frames);
            block3_after_tdnnd1_err = max_abs_error(block3_after_tdnnd1, expected_block3_after_tdnnd1);
            auto block3_tdnnd2 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd2",
                block3_after_tdnnd1,
                544u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd2.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd2_err = max_abs_error(block3_tdnnd2, expected_block3_tdnnd2);
            const auto block3_after_tdnnd2 = concat_channels_ncw(block3_after_tdnnd1, 544u, block3_tdnnd2, 32u, tdnn_frames);
            block3_after_tdnnd2_err = max_abs_error(block3_after_tdnnd2, expected_block3_after_tdnnd2);
            auto block3_tdnnd3 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd3",
                block3_after_tdnnd2,
                576u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd3.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd3_err = max_abs_error(block3_tdnnd3, expected_block3_tdnnd3);
            const auto block3_after_tdnnd3 = concat_channels_ncw(block3_after_tdnnd2, 576u, block3_tdnnd3, 32u, tdnn_frames);
            block3_after_tdnnd3_err = max_abs_error(block3_after_tdnnd3, expected_block3_after_tdnnd3);
            auto block3_tdnnd4 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd4",
                block3_after_tdnnd3,
                608u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd4.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd4_err = max_abs_error(block3_tdnnd4, expected_block3_tdnnd4);
            const auto block3_after_tdnnd4 = concat_channels_ncw(block3_after_tdnnd3, 608u, block3_tdnnd4, 32u, tdnn_frames);
            block3_after_tdnnd4_err = max_abs_error(block3_after_tdnnd4, expected_block3_after_tdnnd4);
            auto block3_tdnnd5 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd5",
                block3_after_tdnnd4,
                640u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd5.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd5_err = max_abs_error(block3_tdnnd5, expected_block3_tdnnd5);
            const auto block3_after_tdnnd5 = concat_channels_ncw(block3_after_tdnnd4, 640u, block3_tdnnd5, 32u, tdnn_frames);
            block3_after_tdnnd5_err = max_abs_error(block3_after_tdnnd5, expected_block3_after_tdnnd5);
            auto block3_tdnnd6 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd6",
                block3_after_tdnnd5,
                672u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd6.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd6_err = max_abs_error(block3_tdnnd6, expected_block3_tdnnd6);
            const auto block3_after_tdnnd6 = concat_channels_ncw(block3_after_tdnnd5, 672u, block3_tdnnd6, 32u, tdnn_frames);
            block3_after_tdnnd6_err = max_abs_error(block3_after_tdnnd6, expected_block3_after_tdnnd6);
            auto block3_tdnnd7 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd7",
                block3_after_tdnnd6,
                704u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd7.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd7_err = max_abs_error(block3_tdnnd7, expected_block3_tdnnd7);
            const auto block3_after_tdnnd7 = concat_channels_ncw(block3_after_tdnnd6, 704u, block3_tdnnd7, 32u, tdnn_frames);
            block3_after_tdnnd7_err = max_abs_error(block3_after_tdnnd7, expected_block3_after_tdnnd7);
            auto block3_tdnnd8 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd8",
                block3_after_tdnnd7,
                736u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd8.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd8_err = max_abs_error(block3_tdnnd8, expected_block3_tdnnd8);
            const auto block3_after_tdnnd8 = concat_channels_ncw(block3_after_tdnnd7, 736u, block3_tdnnd8, 32u, tdnn_frames);
            block3_after_tdnnd8_err = max_abs_error(block3_after_tdnnd8, expected_block3_after_tdnnd8);
            auto block3_tdnnd9 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd9",
                block3_after_tdnnd8,
                768u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd9.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd9_err = max_abs_error(block3_tdnnd9, expected_block3_tdnnd9);
            const auto block3_after_tdnnd9 = concat_channels_ncw(block3_after_tdnnd8, 768u, block3_tdnnd9, 32u, tdnn_frames);
            block3_after_tdnnd9_err = max_abs_error(block3_after_tdnnd9, expected_block3_after_tdnnd9);
            auto block3_tdnnd10 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd10",
                block3_after_tdnnd9,
                800u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd10.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd10_err = max_abs_error(block3_tdnnd10, expected_block3_tdnnd10);
            const auto block3_after_tdnnd10 = concat_channels_ncw(block3_after_tdnnd9, 800u, block3_tdnnd10, 32u, tdnn_frames);
            block3_after_tdnnd10_err = max_abs_error(block3_after_tdnnd10, expected_block3_after_tdnnd10);
            auto block3_tdnnd11 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd11",
                block3_after_tdnnd10,
                832u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd11.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd11_err = max_abs_error(block3_tdnnd11, expected_block3_tdnnd11);
            const auto block3_after_tdnnd11 = concat_channels_ncw(block3_after_tdnnd10, 832u, block3_tdnnd11, 32u, tdnn_frames);
            block3_after_tdnnd11_err = max_abs_error(block3_after_tdnnd11, expected_block3_after_tdnnd11);
            auto block3_tdnnd12 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd12",
                block3_after_tdnnd11,
                864u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd12.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd12_err = max_abs_error(block3_tdnnd12, expected_block3_tdnnd12);
            const auto block3_after_tdnnd12 = concat_channels_ncw(block3_after_tdnnd11, 864u, block3_tdnnd12, 32u, tdnn_frames);
            block3_after_tdnnd12_err = max_abs_error(block3_after_tdnnd12, expected_block3_after_tdnnd12);
            auto block3_tdnnd13 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd13",
                block3_after_tdnnd12,
                896u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd13.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd13_err = max_abs_error(block3_tdnnd13, expected_block3_tdnnd13);
            const auto block3_after_tdnnd13 = concat_channels_ncw(block3_after_tdnnd12, 896u, block3_tdnnd13, 32u, tdnn_frames);
            block3_after_tdnnd13_err = max_abs_error(block3_after_tdnnd13, expected_block3_after_tdnnd13);
            auto block3_tdnnd14 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd14",
                block3_after_tdnnd13,
                928u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd14.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd14_err = max_abs_error(block3_tdnnd14, expected_block3_tdnnd14);
            const auto block3_after_tdnnd14 = concat_channels_ncw(block3_after_tdnnd13, 928u, block3_tdnnd14, 32u, tdnn_frames);
            block3_after_tdnnd14_err = max_abs_error(block3_after_tdnnd14, expected_block3_after_tdnnd14);
            auto block3_tdnnd15 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd15",
                block3_after_tdnnd14,
                960u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd15.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd15_err = max_abs_error(block3_tdnnd15, expected_block3_tdnnd15);
            const auto block3_after_tdnnd15 = concat_channels_ncw(block3_after_tdnnd14, 960u, block3_tdnnd15, 32u, tdnn_frames);
            block3_after_tdnnd15_err = max_abs_error(block3_after_tdnnd15, expected_block3_after_tdnnd15);
            auto block3_tdnnd16 = cpu_campplus_dense_tdnn_layer(
                "cpu_campplus_block3_tdnnd16",
                block3_after_tdnnd15,
                992u,
                tdnn_frames,
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.running_mean"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.running_var"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.cam_layer.linear_local.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.cam_layer.linear1.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.cam_layer.linear1.bias"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.cam_layer.linear2.weight"),
                tensor_as_f32(model, "campplus.xvector.block3.tdnnd16.cam_layer.linear2.bias"),
                2u,
                2u);
            block3_tdnnd16_err = max_abs_error(block3_tdnnd16, expected_block3_tdnnd16);
            const auto block3_after_tdnnd16 = concat_channels_ncw(block3_after_tdnnd15, 992u, block3_tdnnd16, 32u, tdnn_frames);
            block3_after_tdnnd16_err = max_abs_error(block3_after_tdnnd16, expected_block3_after_tdnnd16);
            auto transit3_input = block3_after_tdnnd16;
            cpu_batchnorm1d_inplace(transit3_input,
                                    1024u,
                                    tdnn_frames,
                                    tensor_as_f32(model, "campplus.xvector.transit3.nonlinear.batchnorm.weight"),
                                    tensor_as_f32(model, "campplus.xvector.transit3.nonlinear.batchnorm.bias"),
                                    tensor_as_f32(model, "campplus.xvector.transit3.nonlinear.batchnorm.running_mean"),
                                    tensor_as_f32(model, "campplus.xvector.transit3.nonlinear.batchnorm.running_var"),
                                    true);
            uint32_t transit3_width = 0;
            const auto transit3 = cpu_conv1d_ncw_no_bias(transit3_input,
                                                        tensor_as_f32(model, "campplus.xvector.transit3.linear.weight"),
                                                        1024u,
                                                        tdnn_frames,
                                                        512u,
                                                        1u,
                                                        1u,
                                                        0u,
                                                        1u,
                                                        transit3_width);
            if (transit3_width != tdnn_frames) {
                issues.push_back("campplus_xvector_transit3_width_mismatch");
            }
            transit3_err = max_abs_error(transit3, expected_transit3);
            auto out_nonlinear = transit3;
            cpu_batchnorm1d_inplace(out_nonlinear,
                                    512u,
                                    tdnn_frames,
                                    tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.weight"),
                                    tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.bias"),
                                    tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.running_mean"),
                                    tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.running_var"),
                                    true);
            out_nonlinear_err = max_abs_error(out_nonlinear, expected_out_nonlinear);
            const auto stats = cpu_stats_pool_ncw_unbiased(out_nonlinear, 512u, tdnn_frames);
            stats_err = max_abs_error(stats, expected_stats);
            uint32_t dense_width = 0;
            auto dense = cpu_conv1d_ncw_no_bias(stats,
                                                tensor_as_f32(model, "campplus.xvector.dense.linear.weight"),
                                                1024u,
                                                1u,
                                                192u,
                                                1u,
                                                1u,
                                                0u,
                                                1u,
                                                dense_width);
            if (dense_width != 1u) {
                issues.push_back("campplus_xvector_dense_width_mismatch");
            }
            cpu_batchnorm1d_affine_false_inplace(dense,
                                                 192u,
                                                 1u,
                                                 tensor_as_f32(model, "campplus.xvector.dense.nonlinear.batchnorm.running_mean"),
                                                 tensor_as_f32(model, "campplus.xvector.dense.nonlinear.batchnorm.running_var"));
            dense_err = max_abs_error(dense, expected_dense);
            if (!(err <= 1.0e-4f)) {
                issues.push_back("campplus_head_conv1_bn_relu_tolerance_exceeded");
            }
            if (!(layer1_err <= 2.0e-4f)) {
                issues.push_back("campplus_head_layer1_tolerance_exceeded");
            }
            if (!(layer2_err <= 3.0e-4f)) {
                issues.push_back("campplus_head_layer2_tolerance_exceeded");
            }
            if (!(conv2_err <= 4.0e-4f)) {
                issues.push_back("campplus_head_conv2_bn_relu_tolerance_exceeded");
            }
            if (!(output_err <= 4.0e-4f)) {
                issues.push_back("campplus_head_output_tolerance_exceeded");
            }
            if (!(tdnn_err <= 8.0e-4f)) {
                issues.push_back("campplus_xvector_tdnn_tolerance_exceeded");
            }
            if (!(block1_tdnnd1_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd1_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd1_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd1_tolerance_exceeded");
            }
            if (!(block1_tdnnd2_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd2_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd2_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd2_tolerance_exceeded");
            }
            if (!(block1_tdnnd3_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd3_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd3_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd3_tolerance_exceeded");
            }
            if (!(block1_tdnnd4_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd4_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd4_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd4_tolerance_exceeded");
            }
            if (!(block1_tdnnd5_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd5_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd5_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd5_tolerance_exceeded");
            }
            if (!(block1_tdnnd6_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd6_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd6_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd6_tolerance_exceeded");
            }
            if (!(block1_tdnnd7_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd7_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd7_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd7_tolerance_exceeded");
            }
            if (!(block1_tdnnd8_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd8_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd8_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd8_tolerance_exceeded");
            }
            if (!(block1_tdnnd9_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd9_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd9_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd9_tolerance_exceeded");
            }
            if (!(block1_tdnnd10_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd10_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd10_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd10_tolerance_exceeded");
            }
            if (!(block1_tdnnd11_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd11_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd11_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd11_tolerance_exceeded");
            }
            if (!(block1_tdnnd12_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_tdnnd12_tolerance_exceeded");
            }
            if (!(block1_after_tdnnd12_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block1_after_tdnnd12_tolerance_exceeded");
            }
            if (!(transit1_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_transit1_tolerance_exceeded");
            }
            if (!(block2_tdnnd1_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd1_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd1_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd1_tolerance_exceeded");
            }
            if (!(block2_tdnnd2_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd2_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd2_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd2_tolerance_exceeded");
            }
            if (!(block2_tdnnd3_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd3_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd3_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd3_tolerance_exceeded");
            }
            if (!(block2_tdnnd4_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd4_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd4_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd4_tolerance_exceeded");
            }
            if (!(block2_tdnnd5_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd5_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd5_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd5_tolerance_exceeded");
            }
            if (!(block2_tdnnd6_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd6_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd6_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd6_tolerance_exceeded");
            }
            if (!(block2_tdnnd7_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd7_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd7_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd7_tolerance_exceeded");
            }
            if (!(block2_tdnnd8_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd8_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd8_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd8_tolerance_exceeded");
            }
            if (!(block2_tdnnd9_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd9_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd9_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd9_tolerance_exceeded");
            }
            if (!(block2_tdnnd10_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd10_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd10_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd10_tolerance_exceeded");
            }
            if (!(block2_tdnnd11_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd11_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd11_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd11_tolerance_exceeded");
            }
            if (!(block2_tdnnd12_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd12_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd12_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd12_tolerance_exceeded");
            }
            if (!(block2_tdnnd13_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd13_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd13_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd13_tolerance_exceeded");
            }
            if (!(block2_tdnnd14_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd14_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd14_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd14_tolerance_exceeded");
            }
            if (!(block2_tdnnd15_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd15_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd15_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd15_tolerance_exceeded");
            }
            if (!(block2_tdnnd16_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd16_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd16_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd16_tolerance_exceeded");
            }
            if (!(block2_tdnnd17_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd17_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd17_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd17_tolerance_exceeded");
            }
            if (!(block2_tdnnd18_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd18_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd18_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd18_tolerance_exceeded");
            }
            if (!(block2_tdnnd19_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd19_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd19_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd19_tolerance_exceeded");
            }
            if (!(block2_tdnnd20_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd20_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd20_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd20_tolerance_exceeded");
            }
            if (!(block2_tdnnd21_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd21_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd21_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd21_tolerance_exceeded");
            }
            if (!(block2_tdnnd22_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd22_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd22_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd22_tolerance_exceeded");
            }
            if (!(block2_tdnnd23_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd23_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd23_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd23_tolerance_exceeded");
            }
            if (!(block2_tdnnd24_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_tdnnd24_tolerance_exceeded");
            }
            if (!(block2_after_tdnnd24_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block2_after_tdnnd24_tolerance_exceeded");
            }
            if (!(transit2_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_transit2_tolerance_exceeded");
            }
            if (!(block3_tdnnd1_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd1_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd1_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd1_tolerance_exceeded");
            }
            if (!(block3_tdnnd2_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd2_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd2_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd2_tolerance_exceeded");
            }
            if (!(block3_tdnnd3_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd3_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd3_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd3_tolerance_exceeded");
            }
            if (!(block3_tdnnd4_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd4_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd4_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd4_tolerance_exceeded");
            }
            if (!(block3_tdnnd5_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd5_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd5_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd5_tolerance_exceeded");
            }
            if (!(block3_tdnnd6_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd6_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd6_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd6_tolerance_exceeded");
            }
            if (!(block3_tdnnd7_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd7_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd7_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd7_tolerance_exceeded");
            }
            if (!(block3_tdnnd8_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd8_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd8_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd8_tolerance_exceeded");
            }
            if (!(block3_tdnnd9_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd9_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd9_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd9_tolerance_exceeded");
            }
            if (!(block3_tdnnd10_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd10_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd10_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd10_tolerance_exceeded");
            }
            if (!(block3_tdnnd11_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd11_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd11_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd11_tolerance_exceeded");
            }
            if (!(block3_tdnnd12_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd12_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd12_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd12_tolerance_exceeded");
            }
            if (!(block3_tdnnd13_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd13_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd13_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd13_tolerance_exceeded");
            }
            if (!(block3_tdnnd14_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd14_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd14_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd14_tolerance_exceeded");
            }
            if (!(block3_tdnnd15_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd15_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd15_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd15_tolerance_exceeded");
            }
            if (!(block3_tdnnd16_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_tdnnd16_tolerance_exceeded");
            }
            if (!(block3_after_tdnnd16_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_block3_after_tdnnd16_tolerance_exceeded");
            }
            if (!(transit3_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_transit3_tolerance_exceeded");
            }
            if (!(out_nonlinear_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_out_nonlinear_tolerance_exceeded");
            }
            if (!(stats_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_stats_tolerance_exceeded");
            }
            if (!(dense_err <= 1.0e-3f)) {
                issues.push_back("campplus_xvector_dense_tolerance_exceeded");
            }
        }

        const bool ok = issues.empty();
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_campplus_head_golden\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"campplus_golden_dir\": \"" << json_escape(campplus_golden_dir) << "\",\n";
        std::cout << "  \"output_fbank_f32\": \"" << json_escape(manifest.output_fbank_f32) << "\",\n";
        std::cout << "  \"fbank_frames\": " << manifest.fbank_frames << ",\n";
        std::cout << "  \"expected_shape\": \"[1,32,80," << manifest.fbank_frames << "]\",\n";
        std::cout << "  \"expected_values\": " << expected_values << ",\n";
        std::cout << "  \"expected_layer1_shape\": \"[1,32,40," << manifest.fbank_frames << "]\",\n";
        std::cout << "  \"expected_layer1_values\": " << expected_layer1_values << ",\n";
        std::cout << "  \"expected_layer2_shape\": \"[1,32,20," << manifest.fbank_frames << "]\",\n";
        std::cout << "  \"expected_layer2_values\": " << expected_layer2_values << ",\n";
        std::cout << "  \"expected_conv2_shape\": \"[1,32,10," << manifest.fbank_frames << "]\",\n";
        std::cout << "  \"expected_conv2_values\": " << expected_conv2_values << ",\n";
        std::cout << "  \"expected_output_shape\": \"[1,320," << manifest.fbank_frames << "]\",\n";
        std::cout << "  \"expected_output_values\": " << expected_output_values << ",\n";
        std::cout << "  \"expected_tdnn_shape\": \"[1,128," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_tdnn_values\": " << expected_tdnn_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd1_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd1_values\": " << expected_block1_tdnnd1_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd1_shape\": \"[1,160," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd1_values\": " << expected_block1_after_tdnnd1_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd2_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd2_values\": " << expected_block1_tdnnd2_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd2_shape\": \"[1,192," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd2_values\": " << expected_block1_after_tdnnd2_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd3_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd3_values\": " << expected_block1_tdnnd3_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd3_shape\": \"[1,224," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd3_values\": " << expected_block1_after_tdnnd3_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd4_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd4_values\": " << expected_block1_tdnnd4_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd4_shape\": \"[1,256," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd4_values\": " << expected_block1_after_tdnnd4_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd5_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd5_values\": " << expected_block1_tdnnd5_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd5_shape\": \"[1,288," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd5_values\": " << expected_block1_after_tdnnd5_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd6_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd6_values\": " << expected_block1_tdnnd6_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd6_shape\": \"[1,320," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd6_values\": " << expected_block1_after_tdnnd6_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd7_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd7_values\": " << expected_block1_tdnnd7_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd7_shape\": \"[1,352," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd7_values\": " << expected_block1_after_tdnnd7_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd8_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd8_values\": " << expected_block1_tdnnd8_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd8_shape\": \"[1,384," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd8_values\": " << expected_block1_after_tdnnd8_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd9_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd9_values\": " << expected_block1_tdnnd9_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd9_shape\": \"[1,416," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd9_values\": " << expected_block1_after_tdnnd9_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd10_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd10_values\": " << expected_block1_tdnnd10_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd10_shape\": \"[1,448," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd10_values\": " << expected_block1_after_tdnnd10_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd11_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd11_values\": " << expected_block1_tdnnd11_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd11_shape\": \"[1,480," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd11_values\": " << expected_block1_after_tdnnd11_values << ",\n";
        std::cout << "  \"expected_block1_tdnnd12_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_tdnnd12_values\": " << expected_block1_tdnnd12_values << ",\n";
        std::cout << "  \"expected_block1_after_tdnnd12_shape\": \"[1,512," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block1_after_tdnnd12_values\": " << expected_block1_after_tdnnd12_values << ",\n";
        std::cout << "  \"expected_transit1_shape\": \"[1,256," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_transit1_values\": " << expected_transit1_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd1_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd1_values\": " << expected_block2_tdnnd1_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd1_shape\": \"[1,288," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd1_values\": " << expected_block2_after_tdnnd1_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd2_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd2_values\": " << expected_block2_tdnnd2_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd2_shape\": \"[1,320," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd2_values\": " << expected_block2_after_tdnnd2_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd3_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd3_values\": " << expected_block2_tdnnd3_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd3_shape\": \"[1,352," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd3_values\": " << expected_block2_after_tdnnd3_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd4_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd4_values\": " << expected_block2_tdnnd4_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd4_shape\": \"[1,384," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd4_values\": " << expected_block2_after_tdnnd4_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd5_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd5_values\": " << expected_block2_tdnnd5_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd5_shape\": \"[1,416," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd5_values\": " << expected_block2_after_tdnnd5_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd6_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd6_values\": " << expected_block2_tdnnd6_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd6_shape\": \"[1,448," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd6_values\": " << expected_block2_after_tdnnd6_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd7_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd7_values\": " << expected_block2_tdnnd7_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd7_shape\": \"[1,480," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd7_values\": " << expected_block2_after_tdnnd7_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd8_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd8_values\": " << expected_block2_tdnnd8_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd8_shape\": \"[1,512," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd8_values\": " << expected_block2_after_tdnnd8_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd9_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd9_values\": " << expected_block2_tdnnd9_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd9_shape\": \"[1,544," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd9_values\": " << expected_block2_after_tdnnd9_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd10_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd10_values\": " << expected_block2_tdnnd10_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd10_shape\": \"[1,576," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd10_values\": " << expected_block2_after_tdnnd10_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd11_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd11_values\": " << expected_block2_tdnnd11_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd11_shape\": \"[1,608," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd11_values\": " << expected_block2_after_tdnnd11_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd12_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd12_values\": " << expected_block2_tdnnd12_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd12_shape\": \"[1,640," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd12_values\": " << expected_block2_after_tdnnd12_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd13_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd13_values\": " << expected_block2_tdnnd13_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd13_shape\": \"[1,672," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd13_values\": " << expected_block2_after_tdnnd13_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd14_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd14_values\": " << expected_block2_tdnnd14_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd14_shape\": \"[1,704," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd14_values\": " << expected_block2_after_tdnnd14_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd15_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd15_values\": " << expected_block2_tdnnd15_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd15_shape\": \"[1,736," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd15_values\": " << expected_block2_after_tdnnd15_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd16_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd16_values\": " << expected_block2_tdnnd16_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd16_shape\": \"[1,768," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd16_values\": " << expected_block2_after_tdnnd16_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd17_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd17_values\": " << expected_block2_tdnnd17_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd17_shape\": \"[1,800," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd17_values\": " << expected_block2_after_tdnnd17_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd18_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd18_values\": " << expected_block2_tdnnd18_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd18_shape\": \"[1,832," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd18_values\": " << expected_block2_after_tdnnd18_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd19_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd19_values\": " << expected_block2_tdnnd19_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd19_shape\": \"[1,864," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd19_values\": " << expected_block2_after_tdnnd19_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd20_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd20_values\": " << expected_block2_tdnnd20_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd20_shape\": \"[1,896," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd20_values\": " << expected_block2_after_tdnnd20_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd21_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd21_values\": " << expected_block2_tdnnd21_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd21_shape\": \"[1,928," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd21_values\": " << expected_block2_after_tdnnd21_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd22_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd22_values\": " << expected_block2_tdnnd22_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd22_shape\": \"[1,960," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd22_values\": " << expected_block2_after_tdnnd22_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd23_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd23_values\": " << expected_block2_tdnnd23_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd23_shape\": \"[1,992," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd23_values\": " << expected_block2_after_tdnnd23_values << ",\n";
        std::cout << "  \"expected_block2_tdnnd24_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_tdnnd24_values\": " << expected_block2_tdnnd24_values << ",\n";
        std::cout << "  \"expected_block2_after_tdnnd24_shape\": \"[1,1024," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block2_after_tdnnd24_values\": " << expected_block2_after_tdnnd24_values << ",\n";
        std::cout << "  \"expected_transit2_shape\": \"[1,512," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_transit2_values\": " << expected_transit2_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd1_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd1_values\": " << expected_block3_tdnnd1_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd1_shape\": \"[1,544," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd1_values\": " << expected_block3_after_tdnnd1_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd2_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd2_values\": " << expected_block3_tdnnd2_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd2_shape\": \"[1,576," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd2_values\": " << expected_block3_after_tdnnd2_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd3_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd3_values\": " << expected_block3_tdnnd3_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd3_shape\": \"[1,608," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd3_values\": " << expected_block3_after_tdnnd3_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd4_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd4_values\": " << expected_block3_tdnnd4_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd4_shape\": \"[1,640," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd4_values\": " << expected_block3_after_tdnnd4_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd5_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd5_values\": " << expected_block3_tdnnd5_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd5_shape\": \"[1,672," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd5_values\": " << expected_block3_after_tdnnd5_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd6_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd6_values\": " << expected_block3_tdnnd6_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd6_shape\": \"[1,704," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd6_values\": " << expected_block3_after_tdnnd6_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd7_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd7_values\": " << expected_block3_tdnnd7_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd7_shape\": \"[1,736," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd7_values\": " << expected_block3_after_tdnnd7_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd8_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd8_values\": " << expected_block3_tdnnd8_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd8_shape\": \"[1,768," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd8_values\": " << expected_block3_after_tdnnd8_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd9_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd9_values\": " << expected_block3_tdnnd9_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd9_shape\": \"[1,800," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd9_values\": " << expected_block3_after_tdnnd9_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd10_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd10_values\": " << expected_block3_tdnnd10_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd10_shape\": \"[1,832," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd10_values\": " << expected_block3_after_tdnnd10_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd11_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd11_values\": " << expected_block3_tdnnd11_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd11_shape\": \"[1,864," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd11_values\": " << expected_block3_after_tdnnd11_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd12_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd12_values\": " << expected_block3_tdnnd12_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd12_shape\": \"[1,896," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd12_values\": " << expected_block3_after_tdnnd12_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd13_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd13_values\": " << expected_block3_tdnnd13_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd13_shape\": \"[1,928," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd13_values\": " << expected_block3_after_tdnnd13_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd14_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd14_values\": " << expected_block3_tdnnd14_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd14_shape\": \"[1,960," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd14_values\": " << expected_block3_after_tdnnd14_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd15_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd15_values\": " << expected_block3_tdnnd15_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd15_shape\": \"[1,992," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd15_values\": " << expected_block3_after_tdnnd15_values << ",\n";
        std::cout << "  \"expected_block3_tdnnd16_shape\": \"[1,32," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_tdnnd16_values\": " << expected_block3_tdnnd16_values << ",\n";
        std::cout << "  \"expected_block3_after_tdnnd16_shape\": \"[1,1024," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_block3_after_tdnnd16_values\": " << expected_block3_after_tdnnd16_values << ",\n";
        std::cout << "  \"expected_transit3_shape\": \"[1,512," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_transit3_values\": " << expected_transit3_values << ",\n";
        std::cout << "  \"expected_out_nonlinear_shape\": \"[1,512," << expected_tdnn_frames << "]\",\n";
        std::cout << "  \"expected_out_nonlinear_values\": " << expected_out_nonlinear_values << ",\n";
        std::cout << "  \"expected_stats_shape\": \"[1,1024]\",\n";
        std::cout << "  \"expected_stats_values\": " << expected_stats_values << ",\n";
        std::cout << "  \"expected_dense_shape\": \"[1,192]\",\n";
        std::cout << "  \"expected_dense_values\": " << expected_dense_values << ",\n";
        std::cout << "  \"campplus_head_conv1_bn_relu_max_abs_error\": " << err << ",\n";
        std::cout << "  \"campplus_head_conv1_bn_relu_tolerance\": 0.0001,\n";
        std::cout << "  \"campplus_head_layer1_max_abs_error\": " << layer1_err << ",\n";
        std::cout << "  \"campplus_head_layer1_tolerance\": 0.0002,\n";
        std::cout << "  \"campplus_head_layer2_max_abs_error\": " << layer2_err << ",\n";
        std::cout << "  \"campplus_head_layer2_tolerance\": 0.0003,\n";
        std::cout << "  \"campplus_head_conv2_bn_relu_max_abs_error\": " << conv2_err << ",\n";
        std::cout << "  \"campplus_head_conv2_bn_relu_tolerance\": 0.0004,\n";
        std::cout << "  \"campplus_head_output_max_abs_error\": " << output_err << ",\n";
        std::cout << "  \"campplus_head_output_tolerance\": 0.0004,\n";
        std::cout << "  \"campplus_xvector_tdnn_max_abs_error\": " << tdnn_err << ",\n";
        std::cout << "  \"campplus_xvector_tdnn_tolerance\": 0.0008,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd1_max_abs_error\": " << block1_tdnnd1_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd1_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd1_max_abs_error\": " << block1_after_tdnnd1_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd1_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd2_max_abs_error\": " << block1_tdnnd2_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd2_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd2_max_abs_error\": " << block1_after_tdnnd2_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd2_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd3_max_abs_error\": " << block1_tdnnd3_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd3_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd3_max_abs_error\": " << block1_after_tdnnd3_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd3_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd4_max_abs_error\": " << block1_tdnnd4_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd4_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd4_max_abs_error\": " << block1_after_tdnnd4_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd4_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd5_max_abs_error\": " << block1_tdnnd5_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd5_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd5_max_abs_error\": " << block1_after_tdnnd5_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd5_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd6_max_abs_error\": " << block1_tdnnd6_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd6_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd6_max_abs_error\": " << block1_after_tdnnd6_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd6_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd7_max_abs_error\": " << block1_tdnnd7_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd7_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd7_max_abs_error\": " << block1_after_tdnnd7_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd7_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd8_max_abs_error\": " << block1_tdnnd8_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd8_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd8_max_abs_error\": " << block1_after_tdnnd8_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd8_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd9_max_abs_error\": " << block1_tdnnd9_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd9_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd9_max_abs_error\": " << block1_after_tdnnd9_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd9_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd10_max_abs_error\": " << block1_tdnnd10_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd10_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd10_max_abs_error\": " << block1_after_tdnnd10_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd10_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd11_max_abs_error\": " << block1_tdnnd11_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd11_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd11_max_abs_error\": " << block1_after_tdnnd11_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd11_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd12_max_abs_error\": " << block1_tdnnd12_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_tdnnd12_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd12_max_abs_error\": " << block1_after_tdnnd12_err << ",\n";
        std::cout << "  \"campplus_xvector_block1_after_tdnnd12_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_transit1_max_abs_error\": " << transit1_err << ",\n";
        std::cout << "  \"campplus_xvector_transit1_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd1_max_abs_error\": " << block2_tdnnd1_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd1_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd1_max_abs_error\": " << block2_after_tdnnd1_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd1_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd2_max_abs_error\": " << block2_tdnnd2_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd2_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd2_max_abs_error\": " << block2_after_tdnnd2_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd2_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd3_max_abs_error\": " << block2_tdnnd3_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd3_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd3_max_abs_error\": " << block2_after_tdnnd3_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd3_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd4_max_abs_error\": " << block2_tdnnd4_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd4_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd4_max_abs_error\": " << block2_after_tdnnd4_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd4_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd5_max_abs_error\": " << block2_tdnnd5_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd5_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd5_max_abs_error\": " << block2_after_tdnnd5_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd5_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd6_max_abs_error\": " << block2_tdnnd6_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd6_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd6_max_abs_error\": " << block2_after_tdnnd6_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd6_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd7_max_abs_error\": " << block2_tdnnd7_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd7_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd7_max_abs_error\": " << block2_after_tdnnd7_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd7_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd8_max_abs_error\": " << block2_tdnnd8_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd8_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd8_max_abs_error\": " << block2_after_tdnnd8_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd8_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd9_max_abs_error\": " << block2_tdnnd9_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd9_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd9_max_abs_error\": " << block2_after_tdnnd9_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd9_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd10_max_abs_error\": " << block2_tdnnd10_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd10_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd10_max_abs_error\": " << block2_after_tdnnd10_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd10_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd11_max_abs_error\": " << block2_tdnnd11_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd11_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd11_max_abs_error\": " << block2_after_tdnnd11_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd11_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd12_max_abs_error\": " << block2_tdnnd12_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd12_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd12_max_abs_error\": " << block2_after_tdnnd12_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd12_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd13_max_abs_error\": " << block2_tdnnd13_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd13_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd13_max_abs_error\": " << block2_after_tdnnd13_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd13_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd14_max_abs_error\": " << block2_tdnnd14_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd14_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd14_max_abs_error\": " << block2_after_tdnnd14_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd14_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd15_max_abs_error\": " << block2_tdnnd15_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd15_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd15_max_abs_error\": " << block2_after_tdnnd15_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd15_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd16_max_abs_error\": " << block2_tdnnd16_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd16_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd16_max_abs_error\": " << block2_after_tdnnd16_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd16_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd17_max_abs_error\": " << block2_tdnnd17_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd17_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd17_max_abs_error\": " << block2_after_tdnnd17_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd17_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd18_max_abs_error\": " << block2_tdnnd18_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd18_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd18_max_abs_error\": " << block2_after_tdnnd18_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd18_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd19_max_abs_error\": " << block2_tdnnd19_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd19_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd19_max_abs_error\": " << block2_after_tdnnd19_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd19_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd20_max_abs_error\": " << block2_tdnnd20_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd20_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd20_max_abs_error\": " << block2_after_tdnnd20_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd20_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd21_max_abs_error\": " << block2_tdnnd21_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd21_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd21_max_abs_error\": " << block2_after_tdnnd21_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd21_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd22_max_abs_error\": " << block2_tdnnd22_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd22_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd22_max_abs_error\": " << block2_after_tdnnd22_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd22_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd23_max_abs_error\": " << block2_tdnnd23_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd23_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd23_max_abs_error\": " << block2_after_tdnnd23_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd23_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd24_max_abs_error\": " << block2_tdnnd24_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_tdnnd24_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd24_max_abs_error\": " << block2_after_tdnnd24_err << ",\n";
        std::cout << "  \"campplus_xvector_block2_after_tdnnd24_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_transit2_max_abs_error\": " << transit2_err << ",\n";
        std::cout << "  \"campplus_xvector_transit2_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd1_max_abs_error\": " << block3_tdnnd1_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd1_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd1_max_abs_error\": " << block3_after_tdnnd1_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd1_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd2_max_abs_error\": " << block3_tdnnd2_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd2_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd2_max_abs_error\": " << block3_after_tdnnd2_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd2_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd3_max_abs_error\": " << block3_tdnnd3_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd3_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd3_max_abs_error\": " << block3_after_tdnnd3_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd3_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd4_max_abs_error\": " << block3_tdnnd4_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd4_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd4_max_abs_error\": " << block3_after_tdnnd4_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd4_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd5_max_abs_error\": " << block3_tdnnd5_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd5_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd5_max_abs_error\": " << block3_after_tdnnd5_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd5_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd6_max_abs_error\": " << block3_tdnnd6_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd6_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd6_max_abs_error\": " << block3_after_tdnnd6_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd6_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd7_max_abs_error\": " << block3_tdnnd7_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd7_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd7_max_abs_error\": " << block3_after_tdnnd7_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd7_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd8_max_abs_error\": " << block3_tdnnd8_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd8_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd8_max_abs_error\": " << block3_after_tdnnd8_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd8_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd9_max_abs_error\": " << block3_tdnnd9_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd9_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd9_max_abs_error\": " << block3_after_tdnnd9_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd9_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd10_max_abs_error\": " << block3_tdnnd10_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd10_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd10_max_abs_error\": " << block3_after_tdnnd10_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd10_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd11_max_abs_error\": " << block3_tdnnd11_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd11_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd11_max_abs_error\": " << block3_after_tdnnd11_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd11_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd12_max_abs_error\": " << block3_tdnnd12_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd12_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd12_max_abs_error\": " << block3_after_tdnnd12_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd12_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd13_max_abs_error\": " << block3_tdnnd13_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd13_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd13_max_abs_error\": " << block3_after_tdnnd13_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd13_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd14_max_abs_error\": " << block3_tdnnd14_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd14_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd14_max_abs_error\": " << block3_after_tdnnd14_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd14_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd15_max_abs_error\": " << block3_tdnnd15_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd15_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd15_max_abs_error\": " << block3_after_tdnnd15_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd15_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd16_max_abs_error\": " << block3_tdnnd16_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_tdnnd16_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd16_max_abs_error\": " << block3_after_tdnnd16_err << ",\n";
        std::cout << "  \"campplus_xvector_block3_after_tdnnd16_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_transit3_max_abs_error\": " << transit3_err << ",\n";
        std::cout << "  \"campplus_xvector_transit3_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_out_nonlinear_max_abs_error\": " << out_nonlinear_err << ",\n";
        std::cout << "  \"campplus_xvector_out_nonlinear_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_stats_max_abs_error\": " << stats_err << ",\n";
        std::cout << "  \"campplus_xvector_stats_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_xvector_dense_max_abs_error\": " << dense_err << ",\n";
        std::cout << "  \"campplus_xvector_dense_tolerance\": 0.001,\n";
        std::cout << "  \"campplus_contract_issues\": ";
        print_json_string_array(campplus.issues);
        std::cout << ",\n";
        std::cout << "  \"clone_campplus_head_golden_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_campplus_head_conv1_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_head_layer1_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_head_layer2_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_head_output_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_head_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_tdnn_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd1_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd2_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd3_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd4_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd5_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd6_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd7_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd8_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd9_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd10_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd11_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block1_tdnnd12_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_transit1_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd1_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd2_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd3_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd4_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd5_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd6_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd7_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd8_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd9_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd10_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd11_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd12_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd13_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd14_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd15_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd16_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd17_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd18_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd19_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd20_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd21_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd22_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd23_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block2_tdnnd24_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_transit2_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd1_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd2_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd3_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd4_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd5_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd6_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd7_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd8_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd9_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd10_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd11_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd12_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd13_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd14_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd15_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_block3_tdnnd16_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_transit3_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_out_nonlinear_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_stats_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_xvector_dense_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_style_forward\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native clone semantic/acoustic speech encoders for voice tensor creation\"\n";
        std::cout << "}\n";
        return ok;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_campplus_head_golden\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"campplus_golden_dir\": \"" << json_escape(campplus_golden_dir) << "\",\n";
        std::cout << "  \"ready_native_campplus_head_forward\": false,\n";
        std::cout << "  \"ready_native_campplus_style_forward\": false,\n";
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
                                                       const std::string& stage,
                                                       const std::string& feature_manifest);

