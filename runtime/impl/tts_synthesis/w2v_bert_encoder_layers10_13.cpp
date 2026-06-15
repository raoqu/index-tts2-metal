// w2v-BERT conformer encoder: layers 10-13 (ffn/attention/conv ops, CPU + Metal, plus per-op clone CLI entry points).
// Part of the tts_synthesis translation unit (see ../tts_synthesis.cpp);
// these files are #included in order into a single TU, so definition order
// across files is significant.

std::vector<float> run_w2v_bert_layer10_ffn1_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& layer9_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.output_dense.bias");
    if (tokens == 0 || layer9_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_ffn1_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(layer9_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = layer9_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer10_ffn1_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& layer9_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn1.output_dense.bias");
    if (tokens == 0 || layer9_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_ffn1_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.10.ffn1_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.10.ffn1_layer_norm.bias.resident",
        norm_b,
        layer9_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.10.ffn1.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.10.ffn1.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(layer9_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_ffn1_residual(const std::string& model_bundle_dir,
                                                              const std::string& layer9_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer9_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer9_ffn2_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_ffn1_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer9_ffn2_residual_f32\": \"" << json_escape(layer9_residual_f32) << "\",\n";
        std::cout << "  \"output_layer10_ffn1_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_ffn1_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer9_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer9_ffn2_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer10_ffn1_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_ffn1_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_ffn1_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_ffn1_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 self-attention, convolution, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer9_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer10_ffn1_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer10_ffn1_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

W2VLayerQkv run_w2v_bert_layer10_qkv_cpu(const mit2::Bundle& model,
                                         const std::vector<float>& layer10_ffn1_residual,
                                         uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_v.weight");
    if (tokens == 0 || layer10_ffn1_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        q_w.size() != static_cast<size_t>(1024) * 1024 ||
        k_w.size() != static_cast<size_t>(1024) * 1024 ||
        v_w.size() != static_cast<size_t>(1024) * 1024) {
        throw std::invalid_argument("w2v_bert_layer10_qkv invalid input sizes");
    }
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_v.bias");
    return {
        cpu_linear_rows(q_w, q_b, layer10_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(k_w, k_b, layer10_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(v_w, v_b, layer10_ffn1_residual, tokens, 1024, 1024),
    };
}

W2VLayerQkv run_w2v_bert_layer10_qkv_metal(mit2::MetalContext& metal,
                                           const mit2::Bundle& model,
                                           const std::vector<float>& layer10_ffn1_residual,
                                           uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_v.weight");
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_v.bias");
    return {
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.10.self_attn.linear_q.weight.resident",
                                       q_w,
                                       "w2v_bert.encoder.layers.10.self_attn.linear_q.bias.resident",
                                       q_b,
                                       layer10_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.10.self_attn.linear_k.weight.resident",
                                       k_w,
                                       "w2v_bert.encoder.layers.10.self_attn.linear_k.bias.resident",
                                       k_b,
                                       layer10_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.10.self_attn.linear_v.weight.resident",
                                       v_w,
                                       "w2v_bert.encoder.layers.10.self_attn.linear_v.bias.resident",
                                       v_b,
                                       layer10_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
    };
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_qkv(const std::string& model_bundle_dir,
                                                   const std::string& layer10_ffn1_residual_f32,
                                                   uint32_t tokens,
                                                   const std::string& output_dir) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer10_ffn1_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer10_ffn1_residual");

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
    const auto output_q = (dir / "w2v_layer10_q.f32").string();
    const auto output_k = (dir / "w2v_layer10_k.f32").string();
    const auto output_v = (dir / "w2v_layer10_v.f32").string();
    const auto output_manifest = (dir / "w2v_layer10_qkv.manifest.json").string();

    auto print_failure = [&](const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_qkv\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_ffn1_residual_f32\": \"" << json_escape(layer10_ffn1_residual_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_qkv_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_qkv\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_qkv\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_failure(issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer10_ffn1_residual_f32);
        std::string execution_backend = "metal";
        W2VLayerQkv qkv;
        try {
            mit2::MetalContext metal;
            qkv = run_w2v_bert_layer10_qkv_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            qkv = run_w2v_bert_layer10_qkv_cpu(model, input, tokens);
        }

        std::filesystem::create_directories(dir);
        write_raw_f32(output_q, qkv.q);
        write_raw_f32(output_k, qkv.k);
        write_raw_f32(output_v, qkv.v);
        std::ostringstream manifest_json;
        manifest_json << "{\n";
        manifest_json << "  \"format\": \"mit2-w2v-layer10-qkv-sidecars\",\n";
        manifest_json << "  \"version\": 1,\n";
        manifest_json << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        manifest_json << "  \"w2v_layer10_ffn1_residual_f32\": \"" << json_escape(layer10_ffn1_residual_f32) << "\",\n";
        manifest_json << "  \"w2v_tokens\": " << tokens << ",\n";
        manifest_json << "  \"output_q_f32\": \"" << json_escape(output_q) << "\",\n";
        manifest_json << "  \"output_q_sha256\": \"" << file_sha256_hex(output_q) << "\",\n";
        manifest_json << "  \"output_k_f32\": \"" << json_escape(output_k) << "\",\n";
        manifest_json << "  \"output_k_sha256\": \"" << file_sha256_hex(output_k) << "\",\n";
        manifest_json << "  \"output_v_f32\": \"" << json_escape(output_v) << "\",\n";
        manifest_json << "  \"output_v_sha256\": \"" << file_sha256_hex(output_v) << "\",\n";
        manifest_json << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        manifest_json << "  \"ready_native_w2v_bert_layer10_qkv\": true,\n";
        manifest_json << "  \"ready_native_w2v_bert_semantic_features\": false\n";
        manifest_json << "}\n";
        write_text_file(output_manifest, manifest_json.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_qkv\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_ffn1_residual_f32\": \"" << json_escape(layer10_ffn1_residual_f32) << "\",\n";
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
        std::cout << "  \"layer10_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_ffn1_residual_values\": " << input.size() << ",\n";
        std::cout << "  \"q_values\": " << qkv.q.size() << ",\n";
        std::cout << "  \"k_values\": " << qkv.k.size() << ",\n";
        std::cout << "  \"v_values\": " << qkv.v.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_qkv_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_qkv\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_qkv\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        print_failure({e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_attention_context_cpu(const std::vector<float>& q,
                                                              const std::vector<float>& k,
                                                              const std::vector<float>& v,
                                                              const std::vector<uint32_t>& key_mask,
                                                              uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    if (tokens == 0 || q.size() != static_cast<size_t>(tokens) * heads * head_dim ||
        k.size() != q.size() || v.size() != q.size() || key_mask.size() != tokens) {
        throw std::invalid_argument("w2v_bert_layer10_attention_context invalid input sizes");
    }
    return cpu_cross_attention_heads_masked(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

std::vector<float> run_w2v_bert_layer10_attention_context_metal(mit2::MetalContext& metal,
                                                               const std::vector<float>& q,
                                                               const std::vector<float>& k,
                                                               const std::vector<float>& v,
                                                               const std::vector<uint32_t>& key_mask,
                                                               uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    return metal.cross_attention_heads_masked_f32(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_attention(const std::string& q_f32,
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
    append_raw_f32_count_issue(issues, q_f32, qkv_values, "w2v_layer10_q");
    append_raw_f32_count_issue(issues, k_f32, qkv_values, "w2v_layer10_k");
    append_raw_f32_count_issue(issues, v_f32, qkv_values, "w2v_layer10_v");
    append_raw_u32_count_issue(issues, attention_mask_u32, tokens, "w2v_attention_mask");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::optional<std::vector<float>>& output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_attention\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer10_q_f32\": \"" << json_escape(q_f32) << "\",\n";
        std::cout << "  \"w2v_layer10_k_f32\": \"" << json_escape(k_f32) << "\",\n";
        std::cout << "  \"w2v_layer10_v_f32\": \"" << json_escape(v_f32) << "\",\n";
        std::cout << "  \"w2v_attention_mask_u32\": \"" << json_escape(attention_mask_u32) << "\",\n";
        std::cout << "  \"output_layer10_context_f32\": \"" << json_escape(output_context_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_context_sha256\": \"" << file_sha256_hex(output_context_f32) << "\",\n";
        }
        std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_context_shape\": \"[1," << tokens << ",1024]\",\n";
        if (output) {
            std::cout << "  \"layer10_context_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer10_attention_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_attention_context\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_attention_context\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 attention output projection/residual/norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "not_run", std::nullopt, issues);
        return false;
    }

    try {
        const auto q = read_raw_f32(q_f32);
        const auto k = read_raw_f32(k_f32);
        const auto v = read_raw_f32(v_f32);
        const auto mask = read_raw_u32(attention_mask_u32);
        std::string execution_backend = "cpu_distance_emb";
        const std::vector<float> context = cpu_w2v_bert_cross_attention_with_distance(q, k, v, mask, tokens, dist_emb, 64, 8);
        write_raw_f32(output_context_f32, context);
        print_report(true, execution_backend, context, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "not_run", std::nullopt, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_attention_project_cpu(const mit2::Bundle& model,
                                                              const std::vector<float>& context,
                                                              uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.10.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_out.bias");
    }
    if (tokens == 0 || context.size() != static_cast<size_t>(tokens) * 1024u ||
        out_w.size() != static_cast<size_t>(1024) * 1024 || out_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_attention_project invalid input sizes");
    }
    return cpu_linear_rows(out_w, out_b, context, tokens, 1024, 1024);
}

std::vector<float> run_w2v_bert_layer10_attention_project_metal(mit2::MetalContext& metal,
                                                                const mit2::Bundle& model,
                                                                const std::vector<float>& context,
                                                                uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.10.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn.linear_out.bias");
    }
    return metal.linear_rows_f32_resident("w2v_bert.encoder.layers.10.self_attn.linear_out.weight.resident",
                                          out_w,
                                          "w2v_bert.encoder.layers.10.self_attn.linear_out.bias.resident_or_zero",
                                          out_b,
                                          context,
                                          tokens,
                                          1024,
                                          1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_attention_project(const std::string& model_bundle_dir,
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
                               "w2v_layer10_attention_context");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_attention_project\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_layer10_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_attention_sha256\": \"" << file_sha256_hex(output_attention_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer10_context_values\": " << input->size() << ",\n";
            std::cout << "  \"layer10_attention_projection_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_attention_project_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_attention_projection\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_attention_projection\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 attention residual/norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
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
            projected = run_w2v_bert_layer10_attention_project_metal(metal, model, context, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            projected = run_w2v_bert_layer10_attention_project_cpu(model, context, tokens);
        }
        write_raw_f32(output_attention_f32, projected);
        print_report(true, backend, &context, &projected, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_attention_residual_cpu(const std::vector<float>& ffn1_residual,
                                                               const std::vector<float>& attention_projection,
                                                               uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer10_attention_residual invalid input sizes");
    }
    std::vector<float> residual(expected);
    for (size_t i = 0; i < expected; ++i) {
        residual[i] = ffn1_residual[i] + attention_projection[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer10_attention_residual_metal(mit2::MetalContext& metal,
                                                                 const std::vector<float>& ffn1_residual,
                                                                 const std::vector<float>& attention_projection,
                                                                 uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer10_attention_residual invalid input sizes");
    }
    return metal.add_f32(ffn1_residual, attention_projection);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_attention_residual(const std::string& ffn1_residual_f32,
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
                               "w2v_layer10_ffn1_residual");
    append_raw_f32_count_issue(issues,
                               attention_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer10_attention_projection");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* ffn1_residual,
                            const std::vector<float>* attention_projection,
                            const std::vector<float>* residual,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_attention_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer10_ffn1_residual_f32\": \"" << json_escape(ffn1_residual_f32) << "\",\n";
        std::cout << "  \"w2v_layer10_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_layer10_attention_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_attention_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (ffn1_residual && attention_projection && residual) {
            std::cout << "  \"layer10_ffn1_residual_values\": " << ffn1_residual->size() << ",\n";
            std::cout << "  \"layer10_attention_projection_values\": " << attention_projection->size() << ",\n";
            std::cout << "  \"layer10_attention_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer10_attention_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_attention_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_attention_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 attention norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
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
            residual = run_w2v_bert_layer10_attention_residual_metal(metal,
                                                                     ffn1_residual,
                                                                     attention_projection,
                                                                     tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer10_attention_residual_cpu(ffn1_residual, attention_projection, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &ffn1_residual, &attention_projection, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_attention_norm_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& attention_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn_layer_norm.bias");
    if (tokens == 0 || attention_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_attention_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer10_attention_norm_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& attention_residual,
                                                             uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.self_attn_layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.10.self_attn_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.10.self_attn_layer_norm.bias.resident",
                                             norm_b,
                                             attention_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_attention_norm(const std::string& model_bundle_dir,
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
                               "w2v_layer10_attention_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_attention_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_layer10_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_attention_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer10_attention_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer10_attention_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_attention_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_attention_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_attention_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
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
            normed = run_w2v_bert_layer10_attention_norm_metal(metal, model, attention_residual, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer10_attention_norm_cpu(model, attention_residual, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_residual, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_conv_norm_cpu(const mit2::Bundle& model,
                                                      const std::vector<float>& attention_norm,
                                                      uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.layer_norm.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_conv_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_norm, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer10_conv_norm_metal(mit2::MetalContext& metal,
                                                        const mit2::Bundle& model,
                                                        const std::vector<float>& attention_norm,
                                                        uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.10.conv_module.layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.10.conv_module.layer_norm.bias.resident",
                                             norm_b,
                                             attention_norm,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_conv_norm(const std::string& model_bundle_dir,
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
                               "w2v_layer10_attention_norm");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_conv_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer10_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_conv_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer10_attention_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer10_conv_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_conv_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_conv_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_conv_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 convolution GLU, depthwise/residual, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
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
            normed = run_w2v_bert_layer10_conv_norm_metal(metal, model, attention_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer10_conv_norm_cpu(model, attention_norm, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_norm, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_conv_glu_cpu(const mit2::Bundle& model,
                                                     const std::vector<float>& conv_norm,
                                                     uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer10_conv_glu invalid input sizes");
    }
    const auto projected = cpu_linear_rows(pw1_w, pw1_b, conv_norm, tokens, 2048, 1024);
    return cpu_glu_split(projected, tokens, 1024);
}

std::vector<float> run_w2v_bert_layer10_conv_glu_metal(mit2::MetalContext& metal,
                                                       const mit2::Bundle& model,
                                                       const std::vector<float>& conv_norm,
                                                       uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer10_conv_glu invalid input sizes");
    }
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.weight.resident",
        pw1_w,
        "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.bias.resident",
        pw1_b,
        conv_norm,
        tokens,
        2048,
        1024);
    return metal.glu_split_f32(projected, tokens, 1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_conv_glu(const std::string& model_bundle_dir,
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
                               "w2v_layer10_conv_norm");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_conv_glu\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer10_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_conv_glu_sha256\": \"" << file_sha256_hex(output_glu_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer10_conv_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer10_conv_glu_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_conv_glu_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_conv_glu\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_conv_glu\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 depthwise convolution, convolution residual, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
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
            glu = run_w2v_bert_layer10_conv_glu_metal(metal, model, conv_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            glu = run_w2v_bert_layer10_conv_glu_cpu(model, conv_norm, tokens);
        }
        write_raw_f32(output_glu_f32, glu);
        print_report(true, backend, &conv_norm, &glu, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_conv_depthwise_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& conv_glu,
                                                           uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.bias");
    if (tokens == 0 || conv_glu.size() != static_cast<size_t>(tokens) * 1024u ||
        weight.size() != static_cast<size_t>(1024) * 31 || bias.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_conv_depthwise invalid input sizes");
    }
    return cpu_depthwise_conv1d_causal(conv_glu, weight, bias, tokens, 1024, 31);
}

std::vector<float> run_w2v_bert_layer10_conv_depthwise_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& conv_glu,
                                                             uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.bias");
    return metal.depthwise_conv1d_causal_f32_resident(
        "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.weight.resident",
        weight,
        "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.bias.resident",
        bias,
        conv_glu,
        tokens,
        1024,
        31);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_conv_depthwise(const std::string& model_bundle_dir,
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
                               "w2v_layer10_conv_glu");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_conv_depthwise\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer10_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_conv_depthwise_sha256\": \"" << file_sha256_hex(output_depthwise_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer10_conv_glu_values\": " << input->size() << ",\n";
            std::cout << "  \"layer10_conv_depthwise_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_conv_depthwise_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_conv_depthwise\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_conv_depthwise\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 convolution residual, ffn2, and encoder layers 11-16 to hidden_state_17\"\n";
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
            depthwise = run_w2v_bert_layer10_conv_depthwise_metal(metal, model, conv_glu, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            depthwise = run_w2v_bert_layer10_conv_depthwise_cpu(model, conv_glu, tokens);
        }
        write_raw_f32(output_depthwise_f32, depthwise);
        print_report(true, backend, &conv_glu, &depthwise, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_conv_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& attention_norm,
                                                          const std::vector<float>& conv_depthwise,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_conv_residual invalid input sizes");
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

std::vector<float> run_w2v_bert_layer10_conv_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& attention_norm,
                                                            const std::vector<float>& conv_depthwise,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_conv_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.bias.resident",
        norm_b,
        conv_depthwise,
        tokens,
        1024,
        1e-5f);
    const auto activated = metal.silu_f32(normed);
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.weight.resident",
        pw2_w,
        "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.bias.resident",
        pw2_b,
        activated,
        tokens,
        1024,
        1024);
    return metal.add_f32(attention_norm, projected);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_conv_residual(const std::string& model_bundle_dir,
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
                               "w2v_layer10_attention_norm");
    append_raw_f32_count_issue(issues,
                               conv_depthwise_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer10_conv_depthwise");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_conv_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer10_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer10_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_conv_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (attention_norm && conv_depthwise && residual) {
            std::cout << "  \"layer10_attention_norm_values\": " << attention_norm->size() << ",\n";
            std::cout << "  \"layer10_conv_depthwise_values\": " << conv_depthwise->size() << ",\n";
            std::cout << "  \"layer10_conv_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_conv_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_conv_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_conv_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-10 ffn2 and encoder layers 11-16 to hidden_state_17\"\n";
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
            residual = run_w2v_bert_layer10_conv_residual_metal(metal, model, attention_norm, conv_depthwise, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer10_conv_residual_cpu(model, attention_norm, conv_depthwise, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &attention_norm, &conv_depthwise, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer10_ffn2_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& conv_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_ffn2_residual invalid input sizes");
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

std::vector<float> run_w2v_bert_layer10_ffn2_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& conv_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.10.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer10_ffn2_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.10.ffn2_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.10.ffn2_layer_norm.bias.resident",
        norm_b,
        conv_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.10.ffn2.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.10.ffn2.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(conv_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer10_ffn2_residual(const std::string& model_bundle_dir,
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
                               "w2v_layer10_conv_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer10_ffn2_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_layer10_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer10_ffn2_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer10_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer10_conv_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer10_ffn2_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer10_ffn2_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer10_ffn2_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer10_ffn2_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layers 11-16 to hidden_state_17\"\n";
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
            output = run_w2v_bert_layer10_ffn2_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer10_ffn2_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_ffn1_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& layer10_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.output_dense.bias");
    if (tokens == 0 || layer10_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_ffn1_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(layer10_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = layer10_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer11_ffn1_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& layer10_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn1.output_dense.bias");
    if (tokens == 0 || layer10_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_ffn1_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.11.ffn1_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.11.ffn1_layer_norm.bias.resident",
        norm_b,
        layer10_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.11.ffn1.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.11.ffn1.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(layer10_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_ffn1_residual(const std::string& model_bundle_dir,
                                                              const std::string& layer10_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer10_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer10_ffn2_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_ffn1_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer10_ffn2_residual_f32\": \"" << json_escape(layer10_residual_f32) << "\",\n";
        std::cout << "  \"output_layer11_ffn1_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_ffn1_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer10_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer10_ffn2_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer11_ffn1_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_ffn1_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_ffn1_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_ffn1_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 self-attention, convolution, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer10_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer11_ffn1_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer11_ffn1_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

W2VLayerQkv run_w2v_bert_layer11_qkv_cpu(const mit2::Bundle& model,
                                         const std::vector<float>& layer11_ffn1_residual,
                                         uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_v.weight");
    if (tokens == 0 || layer11_ffn1_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        q_w.size() != static_cast<size_t>(1024) * 1024 ||
        k_w.size() != static_cast<size_t>(1024) * 1024 ||
        v_w.size() != static_cast<size_t>(1024) * 1024) {
        throw std::invalid_argument("w2v_bert_layer11_qkv invalid input sizes");
    }
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_v.bias");
    return {
        cpu_linear_rows(q_w, q_b, layer11_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(k_w, k_b, layer11_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(v_w, v_b, layer11_ffn1_residual, tokens, 1024, 1024),
    };
}

W2VLayerQkv run_w2v_bert_layer11_qkv_metal(mit2::MetalContext& metal,
                                           const mit2::Bundle& model,
                                           const std::vector<float>& layer11_ffn1_residual,
                                           uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_v.weight");
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_v.bias");
    return {
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.11.self_attn.linear_q.weight.resident",
                                       q_w,
                                       "w2v_bert.encoder.layers.11.self_attn.linear_q.bias.resident",
                                       q_b,
                                       layer11_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.11.self_attn.linear_k.weight.resident",
                                       k_w,
                                       "w2v_bert.encoder.layers.11.self_attn.linear_k.bias.resident",
                                       k_b,
                                       layer11_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.11.self_attn.linear_v.weight.resident",
                                       v_w,
                                       "w2v_bert.encoder.layers.11.self_attn.linear_v.bias.resident",
                                       v_b,
                                       layer11_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
    };
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_qkv(const std::string& model_bundle_dir,
                                                   const std::string& layer11_ffn1_residual_f32,
                                                   uint32_t tokens,
                                                   const std::string& output_dir) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer11_ffn1_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer11_ffn1_residual");

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
    const auto output_q = (dir / "w2v_layer11_q.f32").string();
    const auto output_k = (dir / "w2v_layer11_k.f32").string();
    const auto output_v = (dir / "w2v_layer11_v.f32").string();
    const auto output_manifest = (dir / "w2v_layer11_qkv.manifest.json").string();

    auto print_failure = [&](const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_qkv\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_ffn1_residual_f32\": \"" << json_escape(layer11_ffn1_residual_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_qkv_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_qkv\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_qkv\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_failure(issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer11_ffn1_residual_f32);
        std::string execution_backend = "metal";
        W2VLayerQkv qkv;
        try {
            mit2::MetalContext metal;
            qkv = run_w2v_bert_layer11_qkv_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            qkv = run_w2v_bert_layer11_qkv_cpu(model, input, tokens);
        }

        std::filesystem::create_directories(dir);
        write_raw_f32(output_q, qkv.q);
        write_raw_f32(output_k, qkv.k);
        write_raw_f32(output_v, qkv.v);
        std::ostringstream manifest_json;
        manifest_json << "{\n";
        manifest_json << "  \"format\": \"mit2-w2v-layer11-qkv-sidecars\",\n";
        manifest_json << "  \"version\": 1,\n";
        manifest_json << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        manifest_json << "  \"w2v_layer11_ffn1_residual_f32\": \"" << json_escape(layer11_ffn1_residual_f32) << "\",\n";
        manifest_json << "  \"w2v_tokens\": " << tokens << ",\n";
        manifest_json << "  \"output_q_f32\": \"" << json_escape(output_q) << "\",\n";
        manifest_json << "  \"output_q_sha256\": \"" << file_sha256_hex(output_q) << "\",\n";
        manifest_json << "  \"output_k_f32\": \"" << json_escape(output_k) << "\",\n";
        manifest_json << "  \"output_k_sha256\": \"" << file_sha256_hex(output_k) << "\",\n";
        manifest_json << "  \"output_v_f32\": \"" << json_escape(output_v) << "\",\n";
        manifest_json << "  \"output_v_sha256\": \"" << file_sha256_hex(output_v) << "\",\n";
        manifest_json << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        manifest_json << "  \"ready_native_w2v_bert_layer11_qkv\": true,\n";
        manifest_json << "  \"ready_native_w2v_bert_semantic_features\": false\n";
        manifest_json << "}\n";
        write_text_file(output_manifest, manifest_json.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_qkv\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_ffn1_residual_f32\": \"" << json_escape(layer11_ffn1_residual_f32) << "\",\n";
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
        std::cout << "  \"layer11_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_ffn1_residual_values\": " << input.size() << ",\n";
        std::cout << "  \"q_values\": " << qkv.q.size() << ",\n";
        std::cout << "  \"k_values\": " << qkv.k.size() << ",\n";
        std::cout << "  \"v_values\": " << qkv.v.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_qkv_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_qkv\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_qkv\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        print_failure({e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_attention_context_cpu(const std::vector<float>& q,
                                                              const std::vector<float>& k,
                                                              const std::vector<float>& v,
                                                              const std::vector<uint32_t>& key_mask,
                                                              uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    if (tokens == 0 || q.size() != static_cast<size_t>(tokens) * heads * head_dim ||
        k.size() != q.size() || v.size() != q.size() || key_mask.size() != tokens) {
        throw std::invalid_argument("w2v_bert_layer11_attention_context invalid input sizes");
    }
    return cpu_cross_attention_heads_masked(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

std::vector<float> run_w2v_bert_layer11_attention_context_metal(mit2::MetalContext& metal,
                                                               const std::vector<float>& q,
                                                               const std::vector<float>& k,
                                                               const std::vector<float>& v,
                                                               const std::vector<uint32_t>& key_mask,
                                                               uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    return metal.cross_attention_heads_masked_f32(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_attention(const std::string& q_f32,
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
    append_raw_f32_count_issue(issues, q_f32, qkv_values, "w2v_layer11_q");
    append_raw_f32_count_issue(issues, k_f32, qkv_values, "w2v_layer11_k");
    append_raw_f32_count_issue(issues, v_f32, qkv_values, "w2v_layer11_v");
    append_raw_u32_count_issue(issues, attention_mask_u32, tokens, "w2v_attention_mask");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::optional<std::vector<float>>& output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_attention\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer11_q_f32\": \"" << json_escape(q_f32) << "\",\n";
        std::cout << "  \"w2v_layer11_k_f32\": \"" << json_escape(k_f32) << "\",\n";
        std::cout << "  \"w2v_layer11_v_f32\": \"" << json_escape(v_f32) << "\",\n";
        std::cout << "  \"w2v_attention_mask_u32\": \"" << json_escape(attention_mask_u32) << "\",\n";
        std::cout << "  \"output_layer11_context_f32\": \"" << json_escape(output_context_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_context_sha256\": \"" << file_sha256_hex(output_context_f32) << "\",\n";
        }
        std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_context_shape\": \"[1," << tokens << ",1024]\",\n";
        if (output) {
            std::cout << "  \"layer11_context_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer11_attention_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_attention_context\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_attention_context\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 attention output projection/residual/norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "not_run", std::nullopt, issues);
        return false;
    }

    try {
        const auto q = read_raw_f32(q_f32);
        const auto k = read_raw_f32(k_f32);
        const auto v = read_raw_f32(v_f32);
        const auto mask = read_raw_u32(attention_mask_u32);
        std::string execution_backend = "cpu_distance_emb";
        const std::vector<float> context = cpu_w2v_bert_cross_attention_with_distance(q, k, v, mask, tokens, dist_emb, 64, 8);
        write_raw_f32(output_context_f32, context);
        print_report(true, execution_backend, context, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "not_run", std::nullopt, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_attention_project_cpu(const mit2::Bundle& model,
                                                              const std::vector<float>& context,
                                                              uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.11.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_out.bias");
    }
    if (tokens == 0 || context.size() != static_cast<size_t>(tokens) * 1024u ||
        out_w.size() != static_cast<size_t>(1024) * 1024 || out_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_attention_project invalid input sizes");
    }
    return cpu_linear_rows(out_w, out_b, context, tokens, 1024, 1024);
}

std::vector<float> run_w2v_bert_layer11_attention_project_metal(mit2::MetalContext& metal,
                                                                const mit2::Bundle& model,
                                                                const std::vector<float>& context,
                                                                uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.11.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn.linear_out.bias");
    }
    return metal.linear_rows_f32_resident("w2v_bert.encoder.layers.11.self_attn.linear_out.weight.resident",
                                          out_w,
                                          "w2v_bert.encoder.layers.11.self_attn.linear_out.bias.resident_or_zero",
                                          out_b,
                                          context,
                                          tokens,
                                          1024,
                                          1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_attention_project(const std::string& model_bundle_dir,
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
                               "w2v_layer11_attention_context");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_attention_project\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_layer11_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_attention_sha256\": \"" << file_sha256_hex(output_attention_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer11_context_values\": " << input->size() << ",\n";
            std::cout << "  \"layer11_attention_projection_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_attention_project_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_attention_projection\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_attention_projection\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 attention residual/norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
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
            projected = run_w2v_bert_layer11_attention_project_metal(metal, model, context, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            projected = run_w2v_bert_layer11_attention_project_cpu(model, context, tokens);
        }
        write_raw_f32(output_attention_f32, projected);
        print_report(true, backend, &context, &projected, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_attention_residual_cpu(const std::vector<float>& ffn1_residual,
                                                               const std::vector<float>& attention_projection,
                                                               uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer11_attention_residual invalid input sizes");
    }
    std::vector<float> residual(expected);
    for (size_t i = 0; i < expected; ++i) {
        residual[i] = ffn1_residual[i] + attention_projection[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer11_attention_residual_metal(mit2::MetalContext& metal,
                                                                 const std::vector<float>& ffn1_residual,
                                                                 const std::vector<float>& attention_projection,
                                                                 uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer11_attention_residual invalid input sizes");
    }
    return metal.add_f32(ffn1_residual, attention_projection);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_attention_residual(const std::string& ffn1_residual_f32,
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
                               "w2v_layer11_ffn1_residual");
    append_raw_f32_count_issue(issues,
                               attention_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer11_attention_projection");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* ffn1_residual,
                            const std::vector<float>* attention_projection,
                            const std::vector<float>* residual,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_attention_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer11_ffn1_residual_f32\": \"" << json_escape(ffn1_residual_f32) << "\",\n";
        std::cout << "  \"w2v_layer11_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_layer11_attention_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_attention_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (ffn1_residual && attention_projection && residual) {
            std::cout << "  \"layer11_ffn1_residual_values\": " << ffn1_residual->size() << ",\n";
            std::cout << "  \"layer11_attention_projection_values\": " << attention_projection->size() << ",\n";
            std::cout << "  \"layer11_attention_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer11_attention_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_attention_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_attention_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 attention norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
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
            residual = run_w2v_bert_layer11_attention_residual_metal(metal,
                                                                     ffn1_residual,
                                                                     attention_projection,
                                                                     tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer11_attention_residual_cpu(ffn1_residual, attention_projection, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &ffn1_residual, &attention_projection, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_attention_norm_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& attention_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn_layer_norm.bias");
    if (tokens == 0 || attention_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_attention_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer11_attention_norm_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& attention_residual,
                                                             uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.self_attn_layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.11.self_attn_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.11.self_attn_layer_norm.bias.resident",
                                             norm_b,
                                             attention_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_attention_norm(const std::string& model_bundle_dir,
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
                               "w2v_layer11_attention_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_attention_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_layer11_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_attention_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer11_attention_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer11_attention_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_attention_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_attention_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_attention_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
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
            normed = run_w2v_bert_layer11_attention_norm_metal(metal, model, attention_residual, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer11_attention_norm_cpu(model, attention_residual, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_residual, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_conv_norm_cpu(const mit2::Bundle& model,
                                                      const std::vector<float>& attention_norm,
                                                      uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.layer_norm.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_conv_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_norm, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer11_conv_norm_metal(mit2::MetalContext& metal,
                                                        const mit2::Bundle& model,
                                                        const std::vector<float>& attention_norm,
                                                        uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.11.conv_module.layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.11.conv_module.layer_norm.bias.resident",
                                             norm_b,
                                             attention_norm,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_conv_norm(const std::string& model_bundle_dir,
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
                               "w2v_layer11_attention_norm");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_conv_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer11_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_conv_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer11_attention_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer11_conv_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_conv_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_conv_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_conv_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 convolution GLU, depthwise/residual, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
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
            normed = run_w2v_bert_layer11_conv_norm_metal(metal, model, attention_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer11_conv_norm_cpu(model, attention_norm, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_norm, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_conv_glu_cpu(const mit2::Bundle& model,
                                                     const std::vector<float>& conv_norm,
                                                     uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer11_conv_glu invalid input sizes");
    }
    const auto projected = cpu_linear_rows(pw1_w, pw1_b, conv_norm, tokens, 2048, 1024);
    return cpu_glu_split(projected, tokens, 1024);
}

std::vector<float> run_w2v_bert_layer11_conv_glu_metal(mit2::MetalContext& metal,
                                                       const mit2::Bundle& model,
                                                       const std::vector<float>& conv_norm,
                                                       uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer11_conv_glu invalid input sizes");
    }
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.weight.resident",
        pw1_w,
        "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.bias.resident",
        pw1_b,
        conv_norm,
        tokens,
        2048,
        1024);
    return metal.glu_split_f32(projected, tokens, 1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_conv_glu(const std::string& model_bundle_dir,
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
                               "w2v_layer11_conv_norm");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_conv_glu\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer11_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_conv_glu_sha256\": \"" << file_sha256_hex(output_glu_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer11_conv_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer11_conv_glu_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_conv_glu_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_conv_glu\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_conv_glu\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 depthwise convolution, convolution residual, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
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
            glu = run_w2v_bert_layer11_conv_glu_metal(metal, model, conv_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            glu = run_w2v_bert_layer11_conv_glu_cpu(model, conv_norm, tokens);
        }
        write_raw_f32(output_glu_f32, glu);
        print_report(true, backend, &conv_norm, &glu, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_conv_depthwise_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& conv_glu,
                                                           uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.bias");
    if (tokens == 0 || conv_glu.size() != static_cast<size_t>(tokens) * 1024u ||
        weight.size() != static_cast<size_t>(1024) * 31 || bias.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_conv_depthwise invalid input sizes");
    }
    return cpu_depthwise_conv1d_causal(conv_glu, weight, bias, tokens, 1024, 31);
}

std::vector<float> run_w2v_bert_layer11_conv_depthwise_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& conv_glu,
                                                             uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.bias");
    return metal.depthwise_conv1d_causal_f32_resident(
        "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.weight.resident",
        weight,
        "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.bias.resident",
        bias,
        conv_glu,
        tokens,
        1024,
        31);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_conv_depthwise(const std::string& model_bundle_dir,
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
                               "w2v_layer11_conv_glu");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_conv_depthwise\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer11_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_conv_depthwise_sha256\": \"" << file_sha256_hex(output_depthwise_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer11_conv_glu_values\": " << input->size() << ",\n";
            std::cout << "  \"layer11_conv_depthwise_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_conv_depthwise_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_conv_depthwise\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_conv_depthwise\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 convolution residual, ffn2, and encoder layers 12-16 to hidden_state_17\"\n";
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
            depthwise = run_w2v_bert_layer11_conv_depthwise_metal(metal, model, conv_glu, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            depthwise = run_w2v_bert_layer11_conv_depthwise_cpu(model, conv_glu, tokens);
        }
        write_raw_f32(output_depthwise_f32, depthwise);
        print_report(true, backend, &conv_glu, &depthwise, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_conv_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& attention_norm,
                                                          const std::vector<float>& conv_depthwise,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_conv_residual invalid input sizes");
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

std::vector<float> run_w2v_bert_layer11_conv_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& attention_norm,
                                                            const std::vector<float>& conv_depthwise,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_conv_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.bias.resident",
        norm_b,
        conv_depthwise,
        tokens,
        1024,
        1e-5f);
    const auto activated = metal.silu_f32(normed);
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.weight.resident",
        pw2_w,
        "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.bias.resident",
        pw2_b,
        activated,
        tokens,
        1024,
        1024);
    return metal.add_f32(attention_norm, projected);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_conv_residual(const std::string& model_bundle_dir,
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
                               "w2v_layer11_attention_norm");
    append_raw_f32_count_issue(issues,
                               conv_depthwise_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer11_conv_depthwise");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_conv_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer11_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer11_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_conv_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (attention_norm && conv_depthwise && residual) {
            std::cout << "  \"layer11_attention_norm_values\": " << attention_norm->size() << ",\n";
            std::cout << "  \"layer11_conv_depthwise_values\": " << conv_depthwise->size() << ",\n";
            std::cout << "  \"layer11_conv_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_conv_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_conv_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_conv_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-11 ffn2 and encoder layers 12-16 to hidden_state_17\"\n";
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
            residual = run_w2v_bert_layer11_conv_residual_metal(metal, model, attention_norm, conv_depthwise, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer11_conv_residual_cpu(model, attention_norm, conv_depthwise, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &attention_norm, &conv_depthwise, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer11_ffn2_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& conv_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_ffn2_residual invalid input sizes");
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

std::vector<float> run_w2v_bert_layer11_ffn2_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& conv_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.11.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer11_ffn2_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.11.ffn2_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.11.ffn2_layer_norm.bias.resident",
        norm_b,
        conv_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.11.ffn2.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.11.ffn2.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(conv_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer11_ffn2_residual(const std::string& model_bundle_dir,
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
                               "w2v_layer11_conv_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer11_ffn2_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_layer11_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer11_ffn2_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer11_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer11_conv_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer11_ffn2_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer11_ffn2_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer11_ffn2_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer11_ffn2_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layers 12-16 to hidden_state_17\"\n";
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
            output = run_w2v_bert_layer11_ffn2_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer11_ffn2_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_ffn1_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& layer11_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.output_dense.bias");
    if (tokens == 0 || layer11_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_ffn1_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(layer11_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = layer11_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer12_ffn1_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& layer11_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn1.output_dense.bias");
    if (tokens == 0 || layer11_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_ffn1_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.12.ffn1_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.12.ffn1_layer_norm.bias.resident",
        norm_b,
        layer11_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.12.ffn1.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.12.ffn1.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(layer11_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_ffn1_residual(const std::string& model_bundle_dir,
                                                              const std::string& layer11_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer11_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer11_ffn2_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_ffn1_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer11_ffn2_residual_f32\": \"" << json_escape(layer11_residual_f32) << "\",\n";
        std::cout << "  \"output_layer12_ffn1_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_ffn1_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer11_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer11_ffn2_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer12_ffn1_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_ffn1_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_ffn1_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_ffn1_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 self-attention, convolution, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer11_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer12_ffn1_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer12_ffn1_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

W2VLayerQkv run_w2v_bert_layer12_qkv_cpu(const mit2::Bundle& model,
                                         const std::vector<float>& layer12_ffn1_residual,
                                         uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_v.weight");
    if (tokens == 0 || layer12_ffn1_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        q_w.size() != static_cast<size_t>(1024) * 1024 ||
        k_w.size() != static_cast<size_t>(1024) * 1024 ||
        v_w.size() != static_cast<size_t>(1024) * 1024) {
        throw std::invalid_argument("w2v_bert_layer12_qkv invalid input sizes");
    }
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_v.bias");
    return {
        cpu_linear_rows(q_w, q_b, layer12_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(k_w, k_b, layer12_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(v_w, v_b, layer12_ffn1_residual, tokens, 1024, 1024),
    };
}

W2VLayerQkv run_w2v_bert_layer12_qkv_metal(mit2::MetalContext& metal,
                                           const mit2::Bundle& model,
                                           const std::vector<float>& layer12_ffn1_residual,
                                           uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_v.weight");
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_v.bias");
    return {
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.12.self_attn.linear_q.weight.resident",
                                       q_w,
                                       "w2v_bert.encoder.layers.12.self_attn.linear_q.bias.resident",
                                       q_b,
                                       layer12_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.12.self_attn.linear_k.weight.resident",
                                       k_w,
                                       "w2v_bert.encoder.layers.12.self_attn.linear_k.bias.resident",
                                       k_b,
                                       layer12_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.12.self_attn.linear_v.weight.resident",
                                       v_w,
                                       "w2v_bert.encoder.layers.12.self_attn.linear_v.bias.resident",
                                       v_b,
                                       layer12_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
    };
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_qkv(const std::string& model_bundle_dir,
                                                   const std::string& layer12_ffn1_residual_f32,
                                                   uint32_t tokens,
                                                   const std::string& output_dir) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer12_ffn1_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer12_ffn1_residual");

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
    const auto output_q = (dir / "w2v_layer12_q.f32").string();
    const auto output_k = (dir / "w2v_layer12_k.f32").string();
    const auto output_v = (dir / "w2v_layer12_v.f32").string();
    const auto output_manifest = (dir / "w2v_layer12_qkv.manifest.json").string();

    auto print_failure = [&](const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_qkv\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_ffn1_residual_f32\": \"" << json_escape(layer12_ffn1_residual_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_qkv_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_qkv\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_qkv\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_failure(issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer12_ffn1_residual_f32);
        std::string execution_backend = "metal";
        W2VLayerQkv qkv;
        try {
            mit2::MetalContext metal;
            qkv = run_w2v_bert_layer12_qkv_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            qkv = run_w2v_bert_layer12_qkv_cpu(model, input, tokens);
        }

        std::filesystem::create_directories(dir);
        write_raw_f32(output_q, qkv.q);
        write_raw_f32(output_k, qkv.k);
        write_raw_f32(output_v, qkv.v);
        std::ostringstream manifest_json;
        manifest_json << "{\n";
        manifest_json << "  \"format\": \"mit2-w2v-layer12-qkv-sidecars\",\n";
        manifest_json << "  \"version\": 1,\n";
        manifest_json << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        manifest_json << "  \"w2v_layer12_ffn1_residual_f32\": \"" << json_escape(layer12_ffn1_residual_f32) << "\",\n";
        manifest_json << "  \"w2v_tokens\": " << tokens << ",\n";
        manifest_json << "  \"output_q_f32\": \"" << json_escape(output_q) << "\",\n";
        manifest_json << "  \"output_q_sha256\": \"" << file_sha256_hex(output_q) << "\",\n";
        manifest_json << "  \"output_k_f32\": \"" << json_escape(output_k) << "\",\n";
        manifest_json << "  \"output_k_sha256\": \"" << file_sha256_hex(output_k) << "\",\n";
        manifest_json << "  \"output_v_f32\": \"" << json_escape(output_v) << "\",\n";
        manifest_json << "  \"output_v_sha256\": \"" << file_sha256_hex(output_v) << "\",\n";
        manifest_json << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        manifest_json << "  \"ready_native_w2v_bert_layer12_qkv\": true,\n";
        manifest_json << "  \"ready_native_w2v_bert_semantic_features\": false\n";
        manifest_json << "}\n";
        write_text_file(output_manifest, manifest_json.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_qkv\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_ffn1_residual_f32\": \"" << json_escape(layer12_ffn1_residual_f32) << "\",\n";
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
        std::cout << "  \"layer12_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_ffn1_residual_values\": " << input.size() << ",\n";
        std::cout << "  \"q_values\": " << qkv.q.size() << ",\n";
        std::cout << "  \"k_values\": " << qkv.k.size() << ",\n";
        std::cout << "  \"v_values\": " << qkv.v.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_qkv_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_qkv\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_qkv\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        print_failure({e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_attention_context_cpu(const std::vector<float>& q,
                                                              const std::vector<float>& k,
                                                              const std::vector<float>& v,
                                                              const std::vector<uint32_t>& key_mask,
                                                              uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    if (tokens == 0 || q.size() != static_cast<size_t>(tokens) * heads * head_dim ||
        k.size() != q.size() || v.size() != q.size() || key_mask.size() != tokens) {
        throw std::invalid_argument("w2v_bert_layer12_attention_context invalid input sizes");
    }
    return cpu_cross_attention_heads_masked(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

std::vector<float> run_w2v_bert_layer12_attention_context_metal(mit2::MetalContext& metal,
                                                                const std::vector<float>& q,
                                                                const std::vector<float>& k,
                                                                const std::vector<float>& v,
                                                                const std::vector<uint32_t>& key_mask,
                                                                uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    return metal.cross_attention_heads_masked_f32(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_attention(const std::string& q_f32,
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
    append_raw_f32_count_issue(issues, q_f32, qkv_values, "w2v_layer12_q");
    append_raw_f32_count_issue(issues, k_f32, qkv_values, "w2v_layer12_k");
    append_raw_f32_count_issue(issues, v_f32, qkv_values, "w2v_layer12_v");
    append_raw_u32_count_issue(issues, attention_mask_u32, tokens, "w2v_attention_mask");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::optional<std::vector<float>>& output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_attention\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer12_q_f32\": \"" << json_escape(q_f32) << "\",\n";
        std::cout << "  \"w2v_layer12_k_f32\": \"" << json_escape(k_f32) << "\",\n";
        std::cout << "  \"w2v_layer12_v_f32\": \"" << json_escape(v_f32) << "\",\n";
        std::cout << "  \"w2v_attention_mask_u32\": \"" << json_escape(attention_mask_u32) << "\",\n";
        std::cout << "  \"output_layer12_context_f32\": \"" << json_escape(output_context_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_context_sha256\": \"" << file_sha256_hex(output_context_f32) << "\",\n";
        }
        std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_context_shape\": \"[1," << tokens << ",1024]\",\n";
        if (output) {
            std::cout << "  \"layer12_context_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer12_attention_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_attention_context\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_attention_context\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 attention output projection/residual/norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "not_run", std::nullopt, issues);
        return false;
    }

    try {
        const auto q = read_raw_f32(q_f32);
        const auto k = read_raw_f32(k_f32);
        const auto v = read_raw_f32(v_f32);
        const auto mask = read_raw_u32(attention_mask_u32);
        std::string execution_backend = "cpu_distance_emb";
        const std::vector<float> context = cpu_w2v_bert_cross_attention_with_distance(q, k, v, mask, tokens, dist_emb, 64, 8);
        write_raw_f32(output_context_f32, context);
        print_report(true, execution_backend, context, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "not_run", std::nullopt, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_attention_project_cpu(const mit2::Bundle& model,
                                                              const std::vector<float>& context,
                                                              uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.12.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_out.bias");
    }
    if (tokens == 0 || context.size() != static_cast<size_t>(tokens) * 1024u ||
        out_w.size() != static_cast<size_t>(1024) * 1024 || out_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_attention_project invalid input sizes");
    }
    return cpu_linear_rows(out_w, out_b, context, tokens, 1024, 1024);
}

std::vector<float> run_w2v_bert_layer12_attention_project_metal(mit2::MetalContext& metal,
                                                                const mit2::Bundle& model,
                                                                const std::vector<float>& context,
                                                                uint32_t tokens) {
    const auto out_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_out.weight");
    std::vector<float> out_b(1024, 0.0f);
    if (model.find("w2v_bert.encoder.layers.12.self_attn.linear_out.bias")) {
        out_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn.linear_out.bias");
    }
    return metal.linear_rows_f32_resident("w2v_bert.encoder.layers.12.self_attn.linear_out.weight.resident",
                                          out_w,
                                          "w2v_bert.encoder.layers.12.self_attn.linear_out.bias.resident_or_zero",
                                          out_b,
                                          context,
                                          tokens,
                                          1024,
                                          1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_attention_project(const std::string& model_bundle_dir,
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
                               "w2v_layer12_attention_context");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_attention_project\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_context_f32\": \"" << json_escape(context_f32) << "\",\n";
        std::cout << "  \"output_layer12_attention_f32\": \"" << json_escape(output_attention_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_attention_sha256\": \"" << file_sha256_hex(output_attention_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer12_context_values\": " << input->size() << ",\n";
            std::cout << "  \"layer12_attention_projection_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_attention_project_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_attention_projection\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_attention_projection\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 attention residual/norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
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
            projected = run_w2v_bert_layer12_attention_project_metal(metal, model, context, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            projected = run_w2v_bert_layer12_attention_project_cpu(model, context, tokens);
        }
        write_raw_f32(output_attention_f32, projected);
        print_report(true, backend, &context, &projected, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_attention_residual_cpu(const std::vector<float>& ffn1_residual,
                                                               const std::vector<float>& attention_projection,
                                                               uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer12_attention_residual invalid input sizes");
    }
    std::vector<float> residual(expected);
    for (size_t i = 0; i < expected; ++i) {
        residual[i] = ffn1_residual[i] + attention_projection[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer12_attention_residual_metal(mit2::MetalContext& metal,
                                                                 const std::vector<float>& ffn1_residual,
                                                                 const std::vector<float>& attention_projection,
                                                                 uint32_t tokens) {
    const size_t expected = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 || ffn1_residual.size() != expected || attention_projection.size() != expected) {
        throw std::invalid_argument("w2v_bert_layer12_attention_residual invalid input sizes");
    }
    return metal.add_f32(ffn1_residual, attention_projection);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_attention_residual(const std::string& ffn1_residual_f32,
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
                               "w2v_layer12_ffn1_residual");
    append_raw_f32_count_issue(issues,
                               attention_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer12_attention_projection");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::vector<float>* ffn1_residual,
                            const std::vector<float>* attention_projection,
                            const std::vector<float>* residual,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_attention_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer12_ffn1_residual_f32\": \"" << json_escape(ffn1_residual_f32) << "\",\n";
        std::cout << "  \"w2v_layer12_attention_f32\": \"" << json_escape(attention_f32) << "\",\n";
        std::cout << "  \"output_layer12_attention_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_attention_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_attention_projection_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (ffn1_residual && attention_projection && residual) {
            std::cout << "  \"layer12_ffn1_residual_values\": " << ffn1_residual->size() << ",\n";
            std::cout << "  \"layer12_attention_projection_values\": " << attention_projection->size() << ",\n";
            std::cout << "  \"layer12_attention_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer12_attention_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_attention_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_attention_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 attention norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
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
            residual = run_w2v_bert_layer12_attention_residual_metal(metal,
                                                                     ffn1_residual,
                                                                     attention_projection,
                                                                     tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer12_attention_residual_cpu(ffn1_residual, attention_projection, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &ffn1_residual, &attention_projection, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_attention_norm_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& attention_residual,
                                                           uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn_layer_norm.bias");
    if (tokens == 0 || attention_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_attention_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer12_attention_norm_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& attention_residual,
                                                             uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.self_attn_layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.12.self_attn_layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.12.self_attn_layer_norm.bias.resident",
                                             norm_b,
                                             attention_residual,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_attention_norm(const std::string& model_bundle_dir,
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
                               "w2v_layer12_attention_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_attention_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_attention_residual_f32\": \"" << json_escape(attention_residual_f32) << "\",\n";
        std::cout << "  \"output_layer12_attention_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_attention_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_attention_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer12_attention_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer12_attention_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_attention_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_attention_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_attention_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
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
            normed = run_w2v_bert_layer12_attention_norm_metal(metal, model, attention_residual, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer12_attention_norm_cpu(model, attention_residual, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_residual, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_conv_norm_cpu(const mit2::Bundle& model,
                                                      const std::vector<float>& attention_norm,
                                                      uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.layer_norm.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_conv_norm invalid input sizes");
    }
    return cpu_layer_norm_rows(attention_norm, norm_w, norm_b, tokens, 1024, 1e-5f);
}

std::vector<float> run_w2v_bert_layer12_conv_norm_metal(mit2::MetalContext& metal,
                                                        const mit2::Bundle& model,
                                                        const std::vector<float>& attention_norm,
                                                        uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.layer_norm.bias");
    return metal.layernorm_rows_f32_resident("w2v_bert.encoder.layers.12.conv_module.layer_norm.weight.resident",
                                             norm_w,
                                             "w2v_bert.encoder.layers.12.conv_module.layer_norm.bias.resident",
                                             norm_b,
                                             attention_norm,
                                             tokens,
                                             1024,
                                             1e-5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_conv_norm(const std::string& model_bundle_dir,
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
                               "w2v_layer12_attention_norm");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_conv_norm\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"output_layer12_conv_norm_f32\": \"" << json_escape(output_norm_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_conv_norm_sha256\": \"" << file_sha256_hex(output_norm_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer12_attention_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer12_conv_norm_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_conv_norm_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_conv_norm\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_conv_norm\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 convolution GLU, depthwise/residual, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
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
            normed = run_w2v_bert_layer12_conv_norm_metal(metal, model, attention_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            normed = run_w2v_bert_layer12_conv_norm_cpu(model, attention_norm, tokens);
        }
        write_raw_f32(output_norm_f32, normed);
        print_report(true, backend, &attention_norm, &normed, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_conv_glu_cpu(const mit2::Bundle& model,
                                                     const std::vector<float>& conv_norm,
                                                     uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer12_conv_glu invalid input sizes");
    }
    const auto projected = cpu_linear_rows(pw1_w, pw1_b, conv_norm, tokens, 2048, 1024);
    return cpu_glu_split(projected, tokens, 1024);
}

std::vector<float> run_w2v_bert_layer12_conv_glu_metal(mit2::MetalContext& metal,
                                                       const mit2::Bundle& model,
                                                       const std::vector<float>& conv_norm,
                                                       uint32_t tokens) {
    const auto pw1_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.weight");
    const auto pw1_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.bias");
    if (tokens == 0 || conv_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        pw1_w.size() != static_cast<size_t>(2048) * 1024 || pw1_b.size() != 2048) {
        throw std::invalid_argument("w2v_bert_layer12_conv_glu invalid input sizes");
    }
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.weight.resident",
        pw1_w,
        "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.bias.resident",
        pw1_b,
        conv_norm,
        tokens,
        2048,
        1024);
    return metal.glu_split_f32(projected, tokens, 1024);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_conv_glu(const std::string& model_bundle_dir,
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
                               "w2v_layer12_conv_norm");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_conv_glu\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_conv_norm_f32\": \"" << json_escape(conv_norm_f32) << "\",\n";
        std::cout << "  \"output_layer12_conv_glu_f32\": \"" << json_escape(output_glu_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_conv_glu_sha256\": \"" << file_sha256_hex(output_glu_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_conv_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer12_conv_norm_values\": " << input->size() << ",\n";
            std::cout << "  \"layer12_conv_glu_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_conv_glu_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_conv_glu\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_conv_glu\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 depthwise convolution, convolution residual, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
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
            glu = run_w2v_bert_layer12_conv_glu_metal(metal, model, conv_norm, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            glu = run_w2v_bert_layer12_conv_glu_cpu(model, conv_norm, tokens);
        }
        write_raw_f32(output_glu_f32, glu);
        print_report(true, backend, &conv_norm, &glu, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_conv_depthwise_cpu(const mit2::Bundle& model,
                                                           const std::vector<float>& conv_glu,
                                                           uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.bias");
    if (tokens == 0 || conv_glu.size() != static_cast<size_t>(tokens) * 1024u ||
        weight.size() != static_cast<size_t>(1024) * 31 || bias.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_conv_depthwise invalid input sizes");
    }
    return cpu_depthwise_conv1d_causal(conv_glu, weight, bias, tokens, 1024, 31);
}

std::vector<float> run_w2v_bert_layer12_conv_depthwise_metal(mit2::MetalContext& metal,
                                                             const mit2::Bundle& model,
                                                             const std::vector<float>& conv_glu,
                                                             uint32_t tokens) {
    const auto weight = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.weight");
    const auto bias = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.bias");
    return metal.depthwise_conv1d_causal_f32_resident(
        "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.weight.resident",
        weight,
        "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.bias.resident",
        bias,
        conv_glu,
        tokens,
        1024,
        31);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_conv_depthwise(const std::string& model_bundle_dir,
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
                               "w2v_layer12_conv_glu");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_conv_depthwise\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_conv_glu_f32\": \"" << json_escape(conv_glu_f32) << "\",\n";
        std::cout << "  \"output_layer12_conv_depthwise_f32\": \"" << json_escape(output_depthwise_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_conv_depthwise_sha256\": \"" << file_sha256_hex(output_depthwise_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_conv_glu_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer12_conv_glu_values\": " << input->size() << ",\n";
            std::cout << "  \"layer12_conv_depthwise_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_conv_depthwise_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_conv_depthwise\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_conv_depthwise\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 convolution residual, ffn2, and encoder layers 13-16 to hidden_state_17\"\n";
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
            depthwise = run_w2v_bert_layer12_conv_depthwise_metal(metal, model, conv_glu, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            depthwise = run_w2v_bert_layer12_conv_depthwise_cpu(model, conv_glu, tokens);
        }
        write_raw_f32(output_depthwise_f32, depthwise);
        print_report(true, backend, &conv_glu, &depthwise, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_conv_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& attention_norm,
                                                          const std::vector<float>& conv_depthwise,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_conv_residual invalid input sizes");
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

std::vector<float> run_w2v_bert_layer12_conv_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& attention_norm,
                                                            const std::vector<float>& conv_depthwise,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.bias");
    const auto pw2_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.weight");
    const auto pw2_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.bias");
    if (tokens == 0 || attention_norm.size() != static_cast<size_t>(tokens) * 1024u ||
        conv_depthwise.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        pw2_w.size() != static_cast<size_t>(1024) * 1024 || pw2_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_conv_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.bias.resident",
        norm_b,
        conv_depthwise,
        tokens,
        1024,
        1e-5f);
    const auto activated = metal.silu_f32(normed);
    const auto projected = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.weight.resident",
        pw2_w,
        "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.bias.resident",
        pw2_b,
        activated,
        tokens,
        1024,
        1024);
    return metal.add_f32(attention_norm, projected);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_conv_residual(const std::string& model_bundle_dir,
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
                               "w2v_layer12_attention_norm");
    append_raw_f32_count_issue(issues,
                               conv_depthwise_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer12_conv_depthwise");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_conv_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_attention_norm_f32\": \"" << json_escape(attention_norm_f32) << "\",\n";
        std::cout << "  \"w2v_layer12_conv_depthwise_f32\": \"" << json_escape(conv_depthwise_f32) << "\",\n";
        std::cout << "  \"output_layer12_conv_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_conv_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_attention_norm_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_conv_depthwise_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (attention_norm && conv_depthwise && residual) {
            std::cout << "  \"layer12_attention_norm_values\": " << attention_norm->size() << ",\n";
            std::cout << "  \"layer12_conv_depthwise_values\": " << conv_depthwise->size() << ",\n";
            std::cout << "  \"layer12_conv_residual_values\": " << residual->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_conv_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_conv_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_conv_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-12 ffn2 and encoder layers 13-16 to hidden_state_17\"\n";
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
            residual = run_w2v_bert_layer12_conv_residual_metal(metal, model, attention_norm, conv_depthwise, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            residual = run_w2v_bert_layer12_conv_residual_cpu(model, attention_norm, conv_depthwise, tokens);
        }
        write_raw_f32(output_residual_f32, residual);
        print_report(true, backend, &attention_norm, &conv_depthwise, &residual, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer12_ffn2_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& conv_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_ffn2_residual invalid input sizes");
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

std::vector<float> run_w2v_bert_layer12_ffn2_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& conv_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.12.ffn2.output_dense.bias");
    if (tokens == 0 || conv_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer12_ffn2_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.12.ffn2_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.12.ffn2_layer_norm.bias.resident",
        norm_b,
        conv_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.12.ffn2.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.12.ffn2.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(conv_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer12_ffn2_residual(const std::string& model_bundle_dir,
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
                               "w2v_layer12_conv_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer12_ffn2_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_conv_residual_f32\": \"" << json_escape(conv_residual_f32) << "\",\n";
        std::cout << "  \"output_layer12_ffn2_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer12_ffn2_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_conv_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer12_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer12_conv_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer12_ffn2_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer12_ffn2_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer12_ffn2_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer12_ffn2_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT encoder layers 13-16 to hidden_state_17\"\n";
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
            output = run_w2v_bert_layer12_ffn2_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer12_ffn2_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_ffn1_residual_cpu(const mit2::Bundle& model,
                                                          const std::vector<float>& layer12_residual,
                                                          uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.output_dense.bias");
    if (tokens == 0 || layer12_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_ffn1_residual invalid input sizes");
    }
    const auto normed = cpu_layer_norm_rows(layer12_residual, norm_w, norm_b, tokens, 1024, 1e-5f);
    const auto intermediate = cpu_linear_rows(intermediate_w, intermediate_b, normed, tokens, 4096, 1024);
    const auto activated = cpu_silu(intermediate);
    const auto output = cpu_linear_rows(output_w, output_b, activated, tokens, 1024, 4096);
    std::vector<float> residual(output.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = layer12_residual[i] + 0.5f * output[i];
    }
    return residual;
}

std::vector<float> run_w2v_bert_layer13_ffn1_residual_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& model,
                                                            const std::vector<float>& layer12_residual,
                                                            uint32_t tokens) {
    const auto norm_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1_layer_norm.weight");
    const auto norm_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1_layer_norm.bias");
    const auto intermediate_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.weight");
    const auto intermediate_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.bias");
    const auto output_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.output_dense.weight");
    const auto output_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.ffn1.output_dense.bias");
    if (tokens == 0 || layer12_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        norm_w.size() != 1024 || norm_b.size() != 1024 ||
        intermediate_w.size() != static_cast<size_t>(4096) * 1024 || intermediate_b.size() != 4096 ||
        output_w.size() != static_cast<size_t>(1024) * 4096 || output_b.size() != 1024) {
        throw std::invalid_argument("w2v_bert_layer13_ffn1_residual invalid input sizes");
    }
    const auto normed = metal.layernorm_rows_f32_resident(
        "w2v_bert.encoder.layers.13.ffn1_layer_norm.weight.resident",
        norm_w,
        "w2v_bert.encoder.layers.13.ffn1_layer_norm.bias.resident",
        norm_b,
        layer12_residual,
        tokens,
        1024,
        1e-5f);
    const auto intermediate = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.weight.resident",
        intermediate_w,
        "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.bias.resident",
        intermediate_b,
        normed,
        tokens,
        4096,
        1024);
    const auto activated = metal.silu_f32(intermediate);
    const auto output = metal.linear_rows_f32_resident(
        "w2v_bert.encoder.layers.13.ffn1.output_dense.weight.resident",
        output_w,
        "w2v_bert.encoder.layers.13.ffn1.output_dense.bias.resident",
        output_b,
        activated,
        tokens,
        1024,
        4096);
    return metal.add_scaled_f32(layer12_residual, output, 0.5f);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_ffn1_residual(const std::string& model_bundle_dir,
                                                              const std::string& layer12_residual_f32,
                                                              uint32_t tokens,
                                                              const std::string& output_residual_f32) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer12_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer12_ffn2_residual");

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
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_ffn1_residual\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer12_ffn2_residual_f32\": \"" << json_escape(layer12_residual_f32) << "\",\n";
        std::cout << "  \"output_layer13_ffn1_residual_f32\": \"" << json_escape(output_residual_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_ffn1_residual_sha256\": \"" << file_sha256_hex(output_residual_f32) << "\",\n";
            std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        }
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer12_ffn2_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        if (input && output) {
            std::cout << "  \"layer12_ffn2_residual_values\": " << input->size() << ",\n";
            std::cout << "  \"layer13_ffn1_residual_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"has_w2v_bert_model_contract\": " << (ok ? "true" : (w2v_ok ? "true" : "false")) << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_ffn1_residual_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_ffn1_residual\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_ffn1_residual\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 self-attention, convolution, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };

    if (!issues.empty()) {
        print_report(false, "", nullptr, nullptr, issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer12_residual_f32);
        std::string backend = "metal";
        std::vector<float> output;
        try {
            mit2::MetalContext metal;
            output = run_w2v_bert_layer13_ffn1_residual_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            backend = "cpu_fallback";
            output = run_w2v_bert_layer13_ffn1_residual_cpu(model, input, tokens);
        }
        write_raw_f32(output_residual_f32, output);
        print_report(true, backend, &input, &output, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "", nullptr, nullptr, {e.what()});
        return false;
    }
}

W2VLayerQkv run_w2v_bert_layer13_qkv_cpu(const mit2::Bundle& model,
                                         const std::vector<float>& layer13_ffn1_residual,
                                         uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_v.weight");
    if (tokens == 0 || layer13_ffn1_residual.size() != static_cast<size_t>(tokens) * 1024u ||
        q_w.size() != static_cast<size_t>(1024) * 1024 ||
        k_w.size() != static_cast<size_t>(1024) * 1024 ||
        v_w.size() != static_cast<size_t>(1024) * 1024) {
        throw std::invalid_argument("w2v_bert_layer13_qkv invalid input sizes");
    }
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_v.bias");
    return {
        cpu_linear_rows(q_w, q_b, layer13_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(k_w, k_b, layer13_ffn1_residual, tokens, 1024, 1024),
        cpu_linear_rows(v_w, v_b, layer13_ffn1_residual, tokens, 1024, 1024),
    };
}

W2VLayerQkv run_w2v_bert_layer13_qkv_metal(mit2::MetalContext& metal,
                                           const mit2::Bundle& model,
                                           const std::vector<float>& layer13_ffn1_residual,
                                           uint32_t tokens) {
    const auto q_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_q.weight");
    const auto k_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_k.weight");
    const auto v_w = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_v.weight");
    const auto q_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_q.bias");
    const auto k_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_k.bias");
    const auto v_b = tensor_as_f32(model, "w2v_bert.encoder.layers.13.self_attn.linear_v.bias");
    return {
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.13.self_attn.linear_q.weight.resident",
                                       q_w,
                                       "w2v_bert.encoder.layers.13.self_attn.linear_q.bias.resident",
                                       q_b,
                                       layer13_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.13.self_attn.linear_k.weight.resident",
                                       k_w,
                                       "w2v_bert.encoder.layers.13.self_attn.linear_k.bias.resident",
                                       k_b,
                                       layer13_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
        metal.linear_rows_f32_resident("w2v_bert.encoder.layers.13.self_attn.linear_v.weight.resident",
                                       v_w,
                                       "w2v_bert.encoder.layers.13.self_attn.linear_v.bias.resident",
                                       v_b,
                                       layer13_ffn1_residual,
                                       tokens,
                                       1024,
                                       1024),
    };
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_qkv(const std::string& model_bundle_dir,
                                                   const std::string& layer13_ffn1_residual_f32,
                                                   uint32_t tokens,
                                                   const std::string& output_dir) {
    std::vector<std::string> issues;
    if (tokens == 0) {
        issues.push_back("w2v_tokens_must_be_positive");
    }
    append_raw_f32_count_issue(issues,
                               layer13_ffn1_residual_f32,
                               static_cast<uint64_t>(tokens) * 1024u,
                               "w2v_layer13_ffn1_residual");

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
    const auto output_q = (dir / "w2v_layer13_q.f32").string();
    const auto output_k = (dir / "w2v_layer13_k.f32").string();
    const auto output_v = (dir / "w2v_layer13_v.f32").string();
    const auto output_manifest = (dir / "w2v_layer13_qkv.manifest.json").string();

    auto print_failure = [&](const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_qkv\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_ffn1_residual_f32\": \"" << json_escape(layer13_ffn1_residual_f32) << "\",\n";
        std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (w2v_ok ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_qkv_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_qkv\": false,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_qkv\": false,\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_failure(issues);
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto input = read_raw_f32(layer13_ffn1_residual_f32);
        std::string execution_backend = "metal";
        W2VLayerQkv qkv;
        try {
            mit2::MetalContext metal;
            qkv = run_w2v_bert_layer13_qkv_metal(metal, model, input, tokens);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("Metal device unavailable") == std::string::npos) {
                throw;
            }
            execution_backend = "cpu_fallback";
            qkv = run_w2v_bert_layer13_qkv_cpu(model, input, tokens);
        }

        std::filesystem::create_directories(dir);
        write_raw_f32(output_q, qkv.q);
        write_raw_f32(output_k, qkv.k);
        write_raw_f32(output_v, qkv.v);
        std::ostringstream manifest_json;
        manifest_json << "{\n";
        manifest_json << "  \"format\": \"mit2-w2v-layer13-qkv-sidecars\",\n";
        manifest_json << "  \"version\": 1,\n";
        manifest_json << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        manifest_json << "  \"w2v_layer13_ffn1_residual_f32\": \"" << json_escape(layer13_ffn1_residual_f32) << "\",\n";
        manifest_json << "  \"w2v_tokens\": " << tokens << ",\n";
        manifest_json << "  \"output_q_f32\": \"" << json_escape(output_q) << "\",\n";
        manifest_json << "  \"output_q_sha256\": \"" << file_sha256_hex(output_q) << "\",\n";
        manifest_json << "  \"output_k_f32\": \"" << json_escape(output_k) << "\",\n";
        manifest_json << "  \"output_k_sha256\": \"" << file_sha256_hex(output_k) << "\",\n";
        manifest_json << "  \"output_v_f32\": \"" << json_escape(output_v) << "\",\n";
        manifest_json << "  \"output_v_sha256\": \"" << file_sha256_hex(output_v) << "\",\n";
        manifest_json << "  \"execution_backend\": \"" << execution_backend << "\",\n";
        manifest_json << "  \"ready_native_w2v_bert_layer13_qkv\": true,\n";
        manifest_json << "  \"ready_native_w2v_bert_semantic_features\": false\n";
        manifest_json << "}\n";
        write_text_file(output_manifest, manifest_json.str());

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_qkv\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"w2v_layer13_ffn1_residual_f32\": \"" << json_escape(layer13_ffn1_residual_f32) << "\",\n";
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
        std::cout << "  \"layer13_ffn1_residual_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_ffn1_residual_values\": " << input.size() << ",\n";
        std::cout << "  \"q_values\": " << qkv.q.size() << ",\n";
        std::cout << "  \"k_values\": " << qkv.k.size() << ",\n";
        std::cout << "  \"v_values\": " << qkv.v.size() << ",\n";
        std::cout << "  \"has_w2v_bert_model_contract\": true,\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_required << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_present << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_w2v_layer13_qkv_issues\": [],\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_qkv\": true,\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_qkv\": " << (execution_backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        print_failure({e.what()});
        return false;
    }
}

std::vector<float> run_w2v_bert_layer13_attention_context_cpu(const std::vector<float>& q,
                                                              const std::vector<float>& k,
                                                              const std::vector<float>& v,
                                                              const std::vector<uint32_t>& key_mask,
                                                              uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    if (tokens == 0 || q.size() != static_cast<size_t>(tokens) * heads * head_dim ||
        k.size() != q.size() || v.size() != q.size() || key_mask.size() != tokens) {
        throw std::invalid_argument("w2v_bert_layer13_attention_context invalid input sizes");
    }
    return cpu_cross_attention_heads_masked(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

std::vector<float> run_w2v_bert_layer13_attention_context_metal(mit2::MetalContext& metal,
                                                                const std::vector<float>& q,
                                                                const std::vector<float>& k,
                                                                const std::vector<float>& v,
                                                                const std::vector<uint32_t>& key_mask,
                                                                uint32_t tokens) {
    constexpr uint32_t heads = 16;
    constexpr uint32_t head_dim = 64;
    return metal.cross_attention_heads_masked_f32(q, k, v, key_mask, tokens, tokens, heads, head_dim);
}

[[maybe_unused]] bool run_tts_clone_w2v_layer13_attention(const std::string& q_f32,
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
    append_raw_f32_count_issue(issues, q_f32, qkv_values, "w2v_layer13_q");
    append_raw_f32_count_issue(issues, k_f32, qkv_values, "w2v_layer13_k");
    append_raw_f32_count_issue(issues, v_f32, qkv_values, "w2v_layer13_v");
    append_raw_u32_count_issue(issues, attention_mask_u32, tokens, "w2v_attention_mask");

    auto print_report = [&](bool ok,
                            const std::string& backend,
                            const std::optional<std::vector<float>>& output,
                            const std::vector<std::string>& report_issues) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_w2v_bert_layer13_attention_context\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"w2v_layer13_q_f32\": \"" << json_escape(q_f32) << "\",\n";
        std::cout << "  \"w2v_layer13_k_f32\": \"" << json_escape(k_f32) << "\",\n";
        std::cout << "  \"w2v_layer13_v_f32\": \"" << json_escape(v_f32) << "\",\n";
        std::cout << "  \"w2v_attention_mask_u32\": \"" << json_escape(attention_mask_u32) << "\",\n";
        std::cout << "  \"output_layer13_context_f32\": \"" << json_escape(output_context_f32) << "\",\n";
        if (ok) {
            std::cout << "  \"output_layer13_context_sha256\": \"" << file_sha256_hex(output_context_f32) << "\",\n";
        }
        std::cout << "  \"execution_backend\": \"" << backend << "\",\n";
        std::cout << "  \"w2v_tokens\": " << tokens << ",\n";
        std::cout << "  \"layer13_qkv_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"layer13_context_shape\": \"[1," << tokens << ",1024]\",\n";
        std::cout << "  \"attention_mask_shape\": \"[1," << tokens << "]\",\n";
        std::cout << "  \"q_values\": " << qkv_values << ",\n";
        std::cout << "  \"k_values\": " << qkv_values << ",\n";
        std::cout << "  \"v_values\": " << qkv_values << ",\n";
        std::cout << "  \"attention_mask_values\": " << tokens << ",\n";
        if (output) {
            std::cout << "  \"layer13_context_values\": " << output->size() << ",\n";
        }
        std::cout << "  \"clone_w2v_layer13_attention_issues\": ";
        print_json_string_array(report_issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_w2v_bert_layer13_attention_context\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_metal_w2v_bert_layer13_attention_context\": " << (ok && backend == "metal" ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_semantic_features\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT layer-13 attention output projection/residual/norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17\"\n";
        std::cout << "}\n";
    };
    if (!issues.empty()) {
        print_report(false, "not_run", std::nullopt, issues);
        return false;
    }

    try {
        const auto q = read_raw_f32(q_f32);
        const auto k = read_raw_f32(k_f32);
        const auto v = read_raw_f32(v_f32);
        const auto mask = read_raw_u32(attention_mask_u32);
        std::string execution_backend = "cpu_distance_emb";
        const std::vector<float> context = cpu_w2v_bert_cross_attention_with_distance(q, k, v, mask, tokens, dist_emb, 64, 8);
        write_raw_f32(output_context_f32, context);
        print_report(true, execution_backend, context, {});
        return true;
    } catch (const std::exception& e) {
        print_report(false, "not_run", std::nullopt, {e.what()});
        return false;
    }
}

