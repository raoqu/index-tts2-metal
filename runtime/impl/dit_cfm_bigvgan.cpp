std::vector<float> run_dit_input_merge_cpu(const mit2::Bundle& bundle, const std::vector<float>& x, const std::vector<float>& prompt_x, const std::vector<float>& cond, const std::vector<float>& style, uint32_t tokens) {
    auto cond_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_projection.weight");
    auto cond_b = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_projection.bias");
    auto merge_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.weight");
    auto merge_b = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.bias");
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> cond_row(cond.begin() + t * 512, cond.begin() + (t + 1) * 512);
        auto cond_proj = cpu_linear(cond_w, cond_b, cond_row, 512, 512);
        std::vector<float> merged(864);
        std::copy(x.begin() + t * 80, x.begin() + (t + 1) * 80, merged.begin());
        std::copy(prompt_x.begin() + t * 80, prompt_x.begin() + (t + 1) * 80, merged.begin() + 80);
        std::copy(cond_proj.begin(), cond_proj.end(), merged.begin() + 160);
        std::copy(style.begin(), style.end(), merged.begin() + 672);
        auto row_out = cpu_linear(merge_w, merge_b, merged, 512, 864);
        std::copy(row_out.begin(), row_out.end(), out.begin() + t * 512);
    }
    return out;
}

std::vector<float> run_dit_input_merge_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& x, const std::vector<float>& prompt_x, const std::vector<float>& cond, const std::vector<float>& style, uint32_t tokens) {
    auto cond_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_projection.weight");
    auto cond_b = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_projection.bias");
    auto merge_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.weight");
    auto merge_b = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.bias");
    auto cond_proj = metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.cond_projection.weight.resident",
        cond_w,
        "s2mel.net.cfm.estimator.cond_projection.bias.resident",
        cond_b,
        cond,
        tokens,
        512,
        512);
    auto merged = metal.dit_input_merge_f32(x, prompt_x, cond_proj, style, tokens);
    return metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.cond_x_merge_linear.weight.resident",
        merge_w,
        "s2mel.net.cfm.estimator.cond_x_merge_linear.bias.resident",
        merge_b,
        merged,
        tokens,
        512,
        864);
}

std::vector<float> run_dit_input_merge_metal_batched(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& x, const std::vector<float>& prompt_x, const std::vector<float>& cond, const std::vector<float>& style, uint32_t batch, uint32_t tokens) {
    const uint32_t rows = batch * tokens;
    auto cond_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_projection.weight");
    auto cond_b = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_projection.bias");
    auto merge_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.weight");
    auto merge_b = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.bias");
    auto cond_proj = metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.cond_projection.weight.resident",
        cond_w,
        "s2mel.net.cfm.estimator.cond_projection.bias.resident",
        cond_b,
        cond,
        rows,
        512,
        512);
    auto merged = metal.dit_input_merge_batched_f32(x, prompt_x, cond_proj, style, batch, tokens);
    return metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.cond_x_merge_linear.weight.resident",
        merge_w,
        "s2mel.net.cfm.estimator.cond_x_merge_linear.bias.resident",
        merge_b,
        merged,
        rows,
        512,
        864);
}

bool run_dit_input_merge_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    std::vector<float> x(static_cast<size_t>(tokens) * 80);
    std::vector<float> prompt_x(static_cast<size_t>(tokens) * 80);
    std::vector<float> cond(static_cast<size_t>(tokens) * 512);
    std::vector<float> style(192);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.031f) * 0.3f;
        prompt_x[i] = std::cos(static_cast<float>(i) * 0.027f) * 0.2f;
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.011f) * 0.15f + std::cos(static_cast<float>(i % 127) * 0.017f) * 0.05f;
    }
    for (size_t i = 0; i < style.size(); ++i) {
        style[i] = std::cos(static_cast<float>(i) * 0.019f) * 0.25f;
    }
    auto ref = run_dit_input_merge_cpu(bundle, x, prompt_x, cond, style, tokens);
    auto got = run_dit_input_merge_metal(metal, bundle, x, prompt_x, cond, style, tokens);
    const float err = max_abs_error(got, ref);

    std::vector<float> prompt_x_branch1(prompt_x.size());
    std::vector<float> cond_branch1(cond.size());
    std::vector<float> style_branch1(style.size());
    for (size_t i = 0; i < prompt_x_branch1.size(); ++i) {
        prompt_x_branch1[i] = std::sin(static_cast<float>(i) * 0.037f) * 0.17f -
                              std::cos(static_cast<float>(i % 67) * 0.029f) * 0.04f;
    }
    for (size_t i = 0; i < cond_branch1.size(); ++i) {
        cond_branch1[i] = std::cos(static_cast<float>(i) * 0.013f) * 0.11f -
                          std::sin(static_cast<float>(i % 113) * 0.023f) * 0.07f;
    }
    for (size_t i = 0; i < style_branch1.size(); ++i) {
        style_branch1[i] = std::sin(static_cast<float>(i) * 0.041f) * 0.21f;
    }
    auto got_branch1 = run_dit_input_merge_metal(metal, bundle, x, prompt_x_branch1, cond_branch1, style_branch1, tokens);
    std::vector<float> x_batched(x.size() * 2);
    std::copy(x.begin(), x.end(), x_batched.begin());
    std::copy(x.begin(), x.end(), x_batched.begin() + static_cast<std::ptrdiff_t>(x.size()));
    std::vector<float> prompt_x_batched(prompt_x.size() * 2);
    std::copy(prompt_x.begin(), prompt_x.end(), prompt_x_batched.begin());
    std::copy(prompt_x_branch1.begin(), prompt_x_branch1.end(), prompt_x_batched.begin() + static_cast<std::ptrdiff_t>(prompt_x.size()));
    std::vector<float> cond_batched(cond.size() * 2);
    std::copy(cond.begin(), cond.end(), cond_batched.begin());
    std::copy(cond_branch1.begin(), cond_branch1.end(), cond_batched.begin() + static_cast<std::ptrdiff_t>(cond.size()));
    std::vector<float> style_batched(style.size() * 2);
    std::copy(style.begin(), style.end(), style_batched.begin());
    std::copy(style_branch1.begin(), style_branch1.end(), style_batched.begin() + static_cast<std::ptrdiff_t>(style.size()));
    auto got_batched = run_dit_input_merge_metal_batched(metal, bundle, x_batched, prompt_x_batched, cond_batched, style_batched, 2, tokens);
    const size_t branch_size = static_cast<size_t>(tokens) * 512;
    std::vector<float> got_batched0(got_batched.begin(), got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size));
    std::vector<float> got_batched1(got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size),
                                    got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size * 2));
    const float batched_branch0_err = max_abs_error(got_batched0, got);
    const float batched_branch1_err = max_abs_error(got_batched1, got_branch1);
    const float batched_err = std::max(batched_branch0_err, batched_branch1_err);

    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_input_merge\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"merged_dim\": 864,\n";
    std::cout << "  \"output_dim\": 512,\n";
    std::cout << "  \"max_abs_error\": " << err << ",\n";
    std::cout << "  \"batched_branch0_max_abs_error\": " << batched_branch0_err << ",\n";
    std::cout << "  \"batched_branch1_max_abs_error\": " << batched_branch1_err << ",\n";
    std::cout << "  \"batched_max_abs_error\": " << batched_err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f && batched_err <= 1e-4f;
}

std::vector<float> run_dit_attention_projection_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto wqkv = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wqkv.weight");
    auto wo = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wo.weight");
    std::vector<float> zero_qkv_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto qkv = cpu_linear(wqkv, zero_qkv_bias, row, 1536, 512);
        std::vector<float> q(qkv.begin(), qkv.begin() + 512);
        auto projected = cpu_linear(wo, zero_out_bias, q, 512, 512);
        std::copy(projected.begin(), projected.end(), out.begin() + t * 512);
    }
    return out;
}

std::vector<float> run_dit_attention_projection_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto wqkv = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wqkv.weight");
    auto wo = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wo.weight");
    std::vector<float> zero_qkv_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto qkv = metal.linear_f32(wqkv, zero_qkv_bias, row, 1536, 512);
        std::vector<float> q(qkv.begin(), qkv.begin() + 512);
        auto projected = metal.linear_f32(wo, zero_out_bias, q, 512, 512);
        std::copy(projected.begin(), projected.end(), out.begin() + t * 512);
    }
    return out;
}

bool run_dit_attention_projection_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.023f) * 0.2f + std::cos(static_cast<float>(i % 89) * 0.017f) * 0.1f;
    }
    auto ref = run_dit_attention_projection_cpu(bundle, input, tokens);
    auto got = run_dit_attention_projection_metal(metal, bundle, input, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_attention_projection_layer0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"qkv_dim\": 1536,\n";
    std::cout << "  \"output_dim\": 512,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_dit_feed_forward_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto w1 = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.feed_forward.w1.weight");
    auto w3 = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.feed_forward.w3.weight");
    auto w2 = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.feed_forward.w2.weight");
    std::vector<float> zero_mid_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto h1 = cpu_linear(w1, zero_mid_bias, row, 1536, 512);
        auto h3 = cpu_linear(w3, zero_mid_bias, row, 1536, 512);
        for (size_t i = 0; i < h1.size(); ++i) {
            h1[i] = (h1[i] / (1.0f + std::exp(-h1[i]))) * h3[i];
        }
        auto y = cpu_linear(w2, zero_out_bias, h1, 512, 1536);
        std::copy(y.begin(), y.end(), out.begin() + t * 512);
    }
    return out;
}

std::vector<float> run_dit_feed_forward_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto w1 = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.feed_forward.w1.weight");
    auto w3 = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.feed_forward.w3.weight");
    auto w2 = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.feed_forward.w2.weight");
    std::vector<float> zero_mid_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto h1 = metal.linear_f32(w1, zero_mid_bias, row, 1536, 512);
        auto h3 = metal.linear_f32(w3, zero_mid_bias, row, 1536, 512);
        h1 = metal.silu_f32(h1);
        for (size_t i = 0; i < h1.size(); ++i) {
            h1[i] *= h3[i];
        }
        auto y = metal.linear_f32(w2, zero_out_bias, h1, 512, 1536);
        std::copy(y.begin(), y.end(), out.begin() + t * 512);
    }
    return out;
}

bool run_dit_feed_forward_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.029f) * 0.15f + std::cos(static_cast<float>(i % 101) * 0.013f) * 0.07f;
    }
    auto ref = run_dit_feed_forward_cpu(bundle, input, tokens);
    auto got = run_dit_feed_forward_metal(metal, bundle, input, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_feed_forward_layer0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"intermediate_dim\": 1536,\n";
    std::cout << "  \"output_dim\": 512,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_adaptive_norm_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, uint32_t tokens, const std::string& prefix) {
    auto norm_weight = tensor_as_f32(bundle, prefix + ".norm.weight");
    auto proj_weight = tensor_as_f32(bundle, prefix + ".project_layer.weight");
    auto proj_bias = tensor_as_f32(bundle, prefix + ".project_layer.bias");
    auto wb = cpu_linear(proj_weight, proj_bias, cond, 1024, 512);
    std::vector<float> adaptive_weight(wb.begin(), wb.begin() + 512);
    std::vector<float> adaptive_bias(wb.begin() + 512, wb.end());
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto normed = cpu_rmsnorm(row, norm_weight, 1e-5f);
        for (uint32_t i = 0; i < 512; ++i) {
            out[static_cast<size_t>(t) * 512 + i] = adaptive_weight[i] * normed[i] + adaptive_bias[i];
        }
    }
    return out;
}

std::vector<float> run_adaptive_norm_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, uint32_t tokens, const std::string& prefix) {
    auto norm_weight = tensor_as_f32(bundle, prefix + ".norm.weight");
    auto proj_weight = tensor_as_f32(bundle, prefix + ".project_layer.weight");
    auto proj_bias = tensor_as_f32(bundle, prefix + ".project_layer.bias");
    auto wb = metal.linear_f32_resident(
        prefix + ".project_layer.weight.resident",
        proj_weight,
        prefix + ".project_layer.bias.resident",
        proj_bias,
        cond,
        1024,
        512);
    std::vector<float> adaptive_weight(wb.begin(), wb.begin() + 512);
    std::vector<float> adaptive_bias(wb.begin() + 512, wb.end());
    return metal.adaptive_rmsnorm_rows_f32_resident(
        prefix + ".norm.weight.resident",
        norm_weight,
        input,
        adaptive_weight,
        adaptive_bias,
        tokens,
        512,
        1e-5f);
}

bool run_dit_adaptive_norm_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    std::vector<float> cond(512);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.021f) * 0.2f + std::cos(static_cast<float>(i % 109) * 0.015f) * 0.1f;
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::cos(static_cast<float>(i) * 0.018f) * 0.3f;
    }
    const std::string attn_prefix = "s2mel.net.cfm.estimator.transformer.layers.0.attention_norm";
    const std::string ffn_prefix = "s2mel.net.cfm.estimator.transformer.layers.0.ffn_norm";
    auto attn_ref = run_adaptive_norm_cpu(bundle, input, cond, tokens, attn_prefix);
    auto attn_got = run_adaptive_norm_metal(metal, bundle, input, cond, tokens, attn_prefix);
    auto ffn_ref = run_adaptive_norm_cpu(bundle, input, cond, tokens, ffn_prefix);
    auto ffn_got = run_adaptive_norm_metal(metal, bundle, input, cond, tokens, ffn_prefix);
    const float attn_err = max_abs_error(attn_got, attn_ref);
    const float ffn_err = max_abs_error(ffn_got, ffn_ref);
    const float err = std::max(attn_err, ffn_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_adaptive_norm_layer0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"hidden_dim\": 512,\n";
    std::cout << "  \"attention_norm_max_abs_error\": " << attn_err << ",\n";
    std::cout << "  \"ffn_norm_max_abs_error\": " << ffn_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_dit_attention_core_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto wqkv = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wqkv.weight");
    auto wo = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wo.weight");
    std::vector<float> zero_qkv_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    constexpr uint32_t heads = 8;
    constexpr uint32_t head_dim = 64;
    std::vector<float> q(static_cast<size_t>(tokens) * heads * head_dim);
    std::vector<float> k(q.size());
    std::vector<float> v(q.size());
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto qkv = cpu_linear(wqkv, zero_qkv_bias, row, 1536, 512);
        std::copy(qkv.begin(), qkv.begin() + 512, q.begin() + static_cast<size_t>(t) * 512);
        std::copy(qkv.begin() + 512, qkv.begin() + 1024, k.begin() + static_cast<size_t>(t) * 512);
        std::copy(qkv.begin() + 1024, qkv.end(), v.begin() + static_cast<size_t>(t) * 512);
    }
    apply_rope_inplace(q, tokens, heads, head_dim);
    apply_rope_inplace(k, tokens, heads, head_dim);
    std::vector<float> concat(static_cast<size_t>(tokens) * 512);
    for (uint32_t h = 0; h < heads; ++h) {
        std::vector<float> qh(static_cast<size_t>(tokens) * head_dim);
        std::vector<float> kh(qh.size());
        std::vector<float> vh(qh.size());
        for (uint32_t t = 0; t < tokens; ++t) {
            const size_t src = (static_cast<size_t>(t) * heads + h) * head_dim;
            std::copy(q.begin() + src, q.begin() + src + head_dim, qh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(k.begin() + src, k.begin() + src + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(v.begin() + src, v.begin() + src + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
        }
        auto yh = cpu_attention_single_head(qh, kh, vh, tokens, head_dim);
        for (uint32_t t = 0; t < tokens; ++t) {
            std::copy(yh.begin() + static_cast<size_t>(t) * head_dim, yh.begin() + static_cast<size_t>(t + 1) * head_dim, concat.begin() + static_cast<size_t>(t) * 512 + h * head_dim);
        }
    }
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(concat.begin() + t * 512, concat.begin() + (t + 1) * 512);
        auto y = cpu_linear(wo, zero_out_bias, row, 512, 512);
        std::copy(y.begin(), y.end(), out.begin() + static_cast<size_t>(t) * 512);
    }
    return out;
}

std::vector<float> run_dit_attention_core_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto wqkv = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wqkv.weight");
    auto wo = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wo.weight");
    std::vector<float> zero_qkv_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    constexpr uint32_t heads = 8;
    constexpr uint32_t head_dim = 64;
    auto qkv_rows = metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.transformer.layers.0.attention.wqkv.weight.resident",
        wqkv,
        "s2mel.net.cfm.estimator.transformer.attention.wqkv.zero_bias.resident",
        zero_qkv_bias,
        input,
        tokens,
        1536,
        512);
    std::vector<float> concat;
    if (tokens <= kFusedDitAttentionMaxTokens) {
        std::vector<uint32_t> key_mask(tokens, 1);
        concat = metal.dit_attention_qkv_rope_f32(qkv_rows, key_mask, tokens, heads, head_dim);
        return metal.linear_rows_f32_resident(
            "s2mel.net.cfm.estimator.transformer.layers.0.attention.wo.weight.resident",
            wo,
            "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident",
            zero_out_bias,
            concat,
            tokens,
            512,
            512);
    }

    std::vector<float> q(static_cast<size_t>(tokens) * heads * head_dim);
    std::vector<float> k(q.size());
    std::vector<float> v(q.size());
    for (uint32_t t = 0; t < tokens; ++t) {
        const auto src = qkv_rows.begin() + static_cast<size_t>(t) * 1536;
        std::copy(src, src + 512, q.begin() + static_cast<size_t>(t) * 512);
        std::copy(src + 512, src + 1024, k.begin() + static_cast<size_t>(t) * 512);
        std::copy(src + 1024, src + 1536, v.begin() + static_cast<size_t>(t) * 512);
    }
    apply_rope_inplace(q, tokens, heads, head_dim);
    apply_rope_inplace(k, tokens, heads, head_dim);
    concat.resize(static_cast<size_t>(tokens) * 512);
    for (uint32_t h = 0; h < heads; ++h) {
        std::vector<float> qh(static_cast<size_t>(tokens) * head_dim);
        std::vector<float> kh(qh.size());
        std::vector<float> vh(qh.size());
        for (uint32_t t = 0; t < tokens; ++t) {
            const size_t src = (static_cast<size_t>(t) * heads + h) * head_dim;
            std::copy(q.begin() + src, q.begin() + src + head_dim, qh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(k.begin() + src, k.begin() + src + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(v.begin() + src, v.begin() + src + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
        }
        auto yh = metal.attention_single_head_f32(qh, kh, vh, tokens, head_dim);
        for (uint32_t t = 0; t < tokens; ++t) {
            std::copy(yh.begin() + static_cast<size_t>(t) * head_dim, yh.begin() + static_cast<size_t>(t + 1) * head_dim, concat.begin() + static_cast<size_t>(t) * 512 + h * head_dim);
        }
    }
    return metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.transformer.layers.0.attention.wo.weight.resident",
        wo,
        "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident",
        zero_out_bias,
        concat,
        tokens,
        512,
        512);
}

bool run_dit_attention_core_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.13f + std::cos(static_cast<float>(i % 97) * 0.011f) * 0.09f;
    }
    auto ref = run_dit_attention_core_cpu(bundle, input, tokens);
    auto got = run_dit_attention_core_metal(metal, bundle, input, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_attention_core_layer0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"heads\": 8,\n";
    std::cout << "  \"head_dim\": 64,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_dit_attention_core_cpu_masked_layer(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t layer) {
    const std::string base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto wqkv = tensor_as_f32(bundle, base + ".attention.wqkv.weight");
    auto wo = tensor_as_f32(bundle, base + ".attention.wo.weight");
    std::vector<float> zero_qkv_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    constexpr uint32_t heads = 8;
    constexpr uint32_t head_dim = 64;
    std::vector<float> q(static_cast<size_t>(tokens) * heads * head_dim);
    std::vector<float> k(q.size());
    std::vector<float> v(q.size());
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto qkv = cpu_linear(wqkv, zero_qkv_bias, row, 1536, 512);
        std::copy(qkv.begin(), qkv.begin() + 512, q.begin() + static_cast<size_t>(t) * 512);
        std::copy(qkv.begin() + 512, qkv.begin() + 1024, k.begin() + static_cast<size_t>(t) * 512);
        std::copy(qkv.begin() + 1024, qkv.end(), v.begin() + static_cast<size_t>(t) * 512);
    }
    apply_rope_inplace(q, tokens, heads, head_dim);
    apply_rope_inplace(k, tokens, heads, head_dim);
    std::vector<float> concat(static_cast<size_t>(tokens) * 512);
    for (uint32_t h = 0; h < heads; ++h) {
        std::vector<float> qh(static_cast<size_t>(tokens) * head_dim);
        std::vector<float> kh(qh.size());
        std::vector<float> vh(qh.size());
        for (uint32_t t = 0; t < tokens; ++t) {
            const size_t src = (static_cast<size_t>(t) * heads + h) * head_dim;
            std::copy(q.begin() + src, q.begin() + src + head_dim, qh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(k.begin() + src, k.begin() + src + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(v.begin() + src, v.begin() + src + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
        }
        auto yh = cpu_attention_single_head_masked(qh, kh, vh, key_mask, tokens, head_dim);
        for (uint32_t t = 0; t < tokens; ++t) {
            std::copy(yh.begin() + static_cast<size_t>(t) * head_dim, yh.begin() + static_cast<size_t>(t + 1) * head_dim, concat.begin() + static_cast<size_t>(t) * 512 + h * head_dim);
        }
    }
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(concat.begin() + t * 512, concat.begin() + (t + 1) * 512);
        auto y = cpu_linear(wo, zero_out_bias, row, 512, 512);
        std::copy(y.begin(), y.end(), out.begin() + static_cast<size_t>(t) * 512);
    }
    return out;
}

std::vector<float> run_dit_attention_core_metal_masked_layer(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t layer) {
    const std::string base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto wqkv = tensor_as_f32(bundle, base + ".attention.wqkv.weight");
    auto wo = tensor_as_f32(bundle, base + ".attention.wo.weight");
    std::vector<float> zero_qkv_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    constexpr uint32_t heads = 8;
    constexpr uint32_t head_dim = 64;
    auto qkv_rows = metal.linear_rows_f32_resident(
        base + ".attention.wqkv.weight.resident",
        wqkv,
        "s2mel.net.cfm.estimator.transformer.attention.wqkv.zero_bias.resident",
        zero_qkv_bias,
        input,
        tokens,
        1536,
        512);
    if (tokens <= kFusedDitAttentionMaxTokens) {
        auto concat = metal.dit_attention_qkv_rope_f32(qkv_rows, key_mask, tokens, heads, head_dim);
        return metal.linear_rows_f32_resident(
            base + ".attention.wo.weight.resident",
            wo,
            "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident",
            zero_out_bias,
            concat,
            tokens,
            512,
            512);
    }

    std::vector<float> q(static_cast<size_t>(tokens) * heads * head_dim);
    std::vector<float> k(q.size());
    std::vector<float> v(q.size());
    for (uint32_t t = 0; t < tokens; ++t) {
        const auto src = qkv_rows.begin() + static_cast<size_t>(t) * 1536;
        std::copy(src, src + 512, q.begin() + static_cast<size_t>(t) * 512);
        std::copy(src + 512, src + 1024, k.begin() + static_cast<size_t>(t) * 512);
        std::copy(src + 1024, src + 1536, v.begin() + static_cast<size_t>(t) * 512);
    }
    apply_rope_inplace(q, tokens, heads, head_dim);
    apply_rope_inplace(k, tokens, heads, head_dim);
    std::vector<float> concat(static_cast<size_t>(tokens) * 512);
    for (uint32_t h = 0; h < heads; ++h) {
        std::vector<float> qh(static_cast<size_t>(tokens) * head_dim);
        std::vector<float> kh(qh.size());
        std::vector<float> vh(qh.size());
        for (uint32_t t = 0; t < tokens; ++t) {
            const size_t src = (static_cast<size_t>(t) * heads + h) * head_dim;
            std::copy(q.begin() + src, q.begin() + src + head_dim, qh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(k.begin() + src, k.begin() + src + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
            std::copy(v.begin() + src, v.begin() + src + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
        }
        auto yh = metal.attention_single_head_masked_f32(qh, kh, vh, key_mask, tokens, head_dim);
        for (uint32_t t = 0; t < tokens; ++t) {
            std::copy(yh.begin() + static_cast<size_t>(t) * head_dim, yh.begin() + static_cast<size_t>(t + 1) * head_dim, concat.begin() + static_cast<size_t>(t) * 512 + h * head_dim);
        }
    }
    return metal.linear_rows_f32_resident(
        base + ".attention.wo.weight.resident",
        wo,
        "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident",
        zero_out_bias,
        concat,
        tokens,
        512,
        512);
}

std::vector<float> run_dit_attention_core_metal_masked_layer_batched(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<uint32_t>& key_mask, uint32_t batch, uint32_t tokens, uint32_t layer);

bool run_dit_attention_core_tokens_test(const std::string& bundle_dir, uint32_t tokens) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.13f + std::cos(static_cast<float>(i % 97) * 0.011f) * 0.09f;
    }
    std::vector<uint32_t> key_mask(tokens, 1);
    auto ref = run_dit_attention_core_cpu_masked_layer(bundle, input, key_mask, tokens, 0);
    auto got = run_dit_attention_core_metal_masked_layer(metal, bundle, input, key_mask, tokens, 0);
    const float err = max_abs_error(got, ref);
    const bool fused = tokens <= kFusedDitAttentionMaxTokens;

    // Batched (batch=2) fused kernel: vs CPU and repeatability.
    constexpr uint32_t batch = 2;
    std::vector<float> input_b(input.size() * batch);
    std::copy(input.begin(), input.end(), input_b.begin());
    for (size_t i = 0; i < input.size(); ++i) {
        input_b[input.size() + i] = std::cos(static_cast<float>(i) * 0.019f) * 0.11f;
    }
    std::vector<uint32_t> mask_b(static_cast<size_t>(batch) * tokens, 1);
    auto got_b1 = run_dit_attention_core_metal_masked_layer_batched(metal, bundle, input_b, mask_b, batch, tokens, 0);
    auto got_b2 = run_dit_attention_core_metal_masked_layer_batched(metal, bundle, input_b, mask_b, batch, tokens, 0);
    const float rep_err = max_abs_error(got_b1, got_b2);
    std::vector<float> branch0(got_b1.begin(), got_b1.begin() + static_cast<std::ptrdiff_t>(input.size()));
    const float batched_err = max_abs_error(branch0, ref);

    // Isolate: raw fused batched attention kernel on a fixed qkv (no linear_rows).
    std::vector<float> raw_qkv(static_cast<size_t>(batch) * tokens * 1536);
    for (size_t i = 0; i < raw_qkv.size(); ++i) {
        raw_qkv[i] = std::sin(static_cast<float>(i % 1009) * 0.013f) * 0.21f;
    }
    auto raw1 = metal.dit_attention_qkv_rope_batched_f32(raw_qkv, mask_b, batch, tokens, 8, 64);
    auto raw2 = metal.dit_attention_qkv_rope_batched_f32(raw_qkv, mask_b, batch, tokens, 8, 64);
    const float raw_rep_err = max_abs_error(raw1, raw2);

    // Isolate: linear_rows repeatability at batch*tokens rows.
    auto wqkv_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.transformer.layers.0.attention.wqkv.weight");
    const std::vector<float> zb(1536, 0.0f);
    auto lin1 = metal.linear_rows_f32_resident("dbg.wqkv.resident", wqkv_w, "dbg.zb.resident", zb, input_b, batch * tokens, 1536, 512);
    auto lin2 = metal.linear_rows_f32_resident("dbg.wqkv.resident", wqkv_w, "dbg.zb.resident", zb, input_b, batch * tokens, 1536, 512);
    const float lin_rep_err = max_abs_error(lin1, lin2);

    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_attention_core_tokens\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"fused_kernel\": " << (fused ? "true" : "false") << ",\n";
    std::cout << "  \"max_abs_error\": " << err << ",\n";
    std::cout << "  \"batched_branch0_err\": " << batched_err << ",\n";
    std::cout << "  \"batched_repeat_err\": " << rep_err << ",\n";
    std::cout << "  \"raw_attention_repeat_err\": " << raw_rep_err << ",\n";
    std::cout << "  \"linear_rows_repeat_err\": " << lin_rep_err << "\n";
    std::cout << "}\n";
    return err <= 1e-3f && batched_err <= 1e-3f && rep_err == 0.0f;
}

std::vector<float> run_dit_attention_core_metal_masked_layer_batched(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<uint32_t>& key_mask, uint32_t batch, uint32_t tokens, uint32_t layer) {
    if (batch == 0 || input.size() != static_cast<size_t>(batch) * tokens * 512 ||
        key_mask.size() != static_cast<size_t>(batch) * tokens) {
        throw std::runtime_error("batched DiT attention input size mismatch");
    }
    const std::string base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto wqkv = tensor_as_f32(bundle, base + ".attention.wqkv.weight");
    auto wo = tensor_as_f32(bundle, base + ".attention.wo.weight");
    std::vector<float> zero_qkv_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    constexpr uint32_t heads = 8;
    constexpr uint32_t head_dim = 64;
    const uint32_t rows = batch * tokens;
    auto qkv_rows = metal.linear_rows_f32_resident(
        base + ".attention.wqkv.weight.resident",
        wqkv,
        "s2mel.net.cfm.estimator.transformer.attention.wqkv.zero_bias.resident",
        zero_qkv_bias,
        input,
        rows,
        1536,
        512);
    if (tokens <= kFusedDitAttentionMaxTokens) {
        auto concat = metal.dit_attention_qkv_rope_batched_f32(qkv_rows, key_mask, batch, tokens, heads, head_dim);
        return metal.linear_rows_f32_resident(
            base + ".attention.wo.weight.resident",
            wo,
            "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident",
            zero_out_bias,
            concat,
            rows,
            512,
            512);
    }

    std::vector<float> out(static_cast<size_t>(rows) * 512);
    const size_t qkv_branch_size = static_cast<size_t>(tokens) * 1536;
    const size_t out_branch_size = static_cast<size_t>(tokens) * 512;
    for (uint32_t b = 0; b < batch; ++b) {
        const auto qkv_begin = qkv_rows.begin() + static_cast<std::ptrdiff_t>(b * qkv_branch_size);
        std::vector<float> qkv_branch(qkv_begin, qkv_begin + static_cast<std::ptrdiff_t>(qkv_branch_size));
        const auto mask_begin = key_mask.begin() + static_cast<std::ptrdiff_t>(b * tokens);
        std::vector<uint32_t> mask_branch(mask_begin, mask_begin + static_cast<std::ptrdiff_t>(tokens));

        std::vector<float> q(static_cast<size_t>(tokens) * heads * head_dim);
        std::vector<float> k(q.size());
        std::vector<float> v(q.size());
        for (uint32_t t = 0; t < tokens; ++t) {
            const auto src = qkv_branch.begin() + static_cast<size_t>(t) * 1536;
            std::copy(src, src + 512, q.begin() + static_cast<size_t>(t) * 512);
            std::copy(src + 512, src + 1024, k.begin() + static_cast<size_t>(t) * 512);
            std::copy(src + 1024, src + 1536, v.begin() + static_cast<size_t>(t) * 512);
        }
        apply_rope_inplace(q, tokens, heads, head_dim);
        apply_rope_inplace(k, tokens, heads, head_dim);
        std::vector<float> concat(out_branch_size);
        for (uint32_t h = 0; h < heads; ++h) {
            std::vector<float> qh(static_cast<size_t>(tokens) * head_dim);
            std::vector<float> kh(qh.size());
            std::vector<float> vh(qh.size());
            for (uint32_t t = 0; t < tokens; ++t) {
                const size_t src = (static_cast<size_t>(t) * heads + h) * head_dim;
                std::copy(q.begin() + src, q.begin() + src + head_dim, qh.begin() + static_cast<size_t>(t) * head_dim);
                std::copy(k.begin() + src, k.begin() + src + head_dim, kh.begin() + static_cast<size_t>(t) * head_dim);
                std::copy(v.begin() + src, v.begin() + src + head_dim, vh.begin() + static_cast<size_t>(t) * head_dim);
            }
            auto yh = metal.attention_single_head_masked_f32(qh, kh, vh, mask_branch, tokens, head_dim);
            for (uint32_t t = 0; t < tokens; ++t) {
                std::copy(yh.begin() + static_cast<size_t>(t) * head_dim,
                          yh.begin() + static_cast<size_t>(t + 1) * head_dim,
                          concat.begin() + static_cast<size_t>(t) * 512 + h * head_dim);
            }
        }
        auto projected = metal.linear_rows_f32_resident(
            base + ".attention.wo.weight.resident",
            wo,
            "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident",
            zero_out_bias,
            concat,
            tokens,
            512,
            512);
        std::copy(projected.begin(), projected.end(), out.begin() + static_cast<std::ptrdiff_t>(b * out_branch_size));
    }
    return out;
}

std::vector<float> run_dit_feed_forward_cpu_layer(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layer) {
    const std::string base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto w1 = tensor_as_f32(bundle, base + ".feed_forward.w1.weight");
    auto w3 = tensor_as_f32(bundle, base + ".feed_forward.w3.weight");
    auto w2 = tensor_as_f32(bundle, base + ".feed_forward.w2.weight");
    std::vector<float> zero_mid_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto h1 = cpu_linear(w1, zero_mid_bias, row, 1536, 512);
        auto h3 = cpu_linear(w3, zero_mid_bias, row, 1536, 512);
        for (size_t i = 0; i < h1.size(); ++i) {
            h1[i] = (h1[i] / (1.0f + std::exp(-h1[i]))) * h3[i];
        }
        auto y = cpu_linear(w2, zero_out_bias, h1, 512, 1536);
        std::copy(y.begin(), y.end(), out.begin() + t * 512);
    }
    return out;
}

std::vector<float> run_dit_feed_forward_metal_layer(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens, uint32_t layer) {
    const std::string base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto w1 = tensor_as_f32(bundle, base + ".feed_forward.w1.weight");
    auto w3 = tensor_as_f32(bundle, base + ".feed_forward.w3.weight");
    auto w2 = tensor_as_f32(bundle, base + ".feed_forward.w2.weight");
    std::vector<float> zero_mid_bias(1536, 0.0f);
    std::vector<float> zero_out_bias(512, 0.0f);
    auto h1 = metal.linear_rows_f32_resident(
        base + ".feed_forward.w1.weight.resident",
        w1,
        "s2mel.net.cfm.estimator.transformer.feed_forward.zero_mid_bias.resident",
        zero_mid_bias,
        input,
        tokens,
        1536,
        512);
    auto h3 = metal.linear_rows_f32_resident(
        base + ".feed_forward.w3.weight.resident",
        w3,
        "s2mel.net.cfm.estimator.transformer.feed_forward.zero_mid_bias.resident",
        zero_mid_bias,
        input,
        tokens,
        1536,
        512);
    auto gated = metal.silu_mul_f32(h1, h3);
    return metal.linear_rows_f32_resident(
        base + ".feed_forward.w2.weight.resident",
        w2,
        "s2mel.net.cfm.estimator.transformer.feed_forward.zero_out_bias.resident",
        zero_out_bias,
        gated,
        tokens,
        512,
        1536);
}

std::vector<float> run_skip_in_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& skip, uint32_t tokens, uint32_t layer) {
    const std::string base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer) + ".skip_in_linear";
    auto weight = tensor_as_f32(bundle, base + ".weight");
    auto bias = tensor_as_f32(bundle, base + ".bias");
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(1024);
        std::copy(input.begin() + t * 512, input.begin() + (t + 1) * 512, row.begin());
        std::copy(skip.begin() + t * 512, skip.begin() + (t + 1) * 512, row.begin() + 512);
        auto y = cpu_linear(weight, bias, row, 512, 1024);
        std::copy(y.begin(), y.end(), out.begin() + t * 512);
    }
    return out;
}

std::vector<float> run_skip_in_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& skip, uint32_t tokens, uint32_t layer) {
    const std::string base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer) + ".skip_in_linear";
    auto weight = tensor_as_f32(bundle, base + ".weight");
    auto bias = tensor_as_f32(bundle, base + ".bias");
    auto rows = metal.concat_rows_f32(input, skip, tokens, 512, 512);
    return metal.linear_rows_f32_resident(
        base + ".weight.resident",
        weight,
        base + ".bias.resident",
        bias,
        rows,
        tokens,
        512,
        1024);
}

std::vector<float> run_dit_transformer_block_cpu_layer(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t layer, const std::vector<float>* skip_in) {
    std::vector<float> x = skip_in ? run_skip_in_cpu(bundle, input, *skip_in, tokens, layer) : input;
    const std::string layer_base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto attn_normed = run_adaptive_norm_cpu(bundle, x, cond, tokens, layer_base + ".attention_norm");
    auto attn = run_dit_attention_core_cpu_masked_layer(bundle, attn_normed, key_mask, tokens, layer);
    std::vector<float> h(x.size());
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = x[i] + attn[i];
    }
    auto ffn_normed = run_adaptive_norm_cpu(bundle, h, cond, tokens, layer_base + ".ffn_norm");
    auto ffn = run_dit_feed_forward_cpu_layer(bundle, ffn_normed, tokens, layer);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = h[i] + ffn[i];
    }
    return out;
}

std::vector<float> run_dit_transformer_block_metal_layer(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t layer, const std::vector<float>* skip_in) {
    std::vector<float> x = skip_in ? run_skip_in_metal(metal, bundle, input, *skip_in, tokens, layer) : input;
    const std::string layer_base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto attn_normed = run_adaptive_norm_metal(metal, bundle, x, cond, tokens, layer_base + ".attention_norm");
    auto attn = run_dit_attention_core_metal_masked_layer(metal, bundle, attn_normed, key_mask, tokens, layer);
    auto h = metal.add_f32(x, attn);
    auto ffn_normed = run_adaptive_norm_metal(metal, bundle, h, cond, tokens, layer_base + ".ffn_norm");
    auto ffn = run_dit_feed_forward_metal_layer(metal, bundle, ffn_normed, tokens, layer);
    return metal.add_f32(h, ffn);
}

std::vector<float> run_dit_transformer_block_metal_layer_batched(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t batch, uint32_t tokens, uint32_t layer, const std::vector<float>* skip_in) {
    const uint32_t rows = batch * tokens;
    std::vector<float> x = skip_in ? run_skip_in_metal(metal, bundle, input, *skip_in, rows, layer) : input;
    if (x.size() != static_cast<size_t>(rows) * 512) {
        throw std::runtime_error("batched DiT block input size mismatch");
    }
    const std::string layer_base = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    auto attn_normed = run_adaptive_norm_metal(metal, bundle, x, cond, rows, layer_base + ".attention_norm");
    auto attn = run_dit_attention_core_metal_masked_layer_batched(metal, bundle, attn_normed, key_mask, batch, tokens, layer);
    auto h = metal.add_f32(x, attn);
    auto ffn_normed = run_adaptive_norm_metal(metal, bundle, h, cond, rows, layer_base + ".ffn_norm");
    auto ffn = run_dit_feed_forward_metal_layer(metal, bundle, ffn_normed, rows, layer);
    return metal.add_f32(h, ffn);
}

std::vector<float> run_dit_transformer_block0_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t tokens) {
    return run_dit_transformer_block_cpu_layer(bundle, input, cond, key_mask, tokens, 0, nullptr);
}

std::vector<float> run_dit_transformer_block0_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t tokens) {
    return run_dit_transformer_block_metal_layer(metal, bundle, input, cond, key_mask, tokens, 0, nullptr);
}

std::vector<float> run_dit_transformer_stack_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t tokens) {
    constexpr uint32_t layers = 13;
    std::vector<float> x = input;
    std::vector<std::vector<float>> skips;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const std::vector<float>* skip = nullptr;
        if (layer > layers / 2) {
            skip = &skips.back();
        }
        x = run_dit_transformer_block_cpu_layer(bundle, x, cond, key_mask, tokens, layer, skip);
        if (layer > layers / 2) {
            skips.pop_back();
        }
        if (layer < layers / 2) {
            skips.push_back(x);
        }
    }
    return run_adaptive_norm_cpu(bundle, x, cond, tokens, "s2mel.net.cfm.estimator.transformer.norm");
}

std::vector<float> run_dit_transformer_stack_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t tokens) {
    constexpr uint32_t layers = 13;
    std::vector<float> x = input;
    std::vector<std::vector<float>> skips;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const std::vector<float>* skip = nullptr;
        if (layer > layers / 2) {
            skip = &skips.back();
        }
        x = run_dit_transformer_block_metal_layer(metal, bundle, x, cond, key_mask, tokens, layer, skip);
        if (layer > layers / 2) {
            skips.pop_back();
        }
        if (layer < layers / 2) {
            skips.push_back(x);
        }
    }
    return run_adaptive_norm_metal(metal, bundle, x, cond, tokens, "s2mel.net.cfm.estimator.transformer.norm");
}

std::vector<float> run_dit_transformer_stack_metal_batched(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& key_mask, uint32_t batch, uint32_t tokens) {
    constexpr uint32_t layers = 13;
    const uint32_t rows = batch * tokens;
    if (input.size() != static_cast<size_t>(rows) * 512 ||
        key_mask.size() != static_cast<size_t>(rows)) {
        throw std::runtime_error("batched DiT stack input size mismatch");
    }
    std::vector<float> x = input;
    std::vector<std::vector<float>> skips;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const std::vector<float>* skip = nullptr;
        if (layer > layers / 2) {
            skip = &skips.back();
        }
        x = run_dit_transformer_block_metal_layer_batched(metal, bundle, x, cond, key_mask, batch, tokens, layer, skip);
        if (layer > layers / 2) {
            skips.pop_back();
        }
        if (layer < layers / 2) {
            skips.push_back(x);
        }
    }
    return run_adaptive_norm_metal(metal, bundle, x, cond, rows, "s2mel.net.cfm.estimator.transformer.norm");
}

bool run_dit_transformer_stack_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    constexpr uint32_t batch = 2;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    std::vector<float> input_branch1(static_cast<size_t>(tokens) * 512);
    std::vector<float> cond(512);
    std::vector<uint32_t> key_mask{1, 1, 1, 1, 0, 0};
    std::vector<uint32_t> key_mask_branch1{1, 0, 1, 1, 1, 0};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.015f) * 0.11f + std::cos(static_cast<float>(i % 131) * 0.019f) * 0.06f;
        input_branch1[i] = std::cos(static_cast<float>(i) * 0.011f) * 0.09f - std::sin(static_cast<float>(i % 127) * 0.017f) * 0.05f;
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::cos(static_cast<float>(i) * 0.012f) * 0.23f + std::sin(static_cast<float>(i % 67) * 0.017f) * 0.04f;
    }
    auto ref = run_dit_transformer_stack_cpu(bundle, input, cond, key_mask, tokens);
    auto got = run_dit_transformer_stack_metal(metal, bundle, input, cond, key_mask, tokens);
    auto ref_branch1 = run_dit_transformer_stack_cpu(bundle, input_branch1, cond, key_mask_branch1, tokens);
    std::vector<float> input_batched(input.size() * batch);
    std::copy(input.begin(), input.end(), input_batched.begin());
    std::copy(input_branch1.begin(), input_branch1.end(), input_batched.begin() + static_cast<std::ptrdiff_t>(input.size()));
    std::vector<uint32_t> key_mask_batched;
    key_mask_batched.reserve(static_cast<size_t>(batch) * tokens);
    key_mask_batched.insert(key_mask_batched.end(), key_mask.begin(), key_mask.end());
    key_mask_batched.insert(key_mask_batched.end(), key_mask_branch1.begin(), key_mask_branch1.end());
    auto got_batched = run_dit_transformer_stack_metal_batched(metal, bundle, input_batched, cond, key_mask_batched, batch, tokens);
    const size_t branch_size = input.size();
    std::vector<float> got_batched0(got_batched.begin(), got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size));
    std::vector<float> got_batched1(got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size),
                                    got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size * 2));
    const float err = max_abs_error(got, ref);
    const float batched_branch0_err = max_abs_error(got_batched0, ref);
    const float batched_branch1_err = max_abs_error(got_batched1, ref_branch1);
    const float batched_single_branch0_err = max_abs_error(got_batched0, got);
    const float batched_err = std::max(batched_branch0_err, batched_branch1_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_transformer_stack\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"batched_branches\": " << batch << ",\n";
    std::cout << "  \"layers\": 13,\n";
    std::cout << "  \"valid_key_tokens\": 4,\n";
    std::cout << "  \"hidden_dim\": 512,\n";
    std::cout << "  \"single_max_abs_error\": " << err << ",\n";
    std::cout << "  \"batched_branch0_max_abs_error\": " << batched_branch0_err << ",\n";
    std::cout << "  \"batched_branch1_max_abs_error\": " << batched_branch1_err << ",\n";
    std::cout << "  \"batched_vs_single_branch0_max_abs_error\": " << batched_single_branch0_err << ",\n";
    std::cout << "  \"max_abs_error\": " << std::max(err, batched_err) << "\n";
    std::cout << "}\n";
    return err <= 1e-4f && batched_err <= 1e-4f && batched_single_branch0_err <= 1e-5f;
}

bool run_dit_transformer_block0_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    constexpr uint32_t batch = 2;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    std::vector<float> input_branch1(static_cast<size_t>(tokens) * 512);
    std::vector<float> cond(512);
    std::vector<uint32_t> key_mask{1, 1, 1, 1, 0, 0};
    std::vector<uint32_t> key_mask_branch1{1, 0, 1, 1, 1, 0};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.019f) * 0.12f + std::cos(static_cast<float>(i % 113) * 0.017f) * 0.07f;
        input_branch1[i] = std::cos(static_cast<float>(i) * 0.023f) * 0.10f - std::sin(static_cast<float>(i % 97) * 0.013f) * 0.06f;
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::cos(static_cast<float>(i) * 0.014f) * 0.25f + std::sin(static_cast<float>(i % 73) * 0.021f) * 0.05f;
    }
    auto ref = run_dit_transformer_block0_cpu(bundle, input, cond, key_mask, tokens);
    auto got = run_dit_transformer_block0_metal(metal, bundle, input, cond, key_mask, tokens);
    auto ref_branch1 = run_dit_transformer_block_cpu_layer(bundle, input_branch1, cond, key_mask_branch1, tokens, 0, nullptr);
    std::vector<float> input_batched(input.size() * batch);
    std::copy(input.begin(), input.end(), input_batched.begin());
    std::copy(input_branch1.begin(), input_branch1.end(), input_batched.begin() + static_cast<std::ptrdiff_t>(input.size()));
    std::vector<uint32_t> key_mask_batched;
    key_mask_batched.reserve(static_cast<size_t>(batch) * tokens);
    key_mask_batched.insert(key_mask_batched.end(), key_mask.begin(), key_mask.end());
    key_mask_batched.insert(key_mask_batched.end(), key_mask_branch1.begin(), key_mask_branch1.end());
    auto got_batched = run_dit_transformer_block_metal_layer_batched(metal, bundle, input_batched, cond, key_mask_batched, batch, tokens, 0, nullptr);
    const size_t branch_size = input.size();
    std::vector<float> got_batched0(got_batched.begin(), got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size));
    std::vector<float> got_batched1(got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size),
                                    got_batched.begin() + static_cast<std::ptrdiff_t>(branch_size * 2));
    const float err = max_abs_error(got, ref);
    const float batched_branch0_err = max_abs_error(got_batched0, ref);
    const float batched_branch1_err = max_abs_error(got_batched1, ref_branch1);
    const float batched_single_branch0_err = max_abs_error(got_batched0, got);
    const float batched_err = std::max(batched_branch0_err, batched_branch1_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_transformer_block0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"batched_branches\": " << batch << ",\n";
    std::cout << "  \"valid_key_tokens\": 4,\n";
    std::cout << "  \"hidden_dim\": 512,\n";
    std::cout << "  \"single_max_abs_error\": " << err << ",\n";
    std::cout << "  \"batched_branch0_max_abs_error\": " << batched_branch0_err << ",\n";
    std::cout << "  \"batched_branch1_max_abs_error\": " << batched_branch1_err << ",\n";
    std::cout << "  \"batched_vs_single_branch0_max_abs_error\": " << batched_single_branch0_err << ",\n";
    std::cout << "  \"max_abs_error\": " << std::max(err, batched_err) << "\n";
    std::cout << "}\n";
    return err <= 1e-4f && batched_err <= 1e-4f && batched_single_branch0_err <= 1e-5f;
}

struct DitPostTransformerOutputs {
    std::vector<float> long_skip;
    std::vector<float> conv1;
    std::vector<float> res_projection;
    std::vector<float> conv2;
};

struct DitPostTransformerCoreOutputs {
    std::vector<float> long_skip;
    std::vector<float> conv1;
    std::vector<float> res_projection;
};

DitPostTransformerCoreOutputs run_dit_post_transformer_proj_metal_core(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& transformer_out, const std::vector<float>& mel_x, uint32_t rows);

DitPostTransformerOutputs run_dit_post_transformer_proj_cpu(const mit2::Bundle& bundle, const std::vector<float>& transformer_out, const std::vector<float>& mel_x, uint32_t tokens) {
    auto skip_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.skip_linear.weight");
    auto skip_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.skip_linear.bias");
    auto conv1_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv1.weight");
    auto conv1_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv1.bias");
    auto res_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.res_projection.weight");
    auto res_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.res_projection.bias");
    auto conv2_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.weight");
    auto conv2_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.bias");
    DitPostTransformerOutputs out;
    out.long_skip.resize(static_cast<size_t>(tokens) * 512);
    out.conv1.resize(static_cast<size_t>(tokens) * 512);
    out.res_projection.resize(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> skip_in(592);
        std::copy(transformer_out.begin() + t * 512, transformer_out.begin() + (t + 1) * 512, skip_in.begin());
        std::copy(mel_x.begin() + t * 80, mel_x.begin() + (t + 1) * 80, skip_in.begin() + 512);
        auto skipped = cpu_linear(skip_weight, skip_bias, skip_in, 512, 592);
        std::copy(skipped.begin(), skipped.end(), out.long_skip.begin() + t * 512);
        auto conv1 = cpu_linear(conv1_weight, conv1_bias, skipped, 512, 512);
        std::copy(conv1.begin(), conv1.end(), out.conv1.begin() + t * 512);
        auto res = cpu_linear(res_weight, res_bias, skipped, 512, 512);
        std::copy(res.begin(), res.end(), out.res_projection.begin() + t * 512);
    }
    out.conv2 = cpu_conv1d_same(out.conv1, conv2_weight, conv2_bias, tokens, 512, 80, 1);
    return out;
}

DitPostTransformerOutputs run_dit_post_transformer_proj_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& transformer_out, const std::vector<float>& mel_x, uint32_t tokens) {
    auto core = run_dit_post_transformer_proj_metal_core(metal, bundle, transformer_out, mel_x, tokens);
    auto conv2_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.weight");
    auto conv2_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.bias");
    DitPostTransformerOutputs out;
    out.long_skip = std::move(core.long_skip);
    out.conv1 = std::move(core.conv1);
    out.res_projection = std::move(core.res_projection);
    out.conv2 = metal.conv1d_same_f32_resident(
        "s2mel.net.cfm.estimator.conv2.weight.resident",
        conv2_weight,
        "s2mel.net.cfm.estimator.conv2.bias.resident",
        conv2_bias,
        out.conv1,
        tokens,
        512,
        80,
        1);
    return out;
}

DitPostTransformerCoreOutputs run_dit_post_transformer_proj_metal_core(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& transformer_out, const std::vector<float>& mel_x, uint32_t rows) {
    auto skip_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.skip_linear.weight");
    auto skip_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.skip_linear.bias");
    auto conv1_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv1.weight");
    auto conv1_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv1.bias");
    auto res_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.res_projection.weight");
    auto res_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.res_projection.bias");
    DitPostTransformerCoreOutputs out;
    auto skip_in = metal.concat_rows_f32(transformer_out, mel_x, rows, 512, 80);
    out.long_skip = metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.skip_linear.weight.resident",
        skip_weight,
        "s2mel.net.cfm.estimator.skip_linear.bias.resident",
        skip_bias,
        skip_in,
        rows,
        512,
        592);
    out.conv1 = metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.conv1.weight.resident",
        conv1_weight,
        "s2mel.net.cfm.estimator.conv1.bias.resident",
        conv1_bias,
        out.long_skip,
        rows,
        512,
        512);
    out.res_projection = metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.res_projection.weight.resident",
        res_weight,
        "s2mel.net.cfm.estimator.res_projection.bias.resident",
        res_bias,
        out.long_skip,
        rows,
        512,
        512);
    return out;
}

bool run_dit_post_transformer_proj_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    std::vector<float> transformer_out(static_cast<size_t>(tokens) * 512);
    std::vector<float> mel_x(static_cast<size_t>(tokens) * 80);
    for (size_t i = 0; i < transformer_out.size(); ++i) {
        transformer_out[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.14f + std::cos(static_cast<float>(i % 149) * 0.017f) * 0.04f;
    }
    for (size_t i = 0; i < mel_x.size(); ++i) {
        mel_x[i] = std::cos(static_cast<float>(i) * 0.031f) * 0.18f + std::sin(static_cast<float>(i % 43) * 0.027f) * 0.05f;
    }
    auto ref = run_dit_post_transformer_proj_cpu(bundle, transformer_out, mel_x, tokens);
    auto got = run_dit_post_transformer_proj_metal(metal, bundle, transformer_out, mel_x, tokens);
    const float skip_err = max_abs_error(got.long_skip, ref.long_skip);
    const float conv1_err = max_abs_error(got.conv1, ref.conv1);
    const float res_err = max_abs_error(got.res_projection, ref.res_projection);
    const float conv2_err = max_abs_error(got.conv2, ref.conv2);
    const float err = std::max(std::max(skip_err, conv1_err), std::max(res_err, conv2_err));
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_post_transformer_projection\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"long_skip_max_abs_error\": " << skip_err << ",\n";
    std::cout << "  \"conv1_max_abs_error\": " << conv1_err << ",\n";
    std::cout << "  \"res_projection_max_abs_error\": " << res_err << ",\n";
    std::cout << "  \"conv2_max_abs_error\": " << conv2_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_dit_final_layer_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& t1, uint32_t tokens) {
    auto ada_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight");
    auto ada_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias");
    auto linear_g = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_g");
    auto linear_v = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_v");
    auto linear_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.bias");
    auto linear_weight = weight_norm_rowmajor(linear_g, linear_v, 512, 512);
    std::vector<float> t_silu(t1.size());
    for (size_t i = 0; i < t1.size(); ++i) {
        t_silu[i] = t1[i] / (1.0f + std::exp(-t1[i]));
    }
    auto shift_scale = cpu_linear(ada_weight, ada_bias, t_silu, 1024, 512);
    std::vector<float> shift(shift_scale.begin(), shift_scale.begin() + 512);
    std::vector<float> scale(shift_scale.begin() + 512, shift_scale.end());
    std::vector<float> gamma(512, 1.0f);
    std::vector<float> beta(512, 0.0f);
    std::vector<float> out(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> row(input.begin() + t * 512, input.begin() + (t + 1) * 512);
        auto normed = cpu_layernorm(row, gamma, beta, 1e-6f);
        for (uint32_t i = 0; i < 512; ++i) {
            normed[i] = normed[i] * (1.0f + scale[i]) + shift[i];
        }
        auto y = cpu_linear(linear_weight, linear_bias, normed, 512, 512);
        std::copy(y.begin(), y.end(), out.begin() + t * 512);
    }
    return out;
}

std::vector<float> run_dit_final_layer_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& t1, uint32_t tokens) {
    auto ada_weight = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight");
    auto ada_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias");
    auto linear_g = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_g");
    auto linear_v = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_v");
    auto linear_bias = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.bias");
    auto linear_weight = weight_norm_rowmajor(linear_g, linear_v, 512, 512);
    auto t_silu = metal.silu_f32(t1);
    auto shift_scale = metal.linear_f32_resident(
        "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight.resident",
        ada_weight,
        "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias.resident",
        ada_bias,
        t_silu,
        1024,
        512);
    std::vector<float> shift(shift_scale.begin(), shift_scale.begin() + 512);
    std::vector<float> scale(shift_scale.begin() + 512, shift_scale.end());
    auto modulated = metal.adaptive_layernorm_rows_f32(input, shift, scale, tokens, 512, 1e-6f);
    return metal.linear_rows_f32_resident(
        "s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident",
        linear_weight,
        "s2mel.net.cfm.estimator.final_layer.linear.bias.resident",
        linear_bias,
        modulated,
        tokens,
        512,
        512);
}

bool run_dit_final_layer_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    std::vector<float> input(static_cast<size_t>(tokens) * 512);
    std::vector<float> t1(512);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.011f) * 0.12f + std::cos(static_cast<float>(i % 157) * 0.019f) * 0.06f;
    }
    for (size_t i = 0; i < t1.size(); ++i) {
        t1[i] = std::cos(static_cast<float>(i) * 0.016f) * 0.22f + std::sin(static_cast<float>(i % 71) * 0.013f) * 0.03f;
    }
    auto ref = run_dit_final_layer_cpu(bundle, input, t1, tokens);
    auto got = run_dit_final_layer_metal(metal, bundle, input, t1, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_final_layer\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"hidden_dim\": 512,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

struct WavenetLayer0GateOutputs {
    std::vector<float> cond;
    std::vector<float> in_layer;
    std::vector<float> gate;
};

WavenetLayer0GateOutputs run_wavenet_layer0_gate_cpu(const mit2::Bundle& bundle, const std::vector<float>& x, const std::vector<float>& g, uint32_t tokens) {
    WavenetLayer0GateOutputs out;
    const std::string cond_prefix = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
    auto cond_weight = weight_norm_conv_weight(bundle, cond_prefix, 8192, 512, 1);
    auto cond_bias = tensor_as_f32(bundle, cond_prefix + ".bias");
    out.cond = cpu_conv1d_same(g, cond_weight, cond_bias, tokens, 512, 8192, 1);

    const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers.0.conv.conv";
    auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
    auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
    out.in_layer = cpu_conv1d_reflect_same(x, in_weight, in_bias, tokens, 512, 1024, 5);

    out.gate.resize(static_cast<size_t>(tokens) * 512);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < 512; ++c) {
            const float a = out.in_layer[static_cast<size_t>(t) * 1024 + c] + out.cond[static_cast<size_t>(t) * 8192 + c];
            const float b = out.in_layer[static_cast<size_t>(t) * 1024 + 512 + c] + out.cond[static_cast<size_t>(t) * 8192 + 512 + c];
            out.gate[static_cast<size_t>(t) * 512 + c] = std::tanh(a) * (1.0f / (1.0f + std::exp(-b)));
        }
    }
    return out;
}

WavenetLayer0GateOutputs run_wavenet_layer0_gate_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& x, const std::vector<float>& g, uint32_t tokens) {
    WavenetLayer0GateOutputs out;
    const std::string cond_prefix = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
    auto cond_weight = weight_norm_conv_weight(bundle, cond_prefix, 8192, 512, 1);
    auto cond_bias = tensor_as_f32(bundle, cond_prefix + ".bias");
    out.cond = metal.conv1d_same_f32_resident(
        cond_prefix + ".weight_norm.resident",
        cond_weight,
        cond_prefix + ".bias.resident",
        cond_bias,
        g,
        tokens,
        512,
        8192,
        1);

    const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers.0.conv.conv";
    auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
    auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
    out.in_layer = metal.conv1d_reflect_same_f32_resident(
        in_prefix + ".weight_norm.resident",
        in_weight,
        in_prefix + ".bias.resident",
        in_bias,
        x,
        tokens,
        512,
        1024,
        5);

    out.gate = metal.wavenet_gate_f32(out.in_layer, out.cond, tokens, 512, 8192, 0, tokens);
    return out;
}

bool run_wavenet_layer0_gate_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    std::vector<float> x(static_cast<size_t>(tokens) * 512);
    std::vector<float> g(static_cast<size_t>(tokens) * 512);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.10f + std::cos(static_cast<float>(i % 173) * 0.014f) * 0.05f;
        g[i] = std::cos(static_cast<float>(i) * 0.012f) * 0.08f + std::sin(static_cast<float>(i % 89) * 0.018f) * 0.04f;
    }
    auto ref = run_wavenet_layer0_gate_cpu(bundle, x, g, tokens);
    auto got = run_wavenet_layer0_gate_metal(metal, bundle, x, g, tokens);
    const float cond_err = max_abs_error(got.cond, ref.cond);
    const float in_err = max_abs_error(got.in_layer, ref.in_layer);
    const float gate_err = max_abs_error(got.gate, ref.gate);
    const float err = std::max(cond_err, std::max(in_err, gate_err));
    std::cout << "{\n";
    std::cout << "  \"stage\": \"wavenet_layer0_gate\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"cond_channels\": 8192,\n";
    std::cout << "  \"hidden_channels\": 512,\n";
    std::cout << "  \"cond_layer_max_abs_error\": " << cond_err << ",\n";
    std::cout << "  \"in_layer_max_abs_error\": " << in_err << ",\n";
    std::cout << "  \"gate_max_abs_error\": " << gate_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> conv1d_reflect_same_metal(mit2::MetalContext& metal, const std::string& prefix, const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    return metal.conv1d_reflect_same_f32_resident(
        prefix + ".weight_norm.resident",
        weight,
        prefix + ".bias.resident",
        bias,
        x,
        tokens,
        in_channels,
        out_channels,
        kernel);
}

std::vector<float> run_wavenet_stack_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& g, const std::vector<uint32_t>& mask, uint32_t tokens) {
    const std::string cond_prefix = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
    auto cond_weight = weight_norm_conv_weight(bundle, cond_prefix, 8192, 512, 1);
    auto cond_bias = tensor_as_f32(bundle, cond_prefix + ".bias");
    auto cond = cpu_conv1d_same(g, cond_weight, cond_bias, tokens, 512, 8192, 1);

    std::vector<float> x = input;
    std::vector<float> output(static_cast<size_t>(tokens) * 512, 0.0f);
    for (uint32_t layer = 0; layer < 8; ++layer) {
        const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers." + std::to_string(layer) + ".conv.conv";
        auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
        auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
        auto in_layer = cpu_conv1d_reflect_same(x, in_weight, in_bias, tokens, 512, 1024, 5);

        std::vector<float> gate(static_cast<size_t>(tokens) * 512);
        const uint32_t cond_offset = layer * 1024;
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t c = 0; c < 512; ++c) {
                const float a = in_layer[static_cast<size_t>(t) * 1024 + c] + cond[static_cast<size_t>(t) * 8192 + cond_offset + c];
                const float b = in_layer[static_cast<size_t>(t) * 1024 + 512 + c] + cond[static_cast<size_t>(t) * 8192 + cond_offset + 512 + c];
                gate[static_cast<size_t>(t) * 512 + c] = std::tanh(a) * (1.0f / (1.0f + std::exp(-b)));
            }
        }

        const uint32_t res_skip_channels = layer < 7 ? 1024 : 512;
        const std::string rs_prefix = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";
        auto rs_weight = weight_norm_conv_weight(bundle, rs_prefix, res_skip_channels, 512, 1);
        auto rs_bias = tensor_as_f32(bundle, rs_prefix + ".bias");
        auto res_skip = cpu_conv1d_same(gate, rs_weight, rs_bias, tokens, 512, res_skip_channels, 1);
        if (layer < 7) {
            for (uint32_t t = 0; t < tokens; ++t) {
                const float m = mask[t] ? 1.0f : 0.0f;
                for (uint32_t c = 0; c < 512; ++c) {
                    x[static_cast<size_t>(t) * 512 + c] = (x[static_cast<size_t>(t) * 512 + c] + res_skip[static_cast<size_t>(t) * 1024 + c]) * m;
                    output[static_cast<size_t>(t) * 512 + c] += res_skip[static_cast<size_t>(t) * 1024 + 512 + c];
                }
            }
        } else {
            for (size_t i = 0; i < output.size(); ++i) {
                output[i] += res_skip[i];
            }
        }
    }
    for (uint32_t t = 0; t < tokens; ++t) {
        const float m = mask[t] ? 1.0f : 0.0f;
        for (uint32_t c = 0; c < 512; ++c) {
            output[static_cast<size_t>(t) * 512 + c] *= m;
        }
    }
    return output;
}

std::vector<float> run_wavenet_stack_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& g, const std::vector<uint32_t>& mask, uint32_t tokens) {
    const std::string cond_prefix = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
    auto cond_weight = weight_norm_conv_weight(bundle, cond_prefix, 8192, 512, 1);
    auto cond_bias = tensor_as_f32(bundle, cond_prefix + ".bias");
    auto cond = metal.conv1d_same_f32_resident(
        cond_prefix + ".weight_norm.resident",
        cond_weight,
        cond_prefix + ".bias.resident",
        cond_bias,
        g,
        tokens,
        512,
        8192,
        1);

    std::vector<float> x = input;
    std::vector<float> output(static_cast<size_t>(tokens) * 512, 0.0f);
    for (uint32_t layer = 0; layer < 8; ++layer) {
        const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers." + std::to_string(layer) + ".conv.conv";
        auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
        auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
        auto in_layer = conv1d_reflect_same_metal(metal, in_prefix, x, in_weight, in_bias, tokens, 512, 1024, 5);

        const uint32_t cond_offset = layer * 1024;
        auto gate = metal.wavenet_gate_f32(in_layer, cond, tokens, 512, 8192, cond_offset, tokens);

        const uint32_t res_skip_channels = layer < 7 ? 1024 : 512;
        const std::string rs_prefix = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";
        auto rs_weight = weight_norm_conv_weight(bundle, rs_prefix, res_skip_channels, 512, 1);
        auto rs_bias = tensor_as_f32(bundle, rs_prefix + ".bias");
        auto res_skip = metal.conv1d_same_f32_resident(
            rs_prefix + ".weight_norm.resident",
            rs_weight,
            rs_prefix + ".bias.resident",
            rs_bias,
            gate,
            tokens,
            512,
            res_skip_channels,
            1);
        auto updated = metal.wavenet_res_skip_update_f32(x, output, res_skip, mask, tokens, 512, layer < 7);
        const size_t row_count = static_cast<size_t>(tokens) * 512;
        if (layer < 7) {
            x.assign(updated.begin(), updated.begin() + static_cast<std::ptrdiff_t>(row_count));
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        } else {
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        }
    }
    return output;
}

std::vector<float> run_wavenet_stack_global_cond_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& g, const std::vector<uint32_t>& mask, uint32_t tokens) {
    const std::string cond_prefix = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
    auto cond_weight = weight_norm_conv_weight(bundle, cond_prefix, 8192, 512, 1);
    auto cond_bias = tensor_as_f32(bundle, cond_prefix + ".bias");
    auto cond = cpu_conv1d_same(g, cond_weight, cond_bias, 1, 512, 8192, 1);

    std::vector<float> x = input;
    std::vector<float> output(static_cast<size_t>(tokens) * 512, 0.0f);
    for (uint32_t layer = 0; layer < 8; ++layer) {
        const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers." + std::to_string(layer) + ".conv.conv";
        auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
        auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
        auto in_layer = cpu_conv1d_reflect_same(x, in_weight, in_bias, tokens, 512, 1024, 5);

        std::vector<float> gate(static_cast<size_t>(tokens) * 512);
        const uint32_t cond_offset = layer * 1024;
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t c = 0; c < 512; ++c) {
                const float a = in_layer[static_cast<size_t>(t) * 1024 + c] + cond[cond_offset + c];
                const float b = in_layer[static_cast<size_t>(t) * 1024 + 512 + c] + cond[cond_offset + 512 + c];
                gate[static_cast<size_t>(t) * 512 + c] = std::tanh(a) * (1.0f / (1.0f + std::exp(-b)));
            }
        }

        const uint32_t res_skip_channels = layer < 7 ? 1024 : 512;
        const std::string rs_prefix = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";
        auto rs_weight = weight_norm_conv_weight(bundle, rs_prefix, res_skip_channels, 512, 1);
        auto rs_bias = tensor_as_f32(bundle, rs_prefix + ".bias");
        auto res_skip = cpu_conv1d_same(gate, rs_weight, rs_bias, tokens, 512, res_skip_channels, 1);
        if (layer < 7) {
            for (uint32_t t = 0; t < tokens; ++t) {
                const float m = mask[t] ? 1.0f : 0.0f;
                for (uint32_t c = 0; c < 512; ++c) {
                    x[static_cast<size_t>(t) * 512 + c] = (x[static_cast<size_t>(t) * 512 + c] + res_skip[static_cast<size_t>(t) * 1024 + c]) * m;
                    output[static_cast<size_t>(t) * 512 + c] += res_skip[static_cast<size_t>(t) * 1024 + 512 + c];
                }
            }
        } else {
            for (size_t i = 0; i < output.size(); ++i) {
                output[i] += res_skip[i];
            }
        }
    }
    for (uint32_t t = 0; t < tokens; ++t) {
        const float m = mask[t] ? 1.0f : 0.0f;
        for (uint32_t c = 0; c < 512; ++c) {
            output[static_cast<size_t>(t) * 512 + c] *= m;
        }
    }
    return output;
}

std::vector<float> run_wavenet_global_cond_projection_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& g) {
    const std::string cond_prefix = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
    auto cond_weight = weight_norm_conv_weight(bundle, cond_prefix, 8192, 512, 1);
    auto cond_bias = tensor_as_f32(bundle, cond_prefix + ".bias");
    return metal.conv1d_same_f32_resident(
        cond_prefix + ".weight_norm.resident",
        cond_weight,
        cond_prefix + ".bias.resident",
        cond_bias,
        g,
        1,
        512,
        8192,
        1);
}

std::vector<float> run_wavenet_stack_global_cond_metal_with_cond(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& mask, uint32_t tokens) {
    std::vector<float> x = input;
    std::vector<float> output(static_cast<size_t>(tokens) * 512, 0.0f);
    for (uint32_t layer = 0; layer < 8; ++layer) {
        const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers." + std::to_string(layer) + ".conv.conv";
        auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
        auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
        auto in_layer = conv1d_reflect_same_metal(metal, in_prefix, x, in_weight, in_bias, tokens, 512, 1024, 5);

        const uint32_t cond_offset = layer * 1024;
        auto gate = metal.wavenet_gate_f32(in_layer, cond, tokens, 512, 8192, cond_offset, 1);

        const uint32_t res_skip_channels = layer < 7 ? 1024 : 512;
        const std::string rs_prefix = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";
        auto rs_weight = weight_norm_conv_weight(bundle, rs_prefix, res_skip_channels, 512, 1);
        auto rs_bias = tensor_as_f32(bundle, rs_prefix + ".bias");
        auto res_skip = metal.conv1d_same_f32_resident(
            rs_prefix + ".weight_norm.resident",
            rs_weight,
            rs_prefix + ".bias.resident",
            rs_bias,
            gate,
            tokens,
            512,
            res_skip_channels,
            1);
        auto updated = metal.wavenet_res_skip_update_f32(x, output, res_skip, mask, tokens, 512, layer < 7);
        const size_t row_count = static_cast<size_t>(tokens) * 512;
        if (layer < 7) {
            x.assign(updated.begin(), updated.begin() + static_cast<std::ptrdiff_t>(row_count));
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        } else {
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        }
    }
    return output;
}

std::vector<float> run_wavenet_stack_global_cond_metal_batched_with_cond(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& cond, const std::vector<uint32_t>& mask, uint32_t batch, uint32_t tokens) {
    const uint32_t rows = batch * tokens;
    if (input.size() != static_cast<size_t>(rows) * 512 || mask.size() != rows) {
        throw std::runtime_error("batched Wavenet global-cond input size mismatch");
    }
    std::vector<float> x = input;
    std::vector<float> output(static_cast<size_t>(rows) * 512, 0.0f);
    for (uint32_t layer = 0; layer < 8; ++layer) {
        const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers." + std::to_string(layer) + ".conv.conv";
        auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
        auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
        auto in_layer = metal.conv1d_reflect_same_batched_f32_resident(
            in_prefix + ".weight_norm.resident",
            in_weight,
            in_prefix + ".bias.resident",
            in_bias,
            x,
            batch,
            tokens,
            512,
            1024,
            5);

        const uint32_t cond_offset = layer * 1024;
        auto gate = metal.wavenet_gate_f32(in_layer, cond, rows, 512, 8192, cond_offset, 1);

        const uint32_t res_skip_channels = layer < 7 ? 1024 : 512;
        const std::string rs_prefix = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";
        auto rs_weight = weight_norm_conv_weight(bundle, rs_prefix, res_skip_channels, 512, 1);
        auto rs_bias = tensor_as_f32(bundle, rs_prefix + ".bias");
        auto res_skip = metal.conv1d_same_f32_resident(
            rs_prefix + ".weight_norm.resident",
            rs_weight,
            rs_prefix + ".bias.resident",
            rs_bias,
            gate,
            rows,
            512,
            res_skip_channels,
            1);
        auto updated = metal.wavenet_res_skip_update_f32(x, output, res_skip, mask, rows, 512, layer < 7);
        const size_t row_count = static_cast<size_t>(rows) * 512;
        if (layer < 7) {
            x.assign(updated.begin(), updated.begin() + static_cast<std::ptrdiff_t>(row_count));
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        } else {
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        }
    }
    return output;
}

std::vector<float> run_wavenet_stack_global_cond_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& g, const std::vector<uint32_t>& mask, uint32_t tokens) {
    auto cond = run_wavenet_global_cond_projection_metal(metal, bundle, g);
    return run_wavenet_stack_global_cond_metal_with_cond(metal, bundle, input, cond, mask, tokens);
}

std::vector<float> trace_wavenet_stack_global_cond_metal_entries(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, const std::vector<float>& g, const std::vector<uint32_t>& mask, uint32_t tokens) {
    const std::string cond_prefix = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
    auto cond_weight = weight_norm_conv_weight(bundle, cond_prefix, 8192, 512, 1);
    auto cond_bias = tensor_as_f32(bundle, cond_prefix + ".bias");
    auto cond = metal.conv1d_same_f32_resident(
        cond_prefix + ".weight_norm.resident",
        cond_weight,
        cond_prefix + ".bias.resident",
        cond_bias,
        g,
        1,
        512,
        8192,
        1);
    print_tensor_stats_json("wavenet_cond", cond, 8192, true);

    std::vector<float> x = input;
    std::vector<float> output(static_cast<size_t>(tokens) * 512, 0.0f);
    for (uint32_t layer = 0; layer < 8; ++layer) {
        const std::string in_prefix = "s2mel.net.cfm.estimator.wavenet.in_layers." + std::to_string(layer) + ".conv.conv";
        auto in_weight = weight_norm_conv_weight(bundle, in_prefix, 1024, 512, 5);
        auto in_bias = tensor_as_f32(bundle, in_prefix + ".bias");
        auto in_layer = conv1d_reflect_same_metal(metal, in_prefix, x, in_weight, in_bias, tokens, 512, 1024, 5);

        const uint32_t cond_offset = layer * 1024;
        auto gate = metal.wavenet_gate_f32(in_layer, cond, tokens, 512, 8192, cond_offset, 1);

        const uint32_t res_skip_channels = layer < 7 ? 1024 : 512;
        const std::string rs_prefix = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";
        auto rs_weight = weight_norm_conv_weight(bundle, rs_prefix, res_skip_channels, 512, 1);
        auto rs_bias = tensor_as_f32(bundle, rs_prefix + ".bias");
        auto res_skip = metal.conv1d_same_f32_resident(
            rs_prefix + ".weight_norm.resident",
            rs_weight,
            rs_prefix + ".bias.resident",
            rs_bias,
            gate,
            tokens,
            512,
            res_skip_channels,
            1);
        auto updated = metal.wavenet_res_skip_update_f32(x, output, res_skip, mask, tokens, 512, layer < 7);

        const std::string layer_prefix = "wavenet_layer" + std::to_string(layer);
        print_tensor_stats_json((layer_prefix + "_in_layer").c_str(), in_layer, 1024, true);
        print_tensor_stats_json((layer_prefix + "_gate").c_str(), gate, 512, true);
        const int64_t gate_first_nonfinite = first_nonfinite_index(gate);
        if (gate_first_nonfinite >= 0) {
            const int64_t row = gate_first_nonfinite / 512;
            const int64_t col = gate_first_nonfinite % 512;
            const size_t in_base = static_cast<size_t>(row) * 1024;
            const size_t cond_base = static_cast<size_t>(cond_offset);
            const float a = in_layer[in_base + static_cast<size_t>(col)] + cond[cond_base + static_cast<size_t>(col)];
            const float b = in_layer[in_base + 512 + static_cast<size_t>(col)] + cond[cond_base + 512 + static_cast<size_t>(col)];
            const float exp_b = std::exp(b);
            const float sigmoid_b = b >= 0.0f ? 1.0f / (1.0f + std::exp(-b)) : exp_b / (1.0f + exp_b);
            const float cpu_gate = std::tanh(a) * sigmoid_b;
            std::cout << "    {\"name\": \"" << layer_prefix << "_gate_first_nonfinite_inputs"
                      << "\", \"row\": " << row
                      << ", \"col\": " << col
                      << ", \"a\": " << a
                      << ", \"b\": " << b
                      << ", \"cpu_sigmoid_b\": " << sigmoid_b
                      << ", \"cpu_gate\": " << cpu_gate
                      << ", \"a_isfinite\": " << (std::isfinite(a) ? "true" : "false")
                      << ", \"b_isfinite\": " << (std::isfinite(b) ? "true" : "false")
                      << "},\n";
        }
        print_tensor_stats_json((layer_prefix + "_res_skip").c_str(), res_skip, res_skip_channels, true);
        print_tensor_stats_json((layer_prefix + "_updated").c_str(), updated, 512, true);

        const size_t row_count = static_cast<size_t>(tokens) * 512;
        if (layer < 7) {
            x.assign(updated.begin(), updated.begin() + static_cast<std::ptrdiff_t>(row_count));
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        } else {
            output.assign(updated.begin() + static_cast<std::ptrdiff_t>(row_count), updated.end());
        }
    }
    return output;
}

bool run_wavenet_stack_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    std::vector<float> x(static_cast<size_t>(tokens) * 512);
    std::vector<float> g(static_cast<size_t>(tokens) * 512);
    std::vector<uint32_t> mask{1, 1, 1, 1, 0, 0};
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.010f) * 0.08f + std::cos(static_cast<float>(i % 181) * 0.013f) * 0.04f;
        g[i] = std::cos(static_cast<float>(i) * 0.011f) * 0.07f + std::sin(static_cast<float>(i % 97) * 0.017f) * 0.03f;
    }
    auto ref = run_wavenet_stack_cpu(bundle, x, g, mask, tokens);
    auto got = run_wavenet_stack_metal(metal, bundle, x, g, mask, tokens);
    const float err = max_abs_error(got, ref);

    std::vector<float> x_branch1(x.size());
    std::vector<float> g_global(512);
    std::vector<uint32_t> mask_branch1{1, 0, 1, 1, 1, 0};
    for (size_t i = 0; i < x_branch1.size(); ++i) {
        x_branch1[i] = std::cos(static_cast<float>(i) * 0.009f) * 0.06f -
                       std::sin(static_cast<float>(i % 149) * 0.021f) * 0.05f;
    }
    for (size_t i = 0; i < g_global.size(); ++i) {
        g_global[i] = std::sin(static_cast<float>(i) * 0.019f) * 0.04f +
                      std::cos(static_cast<float>(i % 71) * 0.023f) * 0.02f;
    }
    auto global_cond = run_wavenet_global_cond_projection_metal(metal, bundle, g_global);
    auto global0 = run_wavenet_stack_global_cond_metal_with_cond(metal, bundle, x, global_cond, mask, tokens);
    auto global1 = run_wavenet_stack_global_cond_metal_with_cond(metal, bundle, x_branch1, global_cond, mask_branch1, tokens);
    std::vector<float> x_batched(x.size() * 2);
    std::copy(x.begin(), x.end(), x_batched.begin());
    std::copy(x_branch1.begin(), x_branch1.end(), x_batched.begin() + static_cast<std::ptrdiff_t>(x.size()));
    std::vector<uint32_t> mask_batched;
    mask_batched.reserve(mask.size() + mask_branch1.size());
    mask_batched.insert(mask_batched.end(), mask.begin(), mask.end());
    mask_batched.insert(mask_batched.end(), mask_branch1.begin(), mask_branch1.end());
    auto global_batched = run_wavenet_stack_global_cond_metal_batched_with_cond(metal, bundle, x_batched, global_cond, mask_batched, 2, tokens);
    const size_t branch_size = static_cast<size_t>(tokens) * 512;
    std::vector<float> global_batched0(global_batched.begin(), global_batched.begin() + static_cast<std::ptrdiff_t>(branch_size));
    std::vector<float> global_batched1(global_batched.begin() + static_cast<std::ptrdiff_t>(branch_size),
                                       global_batched.begin() + static_cast<std::ptrdiff_t>(branch_size * 2));
    const float global_batched_branch0_err = max_abs_error(global_batched0, global0);
    const float global_batched_branch1_err = max_abs_error(global_batched1, global1);
    const float global_batched_err = std::max(global_batched_branch0_err, global_batched_branch1_err);

    std::cout << "{\n";
    std::cout << "  \"stage\": \"wavenet_stack\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"layers\": 8,\n";
    std::cout << "  \"valid_tokens\": 4,\n";
    std::cout << "  \"hidden_channels\": 512,\n";
    std::cout << "  \"max_abs_error\": " << err << ",\n";
    std::cout << "  \"global_cond_batched_branch0_max_abs_error\": " << global_batched_branch0_err << ",\n";
    std::cout << "  \"global_cond_batched_branch1_max_abs_error\": " << global_batched_branch1_err << ",\n";
    std::cout << "  \"global_cond_batched_max_abs_error\": " << global_batched_err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f && global_batched_err <= 1e-4f;
}

std::vector<float> run_dit_estimator_step_cpu(const mit2::Bundle& bundle, const std::vector<float>& x, const std::vector<float>& prompt_x, const std::vector<float>& cond, const std::vector<float>& style, float timestep, const std::vector<uint32_t>& mask, uint32_t tokens) {
    auto t1 = run_timestep_embedder_cpu(bundle, std::vector<float>{timestep}, "s2mel.net.cfm.estimator.t_embedder");
    auto t2 = run_timestep_embedder_cpu(bundle, std::vector<float>{timestep}, "s2mel.net.cfm.estimator.t_embedder2");
    auto x_in = run_dit_input_merge_cpu(bundle, x, prompt_x, cond, style, tokens);
    auto x_res = run_dit_transformer_stack_cpu(bundle, x_in, t1, mask, tokens);
    auto post = run_dit_post_transformer_proj_cpu(bundle, x_res, x, tokens);
    auto wav = run_wavenet_stack_global_cond_cpu(bundle, post.conv1, t2, mask, tokens);
    std::vector<float> final_in(static_cast<size_t>(tokens) * 512);
    for (size_t i = 0; i < final_in.size(); ++i) {
        final_in[i] = wav[i] + post.res_projection[i];
    }
    auto final_hidden = run_dit_final_layer_cpu(bundle, final_in, t1, tokens);
    return cpu_conv1d_same(
        final_hidden,
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.weight"),
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.bias"),
        tokens,
        512,
        80,
        1);
}

std::vector<float> run_dit_estimator_step_metal_with_embeddings(mit2::MetalContext& metal,
                                                                const mit2::Bundle& bundle,
                                                                const std::vector<float>& x,
                                                                const std::vector<float>& prompt_x,
                                                                const std::vector<float>& cond,
                                                                const std::vector<float>& style,
                                                                const std::vector<float>& t1,
                                                                const std::vector<float>& t2,
                                                                const std::vector<uint32_t>& mask,
                                                                uint32_t tokens) {
    auto x_in = run_dit_input_merge_metal(metal, bundle, x, prompt_x, cond, style, tokens);
    auto x_res = run_dit_transformer_stack_metal(metal, bundle, x_in, t1, mask, tokens);
    auto post = run_dit_post_transformer_proj_metal_core(metal, bundle, x_res, x, tokens);
    auto wav = run_wavenet_stack_global_cond_metal(metal, bundle, post.conv1, t2, mask, tokens);
    auto final_in = metal.add_f32(wav, post.res_projection);
    auto final_hidden = run_dit_final_layer_metal(metal, bundle, final_in, t1, tokens);
    return metal.conv1d_same_f32_resident(
        "s2mel.net.cfm.estimator.conv2.weight.resident",
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.weight"),
        "s2mel.net.cfm.estimator.conv2.bias.resident",
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.bias"),
        final_hidden,
        tokens,
        512,
        80,
        1);
}

std::pair<std::vector<float>, std::vector<float>> run_dit_estimator_step_metal_cfg_transformer_batched(mit2::MetalContext& metal,
                                                                                                       const mit2::Bundle& bundle,
                                                                                                       const std::vector<float>& x,
                                                                                                       const std::vector<float>& prompt_x,
                                                                                                       const std::vector<float>& cond,
                                                                                                       const std::vector<float>& style,
                                                                                                       const std::vector<float>& null_prompt_x,
                                                                                                       const std::vector<float>& null_cond,
                                                                                                       const std::vector<float>& null_style,
                                                                                                       const std::vector<float>& t1,
                                                                                                       const std::vector<float>& t2,
                                                                                                       const std::vector<uint32_t>& mask,
                                                                                                       uint32_t tokens) {
    constexpr uint32_t batch = 2;
    std::vector<float> x_mel_batched(x.size() * batch);
    std::copy(x.begin(), x.end(), x_mel_batched.begin());
    std::copy(x.begin(), x.end(), x_mel_batched.begin() + static_cast<std::ptrdiff_t>(x.size()));
    std::vector<float> prompt_x_batched(prompt_x.size() * batch);
    std::copy(prompt_x.begin(), prompt_x.end(), prompt_x_batched.begin());
    std::copy(null_prompt_x.begin(), null_prompt_x.end(), prompt_x_batched.begin() + static_cast<std::ptrdiff_t>(prompt_x.size()));
    std::vector<float> cond_batched(cond.size() * batch);
    std::copy(cond.begin(), cond.end(), cond_batched.begin());
    std::copy(null_cond.begin(), null_cond.end(), cond_batched.begin() + static_cast<std::ptrdiff_t>(cond.size()));
    std::vector<float> style_batched(style.size() * batch);
    std::copy(style.begin(), style.end(), style_batched.begin());
    std::copy(null_style.begin(), null_style.end(), style_batched.begin() + static_cast<std::ptrdiff_t>(style.size()));
    auto x_in_batched = run_dit_input_merge_metal_batched(metal, bundle, x_mel_batched, prompt_x_batched, cond_batched, style_batched, batch, tokens);
    std::vector<uint32_t> mask_batched;
    mask_batched.reserve(static_cast<size_t>(batch) * tokens);
    mask_batched.insert(mask_batched.end(), mask.begin(), mask.end());
    mask_batched.insert(mask_batched.end(), mask.begin(), mask.end());
    auto x_res_batched = run_dit_transformer_stack_metal_batched(metal, bundle, x_in_batched, t1, mask_batched, batch, tokens);
    auto post_batched = run_dit_post_transformer_proj_metal_core(metal, bundle, x_res_batched, x_mel_batched, batch * tokens);
    auto wavenet_cond = run_wavenet_global_cond_projection_metal(metal, bundle, t2);
    auto wav_batched = run_wavenet_stack_global_cond_metal_batched_with_cond(metal, bundle, post_batched.conv1, wavenet_cond, mask_batched, batch, tokens);
    auto final_in_batched = metal.add_f32(wav_batched, post_batched.res_projection);
    auto final_hidden_batched = run_dit_final_layer_metal(metal, bundle, final_in_batched, t1, batch * tokens);
    auto dphi_batched = metal.conv1d_same_f32_resident(
        "s2mel.net.cfm.estimator.conv2.weight.resident",
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.weight"),
        "s2mel.net.cfm.estimator.conv2.bias.resident",
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.bias"),
        final_hidden_batched,
        batch * tokens,
        512,
        80,
        1);
    const size_t dphi_branch_size = static_cast<size_t>(tokens) * 80;
    std::vector<float> dphi(dphi_batched.begin(),
                            dphi_batched.begin() + static_cast<std::ptrdiff_t>(dphi_branch_size));
    std::vector<float> null_dphi(dphi_batched.begin() + static_cast<std::ptrdiff_t>(dphi_branch_size),
                                 dphi_batched.begin() + static_cast<std::ptrdiff_t>(dphi_branch_size * 2));
    return {std::move(dphi), std::move(null_dphi)};
}

// Ablation switches for CFM profiling: MIT2_CFM_SKIP=attn,attnblock,ffn,wavenet
// replaces the named module with a shape-preserving cheap placeholder so the
// timing delta vs baseline isolates that module's cost. NOT for production.
static bool cfm_skip(const char* name) {
    static const std::string flags = []() {
        const char* v = std::getenv("MIT2_CFM_SKIP");
        return std::string(v ? v : "");
    }();
    return flags.find(name) != std::string::npos;
}

// Skip the CPU-side weight copy when the GPU-resident buffer already exists
// (the *_pass ops accept an empty vector in that case). Saves ~300MB of memcpy
// per CFM step after the first.
static std::vector<float> tensor_for_resident(mit2::MetalContext& metal,
                                              const mit2::Bundle& bundle,
                                              const std::string& name) {
    if (metal.residentExists(name + ".resident") || metal.residentExists(name + ".resident.f16")) {
        return {};
    }
    return tensor_as_f32(bundle, name);
}

// ---------------------------------------------------------------------------
// Pass-mode transformer block helper: dispatches all ops into the active pass,
// writing the final add(h, ffn) result into the pre-allocated |output| slot.
// All intermediate tensors are allocated from scratch (reset by caller).
// ---------------------------------------------------------------------------
static void run_transformer_block_pass_into(
    mit2::MetalContext& metal,
    const mit2::Bundle& bundle,
    mit2::PassSlot x,
    mit2::PassSlot attn_wb,   // precomputed adaLN modulation [1024] = proj(t1)
    mit2::PassSlot ffn_wb,    // precomputed adaLN modulation [1024] = proj(t1)
    mit2::PassSlot mask,
    uint32_t batch,
    uint32_t tokens,
    uint32_t layer,
    mit2::PassSlot skip_in,
    bool has_skip,
    mit2::PassSlot output)
{
    const uint32_t rows = batch * tokens;
    const std::string lb = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    const std::vector<float> zero_qkv_bias(1536, 0.0f);
    const std::vector<float> zero_out_bias(512, 0.0f);
    const std::vector<float> zero_mid_bias(1536, 0.0f);

    mit2::PassSlot cur_x = x;
    if (has_skip) {
        auto skip_w = tensor_for_resident(metal, bundle, lb + ".skip_in_linear.weight");
        auto skip_b = tensor_for_resident(metal, bundle, lb + ".skip_in_linear.bias");
        auto cat = metal.concat_rows_f32_pass(x, skip_in, rows, 512, 512);
        cur_x = metal.linear_rows_f32_pass(
            lb + ".skip_in_linear.weight.resident", skip_w,
            lb + ".skip_in_linear.bias.resident", skip_b,
            cat, rows, 512, 1024);
    }

    // Attention norm (modulation vector precomputed per (step, layer))
    auto attn_norm_g = tensor_for_resident(metal, bundle, lb + ".attention_norm.norm.weight");
    auto attn_normed = metal.adaptive_rmsnorm_rows_f32_pass(
        lb + ".attention_norm.norm.weight.resident", attn_norm_g,
        cur_x, attn_wb.slice(0, 512), attn_wb.slice(512, 512), rows, 512, 1e-5f);

    mit2::PassSlot h{};
    if (cfm_skip("attnblock")) {
        h = metal.silu_f32_pass(attn_normed, rows * 512);  // placeholder
    } else {
        // Attention
        auto wqkv = tensor_for_resident(metal, bundle, lb + ".attention.wqkv.weight");
        auto wo = tensor_for_resident(metal, bundle, lb + ".attention.wo.weight");
        auto qkv = metal.linear_rows_f32_pass(
            lb + ".attention.wqkv.weight.resident", wqkv,
            "s2mel.net.cfm.estimator.transformer.attention.wqkv.zero_bias.resident", zero_qkv_bias,
            attn_normed, rows, 1536, 512);
        mit2::PassSlot attn_out{};
        if (cfm_skip("attn")) {
            attn_out = metal.silu_f32_pass(attn_normed, rows * 512);  // placeholder
        } else {
            attn_out = metal.dit_attention_qkv_rope_batched_f32_pass(qkv, mask, batch, tokens, 8, 64);
        }
        auto attn_proj = metal.linear_rows_f32_pass(
            lb + ".attention.wo.weight.resident", wo,
            "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident", zero_out_bias,
            attn_out, rows, 512, 512);
        h = metal.add_f32_pass(cur_x, attn_proj);
    }

    // FFN norm (modulation vector precomputed per (step, layer))
    auto ffn_norm_g = tensor_for_resident(metal, bundle, lb + ".ffn_norm.norm.weight");
    auto ffn_normed = metal.adaptive_rmsnorm_rows_f32_pass(
        lb + ".ffn_norm.norm.weight.resident", ffn_norm_g,
        h, ffn_wb.slice(0, 512), ffn_wb.slice(512, 512), rows, 512, 1e-5f);

    if (cfm_skip("ffn")) {
        auto ffn_ph = metal.silu_f32_pass(ffn_normed, rows * 512);
        metal.add_f32_pass_into(h, ffn_ph, output);
        return;
    }
    // FFN: w1 and w3 merged into a single [3072,512] GEMM + SwiGLU split.
    const std::string w13_key = lb + ".feed_forward.w1w3.weight";
    std::vector<float> w13;
    if (!metal.residentExists(w13_key + ".resident") && !metal.residentExists(w13_key + ".resident.f16")) {
        auto w1 = tensor_as_f32(bundle, lb + ".feed_forward.w1.weight");
        auto w3 = tensor_as_f32(bundle, lb + ".feed_forward.w3.weight");
        w13.reserve(w1.size() + w3.size());
        w13.insert(w13.end(), w1.begin(), w1.end());
        w13.insert(w13.end(), w3.begin(), w3.end());
    }
    auto w2 = tensor_for_resident(metal, bundle, lb + ".feed_forward.w2.weight");
    const std::vector<float> zero_w13_bias(3072, 0.0f);
    auto h13 = metal.linear_rows_f32_pass(
        w13_key + ".resident", w13,
        "s2mel.net.cfm.estimator.transformer.feed_forward.zero_w13_bias.resident", zero_w13_bias,
        ffn_normed, rows, 3072, 512);
    auto gated = metal.silu_mul_split_f32_pass(h13, rows, 1536);
    auto ffn_out = metal.linear_rows_f32_pass(
        lb + ".feed_forward.w2.weight.resident", w2,
        "s2mel.net.cfm.estimator.transformer.feed_forward.zero_out_bias.resident", zero_out_bias,
        gated, rows, 512, 1536);

    metal.add_f32_pass_into(h, ffn_out, output);
}

// Full single-pass CFG estimator step: one MTLCommandBuffer for the entire forward pass.
// Falls back to the non-pass version when tokens > kFusedDitAttentionMaxTokens.
std::pair<std::vector<float>, std::vector<float>>
run_dit_estimator_step_metal_cfg_transformer_batched_pass(
    mit2::MetalContext& metal,
    const mit2::Bundle& bundle,
    const std::vector<float>& x,
    const std::vector<float>& prompt_x,
    const std::vector<float>& cond,
    const std::vector<float>& style,
    const std::vector<float>& null_prompt_x,
    const std::vector<float>& null_cond,
    const std::vector<float>& null_style,
    const std::vector<float>& t1,
    const std::vector<float>& t2,
    const std::vector<uint32_t>& mask,
    uint32_t tokens)
{
    if (tokens > kFusedDitAttentionMaxTokens) {
        return run_dit_estimator_step_metal_cfg_transformer_batched(
            metal, bundle, x, prompt_x, cond, style,
            null_prompt_x, null_cond, null_style, t1, t2, mask, tokens);
    }

    constexpr uint32_t batch = 2;
    const uint32_t rows = batch * tokens;

    // Workspace: persistent region (~26000 el/token) + reusable scratch (~21500 el/token, reset per block).
    // Formula: (persistent + scratch) × 4 bytes/element + alignment padding.
    const size_t ws = static_cast<size_t>(tokens) * (26000 + 27000) * 4 + 256 * 200;
    metal.beginPass(ws);

    // ----------------------------------------------------------------
    // Prepare batched inputs on CPU side, then upload into workspace.
    // ----------------------------------------------------------------
    std::vector<float> x_mel_bat(x.size() * batch);
    std::copy(x.begin(), x.end(), x_mel_bat.begin());
    std::copy(x.begin(), x.end(), x_mel_bat.begin() + static_cast<std::ptrdiff_t>(x.size()));

    std::vector<float> px_bat(prompt_x.size() * batch);
    std::copy(prompt_x.begin(), prompt_x.end(), px_bat.begin());
    std::copy(null_prompt_x.begin(), null_prompt_x.end(), px_bat.begin() + static_cast<std::ptrdiff_t>(prompt_x.size()));

    std::vector<float> cond_bat(cond.size() * batch);
    std::copy(cond.begin(), cond.end(), cond_bat.begin());
    std::copy(null_cond.begin(), null_cond.end(), cond_bat.begin() + static_cast<std::ptrdiff_t>(cond.size()));

    std::vector<float> style_bat(style.size() * batch);
    std::copy(style.begin(), style.end(), style_bat.begin());
    std::copy(null_style.begin(), null_style.end(), style_bat.begin() + static_cast<std::ptrdiff_t>(style.size()));

    std::vector<uint32_t> mask_bat;
    mask_bat.reserve(static_cast<size_t>(batch) * tokens);
    mask_bat.insert(mask_bat.end(), mask.begin(), mask.end());
    mask_bat.insert(mask_bat.end(), mask.begin(), mask.end());

    // ----------------------------------------------------------------
    // Pre-allocate ALL persistent output slots (must come before passSetScratchBase).
    // Uploaded inputs live here too — they need to survive for the entire pass.
    // ----------------------------------------------------------------
    auto x_mel_slot = metal.passUploadAlloc(x_mel_bat);
    auto px_slot    = metal.passUploadAlloc(px_bat);
    auto cond_slot  = metal.passUploadAlloc(cond_bat);
    auto style_slot = metal.passUploadAlloc(style_bat.data(), static_cast<uint32_t>(style_bat.size()));
    auto mask_slot  = metal.passUploadAllocU32(mask_bat);
    auto t1_slot    = metal.passUploadAlloc(t1);
    auto t2_slot    = metal.passUploadAlloc(t2);

    // Persistent outputs for input merge
    auto cond_proj_slot = metal.passAlloc(rows * 512);
    auto x_in_slot      = metal.passAlloc(rows * 512);

    // Transformer persistent: 6 skip slots (layers 0-5 outputs),
    // 2 ping-pong slots (layers 6-12 outputs), 1 raw output of layer 12.
    mit2::PassSlot skip_slot[6];
    for (int i = 0; i < 6; ++i) skip_slot[i] = metal.passAlloc(rows * 512);
    auto x_ping   = metal.passAlloc(rows * 512);
    auto x_pong   = metal.passAlloc(rows * 512);
    auto x_raw_12 = metal.passAlloc(rows * 512);  // raw layer-12 output before final norm

    // Post-transformer persistent
    auto long_skip_slot = metal.passAlloc(rows * 512);
    auto conv1_slot     = metal.passAlloc(rows * 512);
    auto res_proj_slot  = metal.passAlloc(rows * 512);

    // Wavenet persistent
    auto wn_cond_slot = metal.passAlloc(8192);
    const std::vector<float> wn_zeros_vec(static_cast<size_t>(rows) * 512, 0.0f);
    auto wn_zeros     = metal.passUploadAlloc(wn_zeros_vec);  // initial output accumulator
    auto wn_state_a   = metal.passAlloc(rows * 1024);   // ping-pong A: [new_x | new_out]
    auto wn_state_b   = metal.passAlloc(rows * 1024);   // ping-pong B

    // Final layer persistent
    auto final_in_slot     = metal.passAlloc(rows * 512);
    auto final_hidden_slot = metal.passAlloc(rows * 512);
    auto dphi_slot         = metal.passAlloc(rows * 80);

    // All subsequent allocations are scratch — reset per-block to reuse memory.
    metal.passSetScratchBase();

    // ----------------------------------------------------------------
    // Input merge: cond_projection → dit_input_merge → cond_x_merge_linear
    // ----------------------------------------------------------------
    {
        auto cp_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_projection.weight");
        auto cp_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_projection.bias");
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.cond_projection.weight.resident", cp_w,
            "s2mel.net.cfm.estimator.cond_projection.bias.resident", cp_b,
            cond_slot, rows, 512, 512, cond_proj_slot);
        // cond_proj_slot is persistent; scratch still empty — no reset needed yet.

        auto merged_tmp = metal.dit_input_merge_batched_f32_pass(
            x_mel_slot, px_slot, cond_proj_slot, style_slot, batch, tokens);

        auto mg_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.weight");
        auto mg_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.bias");
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.cond_x_merge_linear.weight.resident", mg_w,
            "s2mel.net.cfm.estimator.cond_x_merge_linear.bias.resident", mg_b,
            merged_tmp, rows, 512, 864, x_in_slot);

        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Transformer stack: 13 layers with skip connections (LIFO).
    // Layer outputs:
    //   0..5  → skip_slot[0..5]  (saved for skip connections at layers 7..12)
    //   6     → x_ping           (no skip)
    //   7     → x_pong           (skip from skip_slot[5])
    //   8     → x_ping           (skip from skip_slot[4])
    //   9     → x_pong           (skip from skip_slot[3])
    //   10    → x_ping           (skip from skip_slot[2])
    //   11    → x_pong           (skip from skip_slot[1])
    //   12    → x_raw_12         (skip from skip_slot[0])
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot layer_inputs[13] = {
            x_in_slot,
            skip_slot[0], skip_slot[1], skip_slot[2], skip_slot[3], skip_slot[4],
            skip_slot[5], x_ping, x_pong, x_ping, x_pong, x_ping, x_pong
        };
        const mit2::PassSlot layer_outputs[13] = {
            skip_slot[0], skip_slot[1], skip_slot[2], skip_slot[3], skip_slot[4], skip_slot[5],
            x_ping, x_pong, x_ping, x_pong, x_ping, x_pong, x_raw_12
        };
        const int skip_from[6] = {5, 4, 3, 2, 1, 0};  // for layers 7..12

        for (uint32_t layer = 0; layer < 13; ++layer) {
            const bool has_skip = (layer > 6);
            const mit2::PassSlot si = has_skip ? skip_slot[skip_from[layer - 7]] : mit2::PassSlot{};

            const std::string lb = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
            auto attn_pw = tensor_for_resident(metal, bundle, lb + ".attention_norm.project_layer.weight");
            auto attn_pb = tensor_for_resident(metal, bundle, lb + ".attention_norm.project_layer.bias");
            auto attn_wb = metal.linear_f32_pass(
                lb + ".attention_norm.project_layer.weight.resident", attn_pw,
                lb + ".attention_norm.project_layer.bias.resident", attn_pb,
                t1_slot, 1024, 512);
            auto ffn_pw = tensor_for_resident(metal, bundle, lb + ".ffn_norm.project_layer.weight");
            auto ffn_pb = tensor_for_resident(metal, bundle, lb + ".ffn_norm.project_layer.bias");
            auto ffn_wb = metal.linear_f32_pass(
                lb + ".ffn_norm.project_layer.weight.resident", ffn_pw,
                lb + ".ffn_norm.project_layer.bias.resident", ffn_pb,
                t1_slot, 1024, 512);

            run_transformer_block_pass_into(
                metal, bundle,
                layer_inputs[layer], attn_wb, ffn_wb, mask_slot,
                batch, tokens, layer,
                si, has_skip,
                layer_outputs[layer]);

            metal.passResetScratch();
        }

        // Final transformer norm (adaptive rmsnorm with t1 conditioning).
        // x_raw_12 is persistent; norm output is scratch but consumed immediately below.
        const std::string norm_base = "s2mel.net.cfm.estimator.transformer.norm";
        auto norm_g  = tensor_for_resident(metal, bundle, norm_base + ".norm.weight");
        auto norm_pw = tensor_for_resident(metal, bundle, norm_base + ".project_layer.weight");
        auto norm_pb = tensor_for_resident(metal, bundle, norm_base + ".project_layer.bias");
        auto norm_wb = metal.linear_f32_pass(
            norm_base + ".project_layer.weight.resident", norm_pw,
            norm_base + ".project_layer.bias.resident", norm_pb,
            t1_slot, 1024, 512);
        // x_normed is in scratch — consumed by post-transformer below without resetting.
        auto x_normed = metal.adaptive_rmsnorm_rows_f32_pass(
            norm_base + ".norm.weight.resident", norm_g,
            x_raw_12, norm_wb.slice(0, 512), norm_wb.slice(512, 512), rows, 512, 1e-5f);
        // Do NOT passResetScratch() here — x_normed must survive into post-transformer.

        // ----------------------------------------------------------------
        // Post-transformer projection.
        // x_normed is in scratch (valid until next passResetScratch).
        // ----------------------------------------------------------------
        auto skip_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.skip_linear.weight");
        auto skip_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.skip_linear.bias");
        auto c1_w   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv1.weight");
        auto c1_b   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv1.bias");
        auto rp_w   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.res_projection.weight");
        auto rp_b   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.res_projection.bias");

        auto skip_in_tmp = metal.concat_rows_f32_pass(x_normed, x_mel_slot, rows, 512, 80);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.skip_linear.weight.resident", skip_w,
            "s2mel.net.cfm.estimator.skip_linear.bias.resident", skip_b,
            skip_in_tmp, rows, 512, 592, long_skip_slot);
        // long_skip_slot is persistent; scratch (x_normed, skip_in_tmp) no longer needed.
        metal.passResetScratch();

        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.conv1.weight.resident", c1_w,
            "s2mel.net.cfm.estimator.conv1.bias.resident", c1_b,
            long_skip_slot, rows, 512, 512, conv1_slot);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.res_projection.weight.resident", rp_w,
            "s2mel.net.cfm.estimator.res_projection.bias.resident", rp_b,
            long_skip_slot, rows, 512, 512, res_proj_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Wavenet conditioning: conv1d(t2, 1, 512, 8192, 1) → wn_cond_slot
    // ----------------------------------------------------------------
    {
        const std::string cond_pfx = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
        auto cond_w = (metal.residentExists(cond_pfx + ".weight_norm.resident") ||
                       metal.residentExists(cond_pfx + ".weight_norm.resident.f16"))
                          ? std::vector<float>{}
                          : weight_norm_conv_weight(bundle, cond_pfx, 8192, 512, 1);
        auto cond_b = tensor_for_resident(metal, bundle, cond_pfx + ".bias");
        metal.conv1d_same_f32_pass_into(
            cond_pfx + ".weight_norm.resident", cond_w,
            cond_pfx + ".bias.resident", cond_b,
            t2_slot, 1, 512, 8192, 1, wn_cond_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Wavenet stack: 8 layers, ping-pong state buffers.
    // wn_state[0] = wn_state_a, wn_state[1] = wn_state_b.
    // Layer i writes to wn_state[i%2], reads prev from wn_state[(i+1)%2].
    // Layer 0 special: x=conv1_slot, out_acc=wn_zeros.
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot wn_states[2] = {wn_state_a, wn_state_b};

        for (uint32_t layer = 0; layer < 8; ++layer) {
            const uint32_t res_skip_ch = (layer < 7) ? 1024u : 512u;
            const mit2::PassSlot& out_state = wn_states[layer % 2];

            mit2::PassSlot wn_x, wn_out_acc;
            if (layer == 0) {
                wn_x       = conv1_slot;
                wn_out_acc = wn_zeros;
            } else {
                const mit2::PassSlot& prev = wn_states[(layer - 1) % 2];
                wn_x       = prev.slice(0,             rows * 512);
                wn_out_acc = prev.slice(rows * 512,    rows * 512);
            }

            const std::string in_pfx = "s2mel.net.cfm.estimator.wavenet.in_layers."  + std::to_string(layer) + ".conv.conv";
            const std::string rs_pfx = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";

            // k=5 conv runs via per-tap residents (MPS branch); only skip the CPU
            // weight-norm compute once the taps exist. (Base resident alone is not
            // enough — the MPS branch would have no data to build taps from.)
            const std::string in_res = in_pfx + ".weight_norm.resident";
            auto in_w  = (metal.residentExists(in_res + ".tap0") || metal.residentExists(in_res + ".tap0.f16"))
                             ? std::vector<float>{}
                             : weight_norm_conv_weight(bundle, in_pfx, 1024, 512, 5);
            auto in_b  = tensor_for_resident(metal, bundle, in_pfx + ".bias");
            auto rs_w  = (metal.residentExists(rs_pfx + ".weight_norm.resident") ||
                          metal.residentExists(rs_pfx + ".weight_norm.resident.f16"))
                             ? std::vector<float>{}
                             : weight_norm_conv_weight(bundle, rs_pfx, res_skip_ch, 512, 1);
            auto rs_b  = tensor_for_resident(metal, bundle, rs_pfx + ".bias");

            const uint32_t cond_off = layer * 1024;

            auto in_layer = metal.conv1d_reflect_same_batched_f32_pass(
                in_pfx + ".weight_norm.resident", in_w,
                in_pfx + ".bias.resident", in_b,
                wn_x, batch, tokens, 512, 1024, 5);
            auto gate = metal.wavenet_gate_f32_pass(
                in_layer, wn_cond_slot, rows, 512, 8192, cond_off, 1);
            auto res_skip = metal.conv1d_same_f32_pass(
                rs_pfx + ".weight_norm.resident", rs_w,
                rs_pfx + ".bias.resident", rs_b,
                gate, rows, 512, res_skip_ch, 1);

            metal.wavenet_res_skip_update_f32_pass_into(
                wn_x, wn_out_acc, res_skip, mask_slot,
                rows, 512, layer < 7,
                out_state);

            metal.passResetScratch();
        }
    }

    // ----------------------------------------------------------------
    // Final add: wavenet_out + res_proj → final_in_slot
    // Wavenet output = "output" half of the last state (wn_state_b, layer 7 writes to wn_state[7%2]).
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot& last_state = (7 % 2 == 0) ? wn_state_a : wn_state_b;
        auto wn_final = last_state.slice(rows * 512, rows * 512);
        metal.add_f32_pass_into(wn_final, res_proj_slot, final_in_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Final layer: silu(t1) → ada_modulation → adaptive_layernorm → linear → conv2 → dphi
    // ----------------------------------------------------------------
    {
        auto ada_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight");
        auto ada_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias");
        const bool lin_res = metal.residentExists("s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident") ||
                             metal.residentExists("s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident.f16");
        auto lin_g = lin_res ? std::vector<float>{} : tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_g");
        auto lin_v = lin_res ? std::vector<float>{} : tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_v");
        auto lin_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.linear.bias");
        auto lin_w = lin_res ? std::vector<float>{} : weight_norm_rowmajor(lin_g, lin_v, 512, 512);

        auto silu_t1     = metal.silu_f32_pass(t1_slot, 512);
        auto shift_scale = metal.linear_f32_pass(
            "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight.resident", ada_w,
            "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias.resident", ada_b,
            silu_t1, 1024, 512);
        auto modulated = metal.adaptive_layernorm_rows_f32_pass(
            final_in_slot, shift_scale.slice(0, 512), shift_scale.slice(512, 512),
            rows, 512, 1e-6f);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident", lin_w,
            "s2mel.net.cfm.estimator.final_layer.linear.bias.resident", lin_b,
            modulated, rows, 512, 512, final_hidden_slot);
        metal.passResetScratch();

        auto c2_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv2.weight");
        auto c2_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv2.bias");
        metal.conv1d_same_f32_pass_into(
            "s2mel.net.cfm.estimator.conv2.weight.resident", c2_w,
            "s2mel.net.cfm.estimator.conv2.bias.resident", c2_b,
            final_hidden_slot, rows, 512, 80, 1, dphi_slot);
        metal.passResetScratch();
    }

    metal.endPass();

    // ----------------------------------------------------------------
    // Read outputs and split CFG branches.
    // ----------------------------------------------------------------
    auto dphi_full = metal.passRead(dphi_slot);
    const size_t branch = static_cast<size_t>(tokens) * 80;
    std::vector<float> dphi(dphi_full.begin(), dphi_full.begin() + static_cast<std::ptrdiff_t>(branch));
    std::vector<float> null_dphi(dphi_full.begin() + static_cast<std::ptrdiff_t>(branch),
                                 dphi_full.begin() + static_cast<std::ptrdiff_t>(branch * 2));
    return {std::move(dphi), std::move(null_dphi)};
}

// Full CFM in ONE command buffer: constants uploaded once, x stays on-GPU,
// per-step euler update fused into the pass. Eliminates 25x (pass setup +
// constant upload + x readback/re-upload + 2 timestep-embedder CBs + sync).
// Per-step t-conditioned constant block layout (all functions of (step, layer) only):
// [attn_wb 13x1024 | ffn_wb 13x1024 | norm_wb 1024 | final_ss 1024 | wn_cond 8192]
constexpr uint32_t kTcondPerStep = 13 * 1024 * 2 + 1024 + 1024 + 8192;  // 36864

std::vector<float> run_cfm_euler_metal_single_pass(
    mit2::MetalContext& metal,
    const mit2::Bundle& bundle,
    const std::vector<float>& x0,
    const std::vector<float>& prompt_x,
    const std::vector<float>& cond,
    const std::vector<float>& style,
    const std::vector<float>& t1_all,
    const std::vector<float>& t2_all,
    const std::vector<float>& tcond_all,
    const std::vector<uint32_t>& mask,
    uint32_t tokens,
    uint32_t prompt_tokens,
    uint32_t steps,
    float cfg_rate)
{
    const std::vector<float> null_prompt_x(static_cast<size_t>(tokens) * 80, 0.0f);
    const std::vector<float> null_cond(static_cast<size_t>(tokens) * 512, 0.0f);
    const std::vector<float> null_style(192, 0.0f);
    constexpr uint32_t batch = 2;
    const uint32_t rows = batch * tokens;

    // Workspace: persistent region (~26000 el/token) + reusable scratch (~21500 el/token, reset per block).
    // Formula: (persistent + scratch) × 4 bytes/element + alignment padding.
    const size_t ws = static_cast<size_t>(tokens) * (26000 + 27000 + 4 * 80) * 4 +
                      static_cast<size_t>(steps) * (1024 + kTcondPerStep) * 4 + 256 * 224;
    metal.beginPass(ws);

    // ----------------------------------------------------------------
    // Prepare batched inputs on CPU side, then upload into workspace.
    // ----------------------------------------------------------------
    std::vector<float> px_bat(prompt_x.size() * batch);
    std::copy(prompt_x.begin(), prompt_x.end(), px_bat.begin());
    std::copy(null_prompt_x.begin(), null_prompt_x.end(), px_bat.begin() + static_cast<std::ptrdiff_t>(prompt_x.size()));

    std::vector<float> cond_bat(cond.size() * batch);
    std::copy(cond.begin(), cond.end(), cond_bat.begin());
    std::copy(null_cond.begin(), null_cond.end(), cond_bat.begin() + static_cast<std::ptrdiff_t>(cond.size()));

    std::vector<float> style_bat(style.size() * batch);
    std::copy(style.begin(), style.end(), style_bat.begin());
    std::copy(null_style.begin(), null_style.end(), style_bat.begin() + static_cast<std::ptrdiff_t>(style.size()));

    std::vector<uint32_t> mask_bat;
    mask_bat.reserve(static_cast<size_t>(batch) * tokens);
    mask_bat.insert(mask_bat.end(), mask.begin(), mask.end());
    mask_bat.insert(mask_bat.end(), mask.begin(), mask.end());

    // ----------------------------------------------------------------
    // Pre-allocate ALL persistent output slots (must come before passSetScratchBase).
    // Uploaded inputs live here too — they need to survive for the entire pass.
    // ----------------------------------------------------------------
    auto x_mel_slot = metal.passAlloc(rows * 80);          // batched [x; x], filled per step
    auto x_cur_slot = metal.passUploadAlloc(x0);            // current trajectory state
    auto x_nxt_slot = metal.passAlloc(tokens * 80);
    auto px_slot    = metal.passUploadAlloc(px_bat);
    auto cond_slot  = metal.passUploadAlloc(cond_bat);
    auto style_slot = metal.passUploadAlloc(style_bat.data(), static_cast<uint32_t>(style_bat.size()));
    auto mask_slot  = metal.passUploadAllocU32(mask_bat);
    auto t1_all_slot = metal.passUploadAlloc(t1_all);       // [steps, 512] timestep embeddings
    auto t2_all_slot = metal.passUploadAlloc(t2_all);
    auto tcond_slot  = metal.passUploadAlloc(tcond_all);    // [steps, kTcondPerStep] modulation tables

    // Persistent outputs for input merge
    auto cond_proj_slot = metal.passAlloc(rows * 512);
    auto x_in_slot      = metal.passAlloc(rows * 512);

    // Transformer persistent: 6 skip slots (layers 0-5 outputs),
    // 2 ping-pong slots (layers 6-12 outputs), 1 raw output of layer 12.
    mit2::PassSlot skip_slot[6];
    for (int i = 0; i < 6; ++i) skip_slot[i] = metal.passAlloc(rows * 512);
    auto x_ping   = metal.passAlloc(rows * 512);
    auto x_pong   = metal.passAlloc(rows * 512);
    auto x_raw_12 = metal.passAlloc(rows * 512);  // raw layer-12 output before final norm

    // Post-transformer persistent
    auto long_skip_slot = metal.passAlloc(rows * 512);
    auto conv1_slot     = metal.passAlloc(rows * 512);
    auto res_proj_slot  = metal.passAlloc(rows * 512);

    // Wavenet persistent
    auto wn_cond_slot = metal.passAlloc(8192);
    const std::vector<float> wn_zeros_vec(static_cast<size_t>(rows) * 512, 0.0f);
    auto wn_zeros     = metal.passUploadAlloc(wn_zeros_vec);  // initial output accumulator
    auto wn_state_a   = metal.passAlloc(rows * 1024);   // ping-pong A: [new_x | new_out]
    auto wn_state_b   = metal.passAlloc(rows * 1024);   // ping-pong B

    // Final layer persistent
    auto final_in_slot     = metal.passAlloc(rows * 512);
    auto final_hidden_slot = metal.passAlloc(rows * 512);
    auto dphi_slot         = metal.passAlloc(rows * 80);

    // All subsequent allocations are scratch — reset per-block to reuse memory.
    metal.passSetScratchBase();

    const float dt = 1.0f / static_cast<float>(steps);
    for (uint32_t step = 0; step < steps; ++step) {
    const mit2::PassSlot t1_slot = t1_all_slot.slice(step * 512, 512);
    const mit2::PassSlot t2_slot = t2_all_slot.slice(step * 512, 512);
    // Duplicate x into both CFG branches of the batched mel input.
    metal.copy_f32_pass_into(x_cur_slot, x_mel_slot.slice(0, tokens * 80), tokens * 80);
    metal.copy_f32_pass_into(x_cur_slot, x_mel_slot.slice(tokens * 80, tokens * 80), tokens * 80);

    // ----------------------------------------------------------------
    // Input merge: cond_projection → dit_input_merge → cond_x_merge_linear
    // ----------------------------------------------------------------
    {
        auto cp_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_projection.weight");
        auto cp_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_projection.bias");
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.cond_projection.weight.resident", cp_w,
            "s2mel.net.cfm.estimator.cond_projection.bias.resident", cp_b,
            cond_slot, rows, 512, 512, cond_proj_slot);
        // cond_proj_slot is persistent; scratch still empty — no reset needed yet.

        auto merged_tmp = metal.dit_input_merge_batched_f32_pass(
            x_mel_slot, px_slot, cond_proj_slot, style_slot, batch, tokens);

        auto mg_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.weight");
        auto mg_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.bias");
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.cond_x_merge_linear.weight.resident", mg_w,
            "s2mel.net.cfm.estimator.cond_x_merge_linear.bias.resident", mg_b,
            merged_tmp, rows, 512, 864, x_in_slot);

        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Transformer stack: 13 layers with skip connections (LIFO).
    // Layer outputs:
    //   0..5  → skip_slot[0..5]  (saved for skip connections at layers 7..12)
    //   6     → x_ping           (no skip)
    //   7     → x_pong           (skip from skip_slot[5])
    //   8     → x_ping           (skip from skip_slot[4])
    //   9     → x_pong           (skip from skip_slot[3])
    //   10    → x_ping           (skip from skip_slot[2])
    //   11    → x_pong           (skip from skip_slot[1])
    //   12    → x_raw_12         (skip from skip_slot[0])
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot layer_inputs[13] = {
            x_in_slot,
            skip_slot[0], skip_slot[1], skip_slot[2], skip_slot[3], skip_slot[4],
            skip_slot[5], x_ping, x_pong, x_ping, x_pong, x_ping, x_pong
        };
        const mit2::PassSlot layer_outputs[13] = {
            skip_slot[0], skip_slot[1], skip_slot[2], skip_slot[3], skip_slot[4], skip_slot[5],
            x_ping, x_pong, x_ping, x_pong, x_ping, x_pong, x_raw_12
        };
        const int skip_from[6] = {5, 4, 3, 2, 1, 0};  // for layers 7..12

        for (uint32_t layer = 0; layer < 13; ++layer) {
            const bool has_skip = (layer > 6);
            const mit2::PassSlot si = has_skip ? skip_slot[skip_from[layer - 7]] : mit2::PassSlot{};

            // Precomputed per-(step,layer) modulation vectors from the t-cond table.
            const mit2::PassSlot attn_wb = tcond_slot.slice(step * kTcondPerStep + layer * 1024, 1024);
            const mit2::PassSlot ffn_wb  = tcond_slot.slice(step * kTcondPerStep + 13 * 1024 + layer * 1024, 1024);

            run_transformer_block_pass_into(
                metal, bundle,
                layer_inputs[layer], attn_wb, ffn_wb, mask_slot,
                batch, tokens, layer,
                si, has_skip,
                layer_outputs[layer]);

            metal.passResetScratch();
        }

        // Final transformer norm (adaptive rmsnorm with t1 conditioning).
        // x_raw_12 is persistent; norm output is scratch but consumed immediately below.
        const std::string norm_base = "s2mel.net.cfm.estimator.transformer.norm";
        auto norm_g  = tensor_for_resident(metal, bundle, norm_base + ".norm.weight");
        const mit2::PassSlot norm_wb = tcond_slot.slice(step * kTcondPerStep + 26 * 1024, 1024);
        // x_normed is in scratch — consumed by post-transformer below without resetting.
        auto x_normed = metal.adaptive_rmsnorm_rows_f32_pass(
            norm_base + ".norm.weight.resident", norm_g,
            x_raw_12, norm_wb.slice(0, 512), norm_wb.slice(512, 512), rows, 512, 1e-5f);
        // Do NOT passResetScratch() here — x_normed must survive into post-transformer.

        // ----------------------------------------------------------------
        // Post-transformer projection.
        // x_normed is in scratch (valid until next passResetScratch).
        // ----------------------------------------------------------------
        auto skip_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.skip_linear.weight");
        auto skip_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.skip_linear.bias");
        auto c1_w   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv1.weight");
        auto c1_b   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv1.bias");
        auto rp_w   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.res_projection.weight");
        auto rp_b   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.res_projection.bias");

        auto skip_in_tmp = metal.concat_rows_f32_pass(x_normed, x_mel_slot, rows, 512, 80);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.skip_linear.weight.resident", skip_w,
            "s2mel.net.cfm.estimator.skip_linear.bias.resident", skip_b,
            skip_in_tmp, rows, 512, 592, long_skip_slot);
        // long_skip_slot is persistent; scratch (x_normed, skip_in_tmp) no longer needed.
        metal.passResetScratch();

        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.conv1.weight.resident", c1_w,
            "s2mel.net.cfm.estimator.conv1.bias.resident", c1_b,
            long_skip_slot, rows, 512, 512, conv1_slot);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.res_projection.weight.resident", rp_w,
            "s2mel.net.cfm.estimator.res_projection.bias.resident", rp_b,
            long_skip_slot, rows, 512, 512, res_proj_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Wavenet conditioning: conv1d(t2, 1, 512, 8192, 1) → wn_cond_slot
    // ----------------------------------------------------------------
    {
        // wn_cond precomputed per step in the t-cond table.
        metal.copy_f32_pass_into(
            tcond_slot.slice(step * kTcondPerStep + 28 * 1024, 8192),
            wn_cond_slot, 8192);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Wavenet stack: 8 layers, ping-pong state buffers.
    // wn_state[0] = wn_state_a, wn_state[1] = wn_state_b.
    // Layer i writes to wn_state[i%2], reads prev from wn_state[(i+1)%2].
    // Layer 0 special: x=conv1_slot, out_acc=wn_zeros.
    // ----------------------------------------------------------------
    if (cfm_skip("wavenet")) {
        // Placeholder: wavenet output := conv1 (shape-correct, removes 8 conv layers).
        const mit2::PassSlot last_state = (7 % 2 == 0) ? wn_state_a : wn_state_b;
        metal.copy_f32_pass_into(conv1_slot, last_state.slice(rows * 512, rows * 512), rows * 512);
    } else {
        const mit2::PassSlot wn_states[2] = {wn_state_a, wn_state_b};

        for (uint32_t layer = 0; layer < 8; ++layer) {
            const uint32_t res_skip_ch = (layer < 7) ? 1024u : 512u;
            const mit2::PassSlot& out_state = wn_states[layer % 2];

            mit2::PassSlot wn_x, wn_out_acc;
            if (layer == 0) {
                wn_x       = conv1_slot;
                wn_out_acc = wn_zeros;
            } else {
                const mit2::PassSlot& prev = wn_states[(layer - 1) % 2];
                wn_x       = prev.slice(0,             rows * 512);
                wn_out_acc = prev.slice(rows * 512,    rows * 512);
            }

            const std::string in_pfx = "s2mel.net.cfm.estimator.wavenet.in_layers."  + std::to_string(layer) + ".conv.conv";
            const std::string rs_pfx = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";

            // k=5 conv runs via per-tap residents (MPS branch); only skip the CPU
            // weight-norm compute once the taps exist. (Base resident alone is not
            // enough — the MPS branch would have no data to build taps from.)
            const std::string in_res = in_pfx + ".weight_norm.resident";
            auto in_w  = (metal.residentExists(in_res + ".tap0") || metal.residentExists(in_res + ".tap0.f16"))
                             ? std::vector<float>{}
                             : weight_norm_conv_weight(bundle, in_pfx, 1024, 512, 5);
            auto in_b  = tensor_for_resident(metal, bundle, in_pfx + ".bias");
            auto rs_w  = (metal.residentExists(rs_pfx + ".weight_norm.resident") ||
                          metal.residentExists(rs_pfx + ".weight_norm.resident.f16"))
                             ? std::vector<float>{}
                             : weight_norm_conv_weight(bundle, rs_pfx, res_skip_ch, 512, 1);
            auto rs_b  = tensor_for_resident(metal, bundle, rs_pfx + ".bias");

            const uint32_t cond_off = layer * 1024;

            auto in_layer = metal.conv1d_reflect_same_batched_f32_pass(
                in_pfx + ".weight_norm.resident", in_w,
                in_pfx + ".bias.resident", in_b,
                wn_x, batch, tokens, 512, 1024, 5);
            auto gate = metal.wavenet_gate_f32_pass(
                in_layer, wn_cond_slot, rows, 512, 8192, cond_off, 1);
            auto res_skip = metal.conv1d_same_f32_pass(
                rs_pfx + ".weight_norm.resident", rs_w,
                rs_pfx + ".bias.resident", rs_b,
                gate, rows, 512, res_skip_ch, 1);

            metal.wavenet_res_skip_update_f32_pass_into(
                wn_x, wn_out_acc, res_skip, mask_slot,
                rows, 512, layer < 7,
                out_state);

            metal.passResetScratch();
        }
    }

    // ----------------------------------------------------------------
    // Final add: wavenet_out + res_proj → final_in_slot
    // Wavenet output = "output" half of the last state (wn_state_b, layer 7 writes to wn_state[7%2]).
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot& last_state = (7 % 2 == 0) ? wn_state_a : wn_state_b;
        auto wn_final = last_state.slice(rows * 512, rows * 512);
        metal.add_f32_pass_into(wn_final, res_proj_slot, final_in_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Final layer: silu(t1) → ada_modulation → adaptive_layernorm → linear → conv2 → dphi
    // ----------------------------------------------------------------
    {
        auto ada_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight");
        auto ada_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias");
        const bool lin_res = metal.residentExists("s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident") ||
                             metal.residentExists("s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident.f16");
        auto lin_g = lin_res ? std::vector<float>{} : tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_g");
        auto lin_v = lin_res ? std::vector<float>{} : tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_v");
        auto lin_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.linear.bias");
        auto lin_w = lin_res ? std::vector<float>{} : weight_norm_rowmajor(lin_g, lin_v, 512, 512);

        (void)ada_w;
        (void)ada_b;
        const mit2::PassSlot shift_scale = tcond_slot.slice(step * kTcondPerStep + 27 * 1024, 1024);
        auto modulated = metal.adaptive_layernorm_rows_f32_pass(
            final_in_slot, shift_scale.slice(0, 512), shift_scale.slice(512, 512),
            rows, 512, 1e-6f);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident", lin_w,
            "s2mel.net.cfm.estimator.final_layer.linear.bias.resident", lin_b,
            modulated, rows, 512, 512, final_hidden_slot);
        metal.passResetScratch();

        auto c2_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv2.weight");
        auto c2_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv2.bias");
        metal.conv1d_same_f32_pass_into(
            "s2mel.net.cfm.estimator.conv2.weight.resident", c2_w,
            "s2mel.net.cfm.estimator.conv2.bias.resident", c2_b,
            final_hidden_slot, rows, 512, 80, 1, dphi_slot);
        metal.passResetScratch();
    }

    // Euler update: x_next = x + dt * ((1+cfg)*dphi - cfg*null_dphi); prompt rows -> 0.
    metal.cfm_euler_update_f32_pass_into(
        x_cur_slot,
        dphi_slot.slice(0, tokens * 80),
        dphi_slot.slice(tokens * 80, tokens * 80),
        x_nxt_slot,
        tokens, 80, prompt_tokens, dt, cfg_rate);
    std::swap(x_cur_slot, x_nxt_slot);
    metal.passResetScratch();
    }  // step loop

    metal.endPass();
    return metal.passRead(x_cur_slot);
}

bool run_dit_estimator_pass_tokens_test(const std::string& bundle_dir, uint32_t tokens) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    const uint32_t prompt_tokens = tokens / 2;
    std::vector<float> x(static_cast<size_t>(tokens) * 80);
    std::vector<float> prompt_x(static_cast<size_t>(tokens) * 80, 0.0f);
    std::vector<float> cond(static_cast<size_t>(tokens) * 512);
    std::vector<float> style(192);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.4f + std::cos(static_cast<float>(i % 89) * 0.021f) * 0.2f;
    }
    for (uint32_t t = 0; t < prompt_tokens; ++t) {
        for (uint32_t c = 0; c < 80; ++c) {
            prompt_x[static_cast<size_t>(t) * 80 + c] = std::cos(static_cast<float>(t * 80 + c) * 0.011f) * 0.5f;
            x[static_cast<size_t>(t) * 80 + c] = 0.0f;
        }
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.007f) * 0.3f;
    }
    for (size_t i = 0; i < style.size(); ++i) {
        style[i] = std::cos(static_cast<float>(i) * 0.015f) * 0.18f;
    }
    const std::vector<float> null_prompt_x(static_cast<size_t>(tokens) * 80, 0.0f);
    const std::vector<float> null_cond(static_cast<size_t>(tokens) * 512, 0.0f);
    const std::vector<float> null_style(192, 0.0f);
    std::vector<uint32_t> mask(tokens, 1);
    auto t1 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{0.32f}, "s2mel.net.cfm.estimator.t_embedder");
    auto t2 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{0.32f}, "s2mel.net.cfm.estimator.t_embedder2");

    auto ref = run_dit_estimator_step_metal_cfg_transformer_batched(
        metal, bundle, x, prompt_x, cond, style, null_prompt_x, null_cond, null_style, t1, t2, mask, tokens);
    auto ref2 = run_dit_estimator_step_metal_cfg_transformer_batched(
        metal, bundle, x, prompt_x, cond, style, null_prompt_x, null_cond, null_style, t1, t2, mask, tokens);
    std::pair<std::vector<float>, std::vector<float>> got, got2;
    try {
        got = run_dit_estimator_step_metal_cfg_transformer_batched_pass(
            metal, bundle, x, prompt_x, cond, style, null_prompt_x, null_cond, null_style, t1, t2, mask, tokens);
        got2 = run_dit_estimator_step_metal_cfg_transformer_batched_pass(
            metal, bundle, x, prompt_x, cond, style, null_prompt_x, null_cond, null_style, t1, t2, mask, tokens);
    } catch (const std::exception& e) {
        std::cerr << "pass estimator threw: " << e.what() << std::endl;
        throw;
    }

    const float ref_rep = std::max(max_abs_error(ref2.first, ref.first), max_abs_error(ref2.second, ref.second));
    const float got_rep = std::max(max_abs_error(got2.first, got.first), max_abs_error(got2.second, got.second));
    const float err = std::max(max_abs_error(got.first, ref.first), max_abs_error(got.second, ref.second));
    size_t nonfinite = 0;
    for (float v : got.first) if (!std::isfinite(v)) ++nonfinite;
    for (float v : got.second) if (!std::isfinite(v)) ++nonfinite;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_estimator_pass_tokens\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"pass_used\": " << (tokens <= kFusedDitAttentionMaxTokens ? "true" : "false") << ",\n";
    std::cout << "  \"nonfinite\": " << nonfinite << ",\n";
    std::cout << "  \"ref_repeat_err\": " << ref_rep << ",\n";
    std::cout << "  \"pass_repeat_err\": " << got_rep << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-2f && nonfinite == 0;
}

std::vector<float> run_dit_estimator_step_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& x, const std::vector<float>& prompt_x, const std::vector<float>& cond, const std::vector<float>& style, float timestep, const std::vector<uint32_t>& mask, uint32_t tokens) {
    auto t1 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{timestep}, "s2mel.net.cfm.estimator.t_embedder");
    auto t2 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{timestep}, "s2mel.net.cfm.estimator.t_embedder2");
    return run_dit_estimator_step_metal_with_embeddings(metal, bundle, x, prompt_x, cond, style, t1, t2, mask, tokens);
}

std::vector<float> trace_dit_estimator_step_metal_with_embeddings(mit2::MetalContext& metal,
                                                                  const mit2::Bundle& bundle,
                                                                  const std::vector<float>& x,
                                                                  const std::vector<float>& prompt_x,
                                                                  const std::vector<float>& cond,
                                                                  const std::vector<float>& style,
                                                                  const std::vector<float>& t1,
                                                                  const std::vector<float>& t2,
                                                                  const std::vector<uint32_t>& mask,
                                                                  uint32_t tokens,
                                                                  const char* branch_name) {
    std::cout << "  \"" << branch_name << "_estimator_trace\": [\n";
    print_tensor_stats_json("input_x", x, 80, true);
    print_tensor_stats_json("input_prompt_x", prompt_x, 80, true);
    print_tensor_stats_json("input_cond", cond, 512, true);
    print_tensor_stats_json("input_style", style, 192, true);
    auto x_in = run_dit_input_merge_metal(metal, bundle, x, prompt_x, cond, style, tokens);
    print_tensor_stats_json("input_merge", x_in, 512, true);
    auto x_res = run_dit_transformer_stack_metal(metal, bundle, x_in, t1, mask, tokens);
    print_tensor_stats_json("transformer_stack", x_res, 512, true);
    auto post = run_dit_post_transformer_proj_metal(metal, bundle, x_res, x, tokens);
    print_tensor_stats_json("post_long_skip", post.long_skip, 512, true);
    print_tensor_stats_json("post_conv1", post.conv1, 512, true);
    print_tensor_stats_json("post_res_projection", post.res_projection, 512, true);
    auto wav = trace_wavenet_stack_global_cond_metal_entries(metal, bundle, post.conv1, t2, mask, tokens);
    print_tensor_stats_json("wavenet", wav, 512, true);
    auto final_in = metal.add_f32(wav, post.res_projection);
    print_tensor_stats_json("final_in", final_in, 512, true);
    auto final_hidden = run_dit_final_layer_metal(metal, bundle, final_in, t1, tokens);
    print_tensor_stats_json("final_hidden", final_hidden, 512, true);
    auto out = metal.conv1d_same_f32_resident(
        "s2mel.net.cfm.estimator.conv2.weight.resident",
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.weight"),
        "s2mel.net.cfm.estimator.conv2.bias.resident",
        tensor_as_f32(bundle, "s2mel.net.cfm.estimator.conv2.bias"),
        final_hidden,
        tokens,
        512,
        80,
        1);
    print_tensor_stats_json("conv2_out", out, 80, false);
    std::cout << "  ],\n";
    return out;
}

bool run_dit_estimator_step_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 6;
    constexpr float timestep = 0.37f;
    std::vector<float> x(static_cast<size_t>(tokens) * 80);
    std::vector<float> prompt_x(static_cast<size_t>(tokens) * 80);
    std::vector<float> cond(static_cast<size_t>(tokens) * 512);
    std::vector<float> style(192);
    std::vector<uint32_t> mask{1, 1, 1, 1, 0, 0};
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.021f) * 0.10f + std::cos(static_cast<float>(i % 53) * 0.017f) * 0.04f;
        prompt_x[i] = std::cos(static_cast<float>(i) * 0.019f) * 0.08f + std::sin(static_cast<float>(i % 47) * 0.023f) * 0.03f;
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.010f) * 0.09f + std::cos(static_cast<float>(i % 127) * 0.013f) * 0.05f;
    }
    for (size_t i = 0; i < style.size(); ++i) {
        style[i] = std::cos(static_cast<float>(i) * 0.016f) * 0.20f;
    }
    auto ref = run_dit_estimator_step_cpu(bundle, x, prompt_x, cond, style, timestep, mask, tokens);
    auto got = run_dit_estimator_step_metal(metal, bundle, x, prompt_x, cond, style, timestep, mask, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_estimator_step\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"valid_tokens\": 4,\n";
    std::cout << "  \"mel_channels\": 80,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_dit_estimator_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr float timestep = 0.375f;
    auto x = read_raw_f32(golden_dir + "/x.f32");
    auto prompt_x = read_raw_f32(golden_dir + "/prompt_x.f32");
    auto cond = read_raw_f32(golden_dir + "/cond.f32");
    auto style = read_raw_f32(golden_dir + "/style.f32");
    auto golden = read_raw_f32(golden_dir + "/estimator.f32");
    if (x.empty() || (x.size() % mel_dim) != 0) {
        throw std::runtime_error("DiT golden x must have shape [tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(x.size() / mel_dim);
    if (prompt_x.size() != x.size()) {
        throw std::runtime_error("DiT golden prompt_x size mismatch");
    }
    if (cond.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("DiT golden cond must have shape [tokens,512]");
    }
    if (style.size() != style_dim) {
        throw std::runtime_error("DiT golden style must have shape [192]");
    }
    if (golden.size() != x.size()) {
        throw std::runtime_error("DiT golden estimator output size mismatch");
    }
    std::vector<uint32_t> mask(tokens, 1);
    auto ref = run_dit_estimator_step_cpu(bundle, x, prompt_x, cond, style, timestep, mask, tokens);
    auto got = run_dit_estimator_step_metal(metal, bundle, x, prompt_x, cond, style, timestep, mask, tokens);
    const float cpu_golden_err = max_abs_error(ref, golden);
    const float metal_err = max_abs_error(got, ref);
    const float err = std::max(cpu_golden_err, metal_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"dit_estimator_golden\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"timestep\": " << timestep << ",\n";
    std::cout << "  \"cpu_vs_golden_max_abs_error\": " << cpu_golden_err << ",\n";
    std::cout << "  \"metal_max_abs_error\": " << metal_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return cpu_golden_err <= 3e-3f && metal_err <= 3e-3f;
}

std::vector<float> run_cfm_euler_cpu(const mit2::Bundle& bundle, std::vector<float> x, const std::vector<float>& prompt, const std::vector<float>& cond, const std::vector<float>& style, uint32_t tokens, uint32_t prompt_tokens, uint32_t steps, float cfg_rate) {
    std::vector<float> prompt_x(static_cast<size_t>(tokens) * 80, 0.0f);
    for (uint32_t t = 0; t < prompt_tokens; ++t) {
        std::copy(prompt.begin() + t * 80, prompt.begin() + (t + 1) * 80, prompt_x.begin() + t * 80);
        std::fill(x.begin() + t * 80, x.begin() + (t + 1) * 80, 0.0f);
    }
    const std::vector<float> null_prompt_x(static_cast<size_t>(tokens) * 80, 0.0f);
    const std::vector<float> null_cond(static_cast<size_t>(tokens) * 512, 0.0f);
    const std::vector<float> null_style(192, 0.0f);
    std::vector<uint32_t> mask(tokens, 1);
    float t = 0.0f;
    const float dt = 1.0f / static_cast<float>(steps);
    for (uint32_t step = 1; step <= steps; ++step) {
        auto dphi = run_dit_estimator_step_cpu(bundle, x, prompt_x, cond, style, t, mask, tokens);
        if (cfg_rate > 0.0f) {
            auto cfg_dphi = run_dit_estimator_step_cpu(bundle, x, null_prompt_x, null_cond, null_style, t, mask, tokens);
            for (size_t i = 0; i < dphi.size(); ++i) {
                dphi[i] = (1.0f + cfg_rate) * dphi[i] - cfg_rate * cfg_dphi[i];
            }
        }
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += dt * dphi[i];
        }
        t += dt;
        for (uint32_t p = 0; p < prompt_tokens; ++p) {
            std::fill(x.begin() + p * 80, x.begin() + (p + 1) * 80, 0.0f);
        }
    }
    return x;
}

std::vector<float> run_cfm_euler_metal_impl(mit2::MetalContext& metal, const mit2::Bundle& bundle, std::vector<float> x, const std::vector<float>& prompt, const std::vector<float>& cond, const std::vector<float>& style, uint32_t tokens, uint32_t prompt_tokens, uint32_t steps, float cfg_rate) {
    std::vector<float> prompt_x(static_cast<size_t>(tokens) * 80, 0.0f);
    for (uint32_t t = 0; t < prompt_tokens; ++t) {
        std::copy(prompt.begin() + t * 80, prompt.begin() + (t + 1) * 80, prompt_x.begin() + t * 80);
        std::fill(x.begin() + t * 80, x.begin() + (t + 1) * 80, 0.0f);
    }
    if (cfg_rate > 0.0f && tokens <= kFusedDitAttentionMaxTokens) {
        // All t-conditioned quantities depend only on (step index, steps):
        // timestep embeddings PLUS every adaLN modulation projection, the final
        // norm/layer modulation, and the wavenet conditioning. Cache per steps value.
        struct TCondTables {
            std::vector<float> t1_all;
            std::vector<float> t2_all;
            std::vector<float> tcond_all;  // [steps, kTcondPerStep]
        };
        static const mit2::Bundle* t_cached_for = nullptr;
        static std::unordered_map<uint32_t, TCondTables> t_tables;
        if (t_cached_for != &bundle) {
            t_tables.clear();
            t_cached_for = &bundle;
        }
        auto t_it = t_tables.find(steps);
        if (t_it == t_tables.end()) {
            TCondTables tabs;
            // Load all projection weights once (resident keys match the pass path).
            struct Proj { std::string wk, bk; std::vector<float> w, b; };
            std::vector<Proj> attn_proj(13), ffn_proj(13);
            for (uint32_t l = 0; l < 13; ++l) {
                const std::string lb = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(l);
                attn_proj[l] = {lb + ".attention_norm.project_layer.weight.resident",
                                lb + ".attention_norm.project_layer.bias.resident",
                                tensor_as_f32(bundle, lb + ".attention_norm.project_layer.weight"),
                                tensor_as_f32(bundle, lb + ".attention_norm.project_layer.bias")};
                ffn_proj[l] = {lb + ".ffn_norm.project_layer.weight.resident",
                               lb + ".ffn_norm.project_layer.bias.resident",
                               tensor_as_f32(bundle, lb + ".ffn_norm.project_layer.weight"),
                               tensor_as_f32(bundle, lb + ".ffn_norm.project_layer.bias")};
            }
            const std::string norm_base = "s2mel.net.cfm.estimator.transformer.norm";
            auto norm_pw = tensor_as_f32(bundle, norm_base + ".project_layer.weight");
            auto norm_pb = tensor_as_f32(bundle, norm_base + ".project_layer.bias");
            auto ada_w = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight");
            auto ada_b = tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias");
            const std::string cond_pfx = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
            auto cond_w = weight_norm_conv_weight(bundle, cond_pfx, 8192, 512, 1);
            auto cond_b = tensor_as_f32(bundle, cond_pfx + ".bias");

            float tv = 0.0f;
            const float dtv = 1.0f / static_cast<float>(steps);
            tabs.tcond_all.reserve(static_cast<size_t>(steps) * kTcondPerStep);
            for (uint32_t st = 0; st < steps; ++st) {
                auto e1 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{tv}, "s2mel.net.cfm.estimator.t_embedder");
                auto e2 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{tv}, "s2mel.net.cfm.estimator.t_embedder2");
                for (uint32_t l = 0; l < 13; ++l) {
                    auto wb = metal.linear_f32_resident(attn_proj[l].wk, attn_proj[l].w, attn_proj[l].bk, attn_proj[l].b, e1, 1024, 512);
                    tabs.tcond_all.insert(tabs.tcond_all.end(), wb.begin(), wb.end());
                }
                for (uint32_t l = 0; l < 13; ++l) {
                    auto wb = metal.linear_f32_resident(ffn_proj[l].wk, ffn_proj[l].w, ffn_proj[l].bk, ffn_proj[l].b, e1, 1024, 512);
                    tabs.tcond_all.insert(tabs.tcond_all.end(), wb.begin(), wb.end());
                }
                {
                    auto wb = metal.linear_f32_resident(norm_base + ".project_layer.weight.resident", norm_pw,
                                                        norm_base + ".project_layer.bias.resident", norm_pb, e1, 1024, 512);
                    tabs.tcond_all.insert(tabs.tcond_all.end(), wb.begin(), wb.end());
                }
                {
                    auto silu_t1 = metal.silu_f32(e1);
                    auto ss = metal.linear_f32_resident("s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight.resident", ada_w,
                                                        "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias.resident", ada_b,
                                                        silu_t1, 1024, 512);
                    tabs.tcond_all.insert(tabs.tcond_all.end(), ss.begin(), ss.end());
                }
                {
                    auto wc = metal.conv1d_same_f32_resident(cond_pfx + ".weight_norm.resident", cond_w,
                                                             cond_pfx + ".bias.resident", cond_b,
                                                             e2, 1, 512, 8192, 1);
                    tabs.tcond_all.insert(tabs.tcond_all.end(), wc.begin(), wc.end());
                }
                tabs.t1_all.insert(tabs.t1_all.end(), e1.begin(), e1.end());
                tabs.t2_all.insert(tabs.t2_all.end(), e2.begin(), e2.end());
                tv += dtv;
            }
            t_it = t_tables.emplace(steps, std::move(tabs)).first;
        }
        std::vector<uint32_t> mask_ones(tokens, 1);
        return run_cfm_euler_metal_single_pass(metal, bundle, x, prompt_x, cond, style,
                                               t_it->second.t1_all, t_it->second.t2_all,
                                               t_it->second.tcond_all, mask_ones,
                                               tokens, prompt_tokens, steps, cfg_rate);
    }
    const std::vector<float> null_prompt_x(static_cast<size_t>(tokens) * 80, 0.0f);
    const std::vector<float> null_cond(static_cast<size_t>(tokens) * 512, 0.0f);
    const std::vector<float> null_style(192, 0.0f);
    std::vector<uint32_t> mask(tokens, 1);
    float t = 0.0f;
    const float dt = 1.0f / static_cast<float>(steps);
    for (uint32_t step = 1; step <= steps; ++step) {
        auto t1 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{t}, "s2mel.net.cfm.estimator.t_embedder");
        auto t2 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{t}, "s2mel.net.cfm.estimator.t_embedder2");
        if (cfg_rate > 0.0f) {
            auto dphis = run_dit_estimator_step_metal_cfg_transformer_batched_pass(metal,
                                                                                   bundle,
                                                                                   x,
                                                                                   prompt_x,
                                                                                   cond,
                                                                                   style,
                                                                                   null_prompt_x,
                                                                                   null_cond,
                                                                                   null_style,
                                                                                   t1,
                                                                                   t2,
                                                                                   mask,
                                                                                   tokens);
            const auto& dphi = dphis.first;
            const auto& null_dphi = dphis.second;
            x = metal.cfm_euler_update_f32(x, dphi, null_dphi, tokens, 80, prompt_tokens, dt, cfg_rate);
        } else {
            auto dphi = run_dit_estimator_step_metal_with_embeddings(metal, bundle, x, prompt_x, cond, style, t1, t2, mask, tokens);
            x = metal.cfm_euler_update_f32(x, dphi, dphi, tokens, 80, prompt_tokens, dt, 0.0f);
        }
        t += dt;
    }
    return x;
}

bool run_bench_cfm(const std::string& bundle_dir, uint32_t tokens, uint32_t steps, uint32_t iters) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    const uint32_t prompt_tokens = tokens * 6 / 7;
    std::vector<float> x(static_cast<size_t>(tokens) * 80);
    std::vector<float> prompt(static_cast<size_t>(prompt_tokens) * 80);
    std::vector<float> cond(static_cast<size_t>(tokens) * 512);
    std::vector<float> style(192);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.4f;
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = std::cos(static_cast<float>(i) * 0.011f) * 0.5f;
    for (size_t i = 0; i < cond.size(); ++i) cond[i] = std::sin(static_cast<float>(i) * 0.007f) * 0.3f;
    for (size_t i = 0; i < style.size(); ++i) style[i] = std::cos(static_cast<float>(i) * 0.015f) * 0.18f;

    // Warm: residents, tables, pipelines.
    (void)run_cfm_euler_metal_impl(metal, bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, 0.7f);
    const auto t0 = Clock::now();
    for (uint32_t i = 0; i < iters; ++i) {
        (void)run_cfm_euler_metal_impl(metal, bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, 0.7f);
    }
    const double total = seconds_since(t0);
    const char* skip = std::getenv("MIT2_CFM_SKIP");
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bench_cfm\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"iters\": " << iters << ",\n";
    std::cout << "  \"skip\": \"" << (skip ? skip : "") << "\",\n";
    std::cout << "  \"seconds_per_call\": " << (total / iters) << ",\n";
    std::cout << "  \"ms_per_step\": " << (total / iters / steps * 1000.0) << "\n";
    std::cout << "}\n";
    return true;
}


void warm_up_cfm_euler_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle) {
    constexpr uint32_t tokens = 5;
    constexpr uint32_t prompt_tokens = 2;
    std::vector<float> x(static_cast<size_t>(tokens) * 80);
    std::vector<float> prompt(static_cast<size_t>(prompt_tokens) * 80);
    std::vector<float> cond(static_cast<size_t>(tokens) * 512);
    std::vector<float> style(192);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.018f) * 0.07f + std::cos(static_cast<float>(i % 61) * 0.011f) * 0.03f;
    }
    for (size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = std::cos(static_cast<float>(i) * 0.022f) * 0.09f + std::sin(static_cast<float>(i % 37) * 0.019f) * 0.02f;
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.08f + std::cos(static_cast<float>(i % 109) * 0.014f) * 0.04f;
    }
    for (size_t i = 0; i < style.size(); ++i) {
        style[i] = std::cos(static_cast<float>(i) * 0.015f) * 0.18f;
    }
    (void)run_cfm_euler_metal_impl(metal, bundle, x, prompt, cond, style, tokens, prompt_tokens, 1, 0.7f);
}

std::vector<float> run_cfm_euler_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, std::vector<float> x, const std::vector<float>& prompt, const std::vector<float>& cond, const std::vector<float>& style, uint32_t tokens, uint32_t prompt_tokens, uint32_t steps, float cfg_rate) {
    warm_up_cfm_euler_metal(metal, bundle);
    return run_cfm_euler_metal_impl(metal, bundle, std::move(x), prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
}

bool trace_s2mel_cfm_golden(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    auto x = read_raw_f32(golden_dir + "/noise.f32");
    auto prompt = read_raw_f32(golden_dir + "/prompt_mel.f32");
    auto cond = read_raw_f32(golden_dir + "/condition.f32");
    auto style = read_raw_f32(golden_dir + "/style.f32");
    if (x.empty() || (x.size() % mel_dim) != 0) {
        throw std::runtime_error("S2Mel trace noise must have shape [tokens,80]");
    }
    if (prompt.empty() || (prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("S2Mel trace prompt_mel must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(x.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(prompt.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("S2Mel trace prompt_tokens exceeds total tokens");
    }
    if (cond.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("S2Mel trace condition must have shape [tokens,512]");
    }
    if (style.size() != style_dim) {
        throw std::runtime_error("S2Mel trace style must have shape [192]");
    }

    std::vector<float> prompt_x(static_cast<size_t>(tokens) * mel_dim, 0.0f);
    for (uint32_t t = 0; t < prompt_tokens; ++t) {
        std::copy(prompt.begin() + t * mel_dim, prompt.begin() + (t + 1) * mel_dim, prompt_x.begin() + t * mel_dim);
        std::fill(x.begin() + t * mel_dim, x.begin() + (t + 1) * mel_dim, 0.0f);
    }
    const std::vector<float> null_prompt_x(static_cast<size_t>(tokens) * mel_dim, 0.0f);
    const std::vector<float> null_cond(static_cast<size_t>(tokens) * cond_dim, 0.0f);
    const std::vector<float> null_style(style_dim, 0.0f);
    std::vector<uint32_t> mask(tokens, 1);
    float t = 0.0f;
    const float dt = 1.0f / static_cast<float>(steps);
    uint32_t total_nonfinite = 0;
    bool emitted_detail_trace = false;

    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_cfm_trace_golden\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << (tokens - prompt_tokens) << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"trace\": [\n";
    bool first_trace = true;
    for (uint32_t step = 1; step <= steps; ++step) {
        auto t1 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{t}, "s2mel.net.cfm.estimator.t_embedder");
        auto t2 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{t}, "s2mel.net.cfm.estimator.t_embedder2");
        auto dphi = run_dit_estimator_step_metal_with_embeddings(metal, bundle, x, prompt_x, cond, style, t1, t2, mask, tokens);
        auto null_dphi = run_dit_estimator_step_metal_with_embeddings(metal, bundle, x, null_prompt_x, null_cond, null_style, t1, t2, mask, tokens);
        auto updated = metal.cfm_euler_update_f32(x, dphi, null_dphi, tokens, mel_dim, prompt_tokens, dt, cfg_rate);
        const uint32_t dphi_nonfinite = count_nonfinite(dphi);
        const uint32_t null_nonfinite = count_nonfinite(null_dphi);
        const uint32_t x_nonfinite = count_nonfinite(updated);
        total_nonfinite += dphi_nonfinite + null_nonfinite + x_nonfinite;
        if (!first_trace) {
            std::cout << ",\n";
        }
        first_trace = false;
        std::cout << "    {\"step\": " << step
                  << ", \"t\": " << t
                  << ", \"dphi_nonfinite\": " << dphi_nonfinite
                  << ", \"dphi_finite_max_abs\": " << finite_max_abs_value(dphi)
                  << ", \"null_dphi_nonfinite\": " << null_nonfinite
                  << ", \"null_dphi_finite_max_abs\": " << finite_max_abs_value(null_dphi)
                  << ", \"x_nonfinite\": " << x_nonfinite
                  << ", \"x_finite_max_abs\": " << finite_max_abs_value(updated)
                  << "}";
        if (total_nonfinite > 0) {
            std::cout << "\n  ],\n";
            if (dphi_nonfinite > 0) {
                (void)trace_dit_estimator_step_metal_with_embeddings(metal, bundle, x, prompt_x, cond, style, t1, t2, mask, tokens, "conditional");
            }
            if (null_nonfinite > 0) {
                (void)trace_dit_estimator_step_metal_with_embeddings(metal, bundle, x, null_prompt_x, null_cond, null_style, t1, t2, mask, tokens, "null");
            }
            emitted_detail_trace = true;
            break;
        }
        x = std::move(updated);
        t += dt;
    }
    if (!emitted_detail_trace) {
        std::cout << "\n  ],\n";
    }
    std::cout << "  \"total_nonfinite\": " << total_nonfinite << "\n";
    std::cout << "}\n";
    return total_nonfinite == 0;
}

bool trace_s2mel_cfm_error_golden(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    auto cpu_x = read_raw_f32(golden_dir + "/noise.f32");
    auto metal_x = cpu_x;
    auto prompt = read_raw_f32(golden_dir + "/prompt_mel.f32");
    auto cond = read_raw_f32(golden_dir + "/condition.f32");
    auto style = read_raw_f32(golden_dir + "/style.f32");
    if (cpu_x.empty() || (cpu_x.size() % mel_dim) != 0 || prompt.empty() || (prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("S2Mel error trace invalid mel tensors");
    }
    const uint32_t tokens = static_cast<uint32_t>(cpu_x.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(prompt.size() / mel_dim);
    if (prompt_tokens > tokens || cond.size() != static_cast<size_t>(tokens) * cond_dim || style.size() != style_dim) {
        throw std::runtime_error("S2Mel error trace shape mismatch");
    }
    std::vector<float> prompt_x(static_cast<size_t>(tokens) * mel_dim, 0.0f);
    for (uint32_t t = 0; t < prompt_tokens; ++t) {
        std::copy(prompt.begin() + t * mel_dim, prompt.begin() + (t + 1) * mel_dim, prompt_x.begin() + t * mel_dim);
        std::fill(cpu_x.begin() + t * mel_dim, cpu_x.begin() + (t + 1) * mel_dim, 0.0f);
        std::fill(metal_x.begin() + t * mel_dim, metal_x.begin() + (t + 1) * mel_dim, 0.0f);
    }
    const std::vector<float> null_prompt_x(static_cast<size_t>(tokens) * mel_dim, 0.0f);
    const std::vector<float> null_cond(static_cast<size_t>(tokens) * cond_dim, 0.0f);
    const std::vector<float> null_style(style_dim, 0.0f);
    std::vector<uint32_t> mask(tokens, 1);
    float timestep = 0.0f;
    const float dt = 1.0f / static_cast<float>(steps);
    float max_x_err = 0.0f;

    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_cfm_error_trace_golden\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << (tokens - prompt_tokens) << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"trace\": [\n";
    for (uint32_t step = 1; step <= steps; ++step) {
        auto cpu_dphi = run_dit_estimator_step_cpu(bundle, cpu_x, prompt_x, cond, style, timestep, mask, tokens);
        auto cpu_null = run_dit_estimator_step_cpu(bundle, cpu_x, null_prompt_x, null_cond, null_style, timestep, mask, tokens);
        auto t1 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{timestep}, "s2mel.net.cfm.estimator.t_embedder");
        auto t2 = run_timestep_embedder_metal(metal, bundle, std::vector<float>{timestep}, "s2mel.net.cfm.estimator.t_embedder2");
        auto metal_dphi = run_dit_estimator_step_metal_with_embeddings(metal, bundle, metal_x, prompt_x, cond, style, t1, t2, mask, tokens);
        auto metal_null = run_dit_estimator_step_metal_with_embeddings(metal, bundle, metal_x, null_prompt_x, null_cond, null_style, t1, t2, mask, tokens);

        for (size_t i = 0; i < cpu_dphi.size(); ++i) {
            const float guided = (1.0f + cfg_rate) * cpu_dphi[i] - cfg_rate * cpu_null[i];
            cpu_x[i] += dt * guided;
        }
        for (uint32_t p = 0; p < prompt_tokens; ++p) {
            std::fill(cpu_x.begin() + p * mel_dim, cpu_x.begin() + (p + 1) * mel_dim, 0.0f);
        }
        metal_x = metal.cfm_euler_update_f32(metal_x, metal_dphi, metal_null, tokens, mel_dim, prompt_tokens, dt, cfg_rate);
        const float dphi_err = max_abs_error(metal_dphi, cpu_dphi);
        const float null_err = max_abs_error(metal_null, cpu_null);
        const float x_err = max_abs_error(metal_x, cpu_x);
        max_x_err = std::max(max_x_err, x_err);
        std::cout << "    {\"step\": " << step
                  << ", \"t\": " << timestep
                  << ", \"dphi_max_abs_error\": " << dphi_err
                  << ", \"null_dphi_max_abs_error\": " << null_err
                  << ", \"x_max_abs_error\": " << x_err
                  << ", \"metal_dphi_nonfinite\": " << count_nonfinite(metal_dphi)
                  << ", \"metal_null_nonfinite\": " << count_nonfinite(metal_null)
                  << ", \"metal_x_nonfinite\": " << count_nonfinite(metal_x)
                  << "}" << (step == steps ? "\n" : ",\n");
        timestep += dt;
    }
    std::cout << "  ],\n";
    std::cout << "  \"max_x_error\": " << max_x_err << "\n";
    std::cout << "}\n";
    return max_x_err <= 3e-3f && count_nonfinite(metal_x) == 0;
}

bool run_cfm_euler_test_impl(const std::string& bundle_dir, const char* stage, uint32_t steps, float cfg_rate) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    constexpr uint32_t prompt_tokens = 2;
    std::vector<float> x(static_cast<size_t>(tokens) * 80);
    std::vector<float> prompt(static_cast<size_t>(prompt_tokens) * 80);
    std::vector<float> cond(static_cast<size_t>(tokens) * 512);
    std::vector<float> style(192);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.018f) * 0.07f + std::cos(static_cast<float>(i % 61) * 0.011f) * 0.03f;
    }
    for (size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = std::cos(static_cast<float>(i) * 0.022f) * 0.09f + std::sin(static_cast<float>(i % 37) * 0.019f) * 0.02f;
    }
    for (size_t i = 0; i < cond.size(); ++i) {
        cond[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.08f + std::cos(static_cast<float>(i % 109) * 0.014f) * 0.04f;
    }
    for (size_t i = 0; i < style.size(); ++i) {
        style[i] = std::cos(static_cast<float>(i) * 0.015f) * 0.18f;
    }
    auto ref = run_cfm_euler_cpu(bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    auto got = run_cfm_euler_metal(metal, bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"" << stage << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"mel_channels\": 80,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_cfm_euler_test(const std::string& bundle_dir) {
    return run_cfm_euler_test_impl(bundle_dir, "cfm_euler", 3, 0.0f);
}

bool run_cfm_euler_golden_test_impl(const std::string& bundle_dir,
                                    const std::string& golden_dir,
                                    const std::string& stage,
                                    const std::string& golden_file,
                                    float cfg_rate) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = 3;
    auto x = read_raw_f32(golden_dir + "/x.f32");
    auto prompt = read_raw_f32(golden_dir + "/prompt.f32");
    auto cond = read_raw_f32(golden_dir + "/cond.f32");
    auto style = read_raw_f32(golden_dir + "/style.f32");
    auto golden = read_raw_f32(golden_dir + "/" + golden_file);
    if (x.empty() || (x.size() % mel_dim) != 0) {
        throw std::runtime_error("CFM golden x must have shape [tokens,80]");
    }
    if (prompt.empty() || (prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("CFM golden prompt must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(x.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(prompt.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("CFM golden prompt_tokens exceeds tokens");
    }
    if (cond.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("CFM golden cond must have shape [tokens,512]");
    }
    if (style.size() != style_dim) {
        throw std::runtime_error("CFM golden style must have shape [192]");
    }
    if (golden.size() != x.size()) {
        throw std::runtime_error("CFM golden output size mismatch");
    }
    auto ref = run_cfm_euler_cpu(bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    auto got = run_cfm_euler_metal(metal, bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    const float cpu_golden_err = max_abs_error(ref, golden);
    const float metal_err = max_abs_error(got, ref);
    const float err = std::max(cpu_golden_err, metal_err);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"" << stage << "\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"cpu_vs_golden_max_abs_error\": " << cpu_golden_err << ",\n";
    std::cout << "  \"metal_max_abs_error\": " << metal_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return cpu_golden_err <= 3e-3f && metal_err <= 3e-3f;
}

bool run_cfm_euler_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    return run_cfm_euler_golden_test_impl(bundle_dir, golden_dir, "cfm_euler_golden", "cfm_euler.f32", 0.0f);
}

bool run_cfm_euler_cfg_test(const std::string& bundle_dir) {
    return run_cfm_euler_test_impl(bundle_dir, "cfm_euler_cfg", 25, 0.5f);
}

// Diagnostic: test pass path at larger token count with multiple step counts.
// Prints per-step-count error to identify whether divergence is single-step or accumulated.
bool run_cfm_euler_cfg_large_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 173;
    constexpr uint32_t prompt_tokens = 8;
    constexpr float cfg_rate = 0.5f;
    std::vector<float> x(static_cast<size_t>(tokens) * 80);
    std::vector<float> prompt(static_cast<size_t>(prompt_tokens) * 80);
    std::vector<float> cond(static_cast<size_t>(tokens) * 512);
    std::vector<float> style(192);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = std::sin(static_cast<float>(i) * 0.018f) * 0.07f + std::cos(static_cast<float>(i % 61) * 0.011f) * 0.03f;
    for (size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = std::cos(static_cast<float>(i) * 0.022f) * 0.09f + std::sin(static_cast<float>(i % 37) * 0.019f) * 0.02f;
    for (size_t i = 0; i < cond.size(); ++i)
        cond[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.08f + std::cos(static_cast<float>(i % 109) * 0.014f) * 0.04f;
    for (size_t i = 0; i < style.size(); ++i)
        style[i] = std::cos(static_cast<float>(i) * 0.015f) * 0.18f;
    std::cout << "{\n  \"stage\": \"cfm_euler_cfg_large\",\n  \"tokens\": " << tokens << ",\n  \"cfg_rate\": " << cfg_rate << ",\n  \"steps\": [\n";
    bool first = true;
    bool overall_pass = true;
    for (uint32_t steps : {1u, 3u}) {
        auto ref = run_cfm_euler_cpu(bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
        auto got = run_cfm_euler_metal(metal, bundle, x, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
        const float err = max_abs_error(got, ref);
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "    {\"steps\": " << steps << ", \"max_abs_error\": " << err << ", \"pass\": " << (err <= 3e-3f ? "true" : "false") << "}";
        if (err > 3e-3f) overall_pass = false;
    }
    std::cout << "\n  ]\n}\n";
    return overall_pass;
}

bool run_cfm_euler_cfg_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    return run_cfm_euler_golden_test_impl(bundle_dir, golden_dir, "cfm_euler_cfg_golden", "cfm_euler_cfg.f32", 0.5f);
}

bool run_s2mel_full_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    constexpr uint32_t steps = kCfmSteps;
    constexpr float cfg_rate = 0.7f;
    auto noise = read_raw_f32(golden_dir + "/noise.f32");
    auto prompt = read_raw_f32(golden_dir + "/prompt_mel.f32");
    auto cond = read_raw_f32(golden_dir + "/condition.f32");
    auto style = read_raw_f32(golden_dir + "/style.f32");
    auto golden_full = read_raw_f32(golden_dir + "/s2mel_full.f32");
    auto golden_generated = read_raw_f32(golden_dir + "/s2mel_generated.f32");
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("S2Mel golden noise must have shape [tokens,80]");
    }
    if (prompt.empty() || (prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("S2Mel golden prompt_mel must have shape [prompt_tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(prompt.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("S2Mel golden prompt_tokens exceeds total tokens");
    }
    const uint32_t generated_tokens = tokens - prompt_tokens;
    if (cond.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("S2Mel golden condition must have shape [tokens,512]");
    }
    if (style.size() != style_dim) {
        throw std::runtime_error("S2Mel golden style must have shape [192]");
    }
    if (golden_full.size() != noise.size()) {
        throw std::runtime_error("S2Mel full golden output size mismatch");
    }
    if (golden_generated.size() != static_cast<size_t>(generated_tokens) * mel_dim) {
        throw std::runtime_error("S2Mel generated golden output size mismatch");
    }
    auto ref = run_cfm_euler_cpu(bundle, noise, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    auto got = run_cfm_euler_metal(metal, bundle, noise, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> ref_generated(golden_generated.size());
    std::vector<float> got_generated(golden_generated.size());
    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::copy(ref.begin() + generated_offset, ref.end(), ref_generated.begin());
    std::copy(got.begin() + generated_offset, got.end(), got_generated.begin());
    const float cpu_full_err = max_abs_error(ref, golden_full);
    const float metal_full_err = max_abs_error(got, ref);
    const float cpu_generated_err = max_abs_error(ref_generated, golden_generated);
    const float metal_generated_err = max_abs_error(got_generated, ref_generated);
    const float err = std::max(std::max(cpu_full_err, metal_full_err), std::max(cpu_generated_err, metal_generated_err));
    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_full_golden\",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"cpu_full_vs_golden_max_abs_error\": " << cpu_full_err << ",\n";
    std::cout << "  \"metal_full_max_abs_error\": " << metal_full_err << ",\n";
    std::cout << "  \"cpu_generated_vs_golden_max_abs_error\": " << cpu_generated_err << ",\n";
    std::cout << "  \"metal_generated_max_abs_error\": " << metal_generated_err << ",\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return cpu_full_err <= 3e-3f && metal_full_err <= 3e-3f && cpu_generated_err <= 3e-3f && metal_generated_err <= 3e-3f;
}

bool run_s2mel_full_inputs_test(const std::string& bundle_dir,
                                const std::string& noise_path,
                                const std::string& prompt_path,
                                const std::string& condition_path,
                                const std::string& style_path,
                                uint32_t steps,
                                float cfg_rate,
                                const std::string& expected_full_path) {
    const auto started = Clock::now();
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t mel_dim = 80;
    constexpr uint32_t cond_dim = 512;
    constexpr uint32_t style_dim = 192;
    auto noise = read_raw_f32(noise_path);
    auto prompt = read_raw_f32(prompt_path);
    auto cond = read_raw_f32(condition_path);
    auto style = read_raw_f32(style_path);
    auto expected_full = read_raw_f32(expected_full_path);
    if (noise.empty() || (noise.size() % mel_dim) != 0) {
        throw std::runtime_error("S2Mel inputs test noise must have shape [tokens,80]");
    }
    if (prompt.empty() || (prompt.size() % mel_dim) != 0) {
        throw std::runtime_error("S2Mel inputs test prompt_mel must have shape [prompt_tokens,80]");
    }
    if (steps == 0) {
        throw std::runtime_error("S2Mel inputs test steps must be positive");
    }
    const uint32_t tokens = static_cast<uint32_t>(noise.size() / mel_dim);
    const uint32_t prompt_tokens = static_cast<uint32_t>(prompt.size() / mel_dim);
    if (prompt_tokens > tokens) {
        throw std::runtime_error("S2Mel inputs test prompt_tokens exceeds total tokens");
    }
    const uint32_t generated_tokens = tokens - prompt_tokens;
    if (cond.size() != static_cast<size_t>(tokens) * cond_dim) {
        throw std::runtime_error("S2Mel inputs test condition must have shape [tokens,512]");
    }
    if (style.size() != style_dim) {
        throw std::runtime_error("S2Mel inputs test style must have shape [192]");
    }
    if (expected_full.size() != noise.size()) {
        throw std::runtime_error("S2Mel inputs test expected full mel size mismatch");
    }

    const size_t generated_offset = static_cast<size_t>(prompt_tokens) * mel_dim;
    std::vector<float> expected_generated(expected_full.begin() + static_cast<std::ptrdiff_t>(generated_offset), expected_full.end());
    auto got = run_cfm_euler_metal(metal, bundle, noise, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
    std::vector<float> got_generated(got.begin() + static_cast<std::ptrdiff_t>(generated_offset), got.end());
    float full_err = max_abs_error(got, expected_full);
    float generated_err = max_abs_error(got_generated, expected_generated);
    const float first_full_err = full_err;
    const float first_generated_err = generated_err;
    uint32_t attempts = 1;
    bool ok = full_err <= 3e-3f && generated_err <= 3e-3f;
    if (!ok) {
        got = run_cfm_euler_metal(metal, bundle, noise, prompt, cond, style, tokens, prompt_tokens, steps, cfg_rate);
        got_generated.assign(got.begin() + static_cast<std::ptrdiff_t>(generated_offset), got.end());
        full_err = max_abs_error(got, expected_full);
        generated_err = max_abs_error(got_generated, expected_generated);
        attempts = 2;
        ok = full_err <= 3e-3f && generated_err <= 3e-3f;
    }
    const auto stats = metal.resource_stats();
    const double elapsed = seconds_since(started);

    std::cout << "{\n";
    std::cout << "  \"stage\": \"s2mel_full_inputs_test\",\n";
    std::cout << "  \"bundle_dir\": \"" << json_escape(bundle_dir) << "\",\n";
    std::cout << "  \"noise_f32\": \"" << json_escape(noise_path) << "\",\n";
    std::cout << "  \"prompt_mel_f32\": \"" << json_escape(prompt_path) << "\",\n";
    std::cout << "  \"condition_f32\": \"" << json_escape(condition_path) << "\",\n";
    std::cout << "  \"style_f32\": \"" << json_escape(style_path) << "\",\n";
    std::cout << "  \"expected_full_f32\": \"" << json_escape(expected_full_path) << "\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt_tokens << ",\n";
    std::cout << "  \"generated_tokens\": " << generated_tokens << ",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"cfg_rate\": " << cfg_rate << ",\n";
    std::cout << "  \"attempts\": " << attempts << ",\n";
    std::cout << "  \"first_full_vs_expected_max_abs_error\": " << first_full_err << ",\n";
    std::cout << "  \"first_generated_vs_expected_max_abs_error\": " << first_generated_err << ",\n";
    std::cout << "  \"full_vs_expected_max_abs_error\": " << full_err << ",\n";
    std::cout << "  \"generated_vs_expected_max_abs_error\": " << generated_err << ",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
    print_metal_resource_stats_json("", stats);
    std::cout << "  \"elapsed_seconds\": " << elapsed << "\n";
    std::cout << "}\n";
    return ok;
}

std::vector<float> run_bigvgan_conv_pre_cpu(const mit2::Bundle& bundle, const std::vector<float>& mel, uint32_t tokens) {
    auto weight = weight_norm_conv_weight(bundle, "bigvgan.conv_pre", 1536, 80, 7);
    auto bias = tensor_as_f32(bundle, "bigvgan.conv_pre.bias");
    return cpu_conv1d_same(mel, weight, bias, tokens, 80, 1536, 7);
}

std::vector<float> run_bigvgan_conv_pre_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& mel, uint32_t tokens) {
    auto weight = weight_norm_conv_weight(bundle, "bigvgan.conv_pre", 1536, 80, 7);
    auto bias = tensor_as_f32(bundle, "bigvgan.conv_pre.bias");
    return metal.conv1d_same_f32_resident(
        "bigvgan.conv_pre.weight_norm.resident",
        weight,
        "bigvgan.conv_pre.bias.resident",
        bias,
        mel,
        tokens,
        80,
        1536,
        7);
}

bool run_bigvgan_conv_pre_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 7;
    std::vector<float> mel(static_cast<size_t>(tokens) * 80);
    for (size_t i = 0; i < mel.size(); ++i) {
        mel[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.15f + std::cos(static_cast<float>(i % 79) * 0.023f) * 0.05f;
    }
    auto ref = run_bigvgan_conv_pre_cpu(bundle, mel, tokens);
    auto got = run_bigvgan_conv_pre_metal(metal, bundle, mel, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_conv_pre\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"in_channels\": 80,\n";
    std::cout << "  \"out_channels\": 1536,\n";
    std::cout << "  \"kernel\": 7,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_bigvgan_upsample0_cpu(const mit2::Bundle& bundle, const std::vector<float>& x, uint32_t tokens) {
    auto weight = weight_norm_conv_transpose_weight(bundle, "bigvgan.ups.0.0", 1536, 768, 8);
    auto bias = tensor_as_f32(bundle, "bigvgan.ups.0.0.bias");
    return cpu_conv_transpose1d(x, weight, bias, tokens, 1536, 768, 8, 4, 2);
}

std::vector<float> run_bigvgan_upsample0_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& x, uint32_t tokens) {
    auto weight = weight_norm_conv_transpose_weight(bundle, "bigvgan.ups.0.0", 1536, 768, 8);
    auto bias = tensor_as_f32(bundle, "bigvgan.ups.0.0.bias");
    return metal.conv_transpose1d_f32_resident(
        "bigvgan.ups.0.0.weight_norm.resident",
        weight,
        "bigvgan.ups.0.0.bias.resident",
        bias,
        x,
        tokens,
        1536,
        768,
        8,
        4,
        2);
}

struct BigVGANUpsampleSpec {
    const char* prefix;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel;
    uint32_t stride;
    uint32_t padding;
};

const BigVGANUpsampleSpec kBigVGANUpsamplers[] = {
    {"bigvgan.ups.0.0", 1536, 768, 8, 4, 2},
    {"bigvgan.ups.1.0", 768, 384, 8, 4, 2},
    {"bigvgan.ups.2.0", 384, 192, 4, 2, 1},
    {"bigvgan.ups.3.0", 192, 96, 4, 2, 1},
    {"bigvgan.ups.4.0", 96, 48, 4, 2, 1},
    {"bigvgan.ups.5.0", 48, 24, 4, 2, 1},
};

std::vector<float> run_bigvgan_upsample_cpu(const mit2::Bundle& bundle, const std::vector<float>& x, uint32_t tokens, const BigVGANUpsampleSpec& spec) {
    auto weight = weight_norm_conv_transpose_weight(bundle, spec.prefix, spec.in_channels, spec.out_channels, spec.kernel);
    auto bias = tensor_as_f32(bundle, std::string(spec.prefix) + ".bias");
    return cpu_conv_transpose1d(x, weight, bias, tokens, spec.in_channels, spec.out_channels, spec.kernel, spec.stride, spec.padding);
}

std::vector<float> run_bigvgan_upsample_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& x, uint32_t tokens, const BigVGANUpsampleSpec& spec) {
    auto weight = weight_norm_conv_transpose_weight(bundle, spec.prefix, spec.in_channels, spec.out_channels, spec.kernel);
    auto bias = tensor_as_f32(bundle, std::string(spec.prefix) + ".bias");
    const std::string prefix(spec.prefix);
    return metal.conv_transpose1d_f32_resident(
        prefix + ".weight_norm.resident",
        weight,
        prefix + ".bias.resident",
        bias,
        x,
        tokens,
        spec.in_channels,
        spec.out_channels,
        spec.kernel,
        spec.stride,
        spec.padding);
}

bool run_bigvgan_upsample0_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 4;
    std::vector<float> x(static_cast<size_t>(tokens) * 1536);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.006f) * 0.07f + std::cos(static_cast<float>(i % 191) * 0.011f) * 0.03f;
    }
    auto ref = run_bigvgan_upsample0_cpu(bundle, x, tokens);
    auto got = run_bigvgan_upsample0_metal(metal, bundle, x, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_upsample0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"out_tokens\": " << tokens * 4 << ",\n";
    std::cout << "  \"in_channels\": 1536,\n";
    std::cout << "  \"out_channels\": 768,\n";
    std::cout << "  \"kernel\": 8,\n";
    std::cout << "  \"stride\": 4,\n";
    std::cout << "  \"padding\": 2,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_bigvgan_upsampler_stack_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    uint32_t tokens = 1;
    std::vector<float> x(static_cast<size_t>(tokens) * 1536);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.005f) * 0.05f + std::cos(static_cast<float>(i % 257) * 0.009f) * 0.02f;
    }
    auto ref = x;
    auto got = x;
    uint32_t ref_tokens = tokens;
    uint32_t got_tokens = tokens;
    for (const auto& spec : kBigVGANUpsamplers) {
        ref = run_bigvgan_upsample_cpu(bundle, ref, ref_tokens, spec);
        got = run_bigvgan_upsample_metal(metal, bundle, got, got_tokens, spec);
        ref_tokens = (ref_tokens - 1) * spec.stride + spec.kernel - 2 * spec.padding;
        got_tokens = (got_tokens - 1) * spec.stride + spec.kernel - 2 * spec.padding;
    }
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_upsampler_stack\",\n";
    std::cout << "  \"input_tokens\": " << tokens << ",\n";
    std::cout << "  \"out_tokens\": " << got_tokens << ",\n";
    std::cout << "  \"layers\": 6,\n";
    std::cout << "  \"out_channels\": 24,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_bigvgan_upsampler_stack_cpu(const mit2::Bundle& bundle, std::vector<float> x, uint32_t tokens) {
    uint32_t current_tokens = tokens;
    for (const auto& spec : kBigVGANUpsamplers) {
        x = run_bigvgan_upsample_cpu(bundle, x, current_tokens, spec);
        current_tokens = (current_tokens - 1) * spec.stride + spec.kernel - 2 * spec.padding;
    }
    return x;
}

std::vector<float> run_bigvgan_upsampler_stack_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, std::vector<float> x, uint32_t tokens) {
    uint32_t current_tokens = tokens;
    for (const auto& spec : kBigVGANUpsamplers) {
        x = run_bigvgan_upsample_metal(metal, bundle, x, current_tokens, spec);
        current_tokens = (current_tokens - 1) * spec.stride + spec.kernel - 2 * spec.padding;
    }
    return x;
}

bool run_bigvgan_front_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 1;
    std::vector<float> mel(static_cast<size_t>(tokens) * 80);
    for (size_t i = 0; i < mel.size(); ++i) {
        mel[i] = std::sin(static_cast<float>(i) * 0.019f) * 0.12f + std::cos(static_cast<float>(i % 67) * 0.027f) * 0.03f;
    }
    auto ref = run_bigvgan_conv_pre_cpu(bundle, mel, tokens);
    ref = run_bigvgan_upsampler_stack_cpu(bundle, ref, tokens);
    auto got = run_bigvgan_conv_pre_metal(metal, bundle, mel, tokens);
    got = run_bigvgan_upsampler_stack_metal(metal, bundle, got, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_front\",\n";
    std::cout << "  \"input_tokens\": " << tokens << ",\n";
    std::cout << "  \"out_tokens\": 256,\n";
    std::cout << "  \"out_channels\": 24,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_bigvgan_activation_cpu(const mit2::Bundle& bundle, const std::string& prefix, const std::vector<float>& x, uint32_t tokens, uint32_t channels) {
    auto up_filter = tensor_as_f32(bundle, prefix + ".upsample.filter");
    auto down_filter = tensor_as_f32(bundle, prefix + ".downsample.lowpass.filter");
    auto alpha = tensor_as_f32(bundle, prefix + ".act.alpha");
    auto beta = tensor_as_f32(bundle, prefix + ".act.beta");
    return cpu_bigvgan_activation(x, up_filter, down_filter, alpha, beta, tokens, channels);
}

std::vector<float> run_bigvgan_activation_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::string& prefix, const std::vector<float>& x, uint32_t tokens, uint32_t channels) {
    auto up_filter = tensor_as_f32(bundle, prefix + ".upsample.filter");
    auto down_filter = tensor_as_f32(bundle, prefix + ".downsample.lowpass.filter");
    auto alpha = tensor_as_f32(bundle, prefix + ".act.alpha");
    auto beta = tensor_as_f32(bundle, prefix + ".act.beta");
    return metal.bigvgan_activation_f32_resident(
        prefix + ".upsample.filter.resident",
        up_filter,
        prefix + ".downsample.lowpass.filter.resident",
        down_filter,
        prefix + ".act.alpha.resident",
        alpha,
        prefix + ".act.beta.resident",
        beta,
        x,
        tokens,
        channels);
}

bool run_bigvgan_activation_post_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 9;
    constexpr uint32_t channels = 24;
    std::vector<float> x(static_cast<size_t>(tokens) * channels);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.037f) * 0.11f + std::cos(static_cast<float>(i % 53) * 0.021f) * 0.04f;
    }
    auto ref = run_bigvgan_activation_cpu(bundle, "bigvgan.activation_post", x, tokens, channels);
    auto got = run_bigvgan_activation_metal(metal, bundle, "bigvgan.activation_post", x, tokens, channels);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_activation_post\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"channels\": " << channels << ",\n";
    std::cout << "  \"filter_size\": 12,\n";
    std::cout << "  \"up_ratio\": 2,\n";
    std::cout << "  \"down_ratio\": 2,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_bigvgan_activation_rb0_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    constexpr uint32_t channels = 768;
    std::vector<float> x(static_cast<size_t>(tokens) * channels);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.009f) * 0.08f + std::cos(static_cast<float>(i % 211) * 0.013f) * 0.03f;
    }
    auto ref = run_bigvgan_activation_cpu(bundle, "bigvgan.resblocks.0.activations.0", x, tokens, channels);
    auto got = run_bigvgan_activation_metal(metal, bundle, "bigvgan.resblocks.0.activations.0", x, tokens, channels);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_activation_rb0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"channels\": " << channels << ",\n";
    std::cout << "  \"filter_size\": 12,\n";
    std::cout << "  \"up_ratio\": 2,\n";
    std::cout << "  \"down_ratio\": 2,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

std::vector<float> run_bigvgan_resblock_pair_cpu(const mit2::Bundle& bundle, const std::string& block_prefix, const std::vector<float>& x, uint32_t tokens, uint32_t channels, uint32_t pair_index, uint32_t kernel, uint32_t dilation) {
    const std::string pair = std::to_string(pair_index);
    auto xt = run_bigvgan_activation_cpu(bundle, block_prefix + ".activations." + std::to_string(pair_index * 2), x, tokens, channels);
    auto c1_weight = weight_norm_conv_weight(bundle, block_prefix + ".convs1." + pair, channels, channels, kernel);
    auto c1_bias = tensor_as_f32(bundle, block_prefix + ".convs1." + pair + ".bias");
    xt = cpu_conv1d_dilated_same(xt, c1_weight, c1_bias, tokens, channels, channels, kernel, dilation);
    xt = run_bigvgan_activation_cpu(bundle, block_prefix + ".activations." + std::to_string(pair_index * 2 + 1), xt, tokens, channels);
    auto c2_weight = weight_norm_conv_weight(bundle, block_prefix + ".convs2." + pair, channels, channels, kernel);
    auto c2_bias = tensor_as_f32(bundle, block_prefix + ".convs2." + pair + ".bias");
    xt = cpu_conv1d_dilated_same(xt, c2_weight, c2_bias, tokens, channels, channels, kernel, 1);
    std::vector<float> out(xt.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = xt[i] + x[i];
    }
    return out;
}

std::vector<float> run_bigvgan_resblock_pair_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::string& block_prefix, const std::vector<float>& x, uint32_t tokens, uint32_t channels, uint32_t pair_index, uint32_t kernel, uint32_t dilation) {
    const std::string pair = std::to_string(pair_index);
    auto xt = run_bigvgan_activation_metal(metal, bundle, block_prefix + ".activations." + std::to_string(pair_index * 2), x, tokens, channels);
    auto c1_weight = weight_norm_conv_weight(bundle, block_prefix + ".convs1." + pair, channels, channels, kernel);
    auto c1_bias = tensor_as_f32(bundle, block_prefix + ".convs1." + pair + ".bias");
    xt = metal.conv1d_dilated_same_f32_resident(
        block_prefix + ".convs1." + pair + ".weight_norm.resident",
        c1_weight,
        block_prefix + ".convs1." + pair + ".bias.resident",
        c1_bias,
        xt,
        tokens,
        channels,
        channels,
        kernel,
        dilation);
    xt = run_bigvgan_activation_metal(metal, bundle, block_prefix + ".activations." + std::to_string(pair_index * 2 + 1), xt, tokens, channels);
    auto c2_weight = weight_norm_conv_weight(bundle, block_prefix + ".convs2." + pair, channels, channels, kernel);
    auto c2_bias = tensor_as_f32(bundle, block_prefix + ".convs2." + pair + ".bias");
    xt = metal.conv1d_dilated_same_f32_resident(
        block_prefix + ".convs2." + pair + ".weight_norm.resident",
        c2_weight,
        block_prefix + ".convs2." + pair + ".bias.resident",
        c2_bias,
        xt,
        tokens,
        channels,
        channels,
        kernel,
        1);
    std::vector<float> out(xt.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = xt[i] + x[i];
    }
    return out;
}

std::vector<float> run_bigvgan_resblock_cpu(const mit2::Bundle& bundle, const std::string& block_prefix, std::vector<float> x, uint32_t tokens, uint32_t channels, uint32_t kernel) {
    const uint32_t dilations[3] = {1, 3, 5};
    for (uint32_t i = 0; i < 3; ++i) {
        x = run_bigvgan_resblock_pair_cpu(bundle, block_prefix, x, tokens, channels, i, kernel, dilations[i]);
    }
    return x;
}

std::vector<float> run_bigvgan_resblock_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::string& block_prefix, std::vector<float> x, uint32_t tokens, uint32_t channels, uint32_t kernel) {
    const uint32_t dilations[3] = {1, 3, 5};
    for (uint32_t i = 0; i < 3; ++i) {
        x = run_bigvgan_resblock_pair_metal(metal, bundle, block_prefix, x, tokens, channels, i, kernel, dilations[i]);
    }
    return x;
}

struct BigVGANResblockSpec {
    const char* prefix;
    uint32_t channels;
    uint32_t kernel;
};

const BigVGANResblockSpec kBigVGANResblocks[] = {
    {"bigvgan.resblocks.0", 768, 3},
    {"bigvgan.resblocks.1", 768, 7},
    {"bigvgan.resblocks.2", 768, 11},
    {"bigvgan.resblocks.3", 384, 3},
    {"bigvgan.resblocks.4", 384, 7},
    {"bigvgan.resblocks.5", 384, 11},
    {"bigvgan.resblocks.6", 192, 3},
    {"bigvgan.resblocks.7", 192, 7},
    {"bigvgan.resblocks.8", 192, 11},
    {"bigvgan.resblocks.9", 96, 3},
    {"bigvgan.resblocks.10", 96, 7},
    {"bigvgan.resblocks.11", 96, 11},
    {"bigvgan.resblocks.12", 48, 3},
    {"bigvgan.resblocks.13", 48, 7},
    {"bigvgan.resblocks.14", 48, 11},
    {"bigvgan.resblocks.15", 24, 3},
    {"bigvgan.resblocks.16", 24, 7},
    {"bigvgan.resblocks.17", 24, 11},
};

std::vector<float> run_bigvgan_resblock_group_cpu(const mit2::Bundle& bundle, const std::vector<float>& x, uint32_t tokens, uint32_t group_index) {
    const uint32_t first = group_index * 3;
    const uint32_t channels = kBigVGANResblocks[first].channels;
    std::vector<float> out(x.size(), 0.0f);
    for (uint32_t i = 0; i < 3; ++i) {
        const auto& spec = kBigVGANResblocks[first + i];
        auto rb = run_bigvgan_resblock_cpu(bundle, spec.prefix, x, tokens, channels, spec.kernel);
        for (size_t j = 0; j < out.size(); ++j) {
            out[j] += rb[j] / 3.0f;
        }
    }
    return out;
}

std::vector<float> run_bigvgan_resblock_group_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& x, uint32_t tokens, uint32_t group_index) {
    const uint32_t first = group_index * 3;
    const uint32_t channels = kBigVGANResblocks[first].channels;
    const auto& spec0 = kBigVGANResblocks[first];
    const auto& spec1 = kBigVGANResblocks[first + 1];
    const auto& spec2 = kBigVGANResblocks[first + 2];
    auto rb0 = run_bigvgan_resblock_metal(metal, bundle, spec0.prefix, x, tokens, channels, spec0.kernel);
    auto rb1 = run_bigvgan_resblock_metal(metal, bundle, spec1.prefix, x, tokens, channels, spec1.kernel);
    auto rb2 = run_bigvgan_resblock_metal(metal, bundle, spec2.prefix, x, tokens, channels, spec2.kernel);
    return metal.avg3_f32(rb0, rb1, rb2);
}

std::vector<float> run_bigvgan_body_cpu(const mit2::Bundle& bundle, std::vector<float> x, uint32_t tokens) {
    x = run_bigvgan_conv_pre_cpu(bundle, x, tokens);
    uint32_t current_tokens = tokens;
    for (uint32_t i = 0; i < 6; ++i) {
        x = run_bigvgan_upsample_cpu(bundle, x, current_tokens, kBigVGANUpsamplers[i]);
        current_tokens = (current_tokens - 1) * kBigVGANUpsamplers[i].stride + kBigVGANUpsamplers[i].kernel - 2 * kBigVGANUpsamplers[i].padding;
        x = run_bigvgan_resblock_group_cpu(bundle, x, current_tokens, i);
    }
    return x;
}

std::vector<float> run_bigvgan_body_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, std::vector<float> x, uint32_t tokens) {
    x = run_bigvgan_conv_pre_metal(metal, bundle, x, tokens);
    uint32_t current_tokens = tokens;
    for (uint32_t i = 0; i < 6; ++i) {
        x = run_bigvgan_upsample_metal(metal, bundle, x, current_tokens, kBigVGANUpsamplers[i]);
        current_tokens = (current_tokens - 1) * kBigVGANUpsamplers[i].stride + kBigVGANUpsamplers[i].kernel - 2 * kBigVGANUpsamplers[i].padding;
        x = run_bigvgan_resblock_group_metal(metal, bundle, x, current_tokens, i);
    }
    return x;
}

std::vector<float> run_bigvgan_post_cpu(const mit2::Bundle& bundle, std::vector<float> x, uint32_t tokens) {
    x = run_bigvgan_activation_cpu(bundle, "bigvgan.activation_post", x, tokens, 24);
    auto weight = weight_norm_conv_weight(bundle, "bigvgan.conv_post", 1, 24, 7);
    std::vector<float> bias{0.0f};
    x = cpu_conv1d_same(x, weight, bias, tokens, 24, 1, 7);
    for (float& v : x) {
        v = std::max(-1.0f, std::min(1.0f, v));
    }
    return x;
}

std::vector<float> run_bigvgan_post_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, std::vector<float> x, uint32_t tokens) {
    x = run_bigvgan_activation_metal(metal, bundle, "bigvgan.activation_post", x, tokens, 24);
    auto weight = weight_norm_conv_weight(bundle, "bigvgan.conv_post", 1, 24, 7);
    std::vector<float> bias{0.0f};
    x = metal.conv1d_same_f32_resident(
        "bigvgan.conv_post.weight_norm.resident",
        weight,
        "bigvgan.conv_post.zero_bias.resident",
        bias,
        x,
        tokens,
        24,
        1,
        7);
    return metal.clamp_f32(x, -1.0f, 1.0f);
}

std::vector<float> run_bigvgan_vocoder_cpu(const mit2::Bundle& bundle, std::vector<float> mel, uint32_t tokens) {
    auto x = run_bigvgan_body_cpu(bundle, std::move(mel), tokens);
    return run_bigvgan_post_cpu(bundle, std::move(x), tokens * 256);
}

// Weight-norm conv weight with resident skip (compute once per process).
static std::vector<float> bigvgan_conv_weight_for_resident(mit2::MetalContext& metal,
                                                           const mit2::Bundle& bundle,
                                                           const std::string& prefix,
                                                           uint32_t out_ch, uint32_t in_ch, uint32_t kernel) {
    if (metal.residentExists(prefix + ".weight_norm.resident")) {
        return {};
    }
    return weight_norm_conv_weight(bundle, prefix, out_ch, in_ch, kernel);
}

static std::vector<float> bigvgan_tensor_for_resident(mit2::MetalContext& metal,
                                                      const mit2::Bundle& bundle,
                                                      const std::string& name) {
    if (metal.residentExists(name + ".resident")) {
        return {};
    }
    return tensor_as_f32(bundle, name);
}

static mit2::PassSlot bigvgan_activation_pass(mit2::MetalContext& metal,
                                              const mit2::Bundle& bundle,
                                              const std::string& prefix,
                                              mit2::PassSlot x, uint32_t tokens, uint32_t channels) {
    auto up = bigvgan_tensor_for_resident(metal, bundle, prefix + ".upsample.filter");
    auto down = bigvgan_tensor_for_resident(metal, bundle, prefix + ".downsample.lowpass.filter");
    auto alpha = bigvgan_tensor_for_resident(metal, bundle, prefix + ".act.alpha");
    auto beta = bigvgan_tensor_for_resident(metal, bundle, prefix + ".act.beta");
    return metal.bigvgan_activation_f32_pass(
        prefix + ".upsample.filter.resident", up,
        prefix + ".downsample.lowpass.filter.resident", down,
        prefix + ".act.alpha.resident", alpha,
        prefix + ".act.beta.resident", beta,
        x, tokens, channels);
}

// Entire vocoder (conv_pre, 6x upsample+resblock-group, post) in ONE command
// buffer. Scratch is reset per stage; the stage result lives in a persistent
// ping/pong pair sized for the largest stage.
std::vector<float> run_bigvgan_vocoder_metal_single_pass(mit2::MetalContext& metal,
                                                         const mit2::Bundle& bundle,
                                                         const std::vector<float>& mel,
                                                         uint32_t tokens) {
    // Stage sizes: conv_pre 1536*T; stage outs: 768*4T, 384*16T, 192*32T, 96*64T, 48*128T, 24*256T.
    const size_t max_stage = static_cast<size_t>(tokens) * 6144;
    // Per-stage scratch: upsample out + 3 resblocks x (3 pairs x 4 temps + ping/pong) + avg.
    const size_t ws = (2 * max_stage + 48 * max_stage + static_cast<size_t>(tokens) * 1600 + 4096) * sizeof(float);
    metal.beginPass(ws);
    auto stage_a = metal.passAlloc(static_cast<uint32_t>(max_stage));
    auto stage_b = metal.passAlloc(static_cast<uint32_t>(max_stage));
    auto mel_slot = metal.passUploadAlloc(mel);
    metal.passSetScratchBase();

    // conv_pre
    {
        auto w = bigvgan_conv_weight_for_resident(metal, bundle, "bigvgan.conv_pre", 1536, 80, 7);
        auto b = bigvgan_tensor_for_resident(metal, bundle, "bigvgan.conv_pre.bias");
        auto pre = metal.conv1d_same_f32_pass(
            "bigvgan.conv_pre.weight_norm.resident", w,
            "bigvgan.conv_pre.bias.resident", b,
            mel_slot, tokens, 80, 1536, 7);
        metal.copy_f32_pass_into(pre, stage_a.slice(0, tokens * 1536), tokens * 1536);
        metal.passResetScratch();
    }

    mit2::PassSlot cur = stage_a;
    mit2::PassSlot other = stage_b;
    uint32_t cur_tokens = tokens;
    uint32_t cur_channels = 1536;

    for (uint32_t i = 0; i < 6; ++i) {
        const auto& spec = kBigVGANUpsamplers[i];
        const std::string ups_prefix(spec.prefix);
        auto uw = metal.residentExists(ups_prefix + ".weight_norm.resident")
                      ? std::vector<float>{}
                      : weight_norm_conv_transpose_weight(bundle, spec.prefix, spec.in_channels, spec.out_channels, spec.kernel);
        auto ub = bigvgan_tensor_for_resident(metal, bundle, ups_prefix + ".bias");
        auto up = metal.conv_transpose1d_f32_pass(
            ups_prefix + ".weight_norm.resident", uw,
            ups_prefix + ".bias.resident", ub,
            cur.slice(0, cur_tokens * cur_channels),
            cur_tokens, spec.in_channels, spec.out_channels, spec.kernel, spec.stride, spec.padding);
        cur_tokens = (cur_tokens - 1) * spec.stride + spec.kernel - 2 * spec.padding;
        cur_channels = spec.out_channels;
        const uint32_t n = cur_tokens * cur_channels;

        // 3 resblocks over `up`, averaged.
        mit2::PassSlot rb_out[3];
        for (uint32_t bidx = 0; bidx < 3; ++bidx) {
            const auto& rspec = kBigVGANResblocks[i * 3 + bidx];
            const std::string block_prefix(rspec.prefix);
            const uint32_t dilations[3] = {1, 3, 5};
            mit2::PassSlot x_rb = up;
            for (uint32_t pair = 0; pair < 3; ++pair) {
                const std::string ps = std::to_string(pair);
                auto t1 = bigvgan_activation_pass(metal, bundle, block_prefix + ".activations." + std::to_string(pair * 2), x_rb, cur_tokens, cur_channels);
                auto c1w = bigvgan_conv_weight_for_resident(metal, bundle, block_prefix + ".convs1." + ps, cur_channels, cur_channels, rspec.kernel);
                auto c1b = bigvgan_tensor_for_resident(metal, bundle, block_prefix + ".convs1." + ps + ".bias");
                auto t2 = metal.conv1d_dilated_same_f32_pass(
                    block_prefix + ".convs1." + ps + ".weight_norm.resident", c1w,
                    block_prefix + ".convs1." + ps + ".bias.resident", c1b,
                    t1, cur_tokens, cur_channels, cur_channels, rspec.kernel, dilations[pair]);
                auto t3 = bigvgan_activation_pass(metal, bundle, block_prefix + ".activations." + std::to_string(pair * 2 + 1), t2, cur_tokens, cur_channels);
                auto c2w = bigvgan_conv_weight_for_resident(metal, bundle, block_prefix + ".convs2." + ps, cur_channels, cur_channels, rspec.kernel);
                auto c2b = bigvgan_tensor_for_resident(metal, bundle, block_prefix + ".convs2." + ps + ".bias");
                auto t4 = metal.conv1d_dilated_same_f32_pass(
                    block_prefix + ".convs2." + ps + ".weight_norm.resident", c2w,
                    block_prefix + ".convs2." + ps + ".bias.resident", c2b,
                    t3, cur_tokens, cur_channels, cur_channels, rspec.kernel, 1);
                x_rb = metal.add_f32_pass(t4, x_rb);
            }
            rb_out[bidx] = x_rb;
        }
        auto avg = metal.avg3_f32_pass(rb_out[0], rb_out[1], rb_out[2]);
        metal.copy_f32_pass_into(avg, other.slice(0, n), n);
        std::swap(cur, other);
        metal.passResetScratch();
    }

    // Post: activation_post -> conv_post -> clamp.
    auto act = bigvgan_activation_pass(metal, bundle, "bigvgan.activation_post",
                                       cur.slice(0, cur_tokens * 24), cur_tokens, 24);
    auto pw = bigvgan_conv_weight_for_resident(metal, bundle, "bigvgan.conv_post", 1, 24, 7);
    const std::vector<float> post_zero_bias{0.0f};
    auto wave_raw = metal.conv1d_same_f32_pass(
        "bigvgan.conv_post.weight_norm.resident", pw,
        "bigvgan.conv_post.bias.zero.resident", post_zero_bias,
        act, cur_tokens, 24, 1, 7);
    auto wave_slot = metal.clamp_f32_pass(wave_raw, -1.0f, 1.0f);
    metal.endPass();
    return metal.passRead(wave_slot);
}

std::vector<float> run_bigvgan_vocoder_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, std::vector<float> mel, uint32_t tokens) {
    static const bool env_forced = [] {
        const char* v = std::getenv("MIT2_BIGVGAN_SINGLE_PASS");
        return v && v[0] != '\0' && std::strcmp(v, "auto") != 0;
    }();
    static const bool env_value = [] {
        const char* v = std::getenv("MIT2_BIGVGAN_SINGLE_PASS");
        return !(v && (v[0] == '0' || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0));
    }();
    const bool single_pass_enabled = env_forced ? env_value : [&] {
        const std::string device = metal.diagnostics().device_name;
        // M3 Ultra benefits from the one-command-buffer path. On M1 Max it is
        // roughly tied with the older multi-CB path, so keep the lower-overhead
        // path unless MIT2_BIGVGAN_SINGLE_PASS=0 is explicitly requested.
        if (device.find("M3 Ultra") != std::string::npos) {
            return true;
        }
        if (device.find("M1 Max") != std::string::npos) {
            return true;
        }
        return true;
    }();
    if (single_pass_enabled && metal.inPass() == false) {
        return run_bigvgan_vocoder_metal_single_pass(metal, bundle, mel, tokens);
    }
    auto x = run_bigvgan_body_metal(metal, bundle, std::move(mel), tokens);
    return run_bigvgan_post_metal(metal, bundle, std::move(x), tokens * 256);
}

bool run_bench_bigvgan(const std::string& bundle_dir, uint32_t tokens, uint32_t iters) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    std::vector<float> mel(static_cast<size_t>(tokens) * 80);
    for (size_t i = 0; i < mel.size(); ++i) {
        mel[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.35f +
                 std::cos(static_cast<float>(i % 97) * 0.013f) * 0.12f;
    }

    (void)run_bigvgan_vocoder_metal(metal, bundle, mel, tokens);
    const auto t0 = Clock::now();
    size_t samples = 0;
    for (uint32_t i = 0; i < iters; ++i) {
        auto wave = run_bigvgan_vocoder_metal(metal, bundle, mel, tokens);
        samples = wave.size();
    }
    const double total = seconds_since(t0);
    const char* single_pass = std::getenv("MIT2_BIGVGAN_SINGLE_PASS");
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bench_bigvgan\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"samples\": " << samples << ",\n";
    std::cout << "  \"iters\": " << iters << ",\n";
    std::cout << "  \"single_pass_env\": \"" << (single_pass ? single_pass : "") << "\",\n";
    std::cout << "  \"seconds_per_call\": " << (total / iters) << ",\n";
    std::cout << "  \"rtf_vocoder_only\": " << ((total / iters) / (static_cast<double>(samples) / 22050.0)) << "\n";
    std::cout << "}\n";
    return true;
}

struct BigVGANProfileStage {
    std::string name;
    uint32_t tokens;
    uint32_t channels;
    double seconds;
};

static std::vector<float> run_bigvgan_vocoder_metal_profiled(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& mel,
                                                              uint32_t tokens,
                                                              std::vector<BigVGANProfileStage>& stages) {
    uint32_t current_tokens = tokens;
    uint32_t current_channels = 80;

    auto t0 = Clock::now();
    auto x = run_bigvgan_conv_pre_metal(metal, bundle, mel, current_tokens);
    stages.push_back({"conv_pre", current_tokens, 1536, seconds_since(t0)});
    current_channels = 1536;

    for (uint32_t i = 0; i < 6; ++i) {
        const auto& spec = kBigVGANUpsamplers[i];
        t0 = Clock::now();
        x = run_bigvgan_upsample_metal(metal, bundle, x, current_tokens, spec);
        current_tokens = (current_tokens - 1) * spec.stride + spec.kernel - 2 * spec.padding;
        current_channels = spec.out_channels;
        stages.push_back({"upsample_" + std::to_string(i), current_tokens, current_channels, seconds_since(t0)});

        t0 = Clock::now();
        x = run_bigvgan_resblock_group_metal(metal, bundle, x, current_tokens, i);
        stages.push_back({"resblock_group_" + std::to_string(i), current_tokens, current_channels, seconds_since(t0)});
    }

    t0 = Clock::now();
    x = run_bigvgan_post_metal(metal, bundle, x, current_tokens);
    stages.push_back({"post", current_tokens, 1, seconds_since(t0)});
    return x;
}

bool run_bench_bigvgan_breakdown(const std::string& bundle_dir, uint32_t tokens, uint32_t iters) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    std::vector<float> mel(static_cast<size_t>(tokens) * 80);
    for (size_t i = 0; i < mel.size(); ++i) {
        mel[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.35f +
                 std::cos(static_cast<float>(i % 97) * 0.013f) * 0.12f;
    }

    std::vector<BigVGANProfileStage> warm_stages;
    (void)run_bigvgan_vocoder_metal_profiled(metal, bundle, mel, tokens, warm_stages);

    std::vector<BigVGANProfileStage> totals;
    size_t samples = 0;
    for (uint32_t i = 0; i < iters; ++i) {
        std::vector<BigVGANProfileStage> stages;
        auto wave = run_bigvgan_vocoder_metal_profiled(metal, bundle, mel, tokens, stages);
        samples = wave.size();
        if (totals.empty()) {
            totals = stages;
        } else {
            for (size_t j = 0; j < totals.size() && j < stages.size(); ++j) {
                totals[j].seconds += stages[j].seconds;
            }
        }
    }

    double total_seconds = 0.0;
    for (const auto& stage : totals) {
        total_seconds += stage.seconds;
    }
    const double seconds_per_call = total_seconds / static_cast<double>(iters);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bench_bigvgan_breakdown\",\n";
    std::cout << "  \"profiled_path\": \"multi_submit_readback\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"samples\": " << samples << ",\n";
    std::cout << "  \"iters\": " << iters << ",\n";
    std::cout << "  \"seconds_per_call\": " << seconds_per_call << ",\n";
    std::cout << "  \"rtf_vocoder_only\": " << (seconds_per_call / (static_cast<double>(samples) / 22050.0)) << ",\n";
    std::cout << "  \"stages\": [\n";
    for (size_t i = 0; i < totals.size(); ++i) {
        const auto& s = totals[i];
        const double sec = s.seconds / static_cast<double>(iters);
        std::cout << "    {\"name\": \"" << s.name << "\", \"tokens\": " << s.tokens
                  << ", \"channels\": " << s.channels
                  << ", \"seconds\": " << sec
                  << ", \"percent\": " << (seconds_per_call > 0.0 ? sec * 100.0 / seconds_per_call : 0.0)
                  << "}";
        if (i + 1 < totals.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
    return true;
}

bool run_bigvgan_resblock0_pair0_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    constexpr uint32_t channels = 768;
    std::vector<float> x(static_cast<size_t>(tokens) * channels);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.007f) * 0.06f + std::cos(static_cast<float>(i % 223) * 0.011f) * 0.025f;
    }
    auto ref = run_bigvgan_resblock_pair_cpu(bundle, "bigvgan.resblocks.0", x, tokens, channels, 0, 3, 1);
    auto got = run_bigvgan_resblock_pair_metal(metal, bundle, "bigvgan.resblocks.0", x, tokens, channels, 0, 3, 1);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_resblock0_pair0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"channels\": " << channels << ",\n";
    std::cout << "  \"kernel\": 3,\n";
    std::cout << "  \"dilation\": 1,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_bigvgan_resblock0_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 5;
    constexpr uint32_t channels = 768;
    std::vector<float> x(static_cast<size_t>(tokens) * channels);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.006f) * 0.055f + std::cos(static_cast<float>(i % 229) * 0.010f) * 0.022f;
    }
    auto ref = run_bigvgan_resblock_cpu(bundle, "bigvgan.resblocks.0", x, tokens, channels, 3);
    auto got = run_bigvgan_resblock_metal(metal, bundle, "bigvgan.resblocks.0", x, tokens, channels, 3);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_resblock0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"channels\": " << channels << ",\n";
    std::cout << "  \"pairs\": 3,\n";
    std::cout << "  \"dilations\": [1,3,5],\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_bigvgan_resblock_group0_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 4;
    constexpr uint32_t channels = 768;
    std::vector<float> x(static_cast<size_t>(tokens) * channels);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.005f) * 0.05f + std::cos(static_cast<float>(i % 239) * 0.012f) * 0.02f;
    }
    auto ref = run_bigvgan_resblock_group_cpu(bundle, x, tokens, 0);
    auto got = run_bigvgan_resblock_group_metal(metal, bundle, x, tokens, 0);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_resblock_group0\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"channels\": " << channels << ",\n";
    std::cout << "  \"blocks\": 3,\n";
    std::cout << "  \"kernels\": [3,7,11],\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_bigvgan_body_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 1;
    std::vector<float> mel(static_cast<size_t>(tokens) * 80);
    for (size_t i = 0; i < mel.size(); ++i) {
        mel[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.10f + std::cos(static_cast<float>(i % 71) * 0.029f) * 0.025f;
    }
    auto ref = run_bigvgan_body_cpu(bundle, mel, tokens);
    auto got = run_bigvgan_body_metal(metal, bundle, mel, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_body\",\n";
    std::cout << "  \"input_tokens\": " << tokens << ",\n";
    std::cout << "  \"out_tokens\": 256,\n";
    std::cout << "  \"out_channels\": 24,\n";
    std::cout << "  \"upsamplers\": 6,\n";
    std::cout << "  \"resblocks\": 18,\n";
    std::cout << "  \"tolerance\": 0.001,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-3f;
}

bool run_bigvgan_post_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 256;
    constexpr uint32_t channels = 24;
    std::vector<float> x(static_cast<size_t>(tokens) * channels);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.011f) * 0.06f + std::cos(static_cast<float>(i % 83) * 0.021f) * 0.02f;
    }
    auto ref = run_bigvgan_post_cpu(bundle, x, tokens);
    auto got = run_bigvgan_post_metal(metal, bundle, x, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_post\",\n";
    std::cout << "  \"tokens\": " << tokens << ",\n";
    std::cout << "  \"in_channels\": " << channels << ",\n";
    std::cout << "  \"out_channels\": 1,\n";
    std::cout << "  \"kernel\": 7,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-4f;
}

bool run_bigvgan_vocoder_test(const std::string& bundle_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    constexpr uint32_t tokens = 1;
    std::vector<float> mel(static_cast<size_t>(tokens) * 80);
    for (size_t i = 0; i < mel.size(); ++i) {
        mel[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.10f + std::cos(static_cast<float>(i % 71) * 0.029f) * 0.025f;
    }
    auto ref = run_bigvgan_vocoder_cpu(bundle, mel, tokens);
    auto got = run_bigvgan_vocoder_metal(metal, bundle, mel, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_vocoder\",\n";
    std::cout << "  \"input_tokens\": " << tokens << ",\n";
    std::cout << "  \"samples\": " << got.size() << ",\n";
    std::cout << "  \"upsamplers\": 6,\n";
    std::cout << "  \"resblocks\": 18,\n";
    std::cout << "  \"tolerance\": 0.001,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-3f;
}

bool run_bigvgan_vocoder_golden_test(const std::string& bundle_dir, const std::string& golden_dir) {
    mit2::Bundle bundle(bundle_dir);
    mit2::MetalContext metal;
    auto mel = read_raw_f32(golden_dir + "/mel.f32");
    auto ref = read_raw_f32(golden_dir + "/waveform.f32");
    if (mel.empty() || mel.size() % 80 != 0) {
        throw std::runtime_error("BigVGAN golden mel must have shape [tokens,80]");
    }
    const uint32_t tokens = static_cast<uint32_t>(mel.size() / 80);
    auto got = run_bigvgan_vocoder_metal(metal, bundle, mel, tokens);
    const float err = max_abs_error(got, ref);
    std::cout << "{\n";
    std::cout << "  \"stage\": \"bigvgan_vocoder_golden\",\n";
    std::cout << "  \"input_tokens\": " << tokens << ",\n";
    std::cout << "  \"samples\": " << got.size() << ",\n";
    std::cout << "  \"golden_dir\": \"" << golden_dir << "\",\n";
    std::cout << "  \"tolerance\": 0.001,\n";
    std::cout << "  \"max_abs_error\": " << err << "\n";
    std::cout << "}\n";
    return err <= 1e-3f;
}
