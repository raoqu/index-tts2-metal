// CAMPPlus speaker-embedding forward pass (dense blocks, transit, style forward) on CPU.
// Part of the tts_synthesis translation unit (see ../tts_synthesis.cpp);
// these files are #included in order into a single TU, so definition order
// across files is significant.

std::vector<float> run_campplus_dense_block_cpu(const mit2::Bundle& model,
                                                const std::string& block_prefix,
                                                std::vector<float> input,
                                                uint32_t in_channels,
                                                uint32_t layer_count,
                                                uint32_t width,
                                                uint32_t local_padding,
                                                uint32_t local_dilation) {
    uint32_t channels = in_channels;
    for (uint32_t i = 1; i <= layer_count; ++i) {
        const std::string prefix = block_prefix + ".tdnnd" + std::to_string(i);
        const auto tdnnd = cpu_campplus_dense_tdnn_layer(
            "cpu_campplus_" + block_prefix + "_tdnnd" + std::to_string(i),
            input,
            channels,
            width,
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.weight"),
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.bias"),
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.running_mean"),
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.running_var"),
            tensor_as_f32(model, prefix + ".linear1.weight"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.weight"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.bias"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.running_mean"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.running_var"),
            tensor_as_f32(model, prefix + ".cam_layer.linear_local.weight"),
            tensor_as_f32(model, prefix + ".cam_layer.linear1.weight"),
            tensor_as_f32(model, prefix + ".cam_layer.linear1.bias"),
            tensor_as_f32(model, prefix + ".cam_layer.linear2.weight"),
            tensor_as_f32(model, prefix + ".cam_layer.linear2.bias"),
            local_padding,
            local_dilation);
        input = concat_channels_ncw(input, channels, tdnnd, 32u, width);
        channels += 32u;
    }
    return input;
}

std::vector<float> run_campplus_transit_cpu(const mit2::Bundle& model,
                                            const std::string& prefix,
                                            std::vector<float> input,
                                            uint32_t in_channels,
                                            uint32_t out_channels,
                                            uint32_t width) {
    cpu_batchnorm1d_inplace(input,
                            in_channels,
                            width,
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.weight"),
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.bias"),
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.running_mean"),
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.running_var"),
                            true);
    uint32_t out_width = 0;
    const auto out = cpu_conv1d_ncw_no_bias(input,
                                           tensor_as_f32(model, prefix + ".linear.weight"),
                                           in_channels,
                                           width,
                                           out_channels,
                                           1u,
                                           1u,
                                           0u,
                                           1u,
                                           out_width);
    if (out_width != width) {
        throw std::runtime_error(prefix + " width mismatch");
    }
    return out;
}

std::vector<float> run_campplus_style_forward_cpu(const mit2::Bundle& model,
                                                  const std::vector<float>& fbank,
                                                  uint32_t fbank_frames) {
    if (fbank_frames == 0 || fbank.size() != static_cast<size_t>(fbank_frames) * 80u) {
        throw std::runtime_error("campplus style fbank shape mismatch");
    }
    const auto conv1 = cpu_campplus_head_conv1_bn_relu(
        fbank,
        tensor_as_f32(model, "campplus.head.conv1.weight"),
        tensor_as_f32(model, "campplus.head.bn1.weight"),
        tensor_as_f32(model, "campplus.head.bn1.bias"),
        tensor_as_f32(model, "campplus.head.bn1.running_mean"),
        tensor_as_f32(model, "campplus.head.bn1.running_var"),
        fbank_frames);
    NchwShape layer0_shape{32u, 80u, fbank_frames};
    NchwShape layer1_block0_shape;
    auto layer1 = cpu_campplus_head_layer1_block0(
        conv1,
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
        layer1_block0_shape);
    NchwShape layer1_block1_shape;
    layer1 = cpu_campplus_head_layer1_block1(
        layer1,
        layer1_block0_shape,
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
        layer1_block1_shape);

    NchwShape layer2_block0_shape;
    auto layer2 = cpu_campplus_head_layer1_block0(
        layer1,
        layer1_block1_shape,
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

    NchwShape conv2_shape;
    auto conv2 = cpu_conv2d_nchw_no_bias(
        layer2,
        layer2_block1_shape,
        tensor_as_f32(model, "campplus.head.conv2.weight"),
        32u,
        3u,
        3u,
        2u,
        1u,
        1u,
        1u,
        conv2_shape);
    cpu_batchnorm2d_inplace(conv2,
                            conv2_shape,
                            tensor_as_f32(model, "campplus.head.bn2.weight"),
                            tensor_as_f32(model, "campplus.head.bn2.bias"),
                            tensor_as_f32(model, "campplus.head.bn2.running_mean"),
                            tensor_as_f32(model, "campplus.head.bn2.running_var"),
                            true);

    uint32_t tdnn_frames = 0;
    auto tdnn = cpu_conv1d_ncw_no_bias(conv2,
                                       tensor_as_f32(model, "campplus.xvector.tdnn.linear.weight"),
                                       320u,
                                       fbank_frames,
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

    const auto block1 = run_campplus_dense_block_cpu(model, "campplus.xvector.block1", tdnn, 128u, 12u, tdnn_frames, 1u, 1u);
    const auto transit1 = run_campplus_transit_cpu(model, "campplus.xvector.transit1", block1, 512u, 256u, tdnn_frames);
    const auto block2 = run_campplus_dense_block_cpu(model, "campplus.xvector.block2", transit1, 256u, 24u, tdnn_frames, 2u, 2u);
    const auto transit2 = run_campplus_transit_cpu(model, "campplus.xvector.transit2", block2, 1024u, 512u, tdnn_frames);
    const auto block3 = run_campplus_dense_block_cpu(model, "campplus.xvector.block3", transit2, 512u, 16u, tdnn_frames, 2u, 2u);
    auto transit3 = run_campplus_transit_cpu(model, "campplus.xvector.transit3", block3, 1024u, 512u, tdnn_frames);
    cpu_batchnorm1d_inplace(transit3,
                            512u,
                            tdnn_frames,
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.weight"),
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.bias"),
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.running_mean"),
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.running_var"),
                            true);
    const auto stats = cpu_stats_pool_ncw_unbiased(transit3, 512u, tdnn_frames);
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
        throw std::runtime_error("campplus xvector dense width mismatch");
    }
    cpu_batchnorm1d_affine_false_inplace(dense,
                                         192u,
                                         1u,
                                         tensor_as_f32(model, "campplus.xvector.dense.nonlinear.batchnorm.running_mean"),
                                         tensor_as_f32(model, "campplus.xvector.dense.nonlinear.batchnorm.running_var"));
    return dense;
}

[[maybe_unused]] bool run_tts_clone_campplus_style_from_features(const std::string& model_bundle_dir,
                                                                 const std::string& feature_manifest,
                                                                 const std::string& output_s2mel_style_f32) {
    CloneFeatureManifest manifest;
    ClonePreprocessManifest preprocess_manifest;
    std::vector<std::string> issues = clone_feature_manifest_issues(feature_manifest, manifest, preprocess_manifest);

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
                               manifest.output_fbank_f32,
                               manifest.fbank_frames * 80u,
                               "campplus_fbank");
    if (!issues.empty()) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_campplus_style_from_features\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"output_s2mel_style_f32\": \"" << json_escape(output_s2mel_style_f32) << "\",\n";
        std::cout << "  \"output_fbank_f32\": \"" << json_escape(manifest.output_fbank_f32) << "\",\n";
        std::cout << "  \"fbank_frames\": " << manifest.fbank_frames << ",\n";
        std::cout << "  \"fbank_values\": " << (manifest.fbank_frames * 80u) << ",\n";
        std::cout << "  \"has_campplus_model_contract\": " << (campplus_ok ? "true" : "false") << ",\n";
        std::cout << "  \"campplus_required_tensor_count\": " << campplus_required << ",\n";
        std::cout << "  \"campplus_required_tensors_present\": " << campplus_present << ",\n";
        std::cout << "  \"campplus_contract_issues\": ";
        print_json_string_array(campplus_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_campplus_style_from_features_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"ready_native_campplus_style_forward\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }

    try {
        mit2::Bundle model(model_bundle_dir);
        const auto fbank = read_raw_f32(manifest.output_fbank_f32);
        const auto style = run_campplus_style_forward_cpu(model, fbank, static_cast<uint32_t>(manifest.fbank_frames));
        write_raw_f32(output_s2mel_style_f32, style);
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_campplus_style_from_features\",\n";
        std::cout << "  \"ok\": true,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"output_fbank_f32\": \"" << json_escape(manifest.output_fbank_f32) << "\",\n";
        std::cout << "  \"output_s2mel_style_f32\": \"" << json_escape(output_s2mel_style_f32) << "\",\n";
        std::cout << "  \"output_s2mel_style_sha256\": \"" << file_sha256_hex(output_s2mel_style_f32) << "\",\n";
        std::cout << "  \"execution_backend\": \"cpu\",\n";
        std::cout << "  \"fbank_frames\": " << manifest.fbank_frames << ",\n";
        std::cout << "  \"fbank_values\": " << fbank.size() << ",\n";
        std::cout << "  \"s2mel_style_shape\": \"[1,192]\",\n";
        std::cout << "  \"s2mel_style_values\": " << style.size() << ",\n";
        std::cout << "  \"has_campplus_model_contract\": true,\n";
        std::cout << "  \"campplus_required_tensor_count\": " << campplus_required << ",\n";
        std::cout << "  \"campplus_required_tensors_present\": " << campplus_present << ",\n";
        std::cout << "  \"campplus_contract_issues\": ";
        print_json_string_array(campplus_issues);
        std::cout << ",\n";
        std::cout << "  \"clone_campplus_style_from_features_issues\": [],\n";
        std::cout << "  \"ready_native_clone_feature_prep\": " << (manifest.ready_native_clone_feature_prep ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_clone_fbank_extraction\": " << (manifest.ready_native_clone_fbank_extraction ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_campplus_style_forward\": true,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_campplus_style_from_features\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"feature_manifest\": \"" << json_escape(feature_manifest) << "\",\n";
        std::cout << "  \"output_s2mel_style_f32\": \"" << json_escape(output_s2mel_style_f32) << "\",\n";
        std::cout << "  \"clone_campplus_style_from_features_issues\": [\"" << json_escape(e.what()) << "\"],\n";
        std::cout << "  \"ready_native_campplus_style_forward\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return false;
    }
}

