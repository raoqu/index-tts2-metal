bool run_export_text_ids_cjk(const std::string& tokenizer_dir,
                             const std::string& text,
                             const std::string& output_text_ids_path,
                             const std::string& format) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    const auto tokenized = tokenize_cjk_text(piece_to_id, text);
    write_raw_u32(output_text_ids_path, tokenized.ids);

    std::cout << "{\n";
    std::cout << "  \"format\": \"" << json_escape(format) << "\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"output\": \"" << json_escape(output_text_ids_path) << "\",\n";
    std::cout << "  \"pieces\": ";
    print_json_string_array(tokenized.pieces);
    std::cout << ",\n";
    std::cout << "  \"ids\": ";
    print_json_u32_array(tokenized.ids);
    std::cout << "\n";
    std::cout << "}\n";
    return true;
}

bool run_text_ids_cjk_version_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<Case> cases{
        {"V2\xe4\xbd\xa0\xe5\xa5\xbd", {space_piece, "V", "2", space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"}},
        {"v2\xe4\xbd\xa0\xe5\xa5\xbd", {space_piece, "V", "2", space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"}},
        {"V3\xe4\xbd\xa0\xe5\xa5\xbd", {space_piece, "V", "3", space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"}},
        {"V20\xe4\xbd\xa0\xe5\xa5\xbd", {space_piece, "V", "2", space_piece, "\xe9\x9b\xb6", space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"}},
        {"X2\xe4\xbd\xa0\xe5\xa5\xbd", {space_piece, "X", space_piece, "\xe4\xba\x8c", space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"}},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_version_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_slash_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    auto with_slash = [&](std::vector<std::string> lhs, std::vector<std::string> rhs) {
        lhs.push_back(space_piece);
        lhs.push_back("/");
        lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        return lhs;
    };
    auto with_two_slashes = [&](std::vector<std::string> first,
                                std::vector<std::string> second,
                                std::vector<std::string> third) {
        auto out = with_slash(std::move(first), std::move(second));
        out.push_back(space_piece);
        out.push_back("/");
        out.insert(out.end(), third.begin(), third.end());
        return out;
    };
    const std::vector<Case> cases{
        {"A/B\xe6\xb5\x8b\xe8\xaf\x95", {space_piece + "A", "/", "B", space_piece, "\xe6\xb5\x8b", space_piece, "\xe8\xaf\x95"}},
        {"\xe4\xbd\xa0\xe5\xa5\xbd/\xe4\xb8\x96\xe7\x95\x8c", {space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd", space_piece, "/", space_piece, "\xe4\xb8\x96", space_piece, "\xe7\x95\x8c"}},
        {"\xe8\xb7\xaf\xe5\xbe\x84/home", {space_piece, "\xe8\xb7\xaf", space_piece, "\xe5\xbe\x84", space_piece, "/", "HO", "ME"}},
        {"\xef\xbc\x91\xef\xbc\x8f\xef\xbc\x92",
         with_slash(tokenize_digit_run_as_cjk_integer("1", false), tokenize_digit_run_as_cjk_integer("2", false))},
        {"\xe6\xaf\x94\xe4\xbe\x8b\xef\xbc\x91\xef\xbc\x8f\xef\xbc\x92",
         [&] {
             std::vector<std::string> out{space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xbe\x8b"};
             auto suffix = with_slash(tokenize_digit_run_as_cjk_integer("1", false), tokenize_digit_run_as_cjk_integer("2", false));
             out.insert(out.end(), suffix.begin(), suffix.end());
             return out;
         }()},
        {"\xef\xbc\x91\xef\xbc\x8f\xef\xbc\x90\xef\xbc\x92",
         with_slash(tokenize_digit_run_as_cjk_integer("1", false), tokenize_digit_run_as_cjk_integer("02", false))},
        {"\xef\xbc\x91\xef\xbc\x8f\xef\xbc\x92\xef\xbc\x8f\xef\xbc\x93",
         with_two_slashes(
             tokenize_digit_run_as_cjk_integer("1", false),
             tokenize_digit_run_as_cjk_integer("2", false),
             tokenize_digit_run_as_cjk_integer("3", false))},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_slash_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"fullwidth_slash_unknown_id_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_temperature_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<std::string> twenty_five{
        space_piece, "\xe4\xba\x8c", space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x94"
    };
    const std::vector<std::string> two{
        space_piece, "\xe4\xb8\xa4"
    };
    const std::vector<std::string> zero_two{
        space_piece, "\xe9\x9b\xb6", space_piece, "\xe4\xb8\xa4"
    };
    const std::vector<std::string> negative_five{
        space_piece, "\xe8\xb4\x9f", space_piece, "\xe4\xba\x94"
    };
    const std::vector<std::string> celsius{
        space_piece, "\xe6\x91\x84", space_piece, "\xe6\xb0\x8f", space_piece, "\xe5\xba\xa6"
    };
    const std::vector<std::string> fahrenheit{
        space_piece, "\xe5\x8d\x8e", space_piece, "\xe6\xb0\x8f", space_piece, "\xe5\xba\xa6"
    };
    auto english_temperature = [&](const std::string& digits, bool celsius_unit) {
        auto out = tokenize_english_number_run(digits);
        append_english_temperature_unit_pieces(out, piece_to_id, celsius_unit);
        return out;
    };
    auto with_prefix = [&](std::vector<std::string> suffix) {
        std::vector<std::string> out{space_piece, "\xe6\xb8\xa9", space_piece, "\xe5\xba\xa6"};
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto concat = [](std::vector<std::string> a, const std::vector<std::string>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    const std::vector<Case> cases{
        {"\xe6\xb8\xa9\xe5\xba\xa6" "25\xc2\xb0" "C", with_prefix(concat(twenty_five, celsius))},
        {"\xe6\xb8\xa9\xe5\xba\xa6" "25\xe2\x84\x83", with_prefix(concat(twenty_five, celsius))},
        {"\xe6\xb8\xa9\xe5\xba\xa6" "25\xc2\xb0" "F", with_prefix(concat(twenty_five, fahrenheit))},
        {"25\xe2\x84\x83", concat(twenty_five, celsius)},
        {"25\xe2\x84\x89", concat(twenty_five, fahrenheit)},
        {"2\xe2\x84\x83", concat(two, celsius)},
        {"02\xe2\x84\x83", concat(zero_two, celsius)},
        {"\xe6\xb8\xa9\xe5\xba\xa6" "2\xc2\xb0" "C", with_prefix(concat(two, celsius))},
        {"25\xc2\xb0" "C", english_temperature("25", true)},
        {"25\xc2\xb0" "F", english_temperature("25", false)},
        {"25\xc2\xb0" "c", english_temperature("25", true)},
        {"25\xc2\xb0" "f", english_temperature("25", false)},
        {"-5", negative_five},
        {"-5\xe2\x84\x83", concat(negative_five, celsius)},
        {"\xe6\xb8\xa9\xe5\xba\xa6" "-5\xc2\xb0" "C", with_prefix(concat(negative_five, celsius))},
        {"\xe6\xb8\xa9\xe5\xba\xa6" "-5\xc2\xb0" "F", with_prefix(concat(negative_five, fahrenheit))},
        {"-5\xe5\xba\xa6", concat(negative_five, {space_piece, "\xe5\xba\xa6"})},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_temperature_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"ascii_degree_temperature_supported\": true,\n";
    std::cout << "  \"ascii_unary_minus_cjk_temperature_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_plus_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    auto with_prefix = [&](const std::vector<std::string>& prefix, std::vector<std::string> suffix) {
        std::vector<std::string> out = prefix;
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto plus_binary = [&](const std::string& lhs, const std::string& rhs) {
        auto out = tokenize_digit_run_as_cjk_integer(lhs, false);
        append_cjk_piece(out, "\xe5\x8a\xa0");
        auto rhs_pieces = tokenize_digit_run_as_cjk_integer(rhs, false);
        out.insert(out.end(), rhs_pieces.begin(), rhs_pieces.end());
        return out;
    };
    auto positive_number = [&](const std::string& digits) {
        std::vector<std::string> out;
        append_cjk_piece(out, "\xe6\xad\xa3");
        auto digit_pieces = tokenize_digit_run_as_cjk_integer(digits, false);
        out.insert(out.end(), digit_pieces.begin(), digit_pieces.end());
        return out;
    };
    auto positive_temperature = [&](const std::string& digits, bool celsius) {
        auto out = positive_number(digits);
        append_temperature_unit_pieces(out, celsius);
        return out;
    };
    auto positive_degree_word = [&](const std::string& digits) {
        auto out = positive_number(digits);
        append_cjk_piece(out, "\xe5\xba\xa6");
        return out;
    };
    auto plus_positive_binary = [&](const std::string& lhs, const std::string& rhs) {
        auto out = tokenize_digit_run_as_cjk_integer(lhs, false);
        append_cjk_piece(out, "\xe5\x8a\xa0");
        auto rhs_pieces = positive_number(rhs);
        out.insert(out.end(), rhs_pieces.begin(), rhs_pieces.end());
        return out;
    };
    const std::vector<std::string> wendu{space_piece, "\xe6\xb8\xa9", space_piece, "\xe5\xba\xa6"};
    const std::vector<std::string> wancheng{space_piece, "\xe5\xae\x8c", space_piece, "\xe6\x88\x90"};
    const std::vector<Case> cases{
        {"1+2", plus_binary("1", "2")},
        {"1++2", plus_positive_binary("1", "2")},
        {"12+34", plus_binary("12", "34")},
        {"\xe5\xae\x8c\xe6\x88\x90" "1+2", with_prefix(wancheng, plus_binary("1", "2"))},
        {"+5", positive_number("5")},
        {"+2\xe2\x84\x83", positive_temperature("2", true)},
        {"+5\xe2\x84\x83", positive_temperature("5", true)},
        {"\xe6\xb8\xa9\xe5\xba\xa6+5\xe2\x84\x83", with_prefix(wendu, positive_temperature("5", true))},
        {"\xe6\xb8\xa9\xe5\xba\xa6+2\xc2\xb0" "C", with_prefix(wendu, positive_temperature("2", true))},
        {"\xe6\xb8\xa9\xe5\xba\xa6+5\xc2\xb0" "C", with_prefix(wendu, positive_temperature("5", true))},
        {"\xe6\xb8\xa9\xe5\xba\xa6+5\xc2\xb0" "F", with_prefix(wendu, positive_temperature("5", false))},
        {"1+2\xe5\xba\xa6",
         [&] {
             auto out = tokenize_digit_run_as_cjk_integer("1", false);
             auto rhs = positive_degree_word("2");
             out.insert(out.end(), rhs.begin(), rhs.end());
             return out;
         }()},
        {"1+2\xe2\x84\x83",
         [&] {
             auto out = tokenize_digit_run_as_cjk_integer("1", false);
             append_cjk_piece(out, "\xe6\xad\xa3");
             auto rhs = tokenize_digit_run_as_cjk_integer("2", false);
             out.insert(out.end(), rhs.begin(), rhs.end());
             append_temperature_unit_pieces(out, true);
             return out;
         }()},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_plus_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }

    const std::vector<std::string> unsupported{"+5\xc2\xb0" "C", "1+2\xc2\xb0" "C", "A+1", "C++"};
    std::vector<std::string> rejected;
    for (const auto& text : unsupported) {
        try {
            (void)tokenize_cjk_text(piece_to_id, text);
        } catch (const std::runtime_error&) {
            rejected.push_back(text);
        }
    }
    const bool unsupported_rejected = rejected.size() == unsupported.size();
    ok = ok && unsupported_rejected;
    std::cout << "  ],\n";
    std::cout << "  \"ascii_plus_binary_supported\": true,\n";
    std::cout << "  \"ascii_plus_unary_cjk_supported\": true,\n";
    std::cout << "  \"english_plus_context_requires_full_text_normalizer\": " << (unsupported_rejected ? "true" : "false") << ",\n";
    std::cout << "  \"rejected\": ";
    print_json_string_array(rejected);
    std::cout << ",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_operator_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    auto with_prefix = [&](const std::vector<std::string>& prefix, std::vector<std::string> suffix) {
        std::vector<std::string> out = prefix;
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto binary_operator = [&](const std::string& lhs,
                               const std::string& rhs,
                               const std::string& op) {
        auto out = tokenize_digit_run_as_cjk_integer(lhs, false);
        if (op == "=") {
            append_cjk_equal_pieces(out);
        } else if (op == "<") {
            append_cjk_less_than_pieces(out);
        } else if (op == ">") {
            append_cjk_greater_than_pieces(out);
        } else if (op == "<=") {
            append_cjk_less_than_pieces(out);
            append_cjk_equal_pieces(out);
        } else if (op == ">=") {
            append_cjk_greater_than_pieces(out);
            append_cjk_equal_pieces(out);
        } else if (op == "*") {
            append_cjk_piece(out, "\xe4\xb9\x98");
        } else {
            throw std::runtime_error("unexpected operator test helper op");
        }
        auto rhs_pieces = tokenize_digit_run_as_cjk_integer(rhs, false);
        out.insert(out.end(), rhs_pieces.begin(), rhs_pieces.end());
        return out;
    };
    auto binary_operator_degree = [&](const std::string& lhs,
                                      const std::string& rhs,
                                      const std::string& op,
                                      const std::string& degree_kind) {
        auto out = tokenize_digit_run_as_cjk_integer(lhs, false);
        if (op == "=") {
            append_cjk_equal_pieces(out);
        } else if (op == "<=") {
            append_cjk_less_than_pieces(out);
            append_cjk_equal_pieces(out);
        } else if (op == ">=") {
            append_cjk_greater_than_pieces(out);
            append_cjk_equal_pieces(out);
        } else if (op == "*") {
            append_cjk_piece(out, "\xe4\xb9\x98");
        } else {
            throw std::runtime_error("unexpected operator degree test helper op");
        }
        auto rhs_pieces = tokenize_operator_rhs_degree_number_as_cjk(rhs, false);
        out.insert(out.end(), rhs_pieces.begin(), rhs_pieces.end());
        if (degree_kind == "C") {
            append_temperature_unit_pieces(out, true);
        } else if (degree_kind == "F") {
            append_temperature_unit_pieces(out, false);
        } else if (degree_kind == "degree") {
            append_cjk_piece(out, "\xe5\xba\xa6");
        } else {
            throw std::runtime_error("unexpected operator degree test helper unit");
        }
        return out;
    };
    const std::vector<std::string> wancheng{space_piece, "\xe5\xae\x8c", space_piece, "\xe6\x88\x90"};
    const std::vector<std::string> wendu{space_piece, "\xe6\xb8\xa9", space_piece, "\xe5\xba\xa6"};
    const std::vector<Case> cases{
        {"1=1", binary_operator("1", "1", "=")},
        {"12=34", binary_operator("12", "34", "=")},
        {"\xef\xbc\x91=\xef\xbc\x92", binary_operator("1", "2", "=")},
        {"\xe5\xae\x8c\xe6\x88\x90" "1=1", with_prefix(wancheng, binary_operator("1", "1", "="))},
        {"1<2", binary_operator("1", "2", "<")},
        {"1>2", binary_operator("1", "2", ">")},
        {"12<34", binary_operator("12", "34", "<")},
        {"1<=2", binary_operator("1", "2", "<=")},
        {"1>=2", binary_operator("1", "2", ">=")},
        {"1\xe2\x89\xa4" "2", binary_operator("1", "2", "<=")},
        {"1\xe2\x89\xa5" "2", binary_operator("1", "2", ">=")},
        {"3\xc3\x97" "4", binary_operator("3", "4", "*")},
        {"\xef\xbc\x93\xc3\x97\xef\xbc\x94", binary_operator("3", "4", "*")},
        {"\xe5\xae\x8c\xe6\x88\x90" "3\xc3\x97" "4", with_prefix(wancheng, binary_operator("3", "4", "*"))},
        {"1=2\xe2\x84\x83", binary_operator_degree("1", "2", "=", "C")},
        {"1=2\xe2\x84\x89", binary_operator_degree("1", "2", "=", "F")},
        {"1=2\xe5\xba\xa6", binary_operator_degree("1", "2", "=", "degree")},
        {"1=25\xe2\x84\x83", binary_operator_degree("1", "25", "=", "C")},
        {"1=102\xe2\x84\x83", binary_operator_degree("1", "102", "=", "C")},
        {"1=112\xe2\x84\x83", binary_operator_degree("1", "112", "=", "C")},
        {"1=120\xe2\x84\x83", binary_operator_degree("1", "120", "=", "C")},
        {"1=302\xe2\x84\x83", binary_operator_degree("1", "302", "=", "C")},
        {"1=1000\xe2\x84\x83", binary_operator_degree("1", "1000", "=", "C")},
        {"\xe6\xb8\xa9\xe5\xba\xa6" "1=2\xe2\x84\x83", with_prefix(wendu, binary_operator_degree("1", "2", "=", "C"))},
        {"1<=2\xe2\x84\x83", binary_operator_degree("1", "2", "<=", "C")},
        {"1>=2\xe5\xba\xa6", binary_operator_degree("1", "2", ">=", "degree")},
        {"1\xe2\x89\xa4" "2\xe2\x84\x83", binary_operator_degree("1", "2", "<=", "C")},
        {"1\xe2\x89\xa5" "2\xe2\x84\x83", binary_operator_degree("1", "2", ">=", "C")},
        {"3\xc3\x97" "2\xe2\x84\x83", binary_operator_degree("3", "2", "*", "C")},
        {"3\xc3\x97" "25\xe2\x84\x83", binary_operator_degree("3", "25", "*", "C")},
        {"3\xc3\x97" "125\xe2\x84\x83", binary_operator_degree("3", "125", "*", "C")},
        {"3\xc3\x97" "1000\xe2\x84\x83", binary_operator_degree("3", "1000", "*", "C")},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_operator_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }

    const std::vector<std::string> unsupported{
        "1\xef\xbc\x9d" "1",
        "1\xef\xbc\x9c" "2",
        "1\xef\xbc\x9e" "2",
        "3*4",
        "3\xef\xbc\x8a" "4",
        "A=1",
        "hello=world",
        "1<2\xe2\x84\x83",
        "1=2\xc2\xb0" "C",
        "1=2026\xe2\x84\x83",
    };
    std::vector<std::string> rejected;
    for (const auto& text : unsupported) {
        try {
            (void)tokenize_cjk_text(piece_to_id, text);
        } catch (const std::runtime_error&) {
            rejected.push_back(text);
        }
    }
    const bool unsupported_rejected = rejected.size() == unsupported.size();
    ok = ok && unsupported_rejected;
    std::cout << "  ],\n";
    std::cout << "  \"ascii_comparison_operators_supported\": true,\n";
    std::cout << "  \"unicode_comparison_operators_supported\": true,\n";
    std::cout << "  \"unicode_multiply_operator_supported\": true,\n";
    std::cout << "  \"operator_degree_rhs_supported\": true,\n";
    std::cout << "  \"operator_degree_rhs_three_digit_supported\": true,\n";
    std::cout << "  \"operator_english_degree_rhs_requires_full_text_normalizer\": " << (unsupported_rejected ? "true" : "false") << ",\n";
    std::cout << "  \"rejected\": ";
    print_json_string_array(rejected);
    std::cout << ",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_measure_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<std::string> mi{space_piece, "\xe7\xb1\xb3"};
    const std::vector<std::string> gongli{space_piece, "\xe5\x85\xac", space_piece, "\xe9\x87\x8c"};
    const std::vector<std::string> yuan{space_piece, "\xe5\x85\x83"};
    const std::vector<std::string> du{space_piece, "\xe5\xba\xa6"};
    auto with_suffix = [](std::vector<std::string> number, const std::vector<std::string>& suffix) {
        number.insert(number.end(), suffix.begin(), suffix.end());
        return number;
    };
    auto measure = [&](const std::string& digits, const std::vector<std::string>& suffix) {
        return with_suffix(tokenize_measure_number_as_cjk(digits, false), suffix);
    };
    auto decimal_measure = [&](const std::string& integer,
                               const std::string& fraction,
                               const std::vector<std::string>& suffix) {
        auto out = tokenize_digit_run_as_cjk_integer(integer, false);
        append_cjk_piece(out, "\xe7\x82\xb9");
        auto fraction_pieces = tokenize_digit_run_as_cjk(fraction, true);
        out.insert(out.end(), fraction_pieces.begin(), fraction_pieces.end());
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto ascii_unit = [&](const std::string& digits, const std::string& suffix_text, const std::string& punctuation = "") {
        auto out = tokenize_english_unit_number_run(digits);
        std::vector<std::string> suffix;
        size_t suffix_end = 0;
        if (!read_ascii_unit_suffix_pieces(piece_to_id,
                                           suffix_text,
                                           0,
                                           english_unit_number_is_singular(digits),
                                           suffix,
                                           suffix_end)) {
            throw std::runtime_error("unexpected ASCII unit suffix test helper suffix");
        }
        out.insert(out.end(), suffix.begin(), suffix.end());
        if (!punctuation.empty()) {
            out.push_back(space_piece + punctuation);
        }
        return out;
    };
    auto ascii_decimal_unit = [&](const std::string& digits,
                                  const std::string& fraction,
                                  const std::string& suffix_text,
                                  const std::string& punctuation = "") {
        auto out = tokenize_english_unit_decimal_number_run(digits, fraction);
        std::vector<std::string> suffix;
        size_t suffix_end = 0;
        if (!read_ascii_unit_suffix_pieces(piece_to_id,
                                           suffix_text,
                                           0,
                                           false,
                                           suffix,
                                           suffix_end)) {
            throw std::runtime_error("unexpected ASCII decimal unit suffix test helper suffix");
        }
        out.insert(out.end(), suffix.begin(), suffix.end());
        if (!punctuation.empty()) {
            out.push_back(space_piece + punctuation);
        }
        return out;
    };
    auto fullwidth_ascii_unit = [&](const std::string& digits, const std::string& suffix_text) {
        auto out = tokenize_digit_run_as_cjk_integer(digits, false);
        std::vector<std::string> suffix;
        size_t suffix_end = 0;
        if (!read_fullwidth_ascii_unit_suffix_pieces(piece_to_id, suffix_text, 0, suffix, suffix_end)) {
            throw std::runtime_error("unexpected fullwidth ASCII unit suffix test helper suffix");
        }
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto fullwidth_ascii_decimal_unit = [&](const std::string& digits,
                                            const std::string& fraction,
                                            const std::string& suffix_text) {
        auto out = tokenize_digit_run_as_cjk_integer(digits, false);
        append_cjk_piece(out, "\xe7\x82\xb9");
        auto fraction_pieces = tokenize_digit_run_as_cjk(fraction, true);
        out.insert(out.end(), fraction_pieces.begin(), fraction_pieces.end());
        std::vector<std::string> suffix;
        size_t suffix_end = 0;
        if (!read_fullwidth_ascii_unit_suffix_pieces(piece_to_id, suffix_text, 0, suffix, suffix_end)) {
            throw std::runtime_error("unexpected fullwidth ASCII decimal unit suffix test helper suffix");
        }
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto operator_measure = [&](const std::string& lhs,
                                const std::string& rhs,
                                const std::string& op,
                                const std::vector<std::string>& suffix) {
        auto out = tokenize_digit_run_as_cjk_integer(lhs, false);
        if (op == "=") {
            append_cjk_equal_pieces(out);
        } else if (op == "<") {
            append_cjk_less_than_pieces(out);
        } else if (op == ">") {
            append_cjk_greater_than_pieces(out);
        } else if (op == "<=") {
            append_cjk_less_than_pieces(out);
            append_cjk_equal_pieces(out);
        } else if (op == ">=") {
            append_cjk_greater_than_pieces(out);
            append_cjk_equal_pieces(out);
        } else if (op == "*") {
            append_cjk_piece(out, "\xe4\xb9\x98");
        } else {
            throw std::runtime_error("unexpected measure operator test helper op");
        }
        auto rhs_pieces = tokenize_operator_rhs_measure_number_as_cjk(rhs, false);
        out.insert(out.end(), rhs_pieces.begin(), rhs_pieces.end());
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto binary_generic_measure = [&](const std::string& lhs,
                                      const std::string& rhs,
                                      const std::string& op,
                                      const std::vector<std::string>& suffix) {
        auto out = tokenize_digit_run_as_cjk_integer(lhs, false);
        if (op == "<") {
            append_cjk_less_than_pieces(out);
        } else if (op == ">") {
            append_cjk_greater_than_pieces(out);
        } else {
            throw std::runtime_error("unexpected generic measure operator test helper op");
        }
        auto rhs_pieces = tokenize_digit_run_as_cjk_integer(rhs, false);
        out.insert(out.end(), rhs_pieces.begin(), rhs_pieces.end());
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    const std::vector<std::string> wendu{space_piece, "\xe6\xb8\xa9", space_piece, "\xe5\xba\xa6"};
    auto with_prefix = [&](const std::vector<std::string>& prefix, std::vector<std::string> suffix) {
        std::vector<std::string> out = prefix;
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    const std::vector<Case> cases{
        {"2\xe7\xb1\xb3", measure("2", mi)},
        {"02\xe7\xb1\xb3", measure("02", mi)},
        {"12\xe7\xb1\xb3", measure("12", mi)},
        {"22\xe7\xb1\xb3", measure("22", mi)},
        {"200\xe7\xb1\xb3", measure("200", mi)},
        {"202\xe7\xb1\xb3", measure("202", mi)},
        {"2\xe5\x85\xac\xe9\x87\x8c", measure("2", gongli)},
        {"2\xe5\x85\x83", measure("2", yuan)},
        {"2\xe5\xba\xa6", measure("2", du)},
        {"25\xe5\xba\xa6", measure("25", du)},
        {"2.5\xe7\xb1\xb3", decimal_measure("2", "5", mi)},
        {"2.05\xe7\xb1\xb3", decimal_measure("2", "05", mi)},
        {"1.0\xe7\xb1\xb3", decimal_measure("1", "0", mi)},
        {"2.5\xe5\x85\xac\xe9\x87\x8c", decimal_measure("2", "5", gongli)},
        {"2.5\xe5\xba\xa6", decimal_measure("2", "5", du)},
        {"\xe6\xb8\xa9\xe5\xba\xa6" "2\xe5\xba\xa6", with_prefix(wendu, measure("2", du))},
        {"2\xe7\xbe\x8e\xe5\x85\x83",
         [] {
             auto out = tokenize_digit_run_as_cjk_integer("2", false);
             append_cjk_piece(out, "\xe7\xbe\x8e");
             append_cjk_piece(out, "\xe5\x85\x83");
             return out;
         }()},
        {"1=2\xe7\xb1\xb3", operator_measure("1", "2", "=", mi)},
        {"1=25\xe7\xb1\xb3", operator_measure("1", "25", "=", mi)},
        {"1=102\xe7\xb1\xb3", operator_measure("1", "102", "=", mi)},
        {"1=101\xe7\xb1\xb3", operator_measure("1", "101", "=", mi)},
        {"1=112\xe7\xb1\xb3", operator_measure("1", "112", "=", mi)},
        {"1=120\xe7\xb1\xb3", operator_measure("1", "120", "=", mi)},
        {"1=1000\xe7\xb1\xb3", operator_measure("1", "1000", "=", mi)},
        {"1<=2\xe7\xb1\xb3", operator_measure("1", "2", "<=", mi)},
        {"1\xe2\x89\xa5" "2\xe7\xb1\xb3", operator_measure("1", "2", ">=", mi)},
        {"3\xc3\x97" "2\xe7\xb1\xb3", operator_measure("3", "2", "*", mi)},
        {"3\xc3\x97" "25\xe7\xb1\xb3", operator_measure("3", "25", "*", mi)},
        {"3\xc3\x97" "202\xe7\xb1\xb3", operator_measure("3", "202", "*", mi)},
        {"3\xc3\x97" "125\xe7\xb1\xb3", operator_measure("3", "125", "*", mi)},
        {"3\xc3\x97" "1000\xe7\xb1\xb3", operator_measure("3", "1000", "*", mi)},
        {"1<2\xe7\xb1\xb3", binary_generic_measure("1", "2", "<", mi)},
        {"1>2\xe7\xb1\xb3", binary_generic_measure("1", "2", ">", mi)},
        {"0kg", ascii_unit("0", "kg")},
        {"1kg", ascii_unit("1", "kg")},
        {"2kg", ascii_unit("2", "kg")},
        {"02kg", ascii_unit("02", "kg")},
        {"200kg", ascii_unit("200", "kg")},
        {"202kg", ascii_unit("202", "kg")},
        {"1000kg", ascii_unit("1000", "kg")},
        {"10000kg", ascii_unit("10000", "kg")},
        {"10001kg", ascii_unit("10001", "kg")},
        {"2KG", ascii_unit("2", "KG")},
        {"2kgs", ascii_unit("2", "kgs")},
        {"2\xef\xbc\xab\xef\xbc\xa7", fullwidth_ascii_unit("2", "\xef\xbc\xab\xef\xbc\xa7")},
        {"2g", ascii_unit("2", "g")},
        {"12g", ascii_unit("12", "g")},
        {"2m", ascii_unit("2", "m")},
        {"2cm", ascii_unit("2", "cm")},
        {"1cm", ascii_unit("1", "cm")},
        {"2mm", ascii_unit("2", "mm")},
        {"2km", ascii_unit("2", "km")},
        {"2ml", ascii_unit("2", "ml")},
        {"1ml", ascii_unit("1", "ml")},
        {"2L", ascii_unit("2", "L")},
        {"2kg.", ascii_unit("2", "kg.", ".")},
        {"2.5kg", ascii_decimal_unit("2", "5", "kg")},
        {"0.5kg", ascii_decimal_unit("0", "5", "kg")},
        {"1.0kg", ascii_decimal_unit("1", "0", "kg")},
        {"1.00kg", ascii_decimal_unit("1", "00", "kg")},
        {"02.5kg", ascii_decimal_unit("02", "5", "kg")},
        {"200.5kg", ascii_decimal_unit("200", "5", "kg")},
        {"202.5kg", ascii_decimal_unit("202", "5", "kg")},
        {"1000.5kg", ascii_decimal_unit("1000", "5", "kg")},
        {"10000.5kg", ascii_decimal_unit("10000", "5", "kg")},
        {"10001.5kg", ascii_decimal_unit("10001", "5", "kg")},
        {"2.05kg", ascii_decimal_unit("2", "05", "kg")},
        {"2.50kg", ascii_decimal_unit("2", "50", "kg")},
        {"2.5kgs", ascii_decimal_unit("2", "5", "kgs")},
        {"2.5\xef\xbc\xab\xef\xbc\xa7", fullwidth_ascii_decimal_unit("2", "5", "\xef\xbc\xab\xef\xbc\xa7")},
        {"2.5g", ascii_decimal_unit("2", "5", "g")},
        {"2.5m", ascii_decimal_unit("2", "5", "m")},
        {"2.5cm", ascii_decimal_unit("2", "5", "cm")},
        {"1.0cm", ascii_decimal_unit("1", "0", "cm")},
        {"2.5ml", ascii_decimal_unit("2", "5", "ml")},
        {"1.0ml", ascii_decimal_unit("1", "0", "ml")},
        {"2.5L", ascii_decimal_unit("2", "5", "L")},
        {"2.5kg.", ascii_decimal_unit("2", "5", "kg.", ".")},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_measure_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }

    const std::vector<std::string> unsupported{
        "1=1001\xe7\xb1\xb3",
        "1=2026\xe7\xb1\xb3",
        "3\xc3\x97" "2026\xe7\xb1\xb3",
        "01000kg",
        "\xe9\x87\x8d\xe9\x87\x8f" "2kg",
        "2kg\xe5\xae\x8c\xe6\x88\x90",
        "\xe9\x87\x8d\xe9\x87\x8f" "2.5kg",
        "2.5kg\xe5\xae\x8c\xe6\x88\x90",
        "\xef\xbc\x92" "kg",
        "\xef\xbc\x92.5kg",
        "2.\xef\xbc\x95" "kg",
    };
    std::vector<std::string> rejected;
    for (const auto& text : unsupported) {
        try {
            (void)tokenize_cjk_text(piece_to_id, text);
        } catch (const std::runtime_error&) {
            rejected.push_back(text);
        }
    }
    const bool unsupported_rejected = rejected.size() == unsupported.size();
    ok = ok && unsupported_rejected;
    std::cout << "  ],\n";
    std::cout << "  \"cjk_measure_two_reading_supported\": true,\n";
    std::cout << "  \"operator_measure_rhs_supported\": true,\n";
    std::cout << "  \"operator_measure_rhs_three_digit_supported\": true,\n";
    std::cout << "  \"ascii_unit_suffix_supported\": true,\n";
    std::cout << "  \"operator_measure_long_rhs_requires_full_text_normalizer\": " << (unsupported_rejected ? "true" : "false") << ",\n";
    std::cout << "  \"rejected\": ";
    print_json_string_array(rejected);
    std::cout << ",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_date_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<std::string> date_pieces{
        space_piece, "\xe4\xba\x8c", space_piece, "\xe9\x9b\xb6", space_piece, "\xe4\xba\x8c", space_piece, "\xe5\x85\xad",
        space_piece, "\xe5\xb9\xb4", space_piece, "\xe5\x85\xad", space_piece, "\xe6\x9c\x88", space_piece, "\xe4\xba\x94",
        space_piece, "\xe6\x97\xa5"
    };
    const std::vector<std::string> date_cjk_prefix{
        space_piece, "\xe6\x97\xa5", space_piece, "\xe6\x9c\x9f"
    };
    auto with_prefix = [&](std::vector<std::string> suffix) {
        std::vector<std::string> out = date_cjk_prefix;
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto with_punctuation = [&](std::vector<std::string> lhs,
                                const std::string& punctuation,
                                std::vector<std::string> rhs) {
        append_normalized_ascii_punctuation(lhs, piece_to_id, punctuation, false);
        lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        return lhs;
    };
    auto with_two_punctuations = [&](std::vector<std::string> first,
                                     const std::string& punctuation,
                                     std::vector<std::string> second,
                                     std::vector<std::string> third) {
        auto out = with_punctuation(std::move(first), punctuation, std::move(second));
        append_normalized_ascii_punctuation(out, piece_to_id, punctuation, false);
        out.insert(out.end(), third.begin(), third.end());
        return out;
    };
    auto with_year_marker = [&](std::vector<std::string> prefix) {
        prefix.push_back(space_piece);
        prefix.push_back("\xe5\xb9\xb4");
        return prefix;
    };
    const std::vector<Case> cases{
        {"2026/06/05", date_pieces},
        {"2026-06-05", date_pieces},
        {"\xe6\x97\xa5\xe6\x9c\x9f" "2026/06/05", with_prefix(date_pieces)},
        {"\xe6\x97\xa5\xe6\x9c\x9f" "2026-06-05", with_prefix(date_pieces)},
        {"2026", tokenize_plain_four_digit_integer_as_cjk("2026")},
        {"\xef\xbc\x92\xef\xbc\x90\xef\xbc\x92\xef\xbc\x96", tokenize_plain_four_digit_integer_as_cjk("2026")},
        {"2026\xe5\xb9\xb4", with_year_marker(tokenize_digit_run_as_cjk_integer("2026", false))},
        {"\xef\xbc\x92\xef\xbc\x90\xef\xbc\x92\xef\xbc\x96\xe5\xb9\xb4",
         with_year_marker(tokenize_digit_run_as_cjk_integer("2026", false))},
        {"\xef\xbc\x91\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90", tokenize_plain_four_digit_integer_as_cjk("1000")},
        {"\xef\xbc\x91\xef\xbc\x90\xef\xbc\x90\xef\xbc\x91", tokenize_plain_four_digit_integer_as_cjk("1001")},
        {"\xef\xbc\x91\xef\xbc\x90\xef\xbc\x91\xef\xbc\x90", tokenize_plain_four_digit_integer_as_cjk("1010")},
        {"\xef\xbc\x91\xef\xbc\x91\xef\xbc\x90\xef\xbc\x90", tokenize_plain_four_digit_integer_as_cjk("1100")},
        {"\xef\xbc\x91\xef\xbc\x92\xef\xbc\x90\xef\xbc\x90", tokenize_plain_four_digit_integer_as_cjk("1200")},
        {"\xef\xbc\x92\xef\xbc\x92\xef\xbc\x90\xef\xbc\x90", tokenize_plain_four_digit_integer_as_cjk("2200")},
        {"\xef\xbc\x92\xef\xbc\x90\xef\xbc\x92\xef\xbc\x96\xef\xbc\x8d\xef\xbc\x90\xef\xbc\x96\xef\xbc\x8d\xef\xbc\x90\xef\xbc\x95",
         with_two_punctuations(
             tokenize_plain_four_digit_integer_as_cjk("2026"),
             "-",
             tokenize_digit_run_as_cjk_integer("06", false),
             tokenize_digit_run_as_cjk_integer("05", false))},
        {"\xef\xbc\x92\xef\xbc\x90\xef\xbc\x92\xef\xbc\x96\xef\xbc\x8f\xef\xbc\x90\xef\xbc\x96\xef\xbc\x8f\xef\xbc\x90\xef\xbc\x95",
         with_two_punctuations(
             tokenize_plain_four_digit_integer_as_cjk("2026"),
             "/",
             tokenize_digit_run_as_cjk_integer("06", false),
             tokenize_digit_run_as_cjk_integer("05", false))},
        {"\xe6\x97\xa5\xe6\x9c\x9f\xef\xbc\x92\xef\xbc\x90\xef\xbc\x92\xef\xbc\x96\xef\xbc\x8d\xef\xbc\x90\xef\xbc\x96\xef\xbc\x8d\xef\xbc\x90\xef\xbc\x95",
         with_prefix(with_two_punctuations(
             tokenize_plain_four_digit_integer_as_cjk("2026"),
             "-",
             tokenize_digit_run_as_cjk_integer("06", false),
             tokenize_digit_run_as_cjk_integer("05", false)))},
        {"\xe6\x97\xa5\xe6\x9c\x9f\xef\xbc\x92\xef\xbc\x90\xef\xbc\x92\xef\xbc\x96\xef\xbc\x8f\xef\xbc\x90\xef\xbc\x96\xef\xbc\x8f\xef\xbc\x90\xef\xbc\x95",
         with_prefix(with_two_punctuations(
             tokenize_plain_four_digit_integer_as_cjk("2026"),
             "/",
             tokenize_digit_run_as_cjk_integer("06", false),
             tokenize_digit_run_as_cjk_integer("05", false)))},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_date_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_time_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<std::string> twelve_thirty{
        space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x8c", space_piece, "\xe7\x82\xb9",
        space_piece, "\xe4\xb8\x89", space_piece, "\xe5\x8d\x81", space_piece, "\xe5\x88\x86"
    };
    const std::vector<std::string> twelve_thirty_forty_five{
        space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x8c", space_piece, "\xe7\x82\xb9",
        space_piece, "\xe4\xb8\x89", space_piece, "\xe5\x8d\x81", space_piece, "\xe5\x88\x86",
        space_piece, "\xe5\x9b\x9b", space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x94",
        space_piece, "\xe7\xa7\x92"
    };
    const std::vector<std::string> nine_oh_five{
        space_piece, "\xe4\xb9\x9d", space_piece, "\xe7\x82\xb9", space_piece, "\xe9\x9b\xb6",
        space_piece, "\xe4\xba\x94", space_piece, "\xe5\x88\x86"
    };
    const std::vector<std::string> twelve_zero{
        space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x8c", space_piece, "\xe7\x82\xb9"
    };
    const std::vector<std::string> twelve_zero_five_seconds{
        space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x8c", space_piece, "\xe7\x82\xb9",
        space_piece, "\xe4\xba\x94", space_piece, "\xe7\xa7\x92"
    };
    auto with_time_prefix = [&](std::vector<std::string> suffix) {
        std::vector<std::string> out{space_piece, "\xe6\x97\xb6", space_piece, "\xe9\x97\xb4"};
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto with_morning_prefix = [&](std::vector<std::string> suffix) {
        std::vector<std::string> out{space_piece, "\xe4\xb8\x8a", space_piece, "\xe5\x8d\x88"};
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    const std::vector<Case> cases{
        {"12:30", twelve_thirty},
        {"12\xef\xbc\x9a" "30", twelve_thirty},
        {"\xe6\x97\xb6\xe9\x97\xb4" "12:30", with_time_prefix(twelve_thirty)},
        {"12:30:45", twelve_thirty_forty_five},
        {"\xe6\x97\xb6\xe9\x97\xb4" "12:30:45", with_time_prefix(twelve_thirty_forty_five)},
        {"9:05", nine_oh_five},
        {"09:05", nine_oh_five},
        {"\xe4\xb8\x8a\xe5\x8d\x88" "9:05", with_morning_prefix(nine_oh_five)},
        {"12:00", twelve_zero},
        {"12:00:05", twelve_zero_five_seconds},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_time_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_currency_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::vector<Case> cases{
        {"\xef\xbf\xa5" "12", tokenize_currency_number_as_cjk("12", "", false)},
        {"\xc2\xa5" "12", tokenize_currency_number_as_cjk("12", "", false)},
        {"\xef\xbf\xa5" "12.5", tokenize_currency_number_as_cjk("12", "5", false)},
        {"\xef\xbf\xa5" "0.5", tokenize_currency_number_as_cjk("0", "5", false)},
        {"\xef\xbf\xa5" "100", tokenize_currency_number_as_cjk("100", "", false)},
        {"\xef\xbf\xa5" "200", tokenize_currency_number_as_cjk("200", "", false)},
        {"\xef\xbf\xa5" "999", tokenize_currency_number_as_cjk("999", "", false)},
        {"\xef\xbf\xa5" "1000", tokenize_currency_number_as_cjk("1000", "", false)},
        {"\xef\xbf\xa5" "2000", tokenize_currency_number_as_cjk("2000", "", false)},
        {"\xef\xbf\xa5" "2200", tokenize_currency_number_as_cjk("2200", "", false)},
        {"\xef\xbf\xa5" "10000", tokenize_currency_number_as_cjk("10000", "", false)},
        {"\xef\xbf\xa5" "10001", tokenize_currency_number_as_cjk("10001", "", false)},
        {"\xef\xbf\xa5" "10010", tokenize_currency_number_as_cjk("10010", "", false)},
        {"\xef\xbf\xa5" "20000", tokenize_currency_number_as_cjk("20000", "", false)},
        {"\xef\xbf\xa5" "10000.5", tokenize_currency_number_as_cjk("10000", "5", false)},
        {"\xef\xbf\xa5" "100.05", tokenize_currency_number_as_cjk("100", "05", false)},
        {"\xef\xbf\xa5\xef\xbc\x91\xef\xbc\x92", tokenize_currency_number_as_cjk("12", "", false)},
        {"\xef\xbf\xa5\xef\xbc\x91\xef\xbc\x92.\xef\xbc\x95", tokenize_currency_number_as_cjk("12", "5", false)},
        {"\xef\xbf\xa5\xef\xbc\x90.\xef\xbc\x95", tokenize_currency_number_as_cjk("0", "5", false)},
        {"\xef\xbf\xa5" "100.\xef\xbc\x90\xef\xbc\x95", tokenize_currency_number_as_cjk("100", "05", false)},
        {"\xef\xbf\xa5" "01", tokenize_leading_zero_currency_as_cjk("01", "", false)},
        {"\xef\xbf\xa5\xef\xbc\x90\xef\xbc\x91", tokenize_leading_zero_currency_as_cjk("01", "", false)},
        {"\xef\xbf\xa5" "05", tokenize_leading_zero_currency_as_cjk("05", "", false)},
        {"\xef\xbf\xa5" "012", tokenize_leading_zero_currency_as_cjk("012", "", false)},
        {"\xef\xbf\xa5" "01000", tokenize_leading_zero_currency_as_cjk("01000", "", false)},
        {"\xef\xbf\xa5" "01000.5", tokenize_leading_zero_currency_as_cjk("01000", "5", false)},
        {"\xef\xbf\xa5\xef\xbc\x90\xef\xbc\x91\xef\xbc\x92", tokenize_leading_zero_currency_as_cjk("012", "", false)},
        {"\xef\xbf\xa5\xef\xbc\x90\xef\xbc\x91\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90",
         tokenize_leading_zero_currency_as_cjk("01000", "", false)},
        {"\xef\xbf\xa5\xef\xbc\x90\xef\xbc\x91\xef\xbc\x90", tokenize_leading_zero_currency_as_cjk("010", "", false)},
        {"\xef\xbf\xa5\xef\xbc\x90\xef\xbc\x91\xef\xbc\x92.\xef\xbc\x93\xef\xbc\x94",
         tokenize_leading_zero_currency_as_cjk("012", "34", false)},
        {"\xef\xbf\xa5" "001", tokenize_multi_leading_zero_yen_as_reference_digits("001")},
        {"\xef\xbf\xa5\xef\xbc\x90\xef\xbc\x90\xef\xbc\x91", tokenize_multi_leading_zero_yen_as_reference_digits("001")},
        {"\xef\xbf\xa5\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90", tokenize_multi_leading_zero_yen_as_reference_digits("000")},
        {"\xc2\xa5" "01", tokenize_leading_zero_currency_as_cjk("01", "", false)},
        {"\xc2\xa5\xef\xbc\x90\xef\xbc\x91", tokenize_leading_zero_currency_as_cjk("01", "", false)},
        {"\xc2\xa5" "012", tokenize_leading_zero_currency_as_cjk("012", "", false)},
        {"\xc2\xa5\xef\xbc\x90\xef\xbc\x91\xef\xbc\x92", tokenize_leading_zero_currency_as_cjk("012", "", false)},
        {"\xc2\xa5" "001", tokenize_multi_leading_zero_yen_as_reference_digits("001")},
        {"\xc2\xa5\xef\xbc\x90\xef\xbc\x90\xef\xbc\x91", tokenize_multi_leading_zero_yen_as_reference_digits("001")},
        {"$12", tokenize_currency_number_as_cjk("12", "", true)},
        {"$12.5", tokenize_currency_number_as_cjk("12", "5", true)},
        {"$1000", tokenize_currency_number_as_cjk("1000", "", true)},
        {"$2000", tokenize_currency_number_as_cjk("2000", "", true)},
        {"$10000", tokenize_currency_number_as_cjk("10000", "", true)},
        {"$10000.5", tokenize_currency_number_as_cjk("10000", "5", true)},
        {"$100.05", tokenize_currency_number_as_cjk("100", "05", true)},
        {"$\xef\xbc\x91\xef\xbc\x92", tokenize_currency_number_as_cjk("12", "", true)},
        {"$\xef\xbc\x91\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90", tokenize_currency_number_as_cjk("1000", "", true)},
        {"$01", tokenize_leading_zero_currency_as_cjk("01", "", true)},
        {"$\xef\xbc\x90\xef\xbc\x91", tokenize_leading_zero_currency_as_cjk("01", "", true)},
        {"$05", tokenize_leading_zero_currency_as_cjk("05", "", true)},
        {"$012", tokenize_leading_zero_currency_as_cjk("012", "", true)},
        {"$01000", tokenize_leading_zero_currency_as_cjk("01000", "", true)},
        {"$01000.5", tokenize_leading_zero_currency_as_cjk("01000", "5", true)},
        {"$\xef\xbc\x90\xef\xbc\x91\xef\xbc\x92", tokenize_leading_zero_currency_as_cjk("012", "", true)},
        {"$\xef\xbc\x90\xef\xbc\x91\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90",
         tokenize_leading_zero_currency_as_cjk("01000", "", true)},
        {"$001", tokenize_multi_leading_zero_dollar_as_reference_digits("001")},
        {"$\xef\xbc\x90\xef\xbc\x90\xef\xbc\x91", tokenize_multi_leading_zero_dollar_as_reference_digits("001")},
        {"$\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90", tokenize_multi_leading_zero_dollar_as_reference_digits("000")},
    };
    const std::vector<std::string> unsupported{
        "\xef\xbf\xa5" "100000",
        "$100000",
        "\xef\xbf\xa5" "010000",
        "$010000",
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_currency_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::vector<std::string> rejected;
    for (const auto& text : unsupported) {
        try {
            (void)tokenize_cjk_text(piece_to_id, text);
        } catch (const std::exception&) {
            rejected.push_back(text);
        }
    }
    const bool unsupported_rejected = rejected.size() == unsupported.size();
    ok = ok && unsupported_rejected;
    std::cout << "  ],\n";
    std::cout << "  \"fullwidth_digit_currency_supported\": true,\n";
    std::cout << "  \"single_leading_zero_currency_supported\": true,\n";
    std::cout << "  \"multi_leading_zero_yen_supported\": true,\n";
    std::cout << "  \"multi_leading_zero_dollar_supported\": true,\n";
    std::cout << "  \"focused_5_digit_currency_supported\": true,\n";
    std::cout << "  \"currency_out_of_scope_requires_full_text_normalizer\": " << (unsupported_rejected ? "true" : "false") << ",\n";
    std::cout << "  \"rejected\": ";
    print_json_string_array(rejected);
    std::cout << ",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_phone_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    auto with_phone_prefix = [&](std::vector<std::string> suffix) {
        std::vector<std::string> out{space_piece, "\xe7\x94\xb5", space_piece, "\xe8\xaf\x9d"};
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto with_punctuation = [&](std::vector<std::string> lhs,
                                const std::string& punctuation,
                                std::vector<std::string> rhs) {
        append_normalized_ascii_punctuation(lhs, piece_to_id, punctuation, false);
        lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        return lhs;
    };
    auto with_ascii_word_punctuation = [&](std::vector<std::string> lhs,
                                           const std::string& punctuation,
                                           std::vector<std::string> rhs) {
        append_normalized_ascii_punctuation(lhs, piece_to_id, punctuation, true);
        lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        return lhs;
    };
    auto with_unknown_punctuation = [&](std::vector<std::string> lhs,
                                        const std::string& punctuation,
                                        std::vector<std::string> rhs) {
        append_unknown_id_punctuation(lhs, piece_to_id, punctuation, false);
        lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        return lhs;
    };
    const std::vector<std::string> nihao{space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"};
    const std::vector<std::string> shijie{space_piece, "\xe4\xb8\x96", space_piece, "\xe7\x95\x8c"};
    const std::vector<Case> cases{
        {"138-0013-8000", tokenize_phone_digits_as_cjk("13800138000")},
        {"\xe7\x94\xb5\xe8\xaf\x9d" "138-0013-8000", with_phone_prefix(tokenize_phone_digits_as_cjk("13800138000"))},
        {"010-12345678", tokenize_phone_digits_as_cjk("01012345678")},
        {"400-800-1234", tokenize_phone_digits_as_cjk("4008001234")},
        {"\xef\xbc\x91\xef\xbc\x93\xef\xbc\x98-\xef\xbc\x90\xef\xbc\x90\xef\xbc\x91\xef\xbc\x93-\xef\xbc\x98\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90",
         tokenize_phone_digits_as_cjk("13800138000")},
        {"138\xef\xbc\x8d" "0013\xef\xbc\x8d" "8000",
         tokenize_fullwidth_hyphen_grouped_3_4_4_as_cjk("138", "0013", "8000")},
        {"\xe7\x94\xb5\xe8\xaf\x9d" "138\xef\xbc\x8d" "0013\xef\xbc\x8d" "8000",
         with_phone_prefix(tokenize_fullwidth_hyphen_grouped_3_4_4_as_cjk("138", "0013", "8000"))},
        {"\xef\xbc\x91\xef\xbc\x93\xef\xbc\x98\xef\xbc\x8d\xef\xbc\x90\xef\xbc\x90\xef\xbc\x91\xef\xbc\x93\xef\xbc\x8d\xef\xbc\x98\xef\xbc\x90\xef\xbc\x90\xef\xbc\x90",
         tokenize_fullwidth_hyphen_grouped_3_4_4_as_cjk("138", "0013", "8000")},
        {"1-2", tokenize_short_hyphen_subtraction_as_cjk("1", "2")},
        {"12-34", tokenize_short_hyphen_subtraction_as_cjk("12", "34")},
        {"\xe5\xae\x8c\xe6\x88\x90" "12-34",
         [&] {
             std::vector<std::string> out{space_piece, "\xe5\xae\x8c", space_piece, "\xe6\x88\x90"};
             auto suffix = tokenize_short_hyphen_subtraction_as_cjk("12", "34");
             out.insert(out.end(), suffix.begin(), suffix.end());
             return out;
         }()},
        {"\xef\xbc\x91\xef\xbc\x8d\xef\xbc\x92",
         with_punctuation(tokenize_digit_run_as_cjk_integer("1", false), "-", tokenize_digit_run_as_cjk_integer("2", false))},
        {"\xef\xbc\x91\xef\xbc\x92\xef\xbc\x8d\xef\xbc\x93\xef\xbc\x94",
         with_punctuation(tokenize_digit_run_as_cjk_integer("12", false), "-", tokenize_digit_run_as_cjk_integer("34", false))},
        {"\xe5\xae\x8c\xe6\x88\x90\xef\xbc\x91\xef\xbc\x92\xef\xbc\x8d\xef\xbc\x93\xef\xbc\x94",
         [&] {
             std::vector<std::string> out{space_piece, "\xe5\xae\x8c", space_piece, "\xe6\x88\x90"};
             auto suffix = with_punctuation(
                 tokenize_digit_run_as_cjk_integer("12", false),
                 "-",
                 tokenize_digit_run_as_cjk_integer("34", false));
             out.insert(out.end(), suffix.begin(), suffix.end());
             return out;
         }()},
        {"\xef\xbc\x91\xe2\x80\x94\xef\xbc\x92",
         with_punctuation(tokenize_digit_run_as_cjk_integer("1", false), "-", tokenize_digit_run_as_cjk_integer("2", false))},
        {"\xef\xbc\x91\xe2\x80\x93\xef\xbc\x92",
         with_unknown_punctuation(tokenize_digit_run_as_cjk_integer("1", false), "\xe2\x80\x93", tokenize_digit_run_as_cjk_integer("2", false))},
        {"\xe4\xbd\xa0\xe5\xa5\xbd~\xe4\xb8\x96\xe7\x95\x8c", with_punctuation(nihao, "-", shijie)},
        {"A~\xe4\xbd\xa0\xe5\xa5\xbd", with_ascii_word_punctuation(tokenize_ascii_run_prefixed(piece_to_id, "A"), "-", nihao)},
        {"\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbd\x9e\xe4\xb8\x96\xe7\x95\x8c", with_punctuation(nihao, "-", shijie)},
        {"A\xef\xbd\x9e\xe4\xbd\xa0\xe5\xa5\xbd", with_ascii_word_punctuation(tokenize_ascii_run_prefixed(piece_to_id, "A"), "-", nihao)},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_phone_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"short_hyphen_subtraction_supported\": true,\n";
    std::cout << "  \"fullwidth_hyphen_grouped_3_4_4_supported\": true,\n";
    std::cout << "  \"fullwidth_dash_punctuation_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_ratio_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    auto alpha_colon_pieces = [&](const std::string& alpha, const std::string& digits, bool cjk_number_context) {
        auto out = tokenize_ascii_run_prefixed(piece_to_id, alpha);
        append_alpha_colon_number_pieces(out, piece_to_id, digits, cjk_number_context);
        return out;
    };
    auto cjk_integer_pieces = [&](const std::string& digits) {
        std::vector<std::string> out;
        append_cjk_integer_under_10000(out, std::stoi(digits));
        return out;
    };
    auto with_ratio_alpha_prefix = [&](std::vector<std::string> suffix) {
        std::vector<std::string> out{space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xbe\x8b"};
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto with_nihao_suffix = [&](std::vector<std::string> prefix) {
        prefix.insert(prefix.end(), {space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"});
        return prefix;
    };
    auto append_comma_then_digits = [&](std::vector<std::string> prefix,
                                        const std::string& digits,
                                        bool previous_ascii_word) {
        append_normalized_ascii_punctuation(prefix, piece_to_id, ",", previous_ascii_word);
        auto digit_pieces = tokenize_digit_run_as_cjk_integer(digits, false);
        prefix.insert(prefix.end(), digit_pieces.begin(), digit_pieces.end());
        return prefix;
    };
    auto append_comma_then_cjk = [&](std::vector<std::string> prefix,
                                     bool previous_ascii_word,
                                     const std::vector<std::string>& suffix) {
        append_normalized_ascii_punctuation(prefix, piece_to_id, ",", previous_ascii_word);
        prefix.insert(prefix.end(), suffix.begin(), suffix.end());
        return prefix;
    };
    const std::vector<std::string> nihao{space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"};
    const std::vector<std::string> goujian{space_piece, "\xe6\x9e\x84", space_piece, "\xe5\xbb\xba"};
    const std::vector<Case> cases{
        {"1:2", {space_piece, "\xe4\xb8\x80", space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xba\x8c"}},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "1:2",
         {space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xbe\x8b", space_piece, "\xe4\xb8\x80", space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xba\x8c"}},
        {"12:60",
         {space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x8c", space_piece, "\xe6\xaf\x94", space_piece, "\xe5\x85\xad", space_piece, "\xe5\x8d\x81"}},
        {"\xef\xbc\x91\xef\xbc\x92:\xef\xbc\x93\xef\xbc\x90",
         {space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x8c", space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xb8\x89", space_piece, "\xe5\x8d\x81"}},
        {"1\xef\xbc\x9a" "2",
         {space_piece, "\xe4\xb8\x80", space_piece + ",", space_piece, "\xe4\xba\x8c"}},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "1\xef\xbc\x9a" "2",
         {space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xbe\x8b", space_piece, "\xe4\xb8\x80", space_piece + ",", space_piece, "\xe4\xba\x8c"}},
        {"\xef\xbc\x91\xef\xbc\x9a\xef\xbc\x92",
         {space_piece, "\xe4\xb8\x80", space_piece + ",", space_piece, "\xe4\xba\x8c"}},
        {"12\xef\xbc\x9a" "60",
         {space_piece, "\xe5\x8d\x81", space_piece, "\xe4\xba\x8c", space_piece + ",", space_piece, "\xe5\x85\xad", space_piece, "\xe5\x8d\x81"}},
        {"\xef\xbc\x91\xef\xbc\x92\xef\xbc\x9a\xef\xbc\x93\xef\xbc\x90\xef\xbc\x9a\xef\xbc\x94\xef\xbc\x95",
         [&] {
             auto out = append_comma_then_digits(tokenize_digit_run_as_cjk_integer("12", false), "30", false);
             return append_comma_then_digits(std::move(out), "45", false);
         }()},
        {"A:1", alpha_colon_pieces("A", "1", false)},
        {"A:01", alpha_colon_pieces("A", "01", false)},
        {"A:12", alpha_colon_pieces("A", "12", false)},
        {"AB:12", alpha_colon_pieces("AB", "12", false)},
        {"ABC:123", alpha_colon_pieces("ABC", "123", false)},
        {"A\xef\xbc\x9a" "1", alpha_colon_pieces("A", "1", false)},
        {"\xef\xbc\xa1\xef\xbc\x9a\xef\xbc\x91",
         append_comma_then_digits(tokenize_ascii_run_prefixed(piece_to_id, "A"), "1", true)},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "A:1", with_ratio_alpha_prefix(alpha_colon_pieces("A", "1", true))},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "AB:12", with_ratio_alpha_prefix(alpha_colon_pieces("AB", "12", true))},
        {"\xe6\xaf\x94\xe4\xbe\x8b\xef\xbc\xa1\xef\xbc\x9a\xef\xbc\x91",
         with_ratio_alpha_prefix(append_comma_then_digits(tokenize_ascii_run_prefixed(piece_to_id, "A"), "1", true))},
        {"A:1\xe4\xbd\xa0\xe5\xa5\xbd", with_nihao_suffix(alpha_colon_pieces("A", "1", true))},
        {"A:1234\xe4\xbd\xa0\xe5\xa5\xbd", with_nihao_suffix(alpha_colon_pieces("A", "1234", true))},
        {"A\xef\xbc\x9a" "1234\xe4\xbd\xa0\xe5\xa5\xbd", with_nihao_suffix(alpha_colon_pieces("A", "1234", true))},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "A:1234",
         with_ratio_alpha_prefix([&] {
             auto out = tokenize_ascii_run_prefixed(piece_to_id, "A");
             append_normalized_ascii_punctuation(out, piece_to_id, ",", true);
             auto digits = cjk_integer_pieces("1234");
             out.insert(out.end(), digits.begin(), digits.end());
             return out;
         }())},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "A\xef\xbc\x9a" "1234",
         with_ratio_alpha_prefix([&] {
             auto out = tokenize_ascii_run_prefixed(piece_to_id, "A");
             append_normalized_ascii_punctuation(out, piece_to_id, ",", true);
             auto digits = cjk_integer_pieces("1234");
             out.insert(out.end(), digits.begin(), digits.end());
             return out;
         }())},
        {"\xe4\xbb\x96\xe8\xaf\xb4:\xe4\xbd\xa0\xe5\xa5\xbd",
         append_comma_then_cjk({space_piece, "\xe4\xbb\x96", space_piece, "\xe8\xaf\xb4"}, false, nihao)},
        {"\xe4\xbb\x96\xe8\xaf\xb4;\xe4\xbd\xa0\xe5\xa5\xbd",
         append_comma_then_cjk({space_piece, "\xe4\xbb\x96", space_piece, "\xe8\xaf\xb4"}, false, nihao)},
        {"A:\xe4\xbd\xa0\xe5\xa5\xbd",
         append_comma_then_cjk(tokenize_ascii_run_prefixed(piece_to_id, "A"), true, nihao)},
        {"A;\xe4\xbd\xa0\xe5\xa5\xbd",
         append_comma_then_cjk(tokenize_ascii_run_prefixed(piece_to_id, "A"), true, nihao)},
        {"hello:\xe4\xbd\xa0\xe5\xa5\xbd",
         append_comma_then_cjk(tokenize_ascii_run_prefixed(piece_to_id, "hello"), true, nihao)},
        {"Rag\xef\xbc\x9a\xe4\xbd\xa0\xe5\xa5\xbd",
         append_comma_then_cjk(tokenize_ascii_run_prefixed(piece_to_id, "Rag"), true, nihao)},
        {"Graph Rag\xef\xbc\x9a\xe6\x9e\x84\xe5\xbb\xba",
         [&] {
             auto out = tokenize_ascii_run_prefixed(piece_to_id, "Graph");
             auto rag = tokenize_ascii_run_prefixed(piece_to_id, "Rag");
             out.insert(out.end(), rag.begin(), rag.end());
             return append_comma_then_cjk(std::move(out), true, goujian);
         }()},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_ratio_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    bool alpha_colon_long_digit_rejected = false;
    try {
        (void)tokenize_cjk_text(piece_to_id, "A:1234");
    } catch (const std::runtime_error&) {
        alpha_colon_long_digit_rejected = true;
    }
    ok = ok && alpha_colon_long_digit_rejected;
    std::cout << "  ],\n";
    std::cout << "  \"fullwidth_colon_digit_separator_supported\": true,\n";
    std::cout << "  \"alpha_colon_digit_supported\": true,\n";
    std::cout << "  \"alpha_colon_long_digit_requires_full_text_normalizer\": " << (alpha_colon_long_digit_rejected ? "true" : "false") << ",\n";
    std::cout << "  \"fullwidth_colon_punctuation_digits_supported\": true,\n";
    std::cout << "  \"ascii_colon_semicolon_cjk_punctuation_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_fraction_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    auto with_ratio_prefix = [&](std::vector<std::string> suffix) {
        std::vector<std::string> out{space_piece, "\xe6\xaf\x94", space_piece, "\xe4\xbe\x8b"};
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto with_done_suffix = [&](std::vector<std::string> prefix) {
        prefix.insert(prefix.end(), {space_piece, "\xe5\xae\x8c", space_piece, "\xe6\x88\x90"});
        return prefix;
    };
    const std::vector<Case> cases{
        {"01/02", [&] { std::vector<std::string> out; append_date_md_pieces(out, 1, 2); return out; }()},
        {"09/5", [&] { std::vector<std::string> out; append_date_md_pieces(out, 9, 5); return out; }()},
        {"10/2", [&] { std::vector<std::string> out; append_date_md_pieces(out, 10, 2); return out; }()},
        {"12/03", [&] { std::vector<std::string> out; append_date_md_pieces(out, 12, 3); return out; }()},
        {"1/2", tokenize_fraction_as_cjk("1", "2")},
        {"3/4", tokenize_fraction_as_cjk("3", "4")},
        {"12/34", tokenize_fraction_as_cjk("12", "34")},
        {"1/20", tokenize_fraction_as_cjk("1", "20")},
        {"2/10", tokenize_fraction_as_cjk("2", "10")},
        {"9/5", tokenize_fraction_as_cjk("9", "5")},
        {"13/20", tokenize_fraction_as_cjk("13", "20")},
        {"20/10", tokenize_fraction_as_cjk("20", "10")},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "1/2", with_ratio_prefix(tokenize_fraction_as_cjk("1", "2"))},
        {"1/2\xe5\xae\x8c\xe6\x88\x90", with_done_suffix(tokenize_fraction_as_cjk("1", "2"))},
        {"1/02", tokenize_leading_zero_denominator_fraction_as_cjk("1", "02")},
        {"2/03", tokenize_leading_zero_denominator_fraction_as_cjk("2", "03")},
        {"9/05", tokenize_leading_zero_denominator_fraction_as_cjk("9", "05")},
        {"1/002", tokenize_leading_zero_denominator_fraction_as_cjk("1", "002")},
        {"2/003", tokenize_leading_zero_denominator_fraction_as_cjk("2", "003")},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "1/02", with_ratio_prefix(tokenize_leading_zero_denominator_fraction_as_cjk("1", "02"))},
        {"1/02\xe5\xae\x8c\xe6\x88\x90", with_done_suffix(tokenize_leading_zero_denominator_fraction_as_cjk("1", "02"))},
        {"1/2/3", tokenize_single_digit_chained_slash_fraction_as_cjk(piece_to_id, "1", "2", "3")},
        {"\xe6\xaf\x94\xe4\xbe\x8b" "1/2/3",
         with_ratio_prefix(tokenize_single_digit_chained_slash_fraction_as_cjk(piece_to_id, "1", "2", "3"))},
        {"1/2/3\xe5\xae\x8c\xe6\x88\x90",
         with_done_suffix(tokenize_single_digit_chained_slash_fraction_as_cjk(piece_to_id, "1", "2", "3"))},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_fraction_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"leading_zero_denominator_supported\": true,\n";
    std::cout << "  \"single_digit_chained_slash_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_quote_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<std::string> tashuo{space_piece, "\xe4\xbb\x96", space_piece, "\xe8\xaf\xb4"};
    const std::vector<std::string> nihao{space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"};
    const std::vector<std::string> shijie{space_piece, "\xe4\xb8\x96", space_piece, "\xe7\x95\x8c"};
    auto append_suffix = [](std::vector<std::string> out, const std::vector<std::string>& suffix) {
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto append_comma = [&](std::vector<std::string> out, bool previous_ascii_word) {
        append_normalized_ascii_punctuation(out, piece_to_id, ",", previous_ascii_word);
        return out;
    };
    auto append_quote = [&](std::vector<std::string> out, bool previous_ascii_word) {
        append_quote_mark_punctuation(out, piece_to_id, previous_ascii_word);
        return out;
    };
    auto append_exclamation = [&](std::vector<std::string> out) {
        append_normalized_ascii_punctuation(out, piece_to_id, "!", false);
        return out;
    };
    auto append_period = [&](std::vector<std::string> out) {
        append_normalized_ascii_punctuation(out, piece_to_id, ".", false);
        return out;
    };
    auto tashuo_colon_quote_nihao = [&](bool with_exclamation) {
        auto out = append_quote(append_comma(tashuo, false), false);
        out = append_suffix(std::move(out), nihao);
        if (with_exclamation) {
            out = append_exclamation(std::move(out));
        }
        return append_quote(std::move(out), false);
    };
    auto tashuo_quote_nihao = [&] {
        auto out = append_quote(tashuo, false);
        out = append_suffix(std::move(out), nihao);
        return append_quote(std::move(out), false);
    };
    auto quoted_nihao = [&] {
        auto out = append_quote({}, false);
        out = append_suffix(std::move(out), nihao);
        return append_quote(std::move(out), false);
    };
    auto quoted_nihao_comma_tashuo = [&] {
        auto out = append_comma(quoted_nihao(), false);
        return append_suffix(std::move(out), tashuo);
    };
    auto tashuo_colon_quote_nihao_period = [&] {
        return append_period(tashuo_colon_quote_nihao(false));
    };
    auto nihao_quote_shijie = [&] {
        auto out = append_quote(nihao, false);
        return append_suffix(std::move(out), shijie);
    };
    auto ascii_quoted = [&] {
        auto out = tokenize_ascii_run_prefixed(piece_to_id, "A");
        append_quote_mark_punctuation(out, piece_to_id, true);
        out.push_back("B");
        return out;
    };
    const std::vector<Case> cases{
        {"\xe4\xbb\x96\xe8\xaf\xb4:\"\xe4\xbd\xa0\xe5\xa5\xbd\"", tashuo_colon_quote_nihao(false)},
        {"\xe4\xbb\x96\xe8\xaf\xb4:\"\xe4\xbd\xa0\xe5\xa5\xbd!\"", tashuo_colon_quote_nihao(true)},
        {"\xe4\xbb\x96\xe8\xaf\xb4:\"\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x81\"", tashuo_colon_quote_nihao(true)},
        {"\xe4\xbb\x96\xe8\xaf\xb4\"\xe4\xbd\xa0\xe5\xa5\xbd\"", tashuo_quote_nihao()},
        {"A\"B", ascii_quoted()},
        {"\"\xe4\xbd\xa0\xe5\xa5\xbd\"", quoted_nihao()},
        {"\xe4\xbd\xa0\xe5\xa5\xbd\"\xe4\xb8\x96\xe7\x95\x8c", nihao_quote_shijie()},
        {"\xe4\xbb\x96\xe8\xaf\xb4\xef\xbc\x9a\xe2\x80\x9c\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x81\xe2\x80\x9d",
         tashuo_colon_quote_nihao(true)},
        {"\xe4\xbb\x96\xe8\xaf\xb4\xe2\x80\x9c\xe4\xbd\xa0\xe5\xa5\xbd\xe2\x80\x9d", tashuo_quote_nihao()},
        {"\xe2\x80\x9c\xe4\xbd\xa0\xe5\xa5\xbd\xe2\x80\x9d\xef\xbc\x8c\xe4\xbb\x96\xe8\xaf\xb4",
         quoted_nihao_comma_tashuo()},
        {"\"\xe4\xbd\xa0\xe5\xa5\xbd\",\xe4\xbb\x96\xe8\xaf\xb4", quoted_nihao_comma_tashuo()},
        {"\xe4\xbb\x96\xe8\xaf\xb4\xef\xbc\x9a\xe2\x80\x9c\xe4\xbd\xa0\xe5\xa5\xbd\xe2\x80\x9d\xe3\x80\x82",
         tashuo_colon_quote_nihao_period()},
        {"\xe4\xbb\x96\xe8\xaf\xb4:\"\xe4\xbd\xa0\xe5\xa5\xbd\".", tashuo_colon_quote_nihao_period()},
        {"\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x8c\xe4\xb8\x96\xe7\x95\x8c",
         append_suffix(append_quote(nihao, false), shijie)},
        {"A\xe3\x80\x8c\xe4\xbd\xa0\xe5\xa5\xbd",
         append_suffix(append_quote(tokenize_ascii_run_prefixed(piece_to_id, "A"), true), nihao)},
        {"\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x8d\xe4\xb8\x96\xe7\x95\x8c",
         append_suffix(append_quote(nihao, false), shijie)},
        {"A\xe3\x80\x8d\xe4\xbd\xa0\xe5\xa5\xbd",
         append_suffix(append_quote(tokenize_ascii_run_prefixed(piece_to_id, "A"), true), nihao)},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_quote_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"ascii_double_quote_supported\": true,\n";
    std::cout << "  \"quote_mark_spacing_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_ellipsis_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<std::string> nihao{space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"};
    const std::vector<std::string> shijie{space_piece, "\xe4\xb8\x96", space_piece, "\xe7\x95\x8c"};
    auto append_suffix = [](std::vector<std::string> out, const std::vector<std::string>& suffix) {
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto append_ellipsis = [&](std::vector<std::string> out, bool previous_ascii_word) {
        append_normalized_ascii_punctuation(out, piece_to_id, "...", previous_ascii_word);
        return out;
    };
    auto nihao_ellipsis_shijie = [&] {
        return append_suffix(append_ellipsis(nihao, false), shijie);
    };
    auto hello_ellipsis_world = [&] {
        auto out = tokenize_ascii_run_prefixed(piece_to_id, "hello");
        append_normalized_ascii_punctuation(out, piece_to_id, "...", true);
        auto suffix = tokenize_ascii_run_unprefixed(piece_to_id, "world");
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    const std::vector<Case> cases{
        {"\xe4\xbd\xa0\xe5\xa5\xbd...\xe4\xb8\x96\xe7\x95\x8c", nihao_ellipsis_shijie()},
        {"\xe4\xbd\xa0\xe5\xa5\xbd\xe2\x80\xa6\xe4\xb8\x96\xe7\x95\x8c", nihao_ellipsis_shijie()},
        {"\xe4\xbd\xa0\xe5\xa5\xbd\xe2\x80\xa6\xe2\x80\xa6\xe4\xb8\x96\xe7\x95\x8c", nihao_ellipsis_shijie()},
        {"hello...world", hello_ellipsis_world()},
        {"3...4",
         append_suffix(append_ellipsis(tokenize_digit_run_as_cjk_integer("3", false), false),
                       tokenize_digit_run_as_cjk_integer("4", false))},
        {"...\xe4\xbd\xa0\xe5\xa5\xbd", append_suffix(append_ellipsis({}, false), nihao)},
        {"\xe4\xbd\xa0\xe5\xa5\xbd...", append_ellipsis(nihao, false)},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_ellipsis_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"ascii_ellipsis_supported\": true,\n";
    std::cout << "  \"unicode_ellipsis_spacing_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_percent_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::vector<std::string> wancheng{
        "\xe2\x96\x81", "\xe5\xae\x8c", "\xe2\x96\x81", "\xe6\x88\x90"};
    auto append_suffix = [](std::vector<std::string> out, const std::vector<std::string>& suffix) {
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto percent = [](const std::string& integer, const std::string& fraction = "") {
        return tokenize_percent_number_as_cjk(integer, fraction);
    };
    auto fullwidth_unknown_percent = [&](const std::string& digits) {
        auto out = tokenize_digit_run_as_cjk_integer(digits, false);
        append_normalized_ascii_punctuation(out, piece_to_id, "%", false);
        return out;
    };

    const std::vector<Case> cases{
        {"5%", percent("5")},
        {"50%", percent("50")},
        {"100%", percent("100")},
        {"101%", percent("101")},
        {"1000%", percent("1000")},
        {"2026%", percent("2026")},
        {"12%", percent("12")},
        {"12.5%", percent("12", "5")},
        {"1000.5%", percent("1000", "5")},
        {"0.5%", percent("0", "5")},
        {"\xe5\xae\x8c\xe6\x88\x90" "50%", append_suffix(wancheng, percent("50"))},
        {"\xe5\xae\x8c\xe6\x88\x90" "1000%", append_suffix(wancheng, percent("1000"))},
        {"50%\xe5\xae\x8c\xe6\x88\x90", append_suffix(percent("50"), wancheng)},
        {"\xef\xbc\x91\xef\xbc\x90\xef\xbc\x90%", percent("100")},
        {"\xef\xbc\x95\xef\xbc\x90\xef\xbc\x85", fullwidth_unknown_percent("50")},
        {"100\xef\xbc\x85\xe5\xae\x8c\xe6\x88\x90", append_suffix(fullwidth_unknown_percent("100"), wancheng)},
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_percent_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"halfwidth_percent_normalizer_supported\": true,\n";
    std::cout << "  \"fullwidth_percent_unknown_id_supported\": true,\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

bool run_text_ids_cjk_no_tokenizer_test(const std::string& tokenizer_dir) {
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    struct Case {
        std::string text;
        std::vector<std::string> pieces;
    };
    const std::string space_piece = "\xe2\x96\x81";
    const std::vector<std::string> nihao{space_piece, "\xe4\xbd\xa0", space_piece, "\xe5\xa5\xbd"};
    const std::vector<std::string> bianhao{
        space_piece, "\xe7\xbc\x96", space_piece, "\xe5\x8f\xb7"};
    const std::vector<std::string> wancheng{
        space_piece, "\xe5\xae\x8c", space_piece, "\xe6\x88\x90"};
    auto append_suffix = [](std::vector<std::string> out, const std::vector<std::string>& suffix) {
        out.insert(out.end(), suffix.begin(), suffix.end());
        return out;
    };
    auto number = [&](const std::string& digits) {
        std::vector<std::string> out{space_piece + "NUMBER"};
        auto number_pieces = tokenize_english_no_number_run(digits);
        out.insert(out.end(), number_pieces.begin(), number_pieces.end());
        return out;
    };
    auto spaced_number = [&](const std::string& digits) {
        std::vector<std::string> out{space_piece + "NUMBER"};
        if (!(digits.size() > 1 && digits[0] == '0')) {
            out.push_back(".");
        }
        auto number_pieces = tokenize_english_spaced_no_number_run(digits);
        out.insert(out.end(), number_pieces.begin(), number_pieces.end());
        return out;
    };
    auto lower_no_decimal = [&](const std::string& digits) {
        auto out = tokenize_ascii_run_prefixed(piece_to_id, "NO");
        out.push_back(space_piece + "POINT");
        for (char digit : digits) {
            if (digit == '0') {
                out.push_back(space_piece + "OH");
            } else {
                append_english_under_100(out, digit - '0');
            }
        }
        return out;
    };
    auto old_no_cjk_number = [&](const std::string& digits) {
        auto out = tokenize_ascii_run_prefixed(piece_to_id, "NO");
        append_normalized_ascii_punctuation(out, piece_to_id, ".", true);
        auto digit_pieces = tokenize_digit_run_as_cjk_integer(digits, false);
        out.insert(out.end(), digit_pieces.begin(), digit_pieces.end());
        return out;
    };
    auto append_period = [&](std::vector<std::string> out) {
        append_normalized_ascii_punctuation(out, piece_to_id, ".", false);
        return out;
    };

    const std::vector<Case> cases{
        {"No.1", number("1")},
        {"NO.1", number("1")},
        {"No.2", number("2")},
        {"No.12", number("12")},
        {"No.01", number("01")},
        {"No.001", number("001")},
        {"No.100", number("100")},
        {"No.101", number("101")},
        {"No.999", number("999")},
        {"No.0000", number("0000")},
        {"No.0001", number("0001")},
        {"No.0010", number("0010")},
        {"No.0101", number("0101")},
        {"No.0999", number("0999")},
        {"No.1000", number("1000")},
        {"No.1001", number("1001")},
        {"No.1010", number("1010")},
        {"No.1999", number("1999")},
        {"No.2001", number("2001")},
        {"No.2026", number("2026")},
        {"No.3001", number("3001")},
        {"No.3100", number("3100")},
        {"No.9999", number("9999")},
        {"No.10000", number("10000")},
        {"No.10001", number("10001")},
        {"No.10100", number("10100")},
        {"No.10101", number("10101")},
        {"No.11000", number("11000")},
        {"No.99999", number("99999")},
        {"NO.2026", number("2026")},
        {"No. 1", spaced_number("1")},
        {"NO. 12", spaced_number("12")},
        {"No. 01", spaced_number("01")},
        {"No. 001", spaced_number("001")},
        {"No. 100", spaced_number("100")},
        {"No. 1000", spaced_number("1000")},
        {"No. 1010", spaced_number("1010")},
        {"No. 2001", spaced_number("2001")},
        {"No. 2026", spaced_number("2026")},
        {"No. 9999", spaced_number("9999")},
        {"No. 10000", spaced_number("10000")},
        {"No. 10001", spaced_number("10001")},
        {"No. 10100", spaced_number("10100")},
        {"No. 10101", spaced_number("10101")},
        {"No. 11000", spaced_number("11000")},
        {"No. 99999", spaced_number("99999")},
        {"no. 2026", spaced_number("2026")},
        {"No.1.", append_period(number("1"))},
        {"no.1", lower_no_decimal("1")},
        {"no.12", lower_no_decimal("12")},
        {"no.01", lower_no_decimal("01")},
        {"no.100", lower_no_decimal("100")},
        {"no.1000", lower_no_decimal("1000")},
        {"no.2026", lower_no_decimal("2026")},
        {"no.10000", lower_no_decimal("10000")},
        {"no.10001", lower_no_decimal("10001")},
        {"no.1.", append_period(lower_no_decimal("1"))},
        {"No.1\xe4\xbd\xa0\xe5\xa5\xbd", append_suffix(old_no_cjk_number("1"), nihao)},
        {"\xe7\xbc\x96\xe5\x8f\xb7No.1", append_suffix(bianhao, old_no_cjk_number("1"))},
        {"NO.12\xe5\xae\x8c\xe6\x88\x90", append_suffix(old_no_cjk_number("12"), wancheng)},
    };
    const std::vector<std::string> unsupported{
        "No.100000",
        "No. 100000",
        "no.100000",
    };

    bool ok = true;
    std::cout << "{\n";
    std::cout << "  \"stage\": \"text_ids_cjk_no_tokenizer\",\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto tokenized = tokenize_cjk_text(piece_to_id, cases[i].text);
        const bool match = tokenized.pieces == cases[i].pieces;
        ok = ok && match;
        std::cout << "    {\n";
        std::cout << "      \"text\": \"" << json_escape(cases[i].text) << "\",\n";
        std::cout << "      \"ok\": " << (match ? "true" : "false") << ",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"expected_pieces\": ";
        print_json_string_array(cases[i].pieces);
        std::cout << "\n";
        std::cout << "    }" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    std::vector<std::string> rejected;
    for (const auto& text : unsupported) {
        try {
            (void)tokenize_cjk_text(piece_to_id, text);
        } catch (const std::exception&) {
            rejected.push_back(text);
        }
    }
    const bool unsupported_rejected = rejected.size() == unsupported.size();
    ok = ok && unsupported_rejected;
    std::cout << "  ],\n";
    std::cout << "  \"ascii_no_number_focused_5_digit_supported\": true,\n";
    std::cout << "  \"ascii_spaced_no_number_supported\": true,\n";
    std::cout << "  \"lowercase_no_decimal_supported\": true,\n";
    std::cout << "  \"cjk_context_no_number_preserved\": true,\n";
    std::cout << "  \"no_number_out_of_scope_requires_full_text_normalizer\": " << (unsupported_rejected ? "true" : "false") << ",\n";
    std::cout << "  \"rejected\": ";
    print_json_string_array(rejected);
    std::cout << ",\n";
    std::cout << "  \"ok\": " << (ok ? "true" : "false") << "\n";
    std::cout << "}\n";
    return ok;
}

struct CjkSegmentRange {
    size_t start = 0;
    size_t end = 0;

    size_t size() const {
        return end - start;
    }
};

static bool cjk_split_tokens_contain(const std::vector<std::string>& split_tokens,
                                     const std::string& piece) {
    return std::find(split_tokens.begin(), split_tokens.end(), piece) != split_tokens.end();
}

static bool cjk_range_contains_piece(const std::vector<std::string>& pieces,
                                     size_t start,
                                     size_t end,
                                     const std::string& needle) {
    for (size_t i = start; i < end; ++i) {
        if (pieces[i] == needle) {
            return true;
        }
    }
    return false;
}

static void append_cjk_token_chunks(std::vector<CjkSegmentRange>& out,
                                    size_t start,
                                    size_t end,
                                    size_t max_tokens) {
    for (size_t pos = start; pos < end;) {
        const size_t next = std::min(pos + max_tokens, end);
        out.push_back(CjkSegmentRange{pos, next});
        pos = next;
    }
}

static std::vector<CjkSegmentRange> split_cjk_ranges_by_token(
    const std::vector<std::string>& pieces,
    size_t start,
    size_t end,
    const std::vector<std::string>& split_tokens,
    size_t max_tokens,
    size_t quick_streaming_tokens = 0) {
    if (start >= end) {
        return {};
    }

    std::vector<CjkSegmentRange> segments;
    size_t current_start = start;
    size_t current_len = 0;
    for (size_t i = start; i < end; ++i) {
        ++current_len;
        const bool split_has_comma =
            cjk_split_tokens_contain(split_tokens, ",") ||
            cjk_split_tokens_contain(split_tokens, "\xe2\x96\x81,");
        std::vector<CjkSegmentRange> sub_segments;
        bool emitted_sub_segments = false;

        if (!split_has_comma &&
            (cjk_range_contains_piece(pieces, current_start, i + 1, ",") ||
             cjk_range_contains_piece(pieces, current_start, i + 1, "\xe2\x96\x81,"))) {
            sub_segments = split_cjk_ranges_by_token(
                pieces,
                current_start,
                i + 1,
                {",", "\xe2\x96\x81,"},
                max_tokens,
                quick_streaming_tokens);
            emitted_sub_segments = true;
        } else if (!cjk_split_tokens_contain(split_tokens, "-") &&
                   cjk_range_contains_piece(pieces, current_start, i + 1, "-")) {
            sub_segments = split_cjk_ranges_by_token(
                pieces,
                current_start,
                i + 1,
                {"-"},
                max_tokens,
                quick_streaming_tokens);
            emitted_sub_segments = true;
        } else if (current_len <= max_tokens) {
            if (cjk_split_tokens_contain(split_tokens, pieces[i]) && current_len > 2) {
                size_t segment_end = i + 1;
                if (segment_end < end &&
                    (pieces[segment_end] == "'" || pieces[segment_end] == "\xe2\x96\x81'")) {
                    ++segment_end;
                    i = segment_end - 1;
                }
                segments.push_back(CjkSegmentRange{current_start, segment_end});
                current_start = segment_end;
                current_len = 0;
            }
            continue;
        } else {
            append_cjk_token_chunks(sub_segments, current_start, i + 1, max_tokens);
            emitted_sub_segments = true;
        }

        if (emitted_sub_segments) {
            segments.insert(segments.end(), sub_segments.begin(), sub_segments.end());
            current_start = i + 1;
            current_len = 0;
        }
    }

    if (current_len > 0) {
        segments.push_back(CjkSegmentRange{current_start, end});
    }

    std::vector<CjkSegmentRange> merged;
    size_t total_tokens = 0;
    for (const auto& segment : segments) {
        const size_t segment_len = segment.size();
        total_tokens += segment_len;
        if (segment_len == 0) {
            continue;
        }
        if (merged.empty()) {
            merged.push_back(segment);
            continue;
        }

        auto& previous = merged.back();
        const size_t merged_len = previous.size() + segment_len;
        if (previous.end == segment.start &&
            merged_len <= max_tokens &&
            total_tokens > quick_streaming_tokens) {
            previous.end = segment.end;
        } else if (previous.end == segment.start && merged_len * 2 <= max_tokens) {
            previous.end = segment.end;
        } else {
            merged.push_back(segment);
        }
    }
    return merged;
}

std::vector<CjkTokenizedSegment> split_cjk_tokenized_text(const CjkTokenizedText& tokenized, uint32_t max_tokens) {
    if (max_tokens == 0) {
        throw std::runtime_error("native CJK segment split max_tokens must be positive");
    }
    static const std::vector<std::string> punctuation_marks_tokens{
        ".",
        "!",
        "?",
        "\xe2\x96\x81.",
        "\xe2\x96\x81?",
        "\xe2\x96\x81...",
    };
    const auto ranges = split_cjk_ranges_by_token(
        tokenized.pieces,
        0,
        tokenized.ids.size(),
        punctuation_marks_tokens,
        static_cast<size_t>(max_tokens));

    std::vector<CjkTokenizedSegment> segments;
    segments.reserve(ranges.size());
    for (const auto& range : ranges) {
        CjkTokenizedSegment segment;
        segment.pieces.assign(tokenized.pieces.begin() + static_cast<std::ptrdiff_t>(range.start),
                              tokenized.pieces.begin() + static_cast<std::ptrdiff_t>(range.end));
        segment.ids.assign(tokenized.ids.begin() + static_cast<std::ptrdiff_t>(range.start),
                           tokenized.ids.begin() + static_cast<std::ptrdiff_t>(range.end));
        segments.push_back(std::move(segment));
    }
    if (segments.empty()) {
        throw std::runtime_error("native CJK segment split produced no segments");
    }
    return segments;
}

bool run_export_text_ids_cjk_segments(const std::string& tokenizer_dir,
                                      const std::string& text,
                                      uint32_t max_tokens,
                                      const std::string& output_dir) {
    if (max_tokens == 0) {
        throw std::runtime_error("native CJK segment export max_tokens must be positive");
    }
    const auto piece_to_id = load_tokenizer_pieces(tokenizer_dir);
    const auto tokenized = tokenize_cjk_text(piece_to_id, text);
    std::filesystem::create_directories(output_dir);

    struct OutputSegment {
        CjkTokenizedSegment tokenized;
        std::string output;
    };
    const auto tokenized_segments = split_cjk_tokenized_text(tokenized, max_tokens);
    std::vector<OutputSegment> segments;
    segments.reserve(tokenized_segments.size());
    for (const auto& tokenized_segment : tokenized_segments) {
        OutputSegment segment;
        segment.tokenized = tokenized_segment;
        char name[32];
        std::snprintf(name, sizeof(name), "segment_%03zu.u32", segments.size());
        segment.output = (std::filesystem::path(output_dir) / name).string();
        write_raw_u32(segment.output, segment.tokenized.ids);
        segments.push_back(std::move(segment));
    }
    if (segments.empty()) {
        throw std::runtime_error("native CJK segment export produced no segments");
    }

    std::cout << "{\n";
    std::cout << "  \"format\": \"mit2-text-ids-cjk-segments\",\n";
    std::cout << "  \"version\": 1,\n";
    std::cout << "  \"tokenizer_dir\": \"" << json_escape(tokenizer_dir) << "\",\n";
    std::cout << "  \"text\": \"" << json_escape(text) << "\",\n";
    std::cout << "  \"output_dir\": \"" << json_escape(output_dir) << "\",\n";
    std::cout << "  \"max_text_tokens_per_segment\": " << max_tokens << ",\n";
    std::cout << "  \"tokens\": ";
    print_json_string_array(tokenized.pieces);
    std::cout << ",\n";
    std::cout << "  \"token_ids\": ";
    print_json_u32_array(tokenized.ids);
    std::cout << ",\n";
    std::cout << "  \"segments\": [\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        std::cout << "    {\n";
        std::cout << "      \"index\": " << i << ",\n";
        std::cout << "      \"output\": \"" << json_escape(segment.output) << "\",\n";
        std::cout << "      \"pieces\": ";
        print_json_string_array(segment.tokenized.pieces);
        std::cout << ",\n";
        std::cout << "      \"ids\": ";
        print_json_u32_array(segment.tokenized.ids);
        std::cout << "\n";
        std::cout << "    }";
        if (i + 1 < segments.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
    return true;
}

bool run_tokenize_cjk_smoke(const std::string& tokenizer_dir, const std::string& text, const std::string& output_text_ids_path) {
    return run_export_text_ids_cjk(tokenizer_dir, text, output_text_ids_path, "mit2-tokenize-cjk-smoke");
}

uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

float splitmix64_uniform_open01(uint64_t& state) {
    constexpr double scale = 1.0 / static_cast<double>(1ull << 53);
    const uint64_t bits = splitmix64_next(state) >> 11;
    double value = (static_cast<double>(bits) + 0.5) * scale;
    if (value <= 0.0) {
        value = scale;
    } else if (value >= 1.0) {
        value = 1.0 - scale;
    }
    return static_cast<float>(value);
}

std::vector<float> make_deterministic_normal_noise(uint32_t tokens, uint32_t channels, uint64_t seed, float temperature) {
    if (tokens == 0 || channels == 0) {
        throw std::runtime_error("noise shape must be non-empty");
    }
    constexpr float two_pi = 6.2831853071795864769f;
    std::vector<float> noise(static_cast<size_t>(tokens) * channels);
    uint64_t state = seed;
    size_t i = 0;
    while (i < noise.size()) {
        const float u1 = splitmix64_uniform_open01(state);
        const float u2 = splitmix64_uniform_open01(state);
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float angle = two_pi * u2;
        noise[i++] = radius * std::cos(angle) * temperature;
        if (i < noise.size()) {
            noise[i++] = radius * std::sin(angle) * temperature;
        }
    }
    return noise;
}

void write_u16_le(std::ofstream& out, uint16_t value) {
    char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void write_u32_le(std::ofstream& out, uint32_t value) {
    char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void write_wav_pcm16(const std::string& path, const std::vector<float>& samples, uint32_t sample_rate) {
    constexpr uint16_t channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    constexpr uint16_t bytes_per_sample = bits_per_sample / 8;
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * bytes_per_sample);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open wav output file: " + path);
    }
    out.write("RIFF", 4);
    write_u32_le(out, 36u + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32_le(out, 16);
    write_u16_le(out, 1);
    write_u16_le(out, channels);
    write_u32_le(out, sample_rate);
    write_u32_le(out, sample_rate * channels * bytes_per_sample);
    write_u16_le(out, channels * bytes_per_sample);
    write_u16_le(out, bits_per_sample);
    out.write("data", 4);
    write_u32_le(out, data_bytes);
    for (float sample : samples) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        const int32_t quantized = static_cast<int32_t>(std::round(clipped * 32767.0f));
        write_u16_le(out, static_cast<uint16_t>(static_cast<int16_t>(quantized)));
    }
    if (!out) {
        throw std::runtime_error("failed to write wav output file: " + path);
    }
}

uint16_t read_u16_le_at(const std::vector<char>& bytes, size_t offset) {
    if (offset + 2 > bytes.size()) {
        throw std::runtime_error("truncated u16");
    }
    return static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset])) |
           static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset + 1]) << 8);
}

uint32_t read_u32_le_at(const std::vector<char>& bytes, size_t offset) {
    if (offset + 4 > bytes.size()) {
        throw std::runtime_error("truncated u32");
    }
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3])) << 24);
}

struct Pcm16MonoWav {
    uint32_t sample_rate = 0;
    std::vector<char> frames;
};

struct CloneAudioQuality {
    uint64_t samples = 0;
    uint64_t nonzero_samples = 0;
    uint64_t near_silence_samples = 0;
    uint64_t clipped_samples = 0;
    int32_t peak_abs = 0;
    double peak = 0.0;
    double rms = 0.0;
    double mean = 0.0;
    double nonzero_ratio = 0.0;
    double near_silence_ratio = 1.0;
    double clipping_ratio = 0.0;
    double duration_seconds = 0.0;
    std::vector<std::string> issues;
};

int16_t pcm16_mono_sample_at(const std::vector<char>& frames, size_t sample_index) {
    const size_t offset = sample_index * 2u;
    const uint16_t raw = static_cast<uint16_t>(static_cast<unsigned char>(frames[offset])) |
                         static_cast<uint16_t>(static_cast<unsigned char>(frames[offset + 1]) << 8);
    return static_cast<int16_t>(raw);
}

std::vector<float> pcm16_mono_wav_to_f32(const Pcm16MonoWav& wav) {
    std::vector<float> out(wav.frames.size() / 2u);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>(pcm16_mono_sample_at(wav.frames, i)) / 32768.0f;
    }
    return out;
}

CloneAudioQuality analyze_clone_audio_quality(const Pcm16MonoWav& wav) {
    CloneAudioQuality quality;
    quality.samples = wav.frames.size() / 2u;
    quality.duration_seconds = wav.sample_rate > 0
        ? static_cast<double>(quality.samples) / static_cast<double>(wav.sample_rate)
        : 0.0;
    long double sum_squares = 0.0L;
    long double sum = 0.0L;
    for (size_t i = 0; i < quality.samples; ++i) {
        const int16_t sample = pcm16_mono_sample_at(wav.frames, i);
        const int32_t abs_sample = std::abs(static_cast<int32_t>(sample));
        quality.peak_abs = std::max(quality.peak_abs, abs_sample);
        if (abs_sample > 0) {
            ++quality.nonzero_samples;
        }
        if (abs_sample <= 32) {
            ++quality.near_silence_samples;
        }
        if (abs_sample >= 32760) {
            ++quality.clipped_samples;
        }
        const long double normalized = static_cast<long double>(sample) / 32768.0L;
        sum += normalized;
        sum_squares += normalized * normalized;
    }
    quality.rms = quality.samples > 0 ? std::sqrt(static_cast<double>(sum_squares / quality.samples)) : 0.0;
    quality.mean = quality.samples > 0 ? static_cast<double>(sum / quality.samples) : 0.0;
    quality.peak = static_cast<double>(quality.peak_abs) / 32768.0;
    quality.nonzero_ratio = quality.samples > 0
        ? static_cast<double>(quality.nonzero_samples) / static_cast<double>(quality.samples)
        : 0.0;
    quality.near_silence_ratio = quality.samples > 0
        ? static_cast<double>(quality.near_silence_samples) / static_cast<double>(quality.samples)
        : 1.0;
    quality.clipping_ratio = quality.samples > 0
        ? static_cast<double>(quality.clipped_samples) / static_cast<double>(quality.samples)
        : 0.0;
    if (quality.samples == 0) {
        quality.issues.push_back("empty_audio");
    }
    if (quality.duration_seconds < 0.05) {
        quality.issues.push_back("too_short_for_clone_reference");
    }
    if (quality.peak < 0.001 || quality.rms < 0.0001 || quality.near_silence_ratio > 0.995) {
        quality.issues.push_back("silent_or_near_silent_audio");
    }
    if (quality.clipping_ratio > 0.5) {
        quality.issues.push_back("mostly_clipped_audio");
    }
    return quality;
}

std::vector<float> resample_linear_f32(const std::vector<float>& input, uint32_t source_rate, uint32_t target_rate) {
    if (input.empty() || source_rate == 0 || target_rate == 0) {
        return {};
    }
    if (source_rate == target_rate) {
        return input;
    }
    const double ratio = static_cast<double>(target_rate) / static_cast<double>(source_rate);
    const size_t out_count = std::max<size_t>(1, static_cast<size_t>(std::llround(static_cast<double>(input.size()) * ratio)));
    std::vector<float> out(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        const double src_pos = static_cast<double>(i) * static_cast<double>(source_rate) / static_cast<double>(target_rate);
        const size_t left = std::min(static_cast<size_t>(std::floor(src_pos)), input.size() - 1);
        const size_t right = std::min(left + 1, input.size() - 1);
        const float frac = static_cast<float>(src_pos - static_cast<double>(left));
        out[i] = input[left] * (1.0f - frac) + input[right] * frac;
    }
    return out;
}

int reflect_index(int index, int size) {
    if (size <= 1) {
        return 0;
    }
    const int period = 2 * size - 2;
    int x = index % period;
    if (x < 0) {
        x += period;
    }
    return x < size ? x : period - x;
}

std::vector<float> hann_window(uint32_t win_length) {
    std::vector<float> window(win_length);
    if (win_length <= 1) {
        std::fill(window.begin(), window.end(), 1.0f);
        return window;
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (uint32_t i = 0; i < win_length; ++i) {
        window[i] = static_cast<float>(0.5 - 0.5 * std::cos((2.0 * pi * i) / static_cast<double>(win_length - 1)));
    }
    return window;
}

double hz_to_mel_slaney(double hz) {
    constexpr double f_sp = 200.0 / 3.0;
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = 15.0;
    const double logstep = std::log(6.4) / 27.0;
    if (hz < min_log_hz) {
        return hz / f_sp;
    }
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

double mel_to_hz_slaney(double mel) {
    constexpr double f_sp = 200.0 / 3.0;
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = 15.0;
    const double logstep = std::log(6.4) / 27.0;
    if (mel < min_log_mel) {
        return mel * f_sp;
    }
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

std::vector<float> build_mel_filterbank_slaney(uint32_t sample_rate,
                                               uint32_t n_fft,
                                               uint32_t n_mels,
                                               double f_min,
                                               double f_max) {
    const uint32_t bins = n_fft / 2 + 1;
    std::vector<float> filters(static_cast<size_t>(n_mels) * bins, 0.0f);
    const double mel_min = hz_to_mel_slaney(f_min);
    const double mel_max = hz_to_mel_slaney(f_max);
    std::vector<double> hz_points(n_mels + 2);
    for (uint32_t i = 0; i < n_mels + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) * static_cast<double>(i) / static_cast<double>(n_mels + 1);
        hz_points[i] = mel_to_hz_slaney(mel);
    }
    std::vector<double> fft_freqs(bins);
    for (uint32_t k = 0; k < bins; ++k) {
        fft_freqs[k] = static_cast<double>(sample_rate) * static_cast<double>(k) / static_cast<double>(n_fft);
    }
    for (uint32_t m = 0; m < n_mels; ++m) {
        const double left = hz_points[m];
        const double center = hz_points[m + 1];
        const double right = hz_points[m + 2];
        const double enorm = 2.0 / (right - left);
        for (uint32_t k = 0; k < bins; ++k) {
            const double freq = fft_freqs[k];
            double weight = 0.0;
            if (freq >= left && freq <= center && center > left) {
                weight = (freq - left) / (center - left);
            } else if (freq >= center && freq <= right && right > center) {
                weight = (right - freq) / (right - center);
            }
            filters[static_cast<size_t>(m) * bins + k] = static_cast<float>(std::max(0.0, weight) * enorm);
        }
    }
    return filters;
}

std::vector<float> extract_clone_mel_22k_f32(const std::vector<float>& audio_22k, uint32_t& frames_out) {
    constexpr uint32_t sample_rate = 22050;
    constexpr uint32_t n_fft = 1024;
    constexpr uint32_t win_length = 1024;
    constexpr uint32_t hop_length = 256;
    constexpr uint32_t n_mels = 80;
    constexpr uint32_t bins = n_fft / 2 + 1;
    if (audio_22k.size() < hop_length) {
        throw std::runtime_error("preprocessed audio is too short for clone mel extraction");
    }
    const uint32_t frames = static_cast<uint32_t>(audio_22k.size() / hop_length);
    frames_out = frames;
    const auto window = hann_window(win_length);
    const auto mel_filter = build_mel_filterbank_slaney(sample_rate, n_fft, n_mels, 0.0, sample_rate / 2.0);
    std::vector<float> mel(static_cast<size_t>(n_mels) * frames, 0.0f);
    std::vector<double> magnitude(bins, 0.0);
    constexpr double pi = 3.141592653589793238462643383279502884;
    const int pad = static_cast<int>((n_fft - hop_length) / 2);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        std::fill(magnitude.begin(), magnitude.end(), 0.0);
        const int frame_start = static_cast<int>(frame * hop_length) - pad;
        for (uint32_t k = 0; k < bins; ++k) {
            double real = 0.0;
            double imag = 0.0;
            for (uint32_t n = 0; n < n_fft; ++n) {
                const int src = reflect_index(frame_start + static_cast<int>(n), static_cast<int>(audio_22k.size()));
                const float sample = audio_22k[static_cast<size_t>(src)] * window[static_cast<size_t>(n)];
                const double angle = -2.0 * pi * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(n_fft);
                real += static_cast<double>(sample) * std::cos(angle);
                imag += static_cast<double>(sample) * std::sin(angle);
            }
            magnitude[k] = std::sqrt(real * real + imag * imag + 1e-9);
        }
        for (uint32_t m = 0; m < n_mels; ++m) {
            double energy = 0.0;
            for (uint32_t k = 0; k < bins; ++k) {
                energy += static_cast<double>(mel_filter[static_cast<size_t>(m) * bins + k]) * magnitude[k];
            }
            mel[static_cast<size_t>(m) * frames + frame] = static_cast<float>(std::log(std::max(energy, 1e-5)));
        }
    }
    return mel;
}

std::vector<float> povey_window(uint32_t win_length) {
    std::vector<float> window(win_length);
    if (win_length <= 1) {
        std::fill(window.begin(), window.end(), 1.0f);
        return window;
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (uint32_t i = 0; i < win_length; ++i) {
        const double hann = 0.5 - 0.5 * std::cos((2.0 * pi * i) / static_cast<double>(win_length - 1));
        window[i] = static_cast<float>(std::pow(hann, 0.85));
    }
    return window;
}

std::vector<float> extract_clone_fbank_16k_f32(const std::vector<float>& audio_16k, uint32_t& frames_out) {
    constexpr uint32_t sample_rate = 16000;
    constexpr uint32_t frame_length = 400;  // 25 ms
    constexpr uint32_t frame_shift = 160;   // 10 ms
    constexpr uint32_t n_fft = 512;
    constexpr uint32_t n_mels = 80;
    constexpr uint32_t bins = n_fft / 2 + 1;
    if (audio_16k.size() < frame_length) {
        throw std::runtime_error("preprocessed audio is too short for clone fbank extraction");
    }
    const uint32_t frames = 1u + static_cast<uint32_t>((audio_16k.size() - frame_length) / frame_shift);
    frames_out = frames;
    const auto window = povey_window(frame_length);
    const auto mel_filter = build_mel_filterbank_slaney(sample_rate, n_fft, n_mels, 20.0, sample_rate / 2.0);
    std::vector<float> fbank(static_cast<size_t>(frames) * n_mels, 0.0f);
    std::vector<double> power(bins, 0.0);
    std::array<double, n_mels> column_sum{};
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const size_t start = static_cast<size_t>(frame) * frame_shift;
        double mean = 0.0;
        for (uint32_t n = 0; n < frame_length; ++n) {
            mean += audio_16k[start + n];
        }
        mean /= static_cast<double>(frame_length);
        std::fill(power.begin(), power.end(), 0.0);
        for (uint32_t k = 0; k < bins; ++k) {
            double real = 0.0;
            double imag = 0.0;
            for (uint32_t n = 0; n < n_fft; ++n) {
                float sample = 0.0f;
                if (n < frame_length) {
                    const float centered = audio_16k[start + n] - static_cast<float>(mean);
                    const float previous = n == 0 ? 0.0f : audio_16k[start + n - 1] - static_cast<float>(mean);
                    const float emphasized = centered - 0.97f * previous;
                    sample = emphasized * window[n];
                }
                const double angle = -2.0 * pi * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(n_fft);
                real += static_cast<double>(sample) * std::cos(angle);
                imag += static_cast<double>(sample) * std::sin(angle);
            }
            power[k] = real * real + imag * imag;
        }
        for (uint32_t m = 0; m < n_mels; ++m) {
            double energy = 0.0;
            for (uint32_t k = 0; k < bins; ++k) {
                energy += static_cast<double>(mel_filter[static_cast<size_t>(m) * bins + k]) * power[k];
            }
            const float value = static_cast<float>(std::log(std::max(energy, 1e-10)));
            fbank[static_cast<size_t>(frame) * n_mels + m] = value;
            column_sum[m] += value;
        }
    }
    for (uint32_t m = 0; m < n_mels; ++m) {
        const float mean = static_cast<float>(column_sum[m] / static_cast<double>(frames));
        for (uint32_t frame = 0; frame < frames; ++frame) {
            fbank[static_cast<size_t>(frame) * n_mels + m] -= mean;
        }
    }
    return fbank;
}

// Iterative Cooley-Tukey radix-2 FFT (in-place, complex, size must be power of 2)
static void fft_inplace_double(std::vector<double>& re, std::vector<double>& im) {
    const uint32_t N = static_cast<uint32_t>(re.size());
    // Bit-reversal permutation
    for (uint32_t i = 1, j = 0; i < N; ++i) {
        uint32_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (uint32_t len = 2; len <= N; len <<= 1) {
        const double ang = -2.0 * pi / static_cast<double>(len);
        const double wre = std::cos(ang);
        const double wim = std::sin(ang);
        for (uint32_t i = 0; i < N; i += len) {
            double ur = 1.0, ui = 0.0;
            for (uint32_t k = 0; k < len / 2; ++k) {
                const double xr = re[i + k + len/2], xi = im[i + k + len/2];
                const double vr = xr * ur - xi * ui;
                const double vi = xr * ui + xi * ur;
                re[i + k + len/2] = re[i + k] - vr;
                im[i + k + len/2] = im[i + k] - vi;
                re[i + k] += vr;
                im[i + k] += vi;
                const double nwr = ur * wre - ui * wim;
                ui = ur * wim + ui * wre;
                ur = nwr;
            }
        }
    }
}

// W2V-BERT mel filter bank: kaldi mel scale, triangularize in mel space, no normalization
// Output: flat array [bins * n_mels] indexed as [k * n_mels + m]
static std::vector<float> build_mel_filterbank_w2v_bert() {
    constexpr uint32_t n_fft = 512;
    constexpr uint32_t n_mels = 80;
    constexpr uint32_t sr = 16000;
    constexpr uint32_t bins = n_fft / 2 + 1;  // 257
    constexpr double f_min = 20.0;
    constexpr double f_max = 8000.0;
    auto hz2mel = [](double hz) { return 1127.0 * std::log(1.0 + hz / 700.0); };
    const double mel_min = hz2mel(f_min);
    const double mel_max = hz2mel(f_max);
    std::vector<double> mel_pts(n_mels + 2);
    for (uint32_t i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * static_cast<double>(i) / static_cast<double>(n_mels + 1);
    }
    // FFT bin frequencies converted to mel space (fft_bin_width = sr / ((bins-1)*2))
    const double fft_bin_hz_width = static_cast<double>(sr) / static_cast<double>((bins - 1) * 2);
    std::vector<double> fft_freqs_mel(bins);
    for (uint32_t k = 0; k < bins; ++k) {
        fft_freqs_mel[k] = hz2mel(static_cast<double>(k) * fft_bin_hz_width);
    }
    std::vector<float> filters(static_cast<size_t>(bins) * n_mels, 0.0f);
    for (uint32_t m = 0; m < n_mels; ++m) {
        const double left = mel_pts[m];
        const double center = mel_pts[m + 1];
        const double right = mel_pts[m + 2];
        for (uint32_t k = 0; k < bins; ++k) {
            const double mel_f = fft_freqs_mel[k];
            double weight = 0.0;
            if (mel_f >= left && mel_f <= center && center > left)
                weight = (mel_f - left) / (center - left);
            else if (mel_f >= center && mel_f <= right && right > center)
                weight = (right - mel_f) / (right - center);
            filters[static_cast<size_t>(k) * n_mels + m] = static_cast<float>(std::max(0.0, weight));
        }
    }
    return filters;
}

// SeamlessM4T-compatible feature extraction for W2V-BERT input.
// Produces stacked 160-dim log-mel features from 16kHz audio.
// out_tokens: number of stacked frame pairs (= num_mel_frames / 2)
std::vector<float> extract_w2v_bert_features_16k_f32(const std::vector<float>& audio_16k, uint32_t& out_tokens) {
    constexpr uint32_t frame_length = 400;
    constexpr uint32_t hop_length = 160;
    constexpr uint32_t n_fft = 512;
    constexpr uint32_t bins = n_fft / 2 + 1;
    constexpr uint32_t n_mels = 80;
    constexpr double preemphasis_coef = 0.97;
    constexpr double scale_factor = 32768.0;  // 2^15 Kaldi compliance
    constexpr double mel_floor = 1.192092955078125e-07;  // 2^-23

    if (audio_16k.size() < frame_length) {
        throw std::runtime_error("audio too short for w2v_bert feature extraction");
    }

    // Povey window (symmetric Hann ^ 0.85), matches SeamlessM4T window_function(400, "povey", periodic=False)
    static const std::vector<float> window_f32 = povey_window(frame_length);
    static const std::vector<float> mel_filters = build_mel_filterbank_w2v_bert();

    const uint32_t num_frames = 1u + static_cast<uint32_t>((audio_16k.size() - frame_length) / hop_length);
    const uint32_t num_stacked = num_frames / 2;
    out_tokens = num_stacked;

    // Pass 1: compute raw log-mel [num_frames, n_mels] before normalization
    std::vector<float> log_mel(static_cast<size_t>(num_frames) * n_mels, 0.0f);
    std::vector<double> buf_re(n_fft, 0.0), buf_im(n_fft, 0.0);

    for (uint32_t frame = 0; frame < num_frames; ++frame) {
        const size_t start = static_cast<size_t>(frame) * hop_length;

        for (uint32_t i = 0; i < frame_length; ++i) {
            buf_re[i] = static_cast<double>(audio_16k[start + i]) * scale_factor;
        }
        for (uint32_t i = frame_length; i < n_fft; ++i) buf_re[i] = 0.0;
        std::fill(buf_im.begin(), buf_im.end(), 0.0);

        // Remove DC offset
        double dc = 0.0;
        for (uint32_t i = 0; i < frame_length; ++i) dc += buf_re[i];
        dc /= static_cast<double>(frame_length);
        for (uint32_t i = 0; i < frame_length; ++i) buf_re[i] -= dc;

        // Pre-emphasis (right-to-left to use original values)
        for (int i = static_cast<int>(frame_length) - 1; i >= 1; --i) {
            buf_re[i] -= preemphasis_coef * buf_re[i - 1];
        }
        buf_re[0] *= (1.0 - preemphasis_coef);

        // Apply Povey window
        for (uint32_t i = 0; i < frame_length; ++i) buf_re[i] *= static_cast<double>(window_f32[i]);

        // FFT
        fft_inplace_double(buf_re, buf_im);

        // Power spectrum → mel filterbank → log
        for (uint32_t m = 0; m < n_mels; ++m) {
            double energy = 0.0;
            for (uint32_t k = 0; k < bins; ++k) {
                const double power = buf_re[k] * buf_re[k] + buf_im[k] * buf_im[k];
                energy += static_cast<double>(mel_filters[static_cast<size_t>(k) * n_mels + m]) * power;
            }
            log_mel[static_cast<size_t>(frame) * n_mels + m] = static_cast<float>(std::log(std::max(energy, mel_floor)));
        }
    }

    // Pass 2: per-mel-bin zero-mean unit-variance normalization (matches SeamlessM4TFeatureExtractor)
    // Formula: (x - mean(col)) / sqrt(var(col, ddof=1) + 1e-7)
    for (uint32_t m = 0; m < n_mels; ++m) {
        double sum = 0.0;
        for (uint32_t f = 0; f < num_frames; ++f) sum += static_cast<double>(log_mel[static_cast<size_t>(f) * n_mels + m]);
        const double mean = sum / static_cast<double>(num_frames);
        double sq_sum = 0.0;
        for (uint32_t f = 0; f < num_frames; ++f) {
            const double d = static_cast<double>(log_mel[static_cast<size_t>(f) * n_mels + m]) - mean;
            sq_sum += d * d;
        }
        const double var = (num_frames > 1) ? sq_sum / static_cast<double>(num_frames - 1) : 0.0;
        const double inv_std = 1.0 / std::sqrt(var + 1e-7);
        for (uint32_t f = 0; f < num_frames; ++f) {
            log_mel[static_cast<size_t>(f) * n_mels + m] = static_cast<float>((static_cast<double>(log_mel[static_cast<size_t>(f) * n_mels + m]) - mean) * inv_std);
        }
    }

    // Pass 3: stack consecutive frame pairs → [num_stacked, 160]
    // frame 2i and 2i+1 → stacked row i: [mel(2i), mel(2i+1)]
    std::vector<float> features(static_cast<size_t>(num_stacked) * 160, 0.0f);
    for (uint32_t i = 0; i < num_stacked; ++i) {
        const size_t dst = static_cast<size_t>(i) * 160;
        const size_t src0 = static_cast<size_t>(2 * i) * n_mels;
        const size_t src1 = static_cast<size_t>(2 * i + 1) * n_mels;
        for (uint32_t m = 0; m < n_mels; ++m) {
            features[dst + m]           = log_mel[src0 + m];
            features[dst + n_mels + m]  = log_mel[src1 + m];
        }
    }

    return features;
}

struct FloatVectorStats {
    float min = 0.0f;
    float max = 0.0f;
    double mean = 0.0;
};

FloatVectorStats compute_float_vector_stats(const std::vector<float>& values) {
    FloatVectorStats stats;
    if (values.empty()) {
        return stats;
    }
    stats.min = values[0];
    stats.max = values[0];
    long double sum = 0.0L;
    for (float value : values) {
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
        sum += value;
    }
    stats.mean = static_cast<double>(sum / static_cast<long double>(values.size()));
    return stats;
}

size_t find_json_field_colon(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    const size_t key_pos = text.find(quoted);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("manifest missing field: " + key);
    }
    const size_t colon = text.find(':', key_pos + quoted.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("manifest field has no value: " + key);
    }
    return colon;
}

std::string parse_json_string_field(const std::string& text, const std::string& key) {
    size_t pos = find_json_field_colon(text, key) + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (pos >= text.size() || text[pos] != '"') {
        throw std::runtime_error("manifest field is not a string: " + key);
    }
    ++pos;
    std::string out;
    while (pos < text.size()) {
        const char ch = text[pos++];
        if (ch == '"') {
            return out;
        }
        if (ch == '\\') {
            if (pos >= text.size()) {
                throw std::runtime_error("manifest string has truncated escape: " + key);
            }
            const char esc = text[pos++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default:
                    throw std::runtime_error("manifest string has unsupported escape: " + key);
            }
        } else {
            out.push_back(ch);
        }
    }
    throw std::runtime_error("manifest string is unterminated: " + key);
}

uint64_t parse_json_u64_field(const std::string& text, const std::string& key) {
    size_t pos = find_json_field_colon(text, key) + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    const size_t start = pos;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (pos == start) {
        throw std::runtime_error("manifest field is not an unsigned integer: " + key);
    }
    return std::stoull(text.substr(start, pos - start));
}

bool parse_json_bool_field(const std::string& text, const std::string& key) {
    size_t pos = find_json_field_colon(text, key) + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (text.compare(pos, 4, "true") == 0) {
        return true;
    }
    if (text.compare(pos, 5, "false") == 0) {
        return false;
    }
    throw std::runtime_error("manifest field is not a boolean: " + key);
}

struct ClonePreprocessManifest {
    std::string format;
    uint64_t version = 0;
    std::string source_audio_wav;
    std::string source_audio_sha256;
    std::string output_f32;
    std::string output_sha256;
    std::string audio_format;
    uint64_t source_sample_rate = 0;
    uint64_t target_sample_rate = 0;
    uint64_t source_samples = 0;
    uint64_t preprocessed_samples = 0;
    bool ready_native_clone_audio_preprocess = false;
    bool ready_native_voice_clone = false;
};

ClonePreprocessManifest parse_clone_preprocess_manifest(const std::string& path) {
    const std::string text = read_text_file(path);
    ClonePreprocessManifest manifest;
    manifest.format = parse_json_string_field(text, "format");
    manifest.version = parse_json_u64_field(text, "version");
    manifest.source_audio_wav = parse_json_string_field(text, "source_audio_wav");
    manifest.source_audio_sha256 = parse_json_string_field(text, "source_audio_sha256");
    manifest.output_f32 = parse_json_string_field(text, "output_f32");
    manifest.output_sha256 = parse_json_string_field(text, "output_sha256");
    manifest.audio_format = parse_json_string_field(text, "audio_format");
    manifest.source_sample_rate = parse_json_u64_field(text, "source_sample_rate");
    manifest.target_sample_rate = parse_json_u64_field(text, "target_sample_rate");
    manifest.source_samples = parse_json_u64_field(text, "source_samples");
    manifest.preprocessed_samples = parse_json_u64_field(text, "preprocessed_samples");
    manifest.ready_native_clone_audio_preprocess = parse_json_bool_field(text, "ready_native_clone_audio_preprocess");
    manifest.ready_native_voice_clone = parse_json_bool_field(text, "ready_native_voice_clone");
    return manifest;
}

struct CloneFeatureManifest {
    std::string format;
    uint64_t version = 0;
    std::string source_audio_wav;
    std::string source_audio_sha256;
    std::string output_dir;
    std::string preprocess_manifest;
    std::string preprocessed_output_f32;
    std::string preprocessed_output_sha256;
    std::string mel_manifest;
    std::string output_mel_f32;
    std::string output_mel_sha256;
    std::string fbank_manifest;
    std::string output_fbank_f32;
    std::string output_fbank_sha256;
    uint64_t preprocessed_sample_rate = 0;
    uint64_t preprocessed_samples = 0;
    uint64_t mel_frames = 0;
    uint64_t fbank_frames = 0;
    bool ready_native_clone_audio_preprocess = false;
    bool ready_native_clone_mel_extraction = false;
    bool ready_native_clone_fbank_extraction = false;
    bool ready_native_clone_feature_prep = false;
    bool ready_native_voice_clone = false;
};

CloneFeatureManifest parse_clone_feature_manifest(const std::string& path) {
    const std::string text = read_text_file(path);
    CloneFeatureManifest manifest;
    manifest.format = parse_json_string_field(text, "format");
    manifest.version = parse_json_u64_field(text, "version");
    manifest.source_audio_wav = parse_json_string_field(text, "source_audio_wav");
    manifest.source_audio_sha256 = parse_json_string_field(text, "source_audio_sha256");
    manifest.output_dir = parse_json_string_field(text, "output_dir");
    manifest.preprocess_manifest = parse_json_string_field(text, "preprocess_manifest");
    manifest.preprocessed_output_f32 = parse_json_string_field(text, "preprocessed_output_f32");
    manifest.preprocessed_output_sha256 = parse_json_string_field(text, "preprocessed_output_sha256");
    manifest.mel_manifest = parse_json_string_field(text, "mel_manifest");
    manifest.output_mel_f32 = parse_json_string_field(text, "output_mel_f32");
    manifest.output_mel_sha256 = parse_json_string_field(text, "output_mel_sha256");
    manifest.fbank_manifest = parse_json_string_field(text, "fbank_manifest");
    manifest.output_fbank_f32 = parse_json_string_field(text, "output_fbank_f32");
    manifest.output_fbank_sha256 = parse_json_string_field(text, "output_fbank_sha256");
    manifest.preprocessed_sample_rate = parse_json_u64_field(text, "preprocessed_sample_rate");
    manifest.preprocessed_samples = parse_json_u64_field(text, "preprocessed_samples");
    manifest.mel_frames = parse_json_u64_field(text, "mel_frames");
    manifest.fbank_frames = parse_json_u64_field(text, "fbank_frames");
    manifest.ready_native_clone_audio_preprocess = parse_json_bool_field(text, "ready_native_clone_audio_preprocess");
    manifest.ready_native_clone_mel_extraction = parse_json_bool_field(text, "ready_native_clone_mel_extraction");
    manifest.ready_native_clone_fbank_extraction = parse_json_bool_field(text, "ready_native_clone_fbank_extraction");
    manifest.ready_native_clone_feature_prep = parse_json_bool_field(text, "ready_native_clone_feature_prep");
    manifest.ready_native_voice_clone = parse_json_bool_field(text, "ready_native_voice_clone");
    return manifest;
}

Pcm16MonoWav read_wav_pcm16_mono_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open wav input file: " + path);
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 44 || std::string(bytes.data(), bytes.data() + 4) != "RIFF" ||
        std::string(bytes.data() + 8, bytes.data() + 12) != "WAVE") {
        throw std::runtime_error("expected RIFF/WAVE file: " + path);
    }
    bool saw_fmt = false;
    bool saw_data = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> frames;
    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const std::string chunk_id(bytes.data() + pos, bytes.data() + pos + 4);
        const uint32_t chunk_size = read_u32_le_at(bytes, pos + 4);
        const size_t data_pos = pos + 8;
        if (data_pos + chunk_size > bytes.size()) {
            throw std::runtime_error("truncated wav chunk in: " + path);
        }
        if (chunk_id == "fmt ") {
            if (chunk_size < 16) {
                throw std::runtime_error("invalid wav fmt chunk in: " + path);
            }
            audio_format = read_u16_le_at(bytes, data_pos);
            channels = read_u16_le_at(bytes, data_pos + 2);
            sample_rate = read_u32_le_at(bytes, data_pos + 4);
            bits_per_sample = read_u16_le_at(bytes, data_pos + 14);
            saw_fmt = true;
        } else if (chunk_id == "data") {
            frames.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_pos),
                          bytes.begin() + static_cast<std::ptrdiff_t>(data_pos + chunk_size));
            saw_data = true;
        }
        pos = data_pos + chunk_size + (chunk_size & 1u);
    }
    if (!saw_fmt || !saw_data || audio_format != 1 || channels != 1 || bits_per_sample != 16 ||
        (frames.size() % 2) != 0) {
        throw std::runtime_error("expected PCM16 mono wav: " + path);
    }
    return Pcm16MonoWav{sample_rate, std::move(frames)};
}

void write_wav_pcm16_mono_bytes(const std::string& path, uint32_t sample_rate, const std::vector<char>& frames) {
    if ((frames.size() % 2) != 0) {
        throw std::runtime_error("PCM16 frame byte count must be even");
    }
    constexpr uint16_t channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    constexpr uint16_t bytes_per_sample = bits_per_sample / 8;
    const uint32_t data_bytes = static_cast<uint32_t>(frames.size());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open wav output file: " + path);
    }
    out.write("RIFF", 4);
    write_u32_le(out, 36u + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32_le(out, 16);
    write_u16_le(out, 1);
    write_u16_le(out, channels);
    write_u32_le(out, sample_rate);
    write_u32_le(out, sample_rate * channels * bytes_per_sample);
    write_u16_le(out, channels * bytes_per_sample);
    write_u16_le(out, bits_per_sample);
    out.write("data", 4);
    write_u32_le(out, data_bytes);
    out.write(frames.data(), static_cast<std::streamsize>(frames.size()));
    if (!out) {
        throw std::runtime_error("failed to write wav output file: " + path);
    }
}

std::vector<float> cpu_softmax(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    const float m = *std::max_element(x.begin(), x.end());
    float denom = 0.0f;
    for (float v : x) {
        denom += std::exp(v - m);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - m) / denom;
    }
    return out;
}

std::vector<float> cpu_embedding(const std::vector<float>& table, const std::vector<uint32_t>& ids, uint32_t width) {
    std::vector<float> out(ids.size() * width);
    for (size_t row = 0; row < ids.size(); ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            out[row * width + col] = table[ids[row] * width + col];
        }
    }
    return out;
}

std::vector<float> cpu_layernorm(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, float eps) {
    float mean = 0.0f;
    for (float v : x) {
        mean += v;
    }
    mean /= static_cast<float>(x.size());
    float var = 0.0f;
    for (float v : x) {
        const float d = v - mean;
        var += d * d;
    }
    var /= static_cast<float>(x.size());
    const float inv_std = 1.0f / std::sqrt(var + eps);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
    return out;
}

std::vector<float> cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& gamma, float eps) {
    float mean_sq = 0.0f;
    for (float v : x) {
        mean_sq += v * v;
    }
    mean_sq /= static_cast<float>(x.size());
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] * inv_rms * gamma[i];
    }
    return out;
}

std::vector<float> cpu_linear(const std::vector<float>& weight, const std::vector<float>& bias, const std::vector<float>& x, uint32_t rows, uint32_t cols) {
    std::vector<float> out(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        float acc = bias[row];
        for (uint32_t col = 0; col < cols; ++col) {
            acc += weight[row * cols + col] * x[col];
        }
        out[row] = acc;
    }
    return out;
}

std::vector<float> cpu_linear_rows(const std::vector<float>& weight, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t rows, uint32_t cols) {
    if (x.size() != static_cast<size_t>(tokens) * cols || weight.size() != static_cast<size_t>(rows) * cols || bias.size() != rows) {
        throw std::runtime_error("cpu_linear_rows shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(tokens) * rows);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t row = 0; row < rows; ++row) {
            float acc = bias[row];
            for (uint32_t col = 0; col < cols; ++col) {
                acc += weight[static_cast<size_t>(row) * cols + col] * x[static_cast<size_t>(t) * cols + col];
            }
            out[static_cast<size_t>(t) * rows + row] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_layer_norm_rows(const std::vector<float>& x,
                                       const std::vector<float>& gamma,
                                       const std::vector<float>& beta,
                                       uint32_t tokens,
                                       uint32_t dim,
                                       float eps = 1e-5f) {
    if (x.size() != static_cast<size_t>(tokens) * dim || gamma.size() != dim || beta.size() != dim) {
        throw std::runtime_error("cpu_layer_norm_rows shape mismatch");
    }
    std::vector<float> out(x.size());
    for (uint32_t t = 0; t < tokens; ++t) {
        const size_t base = static_cast<size_t>(t) * dim;
        float mean = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            mean += x[base + d];
        }
        mean /= static_cast<float>(dim);
        float var = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            const float centered = x[base + d] - mean;
            var += centered * centered;
        }
        var /= static_cast<float>(dim);
        const float inv_std = 1.0f / std::sqrt(var + eps);
        for (uint32_t d = 0; d < dim; ++d) {
            const size_t idx = base + d;
            out[idx] = (x[idx] - mean) * inv_std * gamma[d] + beta[d];
        }
    }
    return out;
}

std::vector<float> cpu_gelu_erf(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x[i];
        out[i] = 0.5f * v * (1.0f + std::erf(v * inv_sqrt2));
    }
    return out;
}

std::vector<float> cpu_silu(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
    return out;
}

std::vector<float> cpu_rmsnorm_rows(const std::vector<float>& x, const std::vector<float>& gamma, uint32_t tokens, uint32_t dim) {
    if (x.size() != static_cast<size_t>(tokens) * dim || gamma.size() != dim) {
        throw std::runtime_error("cpu_rmsnorm_rows shape mismatch");
    }
    std::vector<float> out(x.size());
    const float scale = std::sqrt(static_cast<float>(dim));
    for (uint32_t t = 0; t < tokens; ++t) {
        double sum_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            const float v = x[static_cast<size_t>(t) * dim + d];
            sum_sq += static_cast<double>(v) * v;
        }
        const float inv_norm = scale / std::sqrt(static_cast<float>(sum_sq) + 1e-12f);
        for (uint32_t d = 0; d < dim; ++d) {
            const size_t idx = static_cast<size_t>(t) * dim + d;
            out[idx] = x[idx] * inv_norm * gamma[d];
        }
    }
    return out;
}

std::vector<float> cpu_nearest_interpolate(const std::vector<float>& x, uint32_t in_tokens, uint32_t out_tokens, uint32_t width) {
    std::vector<float> out(static_cast<size_t>(out_tokens) * width);
    const float scale = static_cast<float>(in_tokens) / static_cast<float>(out_tokens);
    for (uint32_t out_t = 0; out_t < out_tokens; ++out_t) {
        uint32_t in_t = static_cast<uint32_t>(std::floor(static_cast<float>(out_t) * scale));
        if (in_t >= in_tokens) {
            in_t = in_tokens - 1;
        }
        for (uint32_t col = 0; col < width; ++col) {
            out[out_t * width + col] = x[in_t * width + col];
        }
    }
    return out;
}

std::vector<float> cpu_conv1d_same(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    std::vector<float> out(static_cast<size_t>(tokens) * out_channels);
    const int pad = static_cast<int>(kernel / 2);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t out_c = 0; out_c < out_channels; ++out_c) {
            float acc = bias[out_c];
            for (uint32_t in_c = 0; in_c < in_channels; ++in_c) {
                for (uint32_t k = 0; k < kernel; ++k) {
                    const int src_t = static_cast<int>(t) + static_cast<int>(k) - pad;
                    if (src_t >= 0 && src_t < static_cast<int>(tokens)) {
                        acc += x[static_cast<size_t>(src_t) * in_channels + in_c] *
                               weight[(static_cast<size_t>(out_c) * in_channels + in_c) * kernel + k];
                    }
                }
            }
            out[static_cast<size_t>(t) * out_channels + out_c] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_depthwise_conv1d_same(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t channels, uint32_t kernel) {
    std::vector<float> out(static_cast<size_t>(tokens) * channels);
    const int pad = static_cast<int>(kernel / 2);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < channels; ++c) {
            float acc = bias[c];
            for (uint32_t k = 0; k < kernel; ++k) {
                const int src_t = static_cast<int>(t) + static_cast<int>(k) - pad;
                if (src_t >= 0 && src_t < static_cast<int>(tokens)) {
                    acc += x[static_cast<size_t>(src_t) * channels + c] *
                           weight[static_cast<size_t>(c) * kernel + k];
                }
            }
            out[static_cast<size_t>(t) * channels + c] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_depthwise_conv1d_causal(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t channels, uint32_t kernel) {
    std::vector<float> out(static_cast<size_t>(tokens) * channels);
    const int left_pad = static_cast<int>(kernel) - 1;
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < channels; ++c) {
            float acc = bias[c];
            for (uint32_t k = 0; k < kernel; ++k) {
                const int src_t = static_cast<int>(t) + static_cast<int>(k) - left_pad;
                if (src_t >= 0 && src_t < static_cast<int>(tokens)) {
                    acc += x[static_cast<size_t>(src_t) * channels + c] *
                           weight[static_cast<size_t>(c) * kernel + k];
                }
            }
            out[static_cast<size_t>(t) * channels + c] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_subsampling_conv2d_relu_flat(const std::vector<float>& x,
                                                    const std::vector<float>& weight,
                                                    const std::vector<float>& bias,
                                                    uint32_t input_tokens,
                                                    uint32_t input_dim,
                                                    uint32_t out_channels,
                                                    uint32_t kernel,
                                                    uint32_t stride) {
    if (input_tokens < kernel || input_dim < kernel || out_channels == 0 || kernel == 0 || stride == 0 ||
        x.size() != static_cast<size_t>(input_tokens) * input_dim ||
        weight.size() != static_cast<size_t>(out_channels) * kernel * kernel ||
        bias.size() != out_channels) {
        throw std::runtime_error("cpu_subsampling_conv2d_relu_flat shape mismatch");
    }
    const uint32_t output_tokens = ((input_tokens - kernel) / stride) + 1;
    const uint32_t output_freq = ((input_dim - kernel) / stride) + 1;
    const uint32_t flat_dim = out_channels * output_freq;
    std::vector<float> out(static_cast<size_t>(output_tokens) * flat_dim);
    for (uint32_t t = 0; t < output_tokens; ++t) {
        const uint32_t src_t0 = t * stride;
        for (uint32_t c = 0; c < out_channels; ++c) {
            for (uint32_t f = 0; f < output_freq; ++f) {
                float acc = bias[c];
                const uint32_t src_f0 = f * stride;
                for (uint32_t kt = 0; kt < kernel; ++kt) {
                    for (uint32_t kf = 0; kf < kernel; ++kf) {
                        acc += x[static_cast<size_t>(src_t0 + kt) * input_dim + src_f0 + kf] *
                               weight[(static_cast<size_t>(c) * kernel + kt) * kernel + kf];
                    }
                }
                out[static_cast<size_t>(t) * flat_dim + c * output_freq + f] = std::max(acc, 0.0f);
            }
        }
    }
    return out;
}

std::vector<float> cpu_campplus_head_conv1_bn_relu(const std::vector<float>& fbank,
                                                   const std::vector<float>& conv_weight,
                                                   const std::vector<float>& bn_weight,
                                                   const std::vector<float>& bn_bias,
                                                   const std::vector<float>& bn_running_mean,
                                                   const std::vector<float>& bn_running_var,
                                                   uint32_t frames,
                                                   float eps = 1.0e-5f) {
    constexpr uint32_t feat_dim = 80;
    constexpr uint32_t out_channels = 32;
    constexpr uint32_t kernel = 3;
    if (frames == 0 ||
        fbank.size() != static_cast<size_t>(frames) * feat_dim ||
        conv_weight.size() != static_cast<size_t>(out_channels) * kernel * kernel ||
        bn_weight.size() != out_channels ||
        bn_bias.size() != out_channels ||
        bn_running_mean.size() != out_channels ||
        bn_running_var.size() != out_channels) {
        throw std::runtime_error("cpu_campplus_head_conv1_bn_relu shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(out_channels) * feat_dim * frames);
    for (uint32_t c = 0; c < out_channels; ++c) {
        const float scale = bn_weight[c] / std::sqrt(bn_running_var[c] + eps);
        const float shift = bn_bias[c] - bn_running_mean[c] * scale;
        for (uint32_t f = 0; f < feat_dim; ++f) {
            for (uint32_t t = 0; t < frames; ++t) {
                float acc = 0.0f;
                for (uint32_t kf = 0; kf < kernel; ++kf) {
                    const int src_f = static_cast<int>(f) + static_cast<int>(kf) - 1;
                    if (src_f < 0 || src_f >= static_cast<int>(feat_dim)) {
                        continue;
                    }
                    for (uint32_t kt = 0; kt < kernel; ++kt) {
                        const int src_t = static_cast<int>(t) + static_cast<int>(kt) - 1;
                        if (src_t < 0 || src_t >= static_cast<int>(frames)) {
                            continue;
                        }
                        const float input = fbank[static_cast<size_t>(src_t) * feat_dim + static_cast<uint32_t>(src_f)];
                        const float weight = conv_weight[(static_cast<size_t>(c) * kernel + kf) * kernel + kt];
                        acc += input * weight;
                    }
                }
                const float y = acc * scale + shift;
                out[(static_cast<size_t>(c) * feat_dim + f) * frames + t] = std::max(y, 0.0f);
            }
        }
    }
    return out;
}

struct NchwShape {
    uint32_t channels = 0;
    uint32_t height = 0;
    uint32_t width = 0;
};

std::vector<float> cpu_conv2d_nchw_no_bias(const std::vector<float>& x,
                                           NchwShape input_shape,
                                           const std::vector<float>& weight,
                                           uint32_t out_channels,
                                           uint32_t kernel_h,
                                           uint32_t kernel_w,
                                           uint32_t stride_h,
                                           uint32_t stride_w,
                                           uint32_t pad_h,
                                           uint32_t pad_w,
                                           NchwShape& output_shape) {
    if (input_shape.channels == 0 || input_shape.height == 0 || input_shape.width == 0 ||
        out_channels == 0 || kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 ||
        x.size() != static_cast<size_t>(input_shape.channels) * input_shape.height * input_shape.width ||
        weight.size() != static_cast<size_t>(out_channels) * input_shape.channels * kernel_h * kernel_w ||
        input_shape.height + 2u * pad_h < kernel_h ||
        input_shape.width + 2u * pad_w < kernel_w) {
        throw std::runtime_error("cpu_conv2d_nchw_no_bias shape mismatch");
    }
    output_shape.channels = out_channels;
    output_shape.height = ((input_shape.height + 2u * pad_h - kernel_h) / stride_h) + 1u;
    output_shape.width = ((input_shape.width + 2u * pad_w - kernel_w) / stride_w) + 1u;
    std::vector<float> out(static_cast<size_t>(output_shape.channels) * output_shape.height * output_shape.width);
    for (uint32_t oc = 0; oc < out_channels; ++oc) {
        for (uint32_t oh = 0; oh < output_shape.height; ++oh) {
            for (uint32_t ow = 0; ow < output_shape.width; ++ow) {
                float acc = 0.0f;
                for (uint32_t ic = 0; ic < input_shape.channels; ++ic) {
                    for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                        const int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
                        if (ih < 0 || ih >= static_cast<int>(input_shape.height)) {
                            continue;
                        }
                        for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                            const int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                            if (iw < 0 || iw >= static_cast<int>(input_shape.width)) {
                                continue;
                            }
                            const float xv = x[(static_cast<size_t>(ic) * input_shape.height + static_cast<uint32_t>(ih)) *
                                               input_shape.width + static_cast<uint32_t>(iw)];
                            const float wv = weight[((static_cast<size_t>(oc) * input_shape.channels + ic) * kernel_h + kh) *
                                                    kernel_w + kw];
                            acc += xv * wv;
                        }
                    }
                }
                out[(static_cast<size_t>(oc) * output_shape.height + oh) * output_shape.width + ow] = acc;
            }
        }
    }
    return out;
}

void cpu_batchnorm2d_inplace(std::vector<float>& x,
                             NchwShape shape,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             const std::vector<float>& running_mean,
                             const std::vector<float>& running_var,
                             bool relu,
                             float eps = 1.0e-5f) {
    if (x.size() != static_cast<size_t>(shape.channels) * shape.height * shape.width ||
        weight.size() != shape.channels ||
        bias.size() != shape.channels ||
        running_mean.size() != shape.channels ||
        running_var.size() != shape.channels) {
        throw std::runtime_error("cpu_batchnorm2d_inplace shape mismatch");
    }
    for (uint32_t c = 0; c < shape.channels; ++c) {
        const float scale = weight[c] / std::sqrt(running_var[c] + eps);
        const float shift = bias[c] - running_mean[c] * scale;
        for (uint32_t h = 0; h < shape.height; ++h) {
            for (uint32_t w = 0; w < shape.width; ++w) {
                float y = x[(static_cast<size_t>(c) * shape.height + h) * shape.width + w] * scale + shift;
                if (relu) {
                    y = std::max(y, 0.0f);
                }
                x[(static_cast<size_t>(c) * shape.height + h) * shape.width + w] = y;
            }
        }
    }
}

std::vector<float> add_relu_nchw(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("add_relu_nchw size mismatch");
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = std::max(a[i] + b[i], 0.0f);
    }
    return out;
}

std::vector<float> cpu_conv1d_ncw_no_bias(const std::vector<float>& x,
                                          const std::vector<float>& weight,
                                          uint32_t in_channels,
                                          uint32_t input_width,
                                          uint32_t out_channels,
                                          uint32_t kernel,
                                          uint32_t stride,
                                          uint32_t padding,
                                          uint32_t dilation,
                                          uint32_t& output_width) {
    if (in_channels == 0 || input_width == 0 || out_channels == 0 || kernel == 0 ||
        stride == 0 || dilation == 0 ||
        x.size() != static_cast<size_t>(in_channels) * input_width ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        input_width + 2u * padding < dilation * (kernel - 1u) + 1u) {
        throw std::runtime_error("cpu_conv1d_ncw_no_bias shape mismatch");
    }
    output_width = ((input_width + 2u * padding - dilation * (kernel - 1u) - 1u) / stride) + 1u;
    std::vector<float> out(static_cast<size_t>(out_channels) * output_width);
    for (uint32_t oc = 0; oc < out_channels; ++oc) {
        for (uint32_t ow = 0; ow < output_width; ++ow) {
            float acc = 0.0f;
            for (uint32_t ic = 0; ic < in_channels; ++ic) {
                for (uint32_t k = 0; k < kernel; ++k) {
                    const int iw = static_cast<int>(ow * stride + k * dilation) - static_cast<int>(padding);
                    if (iw < 0 || iw >= static_cast<int>(input_width)) {
                        continue;
                    }
                    const float xv = x[static_cast<size_t>(ic) * input_width + static_cast<uint32_t>(iw)];
                    const float wv = weight[(static_cast<size_t>(oc) * in_channels + ic) * kernel + k];
                    acc += xv * wv;
                }
            }
            out[static_cast<size_t>(oc) * output_width + ow] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_conv1d_ncw_bias(const std::vector<float>& x,
                                       const std::vector<float>& weight,
                                       const std::vector<float>& bias,
                                       uint32_t in_channels,
                                       uint32_t input_width,
                                       uint32_t out_channels,
                                       uint32_t kernel,
                                       uint32_t stride,
                                       uint32_t padding,
                                       uint32_t dilation,
                                       uint32_t& output_width) {
    auto out = cpu_conv1d_ncw_no_bias(x,
                                      weight,
                                      in_channels,
                                      input_width,
                                      out_channels,
                                      kernel,
                                      stride,
                                      padding,
                                      dilation,
                                      output_width);
    if (bias.size() != out_channels) {
        throw std::runtime_error("cpu_conv1d_ncw_bias bias shape mismatch");
    }
    for (uint32_t oc = 0; oc < out_channels; ++oc) {
        for (uint32_t t = 0; t < output_width; ++t) {
            out[static_cast<size_t>(oc) * output_width + t] += bias[oc];
        }
    }
    return out;
}

void cpu_batchnorm1d_inplace(std::vector<float>& x,
                             uint32_t channels,
                             uint32_t width,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             const std::vector<float>& running_mean,
                             const std::vector<float>& running_var,
                             bool relu,
                             float eps = 1.0e-5f) {
    if (x.size() != static_cast<size_t>(channels) * width ||
        weight.size() != channels ||
        bias.size() != channels ||
        running_mean.size() != channels ||
        running_var.size() != channels) {
        throw std::runtime_error("cpu_batchnorm1d_inplace shape mismatch");
    }
    for (uint32_t c = 0; c < channels; ++c) {
        const float scale = weight[c] / std::sqrt(running_var[c] + eps);
        const float shift = bias[c] - running_mean[c] * scale;
        for (uint32_t t = 0; t < width; ++t) {
            float y = x[static_cast<size_t>(c) * width + t] * scale + shift;
            if (relu) {
                y = std::max(y, 0.0f);
            }
            x[static_cast<size_t>(c) * width + t] = y;
        }
    }
}

void cpu_batchnorm1d_affine_false_inplace(std::vector<float>& x,
                                          uint32_t channels,
                                          uint32_t width,
                                          const std::vector<float>& running_mean,
                                          const std::vector<float>& running_var,
                                          float eps = 1.0e-5f) {
    if (x.size() != static_cast<size_t>(channels) * width ||
        running_mean.size() != channels ||
        running_var.size() != channels) {
        throw std::runtime_error("cpu_batchnorm1d_affine_false_inplace shape mismatch");
    }
    for (uint32_t c = 0; c < channels; ++c) {
        const float scale = 1.0f / std::sqrt(running_var[c] + eps);
        const float shift = -running_mean[c] * scale;
        for (uint32_t t = 0; t < width; ++t) {
            x[static_cast<size_t>(c) * width + t] =
                x[static_cast<size_t>(c) * width + t] * scale + shift;
        }
    }
}

std::vector<float> cpu_stats_pool_ncw_unbiased(const std::vector<float>& x,
                                               uint32_t channels,
                                               uint32_t width) {
    if (width < 2u || x.size() != static_cast<size_t>(channels) * width) {
        throw std::runtime_error("cpu_stats_pool_ncw_unbiased shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels) * 2u);
    for (uint32_t c = 0; c < channels; ++c) {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(c) * width;
        for (uint32_t t = 0; t < width; ++t) {
            sum += x[base + t];
        }
        const double mean = sum / static_cast<double>(width);
        double sq = 0.0;
        for (uint32_t t = 0; t < width; ++t) {
            const double delta = static_cast<double>(x[base + t]) - mean;
            sq += delta * delta;
        }
        out[c] = static_cast<float>(mean);
        out[channels + c] = static_cast<float>(std::sqrt(sq / static_cast<double>(width - 1u)));
    }
    return out;
}

void relu_inplace(std::vector<float>& x) {
    for (float& value : x) {
        value = std::max(value, 0.0f);
    }
}

void sigmoid_inplace(std::vector<float>& x) {
    for (float& value : x) {
        value = 1.0f / (1.0f + std::exp(-value));
    }
}

std::vector<float> concat_channels_ncw(const std::vector<float>& a,
                                       uint32_t a_channels,
                                       const std::vector<float>& b,
                                       uint32_t b_channels,
                                       uint32_t width) {
    if (a.size() != static_cast<size_t>(a_channels) * width ||
        b.size() != static_cast<size_t>(b_channels) * width) {
        throw std::runtime_error("concat_channels_ncw shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(a_channels + b_channels) * width);
    std::copy(a.begin(), a.end(), out.begin());
    std::copy(b.begin(), b.end(), out.begin() + static_cast<std::ptrdiff_t>(a.size()));
    return out;
}

std::vector<float> cpu_cam_context_ncw(const std::vector<float>& x, uint32_t channels, uint32_t width) {
    if (x.size() != static_cast<size_t>(channels) * width || width == 0) {
        throw std::runtime_error("cpu_cam_context_ncw shape mismatch");
    }
    constexpr uint32_t seg_len = 100;
    const uint32_t pooled = (width + seg_len - 1u) / seg_len;
    std::vector<float> context(static_cast<size_t>(channels) * width);
    for (uint32_t c = 0; c < channels; ++c) {
        double mean_acc = 0.0;
        for (uint32_t t = 0; t < width; ++t) {
            mean_acc += x[static_cast<size_t>(c) * width + t];
        }
        const float mean = static_cast<float>(mean_acc / static_cast<double>(width));
        for (uint32_t p = 0; p < pooled; ++p) {
            const uint32_t start = p * seg_len;
            const uint32_t end = std::min(width, start + seg_len);
            double seg_acc = 0.0;
            for (uint32_t t = start; t < end; ++t) {
                seg_acc += x[static_cast<size_t>(c) * width + t];
            }
            const float seg_mean = static_cast<float>(seg_acc / static_cast<double>(end - start));
            for (uint32_t offset = 0; offset < seg_len; ++offset) {
                const uint32_t t = start + offset;
                if (t >= width) {
                    break;
                }
                context[static_cast<size_t>(c) * width + t] = mean + seg_mean;
            }
        }
    }
    return context;
}

std::vector<float> cpu_campplus_dense_tdnn_layer(const std::string& debug_name,
                                                 const std::vector<float>& input,
                                                 uint32_t in_channels,
                                                 uint32_t width,
                                                 const std::vector<float>& nonlinear1_bn_weight,
                                                 const std::vector<float>& nonlinear1_bn_bias,
                                                 const std::vector<float>& nonlinear1_bn_running_mean,
                                                 const std::vector<float>& nonlinear1_bn_running_var,
                                                 const std::vector<float>& linear1_weight,
                                                 const std::vector<float>& nonlinear2_bn_weight,
                                                 const std::vector<float>& nonlinear2_bn_bias,
                                                 const std::vector<float>& nonlinear2_bn_running_mean,
                                                 const std::vector<float>& nonlinear2_bn_running_var,
                                                 const std::vector<float>& cam_linear_local_weight,
                                                 const std::vector<float>& cam_linear1_weight,
                                                 const std::vector<float>& cam_linear1_bias,
                                                 const std::vector<float>& cam_linear2_weight,
                                                 const std::vector<float>& cam_linear2_bias,
                                                 uint32_t local_padding = 1,
                                                 uint32_t local_dilation = 1) {
    constexpr uint32_t bn_channels = 128;
    constexpr uint32_t reduction_channels = 64;
    constexpr uint32_t out_channels = 32;
    if (input.size() != static_cast<size_t>(in_channels) * width) {
        throw std::runtime_error(debug_name + " input shape mismatch");
    }
    auto x = input;
    cpu_batchnorm1d_inplace(x,
                            in_channels,
                            width,
                            nonlinear1_bn_weight,
                            nonlinear1_bn_bias,
                            nonlinear1_bn_running_mean,
                            nonlinear1_bn_running_var,
                            true);
    uint32_t linear1_width = 0;
    x = cpu_conv1d_ncw_no_bias(x, linear1_weight, in_channels, width, bn_channels, 1, 1, 0, 1, linear1_width);
    if (linear1_width != width) {
        throw std::runtime_error(debug_name + " linear1 width mismatch");
    }
    cpu_batchnorm1d_inplace(x,
                            bn_channels,
                            width,
                            nonlinear2_bn_weight,
                            nonlinear2_bn_bias,
                            nonlinear2_bn_running_mean,
                            nonlinear2_bn_running_var,
                            true);
    uint32_t local_width = 0;
    auto local = cpu_conv1d_ncw_no_bias(
        x, cam_linear_local_weight, bn_channels, width, out_channels, 3, 1, local_padding, local_dilation, local_width);
    if (local_width != width) {
        throw std::runtime_error(debug_name + " local width mismatch");
    }
    auto context = cpu_cam_context_ncw(x, bn_channels, width);
    uint32_t context1_width = 0;
    auto context1 = cpu_conv1d_ncw_bias(
        context, cam_linear1_weight, cam_linear1_bias, bn_channels, width, reduction_channels, 1, 1, 0, 1, context1_width);
    if (context1_width != width) {
        throw std::runtime_error(debug_name + " context1 width mismatch");
    }
    relu_inplace(context1);
    uint32_t context2_width = 0;
    auto mask = cpu_conv1d_ncw_bias(
        context1, cam_linear2_weight, cam_linear2_bias, reduction_channels, width, out_channels, 1, 1, 0, 1, context2_width);
    if (context2_width != width) {
        throw std::runtime_error(debug_name + " context2 width mismatch");
    }
    sigmoid_inplace(mask);
    for (size_t i = 0; i < local.size(); ++i) {
        local[i] *= mask[i];
    }
    return local;
}

std::vector<float> cpu_campplus_head_layer1_block0(const std::vector<float>& input,
                                                   NchwShape input_shape,
                                                   const std::vector<float>& conv1_weight,
                                                   const std::vector<float>& bn1_weight,
                                                   const std::vector<float>& bn1_bias,
                                                   const std::vector<float>& bn1_running_mean,
                                                   const std::vector<float>& bn1_running_var,
                                                   const std::vector<float>& conv2_weight,
                                                   const std::vector<float>& bn2_weight,
                                                   const std::vector<float>& bn2_bias,
                                                   const std::vector<float>& bn2_running_mean,
                                                   const std::vector<float>& bn2_running_var,
                                                   const std::vector<float>& shortcut_weight,
                                                   const std::vector<float>& shortcut_bn_weight,
                                                   const std::vector<float>& shortcut_bn_bias,
                                                   const std::vector<float>& shortcut_bn_running_mean,
                                                   const std::vector<float>& shortcut_bn_running_var,
                                                   NchwShape& output_shape) {
    NchwShape conv1_shape;
    auto out = cpu_conv2d_nchw_no_bias(input, input_shape, conv1_weight, 32, 3, 3, 2, 1, 1, 1, conv1_shape);
    cpu_batchnorm2d_inplace(out, conv1_shape, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, true);
    NchwShape conv2_shape;
    out = cpu_conv2d_nchw_no_bias(out, conv1_shape, conv2_weight, 32, 3, 3, 1, 1, 1, 1, conv2_shape);
    cpu_batchnorm2d_inplace(out, conv2_shape, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, false);
    NchwShape shortcut_shape;
    auto shortcut = cpu_conv2d_nchw_no_bias(input, input_shape, shortcut_weight, 32, 1, 1, 2, 1, 0, 0, shortcut_shape);
    cpu_batchnorm2d_inplace(shortcut,
                            shortcut_shape,
                            shortcut_bn_weight,
                            shortcut_bn_bias,
                            shortcut_bn_running_mean,
                            shortcut_bn_running_var,
                            false);
    if (conv2_shape.channels != shortcut_shape.channels ||
        conv2_shape.height != shortcut_shape.height ||
        conv2_shape.width != shortcut_shape.width) {
        throw std::runtime_error("campplus layer1 block0 shortcut shape mismatch");
    }
    output_shape = conv2_shape;
    return add_relu_nchw(out, shortcut);
}

std::vector<float> cpu_campplus_head_layer1_block1(const std::vector<float>& input,
                                                   NchwShape input_shape,
                                                   const std::vector<float>& conv1_weight,
                                                   const std::vector<float>& bn1_weight,
                                                   const std::vector<float>& bn1_bias,
                                                   const std::vector<float>& bn1_running_mean,
                                                   const std::vector<float>& bn1_running_var,
                                                   const std::vector<float>& conv2_weight,
                                                   const std::vector<float>& bn2_weight,
                                                   const std::vector<float>& bn2_bias,
                                                   const std::vector<float>& bn2_running_mean,
                                                   const std::vector<float>& bn2_running_var,
                                                   NchwShape& output_shape) {
    NchwShape conv1_shape;
    auto out = cpu_conv2d_nchw_no_bias(input, input_shape, conv1_weight, 32, 3, 3, 1, 1, 1, 1, conv1_shape);
    cpu_batchnorm2d_inplace(out, conv1_shape, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, true);
    NchwShape conv2_shape;
    out = cpu_conv2d_nchw_no_bias(out, conv1_shape, conv2_weight, 32, 3, 3, 1, 1, 1, 1, conv2_shape);
    cpu_batchnorm2d_inplace(out, conv2_shape, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, false);
    if (conv2_shape.channels != input_shape.channels ||
        conv2_shape.height != input_shape.height ||
        conv2_shape.width != input_shape.width) {
        throw std::runtime_error("campplus layer1 block1 residual shape mismatch");
    }
    output_shape = conv2_shape;
    return add_relu_nchw(out, input);
}

std::vector<float> cpu_conv1d_dilated_same(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t dilation) {
    std::vector<float> out(static_cast<size_t>(tokens) * out_channels);
    const int pad = static_cast<int>((kernel - 1) * dilation / 2);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t out_c = 0; out_c < out_channels; ++out_c) {
            float acc = bias[out_c];
            for (uint32_t in_c = 0; in_c < in_channels; ++in_c) {
                for (uint32_t k = 0; k < kernel; ++k) {
                    const int src_t = static_cast<int>(t) + static_cast<int>(k * dilation) - pad;
                    if (src_t >= 0 && src_t < static_cast<int>(tokens)) {
                        acc += x[static_cast<size_t>(src_t) * in_channels + in_c] *
                               weight[(static_cast<size_t>(out_c) * in_channels + in_c) * kernel + k];
                    }
                }
            }
            out[static_cast<size_t>(t) * out_channels + out_c] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_conv_transpose1d(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t stride, uint32_t padding) {
    const uint32_t out_tokens = (tokens - 1) * stride + kernel - 2 * padding;
    std::vector<float> out(static_cast<size_t>(out_tokens) * out_channels);
    for (uint32_t out_t = 0; out_t < out_tokens; ++out_t) {
        for (uint32_t out_c = 0; out_c < out_channels; ++out_c) {
            float acc = bias[out_c];
            for (uint32_t in_c = 0; in_c < in_channels; ++in_c) {
                for (uint32_t k = 0; k < kernel; ++k) {
                    const int numerator = static_cast<int>(out_t) + static_cast<int>(padding) - static_cast<int>(k);
                    if (numerator >= 0 && (static_cast<uint32_t>(numerator) % stride) == 0) {
                        const uint32_t in_t = static_cast<uint32_t>(numerator) / stride;
                        if (in_t < tokens) {
                            const size_t x_idx = static_cast<size_t>(in_t) * in_channels + in_c;
                            const size_t w_idx = (static_cast<size_t>(in_c) * out_channels + out_c) * kernel + k;
                            acc += x[x_idx] * weight[w_idx];
                        }
                    }
                }
            }
            out[static_cast<size_t>(out_t) * out_channels + out_c] = acc;
        }
    }
    return out;
}

float clamped_channel_sample(const std::vector<float>& x, int t, uint32_t c, uint32_t tokens, uint32_t channels) {
    if (t < 0) {
        t = 0;
    } else if (t >= static_cast<int>(tokens)) {
        t = static_cast<int>(tokens) - 1;
    }
    return x[static_cast<size_t>(t) * channels + c];
}

float bigvgan_upsample_value(const std::vector<float>& x, const std::vector<float>& filter, uint32_t up_t, uint32_t c, uint32_t tokens, uint32_t channels) {
    constexpr uint32_t ratio = 2;
    constexpr uint32_t kernel = 12;
    constexpr uint32_t input_pad = 5;
    constexpr uint32_t crop_left = 15;
    const uint32_t full_t = up_t + crop_left;
    float acc = 0.0f;
    for (uint32_t k = 0; k < kernel; ++k) {
        if (full_t >= k && ((full_t - k) % ratio) == 0) {
            const uint32_t padded_t = (full_t - k) / ratio;
            const int src_t = static_cast<int>(padded_t) - static_cast<int>(input_pad);
            acc += clamped_channel_sample(x, src_t, c, tokens, channels) * filter[k];
        }
    }
    return static_cast<float>(ratio) * acc;
}

std::vector<float> cpu_bigvgan_activation(const std::vector<float>& x, const std::vector<float>& up_filter, const std::vector<float>& down_filter, const std::vector<float>& alpha_log, const std::vector<float>& beta_log, uint32_t tokens, uint32_t channels) {
    std::vector<float> out(static_cast<size_t>(tokens) * channels);
    const uint32_t up_tokens = tokens * 2;
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < channels; ++c) {
            const float alpha = std::exp(alpha_log[c]);
            const float beta = std::exp(beta_log[c]);
            float acc = 0.0f;
            for (uint32_t k = 0; k < 12; ++k) {
                int up_t = static_cast<int>(t * 2 + k) - 5;
                if (up_t < 0) {
                    up_t = 0;
                } else if (up_t >= static_cast<int>(up_tokens)) {
                    up_t = static_cast<int>(up_tokens) - 1;
                }
                const float y = bigvgan_upsample_value(x, up_filter, static_cast<uint32_t>(up_t), c, tokens, channels);
                const float s = std::sin(y * alpha);
                acc += (y + (s * s) / (beta + 1e-9f)) * down_filter[k];
            }
            out[static_cast<size_t>(t) * channels + c] = acc;
        }
    }
    return out;
}

std::vector<float> reflect_pad_1d(const std::vector<float>& x, uint32_t tokens, uint32_t channels, uint32_t pad_left, uint32_t pad_right) {
    const uint32_t out_tokens = tokens + pad_left + pad_right;
    std::vector<float> out(static_cast<size_t>(out_tokens) * channels);
    for (uint32_t t = 0; t < out_tokens; ++t) {
        int src = static_cast<int>(t) - static_cast<int>(pad_left);
        while (src < 0 || src >= static_cast<int>(tokens)) {
            if (src < 0) {
                src = -src;
            }
            if (src >= static_cast<int>(tokens)) {
                src = 2 * static_cast<int>(tokens) - 2 - src;
            }
        }
        for (uint32_t c = 0; c < channels; ++c) {
            out[static_cast<size_t>(t) * channels + c] = x[static_cast<size_t>(src) * channels + c];
        }
    }
    return out;
}

std::vector<float> slice_tokens(const std::vector<float>& x, uint32_t start_token, uint32_t tokens, uint32_t channels) {
    std::vector<float> out(static_cast<size_t>(tokens) * channels);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::copy(x.begin() + static_cast<size_t>(start_token + t) * channels,
                  x.begin() + static_cast<size_t>(start_token + t + 1) * channels,
                  out.begin() + static_cast<size_t>(t) * channels);
    }
    return out;
}

std::vector<float> cpu_conv1d_reflect_same(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    const uint32_t pad_left = kernel / 2;
    const uint32_t pad_right = kernel - 1 - pad_left;
    auto padded = reflect_pad_1d(x, tokens, in_channels, pad_left, pad_right);
    auto full = cpu_conv1d_same(padded, weight, bias, tokens + pad_left + pad_right, in_channels, out_channels, kernel);
    return slice_tokens(full, pad_left, tokens, out_channels);
}

std::vector<float> cpu_groupnorm1(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, uint32_t tokens, uint32_t channels, float eps) {
    const size_t count = static_cast<size_t>(tokens) * channels;
    float mean = 0.0f;
    for (float v : x) {
        mean += v;
    }
    mean /= static_cast<float>(count);
    float var = 0.0f;
    for (float v : x) {
        const float d = v - mean;
        var += d * d;
    }
    var /= static_cast<float>(count);
    const float inv_std = 1.0f / std::sqrt(var + eps);
    std::vector<float> out(count);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < channels; ++c) {
            const size_t i = static_cast<size_t>(t) * channels + c;
            out[i] = (x[i] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
    return out;
}

std::vector<float> cpu_mish(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] * std::tanh(std::log1p(std::exp(x[i])));
    }
    return out;
}

std::vector<float> cpu_mask_rows(const std::vector<float>& x, const std::vector<uint32_t>& mask, uint32_t tokens, uint32_t width) {
    std::vector<float> out = x;
    for (uint32_t t = 0; t < tokens; ++t) {
        if (mask[t] == 0) {
            std::fill(out.begin() + static_cast<size_t>(t) * width,
                      out.begin() + static_cast<size_t>(t + 1) * width,
                      0.0f);
        }
    }
    return out;
}

std::vector<float> cpu_glu_split(const std::vector<float>& x, uint32_t tokens, uint32_t width) {
    std::vector<float> out(static_cast<size_t>(tokens) * width);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t c = 0; c < width; ++c) {
            const size_t base = static_cast<size_t>(t) * width * 2;
            const float a = x[base + c];
            const float b = x[base + width + c];
            out[static_cast<size_t>(t) * width + c] = a / (1.0f + std::exp(-b));
        }
    }
    return out;
}

std::vector<float> cpu_gelu_new(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    constexpr float c = 0.7978845608028654f;
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x[i];
        if (v > 10.0f) {
            out[i] = v;
        } else if (v < -10.0f) {
            out[i] = 0.0f;
        } else {
            out[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
        }
    }
    return out;
}

std::vector<float> cpu_timestep_embedding(const std::vector<float>& timesteps, const std::vector<float>& freqs, float scale) {
    const size_t half = freqs.size();
    std::vector<float> out(timesteps.size() * half * 2);
    for (size_t b = 0; b < timesteps.size(); ++b) {
        for (size_t i = 0; i < half; ++i) {
            const float arg = scale * timesteps[b] * freqs[i];
            out[b * half * 2 + i] = std::cos(arg);
            out[b * half * 2 + half + i] = std::sin(arg);
        }
    }
    return out;
}

std::vector<float> cpu_attention_single_head_masked(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t head_dim) {
    std::vector<float> out(static_cast<size_t>(tokens) * head_dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (uint32_t tq = 0; tq < tokens; ++tq) {
        std::vector<float> scores(tokens);
        float max_score = -std::numeric_limits<float>::infinity();
        for (uint32_t tk = 0; tk < tokens; ++tk) {
            if (key_mask[tk] == 0) {
                scores[tk] = -std::numeric_limits<float>::infinity();
                continue;
            }
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                score += q[static_cast<size_t>(tq) * head_dim + d] * k[static_cast<size_t>(tk) * head_dim + d];
            }
            score *= scale;
            scores[tk] = score;
            max_score = std::max(max_score, score);
        }
        float denom = 0.0f;
        for (float& s : scores) {
            if (!std::isfinite(s)) {
                s = 0.0f;
                continue;
            }
            s = std::exp(s - max_score);
            denom += s;
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t tk = 0; tk < tokens; ++tk) {
                acc += (scores[tk] / denom) * v[static_cast<size_t>(tk) * head_dim + d];
            }
            out[static_cast<size_t>(tq) * head_dim + d] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_attention_single_head(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens, uint32_t head_dim) {
    return cpu_attention_single_head_masked(q, k, v, std::vector<uint32_t>(tokens, 1), tokens, head_dim);
}

std::vector<float> cpu_attention_single_head_causal(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens, uint32_t head_dim) {
    std::vector<float> out(static_cast<size_t>(tokens) * head_dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (uint32_t tq = 0; tq < tokens; ++tq) {
        std::vector<float> scores(tq + 1);
        float max_score = -std::numeric_limits<float>::infinity();
        for (uint32_t tk = 0; tk <= tq; ++tk) {
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                score += q[static_cast<size_t>(tq) * head_dim + d] * k[static_cast<size_t>(tk) * head_dim + d];
            }
            score *= scale;
            scores[tk] = score;
            max_score = std::max(max_score, score);
        }
        float denom = 0.0f;
        for (float& s : scores) {
            s = std::exp(s - max_score);
            denom += s;
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t tk = 0; tk <= tq; ++tk) {
                acc += (scores[tk] / denom) * v[static_cast<size_t>(tk) * head_dim + d];
            }
            out[static_cast<size_t>(tq) * head_dim + d] = acc;
        }
    }
    return out;
}

std::vector<float> cpu_attention_single_query(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t key_tokens, uint32_t head_dim) {
    std::vector<float> out(head_dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(key_tokens);
    float max_score = -std::numeric_limits<float>::infinity();
    for (uint32_t tk = 0; tk < key_tokens; ++tk) {
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            score += q[d] * k[static_cast<size_t>(tk) * head_dim + d];
        }
        score *= scale;
        scores[tk] = score;
        max_score = std::max(max_score, score);
    }
    float denom = 0.0f;
    for (float& s : scores) {
        s = std::exp(s - max_score);
        denom += s;
    }
    for (uint32_t d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (uint32_t tk = 0; tk < key_tokens; ++tk) {
            acc += (scores[tk] / denom) * v[static_cast<size_t>(tk) * head_dim + d];
        }
        out[d] = acc;
    }
    return out;
}

std::vector<float> cpu_conformer_rel_attention_context(const std::vector<float>& q,
                                                       const std::vector<float>& k,
                                                       const std::vector<float>& v,
                                                       const std::vector<float>& p,
                                                       const std::vector<float>& bias_u,
                                                       const std::vector<float>& bias_v,
                                                       const std::vector<uint32_t>& key_mask,
                                                       uint32_t tokens,
                                                       uint32_t heads,
                                                       uint32_t head_dim) {
    const uint32_t dim = heads * head_dim;
    if (tokens == 0 || heads == 0 || head_dim == 0 ||
        q.size() != static_cast<size_t>(tokens) * dim ||
        k.size() != static_cast<size_t>(tokens) * dim ||
        v.size() != static_cast<size_t>(tokens) * dim ||
        p.size() != static_cast<size_t>(tokens) * dim ||
        bias_u.size() != dim ||
        bias_v.size() != dim ||
        key_mask.size() != tokens) {
        throw std::runtime_error("cpu_conformer_rel_attention_context shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(tokens) * dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t tq = 0; tq < tokens; ++tq) {
            std::vector<float> scores(tokens);
            float max_score = -std::numeric_limits<float>::infinity();
            for (uint32_t tk = 0; tk < tokens; ++tk) {
                if (key_mask[tk] == 0) {
                    scores[tk] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float ac = 0.0f;
                float bd = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const size_t q_idx = static_cast<size_t>(tq) * dim + h * head_dim + d;
                    const size_t k_idx = static_cast<size_t>(tk) * dim + h * head_dim + d;
                    const size_t b_idx = static_cast<size_t>(h) * head_dim + d;
                    const float qv = q[q_idx];
                    ac += (qv + bias_u[b_idx]) * k[k_idx];
                    bd += (qv + bias_v[b_idx]) * p[k_idx];
                }
                const float score = (ac + bd) * scale;
                scores[tk] = score;
                max_score = std::max(max_score, score);
            }
            float denom = 0.0f;
            for (float& score : scores) {
                if (!std::isfinite(score)) {
                    score = 0.0f;
                    continue;
                }
                score = std::exp(score - max_score);
                denom += score;
            }
            if (denom == 0.0f) {
                throw std::runtime_error("Conformer relative attention encountered fully masked row");
            }
            for (uint32_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (uint32_t tk = 0; tk < tokens; ++tk) {
                    const size_t v_idx = static_cast<size_t>(tk) * dim + h * head_dim + d;
                    acc += (scores[tk] / denom) * v[v_idx];
                }
                out[static_cast<size_t>(tq) * dim + h * head_dim + d] = acc;
            }
        }
    }
    return out;
}

std::vector<float> cpu_cross_attention_heads_masked(const std::vector<float>& q,
                                                    const std::vector<float>& k,
                                                    const std::vector<float>& v,
                                                    const std::vector<uint32_t>& key_mask,
                                                    uint32_t query_tokens,
                                                    uint32_t key_tokens,
                                                    uint32_t heads,
                                                    uint32_t head_dim) {
    const uint32_t inner = heads * head_dim;
    if (q.size() != static_cast<size_t>(query_tokens) * inner ||
        k.size() != static_cast<size_t>(key_tokens) * inner ||
        v.size() != static_cast<size_t>(key_tokens) * inner ||
        key_mask.size() != key_tokens) {
        throw std::runtime_error("cpu_cross_attention_heads_masked shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(query_tokens) * inner);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t tq = 0; tq < query_tokens; ++tq) {
            std::vector<float> scores(key_tokens);
            float max_score = -std::numeric_limits<float>::infinity();
            for (uint32_t tk = 0; tk < key_tokens; ++tk) {
                if (key_mask[tk] == 0) {
                    scores[tk] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float score = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const size_t qi = static_cast<size_t>(tq) * inner + h * head_dim + d;
                    const size_t ki = static_cast<size_t>(tk) * inner + h * head_dim + d;
                    score += q[qi] * k[ki];
                }
                score *= scale;
                scores[tk] = score;
                max_score = std::max(max_score, score);
            }
            float denom = 0.0f;
            for (float& score : scores) {
                if (!std::isfinite(score)) {
                    score = 0.0f;
                    continue;
                }
                score = std::exp(score - max_score);
                denom += score;
            }
            for (uint32_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (uint32_t tk = 0; tk < key_tokens; ++tk) {
                    const size_t vi = static_cast<size_t>(tk) * inner + h * head_dim + d;
                    acc += (scores[tk] / denom) * v[vi];
                }
                const size_t oi = static_cast<size_t>(tq) * inner + h * head_dim + d;
                out[oi] = acc;
            }
        }
    }
    return out;
}

// W2V-BERT relative-key attention with asymmetric distance bias.
// distance_emb: [left_max + right_max + 1, head_dim] (e.g. [73, 64] for w2v-bert-2.0)
// HuggingFace formula: distance = tk - tq (KEY minus QUERY), clamped to [-left_max, right_max]
// dist_idx = clamped_dist + left_max
// score += Q[tq,h] · distance_emb[dist_idx]  (relative_key einsum "bhld,lrd->bhlr")
std::vector<float> cpu_w2v_bert_cross_attention_with_distance(
        const std::vector<float>& q,
        const std::vector<float>& k,
        const std::vector<float>& v,
        const std::vector<uint32_t>& key_mask,
        uint32_t tokens,
        const std::vector<float>& distance_emb,
        int32_t left_max,
        int32_t right_max) {
    constexpr uint32_t heads = 16, head_dim = 64;
    const uint32_t inner = heads * head_dim;
    const uint32_t dist_size = static_cast<uint32_t>(left_max + right_max + 1);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (q.size() != static_cast<size_t>(tokens) * inner ||
        k.size() != q.size() || v.size() != q.size() ||
        key_mask.size() != tokens) {
        throw std::invalid_argument("cpu_w2v_bert_cross_attention_with_distance shape mismatch");
    }
    // Fallback to plain attention when no distance embedding is provided
    if (distance_emb.empty()) {
        return cpu_cross_attention_heads_masked(q, k, v, key_mask, tokens, tokens, heads, head_dim);
    }
    if (distance_emb.size() != static_cast<size_t>(dist_size) * head_dim) {
        throw std::invalid_argument("cpu_w2v_bert_cross_attention_with_distance: distance_emb size mismatch");
    }
    std::vector<float> out(static_cast<size_t>(tokens) * inner);
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t tq = 0; tq < tokens; ++tq) {
            std::vector<float> scores(tokens);
            float max_score = -std::numeric_limits<float>::infinity();
            for (uint32_t tk = 0; tk < tokens; ++tk) {
                if (key_mask[tk] == 0) {
                    scores[tk] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dot += q[tq * inner + h * head_dim + d] * k[tk * inner + h * head_dim + d];
                }
                dot *= scale;
                const int32_t neg_dist = static_cast<int32_t>(tk) - static_cast<int32_t>(tq);
                const int32_t clamped = std::max(-left_max, std::min(right_max, neg_dist));
                const uint32_t dist_idx = static_cast<uint32_t>(clamped + left_max);
                float pos = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    pos += q[tq * inner + h * head_dim + d] * distance_emb[dist_idx * head_dim + d];
                }
                scores[tk] = dot + pos * scale;
                max_score = std::max(max_score, scores[tk]);
            }
            float denom = 0.0f;
            for (float& s : scores) {
                if (!std::isfinite(s)) { s = 0.0f; continue; }
                s = std::exp(s - max_score);
                denom += s;
            }
            for (uint32_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (uint32_t tk = 0; tk < tokens; ++tk) {
                    acc += (scores[tk] / denom) * v[tk * inner + h * head_dim + d];
                }
                out[tq * inner + h * head_dim + d] = acc;
            }
        }
    }
    return out;
}

void apply_rope_inplace(std::vector<float>& x, uint32_t tokens, uint32_t heads, uint32_t head_dim) {
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t pair = 0; pair < head_dim / 2; ++pair) {
            const float theta = static_cast<float>(t) / std::pow(10000.0f, static_cast<float>(pair * 2) / static_cast<float>(head_dim));
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            for (uint32_t h = 0; h < heads; ++h) {
                const size_t base = (static_cast<size_t>(t) * heads + h) * head_dim + pair * 2;
                const float a = x[base];
                const float b = x[base + 1];
                x[base] = a * c - b * s;
                x[base + 1] = b * c + a * s;
            }
        }
    }
}


float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<float> tensor_as_f32(const mit2::Bundle& bundle, const std::string& name) {
    const auto* info = bundle.find(name);
    if (!info) {
        throw std::runtime_error("missing tensor: " + name);
    }
    const uint8_t* data = bundle.tensor_data(*info);
    size_t count = 1;
    for (int64_t dim : info->shape) {
        count *= static_cast<size_t>(dim);
    }
    std::vector<float> out(count);
    if (info->dtype == "f32") {
        std::memcpy(out.data(), data, count * sizeof(float));
    } else if (info->dtype == "f16") {
        const auto* h = reinterpret_cast<const uint16_t*>(data);
        for (size_t i = 0; i < count; ++i) {
            out[i] = half_to_float(h[i]);
        }
    } else {
        throw std::runtime_error("tensor is not floating point: " + name + " dtype=" + info->dtype);
    }
    return out;
}

uint64_t tensor_element_count(const mit2::TensorInfo& info) {
    uint64_t count = 1;
    for (int64_t dim : info.shape) {
        if (dim <= 0) {
            throw std::runtime_error("tensor has non-positive dimension: " + info.name);
        }
        count *= static_cast<uint64_t>(dim);
    }
    return count;
}

void require_f32_tensor_bytes(const mit2::TensorInfo& info) {
    const uint64_t expected = tensor_element_count(info) * sizeof(float);
    if (info.nbytes != expected) {
        throw std::runtime_error("tensor byte count mismatch: " + info.name);
    }
}

uint64_t dtype_byte_width(const std::string& dtype) {
    if (dtype == "f32" || dtype == "i32" || dtype == "u32") {
        return 4;
    }
    if (dtype == "f16") {
        return 2;
    }
    if (dtype == "i64") {
        return 8;
    }
    if (dtype == "u8") {
        return 1;
    }
    throw std::runtime_error("unsupported tensor dtype in bundle contract: " + dtype);
}

void require_tensor_dtype_bytes(const mit2::TensorInfo& info) {
    const uint64_t expected = tensor_element_count(info) * dtype_byte_width(info.dtype);
    if (info.nbytes != expected) {
        throw std::runtime_error("tensor dtype byte count mismatch: " + info.name);
    }
}

std::string sha256_hex(const uint8_t* data, uint64_t nbytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    uint64_t offset = 0;
    while (offset < nbytes) {
        const uint64_t remaining = nbytes - offset;
        const auto chunk = static_cast<CC_LONG>(
            std::min<uint64_t>(remaining, static_cast<uint64_t>(std::numeric_limits<CC_LONG>::max())));
        CC_SHA256_Update(&ctx, data + offset, chunk);
        offset += static_cast<uint64_t>(chunk);
    }
    CC_SHA256_Final(digest.data(), &ctx);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

std::string file_sha256_hex(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file for sha256: " + path);
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    std::array<char, 1024 * 1024> buffer{};
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) {
            CC_SHA256_Update(&ctx, buffer.data(), static_cast<CC_LONG>(got));
        }
    }
    if (in.bad()) {
        throw std::runtime_error("failed to read file for sha256: " + path);
    }
    CC_SHA256_Final(digest.data(), &ctx);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        throw std::runtime_error("alignment must be non-zero");
    }
    const uint64_t rem = value % alignment;
    if (rem == 0) {
        return value;
    }
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - rem)) {
        throw std::overflow_error("align_up overflow");
    }
    return value + (alignment - rem);
}

[[maybe_unused]] uint32_t parse_positive_u32_arg(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(name + " must be a positive u32");
    }
    return static_cast<uint32_t>(parsed);
}

void write_zero_padding(std::ofstream& out, uint64_t target_offset) {
    const auto current_pos = out.tellp();
    if (current_pos < 0) {
        throw std::runtime_error("failed to query output offset");
    }
    uint64_t current = static_cast<uint64_t>(current_pos);
    if (target_offset < current) {
        throw std::runtime_error("padding target moved backwards");
    }
    std::array<char, 4096> zeros{};
    while (current < target_offset) {
        const uint64_t remaining = target_offset - current;
        const auto chunk = static_cast<std::streamsize>(std::min<uint64_t>(remaining, zeros.size()));
        out.write(zeros.data(), chunk);
        if (!out) {
            throw std::runtime_error("failed to write bundle padding");
        }
        current += static_cast<uint64_t>(chunk);
    }
}

struct NativeBundleTensorPayload {
    std::string name;
    std::vector<int64_t> shape;
    std::string component;
    std::vector<float> values;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    std::string sha256;
    std::string dtype = "f32";
    std::vector<uint8_t> bytes;
};

void require_f32_value_count(const std::vector<float>& values, uint64_t expected, const std::string& name) {
    if (values.size() != expected) {
        std::ostringstream msg;
        msg << name << " expected " << expected << " f32 values, got " << values.size();
        throw std::runtime_error(msg.str());
    }
}

std::string shape_json_string(const std::vector<int64_t>& shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

void write_native_f32_bundle(const std::string& output_dir,
                             std::vector<NativeBundleTensorPayload>& tensors,
                             const std::string& metadata_json) {
    constexpr uint32_t version = 1;
    constexpr uint32_t alignment = 4096;
    std::filesystem::create_directories(output_dir);
    const std::string weights_path = (std::filesystem::path(output_dir) / "weights.bin").string();
    std::ofstream weights(weights_path, std::ios::binary);
    if (!weights) {
        throw std::runtime_error("failed to open bundle weights output: " + weights_path);
    }
    weights.write("MIT2", 4);
    write_u32_le(weights, version);
    write_u32_le(weights, alignment);
    for (auto& tensor : tensors) {
        const auto pos = weights.tellp();
        if (pos < 0) {
            throw std::runtime_error("failed to query bundle weights offset");
        }
        tensor.offset = align_up_u64(static_cast<uint64_t>(pos), alignment);
        write_zero_padding(weights, tensor.offset);
        if (!tensor.bytes.empty()) {
            tensor.nbytes = static_cast<uint64_t>(tensor.bytes.size());
            tensor.sha256 = sha256_hex(tensor.bytes.data(), tensor.nbytes);
            weights.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                          static_cast<std::streamsize>(tensor.nbytes));
        } else {
            tensor.dtype = "f32";
            tensor.nbytes = static_cast<uint64_t>(tensor.values.size() * sizeof(float));
            tensor.sha256 = sha256_hex(reinterpret_cast<const uint8_t*>(tensor.values.data()), tensor.nbytes);
            weights.write(reinterpret_cast<const char*>(tensor.values.data()),
                          static_cast<std::streamsize>(tensor.nbytes));
        }
        if (!weights) {
            throw std::runtime_error("failed to write bundle tensor: " + tensor.name);
        }
    }
    weights.close();
    if (!weights) {
        throw std::runtime_error("failed to finalize bundle weights: " + weights_path);
    }

    std::ostringstream manifest;
    manifest << "{\n";
    manifest << "  \"alignment\": " << alignment << ",\n";
    manifest << "  \"endianness\": \"little\",\n";
    manifest << "  \"format\": \"MIT2\",\n";
    manifest << "  \"metadata\": " << metadata_json << ",\n";
    manifest << "  \"tensors\": [\n";
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        manifest << "    {\n";
        manifest << "      \"component\": \"" << json_escape(tensor.component) << "\",\n";
        manifest << "      \"dtype\": \"" << json_escape(tensor.dtype) << "\",\n";
        manifest << "      \"layout\": \"row_major\",\n";
        manifest << "      \"name\": \"" << json_escape(tensor.name) << "\",\n";
        manifest << "      \"nbytes\": " << tensor.nbytes << ",\n";
        manifest << "      \"offset\": " << tensor.offset << ",\n";
        manifest << "      \"sha256\": \"" << tensor.sha256 << "\",\n";
        manifest << "      \"shape\": " << shape_json_string(tensor.shape) << "\n";
        manifest << "    }" << (i + 1 == tensors.size() ? "\n" : ",\n");
    }
    manifest << "  ],\n";
    manifest << "  \"version\": " << version << ",\n";
    manifest << "  \"weights_file\": \"weights.bin\"\n";
    manifest << "}\n";
    write_text_file((std::filesystem::path(output_dir) / "manifest.json").string(), manifest.str());
}

// Single-file variant: one file holding the MIT2 weights stream followed by the
// manifest JSON and a footer ([u64 manifest_offset][u64 manifest_size]"MIT2VOIC"),
// so a voice profile is a plain file instead of a directory.
void write_native_f32_bundle_single_file(const std::string& output_path,
                                         std::vector<NativeBundleTensorPayload>& tensors,
                                         const std::string& metadata_json) {
    constexpr uint32_t version = 1;
    constexpr uint32_t alignment = 4096;
    const auto parent = std::filesystem::path(output_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open single-file bundle output: " + output_path);
    }
    out.write("MIT2", 4);
    write_u32_le(out, version);
    write_u32_le(out, alignment);
    for (auto& tensor : tensors) {
        const auto pos = out.tellp();
        if (pos < 0) {
            throw std::runtime_error("failed to query bundle output offset");
        }
        tensor.offset = align_up_u64(static_cast<uint64_t>(pos), alignment);
        write_zero_padding(out, tensor.offset);
        if (!tensor.bytes.empty()) {
            tensor.nbytes = static_cast<uint64_t>(tensor.bytes.size());
            tensor.sha256 = sha256_hex(tensor.bytes.data(), tensor.nbytes);
            out.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                      static_cast<std::streamsize>(tensor.nbytes));
        } else {
            tensor.dtype = "f32";
            tensor.nbytes = static_cast<uint64_t>(tensor.values.size() * sizeof(float));
            tensor.sha256 = sha256_hex(reinterpret_cast<const uint8_t*>(tensor.values.data()), tensor.nbytes);
            out.write(reinterpret_cast<const char*>(tensor.values.data()),
                      static_cast<std::streamsize>(tensor.nbytes));
        }
        if (!out) {
            throw std::runtime_error("failed to write bundle tensor: " + tensor.name);
        }
    }

    std::ostringstream manifest;
    manifest << "{\n";
    manifest << "  \"alignment\": " << alignment << ",\n";
    manifest << "  \"endianness\": \"little\",\n";
    manifest << "  \"format\": \"MIT2\",\n";
    manifest << "  \"metadata\": " << metadata_json << ",\n";
    manifest << "  \"tensors\": [\n";
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        manifest << "    {\n";
        manifest << "      \"component\": \"" << json_escape(tensor.component) << "\",\n";
        manifest << "      \"dtype\": \"" << json_escape(tensor.dtype) << "\",\n";
        manifest << "      \"layout\": \"row_major\",\n";
        manifest << "      \"name\": \"" << json_escape(tensor.name) << "\",\n";
        manifest << "      \"nbytes\": " << tensor.nbytes << ",\n";
        manifest << "      \"offset\": " << tensor.offset << ",\n";
        manifest << "      \"sha256\": \"" << tensor.sha256 << "\",\n";
        manifest << "      \"shape\": " << shape_json_string(tensor.shape) << "\n";
        manifest << "    }" << (i + 1 == tensors.size() ? "\n" : ",\n");
    }
    manifest << "  ],\n";
    manifest << "  \"version\": " << version << "\n";
    manifest << "}\n";
    const std::string manifest_text = manifest.str();

    const auto manifest_pos = out.tellp();
    if (manifest_pos < 0) {
        throw std::runtime_error("failed to query manifest offset");
    }
    out.write(manifest_text.data(), static_cast<std::streamsize>(manifest_text.size()));
    const uint64_t manifest_offset = static_cast<uint64_t>(manifest_pos);
    const uint64_t manifest_size = static_cast<uint64_t>(manifest_text.size());
    for (int part = 0; part < 2; ++part) {
        const uint64_t value = part == 0 ? manifest_offset : manifest_size;
        for (int i = 0; i < 8; ++i) {
            const char byte = static_cast<char>((value >> (8 * i)) & 0xff);
            out.write(&byte, 1);
        }
    }
    out.write("MIT2VOIC", 8);
    out.close();
    if (!out) {
        throw std::runtime_error("failed to finalize single-file bundle: " + output_path);
    }
}

struct BundleIntegrityReport {
    uint64_t checksum_verified_count = 0;
    uint64_t aligned_tensor_count = 0;
    uint64_t checked_interval_count = 0;
};

BundleIntegrityReport validate_bundle_integrity(const mit2::Bundle& bundle, bool verify_sha256) {
    BundleIntegrityReport report;
    std::vector<const mit2::TensorInfo*> sorted;
    sorted.reserve(bundle.tensors().size());
    for (const auto& tensor : bundle.tensors()) {
        if (tensor.shape.empty()) {
            throw std::runtime_error("tensor has empty shape: " + tensor.name);
        }
        require_tensor_dtype_bytes(tensor);
        if (tensor.offset % bundle.alignment() != 0) {
            throw std::runtime_error("tensor offset is not bundle-aligned: " + tensor.name);
        }
        ++report.aligned_tensor_count;
        sorted.push_back(&tensor);
    }
    std::sort(sorted.begin(), sorted.end(), [](const mit2::TensorInfo* a, const mit2::TensorInfo* b) {
        if (a->offset != b->offset) {
            return a->offset < b->offset;
        }
        return a->name < b->name;
    });
    uint64_t previous_end = 12;
    for (const auto* tensor : sorted) {
        if (tensor->offset < previous_end) {
            throw std::runtime_error("tensor payload ranges overlap: " + tensor->name);
        }
        previous_end = tensor->offset + tensor->nbytes;
        ++report.checked_interval_count;
    }
    if (verify_sha256) {
        for (const auto& tensor : bundle.tensors()) {
            if (tensor.sha256.empty()) {
                throw std::runtime_error("tensor missing sha256: " + tensor.name);
            }
            const std::string actual = sha256_hex(bundle.tensor_data(tensor), tensor.nbytes);
            if (actual != tensor.sha256) {
                throw std::runtime_error("tensor sha256 mismatch: " + tensor.name);
            }
            ++report.checksum_verified_count;
        }
    }
    return report;
}

const mit2::TensorInfo& require_voice_tensor(const mit2::Bundle& bundle, const std::string& name) {
    const auto* info = bundle.find(name);
    if (!info) {
        throw std::runtime_error("voice bundle missing tensor: " + name);
    }
    if (info->dtype != "f32") {
        throw std::runtime_error("voice tensor must use f32 dtype: " + name + " dtype=" + info->dtype);
    }
    if (info->component != "voice") {
        throw std::runtime_error("voice tensor must have component=voice: " + name + " component=" + info->component);
    }
    require_f32_tensor_bytes(*info);
    return *info;
}

struct RequiredTensorSpec {
    std::string name;
    std::string component;
    std::vector<int64_t> shape;
};

const mit2::TensorInfo& require_model_tensor(const mit2::Bundle& bundle, const RequiredTensorSpec& spec) {
    const auto* info = bundle.find(spec.name);
    if (!info) {
        throw std::runtime_error("model bundle missing tensor: " + spec.name);
    }
    if (info->dtype != "f32") {
        throw std::runtime_error("model tensor must use f32 dtype: " + spec.name + " dtype=" + info->dtype);
    }
    if (info->component != spec.component) {
        throw std::runtime_error("model tensor component mismatch: " + spec.name + " component=" + info->component);
    }
    if (info->shape != spec.shape) {
        throw std::runtime_error("model tensor shape mismatch: " + spec.name);
    }
    require_f32_tensor_bytes(*info);
    return *info;
}

void print_shape_json(const std::vector<int64_t>& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << shape[i];
    }
    std::cout << "]";
}

void print_tensor_contract_json(const mit2::TensorInfo& info, bool trailing_comma) {
    std::cout << "    {\n";
    std::cout << "      \"name\": \"" << json_escape(info.name) << "\",\n";
    std::cout << "      \"dtype\": \"" << json_escape(info.dtype) << "\",\n";
    std::cout << "      \"shape\": ";
    print_shape_json(info.shape);
    std::cout << ",\n";
    std::cout << "      \"nbytes\": " << info.nbytes << ",\n";
    std::cout << "      \"component\": \"" << json_escape(info.component) << "\"\n";
    std::cout << "    }";
    if (trailing_comma) {
        std::cout << ",";
    }
    std::cout << "\n";
}

bool inspect_model_bundle_contract(const std::string& model_bundle_dir) {
    mit2::Bundle model(model_bundle_dir);
    const auto integrity = validate_bundle_integrity(model, true);
    const std::vector<RequiredTensorSpec> specs{
        {"gpt.text_embedding.weight", "gpt", {12001, 1280}},
        {"gpt.mel_embedding.weight", "gpt", {8194, 1280}},
        {"gpt.mel_pos_embedding.emb.weight", "gpt", {1818, 1280}},
        {"gpt.final_norm.weight", "gpt", {1280}},
        {"gpt.mel_head.weight", "gpt", {8194, 1280}},
        {"gpt.conditioning_encoder.embed.out.0.weight", "gpt", {512, 261632}},
        {"gpt.perceiver_encoder.proj_context.weight", "gpt", {1280, 512}},
        {"gpt.perceiver_encoder.layers.0.0.to_q.weight", "gpt", {512, 1280}},
        {"gpt.emo_perceiver_encoder.proj_context.weight", "gpt", {1024, 512}},
        {"gpt.emovec_layer.weight", "gpt", {1280, 1024}},
        {"gpt.emo_layer.weight", "gpt", {1280, 1280}},
        {"semantic_codec.quantizer.quantizers.0.codebook.weight", "semantic_codec", {8192, 8}},
        {"s2mel.net.gpt_layer.0.weight", "s2mel", {256, 1280}},
        {"s2mel.net.length_regulator.content_in_proj.weight", "s2mel", {512, 1024}},
        {"s2mel.net.length_regulator.model.0.weight", "s2mel", {512, 512, 3}},
        {"s2mel.net.cfm.estimator.x_embedder.weight_v", "s2mel", {512, 80}},
        {"s2mel.net.cfm.estimator.cond_projection.weight", "s2mel", {512, 512}},
        {"s2mel.net.cfm.estimator.transformer.layers.0.attention.wqkv.weight", "s2mel", {1536, 512}},
        {"s2mel.net.cfm.estimator.final_layer.linear.weight_v", "s2mel", {512, 512}},
        {"s2mel.net.cfm.estimator.wavenet.in_layers.0.conv.conv.weight_v", "s2mel", {1024, 512, 5}},
        {"bigvgan.conv_pre.weight_v", "bigvgan", {1536, 80, 7}},
        {"bigvgan.ups.0.0.weight_v", "bigvgan", {1536, 768, 8}},
        {"bigvgan.resblocks.0.convs1.0.weight_v", "bigvgan", {768, 768, 3}},
        {"bigvgan.conv_post.weight_v", "bigvgan", {1, 24, 7}},
    };

    std::unordered_map<std::string, uint64_t> component_counts;
    std::unordered_map<std::string, uint64_t> component_bytes;
    for (const auto& tensor : model.tensors()) {
        component_counts[tensor.component] += 1;
        component_bytes[tensor.component] += tensor.nbytes;
    }
    for (const auto& required : std::vector<std::string>{"gpt", "semantic_codec", "s2mel", "bigvgan"}) {
        if (component_counts[required] == 0) {
            throw std::runtime_error("model bundle missing component: " + required);
        }
    }

    std::vector<const mit2::TensorInfo*> required_tensors;
    required_tensors.reserve(specs.size());
    for (const auto& spec : specs) {
        required_tensors.push_back(&require_model_tensor(model, spec));
    }

    std::cout << "{\n";
    std::cout << "  \"stage\": \"model_bundle_contract\",\n";
    std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
    std::cout << "  \"ok\": true,\n";
    std::cout << "  \"format\": \"MIT2\",\n";
    std::cout << "  \"version\": " << model.version() << ",\n";
    std::cout << "  \"endianness\": \"" << json_escape(model.endianness()) << "\",\n";
    std::cout << "  \"alignment\": " << model.alignment() << ",\n";
    std::cout << "  \"weights_file\": \"" << json_escape(model.weights_file()) << "\",\n";
    std::cout << "  \"weights_bytes\": " << model.weights_size() << ",\n";
    std::cout << "  \"tensor_count\": " << model.tensor_count() << ",\n";
    std::cout << "  \"tensor_bytes\": " << model.total_tensor_bytes() << ",\n";
    std::cout << "  \"integrity\": {\n";
    std::cout << "    \"aligned_tensor_count\": " << integrity.aligned_tensor_count << ",\n";
    std::cout << "    \"checked_interval_count\": " << integrity.checked_interval_count << ",\n";
    std::cout << "    \"sha256_verified_count\": " << integrity.checksum_verified_count << "\n";
    std::cout << "  },\n";
    std::cout << "  \"component_counts\": {\n";
    const std::vector<std::string> components{"gpt", "semantic_codec", "s2mel", "bigvgan"};
    for (size_t i = 0; i < components.size(); ++i) {
        const auto& component = components[i];
        std::cout << "    \"" << component << "\": " << component_counts[component];
        std::cout << (i + 1 == components.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"component_bytes\": {\n";
    for (size_t i = 0; i < components.size(); ++i) {
        const auto& component = components[i];
        std::cout << "    \"" << component << "\": " << component_bytes[component];
        std::cout << (i + 1 == components.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"required_tensor_count\": " << required_tensors.size() << ",\n";
    std::cout << "  \"required_tensors\": [\n";
    for (size_t i = 0; i < required_tensors.size(); ++i) {
        print_tensor_contract_json(*required_tensors[i], i + 1 != required_tensors.size());
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
    return true;
}

[[maybe_unused]] bool tensor_name_has_prefix(const mit2::Bundle& bundle, const std::string& prefix) {
    for (const auto& tensor : bundle.tensors()) {
        if (tensor.name.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

struct CampplusModelContract {
    std::vector<RequiredTensorSpec> specs;
    std::vector<const mit2::TensorInfo*> present_tensors;
    std::vector<std::string> issues;
    bool ok = false;
};

struct S2MelPromptConditionContract {
    std::vector<RequiredTensorSpec> specs;
    std::vector<const mit2::TensorInfo*> present_tensors;
    std::vector<std::string> issues;
    bool ok = false;
};

struct SemanticCodecQuantizeContract {
    std::vector<RequiredTensorSpec> specs;
    std::vector<const mit2::TensorInfo*> present_tensors;
    std::vector<std::string> issues;
    bool ok = false;
};

struct W2VBertModelContract {
    std::vector<RequiredTensorSpec> specs;
    std::vector<const mit2::TensorInfo*> present_tensors;
    std::vector<std::string> issues;
    bool ok = false;
};

W2VBertModelContract inspect_w2v_bert_model_contract(const mit2::Bundle& model) {
    W2VBertModelContract contract;
    contract.specs = {
        {"w2v_bert.feature_projection.layer_norm.weight", "w2v_bert", {160}},
        {"w2v_bert.feature_projection.layer_norm.bias", "w2v_bert", {160}},
        {"w2v_bert.feature_projection.projection.weight", "w2v_bert", {1024, 160}},
        {"w2v_bert.feature_projection.projection.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.0.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.0.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.0.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.0.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.0.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.0.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.0.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.0.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.0.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.0.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.0.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.0.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.final_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.0.final_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.1.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.1.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.1.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.1.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.1.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.1.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.1.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.1.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.1.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.1.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.1.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.1.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.final_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.1.final_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.2.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.2.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.2.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.2.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.2.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.2.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.2.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.2.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.2.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.2.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.2.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.2.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.2.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.2.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.2.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.3.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.3.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.3.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.3.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.3.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.3.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.3.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.3.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.3.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.3.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.3.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.3.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.3.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.3.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.final_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.3.final_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.4.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.4.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.4.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.4.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.4.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.4.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.4.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.4.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.4.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.4.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.4.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.4.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.4.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.4.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.4.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.5.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.5.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.5.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.5.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.5.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.5.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.5.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.5.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.5.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.5.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.5.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.5.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.5.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.5.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.5.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.6.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.6.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.6.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.6.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.6.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.6.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.6.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.6.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.6.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.6.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.6.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.6.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.6.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.6.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.6.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.7.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.7.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.7.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.7.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.7.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.7.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.7.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.7.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.7.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.7.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.7.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.7.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.7.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.7.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.7.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.8.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.8.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.8.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.8.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.8.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.8.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.8.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.8.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.8.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.8.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.8.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.8.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.8.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.8.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.8.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.9.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.9.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.9.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.9.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.9.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.9.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.9.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.9.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.9.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.9.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.9.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.9.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.9.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.9.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.9.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.10.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.10.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.10.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.10.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.10.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.10.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.10.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.10.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.10.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.10.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.10.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.10.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.10.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.11.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.11.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.11.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.11.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.11.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.11.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.11.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.11.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.11.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.11.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.11.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.11.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.11.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.12.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.12.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.12.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.12.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.12.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.12.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.12.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.12.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.12.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.12.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.12.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.12.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.12.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.13.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.13.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.13.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.self_attn.linear_q.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.13.self_attn.linear_q.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.self_attn.linear_k.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.13.self_attn.linear_k.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.self_attn.linear_v.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.13.self_attn.linear_v.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.self_attn.linear_out.weight", "w2v_bert", {1024, 1024}},
        {"w2v_bert.encoder.layers.13.self_attn_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.self_attn_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.conv_module.layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.conv_module.layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.weight", "w2v_bert", {2048, 1024, 1}},
        {"w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.bias", "w2v_bert", {2048}},
        {"w2v_bert.encoder.layers.13.conv_module.depthwise_conv.weight", "w2v_bert", {1024, 1, 31}},
        {"w2v_bert.encoder.layers.13.conv_module.depthwise_conv.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.weight", "w2v_bert", {1024, 1024, 1}},
        {"w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.ffn2_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.ffn2_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.13.ffn2.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.13.ffn2.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.13.ffn2.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.13.ffn2.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.14.ffn1_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.14.ffn1_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.14.ffn1.intermediate_dense.weight", "w2v_bert", {4096, 1024}},
        {"w2v_bert.encoder.layers.14.ffn1.intermediate_dense.bias", "w2v_bert", {4096}},
        {"w2v_bert.encoder.layers.14.ffn1.output_dense.weight", "w2v_bert", {1024, 4096}},
        {"w2v_bert.encoder.layers.14.ffn1.output_dense.bias", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.17.final_layer_norm.weight", "w2v_bert", {1024}},
        {"w2v_bert.encoder.layers.17.final_layer_norm.bias", "w2v_bert", {1024}},
        {"w2v_bert.stats.mean", "w2v_bert", {1024}},
        {"w2v_bert.stats.std", "w2v_bert", {1024}},
    };
    contract.present_tensors.reserve(contract.specs.size());
    for (const auto& spec : contract.specs) {
        const auto* info = model.find(spec.name);
        if (!info) {
            contract.issues.push_back("missing_w2v_bert_tensor:" + spec.name);
            continue;
        }
        if (info->dtype != "f32") {
            contract.issues.push_back("w2v_bert_tensor_dtype_mismatch:" + spec.name);
            continue;
        }
        if (info->component != spec.component) {
            contract.issues.push_back("w2v_bert_tensor_component_mismatch:" + spec.name);
            continue;
        }
        if (info->shape != spec.shape) {
            contract.issues.push_back("w2v_bert_tensor_shape_mismatch:" + spec.name);
            continue;
        }
        contract.present_tensors.push_back(info);
    }
    contract.ok = contract.issues.empty();
    return contract;
}

SemanticCodecQuantizeContract inspect_semantic_codec_quantize_contract(const mit2::Bundle& model) {
    SemanticCodecQuantizeContract contract;
    contract.specs = {
        {"semantic_codec.quantizer.quantizers.0.in_project.weight_g", "semantic_codec", {8, 1, 1}},
        {"semantic_codec.quantizer.quantizers.0.in_project.weight_v", "semantic_codec", {8, 1024, 1}},
        {"semantic_codec.quantizer.quantizers.0.in_project.bias", "semantic_codec", {8}},
        {"semantic_codec.quantizer.quantizers.0.codebook.weight", "semantic_codec", {8192, 8}},
        {"semantic_codec.quantizer.quantizers.0.out_project.weight_g", "semantic_codec", {1024, 1, 1}},
        {"semantic_codec.quantizer.quantizers.0.out_project.weight_v", "semantic_codec", {1024, 8, 1}},
        {"semantic_codec.quantizer.quantizers.0.out_project.bias", "semantic_codec", {1024}},
    };
    contract.present_tensors.reserve(contract.specs.size());
    for (const auto& spec : contract.specs) {
        const auto* info = model.find(spec.name);
        if (!info) {
            contract.issues.push_back("missing_semantic_codec_quantize_tensor:" + spec.name);
            continue;
        }
        if (info->dtype != "f32") {
            contract.issues.push_back("semantic_codec_quantize_tensor_dtype_mismatch:" + spec.name);
            continue;
        }
        if (info->component != spec.component) {
            contract.issues.push_back("semantic_codec_quantize_tensor_component_mismatch:" + spec.name);
            continue;
        }
        if (info->shape != spec.shape) {
            contract.issues.push_back("semantic_codec_quantize_tensor_shape_mismatch:" + spec.name);
            continue;
        }
        contract.present_tensors.push_back(info);
    }
    contract.ok = contract.issues.empty();
    return contract;
}

S2MelPromptConditionContract inspect_s2mel_prompt_condition_contract(const mit2::Bundle& model) {
    S2MelPromptConditionContract contract;
    contract.specs = {
        {"s2mel.net.length_regulator.content_in_proj.weight", "s2mel", {512, 1024}},
        {"s2mel.net.length_regulator.content_in_proj.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.0.weight", "s2mel", {512, 512, 3}},
        {"s2mel.net.length_regulator.model.0.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.1.weight", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.1.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.3.weight", "s2mel", {512, 512, 3}},
        {"s2mel.net.length_regulator.model.3.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.4.weight", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.4.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.6.weight", "s2mel", {512, 512, 3}},
        {"s2mel.net.length_regulator.model.6.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.7.weight", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.7.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.9.weight", "s2mel", {512, 512, 3}},
        {"s2mel.net.length_regulator.model.9.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.10.weight", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.10.bias", "s2mel", {512}},
        {"s2mel.net.length_regulator.model.12.weight", "s2mel", {512, 512, 1}},
        {"s2mel.net.length_regulator.model.12.bias", "s2mel", {512}},
    };
    contract.present_tensors.reserve(contract.specs.size());
    for (const auto& spec : contract.specs) {
        const auto* info = model.find(spec.name);
        if (!info) {
            contract.issues.push_back("missing_s2mel_prompt_condition_tensor:" + spec.name);
            continue;
        }
        if (info->dtype != "f32") {
            contract.issues.push_back("s2mel_prompt_condition_tensor_dtype_mismatch:" + spec.name);
            continue;
        }
        if (info->component != spec.component) {
            contract.issues.push_back("s2mel_prompt_condition_tensor_component_mismatch:" + spec.name);
            continue;
        }
        if (info->shape != spec.shape) {
            contract.issues.push_back("s2mel_prompt_condition_tensor_shape_mismatch:" + spec.name);
            continue;
        }
        contract.present_tensors.push_back(info);
    }
    contract.ok = contract.issues.empty();
    return contract;
}

CampplusModelContract inspect_campplus_model_contract(const mit2::Bundle& model) {
    CampplusModelContract contract;
    contract.specs = {
        {"campplus.head.conv1.weight", "campplus", {32, 1, 3, 3}},
        {"campplus.head.bn1.weight", "campplus", {32}},
        {"campplus.head.bn1.running_mean", "campplus", {32}},
        {"campplus.head.layer1.0.conv1.weight", "campplus", {32, 32, 3, 3}},
        {"campplus.head.layer2.0.shortcut.0.weight", "campplus", {32, 32, 1, 1}},
        {"campplus.xvector.tdnn.linear.weight", "campplus", {128, 320, 5}},
        {"campplus.xvector.block1.tdnnd1.linear1.weight", "campplus", {128, 128, 1}},
        {"campplus.xvector.block1.tdnnd1.cam_layer.linear_local.weight", "campplus", {32, 128, 3}},
        {"campplus.xvector.transit1.linear.weight", "campplus", {256, 512, 1}},
        {"campplus.xvector.block2.tdnnd1.linear1.weight", "campplus", {128, 256, 1}},
        {"campplus.xvector.block3.tdnnd1.linear1.weight", "campplus", {128, 512, 1}},
        {"campplus.xvector.block3.tdnnd2.linear1.weight", "campplus", {128, 544, 1}},
        {"campplus.xvector.block3.tdnnd3.linear1.weight", "campplus", {128, 576, 1}},
        {"campplus.xvector.block3.tdnnd4.linear1.weight", "campplus", {128, 608, 1}},
        {"campplus.xvector.block3.tdnnd5.linear1.weight", "campplus", {128, 640, 1}},
        {"campplus.xvector.block3.tdnnd6.linear1.weight", "campplus", {128, 672, 1}},
        {"campplus.xvector.block3.tdnnd7.linear1.weight", "campplus", {128, 704, 1}},
        {"campplus.xvector.block3.tdnnd8.linear1.weight", "campplus", {128, 736, 1}},
        {"campplus.xvector.block3.tdnnd9.linear1.weight", "campplus", {128, 768, 1}},
        {"campplus.xvector.block3.tdnnd10.linear1.weight", "campplus", {128, 800, 1}},
        {"campplus.xvector.block3.tdnnd11.linear1.weight", "campplus", {128, 832, 1}},
        {"campplus.xvector.block3.tdnnd12.linear1.weight", "campplus", {128, 864, 1}},
        {"campplus.xvector.block3.tdnnd13.linear1.weight", "campplus", {128, 896, 1}},
        {"campplus.xvector.block3.tdnnd14.linear1.weight", "campplus", {128, 928, 1}},
        {"campplus.xvector.block3.tdnnd15.linear1.weight", "campplus", {128, 960, 1}},
        {"campplus.xvector.block3.tdnnd16.linear1.weight", "campplus", {128, 992, 1}},
        {"campplus.xvector.transit3.linear.weight", "campplus", {512, 1024, 1}},
        {"campplus.xvector.out_nonlinear.batchnorm.weight", "campplus", {512}},
        {"campplus.xvector.dense.linear.weight", "campplus", {192, 1024, 1}},
        {"campplus.xvector.dense.nonlinear.batchnorm.running_mean", "campplus", {192}},
    };
    contract.present_tensors.reserve(contract.specs.size());
    for (const auto& spec : contract.specs) {
        const auto* info = model.find(spec.name);
        if (!info) {
            contract.issues.push_back("missing_campplus_tensor:" + spec.name);
            continue;
        }
        if (info->dtype != "f32") {
            contract.issues.push_back("campplus_tensor_dtype_mismatch:" + spec.name);
            continue;
        }
        if (info->component != spec.component) {
            contract.issues.push_back("campplus_tensor_component_mismatch:" + spec.name);
            continue;
        }
        if (info->shape != spec.shape) {
            contract.issues.push_back("campplus_tensor_shape_mismatch:" + spec.name);
            continue;
        }
        contract.present_tensors.push_back(info);
    }
    contract.ok = contract.issues.empty();
    return contract;
}

[[maybe_unused]] bool inspect_tts_clone_encoder_model_readiness(const std::string& model_bundle_dir) {
    try {
        mit2::Bundle model(model_bundle_dir);
        const auto integrity = validate_bundle_integrity(model, true);
        std::unordered_map<std::string, uint64_t> component_counts;
        std::unordered_map<std::string, uint64_t> component_bytes;
        for (const auto& tensor : model.tensors()) {
            component_counts[tensor.component] += 1;
            component_bytes[tensor.component] += tensor.nbytes;
        }

        const auto campplus = inspect_campplus_model_contract(model);
        const auto w2v_bert = inspect_w2v_bert_model_contract(model);
        const auto semantic_quantize = inspect_semantic_codec_quantize_contract(model);
        const auto prompt_condition = inspect_s2mel_prompt_condition_contract(model);
        const bool has_campplus = campplus.ok;
        const bool has_w2v_bert = w2v_bert.ok;
        const bool has_semantic_codec_quantize = semantic_quantize.ok;
        const bool has_s2mel_prompt_condition = prompt_condition.ok;
        const bool has_hot_vq2emb =
            model.find("semantic_codec.quantizer.quantizers.0.codebook.weight") != nullptr &&
            model.find("semantic_codec.quantizer.quantizers.0.out_project.weight_v") != nullptr;

        std::vector<std::string> issues;
        if (!has_campplus) {
            issues.push_back("missing_clone_encoder_component_campplus");
        }
        if (!has_w2v_bert) {
            issues.push_back("missing_clone_encoder_component_w2v_bert");
        }
        if (!has_semantic_codec_quantize) {
            issues.push_back("missing_native_semantic_codec_quantize_contract");
        }
        if (!has_s2mel_prompt_condition) {
            issues.push_back("missing_s2mel_prompt_condition_contract");
        }
        const bool ok = issues.empty();

        const std::vector<std::string> clone_components{
            "campplus",
            "w2v_bert",
            "semantic_model",
            "semantic_codec",
            "s2mel",
            "gpt",
            "bigvgan",
        };

        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_encoder_model_readiness\",\n";
        std::cout << "  \"ok\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"format\": \"MIT2\",\n";
        std::cout << "  \"version\": " << model.version() << ",\n";
        std::cout << "  \"weights_bytes\": " << model.weights_size() << ",\n";
        std::cout << "  \"tensor_count\": " << model.tensor_count() << ",\n";
        std::cout << "  \"tensor_bytes\": " << model.total_tensor_bytes() << ",\n";
        std::cout << "  \"integrity\": {\n";
        std::cout << "    \"aligned_tensor_count\": " << integrity.aligned_tensor_count << ",\n";
        std::cout << "    \"checked_interval_count\": " << integrity.checked_interval_count << ",\n";
        std::cout << "    \"sha256_verified_count\": " << integrity.checksum_verified_count << "\n";
        std::cout << "  },\n";
        std::cout << "  \"component_counts\": {\n";
        for (size_t i = 0; i < clone_components.size(); ++i) {
            const auto& component = clone_components[i];
            std::cout << "    \"" << component << "\": " << component_counts[component];
            std::cout << (i + 1 == clone_components.size() ? "\n" : ",\n");
        }
        std::cout << "  },\n";
        std::cout << "  \"component_bytes\": {\n";
        for (size_t i = 0; i < clone_components.size(); ++i) {
            const auto& component = clone_components[i];
            std::cout << "    \"" << component << "\": " << component_bytes[component];
            std::cout << (i + 1 == clone_components.size() ? "\n" : ",\n");
        }
        std::cout << "  },\n";
        std::cout << "  \"clone_encoder_model_issues\": ";
        print_json_string_array(issues);
        std::cout << ",\n";
        std::cout << "  \"available_clone_encoder_components\": [\n";
        bool printed_component = false;
        for (const auto& component : std::vector<std::string>{"campplus", "w2v_bert", "semantic_model"}) {
            if (component_counts[component] > 0) {
                if (printed_component) {
                    std::cout << ",\n";
                }
                std::cout << "    \"" << component << "\"";
                printed_component = true;
            }
        }
        if (printed_component) {
            std::cout << "\n";
        }
        std::cout << "  ],\n";
        std::cout << "  \"available_hot_components\": [\n";
        const std::vector<std::string> hot_components{"gpt", "semantic_codec", "s2mel", "bigvgan"};
        for (size_t i = 0; i < hot_components.size(); ++i) {
            const auto& component = hot_components[i];
            std::cout << "    \"" << component << "\"";
            std::cout << (i + 1 == hot_components.size() ? "\n" : ",\n");
        }
        std::cout << "  ],\n";
        std::cout << "  \"available_hot_semantic_codec_vq2emb\": " << (has_hot_vq2emb ? "true" : "false") << ",\n";
        std::cout << "  \"has_campplus_model_contract\": " << (has_campplus ? "true" : "false") << ",\n";
        std::cout << "  \"campplus_required_tensor_count\": " << campplus.specs.size() << ",\n";
        std::cout << "  \"campplus_required_tensors_present\": " << campplus.present_tensors.size() << ",\n";
        std::cout << "  \"campplus_contract_issues\": ";
        print_json_string_array(campplus.issues);
        std::cout << ",\n";
        std::cout << "  \"campplus_required_tensors\": [\n";
        for (size_t i = 0; i < campplus.present_tensors.size(); ++i) {
            print_tensor_contract_json(*campplus.present_tensors[i], i + 1 != campplus.present_tensors.size());
        }
        std::cout << "  ],\n";
        std::cout << "  \"has_w2v_bert_model_contract\": " << (has_w2v_bert ? "true" : "false") << ",\n";
        std::cout << "  \"w2v_bert_required_tensor_count\": " << w2v_bert.specs.size() << ",\n";
        std::cout << "  \"w2v_bert_required_tensors_present\": " << w2v_bert.present_tensors.size() << ",\n";
        std::cout << "  \"w2v_bert_contract_issues\": ";
        print_json_string_array(w2v_bert.issues);
        std::cout << ",\n";
        std::cout << "  \"w2v_bert_required_tensors\": [\n";
        for (size_t i = 0; i < w2v_bert.present_tensors.size(); ++i) {
            print_tensor_contract_json(*w2v_bert.present_tensors[i], i + 1 != w2v_bert.present_tensors.size());
        }
        std::cout << "  ],\n";
        std::cout << "  \"has_semantic_codec_quantize_contract\": " << (has_semantic_codec_quantize ? "true" : "false") << ",\n";
        std::cout << "  \"semantic_codec_quantize_required_tensor_count\": " << semantic_quantize.specs.size() << ",\n";
        std::cout << "  \"semantic_codec_quantize_required_tensors_present\": " << semantic_quantize.present_tensors.size() << ",\n";
        std::cout << "  \"semantic_codec_quantize_contract_issues\": ";
        print_json_string_array(semantic_quantize.issues);
        std::cout << ",\n";
        std::cout << "  \"has_s2mel_prompt_encoder_contract\": " << (has_s2mel_prompt_condition ? "true" : "false") << ",\n";
        std::cout << "  \"has_s2mel_prompt_condition_contract\": " << (has_s2mel_prompt_condition ? "true" : "false") << ",\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensor_count\": " << prompt_condition.specs.size() << ",\n";
        std::cout << "  \"s2mel_prompt_condition_required_tensors_present\": " << prompt_condition.present_tensors.size() << ",\n";
        std::cout << "  \"s2mel_prompt_condition_contract_issues\": ";
        print_json_string_array(prompt_condition.issues);
        std::cout << ",\n";
        std::cout << "  \"required_clone_encoder_models\": [\n";
        std::cout << "    {\"component\": \"campplus\", \"produces\": \"s2mel_style\", \"reference\": \"funasr/campplus campplus_cn_common.bin\", \"required_tensor_prefix\": \"campplus.\"},\n";
        std::cout << "    {\"component\": \"w2v_bert\", \"produces\": \"semantic features before MaskGCT quantize\", \"reference\": \"facebook/w2v-bert-2.0\", \"required_tensor_prefix\": \"w2v_bert.\"},\n";
        std::cout << "    {\"component\": \"semantic_codec_quantize\", \"produces\": \"continuous S_ref plus semantic code indices from spk_cond_emb\", \"reference\": \"MaskGCT semantic_codec.quantize\", \"required_tensor_prefix\": \"semantic_codec.quantizer.quantizers.0.\"},\n";
        std::cout << "    {\"component\": \"s2mel_prompt_condition\", \"produces\": \"s2mel_prompt\", \"reference\": \"IndexTTS2 s2mel.models['length_regulator'](S_ref, ylens=mel_frames)\", \"required_tensor_prefix\": \"s2mel.net.length_regulator.\"}\n";
        std::cout << "  ],\n";
        std::cout << "  \"ready_native_clone_encoder_models\": " << (ok ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_w2v_bert_model_contract\": " << (has_w2v_bert ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_semantic_codec_quantize_from_spk_cond\": " << (has_semantic_codec_quantize ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_s2mel_prompt_condition_from_sref\": " << (has_s2mel_prompt_condition ? "true" : "false") << ",\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"next_native_boundary\": \"native W2V-BERT semantic features to produce spk_cond_emb from clone audio\"\n";
        std::cout << "}\n";
        return ok;
    } catch (const std::exception& e) {
        std::cout << "{\n";
        std::cout << "  \"stage\": \"tts_clone_encoder_model_readiness\",\n";
        std::cout << "  \"ok\": false,\n";
        std::cout << "  \"product_surface_version\": 1,\n";
        std::cout << "  \"binary\": \"mit2_tts\",\n";
        std::cout << "  \"model_bundle_dir\": \"" << json_escape(model_bundle_dir) << "\",\n";
        std::cout << "  \"ready_native_clone_encoder_models\": false,\n";
        std::cout << "  \"ready_native_voice_clone\": false,\n";
        std::cout << "  \"error\": \"" << json_escape(e.what()) << "\"\n";
        std::cout << "}\n";
        return false;
    }
}

bool inspect_voice_bundle_contract(const std::string& voice_bundle_dir) {
    mit2::Bundle voice(voice_bundle_dir);
    const auto integrity = validate_bundle_integrity(voice, true);
    const auto& spk = require_voice_tensor(voice, "spk_cond_emb");
    const auto& style = require_voice_tensor(voice, "s2mel_style");
    const auto& prompt = require_voice_tensor(voice, "s2mel_prompt");
    const auto& mel = require_voice_tensor(voice, "mel");

    constexpr int64_t semantic_dim = 1024;
    constexpr int64_t style_dim = 192;
    constexpr int64_t cond_dim = 512;
    constexpr int64_t mel_dim = 80;
    if (spk.shape.size() != 3 || spk.shape[0] != 1 || spk.shape[1] <= 0 || spk.shape[2] != semantic_dim) {
        throw std::runtime_error("voice spk_cond_emb must have shape [1,tokens>0,1024]");
    }
    if (style.shape.size() != 2 || style.shape[0] != 1 || style.shape[1] != style_dim) {
        throw std::runtime_error("voice s2mel_style must have shape [1,192]");
    }
    if (prompt.shape.size() != 3 || prompt.shape[0] != 1 || prompt.shape[1] <= 0 || prompt.shape[2] != cond_dim) {
        throw std::runtime_error("voice s2mel_prompt must have shape [1,tokens>0,512]");
    }
    if (mel.shape.size() != 3 || mel.shape[0] != 1 || mel.shape[1] != mel_dim || mel.shape[2] <= 0) {
        throw std::runtime_error("voice mel must have shape [1,80,tokens>0]");
    }
    if (prompt.shape[1] != mel.shape[2]) {
        throw std::runtime_error("voice prompt token count must match mel frame count");
    }

    std::cout << "{\n";
    std::cout << "  \"stage\": \"voice_bundle_contract\",\n";
    std::cout << "  \"voice_bundle_dir\": \"" << json_escape(voice_bundle_dir) << "\",\n";
    std::cout << "  \"ok\": true,\n";
    std::cout << "  \"format\": \"MIT2\",\n";
    std::cout << "  \"version\": " << voice.version() << ",\n";
    std::cout << "  \"endianness\": \"" << json_escape(voice.endianness()) << "\",\n";
    std::cout << "  \"alignment\": " << voice.alignment() << ",\n";
    std::cout << "  \"weights_file\": \"" << json_escape(voice.weights_file()) << "\",\n";
    std::cout << "  \"weights_bytes\": " << voice.weights_size() << ",\n";
    std::cout << "  \"tensor_count\": " << voice.tensor_count() << ",\n";
    std::cout << "  \"tensor_bytes\": " << voice.total_tensor_bytes() << ",\n";
    std::cout << "  \"integrity\": {\n";
    std::cout << "    \"aligned_tensor_count\": " << integrity.aligned_tensor_count << ",\n";
    std::cout << "    \"checked_interval_count\": " << integrity.checked_interval_count << ",\n";
    std::cout << "    \"sha256_verified_count\": " << integrity.checksum_verified_count << "\n";
    std::cout << "  },\n";
    std::cout << "  \"spk_cond_tokens\": " << spk.shape[1] << ",\n";
    std::cout << "  \"prompt_tokens\": " << prompt.shape[1] << ",\n";
    std::cout << "  \"mel_frames\": " << mel.shape[2] << ",\n";
    std::cout << "  \"required_tensors\": [\n";
    print_tensor_contract_json(spk, true);
    print_tensor_contract_json(style, true);
    print_tensor_contract_json(prompt, true);
    print_tensor_contract_json(mel, false);
    std::cout << "  ]\n";
    std::cout << "}\n";
    return true;
}

uint32_t infer_voice_prompt_tokens(const mit2::Bundle& voice) {
    const auto* prompt = voice.find("s2mel_prompt");
    const auto* mel = voice.find("mel");

    constexpr int64_t cond_dim = 512;
    constexpr int64_t mel_dim = 80;
    if (!prompt || prompt->shape.size() != 3 || prompt->shape[0] != 1 || prompt->shape[1] <= 0 ||
        prompt->shape[2] != cond_dim) {
        throw std::runtime_error("voice s2mel_prompt must have shape [1,tokens>0,512]");
    }
    if (!mel || mel->shape.size() != 3 || mel->shape[0] != 1 || mel->shape[1] != mel_dim || mel->shape[2] <= 0) {
        throw std::runtime_error("voice mel must have shape [1,80,tokens>0]");
    }
    if (prompt->shape[1] != mel->shape[2]) {
        throw std::runtime_error("voice prompt token count must match mel frame count");
    }
    return static_cast<uint32_t>(prompt->shape[1]);
}

uint32_t resolve_voice_prompt_tokens(const mit2::Bundle& voice, uint32_t requested_prompt_tokens) {
    if (requested_prompt_tokens != 0) {
        return requested_prompt_tokens;
    }
    return infer_voice_prompt_tokens(voice);
}

std::vector<float> run_gpt_layer_cpu(const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto w0 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.0.weight");
    auto b0 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.0.bias");
    auto w1 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.1.weight");
    auto b1 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.1.bias");
    auto w2 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.2.weight");
    auto b2 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.2.bias");
    std::vector<float> out(tokens * 1024);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::vector<float> x0(input.begin() + t * 1280, input.begin() + (t + 1) * 1280);
        auto x1 = cpu_linear(w0, b0, x0, 256, 1280);
        auto x2 = cpu_linear(w1, b1, x1, 128, 256);
        auto x3 = cpu_linear(w2, b2, x2, 1024, 128);
        std::copy(x3.begin(), x3.end(), out.begin() + t * 1024);
    }
    return out;
}

std::vector<float> run_gpt_layer_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t tokens) {
    auto w0 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.0.weight");
    auto b0 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.0.bias");
    auto w1 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.1.weight");
    auto b1 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.1.bias");
    auto w2 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.2.weight");
    auto b2 = tensor_as_f32(bundle, "s2mel.net.gpt_layer.2.bias");
    auto x1 = metal.linear_rows_f32_resident(
        "s2mel.net.gpt_layer.0.weight.resident",
        w0,
        "s2mel.net.gpt_layer.0.bias.resident",
        b0,
        input,
        tokens,
        256,
        1280);
    auto x2 = metal.linear_rows_f32_resident(
        "s2mel.net.gpt_layer.1.weight.resident",
        w1,
        "s2mel.net.gpt_layer.1.bias.resident",
        b1,
        x1,
        tokens,
        128,
        256);
    return metal.linear_rows_f32_resident(
        "s2mel.net.gpt_layer.2.weight.resident",
        w2,
        "s2mel.net.gpt_layer.2.bias.resident",
        b2,
        x2,
        tokens,
        1024,
        128);
}
