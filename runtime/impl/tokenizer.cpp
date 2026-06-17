struct TokenizerPieceInfo {
    uint32_t id = 0;
    float score = 0.0f;
    bool has_score = false;
};

// Fallback dictionary for punctuation pieces missing from the tokenizer vocab.
// Loaded from tokenizer/punct_fallback.tsv: each line is "<piece>\t<replacement>";
// an empty replacement drops the piece. Pieces with no entry are skipped with a
// warning instead of aborting synthesis.
struct PunctFallbackConfig {
    std::unordered_map<std::string, std::string> map;
    bool loaded = false;
};

PunctFallbackConfig& punct_fallback_config() {
    static PunctFallbackConfig config;
    return config;
}

void load_punct_fallback(const std::string& tokenizer_dir) {
    auto& config = punct_fallback_config();
    if (config.loaded) {
        return;
    }
    config.loaded = true;
    std::ifstream fp(tokenizer_dir + "/punct_fallback.tsv");
    if (!fp) {
        return;  // optional file; absence means warn-and-skip for all unknown pieces
    }
    std::string line;
    while (std::getline(fp, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            config.map[line] = "";
        } else {
            config.map[line.substr(0, tab)] = line.substr(tab + 1);
        }
    }
}

std::unordered_map<std::string, TokenizerPieceInfo> load_tokenizer_pieces(const std::string& tokenizer_dir) {
    load_punct_fallback(tokenizer_dir);
    const std::string path = tokenizer_dir + "/pieces.tsv";
    std::ifstream fp(path);
    if (!fp) {
        throw std::runtime_error("failed to open tokenizer pieces file: " + path);
    }
    std::unordered_map<std::string, TokenizerPieceInfo> piece_to_id;
    std::string line;
    while (std::getline(fp, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("malformed tokenizer pieces line: " + line);
        }
        const size_t score_tab = line.find('\t', tab + 1);
        const uint32_t token_id = static_cast<uint32_t>(std::stoul(line.substr(0, tab)));
        const std::string piece = score_tab == std::string::npos ? line.substr(tab + 1) : line.substr(tab + 1, score_tab - tab - 1);
        TokenizerPieceInfo info;
        info.id = token_id;
        if (score_tab != std::string::npos) {
            info.score = std::stof(line.substr(score_tab + 1));
            info.has_score = true;
        }
        piece_to_id[piece] = info;
    }
    return piece_to_id;
}

bool utf8_next(const std::string& text, size_t& pos, uint32_t& codepoint, std::string& bytes) {
    if (pos >= text.size()) {
        return false;
    }
    const unsigned char lead = static_cast<unsigned char>(text[pos]);
    size_t len = 0;
    uint32_t cp = 0;
    if (lead < 0x80) {
        len = 1;
        cp = lead;
    } else if ((lead & 0xe0) == 0xc0) {
        len = 2;
        cp = lead & 0x1f;
    } else if ((lead & 0xf0) == 0xe0) {
        len = 3;
        cp = lead & 0x0f;
    } else if ((lead & 0xf8) == 0xf0) {
        len = 4;
        cp = lead & 0x07;
    } else {
        throw std::runtime_error("invalid utf-8 leading byte in tokenizer input");
    }
    if (pos + len > text.size()) {
        throw std::runtime_error("truncated utf-8 in tokenizer input");
    }
    for (size_t i = 1; i < len; ++i) {
        const unsigned char cont = static_cast<unsigned char>(text[pos + i]);
        if ((cont & 0xc0) != 0x80) {
            throw std::runtime_error("invalid utf-8 continuation byte in tokenizer input");
        }
        cp = (cp << 6) | (cont & 0x3f);
    }
    bytes = text.substr(pos, len);
    codepoint = cp;
    pos += len;
    return true;
}

bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x3400 && cp <= 0x4dbf) ||
           (cp >= 0x4e00 && cp <= 0x9fff) ||
           (cp >= 0xf900 && cp <= 0xfaff) ||
           (cp >= 0x20000 && cp <= 0x2ebef);
}

// Maps CJK Radicals Supplement (U+2E80–U+2EFF), Kangxi Radicals (U+2F00–U+2FDF),
// and CJK Strokes (U+31C0–U+31EF) lookalike characters to their canonical unified
// ideograph, per Unicode's Equivalent_Unified_Ideograph property. These glyphs are
// visually identical to ordinary Hanzi but are distinct codepoints absent from the
// tokenizer vocab, so callers occasionally paste them in place of the real character.
uint32_t equivalent_unified_ideograph(uint32_t cp) {
    if (cp < 0x2E80 || cp > 0x31E1) {
        return 0;
    }
    static const std::unordered_map<uint32_t, uint32_t> table = {
    {0x2E81,0x5382}, {0x2E82,0x4E5B}, {0x2E83,0x4E5A}, {0x2E84,0x4E59}, {0x2E85,0x4EBB}, {0x2E86,0x5182},
    {0x2E87,0x20628}, {0x2E88,0x5200}, {0x2E89,0x5202}, {0x2E8A,0x535C}, {0x2E8B,0x353E}, {0x2E8C,0x5C0F},
    {0x2E8D,0x5C0F}, {0x2E8E,0x5140}, {0x2E8F,0x5C23}, {0x2E90,0x5C22}, {0x2E91,0x21BC2}, {0x2E92,0x5DF3},
    {0x2E93,0x5E7A}, {0x2E94,0x5F51}, {0x2E95,0x2B739}, {0x2E96,0x5FC4}, {0x2E97,0x5FC3}, {0x2E98,0x624C},
    {0x2E99,0x6535}, {0x2E9B,0x65E1}, {0x2E9C,0x65E5}, {0x2E9D,0x6708}, {0x2E9E,0x6B7A}, {0x2E9F,0x6BCD},
    {0x2EA0,0x6C11}, {0x2EA1,0x6C35}, {0x2EA2,0x6C3A}, {0x2EA3,0x706C}, {0x2EA4,0x722B}, {0x2EA5,0x722B},
    {0x2EA6,0x4E2C}, {0x2EA7,0x725B}, {0x2EA8,0x72AD}, {0x2EA9,0x738B}, {0x2EAA,0x24D14}, {0x2EAB,0x76EE},
    {0x2EAC,0x793A}, {0x2EAD,0x793B}, {0x2EAE,0x25AD7}, {0x2EAF,0x7CF9}, {0x2EB0,0x7E9F}, {0x2EB1,0x7F53},
    {0x2EB2,0x7F52}, {0x2EB3,0x34C1}, {0x2EB4,0x5197}, {0x2EB5,0x2626B}, {0x2EB6,0x7F8A}, {0x2EB7,0x2634C},
    {0x2EB8,0x2634B}, {0x2EB9,0x8002}, {0x2EBA,0x8080}, {0x2EBB,0x807F}, {0x2EBC,0x8089}, {0x2EBD,0x26951},
    {0x2EBE,0x8279}, {0x2EBF,0x8279}, {0x2EC0,0x8279}, {0x2EC1,0x864E}, {0x2EC2,0x8864}, {0x2EC3,0x8980},
    {0x2EC4,0x897F}, {0x2EC5,0x89C1}, {0x2EC6,0x89D2}, {0x2EC7,0x278B2}, {0x2EC8,0x8BA0}, {0x2EC9,0x8D1D},
    {0x2ECA,0x27FB7}, {0x2ECB,0x8F66}, {0x2ECC,0x8FB6}, {0x2ECD,0x8FB6}, {0x2ECE,0x8FB6}, {0x2ECF,0x9091},
    {0x2ED0,0x9485}, {0x2ED1,0x9577}, {0x2ED2,0x9578}, {0x2ED3,0x957F}, {0x2ED4,0x95E8}, {0x2ED5,0x28E0F},
    {0x2ED6,0x961D}, {0x2ED7,0x96E8}, {0x2ED8,0x9752}, {0x2ED9,0x97E6}, {0x2EDA,0x9875}, {0x2EDB,0x98CE},
    {0x2EDC,0x98DE}, {0x2EDD,0x98DF}, {0x2EDE,0x2967F}, {0x2EDF,0x98E0}, {0x2EE0,0x9963}, {0x2EE1,0x29810},
    {0x2EE2,0x9A6C}, {0x2EE3,0x9AA8}, {0x2EE4,0x9B3C}, {0x2EE5,0x9C7C}, {0x2EE6,0x9E1F}, {0x2EE7,0x5364},
    {0x2EE8,0x9EA6}, {0x2EE9,0x9EC4}, {0x2EEA,0x9EFE}, {0x2EEB,0x6589}, {0x2EEC,0x9F50}, {0x2EED,0x6B6F},
    {0x2EEE,0x9F7F}, {0x2EEF,0x7ADC}, {0x2EF0,0x9F99}, {0x2EF1,0x9F9C}, {0x2EF2,0x4E80}, {0x2EF3,0x9F9F},
    {0x2F00,0x4E00}, {0x2F01,0x4E28}, {0x2F02,0x4E36}, {0x2F03,0x4E3F}, {0x2F04,0x4E59}, {0x2F05,0x4E85},
    {0x2F06,0x4E8C}, {0x2F07,0x4EA0}, {0x2F08,0x4EBA}, {0x2F09,0x513F}, {0x2F0A,0x5165}, {0x2F0B,0x516B},
    {0x2F0C,0x5182}, {0x2F0D,0x5196}, {0x2F0E,0x51AB}, {0x2F0F,0x51E0}, {0x2F10,0x51F5}, {0x2F11,0x5200},
    {0x2F12,0x529B}, {0x2F13,0x52F9}, {0x2F14,0x5315}, {0x2F15,0x531A}, {0x2F16,0x5338}, {0x2F17,0x5341},
    {0x2F18,0x535C}, {0x2F19,0x5369}, {0x2F1A,0x5382}, {0x2F1B,0x53B6}, {0x2F1C,0x53C8}, {0x2F1D,0x53E3},
    {0x2F1E,0x56D7}, {0x2F1F,0x571F}, {0x2F20,0x58EB}, {0x2F21,0x5902}, {0x2F22,0x590A}, {0x2F23,0x5915},
    {0x2F24,0x5927}, {0x2F25,0x5973}, {0x2F26,0x5B50}, {0x2F27,0x5B80}, {0x2F28,0x5BF8}, {0x2F29,0x5C0F},
    {0x2F2A,0x5C22}, {0x2F2B,0x5C38}, {0x2F2C,0x5C6E}, {0x2F2D,0x5C71}, {0x2F2E,0x5DDB}, {0x2F2F,0x5DE5},
    {0x2F30,0x5DF1}, {0x2F31,0x5DFE}, {0x2F32,0x5E72}, {0x2F33,0x5E7A}, {0x2F34,0x5E7F}, {0x2F35,0x5EF4},
    {0x2F36,0x5EFE}, {0x2F37,0x5F0B}, {0x2F38,0x5F13}, {0x2F39,0x5F50}, {0x2F3A,0x5F61}, {0x2F3B,0x5F73},
    {0x2F3C,0x5FC3}, {0x2F3D,0x6208}, {0x2F3E,0x6236}, {0x2F3F,0x624B}, {0x2F40,0x652F}, {0x2F41,0x6534},
    {0x2F42,0x6587}, {0x2F43,0x6597}, {0x2F44,0x65A4}, {0x2F45,0x65B9}, {0x2F46,0x65E0}, {0x2F47,0x65E5},
    {0x2F48,0x66F0}, {0x2F49,0x6708}, {0x2F4A,0x6728}, {0x2F4B,0x6B20}, {0x2F4C,0x6B62}, {0x2F4D,0x6B79},
    {0x2F4E,0x6BB3}, {0x2F4F,0x6BCB}, {0x2F50,0x6BD4}, {0x2F51,0x6BDB}, {0x2F52,0x6C0F}, {0x2F53,0x6C14},
    {0x2F54,0x6C34}, {0x2F55,0x706B}, {0x2F56,0x722A}, {0x2F57,0x7236}, {0x2F58,0x723B}, {0x2F59,0x723F},
    {0x2F5A,0x7247}, {0x2F5B,0x7259}, {0x2F5C,0x725B}, {0x2F5D,0x72AC}, {0x2F5E,0x7384}, {0x2F5F,0x7389},
    {0x2F60,0x74DC}, {0x2F61,0x74E6}, {0x2F62,0x7518}, {0x2F63,0x751F}, {0x2F64,0x7528}, {0x2F65,0x7530},
    {0x2F66,0x758B}, {0x2F67,0x7592}, {0x2F68,0x7676}, {0x2F69,0x767D}, {0x2F6A,0x76AE}, {0x2F6B,0x76BF},
    {0x2F6C,0x76EE}, {0x2F6D,0x77DB}, {0x2F6E,0x77E2}, {0x2F6F,0x77F3}, {0x2F70,0x793A}, {0x2F71,0x79B8},
    {0x2F72,0x79BE}, {0x2F73,0x7A74}, {0x2F74,0x7ACB}, {0x2F75,0x7AF9}, {0x2F76,0x7C73}, {0x2F77,0x7CF8},
    {0x2F78,0x7F36}, {0x2F79,0x7F51}, {0x2F7A,0x7F8A}, {0x2F7B,0x7FBD}, {0x2F7C,0x8001}, {0x2F7D,0x800C},
    {0x2F7E,0x8012}, {0x2F7F,0x8033}, {0x2F80,0x807F}, {0x2F81,0x8089}, {0x2F82,0x81E3}, {0x2F83,0x81EA},
    {0x2F84,0x81F3}, {0x2F85,0x81FC}, {0x2F86,0x820C}, {0x2F87,0x821B}, {0x2F88,0x821F}, {0x2F89,0x826E},
    {0x2F8A,0x8272}, {0x2F8B,0x8278}, {0x2F8C,0x864D}, {0x2F8D,0x866B}, {0x2F8E,0x8840}, {0x2F8F,0x884C},
    {0x2F90,0x8863}, {0x2F91,0x897E}, {0x2F92,0x898B}, {0x2F93,0x89D2}, {0x2F94,0x8A00}, {0x2F95,0x8C37},
    {0x2F96,0x8C46}, {0x2F97,0x8C55}, {0x2F98,0x8C78}, {0x2F99,0x8C9D}, {0x2F9A,0x8D64}, {0x2F9B,0x8D70},
    {0x2F9C,0x8DB3}, {0x2F9D,0x8EAB}, {0x2F9E,0x8ECA}, {0x2F9F,0x8F9B}, {0x2FA0,0x8FB0}, {0x2FA1,0x8FB5},
    {0x2FA2,0x9091}, {0x2FA3,0x9149}, {0x2FA4,0x91C6}, {0x2FA5,0x91CC}, {0x2FA6,0x91D1}, {0x2FA7,0x9577},
    {0x2FA8,0x9580}, {0x2FA9,0x961C}, {0x2FAA,0x96B6}, {0x2FAB,0x96B9}, {0x2FAC,0x96E8}, {0x2FAD,0x9751},
    {0x2FAE,0x975E}, {0x2FAF,0x9762}, {0x2FB0,0x9769}, {0x2FB1,0x97CB}, {0x2FB2,0x97ED}, {0x2FB3,0x97F3},
    {0x2FB4,0x9801}, {0x2FB5,0x98A8}, {0x2FB6,0x98DB}, {0x2FB7,0x98DF}, {0x2FB8,0x9996}, {0x2FB9,0x9999},
    {0x2FBA,0x99AC}, {0x2FBB,0x9AA8}, {0x2FBC,0x9AD8}, {0x2FBD,0x9ADF}, {0x2FBE,0x9B25}, {0x2FBF,0x9B2F},
    {0x2FC0,0x9B32}, {0x2FC1,0x9B3C}, {0x2FC2,0x9B5A}, {0x2FC3,0x9CE5}, {0x2FC4,0x9E75}, {0x2FC5,0x9E7F},
    {0x2FC6,0x9EA5}, {0x2FC7,0x9EBB}, {0x2FC8,0x9EC3}, {0x2FC9,0x9ECD}, {0x2FCA,0x9ED1}, {0x2FCB,0x9EF9},
    {0x2FCC,0x9EFD}, {0x2FCD,0x9F0E}, {0x2FCE,0x9F13}, {0x2FCF,0x9F20}, {0x2FD0,0x9F3B}, {0x2FD1,0x9F4A},
    {0x2FD2,0x9F52}, {0x2FD3,0x9F8D}, {0x2FD4,0x9F9C}, {0x2FD5,0x9FA0}, {0x31C6,0x200CC}, {0x31CF,0x4E40},
    {0x31D0,0x4E00}, {0x31D1,0x4E28}, {0x31D2,0x4E3F}, {0x31D3,0x4E3F}, {0x31D4,0x4E36}, {0x31D5,0x200CD},
    {0x31D6,0x4E5B}, {0x31D7,0x200CA}, {0x31D8,0x200CE}, {0x31D9,0x2010C}, {0x31DA,0x4E85}, {0x31DB,0x21FE8},
    {0x31DC,0x200CB}, {0x31DD,0x4E40}, {0x31DE,0x200D1}, {0x31DF,0x4E5A}, {0x31E0,0x4E59}, {0x31E1,0x2010E},    };
    const auto it = table.find(cp);
    return it == table.end() ? 0 : it->second;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

// Rewrites radical-block lookalikes to canonical unified ideographs before
// tokenization so all downstream CJK detection and vocab lookup sees real Hanzi.
std::string normalize_cjk_radicals(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    size_t pos = 0;
    uint32_t cp = 0;
    std::string bytes;
    while (utf8_next(text, pos, cp, bytes)) {
        const uint32_t mapped = equivalent_unified_ideograph(cp);
        if (mapped != 0) {
            append_utf8(out, mapped);
        } else {
            out += bytes;
        }
    }
    return out;
}

struct CjkTokenizedText {
    std::vector<std::string> pieces;
    std::vector<uint32_t> ids;
};

struct CjkTokenizedSegment {
    std::vector<std::string> pieces;
    std::vector<uint32_t> ids;
};

// Full native text tokenizer: faithful reimplementation of the reference
// IndexTTS2 Python frontend (TextNormalizer + SentencePiece), replacing the
// narrow hand-rolled `tokenize_cjk_text` for arbitrary input. Loads bpe.model
// and the wetext .fst grammars from <tokenizer_dir> once (process-cached).
CjkTokenizedText tokenize_text_full(const std::string& tokenizer_dir, const std::string& text) {
    const mit2::TextFrontend& frontend = mit2::get_cached_text_frontend(tokenizer_dir);
    mit2::FrontendTokenized t = frontend.tokenize(text);
    CjkTokenizedText out;
    out.pieces = std::move(t.pieces);
    out.ids = std::move(t.ids);
    return out;
}

bool is_ascii_alpha(uint32_t cp) {
    return cp < 128 && std::isalpha(static_cast<unsigned char>(cp)) != 0;
}

bool is_ascii_digit(uint32_t cp) {
    return cp < 128 && std::isdigit(static_cast<unsigned char>(cp)) != 0;
}

bool is_ascii_alpha_byte(char ch) {
    return std::isalpha(static_cast<unsigned char>(ch)) != 0;
}

bool is_ascii_space(uint32_t cp) {
    return cp == ' ' || cp == '\n' || cp == '\r' || cp == '\t';
}

bool is_ascii_punctuation(uint32_t cp) {
    return cp < 128 && std::ispunct(static_cast<unsigned char>(cp)) != 0;
}

bool is_ascii_punctuation_piece(const std::string& piece) {
    const std::string space_piece = "\xe2\x96\x81";
    std::string value = piece;
    if (value.rfind(space_piece, 0) == 0) {
        value = value.substr(space_piece.size());
    }
    return value.size() == 1 && is_ascii_punctuation(static_cast<unsigned char>(value[0]));
}

bool is_fullwidth_comma_like(uint32_t cp) {
    return cp == 0xff0c || cp == 0x3001 || cp == 0xff1a || cp == 0xff1b;
}

bool is_fullwidth_quote_like(uint32_t cp) {
    return cp == 0xff08 || cp == 0xff09 || cp == 0x201c || cp == 0x201d ||
           cp == 0x2018 || cp == 0x2019;
}

bool is_quote_or_bracket_like(uint32_t cp) {
    return is_fullwidth_quote_like(cp) ||
           cp == '[' || cp == ']' || cp == '(' || cp == ')' ||
           cp == 0x3010 || cp == 0x3011 ||
           cp == 0x300a || cp == 0x300b ||
           cp == 0x300c || cp == 0x300d;
}

bool is_quote_mark_like(uint32_t cp) {
    return cp == '"' || cp == 0x201c || cp == 0x201d ||
           cp == 0x2018 || cp == 0x2019;
}

bool is_dash_like(uint32_t cp) {
    // 0x00b7 middle dot (人名间隔号): reference TextNormalizer maps "·" -> "-"
    return cp == 0x2014 || cp == 0xff0d || cp == 0x2212 || cp == 0x00b7 || cp == 0xff5e;
}

bool is_unknown_dash_like(uint32_t cp) {
    return cp == 0x2013;
}

bool try_append_normalized_ascii_punctuation(
    std::vector<std::string>& pieces,
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    const std::string& ascii_piece,
    bool previous_was_ascii_word) {
    const std::string space_piece = "\xe2\x96\x81";
    const std::string prefixed = space_piece + ascii_piece;
    if (previous_was_ascii_word && piece_to_id.find(ascii_piece) != piece_to_id.end()) {
        pieces.push_back(ascii_piece);
    } else if (!pieces.empty() && pieces.back() == "'" && piece_to_id.find(ascii_piece) != piece_to_id.end()) {
        pieces.push_back(ascii_piece);
    } else if (piece_to_id.find(prefixed) != piece_to_id.end()) {
        pieces.push_back(prefixed);
    } else if (piece_to_id.find(ascii_piece) != piece_to_id.end()) {
        pieces.push_back(space_piece);
        pieces.push_back(ascii_piece);
    } else if (ascii_piece == "%" && piece_to_id.find("<unk>") != piece_to_id.end()) {
        pieces.push_back(space_piece);
        pieces.push_back(ascii_piece);
    } else if (ascii_piece == "/" && piece_to_id.find("<unk>") != piece_to_id.end()) {
        if (!previous_was_ascii_word) {
            pieces.push_back(space_piece);
        }
        pieces.push_back(ascii_piece);
    } else {
        return false;
    }
    return true;
}

void append_normalized_ascii_punctuation(
    std::vector<std::string>& pieces,
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    const std::string& ascii_piece,
    bool previous_was_ascii_word) {
    if (try_append_normalized_ascii_punctuation(pieces, piece_to_id, ascii_piece, previous_was_ascii_word)) {
        return;
    }
    const auto& fallback = punct_fallback_config().map;
    const auto it = fallback.find(ascii_piece);
    if (it != fallback.end()) {
        if (it->second.empty()) {
            return;  // configured drop
        }
        if (try_append_normalized_ascii_punctuation(pieces, piece_to_id, it->second, previous_was_ascii_word)) {
            return;
        }
        std::cerr << "warning: punct_fallback maps '" << ascii_piece << "' to '" << it->second
                  << "' but the replacement is not encodable either; skipping" << std::endl;
        return;
    }
    std::cerr << "warning: tokenizer vocab cannot encode punctuation piece '" << ascii_piece
              << "'; skipping (add a mapping to tokenizer/punct_fallback.tsv to control this)" << std::endl;
}

void append_quote_mark_punctuation(
    std::vector<std::string>& pieces,
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    bool previous_was_ascii_word) {
    if ((previous_was_ascii_word || (!pieces.empty() && is_ascii_punctuation_piece(pieces.back()))) &&
        piece_to_id.find("'") != piece_to_id.end()) {
        pieces.push_back("'");
        return;
    }
    append_normalized_ascii_punctuation(pieces, piece_to_id, "'", previous_was_ascii_word);
}

void append_unknown_id_punctuation(std::vector<std::string>& pieces,
                                   const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
                                   const std::string& punctuation_piece,
                                   bool previous_was_ascii_word) {
    const std::string space_piece = "\xe2\x96\x81";
    if (piece_to_id.find("<unk>") == piece_to_id.end()) {
        std::cerr << "warning: tokenizer vocab cannot encode punctuation piece '" << punctuation_piece
                  << "' without <unk>; skipping" << std::endl;
        return;
    }
    if (!previous_was_ascii_word) {
        pieces.push_back(space_piece);
    }
    pieces.push_back(punctuation_piece);
}

bool normalize_fullwidth_alnum(uint32_t& cp, std::string& bytes) {
    if (cp >= 0xff10 && cp <= 0xff19) {
        cp = '0' + (cp - 0xff10);
    } else if (cp >= 0xff21 && cp <= 0xff3a) {
        cp = 'A' + (cp - 0xff21);
    } else if (cp >= 0xff41 && cp <= 0xff5a) {
        cp = 'a' + (cp - 0xff41);
    } else {
        return false;
    }
    bytes.assign(1, static_cast<char>(cp));
    return true;
}

bool suffix_has_cjk(const std::string& text, size_t pos) {
    while (pos < text.size()) {
        uint32_t cp = 0;
        std::string bytes;
        utf8_next(text, pos, cp, bytes);
        normalize_fullwidth_alnum(cp, bytes);
        if (is_cjk_codepoint(cp)) {
            return true;
        }
    }
    return false;
}

struct SignedCjkNumberProbe {
    bool has_digit = false;
    bool supports_cjk_signed_number = false;
    bool has_temperature_unit = false;
};

SignedCjkNumberProbe probe_signed_cjk_number_suffix(const std::string& text,
                                                    size_t digit_pos,
                                                    bool seen_cjk_context) {
    SignedCjkNumberProbe probe;
    uint32_t digit_cp = 0;
    std::string digit_bytes;
    if (!utf8_next(text, digit_pos, digit_cp, digit_bytes)) {
        return probe;
    }
    normalize_fullwidth_alnum(digit_cp, digit_bytes);
    if (!is_ascii_digit(digit_cp)) {
        return probe;
    }
    probe.has_digit = true;
    size_t after_digit_probe = digit_pos;
    while (after_digit_probe < text.size()) {
        const size_t saved = after_digit_probe;
        uint32_t next_digit_cp = 0;
        std::string next_digit_bytes;
        utf8_next(text, after_digit_probe, next_digit_cp, next_digit_bytes);
        normalize_fullwidth_alnum(next_digit_cp, next_digit_bytes);
        if (!is_ascii_digit(next_digit_cp)) {
            after_digit_probe = saved;
            break;
        }
    }
    probe.supports_cjk_signed_number = after_digit_probe == text.size();
    if (!probe.supports_cjk_signed_number && after_digit_probe < text.size()) {
        size_t suffix_pos = after_digit_probe;
        uint32_t suffix_cp = 0;
        std::string suffix_bytes;
        utf8_next(text, suffix_pos, suffix_cp, suffix_bytes);
        normalize_fullwidth_alnum(suffix_cp, suffix_bytes);
        probe.has_temperature_unit = suffix_cp == 0x2103 || suffix_cp == 0x2109;
        probe.supports_cjk_signed_number = is_cjk_codepoint(suffix_cp) ||
                                           probe.has_temperature_unit ||
                                           suffix_cp == '.' || suffix_cp == 0xff0e;
        if (!probe.supports_cjk_signed_number && suffix_cp == 0x00b0 && suffix_pos < text.size()) {
            uint32_t unit_cp = 0;
            std::string unit_bytes;
            utf8_next(text, suffix_pos, unit_cp, unit_bytes);
            normalize_fullwidth_alnum(unit_cp, unit_bytes);
            probe.has_temperature_unit = unit_cp == 'C' || unit_cp == 'c' ||
                                         unit_cp == 'F' || unit_cp == 'f';
            probe.supports_cjk_signed_number = probe.has_temperature_unit && seen_cjk_context;
        }
    }
    return probe;
}

std::string ascii_upper(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool ascii_run_is_alpha(const std::string& run) {
    return !run.empty() && std::all_of(run.begin(), run.end(), is_ascii_alpha_byte);
}

bool ascii_run_has_uppercase(const std::string& run) {
    return std::any_of(run.begin(), run.end(), [](char ch) {
        return std::isupper(static_cast<unsigned char>(ch)) != 0;
    });
}

std::string normalized_digit_piece(char digit, bool decimal_fraction) {
    switch (digit) {
        case '0':
            return "\xe9\x9b\xb6";
        case '1':
            return decimal_fraction ? "\xe4\xb8\x80" : "\xe5\xb9\xba";
        case '2':
            return "\xe4\xba\x8c";
        case '3':
            return "\xe4\xb8\x89";
        case '4':
            return "\xe5\x9b\x9b";
        case '5':
            return "\xe4\xba\x94";
        case '6':
            return "\xe5\x85\xad";
        case '7':
            return "\xe4\xb8\x83";
        case '8':
            return "\xe5\x85\xab";
        case '9':
            return "\xe4\xb9\x9d";
        default:
            throw std::runtime_error("native CJK text-id export received non-digit in digit normalizer");
    }
}

std::vector<std::string> tokenize_digit_run_as_cjk(const std::string& run, bool decimal_fraction) {
    const std::string space_piece = "\xe2\x96\x81";
    std::vector<std::string> out;
    out.reserve(run.size() * 2);
    for (char digit : run) {
        out.push_back(space_piece);
        out.push_back(normalized_digit_piece(digit, decimal_fraction));
    }
    return out;
}

std::vector<std::string> tokenize_digit_run_as_cjk_integer(const std::string& run, bool decimal_fraction) {
    const std::string space_piece = "\xe2\x96\x81";
    if (decimal_fraction || run.size() > 2) {
        return tokenize_digit_run_as_cjk(run, decimal_fraction);
    }
    if (run.size() == 1) {
        std::vector<std::string> out;
        out.push_back(space_piece);
        out.push_back(normalized_digit_piece(run[0], true));
        return out;
    }
    if (run[0] == '0') {
        return tokenize_digit_run_as_cjk(run, true);
    }
    std::vector<std::string> out;
    if (run[0] != '1') {
        out.push_back(space_piece);
        out.push_back(normalized_digit_piece(run[0], true));
    }
    out.push_back(space_piece);
    out.push_back("\xe5\x8d\x81");
    if (run[1] != '0') {
        out.push_back(space_piece);
        out.push_back(normalized_digit_piece(run[1], true));
    }
    return out;
}

void append_cjk_piece(std::vector<std::string>& out, const std::string& piece) {
    const std::string space_piece = "\xe2\x96\x81";
    out.push_back(space_piece);
    out.push_back(piece);
}

std::vector<std::string> tokenize_operator_rhs_degree_number_as_cjk(const std::string& run,
                                                                    bool decimal_fraction) {
    if (decimal_fraction || run.empty() || run.size() > 4 || (run.size() == 4 && run != "1000")) {
        throw std::runtime_error("native CJK text-id export operator degree RHS needs full TextNormalizer for this number: " + run);
    }
    for (char digit : run) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export operator degree RHS received non-digit");
        }
    }
    if (run == "1000") {
        std::vector<std::string> out;
        append_cjk_piece(out, "\xe4\xb8\x80");
        append_cjk_piece(out, "\xe7\x99\xbe");
        append_cjk_piece(out, "\xe9\x9b\xb6");
        return out;
    }
    if (run.size() == 3) {
        std::vector<std::string> out;
        if (run[1] == '0') {
            if (run[0] != '1') {
                append_cjk_piece(out, normalized_digit_piece(run[0], true));
            }
            append_cjk_piece(out, "\xe5\x8d\x81");
            if (run[2] != '0') {
                append_cjk_piece(out, run[2] == '2' ? "\xe4\xb8\xa4" : normalized_digit_piece(run[2], true));
            }
            return out;
        }
        append_cjk_piece(out, normalized_digit_piece(run[0], true));
        auto tail = tokenize_digit_run_as_cjk_integer(run.substr(1), false);
        out.insert(out.end(), tail.begin(), tail.end());
        return out;
    }
    std::vector<std::string> out;
    out.reserve(run.size() * 2);
    for (size_t i = 0; i < run.size(); ++i) {
        if (i + 1 == run.size() && run[i] == '2') {
            append_cjk_piece(out, "\xe4\xb8\xa4");
        } else {
            append_cjk_piece(out, normalized_digit_piece(run[i], true));
        }
    }
    return out;
}

std::vector<std::string> tokenize_temperature_number_as_cjk(const std::string& run,
                                                            bool decimal_fraction) {
    if (decimal_fraction || run.empty()) {
        return tokenize_digit_run_as_cjk_integer(run, decimal_fraction);
    }
    if (run == "2") {
        std::vector<std::string> out;
        append_cjk_piece(out, "\xe4\xb8\xa4");
        return out;
    }
    if (run.size() > 1 && run[0] == '0') {
        std::vector<std::string> out;
        out.reserve(run.size() * 2);
        for (size_t i = 0; i < run.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(run[i]))) {
                throw std::runtime_error("native CJK text-id export temperature helper received non-digit");
            }
            if (i + 1 == run.size() && run[i] == '2') {
                append_cjk_piece(out, "\xe4\xb8\xa4");
            } else {
                append_cjk_piece(out, normalized_digit_piece(run[i], true));
            }
        }
        return out;
    }
    return tokenize_digit_run_as_cjk_integer(run, decimal_fraction);
}

bool native_operator_rhs_degree_number_context(const std::string& text, size_t digit_pos) {
    uint32_t digit_cp = 0;
    std::string digit_bytes;
    if (!utf8_next(text, digit_pos, digit_cp, digit_bytes)) {
        return false;
    }
    normalize_fullwidth_alnum(digit_cp, digit_bytes);
    if (!is_ascii_digit(digit_cp)) {
        return false;
    }
    size_t after_digits = digit_pos;
    while (after_digits < text.size()) {
        const size_t saved = after_digits;
        uint32_t next_cp = 0;
        std::string next_bytes;
        utf8_next(text, after_digits, next_cp, next_bytes);
        normalize_fullwidth_alnum(next_cp, next_bytes);
        if (!is_ascii_digit(next_cp)) {
            after_digits = saved;
            break;
        }
    }
    if (after_digits == text.size()) {
        return false;
    }
    size_t suffix_pos = after_digits;
    uint32_t suffix_cp = 0;
    std::string suffix_bytes;
    utf8_next(text, suffix_pos, suffix_cp, suffix_bytes);
    normalize_fullwidth_alnum(suffix_cp, suffix_bytes);
    return suffix_cp == 0x2103 || suffix_cp == 0x2109 || suffix_cp == 0x5ea6;
}

void append_cjk_integer_under_10000(std::vector<std::string>& out, int value);

size_t native_digit_run_length_at(const std::string& text, size_t digit_pos) {
    size_t len = 0;
    size_t pos = digit_pos;
    while (pos < text.size()) {
        const size_t saved = pos;
        uint32_t cp = 0;
        std::string bytes;
        utf8_next(text, pos, cp, bytes);
        normalize_fullwidth_alnum(cp, bytes);
        if (!is_ascii_digit(cp)) {
            pos = saved;
            break;
        }
        ++len;
    }
    return len;
}

bool read_cjk_measure_suffix_pieces(const std::string& text,
                                    size_t suffix_pos,
                                    std::vector<std::string>& suffix,
                                    size_t& suffix_end) {
    size_t pos = suffix_pos;
    uint32_t cp = 0;
    std::string bytes;
    if (!utf8_next(text, pos, cp, bytes)) {
        return false;
    }
    suffix.clear();
    if (cp == 0x7c73 || cp == 0x5143 || cp == 0x5ea6) {
        append_cjk_piece(suffix, bytes);
        suffix_end = pos;
        return true;
    }
    if (cp == 0x516c && pos < text.size()) {
        size_t next_pos = pos;
        uint32_t next_cp = 0;
        std::string next_bytes;
        utf8_next(text, next_pos, next_cp, next_bytes);
        if (next_cp == 0x91cc) {
            append_cjk_piece(suffix, bytes);
            append_cjk_piece(suffix, next_bytes);
            suffix_end = next_pos;
            return true;
        }
    }
    return false;
}

bool native_cjk_measure_number_context(const std::string& text, size_t digit_pos) {
    size_t after_digits = digit_pos;
    while (after_digits < text.size()) {
        const size_t saved = after_digits;
        uint32_t cp = 0;
        std::string bytes;
        utf8_next(text, after_digits, cp, bytes);
        normalize_fullwidth_alnum(cp, bytes);
        if (!is_ascii_digit(cp)) {
            after_digits = saved;
            break;
        }
    }
    std::vector<std::string> suffix;
    size_t suffix_end = after_digits;
    return read_cjk_measure_suffix_pieces(text, after_digits, suffix, suffix_end);
}

std::vector<std::string> tokenize_measure_number_as_cjk(const std::string& run,
                                                        bool decimal_fraction) {
    if (decimal_fraction || run.empty() || run.size() > 4) {
        throw std::runtime_error("native CJK text-id export measure number needs full TextNormalizer for this number: " + run);
    }
    for (char digit : run) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export measure number received non-digit");
        }
    }
    if (run == "2") {
        std::vector<std::string> out;
        append_cjk_piece(out, "\xe4\xb8\xa4");
        return out;
    }
    if (run.size() > 1 && run[0] == '0') {
        std::vector<std::string> out;
        for (size_t i = 0; i < run.size(); ++i) {
            if (i + 1 == run.size() && run[i] == '2') {
                append_cjk_piece(out, "\xe4\xb8\xa4");
            } else {
                append_cjk_piece(out, normalized_digit_piece(run[i], true));
            }
        }
        return out;
    }
    std::vector<std::string> out;
    append_cjk_integer_under_10000(out, std::stoi(run));
    return out;
}

std::vector<std::string> tokenize_operator_rhs_measure_number_as_cjk(const std::string& run,
                                                                     bool decimal_fraction) {
    if (decimal_fraction || run.empty() || run.size() > 4 || (run.size() == 4 && run != "1000")) {
        throw std::runtime_error("native CJK text-id export operator measure RHS needs full TextNormalizer for this number: " + run);
    }
    for (char digit : run) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export operator measure RHS received non-digit");
        }
    }
    if (run == "1000") {
        std::vector<std::string> out;
        append_cjk_piece(out, "\xe4\xb8\x80");
        append_cjk_piece(out, "\xe7\x99\xbe");
        append_cjk_piece(out, "\xe9\x9b\xb6");
        return out;
    }
    if (run.size() == 3) {
        std::vector<std::string> out;
        if (run[1] == '0') {
            if (run[0] != '1') {
                append_cjk_piece(out, normalized_digit_piece(run[0], true));
            }
            append_cjk_piece(out, "\xe5\x8d\x81");
            if (run[2] != '0') {
                append_cjk_piece(out, run[2] == '2' ? "\xe4\xb8\xa4" : normalized_digit_piece(run[2], true));
            }
            return out;
        }
        append_cjk_piece(out, normalized_digit_piece(run[0], true));
        auto tail = tokenize_digit_run_as_cjk_integer(run.substr(1), false);
        out.insert(out.end(), tail.begin(), tail.end());
        return out;
    }
    std::vector<std::string> out;
    for (size_t i = 0; i < run.size(); ++i) {
        if (i + 1 == run.size() && run[i] == '2') {
            append_cjk_piece(out, "\xe4\xb8\xa4");
        } else {
            append_cjk_piece(out, normalized_digit_piece(run[i], true));
        }
    }
    return out;
}

bool native_operator_rhs_digit_context(const std::string& text,
                                       size_t digit_pos,
                                       bool allow_degree_number_context = false) {
    uint32_t digit_cp = 0;
    std::string digit_bytes;
    if (!utf8_next(text, digit_pos, digit_cp, digit_bytes)) {
        return false;
    }
    normalize_fullwidth_alnum(digit_cp, digit_bytes);
    if (!is_ascii_digit(digit_cp)) {
        return false;
    }
    size_t after_digits = digit_pos;
    while (after_digits < text.size()) {
        const size_t saved = after_digits;
        uint32_t next_cp = 0;
        std::string next_bytes;
        utf8_next(text, after_digits, next_cp, next_bytes);
        normalize_fullwidth_alnum(next_cp, next_bytes);
        if (!is_ascii_digit(next_cp)) {
            after_digits = saved;
            break;
        }
    }
    if (after_digits == text.size()) {
        return true;
    }
    size_t suffix_pos = after_digits;
    uint32_t suffix_cp = 0;
    std::string suffix_bytes;
    utf8_next(text, suffix_pos, suffix_cp, suffix_bytes);
    normalize_fullwidth_alnum(suffix_cp, suffix_bytes);
    if (allow_degree_number_context && (suffix_cp == 0x2103 || suffix_cp == 0x2109)) {
        return true;
    }
    return is_cjk_codepoint(suffix_cp);
}

void append_cjk_equal_pieces(std::vector<std::string>& out) {
    append_cjk_piece(out, "\xe7\xad\x89");
    append_cjk_piece(out, "\xe4\xba\x8e");
}

void append_cjk_less_than_pieces(std::vector<std::string>& out) {
    append_cjk_piece(out, "\xe5\xb0\x8f");
    append_cjk_piece(out, "\xe4\xba\x8e");
}

void append_cjk_greater_than_pieces(std::vector<std::string>& out) {
    append_cjk_piece(out, "\xe5\xa4\xa7");
    append_cjk_piece(out, "\xe4\xba\x8e");
}

std::vector<std::string> tokenize_phone_digits_as_cjk(const std::string& digits) {
    if (digits.empty()) {
        throw std::runtime_error("native CJK text-id export phone helper expects digits");
    }
    std::vector<std::string> out;
    out.reserve(digits.size() * 2);
    for (char digit : digits) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in phone normalizer");
        }
        append_cjk_piece(out, normalized_digit_piece(digit, true));
    }
    return out;
}

std::vector<std::string> tokenize_phone_digits_as_cjk_yao(const std::string& digits) {
    if (digits.empty()) {
        throw std::runtime_error("native CJK text-id export phone yao helper expects digits");
    }
    std::vector<std::string> out;
    out.reserve(digits.size() * 2);
    for (char digit : digits) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in phone yao normalizer");
        }
        append_cjk_piece(out, normalized_digit_piece(digit, false));
    }
    return out;
}

void append_cjk_integer_under_100(std::vector<std::string>& out, int value) {
    if (value < 0 || value >= 100) {
        throw std::runtime_error("native CJK text-id export CJK integer helper expects 0..99");
    }
    if (value < 10) {
        append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + value), true));
        return;
    }
    const int tens = value / 10;
    const int ones = value % 10;
    if (tens != 1) {
        append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + tens), true));
    }
    append_cjk_piece(out, "\xe5\x8d\x81");
    if (ones != 0) {
        append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + ones), true));
    }
}

void append_cjk_integer_under_100_explicit_ten(std::vector<std::string>& out, int value) {
    if (value < 0 || value >= 100) {
        throw std::runtime_error("native CJK text-id export explicit-ten helper expects 0..99");
    }
    if (value < 10) {
        append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + value), true));
        return;
    }
    const int tens = value / 10;
    const int ones = value % 10;
    append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + tens), true));
    append_cjk_piece(out, "\xe5\x8d\x81");
    if (ones != 0) {
        append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + ones), true));
    }
}

void append_cjk_integer_under_10000(std::vector<std::string>& out, int value) {
    if (value < 0 || value >= 10000) {
        throw std::runtime_error("native CJK text-id export CJK integer helper expects 0..9999");
    }
    if (value < 100) {
        append_cjk_integer_under_100(out, value);
        return;
    }
    if (value < 1000) {
        const int hundreds = value / 100;
        append_cjk_piece(out, hundreds == 2 ? "\xe4\xb8\xa4" : normalized_digit_piece(static_cast<char>('0' + hundreds), true));
        append_cjk_piece(out, "\xe7\x99\xbe");
        const int rest = value % 100;
        if (rest != 0) {
            if (rest < 10) {
                append_cjk_piece(out, "\xe9\x9b\xb6");
                append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + rest), true));
            } else {
                append_cjk_integer_under_100_explicit_ten(out, rest);
            }
        }
        return;
    }

    const int thousands = value / 1000;
    append_cjk_piece(out, thousands == 2 ? "\xe4\xb8\xa4" : normalized_digit_piece(static_cast<char>('0' + thousands), true));
    append_cjk_piece(out, "\xe5\x8d\x83");
    const int rest = value % 1000;
    if (rest == 0) {
        return;
    }
    if (rest < 100) {
        append_cjk_piece(out, "\xe9\x9b\xb6");
        if (rest < 10) {
            append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + rest), true));
        } else {
            append_cjk_integer_under_100_explicit_ten(out, rest);
        }
    } else {
        append_cjk_integer_under_10000(out, rest);
    }
}

void append_cjk_integer_under_100000(std::vector<std::string>& out, int value) {
    if (value < 0 || value >= 100000) {
        throw std::runtime_error("native CJK text-id export CJK currency integer helper expects 0..99999");
    }
    if (value < 10000) {
        append_cjk_integer_under_10000(out, value);
        return;
    }
    const int ten_thousands = value / 10000;
    append_cjk_piece(out, ten_thousands == 2 ? "\xe4\xb8\xa4" : normalized_digit_piece(static_cast<char>('0' + ten_thousands), true));
    append_cjk_piece(out, "\xe4\xb8\x87");
    const int rest = value % 10000;
    if (rest == 0) {
        return;
    }
    if (rest < 1000) {
        append_cjk_piece(out, "\xe9\x9b\xb6");
        if (rest < 100) {
            append_cjk_integer_under_100_explicit_ten(out, rest);
        } else {
            append_cjk_integer_under_10000(out, rest);
        }
    } else {
        append_cjk_integer_under_10000(out, rest);
    }
}

std::vector<std::string> tokenize_plain_four_digit_integer_as_cjk(const std::string& run) {
    if (run.size() != 4 || run[0] == '0') {
        throw std::runtime_error("native CJK text-id export plain integer helper expects non-leading-zero 4 digits");
    }
    std::vector<std::string> out;
    append_cjk_integer_under_10000(out, std::stoi(run));
    return out;
}

std::vector<std::string> tokenize_fullwidth_hyphen_grouped_3_4_4_as_cjk(
    const std::string& group0,
    const std::string& group1,
    const std::string& group2) {
    if (group0.size() != 3 || group1.size() != 4 || group2.size() != 4) {
        throw std::runtime_error("native CJK text-id export fullwidth hyphen grouped-number helper expects 3-4-4 digits");
    }
    std::vector<std::string> out = tokenize_phone_digits_as_cjk_yao(group0);
    append_cjk_piece(out, "-");
    auto middle = tokenize_phone_digits_as_cjk_yao(group1);
    out.insert(out.end(), middle.begin(), middle.end());
    append_cjk_piece(out, "-");
    append_cjk_integer_under_10000(out, std::stoi(group2));
    return out;
}

std::vector<std::string> tokenize_short_hyphen_subtraction_as_cjk(const std::string& lhs,
                                                                  const std::string& rhs) {
    if (lhs.empty() || rhs.empty() || lhs.size() > 2 || rhs.size() > 2 ||
        (lhs.size() > 1 && lhs[0] == '0') ||
        (rhs.size() > 1 && rhs[0] == '0')) {
        throw std::runtime_error("native CJK text-id export short hyphen subtraction helper expects 1-2 digits without leading zeros");
    }
    for (char digit : lhs + rhs) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in short hyphen subtraction normalizer");
        }
    }
    std::vector<std::string> out;
    append_cjk_integer_under_100(out, std::stoi(lhs));
    append_cjk_piece(out, "\xe5\x87\x8f");
    append_cjk_integer_under_100(out, std::stoi(rhs));
    return out;
}

std::vector<std::string> tokenize_currency_number_as_cjk(const std::string& integer,
                                                         const std::string& fraction,
                                                         bool dollar) {
    if (integer.empty() || integer.size() > 5 || (!fraction.empty() && fraction.size() > 6) ||
        (integer.size() > 1 && integer[0] == '0')) {
        throw std::runtime_error("native CJK text-id export supports currency numbers with focused 1-5 integer digits without leading zeros and up to 6 fraction digits");
    }
    for (char digit : integer + fraction) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in currency normalizer");
        }
    }
    std::vector<std::string> out;
    append_cjk_integer_under_100000(out, std::stoi(integer));
    if (!fraction.empty()) {
        append_cjk_piece(out, "\xe7\x82\xb9");
        auto fraction_pieces = tokenize_digit_run_as_cjk(fraction, true);
        out.insert(out.end(), fraction_pieces.begin(), fraction_pieces.end());
    }
    if (dollar) {
        append_cjk_piece(out, "\xe7\xbe\x8e");
    }
    append_cjk_piece(out, "\xe5\x85\x83");
    return out;
}

std::vector<std::string> tokenize_leading_zero_currency_as_cjk(const std::string& integer,
                                                               const std::string& fraction,
                                                               bool dollar) {
    if (integer.size() < 2 || integer.size() > 5 || integer[0] != '0' || integer[1] == '0' ||
        (!fraction.empty() && fraction.size() > 6)) {
        throw std::runtime_error("native CJK text-id export leading-zero currency helper expects focused 0N... currency with nonzero second digit");
    }
    for (char digit : integer + fraction) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in leading-zero currency normalizer");
        }
    }
    std::vector<std::string> out;
    append_cjk_piece(out, "\xe9\x9b\xb6");
    if (dollar) {
        append_cjk_piece(out, "\xe7\xbe\x8e");
    }
    append_cjk_piece(out, "\xe5\x85\x83");
    append_cjk_integer_under_10000(out, std::stoi(integer.substr(1)));
    if (!fraction.empty()) {
        append_cjk_piece(out, "\xe7\x82\xb9");
        auto fraction_pieces = tokenize_digit_run_as_cjk(fraction, true);
        out.insert(out.end(), fraction_pieces.begin(), fraction_pieces.end());
    }
    return out;
}

std::vector<std::string> tokenize_multi_leading_zero_yen_as_reference_digits(const std::string& integer) {
    if (integer.size() < 3 || integer[0] != '0' || integer[1] != '0') {
        throw std::runtime_error("native CJK text-id export multi-leading-zero yen helper expects 00N+");
    }
    std::vector<std::string> out;
    out.push_back("\xe2\x96\x81");
    out.push_back("\xc2\xa5");
    for (char digit : integer) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in multi-leading-zero yen normalizer");
        }
        append_cjk_piece(out, normalized_digit_piece(digit, false));
    }
    return out;
}

std::vector<std::string> tokenize_multi_leading_zero_dollar_as_reference_digits(const std::string& integer) {
    if (integer.size() < 3 || integer[0] != '0' || integer[1] != '0') {
        throw std::runtime_error("native CJK text-id export multi-leading-zero dollar helper expects 00N+");
    }
    std::vector<std::string> out;
    out.push_back("\xe2\x96\x81.");
    for (char digit : integer) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in multi-leading-zero dollar normalizer");
        }
        append_cjk_piece(out, normalized_digit_piece(digit, false));
    }
    return out;
}

std::vector<std::string> tokenize_percent_number_as_cjk(const std::string& integer, const std::string& fraction) {
    if (integer.empty() || integer.size() > 4 || (integer.size() == 4 && integer[0] == '0') ||
        (!fraction.empty() && fraction.size() > 6)) {
        throw std::runtime_error("native CJK text-id export supports percentage numbers with 1-3 digits or non-leading-zero 4 digits and up to 6 fraction digits");
    }
    for (char digit : integer + fraction) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in percentage normalizer");
        }
    }
    std::vector<std::string> out;
    append_cjk_piece(out, "\xe7\x99\xbe");
    append_cjk_piece(out, "\xe5\x88\x86");
    append_cjk_piece(out, "\xe4\xb9\x8b");

    const int value = std::stoi(integer);
    if (value < 100) {
        append_cjk_integer_under_100(out, value);
    } else if (value < 1000) {
        append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + value / 100), true));
        append_cjk_piece(out, "\xe7\x99\xbe");
        const int rest = value % 100;
        if (rest != 0) {
            if (rest < 10) {
                append_cjk_piece(out, "\xe9\x9b\xb6");
                append_cjk_piece(out, normalized_digit_piece(static_cast<char>('0' + rest), true));
            } else {
                append_cjk_integer_under_100(out, rest);
            }
        }
    } else {
        append_cjk_integer_under_10000(out, value);
    }

    if (!fraction.empty()) {
        append_cjk_piece(out, "\xe7\x82\xb9");
        auto fraction_pieces = tokenize_digit_run_as_cjk(fraction, true);
        out.insert(out.end(), fraction_pieces.begin(), fraction_pieces.end());
    }
    return out;
}

void append_temperature_unit_pieces(std::vector<std::string>& out, bool celsius) {
    if (celsius) {
        append_cjk_piece(out, "\xe6\x91\x84");
    } else {
        append_cjk_piece(out, "\xe5\x8d\x8e");
    }
    append_cjk_piece(out, "\xe6\xb0\x8f");
    append_cjk_piece(out, "\xe5\xba\xa6");
}

void append_date_ymd_pieces(std::vector<std::string>& out, const std::string& year, int month, int day) {
    if (year.size() != 4 || month < 1 || month > 12 || day < 1 || day > 31) {
        throw std::runtime_error("native CJK text-id export date helper expects YYYY plus valid month/day");
    }
    auto year_pieces = tokenize_digit_run_as_cjk(year, true);
    out.insert(out.end(), year_pieces.begin(), year_pieces.end());
    append_cjk_piece(out, "\xe5\xb9\xb4");
    append_cjk_integer_under_100(out, month);
    append_cjk_piece(out, "\xe6\x9c\x88");
    append_cjk_integer_under_100(out, day);
    append_cjk_piece(out, "\xe6\x97\xa5");
}

void append_date_md_pieces(std::vector<std::string>& out, int month, int day) {
    if (month < 1 || month > 12 || day < 1 || day > 31) {
        throw std::runtime_error("native CJK text-id export month/day helper expects valid month/day");
    }
    append_cjk_integer_under_100(out, month);
    append_cjk_piece(out, "\xe6\x9c\x88");
    append_cjk_integer_under_100(out, day);
    append_cjk_piece(out, "\xe6\x97\xa5");
}

std::vector<std::string> tokenize_fraction_as_cjk(const std::string& numerator,
                                                  const std::string& denominator) {
    if (numerator.empty() || denominator.empty() || numerator.size() > 2 || denominator.size() > 2 ||
        (numerator.size() > 1 && numerator[0] == '0') ||
        (denominator.size() > 1 && denominator[0] == '0')) {
        throw std::runtime_error("native CJK text-id export fraction helper expects 1-2 ASCII digits without leading zeros");
    }
    for (char digit : numerator + denominator) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in fraction normalizer");
        }
    }
    const int numerator_value = std::stoi(numerator);
    const int denominator_value = std::stoi(denominator);
    if (denominator_value == 0) {
        throw std::runtime_error("native CJK text-id export fraction helper received zero denominator");
    }
    std::vector<std::string> out;
    append_cjk_integer_under_100(out, denominator_value);
    append_cjk_piece(out, "\xe5\x88\x86");
    append_cjk_piece(out, "\xe4\xb9\x8b");
    append_cjk_integer_under_100(out, numerator_value);
    return out;
}

std::vector<std::string> tokenize_leading_zero_denominator_fraction_as_cjk(
    const std::string& numerator,
    const std::string& denominator) {
    if (numerator.empty() || numerator.size() != 1 ||
        denominator.size() < 2 || denominator.size() > 3 || denominator[0] != '0') {
        throw std::runtime_error("native CJK text-id export leading-zero denominator fraction helper expects N/0D or N/00D");
    }
    for (char digit : numerator + denominator) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::runtime_error("native CJK text-id export received non-digit in leading-zero denominator fraction normalizer");
        }
    }
    std::vector<std::string> out;
    append_cjk_piece(out, "\xe9\x9b\xb6");
    append_cjk_piece(out, "\xe5\x88\x86");
    append_cjk_piece(out, "\xe4\xb9\x8b");
    auto numerator_pieces = tokenize_digit_run_as_cjk(numerator + denominator.substr(1), true);
    out.insert(out.end(), numerator_pieces.begin(), numerator_pieces.end());
    return out;
}

std::vector<std::string> tokenize_single_digit_chained_slash_fraction_as_cjk(
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    const std::string& prefix,
    const std::string& numerator,
    const std::string& denominator) {
    if (prefix.size() != 1 || numerator.size() != 1 || denominator.size() != 1 ||
        !std::isdigit(static_cast<unsigned char>(prefix[0])) ||
        !std::isdigit(static_cast<unsigned char>(numerator[0])) ||
        !std::isdigit(static_cast<unsigned char>(denominator[0]))) {
        throw std::runtime_error("native CJK text-id export chained slash helper expects N/N/N ASCII digits");
    }
    std::vector<std::string> out = tokenize_digit_run_as_cjk(prefix, true);
    append_normalized_ascii_punctuation(out, piece_to_id, "/", false);
    auto fraction_pieces = tokenize_fraction_as_cjk(numerator, denominator);
    out.insert(out.end(), fraction_pieces.begin(), fraction_pieces.end());
    return out;
}

void append_time_hms_pieces(std::vector<std::string>& out,
                            const std::string& hour,
                            const std::string& minute,
                            const std::optional<std::string>& second) {
    if (hour.empty() || hour.size() > 2 || minute.size() != 2 ||
        (second.has_value() && second->size() != 2)) {
        throw std::runtime_error("native CJK text-id export time helper expects H:MM, HH:MM, or HH:MM:SS");
    }
    const int hour_value = std::stoi(hour);
    const int minute_value = std::stoi(minute);
    if (hour_value < 0 || hour_value > 24 || minute_value < 0 || minute_value > 59 ||
        (hour.size() == 1 && hour[0] == '0')) {
        throw std::runtime_error("native CJK text-id export time helper received unsupported hour/minute");
    }
    if (hour_value == 0) {
        append_cjk_piece(out, "\xe9\x9b\xb6");
    } else {
        append_cjk_integer_under_100(out, hour_value);
    }
    append_cjk_piece(out, "\xe7\x82\xb9");

    if (minute_value > 0) {
        if (minute[0] == '0' && minute_value < 10) {
            append_cjk_piece(out, "\xe9\x9b\xb6");
            append_cjk_piece(out, normalized_digit_piece(minute[1], true));
        } else {
            append_cjk_integer_under_100(out, minute_value);
        }
        append_cjk_piece(out, "\xe5\x88\x86");
    }

    if (second.has_value()) {
        const int second_value = std::stoi(*second);
        if (second_value < 0 || second_value > 59) {
            throw std::runtime_error("native CJK text-id export time helper received unsupported seconds");
        }
        if (second_value > 0) {
            append_cjk_integer_under_100(out, second_value);
            append_cjk_piece(out, "\xe7\xa7\x92");
        }
    }
}

bool parse_date_separator(uint32_t cp, char& separator) {
    if (cp == '/' || cp == '-') {
        separator = static_cast<char>(cp);
        return true;
    }
    return false;
}

bool parse_time_separator(uint32_t cp) {
    return cp == ':' || cp == 0xff1a;
}

bool parse_currency_symbol(uint32_t cp, bool& dollar) {
    if (cp == '$') {
        dollar = true;
        return true;
    }
    if (cp == 0x00a5 || cp == 0xffe5) {
        dollar = false;
        return true;
    }
    return false;
}

bool read_ascii_digit_run_at(const std::string& text,
                             size_t start,
                             size_t max_digits,
                             std::string& run,
                             size_t& end,
                             bool* saw_fullwidth = nullptr) {
    run.clear();
    if (saw_fullwidth) {
        *saw_fullwidth = false;
    }
    size_t pos = start;
    while (pos < text.size() && run.size() < max_digits) {
        const size_t saved = pos;
        uint32_t cp = 0;
        std::string bytes;
        utf8_next(text, pos, cp, bytes);
        const bool from_fullwidth = normalize_fullwidth_alnum(cp, bytes);
        if (!is_ascii_digit(cp)) {
            pos = saved;
            break;
        }
        if (saw_fullwidth && from_fullwidth) {
            *saw_fullwidth = true;
        }
        run += bytes;
    }
    end = pos;
    return !run.empty();
}

void append_english_under_100(std::vector<std::string>& out, int value) {
    static const char* ones[] = {
        "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
        "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN",
        "SEVENTEEN", "EIGHTEEN", "NINETEEN"
    };
    static const char* tens[] = {
        "", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"
    };
    const std::string space_piece = "\xe2\x96\x81";
    if (value < 0 || value >= 100) {
        throw std::runtime_error("native CJK text-id export English number helper expects 0..99");
    }
    if (value < 20) {
        out.push_back(space_piece + ones[value]);
        return;
    }
    out.push_back(space_piece + tens[value / 10]);
    if (value % 10 != 0) {
        out.push_back(space_piece + ones[value % 10]);
    }
}

std::vector<std::string> tokenize_english_number_run(const std::string& run) {
    if (run.empty() || run.size() > 3) {
        throw std::runtime_error("native CJK text-id export supports only 1-3 digit English number runs");
    }
    int value = 0;
    for (char ch : run) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::runtime_error("native CJK text-id export received non-digit in English number normalizer");
        }
        value = value * 10 + (ch - '0');
    }

    const std::string space_piece = "\xe2\x96\x81";
    std::vector<std::string> out;
    if (run.size() > 1 && run[0] == '0') {
        size_t pos = 0;
        while (pos + 1 < run.size() && run[pos] == '0') {
            out.push_back(space_piece + "OH");
            ++pos;
        }
        const int remainder = std::stoi(run.substr(pos));
        append_english_under_100(out, remainder);
        return out;
    }
    if (value < 100) {
        append_english_under_100(out, value);
        return out;
    }
    out.push_back(space_piece + std::string(value / 100 == 1 ? "ONE" : value / 100 == 2 ? "TWO" :
                                           value / 100 == 3 ? "THREE" : value / 100 == 4 ? "FOUR" :
                                           value / 100 == 5 ? "FIVE" : value / 100 == 6 ? "SIX" :
                                           value / 100 == 7 ? "SEVEN" : value / 100 == 8 ? "EIGHT" : "NINE"));
    out.push_back(space_piece + "HUNDRED");
    if (value % 100 != 0) {
        out.push_back(space_piece + "AND");
        append_english_under_100(out, value % 100);
    }
    return out;
}

std::vector<std::string> tokenize_english_no_number_run(const std::string& run) {
    if (run.empty() || run.size() > 5) {
        throw std::runtime_error("native CJK text-id export supports only focused English No. number runs");
    }
    for (char ch : run) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::runtime_error("native CJK text-id export received non-digit in English No. number normalizer");
        }
    }
    if (run.size() <= 3) {
        return tokenize_english_number_run(run);
    }

    const std::string space_piece = "\xe2\x96\x81";
    if (run.size() == 5) {
        if (run[0] != '0') {
            const int thousands = std::stoi(run.substr(0, 2));
            const int rest = std::stoi(run.substr(2));
            std::vector<std::string> out;
            append_english_under_100(out, thousands);
            out.push_back(space_piece + "THOUSAND");
            if (rest == 0) {
                return out;
            }
            if (rest < 100) {
                out.push_back(space_piece + "AND");
                append_english_under_100(out, rest);
                return out;
            }
            auto rest_pieces = tokenize_english_number_run(run.substr(2));
            out.insert(out.end(), rest_pieces.begin(), rest_pieces.end());
            return out;
        }
        throw std::runtime_error("native CJK text-id export English No. number needs full TextNormalizer: " + run);
    }

    std::vector<std::string> out;
    if (run[0] == '0') {
        size_t pos = 0;
        while (pos + 1 < run.size() && run[pos] == '0') {
            out.push_back(space_piece + "OH");
            ++pos;
        }
        auto remainder = tokenize_english_number_run(run.substr(pos));
        out.insert(out.end(), remainder.begin(), remainder.end());
        return out;
    }

    const int thousands = run[0] - '0';
    const int rest = std::stoi(run.substr(1));
    if (thousands >= 3) {
        append_english_under_100(out, thousands);
        out.push_back(space_piece + "THOUSAND");
        if (rest == 0) {
            return out;
        }
        if (rest < 100) {
            out.push_back(space_piece + "AND");
            append_english_under_100(out, rest);
            return out;
        }
        auto rest_pieces = tokenize_english_number_run(run.substr(1));
        out.insert(out.end(), rest_pieces.begin(), rest_pieces.end());
        return out;
    }

    if (rest == 0) {
        append_english_under_100(out, thousands);
        out.push_back(space_piece + "THOUSAND");
        return out;
    }
    if (rest < 10) {
        append_english_under_100(out, thousands);
        out.push_back(space_piece + "THOUSAND");
        append_english_under_100(out, rest);
        return out;
    }

    const int first_two = std::stoi(run.substr(0, 2));
    const int last_two = std::stoi(run.substr(2, 2));
    append_english_under_100(out, first_two);
    if (last_two == 0) {
        out.push_back(space_piece + "HUNDRED");
    } else {
        append_english_under_100(out, last_two);
    }
    return out;
}

std::vector<std::string> tokenize_english_spaced_no_number_run(const std::string& run) {
    if (run.empty() || run.size() > 5) {
        throw std::runtime_error("native CJK text-id export supports only focused spaced No. number runs");
    }
    for (char ch : run) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::runtime_error("native CJK text-id export received non-digit in spaced No. number normalizer");
        }
    }
    if (run.size() > 1 && run[0] == '0') {
        return tokenize_english_no_number_run(run);
    }
    if (run.size() <= 3) {
        return tokenize_english_number_run(run);
    }
    if (run.size() == 4) {
        const int thousands = run[0] - '0';
        const int rest = std::stoi(run.substr(1));
        std::vector<std::string> out;
        append_english_under_100(out, thousands);
        const std::string space_piece = "\xe2\x96\x81";
        out.push_back(space_piece + "THOUSAND");
        if (rest == 0) {
            return out;
        }
        if (rest < 100) {
            out.push_back(space_piece + "AND");
            append_english_under_100(out, rest);
            return out;
        }
        auto rest_pieces = tokenize_english_number_run(run.substr(1));
        out.insert(out.end(), rest_pieces.begin(), rest_pieces.end());
        return out;
    }
    if (run.size() == 5 && run[0] != '0') {
        const int thousands = std::stoi(run.substr(0, 2));
        const int rest = std::stoi(run.substr(2));
        const std::string space_piece = "\xe2\x96\x81";
        std::vector<std::string> out;
        append_english_under_100(out, thousands);
        out.push_back(space_piece + "THOUSAND");
        if (rest == 0) {
            return out;
        }
        if (rest < 100) {
            out.push_back(space_piece + "AND");
            append_english_under_100(out, rest);
            return out;
        }
        auto rest_pieces = tokenize_english_number_run(run.substr(2));
        out.insert(out.end(), rest_pieces.begin(), rest_pieces.end());
        return out;
    }
    throw std::runtime_error("native CJK text-id export spaced No. number needs full TextNormalizer for this number: " + run);
}

std::vector<std::string> tokenize_english_unit_number_run(const std::string& run) {
    if (run.empty() || run.size() > 5) {
        throw std::runtime_error("native CJK text-id export supports only focused English unit number runs");
    }
    for (char ch : run) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::runtime_error("native CJK text-id export received non-digit in English unit number normalizer");
        }
    }
    if (run.size() > 1 && run[0] == '0') {
        if (run.size() > 3) {
            throw std::runtime_error("native CJK text-id export English unit leading-zero number needs full TextNormalizer: " + run);
        }
        const std::string space_piece = "\xe2\x96\x81";
        std::vector<std::string> out;
        size_t pos = 0;
        while (pos + 2 < run.size() && run[pos] == '0') {
            out.push_back(space_piece + "OH");
            ++pos;
        }
        if (pos < run.size() && run[pos] == '0') {
            out.push_back(space_piece + "ZERO");
            ++pos;
        }
        if (pos < run.size()) {
            append_english_under_100(out, std::stoi(run.substr(pos)));
        }
        return out;
    }
    if (run.size() <= 3 || run.size() == 4 || run[0] != '0') {
        return tokenize_english_spaced_no_number_run(run);
    }
    throw std::runtime_error("native CJK text-id export English unit number needs full TextNormalizer: " + run);
}

bool english_unit_number_is_singular(const std::string& run) {
    if (run.empty() || run.size() > 5) {
        throw std::runtime_error("native CJK text-id export English unit singular probe expects focused digits");
    }
    for (char ch : run) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::runtime_error("native CJK text-id export English unit singular probe received non-digit");
        }
    }
    return std::stoi(run) == 1;
}

std::vector<std::string> tokenize_english_unit_decimal_number_run(const std::string& integer_run,
                                                                  const std::string& fraction_run) {
    if (fraction_run.empty()) {
        throw std::runtime_error("native CJK text-id export English unit decimal helper expects fractional digits");
    }
    std::vector<std::string> out = (integer_run.size() > 1 && integer_run[0] == '0')
        ? tokenize_english_number_run(integer_run)
        : tokenize_english_unit_number_run(integer_run);
    const std::string space_piece = "\xe2\x96\x81";
    out.push_back(space_piece + "POINT");
    for (char ch : fraction_run) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::runtime_error("native CJK text-id export received non-digit in English unit decimal normalizer");
        }
        if (ch == '0') {
            out.push_back(space_piece + "OH");
        } else {
            append_english_under_100(out, ch - '0');
        }
    }
    return out;
}

void append_alpha_colon_number_pieces(
    std::vector<std::string>& out,
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    const std::string& digits,
    bool cjk_number_context) {
    const std::string space_piece = "\xe2\x96\x81";
    if (digits.empty() || digits.size() > 3) {
        if (!(cjk_number_context && digits.size() == 4)) {
            throw std::runtime_error("native CJK text-id export alpha-colon helper expects 1-3 digits, or 4 digits in CJK context");
        }
    }
    if (cjk_number_context) {
        if (piece_to_id.find(",") == piece_to_id.end()) {
            throw std::runtime_error("native CJK text-id export cannot encode alpha-colon comma piece");
        }
        out.push_back(",");
        if (digits.size() == 4) {
            append_cjk_integer_under_10000(out, std::stoi(digits));
        } else {
            auto digit_pieces = tokenize_digit_run_as_cjk_integer(digits, false);
            out.insert(out.end(), digit_pieces.begin(), digit_pieces.end());
        }
    } else {
        const std::string prefixed_comma = space_piece + ",";
        if (piece_to_id.find(prefixed_comma) == piece_to_id.end()) {
            throw std::runtime_error("native CJK text-id export cannot encode alpha-colon prefixed comma piece");
        }
        out.push_back(prefixed_comma);
        auto digit_pieces = tokenize_english_number_run(digits);
        out.insert(out.end(), digit_pieces.begin(), digit_pieces.end());
    }
}

std::vector<std::string> tokenize_ascii_run_unprefixed(
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    const std::string& run) {
    const std::string upper = ascii_upper(run);
    bool has_scores = false;
    for (size_t start = 0; start < upper.size() && !has_scores; ++start) {
        for (size_t len = 1; start + len <= upper.size(); ++len) {
            auto it = piece_to_id.find(upper.substr(start, len));
            if (it != piece_to_id.end() && it->second.has_score) {
                has_scores = true;
                break;
            }
        }
    }
    if (has_scores) {
        const float neg_inf = -std::numeric_limits<float>::infinity();
        std::vector<float> best(upper.size() + 1, neg_inf);
        std::vector<size_t> prev(upper.size() + 1, std::numeric_limits<size_t>::max());
        std::vector<std::string> prev_piece(upper.size() + 1);
        best[0] = 0.0f;
        for (size_t pos = 0; pos < upper.size(); ++pos) {
            if (!std::isfinite(best[pos])) {
                continue;
            }
            for (size_t len = 1; pos + len <= upper.size(); ++len) {
                const std::string piece = upper.substr(pos, len);
                auto it = piece_to_id.find(piece);
                if (it == piece_to_id.end() || !it->second.has_score) {
                    continue;
                }
                const float candidate = best[pos] + it->second.score;
                const size_t next = pos + len;
                const bool better = candidate > best[next] ||
                                    (candidate == best[next] && len < prev_piece[next].size());
                if (better) {
                    best[next] = candidate;
                    prev[next] = pos;
                    prev_piece[next] = piece;
                }
            }
        }
        if (std::isfinite(best[upper.size()])) {
            std::vector<std::string> out;
            size_t pos = upper.size();
            while (pos > 0) {
                out.push_back(prev_piece[pos]);
                pos = prev[pos];
            }
            std::reverse(out.begin(), out.end());
            return out;
        }
    }

    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < upper.size()) {
        bool matched = false;
        for (size_t len = upper.size() - pos; len > 0; --len) {
            const std::string piece = upper.substr(pos, len);
            if (piece_to_id.find(piece) != piece_to_id.end()) {
                out.push_back(piece);
                pos += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            throw std::runtime_error("native CJK text-id export cannot encode unprefixed ASCII run without full SentencePiece: " + run);
        }
    }
    return out;
}

std::vector<std::string> tokenize_ascii_run_prefixed(
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    const std::string& run) {
    const std::string upper = ascii_upper(run);
    const std::string space_piece = "\xe2\x96\x81";
    struct ScoredAsciiCandidate {
        bool ok = false;
        float score = -std::numeric_limits<float>::infinity();
        std::vector<std::string> pieces;
    };
    auto best_scored = [&](bool standalone_space_prefix) {
        ScoredAsciiCandidate result;
        if (standalone_space_prefix) {
            auto space_it = piece_to_id.find(space_piece);
            if (space_it == piece_to_id.end() || !space_it->second.has_score) {
                return result;
            }
        }
        const float neg_inf = -std::numeric_limits<float>::infinity();
        std::vector<float> best(upper.size() + 1, neg_inf);
        std::vector<size_t> prev(upper.size() + 1, std::numeric_limits<size_t>::max());
        std::vector<std::string> prev_piece(upper.size() + 1);
        best[0] = standalone_space_prefix ? piece_to_id.find(space_piece)->second.score : 0.0f;
        for (size_t pos = 0; pos < upper.size(); ++pos) {
            if (!std::isfinite(best[pos])) {
                continue;
            }
            for (size_t len = 1; pos + len <= upper.size(); ++len) {
                const std::string piece =
                    (!standalone_space_prefix && pos == 0 ? space_piece : std::string()) +
                    upper.substr(pos, len);
                auto it = piece_to_id.find(piece);
                if (it == piece_to_id.end() || !it->second.has_score) {
                    continue;
                }
                const float candidate = best[pos] + it->second.score;
                const size_t next = pos + len;
                const bool better = candidate > best[next] ||
                                    (candidate == best[next] && len < prev_piece[next].size());
                if (better) {
                    best[next] = candidate;
                    prev[next] = pos;
                    prev_piece[next] = piece;
                }
            }
        }
        if (std::isfinite(best[upper.size()])) {
            std::vector<std::string> out;
            size_t pos = upper.size();
            while (pos > 0) {
                out.push_back(prev_piece[pos]);
                pos = prev[pos];
            }
            std::reverse(out.begin(), out.end());
            if (standalone_space_prefix) {
                out.insert(out.begin(), space_piece);
            }
            result.ok = true;
            result.score = best[upper.size()];
            result.pieces = std::move(out);
        }
        return result;
    };

    const auto direct = best_scored(false);
    const auto standalone = best_scored(true);
    if (direct.ok || standalone.ok) {
        if (!standalone.ok) {
            return direct.pieces;
        }
        if (!direct.ok) {
            return standalone.pieces;
        }
        if (standalone.score > direct.score ||
            (standalone.score == direct.score && standalone.pieces.size() < direct.pieces.size())) {
            return standalone.pieces;
        }
        return direct.pieces;
    }

    const std::string piece = space_piece + upper;
    if (piece_to_id.find(piece) == piece_to_id.end()) {
        auto unprefixed = tokenize_ascii_run_unprefixed(piece_to_id, run);
        std::vector<std::string> out;
        out.push_back(space_piece);
        out.insert(out.end(), unprefixed.begin(), unprefixed.end());
        return out;
    }
    return {piece};
}

void append_english_temperature_unit_pieces(
    std::vector<std::string>& out,
    const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
    bool celsius) {
    auto degrees = tokenize_ascii_run_prefixed(piece_to_id, "DEGREES");
    out.insert(out.end(), degrees.begin(), degrees.end());
    auto unit = tokenize_ascii_run_prefixed(piece_to_id, celsius ? "CELSIUS" : "FAHRENHEIT");
    out.insert(out.end(), unit.begin(), unit.end());
}

bool read_ascii_unit_suffix_pieces(const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
                                   const std::string& text,
                                   size_t suffix_pos,
                                   bool singular,
                                   std::vector<std::string>& suffix,
                                   size_t& suffix_end) {
    size_t pos = suffix_pos;
    std::string unit;
    while (pos < text.size()) {
        const size_t saved = pos;
        uint32_t cp = 0;
        std::string bytes;
        utf8_next(text, pos, cp, bytes);
        if (normalize_fullwidth_alnum(cp, bytes)) {
            pos = saved;
            break;
        }
        if (!is_ascii_alpha(cp)) {
            pos = saved;
            break;
        }
        unit += static_cast<char>(std::tolower(static_cast<unsigned char>(cp)));
    }
    if (unit.empty()) {
        return false;
    }
    if (pos < text.size()) {
        size_t boundary_pos = pos;
        uint32_t boundary_cp = 0;
        std::string boundary_bytes;
        utf8_next(text, boundary_pos, boundary_cp, boundary_bytes);
        normalize_fullwidth_alnum(boundary_cp, boundary_bytes);
        if (!is_ascii_punctuation(boundary_cp)) {
            return false;
        }
    }

    std::string unit_word;
    if (unit == "kg") {
        unit_word = singular ? "KILOGRAM" : "KILOGRAMS";
    } else if (unit == "kgs") {
        unit_word = "KGS";
    } else if (unit == "cm") {
        unit_word = singular ? "CENTIMETER" : "CENTIMETERS";
    } else if (unit == "mm") {
        unit_word = singular ? "MILLIMETER" : "MILLIMETERS";
    } else if (unit == "km") {
        unit_word = singular ? "KILOMETER" : "KILOMETERS";
    } else if (unit == "ml") {
        unit_word = singular ? "MILLILITER" : "MILLILITERS";
    } else if (unit == "g") {
        unit_word = "G";
    } else if (unit == "m") {
        unit_word = "M";
    } else if (unit == "l") {
        unit_word = "L";
    } else {
        return false;
    }

    suffix = tokenize_ascii_run_prefixed(piece_to_id, unit_word);
    suffix_end = pos;
    return true;
}

bool read_fullwidth_ascii_unit_suffix_pieces(const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
                                             const std::string& text,
                                             size_t suffix_pos,
                                             std::vector<std::string>& suffix,
                                             size_t& suffix_end) {
    size_t pos = suffix_pos;
    std::string unit;
    bool any_fullwidth = false;
    while (pos < text.size()) {
        const size_t saved = pos;
        uint32_t cp = 0;
        std::string bytes;
        utf8_next(text, pos, cp, bytes);
        const bool from_fullwidth = normalize_fullwidth_alnum(cp, bytes);
        if (!from_fullwidth || !is_ascii_alpha(cp)) {
            pos = saved;
            break;
        }
        any_fullwidth = true;
        unit += static_cast<char>(std::tolower(static_cast<unsigned char>(cp)));
    }
    if (!any_fullwidth || (unit != "kg" && unit != "kgs")) {
        return false;
    }
    if (pos < text.size()) {
        size_t boundary_pos = pos;
        uint32_t boundary_cp = 0;
        std::string boundary_bytes;
        utf8_next(text, boundary_pos, boundary_cp, boundary_bytes);
        normalize_fullwidth_alnum(boundary_cp, boundary_bytes);
        if (!is_ascii_punctuation(boundary_cp)) {
            return false;
        }
    }
    suffix = tokenize_ascii_run_prefixed(piece_to_id, ascii_upper(unit));
    suffix_end = pos;
    return true;
}

CjkTokenizedText tokenize_cjk_text(const std::unordered_map<std::string, TokenizerPieceInfo>& piece_to_id,
                                   const std::string& raw_text) {
    const std::string text = normalize_cjk_radicals(raw_text);
    const std::string space_piece = "\xe2\x96\x81";
    if (piece_to_id.find(space_piece) == piece_to_id.end()) {
        throw std::runtime_error("tokenizer pieces file does not contain SentencePiece space marker");
    }

    std::vector<std::string> pieces;
    size_t pos = 0;
    bool previous_was_ascii_word = false;
    bool ascii_run_follows_punctuation = false;
    bool previous_alpha_run_had_uppercase = false;
    size_t previous_alpha_run_length = 0;
    std::string previous_alpha_run_upper;
    bool previous_alpha_run_allows_english_number = false;
    bool previous_alpha_run_after_digit = false;
    bool previous_was_digit_run = false;
    bool decimal_fraction_digits = false;
    bool digit_run_allowed_after_punctuation = false;
    bool operator_rhs_degree_number_context = false;
    bool operator_rhs_measure_number_context = false;
    bool suppress_measure_number_context = false;
    bool signed_cjk_number_context = false;
    bool seen_cjk_context = false;
    while (pos < text.size()) {
        uint32_t cp = 0;
        std::string bytes;
        utf8_next(text, pos, cp, bytes);
        const bool current_from_fullwidth_alnum = normalize_fullwidth_alnum(cp, bytes);
        if (operator_rhs_degree_number_context && !is_ascii_digit(cp)) {
            operator_rhs_degree_number_context = false;
        }
        if (operator_rhs_measure_number_context && !is_ascii_digit(cp)) {
            operator_rhs_measure_number_context = false;
        }
        if (suppress_measure_number_context && !is_ascii_digit(cp)) {
            suppress_measure_number_context = false;
        }
        if (signed_cjk_number_context && !is_ascii_digit(cp)) {
            signed_cjk_number_context = false;
        }
        if (is_ascii_space(cp)) {
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = false;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = cp == 0xff0f || cp == 0x2026;
            continue;
        }
        if (is_cjk_codepoint(cp)) {
            pieces.push_back(space_piece);
            pieces.push_back(bytes);
            seen_cjk_context = true;
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = false;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = cp == 0xff0f || cp == 0x2026;
            continue;
        }
        if (cp == 0xff1a && previous_alpha_run_had_uppercase && previous_alpha_run_allows_english_number) {
            std::string alpha_colon_digits;
            size_t alpha_colon_end = pos;
            if (!read_ascii_digit_run_at(text, pos, 4, alpha_colon_digits, alpha_colon_end)) {
                if (seen_cjk_context || suffix_has_cjk(text, pos)) {
                    append_normalized_ascii_punctuation(pieces, piece_to_id, ",", previous_was_ascii_word);
                    previous_was_ascii_word = false;
                    ascii_run_follows_punctuation = true;
                    previous_alpha_run_had_uppercase = false;
                    previous_alpha_run_length = 0;
                    previous_alpha_run_allows_english_number = false;
                    previous_alpha_run_after_digit = false;
                    previous_was_digit_run = false;
                    decimal_fraction_digits = false;
                    digit_run_allowed_after_punctuation = false;
                    continue;
                }
                throw std::runtime_error("native CJK text-id export cannot encode alpha-colon form without trailing digits");
            }
            if (alpha_colon_end < text.size()) {
                size_t next_pos = alpha_colon_end;
                uint32_t next_cp = 0;
                std::string next_bytes;
                utf8_next(text, next_pos, next_cp, next_bytes);
                normalize_fullwidth_alnum(next_cp, next_bytes);
                if (is_ascii_digit(next_cp)) {
                    throw std::runtime_error("native CJK text-id export cannot encode alpha-colon digit run longer than 4 digits without full TextNormalizer");
                }
            }
            const bool cjk_number_context = seen_cjk_context || suffix_has_cjk(text, alpha_colon_end);
            append_alpha_colon_number_pieces(pieces, piece_to_id, alpha_colon_digits, cjk_number_context);
            pos = alpha_colon_end;
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = false;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = true;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = cp == 0xff0f;
            continue;
        }
        if (is_unknown_dash_like(cp)) {
            append_unknown_id_punctuation(pieces, piece_to_id, bytes, previous_was_ascii_word);
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = true;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = true;
            continue;
        }
        if (is_fullwidth_comma_like(cp) || cp == 0x3002 || cp == 0xff0e || cp == 0xff1f || is_dash_like(cp)) {
            const std::string ascii_piece = (cp == 0x3002 || cp == 0xff0e) ? "." : cp == 0xff1f ? "?" : ",";
            const std::string normalized = is_dash_like(cp) ? "-" : ascii_piece;
            append_normalized_ascii_punctuation(pieces, piece_to_id, normalized, previous_was_ascii_word);
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = true;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = normalized == "." || normalized == "-" || cp == 0xff1a;
            continue;
        }
        if (cp == 0xff01 || cp == 0xff05 || cp == 0xff0f || cp == '"' || is_quote_or_bracket_like(cp) || cp == 0x2026) {
            const std::string ascii_piece = cp == 0xff01 ? "!" : cp == 0xff05 ? "%" : cp == 0xff0f ? "/" : cp == 0x2026 ? "..." : "'";
            if (cp == 0x2026 && pos < text.size()) {
                const size_t saved = pos;
                uint32_t next_cp = 0;
                std::string next_bytes;
                if (utf8_next(text, pos, next_cp, next_bytes) && next_cp != 0x2026) {
                    pos = saved;
                }
            }
            if (is_quote_mark_like(cp)) {
                append_quote_mark_punctuation(pieces, piece_to_id, previous_was_ascii_word);
            } else {
                append_normalized_ascii_punctuation(pieces, piece_to_id, ascii_piece, previous_was_ascii_word);
            }
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = true;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = cp == 0xff0f || cp == 0x2026;
            continue;
        }
        bool currency_dollar = false;
        if (parse_currency_symbol(cp, currency_dollar)) {
            std::string integer_run;
            size_t integer_end = pos;
            if (!read_ascii_digit_run_at(text, pos, 5, integer_run, integer_end)) {
                throw std::runtime_error("native CJK text-id export supports currency symbols only before digit amounts");
            }
            if (integer_end < text.size()) {
                size_t next_pos = integer_end;
                uint32_t next_cp = 0;
                std::string next_bytes;
                utf8_next(text, next_pos, next_cp, next_bytes);
                normalize_fullwidth_alnum(next_cp, next_bytes);
                if (is_ascii_digit(next_cp)) {
                    throw std::runtime_error("native CJK text-id export currency integer amount exceeds focused native range");
                }
            }

            std::string fraction_run;
            size_t currency_end = integer_end;
            if (integer_end < text.size()) {
                size_t dot_pos = integer_end;
                uint32_t dot_cp = 0;
                std::string dot_bytes;
                utf8_next(text, dot_pos, dot_cp, dot_bytes);
                if (dot_cp == '.') {
                    std::string candidate_fraction;
                    size_t fraction_end = dot_pos;
                    if (read_ascii_digit_run_at(text, dot_pos, 6, candidate_fraction, fraction_end)) {
                        bool next_is_digit = false;
                        if (fraction_end < text.size()) {
                            size_t next_pos = fraction_end;
                            uint32_t next_cp = 0;
                            std::string next_bytes;
                            utf8_next(text, next_pos, next_cp, next_bytes);
                            normalize_fullwidth_alnum(next_cp, next_bytes);
                            next_is_digit = is_ascii_digit(next_cp);
                        }
                        if (!next_is_digit) {
                            fraction_run = candidate_fraction;
                            currency_end = fraction_end;
                        }
                    }
                }
            }

            std::vector<std::string> currency_pieces;
            if (fraction_run.empty() && integer_run.size() >= 3 &&
                integer_run[0] == '0' && integer_run[1] == '0') {
                currency_pieces = currency_dollar
                    ? tokenize_multi_leading_zero_dollar_as_reference_digits(integer_run)
                    : tokenize_multi_leading_zero_yen_as_reference_digits(integer_run);
            } else if (integer_run.size() >= 2 && integer_run.size() <= 5 &&
                       integer_run[0] == '0' && integer_run[1] != '0') {
                currency_pieces = tokenize_leading_zero_currency_as_cjk(integer_run, fraction_run, currency_dollar);
            } else {
                currency_pieces = tokenize_currency_number_as_cjk(integer_run, fraction_run, currency_dollar);
            }
            pieces.insert(pieces.end(), currency_pieces.begin(), currency_pieces.end());
            pos = currency_end;
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = false;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = false;
            continue;
        }
        if (is_ascii_digit(cp)) {
            if (ascii_run_follows_punctuation && !digit_run_allowed_after_punctuation) {
                // Identifier-style digit run right after punctuation (e.g. the
                // "2023" in "CVE-2023-3420"): the full TextNormalizer is not
                // available, so read it digit-by-digit instead of failing.
                std::string id_run = bytes;
                while (pos < text.size()) {
                    const size_t saved = pos;
                    uint32_t next_cp = 0;
                    std::string next_bytes;
                    utf8_next(text, pos, next_cp, next_bytes);
                    normalize_fullwidth_alnum(next_cp, next_bytes);
                    if (!is_ascii_digit(next_cp)) {
                        pos = saved;
                        break;
                    }
                    id_run += next_bytes;
                }
                std::cerr << "warning: digit run '" << id_run
                          << "' after punctuation has no native number normalization; reading digit-by-digit" << std::endl;
                const auto digit_pieces = tokenize_digit_run_as_cjk(id_run, true);
                pieces.insert(pieces.end(), digit_pieces.begin(), digit_pieces.end());
                previous_was_ascii_word = false;
                ascii_run_follows_punctuation = false;
                previous_alpha_run_had_uppercase = false;
                previous_alpha_run_length = 0;
                previous_alpha_run_allows_english_number = false;
                previous_alpha_run_after_digit = false;
                previous_was_digit_run = true;
                decimal_fraction_digits = false;
                digit_run_allowed_after_punctuation = false;
                continue;
            }
            const bool digit_run_is_operator_rhs_degree = operator_rhs_degree_number_context;
            const bool digit_run_is_operator_rhs_measure = operator_rhs_measure_number_context;
            const bool digit_run_suppresses_measure = suppress_measure_number_context;
            const bool digit_run_is_signed_cjk_number = signed_cjk_number_context;
            operator_rhs_degree_number_context = false;
            operator_rhs_measure_number_context = false;
            suppress_measure_number_context = false;
            signed_cjk_number_context = false;
            std::string run = bytes;
            bool digit_run_from_fullwidth = current_from_fullwidth_alnum;
            while (pos < text.size()) {
                const size_t saved = pos;
                uint32_t next_cp = 0;
                std::string next_bytes;
                utf8_next(text, pos, next_cp, next_bytes);
                const bool next_from_fullwidth = normalize_fullwidth_alnum(next_cp, next_bytes);
                if (!is_ascii_digit(next_cp)) {
                    pos = saved;
                    break;
                }
                digit_run_from_fullwidth = digit_run_from_fullwidth || next_from_fullwidth;
                run += next_bytes;
            }
            const size_t after_digits = pos;
            bool has_date = false;
            int date_month = 0;
            int date_day = 0;
            size_t date_end = after_digits;
            bool has_month_day_date = false;
            int month_day_month = 0;
            int month_day_day = 0;
            size_t month_day_end = after_digits;
            bool has_fraction = false;
            std::string fraction_denominator;
            size_t fraction_end = after_digits;
            bool has_leading_zero_denominator_fraction = false;
            std::string leading_zero_denominator;
            size_t leading_zero_denominator_fraction_end = after_digits;
            bool has_chained_slash_fraction = false;
            std::string chained_slash_numerator;
            std::string chained_slash_denominator;
            size_t chained_slash_fraction_end = after_digits;
            bool has_short_hyphen_subtraction = false;
            std::string short_hyphen_subtraction_rhs;
            size_t short_hyphen_subtraction_end = after_digits;
            bool has_phone = false;
            std::string phone_digits;
            size_t phone_end = after_digits;
            bool has_fullwidth_hyphen_grouped_number = false;
            std::string fullwidth_hyphen_group0;
            std::string fullwidth_hyphen_group1;
            std::string fullwidth_hyphen_group2;
            size_t fullwidth_hyphen_grouped_number_end = after_digits;
            bool has_time = false;
            std::string time_minute;
            std::optional<std::string> time_second;
            size_t time_end = after_digits;
            bool has_fullwidth_colon_digit_separator = false;
            std::string fullwidth_colon_rhs;
            size_t fullwidth_colon_end = after_digits;
            std::string percent_fraction;
            bool has_percent = false;
            size_t percent_end = after_digits;
            bool has_temperature_unit = false;
            bool temperature_unit_is_english = false;
            bool temperature_celsius = true;
            size_t temperature_end = after_digits;
            bool has_operator_rhs_degree_word = false;
            size_t operator_rhs_degree_word_end = after_digits;
            bool has_cjk_measure_unit = false;
            std::vector<std::string> cjk_measure_suffix_pieces;
            size_t cjk_measure_end = after_digits;
            bool has_ascii_unit_suffix = false;
            std::vector<std::string> ascii_unit_suffix_pieces;
            size_t ascii_unit_end = after_digits;
            bool has_ascii_decimal_unit_suffix = false;
            std::string ascii_unit_fraction_digits;
            std::vector<std::string> ascii_decimal_unit_suffix_pieces;
            size_t ascii_decimal_unit_end = after_digits;
            bool has_fullwidth_ascii_unit_suffix = false;
            std::vector<std::string> fullwidth_ascii_unit_suffix_pieces;
            size_t fullwidth_ascii_unit_end = after_digits;
            if (!previous_alpha_run_after_digit && after_digits < text.size() &&
                read_fullwidth_ascii_unit_suffix_pieces(piece_to_id,
                                                       text,
                                                       after_digits,
                                                       fullwidth_ascii_unit_suffix_pieces,
                                                       fullwidth_ascii_unit_end)) {
                has_fullwidth_ascii_unit_suffix = true;
            }
            if (!has_fullwidth_ascii_unit_suffix &&
                !previous_alpha_run_had_uppercase && !previous_alpha_run_after_digit && after_digits < text.size()) {
                size_t suffix_pos = after_digits;
                uint32_t suffix_cp = 0;
                std::string suffix_bytes;
                utf8_next(text, suffix_pos, suffix_cp, suffix_bytes);
                const bool suffix_from_fullwidth = normalize_fullwidth_alnum(suffix_cp, suffix_bytes);
                if (is_ascii_alpha(suffix_cp)) {
                    if (seen_cjk_context || decimal_fraction_digits || digit_run_from_fullwidth || suffix_from_fullwidth ||
                        !read_ascii_unit_suffix_pieces(piece_to_id,
                                                       text,
                                                       after_digits,
                                                       english_unit_number_is_singular(run),
                                                       ascii_unit_suffix_pieces,
                                                       ascii_unit_end)) {
                        throw std::runtime_error(
                            "native CJK text-id export cannot encode ASCII unit suffix after digit run without full TextNormalizer: " +
                            run);
                    }
                    has_ascii_unit_suffix = true;
                } else if (suffix_cp == '.' && !seen_cjk_context && !decimal_fraction_digits &&
                           !digit_run_from_fullwidth) {
                    size_t fraction_pos = suffix_pos;
                    std::string fraction;
                    bool fraction_from_fullwidth = false;
                    while (fraction_pos < text.size()) {
                        const size_t saved = fraction_pos;
                        uint32_t fraction_cp = 0;
                        std::string fraction_bytes;
                        utf8_next(text, fraction_pos, fraction_cp, fraction_bytes);
                        const bool digit_from_fullwidth = normalize_fullwidth_alnum(fraction_cp, fraction_bytes);
                        if (!is_ascii_digit(fraction_cp)) {
                            fraction_pos = saved;
                            break;
                        }
                        fraction_from_fullwidth = fraction_from_fullwidth || digit_from_fullwidth;
                        fraction += fraction_bytes;
                    }
                    if (!fraction.empty() && !fraction_from_fullwidth && fraction_pos < text.size()) {
                        size_t unit_probe_pos = fraction_pos;
                        uint32_t unit_probe_cp = 0;
                        std::string unit_probe_bytes;
                        utf8_next(text, unit_probe_pos, unit_probe_cp, unit_probe_bytes);
                        const bool unit_probe_from_fullwidth = normalize_fullwidth_alnum(unit_probe_cp, unit_probe_bytes);
                        if (is_ascii_alpha(unit_probe_cp) && !unit_probe_from_fullwidth &&
                            read_ascii_unit_suffix_pieces(piece_to_id,
                                                          text,
                                                          fraction_pos,
                                                          false,
                                                          ascii_decimal_unit_suffix_pieces,
                                                          ascii_decimal_unit_end)) {
                            has_ascii_decimal_unit_suffix = true;
                            ascii_unit_fraction_digits = fraction;
                        }
                    }
                }
            }
            if (!decimal_fraction_digits && run.size() == 4 && after_digits < text.size()) {
                size_t lookahead = after_digits;
                uint32_t sep_cp = 0;
                std::string sep_bytes;
                utf8_next(text, lookahead, sep_cp, sep_bytes);
                char date_separator = 0;
                if (parse_date_separator(sep_cp, date_separator)) {
                    std::string month_run;
                    size_t month_end = lookahead;
                    if (read_ascii_digit_run_at(text, lookahead, 2, month_run, month_end) && month_end < text.size()) {
                        size_t sep2_pos = month_end;
                        uint32_t sep2_cp = 0;
                        std::string sep2_bytes;
                        utf8_next(text, sep2_pos, sep2_cp, sep2_bytes);
                        char date_separator2 = 0;
                        if (parse_date_separator(sep2_cp, date_separator2) && date_separator2 == date_separator) {
                            std::string day_run;
                            size_t day_end = sep2_pos;
                            if (read_ascii_digit_run_at(text, sep2_pos, 2, day_run, day_end)) {
                                bool next_is_digit = false;
                                if (day_end < text.size()) {
                                    size_t next_pos = day_end;
                                    uint32_t next_cp = 0;
                                    std::string next_bytes;
                                    utf8_next(text, next_pos, next_cp, next_bytes);
                                    normalize_fullwidth_alnum(next_cp, next_bytes);
                                    next_is_digit = is_ascii_digit(next_cp);
                                }
                                const int month_value = std::stoi(month_run);
                                const int day_value = std::stoi(day_run);
                                if (!next_is_digit && month_value >= 1 && month_value <= 12 && day_value >= 1 && day_value <= 31) {
                                    has_date = true;
                                    date_month = month_value;
                                    date_day = day_value;
                                    date_end = day_end;
                                }
                            }
                        }
                    }
                }
            }
            if (!has_date && !decimal_fraction_digits && !digit_run_from_fullwidth &&
                run.size() <= 2 && after_digits < text.size()) {
                size_t lookahead = after_digits;
                uint32_t sep_cp = 0;
                std::string sep_bytes;
                utf8_next(text, lookahead, sep_cp, sep_bytes);
                if (sep_cp == '/') {
                    std::string rhs_run;
                    bool rhs_from_fullwidth = false;
                    size_t rhs_end = lookahead;
                    if (read_ascii_digit_run_at(text, lookahead, 2, rhs_run, rhs_end, &rhs_from_fullwidth) &&
                        !rhs_from_fullwidth) {
                        bool next_is_digit = false;
                        bool next_is_slash = false;
                        if (rhs_end < text.size()) {
                            size_t next_pos = rhs_end;
                            uint32_t next_cp = 0;
                            std::string next_bytes;
                            utf8_next(text, next_pos, next_cp, next_bytes);
                            normalize_fullwidth_alnum(next_cp, next_bytes);
                            next_is_digit = is_ascii_digit(next_cp);
                            next_is_slash = next_cp == '/';
                        }
                        const int lhs_value = std::stoi(run);
                        const int rhs_value = std::stoi(rhs_run);
                        const bool month_day_shape = run.size() == 2 || lhs_value >= 10;
                        if (!next_is_digit && !next_is_slash && month_day_shape &&
                            lhs_value >= 1 && lhs_value <= 12 && rhs_value >= 1 && rhs_value <= 31) {
                            has_month_day_date = true;
                            month_day_month = lhs_value;
                            month_day_day = rhs_value;
                            month_day_end = rhs_end;
                        } else if (!next_is_digit && !next_is_slash &&
                                   !(run.size() > 1 && run[0] == '0') &&
                                   !(rhs_run.size() > 1 && rhs_run[0] == '0') &&
                                   rhs_value != 0) {
                            has_fraction = true;
                            fraction_denominator = rhs_run;
                            fraction_end = rhs_end;
                        } else if (!next_is_digit && next_is_slash &&
                                   run.size() == 1 && rhs_run.size() == 1 &&
                                   !(run[0] == '0') && rhs_value != 0) {
                            size_t chain_denominator_start = rhs_end;
                            uint32_t chain_separator_cp = 0;
                            std::string chain_separator_bytes;
                            utf8_next(text, chain_denominator_start, chain_separator_cp, chain_separator_bytes);
                            if (chain_separator_cp != '/') {
                                throw std::runtime_error("native CJK text-id export chained slash helper lost slash separator");
                            }
                            std::string chain_denominator_run;
                            bool chain_denominator_from_fullwidth = false;
                            size_t chain_denominator_end = chain_denominator_start;
                            if (read_ascii_digit_run_at(text,
                                                        chain_denominator_start,
                                                        1,
                                                        chain_denominator_run,
                                                        chain_denominator_end,
                                                        &chain_denominator_from_fullwidth) &&
                                !chain_denominator_from_fullwidth && chain_denominator_run.size() == 1) {
                                bool chain_next_is_digit = false;
                                bool chain_next_is_slash = false;
                                if (chain_denominator_end < text.size()) {
                                    size_t chain_next_pos = chain_denominator_end;
                                    uint32_t chain_next_cp = 0;
                                    std::string chain_next_bytes;
                                    utf8_next(text, chain_next_pos, chain_next_cp, chain_next_bytes);
                                    normalize_fullwidth_alnum(chain_next_cp, chain_next_bytes);
                                    chain_next_is_digit = is_ascii_digit(chain_next_cp);
                                    chain_next_is_slash = chain_next_cp == '/';
                                }
                                if (!chain_next_is_digit && !chain_next_is_slash &&
                                    chain_denominator_run[0] != '0') {
                                    has_chained_slash_fraction = true;
                                    chained_slash_numerator = rhs_run;
                                    chained_slash_denominator = chain_denominator_run;
                                    chained_slash_fraction_end = chain_denominator_end;
                                }
                            }
                        }
                    }
                    if (!has_month_day_date && !has_fraction && !has_chained_slash_fraction && run.size() == 1) {
                        std::string leading_rhs_run;
                        bool leading_rhs_from_fullwidth = false;
                        size_t leading_rhs_end = lookahead;
                        if (read_ascii_digit_run_at(text, lookahead, 3, leading_rhs_run, leading_rhs_end, &leading_rhs_from_fullwidth) &&
                            !leading_rhs_from_fullwidth &&
                            leading_rhs_run.size() >= 2 &&
                            leading_rhs_run[0] == '0') {
                            bool next_is_digit = false;
                            bool next_is_slash = false;
                            if (leading_rhs_end < text.size()) {
                                size_t next_pos = leading_rhs_end;
                                uint32_t next_cp = 0;
                                std::string next_bytes;
                                utf8_next(text, next_pos, next_cp, next_bytes);
                                normalize_fullwidth_alnum(next_cp, next_bytes);
                                next_is_digit = is_ascii_digit(next_cp);
                                next_is_slash = next_cp == '/';
                            }
                            if (!next_is_digit && !next_is_slash) {
                                has_leading_zero_denominator_fraction = true;
                                leading_zero_denominator = leading_rhs_run;
                                leading_zero_denominator_fraction_end = leading_rhs_end;
                            }
                        }
                    }
                }
            }
            if (!has_date && !has_month_day_date && !has_fraction && !has_leading_zero_denominator_fraction &&
                !has_chained_slash_fraction && !decimal_fraction_digits &&
                !digit_run_from_fullwidth && run.size() <= 2 && after_digits < text.size()) {
                size_t lookahead = after_digits;
                uint32_t sep_cp = 0;
                std::string sep_bytes;
                utf8_next(text, lookahead, sep_cp, sep_bytes);
                if (sep_cp == '-') {
                    std::string rhs_run;
                    bool rhs_from_fullwidth = false;
                    size_t rhs_end = lookahead;
                    if (read_ascii_digit_run_at(text, lookahead, 2, rhs_run, rhs_end, &rhs_from_fullwidth) &&
                        !rhs_from_fullwidth &&
                        !(run.size() > 1 && run[0] == '0') &&
                        !(rhs_run.size() > 1 && rhs_run[0] == '0')) {
                        bool next_is_digit = false;
                        bool next_is_hyphen = false;
                        if (rhs_end < text.size()) {
                            size_t next_pos = rhs_end;
                            uint32_t next_cp = 0;
                            std::string next_bytes;
                            utf8_next(text, next_pos, next_cp, next_bytes);
                            normalize_fullwidth_alnum(next_cp, next_bytes);
                            next_is_digit = is_ascii_digit(next_cp);
                            next_is_hyphen = next_cp == '-';
                        }
                        if (!next_is_digit && !next_is_hyphen) {
                            has_short_hyphen_subtraction = true;
                            short_hyphen_subtraction_rhs = rhs_run;
                            short_hyphen_subtraction_end = rhs_end;
                        }
                    }
                }
            }
            if (!has_date && !has_month_day_date && !has_fraction && !has_leading_zero_denominator_fraction &&
                !has_chained_slash_fraction && !has_short_hyphen_subtraction &&
                !decimal_fraction_digits && run.size() == 3 && after_digits < text.size()) {
                std::vector<std::string> phone_groups{run};
                size_t phone_cursor = after_digits;
                bool saw_phone_separator = false;
                while (phone_cursor < text.size()) {
                    size_t sep_pos = phone_cursor;
                    uint32_t sep_cp = 0;
                    std::string sep_bytes;
                    utf8_next(text, sep_pos, sep_cp, sep_bytes);
                    if (sep_cp != '-') {
                        break;
                    }
                    saw_phone_separator = true;
                    std::string group_run;
                    size_t group_end = sep_pos;
                    if (!read_ascii_digit_run_at(text, sep_pos, 8, group_run, group_end)) {
                        break;
                    }
                    bool next_is_digit = false;
                    if (group_end < text.size()) {
                        size_t next_pos = group_end;
                        uint32_t next_cp = 0;
                        std::string next_bytes;
                        utf8_next(text, next_pos, next_cp, next_bytes);
                        normalize_fullwidth_alnum(next_cp, next_bytes);
                        next_is_digit = is_ascii_digit(next_cp);
                    }
                    if (next_is_digit) {
                        break;
                    }
                    phone_groups.push_back(group_run);
                    phone_cursor = group_end;
                }
                if (saw_phone_separator) {
                    const bool phone_3_4_4 = phone_groups.size() == 3 &&
                                             phone_groups[0].size() == 3 &&
                                             phone_groups[1].size() == 4 &&
                                             phone_groups[2].size() == 4;
                    const bool phone_3_3_4 = phone_groups.size() == 3 &&
                                             phone_groups[0].size() == 3 &&
                                             phone_groups[1].size() == 3 &&
                                             phone_groups[2].size() == 4;
                    const bool phone_3_8 = phone_groups.size() == 2 &&
                                           phone_groups[0].size() == 3 &&
                                           phone_groups[1].size() == 8;
                    if (phone_3_4_4 || phone_3_3_4 || phone_3_8) {
                        has_phone = true;
                        phone_digits.clear();
                        for (const auto& group : phone_groups) {
                            phone_digits += group;
                        }
                        phone_end = phone_cursor;
                    }
                }
            }
            if (!has_date && !has_month_day_date && !has_fraction && !has_leading_zero_denominator_fraction &&
                !has_chained_slash_fraction && !has_short_hyphen_subtraction && !has_phone &&
                !decimal_fraction_digits && run.size() == 3 && after_digits < text.size()) {
                size_t sep0_end = after_digits;
                uint32_t sep0_cp = 0;
                std::string sep0_bytes;
                utf8_next(text, sep0_end, sep0_cp, sep0_bytes);
                if (sep0_cp == 0xff0d) {
                    std::string group1;
                    size_t group1_end = sep0_end;
                    if (read_ascii_digit_run_at(text, sep0_end, 4, group1, group1_end) &&
                        group1.size() == 4 && group1_end < text.size()) {
                        size_t sep1_end = group1_end;
                        uint32_t sep1_cp = 0;
                        std::string sep1_bytes;
                        utf8_next(text, sep1_end, sep1_cp, sep1_bytes);
                        if (sep1_cp == 0xff0d) {
                            std::string group2;
                            size_t group2_end = sep1_end;
                            if (read_ascii_digit_run_at(text, sep1_end, 4, group2, group2_end) &&
                                group2.size() == 4) {
                                bool next_is_digit = false;
                                if (group2_end < text.size()) {
                                    size_t next_pos = group2_end;
                                    uint32_t next_cp = 0;
                                    std::string next_bytes;
                                    utf8_next(text, next_pos, next_cp, next_bytes);
                                    normalize_fullwidth_alnum(next_cp, next_bytes);
                                    next_is_digit = is_ascii_digit(next_cp);
                                }
                                if (!next_is_digit) {
                                    has_fullwidth_hyphen_grouped_number = true;
                                    fullwidth_hyphen_group0 = run;
                                    fullwidth_hyphen_group1 = group1;
                                    fullwidth_hyphen_group2 = group2;
                                    fullwidth_hyphen_grouped_number_end = group2_end;
                                }
                            }
                        }
                    }
                }
            }
            if (!has_date && !has_month_day_date && !has_fraction && !has_leading_zero_denominator_fraction &&
                !has_chained_slash_fraction && !has_short_hyphen_subtraction && !has_phone &&
                !has_fullwidth_hyphen_grouped_number &&
                !decimal_fraction_digits && !digit_run_from_fullwidth &&
                run.size() <= 2 && after_digits < text.size()) {
                size_t lookahead = after_digits;
                uint32_t sep_cp = 0;
                std::string sep_bytes;
                utf8_next(text, lookahead, sep_cp, sep_bytes);
                if (parse_time_separator(sep_cp)) {
                    std::string minute_run;
                    bool minute_from_fullwidth = false;
                    size_t minute_end = lookahead;
                    if (read_ascii_digit_run_at(text, lookahead, 2, minute_run, minute_end, &minute_from_fullwidth) &&
                        minute_run.size() == 2 && !minute_from_fullwidth) {
                        bool next_is_digit = false;
                        bool has_seconds_candidate = false;
                        std::string second_run;
                        size_t second_end = minute_end;
                        if (minute_end < text.size()) {
                            size_t next_pos = minute_end;
                            uint32_t next_cp = 0;
                            std::string next_bytes;
                            utf8_next(text, next_pos, next_cp, next_bytes);
                            const bool next_from_fullwidth = normalize_fullwidth_alnum(next_cp, next_bytes);
                            next_is_digit = is_ascii_digit(next_cp);
                            if (parse_time_separator(next_cp)) {
                                bool second_from_fullwidth = false;
                                if (read_ascii_digit_run_at(text, next_pos, 2, second_run, second_end, &second_from_fullwidth) &&
                                    second_run.size() == 2 && !second_from_fullwidth) {
                                    bool second_next_is_digit = false;
                                    if (second_end < text.size()) {
                                        size_t after_second_pos = second_end;
                                        uint32_t after_second_cp = 0;
                                        std::string after_second_bytes;
                                        utf8_next(text, after_second_pos, after_second_cp, after_second_bytes);
                                        normalize_fullwidth_alnum(after_second_cp, after_second_bytes);
                                        second_next_is_digit = is_ascii_digit(after_second_cp);
                                    }
                                    if (!second_next_is_digit) {
                                        has_seconds_candidate = true;
                                    }
                                }
                            } else {
                                (void)next_from_fullwidth;
                            }
                        }
                        const int hour_value = std::stoi(run);
                        const int minute_value = std::stoi(minute_run);
                        if (!next_is_digit && hour_value >= 0 && hour_value <= 24 && minute_value >= 0 && minute_value <= 59 &&
                            !(run.size() == 1 && run[0] == '0')) {
                            has_time = true;
                            time_minute = minute_run;
                            time_end = minute_end;
                            if (has_seconds_candidate) {
                                const int second_value = std::stoi(second_run);
                                if (second_value >= 0 && second_value <= 59) {
                                    time_second = second_run;
                                    time_end = second_end;
                                } else {
                                    has_time = false;
                                    time_second.reset();
                                }
                            }
                        }
                    }
                }
            }
            if (!has_date && !has_month_day_date && !has_fraction && !has_leading_zero_denominator_fraction &&
                !has_chained_slash_fraction && !has_short_hyphen_subtraction && !has_time &&
                !decimal_fraction_digits && after_digits < text.size()) {
                size_t lookahead = after_digits;
                uint32_t sep_cp = 0;
                std::string sep_bytes;
                utf8_next(text, lookahead, sep_cp, sep_bytes);
                if (sep_cp == 0xff1a) {
                    std::string rhs_run;
                    size_t rhs_end = lookahead;
                    if (read_ascii_digit_run_at(text, lookahead, 2, rhs_run, rhs_end)) {
                        bool next_is_digit = false;
                        if (rhs_end < text.size()) {
                            size_t next_pos = rhs_end;
                            uint32_t next_cp = 0;
                            std::string next_bytes;
                            utf8_next(text, next_pos, next_cp, next_bytes);
                            normalize_fullwidth_alnum(next_cp, next_bytes);
                            next_is_digit = is_ascii_digit(next_cp);
                        }
                        if (!next_is_digit) {
                            has_fullwidth_colon_digit_separator = true;
                            fullwidth_colon_rhs = rhs_run;
                            fullwidth_colon_end = rhs_end;
                        }
                    }
                }
            }
            if (!has_date && !has_month_day_date && !has_fraction && !has_leading_zero_denominator_fraction &&
                !has_chained_slash_fraction && !has_short_hyphen_subtraction && !has_time &&
                !has_fullwidth_colon_digit_separator &&
                !decimal_fraction_digits && after_digits < text.size()) {
                size_t lookahead = after_digits;
                uint32_t next_cp = 0;
                std::string next_bytes;
                utf8_next(text, lookahead, next_cp, next_bytes);
                normalize_fullwidth_alnum(next_cp, next_bytes);
                if (next_cp == 0x2103 || next_cp == 0x2109) {
                    has_temperature_unit = true;
                    temperature_celsius = next_cp == 0x2103;
                    temperature_end = lookahead;
                } else if (next_cp == 0x00b0 && lookahead < text.size()) {
                    size_t unit_pos = lookahead;
                    uint32_t unit_cp = 0;
                    std::string unit_bytes;
                    utf8_next(text, unit_pos, unit_cp, unit_bytes);
                    normalize_fullwidth_alnum(unit_cp, unit_bytes);
                    if (unit_cp == 'C' || unit_cp == 'c' || unit_cp == 'F' || unit_cp == 'f') {
                        has_temperature_unit = true;
                        temperature_unit_is_english = !seen_cjk_context;
                        temperature_celsius = unit_cp == 'C' || unit_cp == 'c';
                        temperature_end = unit_pos;
                    }
                } else if (next_cp == '%') {
                    has_percent = true;
                    percent_end = lookahead;
                } else if (next_cp == '.') {
                    size_t fraction_pos = lookahead;
                    std::string fraction;
                    while (fraction_pos < text.size()) {
                        const size_t saved = fraction_pos;
                        uint32_t fraction_cp = 0;
                        std::string fraction_bytes;
                        utf8_next(text, fraction_pos, fraction_cp, fraction_bytes);
                        normalize_fullwidth_alnum(fraction_cp, fraction_bytes);
                        if (!is_ascii_digit(fraction_cp)) {
                            fraction_pos = saved;
                            break;
                        }
                        fraction += fraction_bytes;
                    }
                    if (!fraction.empty() && fraction_pos < text.size()) {
                        size_t percent_pos = fraction_pos;
                        uint32_t percent_cp = 0;
                        std::string percent_bytes;
                        utf8_next(text, percent_pos, percent_cp, percent_bytes);
                        if (percent_cp == '%') {
                            has_percent = true;
                            percent_fraction = fraction;
                            percent_end = percent_pos;
                        }
                    }
                }
            }
            if ((digit_run_is_operator_rhs_degree || digit_run_is_signed_cjk_number) &&
                !has_temperature_unit && after_digits < text.size()) {
                size_t degree_pos = after_digits;
                uint32_t degree_cp = 0;
                std::string degree_bytes;
                utf8_next(text, degree_pos, degree_cp, degree_bytes);
                if (degree_cp == 0x5ea6) {
                    has_operator_rhs_degree_word = true;
                    operator_rhs_degree_word_end = degree_pos;
                }
            }
            if (!digit_run_suppresses_measure && !has_temperature_unit && !has_operator_rhs_degree_word &&
                !has_percent && !has_fullwidth_colon_digit_separator && after_digits < text.size()) {
                has_cjk_measure_unit = read_cjk_measure_suffix_pieces(
                    text,
                    after_digits,
                    cjk_measure_suffix_pieces,
                    cjk_measure_end);
            }
            bool later_has_cjk = false;
            if (!has_percent && after_digits < text.size()) {
                later_has_cjk = suffix_has_cjk(text, after_digits);
            }
            bool followed_by_year_marker = false;
            if (after_digits < text.size()) {
                size_t next_pos = after_digits;
                uint32_t next_cp = 0;
                std::string next_bytes;
                utf8_next(text, next_pos, next_cp, next_bytes);
                followed_by_year_marker = next_cp == 0x5e74;
            }
            std::vector<std::string> run_pieces;
            if (has_date) {
                pos = date_end;
                append_date_ymd_pieces(run_pieces, run, date_month, date_day);
            } else if (has_month_day_date) {
                pos = month_day_end;
                append_date_md_pieces(run_pieces, month_day_month, month_day_day);
            } else if (has_fraction) {
                pos = fraction_end;
                run_pieces = tokenize_fraction_as_cjk(run, fraction_denominator);
            } else if (has_leading_zero_denominator_fraction) {
                pos = leading_zero_denominator_fraction_end;
                run_pieces = tokenize_leading_zero_denominator_fraction_as_cjk(run, leading_zero_denominator);
            } else if (has_chained_slash_fraction) {
                pos = chained_slash_fraction_end;
                run_pieces = tokenize_single_digit_chained_slash_fraction_as_cjk(
                    piece_to_id,
                    run,
                    chained_slash_numerator,
                    chained_slash_denominator);
            } else if (has_short_hyphen_subtraction) {
                pos = short_hyphen_subtraction_end;
                run_pieces = tokenize_short_hyphen_subtraction_as_cjk(run, short_hyphen_subtraction_rhs);
            } else if (has_phone) {
                pos = phone_end;
                run_pieces = tokenize_phone_digits_as_cjk(phone_digits);
            } else if (has_fullwidth_hyphen_grouped_number) {
                pos = fullwidth_hyphen_grouped_number_end;
                run_pieces = tokenize_fullwidth_hyphen_grouped_3_4_4_as_cjk(
                    fullwidth_hyphen_group0,
                    fullwidth_hyphen_group1,
                    fullwidth_hyphen_group2);
            } else if (has_time) {
                pos = time_end;
                append_time_hms_pieces(run_pieces, run, time_minute, time_second);
            } else if (has_fullwidth_colon_digit_separator) {
                pos = fullwidth_colon_end;
                run_pieces = tokenize_digit_run_as_cjk_integer(run, decimal_fraction_digits);
                append_normalized_ascii_punctuation(run_pieces, piece_to_id, ",", false);
                auto rhs_pieces = tokenize_digit_run_as_cjk_integer(fullwidth_colon_rhs, false);
                run_pieces.insert(run_pieces.end(), rhs_pieces.begin(), rhs_pieces.end());
            } else if (has_temperature_unit) {
                pos = temperature_end;
                if (temperature_unit_is_english) {
                    if (digit_run_from_fullwidth || decimal_fraction_digits || run.size() > 3) {
                        throw std::runtime_error("native CJK text-id export cannot encode English temperature number without full TextNormalizer: " + run);
                    }
                    run_pieces = tokenize_english_number_run(run);
                    append_english_temperature_unit_pieces(run_pieces, piece_to_id, temperature_celsius);
                } else if (digit_run_is_operator_rhs_degree) {
                    run_pieces = tokenize_operator_rhs_degree_number_as_cjk(run, decimal_fraction_digits);
                    append_temperature_unit_pieces(run_pieces, temperature_celsius);
                } else if (digit_run_is_signed_cjk_number) {
                    run_pieces = tokenize_digit_run_as_cjk_integer(run, decimal_fraction_digits);
                    append_temperature_unit_pieces(run_pieces, temperature_celsius);
                } else {
                    run_pieces = tokenize_temperature_number_as_cjk(run, decimal_fraction_digits);
                    append_temperature_unit_pieces(run_pieces, temperature_celsius);
                }
            } else if (has_operator_rhs_degree_word) {
                pos = operator_rhs_degree_word_end;
                run_pieces = digit_run_is_operator_rhs_degree
                    ? tokenize_operator_rhs_degree_number_as_cjk(run, decimal_fraction_digits)
                    : tokenize_digit_run_as_cjk_integer(run, decimal_fraction_digits);
                append_cjk_piece(run_pieces, "\xe5\xba\xa6");
            } else if (has_cjk_measure_unit) {
                pos = cjk_measure_end;
                if (decimal_fraction_digits) {
                    run_pieces = tokenize_digit_run_as_cjk(run, true);
                } else {
                    run_pieces = digit_run_is_operator_rhs_measure
                        ? tokenize_operator_rhs_measure_number_as_cjk(run, false)
                        : tokenize_measure_number_as_cjk(run, false);
                }
                run_pieces.insert(run_pieces.end(), cjk_measure_suffix_pieces.begin(), cjk_measure_suffix_pieces.end());
            } else if (has_ascii_decimal_unit_suffix) {
                pos = ascii_decimal_unit_end;
                run_pieces = tokenize_english_unit_decimal_number_run(run, ascii_unit_fraction_digits);
                run_pieces.insert(run_pieces.end(), ascii_decimal_unit_suffix_pieces.begin(), ascii_decimal_unit_suffix_pieces.end());
            } else if (has_ascii_unit_suffix) {
                pos = ascii_unit_end;
                run_pieces = tokenize_english_unit_number_run(run);
                run_pieces.insert(run_pieces.end(), ascii_unit_suffix_pieces.begin(), ascii_unit_suffix_pieces.end());
            } else if (has_fullwidth_ascii_unit_suffix) {
                pos = fullwidth_ascii_unit_end;
                if (seen_cjk_context || digit_run_from_fullwidth) {
                    throw std::runtime_error("native CJK text-id export fullwidth ASCII unit suffix needs full TextNormalizer for this number: " + run);
                }
                run_pieces = decimal_fraction_digits
                    ? tokenize_digit_run_as_cjk(run, true)
                    : tokenize_digit_run_as_cjk_integer(run, false);
                run_pieces.insert(run_pieces.end(), fullwidth_ascii_unit_suffix_pieces.begin(), fullwidth_ascii_unit_suffix_pieces.end());
            } else if (has_percent) {
                pos = percent_end;
                run_pieces = tokenize_percent_number_as_cjk(run, percent_fraction);
            } else if (previous_alpha_run_length == 1 && previous_alpha_run_upper == "V") {
                // Version-style "V8" / "V12": digit pieces are not in the vocab
                // (they map to <unk>), so read the number out — 八/十二 in CJK
                // context, EIGHT/TWELVE in English context.
                run_pieces = (later_has_cjk || seen_cjk_context)
                    ? tokenize_digit_run_as_cjk_integer(run, false)
                    : tokenize_english_number_run(run);
            } else if (previous_alpha_run_had_uppercase && previous_alpha_run_allows_english_number && !later_has_cjk) {
                if ((previous_alpha_run_length < 2 && !previous_alpha_run_after_digit) || run.size() > 3) {
                    throw std::runtime_error("native CJK text-id export cannot encode uppercase alphanumeric digit run without full TextNormalizer: " + run);
                }
                run_pieces = tokenize_english_number_run(run);
            } else if (!decimal_fraction_digits && run.size() == 4 && run[0] != '0' && !followed_by_year_marker) {
                run_pieces = tokenize_plain_four_digit_integer_as_cjk(run);
            } else {
                run_pieces = tokenize_digit_run_as_cjk_integer(run, decimal_fraction_digits);
            }
            pieces.insert(pieces.end(), run_pieces.begin(), run_pieces.end());
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = false;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = true;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = false;
            continue;
        }
        if (is_ascii_alpha(cp)) {
            const bool alpha_run_follows_punctuation = ascii_run_follows_punctuation;
            const bool alpha_run_after_digit = previous_was_digit_run;
            bool alpha_run_from_fullwidth = current_from_fullwidth_alnum;
            std::string run = bytes;
            while (pos < text.size()) {
                const size_t saved = pos;
                uint32_t next_cp = 0;
                std::string next_bytes;
                utf8_next(text, pos, next_cp, next_bytes);
                const bool next_from_fullwidth = normalize_fullwidth_alnum(next_cp, next_bytes);
                if (!is_ascii_alpha(next_cp)) {
                    pos = saved;
                    break;
                }
                alpha_run_from_fullwidth = alpha_run_from_fullwidth || next_from_fullwidth;
                run += next_bytes;
            }
            if (!alpha_run_follows_punctuation && !alpha_run_after_digit && !alpha_run_from_fullwidth &&
                !seen_cjk_context && (run == "No" || run == "NO" || run == "no") && pos < text.size()) {
                size_t dot_pos = pos;
                uint32_t dot_cp = 0;
                std::string dot_bytes;
                utf8_next(text, dot_pos, dot_cp, dot_bytes);
                if (dot_cp == '.') {
                    size_t spaced_digit_pos = dot_pos;
                    bool saw_no_space = false;
                    while (spaced_digit_pos < text.size()) {
                        const size_t saved_space = spaced_digit_pos;
                        uint32_t space_cp = 0;
                        std::string space_bytes;
                        utf8_next(text, spaced_digit_pos, space_cp, space_bytes);
                        if (space_cp != ' ') {
                            spaced_digit_pos = saved_space;
                            break;
                        }
                        saw_no_space = true;
                    }
                    if (saw_no_space) {
                        std::string spaced_number_run;
                        size_t spaced_number_end = spaced_digit_pos;
                        bool saw_fullwidth_spaced_number = false;
                        if (read_ascii_digit_run_at(text,
                                                    spaced_digit_pos,
                                                    5,
                                                    spaced_number_run,
                                                    spaced_number_end,
                                                    &saw_fullwidth_spaced_number)) {
                            bool next_is_digit = false;
                            const bool next_or_suffix_has_cjk = suffix_has_cjk(text, spaced_number_end);
                            if (spaced_number_end < text.size()) {
                                size_t next_pos = spaced_number_end;
                                uint32_t next_cp = 0;
                                std::string next_bytes;
                                utf8_next(text, next_pos, next_cp, next_bytes);
                                normalize_fullwidth_alnum(next_cp, next_bytes);
                                next_is_digit = is_ascii_digit(next_cp);
                            }
                            if (!saw_fullwidth_spaced_number && !next_is_digit) {
                                if (next_or_suffix_has_cjk) {
                                    auto no_pieces = tokenize_ascii_run_prefixed(piece_to_id, "NO");
                                    pieces.insert(pieces.end(), no_pieces.begin(), no_pieces.end());
                                    pieces.push_back(".");
                                    auto digit_pieces = tokenize_digit_run_as_cjk_integer(spaced_number_run, false);
                                    pieces.insert(pieces.end(), digit_pieces.begin(), digit_pieces.end());
                                } else {
                                    const std::string number_piece = space_piece + "NUMBER";
                                    if (piece_to_id.find(number_piece) == piece_to_id.end()) {
                                        throw std::runtime_error("native CJK text-id export cannot encode spaced No. prefix without NUMBER piece");
                                    }
                                    pieces.push_back(number_piece);
                                    if (!(spaced_number_run.size() > 1 && spaced_number_run[0] == '0')) {
                                        if (piece_to_id.find(".") == piece_to_id.end()) {
                                            throw std::runtime_error("native CJK text-id export cannot encode spaced No. prefix without dot piece");
                                        }
                                        pieces.push_back(".");
                                    }
                                    auto number_pieces = tokenize_english_spaced_no_number_run(spaced_number_run);
                                    pieces.insert(pieces.end(), number_pieces.begin(), number_pieces.end());
                                }
                                pos = spaced_number_end;
                                previous_was_ascii_word = false;
                                ascii_run_follows_punctuation = false;
                                previous_alpha_run_had_uppercase = false;
                                previous_alpha_run_length = 0;
                                previous_alpha_run_allows_english_number = false;
                                previous_alpha_run_after_digit = false;
                                previous_was_digit_run = true;
                                decimal_fraction_digits = false;
                                digit_run_allowed_after_punctuation = false;
                                continue;
                            }
                        }
                    }
                    std::string number_run;
                    size_t number_end = dot_pos;
                    bool saw_fullwidth_number = false;
                    if (read_ascii_digit_run_at(text, dot_pos, 5, number_run, number_end, &saw_fullwidth_number)) {
                        bool next_is_digit = false;
                        bool next_or_suffix_has_cjk = suffix_has_cjk(text, number_end);
                        if (number_end < text.size()) {
                            size_t next_pos = number_end;
                            uint32_t next_cp = 0;
                            std::string next_bytes;
                            utf8_next(text, next_pos, next_cp, next_bytes);
                            normalize_fullwidth_alnum(next_cp, next_bytes);
                            next_is_digit = is_ascii_digit(next_cp);
                        }
                        if (!saw_fullwidth_number && !next_is_digit && !next_or_suffix_has_cjk) {
                            if (run == "no") {
                                auto no_pieces = tokenize_ascii_run_prefixed(piece_to_id, "NO");
                                pieces.insert(pieces.end(), no_pieces.begin(), no_pieces.end());
                                pieces.push_back(space_piece + "POINT");
                                for (char digit : number_run) {
                                    if (digit == '0') {
                                        pieces.push_back(space_piece + "OH");
                                    } else {
                                        append_english_under_100(pieces, digit - '0');
                                    }
                                }
                            } else {
                                const std::string number_piece = space_piece + "NUMBER";
                                if (piece_to_id.find(number_piece) == piece_to_id.end()) {
                                    throw std::runtime_error("native CJK text-id export cannot encode No. prefix without NUMBER piece");
                                }
                                pieces.push_back(number_piece);
                                auto number_pieces = tokenize_english_no_number_run(number_run);
                                pieces.insert(pieces.end(), number_pieces.begin(), number_pieces.end());
                            }
                            pos = number_end;
                            previous_was_ascii_word = false;
                            ascii_run_follows_punctuation = false;
                            previous_alpha_run_had_uppercase = false;
                            previous_alpha_run_length = 0;
                            previous_alpha_run_allows_english_number = false;
                            previous_alpha_run_after_digit = false;
                            previous_was_digit_run = true;
                            decimal_fraction_digits = false;
                            digit_run_allowed_after_punctuation = false;
                            continue;
                        }
                    }
                    if (!suffix_has_cjk(text, dot_pos)) {
                        throw std::runtime_error("native CJK text-id export cannot encode this No. number form without full TextNormalizer: " + run);
                    }
                }
            }
            bool consumed_mixed_piece = false;
            if (!alpha_run_follows_punctuation && !alpha_run_from_fullwidth && pos < text.size()) {
                size_t lookahead = pos;
                uint32_t next_cp = 0;
                std::string next_bytes;
                utf8_next(text, lookahead, next_cp, next_bytes);
                normalize_fullwidth_alnum(next_cp, next_bytes);
                if (is_ascii_digit(next_cp)) {
                    const std::string mixed_piece = ascii_upper(run) + next_bytes;
                    if (piece_to_id.find(mixed_piece) != piece_to_id.end()) {
                        pieces.push_back(space_piece);
                        pieces.push_back(mixed_piece);
                        pos = lookahead;
                        consumed_mixed_piece = true;
                    }
                }
            }
            if (consumed_mixed_piece) {
                previous_was_ascii_word = true;
                ascii_run_follows_punctuation = false;
                previous_alpha_run_had_uppercase = false;
                previous_alpha_run_length = 0;
                previous_alpha_run_allows_english_number = false;
                previous_alpha_run_after_digit = false;
                previous_was_digit_run = false;
                decimal_fraction_digits = false;
                digit_run_allowed_after_punctuation = true;
                continue;
            }
            if (alpha_run_after_digit && run.size() == 1 && piece_to_id.find(ascii_upper(run)) != piece_to_id.end()) {
                pieces.push_back(space_piece);
                pieces.push_back(ascii_upper(run));
            } else if (ascii_run_follows_punctuation) {
                auto run_pieces = tokenize_ascii_run_unprefixed(piece_to_id, run);
                pieces.insert(pieces.end(), run_pieces.begin(), run_pieces.end());
            } else {
                if (!ascii_run_is_alpha(run)) {
                    const std::string piece = space_piece + ascii_upper(run);
                    if (piece_to_id.find(piece) == piece_to_id.end()) {
                        throw std::runtime_error("native CJK text-id export cannot encode ASCII run without full SentencePiece: " + run);
                    }
                    pieces.push_back(piece);
                } else {
                    auto run_pieces = tokenize_ascii_run_prefixed(piece_to_id, run);
                    pieces.insert(pieces.end(), run_pieces.begin(), run_pieces.end());
                }
            }
            previous_was_ascii_word = true;
            ascii_run_follows_punctuation = false;
            previous_alpha_run_had_uppercase = ascii_run_has_uppercase(run);
            previous_alpha_run_length = run.size();
            previous_alpha_run_upper = ascii_upper(run);
            previous_alpha_run_allows_english_number = !alpha_run_follows_punctuation && !alpha_run_from_fullwidth;
            previous_alpha_run_after_digit = alpha_run_after_digit;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = false;
            continue;
        }
        if (is_ascii_punctuation(cp)) {
            if (cp == '.' && pos + 1 < text.size() && text[pos] == '.' && text[pos + 1] == '.') {
                pos += 2;
                append_normalized_ascii_punctuation(pieces, piece_to_id, "...", previous_was_ascii_word);
                previous_was_ascii_word = false;
                ascii_run_follows_punctuation = true;
                previous_alpha_run_had_uppercase = false;
                previous_alpha_run_length = 0;
                previous_alpha_run_allows_english_number = false;
                previous_alpha_run_after_digit = false;
                previous_was_digit_run = false;
                decimal_fraction_digits = false;
                digit_run_allowed_after_punctuation = true;
                continue;
            }
            if (cp == '+' && !previous_was_ascii_word) {
                if (previous_was_digit_run && pos < text.size()) {
                    size_t next_plus_pos = pos;
                    uint32_t next_plus_cp = 0;
                    std::string next_plus_bytes;
                    utf8_next(text, next_plus_pos, next_plus_cp, next_plus_bytes);
                    if (next_plus_cp == '+') {
                        const auto next_plus_probe = probe_signed_cjk_number_suffix(text, next_plus_pos, seen_cjk_context);
                        if (next_plus_probe.supports_cjk_signed_number) {
                            append_cjk_piece(pieces, "\xe5\x8a\xa0");
                            previous_was_ascii_word = false;
                            ascii_run_follows_punctuation = false;
                            previous_alpha_run_had_uppercase = false;
                            previous_alpha_run_length = 0;
                            previous_alpha_run_allows_english_number = false;
                            previous_alpha_run_after_digit = false;
                            previous_was_digit_run = false;
                            decimal_fraction_digits = false;
                            digit_run_allowed_after_punctuation = false;
                            signed_cjk_number_context = false;
                            seen_cjk_context = true;
                            continue;
                        }
                    }
                }
                const auto plus_probe = probe_signed_cjk_number_suffix(text, pos, seen_cjk_context);
                if (plus_probe.has_digit && previous_was_digit_run) {
                    if (plus_probe.has_temperature_unit && !plus_probe.supports_cjk_signed_number) {
                        throw std::runtime_error("native CJK text-id export cannot encode ASCII plus English temperature without full TextNormalizer");
                    }
                    const bool plus_degree_rhs = plus_probe.has_temperature_unit ||
                                                 native_operator_rhs_degree_number_context(text, pos);
                    append_cjk_piece(pieces, plus_degree_rhs ? "\xe6\xad\xa3" : "\xe5\x8a\xa0");
                    previous_was_ascii_word = false;
                    ascii_run_follows_punctuation = false;
                    previous_alpha_run_had_uppercase = false;
                    previous_alpha_run_length = 0;
                    previous_alpha_run_allows_english_number = false;
                    previous_alpha_run_after_digit = false;
                    previous_was_digit_run = false;
                    decimal_fraction_digits = false;
                    digit_run_allowed_after_punctuation = true;
                    signed_cjk_number_context = plus_degree_rhs;
                    seen_cjk_context = true;
                    continue;
                }
                if (!previous_was_digit_run && plus_probe.supports_cjk_signed_number) {
                    append_cjk_piece(pieces, "\xe6\xad\xa3");
                    previous_was_ascii_word = false;
                    ascii_run_follows_punctuation = false;
                    previous_alpha_run_had_uppercase = false;
                    previous_alpha_run_length = 0;
                    previous_alpha_run_allows_english_number = false;
                    previous_alpha_run_after_digit = false;
                    previous_was_digit_run = false;
                    decimal_fraction_digits = false;
                    digit_run_allowed_after_punctuation = true;
                    signed_cjk_number_context = plus_probe.has_temperature_unit ||
                                                native_operator_rhs_degree_number_context(text, pos);
                    seen_cjk_context = true;
                    continue;
                }
            }
            if (cp == '-' && !previous_was_digit_run && !previous_was_ascii_word) {
                const auto minus_probe = probe_signed_cjk_number_suffix(text, pos, seen_cjk_context);
                if (minus_probe.supports_cjk_signed_number) {
                    append_cjk_piece(pieces, "\xe8\xb4\x9f");
                    previous_was_ascii_word = false;
                    ascii_run_follows_punctuation = false;
                    previous_alpha_run_had_uppercase = false;
                    previous_alpha_run_length = 0;
                    previous_alpha_run_allows_english_number = false;
                    previous_alpha_run_after_digit = false;
                    previous_was_digit_run = false;
                    decimal_fraction_digits = false;
                    digit_run_allowed_after_punctuation = false;
                    signed_cjk_number_context = minus_probe.has_temperature_unit ||
                                                native_operator_rhs_degree_number_context(text, pos);
                    seen_cjk_context = true;
                    continue;
                }
            }
            if ((cp == '=' || cp == '<' || cp == '>') && previous_was_digit_run) {
                bool handled_operator = false;
                if ((cp == '<' || cp == '>') && pos < text.size()) {
                    size_t after_equal = pos;
                    uint32_t maybe_equal_cp = 0;
                    std::string maybe_equal_bytes;
                    utf8_next(text, after_equal, maybe_equal_cp, maybe_equal_bytes);
                    if (maybe_equal_cp == '=' && native_operator_rhs_digit_context(text, after_equal, true)) {
                        operator_rhs_degree_number_context = native_operator_rhs_degree_number_context(text, after_equal);
                        operator_rhs_measure_number_context = native_cjk_measure_number_context(text, after_equal);
                        if (cp == '<') {
                            append_cjk_less_than_pieces(pieces);
                        } else {
                            append_cjk_greater_than_pieces(pieces);
                        }
                        append_cjk_equal_pieces(pieces);
                        pos = after_equal;
                        handled_operator = true;
                    }
                }
                if (!handled_operator && native_operator_rhs_digit_context(text, pos, cp == '=')) {
                    operator_rhs_degree_number_context = cp == '=' && native_operator_rhs_degree_number_context(text, pos);
                    const bool rhs_measure = native_cjk_measure_number_context(text, pos);
                    if (rhs_measure) {
                        if (cp == '=' || native_digit_run_length_at(text, pos) > 1) {
                            operator_rhs_measure_number_context = true;
                        } else if (cp == '<' || cp == '>') {
                            suppress_measure_number_context = true;
                        }
                    }
                    if (cp == '=') {
                        append_cjk_equal_pieces(pieces);
                    } else if (cp == '<') {
                        append_cjk_less_than_pieces(pieces);
                    } else {
                        append_cjk_greater_than_pieces(pieces);
                    }
                    handled_operator = true;
                }
                if (handled_operator) {
                    previous_was_ascii_word = false;
                    ascii_run_follows_punctuation = false;
                    previous_alpha_run_had_uppercase = false;
                    previous_alpha_run_length = 0;
                    previous_alpha_run_allows_english_number = false;
                    previous_alpha_run_after_digit = false;
                    previous_was_digit_run = false;
                    decimal_fraction_digits = false;
                    digit_run_allowed_after_punctuation = true;
                    seen_cjk_context = true;
                    continue;
                }
            }
            if (cp == ':' && previous_alpha_run_had_uppercase && previous_alpha_run_allows_english_number) {
                std::string alpha_colon_digits;
                size_t alpha_colon_end = pos;
                if (!read_ascii_digit_run_at(text, pos, 4, alpha_colon_digits, alpha_colon_end)) {
                    if (seen_cjk_context || suffix_has_cjk(text, pos)) {
                        append_normalized_ascii_punctuation(pieces, piece_to_id, ",", previous_was_ascii_word);
                        previous_was_ascii_word = false;
                        ascii_run_follows_punctuation = true;
                        previous_alpha_run_had_uppercase = false;
                        previous_alpha_run_length = 0;
                        previous_alpha_run_allows_english_number = false;
                        previous_alpha_run_after_digit = false;
                        previous_was_digit_run = false;
                        decimal_fraction_digits = false;
                        digit_run_allowed_after_punctuation = false;
                        continue;
                    }
                    throw std::runtime_error("native CJK text-id export cannot encode alpha-colon form without trailing digits");
                }
                if (alpha_colon_end < text.size()) {
                    size_t next_pos = alpha_colon_end;
                    uint32_t next_cp = 0;
                    std::string next_bytes;
                    utf8_next(text, next_pos, next_cp, next_bytes);
                    normalize_fullwidth_alnum(next_cp, next_bytes);
                    if (is_ascii_digit(next_cp)) {
                        throw std::runtime_error("native CJK text-id export cannot encode alpha-colon digit run longer than 4 digits without full TextNormalizer");
                    }
                }
                const bool cjk_number_context = seen_cjk_context || suffix_has_cjk(text, alpha_colon_end);
                append_alpha_colon_number_pieces(pieces, piece_to_id, alpha_colon_digits, cjk_number_context);
                pos = alpha_colon_end;
                previous_was_ascii_word = false;
                ascii_run_follows_punctuation = false;
                previous_alpha_run_had_uppercase = false;
                previous_alpha_run_length = 0;
                previous_alpha_run_allows_english_number = false;
                previous_alpha_run_after_digit = false;
                previous_was_digit_run = true;
                decimal_fraction_digits = false;
                digit_run_allowed_after_punctuation = false;
                continue;
            }
            if (cp == ':' && previous_was_digit_run) {
                const size_t saved = pos;
                uint32_t next_cp = 0;
                std::string next_bytes;
                if (utf8_next(text, pos, next_cp, next_bytes)) {
                    normalize_fullwidth_alnum(next_cp, next_bytes);
                    if (is_ascii_digit(next_cp)) {
                        pos = saved;
                        append_cjk_piece(pieces, "\xe6\xaf\x94");
                        previous_was_ascii_word = false;
                        ascii_run_follows_punctuation = false;
                        previous_alpha_run_had_uppercase = false;
                        previous_alpha_run_length = 0;
                        previous_alpha_run_allows_english_number = false;
                        previous_alpha_run_after_digit = false;
                        previous_was_digit_run = false;
                        decimal_fraction_digits = false;
                        digit_run_allowed_after_punctuation = false;
                        continue;
                    }
                }
                pos = saved;
            }
            if (cp == '.' && previous_was_digit_run) {
                const size_t saved = pos;
                uint32_t next_cp = 0;
                std::string next_bytes;
                if (utf8_next(text, pos, next_cp, next_bytes) && is_ascii_digit(next_cp)) {
                    pos = saved;
                    pieces.push_back(space_piece);
                    pieces.push_back("\xe7\x82\xb9");
                    previous_was_ascii_word = false;
                    ascii_run_follows_punctuation = false;
                    previous_alpha_run_had_uppercase = false;
                    previous_alpha_run_length = 0;
                    previous_alpha_run_allows_english_number = false;
                    previous_alpha_run_after_digit = false;
                    previous_was_digit_run = false;
                    decimal_fraction_digits = true;
                    digit_run_allowed_after_punctuation = false;
                    continue;
                }
                pos = saved;
            }
            if ((cp == ':' || cp == ';') && (seen_cjk_context || suffix_has_cjk(text, pos))) {
                append_normalized_ascii_punctuation(pieces, piece_to_id, ",", previous_was_ascii_word);
                previous_was_ascii_word = false;
                ascii_run_follows_punctuation = true;
                previous_alpha_run_had_uppercase = false;
                previous_alpha_run_length = 0;
                previous_alpha_run_allows_english_number = false;
                previous_alpha_run_after_digit = false;
                previous_was_digit_run = false;
                decimal_fraction_digits = false;
                digit_run_allowed_after_punctuation = false;
                continue;
            }
            const std::string normalized = cp == '~' ? "-" : bytes;
            append_normalized_ascii_punctuation(pieces, piece_to_id, normalized, previous_was_ascii_word);
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = true;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = cp == '.' || normalized == "-";
            continue;
        }
        if ((cp == 0x2264 || cp == 0x2265 || cp == 0x00d7) && previous_was_digit_run &&
            native_operator_rhs_digit_context(text, pos, true)) {
            operator_rhs_degree_number_context = native_operator_rhs_degree_number_context(text, pos);
            operator_rhs_measure_number_context = native_cjk_measure_number_context(text, pos);
            if (cp == 0x2264) {
                append_cjk_less_than_pieces(pieces);
                append_cjk_equal_pieces(pieces);
            } else if (cp == 0x2265) {
                append_cjk_greater_than_pieces(pieces);
                append_cjk_equal_pieces(pieces);
            } else {
                append_cjk_piece(pieces, "\xe4\xb9\x98");
            }
            previous_was_ascii_word = false;
            ascii_run_follows_punctuation = false;
            previous_alpha_run_had_uppercase = false;
            previous_alpha_run_length = 0;
            previous_alpha_run_allows_english_number = false;
            previous_alpha_run_after_digit = false;
            previous_was_digit_run = false;
            decimal_fraction_digits = false;
            digit_run_allowed_after_punctuation = true;
            seen_cjk_context = true;
            continue;
        }
        throw std::runtime_error("native CJK text-id export currently supports CJK, whitespace, and limited ASCII pieces only");
    }
    if (pieces.empty()) {
        throw std::runtime_error("tokenizer input produced no native text pieces");
    }

    std::vector<uint32_t> ids;
    ids.reserve(pieces.size());
    for (const auto& piece : pieces) {
        auto it = piece_to_id.find(piece);
        if (it == piece_to_id.end()) {
            if (piece == "%" || piece == "/" || piece == "\xe2\x80\x93" || piece == "\xc2\xa5") {
                auto unk = piece_to_id.find("<unk>");
                if (unk != piece_to_id.end()) {
                    ids.push_back(unk->second.id);
                    continue;
                }
            }
            if (piece.size() == 1 && std::isdigit(static_cast<unsigned char>(piece[0])) != 0) {
                auto unk = piece_to_id.find("<unk>");
                if (unk != piece_to_id.end()) {
                    ids.push_back(unk->second.id);
                    continue;
                }
            }
            throw std::runtime_error("tokenizer pieces file does not contain piece: " + piece);
        }
        ids.push_back(it->second.id);
    }
    return {std::move(pieces), std::move(ids)};
}

void print_json_string_array(const std::vector<std::string>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << "\"" << json_escape(values[i]) << "\"";
    }
    std::cout << "]";
}

void print_json_u32_array(const std::vector<uint32_t>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    std::cout << "]";
}
