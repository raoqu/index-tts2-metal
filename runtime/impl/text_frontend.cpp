// Implementation of mit2::TextFrontend — see runtime/include/mit2/text_frontend.hpp.
//
// Faithful native port of the reference Python text frontend:
//   - indextts/utils/front.py     (TextNormalizer, TextTokenizer, tokenize_by_CJK_char)
//   - wetext/utils.py             (preprocess/should_normalize/tag/reorder/verbalize/postprocess)
//   - wetext/token_parser.py      (Token, TokenParser, order tables)
// using the SAME assets the reference uses (bpe.model + wetext .fst grammars),
// so output token ids are byte-identical to the reference for arbitrary text.
//
// Third-party headers (kaldifst/OpenFST, sentencepiece) are confined to this
// translation unit; the public header exposes only std types.

#include "mit2/text_frontend.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "kaldifst/csrc/text-normalizer.h"
#include <sentencepiece_processor.h>

namespace mit2 {
namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Decode one codepoint at byte offset i. Returns number of bytes consumed (>=1)
// and writes the codepoint to cp. Invalid bytes are passed through as 1 byte.
size_t utf8_decode(const std::string& s, size_t i, uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        cp = c;
        return 1;
    }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) {
        cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        return 2;
    }
    if ((c >> 4) == 0xE && i + 2 < s.size()) {
        cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        return 3;
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size()) {
        cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
             ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        return 4;
    }
    cp = c;
    return 1;
}

std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

size_t utf8_count(const std::string& s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        i += utf8_decode(s, i, cp);
        ++n;
    }
    return n;
}

// Python str.strip(): trims ASCII + common Unicode whitespace. The reference
// only ever sees ASCII/Chinese, so ASCII whitespace trimming matches.
std::string py_strip(const std::string& s) {
    size_t b = 0, e = s.size();
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    while (b < e && is_ws(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && is_ws(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string py_rstrip(const std::string& s) {
    size_t e = s.size();
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    while (e > 0 && is_ws(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(0, e);
}

// ASCII + fullwidth-latin uppercasing (matches Python str.upper() on the
// scripts the reference encounters).
std::string py_upper(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        size_t n = utf8_decode(s, i, cp);
        if (cp >= 'a' && cp <= 'z') {
            out.push_back(static_cast<char>(cp - 32));
        } else if (cp >= 0xFF41 && cp <= 0xFF5A) {  // fullwidth a-z
            out += utf8_encode(cp - 0x20);
        } else {
            out.append(s, i, n);
        }
        i += n;
    }
    return out;
}

bool is_cjk_basic(uint32_t cp) {  // front.py [一-鿿]
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

// tokenize_by_CJK_char CJK_RANGE_PATTERN ranges.
bool is_cjk_split(uint32_t cp) {
    return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0xA4CF) ||
           (cp >= 0xA840 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF65 && cp <= 0xFFDC) ||
           (cp >= 0x20000 && cp <= 0x2FFFF);
}

bool has_ascii_digit_or_fullwidth(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        i += utf8_decode(s, i, cp);
        if (cp >= '0' && cp <= '9') return true;
        if (cp >= 0xFF10 && cp <= 0xFF19) return true;  // fullwidth digits
    }
    return false;
}

// ---------------------------------------------------------------------------
// Punctuation replacement maps (front.py char_rep_map / zh_char_rep_map)
// ---------------------------------------------------------------------------

using ReplPair = std::pair<std::string, std::string>;

const std::vector<ReplPair>& char_rep_map() {
    static const std::vector<ReplPair> m = {
        {"\xEF\xBC\x9A", ","},  // ：
        {"\xEF\xBC\x9B", ","},  // ；
        {";", ","},
        {"\xEF\xBC\x8C", ","},  // ，
        {"\xE3\x80\x82", "."},  // 。
        {"\xEF\xBC\x81", "!"},  // ！
        {"\xEF\xBC\x9F", "?"},  // ？
        {"\n", " "},
        {"\xC2\xB7", "-"},      // ·
        {"\xE3\x80\x81", ","},  // 、
        {"...", "\xE2\x80\xA6"},                  // ... -> …
        {",,,", "\xE2\x80\xA6"},                  // ,,, -> …
        {"\xEF\xBC\x8C\xEF\xBC\x8C\xEF\xBC\x8C", "\xE2\x80\xA6"},  // ，，， -> …
        {"\xE2\x80\xA6\xE2\x80\xA6", "\xE2\x80\xA6"},              // …… -> …
        {"\xE2\x80\x9C", "'"},  // “
        {"\xE2\x80\x9D", "'"},  // ”
        {"\"", "'"},
        {"\xE2\x80\x98", "'"},  // ‘
        {"\xE2\x80\x99", "'"},  // ’
        {"\xEF\xBC\x88", "'"},  // （
        {"\xEF\xBC\x89", "'"},  // ）
        {"(", "'"},
        {")", "'"},
        {"\xE3\x80\x8A", "'"},  // 《
        {"\xE3\x80\x8B", "'"},  // 》
        {"\xE3\x80\x90", "'"},  // 【
        {"\xE3\x80\x91", "'"},  // 】
        {"[", "'"},
        {"]", "'"},
        {"\xE2\x80\x94", "-"},  // —
        {"\xEF\xBD\x9E", "-"},  // ～
        {"~", "-"},
        {"\xE3\x80\x8C", "'"},  // 「
        {"\xE3\x80\x8D", "'"},  // 」
        {":", ","},
    };
    return m;
}

const std::vector<ReplPair>& zh_char_rep_map() {
    static const std::vector<ReplPair> m = [] {
        std::vector<ReplPair> v;
        v.push_back({"$", "."});
        const auto& base = char_rep_map();
        v.insert(v.end(), base.begin(), base.end());
        return v;
    }();
    return m;
}

// Emulate re.compile("|".join(escape(k) for k in keys)).sub(repl): leftmost
// scan, first matching alternative (in map order) wins at each position.
std::string apply_rep_map(const std::string& text, const std::vector<ReplPair>& m) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        bool matched = false;
        for (const auto& kv : m) {
            const std::string& key = kv.first;
            if (!key.empty() && text.compare(i, key.size(), key) == 0) {
                out += kv.second;
                i += key.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            out.push_back(text[i]);
            ++i;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Regex-based helpers (contraction / email / pinyin / names)
// ---------------------------------------------------------------------------

std::string apply_contraction(const std::string& text) {
    // (what|where|who|which|how|t?here|it|s?he|that|this)'s  -> \1 is  (icase)
    static const std::regex re(
        R"((what|where|who|which|how|t?here|it|s?he|that|this)'s)",
        std::regex::icase | std::regex::ECMAScript);
    return std::regex_replace(text, re, "$1 is");
}

bool is_email(const std::string& s) {
    static const std::regex re(R"(^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$)");
    return std::regex_match(s, re);
}

bool has_ascii_alpha(const std::string& s) {
    for (unsigned char c : s) {
        if (std::isalpha(c)) return true;
    }
    return false;
}

bool has_cjk_basic(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        i += utf8_decode(s, i, cp);
        if (is_cjk_basic(cp)) return true;
    }
    return false;
}

// PINYIN_TONE_PATTERN without the (?<![a-z]) lookbehind; the lookbehind is
// enforced by checking the preceding byte is not an ASCII letter.
const std::regex& pinyin_syllable_re() {
    static const std::regex re(
        R"(((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiou\xC3\xBCv]|[ae]i|u[aio]|ao|ou|i[aue]|[u\xC3\xBCv]e|[uv\xC3\xBC]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5]))",
        std::regex::icase | std::regex::ECMAScript);
    return re;
}

// Find non-overlapping pinyin-with-tone matches honoring the lookbehind.
std::vector<std::string> find_pinyin(const std::string& text) {
    std::vector<std::string> out;
    auto begin = std::sregex_iterator(text.begin(), text.end(), pinyin_syllable_re());
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const auto& m = *it;
        size_t pos = static_cast<size_t>(m.position(0));
        if (pos > 0) {
            unsigned char prev = static_cast<unsigned char>(text[pos - 1]);
            if (std::isalpha(prev)) continue;  // (?<![a-z]) with icase excludes A-Z too
        }
        out.push_back(m.str(0));
    }
    return out;
}

std::string correct_pinyin(const std::string& pinyin) {
    if (pinyin.empty()) return pinyin;
    char c0 = pinyin[0];
    if (c0 != 'j' && c0 != 'q' && c0 != 'x' && c0 != 'J' && c0 != 'Q' && c0 != 'X') {
        return pinyin;  // not uppercased (matches reference early return)
    }
    // ([jqx])[uü](n|e|an)*(\d) -> \1 v \2 \3   (icase)
    static const std::regex re(R"(([jqx])[u\xC3\xBC]((?:n|e|an)*)(\d))",
                               std::regex::icase | std::regex::ECMAScript);
    std::string fixed = std::regex_replace(pinyin, re, "$1v$2$3");
    return py_upper(fixed);
}

// Replace all non-overlapping occurrences of `needle` in `hay` with `repl`.
std::string replace_all(std::string hay, const std::string& needle, const std::string& repl) {
    if (needle.empty()) return hay;
    size_t pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        hay.replace(pos, needle.size(), repl);
        pos += repl.size();
    }
    return hay;
}

std::string placeholder(const char* prefix, size_t i) {
    std::string p = "<";
    p += prefix;
    p += static_cast<char>('a' + static_cast<int>(i));
    p += ">";
    return p;
}

// dedup preserving first-seen order
std::vector<std::string> unique_keep_order(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    for (const auto& s : in) {
        if (std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
    }
    return out;
}

// NAME_PATTERN: [一-鿿]+([-·—][一-鿿]+){1,2}
std::vector<std::string> find_names(const std::string& text) {
    std::vector<std::string> names;
    // decode to codepoints with byte spans
    std::vector<uint32_t> cps;
    std::vector<size_t> starts;
    std::vector<size_t> lens;
    for (size_t i = 0; i < text.size();) {
        uint32_t cp = 0;
        size_t n = utf8_decode(text, i, cp);
        cps.push_back(cp);
        starts.push_back(i);
        lens.push_back(n);
        i += n;
    }
    auto is_sep = [](uint32_t cp) { return cp == '-' || cp == 0x00B7 || cp == 0x2014; };
    size_t k = 0;
    while (k < cps.size()) {
        if (!is_cjk_basic(cps[k])) {
            ++k;
            continue;
        }
        size_t match_begin = k;
        // first CJK run
        while (k < cps.size() && is_cjk_basic(cps[k])) ++k;
        size_t groups = 0;
        size_t match_end = k;  // exclusive index into cps
        while (groups < 2 && k < cps.size() && is_sep(cps[k]) && k + 1 < cps.size() &&
               is_cjk_basic(cps[k + 1])) {
            ++k;  // consume sep
            while (k < cps.size() && is_cjk_basic(cps[k])) ++k;  // consume CJK run
            ++groups;
            match_end = k;
        }
        if (groups >= 1) {
            size_t bstart = starts[match_begin];
            size_t bend = (match_end < cps.size()) ? starts[match_end]
                                                   : (starts.back() + lens.back());
            names.push_back(text.substr(bstart, bend - bstart));
        }
        // if no group formed, k already advanced past the CJK run; continue
    }
    return names;
}

// ---------------------------------------------------------------------------
// tokenize_by_CJK_char (common.py)
// ---------------------------------------------------------------------------
std::string tokenize_by_cjk_char(const std::string& line_in) {
    std::string line = py_strip(line_in);
    std::vector<std::string> tokens;
    std::string buf;
    auto flush = [&] {
        std::string t = py_strip(buf);
        if (!t.empty()) tokens.push_back(py_upper(t));
        buf.clear();
    };
    for (size_t i = 0; i < line.size();) {
        uint32_t cp = 0;
        size_t n = utf8_decode(line, i, cp);
        if (is_cjk_split(cp)) {
            flush();
            tokens.push_back(py_upper(line.substr(i, n)));  // single CJK char
        } else {
            buf.append(line, i, n);
        }
        i += n;
    }
    flush();
    std::string out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) out.push_back(' ');
        out += tokens[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// TokenParser (wetext/token_parser.py)
// ---------------------------------------------------------------------------
const char* kEOS = "<EOS>";

using OrderMap = std::map<std::string, std::vector<std::string>>;

const OrderMap& tn_orders() {
    static const OrderMap m = {
        {"date", {"year", "month", "day"}},
        {"fraction", {"denominator", "numerator"}},
        {"measure", {"denominator", "numerator", "value"}},
        {"money", {"value", "currency"}},
        {"time", {"noon", "hour", "minute", "second"}},
    };
    return m;
}

const OrderMap& en_tn_orders() {
    static const OrderMap m = {
        {"date", {"preserve_order", "text", "day", "month", "year"}},
        {"money", {"integer_part", "fractional_part", "quantity", "currency_maj"}},
    };
    return m;
}

struct TokenItem {
    std::string name;
    std::vector<std::string> order;
    std::map<std::string, std::string> members;
    void append(const std::string& k, const std::string& v) {
        order.push_back(k);
        members[k] = v;
    }
    std::string string(const OrderMap& orders) const {
        std::string output = name + " {";
        std::vector<std::string> use_order = order;
        auto it = orders.find(name);
        if (it != orders.end()) {
            auto po = members.find("preserve_order");
            if (po == members.end() || po->second != "true") {
                use_order = it->second;
            }
        }
        for (const auto& key : use_order) {
            auto mit = members.find(key);
            if (mit == members.end()) continue;
            output += " " + key + ": \"" + mit->second + "\"";
        }
        return output + " }";
    }
};

class TokenParser {
public:
    explicit TokenParser(const OrderMap& orders) : orders_(orders) {}

    std::string reorder(const std::string& input) {
        parse(input);
        std::string output;
        for (const auto& tok : tokens_) {
            output += tok.string(orders_) + " ";
        }
        return py_strip(output);
    }

private:
    const OrderMap& orders_;
    std::string text_;
    size_t index_ = 0;
    std::string ch_;  // current "char": a single byte, or kEOS
    std::vector<TokenItem> tokens_;

    void load(const std::string& input) {
        index_ = 0;
        text_ = input;
        ch_ = std::string(1, input[0]);
        tokens_.clear();
    }
    bool read() {
        if (index_ < text_.size() - 1) {
            ++index_;
            ch_ = std::string(1, text_[index_]);
            return true;
        }
        ch_ = kEOS;
        return false;
    }
    bool is_eos() const { return ch_ == kEOS; }
    bool parse_ws() {
        bool not_eos = !is_eos();
        while (not_eos && ch_ == " ") not_eos = read();
        return not_eos;
    }
    bool parse_char(const std::string& exp) {
        if (ch_ == exp) {
            read();
            return true;
        }
        return false;
    }
    void parse_chars(const std::string& exp) {
        for (char x : exp) parse_char(std::string(1, x));
    }
    std::string parse_key() {
        std::string key;
        while (!is_eos() && ch_.size() == 1 &&
               (std::isalpha(static_cast<unsigned char>(ch_[0])) || ch_[0] == '_')) {
            key += ch_;
            read();
        }
        return key;
    }
    std::string parse_value() {
        std::string value;
        bool escape = false;
        while (ch_ != "\"") {
            if (is_eos()) break;  // safety
            value += ch_;
            escape = (ch_ == "\\");
            read();
            if (escape) {
                escape = false;
                value += ch_;
                read();
            }
        }
        return value;
    }
    void parse(const std::string& input) {
        load(input);
        while (parse_ws()) {
            std::string name = parse_key();
            parse_chars(" { ");
            TokenItem token;
            token.name = name;
            while (parse_ws()) {
                if (ch_ == "}") {
                    parse_char("}");
                    break;
                }
                std::string key = parse_key();
                parse_chars(": \"");
                std::string value = parse_value();
                parse_char("\"");
                token.append(key, value);
            }
            tokens_.push_back(std::move(token));
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// TextFrontend::Impl
// ---------------------------------------------------------------------------
struct TextFrontend::Impl {
    sentencepiece::SentencePieceProcessor sp;
    std::unique_ptr<kaldifst::TextNormalizer> zh_tagger;
    std::unique_ptr<kaldifst::TextNormalizer> zh_verbalizer;
    std::unique_ptr<kaldifst::TextNormalizer> en_tagger;
    std::unique_ptr<kaldifst::TextNormalizer> en_verbalizer;

    explicit Impl(const std::string& dir) {
        const std::string bpe = dir + "/bpe.model";
        auto st = sp.Load(bpe);
        if (!st.ok()) {
            throw std::runtime_error("text frontend: failed to load " + bpe + ": " + st.ToString());
        }
        const std::string f = dir + "/fsts";
        zh_tagger = std::make_unique<kaldifst::TextNormalizer>(f + "/zh/tn/tagger.fst");
        zh_verbalizer = std::make_unique<kaldifst::TextNormalizer>(f + "/zh/tn/verbalizer.fst");
        en_tagger = std::make_unique<kaldifst::TextNormalizer>(f + "/en/tn/tagger.fst");
        en_verbalizer = std::make_unique<kaldifst::TextNormalizer>(f + "/en/tn/verbalizer.fst");
    }

    // wetext normalize() for a fixed language (front.py uses fixed lang per path).
    std::string wetext_normalize(const std::string& text_in, bool zh) const {
        std::string text = py_strip(text_in);  // preprocess (traditional_to_simple disabled)
        if (has_ascii_digit_or_fullwidth(text)) {  // should_normalize (tn): requires a digit
            kaldifst::TextNormalizer* tagger = zh ? zh_tagger.get() : en_tagger.get();
            kaldifst::TextNormalizer* verb = zh ? zh_verbalizer.get() : en_verbalizer.get();
            std::string tagged = py_strip(tagger->Normalize(text));
            TokenParser parser(zh ? tn_orders() : en_tn_orders());
            std::string reordered = parser.reorder(tagged);
            text = py_strip(verb->Normalize(reordered));
        }
        return py_strip(text);  // postprocess (all options disabled)
    }

    bool use_chinese(const std::string& s) const {
        if (has_cjk_basic(s) || !has_ascii_alpha(s) || is_email(s)) return true;
        return !find_pinyin(s).empty();
    }

    std::string normalize(const std::string& text_in) const {
        std::string result;
        if (use_chinese(text_in)) {
            std::string text = apply_contraction(text_in);
            // save_pinyin_tones(text.rstrip())
            std::string replaced = py_rstrip(text);
            std::vector<std::string> pinyin_list = unique_keep_order(find_pinyin(replaced));
            for (size_t i = 0; i < pinyin_list.size(); ++i) {
                replaced = replace_all(replaced, pinyin_list[i], placeholder("pinyin_", i));
            }
            // save_names
            std::vector<std::string> name_list = unique_keep_order(find_names(replaced));
            for (size_t i = 0; i < name_list.size(); ++i) {
                replaced = replace_all(replaced, name_list[i], placeholder("n_", i));
            }
            // front.py wraps the normalizer in try/except (result="" on failure)
            // so a grammar/FST edge case never aborts synthesis.
            try {
                result = wetext_normalize(replaced, /*zh=*/true);
            } catch (...) {
                result = "";
            }
            // restore_names
            for (size_t i = 0; i < name_list.size(); ++i) {
                result = replace_all(result, placeholder("n_", i), name_list[i]);
            }
            // restore_pinyin_tones (with correct_pinyin)
            for (size_t i = 0; i < pinyin_list.size(); ++i) {
                result = replace_all(result, placeholder("pinyin_", i), correct_pinyin(pinyin_list[i]));
            }
            result = apply_rep_map(result, zh_char_rep_map());
        } else {
            std::string text = apply_contraction(text_in);
            try {
                result = wetext_normalize(text, /*zh=*/false);  // en: result=text on failure
            } catch (...) {
                result = text;
            }
            result = apply_rep_map(result, char_rep_map());
        }
        return result;
    }

    FrontendTokenized tokenize(const std::string& text) const {
        FrontendTokenized out;
        if (text.empty()) return out;
        std::string encode_input;
        if (utf8_count(py_strip(text)) == 1) {
            encode_input = text;  // len(text.strip())==1 -> encode raw
        } else {
            std::string normalized = normalize(text);
            encode_input = tokenize_by_cjk_char(normalized);
        }
        sp.Encode(encode_input, &out.pieces);
        out.ids.reserve(out.pieces.size());
        for (const auto& piece : out.pieces) {
            out.ids.push_back(static_cast<uint32_t>(sp.PieceToId(piece)));
        }
        return out;
    }
};

TextFrontend::TextFrontend(const std::string& tokenizer_dir)
    : impl_(std::make_unique<Impl>(tokenizer_dir)) {}
TextFrontend::~TextFrontend() = default;
TextFrontend::TextFrontend(TextFrontend&&) noexcept = default;
TextFrontend& TextFrontend::operator=(TextFrontend&&) noexcept = default;

FrontendTokenized TextFrontend::tokenize(const std::string& text) const {
    return impl_->tokenize(text);
}
int TextFrontend::unk_id() const { return impl_->sp.unk_id(); }
std::string TextFrontend::normalize(const std::string& text) const {
    return impl_->normalize(text);
}

const TextFrontend& get_cached_text_frontend(const std::string& tokenizer_dir) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::unique_ptr<TextFrontend>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(tokenizer_dir);
    if (it == cache.end()) {
        it = cache.emplace(tokenizer_dir, std::make_unique<TextFrontend>(tokenizer_dir)).first;
    }
    return *it->second;
}

}  // namespace mit2
