// Full native text frontend for IndexTTS2.
//
// Faithful reimplementation of the reference Python tokenizer
// (indextts/utils/front.py: TextNormalizer + TextTokenizer) so the native
// runtime can tokenize ARBITRARY input text byte-identically to the reference,
// instead of the narrow hand-rolled `tokenize_cjk_text` state machine that
// throws on anything outside a curated fixture set.
//
// Pipeline (== front.py TextTokenizer.encode):
//   1. TextNormalizer.normalize  -> wetext TN over OpenFST grammars
//      (numbers/dates/units/currency -> spoken Chinese/English), plus pinyin /
//      name protection and punctuation normalization.
//   2. tokenize_by_CJK_char      -> split each CJK char, uppercase ASCII runs.
//   3. SentencePiece.Encode      -> BPE pieces over bpe.model; ids via PieceToId.
//
// Heavy third-party headers (OpenFST / kaldifst / sentencepiece) are confined
// to text_frontend.cpp; this header exposes only std types so the rest of the
// runtime (compiled with -Wpedantic) need not see them.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mit2 {

struct FrontendTokenized {
    std::vector<std::string> pieces;  // SentencePiece pieces (str), == reference tokenize()
    std::vector<uint32_t> ids;        // PieceToId(piece) for each piece
};

// Loads bpe.model + the wetext FST grammars from <tokenizer_dir> once and
// tokenizes text. Construction is relatively heavy (~8MB FSTs + bpe model);
// reuse a single instance. Thread-compatible: tokenize() is const and does not
// mutate shared state.
class TextFrontend {
public:
    // tokenizer_dir must contain `bpe.model` and `fsts/{zh,en}/tn/{tagger,verbalizer}.fst`.
    explicit TextFrontend(const std::string& tokenizer_dir);
    ~TextFrontend();
    TextFrontend(TextFrontend&&) noexcept;
    TextFrontend& operator=(TextFrontend&&) noexcept;
    TextFrontend(const TextFrontend&) = delete;
    TextFrontend& operator=(const TextFrontend&) = delete;

    FrontendTokenized tokenize(const std::string& text) const;
    int unk_id() const;

    // Normalized text (after TextNormalizer.normalize) — exposed for parity tests.
    std::string normalize(const std::string& text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Process-wide cache keyed by tokenizer_dir so the FSTs/bpe model load once.
const TextFrontend& get_cached_text_frontend(const std::string& tokenizer_dir);

}  // namespace mit2
