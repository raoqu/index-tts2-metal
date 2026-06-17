# Vendored third-party dependencies

These sources are vendored (committed in-tree) so the native runtime builds with
**no network access** — `cmake`/`build.sh` never download anything. They back the
full native text frontend (`runtime/impl/text_frontend.cpp`), which replaces the
old narrow CJK tokenizer and matches the reference IndexTTS2 Python frontend.

| Dir | Upstream | Pinned version | License |
|-----|----------|----------------|---------|
| `sentencepiece/` | https://github.com/google/sentencepiece | `v0.2.0` | Apache-2.0 |
| `kaldifst/`      | https://github.com/k2-fsa/kaldifst       | `v1.7.17` | Apache-2.0 |
| `openfst/`       | https://github.com/csukuangfj/openfst    | `sherpa-onnx-2024-06-13` | Apache-2.0 |

How they are built (see the top-level `CMakeLists.txt`):
- `sentencepiece/` — `add_subdirectory` with `SPM_ABSL_PROVIDER=internal` (uses the
  bundled minimal abseil under `sentencepiece/third_party/absl`, no network), built
  as `sentencepiece-static`.
- `openfst/` — `add_subdirectory` with all extensions disabled (`HAVE_*` OFF,
  `HAVE_SCRIPT` ON); only the core `fst` library is built. Already carries the
  kaldifst test/extension strip patch (the upstream sed in kaldifst's
  `cmake/openfst.cmake`), so no PATCH_COMMAND is needed.
- `kaldifst/` — we do NOT use kaldifst's own CMake (it FetchContent-downloads
  openfst). Instead the top-level `CMakeLists.txt` compiles the ten
  `kaldifst/kaldifst/csrc/*.cc` files directly into a `kaldifst_core` static lib
  that links `fst`. This is what provides `kaldifst::TextNormalizer`.

## Local patches

- `sentencepiece/third_party/absl/flags/flag.cc`: renamed the global `help` flag
  to `spm_help` (symbol `FLAGS_help` -> `FLAGS_spm_help`). Both SentencePiece's
  bundled abseil and OpenFST define a global `FLAGS_help`; linking both static
  libs into `mtts` caused a duplicate-symbol error. The flag is only used by
  SentencePiece's command-line tools, which we do not build. (Search the file
  for "mit2 vendor patch".)

If you re-vendor any of these from upstream, re-apply the patch above.

## Re-vendoring

The sources came from CMake `FetchContent` of the pinned tags, copied out of
`build/_deps/*-src`, with `.git`, `*.bak`, and SentencePiece's `data/`, `python/`,
and `doc/` directories removed to keep the tree small.
