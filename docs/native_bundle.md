# Native Bundle Specification

The native bundle is deliberately fixed for IndexTTS2. It is not a generic
tensor exchange format.

## Files

- `manifest.json`: tensor directory, metadata, offsets, shape, dtype, layout,
  and checksum.
- `weights.bin`: aligned little-endian tensor payloads.
- `tokenizer/`: copied tokenizer assets needed by the text frontend.
- `voices/*/manifest.json` plus `weights.bin`: optional converted voice
  profiles.

## Alignment

`weights.bin` starts with:

```text
bytes 0..3   "MIT2"
uint32       version
uint32       alignment
```

Each tensor payload starts at a 4096-byte aligned offset. This matches the
runtime goal of mmap-backed, page-friendly weight slices and no-copy Metal
buffer wrapping where possible.

## Dtypes

Initial supported dtypes:

- `f32`
- `f16`
- `i64`
- `i32`
- `u32`
- `u8`

Default conversion preserves checkpoint dtype unless `--force-dtype` is set.
Quantized formats are intentionally absent until fp16/fp32 parity is stable.

## Voice Profile Tensors

Native voice profile conversion preserves:

- `spk_cond_emb`
- `s2mel_style`
- `s2mel_prompt`
- `mel`

The metadata carries `voice_name`, `ref_audio_path`, and `created_at` when the
source `.pt` profile provides them.

## Validation Rules

The loader must reject:

- unsupported `format` or `version`,
- unsupported endianness,
- invalid `weights.bin` MIT2 header,
- manifest/header version or alignment mismatch,
- missing `weights.bin`,
- duplicate tensor names,
- zero-byte tensors,
- unaligned tensor offsets,
- overlapping tensor payload ranges,
- dtype/shape byte-count mismatches,
- tensors extending beyond the mapped file,
- missing required tensors at stage-specific boundaries.

Python validation additionally checks SHA-256 for each tensor payload. The
native `--inspect-model-bundle` and `--inspect-voice-bundle` contract gates
also verify every tensor payload SHA-256, so bundle corruption is caught before
TTS when the gates are enabled.
`metal_indextts2.tools.convert_voice --validate-source` also reloads the
written bundle and compares every required voice tensor byte-exactly against
the source `.pt` profile after any requested dtype conversion.

The native runtime exposes model and voice-profile contract gates:

```bash
./build/mit2_runtime --inspect-model-bundle artifacts/test_model_bundle
./build/mit2_runtime --inspect-voice-bundle artifacts/test_voice_bundle
```

The model gate verifies MIT2 format/version/endianness/alignment, weights
header consistency, non-overlapping aligned payloads, dtype byte counts, full
payload SHA-256, required GPT/semantic-codec/S2Mel/BigVGAN components, and 24
required sentinel tensors. It prints integrity counters such as
`sha256_verified_count`.
Clone-time encoder components are separate model-bundle contracts. CAMPPlus
weights are exported with tensor names prefixed by `campplus.` and
`component=campplus`; they are validated by
`mit2_tts --clone-encoder-model-readiness MODEL_BUNDLE_DIR` rather than by the
hot cached-voice `--inspect-model-bundle` gate. That clone gate requires
CAMPPlus sentinel tensors from the FCM head, xvector TDNN, CAM dense TDNN
blocks, transit layers, and final dense projection to match the expected f32
shapes before `has_campplus_model_contract` is true.

The voice gate accepts only native voice bundles that contain f32 `component=voice`
entries for `spk_cond_emb` (`[1,tokens>0,1024]`), `s2mel_style` (`[1,192]`),
`s2mel_prompt` (`[1,tokens>0,512]`), and `mel` (`[1,80,tokens>0]`). It also
checks tensor byte counts, full payload SHA-256, and requires `s2mel_prompt`
token count to match `mel` frame count. Each command prints a JSON report with
`ok: true` on success and exits non-zero on contract violations.
`metal_indextts2.tools.synthesize_native_hot --validate-voice-contract`
invokes the same gate before synthesis and stores the report as
`voice_contract`.
`metal_indextts2.tools.synthesize_native_hot --validate-model-contract`
invokes the model gate before synthesis and stores the report as
`model_contract`.
