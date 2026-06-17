# Pure Metal IndexTTS2 Optimization Plan

## Operating Principles

- Target IndexTTS2 specifically; do not build a generic Torch, ONNX, or MLX runtime.
- Keep PyTorch IndexTTS2 as the source of truth until every native stage has golden tests.
- Port the hot TTS path before the cold voice-clone path.
- Every step must have a local validation artifact: tensor diff, token diff, audio diff, benchmark, or build/test result.
- Optimization follows correctness. Quantization follows stable fp16/fp32 parity.

## Phase 0: Repository and Baseline Audit

Status: partial

Deliverables:

- Confirm canonical working directory. Current implementation uses `metal-indextts` and reads PyTorch reference assets from `index-tts`.
- Record git status and avoid modifying unrelated dirty files. Current implementation directory is not a git repository.
- Add a profiling harness around `IndexTTS2.infer_generator`. Implemented as `metal_indextts2.tools.profile_baseline`.
- Capture baseline timings on Apple Silicon:
  - clone from new reference audio,
  - TTS from cached voice profile,
  - short text,
  - long multi-segment text,
  - emotion audio,
  - emotion vector,
  - emotion text if Qwen is available.

Validation:

- `python` baseline script emits JSON with `gpt_gen_time`, `gpt_forward_time`, `s2mel_time`, `bigvgan_time`, `total`, `audio_len`, `rtf`.
- Baseline output wavs are saved and checksummed.
- The same input can be rerun without changing voice profile tensors.

## Phase 1: Golden Test Corpus

Status: partial

Deliverables:

- Build `tests/golden/` generator using the existing PyTorch pipeline. Implemented as `metal_indextts2.tools.generate_golden`.
- Save stage outputs for fixed fixtures:
  - text tokens and segment boundaries,
  - voice profile tensors,
  - emotion vector and merged emotion vector,
  - GPT input embeddings,
  - first N logits for greedy decode,
  - generated mel codes,
  - GPT latent,
  - S2Mel length-regulator output,
  - CFM outputs at selected diffusion steps,
  - final mel,
  - BigVGAN waveform.
- Provide deterministic settings:
  - greedy decode first,
  - fixed RNG seed for sampling tests,
  - fixed `max_text_tokens_per_segment`,
  - fixed `max_mel_tokens`.

Validation:

- Golden files include shapes, dtype, checksum, max/mean value, and source commit.
- A comparison script reports absolute error, relative error, cosine similarity, token exact match, and audio metrics.
- Golden generation is independent of future native runtime code.

## Phase 2: Native Model Bundle Specification

Status: partial

Deliverables:

- Define a stable IndexTTS2 native bundle. Initial implementation exists in `metal_indextts2.bundle` and `docs/native_bundle.md`:
  - `manifest.json`,
  - `weights.bin`,
  - tokenizer assets,
  - optional `voices/*.bin`.
- Specify tensor layout per component:
  - GPT,
  - semantic codec vq embedding path needed by hot TTS,
  - S2Mel gpt layer,
  - length regulator,
  - DiT/CFM,
  - BigVGAN.
- Decide default dtype:
  - fp16 weights where safe,
  - fp32 accumulators for normalization, softmax, selected audio-sensitive ops.
- Define alignment, offset, checksum, and endianness rules.

Validation:

- Converter can round-trip every tensor shape and checksum.
- Native loader can list all expected tensors and reject missing/unexpected tensors.
- Tensor layout tests compare converted tensor views against PyTorch tensors.

## Phase 3: Native Runtime Skeleton

Status: partial

Deliverables:

- Create C/C++/Objective-C++ runtime boundary. Initial implementation exists under `runtime/`:
  - C/C++ owns model semantics, scheduler, tensor metadata.
  - Objective-C++ owns Metal device, queue, library, buffers, pipeline state.
- Add Metal runtime diagnostics:
  - selected device,
  - unified memory size,
  - supported GPU family,
  - pipeline compile mode,
  - command-buffer submission count,
  - kernel GPU elapsed time,
  - buffer allocation count and cumulative allocated bytes,
  - peak/live buffer bytes. Process RSS and peak RSS sampling are implemented in `mit2_runtime --diagnostics` and hot WAV JSON outputs; `MetalContext::resource_stats()` now tracks per-context command-buffer submissions, buffer allocations, cumulative allocated bytes, and command-buffer GPU elapsed seconds for diagnostics, hot gates, and benchmark summaries.
- Add command-line test runner. Implemented as `mit2_runtime`:
  - load model bundle,
  - load golden fixture,
  - run one selected op/stage,
  - dump diff JSON.

Validation:

- Runtime builds on macOS with `make` or CMake.
- Metal library compiles from source and can also use precompiled `.metallib`.
- A no-op tensor copy test passes for multiple shapes and alignments.

## Phase 4: Tensor Allocator and Buffer Residency

Status: partial

Deliverables:

- Implement persistent weight buffers and scratch buffer allocator. The native
  `ScratchAllocator` now has alignment-safe allocation, checkpoint/rewind,
  reset with retained high-water peak, and overflow/invalid-alignment checks
  covered by `mit2_runtime --test-scratch-allocator`. `mit2_runtime
  --plan-hot-scratch MAX_PREFIX_TOKENS MAX_CODES PROMPT_TOKENS` now produces a
  ds4-style reusable scratch capacity estimate for GPT KV/logits/codes,
  condition export, S2Mel/CFM, and BigVGAN feature buffers; the companion
  `--plan-hot-scratch-inputs CONDS_F32 TEXT_IDS_U32 MAX_CODES PROMPT_TOKENS`
  mode derives the prefix budget from real frontend artifacts as
  `cond_tokens + text_tokens + 2`. Seeded and shared seeded synthesis summaries
  now emit the same `planned_scratch_*` fields, and benchmark summaries extract
  and exact-match them so real hot runs are tied to the reusable scratch plan;
  the benchmark layer also derives capacity/unshared and reuse-savings ratios
  and exposes planned-scratch capacity/reuse budget checks for regression
  gates. Short staged-vs-shared resource comparisons now include planned-scratch
  numeric deltas and exact invariants so the same-process experiment cannot
  diverge from the staged reusable scratch plan for identical inputs. Seeded,
  sampled-seeded, and shared-seeded hot summaries now also report
  `planned_scratch_actual_codes`, actual generated/total mel tokens, slack
  against the planned `max_codes`/mel-token envelope, and
  `planned_scratch_covers_actual`, tying the ds4-style reusable plan to the
  actual generated sequence rather than only to the requested upper bound.
  `mit2_runtime --test-metal-scratch-arena` verifies the Metal-side primitive
  for binding two non-overlapping `ScratchAllocator` offsets inside one
  preallocated `MTLBuffer`; the real hot condition merge now uses one
  arena-backed `MTLBuffer` for prompt, generated, and output slices, and
  `add_f32` now uses arena-backed input/output slices for residual and S_infer
  additions across the hot graph. The acoustic CFM Euler update also uses
  arena-backed `x`/`dphi`/CFG-`dphi`/output slices, moving another repeated
  per-step allocation cluster to the ds4-style offset-dispatch pattern. DiT
  `concat_rows_f32`, `dit_input_merge_f32`, and
  `dit_input_merge_batched_f32` now use the same arena-backed input/output
  packing for skip and CFG input assembly. Wavenet `wavenet_gate_f32` and
  `wavenet_res_skip_update_f32` now use arena-backed packing for gate,
  residual, skip, mask, and output slices in the acoustic Wavenet stack. The
  shared `linear_rows_f32` primitive now also uses arena-backed
  weight/bias/input/output packing, covering the repeated projection calls
  across GPT conditioning/decode, Perceiver, S2Mel, DiT, Wavenet, and BigVGAN
  paths. Row-wise `layernorm_rows_f32`, `adaptive_layernorm_rows_f32`,
  `adaptive_rmsnorm_rows_f32`, `rmsnorm_rows_f32`, and
  `rmsnorm_rows_eps_f32` now use arena-backed x/parameter/output packing for
  repeated normalization calls across the same hot graph. Simple hot helpers
  used by Conformer/Perceiver/DiT feed-forward blocks and GPT/S2Mel
  embedding/decode, including `silu_mul_f32`, `mask_rows_f32`,
  `glu_split_f32`, `geglu_erf_split_f32`, `embedding_f32`, and single-row
  `linear_f32`, now also use arena-backed input/parameter/output packing. The
  remaining pointwise and single-vector hot helpers, including `silu_f32`,
  `gelu_f32`, `tanh_f32`, `clamp_f32`, `softmax_f32`, `layernorm_f32`,
  `rmsnorm_f32`, and `timestep_embedding_f32`, now follow the same
  offset-binding pattern for GPT decode and diffusion setup. Length-regulator
  and BigVGAN helpers including `nearest_interpolate_f32`, `groupnorm1_f32`,
  `mish_f32`, and `avg3_f32` now also use arena-backed tensor slices. Hot 1D
  convolution helpers, including same, reflect-same, batched reflect-same,
  depthwise, dilated, and transposed 1D conv, now pack
  input/weight/bias/output tensors into arena slices while preserving scalar
  parameter buffers. GPT/basic attention helpers, including full-head, causal,
  single-query, cached GPT, and masked single-head attention, now use
  arena-backed Q/K/V/cache/mask/output tensor slices. Cross-attention,
  Conformer relative attention, and DiT QKV/RoPE attention helpers now use the
  same arena-backed tensor-slice convention. GPT/emotion subsampling and BigVGAN
  anti-aliased activation helpers now also use arena-backed tensor slices.
  `MetalContext` now owns a grow-only reusable scratch arena for those primitive
  tensor slices, replacing per-dispatch arena allocation with same-context
  `MTLBuffer` reuse. Scalar kernel arguments now use a same-context reusable
  constant-buffer ring, preserving the existing offset-0 scalar ABI while
  avoiding per-dispatch 4-byte buffer churn. The short fixed-seed hot smoke
  remains byte-exact and lowers counted GPT buffer allocations from 22,831
  before primitive arena reuse to 33, and the acoustic WAV stage from 2,837 to
  41. GPT cached decode/generation now has the first keyed resident weight
  path: qkv, attention projection, MLP projection, mel-head linear weights, and
  the cached path's `ln_1`, `ln_2`, `gpt.gpt.ln_f`, and `gpt.final_norm`
  LayerNorm parameters are uploaded once per `MetalContext` and reused by
  resident LayerNorm helpers. For GPT's 1280-wide full-sequence path, row-wise
  LayerNorm now uses a high-width serial-row Metal branch above width 1024,
  keeping resident parameters and one dispatch per LayerNorm while avoiding the
  medium-text drift seen in the fully parallel row reduction; dense GPT
  projections remain resident and batched. GPT text, text-position, mel-token, and
  mel-position embedding tables now also use keyed resident buffers in prepare,
  greedy, and latent-forward input assembly. The semantic codec hot `vq2emb`
  path now also keeps the codebook and out-project weight/bias resident. S2Mel
  length-regulator now keeps `content_in_proj`, convolution, and GroupNorm
  weights resident across the front projection and convolutional LR stack. The
  S2Mel/CFM Wavenet stack now keeps the global condition projection, eight
  reflect-padded input convolutions, and eight res/skip convolutions resident
  across diffusion steps. DiT timestep MLPs, input merge projections,
  transformer attention/FFN projections, skip-in/post-transformer projections,
  adaptive RMSNorm gamma, and final-layer modulation/projection now also use
  resident linear/norm/conv weights in the CFM loop. The
  BigVGAN acoustic path now keeps `conv_pre`, the six ConvTranspose upsamplers,
  all AMP residual-block convolutions, SnakeBeta activation/filter parameters,
  and `conv_post` vocoder weights resident. The
  short fixed-seed smoke remains byte-exact and GPT generation stays around
  4.8s on the current host, while counted allocation bytes include one-time
  resident GPT linear, LayerNorm, embedding, semantic `vq2emb`, LR stack, and
  DiT linear/norm, Wavenet/BigVGAN body/front-path convolution, and
  activation-parameter uploads.
  Remaining Phase 4 work is
  expanding persistent weight residency to the broader GPT conditioning/front-end,
  Conformer/S2Mel/CFM norm and remaining helper weights, and production-length scratch integration
  across the full chain.
- Objective-C++ ARC is enabled for the Metal runtime, fixing retained `newBuffer*` objects that previously drove the short condition-to-WAV path to roughly 145 GB RSS. The short same-process from-codes hot gate now completes at roughly 3.10 GB peak RSS, while production-length GPT-to-waveform still needs ds4-style persistent weight residency plus full-chain scratch integration beyond primitive calls.
- Preallocate maximum hot-path buffers:
  - GPT KV cache,
  - logits,
  - mel codes,
  - S2Mel sequences,
  - CFM diffusion state,
  - BigVGAN intermediate feature maps.
- Add debug flags:
  - disable fusion,
  - disable command batching,
  - capture Metal workload,
  - dump intermediate tensors.

Validation:

- Repeated runs do not increase live allocation count.
- Peak memory is reported and bounded by planned tensor sizes.
- Tensor lifetime tests catch overlapping writes.

## Phase 5: Primitive Kernels

Status: partial

Deliverables:

- Implement and test core kernels:
  - copy, slice, concat, transpose. Copy is implemented and tested.
  - embedding lookup. Implemented and tested.
  - elementwise add/mul/scale/clamp/tanh/sigmoid/silu/mish. Add, SILU, clamp, tanh, and Mish are implemented and tested.
  - LayerNorm, RMSNorm, GroupNorm. LayerNorm, batched LayerNorm, and RMSNorm are implemented and tested.
  - softmax. Implemented and tested for one-row GPT logits.
  - top-k/top-p/repetition-penalty support,
  - GEMV/GEMM. Row-major linear/GEMV and batched row-major linear over token rows are implemented and tested for GPT hidden-width fixtures.
  - Conv1d,
  - ConvTranspose1d,
  - nearest interpolation. Implemented and tested against CPU reference plus PyTorch index behavior.

Validation:

- Each kernel has shape fixtures from IndexTTS2, not only synthetic square tensors.
- Error thresholds are documented per dtype.
- CPU reference and PyTorch reference comparisons both pass.

## Phase 6: GPT Autoregressive Decode

Status: partial

Deliverables:

- Port `UnifiedVoice` hot decode path:
  - conditioning latent input. The `prepare_gpt_inputs` conditional-latent padding and attention-mask construction path is implemented as `mit2_runtime --test-gpt-prepare-inputs`.
  - conditioning front-end. `metal_indextts2.tools.generate_gpt_subsampling_golden` now saves a short `gpt.conditioning_encoder.embed` boundary, and `mit2_runtime --test-gpt-subsampling-golden` reproduces Conv2dSubsampling2's stride-2 conv, ReLU, PyTorch flatten order, 261632-wide linear projection, positional scale, and output mask from converted weights. `mit2_runtime --test-gpt-subsampling-metal-linear-golden` now runs the stride-2 conv/ReLU/PyTorch flatten through resident Metal `subsampling_conv2d_relu_flat_f32_resident` and that wide projection through resident Metal `linear_rows_f32_resident`, giving a focused gate for the largest subsampling front-end subgraph. `metal_indextts2.tools.generate_gpt_conformer_attn_golden`, `generate_gpt_conformer_conv_golden`, and `generate_gpt_conformer_ff_golden` save layer-0 Conformer subgraph boundaries for relative self-attention, convolution, and feed-forward tail; `mit2_runtime --test-gpt-conformer-attn-golden`, `--test-gpt-conformer-conv-golden`, and `--test-gpt-conformer-ff-golden` reproduce those subgraphs from converted weights; `--test-gpt-conformer-ff-metal-golden` runs the feed-forward tail through resident Metal batched LayerNorm, `linear_rows_f32_resident`, SiLU, residual add, and final resident batched LayerNorm; `--test-gpt-conformer-attn-metal-proj-golden` runs resident attention norm, q/k/v/pos projections, relative attention score/softmax/context, output projection, and residual add through Metal via resident-bias `conformer_rel_attention_context_f32_resident`; `--test-gpt-conformer-conv-metal-golden` runs convolution-module resident LayerNorms, resident pointwise projections, resident depthwise Conv1d, SiLU, and residual add through Metal, including mask application through `mask_rows_f32` and GLU through `glu_split_f32`. `metal_indextts2.tools.generate_gpt_conformer_block_golden` saves the full layer-0 Conformer block boundary, and `mit2_runtime --test-gpt-conformer-block-golden` composes attention, convolution, feed-forward, and final normalization in true Conformer order. `metal_indextts2.tools.generate_gpt_conformer_stack_golden` now saves the six-layer Conformer stack boundary, and `mit2_runtime --test-gpt-conformer-stack-golden` reuses the parameterized native block implementation for all six `gpt.conditioning_encoder.encoders.*` layers plus `after_norm`; `--test-gpt-conformer-stack-metal-ff-golden` runs all six resident feed-forward tails plus resident `after_norm` through Metal; `--test-gpt-conformer-stack-metal-attn-ff-golden` additionally runs resident attention norm/projection/core/bias/output/residual sub-ops through Metal; `--test-gpt-conformer-stack-metal-attn-conv-ff-golden` also runs all six convolution modules through resident Metal-backed LayerNorm/mask/Linear/GLU/resident-depthwise-conv/SiLU/add kernels, and `mit2_runtime --export-gpt-conformer-stack` exports this post-Conformer `conditioning_context.f32` through that resident Metal-attn-core-conv-FF stack path. `metal_indextts2.tools.generate_gpt_perceiver_golden` saves the downstream `conditioning_encoder` output boundary plus Perceiver mask/output, and `mit2_runtime --test-gpt-perceiver-golden` now runs `gpt.perceiver_encoder` through resident-weight Metal `linear_rows_f32_resident`, `cross_attention_heads_masked_f32`, `geglu_erf_split_f32`, residual add, and `rmsnorm_rows_f32_resident` for 16-token and 64-token context fixtures. `mit2_runtime --export-gpt-perceiver` exports `speech_conditioning_latent.f32` from `conditioning_context.f32` plus `perceiver_mask.u32`, and `generate_gpt_frontend`/`synthesize_native_hot` can opt into the Perceiver path through `--native-perceiver` with `perceiver_source: native_metal_resident_linear_cross_attn_geglu_rmsnorm`. `metal_indextts2.tools.generate_gpt_emovec_golden` now saves a short `gpt.get_emovec` boundary, and `mit2_runtime --test-gpt-emovec-golden`/`--export-gpt-emovec` reproduce the emotion subsampling, four-layer/four-head `gpt.emo_conditioning_encoder`, `gpt.emo_perceiver_encoder`, `emovec_layer`, and `emo_layer` path from converted weights; the Metal-linear gate now runs emotion subsampling conv/projection, the four-layer resident-weight emotion Conformer stack, resident-weight emotion Perceiver, and final projections through Metal-backed resident kernels. `generate_gpt_frontend` and `synthesize_native_hot` now expose full-context native emovec with `--native-emovec`; `--native-emovec-input-tokens 0` consumes the full voice conditioning sequence and positive `N` remains a truncated debug path. These move the native boundary inside GPT conditioning and emotion-vector computation for the production-length cached voice profile, while clone-time feature extraction remains Python-owned.
  - text and mel embeddings. Text embedding plus text-position embedding lookup for the generation prefill input is implemented as part of `mit2_runtime --test-gpt-prepare-inputs`; generated mel-token embedding plus mel-position embedding is implemented as part of `mit2_runtime --test-gpt-greedy`.
  - GPT2 transformer layers. Layer 0 `ln_1 -> c_attn(qkv)` projection is implemented as `mit2_runtime --test-gpt-layer0-qkv`; layer 0 causal self-attention plus `c_proj` is implemented as `mit2_runtime --test-gpt-layer0-attn`; layer 0 cached single-query attention against K/V prefixes is implemented as `mit2_runtime --test-gpt-layer0-kv-attn`; full layer 0 attention residual, `ln_2`, MLP `c_fc -> gelu_new -> c_proj`, and final residual are implemented as `mit2_runtime --test-gpt-layer0-block`; the 24-layer GPT2 transformer stack plus `gpt.gpt.ln_f` is implemented as `mit2_runtime --test-gpt-transformer-stack`; IndexTTS2 `final_norm -> mel_head` logits and last-token greedy argmax parity are implemented as `mit2_runtime --test-gpt-logits`; no-KV full-sequence greedy mel-token loop with generated mel-token embeddings is implemented as `mit2_runtime --test-gpt-greedy`.
  - KV cache. Layer 0 single-query K/V-cache attention semantics are implemented as `mit2_runtime --test-gpt-layer0-kv-attn`; full 24-layer one-step cached decode logits parity is implemented as `mit2_runtime --test-gpt-kv-decode`; multi-step cached greedy loop with per-layer persistent K/V append is implemented as `mit2_runtime --test-gpt-kv-greedy`. The Metal cached decode path now fuses all GPT attention heads for the current token into `gpt_cached_attention_f32`, keeps decode weights pre-transposed for the duration of a generation call, and reuses mel embeddings/final head weights instead of copying and transposing them every step. Short, 16-token, and 30-step PyTorch greedy generated-code parity are implemented with `metal_indextts2.tools.generate_gpt_golden` plus `mit2_runtime --test-gpt-kv-greedy-golden`; the medium-text 96-code CPU-LR fixture is byte-exact through the raw-input native compare gate `--test-gpt-kv-codes-inputs` in 9.75 seconds on the current Apple Silicon test host. `generate_gpt_golden --length-regulator-device cpu` provides a stable CPU length-regulator reference for longer fixtures where MPS GroupNorm/Conv reductions drift from CPU/native math.
  - final norm and mel head,
  - greedy decode.
- Native cached GPT code export can now start from raw `conds_latent.f32` and `text_ids.u32` plus a caller-provided max step count through `--export-gpt-kv-codes-inputs`, rather than reading the expected golden `codes.u32` to determine decode length. `--test-gpt-kv-codes-inputs` wraps the same raw-input native decode path with an in-runtime expected-code comparison; the short smoke and medium-text 96-code fixture are byte-exact with their PyTorch golden codes.
- Keep sampling support as a substep after logits parity:
  - temperature,
  - top-k,
  - top-p,
  - repetition penalty,
  - seeded RNG.
  Native sampling processors are implemented as `--test-gpt-sampling-processors`.
  Raw frontend-input sampled code export is implemented as
  `--export-gpt-kv-codes-inputs-sampled`, and
  `--test-gpt-sampled-inputs-determinism` now verifies same-seed reproducibility
  on real `conds_latent.f32`/`text_ids.u32` frontend tensors while reporting the
  alternate-seed codes for stochastic-control drift checks.

Validation:

- First-token logits match PyTorch within tolerance.
- Greedy generated code sequence exactly matches for golden fixtures.
- KV cached decode matches non-cached prefill/decode for short fixtures.
- Per-token latency is reported.

## Phase 7: GPT Latent Forward

Status: partial

Deliverables:

- Port the second GPT forward used after generated codes.
- Port or reuse required `speech_conditioning_latent`, text token, mel code, and emotion vector handling.
- Port `s2mel.models['gpt_layer']` projection. Implemented as `mit2_runtime --test-gpt-layer` using real converted bundle weights; the Metal path runs all three projection layers through batched resident `linear_rows_f32_resident`.
- Short PyTorch golden parity for the second GPT forward, `gpt_layer`, semantic `vq2emb`, and combined `S_infer` is implemented as `mit2_runtime --test-gpt-latent-golden`.

Validation:

- GPT latent tensor matches golden fixture by max error and cosine similarity.
- `gpt_layer` output matches golden fixture.
- Runtime reports separate timing for AR decode and latent forward.

## Phase 8: Semantic Codec Hot Subset

Status: partial

Deliverables:

- Port only `semantic_codec.quantizer.vq2emb` first because hot TTS uses generated codes to fetch embeddings. Implemented as `mit2_runtime --test-vq2emb` for Metal codebook lookup plus batched Metal weight-normalized out projection using real MaskGCT semantic codec weights.
- Defer full `semantic_codec.quantize` until clone path.

Validation:

- `S_infer` matches PyTorch for golden code sequences. Short PyTorch golden comparison is implemented as part of `mit2_runtime --test-gpt-latent-golden`; the Metal hot path now runs `vq2emb` through `embedding_f32` plus batched `linear_rows_f32`, combines `gpt_layer` and `vq2emb` with `add_f32` before length regulation, then assembles cached prompt rows plus length-regulator rows through Metal `hot_condition_merge_f32`; production-length S2Mel golden comparison remains.
- Codebook index bounds and shape assertions are enforced.

## Phase 9: S2Mel Length Regulator

Status: partial

Deliverables:

- Port continuous `InterpolateRegulator` path:
  - linear content projection. Implemented as part of `mit2_runtime --test-length-regulator-front` using real bundle weights; the Metal path runs `content_in_proj` through batched `linear_rows_f32`.
  - nearest interpolation to target length. Implemented and cross-checked against PyTorch nearest index behavior.
  - Conv1d + GroupNorm + Mish stack. Implemented as part of `mit2_runtime --test-length-regulator-full` using real bundle weights.
  - final Conv1d. Implemented as part of `mit2_runtime --test-length-regulator-full`.

Validation:

- `prompt_condition` loaded from voice profile is accepted without recomputation.
- `cond` for generated speech matches golden fixture. Short PyTorch golden comparison is implemented as `mit2_runtime --test-length-regulator-golden`.
- Target length rounding rule `(code_lens * 1.72).long()` is preserved in `metal_indextts2.tools.generate_gpt_golden` and consumed by `mit2_runtime --test-length-regulator-golden`; short, 16-token, max-64/30-code CPU-LR, and medium-text 96-code CPU-LR fixtures pass. Native nearest interpolation now uses the PyTorch-compatible float32 scale rule, with a 96->165 primitive edge gate plus `--export-length-regulator-golden`/`--export-length-regulator-stages-golden` diagnostics for longer LR fixtures; full production-length full-stage coverage remains.

## Phase 10: DiT / CFM Solver

Status: partial

Deliverables:

- Port CFM Euler loop with 25 diffusion steps. A minimal unbatched fixed-step loop is implemented as `--test-cfm-euler`; no-CFG PyTorch Euler golden parity is implemented as `--test-cfm-euler-golden`; CFG conditional/null solver semantics with 25 production steps is implemented as `--test-cfm-euler-cfg`; CFG PyTorch Euler golden parity is implemented as `--test-cfm-euler-cfg-golden`; voice-conditioned short-prompt, max-64/30-code CPU-LR, and medium-text 96-code/165-generated-frame S2Mel full-mel golden parity are implemented as `--test-s2mel-full-golden`, and those same short/max-64/medium acoustic boundaries now have a lightweight Metal-only explicit tensor gate through `--test-s2mel-full-inputs`. The explicit-input gate prints first-attempt and final errors and replays once in the same Metal context only after an initial tolerance failure, which makes cold-run drift visible while keeping the normal path single-pass. The medium-text 173-total-frame fixture is stabilized by saturating Wavenet gate `tanh` inputs before the Metal math path, routing DiT attention sequences above 128 tokens through the existing per-head Metal attention fallback instead of the short-sequence fused QKV/RoPE kernel, and running a one-step Metal-only CFM warm-up before full Metal CFM replay to avoid cold-run drift. `--trace-s2mel-cfm-golden` and `--trace-s2mel-cfm-error-golden` remain available for long-CFM nonfinite and drift localization. The Metal solver now fuses CFG branch combination, fixed-step Euler update, and prompt-region zeroing through `cfm_euler_update_f32`; the conditional/null branches share each step's Metal timestep embeddings instead of recomputing `t_embedder`/`t_embedder2`; the CFG estimator path now batches input merge with per-branch style through `dit_input_merge_batched_f32`, batches the DiT transformer stack across conditional/null branches with `dit_attention_qkv_rope_batched_f32`, batches the row-wise post-transformer projection core and Wavenet-after final adaptive layer, skips the unused intermediate post-transformer `conv2` in estimator execution, reuses the shared per-step Wavenet global-condition projection across CFG branches, batches the Wavenet convolution stack with branch-isolated reflect padding through `conv1d_reflect_same_batched_f32`, and batches the final kernel=1 output `conv2`; remaining estimator CFG work is persistent buffer planning and full production-length prompt/full-mel golden-fixture comparison.
- Port DiT estimator:
  - timestep embedder. Implemented as `mit2_runtime --test-timestep-embedder` for `t_embedder` and `t_embedder2` using real bundle weights; the Metal path runs the MLP projections with batched `linear_rows_f32`.
  - style conditioning,
  - condition projection. Implemented as part of `mit2_runtime --test-dit-input-merge`, with the Metal path running batched `linear_rows_f32`.
  - pre-transformer input merge with prompt mel, current mel, semantic condition, and repeated style. Implemented as `mit2_runtime --test-dit-input-merge`; the Metal path assembles the 864-wide rows through `dit_input_merge_f32` before the batched `cond_x_merge_linear`, and the CFG path uses `dit_input_merge_batched_f32` to pack conditional/null branches with separate style vectors.
  - transformer with non-causal attention mask. Layer 0 adaptive norms, attention projection (`wqkv` and `wo`), feed-forward projection (`w1/w3/w2`), unmasked RoPE/SDPA attention core, full layer 0 block with key mask/residual composition, and all-layer transformer stack with U-ViT skip-in connections plus final adaptive norm are implemented as `--test-dit-adaptive-norm`, `--test-dit-attention-proj`, `--test-dit-feed-forward`, `--test-dit-attention-core`, `--test-dit-transformer-block0`, and `--test-dit-transformer-stack`; the Metal path now batches input merge projections, adaptive RMSNorm rows through `adaptive_rmsnorm_rows_f32`, attention `wqkv`/`wo`, short-sequence QKV split/RoPE plus multi-head masked attention through `dit_attention_qkv_rope_f32` with long-sequence fallback, branch-isolated batched short-sequence QKV/RoPE attention through `dit_attention_qkv_rope_batched_f32`, batched transformer block/stack execution for the CFG conditional/null pair, attention/FFN residual adds through `add_f32`, feed-forward `w1`/`w3`/`w2` with fused `silu_mul_f32`, skip-in row assembly/projection with `concat_rows_f32`, post-transformer `skip_linear`/`conv1`/`res_projection`, and final adaptive output LayerNorm/modulation/linear with `dit_input_merge_f32`/`adaptive_rmsnorm_rows_f32`/`dit_attention_qkv_rope_f32`/`dit_attention_qkv_rope_batched_f32`/`add_f32`/`concat_rows_f32`/`adaptive_layernorm_rows_f32`/`linear_rows_f32`, reducing repeated transformer and row-wise tail command dispatches in the CFM acoustic loop.
  - long skip connection. Implemented as part of `--test-dit-post-transformer-proj`, along with `conv1`, `res_projection`, and `conv2` direct projection paths.
  - Wavenet final path. Final adaptive output layer is implemented as `--test-dit-final-layer`; Wavenet layer 0 condition/input gate is implemented as `--test-wavenet-layer0-gate`; full Wavenet residual/skip stack is implemented as `--test-wavenet-stack`. The Wavenet reflect-padded input convolution now runs through `conv1d_reflect_same_f32`, and the CFG path uses `conv1d_reflect_same_batched_f32` to batch conditional/null Wavenet stacks while preserving per-branch reflect padding; gate activation runs through `wavenet_gate_f32`; and residual update/skip accumulation/final masking run through `wavenet_res_skip_update_f32` in the Metal Wavenet layer and stack paths instead of CPU pad/slice and token/channel loops.
  - full DiT estimator single step. Implemented as `--test-dit-estimator-step` by composing timestep embedders, input merge, transformer stack, long skip, Wavenet, final adaptive layer, and output `conv2`; PyTorch golden parity for the same estimator boundary is implemented as `--test-dit-estimator-golden`.
- Preserve CFG stacking semantics. The native solver now preserves the conditional/null branch formula, reuses the per-step timestep embeddings across CFG branches, batches input merge, the DiT transformer stack, post-transformer row-wise projection core, Wavenet convolution stack with branch-isolated reflect padding, final adaptive layer, and final kernel=1 output `conv2` across the two CFG branches, reuses the shared Wavenet global-condition projection, and performs the post-estimator CFG/Euler update on Metal; production-length resource planning remains.

Validation:

- One DiT estimator call matches golden fixture.
- Selected Euler step outputs match fixture.
- Final mel output matches fixture within documented tolerance.
- Diffusion time is reported independently.

## Phase 11: BigVGAN Vocoder

Status: partial

Deliverables:

- Port BigVGAN inference:
  - pre-conv. Implemented as `mit2_runtime --test-bigvgan-conv-pre` using converted BigVGAN generator weights.
  - upsample ConvTranspose1d stack. Generic Metal `ConvTranspose1d`, the first BigVGAN upsampler `ups.0.0`, the complete 6-layer upsampler stack, and the composed `conv_pre -> upsamplers` front path are implemented as `mit2_runtime --test-bigvgan-upsample0`, `--test-bigvgan-upsampler-stack`, and `--test-bigvgan-front`.
  - AMP residual blocks. The first `resblocks.0` activation/conv pair, complete `resblocks.0` AMPBlock1 with dilation 1/3/5 residual pairs, first 3-block kernel-size group with Metal `avg3_f32` residual-group averaging, and complete 18-block upsample/resblock body are implemented as `mit2_runtime --test-bigvgan-resblock0-pair0`, `--test-bigvgan-resblock0`, `--test-bigvgan-resblock-group0`, and `--test-bigvgan-body`.
  - Snake/SnakeBeta activations. Fused Metal alias-free SnakeBeta activation is implemented as `mit2_runtime --test-bigvgan-activation-post` and `--test-bigvgan-activation-rb0`.
  - anti-alias up/down filters. The fixed BigVGAN upsample/downsample filter path is implemented inside the fused activation kernel.
  - post-conv and final clamp/tanh. Implemented as `mit2_runtime --test-bigvgan-post`; complete synthetic mel-to-waveform native vocoder parity is implemented as `mit2_runtime --test-bigvgan-vocoder`; PyTorch waveform golden comparison is implemented as `mit2_runtime --test-bigvgan-vocoder-golden` for synthetic mel plus S2Mel-generated short, max-64/30-code CPU-LR, and medium-text 165-frame mel fixtures.
- Evaluate fusion opportunities:
  - activation + filter,
  - conv + bias + activation,
  - residual block accumulation.

Validation:

- BigVGAN output waveform matches golden fixture within audio-tolerant thresholds.
- Listening checks pass on short and long examples.
- Vocoder RTF is reported.

## Phase 12: End-to-End Hot TTS

Status: partial

Deliverables:

- Native CLI accepts:
  - model bundle path,
  - native voice profile,
  - text,
  - generation parameters,
  - output wav path.
- Implement text processing either by:
  - reusing Python-generated token ids for first milestone,
  - using native `--export-text-ids-cjk` / `--native-cjk-text-ids` for narrow CJK plus limited ASCII/ASCII-punctuation fixtures, including focused temperature/operator/measure-unit and pure-ASCII unit suffix forms, then
  - bridging reference tokenizer segmentation through `--all-segments` for multi-segment synthesis, then
  - porting tokenizer/normalizer assets for full native CLI.

Validation:

- End-to-end greedy TTS from cached voice profile passes token, mel, and audio checks.
- Multi-segment text concatenation and interval silence match reference behavior.
- Native stage timing schema matches Python baseline schema.
- `synthesize_native_hot` reports now include a first-class `runtime_summary`
  block for each single-segment run, and aggregate/per-segment runtime summaries
  for multi-segment runs. The summary promotes native stage counts/seconds,
  predicted codes, resident peak bytes, command-buffer counts, allocation
  counts/bytes, GPU elapsed seconds, native GPT/condition/noise/acoustic phase
  seconds, native GPT decode code throughput/latency fields, and
  planned-scratch fields when emitted by the runtime. Multi-segment
  aggregate summaries now also merge per-segment planned scratch: capacity and
  phase peaks keep the maximum per-segment requirement, actual code/frame/slack
  counters and reuse-saves totals are summed across the full segmented run, and
  per-segment planned-scratch records remain attached for diagnosis. This keeps
  ds4-style resource regressions visible before benchmark summarization.
  Benchmark summarization now uses raw `runtime_json` when present and falls
  back to `runtime_summary` for trimmed reports, preserving resource budgets
  and baseline comparison coverage. `synthesize_native_hot --compact-report`
  now omits raw runtime stdout/stderr/json payloads while keeping
  `runtime_summary`, sidecar metadata, and benchmark-compatible resource fields;
  `benchmark_native_hot --compact-reports` forwards that mode to every case in
  the benchmark matrix. Benchmark summaries now record a reproducibility
  `config` block with selected model/runtime/voice inputs, case lists or
  from-report entries, generation controls, comparison thresholds, resource
  budgets, and wrapper flags such as `compact_reports`.
- `synthesize_native_hot` now validates generation controls through the same
  helper in the CLI and Python `run(args)` entry point before native runtime
  work starts, rejecting invalid max token/step budgets, non-finite or
  non-positive temperatures, invalid top-k/top-p/repetition-penalty sampling
  values, negative CFG/silence settings, and negative native emovec truncation.
  `benchmark_native_hot` now performs its own wrapper-level validation for
  segment/full-acoustic budgets, synthesis control passthrough values, and
  benchmark resource/regression thresholds, requested case names and uniqueness,
  clone/emotion required inputs, and `--from-reports`/`--baseline-summary`
  file inputs before it creates output directories or launches synthesis
  subprocesses, including JSON parseability, baseline `cases`, and report
  `output_wav` presence; archived reports may omit the WAV file only when
  `output_wav_sha256` is already recorded, and relative `output_wav` paths are
  resolved from the report JSON directory.
- A short cached-voice hot-path golden gate is implemented as `--test-hot-tts-golden`, chaining S2Mel Metal generated mel directly into BigVGAN Metal waveform from GPT-derived fixture inputs. `--test-hot-tts-from-gpt-golden` moves the native boundary earlier by running GPT-derived `S_infer` through the native length regulator, cached voice prompt/style, S2Mel Metal, and BigVGAN Metal; this now passes on the medium-text 96-code/165-frame waveform fixture. `--test-hot-tts-from-codes-golden` verifies conds/text/codes through native GPT latent forward, `gpt_layer`, semantic `vq2emb`, `S_infer`, length regulator, Metal `hot_condition_merge_f32` voice-conditioned S2Mel condition assembly, and BigVGAN waveform in one process; this also passes on the medium-text fixture with 42,240 output samples. Generated-condition hot gates keep strict upstream tolerances (`latent`, `S_infer`, length regulator, condition) and use explicit end-to-end acoustic tolerances (`mel_tolerance=0.006`, `wave_tolerance=0.01`) because small condition-level fp32 differences are amplified by the 25-step CFM and vocoder replay. The hybrid text-to-native-hot CLI, all-segments reference-tokenizer bridge, narrow native CJK/limited-ASCII segmented text-id bridge, clone-audio smoke, compact narrow-CJK cached-voice binary TTS entries `--tts-cjk`, `--tts-cjk-segments`, `--tts-cjk-sampled`, `--tts-cjk-segments-sampled`, lower-argument preset entries `--tts-cjk-preset` and `--tts-cjk-segments-preset`, launcher-oriented auto preset entry `--tts-cjk-auto-preset`, launcher-shaped standard product entry `--tts` and dedicated `mit2_tts` binary with no-bundle `--capabilities` JSON, and no-synthesis per-request text gate `--tts-cjk-text-readiness`, narrow-CJK cached-voice binary TTS entry `--synthesize-hot-text-cjk-seeded`, narrow-CJK sampled binary TTS entry `--synthesize-hot-text-cjk-sampled-seeded`, narrow-CJK multi-segment binary TTS entry `--synthesize-hot-text-cjk-segments-seeded`, and narrow-CJK sampled multi-segment binary TTS entry `--synthesize-hot-text-cjk-segments-sampled-seeded` are validated; the compact entries can also use `--tts-validate-bundles` to run native model/voice contract gates before synthesis, the `--tts` and `mit2_tts` entries validate both bundles by default, final `tts_cjk_product` summaries now include generated codes in the last JSON object, `--tts-cjk-auto-preset`, `--tts`, and `mit2_tts` report their single/segmented choice as `auto_segmented`, `--tts-product-readiness` provides a no-synthesis startup gate that validates both bundles and emits the final cached-CJK/general-text/native-clone readiness flags, `--tts-cjk-text-readiness` reports native text support, token ids, and preset segmentation plans before synthesis, and the four narrow-CJK text binary commands accept `PROMPT_TOKENS=0` to infer the prompt count from the native voice bundle. Full native TextNormalizer/SentencePiece integration, full production-length acoustic validation, a general multi-segment product entry beyond the narrow native CJK tokenizer bridge, and same-process production resource planning are still required for full cached-voice TTS.
- A golden-assisted native WAV output command is implemented as `--synthesize-hot-gpt-golden`, running GPT-derived `S_infer` through the native length regulator, Metal `hot_condition_merge_f32` prompt/LR condition assembly, cached voice prompt/style, S2Mel Metal, BigVGAN Metal, and PCM16 WAV writing. This is a smoke path for the acoustic half of the CLI, not yet the full text-to-wave native interface.
- A native hot-path smoke wrapper is implemented as `--synthesize-hot-native-golden`, running `conds/text -> native cached-GPT greedy codes -> native condition -> S2Mel Metal -> BigVGAN Metal -> PCM16 WAV` in one process from golden fixture inputs while writing codes/condition sidecars. The same stages can still be run explicitly with `--export-gpt-kv-codes-golden`, `--export-gpt-kv-codes-inputs`, `--export-hot-codes-condition-input`, `--export-hot-codes-condition-inputs`, `--test-hot-tts-condition-golden`, `--synthesize-hot-condition-golden`, and `--synthesize-hot-condition-inputs` for artifact-level debugging. The raw-input condition export computes generated condition length from `floor(code_len * 1.72)` natively and no longer requires `gpt_golden_dir` or `s2mel_golden_dir` to build the condition tensor. The raw condition/noise WAV command runs S2Mel and BigVGAN from explicit tensors plus `prompt_tokens`, `steps`, and `cfg_rate`, and its WAV output is byte-exact with the previous golden-dir condition synth smoke.
- A hybrid text-to-native-hot CLI is implemented as `python -m metal_indextts2.tools.synthesize_native_hot`, taking cached voice/text inputs, generating only the minimal GPT frontend tensors with `metal_indextts2.tools.generate_gpt_frontend`, then invoking the one-process raw native wrapper. The frontend directory contains `conds_latent.f32`, `text_ids.u32`, and optional debug prefix tensors; it no longer contains reference-generated `codes`, `gpt_latent`, `gpt_layer`, `vq2emb`, `s_infer`, or `length_regulator`. Text token ids can now be generated independently through `metal_indextts2.tools.generate_text_ids`, which loads only the IndexTTS2 tokenizer/normalizer and writes `mit2-text-ids`; `generate_gpt_frontend` and `synthesize_native_hot` both accept `--text-ids-file` to reuse that boundary. `metal_indextts2.tools.export_tokenizer_vocab` now exports `bpe.model` ids and scores to `pieces.tsv`; `mit2_runtime --export-text-ids-cjk` generates byte-exact `text_ids.u32` for focused CJK and limited ASCII/ASCII-punctuation fixtures (`你好`, `中文测试`, `A?`, `你好?`, `你好,`, `HELLO,你好 WORLD?`, `AI你好,INDEX?`, `METAL你好 WORLD.`, `你好, INDEX?`, `HELLO,INDEX?`, `HELLOWORLD你好`, `OPENAI你好`, `SUPERLONGWORD?`, `METALINDEXTTS`, `HelloWorld?`, `0123456789`, `123`, `abc123你好`, `2026年`, `3.14`, `10086`, `电话12345`, `A1`, `A123`, `AI2026`, `AB1C2`, `AB1C2?`, `AB1C2你好`, `AB1C2,你好`, `AB1C2 你好`, `AB1C2END你好`, `X1Y2你好`, `AB12`, `ABC123`, `OPEN123`, `AB20`, `AB21`, `AB101`, `AB120`, `AB100`, `TEST007`, `TEST010`, `TEST0`, `ABC123你好`, `ＡＢＣ１２３`, `ａｂｃ１２３你好`, `２０２６年`, `ＯＰＥＮ１２３`, `ＨｅｌｌｏＷｏｒｌｄ？`, `测试ＡＢ１２结束`, `１２`, `12`, `12结束`, `21结束`, `99结束`, `Ａ１`, `Ａ１２３`) from that exported vocabulary, including score-based pure-letter ASCII run splitting with a prefixed first piece or `▁` plus unprefixed fallback, score-based unprefixed ASCII splitting after punctuation, simple digit-run normalization, exact vocab-backed `alpha+first digit` mixed pieces, digit-after-single-alpha continuations after prior digit context, a conservative uppercase-alpha English number-word subpath that switches back to Chinese digit normalization when later text enters CJK context, fullwidth punctuation normalization, fullwidth alphanumeric normalization, and 1-2 digit Chinese integer forms. More complex mixed-script and punctuation forms still intentionally require full native `TextNormalizer`, because the Python frontend may emit exact mixed pieces or switch between English and Chinese number normalization by context. `--export-text-ids-cjk-segments` cuts those narrow native text ids at tokenizer-compatible punctuation boundaries within the token budget into per-segment `.u32` files, and `--native-cjk-text-ids` lets `generate_gpt_frontend`/`synthesize_native_hot` call those native text-id exports directly. This is an integrated narrow native text boundary, not a full native `TextNormalizer`/SentencePiece port. `--all-segments` now bridges the reference TextNormalizer/tokenizer segmentation for general longer input, or uses narrow native segment ids when combined with `--native-cjk-text-ids`; it synthesizes each segment through the same native hot path with per-segment deterministic seeds and concatenates PCM16 mono segment WAVs with configurable zero interval silence. It is not combined with `--text-ids-file` or `--noise-file` because those are single-segment boundaries. `--native-perceiver` can run `gpt.perceiver_encoder` in the native runtime from a PyTorch-generated Conformer context through resident-weight Metal linear/cross-attention/GEGLU/RMSNorm kernels. `--native-conformer-stack --native-perceiver` moves the hybrid frontend boundary further by exporting the six-layer Conformer stack natively from Python-prepared post-subsampling stack inputs through resident-weight Metal norm/projection kernels, then chaining that context into native Metal Perceiver export. `--native-frontend-tail` then calls `mit2_runtime --export-gpt-frontend-tail` to build `conds_latent`, `fake_inputs`, `inputs_embeds`, and `attention_mask` from native `speech_conditioning_latent`, native or PyTorch `emovec`, and `text_ids`, removing duration embedding lookup and `prepare_gpt_inputs` from Python. `--native-emovec` now consumes the full native `gpt.get_emovec` path by default, with emotion subsampling conv/projection, the four-layer resident-weight emotion Conformer stack, resident-weight emotion Perceiver, and final emovec projections running through resident Metal-backed kernels, and records `emovec_source: native_full_metal_subsampling_conformer_perceiver_linear`; `--native-emovec-input-tokens N` remains a truncated debug path and intentionally changes output when `N` is shorter than the full voice conditioning sequence. `--native-subsampling` exposes a runtime path through `mit2_runtime --export-gpt-subsampling`; it uses resident Metal `subsampling_conv2d_relu_flat_f32_resident` for conv/ReLU/PyTorch flatten and resident Metal `linear_rows_f32_resident` for the 261632-wide projection, with manifests recording `subsampling_source: native_metal_resident_conv_linear`. The resulting seeded short WAV/codes/noise are byte-exact with the prior PyTorch-tail frontend path when full PyTorch emovec is used, while manifests/reports record `emovec_source`, `subsampling_source`, `conformer_source: native_stack_metal_resident_attn_core_conv_ff`, `perceiver_source: native_metal_resident_linear_cross_attn_geglu_rmsnorm`, `frontend_tail_source`, and `text_ids_source`. The fixed-noise reproduction path uses `--synthesize-hot-inputs`; the default path now uses `--synthesize-hot-inputs-seeded`, which runs native cached GPT decode first, derives the real code/condition length, writes deterministic native noise from `seed`/`temperature`, then runs GPT latent/semantic/length-regulator condition export, S2Mel, BigVGAN, and PCM16 WAV writing from explicit tensors. `--shared-runtime-stages` exposes the shared-bundle greedy seeded runtime as an opt-in Python CLI and benchmark path; it reports the same native phase timing/resource counters as the default staged path but remains an allocator-planning experiment because mapped bundle pages stay resident across stages and can raise peak RSS. The CLI can reuse an existing native `--voice-bundle`, automatically convert `index-tts/voices/<voice>.pt` into a native bundle under the work directory, or call `clone_voice` via `--clone-audio` and then convert the cloned profile before synthesis. It no longer runs `generate_gpt_golden`, `generate_s2mel_golden`, multiple runtime subprocesses, or `--synthesize-hot-native-golden` for the normal path. The smoke artifacts `artifacts/native_text_native_emovec_tail_cli.wav`, `artifacts/native_text_native_emovec_metal_linear_tail_cli.wav`, `artifacts/native_text_native_tail_cli.wav`, `artifacts/native_text_native_stack_metal_attn_core_conv_ff_tail_cli.wav`, `artifacts/native_text_native_subsampling_metal_conv_linear_cli.wav`, `artifacts/native_text_native_conformer_perceiver_cli.wav`, `artifacts/native_text_native_perceiver_cli.wav`, `artifacts/native_text_native_ids_cli.wav`, `artifacts/native_text_ids_cli.wav`, `artifacts/native_text_frontend_cli.wav`, `artifacts/native_hot_seeded_default_a.wav`, `artifacts/native_text_single_cli.wav`, and `artifacts/native_hot_inputs.wav` are validated as PCM16 mono 22050 Hz with native codes matching the generated golden codes where the frontend boundary is byte-equivalent; seeded native runs are byte-stable for WAV/noise/codes/condition sidecars, and the fixed-noise raw CLI and one-command native raw-input outputs are byte-exact with the earlier native condition WAV smoke. The new Metal-Perceiver plus batched-DiT direct native smoke `artifacts/native_hot_seeded_perceiver_metal_batched_dit.wav` completes 25-step seeded synthesis in 37.10s with 2.45GB native peak RSS and is byte-exact with the prior deterministic WAV/codes/noise. The full native-emovec hybrid CLI smoke `artifacts/native_text_native_full_emovec_metal_perceiver_batched_cli.wav` completes in 57.99s with runtime peak RSS about 2.45GB and whole-process peak RSS about 7.73GB; it predicts codes `[4039, 6947, 2248]`, records `emovec_source: native_full_metal_subsampling_conformer_perceiver_linear`, and is byte-exact with the direct native Metal-Perceiver plus batched-DiT WAV/codes/noise. The native-CJK-id full-emovec CLI smoke `artifacts/native_text_native_cjk_ids_full_emovec_cli.wav` completes in 57.65s, records `text_ids_source: native_cjk`, and is byte-exact with the full native-emovec WAV/codes/noise. The mixed alpha+digit CJK-context smoke `artifacts/native_text_native_mixed_alnum_context_validated_smoke.wav` runs `AB1C2你好` through `--native-cjk-text-ids` plus source and native voice-contract validation, predicts code `[4588]`, writes SHA-256 `3f9f94900a30110e02f5b399ce69fa2f4ac0f0a9968e875f0538e746135dd442`, and its generated frontend text ids are byte-exact with `generate_text_ids`. The reference-tokenizer all-segments smoke `artifacts/native_text_all_segments_two_smoke.wav` splits `你好你好` into two segments, completes in 90.31s, writes report format `mit2-native-hot-multisegment-synthesis-report`, and validates exact PCM16 concatenation of segment0 + 200 ms zero silence + segment1. The native-CJK-segment smoke `artifacts/native_text_all_segments_native_cjk_smoke.wav` splits the same text via `--export-text-ids-cjk-segments`, completes in 88.94s with `steps=1`, records `text_ids_source: native_cjk_segments`, predicts segment codes `[[4039], [4039]]`, and writes SHA-256 `1c5a327e6e65904c2e157838a63c90d83cb8a2e11ccc93f662d6a08918c0d498`. The clone-audio smoke `artifacts/native_clone_full_emovec_metal_perceiver_batched_cli.wav` creates and converts `mit2_clone_smoke_full_emovec`, completes in 77.38s with runtime peak RSS about 2.45GB and whole-process peak RSS about 8.61GB, and writes the same deterministic short-fixture WAV hash. This is the current text-to-WAV bridge, not the final all-native tokenizer/normalizer, clone-time encoder stack, or fully Metal-optimized GPT conditioning front-end.
- Native text-id export also now normalizes common fullwidth Chinese punctuation (`，。．！？；：（）“”……`) to the same ASCII pieces as the reference tokenizer, with byte-exact checks for `你好，世界。`, `你好．世界`, `３．１４`, `你好.14`, `你好！`, `你好？`, `测试：ABC123；结束`, `（测试）`, `你好、世界`, `“你好”`, and `你好……结束`.
- Native text-id export now matches the reference V-version tokenization quirk for `V2`-style inputs: `V2你好`, `v2你好`, `V3你好`, and `V20你好` preserve the first version digit as the reference unknown/raw digit token while keeping later digits in the CJK digit path; the focused runtime gate `--test-text-ids-cjk-version-tokenizer` also keeps `X2你好` on the existing CJK digit path.
- Native text-id export now matches focused slash punctuation forms by mapping `/` and fullwidth `／` through the reference unknown-id path with context-aware `▁` insertion; `--test-text-ids-cjk-slash-tokenizer` covers `A/B测试`, `你好/世界`, `路径/home`, `１／２`, `比例１／２`, `１／０２`, and `１／２／３`.
- Native text-id export now matches focused temperature-unit forms: `℃`/`℉` and CJK-context `°C`/`°F` normalize to `摄氏度`/`华氏度`, while pure English `25°C`/`25°F` follows the reference English `DEGREES CELSIUS/FAHRENHEIT` piece path; `--test-text-ids-cjk-temperature-tokenizer` covers `温度25°C`, `温度25℃`, `温度25°F`, `25℃`, `25℉`, `2℃`, `02℃`, `温度2°C`, `25°C`, and `25°F`, including the reference `两` reading for focused two-degree forms. Focused ASCII unary-minus CJK numeric contexts now also match the reference `负...` pieces for `-5`, `-5℃`, `今天气温-5℃`, `-5度`, `温度-5°C`, and `温度-5°F`, while pure English unit forms such as `-5°C`, `-5kg`, and ASCII-word contexts such as `A-5` remain outside the narrow native bridge rather than producing mismatched ids.
- Native text-id export now matches focused comparison and multiplication forms: `1=1`, `12=34`, `１=２`, `1<2`, `1>2`, `1<=2`, `1>=2`, `1≤2`, `1≥2`, `3×4`, `３×４`, and CJK-prefixed variants normalize to the reference `等于`/`小于`/`大于`/`乘` pieces through `--test-text-ids-cjk-operator-tokenizer`. Focused degree RHS forms such as `1=2℃`, `1=2℉`, `1=2度`, `1=25℃`, `温度1=2℃`, `1<=2℃`, `1≥2℃`, `3×2℃`, `3×25℃`, and focused three-digit plus exact-`1000℃` RHS forms such as `1=102℃`, `1=112℃`, `1=120℃`, `1=302℃`, `1=1000℃`, and `3×1000℃` now also match the reference. Fullwidth comparison operators, asterisk multiplication, ASCII-word operator contexts, bare `<`/`>` with `℃`, English `°C`/`°F` RHS, and broader four-plus-digit degree RHS numbers such as `1=2026℃` remain rejected until the full native TextNormalizer/SentencePiece path owns those mixed contexts.
- Native text-id export now also matches focused operator measure RHS forms such as `1=101米`, `1=102米`, `1=112米`, `1=120米`, `1=1000米`, `3×125米`, `3×202米`, and `3×1000米` through `--test-text-ids-cjk-measure-tokenizer`, preserving the same operator-specific reference pieces as the Python `TextNormalizer`; other four-plus-digit measure RHS forms such as `1=2026米` and `3×2026米` remain rejected until the full native TextNormalizer/SentencePiece path owns those broader mixed contexts.
- Native text-id export now also matches focused CJK decimal measure forms such as `2.5米`, `2.05米`, `1.0米`, `2.5公里`, `2.5度`, and CJK-prefixed `长度2.5米`, keeping the reference `点` plus fractional digit pieces and consuming the CJK unit suffix without falling back to Python.
- Native text-id export now also matches focused ASCII/fullwidth-suffix unit
  expansions for explicit `kgs`, fullwidth `ＫＧ` suffixes, and
  non-leading-zero five-digit unit numbers, including `2kgs`, `2.5kgs`,
  `2ＫＧ`, `2.5ＫＧ`, `10001kg`, and `10001.5kg`, while keeping long
  leading-zero forms such as `01000kg`, CJK-surrounded unit snippets, and
  fullwidth digit/fraction mixtures delegated to the full native
  `TextNormalizer`/SentencePiece path.
- Native text-id export now matches focused ASCII plus numeric forms:
  `1+2`, `12+34`, `完成1+2`, and `1++2` emit the reference `加`/`正`
  pieces; `+5`, `+2℃`, `+5℃`, `温度+2°C`, `温度+5℃`, `温度+5°C`, `温度+5°F`,
  `1+2度`, and `1+2℃`
  emit the reference `正...` pieces. Pure English or unknown plus contexts
  such as `+5°C`, `1+2°C`, `A+1`, and `C++` remain outside the narrow native
  bridge rather than producing mismatched ids.
- Native text-id export now matches focused ASCII-separated date forms:
  `YYYY/MM/DD` and `YYYY-MM-DD` normalize to 年/月/日 with byte-exact checks for
  `2026/06/05`, `日期2026/06/05`, `2026-06-05`, and `日期2026-06-05`; already
  normalized Chinese-unit dates such as `2026年06月05日` keep their existing
  reference behavior.
- Native text-id export now matches focused standalone 4-digit integer forms:
  `2026`, `２０２６`, `１０００`, `１００１`, `１０１０`, `１１００`,
  `１２００`, and `２２００` normalize to the reference Chinese cardinal
  pieces, including `两千`/`两百`, while `2026年` and `２０２６年` remain on
  the reference per-digit year path. Fullwidth separator date-like forms such
  as `２０２６－０６－０５`, `２０２６／０６／０５`,
  `日期２０２６－０６－０５`, and `日期２０２６／０６／０５` preserve
  punctuation separators and now match the reference first-run cardinal
  normalization.
- Native text-id export now matches focused ASCII time forms:
  `H:MM`, `HH:MM`, and optional `:SS` normalize to 点/分/秒 with byte-exact checks
  for `12:30`, `12：30`, `时间12:30`, `12:30:45`, `时间12:30:45`, `9:05`,
  `09:05`, `上午9:05`, `12:00`, and `12:00:05`; invalid-minute and
  fullwidth-digit colon forms remain outside the narrow native CJK bridge.
- Native text-id export now matches focused currency-prefix forms:
  `￥`/`¥` amounts normalize to 元 and `$` amounts normalize to 美元, with
  byte-exact checks for `￥12`, `¥12`, `￥12.5`, `￥0.5`, `￥100`, `￥999`,
  `￥1000`, `￥200`, `￥2000`, `￥2200`, `￥10000`, `￥10001`, `￥10010`,
  `￥20000`, `￥10000.5`, `￥100.05`, `￥１２`,
  `￥１２.５`, `￥０.５`, `￥100.０５`, `$12`, `$12.5`, `$0.5`,
  `$100`, `$1000`, `$2000`, `$10000`, `$10000.5`, `$100.05`, `$１２`, and
  `$１０００`. Focused leading-zero currency amounts with a single leading zero
  and nonzero second digit, such as `￥01`, `￥05`, `￥012`, `￥０１２`,
  `￥０１０`, `￥01000`, `￥01000.5`, `￥０１２.３４`, `¥012`, `$012`,
  `$01000`, and `$01000.5`, now follow the reference
  `零元...` or `零美元...` pieces. Focused multi-leading-zero yen forms such
  as `￥001`, `￥００１`, `￥０００`, `¥001`, and `¥００１` now follow the
  reference raw `¥` unknown-id plus per-digit pieces. Focused multi-leading-zero
  dollar forms such as `$001`, `$００１`, and `$０００` follow the reference
  `▁.` plus per-digit pieces; broader currency amounts such as `￥100000`,
  `$100000`, and long leading-zero forms such as `￥010000` and `$010000`
  remain outside the narrow native CJK bridge.
- Native text-id export now matches focused hyphenated phone/grouped-number
  forms: ASCII `3-4-4`, `3-3-4`, and `3-8` digit groupings normalize to
  per-digit Chinese number pieces with byte-exact checks for
  `138-0013-8000`, `电话138-0013-8000`, `010-12345678`,
  `400-800-1234`, and `１３８-００１３-８０００`; focused `3－4－4`
  fullwidth-hyphen groupings such as `138－0013－8000`,
  `电话138－0013－8000`, and `１３８－００１３－８０００` follow the reference
  `幺...-...-八千` piece path. Focused short hyphenated arithmetic-like forms
  such as `1-2`, `12-34`, `比例12-34`, and `完成12-34` now follow the reference
  `减` pieces.
- Native text-id export now matches focused ASCII digit-ratio forms: after
  valid time forms have already been handled, `N:N` forms normalize the ASCII
  colon to `比`, with byte-exact checks for `1:2`, `比例1:2`, `12:60`, and
  `１２:３０`. Fullwidth-colon digit separators such as `1：2`, `比例1：2`,
  `１：２`, and `12：60` now follow the reference comma-like pieces. Focused
  alpha-colon digit forms such as `A:1`, `A:01`, `AB:12`, `ABC:123`,
  `比例A:1`, `比例AB:12`, `A：1`, and `A:1你好` now match the reference
  English-number or CJK-number comma paths, while longer alpha-colon digit runs
  remain outside the narrow native CJK bridge.
- Native text-id export now matches focused ASCII `No.` numbering forms:
  unspaced `No.1`/`NO.1`/`No.2026` follow the reference `NUMBER ...` pieces,
  lowercase unspaced `no.2026` keeps the reference `NO POINT ...` path, spaced
  pure-ASCII forms such as `No. 1`, `No. 001`, `No. 2026`, `No. 10000`, `No. 10001`, and
  `no. 2026` follow the reference spaced-number pieces, and CJK-context forms
  such as `No.1你好`, `编号No.1`, and `NO.12完成` keep the reference `NO . 中文数字`
  path. Focused 5-digit unspaced forms such as `No.10000`, `No.10001`,
  `No.10100`, `No.10101`, `No.11000`, `No.99999`, `no.10000`, and `no.10001`
  also match the reference pieces. Out-of-scope forms
  such as `No.100000`, `No. 100000`, and `no.100000` remain rejected until the
  full native TextNormalizer owns those rules.
- Native text-id export now matches focused ASCII slash numeric forms:
  month/day shapes such as `01/02`, `09/5`, `10/2`, and `12/03` normalize to
  月/日 pieces, while no-leading-zero fractions such as `1/2`, `12/34`,
  `1/20`, `9/5`, and `比例1/2` normalize to the reference `分之` pieces.
  Focused leading-zero denominator forms such as `1/02`, `2/03`, `9/05`,
  `1/002`, `2/003`, `比例1/02`, and `1/02完成` now follow the reference
  `零分之...` pieces. Focused single-digit chained slash forms such as
  `1/2/3`, `比例1/2/3`, and `1/2/3完成` now follow the reference
  `一 / 三分之二` style pieces. Fullwidth slash punctuation forms such as
  `１／２`, `比例１／２`, `１／０２`, and `１／２／３` now follow the reference
  unknown-id slash path; broader chained slash and general number normalization
  beyond the focused 4-digit integer path remain outside the narrow native CJK
  bridge.
- Native text-id punctuation handling now uses one context-aware ASCII punctuation helper for both ASCII and normalized Unicode punctuation, fixing the CJK-context hyphen boundary (`你好-世界` -> `▁ -`) while keeping ASCII-word context (`A-B` -> `-`). It also normalizes reference-supported dash/bracket-like forms with byte-exact checks for `A-B`, `你好-世界`, `A—B`, `A–B`, `你好–世界`, `你好—世界`, `Ａ－Ｂ`, `[测试]`, `【测试】`, `《测试》`, and `(测试)`; en dash follows the reference unknown-id punctuation path instead of being collapsed to `-`.
- Native text-id export now matches focused fullwidth/Unicode dash numeric punctuation forms. `１－２`, `１２－３４`, `完成１２－３４`, and `１—２` follow the reference `-` punctuation pieces instead of the ASCII hyphen subtraction normalizer, while `１–２` preserves the reference unknown-id en dash path.
- Native text-id export now matches additional fullwidth-colon digit punctuation forms: `１２：３０：４５` emits comma-like pieces for both separators, and fullwidth alpha-colon forms such as `Ａ：１` and `比例Ａ：１` follow the reference comma plus CJK-number pieces.
- Native text-id export now matches focused halfwidth colon/semicolon CJK
  punctuation forms: `他说:你好`, `他说;你好`, `你好:世界`, `A:你好`,
  `A;你好`, and `hello:你好` map to the reference comma-like pieces, while
  existing specialized `1:2`, `12:30`, and `A:1` normalizers keep priority.
  URL/email/operator-like ASCII forms remain outside the narrow native bridge.
- Native text-id export now matches focused ASCII double-quote and quote-mark
  spacing forms: `他说:"你好"`, `他说:"你好!"`, `他说:"你好！"`,
  `他说"你好"`, `A"B`, `"你好"`, `你好"世界`, `他说：“你好！”`, and
  `他说“你好”` use the reference apostrophe pieces, including no extra `▁`
  after comma-like punctuation, before a closing quote after `!`, or before
  comma/period punctuation that follows a closing quote in forms such as
  `“你好”，他说` and `他说：“你好”。`.
- Native text-id export now matches focused ASCII ellipsis forms: `你好...世界`,
  `hello...world`, `3...4`, `...你好`, and `你好...` collapse to the same
  reference `...` pieces as `…`/`……`, while preserving the surrounding CJK,
  ASCII, or digit tokenization context.
- Native text-id export now supports the reference halfwidth-percent normalizer for focused 0-999 plus non-leading-zero 4-digit numeric forms, including integer and decimal percentages, with byte-exact checks for `5%`, `50%`, `100%`, `101%`, `1000%`, `2026%`, `12%`, `12.5%`, `1000.5%`, `0.5%`, `完成50%`, `完成1000%`, `50%完成`, and `１００%`; focused fullwidth percent forms such as `５０％` and `100％完成` now match the reference unknown `%` id path generated with `--allow-unknown`.
- The expanded native-CJK text-id bridge now has an end-to-end cached-voice smoke on `测试：ABC123；结束`, covering fullwidth punctuation plus uppercase-alpha digit normalization inside `synthesize_native_hot --native-cjk-text-ids --native-subsampling --native-conformer-stack --native-perceiver --native-frontend-tail --native-emovec`. The generated native frontend `text_ids.u32` is byte-exact with `generate_text_ids`, the run predicts code `[4588]`, writes PCM16 mono 22050 Hz output `artifacts/native_text_native_expanded_ids_smoke.wav` with SHA-256 `b120e0f7eb50244293eab954c092eea25c1db40900a749153ccd3ac3a422c069`, and records native peak RSS about 4.39GB for this 1-code/1-step smoke.
- The same expanded text now passes the native segmented TTS bridge with `--all-segments --native-cjk-text-ids --max-text-tokens-per-segment 12`, cutting three reference-matching native segment files (`测试：`, `ABC123；`, `结束`), predicting segment codes `[4588]`, `[1707]`, and `[4588]`, and writing concatenated PCM16 mono 22050 Hz output `artifacts/native_text_all_segments_expanded_cjk_smoke.wav` with SHA-256 `9f608d0923bbc5d2d172f5d45fe2da93a7ca79db848b9bcf6b2afa47fbcfe87d`.
- The narrow hand-rolled `tokenize_cjk_text` state machine is no longer the server tokenizer: it threw `native CJK text-id export currently supports CJK, whitespace, and limited ASCII pieces only` on any real-world input outside its curated fixtures. The server synthesis path (`synthesis_pipeline.cpp`, `product_and_readiness.cpp`) now calls `tokenize_text_full(tokenizer_dir, text)` → `mit2::TextFrontend` (`runtime/impl/text_frontend.cpp`, `runtime/include/mit2/text_frontend.hpp`), a faithful native port of the reference Python frontend (`indextts/utils/front.py` TextNormalizer+TextTokenizer, `wetext/utils.py`, `wetext/token_parser.py`). It links **SentencePiece** (`bpe.model`) and **kaldifst/OpenFST** (applying the vendored wetext `.fst` TN grammars in `bin/tokenizer/fsts/{zh,en}/tn/{tagger,verbalizer}.fst`), both pulled in via CMake `FetchContent` (kaldifst `v1.7.17`, sentencepiece `v0.2.0`). The pipeline is `normalize` (contraction → pinyin/name protect → OpenFST tagger→TokenParser reorder→verbalizer when the text contains a digit → punctuation char-map) → `tokenize_by_CJK_char` → SentencePiece encode, with the normalizer wrapped in try/except like the reference so grammar edge cases never abort synthesis. Output token ids are byte-identical to the reference; `tests/test_tokenizer_parity.py` gates 33 cases (the formerly-failing long mixed text, numbers/dates/percent/currency/time/fractions, English brand names + code/backticks, emails, names, jqx-pinyin) against committed golden ids in `tests/tok_corpus.expected.txt` (regenerate with `tests/ref_runner.py`). The narrow `tokenize_cjk_text` and its `--export-text-ids-cjk*` CLIs/tests are retained unchanged. This is the "Full native TextNormalizer/SentencePiece integration" previously listed as remaining for the cached-voice TTS path.

## Phase 13: Voice Profile Conversion

Status: partial

Deliverables:

- Convert existing `.pt` voice profiles to native binary format.
- Preserve:
  - voice name,
  - reference audio path metadata,
  - `spk_cond_emb`,
  - `s2mel_style`,
  - `s2mel_prompt`,
  - `mel`.

Validation:

- Native runtime loads profiles without Python/Torch.
- Tensor checksums match `.pt` source tensors after conversion.
- Cached profile TTS output matches reference.
- The hybrid TTS CLI now auto-converts a cached `.pt` voice profile when `--voice-bundle` is omitted, and the resulting native voice bundle has been used successfully by the native hot S2Mel/BigVGAN path.
- `metal_indextts2.tools.convert_voice --validate-source` now reloads the written native voice bundle and compares `spk_cond_emb`, `s2mel_style`, `s2mel_prompt`, and `mel` byte-exactly against the source `.pt` tensors after dtype conversion. The real `voice_a7bd52e4.pt` profile validates with all four tensors matching, and the resulting `artifacts/voice_a7bd52e4_validated_bundle` drives a native hot smoke that predicts code `[4039]` and writes PCM16 mono 22050 Hz output `artifacts/native_text_validated_voice_bundle_smoke.wav` with SHA-256 `69dcc4124aab3388c894e9fee95466d0703759eb4084f465d2b4074c548cfc92`.
- `synthesize_native_hot --validate-voice-source` now applies the same byte-exact source validation to auto-converted cached voices and `--clone-audio` voices before the native runtime is invoked. The cached auto-convert smoke with `--validate-voice-contract` writes `artifacts/native_text_auto_validated_voice_smoke.wav` with SHA-256 `1eb380f6718bcdf1d3254d9ede10b75018d23f3a43c634c122f267b85b07a566`; the clone-audio auto-convert smoke creates `mit2_clone_validated_voice_source`, validates the cloned `.pt` against the native bundle with all four tensors matching, predicts code `[4039]`, and writes `artifacts/native_clone_validated_voice_source_smoke.wav` with SHA-256 `1eb380f6718bcdf1d3254d9ede10b75018d23f3a43c634c122f267b85b07a566`.
- `mit2_runtime --inspect-voice-bundle` now provides a native-only voice bundle contract gate. It verifies required f32 `component=voice` tensors, shape/byte-count consistency, MIT2 header/alignment consistency, non-overlapping aligned payload ranges, full payload SHA-256, and matching `s2mel_prompt`/`mel` prompt token counts; both `artifacts/test_voice_bundle` and `artifacts/voice_a7bd52e4_validated_bundle` pass with `ok: true`, `spk_cond_tokens: 548`, and `prompt_tokens: 944`.
- `synthesize_native_hot --validate-voice-contract` now runs that native-only contract gate before synthesis for explicit, cached auto-converted, and clone-audio voice bundles, and records the resulting native report under `voice_contract`.
- `mit2_runtime --inspect-model-bundle` now provides a native-only model bundle contract gate. It verifies non-empty tensor shapes, dtype byte-count consistency, MIT2 header/alignment consistency, non-overlapping aligned payload ranges, full payload SHA-256, required component presence for `gpt`, `semantic_codec`, `s2mel`, and `bigvgan`, plus 24 required sentinel tensors spanning GPT embeddings/head, conditioning/emotion/perceiver paths, semantic codec codebook, S2Mel length-regulator/DiT/Wavenet paths, and BigVGAN front/body/post tensors. `artifacts/test_model_bundle` passes with `ok: true`, `weights_bytes: 5317937152`, `tensor_count: 2755`, `tensor_bytes: 5312776880`, `sha256_verified_count: 2755`, and `required_tensor_count: 24`.
- `synthesize_native_hot --validate-model-contract` now runs that native-only model contract gate before synthesis and records the resulting native report under `model_contract`. The 1-step native-hot smoke `artifacts/native_text_model_integrity_contract_smoke.wav` records both `model_contract.ok: true` and `voice_contract.ok: true`, predicts code `[4039]`, verifies 2755 model tensor SHA-256 payloads plus 4 voice tensor SHA-256 payloads, and writes SHA-256 `1eb380f6718bcdf1d3254d9ede10b75018d23f3a43c634c122f267b85b07a566`.

## Phase 14: Clone-Time Native Path

Status: partial

Deliverables:

- Port clone-time pieces after hot TTS is stable:
  - audio loading/resampling,
  - W2V-BERT feature path or compatible exported subset,
  - semantic model hidden-state extraction,
  - semantic codec quantize,
  - mel spectrogram,
  - CAMPPlus style encoder,
  - length regulator prompt condition.
- Lock the model-bundle side of the clone encoder boundary before implementing
  the encoder forwards. `mit2_tts --clone-encoder-model-readiness
  MODEL_BUNDLE_DIR` now validates the current bundle integrity and reports the
  required clone-time model contracts for CAMPPlus, W2V-BERT/semantic model,
  MaskGCT quantize, and the S2Mel length-regulator prompt-condition path.
  Current hot-path bundles expose the semantic-codec encoder/quantizer weights
  and the S2Mel prompt-condition contract, but still lack CAMPPlus and
  W2V-BERT/semantic model contracts, so they correctly remain
  `ready_native_clone_encoder_models=false`.
- Export CAMPPlus weights into the native bundle. `convert_model` now accepts
  `--campplus-checkpoint` and writes the funasr CAMPPlus state dict as
  `campplus.*` tensors with `component=campplus`; a real cached
  `campplus_cn_common.bin` conversion produced 937 tensors and is recognized by
  `mit2_tts --clone-encoder-model-readiness` as
  `has_campplus_model_contract=true` after 30 sentinel tensor checks across the
  FCM head, xvector TDNN, CAM dense blocks, transit layers,
  out-nonlinear/stats, and dense/style output. Native CAMPPlus
  fbank-to-`s2mel_style` replay is now complete.
- Export W2V-BERT weights and normalization stats into the native bundle.
  `convert_model` now accepts `--w2v-bert-dir` and `--w2v-stats`, loads
  `facebook/w2v-bert-2.0` through `Wav2Vec2BertModel.from_pretrained`, writes
  the state dict as `w2v_bert.*` tensors, and stores
  `wav2vec2bert_stats.pt` mean/variance as `w2v_bert.stats.mean` and
  `w2v_bert.stats.std`. `mit2_tts --clone-encoder-model-readiness` now checks
  412 f32 W2V-BERT sentinel tensors across feature projection, layer-0
  ffn1, attention/conv, ffn2, layer-0 final norm, layer-1 ffn1
  LayerNorm/intermediate/output dense, layer-1 self-attention Q/K/V/out
  projections plus attention LayerNorm, layer-1 convolution-module LayerNorm,
  layer-1 convolution pointwise GLU, depthwise conv, activation/projection
  residual, ffn2, final norm, layer-2 ffn1 LayerNorm/intermediate/output dense,
  layer-2 self-attention Q/K/V/out projections plus attention LayerNorm,
  layer-2 convolution-module LayerNorm, layer-2 convolution GLU, layer-2
  convolution depthwise/residual, layer-2 ffn2, layer-3 ffn1
  LayerNorm/intermediate/output dense plus self-attention Q/K/V projections,
  self-attention output projection, attention LayerNorm, convolution-module
  LayerNorm, convolution GLU, convolution depthwise/residual, layer-3 ffn2,
  layer-3 final norm, layer-4 ffn1 LayerNorm/intermediate/output dense,
  layer-4 self-attention Q/K/V and output projections, attention LayerNorm, convolution-module LayerNorm, convolution GLU, convolution depthwise/residual, ffn2, layer-5 ffn1, self-attention Q/K/V, attention context/projection/residual, attention LayerNorm, convolution-module LayerNorm, convolution GLU, causal depthwise convolution, convolution residual projection, ffn2 half-residual, layer-6 ffn1 half-residual, layer-6 self-attention Q/K/V/out projections, layer-6 attention LayerNorm, layer-6 convolution-module LayerNorm, layer-6 convolution GLU, layer-6 causal depthwise convolution, layer-6 convolution residual projection, layer-6 ffn2 half-residual, layer-7 ffn1 half-residual, layer-7 self-attention Q/K/V/out projections, layer-7 attention LayerNorm, layer-7 convolution-module LayerNorm, layer-7 convolution GLU, layer-7 causal depthwise convolution, layer-7 convolution residual projection, layer-7 ffn2 half-residual, layer-8 ffn1 half-residual, layer-8 self-attention Q/K/V/out projections, layer-8 attention LayerNorm, layer-8 convolution-module LayerNorm, layer-8 convolution GLU, layer-8 causal depthwise convolution, layer-8 convolution residual projection, layer-8 ffn2 half-residual, layer-9 ffn1 half-residual, layer-9 self-attention Q/K/V/out projections, layer-9 attention LayerNorm/convolution residual/ffn2 half-residual, layer-10 ffn1 half-residual, layer-10 self-attention Q/K/V/context/output projection/attention LayerNorm, layer-10 convolution-module LayerNorm/GLU/depthwise/residual, layer-10 ffn2 half-residual, layer-11 ffn1 half-residual, layer-11 self-attention Q/K/V/context/output projection/residual/attention LayerNorm, layer-11 convolution-module LayerNorm/GLU/depthwise convolution/residual, layer-11 ffn2 half-residual, layer-12 ffn1 half-residual, layer-12 self-attention Q/K/V/output projection/attention LayerNorm, layer-12 convolution-module LayerNorm/GLU/depthwise convolution/residual, layer-12 ffn2 half-residual, layer-13 ffn1 half-residual, layer-13 self-attention Q/K/V/context/output projection/residual/attention LayerNorm, layer-13 convolution-module LayerNorm/GLU/depthwise convolution/residual, layer-13 ffn2 half-residual, layer-14 ffn1 half-residual, layer-17 final norm,
  and mean/std stats before reporting
  `has_w2v_bert_model_contract=true`. This locks the model-bundle contract for
  the semantic encoder; the W2V-BERT feature extractor/forward that produces
  `spk_cond_emb` from clone audio remains to be implemented.
- Establish the W2V-BERT semantic-feature parity target.
  `generate_w2v_bert_golden` now consumes a native clone feature manifest,
  reads the 16 kHz f32 audio sidecar, runs `SeamlessM4TFeatureExtractor` and
  `Wav2Vec2BertModel` from the local `facebook/w2v-bert-2.0` directory, writes
  `w2v_input_features.f32`, `w2v_attention_mask.u32`,
  `w2v_hidden_state_17.f32`, and normalized `spk_cond_emb.f32`, and records a
  `mit2-w2v-bert-semantic-golden` manifest. This gives the future Metal
  W2V-BERT forward an exact input/output fixture while keeping
  `ready_native_w2v_bert_semantic_features=false`.
- Close the W2V-BERT stats-normalization sub-boundary after hidden-state
  extraction. `mit2_tts --clone-w2v-normalize MODEL_BUNDLE_DIR
  W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32` now validates the
  W2V-BERT model/stats contract, consumes `w2v_hidden_state_17.f32`
  `[1,tokens,1024]`, and writes normalized `spk_cond_emb.f32` with
  `w2v_bert.stats.mean/std`, preferring Metal and reporting `cpu_fallback` only
  when no Metal device is available. This leaves the native W2V-BERT feature
  extractor/encoder forward as the remaining semantic-feature step from clone
  audio to `w2v_hidden_state_17.f32`.
- Close the W2V-BERT feature-projection subgraph after HF feature extraction.
  `mit2_tts --clone-w2v-feature-project MODEL_BUNDLE_DIR
  W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32` now
  validates the W2V-BERT model contract, consumes `w2v_input_features.f32`
  `[1,tokens,160]`, and writes the feature-projection output
  `[1,tokens,1024]` through native LayerNorm plus 160-to-1024 projection,
  preferring resident Metal weights where available. The remaining W2V-BERT
  forward work is the encoder stack from feature projection to
  `hidden_state_17`.
- Enter the true layer-0 Conformer order with the first feed-forward LayerNorm.
  `mit2_tts --clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR
  W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32` now validates the
  expanded W2V-BERT contract including
  `encoder.layers.0.ffn1_layer_norm.{weight,bias}`, applies native LayerNorm
  rows with resident Metal parameters where available, and writes
  `[1,tokens,1024]`.
- Close the first layer-0 ffn1 dense boundary.
  `mit2_tts --clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR
  W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32` now validates the
  expanded W2V-BERT contract including
  `encoder.layers.0.ffn1.intermediate_dense.{weight,bias}`, applies the native
  1024-to-4096 projection with resident Metal weights where available, and
  writes `[1,tokens,4096]`.
- Close the first layer-0 ffn1 swish activation boundary.
  `mit2_tts --clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32
  W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32` now consumes the intermediate dense
  sidecar, applies swish/SILU through Metal when available, and writes
  `[1,tokens,4096]`.
- Close the first layer-0 ffn1 output projection boundary.
  `mit2_tts --clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR
  W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32` now validates the
  expanded W2V-BERT contract including
  `encoder.layers.0.ffn1.output_dense.{weight,bias}`, applies the native
  4096-to-1024 projection with resident Metal weights where available, and
  writes `[1,tokens,1024]`.
- Close the first layer-0 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32
  W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32` now consumes the
  feature-projection sidecar plus ffn1 output sidecar and writes
  `feature_projection + 0.5 * ffn1_output` as `[1,tokens,1024]`, using Metal
  `add_scaled_f32` when available.
- Enter W2V-BERT encoder layer 0 with native self-attention projections.
  `mit2_tts --clone-w2v-layer0-qkv MODEL_BUNDLE_DIR
  W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR` now consumes
  feature-projection output `[1,tokens,1024]`, runs the native
  `encoder.layers.0.self_attn.linear_q/k/v.weight` projections with resident
  Metal weights where available, and writes Q/K/V sidecars plus a manifest.
  The remaining layer-0 work is attention scores/context/output projection and
  the feed-forward/convolution stack, followed by the remaining encoder layers
  to `hidden_state_17`.
- Close the first layer-0 self-attention context boundary.
  `mit2_tts --clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32
  W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32` now consumes Q/K/V
  sidecars plus the attention mask and writes 16-head masked self-attention
  context `[1,tokens,1024]`, preferring the existing Metal cross-attention
  primitive when available. The layer-0 output projection and the remaining
  encoder stack to `hidden_state_17` remain.
- Close the layer-0 self-attention output projection boundary.
  `mit2_tts --clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR
  W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32` now validates the expanded
  W2V-BERT contract including `encoder.layers.0.self_attn.linear_out.weight`,
  applies the native 1024-to-1024 attention output projection with resident
  Metal weights where available, and writes `[1,tokens,1024]`.
- Close the layer-0 self-attention residual boundary.
  `mit2_tts --clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32
  W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32` now adds the
  feature-projection input and attention projection through Metal `add_f32`
  when available, writing `[1,tokens,1024]`.
- Close the layer-0 self-attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR
  W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32` now validates
  the expanded W2V-BERT contract including
  `encoder.layers.0.self_attn_layer_norm.{weight,bias}`, applies resident Metal
  LayerNorm rows where available, and writes `[1,tokens,1024]`.
- Close the layer-0 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR
  W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32` now validates the
  expanded W2V-BERT contract including
  `encoder.layers.0.conv_module.layer_norm.{weight,bias}`, applies resident
  Metal LayerNorm rows where available, and writes `[1,tokens,1024]`.
- Close the layer-0 convolution pointwise-GLU boundary.
  `mit2_tts --clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32
  W2V_TOKENS OUTPUT_CONV_GLU_F32` now validates
  `encoder.layers.0.conv_module.pointwise_conv1.{weight,bias}`, applies the
  1024-to-2048 projection with resident Metal weights where available, runs GLU
  through Metal `glu_split_f32`, and writes `[1,tokens,1024]`.
- Close the layer-0 convolution depthwise boundary.
  `mit2_tts --clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR
  W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32` now validates
  `encoder.layers.0.conv_module.depthwise_conv.{weight,bias}`, applies the
  31-tap causal depthwise Conv1d through resident Metal weights where
  available, and writes `[1,tokens,1024]`.
- Close the layer-0 convolution projection/residual boundary.
  `mit2_tts --clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR
  W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_CONV_RESIDUAL_F32` now validates
  `encoder.layers.0.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.0.conv_module.pointwise_conv2.{weight,bias}`, applies
  resident Metal LayerNorm rows, SiLU, 1024-to-1024 projection, and residual
  add where available, and writes `[1,tokens,1024]`.
- Close the layer-0 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR
  W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32` now validates
  `encoder.layers.0.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.0.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.0.ffn2.output_dense.{weight,bias}`, applies resident Metal
  LayerNorm rows, 1024-to-4096 projection, SiLU, 4096-to-1024 projection, and
  `conv_residual + 0.5 * ffn2_output` where available, and writes
  `[1,tokens,1024]`.
- Close the layer-0 final LayerNorm boundary.
  `mit2_tts --clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR
  W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32` now validates
  `encoder.layers.0.final_layer_norm.{weight,bias}`, applies resident Metal
  LayerNorm rows where available, and writes the layer-0 output
  `[1,tokens,1024]`.
- Enter encoder layer 1 with the first feed-forward LayerNorm.
  `mit2_tts --clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32
  W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32` now validates
  `encoder.layers.1.ffn1_layer_norm.{weight,bias}`, applies resident Metal
  LayerNorm rows where available, and writes `[1,tokens,1024]`.
- Close the first layer-1 ffn1 dense boundary.
  `mit2_tts --clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR
  W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32`
  now validates `encoder.layers.1.ffn1.intermediate_dense.{weight,bias}`,
  applies the native 1024-to-4096 projection with resident Metal weights where
  available, and writes `[1,tokens,4096]`.
- Close the first layer-1 ffn1 swish activation boundary.
  `mit2_tts --clone-w2v-layer1-ffn1-activate
  W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS
  OUTPUT_LAYER1_FFN1_ACTIVATED_F32` now consumes the intermediate dense
  sidecar, applies swish/SILU through Metal when available, and writes
  `[1,tokens,4096]`.
- Close the first layer-1 ffn1 output projection boundary.
  `mit2_tts --clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR
  W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32`
  now validates `encoder.layers.1.ffn1.output_dense.{weight,bias}`, applies
  the native 4096-to-1024 projection with resident Metal weights where
  available, and writes `[1,tokens,1024]`.
- Close the first layer-1 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32
  W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32`
  now consumes layer-0 output and layer-1 ffn1 output sidecars, applies
  `layer0 + 0.5 * layer1_ffn1_output` through Metal `add_scaled_f32` when
  available, and writes `[1,tokens,1024]`. The remaining W2V-BERT encoder work
  is layer-1 self-attention and the rest of the encoder stack to
  `hidden_state_17`.
- Start the first layer-1 self-attention boundary.
  `mit2_tts --clone-w2v-layer1-qkv MODEL_BUNDLE_DIR
  W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` now validates
  `encoder.layers.1.self_attn.linear_q/k/v.weight`, applies native resident
  Metal row-wise projections when available, and writes `w2v_layer1_q.f32`,
  `w2v_layer1_k.f32`, `w2v_layer1_v.f32`, and a manifest. The next native
  boundary is layer-1 attention context.
- Close the first layer-1 attention context boundary.
  `mit2_tts --clone-w2v-layer1-attention W2V_LAYER1_Q_F32
  W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER1_CONTEXT_F32` now consumes layer-1 Q/K/V sidecars, runs masked
  16-head attention through Metal when available, and writes `[1,tokens,1024]`.
  The remaining W2V-BERT encoder work is layer-1 attention output projection,
  residual/norm, and the rest of the encoder stack to `hidden_state_17`.
- Close the first layer-1 attention output projection boundary.
  `mit2_tts --clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32` now validates
  `encoder.layers.1.self_attn.linear_out.weight`, applies the native
  1024-to-1024 projection with resident Metal weights where available, and
  writes `[1,tokens,1024]`. The remaining W2V-BERT encoder work is layer-1
  attention residual/norm and the rest of the encoder stack to
  `hidden_state_17`.
- Close the first layer-1 attention residual boundary.
  `mit2_tts --clone-w2v-layer1-attention-residual
  W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32` now consumes the layer-1 ffn1 residual
  and layer-1 attention projection sidecars, applies an elementwise residual add
  through Metal `add_f32` when available, and writes `[1,tokens,1024]`.
- Close the first layer-1 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER1_ATTENTION_NORM_F32` now validates the expanded W2V-BERT
  contract including `encoder.layers.1.self_attn_layer_norm.{weight,bias}`,
  applies row LayerNorm with `eps=1e-5` through resident Metal when available,
  and writes `[1,tokens,1024]`.
- Start the first layer-1 convolution module boundary.
  `mit2_tts --clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32` now
  validates the expanded W2V-BERT contract including
  `encoder.layers.1.conv_module.layer_norm.{weight,bias}`, applies row
  LayerNorm with `eps=1e-5` through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the first layer-1 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32` now validates
  the expanded W2V-BERT contract including
  `encoder.layers.1.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native pointwise projection and GLU split through resident Metal when
  available, and writes `[1,tokens,1024]`.
- Close the first layer-1 convolution depthwise boundary.
  `mit2_tts --clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32` now
  validates the expanded W2V-BERT contract including
  `encoder.layers.1.conv_module.depthwise_conv.{weight,bias}`, applies causal
  depthwise Conv1d through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the first layer-1 convolution residual boundary.
  `mit2_tts --clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER1_CONV_RESIDUAL_F32` now validates the expanded W2V-BERT contract
  including `encoder.layers.1.conv_module.depthwise_layer_norm.{weight,bias}`
  and `encoder.layers.1.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the layer-1 second feed-forward and final-norm boundary.
  `mit2_tts --clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32`
  now validates the expanded W2V-BERT contract including
  `encoder.layers.1.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.1.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.1.ffn2.output_dense.{weight,bias}`, applies the second
  feed-forward half-residual through resident Metal when available, and writes
  `[1,tokens,1024]`. `mit2_tts --clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR
  W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32` then validates
  `encoder.layers.1.final_layer_norm.{weight,bias}`, applies row LayerNorm
  through resident Metal when available, and writes `[1,tokens,1024]`.
- Open the layer-2 ffn1 LayerNorm boundary.
  `mit2_tts --clone-w2v-layer2-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER1_F32
  W2V_TOKENS OUTPUT_LAYER2_FFN1_NORM_F32` now validates
  `encoder.layers.2.ffn1_layer_norm.{weight,bias}`, applies row LayerNorm
  through resident Metal when available, and writes `[1,tokens,1024]`.
- Open the layer-2 ffn1 intermediate dense boundary.
  `mit2_tts --clone-w2v-layer2-ffn1-intermediate MODEL_BUNDLE_DIR
  W2V_LAYER2_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_INTERMEDIATE_F32`
  now validates `encoder.layers.2.ffn1.intermediate_dense.{weight,bias}`,
  applies the 1024-to-4096 projection through resident Metal when available,
  and writes `[1,tokens,4096]`.
- Close the layer-2 ffn1 swish activation boundary.
  `mit2_tts --clone-w2v-layer2-ffn1-activate
  W2V_LAYER2_FFN1_INTERMEDIATE_F32 W2V_TOKENS
  OUTPUT_LAYER2_FFN1_ACTIVATED_F32` now applies native swish/SiLU through Metal
  when available and writes `[1,tokens,4096]`.
- Close the layer-2 ffn1 output projection boundary.
  `mit2_tts --clone-w2v-layer2-ffn1-output MODEL_BUNDLE_DIR
  W2V_LAYER2_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_OUTPUT_F32`
  now validates `encoder.layers.2.ffn1.output_dense.{weight,bias}`, applies the
  4096-to-1024 projection through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the layer-2 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer2-ffn1-residual W2V_LAYER1_F32
  W2V_LAYER2_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_RESIDUAL_F32`
  now applies the Conformer half-residual `layer1 + 0.5 *
  layer2_ffn1_output` through Metal when available and writes
  `[1,tokens,1024]`.
- Close the layer-2 self-attention Q/K/V projection boundary.
  `mit2_tts --clone-w2v-layer2-qkv MODEL_BUNDLE_DIR
  W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` now validates
  `encoder.layers.2.self_attn.linear_{q,k,v}.weight`, applies the three
  1024-to-1024 projections through resident Metal when available, and writes
  `w2v_layer2_q/k/v.f32` plus a manifest.
- Close the layer-2 self-attention context boundary.
  `mit2_tts --clone-w2v-layer2-attention W2V_LAYER2_Q_F32 W2V_LAYER2_K_F32
  W2V_LAYER2_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER2_CONTEXT_F32` now consumes layer-2 Q/K/V sidecars plus the W2V
  attention mask, runs masked 16-head attention through Metal when available,
  and writes `[1,tokens,1024]`.
- Close the layer-2 attention output projection boundary.
  `mit2_tts --clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32` validates
  `encoder.layers.2.self_attn.linear_out.weight`, applies the native projection
  through resident Metal when available, and writes `[1,tokens,1024]`.
- Close the layer-2 attention residual boundary.
  `mit2_tts --clone-w2v-layer2-attention-residual
  W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32` adds the projected attention sidecar back
  to the layer-2 ffn1 residual sidecar and writes `[1,tokens,1024]`.
- Close the layer-2 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32`
  validates `encoder.layers.2.self_attn_layer_norm.{weight,bias}`, applies row
  LayerNorm through resident Metal when available, and writes `[1,tokens,1024]`.
- Close the layer-2 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32`
  validates `encoder.layers.2.conv_module.layer_norm.{weight,bias}`, applies
  row LayerNorm through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the layer-2 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32`
  validates `encoder.layers.2.conv_module.pointwise_conv1.{weight,bias}`,
  applies the native pointwise projection plus GLU split through resident Metal
  when available, and writes `[1,tokens,1024]`.
- Close the layer-2 convolution depthwise boundary.
  `mit2_tts --clone-w2v-layer2-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER2_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_DEPTHWISE_F32`
  validates `encoder.layers.2.conv_module.depthwise_conv.{weight,bias}`,
  applies causal depthwise Conv1d through resident Metal when available, and
  writes `[1,tokens,1024]`.
- Close the layer-2 convolution residual boundary.
  `mit2_tts --clone-w2v-layer2-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER2_ATTENTION_NORM_F32 W2V_LAYER2_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER2_CONV_RESIDUAL_F32` validates
  `encoder.layers.2.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.2.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the layer-2 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer2-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER2_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_FFN2_RESIDUAL_F32`
  validates `encoder.layers.2.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.2.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.2.ffn2.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the 0.5 residual through resident Metal when available,
  and writes `[1,tokens,1024]`.
- Close the layer-3 ffn1 LayerNorm boundary.
  `mit2_tts --clone-w2v-layer3-ffn1-norm MODEL_BUNDLE_DIR
  W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_NORM_F32`
  validates `encoder.layers.3.ffn1_layer_norm.{weight,bias}`, applies row
  LayerNorm through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the layer-3 ffn1 intermediate dense boundary.
  `mit2_tts --clone-w2v-layer3-ffn1-intermediate MODEL_BUNDLE_DIR
  W2V_LAYER3_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_INTERMEDIATE_F32`
  validates the expanded W2V-BERT contract including
  `encoder.layers.3.ffn1.intermediate_dense.{weight,bias}`, consumes the
  layer-3 ffn1 norm sidecar `[1,tokens,1024]`, applies the native 1024-to-4096
  projection through resident Metal when available, and writes
  `[1,tokens,4096]`.
- Close the layer-3 ffn1 swish activation boundary.
  `mit2_tts --clone-w2v-layer3-ffn1-activate
  W2V_LAYER3_FFN1_INTERMEDIATE_F32 W2V_TOKENS
  OUTPUT_LAYER3_FFN1_ACTIVATED_F32` consumes the layer-3 ffn1 intermediate
  sidecar `[1,tokens,4096]`, applies native swish/SILU through Metal when
  available, and writes `[1,tokens,4096]`.
- Close the layer-3 ffn1 output projection boundary.
  `mit2_tts --clone-w2v-layer3-ffn1-output MODEL_BUNDLE_DIR
  W2V_LAYER3_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_OUTPUT_F32`
  validates the expanded W2V-BERT contract including
  `encoder.layers.3.ffn1.output_dense.{weight,bias}`, consumes the layer-3
  ffn1 activated sidecar `[1,tokens,4096]`, applies the native 4096-to-1024
  projection through resident Metal when available, and writes
  `[1,tokens,1024]`.
- Close the layer-3 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer3-ffn1-residual W2V_LAYER2_FFN2_RESIDUAL_F32
  W2V_LAYER3_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_RESIDUAL_F32`
  consumes the layer-2 ffn2 residual and layer-3 ffn1 output sidecars
  `[1,tokens,1024]`, applies `layer2 + 0.5 * layer3_ffn1_output` through Metal
  when available, and writes `[1,tokens,1024]`.
- Enter W2V-BERT encoder layer 3 with native self-attention projections.
  `mit2_tts --clone-w2v-layer3-qkv MODEL_BUNDLE_DIR
  W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` validates
  `encoder.layers.3.self_attn.linear_{q,k,v}.weight`, consumes layer-3 ffn1
  residual `[1,tokens,1024]`, applies the three native projections through
  resident Metal when available, and writes `w2v_layer3_q.f32`,
  `w2v_layer3_k.f32`, `w2v_layer3_v.f32`, and a sidecar manifest.
- Close the W2V-BERT layer-3 self-attention context boundary.
  `mit2_tts --clone-w2v-layer3-attention W2V_LAYER3_Q_F32 W2V_LAYER3_K_F32
  W2V_LAYER3_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER3_CONTEXT_F32` consumes `w2v_layer3_q/k/v.f32` and the W2V
  attention mask, runs masked 16-head attention through Metal when available,
  writes `[1,tokens,1024]`, and leaves layer-3 attention output
  projection/residual/norm plus the rest of layer 3 and encoder layers 4-16 as
  the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 attention output projection boundary.
  `mit2_tts --clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32` validates
  `encoder.layers.3.self_attn.linear_out.weight`, consumes layer-3 attention
  context `[1,tokens,1024]`, applies the native projection with resident Metal
  weights where available, writes `[1,tokens,1024]`, and leaves layer-3
  attention residual/norm, convolution, ffn2, and encoder layers 4-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 attention residual boundary.
  `mit2_tts --clone-w2v-layer3-attention-residual
  W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32` adds the projected attention sidecar
  back to the layer-3 ffn1 residual through Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-3 attention norm, convolution, ffn2, and
  encoder layers 4-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER3_ATTENTION_NORM_F32` validates
  `encoder.layers.3.self_attn_layer_norm.{weight,bias}`, applies row LayerNorm
  through resident Metal when available, writes `[1,tokens,1024]`, and leaves
  layer-3 convolution, ffn2, and encoder layers 4-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32`
  validates `encoder.layers.3.conv_module.layer_norm.{weight,bias}`, applies
  row LayerNorm through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-3 convolution GLU/depthwise/residual, ffn2, and encoder
  layers 4-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32` validates
  `encoder.layers.3.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native 1024-to-2048 projection plus GLU split through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-3 convolution
  depthwise/residual, ffn2, and encoder layers 4-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 convolution depthwise boundary.
  `mit2_tts --clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32`
  validates `encoder.layers.3.conv_module.depthwise_conv.{weight,bias}`,
  applies causal depthwise Conv1d through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-3 convolution residual, ffn2, and encoder
  layers 4-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 convolution residual boundary.
  `mit2_tts --clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER3_CONV_RESIDUAL_F32` validates
  `encoder.layers.3.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.3.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-3 ffn2 plus encoder layers 4-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32`
  validates `encoder.layers.3.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.3.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.3.ffn2.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the 0.5 residual through resident Metal when available,
  writes `[1,tokens,1024]`, and leaves encoder layers 4-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-3 final LayerNorm boundary.
  `mit2_tts --clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR
  W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32` validates
  `encoder.layers.3.final_layer_norm.{weight,bias}`, applies row LayerNorm
  through resident Metal when available, writes `[1,tokens,1024]`, and leaves
  encoder layers 4-16 as the remaining native W2V-BERT encoder gap.
- Enter W2V-BERT encoder layer 4 with the ffn1 LayerNorm boundary.
  `mit2_tts --clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32
  W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32` validates
  `encoder.layers.4.ffn1_layer_norm.{weight,bias}`, applies row LayerNorm
  through resident Metal when available, writes `[1,tokens,1024]`, and leaves
  layer-4 ffn1 dense/swish/output plus encoder layers 5-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 ffn1 intermediate dense boundary.
  `mit2_tts --clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR
  W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32`
  validates `encoder.layers.4.ffn1.intermediate_dense.{weight,bias}`, applies
  the native 1024-to-4096 projection through resident Metal when available,
  writes `[1,tokens,4096]`, and leaves layer-4 ffn1 swish/output plus encoder
  layers 5-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 ffn1 activation boundary.
  `mit2_tts --clone-w2v-layer4-ffn1-activate
  W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS
  OUTPUT_LAYER4_FFN1_ACTIVATED_F32` consumes the layer-4 ffn1 intermediate
  `[1,tokens,4096]` sidecar, applies native SiLU/swish through Metal when
  available, writes `[1,tokens,4096]`, and leaves layer-4 ffn1 output
  projection/half-residual plus encoder layers 5-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 ffn1 output projection boundary.
  `mit2_tts --clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR
  W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32`
  validates `encoder.layers.4.ffn1.output_dense.{weight,bias}`, applies the
  native 4096-to-1024 projection through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-4 ffn1 half-residual plus encoder layers
  5-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32
  W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32`
  consumes layer-3 hidden and layer-4 ffn1 output sidecars `[1,tokens,1024]`,
  applies `layer3 + 0.5 * layer4_ffn1_output` through Metal when available,
  writes `[1,tokens,1024]`, and leaves layer-4 self-attention Q/K/V plus
  encoder layers 5-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 self-attention Q/K/V sidecar boundary.
  `mit2_tts --clone-w2v-layer4-qkv MODEL_BUNDLE_DIR
  W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` validates
  `encoder.layers.4.self_attn.linear_{q,k,v}.weight`, applies the three native
  1024-to-1024 projections through resident Metal when available, writes
  layer-4 Q/K/V sidecars plus a manifest, and leaves layer-4 attention
  scores/context plus encoder layers 5-16 as the remaining native W2V-BERT
  encoder gap.
- Close the W2V-BERT layer-4 self-attention context boundary.
  `mit2_tts --clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32
  W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER4_CONTEXT_F32` consumes layer-4 Q/K/V and the W2V attention mask,
  applies 16-head masked attention through Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-4 attention output projection/residual/norm
  plus encoder layers 5-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 attention output projection boundary.
  `mit2_tts --clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32` validates
  `encoder.layers.4.self_attn.linear_out.weight`, applies the native
  1024-to-1024 projection through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-4 attention residual/norm plus
  encoder layers 5-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 attention residual boundary.
  `mit2_tts --clone-w2v-layer4-attention-residual
  W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32` adds the projected attention sidecar
  back to the layer-4 ffn1 residual through Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-4 attention norm plus encoder layers 5-16
  as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER4_ATTENTION_NORM_F32` validates
  `encoder.layers.4.self_attn_layer_norm.{weight,bias}`, applies row LayerNorm
  through resident Metal when available, writes `[1,tokens,1024]`, and leaves
  layer-4 convolution plus encoder layers 5-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32`
  validates `encoder.layers.4.conv_module.layer_norm.{weight,bias}`, applies
  row LayerNorm through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-4 convolution GLU/depthwise/residual, ffn2, and encoder
  layers 5-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32` validates
  `encoder.layers.4.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native 1024-to-2048 projection plus GLU split through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-4 convolution
  depthwise/residual, ffn2, and encoder layers 5-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 convolution depthwise boundary.
  `mit2_tts --clone-w2v-layer4-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER4_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_DEPTHWISE_F32`
  validates `encoder.layers.4.conv_module.depthwise_conv.{weight,bias}`,
  applies causal depthwise Conv1d through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-4 convolution residual, ffn2, and encoder
  layers 5-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 convolution residual boundary.
  `mit2_tts --clone-w2v-layer4-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER4_ATTENTION_NORM_F32 W2V_LAYER4_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER4_CONV_RESIDUAL_F32` validates
  `encoder.layers.4.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.4.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-4 ffn2 plus encoder layers 5-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-4 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer4-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER4_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_FFN2_RESIDUAL_F32`
  validates `encoder.layers.4.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.4.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.4.ffn2.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the 0.5 residual through resident Metal when available,
  writes `[1,tokens,1024]`, and hands off to the layer-5 ffn1 boundary.
- Close the W2V-BERT layer-5 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer5-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER4_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN1_RESIDUAL_F32`
  validates `encoder.layers.5.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.5.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.5.ffn1.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the 0.5 residual through resident Metal when available,
  writes `[1,tokens,1024]`, and hands off to the layer-5 self-attention Q/K/V
  boundary.
- Close the W2V-BERT layer-5 self-attention Q/K/V sidecar boundary.
  `mit2_tts --clone-w2v-layer5-qkv MODEL_BUNDLE_DIR
  W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` validates
  `encoder.layers.5.self_attn.linear_{q,k,v}.weight`, applies the three native
  1024-to-1024 projections through resident Metal when available, writes
  layer-5 Q/K/V sidecars plus a manifest, and hands off to the layer-5
  attention context boundary.
- Close the W2V-BERT layer-5 self-attention context boundary.
  `mit2_tts --clone-w2v-layer5-attention W2V_LAYER5_Q_F32
  W2V_LAYER5_K_F32 W2V_LAYER5_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER5_CONTEXT_F32` consumes layer-5 Q/K/V sidecars plus the W2V
  attention mask, applies 16-head masked attention through Metal when
  available, writes `[1,tokens,1024]`, and hands off to the layer-5 attention
  output projection boundary.
- Close the W2V-BERT layer-5 attention output projection boundary.
  `mit2_tts --clone-w2v-layer5-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER5_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_F32` validates
  `encoder.layers.5.self_attn.linear_out.weight`, consumes layer-5 attention
  context, applies the native projection through resident Metal when available,
  writes `[1,tokens,1024]`, and hands off to the layer-5 attention residual
  boundary.
- Close the W2V-BERT layer-5 attention residual boundary.
  `mit2_tts --clone-w2v-layer5-attention-residual
  W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_LAYER5_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER5_ATTENTION_RESIDUAL_F32` consumes layer-5 ffn1 residual and
  attention projection sidecars, applies the residual add through Metal when
  available, writes `[1,tokens,1024]`, and hands off to the layer-5 attention
  norm boundary.
- Close the W2V-BERT layer-5 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer5-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER5_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER5_ATTENTION_NORM_F32` validates
  `encoder.layers.5.self_attn_layer_norm.{weight,bias}`, consumes the layer-5
  attention residual sidecar, applies row LayerNorm through resident Metal when
  available, writes `[1,tokens,1024]`, and hands off to the layer-5 convolution
  LayerNorm boundary.
- Close the W2V-BERT layer-5 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer5-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER5_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_NORM_F32`
  validates `encoder.layers.5.conv_module.layer_norm.{weight,bias}`, consumes
  the layer-5 attention-norm sidecar, applies row LayerNorm through resident
  Metal when available, writes `[1,tokens,1024]`, and hands off to the layer-5
  convolution GLU boundary.
- Close the W2V-BERT layer-5 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer5-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER5_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_GLU_F32` validates
  `encoder.layers.5.conv_module.pointwise_conv1.{weight,bias}`, consumes the
  layer-5 conv-norm sidecar, applies the native 1024-to-2048 projection plus GLU
  split through resident Metal when available, writes `[1,tokens,1024]`, and
  hands off to the layer-5 convolution depthwise boundary.
- Close the W2V-BERT layer-5 convolution depthwise boundary.
  `mit2_tts --clone-w2v-layer5-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER5_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_DEPTHWISE_F32` validates
  `encoder.layers.5.conv_module.depthwise_conv.{weight,bias}`, consumes the
  layer-5 conv-GLU sidecar, applies causal depthwise Conv1d through resident
  Metal when available, writes `[1,tokens,1024]`, and hands off to the layer-5
  convolution residual boundary.
- Close the W2V-BERT layer-5 convolution residual boundary.
  `mit2_tts --clone-w2v-layer5-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER5_ATTENTION_NORM_F32 W2V_LAYER5_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER5_CONV_RESIDUAL_F32` validates
  `encoder.layers.5.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.5.conv_module.pointwise_conv2.{weight,bias}`, consumes the
  layer-5 attention-norm and conv-depthwise sidecars, applies depthwise
  LayerNorm, SiLU, pointwise projection, and the attention-norm residual add
  through resident Metal when available, writes `[1,tokens,1024]`, and hands
  off to the layer-5 ffn2 boundary.
- Close the W2V-BERT layer-5 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer5-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER5_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN2_RESIDUAL_F32`
  validates `encoder.layers.5.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.5.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.5.ffn2.output_dense.{weight,bias}`, consumes the layer-5
  conv-residual sidecar, applies LayerNorm, SiLU feed-forward, and the second
  half-residual through resident Metal when available, writes
  `[1,tokens,1024]`, and hands off to the layer-6 ffn1 boundary.
- Close the W2V-BERT layer-6 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer6-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER5_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN1_RESIDUAL_F32`
  validates `encoder.layers.6.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.6.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.6.ffn1.output_dense.{weight,bias}`, consumes the layer-5
  ffn2-residual sidecar, applies LayerNorm, SiLU feed-forward, and the first
  half-residual through resident Metal when available, writes
  `[1,tokens,1024]`, and hands off to the layer-6 Q/K/V boundary.
- Close the W2V-BERT layer-6 self-attention Q/K/V sidecar boundary.
  `mit2_tts --clone-w2v-layer6-qkv MODEL_BUNDLE_DIR
  W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` validates
  `encoder.layers.6.self_attn.linear_{q,k,v,out}.weight`, consumes the
  layer-6 ffn1 residual sidecar, applies the native Q/K/V projections through
  resident Metal when available, writes `w2v_layer6_{q,k,v}.f32` plus a
  `w2v_layer6_qkv.manifest.json`, and leaves layer-6 attention
  scores/context, attention output projection, convolution, ffn2, and encoder
  layers 7-16
  as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 self-attention context boundary.
  `mit2_tts --clone-w2v-layer6-attention W2V_LAYER6_Q_F32 W2V_LAYER6_K_F32
  W2V_LAYER6_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER6_CONTEXT_F32` consumes layer-6 Q/K/V sidecars plus the W2V
  attention mask, applies 16-head masked attention through Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-6 attention output
  projection/residual/norm, convolution, ffn2, and encoder layers 7-16
  as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 attention output projection boundary.
  `mit2_tts --clone-w2v-layer6-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER6_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_F32` consumes the
  layer-6 attention context sidecar, validates
  `encoder.layers.6.self_attn.linear_out.weight`, applies the native
  1024-to-1024 projection through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-6 attention residual/norm, convolution,
  ffn2, and encoder layers 7-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 attention residual boundary.
  `mit2_tts --clone-w2v-layer6-attention-residual
  W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_LAYER6_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER6_ATTENTION_RESIDUAL_F32` consumes layer-6 ffn1 residual and
  attention projection sidecars, applies the residual add through Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-6 attention norm,
  convolution, ffn2, and encoder layers 7-16 as the remaining native W2V-BERT
  encoder gap.
- Close the W2V-BERT layer-6 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer6-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER6_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER6_ATTENTION_NORM_F32` consumes the layer-6 attention residual
  sidecar, validates `encoder.layers.6.self_attn_layer_norm.{weight,bias}`,
  applies the 1024-wide LayerNorm through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-6 convolution, ffn2, and encoder layers
  7-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer6-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER6_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_NORM_F32`
  consumes the layer-6 attention LayerNorm sidecar, validates
  `encoder.layers.6.conv_module.layer_norm.{weight,bias}`, applies the
  1024-wide convolution-module LayerNorm through resident Metal when available,
  writes `[1,tokens,1024]`, and leaves layer-6 convolution GLU,
  depthwise/residual, ffn2, and encoder layers 7-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer6-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER6_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_GLU_F32` consumes the
  layer-6 convolution-module LayerNorm sidecar, validates
  `encoder.layers.6.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native 1024-to-2048 projection plus GLU split through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-6 depthwise
  convolution/residual, ffn2, and encoder layers 7-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer6-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER6_CONV_GLU_F32 W2V_TOKENS
  OUTPUT_LAYER6_CONV_DEPTHWISE_F32` consumes the layer-6 convolution GLU
  sidecar, validates `encoder.layers.6.conv_module.depthwise_conv.{weight,bias}`,
  applies causal depthwise Conv1d through resident Metal when available,
  writes `[1,tokens,1024]`, and leaves layer-6 convolution residual, ffn2, and
  encoder layers 7-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 convolution residual boundary.
  `mit2_tts --clone-w2v-layer6-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER6_ATTENTION_NORM_F32 W2V_LAYER6_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER6_CONV_RESIDUAL_F32` consumes the layer-6 attention LayerNorm and
  causal depthwise sidecars, validates
  `encoder.layers.6.conv_module.depthwise_layer_norm.{weight,bias}` plus
  `encoder.layers.6.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and residual add through
  resident Metal when available, writes `[1,tokens,1024]`, and leaves layer-6
  ffn2 plus encoder layers 7-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-6 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer6-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER6_CONV_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER6_FFN2_RESIDUAL_F32` consumes the layer-6 convolution residual
  sidecar, validates `encoder.layers.6.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.6.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.6.ffn2.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the second half-residual through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves encoder layers 7-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer7-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER6_FFN2_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER7_FFN1_RESIDUAL_F32` consumes the layer-6 ffn2 residual sidecar,
  validates `encoder.layers.7.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.7.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.7.ffn1.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the first half-residual through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-7 self-attention,
  convolution, ffn2, plus encoder layers 8-16 as the remaining native W2V-BERT
  encoder gap.
- Close the W2V-BERT layer-7 self-attention Q/K/V boundary.
  `mit2_tts --clone-w2v-layer7-qkv MODEL_BUNDLE_DIR
  W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` consumes the layer-7 ffn1
  residual sidecar, validates `encoder.layers.7.self_attn.linear_{q,k,v,out}.weight`,
  applies Q/K/V projections through resident Metal when available, writes
  `w2v_layer7_{q,k,v}.f32` plus `w2v_layer7_qkv.manifest.json`, and leaves
  layer-7 attention scores/context/projection/residual/norm, convolution, ffn2,
  plus encoder layers 8-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 self-attention context boundary.
  `mit2_tts --clone-w2v-layer7-attention W2V_LAYER7_Q_F32 W2V_LAYER7_K_F32
  W2V_LAYER7_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER7_CONTEXT_F32` consumes the layer-7 Q/K/V sidecars and attention
  mask, applies 16-head masked attention through Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-7 attention output
  projection/residual/norm, convolution, ffn2, plus encoder layers 8-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 attention output-projection boundary.
  `mit2_tts --clone-w2v-layer7-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER7_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_F32` consumes the
  layer-7 context sidecar, validates `encoder.layers.7.self_attn.linear_out.weight`,
  applies the resident Metal projection when available, writes `[1,tokens,1024]`,
  and leaves layer-7 attention residual/norm, convolution, ffn2, plus encoder
  layers 8-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 attention-residual boundary.
  `mit2_tts --clone-w2v-layer7-attention-residual W2V_LAYER7_FFN1_RESIDUAL_F32
  W2V_LAYER7_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_RESIDUAL_F32`
  consumes the layer-7 ffn1 residual and projected attention sidecars, applies
  the residual add through Metal when available, writes `[1,tokens,1024]`, and
  leaves layer-7 attention norm, convolution, ffn2, plus encoder layers 8-16 as
  the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 attention-LayerNorm boundary.
  `mit2_tts --clone-w2v-layer7-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER7_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER7_ATTENTION_NORM_F32` consumes the layer-7 attention residual,
  validates `encoder.layers.7.self_attn_layer_norm.{weight,bias}`, applies
  1024-wide LayerNorm through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-7 convolution, ffn2, plus encoder layers
  8-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer7-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER7_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_NORM_F32`
  consumes the layer-7 attention norm sidecar, validates
  `encoder.layers.7.conv_module.layer_norm.{weight,bias}`, applies 1024-wide
  LayerNorm through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-7 convolution GLU/depthwise/residual, ffn2, plus encoder
  layers 8-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 convolution-GLU boundary.
  `mit2_tts --clone-w2v-layer7-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER7_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_GLU_F32` consumes the
  layer-7 conv-norm sidecar, validates
  `encoder.layers.7.conv_module.pointwise_conv1.{weight,bias}`, applies the
  1024-to-2048 pointwise projection plus GLU split through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-7 depthwise
  convolution/residual, ffn2, plus encoder layers 8-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer7-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER7_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_DEPTHWISE_F32`
  consumes the layer-7 conv-GLU sidecar, validates
  `encoder.layers.7.conv_module.depthwise_conv.{weight,bias}`, applies causal
  depthwise Conv1d through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-7 convolution residual, ffn2, plus
  encoder layers 8-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 convolution-residual boundary.
  `mit2_tts --clone-w2v-layer7-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER7_ATTENTION_NORM_F32 W2V_LAYER7_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER7_CONV_RESIDUAL_F32` consumes layer-7 attention-norm and
  conv-depthwise sidecars, validates
  `encoder.layers.7.conv_module.depthwise_layer_norm.{weight,bias}` plus
  `encoder.layers.7.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the residual add through
  resident Metal when available, writes `[1,tokens,1024]`, and leaves layer-7
  ffn2 plus encoder layers 8-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-7 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer7-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER7_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN2_RESIDUAL_F32`
  consumes the layer-7 convolution-residual sidecar, validates
  `encoder.layers.7.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.7.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.7.ffn2.output_dense.{weight,bias}`, applies LayerNorm,
  intermediate dense, SiLU, output dense, and the 0.5 feed-forward residual
  through resident Metal when available, writes `[1,tokens,1024]`, and leaves
  encoder layers 8-16 as the remaining native W2V-BERT encoder gap.
- Open and close the W2V-BERT layer-8 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer8-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER7_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN1_RESIDUAL_F32`
  consumes the layer-7 ffn2-residual sidecar, validates
  `encoder.layers.8.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.8.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.8.ffn1.output_dense.{weight,bias}`, applies LayerNorm,
  intermediate dense, SiLU, output dense, and the first 0.5 feed-forward
  residual through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-8 self-attention, convolution, ffn2, plus encoder layers
  9-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 self-attention Q/K/V projection boundary.
  `mit2_tts --clone-w2v-layer8-qkv MODEL_BUNDLE_DIR
  W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` consumes the layer-8
  ffn1-residual sidecar, validates
  `encoder.layers.8.self_attn.linear_{q,k,v,out}.weight`, applies Q/K/V
  projections through resident Metal when available, writes
  `w2v_layer8_q.f32`, `w2v_layer8_k.f32`, `w2v_layer8_v.f32`, and a
  `w2v_layer8_qkv.manifest.json`, and leaves layer-8 attention
  context/projection/residual/norm, convolution, ffn2, plus encoder
  layers 9-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 self-attention context boundary.
  `mit2_tts --clone-w2v-layer8-attention W2V_LAYER8_Q_F32 W2V_LAYER8_K_F32
  W2V_LAYER8_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER8_CONTEXT_F32` consumes the layer-8 Q/K/V sidecars plus the W2V
  attention mask, applies 16-head masked scaled dot-product attention through
  resident Metal when available, writes `[1,tokens,1024]`, and leaves layer-8
  attention output projection/residual/norm, convolution, ffn2, plus encoder
  layers 9-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 self-attention output projection boundary.
  `mit2_tts --clone-w2v-layer8-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER8_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_F32` consumes the
  layer-8 attention context, validates
  `encoder.layers.8.self_attn.linear_out.weight`, applies the output projection
  through resident Metal when available, writes `[1,tokens,1024]`, and leaves
  layer-8 attention residual/norm, convolution, ffn2, plus encoder layers 9-16
  as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 self-attention residual boundary.
  `mit2_tts --clone-w2v-layer8-attention-residual
  W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_LAYER8_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER8_ATTENTION_RESIDUAL_F32` consumes the layer-8 ffn1 residual and
  attention projection sidecars, adds them through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-8 attention norm,
  convolution, ffn2, plus encoder layers 9-16 as the remaining native W2V-BERT
  encoder gap.
- Close the W2V-BERT layer-8 self-attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer8-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER8_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER8_ATTENTION_NORM_F32` consumes the layer-8 attention residual,
  validates `encoder.layers.8.self_attn_layer_norm.{weight,bias}`, applies
  1024-wide LayerNorm through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-8 convolution-module LayerNorm,
  convolution GLU/depthwise/residual, ffn2, plus encoder layers 9-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer8-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER8_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_NORM_F32`
  consumes the layer-8 attention norm, validates
  `encoder.layers.8.conv_module.layer_norm.{weight,bias}`, applies 1024-wide
  LayerNorm through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-8 convolution GLU/depthwise/residual, ffn2, plus encoder
  layers 9-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer8-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER8_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_GLU_F32` consumes the
  layer-8 convolution-module LayerNorm sidecar, validates
  `encoder.layers.8.conv_module.pointwise_conv1.{weight,bias}`, applies the
  1024-to-2048 pointwise projection plus GLU split through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-8 depthwise
  convolution/residual, ffn2, plus encoder layers 9-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer8-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER8_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_DEPTHWISE_F32`
  consumes the layer-8 convolution GLU sidecar, validates
  `encoder.layers.8.conv_module.depthwise_conv.{weight,bias}`, applies causal
  depthwise Conv1d through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-8 convolution residual, ffn2, plus
  encoder layers 9-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 convolution residual boundary.
  `mit2_tts --clone-w2v-layer8-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER8_ATTENTION_NORM_F32 W2V_LAYER8_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER8_CONV_RESIDUAL_F32` consumes the layer-8 attention-norm and
  conv-depthwise sidecars, validates
  `encoder.layers.8.conv_module.depthwise_layer_norm.{weight,bias}` plus
  `encoder.layers.8.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes `[1,tokens,1024]`,
  and leaves layer-8 ffn2 plus encoder layers 9-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-8 ffn2 residual boundary.
  `mit2_tts --clone-w2v-layer8-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER8_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN2_RESIDUAL_F32`
  consumes the layer-8 convolution residual sidecar, validates
  `encoder.layers.8.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.8.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.8.ffn2.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the 0.5 residual through resident Metal when available,
  writes `[1,tokens,1024]`, and leaves layer-9 ffn1 plus layer-9
  self-attention, convolution, ffn2, and encoder layers 10-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 ffn1 residual boundary.
  `mit2_tts --clone-w2v-layer9-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER8_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN1_RESIDUAL_F32`
  consumes the layer-8 ffn2 residual sidecar, validates
  `encoder.layers.9.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.9.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.9.ffn1.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, and the first 0.5 residual through resident Metal when
  available, writes `[1,tokens,1024]`, and leaves layer-9 self-attention Q/K/V
  plus attention context/projection/residual/norm, convolution, ffn2, and
  encoder layers 10-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 self-attention Q/K/V boundary.
  `mit2_tts --clone-w2v-layer9-qkv MODEL_BUNDLE_DIR
  W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` consumes the layer-9
  ffn1 residual sidecar, validates
  `encoder.layers.9.self_attn.linear_{q,k,v,out}.weight`, applies Q/K/V
  projections through resident Metal when available, writes `w2v_layer9_q.f32`,
  `w2v_layer9_k.f32`, `w2v_layer9_v.f32`, and a
  `w2v_layer9_qkv.manifest.json`, and leaves layer-9 attention
  context/projection/residual/norm, convolution, ffn2, plus encoder layers
  10-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 self-attention context boundary.
  `mit2_tts --clone-w2v-layer9-attention W2V_LAYER9_Q_F32 W2V_LAYER9_K_F32
  W2V_LAYER9_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER9_CONTEXT_F32` consumes layer-9 Q/K/V sidecars plus the W2V
  attention mask, applies 16-head masked attention through resident Metal when
  available, writes `w2v_layer9_context.f32` `[1,tokens,1024]`, and leaves
  layer-9 attention output projection/residual/norm, convolution, ffn2, plus
  encoder layers 10-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 self-attention output projection boundary.
  `mit2_tts --clone-w2v-layer9-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER9_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_F32` consumes
  layer-9 attention context, validates
  `encoder.layers.9.self_attn.linear_out.weight`, applies the layer-9
  attention output projection through resident Metal when available, writes
  `[1,tokens,1024]`, and leaves layer-9 attention residual/norm, convolution,
  ffn2, plus encoder layers 10-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 self-attention residual boundary.
  `mit2_tts --clone-w2v-layer9-attention-residual
  W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_LAYER9_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER9_ATTENTION_RESIDUAL_F32` consumes layer-9 ffn1 residual and
  attention projection sidecars, adds them through resident Metal when
  available, writes `w2v_layer9_attention_residual.f32` `[1,tokens,1024]`,
  and leaves layer-9 attention norm, convolution, ffn2, plus encoder layers
  10-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 self-attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer9-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER9_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER9_ATTENTION_NORM_F32` consumes layer-9 attention residual,
  validates `encoder.layers.9.self_attn_layer_norm.{weight,bias}`, applies
  1024-wide LayerNorm through resident Metal when available, writes
  `w2v_layer9_attention_norm.f32` `[1,tokens,1024]`, and leaves layer-9
  convolution-module LayerNorm/GLU/depthwise/residual, ffn2, plus encoder
  layers 10-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer9-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER9_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_NORM_F32`
  consumes layer-9 attention norm, validates
  `encoder.layers.9.conv_module.layer_norm.{weight,bias}`, applies 1024-wide
  LayerNorm through resident Metal when available, writes
  `w2v_layer9_conv_norm.f32` `[1,tokens,1024]`, and leaves layer-9
  convolution GLU/depthwise/residual, ffn2, plus encoder layers 10-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer9-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER9_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_GLU_F32` consumes
  layer-9 conv norm, validates
  `encoder.layers.9.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native 1024-to-2048 projection plus GLU split through resident Metal when
  available, writes `w2v_layer9_conv_glu.f32` `[1,tokens,1024]`, and leaves
  layer-9 depthwise convolution/residual, ffn2, plus encoder layers 10-16 as
  the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer9-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER9_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_DEPTHWISE_F32`
  consumes layer-9 conv GLU, validates
  `encoder.layers.9.conv_module.depthwise_conv.{weight,bias}`, applies causal
  depthwise Conv1d through resident Metal when available, writes
  `w2v_layer9_conv_depthwise.f32` `[1,tokens,1024]`, and leaves layer-9
  convolution residual, ffn2, plus encoder layers 10-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 convolution residual boundary.
  `mit2_tts --clone-w2v-layer9-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER9_ATTENTION_NORM_F32 W2V_LAYER9_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER9_CONV_RESIDUAL_F32` consumes layer-9 attention norm and
  conv-depthwise sidecars, validates
  `encoder.layers.9.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.9.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes
  `w2v_layer9_conv_residual.f32` `[1,tokens,1024]`, and leaves layer-9 ffn2
  plus encoder layers 10-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-9 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer9-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER9_CONV_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER9_FFN2_RESIDUAL_F32` consumes layer-9 conv-residual sidecar,
  validates `encoder.layers.9.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.9.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.9.ffn2.output_dense.{weight,bias}`, applies LayerNorm,
  SiLU feed-forward, output projection, and the 0.5 residual through resident
  Metal when available, writes `w2v_layer9_ffn2_residual.f32`
  `[1,tokens,1024]`, and leaves encoder layers 10-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer10-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER9_FFN2_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER10_FFN1_RESIDUAL_F32` consumes layer-9 ffn2-residual sidecar,
  validates `encoder.layers.10.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.10.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.10.ffn1.output_dense.{weight,bias}`, applies LayerNorm,
  SiLU feed-forward, output projection, and the first 0.5 residual through
  resident Metal when available, writes `w2v_layer10_ffn1_residual.f32`
  `[1,tokens,1024]`, and leaves layer-10 self-attention/convolution/ffn2 plus
  encoder layers 11-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 Q/K/V projection boundary.
  `mit2_tts --clone-w2v-layer10-qkv MODEL_BUNDLE_DIR
  W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` consumes layer-10
  ffn1-residual sidecar, validates
  `encoder.layers.10.self_attn.linear_{q,k,v}.weight`, applies Q/K/V
  projections through resident Metal when available, writes
  `w2v_layer10_q.f32`, `w2v_layer10_k.f32`, `w2v_layer10_v.f32`, and
  `w2v_layer10_qkv.manifest.json`, and leaves layer-10 attention
  context/projection/residual/norm, convolution, ffn2, plus encoder layers
  11-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 attention context boundary.
  `mit2_tts --clone-w2v-layer10-attention W2V_LAYER10_Q_F32
  W2V_LAYER10_K_F32 W2V_LAYER10_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER10_CONTEXT_F32` consumes layer-10 Q/K/V sidecars plus the
  attention mask, applies 16-head masked scaled dot-product attention through
  Metal when available, writes `w2v_layer10_context.f32` `[1,tokens,1024]`,
  and leaves layer-10 attention output projection/residual/norm, convolution,
  ffn2, plus encoder layers 11-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 attention output projection boundary.
  `mit2_tts --clone-w2v-layer10-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER10_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_F32` validates
  `encoder.layers.10.self_attn.linear_out.weight`, applies the attention
  output projection through resident Metal when available, writes
  `w2v_layer10_attention.f32` `[1,tokens,1024]`, and leaves layer-10 attention
  residual/norm, convolution, ffn2, plus encoder layers 11-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 attention residual boundary.
  `mit2_tts --clone-w2v-layer10-attention-residual
  W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_LAYER10_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER10_ATTENTION_RESIDUAL_F32` consumes layer-10 ffn1 residual and
  attention projection sidecars `[1,tokens,1024]`, adds them through resident
  Metal when available, writes `w2v_layer10_attention_residual.f32`, and leaves
  layer-10 attention norm, convolution, ffn2, plus encoder layers 11-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer10-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER10_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER10_ATTENTION_NORM_F32` validates
  `encoder.layers.10.self_attn_layer_norm.{weight,bias}`, applies 1024-wide
  LayerNorm through resident Metal when available, writes
  `w2v_layer10_attention_norm.f32` `[1,tokens,1024]`, and leaves layer-10
  convolution-module LayerNorm/GLU/depthwise/residual, ffn2, plus encoder
  layers 11-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer10-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER10_ATTENTION_NORM_F32 W2V_TOKENS
  OUTPUT_LAYER10_CONV_NORM_F32` validates
  `encoder.layers.10.conv_module.layer_norm.{weight,bias}`, applies 1024-wide
  convolution-module LayerNorm through resident Metal when available, writes
  `w2v_layer10_conv_norm.f32` `[1,tokens,1024]`, and leaves layer-10
  convolution GLU/depthwise/residual, ffn2, plus encoder layers 11-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer10-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER10_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_GLU_F32` validates
  `encoder.layers.10.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native 1024-to-2048 pointwise projection plus GLU split through resident Metal
  when available, writes `w2v_layer10_conv_glu.f32` `[1,tokens,1024]`, and
  leaves layer-10 depthwise convolution/residual, ffn2, plus encoder layers
  11-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer10-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER10_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_DEPTHWISE_F32`
  validates `encoder.layers.10.conv_module.depthwise_conv.{weight,bias}`,
  applies causal depthwise Conv1d through resident Metal when available, writes
  `w2v_layer10_conv_depthwise.f32` `[1,tokens,1024]`, and leaves layer-10
  convolution residual, ffn2, plus encoder layers 11-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 convolution residual boundary.
  `mit2_tts --clone-w2v-layer10-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER10_ATTENTION_NORM_F32 W2V_LAYER10_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER10_CONV_RESIDUAL_F32` validates
  `encoder.layers.10.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.10.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes
  `w2v_layer10_conv_residual.f32` `[1,tokens,1024]`, and leaves layer-10 ffn2
  plus encoder layers 11-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-10 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer10-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER10_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN2_RESIDUAL_F32`
  validates `encoder.layers.10.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.10.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.10.ffn2.output_dense.{weight,bias}`, applies LayerNorm,
  SiLU feed-forward, output projection, and the 0.5 residual through resident
  Metal when available, writes `w2v_layer10_ffn2_residual.f32`
  `[1,tokens,1024]`, and leaves encoder layers 11-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer11-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER10_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN1_RESIDUAL_F32`
  validates `encoder.layers.11.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.11.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.11.ffn1.output_dense.{weight,bias}`, applies LayerNorm,
  SiLU feed-forward, output projection, and the first 0.5 residual through
  resident Metal when available, writes `w2v_layer11_ffn1_residual.f32`
  `[1,tokens,1024]`, and leaves layer-11 self-attention/convolution/ffn2 plus
  encoder layers 12-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 self-attention Q/K/V projection boundary.
  `mit2_tts --clone-w2v-layer11-qkv MODEL_BUNDLE_DIR
  W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` validates
  `encoder.layers.11.self_attn.linear_{q,k,v}.weight`, applies the three
  1024-wide projections through resident Metal when available, writes
  `w2v_layer11_q.f32`, `w2v_layer11_k.f32`, `w2v_layer11_v.f32`, and
  `w2v_layer11_qkv.manifest.json`, and leaves layer-11 attention
  context/projection/residual/norm, convolution, ffn2, plus encoder layers
  12-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 attention-context boundary.
  `mit2_tts --clone-w2v-layer11-attention W2V_LAYER11_Q_F32
  W2V_LAYER11_K_F32 W2V_LAYER11_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER11_CONTEXT_F32` consumes layer-11 Q/K/V sidecars plus the
  attention mask, applies 16-head masked scaled dot-product attention through
  Metal when available, writes `w2v_layer11_context.f32` `[1,tokens,1024]`,
  and leaves layer-11 attention output projection/residual/norm, convolution,
  ffn2, plus encoder layers 12-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 attention output projection boundary.
  `mit2_tts --clone-w2v-layer11-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER11_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_F32` validates
  `encoder.layers.11.self_attn.linear_out.weight`, applies the output
  projection through resident Metal when available, writes
  `w2v_layer11_attention.f32` `[1,tokens,1024]`, and leaves layer-11 attention
  residual/norm, convolution, ffn2, plus encoder layers 12-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 attention residual boundary.
  `mit2_tts --clone-w2v-layer11-attention-residual
  W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_LAYER11_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER11_ATTENTION_RESIDUAL_F32` consumes layer-11 ffn1 residual and
  attention projection sidecars `[1,tokens,1024]`, adds them through resident
  Metal when available, writes `w2v_layer11_attention_residual.f32`, and leaves
  layer-11 attention norm, convolution, ffn2, plus encoder layers 12-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer11-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER11_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER11_ATTENTION_NORM_F32` validates
  `encoder.layers.11.self_attn_layer_norm.{weight,bias}`, applies 1024-wide
  LayerNorm through resident Metal when available, writes
  `w2v_layer11_attention_norm.f32` `[1,tokens,1024]`, and leaves layer-11
  convolution-module LayerNorm/GLU/depthwise/residual, ffn2, plus encoder
  layers 12-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer11-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER11_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_NORM_F32`
  validates `encoder.layers.11.conv_module.layer_norm.{weight,bias}`, applies
  1024-wide convolution-module LayerNorm through resident Metal when available,
  writes `w2v_layer11_conv_norm.f32` `[1,tokens,1024]`, and leaves layer-11
  convolution GLU/depthwise/residual, ffn2, plus encoder layers 12-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer11-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER11_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_GLU_F32` validates
  `encoder.layers.11.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native 1024-to-2048 pointwise projection plus GLU split through resident
  Metal when available, writes `w2v_layer11_conv_glu.f32` `[1,tokens,1024]`,
  and leaves layer-11 depthwise convolution/residual, ffn2, plus encoder layers
  12-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer11-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER11_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_DEPTHWISE_F32`
  validates `encoder.layers.11.conv_module.depthwise_conv.{weight,bias}`,
  applies causal depthwise Conv1d through resident Metal when available, writes
  `w2v_layer11_conv_depthwise.f32` `[1,tokens,1024]`, and leaves layer-11
  convolution residual, ffn2, plus encoder layers 12-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 convolution residual boundary.
  `mit2_tts --clone-w2v-layer11-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER11_ATTENTION_NORM_F32 W2V_LAYER11_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER11_CONV_RESIDUAL_F32` validates
  `encoder.layers.11.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.11.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes
  `w2v_layer11_conv_residual.f32` `[1,tokens,1024]`, and leaves layer-11 ffn2
  plus encoder layers 12-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-11 ffn2 residual boundary.
  `mit2_tts --clone-w2v-layer11-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER11_CONV_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER11_FFN2_RESIDUAL_F32` validates
  `encoder.layers.11.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.11.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.11.ffn2.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, output projection, and the second 0.5 residual through resident
  Metal when available, writes `w2v_layer11_ffn2_residual.f32`
  `[1,tokens,1024]`, and leaves encoder layers 12-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 ffn1 residual boundary.
  `mit2_tts --clone-w2v-layer12-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER11_FFN2_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER12_FFN1_RESIDUAL_F32` validates
  `encoder.layers.12.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.12.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.12.ffn1.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, output projection, and the first 0.5 residual through resident
  Metal when available, writes `w2v_layer12_ffn1_residual.f32`
  `[1,tokens,1024]`, and leaves layer-12 self-attention, convolution, ffn2,
  plus encoder layers 13-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 self-attention Q/K/V projection boundary.
  `mit2_tts --clone-w2v-layer12-qkv MODEL_BUNDLE_DIR
  W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` validates
  `encoder.layers.12.self_attn.linear_{q,k,v}.weight`, applies the three
  1024-wide self-attention projections through resident Metal when available,
  writes `w2v_layer12_q.f32`, `w2v_layer12_k.f32`, `w2v_layer12_v.f32`, and
  `w2v_layer12_qkv.manifest.json`, and leaves layer-12 attention
  context/projection/residual/norm, convolution, ffn2, plus encoder layers 13-16
  as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 self-attention context boundary.
  `mit2_tts --clone-w2v-layer12-attention W2V_LAYER12_Q_F32
  W2V_LAYER12_K_F32 W2V_LAYER12_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER12_CONTEXT_F32` consumes the layer-12 Q/K/V sidecars plus the
  attention mask, applies 16-head masked scaled dot-product attention through
  Metal when available, writes `w2v_layer12_context.f32` `[1,tokens,1024]`, and
  leaves layer-12 attention output projection/residual/norm, convolution, ffn2,
  plus encoder layers 13-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 attention output projection boundary.
  `mit2_tts --clone-w2v-layer12-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER12_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_F32` validates
  `encoder.layers.12.self_attn.linear_out.weight`, applies the attention output
  projection through resident Metal when available, writes
  `w2v_layer12_attention.f32` `[1,tokens,1024]`, and leaves layer-12 attention
  residual/norm, convolution, ffn2, plus encoder layers 13-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 attention residual boundary.
  `mit2_tts --clone-w2v-layer12-attention-residual
  W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_LAYER12_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER12_ATTENTION_RESIDUAL_F32` consumes layer-12 ffn1 residual and
  attention projection sidecars `[1,tokens,1024]`, adds them through resident
  Metal when available, writes `w2v_layer12_attention_residual.f32`, and leaves
  layer-12 attention norm, convolution, ffn2, plus encoder layers 13-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer12-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER12_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER12_ATTENTION_NORM_F32` consumes layer-12 attention residual
  `[1,tokens,1024]`, validates
  `encoder.layers.12.self_attn_layer_norm.{weight,bias}`, applies the
  1024-wide LayerNorm through resident Metal when available, writes
  `w2v_layer12_attention_norm.f32`, and leaves layer-12 convolution, ffn2, plus
  encoder layers 13-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer12-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER12_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_NORM_F32`
  consumes layer-12 attention norm `[1,tokens,1024]`, validates
  `encoder.layers.12.conv_module.layer_norm.{weight,bias}`, applies the
  1024-wide row LayerNorm through resident Metal when available, writes
  `w2v_layer12_conv_norm.f32`, and leaves layer-12 convolution
  GLU/depthwise/residual, ffn2, plus encoder layers 13-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer12-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER12_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_GLU_F32` consumes
  layer-12 conv norm `[1,tokens,1024]`, validates
  `encoder.layers.12.conv_module.pointwise_conv1.{weight,bias}`, applies the
  1024-to-2048 projection plus GLU split through resident Metal when available,
  writes `w2v_layer12_conv_glu.f32`, and leaves layer-12 depthwise
  convolution/residual, ffn2, plus encoder layers 13-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer12-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER12_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_DEPTHWISE_F32`
  consumes layer-12 conv GLU `[1,tokens,1024]`, validates
  `encoder.layers.12.conv_module.depthwise_conv.{weight,bias}`, applies causal
  depthwise Conv1d through resident Metal when available, writes
  `w2v_layer12_conv_depthwise.f32`, and leaves layer-12 convolution residual,
  ffn2, plus encoder layers 13-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 convolution residual boundary.
  `mit2_tts --clone-w2v-layer12-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER12_ATTENTION_NORM_F32 W2V_LAYER12_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER12_CONV_RESIDUAL_F32` consumes layer-12 attention norm and
  conv-depthwise sidecars `[1,tokens,1024]`, validates
  `encoder.layers.12.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.12.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes
  `w2v_layer12_conv_residual.f32`, and leaves layer-12 ffn2 plus encoder layers
  13-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-12 ffn2 residual boundary.
  `mit2_tts --clone-w2v-layer12-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER12_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN2_RESIDUAL_F32`
  consumes layer-12 conv residual `[1,tokens,1024]`, validates
  `encoder.layers.12.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.12.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.12.ffn2.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, output projection, and the second 0.5 residual through resident
  Metal when available, writes `w2v_layer12_ffn2_residual.f32`, and leaves
  encoder layers 13-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 ffn1 residual boundary.
  `mit2_tts --clone-w2v-layer13-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER12_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN1_RESIDUAL_F32`
  consumes layer-12 ffn2 residual `[1,tokens,1024]`, validates
  `encoder.layers.13.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.13.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.13.ffn1.output_dense.{weight,bias}`, applies LayerNorm, SiLU
  feed-forward, output projection, and the first 0.5 residual through resident
  Metal when available, writes `w2v_layer13_ffn1_residual.f32`, and leaves
  layer-13 self-attention, convolution, ffn2, plus encoder layers 14-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 self-attention Q/K/V boundary.
  `mit2_tts --clone-w2v-layer13-qkv MODEL_BUNDLE_DIR
  W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR` validates
  `encoder.layers.13.self_attn.linear_{q,k,v}.weight`, applies the three
  1024-wide self-attention projections through resident Metal when available,
  writes `w2v_layer13_q.f32`, `w2v_layer13_k.f32`, `w2v_layer13_v.f32`, and
  `w2v_layer13_qkv.manifest.json`, and leaves layer-13 attention
  context/projection/residual/norm, convolution, ffn2, plus encoder layers
  14-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 attention context boundary.
  `mit2_tts --clone-w2v-layer13-attention W2V_LAYER13_Q_F32
  W2V_LAYER13_K_F32 W2V_LAYER13_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS
  OUTPUT_LAYER13_CONTEXT_F32` consumes layer-13 Q/K/V sidecars plus the
  attention mask, applies 16-head masked scaled dot-product attention through
  Metal when available, writes `w2v_layer13_context.f32` `[1,tokens,1024]`,
  and leaves layer-13 attention output projection/residual/norm, convolution,
  ffn2, plus encoder layers 14-16 as the remaining native W2V-BERT encoder
  gap.
- Close the W2V-BERT layer-13 attention output projection boundary.
  `mit2_tts --clone-w2v-layer13-attention-project MODEL_BUNDLE_DIR
  W2V_LAYER13_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_F32` validates
  `encoder.layers.13.self_attn.linear_out.weight`, applies the 1024-wide
  attention output projection through resident Metal when available, writes
  `w2v_layer13_attention.f32`, and leaves layer-13 attention residual/norm,
  convolution, ffn2, plus encoder layers 14-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 attention residual boundary.
  `mit2_tts --clone-w2v-layer13-attention-residual
  W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_LAYER13_ATTENTION_F32 W2V_TOKENS
  OUTPUT_LAYER13_ATTENTION_RESIDUAL_F32` consumes layer-13 ffn1 residual and
  attention projection sidecars `[1,tokens,1024]`, adds them through resident
  Metal when available, writes `w2v_layer13_attention_residual.f32`, and leaves
  layer-13 attention norm, convolution, ffn2, plus encoder layers 14-16 as the
  remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 attention LayerNorm boundary.
  `mit2_tts --clone-w2v-layer13-attention-norm MODEL_BUNDLE_DIR
  W2V_LAYER13_ATTENTION_RESIDUAL_F32 W2V_TOKENS
  OUTPUT_LAYER13_ATTENTION_NORM_F32` validates
  `encoder.layers.13.self_attn_layer_norm.{weight,bias}`, applies the
  1024-wide LayerNorm through resident Metal when available, writes
  `w2v_layer13_attention_norm.f32`, and leaves layer-13 convolution-module
  LayerNorm, convolution GLU/depthwise/residual, ffn2, plus encoder layers
  14-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 convolution-module LayerNorm boundary.
  `mit2_tts --clone-w2v-layer13-conv-norm MODEL_BUNDLE_DIR
  W2V_LAYER13_ATTENTION_NORM_F32 W2V_TOKENS
  OUTPUT_LAYER13_CONV_NORM_F32` validates
  `encoder.layers.13.conv_module.layer_norm.{weight,bias}`, applies the
  1024-wide row LayerNorm through resident Metal when available, writes
  `w2v_layer13_conv_norm.f32`, and leaves layer-13 convolution GLU,
  depthwise/residual, ffn2, plus encoder layers 14-16 as the remaining native
  W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 convolution GLU boundary.
  `mit2_tts --clone-w2v-layer13-conv-glu MODEL_BUNDLE_DIR
  W2V_LAYER13_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_GLU_F32` validates
  `encoder.layers.13.conv_module.pointwise_conv1.{weight,bias}`, applies the
  native 1024-to-2048 projection plus GLU split through resident Metal when
  available, writes `w2v_layer13_conv_glu.f32`, and leaves layer-13 depthwise
  convolution/residual, ffn2, plus encoder layers 14-16 as the remaining
  native W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 causal depthwise convolution boundary.
  `mit2_tts --clone-w2v-layer13-conv-depthwise MODEL_BUNDLE_DIR
  W2V_LAYER13_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_DEPTHWISE_F32`
  validates `encoder.layers.13.conv_module.depthwise_conv.{weight,bias}`,
  applies causal depthwise Conv1d through resident Metal when available,
  writes `w2v_layer13_conv_depthwise.f32`, and leaves layer-13 convolution
  residual, ffn2, plus encoder layers 14-16 as the remaining native W2V-BERT
  encoder gap.
- Close the W2V-BERT layer-13 convolution residual boundary.
  `mit2_tts --clone-w2v-layer13-conv-residual MODEL_BUNDLE_DIR
  W2V_LAYER13_ATTENTION_NORM_F32 W2V_LAYER13_CONV_DEPTHWISE_F32 W2V_TOKENS
  OUTPUT_LAYER13_CONV_RESIDUAL_F32` validates
  `encoder.layers.13.conv_module.depthwise_layer_norm.{weight,bias}` and
  `encoder.layers.13.conv_module.pointwise_conv2.{weight,bias}`, applies
  depthwise LayerNorm, SiLU, pointwise projection, and the attention-norm
  residual add through resident Metal when available, writes
  `w2v_layer13_conv_residual.f32`, and leaves layer-13 ffn2 plus encoder
  layers 14-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-13 ffn2 half-residual boundary.
  `mit2_tts --clone-w2v-layer13-ffn2-residual MODEL_BUNDLE_DIR
  W2V_LAYER13_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN2_RESIDUAL_F32`
  validates `encoder.layers.13.ffn2_layer_norm.{weight,bias}`,
  `encoder.layers.13.ffn2.intermediate_dense.{weight,bias}`, and
  `encoder.layers.13.ffn2.output_dense.{weight,bias}`, applies LayerNorm,
  SiLU feed-forward, output projection, and the second 0.5 residual through
  resident Metal when available, writes `w2v_layer13_ffn2_residual.f32`, and
  leaves encoder layers 14-16 as the remaining native W2V-BERT encoder gap.
- Close the W2V-BERT layer-14 ffn1 half-residual boundary.
  `mit2_tts --clone-w2v-layer14-ffn1-residual MODEL_BUNDLE_DIR
  W2V_LAYER13_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN1_RESIDUAL_F32`
  validates `encoder.layers.14.ffn1_layer_norm.{weight,bias}`,
  `encoder.layers.14.ffn1.intermediate_dense.{weight,bias}`, and
  `encoder.layers.14.ffn1.output_dense.{weight,bias}`, applies LayerNorm,
  SiLU feed-forward, output projection, and the first 0.5 residual through
  resident Metal when available, writes `w2v_layer14_ffn1_residual.f32`, and
  leaves layer-14 self-attention/convolution/ffn2 plus encoder layers 15-16 as
  the remaining native W2V-BERT encoder gap.
- Close the layer-17 final LayerNorm boundary.
  `mit2_tts --clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32
  W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32` now validates
  `encoder.layers.17.final_layer_norm.{weight,bias}`, applies the final encoder
  LayerNorm through resident Metal when available, and writes
  `w2v_hidden_state_17.f32` for `--clone-w2v-normalize`. The remaining W2V-BERT
  encoder work is layer-14 self-attention/convolution/ffn2 plus layers 15-16
  to produce the layer-16 sidecar.
- Establish the CAMPPlus style parity target. `generate_campplus_golden` now
  runs the PyTorch CAMPPlus reference from a native clone feature manifest's
  fbank sidecar and writes `s2mel_style.f32` plus the full FCM head
  intermediates `campplus_head_conv1_bn_relu.f32` and
  `campplus_head_layer1.f32`, `campplus_head_layer2.f32`,
  `campplus_head_conv2_bn_relu.f32`, `campplus_head_output.f32`, `campplus_xvector_tdnn.f32`,
  `campplus_xvector_block1_tdnnd1.f32`, and
  `campplus_xvector_block1_after_tdnnd1.f32`,
  `campplus_xvector_block1_tdnnd2.f32`, and
  `campplus_xvector_block1_after_tdnnd2.f32`,
  `campplus_xvector_block1_tdnnd3.f32`, and
  `campplus_xvector_block1_after_tdnnd3.f32`,
  `campplus_xvector_block1_tdnnd4.f32`, and
  `campplus_xvector_block1_after_tdnnd4.f32`,
  `campplus_xvector_block1_tdnnd5.f32`, and
  `campplus_xvector_block1_after_tdnnd5.f32`,
  `campplus_xvector_block1_tdnnd6.f32`, and
  `campplus_xvector_block1_after_tdnnd6.f32`,
  `campplus_xvector_block1_tdnnd7.f32`, and
  `campplus_xvector_block1_after_tdnnd7.f32`,
  `campplus_xvector_block1_tdnnd8.f32`, and
  `campplus_xvector_block1_after_tdnnd8.f32`,
  `campplus_xvector_block1_tdnnd9.f32`, and
  `campplus_xvector_block1_after_tdnnd9.f32`,
  `campplus_xvector_block1_tdnnd10.f32`, and
  `campplus_xvector_block1_after_tdnnd10.f32`,
  `campplus_xvector_block1_tdnnd11.f32`, and
  `campplus_xvector_block1_after_tdnnd11.f32`,
  `campplus_xvector_block1_tdnnd12.f32`, and
  `campplus_xvector_block1_after_tdnnd12.f32`, and
  `campplus_xvector_transit1.f32`,
  `campplus_xvector_block2_tdnnd1.f32`, and
  `campplus_xvector_block2_after_tdnnd1.f32`,
  `campplus_xvector_block2_tdnnd2.f32`, and
  `campplus_xvector_block2_after_tdnnd2.f32`,
  `campplus_xvector_block2_tdnnd3.f32`, and
  `campplus_xvector_block2_after_tdnnd3.f32`,
  `campplus_xvector_block2_tdnnd4.f32`, and
  `campplus_xvector_block2_after_tdnnd4.f32`,
  `campplus_xvector_block2_tdnnd5.f32`, and
  `campplus_xvector_block2_after_tdnnd5.f32`,
  `campplus_xvector_block2_tdnnd6.f32`, and
  `campplus_xvector_block2_after_tdnnd6.f32`,
  `campplus_xvector_block2_tdnnd7.f32`, and
  `campplus_xvector_block2_after_tdnnd7.f32`,
  `campplus_xvector_block2_tdnnd8.f32`, and
  `campplus_xvector_block2_after_tdnnd8.f32`,
  `campplus_xvector_block2_tdnnd9.f32`, and
  `campplus_xvector_block2_after_tdnnd9.f32`,
  `campplus_xvector_block2_tdnnd10.f32`, and
  `campplus_xvector_block2_after_tdnnd10.f32`,
  `campplus_xvector_block2_tdnnd11.f32`, and
  `campplus_xvector_block2_after_tdnnd11.f32`,
  `campplus_xvector_block2_tdnnd12.f32`, and
  `campplus_xvector_block2_after_tdnnd12.f32`,
  `campplus_xvector_block2_tdnnd13.f32`, and
  `campplus_xvector_block2_after_tdnnd13.f32`,
  `campplus_xvector_block2_tdnnd14.f32`,
  `campplus_xvector_block2_after_tdnnd14.f32`,
  `campplus_xvector_block2_tdnnd15.f32`, and
  `campplus_xvector_block2_after_tdnnd15.f32`,
  `campplus_xvector_block2_tdnnd16.f32`, and
  `campplus_xvector_block2_after_tdnnd16.f32`,
  `campplus_xvector_block2_tdnnd17.f32`, and
  `campplus_xvector_block2_after_tdnnd17.f32`,
  `campplus_xvector_block2_tdnnd18.f32`, and
  `campplus_xvector_block2_after_tdnnd18.f32`,
  `campplus_xvector_block2_tdnnd19.f32`, and
  `campplus_xvector_block2_after_tdnnd19.f32`,
  `campplus_xvector_block2_tdnnd20.f32`, and
  `campplus_xvector_block2_after_tdnnd20.f32`,
  `campplus_xvector_block2_tdnnd21.f32`, and
  `campplus_xvector_block2_after_tdnnd21.f32`,
  `campplus_xvector_block2_tdnnd22.f32`, and
  `campplus_xvector_block2_after_tdnnd22.f32`,
  `campplus_xvector_block2_tdnnd23.f32`, and
  `campplus_xvector_block2_after_tdnnd23.f32`,
  `campplus_xvector_block2_tdnnd24.f32`, and
  `campplus_xvector_block2_after_tdnnd24.f32`, and
  `campplus_xvector_transit2.f32`, and
  `campplus_xvector_block3_tdnnd1.f32`, and
  `campplus_xvector_block3_after_tdnnd1.f32`, and
  `campplus_xvector_block3_tdnnd2.f32`, and
  `campplus_xvector_block3_after_tdnnd2.f32`, and
  `campplus_xvector_block3_tdnnd3.f32`, and
  `campplus_xvector_block3_after_tdnnd3.f32`,
  `campplus_xvector_block3_tdnnd4.f32`, and
  `campplus_xvector_block3_after_tdnnd4.f32`,
  `campplus_xvector_block3_tdnnd5.f32`, and
  `campplus_xvector_block3_after_tdnnd5.f32`,
  `campplus_xvector_block3_tdnnd6.f32`, and
  `campplus_xvector_block3_after_tdnnd6.f32`,
  `campplus_xvector_block3_tdnnd7.f32`, and
  `campplus_xvector_block3_after_tdnnd7.f32`,
  `campplus_xvector_block3_tdnnd8.f32`, and
  `campplus_xvector_block3_after_tdnnd8.f32`,
  `campplus_xvector_block3_tdnnd9.f32`, and
  `campplus_xvector_block3_after_tdnnd9.f32`,
  `campplus_xvector_block3_tdnnd10.f32`, and
  `campplus_xvector_block3_after_tdnnd10.f32`,
  `campplus_xvector_block3_tdnnd11.f32`, and
  `campplus_xvector_block3_after_tdnnd11.f32`,
  `campplus_xvector_block3_tdnnd12.f32`, and
  `campplus_xvector_block3_after_tdnnd12.f32`,
  `campplus_xvector_block3_tdnnd13.f32`, and
  `campplus_xvector_block3_after_tdnnd13.f32`,
  `campplus_xvector_block3_tdnnd14.f32`, and
  `campplus_xvector_block3_after_tdnnd14.f32`,
  `campplus_xvector_block3_tdnnd15.f32`, and
  `campplus_xvector_block3_after_tdnnd15.f32`,
  `campplus_xvector_block3_tdnnd16.f32`, and
  `campplus_xvector_block3_after_tdnnd16.f32`,
  `campplus_xvector_transit3.f32`, `campplus_xvector_out_nonlinear.f32`,
  `campplus_xvector_stats.f32`, and `campplus_xvector_dense.f32`; `mit2_tts
  --clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST
  S2MEL_STYLE_F32` validates the CAMPPlus model contract, feature fbank
  readiness, and `[1,192]` style sidecar before the broader encoder-output
  readiness gate. `mit2_tts --clone-campplus-style-from-features
  MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32` now runs the same
  native CAMPPlus fbank-to-style forward as a product sidecar command and
  writes `[1,192]` `s2mel_style` for the later voice-bundle writer. `mit2_tts
  --clone-campplus-head-golden MODEL_BUNDLE_DIR
  FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR` now replays the native fbank through the
  first CAMPPlus `conv1 -> bn1 -> relu` boundary and the two-block `layer1`
  and `layer2` residual downsampling boundaries, final head `conv2 -> bn2 ->
  relu`, and contiguous reshape to the `[1,320,T]` xvector input from native
  bundle weights, then runs the xvector TDNN Conv1d/BatchNorm/ReLU boundary.
  A real `voice_a7bd52e4.wav` fixture produced 1095 fbank frames, style SHA-256
  `88863411d34ee601e9b650b79cd885103fd56d68b843bb6f9928fc299853c998`, head
  SHA-256 `df100a0fb0d1a7d17ce256b4f5383723f0857cde56b2c0f3f156644aaf79bf6b`,
  layer1 SHA-256 `0f3da379997b8bddc9abedff97f9adcf3cf088e87949146c6916ef7abd531147`,
  layer2 SHA-256 `baa5a4c040c692dae49f6ecb439ec082529eb575ef8efd72b528af114a11f062`,
  head-output SHA-256 `f801366cb9ac813c590299aba65e5b70fa93be1c786a909294322fa41120062f`,
  TDNN SHA-256 `c8d8c6db1b7649cfd99b92fc971a1a19266d89835b3723ae13804f74b05fb51a`,
  native `conv1_bn_relu` max absolute error `1.90735e-06`, and native `layer1`
  max absolute error `1.52588e-05`, native `layer2` max absolute error
  `1.90735e-05`, and native final head-output max absolute error `1.23978e-05`
  plus native xvector TDNN max absolute error `1.48267e-06`, block1.tdnnd1 SHA-256 `127a7686d0635547ff5ff500130e04ae47b4d8726b7342d0aa43ce0b4d6a985e`, block1-after-tdnnd1 SHA-256 `229c68466a1db33e2e12d1502a1b2d01d45edd14dd384d28486ba1e67d074c5e`, native block1.tdnnd1 max absolute error `4.52995e-06`, block1.tdnnd2 SHA-256 `ecb3bf378fd5f7d45002e92aa731a45baa8fcc0a76326a91a14c055cb6cc7c88`, block1-after-tdnnd2 SHA-256 `e740dc5e4c397e0897e2541714631ca475aee7624e0c61efc466491ba3c4cd68`, native block1.tdnnd2 max absolute error `4.05312e-06`, block1.tdnnd3 SHA-256 `3f74909daf437c2f8f680f0884c1cc7726c408d5b22013c156d7ce3f4574d8f8`, block1-after-tdnnd3 SHA-256 `a1b660470385c152ed826f41c8d96f277331b4afccaeeff4cd5279751d48170f`, native block1.tdnnd3 max absolute error `5.24521e-06`, block1.tdnnd4 SHA-256 `81e2d51ac57eb2fe6988a1d94df95288f65ab78c5d8613d9ca0cdbc5d33f35b0`, block1-after-tdnnd4 SHA-256 `2d3f13a413bf8335a1410e768842fe2a6948f4697d2b7e0b85e5af6e92c8d8ef`, native block1.tdnnd4 max absolute error `5.00679e-06`, block1.tdnnd5 SHA-256 `92bc9485ad0fe5a37bddf4866b5ce90145557959facef7017ef99a8ba77736fc`, block1-after-tdnnd5 SHA-256 `1a4183bbda6d4a60ac6a4502bfdc2164715c640b21d0ead2947eeed3efaf54b1`, native block1.tdnnd5 max absolute error `3.57628e-06`, block1.tdnnd6 SHA-256 `cd3d2a7bfad5d50759a4b766d5bc0764bb54e81b3700d3e8662525a9fc21de07`, block1-after-tdnnd6 SHA-256 `3b3a719bb3fc5c0b1faebb6cd48dfadfef9a8ebd4a19e39fdf676d0b692cedd5`, native block1.tdnnd6 max absolute error `5.24521e-06`, block1.tdnnd7 SHA-256 `e0d0900eaf749a30d7bd1aaeb4488dccb46fad978b73d69ccaf47cf10fae7c56`, block1-after-tdnnd7 SHA-256 `49ea507dd3509d389b7a590f703ee26626a3b18eb937d214d44ec23f3bbe63fc`, native block1.tdnnd7 max absolute error `6.4373e-06`, block1.tdnnd8 SHA-256 `072b12dac6bd53cc1b4963b79283ce46c271c0dff88e253c6e1e86357b044255`, block1-after-tdnnd8 SHA-256 `d600ec0d1bcad688a9693a66f86f539c88fdcdde66f60fffd619c3ef2971c59e`, native block1.tdnnd8 max absolute error `6.19888e-06`, block1.tdnnd9 SHA-256 `ce6ec375a908eab42a7b55ca968ba4a3b146b84dfd874034fa4993d5339bf31d`, block1-after-tdnnd9 SHA-256 `2f9c66c6f8cfcfddfdcf3a8c69df91a897ecb5c1cb3de0e799c98ae3c7c8c954`, native block1.tdnnd9 max absolute error `7.62939e-06`, block1.tdnnd10 SHA-256 `e9e6dd40019167ddb36fca1f91c876c30030bd06f5b36364ece60c20633e14af`, block1-after-tdnnd10 SHA-256 `e91ca5ed9967f72c2f2793ee231b3ff3497694a8ada564a6b0a2a7806c30e522`, native block1.tdnnd10 max absolute error `1.43051e-05`, block1.tdnnd11 SHA-256 `7d4ab602dcd26536b8f7811fe1d91bcee059d15bd1ac70863de692630b5d9a81`, block1-after-tdnnd11 SHA-256 `e8f41ffdbb1ac7a79849ec15de93e61d0efb2b44f9f3312d3a97d3f99ea20b0e`, native block1.tdnnd11 max absolute error `7.7486e-06`, block1.tdnnd12 SHA-256 `bc377d6075694b6cafd275ec9cffbb1cf606f11b9bb7518466f8f3544189f958`, block1-after-tdnnd12 SHA-256 `7eafc5d777cc5365f7160d13da5cbf13de028fb74a39dbc4bc77982ba11e206e`, native block1.tdnnd12 max absolute error `6.4373e-06`, transit1 SHA-256 `a78d967ca7d33b3a369b2862a1fa72c3f6783d5aab05d5edd86d7a174fd454b8`, native transit1 max absolute error `3.62396e-05`, block2.tdnnd1 SHA-256 `2455bcfe4ebe44ec63c31719d611440ad7c67aeb32b5b3d5415cf547d6489867`, block2-after-tdnnd1 SHA-256 `8097e9ae0ac1b9a04f3ded20aa947ef32e5b3ddd587bb7d0ee7ddc623f09e0e2`, native block2.tdnnd1 max absolute error `5.24521e-06`, native block2-after-tdnnd1 max absolute error `3.62396e-05`, block2.tdnnd2 SHA-256 `afba868b9d969446904654d36a8f34688c0a26782490f70f9e91f8de770a2070`, block2-after-tdnnd2 SHA-256 `38c111eab44007003e65f4dd09e6d573da278d5230222bf410e23e575f9a6210`, native block2.tdnnd2 max absolute error `4.76837e-06`, and native block2-after-tdnnd2 max absolute error `3.62396e-05` against PyTorch.
  The same fixture now also records block2.tdnnd3 SHA-256 `0577f35fbb900ce32c1704882599703f52aead31dc0d41a747e86aa17658d3f9`, block2-after-tdnnd3 SHA-256 `d622a9248519efd064ba7c540b973afad91c115782614ed85fee9028fb23922c`, native block2.tdnnd3 max absolute error `4.76837e-06`, and native block2-after-tdnnd3 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd4 SHA-256 `bb88519bafe5bb64228c05532a0bb1091a7a0292d2902ffe55c98c17dc63662d`, block2-after-tdnnd4 SHA-256 `57b24b0db7ce214a2c2abffec30aade36bdffe8ac20e0093b90b91ad7c66f9af`, native block2.tdnnd4 max absolute error `4.41074e-06`, and native block2-after-tdnnd4 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd5 SHA-256 `aa989c0a730257a74500aa2c59f5bd45fa720310f6823e0b09aeab3ae592789c`, block2-after-tdnnd5 SHA-256 `ea5edf07b078fb26f476518e5ba6591450bb45ab1455985d6f511f0ad2a895b3`, native block2.tdnnd5 max absolute error `5.72205e-06`, and native block2-after-tdnnd5 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd6 SHA-256 `4f0335237775db51867767a267d3d91828989e08b90b5f5b48094d3f96ce47c6`, block2-after-tdnnd6 SHA-256 `c2f715e16e38cdd51954e0cbf89c53d116ca30bd67b28c483bf70b875f11b1c4`, native block2.tdnnd6 max absolute error `7.86781e-06`, and native block2-after-tdnnd6 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd7 SHA-256 `faae6c70c902b2c61be1008d4184316337b4d3f8c7ce0b09d82ca8b9bcc4e02a`, block2-after-tdnnd7 SHA-256 `31b21e317714691cc1e9c36b8b2e4c246520563921280d3e0be8a6cd57d2fb69`, native block2.tdnnd7 max absolute error `6.67572e-06`, and native block2-after-tdnnd7 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd8 SHA-256 `9e0de6d123e86ed9dc439f100e84756238259011e2eadbc529e858c14bccebb4`, block2-after-tdnnd8 SHA-256 `94803007593da91e5d7d836013b91f644a9fa4071c5de149dcf9444daae73aa4`, native block2.tdnnd8 max absolute error `4.76837e-06`, and native block2-after-tdnnd8 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd9 SHA-256 `63cdc534dccf352ad1fe84c82b13be4a0734ad4035d1d50985f281c23c6a1c59`, block2-after-tdnnd9 SHA-256 `cbc2af43d5f968854e9fd91522c592e080c78d96fa30e234d21d9fef880523f2`, native block2.tdnnd9 max absolute error `4.29153e-06`, and native block2-after-tdnnd9 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd10 SHA-256 `472b2ddbfe883e825998beb16f64594767e137aab4b7893534db6b124217f962`, block2-after-tdnnd10 SHA-256 `b1de5ebe6732028ae5775b00a9fc4de821eec2850fbde5f21907b2f7a46cc425`, native block2.tdnnd10 max absolute error `4.41074e-06`, and native block2-after-tdnnd10 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd11 SHA-256 `16f84aa69d258e25c62a057b6eb8c4ffa3f73ce92c0c78aec5b82d58eca42daf`, block2-after-tdnnd11 SHA-256 `024f084ccc5b16a660589f878616276f7e8a7945eaa7e6a4a2097d0c4516940c`, native block2.tdnnd11 max absolute error `6.91414e-06`, and native block2-after-tdnnd11 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd12 SHA-256 `03fbe593b2f1fef96ed6298e161227c3c35cdf6bc633412e7bb2c420046a7e50`, block2-after-tdnnd12 SHA-256 `7102d73dc11381c7e5bd4821b6a4ffaddf01fa82c69ded6e43d9271322983b99`, native block2.tdnnd12 max absolute error `5.72205e-06`, and native block2-after-tdnnd12 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd13 SHA-256 `78a2bea2035d5e62a6762c8c3895bf5c673a9069cca85f9b88cc40031f606887`, block2-after-tdnnd13 SHA-256 `d3ecd0b39930f6cdb6b66b0e6078a00338ced675c17e5fb737120862fdded48f`, native block2.tdnnd13 max absolute error `5.00679e-06`, and native block2-after-tdnnd13 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd14 SHA-256 `19fea0c57a0b822535ecc43f6fa3eaeb7c719e4345ff0b3036d92e226b0f1412`, block2-after-tdnnd14 SHA-256 `9988d71fbb62e2f4b0534025c349b9a66722b93d8cc0f43c4d5b952f30085773`, native block2.tdnnd14 max absolute error `8.10623e-06`, and native block2-after-tdnnd14 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd15 SHA-256 `caa29ce67ee29a7ff191c36236c1df682fd046e1df14f78e9d40c761b50755a5`, block2-after-tdnnd15 SHA-256 `19e3c967d2041e5a95576b5e1693cbb5175967f2c74fa9c26bdfaa2915cbfa09`, native block2.tdnnd15 max absolute error `6.31809e-06`, and native block2-after-tdnnd15 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd16 SHA-256 `a02293a540ecc3bb18628c290fd8c85e3b040de33f206a4a2422ca2a381a220b`, block2-after-tdnnd16 SHA-256 `784751f04de4633684777a62b3c92aef90a11bda03270ce24c2b014f13e76c81`, native block2.tdnnd16 max absolute error `9.31323e-06`, and native block2-after-tdnnd16 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd17 SHA-256 `3296f935249007404d94129747795ecb99311ae7b5231cdfafad6f8c99b07f00`, block2-after-tdnnd17 SHA-256 `80e57bc1635ad08367ad70d0582005a0428c07135da1007d3a20eed9e6fb781d`, native block2.tdnnd17 max absolute error `4.73857e-06`, and native block2-after-tdnnd17 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd18 SHA-256 `834ac843ce0f0bc8ba3719307d4f0ba7305f4637f6bb5c9dd69e7a08937f44d2`, block2-after-tdnnd18 SHA-256 `58635301755214662d814ffd513a5cca07d95429aed2799598149b2cbf7a8149`, native block2.tdnnd18 max absolute error `9.29832e-06`, and native block2-after-tdnnd18 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd19 SHA-256 `fed8919dbd9dd13492c0f5714d95c88127d00c959a4ad2ec2b1ccf1ed63d4c11`, block2-after-tdnnd19 SHA-256 `18b3f6b6080706973263a76999719f3a494b14b2b533aacf8de65ba2319ed2fb`, native block2.tdnnd19 max absolute error `2.57492e-05`, and native block2-after-tdnnd19 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd20 SHA-256 `83669a623608b332aa5ce7b3023a02d6c54080deca9d838b1da914b264783097`, block2-after-tdnnd20 SHA-256 `8e3cda7f65421dc89285b59c86ee041960ba67f15bdf6c00a789035c18064fb2`, native block2.tdnnd20 max absolute error `5.00679e-06`, and native block2-after-tdnnd20 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd21 SHA-256 `5f834ada00ffcbaaaeb95381d3abb4dc0ba22d72c447c6a4bb18b44cc42b466c`, block2-after-tdnnd21 SHA-256 `9bfdebeb9bc54475e342c11477162e37fd9bba0f50fe7f74662dfa8566749bc5`, native block2.tdnnd21 max absolute error `6.4373e-06`, and native block2-after-tdnnd21 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd22 SHA-256 `e39862966407beceb03a4700399355c394f270998df64c5a5846659d811529ed`, block2-after-tdnnd22 SHA-256 `fe336a4ff02946c9a6730fb525a26dff8fe7016a8819784c39c1876e6c7f0f01`, native block2.tdnnd22 max absolute error `1.00136e-05`, and native block2-after-tdnnd22 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd23 SHA-256 `6ac0c1c499f4a8ced3065c694d3f9d80249742e59b5c7aff904818e89e34e412`, block2-after-tdnnd23 SHA-256 `24ecfda5592784ac648af5f79c8c7952df85562224f68afb61782f4260f05e21`, native block2.tdnnd23 max absolute error `8.16584e-06`, and native block2-after-tdnnd23 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records block2.tdnnd24 SHA-256 `d966c29db4f31ac78e1ed5f4028128ee4fd8d34219dff8a2939ffb8a5ad6356b`, block2-after-tdnnd24 SHA-256 `30b5d50cdfcda0229d97cad0aedf7c8562ca4f43dc93e0cff3e9d34edb7228b7`, native block2.tdnnd24 max absolute error `1.09673e-05`, and native block2-after-tdnnd24 max absolute error `3.62396e-05` against PyTorch.
  The same fixture also records transit2 SHA-256 `14e5bbe8937e2566774898e5d8de955f3bf0071bf4fd25b832bf6d3147d7e72b` and native transit2 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd1 SHA-256 `7b8ee104c697199ee8f94adf1a4f763d0c290c89659286cb9191ca833c8087a5`, block3-after-tdnnd1 SHA-256 `4650137bd373e38abae47b7a5b3d9e92ecc2dd5295124b48ee5e749bf4e23997`, native block3.tdnnd1 max absolute error `4.05312e-06`, and native block3-after-tdnnd1 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd2 SHA-256 `518c0b93d3b0f0be1c9f35fca3ecee191b83ce845b879fcbd10eeb96c0f68fd3`, block3-after-tdnnd2 SHA-256 `b10673ce1be1e7addf604b6faac547376354aba0ed77479c1df57dee05141d05`, native block3.tdnnd2 max absolute error `8.46386e-06`, and native block3-after-tdnnd2 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd3 SHA-256 `cce9ec3619116cbcbf362db6141f728c381e54ec0edbec4c986aa47afbdaf400`, block3-after-tdnnd3 SHA-256 `eb4753db2330eafd264009e4ed8bdd140e81d74ba2987849dffcc4cf69cf0439`, native block3.tdnnd3 max absolute error `1.12057e-05`, and native block3-after-tdnnd3 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd4 SHA-256 `378a2199adf224fc08492c4d252a6b23f25cac6495168de1a470ff9f2d8f61b3`, block3-after-tdnnd4 SHA-256 `3304df3fb00f64bd9ba4faa69bcca9f39615f466df366214ee16d9d5857423da`, native block3.tdnnd4 max absolute error `9.53674e-06`, and native block3-after-tdnnd4 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd5 SHA-256 `eb9b2b34dfd98eedf6d9093443ceeeb21c736c2f807f35826fff717b6493312d`, block3-after-tdnnd5 SHA-256 `36f1e1398bf6c28d4719022c0313f983c0d6c2f030f600cfb40a230f6f9785ee`, native block3.tdnnd5 max absolute error `7.39098e-06`, and native block3-after-tdnnd5 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd6 SHA-256 `34099aec0a9f7f08d77cda979dd85a505b74afc70869c6bfa3ccc2363b0dd5a5`, block3-after-tdnnd6 SHA-256 `ab636f5b771eb32e51e38b0a3e4e0644cd11526515d860cec09155b12bed0bbf`, native block3.tdnnd6 max absolute error `7.62939e-06`, and native block3-after-tdnnd6 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd7 SHA-256 `09d59bcc1db5ae2cf150896ee04fbe2ccd79a64ecabf99a4a6e49fae3b3f8002`, block3-after-tdnnd7 SHA-256 `abc8118cde32754ef877ebc730ba811977ab30c314ba66d37465d392511f70bb`, native block3.tdnnd7 max absolute error `6.67572e-06`, and native block3-after-tdnnd7 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd8 SHA-256 `efd76b8eda715083d984d4a6b7c4f33c5506fe5ef83773a404a98b912b2b9ad8`, block3-after-tdnnd8 SHA-256 `6f53d4769f55f1becdb34dcd577de1010d54039a15e4f66e3117e451b1f6a332`, native block3.tdnnd8 max absolute error `5.42402e-06`, and native block3-after-tdnnd8 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd9 SHA-256 `e867ae4e61414aaf7d67995c678758d9acb5a880f9bb1e379200df0617eafa22`, block3-after-tdnnd9 SHA-256 `36cecaed68c4180eabacaf162778c297e4fffcd017ebfe2f3e20cb46841ed496`, native block3.tdnnd9 max absolute error `1.19209e-05`, and native block3-after-tdnnd9 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd10 SHA-256 `81f71b3862a9f4334641da9c8b6391648245b4831a8ba4c1846a148b9d49fc19`, block3-after-tdnnd10 SHA-256 `bc26fb529d5c3151e634172a17c487333a2fd6d35c41fd3abe11536107a9ebeb`, native block3.tdnnd10 max absolute error `6.19888e-06`, and native block3-after-tdnnd10 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd11 SHA-256 `f665860ac9eadf085523919ca4d712f39091f5200e22476bd78df7195c009cb6`, block3-after-tdnnd11 SHA-256 `4254aad908807dea315d9b9d6cb20deebe66e07d13df3a3ddaa038d4e2390ff0`, native block3.tdnnd11 max absolute error `1.00136e-05`, and native block3-after-tdnnd11 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd12 SHA-256 `3b506482d5726aa487c1d253af8a0b17145fa84d2a99d61b978ef6551b1da57e`, block3-after-tdnnd12 SHA-256 `f39b9c417496bfd898272614c4fdc2f63d81379853c9de3290ffb4eb2f2ca40c`, native block3.tdnnd12 max absolute error `5.00679e-06`, and native block3-after-tdnnd12 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd13 SHA-256 `46d30d31e3bc7242349379ae199c2b9feb631b912894cd87c14d4ea7e3b8adf4`, block3-after-tdnnd13 SHA-256 `b5c4ed9e463b5bb8b008227f57917eebc9a281926499fb0fdea187d6a222c7c8`, native block3.tdnnd13 max absolute error `8.10623e-06`, and native block3-after-tdnnd13 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd14 SHA-256 `1d2efebe3d2dedad6f094319d0dd40d68f7899e58f96ab95383cc376a242d5d1`, block3-after-tdnnd14 SHA-256 `f1f5c395d22bc0ac3f3f719088fa27315ea73c98deafe030904af81e1d3ca3b0`, native block3.tdnnd14 max absolute error `8.82149e-06`, and native block3-after-tdnnd14 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd15 SHA-256 `21ecec974812b840866b9cd55d3928488b683d8bd804f1dcb85bcb46fe9e6540`, block3-after-tdnnd15 SHA-256 `c1e6c3109c2e433693ec4281ce88ea81472c152f3704cc73bf2b76b0622b132f`, native block3.tdnnd15 max absolute error `1.28746e-05`, and native block3-after-tdnnd15 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records block3.tdnnd16 SHA-256 `8d3d12808fa92627b096f5c47240fc5a2316e61f025d0368d24ba26f3f733a0d`, block3-after-tdnnd16 SHA-256 `94a773bcaf88a3d5ff76c018d2d84a3fb511d61d606ad7e17782bc50cfa14640`, native block3.tdnnd16 max absolute error `1.09673e-05`, and native block3-after-tdnnd16 max absolute error `2.86102e-05` against PyTorch.
  The same fixture also records transit3 SHA-256 `92f7c78ca556b963566b6f81119f29c9084bb49b2884d4a981bc4ee38bd6c39d` and native transit3 max absolute error `3.71933e-05` against PyTorch.
  The same fixture now records out-nonlinear SHA-256 `0f11ee6be0009b116b01552c1421892b8e6ab719e345ea48b5659d1c3ac19b64`, stats SHA-256 `f7768d1801ee88f716b3b651c1ac6220f362f9018797c588ec39815e76e4a626`, and dense/style SHA-256 `88863411d34ee601e9b650b79cd885103fd56d68b843bb6f9928fc299853c998`, with native max absolute errors `1.25617e-05`, `4.76837e-07`, and `3.99351e-06` respectively. Native CAMPPlus fbank-to-`s2mel_style` parity and sidecar generation are complete; the remaining upstream clone work is W2V-BERT/semantic feature extraction to produce `spk_cond_emb` from audio.
- Close the semantic-codec quantize sub-boundary once `spk_cond_emb` is
  available. `mit2_tts --clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32
  SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32` now validates the native
  semantic-codec quantizer tensor contract, consumes `spk_cond_emb`
  `[1,tokens,1024]`, runs weight-normalized in/out projection plus 8192-way
  Euclidean codebook selection, writes continuous `S_ref` `[1,tokens,1024]`,
  and emits the selected semantic code indices as raw `u32`. This now prefers a fused Metal nearest-neighbor/projection path and reports `cpu_fallback` only when no Metal device is available for tests.
- Close the S2Mel prompt-condition sub-boundary once `S_ref` is available.
  `mit2_tts --clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST
  S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32` now validates the native
  feature manifest plus the S2Mel length-regulator tensor contract, consumes
  continuous MaskGCT `S_ref` `[1,tokens,1024]`, derives `prompt_tokens` from
  native mel frames, and writes `s2mel_prompt` `[1,mel_frames,512]`. The command
  prefers Metal and reports an explicit `cpu_fallback` backend when no Metal
  device is available for tests. `mit2_tts
  --clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST
  SPK_COND_F32 SPK_TOKENS OUTPUT_DIR` now chains semantic quantize and S2Mel
  prompt-condition into one product sidecar command, writing `s_ref.f32`,
  `semantic_codes.u32`, `s2mel_prompt.f32`, and a manifest from a supplied
  `spk_cond_emb`. The remaining clone gap is now specifically W2V-BERT/semantic
  feature extraction to produce `spk_cond_emb` from clone audio.

Validation:

- Native clone profile matches PyTorch-created profile.
- Same reference audio produces stable voice profile checksum under deterministic preprocessing.
- Hot TTS from native-created profile passes audio checks.

## Phase 15: Sampling, Emotion, and User Controls

Status: partial

Deliverables:

- Add stochastic sampling parity:
  - top-p, top-k, temperature, and repetition penalty are implemented in native seeded sampling,
  - length penalty / beam behavior remain if retained.
- Add emotion vector controls:
  - direct raw `float32[1280]` emovec injection is implemented through
    `--emovec-file` in `generate_gpt_frontend` and `synthesize_native_hot`,
    mutually exclusive with `--native-emovec`, with upfront 5120-byte raw-vector
    validation before model/runtime work,
  - weighted raw emovec mixing is implemented through repeatable
    `--emovec-mix PATH=WEIGHT`, using a float64 accumulator before writing the
    final `float32[1280]` vector, with each component path and finite weight
    validated up front,
  - IndexTTS2/WebUI-style 8-dimensional slider input is implemented through
    `--emotion-vector`, with optional WebUI normalization bypass
    (`--emotion-vector-raw`) and random prototype selection
    (`--emotion-vector-random`); `--emotion-vector-random-seed` makes random
    prototype selection reproducible for benchmark/report comparisons,
  - browser/UI slider wiring remains planned.
- Add emotion audio profile support.
  - cached emotion profile source is implemented through `--emotion-voice-name`,
  - raw emotion reference audio source is implemented through `--emotion-audio`
    for the Python-owned cold extraction boundary,
  - both feed the current emovec merge path and can use native emovec export.
- Keep Qwen emotion text inference optional through Python or a separate native model plan.

Validation:

- Fixed seeded sampling produces reproducible native output.
  `--test-gpt-sampled-inputs-determinism` covers the code-generation boundary
  directly; the current short frontend fixture repeats `[6337, 4568, 4724]` for
  seed `20240605` and reports `[6049, 6947, 4891]` for seed `20240606`.
- Direct emovec-file injection produces a valid native-hot TTS smoke and records
  the injected vector path in the frontend manifest/report.
- Weighted emovec mixing preserves expected tensor math and produces a valid
  native-hot TTS smoke with component paths/weights recorded in the report.
- Emotion-vector mixtures preserve expected tensor math.
- 8-dimensional emotion slider overlay produces valid native-hot TTS smokes,
  records normalized vectors and selected prototype indices, and preserves the
  base emovec for an all-zero slider vector.
- Emotion audio/profile prompt path works from cached profile; raw audio prompt
  extraction is exposed as a cold Python boundary.

## Phase 16: Quantization and Advanced Optimization

Status: planned

Deliverables:

- Profile fp16/fp32 native runtime first.
- Introduce quantization selectively:
  - GPT linear weights,
  - S2Mel transformer weights,
  - BigVGAN conv weights only if quality survives.
- Evaluate formats:
  - fp16,
  - int8,
  - q4 variants inspired by ggml,
  - mixed precision per component.
- Fuse hot kernels only after individual kernels pass tests.

Validation:

- Quantized model has side-by-side quality metrics against fp16/fp32 native and PyTorch.
- Regression suite blocks token drift beyond accepted stochastic boundaries.
- RTF, memory, and quality deltas are tracked per quantization scheme.

## Phase 17: Benchmark and Regression Suite

Status: partial

Deliverables:

- Add repeatable benchmark matrix. Implemented as
  `python -m metal_indextts2.tools.benchmark_native_hot` for the current native
  hot path:
  - short greedy text,
  - short sampled text,
  - all-segments text bridge,
  - native CJK segmented text-id bridge,
  - native CJK mixed alpha+digit single-segment text-id bridge through
    `native_cjk_mixed`,
  - explicit `shared_short_greedy` default-vs-shared resource comparison case,
  - cached emotion profile, raw emotion audio, and 8-dimensional emotion vector
    cases through `emotion_profile`, `emotion_audio`, and `emotion_vector`,
  - full-acoustic budget cases through `short_acoustic_full`,
    `medium_acoustic_full`, `long_acoustic_full_segments`, and
    `clone_acoustic_full`,
  - medium greedy text,
  - long all-segments text bridge,
  - fresh clone smoke when `--clone-audio` is provided,
  - existing-report summarization through `--from-reports`.
  Benchmark execution now forwards `--validate-model-contract`,
  `--validate-voice-source`, and `--validate-voice-contract` to the synthesis
  CLI. Clone benchmark cases intentionally omit the benchmark's default
  `--voice-bundle`, so `clone` and `clone_acoustic_full` measure a freshly
  cloned and converted profile instead of accidentally reusing the cached
  speaker bundle. `--python-executable` lets the benchmark wrapper run under a
  lightweight Python while launching `synthesize_native_hot` under the
  IndexTTS2/Torch environment. Benchmark summaries now extract
  `model_contract_ok`, model tensor and integrity counters,
  `voice_validation_ok`, `voice_contract_ok`, and native voice contract
  token/byte/integrity counters from synthesis reports. They also extract GPT
  frontend/control fields such as `gpt_emovec_source`, `gpt_emotion_source`,
  `gpt_emotion_alpha`, `gpt_emotion_vector`,
  `gpt_emotion_vector_indices`, `gpt_emotion_vector_random`,
  `gpt_emotion_vector_random_seed`, `gpt_subsampling_source`,
  `gpt_conformer_source`, `gpt_perceiver_source`,
  `gpt_frontend_tail_source`, `gpt_emovec_mix`, and
  `generation_shared_runtime_stages`, plus GPT frontend tensor dtype/shape/SHA
  fields such as `gpt_conds_latent_sha256`,
  `gpt_speech_conditioning_latent_sha256`, `gpt_text_ids_tensor_sha256`,
  `gpt_fake_inputs_sha256`, `gpt_inputs_embeds_sha256`, and
  `gpt_attention_mask_sha256` when synthesis reports include those summaries.
  When older synthesis reports only contain `gpt_frontend_dir`, the benchmark
  summary backfills missing frontend tensor summaries from
  `gpt_frontend/manifest.json`, resolving relative frontend paths from the
  report JSON directory so existing artifacts can be covered by the same hash
  comparison. For `--all-segments` reports, summary extraction also
  promotes per-segment GPT frontend tensor/control lists such as
  `segment_gpt_conds_latent_sha256`,
  `segment_gpt_speech_conditioning_latent_sha256`,
  `segment_gpt_text_ids_tensor_sha256`, and
  `segment_gpt_inputs_embeds_sha256`. Summary extraction also hashes the native
  hot sidecar artifacts referenced by synthesis reports (`codes_u32`,
  `condition_f32`, and `noise_f32`) and records their byte counts plus SHA-256
  values, including per-segment `segment_*_sha256` lists for `--all-segments`
  reports.
  Baseline comparisons exact-match the model/voice contract byte counts,
  tensor counts, prompt-token counters, SHA/integrity counters, GPT
  emotion/native-frontend control fields, single- and multi-segment frontend
  tensor hashes, native runtime stage counts, and hot-path sidecar hashes when
  present, so converted bundle layout, frontend payload drift, single- or
  multi-segment native code/condition/noise drift, accidental control path
  changes, or scheduling stage-count changes fail regression checks instead of
  only surfacing as a later audio difference.
  Absolute benchmark budgets
  (`--max-runtime-peak-bytes`, `--max-runtime-command-buffers-submitted`,
  `--max-runtime-buffer-allocations`, `--max-runtime-buffer-bytes-allocated`,
  `--max-runtime-gpu-elapsed-seconds`, `--max-rtf`, `--max-elapsed-seconds`,
  `--min-audio-samples-per-second`) now mark violating cases failed and make
  the CLI exit non-zero without requiring a baseline summary.
  A medium/long production-input smoke has been run with short acoustic output
  (`max_mel_tokens=1`, `steps=1`) to validate the real text path and long
  all-segments bridge. Short cached-voice, medium tokenizer text, long
  segmented tokenizer text, and clone 25-step full-acoustic benchmark baseline
  artifacts have been created.
- Track:
  - RTF when elapsed time and WAV duration are available,
  - predicted GPT codes,
  - segment predicted codes for multi-segment reports,
  - text-id source boundary for tokenizer/native-CJK cases,
  - output WAV SHA-256,
  - peak native runtime RSS,
  - runtime stage counts,
  - GPT code count,
  - generated mel frame count,
  - audio sample count,
  - wrapper-level voice prep, segmentation, frontend, native runtime,
    postprocess, and concat timing fields when reports include them,
  - native hot GPT code export, condition export, noise, acoustic, and total
    runtime phase timing when runtime JSON includes them,
  - native runtime command-buffer submissions, buffer allocations, cumulative
    allocated bytes, and GPU elapsed seconds when runtime JSON includes them,
  - whole-case GPT codes/sec, mel frames/sec, and audio samples/sec when
    elapsed time is available,
  - per-case pass/fail status.
  Kernel-level GPT/S2Mel/BigVGAN timing beyond native hot phase timing remains
  planned.

Validation:

- Local script writes `summary.json` in
  `mit2-native-hot-benchmark-summary` format. It was validated by summarizing
  existing short greedy, sampled, all-segments, and clone synthesis reports
  into `artifacts/native_hot_benchmark_summary_smoke/summary.json`.
- The executed production-input smoke writes
  `artifacts/native_hot_benchmark_prod_input_smoke/summary.json` for
  `medium_greedy` and `long_all_segments`, with real medium/long text inputs,
  short acoustic output (`max_mel_tokens=1`, `steps=1`), and native phase timing
  plus memory counters.
- The native CJK segmented benchmark summary writes
  `artifacts/native_hot_benchmark_native_cjk_segments_smoke/summary.json` from
  the existing `artifacts/native_text_all_segments_native_cjk_smoke/report.json`
  and compares `text_ids_source: native_cjk_segments` in
  `comparison.json`.
- Runtime command-buffer submissions, buffer allocations, cumulative allocated
  bytes, and command-buffer GPU elapsed seconds are aggregated into
  `runtime_command_buffers_submitted`, `runtime_buffer_allocations`,
  `runtime_buffer_bytes_allocated`, and `runtime_gpu_elapsed_seconds`, with
  per-stage and per-field breakdowns when runtime JSON includes matching
  resource fields. When both `short_greedy` and `shared_short_greedy` are in
  one summary, `resource_comparisons` records default-vs-shared delta/ratio
  metrics. Baseline comparisons treat increases outside the elapsed-regression
  threshold as regressions.
- `--baseline-summary` now compares the current summary against a previous
  JSON, writes `comparison.json`, and exits non-zero on deterministic WAV hash,
  predicted-code, segment-code, RTF/elapsed, runtime-peak RSS, or whole-case
  throughput regressions outside the configured thresholds. It was validated
  against
  `artifacts/native_hot_benchmark_summary_smoke/summary.json` with
  `comparison.failed=0`.
- Absolute benchmark budgets were validated with
  `artifacts/native_hot_benchmark_budget_pass_smoke/summary.json`
  (`budget_failed=0`) and an intentionally impossible runtime RSS budget in
  `artifacts/native_hot_benchmark_budget_fail_smoke/summary.json`
  (`budget_failed=1`, `failed=1`, non-zero CLI exit).
- Speed regressions require explicit acceptance through the configured
  threshold flags.
- Correctness regressions fail by default unless the caller disables hash/code
  matching for explicitly stochastic cases.

## Phase 18: Packaging and Developer Workflow

Status: planned

Deliverables:

- Build commands for:
  - converter,
  - runtime,
  - tests,
  - benchmarks.
- Document hardware assumptions:
  - Apple Silicon,
  - macOS version,
  - memory budget,
  - Xcode/Metal tools.
- Provide minimal examples:
  - convert model,
  - convert voice,
  - synthesize wav,
  - run benchmark.

Validation:

- A clean machine can follow docs from model download to generated wav.
- All paths are configurable and do not depend on the current working directory.

## Tracking Checklist

- [x] Baseline profiling harness exists.
- [x] Golden corpus generator exists.
- [x] Native bundle spec is finalized.
- [x] Model converter validates all tensor names/shapes/checksums.
- [x] Metal runtime skeleton builds.
- [x] Primitive kernel tests pass.
- [ ] GPT greedy decode matches reference. Short synthetic no-KV/KV-cached CPU-vs-Metal parity gates, short, 16-token, max-64/30-step, and medium-text 96-code PyTorch generated-code gates now pass, with the medium inputs path compared inside the native runtime through `--test-gpt-kv-codes-inputs`; broader production-length generated-code validation, beam behavior, and PyTorch RNG parity remain.
- [x] GPT latent forward matches reference. Short, 16-token, max-64/30-code, and medium-text 96-code PyTorch golden parity passes for generated-code latent, resident-Metal `gpt_layer`, resident-Metal `vq2emb`, and `S_infer`, with GPT dense projections resident/batched, 1280-wide GPT LayerNorm using resident parameters plus the high-width serial-row Metal branch, `S_infer` add running through Metal `add_f32`, and prompt/LR condition assembly through Metal `hot_condition_merge_f32`; full production-length S2Mel golden comparison remains.
- [x] Semantic `vq2emb` hot subset matches reference.
- [x] Length regulator matches reference.
- [ ] DiT estimator and CFM solver match reference. PyTorch golden parity passes for a short DiT estimator step, no-CFG and CFG CFM Euler fixtures, a voice-conditioned short-prompt S2Mel full-mel fixture, a max-64/30-code CPU-LR 59-frame S2Mel full-mel fixture, and a medium-text 173-frame S2Mel full-mel fixture; the short, max-64/30-code CPU-LR, and medium full-mel boundaries also pass the explicit Metal-only `--test-s2mel-full-inputs` gate. CPU-vs-Metal synthetic tests pass for unbatched no-CFG and 25-step CFG solver paths, with DiT timestep MLPs, adaptive RMSNorm rows, batched CFG input merge, Wavenet reflect convolution/gate/residual/skip updates, final adaptive output LayerNorm/modulation/linear, shared per-step CFG timestep embeddings, batched branch-isolated CFG Wavenet scheduling, and post-estimator CFG/Euler update running through Metal primitives in the Metal estimator/solver path. Full production-length prompt/full-mel PyTorch golden comparison and production resource planning remain.
- [x] BigVGAN matches reference. Native BigVGAN generator weight bundling, `conv_pre`, complete ConvTranspose upsampler stack, composed front path, alias-free SnakeBeta activation, complete first AMP residual block, first 3-block residual group with Metal `avg3_f32` averaging, complete 18-block upsample/resblock body, post-conv, final clamp/tanh, synthetic mel-to-waveform CPU-vs-Metal parity, and PyTorch waveform golden comparison for synthetic and S2Mel-generated mel are implemented.
- [ ] End-to-end cached voice TTS works. Same-process short native hot-path WAV smoke from golden fixture inputs, same-process from-codes-to-waveform with Metal condition merge, GPT-derived acoustic hot-path golden gates with Metal condition merge, medium-text from-codes cond/text/codes-to-waveform validation, artifact-level native GPT-code-to-condition-to-waveform/WAV validation, golden-assisted native WAV output, the hybrid text-to-native-hot CLI, all-segments reference-tokenizer bridge, narrow native CJK/limited-ASCII plus scored ASCII, simple-digit, focused standalone 4-digit cardinal, focused unary-minus CJK number/temperature, uppercase-alpha number-word, fullwidth-punctuation, fullwidth-alphanumeric, fullwidth slash punctuation, non-leading-zero 4-digit halfwidth-percent, focused CJK decimal measure, focused operator measure RHS including `1000` RHS, focused fullwidth-percent unknown-id text-id bridge, clone-audio smoke, and machine-readable `mit2_tts` capability/text-readiness/readiness contracts pass. `mit2_tts --capabilities` and `--readiness` now emit `start_sh_replacement_audit`, which marks cached-voice focused-CJK TTS as currently replaceable by the standalone binary and keeps full clone+TTS replacement false with explicit missing surfaces for general text, native raw-audio clone profile creation, and production-length resource planning. `mit2_tts --preflight MODEL_BUNDLE VOICE_BUNDLE TEXT` now combines model/voice contract validation and text readiness into one no-synthesis `tts_product_preflight` gate with `ready_to_synthesize`, token ids, preset segmentation, per-preset ds4-style hot scratch `resource_plan` estimates, and the same audit. `mit2_tts --clone-preflight AUDIO_WAV` now gives the clone path a native PCM16 mono WAV input gate with audio metadata/SHA, peak/RMS/mean, silence/clipping quality metrics, 16 kHz resample target reporting, and invalid/silent-audio rejection. `mit2_tts --clone-preprocess AUDIO_WAV OUTPUT_F32` now writes normalized mono f32 audio resampled to 16 kHz plus `OUTPUT_F32.manifest.json` with source/output SHA, quality, and resample metadata for future native clone encoders; `mit2_tts --clone-readiness PREPROCESS_MANIFEST` validates that sidecar and reports the remaining native voice bundle tensor requirements; `mit2_tts --clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32` now writes the native 22.05 kHz S2Mel reference mel tensor matching the voice-profile `mel` contract; `mit2_tts --clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32` now writes the native Kaldi-style 80-bin CAMPPlus input sidecar from the 16 kHz clone audio; `mit2_tts --clone-prepare-features AUDIO_WAV OUTPUT_DIR` now combines the native quality gate, 16 kHz f32 sidecar, S2Mel mel extraction, CAMPPlus fbank extraction, per-sidecar manifests, and a top-level clone feature manifest into one launcher-facing command; `mit2_tts --clone-feature-readiness FEATURE_MANIFEST` validates that combined feature manifest plus nested preprocess/mel/fbank manifests and raw sidecar SHA/size/frame/sample contracts; `mit2_tts --clone-encoder-readiness FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32` now validates future encoder outputs against the prepared feature manifest by deriving prompt length from native mel frames and checking the three remaining raw f32 tensor size contracts; `mit2_tts --clone-write-voice-bundle PREPROCESS_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS MEL_F32 OUTPUT_VOICE_BUNDLE` writes and read-back-validates native MIT2 voice bundles from the four raw f32 clone tensors; `mit2_tts --clone-write-voice-bundle-from-features FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS OUTPUT_VOICE_BUNDLE` validates feature-prep output and automatically consumes its native mel so future encoders only provide the three remaining raw tensors, while full native clone-time W2V-BERT/semantic feature extraction remains. Full native TextNormalizer/SentencePiece integration, W2V-BERT/semantic feature extraction, full-length acoustic validation, and same-process production resource planning remain.
- [x] Native voice profile conversion works.
- [ ] Clone-time native path works.
- [ ] Sampling and emotion controls work.
- [ ] Quantized variants are evaluated.
- [ ] Benchmark and regression suite is stable.
- [ ] Packaging docs are complete.
