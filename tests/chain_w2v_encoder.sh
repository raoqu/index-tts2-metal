#!/bin/bash
# Integration test: chains W2V-BERT sidecar commands into spk_cond_emb
# Usage: ./tests/chain_w2v_encoder.sh MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_ATTENTION_MASK_U32 TOKENS OUTPUT_SPK_COND_F32
set -euo pipefail

BIN="${MIT2_TTS_BIN:-./build/mtts}"
BUNDLE="$1"
FEATURES="$2"
MASK="$3"
TOKENS="$4"
OUTPUT="$5"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "=== W2V-BERT Encoder Chain ==="
echo "Bundle: $BUNDLE"
echo "Tokens: $TOKENS"

# Feature projection
echo "[1/??] Feature projection..."
$BIN --clone-w2v-feature-project "$BUNDLE" "$FEATURES" "$TOKENS" "$TMP/fp.f32" > /dev/null || { echo "FAIL: feature projection"; exit 1; }

# Layer 0: individual ffn1 steps + attention + conv + ffn2 + final_norm
echo "[?] Layer 0..."
$BIN --clone-w2v-layer0-ffn1-norm "$BUNDLE" "$TMP/fp.f32" "$TOKENS" "$TMP/l0_fn.f32" > /dev/null
$BIN --clone-w2v-layer0-ffn1-intermediate "$BUNDLE" "$TMP/l0_fn.f32" "$TOKENS" "$TMP/l0_fi.f32" > /dev/null
$BIN --clone-w2v-layer0-ffn1-activate "$TMP/l0_fi.f32" "$TOKENS" "$TMP/l0_fa.f32" > /dev/null
$BIN --clone-w2v-layer0-ffn1-output "$BUNDLE" "$TMP/l0_fa.f32" "$TOKENS" "$TMP/l0_fo.f32" > /dev/null
$BIN --clone-w2v-layer0-ffn1-residual "$TMP/fp.f32" "$TMP/l0_fo.f32" "$TOKENS" "$TMP/l0_fr.f32" > /dev/null
$BIN --clone-w2v-layer0-qkv "$BUNDLE" "$TMP/l0_fr.f32" "$TOKENS" "$TMP/l0_qkv" > /dev/null
$BIN --clone-w2v-layer0-attention "$TMP/l0_qkv/q.f32" "$TMP/l0_qkv/k.f32" "$TMP/l0_qkv/v.f32" "$MASK" "$TOKENS" "$TMP/l0_at.f32" > /dev/null
$BIN --clone-w2v-layer0-attention-project "$BUNDLE" "$TMP/l0_at.f32" "$TOKENS" "$TMP/l0_ap.f32" > /dev/null
$BIN --clone-w2v-layer0-attention-residual "$TMP/l0_fr.f32" "$TMP/l0_ap.f32" "$TOKENS" "$TMP/l0_ar.f32" > /dev/null
$BIN --clone-w2v-layer0-attention-norm "$BUNDLE" "$TMP/l0_ar.f32" "$TOKENS" "$TMP/l0_an.f32" > /dev/null
$BIN --clone-w2v-layer0-conv-norm "$BUNDLE" "$TMP/l0_an.f32" "$TOKENS" "$TMP/l0_cn.f32" > /dev/null
$BIN --clone-w2v-layer0-conv-glu "$BUNDLE" "$TMP/l0_cn.f32" "$TOKENS" "$TMP/l0_cg.f32" > /dev/null
$BIN --clone-w2v-layer0-conv-depthwise "$BUNDLE" "$TMP/l0_cg.f32" "$TOKENS" "$TMP/l0_cd.f32" > /dev/null
$BIN --clone-w2v-layer0-conv-residual "$BUNDLE" "$TMP/l0_an.f32" "$TMP/l0_cd.f32" "$TOKENS" "$TMP/l0_cr.f32" > /dev/null
$BIN --clone-w2v-layer0-ffn2-residual "$BUNDLE" "$TMP/l0_cr.f32" "$TOKENS" "$TMP/l0_f2.f32" > /dev/null
$BIN --clone-w2v-layer0-final-norm "$BUNDLE" "$TMP/l0_f2.f32" "$TOKENS" "$TMP/layer0.f32" > /dev/null
echo "  Layer 0 done"

# Layers 1-3
for L in 1 2 3; do
    echo "[?] Layer $L..."
    PREV=$((L-1))
    $BIN --clone-w2v-layer${L}-ffn1-norm "$BUNDLE" "$TMP/layer${PREV}.f32" "$TOKENS" "$TMP/l${L}_fn.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-ffn1-intermediate "$BUNDLE" "$TMP/l${L}_fn.f32" "$TOKENS" "$TMP/l${L}_fi.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-ffn1-activate "$TMP/l${L}_fi.f32" "$TOKENS" "$TMP/l${L}_fa.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-ffn1-output "$BUNDLE" "$TMP/l${L}_fa.f32" "$TOKENS" "$TMP/l${L}_fo.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-ffn1-residual "$TMP/layer${PREV}.f32" "$TMP/l${L}_fo.f32" "$TOKENS" "$TMP/l${L}_fr.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-qkv "$BUNDLE" "$TMP/l${L}_fr.f32" "$TOKENS" "$TMP/l${L}_qkv" > /dev/null
    $BIN --clone-w2v-layer${L}-attention "$TMP/l${L}_qkv/q.f32" "$TMP/l${L}_qkv/k.f32" "$TMP/l${L}_qkv/v.f32" "$MASK" "$TOKENS" "$TMP/l${L}_at.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-attention-project "$BUNDLE" "$TMP/l${L}_at.f32" "$TOKENS" "$TMP/l${L}_ap.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-attention-residual "$TMP/l${L}_fr.f32" "$TMP/l${L}_ap.f32" "$TOKENS" "$TMP/l${L}_ar.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-attention-norm "$BUNDLE" "$TMP/l${L}_ar.f32" "$TOKENS" "$TMP/l${L}_an.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-norm "$BUNDLE" "$TMP/l${L}_an.f32" "$TOKENS" "$TMP/l${L}_cn.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-glu "$BUNDLE" "$TMP/l${L}_cn.f32" "$TOKENS" "$TMP/l${L}_cg.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-depthwise "$BUNDLE" "$TMP/l${L}_cg.f32" "$TOKENS" "$TMP/l${L}_cd.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-residual "$BUNDLE" "$TMP/l${L}_an.f32" "$TMP/l${L}_cd.f32" "$TOKENS" "$TMP/l${L}_cr.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-ffn2-residual "$BUNDLE" "$TMP/l${L}_cr.f32" "$TOKENS" "$TMP/l${L}_f2.f32" > /dev/null
    # final_norm: exists for layers 0,1,3; layer2 skips it
    if [ "$L" = "2" ]; then
        cp "$TMP/l${L}_f2.f32" "$TMP/layer${L}.f32"
    else
        $BIN --clone-w2v-layer${L}-final-norm "$BUNDLE" "$TMP/l${L}_f2.f32" "$TOKENS" "$TMP/layer${L}.f32" > /dev/null
    fi
    echo "  Layer $L done"
done

# Layers 4-16 (simplified path)
for L in $(seq 4 16); do
    echo "[?] Layer $L..."
    PREV=$((L-1))
    $BIN --clone-w2v-layer${L}-ffn1-residual "$BUNDLE" "$TMP/layer${PREV}.f32" "$TOKENS" "$TMP/l${L}_fr.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-qkv "$BUNDLE" "$TMP/l${L}_fr.f32" "$TOKENS" "$TMP/l${L}_qkv" > /dev/null
    $BIN --clone-w2v-layer${L}-attention "$TMP/l${L}_qkv/q.f32" "$TMP/l${L}_qkv/k.f32" "$TMP/l${L}_qkv/v.f32" "$MASK" "$TOKENS" "$TMP/l${L}_at.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-attention-project "$BUNDLE" "$TMP/l${L}_at.f32" "$TOKENS" "$TMP/l${L}_ap.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-attention-residual "$TMP/l${L}_fr.f32" "$TMP/l${L}_ap.f32" "$TOKENS" "$TMP/l${L}_ar.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-attention-norm "$BUNDLE" "$TMP/l${L}_ar.f32" "$TOKENS" "$TMP/l${L}_an.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-norm "$BUNDLE" "$TMP/l${L}_an.f32" "$TOKENS" "$TMP/l${L}_cn.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-glu "$BUNDLE" "$TMP/l${L}_cn.f32" "$TOKENS" "$TMP/l${L}_cg.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-depthwise "$BUNDLE" "$TMP/l${L}_cg.f32" "$TOKENS" "$TMP/l${L}_cd.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-conv-residual "$BUNDLE" "$TMP/l${L}_an.f32" "$TMP/l${L}_cd.f32" "$TOKENS" "$TMP/l${L}_cr.f32" > /dev/null
    $BIN --clone-w2v-layer${L}-ffn2-residual "$BUNDLE" "$TMP/l${L}_cr.f32" "$TOKENS" "$TMP/layer${L}.f32" > /dev/null
    echo "  Layer $L done"
done

# Layer 17 final norm
echo "[?] Layer 17 final norm..."
$BIN --clone-w2v-layer17-final-norm "$BUNDLE" "$TMP/layer16.f32" "$TOKENS" "$TMP/hs17.f32" > /dev/null

# Normalize
echo "[?] Normalize..."
$BIN --clone-w2v-normalize "$BUNDLE" "$TMP/hs17.f32" "$TOKENS" "$OUTPUT" > /dev/null

echo "=== Done: $OUTPUT ==="
