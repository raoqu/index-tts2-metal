"""
Compare native vs Python W2V-BERT intermediate outputs layer by layer.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

NATIVE_DIR = "/tmp/mit2_enc_debug"
MODEL_DIR = "model/w2v-bert-2.0"
FEATURES_F32 = "sample/qin.pt.clone_work/w2v_features.f32"
TOKENS = 364

def load_native(name):
    path = os.path.join(NATIVE_DIR, name)
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, 1024) if data.size % 1024 == 0 else data

def corr(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

def compare(label, native_name, py_tensor):
    native = load_native(native_name)
    if native is None:
        print(f"  MISS {label}: no file {native_name}")
        return 0.0
    py = py_tensor.detach().float().numpy().reshape(-1, 1024) if hasattr(py_tensor, 'detach') else np.array(py_tensor).reshape(-1, 1024)
    c = corr(native, py)
    flag = "OK  " if c > 0.99 else ("WARN" if c > 0.9 else "FAIL")
    max_diff = float(np.abs(native - py).max())
    print(f"  {flag} {label}: corr={c:.4f} max_diff={max_diff:.4f}")
    return c

# Load model
print(f"Loading W2V-BERT model...")
from transformers import Wav2Vec2BertModel, AutoConfig
config = AutoConfig.from_pretrained(MODEL_DIR)
model = Wav2Vec2BertModel.from_pretrained(MODEL_DIR)
model.eval()

# Load features (T, 160)
features = np.fromfile(FEATURES_F32, dtype=np.float32).reshape(TOKENS, 160)
input_tensor = torch.tensor(features).unsqueeze(0)  # [1, T, 160]
mask = torch.ones(1, TOKENS, dtype=torch.long)

with torch.no_grad():
    fp_out = model.feature_projection(input_tensor)
    if isinstance(fp_out, tuple):
        fp_out = fp_out[0]

compare("fp (feature_projection)", "fp.f32", fp_out[0])

encoder = model.encoder
extended_attention_mask = model.get_extended_attention_mask(mask, (1, TOKENS))

layer_prefixes = {
    0: 'a', 1: 'b1', 2: 'b2', 3: 'b3', 4: 'c4',
    5: 'e5', 6: 'e6', 7: 'e7', 8: 'e8', 9: 'e9',
    10: 'e10', 11: 'e11', 12: 'e12', 13: 'e13',
    14: 'f14', 15: 'd15', 16: 'd16'
}

with torch.no_grad():
    hs = fp_out  # [1, T, 1024]

    for layer_idx in range(17):
        layer = encoder.layers[layer_idx]
        pfx = layer_prefixes[layer_idx]

        print(f"\n=== Layer {layer_idx} ===")

        # FFN1: norm is layer.ffn1_layer_norm, not inside layer.ffn1
        h = hs
        fn = layer.ffn1_layer_norm(h)
        compare(f"l{layer_idx} ffn1_norm (fn)", f"{pfx}_fn.f32", fn[0])

        fi = layer.ffn1.intermediate_dense(fn)
        compare(f"l{layer_idx} ffn1_intermediate (fi)", f"{pfx}_fi.f32", fi[0])

        fa = layer.ffn1.intermediate_act_fn(fi)
        compare(f"l{layer_idx} ffn1_activate (fa)", f"{pfx}_fa.f32", fa[0])

        fo = layer.ffn1.output_dense(fa)
        compare(f"l{layer_idx} ffn1_output (fo)", f"{pfx}_fo.f32", fo[0])

        fr = h + 0.5 * fo  # ffn1_residual
        compare(f"l{layer_idx} ffn1_residual (fr)", f"{pfx}_fr.f32", fr[0])

        # Self-Attention (pre-norm)
        an = layer.self_attn_layer_norm(fr)
        compare(f"l{layer_idx} attention_norm (an)", f"{pfx}_an.f32", an[0])

        # Manual attention to get intermediate attention output (context before projection)
        attn_out, _ = layer.self_attn(an, attention_mask=extended_attention_mask)
        attn_out_drop = layer.self_attn_dropout(attn_out)

        ar = fr + attn_out_drop  # attention_residual
        compare(f"l{layer_idx} attention_residual (ar)", f"{pfx}_ar.f32", ar[0])

        # Convolution module
        # conv_module forward:
        conv_ln = layer.conv_module.layer_norm(ar)
        compare(f"l{layer_idx} conv_norm (cn)", f"{pfx}_cn.f32", conv_ln[0])

        # pointwise_conv1 → GLU
        conv_t = conv_ln.transpose(1, 2)  # [1, 1024, T]
        pw1 = layer.conv_module.pointwise_conv1(conv_t)  # [1, 2048, T]
        glu_out = layer.conv_module.glu(pw1)  # [1, 1024, T] via GLU(dim=1)
        glu_t_back = glu_out.transpose(1, 2)  # [1, T, 1024]
        compare(f"l{layer_idx} conv_glu (cg)", f"{pfx}_cg.f32", glu_t_back[0])

        # causal depthwise conv (padding kernel-1=30 on left)
        pad_size = layer.conv_module.depthwise_conv.kernel_size[0] - 1
        dw_padded = F.pad(glu_out, (pad_size, 0))  # [1, 1024, T+30]
        dw_out = layer.conv_module.depthwise_conv(dw_padded)  # [1, 1024, T]
        dw_t = dw_out.transpose(1, 2)  # [1, T, 1024]
        compare(f"l{layer_idx} conv_depthwise (cd)", f"{pfx}_cd.f32", dw_t[0])

        # depthwise_layer_norm + activation + pointwise_conv2
        dw_ln = layer.conv_module.depthwise_layer_norm(dw_t)
        dw_act = layer.conv_module.activation(dw_ln)
        pw2 = layer.conv_module.pointwise_conv2(dw_act.transpose(1, 2)).transpose(1, 2)  # [1, T, 1024]

        cr = ar + pw2  # conv_residual
        compare(f"l{layer_idx} conv_residual (cr)", f"{pfx}_cr.f32", cr[0])

        # FFN2: norm is layer.ffn2_layer_norm, not inside layer.ffn2
        fn2 = layer.ffn2_layer_norm(cr)
        fi2 = layer.ffn2.intermediate_dense(fn2)
        fa2 = layer.ffn2.intermediate_act_fn(fi2)
        fo2 = layer.ffn2.output_dense(fa2)
        f2 = cr + 0.5 * fo2  # ffn2_residual
        compare(f"l{layer_idx} ffn2_residual (f2)", f"{pfx}_f2.f32", f2[0])

        # final_layer_norm
        layer_out = layer.final_layer_norm(f2)
        compare(f"l{layer_idx} final_norm", f"layer{layer_idx}.f32", layer_out[0])

        hs = layer_out

    # Final spk_cond_emb comparison
    print(f"\n=== Final spk_cond_emb ===")
    py_ref = torch.load("sample/qin.pt", weights_only=False)
    if isinstance(py_ref, dict) and 'spk_cond_emb' in py_ref:
        py_spk = py_ref['spk_cond_emb'].float().numpy().reshape(-1, 1024)
        native_spk = np.fromfile("/tmp/spk_cond_test.f32", dtype=np.float32).reshape(-1, 1024)
        c = corr(native_spk, py_spk)
        flag = "OK  " if c > 0.99 else ("WARN" if c > 0.9 else "FAIL")
        print(f"  {flag} spk_cond_emb: corr={c:.4f}")
    else:
        print("  No Python reference spk_cond_emb")

print("\nDone.")
