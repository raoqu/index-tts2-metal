from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import tensor_summary
from metal_indextts2.tools.common import add_reference_args, load_indextts2, write_json


def _as_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _write_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = path.name
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    profile = tts.load_voice(args.voice_name) if args.voice_name else tts._extract_voice_features(args.prompt_audio)
    spk_cond_emb = profile["spk_cond_emb"].to(tts.device)
    input_tokens = min(int(args.input_tokens), int(spk_cond_emb.shape[1]))
    if input_tokens < 3:
        raise ValueError("--input-tokens must leave at least 3 conditioning frames")

    x = spk_cond_emb[:, :input_tokens, :]
    mask = torch.ones((1, 1, input_tokens), dtype=torch.bool, device=tts.device)
    layer = tts.gpt.conditioning_encoder.encoders[int(args.layer_index)]

    with torch.no_grad():
        with torch.amp.autocast(tts.device, enabled=tts.dtype is not None, dtype=tts.dtype):
            subsampled, _pos_emb, _out_mask = tts.gpt.conditioning_encoder.embed(x, mask)
            normed = layer.norm_ff(subsampled)
            ff_raw = layer.feed_forward(normed)
            ff_residual = subsampled + float(layer.ff_scale) * ff_raw
            if layer.conv_module is not None:
                ff_tail = layer.norm_final(ff_residual)
            else:
                ff_tail = ff_residual

    input_np = _as_numpy(subsampled.squeeze(0), np.float32)
    normed_np = _as_numpy(normed.squeeze(0), np.float32)
    raw_np = _as_numpy(ff_raw.squeeze(0), np.float32)
    tail_np = _as_numpy(ff_tail.squeeze(0), np.float32)

    manifest = {
        "format": "mit2-gpt-conformer-ff-golden",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).expanduser().resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "voice_name": profile.get("voice_name") or args.voice_name,
        "condition_type": tts.gpt.condition_type,
        "layer_index": int(args.layer_index),
        "ff_scale": float(layer.ff_scale),
        "input_tokens": int(input_np.shape[0]),
        "input_dim": int(input_np.shape[1]),
        "hidden_units": int(layer.feed_forward.w_1.out_features),
        "tensors": {
            "ff_input": _write_raw(out_dir / "ff_input.f32", input_np),
            "ff_normed": _write_raw(out_dir / "ff_normed.f32", normed_np),
            "ff_raw": _write_raw(out_dir / "ff_raw.f32", raw_np),
            "ff_tail": _write_raw(out_dir / "ff_tail.f32", tail_np),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GPT conditioning Conformer feed-forward golden fixture.")
    add_reference_args(parser)
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-tokens", type=int, default=9)
    parser.add_argument("--layer-index", type=int, default=0)
    args = parser.parse_args()
    if not args.voice_name and not args.prompt_audio:
        raise SystemExit("provide --voice-name or --prompt-audio")
    if args.input_tokens < 3:
        raise SystemExit("--input-tokens must be at least 3")
    if args.layer_index < 0:
        raise SystemExit("--layer-index must be non-negative")
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
