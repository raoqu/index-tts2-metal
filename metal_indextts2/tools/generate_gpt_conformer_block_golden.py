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
            subsampled, pos_emb, out_mask = tts.gpt.conditioning_encoder.embed(x, mask)
            block_out, block_mask, _att_cache, _cnn_cache = layer(subsampled, out_mask, pos_emb, out_mask)

    input_np = _as_numpy(subsampled.squeeze(0), np.float32)
    pos_np = _as_numpy(pos_emb.squeeze(0), np.float32)
    mask_np = _as_numpy(out_mask.squeeze(0).squeeze(0), np.uint32)
    out_np = _as_numpy(block_out.squeeze(0), np.float32)
    out_mask_np = _as_numpy(block_mask.squeeze(0).squeeze(0), np.uint32)

    manifest = {
        "format": "mit2-gpt-conformer-block-golden",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).expanduser().resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "voice_name": profile.get("voice_name") or args.voice_name,
        "condition_type": tts.gpt.condition_type,
        "layer_index": int(args.layer_index),
        "input_tokens": int(input_np.shape[0]),
        "input_dim": int(input_np.shape[1]),
        "tensors": {
            "block_input": _write_raw(out_dir / "block_input.f32", input_np),
            "pos_emb": _write_raw(out_dir / "pos_emb.f32", pos_np),
            "block_mask": _write_raw(out_dir / "block_mask.u32", mask_np),
            "block_output": _write_raw(out_dir / "block_output.f32", out_np),
            "block_output_mask": _write_raw(out_dir / "block_output_mask.u32", out_mask_np),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GPT conditioning Conformer layer golden fixture.")
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
