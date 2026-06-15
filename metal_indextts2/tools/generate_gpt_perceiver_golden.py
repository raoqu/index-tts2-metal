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
    cond_len = torch.tensor([spk_cond_emb.shape[-1]], device=tts.device)

    with torch.no_grad():
        with torch.amp.autocast(tts.device, enabled=tts.dtype is not None, dtype=tts.dtype):
            if tts.gpt.condition_type != "conformer_perceiver":
                raise ValueError(f"expected conformer_perceiver conditioning, got {tts.gpt.condition_type!r}")
            context, mask = tts.gpt.conditioning_encoder(spk_cond_emb, cond_len)
            context_tokens = min(int(args.context_tokens), int(context.shape[1]))
            context = context[:, :context_tokens, :]
            context_mask = mask.squeeze(1)[:, :context_tokens]
            perceiver_mask = torch.cat(
                [
                    torch.ones((context_mask.shape[0], tts.gpt.cond_num), dtype=torch.bool, device=context_mask.device),
                    context_mask.bool(),
                ],
                dim=1,
            )
            perceiver = tts.gpt.perceiver_encoder(context, perceiver_mask)

    context_np = _as_numpy(context.squeeze(0), np.float32)
    mask_np = _as_numpy(perceiver_mask.squeeze(0), np.uint32)
    perceiver_np = _as_numpy(perceiver.squeeze(0), np.float32)

    manifest = {
        "format": "mit2-gpt-perceiver-golden",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).expanduser().resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "voice_name": profile.get("voice_name") or args.voice_name,
        "condition_type": tts.gpt.condition_type,
        "context_tokens": int(context_np.shape[0]),
        "context_dim": int(context_np.shape[1]),
        "latents": int(perceiver_np.shape[0]),
        "latent_dim": int(perceiver_np.shape[1]),
        "tensors": {
            "conditioning_context": _write_raw(out_dir / "conditioning_context.f32", context_np),
            "perceiver_mask": _write_raw(out_dir / "perceiver_mask.u32", mask_np),
            "perceiver": _write_raw(out_dir / "perceiver.f32", perceiver_np),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GPT conditioning PerceiverResampler golden fixture.")
    add_reference_args(parser)
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-tokens", type=int, default=16)
    args = parser.parse_args()
    if not args.voice_name and not args.prompt_audio:
        raise SystemExit("provide --voice-name or --prompt-audio")
    if args.context_tokens <= 0:
        raise SystemExit("--context-tokens must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
