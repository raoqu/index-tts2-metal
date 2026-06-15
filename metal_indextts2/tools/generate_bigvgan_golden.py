from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metal_indextts2.metrics import file_sha256, tensor_summary
from metal_indextts2.tools.common import add_index_tts_repo, write_json


def synthetic_mel(tokens: int) -> np.ndarray:
    values = np.empty((tokens, 80), dtype=np.float32)
    flat = values.reshape(-1)
    for i in range(flat.size):
        flat[i] = np.sin(np.float32(i) * np.float32(0.017)) * np.float32(0.10) + np.cos(
            np.float32(i % 71) * np.float32(0.029)
        ) * np.float32(0.025)
    return values


def write_f32(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(value, dtype=np.float32).tofile(path)


def read_mel(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0 or arr.size % 80 != 0:
        raise ValueError(f"mel file must have shape [tokens,80]: {path}")
    return np.ascontiguousarray(arr.reshape(arr.size // 80, 80))


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = add_index_tts_repo(args.index_tts_repo)
    from indextts.s2mel.modules.bigvgan import bigvgan

    config_path = Path(args.bigvgan_config).expanduser().resolve()
    checkpoint_path = Path(args.bigvgan_checkpoint).expanduser().resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    h = bigvgan.load_hparams_from_json(config_path)
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["generator"])
    model.remove_weight_norm()
    model.eval()

    if args.mel_file:
        mel = read_mel(Path(args.mel_file))
        mel_source = str(Path(args.mel_file).resolve())
    else:
        mel = synthetic_mel(args.tokens)
        mel_source = "synthetic"
    tokens = int(mel.shape[0])
    with torch.no_grad():
        x = torch.from_numpy(mel.T).unsqueeze(0)
        wav = model(x).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    mel_path = out_dir / "mel.f32"
    wav_path = out_dir / "waveform.f32"
    write_f32(mel_path, mel)
    write_f32(wav_path, wav)
    manifest: dict[str, Any] = {
        "format": "mit2-bigvgan-golden",
        "version": 1,
        "source_repo": str(repo),
        "bigvgan_config": str(config_path),
        "bigvgan_checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "tokens": tokens,
        "samples": int(wav.shape[0]),
        "mel_source": mel_source,
        "mel": {
            "path": str(mel_path),
            "shape": [tokens, 80],
            "sha256": file_sha256(mel_path),
            "summary": tensor_summary(mel),
        },
        "waveform": {
            "path": str(wav_path),
            "shape": [int(wav.shape[0])],
            "sha256": file_sha256(wav_path),
            "summary": tensor_summary(wav),
        },
        "final_activation": "tanh" if h.get("use_tanh_at_final", True) else "clamp",
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PyTorch BigVGAN waveform golden fixture.")
    parser.add_argument("--index-tts-repo", default="index-tts")
    parser.add_argument(
        "--bigvgan-checkpoint",
        default="index-tts/checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513/bigvgan_generator.pt",
    )
    parser.add_argument(
        "--bigvgan-config",
        default="index-tts/checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513/config.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--mel-file", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
