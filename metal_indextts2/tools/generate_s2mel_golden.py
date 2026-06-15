from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.bundle import read_tensor
from metal_indextts2.metrics import tensor_summary
from metal_indextts2.tools.common import add_reference_args, load_indextts2, write_json


def _write_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = path.name
    return summary


def _read_raw_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(shape))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    device = torch.device(tts.device)

    gpt_dir = Path(args.gpt_golden_dir)
    lr = np.fromfile(gpt_dir / "length_regulator.f32", dtype=np.float32)
    if lr.size == 0 or lr.size % 512 != 0:
        raise ValueError("length_regulator.f32 must have shape [tokens,512]")
    gen_tokens = lr.size // 512
    gen_condition_np = np.ascontiguousarray(lr.reshape(gen_tokens, 512))

    prompt_condition_np = np.asarray(read_tensor(args.voice_bundle, "s2mel_prompt"), dtype=np.float32)
    ref_mel_np = np.asarray(read_tensor(args.voice_bundle, "mel"), dtype=np.float32)
    style_np = np.asarray(read_tensor(args.voice_bundle, "s2mel_style"), dtype=np.float32)
    if prompt_condition_np.ndim != 3 or prompt_condition_np.shape[0] != 1 or prompt_condition_np.shape[2] != 512:
        raise ValueError("voice s2mel_prompt must have shape [1,prompt_tokens,512]")
    if ref_mel_np.ndim != 3 or ref_mel_np.shape[0] != 1 or ref_mel_np.shape[1] != 80:
        raise ValueError("voice mel must have shape [1,80,prompt_tokens]")
    if style_np.size != 192:
        raise ValueError("voice s2mel_style must contain 192 floats")

    source_prompt_tokens = int(ref_mel_np.shape[2])
    prompt_tokens = source_prompt_tokens
    if args.prompt_tokens is not None:
        if args.prompt_tokens <= 0 or args.prompt_tokens > source_prompt_tokens:
            raise ValueError(f"--prompt-tokens must be in [1,{source_prompt_tokens}]")
        prompt_tokens = args.prompt_tokens
        prompt_condition_np = np.ascontiguousarray(prompt_condition_np[:, :prompt_tokens, :])
        ref_mel_np = np.ascontiguousarray(ref_mel_np[:, :, :prompt_tokens])
    if int(prompt_condition_np.shape[1]) != prompt_tokens:
        raise ValueError("voice s2mel_prompt and mel prompt lengths differ")

    prompt_condition = torch.from_numpy(np.ascontiguousarray(prompt_condition_np)).to(device)
    gen_condition = torch.from_numpy(gen_condition_np[None]).to(device)
    cat_condition = torch.cat([prompt_condition.float(), gen_condition.float()], dim=1)
    ref_mel = torch.from_numpy(np.ascontiguousarray(ref_mel_np)).to(device).float()
    style = torch.from_numpy(np.ascontiguousarray(style_np.reshape(1, 192))).to(device).float()
    x_lens = torch.tensor([cat_condition.size(1)], dtype=torch.long, device=device)
    t_span = torch.linspace(0, 1, args.steps + 1, device=device)

    torch.manual_seed(args.seed)
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=False):
            noise = torch.randn([1, 80, cat_condition.size(1)], device=device) * args.temperature
            full = tts.s2mel.models["cfm"].solve_euler(
                noise.clone(),
                x_lens,
                ref_mel,
                cat_condition.clone(),
                style,
                None,
                t_span,
                inference_cfg_rate=args.cfg_rate,
            )
    generated = full[:, :, prompt_tokens:]

    noise_np = np.ascontiguousarray(noise.detach().cpu().numpy()[0].T.astype(np.float32))
    prompt_mel_np = np.ascontiguousarray(ref_mel.detach().cpu().numpy()[0].T.astype(np.float32))
    cat_condition_out_np = np.ascontiguousarray(cat_condition.detach().cpu().numpy()[0].astype(np.float32))
    style_out_np = np.ascontiguousarray(style.detach().cpu().numpy()[0].astype(np.float32))
    full_np = np.ascontiguousarray(full.detach().cpu().numpy()[0].T.astype(np.float32))
    generated_np = np.ascontiguousarray(generated.detach().cpu().numpy()[0].T.astype(np.float32))

    manifest = {
        "format": "mit2-s2mel-golden",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "voice_bundle": str(Path(args.voice_bundle).resolve()),
        "gpt_golden_dir": str(gpt_dir.resolve()),
        "prompt_tokens": prompt_tokens,
        "source_prompt_tokens": source_prompt_tokens,
        "generated_tokens": gen_tokens,
        "total_tokens": int(cat_condition.size(1)),
        "steps": args.steps,
        "cfg_rate": args.cfg_rate,
        "temperature": args.temperature,
        "seed": args.seed,
        "tensors": {
            "noise": _write_raw(out_dir / "noise.f32", noise_np),
            "prompt_mel": _write_raw(out_dir / "prompt_mel.f32", prompt_mel_np),
            "condition": _write_raw(out_dir / "condition.f32", cat_condition_out_np),
            "style": _write_raw(out_dir / "style.f32", style_out_np),
            "s2mel_full": _write_raw(out_dir / "s2mel_full.f32", full_np),
            "s2mel_generated": _write_raw(out_dir / "s2mel_generated.f32", generated_np),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PyTorch S2Mel full-mel golden fixture from GPT and voice bundles.")
    add_reference_args(parser)
    parser.add_argument("--gpt-golden-dir", required=True)
    parser.add_argument("--voice-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20240605)
    parser.add_argument("--prompt-tokens", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
