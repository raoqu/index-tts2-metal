from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import tensor_summary
from metal_indextts2.tools.common import add_reference_args, load_indextts2, write_json


def _write_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = path.name
    return summary


def _series(size: int, sin_scale: float, cos_scale: float, sin_step: float, cos_step: float, modulo: int) -> np.ndarray:
    idx = np.arange(size, dtype=np.float32)
    return np.sin(idx * sin_step) * sin_scale + np.cos((idx % modulo) * cos_step) * cos_scale


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    tokens = args.tokens
    if args.prompt_tokens > tokens:
        raise ValueError("--prompt-tokens must be <= --tokens")
    x_np = _series(tokens * 80, 0.11, 0.07, 0.019, 0.013, 97).reshape(tokens, 80).astype(np.float32)
    prompt_np = _series(tokens * 80, 0.09, 0.05, 0.023, 0.017, 89).reshape(tokens, 80).astype(np.float32)
    prompt_short_np = np.ascontiguousarray(prompt_np[: args.prompt_tokens])
    cond_np = _series(tokens * 512, 0.08, 0.04, 0.011, 0.007, 127).reshape(tokens, 512).astype(np.float32)
    style_np = _series(192, 0.12, 0.06, 0.016, 0.021, 53).astype(np.float32)

    device = torch.device(tts.device)
    x = torch.from_numpy(x_np.T[None]).to(device)
    prompt = torch.from_numpy(prompt_np.T[None]).to(device)
    prompt_short = torch.from_numpy(prompt_short_np.T[None]).to(device)
    cond = torch.from_numpy(cond_np[None]).to(device)
    style = torch.from_numpy(style_np[None]).to(device)
    x_lens = torch.tensor([tokens], dtype=torch.long, device=device)
    timestep = torch.tensor([args.timestep], dtype=torch.float32, device=device)
    t_span = torch.linspace(0, 1, args.steps + 1, device=device)
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=False):
            output = tts.s2mel.models["cfm"].estimator(x.float(), prompt.float(), x_lens, timestep, style.float(), cond.float())
            cfm = tts.s2mel.models["cfm"].solve_euler(
                x.float().clone(),
                x_lens,
                prompt_short.float(),
                cond.float().clone(),
                style.float(),
                None,
                t_span,
                inference_cfg_rate=0.0,
            )
            cfm_cfg = tts.s2mel.models["cfm"].solve_euler(
                x.float().clone(),
                x_lens,
                prompt_short.float(),
                cond.float().clone(),
                style.float(),
                None,
                t_span,
                inference_cfg_rate=args.cfg_rate,
            )
    output_np = np.ascontiguousarray(output.detach().cpu().numpy()[0].T.astype(np.float32))
    cfm_np = np.ascontiguousarray(cfm.detach().cpu().numpy()[0].T.astype(np.float32))
    cfm_cfg_np = np.ascontiguousarray(cfm_cfg.detach().cpu().numpy()[0].T.astype(np.float32))
    manifest = {
        "format": "mit2-dit-golden",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "tokens": tokens,
        "timestep": args.timestep,
        "prompt_tokens": args.prompt_tokens,
        "steps": args.steps,
        "cfg_rate": args.cfg_rate,
        "tensors": {
            "x": _write_raw(out_dir / "x.f32", x_np),
            "prompt_x": _write_raw(out_dir / "prompt_x.f32", prompt_np),
            "prompt": _write_raw(out_dir / "prompt.f32", prompt_short_np),
            "cond": _write_raw(out_dir / "cond.f32", cond_np),
            "style": _write_raw(out_dir / "style.f32", style_np),
            "estimator": _write_raw(out_dir / "estimator.f32", output_np),
            "cfm_euler": _write_raw(out_dir / "cfm_euler.f32", cfm_np),
            "cfm_euler_cfg": _write_raw(out_dir / "cfm_euler_cfg.f32", cfm_cfg_np),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PyTorch DiT estimator-step golden fixture.")
    add_reference_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--prompt-tokens", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cfg-rate", type=float, default=0.5)
    parser.add_argument("--timestep", type=float, default=0.375)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
