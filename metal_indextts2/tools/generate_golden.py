from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import file_sha256, tensor_summary
from metal_indextts2.tools.common import add_reference_args, captured_stdout, load_indextts2, write_json


def _save_npz(path: Path, tensors: dict[str, Any]) -> dict[str, Any]:
    arrays = {}
    summary = {}
    for name, value in tensors.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.ascontiguousarray(np.asarray(value))
        arrays[name] = arr
        summary[name] = tensor_summary(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)

    profile = tts.load_voice(args.voice_name) if args.voice_name else tts._extract_voice_features(args.prompt_audio)
    profile.setdefault("voice_name", args.voice_name or Path(args.prompt_audio).stem)

    tokens = tts.tokenizer.tokenize(args.text)
    segments = tts.tokenizer.split_segments(tokens, args.max_text_tokens_per_segment)
    token_ids = tts.tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [tts.tokenizer.convert_tokens_to_ids(seg) for seg in segments]

    voice_summary = _save_npz(
        out_dir / "voice_profile.npz",
        {
            "spk_cond_emb": profile["spk_cond_emb"],
            "s2mel_style": profile["s2mel_style"],
            "s2mel_prompt": profile["s2mel_prompt"],
            "mel": profile["mel"],
        },
    )

    output_wav = out_dir / "reference.wav"
    with captured_stdout() as buf:
        tts.infer(
            profile,
            args.text,
            str(output_wav),
            verbose=args.verbose,
            max_text_tokens_per_segment=args.max_text_tokens_per_segment,
            do_sample=not args.greedy,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            max_mel_tokens=args.max_mel_tokens,
        )

    manifest = {
        "format": "mit2-golden",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "voice_name": profile.get("voice_name"),
        "text": args.text,
        "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
        "tokens": tokens,
        "token_ids": token_ids,
        "segments": segments,
        "segment_ids": segment_ids,
        "generation": {
            "greedy": args.greedy,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
            "max_mel_tokens": args.max_mel_tokens,
        },
        "voice_tensors": voice_summary,
        "audio": {
            "path": str(output_wav),
            "sha256": file_sha256(output_wav) if output_wav.exists() else None,
        },
        "raw_log": buf.getvalue(),
    }
    write_json(out_dir / "golden_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PyTorch golden artifacts for IndexTTS2 parity tests.")
    add_reference_args(parser)
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--max-mel-tokens", type=int, default=1500)
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.voice_name and not args.prompt_audio:
        raise SystemExit("provide --voice-name or --prompt-audio")
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
