from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from metal_indextts2.metrics import file_sha256
from metal_indextts2.tools.common import add_reference_args, captured_stdout, load_indextts2, write_json

TIME_PATTERNS = {
    "gpt_gen_time": r">> gpt_gen_time:\s*([0-9.]+)",
    "gpt_forward_time": r">> gpt_forward_time:\s*([0-9.]+)",
    "s2mel_time": r">> s2mel_time:\s*([0-9.]+)",
    "bigvgan_time": r">> bigvgan_time:\s*([0-9.]+)",
    "total": r">> Total inference time:\s*([0-9.]+)",
    "audio_len": r">> Generated audio length:\s*([0-9.]+)",
    "rtf": r">> RTF:\s*([0-9.]+)",
}


def parse_timings(log: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pattern in TIME_PATTERNS.items():
        m = re.search(pattern, log)
        if m:
            out[key] = float(m.group(1))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_wav = out_dir / "baseline.wav"

    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    if args.voice_name:
        prompt = tts.load_voice(args.voice_name)
    elif args.prompt_audio:
        prompt = args.prompt_audio
    else:
        raise ValueError("provide --voice-name for cached voice TTS or --prompt-audio for clone-from-audio baseline")

    generation_kwargs = {
        "do_sample": not args.greedy,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "max_mel_tokens": args.max_mel_tokens,
    }
    started = time.perf_counter()
    with captured_stdout() as buf:
        result = tts.infer(
            prompt,
            args.text,
            str(output_wav),
            verbose=args.verbose,
            max_text_tokens_per_segment=args.max_text_tokens_per_segment,
            **generation_kwargs,
        )
    elapsed = time.perf_counter() - started
    log = buf.getvalue()
    timings = parse_timings(log)
    timings.setdefault("total_wall", elapsed)

    report = {
        "mode": "cached_voice" if args.voice_name else "clone_from_audio",
        "text": args.text,
        "output_wav": str(output_wav),
        "output_sha256": file_sha256(output_wav) if output_wav.exists() else None,
        "infer_result": result,
        "timings": timings,
        "generation": generation_kwargs,
        "raw_log": log,
    }
    write_json(out_dir / "baseline.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyTorch IndexTTS2 baseline and emit timing/checksum JSON.")
    add_reference_args(parser)
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling for deterministic golden baselines.")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--max-mel-tokens", type=int, default=1500)
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
