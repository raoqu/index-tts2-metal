from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_indextts2.metrics import file_sha256
from metal_indextts2.tools.common import add_reference_args, captured_stdout, load_indextts2, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference IndexTTS2 clone/TTS entrypoint used until native stages land.")
    add_reference_args(parser)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--voice-name", default=None, help="Use an existing cached voice profile.")
    parser.add_argument("--clone-audio", default=None, help="Create or refresh a voice profile from reference audio.")
    parser.add_argument("--clone-name", default=None)
    parser.add_argument("--overwrite-voice", action="store_true")
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--max-mel-tokens", type=int, default=1500)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.voice_name and not args.clone_audio:
        raise SystemExit("provide --voice-name or --clone-audio")

    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    cloned_voice = None
    if args.clone_audio:
        cloned_voice = tts.clone_voice(
            args.clone_audio,
            voice_name=args.clone_name,
            overwrite=args.overwrite_voice,
            verbose=args.verbose,
        )
        voice_name = cloned_voice
    else:
        voice_name = args.voice_name

    profile = tts.load_voice(voice_name)
    output_wav = Path(args.output_wav)
    with captured_stdout() as buf:
        result = tts.infer(
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

    report = {
        "mode": "clone_and_tts" if args.clone_audio else "cached_voice_tts",
        "voice_name": voice_name,
        "cloned_voice": cloned_voice,
        "output_wav": str(output_wav),
        "output_sha256": file_sha256(output_wav) if output_wav.exists() else None,
        "infer_result": result,
        "raw_log": buf.getvalue(),
    }
    if args.report_json:
        write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
