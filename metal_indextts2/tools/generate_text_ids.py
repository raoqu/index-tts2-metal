from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import tensor_summary
from metal_indextts2.tools.common import write_json


def _add_index_tts_repo(index_tts_repo: str | os.PathLike[str]) -> Path:
    repo = Path(index_tts_repo).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"IndexTTS repository not found: {repo}")
    sys.path.insert(0, str(repo))
    return repo


def _write_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = path.name
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    _add_index_tts_repo(args.index_tts_repo)
    from indextts.utils.front import TextNormalizer, TextTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "text_ids.u32"
    bpe_model = Path(args.bpe_model) if args.bpe_model else Path(args.checkpoint_dir) / "bpe.model"

    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(bpe_model), normalizer)
    tokens = tokenizer.tokenize(args.text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if tokenizer.unk_token_id in token_ids and not args.allow_unknown:
        unknown = [token for token, token_id in zip(tokens, token_ids) if token_id == tokenizer.unk_token_id]
        raise ValueError(f"text contains unknown tokenizer pieces: {unknown}")
    segments = tokenizer.split_segments(tokens, args.max_text_tokens_per_segment)
    if not segments:
        raise ValueError("text produced no tokenizer segments")
    if args.segment_index >= len(segments):
        raise ValueError(f"--segment-index {args.segment_index} out of range for {len(segments)} segments")
    segment = segments[args.segment_index]
    text_ids = tokenizer.convert_tokens_to_ids(segment)
    text_np = np.ascontiguousarray(np.asarray(text_ids, dtype=np.uint32))

    manifest = {
        "format": "mit2-text-ids",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).expanduser().resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "bpe_model": str(bpe_model.expanduser().resolve()),
        "text": args.text,
        "tokens": tokens,
        "token_ids": token_ids,
        "segment_index": args.segment_index,
        "segment_tokens": segment,
        "text_ids": text_ids,
        "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
        "tensors": {
            "text_ids": _write_raw(output_path, text_np),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IndexTTS2 text token ids without loading the TTS models.")
    parser.add_argument("--index-tts-repo", default="index-tts")
    parser.add_argument("--checkpoint-dir", default="index-tts/checkpoints")
    parser.add_argument("--bpe-model", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--allow-unknown", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
