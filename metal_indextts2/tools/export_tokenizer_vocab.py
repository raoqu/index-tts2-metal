from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metal_indextts2.tools.common import write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sentencepiece as spm

    bpe_model = Path(args.bpe_model).expanduser().resolve()
    if not bpe_model.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {bpe_model}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    processor = spm.SentencePieceProcessor()
    processor.Load(str(bpe_model))

    with output.open("w", encoding="utf-8", newline="\n") as fp:
        for token_id in range(processor.GetPieceSize()):
            fp.write(f"{token_id}\t{processor.IdToPiece(token_id)}\t{processor.GetScore(token_id):.9g}\n")

    summary = {
        "format": "mit2-tokenizer-pieces",
        "version": 1,
        "bpe_model": str(bpe_model),
        "output": str(output),
        "piece_count": processor.GetPieceSize(),
        "unk_id": processor.unk_id(),
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
        "pad_id": processor.pad_id(),
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IndexTTS2 SentencePiece vocabulary for lightweight native tokenizer smoke tests.")
    parser.add_argument("--bpe-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
