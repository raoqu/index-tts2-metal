#!/usr/bin/env python3
"""Native tokenizer parity gate.

Verifies that the native runtime's full text frontend (TextNormalizer +
SentencePiece, replacing the narrow hand-rolled CJK tokenizer) produces token
ids byte-identical to the reference IndexTTS2 Python frontend.

The expected ids in tests/tok_corpus.expected.txt were generated from the
reference (indextts.utils.front.TextTokenizer) so this test needs no Python
deps / venv -- only the built `mtts` binary. Regenerate the golden with:

    .venv/bin/python ref_runner.py tests/tok_corpus.txt > tests/tok_corpus.expected.txt

Usage: python3 tests/test_tokenizer_parity.py [MTTS_BIN] [MODEL_BUNDLE_DIR]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MTTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "mtts"
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "bin")
CORPUS = ROOT / "tests" / "tok_corpus.txt"
EXPECTED = ROOT / "tests" / "tok_corpus.expected.txt"


def native_ids(text: str) -> list[int]:
    out = subprocess.run(
        [str(MTTS), "--tts-cjk-text-readiness", BUNDLE, text],
        capture_output=True, text=True, check=True,
    ).stdout
    return list(json.loads(out)["token_ids"])


def main() -> int:
    if not MTTS.exists():
        print(f"FAIL: mtts binary not found at {MTTS} (build it first)")
        return 2
    texts = [l.rstrip("\n") for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    expected = [l.strip() for l in EXPECTED.read_text().splitlines() if l.strip()]
    if len(texts) != len(expected):
        print(f"FAIL: corpus ({len(texts)}) / expected ({len(expected)}) length mismatch")
        return 2

    failures = 0
    for i, (text, exp_line) in enumerate(zip(texts, expected)):
        exp = [int(x) for x in exp_line.split()]
        got = native_ids(text)
        if got != exp:
            failures += 1
            print(f"FAIL [{i}] {text!r}\n  expected: {exp}\n  got:      {got}")
    total = len(texts)
    if failures:
        print(f"\n{failures}/{total} cases differ from the reference tokenizer.")
        return 1
    print(f"PASS: all {total} cases byte-identical to reference tokenizer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
