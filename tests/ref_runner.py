#!/usr/bin/env python3
# Reference tokenizer runner: emits one line of space-separated token ids per
# input line, using the IndexTTS2 Python frontend (indextts.utils.front).
# Used to (re)generate tests/tok_corpus.expected.txt for the native parity gate.
# Run from the reference repo with its venv, e.g.:
#   cd ~/research/index-tts && .venv/bin/python <this> tok_corpus.txt > expected.txt

import sys
sys.path.insert(0,".")
from indextts.utils.front import TextNormalizer, TextTokenizer
norm=TextNormalizer(); norm.load()
tok=TextTokenizer("checkpoints/bpe.model", norm)
for line in open(sys.argv[1], encoding="utf-8"):
    line=line.rstrip("\n")
    if not line: continue
    ids=tok.convert_tokens_to_ids(tok.tokenize(line))
    print(" ".join(map(str,ids)))
