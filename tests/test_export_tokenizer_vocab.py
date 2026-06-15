from __future__ import annotations

import argparse
import sys
import types

from metal_indextts2.tools import export_tokenizer_vocab


def test_export_tokenizer_vocab_writes_sentencepiece_scores(tmp_path, monkeypatch):
    class FakeProcessor:
        def Load(self, path: str) -> None:
            self.path = path

        def GetPieceSize(self) -> int:
            return 2

        def IdToPiece(self, token_id: int) -> str:
            return ["A", "BC"][token_id]

        def GetScore(self, token_id: int) -> float:
            return [-1.25, -2.5][token_id]

        def unk_id(self) -> int:
            return 0

        def bos_id(self) -> int:
            return -1

        def eos_id(self) -> int:
            return -1

        def pad_id(self) -> int:
            return -1

    fake_spm = types.SimpleNamespace(SentencePieceProcessor=FakeProcessor)
    monkeypatch.setitem(sys.modules, "sentencepiece", fake_spm)
    bpe_model = tmp_path / "bpe.model"
    bpe_model.write_bytes(b"fake")
    output = tmp_path / "pieces.tsv"

    summary = export_tokenizer_vocab.run(
        argparse.Namespace(
            bpe_model=str(bpe_model),
            output=str(output),
            summary_json=None,
        )
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "0\tA\t-1.25",
        "1\tBC\t-2.5",
    ]
    assert summary["piece_count"] == 2
