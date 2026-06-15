from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from metal_indextts2.metrics import file_sha256
from metal_indextts2.tools.generate_w2v_bert_golden import (
    _read_feature_manifest,
    _read_preprocessed_audio_from_manifest,
    run,
)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def to(self, device):
        return self


def _write_feature_manifest(tmp_path: Path, *, sha: str | None = None) -> Path:
    audio = np.linspace(-0.1, 0.1, 8, dtype=np.float32)
    audio_path = tmp_path / "clone_ref.16k.f32"
    audio.tofile(audio_path)
    manifest_path = tmp_path / "clone_features.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "mit2-clone-feature-prep",
                "ready_native_clone_audio_preprocess": True,
                "preprocessed_output_f32": str(audio_path),
                "preprocessed_output_sha256": sha if sha is not None else file_sha256(audio_path),
                "preprocessed_sample_rate": 16000,
                "preprocessed_samples": int(audio.size),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_generate_w2v_bert_golden_reads_preprocessed_audio(tmp_path: Path):
    manifest_path = _write_feature_manifest(tmp_path)

    manifest = _read_feature_manifest(manifest_path)
    audio, audio_path, samples = _read_preprocessed_audio_from_manifest(manifest)

    assert audio_path == tmp_path / "clone_ref.16k.f32"
    assert samples == 8
    np.testing.assert_allclose(audio, np.linspace(-0.1, 0.1, 8, dtype=np.float32))


def test_generate_w2v_bert_golden_rejects_audio_sha_mismatch(tmp_path: Path):
    manifest_path = _write_feature_manifest(tmp_path, sha="bad")
    manifest = _read_feature_manifest(manifest_path)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _read_preprocessed_audio_from_manifest(manifest)


def test_generate_w2v_bert_golden_run_writes_normalized_spk_cond(tmp_path: Path, monkeypatch):
    manifest_path = _write_feature_manifest(tmp_path)
    w2v_dir = tmp_path / "w2v_bert"
    w2v_dir.mkdir()
    stats_path = tmp_path / "wav2vec2bert_stats.pt"
    stats_path.write_bytes(b"fake")
    out_dir = tmp_path / "golden"

    class FakeTorch:
        @staticmethod
        def load(path, map_location=None):
            assert str(path) == str(stats_path.resolve())
            assert map_location == "cpu"
            return {
                "mean": FakeTensor(np.ones((1024,), dtype=np.float32)),
                "var": FakeTensor(np.full((1024,), 4.0, dtype=np.float32)),
            }

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def no_grad():
            class Guard:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Guard()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(path):
            assert path == str(w2v_dir.resolve())
            return FakeProcessor()

        def __call__(self, audio, sampling_rate, return_tensors):
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            assert audio.shape == (8,)
            return {
                "input_features": FakeTensor(np.ones((1, 2, 160), dtype=np.float32)),
                "attention_mask": FakeTensor(np.array([[1, 1]], dtype=np.int64)),
            }

    class FakeOutput:
        def __init__(self):
            hidden = [FakeTensor(np.zeros((1, 2, 1024), dtype=np.float32)) for _ in range(18)]
            hidden[17] = FakeTensor(np.full((1, 2, 1024), 5.0, dtype=np.float32))
            self.hidden_states = hidden

    class FakeW2VModel:
        @staticmethod
        def from_pretrained(path):
            assert path == str(w2v_dir.resolve())
            return FakeW2VModel()

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, *, input_features, attention_mask, output_hidden_states):
            assert output_hidden_states is True
            np.testing.assert_array_equal(input_features.numpy(), np.ones((1, 2, 160), dtype=np.float32))
            np.testing.assert_array_equal(attention_mask.numpy(), np.array([[1, 1]], dtype=np.int64))
            return FakeOutput()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(SeamlessM4TFeatureExtractor=FakeProcessor, Wav2Vec2BertModel=FakeW2VModel),
    )

    result = run(
        types.SimpleNamespace(
            feature_manifest=str(manifest_path),
            w2v_bert_dir=str(w2v_dir),
            w2v_stats=str(stats_path),
            output_dir=str(out_dir),
            device="cpu",
        )
    )

    assert result["format"] == "mit2-w2v-bert-semantic-golden"
    assert result["spk_cond_tokens"] == 2
    assert result["ready_reference_w2v_bert_semantic_features"] is True
    assert result["ready_native_w2v_bert_semantic_features"] is False
    spk_cond = np.fromfile(out_dir / "spk_cond_emb.f32", dtype=np.float32).reshape(1, 2, 1024)
    np.testing.assert_array_equal(spk_cond, np.full((1, 2, 1024), 2.0, dtype=np.float32))
    mask = np.fromfile(out_dir / "w2v_attention_mask.u32", dtype=np.uint32).reshape(1, 2)
    np.testing.assert_array_equal(mask, np.array([[1, 1]], dtype=np.uint32))
    written_manifest = json.loads((out_dir / "w2v_bert_semantic_golden.manifest.json").read_text(encoding="utf-8"))
    assert written_manifest["artifacts"]["spk_cond_emb"]["sha256"] == file_sha256(out_dir / "spk_cond_emb.f32")
