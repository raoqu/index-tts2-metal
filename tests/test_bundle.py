from __future__ import annotations

import numpy as np

from metal_indextts2.bundle import read_tensor, write_bundle
from metal_indextts2.tools import convert_model
from metal_indextts2.tools.convert_model import _iter_tensors


def test_bundle_round_trip(tmp_path):
    manifest = write_bundle(
        tmp_path,
        [
            ("gpt.weight", np.arange(12, dtype=np.float32).reshape(3, 4), "gpt"),
            ("voice.codes", np.arange(5, dtype=np.int32), "voice"),
        ],
        metadata={"unit": True},
    )

    assert manifest["format"] == "MIT2"
    assert len(manifest["tensors"]) == 2
    assert manifest["tensors"][0]["offset"] % manifest["alignment"] == 0

    arr = read_tensor(tmp_path, "gpt.weight")
    np.testing.assert_array_equal(arr, np.arange(12, dtype=np.float32).reshape(3, 4))


def test_convert_model_recurses_nested_tensors():
    nested = {
        "net": {
            "gpt_layer": {
                "0": {"weight": np.ones((2, 3), dtype=np.float32)},
            }
        },
        "epoch": 1,
    }
    names = [name for name, _, _ in _iter_tensors(nested, "s2mel", "s2mel")]
    assert names == ["s2mel.net.gpt_layer.0.weight"]


def test_convert_model_includes_campplus_checkpoint(tmp_path, monkeypatch):
    class FakeTorch:
        @staticmethod
        def load(path, map_location=None):
            assert map_location == "cpu"
            assert str(path).endswith("campplus_cn_common.bin")
            return {
                "head": {
                    "conv1": {
                        "weight": np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3),
                    }
                },
                "xvector": {
                    "dense": {
                        "linear": {
                            "weight": np.arange(6, dtype=np.float32).reshape(2, 3, 1),
                        }
                    }
                },
            }

    monkeypatch.setattr(convert_model, "_load_torch", lambda: FakeTorch)
    manifest = convert_model.convert(
        tmp_path / "checkpoint",
        tmp_path / "bundle",
        campplus_checkpoint=tmp_path / "campplus_cn_common.bin",
    )

    by_name = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    assert by_name["campplus.head.conv1.weight"]["component"] == "campplus"
    assert by_name["campplus.xvector.dense.linear.weight"]["component"] == "campplus"
    np.testing.assert_array_equal(
        read_tensor(tmp_path / "bundle", "campplus.xvector.dense.linear.weight"),
        np.arange(6, dtype=np.float32).reshape(2, 3, 1),
    )
    assert str(tmp_path / "campplus_cn_common.bin") in manifest["metadata"]["source_files"]


def test_convert_model_includes_w2v_bert_and_stats(tmp_path, monkeypatch):
    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self.array

    class FakeW2VModel:
        def state_dict(self):
            return {
                "feature_projection.layer_norm.weight": FakeTensor(np.ones((160,), dtype=np.float32)),
                "feature_projection.projection.weight": FakeTensor(np.ones((1024, 160), dtype=np.float32)),
                "encoder.layers.17.final_layer_norm.weight": FakeTensor(np.ones((1024,), dtype=np.float32)),
            }

    class FakeWav2Vec2BertModel:
        @staticmethod
        def from_pretrained(path):
            assert str(path).endswith("w2v_bert")
            return FakeW2VModel()

    class FakeTorch:
        @staticmethod
        def load(path, map_location=None):
            assert map_location == "cpu"
            assert str(path).endswith("wav2vec2bert_stats.pt")
            return {
                "mean": FakeTensor(np.arange(1024, dtype=np.float32)),
                "var": FakeTensor(np.full((1024,), 4.0, dtype=np.float32)),
            }

        @staticmethod
        def sqrt(tensor):
            return FakeTensor(np.sqrt(tensor.array))

    monkeypatch.setattr(convert_model, "_load_torch", lambda: FakeTorch)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type("FakeTransformers", (), {"Wav2Vec2BertModel": FakeWav2Vec2BertModel}),
    )

    manifest = convert_model.convert(
        tmp_path / "checkpoint",
        tmp_path / "bundle",
        w2v_bert_dir=tmp_path / "w2v_bert",
        w2v_stats=tmp_path / "wav2vec2bert_stats.pt",
    )

    by_name = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    assert by_name["w2v_bert.feature_projection.layer_norm.weight"]["component"] == "w2v_bert"
    assert by_name["w2v_bert.encoder.layers.17.final_layer_norm.weight"]["component"] == "w2v_bert"
    assert by_name["w2v_bert.stats.mean"]["component"] == "w2v_bert"
    assert by_name["w2v_bert.stats.std"]["component"] == "w2v_bert"
    np.testing.assert_array_equal(
        read_tensor(tmp_path / "bundle", "w2v_bert.stats.std"),
        np.full((1024,), 2.0, dtype=np.float32),
    )
    assert str(tmp_path / "w2v_bert") in manifest["metadata"]["source_files"]
    assert str(tmp_path / "wav2vec2bert_stats.pt") in manifest["metadata"]["source_files"]
