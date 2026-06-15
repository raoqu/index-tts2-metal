from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metal_indextts2.metrics import file_sha256
from metal_indextts2.tools.generate_campplus_golden import _read_fbank_from_manifest, _read_feature_manifest


def test_generate_campplus_golden_reads_feature_manifest_fbank(tmp_path: Path):
    fbank = np.arange(160, dtype=np.float32).reshape(2, 80)
    fbank_path = tmp_path / "fbank.f32"
    fbank.tofile(fbank_path)
    manifest_path = tmp_path / "clone_features.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "mit2-clone-feature-prep",
                "ready_native_clone_fbank_extraction": True,
                "output_fbank_f32": str(fbank_path),
                "output_fbank_sha256": file_sha256(fbank_path),
                "fbank_frames": 2,
            }
        ),
        encoding="utf-8",
    )

    manifest = _read_feature_manifest(manifest_path)
    loaded, loaded_path, frames = _read_fbank_from_manifest(manifest)

    assert loaded_path == fbank_path
    assert frames == 2
    np.testing.assert_array_equal(loaded, fbank)


def test_generate_campplus_golden_rejects_fbank_sha_mismatch(tmp_path: Path):
    fbank_path = tmp_path / "fbank.f32"
    np.zeros((2, 80), dtype=np.float32).tofile(fbank_path)
    manifest = {
        "format": "mit2-clone-feature-prep",
        "ready_native_clone_fbank_extraction": True,
        "output_fbank_f32": str(fbank_path),
        "output_fbank_sha256": "bad",
        "fbank_frames": 2,
    }

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _read_fbank_from_manifest(manifest)
