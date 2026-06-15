from __future__ import annotations

from pathlib import Path

import numpy as np

from metal_indextts2.bundle import write_bundle
from metal_indextts2.tools import convert_voice


def _voice_profile() -> dict[str, object]:
    return {
        "voice_name": "unit_voice",
        "ref_audio_path": "refs/unit.wav",
        "created_at": "2026-06-05T00:00:00Z",
        "spk_cond_emb": np.arange(12, dtype=np.float32).reshape(3, 4),
        "s2mel_style": np.arange(4, dtype=np.float32),
        "s2mel_prompt": np.arange(8, dtype=np.float32).reshape(2, 4),
        "mel": np.arange(15, dtype=np.float32).reshape(3, 5),
    }


def test_convert_voice_validates_source_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(convert_voice, "load_voice", lambda path: _voice_profile())
    voice = Path("voice.pt")
    output = tmp_path / "voice_bundle"

    manifest = convert_voice.convert_voice(voice, output)
    validation = convert_voice.validate_voice_bundle(voice, output)

    assert manifest["metadata"]["voice_name"] == "unit_voice"
    assert validation["format"] == "mit2-voice-bundle-validation"
    assert validation["ok"] is True
    assert [item["name"] for item in validation["tensors"]] == list(convert_voice.VOICE_TENSOR_KEYS)
    assert all(item["matches"] for item in validation["tensors"])


def test_convert_voice_validation_respects_force_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(convert_voice, "load_voice", lambda path: _voice_profile())
    voice = Path("voice.pt")
    output = tmp_path / "voice_bundle"

    convert_voice.convert_voice(voice, output, force_dtype="f16")
    validation = convert_voice.validate_voice_bundle(voice, output, force_dtype="f16")

    assert validation["ok"] is True
    assert {item["dtype"] for item in validation["tensors"]} == {"float16"}


def test_no_torch_voice_profile_pt_round_trips_from_native_bundle(tmp_path):
    bundle = tmp_path / "voice_bundle"
    output = tmp_path / "voice.pt"
    profile = _voice_profile()
    write_bundle(
        bundle,
        [(key, profile[key], "voice") for key in convert_voice.VOICE_TENSOR_KEYS],
        metadata={"voice_name": "unit_voice"},
    )

    manifest = convert_voice.write_mit2_voice_profile_pt(
        bundle,
        output,
        voice_name="unit_voice",
        ref_audio_path="refs/unit.wav",
    )
    loaded = convert_voice.load_voice(output)

    assert manifest["format"] == convert_voice.VOICE_PROFILE_PT_FORMAT
    assert output.exists()
    assert loaded["voice_name"] == "unit_voice"
    assert loaded["ref_audio_path"] == "refs/unit.wav"
    for key in convert_voice.VOICE_TENSOR_KEYS:
        np.testing.assert_array_equal(loaded[key], profile[key])


def test_validate_voice_bundle_reports_tensor_mismatch(tmp_path, monkeypatch):
    profile = _voice_profile()
    monkeypatch.setattr(convert_voice, "load_voice", lambda path: profile)
    output = tmp_path / "voice_bundle"
    wrong_profile = dict(profile)
    wrong_profile["mel"] = np.zeros_like(profile["mel"])
    write_bundle(
        output,
        [(key, wrong_profile[key], "voice") for key in convert_voice.VOICE_TENSOR_KEYS],
        metadata={"voice_name": "unit_voice"},
    )

    validation = convert_voice.validate_voice_bundle(Path("voice.pt"), output)
    mel_result = next(item for item in validation["tensors"] if item["name"] == "mel")

    assert validation["ok"] is False
    assert mel_result["matches"] is False
    assert mel_result["converted_sha256"] != mel_result["sha256"]
