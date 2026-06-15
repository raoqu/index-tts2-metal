from __future__ import annotations

import argparse

import numpy as np
import pytest

from metal_indextts2.tools import generate_gpt_frontend


def test_load_emovec_file_requires_1280_float32_values(tmp_path):
    good = tmp_path / "emovec.f32"
    np.arange(1280, dtype=np.float32).tofile(good)
    loaded = generate_gpt_frontend._load_emovec_file(good)
    assert loaded.shape == (1280,)
    assert loaded.dtype == np.float32

    bad = tmp_path / "bad.f32"
    np.arange(1279, dtype=np.float32).tofile(bad)
    with pytest.raises(ValueError, match="1280"):
        generate_gpt_frontend._load_emovec_file(bad)


def test_validate_emovec_file_rejects_missing_or_wrong_size(tmp_path):
    missing = tmp_path / "missing.f32"
    with pytest.raises(ValueError, match="does not exist"):
        generate_gpt_frontend._validate_emovec_file_arg(missing)

    bad = tmp_path / "bad.f32"
    np.arange(1279, dtype=np.float32).tofile(bad)
    with pytest.raises(ValueError, match="exactly 1280"):
        generate_gpt_frontend._validate_emovec_file_arg(bad)


def test_load_emovec_mix_preserves_weighted_tensor_math(tmp_path):
    first = tmp_path / "first.f32"
    second = tmp_path / "second.f32"
    a = np.arange(1280, dtype=np.float32)
    b = np.full(1280, 3.0, dtype=np.float32)
    a.tofile(first)
    b.tofile(second)

    mixed, components = generate_gpt_frontend._load_emovec_mix(
        [f"{first}=0.25", f"{second}=2.0"]
    )

    np.testing.assert_allclose(mixed, a * np.float32(0.25) + b * np.float32(2.0))
    assert components == [
        {"path": str(first.resolve()), "weight": 0.25},
        {"path": str(second.resolve()), "weight": 2.0},
    ]


def test_emovec_mix_requires_finite_weight(tmp_path):
    vec = tmp_path / "emovec.f32"
    np.zeros(1280, dtype=np.float32).tofile(vec)
    with pytest.raises(ValueError, match="finite"):
        generate_gpt_frontend._load_emovec_mix([f"{vec}=nan"])


def test_validate_emotion_args_checks_emovec_file_before_model_work(tmp_path):
    bad = tmp_path / "bad.f32"
    np.arange(1279, dtype=np.float32).tofile(bad)
    args = argparse.Namespace(
        emovec_file=str(bad),
        emovec_mix=None,
        native_emovec=False,
        emotion_audio=None,
        emotion_voice_name=None,
        emotion_alpha=1.0,
        emotion_vector=None,
        emotion_vector_random=False,
        emotion_vector_random_seed=None,
    )
    with pytest.raises(ValueError, match="--emovec-file.*exactly 1280"):
        generate_gpt_frontend._validate_emotion_args(args)


def test_validate_emotion_args_checks_emovec_mix_components(tmp_path):
    good = tmp_path / "good.f32"
    np.zeros(1280, dtype=np.float32).tofile(good)
    bad = tmp_path / "bad.f32"
    np.zeros(1279, dtype=np.float32).tofile(bad)

    args = argparse.Namespace(
        emovec_file=None,
        emovec_mix=[f"{good}=0.25", f"{bad}=0.75"],
        native_emovec=False,
        emotion_audio=None,
        emotion_voice_name=None,
        emotion_alpha=1.0,
        emotion_vector=None,
        emotion_vector_random=False,
        emotion_vector_random_seed=None,
    )
    with pytest.raises(ValueError, match="--emovec-mix.*exactly 1280"):
        generate_gpt_frontend._validate_emotion_args(args)

    args.emovec_mix = [f"{good}=oops"]
    with pytest.raises(ValueError, match="finite float"):
        generate_gpt_frontend._validate_emotion_args(args)


def test_emotion_vector_random_indices_can_be_seeded():
    assert generate_gpt_frontend._emotion_vector_random_indices([4, 5, 6], 1234) == \
        generate_gpt_frontend._emotion_vector_random_indices([4, 5, 6], 1234)
    assert generate_gpt_frontend._emotion_vector_random_indices([4, 5, 6], 1234) != \
        generate_gpt_frontend._emotion_vector_random_indices([4, 5, 6], 1235)


def test_parse_emotion_vector_requires_eight_finite_values():
    parsed = generate_gpt_frontend._parse_emotion_vector("0,0.1,0,0,0,0,0,0.2")
    assert parsed == [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    with pytest.raises(ValueError, match="8"):
        generate_gpt_frontend._parse_emotion_vector("0,0")
    with pytest.raises(ValueError, match="finite"):
        generate_gpt_frontend._parse_emotion_vector("0,0,0,0,0,0,0,nan")


def test_normalize_emotion_vector_matches_webui_bias_and_sum_cap():
    values = [1.0] * 8
    normalized = generate_gpt_frontend._normalize_emotion_vector(values)
    assert pytest.approx(sum(normalized), abs=1e-7) == 0.8
    assert normalized[0] > normalized[1]
    assert normalized[-1] < normalized[0]


def test_emovec_file_is_mutually_exclusive_with_native_emovec(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_gpt_frontend, "load_indextts2", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        emovec_file=str(tmp_path / "emovec.f32"),
        emovec_mix=None,
        native_emovec=True,
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_gpt_frontend.run(args)


def test_emovec_mix_is_mutually_exclusive_with_other_emovec_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_gpt_frontend, "load_indextts2", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        emovec_file=str(tmp_path / "emovec.f32"),
        emovec_mix=[f"{tmp_path / 'other.f32'}=1.0"],
        native_emovec=False,
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_gpt_frontend.run(args)


def test_native_emovec_can_use_emotion_profile_source():
    args = argparse.Namespace(
        emovec_file=None,
        emovec_mix=None,
        native_emovec=True,
        emotion_audio=None,
        emotion_voice_name="emotion_ref",
        emotion_alpha=0.4,
    )
    generate_gpt_frontend._validate_emotion_args(args)


def test_explicit_emovec_override_rejects_emotion_source(tmp_path):
    args = argparse.Namespace(
        emovec_file=str(tmp_path / "emovec.f32"),
        emovec_mix=None,
        native_emovec=False,
        emotion_audio=str(tmp_path / "emotion.wav"),
        emotion_voice_name=None,
        emotion_alpha=1.0,
    )
    with pytest.raises(ValueError, match="emotion audio/profile"):
        generate_gpt_frontend._validate_emotion_args(args)


def test_emotion_vector_rejects_emotion_source(tmp_path):
    args = argparse.Namespace(
        emovec_file=None,
        emovec_mix=None,
        native_emovec=True,
        emotion_audio=str(tmp_path / "emotion.wav"),
        emotion_voice_name=None,
        emotion_alpha=1.0,
        emotion_vector="0,0,0,0,0,0,0,0",
    )
    with pytest.raises(ValueError, match="emotion audio/profile"):
        generate_gpt_frontend._validate_emotion_args(args)


def test_emotion_vector_random_seed_requires_random_mode():
    args = argparse.Namespace(
        emovec_file=None,
        emovec_mix=None,
        native_emovec=False,
        emotion_audio=None,
        emotion_voice_name=None,
        emotion_alpha=1.0,
        emotion_vector="0,0,0,0,0,0,0,0",
        emotion_vector_random=False,
        emotion_vector_random_seed=123,
    )
    with pytest.raises(ValueError, match="requires --emotion-vector-random"):
        generate_gpt_frontend._validate_emotion_args(args)
