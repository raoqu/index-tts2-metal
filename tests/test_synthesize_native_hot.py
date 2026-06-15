from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path

import pytest

from metal_indextts2.tools import synthesize_native_hot


def test_native_cjk_segments_helper_exports_segment_files(tmp_path, monkeypatch):
    calls = []

    def fake_run_runtime(command):
        calls.append(command)
        output_dir = Path(command[-1])
        output_dir.mkdir(parents=True, exist_ok=True)
        segment0 = output_dir / "segment_000.u32"
        segment1 = output_dir / "segment_001.u32"
        segment0.write_bytes(struct.pack("<4I", 10201, 208, 10201, 1260))
        segment1.write_bytes(struct.pack("<4I", 10201, 208, 10201, 1260))
        return (
            subprocess.CompletedProcess(command, 0, "", ""),
            [
                {
                    "format": "mit2-text-ids-cjk-segments",
                    "segments": [
                        {"index": 0, "output": str(segment0), "pieces": ["▁", "你", "▁", "好"]},
                        {"index": 1, "output": str(segment1), "pieces": ["▁", "你", "▁", "好"]},
                    ],
                }
            ],
            0.01,
        )

    monkeypatch.setattr(synthesize_native_hot, "_run_runtime", fake_run_runtime)
    args = argparse.Namespace(
        runtime="./build/mtts",
        tokenizer_dir=None,
        model_bundle="bin",
        text="你好你好",
        max_text_tokens_per_segment=4,
    )

    manifest = synthesize_native_hot._run_native_cjk_segments(args, tmp_path)

    assert calls[0][1] == "--export-text-ids-cjk-segments"
    assert calls[0][3] == "你好你好"
    assert calls[0][4] == "4"
    assert len(manifest["segments"]) == 2
    assert (tmp_path / "segment_000.u32").stat().st_size == 16


def _voice_args(tmp_path, **overrides):
    values = {
        "voice_name": "voice_a",
        "voice_bundle": None,
        "voice_output_dir": None,
        "voice_force_dtype": None,
        "validate_voice_source": True,
        "clone_audio": None,
        "clone_name": None,
        "overwrite_voice": False,
        "verbose": False,
        "index_tts_repo": "/repo",
        "checkpoint_dir": "/ckpt",
        "cfg_path": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _synthesis_args(tmp_path, **overrides):
    values = {
        "voice_name": "voice_a",
        "voice_bundle": str(tmp_path / "voice"),
        "prompt_audio": None,
        "clone_audio": None,
        "prompt_tokens": 8,
        "max_text_tokens_per_segment": 120,
        "max_mel_tokens": 3,
        "steps": 2,
        "cfg_rate": 0.7,
        "temperature": 1.0,
        "interval_silence_ms": 200,
        "native_emovec": True,
        "native_emovec_input_tokens": 0,
        "gpt_temperature": 0.8,
        "gpt_top_k": 30,
        "gpt_top_p": 0.8,
        "gpt_repetition_penalty": 10.0,
        "emotion_vector": None,
        "emotion_vector_random": False,
        "emotion_vector_random_seed": None,
        "all_segments": False,
        "text_ids_file": None,
        "noise_file": None,
        "gpt_do_sample": False,
        "shared_runtime_stages": False,
        "work_dir": str(tmp_path / "work"),
        "output_wav": str(tmp_path / "out.wav"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_generation_args_rejects_invalid_sampling_controls(tmp_path):
    invalid_cases = [
        ({"max_text_tokens_per_segment": 0}, "--max-text-tokens-per-segment"),
        ({"max_mel_tokens": 0}, "--max-mel-tokens"),
        ({"steps": 0}, "--steps"),
        ({"cfg_rate": -0.1}, "--cfg-rate"),
        ({"temperature": float("nan")}, "--temperature"),
        ({"gpt_temperature": 0.0}, "--gpt-temperature"),
        ({"gpt_top_k": -1}, "--gpt-top-k"),
        ({"gpt_top_p": 1.1}, "--gpt-top-p"),
        ({"gpt_repetition_penalty": float("inf")}, "--gpt-repetition-penalty"),
    ]

    for overrides, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            synthesize_native_hot._validate_generation_args(_synthesis_args(tmp_path, **overrides))


def test_run_validates_generation_args_before_runtime_work(tmp_path):
    with pytest.raises(ValueError, match="--max-mel-tokens"):
        synthesize_native_hot.run(_synthesis_args(tmp_path, max_mel_tokens=0))


def test_run_validates_emovec_file_before_runtime_work(tmp_path):
    bad = tmp_path / "bad_emovec.f32"
    bad.write_bytes(b"\0" * 4)
    args = _synthesis_args(
        tmp_path,
        native_emovec=False,
        emovec_file=str(bad),
        emovec_mix=None,
    )

    with pytest.raises(ValueError, match="--emovec-file.*exactly 1280"):
        synthesize_native_hot.run(args)

    assert not Path(args.work_dir).exists()


def test_prepare_voice_validates_auto_converted_cached_profile(tmp_path, monkeypatch):
    calls = []
    profile = tmp_path / "voice_a.pt"

    def fake_convert_voice(path, output_dir, *, force_dtype=None):
        calls.append(("convert", path, output_dir, force_dtype))
        return {"metadata": {"voice_name": "voice_a"}}

    def fake_validate(path, output_dir, *, force_dtype=None):
        calls.append(("validate", path, output_dir, force_dtype))
        return {"format": "mit2-voice-bundle-validation", "ok": True}

    monkeypatch.setattr(synthesize_native_hot, "_resolve_voice_profile_path", lambda args, voice_name: profile)
    monkeypatch.setattr(synthesize_native_hot.convert_voice, "convert_voice", fake_convert_voice)
    monkeypatch.setattr(synthesize_native_hot.convert_voice, "validate_voice_bundle", fake_validate)

    voice_name, voice_bundle, voice_manifest, voice_validation = synthesize_native_hot._prepare_voice(_voice_args(tmp_path), tmp_path)

    assert voice_name == "voice_a"
    assert Path(voice_bundle) == tmp_path / "voice"
    assert voice_manifest == {"metadata": {"voice_name": "voice_a"}}
    assert voice_validation == {"format": "mit2-voice-bundle-validation", "ok": True}
    assert calls == [
        ("convert", profile, tmp_path / "voice", None),
        ("validate", profile, tmp_path / "voice", None),
    ]


def test_prepare_voice_validates_auto_converted_clone_profile(tmp_path, monkeypatch):
    class FakeTTS:
        def clone_voice(self, clone_audio, *, voice_name, overwrite, verbose):
            assert clone_audio == "refs/a.wav"
            assert voice_name == "clone_a"
            assert overwrite is True
            assert verbose is False
            return "clone_a"

    profile = tmp_path / "clone_a.pt"
    monkeypatch.setattr(synthesize_native_hot, "load_indextts2", lambda *args, **kwargs: FakeTTS())
    monkeypatch.setattr(synthesize_native_hot, "_resolve_voice_profile_path", lambda args, voice_name: profile)
    monkeypatch.setattr(
        synthesize_native_hot.convert_voice,
        "convert_voice",
        lambda path, output_dir, *, force_dtype=None: {"metadata": {"voice_name": "clone_a"}},
    )
    monkeypatch.setattr(
        synthesize_native_hot.convert_voice,
        "validate_voice_bundle",
        lambda path, output_dir, *, force_dtype=None: {"format": "mit2-voice-bundle-validation", "ok": True},
    )

    args = _voice_args(
        tmp_path,
        voice_name=None,
        clone_audio="refs/a.wav",
        clone_name="clone_a",
        overwrite_voice=True,
        voice_output_dir=str(tmp_path / "clone_bundle"),
    )

    voice_name, voice_bundle, voice_manifest, voice_validation = synthesize_native_hot._prepare_voice(args, tmp_path)

    assert voice_name == "clone_a"
    assert Path(voice_bundle) == tmp_path / "clone_bundle"
    assert voice_manifest["metadata"]["voice_name"] == "clone_a"
    assert voice_validation["ok"] is True


def test_prepare_voice_source_validation_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(synthesize_native_hot, "_resolve_voice_profile_path", lambda args, voice_name: tmp_path / "voice_a.pt")
    monkeypatch.setattr(
        synthesize_native_hot.convert_voice,
        "convert_voice",
        lambda path, output_dir, *, force_dtype=None: {"metadata": {"voice_name": "voice_a"}},
    )
    monkeypatch.setattr(
        synthesize_native_hot.convert_voice,
        "validate_voice_bundle",
        lambda path, output_dir, *, force_dtype=None: {"format": "mit2-voice-bundle-validation", "ok": False},
    )

    try:
        synthesize_native_hot._prepare_voice(_voice_args(tmp_path), tmp_path)
    except ValueError as exc:
        assert "failed source validation" in str(exc)
    else:
        raise AssertionError("source validation failure should raise")


def test_validate_model_contract_uses_native_runtime(tmp_path, monkeypatch):
    calls = []
    model_bundle = tmp_path / "model"
    model_bundle.mkdir()

    def fake_run_runtime(command):
        calls.append(command)
        return (
            subprocess.CompletedProcess(command, 0, '{"stage":"model_bundle_contract","ok":true}', ""),
            [
                {
                    "stage": "model_bundle_contract",
                    "ok": True,
                    "model_bundle_dir": str(model_bundle),
                    "tensor_count": 2755,
                    "required_tensor_count": 24,
                }
            ],
            0.03,
        )

    monkeypatch.setattr(synthesize_native_hot, "_run_runtime", fake_run_runtime)
    runtime = tmp_path / "mit2_runtime"
    args = argparse.Namespace(runtime=str(runtime), model_bundle=str(model_bundle))

    report = synthesize_native_hot._validate_model_contract(args)

    assert calls == [[str(runtime), "--inspect-model-bundle", str(model_bundle)]]
    assert report["format"] == "mit2-native-model-contract-validation"
    assert report["ok"] is True
    assert report["runtime_seconds"] == 0.03
    assert report["runtime_report"]["required_tensor_count"] == 24


def test_validate_model_contract_missing_report_raises(monkeypatch):
    monkeypatch.setattr(
        synthesize_native_hot,
        "_run_runtime",
        lambda command: (subprocess.CompletedProcess(command, 0, "{}", ""), [{"stage": "other", "ok": True}], 0.01),
    )
    args = argparse.Namespace(runtime="./build/mtts", model_bundle="artifacts/model")

    try:
        synthesize_native_hot._validate_model_contract(args)
    except ValueError as exc:
        assert "model_bundle_contract" in str(exc)
    else:
        raise AssertionError("missing native model contract report should raise")


def test_validate_model_contract_failure_raises(monkeypatch):
    monkeypatch.setattr(
        synthesize_native_hot,
        "_run_runtime",
        lambda command: (
            subprocess.CompletedProcess(command, 0, '{"stage":"model_bundle_contract","ok":false}', ""),
            [{"stage": "model_bundle_contract", "ok": False}],
            0.01,
        ),
    )
    args = argparse.Namespace(runtime="./build/mtts", model_bundle="artifacts/model")

    try:
        synthesize_native_hot._validate_model_contract(args)
    except ValueError as exc:
        assert "failed contract validation" in str(exc)
    else:
        raise AssertionError("failed native model contract report should raise")


def test_validate_voice_contract_uses_native_runtime(tmp_path, monkeypatch):
    calls = []
    voice_bundle = tmp_path / "voice"
    voice_bundle.mkdir()

    def fake_run_runtime(command):
        calls.append(command)
        return (
            subprocess.CompletedProcess(command, 0, '{"stage":"voice_bundle_contract","ok":true}', ""),
            [
                {
                    "stage": "voice_bundle_contract",
                    "ok": True,
                    "voice_bundle_dir": str(voice_bundle),
                    "prompt_tokens": 944,
                }
            ],
            0.02,
        )

    monkeypatch.setattr(synthesize_native_hot, "_run_runtime", fake_run_runtime)
    runtime = tmp_path / "mit2_runtime"
    args = argparse.Namespace(runtime=str(runtime))

    report = synthesize_native_hot._validate_voice_contract(args, str(voice_bundle))

    assert calls == [[str(runtime), "--inspect-voice-bundle", str(voice_bundle)]]
    assert report["format"] == "mit2-native-voice-contract-validation"
    assert report["ok"] is True
    assert report["runtime_seconds"] == 0.02
    assert report["runtime_report"]["prompt_tokens"] == 944


def test_validate_voice_contract_missing_report_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        synthesize_native_hot,
        "_run_runtime",
        lambda command: (subprocess.CompletedProcess(command, 0, "{}", ""), [{"stage": "other", "ok": True}], 0.01),
    )
    args = argparse.Namespace(runtime="./build/mtts")

    try:
        synthesize_native_hot._validate_voice_contract(args, str(tmp_path / "voice"))
    except ValueError as exc:
        assert "voice_bundle_contract" in str(exc)
    else:
        raise AssertionError("missing native voice contract report should raise")


def test_validate_voice_contract_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        synthesize_native_hot,
        "_run_runtime",
        lambda command: (
            subprocess.CompletedProcess(command, 0, '{"stage":"voice_bundle_contract","ok":false}', ""),
            [{"stage": "voice_bundle_contract", "ok": False}],
            0.01,
        ),
    )
    args = argparse.Namespace(runtime="./build/mtts")

    try:
        synthesize_native_hot._validate_voice_contract(args, str(tmp_path / "voice"))
    except ValueError as exc:
        assert "failed contract validation" in str(exc)
    else:
        raise AssertionError("failed native voice contract report should raise")


def test_run_single_segment_reports_frontend_tensor_summaries(tmp_path, monkeypatch):
    def fake_frontend_run(args):
        return {
            "format": "mit2-gpt-frontend",
            "emovec_source": "native_full_metal_subsampling_conformer_perceiver_linear",
            "emotion_source": "speaker",
            "emotion_alpha": 1.0,
            "text_ids_source": "native_cjk",
            "text_ids": [10201, 208, 10201, 1260],
            "segment_tokens": ["▁", "你", "▁", "好"],
            "tensors": {
                "conds_latent": {"dtype": "float32", "shape": [34, 1280], "sha256": "conds-sha"},
                "speech_conditioning_latent": {"dtype": "float32", "shape": [32, 1280], "sha256": "speech-sha"},
                "text_ids": {"dtype": "uint32", "shape": [4], "sha256": "text-sha"},
                "fake_inputs": {"dtype": "uint32", "shape": [41], "sha256": "fake-sha"},
                "inputs_embeds": {"dtype": "float32", "shape": [40, 1280], "sha256": "embeds-sha"},
                "attention_mask": {"dtype": "uint32", "shape": [41], "sha256": "mask-sha"},
            },
            "generation": {"emovec_mix": None},
        }

    def fake_run_runtime(command):
        output_wav = Path(command[-1])
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"wav")
        output_wav.with_name(output_wav.name + ".codes.u32").write_bytes(struct.pack("<I", 4039))
        return (
            subprocess.CompletedProcess(command, 0, "{}", ""),
            [
                {
                    "stage": "gpt_kv_codes_inputs_export",
                    "elapsed_seconds": 1.0,
                    "predicted_codes": [4039],
                    "raw_codes_per_second": 1.0,
                    "seconds_per_raw_code": 1.0,
                    "codes_per_second": 1.0,
                    "seconds_per_code": 1.0,
                },
                {
                    "stage": "hot_tts_inputs_seeded_wav",
                    "elapsed_seconds": 2.0,
                    "gpt_seconds": 1.0,
                    "condition_seconds": 0.5,
                    "noise_seconds": 0.1,
                    "acoustic_seconds": 0.4,
                    "predicted_codes": [4039],
                    "resident_peak_bytes": 1234,
                    "gpt_command_buffers_submitted": 3,
                    "condition_command_buffers_submitted": 2,
                    "gpt_buffer_allocations": 10,
                    "condition_buffer_allocations": 4,
                    "gpt_buffer_bytes_allocated": 100,
                    "acoustic_gpu_elapsed_seconds": 0.25,
                    "planned_scratch_ok": True,
                }
            ],
            0.01,
        )

    monkeypatch.setattr(synthesize_native_hot.generate_gpt_frontend, "run", fake_frontend_run)
    monkeypatch.setattr(synthesize_native_hot, "_run_runtime", fake_run_runtime)
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/ckpt",
        cfg_path=None,
        prompt_audio=None,
        emotion_audio=None,
        emotion_voice_name=None,
        emotion_alpha=1.0,
        emotion_vector="0,0,0,0,0,0,0,0",
        emotion_vector_raw=False,
        emotion_vector_random=True,
        emotion_vector_random_seed=321,
        text="你好",
        text_ids_file=None,
        native_cjk_text_ids=True,
        tokenizer_dir=None,
        native_subsampling=True,
        native_emovec=True,
        emovec_file=None,
        emovec_mix=None,
        native_emovec_input_tokens=0,
        native_conformer_stack=True,
        native_perceiver=True,
        native_frontend_tail=True,
        model_bundle=str(tmp_path / "model"),
        runtime="./runtime",
        max_text_tokens_per_segment=120,
        segment_index=0,
        max_mel_tokens=1,
        noise_file=None,
        gpt_do_sample=False,
        seed=20240605,
        temperature=1.0,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        steps=1,
        cfg_rate=0.7,
        shared_runtime_stages=False,
        clone_audio=None,
    )

    report = synthesize_native_hot._run_single_segment(
        args,
        work_dir=tmp_path / "work",
        output_wav=tmp_path / "out.wav",
        voice_name="voice",
        voice_bundle=str(tmp_path / "voice"),
        voice_manifest=None,
        voice_validation=None,
        model_contract=None,
        voice_contract=None,
        prompt_tokens=8,
        segment_index=0,
        frontend_dir_name="gpt_frontend",
    )

    assert report["gpt"]["conds_latent"]["sha256"] == "conds-sha"
    assert report["gpt"]["speech_conditioning_latent"]["sha256"] == "speech-sha"
    assert report["gpt"]["text_ids_tensor"]["sha256"] == "text-sha"
    assert report["gpt"]["fake_inputs"]["sha256"] == "fake-sha"
    assert report["gpt"]["inputs_embeds"]["sha256"] == "embeds-sha"
    assert report["gpt"]["attention_mask"]["sha256"] == "mask-sha"
    assert report["generation"]["emotion_vector_random"] is True
    assert report["generation"]["emotion_vector_random_seed"] == 321
    assert report["runtime_summary"]["stage_counts"]["hot_tts_inputs_seeded_wav"] == 1
    assert report["runtime_summary"]["stage_counts"]["gpt_kv_codes_inputs_export"] == 1
    assert report["runtime_summary"]["native_total_seconds"] == 2.0
    assert report["runtime_summary"]["native_phase_seconds"]["gpt_seconds"] == 1.0
    assert report["runtime_summary"]["resident_peak_bytes"] == 1234
    assert report["runtime_summary"]["command_buffers_submitted"] == 5
    assert report["runtime_summary"]["buffer_allocations"] == 14
    assert report["runtime_summary"]["buffer_bytes_allocated"] == 100
    assert report["runtime_summary"]["gpu_elapsed_seconds"] == 0.25
    assert report["runtime_summary"]["predicted_codes"] == [4039]
    assert report["runtime_summary"]["planned_scratch"] == {"planned_scratch_ok": True}
    assert report["runtime_summary"]["gpt_decode"]["raw_codes_per_second"] == 1.0
    assert report["runtime_summary"]["gpt_decode"]["seconds_per_raw_code"] == 1.0


def test_combine_runtime_summaries_aggregates_segments():
    summary = synthesize_native_hot._combine_runtime_summaries(
        [
            {
                "stage_counts": {"hot_tts_inputs_seeded_wav": 1},
                "stage_seconds": {"hot_tts_inputs_seeded_wav": 2.0},
                "native_total_seconds": 2.0,
                "native_phase_seconds": {"gpt_seconds": 1.0, "acoustic_seconds": 0.5},
                "resident_peak_bytes": 1000,
                "command_buffers_submitted": 3,
                "buffer_allocations": 10,
                "buffer_bytes_allocated": 100,
                "gpu_elapsed_seconds": 0.2,
                "predicted_codes": [1],
                "planned_scratch": {
                    "planned_scratch_ok": True,
                    "planned_scratch_alignment": 256,
                    "planned_scratch_source": "inputs",
                    "planned_scratch_max_prefix_tokens": 40,
                    "planned_scratch_max_codes": 3,
                    "planned_scratch_prompt_tokens": 8,
                    "planned_scratch_generated_tokens": 5,
                    "planned_scratch_total_mel_tokens": 13,
                    "planned_scratch_gpt_phase_peak_bytes": 1000,
                    "planned_scratch_acoustic_phase_peak_bytes": 4000,
                    "planned_scratch_unshared_phase_peak_total_bytes": 6000,
                    "planned_scratch_capacity_bytes": 5000,
                    "planned_scratch_reuse_saves_bytes": 1000,
                    "planned_scratch_actual_codes": 1,
                    "planned_scratch_actual_generated_tokens": 5,
                    "planned_scratch_actual_total_mel_tokens": 13,
                    "planned_scratch_code_slack": 2,
                    "planned_scratch_generated_token_slack": 0,
                    "planned_scratch_total_mel_token_slack": 0,
                    "planned_scratch_covers_actual": True,
                },
            },
            {
                "stage_counts": {"hot_tts_inputs_seeded_wav": 1, "other": 1},
                "stage_seconds": {"hot_tts_inputs_seeded_wav": 3.0, "other": 0.5},
                "native_total_seconds": 3.0,
                "native_phase_seconds": {"gpt_seconds": 2.0, "condition_seconds": 0.25},
                "resident_peak_bytes": 1500,
                "command_buffers_submitted": 4,
                "buffer_allocations": 20,
                "buffer_bytes_allocated": 200,
                "gpu_elapsed_seconds": 0.3,
                "predicted_codes": [2, 3],
                "planned_scratch": {
                    "planned_scratch_ok": True,
                    "planned_scratch_alignment": 256,
                    "planned_scratch_source": "inputs",
                    "planned_scratch_max_prefix_tokens": 42,
                    "planned_scratch_max_codes": 4,
                    "planned_scratch_prompt_tokens": 8,
                    "planned_scratch_generated_tokens": 7,
                    "planned_scratch_total_mel_tokens": 15,
                    "planned_scratch_gpt_phase_peak_bytes": 1200,
                    "planned_scratch_acoustic_phase_peak_bytes": 5000,
                    "planned_scratch_unshared_phase_peak_total_bytes": 8000,
                    "planned_scratch_capacity_bytes": 6500,
                    "planned_scratch_reuse_saves_bytes": 1500,
                    "planned_scratch_actual_codes": 2,
                    "planned_scratch_actual_generated_tokens": 7,
                    "planned_scratch_actual_total_mel_tokens": 15,
                    "planned_scratch_code_slack": 2,
                    "planned_scratch_generated_token_slack": 0,
                    "planned_scratch_total_mel_token_slack": 0,
                    "planned_scratch_covers_actual": True,
                },
            },
        ]
    )

    assert summary is not None
    assert summary["segments"] == 2
    assert summary["stage_counts"] == {"hot_tts_inputs_seeded_wav": 2, "other": 1}
    assert summary["stage_seconds"]["hot_tts_inputs_seeded_wav"] == 5.0
    assert summary["native_total_seconds"] == 5.0
    assert summary["native_phase_seconds"]["gpt_seconds"] == 3.0
    assert summary["resident_peak_bytes"] == 1500
    assert summary["command_buffers_submitted"] == 7
    assert summary["buffer_allocations"] == 30
    assert summary["buffer_bytes_allocated"] == 300
    assert summary["gpu_elapsed_seconds"] == 0.5
    assert summary["segment_predicted_codes"] == [[1], [2, 3]]
    assert summary["planned_scratch"]["planned_scratch_segments"] == 2
    assert summary["planned_scratch"]["planned_scratch_ok"] is True
    assert summary["planned_scratch"]["planned_scratch_covers_actual"] is True
    assert summary["planned_scratch"]["planned_scratch_alignment"] == 256
    assert summary["planned_scratch"]["planned_scratch_source"] == "inputs"
    assert summary["planned_scratch"]["planned_scratch_max_prefix_tokens"] == 42
    assert summary["planned_scratch"]["planned_scratch_max_codes"] == 4
    assert summary["planned_scratch"]["planned_scratch_capacity_bytes"] == 6500
    assert summary["planned_scratch"]["planned_scratch_acoustic_phase_peak_bytes"] == 5000
    assert summary["planned_scratch"]["planned_scratch_unshared_phase_peak_total_bytes"] == 14000
    assert summary["planned_scratch"]["planned_scratch_reuse_saves_bytes"] == 2500
    assert summary["planned_scratch"]["planned_scratch_actual_codes"] == 3
    assert summary["planned_scratch"]["planned_scratch_actual_generated_tokens"] == 12
    assert summary["planned_scratch"]["planned_scratch_code_slack"] == 4
    assert len(summary["segment_planned_scratch"]) == 2


def test_compact_report_payload_removes_raw_runtime_payloads():
    report = {
        "format": "mit2-native-hot-multisegment-synthesis-report",
        "runtime_json": [{"stage": "top"}],
        "runtime_stdout": "{}",
        "runtime_stderr": "debug",
        "runtime_summary": {"stage_counts": {"top": 1}},
        "segments": [
            {
                "index": 0,
                "report": {
                    "runtime_json": [{"stage": "segment"}],
                    "runtime_stdout": "{}",
                    "runtime_stderr": "debug",
                    "runtime_summary": {"stage_counts": {"segment": 1}},
                },
            }
        ],
    }

    compact = synthesize_native_hot._compact_report_payload(report)

    assert "runtime_json" not in compact
    assert "runtime_stdout" not in compact
    assert "runtime_stderr" not in compact
    assert compact["runtime_summary"] == {"stage_counts": {"top": 1}}
    nested = compact["segments"][0]["report"]
    assert "runtime_json" not in nested
    assert "runtime_stdout" not in nested
    assert "runtime_stderr" not in nested
    assert nested["runtime_summary"] == {"stage_counts": {"segment": 1}}
