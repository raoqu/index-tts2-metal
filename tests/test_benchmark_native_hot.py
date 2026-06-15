from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import pytest

import metal_indextts2.tools.benchmark_native_hot as benchmark_native_hot
from metal_indextts2.tools.benchmark_native_hot import (
    _case_args,
    _runtime_stage_metrics,
    _summarize_report,
    _validate_benchmark_args,
    apply_summary_budgets,
    compare_benchmark_summaries,
    compare_summary_resource_cases,
)


def _summary(
    *,
    wav_hash: str = "abc",
    peak: int = 1000,
    samples_per_second: float = 100.0,
    text_ids_source: str | None = None,
    generation_steps: int | None = None,
    generation_max_mel_tokens: int | None = None,
    model_contract_ok: bool | None = None,
    voice_validation_ok: bool | None = None,
    voice_contract_ok: bool | None = None,
    extra_case_fields: dict | None = None,
) -> dict:
    case = {
        "case": "short_greedy",
        "status": "passed",
        "output_wav_sha256": wav_hash,
        "predicted_codes": [1, 2, 3],
        "elapsed_seconds": 10.0,
        "rtf": 2.0,
        "runtime_peak_bytes": peak,
        "audio_samples_per_second": samples_per_second,
        "runtime_seconds": 5.0,
        "text_ids_source": text_ids_source,
    }
    if generation_steps is not None:
        case["generation_steps"] = generation_steps
    if generation_max_mel_tokens is not None:
        case["generation_max_mel_tokens"] = generation_max_mel_tokens
    if model_contract_ok is not None:
        case["model_contract_ok"] = model_contract_ok
    if voice_validation_ok is not None:
        case["voice_validation_ok"] = voice_validation_ok
    if voice_contract_ok is not None:
        case["voice_contract_ok"] = voice_contract_ok
    if extra_case_fields:
        case.update(extra_case_fields)
    return {
        "format": "mit2-native-hot-benchmark-summary",
        "version": 1,
        "cases": [case],
        "passed": 1,
        "failed": 0,
    }


def _case_namespace(**overrides):
    values = {
        "index_tts_repo": "/repo",
        "checkpoint_dir": "/repo/checkpoints",
        "python_executable": "/venv/bin/python",
        "model_bundle": "artifacts/model",
        "runtime": "./runtime",
        "max_mel_tokens": 1,
        "prompt_tokens": 8,
        "steps": 1,
        "full_acoustic_max_mel_tokens": 8,
        "full_acoustic_steps": 25,
        "cfg_rate": 0.7,
        "temperature": 1.0,
        "seed": 1,
        "voice_bundle": "artifacts/voice",
        "voice_name": "voice",
        "validate_voice_source": False,
        "validate_model_contract": False,
        "validate_voice_contract": False,
        "text_ids_file": "artifacts/native_text_ids_cjk.u32",
        "native_flags": True,
        "short_text": "短",
        "medium_text": "中等长度文本",
        "native_cjk_mixed_text": "AB1C2你好",
        "multi_text": "多段",
        "long_text": "长文本",
        "segment_tokens": 4,
        "long_segment_tokens": 80,
        "interval_silence_ms": 200,
        "gpt_temperature": 0.8,
        "gpt_top_k": 30,
        "gpt_top_p": 0.8,
        "gpt_repetition_penalty": 10.0,
        "clone_audio": None,
        "clone_name": "clone",
        "shared_runtime_stages": False,
        "emotion_voice_name": None,
        "emotion_audio": None,
        "emotion_alpha": 1.0,
        "emotion_vector": "0,0,0,0,0,0,0,0",
        "emotion_vector_raw": False,
        "emotion_vector_random": False,
        "emotion_vector_random_seed": None,
        "cases": ["short_greedy"],
        "from_reports": None,
        "baseline_summary": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_compare_benchmark_summaries_passes_identical_summary():
    comparison = compare_benchmark_summaries(_summary(), _summary())
    assert comparison["failed"] == 0
    assert comparison["cases"][0]["status"] == "passed"


def test_run_records_compact_report_config(tmp_path, monkeypatch):
    def fake_run_case(args, case, out_dir):
        return {"case": case, "status": "passed"}

    monkeypatch.setattr(benchmark_native_hot, "_run_case", fake_run_case)
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        from_reports=None,
        cases=["short_greedy"],
        baseline_summary=None,
        compact_reports=True,
        runtime="./runtime",
        model_bundle="artifacts/model",
        voice_bundle="artifacts/voice",
        seed=20240605,
        max_mel_tokens=3,
        steps=2,
        max_rtf=5.0,
        require_hash_match=True,
    )

    summary = benchmark_native_hot.run(args)
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["config"]["compact_reports"] is True
    assert summary["config"]["cases"] == ["short_greedy"]
    assert summary["config"]["runtime"] == "./runtime"
    assert summary["config"]["model_bundle"] == "artifacts/model"
    assert summary["config"]["voice_bundle"] == "artifacts/voice"
    assert summary["config"]["seed"] == 20240605
    assert summary["config"]["max_mel_tokens"] == 3
    assert summary["config"]["steps"] == 2
    assert summary["config"]["max_rtf"] == 5.0
    assert summary["config"]["require_hash_match"] is True
    assert written["config"]["compact_reports"] is True
    assert written["config"]["cases"] == ["short_greedy"]


def test_validate_benchmark_args_rejects_invalid_generation_controls():
    invalid_cases = [
        ({"max_mel_tokens": 0}, "--max-mel-tokens"),
        ({"prompt_tokens": 0}, "--prompt-tokens"),
        ({"steps": 0}, "--steps"),
        ({"full_acoustic_max_mel_tokens": 0}, "--full-acoustic-max-mel-tokens"),
        ({"full_acoustic_steps": 0}, "--full-acoustic-steps"),
        ({"segment_tokens": 0}, "--segment-tokens"),
        ({"long_segment_tokens": 0}, "--long-segment-tokens"),
        ({"interval_silence_ms": -1}, "--interval-silence-ms"),
        ({"cfg_rate": -0.1}, "--cfg-rate"),
        ({"temperature": float("nan")}, "--temperature"),
        ({"gpt_temperature": 0.0}, "--gpt-temperature"),
        ({"gpt_top_k": -1}, "--gpt-top-k"),
        ({"gpt_top_p": 1.1}, "--gpt-top-p"),
        ({"gpt_repetition_penalty": float("inf")}, "--gpt-repetition-penalty"),
    ]

    for overrides, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            _validate_benchmark_args(_case_namespace(**overrides))


def test_validate_benchmark_args_rejects_invalid_budget_controls():
    invalid_cases = [
        ({"max_rtf_regression": -0.1}, "--max-rtf-regression"),
        ({"max_elapsed_regression": float("nan")}, "--max-elapsed-regression"),
        ({"max_memory_regression": -0.1}, "--max-memory-regression"),
        ({"max_throughput_regression": -0.1}, "--max-throughput-regression"),
        ({"max_runtime_peak_bytes": -1}, "--max-runtime-peak-bytes"),
        ({"max_runtime_command_buffers_submitted": -1}, "--max-runtime-command-buffers-submitted"),
        ({"max_runtime_buffer_allocations": -1}, "--max-runtime-buffer-allocations"),
        ({"max_runtime_buffer_bytes_allocated": -1}, "--max-runtime-buffer-bytes-allocated"),
        ({"max_runtime_gpu_elapsed_seconds": -0.1}, "--max-runtime-gpu-elapsed-seconds"),
        ({"max_rtf": -0.1}, "--max-rtf"),
        ({"max_elapsed_seconds": -0.1}, "--max-elapsed-seconds"),
        ({"min_audio_samples_per_second": -0.1}, "--min-audio-samples-per-second"),
        ({"max_planned_scratch_capacity_bytes": -1}, "--max-planned-scratch-capacity-bytes"),
        ({"min_planned_scratch_reuse_saves_bytes": -1}, "--min-planned-scratch-reuse-saves-bytes"),
        ({"min_planned_scratch_reuse_saves_ratio": float("inf")}, "--min-planned-scratch-reuse-saves-ratio"),
    ]

    for overrides, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            _validate_benchmark_args(_case_namespace(**overrides))


def test_validate_benchmark_args_rejects_invalid_case_configuration():
    invalid_cases = [
        ({"cases": []}, "--cases requires at least one benchmark case"),
        ({"cases": ["missing_case"]}, "unknown benchmark case"),
        ({"cases": ["clone"], "clone_audio": None}, "clone cases require --clone-audio"),
        ({"cases": ["emotion_profile"], "emotion_voice_name": None}, "emotion_profile case requires --emotion-voice-name"),
        ({"cases": ["emotion_audio"], "emotion_audio": None}, "emotion_audio case requires --emotion-audio"),
        ({"from_reports": []}, "--from-reports requires at least one"),
        ({"from_reports": ["short_greedy"]}, "--from-reports entries must be CASE=REPORT_JSON"),
        ({"from_reports": ["=report.json"]}, "--from-reports case name must be non-empty"),
        ({"from_reports": ["short_greedy="]}, "--from-reports report path must be non-empty"),
        ({"from_reports": ["short_greedy=/missing/report.json"]}, "--from-reports report does not exist"),
        ({"cases": ["short_greedy", "short_greedy"]}, "duplicate benchmark case: short_greedy"),
        ({"baseline_summary": "/missing/baseline.json"}, "--baseline-summary does not exist"),
    ]

    for overrides, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            _validate_benchmark_args(_case_namespace(**overrides))


def test_run_validates_benchmark_args_before_writing_output(tmp_path):
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=None, cases=["short_greedy"], max_mel_tokens=0)

    with pytest.raises(ValueError, match="--max-mel-tokens"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_case_configuration_before_writing_output(tmp_path):
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=None, cases=["missing_case"])

    with pytest.raises(ValueError, match="unknown benchmark case"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_duplicate_cases_before_writing_output(tmp_path):
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=None, cases=["short_greedy", "short_greedy"])

    with pytest.raises(ValueError, match="duplicate benchmark case: short_greedy"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_report_paths_before_writing_output(tmp_path):
    out_dir = tmp_path / "bench"
    args = _case_namespace(
        output_dir=str(out_dir),
        from_reports=[f"short_greedy={tmp_path / 'missing.json'}"],
        cases=["short_greedy"],
    )

    with pytest.raises(ValueError, match="--from-reports report does not exist"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_duplicate_from_reports_before_writing_output(tmp_path):
    wav_path = tmp_path / "out.wav"
    with wave.open(str(wav_path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(22050)
        fp.writeframes(b"\x00\x00" * 4)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-synthesis-report",
                "output_wav": str(wav_path),
                "output_wav_sha256": "abc",
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "bench"
    args = _case_namespace(
        output_dir=str(out_dir),
        from_reports=[f"short_greedy={report_path}", f"short_greedy={report_path}"],
    )

    with pytest.raises(ValueError, match="duplicate benchmark case: short_greedy"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_report_json_before_writing_output(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text("{not json", encoding="utf-8")
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=[f"short_greedy={report_path}"])

    with pytest.raises(ValueError, match="--from-reports report must be valid JSON"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_report_shape_before_writing_output(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"format": "mit2-native-hot-synthesis-report"}), encoding="utf-8")
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=[f"short_greedy={report_path}"])

    with pytest.raises(ValueError, match="--from-reports report missing output_wav"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_report_wav_or_hash_before_writing_output(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-synthesis-report",
                "output_wav": str(tmp_path / "missing.wav"),
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=[f"short_greedy={report_path}"])

    with pytest.raises(ValueError, match="output_wav does not exist and output_wav_sha256 is missing"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_accepts_archived_report_with_output_hash(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-synthesis-report",
                "output_wav": str(tmp_path / "missing.wav"),
                "output_wav_sha256": "abc",
                "elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=[f"short_greedy={report_path}"])

    summary = benchmark_native_hot.run(args)

    assert summary["failed"] == 0
    assert summary["cases"][0]["output_wav_sha256"] == "abc"
    assert summary["cases"][0]["audio_seconds"] is None


def test_run_resolves_relative_report_wav_from_report_dir(tmp_path):
    report_dir = tmp_path / "case"
    report_dir.mkdir()
    wav_path = report_dir / "out.wav"
    with wave.open(str(wav_path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(22050)
        fp.writeframes(b"\x00\x00" * 4)
    report_path = report_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-synthesis-report",
                "output_wav": "out.wav",
                "elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=[f"short_greedy={report_path}"])

    summary = benchmark_native_hot.run(args)

    assert summary["failed"] == 0
    assert summary["cases"][0]["output_wav"] == str(wav_path.resolve())
    assert summary["cases"][0]["output_wav_sha256"] == hashlib.sha256(wav_path.read_bytes()).hexdigest()
    assert summary["cases"][0]["audio_seconds"] == 4 / 22050


def test_run_validates_baseline_summary_before_writing_output(tmp_path):
    out_dir = tmp_path / "bench"
    args = _case_namespace(
        output_dir=str(out_dir),
        from_reports=None,
        cases=["short_greedy"],
        baseline_summary=str(tmp_path / "missing_baseline.json"),
    )

    with pytest.raises(ValueError, match="--baseline-summary does not exist"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_baseline_summary_json_before_writing_output(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{not json", encoding="utf-8")
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=None, cases=["short_greedy"], baseline_summary=str(baseline_path))

    with pytest.raises(ValueError, match="--baseline-summary must be valid JSON"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_run_validates_baseline_summary_shape_before_writing_output(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"format": "mit2-native-hot-benchmark-summary"}), encoding="utf-8")
    out_dir = tmp_path / "bench"
    args = _case_namespace(output_dir=str(out_dir), from_reports=None, cases=["short_greedy"], baseline_summary=str(baseline_path))

    with pytest.raises(ValueError, match="benchmark summary must contain a cases list"):
        benchmark_native_hot.run(args)

    assert not out_dir.exists()


def test_main_reports_benchmark_validation_as_system_exit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark_native_hot", "--max-mel-tokens", "0"])

    with pytest.raises(SystemExit, match="--max-mel-tokens must be positive"):
        benchmark_native_hot.main()


def test_compare_benchmark_summaries_fails_correctness_and_memory_regression():
    comparison = compare_benchmark_summaries(
        _summary(wav_hash="changed", peak=1300),
        _summary(),
        max_memory_regression=0.10,
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["output_wav_sha256"]["status"] == "failed"
    assert checks["runtime_peak_bytes"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_throughput_regression():
    comparison = compare_benchmark_summaries(
        _summary(samples_per_second=80.0),
        _summary(samples_per_second=100.0),
        max_throughput_regression=0.10,
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["audio_samples_per_second"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_stage_timing_regression():
    current = _summary()
    current["cases"][0]["runtime_seconds"] = 6.0
    comparison = compare_benchmark_summaries(current, _summary(), max_elapsed_regression=0.10)
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["runtime_seconds"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_command_buffer_regression():
    current = _summary()
    current["cases"][0]["runtime_command_buffers_submitted"] = 120
    baseline = _summary()
    baseline["cases"][0]["runtime_command_buffers_submitted"] = 100
    comparison = compare_benchmark_summaries(current, baseline, max_elapsed_regression=0.10)
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["runtime_command_buffers_submitted"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_runtime_resource_regression():
    current = _summary()
    current["cases"][0]["runtime_buffer_allocations"] = 120
    current["cases"][0]["runtime_buffer_bytes_allocated"] = 1200
    current["cases"][0]["runtime_gpu_elapsed_seconds"] = 1.2
    baseline = _summary()
    baseline["cases"][0]["runtime_buffer_allocations"] = 100
    baseline["cases"][0]["runtime_buffer_bytes_allocated"] = 1000
    baseline["cases"][0]["runtime_gpu_elapsed_seconds"] = 1.0
    comparison = compare_benchmark_summaries(current, baseline, max_elapsed_regression=0.10)
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["runtime_buffer_allocations"]["status"] == "failed"
    assert checks["runtime_buffer_bytes_allocated"]["status"] == "failed"
    assert checks["runtime_gpu_elapsed_seconds"]["status"] == "failed"


def test_apply_summary_budgets_marks_failed_case():
    cases = [
        {
            "case": "short_greedy",
            "status": "passed",
            "runtime_peak_bytes": 1200,
            "runtime_command_buffers_submitted": 10,
            "runtime_buffer_allocations": 20,
            "runtime_buffer_bytes_allocated": 2000,
            "runtime_gpu_elapsed_seconds": 1.5,
            "rtf": 2.5,
            "elapsed_seconds": 11.0,
            "audio_samples_per_second": 80.0,
            "planned_scratch_capacity_bytes": 2000,
            "planned_scratch_reuse_saves_bytes": 50,
            "planned_scratch_reuse_saves_ratio": 0.05,
        }
    ]

    checks = apply_summary_budgets(
        cases,
        max_runtime_peak_bytes=1000,
        max_runtime_command_buffers_submitted=9,
        max_runtime_buffer_allocations=25,
        max_runtime_buffer_bytes_allocated=1000,
        max_runtime_gpu_elapsed_seconds=2.0,
        max_rtf=3.0,
        max_elapsed_seconds=12.0,
        min_audio_samples_per_second=100.0,
        max_planned_scratch_capacity_bytes=1500,
        min_planned_scratch_reuse_saves_bytes=100,
        min_planned_scratch_reuse_saves_ratio=0.1,
    )

    assert cases[0]["status"] == "failed"
    by_field = {check["field"]: check for check in checks}
    assert by_field["runtime_peak_bytes"]["status"] == "failed"
    assert by_field["runtime_command_buffers_submitted"]["status"] == "failed"
    assert by_field["runtime_buffer_allocations"]["status"] == "passed"
    assert by_field["runtime_buffer_bytes_allocated"]["status"] == "failed"
    assert by_field["runtime_gpu_elapsed_seconds"]["status"] == "passed"
    assert by_field["rtf"]["status"] == "passed"
    assert by_field["elapsed_seconds"]["status"] == "passed"
    assert by_field["audio_samples_per_second"]["status"] == "failed"
    assert by_field["planned_scratch_capacity_bytes"]["status"] == "failed"
    assert by_field["planned_scratch_reuse_saves_bytes"]["status"] == "failed"
    assert by_field["planned_scratch_reuse_saves_ratio"]["status"] == "failed"


def test_apply_summary_budgets_skips_missing_fields_without_failing_case():
    cases = [{"case": "short_greedy", "status": "passed"}]

    checks = apply_summary_budgets(
        cases,
        max_runtime_peak_bytes=1000,
        max_runtime_command_buffers_submitted=1,
        max_runtime_buffer_allocations=1,
        max_runtime_buffer_bytes_allocated=1,
        max_runtime_gpu_elapsed_seconds=1.0,
        max_rtf=2.0,
        max_planned_scratch_capacity_bytes=1,
        min_planned_scratch_reuse_saves_bytes=1,
        min_planned_scratch_reuse_saves_ratio=0.1,
    )

    assert cases[0]["status"] == "passed"
    assert {check["status"] for check in checks} == {"skipped"}


def test_runtime_stage_metrics_aggregates_resource_stats():
    metrics = _runtime_stage_metrics(
        {
            "runtime_json": [
                {
                    "stage": "hot_tts_condition_inputs_export",
                    "gpt_command_buffers_submitted": 3,
                    "gpt_buffer_allocations": 10,
                    "gpt_buffer_bytes_allocated": 1000,
                    "gpt_gpu_elapsed_seconds": 0.25,
                    "condition_command_buffers_submitted": 1,
                    "condition_buffer_allocations": 4,
                    "condition_buffer_bytes_allocated": 200,
                    "condition_gpu_elapsed_seconds": 0.05,
                    "planned_scratch_ok": True,
                    "planned_scratch_unshared_phase_peak_total_bytes": 13435148,
                    "planned_scratch_capacity_bytes": 10600716,
                    "planned_scratch_reuse_saves_bytes": 2834432,
                    "planned_scratch_actual_codes": 1,
                    "planned_scratch_actual_generated_tokens": 1,
                    "planned_scratch_actual_total_mel_tokens": 9,
                    "planned_scratch_code_slack": 2,
                    "planned_scratch_generated_token_slack": 4,
                    "planned_scratch_total_mel_token_slack": 4,
                    "planned_scratch_covers_actual": True,
                },
                {
                    "stage": "gpt_kv_codes_inputs_export",
                    "predicted_codes": [4039, 6947],
                    "raw_codes_per_second": 48.0,
                    "seconds_per_raw_code": 1.0 / 48.0,
                    "codes_per_second": 48.0,
                    "seconds_per_code": 1.0 / 48.0,
                }
            ]
        },
        elapsed=1.0,
    )
    assert metrics["runtime_command_buffers_submitted"] == 4
    assert metrics["runtime_buffer_allocations"] == 14
    assert metrics["runtime_buffer_bytes_allocated"] == 1200
    assert abs(metrics["runtime_gpu_elapsed_seconds"] - 0.3) < 1e-12
    assert metrics["runtime_stage_buffer_allocations"]["hot_tts_condition_inputs_export"] == 14
    assert metrics["planned_scratch_ok"] is True
    assert metrics["planned_scratch_capacity_bytes"] == 10600716
    assert metrics["planned_scratch_reuse_saves_bytes"] == 2834432
    assert metrics["planned_scratch_capacity_to_unshared_ratio"] == 10600716 / 13435148
    assert metrics["planned_scratch_reuse_saves_ratio"] == 2834432 / 13435148
    assert metrics["planned_scratch_actual_codes"] == 1
    assert metrics["planned_scratch_code_slack"] == 2
    assert metrics["planned_scratch_covers_actual"] is True
    assert metrics["native_gpt_raw_codes_per_second"] == 48.0
    assert metrics["native_gpt_seconds_per_raw_code"] == 1.0 / 48.0
    assert metrics["native_gpt_output_codes_per_second"] == 48.0


def test_runtime_stage_metrics_uses_runtime_summary_without_runtime_json():
    metrics = _runtime_stage_metrics(
        {
            "runtime_summary": {
                "stage_counts": {"hot_tts_inputs_seeded_wav": 1},
                "stage_seconds": {"hot_tts_inputs_seeded_wav": 2.5},
                "native_total_seconds": 2.5,
                "native_phase_seconds": {
                    "gpt_seconds": 1.25,
                    "condition_seconds": 0.5,
                    "noise_seconds": 0.05,
                    "acoustic_seconds": 0.7,
                },
                "command_buffers_submitted": 5,
                "buffer_allocations": 14,
                "buffer_bytes_allocated": 1200,
                "gpu_elapsed_seconds": 0.3,
                "predicted_codes": [4039, 6947],
                "planned_scratch": {
                    "planned_scratch_unshared_phase_peak_total_bytes": 12000,
                    "planned_scratch_capacity_bytes": 10000,
                    "planned_scratch_reuse_saves_bytes": 2000,
                    "planned_scratch_ok": True,
                },
                "gpt_decode": {
                    "raw_codes_per_second": 24.0,
                    "seconds_per_raw_code": 1.0 / 24.0,
                    "codes_per_second": 20.0,
                    "seconds_per_code": 0.05,
                },
            },
        },
        elapsed=5.0,
    )

    assert metrics["runtime_stage_counts"] == {"hot_tts_inputs_seeded_wav": 1}
    assert metrics["runtime_stage_seconds"] == {"hot_tts_inputs_seeded_wav": 2.5}
    assert metrics["native_runtime_total_seconds"] == 2.5
    assert metrics["native_gpt_seconds"] == 1.25
    assert metrics["native_condition_seconds"] == 0.5
    assert metrics["native_acoustic_seconds"] == 0.7
    assert metrics["runtime_command_buffers_submitted"] == 5
    assert metrics["runtime_buffer_allocations"] == 14
    assert metrics["runtime_buffer_bytes_allocated"] == 1200
    assert metrics["runtime_gpu_elapsed_seconds"] == 0.3
    assert metrics["gpt_codes"] == 2
    assert metrics["gpt_codes_per_second"] == 0.4
    assert metrics["native_gpt_raw_codes_per_second"] == 24.0
    assert metrics["native_gpt_seconds_per_raw_code"] == 1.0 / 24.0
    assert metrics["native_gpt_output_codes_per_second"] == 20.0
    assert metrics["native_gpt_seconds_per_output_code"] == 0.05
    assert metrics["planned_scratch_ok"] is True
    assert metrics["planned_scratch_capacity_to_unshared_ratio"] == 10000 / 12000
    assert metrics["planned_scratch_reuse_saves_ratio"] == 2000 / 12000


def test_compare_summary_resource_cases_pairs_short_and_shared():
    summary = {
        "format": "mit2-native-hot-benchmark-summary",
        "version": 1,
        "cases": [
            {
                "case": "short_greedy",
                "status": "passed",
                "runtime_peak_bytes": 1000,
                "runtime_buffer_allocations": 100,
                "runtime_gpu_elapsed_seconds": 1.0,
                "native_runtime_total_seconds": 3.0,
                "planned_scratch_ok": True,
                "planned_scratch_segments": 2,
                "planned_scratch_max_prefix_tokens": 40,
                "planned_scratch_capacity_bytes": 10000,
                "planned_scratch_unshared_phase_peak_total_bytes": 12000,
                "planned_scratch_reuse_saves_bytes": 2000,
                "planned_scratch_capacity_to_unshared_ratio": 10000 / 12000,
                "planned_scratch_reuse_saves_ratio": 2000 / 12000,
                "planned_scratch_actual_codes": 1,
                "planned_scratch_actual_generated_tokens": 1,
                "planned_scratch_actual_total_mel_tokens": 9,
                "planned_scratch_code_slack": 2,
                "planned_scratch_generated_token_slack": 4,
                "planned_scratch_total_mel_token_slack": 4,
                "planned_scratch_covers_actual": True,
            },
            {
                "case": "shared_short_greedy",
                "status": "passed",
                "runtime_peak_bytes": 1500,
                "runtime_buffer_allocations": 120,
                "runtime_gpu_elapsed_seconds": 1.1,
                "native_runtime_total_seconds": 2.7,
                "planned_scratch_ok": True,
                "planned_scratch_segments": 2,
                "planned_scratch_max_prefix_tokens": 40,
                "planned_scratch_capacity_bytes": 10000,
                "planned_scratch_unshared_phase_peak_total_bytes": 12000,
                "planned_scratch_reuse_saves_bytes": 2000,
                "planned_scratch_capacity_to_unshared_ratio": 10000 / 12000,
                "planned_scratch_reuse_saves_ratio": 2000 / 12000,
                "planned_scratch_actual_codes": 1,
                "planned_scratch_actual_generated_tokens": 1,
                "planned_scratch_actual_total_mel_tokens": 9,
                "planned_scratch_code_slack": 2,
                "planned_scratch_generated_token_slack": 4,
                "planned_scratch_total_mel_token_slack": 4,
                "planned_scratch_covers_actual": True,
            },
        ],
    }

    comparisons = compare_summary_resource_cases(summary)
    assert comparisons[0]["base_case"] == "short_greedy"
    assert comparisons[0]["variant_case"] == "shared_short_greedy"
    metrics = {item["field"]: item for item in comparisons[0]["metrics"]}
    assert metrics["runtime_peak_bytes"]["ratio"] == 1.5
    assert metrics["runtime_buffer_allocations"]["delta"] == 20
    assert abs(metrics["runtime_gpu_elapsed_seconds"]["relative_delta"] - 0.1) < 1e-12
    assert abs(metrics["native_runtime_total_seconds"]["delta"] + 0.3) < 1e-12
    assert metrics["planned_scratch_capacity_bytes"]["delta"] == 0
    assert metrics["planned_scratch_segments"]["delta"] == 0
    assert metrics["planned_scratch_reuse_saves_ratio"]["ratio"] == 1.0
    assert metrics["planned_scratch_actual_codes"]["delta"] == 0
    invariants = {item["field"]: item for item in comparisons[0]["invariants"]}
    assert invariants["planned_scratch_ok"]["status"] == "passed"
    assert invariants["planned_scratch_segments"]["status"] == "passed"
    assert invariants["planned_scratch_max_prefix_tokens"]["status"] == "passed"
    assert invariants["planned_scratch_capacity_bytes"]["status"] == "passed"
    assert invariants["planned_scratch_actual_total_mel_tokens"]["status"] == "passed"
    assert invariants["planned_scratch_covers_actual"]["status"] == "passed"
    assert comparisons[0]["status"] == "measured"


def test_compare_summary_resource_cases_fails_planned_scratch_mismatch():
    summary = {
        "format": "mit2-native-hot-benchmark-summary",
        "version": 1,
        "cases": [
            {
                "case": "short_greedy",
                "status": "passed",
                "planned_scratch_capacity_bytes": 10000,
                "planned_scratch_reuse_saves_ratio": 0.1,
                "planned_scratch_covers_actual": True,
            },
            {
                "case": "shared_short_greedy",
                "status": "passed",
                "planned_scratch_capacity_bytes": 12000,
                "planned_scratch_reuse_saves_ratio": 0.2,
                "planned_scratch_covers_actual": False,
            },
        ],
    }

    comparisons = compare_summary_resource_cases(summary)
    assert comparisons[0]["status"] == "failed"
    invariants = {item["field"]: item for item in comparisons[0]["invariants"]}
    assert invariants["planned_scratch_capacity_bytes"]["status"] == "failed"
    assert invariants["planned_scratch_reuse_saves_ratio"]["status"] == "failed"
    assert invariants["planned_scratch_covers_actual"]["status"] == "failed"


def test_compare_benchmark_summaries_skips_new_segment_codes_against_old_baseline():
    current = _summary()
    current["cases"][0]["segment_predicted_codes"] = [[1], [2]]
    baseline = _summary()
    comparison = compare_benchmark_summaries(current, baseline)
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["segment_predicted_codes"]["status"] == "skipped"
    assert comparison["failed"] == 0


def test_compare_benchmark_summaries_fails_text_ids_source_drift():
    comparison = compare_benchmark_summaries(
        _summary(text_ids_source="tokenizer"),
        _summary(text_ids_source="native_cjk_segments"),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["text_ids_source"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_generation_budget_drift():
    comparison = compare_benchmark_summaries(
        _summary(generation_steps=1, generation_max_mel_tokens=1),
        _summary(generation_steps=25, generation_max_mel_tokens=3),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["generation_steps"]["status"] == "failed"
    assert checks["generation_max_mel_tokens"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_planned_scratch_drift():
    comparison = compare_benchmark_summaries(
        _summary(
            extra_case_fields={
                "planned_scratch_segments": 2,
                "planned_scratch_capacity_bytes": 1000,
                "planned_scratch_capacity_to_unshared_ratio": 0.5,
            }
        ),
        _summary(
            extra_case_fields={
                "planned_scratch_segments": 1,
                "planned_scratch_capacity_bytes": 2000,
                "planned_scratch_capacity_to_unshared_ratio": 0.75,
            }
        ),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["planned_scratch_segments"]["status"] == "failed"
    assert checks["planned_scratch_capacity_bytes"]["status"] == "failed"
    assert checks["planned_scratch_capacity_to_unshared_ratio"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_gpt_control_drift():
    current_fields = {
        "gpt_emovec_source": "native_emotion_vector_overlay",
        "gpt_emotion_source": "speaker",
        "gpt_emotion_alpha": 0.1875,
        "gpt_emotion_vector": [0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gpt_emotion_vector_indices": [2, 11, 1, 5, 3, 3, 0, 2],
        "gpt_emotion_vector_random": True,
        "gpt_emotion_vector_random_seed": 123,
        "gpt_frontend_tail_source": "native",
        "generation_shared_runtime_stages": False,
        "generation_emotion_vector_random": True,
        "generation_emotion_vector_random_seed": 123,
        "gpt_conds_latent_sha256": "conds-current",
        "gpt_text_ids_tensor_sha256": "text-current",
        "condition_f32_sha256": "condition-current",
        "noise_f32_bytes": 2880,
    }
    baseline_fields = {
        "gpt_emovec_source": "native_full_metal_subsampling_conformer_perceiver_linear",
        "gpt_emotion_source": "speaker",
        "gpt_emotion_alpha": 1.0,
        "gpt_emotion_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gpt_emotion_vector_indices": [2, 11, 1, 5, 3, 3, 0, 1],
        "gpt_emotion_vector_random": True,
        "gpt_emotion_vector_random_seed": 124,
        "gpt_frontend_tail_source": "native",
        "generation_shared_runtime_stages": True,
        "generation_emotion_vector_random": True,
        "generation_emotion_vector_random_seed": 124,
        "gpt_conds_latent_sha256": "conds-baseline",
        "gpt_text_ids_tensor_sha256": "text-current",
        "condition_f32_sha256": "condition-baseline",
        "noise_f32_bytes": 2880,
    }
    comparison = compare_benchmark_summaries(
        _summary(extra_case_fields=current_fields),
        _summary(extra_case_fields=baseline_fields),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["gpt_emovec_source"]["status"] == "failed"
    assert checks["gpt_emotion_alpha"]["status"] == "failed"
    assert checks["gpt_emotion_vector"]["status"] == "failed"
    assert checks["gpt_emotion_vector_random_seed"]["status"] == "failed"
    assert checks["gpt_frontend_tail_source"]["status"] == "passed"
    assert checks["generation_shared_runtime_stages"]["status"] == "failed"
    assert checks["generation_emotion_vector_random_seed"]["status"] == "failed"
    assert checks["gpt_conds_latent_sha256"]["status"] == "failed"
    assert checks["gpt_text_ids_tensor_sha256"]["status"] == "passed"
    assert checks["condition_f32_sha256"]["status"] == "failed"
    assert checks["noise_f32_bytes"]["status"] == "passed"


def test_compare_benchmark_summaries_fails_segment_sidecar_drift():
    current_fields = {
        "segment_codes_u32_bytes": [4, 4],
        "segment_codes_u32_sha256": ["codes-0", "codes-1"],
        "segment_condition_f32_bytes": [18432, 18432],
        "segment_condition_f32_sha256": ["condition-0", "condition-current"],
        "segment_noise_f32_bytes": [2880, 2880],
        "segment_noise_f32_sha256": ["noise-0", "noise-1"],
    }
    baseline_fields = {
        "segment_codes_u32_bytes": [4, 4],
        "segment_codes_u32_sha256": ["codes-0", "codes-1"],
        "segment_condition_f32_bytes": [18432, 18432],
        "segment_condition_f32_sha256": ["condition-0", "condition-baseline"],
        "segment_noise_f32_bytes": [2880, 2880],
        "segment_noise_f32_sha256": ["noise-0", "noise-1"],
    }
    comparison = compare_benchmark_summaries(
        _summary(extra_case_fields=current_fields),
        _summary(extra_case_fields=baseline_fields),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["segment_codes_u32_sha256"]["status"] == "passed"
    assert checks["segment_condition_f32_sha256"]["status"] == "failed"
    assert checks["segment_noise_f32_sha256"]["status"] == "passed"


def test_compare_benchmark_summaries_fails_segment_gpt_frontend_drift():
    current_fields = {
        "segment_gpt_frontend_format": ["mit2-gpt-frontend", "mit2-gpt-frontend"],
        "segment_gpt_conds_latent_sha256": ["conds-0", "conds-current"],
        "segment_gpt_text_ids_tensor_sha256": ["text-0", "text-1"],
        "segment_gpt_attention_mask_sha256": ["mask-0", "mask-1"],
    }
    baseline_fields = {
        "segment_gpt_frontend_format": ["mit2-gpt-frontend", "mit2-gpt-frontend"],
        "segment_gpt_conds_latent_sha256": ["conds-0", "conds-baseline"],
        "segment_gpt_text_ids_tensor_sha256": ["text-0", "text-1"],
        "segment_gpt_attention_mask_sha256": ["mask-0", "mask-1"],
    }
    comparison = compare_benchmark_summaries(
        _summary(extra_case_fields=current_fields),
        _summary(extra_case_fields=baseline_fields),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["segment_gpt_frontend_format"]["status"] == "passed"
    assert checks["segment_gpt_conds_latent_sha256"]["status"] == "failed"
    assert checks["segment_gpt_text_ids_tensor_sha256"]["status"] == "passed"


def test_compare_benchmark_summaries_fails_runtime_stage_count_drift():
    current_fields = {
        "runtime_stage_counts": {
            "gpt_kv_codes_inputs_export": 1,
            "hot_tts_condition_inputs_export": 1,
            "hot_tts_condition_inputs_wav": 1,
            "hot_tts_inputs_seeded_wav": 1,
        }
    }
    baseline_fields = {
        "runtime_stage_counts": {
            "gpt_kv_codes_inputs_export": 1,
            "hot_tts_condition_inputs_export": 1,
            "hot_tts_condition_inputs_wav": 1,
        }
    }
    comparison = compare_benchmark_summaries(
        _summary(extra_case_fields=current_fields),
        _summary(extra_case_fields=baseline_fields),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["runtime_stage_counts"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_voice_gate_drift():
    comparison = compare_benchmark_summaries(
        _summary(voice_validation_ok=True, voice_contract_ok=False),
        _summary(voice_validation_ok=True, voice_contract_ok=True),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["voice_validation_ok"]["status"] == "passed"
    assert checks["voice_contract_ok"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_model_gate_drift():
    comparison = compare_benchmark_summaries(
        _summary(model_contract_ok=False),
        _summary(model_contract_ok=True),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["model_contract_ok"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_model_integrity_drift():
    current_fields = {
        "model_contract_ok": True,
        "model_contract_weights_bytes": 5317937152,
        "model_contract_tensor_count": 2755,
        "model_contract_required_tensor_count": 24,
        "model_contract_integrity_sha256_verified_count": 2754,
    }
    baseline_fields = {
        "model_contract_ok": True,
        "model_contract_weights_bytes": 5317937152,
        "model_contract_tensor_count": 2755,
        "model_contract_required_tensor_count": 24,
        "model_contract_integrity_sha256_verified_count": 2755,
    }
    comparison = compare_benchmark_summaries(
        _summary(extra_case_fields=current_fields),
        _summary(extra_case_fields=baseline_fields),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["model_contract_ok"]["status"] == "passed"
    assert checks["model_contract_integrity_sha256_verified_count"]["status"] == "failed"


def test_compare_benchmark_summaries_fails_voice_integrity_drift():
    current_fields = {
        "voice_contract_ok": True,
        "voice_contract_weights_bytes": 4488192,
        "voice_contract_tensor_count": 4,
        "voice_contract_prompt_tokens": 943,
        "voice_contract_integrity_sha256_verified_count": 4,
    }
    baseline_fields = {
        "voice_contract_ok": True,
        "voice_contract_weights_bytes": 4488192,
        "voice_contract_tensor_count": 4,
        "voice_contract_prompt_tokens": 944,
        "voice_contract_integrity_sha256_verified_count": 4,
    }
    comparison = compare_benchmark_summaries(
        _summary(extra_case_fields=current_fields),
        _summary(extra_case_fields=baseline_fields),
    )
    assert comparison["failed"] == 1
    checks = {item["field"]: item for item in comparison["cases"][0]["checks"]}
    assert checks["voice_contract_ok"]["status"] == "passed"
    assert checks["voice_contract_prompt_tokens"]["status"] == "failed"


def test_summarize_report_extracts_voice_gate_metrics(tmp_path):
    wav_path = tmp_path / "out.wav"
    with wave.open(str(wav_path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(22050)
        fp.writeframes(b"\x00\x00" * 4)
    codes_path = tmp_path / "out.wav.codes.u32"
    condition_path = tmp_path / "out.wav.condition.f32"
    noise_path = tmp_path / "out.wav.noise.f32"
    codes_path.write_bytes(b"codes")
    condition_path.write_bytes(b"condition")
    noise_path.write_bytes(b"noise")
    frontend_dir = tmp_path / "gpt_frontend"
    frontend_dir.mkdir()
    (frontend_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mit2-gpt-frontend",
                "tensors": {
                    "conds_latent": {
                        "dtype": "float32",
                        "shape": [34, 1280],
                        "sha256": "conds-sha",
                    },
                    "speech_conditioning_latent": {
                        "dtype": "float32",
                        "shape": [32, 1280],
                        "sha256": "speech-sha",
                    },
                    "text_ids": {
                        "dtype": "uint32",
                        "shape": [4],
                        "sha256": "text-sha",
                    },
                    "fake_inputs": {
                        "dtype": "uint32",
                        "shape": [41],
                        "sha256": "fake-sha",
                    },
                    "inputs_embeds": {
                        "dtype": "float32",
                        "shape": [40, 1280],
                        "sha256": "embeds-sha",
                    },
                    "attention_mask": {
                        "dtype": "uint32",
                        "shape": [41],
                        "sha256": "mask-sha",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-synthesis-report",
                "output_wav": str(wav_path),
                "output_wav_sha256": "abc",
                "codes_u32": str(codes_path),
                "condition_f32": str(condition_path),
                "noise_f32": str(noise_path),
                "elapsed_seconds": 1.0,
                "gpt_frontend_dir": str(frontend_dir),
                "generation": {
                    "max_mel_tokens": 1,
                    "prompt_tokens": 8,
                    "steps": 1,
                    "emotion_vector_random": True,
                    "emotion_vector_random_seed": 123,
                },
                "gpt": {
                    "frontend_format": "mit2-gpt-frontend",
                    "emovec_source": "native_emotion_vector_overlay",
                    "emotion_source": "speaker",
                    "emotion_alpha": 0.1875,
                    "emotion_vector": [0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "emotion_vector_indices": [2, 11, 1, 5, 3, 3, 0, 2],
                    "emotion_vector_random": True,
                    "emotion_vector_random_seed": 123,
                    "subsampling_source": "native_metal_resident_conv_linear",
                    "conformer_source": "native_stack_metal_resident_attn_core_conv_ff",
                    "perceiver_source": "native_metal_resident_linear_cross_attn_geglu_rmsnorm",
                    "frontend_tail_source": "native",
                    "emovec_mix": [{"path": "a.f32", "weight": 1.0}],
                },
                "model_contract": {
                    "ok": True,
                    "runtime_report": {
                        "weights_bytes": 5320155392,
                        "tensor_count": 2755,
                        "tensor_bytes": 5312776880,
                        "required_tensor_count": 24,
                        "integrity": {
                            "aligned_tensor_count": 2755,
                            "checked_interval_count": 2755,
                            "sha256_verified_count": 2755,
                        },
                    },
                },
                "voice_validation": {"ok": True},
                "voice_contract": {
                    "ok": True,
                    "runtime_report": {
                        "weights_bytes": 4485120,
                        "spk_cond_tokens": 548,
                        "prompt_tokens": 944,
                        "mel_frames": 944,
                        "tensor_count": 4,
                        "tensor_bytes": 4480768,
                        "integrity": {
                            "aligned_tensor_count": 4,
                            "checked_interval_count": 4,
                            "sha256_verified_count": 4,
                        },
                    },
                },
                "runtime_json": [],
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report("short_greedy", report_path)

    assert summary["model_contract_ok"] is True
    assert summary["gpt_frontend_format"] == "mit2-gpt-frontend"
    assert summary["gpt_emovec_source"] == "native_emotion_vector_overlay"
    assert summary["gpt_emotion_alpha"] == 0.1875
    assert summary["gpt_emotion_vector"] == [0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert summary["gpt_emotion_vector_indices"] == [2, 11, 1, 5, 3, 3, 0, 2]
    assert summary["gpt_emotion_vector_random"] is True
    assert summary["gpt_emotion_vector_random_seed"] == 123
    assert summary["generation_emotion_vector_random"] is True
    assert summary["generation_emotion_vector_random_seed"] == 123
    assert summary["gpt_subsampling_source"] == "native_metal_resident_conv_linear"
    assert summary["gpt_frontend_tail_source"] == "native"
    assert summary["gpt_emovec_mix"] == [{"path": "a.f32", "weight": 1.0}]
    assert summary["gpt_conds_latent_dtype"] == "float32"
    assert summary["gpt_conds_latent_shape"] == [34, 1280]
    assert summary["gpt_conds_latent_sha256"] == "conds-sha"
    assert summary["gpt_speech_conditioning_latent_shape"] == [32, 1280]
    assert summary["gpt_speech_conditioning_latent_sha256"] == "speech-sha"
    assert summary["gpt_text_ids_tensor_sha256"] == "text-sha"
    assert summary["gpt_fake_inputs_sha256"] == "fake-sha"
    assert summary["gpt_inputs_embeds_sha256"] == "embeds-sha"
    assert summary["gpt_attention_mask_sha256"] == "mask-sha"
    assert summary["codes_u32_bytes"] == 5
    assert summary["codes_u32_sha256"] == hashlib.sha256(b"codes").hexdigest()
    assert summary["condition_f32_bytes"] == 9
    assert summary["condition_f32_sha256"] == hashlib.sha256(b"condition").hexdigest()
    assert summary["noise_f32_bytes"] == 5
    assert summary["noise_f32_sha256"] == hashlib.sha256(b"noise").hexdigest()
    assert summary["model_contract_weights_bytes"] == 5320155392
    assert summary["model_contract_tensor_count"] == 2755
    assert summary["model_contract_required_tensor_count"] == 24
    assert summary["model_contract_integrity_sha256_verified_count"] == 2755
    assert summary["voice_validation_ok"] is True
    assert summary["voice_contract_ok"] is True
    assert summary["voice_contract_weights_bytes"] == 4485120
    assert summary["voice_contract_spk_cond_tokens"] == 548
    assert summary["voice_contract_prompt_tokens"] == 944
    assert summary["voice_contract_mel_frames"] == 944
    assert summary["voice_contract_integrity_checked_interval_count"] == 4


def test_summarize_report_backfills_relative_frontend_manifest(tmp_path):
    wav_path = tmp_path / "out.wav"
    with wave.open(str(wav_path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(22050)
        fp.writeframes(b"\x00\x00" * 4)
    frontend_dir = tmp_path / "gpt_frontend"
    frontend_dir.mkdir()
    (frontend_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mit2-gpt-frontend",
                "tensors": {
                    "conds_latent": {
                        "dtype": "float32",
                        "shape": [34, 1280],
                        "sha256": "relative-conds-sha",
                    },
                    "text_ids": {
                        "dtype": "uint32",
                        "shape": [4],
                        "sha256": "relative-text-sha",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-synthesis-report",
                "output_wav": str(wav_path),
                "output_wav_sha256": "abc",
                "gpt_frontend_dir": "gpt_frontend",
                "gpt": {"frontend_format": "mit2-gpt-frontend"},
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report("short_greedy", report_path)

    assert summary["gpt_conds_latent_sha256"] == "relative-conds-sha"
    assert summary["gpt_text_ids_tensor_sha256"] == "relative-text-sha"


def test_summarize_report_extracts_segment_sidecar_metrics(tmp_path):
    wav_path = tmp_path / "out.wav"
    with wave.open(str(wav_path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(22050)
        fp.writeframes(b"\x00\x00" * 4)
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    sidecars = {
        "segments/segment_000.wav.codes.u32": b"codes0",
        "segments/segment_000.wav.condition.f32": b"condition0",
        "segments/segment_000.wav.noise.f32": b"noise0",
        "segments/segment_001.wav.codes.u32": b"codes1",
        "segments/segment_001.wav.condition.f32": b"condition1",
        "segments/segment_001.wav.noise.f32": b"noise1",
    }
    for rel_path, payload in sidecars.items():
        (tmp_path / rel_path).write_bytes(payload)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-multisegment-synthesis-report",
                "output_wav": str(wav_path),
                "output_wav_sha256": "abc",
                "generation": {"all_segments": True, "segments": 2},
                "segments": [
                    {
                        "index": 0,
                        "report": {
                            "codes_u32": "segments/segment_000.wav.codes.u32",
                            "condition_f32": "segments/segment_000.wav.condition.f32",
                            "noise_f32": "segments/segment_000.wav.noise.f32",
                            "gpt": {
                                "frontend_format": "mit2-gpt-frontend",
                                "conds_latent": {
                                    "dtype": "float32",
                                    "shape": [34, 1280],
                                    "sha256": "conds-0",
                                },
                                "text_ids_tensor": {
                                    "dtype": "uint32",
                                    "shape": [4],
                                    "sha256": "text-0",
                                },
                                "attention_mask": {
                                    "dtype": "uint32",
                                    "shape": [41],
                                    "sha256": "mask-0",
                                },
                            },
                        },
                    },
                    {
                        "index": 1,
                        "report": {
                            "codes_u32": "segments/segment_001.wav.codes.u32",
                            "condition_f32": "segments/segment_001.wav.condition.f32",
                            "noise_f32": "segments/segment_001.wav.noise.f32",
                            "gpt": {
                                "frontend_format": "mit2-gpt-frontend",
                                "conds_latent": {
                                    "dtype": "float32",
                                    "shape": [34, 1280],
                                    "sha256": "conds-1",
                                },
                                "text_ids_tensor": {
                                    "dtype": "uint32",
                                    "shape": [4],
                                    "sha256": "text-1",
                                },
                                "attention_mask": {
                                    "dtype": "uint32",
                                    "shape": [41],
                                    "sha256": "mask-1",
                                },
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report("long_all_segments", report_path)

    assert summary["segment_codes_u32_bytes"] == [6, 6]
    assert summary["segment_codes_u32_sha256"] == [
        hashlib.sha256(b"codes0").hexdigest(),
        hashlib.sha256(b"codes1").hexdigest(),
    ]
    assert summary["segment_condition_f32_bytes"] == [10, 10]
    assert summary["segment_condition_f32_sha256"] == [
        hashlib.sha256(b"condition0").hexdigest(),
        hashlib.sha256(b"condition1").hexdigest(),
    ]
    assert summary["segment_noise_f32_bytes"] == [6, 6]
    assert summary["segment_noise_f32_sha256"] == [
        hashlib.sha256(b"noise0").hexdigest(),
        hashlib.sha256(b"noise1").hexdigest(),
    ]
    assert summary["segment_gpt_frontend_format"] == ["mit2-gpt-frontend", "mit2-gpt-frontend"]
    assert summary["segment_gpt_conds_latent_dtype"] == ["float32", "float32"]
    assert summary["segment_gpt_conds_latent_shape"] == [[34, 1280], [34, 1280]]
    assert summary["segment_gpt_conds_latent_sha256"] == ["conds-0", "conds-1"]
    assert summary["segment_gpt_text_ids_tensor_sha256"] == ["text-0", "text-1"]
    assert summary["segment_gpt_attention_mask_sha256"] == ["mask-0", "mask-1"]


def test_summarize_report_uses_runtime_summary_without_runtime_json(tmp_path):
    wav_path = tmp_path / "out.wav"
    with wave.open(str(wav_path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(22050)
        fp.writeframes(b"\x00\x00" * 8)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "mit2-native-hot-synthesis-report",
                "output_wav": str(wav_path),
                "output_wav_sha256": "abc",
                "elapsed_seconds": 4.0,
                "runtime_summary": {
                    "stage_counts": {"hot_tts_inputs_seeded_wav": 1},
                    "stage_seconds": {"hot_tts_inputs_seeded_wav": 2.0},
                    "native_total_seconds": 2.0,
                    "native_phase_seconds": {"gpt_seconds": 1.0, "acoustic_seconds": 0.5},
                    "resident_peak_bytes": 1234,
                    "command_buffers_submitted": 5,
                    "buffer_allocations": 14,
                    "buffer_bytes_allocated": 1200,
                    "gpu_elapsed_seconds": 0.3,
                    "predicted_codes": [4039],
                    "segment_predicted_codes": [[4039], [6947, 2248]],
                    "planned_scratch": {
                        "planned_scratch_segments": 2,
                        "planned_scratch_unshared_phase_peak_total_bytes": 14000,
                        "planned_scratch_capacity_bytes": 6500,
                        "planned_scratch_reuse_saves_bytes": 2500,
                        "planned_scratch_actual_codes": 3,
                        "planned_scratch_actual_generated_tokens": 12,
                        "planned_scratch_covers_actual": True,
                        "planned_scratch_ok": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report("short_greedy", report_path)

    assert summary["runtime_peak_bytes"] == 1234
    assert summary["predicted_codes"] == [4039]
    assert summary["runtime_stage_counts"] == {"hot_tts_inputs_seeded_wav": 1}
    assert summary["native_runtime_total_seconds"] == 2.0
    assert summary["native_gpt_seconds"] == 1.0
    assert summary["runtime_command_buffers_submitted"] == 5
    assert summary["runtime_buffer_allocations"] == 14
    assert summary["runtime_buffer_bytes_allocated"] == 1200
    assert summary["runtime_gpu_elapsed_seconds"] == 0.3
    assert summary["gpt_codes"] == 1
    assert summary["segment_predicted_codes"] == [[4039], [6947, 2248]]
    assert summary["planned_scratch_segments"] == 2
    assert summary["planned_scratch_capacity_bytes"] == 6500
    assert summary["planned_scratch_actual_codes"] == 3
    assert summary["planned_scratch_actual_generated_tokens"] == 12
    assert summary["planned_scratch_capacity_to_unshared_ratio"] == 6500 / 14000
    assert summary["planned_scratch_reuse_saves_ratio"] == 2500 / 14000
    assert summary["planned_scratch_covers_actual"] is True


def test_medium_case_does_not_reuse_short_text_ids_file():
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/repo/checkpoints",
        python_executable="/venv/bin/python",
        model_bundle="artifacts/model",
        runtime="./runtime",
        max_mel_tokens=3,
        prompt_tokens=8,
        steps=2,
        cfg_rate=0.7,
        temperature=1.0,
        seed=1,
        voice_bundle="artifacts/voice",
        voice_name="voice",
        text_ids_file="artifacts/native_text_ids_cjk.u32",
        native_flags=False,
        short_text="短",
        medium_text="中等长度文本",
        multi_text="多段",
        long_text="长文本",
        segment_tokens=4,
        long_segment_tokens=80,
        interval_silence_ms=200,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        clone_audio=None,
        clone_name="clone",
        shared_runtime_stages=False,
    )
    command = _case_args(args, "medium_greedy", Path("artifacts/bench/medium_greedy"))
    assert "--text-ids-file" not in command
    assert command[-2:] == ["--text", "中等长度文本"]


def test_native_cjk_segments_case_uses_native_segmented_text_ids():
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/repo/checkpoints",
        python_executable="/venv/bin/python",
        model_bundle="artifacts/model",
        runtime="./runtime",
        max_mel_tokens=1,
        prompt_tokens=8,
        steps=1,
        cfg_rate=0.7,
        temperature=1.0,
        seed=1,
        voice_bundle="artifacts/voice",
        voice_name="voice",
        text_ids_file="artifacts/native_text_ids_cjk.u32",
        native_flags=False,
        short_text="短",
        medium_text="中等长度文本",
        multi_text="你好你好",
        long_text="长文本",
        segment_tokens=4,
        long_segment_tokens=80,
        interval_silence_ms=200,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        clone_audio=None,
        clone_name="clone",
        shared_runtime_stages=False,
    )
    command = _case_args(args, "native_cjk_segments", Path("artifacts/bench/native_cjk_segments"))

    assert "--text-ids-file" not in command
    assert "--all-segments" in command
    assert "--native-cjk-text-ids" in command
    assert command[command.index("--max-text-tokens-per-segment") + 1] == "4"


def test_native_cjk_mixed_case_uses_single_segment_native_text_ids():
    args = _case_namespace(
        text_ids_file="artifacts/native_text_ids_cjk.u32",
        native_cjk_mixed_text="AB1C2你好",
        native_flags=False,
    )
    command = _case_args(args, "native_cjk_mixed", Path("artifacts/bench/native_cjk_mixed"))

    assert "--text-ids-file" not in command
    assert "--native-cjk-text-ids" in command
    assert "--all-segments" not in command
    assert command[-1] == "--native-cjk-text-ids"
    assert command[command.index("--text") + 1] == "AB1C2你好"


def test_case_args_uses_configured_python_executable():
    args = _case_namespace(python_executable="/custom/python")
    command = _case_args(args, "short_greedy", Path("artifacts/bench/short_greedy"))

    assert command[0] == "/custom/python"


def test_compact_reports_flag_forwards_to_synthesis():
    compact = _case_args(
        _case_namespace(compact_reports=True),
        "short_greedy",
        Path("artifacts/bench/short_greedy"),
    )
    default = _case_args(
        _case_namespace(compact_reports=False),
        "short_greedy",
        Path("artifacts/bench/short_greedy"),
    )

    assert "--compact-report" in compact
    assert "--compact-report" not in default


def test_shared_runtime_stages_flag_applies_to_greedy_not_sampled():
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/repo/checkpoints",
        python_executable="/venv/bin/python",
        model_bundle="artifacts/model",
        runtime="./runtime",
        max_mel_tokens=3,
        prompt_tokens=8,
        steps=2,
        cfg_rate=0.7,
        temperature=1.0,
        seed=1,
        voice_bundle="artifacts/voice",
        voice_name="voice",
        text_ids_file=None,
        native_flags=False,
        short_text="短",
        medium_text="中等长度文本",
        multi_text="多段",
        long_text="长文本",
        segment_tokens=4,
        long_segment_tokens=80,
        interval_silence_ms=200,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        clone_audio=None,
        clone_name="clone",
        shared_runtime_stages=True,
    )
    greedy = _case_args(args, "short_greedy", Path("artifacts/bench/short_greedy"))
    sampled = _case_args(args, "short_sampled", Path("artifacts/bench/short_sampled"))

    assert "--shared-runtime-stages" in greedy
    assert "--shared-runtime-stages" not in sampled


def test_shared_short_greedy_case_is_independent_opt_in():
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/repo/checkpoints",
        python_executable="/venv/bin/python",
        model_bundle="artifacts/model",
        runtime="./runtime",
        max_mel_tokens=3,
        prompt_tokens=8,
        steps=2,
        cfg_rate=0.7,
        temperature=1.0,
        seed=1,
        voice_bundle="artifacts/voice",
        voice_name="voice",
        text_ids_file="artifacts/native_text_ids_cjk.u32",
        native_flags=False,
        short_text="短",
        medium_text="中等长度文本",
        multi_text="多段",
        long_text="长文本",
        segment_tokens=4,
        long_segment_tokens=80,
        interval_silence_ms=200,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        clone_audio=None,
        clone_name="clone",
        shared_runtime_stages=False,
    )
    command = _case_args(args, "shared_short_greedy", Path("artifacts/bench/shared_short_greedy"))

    assert "--shared-runtime-stages" in command
    assert command[command.index("--text-ids-file") + 1] == "artifacts/native_text_ids_cjk.u32"
    assert command[-2:] == ["--text", "短"]


def test_emotion_profile_case_requires_profile_and_passes_alpha():
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/repo/checkpoints",
        python_executable="/venv/bin/python",
        model_bundle="artifacts/model",
        runtime="./runtime",
        max_mel_tokens=3,
        prompt_tokens=8,
        steps=2,
        cfg_rate=0.7,
        temperature=1.0,
        seed=1,
        voice_bundle="artifacts/voice",
        voice_name="voice",
        text_ids_file="artifacts/native_text_ids_cjk.u32",
        native_flags=True,
        short_text="短",
        medium_text="中等长度文本",
        multi_text="多段",
        long_text="长文本",
        segment_tokens=4,
        long_segment_tokens=80,
        interval_silence_ms=200,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        clone_audio=None,
        clone_name="clone",
        shared_runtime_stages=False,
        emotion_voice_name=None,
        emotion_alpha=0.4,
    )

    try:
        _case_args(args, "emotion_profile", Path("artifacts/bench/emotion_profile"))
    except ValueError as exc:
        assert "requires --emotion-voice-name" in str(exc)
    else:
        raise AssertionError("emotion_profile should require --emotion-voice-name")

    args.emotion_voice_name = "emotion_ref"
    command = _case_args(args, "emotion_profile", Path("artifacts/bench/emotion_profile"))
    assert "--native-emovec" in command
    assert command[command.index("--text-ids-file") + 1] == "artifacts/native_text_ids_cjk.u32"
    assert command[command.index("--emotion-voice-name") + 1] == "emotion_ref"
    assert command[command.index("--emotion-alpha") + 1] == "0.4"


def test_emotion_audio_case_requires_audio_and_passes_reference():
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/repo/checkpoints",
        python_executable="/venv/bin/python",
        model_bundle="artifacts/model",
        runtime="./runtime",
        max_mel_tokens=3,
        prompt_tokens=8,
        steps=2,
        cfg_rate=0.7,
        temperature=1.0,
        seed=1,
        voice_bundle="artifacts/voice",
        voice_name="voice",
        text_ids_file=None,
        native_flags=False,
        short_text="短",
        medium_text="中等长度文本",
        multi_text="多段",
        long_text="长文本",
        segment_tokens=4,
        long_segment_tokens=80,
        interval_silence_ms=200,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        clone_audio=None,
        clone_name="clone",
        shared_runtime_stages=False,
        emotion_audio=None,
        emotion_alpha=0.6,
    )

    try:
        _case_args(args, "emotion_audio", Path("artifacts/bench/emotion_audio"))
    except ValueError as exc:
        assert "requires --emotion-audio" in str(exc)
    else:
        raise AssertionError("emotion_audio should require --emotion-audio")

    args.emotion_audio = "refs/emotion.wav"
    command = _case_args(args, "emotion_audio", Path("artifacts/bench/emotion_audio"))
    assert "--native-emovec" not in command
    assert command[command.index("--emotion-audio") + 1] == "refs/emotion.wav"
    assert command[command.index("--emotion-alpha") + 1] == "0.6"


def test_emotion_vector_case_passes_slider_options():
    args = argparse.Namespace(
        index_tts_repo="/repo",
        checkpoint_dir="/repo/checkpoints",
        python_executable="/venv/bin/python",
        model_bundle="artifacts/model",
        runtime="./runtime",
        max_mel_tokens=3,
        prompt_tokens=8,
        steps=2,
        cfg_rate=0.7,
        temperature=1.0,
        seed=1,
        voice_bundle="artifacts/voice",
        voice_name="voice",
        text_ids_file="artifacts/native_text_ids_cjk.u32",
        native_flags=True,
        short_text="短",
        medium_text="中等长度文本",
        multi_text="多段",
        long_text="长文本",
        segment_tokens=4,
        long_segment_tokens=80,
        interval_silence_ms=200,
        gpt_temperature=0.8,
        gpt_top_k=30,
        gpt_top_p=0.8,
        gpt_repetition_penalty=10.0,
        clone_audio=None,
        clone_name="clone",
        shared_runtime_stages=False,
        emotion_vector="0.2,0,0,0,0,0,0,0",
        emotion_alpha=0.75,
        emotion_vector_raw=True,
        emotion_vector_random=True,
        emotion_vector_random_seed=123,
    )
    command = _case_args(args, "emotion_vector", Path("artifacts/bench/emotion_vector"))

    assert command[command.index("--text-ids-file") + 1] == "artifacts/native_text_ids_cjk.u32"
    assert command[command.index("--emotion-vector") + 1] == "0.2,0,0,0,0,0,0,0"
    assert command[command.index("--emotion-alpha") + 1] == "0.75"
    assert "--emotion-vector-raw" in command
    assert "--emotion-vector-random" in command
    assert command[command.index("--emotion-vector-random-seed") + 1] == "123"


def test_short_full_acoustic_case_uses_full_acoustic_budget_and_short_ids():
    args = _case_namespace(max_mel_tokens=1, steps=1, full_acoustic_max_mel_tokens=8, full_acoustic_steps=25)
    command = _case_args(args, "short_acoustic_full", Path("artifacts/bench/short_acoustic_full"))

    assert command[command.index("--max-mel-tokens") + 1] == "8"
    assert command[command.index("--steps") + 1] == "25"
    assert command[command.index("--text-ids-file") + 1] == "artifacts/native_text_ids_cjk.u32"
    assert command[-2:] == ["--text", "短"]


def test_medium_full_acoustic_case_does_not_reuse_short_text_ids():
    args = _case_namespace(max_mel_tokens=1, steps=1, full_acoustic_max_mel_tokens=8, full_acoustic_steps=25)
    command = _case_args(args, "medium_acoustic_full", Path("artifacts/bench/medium_acoustic_full"))

    assert command[command.index("--max-mel-tokens") + 1] == "8"
    assert command[command.index("--steps") + 1] == "25"
    assert "--text-ids-file" not in command
    assert command[-2:] == ["--text", "中等长度文本"]


def test_long_full_acoustic_case_uses_segmented_long_text():
    args = _case_namespace(
        max_mel_tokens=1,
        steps=1,
        full_acoustic_max_mel_tokens=6,
        full_acoustic_steps=25,
        long_segment_tokens=18,
    )
    command = _case_args(args, "long_acoustic_full_segments", Path("artifacts/bench/long_acoustic_full_segments"))

    assert command[command.index("--max-mel-tokens") + 1] == "6"
    assert command[command.index("--steps") + 1] == "25"
    assert "--all-segments" in command
    assert command[command.index("--max-text-tokens-per-segment") + 1] == "18"
    assert "--text-ids-file" not in command


def test_clone_full_acoustic_case_requires_clone_audio_and_full_budget():
    args = _case_namespace(clone_audio=None, full_acoustic_max_mel_tokens=3, full_acoustic_steps=25)
    try:
        _case_args(args, "clone_acoustic_full", Path("artifacts/bench/clone_acoustic_full"))
    except ValueError as exc:
        assert "requires --clone-audio" in str(exc)
    else:
        raise AssertionError("clone_acoustic_full should require --clone-audio")

    args.clone_audio = "refs/voice.wav"
    command = _case_args(args, "clone_acoustic_full", Path("artifacts/bench/clone_acoustic_full"))
    assert command[command.index("--max-mel-tokens") + 1] == "3"
    assert command[command.index("--steps") + 1] == "25"
    assert command[command.index("--text-ids-file") + 1] == "artifacts/native_text_ids_cjk.u32"
    assert command[command.index("--clone-audio") + 1] == "refs/voice.wav"
    assert "--voice-bundle" not in command
    assert "--voice-name" not in command


def test_clone_case_omits_default_voice_bundle_for_fresh_clone():
    args = _case_namespace(clone_audio="refs/voice.wav", voice_bundle="artifacts/voice", voice_name="voice")
    command = _case_args(args, "clone", Path("artifacts/bench/clone"))

    assert command[command.index("--clone-audio") + 1] == "refs/voice.wav"
    assert command[command.index("--clone-name") + 1] == "clone"
    assert "--voice-bundle" not in command
    assert "--voice-name" not in command


def test_benchmark_forwards_validation_flags():
    args = _case_namespace(validate_voice_source=True, validate_model_contract=True, validate_voice_contract=True)
    cached = _case_args(args, "short_greedy", Path("artifacts/bench/short_greedy"))
    assert "--validate-voice-source" in cached
    assert "--validate-model-contract" in cached
    assert "--validate-voice-contract" in cached
    assert cached[cached.index("--voice-bundle") + 1] == "artifacts/voice"

    args.clone_audio = "refs/voice.wav"
    clone = _case_args(args, "clone", Path("artifacts/bench/clone"))
    assert "--validate-voice-source" in clone
    assert "--validate-model-contract" in clone
    assert "--validate-voice-contract" in clone
    assert "--voice-bundle" not in clone
