from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Iterable

from metal_indextts2.metrics import file_sha256
from metal_indextts2.tools.common import add_reference_args, write_json


_SHORT_TEXT_ID_CASES = {
    "short_greedy",
    "short_sampled",
    "shared_short_greedy",
    "short_acoustic_full",
    "clone_acoustic_full",
    "emotion_profile",
    "emotion_audio",
    "emotion_vector",
}

_FULL_ACOUSTIC_CASES = {
    "short_acoustic_full",
    "medium_acoustic_full",
    "long_acoustic_full_segments",
    "clone_acoustic_full",
}

_CLONE_CASES = {
    "clone",
    "clone_acoustic_full",
}

_BENCHMARK_CASES = {
    "short_greedy",
    "shared_short_greedy",
    "short_acoustic_full",
    "short_sampled",
    "medium_greedy",
    "medium_acoustic_full",
    "emotion_profile",
    "emotion_audio",
    "emotion_vector",
    "all_segments",
    "native_cjk_segments",
    "native_cjk_mixed",
    "long_all_segments",
    "long_acoustic_full_segments",
    "clone",
    "clone_acoustic_full",
}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _positive_int_arg(args: argparse.Namespace, name: str, flag: str) -> None:
    value = getattr(args, name, None)
    if value is not None and value <= 0:
        raise ValueError(f"{flag} must be positive")


def _non_negative_int_arg(args: argparse.Namespace, name: str, flag: str) -> None:
    value = getattr(args, name, None)
    if value is not None and value < 0:
        raise ValueError(f"{flag} must be non-negative")


def _non_negative_float_arg(args: argparse.Namespace, name: str, flag: str) -> None:
    value = getattr(args, name, None)
    if value is not None and (not _is_finite_number(value) or float(value) < 0.0):
        raise ValueError(f"{flag} must be finite and non-negative")


def _read_json_object_for_validation(path: Path, flag: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag} must be valid JSON: {path}") from exc
    except OSError as exc:
        raise ValueError(f"{flag} cannot be read: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{flag} must contain a JSON object: {path}")
    return value


def _resolve_report_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def _validate_benchmark_args(args: argparse.Namespace) -> None:
    from_reports = getattr(args, "from_reports", None)
    if from_reports is not None:
        if not from_reports:
            raise ValueError("--from-reports requires at least one CASE=REPORT_JSON entry")
        seen_cases: set[str] = set()
        for item in from_reports:
            if "=" not in item:
                raise ValueError("--from-reports entries must be CASE=REPORT_JSON")
            case, report = item.split("=", 1)
            if not case:
                raise ValueError("--from-reports case name must be non-empty")
            if case in seen_cases:
                raise ValueError(f"duplicate benchmark case: {case}")
            seen_cases.add(case)
            if not report:
                raise ValueError("--from-reports report path must be non-empty")
            if not Path(report).is_file():
                raise ValueError(f"--from-reports report does not exist: {report}")
            report_path = Path(report)
            report_json = _read_json_object_for_validation(report_path, "--from-reports report")
            output_wav = report_json.get("output_wav")
            if not isinstance(output_wav, str) or not output_wav:
                raise ValueError(f"--from-reports report missing output_wav: {report}")
            output_wav_path = _resolve_report_path(output_wav, report_path.parent)
            if not report_json.get("output_wav_sha256") and not output_wav_path.is_file():
                raise ValueError(f"--from-reports report output_wav does not exist and output_wav_sha256 is missing: {output_wav}")
    else:
        cases = list(getattr(args, "cases", []) or [])
        if not cases:
            raise ValueError("--cases requires at least one benchmark case")
        seen_cases: set[str] = set()
        for case in cases:
            if case in seen_cases:
                raise ValueError(f"duplicate benchmark case: {case}")
            seen_cases.add(case)
        unknown = sorted(set(cases) - _BENCHMARK_CASES)
        if unknown:
            raise ValueError(f"unknown benchmark case: {unknown[0]}")
        if any(case in _CLONE_CASES for case in cases) and not getattr(args, "clone_audio", None):
            raise ValueError("clone cases require --clone-audio")
        if "emotion_profile" in cases and not getattr(args, "emotion_voice_name", None):
            raise ValueError("emotion_profile case requires --emotion-voice-name")
        if "emotion_audio" in cases and not getattr(args, "emotion_audio", None):
            raise ValueError("emotion_audio case requires --emotion-audio")
    baseline_summary = getattr(args, "baseline_summary", None)
    if baseline_summary and not Path(baseline_summary).is_file():
        raise ValueError(f"--baseline-summary does not exist: {baseline_summary}")
    if baseline_summary:
        baseline_json = _read_json_object_for_validation(Path(baseline_summary), "--baseline-summary")
        _summary_case_map(baseline_json)
    for name, flag in (
        ("max_mel_tokens", "--max-mel-tokens"),
        ("prompt_tokens", "--prompt-tokens"),
        ("steps", "--steps"),
        ("full_acoustic_max_mel_tokens", "--full-acoustic-max-mel-tokens"),
        ("full_acoustic_steps", "--full-acoustic-steps"),
        ("segment_tokens", "--segment-tokens"),
        ("long_segment_tokens", "--long-segment-tokens"),
    ):
        _positive_int_arg(args, name, flag)
    _non_negative_int_arg(args, "interval_silence_ms", "--interval-silence-ms")
    cfg_rate = getattr(args, "cfg_rate", 0.0)
    if not _is_finite_number(cfg_rate) or float(cfg_rate) < 0.0:
        raise ValueError("--cfg-rate must be finite and non-negative")
    temperature = getattr(args, "temperature", 1.0)
    if not _is_finite_number(temperature) or float(temperature) <= 0.0:
        raise ValueError("--temperature must be finite and positive")
    gpt_temperature = getattr(args, "gpt_temperature", 1.0)
    if not _is_finite_number(gpt_temperature) or float(gpt_temperature) <= 0.0:
        raise ValueError("--gpt-temperature must be finite and positive")
    _non_negative_int_arg(args, "gpt_top_k", "--gpt-top-k")
    gpt_top_p = getattr(args, "gpt_top_p", 1.0)
    if not _is_finite_number(gpt_top_p) or not (0.0 < float(gpt_top_p) <= 1.0):
        raise ValueError("--gpt-top-p must be finite and in (0, 1]")
    gpt_repetition_penalty = getattr(args, "gpt_repetition_penalty", 1.0)
    if (
        not _is_finite_number(gpt_repetition_penalty)
        or float(gpt_repetition_penalty) <= 0.0
    ):
        raise ValueError("--gpt-repetition-penalty must be finite and positive")
    for name, flag in (
        ("max_rtf_regression", "--max-rtf-regression"),
        ("max_elapsed_regression", "--max-elapsed-regression"),
        ("max_memory_regression", "--max-memory-regression"),
        ("max_throughput_regression", "--max-throughput-regression"),
        ("max_runtime_gpu_elapsed_seconds", "--max-runtime-gpu-elapsed-seconds"),
        ("max_rtf", "--max-rtf"),
        ("max_elapsed_seconds", "--max-elapsed-seconds"),
        ("min_audio_samples_per_second", "--min-audio-samples-per-second"),
        ("min_planned_scratch_reuse_saves_ratio", "--min-planned-scratch-reuse-saves-ratio"),
    ):
        _non_negative_float_arg(args, name, flag)
    for name, flag in (
        ("max_runtime_peak_bytes", "--max-runtime-peak-bytes"),
        ("max_runtime_command_buffers_submitted", "--max-runtime-command-buffers-submitted"),
        ("max_runtime_buffer_allocations", "--max-runtime-buffer-allocations"),
        ("max_runtime_buffer_bytes_allocated", "--max-runtime-buffer-bytes-allocated"),
        ("max_planned_scratch_capacity_bytes", "--max-planned-scratch-capacity-bytes"),
        ("min_planned_scratch_reuse_saves_bytes", "--min-planned-scratch-reuse-saves-bytes"),
    ):
        _non_negative_int_arg(args, name, flag)


def _runtime_peak_bytes(report: dict[str, Any]) -> int | None:
    peaks: list[int] = []
    for item in _runtime_items(report):
        value = item.get("resident_peak_bytes")
        if isinstance(value, int):
            peaks.append(value)
    if peaks:
        return max(peaks)
    summary = report.get("runtime_summary")
    if isinstance(summary, dict):
        value = summary.get("resident_peak_bytes")
        if isinstance(value, int):
            return value
    nested_peaks: list[int] = []
    for segment in report.get("segments", []):
        if not isinstance(segment, dict):
            continue
        nested = segment.get("report")
        if isinstance(nested, dict):
            value = _runtime_peak_bytes(nested)
            if isinstance(value, int):
                nested_peaks.append(value)
    return max(nested_peaks) if nested_peaks else None


def _runtime_items(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in report.get("runtime_json", []):
        if isinstance(item, dict):
            yield item
    for segment in report.get("segments", []):
        if not isinstance(segment, dict):
            continue
        nested = segment.get("report")
        if isinstance(nested, dict):
            yield from _runtime_items(nested)


def _predicted_codes(report: dict[str, Any]) -> list[int] | None:
    for item in report.get("runtime_json", []):
        if not isinstance(item, dict):
            continue
        codes = item.get("predicted_codes")
        if isinstance(codes, list):
            return [int(v) for v in codes]
    summary = report.get("runtime_summary")
    if isinstance(summary, dict):
        codes = summary.get("predicted_codes")
        if isinstance(codes, list):
            return [int(v) for v in codes]
    return None


def _segment_predicted_codes(report: dict[str, Any]) -> list[list[int]] | None:
    summary = report.get("runtime_summary")
    if isinstance(summary, dict):
        codes = summary.get("segment_predicted_codes")
        if isinstance(codes, list):
            return [[int(v) for v in item] for item in codes if isinstance(item, list)]
    out: list[list[int]] = []
    for segment in report.get("segments", []):
        if not isinstance(segment, dict):
            continue
        nested = segment.get("report")
        if not isinstance(nested, dict):
            continue
        codes = _predicted_codes(nested)
        if codes is not None:
            out.append(codes)
    return out or None


def _text_ids_source(report: dict[str, Any]) -> str | None:
    generation = report.get("generation")
    if isinstance(generation, dict):
        source = generation.get("text_ids_source")
        if isinstance(source, str) and source:
            return source
    gpt = report.get("gpt")
    if isinstance(gpt, dict):
        source = gpt.get("text_ids_source")
        if isinstance(source, str) and source:
            return source
    return None


def _gpt_control_metrics(report: dict[str, Any]) -> dict[str, Any]:
    gpt = report.get("gpt")
    if not isinstance(gpt, dict):
        for segment in report.get("segments", []):
            if not isinstance(segment, dict):
                continue
            nested = segment.get("report")
            if isinstance(nested, dict):
                return _gpt_control_metrics(nested)
        return {}
    out: dict[str, Any] = {}
    for key in (
        "frontend_format",
        "emovec_source",
        "emotion_source",
        "emotion_alpha",
        "emotion_vector",
        "emotion_vector_indices",
        "emotion_vector_random",
        "emotion_vector_random_seed",
        "subsampling_source",
        "conformer_source",
        "perceiver_source",
        "frontend_tail_source",
        "emovec_mix",
    ):
        if key in gpt:
            out[f"gpt_{key}"] = gpt.get(key)
    for tensor_name in (
        "conds_latent",
        "speech_conditioning_latent",
        "text_ids_tensor",
        "fake_inputs",
        "inputs_embeds",
        "attention_mask",
    ):
        tensor = gpt.get(tensor_name)
        if not isinstance(tensor, dict):
            continue
        prefix = f"gpt_{tensor_name}"
        for key in ("dtype", "shape", "sha256"):
            if key in tensor:
                out[f"{prefix}_{key}"] = tensor.get(key)
    return out


def _segment_gpt_metrics(report: dict[str, Any]) -> dict[str, Any]:
    segment_reports: list[dict[str, Any]] = []
    for segment in report.get("segments", []):
        if not isinstance(segment, dict):
            continue
        nested = segment.get("report")
        if isinstance(nested, dict) and isinstance(nested.get("gpt"), dict):
            segment_reports.append(nested)
    if not segment_reports:
        return {}
    segment_summaries = [_gpt_control_metrics(nested) for nested in segment_reports]
    out: dict[str, Any] = {}
    for field in (
        "gpt_frontend_format",
        "gpt_emovec_source",
        "gpt_emotion_source",
        "gpt_emotion_alpha",
        "gpt_emotion_vector",
        "gpt_emotion_vector_indices",
        "gpt_emotion_vector_random",
        "gpt_emotion_vector_random_seed",
        "gpt_subsampling_source",
        "gpt_conformer_source",
        "gpt_perceiver_source",
        "gpt_frontend_tail_source",
        "gpt_emovec_mix",
        "gpt_conds_latent_dtype",
        "gpt_conds_latent_shape",
        "gpt_conds_latent_sha256",
        "gpt_speech_conditioning_latent_dtype",
        "gpt_speech_conditioning_latent_shape",
        "gpt_speech_conditioning_latent_sha256",
        "gpt_text_ids_tensor_dtype",
        "gpt_text_ids_tensor_shape",
        "gpt_text_ids_tensor_sha256",
        "gpt_fake_inputs_dtype",
        "gpt_fake_inputs_shape",
        "gpt_fake_inputs_sha256",
        "gpt_inputs_embeds_dtype",
        "gpt_inputs_embeds_shape",
        "gpt_inputs_embeds_sha256",
        "gpt_attention_mask_dtype",
        "gpt_attention_mask_shape",
        "gpt_attention_mask_sha256",
    ):
        values = [summary.get(field) for summary in segment_summaries]
        if any(value is not None for value in values):
            out[f"segment_{field}"] = values
    return out


def _fill_gpt_frontend_manifest_tensors(report: dict[str, Any], base_dir: Path) -> None:
    gpt = report.get("gpt")
    frontend_dir = report.get("gpt_frontend_dir")
    if isinstance(gpt, dict) and isinstance(frontend_dir, str) and frontend_dir:
        manifest_path = _resolve_report_path(frontend_dir, base_dir) / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
            if isinstance(manifest, dict):
                tensors = manifest.get("tensors")
                if isinstance(tensors, dict):
                    for report_key, manifest_key in (
                        ("conds_latent", "conds_latent"),
                        ("speech_conditioning_latent", "speech_conditioning_latent"),
                        ("text_ids_tensor", "text_ids"),
                        ("fake_inputs", "fake_inputs"),
                        ("inputs_embeds", "inputs_embeds"),
                        ("attention_mask", "attention_mask"),
                    ):
                        if report_key not in gpt and isinstance(tensors.get(manifest_key), dict):
                            gpt[report_key] = tensors[manifest_key]
    for segment in report.get("segments", []):
        if not isinstance(segment, dict):
            continue
        nested = segment.get("report")
        if isinstance(nested, dict):
            _fill_gpt_frontend_manifest_tensors(nested, base_dir)


def _planned_scratch_derived_metrics(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    capacity = fields.get("planned_scratch_capacity_bytes")
    unshared = fields.get("planned_scratch_unshared_phase_peak_total_bytes")
    saved = fields.get("planned_scratch_reuse_saves_bytes")
    if isinstance(capacity, int) and isinstance(unshared, int) and unshared > 0:
        out["planned_scratch_capacity_to_unshared_ratio"] = float(capacity) / float(unshared)
    if isinstance(saved, int) and isinstance(unshared, int) and unshared > 0:
        out["planned_scratch_reuse_saves_ratio"] = float(saved) / float(unshared)
    return out


def _runtime_stage_metrics_from_summary(report: dict[str, Any], elapsed: float | None) -> dict[str, Any]:
    summary = report.get("runtime_summary")
    if not isinstance(summary, dict):
        return {}
    stage_counts = summary.get("stage_counts")
    stage_seconds = summary.get("stage_seconds")
    phase_seconds = summary.get("native_phase_seconds")
    planned_scratch = summary.get("planned_scratch")
    predicted_codes = summary.get("predicted_codes")
    segment_predicted_codes = summary.get("segment_predicted_codes")
    gpt_decode = summary.get("gpt_decode")

    gpt_codes = 0
    if isinstance(predicted_codes, list):
        gpt_codes = len(predicted_codes)
    elif isinstance(segment_predicted_codes, list):
        for item in segment_predicted_codes:
            if isinstance(item, list):
                gpt_codes += len(item)

    out: dict[str, Any] = {
        "runtime_stage_counts": stage_counts if isinstance(stage_counts, dict) else {},
        "runtime_stage_seconds": stage_seconds if isinstance(stage_seconds, dict) and stage_seconds else None,
        "runtime_command_buffers_submitted": summary.get("command_buffers_submitted"),
        "runtime_buffer_allocations": summary.get("buffer_allocations"),
        "runtime_buffer_bytes_allocated": summary.get("buffer_bytes_allocated"),
        "runtime_gpu_elapsed_seconds": summary.get("gpu_elapsed_seconds"),
        "gpt_codes": gpt_codes or None,
    }
    if isinstance(summary.get("native_total_seconds"), (int, float)):
        out["native_runtime_total_seconds"] = float(summary["native_total_seconds"])
    if isinstance(phase_seconds, dict):
        for key, value in phase_seconds.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("native_"):
                out[key] = float(value)
            elif key.endswith("_seconds"):
                out[f"native_{key}"] = float(value)
    if isinstance(planned_scratch, dict):
        out.update(planned_scratch)
        out.update(_planned_scratch_derived_metrics(planned_scratch))
    if isinstance(gpt_decode, dict):
        mapping = {
            "raw_codes_per_second": "native_gpt_raw_codes_per_second",
            "seconds_per_raw_code": "native_gpt_seconds_per_raw_code",
            "codes_per_second": "native_gpt_output_codes_per_second",
            "seconds_per_code": "native_gpt_seconds_per_output_code",
        }
        for src, dst in mapping.items():
            value = gpt_decode.get(src)
            if isinstance(value, (int, float)):
                out[dst] = float(value)
    if elapsed is not None and elapsed > 0 and gpt_codes:
        out["gpt_codes_per_second"] = gpt_codes / elapsed
    return out


def _runtime_stage_metrics(report: dict[str, Any], elapsed: float | None) -> dict[str, Any]:
    items = list(_runtime_items(report))
    if not items:
        summary_metrics = _runtime_stage_metrics_from_summary(report, elapsed)
        if summary_metrics:
            return summary_metrics
    gpt_codes = 0
    generated_mel_frames = 0
    audio_samples = 0
    stage_counts: dict[str, int] = {}
    stage_seconds: dict[str, float] = {}
    native_phase_seconds: dict[str, float] = {}
    native_total_seconds: float | None = None
    command_buffers = 0
    stage_command_buffers: dict[str, int] = {}
    command_buffer_fields: dict[str, int] = {}
    buffer_allocations = 0
    stage_buffer_allocations: dict[str, int] = {}
    buffer_allocation_fields: dict[str, int] = {}
    buffer_bytes_allocated = 0
    stage_buffer_bytes: dict[str, int] = {}
    buffer_byte_fields: dict[str, int] = {}
    planned_scratch_fields: dict[str, Any] = {}
    gpu_elapsed_seconds = 0.0
    stage_gpu_elapsed_seconds: dict[str, float] = {}
    gpu_elapsed_fields: dict[str, float] = {}
    native_gpt_decode: dict[str, float] = {}
    for item in items:
        stage = item.get("stage")
        if isinstance(stage, str):
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            item_elapsed = item.get("elapsed_seconds")
            if isinstance(item_elapsed, (int, float)):
                stage_seconds[stage] = stage_seconds.get(stage, 0.0) + float(item_elapsed)
                if stage in {"hot_tts_inputs_seeded_wav", "hot_tts_inputs_sampled_seeded_wav", "hot_tts_inputs_wav"}:
                    native_total_seconds = float(item_elapsed)
                    for phase in ("gpt", "condition", "noise", "acoustic"):
                        value = item.get(f"{phase}_seconds")
                        if isinstance(value, (int, float)):
                            native_phase_seconds[f"native_{phase}_seconds"] = float(value)
        item_command_buffers = 0
        item_buffer_allocations = 0
        item_buffer_bytes = 0
        item_gpu_elapsed = 0.0
        for key, value in item.items():
            if key.endswith("command_buffers_submitted") and isinstance(value, int):
                command_buffers += value
                item_command_buffers += value
                command_buffer_fields[key] = command_buffer_fields.get(key, 0) + value
            if key.endswith("buffer_allocations") and isinstance(value, int):
                buffer_allocations += value
                item_buffer_allocations += value
                buffer_allocation_fields[key] = buffer_allocation_fields.get(key, 0) + value
            if key.endswith("buffer_bytes_allocated") and isinstance(value, int):
                buffer_bytes_allocated += value
                item_buffer_bytes += value
                buffer_byte_fields[key] = buffer_byte_fields.get(key, 0) + value
            if key.startswith("planned_scratch_"):
                planned_scratch_fields[key] = value
            if key.endswith("gpu_elapsed_seconds") and isinstance(value, (int, float)):
                gpu_elapsed_seconds += float(value)
                item_gpu_elapsed += float(value)
                gpu_elapsed_fields[key] = gpu_elapsed_fields.get(key, 0.0) + float(value)
        if isinstance(stage, str) and item_command_buffers:
            stage_command_buffers[stage] = stage_command_buffers.get(stage, 0) + item_command_buffers
        if isinstance(stage, str) and item_buffer_allocations:
            stage_buffer_allocations[stage] = stage_buffer_allocations.get(stage, 0) + item_buffer_allocations
        if isinstance(stage, str) and item_buffer_bytes:
            stage_buffer_bytes[stage] = stage_buffer_bytes.get(stage, 0) + item_buffer_bytes
        if isinstance(stage, str) and item_gpu_elapsed:
            stage_gpu_elapsed_seconds[stage] = stage_gpu_elapsed_seconds.get(stage, 0.0) + item_gpu_elapsed
        codes = item.get("predicted_codes")
        if isinstance(codes, list):
            gpt_codes += len(codes)
        if stage == "gpt_kv_codes_inputs_export":
            for src, dst in (
                ("raw_codes_per_second", "native_gpt_raw_codes_per_second"),
                ("seconds_per_raw_code", "native_gpt_seconds_per_raw_code"),
                ("codes_per_second", "native_gpt_output_codes_per_second"),
                ("seconds_per_code", "native_gpt_seconds_per_output_code"),
            ):
                value = item.get(src)
                if isinstance(value, (int, float)):
                    native_gpt_decode[dst] = float(value)
        if stage == "hot_tts_condition_inputs_wav":
            generated = item.get("generated_tokens")
            samples = item.get("samples")
            if isinstance(generated, int):
                generated_mel_frames += generated
            if isinstance(samples, int):
                audio_samples += samples
    out: dict[str, Any] = {
        "runtime_stage_counts": stage_counts,
        "runtime_stage_seconds": stage_seconds or None,
        "runtime_stage_command_buffers": stage_command_buffers or None,
        "runtime_command_buffer_fields": command_buffer_fields or None,
        "runtime_command_buffers_submitted": command_buffers or None,
        "runtime_stage_buffer_allocations": stage_buffer_allocations or None,
        "runtime_buffer_allocation_fields": buffer_allocation_fields or None,
        "runtime_buffer_allocations": buffer_allocations or None,
        "runtime_stage_buffer_bytes_allocated": stage_buffer_bytes or None,
        "runtime_buffer_byte_fields": buffer_byte_fields or None,
        "runtime_buffer_bytes_allocated": buffer_bytes_allocated or None,
        "runtime_stage_gpu_elapsed_seconds": stage_gpu_elapsed_seconds or None,
        "runtime_gpu_elapsed_fields": gpu_elapsed_fields or None,
        "runtime_gpu_elapsed_seconds": gpu_elapsed_seconds or None,
        "gpt_codes": gpt_codes or None,
        "generated_mel_frames": generated_mel_frames or None,
        "audio_samples": audio_samples or None,
    }
    out.update(planned_scratch_fields)
    out.update(_planned_scratch_derived_metrics(planned_scratch_fields))
    out.update(native_gpt_decode)
    if native_total_seconds is not None:
        out["native_runtime_total_seconds"] = native_total_seconds
    out.update(native_phase_seconds)
    if elapsed is not None and elapsed > 0:
        if gpt_codes:
            out["gpt_codes_per_second"] = gpt_codes / elapsed
        if generated_mel_frames:
            out["mel_frames_per_second"] = generated_mel_frames / elapsed
        if audio_samples:
            out["audio_samples_per_second"] = audio_samples / elapsed
    return out


def _timing_metrics(report: dict[str, Any]) -> dict[str, Any]:
    timing = report.get("timing")
    if not isinstance(timing, dict):
        return {"timing": None}
    out: dict[str, Any] = {"timing": timing}
    for field in (
        "prepare_voice_seconds",
        "split_segments_seconds",
        "frontend_seconds",
        "runtime_seconds",
        "postprocess_seconds",
        "segment_seconds",
        "concat_seconds",
    ):
        value = timing.get(field)
        if isinstance(value, (int, float)):
            out[field] = float(value)
    return out


def _generation_metrics(report: dict[str, Any]) -> dict[str, Any]:
    generation = report.get("generation")
    if not isinstance(generation, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "max_mel_tokens",
        "prompt_tokens",
        "steps",
        "cfg_rate",
        "temperature",
        "seed",
        "gpt_do_sample",
        "gpt_temperature",
        "gpt_top_k",
        "gpt_top_p",
        "gpt_repetition_penalty",
        "shared_runtime_stages",
        "all_segments",
        "segments",
        "max_text_tokens_per_segment",
        "interval_silence_ms",
        "emotion_vector_random",
        "emotion_vector_random_seed",
    ):
        if key in generation:
            out[f"generation_{key}"] = generation.get(key)
    return out


def _voice_gate_metrics(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    model_contract = report.get("model_contract")
    if isinstance(model_contract, dict):
        out["model_contract_ok"] = bool(model_contract.get("ok"))
        runtime_report = model_contract.get("runtime_report")
        if isinstance(runtime_report, dict):
            for key in ("weights_bytes", "tensor_count", "tensor_bytes", "required_tensor_count"):
                value = runtime_report.get(key)
                if isinstance(value, int):
                    out[f"model_contract_{key}"] = value
            integrity = runtime_report.get("integrity")
            if isinstance(integrity, dict):
                for key in ("aligned_tensor_count", "checked_interval_count", "sha256_verified_count"):
                    value = integrity.get(key)
                    if isinstance(value, int):
                        out[f"model_contract_integrity_{key}"] = value
    voice_validation = report.get("voice_validation")
    if isinstance(voice_validation, dict):
        out["voice_validation_ok"] = bool(voice_validation.get("ok"))
    voice_contract = report.get("voice_contract")
    if isinstance(voice_contract, dict):
        out["voice_contract_ok"] = bool(voice_contract.get("ok"))
        runtime_report = voice_contract.get("runtime_report")
        if isinstance(runtime_report, dict):
            for key in ("weights_bytes", "spk_cond_tokens", "prompt_tokens", "mel_frames", "tensor_count", "tensor_bytes"):
                value = runtime_report.get(key)
                if isinstance(value, int):
                    out[f"voice_contract_{key}"] = value
            integrity = runtime_report.get("integrity")
            if isinstance(integrity, dict):
                for key in ("aligned_tensor_count", "checked_interval_count", "sha256_verified_count"):
                    value = integrity.get(key)
                    if isinstance(value, int):
                        out[f"voice_contract_integrity_{key}"] = value
    return out


def _sidecar_file_metrics(path_value: Any, base_dir: Path) -> dict[str, Any] | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = _resolve_report_path(path_value, base_dir)
    if not path.exists() or not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _sidecar_metrics(report: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("codes_u32", "condition_f32", "noise_f32"):
        metrics = _sidecar_file_metrics(report.get(field), base_dir)
        if metrics is None:
            continue
        for key in ("path", "bytes", "sha256"):
            out[f"{field}_{key}"] = metrics[key]
    segment_reports: list[dict[str, Any]] = []
    for segment in report.get("segments", []):
        if not isinstance(segment, dict):
            continue
        nested = segment.get("report")
        if isinstance(nested, dict):
            segment_reports.append(nested)
    if segment_reports:
        for field in ("codes_u32", "condition_f32", "noise_f32"):
            paths: list[str] = []
            byte_counts: list[int] = []
            hashes: list[str] = []
            for nested in segment_reports:
                metrics = _sidecar_file_metrics(nested.get(field), base_dir)
                if metrics is None:
                    continue
                paths.append(str(metrics["path"]))
                byte_counts.append(int(metrics["bytes"]))
                hashes.append(str(metrics["sha256"]))
            if paths:
                out[f"segment_{field}_paths"] = paths
                out[f"segment_{field}_bytes"] = byte_counts
                out[f"segment_{field}_sha256"] = hashes
    return out


def _wav_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    with wave.open(str(path), "rb") as fp:
        frames = fp.getnframes()
        rate = fp.getframerate()
    if rate <= 0:
        return None
    return frames / float(rate)


def _summarize_report(case: str, report_path: Path, elapsed_seconds: float | None = None) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _fill_gpt_frontend_manifest_tensors(report, report_path.parent)
    output_wav = _resolve_report_path(report["output_wav"], report_path.parent)
    audio_seconds = _wav_seconds(output_wav)
    elapsed = elapsed_seconds if elapsed_seconds is not None else report.get("elapsed_seconds")
    out: dict[str, Any] = {
        "case": case,
        "status": "passed",
        "report_json": str(report_path.resolve()),
        "format": report.get("format"),
        "output_wav": str(output_wav.resolve()),
        "output_wav_sha256": report.get("output_wav_sha256") or file_sha256(output_wav),
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed,
        "rtf": (elapsed / audio_seconds) if elapsed is not None and audio_seconds and audio_seconds > 0 else None,
        "runtime_peak_bytes": _runtime_peak_bytes(report),
        "predicted_codes": _predicted_codes(report),
        "segment_predicted_codes": _segment_predicted_codes(report),
        "text_ids_source": _text_ids_source(report),
        "generation": report.get("generation"),
    }
    out.update(_generation_metrics(report))
    out.update(_gpt_control_metrics(report))
    out.update(_segment_gpt_metrics(report))
    out.update(_voice_gate_metrics(report))
    out.update(_sidecar_metrics(report, report_path.parent))
    out.update(_timing_metrics(report))
    out.update(_runtime_stage_metrics(report, elapsed))
    if report.get("format") == "mit2-native-hot-multisegment-synthesis-report":
        out["segments"] = report.get("generation", {}).get("segments")
        out["concat"] = report.get("concat")
    return out


def _summary_case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError("benchmark summary must contain a cases list")
    out: dict[str, dict[str, Any]] = {}
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("benchmark summary cases must be objects")
        case = item.get("case")
        if not isinstance(case, str) or not case:
            raise ValueError("benchmark summary case is missing a string case name")
        if case in out:
            raise ValueError(f"duplicate benchmark case: {case}")
        out[case] = item
    return out


def _case_names(current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> Iterable[str]:
    return sorted(set(current) | set(baseline))


def _paired_numeric_delta(field: str, base_case: dict[str, Any], variant_case: dict[str, Any]) -> dict[str, Any]:
    base = base_case.get(field)
    variant = variant_case.get(field)
    out: dict[str, Any] = {
        "field": field,
        "base": base,
        "variant": variant,
        "status": "skipped",
    }
    if not isinstance(base, (int, float)) or not isinstance(variant, (int, float)):
        return out
    out["delta"] = float(variant) - float(base)
    if base != 0:
        out["ratio"] = float(variant) / float(base)
        out["relative_delta"] = (float(variant) - float(base)) / float(base)
    out["status"] = "measured"
    return out


def _paired_exact_match(field: str, base_case: dict[str, Any], variant_case: dict[str, Any]) -> dict[str, Any]:
    base = base_case.get(field)
    variant = variant_case.get(field)
    out: dict[str, Any] = {
        "field": field,
        "base": base,
        "variant": variant,
        "status": "skipped",
    }
    if base is None and variant is None:
        return out
    out["status"] = "passed" if base == variant else "failed"
    return out


def compare_summary_resource_cases(summary: dict[str, Any]) -> list[dict[str, Any]]:
    cases = _summary_case_map(summary)
    pairs = [("short_greedy", "shared_short_greedy")]
    fields = (
        "elapsed_seconds",
        "rtf",
        "runtime_peak_bytes",
        "runtime_command_buffers_submitted",
        "runtime_buffer_allocations",
        "runtime_buffer_bytes_allocated",
        "runtime_gpu_elapsed_seconds",
        "audio_samples_per_second",
        "native_runtime_total_seconds",
        "native_gpt_seconds",
        "native_condition_seconds",
        "native_acoustic_seconds",
        "planned_scratch_segments",
        "planned_scratch_unshared_phase_peak_total_bytes",
        "planned_scratch_capacity_bytes",
        "planned_scratch_reuse_saves_bytes",
        "planned_scratch_capacity_to_unshared_ratio",
        "planned_scratch_reuse_saves_ratio",
        "planned_scratch_actual_codes",
        "planned_scratch_actual_generated_tokens",
        "planned_scratch_actual_total_mel_tokens",
        "planned_scratch_code_slack",
        "planned_scratch_generated_token_slack",
        "planned_scratch_total_mel_token_slack",
    )
    exact_fields = (
        "planned_scratch_ok",
        "planned_scratch_segments",
        "planned_scratch_alignment",
        "planned_scratch_max_prefix_tokens",
        "planned_scratch_max_codes",
        "planned_scratch_prompt_tokens",
        "planned_scratch_source",
        "planned_scratch_cond_tokens",
        "planned_scratch_text_tokens",
        "planned_scratch_generated_tokens",
        "planned_scratch_total_mel_tokens",
        "planned_scratch_gpt_kv_cache_bytes",
        "planned_scratch_gpt_phase_peak_bytes",
        "planned_scratch_condition_tensor_bytes",
        "planned_scratch_condition_phase_peak_bytes",
        "planned_scratch_acoustic_mel_tensor_bytes",
        "planned_scratch_acoustic_phase_peak_bytes",
        "planned_scratch_unshared_phase_peak_total_bytes",
        "planned_scratch_capacity_bytes",
        "planned_scratch_reuse_saves_bytes",
        "planned_scratch_capacity_to_unshared_ratio",
        "planned_scratch_reuse_saves_ratio",
        "planned_scratch_actual_codes",
        "planned_scratch_actual_generated_tokens",
        "planned_scratch_actual_total_mel_tokens",
        "planned_scratch_code_slack",
        "planned_scratch_generated_token_slack",
        "planned_scratch_total_mel_token_slack",
        "planned_scratch_covers_actual",
    )
    comparisons: list[dict[str, Any]] = []
    for base_name, variant_name in pairs:
        base_case = cases.get(base_name)
        variant_case = cases.get(variant_name)
        if base_case is None or variant_case is None:
            continue
        comparison = {
            "base_case": base_name,
            "variant_case": variant_name,
            "status": "measured",
            "metrics": [_paired_numeric_delta(field, base_case, variant_case) for field in fields],
            "invariants": [_paired_exact_match(field, base_case, variant_case) for field in exact_fields],
        }
        if base_case.get("status", "passed") != "passed" or variant_case.get("status", "passed") != "passed":
            comparison["status"] = "skipped"
        elif any(check.get("status") == "failed" for check in comparison["invariants"]):
            comparison["status"] = "failed"
        comparisons.append(comparison)
    return comparisons


def _numeric_comparison(
    field: str,
    current_case: dict[str, Any],
    baseline_case: dict[str, Any],
    max_relative_regression: float,
) -> dict[str, Any]:
    current = current_case.get(field)
    baseline = baseline_case.get(field)
    out: dict[str, Any] = {
        "field": field,
        "current": current,
        "baseline": baseline,
        "status": "skipped",
    }
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)) or baseline <= 0:
        return out
    relative_delta = (float(current) - float(baseline)) / float(baseline)
    out["relative_delta"] = relative_delta
    out["max_relative_regression"] = max_relative_regression
    out["status"] = "failed" if relative_delta > max_relative_regression else "passed"
    return out


def _throughput_comparison(
    field: str,
    current_case: dict[str, Any],
    baseline_case: dict[str, Any],
    max_relative_regression: float,
) -> dict[str, Any]:
    current = current_case.get(field)
    baseline = baseline_case.get(field)
    out: dict[str, Any] = {
        "field": field,
        "current": current,
        "baseline": baseline,
        "status": "skipped",
    }
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)) or baseline <= 0:
        return out
    relative_delta = (float(current) - float(baseline)) / float(baseline)
    out["relative_delta"] = relative_delta
    out["max_relative_regression"] = max_relative_regression
    out["status"] = "failed" if relative_delta < -max_relative_regression else "passed"
    return out


def _string_match_comparison(field: str, current_case: dict[str, Any], baseline_case: dict[str, Any]) -> dict[str, Any]:
    current = current_case.get(field)
    baseline = baseline_case.get(field)
    out: dict[str, Any] = {
        "field": field,
        "current": current,
        "baseline": baseline,
        "status": "skipped",
    }
    if baseline is None:
        return out
    out["status"] = "passed" if current == baseline else "failed"
    return out


def _exact_match_comparison(field: str, current_case: dict[str, Any], baseline_case: dict[str, Any]) -> dict[str, Any]:
    current = current_case.get(field)
    baseline = baseline_case.get(field)
    out: dict[str, Any] = {
        "field": field,
        "current": current,
        "baseline": baseline,
        "status": "skipped",
    }
    if baseline is None:
        return out
    out["status"] = "passed" if current == baseline else "failed"
    return out


def compare_benchmark_summaries(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_rtf_regression: float = 0.10,
    max_elapsed_regression: float = 0.10,
    max_memory_regression: float = 0.10,
    max_throughput_regression: float = 0.10,
    require_hash_match: bool = True,
    require_code_match: bool = True,
    allow_new_cases: bool = False,
) -> dict[str, Any]:
    current_cases = _summary_case_map(current)
    baseline_cases = _summary_case_map(baseline)
    compared_cases: list[dict[str, Any]] = []
    failed = 0
    for case_name in _case_names(current_cases, baseline_cases):
        current_case = current_cases.get(case_name)
        baseline_case = baseline_cases.get(case_name)
        case_report: dict[str, Any] = {"case": case_name, "status": "passed", "checks": []}
        if current_case is None:
            case_report["status"] = "failed"
            case_report["checks"].append({"field": "case", "status": "failed", "reason": "missing_current_case"})
        elif baseline_case is None:
            status = "passed" if allow_new_cases else "failed"
            case_report["status"] = status
            case_report["checks"].append({"field": "case", "status": status, "reason": "new_case"})
        else:
            if current_case.get("status", "passed") != "passed":
                case_report["checks"].append(
                    {"field": "status", "status": "failed", "current": current_case.get("status")}
                )
            if require_hash_match:
                current_hash = current_case.get("output_wav_sha256")
                baseline_hash = baseline_case.get("output_wav_sha256")
                status = "passed" if current_hash == baseline_hash and current_hash is not None else "failed"
                case_report["checks"].append(
                    {
                        "field": "output_wav_sha256",
                        "status": status,
                        "current": current_hash,
                        "baseline": baseline_hash,
                    }
                )
            if require_code_match:
                current_codes = current_case.get("predicted_codes")
                baseline_codes = baseline_case.get("predicted_codes")
                if current_codes is None and baseline_codes is None:
                    status = "skipped"
                else:
                    status = "passed" if current_codes == baseline_codes else "failed"
                case_report["checks"].append(
                    {
                        "field": "predicted_codes",
                        "status": status,
                        "current": current_codes,
                        "baseline": baseline_codes,
                    }
                )
                current_segment_codes = current_case.get("segment_predicted_codes")
                baseline_segment_codes = baseline_case.get("segment_predicted_codes")
                if baseline_segment_codes is None:
                    status = "skipped"
                else:
                    status = "passed" if current_segment_codes == baseline_segment_codes else "failed"
                case_report["checks"].append(
                    {
                        "field": "segment_predicted_codes",
                        "status": status,
                        "current": current_segment_codes,
                        "baseline": baseline_segment_codes,
                    }
                )
            case_report["checks"].append(_string_match_comparison("text_ids_source", current_case, baseline_case))
            for field in (
                "gpt_frontend_format",
                "gpt_emovec_source",
                "gpt_emotion_source",
                "gpt_emotion_alpha",
                "gpt_emotion_vector",
                "gpt_emotion_vector_indices",
                "gpt_emotion_vector_random",
                "gpt_emotion_vector_random_seed",
                "gpt_subsampling_source",
                "gpt_conformer_source",
                "gpt_perceiver_source",
                "gpt_frontend_tail_source",
                "gpt_emovec_mix",
                "gpt_conds_latent_dtype",
                "gpt_conds_latent_shape",
                "gpt_conds_latent_sha256",
                "gpt_speech_conditioning_latent_dtype",
                "gpt_speech_conditioning_latent_shape",
                "gpt_speech_conditioning_latent_sha256",
                "gpt_text_ids_tensor_dtype",
                "gpt_text_ids_tensor_shape",
                "gpt_text_ids_tensor_sha256",
                "gpt_fake_inputs_dtype",
                "gpt_fake_inputs_shape",
                "gpt_fake_inputs_sha256",
                "gpt_inputs_embeds_dtype",
                "gpt_inputs_embeds_shape",
                "gpt_inputs_embeds_sha256",
                "gpt_attention_mask_dtype",
                "gpt_attention_mask_shape",
                "gpt_attention_mask_sha256",
                "segment_gpt_frontend_format",
                "segment_gpt_emovec_source",
                "segment_gpt_emotion_source",
                "segment_gpt_emotion_alpha",
                "segment_gpt_emotion_vector",
                "segment_gpt_emotion_vector_indices",
                "segment_gpt_emotion_vector_random",
                "segment_gpt_emotion_vector_random_seed",
                "segment_gpt_subsampling_source",
                "segment_gpt_conformer_source",
                "segment_gpt_perceiver_source",
                "segment_gpt_frontend_tail_source",
                "segment_gpt_emovec_mix",
                "segment_gpt_conds_latent_dtype",
                "segment_gpt_conds_latent_shape",
                "segment_gpt_conds_latent_sha256",
                "segment_gpt_speech_conditioning_latent_dtype",
                "segment_gpt_speech_conditioning_latent_shape",
                "segment_gpt_speech_conditioning_latent_sha256",
                "segment_gpt_text_ids_tensor_dtype",
                "segment_gpt_text_ids_tensor_shape",
                "segment_gpt_text_ids_tensor_sha256",
                "segment_gpt_fake_inputs_dtype",
                "segment_gpt_fake_inputs_shape",
                "segment_gpt_fake_inputs_sha256",
                "segment_gpt_inputs_embeds_dtype",
                "segment_gpt_inputs_embeds_shape",
                "segment_gpt_inputs_embeds_sha256",
                "segment_gpt_attention_mask_dtype",
                "segment_gpt_attention_mask_shape",
                "segment_gpt_attention_mask_sha256",
                "codes_u32_bytes",
                "codes_u32_sha256",
                "condition_f32_bytes",
                "condition_f32_sha256",
                "noise_f32_bytes",
                "noise_f32_sha256",
                "segment_codes_u32_bytes",
                "segment_codes_u32_sha256",
                "segment_condition_f32_bytes",
                "segment_condition_f32_sha256",
                "segment_noise_f32_bytes",
                "segment_noise_f32_sha256",
                "runtime_stage_counts",
                "planned_scratch_ok",
                "planned_scratch_segments",
                "planned_scratch_alignment",
                "planned_scratch_max_prefix_tokens",
                "planned_scratch_max_codes",
                "planned_scratch_prompt_tokens",
                "planned_scratch_source",
                "planned_scratch_cond_tokens",
                "planned_scratch_text_tokens",
                "planned_scratch_generated_tokens",
                "planned_scratch_total_mel_tokens",
                "planned_scratch_gpt_kv_cache_bytes",
                "planned_scratch_gpt_phase_peak_bytes",
                "planned_scratch_condition_tensor_bytes",
                "planned_scratch_condition_phase_peak_bytes",
                "planned_scratch_acoustic_mel_tensor_bytes",
                "planned_scratch_acoustic_phase_peak_bytes",
                "planned_scratch_unshared_phase_peak_total_bytes",
                "planned_scratch_capacity_bytes",
                "planned_scratch_reuse_saves_bytes",
                "planned_scratch_capacity_to_unshared_ratio",
                "planned_scratch_reuse_saves_ratio",
                "planned_scratch_actual_codes",
                "planned_scratch_actual_generated_tokens",
                "planned_scratch_actual_total_mel_tokens",
                "planned_scratch_code_slack",
                "planned_scratch_generated_token_slack",
                "planned_scratch_total_mel_token_slack",
                "planned_scratch_covers_actual",
            ):
                case_report["checks"].append(_exact_match_comparison(field, current_case, baseline_case))
            case_report["checks"].append(_exact_match_comparison("model_contract_ok", current_case, baseline_case))
            case_report["checks"].append(_exact_match_comparison("voice_validation_ok", current_case, baseline_case))
            case_report["checks"].append(_exact_match_comparison("voice_contract_ok", current_case, baseline_case))
            for field in (
                "model_contract_weights_bytes",
                "model_contract_tensor_count",
                "model_contract_tensor_bytes",
                "model_contract_required_tensor_count",
                "model_contract_integrity_aligned_tensor_count",
                "model_contract_integrity_checked_interval_count",
                "model_contract_integrity_sha256_verified_count",
                "voice_contract_weights_bytes",
                "voice_contract_tensor_count",
                "voice_contract_tensor_bytes",
                "voice_contract_spk_cond_tokens",
                "voice_contract_prompt_tokens",
                "voice_contract_mel_frames",
                "voice_contract_integrity_aligned_tensor_count",
                "voice_contract_integrity_checked_interval_count",
                "voice_contract_integrity_sha256_verified_count",
            ):
                case_report["checks"].append(_exact_match_comparison(field, current_case, baseline_case))
            for field in (
                "generation_max_mel_tokens",
                "generation_prompt_tokens",
                "generation_steps",
                "generation_cfg_rate",
                "generation_temperature",
                "generation_seed",
                "generation_gpt_do_sample",
                "generation_gpt_temperature",
                "generation_gpt_top_k",
                "generation_gpt_top_p",
                "generation_gpt_repetition_penalty",
                "generation_shared_runtime_stages",
                "generation_all_segments",
                "generation_segments",
                "generation_max_text_tokens_per_segment",
                "generation_interval_silence_ms",
                "generation_emotion_vector_random",
                "generation_emotion_vector_random_seed",
            ):
                case_report["checks"].append(_exact_match_comparison(field, current_case, baseline_case))
            case_report["checks"].append(
                _numeric_comparison("rtf", current_case, baseline_case, max_rtf_regression)
            )
            case_report["checks"].append(
                _numeric_comparison("elapsed_seconds", current_case, baseline_case, max_elapsed_regression)
            )
            case_report["checks"].append(
                _numeric_comparison("runtime_peak_bytes", current_case, baseline_case, max_memory_regression)
            )
            for field in (
                "prepare_voice_seconds",
                "split_segments_seconds",
                "frontend_seconds",
                "runtime_seconds",
                "postprocess_seconds",
                "segment_seconds",
                "concat_seconds",
                "native_runtime_total_seconds",
                "native_gpt_seconds",
                "native_condition_seconds",
                "native_noise_seconds",
                "native_acoustic_seconds",
                "runtime_command_buffers_submitted",
                "runtime_buffer_allocations",
                "runtime_buffer_bytes_allocated",
                "runtime_gpu_elapsed_seconds",
            ):
                case_report["checks"].append(
                    _numeric_comparison(field, current_case, baseline_case, max_elapsed_regression)
                )
            for field in ("gpt_codes_per_second", "mel_frames_per_second", "audio_samples_per_second"):
                case_report["checks"].append(
                    _throughput_comparison(field, current_case, baseline_case, max_throughput_regression)
                )
            if any(check.get("status") == "failed" for check in case_report["checks"]):
                case_report["status"] = "failed"
        if case_report["status"] != "passed":
            failed += 1
        compared_cases.append(case_report)
    return {
        "format": "mit2-native-hot-benchmark-comparison",
        "version": 1,
        "cases": compared_cases,
        "passed": len(compared_cases) - failed,
        "failed": failed,
        "thresholds": {
            "max_rtf_regression": max_rtf_regression,
            "max_elapsed_regression": max_elapsed_regression,
            "max_memory_regression": max_memory_regression,
            "max_throughput_regression": max_throughput_regression,
            "require_hash_match": require_hash_match,
            "require_code_match": require_code_match,
            "allow_new_cases": allow_new_cases,
        },
    }


def _max_budget_check(case: dict[str, Any], field: str, limit: float | int | None) -> dict[str, Any] | None:
    if limit is None:
        return None
    current = case.get(field)
    out: dict[str, Any] = {
        "case": case.get("case"),
        "field": field,
        "current": current,
        "budget": limit,
        "status": "skipped",
        "direction": "max",
    }
    if current is None:
        return out
    out["status"] = "passed" if float(current) <= float(limit) else "failed"
    return out


def _min_budget_check(case: dict[str, Any], field: str, limit: float | int | None) -> dict[str, Any] | None:
    if limit is None:
        return None
    current = case.get(field)
    out: dict[str, Any] = {
        "case": case.get("case"),
        "field": field,
        "current": current,
        "budget": limit,
        "status": "skipped",
        "direction": "min",
    }
    if current is None:
        return out
    out["status"] = "passed" if float(current) >= float(limit) else "failed"
    return out


def apply_summary_budgets(
    cases: list[dict[str, Any]],
    *,
    max_runtime_peak_bytes: int | None = None,
    max_runtime_command_buffers_submitted: int | None = None,
    max_runtime_buffer_allocations: int | None = None,
    max_runtime_buffer_bytes_allocated: int | None = None,
    max_runtime_gpu_elapsed_seconds: float | None = None,
    max_rtf: float | None = None,
    max_elapsed_seconds: float | None = None,
    min_audio_samples_per_second: float | None = None,
    max_planned_scratch_capacity_bytes: int | None = None,
    min_planned_scratch_reuse_saves_bytes: int | None = None,
    min_planned_scratch_reuse_saves_ratio: float | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for case in cases:
        for check in (
            _max_budget_check(case, "runtime_peak_bytes", max_runtime_peak_bytes),
            _max_budget_check(
                case,
                "runtime_command_buffers_submitted",
                max_runtime_command_buffers_submitted,
            ),
            _max_budget_check(case, "runtime_buffer_allocations", max_runtime_buffer_allocations),
            _max_budget_check(case, "runtime_buffer_bytes_allocated", max_runtime_buffer_bytes_allocated),
            _max_budget_check(case, "runtime_gpu_elapsed_seconds", max_runtime_gpu_elapsed_seconds),
            _max_budget_check(case, "rtf", max_rtf),
            _max_budget_check(case, "elapsed_seconds", max_elapsed_seconds),
            _min_budget_check(case, "audio_samples_per_second", min_audio_samples_per_second),
            _max_budget_check(case, "planned_scratch_capacity_bytes", max_planned_scratch_capacity_bytes),
            _min_budget_check(case, "planned_scratch_reuse_saves_bytes", min_planned_scratch_reuse_saves_bytes),
            _min_budget_check(case, "planned_scratch_reuse_saves_ratio", min_planned_scratch_reuse_saves_ratio),
        ):
            if check is None:
                continue
            checks.append(check)
            if check["status"] == "failed":
                case["status"] = "failed"
    return checks


def _case_args(args: argparse.Namespace, case: str, case_dir: Path) -> list[str]:
    max_mel_tokens = (
        getattr(args, "full_acoustic_max_mel_tokens", args.max_mel_tokens)
        if case in _FULL_ACOUSTIC_CASES
        else args.max_mel_tokens
    )
    steps = (
        getattr(args, "full_acoustic_steps", args.steps)
        if case in _FULL_ACOUSTIC_CASES
        else args.steps
    )
    base = [
        getattr(args, "python_executable", sys.executable),
        "-m",
        "metal_indextts2.tools.synthesize_native_hot",
        "--index-tts-repo",
        args.index_tts_repo,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--model-bundle",
        args.model_bundle,
        "--runtime",
        args.runtime,
        "--work-dir",
        str(case_dir),
        "--output-wav",
        str(case_dir.with_suffix(".wav")),
        "--report-json",
        str(case_dir / "report.json"),
        "--max-mel-tokens",
        str(max_mel_tokens),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--steps",
        str(steps),
        "--cfg-rate",
        str(args.cfg_rate),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
    ]
    if args.voice_bundle and case not in _CLONE_CASES:
        base += ["--voice-bundle", args.voice_bundle]
    if getattr(args, "validate_voice_source", False):
        base += ["--validate-voice-source"]
    if getattr(args, "validate_model_contract", False):
        base += ["--validate-model-contract"]
    if getattr(args, "validate_voice_contract", False):
        base += ["--validate-voice-contract"]
    if args.voice_name and case not in _CLONE_CASES:
        base += ["--voice-name", args.voice_name]
    if args.text_ids_file and case in _SHORT_TEXT_ID_CASES:
        base += ["--text-ids-file", args.text_ids_file]
    if args.native_flags:
        base += [
            "--native-subsampling",
            "--native-conformer-stack",
            "--native-perceiver",
            "--native-frontend-tail",
            "--native-emovec",
        ]
    if getattr(args, "compact_reports", False):
        base += ["--compact-report"]
    if case == "shared_short_greedy" or (getattr(args, "shared_runtime_stages", False) and case != "short_sampled"):
        base += ["--shared-runtime-stages"]
    if case in {"short_greedy", "shared_short_greedy"}:
        return base + ["--text", args.short_text]
    if case == "short_acoustic_full":
        return base + ["--text", args.short_text]
    if case == "short_sampled":
        return base + [
            "--text",
            args.short_text,
            "--gpt-do-sample",
            "--gpt-temperature",
            str(args.gpt_temperature),
            "--gpt-top-k",
            str(args.gpt_top_k),
            "--gpt-top-p",
            str(args.gpt_top_p),
            "--gpt-repetition-penalty",
            str(args.gpt_repetition_penalty),
        ]
    if case == "medium_greedy":
        return base + ["--text", args.medium_text]
    if case == "medium_acoustic_full":
        return base + ["--text", args.medium_text]
    if case == "emotion_profile":
        emotion_voice_name = getattr(args, "emotion_voice_name", None)
        if not emotion_voice_name:
            raise ValueError("emotion_profile case requires --emotion-voice-name")
        return base + [
            "--text",
            args.short_text,
            "--emotion-voice-name",
            emotion_voice_name,
            "--emotion-alpha",
            str(getattr(args, "emotion_alpha", 1.0)),
        ]
    if case == "emotion_audio":
        emotion_audio = getattr(args, "emotion_audio", None)
        if not emotion_audio:
            raise ValueError("emotion_audio case requires --emotion-audio")
        return base + [
            "--text",
            args.short_text,
            "--emotion-audio",
            emotion_audio,
            "--emotion-alpha",
            str(getattr(args, "emotion_alpha", 1.0)),
        ]
    if case == "emotion_vector":
        command = base + [
            "--text",
            args.short_text,
            "--emotion-vector",
            getattr(args, "emotion_vector", "0,0,0,0,0,0,0,0"),
            "--emotion-alpha",
            str(getattr(args, "emotion_alpha", 1.0)),
        ]
        if getattr(args, "emotion_vector_raw", False):
            command.append("--emotion-vector-raw")
        if getattr(args, "emotion_vector_random", False):
            command.append("--emotion-vector-random")
            if getattr(args, "emotion_vector_random_seed", None) is not None:
                command += ["--emotion-vector-random-seed", str(args.emotion_vector_random_seed)]
        return command
    if case == "all_segments":
        return base + [
            "--text",
            args.multi_text,
            "--all-segments",
            "--max-text-tokens-per-segment",
            str(args.segment_tokens),
            "--interval-silence-ms",
            str(args.interval_silence_ms),
        ]
    if case == "native_cjk_segments":
        return base + [
            "--text",
            args.multi_text,
            "--all-segments",
            "--native-cjk-text-ids",
            "--max-text-tokens-per-segment",
            str(args.segment_tokens),
            "--interval-silence-ms",
            str(args.interval_silence_ms),
        ]
    if case == "native_cjk_mixed":
        return base + [
            "--text",
            getattr(args, "native_cjk_mixed_text", "AB1C2你好"),
            "--native-cjk-text-ids",
        ]
    if case == "long_all_segments":
        return base + [
            "--text",
            args.long_text,
            "--all-segments",
            "--max-text-tokens-per-segment",
            str(args.long_segment_tokens),
            "--interval-silence-ms",
            str(args.interval_silence_ms),
        ]
    if case == "long_acoustic_full_segments":
        return base + [
            "--text",
            args.long_text,
            "--all-segments",
            "--max-text-tokens-per-segment",
            str(args.long_segment_tokens),
            "--interval-silence-ms",
            str(args.interval_silence_ms),
        ]
    if case == "clone":
        if not args.clone_audio:
            raise ValueError("clone case requires --clone-audio")
        return base + [
            "--clone-audio",
            args.clone_audio,
            "--clone-name",
            args.clone_name,
            "--overwrite-voice",
            "--text",
            args.short_text,
        ]
    if case == "clone_acoustic_full":
        if not args.clone_audio:
            raise ValueError("clone_acoustic_full case requires --clone-audio")
        return base + [
            "--clone-audio",
            args.clone_audio,
            "--clone-name",
            args.clone_name,
            "--overwrite-voice",
            "--text",
            args.short_text,
        ]
    raise ValueError(f"unknown benchmark case: {case}")


def _run_case(args: argparse.Namespace, case: str, out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / case
    command = _case_args(args, case, case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    (case_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (case_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        return {
            "case": case,
            "status": "failed",
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "command": command,
            "stdout": str(case_dir / "stdout.txt"),
            "stderr": str(case_dir / "stderr.txt"),
        }
    summary = _summarize_report(case, case_dir / "report.json", elapsed)
    summary["status"] = "passed"
    summary["command"] = command
    return summary


def _benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in (
        "python_executable",
        "runtime",
        "model_bundle",
        "voice_bundle",
        "voice_name",
        "validate_voice_source",
        "validate_model_contract",
        "validate_voice_contract",
        "clone_audio",
        "clone_name",
        "text_ids_file",
        "cases",
        "from_reports",
        "baseline_summary",
        "short_text",
        "multi_text",
        "medium_text",
        "native_cjk_mixed_text",
        "long_text",
        "segment_tokens",
        "long_segment_tokens",
        "interval_silence_ms",
        "max_mel_tokens",
        "prompt_tokens",
        "steps",
        "full_acoustic_max_mel_tokens",
        "full_acoustic_steps",
        "cfg_rate",
        "temperature",
        "seed",
        "gpt_temperature",
        "gpt_top_k",
        "gpt_top_p",
        "gpt_repetition_penalty",
        "emotion_voice_name",
        "emotion_audio",
        "emotion_alpha",
        "emotion_vector",
        "emotion_vector_raw",
        "emotion_vector_random",
        "emotion_vector_random_seed",
        "native_flags",
        "shared_runtime_stages",
        "compact_reports",
        "require_hash_match",
        "require_code_match",
        "allow_new_cases",
        "max_rtf_regression",
        "max_elapsed_regression",
        "max_memory_regression",
        "max_throughput_regression",
        "max_runtime_peak_bytes",
        "max_runtime_command_buffers_submitted",
        "max_runtime_buffer_allocations",
        "max_runtime_buffer_bytes_allocated",
        "max_runtime_gpu_elapsed_seconds",
        "max_rtf",
        "max_elapsed_seconds",
        "min_audio_samples_per_second",
        "max_planned_scratch_capacity_bytes",
        "min_planned_scratch_reuse_saves_bytes",
        "min_planned_scratch_reuse_saves_ratio",
    ):
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if isinstance(value, tuple):
            value = list(value)
        config[key] = value
    return config


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_benchmark_args(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_reports:
        cases = []
        for item in args.from_reports:
            if "=" not in item:
                raise ValueError("--from-reports entries must be CASE=REPORT_JSON")
            case, report = item.split("=", 1)
            cases.append(_summarize_report(case, Path(report)))
    else:
        cases = [_run_case(args, case, out_dir) for case in args.cases]
    budget_checks = apply_summary_budgets(
        cases,
        max_runtime_peak_bytes=getattr(args, "max_runtime_peak_bytes", None),
        max_runtime_command_buffers_submitted=getattr(args, "max_runtime_command_buffers_submitted", None),
        max_runtime_buffer_allocations=getattr(args, "max_runtime_buffer_allocations", None),
        max_runtime_buffer_bytes_allocated=getattr(args, "max_runtime_buffer_bytes_allocated", None),
        max_runtime_gpu_elapsed_seconds=getattr(args, "max_runtime_gpu_elapsed_seconds", None),
        max_rtf=getattr(args, "max_rtf", None),
        max_elapsed_seconds=getattr(args, "max_elapsed_seconds", None),
        min_audio_samples_per_second=getattr(args, "min_audio_samples_per_second", None),
        max_planned_scratch_capacity_bytes=getattr(args, "max_planned_scratch_capacity_bytes", None),
        min_planned_scratch_reuse_saves_bytes=getattr(args, "min_planned_scratch_reuse_saves_bytes", None),
        min_planned_scratch_reuse_saves_ratio=getattr(args, "min_planned_scratch_reuse_saves_ratio", None),
    )
    passed = sum(1 for case in cases if case.get("status", "passed") == "passed")
    summary: dict[str, Any] = {
        "format": "mit2-native-hot-benchmark-summary",
        "version": 1,
        "config": _benchmark_config(args),
        "cases": cases,
        "passed": passed,
        "failed": len(cases) - passed,
    }
    if budget_checks:
        summary["budget_checks"] = budget_checks
        summary["budget_failed"] = sum(1 for check in budget_checks if check.get("status") == "failed")
    resource_comparisons = compare_summary_resource_cases(summary)
    if resource_comparisons:
        summary["resource_comparisons"] = resource_comparisons
    if args.baseline_summary:
        baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
        comparison = compare_benchmark_summaries(
            summary,
            baseline,
            max_rtf_regression=args.max_rtf_regression,
            max_elapsed_regression=args.max_elapsed_regression,
            max_memory_regression=args.max_memory_regression,
            max_throughput_regression=args.max_throughput_regression,
            require_hash_match=args.require_hash_match,
            require_code_match=args.require_code_match,
            allow_new_cases=args.allow_new_cases,
        )
        summary["comparison"] = comparison
        write_json(out_dir / "comparison.json", comparison)
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize the native hot TTS benchmark matrix.")
    add_reference_args(parser)
    parser.add_argument("--runtime", default="./build/mtts")
    parser.add_argument("--model-bundle", default="bin")
    parser.add_argument("--voice-bundle", default="artifacts/test_voice_bundle")
    parser.add_argument("--voice-name", default="voice_a7bd52e4")
    parser.add_argument(
        "--validate-voice-source",
        action="store_true",
        help="Forward source byte-exact validation to synthesize_native_hot for auto-converted cached/clone voices.",
    )
    parser.add_argument(
        "--validate-model-contract",
        action="store_true",
        help="Forward native mit2_runtime --inspect-model-bundle validation to each benchmark case.",
    )
    parser.add_argument(
        "--validate-voice-contract",
        action="store_true",
        help="Forward native mit2_runtime --inspect-voice-bundle validation to each benchmark case.",
    )
    parser.add_argument("--clone-audio", default=None)
    parser.add_argument("--clone-name", default="mit2_benchmark_clone")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for synthesize_native_hot child processes.",
    )
    parser.add_argument("--text-ids-file", default=None)
    parser.add_argument("--output-dir", default="artifacts/native_hot_benchmark")
    parser.add_argument("--cases", nargs="+", default=["short_greedy", "short_sampled", "all_segments"])
    parser.add_argument("--from-reports", nargs="*", default=None)
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--max-rtf-regression", type=float, default=0.10)
    parser.add_argument("--max-elapsed-regression", type=float, default=0.10)
    parser.add_argument("--max-memory-regression", type=float, default=0.10)
    parser.add_argument("--max-throughput-regression", type=float, default=0.10)
    parser.add_argument("--max-runtime-peak-bytes", type=int, default=None)
    parser.add_argument("--max-runtime-command-buffers-submitted", type=int, default=None)
    parser.add_argument("--max-runtime-buffer-allocations", type=int, default=None)
    parser.add_argument("--max-runtime-buffer-bytes-allocated", type=int, default=None)
    parser.add_argument("--max-runtime-gpu-elapsed-seconds", type=float, default=None)
    parser.add_argument("--max-rtf", type=float, default=None)
    parser.add_argument("--max-elapsed-seconds", type=float, default=None)
    parser.add_argument("--min-audio-samples-per-second", type=float, default=None)
    parser.add_argument("--max-planned-scratch-capacity-bytes", type=int, default=None)
    parser.add_argument("--min-planned-scratch-reuse-saves-bytes", type=int, default=None)
    parser.add_argument("--min-planned-scratch-reuse-saves-ratio", type=float, default=None)
    parser.add_argument("--require-hash-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-code-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-new-cases", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--short-text", default="你好")
    parser.add_argument("--multi-text", default="你好你好")
    parser.add_argument("--medium-text", default="你好，这是 IndexTTS2 的 Metal 中等长度基准测试。")
    parser.add_argument(
        "--native-cjk-mixed-text",
        default="AB1C2你好",
        help="Single-segment native-CJK tokenizer regression text for mixed alpha+digit CJK context.",
    )
    parser.add_argument(
        "--long-text",
        default=(
            "你好，这是 IndexTTS2 的 Metal 长文本基准测试。"
            "我们会把输入切成多个段落，逐段运行 native hot path，"
            "并用固定随机种子检查输出、内存和速度是否稳定。"
        ),
    )
    parser.add_argument("--segment-tokens", type=int, default=4)
    parser.add_argument("--long-segment-tokens", type=int, default=80)
    parser.add_argument("--interval-silence-ms", type=int, default=200)
    parser.add_argument("--max-mel-tokens", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--full-acoustic-max-mel-tokens", type=int, default=8)
    parser.add_argument("--full-acoustic-steps", type=int, default=25)
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20240605)
    parser.add_argument("--gpt-temperature", type=float, default=0.8)
    parser.add_argument("--gpt-top-k", type=int, default=30)
    parser.add_argument("--gpt-top-p", type=float, default=0.8)
    parser.add_argument("--gpt-repetition-penalty", type=float, default=10.0)
    parser.add_argument("--emotion-voice-name", default=None)
    parser.add_argument("--emotion-audio", default=None)
    parser.add_argument("--emotion-alpha", type=float, default=1.0)
    parser.add_argument(
        "--emotion-vector",
        default="0,0,0,0,0,0,0,0",
        help="8 comma-separated slider values used by the emotion_vector benchmark case.",
    )
    parser.add_argument("--emotion-vector-raw", action="store_true")
    parser.add_argument("--emotion-vector-random", action="store_true")
    parser.add_argument("--emotion-vector-random-seed", type=int, default=None)
    parser.add_argument("--native-flags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shared-runtime-stages", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compact-reports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward --compact-report to synthesize_native_hot so per-case reports omit raw runtime payloads.",
    )
    args = parser.parse_args()
    try:
        summary = run(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    if summary.get("failed", 0) > 0 or summary.get("comparison", {}).get("failed", 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
