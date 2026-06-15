from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

from metal_indextts2.bundle import load_manifest
from metal_indextts2.metrics import file_sha256
from metal_indextts2.tools import convert_voice
from metal_indextts2.tools import generate_gpt_frontend
from metal_indextts2.tools.common import add_reference_args, load_indextts2, write_json


def _parse_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(value, dict):
            out.append(value)
        idx = start + end
    return out


def _namespace(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_generation_args(args: argparse.Namespace) -> None:
    if getattr(args, "max_text_tokens_per_segment", 1) <= 0:
        raise ValueError("--max-text-tokens-per-segment must be positive")
    if getattr(args, "max_mel_tokens", 1) <= 0:
        raise ValueError("--max-mel-tokens must be positive")
    if getattr(args, "steps", 1) <= 0:
        raise ValueError("--steps must be positive")
    if not _is_finite_number(getattr(args, "cfg_rate", 0.0)) or float(args.cfg_rate) < 0.0:
        raise ValueError("--cfg-rate must be finite and non-negative")
    if not _is_finite_number(getattr(args, "temperature", 1.0)) or float(args.temperature) <= 0.0:
        raise ValueError("--temperature must be finite and positive")
    if getattr(args, "interval_silence_ms", 0) < 0:
        raise ValueError("--interval-silence-ms must be non-negative")
    if getattr(args, "native_emovec", False) and getattr(args, "native_emovec_input_tokens", 0) < 0:
        raise ValueError("--native-emovec-input-tokens must be non-negative")
    if not _is_finite_number(getattr(args, "gpt_temperature", 1.0)) or float(args.gpt_temperature) <= 0.0:
        raise ValueError("--gpt-temperature must be finite and positive")
    if getattr(args, "gpt_top_k", 0) < 0:
        raise ValueError("--gpt-top-k must be non-negative")
    if not _is_finite_number(getattr(args, "gpt_top_p", 1.0)) or not (0.0 < float(args.gpt_top_p) <= 1.0):
        raise ValueError("--gpt-top-p must be finite and in (0, 1]")
    if (
        not _is_finite_number(getattr(args, "gpt_repetition_penalty", 1.0))
        or float(args.gpt_repetition_penalty) <= 0.0
    ):
        raise ValueError("--gpt-repetition-penalty must be finite and positive")


def _run_runtime(command: list[str]) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]], float]:
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed, _parse_json_objects(completed.stdout), time.perf_counter() - started


def _runtime_summary(runtime_json: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    stage_seconds: dict[str, float] = {}
    command_buffers = 0
    buffer_allocations = 0
    buffer_bytes_allocated = 0
    gpu_elapsed_seconds = 0.0
    resident_peak_bytes: list[int] = []
    predicted_codes: list[int] | None = None
    planned_scratch: dict[str, Any] = {}
    native_phase_seconds: dict[str, float] = {}
    native_total_seconds: float | None = None
    gpt_decode: dict[str, float] = {}

    for item in runtime_json:
        if not isinstance(item, dict):
            continue
        stage = item.get("stage")
        if isinstance(stage, str):
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            elapsed = item.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                stage_seconds[stage] = stage_seconds.get(stage, 0.0) + float(elapsed)
            if stage in {"hot_tts_inputs_seeded_wav", "hot_tts_inputs_sampled_seeded_wav", "hot_tts_inputs_wav"}:
                if isinstance(elapsed, (int, float)):
                    native_total_seconds = float(elapsed)
                for phase in ("gpt", "condition", "noise", "acoustic"):
                    value = item.get(f"{phase}_seconds")
                    if isinstance(value, (int, float)):
                        native_phase_seconds[f"{phase}_seconds"] = float(value)
        value = item.get("resident_peak_bytes")
        if isinstance(value, int):
            resident_peak_bytes.append(value)
        codes = item.get("predicted_codes")
        if predicted_codes is None and isinstance(codes, list):
            predicted_codes = [int(v) for v in codes]
        if item.get("stage") == "gpt_kv_codes_inputs_export":
            for key in ("raw_codes_per_second", "seconds_per_raw_code", "codes_per_second", "seconds_per_code"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    gpt_decode[key] = float(value)
        for key, value in item.items():
            if key.endswith("command_buffers_submitted") and isinstance(value, int):
                command_buffers += value
            elif key.endswith("buffer_allocations") and isinstance(value, int):
                buffer_allocations += value
            elif key.endswith("buffer_bytes_allocated") and isinstance(value, int):
                buffer_bytes_allocated += value
            elif key.endswith("gpu_elapsed_seconds") and isinstance(value, (int, float)):
                gpu_elapsed_seconds += float(value)
            elif key.startswith("planned_scratch_"):
                planned_scratch[key] = value

    return {
        "format": "mit2-native-hot-runtime-summary",
        "stage_counts": stage_counts,
        "stage_seconds": stage_seconds,
        "native_total_seconds": native_total_seconds,
        "native_phase_seconds": native_phase_seconds,
        "resident_peak_bytes": max(resident_peak_bytes) if resident_peak_bytes else None,
        "command_buffers_submitted": command_buffers or None,
        "buffer_allocations": buffer_allocations or None,
        "buffer_bytes_allocated": buffer_bytes_allocated or None,
        "gpu_elapsed_seconds": gpu_elapsed_seconds or None,
        "predicted_codes": predicted_codes,
        "planned_scratch": planned_scratch or None,
        "gpt_decode": gpt_decode or None,
    }


_PLANNED_SCRATCH_SUM_KEYS = {
    "planned_scratch_unshared_phase_peak_total_bytes",
    "planned_scratch_reuse_saves_bytes",
    "planned_scratch_actual_codes",
    "planned_scratch_actual_generated_tokens",
    "planned_scratch_actual_total_mel_tokens",
    "planned_scratch_code_slack",
    "planned_scratch_generated_token_slack",
    "planned_scratch_total_mel_token_slack",
}

_PLANNED_SCRATCH_MAX_KEYS = {
    "planned_scratch_max_prefix_tokens",
    "planned_scratch_max_codes",
    "planned_scratch_prompt_tokens",
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
    "planned_scratch_capacity_bytes",
}

_PLANNED_SCRATCH_ALL_TRUE_KEYS = {
    "planned_scratch_ok",
    "planned_scratch_covers_actual",
}

_PLANNED_SCRATCH_SAME_KEYS = {
    "planned_scratch_alignment",
    "planned_scratch_source",
}


def _combine_planned_scratch(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    segments = [
        summary.get("planned_scratch")
        for summary in summaries
        if isinstance(summary.get("planned_scratch"), dict)
    ]
    planned_segments = [dict(item) for item in segments if isinstance(item, dict)]
    if not planned_segments:
        return None, []

    combined: dict[str, Any] = {"planned_scratch_segments": len(planned_segments)}
    for key in _PLANNED_SCRATCH_SUM_KEYS:
        values = [item.get(key) for item in planned_segments if isinstance(item.get(key), int)]
        if values:
            combined[key] = sum(int(value) for value in values)
    for key in _PLANNED_SCRATCH_MAX_KEYS:
        values = [item.get(key) for item in planned_segments if isinstance(item.get(key), int)]
        if values:
            combined[key] = max(int(value) for value in values)
    for key in _PLANNED_SCRATCH_ALL_TRUE_KEYS:
        values = [item.get(key) for item in planned_segments if isinstance(item.get(key), bool)]
        if values:
            combined[key] = len(values) == len(planned_segments) and all(values)
    for key in _PLANNED_SCRATCH_SAME_KEYS:
        values = [item.get(key) for item in planned_segments if item.get(key) is not None]
        if values:
            first = values[0]
            combined[key] = first if all(value == first for value in values) else "mixed"
    return combined, planned_segments


def _combine_runtime_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not summaries:
        return None
    stage_counts: dict[str, int] = {}
    stage_seconds: dict[str, float] = {}
    native_phase_seconds: dict[str, float] = {}
    resident_peak_bytes: list[int] = []
    command_buffers = 0
    buffer_allocations = 0
    buffer_bytes_allocated = 0
    gpu_elapsed_seconds = 0.0
    native_total_seconds = 0.0
    saw_native_total = False
    segment_predicted_codes: list[list[int]] = []

    for summary in summaries:
        counts = summary.get("stage_counts")
        if isinstance(counts, dict):
            for key, value in counts.items():
                if isinstance(key, str) and isinstance(value, int):
                    stage_counts[key] = stage_counts.get(key, 0) + value
        seconds = summary.get("stage_seconds")
        if isinstance(seconds, dict):
            for key, value in seconds.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    stage_seconds[key] = stage_seconds.get(key, 0.0) + float(value)
        phase_seconds = summary.get("native_phase_seconds")
        if isinstance(phase_seconds, dict):
            for key, value in phase_seconds.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    native_phase_seconds[key] = native_phase_seconds.get(key, 0.0) + float(value)
        value = summary.get("resident_peak_bytes")
        if isinstance(value, int):
            resident_peak_bytes.append(value)
        value = summary.get("command_buffers_submitted")
        if isinstance(value, int):
            command_buffers += value
        value = summary.get("buffer_allocations")
        if isinstance(value, int):
            buffer_allocations += value
        value = summary.get("buffer_bytes_allocated")
        if isinstance(value, int):
            buffer_bytes_allocated += value
        value = summary.get("gpu_elapsed_seconds")
        if isinstance(value, (int, float)):
            gpu_elapsed_seconds += float(value)
        value = summary.get("native_total_seconds")
        if isinstance(value, (int, float)):
            native_total_seconds += float(value)
            saw_native_total = True
        codes = summary.get("predicted_codes")
        if isinstance(codes, list):
            segment_predicted_codes.append([int(v) for v in codes])

    planned_scratch, segment_planned_scratch = _combine_planned_scratch(summaries)
    return {
        "format": "mit2-native-hot-runtime-summary",
        "segments": len(summaries),
        "stage_counts": stage_counts,
        "stage_seconds": stage_seconds,
        "native_total_seconds": native_total_seconds if saw_native_total else None,
        "native_phase_seconds": native_phase_seconds,
        "resident_peak_bytes": max(resident_peak_bytes) if resident_peak_bytes else None,
        "command_buffers_submitted": command_buffers or None,
        "buffer_allocations": buffer_allocations or None,
        "buffer_bytes_allocated": buffer_bytes_allocated or None,
        "gpu_elapsed_seconds": gpu_elapsed_seconds or None,
        "segment_predicted_codes": segment_predicted_codes or None,
        "planned_scratch": planned_scratch,
        "segment_planned_scratch": segment_planned_scratch or None,
    }


def _compact_report_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact_report_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    omitted = {"runtime_json", "runtime_stdout", "runtime_stderr"}
    return {
        key: _compact_report_payload(item)
        for key, item in value.items()
        if key not in omitted
    }


def _validate_model_contract(args: argparse.Namespace) -> dict[str, Any]:
    if not args.runtime:
        raise ValueError("--validate-model-contract requires --runtime")
    if not args.model_bundle:
        raise ValueError("--validate-model-contract requires --model-bundle")
    command = [
        str(Path(args.runtime)),
        "--inspect-model-bundle",
        str(Path(args.model_bundle)),
    ]
    completed, runtime_json, runtime_seconds = _run_runtime(command)
    contract = next((item for item in runtime_json if item.get("stage") == "model_bundle_contract"), None)
    if contract is None:
        raise ValueError("native model contract validation did not return a model_bundle_contract report")
    if not contract.get("ok"):
        raise ValueError(f"native model bundle failed contract validation: {contract}")
    return {
        "format": "mit2-native-model-contract-validation",
        "ok": True,
        "runtime_seconds": runtime_seconds,
        "runtime_command": command,
        "runtime_report": contract,
        "runtime_stdout": completed.stdout,
        "runtime_stderr": completed.stderr,
    }


def _validate_voice_contract(args: argparse.Namespace, voice_bundle: str) -> dict[str, Any]:
    if not args.runtime:
        raise ValueError("--validate-voice-contract requires --runtime")
    command = [
        str(Path(args.runtime)),
        "--inspect-voice-bundle",
        str(Path(voice_bundle)),
    ]
    completed, runtime_json, runtime_seconds = _run_runtime(command)
    contract = next((item for item in runtime_json if item.get("stage") == "voice_bundle_contract"), None)
    if contract is None:
        raise ValueError("native voice contract validation did not return a voice_bundle_contract report")
    if not contract.get("ok"):
        raise ValueError(f"native voice bundle failed contract validation: {contract}")
    return {
        "format": "mit2-native-voice-contract-validation",
        "ok": True,
        "runtime_seconds": runtime_seconds,
        "runtime_command": command,
        "runtime_report": contract,
        "runtime_stdout": completed.stdout,
        "runtime_stderr": completed.stderr,
    }


def _split_reference_segments(args: argparse.Namespace) -> list[list[str]]:
    repo = Path(args.index_tts_repo).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from indextts.utils.front import TextNormalizer, TextTokenizer

    bpe_model = Path(args.checkpoint_dir).expanduser().resolve() / "bpe.model"
    tokenizer = TextTokenizer(str(bpe_model), TextNormalizer())
    tokens = tokenizer.tokenize(args.text)
    segments = tokenizer.split_segments(tokens, args.max_text_tokens_per_segment)
    if not segments:
        raise ValueError("text produced no tokenizer segments")
    return segments


def _run_native_cjk_segments(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if not args.runtime:
        raise ValueError("--native-cjk-text-ids requires --runtime")
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else None
    if tokenizer_dir is None:
        if not args.model_bundle:
            raise ValueError("--native-cjk-text-ids requires --model-bundle or --tokenizer-dir")
        tokenizer_dir = Path(args.model_bundle) / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, runtime_json, _ = _run_runtime(
        [
            str(Path(args.runtime)),
            "--export-text-ids-cjk-segments",
            str(tokenizer_dir),
            str(args.text),
            str(args.max_text_tokens_per_segment),
            str(output_dir),
        ]
    )
    manifest = next((item for item in runtime_json if item.get("format") == "mit2-text-ids-cjk-segments"), None)
    if manifest is None:
        raise ValueError("native CJK segment export did not return a segment manifest")
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("native CJK segment export produced no segments")
    for index, segment in enumerate(segments):
        output = segment.get("output")
        if not output:
            raise ValueError(f"native CJK segment {index} did not include an output path")
        path = Path(output)
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"native CJK segment {index} output is missing or empty: {path}")
    return manifest


def _voice_prompt_tokens(voice_bundle: str) -> int:
    manifest = load_manifest(voice_bundle)
    for tensor in manifest.get("tensors", []):
        if tensor.get("name") == "s2mel_prompt":
            shape = tensor.get("shape", [])
            if len(shape) == 3 and int(shape[0]) == 1 and int(shape[2]) == 512:
                return int(shape[1])
    raise ValueError(f"voice bundle is missing s2mel_prompt [1,tokens,512]: {voice_bundle}")


def _resolve_voice_profile_path(args: argparse.Namespace, voice_name: str) -> Path:
    candidate = Path(args.index_tts_repo).expanduser().resolve() / "voices" / f"{voice_name}.pt"
    if candidate.exists():
        return candidate

    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    if not hasattr(tts, "_voice_profile_path"):
        raise FileNotFoundError(f"voice profile not found and reference loader has no _voice_profile_path: {candidate}")
    profile_path = Path(tts._voice_profile_path(voice_name))
    if not profile_path.exists():
        raise FileNotFoundError(f"voice profile not found: {profile_path}")
    return profile_path


def _prepare_voice(args: argparse.Namespace, work_dir: Path) -> tuple[str | None, str, dict[str, Any] | None, dict[str, Any] | None]:
    voice_name = args.voice_name
    voice_manifest = None
    voice_validation = None

    if args.clone_audio:
        tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
        voice_name = tts.clone_voice(
            args.clone_audio,
            voice_name=args.clone_name,
            overwrite=args.overwrite_voice,
            verbose=args.verbose,
        )

    if args.voice_bundle:
        return voice_name, str(Path(args.voice_bundle)), voice_manifest, voice_validation

    if not voice_name:
        raise ValueError("provide --voice-bundle, --voice-name, or --clone-audio")

    voice_profile = _resolve_voice_profile_path(args, voice_name)
    voice_output_dir = Path(args.voice_output_dir) if args.voice_output_dir else work_dir / "voice"
    voice_manifest = convert_voice.convert_voice(voice_profile, voice_output_dir, force_dtype=args.voice_force_dtype)
    if getattr(args, "validate_voice_source", False):
        voice_validation = convert_voice.validate_voice_bundle(voice_profile, voice_output_dir, force_dtype=args.voice_force_dtype)
        if not voice_validation["ok"]:
            raise ValueError(f"native voice bundle failed source validation: {voice_validation}")
    return voice_name, str(voice_output_dir), voice_manifest, voice_validation


def _read_wav_pcm16_mono(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as fp:
        channels = fp.getnchannels()
        sample_width = fp.getsampwidth()
        sample_rate = fp.getframerate()
        if channels != 1 or sample_width != 2:
            raise ValueError(f"expected PCM16 mono WAV, got channels={channels}, sample_width={sample_width}: {path}")
        frames = fp.readframes(fp.getnframes())
    return sample_rate, frames


def _write_wav_pcm16_mono(path: Path, sample_rate: int, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(sample_rate)
        fp.writeframes(frames)


def _concat_segment_wavs(segment_wavs: list[Path], output_wav: Path, interval_silence_ms: int) -> dict[str, Any]:
    if not segment_wavs:
        raise ValueError("no segment WAVs to concatenate")
    sample_rate = None
    chunks: list[bytes] = []
    for index, path in enumerate(segment_wavs):
        wav_sample_rate, frames = _read_wav_pcm16_mono(path)
        if sample_rate is None:
            sample_rate = wav_sample_rate
        elif sample_rate != wav_sample_rate:
            raise ValueError(f"segment sample-rate mismatch: {path} has {wav_sample_rate}, expected {sample_rate}")
        chunks.append(frames)
        if index < len(segment_wavs) - 1 and interval_silence_ms > 0:
            silence_frames = int(sample_rate * interval_silence_ms / 1000.0)
            chunks.append(b"\x00" * silence_frames * 2)
    assert sample_rate is not None
    joined = b"".join(chunks)
    _write_wav_pcm16_mono(output_wav, sample_rate, joined)
    return {
        "sample_rate": sample_rate,
        "samples": len(joined) // 2,
        "interval_silence_ms": interval_silence_ms,
        "segments": len(segment_wavs),
    }


def _run_single_segment(
    args: argparse.Namespace,
    *,
    work_dir: Path,
    output_wav: Path,
    voice_name: str | None,
    voice_bundle: str,
    voice_manifest: dict[str, Any] | None,
    voice_validation: dict[str, Any] | None,
    model_contract: dict[str, Any] | None,
    voice_contract: dict[str, Any] | None,
    prompt_tokens: int,
    segment_index: int,
    frontend_dir_name: str,
    segment_seed: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    frontend_dir = work_dir / frontend_dir_name
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    frontend_started = time.perf_counter()
    frontend_manifest = generate_gpt_frontend.run(
        _namespace(
            index_tts_repo=args.index_tts_repo,
            checkpoint_dir=args.checkpoint_dir,
            cfg_path=args.cfg_path,
            voice_name=voice_name,
            prompt_audio=None if voice_name else args.prompt_audio,
            emotion_audio=args.emotion_audio,
            emotion_voice_name=args.emotion_voice_name,
            emotion_alpha=args.emotion_alpha,
            emotion_vector=args.emotion_vector,
            emotion_vector_raw=args.emotion_vector_raw,
            emotion_vector_random=args.emotion_vector_random,
            emotion_vector_random_seed=args.emotion_vector_random_seed,
            text=args.text,
            output_dir=str(frontend_dir),
            text_ids_file=args.text_ids_file,
            native_cjk_text_ids=args.native_cjk_text_ids,
            tokenizer_dir=args.tokenizer_dir,
            native_subsampling=args.native_subsampling,
            native_emovec=args.native_emovec,
            emovec_file=args.emovec_file,
            emovec_mix=args.emovec_mix,
            native_emovec_input_tokens=args.native_emovec_input_tokens,
            native_conformer_stack=args.native_conformer_stack,
            native_perceiver=args.native_perceiver,
            native_frontend_tail=args.native_frontend_tail,
            model_bundle=args.model_bundle,
            runtime=args.runtime,
            max_text_tokens_per_segment=args.max_text_tokens_per_segment,
            segment_index=segment_index,
            max_mel_tokens=args.max_mel_tokens,
        )
    )
    frontend_seconds = time.perf_counter() - frontend_started

    codes_path = output_wav.with_name(output_wav.name + ".codes.u32")
    condition_path = output_wav.with_name(output_wav.name + ".condition.f32")
    noise_path = Path(args.noise_file) if args.noise_file else output_wav.with_name(output_wav.name + ".noise.f32")

    if args.noise_file:
        noise_summary = {"path": str(noise_path.resolve()), "source": "provided"}
        command = [
            str(Path(args.runtime)),
            "--synthesize-hot-inputs",
            str(Path(args.model_bundle)),
            str(Path(voice_bundle)),
            str(frontend_dir / "conds_latent.f32"),
            str(frontend_dir / "text_ids.u32"),
            str(args.max_mel_tokens),
            str(noise_path),
            str(prompt_tokens),
            str(args.steps),
            str(args.cfg_rate),
            str(output_wav),
        ]
    elif args.gpt_do_sample:
        effective_seed = int(args.seed if segment_seed is None else segment_seed)
        noise_summary = {
            "path": str(noise_path.resolve()),
            "source": "native_seeded",
            "dtype": "float32",
            "seed": effective_seed,
            "temperature": args.temperature,
        }
        command = [
            str(Path(args.runtime)),
            "--synthesize-hot-inputs-sampled-seeded",
            str(Path(args.model_bundle)),
            str(Path(voice_bundle)),
            str(frontend_dir / "conds_latent.f32"),
            str(frontend_dir / "text_ids.u32"),
            str(args.max_mel_tokens),
            str(effective_seed),
            str(args.gpt_temperature),
            str(args.gpt_top_k),
            str(args.gpt_top_p),
            str(args.gpt_repetition_penalty),
            str(effective_seed),
            str(args.temperature),
            str(prompt_tokens),
            str(args.steps),
            str(args.cfg_rate),
            str(output_wav),
        ]
    else:
        effective_seed = int(args.seed if segment_seed is None else segment_seed)
        noise_summary = {
            "path": str(noise_path.resolve()),
            "source": "native_seeded",
            "dtype": "float32",
            "seed": effective_seed,
            "temperature": args.temperature,
        }
        command = [
            str(Path(args.runtime)),
            "--synthesize-hot-inputs-seeded-shared" if getattr(args, "shared_runtime_stages", False) else "--synthesize-hot-inputs-seeded",
            str(Path(args.model_bundle)),
            str(Path(voice_bundle)),
            str(frontend_dir / "conds_latent.f32"),
            str(frontend_dir / "text_ids.u32"),
            str(args.max_mel_tokens),
            str(effective_seed),
            str(args.temperature),
            str(prompt_tokens),
            str(args.steps),
            str(args.cfg_rate),
            str(output_wav),
        ]
    completed, runtime_json, runtime_seconds = _run_runtime(command)
    runtime_summary = _runtime_summary(runtime_json)
    post_started = time.perf_counter()
    codes_count = codes_path.stat().st_size // 4
    if codes_count <= 0:
        raise ValueError("native GPT code export produced no codes")
    generated_tokens = int(codes_count * 1.72)
    if generated_tokens <= 0:
        raise ValueError("native target length calculation produced no generated tokens")
    total_tokens = prompt_tokens + generated_tokens
    output_wav_sha256 = file_sha256(output_wav)
    postprocess_seconds = time.perf_counter() - post_started
    elapsed_seconds = time.perf_counter() - started

    return {
        "format": "mit2-native-hot-synthesis-report",
        "version": 1,
        "elapsed_seconds": elapsed_seconds,
        "timing": {
            "frontend_seconds": frontend_seconds,
            "runtime_seconds": runtime_seconds,
            "postprocess_seconds": postprocess_seconds,
        },
        "text": args.text,
        "voice_name": voice_name,
        "clone_audio": args.clone_audio,
        "prompt_audio": args.prompt_audio,
        "model_bundle": str(Path(args.model_bundle).resolve()),
        "model_contract": model_contract,
        "voice_bundle": str(Path(voice_bundle).resolve()),
        "voice_manifest": voice_manifest,
        "voice_validation": voice_validation,
        "voice_contract": voice_contract,
        "work_dir": str(work_dir.resolve()),
        "gpt_frontend_dir": str(frontend_dir.resolve()),
        "output_wav": str(output_wav.resolve()),
        "output_wav_sha256": output_wav_sha256,
        "codes_u32": str(codes_path.resolve()),
        "condition_f32": str(condition_path.resolve()),
        "noise_f32": str(noise_path.resolve()),
        "generation": {
            "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
            "segment_index": segment_index,
            "max_mel_tokens": args.max_mel_tokens,
            "prompt_tokens": prompt_tokens,
            "steps": args.steps,
            "cfg_rate": args.cfg_rate,
            "temperature": args.temperature,
            "seed": int(args.seed if segment_seed is None else segment_seed),
            "gpt_do_sample": bool(args.gpt_do_sample),
            "shared_runtime_stages": bool(getattr(args, "shared_runtime_stages", False)),
            "gpt_temperature": args.gpt_temperature if args.gpt_do_sample else None,
            "gpt_top_k": args.gpt_top_k if args.gpt_do_sample else None,
            "gpt_top_p": args.gpt_top_p if args.gpt_do_sample else None,
            "gpt_repetition_penalty": args.gpt_repetition_penalty if args.gpt_do_sample else None,
            "emotion_vector_random": bool(getattr(args, "emotion_vector_random", False)) if args.emotion_vector else None,
            "emotion_vector_random_seed": (
                int(args.emotion_vector_random_seed)
                if args.emotion_vector and getattr(args, "emotion_vector_random_seed", None) is not None
                else None
            ),
        },
        "gpt": {
            "frontend_format": frontend_manifest.get("format"),
            "emovec_source": frontend_manifest.get("emovec_source"),
            "emotion_source": frontend_manifest.get("emotion_source"),
            "emotion_alpha": frontend_manifest.get("emotion_alpha"),
            "emotion_vector": frontend_manifest.get("emotion_vector"),
            "emotion_vector_indices": frontend_manifest.get("emotion_vector_indices"),
            "emotion_vector_random": frontend_manifest.get("emotion_vector_random"),
            "emotion_vector_random_seed": frontend_manifest.get("emotion_vector_random_seed"),
            "subsampling_source": frontend_manifest.get("subsampling_source"),
            "conformer_source": frontend_manifest.get("conformer_source"),
            "perceiver_source": frontend_manifest.get("perceiver_source"),
            "frontend_tail_source": frontend_manifest.get("frontend_tail_source"),
            "emovec_mix": frontend_manifest.get("generation", {}).get("emovec_mix"),
            "text_ids_source": frontend_manifest.get("text_ids_source"),
            "text_ids": frontend_manifest.get("text_ids"),
            "segment_tokens": frontend_manifest.get("segment_tokens"),
            "conds_latent": frontend_manifest.get("tensors", {}).get("conds_latent"),
            "speech_conditioning_latent": frontend_manifest.get("tensors", {}).get("speech_conditioning_latent"),
            "text_ids_tensor": frontend_manifest.get("tensors", {}).get("text_ids"),
            "fake_inputs": frontend_manifest.get("tensors", {}).get("fake_inputs"),
            "inputs_embeds": frontend_manifest.get("tensors", {}).get("inputs_embeds"),
            "attention_mask": frontend_manifest.get("tensors", {}).get("attention_mask"),
        },
        "s2mel": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
            "noise": noise_summary,
        },
        "runtime_command": command,
        "runtime_summary": runtime_summary,
        "runtime_json": runtime_json,
        "runtime_stdout": completed.stdout,
        "runtime_stderr": completed.stderr,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if not args.voice_name and not args.prompt_audio and not args.clone_audio:
        raise ValueError("provide --voice-name, --prompt-audio, or --clone-audio")
    if args.prompt_audio and not args.voice_bundle and not args.clone_audio:
        raise ValueError("--prompt-audio without --voice-bundle is ambiguous; use --clone-audio for auto voice conversion")
    if args.prompt_tokens is not None and args.prompt_tokens <= 0:
        raise ValueError("--prompt-tokens must be positive")
    _validate_generation_args(args)
    generate_gpt_frontend._validate_emotion_args(args)
    if args.all_segments and args.text_ids_file:
        raise ValueError("--all-segments cannot be combined with --text-ids-file")
    if args.all_segments and args.noise_file:
        raise ValueError("--all-segments cannot reuse one --noise-file across segments")
    if args.gpt_do_sample and args.noise_file:
        raise ValueError("--gpt-do-sample currently requires native seeded acoustic noise; omit --noise-file")
    if getattr(args, "shared_runtime_stages", False) and args.noise_file:
        raise ValueError("--shared-runtime-stages currently requires native seeded acoustic noise; omit --noise-file")
    if getattr(args, "shared_runtime_stages", False) and args.gpt_do_sample:
        raise ValueError("--shared-runtime-stages currently supports greedy GPT decode only")

    work_dir = Path(args.work_dir) if args.work_dir else Path(args.output_wav).with_suffix("").parent / (
        Path(args.output_wav).with_suffix("").name + ".native_hot_work"
    )
    output_wav = Path(args.output_wav)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_contract = _validate_model_contract(args) if getattr(args, "validate_model_contract", False) else None
    prepare_voice_started = time.perf_counter()
    voice_name, voice_bundle, voice_manifest, voice_validation = _prepare_voice(args, work_dir)
    voice_contract = _validate_voice_contract(args, voice_bundle) if getattr(args, "validate_voice_contract", False) else None
    prepare_voice_seconds = time.perf_counter() - prepare_voice_started
    prompt_tokens = args.prompt_tokens if args.prompt_tokens is not None else _voice_prompt_tokens(voice_bundle)

    if args.all_segments:
        split_started = time.perf_counter()
        native_cjk_segments_manifest = None
        if args.native_cjk_text_ids:
            native_cjk_segments_manifest = _run_native_cjk_segments(args, work_dir / "native_cjk_segments")
            segments = [list(item.get("pieces", [])) for item in native_cjk_segments_manifest["segments"]]
            segment_text_ids_files = [str(Path(item["output"])) for item in native_cjk_segments_manifest["segments"]]
            text_ids_source = "native_cjk_segments"
        else:
            segments = _split_reference_segments(args)
            segment_text_ids_files = [None] * len(segments)
            text_ids_source = "tokenizer"
        split_segments_seconds = time.perf_counter() - split_started
        segment_reports = []
        segment_wavs = []
        for segment_index in range(len(segments)):
            segment_wav = work_dir / "segments" / f"segment_{segment_index:03d}.wav"
            segment_args = args
            if segment_text_ids_files[segment_index] is not None:
                segment_arg_values = vars(args).copy()
                segment_arg_values["text_ids_file"] = segment_text_ids_files[segment_index]
                segment_arg_values["native_cjk_text_ids"] = False
                segment_arg_values["all_segments"] = False
                segment_arg_values["segment_index"] = segment_index
                segment_args = argparse.Namespace(**segment_arg_values)
            segment_report = _run_single_segment(
                segment_args,
                work_dir=work_dir / "segments" / f"segment_{segment_index:03d}",
                output_wav=segment_wav,
                voice_name=voice_name,
                voice_bundle=voice_bundle,
                voice_manifest=voice_manifest if segment_index == 0 else None,
                voice_validation=voice_validation if segment_index == 0 else None,
                model_contract=model_contract if segment_index == 0 else None,
                voice_contract=voice_contract if segment_index == 0 else None,
                prompt_tokens=prompt_tokens,
                segment_index=segment_index,
                frontend_dir_name="gpt_frontend",
                segment_seed=int(args.seed) + segment_index,
            )
            segment_reports.append(segment_report)
            segment_wavs.append(segment_wav)
        concat_started = time.perf_counter()
        concat = _concat_segment_wavs(segment_wavs, output_wav, args.interval_silence_ms)
        concat_seconds = time.perf_counter() - concat_started
        output_wav_sha256 = file_sha256(output_wav)
        elapsed_seconds = time.perf_counter() - started
        runtime_summaries = [item.get("runtime_summary") for item in segment_reports if isinstance(item.get("runtime_summary"), dict)]
        report: dict[str, Any] = {
            "format": "mit2-native-hot-multisegment-synthesis-report",
            "version": 1,
            "elapsed_seconds": elapsed_seconds,
            "timing": {
                "prepare_voice_seconds": prepare_voice_seconds,
                "split_segments_seconds": split_segments_seconds,
                "segment_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in segment_reports),
                "concat_seconds": concat_seconds,
            },
            "text": args.text,
            "voice_name": voice_name,
            "clone_audio": args.clone_audio,
            "prompt_audio": args.prompt_audio,
            "model_bundle": str(Path(args.model_bundle).resolve()),
            "model_contract": model_contract,
            "voice_bundle": str(Path(voice_bundle).resolve()),
            "voice_manifest": voice_manifest,
            "voice_validation": voice_validation,
            "voice_contract": voice_contract,
            "work_dir": str(work_dir.resolve()),
            "output_wav": str(output_wav.resolve()),
            "output_wav_sha256": output_wav_sha256,
            "generation": {
                "all_segments": True,
                "segments": len(segments),
                "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
                "max_mel_tokens": args.max_mel_tokens,
                "prompt_tokens": prompt_tokens,
                "steps": args.steps,
                "cfg_rate": args.cfg_rate,
                "temperature": args.temperature,
                "seed": args.seed,
                "gpt_do_sample": bool(args.gpt_do_sample),
                "shared_runtime_stages": bool(getattr(args, "shared_runtime_stages", False)),
                "gpt_temperature": args.gpt_temperature if args.gpt_do_sample else None,
                "gpt_top_k": args.gpt_top_k if args.gpt_do_sample else None,
                "gpt_top_p": args.gpt_top_p if args.gpt_do_sample else None,
                "gpt_repetition_penalty": args.gpt_repetition_penalty if args.gpt_do_sample else None,
                "interval_silence_ms": args.interval_silence_ms,
                "text_ids_source": text_ids_source,
                "emotion_vector_random": bool(getattr(args, "emotion_vector_random", False)) if args.emotion_vector else None,
                "emotion_vector_random_seed": (
                    int(args.emotion_vector_random_seed)
                    if args.emotion_vector and getattr(args, "emotion_vector_random_seed", None) is not None
                    else None
                ),
            },
            "segments": [
                {
                    "index": i,
                    "tokens": segments[i],
                    "text_ids_file": segment_text_ids_files[i],
                    "report": segment_reports[i],
                }
                for i in range(len(segments))
            ],
            "concat": concat,
            "runtime_summary": _combine_runtime_summaries(runtime_summaries),
            "runtime_summaries": runtime_summaries,
        }
        if native_cjk_segments_manifest is not None:
            report["native_cjk_segments"] = native_cjk_segments_manifest
    else:
        report = _run_single_segment(
            args,
            work_dir=work_dir,
            output_wav=output_wav,
            voice_name=voice_name,
            voice_bundle=voice_bundle,
            voice_manifest=voice_manifest,
            voice_validation=voice_validation,
            model_contract=model_contract,
            voice_contract=voice_contract,
            prompt_tokens=prompt_tokens,
            segment_index=args.segment_index,
            frontend_dir_name="gpt_frontend",
        )
        timing = dict(report.get("timing", {}))
        timing["prepare_voice_seconds"] = prepare_voice_seconds
        report["timing"] = timing
        report["elapsed_seconds"] = time.perf_counter() - started
    if getattr(args, "compact_report", False):
        report = _compact_report_payload(report)
    if args.report_json:
        write_json(args.report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate native hot-path TTS from text by building frontend fixtures and invoking the Metal runtime."
    )
    add_reference_args(parser)
    parser.add_argument("--runtime", default="./build/mtts")
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument(
        "--validate-model-contract",
        action="store_true",
        help="Run mit2_runtime --inspect-model-bundle before synthesis and include the native model contract report.",
    )
    parser.add_argument("--voice-bundle", default=None)
    parser.add_argument("--voice-output-dir", default=None)
    parser.add_argument("--voice-force-dtype", choices=("f32", "f16"), default=None)
    parser.add_argument(
        "--validate-voice-source",
        action="store_true",
        help="When auto-converting a .pt voice profile, compare the native bundle byte-exactly against the source tensors.",
    )
    parser.add_argument(
        "--validate-voice-contract",
        action="store_true",
        help="Run mit2_runtime --inspect-voice-bundle before synthesis and include the native contract report.",
    )
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--emotion-audio", default=None)
    parser.add_argument("--emotion-voice-name", default=None)
    parser.add_argument("--emotion-alpha", type=float, default=1.0)
    parser.add_argument(
        "--emotion-vector",
        default=None,
        help="Optional 8 comma-separated emotion slider values: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm.",
    )
    parser.add_argument("--emotion-vector-raw", action="store_true", help="Use --emotion-vector values without WebUI bias/sum normalization.")
    parser.add_argument("--emotion-vector-random", action="store_true", help="Use random emotion-matrix rows instead of speaker-style nearest rows.")
    parser.add_argument("--emotion-vector-random-seed", type=int, default=None, help="Seed for deterministic --emotion-vector-random prototype selection.")
    parser.add_argument("--clone-audio", default=None)
    parser.add_argument("--clone-name", default=None)
    parser.add_argument("--overwrite-voice", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument(
        "--compact-report",
        action="store_true",
        help="Omit raw runtime stdout/stderr/json payloads while preserving runtime_summary and sidecar metadata.",
    )
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--all-segments", action="store_true")
    parser.add_argument("--interval-silence-ms", type=int, default=200)
    parser.add_argument("--text-ids-file", default=None)
    parser.add_argument("--native-cjk-text-ids", action="store_true")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--native-subsampling", action="store_true")
    parser.add_argument("--native-emovec", action="store_true")
    parser.add_argument(
        "--emovec-file",
        default=None,
        help="Optional raw float32[1280] emotion vector to inject instead of computing gpt.get_emovec.",
    )
    parser.add_argument(
        "--emovec-mix",
        action="append",
        default=None,
        metavar="PATH=WEIGHT",
        help="Repeatable weighted raw float32[1280] emotion vector component. The vectors are summed as weight * vector.",
    )
    parser.add_argument(
        "--native-emovec-input-tokens",
        type=int,
        default=0,
        help="Conditioning tokens for native emovec; 0 uses the full voice conditioning sequence.",
    )
    parser.add_argument("--native-conformer-stack", action="store_true")
    parser.add_argument("--native-perceiver", action="store_true")
    parser.add_argument("--native-frontend-tail", action="store_true")
    parser.add_argument("--max-mel-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=None)
    parser.add_argument("--noise-file", default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20240605)
    parser.add_argument("--gpt-do-sample", action="store_true")
    parser.add_argument(
        "--shared-runtime-stages",
        action="store_true",
        help="Use the experimental shared-bundle greedy seeded runtime path.",
    )
    parser.add_argument("--gpt-temperature", type=float, default=0.8)
    parser.add_argument("--gpt-top-k", type=int, default=30)
    parser.add_argument("--gpt-top-p", type=float, default=0.8)
    parser.add_argument("--gpt-repetition-penalty", type=float, default=10.0)
    args = parser.parse_args()
    if args.text_ids_file and args.native_cjk_text_ids:
        raise SystemExit("--text-ids-file and --native-cjk-text-ids are mutually exclusive")
    if args.all_segments and args.text_ids_file:
        raise SystemExit("--all-segments cannot be combined with --text-ids-file")
    if args.all_segments and args.noise_file:
        raise SystemExit("--all-segments cannot reuse one --noise-file across segments")
    if args.gpt_do_sample and args.noise_file:
        raise SystemExit("--gpt-do-sample currently requires native seeded acoustic noise; omit --noise-file")
    if args.shared_runtime_stages and args.noise_file:
        raise SystemExit("--shared-runtime-stages currently requires native seeded acoustic noise; omit --noise-file")
    if args.shared_runtime_stages and args.gpt_do_sample:
        raise SystemExit("--shared-runtime-stages currently supports greedy GPT decode only")
    try:
        _validate_generation_args(args)
        generate_gpt_frontend._validate_emotion_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
