from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import tensor_summary
from metal_indextts2.tools.common import add_reference_args, load_indextts2, write_json


def _as_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _write_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = path.name
    return summary


def _drop_optional_batch(value: Any, dtype: np.dtype) -> np.ndarray:
    arr = _as_numpy(value, dtype)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return np.ascontiguousarray(arr)


def _run_native_subsampling(args: argparse.Namespace, out_dir: Path, spk_cond_emb: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not args.model_bundle:
        raise ValueError("--native-conformer-stack requires --model-bundle")
    if not args.runtime:
        raise ValueError("--native-conformer-stack requires --runtime")
    input_path = out_dir / "spk_cond_emb.f32"
    stack_input_path = out_dir / "conformer_stack_input.f32"
    pos_emb_path = out_dir / "conformer_pos_emb.f32"
    mask_path = out_dir / "conditioning_mask.u32"
    spk_np = _drop_optional_batch(spk_cond_emb, np.float32)
    if spk_np.ndim != 2 or spk_np.shape[1] != 1024:
        raise ValueError(f"spk_cond_emb must have shape [tokens,1024], got {spk_np.shape}")
    _write_raw(input_path, spk_np)
    subprocess.run(
        [
            str(Path(args.runtime)),
            "--export-gpt-subsampling",
            str(Path(args.model_bundle)),
            str(input_path),
            str(stack_input_path),
            str(pos_emb_path),
            str(mask_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stack = np.fromfile(stack_input_path, dtype=np.float32)
    pos = np.fromfile(pos_emb_path, dtype=np.float32)
    mask = np.fromfile(mask_path, dtype=np.uint32)
    if mask.size == 0:
        raise ValueError("native subsampling produced empty mask")
    expected = mask.size * 512
    if stack.size != expected or pos.size != expected:
        raise ValueError(
            f"native subsampling produced stack={stack.size}, pos={pos.size}, expected {expected}"
        )
    return (
        np.ascontiguousarray(stack.reshape(mask.size, 512)),
        np.ascontiguousarray(pos.reshape(mask.size, 512)),
        np.ascontiguousarray(mask),
    )


def _run_native_cjk_text_ids(args: argparse.Namespace, out_dir: Path) -> list[int]:
    if not args.runtime:
        raise ValueError("--native-cjk-text-ids requires --runtime")
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else None
    if tokenizer_dir is None:
        if not args.model_bundle:
            raise ValueError("--native-cjk-text-ids requires --model-bundle or --tokenizer-dir")
        tokenizer_dir = Path(args.model_bundle) / "tokenizer"
    output_path = out_dir / "text_ids.native_cjk.u32"
    subprocess.run(
        [
            str(Path(args.runtime)),
            "--export-text-ids-cjk",
            str(tokenizer_dir),
            str(args.text),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text_ids = np.fromfile(output_path, dtype=np.uint32).astype(np.int64).tolist()
    if not text_ids:
        raise ValueError("native CJK text-id export produced no ids")
    return text_ids


def _run_native_emovec(
    args: argparse.Namespace,
    out_dir: Path,
    spk_cond_emb: Any,
    *,
    prefix: str = "emovec",
) -> tuple[np.ndarray, int]:
    if not args.model_bundle:
        raise ValueError("--native-emovec requires --model-bundle")
    if not args.runtime:
        raise ValueError("--native-emovec requires --runtime")
    requested_tokens = int(args.native_emovec_input_tokens)
    if requested_tokens < 0:
        raise ValueError("--native-emovec-input-tokens must be non-negative")
    input_path = out_dir / f"{prefix}_spk_cond_emb.f32"
    output_path = out_dir / f"{prefix}.native.f32"
    spk_np = _drop_optional_batch(spk_cond_emb, np.float32)
    if spk_np.ndim != 2 or spk_np.shape[1] != 1024:
        raise ValueError(f"spk_cond_emb must have shape [tokens,1024], got {spk_np.shape}")
    input_tokens = spk_np.shape[0] if requested_tokens == 0 else requested_tokens
    if input_tokens < 3:
        raise ValueError("--native-emovec-input-tokens must be 0 for full context or at least 3")
    if input_tokens > spk_np.shape[0]:
        raise ValueError(f"--native-emovec-input-tokens {input_tokens} exceeds available tokens {spk_np.shape[0]}")
    _write_raw(input_path, spk_np[:input_tokens])
    subprocess.run(
        [
            str(Path(args.runtime)),
            "--export-gpt-emovec",
            str(Path(args.model_bundle)),
            str(input_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = np.fromfile(output_path, dtype=np.float32)
    if native.size != 1280:
        raise ValueError(f"native emovec produced {native.size} floats, expected 1280")
    return np.ascontiguousarray(native), int(input_tokens)


def _validate_emovec_file_arg(path: str | Path, *, flag: str = "--emovec-file") -> Path:
    emovec_path = Path(path).expanduser().resolve()
    try:
        stat = emovec_path.stat()
    except OSError as exc:
        raise ValueError(f"{flag} does not exist: {emovec_path}") from exc
    if not emovec_path.is_file():
        raise ValueError(f"{flag} must be a file: {emovec_path}")
    expected_bytes = 1280 * np.dtype(np.float32).itemsize
    if stat.st_size != expected_bytes:
        raise ValueError(
            f"{flag} must contain exactly 1280 float32 values "
            f"({expected_bytes} bytes), got {stat.st_size} bytes: {emovec_path}"
        )
    return emovec_path


def _load_emovec_file(path: str | Path, *, flag: str = "--emovec-file") -> np.ndarray:
    emovec_path = _validate_emovec_file_arg(path, flag=flag)
    values = np.fromfile(emovec_path, dtype=np.float32)
    if values.size != 1280:
        raise ValueError(f"{flag} must contain 1280 float32 values, got {values.size}: {emovec_path}")
    return np.ascontiguousarray(values)


def _parse_emovec_mix_entry(entry: str) -> tuple[Path, float]:
    if "=" not in entry:
        raise ValueError("--emovec-mix entries must use PATH=WEIGHT")
    path_text, weight_text = entry.rsplit("=", 1)
    if not path_text:
        raise ValueError("--emovec-mix path must not be empty")
    try:
        weight = float(weight_text)
    except ValueError as exc:
        raise ValueError(f"--emovec-mix weight must be a finite float: {entry}") from exc
    if not np.isfinite(weight):
        raise ValueError(f"--emovec-mix weight must be finite: {entry}")
    return Path(path_text).expanduser().resolve(), weight


def _load_emovec_mix(entries: list[str] | tuple[str, ...]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not entries:
        raise ValueError("--emovec-mix requires at least one PATH=WEIGHT entry")
    total = np.zeros(1280, dtype=np.float64)
    components: list[dict[str, Any]] = []
    for entry in entries:
        path, weight = _parse_emovec_mix_entry(entry)
        values = _load_emovec_file(path, flag="--emovec-mix")
        total += values.astype(np.float64) * weight
        components.append({"path": str(path), "weight": weight})
    return np.ascontiguousarray(total.astype(np.float32)), components


def _validate_emovec_mix_args(entries: list[str] | tuple[str, ...]) -> None:
    if not entries:
        raise ValueError("--emovec-mix requires at least one PATH=WEIGHT entry")
    for entry in entries:
        path, _ = _parse_emovec_mix_entry(entry)
        _validate_emovec_file_arg(path, flag="--emovec-mix")


def _parse_emotion_vector(text: str) -> list[float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 8:
        raise ValueError("--emotion-vector must contain 8 comma-separated values")
    values: list[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"--emotion-vector values must be finite floats: {text}") from exc
        if not np.isfinite(value):
            raise ValueError(f"--emotion-vector values must be finite: {text}")
        values.append(value)
    return values


def _normalize_emotion_vector(values: list[float], *, apply_bias: bool = True) -> list[float]:
    out = list(values)
    if apply_bias:
        emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
        out = [value * bias for value, bias in zip(out, emo_bias)]
    emo_sum = sum(out)
    if emo_sum > 0.8:
        scale = 0.8 / emo_sum
        out = [value * scale for value in out]
    return out


def _explicit_emovec_override_count(args: argparse.Namespace) -> int:
    return sum(
        1
        for enabled in (
            bool(getattr(args, "emovec_file", None)),
            bool(getattr(args, "emovec_mix", None)),
        )
        if enabled
    )


def _emotion_source_count(args: argparse.Namespace) -> int:
    return sum(
        1
        for enabled in (
            bool(getattr(args, "emotion_audio", None)),
            bool(getattr(args, "emotion_voice_name", None)),
        )
        if enabled
    )


def _validate_emotion_args(args: argparse.Namespace) -> None:
    if _explicit_emovec_override_count(args) > 1:
        raise ValueError("--emovec-file and --emovec-mix are mutually exclusive")
    if _explicit_emovec_override_count(args) and getattr(args, "native_emovec", False):
        raise ValueError("--emovec-file/--emovec-mix and --native-emovec are mutually exclusive")
    if _explicit_emovec_override_count(args) and getattr(args, "emotion_vector", None):
        raise ValueError("--emovec-file/--emovec-mix cannot be combined with --emotion-vector")
    if _emotion_source_count(args) > 1:
        raise ValueError("--emotion-audio and --emotion-voice-name are mutually exclusive")
    if _explicit_emovec_override_count(args) and _emotion_source_count(args):
        raise ValueError("--emovec-file/--emovec-mix cannot be combined with emotion audio/profile sources")
    if getattr(args, "emotion_vector", None) and _emotion_source_count(args):
        raise ValueError("--emotion-vector cannot be combined with emotion audio/profile sources")
    emotion_alpha = float(getattr(args, "emotion_alpha", 1.0))
    if not np.isfinite(emotion_alpha):
        raise ValueError("--emotion-alpha must be finite")
    if getattr(args, "emovec_file", None):
        _validate_emovec_file_arg(args.emovec_file, flag="--emovec-file")
    if getattr(args, "emovec_mix", None):
        _validate_emovec_mix_args(args.emovec_mix)
    if getattr(args, "emotion_vector", None):
        _parse_emotion_vector(args.emotion_vector)
    if getattr(args, "emotion_vector_random_seed", None) is not None:
        if not getattr(args, "emotion_vector_random", False):
            raise ValueError("--emotion-vector-random-seed requires --emotion-vector-random")
        int(getattr(args, "emotion_vector_random_seed"))


def _most_similar_cosine_index(query: Any, matrix: Any) -> Any:
    import torch
    import torch.nn.functional as F

    similarities = F.cosine_similarity(query.float(), matrix.float(), dim=1)
    return torch.argmax(similarities)


def _emotion_vector_random_indices(emo_num: Any, random_seed: int | None = None) -> list[int]:
    import random

    rng = random.Random(random_seed) if random_seed is not None else random
    return [rng.randint(0, int(count) - 1) for count in emo_num]


def _emotion_vector_to_emovec(
    tts: Any,
    style: Any,
    values: list[float],
    *,
    use_random: bool,
    random_seed: int | None = None,
) -> Any:
    import torch

    weight_vector = torch.tensor(values, device=style.device, dtype=style.dtype)
    if use_random:
        indices = _emotion_vector_random_indices(tts.emo_num, random_seed)
    else:
        indices = [_most_similar_cosine_index(style, matrix) for matrix in tts.spk_matrix]
    matrices = [matrix[index].unsqueeze(0) for index, matrix in zip(indices, tts.emo_matrix)]
    emo_matrix = torch.cat(matrices, 0).to(device=style.device, dtype=style.dtype)
    emovec_mat = torch.sum(weight_vector.unsqueeze(1) * emo_matrix, 0).unsqueeze(0)
    return emovec_mat, [
        int(index.detach().cpu().item()) if hasattr(index, "detach") else int(index)
        for index in indices
    ]


def _run_native_perceiver(args: argparse.Namespace, out_dir: Path, context: Any, mask: Any) -> np.ndarray:
    if not args.model_bundle:
        raise ValueError("--native-perceiver requires --model-bundle")
    if not args.runtime:
        raise ValueError("--native-perceiver requires --runtime")
    context_path = out_dir / "conditioning_context.f32"
    mask_path = out_dir / "perceiver_mask.u32"
    output_path = out_dir / "speech_conditioning_latent.native.f32"
    _write_raw(context_path, _as_numpy(context.squeeze(0), np.float32))
    _write_raw(mask_path, _as_numpy(mask.squeeze(0), np.uint32))
    subprocess.run(
        [
            str(Path(args.runtime)),
            "--export-gpt-perceiver",
            str(Path(args.model_bundle)),
            str(context_path),
            str(mask_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = np.fromfile(output_path, dtype=np.float32)
    if native.size != 32 * 1280:
        raise ValueError(f"native perceiver produced {native.size} floats, expected {32 * 1280}")
    return np.ascontiguousarray(native.reshape(32, 1280))


def _run_native_conformer_stack(args: argparse.Namespace, out_dir: Path, stack_input: Any, pos_emb: Any, mask: Any) -> np.ndarray:
    if not args.model_bundle:
        raise ValueError("--native-conformer-stack requires --model-bundle")
    if not args.runtime:
        raise ValueError("--native-conformer-stack requires --runtime")
    stack_input_path = out_dir / "conformer_stack_input.f32"
    pos_emb_path = out_dir / "conformer_pos_emb.f32"
    mask_path = out_dir / "conditioning_mask.u32"
    output_path = out_dir / "conditioning_context.native_stack.f32"
    stack_np = _drop_optional_batch(stack_input, np.float32)
    pos_np = _drop_optional_batch(pos_emb, np.float32)
    mask_np = np.ascontiguousarray(_as_numpy(mask, np.uint32).reshape(-1))
    if stack_np.ndim != 2 or stack_np.shape[1] != 512:
        raise ValueError(f"conformer stack input must have shape [tokens,512], got {stack_np.shape}")
    if pos_np.shape != stack_np.shape:
        raise ValueError(f"conformer pos_emb shape {pos_np.shape} does not match stack input {stack_np.shape}")
    if mask_np.size != stack_np.shape[0]:
        raise ValueError(f"conditioning mask length {mask_np.size} does not match stack tokens {stack_np.shape[0]}")
    _write_raw(stack_input_path, stack_np)
    _write_raw(pos_emb_path, pos_np)
    _write_raw(mask_path, mask_np)
    subprocess.run(
        [
            str(Path(args.runtime)),
            "--export-gpt-conformer-stack",
            str(Path(args.model_bundle)),
            str(stack_input_path),
            str(pos_emb_path),
            str(mask_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = np.fromfile(output_path, dtype=np.float32)
    expected = stack_np.shape[0] * 512
    if native.size != expected:
        raise ValueError(f"native conformer stack produced {native.size} floats, expected {expected}")
    return np.ascontiguousarray(native.reshape(stack_np.shape[0], 512))


def _run_native_frontend_tail(
    args: argparse.Namespace,
    out_dir: Path,
    speech_conditioning_latent: Any,
    emovec: Any,
    text_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not args.model_bundle:
        raise ValueError("--native-frontend-tail requires --model-bundle")
    if not args.runtime:
        raise ValueError("--native-frontend-tail requires --runtime")
    speech_path = out_dir / "speech_conditioning_latent.f32"
    emovec_path = out_dir / "emovec.f32"
    text_path = out_dir / "text_ids.u32"
    conds_path = out_dir / "conds_latent.f32"
    fake_path = out_dir / "fake_inputs.u32"
    inputs_path = out_dir / "inputs_embeds.f32"
    mask_path = out_dir / "attention_mask.u32"
    speech_np = _drop_optional_batch(speech_conditioning_latent, np.float32)
    emovec_np = _drop_optional_batch(emovec, np.float32).reshape(-1)
    text_np = np.ascontiguousarray(np.asarray(text_ids, dtype=np.uint32))
    if speech_np.shape != (32, 1280):
        raise ValueError(f"speech_conditioning_latent must have shape [32,1280], got {speech_np.shape}")
    if emovec_np.shape != (1280,):
        raise ValueError(f"emovec must have shape [1280], got {emovec_np.shape}")
    _write_raw(speech_path, speech_np)
    _write_raw(emovec_path, emovec_np)
    _write_raw(text_path, text_np)
    subprocess.run(
        [
            str(Path(args.runtime)),
            "--export-gpt-frontend-tail",
            str(Path(args.model_bundle)),
            str(speech_path),
            str(emovec_path),
            str(text_path),
            str(conds_path),
            str(fake_path),
            str(inputs_path),
            str(mask_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    conds = np.fromfile(conds_path, dtype=np.float32)
    fake = np.fromfile(fake_path, dtype=np.uint32)
    inputs = np.fromfile(inputs_path, dtype=np.float32)
    mask = np.fromfile(mask_path, dtype=np.uint32)
    if conds.size != 34 * 1280:
        raise ValueError(f"native frontend tail produced {conds.size} cond floats, expected {34 * 1280}")
    if fake.size == 0 or mask.size != fake.size:
        raise ValueError("native frontend tail produced inconsistent fake_inputs/attention_mask")
    expected_inputs = (fake.size - 1) * 1280
    if inputs.size != expected_inputs:
        raise ValueError(f"native frontend tail produced {inputs.size} input floats, expected {expected_inputs}")
    return (
        np.ascontiguousarray(conds.reshape(34, 1280)),
        speech_np,
        text_np,
        np.ascontiguousarray(fake),
        np.ascontiguousarray(inputs.reshape(fake.size - 1, 1280)),
        np.ascontiguousarray(mask),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_emotion_args(args)

    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    profile = tts.load_voice(args.voice_name) if args.voice_name else tts._extract_voice_features(args.prompt_audio)
    profile.setdefault("voice_name", args.voice_name or Path(args.prompt_audio).stem)
    emotion_profile = None
    if getattr(args, "emotion_voice_name", None):
        if args.voice_name and args.emotion_voice_name == args.voice_name:
            emotion_profile = profile
        else:
            emotion_profile = tts.load_voice(args.emotion_voice_name)
        emotion_profile.setdefault("voice_name", args.emotion_voice_name)
        emotion_source = f"voice:{args.emotion_voice_name}"
    elif getattr(args, "emotion_audio", None):
        emotion_profile = tts._extract_voice_features(args.emotion_audio)
        emotion_profile.setdefault("voice_name", Path(args.emotion_audio).stem)
        emotion_source = str(Path(args.emotion_audio).expanduser().resolve())
    else:
        emotion_source = "speaker"

    if args.text_ids_file:
        text_ids = np.fromfile(args.text_ids_file, dtype=np.uint32).astype(np.int64).tolist()
        if not text_ids:
            raise ValueError("--text-ids-file produced no ids")
        tokens = None
        segment = None
        text_ids_source = str(Path(args.text_ids_file).resolve())
    elif args.native_cjk_text_ids:
        text_ids = _run_native_cjk_text_ids(args, out_dir)
        tokens = None
        segment = None
        text_ids_source = "native_cjk"
    else:
        tokens = tts.tokenizer.tokenize(args.text)
        segments = tts.tokenizer.split_segments(tokens, args.max_text_tokens_per_segment)
        if not segments:
            raise ValueError("text produced no tokenizer segments")
        if args.segment_index >= len(segments):
            raise ValueError(f"--segment-index {args.segment_index} out of range for {len(segments)} segments")
        segment = segments[args.segment_index]
        text_ids = tts.tokenizer.convert_tokens_to_ids(segment)
        text_ids_source = "tokenizer"
    text_tensor = torch.tensor(text_ids, dtype=torch.int32, device=tts.device).unsqueeze(0)

    spk_cond_emb = profile["spk_cond_emb"].to(tts.device)
    style = profile["s2mel_style"].to(tts.device)
    emo_cond_emb = emotion_profile["spk_cond_emb"].to(tts.device) if emotion_profile is not None else spk_cond_emb
    emotion_alpha = float(getattr(args, "emotion_alpha", 1.0))
    effective_emotion_alpha = emotion_alpha if emotion_profile is not None else 1.0
    emotion_vector_values = None
    emotion_vector_indices = None
    emotion_vector_random_seed = None
    if getattr(args, "emotion_vector", None):
        parsed_vector = _parse_emotion_vector(args.emotion_vector)
        if getattr(args, "emotion_vector_raw", False):
            emotion_vector_values = parsed_vector
        else:
            emotion_vector_values = _normalize_emotion_vector(parsed_vector, apply_bias=True)
        vector_scale = max(0.0, min(1.0, emotion_alpha))
        effective_emotion_alpha = vector_scale
        if vector_scale != 1.0:
            emotion_vector_values = [int(value * vector_scale * 10000) / 10000 for value in emotion_vector_values]
        if getattr(args, "emotion_vector_random", False) and getattr(args, "emotion_vector_random_seed", None) is not None:
            emotion_vector_random_seed = int(args.emotion_vector_random_seed)
    with torch.no_grad():
        with torch.amp.autocast(text_tensor.device.type, enabled=tts.dtype is not None, dtype=tts.dtype):
            cond_len = torch.tensor([spk_cond_emb.shape[-1]], device=text_tensor.device)
            emo_cond_len = torch.tensor([emo_cond_emb.shape[-1]], device=text_tensor.device)
            emovec_input_tokens = None
            emotion_emovec_input_tokens = None
            emovec_mix_components = None
            if getattr(args, "emovec_file", None):
                emovec_np = _load_emovec_file(args.emovec_file)
                emovec = torch.from_numpy(emovec_np).to(device=text_tensor.device, dtype=spk_cond_emb.dtype).unsqueeze(0)
                emovec_source = str(Path(args.emovec_file).expanduser().resolve())
            elif getattr(args, "emovec_mix", None):
                emovec_np, emovec_mix_components = _load_emovec_mix(args.emovec_mix)
                emovec = torch.from_numpy(emovec_np).to(device=text_tensor.device, dtype=spk_cond_emb.dtype).unsqueeze(0)
                emovec_source = "weighted_mix"
            elif args.native_emovec:
                base_emovec_np, emovec_input_tokens = _run_native_emovec(args, out_dir, spk_cond_emb)
                if emotion_profile is not None:
                    if torch.equal(spk_cond_emb, emo_cond_emb):
                        emo_emovec_np = base_emovec_np
                        emotion_emovec_input_tokens = emovec_input_tokens
                    else:
                        emo_emovec_np, emotion_emovec_input_tokens = _run_native_emovec(
                            args,
                            out_dir,
                            emo_cond_emb,
                            prefix="emovec_emotion",
                        )
                    emovec_np = base_emovec_np + np.float32(effective_emotion_alpha) * (emo_emovec_np - base_emovec_np)
                else:
                    emovec_np = base_emovec_np
                emovec = torch.from_numpy(np.ascontiguousarray(emovec_np)).to(
                    device=text_tensor.device,
                    dtype=spk_cond_emb.dtype,
                ).unsqueeze(0)
                if emotion_profile is None and emovec_input_tokens == int(spk_cond_emb.shape[1]):
                    emovec_source = "native_full_metal_subsampling_conformer_perceiver_linear"
                elif emotion_profile is None:
                    emovec_source = f"native_debug_first_{emovec_input_tokens}_metal_subsampling_conformer_perceiver_linear"
                else:
                    emovec_source = "native_emotion_merge_metal_subsampling_conformer_perceiver_linear"
            else:
                emovec_source = "pytorch" if emotion_profile is None else "pytorch_emotion_merge"
                emovec = tts.gpt.merge_emovec(
                    spk_cond_emb,
                    emo_cond_emb,
                    cond_len,
                    emo_cond_len,
                    alpha=effective_emotion_alpha,
                )
            if emotion_vector_values is not None:
                emovec_mat, emotion_vector_indices = _emotion_vector_to_emovec(
                    tts,
                    style,
                    emotion_vector_values,
                    use_random=bool(getattr(args, "emotion_vector_random", False)),
                    random_seed=emotion_vector_random_seed,
                )
                weight_sum = sum(emotion_vector_values)
                emovec = emovec_mat + (1.0 - weight_sum) * emovec
                if emovec_source.startswith("native"):
                    emovec_source = "native_emotion_vector_overlay"
                else:
                    emovec_source = "pytorch_emotion_vector_overlay"
            conformer_source = "pytorch"
            perceiver_source = "pytorch"
            subsampling_source = "pytorch"
            frontend_tail_source = "pytorch"
            if args.native_conformer_stack:
                if tts.gpt.condition_type != "conformer_perceiver":
                    raise ValueError(f"--native-conformer-stack only supports conformer_perceiver, got {tts.gpt.condition_type!r}")
                if args.native_subsampling:
                    stack_np, pos_np, mask_np = _run_native_subsampling(args, out_dir, spk_cond_emb)
                    stack_input = stack_np
                    pos_emb = pos_np
                    conditioning_mask = torch.from_numpy(mask_np.astype(np.bool_)).to(device=text_tensor.device).view(1, 1, -1)
                    subsampling_source = "native_metal_resident_conv_linear"
                else:
                    input_tokens = spk_cond_emb.shape[1]
                    input_mask = torch.ones((spk_cond_emb.shape[0], 1, input_tokens), dtype=torch.bool, device=spk_cond_emb.device)
                    stack_input, pos_emb, conditioning_mask = tts.gpt.conditioning_encoder.embed(spk_cond_emb, input_mask)
                conditioning_context_np = _run_native_conformer_stack(args, out_dir, stack_input, pos_emb, conditioning_mask)
                conditioning_context = torch.from_numpy(conditioning_context_np).to(
                    device=text_tensor.device,
                    dtype=spk_cond_emb.dtype,
                ).unsqueeze(0)
                conformer_source = "native_stack_metal_resident_attn_core_conv_ff"
            elif args.native_perceiver:
                if tts.gpt.condition_type != "conformer_perceiver":
                    raise ValueError(f"--native-perceiver only supports conformer_perceiver, got {tts.gpt.condition_type!r}")
                conditioning_context, conditioning_mask = tts.gpt.conditioning_encoder(spk_cond_emb, cond_len)
            if args.native_conformer_stack or args.native_perceiver:
                perceiver_mask = torch.cat(
                    [
                        torch.ones(
                            (conditioning_mask.shape[0], tts.gpt.cond_num),
                            dtype=torch.bool,
                            device=conditioning_mask.device,
                        ),
                        conditioning_mask.squeeze(1).bool(),
                    ],
                    dim=1,
                )
                if args.native_perceiver:
                    speech_cond_np = _run_native_perceiver(args, out_dir, conditioning_context, perceiver_mask)
                    speech_conditioning_latent = torch.from_numpy(speech_cond_np).to(
                        device=text_tensor.device,
                        dtype=conditioning_context.dtype,
                    ).unsqueeze(0)
                    perceiver_source = "native_metal_resident_linear_cross_attn_geglu_rmsnorm"
                else:
                    speech_conditioning_latent = tts.gpt.perceiver_encoder(conditioning_context, perceiver_mask)
            else:
                speech_conditioning_latent = tts.gpt.get_conditioning(
                    spk_cond_emb.transpose(1, 2),
                    cond_len,
                )
            if args.native_frontend_tail:
                conds_np, speech_cond_np, text_np, fake_inputs_np, inputs_embeds_np, attention_mask_np = _run_native_frontend_tail(
                    args,
                    out_dir,
                    speech_conditioning_latent,
                    emovec,
                    text_ids,
                )
                frontend_tail_source = "native"
            else:
                duration_index = torch.zeros(text_tensor.size(0), device=text_tensor.device, dtype=torch.long)
                duration_emb = tts.gpt.speed_emb(duration_index)
                duration_emb_half = tts.gpt.speed_emb(torch.ones_like(duration_index))
                conds_latent = torch.cat(
                    (
                        speech_conditioning_latent + emovec.unsqueeze(1),
                        duration_emb_half.unsqueeze(1),
                        duration_emb.unsqueeze(1),
                    ),
                    dim=1,
                )
                fake_inputs, inputs_embeds, attention_mask = tts.gpt.prepare_gpt_inputs(conds_latent, text_tensor)
                conds_np = _as_numpy(conds_latent.squeeze(0), np.float32)
                speech_cond_np = _as_numpy(speech_conditioning_latent.squeeze(0), np.float32)
                text_np = np.ascontiguousarray(np.asarray(text_ids, dtype=np.uint32))
                fake_inputs_np = _as_numpy(fake_inputs.squeeze(0), np.uint32)
                inputs_embeds_np = _as_numpy(inputs_embeds.squeeze(0), np.float32)
                attention_mask_np = _as_numpy(attention_mask.squeeze(0), np.uint32)

    manifest = {
        "format": "mit2-gpt-frontend",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "voice_name": profile.get("voice_name"),
        "text": args.text,
        "tokens": tokens,
        "segment_index": args.segment_index,
        "segment_tokens": segment,
        "text_ids_source": text_ids_source,
        "emovec_source": emovec_source,
        "emotion_source": emotion_source,
        "emotion_alpha": effective_emotion_alpha,
        "emotion_vector": emotion_vector_values,
        "emotion_vector_indices": emotion_vector_indices,
        "emotion_vector_random": bool(getattr(args, "emotion_vector_random", False)) if emotion_vector_values is not None else None,
        "emotion_vector_random_seed": emotion_vector_random_seed,
        "subsampling_source": subsampling_source,
        "conformer_source": conformer_source,
        "perceiver_source": perceiver_source,
        "frontend_tail_source": frontend_tail_source,
        "text_ids": text_ids,
        "generation": {
            "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
            "max_mel_tokens": args.max_mel_tokens,
            "native_emovec_requested_tokens": int(args.native_emovec_input_tokens) if args.native_emovec else None,
            "native_emovec_effective_tokens": emovec_input_tokens,
            "native_emotion_emovec_effective_tokens": emotion_emovec_input_tokens,
            "emovec_mix": emovec_mix_components,
            "emotion_vector_random": bool(getattr(args, "emotion_vector_random", False)) if emotion_vector_values is not None else None,
            "emotion_vector_random_seed": emotion_vector_random_seed,
        },
        "tensors": {
            "conds_latent": _write_raw(out_dir / "conds_latent.f32", conds_np),
            "speech_conditioning_latent": _write_raw(out_dir / "speech_conditioning_latent.f32", speech_cond_np),
            "text_ids": _write_raw(out_dir / "text_ids.u32", text_np),
            "fake_inputs": _write_raw(out_dir / "fake_inputs.u32", fake_inputs_np),
            "inputs_embeds": _write_raw(out_dir / "inputs_embeds.f32", inputs_embeds_np),
            "attention_mask": _write_raw(out_dir / "attention_mask.u32", attention_mask_np),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate minimal GPT frontend tensors for native hot-path TTS.")
    add_reference_args(parser)
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
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--model-bundle", default=None)
    parser.add_argument("--runtime", default="./build/mtts")
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--max-mel-tokens", type=int, default=8)
    args = parser.parse_args()
    if not args.voice_name and not args.prompt_audio:
        raise SystemExit("provide --voice-name or --prompt-audio")
    if args.text_ids_file and args.native_cjk_text_ids:
        raise SystemExit("--text-ids-file and --native-cjk-text-ids are mutually exclusive")
    if args.native_subsampling and not args.native_conformer_stack:
        raise SystemExit("--native-subsampling requires --native-conformer-stack")
    try:
        _validate_emotion_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.native_emovec and args.native_emovec_input_tokens < 0:
        raise SystemExit("--native-emovec-input-tokens must be non-negative")
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
