from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from metal_indextts2.bundle import normalize_array, read_tensor, sha256_bytes, write_bundle

VOICE_TENSOR_KEYS = ("spk_cond_emb", "s2mel_style", "s2mel_prompt", "mel")
VOICE_PROFILE_PT_FORMAT = "mit2-voice-profile-pt"


def load_mit2_voice_profile_pt(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != VOICE_PROFILE_PT_FORMAT:
            raise ValueError(f"unsupported MIT2 voice profile format: {manifest.get('format')}")
        tensors: dict[str, Any] = {}
        tensor_records = {record["name"]: record for record in manifest.get("tensors", [])}
        missing = [key for key in VOICE_TENSOR_KEYS if key not in tensor_records]
        if missing:
            raise KeyError(f"voice profile is missing tensors: {missing}")
        for key in VOICE_TENSOR_KEYS:
            record = tensor_records[key]
            payload = zf.read(record["file"])
            if sha256_bytes(payload) != record["sha256"]:
                raise ValueError(f"checksum mismatch for {key}")
            tensors[key] = np.load(io.BytesIO(payload), allow_pickle=False)
        metadata = manifest.get("metadata", {})
        tensors["voice_name"] = metadata.get("voice_name") or path.stem
        tensors["ref_audio_path"] = metadata.get("ref_audio_path")
        tensors["created_at"] = metadata.get("created_at")
        return tensors


def write_mit2_voice_profile_pt(
    bundle_dir: Path,
    output: Path,
    *,
    voice_name: str | None = None,
    ref_audio_path: str | None = None,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "voice_name": voice_name or output.stem,
        "ref_audio_path": ref_audio_path,
        "source_bundle": str(bundle_dir),
        "stage": "native_clone_voice_profile",
    }
    tensor_records: list[dict[str, Any]] = []
    tensor_payloads: dict[str, bytes] = {}
    for key in VOICE_TENSOR_KEYS:
        arr = np.ascontiguousarray(read_tensor(bundle_dir, key))
        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        payload = buffer.getvalue()
        tensor_file = f"tensors/{key}.npy"
        tensor_payloads[tensor_file] = payload
        tensor_records.append(
            {
                "name": key,
                "file": tensor_file,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
                "sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "format": VOICE_PROFILE_PT_FORMAT,
        "version": 1,
        "metadata": metadata,
        "tensors": tensor_records,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        for name, payload in tensor_payloads.items():
            zf.writestr(name, payload)
    return manifest


def load_voice(path: Path) -> dict[str, Any]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            if "manifest.json" in zf.namelist():
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                if manifest.get("format") == VOICE_PROFILE_PT_FORMAT:
                    return load_mit2_voice_profile_pt(path)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required to read torch.save voice profiles. "
            "MIT2 no-torch voice profile .pt files are supported without torch."
        ) from exc

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"voice profile is not a dict: {path}")
    missing = [k for k in VOICE_TENSOR_KEYS if k not in obj]
    if missing:
        raise KeyError(f"voice profile is missing tensors: {missing}")
    return obj


def convert_voice(path: Path, output: Path, *, force_dtype: str | None = None) -> dict[str, Any]:
    profile = load_voice(path)
    tensors = [(key, profile[key], "voice") for key in VOICE_TENSOR_KEYS]
    metadata = {
        "source": str(path),
        "voice_name": profile.get("voice_name") or path.stem,
        "ref_audio_path": profile.get("ref_audio_path"),
        "created_at": profile.get("created_at"),
        "stage": "converted_voice_profile",
    }
    return write_bundle(output, tensors, metadata=metadata, force_dtype=force_dtype)


def validate_voice_bundle(path: Path, bundle_dir: Path, *, force_dtype: str | None = None) -> dict[str, Any]:
    profile = load_voice(path)
    tensor_results: list[dict[str, Any]] = []
    for key in VOICE_TENSOR_KEYS:
        source = normalize_array(profile[key], force_dtype)
        converted = read_tensor(bundle_dir, key)
        source_bytes = bytes(memoryview(np.ascontiguousarray(source)).cast("B"))
        converted_bytes = bytes(memoryview(np.ascontiguousarray(converted)).cast("B"))
        result = {
            "name": key,
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "sha256": sha256_bytes(source_bytes),
            "matches": source.shape == converted.shape
            and source.dtype == converted.dtype
            and source_bytes == converted_bytes,
        }
        if not result["matches"]:
            result["converted_shape"] = list(converted.shape)
            result["converted_dtype"] = str(converted.dtype)
            result["converted_sha256"] = sha256_bytes(converted_bytes)
        tensor_results.append(result)

    ok = all(result["matches"] for result in tensor_results)
    return {
        "format": "mit2-voice-bundle-validation",
        "source": str(path),
        "bundle": str(bundle_dir),
        "force_dtype": force_dtype,
        "ok": ok,
        "tensors": tensor_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an IndexTTS2 .pt voice profile into native MIT2 format.")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-dtype", choices=("f32", "f16"), default=None)
    parser.add_argument(
        "--validate-source",
        action="store_true",
        help="Read the written bundle back and compare required tensors byte-exactly against the source profile.",
    )
    args = parser.parse_args()

    manifest = convert_voice(Path(args.voice), Path(args.output), force_dtype=args.force_dtype)
    validation = None
    if args.validate_source:
        validation = validate_voice_bundle(Path(args.voice), Path(args.output), force_dtype=args.force_dtype)
        if not validation["ok"]:
            raise SystemExit(json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=False))
    summary = {
        "output": str(Path(args.output).resolve()),
        "voice_name": manifest["metadata"].get("voice_name"),
        "tensor_count": len(manifest["tensors"]),
    }
    if validation is not None:
        summary["validation"] = validation
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
