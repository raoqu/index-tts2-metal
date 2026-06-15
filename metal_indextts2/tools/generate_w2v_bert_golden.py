from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import file_sha256, tensor_summary
from metal_indextts2.tools.common import write_json


def _write_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = str(path)
    summary["sha256"] = file_sha256(path)
    return summary


def _tensor_to_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def _read_feature_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mit2-clone-feature-prep":
        raise ValueError("feature manifest must have format=mit2-clone-feature-prep")
    if not manifest.get("ready_native_clone_audio_preprocess"):
        raise ValueError("feature manifest audio preprocess is not ready")
    return manifest


def _read_preprocessed_audio_from_manifest(manifest: dict[str, Any]) -> tuple[np.ndarray, Path, int]:
    audio_path = Path(str(manifest.get("preprocessed_output_f32", "")))
    samples = int(manifest.get("preprocessed_samples", 0))
    sample_rate = int(manifest.get("preprocessed_sample_rate", 0))
    if sample_rate != 16000:
        raise ValueError(f"preprocessed_sample_rate must be 16000, got {sample_rate}")
    if samples <= 0:
        raise ValueError("feature manifest preprocessed_samples must be positive")
    if not audio_path.exists():
        raise FileNotFoundError(f"preprocessed audio sidecar not found: {audio_path}")
    arr = np.fromfile(audio_path, dtype=np.float32)
    if arr.size != samples:
        raise ValueError(f"preprocessed audio sidecar must contain {samples} f32 values, got {arr.size}")
    expected_sha = str(manifest.get("preprocessed_output_sha256", ""))
    actual_sha = file_sha256(audio_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("preprocessed audio sidecar sha256 mismatch")
    return np.ascontiguousarray(arr, dtype=np.float32), audio_path, samples


def _load_w2v_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, dict) or "mean" not in stats or "var" not in stats:
        raise TypeError(f"{path} did not contain wav2vec2bert mean/var stats")
    mean = _tensor_to_numpy(stats["mean"], np.float32)
    var = _tensor_to_numpy(stats["var"], np.float32)
    if mean.shape != (1024,) or var.shape != (1024,):
        raise ValueError(f"wav2vec2bert stats must be [1024], got mean={mean.shape}, var={var.shape}")
    std = np.sqrt(var, dtype=np.float32)
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("wav2vec2bert std must be finite and positive")
    return mean, std


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    feature_manifest = _read_feature_manifest(feature_manifest_path)
    audio, audio_path, samples = _read_preprocessed_audio_from_manifest(feature_manifest)
    w2v_dir = Path(args.w2v_bert_dir).expanduser().resolve()
    stats_path = Path(args.w2v_stats).expanduser().resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = SeamlessM4TFeatureExtractor.from_pretrained(str(w2v_dir))
    model = Wav2Vec2BertModel.from_pretrained(str(w2v_dir))
    device = torch.device(args.device)
    model.eval()
    model.to(device)
    mean, std = _load_w2v_stats(stats_path)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    input_features_np = _tensor_to_numpy(input_features, np.float32)
    attention_mask_np = _tensor_to_numpy(attention_mask, np.uint32)
    hidden = _tensor_to_numpy(outputs.hidden_states[17], np.float32)
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 1024:
        raise ValueError(f"W2V-BERT hidden_states[17] must have shape [1,tokens,1024], got {hidden.shape}")
    spk_cond_emb = np.ascontiguousarray((hidden - mean.reshape(1, 1, 1024)) / std.reshape(1, 1, 1024), dtype=np.float32)
    tokens = int(spk_cond_emb.shape[1])

    manifest = {
        "format": "mit2-w2v-bert-semantic-golden",
        "version": 1,
        "feature_manifest": str(feature_manifest_path),
        "preprocessed_audio_f32": str(audio_path),
        "preprocessed_audio_sha256": file_sha256(audio_path),
        "preprocessed_samples": samples,
        "sample_rate": 16000,
        "w2v_bert_dir": str(w2v_dir),
        "w2v_stats": str(stats_path),
        "w2v_hidden_layer": 17,
        "spk_cond_tokens": tokens,
        "ready_reference_w2v_bert_semantic_features": True,
        "ready_native_w2v_bert_semantic_features": False,
        "next_native_boundary": "native W2V-BERT feature extractor/forward to reproduce spk_cond_emb",
        "artifacts": {
            "input_features": _write_raw(out_dir / "w2v_input_features.f32", input_features_np),
            "attention_mask": _write_raw(out_dir / "w2v_attention_mask.u32", attention_mask_np),
            "hidden_state_17": _write_raw(out_dir / "w2v_hidden_state_17.f32", hidden),
            "spk_cond_emb": _write_raw(out_dir / "spk_cond_emb.f32", spk_cond_emb),
        },
    }
    write_json(out_dir / "w2v_bert_semantic_golden.manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate W2V-BERT semantic feature golden sidecars from native clone features.")
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--w2v-bert-dir", required=True)
    parser.add_argument("--w2v-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
