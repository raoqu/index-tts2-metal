#!/usr/bin/env python3
"""Extract W2V-BERT input features from preprocessed 16kHz audio.

This tool uses the HuggingFace SeamlessM4TFeatureExtractor to convert
16kHz mono f32 audio into w2v_input_features.f32 [1, tokens, 160]
and w2v_attention_mask.u32 [1, tokens].

Usage:
    python -m metal_indextts2.tools.extract_w2v_features \\
        --w2v-bert-dir /path/to/w2v_bert_model \\
        --preprocess-manifest clone_preprocess.manifest.json \\
        --output-dir /tmp/w2v_features

Outputs:
    w2v_input_features.f32  - float32 [1, tokens, 160]
    w2v_attention_mask.u32   - uint32 [1, tokens]
    manifest.json           - extraction metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract W2V-BERT input features from audio")
    parser.add_argument("--w2v-bert-dir", required=True, help="Path to W2V-BERT HuggingFace model directory")
    parser.add_argument("--preprocess-manifest", required=True, help="Path to clone preprocess manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for feature files")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu, mps, cuda)")
    args = parser.parse_args()

    try:
        import torch
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError:
        print("ERROR: torch and transformers required. Install with: pip install torch transformers", file=sys.stderr)
        sys.exit(1)

    w2v_dir = Path(args.w2v_bert_dir).expanduser().resolve()
    manifest_path = Path(args.preprocess_manifest).expanduser().resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read preprocess manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mit2-clone-preprocess":
        print("ERROR: manifest must have format=mit2-clone-preprocess", file=sys.stderr)
        sys.exit(1)

    audio_path = Path(str(manifest.get("preprocessed_output_f32", "")))
    sample_rate = int(manifest.get("preprocessed_sample_rate", 0))
    samples = int(manifest.get("preprocessed_samples", 0))

    if sample_rate != 16000:
        print(f"ERROR: preprocessed audio must be 16000 Hz, got {sample_rate}", file=sys.stderr)
        sys.exit(1)

    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Load audio as numpy
    audio = np.fromfile(audio_path, dtype=np.float32)
    if len(audio) != samples:
        print(f"WARNING: audio samples {len(audio)} != manifest samples {samples}")

    # Load feature extractor
    processor = SeamlessM4TFeatureExtractor.from_pretrained(str(w2v_dir))

    # Extract features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]

    # Convert to numpy
    input_features_np = input_features.detach().cpu().numpy().astype(np.float32)
    attention_mask_np = attention_mask.detach().cpu().numpy().astype(np.uint32)

    tokens = int(input_features_np.shape[1])

    # Write outputs
    features_path = out_dir / "w2v_input_features.f32"
    mask_path = out_dir / "w2v_attention_mask.u32"

    np.ascontiguousarray(input_features_np).tofile(features_path)
    np.ascontiguousarray(attention_mask_np).tofile(mask_path)

    # Write manifest
    output_manifest = {
        "format": "mit2-w2v-feature-extraction",
        "version": 1,
        "w2v_bert_dir": str(w2v_dir),
        "preprocess_manifest": str(manifest_path),
        "preprocessed_audio_f32": str(audio_path),
        "sample_rate": 16000,
        "tokens": tokens,
        "input_features_shape": list(input_features_np.shape),
        "attention_mask_shape": list(attention_mask_np.shape),
        "ready_native_w2v_feature_extraction": False,
        "tool": "python_huggingface",
        "next_native_boundary": "Implement Conv1d W2V-BERT feature encoder in C++/Metal",
        "outputs": {
            "w2v_input_features_f32": str(features_path),
            "w2v_attention_mask_u32": str(mask_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "stage": "w2v_feature_extraction",
        "ok": True,
        "tokens": tokens,
        "features_shape": list(input_features_np.shape),
        "features_path": str(features_path),
        "mask_path": str(mask_path),
    }, indent=2))


if __name__ == "__main__":
    main()
