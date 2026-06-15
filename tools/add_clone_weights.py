#!/usr/bin/env python3
"""
Add W2V-BERT and CAMPPlus weights to the MIT2 model bundle so --clone can run.

Usage:
    python3 tools/add_clone_weights.py [--bundle-dir BUNDLE_DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

BUNDLE_DIR_DEFAULT = Path(__file__).parent.parent / "bin"

W2V_BERT_SAFETENSORS = (
    Path("index-tts/checkpoints/hf_cache")
    / "models--facebook--w2v-bert-2.0"
    / "snapshots"
    / "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
    / "model.safetensors"
)

CAMPPLUS_BIN = (
    Path("index-tts/checkpoints/hf_cache")
    / "models--funasr--campplus"
    / "snapshots"
    / "81a8afba4ca420cf6f845f157d5fc1d365286821"
    / "campplus_cn_common.bin"
)

ALIGNMENT = 4096


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pad_to_alignment(offset: int, align: int = ALIGNMENT) -> int:
    return math.ceil(offset / align) * align


def append_tensor(
    weights_fh,
    tensor_records: list,
    name: str,
    component: str,
    arr: np.ndarray,
    current_offset: int,
) -> int:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    data = arr.tobytes()
    nbytes = len(data)
    padded_offset = pad_to_alignment(current_offset)
    padding = padded_offset - current_offset
    if padding > 0:
        weights_fh.write(b"\x00" * padding)
    weights_fh.write(data)
    sha = sha256_bytes(data)
    tensor_records.append(
        {
            "component": component,
            "dtype": "f32",
            "layout": "row_major",
            "name": name,
            "nbytes": nbytes,
            "offset": padded_offset,
            "sha256": sha,
            "shape": list(arr.shape),
        }
    )
    return padded_offset + nbytes


def load_bundle_stats(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    weights_path = bundle_dir / manifest["weights_file"]
    tensors = {t["name"]: t for t in manifest["tensors"]}
    mean_info = tensors["wav2vec2bert_stats.mean"]
    var_info = tensors["wav2vec2bert_stats.var"]
    with open(weights_path, "rb") as fh:
        fh.seek(mean_info["offset"])
        mean_data = np.frombuffer(fh.read(mean_info["nbytes"]), dtype=np.float32).copy()
        fh.seek(var_info["offset"])
        var_data = np.frombuffer(fh.read(var_info["nbytes"]), dtype=np.float32).copy()
    std_data = np.sqrt(var_data)
    return mean_data, std_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR_DEFAULT)
    parser.add_argument(
        "--w2v-safetensors", type=Path, default=W2V_BERT_SAFETENSORS
    )
    parser.add_argument("--campplus-bin", type=Path, default=CAMPPLUS_BIN)
    args = parser.parse_args()

    bundle_dir: Path = args.bundle_dir
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    weights_path = bundle_dir / manifest["weights_file"]

    existing_names = {t["name"] for t in manifest["tensors"]}

    # Find current end of weights.bin
    last_tensor = max(manifest["tensors"], key=lambda t: t["offset"] + t["nbytes"])
    current_end = last_tensor["offset"] + last_tensor["nbytes"]
    print(f"Bundle: {bundle_dir}")
    print(f"Existing tensors: {len(manifest['tensors'])}")
    print(f"Current weights end: {current_end / 1024 / 1024:.1f} MB")

    new_records: list = []

    with open(weights_path, "ab") as fh:
        current_offset = current_end

        # ── W2V-BERT ───────────────────────────────────────────────────────
        print(f"\nExtracting W2V-BERT from {args.w2v_safetensors} ...")
        sf = safe_open(str(args.w2v_safetensors), framework="pt", device="cpu")
        sf_keys = list(sf.keys())
        added_w2v = 0

        for raw_key in sf_keys:
            bundle_name = "w2v_bert." + raw_key
            if bundle_name in existing_names:
                continue
            arr = sf.get_tensor(raw_key).float().numpy()
            current_offset = append_tensor(
                fh, new_records, bundle_name, "w2v_bert", arr, current_offset
            )
            added_w2v += 1

        # Zero-fill missing conv biases (pointwise_conv1/2 and depthwise_conv have no
        # bias stored in the safetensors file but the C++ forward pass expects them).
        for layer_idx in range(18):
            prefix = f"encoder.layers.{layer_idx}.conv_module"
            for bias_name, size in [
                ("pointwise_conv1.bias", 2048),
                ("depthwise_conv.bias", 1024),
                ("pointwise_conv2.bias", 1024),
            ]:
                raw_key = f"{prefix}.{bias_name}"
                bundle_name = f"w2v_bert.{raw_key}"
                if bundle_name in existing_names or raw_key in sf_keys:
                    continue
                # Also skip if already in new_records
                new_names = {r["name"] for r in new_records}
                if bundle_name in new_names:
                    continue
                arr = np.zeros(size, dtype=np.float32)
                current_offset = append_tensor(
                    fh, new_records, bundle_name, "w2v_bert", arr, current_offset
                )
                added_w2v += 1

        # Stats: mean and std
        print("Extracting w2v_bert.stats from existing wav2vec2bert_stats ...")
        mean_arr, std_arr = load_bundle_stats(bundle_dir)
        for stat_name, arr in [
            ("w2v_bert.stats.mean", mean_arr),
            ("w2v_bert.stats.std", std_arr),
        ]:
            if stat_name not in existing_names:
                current_offset = append_tensor(
                    fh, new_records, stat_name, "w2v_bert", arr, current_offset
                )
                added_w2v += 1

        print(f"  Added {added_w2v} W2V-BERT tensors")

        # ── CAMPPlus ───────────────────────────────────────────────────────
        print(f"\nExtracting CAMPPlus from {args.campplus_bin} ...")
        state_dict = torch.load(str(args.campplus_bin), map_location="cpu", weights_only=False)
        added_camp = 0
        for raw_key, param in state_dict.items():
            if not hasattr(param, "numpy"):
                continue  # skip non-tensor entries
            bundle_name = "campplus." + raw_key
            if bundle_name in existing_names:
                continue
            arr = param.float().detach().numpy()
            current_offset = append_tensor(
                fh, new_records, bundle_name, "campplus", arr, current_offset
            )
            added_camp += 1
        print(f"  Added {added_camp} CAMPPlus tensors")

    # Update manifest
    manifest["tensors"].extend(new_records)
    manifest["tensors"].sort(key=lambda t: t["offset"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    )

    total_new = len(new_records)
    new_size = current_offset
    print(f"\nDone. Added {total_new} new tensors.")
    print(f"Bundle now has {len(manifest['tensors'])} tensors.")
    print(f"New weights size: {new_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
