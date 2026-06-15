from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import file_sha256, tensor_summary
from metal_indextts2.tools.common import add_index_tts_repo, write_json


def _write_raw_f32(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array, dtype=np.float32)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = str(path)
    summary["sha256"] = file_sha256(path)
    return summary


def _read_feature_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mit2-clone-feature-prep":
        raise ValueError("feature manifest must have format=mit2-clone-feature-prep")
    if not manifest.get("ready_native_clone_fbank_extraction"):
        raise ValueError("feature manifest fbank extraction is not ready")
    return manifest


def _read_fbank_from_manifest(manifest: dict[str, Any]) -> tuple[np.ndarray, Path, int]:
    fbank_path = Path(str(manifest.get("output_fbank_f32", "")))
    frames = int(manifest.get("fbank_frames", 0))
    if frames <= 0:
        raise ValueError("feature manifest fbank_frames must be positive")
    if not fbank_path.exists():
        raise FileNotFoundError(f"fbank sidecar not found: {fbank_path}")
    arr = np.fromfile(fbank_path, dtype=np.float32)
    expected = frames * 80
    if arr.size != expected:
        raise ValueError(f"fbank sidecar must contain {expected} f32 values, got {arr.size}")
    expected_sha = str(manifest.get("output_fbank_sha256", ""))
    actual_sha = file_sha256(fbank_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("fbank sidecar sha256 mismatch")
    return np.ascontiguousarray(arr.reshape(frames, 80), dtype=np.float32), fbank_path, frames


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    repo = add_index_tts_repo(args.index_tts_repo)
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    manifest = _read_feature_manifest(feature_manifest_path)
    fbank, fbank_path, frames = _read_fbank_from_manifest(manifest)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.campplus_checkpoint).expanduser().resolve()
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    fbank_tensor = torch.from_numpy(fbank).unsqueeze(0).to(device)
    with torch.no_grad():
        head_input = fbank_tensor.permute(0, 2, 1)
        head_input_4d = head_input.unsqueeze(1)
        head_conv1_bn_relu = torch.relu(model.head.bn1(model.head.conv1(head_input_4d)))
        head_layer1 = model.head.layer1(head_conv1_bn_relu)
        head_layer2 = model.head.layer2(head_layer1)
        head_conv2_bn_relu = torch.relu(model.head.bn2(model.head.conv2(head_layer2)))
        head_shape = head_conv2_bn_relu.shape
        head_output = head_conv2_bn_relu.reshape(head_shape[0], head_shape[1] * head_shape[2], head_shape[3])
        xvector_tdnn = model.xvector.tdnn(head_output)
        xvector_block1_tdnnd1 = model.xvector.block1.tdnnd1(xvector_tdnn)
        xvector_block1_after_tdnnd1 = torch.cat([xvector_tdnn, xvector_block1_tdnnd1], dim=1)
        xvector_block1_tdnnd2 = model.xvector.block1.tdnnd2(xvector_block1_after_tdnnd1)
        xvector_block1_after_tdnnd2 = torch.cat([xvector_block1_after_tdnnd1, xvector_block1_tdnnd2], dim=1)
        xvector_block1_tdnnd3 = model.xvector.block1.tdnnd3(xvector_block1_after_tdnnd2)
        xvector_block1_after_tdnnd3 = torch.cat([xvector_block1_after_tdnnd2, xvector_block1_tdnnd3], dim=1)
        xvector_block1_tdnnd4 = model.xvector.block1.tdnnd4(xvector_block1_after_tdnnd3)
        xvector_block1_after_tdnnd4 = torch.cat([xvector_block1_after_tdnnd3, xvector_block1_tdnnd4], dim=1)
        xvector_block1_tdnnd5 = model.xvector.block1.tdnnd5(xvector_block1_after_tdnnd4)
        xvector_block1_after_tdnnd5 = torch.cat([xvector_block1_after_tdnnd4, xvector_block1_tdnnd5], dim=1)
        xvector_block1_tdnnd6 = model.xvector.block1.tdnnd6(xvector_block1_after_tdnnd5)
        xvector_block1_after_tdnnd6 = torch.cat([xvector_block1_after_tdnnd5, xvector_block1_tdnnd6], dim=1)
        xvector_block1_tdnnd7 = model.xvector.block1.tdnnd7(xvector_block1_after_tdnnd6)
        xvector_block1_after_tdnnd7 = torch.cat([xvector_block1_after_tdnnd6, xvector_block1_tdnnd7], dim=1)
        xvector_block1_tdnnd8 = model.xvector.block1.tdnnd8(xvector_block1_after_tdnnd7)
        xvector_block1_after_tdnnd8 = torch.cat([xvector_block1_after_tdnnd7, xvector_block1_tdnnd8], dim=1)
        xvector_block1_tdnnd9 = model.xvector.block1.tdnnd9(xvector_block1_after_tdnnd8)
        xvector_block1_after_tdnnd9 = torch.cat([xvector_block1_after_tdnnd8, xvector_block1_tdnnd9], dim=1)
        xvector_block1_tdnnd10 = model.xvector.block1.tdnnd10(xvector_block1_after_tdnnd9)
        xvector_block1_after_tdnnd10 = torch.cat([xvector_block1_after_tdnnd9, xvector_block1_tdnnd10], dim=1)
        xvector_block1_tdnnd11 = model.xvector.block1.tdnnd11(xvector_block1_after_tdnnd10)
        xvector_block1_after_tdnnd11 = torch.cat([xvector_block1_after_tdnnd10, xvector_block1_tdnnd11], dim=1)
        xvector_block1_tdnnd12 = model.xvector.block1.tdnnd12(xvector_block1_after_tdnnd11)
        xvector_block1_after_tdnnd12 = torch.cat([xvector_block1_after_tdnnd11, xvector_block1_tdnnd12], dim=1)
        xvector_transit1 = model.xvector.transit1(xvector_block1_after_tdnnd12)
        xvector_block2_tdnnd1 = model.xvector.block2.tdnnd1(xvector_transit1)
        xvector_block2_after_tdnnd1 = torch.cat([xvector_transit1, xvector_block2_tdnnd1], dim=1)
        xvector_block2_tdnnd2 = model.xvector.block2.tdnnd2(xvector_block2_after_tdnnd1)
        xvector_block2_after_tdnnd2 = torch.cat([xvector_block2_after_tdnnd1, xvector_block2_tdnnd2], dim=1)
        xvector_block2_tdnnd3 = model.xvector.block2.tdnnd3(xvector_block2_after_tdnnd2)
        xvector_block2_after_tdnnd3 = torch.cat([xvector_block2_after_tdnnd2, xvector_block2_tdnnd3], dim=1)
        xvector_block2_tdnnd4 = model.xvector.block2.tdnnd4(xvector_block2_after_tdnnd3)
        xvector_block2_after_tdnnd4 = torch.cat([xvector_block2_after_tdnnd3, xvector_block2_tdnnd4], dim=1)
        xvector_block2_tdnnd5 = model.xvector.block2.tdnnd5(xvector_block2_after_tdnnd4)
        xvector_block2_after_tdnnd5 = torch.cat([xvector_block2_after_tdnnd4, xvector_block2_tdnnd5], dim=1)
        xvector_block2_tdnnd6 = model.xvector.block2.tdnnd6(xvector_block2_after_tdnnd5)
        xvector_block2_after_tdnnd6 = torch.cat([xvector_block2_after_tdnnd5, xvector_block2_tdnnd6], dim=1)
        xvector_block2_tdnnd7 = model.xvector.block2.tdnnd7(xvector_block2_after_tdnnd6)
        xvector_block2_after_tdnnd7 = torch.cat([xvector_block2_after_tdnnd6, xvector_block2_tdnnd7], dim=1)
        xvector_block2_tdnnd8 = model.xvector.block2.tdnnd8(xvector_block2_after_tdnnd7)
        xvector_block2_after_tdnnd8 = torch.cat([xvector_block2_after_tdnnd7, xvector_block2_tdnnd8], dim=1)
        xvector_block2_tdnnd9 = model.xvector.block2.tdnnd9(xvector_block2_after_tdnnd8)
        xvector_block2_after_tdnnd9 = torch.cat([xvector_block2_after_tdnnd8, xvector_block2_tdnnd9], dim=1)
        xvector_block2_tdnnd10 = model.xvector.block2.tdnnd10(xvector_block2_after_tdnnd9)
        xvector_block2_after_tdnnd10 = torch.cat([xvector_block2_after_tdnnd9, xvector_block2_tdnnd10], dim=1)
        xvector_block2_tdnnd11 = model.xvector.block2.tdnnd11(xvector_block2_after_tdnnd10)
        xvector_block2_after_tdnnd11 = torch.cat([xvector_block2_after_tdnnd10, xvector_block2_tdnnd11], dim=1)
        xvector_block2_tdnnd12 = model.xvector.block2.tdnnd12(xvector_block2_after_tdnnd11)
        xvector_block2_after_tdnnd12 = torch.cat([xvector_block2_after_tdnnd11, xvector_block2_tdnnd12], dim=1)
        xvector_block2_tdnnd13 = model.xvector.block2.tdnnd13(xvector_block2_after_tdnnd12)
        xvector_block2_after_tdnnd13 = torch.cat([xvector_block2_after_tdnnd12, xvector_block2_tdnnd13], dim=1)
        xvector_block2_tdnnd14 = model.xvector.block2.tdnnd14(xvector_block2_after_tdnnd13)
        xvector_block2_after_tdnnd14 = torch.cat([xvector_block2_after_tdnnd13, xvector_block2_tdnnd14], dim=1)
        xvector_block2_tdnnd15 = model.xvector.block2.tdnnd15(xvector_block2_after_tdnnd14)
        xvector_block2_after_tdnnd15 = torch.cat([xvector_block2_after_tdnnd14, xvector_block2_tdnnd15], dim=1)
        xvector_block2_tdnnd16 = model.xvector.block2.tdnnd16(xvector_block2_after_tdnnd15)
        xvector_block2_after_tdnnd16 = torch.cat([xvector_block2_after_tdnnd15, xvector_block2_tdnnd16], dim=1)
        xvector_block2_tdnnd17 = model.xvector.block2.tdnnd17(xvector_block2_after_tdnnd16)
        xvector_block2_after_tdnnd17 = torch.cat([xvector_block2_after_tdnnd16, xvector_block2_tdnnd17], dim=1)
        xvector_block2_tdnnd18 = model.xvector.block2.tdnnd18(xvector_block2_after_tdnnd17)
        xvector_block2_after_tdnnd18 = torch.cat([xvector_block2_after_tdnnd17, xvector_block2_tdnnd18], dim=1)
        xvector_block2_tdnnd19 = model.xvector.block2.tdnnd19(xvector_block2_after_tdnnd18)
        xvector_block2_after_tdnnd19 = torch.cat([xvector_block2_after_tdnnd18, xvector_block2_tdnnd19], dim=1)
        xvector_block2_tdnnd20 = model.xvector.block2.tdnnd20(xvector_block2_after_tdnnd19)
        xvector_block2_after_tdnnd20 = torch.cat([xvector_block2_after_tdnnd19, xvector_block2_tdnnd20], dim=1)
        xvector_block2_tdnnd21 = model.xvector.block2.tdnnd21(xvector_block2_after_tdnnd20)
        xvector_block2_after_tdnnd21 = torch.cat([xvector_block2_after_tdnnd20, xvector_block2_tdnnd21], dim=1)
        xvector_block2_tdnnd22 = model.xvector.block2.tdnnd22(xvector_block2_after_tdnnd21)
        xvector_block2_after_tdnnd22 = torch.cat([xvector_block2_after_tdnnd21, xvector_block2_tdnnd22], dim=1)
        xvector_block2_tdnnd23 = model.xvector.block2.tdnnd23(xvector_block2_after_tdnnd22)
        xvector_block2_after_tdnnd23 = torch.cat([xvector_block2_after_tdnnd22, xvector_block2_tdnnd23], dim=1)
        xvector_block2_tdnnd24 = model.xvector.block2.tdnnd24(xvector_block2_after_tdnnd23)
        xvector_block2_after_tdnnd24 = torch.cat([xvector_block2_after_tdnnd23, xvector_block2_tdnnd24], dim=1)
        xvector_transit2 = model.xvector.transit2(xvector_block2_after_tdnnd24)
        xvector_block3_tdnnd1 = model.xvector.block3.tdnnd1(xvector_transit2)
        xvector_block3_after_tdnnd1 = torch.cat([xvector_transit2, xvector_block3_tdnnd1], dim=1)
        xvector_block3_tdnnd2 = model.xvector.block3.tdnnd2(xvector_block3_after_tdnnd1)
        xvector_block3_after_tdnnd2 = torch.cat([xvector_block3_after_tdnnd1, xvector_block3_tdnnd2], dim=1)
        xvector_block3_tdnnd3 = model.xvector.block3.tdnnd3(xvector_block3_after_tdnnd2)
        xvector_block3_after_tdnnd3 = torch.cat([xvector_block3_after_tdnnd2, xvector_block3_tdnnd3], dim=1)
        xvector_block3_tdnnd4 = model.xvector.block3.tdnnd4(xvector_block3_after_tdnnd3)
        xvector_block3_after_tdnnd4 = torch.cat([xvector_block3_after_tdnnd3, xvector_block3_tdnnd4], dim=1)
        xvector_block3_tdnnd5 = model.xvector.block3.tdnnd5(xvector_block3_after_tdnnd4)
        xvector_block3_after_tdnnd5 = torch.cat([xvector_block3_after_tdnnd4, xvector_block3_tdnnd5], dim=1)
        xvector_block3_tdnnd6 = model.xvector.block3.tdnnd6(xvector_block3_after_tdnnd5)
        xvector_block3_after_tdnnd6 = torch.cat([xvector_block3_after_tdnnd5, xvector_block3_tdnnd6], dim=1)
        xvector_block3_tdnnd7 = model.xvector.block3.tdnnd7(xvector_block3_after_tdnnd6)
        xvector_block3_after_tdnnd7 = torch.cat([xvector_block3_after_tdnnd6, xvector_block3_tdnnd7], dim=1)
        xvector_block3_tdnnd8 = model.xvector.block3.tdnnd8(xvector_block3_after_tdnnd7)
        xvector_block3_after_tdnnd8 = torch.cat([xvector_block3_after_tdnnd7, xvector_block3_tdnnd8], dim=1)
        xvector_block3_tdnnd9 = model.xvector.block3.tdnnd9(xvector_block3_after_tdnnd8)
        xvector_block3_after_tdnnd9 = torch.cat([xvector_block3_after_tdnnd8, xvector_block3_tdnnd9], dim=1)
        xvector_block3_tdnnd10 = model.xvector.block3.tdnnd10(xvector_block3_after_tdnnd9)
        xvector_block3_after_tdnnd10 = torch.cat([xvector_block3_after_tdnnd9, xvector_block3_tdnnd10], dim=1)
        xvector_block3_tdnnd11 = model.xvector.block3.tdnnd11(xvector_block3_after_tdnnd10)
        xvector_block3_after_tdnnd11 = torch.cat([xvector_block3_after_tdnnd10, xvector_block3_tdnnd11], dim=1)
        xvector_block3_tdnnd12 = model.xvector.block3.tdnnd12(xvector_block3_after_tdnnd11)
        xvector_block3_after_tdnnd12 = torch.cat([xvector_block3_after_tdnnd11, xvector_block3_tdnnd12], dim=1)
        xvector_block3_tdnnd13 = model.xvector.block3.tdnnd13(xvector_block3_after_tdnnd12)
        xvector_block3_after_tdnnd13 = torch.cat([xvector_block3_after_tdnnd12, xvector_block3_tdnnd13], dim=1)
        xvector_block3_tdnnd14 = model.xvector.block3.tdnnd14(xvector_block3_after_tdnnd13)
        xvector_block3_after_tdnnd14 = torch.cat([xvector_block3_after_tdnnd13, xvector_block3_tdnnd14], dim=1)
        xvector_block3_tdnnd15 = model.xvector.block3.tdnnd15(xvector_block3_after_tdnnd14)
        xvector_block3_after_tdnnd15 = torch.cat([xvector_block3_after_tdnnd14, xvector_block3_tdnnd15], dim=1)
        xvector_block3_tdnnd16 = model.xvector.block3.tdnnd16(xvector_block3_after_tdnnd15)
        xvector_block3_after_tdnnd16 = torch.cat([xvector_block3_after_tdnnd15, xvector_block3_tdnnd16], dim=1)
        xvector_transit3 = model.xvector.transit3(xvector_block3_after_tdnnd16)
        xvector_out_nonlinear = model.xvector.out_nonlinear(xvector_transit3)
        xvector_stats = model.xvector.stats(xvector_out_nonlinear)
        xvector_dense = model.xvector.dense(xvector_stats)
        style = model(fbank_tensor).detach().cpu().numpy()
    style = np.ascontiguousarray(style.reshape(1, 192), dtype=np.float32)
    head_conv1_bn_relu_np = np.ascontiguousarray(head_conv1_bn_relu.detach().cpu().numpy(), dtype=np.float32)
    head_layer1_np = np.ascontiguousarray(head_layer1.detach().cpu().numpy(), dtype=np.float32)
    head_layer2_np = np.ascontiguousarray(head_layer2.detach().cpu().numpy(), dtype=np.float32)
    head_conv2_bn_relu_np = np.ascontiguousarray(head_conv2_bn_relu.detach().cpu().numpy(), dtype=np.float32)
    head_output_np = np.ascontiguousarray(head_output.detach().cpu().numpy(), dtype=np.float32)
    xvector_tdnn_np = np.ascontiguousarray(xvector_tdnn.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_tdnnd1_np = np.ascontiguousarray(xvector_block1_tdnnd1.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd1_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd1.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd2_np = np.ascontiguousarray(xvector_block1_tdnnd2.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd2_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd2.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd3_np = np.ascontiguousarray(xvector_block1_tdnnd3.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd3_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd3.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd4_np = np.ascontiguousarray(xvector_block1_tdnnd4.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd4_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd4.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd5_np = np.ascontiguousarray(xvector_block1_tdnnd5.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd5_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd5.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd6_np = np.ascontiguousarray(xvector_block1_tdnnd6.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd6_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd6.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd7_np = np.ascontiguousarray(xvector_block1_tdnnd7.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd7_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd7.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd8_np = np.ascontiguousarray(xvector_block1_tdnnd8.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd8_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd8.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd9_np = np.ascontiguousarray(xvector_block1_tdnnd9.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd9_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd9.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd10_np = np.ascontiguousarray(xvector_block1_tdnnd10.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd10_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd10.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd11_np = np.ascontiguousarray(xvector_block1_tdnnd11.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd11_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd11.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block1_tdnnd12_np = np.ascontiguousarray(xvector_block1_tdnnd12.detach().cpu().numpy(), dtype=np.float32)
    xvector_block1_after_tdnnd12_np = np.ascontiguousarray(
        xvector_block1_after_tdnnd12.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_transit1_np = np.ascontiguousarray(xvector_transit1.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_tdnnd1_np = np.ascontiguousarray(xvector_block2_tdnnd1.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd1_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd1.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd2_np = np.ascontiguousarray(xvector_block2_tdnnd2.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd2_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd2.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd3_np = np.ascontiguousarray(xvector_block2_tdnnd3.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd3_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd3.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd4_np = np.ascontiguousarray(xvector_block2_tdnnd4.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd4_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd4.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd5_np = np.ascontiguousarray(xvector_block2_tdnnd5.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd5_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd5.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd6_np = np.ascontiguousarray(xvector_block2_tdnnd6.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd6_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd6.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd7_np = np.ascontiguousarray(xvector_block2_tdnnd7.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd7_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd7.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd8_np = np.ascontiguousarray(xvector_block2_tdnnd8.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd8_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd8.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd9_np = np.ascontiguousarray(xvector_block2_tdnnd9.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd9_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd9.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd10_np = np.ascontiguousarray(xvector_block2_tdnnd10.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd10_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd10.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd11_np = np.ascontiguousarray(xvector_block2_tdnnd11.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd11_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd11.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd12_np = np.ascontiguousarray(xvector_block2_tdnnd12.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd12_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd12.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd13_np = np.ascontiguousarray(xvector_block2_tdnnd13.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd13_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd13.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd14_np = np.ascontiguousarray(xvector_block2_tdnnd14.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd14_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd14.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd15_np = np.ascontiguousarray(xvector_block2_tdnnd15.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd15_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd15.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd16_np = np.ascontiguousarray(xvector_block2_tdnnd16.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd16_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd16.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd17_np = np.ascontiguousarray(xvector_block2_tdnnd17.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd17_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd17.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd18_np = np.ascontiguousarray(xvector_block2_tdnnd18.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd18_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd18.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd19_np = np.ascontiguousarray(xvector_block2_tdnnd19.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd19_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd19.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd20_np = np.ascontiguousarray(xvector_block2_tdnnd20.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd20_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd20.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd21_np = np.ascontiguousarray(xvector_block2_tdnnd21.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd21_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd21.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd22_np = np.ascontiguousarray(xvector_block2_tdnnd22.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd22_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd22.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd23_np = np.ascontiguousarray(xvector_block2_tdnnd23.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd23_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd23.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block2_tdnnd24_np = np.ascontiguousarray(xvector_block2_tdnnd24.detach().cpu().numpy(), dtype=np.float32)
    xvector_block2_after_tdnnd24_np = np.ascontiguousarray(
        xvector_block2_after_tdnnd24.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_transit2_np = np.ascontiguousarray(xvector_transit2.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_tdnnd1_np = np.ascontiguousarray(xvector_block3_tdnnd1.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd1_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd1.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd2_np = np.ascontiguousarray(xvector_block3_tdnnd2.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd2_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd2.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd3_np = np.ascontiguousarray(xvector_block3_tdnnd3.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd3_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd3.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd4_np = np.ascontiguousarray(xvector_block3_tdnnd4.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd4_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd4.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd5_np = np.ascontiguousarray(xvector_block3_tdnnd5.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd5_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd5.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd6_np = np.ascontiguousarray(xvector_block3_tdnnd6.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd6_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd6.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd7_np = np.ascontiguousarray(xvector_block3_tdnnd7.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd7_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd7.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd8_np = np.ascontiguousarray(xvector_block3_tdnnd8.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd8_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd8.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd9_np = np.ascontiguousarray(xvector_block3_tdnnd9.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd9_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd9.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd10_np = np.ascontiguousarray(xvector_block3_tdnnd10.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd10_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd10.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd11_np = np.ascontiguousarray(xvector_block3_tdnnd11.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd11_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd11.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd12_np = np.ascontiguousarray(xvector_block3_tdnnd12.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd12_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd12.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd13_np = np.ascontiguousarray(xvector_block3_tdnnd13.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd13_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd13.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd14_np = np.ascontiguousarray(xvector_block3_tdnnd14.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd14_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd14.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd15_np = np.ascontiguousarray(xvector_block3_tdnnd15.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd15_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd15.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_block3_tdnnd16_np = np.ascontiguousarray(xvector_block3_tdnnd16.detach().cpu().numpy(), dtype=np.float32)
    xvector_block3_after_tdnnd16_np = np.ascontiguousarray(
        xvector_block3_after_tdnnd16.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_transit3_np = np.ascontiguousarray(xvector_transit3.detach().cpu().numpy(), dtype=np.float32)
    xvector_out_nonlinear_np = np.ascontiguousarray(
        xvector_out_nonlinear.detach().cpu().numpy(), dtype=np.float32
    )
    xvector_stats_np = np.ascontiguousarray(xvector_stats.detach().cpu().numpy(), dtype=np.float32)
    xvector_dense_np = np.ascontiguousarray(xvector_dense.detach().cpu().numpy(), dtype=np.float32)
    style_summary = _write_raw_f32(out_dir / "s2mel_style.f32", style)
    head_conv1_bn_relu_summary = _write_raw_f32(out_dir / "campplus_head_conv1_bn_relu.f32", head_conv1_bn_relu_np)
    head_layer1_summary = _write_raw_f32(out_dir / "campplus_head_layer1.f32", head_layer1_np)
    head_layer2_summary = _write_raw_f32(out_dir / "campplus_head_layer2.f32", head_layer2_np)
    head_conv2_bn_relu_summary = _write_raw_f32(out_dir / "campplus_head_conv2_bn_relu.f32", head_conv2_bn_relu_np)
    head_output_summary = _write_raw_f32(out_dir / "campplus_head_output.f32", head_output_np)
    xvector_tdnn_summary = _write_raw_f32(out_dir / "campplus_xvector_tdnn.f32", xvector_tdnn_np)
    xvector_block1_tdnnd1_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd1.f32", xvector_block1_tdnnd1_np
    )
    xvector_block1_after_tdnnd1_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd1.f32", xvector_block1_after_tdnnd1_np
    )
    xvector_block1_tdnnd2_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd2.f32", xvector_block1_tdnnd2_np
    )
    xvector_block1_after_tdnnd2_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd2.f32", xvector_block1_after_tdnnd2_np
    )
    xvector_block1_tdnnd3_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd3.f32", xvector_block1_tdnnd3_np
    )
    xvector_block1_after_tdnnd3_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd3.f32", xvector_block1_after_tdnnd3_np
    )
    xvector_block1_tdnnd4_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd4.f32", xvector_block1_tdnnd4_np
    )
    xvector_block1_after_tdnnd4_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd4.f32", xvector_block1_after_tdnnd4_np
    )
    xvector_block1_tdnnd5_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd5.f32", xvector_block1_tdnnd5_np
    )
    xvector_block1_after_tdnnd5_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd5.f32", xvector_block1_after_tdnnd5_np
    )
    xvector_block1_tdnnd6_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd6.f32", xvector_block1_tdnnd6_np
    )
    xvector_block1_after_tdnnd6_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd6.f32", xvector_block1_after_tdnnd6_np
    )
    xvector_block1_tdnnd7_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd7.f32", xvector_block1_tdnnd7_np
    )
    xvector_block1_after_tdnnd7_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd7.f32", xvector_block1_after_tdnnd7_np
    )
    xvector_block1_tdnnd8_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd8.f32", xvector_block1_tdnnd8_np
    )
    xvector_block1_after_tdnnd8_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd8.f32", xvector_block1_after_tdnnd8_np
    )
    xvector_block1_tdnnd9_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd9.f32", xvector_block1_tdnnd9_np
    )
    xvector_block1_after_tdnnd9_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd9.f32", xvector_block1_after_tdnnd9_np
    )
    xvector_block1_tdnnd10_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd10.f32", xvector_block1_tdnnd10_np
    )
    xvector_block1_after_tdnnd10_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd10.f32", xvector_block1_after_tdnnd10_np
    )
    xvector_block1_tdnnd11_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd11.f32", xvector_block1_tdnnd11_np
    )
    xvector_block1_after_tdnnd11_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd11.f32", xvector_block1_after_tdnnd11_np
    )
    xvector_block1_tdnnd12_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_tdnnd12.f32", xvector_block1_tdnnd12_np
    )
    xvector_block1_after_tdnnd12_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block1_after_tdnnd12.f32", xvector_block1_after_tdnnd12_np
    )
    xvector_transit1_summary = _write_raw_f32(out_dir / "campplus_xvector_transit1.f32", xvector_transit1_np)
    xvector_block2_tdnnd1_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd1.f32", xvector_block2_tdnnd1_np
    )
    xvector_block2_after_tdnnd1_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd1.f32", xvector_block2_after_tdnnd1_np
    )
    xvector_block2_tdnnd2_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd2.f32", xvector_block2_tdnnd2_np
    )
    xvector_block2_after_tdnnd2_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd2.f32", xvector_block2_after_tdnnd2_np
    )
    xvector_block2_tdnnd3_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd3.f32", xvector_block2_tdnnd3_np
    )
    xvector_block2_after_tdnnd3_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd3.f32", xvector_block2_after_tdnnd3_np
    )
    xvector_block2_tdnnd4_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd4.f32", xvector_block2_tdnnd4_np
    )
    xvector_block2_after_tdnnd4_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd4.f32", xvector_block2_after_tdnnd4_np
    )
    xvector_block2_tdnnd5_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd5.f32", xvector_block2_tdnnd5_np
    )
    xvector_block2_after_tdnnd5_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd5.f32", xvector_block2_after_tdnnd5_np
    )
    xvector_block2_tdnnd6_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd6.f32", xvector_block2_tdnnd6_np
    )
    xvector_block2_after_tdnnd6_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd6.f32", xvector_block2_after_tdnnd6_np
    )
    xvector_block2_tdnnd7_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd7.f32", xvector_block2_tdnnd7_np
    )
    xvector_block2_after_tdnnd7_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd7.f32", xvector_block2_after_tdnnd7_np
    )
    xvector_block2_tdnnd8_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd8.f32", xvector_block2_tdnnd8_np
    )
    xvector_block2_after_tdnnd8_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd8.f32", xvector_block2_after_tdnnd8_np
    )
    xvector_block2_tdnnd9_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd9.f32", xvector_block2_tdnnd9_np
    )
    xvector_block2_after_tdnnd9_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd9.f32", xvector_block2_after_tdnnd9_np
    )
    xvector_block2_tdnnd10_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd10.f32", xvector_block2_tdnnd10_np
    )
    xvector_block2_after_tdnnd10_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd10.f32", xvector_block2_after_tdnnd10_np
    )
    xvector_block2_tdnnd11_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd11.f32", xvector_block2_tdnnd11_np
    )
    xvector_block2_after_tdnnd11_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd11.f32", xvector_block2_after_tdnnd11_np
    )
    xvector_block2_tdnnd12_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd12.f32", xvector_block2_tdnnd12_np
    )
    xvector_block2_after_tdnnd12_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd12.f32", xvector_block2_after_tdnnd12_np
    )
    xvector_block2_tdnnd13_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd13.f32", xvector_block2_tdnnd13_np
    )
    xvector_block2_after_tdnnd13_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd13.f32", xvector_block2_after_tdnnd13_np
    )
    xvector_block2_tdnnd14_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd14.f32", xvector_block2_tdnnd14_np
    )
    xvector_block2_after_tdnnd14_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd14.f32", xvector_block2_after_tdnnd14_np
    )
    xvector_block2_tdnnd15_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd15.f32", xvector_block2_tdnnd15_np
    )
    xvector_block2_after_tdnnd15_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd15.f32", xvector_block2_after_tdnnd15_np
    )
    xvector_block2_tdnnd16_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd16.f32", xvector_block2_tdnnd16_np
    )
    xvector_block2_after_tdnnd16_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd16.f32", xvector_block2_after_tdnnd16_np
    )
    xvector_block2_tdnnd17_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd17.f32", xvector_block2_tdnnd17_np
    )
    xvector_block2_after_tdnnd17_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd17.f32", xvector_block2_after_tdnnd17_np
    )
    xvector_block2_tdnnd18_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd18.f32", xvector_block2_tdnnd18_np
    )
    xvector_block2_after_tdnnd18_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd18.f32", xvector_block2_after_tdnnd18_np
    )
    xvector_block2_tdnnd19_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd19.f32", xvector_block2_tdnnd19_np
    )
    xvector_block2_after_tdnnd19_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd19.f32", xvector_block2_after_tdnnd19_np
    )
    xvector_block2_tdnnd20_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd20.f32", xvector_block2_tdnnd20_np
    )
    xvector_block2_after_tdnnd20_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd20.f32", xvector_block2_after_tdnnd20_np
    )
    xvector_block2_tdnnd21_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd21.f32", xvector_block2_tdnnd21_np
    )
    xvector_block2_after_tdnnd21_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd21.f32", xvector_block2_after_tdnnd21_np
    )
    xvector_block2_tdnnd22_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd22.f32", xvector_block2_tdnnd22_np
    )
    xvector_block2_after_tdnnd22_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd22.f32", xvector_block2_after_tdnnd22_np
    )
    xvector_block2_tdnnd23_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd23.f32", xvector_block2_tdnnd23_np
    )
    xvector_block2_after_tdnnd23_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd23.f32", xvector_block2_after_tdnnd23_np
    )
    xvector_block2_tdnnd24_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_tdnnd24.f32", xvector_block2_tdnnd24_np
    )
    xvector_block2_after_tdnnd24_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block2_after_tdnnd24.f32", xvector_block2_after_tdnnd24_np
    )
    xvector_transit2_summary = _write_raw_f32(out_dir / "campplus_xvector_transit2.f32", xvector_transit2_np)
    xvector_block3_tdnnd1_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd1.f32", xvector_block3_tdnnd1_np
    )
    xvector_block3_after_tdnnd1_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd1.f32", xvector_block3_after_tdnnd1_np
    )
    xvector_block3_tdnnd2_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd2.f32", xvector_block3_tdnnd2_np
    )
    xvector_block3_after_tdnnd2_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd2.f32", xvector_block3_after_tdnnd2_np
    )
    xvector_block3_tdnnd3_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd3.f32", xvector_block3_tdnnd3_np
    )
    xvector_block3_after_tdnnd3_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd3.f32", xvector_block3_after_tdnnd3_np
    )
    xvector_block3_tdnnd4_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd4.f32", xvector_block3_tdnnd4_np
    )
    xvector_block3_after_tdnnd4_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd4.f32", xvector_block3_after_tdnnd4_np
    )
    xvector_block3_tdnnd5_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd5.f32", xvector_block3_tdnnd5_np
    )
    xvector_block3_after_tdnnd5_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd5.f32", xvector_block3_after_tdnnd5_np
    )
    xvector_block3_tdnnd6_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd6.f32", xvector_block3_tdnnd6_np
    )
    xvector_block3_after_tdnnd6_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd6.f32", xvector_block3_after_tdnnd6_np
    )
    xvector_block3_tdnnd7_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd7.f32", xvector_block3_tdnnd7_np
    )
    xvector_block3_after_tdnnd7_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd7.f32", xvector_block3_after_tdnnd7_np
    )
    xvector_block3_tdnnd8_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd8.f32", xvector_block3_tdnnd8_np
    )
    xvector_block3_after_tdnnd8_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd8.f32", xvector_block3_after_tdnnd8_np
    )
    xvector_block3_tdnnd9_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd9.f32", xvector_block3_tdnnd9_np
    )
    xvector_block3_after_tdnnd9_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd9.f32", xvector_block3_after_tdnnd9_np
    )
    xvector_block3_tdnnd10_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd10.f32", xvector_block3_tdnnd10_np
    )
    xvector_block3_after_tdnnd10_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd10.f32", xvector_block3_after_tdnnd10_np
    )
    xvector_block3_tdnnd11_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd11.f32", xvector_block3_tdnnd11_np
    )
    xvector_block3_after_tdnnd11_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd11.f32", xvector_block3_after_tdnnd11_np
    )
    xvector_block3_tdnnd12_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd12.f32", xvector_block3_tdnnd12_np
    )
    xvector_block3_after_tdnnd12_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd12.f32", xvector_block3_after_tdnnd12_np
    )
    xvector_block3_tdnnd13_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd13.f32", xvector_block3_tdnnd13_np
    )
    xvector_block3_after_tdnnd13_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd13.f32", xvector_block3_after_tdnnd13_np
    )
    xvector_block3_tdnnd14_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd14.f32", xvector_block3_tdnnd14_np
    )
    xvector_block3_after_tdnnd14_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd14.f32", xvector_block3_after_tdnnd14_np
    )
    xvector_block3_tdnnd15_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd15.f32", xvector_block3_tdnnd15_np
    )
    xvector_block3_after_tdnnd15_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd15.f32", xvector_block3_after_tdnnd15_np
    )
    xvector_block3_tdnnd16_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_tdnnd16.f32", xvector_block3_tdnnd16_np
    )
    xvector_block3_after_tdnnd16_summary = _write_raw_f32(
        out_dir / "campplus_xvector_block3_after_tdnnd16.f32", xvector_block3_after_tdnnd16_np
    )
    xvector_transit3_summary = _write_raw_f32(out_dir / "campplus_xvector_transit3.f32", xvector_transit3_np)
    xvector_out_nonlinear_summary = _write_raw_f32(
        out_dir / "campplus_xvector_out_nonlinear.f32", xvector_out_nonlinear_np
    )
    xvector_stats_summary = _write_raw_f32(out_dir / "campplus_xvector_stats.f32", xvector_stats_np)
    xvector_dense_summary = _write_raw_f32(out_dir / "campplus_xvector_dense.f32", xvector_dense_np)

    result: dict[str, Any] = {
        "format": "mit2-campplus-style-golden",
        "version": 1,
        "source_repo": str(repo),
        "feature_manifest": str(feature_manifest_path),
        "campplus_checkpoint": str(checkpoint_path),
        "device": args.device,
        "fbank": {
            "path": str(fbank_path),
            "shape": [frames, 80],
            "sha256": file_sha256(fbank_path),
            "summary": tensor_summary(fbank),
        },
        "style": style_summary,
        "campplus_head": {
            "conv1_bn_relu": head_conv1_bn_relu_summary,
            "layer1": head_layer1_summary,
            "layer2": head_layer2_summary,
            "conv2_bn_relu": head_conv2_bn_relu_summary,
            "output": head_output_summary,
        },
        "campplus_xvector": {
            "tdnn": xvector_tdnn_summary,
            "block1_tdnnd1": xvector_block1_tdnnd1_summary,
            "block1_after_tdnnd1": xvector_block1_after_tdnnd1_summary,
            "block1_tdnnd2": xvector_block1_tdnnd2_summary,
            "block1_after_tdnnd2": xvector_block1_after_tdnnd2_summary,
            "block1_tdnnd3": xvector_block1_tdnnd3_summary,
            "block1_after_tdnnd3": xvector_block1_after_tdnnd3_summary,
            "block1_tdnnd4": xvector_block1_tdnnd4_summary,
            "block1_after_tdnnd4": xvector_block1_after_tdnnd4_summary,
            "block1_tdnnd5": xvector_block1_tdnnd5_summary,
            "block1_after_tdnnd5": xvector_block1_after_tdnnd5_summary,
            "block1_tdnnd6": xvector_block1_tdnnd6_summary,
            "block1_after_tdnnd6": xvector_block1_after_tdnnd6_summary,
            "block1_tdnnd7": xvector_block1_tdnnd7_summary,
            "block1_after_tdnnd7": xvector_block1_after_tdnnd7_summary,
            "block1_tdnnd8": xvector_block1_tdnnd8_summary,
            "block1_after_tdnnd8": xvector_block1_after_tdnnd8_summary,
            "block1_tdnnd9": xvector_block1_tdnnd9_summary,
            "block1_after_tdnnd9": xvector_block1_after_tdnnd9_summary,
            "block1_tdnnd10": xvector_block1_tdnnd10_summary,
            "block1_after_tdnnd10": xvector_block1_after_tdnnd10_summary,
            "block1_tdnnd11": xvector_block1_tdnnd11_summary,
            "block1_after_tdnnd11": xvector_block1_after_tdnnd11_summary,
            "block1_tdnnd12": xvector_block1_tdnnd12_summary,
            "block1_after_tdnnd12": xvector_block1_after_tdnnd12_summary,
            "transit1": xvector_transit1_summary,
            "block2_tdnnd1": xvector_block2_tdnnd1_summary,
            "block2_after_tdnnd1": xvector_block2_after_tdnnd1_summary,
            "block2_tdnnd2": xvector_block2_tdnnd2_summary,
            "block2_after_tdnnd2": xvector_block2_after_tdnnd2_summary,
            "block2_tdnnd3": xvector_block2_tdnnd3_summary,
            "block2_after_tdnnd3": xvector_block2_after_tdnnd3_summary,
            "block2_tdnnd4": xvector_block2_tdnnd4_summary,
            "block2_after_tdnnd4": xvector_block2_after_tdnnd4_summary,
            "block2_tdnnd5": xvector_block2_tdnnd5_summary,
            "block2_after_tdnnd5": xvector_block2_after_tdnnd5_summary,
            "block2_tdnnd6": xvector_block2_tdnnd6_summary,
            "block2_after_tdnnd6": xvector_block2_after_tdnnd6_summary,
            "block2_tdnnd7": xvector_block2_tdnnd7_summary,
            "block2_after_tdnnd7": xvector_block2_after_tdnnd7_summary,
            "block2_tdnnd8": xvector_block2_tdnnd8_summary,
            "block2_after_tdnnd8": xvector_block2_after_tdnnd8_summary,
            "block2_tdnnd9": xvector_block2_tdnnd9_summary,
            "block2_after_tdnnd9": xvector_block2_after_tdnnd9_summary,
            "block2_tdnnd10": xvector_block2_tdnnd10_summary,
            "block2_after_tdnnd10": xvector_block2_after_tdnnd10_summary,
            "block2_tdnnd11": xvector_block2_tdnnd11_summary,
            "block2_after_tdnnd11": xvector_block2_after_tdnnd11_summary,
            "block2_tdnnd12": xvector_block2_tdnnd12_summary,
            "block2_after_tdnnd12": xvector_block2_after_tdnnd12_summary,
            "block2_tdnnd13": xvector_block2_tdnnd13_summary,
            "block2_after_tdnnd13": xvector_block2_after_tdnnd13_summary,
            "block2_tdnnd14": xvector_block2_tdnnd14_summary,
            "block2_after_tdnnd14": xvector_block2_after_tdnnd14_summary,
            "block2_tdnnd15": xvector_block2_tdnnd15_summary,
            "block2_after_tdnnd15": xvector_block2_after_tdnnd15_summary,
            "block2_tdnnd16": xvector_block2_tdnnd16_summary,
            "block2_after_tdnnd16": xvector_block2_after_tdnnd16_summary,
            "block2_tdnnd17": xvector_block2_tdnnd17_summary,
            "block2_after_tdnnd17": xvector_block2_after_tdnnd17_summary,
            "block2_tdnnd18": xvector_block2_tdnnd18_summary,
            "block2_after_tdnnd18": xvector_block2_after_tdnnd18_summary,
            "block2_tdnnd19": xvector_block2_tdnnd19_summary,
            "block2_after_tdnnd19": xvector_block2_after_tdnnd19_summary,
            "block2_tdnnd20": xvector_block2_tdnnd20_summary,
            "block2_after_tdnnd20": xvector_block2_after_tdnnd20_summary,
            "block2_tdnnd21": xvector_block2_tdnnd21_summary,
            "block2_after_tdnnd21": xvector_block2_after_tdnnd21_summary,
            "block2_tdnnd22": xvector_block2_tdnnd22_summary,
            "block2_after_tdnnd22": xvector_block2_after_tdnnd22_summary,
            "block2_tdnnd23": xvector_block2_tdnnd23_summary,
            "block2_after_tdnnd23": xvector_block2_after_tdnnd23_summary,
            "block2_tdnnd24": xvector_block2_tdnnd24_summary,
            "block2_after_tdnnd24": xvector_block2_after_tdnnd24_summary,
            "transit2": xvector_transit2_summary,
            "block3_tdnnd1": xvector_block3_tdnnd1_summary,
            "block3_after_tdnnd1": xvector_block3_after_tdnnd1_summary,
            "block3_tdnnd2": xvector_block3_tdnnd2_summary,
            "block3_after_tdnnd2": xvector_block3_after_tdnnd2_summary,
            "block3_tdnnd3": xvector_block3_tdnnd3_summary,
            "block3_after_tdnnd3": xvector_block3_after_tdnnd3_summary,
            "block3_tdnnd4": xvector_block3_tdnnd4_summary,
            "block3_after_tdnnd4": xvector_block3_after_tdnnd4_summary,
            "block3_tdnnd5": xvector_block3_tdnnd5_summary,
            "block3_after_tdnnd5": xvector_block3_after_tdnnd5_summary,
            "block3_tdnnd6": xvector_block3_tdnnd6_summary,
            "block3_after_tdnnd6": xvector_block3_after_tdnnd6_summary,
            "block3_tdnnd7": xvector_block3_tdnnd7_summary,
            "block3_after_tdnnd7": xvector_block3_after_tdnnd7_summary,
            "block3_tdnnd8": xvector_block3_tdnnd8_summary,
            "block3_after_tdnnd8": xvector_block3_after_tdnnd8_summary,
            "block3_tdnnd9": xvector_block3_tdnnd9_summary,
            "block3_after_tdnnd9": xvector_block3_after_tdnnd9_summary,
            "block3_tdnnd10": xvector_block3_tdnnd10_summary,
            "block3_after_tdnnd10": xvector_block3_after_tdnnd10_summary,
            "block3_tdnnd11": xvector_block3_tdnnd11_summary,
            "block3_after_tdnnd11": xvector_block3_after_tdnnd11_summary,
            "block3_tdnnd12": xvector_block3_tdnnd12_summary,
            "block3_after_tdnnd12": xvector_block3_after_tdnnd12_summary,
            "block3_tdnnd13": xvector_block3_tdnnd13_summary,
            "block3_after_tdnnd13": xvector_block3_after_tdnnd13_summary,
            "block3_tdnnd14": xvector_block3_tdnnd14_summary,
            "block3_after_tdnnd14": xvector_block3_after_tdnnd14_summary,
            "block3_tdnnd15": xvector_block3_tdnnd15_summary,
            "block3_after_tdnnd15": xvector_block3_after_tdnnd15_summary,
            "block3_tdnnd16": xvector_block3_tdnnd16_summary,
            "block3_after_tdnnd16": xvector_block3_after_tdnnd16_summary,
            "transit3": xvector_transit3_summary,
            "out_nonlinear": xvector_out_nonlinear_summary,
            "stats": xvector_stats_summary,
            "dense": xvector_dense_summary,
        },
        "ready_reference_campplus_style": True,
        "ready_reference_campplus_head": True,
        "ready_native_voice_clone": False,
        "next_native_boundary": "native clone semantic/acoustic speech encoders for voice tensor creation",
    }
    write_json(out_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PyTorch CAMPPlus style golden from native clone fbank features.")
    parser.add_argument("--index-tts-repo", default="index-tts")
    parser.add_argument(
        "--campplus-checkpoint",
        default="~/.cache/huggingface/hub/models--funasr--campplus/snapshots/81a8afba4ca420cf6f845f157d5fc1d365286821/campplus_cn_common.bin",
    )
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
