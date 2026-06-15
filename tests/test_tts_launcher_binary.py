import json
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from metal_indextts2.bundle import write_bundle


ROOT = Path(__file__).resolve().parents[1]
MIT2_TTS = ROOT / "build" / "mtts"
MODEL_BUNDLE = ROOT / "bin"
VOICE_BUNDLE = ROOT / "artifacts" / "test_voice_bundle"


def _require_launcher_artifacts() -> None:
    if not MIT2_TTS.exists():
        pytest.skip("build/mtts is not available")
    if not (MODEL_BUNDLE / "tokenizer" / "pieces.tsv").exists():
        pytest.skip("test tokenizer bundle is not available")


def _parse_json_stream(stdout: str) -> list[dict]:
    decoder = json.JSONDecoder()
    reports = []
    pos = 0
    while pos < len(stdout):
        while pos < len(stdout) and stdout[pos].isspace():
            pos += 1
        if pos >= len(stdout):
            break
        report, pos = decoder.raw_decode(stdout, pos)
        reports.append(report)
    return reports


def _run_json(*args: str, check: bool = True) -> dict:
    _require_launcher_artifacts()
    proc = subprocess.run(
        [str(MIT2_TTS), "--verbose", *args],
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def _run_last_json(*args: str, check: bool = True) -> dict:
    _require_launcher_artifacts()
    proc = subprocess.run(
        [str(MIT2_TTS), "--verbose", *args],
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return _parse_json_stream(proc.stdout)[-1]


def _write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack("<" + "f" * len(values), *values))


def _write_campplus_contract_bundle(bundle_dir: Path) -> None:
    shortcut_identity = np.eye(32, dtype=np.float32).reshape(32, 32, 1, 1)
    write_bundle(
        bundle_dir,
        [
            ("campplus.head.conv1.weight", np.ones((32, 1, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.bn1.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.bn1.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.bn1.running_mean", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.bn1.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.conv1.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn1.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn1.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn1.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn1.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.conv2.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn2.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn2.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.bn2.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.shortcut.0.weight", shortcut_identity, "campplus"),
            ("campplus.head.layer2.0.shortcut.0.weight", np.ones((32, 32, 1, 1), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.shortcut.1.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.shortcut.1.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.shortcut.1.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.0.shortcut.1.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.conv1.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn1.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn1.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn1.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn1.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.conv2.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn2.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn2.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer1.1.bn2.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.conv1.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn1.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn1.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn1.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn1.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.conv2.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn2.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn2.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.bn2.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.shortcut.1.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.shortcut.1.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.shortcut.1.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.0.shortcut.1.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.conv1.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn1.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn1.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn1.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn1.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.conv2.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn2.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn2.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.layer2.1.bn2.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.conv2.weight", np.zeros((32, 32, 3, 3), dtype=np.float32), "campplus"),
            ("campplus.head.bn2.weight", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.head.bn2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.bn2.running_mean", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.head.bn2.running_var", np.ones((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.tdnn.linear.weight", np.ones((128, 320, 5), dtype=np.float32), "campplus"),
            ("campplus.xvector.tdnn.nonlinear.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.tdnn.nonlinear.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.tdnn.nonlinear.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.tdnn.nonlinear.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear1.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.linear1.weight", np.ones((128, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd1.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd1.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd1.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.weight", np.ones((160,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.bias", np.zeros((160,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.running_mean", np.zeros((160,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear1.batchnorm.running_var", np.ones((160,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.linear1.weight", np.ones((128, 160, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd2.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd2.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd2.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.weight", np.ones((192,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.bias", np.zeros((192,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.running_mean", np.zeros((192,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear1.batchnorm.running_var", np.ones((192,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.linear1.weight", np.ones((128, 192, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd3.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd3.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd3.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.weight", np.ones((224,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.bias", np.zeros((224,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.running_mean", np.zeros((224,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear1.batchnorm.running_var", np.ones((224,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.linear1.weight", np.ones((128, 224, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd4.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd4.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd4.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.weight", np.ones((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.bias", np.zeros((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.running_mean", np.zeros((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear1.batchnorm.running_var", np.ones((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.linear1.weight", np.ones((128, 256, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd5.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd5.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd5.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.weight", np.ones((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.bias", np.zeros((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.running_mean", np.zeros((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear1.batchnorm.running_var", np.ones((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.linear1.weight", np.ones((128, 288, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd6.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd6.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd6.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.weight", np.ones((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.bias", np.zeros((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.running_mean", np.zeros((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear1.batchnorm.running_var", np.ones((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.linear1.weight", np.ones((128, 320, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd7.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd7.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd7.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.weight", np.ones((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.bias", np.zeros((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.running_mean", np.zeros((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear1.batchnorm.running_var", np.ones((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.linear1.weight", np.ones((128, 352, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd8.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd8.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd8.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.weight", np.ones((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.bias", np.zeros((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.running_mean", np.zeros((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear1.batchnorm.running_var", np.ones((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.linear1.weight", np.ones((128, 384, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd9.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd9.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd9.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.weight", np.ones((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.bias", np.zeros((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.running_mean", np.zeros((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear1.batchnorm.running_var", np.ones((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.linear1.weight", np.ones((128, 416, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd10.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd10.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd10.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.weight", np.ones((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.bias", np.zeros((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.running_mean", np.zeros((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear1.batchnorm.running_var", np.ones((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.linear1.weight", np.ones((128, 448, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd11.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd11.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd11.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.weight", np.ones((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.bias", np.zeros((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.running_mean", np.zeros((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear1.batchnorm.running_var", np.ones((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.linear1.weight", np.ones((128, 480, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block1.tdnnd12.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block1.tdnnd12.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block1.tdnnd12.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit1.nonlinear.batchnorm.weight", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit1.nonlinear.batchnorm.bias", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit1.nonlinear.batchnorm.running_mean", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit1.nonlinear.batchnorm.running_var", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit1.linear.weight", np.ones((256, 512, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.weight", np.ones((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.bias", np.zeros((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.running_mean", np.zeros((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear1.batchnorm.running_var", np.ones((256,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.linear1.weight", np.ones((128, 256, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd1.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd1.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd1.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.weight", np.ones((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.bias", np.zeros((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.running_mean", np.zeros((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear1.batchnorm.running_var", np.ones((288,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.linear1.weight", np.ones((128, 288, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd2.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd2.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd2.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.weight", np.ones((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.bias", np.zeros((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.running_mean", np.zeros((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear1.batchnorm.running_var", np.ones((320,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.linear1.weight", np.ones((128, 320, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd3.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd3.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd3.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.weight", np.ones((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.bias", np.zeros((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.running_mean", np.zeros((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear1.batchnorm.running_var", np.ones((352,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.linear1.weight", np.ones((128, 352, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd4.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd4.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd4.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.weight", np.ones((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.bias", np.zeros((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.running_mean", np.zeros((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear1.batchnorm.running_var", np.ones((384,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.linear1.weight", np.ones((128, 384, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd5.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd5.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd5.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.weight", np.ones((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.bias", np.zeros((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.running_mean", np.zeros((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear1.batchnorm.running_var", np.ones((416,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.linear1.weight", np.ones((128, 416, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd6.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd6.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd6.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.weight", np.ones((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.bias", np.zeros((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.running_mean", np.zeros((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear1.batchnorm.running_var", np.ones((448,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.linear1.weight", np.ones((128, 448, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd7.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd7.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd7.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.weight", np.ones((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.bias", np.zeros((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.running_mean", np.zeros((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear1.batchnorm.running_var", np.ones((480,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.linear1.weight", np.ones((128, 480, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd8.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd8.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd8.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.weight", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.bias", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.running_mean", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear1.batchnorm.running_var", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.linear1.weight", np.ones((128, 512, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd9.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd9.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd9.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.weight", np.ones((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.bias", np.zeros((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.running_mean", np.zeros((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear1.batchnorm.running_var", np.ones((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.linear1.weight", np.ones((128, 544, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd10.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd10.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd10.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.weight", np.ones((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.bias", np.zeros((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.running_mean", np.zeros((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear1.batchnorm.running_var", np.ones((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.linear1.weight", np.ones((128, 576, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd11.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd11.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd11.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.weight", np.ones((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.bias", np.zeros((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.running_mean", np.zeros((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear1.batchnorm.running_var", np.ones((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.linear1.weight", np.ones((128, 608, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd12.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd12.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd12.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.weight", np.ones((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.bias", np.zeros((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.running_mean", np.zeros((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear1.batchnorm.running_var", np.ones((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.linear1.weight", np.ones((128, 640, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd13.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd13.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd13.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.weight", np.ones((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.bias", np.zeros((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.running_mean", np.zeros((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear1.batchnorm.running_var", np.ones((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.linear1.weight", np.ones((128, 672, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd14.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd14.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd14.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.weight", np.ones((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.bias", np.zeros((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.running_mean", np.zeros((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear1.batchnorm.running_var", np.ones((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.linear1.weight", np.ones((128, 704, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd15.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd15.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd15.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.weight", np.ones((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.bias", np.zeros((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.running_mean", np.zeros((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear1.batchnorm.running_var", np.ones((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.linear1.weight", np.ones((128, 736, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd16.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd16.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd16.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.weight", np.ones((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.bias", np.zeros((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.running_mean", np.zeros((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear1.batchnorm.running_var", np.ones((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.linear1.weight", np.ones((128, 768, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd17.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd17.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd17.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.weight", np.ones((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.bias", np.zeros((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.running_mean", np.zeros((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear1.batchnorm.running_var", np.ones((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.linear1.weight", np.ones((128, 800, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd18.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd18.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd18.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.weight", np.ones((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.bias", np.zeros((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.running_mean", np.zeros((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear1.batchnorm.running_var", np.ones((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.linear1.weight", np.ones((128, 832, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd19.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd19.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd19.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.weight", np.ones((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.bias", np.zeros((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.running_mean", np.zeros((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear1.batchnorm.running_var", np.ones((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.linear1.weight", np.ones((128, 864, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd20.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd20.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd20.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.weight", np.ones((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.bias", np.zeros((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.running_mean", np.zeros((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear1.batchnorm.running_var", np.ones((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.linear1.weight", np.ones((128, 896, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd21.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd21.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd21.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.weight", np.ones((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.bias", np.zeros((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.running_mean", np.zeros((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear1.batchnorm.running_var", np.ones((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.linear1.weight", np.ones((128, 928, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd22.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd22.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd22.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.weight", np.ones((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.bias", np.zeros((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.running_mean", np.zeros((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear1.batchnorm.running_var", np.ones((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.linear1.weight", np.ones((128, 960, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd23.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd23.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd23.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.weight", np.ones((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.bias", np.zeros((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.running_mean", np.zeros((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear1.batchnorm.running_var", np.ones((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.linear1.weight", np.ones((128, 992, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block2.tdnnd24.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block2.tdnnd24.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block2.tdnnd24.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit2.nonlinear.batchnorm.weight", np.ones((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit2.nonlinear.batchnorm.bias", np.zeros((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit2.nonlinear.batchnorm.running_mean", np.zeros((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit2.nonlinear.batchnorm.running_var", np.ones((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit2.linear.weight", np.ones((512, 1024, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.weight", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.bias", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.running_mean", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear1.batchnorm.running_var", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.linear1.weight", np.ones((128, 512, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd1.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd1.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd1.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.weight", np.ones((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.bias", np.zeros((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.running_mean", np.zeros((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear1.batchnorm.running_var", np.ones((544,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.linear1.weight", np.ones((128, 544, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd2.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd2.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd2.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.weight", np.ones((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.bias", np.zeros((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.running_mean", np.zeros((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear1.batchnorm.running_var", np.ones((576,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.linear1.weight", np.ones((128, 576, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd3.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd3.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd3.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.weight", np.ones((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.bias", np.zeros((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.running_mean", np.zeros((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear1.batchnorm.running_var", np.ones((608,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.linear1.weight", np.ones((128, 608, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd4.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd4.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd4.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.weight", np.ones((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.bias", np.zeros((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.running_mean", np.zeros((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear1.batchnorm.running_var", np.ones((640,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.linear1.weight", np.ones((128, 640, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd5.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd5.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd5.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.weight", np.ones((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.bias", np.zeros((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.running_mean", np.zeros((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear1.batchnorm.running_var", np.ones((672,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.linear1.weight", np.ones((128, 672, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd6.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd6.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd6.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.weight", np.ones((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.bias", np.zeros((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.running_mean", np.zeros((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear1.batchnorm.running_var", np.ones((704,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.linear1.weight", np.ones((128, 704, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd7.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd7.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd7.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.weight", np.ones((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.bias", np.zeros((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.running_mean", np.zeros((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear1.batchnorm.running_var", np.ones((736,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.linear1.weight", np.ones((128, 736, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd8.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd8.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd8.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.weight", np.ones((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.bias", np.zeros((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.running_mean", np.zeros((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear1.batchnorm.running_var", np.ones((768,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.linear1.weight", np.ones((128, 768, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd9.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd9.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd9.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.weight", np.ones((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.bias", np.zeros((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.running_mean", np.zeros((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear1.batchnorm.running_var", np.ones((800,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.linear1.weight", np.ones((128, 800, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd10.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd10.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd10.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.weight", np.ones((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.bias", np.zeros((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.running_mean", np.zeros((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear1.batchnorm.running_var", np.ones((832,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.linear1.weight", np.ones((128, 832, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd11.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd11.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd11.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.weight", np.ones((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.bias", np.zeros((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.running_mean", np.zeros((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear1.batchnorm.running_var", np.ones((864,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.linear1.weight", np.ones((128, 864, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd12.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd12.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd12.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.weight", np.ones((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.bias", np.zeros((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.running_mean", np.zeros((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear1.batchnorm.running_var", np.ones((896,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.linear1.weight", np.ones((128, 896, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd13.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd13.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd13.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.weight", np.ones((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.bias", np.zeros((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.running_mean", np.zeros((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear1.batchnorm.running_var", np.ones((928,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.linear1.weight", np.ones((128, 928, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd14.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd14.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd14.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.weight", np.ones((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.bias", np.zeros((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.running_mean", np.zeros((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear1.batchnorm.running_var", np.ones((960,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.linear1.weight", np.ones((128, 960, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd15.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd15.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd15.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.weight", np.ones((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.bias", np.zeros((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.running_mean", np.zeros((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear1.batchnorm.running_var", np.ones((992,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.linear1.weight", np.ones((128, 992, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.weight", np.ones((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.bias", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.running_mean", np.zeros((128,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.nonlinear2.batchnorm.running_var", np.ones((128,), dtype=np.float32), "campplus"),
            (
                "campplus.xvector.block3.tdnnd16.cam_layer.linear_local.weight",
                np.ones((32, 128, 3), dtype=np.float32),
                "campplus",
            ),
            ("campplus.xvector.block3.tdnnd16.cam_layer.linear1.weight", np.zeros((64, 128, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.cam_layer.linear1.bias", np.zeros((64,), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.cam_layer.linear2.weight", np.zeros((32, 64, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.block3.tdnnd16.cam_layer.linear2.bias", np.zeros((32,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit3.nonlinear.batchnorm.weight", np.ones((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit3.nonlinear.batchnorm.bias", np.zeros((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit3.nonlinear.batchnorm.running_mean", np.zeros((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit3.nonlinear.batchnorm.running_var", np.ones((1024,), dtype=np.float32), "campplus"),
            ("campplus.xvector.transit3.linear.weight", np.ones((512, 1024, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.out_nonlinear.batchnorm.weight", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.out_nonlinear.batchnorm.bias", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.out_nonlinear.batchnorm.running_mean", np.zeros((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.out_nonlinear.batchnorm.running_var", np.ones((512,), dtype=np.float32), "campplus"),
            ("campplus.xvector.dense.linear.weight", np.ones((192, 1024, 1), dtype=np.float32), "campplus"),
            ("campplus.xvector.dense.nonlinear.batchnorm.running_mean", np.zeros((192,), dtype=np.float32), "campplus"),
            ("campplus.xvector.dense.nonlinear.batchnorm.running_var", np.ones((192,), dtype=np.float32), "campplus"),
            ("semantic_codec.encoder.0.weight", np.ones((2, 2), dtype=np.float32), "semantic_codec"),
            (
                "semantic_codec.quantizer.quantizers.0.in_project.weight_g",
                np.ones((8, 1, 1), dtype=np.float32),
                "semantic_codec",
            ),
            (
                "semantic_codec.quantizer.quantizers.0.in_project.weight_v",
                np.ones((8, 1024, 1), dtype=np.float32),
                "semantic_codec",
            ),
            (
                "semantic_codec.quantizer.quantizers.0.in_project.bias",
                np.zeros((8,), dtype=np.float32),
                "semantic_codec",
            ),
            (
                "semantic_codec.quantizer.quantizers.0.codebook.weight",
                np.ones((8192, 8), dtype=np.float32),
                "semantic_codec",
            ),
            (
                "semantic_codec.quantizer.quantizers.0.out_project.weight_g",
                np.ones((1024, 1, 1), dtype=np.float32),
                "semantic_codec",
            ),
            (
                "semantic_codec.quantizer.quantizers.0.out_project.weight_v",
                np.ones((1024, 8, 1), dtype=np.float32),
                "semantic_codec",
            ),
            (
                "semantic_codec.quantizer.quantizers.0.out_project.bias",
                np.zeros((1024,), dtype=np.float32),
                "semantic_codec",
            ),
        ],
        metadata={"unit": "campplus-readiness"},
    )


def _write_w2v_bert_contract_bundle(
    bundle_dir: Path,
    *,
    layer_norm_weight: np.ndarray | None = None,
    layer_norm_bias: np.ndarray | None = None,
    projection_weight: np.ndarray | None = None,
    projection_bias: np.ndarray | None = None,
    layer0_ffn1_norm_weight: np.ndarray | None = None,
    layer0_ffn1_norm_bias: np.ndarray | None = None,
    layer0_ffn1_intermediate_weight: np.ndarray | None = None,
    layer0_ffn1_intermediate_bias: np.ndarray | None = None,
    layer0_ffn1_output_weight: np.ndarray | None = None,
    layer0_ffn1_output_bias: np.ndarray | None = None,
    layer0_q_weight: np.ndarray | None = None,
    layer0_k_weight: np.ndarray | None = None,
    layer0_v_weight: np.ndarray | None = None,
    layer0_out_weight: np.ndarray | None = None,
    layer0_self_attn_norm_weight: np.ndarray | None = None,
    layer0_self_attn_norm_bias: np.ndarray | None = None,
    layer0_conv_norm_weight: np.ndarray | None = None,
    layer0_conv_norm_bias: np.ndarray | None = None,
    layer0_conv_pointwise1_weight: np.ndarray | None = None,
    layer0_conv_pointwise1_bias: np.ndarray | None = None,
    layer0_conv_depthwise_weight: np.ndarray | None = None,
    layer0_conv_depthwise_bias: np.ndarray | None = None,
    layer0_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer0_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer0_conv_pointwise2_weight: np.ndarray | None = None,
    layer0_conv_pointwise2_bias: np.ndarray | None = None,
    layer0_ffn2_norm_weight: np.ndarray | None = None,
    layer0_ffn2_norm_bias: np.ndarray | None = None,
    layer0_ffn2_intermediate_weight: np.ndarray | None = None,
    layer0_ffn2_intermediate_bias: np.ndarray | None = None,
    layer0_ffn2_output_weight: np.ndarray | None = None,
    layer0_ffn2_output_bias: np.ndarray | None = None,
    layer0_final_norm_weight: np.ndarray | None = None,
    layer0_final_norm_bias: np.ndarray | None = None,
    layer1_ffn1_norm_weight: np.ndarray | None = None,
    layer1_ffn1_norm_bias: np.ndarray | None = None,
    layer1_ffn1_intermediate_weight: np.ndarray | None = None,
    layer1_ffn1_intermediate_bias: np.ndarray | None = None,
    layer1_ffn1_output_weight: np.ndarray | None = None,
    layer1_ffn1_output_bias: np.ndarray | None = None,
    layer1_q_weight: np.ndarray | None = None,
    layer1_k_weight: np.ndarray | None = None,
    layer1_v_weight: np.ndarray | None = None,
    layer1_out_weight: np.ndarray | None = None,
    layer1_self_attn_norm_weight: np.ndarray | None = None,
    layer1_self_attn_norm_bias: np.ndarray | None = None,
    layer1_conv_norm_weight: np.ndarray | None = None,
    layer1_conv_norm_bias: np.ndarray | None = None,
    layer1_conv_pointwise1_weight: np.ndarray | None = None,
    layer1_conv_pointwise1_bias: np.ndarray | None = None,
    layer1_conv_depthwise_weight: np.ndarray | None = None,
    layer1_conv_depthwise_bias: np.ndarray | None = None,
    layer1_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer1_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer1_conv_pointwise2_weight: np.ndarray | None = None,
    layer1_conv_pointwise2_bias: np.ndarray | None = None,
    layer1_ffn2_norm_weight: np.ndarray | None = None,
    layer1_ffn2_norm_bias: np.ndarray | None = None,
    layer1_ffn2_intermediate_weight: np.ndarray | None = None,
    layer1_ffn2_intermediate_bias: np.ndarray | None = None,
    layer1_ffn2_output_weight: np.ndarray | None = None,
    layer1_ffn2_output_bias: np.ndarray | None = None,
    layer1_final_norm_weight: np.ndarray | None = None,
    layer1_final_norm_bias: np.ndarray | None = None,
    layer2_ffn1_norm_weight: np.ndarray | None = None,
    layer2_ffn1_norm_bias: np.ndarray | None = None,
    layer2_ffn1_intermediate_weight: np.ndarray | None = None,
    layer2_ffn1_intermediate_bias: np.ndarray | None = None,
    layer2_ffn1_output_weight: np.ndarray | None = None,
    layer2_ffn1_output_bias: np.ndarray | None = None,
    layer2_q_weight: np.ndarray | None = None,
    layer2_k_weight: np.ndarray | None = None,
    layer2_v_weight: np.ndarray | None = None,
    layer2_out_weight: np.ndarray | None = None,
    layer2_self_attn_norm_weight: np.ndarray | None = None,
    layer2_self_attn_norm_bias: np.ndarray | None = None,
    layer2_conv_norm_weight: np.ndarray | None = None,
    layer2_conv_norm_bias: np.ndarray | None = None,
    layer2_conv_pointwise1_weight: np.ndarray | None = None,
    layer2_conv_pointwise1_bias: np.ndarray | None = None,
    layer2_conv_depthwise_weight: np.ndarray | None = None,
    layer2_conv_depthwise_bias: np.ndarray | None = None,
    layer2_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer2_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer2_conv_pointwise2_weight: np.ndarray | None = None,
    layer2_conv_pointwise2_bias: np.ndarray | None = None,
    layer2_ffn2_norm_weight: np.ndarray | None = None,
    layer2_ffn2_norm_bias: np.ndarray | None = None,
    layer2_ffn2_intermediate_weight: np.ndarray | None = None,
    layer2_ffn2_intermediate_bias: np.ndarray | None = None,
    layer2_ffn2_output_weight: np.ndarray | None = None,
    layer2_ffn2_output_bias: np.ndarray | None = None,
    layer3_ffn1_norm_weight: np.ndarray | None = None,
    layer3_ffn1_norm_bias: np.ndarray | None = None,
    layer3_ffn1_intermediate_weight: np.ndarray | None = None,
    layer3_ffn1_intermediate_bias: np.ndarray | None = None,
    layer3_ffn1_output_weight: np.ndarray | None = None,
    layer3_ffn1_output_bias: np.ndarray | None = None,
    layer3_q_weight: np.ndarray | None = None,
    layer3_k_weight: np.ndarray | None = None,
    layer3_v_weight: np.ndarray | None = None,
    layer3_out_weight: np.ndarray | None = None,
    layer3_self_attn_norm_weight: np.ndarray | None = None,
    layer3_self_attn_norm_bias: np.ndarray | None = None,
    layer3_conv_norm_weight: np.ndarray | None = None,
    layer3_conv_norm_bias: np.ndarray | None = None,
    layer3_conv_pointwise1_weight: np.ndarray | None = None,
    layer3_conv_pointwise1_bias: np.ndarray | None = None,
    layer3_conv_depthwise_weight: np.ndarray | None = None,
    layer3_conv_depthwise_bias: np.ndarray | None = None,
    layer3_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer3_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer3_conv_pointwise2_weight: np.ndarray | None = None,
    layer3_conv_pointwise2_bias: np.ndarray | None = None,
    layer3_ffn2_norm_weight: np.ndarray | None = None,
    layer3_ffn2_norm_bias: np.ndarray | None = None,
    layer3_ffn2_intermediate_weight: np.ndarray | None = None,
    layer3_ffn2_intermediate_bias: np.ndarray | None = None,
    layer3_ffn2_output_weight: np.ndarray | None = None,
    layer3_ffn2_output_bias: np.ndarray | None = None,
    layer3_final_norm_weight: np.ndarray | None = None,
    layer3_final_norm_bias: np.ndarray | None = None,
    layer4_ffn1_norm_weight: np.ndarray | None = None,
    layer4_ffn1_norm_bias: np.ndarray | None = None,
    layer4_ffn1_intermediate_weight: np.ndarray | None = None,
    layer4_ffn1_intermediate_bias: np.ndarray | None = None,
    layer4_ffn1_output_weight: np.ndarray | None = None,
    layer4_ffn1_output_bias: np.ndarray | None = None,
    layer4_q_weight: np.ndarray | None = None,
    layer4_k_weight: np.ndarray | None = None,
    layer4_v_weight: np.ndarray | None = None,
    layer4_out_weight: np.ndarray | None = None,
    layer4_self_attn_norm_weight: np.ndarray | None = None,
    layer4_self_attn_norm_bias: np.ndarray | None = None,
    layer4_conv_norm_weight: np.ndarray | None = None,
    layer4_conv_norm_bias: np.ndarray | None = None,
    layer4_conv_pointwise1_weight: np.ndarray | None = None,
    layer4_conv_pointwise1_bias: np.ndarray | None = None,
    layer4_conv_depthwise_weight: np.ndarray | None = None,
    layer4_conv_depthwise_bias: np.ndarray | None = None,
    layer4_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer4_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer4_conv_pointwise2_weight: np.ndarray | None = None,
    layer4_conv_pointwise2_bias: np.ndarray | None = None,
    layer4_ffn2_norm_weight: np.ndarray | None = None,
    layer4_ffn2_norm_bias: np.ndarray | None = None,
    layer4_ffn2_intermediate_weight: np.ndarray | None = None,
    layer4_ffn2_intermediate_bias: np.ndarray | None = None,
    layer4_ffn2_output_weight: np.ndarray | None = None,
    layer4_ffn2_output_bias: np.ndarray | None = None,
    layer5_ffn1_norm_weight: np.ndarray | None = None,
    layer5_ffn1_norm_bias: np.ndarray | None = None,
    layer5_ffn1_intermediate_weight: np.ndarray | None = None,
    layer5_ffn1_intermediate_bias: np.ndarray | None = None,
    layer5_ffn1_output_weight: np.ndarray | None = None,
    layer5_ffn1_output_bias: np.ndarray | None = None,
    layer5_q_weight: np.ndarray | None = None,
    layer5_k_weight: np.ndarray | None = None,
    layer5_v_weight: np.ndarray | None = None,
    layer5_out_weight: np.ndarray | None = None,
    layer5_self_attn_norm_weight: np.ndarray | None = None,
    layer5_self_attn_norm_bias: np.ndarray | None = None,
    layer5_conv_norm_weight: np.ndarray | None = None,
    layer5_conv_norm_bias: np.ndarray | None = None,
    layer5_conv_pointwise1_weight: np.ndarray | None = None,
    layer5_conv_pointwise1_bias: np.ndarray | None = None,
    layer5_conv_depthwise_weight: np.ndarray | None = None,
    layer5_conv_depthwise_bias: np.ndarray | None = None,
    layer5_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer5_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer5_conv_pointwise2_weight: np.ndarray | None = None,
    layer5_conv_pointwise2_bias: np.ndarray | None = None,
    layer5_ffn2_norm_weight: np.ndarray | None = None,
    layer5_ffn2_norm_bias: np.ndarray | None = None,
    layer5_ffn2_intermediate_weight: np.ndarray | None = None,
    layer5_ffn2_intermediate_bias: np.ndarray | None = None,
    layer5_ffn2_output_weight: np.ndarray | None = None,
    layer5_ffn2_output_bias: np.ndarray | None = None,
    layer6_ffn1_norm_weight: np.ndarray | None = None,
    layer6_ffn1_norm_bias: np.ndarray | None = None,
    layer6_ffn1_intermediate_weight: np.ndarray | None = None,
    layer6_ffn1_intermediate_bias: np.ndarray | None = None,
    layer6_ffn1_output_weight: np.ndarray | None = None,
    layer6_ffn1_output_bias: np.ndarray | None = None,
    layer6_q_weight: np.ndarray | None = None,
    layer6_k_weight: np.ndarray | None = None,
    layer6_v_weight: np.ndarray | None = None,
    layer6_out_weight: np.ndarray | None = None,
    layer6_self_attn_norm_weight: np.ndarray | None = None,
    layer6_self_attn_norm_bias: np.ndarray | None = None,
    layer6_conv_norm_weight: np.ndarray | None = None,
    layer6_conv_norm_bias: np.ndarray | None = None,
    layer6_conv_pointwise1_weight: np.ndarray | None = None,
    layer6_conv_pointwise1_bias: np.ndarray | None = None,
    layer6_conv_depthwise_weight: np.ndarray | None = None,
    layer6_conv_depthwise_bias: np.ndarray | None = None,
    layer6_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer6_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer6_conv_pointwise2_weight: np.ndarray | None = None,
    layer6_conv_pointwise2_bias: np.ndarray | None = None,
    layer6_ffn2_norm_weight: np.ndarray | None = None,
    layer6_ffn2_norm_bias: np.ndarray | None = None,
    layer6_ffn2_intermediate_weight: np.ndarray | None = None,
    layer6_ffn2_intermediate_bias: np.ndarray | None = None,
    layer6_ffn2_output_weight: np.ndarray | None = None,
    layer6_ffn2_output_bias: np.ndarray | None = None,
    layer7_ffn1_norm_weight: np.ndarray | None = None,
    layer7_ffn1_norm_bias: np.ndarray | None = None,
    layer7_ffn1_intermediate_weight: np.ndarray | None = None,
    layer7_ffn1_intermediate_bias: np.ndarray | None = None,
    layer7_ffn1_output_weight: np.ndarray | None = None,
    layer7_ffn1_output_bias: np.ndarray | None = None,
    layer7_q_weight: np.ndarray | None = None,
    layer7_k_weight: np.ndarray | None = None,
    layer7_v_weight: np.ndarray | None = None,
    layer7_out_weight: np.ndarray | None = None,
    layer7_self_attn_norm_weight: np.ndarray | None = None,
    layer7_self_attn_norm_bias: np.ndarray | None = None,
    layer7_conv_norm_weight: np.ndarray | None = None,
    layer7_conv_norm_bias: np.ndarray | None = None,
    layer7_conv_pointwise1_weight: np.ndarray | None = None,
    layer7_conv_pointwise1_bias: np.ndarray | None = None,
    layer7_conv_depthwise_weight: np.ndarray | None = None,
    layer7_conv_depthwise_bias: np.ndarray | None = None,
    layer7_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer7_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer7_conv_pointwise2_weight: np.ndarray | None = None,
    layer7_conv_pointwise2_bias: np.ndarray | None = None,
    layer7_ffn2_norm_weight: np.ndarray | None = None,
    layer7_ffn2_norm_bias: np.ndarray | None = None,
    layer7_ffn2_intermediate_weight: np.ndarray | None = None,
    layer7_ffn2_intermediate_bias: np.ndarray | None = None,
    layer7_ffn2_output_weight: np.ndarray | None = None,
    layer7_ffn2_output_bias: np.ndarray | None = None,
    layer8_ffn1_norm_weight: np.ndarray | None = None,
    layer8_ffn1_norm_bias: np.ndarray | None = None,
    layer8_ffn1_intermediate_weight: np.ndarray | None = None,
    layer8_ffn1_intermediate_bias: np.ndarray | None = None,
    layer8_ffn1_output_weight: np.ndarray | None = None,
    layer8_ffn1_output_bias: np.ndarray | None = None,
    layer8_q_weight: np.ndarray | None = None,
    layer8_k_weight: np.ndarray | None = None,
    layer8_v_weight: np.ndarray | None = None,
    layer8_out_weight: np.ndarray | None = None,
    layer8_self_attn_norm_weight: np.ndarray | None = None,
    layer8_self_attn_norm_bias: np.ndarray | None = None,
    layer8_conv_norm_weight: np.ndarray | None = None,
    layer8_conv_norm_bias: np.ndarray | None = None,
    layer8_conv_pointwise1_weight: np.ndarray | None = None,
    layer8_conv_pointwise1_bias: np.ndarray | None = None,
    layer8_conv_depthwise_weight: np.ndarray | None = None,
    layer8_conv_depthwise_bias: np.ndarray | None = None,
    layer8_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer8_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer8_conv_pointwise2_weight: np.ndarray | None = None,
    layer8_conv_pointwise2_bias: np.ndarray | None = None,
    layer8_ffn2_norm_weight: np.ndarray | None = None,
    layer8_ffn2_norm_bias: np.ndarray | None = None,
    layer8_ffn2_intermediate_weight: np.ndarray | None = None,
    layer8_ffn2_intermediate_bias: np.ndarray | None = None,
    layer8_ffn2_output_weight: np.ndarray | None = None,
    layer8_ffn2_output_bias: np.ndarray | None = None,
    layer9_ffn1_norm_weight: np.ndarray | None = None,
    layer9_ffn1_norm_bias: np.ndarray | None = None,
    layer9_ffn1_intermediate_weight: np.ndarray | None = None,
    layer9_ffn1_intermediate_bias: np.ndarray | None = None,
    layer9_ffn1_output_weight: np.ndarray | None = None,
    layer9_ffn1_output_bias: np.ndarray | None = None,
    layer9_q_weight: np.ndarray | None = None,
    layer9_k_weight: np.ndarray | None = None,
    layer9_v_weight: np.ndarray | None = None,
    layer9_out_weight: np.ndarray | None = None,
    layer9_self_attn_norm_weight: np.ndarray | None = None,
    layer9_self_attn_norm_bias: np.ndarray | None = None,
    layer9_conv_norm_weight: np.ndarray | None = None,
    layer9_conv_norm_bias: np.ndarray | None = None,
    layer9_conv_pointwise1_weight: np.ndarray | None = None,
    layer9_conv_pointwise1_bias: np.ndarray | None = None,
    layer9_conv_depthwise_weight: np.ndarray | None = None,
    layer9_conv_depthwise_bias: np.ndarray | None = None,
    layer9_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer9_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer9_conv_pointwise2_weight: np.ndarray | None = None,
    layer9_conv_pointwise2_bias: np.ndarray | None = None,
    layer9_ffn2_norm_weight: np.ndarray | None = None,
    layer9_ffn2_norm_bias: np.ndarray | None = None,
    layer9_ffn2_intermediate_weight: np.ndarray | None = None,
    layer9_ffn2_intermediate_bias: np.ndarray | None = None,
    layer9_ffn2_output_weight: np.ndarray | None = None,
    layer9_ffn2_output_bias: np.ndarray | None = None,
    layer10_ffn1_norm_weight: np.ndarray | None = None,
    layer10_ffn1_norm_bias: np.ndarray | None = None,
    layer10_ffn1_intermediate_weight: np.ndarray | None = None,
    layer10_ffn1_intermediate_bias: np.ndarray | None = None,
    layer10_ffn1_output_weight: np.ndarray | None = None,
    layer10_ffn1_output_bias: np.ndarray | None = None,
    layer10_q_weight: np.ndarray | None = None,
    layer10_k_weight: np.ndarray | None = None,
    layer10_v_weight: np.ndarray | None = None,
    layer10_out_weight: np.ndarray | None = None,
    layer10_self_attn_norm_weight: np.ndarray | None = None,
    layer10_self_attn_norm_bias: np.ndarray | None = None,
    layer10_conv_norm_weight: np.ndarray | None = None,
    layer10_conv_norm_bias: np.ndarray | None = None,
    layer10_conv_pointwise1_weight: np.ndarray | None = None,
    layer10_conv_pointwise1_bias: np.ndarray | None = None,
    layer10_conv_depthwise_weight: np.ndarray | None = None,
    layer10_conv_depthwise_bias: np.ndarray | None = None,
    layer10_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer10_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer10_conv_pointwise2_weight: np.ndarray | None = None,
    layer10_conv_pointwise2_bias: np.ndarray | None = None,
    layer10_ffn2_norm_weight: np.ndarray | None = None,
    layer10_ffn2_norm_bias: np.ndarray | None = None,
    layer10_ffn2_intermediate_weight: np.ndarray | None = None,
    layer10_ffn2_intermediate_bias: np.ndarray | None = None,
    layer10_ffn2_output_weight: np.ndarray | None = None,
    layer10_ffn2_output_bias: np.ndarray | None = None,
    layer11_ffn1_norm_weight: np.ndarray | None = None,
    layer11_ffn1_norm_bias: np.ndarray | None = None,
    layer11_ffn1_intermediate_weight: np.ndarray | None = None,
    layer11_ffn1_intermediate_bias: np.ndarray | None = None,
    layer11_ffn1_output_weight: np.ndarray | None = None,
    layer11_ffn1_output_bias: np.ndarray | None = None,
    layer11_q_weight: np.ndarray | None = None,
    layer11_k_weight: np.ndarray | None = None,
    layer11_v_weight: np.ndarray | None = None,
    layer11_out_weight: np.ndarray | None = None,
    layer11_self_attn_norm_weight: np.ndarray | None = None,
    layer11_self_attn_norm_bias: np.ndarray | None = None,
    layer11_conv_norm_weight: np.ndarray | None = None,
    layer11_conv_norm_bias: np.ndarray | None = None,
    layer11_conv_pointwise1_weight: np.ndarray | None = None,
    layer11_conv_pointwise1_bias: np.ndarray | None = None,
    layer11_conv_depthwise_weight: np.ndarray | None = None,
    layer11_conv_depthwise_bias: np.ndarray | None = None,
    layer11_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer11_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer11_conv_pointwise2_weight: np.ndarray | None = None,
    layer11_conv_pointwise2_bias: np.ndarray | None = None,
    layer11_ffn2_norm_weight: np.ndarray | None = None,
    layer11_ffn2_norm_bias: np.ndarray | None = None,
    layer11_ffn2_intermediate_weight: np.ndarray | None = None,
    layer11_ffn2_intermediate_bias: np.ndarray | None = None,
    layer11_ffn2_output_weight: np.ndarray | None = None,
    layer11_ffn2_output_bias: np.ndarray | None = None,
    layer12_ffn1_norm_weight: np.ndarray | None = None,
    layer12_ffn1_norm_bias: np.ndarray | None = None,
    layer12_ffn1_intermediate_weight: np.ndarray | None = None,
    layer12_ffn1_intermediate_bias: np.ndarray | None = None,
    layer12_ffn1_output_weight: np.ndarray | None = None,
    layer12_ffn1_output_bias: np.ndarray | None = None,
    layer12_q_weight: np.ndarray | None = None,
    layer12_k_weight: np.ndarray | None = None,
    layer12_v_weight: np.ndarray | None = None,
    layer12_out_weight: np.ndarray | None = None,
    layer12_self_attn_norm_weight: np.ndarray | None = None,
    layer12_self_attn_norm_bias: np.ndarray | None = None,
    layer12_conv_norm_weight: np.ndarray | None = None,
    layer12_conv_norm_bias: np.ndarray | None = None,
    layer12_conv_pointwise1_weight: np.ndarray | None = None,
    layer12_conv_pointwise1_bias: np.ndarray | None = None,
    layer12_conv_depthwise_weight: np.ndarray | None = None,
    layer12_conv_depthwise_bias: np.ndarray | None = None,
    layer12_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer12_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer12_conv_pointwise2_weight: np.ndarray | None = None,
    layer12_conv_pointwise2_bias: np.ndarray | None = None,
    layer12_ffn2_norm_weight: np.ndarray | None = None,
    layer12_ffn2_norm_bias: np.ndarray | None = None,
    layer12_ffn2_intermediate_weight: np.ndarray | None = None,
    layer12_ffn2_intermediate_bias: np.ndarray | None = None,
    layer12_ffn2_output_weight: np.ndarray | None = None,
    layer12_ffn2_output_bias: np.ndarray | None = None,
    layer13_ffn1_norm_weight: np.ndarray | None = None,
    layer13_ffn1_norm_bias: np.ndarray | None = None,
    layer13_ffn1_intermediate_weight: np.ndarray | None = None,
    layer13_ffn1_intermediate_bias: np.ndarray | None = None,
    layer13_ffn1_output_weight: np.ndarray | None = None,
    layer13_ffn1_output_bias: np.ndarray | None = None,
    layer13_q_weight: np.ndarray | None = None,
    layer13_k_weight: np.ndarray | None = None,
    layer13_v_weight: np.ndarray | None = None,
    layer13_out_weight: np.ndarray | None = None,
    layer13_self_attn_norm_weight: np.ndarray | None = None,
    layer13_self_attn_norm_bias: np.ndarray | None = None,
    layer13_conv_norm_weight: np.ndarray | None = None,
    layer13_conv_norm_bias: np.ndarray | None = None,
    layer13_conv_pointwise1_weight: np.ndarray | None = None,
    layer13_conv_pointwise1_bias: np.ndarray | None = None,
    layer13_conv_depthwise_weight: np.ndarray | None = None,
    layer13_conv_depthwise_bias: np.ndarray | None = None,
    layer13_conv_depthwise_norm_weight: np.ndarray | None = None,
    layer13_conv_depthwise_norm_bias: np.ndarray | None = None,
    layer13_conv_pointwise2_weight: np.ndarray | None = None,
    layer13_conv_pointwise2_bias: np.ndarray | None = None,
    layer13_ffn2_norm_weight: np.ndarray | None = None,
    layer13_ffn2_norm_bias: np.ndarray | None = None,
    layer13_ffn2_intermediate_weight: np.ndarray | None = None,
    layer13_ffn2_intermediate_bias: np.ndarray | None = None,
    layer13_ffn2_output_weight: np.ndarray | None = None,
    layer13_ffn2_output_bias: np.ndarray | None = None,
    layer14_ffn2_norm_weight: np.ndarray | None = None,
    layer14_ffn2_norm_bias: np.ndarray | None = None,
    layer14_ffn2_intermediate_weight: np.ndarray | None = None,
    layer14_ffn2_intermediate_bias: np.ndarray | None = None,
    layer14_ffn2_output_weight: np.ndarray | None = None,
    layer14_ffn2_output_bias: np.ndarray | None = None,
    layer14_ffn1_norm_weight: np.ndarray | None = None,
    layer14_ffn1_norm_bias: np.ndarray | None = None,
    layer14_ffn1_intermediate_weight: np.ndarray | None = None,
    layer14_ffn1_intermediate_bias: np.ndarray | None = None,
    layer14_ffn1_output_weight: np.ndarray | None = None,
    layer14_ffn1_output_bias: np.ndarray | None = None,
    layer17_final_norm_weight: np.ndarray | None = None,
    layer17_final_norm_bias: np.ndarray | None = None,
    stats_mean: np.ndarray | None = None,
    stats_std: np.ndarray | None = None,
) -> None:
    ln_weight = (
        np.ones((160,), dtype=np.float32)
        if layer_norm_weight is None
        else np.asarray(layer_norm_weight, dtype=np.float32)
    )
    ln_bias = (
        np.zeros((160,), dtype=np.float32)
        if layer_norm_bias is None
        else np.asarray(layer_norm_bias, dtype=np.float32)
    )
    proj_weight = (
        np.ones((1024, 160), dtype=np.float32)
        if projection_weight is None
        else np.asarray(projection_weight, dtype=np.float32)
    )
    proj_bias = (
        np.zeros((1024,), dtype=np.float32)
        if projection_bias is None
        else np.asarray(projection_bias, dtype=np.float32)
    )
    ffn1_norm_weight = (
        np.ones((1024,), dtype=np.float32)
        if layer0_ffn1_norm_weight is None
        else np.asarray(layer0_ffn1_norm_weight, dtype=np.float32)
    )
    ffn1_norm_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_ffn1_norm_bias is None
        else np.asarray(layer0_ffn1_norm_bias, dtype=np.float32)
    )
    ffn1_intermediate_weight = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer0_ffn1_intermediate_weight is None
        else np.asarray(layer0_ffn1_intermediate_weight, dtype=np.float32)
    )
    ffn1_intermediate_bias = (
        np.zeros((4096,), dtype=np.float32)
        if layer0_ffn1_intermediate_bias is None
        else np.asarray(layer0_ffn1_intermediate_bias, dtype=np.float32)
    )
    ffn1_output_weight = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer0_ffn1_output_weight is None
        else np.asarray(layer0_ffn1_output_weight, dtype=np.float32)
    )
    ffn1_output_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_ffn1_output_bias is None
        else np.asarray(layer0_ffn1_output_bias, dtype=np.float32)
    )
    q_weight = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer0_q_weight is None
        else np.asarray(layer0_q_weight, dtype=np.float32)
    )
    k_weight = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer0_k_weight is None
        else np.asarray(layer0_k_weight, dtype=np.float32)
    )
    v_weight = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer0_v_weight is None
        else np.asarray(layer0_v_weight, dtype=np.float32)
    )
    out_weight = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer0_out_weight is None
        else np.asarray(layer0_out_weight, dtype=np.float32)
    )
    self_attn_norm_weight = (
        np.ones((1024,), dtype=np.float32)
        if layer0_self_attn_norm_weight is None
        else np.asarray(layer0_self_attn_norm_weight, dtype=np.float32)
    )
    self_attn_norm_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_self_attn_norm_bias is None
        else np.asarray(layer0_self_attn_norm_bias, dtype=np.float32)
    )
    conv_norm_weight = (
        np.ones((1024,), dtype=np.float32)
        if layer0_conv_norm_weight is None
        else np.asarray(layer0_conv_norm_weight, dtype=np.float32)
    )
    conv_norm_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_conv_norm_bias is None
        else np.asarray(layer0_conv_norm_bias, dtype=np.float32)
    )
    conv_pointwise1_weight = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer0_conv_pointwise1_weight is None
        else np.asarray(layer0_conv_pointwise1_weight, dtype=np.float32)
    )
    conv_pointwise1_bias = (
        np.zeros((2048,), dtype=np.float32)
        if layer0_conv_pointwise1_bias is None
        else np.asarray(layer0_conv_pointwise1_bias, dtype=np.float32)
    )
    conv_depthwise_weight = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer0_conv_depthwise_weight is None
        else np.asarray(layer0_conv_depthwise_weight, dtype=np.float32)
    )
    conv_depthwise_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_conv_depthwise_bias is None
        else np.asarray(layer0_conv_depthwise_bias, dtype=np.float32)
    )
    conv_depthwise_norm_weight = (
        np.ones((1024,), dtype=np.float32)
        if layer0_conv_depthwise_norm_weight is None
        else np.asarray(layer0_conv_depthwise_norm_weight, dtype=np.float32)
    )
    conv_depthwise_norm_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_conv_depthwise_norm_bias is None
        else np.asarray(layer0_conv_depthwise_norm_bias, dtype=np.float32)
    )
    conv_pointwise2_weight = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer0_conv_pointwise2_weight is None
        else np.asarray(layer0_conv_pointwise2_weight, dtype=np.float32)
    )
    conv_pointwise2_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_conv_pointwise2_bias is None
        else np.asarray(layer0_conv_pointwise2_bias, dtype=np.float32)
    )
    ffn2_norm_weight = (
        np.ones((1024,), dtype=np.float32)
        if layer0_ffn2_norm_weight is None
        else np.asarray(layer0_ffn2_norm_weight, dtype=np.float32)
    )
    ffn2_norm_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_ffn2_norm_bias is None
        else np.asarray(layer0_ffn2_norm_bias, dtype=np.float32)
    )
    ffn2_intermediate_weight = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer0_ffn2_intermediate_weight is None
        else np.asarray(layer0_ffn2_intermediate_weight, dtype=np.float32)
    )
    ffn2_intermediate_bias = (
        np.zeros((4096,), dtype=np.float32)
        if layer0_ffn2_intermediate_bias is None
        else np.asarray(layer0_ffn2_intermediate_bias, dtype=np.float32)
    )
    ffn2_output_weight = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer0_ffn2_output_weight is None
        else np.asarray(layer0_ffn2_output_weight, dtype=np.float32)
    )
    ffn2_output_bias = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_ffn2_output_bias is None
        else np.asarray(layer0_ffn2_output_bias, dtype=np.float32)
    )
    layer0_final_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer0_final_norm_weight is None
        else np.asarray(layer0_final_norm_weight, dtype=np.float32)
    )
    layer0_final_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer0_final_norm_bias is None
        else np.asarray(layer0_final_norm_bias, dtype=np.float32)
    )
    layer1_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer1_ffn1_norm_weight is None
        else np.asarray(layer1_ffn1_norm_weight, dtype=np.float32)
    )
    layer1_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_ffn1_norm_bias is None
        else np.asarray(layer1_ffn1_norm_bias, dtype=np.float32)
    )
    layer1_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer1_ffn1_intermediate_weight is None
        else np.asarray(layer1_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer1_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer1_ffn1_intermediate_bias is None
        else np.asarray(layer1_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer1_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer1_ffn1_output_weight is None
        else np.asarray(layer1_ffn1_output_weight, dtype=np.float32)
    )
    layer1_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_ffn1_output_bias is None
        else np.asarray(layer1_ffn1_output_bias, dtype=np.float32)
    )
    layer1_q_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer1_q_weight is None
        else np.asarray(layer1_q_weight, dtype=np.float32)
    )
    layer1_k_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer1_k_weight is None
        else np.asarray(layer1_k_weight, dtype=np.float32)
    )
    layer1_v_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer1_v_weight is None
        else np.asarray(layer1_v_weight, dtype=np.float32)
    )
    layer1_out_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer1_out_weight is None
        else np.asarray(layer1_out_weight, dtype=np.float32)
    )
    layer1_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer1_self_attn_norm_weight is None
        else np.asarray(layer1_self_attn_norm_weight, dtype=np.float32)
    )
    layer1_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_self_attn_norm_bias is None
        else np.asarray(layer1_self_attn_norm_bias, dtype=np.float32)
    )
    layer1_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer1_conv_norm_weight is None
        else np.asarray(layer1_conv_norm_weight, dtype=np.float32)
    )
    layer1_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_conv_norm_bias is None
        else np.asarray(layer1_conv_norm_bias, dtype=np.float32)
    )
    layer1_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer1_conv_pointwise1_weight is None
        else np.asarray(layer1_conv_pointwise1_weight, dtype=np.float32)
    )
    layer1_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer1_conv_pointwise1_bias is None
        else np.asarray(layer1_conv_pointwise1_bias, dtype=np.float32)
    )
    layer1_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer1_conv_depthwise_weight is None
        else np.asarray(layer1_conv_depthwise_weight, dtype=np.float32)
    )
    layer1_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_conv_depthwise_bias is None
        else np.asarray(layer1_conv_depthwise_bias, dtype=np.float32)
    )
    layer1_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer1_conv_depthwise_norm_weight is None
        else np.asarray(layer1_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer1_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_conv_depthwise_norm_bias is None
        else np.asarray(layer1_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer1_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer1_conv_pointwise2_weight is None
        else np.asarray(layer1_conv_pointwise2_weight, dtype=np.float32)
    )
    layer1_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_conv_pointwise2_bias is None
        else np.asarray(layer1_conv_pointwise2_bias, dtype=np.float32)
    )
    layer1_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer1_ffn2_norm_weight is None
        else np.asarray(layer1_ffn2_norm_weight, dtype=np.float32)
    )
    layer1_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_ffn2_norm_bias is None
        else np.asarray(layer1_ffn2_norm_bias, dtype=np.float32)
    )
    layer1_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer1_ffn2_intermediate_weight is None
        else np.asarray(layer1_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer1_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer1_ffn2_intermediate_bias is None
        else np.asarray(layer1_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer1_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer1_ffn2_output_weight is None
        else np.asarray(layer1_ffn2_output_weight, dtype=np.float32)
    )
    layer1_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_ffn2_output_bias is None
        else np.asarray(layer1_ffn2_output_bias, dtype=np.float32)
    )
    layer1_final_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer1_final_norm_weight is None
        else np.asarray(layer1_final_norm_weight, dtype=np.float32)
    )
    layer1_final_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer1_final_norm_bias is None
        else np.asarray(layer1_final_norm_bias, dtype=np.float32)
    )
    layer2_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer2_ffn1_norm_weight is None
        else np.asarray(layer2_ffn1_norm_weight, dtype=np.float32)
    )
    layer2_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_ffn1_norm_bias is None
        else np.asarray(layer2_ffn1_norm_bias, dtype=np.float32)
    )
    layer2_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer2_ffn1_intermediate_weight is None
        else np.asarray(layer2_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer2_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer2_ffn1_intermediate_bias is None
        else np.asarray(layer2_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer2_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer2_ffn1_output_weight is None
        else np.asarray(layer2_ffn1_output_weight, dtype=np.float32)
    )
    layer2_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_ffn1_output_bias is None
        else np.asarray(layer2_ffn1_output_bias, dtype=np.float32)
    )
    layer2_q_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer2_q_weight is None
        else np.asarray(layer2_q_weight, dtype=np.float32)
    )
    layer2_k_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer2_k_weight is None
        else np.asarray(layer2_k_weight, dtype=np.float32)
    )
    layer2_v_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer2_v_weight is None
        else np.asarray(layer2_v_weight, dtype=np.float32)
    )
    layer2_out_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer2_out_weight is None
        else np.asarray(layer2_out_weight, dtype=np.float32)
    )
    layer2_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer2_self_attn_norm_weight is None
        else np.asarray(layer2_self_attn_norm_weight, dtype=np.float32)
    )
    layer2_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_self_attn_norm_bias is None
        else np.asarray(layer2_self_attn_norm_bias, dtype=np.float32)
    )
    layer2_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer2_conv_norm_weight is None
        else np.asarray(layer2_conv_norm_weight, dtype=np.float32)
    )
    layer2_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_conv_norm_bias is None
        else np.asarray(layer2_conv_norm_bias, dtype=np.float32)
    )
    layer2_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer2_conv_pointwise1_weight is None
        else np.asarray(layer2_conv_pointwise1_weight, dtype=np.float32)
    )
    layer2_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer2_conv_pointwise1_bias is None
        else np.asarray(layer2_conv_pointwise1_bias, dtype=np.float32)
    )
    layer2_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer2_conv_depthwise_weight is None
        else np.asarray(layer2_conv_depthwise_weight, dtype=np.float32)
    )
    layer2_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_conv_depthwise_bias is None
        else np.asarray(layer2_conv_depthwise_bias, dtype=np.float32)
    )
    layer2_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer2_conv_depthwise_norm_weight is None
        else np.asarray(layer2_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer2_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_conv_depthwise_norm_bias is None
        else np.asarray(layer2_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer2_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer2_conv_pointwise2_weight is None
        else np.asarray(layer2_conv_pointwise2_weight, dtype=np.float32)
    )
    layer2_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_conv_pointwise2_bias is None
        else np.asarray(layer2_conv_pointwise2_bias, dtype=np.float32)
    )
    layer2_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer2_ffn2_norm_weight is None
        else np.asarray(layer2_ffn2_norm_weight, dtype=np.float32)
    )
    layer2_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_ffn2_norm_bias is None
        else np.asarray(layer2_ffn2_norm_bias, dtype=np.float32)
    )
    layer2_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer2_ffn2_intermediate_weight is None
        else np.asarray(layer2_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer2_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer2_ffn2_intermediate_bias is None
        else np.asarray(layer2_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer2_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer2_ffn2_output_weight is None
        else np.asarray(layer2_ffn2_output_weight, dtype=np.float32)
    )
    layer2_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer2_ffn2_output_bias is None
        else np.asarray(layer2_ffn2_output_bias, dtype=np.float32)
    )
    layer3_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer3_ffn1_norm_weight is None
        else np.asarray(layer3_ffn1_norm_weight, dtype=np.float32)
    )
    layer3_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_ffn1_norm_bias is None
        else np.asarray(layer3_ffn1_norm_bias, dtype=np.float32)
    )
    layer3_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer3_ffn1_intermediate_weight is None
        else np.asarray(layer3_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer3_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer3_ffn1_intermediate_bias is None
        else np.asarray(layer3_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer3_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer3_ffn1_output_weight is None
        else np.asarray(layer3_ffn1_output_weight, dtype=np.float32)
    )
    layer3_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_ffn1_output_bias is None
        else np.asarray(layer3_ffn1_output_bias, dtype=np.float32)
    )
    layer3_q_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer3_q_weight is None
        else np.asarray(layer3_q_weight, dtype=np.float32)
    )
    layer3_k_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer3_k_weight is None
        else np.asarray(layer3_k_weight, dtype=np.float32)
    )
    layer3_v_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer3_v_weight is None
        else np.asarray(layer3_v_weight, dtype=np.float32)
    )
    layer3_out_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer3_out_weight is None
        else np.asarray(layer3_out_weight, dtype=np.float32)
    )
    layer3_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer3_self_attn_norm_weight is None
        else np.asarray(layer3_self_attn_norm_weight, dtype=np.float32)
    )
    layer3_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_self_attn_norm_bias is None
        else np.asarray(layer3_self_attn_norm_bias, dtype=np.float32)
    )
    layer3_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer3_conv_norm_weight is None
        else np.asarray(layer3_conv_norm_weight, dtype=np.float32)
    )
    layer3_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_conv_norm_bias is None
        else np.asarray(layer3_conv_norm_bias, dtype=np.float32)
    )
    layer3_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer3_conv_pointwise1_weight is None
        else np.asarray(layer3_conv_pointwise1_weight, dtype=np.float32)
    )
    layer3_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer3_conv_pointwise1_bias is None
        else np.asarray(layer3_conv_pointwise1_bias, dtype=np.float32)
    )
    layer3_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer3_conv_depthwise_weight is None
        else np.asarray(layer3_conv_depthwise_weight, dtype=np.float32)
    )
    layer3_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_conv_depthwise_bias is None
        else np.asarray(layer3_conv_depthwise_bias, dtype=np.float32)
    )
    layer3_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer3_conv_depthwise_norm_weight is None
        else np.asarray(layer3_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer3_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_conv_depthwise_norm_bias is None
        else np.asarray(layer3_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer3_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer3_conv_pointwise2_weight is None
        else np.asarray(layer3_conv_pointwise2_weight, dtype=np.float32)
    )
    layer3_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_conv_pointwise2_bias is None
        else np.asarray(layer3_conv_pointwise2_bias, dtype=np.float32)
    )
    layer3_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer3_ffn2_norm_weight is None
        else np.asarray(layer3_ffn2_norm_weight, dtype=np.float32)
    )
    layer3_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_ffn2_norm_bias is None
        else np.asarray(layer3_ffn2_norm_bias, dtype=np.float32)
    )
    layer3_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer3_ffn2_intermediate_weight is None
        else np.asarray(layer3_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer3_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer3_ffn2_intermediate_bias is None
        else np.asarray(layer3_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer3_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer3_ffn2_output_weight is None
        else np.asarray(layer3_ffn2_output_weight, dtype=np.float32)
    )
    layer3_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_ffn2_output_bias is None
        else np.asarray(layer3_ffn2_output_bias, dtype=np.float32)
    )
    layer3_final_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer3_final_norm_weight is None
        else np.asarray(layer3_final_norm_weight, dtype=np.float32)
    )
    layer3_final_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer3_final_norm_bias is None
        else np.asarray(layer3_final_norm_bias, dtype=np.float32)
    )
    layer4_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer4_ffn1_norm_weight is None
        else np.asarray(layer4_ffn1_norm_weight, dtype=np.float32)
    )
    layer4_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_ffn1_norm_bias is None
        else np.asarray(layer4_ffn1_norm_bias, dtype=np.float32)
    )
    layer4_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer4_ffn1_intermediate_weight is None
        else np.asarray(layer4_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer4_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer4_ffn1_intermediate_bias is None
        else np.asarray(layer4_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer4_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer4_ffn1_output_weight is None
        else np.asarray(layer4_ffn1_output_weight, dtype=np.float32)
    )
    layer4_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_ffn1_output_bias is None
        else np.asarray(layer4_ffn1_output_bias, dtype=np.float32)
    )
    layer4_q_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer4_q_weight is None
        else np.asarray(layer4_q_weight, dtype=np.float32)
    )
    layer4_k_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer4_k_weight is None
        else np.asarray(layer4_k_weight, dtype=np.float32)
    )
    layer4_v_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer4_v_weight is None
        else np.asarray(layer4_v_weight, dtype=np.float32)
    )
    layer4_out_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer4_out_weight is None
        else np.asarray(layer4_out_weight, dtype=np.float32)
    )
    layer4_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer4_self_attn_norm_weight is None
        else np.asarray(layer4_self_attn_norm_weight, dtype=np.float32)
    )
    layer4_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_self_attn_norm_bias is None
        else np.asarray(layer4_self_attn_norm_bias, dtype=np.float32)
    )
    layer4_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer4_conv_norm_weight is None
        else np.asarray(layer4_conv_norm_weight, dtype=np.float32)
    )
    layer4_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_conv_norm_bias is None
        else np.asarray(layer4_conv_norm_bias, dtype=np.float32)
    )
    layer4_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer4_conv_pointwise1_weight is None
        else np.asarray(layer4_conv_pointwise1_weight, dtype=np.float32)
    )
    layer4_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer4_conv_pointwise1_bias is None
        else np.asarray(layer4_conv_pointwise1_bias, dtype=np.float32)
    )
    layer4_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer4_conv_depthwise_weight is None
        else np.asarray(layer4_conv_depthwise_weight, dtype=np.float32)
    )
    layer4_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_conv_depthwise_bias is None
        else np.asarray(layer4_conv_depthwise_bias, dtype=np.float32)
    )
    layer4_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer4_conv_depthwise_norm_weight is None
        else np.asarray(layer4_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer4_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_conv_depthwise_norm_bias is None
        else np.asarray(layer4_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer4_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer4_conv_pointwise2_weight is None
        else np.asarray(layer4_conv_pointwise2_weight, dtype=np.float32)
    )
    layer4_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_conv_pointwise2_bias is None
        else np.asarray(layer4_conv_pointwise2_bias, dtype=np.float32)
    )
    layer4_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer4_ffn2_norm_weight is None
        else np.asarray(layer4_ffn2_norm_weight, dtype=np.float32)
    )
    layer4_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_ffn2_norm_bias is None
        else np.asarray(layer4_ffn2_norm_bias, dtype=np.float32)
    )
    layer4_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer4_ffn2_intermediate_weight is None
        else np.asarray(layer4_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer4_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer4_ffn2_intermediate_bias is None
        else np.asarray(layer4_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer4_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer4_ffn2_output_weight is None
        else np.asarray(layer4_ffn2_output_weight, dtype=np.float32)
    )
    layer4_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer4_ffn2_output_bias is None
        else np.asarray(layer4_ffn2_output_bias, dtype=np.float32)
    )
    layer5_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer5_ffn1_norm_weight is None
        else np.asarray(layer5_ffn1_norm_weight, dtype=np.float32)
    )
    layer5_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_ffn1_norm_bias is None
        else np.asarray(layer5_ffn1_norm_bias, dtype=np.float32)
    )
    layer5_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer5_ffn1_intermediate_weight is None
        else np.asarray(layer5_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer5_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer5_ffn1_intermediate_bias is None
        else np.asarray(layer5_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer5_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer5_ffn1_output_weight is None
        else np.asarray(layer5_ffn1_output_weight, dtype=np.float32)
    )
    layer5_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_ffn1_output_bias is None
        else np.asarray(layer5_ffn1_output_bias, dtype=np.float32)
    )
    layer5_q_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer5_q_weight is None
        else np.asarray(layer5_q_weight, dtype=np.float32)
    )
    layer5_k_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer5_k_weight is None
        else np.asarray(layer5_k_weight, dtype=np.float32)
    )
    layer5_v_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer5_v_weight is None
        else np.asarray(layer5_v_weight, dtype=np.float32)
    )
    layer5_out_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer5_out_weight is None
        else np.asarray(layer5_out_weight, dtype=np.float32)
    )
    layer5_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer5_self_attn_norm_weight is None
        else np.asarray(layer5_self_attn_norm_weight, dtype=np.float32)
    )
    layer5_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_self_attn_norm_bias is None
        else np.asarray(layer5_self_attn_norm_bias, dtype=np.float32)
    )
    layer5_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer5_conv_norm_weight is None
        else np.asarray(layer5_conv_norm_weight, dtype=np.float32)
    )
    layer5_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_conv_norm_bias is None
        else np.asarray(layer5_conv_norm_bias, dtype=np.float32)
    )
    layer5_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer5_conv_pointwise1_weight is None
        else np.asarray(layer5_conv_pointwise1_weight, dtype=np.float32)
    )
    layer5_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer5_conv_pointwise1_bias is None
        else np.asarray(layer5_conv_pointwise1_bias, dtype=np.float32)
    )
    layer5_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer5_conv_depthwise_weight is None
        else np.asarray(layer5_conv_depthwise_weight, dtype=np.float32)
    )
    layer5_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_conv_depthwise_bias is None
        else np.asarray(layer5_conv_depthwise_bias, dtype=np.float32)
    )
    layer5_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer5_conv_depthwise_norm_weight is None
        else np.asarray(layer5_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer5_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_conv_depthwise_norm_bias is None
        else np.asarray(layer5_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer5_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer5_conv_pointwise2_weight is None
        else np.asarray(layer5_conv_pointwise2_weight, dtype=np.float32)
    )
    layer5_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_conv_pointwise2_bias is None
        else np.asarray(layer5_conv_pointwise2_bias, dtype=np.float32)
    )
    layer5_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer5_ffn2_norm_weight is None
        else np.asarray(layer5_ffn2_norm_weight, dtype=np.float32)
    )
    layer5_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_ffn2_norm_bias is None
        else np.asarray(layer5_ffn2_norm_bias, dtype=np.float32)
    )
    layer5_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer5_ffn2_intermediate_weight is None
        else np.asarray(layer5_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer5_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer5_ffn2_intermediate_bias is None
        else np.asarray(layer5_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer5_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer5_ffn2_output_weight is None
        else np.asarray(layer5_ffn2_output_weight, dtype=np.float32)
    )
    layer5_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer5_ffn2_output_bias is None
        else np.asarray(layer5_ffn2_output_bias, dtype=np.float32)
    )
    layer6_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer6_ffn1_norm_weight is None
        else np.asarray(layer6_ffn1_norm_weight, dtype=np.float32)
    )
    layer6_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_ffn1_norm_bias is None
        else np.asarray(layer6_ffn1_norm_bias, dtype=np.float32)
    )
    layer6_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer6_ffn1_intermediate_weight is None
        else np.asarray(layer6_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer6_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer6_ffn1_intermediate_bias is None
        else np.asarray(layer6_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer6_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer6_ffn1_output_weight is None
        else np.asarray(layer6_ffn1_output_weight, dtype=np.float32)
    )
    layer6_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_ffn1_output_bias is None
        else np.asarray(layer6_ffn1_output_bias, dtype=np.float32)
    )
    layer6_q_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer6_q_weight is None
        else np.asarray(layer6_q_weight, dtype=np.float32)
    )
    layer6_k_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer6_k_weight is None
        else np.asarray(layer6_k_weight, dtype=np.float32)
    )
    layer6_v_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer6_v_weight is None
        else np.asarray(layer6_v_weight, dtype=np.float32)
    )
    layer6_out_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer6_out_weight is None
        else np.asarray(layer6_out_weight, dtype=np.float32)
    )
    layer6_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer6_self_attn_norm_weight is None
        else np.asarray(layer6_self_attn_norm_weight, dtype=np.float32)
    )
    layer6_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_self_attn_norm_bias is None
        else np.asarray(layer6_self_attn_norm_bias, dtype=np.float32)
    )
    layer6_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer6_conv_norm_weight is None
        else np.asarray(layer6_conv_norm_weight, dtype=np.float32)
    )
    layer6_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_conv_norm_bias is None
        else np.asarray(layer6_conv_norm_bias, dtype=np.float32)
    )
    layer6_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer6_conv_pointwise1_weight is None
        else np.asarray(layer6_conv_pointwise1_weight, dtype=np.float32)
    )
    layer6_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer6_conv_pointwise1_bias is None
        else np.asarray(layer6_conv_pointwise1_bias, dtype=np.float32)
    )
    layer6_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer6_conv_depthwise_weight is None
        else np.asarray(layer6_conv_depthwise_weight, dtype=np.float32)
    )
    layer6_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_conv_depthwise_bias is None
        else np.asarray(layer6_conv_depthwise_bias, dtype=np.float32)
    )
    layer6_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer6_conv_depthwise_norm_weight is None
        else np.asarray(layer6_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer6_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_conv_depthwise_norm_bias is None
        else np.asarray(layer6_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer6_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer6_conv_pointwise2_weight is None
        else np.asarray(layer6_conv_pointwise2_weight, dtype=np.float32)
    )
    layer6_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_conv_pointwise2_bias is None
        else np.asarray(layer6_conv_pointwise2_bias, dtype=np.float32)
    )
    layer6_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer6_ffn2_norm_weight is None
        else np.asarray(layer6_ffn2_norm_weight, dtype=np.float32)
    )
    layer6_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_ffn2_norm_bias is None
        else np.asarray(layer6_ffn2_norm_bias, dtype=np.float32)
    )
    layer6_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer6_ffn2_intermediate_weight is None
        else np.asarray(layer6_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer6_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer6_ffn2_intermediate_bias is None
        else np.asarray(layer6_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer6_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer6_ffn2_output_weight is None
        else np.asarray(layer6_ffn2_output_weight, dtype=np.float32)
    )
    layer6_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer6_ffn2_output_bias is None
        else np.asarray(layer6_ffn2_output_bias, dtype=np.float32)
    )
    layer7_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer7_ffn1_norm_weight is None
        else np.asarray(layer7_ffn1_norm_weight, dtype=np.float32)
    )
    layer7_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_ffn1_norm_bias is None
        else np.asarray(layer7_ffn1_norm_bias, dtype=np.float32)
    )
    layer7_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer7_ffn1_intermediate_weight is None
        else np.asarray(layer7_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer7_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer7_ffn1_intermediate_bias is None
        else np.asarray(layer7_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer7_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer7_ffn1_output_weight is None
        else np.asarray(layer7_ffn1_output_weight, dtype=np.float32)
    )
    layer7_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_ffn1_output_bias is None
        else np.asarray(layer7_ffn1_output_bias, dtype=np.float32)
    )
    layer7_q_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer7_q_weight is None
        else np.asarray(layer7_q_weight, dtype=np.float32)
    )
    layer7_k_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer7_k_weight is None
        else np.asarray(layer7_k_weight, dtype=np.float32)
    )
    layer7_v_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer7_v_weight is None
        else np.asarray(layer7_v_weight, dtype=np.float32)
    )
    layer7_out_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer7_out_weight is None
        else np.asarray(layer7_out_weight, dtype=np.float32)
    )
    layer7_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer7_self_attn_norm_weight is None
        else np.asarray(layer7_self_attn_norm_weight, dtype=np.float32)
    )
    layer7_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_self_attn_norm_bias is None
        else np.asarray(layer7_self_attn_norm_bias, dtype=np.float32)
    )
    layer7_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer7_conv_norm_weight is None
        else np.asarray(layer7_conv_norm_weight, dtype=np.float32)
    )
    layer7_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_conv_norm_bias is None
        else np.asarray(layer7_conv_norm_bias, dtype=np.float32)
    )
    layer7_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer7_conv_pointwise1_weight is None
        else np.asarray(layer7_conv_pointwise1_weight, dtype=np.float32)
    )
    layer7_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer7_conv_pointwise1_bias is None
        else np.asarray(layer7_conv_pointwise1_bias, dtype=np.float32)
    )
    layer7_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer7_conv_depthwise_weight is None
        else np.asarray(layer7_conv_depthwise_weight, dtype=np.float32)
    )
    layer7_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_conv_depthwise_bias is None
        else np.asarray(layer7_conv_depthwise_bias, dtype=np.float32)
    )
    layer7_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer7_conv_depthwise_norm_weight is None
        else np.asarray(layer7_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer7_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_conv_depthwise_norm_bias is None
        else np.asarray(layer7_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer7_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer7_conv_pointwise2_weight is None
        else np.asarray(layer7_conv_pointwise2_weight, dtype=np.float32)
    )
    layer7_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_conv_pointwise2_bias is None
        else np.asarray(layer7_conv_pointwise2_bias, dtype=np.float32)
    )
    layer7_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer7_ffn2_norm_weight is None
        else np.asarray(layer7_ffn2_norm_weight, dtype=np.float32)
    )
    layer7_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_ffn2_norm_bias is None
        else np.asarray(layer7_ffn2_norm_bias, dtype=np.float32)
    )
    layer7_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer7_ffn2_intermediate_weight is None
        else np.asarray(layer7_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer7_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer7_ffn2_intermediate_bias is None
        else np.asarray(layer7_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer7_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer7_ffn2_output_weight is None
        else np.asarray(layer7_ffn2_output_weight, dtype=np.float32)
    )
    layer7_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer7_ffn2_output_bias is None
        else np.asarray(layer7_ffn2_output_bias, dtype=np.float32)
    )
    layer8_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer8_ffn1_norm_weight is None
        else np.asarray(layer8_ffn1_norm_weight, dtype=np.float32)
    )
    layer8_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_ffn1_norm_bias is None
        else np.asarray(layer8_ffn1_norm_bias, dtype=np.float32)
    )
    layer8_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer8_ffn1_intermediate_weight is None
        else np.asarray(layer8_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer8_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer8_ffn1_intermediate_bias is None
        else np.asarray(layer8_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer8_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer8_ffn1_output_weight is None
        else np.asarray(layer8_ffn1_output_weight, dtype=np.float32)
    )
    layer8_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_ffn1_output_bias is None
        else np.asarray(layer8_ffn1_output_bias, dtype=np.float32)
    )
    layer8_q_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer8_q_weight is None
        else np.asarray(layer8_q_weight, dtype=np.float32)
    )
    layer8_k_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer8_k_weight is None
        else np.asarray(layer8_k_weight, dtype=np.float32)
    )
    layer8_v_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer8_v_weight is None
        else np.asarray(layer8_v_weight, dtype=np.float32)
    )
    layer8_out_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer8_out_weight is None
        else np.asarray(layer8_out_weight, dtype=np.float32)
    )
    layer8_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer8_self_attn_norm_weight is None
        else np.asarray(layer8_self_attn_norm_weight, dtype=np.float32)
    )
    layer8_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_self_attn_norm_bias is None
        else np.asarray(layer8_self_attn_norm_bias, dtype=np.float32)
    )
    layer8_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer8_conv_norm_weight is None
        else np.asarray(layer8_conv_norm_weight, dtype=np.float32)
    )
    layer8_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_conv_norm_bias is None
        else np.asarray(layer8_conv_norm_bias, dtype=np.float32)
    )
    layer8_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer8_conv_pointwise1_weight is None
        else np.asarray(layer8_conv_pointwise1_weight, dtype=np.float32)
    )
    layer8_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer8_conv_pointwise1_bias is None
        else np.asarray(layer8_conv_pointwise1_bias, dtype=np.float32)
    )
    layer8_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer8_conv_depthwise_weight is None
        else np.asarray(layer8_conv_depthwise_weight, dtype=np.float32)
    )
    layer8_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_conv_depthwise_bias is None
        else np.asarray(layer8_conv_depthwise_bias, dtype=np.float32)
    )
    layer8_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer8_conv_depthwise_norm_weight is None
        else np.asarray(layer8_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer8_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_conv_depthwise_norm_bias is None
        else np.asarray(layer8_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer8_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer8_conv_pointwise2_weight is None
        else np.asarray(layer8_conv_pointwise2_weight, dtype=np.float32)
    )
    layer8_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_conv_pointwise2_bias is None
        else np.asarray(layer8_conv_pointwise2_bias, dtype=np.float32)
    )
    layer8_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer8_ffn2_norm_weight is None
        else np.asarray(layer8_ffn2_norm_weight, dtype=np.float32)
    )
    layer8_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_ffn2_norm_bias is None
        else np.asarray(layer8_ffn2_norm_bias, dtype=np.float32)
    )
    layer8_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer8_ffn2_intermediate_weight is None
        else np.asarray(layer8_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer8_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer8_ffn2_intermediate_bias is None
        else np.asarray(layer8_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer8_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer8_ffn2_output_weight is None
        else np.asarray(layer8_ffn2_output_weight, dtype=np.float32)
    )
    layer8_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer8_ffn2_output_bias is None
        else np.asarray(layer8_ffn2_output_bias, dtype=np.float32)
    )
    layer9_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer9_ffn1_norm_weight is None
        else np.asarray(layer9_ffn1_norm_weight, dtype=np.float32)
    )
    layer9_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_ffn1_norm_bias is None
        else np.asarray(layer9_ffn1_norm_bias, dtype=np.float32)
    )
    layer9_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer9_ffn1_intermediate_weight is None
        else np.asarray(layer9_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer9_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer9_ffn1_intermediate_bias is None
        else np.asarray(layer9_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer9_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer9_ffn1_output_weight is None
        else np.asarray(layer9_ffn1_output_weight, dtype=np.float32)
    )
    layer9_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_ffn1_output_bias is None
        else np.asarray(layer9_ffn1_output_bias, dtype=np.float32)
    )
    layer9_q_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer9_q_weight is None
        else np.asarray(layer9_q_weight, dtype=np.float32)
    )
    layer9_k_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer9_k_weight is None
        else np.asarray(layer9_k_weight, dtype=np.float32)
    )
    layer9_v_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer9_v_weight is None
        else np.asarray(layer9_v_weight, dtype=np.float32)
    )
    layer9_out_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer9_out_weight is None
        else np.asarray(layer9_out_weight, dtype=np.float32)
    )
    layer9_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer9_self_attn_norm_weight is None
        else np.asarray(layer9_self_attn_norm_weight, dtype=np.float32)
    )
    layer9_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_self_attn_norm_bias is None
        else np.asarray(layer9_self_attn_norm_bias, dtype=np.float32)
    )
    layer9_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer9_conv_norm_weight is None
        else np.asarray(layer9_conv_norm_weight, dtype=np.float32)
    )
    layer9_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_conv_norm_bias is None
        else np.asarray(layer9_conv_norm_bias, dtype=np.float32)
    )
    layer9_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer9_conv_pointwise1_weight is None
        else np.asarray(layer9_conv_pointwise1_weight, dtype=np.float32)
    )
    layer9_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer9_conv_pointwise1_bias is None
        else np.asarray(layer9_conv_pointwise1_bias, dtype=np.float32)
    )
    layer9_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer9_conv_depthwise_weight is None
        else np.asarray(layer9_conv_depthwise_weight, dtype=np.float32)
    )
    layer9_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_conv_depthwise_bias is None
        else np.asarray(layer9_conv_depthwise_bias, dtype=np.float32)
    )
    layer9_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer9_conv_depthwise_norm_weight is None
        else np.asarray(layer9_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer9_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_conv_depthwise_norm_bias is None
        else np.asarray(layer9_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer9_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer9_conv_pointwise2_weight is None
        else np.asarray(layer9_conv_pointwise2_weight, dtype=np.float32)
    )
    layer9_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_conv_pointwise2_bias is None
        else np.asarray(layer9_conv_pointwise2_bias, dtype=np.float32)
    )
    layer9_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer9_ffn2_norm_weight is None
        else np.asarray(layer9_ffn2_norm_weight, dtype=np.float32)
    )
    layer9_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_ffn2_norm_bias is None
        else np.asarray(layer9_ffn2_norm_bias, dtype=np.float32)
    )
    layer9_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer9_ffn2_intermediate_weight is None
        else np.asarray(layer9_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer9_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer9_ffn2_intermediate_bias is None
        else np.asarray(layer9_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer9_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer9_ffn2_output_weight is None
        else np.asarray(layer9_ffn2_output_weight, dtype=np.float32)
    )
    layer9_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer9_ffn2_output_bias is None
        else np.asarray(layer9_ffn2_output_bias, dtype=np.float32)
    )
    layer10_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer10_ffn1_norm_weight is None
        else np.asarray(layer10_ffn1_norm_weight, dtype=np.float32)
    )
    layer10_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_ffn1_norm_bias is None
        else np.asarray(layer10_ffn1_norm_bias, dtype=np.float32)
    )
    layer10_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer10_ffn1_intermediate_weight is None
        else np.asarray(layer10_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer10_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer10_ffn1_intermediate_bias is None
        else np.asarray(layer10_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer10_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer10_ffn1_output_weight is None
        else np.asarray(layer10_ffn1_output_weight, dtype=np.float32)
    )
    layer10_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_ffn1_output_bias is None
        else np.asarray(layer10_ffn1_output_bias, dtype=np.float32)
    )
    layer10_q_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer10_q_weight is None
        else np.asarray(layer10_q_weight, dtype=np.float32)
    )
    layer10_k_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer10_k_weight is None
        else np.asarray(layer10_k_weight, dtype=np.float32)
    )
    layer10_v_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer10_v_weight is None
        else np.asarray(layer10_v_weight, dtype=np.float32)
    )
    layer10_out_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer10_out_weight is None
        else np.asarray(layer10_out_weight, dtype=np.float32)
    )
    layer10_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer10_self_attn_norm_weight is None
        else np.asarray(layer10_self_attn_norm_weight, dtype=np.float32)
    )
    layer10_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_self_attn_norm_bias is None
        else np.asarray(layer10_self_attn_norm_bias, dtype=np.float32)
    )
    layer10_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer10_conv_norm_weight is None
        else np.asarray(layer10_conv_norm_weight, dtype=np.float32)
    )
    layer10_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_conv_norm_bias is None
        else np.asarray(layer10_conv_norm_bias, dtype=np.float32)
    )
    layer10_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer10_conv_pointwise1_weight is None
        else np.asarray(layer10_conv_pointwise1_weight, dtype=np.float32)
    )
    layer10_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer10_conv_pointwise1_bias is None
        else np.asarray(layer10_conv_pointwise1_bias, dtype=np.float32)
    )
    layer10_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer10_conv_depthwise_weight is None
        else np.asarray(layer10_conv_depthwise_weight, dtype=np.float32)
    )
    layer10_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_conv_depthwise_bias is None
        else np.asarray(layer10_conv_depthwise_bias, dtype=np.float32)
    )
    layer10_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer10_conv_depthwise_norm_weight is None
        else np.asarray(layer10_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer10_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_conv_depthwise_norm_bias is None
        else np.asarray(layer10_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer10_conv_pointwise2_weight_arr = (
        np.eye(1024, dtype=np.float32).reshape(1024, 1024, 1)
        if layer10_conv_pointwise2_weight is None
        else np.asarray(layer10_conv_pointwise2_weight, dtype=np.float32)
    )
    layer10_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_conv_pointwise2_bias is None
        else np.asarray(layer10_conv_pointwise2_bias, dtype=np.float32)
    )
    layer10_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer10_ffn2_norm_weight is None
        else np.asarray(layer10_ffn2_norm_weight, dtype=np.float32)
    )
    layer10_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_ffn2_norm_bias is None
        else np.asarray(layer10_ffn2_norm_bias, dtype=np.float32)
    )
    layer10_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer10_ffn2_intermediate_weight is None
        else np.asarray(layer10_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer10_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer10_ffn2_intermediate_bias is None
        else np.asarray(layer10_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer10_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer10_ffn2_output_weight is None
        else np.asarray(layer10_ffn2_output_weight, dtype=np.float32)
    )
    layer10_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer10_ffn2_output_bias is None
        else np.asarray(layer10_ffn2_output_bias, dtype=np.float32)
    )
    layer11_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer11_ffn1_norm_weight is None
        else np.asarray(layer11_ffn1_norm_weight, dtype=np.float32)
    )
    layer11_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_ffn1_norm_bias is None
        else np.asarray(layer11_ffn1_norm_bias, dtype=np.float32)
    )
    layer11_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer11_ffn1_intermediate_weight is None
        else np.asarray(layer11_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer11_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer11_ffn1_intermediate_bias is None
        else np.asarray(layer11_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer11_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer11_ffn1_output_weight is None
        else np.asarray(layer11_ffn1_output_weight, dtype=np.float32)
    )
    layer11_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_ffn1_output_bias is None
        else np.asarray(layer11_ffn1_output_bias, dtype=np.float32)
    )
    layer11_q_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer11_q_weight is None
        else np.asarray(layer11_q_weight, dtype=np.float32)
    )
    layer11_k_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer11_k_weight is None
        else np.asarray(layer11_k_weight, dtype=np.float32)
    )
    layer11_v_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer11_v_weight is None
        else np.asarray(layer11_v_weight, dtype=np.float32)
    )
    layer11_out_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer11_out_weight is None
        else np.asarray(layer11_out_weight, dtype=np.float32)
    )
    layer11_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer11_self_attn_norm_weight is None
        else np.asarray(layer11_self_attn_norm_weight, dtype=np.float32)
    )
    layer11_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_self_attn_norm_bias is None
        else np.asarray(layer11_self_attn_norm_bias, dtype=np.float32)
    )
    layer11_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer11_conv_norm_weight is None
        else np.asarray(layer11_conv_norm_weight, dtype=np.float32)
    )
    layer11_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_conv_norm_bias is None
        else np.asarray(layer11_conv_norm_bias, dtype=np.float32)
    )
    layer11_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer11_conv_pointwise1_weight is None
        else np.asarray(layer11_conv_pointwise1_weight, dtype=np.float32)
    )
    layer11_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer11_conv_pointwise1_bias is None
        else np.asarray(layer11_conv_pointwise1_bias, dtype=np.float32)
    )
    layer11_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer11_conv_depthwise_weight is None
        else np.asarray(layer11_conv_depthwise_weight, dtype=np.float32)
    )
    layer11_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_conv_depthwise_bias is None
        else np.asarray(layer11_conv_depthwise_bias, dtype=np.float32)
    )
    layer11_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer11_conv_depthwise_norm_weight is None
        else np.asarray(layer11_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer11_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_conv_depthwise_norm_bias is None
        else np.asarray(layer11_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer11_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer11_conv_pointwise2_weight is None
        else np.asarray(layer11_conv_pointwise2_weight, dtype=np.float32)
    )
    layer11_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_conv_pointwise2_bias is None
        else np.asarray(layer11_conv_pointwise2_bias, dtype=np.float32)
    )
    layer11_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer11_ffn2_norm_weight is None
        else np.asarray(layer11_ffn2_norm_weight, dtype=np.float32)
    )
    layer11_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_ffn2_norm_bias is None
        else np.asarray(layer11_ffn2_norm_bias, dtype=np.float32)
    )
    layer11_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer11_ffn2_intermediate_weight is None
        else np.asarray(layer11_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer11_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer11_ffn2_intermediate_bias is None
        else np.asarray(layer11_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer11_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer11_ffn2_output_weight is None
        else np.asarray(layer11_ffn2_output_weight, dtype=np.float32)
    )
    layer11_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer11_ffn2_output_bias is None
        else np.asarray(layer11_ffn2_output_bias, dtype=np.float32)
    )
    layer12_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer12_ffn1_norm_weight is None
        else np.asarray(layer12_ffn1_norm_weight, dtype=np.float32)
    )
    layer12_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_ffn1_norm_bias is None
        else np.asarray(layer12_ffn1_norm_bias, dtype=np.float32)
    )
    layer12_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer12_ffn1_intermediate_weight is None
        else np.asarray(layer12_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer12_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer12_ffn1_intermediate_bias is None
        else np.asarray(layer12_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer12_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer12_ffn1_output_weight is None
        else np.asarray(layer12_ffn1_output_weight, dtype=np.float32)
    )
    layer12_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_ffn1_output_bias is None
        else np.asarray(layer12_ffn1_output_bias, dtype=np.float32)
    )
    layer12_q_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer12_q_weight is None
        else np.asarray(layer12_q_weight, dtype=np.float32)
    )
    layer12_k_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer12_k_weight is None
        else np.asarray(layer12_k_weight, dtype=np.float32)
    )
    layer12_v_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer12_v_weight is None
        else np.asarray(layer12_v_weight, dtype=np.float32)
    )
    layer12_out_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer12_out_weight is None
        else np.asarray(layer12_out_weight, dtype=np.float32)
    )
    layer12_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer12_self_attn_norm_weight is None
        else np.asarray(layer12_self_attn_norm_weight, dtype=np.float32)
    )
    layer12_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_self_attn_norm_bias is None
        else np.asarray(layer12_self_attn_norm_bias, dtype=np.float32)
    )
    layer12_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer12_conv_norm_weight is None
        else np.asarray(layer12_conv_norm_weight, dtype=np.float32)
    )
    layer12_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_conv_norm_bias is None
        else np.asarray(layer12_conv_norm_bias, dtype=np.float32)
    )
    layer12_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer12_conv_pointwise1_weight is None
        else np.asarray(layer12_conv_pointwise1_weight, dtype=np.float32)
    )
    layer12_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer12_conv_pointwise1_bias is None
        else np.asarray(layer12_conv_pointwise1_bias, dtype=np.float32)
    )
    layer12_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer12_conv_depthwise_weight is None
        else np.asarray(layer12_conv_depthwise_weight, dtype=np.float32)
    )
    layer12_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_conv_depthwise_bias is None
        else np.asarray(layer12_conv_depthwise_bias, dtype=np.float32)
    )
    layer12_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer12_conv_depthwise_norm_weight is None
        else np.asarray(layer12_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer12_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_conv_depthwise_norm_bias is None
        else np.asarray(layer12_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer12_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer12_conv_pointwise2_weight is None
        else np.asarray(layer12_conv_pointwise2_weight, dtype=np.float32)
    )
    layer12_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_conv_pointwise2_bias is None
        else np.asarray(layer12_conv_pointwise2_bias, dtype=np.float32)
    )
    layer12_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer12_ffn2_norm_weight is None
        else np.asarray(layer12_ffn2_norm_weight, dtype=np.float32)
    )
    layer12_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_ffn2_norm_bias is None
        else np.asarray(layer12_ffn2_norm_bias, dtype=np.float32)
    )
    layer12_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer12_ffn2_intermediate_weight is None
        else np.asarray(layer12_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer12_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer12_ffn2_intermediate_bias is None
        else np.asarray(layer12_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer12_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer12_ffn2_output_weight is None
        else np.asarray(layer12_ffn2_output_weight, dtype=np.float32)
    )
    layer12_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer12_ffn2_output_bias is None
        else np.asarray(layer12_ffn2_output_bias, dtype=np.float32)
    )
    layer13_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer13_ffn1_norm_weight is None
        else np.asarray(layer13_ffn1_norm_weight, dtype=np.float32)
    )
    layer13_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_ffn1_norm_bias is None
        else np.asarray(layer13_ffn1_norm_bias, dtype=np.float32)
    )
    layer13_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer13_ffn1_intermediate_weight is None
        else np.asarray(layer13_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer13_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer13_ffn1_intermediate_bias is None
        else np.asarray(layer13_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer13_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer13_ffn1_output_weight is None
        else np.asarray(layer13_ffn1_output_weight, dtype=np.float32)
    )
    layer13_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_ffn1_output_bias is None
        else np.asarray(layer13_ffn1_output_bias, dtype=np.float32)
    )
    layer13_q_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer13_q_weight is None
        else np.asarray(layer13_q_weight, dtype=np.float32)
    )
    layer13_k_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer13_k_weight is None
        else np.asarray(layer13_k_weight, dtype=np.float32)
    )
    layer13_v_weight_arr = (
        np.ones((1024, 1024), dtype=np.float32)
        if layer13_v_weight is None
        else np.asarray(layer13_v_weight, dtype=np.float32)
    )
    layer13_out_weight_arr = (
        np.eye(1024, dtype=np.float32)
        if layer13_out_weight is None
        else np.asarray(layer13_out_weight, dtype=np.float32)
    )
    layer13_self_attn_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer13_self_attn_norm_weight is None
        else np.asarray(layer13_self_attn_norm_weight, dtype=np.float32)
    )
    layer13_self_attn_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_self_attn_norm_bias is None
        else np.asarray(layer13_self_attn_norm_bias, dtype=np.float32)
    )
    layer13_conv_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer13_conv_norm_weight is None
        else np.asarray(layer13_conv_norm_weight, dtype=np.float32)
    )
    layer13_conv_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_conv_norm_bias is None
        else np.asarray(layer13_conv_norm_bias, dtype=np.float32)
    )
    layer13_conv_pointwise1_weight_arr = (
        np.ones((2048, 1024, 1), dtype=np.float32)
        if layer13_conv_pointwise1_weight is None
        else np.asarray(layer13_conv_pointwise1_weight, dtype=np.float32)
    )
    layer13_conv_pointwise1_bias_arr = (
        np.zeros((2048,), dtype=np.float32)
        if layer13_conv_pointwise1_bias is None
        else np.asarray(layer13_conv_pointwise1_bias, dtype=np.float32)
    )
    layer13_conv_depthwise_weight_arr = (
        np.ones((1024, 1, 31), dtype=np.float32)
        if layer13_conv_depthwise_weight is None
        else np.asarray(layer13_conv_depthwise_weight, dtype=np.float32)
    )
    layer13_conv_depthwise_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_conv_depthwise_bias is None
        else np.asarray(layer13_conv_depthwise_bias, dtype=np.float32)
    )
    layer13_conv_depthwise_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer13_conv_depthwise_norm_weight is None
        else np.asarray(layer13_conv_depthwise_norm_weight, dtype=np.float32)
    )
    layer13_conv_depthwise_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_conv_depthwise_norm_bias is None
        else np.asarray(layer13_conv_depthwise_norm_bias, dtype=np.float32)
    )
    layer13_conv_pointwise2_weight_arr = (
        np.ones((1024, 1024, 1), dtype=np.float32)
        if layer13_conv_pointwise2_weight is None
        else np.asarray(layer13_conv_pointwise2_weight, dtype=np.float32)
    )
    layer13_conv_pointwise2_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_conv_pointwise2_bias is None
        else np.asarray(layer13_conv_pointwise2_bias, dtype=np.float32)
    )
    layer13_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer13_ffn2_norm_weight is None
        else np.asarray(layer13_ffn2_norm_weight, dtype=np.float32)
    )
    layer13_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_ffn2_norm_bias is None
        else np.asarray(layer13_ffn2_norm_bias, dtype=np.float32)
    )
    layer13_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer13_ffn2_intermediate_weight is None
        else np.asarray(layer13_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer13_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer13_ffn2_intermediate_bias is None
        else np.asarray(layer13_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer13_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer13_ffn2_output_weight is None
        else np.asarray(layer13_ffn2_output_weight, dtype=np.float32)
    )
    layer13_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer13_ffn2_output_bias is None
        else np.asarray(layer13_ffn2_output_bias, dtype=np.float32)
    )
    layer14_ffn1_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer14_ffn1_norm_weight is None
        else np.asarray(layer14_ffn1_norm_weight, dtype=np.float32)
    )
    layer14_ffn1_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer14_ffn1_norm_bias is None
        else np.asarray(layer14_ffn1_norm_bias, dtype=np.float32)
    )
    layer14_ffn1_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer14_ffn1_intermediate_weight is None
        else np.asarray(layer14_ffn1_intermediate_weight, dtype=np.float32)
    )
    layer14_ffn1_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer14_ffn1_intermediate_bias is None
        else np.asarray(layer14_ffn1_intermediate_bias, dtype=np.float32)
    )
    layer14_ffn1_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer14_ffn1_output_weight is None
        else np.asarray(layer14_ffn1_output_weight, dtype=np.float32)
    )
    layer14_ffn1_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer14_ffn1_output_bias is None
        else np.asarray(layer14_ffn1_output_bias, dtype=np.float32)
    )
    layer14_ffn2_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer14_ffn2_norm_weight is None
        else np.asarray(layer14_ffn2_norm_weight, dtype=np.float32)
    )
    layer14_ffn2_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer14_ffn2_norm_bias is None
        else np.asarray(layer14_ffn2_norm_bias, dtype=np.float32)
    )
    layer14_ffn2_intermediate_weight_arr = (
        np.ones((4096, 1024), dtype=np.float32)
        if layer14_ffn2_intermediate_weight is None
        else np.asarray(layer14_ffn2_intermediate_weight, dtype=np.float32)
    )
    layer14_ffn2_intermediate_bias_arr = (
        np.zeros((4096,), dtype=np.float32)
        if layer14_ffn2_intermediate_bias is None
        else np.asarray(layer14_ffn2_intermediate_bias, dtype=np.float32)
    )
    layer14_ffn2_output_weight_arr = (
        np.ones((1024, 4096), dtype=np.float32)
        if layer14_ffn2_output_weight is None
        else np.asarray(layer14_ffn2_output_weight, dtype=np.float32)
    )
    layer14_ffn2_output_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer14_ffn2_output_bias is None
        else np.asarray(layer14_ffn2_output_bias, dtype=np.float32)
    )
    layer17_final_norm_weight_arr = (
        np.ones((1024,), dtype=np.float32)
        if layer17_final_norm_weight is None
        else np.asarray(layer17_final_norm_weight, dtype=np.float32)
    )
    layer17_final_norm_bias_arr = (
        np.zeros((1024,), dtype=np.float32)
        if layer17_final_norm_bias is None
        else np.asarray(layer17_final_norm_bias, dtype=np.float32)
    )
    mean = np.zeros((1024,), dtype=np.float32) if stats_mean is None else np.asarray(stats_mean, dtype=np.float32)
    std = np.ones((1024,), dtype=np.float32) if stats_std is None else np.asarray(stats_std, dtype=np.float32)
    write_bundle(
        bundle_dir,
        [
            ("w2v_bert.feature_projection.layer_norm.weight", ln_weight, "w2v_bert"),
            ("w2v_bert.feature_projection.layer_norm.bias", ln_bias, "w2v_bert"),
            (
                "w2v_bert.feature_projection.projection.weight",
                proj_weight,
                "w2v_bert",
            ),
            ("w2v_bert.feature_projection.projection.bias", proj_bias, "w2v_bert"),
            ("w2v_bert.encoder.layers.0.ffn1_layer_norm.weight", ffn1_norm_weight, "w2v_bert"),
            ("w2v_bert.encoder.layers.0.ffn1_layer_norm.bias", ffn1_norm_bias, "w2v_bert"),
            (
                "w2v_bert.encoder.layers.0.ffn1.intermediate_dense.weight",
                ffn1_intermediate_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn1.intermediate_dense.bias",
                ffn1_intermediate_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn1.output_dense.weight",
                ffn1_output_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn1.output_dense.bias",
                ffn1_output_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.self_attn.linear_q.weight",
                q_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.self_attn.linear_k.weight",
                k_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.self_attn.linear_v.weight",
                v_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.self_attn.linear_out.weight",
                out_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.self_attn_layer_norm.weight",
                self_attn_norm_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.self_attn_layer_norm.bias",
                self_attn_norm_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.layer_norm.weight",
                conv_norm_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.layer_norm.bias",
                conv_norm_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.weight",
                conv_pointwise1_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.pointwise_conv1.bias",
                conv_pointwise1_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.weight",
                conv_depthwise_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.depthwise_conv.bias",
                conv_depthwise_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.weight",
                conv_depthwise_norm_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.depthwise_layer_norm.bias",
                conv_depthwise_norm_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.weight",
                conv_pointwise2_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.conv_module.pointwise_conv2.bias",
                conv_pointwise2_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn2_layer_norm.weight",
                ffn2_norm_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn2_layer_norm.bias",
                ffn2_norm_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.weight",
                ffn2_intermediate_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn2.intermediate_dense.bias",
                ffn2_intermediate_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn2.output_dense.weight",
                ffn2_output_weight,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.ffn2.output_dense.bias",
                ffn2_output_bias,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.final_layer_norm.weight",
                layer0_final_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.0.final_layer_norm.bias",
                layer0_final_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn1_layer_norm.weight",
                layer1_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn1_layer_norm.bias",
                layer1_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn1.intermediate_dense.weight",
                layer1_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn1.intermediate_dense.bias",
                layer1_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn1.output_dense.weight",
                layer1_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn1.output_dense.bias",
                layer1_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.self_attn.linear_q.weight",
                layer1_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.self_attn.linear_k.weight",
                layer1_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.self_attn.linear_v.weight",
                layer1_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.self_attn.linear_out.weight",
                layer1_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.self_attn_layer_norm.weight",
                layer1_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.self_attn_layer_norm.bias",
                layer1_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.layer_norm.weight",
                layer1_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.layer_norm.bias",
                layer1_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.weight",
                layer1_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.pointwise_conv1.bias",
                layer1_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.weight",
                layer1_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.depthwise_conv.bias",
                layer1_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.weight",
                layer1_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.depthwise_layer_norm.bias",
                layer1_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.weight",
                layer1_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.conv_module.pointwise_conv2.bias",
                layer1_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn2_layer_norm.weight",
                layer1_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn2_layer_norm.bias",
                layer1_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.weight",
                layer1_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn2.intermediate_dense.bias",
                layer1_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn2.output_dense.weight",
                layer1_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.ffn2.output_dense.bias",
                layer1_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.final_layer_norm.weight",
                layer1_final_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.1.final_layer_norm.bias",
                layer1_final_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn1_layer_norm.weight",
                layer2_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn1_layer_norm.bias",
                layer2_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn1.intermediate_dense.weight",
                layer2_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn1.intermediate_dense.bias",
                layer2_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn1.output_dense.weight",
                layer2_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn1.output_dense.bias",
                layer2_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.self_attn.linear_q.weight",
                layer2_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.self_attn.linear_k.weight",
                layer2_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.self_attn.linear_v.weight",
                layer2_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.self_attn.linear_out.weight",
                layer2_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.self_attn_layer_norm.weight",
                layer2_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.self_attn_layer_norm.bias",
                layer2_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.layer_norm.weight",
                layer2_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.layer_norm.bias",
                layer2_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.pointwise_conv1.weight",
                layer2_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.pointwise_conv1.bias",
                layer2_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.depthwise_conv.weight",
                layer2_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.depthwise_conv.bias",
                layer2_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.depthwise_layer_norm.weight",
                layer2_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.depthwise_layer_norm.bias",
                layer2_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.pointwise_conv2.weight",
                layer2_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.conv_module.pointwise_conv2.bias",
                layer2_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            ("w2v_bert.encoder.layers.2.ffn2_layer_norm.weight", layer2_ffn2_norm_weight_arr, "w2v_bert"),
            ("w2v_bert.encoder.layers.2.ffn2_layer_norm.bias", layer2_ffn2_norm_bias_arr, "w2v_bert"),
            (
                "w2v_bert.encoder.layers.2.ffn2.intermediate_dense.weight",
                layer2_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn2.intermediate_dense.bias",
                layer2_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn2.output_dense.weight",
                layer2_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.2.ffn2.output_dense.bias",
                layer2_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            ("w2v_bert.encoder.layers.3.ffn1_layer_norm.weight", layer3_ffn1_norm_weight_arr, "w2v_bert"),
            ("w2v_bert.encoder.layers.3.ffn1_layer_norm.bias", layer3_ffn1_norm_bias_arr, "w2v_bert"),
            (
                "w2v_bert.encoder.layers.3.ffn1.intermediate_dense.weight",
                layer3_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn1.intermediate_dense.bias",
                layer3_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn1.output_dense.weight",
                layer3_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn1.output_dense.bias",
                layer3_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.self_attn.linear_q.weight",
                layer3_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.self_attn.linear_k.weight",
                layer3_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.self_attn.linear_v.weight",
                layer3_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.self_attn.linear_out.weight",
                layer3_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.self_attn_layer_norm.weight",
                layer3_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.self_attn_layer_norm.bias",
                layer3_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.layer_norm.weight",
                layer3_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.layer_norm.bias",
                layer3_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.pointwise_conv1.weight",
                layer3_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.pointwise_conv1.bias",
                layer3_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.depthwise_conv.weight",
                layer3_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.depthwise_conv.bias",
                layer3_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.depthwise_layer_norm.weight",
                layer3_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.depthwise_layer_norm.bias",
                layer3_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.pointwise_conv2.weight",
                layer3_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.conv_module.pointwise_conv2.bias",
                layer3_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn2_layer_norm.weight",
                layer3_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn2_layer_norm.bias",
                layer3_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn2.intermediate_dense.weight",
                layer3_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn2.intermediate_dense.bias",
                layer3_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn2.output_dense.weight",
                layer3_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.ffn2.output_dense.bias",
                layer3_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.final_layer_norm.weight",
                layer3_final_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.3.final_layer_norm.bias",
                layer3_final_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn1_layer_norm.weight",
                layer4_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn1_layer_norm.bias",
                layer4_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn1.intermediate_dense.weight",
                layer4_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn1.intermediate_dense.bias",
                layer4_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn1.output_dense.weight",
                layer4_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn1.output_dense.bias",
                layer4_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.self_attn.linear_q.weight",
                layer4_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.self_attn.linear_k.weight",
                layer4_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.self_attn.linear_v.weight",
                layer4_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.self_attn.linear_out.weight",
                layer4_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.self_attn_layer_norm.weight",
                layer4_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.self_attn_layer_norm.bias",
                layer4_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.layer_norm.weight",
                layer4_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.layer_norm.bias",
                layer4_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.pointwise_conv1.weight",
                layer4_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.pointwise_conv1.bias",
                layer4_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.depthwise_conv.weight",
                layer4_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.depthwise_conv.bias",
                layer4_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.depthwise_layer_norm.weight",
                layer4_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.depthwise_layer_norm.bias",
                layer4_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.pointwise_conv2.weight",
                layer4_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.conv_module.pointwise_conv2.bias",
                layer4_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn2_layer_norm.weight",
                layer4_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn2_layer_norm.bias",
                layer4_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn2.intermediate_dense.weight",
                layer4_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn2.intermediate_dense.bias",
                layer4_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn2.output_dense.weight",
                layer4_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.4.ffn2.output_dense.bias",
                layer4_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn1_layer_norm.weight",
                layer5_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn1_layer_norm.bias",
                layer5_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn1.intermediate_dense.weight",
                layer5_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn1.intermediate_dense.bias",
                layer5_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn1.output_dense.weight",
                layer5_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn1.output_dense.bias",
                layer5_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.self_attn.linear_q.weight",
                layer5_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.self_attn.linear_k.weight",
                layer5_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.self_attn.linear_v.weight",
                layer5_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.self_attn.linear_out.weight",
                layer5_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.self_attn_layer_norm.weight",
                layer5_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.self_attn_layer_norm.bias",
                layer5_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.layer_norm.weight",
                layer5_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.layer_norm.bias",
                layer5_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.pointwise_conv1.weight",
                layer5_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.pointwise_conv1.bias",
                layer5_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.depthwise_conv.weight",
                layer5_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.depthwise_conv.bias",
                layer5_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.depthwise_layer_norm.weight",
                layer5_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.depthwise_layer_norm.bias",
                layer5_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.pointwise_conv2.weight",
                layer5_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.conv_module.pointwise_conv2.bias",
                layer5_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn2_layer_norm.weight",
                layer5_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn2_layer_norm.bias",
                layer5_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn2.intermediate_dense.weight",
                layer5_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn2.intermediate_dense.bias",
                layer5_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn2.output_dense.weight",
                layer5_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.5.ffn2.output_dense.bias",
                layer5_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn1_layer_norm.weight",
                layer6_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn1_layer_norm.bias",
                layer6_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn1.intermediate_dense.weight",
                layer6_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn1.intermediate_dense.bias",
                layer6_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn1.output_dense.weight",
                layer6_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn1.output_dense.bias",
                layer6_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.self_attn.linear_q.weight",
                layer6_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.self_attn.linear_k.weight",
                layer6_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.self_attn.linear_v.weight",
                layer6_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.self_attn.linear_out.weight",
                layer6_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.self_attn_layer_norm.weight",
                layer6_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.self_attn_layer_norm.bias",
                layer6_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.layer_norm.weight",
                layer6_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.layer_norm.bias",
                layer6_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.pointwise_conv1.weight",
                layer6_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.pointwise_conv1.bias",
                layer6_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.depthwise_conv.weight",
                layer6_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.depthwise_conv.bias",
                layer6_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.depthwise_layer_norm.weight",
                layer6_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.depthwise_layer_norm.bias",
                layer6_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.pointwise_conv2.weight",
                layer6_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.conv_module.pointwise_conv2.bias",
                layer6_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn2_layer_norm.weight",
                layer6_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn2_layer_norm.bias",
                layer6_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn2.intermediate_dense.weight",
                layer6_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn2.intermediate_dense.bias",
                layer6_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn2.output_dense.weight",
                layer6_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.6.ffn2.output_dense.bias",
                layer6_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn1_layer_norm.weight",
                layer7_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn1_layer_norm.bias",
                layer7_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn1.intermediate_dense.weight",
                layer7_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn1.intermediate_dense.bias",
                layer7_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn1.output_dense.weight",
                layer7_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn1.output_dense.bias",
                layer7_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.self_attn.linear_q.weight",
                layer7_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.self_attn.linear_k.weight",
                layer7_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.self_attn.linear_v.weight",
                layer7_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.self_attn.linear_out.weight",
                layer7_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.self_attn_layer_norm.weight",
                layer7_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.self_attn_layer_norm.bias",
                layer7_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.layer_norm.weight",
                layer7_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.layer_norm.bias",
                layer7_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.pointwise_conv1.weight",
                layer7_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.pointwise_conv1.bias",
                layer7_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.depthwise_conv.weight",
                layer7_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.depthwise_conv.bias",
                layer7_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.depthwise_layer_norm.weight",
                layer7_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.depthwise_layer_norm.bias",
                layer7_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.pointwise_conv2.weight",
                layer7_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.conv_module.pointwise_conv2.bias",
                layer7_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn2_layer_norm.weight",
                layer7_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn2_layer_norm.bias",
                layer7_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn2.intermediate_dense.weight",
                layer7_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn2.intermediate_dense.bias",
                layer7_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn2.output_dense.weight",
                layer7_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.7.ffn2.output_dense.bias",
                layer7_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn1_layer_norm.weight",
                layer8_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn1_layer_norm.bias",
                layer8_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn1.intermediate_dense.weight",
                layer8_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn1.intermediate_dense.bias",
                layer8_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn1.output_dense.weight",
                layer8_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn1.output_dense.bias",
                layer8_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.self_attn.linear_q.weight",
                layer8_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.self_attn.linear_k.weight",
                layer8_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.self_attn.linear_v.weight",
                layer8_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.self_attn.linear_out.weight",
                layer8_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.self_attn_layer_norm.weight",
                layer8_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.self_attn_layer_norm.bias",
                layer8_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.layer_norm.weight",
                layer8_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.layer_norm.bias",
                layer8_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.pointwise_conv1.weight",
                layer8_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.pointwise_conv1.bias",
                layer8_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.depthwise_conv.weight",
                layer8_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.depthwise_conv.bias",
                layer8_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.depthwise_layer_norm.weight",
                layer8_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.depthwise_layer_norm.bias",
                layer8_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.pointwise_conv2.weight",
                layer8_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.conv_module.pointwise_conv2.bias",
                layer8_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn2_layer_norm.weight",
                layer8_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn2_layer_norm.bias",
                layer8_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn2.intermediate_dense.weight",
                layer8_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn2.intermediate_dense.bias",
                layer8_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn2.output_dense.weight",
                layer8_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.8.ffn2.output_dense.bias",
                layer8_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn1_layer_norm.weight",
                layer9_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn1_layer_norm.bias",
                layer9_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn1.intermediate_dense.weight",
                layer9_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn1.intermediate_dense.bias",
                layer9_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn1.output_dense.weight",
                layer9_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn1.output_dense.bias",
                layer9_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.self_attn.linear_q.weight",
                layer9_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.self_attn.linear_k.weight",
                layer9_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.self_attn.linear_v.weight",
                layer9_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.self_attn.linear_out.weight",
                layer9_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.self_attn_layer_norm.weight",
                layer9_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.self_attn_layer_norm.bias",
                layer9_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.layer_norm.weight",
                layer9_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.layer_norm.bias",
                layer9_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.pointwise_conv1.weight",
                layer9_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.pointwise_conv1.bias",
                layer9_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.depthwise_conv.weight",
                layer9_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.depthwise_conv.bias",
                layer9_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.depthwise_layer_norm.weight",
                layer9_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.depthwise_layer_norm.bias",
                layer9_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.pointwise_conv2.weight",
                layer9_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.conv_module.pointwise_conv2.bias",
                layer9_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn2_layer_norm.weight",
                layer9_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn2_layer_norm.bias",
                layer9_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn2.intermediate_dense.weight",
                layer9_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn2.intermediate_dense.bias",
                layer9_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn2.output_dense.weight",
                layer9_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.9.ffn2.output_dense.bias",
                layer9_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn1_layer_norm.weight",
                layer10_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn1_layer_norm.bias",
                layer10_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.weight",
                layer10_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn1.intermediate_dense.bias",
                layer10_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn1.output_dense.weight",
                layer10_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn1.output_dense.bias",
                layer10_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.self_attn.linear_q.weight",
                layer10_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.self_attn.linear_k.weight",
                layer10_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.self_attn.linear_v.weight",
                layer10_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.self_attn.linear_out.weight",
                layer10_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.self_attn_layer_norm.weight",
                layer10_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.self_attn_layer_norm.bias",
                layer10_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.layer_norm.weight",
                layer10_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.layer_norm.bias",
                layer10_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.weight",
                layer10_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.pointwise_conv1.bias",
                layer10_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.weight",
                layer10_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.depthwise_conv.bias",
                layer10_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.weight",
                layer10_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.depthwise_layer_norm.bias",
                layer10_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.weight",
                layer10_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.conv_module.pointwise_conv2.bias",
                layer10_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn2_layer_norm.weight",
                layer10_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn2_layer_norm.bias",
                layer10_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.weight",
                layer10_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn2.intermediate_dense.bias",
                layer10_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn2.output_dense.weight",
                layer10_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.10.ffn2.output_dense.bias",
                layer10_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn1_layer_norm.weight",
                layer11_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn1_layer_norm.bias",
                layer11_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.weight",
                layer11_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn1.intermediate_dense.bias",
                layer11_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn1.output_dense.weight",
                layer11_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn1.output_dense.bias",
                layer11_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.self_attn.linear_q.weight",
                layer11_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.self_attn.linear_k.weight",
                layer11_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.self_attn.linear_v.weight",
                layer11_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.self_attn.linear_out.weight",
                layer11_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.self_attn_layer_norm.weight",
                layer11_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.self_attn_layer_norm.bias",
                layer11_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.layer_norm.weight",
                layer11_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.layer_norm.bias",
                layer11_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.weight",
                layer11_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.pointwise_conv1.bias",
                layer11_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.weight",
                layer11_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.depthwise_conv.bias",
                layer11_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.weight",
                layer11_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.depthwise_layer_norm.bias",
                layer11_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.weight",
                layer11_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.conv_module.pointwise_conv2.bias",
                layer11_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn2_layer_norm.weight",
                layer11_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn2_layer_norm.bias",
                layer11_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.weight",
                layer11_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn2.intermediate_dense.bias",
                layer11_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn2.output_dense.weight",
                layer11_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.11.ffn2.output_dense.bias",
                layer11_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn1_layer_norm.weight",
                layer12_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn1_layer_norm.bias",
                layer12_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.weight",
                layer12_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn1.intermediate_dense.bias",
                layer12_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn1.output_dense.weight",
                layer12_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn1.output_dense.bias",
                layer12_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.self_attn.linear_q.weight",
                layer12_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.self_attn.linear_k.weight",
                layer12_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.self_attn.linear_v.weight",
                layer12_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.self_attn.linear_out.weight",
                layer12_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.self_attn_layer_norm.weight",
                layer12_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.self_attn_layer_norm.bias",
                layer12_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.layer_norm.weight",
                layer12_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.layer_norm.bias",
                layer12_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.weight",
                layer12_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.pointwise_conv1.bias",
                layer12_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.weight",
                layer12_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.depthwise_conv.bias",
                layer12_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.weight",
                layer12_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.depthwise_layer_norm.bias",
                layer12_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.weight",
                layer12_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.conv_module.pointwise_conv2.bias",
                layer12_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn2_layer_norm.weight",
                layer12_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn2_layer_norm.bias",
                layer12_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.weight",
                layer12_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn2.intermediate_dense.bias",
                layer12_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn2.output_dense.weight",
                layer12_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.12.ffn2.output_dense.bias",
                layer12_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn1_layer_norm.weight",
                layer13_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn1_layer_norm.bias",
                layer13_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.weight",
                layer13_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn1.intermediate_dense.bias",
                layer13_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn1.output_dense.weight",
                layer13_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn1.output_dense.bias",
                layer13_ffn1_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.self_attn.linear_q.weight",
                layer13_q_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.self_attn.linear_k.weight",
                layer13_k_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.self_attn.linear_v.weight",
                layer13_v_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.self_attn.linear_out.weight",
                layer13_out_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.self_attn_layer_norm.weight",
                layer13_self_attn_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.self_attn_layer_norm.bias",
                layer13_self_attn_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.layer_norm.weight",
                layer13_conv_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.layer_norm.bias",
                layer13_conv_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.weight",
                layer13_conv_pointwise1_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.pointwise_conv1.bias",
                layer13_conv_pointwise1_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.weight",
                layer13_conv_depthwise_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.depthwise_conv.bias",
                layer13_conv_depthwise_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.weight",
                layer13_conv_depthwise_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.depthwise_layer_norm.bias",
                layer13_conv_depthwise_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.weight",
                layer13_conv_pointwise2_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.conv_module.pointwise_conv2.bias",
                layer13_conv_pointwise2_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn2_layer_norm.weight",
                layer13_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn2_layer_norm.bias",
                layer13_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.weight",
                layer13_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn2.intermediate_dense.bias",
                layer13_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn2.output_dense.weight",
                layer13_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.13.ffn2.output_dense.bias",
                layer13_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn1_layer_norm.weight",
                layer14_ffn1_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn1_layer_norm.bias",
                layer14_ffn1_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.weight",
                layer14_ffn1_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn1.intermediate_dense.bias",
                layer14_ffn1_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn1.output_dense.weight",
                layer14_ffn1_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn1.output_dense.bias",
                layer14_ffn1_output_bias_arr,
                "w2v_bert",
            ),

            (
                "w2v_bert.encoder.layers.14.ffn2_layer_norm.weight",
                layer14_ffn2_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn2_layer_norm.bias",
                layer14_ffn2_norm_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn2.intermediate_dense.weight",
                layer14_ffn2_intermediate_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn2.intermediate_dense.bias",
                layer14_ffn2_intermediate_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn2.output_dense.weight",
                layer14_ffn2_output_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.14.ffn2.output_dense.bias",
                layer14_ffn2_output_bias_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.17.final_layer_norm.weight",
                layer17_final_norm_weight_arr,
                "w2v_bert",
            ),
            (
                "w2v_bert.encoder.layers.17.final_layer_norm.bias",
                layer17_final_norm_bias_arr,
                "w2v_bert",
            ),
            ("w2v_bert.stats.mean", mean, "w2v_bert"),
            ("w2v_bert.stats.std", std, "w2v_bert"),
        ],
        metadata={"unit": "w2v-bert-readiness"},
    )


def test_mit2_tts_capabilities_exposes_machine_readable_product_surface():
    report = _run_json("--capabilities")

    assert report["stage"] == "tts_launcher_capabilities"
    assert report["ok"] is True
    assert report["product_surface_version"] == 1
    assert report["binary"] == "mit2_tts"
    assert report["native_text_surface"] == "focused_cjk_limited_ascii"
    assert report["supports_cached_voice_tts_cjk"] is True
    assert report["supports_cached_voice_tts_general_text"] is False
    assert report["supports_native_voice_clone"] is False
    assert report["supported_product_surfaces"] == ["cached_voice_cjk_text_to_wav"]
    assert "cached_voice_general_text_to_wav" in report["unsupported_product_surfaces"]
    assert "native_clone_audio_to_voice_bundle" in report["unsupported_product_surfaces"]
    assert report["default_preset"] == "standard"
    assert [preset["name"] for preset in report["presets"]] == ["smoke", "short", "standard"]
    assert report["synthesis_preflight_command"] == "--preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT"
    assert "--preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT" in report["native_product_commands"]
    audit = report["start_sh_replacement_audit"]
    assert audit["can_replace_start_sh_cached_voice_cjk"] is True
    assert audit["can_replace_start_sh_full_clone_tts"] is False
    assert {gap["surface"] for gap in audit["missing_surfaces"]} == {
        "cached_voice_general_text_to_wav",
        "native_clone_audio_to_voice_bundle",
        "production_length_resource_planning",
    }
    assert "--readiness MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR" in audit["recommended_preflight_commands"]
    assert "--preflight MODEL_BUNDLE_DIR VOICE_BUNDLE_DIR TEXT" in audit["recommended_preflight_commands"]
    assert report["clone_audio_preflight_command"] == "--clone-preflight AUDIO_WAV"
    assert report["clone_audio_preprocess_command"] == "--clone-preprocess AUDIO_WAV OUTPUT_F32"
    assert report["clone_readiness_command"] == "--clone-readiness PREPROCESS_MANIFEST"
    assert report["clone_mel_extraction_command"] == "--clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32"
    assert report["clone_fbank_extraction_command"] == "--clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32"
    assert report["clone_feature_prepare_command"] == "--clone-prepare-features AUDIO_WAV OUTPUT_DIR"
    assert report["clone_feature_readiness_command"] == "--clone-feature-readiness FEATURE_MANIFEST"
    assert report["clone_command"] == "--clone AUDIO_WAV OUTPUT_VOICE_BUNDLE"
    assert report["clone_encoder_model_readiness_command"] == "--clone-encoder-model-readiness MODEL_BUNDLE_DIR"
    assert (
        report["clone_campplus_style_readiness_command"]
        == "--clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST S2MEL_STYLE_F32"
    )
    assert (
        report["clone_campplus_style_from_features_command"]
        == "--clone-campplus-style-from-features MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32"
    )
    assert (
        report["clone_campplus_head_golden_command"]
        == "--clone-campplus-head-golden MODEL_BUNDLE_DIR FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR"
    )
    assert (
        report["clone_w2v_feature_project_command"]
        == "--clone-w2v-feature-project MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32"
    )
    assert (
        report["clone_w2v_layer0_ffn1_norm_command"]
        == "--clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32"
    )
    assert (
        report["clone_w2v_layer0_ffn1_intermediate_command"]
        == "--clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32"
    )
    assert (
        report["clone_w2v_layer0_ffn1_activate_command"]
        == "--clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32"
    )
    assert (
        report["clone_w2v_layer0_ffn1_output_command"]
        == "--clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32"
    )
    assert (
        report["clone_w2v_layer0_ffn1_residual_command"]
        == "--clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32 W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer0_qkv_command"]
        == "--clone-w2v-layer0-qkv MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer0_attention_command"]
        == "--clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer0_attention_project_command"]
        == "--clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer0_attention_residual_command"]
        == "--clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32 W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer0_attention_norm_command"]
        == "--clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer0_conv_norm_command"]
        == "--clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer0_conv_glu_command"]
        == "--clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32 W2V_TOKENS OUTPUT_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer0_conv_depthwise_command"]
        == "--clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer0_conv_residual_command"]
        == "--clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer0_ffn2_residual_command"]
        == "--clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer0_final_norm_command"]
        == "--clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32"
    )
    assert (
        report["clone_w2v_layer1_ffn1_norm_command"]
        == "--clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32"
    )
    assert (
        report["clone_w2v_layer1_ffn1_intermediate_command"]
        == "--clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32"
    )
    assert (
        report["clone_w2v_layer1_ffn1_activate_command"]
        == "--clone-w2v-layer1-ffn1-activate W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_ACTIVATED_F32"
    )
    assert (
        report["clone_w2v_layer1_ffn1_output_command"]
        == "--clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32"
    )
    assert (
        report["clone_w2v_layer1_ffn1_residual_command"]
        == "--clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32 W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer1_qkv_command"]
        == "--clone-w2v-layer1-qkv MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer1_attention_command"]
        == "--clone-w2v-layer1-attention W2V_LAYER1_Q_F32 W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER1_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer1_attention_project_command"]
        == "--clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer1_attention_residual_command"]
        == "--clone-w2v-layer1-attention-residual W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer1_attention_norm_command"]
        == "--clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer1_conv_norm_command"]
        == "--clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer1_conv_glu_command"]
        == "--clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer1_conv_depthwise_command"]
        == "--clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer1_conv_residual_command"]
        == "--clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer1_ffn2_residual_command"]
        == "--clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer1_final_norm_command"]
        == "--clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32"
    )
    assert (
        report["clone_w2v_layer2_ffn1_norm_command"]
        == "--clone-w2v-layer2-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER1_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_NORM_F32"
    )
    assert (
        report["clone_w2v_layer2_ffn1_intermediate_command"]
        == "--clone-w2v-layer2-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_INTERMEDIATE_F32"
    )
    assert (
        report["clone_w2v_layer2_ffn1_activate_command"]
        == "--clone-w2v-layer2-ffn1-activate W2V_LAYER2_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_ACTIVATED_F32"
    )
    assert (
        report["clone_w2v_layer2_ffn1_output_command"]
        == "--clone-w2v-layer2-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_OUTPUT_F32"
    )
    assert (
        report["clone_w2v_layer2_ffn1_residual_command"]
        == "--clone-w2v-layer2-ffn1-residual W2V_LAYER1_F32 W2V_LAYER2_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER2_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer2_qkv_command"]
        == "--clone-w2v-layer2-qkv MODEL_BUNDLE_DIR W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer2_attention_command"]
        == "--clone-w2v-layer2-attention W2V_LAYER2_Q_F32 W2V_LAYER2_K_F32 W2V_LAYER2_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER2_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer2_attention_project_command"]
        == "--clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer2_attention_residual_command"]
        == "--clone-w2v-layer2-attention-residual W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer2_attention_norm_command"]
        == "--clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer2_conv_norm_command"]
        == "--clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer2_conv_glu_command"]
        == "--clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer2_conv_depthwise_command"]
        == "--clone-w2v-layer2-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER2_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer2_conv_residual_command"]
        == "--clone-w2v-layer2-conv-residual MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_LAYER2_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer2_ffn2_residual_command"]
        == "--clone-w2v-layer2-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER2_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer3_ffn1_norm_command"]
        == "--clone-w2v-layer3-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_NORM_F32"
    )
    assert (
        report["clone_w2v_layer3_ffn1_intermediate_command"]
        == "--clone-w2v-layer3-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_INTERMEDIATE_F32"
    )
    assert (
        report["clone_w2v_layer3_ffn1_activate_command"]
        == "--clone-w2v-layer3-ffn1-activate W2V_LAYER3_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_ACTIVATED_F32"
    )
    assert (
        report["clone_w2v_layer3_ffn1_output_command"]
        == "--clone-w2v-layer3-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_OUTPUT_F32"
    )
    assert (
        report["clone_w2v_layer3_ffn1_residual_command"]
        == "--clone-w2v-layer3-ffn1-residual W2V_LAYER2_FFN2_RESIDUAL_F32 W2V_LAYER3_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER3_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer3_qkv_command"]
        == "--clone-w2v-layer3-qkv MODEL_BUNDLE_DIR W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer3_attention_command"]
        == "--clone-w2v-layer3-attention W2V_LAYER3_Q_F32 W2V_LAYER3_K_F32 W2V_LAYER3_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER3_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer3_attention_project_command"]
        == "--clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer3_attention_residual_command"]
        == "--clone-w2v-layer3-attention-residual W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer3_attention_norm_command"]
        == "--clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer3_conv_norm_command"]
        == "--clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer3_conv_glu_command"]
        == "--clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer3_conv_depthwise_command"]
        == "--clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer3_conv_residual_command"]
        == "--clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer3_ffn2_residual_command"]
        == "--clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer3_final_norm_command"]
        == "--clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32"
    )
    assert (
        report["clone_w2v_layer4_ffn1_norm_command"]
        == "--clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32"
    )
    assert (
        report["clone_w2v_layer4_ffn1_intermediate_command"]
        == "--clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32"
    )
    assert (
        report["clone_w2v_layer4_ffn1_activate_command"]
        == "--clone-w2v-layer4-ffn1-activate W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_ACTIVATED_F32"
    )
    assert (
        report["clone_w2v_layer4_ffn1_output_command"]
        == "--clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32"
    )
    assert (
        report["clone_w2v_layer4_ffn1_residual_command"]
        == "--clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32 W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer4_qkv_command"]
        == "--clone-w2v-layer4-qkv MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer4_attention_command"]
        == "--clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32 W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER4_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer4_attention_project_command"]
        == "--clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer4_attention_residual_command"]
        == "--clone-w2v-layer4-attention-residual W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer4_attention_norm_command"]
        == "--clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer4_conv_norm_command"]
        == "--clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer4_conv_glu_command"]
        == "--clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer4_conv_depthwise_command"]
        == "--clone-w2v-layer4-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER4_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer4_conv_residual_command"]
        == "--clone-w2v-layer4-conv-residual MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_LAYER4_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer4_ffn2_residual_command"]
        == "--clone-w2v-layer4-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER4_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer5_ffn1_residual_command"]
        == "--clone-w2v-layer5-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER4_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer5_qkv_command"]
        == "--clone-w2v-layer5-qkv MODEL_BUNDLE_DIR W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer5_attention_command"]
        == "--clone-w2v-layer5-attention W2V_LAYER5_Q_F32 W2V_LAYER5_K_F32 W2V_LAYER5_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER5_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer5_attention_project_command"]
        == "--clone-w2v-layer5-attention-project MODEL_BUNDLE_DIR W2V_LAYER5_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer5_attention_residual_command"]
        == "--clone-w2v-layer5-attention-residual W2V_LAYER5_FFN1_RESIDUAL_F32 W2V_LAYER5_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer5_attention_norm_command"]
        == "--clone-w2v-layer5-attention-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer5_conv_norm_command"]
        == "--clone-w2v-layer5-conv-norm MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer5_conv_glu_command"]
        == "--clone-w2v-layer5-conv-glu MODEL_BUNDLE_DIR W2V_LAYER5_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer5_conv_depthwise_command"]
        == "--clone-w2v-layer5-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER5_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer5_conv_residual_command"]
        == "--clone-w2v-layer5-conv-residual MODEL_BUNDLE_DIR W2V_LAYER5_ATTENTION_NORM_F32 W2V_LAYER5_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER5_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer5_ffn2_residual_command"]
        == "--clone-w2v-layer5-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER5_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER5_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer6_ffn1_residual_command"]
        == "--clone-w2v-layer6-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER5_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer6_qkv_command"]
        == "--clone-w2v-layer6-qkv MODEL_BUNDLE_DIR W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer6_attention_command"]
        == "--clone-w2v-layer6-attention W2V_LAYER6_Q_F32 W2V_LAYER6_K_F32 W2V_LAYER6_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER6_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer6_attention_project_command"]
        == "--clone-w2v-layer6-attention-project MODEL_BUNDLE_DIR W2V_LAYER6_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer6_attention_residual_command"]
        == "--clone-w2v-layer6-attention-residual W2V_LAYER6_FFN1_RESIDUAL_F32 W2V_LAYER6_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer6_attention_norm_command"]
        == "--clone-w2v-layer6-attention-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer6_conv_norm_command"]
        == "--clone-w2v-layer6-conv-norm MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer6_conv_glu_command"]
        == "--clone-w2v-layer6-conv-glu MODEL_BUNDLE_DIR W2V_LAYER6_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer6_conv_depthwise_command"]
        == "--clone-w2v-layer6-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER6_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer6_conv_residual_command"]
        == "--clone-w2v-layer6-conv-residual MODEL_BUNDLE_DIR W2V_LAYER6_ATTENTION_NORM_F32 W2V_LAYER6_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER6_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer6_ffn2_residual_command"]
        == "--clone-w2v-layer6-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER6_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER6_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer7_ffn1_residual_command"]
        == "--clone-w2v-layer7-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER6_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer7_qkv_command"]
        == "--clone-w2v-layer7-qkv MODEL_BUNDLE_DIR W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer7_attention_command"]
        == "--clone-w2v-layer7-attention W2V_LAYER7_Q_F32 W2V_LAYER7_K_F32 W2V_LAYER7_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER7_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer7_attention_project_command"]
        == "--clone-w2v-layer7-attention-project MODEL_BUNDLE_DIR W2V_LAYER7_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer7_attention_residual_command"]
        == "--clone-w2v-layer7-attention-residual W2V_LAYER7_FFN1_RESIDUAL_F32 W2V_LAYER7_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer7_attention_norm_command"]
        == "--clone-w2v-layer7-attention-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer7_conv_norm_command"]
        == "--clone-w2v-layer7-conv-norm MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer7_conv_glu_command"]
        == "--clone-w2v-layer7-conv-glu MODEL_BUNDLE_DIR W2V_LAYER7_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer7_conv_depthwise_command"]
        == "--clone-w2v-layer7-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER7_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer7_conv_residual_command"]
        == "--clone-w2v-layer7-conv-residual MODEL_BUNDLE_DIR W2V_LAYER7_ATTENTION_NORM_F32 W2V_LAYER7_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER7_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer7_ffn2_residual_command"]
        == "--clone-w2v-layer7-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER7_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER7_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer8_ffn1_residual_command"]
        == "--clone-w2v-layer8-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER7_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer8_qkv_command"]
        == "--clone-w2v-layer8-qkv MODEL_BUNDLE_DIR W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer8_attention_command"]
        == "--clone-w2v-layer8-attention W2V_LAYER8_Q_F32 W2V_LAYER8_K_F32 W2V_LAYER8_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER8_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer8_attention_project_command"]
        == "--clone-w2v-layer8-attention-project MODEL_BUNDLE_DIR W2V_LAYER8_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer8_attention_residual_command"]
        == "--clone-w2v-layer8-attention-residual W2V_LAYER8_FFN1_RESIDUAL_F32 W2V_LAYER8_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer8_attention_norm_command"]
        == "--clone-w2v-layer8-attention-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer8_conv_norm_command"]
        == "--clone-w2v-layer8-conv-norm MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer8_conv_glu_command"]
        == "--clone-w2v-layer8-conv-glu MODEL_BUNDLE_DIR W2V_LAYER8_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer8_conv_depthwise_command"]
        == "--clone-w2v-layer8-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER8_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer8_conv_residual_command"]
        == "--clone-w2v-layer8-conv-residual MODEL_BUNDLE_DIR W2V_LAYER8_ATTENTION_NORM_F32 W2V_LAYER8_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER8_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer8_ffn2_residual_command"]
        == "--clone-w2v-layer8-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER8_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER8_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer9_ffn1_residual_command"]
        == "--clone-w2v-layer9-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER8_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer9_qkv_command"]
        == "--clone-w2v-layer9-qkv MODEL_BUNDLE_DIR W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer9_attention_command"]
        == "--clone-w2v-layer9-attention W2V_LAYER9_Q_F32 W2V_LAYER9_K_F32 W2V_LAYER9_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER9_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer9_attention_project_command"]
        == "--clone-w2v-layer9-attention-project MODEL_BUNDLE_DIR W2V_LAYER9_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer9_attention_residual_command"]
        == "--clone-w2v-layer9-attention-residual W2V_LAYER9_FFN1_RESIDUAL_F32 W2V_LAYER9_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer9_attention_norm_command"]
        == "--clone-w2v-layer9-attention-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer9_conv_norm_command"]
        == "--clone-w2v-layer9-conv-norm MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer9_conv_glu_command"]
        == "--clone-w2v-layer9-conv-glu MODEL_BUNDLE_DIR W2V_LAYER9_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer9_conv_depthwise_command"]
        == "--clone-w2v-layer9-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER9_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer9_conv_residual_command"]
        == "--clone-w2v-layer9-conv-residual MODEL_BUNDLE_DIR W2V_LAYER9_ATTENTION_NORM_F32 W2V_LAYER9_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER9_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer9_ffn2_residual_command"]
        == "--clone-w2v-layer9-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER9_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER9_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer10_ffn1_residual_command"]
        == "--clone-w2v-layer10-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER9_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer10_qkv_command"]
        == "--clone-w2v-layer10-qkv MODEL_BUNDLE_DIR W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer10_attention_command"]
        == "--clone-w2v-layer10-attention W2V_LAYER10_Q_F32 W2V_LAYER10_K_F32 W2V_LAYER10_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER10_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer10_attention_project_command"]
        == "--clone-w2v-layer10-attention-project MODEL_BUNDLE_DIR W2V_LAYER10_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer10_attention_residual_command"]
        == "--clone-w2v-layer10-attention-residual W2V_LAYER10_FFN1_RESIDUAL_F32 W2V_LAYER10_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer10_attention_norm_command"]
        == "--clone-w2v-layer10-attention-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer10_conv_norm_command"]
        == "--clone-w2v-layer10-conv-norm MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer10_conv_glu_command"]
        == "--clone-w2v-layer10-conv-glu MODEL_BUNDLE_DIR W2V_LAYER10_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer10_conv_depthwise_command"]
        == "--clone-w2v-layer10-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER10_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer10_conv_residual_command"]
        == "--clone-w2v-layer10-conv-residual MODEL_BUNDLE_DIR W2V_LAYER10_ATTENTION_NORM_F32 W2V_LAYER10_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER10_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer10_ffn2_residual_command"]
        == "--clone-w2v-layer10-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER10_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER10_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer11_ffn1_residual_command"]
        == "--clone-w2v-layer11-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER10_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer11_qkv_command"]
        == "--clone-w2v-layer11-qkv MODEL_BUNDLE_DIR W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer11_attention_command"]
        == "--clone-w2v-layer11-attention W2V_LAYER11_Q_F32 W2V_LAYER11_K_F32 W2V_LAYER11_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER11_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer11_attention_project_command"]
        == "--clone-w2v-layer11-attention-project MODEL_BUNDLE_DIR W2V_LAYER11_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer11_attention_residual_command"]
        == "--clone-w2v-layer11-attention-residual W2V_LAYER11_FFN1_RESIDUAL_F32 W2V_LAYER11_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer11_attention_norm_command"]
        == "--clone-w2v-layer11-attention-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer11_conv_norm_command"]
        == "--clone-w2v-layer11-conv-norm MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer11_conv_glu_command"]
        == "--clone-w2v-layer11-conv-glu MODEL_BUNDLE_DIR W2V_LAYER11_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer11_conv_depthwise_command"]
        == "--clone-w2v-layer11-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER11_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer11_conv_residual_command"]
        == "--clone-w2v-layer11-conv-residual MODEL_BUNDLE_DIR W2V_LAYER11_ATTENTION_NORM_F32 W2V_LAYER11_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER11_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer11_ffn2_residual_command"]
        == "--clone-w2v-layer11-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER11_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER11_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer12_ffn1_residual_command"]
        == "--clone-w2v-layer12-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER11_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer12_qkv_command"]
        == "--clone-w2v-layer12-qkv MODEL_BUNDLE_DIR W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer12_attention_command"]
        == "--clone-w2v-layer12-attention W2V_LAYER12_Q_F32 W2V_LAYER12_K_F32 W2V_LAYER12_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER12_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer12_attention_project_command"]
        == "--clone-w2v-layer12-attention-project MODEL_BUNDLE_DIR W2V_LAYER12_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer12_attention_residual_command"]
        == "--clone-w2v-layer12-attention-residual W2V_LAYER12_FFN1_RESIDUAL_F32 W2V_LAYER12_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer12_attention_norm_command"]
        == "--clone-w2v-layer12-attention-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer12_conv_norm_command"]
        == "--clone-w2v-layer12-conv-norm MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer12_conv_glu_command"]
        == "--clone-w2v-layer12-conv-glu MODEL_BUNDLE_DIR W2V_LAYER12_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer12_conv_depthwise_command"]
        == "--clone-w2v-layer12-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER12_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer12_conv_residual_command"]
        == "--clone-w2v-layer12-conv-residual MODEL_BUNDLE_DIR W2V_LAYER12_ATTENTION_NORM_F32 W2V_LAYER12_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER12_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer12_ffn2_residual_command"]
        == "--clone-w2v-layer12-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER12_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER12_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer13_ffn1_residual_command"]
        == "--clone-w2v-layer13-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER12_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer13_qkv_command"]
        == "--clone-w2v-layer13-qkv MODEL_BUNDLE_DIR W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_w2v_layer13_attention_command"]
        == "--clone-w2v-layer13-attention W2V_LAYER13_Q_F32 W2V_LAYER13_K_F32 W2V_LAYER13_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER13_CONTEXT_F32"
    )
    assert (
        report["clone_w2v_layer13_attention_project_command"]
        == "--clone-w2v-layer13-attention-project MODEL_BUNDLE_DIR W2V_LAYER13_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_F32"
    )
    assert (
        report["clone_w2v_layer13_attention_residual_command"]
        == "--clone-w2v-layer13-attention-residual W2V_LAYER13_FFN1_RESIDUAL_F32 W2V_LAYER13_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer13_attention_norm_command"]
        == "--clone-w2v-layer13-attention-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_ATTENTION_NORM_F32"
    )
    assert (
        report["clone_w2v_layer13_conv_norm_command"]
        == "--clone-w2v-layer13-conv-norm MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_NORM_F32"
    )
    assert (
        report["clone_w2v_layer13_conv_glu_command"]
        == "--clone-w2v-layer13-conv-glu MODEL_BUNDLE_DIR W2V_LAYER13_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_GLU_F32"
    )
    assert (
        report["clone_w2v_layer13_conv_depthwise_command"]
        == "--clone-w2v-layer13-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER13_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_DEPTHWISE_F32"
    )
    assert (
        report["clone_w2v_layer13_conv_residual_command"]
        == "--clone-w2v-layer13-conv-residual MODEL_BUNDLE_DIR W2V_LAYER13_ATTENTION_NORM_F32 W2V_LAYER13_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER13_CONV_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer13_ffn2_residual_command"]
        == "--clone-w2v-layer13-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER13_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER13_FFN2_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer14_ffn1_residual_command"]
        == "--clone-w2v-layer14-ffn1-residual MODEL_BUNDLE_DIR W2V_LAYER13_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER14_FFN1_RESIDUAL_F32"
    )
    assert (
        report["clone_w2v_layer17_final_norm_command"]
        == "--clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32 W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32"
    )
    assert (
        report["clone_w2v_normalize_command"]
        == "--clone-w2v-normalize MODEL_BUNDLE_DIR W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32"
    )
    assert (
        report["clone_semantic_quantize_command"]
        == "--clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32 SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32"
    )
    assert (
        report["clone_semantic_prompt_from_spk_cond_command"]
        == "--clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS OUTPUT_DIR"
    )
    assert (
        report["clone_s2mel_prompt_from_sref_command"]
        == "--clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32"
    )
    assert (
        report["clone_encoder_readiness_command"]
        == "--clone-encoder-readiness FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32"
    )
    assert (
        report["clone_voice_bundle_writer_command"]
        == "--clone-write-voice-bundle PREPROCESS_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS MEL_F32 OUTPUT_VOICE_BUNDLE"
    )
    assert (
        report["clone_voice_bundle_writer_from_features_command"]
        == "--clone-write-voice-bundle-from-features FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS S2MEL_STYLE_F32 S2MEL_PROMPT_F32 PROMPT_TOKENS OUTPUT_VOICE_BUNDLE"
    )
    assert "--clone-preflight AUDIO_WAV" in report["native_product_commands"]
    assert "--clone-preprocess AUDIO_WAV OUTPUT_F32" in report["native_product_commands"]
    assert "--clone-readiness PREPROCESS_MANIFEST" in report["native_product_commands"]
    assert "--clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32" in report["native_product_commands"]
    assert "--clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32" in report["native_product_commands"]
    assert "--clone-prepare-features AUDIO_WAV OUTPUT_DIR" in report["native_product_commands"]
    assert "--clone-feature-readiness FEATURE_MANIFEST" in report["native_product_commands"]
    assert "--clone AUDIO_WAV OUTPUT_VOICE_BUNDLE" in report["native_product_commands"]
    assert report["clone_encoder_model_readiness_command"] in report["native_product_commands"]
    assert report["clone_campplus_style_readiness_command"] in report["native_product_commands"]
    assert report["clone_campplus_style_from_features_command"] in report["native_product_commands"]
    assert report["clone_campplus_head_golden_command"] in report["native_product_commands"]
    assert report["clone_w2v_feature_project_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_ffn1_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_ffn1_intermediate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_ffn1_activate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_ffn1_output_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_ffn1_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_attention_project_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_attention_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_attention_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_conv_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_conv_glu_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_conv_depthwise_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_conv_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_ffn2_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer0_final_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_ffn1_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_ffn1_intermediate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_ffn1_activate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_ffn1_output_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_ffn1_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_attention_project_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_attention_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_attention_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_conv_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_conv_glu_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_conv_depthwise_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_conv_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_ffn2_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer1_final_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_ffn1_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_ffn1_intermediate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_ffn1_activate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_ffn1_output_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_ffn1_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_attention_project_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_attention_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_attention_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_conv_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_conv_glu_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_conv_depthwise_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_conv_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer2_ffn2_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_ffn1_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_ffn1_intermediate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_ffn1_activate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_ffn1_output_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_ffn1_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_attention_project_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_attention_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_attention_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_conv_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_conv_glu_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_conv_depthwise_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_conv_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_ffn2_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer3_final_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_ffn1_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_ffn1_intermediate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_ffn1_activate_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_ffn1_output_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_ffn1_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_attention_project_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_attention_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_attention_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_conv_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_conv_glu_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_conv_depthwise_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_conv_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer4_ffn2_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_ffn1_residual_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_attention_project_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer5_attention_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_conv_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_conv_glu_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer5_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer5_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer5_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer6_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer6_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer6_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer7_attention_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer7_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer8_attention_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer8_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer9_attention_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer9_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer9_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer10_attention_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer10_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer11_attention_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer11_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer12_attention_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer12_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_qkv_command"] in report["native_product_commands"]
    assert report["clone_w2v_layer13_attention_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_attention_project_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_attention_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_attention_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_conv_norm_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_conv_glu_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_conv_depthwise_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_conv_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer13_ffn2_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer14_ffn1_residual_command"] in report[
        "native_product_commands"
    ]
    assert report["clone_w2v_layer17_final_norm_command"] in report["native_product_commands"]
    assert report["clone_w2v_normalize_command"] in report["native_product_commands"]
    assert report["clone_semantic_quantize_command"] in report["native_product_commands"]
    assert report["clone_semantic_prompt_from_spk_cond_command"] in report["native_product_commands"]
    assert report["clone_s2mel_prompt_from_sref_command"] in report["native_product_commands"]
    assert report["clone_encoder_readiness_command"] in report["native_product_commands"]
    assert report["clone_voice_bundle_writer_command"] in report["native_product_commands"]
    assert report["clone_voice_bundle_writer_from_features_command"] in report["native_product_commands"]
    assert "--clone-preflight AUDIO_WAV" in audit["recommended_preflight_commands"]
    assert "--clone-preprocess AUDIO_WAV OUTPUT_F32" in audit["recommended_preflight_commands"]
    assert "--clone-readiness PREPROCESS_MANIFEST" in audit["recommended_preflight_commands"]
    assert "--clone-extract-mel PREPROCESS_MANIFEST OUTPUT_MEL_F32" in audit["recommended_preflight_commands"]
    assert "--clone-extract-fbank PREPROCESS_MANIFEST OUTPUT_FBANK_F32" in audit["recommended_preflight_commands"]
    assert "--clone-prepare-features AUDIO_WAV OUTPUT_DIR" in audit["recommended_preflight_commands"]
    assert "--clone-feature-readiness FEATURE_MANIFEST" in audit["recommended_preflight_commands"]
    assert report["clone_encoder_model_readiness_command"] in audit["recommended_preflight_commands"]
    assert report["clone_campplus_style_readiness_command"] in audit["recommended_preflight_commands"]
    assert report["clone_campplus_style_from_features_command"] in audit["recommended_preflight_commands"]
    assert report["clone_campplus_head_golden_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_feature_project_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_ffn1_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_ffn1_intermediate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_ffn1_activate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_ffn1_output_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_ffn1_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_attention_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_attention_project_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_attention_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_attention_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_conv_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_conv_glu_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_conv_depthwise_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_conv_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_ffn2_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer0_final_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_ffn1_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_ffn1_intermediate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_ffn1_activate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_ffn1_output_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_ffn1_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_attention_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_attention_project_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_attention_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_attention_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_conv_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_conv_glu_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_conv_depthwise_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_conv_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_ffn2_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer1_final_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_ffn1_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_ffn1_intermediate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_ffn1_activate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_ffn1_output_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_ffn1_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_attention_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_attention_project_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_attention_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_attention_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_conv_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_conv_glu_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_conv_depthwise_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_conv_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer2_ffn2_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_ffn1_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_ffn1_intermediate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_ffn1_activate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_ffn1_output_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_ffn1_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_attention_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_attention_project_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_attention_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_attention_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_conv_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_conv_glu_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_conv_depthwise_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_conv_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_ffn2_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer3_final_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_ffn1_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_ffn1_intermediate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_ffn1_activate_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_ffn1_output_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_ffn1_residual_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_attention_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer4_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer4_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer4_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer4_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer4_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer4_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer4_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer4_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer5_attention_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer5_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer5_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_qkv_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_attention_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer6_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_qkv_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_attention_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer7_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_qkv_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_attention_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer8_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer9_attention_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer9_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer9_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_qkv_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_attention_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer10_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_qkv_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_layer11_attention_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer11_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_qkv_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_attention_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer12_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_qkv_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_attention_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_attention_project_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_attention_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_attention_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_conv_norm_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_conv_glu_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_conv_depthwise_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_conv_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer13_ffn2_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer14_ffn1_residual_command"] in audit[
        "recommended_preflight_commands"
    ]
    assert report["clone_w2v_layer17_final_norm_command"] in audit["recommended_preflight_commands"]
    assert report["clone_w2v_normalize_command"] in audit["recommended_preflight_commands"]
    assert report["clone_semantic_quantize_command"] in audit["recommended_preflight_commands"]
    assert report["clone_semantic_prompt_from_spk_cond_command"] in audit["recommended_preflight_commands"]
    assert report["clone_s2mel_prompt_from_sref_command"] in audit["recommended_preflight_commands"]
    assert report["clone_encoder_readiness_command"] in audit["recommended_preflight_commands"]
    assert report["clone_voice_bundle_writer_command"] in audit["recommended_preflight_commands"]
    assert report["clone_voice_bundle_writer_from_features_command"] in audit["recommended_preflight_commands"]


def test_mit2_tts_readiness_reports_start_sh_replacement_audit():
    if not (VOICE_BUNDLE / "manifest.json").exists():
        pytest.skip("test voice bundle is not available")

    report = _run_last_json("--readiness", str(MODEL_BUNDLE), str(VOICE_BUNDLE))

    assert report["stage"] == "tts_product_readiness"
    assert report["ok"] is True
    assert report["ready_cached_voice_tts_cjk"] is True
    assert report["ready_cached_voice_tts_general_text"] is False
    assert report["ready_native_voice_clone"] is False
    audit = report["start_sh_replacement_audit"]
    assert audit["can_replace_start_sh_cached_voice_cjk"] is True
    assert audit["can_replace_start_sh_full_clone_tts"] is False
    assert {gap["current_boundary"] for gap in audit["missing_surfaces"]} == {
        "python_owned",
        "incomplete",
    }
    assert "--clone-encoder-model-readiness MODEL_BUNDLE_DIR" in audit["recommended_preflight_commands"]
    assert "--clone-campplus-style-readiness MODEL_BUNDLE_DIR FEATURE_MANIFEST S2MEL_STYLE_F32" in audit[
        "recommended_preflight_commands"
    ]
    assert (
        "--clone-campplus-style-from-features MODEL_BUNDLE_DIR FEATURE_MANIFEST OUTPUT_S2MEL_STYLE_F32"
        in audit["recommended_preflight_commands"]
    )
    assert "--clone-campplus-head-golden MODEL_BUNDLE_DIR FEATURE_MANIFEST CAMPPLUS_GOLDEN_DIR" in audit[
        "recommended_preflight_commands"
    ]
    assert (
        "--clone-w2v-feature-project MODEL_BUNDLE_DIR W2V_INPUT_FEATURES_F32 W2V_TOKENS OUTPUT_W2V_FEATURE_PROJECTION_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-ffn1-norm MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_FFN1_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-ffn1-intermediate MODEL_BUNDLE_DIR W2V_FFN1_NORM_F32 W2V_TOKENS OUTPUT_FFN1_INTERMEDIATE_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-ffn1-activate W2V_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_FFN1_ACTIVATED_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-ffn1-output MODEL_BUNDLE_DIR W2V_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_FFN1_OUTPUT_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-ffn1-residual W2V_FEATURE_PROJECTION_F32 W2V_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_FFN1_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-qkv MODEL_BUNDLE_DIR W2V_FEATURE_PROJECTION_F32 W2V_TOKENS OUTPUT_DIR"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-attention W2V_Q_F32 W2V_K_F32 W2V_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_CONTEXT_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-attention-project MODEL_BUNDLE_DIR W2V_CONTEXT_F32 W2V_TOKENS OUTPUT_ATTENTION_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-attention-residual W2V_FEATURE_PROJECTION_F32 W2V_ATTENTION_F32 W2V_TOKENS OUTPUT_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-attention-norm MODEL_BUNDLE_DIR W2V_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_ATTENTION_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-conv-norm MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_CONV_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-conv-glu MODEL_BUNDLE_DIR W2V_CONV_NORM_F32 W2V_TOKENS OUTPUT_CONV_GLU_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-conv-depthwise MODEL_BUNDLE_DIR W2V_CONV_GLU_F32 W2V_TOKENS OUTPUT_CONV_DEPTHWISE_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-conv-residual MODEL_BUNDLE_DIR W2V_ATTENTION_NORM_F32 W2V_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_CONV_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-ffn2-residual MODEL_BUNDLE_DIR W2V_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_FFN2_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer0-final-norm MODEL_BUNDLE_DIR W2V_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER0_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER0_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_INTERMEDIATE_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-ffn1-activate W2V_LAYER1_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_ACTIVATED_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_OUTPUT_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-ffn1-residual W2V_LAYER0_F32 W2V_LAYER1_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER1_FFN1_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-qkv MODEL_BUNDLE_DIR W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-attention W2V_LAYER1_Q_F32 W2V_LAYER1_K_F32 W2V_LAYER1_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER1_CONTEXT_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-attention-project MODEL_BUNDLE_DIR W2V_LAYER1_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-attention-residual W2V_LAYER1_FFN1_RESIDUAL_F32 W2V_LAYER1_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-attention-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_ATTENTION_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-conv-norm MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-conv-glu MODEL_BUNDLE_DIR W2V_LAYER1_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_GLU_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER1_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_DEPTHWISE_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-conv-residual MODEL_BUNDLE_DIR W2V_LAYER1_ATTENTION_NORM_F32 W2V_LAYER1_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER1_CONV_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER1_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_FFN2_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer1-final-norm MODEL_BUNDLE_DIR W2V_LAYER1_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER1_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer2-attention-project MODEL_BUNDLE_DIR W2V_LAYER2_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer2-attention-residual W2V_LAYER2_FFN1_RESIDUAL_F32 W2V_LAYER2_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer2-attention-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER2_ATTENTION_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer2-conv-norm MODEL_BUNDLE_DIR W2V_LAYER2_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer2-conv-glu MODEL_BUNDLE_DIR W2V_LAYER2_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER2_CONV_GLU_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-attention-project MODEL_BUNDLE_DIR W2V_LAYER3_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-attention-residual W2V_LAYER3_FFN1_RESIDUAL_F32 W2V_LAYER3_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-attention-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_ATTENTION_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-conv-norm MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-conv-glu MODEL_BUNDLE_DIR W2V_LAYER3_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_GLU_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-conv-depthwise MODEL_BUNDLE_DIR W2V_LAYER3_CONV_GLU_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_DEPTHWISE_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-conv-residual MODEL_BUNDLE_DIR W2V_LAYER3_ATTENTION_NORM_F32 W2V_LAYER3_CONV_DEPTHWISE_F32 W2V_TOKENS OUTPUT_LAYER3_CONV_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-ffn2-residual MODEL_BUNDLE_DIR W2V_LAYER3_CONV_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_FFN2_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer3-final-norm MODEL_BUNDLE_DIR W2V_LAYER3_FFN2_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER3_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-ffn1-norm MODEL_BUNDLE_DIR W2V_LAYER3_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-ffn1-intermediate MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_INTERMEDIATE_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-ffn1-activate W2V_LAYER4_FFN1_INTERMEDIATE_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_ACTIVATED_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-ffn1-output MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_ACTIVATED_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_OUTPUT_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-ffn1-residual W2V_LAYER3_F32 W2V_LAYER4_FFN1_OUTPUT_F32 W2V_TOKENS OUTPUT_LAYER4_FFN1_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-qkv MODEL_BUNDLE_DIR W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_TOKENS OUTPUT_DIR"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-attention W2V_LAYER4_Q_F32 W2V_LAYER4_K_F32 W2V_LAYER4_V_F32 W2V_ATTENTION_MASK_U32 W2V_TOKENS OUTPUT_LAYER4_CONTEXT_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-attention-project MODEL_BUNDLE_DIR W2V_LAYER4_CONTEXT_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-attention-residual W2V_LAYER4_FFN1_RESIDUAL_F32 W2V_LAYER4_ATTENTION_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_RESIDUAL_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-attention-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_RESIDUAL_F32 W2V_TOKENS OUTPUT_LAYER4_ATTENTION_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-conv-norm MODEL_BUNDLE_DIR W2V_LAYER4_ATTENTION_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_NORM_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer4-conv-glu MODEL_BUNDLE_DIR W2V_LAYER4_CONV_NORM_F32 W2V_TOKENS OUTPUT_LAYER4_CONV_GLU_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-layer17-final-norm MODEL_BUNDLE_DIR W2V_LAYER16_F32 W2V_TOKENS OUTPUT_W2V_HIDDEN_STATE_17_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-w2v-normalize MODEL_BUNDLE_DIR W2V_HIDDEN_STATE_17_F32 W2V_TOKENS OUTPUT_SPK_COND_F32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-semantic-quantize MODEL_BUNDLE_DIR SPK_COND_F32 SPK_TOKENS OUTPUT_S_REF_F32 OUTPUT_CODES_U32"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-semantic-prompt-from-spk-cond MODEL_BUNDLE_DIR FEATURE_MANIFEST SPK_COND_F32 SPK_TOKENS OUTPUT_DIR"
        in audit["recommended_preflight_commands"]
    )
    assert (
        "--clone-s2mel-prompt-from-sref MODEL_BUNDLE_DIR FEATURE_MANIFEST S_REF_F32 S_REF_TOKENS OUTPUT_S2MEL_PROMPT_F32"
        in audit["recommended_preflight_commands"]
    )


def test_mit2_tts_clone_encoder_model_readiness_reports_missing_clone_models():
    report = _run_json("--clone-encoder-model-readiness", str(MODEL_BUNDLE), check=False)

    assert report["stage"] == "tts_clone_encoder_model_readiness"
    assert report["ok"] is False
    assert report["product_surface_version"] == 1
    assert report["binary"] == "mit2_tts"
    assert report["model_bundle_dir"] == str(MODEL_BUNDLE)
    assert report["ready_native_clone_encoder_models"] is False
    assert report["ready_native_voice_clone"] is False
    assert report["available_hot_semantic_codec_vq2emb"] is True
    assert report["component_counts"]["gpt"] > 0
    assert report["component_counts"]["semantic_codec"] > 0
    assert report["component_counts"]["s2mel"] > 0
    assert report["component_counts"]["bigvgan"] > 0
    assert report["component_counts"]["campplus"] == 0
    assert report["available_clone_encoder_components"] == []
    assert report["has_campplus_model_contract"] is False
    assert report["has_w2v_bert_model_contract"] is False
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 0
    assert "missing_w2v_bert_tensor:w2v_bert.feature_projection.layer_norm.weight" in report[
        "w2v_bert_contract_issues"
    ]
    assert report["ready_native_w2v_bert_model_contract"] is False
    assert report["has_semantic_codec_quantize_contract"] is True
    assert report["semantic_codec_quantize_required_tensor_count"] == 7
    assert report["semantic_codec_quantize_required_tensors_present"] == 7
    assert report["semantic_codec_quantize_contract_issues"] == []
    assert report["ready_native_semantic_codec_quantize_from_spk_cond"] is True
    assert report["has_s2mel_prompt_encoder_contract"] is True
    assert report["has_s2mel_prompt_condition_contract"] is True
    assert report["s2mel_prompt_condition_required_tensor_count"] == 20
    assert report["s2mel_prompt_condition_required_tensors_present"] == 20
    assert report["s2mel_prompt_condition_contract_issues"] == []
    assert report["ready_native_s2mel_prompt_condition_from_sref"] is True
    assert {model["component"] for model in report["required_clone_encoder_models"]} == {
        "campplus",
        "w2v_bert",
        "semantic_codec_quantize",
        "s2mel_prompt_condition",
    }
    assert set(report["clone_encoder_model_issues"]) == {
        "missing_clone_encoder_component_campplus",
        "missing_clone_encoder_component_w2v_bert",
    }


def test_mit2_tts_clone_encoder_model_readiness_detects_campplus_component(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    _write_campplus_contract_bundle(bundle_dir)

    report = _run_json("--clone-encoder-model-readiness", str(bundle_dir), check=False)

    assert report["stage"] == "tts_clone_encoder_model_readiness"
    assert report["ok"] is False
    assert report["has_campplus_model_contract"] is True
    assert report["campplus_required_tensor_count"] == 30
    assert report["campplus_required_tensors_present"] == 30
    assert report["campplus_contract_issues"] == []
    assert report["has_semantic_codec_quantize_contract"] is True
    assert report["semantic_codec_quantize_required_tensor_count"] == 7
    assert report["semantic_codec_quantize_required_tensors_present"] == 7
    assert report["semantic_codec_quantize_contract_issues"] == []
    assert report["has_s2mel_prompt_condition_contract"] is False
    assert report["has_w2v_bert_model_contract"] is False
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 0
    assert report["ready_native_w2v_bert_model_contract"] is False
    assert report["component_counts"]["campplus"] == 815
    assert report["available_clone_encoder_components"] == ["campplus"]
    assert "missing_clone_encoder_component_campplus" not in report["clone_encoder_model_issues"]
    assert "missing_clone_encoder_component_w2v_bert" in report["clone_encoder_model_issues"]
    assert "missing_s2mel_prompt_condition_contract" in report["clone_encoder_model_issues"]


def test_mit2_tts_clone_encoder_model_readiness_detects_w2v_bert_contract(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    _write_w2v_bert_contract_bundle(bundle_dir)

    report = _run_json("--clone-encoder-model-readiness", str(bundle_dir), check=False)

    assert report["stage"] == "tts_clone_encoder_model_readiness"
    assert report["ok"] is False
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["ready_native_w2v_bert_model_contract"] is True
    assert report["component_counts"]["w2v_bert"] == 418
    assert report["available_clone_encoder_components"] == ["w2v_bert"]
    assert "missing_clone_encoder_component_w2v_bert" not in report["clone_encoder_model_issues"]
    assert "missing_clone_encoder_component_campplus" in report["clone_encoder_model_issues"]
    assert "missing_native_semantic_codec_quantize_contract" in report["clone_encoder_model_issues"]
    assert "missing_s2mel_prompt_condition_contract" in report["clone_encoder_model_issues"]


def test_mit2_tts_clone_w2v_feature_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    projection_weight = np.zeros((1024, 160), dtype=np.float32)
    projection_weight[:160, :] = np.eye(160, dtype=np.float32)
    projection_bias = np.full((1024,), 0.25, dtype=np.float32)
    projection_bias[:160] = 0.0
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )
    features = np.arange(2 * 160, dtype=np.float32).reshape(2, 160) / 10.0
    features_path = tmp_path / "w2v_input_features.f32"
    out_path = tmp_path / "w2v_feature_projection.f32"
    features.tofile(features_path)

    report = _run_json(
        "--clone-w2v-feature-project",
        str(bundle_dir),
        str(features_path),
        "2",
        str(out_path),
    )

    mean = features.mean(axis=1, keepdims=True)
    var = ((features - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (features - mean) / np.sqrt(var + np.float32(1e-5))
    expected = np.full((2, 1024), 0.25, dtype=np.float32)
    expected[:, :160] = normed
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_feature_projection"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["input_features_shape"] == "[1,2,160]"
    assert report["feature_projection_shape"] == "[1,2,1024]"
    assert report["input_feature_values"] == 2 * 160
    assert report["feature_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_feature_project_issues"] == []
    assert report["ready_native_w2v_bert_feature_projection"] is True
    assert report["ready_metal_w2v_bert_feature_projection"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer0_ffn1_norm_writes_norm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.full((1024,), 1.5, dtype=np.float32)
    beta = np.full((1024,), -0.25, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_ffn1_norm_weight=gamma,
        layer0_ffn1_norm_bias=beta,
    )
    feature_projection = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 100.0
    feature_projection_path = tmp_path / "w2v_feature_projection.f32"
    out_path = tmp_path / "w2v_layer0_ffn1_norm.f32"
    feature_projection.tofile(feature_projection_path)

    report = _run_json(
        "--clone-w2v-layer0-ffn1-norm",
        str(bundle_dir),
        str(feature_projection_path),
        "2",
        str(out_path),
    )

    mean = feature_projection.mean(axis=1, keepdims=True)
    var = ((feature_projection - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((feature_projection - mean) / np.sqrt(var + np.float32(1e-5))) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_ffn1_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["feature_projection_shape"] == "[1,2,1024]"
    assert report["ffn1_norm_shape"] == "[1,2,1024]"
    assert report["feature_projection_values"] == 2 * 1024
    assert report["ffn1_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_ffn1_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer0_ffn1_norm"] is True
    assert report["ready_metal_w2v_bert_layer0_ffn1_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-5)


def test_mit2_tts_clone_w2v_layer0_ffn1_intermediate_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((4096, 1024), dtype=np.float32)
    weight[:1024, :] = np.eye(1024, dtype=np.float32)
    bias = np.full((4096,), 0.125, dtype=np.float32)
    bias[:1024] = -0.5
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_ffn1_intermediate_weight=weight,
        layer0_ffn1_intermediate_bias=bias,
    )
    ffn1_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 100.0
    norm_path = tmp_path / "w2v_layer0_ffn1_norm.f32"
    out_path = tmp_path / "w2v_layer0_ffn1_intermediate.f32"
    ffn1_norm.tofile(norm_path)

    report = _run_json(
        "--clone-w2v-layer0-ffn1-intermediate",
        str(bundle_dir),
        str(norm_path),
        "2",
        str(out_path),
    )

    expected = np.full((2, 4096), 0.125, dtype=np.float32)
    expected[:, :1024] = ffn1_norm - 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_ffn1_intermediate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["ffn1_norm_shape"] == "[1,2,1024]"
    assert report["ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["ffn1_norm_values"] == 2 * 1024
    assert report["ffn1_intermediate_values"] == 2 * 4096
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_ffn1_intermediate_issues"] == []
    assert report["ready_native_w2v_bert_layer0_ffn1_intermediate"] is True
    assert report["ready_metal_w2v_bert_layer0_ffn1_intermediate"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_ffn1_activate_writes_swish(tmp_path: Path):
    intermediate = np.linspace(-3.0, 3.0, 2 * 4096, dtype=np.float32).reshape(2, 4096)
    intermediate_path = tmp_path / "w2v_layer0_ffn1_intermediate.f32"
    out_path = tmp_path / "w2v_layer0_ffn1_activated.f32"
    intermediate.tofile(intermediate_path)

    report = _run_json(
        "--clone-w2v-layer0-ffn1-activate",
        str(intermediate_path),
        "2",
        str(out_path),
    )

    expected = intermediate / (1.0 + np.exp(-intermediate))
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_ffn1_activate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["ffn1_activated_shape"] == "[1,2,4096]"
    assert report["ffn1_intermediate_values"] == 2 * 4096
    assert report["ffn1_activated_values"] == 2 * 4096
    assert report["clone_w2v_layer0_ffn1_activate_issues"] == []
    assert report["ready_native_w2v_bert_layer0_ffn1_activation"] is True
    assert report["ready_metal_w2v_bert_layer0_ffn1_activation"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_ffn1_output_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 4096), dtype=np.float32)
    weight[:, :1024] = np.eye(1024, dtype=np.float32)
    bias = np.full((1024,), 0.375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_ffn1_output_weight=weight,
        layer0_ffn1_output_bias=bias,
    )
    activated = np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) / 100.0
    activated_path = tmp_path / "w2v_layer0_ffn1_activated.f32"
    out_path = tmp_path / "w2v_layer0_ffn1_output.f32"
    activated.tofile(activated_path)

    report = _run_json(
        "--clone-w2v-layer0-ffn1-output",
        str(bundle_dir),
        str(activated_path),
        "2",
        str(out_path),
    )

    expected = activated[:, :1024] + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_ffn1_output"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["ffn1_activated_shape"] == "[1,2,4096]"
    assert report["ffn1_output_shape"] == "[1,2,1024]"
    assert report["ffn1_activated_values"] == 2 * 4096
    assert report["ffn1_output_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_ffn1_output_issues"] == []
    assert report["ready_native_w2v_bert_layer0_ffn1_output"] is True
    assert report["ready_metal_w2v_bert_layer0_ffn1_output"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_ffn1_residual_writes_half_residual(tmp_path: Path):
    feature_projection = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 100.0
    ffn1_output = np.linspace(-2.0, 2.0, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    projection_path = tmp_path / "w2v_feature_projection.f32"
    output_path = tmp_path / "w2v_layer0_ffn1_output.f32"
    residual_path = tmp_path / "w2v_layer0_ffn1_residual.f32"
    feature_projection.tofile(projection_path)
    ffn1_output.tofile(output_path)

    report = _run_json(
        "--clone-w2v-layer0-ffn1-residual",
        str(projection_path),
        str(output_path),
        "2",
        str(residual_path),
    )

    expected = feature_projection + 0.5 * ffn1_output
    actual = np.fromfile(residual_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["feature_projection_shape"] == "[1,2,1024]"
    assert report["ffn1_output_shape"] == "[1,2,1024]"
    assert report["ffn1_residual_shape"] == "[1,2,1024]"
    assert report["feature_projection_values"] == 2 * 1024
    assert report["ffn1_output_values"] == 2 * 1024
    assert report["ffn1_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer0_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer0_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer0_ffn1_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_q_weight=eye,
        layer0_k_weight=2.0 * eye,
        layer0_v_weight=-eye,
    )
    feature_projection = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 100.0
    feature_projection_path = tmp_path / "w2v_feature_projection.f32"
    out_dir = tmp_path / "qkv"
    feature_projection.tofile(feature_projection_path)

    report = _run_json(
        "--clone-w2v-layer0-qkv",
        str(bundle_dir),
        str(feature_projection_path),
        "2",
        str(out_dir),
    )

    q = np.fromfile(out_dir / "w2v_layer0_q.f32", dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(out_dir / "w2v_layer0_k.f32", dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(out_dir / "w2v_layer0_v.f32", dtype=np.float32).reshape(2, 1024)
    manifest = json.loads((out_dir / "w2v_layer0_qkv.manifest.json").read_text(encoding="utf-8"))
    assert report["stage"] == "tts_clone_w2v_bert_layer0_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["feature_projection_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["feature_projection_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer0_qkv"] is True
    assert report["ready_metal_w2v_bert_layer0_qkv"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert manifest["format"] == "mit2-w2v-layer0-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer0_qkv"] is True
    np.testing.assert_allclose(q, feature_projection, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(k, 2.0 * feature_projection, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(v, -feature_projection, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer0_q.f32"
    k_path = tmp_path / "w2v_layer0_k.f32"
    v_path = tmp_path / "w2v_layer0_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer0_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32),
            np.arange(1024, dtype=np.float32) + 100.0,
        ]
    )
    mask = np.array([1, 1], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer0-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.repeat(((v[0] + v[1]) / 2.0)[None, :], 2, axis=0)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["context_shape"] == "[1,2,1024]"
    assert report["context_values"] == 2 * 1024
    assert report["clone_w2v_layer0_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer0_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer0_attention_context"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer0_out_weight=eye)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 50.0
    context_path = tmp_path / "w2v_layer0_context.f32"
    out_path = tmp_path / "w2v_layer0_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer0-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["context_shape"] == "[1,2,1024]"
    assert report["attention_projection_shape"] == "[1,2,1024]"
    assert report["context_values"] == 2 * 1024
    assert report["attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer0_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer0_attention_projection"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, context, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_attention_residual_writes_sum(tmp_path: Path):
    feature_projection = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 100.0
    attention = np.flip(feature_projection, axis=1).copy() * 0.25
    feature_path = tmp_path / "w2v_feature_projection.f32"
    attention_path = tmp_path / "w2v_layer0_attention.f32"
    out_path = tmp_path / "w2v_layer0_attention_residual.f32"
    feature_projection.tofile(feature_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer0-attention-residual",
        str(feature_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["feature_projection_shape"] == "[1,2,1024]"
    assert report["attention_projection_shape"] == "[1,2,1024]"
    assert report["residual_shape"] == "[1,2,1024]"
    assert report["feature_projection_values"] == 2 * 1024
    assert report["attention_projection_values"] == 2 * 1024
    assert report["residual_values"] == 2 * 1024
    assert report["clone_w2v_layer0_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer0_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer0_attention_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, feature_projection + attention, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.75, 1.25, 1024, dtype=np.float32)
    bias = np.linspace(-0.1, 0.1, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_self_attn_norm_weight=weight,
        layer0_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 80.0
    residual_path = tmp_path / "w2v_layer0_attention_residual.f32"
    out_path = tmp_path / "w2v_layer0_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer0-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["attention_residual_shape"] == "[1,2,1024]"
    assert report["attention_norm_shape"] == "[1,2,1024]"
    assert report["attention_residual_values"] == 2 * 1024
    assert report["attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer0_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer0_attention_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer0_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(1.1, 0.9, 1024, dtype=np.float32)
    bias = np.linspace(0.05, -0.05, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_conv_norm_weight=weight,
        layer0_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 64.0
    attention_norm_path = tmp_path / "w2v_layer0_attention_norm.f32"
    out_path = tmp_path / "w2v_layer0_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer0-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["attention_norm_shape"] == "[1,2,1024]"
    assert report["conv_norm_shape"] == "[1,2,1024]"
    assert report["attention_norm_values"] == 2 * 1024
    assert report["conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer0_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer0_conv_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer0_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = 0.25
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_conv_pointwise1_weight=weight,
        layer0_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 128.0
    conv_norm_path = tmp_path / "w2v_layer0_conv_norm.f32"
    out_path = tmp_path / "w2v_layer0_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer0-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm + 0.25) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["conv_norm_shape"] == "[1,2,1024]"
    assert report["conv_glu_shape"] == "[1,2,1024]"
    assert report["conv_norm_values"] == 2 * 1024
    assert report["conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer0_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer0_conv_glu"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 29] = 1.0
    bias = np.linspace(-0.2, 0.2, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_conv_depthwise_weight=weight,
        layer0_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 256.0
    conv_glu_path = tmp_path / "w2v_layer0_conv_glu.f32"
    out_path = tmp_path / "w2v_layer0_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer0-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = np.vstack([np.zeros((1, 1024), dtype=np.float32), conv_glu[:1]]) + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["conv_glu_shape"] == "[1,2,1024]"
    assert report["conv_depthwise_shape"] == "[1,2,1024]"
    assert report["conv_glu_values"] == 2 * 1024
    assert report["conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer0_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer0_conv_depthwise"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer0_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.ones((1024,), dtype=np.float32)
    norm_bias = np.zeros((1024,), dtype=np.float32)
    pw2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pw2_weight[:, :, 0] = np.eye(1024, dtype=np.float32)
    pw2_bias = np.full((1024,), 0.125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_conv_depthwise_norm_weight=norm_weight,
        layer0_conv_depthwise_norm_bias=norm_bias,
        layer0_conv_pointwise2_weight=pw2_weight,
        layer0_conv_pointwise2_bias=pw2_bias,
    )
    attention_norm = np.full((2, 1024), 0.5, dtype=np.float32)
    conv_depthwise = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    attention_norm_path = tmp_path / "w2v_layer0_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer0_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer0_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer0-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    mean = conv_depthwise.mean(axis=1, keepdims=True)
    var = ((conv_depthwise - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_depthwise - mean) / np.sqrt(var + 1e-5)
    expected = attention_norm + (normed / (1.0 + np.exp(-normed))) + 0.125
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["attention_norm_shape"] == "[1,2,1024]"
    assert report["conv_depthwise_shape"] == "[1,2,1024]"
    assert report["conv_residual_shape"] == "[1,2,1024]"
    assert report["attention_norm_values"] == 2 * 1024
    assert report["conv_depthwise_values"] == 2 * 1024
    assert report["conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer0_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer0_conv_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer0_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.25, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_ffn2_intermediate_weight=intermediate_weight,
        layer0_ffn2_intermediate_bias=intermediate_bias,
        layer0_ffn2_output_weight=output_weight,
        layer0_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer0_conv_residual.f32"
    out_path = tmp_path / "w2v_layer0_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer0-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) + 0.25
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["conv_residual_shape"] == "[1,2,1024]"
    assert report["ffn2_residual_shape"] == "[1,2,1024]"
    assert report["conv_residual_values"] == 2 * 1024
    assert report["ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer0_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer0_ffn2_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer0_final_norm_writes_layer0_hidden(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.75, 1.25, 1024, dtype=np.float32)
    beta = np.linspace(-0.1, 0.1, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer0_final_norm_weight=gamma,
        layer0_final_norm_bias=beta,
    )
    ffn2_residual = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    ffn2_residual_path = tmp_path / "w2v_layer0_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer0.f32"
    ffn2_residual.tofile(ffn2_residual_path)

    report = _run_json(
        "--clone-w2v-layer0-final-norm",
        str(bundle_dir),
        str(ffn2_residual_path),
        "2",
        str(out_path),
    )

    mean = ffn2_residual.mean(axis=1, keepdims=True)
    var = ((ffn2_residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((ffn2_residual - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer0_final_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer0_shape"] == "[1,2,1024]"
    assert report["ffn2_residual_values"] == 2 * 1024
    assert report["layer0_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer0_final_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer0_final_norm"] is True
    assert report["ready_metal_w2v_bert_layer0_final_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_ffn1_norm_writes_layer1_norm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.5, 1.5, 1024, dtype=np.float32)
    beta = np.linspace(-0.2, 0.2, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_ffn1_norm_weight=gamma,
        layer1_ffn1_norm_bias=beta,
    )
    layer0 = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    layer0_path = tmp_path / "w2v_layer0.f32"
    out_path = tmp_path / "w2v_layer1_ffn1_norm.f32"
    layer0.tofile(layer0_path)

    report = _run_json(
        "--clone-w2v-layer1-ffn1-norm",
        str(bundle_dir),
        str(layer0_path),
        "2",
        str(out_path),
    )

    mean = layer0.mean(axis=1, keepdims=True)
    var = ((layer0 - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((layer0 - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_ffn1_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer0_shape"] == "[1,2,1024]"
    assert report["layer1_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer0_values"] == 2 * 1024
    assert report["layer1_ffn1_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_ffn1_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer1_ffn1_norm"] is True
    assert report["ready_metal_w2v_bert_layer1_ffn1_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_ffn1_intermediate_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((4096, 1024), dtype=np.float32)
    weight[:1024, :] = np.eye(1024, dtype=np.float32)
    bias = np.full((4096,), 0.125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_ffn1_intermediate_weight=weight,
        layer1_ffn1_intermediate_bias=bias,
    )
    ffn1_norm = np.tile(np.linspace(-0.5, 0.5, 1024, dtype=np.float32), (2, 1))
    ffn1_norm_path = tmp_path / "w2v_layer1_ffn1_norm.f32"
    out_path = tmp_path / "w2v_layer1_ffn1_intermediate.f32"
    ffn1_norm.tofile(ffn1_norm_path)

    report = _run_json(
        "--clone-w2v-layer1-ffn1-intermediate",
        str(bundle_dir),
        str(ffn1_norm_path),
        "2",
        str(out_path),
    )

    expected = np.full((2, 4096), 0.125, dtype=np.float32)
    expected[:, :1024] += ffn1_norm
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_ffn1_intermediate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer1_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer1_ffn1_norm_values"] == 2 * 1024
    assert report["layer1_ffn1_intermediate_values"] == 2 * 4096
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_ffn1_intermediate_issues"] == []
    assert report["ready_native_w2v_bert_layer1_ffn1_intermediate"] is True
    assert report["ready_metal_w2v_bert_layer1_ffn1_intermediate"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_ffn1_activate_writes_swish(tmp_path: Path):
    intermediate = np.linspace(-3.0, 3.0, 2 * 4096, dtype=np.float32)
    intermediate_path = tmp_path / "w2v_layer1_ffn1_intermediate.f32"
    out_path = tmp_path / "w2v_layer1_ffn1_activated.f32"
    intermediate.tofile(intermediate_path)

    report = _run_json(
        "--clone-w2v-layer1-ffn1-activate",
        str(intermediate_path),
        "2",
        str(out_path),
    )

    expected = intermediate / (1.0 + np.exp(-intermediate))
    actual = np.fromfile(out_path, dtype=np.float32)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_ffn1_activate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer1_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer1_ffn1_intermediate_values"] == 2 * 4096
    assert report["layer1_ffn1_activated_values"] == 2 * 4096
    assert report["clone_w2v_layer1_ffn1_activate_issues"] == []
    assert report["ready_native_w2v_bert_layer1_ffn1_activation"] is True
    assert report["ready_metal_w2v_bert_layer1_ffn1_activation"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_ffn1_output_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 4096), dtype=np.float32)
    weight[:, :1024] = np.eye(1024, dtype=np.float32)
    bias = np.full((1024,), -0.25, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_ffn1_output_weight=weight,
        layer1_ffn1_output_bias=bias,
    )
    activated = np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) / 128.0
    activated_path = tmp_path / "w2v_layer1_ffn1_activated.f32"
    out_path = tmp_path / "w2v_layer1_ffn1_output.f32"
    activated.tofile(activated_path)

    report = _run_json(
        "--clone-w2v-layer1-ffn1-output",
        str(bundle_dir),
        str(activated_path),
        "2",
        str(out_path),
    )

    expected = activated[:, :1024] + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_ffn1_output"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer1_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer1_ffn1_activated_values"] == 2 * 4096
    assert report["layer1_ffn1_output_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_ffn1_output_issues"] == []
    assert report["ready_native_w2v_bert_layer1_ffn1_output"] is True
    assert report["ready_metal_w2v_bert_layer1_ffn1_output"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_ffn1_residual_writes_half_residual(tmp_path: Path):
    layer0 = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 64.0
    ffn1_output = np.linspace(-1.5, 1.5, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    layer0_path = tmp_path / "w2v_layer0.f32"
    ffn1_output_path = tmp_path / "w2v_layer1_ffn1_output.f32"
    out_path = tmp_path / "w2v_layer1_ffn1_residual.f32"
    layer0.tofile(layer0_path)
    ffn1_output.tofile(ffn1_output_path)

    report = _run_json(
        "--clone-w2v-layer1-ffn1-residual",
        str(layer0_path),
        str(ffn1_output_path),
        "2",
        str(out_path),
    )

    expected = layer0 + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer0_shape"] == "[1,2,1024]"
    assert report["layer1_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer1_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer0_values"] == 2 * 1024
    assert report["layer1_ffn1_output_values"] == 2 * 1024
    assert report["layer1_ffn1_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer1_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer1_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer1_ffn1_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer1_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_q_weight=eye,
        layer1_k_weight=2.0 * eye,
        layer1_v_weight=-eye,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 100.0
    residual_path = tmp_path / "w2v_layer1_ffn1_residual.f32"
    out_dir = tmp_path / "layer1_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer1-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q_path = out_dir / "w2v_layer1_q.f32"
    k_path = out_dir / "w2v_layer1_k.f32"
    v_path = out_dir / "w2v_layer1_v.f32"
    manifest_path = out_dir / "w2v_layer1_qkv.manifest.json"
    q = np.fromfile(q_path, dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(k_path, dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(v_path, dtype=np.float32).reshape(2, 1024)
    manifest = json.loads(manifest_path.read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer1_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer1_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer1_qkv"] is True
    assert report["ready_metal_w2v_bert_layer1_qkv"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert manifest["format"] == "mit2-w2v-layer1-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer1_qkv"] is True
    np.testing.assert_allclose(q, residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, 2.0 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, -residual, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer1_q.f32"
    k_path = tmp_path / "w2v_layer1_k.f32"
    v_path = tmp_path / "w2v_layer1_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer1_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32),
            np.arange(1024, dtype=np.float32) + 100.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer1-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.repeat(v[:1], 2, axis=0)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer1_context_shape"] == "[1,2,1024]"
    assert report["layer1_context_values"] == 2 * 1024
    assert report["clone_w2v_layer1_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer1_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer1_attention_context"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer1_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer1_out_weight=eye)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 75.0
    context_path = tmp_path / "w2v_layer1_context.f32"
    out_path = tmp_path / "w2v_layer1_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer1-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_context_shape"] == "[1,2,1024]"
    assert report["layer1_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer1_context_values"] == 2 * 1024
    assert report["layer1_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer1_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer1_attention_projection"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, context, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer1_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 90.0
    attention = np.flip(ffn1_residual, axis=1).copy() * 0.2
    ffn1_residual_path = tmp_path / "w2v_layer1_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer1_attention.f32"
    out_path = tmp_path / "w2v_layer1_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer1-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = ffn1_residual + attention
    assert report["stage"] == "tts_clone_w2v_bert_layer1_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer1_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer1_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer1_ffn1_residual_values"] == 2 * 1024
    assert report["layer1_attention_projection_values"] == 2 * 1024
    assert report["layer1_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer1_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer1_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer1_attention_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer1_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.8, 1.2, 1024, dtype=np.float32)
    bias = np.linspace(0.1, -0.1, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_self_attn_norm_weight=weight,
        layer1_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 70.0
    residual_path = tmp_path / "w2v_layer1_attention_residual.f32"
    out_path = tmp_path / "w2v_layer1_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer1-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((residual - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer1_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer1_attention_residual_values"] == 2 * 1024
    assert report["layer1_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer1_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer1_attention_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(1.15, 0.85, 1024, dtype=np.float32)
    bias = np.linspace(-0.03, 0.07, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_conv_norm_weight=weight,
        layer1_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 80.0
    attention_norm_path = tmp_path / "w2v_layer1_attention_norm.f32"
    out_path = tmp_path / "w2v_layer1_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer1-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer1_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer1_attention_norm_values"] == 2 * 1024
    assert report["layer1_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer1_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer1_conv_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = 0.125
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_conv_pointwise1_weight=weight,
        layer1_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 96.0
    conv_norm_path = tmp_path / "w2v_layer1_conv_norm.f32"
    out_path = tmp_path / "w2v_layer1_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer1-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm + 0.125) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer1_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer1_conv_norm_values"] == 2 * 1024
    assert report["layer1_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer1_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer1_conv_glu"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer1_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0
    bias = np.full((1024,), 0.0625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_conv_depthwise_weight=weight,
        layer1_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 112.0
    conv_glu_path = tmp_path / "w2v_layer1_conv_glu.f32"
    out_path = tmp_path / "w2v_layer1_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer1-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = conv_glu + 0.0625
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer1_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer1_conv_glu_values"] == 2 * 1024
    assert report["layer1_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer1_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer1_conv_depthwise"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer1_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.ones((1024,), dtype=np.float32)
    norm_bias = np.zeros((1024,), dtype=np.float32)
    pw2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pw2_weight[:, :, 0] = np.eye(1024, dtype=np.float32)
    pw2_bias = np.full((1024,), 0.125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_conv_depthwise_norm_weight=norm_weight,
        layer1_conv_depthwise_norm_bias=norm_bias,
        layer1_conv_pointwise2_weight=pw2_weight,
        layer1_conv_pointwise2_bias=pw2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 130.0
    conv_depthwise = np.full((2, 1024), 0.25, dtype=np.float32)
    attention_norm_path = tmp_path / "w2v_layer1_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer1_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer1_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer1-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    expected = attention_norm + 0.125
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer1_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer1_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer1_attention_norm_values"] == 2 * 1024
    assert report["layer1_conv_depthwise_values"] == 2 * 1024
    assert report["layer1_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer1_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer1_conv_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.25, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_ffn2_intermediate_weight=intermediate_weight,
        layer1_ffn2_intermediate_bias=intermediate_bias,
        layer1_ffn2_output_weight=output_weight,
        layer1_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer1_conv_residual.f32"
    out_path = tmp_path / "w2v_layer1_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer1-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) + 0.25
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer1_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer1_conv_residual_values"] == 2 * 1024
    assert report["layer1_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer1_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer1_ffn2_residual"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer1_final_norm_writes_layer1_hidden(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.75, 1.25, 1024, dtype=np.float32)
    beta = np.linspace(-0.1, 0.1, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer1_final_norm_weight=gamma,
        layer1_final_norm_bias=beta,
    )
    ffn2_residual = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    ffn2_residual_path = tmp_path / "w2v_layer1_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer1.f32"
    ffn2_residual.tofile(ffn2_residual_path)

    report = _run_json(
        "--clone-w2v-layer1-final-norm",
        str(bundle_dir),
        str(ffn2_residual_path),
        "2",
        str(out_path),
    )

    mean = ffn2_residual.mean(axis=1, keepdims=True)
    var = ((ffn2_residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((ffn2_residual - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer1_final_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer1_shape"] == "[1,2,1024]"
    assert report["layer1_ffn2_residual_values"] == 2 * 1024
    assert report["layer1_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer1_final_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer1_final_norm"] is True
    assert report["ready_metal_w2v_bert_layer1_final_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_ffn1_norm_writes_layer2_norm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.5, 1.5, 1024, dtype=np.float32)
    beta = np.linspace(-0.2, 0.2, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_ffn1_norm_weight=gamma,
        layer2_ffn1_norm_bias=beta,
    )
    layer1 = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    layer1_path = tmp_path / "w2v_layer1.f32"
    out_path = tmp_path / "w2v_layer2_ffn1_norm.f32"
    layer1.tofile(layer1_path)

    report = _run_json(
        "--clone-w2v-layer2-ffn1-norm",
        str(bundle_dir),
        str(layer1_path),
        "2",
        str(out_path),
    )

    mean = layer1.mean(axis=1, keepdims=True)
    var = ((layer1 - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((layer1 - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_ffn1_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_shape"] == "[1,2,1024]"
    assert report["layer2_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer1_values"] == 2 * 1024
    assert report["layer2_ffn1_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_ffn1_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer2_ffn1_norm"] is True
    assert report["ready_metal_w2v_bert_layer2_ffn1_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_ffn1_intermediate_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((4096, 1024), dtype=np.float32)
    weight[:, 0] = np.linspace(0.01, 0.02, 4096, dtype=np.float32)
    weight[:, 7] = np.linspace(-0.02, 0.01, 4096, dtype=np.float32)
    bias = np.linspace(-0.25, 0.25, 4096, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_ffn1_intermediate_weight=weight,
        layer2_ffn1_intermediate_bias=bias,
    )
    normed = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    normed_path = tmp_path / "w2v_layer2_ffn1_norm.f32"
    out_path = tmp_path / "w2v_layer2_ffn1_intermediate.f32"
    normed.tofile(normed_path)

    report = _run_json(
        "--clone-w2v-layer2-ffn1-intermediate",
        str(bundle_dir),
        str(normed_path),
        "2",
        str(out_path),
    )

    expected = normed @ weight.T + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_ffn1_intermediate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer2_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer2_ffn1_norm_values"] == 2 * 1024
    assert report["layer2_ffn1_intermediate_values"] == 2 * 4096
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_ffn1_intermediate_issues"] == []
    assert report["ready_native_w2v_bert_layer2_ffn1_intermediate"] is True
    assert report["ready_metal_w2v_bert_layer2_ffn1_intermediate"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_ffn1_activate_writes_swish(tmp_path: Path):
    intermediate = np.linspace(-3.0, 3.0, 2 * 4096, dtype=np.float32).reshape(2, 4096)
    intermediate_path = tmp_path / "w2v_layer2_ffn1_intermediate.f32"
    out_path = tmp_path / "w2v_layer2_ffn1_activated.f32"
    intermediate.tofile(intermediate_path)

    report = _run_json(
        "--clone-w2v-layer2-ffn1-activate",
        str(intermediate_path),
        "2",
        str(out_path),
    )

    expected = intermediate / (1.0 + np.exp(-intermediate))
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_ffn1_activate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer2_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer2_ffn1_intermediate_values"] == 2 * 4096
    assert report["layer2_ffn1_activated_values"] == 2 * 4096
    assert report["clone_w2v_layer2_ffn1_activate_issues"] == []
    assert report["ready_native_w2v_bert_layer2_ffn1_activation"] is True
    assert report["ready_metal_w2v_bert_layer2_ffn1_activation"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer2_ffn1_output_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 4096), dtype=np.float32)
    weight[:, :1024] = np.eye(1024, dtype=np.float32)
    bias = np.linspace(-0.3, 0.3, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_ffn1_output_weight=weight,
        layer2_ffn1_output_bias=bias,
    )
    activated = np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096) / 256.0
    activated_path = tmp_path / "w2v_layer2_ffn1_activated.f32"
    out_path = tmp_path / "w2v_layer2_ffn1_output.f32"
    activated.tofile(activated_path)

    report = _run_json(
        "--clone-w2v-layer2-ffn1-output",
        str(bundle_dir),
        str(activated_path),
        "2",
        str(out_path),
    )

    expected = activated[:, :1024] + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_ffn1_output"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer2_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer2_ffn1_activated_values"] == 2 * 4096
    assert report["layer2_ffn1_output_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_ffn1_output_issues"] == []
    assert report["ready_native_w2v_bert_layer2_ffn1_output"] is True
    assert report["ready_metal_w2v_bert_layer2_ffn1_output"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_ffn1_residual_writes_half_residual(tmp_path: Path):
    layer1 = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 128.0
    ffn1_output = np.linspace(-2.0, 2.0, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    layer1_path = tmp_path / "w2v_layer1.f32"
    ffn1_output_path = tmp_path / "w2v_layer2_ffn1_output.f32"
    out_path = tmp_path / "w2v_layer2_ffn1_residual.f32"
    layer1.tofile(layer1_path)
    ffn1_output.tofile(ffn1_output_path)

    report = _run_json(
        "--clone-w2v-layer2-ffn1-residual",
        str(layer1_path),
        str(ffn1_output_path),
        "2",
        str(out_path),
    )

    expected = layer1 + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer1_shape"] == "[1,2,1024]"
    assert report["layer2_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer2_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer1_values"] == 2 * 1024
    assert report["layer2_ffn1_output_values"] == 2 * 1024
    assert report["layer2_ffn1_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer2_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer2_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer2_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer2_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_q_weight=eye,
        layer2_k_weight=2.0 * eye,
        layer2_v_weight=-eye,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 200.0
    residual_path = tmp_path / "w2v_layer2_ffn1_residual.f32"
    out_dir = tmp_path / "layer2_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer2-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q_path = out_dir / "w2v_layer2_q.f32"
    k_path = out_dir / "w2v_layer2_k.f32"
    v_path = out_dir / "w2v_layer2_v.f32"
    manifest_path = out_dir / "w2v_layer2_qkv.manifest.json"
    q = np.fromfile(q_path, dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(k_path, dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(v_path, dtype=np.float32).reshape(2, 1024)
    manifest = json.loads(manifest_path.read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer2_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer2_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer2_qkv"] is True
    assert report["ready_metal_w2v_bert_layer2_qkv"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert manifest["format"] == "mit2-w2v-layer2-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer2_qkv"] is True
    np.testing.assert_allclose(q, residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, 2.0 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, -residual, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer2_q.f32"
    k_path = tmp_path / "w2v_layer2_k.f32"
    v_path = tmp_path / "w2v_layer2_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer2_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 50.0,
            np.arange(1024, dtype=np.float32) + 150.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer2-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.repeat(v[:1], 2, axis=0)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer2_context_shape"] == "[1,2,1024]"
    assert report["layer2_context_values"] == 2 * 1024
    assert report["clone_w2v_layer2_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer2_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer2_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer2_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer2_out_weight=eye)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 125.0
    context_path = tmp_path / "w2v_layer2_context.f32"
    out_path = tmp_path / "w2v_layer2_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer2-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_context_shape"] == "[1,2,1024]"
    assert report["layer2_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer2_context_values"] == 2 * 1024
    assert report["layer2_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer2_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer2_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, context, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer2_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 150.0
    attention = np.flip(ffn1_residual, axis=1).copy() * 0.125
    ffn1_residual_path = tmp_path / "w2v_layer2_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer2_attention.f32"
    out_path = tmp_path / "w2v_layer2_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer2-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = ffn1_residual + attention
    assert report["stage"] == "tts_clone_w2v_bert_layer2_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer2_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer2_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer2_ffn1_residual_values"] == 2 * 1024
    assert report["layer2_attention_projection_values"] == 2 * 1024
    assert report["layer2_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer2_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer2_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer2_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer2_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(1.25, 0.75, 1024, dtype=np.float32)
    bias = np.linspace(-0.08, 0.08, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_self_attn_norm_weight=weight,
        layer2_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 95.0
    residual_path = tmp_path / "w2v_layer2_attention_residual.f32"
    out_path = tmp_path / "w2v_layer2_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer2-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((residual - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer2_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer2_attention_residual_values"] == 2 * 1024
    assert report["layer2_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer2_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer2_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.7, 1.3, 1024, dtype=np.float32)
    bias = np.linspace(0.06, -0.06, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_conv_norm_weight=weight,
        layer2_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 110.0
    attention_norm_path = tmp_path / "w2v_layer2_attention_norm.f32"
    out_path = tmp_path / "w2v_layer2_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer2-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((attention_norm - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer2_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer2_attention_norm_values"] == 2 * 1024
    assert report["layer2_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer2_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer2_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = -0.25
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_conv_pointwise1_weight=weight,
        layer2_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 128.0
    conv_norm_path = tmp_path / "w2v_layer2_conv_norm.f32"
    out_path = tmp_path / "w2v_layer2_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer2-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm - 0.25) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer2_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer2_conv_norm_values"] == 2 * 1024
    assert report["layer2_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer2_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer2_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer2_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0
    bias = np.full((1024,), -0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_conv_depthwise_weight=weight,
        layer2_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 144.0
    conv_glu_path = tmp_path / "w2v_layer2_conv_glu.f32"
    out_path = tmp_path / "w2v_layer2_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer2-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = conv_glu - 0.03125
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer2_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer2_conv_glu_values"] == 2 * 1024
    assert report["layer2_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer2_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer2_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer2_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.ones((1024,), dtype=np.float32)
    norm_bias = np.zeros((1024,), dtype=np.float32)
    pw2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pw2_weight[:, :, 0] = np.eye(1024, dtype=np.float32)
    pw2_bias = np.full((1024,), 0.09375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_conv_depthwise_norm_weight=norm_weight,
        layer2_conv_depthwise_norm_bias=norm_bias,
        layer2_conv_pointwise2_weight=pw2_weight,
        layer2_conv_pointwise2_bias=pw2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 156.0
    conv_depthwise = np.full((2, 1024), 0.25, dtype=np.float32)
    attention_norm_path = tmp_path / "w2v_layer2_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer2_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer2_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer2-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    expected = attention_norm + 0.09375
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer2_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer2_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer2_attention_norm_values"] == 2 * 1024
    assert report["layer2_conv_depthwise_values"] == 2 * 1024
    assert report["layer2_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer2_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer2_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer2_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.1875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer2_ffn2_intermediate_weight=intermediate_weight,
        layer2_ffn2_intermediate_bias=intermediate_bias,
        layer2_ffn2_output_weight=output_weight,
        layer2_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-0.75, 0.75, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer2_conv_residual.f32"
    out_path = tmp_path / "w2v_layer2_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer2-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) + 0.1875
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer2_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer2_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer2_conv_residual_values"] == 2 * 1024
    assert report["layer2_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer2_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer2_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer2_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_ffn1_norm_writes_layer3_norm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.8, 1.2, 1024, dtype=np.float32)
    beta = np.linspace(-0.15, 0.15, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_ffn1_norm_weight=gamma,
        layer3_ffn1_norm_bias=beta,
    )
    layer2 = np.tile(np.linspace(-0.5, 0.5, 1024, dtype=np.float32), (2, 1))
    layer2_path = tmp_path / "w2v_layer2_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer3_ffn1_norm.f32"
    layer2.tofile(layer2_path)

    report = _run_json(
        "--clone-w2v-layer3-ffn1-norm",
        str(bundle_dir),
        str(layer2_path),
        "2",
        str(out_path),
    )

    mean = layer2.mean(axis=1, keepdims=True)
    var = ((layer2 - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((layer2 - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_ffn1_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer3_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer2_ffn2_residual_values"] == 2 * 1024
    assert report["layer3_ffn1_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_ffn1_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer3_ffn1_norm"] is True
    assert report["ready_metal_w2v_bert_layer3_ffn1_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_ffn1_intermediate_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((4096, 1024), dtype=np.float32)
    weight[:, 0] = np.linspace(0.015, 0.025, 4096, dtype=np.float32)
    weight[:, 11] = np.linspace(-0.018, 0.012, 4096, dtype=np.float32)
    bias = np.linspace(-0.2, 0.2, 4096, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_ffn1_intermediate_weight=weight,
        layer3_ffn1_intermediate_bias=bias,
    )
    normed = np.tile(np.linspace(-0.8, 0.8, 1024, dtype=np.float32), (2, 1))
    normed_path = tmp_path / "w2v_layer3_ffn1_norm.f32"
    out_path = tmp_path / "w2v_layer3_ffn1_intermediate.f32"
    normed.tofile(normed_path)

    report = _run_json(
        "--clone-w2v-layer3-ffn1-intermediate",
        str(bundle_dir),
        str(normed_path),
        "2",
        str(out_path),
    )

    expected = normed @ weight.T + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_ffn1_intermediate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer3_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer3_ffn1_norm_values"] == 2 * 1024
    assert report["layer3_ffn1_intermediate_values"] == 2 * 4096
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_ffn1_intermediate_issues"] == []
    assert report["ready_native_w2v_bert_layer3_ffn1_intermediate"] is True
    assert report["ready_metal_w2v_bert_layer3_ffn1_intermediate"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_ffn1_activate_writes_silu(tmp_path: Path):
    intermediate = np.tile(np.linspace(-4.0, 4.0, 4096, dtype=np.float32), (2, 1))
    intermediate_path = tmp_path / "w2v_layer3_ffn1_intermediate.f32"
    out_path = tmp_path / "w2v_layer3_ffn1_activated.f32"
    intermediate.tofile(intermediate_path)

    report = _run_json(
        "--clone-w2v-layer3-ffn1-activate",
        str(intermediate_path),
        "2",
        str(out_path),
    )

    expected = intermediate / (1.0 + np.exp(-intermediate))
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_ffn1_activate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer3_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer3_ffn1_intermediate_values"] == 2 * 4096
    assert report["layer3_ffn1_activated_values"] == 2 * 4096
    assert report["clone_w2v_layer3_ffn1_activate_issues"] == []
    assert report["ready_native_w2v_bert_layer3_ffn1_activation"] is True
    assert report["ready_metal_w2v_bert_layer3_ffn1_activation"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_ffn1_output_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 4096), dtype=np.float32)
    weight[:, :1024] = np.eye(1024, dtype=np.float32)
    weight[:, 1024:2048] = 0.125 * np.eye(1024, dtype=np.float32)
    bias = np.linspace(-0.1, 0.1, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_ffn1_output_weight=weight,
        layer3_ffn1_output_bias=bias,
    )
    activated = np.tile(np.linspace(-0.6, 0.6, 4096, dtype=np.float32), (2, 1))
    activated_path = tmp_path / "w2v_layer3_ffn1_activated.f32"
    out_path = tmp_path / "w2v_layer3_ffn1_output.f32"
    activated.tofile(activated_path)

    report = _run_json(
        "--clone-w2v-layer3-ffn1-output",
        str(bundle_dir),
        str(activated_path),
        "2",
        str(out_path),
    )

    expected = activated @ weight.T + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_ffn1_output"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer3_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer3_ffn1_activated_values"] == 2 * 4096
    assert report["layer3_ffn1_output_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_ffn1_output_issues"] == []
    assert report["ready_native_w2v_bert_layer3_ffn1_output"] is True
    assert report["ready_metal_w2v_bert_layer3_ffn1_output"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_ffn1_residual_writes_half_residual(tmp_path: Path):
    layer2 = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 192.0
    ffn1_output = np.linspace(-1.5, 1.5, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    layer2_path = tmp_path / "w2v_layer2_ffn2_residual.f32"
    ffn1_output_path = tmp_path / "w2v_layer3_ffn1_output.f32"
    out_path = tmp_path / "w2v_layer3_ffn1_residual.f32"
    layer2.tofile(layer2_path)
    ffn1_output.tofile(ffn1_output_path)

    report = _run_json(
        "--clone-w2v-layer3-ffn1-residual",
        str(layer2_path),
        str(ffn1_output_path),
        "2",
        str(out_path),
    )

    expected = layer2 + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer2_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer3_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer3_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer2_ffn2_residual_values"] == 2 * 1024
    assert report["layer3_ffn1_output_values"] == 2 * 1024
    assert report["layer3_ffn1_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer3_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer3_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer3_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer3_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_q_weight=eye,
        layer3_k_weight=1.5 * eye,
        layer3_v_weight=-0.5 * eye,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 240.0
    residual_path = tmp_path / "w2v_layer3_ffn1_residual.f32"
    out_dir = tmp_path / "layer3_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer3-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q_path = out_dir / "w2v_layer3_q.f32"
    k_path = out_dir / "w2v_layer3_k.f32"
    v_path = out_dir / "w2v_layer3_v.f32"
    manifest_path = out_dir / "w2v_layer3_qkv.manifest.json"
    q = np.fromfile(q_path, dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(k_path, dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(v_path, dtype=np.float32).reshape(2, 1024)
    manifest = json.loads(manifest_path.read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer3_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer3_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer3_qkv"] is True
    assert report["ready_metal_w2v_bert_layer3_qkv"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert manifest["format"] == "mit2-w2v-layer3-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer3_qkv"] is True
    np.testing.assert_allclose(q, residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, 1.5 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, -0.5 * residual, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer3_q.f32"
    k_path = tmp_path / "w2v_layer3_k.f32"
    v_path = tmp_path / "w2v_layer3_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer3_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 25.0,
            np.arange(1024, dtype=np.float32) + 125.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer3-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer3_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer3_context_shape"] == "[1,2,1024]"
    assert report["layer3_context_values"] == 2 * 1024
    assert report["clone_w2v_layer3_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer3_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer3_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1024), dtype=np.float32)
    weight[:, :] = np.eye(1024, dtype=np.float32)
    weight[:, 0] += np.linspace(0.0, 0.01, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer3_out_weight=weight)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 160.0
    context_path = tmp_path / "w2v_layer3_context.f32"
    out_path = tmp_path / "w2v_layer3_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer3-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    expected = context @ weight.T
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_context_shape"] == "[1,2,1024]"
    assert report["layer3_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer3_context_values"] == 2 * 1024
    assert report["layer3_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer3_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer3_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 170.0
    attention = np.flip(ffn1_residual, axis=1).copy() * 0.0625
    ffn1_residual_path = tmp_path / "w2v_layer3_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer3_attention.f32"
    out_path = tmp_path / "w2v_layer3_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer3-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = ffn1_residual + attention
    assert report["stage"] == "tts_clone_w2v_bert_layer3_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer3_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer3_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer3_ffn1_residual_values"] == 2 * 1024
    assert report["layer3_attention_projection_values"] == 2 * 1024
    assert report["layer3_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer3_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer3_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer3_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer3_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(1.15, 0.85, 1024, dtype=np.float32)
    bias = np.linspace(-0.05, 0.05, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_self_attn_norm_weight=weight,
        layer3_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 180.0
    residual_path = tmp_path / "w2v_layer3_attention_residual.f32"
    out_path = tmp_path / "w2v_layer3_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer3-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((residual - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer3_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer3_attention_residual_values"] == 2 * 1024
    assert report["layer3_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer3_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer3_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.65, 1.35, 1024, dtype=np.float32)
    bias = np.linspace(0.07, -0.07, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_conv_norm_weight=weight,
        layer3_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 190.0
    attention_norm_path = tmp_path / "w2v_layer3_attention_norm.f32"
    out_path = tmp_path / "w2v_layer3_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer3-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((attention_norm - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer3_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer3_attention_norm_values"] == 2 * 1024
    assert report["layer3_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer3_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer3_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = -0.125
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_conv_pointwise1_weight=weight,
        layer3_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 200.0
    conv_norm_path = tmp_path / "w2v_layer3_conv_norm.f32"
    out_path = tmp_path / "w2v_layer3_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer3-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm - 0.125) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer3_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer3_conv_norm_values"] == 2 * 1024
    assert report["layer3_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer3_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer3_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer3_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0
    bias = np.full((1024,), -0.015625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_conv_depthwise_weight=weight,
        layer3_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 210.0
    conv_glu_path = tmp_path / "w2v_layer3_conv_glu.f32"
    out_path = tmp_path / "w2v_layer3_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer3-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = conv_glu - 0.015625
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer3_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer3_conv_glu_values"] == 2 * 1024
    assert report["layer3_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer3_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer3_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer3_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.ones((1024,), dtype=np.float32)
    norm_bias = np.zeros((1024,), dtype=np.float32)
    pw2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pw2_weight[:, :, 0] = np.eye(1024, dtype=np.float32)
    pw2_bias = np.full((1024,), 0.046875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_conv_depthwise_norm_weight=norm_weight,
        layer3_conv_depthwise_norm_bias=norm_bias,
        layer3_conv_pointwise2_weight=pw2_weight,
        layer3_conv_pointwise2_bias=pw2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 240.0
    conv_depthwise = np.full((2, 1024), -0.125, dtype=np.float32)
    attention_norm_path = tmp_path / "w2v_layer3_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer3_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer3_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer3-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    expected = attention_norm + 0.046875
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer3_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer3_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer3_attention_norm_values"] == 2 * 1024
    assert report["layer3_conv_depthwise_values"] == 2 * 1024
    assert report["layer3_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer3_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer3_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.0625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_ffn2_intermediate_weight=intermediate_weight,
        layer3_ffn2_intermediate_bias=intermediate_bias,
        layer3_ffn2_output_weight=output_weight,
        layer3_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-0.5, 0.5, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer3_conv_residual.f32"
    out_path = tmp_path / "w2v_layer3_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer3-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) - 0.0625
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer3_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer3_conv_residual_values"] == 2 * 1024
    assert report["layer3_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer3_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer3_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer3_final_norm_writes_layer3_hidden(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.7, 1.3, 1024, dtype=np.float32)
    beta = np.linspace(-0.08, 0.08, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer3_final_norm_weight=gamma,
        layer3_final_norm_bias=beta,
    )
    ffn2_residual = np.tile(np.linspace(-0.9, 0.9, 1024, dtype=np.float32), (2, 1))
    ffn2_residual_path = tmp_path / "w2v_layer3_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer3.f32"
    ffn2_residual.tofile(ffn2_residual_path)

    report = _run_json(
        "--clone-w2v-layer3-final-norm",
        str(bundle_dir),
        str(ffn2_residual_path),
        "2",
        str(out_path),
    )

    mean = ffn2_residual.mean(axis=1, keepdims=True)
    var = ((ffn2_residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((ffn2_residual - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer3_final_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer3_shape"] == "[1,2,1024]"
    assert report["layer3_ffn2_residual_values"] == 2 * 1024
    assert report["layer3_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer3_final_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer3_final_norm"] is True
    assert report["ready_metal_w2v_bert_layer3_final_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_ffn1_norm_writes_layer4_norm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.85, 1.15, 1024, dtype=np.float32)
    beta = np.linspace(-0.12, 0.12, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_ffn1_norm_weight=gamma,
        layer4_ffn1_norm_bias=beta,
    )
    layer3 = np.tile(np.linspace(-0.65, 0.65, 1024, dtype=np.float32), (2, 1))
    layer3_path = tmp_path / "w2v_layer3.f32"
    out_path = tmp_path / "w2v_layer4_ffn1_norm.f32"
    layer3.tofile(layer3_path)

    report = _run_json(
        "--clone-w2v-layer4-ffn1-norm",
        str(bundle_dir),
        str(layer3_path),
        "2",
        str(out_path),
    )

    mean = layer3.mean(axis=1, keepdims=True)
    var = ((layer3 - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((layer3 - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_ffn1_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_shape"] == "[1,2,1024]"
    assert report["layer4_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer3_values"] == 2 * 1024
    assert report["layer4_ffn1_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_ffn1_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer4_ffn1_norm"] is True
    assert report["ready_metal_w2v_bert_layer4_ffn1_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_ffn1_intermediate_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((4096, 1024), dtype=np.float32)
    weight[:, 3] = np.linspace(0.012, 0.022, 4096, dtype=np.float32)
    weight[:, 19] = np.linspace(-0.02, 0.01, 4096, dtype=np.float32)
    bias = np.linspace(0.25, -0.25, 4096, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_ffn1_intermediate_weight=weight,
        layer4_ffn1_intermediate_bias=bias,
    )
    normed = np.tile(np.linspace(-0.7, 0.7, 1024, dtype=np.float32), (2, 1))
    normed_path = tmp_path / "w2v_layer4_ffn1_norm.f32"
    out_path = tmp_path / "w2v_layer4_ffn1_intermediate.f32"
    normed.tofile(normed_path)

    report = _run_json(
        "--clone-w2v-layer4-ffn1-intermediate",
        str(bundle_dir),
        str(normed_path),
        "2",
        str(out_path),
    )

    expected = normed @ weight.T + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_ffn1_intermediate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_ffn1_norm_shape"] == "[1,2,1024]"
    assert report["layer4_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer4_ffn1_norm_values"] == 2 * 1024
    assert report["layer4_ffn1_intermediate_values"] == 2 * 4096
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_ffn1_intermediate_issues"] == []
    assert report["ready_native_w2v_bert_layer4_ffn1_intermediate"] is True
    assert report["ready_metal_w2v_bert_layer4_ffn1_intermediate"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_ffn1_activate_writes_silu(tmp_path: Path):
    intermediate = np.tile(np.linspace(-4.5, 4.5, 4096, dtype=np.float32), (2, 1))
    intermediate_path = tmp_path / "w2v_layer4_ffn1_intermediate.f32"
    out_path = tmp_path / "w2v_layer4_ffn1_activated.f32"
    intermediate.tofile(intermediate_path)

    report = _run_json(
        "--clone-w2v-layer4-ffn1-activate",
        str(intermediate_path),
        "2",
        str(out_path),
    )

    expected = intermediate / (1.0 + np.exp(-intermediate))
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 4096)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_ffn1_activate"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_ffn1_intermediate_shape"] == "[1,2,4096]"
    assert report["layer4_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer4_ffn1_intermediate_values"] == 2 * 4096
    assert report["layer4_ffn1_activated_values"] == 2 * 4096
    assert report["clone_w2v_layer4_ffn1_activate_issues"] == []
    assert report["ready_native_w2v_bert_layer4_ffn1_activation"] is True
    assert report["ready_metal_w2v_bert_layer4_ffn1_activation"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_ffn1_output_writes_dense(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 4096), dtype=np.float32)
    weight[:, :1024] = np.eye(1024, dtype=np.float32)
    weight[:, 2048:3072] = 0.0625 * np.eye(1024, dtype=np.float32)
    bias = np.linspace(0.08, -0.08, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_ffn1_output_weight=weight,
        layer4_ffn1_output_bias=bias,
    )
    activated = np.tile(np.linspace(-0.75, 0.75, 4096, dtype=np.float32), (2, 1))
    activated_path = tmp_path / "w2v_layer4_ffn1_activated.f32"
    out_path = tmp_path / "w2v_layer4_ffn1_output.f32"
    activated.tofile(activated_path)

    report = _run_json(
        "--clone-w2v-layer4-ffn1-output",
        str(bundle_dir),
        str(activated_path),
        "2",
        str(out_path),
    )

    expected = activated @ weight.T + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_ffn1_output"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_ffn1_activated_shape"] == "[1,2,4096]"
    assert report["layer4_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer4_ffn1_activated_values"] == 2 * 4096
    assert report["layer4_ffn1_output_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_ffn1_output_issues"] == []
    assert report["ready_native_w2v_bert_layer4_ffn1_output"] is True
    assert report["ready_metal_w2v_bert_layer4_ffn1_output"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_ffn1_residual_writes_half_residual(tmp_path: Path):
    layer3 = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 224.0
    ffn1_output = np.linspace(-1.25, 1.25, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    layer3_path = tmp_path / "w2v_layer3.f32"
    ffn1_output_path = tmp_path / "w2v_layer4_ffn1_output.f32"
    out_path = tmp_path / "w2v_layer4_ffn1_residual.f32"
    layer3.tofile(layer3_path)
    ffn1_output.tofile(ffn1_output_path)

    report = _run_json(
        "--clone-w2v-layer4-ffn1-residual",
        str(layer3_path),
        str(ffn1_output_path),
        "2",
        str(out_path),
    )

    expected = layer3 + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer3_shape"] == "[1,2,1024]"
    assert report["layer4_ffn1_output_shape"] == "[1,2,1024]"
    assert report["layer4_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer3_values"] == 2 * 1024
    assert report["layer4_ffn1_output_values"] == 2 * 1024
    assert report["layer4_ffn1_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer4_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer4_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer4_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer4_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_q_weight=eye,
        layer4_k_weight=0.25 * eye,
        layer4_v_weight=-1.25 * eye,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 256.0
    residual_path = tmp_path / "w2v_layer4_ffn1_residual.f32"
    out_dir = tmp_path / "layer4_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer4-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q_path = out_dir / "w2v_layer4_q.f32"
    k_path = out_dir / "w2v_layer4_k.f32"
    v_path = out_dir / "w2v_layer4_v.f32"
    manifest_path = out_dir / "w2v_layer4_qkv.manifest.json"
    q = np.fromfile(q_path, dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(k_path, dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(v_path, dtype=np.float32).reshape(2, 1024)
    manifest = json.loads(manifest_path.read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer4_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer4_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer4_qkv"] is True
    assert report["ready_metal_w2v_bert_layer4_qkv"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert manifest["format"] == "mit2-w2v-layer4-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer4_qkv"] is True
    np.testing.assert_allclose(q, residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, 0.25 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, -1.25 * residual, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer4_q.f32"
    k_path = tmp_path / "w2v_layer4_k.f32"
    v_path = tmp_path / "w2v_layer4_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer4_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 40.0,
            np.arange(1024, dtype=np.float32) + 140.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer4-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer4_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer4_context_shape"] == "[1,2,1024]"
    assert report["layer4_context_values"] == 2 * 1024
    assert report["clone_w2v_layer4_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer4_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer4_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.eye(1024, dtype=np.float32)
    weight[:, 1] += np.linspace(0.0, 0.02, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer4_out_weight=weight)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 192.0
    context_path = tmp_path / "w2v_layer4_context.f32"
    out_path = tmp_path / "w2v_layer4_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer4-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    expected = context @ weight.T
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_context_shape"] == "[1,2,1024]"
    assert report["layer4_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer4_context_values"] == 2 * 1024
    assert report["layer4_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer4_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer4_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 224.0
    attention = np.flip(ffn1_residual, axis=1).copy() * -0.125
    ffn1_residual_path = tmp_path / "w2v_layer4_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer4_attention.f32"
    out_path = tmp_path / "w2v_layer4_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer4-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = ffn1_residual + attention
    assert report["stage"] == "tts_clone_w2v_bert_layer4_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer4_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer4_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer4_ffn1_residual_values"] == 2 * 1024
    assert report["layer4_attention_projection_values"] == 2 * 1024
    assert report["layer4_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer4_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer4_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer4_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer4_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.75, 1.25, 1024, dtype=np.float32)
    bias = np.linspace(0.04, -0.04, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_self_attn_norm_weight=weight,
        layer4_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 240.0
    residual_path = tmp_path / "w2v_layer4_attention_residual.f32"
    out_path = tmp_path / "w2v_layer4_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer4-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((residual - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer4_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer4_attention_residual_values"] == 2 * 1024
    assert report["layer4_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer4_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer4_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.85, 1.15, 1024, dtype=np.float32)
    bias = np.linspace(0.03, -0.03, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_conv_norm_weight=weight,
        layer4_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 260.0
    attention_norm_path = tmp_path / "w2v_layer4_attention_norm.f32"
    out_path = tmp_path / "w2v_layer4_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer4-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((attention_norm - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer4_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer4_attention_norm_values"] == 2 * 1024
    assert report["layer4_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer4_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer4_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = 0.25
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_conv_pointwise1_weight=weight,
        layer4_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 280.0
    conv_norm_path = tmp_path / "w2v_layer4_conv_norm.f32"
    out_path = tmp_path / "w2v_layer4_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer4-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm + 0.25) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer4_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer4_conv_norm_values"] == 2 * 1024
    assert report["layer4_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer4_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer4_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0
    bias = np.full((1024,), 0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_conv_depthwise_weight=weight,
        layer4_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 300.0
    conv_glu_path = tmp_path / "w2v_layer4_conv_glu.f32"
    out_path = tmp_path / "w2v_layer4_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer4-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = conv_glu + 0.03125
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer4_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer4_conv_glu_values"] == 2 * 1024
    assert report["layer4_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer4_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer4_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.full((1024,), 1.25, dtype=np.float32)
    norm_bias = np.full((1024,), -0.125, dtype=np.float32)
    pw2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pw2_weight[:, :, 0] = np.eye(1024, dtype=np.float32)
    pw2_bias = np.full((1024,), 0.0625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_conv_depthwise_norm_weight=norm_weight,
        layer4_conv_depthwise_norm_bias=norm_bias,
        layer4_conv_pointwise2_weight=pw2_weight,
        layer4_conv_pointwise2_bias=pw2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 320.0
    conv_depthwise = np.linspace(-0.75, 0.75, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    attention_norm_path = tmp_path / "w2v_layer4_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer4_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer4_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer4-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    mean = conv_depthwise.mean(axis=1, keepdims=True)
    var = ((conv_depthwise - mean) ** 2).mean(axis=1, keepdims=True)
    normed = ((conv_depthwise - mean) / np.sqrt(var + 1e-5)) * 1.25 - 0.125
    activated = normed / (1.0 + np.exp(-normed))
    expected = attention_norm + activated + 0.0625
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer4_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer4_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer4_attention_norm_values"] == 2 * 1024
    assert report["layer4_conv_depthwise_values"] == 2 * 1024
    assert report["layer4_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer4_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer4_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer4_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer4_ffn2_intermediate_weight=intermediate_weight,
        layer4_ffn2_intermediate_bias=intermediate_bias,
        layer4_ffn2_output_weight=output_weight,
        layer4_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-0.625, 0.625, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer4_conv_residual.f32"
    out_path = tmp_path / "w2v_layer4_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer4-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) - 0.03125
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer4_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer4_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer4_conv_residual_values"] == 2 * 1024
    assert report["layer4_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer4_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer4_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer4_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_ffn1_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.046875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_ffn1_intermediate_weight=intermediate_weight,
        layer5_ffn1_intermediate_bias=intermediate_bias,
        layer5_ffn1_output_weight=output_weight,
        layer5_ffn1_output_bias=output_bias,
    )
    layer4 = np.tile(np.linspace(-0.5, 0.5, 1024, dtype=np.float32), (2, 1))
    layer4_path = tmp_path / "w2v_layer4_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer5_ffn1_residual.f32"
    layer4.tofile(layer4_path)

    report = _run_json(
        "--clone-w2v-layer5-ffn1-residual",
        str(bundle_dir),
        str(layer4_path),
        "2",
        str(out_path),
    )

    mean = layer4.mean(axis=1, keepdims=True)
    var = ((layer4 - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer4 - mean) / np.sqrt(var + 1e-5)
    ffn1_output = normed / (1.0 + np.exp(-normed)) + 0.046875
    expected = layer4 + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer4_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer5_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer4_ffn2_residual_values"] == 2 * 1024
    assert report["layer5_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer5_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer5_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_q_weight=0.5 * eye,
        layer5_k_weight=-0.75 * eye,
        layer5_v_weight=1.5 * eye,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 384.0
    residual_path = tmp_path / "w2v_layer5_ffn1_residual.f32"
    out_dir = tmp_path / "layer5_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer5-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q_path = out_dir / "w2v_layer5_q.f32"
    k_path = out_dir / "w2v_layer5_k.f32"
    v_path = out_dir / "w2v_layer5_v.f32"
    manifest_path = out_dir / "w2v_layer5_qkv.manifest.json"
    q = np.fromfile(q_path, dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(k_path, dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(v_path, dtype=np.float32).reshape(2, 1024)
    manifest = json.loads(manifest_path.read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer5_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer5_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer5_qkv"] is True
    assert report["ready_metal_w2v_bert_layer5_qkv"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert manifest["format"] == "mit2-w2v-layer5-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer5_qkv"] is True
    np.testing.assert_allclose(q, 0.5 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, -0.75 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, 1.5 * residual, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer5_q.f32"
    k_path = tmp_path / "w2v_layer5_k.f32"
    v_path = tmp_path / "w2v_layer5_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer5_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 55.0,
            np.arange(1024, dtype=np.float32) + 155.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer5-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer5_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer5_context_shape"] == "[1,2,1024]"
    assert report["layer5_context_values"] == 2 * 1024
    assert report["clone_w2v_layer5_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer5_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer5_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.eye(1024, dtype=np.float32)
    weight[:, 2] += np.linspace(-0.01, 0.015, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer5_out_weight=weight)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 288.0
    context_path = tmp_path / "w2v_layer5_context.f32"
    out_path = tmp_path / "w2v_layer5_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer5-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    expected = context @ weight.T
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_context_shape"] == "[1,2,1024]"
    assert report["layer5_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer5_context_values"] == 2 * 1024
    assert report["layer5_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer5_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer5_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 304.0
    attention = np.flip(ffn1_residual, axis=1).copy() * 0.0625
    ffn1_residual_path = tmp_path / "w2v_layer5_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer5_attention.f32"
    out_path = tmp_path / "w2v_layer5_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer5-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = ffn1_residual + attention
    assert report["stage"] == "tts_clone_w2v_bert_layer5_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer5_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer5_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer5_ffn1_residual_values"] == 2 * 1024
    assert report["layer5_attention_projection_values"] == 2 * 1024
    assert report["layer5_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer5_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer5_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer5_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_mit2_tts_clone_w2v_layer5_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.65, 1.35, 1024, dtype=np.float32)
    bias = np.linspace(-0.035, 0.035, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_self_attn_norm_weight=weight,
        layer5_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 320.0
    residual_path = tmp_path / "w2v_layer5_attention_residual.f32"
    out_path = tmp_path / "w2v_layer5_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer5-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((residual - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer5_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer5_attention_residual_values"] == 2 * 1024
    assert report["layer5_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer5_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer5_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(1.2, 0.7, 1024, dtype=np.float32)
    bias = np.linspace(0.025, -0.025, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_conv_norm_weight=weight,
        layer5_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 338.0
    attention_norm_path = tmp_path / "w2v_layer5_attention_norm.f32"
    out_path = tmp_path / "w2v_layer5_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer5-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((attention_norm - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer5_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer5_attention_norm_values"] == 2 * 1024
    assert report["layer5_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer5_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer5_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = -0.125
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_conv_pointwise1_weight=weight,
        layer5_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 352.0
    conv_norm_path = tmp_path / "w2v_layer5_conv_norm.f32"
    out_path = tmp_path / "w2v_layer5_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer5-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm - 0.125) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer5_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer5_conv_norm_values"] == 2 * 1024
    assert report["layer5_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer5_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer5_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0
    bias = np.full((1024,), -0.046875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_conv_depthwise_weight=weight,
        layer5_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 384.0
    conv_glu_path = tmp_path / "w2v_layer5_conv_glu.f32"
    out_path = tmp_path / "w2v_layer5_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer5-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = conv_glu - 0.046875
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer5_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer5_conv_glu_values"] == 2 * 1024
    assert report["layer5_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer5_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer5_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.full((1024,), 0.875, dtype=np.float32)
    norm_bias = np.full((1024,), 0.09375, dtype=np.float32)
    pw2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pw2_weight[:, :, 0] = np.eye(1024, dtype=np.float32)
    pw2_bias = np.full((1024,), -0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_conv_depthwise_norm_weight=norm_weight,
        layer5_conv_depthwise_norm_bias=norm_bias,
        layer5_conv_pointwise2_weight=pw2_weight,
        layer5_conv_pointwise2_bias=pw2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 416.0
    conv_depthwise = np.linspace(-0.5, 0.5, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    attention_norm_path = tmp_path / "w2v_layer5_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer5_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer5_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer5-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    mean = conv_depthwise.mean(axis=1, keepdims=True)
    var = ((conv_depthwise - mean) ** 2).mean(axis=1, keepdims=True)
    normed = ((conv_depthwise - mean) / np.sqrt(var + 1e-5)) * 0.875 + 0.09375
    activated = normed / (1.0 + np.exp(-normed))
    expected = attention_norm + activated - 0.03125
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer5_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer5_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer5_attention_norm_values"] == 2 * 1024
    assert report["layer5_conv_depthwise_values"] == 2 * 1024
    assert report["layer5_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer5_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer5_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer5_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.0234375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer5_ffn2_intermediate_weight=intermediate_weight,
        layer5_ffn2_intermediate_bias=intermediate_bias,
        layer5_ffn2_output_weight=output_weight,
        layer5_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-0.8, 0.8, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer5_conv_residual.f32"
    out_path = tmp_path / "w2v_layer5_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer5-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) + 0.0234375
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer5_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer5_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer5_conv_residual_values"] == 2 * 1024
    assert report["layer5_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer5_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer5_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer5_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_ffn1_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.015625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_ffn1_intermediate_weight=intermediate_weight,
        layer6_ffn1_intermediate_bias=intermediate_bias,
        layer6_ffn1_output_weight=output_weight,
        layer6_ffn1_output_bias=output_bias,
    )
    layer5_residual = np.tile(np.linspace(-0.7, 0.7, 1024, dtype=np.float32), (2, 1))
    layer5_residual_path = tmp_path / "w2v_layer5_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer6_ffn1_residual.f32"
    layer5_residual.tofile(layer5_residual_path)

    report = _run_json(
        "--clone-w2v-layer6-ffn1-residual",
        str(bundle_dir),
        str(layer5_residual_path),
        "2",
        str(out_path),
    )

    mean = layer5_residual.mean(axis=1, keepdims=True)
    var = ((layer5_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer5_residual - mean) / np.sqrt(var + 1e-5)
    ffn1_output = normed / (1.0 + np.exp(-normed)) - 0.015625
    expected = layer5_residual + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer5_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer6_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer5_ffn2_residual_values"] == 2 * 1024
    assert report["layer6_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer6_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer6_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 self-attention, convolution, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_q_weight=0.25 * eye,
        layer6_k_weight=-1.25 * eye,
        layer6_v_weight=1.75 * eye,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 448.0
    residual_path = tmp_path / "w2v_layer6_ffn1_residual.f32"
    out_dir = tmp_path / "layer6_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer6-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q_path = out_dir / "w2v_layer6_q.f32"
    k_path = out_dir / "w2v_layer6_k.f32"
    v_path = out_dir / "w2v_layer6_v.f32"
    manifest_path = out_dir / "w2v_layer6_qkv.manifest.json"
    q = np.fromfile(q_path, dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(k_path, dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(v_path, dtype=np.float32).reshape(2, 1024)
    manifest = json.loads(manifest_path.read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer6_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer6_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer6_qkv"] is True
    assert report["ready_metal_w2v_bert_layer6_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert manifest["format"] == "mit2-w2v-layer6-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer6_qkv"] is True
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 attention scores/context, convolution, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(q, 0.25 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, -1.25 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, 1.75 * residual, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer6_q.f32"
    k_path = tmp_path / "w2v_layer6_k.f32"
    v_path = tmp_path / "w2v_layer6_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer6_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 65.0,
            np.arange(1024, dtype=np.float32) + 165.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer6-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer6_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer6_context_shape"] == "[1,2,1024]"
    assert report["layer6_context_values"] == 2 * 1024
    assert report["clone_w2v_layer6_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer6_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer6_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 attention output projection/residual/norm, convolution, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.eye(1024, dtype=np.float32)
    weight[:, 3] += np.linspace(0.02, -0.0125, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer6_out_weight=weight)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 320.0
    context_path = tmp_path / "w2v_layer6_context.f32"
    out_path = tmp_path / "w2v_layer6_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer6-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    expected = context @ weight.T
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_context_shape"] == "[1,2,1024]"
    assert report["layer6_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer6_context_values"] == 2 * 1024
    assert report["layer6_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer6_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer6_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 attention residual/norm, convolution, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 352.0
    attention = np.flip(ffn1_residual, axis=1).copy() * -0.03125
    ffn1_residual_path = tmp_path / "w2v_layer6_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer6_attention.f32"
    out_path = tmp_path / "w2v_layer6_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer6-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = ffn1_residual + attention
    assert report["stage"] == "tts_clone_w2v_bert_layer6_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer6_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer6_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer6_ffn1_residual_values"] == 2 * 1024
    assert report["layer6_attention_projection_values"] == 2 * 1024
    assert report["layer6_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer6_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer6_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer6_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 attention norm, convolution, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.7, 1.25, 1024, dtype=np.float32)
    bias = np.linspace(-0.028, 0.032, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_self_attn_norm_weight=weight,
        layer6_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 384.0
    residual_path = tmp_path / "w2v_layer6_attention_residual.f32"
    out_path = tmp_path / "w2v_layer6_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer6-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((residual - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer6_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer6_attention_residual_values"] == 2 * 1024
    assert report["layer6_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer6_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer6_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 convolution, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(1.15, 0.72, 1024, dtype=np.float32)
    bias = np.linspace(0.018, -0.027, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_conv_norm_weight=weight,
        layer6_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 448.0
    attention_norm_path = tmp_path / "w2v_layer6_attention_norm.f32"
    out_path = tmp_path / "w2v_layer6_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer6-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((attention_norm - mean) / np.sqrt(var + 1e-5)) * weight.reshape(1, 1024) + bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer6_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer6_attention_norm_values"] == 2 * 1024
    assert report["layer6_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer6_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer6_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 convolution GLU, depthwise/residual, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = 0.09375
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_conv_pointwise1_weight=weight,
        layer6_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    conv_norm_path = tmp_path / "w2v_layer6_conv_norm.f32"
    out_path = tmp_path / "w2v_layer6_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer6-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm + 0.09375) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer6_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer6_conv_norm_values"] == 2 * 1024
    assert report["layer6_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer6_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer6_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 depthwise convolution, convolution residual, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0
    bias = np.full((1024,), 0.0390625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_conv_depthwise_weight=weight,
        layer6_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 576.0
    conv_glu_path = tmp_path / "w2v_layer6_conv_glu.f32"
    out_path = tmp_path / "w2v_layer6_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer6-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = conv_glu + 0.0390625
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer6_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer6_conv_glu_values"] == 2 * 1024
    assert report["layer6_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer6_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer6_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 convolution residual, ffn2, and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(-0.25, 0.25, 1024, dtype=np.float32)
    pointwise2_weight = np.eye(1024, dtype=np.float32).reshape(1024, 1024, 1)
    pointwise2_bias = np.full((1024,), 0.015625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_conv_depthwise_norm_weight=norm_weight,
        layer6_conv_depthwise_norm_bias=norm_bias,
        layer6_conv_pointwise2_weight=pointwise2_weight,
        layer6_conv_pointwise2_bias=pointwise2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 640.0
    conv_depthwise = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    attention_norm_path = tmp_path / "w2v_layer6_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer6_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer6_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer6-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = attention_norm + activated.reshape(1, 1024) + pointwise2_bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer6_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer6_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer6_attention_norm_values"] == 2 * 1024
    assert report["layer6_conv_depthwise_values"] == 2 * 1024
    assert report["layer6_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer6_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer6_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-6 ffn2 and encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer6_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.017578125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer6_ffn2_intermediate_weight=intermediate_weight,
        layer6_ffn2_intermediate_bias=intermediate_bias,
        layer6_ffn2_output_weight=output_weight,
        layer6_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-0.7, 0.7, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer6_conv_residual.f32"
    out_path = tmp_path / "w2v_layer6_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer6-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) + 0.017578125
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer6_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer6_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer6_conv_residual_values"] == 2 * 1024
    assert report["layer6_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer6_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer6_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer6_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 7-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_ffn1_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.01171875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_ffn1_intermediate_weight=intermediate_weight,
        layer7_ffn1_intermediate_bias=intermediate_bias,
        layer7_ffn1_output_weight=output_weight,
        layer7_ffn1_output_bias=output_bias,
    )
    layer6_residual = np.tile(np.linspace(-0.65, 0.65, 1024, dtype=np.float32), (2, 1))
    layer6_residual_path = tmp_path / "w2v_layer6_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer7_ffn1_residual.f32"
    layer6_residual.tofile(layer6_residual_path)

    report = _run_json(
        "--clone-w2v-layer7-ffn1-residual",
        str(bundle_dir),
        str(layer6_residual_path),
        "2",
        str(out_path),
    )

    mean = layer6_residual.mean(axis=1, keepdims=True)
    var = ((layer6_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer6_residual - mean) / np.sqrt(var + 1e-5)
    ffn1_output = normed / (1.0 + np.exp(-normed)) - 0.01171875
    expected = layer6_residual + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer6_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer7_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer6_ffn2_residual_values"] == 2 * 1024
    assert report["layer7_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer7_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer7_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 self-attention, convolution, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_q_weight=-0.5 * eye,
        layer7_k_weight=0.75 * eye,
        layer7_v_weight=1.125 * eye,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 384.0
    residual_path = tmp_path / "w2v_layer7_ffn1_residual.f32"
    out_dir = tmp_path / "layer7_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer7-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q_path = out_dir / "w2v_layer7_q.f32"
    k_path = out_dir / "w2v_layer7_k.f32"
    v_path = out_dir / "w2v_layer7_v.f32"
    manifest_path = out_dir / "w2v_layer7_qkv.manifest.json"
    q = np.fromfile(q_path, dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(k_path, dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(v_path, dtype=np.float32).reshape(2, 1024)
    manifest = json.loads(manifest_path.read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer7_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer7_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer7_qkv"] is True
    assert report["ready_metal_w2v_bert_layer7_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert manifest["format"] == "mit2-w2v-layer7-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer7_qkv"] is True
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 attention scores/context, convolution, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(q, -0.5 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, 0.75 * residual, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, 1.125 * residual, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer7_q.f32"
    k_path = tmp_path / "w2v_layer7_k.f32"
    v_path = tmp_path / "w2v_layer7_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer7_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 77.0,
            np.arange(1024, dtype=np.float32) + 177.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer7-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer7_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer7_context_shape"] == "[1,2,1024]"
    assert report["layer7_context_values"] == 2 * 1024
    assert report["clone_w2v_layer7_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer7_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer7_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 attention output projection/residual/norm, convolution, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.eye(1024, dtype=np.float32)
    weight[:, 7] += np.linspace(-0.015625, 0.025, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer7_out_weight=weight)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 448.0
    context_path = tmp_path / "w2v_layer7_context.f32"
    out_path = tmp_path / "w2v_layer7_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer7-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    expected = context @ weight.T
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_context_shape"] == "[1,2,1024]"
    assert report["layer7_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer7_context_values"] == 2 * 1024
    assert report["layer7_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer7_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer7_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 attention residual/norm, convolution, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    attention = np.flip(ffn1_residual, axis=1).copy() * 0.046875
    ffn1_residual_path = tmp_path / "w2v_layer7_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer7_attention.f32"
    out_path = tmp_path / "w2v_layer7_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer7-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = ffn1_residual + attention
    assert report["stage"] == "tts_clone_w2v_bert_layer7_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer7_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer7_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer7_ffn1_residual_values"] == 2 * 1024
    assert report["layer7_attention_projection_values"] == 2 * 1024
    assert report["layer7_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer7_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer7_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer7_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 attention norm, convolution, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.75, 1.3, 1024, dtype=np.float32)
    bias = np.linspace(-0.03125, 0.02734375, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_self_attn_norm_weight=weight,
        layer7_self_attn_norm_bias=bias,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    residual_path = tmp_path / "w2v_layer7_attention_residual.f32"
    out_path = tmp_path / "w2v_layer7_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer7-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer7_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer7_attention_residual_values"] == 2 * 1024
    assert report["layer7_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer7_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer7_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 convolution, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(1.2, 0.68, 1024, dtype=np.float32)
    bias = np.linspace(0.0234375, -0.033203125, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_conv_norm_weight=weight,
        layer7_conv_norm_bias=bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 544.0
    attention_norm_path = tmp_path / "w2v_layer7_attention_norm.f32"
    out_path = tmp_path / "w2v_layer7_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer7-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer7_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer7_attention_norm_values"] == 2 * 1024
    assert report["layer7_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer7_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer7_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 convolution GLU, depthwise/residual, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = -0.078125
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_conv_pointwise1_weight=weight,
        layer7_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 608.0
    conv_norm_path = tmp_path / "w2v_layer7_conv_norm.f32"
    out_path = tmp_path / "w2v_layer7_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer7-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm - 0.078125) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer7_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer7_conv_norm_values"] == 2 * 1024
    assert report["layer7_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer7_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer7_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 depthwise convolution, convolution residual, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0
    bias = np.full((1024,), -0.044921875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_conv_depthwise_weight=weight,
        layer7_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 672.0
    conv_glu_path = tmp_path / "w2v_layer7_conv_glu.f32"
    out_path = tmp_path / "w2v_layer7_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer7-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = conv_glu - 0.044921875
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer7_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer7_conv_glu_values"] == 2 * 1024
    assert report["layer7_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer7_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer7_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 convolution residual, ffn2, and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(-0.22, 0.28, 1024, dtype=np.float32)
    pointwise2_weight = np.eye(1024, dtype=np.float32).reshape(1024, 1024, 1)
    pointwise2_bias = np.full((1024,), -0.01953125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_conv_depthwise_norm_weight=norm_weight,
        layer7_conv_depthwise_norm_bias=norm_bias,
        layer7_conv_pointwise2_weight=pointwise2_weight,
        layer7_conv_pointwise2_bias=pointwise2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 704.0
    conv_depthwise = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 576.0
    attention_norm_path = tmp_path / "w2v_layer7_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer7_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer7_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer7-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = attention_norm + activated.reshape(1, 1024) + pointwise2_bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer7_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer7_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer7_attention_norm_values"] == 2 * 1024
    assert report["layer7_conv_depthwise_values"] == 2 * 1024
    assert report["layer7_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer7_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer7_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-7 ffn2 and encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer7_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.013671875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer7_ffn2_intermediate_weight=intermediate_weight,
        layer7_ffn2_intermediate_bias=intermediate_bias,
        layer7_ffn2_output_weight=output_weight,
        layer7_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-0.8, 0.8, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer7_conv_residual.f32"
    out_path = tmp_path / "w2v_layer7_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer7-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) - 0.013671875
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer7_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer7_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer7_conv_residual_values"] == 2 * 1024
    assert report["layer7_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer7_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer7_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer7_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 8-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_ffn1_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.009765625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_ffn1_intermediate_weight=intermediate_weight,
        layer8_ffn1_intermediate_bias=intermediate_bias,
        layer8_ffn1_output_weight=output_weight,
        layer8_ffn1_output_bias=output_bias,
    )
    layer7_residual = np.tile(np.linspace(-0.9, 0.9, 1024, dtype=np.float32), (2, 1))
    layer7_residual_path = tmp_path / "w2v_layer7_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer8_ffn1_residual.f32"
    layer7_residual.tofile(layer7_residual_path)

    report = _run_json(
        "--clone-w2v-layer8-ffn1-residual",
        str(bundle_dir),
        str(layer7_residual_path),
        "2",
        str(out_path),
    )

    mean = layer7_residual.mean(axis=1, keepdims=True)
    var = ((layer7_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer7_residual - mean) / np.sqrt(var + 1e-5)
    ffn1_output = normed / (1.0 + np.exp(-normed)) + 0.009765625
    expected = layer7_residual + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer7_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer8_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer7_ffn2_residual_values"] == 2 * 1024
    assert report["layer8_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer8_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer8_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 self-attention, convolution, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_q_weight=eye * 0.5,
        layer8_k_weight=eye * -1.25,
        layer8_v_weight=eye * 1.75,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 768.0
    residual_path = tmp_path / "w2v_layer8_ffn1_residual.f32"
    out_dir = tmp_path / "layer8_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer8-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q = np.fromfile(out_dir / "w2v_layer8_q.f32", dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(out_dir / "w2v_layer8_k.f32", dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(out_dir / "w2v_layer8_v.f32", dtype=np.float32).reshape(2, 1024)
    manifest = json.loads((out_dir / "w2v_layer8_qkv.manifest.json").read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer8_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer8_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer8_qkv"] is True
    assert report["ready_metal_w2v_bert_layer8_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    assert manifest["format"] == "mit2-w2v-layer8-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer8_qkv"] is True
    assert manifest["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(q, residual * 0.5, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, residual * -1.25, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, residual * 1.75, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer8_q.f32"
    k_path = tmp_path / "w2v_layer8_k.f32"
    v_path = tmp_path / "w2v_layer8_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer8_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 97.0,
            np.arange(1024, dtype=np.float32) + 197.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer8-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer8_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer8_context_shape"] == "[1,2,1024]"
    assert report["layer8_context_values"] == 2 * 1024
    assert report["clone_w2v_layer8_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer8_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer8_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 attention output projection/residual/norm, convolution, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_attention_project_writes_projection(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer8_out_weight=eye * 2.25)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 256.0
    context_path = tmp_path / "w2v_layer8_context.f32"
    out_path = tmp_path / "w2v_layer8_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer8-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_context_shape"] == "[1,2,1024]"
    assert report["layer8_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer8_context_values"] == 2 * 1024
    assert report["layer8_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer8_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer8_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 attention residual/norm, convolution, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, context * 2.25, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1024.0
    attention = np.flip(ffn1_residual, axis=1).copy() * -0.75
    ffn1_residual_path = tmp_path / "w2v_layer8_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer8_attention.f32"
    out_path = tmp_path / "w2v_layer8_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer8-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer8_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer8_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer8_ffn1_residual_values"] == 2 * 1024
    assert report["layer8_attention_projection_values"] == 2 * 1024
    assert report["layer8_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer8_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer8_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer8_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 attention norm, convolution, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, ffn1_residual + attention, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_attention_norm_writes_layernorm(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.8, 1.2, 1024, dtype=np.float32)
    beta = np.linspace(-0.03, 0.03, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_self_attn_norm_weight=gamma,
        layer8_self_attn_norm_bias=beta,
    )
    residual = np.stack(
        [
            np.linspace(-2.0, 2.0, 1024, dtype=np.float32),
            np.linspace(2.5, -1.5, 1024, dtype=np.float32),
        ]
    )
    residual_path = tmp_path / "w2v_layer8_attention_residual.f32"
    out_path = tmp_path / "w2v_layer8_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer8-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer8_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer8_attention_residual_values"] == 2 * 1024
    assert report["layer8_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer8_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer8_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.7, 1.25, 1024, dtype=np.float32)
    bias = np.linspace(0.025, -0.025, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_conv_norm_weight=weight,
        layer8_conv_norm_bias=bias,
    )
    attention_norm = np.stack(
        [
            np.linspace(-1.5, 2.25, 1024, dtype=np.float32),
            np.linspace(1.75, -2.0, 1024, dtype=np.float32),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer8_attention_norm.f32"
    out_path = tmp_path / "w2v_layer8_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer8-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer8_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer8_attention_norm_values"] == 2 * 1024
    assert report["layer8_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer8_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer8_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 convolution GLU, depthwise/residual, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = 0.0625
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_conv_pointwise1_weight=weight,
        layer8_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    conv_norm_path = tmp_path / "w2v_layer8_conv_norm.f32"
    out_path = tmp_path / "w2v_layer8_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer8-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm + 0.0625) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer8_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer8_conv_norm_values"] == 2 * 1024
    assert report["layer8_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer8_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer8_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 depthwise convolution, convolution residual, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_conv_depthwise_writes_depthwise_conv(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.25
    weight[:, 0, 29] = -0.5
    bias = np.full((1024,), 0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_conv_depthwise_weight=weight,
        layer8_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 256.0
    conv_glu_path = tmp_path / "w2v_layer8_conv_glu.f32"
    out_path = tmp_path / "w2v_layer8_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer8-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = np.empty_like(conv_glu)
    expected[0] = conv_glu[0] * 1.25 + bias
    expected[1] = conv_glu[1] * 1.25 - conv_glu[0] * 0.5 + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer8_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer8_conv_glu_values"] == 2 * 1024
    assert report["layer8_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer8_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer8_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 convolution residual, ffn2, and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_conv_residual_writes_activation_projection_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(-0.18, 0.32, 1024, dtype=np.float32)
    pointwise2_weight = np.eye(1024, dtype=np.float32).reshape(1024, 1024, 1)
    pointwise2_bias = np.full((1024,), 0.0234375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_conv_depthwise_norm_weight=norm_weight,
        layer8_conv_depthwise_norm_bias=norm_bias,
        layer8_conv_pointwise2_weight=pointwise2_weight,
        layer8_conv_pointwise2_bias=pointwise2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 640.0
    conv_depthwise = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    attention_norm_path = tmp_path / "w2v_layer8_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer8_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer8_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer8-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = attention_norm + activated.reshape(1, 1024) + pointwise2_bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer8_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer8_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer8_attention_norm_values"] == 2 * 1024
    assert report["layer8_conv_depthwise_values"] == 2 * 1024
    assert report["layer8_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer8_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer8_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-8 ffn2 and encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer8_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.0078125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer8_ffn2_intermediate_weight=intermediate_weight,
        layer8_ffn2_intermediate_bias=intermediate_bias,
        layer8_ffn2_output_weight=output_weight,
        layer8_ffn2_output_bias=output_bias,
    )
    conv_residual = np.tile(np.linspace(-0.7, 0.7, 1024, dtype=np.float32), (2, 1))
    conv_residual_path = tmp_path / "w2v_layer8_conv_residual.f32"
    out_path = tmp_path / "w2v_layer8_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer8-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) - 0.0078125
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer8_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer8_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer8_conv_residual_values"] == 2 * 1024
    assert report["layer8_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer8_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer8_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer8_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 9-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_ffn1_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.01171875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_ffn1_intermediate_weight=intermediate_weight,
        layer9_ffn1_intermediate_bias=intermediate_bias,
        layer9_ffn1_output_weight=output_weight,
        layer9_ffn1_output_bias=output_bias,
    )
    layer8_residual = np.tile(np.linspace(-0.65, 0.65, 1024, dtype=np.float32), (2, 1))
    layer8_residual_path = tmp_path / "w2v_layer8_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer9_ffn1_residual.f32"
    layer8_residual.tofile(layer8_residual_path)

    report = _run_json(
        "--clone-w2v-layer9-ffn1-residual",
        str(bundle_dir),
        str(layer8_residual_path),
        "2",
        str(out_path),
    )

    mean = layer8_residual.mean(axis=1, keepdims=True)
    var = ((layer8_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer8_residual - mean) / np.sqrt(var + 1e-5)
    ffn1_output = normed / (1.0 + np.exp(-normed)) + 0.01171875
    expected = layer8_residual + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer8_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer9_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer8_ffn2_residual_values"] == 2 * 1024
    assert report["layer9_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer9_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer9_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 self-attention, convolution, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_q_weight=eye * -0.25,
        layer9_k_weight=eye * 0.75,
        layer9_v_weight=eye * 1.5,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 896.0
    residual_path = tmp_path / "w2v_layer9_ffn1_residual.f32"
    out_dir = tmp_path / "layer9_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer9-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q = np.fromfile(out_dir / "w2v_layer9_q.f32", dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(out_dir / "w2v_layer9_k.f32", dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(out_dir / "w2v_layer9_v.f32", dtype=np.float32).reshape(2, 1024)
    manifest = json.loads((out_dir / "w2v_layer9_qkv.manifest.json").read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer9_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer9_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer9_qkv"] is True
    assert report["ready_metal_w2v_bert_layer9_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    assert manifest["format"] == "mit2-w2v-layer9-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer9_qkv"] is True
    assert manifest["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(q, residual * -0.25, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, residual * 0.75, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, residual * 1.5, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer9_q.f32"
    k_path = tmp_path / "w2v_layer9_k.f32"
    v_path = tmp_path / "w2v_layer9_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer9_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 211.0,
            np.arange(1024, dtype=np.float32) + 421.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer9-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer9_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["heads"] == 16
    assert report["head_dim"] == 64
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["mask_values"] == 2
    assert report["layer9_context_shape"] == "[1,2,1024]"
    assert report["layer9_context_values"] == 2 * 1024
    assert report["clone_w2v_layer9_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer9_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer9_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 attention output projection/residual/norm, convolution, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_attention_project_writes_projection(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer9_out_weight=eye * -1.5)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 384.0
    context_path = tmp_path / "w2v_layer9_context.f32"
    out_path = tmp_path / "w2v_layer9_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer9-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_context_shape"] == "[1,2,1024]"
    assert report["layer9_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer9_context_values"] == 2 * 1024
    assert report["layer9_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer9_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer9_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 attention residual/norm, convolution, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, context * -1.5, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1536.0
    attention = np.flip(ffn1_residual, axis=1).copy() * 0.625
    ffn1_residual_path = tmp_path / "w2v_layer9_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer9_attention.f32"
    out_path = tmp_path / "w2v_layer9_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer9-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer9_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer9_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer9_ffn1_residual_values"] == 2 * 1024
    assert report["layer9_attention_projection_values"] == 2 * 1024
    assert report["layer9_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer9_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer9_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer9_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 attention norm, convolution, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, ffn1_residual + attention, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_attention_norm_writes_layernorm(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(1.15, 0.65, 1024, dtype=np.float32)
    beta = np.linspace(0.04, -0.04, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_self_attn_norm_weight=gamma,
        layer9_self_attn_norm_bias=beta,
    )
    residual = np.stack(
        [
            np.linspace(-1.75, 2.25, 1024, dtype=np.float32),
            np.linspace(3.0, -0.75, 1024, dtype=np.float32),
        ]
    )
    residual_path = tmp_path / "w2v_layer9_attention_residual.f32"
    out_path = tmp_path / "w2v_layer9_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer9-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer9_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer9_attention_residual_values"] == 2 * 1024
    assert report["layer9_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer9_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer9_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.55, 1.35, 1024, dtype=np.float32)
    bias = np.linspace(-0.035, 0.035, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_conv_norm_weight=weight,
        layer9_conv_norm_bias=bias,
    )
    attention_norm = np.stack(
        [
            np.linspace(-2.25, 1.5, 1024, dtype=np.float32),
            np.linspace(2.0, -1.75, 1024, dtype=np.float32),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer9_attention_norm.f32"
    out_path = tmp_path / "w2v_layer9_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer9-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer9_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer9_attention_norm_values"] == 2 * 1024
    assert report["layer9_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer9_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer9_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 convolution GLU, depthwise/residual, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = -0.125
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_conv_pointwise1_weight=weight,
        layer9_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 640.0
    conv_norm_path = tmp_path / "w2v_layer9_conv_norm.f32"
    out_path = tmp_path / "w2v_layer9_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer9-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm - 0.125) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer9_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer9_conv_norm_values"] == 2 * 1024
    assert report["layer9_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer9_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer9_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 depthwise convolution, convolution residual, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_conv_depthwise_writes_depthwise_conv(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = -1.125
    weight[:, 0, 29] = 0.375
    bias = np.full((1024,), -0.046875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_conv_depthwise_weight=weight,
        layer9_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 320.0
    conv_glu_path = tmp_path / "w2v_layer9_conv_glu.f32"
    out_path = tmp_path / "w2v_layer9_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer9-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = np.empty_like(conv_glu)
    expected[0] = conv_glu[0] * -1.125 + bias
    expected[1] = conv_glu[1] * -1.125 + conv_glu[0] * 0.375 + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer9_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer9_conv_glu_values"] == 2 * 1024
    assert report["layer9_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer9_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer9_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 convolution residual, ffn2, and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_conv_residual_writes_activation_projection_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(0.28, -0.22, 1024, dtype=np.float32)
    pointwise2_weight = np.eye(1024, dtype=np.float32).reshape(1024, 1024, 1)
    pointwise2_bias = np.full((1024,), -0.015625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_conv_depthwise_norm_weight=norm_weight,
        layer9_conv_depthwise_norm_bias=norm_bias,
        layer9_conv_pointwise2_weight=pointwise2_weight,
        layer9_conv_pointwise2_bias=pointwise2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 704.0
    conv_depthwise = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 448.0
    attention_norm_path = tmp_path / "w2v_layer9_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer9_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer9_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer9-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = attention_norm + activated.reshape(1, 1024) + pointwise2_bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer9_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer9_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer9_attention_norm_values"] == 2 * 1024
    assert report["layer9_conv_depthwise_values"] == 2 * 1024
    assert report["layer9_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer9_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer9_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-9 ffn2 and encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer9_ffn2_residual_writes_feed_forward_half_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(-0.24, 0.32, 1024, dtype=np.float32)
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_bias = np.tile(norm_bias, 4).astype(np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    for i in range(1024):
        output_weight[i, i] = 1.0
    output_bias = np.full((1024,), 0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer9_ffn2_norm_weight=norm_weight,
        layer9_ffn2_norm_bias=norm_bias,
        layer9_ffn2_intermediate_weight=intermediate_weight,
        layer9_ffn2_intermediate_bias=intermediate_bias,
        layer9_ffn2_output_weight=output_weight,
        layer9_ffn2_output_bias=output_bias,
    )
    conv_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 832.0
    conv_residual_path = tmp_path / "w2v_layer9_conv_residual.f32"
    out_path = tmp_path / "w2v_layer9_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer9-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = conv_residual + 0.5 * (
        activated.reshape(1, 1024) + output_bias.reshape(1, 1024)
    )
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer9_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer9_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer9_conv_residual_values"] == 2 * 1024
    assert report["layer9_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer9_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer9_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer9_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 10-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_ffn1_residual_writes_feed_forward_half_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(0.18, -0.34, 1024, dtype=np.float32)
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_bias = np.tile(norm_bias, 4).astype(np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    for i in range(1024):
        output_weight[i, i] = 1.0
    output_bias = np.full((1024,), -0.0234375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_ffn1_norm_weight=norm_weight,
        layer10_ffn1_norm_bias=norm_bias,
        layer10_ffn1_intermediate_weight=intermediate_weight,
        layer10_ffn1_intermediate_bias=intermediate_bias,
        layer10_ffn1_output_weight=output_weight,
        layer10_ffn1_output_bias=output_bias,
    )
    layer9_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 896.0
    layer9_residual_path = tmp_path / "w2v_layer9_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer10_ffn1_residual.f32"
    layer9_residual.tofile(layer9_residual_path)

    report = _run_json(
        "--clone-w2v-layer10-ffn1-residual",
        str(bundle_dir),
        str(layer9_residual_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = layer9_residual + 0.5 * (
        activated.reshape(1, 1024) + output_bias.reshape(1, 1024)
    )
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer9_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer10_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer9_ffn2_residual_values"] == 2 * 1024
    assert report["layer10_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer10_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer10_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 self-attention, convolution, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_q_weight=eye * 0.3125,
        layer10_k_weight=eye * -0.625,
        layer10_v_weight=eye * 1.875,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 960.0
    residual_path = tmp_path / "w2v_layer10_ffn1_residual.f32"
    out_dir = tmp_path / "layer10_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer10-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q = np.fromfile(out_dir / "w2v_layer10_q.f32", dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(out_dir / "w2v_layer10_k.f32", dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(out_dir / "w2v_layer10_v.f32", dtype=np.float32).reshape(2, 1024)
    manifest = json.loads((out_dir / "w2v_layer10_qkv.manifest.json").read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer10_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer10_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer10_qkv"] is True
    assert report["ready_metal_w2v_bert_layer10_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    assert manifest["format"] == "mit2-w2v-layer10-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer10_qkv"] is True
    assert manifest["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(q, residual * 0.3125, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, residual * -0.625, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, residual * 1.875, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer10_q.f32"
    k_path = tmp_path / "w2v_layer10_k.f32"
    v_path = tmp_path / "w2v_layer10_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer10_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 313.0,
            np.arange(1024, dtype=np.float32) + 557.0,
        ]
    )
    mask = np.array([1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer10-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile(v[0], (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer10_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_context_shape"] == "[1,2,1024]"
    assert report["layer10_context_values"] == 2 * 1024
    assert report["clone_w2v_layer10_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer10_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer10_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 attention output projection/residual/norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_attention_project_writes_projection(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer10_out_weight=eye * -0.75)
    context = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    context_path = tmp_path / "w2v_layer10_context.f32"
    out_path = tmp_path / "w2v_layer10_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer10-attention-project",
        str(bundle_dir),
        str(context_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_context_shape"] == "[1,2,1024]"
    assert report["layer10_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer10_context_values"] == 2 * 1024
    assert report["layer10_attention_projection_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer10_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer10_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 attention residual/norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, context * -0.75, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_attention_residual_writes_sum(tmp_path: Path):
    ffn1_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1792.0
    attention = np.flip(ffn1_residual, axis=1).copy() * -0.375
    ffn1_residual_path = tmp_path / "w2v_layer10_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer10_attention.f32"
    out_path = tmp_path / "w2v_layer10_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer10-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer10_attention_projection_shape"] == "[1,2,1024]"
    assert report["layer10_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer10_ffn1_residual_values"] == 2 * 1024
    assert report["layer10_attention_projection_values"] == 2 * 1024
    assert report["layer10_attention_residual_values"] == 2 * 1024
    assert report["clone_w2v_layer10_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer10_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer10_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 attention norm, convolution, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, ffn1_residual + attention, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_attention_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(1.25, 0.55, 1024, dtype=np.float32)
    beta = np.linspace(0.06, -0.06, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_self_attn_norm_weight=gamma,
        layer10_self_attn_norm_bias=beta,
    )
    residual = np.stack(
        [
            np.linspace(-2.25, 1.75, 1024, dtype=np.float32),
            np.linspace(2.75, -1.25, 1024, dtype=np.float32),
        ]
    )
    residual_path = tmp_path / "w2v_layer10_attention_residual.f32"
    out_path = tmp_path / "w2v_layer10_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer10-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_attention_residual_shape"] == "[1,2,1024]"
    assert report["layer10_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer10_attention_residual_values"] == 2 * 1024
    assert report["layer10_attention_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer10_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer10_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.linspace(0.5, 1.3, 1024, dtype=np.float32)
    bias = np.linspace(-0.045, 0.045, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_conv_norm_weight=weight,
        layer10_conv_norm_bias=bias,
    )
    attention_norm = np.stack(
        [
            np.linspace(-2.5, 1.25, 1024, dtype=np.float32),
            np.linspace(2.25, -1.5, 1024, dtype=np.float32),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer10_attention_norm.f32"
    out_path = tmp_path / "w2v_layer10_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer10-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * weight + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer10_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer10_attention_norm_values"] == 2 * 1024
    assert report["layer10_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer10_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer10_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 convolution GLU, depthwise/residual, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32)
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = 0.1875
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_conv_pointwise1_weight=weight,
        layer10_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 704.0
    conv_norm_path = tmp_path / "w2v_layer10_conv_norm.f32"
    out_path = tmp_path / "w2v_layer10_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer10-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    expected = (conv_norm + 0.1875) * 0.5
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer10_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer10_conv_norm_values"] == 2 * 1024
    assert report["layer10_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer10_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer10_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 depthwise convolution, convolution residual, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_conv_depthwise_writes_depthwise_conv(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 1.0625
    weight[:, 0, 29] = -0.3125
    bias = np.full((1024,), 0.0390625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_conv_depthwise_weight=weight,
        layer10_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 384.0
    conv_glu_path = tmp_path / "w2v_layer10_conv_glu.f32"
    out_path = tmp_path / "w2v_layer10_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer10-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "2",
        str(out_path),
    )

    expected = np.empty_like(conv_glu)
    expected[0] = conv_glu[0] * 1.0625 + bias
    expected[1] = conv_glu[1] * 1.0625 + conv_glu[0] * -0.3125 + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer10_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer10_conv_glu_values"] == 2 * 1024
    assert report["layer10_conv_depthwise_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer10_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer10_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 convolution residual, ffn2, and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_conv_residual_writes_activation_projection_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(0.18, -0.26, 1024, dtype=np.float32)
    pointwise2_weight = np.eye(1024, dtype=np.float32).reshape(1024, 1024, 1)
    pointwise2_bias = np.full((1024,), 0.0234375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_conv_depthwise_norm_weight=norm_weight,
        layer10_conv_depthwise_norm_bias=norm_bias,
        layer10_conv_pointwise2_weight=pointwise2_weight,
        layer10_conv_pointwise2_bias=pointwise2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 768.0
    conv_depthwise = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 512.0
    attention_norm_path = tmp_path / "w2v_layer10_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer10_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer10_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer10-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = attention_norm + activated.reshape(1, 1024) + pointwise2_bias.reshape(1, 1024)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer10_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer10_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer10_attention_norm_values"] == 2 * 1024
    assert report["layer10_conv_depthwise_values"] == 2 * 1024
    assert report["layer10_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer10_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer10_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-10 ffn2 and encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer10_ffn2_residual_writes_feed_forward_half_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(-0.18, 0.34, 1024, dtype=np.float32)
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_bias = np.tile(norm_bias, 4).astype(np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    for i in range(1024):
        output_weight[i, i] = 1.0
    output_bias = np.full((1024,), -0.02734375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer10_ffn2_norm_weight=norm_weight,
        layer10_ffn2_norm_bias=norm_bias,
        layer10_ffn2_intermediate_weight=intermediate_weight,
        layer10_ffn2_intermediate_bias=intermediate_bias,
        layer10_ffn2_output_weight=output_weight,
        layer10_ffn2_output_bias=output_bias,
    )
    conv_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 896.0
    conv_residual_path = tmp_path / "w2v_layer10_conv_residual.f32"
    out_path = tmp_path / "w2v_layer10_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer10-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = conv_residual + 0.5 * (
        activated.reshape(1, 1024) + output_bias.reshape(1, 1024)
    )
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer10_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer10_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer10_conv_residual_values"] == 2 * 1024
    assert report["layer10_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer10_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer10_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer10_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 11-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_ffn1_residual_writes_feed_forward_half_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.zeros((1024,), dtype=np.float32)
    norm_bias = np.linspace(0.22, -0.3, 1024, dtype=np.float32)
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_bias = np.tile(norm_bias, 4).astype(np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    for i in range(1024):
        output_weight[i, i] = 1.0
    output_bias = np.full((1024,), 0.01953125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_ffn1_norm_weight=norm_weight,
        layer11_ffn1_norm_bias=norm_bias,
        layer11_ffn1_intermediate_weight=intermediate_weight,
        layer11_ffn1_intermediate_bias=intermediate_bias,
        layer11_ffn1_output_weight=output_weight,
        layer11_ffn1_output_bias=output_bias,
    )
    layer10_residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 960.0
    layer10_residual_path = tmp_path / "w2v_layer10_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer11_ffn1_residual.f32"
    layer10_residual.tofile(layer10_residual_path)

    report = _run_json(
        "--clone-w2v-layer11-ffn1-residual",
        str(bundle_dir),
        str(layer10_residual_path),
        "2",
        str(out_path),
    )

    activated = norm_bias / (1.0 + np.exp(-norm_bias))
    expected = layer10_residual + 0.5 * (
        activated.reshape(1, 1024) + output_bias.reshape(1, 1024)
    )
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer10_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer11_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer10_ffn2_residual_values"] == 2 * 1024
    assert report["layer11_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer11_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer11_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 self-attention, convolution, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_q_weight=eye * -0.4375,
        layer11_k_weight=eye * 0.6875,
        layer11_v_weight=eye * -1.25,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1024.0
    residual_path = tmp_path / "w2v_layer11_ffn1_residual.f32"
    out_dir = tmp_path / "layer11_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer11-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q = np.fromfile(out_dir / "w2v_layer11_q.f32", dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(out_dir / "w2v_layer11_k.f32", dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(out_dir / "w2v_layer11_v.f32", dtype=np.float32).reshape(2, 1024)
    manifest = json.loads((out_dir / "w2v_layer11_qkv.manifest.json").read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer11_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer11_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer11_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer11_qkv"] is True
    assert report["ready_metal_w2v_bert_layer11_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    assert manifest["format"] == "mit2-w2v-layer11-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer11_qkv"] is True
    assert manifest["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(q, residual * -0.4375, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, residual * 0.6875, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, residual * -1.25, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer11_q.f32"
    k_path = tmp_path / "w2v_layer11_k.f32"
    v_path = tmp_path / "w2v_layer11_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer11_context.f32"
    q = np.zeros((3, 1024), dtype=np.float32)
    k = np.zeros((3, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) + 211.0,
            np.arange(1024, dtype=np.float32) + 433.0,
            np.arange(1024, dtype=np.float32) + 877.0,
        ]
    )
    mask = np.array([0, 1, 0], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer11-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    expected = np.tile(v[1], (3, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer11_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer11_context_shape"] == "[1,3,1024]"
    assert report["layer11_context_values"] == 3 * 1024
    assert report["clone_w2v_layer11_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer11_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer11_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 attention output projection/residual/norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_attention_project_writes_projection(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer11_out_weight=eye * 1.3125)
    context = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 640.0
    context_path = tmp_path / "w2v_layer11_context.f32"
    out_path = tmp_path / "w2v_layer11_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer11-attention-project",
        str(bundle_dir),
        str(context_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer11_context_shape"] == "[1,3,1024]"
    assert report["layer11_attention_projection_shape"] == "[1,3,1024]"
    assert report["layer11_context_values"] == 3 * 1024
    assert report["layer11_attention_projection_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer11_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer11_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 attention residual/norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, context * 1.3125, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_attention_residual_writes_sum(
    tmp_path: Path,
):
    ffn1_residual = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 2048.0
    attention = np.flip(ffn1_residual, axis=1).copy() * 0.8125
    ffn1_residual_path = tmp_path / "w2v_layer11_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer11_attention.f32"
    out_path = tmp_path / "w2v_layer11_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer11-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer11_ffn1_residual_shape"] == "[1,3,1024]"
    assert report["layer11_attention_projection_shape"] == "[1,3,1024]"
    assert report["layer11_attention_residual_shape"] == "[1,3,1024]"
    assert report["layer11_ffn1_residual_values"] == 3 * 1024
    assert report["layer11_attention_projection_values"] == 3 * 1024
    assert report["layer11_attention_residual_values"] == 3 * 1024
    assert report["clone_w2v_layer11_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer11_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer11_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 attention norm, convolution, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, ffn1_residual + attention, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_attention_norm_writes_layernorm(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.72, 1.18, 1024, dtype=np.float32)
    beta = np.linspace(-0.035, 0.055, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_self_attn_norm_weight=gamma,
        layer11_self_attn_norm_bias=beta,
    )
    residual = np.stack(
        [
            np.linspace(-1.875, 2.125, 1024, dtype=np.float32),
            np.linspace(2.5, -1.5, 1024, dtype=np.float32),
            np.sin(np.linspace(-1.0, 1.0, 1024, dtype=np.float32)),
        ]
    )
    residual_path = tmp_path / "w2v_layer11_attention_residual.f32"
    out_path = tmp_path / "w2v_layer11_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer11-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "3",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer11_attention_residual_shape"] == "[1,3,1024]"
    assert report["layer11_attention_norm_shape"] == "[1,3,1024]"
    assert report["layer11_attention_residual_values"] == 3 * 1024
    assert report["layer11_attention_norm_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer11_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer11_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_conv_norm_writes_layernorm(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(1.35, 0.65, 1024, dtype=np.float32)
    beta = np.linspace(0.04, -0.025, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_conv_norm_weight=gamma,
        layer11_conv_norm_bias=beta,
    )
    attention_norm = np.stack(
        [
            np.linspace(-2.0, 1.5, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.5, 1.5, 1024, dtype=np.float32)),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer11_attention_norm.f32"
    out_path = tmp_path / "w2v_layer11_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer11-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer11_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer11_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer11_attention_norm_values"] == 2 * 1024
    assert report["layer11_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer11_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer11_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 convolution GLU, depthwise/residual, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32) * 0.75
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = -0.125
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_conv_pointwise1_weight=weight,
        layer11_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 768.0
    conv_norm_path = tmp_path / "w2v_layer11_conv_norm.f32"
    out_path = tmp_path / "w2v_layer11_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer11-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    projected_a = conv_norm * 0.75 - 0.125
    projected_b = np.zeros_like(conv_norm)
    expected = projected_a / (1.0 + np.exp(-projected_b))
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer11_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer11_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer11_conv_norm_values"] == 2 * 1024
    assert report["layer11_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer11_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer11_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 depthwise convolution, convolution residual, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = 0.875
    weight[:, 0, 29] = 0.21875
    bias = np.full((1024,), -0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_conv_depthwise_weight=weight,
        layer11_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 512.0
    conv_glu_path = tmp_path / "w2v_layer11_conv_glu.f32"
    out_path = tmp_path / "w2v_layer11_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer11-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "3",
        str(out_path),
    )

    expected = np.empty_like(conv_glu)
    expected[0] = conv_glu[0] * 0.875 + bias
    expected[1:] = conv_glu[1:] * 0.875 + conv_glu[:-1] * 0.21875 + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer11_conv_glu_shape"] == "[1,3,1024]"
    assert report["layer11_conv_depthwise_shape"] == "[1,3,1024]"
    assert report["layer11_conv_glu_values"] == 3 * 1024
    assert report["layer11_conv_depthwise_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer11_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer11_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 convolution residual, ffn2, and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.full((1024,), 0.875, dtype=np.float32)
    norm_bias = np.full((1024,), 0.09375, dtype=np.float32)
    pointwise2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pointwise2_weight[:, :, 0] = np.eye(1024, dtype=np.float32)
    pointwise2_bias = np.full((1024,), -0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_conv_depthwise_norm_weight=norm_weight,
        layer11_conv_depthwise_norm_bias=norm_bias,
        layer11_conv_pointwise2_weight=pointwise2_weight,
        layer11_conv_pointwise2_bias=pointwise2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 896.0
    conv_depthwise = np.stack(
        [
            np.linspace(-0.7, 0.9, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.0, 1.0, 1024, dtype=np.float32)),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer11_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer11_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer11_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer11-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    mean = conv_depthwise.mean(axis=1, keepdims=True)
    var = ((conv_depthwise - mean) ** 2).mean(axis=1, keepdims=True)
    normed = ((conv_depthwise - mean) / np.sqrt(var + 1e-5)) * norm_weight + norm_bias
    expected = attention_norm + normed / (1.0 + np.exp(-normed)) - 0.03125
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer11_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer11_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer11_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer11_attention_norm_values"] == 2 * 1024
    assert report["layer11_conv_depthwise_values"] == 2 * 1024
    assert report["layer11_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer11_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer11_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-11 ffn2 and encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer11_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.015625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer11_ffn2_intermediate_weight=intermediate_weight,
        layer11_ffn2_intermediate_bias=intermediate_bias,
        layer11_ffn2_output_weight=output_weight,
        layer11_ffn2_output_bias=output_bias,
    )
    conv_residual = np.stack(
        [
            np.linspace(-0.9, 0.9, 1024, dtype=np.float32),
            np.sin(np.linspace(-1.25, 1.25, 1024, dtype=np.float32)),
        ]
    )
    conv_residual_path = tmp_path / "w2v_layer11_conv_residual.f32"
    out_path = tmp_path / "w2v_layer11_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer11-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) + 0.015625
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer11_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer11_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer11_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer11_conv_residual_values"] == 2 * 1024
    assert report["layer11_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer11_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer11_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer11_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 12-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_ffn1_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.0234375, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_ffn1_intermediate_weight=intermediate_weight,
        layer12_ffn1_intermediate_bias=intermediate_bias,
        layer12_ffn1_output_weight=output_weight,
        layer12_ffn1_output_bias=output_bias,
    )
    layer11_residual = np.stack(
        [
            np.linspace(-0.8, 0.85, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.2, 1.2, 1024, dtype=np.float32)),
        ]
    )
    layer11_residual_path = tmp_path / "w2v_layer11_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer12_ffn1_residual.f32"
    layer11_residual.tofile(layer11_residual_path)

    report = _run_json(
        "--clone-w2v-layer12-ffn1-residual",
        str(bundle_dir),
        str(layer11_residual_path),
        "2",
        str(out_path),
    )

    mean = layer11_residual.mean(axis=1, keepdims=True)
    var = ((layer11_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer11_residual - mean) / np.sqrt(var + 1e-5)
    ffn1_output = normed / (1.0 + np.exp(-normed)) - 0.0234375
    expected = layer11_residual + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer11_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer12_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer11_ffn2_residual_values"] == 2 * 1024
    assert report["layer12_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer12_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer12_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 self-attention, convolution, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_q_weight=eye * 0.3125,
        layer12_k_weight=eye * -0.5625,
        layer12_v_weight=eye * 1.375,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1536.0
    residual_path = tmp_path / "w2v_layer12_ffn1_residual.f32"
    out_dir = tmp_path / "layer12_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer12-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q = np.fromfile(out_dir / "w2v_layer12_q.f32", dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(out_dir / "w2v_layer12_k.f32", dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(out_dir / "w2v_layer12_v.f32", dtype=np.float32).reshape(2, 1024)
    manifest = json.loads((out_dir / "w2v_layer12_qkv.manifest.json").read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer12_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer12_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer12_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer12_qkv"] is True
    assert report["ready_metal_w2v_bert_layer12_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    assert manifest["format"] == "mit2-w2v-layer12-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer12_qkv"] is True
    assert manifest["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(q, residual * 0.3125, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, residual * -0.5625, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, residual * 1.375, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer12_q.f32"
    k_path = tmp_path / "w2v_layer12_k.f32"
    v_path = tmp_path / "w2v_layer12_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer12_context.f32"
    q = np.zeros((3, 1024), dtype=np.float32)
    k = np.zeros((3, 1024), dtype=np.float32)
    v = np.stack(
        [
            np.arange(1024, dtype=np.float32) - 31.0,
            np.arange(1024, dtype=np.float32) + 719.0,
            np.arange(1024, dtype=np.float32) + 2048.0,
        ]
    )
    mask = np.array([1, 0, 1], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer12-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    expected = np.tile((v[0] + v[2]) * 0.5, (3, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer12_attention"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer12_context_shape"] == "[1,3,1024]"
    assert report["layer12_context_values"] == 3 * 1024
    assert report["clone_w2v_layer12_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer12_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer12_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 attention output projection/residual/norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_attention_project_writes_projection(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer12_out_weight=eye * -0.8125)
    context = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 512.0
    context_path = tmp_path / "w2v_layer12_context.f32"
    out_path = tmp_path / "w2v_layer12_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer12-attention-project",
        str(bundle_dir),
        str(context_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer12_context_shape"] == "[1,3,1024]"
    assert report["layer12_attention_projection_shape"] == "[1,3,1024]"
    assert report["layer12_context_values"] == 3 * 1024
    assert report["layer12_attention_projection_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer12_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer12_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 attention residual/norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, context * -0.8125, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_attention_residual_writes_sum(
    tmp_path: Path,
):
    ffn1_residual = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 1792.0
    attention = np.flip(ffn1_residual, axis=0).copy() * -0.4375
    ffn1_residual_path = tmp_path / "w2v_layer12_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer12_attention.f32"
    out_path = tmp_path / "w2v_layer12_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer12-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer12_ffn1_residual_shape"] == "[1,3,1024]"
    assert report["layer12_attention_projection_shape"] == "[1,3,1024]"
    assert report["layer12_attention_residual_shape"] == "[1,3,1024]"
    assert report["layer12_ffn1_residual_values"] == 3 * 1024
    assert report["layer12_attention_projection_values"] == 3 * 1024
    assert report["layer12_attention_residual_values"] == 3 * 1024
    assert report["clone_w2v_layer12_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer12_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer12_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 attention norm, convolution, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, ffn1_residual + attention, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_attention_norm_writes_layernorm(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.68, 1.24, 1024, dtype=np.float32)
    beta = np.linspace(-0.045, 0.065, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_self_attn_norm_weight=gamma,
        layer12_self_attn_norm_bias=beta,
    )
    residual = np.stack(
        [
            np.linspace(-2.125, 1.875, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.25, 1.25, 1024, dtype=np.float32)),
            np.linspace(2.25, -1.75, 1024, dtype=np.float32),
        ]
    )
    residual_path = tmp_path / "w2v_layer12_attention_residual.f32"
    out_path = tmp_path / "w2v_layer12_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer12-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "3",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer12_attention_residual_shape"] == "[1,3,1024]"
    assert report["layer12_attention_norm_shape"] == "[1,3,1024]"
    assert report["layer12_attention_residual_values"] == 3 * 1024
    assert report["layer12_attention_norm_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer12_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer12_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_conv_norm_writes_layernorm(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.74, 1.16, 1024, dtype=np.float32)
    beta = np.linspace(-0.025, 0.075, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_conv_norm_weight=gamma,
        layer12_conv_norm_bias=beta,
    )
    attention_norm = np.stack(
        [
            np.linspace(-1.625, 2.375, 1024, dtype=np.float32),
            np.sin(np.linspace(-1.75, 1.75, 1024, dtype=np.float32)),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer12_attention_norm.f32"
    out_path = tmp_path / "w2v_layer12_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer12-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer12_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer12_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer12_attention_norm_values"] == 2 * 1024
    assert report["layer12_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer12_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer12_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 convolution GLU, depthwise/residual, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    weight[:1024, :, 0] = np.eye(1024, dtype=np.float32) * -0.625
    bias = np.zeros((2048,), dtype=np.float32)
    bias[:1024] = 0.1875
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_conv_pointwise1_weight=weight,
        layer12_conv_pointwise1_bias=bias,
    )
    conv_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 896.0
    conv_norm_path = tmp_path / "w2v_layer12_conv_norm.f32"
    out_path = tmp_path / "w2v_layer12_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer12-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    projected_a = conv_norm * -0.625 + 0.1875
    projected_b = np.zeros_like(conv_norm)
    expected = projected_a / (1.0 + np.exp(-projected_b))
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer12_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer12_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer12_conv_norm_values"] == 2 * 1024
    assert report["layer12_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer12_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer12_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 depthwise convolution, convolution residual, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_conv_depthwise_writes_depthwise_conv(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, 30] = -0.5625
    weight[:, 0, 29] = 0.3125
    bias = np.full((1024,), 0.046875, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_conv_depthwise_weight=weight,
        layer12_conv_depthwise_bias=bias,
    )
    conv_glu = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 640.0
    conv_glu_path = tmp_path / "w2v_layer12_conv_glu.f32"
    out_path = tmp_path / "w2v_layer12_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer12-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "3",
        str(out_path),
    )

    expected = np.empty_like(conv_glu)
    expected[0] = conv_glu[0] * -0.5625 + bias
    expected[1:] = conv_glu[1:] * -0.5625 + conv_glu[:-1] * 0.3125 + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer12_conv_glu_shape"] == "[1,3,1024]"
    assert report["layer12_conv_depthwise_shape"] == "[1,3,1024]"
    assert report["layer12_conv_glu_values"] == 3 * 1024
    assert report["layer12_conv_depthwise_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer12_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer12_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 convolution residual, ffn2, and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_conv_residual_writes_activation_projection_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.full((1024,), 0.8125, dtype=np.float32)
    norm_bias = np.full((1024,), -0.046875, dtype=np.float32)
    pointwise2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    pointwise2_weight[:, :, 0] = np.eye(1024, dtype=np.float32) * -0.75
    pointwise2_bias = np.full((1024,), 0.0625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_conv_depthwise_norm_weight=norm_weight,
        layer12_conv_depthwise_norm_bias=norm_bias,
        layer12_conv_pointwise2_weight=pointwise2_weight,
        layer12_conv_pointwise2_bias=pointwise2_bias,
    )
    attention_norm = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1024.0
    conv_depthwise = np.stack(
        [
            np.linspace(-0.9, 0.7, 1024, dtype=np.float32),
            np.sin(np.linspace(-1.0, 1.0, 1024, dtype=np.float32)),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer12_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer12_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer12_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer12-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    mean = conv_depthwise.mean(axis=1, keepdims=True)
    var = ((conv_depthwise - mean) ** 2).mean(axis=1, keepdims=True)
    normed = ((conv_depthwise - mean) / np.sqrt(var + 1e-5)) * norm_weight + norm_bias
    expected = attention_norm + (normed / (1.0 + np.exp(-normed))) * -0.75 + 0.0625
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer12_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer12_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer12_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer12_attention_norm_values"] == 2 * 1024
    assert report["layer12_conv_depthwise_values"] == 2 * 1024
    assert report["layer12_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer12_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer12_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-12 ffn2 and encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer12_ffn2_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), 0.015625, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer12_ffn2_intermediate_weight=intermediate_weight,
        layer12_ffn2_intermediate_bias=intermediate_bias,
        layer12_ffn2_output_weight=output_weight,
        layer12_ffn2_output_bias=output_bias,
    )
    conv_residual = np.stack(
        [
            np.linspace(-0.9, 0.9, 1024, dtype=np.float32),
            np.sin(np.linspace(-1.25, 1.25, 1024, dtype=np.float32)),
        ]
    )
    conv_residual_path = tmp_path / "w2v_layer12_conv_residual.f32"
    out_path = tmp_path / "w2v_layer12_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer12-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5)
    ffn2_output = normed / (1.0 + np.exp(-normed)) + 0.015625
    expected = conv_residual + 0.5 * ffn2_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer12_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer12_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer12_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer12_conv_residual_values"] == 2 * 1024
    assert report["layer12_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer12_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer12_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer12_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 13-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_ffn1_residual_writes_feed_forward_half_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    intermediate_weight[:1024, :] = np.eye(1024, dtype=np.float32)
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    output_weight[:, :1024] = np.eye(1024, dtype=np.float32)
    output_bias = np.full((1024,), -0.03125, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_ffn1_intermediate_weight=intermediate_weight,
        layer13_ffn1_intermediate_bias=intermediate_bias,
        layer13_ffn1_output_weight=output_weight,
        layer13_ffn1_output_bias=output_bias,
    )
    layer12_residual = np.stack(
        [
            np.linspace(-0.75, 0.95, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.2, 1.2, 1024, dtype=np.float32)),
        ]
    )
    layer12_residual_path = tmp_path / "w2v_layer12_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer13_ffn1_residual.f32"
    layer12_residual.tofile(layer12_residual_path)

    report = _run_json(
        "--clone-w2v-layer13-ffn1-residual",
        str(bundle_dir),
        str(layer12_residual_path),
        "2",
        str(out_path),
    )

    mean = layer12_residual.mean(axis=1, keepdims=True)
    var = ((layer12_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer12_residual - mean) / np.sqrt(var + 1e-5)
    ffn1_output = normed / (1.0 + np.exp(-normed)) - 0.03125
    expected = layer12_residual + 0.5 * ffn1_output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer12_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer13_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer12_ffn2_residual_values"] == 2 * 1024
    assert report["layer13_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer13_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer13_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 self-attention, convolution, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_qkv_writes_sidecars(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_q_weight=eye * -0.25,
        layer13_k_weight=eye * 0.6875,
        layer13_v_weight=eye * 1.125,
    )
    residual = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 1280.0
    residual_path = tmp_path / "w2v_layer13_ffn1_residual.f32"
    out_dir = tmp_path / "layer13_qkv"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer13-qkv",
        str(bundle_dir),
        str(residual_path),
        "2",
        str(out_dir),
    )

    q = np.fromfile(out_dir / "w2v_layer13_q.f32", dtype=np.float32).reshape(2, 1024)
    k = np.fromfile(out_dir / "w2v_layer13_k.f32", dtype=np.float32).reshape(2, 1024)
    v = np.fromfile(out_dir / "w2v_layer13_v.f32", dtype=np.float32).reshape(2, 1024)
    manifest = json.loads((out_dir / "w2v_layer13_qkv.manifest.json").read_text())
    assert report["stage"] == "tts_clone_w2v_bert_layer13_qkv"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer13_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["qkv_shape"] == "[1,2,1024]"
    assert report["layer13_ffn1_residual_values"] == 2 * 1024
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_qkv_issues"] == []
    assert report["ready_native_w2v_bert_layer13_qkv"] is True
    assert report["ready_metal_w2v_bert_layer13_qkv"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 attention context/projection/residual/norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    assert manifest["format"] == "mit2-w2v-layer13-qkv-sidecars"
    assert manifest["ready_native_w2v_bert_layer13_qkv"] is True
    assert manifest["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(q, residual * -0.25, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(k, residual * 0.6875, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(v, residual * 1.125, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_attention_writes_context(tmp_path: Path):
    q_path = tmp_path / "w2v_layer13_q.f32"
    k_path = tmp_path / "w2v_layer13_k.f32"
    v_path = tmp_path / "w2v_layer13_v.f32"
    mask_path = tmp_path / "w2v_attention_mask.u32"
    out_path = tmp_path / "w2v_layer13_context.f32"
    q = np.zeros((2, 1024), dtype=np.float32)
    k = np.zeros((2, 1024), dtype=np.float32)
    v = np.zeros((2, 1024), dtype=np.float32)
    v[0, 0] = 0.25
    v[1, 0] = 2.0
    mask = np.array([1, 1], dtype=np.uint32)
    q.tofile(q_path)
    k.tofile(k_path)
    v.tofile(v_path)
    mask.tofile(mask_path)

    report = _run_json(
        "--clone-w2v-layer13-attention",
        str(q_path),
        str(k_path),
        str(v_path),
        str(mask_path),
        "2",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    expected = np.tile((v[0] + v[1]) * 0.5, (2, 1))
    assert report["stage"] == "tts_clone_w2v_bert_layer13_attention_context"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer13_qkv_shape"] == "[1,2,1024]"
    assert report["layer13_context_shape"] == "[1,2,1024]"
    assert report["attention_mask_shape"] == "[1,2]"
    assert report["q_values"] == 2 * 1024
    assert report["k_values"] == 2 * 1024
    assert report["v_values"] == 2 * 1024
    assert report["attention_mask_values"] == 2
    assert report["layer13_context_values"] == 2 * 1024
    assert report["clone_w2v_layer13_attention_issues"] == []
    assert report["ready_native_w2v_bert_layer13_attention_context"] is True
    assert report["ready_metal_w2v_bert_layer13_attention_context"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 attention output projection/residual/norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_attention_project_writes_projection(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    eye = np.eye(1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer13_out_weight=eye * 0.59375)
    context = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 768.0
    context_path = tmp_path / "w2v_layer13_context.f32"
    out_path = tmp_path / "w2v_layer13_attention.f32"
    context.tofile(context_path)

    report = _run_json(
        "--clone-w2v-layer13-attention-project",
        str(bundle_dir),
        str(context_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_attention_project"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer13_context_shape"] == "[1,3,1024]"
    assert report["layer13_attention_projection_shape"] == "[1,3,1024]"
    assert report["layer13_context_values"] == 3 * 1024
    assert report["layer13_attention_projection_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_attention_project_issues"] == []
    assert report["ready_native_w2v_bert_layer13_attention_projection"] is True
    assert report["ready_metal_w2v_bert_layer13_attention_projection"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 attention residual/norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, context * 0.59375, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_attention_residual_writes_sum(
    tmp_path: Path,
):
    ffn1_residual = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 1536.0
    attention = np.flip(ffn1_residual, axis=0).copy() * 0.28125
    ffn1_residual_path = tmp_path / "w2v_layer13_ffn1_residual.f32"
    attention_path = tmp_path / "w2v_layer13_attention.f32"
    out_path = tmp_path / "w2v_layer13_attention_residual.f32"
    ffn1_residual.tofile(ffn1_residual_path)
    attention.tofile(attention_path)

    report = _run_json(
        "--clone-w2v-layer13-attention-residual",
        str(ffn1_residual_path),
        str(attention_path),
        "3",
        str(out_path),
    )

    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_attention_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer13_ffn1_residual_shape"] == "[1,3,1024]"
    assert report["layer13_attention_projection_shape"] == "[1,3,1024]"
    assert report["layer13_attention_residual_shape"] == "[1,3,1024]"
    assert report["layer13_ffn1_residual_values"] == 3 * 1024
    assert report["layer13_attention_projection_values"] == 3 * 1024
    assert report["layer13_attention_residual_values"] == 3 * 1024
    assert report["clone_w2v_layer13_attention_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer13_attention_residual"] is True
    assert report["ready_metal_w2v_bert_layer13_attention_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 attention norm, convolution, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, ffn1_residual + attention, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_attention_norm_writes_layernorm(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.72, 1.18, 1024, dtype=np.float32)
    beta = np.linspace(-0.035, 0.055, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_self_attn_norm_weight=gamma,
        layer13_self_attn_norm_bias=beta,
    )
    residual = np.stack(
        [
            np.linspace(-1.875, 2.125, 1024, dtype=np.float32),
            np.sin(np.linspace(-1.5, 1.5, 1024, dtype=np.float32)),
            np.linspace(1.625, -2.375, 1024, dtype=np.float32),
        ]
    )
    residual_path = tmp_path / "w2v_layer13_attention_residual.f32"
    out_path = tmp_path / "w2v_layer13_attention_norm.f32"
    residual.tofile(residual_path)

    report = _run_json(
        "--clone-w2v-layer13-attention-norm",
        str(bundle_dir),
        str(residual_path),
        "3",
        str(out_path),
    )

    mean = residual.mean(axis=1, keepdims=True)
    var = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (residual - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_attention_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer13_attention_residual_shape"] == "[1,3,1024]"
    assert report["layer13_attention_norm_shape"] == "[1,3,1024]"
    assert report["layer13_attention_residual_values"] == 3 * 1024
    assert report["layer13_attention_norm_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_attention_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer13_attention_norm"] is True
    assert report["ready_metal_w2v_bert_layer13_attention_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 convolution-module LayerNorm, convolution GLU/depthwise/residual, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_conv_norm_writes_layernorm(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.76, 1.14, 1024, dtype=np.float32)
    beta = np.linspace(-0.02, 0.08, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_conv_norm_weight=gamma,
        layer13_conv_norm_bias=beta,
    )
    attention_norm = np.stack(
        [
            np.linspace(-1.375, 2.625, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.25, 1.75, 1024, dtype=np.float32)),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer13_attention_norm.f32"
    out_path = tmp_path / "w2v_layer13_conv_norm.f32"
    attention_norm.tofile(attention_norm_path)

    report = _run_json(
        "--clone-w2v-layer13-conv-norm",
        str(bundle_dir),
        str(attention_norm_path),
        "2",
        str(out_path),
    )

    mean = attention_norm.mean(axis=1, keepdims=True)
    var = ((attention_norm - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (attention_norm - mean) / np.sqrt(var + 1e-5) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_conv_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer13_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer13_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer13_attention_norm_values"] == 2 * 1024
    assert report["layer13_conv_norm_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_conv_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer13_conv_norm"] is True
    assert report["ready_metal_w2v_bert_layer13_conv_norm"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 convolution GLU, depthwise/residual, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_conv_glu_writes_pointwise_glu(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((2048, 1024, 1), dtype=np.float32)
    for i in range(1024):
        weight[i, i, 0] = 0.625
        weight[1024 + i, i, 0] = -0.375
    bias = np.concatenate(
        [
            np.linspace(-0.015, 0.025, 1024, dtype=np.float32),
            np.linspace(0.045, -0.035, 1024, dtype=np.float32),
        ]
    )
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_conv_pointwise1_weight=weight,
        layer13_conv_pointwise1_bias=bias,
    )
    conv_norm = np.stack(
        [
            np.linspace(-1.125, 1.875, 1024, dtype=np.float32),
            np.sin(np.linspace(-0.75, 2.25, 1024, dtype=np.float32)),
        ]
    )
    conv_norm_path = tmp_path / "w2v_layer13_conv_norm.f32"
    out_path = tmp_path / "w2v_layer13_conv_glu.f32"
    conv_norm.tofile(conv_norm_path)

    report = _run_json(
        "--clone-w2v-layer13-conv-glu",
        str(bundle_dir),
        str(conv_norm_path),
        "2",
        str(out_path),
    )

    projected = conv_norm @ weight.reshape(2048, 1024).T + bias
    value = projected[:, :1024]
    gate = projected[:, 1024:]
    expected = value / (1.0 + np.exp(-gate))
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_conv_glu"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer13_conv_norm_shape"] == "[1,2,1024]"
    assert report["layer13_conv_glu_shape"] == "[1,2,1024]"
    assert report["layer13_conv_norm_values"] == 2 * 1024
    assert report["layer13_conv_glu_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_conv_glu_issues"] == []
    assert report["ready_native_w2v_bert_layer13_conv_glu"] is True
    assert report["ready_metal_w2v_bert_layer13_conv_glu"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 depthwise convolution, convolution residual, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_conv_depthwise_writes_depthwise_conv(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    weight = np.zeros((1024, 1, 31), dtype=np.float32)
    weight[:, 0, -1] = 0.5
    weight[:, 0, -2] = -0.125
    bias = np.linspace(-0.03, 0.04, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_conv_depthwise_weight=weight,
        layer13_conv_depthwise_bias=bias,
    )
    conv_glu = np.stack(
        [
            np.linspace(-0.75, 1.25, 1024, dtype=np.float32),
            np.cos(np.linspace(-0.5, 1.5, 1024, dtype=np.float32)),
            np.linspace(1.5, -0.25, 1024, dtype=np.float32),
        ]
    )
    conv_glu_path = tmp_path / "w2v_layer13_conv_glu.f32"
    out_path = tmp_path / "w2v_layer13_conv_depthwise.f32"
    conv_glu.tofile(conv_glu_path)

    report = _run_json(
        "--clone-w2v-layer13-conv-depthwise",
        str(bundle_dir),
        str(conv_glu_path),
        "3",
        str(out_path),
    )

    expected = np.empty_like(conv_glu)
    for t in range(3):
        previous = conv_glu[t - 1] if t > 0 else np.zeros((1024,), dtype=np.float32)
        expected[t] = conv_glu[t] * 0.5 + previous * -0.125 + bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(3, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_conv_depthwise"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 3
    assert report["layer13_conv_glu_shape"] == "[1,3,1024]"
    assert report["layer13_conv_depthwise_shape"] == "[1,3,1024]"
    assert report["layer13_conv_glu_values"] == 3 * 1024
    assert report["layer13_conv_depthwise_values"] == 3 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_conv_depthwise_issues"] == []
    assert report["ready_native_w2v_bert_layer13_conv_depthwise"] is True
    assert report["ready_metal_w2v_bert_layer13_conv_depthwise"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 convolution residual, ffn2, and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_conv_residual_writes_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.linspace(0.84, 1.16, 1024, dtype=np.float32)
    norm_bias = np.linspace(-0.025, 0.035, 1024, dtype=np.float32)
    pw2_weight = np.zeros((1024, 1024, 1), dtype=np.float32)
    for i in range(1024):
        pw2_weight[i, i, 0] = 0.3125
    pw2_bias = np.linspace(0.015, -0.02, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_conv_depthwise_norm_weight=norm_weight,
        layer13_conv_depthwise_norm_bias=norm_bias,
        layer13_conv_pointwise2_weight=pw2_weight,
        layer13_conv_pointwise2_bias=pw2_bias,
    )
    attention_norm = np.stack(
        [
            np.linspace(-1.2, 0.8, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.0, 1.0, 1024, dtype=np.float32)),
        ]
    )
    conv_depthwise = np.stack(
        [
            np.linspace(1.25, -0.75, 1024, dtype=np.float32),
            np.sin(np.linspace(-0.25, 2.0, 1024, dtype=np.float32)),
        ]
    )
    attention_norm_path = tmp_path / "w2v_layer13_attention_norm.f32"
    conv_depthwise_path = tmp_path / "w2v_layer13_conv_depthwise.f32"
    out_path = tmp_path / "w2v_layer13_conv_residual.f32"
    attention_norm.tofile(attention_norm_path)
    conv_depthwise.tofile(conv_depthwise_path)

    report = _run_json(
        "--clone-w2v-layer13-conv-residual",
        str(bundle_dir),
        str(attention_norm_path),
        str(conv_depthwise_path),
        "2",
        str(out_path),
    )

    mean = conv_depthwise.mean(axis=1, keepdims=True)
    var = ((conv_depthwise - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_depthwise - mean) / np.sqrt(var + 1e-5) * norm_weight + norm_bias
    activated = normed / (1.0 + np.exp(-normed))
    expected = attention_norm + activated * 0.3125 + pw2_bias
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_conv_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer13_attention_norm_shape"] == "[1,2,1024]"
    assert report["layer13_conv_depthwise_shape"] == "[1,2,1024]"
    assert report["layer13_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer13_attention_norm_values"] == 2 * 1024
    assert report["layer13_conv_depthwise_values"] == 2 * 1024
    assert report["layer13_conv_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_conv_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer13_conv_residual"] is True
    assert report["ready_metal_w2v_bert_layer13_conv_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-13 ffn2 and encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer13_ffn2_residual_writes_half_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.linspace(0.9, 1.1, 1024, dtype=np.float32)
    norm_bias = np.linspace(-0.015, 0.025, 1024, dtype=np.float32)
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    for i in range(1024):
        intermediate_weight[i, i] = 0.5
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    intermediate_bias[:1024] = np.linspace(-0.02, 0.02, 1024, dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    for i in range(1024):
        output_weight[i, i] = 0.25
    output_bias = np.linspace(0.01, -0.03, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer13_ffn2_norm_weight=norm_weight,
        layer13_ffn2_norm_bias=norm_bias,
        layer13_ffn2_intermediate_weight=intermediate_weight,
        layer13_ffn2_intermediate_bias=intermediate_bias,
        layer13_ffn2_output_weight=output_weight,
        layer13_ffn2_output_bias=output_bias,
    )
    conv_residual = np.stack(
        [
            np.linspace(-1.4, 1.6, 1024, dtype=np.float32),
            np.sin(np.linspace(-1.2, 1.4, 1024, dtype=np.float32)),
        ]
    )
    conv_residual_path = tmp_path / "w2v_layer13_conv_residual.f32"
    out_path = tmp_path / "w2v_layer13_ffn2_residual.f32"
    conv_residual.tofile(conv_residual_path)

    report = _run_json(
        "--clone-w2v-layer13-ffn2-residual",
        str(bundle_dir),
        str(conv_residual_path),
        "2",
        str(out_path),
    )

    mean = conv_residual.mean(axis=1, keepdims=True)
    var = ((conv_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (conv_residual - mean) / np.sqrt(var + 1e-5) * norm_weight + norm_bias
    intermediate = normed * 0.5 + intermediate_bias[:1024]
    activated = intermediate / (1.0 + np.exp(-intermediate))
    output = activated * 0.25 + output_bias
    expected = conv_residual + 0.5 * output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer13_ffn2_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer13_conv_residual_shape"] == "[1,2,1024]"
    assert report["layer13_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer13_conv_residual_values"] == 2 * 1024
    assert report["layer13_ffn2_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer13_ffn2_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer13_ffn2_residual"] is True
    assert report["ready_metal_w2v_bert_layer13_ffn2_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT encoder layers 14-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer14_ffn1_residual_writes_half_residual(
    tmp_path: Path,
):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.linspace(0.88, 1.12, 1024, dtype=np.float32)
    norm_bias = np.linspace(0.02, -0.015, 1024, dtype=np.float32)
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    for i in range(1024):
        intermediate_weight[i, i] = -0.375
    intermediate_bias = np.zeros((4096,), dtype=np.float32)
    intermediate_bias[:1024] = np.linspace(0.025, -0.01, 1024, dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    for i in range(1024):
        output_weight[i, i] = 0.1875
    output_bias = np.linspace(-0.02, 0.018, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer14_ffn1_norm_weight=norm_weight,
        layer14_ffn1_norm_bias=norm_bias,
        layer14_ffn1_intermediate_weight=intermediate_weight,
        layer14_ffn1_intermediate_bias=intermediate_bias,
        layer14_ffn1_output_weight=output_weight,
        layer14_ffn1_output_bias=output_bias,
    )
    layer13_residual = np.stack(
        [
            np.linspace(1.2, -1.1, 1024, dtype=np.float32),
            np.cos(np.linspace(-1.3, 1.5, 1024, dtype=np.float32)),
        ]
    )
    layer13_residual_path = tmp_path / "w2v_layer13_ffn2_residual.f32"
    out_path = tmp_path / "w2v_layer14_ffn1_residual.f32"
    layer13_residual.tofile(layer13_residual_path)

    report = _run_json(
        "--clone-w2v-layer14-ffn1-residual",
        str(bundle_dir),
        str(layer13_residual_path),
        "2",
        str(out_path),
    )

    mean = layer13_residual.mean(axis=1, keepdims=True)
    var = ((layer13_residual - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (layer13_residual - mean) / np.sqrt(var + 1e-5) * norm_weight + norm_bias
    intermediate = normed * -0.375 + intermediate_bias[:1024]
    activated = intermediate / (1.0 + np.exp(-intermediate))
    output = activated * 0.1875 + output_bias
    expected = layer13_residual + 0.5 * output
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer14_ffn1_residual"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer13_ffn2_residual_shape"] == "[1,2,1024]"
    assert report["layer14_ffn1_residual_shape"] == "[1,2,1024]"
    assert report["layer13_ffn2_residual_values"] == 2 * 1024
    assert report["layer14_ffn1_residual_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer14_ffn1_residual_issues"] == []
    assert report["ready_native_w2v_bert_layer14_ffn1_residual"] is True
    assert report["ready_metal_w2v_bert_layer14_ffn1_residual"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    assert (
        report["next_native_boundary"]
        == "native W2V-BERT layer-14 self-attention, convolution, ffn2, and encoder layers 15-16 to hidden_state_17"
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_layer14_ffn2_residual_writes_residual(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    norm_weight = np.linspace(0.65, 1.35, 1024, dtype=np.float32)
    norm_bias = np.linspace(-0.04, 0.04, 1024, dtype=np.float32)
    intermediate_weight = np.zeros((4096, 1024), dtype=np.float32)
    np.fill_diagonal(intermediate_weight[:1024, :], 0.5)
    intermediate_bias = np.linspace(-0.5, 0.5, 4096, dtype=np.float32)
    output_weight = np.zeros((1024, 4096), dtype=np.float32)
    np.fill_diagonal(output_weight[:, :1024], -0.25)
    output_bias = np.linspace(0.1, -0.1, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, layer14_ffn2_norm_weight=norm_weight, layer14_ffn2_norm_bias=norm_bias, layer14_ffn2_intermediate_weight=intermediate_weight, layer14_ffn2_intermediate_bias=intermediate_bias, layer14_ffn2_output_weight=output_weight, layer14_ffn2_output_bias=output_bias)
    cv = np.stack([np.linspace(-1.0, 1.0, 1024, dtype=np.float32), np.cos(np.linspace(-1.0, 2.0, 1024, dtype=np.float32))])
    cv_path = tmp_path / "cv.f32"; out_path = tmp_path / "out.f32"; cv.tofile(cv_path)
    r = _run_json("--clone-w2v-layer14-ffn2-residual", str(bundle_dir), str(cv_path), "2", str(out_path))
    m = cv.mean(axis=1, keepdims=True)
    v = ((cv - m) ** 2).mean(axis=1, keepdims=True)
    n = ((cv - m) / np.sqrt(v + 1e-5)) * norm_weight + norm_bias
    inter = n @ intermediate_weight[:1024, :].T + intermediate_bias[:1024]
    inter_f = np.zeros((2, 4096), dtype=np.float32); inter_f[:, :1024] = inter
    act = inter_f / (1.0 + np.exp(-inter_f))
    expected = cv + 0.5 * (act @ output_weight.T + output_bias)
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert r["ok"] is True
    assert r["stage"] == "tts_clone_w2v_bert_layer14_ffn2_residual"
    assert r["w2v_tokens"] == 2
    assert r["ready_native_w2v_bert_layer14_ffn2_residual"] is True
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

def test_mit2_tts_clone_w2v_chain_to_spk_cond(tmp_path: Path):
    """Chain: layer17 final norm -> w2v normalize -> spk_cond_emb."""
    d = tmp_path / "bundle"
    g = np.linspace(0.6, 1.4, 1024, dtype=np.float32)
    b = np.linspace(-0.05, 0.05, 1024, dtype=np.float32)
    sm = np.linspace(-0.5, 0.5, 1024, dtype=np.float32)
    ss = np.linspace(0.5, 1.5, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(d, layer17_final_norm_weight=g, layer17_final_norm_bias=b, stats_mean=sm, stats_std=ss)
    l16 = np.stack([np.linspace(-1., 1., 1024, dtype=np.float32), np.cos(np.linspace(-1., 2., 1024, dtype=np.float32))])
    l16p = tmp_path / "l16.f32"; l16.tofile(l16p)
    l17p = tmp_path / "l17.f32"
    assert _run_json("--clone-w2v-layer17-final-norm", str(d), str(l16p), "2", str(l17p))["ok"]
    sp = tmp_path / "sp.f32"
    assert _run_json("--clone-w2v-normalize", str(d), str(l17p), "2", str(sp))["ok"]
    l17 = np.fromfile(l17p, dtype=np.float32)
    sp_out = np.fromfile(sp, dtype=np.float32)
    np.testing.assert_allclose(sp_out.reshape(2,1024), (l17.reshape(2,1024)-sm)/ss, rtol=0, atol=1e-5)

def test_mit2_tts_clone_w2v_layer17_final_norm_writes_hidden_state_17(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    gamma = np.linspace(0.6, 1.4, 1024, dtype=np.float32)
    beta = np.linspace(-0.05, 0.05, 1024, dtype=np.float32)
    _write_w2v_bert_contract_bundle(
        bundle_dir,
        layer17_final_norm_weight=gamma,
        layer17_final_norm_bias=beta,
    )
    layer16 = np.tile(np.linspace(-1.0, 1.0, 1024, dtype=np.float32), (2, 1))
    layer16_path = tmp_path / "w2v_layer16.f32"
    out_path = tmp_path / "w2v_hidden_state_17.f32"
    layer16.tofile(layer16_path)

    report = _run_json(
        "--clone-w2v-layer17-final-norm",
        str(bundle_dir),
        str(layer16_path),
        "2",
        str(out_path),
    )

    mean = layer16.mean(axis=1, keepdims=True)
    var = ((layer16 - mean) ** 2).mean(axis=1, keepdims=True)
    expected = ((layer16 - mean) / np.sqrt(var + 1e-5)) * gamma + beta
    actual = np.fromfile(out_path, dtype=np.float32).reshape(2, 1024)
    assert report["stage"] == "tts_clone_w2v_bert_layer17_final_norm"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["layer16_shape"] == "[1,2,1024]"
    assert report["w2v_hidden_state_17_shape"] == "[1,2,1024]"
    assert report["layer16_values"] == 2 * 1024
    assert report["w2v_hidden_state_17_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_layer17_final_norm_issues"] == []
    assert report["ready_native_w2v_bert_layer17_final_norm"] is True
    assert report["ready_metal_w2v_bert_layer17_final_norm"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert report["ready_native_voice_clone"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_mit2_tts_clone_w2v_normalize_writes_spk_cond(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    mean = np.full((1024,), 1.0, dtype=np.float32)
    std = np.full((1024,), 2.0, dtype=np.float32)
    _write_w2v_bert_contract_bundle(bundle_dir, stats_mean=mean, stats_std=std)
    hidden = np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024)
    hidden_path = tmp_path / "w2v_hidden_state_17.f32"
    out_path = tmp_path / "spk_cond_emb.f32"
    hidden.tofile(hidden_path)

    report = _run_json(
        "--clone-w2v-normalize",
        str(bundle_dir),
        str(hidden_path),
        "2",
        str(out_path),
    )

    expected = (hidden.reshape(-1) - 1.0) / 2.0
    actual = np.fromfile(out_path, dtype=np.float32)
    assert report["stage"] == "tts_clone_w2v_bert_normalize"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["w2v_tokens"] == 2
    assert report["w2v_hidden_shape"] == "[1,2,1024]"
    assert report["spk_cond_shape"] == "[1,2,1024]"
    assert report["spk_cond_values"] == 2 * 1024
    assert report["has_w2v_bert_model_contract"] is True
    assert report["w2v_bert_required_tensors_present"] == 412
    assert report["w2v_bert_contract_issues"] == []
    assert report["clone_w2v_normalize_issues"] == []
    assert report["ready_native_w2v_bert_stats_normalize"] is True
    assert report["ready_metal_w2v_bert_stats_normalize"] is (report["execution_backend"] == "metal")
    assert report["ready_native_w2v_bert_semantic_features"] is False
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_mit2_tts_clone_w2v_normalize_reports_missing_contract(tmp_path: Path):
    bundle_dir = tmp_path / "model_bundle"
    write_bundle(
        bundle_dir,
        [("unrelated.weight", np.zeros((1,), dtype=np.float32), "unrelated")],
        metadata={"unit": "missing-w2v"},
    )
    hidden_path = tmp_path / "w2v_hidden_state_17.f32"
    out_path = tmp_path / "spk_cond_emb.f32"
    np.zeros((1024,), dtype=np.float32).tofile(hidden_path)

    report = _run_json(
        "--clone-w2v-normalize",
        str(bundle_dir),
        str(hidden_path),
        "1",
        str(out_path),
        check=False,
    )

    assert report["stage"] == "tts_clone_w2v_bert_normalize"
    assert report["ok"] is False
    assert report["has_w2v_bert_model_contract"] is False
    assert report["w2v_bert_required_tensor_count"] == 412
    assert report["w2v_bert_required_tensors_present"] == 0
    assert "w2v_bert_model_contract_not_ready" in report["clone_w2v_normalize_issues"]
    assert "missing_w2v_bert_tensor:w2v_bert.feature_projection.layer_norm.weight" in report[
        "w2v_bert_contract_issues"
    ]
    assert report["ready_native_w2v_bert_stats_normalize"] is False
    assert report["ready_metal_w2v_bert_stats_normalize"] is False
    assert report["ready_native_w2v_bert_semantic_features"] is False
    assert not out_path.exists()


def test_mit2_tts_preflight_reports_ready_to_synthesize_surface():
    if not (VOICE_BUNDLE / "manifest.json").exists():
        pytest.skip("test voice bundle is not available")

    report = _run_last_json("--preflight", str(MODEL_BUNDLE), str(VOICE_BUNDLE), "1=1000℃")

    assert report["stage"] == "tts_product_preflight"
    assert report["ok"] is True
    assert report["model_contract_ok"] is True
    assert report["voice_contract_ok"] is True
    assert report["ready_native_cjk_text"] is True
    assert report["ready_cached_voice_tts_cjk_text"] is True
    assert report["ready_to_synthesize"] is True
    assert report["token_ids"] == [
        10201,
        7,
        10201,
        4275,
        10201,
        85,
        10201,
        7,
        10201,
        3878,
        10201,
        6500,
        10201,
        2313,
        10201,
        2942,
        10201,
        1726,
    ]
    assert [preset["name"] for preset in report["presets"]] == ["smoke", "short", "standard"]
    by_name = {preset["name"]: preset for preset in report["presets"]}
    assert by_name["smoke"]["resource_plan"]["segment_count"] == by_name["smoke"]["segment_count"]
    assert by_name["smoke"]["resource_plan"]["max_segment_text_tokens"] == 4
    assert len(by_name["smoke"]["resource_plan"]["segment_plans"]) == 5
    assert by_name["short"]["resource_plan"]["aggregate_segment_plan"]["max_codes"] == 32
    assert by_name["short"]["resource_plan"]["aggregate_segment_plan"]["text_tokens"] == 18
    assert by_name["standard"]["resource_plan"]["aggregate_segment_plan"]["max_codes"] == 128
    assert by_name["standard"]["resource_plan"]["aggregate_segment_plan"]["planned_scratch_capacity_bytes"] > 0
    assert by_name["standard"]["resource_plan"]["aggregate_segment_plan"]["scratch_reuse_saves_bytes"] > 0
    assert report["start_sh_replacement_audit"]["can_replace_start_sh_cached_voice_cjk"] is True


def test_mit2_tts_preflight_rejects_unsupported_text_before_synthesis():
    if not (VOICE_BUNDLE / "manifest.json").exists():
        pytest.skip("test voice bundle is not available")

    report = _run_last_json("--preflight", str(MODEL_BUNDLE), str(VOICE_BUNDLE), "1=2026℃", check=False)

    assert report["stage"] == "tts_product_preflight"
    assert report["ok"] is False
    assert report["model_contract_ok"] is True
    assert report["voice_contract_ok"] is True
    assert report["ready_native_cjk_text"] is False
    assert report["ready_cached_voice_tts_cjk_text"] is False
    assert report["ready_to_synthesize"] is False
    assert report["python_boundary"] == "full TextNormalizer/SentencePiece for general text"
    assert "2026" in report["text_error"]
    assert report["start_sh_replacement_audit"]["can_replace_start_sh_cached_voice_cjk"] is False


def test_mit2_tts_clone_preflight_accepts_pcm16_mono_wav(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))

    report = _run_json("--clone-preflight", str(wav_path))

    assert report["stage"] == "tts_clone_audio_preflight"
    assert report["ok"] is True
    assert report["audio_format"] == "pcm16_mono_wav"
    assert report["sample_rate"] == 22050
    assert report["samples"] == 2205
    assert report["frame_bytes"] == 4410
    assert report["audio_peak_abs_i16"] == 4096
    assert report["audio_peak_normalized"] > 0.12
    assert report["audio_rms_normalized"] > 0.12
    assert report["nonzero_sample_ratio"] == 1
    assert report["near_silence_sample_ratio"] == 0
    assert report["clipping_sample_ratio"] == 0
    assert report["recommended_resample_rate"] == 16000
    assert report["ready_native_clone_audio_quality"] is True
    assert report["quality_issues"] == []
    assert report["ready_native_clone_audio_input"] is True
    assert report["ready_native_voice_clone"] is False
    assert report["python_boundary_after_input"] == "clone-time mel extraction and speech encoders for voice tensor creation"


def test_mit2_tts_clone_preflight_rejects_silent_audio(tmp_path: Path):
    wav_path = tmp_path / "silent_ref.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x00" * 2205))

    report = _run_json("--clone-preflight", str(wav_path), check=False)

    assert report["stage"] == "tts_clone_audio_preflight"
    assert report["ok"] is False
    assert report["audio_format"] == "pcm16_mono_wav"
    assert report["ready_native_clone_audio_quality"] is False
    assert report["ready_native_clone_audio_input"] is False
    assert "silent_or_near_silent_audio" in report["quality_issues"]


def test_mit2_tts_clone_preflight_rejects_invalid_audio(tmp_path: Path):
    wav_path = tmp_path / "not_audio.wav"
    wav_path.write_bytes(b"not a wav")

    report = _run_json("--clone-preflight", str(wav_path), check=False)

    assert report["stage"] == "tts_clone_audio_preflight"
    assert report["ok"] is False
    assert report["ready_native_clone_audio_input"] is False
    assert report["ready_native_voice_clone"] is False
    assert report["expected_audio_format"] == "pcm16_mono_wav"


def test_mit2_tts_clone_preprocess_writes_resampled_f32(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    out_path = tmp_path / "clone_ref.16k.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))

    report = _run_json("--clone-preprocess", str(wav_path), str(out_path))

    assert report["stage"] == "tts_clone_audio_preprocess"
    assert report["ok"] is True
    assert report["source_sample_rate"] == 22050
    assert report["target_sample_rate"] == 16000
    assert report["source_samples"] == 2205
    assert report["preprocessed_audio_format"] == "f32_mono_raw"
    assert report["preprocessed_sample_rate"] == 16000
    assert report["preprocessed_samples"] == 1600
    assert report["ready_native_clone_audio_preprocess"] is True
    assert report["ready_native_voice_clone"] is False
    manifest_path = Path(report["preprocess_manifest"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "mit2-clone-audio-preprocess"
    assert manifest["version"] == 1
    assert manifest["output_f32"] == str(out_path)
    assert manifest["target_sample_rate"] == 16000
    assert manifest["preprocessed_samples"] == 1600
    assert manifest["ready_native_clone_audio_preprocess"] is True
    assert manifest["ready_native_voice_clone"] is False
    data = out_path.read_bytes()
    assert len(data) == 1600 * 4
    first, last = struct.unpack("<ff", data[:4] + data[-4:])
    assert first == pytest.approx(0.125)
    assert last == pytest.approx(0.125)


def test_mit2_tts_clone_readiness_validates_preprocess_manifest(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    out_path = tmp_path / "clone_ref.16k.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    preprocess = _run_json("--clone-preprocess", str(wav_path), str(out_path))

    report = _run_json("--clone-readiness", preprocess["preprocess_manifest"])

    assert report["stage"] == "tts_clone_readiness"
    assert report["ok"] is True
    assert report["manifest_parsed"] is True
    assert report["preprocessed_audio_format"] == "f32_mono_raw"
    assert report["preprocessed_sample_rate"] == 16000
    assert report["preprocessed_samples"] == 1600
    assert report["ready_native_clone_audio_preprocess"] is True
    assert report["ready_native_voice_bundle_creation"] is False
    assert report["ready_native_voice_clone"] is False
    assert report["clone_readiness_issues"] == []
    assert {tensor["name"] for tensor in report["required_voice_bundle_tensors"]} == {
        "spk_cond_emb",
        "s2mel_style",
        "s2mel_prompt",
        "mel",
    }


def test_mit2_tts_clone_readiness_rejects_tampered_preprocess_output(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    out_path = tmp_path / "clone_ref.16k.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    preprocess = _run_json("--clone-preprocess", str(wav_path), str(out_path))
    out_path.write_bytes(out_path.read_bytes() + b"bad")

    report = _run_json("--clone-readiness", preprocess["preprocess_manifest"], check=False)

    assert report["stage"] == "tts_clone_readiness"
    assert report["ok"] is False
    assert report["ready_native_clone_audio_preprocess"] is False
    assert "preprocessed_output_f32_size_mismatch" in report["clone_readiness_issues"]
    assert "preprocessed_output_f32_sha256_mismatch" in report["clone_readiness_issues"]
    assert report["ready_native_voice_clone"] is False


def test_mit2_tts_clone_extract_mel_writes_voice_mel_sidecar(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    out_path = tmp_path / "clone_ref.16k.f32"
    mel_path = tmp_path / "mel.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    preprocess = _run_json("--clone-preprocess", str(wav_path), str(out_path))

    report = _run_json("--clone-extract-mel", preprocess["preprocess_manifest"], str(mel_path))

    assert report["stage"] == "tts_clone_extract_mel"
    assert report["ok"] is True
    assert report["source_sample_rate"] == 16000
    assert report["mel_sample_rate"] == 22050
    assert report["n_fft"] == 1024
    assert report["win_length"] == 1024
    assert report["hop_length"] == 256
    assert report["n_mels"] == 80
    assert report["mel_frames"] == 8
    assert report["mel_values"] == 80 * 8
    assert report["mel_layout"] == "[1,80,frames]_row_major_flat"
    assert report["ready_native_clone_mel_extraction"] is True
    assert report["ready_native_voice_clone"] is False
    assert mel_path.exists()
    assert len(mel_path.read_bytes()) == 80 * 8 * 4
    manifest = json.loads(Path(report["mel_manifest"]).read_text())
    assert manifest["format"] == "mit2-clone-mel-extract"
    assert manifest["output_mel_f32"] == str(mel_path)
    assert manifest["mel_frames"] == 8
    assert manifest["ready_native_clone_mel_extraction"] is True


def test_mit2_tts_clone_extract_fbank_writes_campplus_input_sidecar(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    out_path = tmp_path / "clone_ref.16k.f32"
    fbank_path = tmp_path / "fbank.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    preprocess = _run_json("--clone-preprocess", str(wav_path), str(out_path))

    report = _run_json("--clone-extract-fbank", preprocess["preprocess_manifest"], str(fbank_path))

    assert report["stage"] == "tts_clone_extract_fbank"
    assert report["ok"] is True
    assert report["sample_rate"] == 16000
    assert report["frame_length_samples"] == 400
    assert report["frame_shift_samples"] == 160
    assert report["frame_length_ms"] == 25
    assert report["frame_shift_ms"] == 10
    assert report["n_fft"] == 512
    assert report["num_mel_bins"] == 80
    assert report["dither"] == 0
    assert report["mean_normalized"] is True
    assert report["fbank_frames"] == 8
    assert report["fbank_values"] == 80 * 8
    assert report["fbank_layout"] == "[frames,80]_row_major_flat"
    assert report["ready_native_clone_fbank_extraction"] is True
    assert report["ready_native_voice_clone"] is False
    assert abs(report["fbank_mean"]) < 1e-5
    assert fbank_path.exists()
    assert len(fbank_path.read_bytes()) == 80 * 8 * 4
    manifest = json.loads(Path(report["fbank_manifest"]).read_text())
    assert manifest["format"] == "mit2-clone-fbank-extract"
    assert manifest["output_fbank_f32"] == str(fbank_path)
    assert manifest["fbank_frames"] == 8
    assert manifest["feature_contract"] == "kaldi_style_fbank_for_campplus"
    assert manifest["ready_native_clone_fbank_extraction"] is True


def test_mit2_tts_clone_prepare_features_writes_native_clone_sidecars(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    output_dir = tmp_path / "features"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))

    report = _run_json("--clone-prepare-features", str(wav_path), str(output_dir))

    assert report["stage"] == "tts_clone_prepare_features"
    assert report["ok"] is True
    assert report["preprocessed_sample_rate"] == 16000
    assert report["preprocessed_samples"] == 1600
    assert report["mel_frames"] == 8
    assert report["mel_values"] == 80 * 8
    assert report["fbank_frames"] == 8
    assert report["fbank_values"] == 80 * 8
    assert report["ready_native_clone_audio_preprocess"] is True
    assert report["ready_native_clone_mel_extraction"] is True
    assert report["ready_native_clone_fbank_extraction"] is True
    assert report["ready_native_clone_feature_prep"] is True
    assert report["ready_native_voice_clone"] is False
    assert Path(report["preprocessed_output_f32"]).exists()
    assert Path(report["output_mel_f32"]).exists()
    assert Path(report["output_fbank_f32"]).exists()
    assert len(Path(report["preprocessed_output_f32"]).read_bytes()) == 1600 * 4
    assert len(Path(report["output_mel_f32"]).read_bytes()) == 80 * 8 * 4
    assert len(Path(report["output_fbank_f32"]).read_bytes()) == 80 * 8 * 4
    features_manifest = json.loads(Path(report["features_manifest"]).read_text())
    assert features_manifest["format"] == "mit2-clone-feature-prep"
    assert features_manifest["ready_native_clone_feature_prep"] is True
    assert features_manifest["ready_native_voice_clone"] is False
    preprocess_manifest = json.loads(Path(report["preprocess_manifest"]).read_text())
    assert preprocess_manifest["format"] == "mit2-clone-audio-preprocess"
    assert preprocess_manifest["ready_native_clone_audio_preprocess"] is True
    mel_manifest = json.loads(Path(report["mel_manifest"]).read_text())
    assert mel_manifest["format"] == "mit2-clone-mel-extract"
    assert mel_manifest["ready_native_clone_mel_extraction"] is True
    fbank_manifest = json.loads(Path(report["fbank_manifest"]).read_text())
    assert fbank_manifest["format"] == "mit2-clone-fbank-extract"
    assert fbank_manifest["ready_native_clone_fbank_extraction"] is True
    readiness = _run_json("--clone-feature-readiness", report["features_manifest"])
    assert readiness["stage"] == "tts_clone_feature_readiness"
    assert readiness["ok"] is True
    assert readiness["ready_native_clone_audio_preprocess"] is True
    assert readiness["ready_native_clone_mel_extraction"] is True
    assert readiness["ready_native_clone_fbank_extraction"] is True
    assert readiness["ready_native_clone_feature_prep"] is True
    assert readiness["ready_native_voice_clone"] is False
    assert readiness["clone_feature_readiness_issues"] == []


def test_mit2_tts_clone_command_creates_fast_voice_bundle(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    wav_path = tmp_path / "clone_ref.wav"
    bundle_path = tmp_path / "fast_voice_bundle"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))

    report = _run_last_json("--clone", str(wav_path), str(bundle_path))

    assert report["stage"] == "tts_clone"
    assert report["ok"] is True
    assert report["format"] == "MIT2"
    assert report["output_voice_bundle"] == str(bundle_path)
    assert report["tensor_count"] == 4
    assert report["spk_cond_tokens"] == 3
    assert report["prompt_tokens"] == 8
    assert report["mel_frames"] == 8
    assert report["ready_native_voice_bundle_creation"] is True
    assert report["ready_native_voice_clone"] is False
    assert bundle_path.exists()
    assert bundle_path.is_file() or (bundle_path / "manifest.json").exists()
    readiness = _run_last_json("--readiness", str(MODEL_BUNDLE), str(bundle_path))
    assert readiness["stage"] == "tts_product_readiness"
    assert readiness["ok"] is True
    assert readiness["voice_prompt_tokens"] == 8


def test_mit2_tts_clone_feature_readiness_rejects_tampered_sidecar(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    output_dir = tmp_path / "features"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(output_dir))
    fbank_path = Path(prepared["output_fbank_f32"])
    fbank_path.write_bytes(fbank_path.read_bytes() + b"bad")

    report = _run_json("--clone-feature-readiness", prepared["features_manifest"], check=False)

    assert report["stage"] == "tts_clone_feature_readiness"
    assert report["ok"] is False
    assert report["ready_native_clone_feature_prep"] is False
    assert report["ready_native_voice_clone"] is False
    assert "feature_output_fbank_f32_size_mismatch" in report["clone_feature_readiness_issues"]
    assert "feature_output_fbank_f32_sha256_mismatch" in report["clone_feature_readiness_issues"]


def test_mit2_tts_clone_campplus_style_readiness_accepts_style_boundary(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    model_bundle = tmp_path / "campplus_model_bundle"
    style_path = tmp_path / "s2mel_style.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    _write_campplus_contract_bundle(model_bundle)
    _write_f32(style_path, [0.02] * 192)

    report = _run_json(
        "--clone-campplus-style-readiness",
        str(model_bundle),
        prepared["features_manifest"],
        str(style_path),
    )

    assert report["stage"] == "tts_clone_campplus_style_readiness"
    assert report["ok"] is True
    assert report["has_campplus_model_contract"] is True
    assert report["campplus_contract_issues"] == []
    assert report["fbank_frames"] == prepared["fbank_frames"]
    assert report["fbank_values"] == 80 * prepared["fbank_frames"]
    assert report["expected_style"]["shape"] == "[1,192]"
    assert report["ready_native_clone_feature_prep"] is True
    assert report["ready_native_clone_fbank_extraction"] is True
    assert report["ready_reference_campplus_style"] is True
    assert report["ready_native_campplus_style_forward"] is False
    assert report["ready_native_voice_clone"] is False
    assert report["clone_campplus_style_readiness_issues"] == []


def test_mit2_tts_clone_campplus_style_from_features_writes_style_sidecar(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    model_bundle = tmp_path / "campplus_model_bundle"
    style_path = tmp_path / "s2mel_style.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    _write_campplus_contract_bundle(model_bundle)

    report = _run_json(
        "--clone-campplus-style-from-features",
        str(model_bundle),
        prepared["features_manifest"],
        str(style_path),
    )

    assert report["stage"] == "tts_clone_campplus_style_from_features"
    assert report["ok"] is True
    assert report["execution_backend"] == "cpu"
    assert report["has_campplus_model_contract"] is True
    assert report["campplus_contract_issues"] == []
    assert report["fbank_frames"] == prepared["fbank_frames"]
    assert report["fbank_values"] == 80 * prepared["fbank_frames"]
    assert report["s2mel_style_shape"] == "[1,192]"
    assert report["s2mel_style_values"] == 192
    assert report["ready_native_clone_feature_prep"] is True
    assert report["ready_native_clone_fbank_extraction"] is True
    assert report["ready_native_campplus_style_forward"] is True
    assert report["ready_native_voice_clone"] is False
    assert report["clone_campplus_style_from_features_issues"] == []

    style = np.fromfile(style_path, dtype=np.float32)
    assert style.shape == (192,)
    assert np.all(np.isfinite(style))
    assert np.allclose(style, 0.0)

    readiness = _run_json(
        "--clone-campplus-style-readiness",
        str(model_bundle),
        prepared["features_manifest"],
        str(style_path),
    )
    assert readiness["ok"] is True
    assert readiness["ready_reference_campplus_style"] is True


def test_mit2_tts_clone_campplus_head_golden_replays_first_head_boundary(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    model_bundle = tmp_path / "campplus_model_bundle"
    golden_dir = tmp_path / "campplus_golden"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    _write_campplus_contract_bundle(model_bundle)

    fbank = np.fromfile(prepared["output_fbank_f32"], dtype=np.float32).reshape(prepared["fbank_frames"], 80)
    frames = fbank.shape[0]
    expected = np.zeros((32, 80, frames), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(1.0 + 1.0e-5))
    shift = np.float32(-scale)
    padded = np.pad(fbank.T, ((1, 1), (1, 1)), mode="constant")
    for freq in range(80):
        for frame in range(frames):
            window_sum = np.float32(padded[freq : freq + 3, frame : frame + 3].sum())
            expected[:, freq, frame] = np.maximum(np.float32(window_sum * scale + shift), np.float32(0.0))
    expected_layer1 = expected[:, ::2, :]
    expected_layer2 = expected_layer1[:, ::2, :]
    expected_conv2 = np.zeros((32, 10, frames), dtype=np.float32)
    expected_output = expected_conv2.reshape(320, frames)
    tdnn_frames = ((frames + 4 - 5) // 2) + 1
    expected_tdnn = np.zeros((128, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd1 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd1 = np.zeros((160, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd2 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd2 = np.zeros((192, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd3 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd3 = np.zeros((224, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd4 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd4 = np.zeros((256, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd5 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd5 = np.zeros((288, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd6 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd6 = np.zeros((320, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd7 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd7 = np.zeros((352, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd8 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd8 = np.zeros((384, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd9 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd9 = np.zeros((416, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd10 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd10 = np.zeros((448, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd11 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd11 = np.zeros((480, tdnn_frames), dtype=np.float32)
    expected_block1_tdnnd12 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block1_after_tdnnd12 = np.zeros((512, tdnn_frames), dtype=np.float32)
    expected_transit1 = np.zeros((256, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd1 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd1 = np.zeros((288, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd2 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd2 = np.zeros((320, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd3 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd3 = np.zeros((352, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd4 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd4 = np.zeros((384, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd5 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd5 = np.zeros((416, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd6 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd6 = np.zeros((448, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd7 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd7 = np.zeros((480, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd8 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd8 = np.zeros((512, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd9 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd9 = np.zeros((544, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd10 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd10 = np.zeros((576, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd11 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd11 = np.zeros((608, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd12 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd12 = np.zeros((640, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd13 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd13 = np.zeros((672, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd14 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd14 = np.zeros((704, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd15 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd15 = np.zeros((736, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd16 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd16 = np.zeros((768, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd17 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd17 = np.zeros((800, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd18 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd18 = np.zeros((832, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd19 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd19 = np.zeros((864, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd20 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd20 = np.zeros((896, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd21 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd21 = np.zeros((928, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd22 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd22 = np.zeros((960, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd23 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd23 = np.zeros((992, tdnn_frames), dtype=np.float32)
    expected_block2_tdnnd24 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block2_after_tdnnd24 = np.zeros((1024, tdnn_frames), dtype=np.float32)
    expected_transit2 = np.zeros((512, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd1 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd1 = np.zeros((544, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd2 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd2 = np.zeros((576, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd3 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd3 = np.zeros((608, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd4 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd4 = np.zeros((640, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd5 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd5 = np.zeros((672, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd6 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd6 = np.zeros((704, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd7 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd7 = np.zeros((736, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd8 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd8 = np.zeros((768, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd9 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd9 = np.zeros((800, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd10 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd10 = np.zeros((832, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd11 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd11 = np.zeros((864, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd12 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd12 = np.zeros((896, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd13 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd13 = np.zeros((928, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd14 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd14 = np.zeros((960, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd15 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd15 = np.zeros((992, tdnn_frames), dtype=np.float32)
    expected_block3_tdnnd16 = np.zeros((32, tdnn_frames), dtype=np.float32)
    expected_block3_after_tdnnd16 = np.zeros((1024, tdnn_frames), dtype=np.float32)
    expected_transit3 = np.zeros((512, tdnn_frames), dtype=np.float32)
    expected_out_nonlinear = np.zeros((512, tdnn_frames), dtype=np.float32)
    expected_stats = np.zeros((1024,), dtype=np.float32)
    expected_dense = np.zeros((192,), dtype=np.float32)
    golden_dir.mkdir()
    expected.reshape(1, 32, 80, frames).tofile(golden_dir / "campplus_head_conv1_bn_relu.f32")
    expected_layer1.reshape(1, 32, 40, frames).tofile(golden_dir / "campplus_head_layer1.f32")
    expected_layer2.reshape(1, 32, 20, frames).tofile(golden_dir / "campplus_head_layer2.f32")
    expected_conv2.reshape(1, 32, 10, frames).tofile(golden_dir / "campplus_head_conv2_bn_relu.f32")
    expected_output.reshape(1, 320, frames).tofile(golden_dir / "campplus_head_output.f32")
    expected_tdnn.reshape(1, 128, tdnn_frames).tofile(golden_dir / "campplus_xvector_tdnn.f32")
    expected_block1_tdnnd1.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd1.f32")
    expected_block1_after_tdnnd1.reshape(1, 160, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd1.f32"
    )
    expected_block1_tdnnd2.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd2.f32")
    expected_block1_after_tdnnd2.reshape(1, 192, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd2.f32"
    )
    expected_block1_tdnnd3.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd3.f32")
    expected_block1_after_tdnnd3.reshape(1, 224, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd3.f32"
    )
    expected_block1_tdnnd4.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd4.f32")
    expected_block1_after_tdnnd4.reshape(1, 256, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd4.f32"
    )
    expected_block1_tdnnd5.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd5.f32")
    expected_block1_after_tdnnd5.reshape(1, 288, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd5.f32"
    )
    expected_block1_tdnnd6.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd6.f32")
    expected_block1_after_tdnnd6.reshape(1, 320, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd6.f32"
    )
    expected_block1_tdnnd7.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd7.f32")
    expected_block1_after_tdnnd7.reshape(1, 352, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd7.f32"
    )
    expected_block1_tdnnd8.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd8.f32")
    expected_block1_after_tdnnd8.reshape(1, 384, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd8.f32"
    )
    expected_block1_tdnnd9.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd9.f32")
    expected_block1_after_tdnnd9.reshape(1, 416, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd9.f32"
    )
    expected_block1_tdnnd10.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd10.f32")
    expected_block1_after_tdnnd10.reshape(1, 448, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd10.f32"
    )
    expected_block1_tdnnd11.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd11.f32")
    expected_block1_after_tdnnd11.reshape(1, 480, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd11.f32"
    )
    expected_block1_tdnnd12.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block1_tdnnd12.f32")
    expected_block1_after_tdnnd12.reshape(1, 512, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block1_after_tdnnd12.f32"
    )
    expected_transit1.reshape(1, 256, tdnn_frames).tofile(golden_dir / "campplus_xvector_transit1.f32")
    expected_block2_tdnnd1.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd1.f32")
    expected_block2_after_tdnnd1.reshape(1, 288, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd1.f32"
    )
    expected_block2_tdnnd2.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd2.f32")
    expected_block2_after_tdnnd2.reshape(1, 320, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd2.f32"
    )
    expected_block2_tdnnd3.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd3.f32")
    expected_block2_after_tdnnd3.reshape(1, 352, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd3.f32"
    )
    expected_block2_tdnnd4.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd4.f32")
    expected_block2_after_tdnnd4.reshape(1, 384, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd4.f32"
    )
    expected_block2_tdnnd5.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd5.f32")
    expected_block2_after_tdnnd5.reshape(1, 416, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd5.f32"
    )
    expected_block2_tdnnd6.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd6.f32")
    expected_block2_after_tdnnd6.reshape(1, 448, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd6.f32"
    )
    expected_block2_tdnnd7.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd7.f32")
    expected_block2_after_tdnnd7.reshape(1, 480, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd7.f32"
    )
    expected_block2_tdnnd8.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd8.f32")
    expected_block2_after_tdnnd8.reshape(1, 512, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd8.f32"
    )
    expected_block2_tdnnd9.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd9.f32")
    expected_block2_after_tdnnd9.reshape(1, 544, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd9.f32"
    )
    expected_block2_tdnnd10.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd10.f32")
    expected_block2_after_tdnnd10.reshape(1, 576, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd10.f32"
    )
    expected_block2_tdnnd11.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd11.f32")
    expected_block2_after_tdnnd11.reshape(1, 608, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd11.f32"
    )
    expected_block2_tdnnd12.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd12.f32")
    expected_block2_after_tdnnd12.reshape(1, 640, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd12.f32"
    )
    expected_block2_tdnnd13.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd13.f32")
    expected_block2_after_tdnnd13.reshape(1, 672, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd13.f32"
    )
    expected_block2_tdnnd14.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd14.f32")
    expected_block2_after_tdnnd14.reshape(1, 704, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd14.f32"
    )
    expected_block2_tdnnd15.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd15.f32")
    expected_block2_after_tdnnd15.reshape(1, 736, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd15.f32"
    )
    expected_block2_tdnnd16.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd16.f32")
    expected_block2_after_tdnnd16.reshape(1, 768, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd16.f32"
    )
    expected_block2_tdnnd17.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd17.f32")
    expected_block2_after_tdnnd17.reshape(1, 800, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd17.f32"
    )
    expected_block2_tdnnd18.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd18.f32")
    expected_block2_after_tdnnd18.reshape(1, 832, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd18.f32"
    )
    expected_block2_tdnnd19.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd19.f32")
    expected_block2_after_tdnnd19.reshape(1, 864, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd19.f32"
    )
    expected_block2_tdnnd20.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd20.f32")
    expected_block2_after_tdnnd20.reshape(1, 896, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd20.f32"
    )
    expected_block2_tdnnd21.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd21.f32")
    expected_block2_after_tdnnd21.reshape(1, 928, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd21.f32"
    )
    expected_block2_tdnnd22.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd22.f32")
    expected_block2_after_tdnnd22.reshape(1, 960, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd22.f32"
    )
    expected_block2_tdnnd23.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd23.f32")
    expected_block2_after_tdnnd23.reshape(1, 992, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd23.f32"
    )
    expected_block2_tdnnd24.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block2_tdnnd24.f32")
    expected_block2_after_tdnnd24.reshape(1, 1024, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block2_after_tdnnd24.f32"
    )
    expected_transit2.reshape(1, 512, tdnn_frames).tofile(golden_dir / "campplus_xvector_transit2.f32")
    expected_block3_tdnnd1.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd1.f32")
    expected_block3_after_tdnnd1.reshape(1, 544, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd1.f32"
    )
    expected_block3_tdnnd2.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd2.f32")
    expected_block3_after_tdnnd2.reshape(1, 576, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd2.f32"
    )
    expected_block3_tdnnd3.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd3.f32")
    expected_block3_after_tdnnd3.reshape(1, 608, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd3.f32"
    )
    expected_block3_tdnnd4.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd4.f32")
    expected_block3_after_tdnnd4.reshape(1, 640, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd4.f32"
    )
    expected_block3_tdnnd5.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd5.f32")
    expected_block3_after_tdnnd5.reshape(1, 672, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd5.f32"
    )
    expected_block3_tdnnd6.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd6.f32")
    expected_block3_after_tdnnd6.reshape(1, 704, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd6.f32"
    )
    expected_block3_tdnnd7.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd7.f32")
    expected_block3_after_tdnnd7.reshape(1, 736, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd7.f32"
    )
    expected_block3_tdnnd8.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd8.f32")
    expected_block3_after_tdnnd8.reshape(1, 768, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd8.f32"
    )
    expected_block3_tdnnd9.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd9.f32")
    expected_block3_after_tdnnd9.reshape(1, 800, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd9.f32"
    )
    expected_block3_tdnnd10.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd10.f32")
    expected_block3_after_tdnnd10.reshape(1, 832, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd10.f32"
    )
    expected_block3_tdnnd11.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd11.f32")
    expected_block3_after_tdnnd11.reshape(1, 864, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd11.f32"
    )
    expected_block3_tdnnd12.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd12.f32")
    expected_block3_after_tdnnd12.reshape(1, 896, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd12.f32"
    )
    expected_block3_tdnnd13.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd13.f32")
    expected_block3_after_tdnnd13.reshape(1, 928, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd13.f32"
    )
    expected_block3_tdnnd14.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd14.f32")
    expected_block3_after_tdnnd14.reshape(1, 960, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd14.f32"
    )
    expected_block3_tdnnd15.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd15.f32")
    expected_block3_after_tdnnd15.reshape(1, 992, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd15.f32"
    )
    expected_block3_tdnnd16.reshape(1, 32, tdnn_frames).tofile(golden_dir / "campplus_xvector_block3_tdnnd16.f32")
    expected_block3_after_tdnnd16.reshape(1, 1024, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_block3_after_tdnnd16.f32"
    )
    expected_transit3.reshape(1, 512, tdnn_frames).tofile(golden_dir / "campplus_xvector_transit3.f32")
    expected_out_nonlinear.reshape(1, 512, tdnn_frames).tofile(
        golden_dir / "campplus_xvector_out_nonlinear.f32"
    )
    expected_stats.reshape(1, 1024).tofile(golden_dir / "campplus_xvector_stats.f32")
    expected_dense.reshape(1, 192).tofile(golden_dir / "campplus_xvector_dense.f32")

    report = _run_json(
        "--clone-campplus-head-golden",
        str(model_bundle),
        prepared["features_manifest"],
        str(golden_dir),
    )

    assert report["stage"] == "tts_clone_campplus_head_golden"
    assert report["ok"] is True
    assert report["expected_shape"] == f"[1,32,80,{frames}]"
    assert report["expected_layer1_shape"] == f"[1,32,40,{frames}]"
    assert report["expected_layer2_shape"] == f"[1,32,20,{frames}]"
    assert report["expected_conv2_shape"] == f"[1,32,10,{frames}]"
    assert report["expected_output_shape"] == f"[1,320,{frames}]"
    assert report["expected_tdnn_shape"] == f"[1,128,{tdnn_frames}]"
    assert report["expected_block1_tdnnd1_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd1_shape"] == f"[1,160,{tdnn_frames}]"
    assert report["expected_block1_tdnnd2_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd2_shape"] == f"[1,192,{tdnn_frames}]"
    assert report["expected_block1_tdnnd3_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd3_shape"] == f"[1,224,{tdnn_frames}]"
    assert report["expected_block1_tdnnd4_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd4_shape"] == f"[1,256,{tdnn_frames}]"
    assert report["expected_block1_tdnnd5_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd5_shape"] == f"[1,288,{tdnn_frames}]"
    assert report["expected_block1_tdnnd6_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd6_shape"] == f"[1,320,{tdnn_frames}]"
    assert report["expected_block1_tdnnd7_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd7_shape"] == f"[1,352,{tdnn_frames}]"
    assert report["expected_block1_tdnnd8_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd8_shape"] == f"[1,384,{tdnn_frames}]"
    assert report["expected_block1_tdnnd9_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd9_shape"] == f"[1,416,{tdnn_frames}]"
    assert report["expected_block1_tdnnd10_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd10_shape"] == f"[1,448,{tdnn_frames}]"
    assert report["expected_block1_tdnnd11_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd11_shape"] == f"[1,480,{tdnn_frames}]"
    assert report["expected_block1_tdnnd12_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block1_after_tdnnd12_shape"] == f"[1,512,{tdnn_frames}]"
    assert report["expected_transit1_shape"] == f"[1,256,{tdnn_frames}]"
    assert report["expected_block2_tdnnd1_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd1_shape"] == f"[1,288,{tdnn_frames}]"
    assert report["expected_block2_tdnnd2_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd2_shape"] == f"[1,320,{tdnn_frames}]"
    assert report["expected_block2_tdnnd3_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd3_shape"] == f"[1,352,{tdnn_frames}]"
    assert report["expected_block2_tdnnd4_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd4_shape"] == f"[1,384,{tdnn_frames}]"
    assert report["expected_block2_tdnnd5_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd5_shape"] == f"[1,416,{tdnn_frames}]"
    assert report["expected_block2_tdnnd6_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd6_shape"] == f"[1,448,{tdnn_frames}]"
    assert report["expected_block2_tdnnd7_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd7_shape"] == f"[1,480,{tdnn_frames}]"
    assert report["expected_block2_tdnnd8_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd8_shape"] == f"[1,512,{tdnn_frames}]"
    assert report["expected_block2_tdnnd9_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd9_shape"] == f"[1,544,{tdnn_frames}]"
    assert report["expected_block2_tdnnd10_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd10_shape"] == f"[1,576,{tdnn_frames}]"
    assert report["expected_block2_tdnnd11_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd11_shape"] == f"[1,608,{tdnn_frames}]"
    assert report["expected_block2_tdnnd12_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd12_shape"] == f"[1,640,{tdnn_frames}]"
    assert report["expected_block2_tdnnd13_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd13_shape"] == f"[1,672,{tdnn_frames}]"
    assert report["expected_block2_tdnnd14_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd14_shape"] == f"[1,704,{tdnn_frames}]"
    assert report["expected_block2_tdnnd15_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd15_shape"] == f"[1,736,{tdnn_frames}]"
    assert report["expected_block2_tdnnd16_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd16_shape"] == f"[1,768,{tdnn_frames}]"
    assert report["expected_block2_tdnnd17_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd17_shape"] == f"[1,800,{tdnn_frames}]"
    assert report["expected_block2_tdnnd18_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd18_shape"] == f"[1,832,{tdnn_frames}]"
    assert report["expected_block2_tdnnd19_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd19_shape"] == f"[1,864,{tdnn_frames}]"
    assert report["expected_block2_tdnnd20_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd20_shape"] == f"[1,896,{tdnn_frames}]"
    assert report["expected_block2_tdnnd21_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd21_shape"] == f"[1,928,{tdnn_frames}]"
    assert report["expected_block2_tdnnd22_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd22_shape"] == f"[1,960,{tdnn_frames}]"
    assert report["expected_block2_tdnnd23_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd23_shape"] == f"[1,992,{tdnn_frames}]"
    assert report["expected_block2_tdnnd24_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block2_after_tdnnd24_shape"] == f"[1,1024,{tdnn_frames}]"
    assert report["expected_transit2_shape"] == f"[1,512,{tdnn_frames}]"
    assert report["expected_block3_tdnnd1_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd1_shape"] == f"[1,544,{tdnn_frames}]"
    assert report["expected_block3_tdnnd2_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd2_shape"] == f"[1,576,{tdnn_frames}]"
    assert report["expected_block3_tdnnd3_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd3_shape"] == f"[1,608,{tdnn_frames}]"
    assert report["expected_block3_tdnnd4_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd4_shape"] == f"[1,640,{tdnn_frames}]"
    assert report["expected_block3_tdnnd5_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd5_shape"] == f"[1,672,{tdnn_frames}]"
    assert report["expected_block3_tdnnd6_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd6_shape"] == f"[1,704,{tdnn_frames}]"
    assert report["expected_block3_tdnnd7_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd7_shape"] == f"[1,736,{tdnn_frames}]"
    assert report["expected_block3_tdnnd8_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd8_shape"] == f"[1,768,{tdnn_frames}]"
    assert report["expected_block3_tdnnd9_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd9_shape"] == f"[1,800,{tdnn_frames}]"
    assert report["expected_block3_tdnnd10_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd10_shape"] == f"[1,832,{tdnn_frames}]"
    assert report["expected_block3_tdnnd11_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd11_shape"] == f"[1,864,{tdnn_frames}]"
    assert report["expected_block3_tdnnd12_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd12_shape"] == f"[1,896,{tdnn_frames}]"
    assert report["expected_block3_tdnnd13_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd13_shape"] == f"[1,928,{tdnn_frames}]"
    assert report["expected_block3_tdnnd14_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd14_shape"] == f"[1,960,{tdnn_frames}]"
    assert report["expected_block3_tdnnd15_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd15_shape"] == f"[1,992,{tdnn_frames}]"
    assert report["expected_block3_tdnnd16_shape"] == f"[1,32,{tdnn_frames}]"
    assert report["expected_block3_after_tdnnd16_shape"] == f"[1,1024,{tdnn_frames}]"
    assert report["expected_transit3_shape"] == f"[1,512,{tdnn_frames}]"
    assert report["expected_out_nonlinear_shape"] == f"[1,512,{tdnn_frames}]"
    assert report["expected_stats_shape"] == "[1,1024]"
    assert report["expected_dense_shape"] == "[1,192]"
    assert report["campplus_head_conv1_bn_relu_max_abs_error"] <= 1.0e-4
    assert report["campplus_head_layer1_max_abs_error"] <= 2.0e-4
    assert report["campplus_head_layer2_max_abs_error"] <= 3.0e-4
    assert report["campplus_head_conv2_bn_relu_max_abs_error"] <= 4.0e-4
    assert report["campplus_head_output_max_abs_error"] <= 4.0e-4
    assert report["campplus_xvector_tdnn_max_abs_error"] <= 8.0e-4
    assert report["campplus_xvector_block1_tdnnd1_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd1_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd2_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd2_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd3_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd3_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd4_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd4_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd5_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd5_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd6_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd6_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd7_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd7_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd8_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd8_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd9_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd9_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd10_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd10_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd11_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd11_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_tdnnd12_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block1_after_tdnnd12_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_transit1_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd1_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd1_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd2_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd2_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd3_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd3_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd4_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd4_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd5_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd5_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd6_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd6_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd7_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd7_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd8_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd8_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd9_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd9_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd10_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd10_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd11_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd11_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd12_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd12_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd13_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd13_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd14_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd14_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd15_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd15_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd16_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd16_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd17_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd17_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd18_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd18_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd19_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd19_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd20_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd20_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd21_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd21_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd22_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd22_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd23_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd23_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_tdnnd24_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block2_after_tdnnd24_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_transit2_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd1_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd1_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd2_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd2_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd3_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd3_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd4_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd4_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd5_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd5_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd6_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd6_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd7_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd7_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd8_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd8_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd9_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd9_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd10_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd10_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd11_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd11_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd12_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd12_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd13_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd13_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd14_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd14_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd15_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd15_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_tdnnd16_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_block3_after_tdnnd16_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_transit3_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_out_nonlinear_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_stats_max_abs_error"] <= 1.0e-3
    assert report["campplus_xvector_dense_max_abs_error"] <= 1.0e-3
    assert report["ready_native_campplus_head_conv1_forward"] is True
    assert report["ready_native_campplus_head_layer1_forward"] is True
    assert report["ready_native_campplus_head_layer2_forward"] is True
    assert report["ready_native_campplus_head_output_forward"] is True
    assert report["ready_native_campplus_head_forward"] is True
    assert report["ready_native_campplus_xvector_tdnn_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd1_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd2_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd3_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd4_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd5_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd6_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd7_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd8_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd9_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd10_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd11_forward"] is True
    assert report["ready_native_campplus_xvector_block1_tdnnd12_forward"] is True
    assert report["ready_native_campplus_xvector_transit1_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd1_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd2_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd3_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd4_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd5_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd6_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd7_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd8_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd9_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd10_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd11_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd12_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd13_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd14_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd15_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd16_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd17_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd18_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd19_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd20_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd21_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd22_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd23_forward"] is True
    assert report["ready_native_campplus_xvector_block2_tdnnd24_forward"] is True
    assert report["ready_native_campplus_xvector_transit2_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd1_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd2_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd3_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd4_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd5_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd6_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd7_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd8_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd9_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd10_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd11_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd12_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd13_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd14_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd15_forward"] is True
    assert report["ready_native_campplus_xvector_block3_tdnnd16_forward"] is True
    assert report["ready_native_campplus_xvector_transit3_forward"] is True
    assert report["ready_native_campplus_xvector_out_nonlinear_forward"] is True
    assert report["ready_native_campplus_xvector_stats_forward"] is True
    assert report["ready_native_campplus_xvector_dense_forward"] is True
    assert report["ready_native_campplus_style_forward"] is True
    assert report["ready_native_voice_clone"] is False
    assert report["next_native_boundary"] == "native clone semantic/acoustic speech encoders for voice tensor creation"
    assert report["clone_campplus_head_golden_issues"] == []


def test_mit2_tts_clone_write_voice_bundle_creates_native_mit2_bundle(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    wav_path = tmp_path / "clone_ref.wav"
    preprocessed_path = tmp_path / "clone_ref.16k.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    preprocess = _run_json("--clone-preprocess", str(wav_path), str(preprocessed_path))
    mel_extract = _run_json("--clone-extract-mel", preprocess["preprocess_manifest"], str(tmp_path / "mel.f32"))
    spk_path = tmp_path / "spk_cond_emb.f32"
    style_path = tmp_path / "s2mel_style.f32"
    prompt_path = tmp_path / "s2mel_prompt.f32"
    mel_path = Path(mel_extract["output_mel_f32"])
    bundle_path = tmp_path / "native_voice_bundle"
    prompt_tokens = int(mel_extract["mel_frames"])
    _write_f32(spk_path, [0.01] * (2 * 1024))
    _write_f32(style_path, [0.02] * 192)
    _write_f32(prompt_path, [0.03] * (prompt_tokens * 512))

    report = _run_json(
        "--clone-write-voice-bundle",
        preprocess["preprocess_manifest"],
        str(spk_path),
        "2",
        str(style_path),
        str(prompt_path),
        str(prompt_tokens),
        str(mel_path),
        str(bundle_path),
    )

    assert report["stage"] == "tts_clone_write_voice_bundle"
    assert report["ok"] is True
    assert report["format"] == "MIT2"
    assert report["output_voice_bundle"] == str(bundle_path)
    assert report["spk_cond_tokens"] == 2
    assert report["prompt_tokens"] == prompt_tokens
    assert report["mel_frames"] == prompt_tokens
    assert report["integrity"]["aligned_tensor_count"] == 4
    assert report["integrity"]["sha256_verified_count"] == 4
    assert report["ready_native_clone_audio_preprocess"] is True
    assert report["ready_native_voice_bundle_creation"] is True
    assert report["ready_native_voice_clone"] is False
    readiness = _run_last_json("--readiness", str(MODEL_BUNDLE), str(bundle_path))
    assert readiness["stage"] == "tts_product_readiness"
    assert readiness["ok"] is True
    assert readiness["voice_prompt_tokens"] == prompt_tokens


def test_mit2_tts_clone_write_voice_bundle_from_features_uses_prepared_mel(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    spk_path = tmp_path / "spk_cond_emb.f32"
    style_path = tmp_path / "s2mel_style.f32"
    prompt_path = tmp_path / "s2mel_prompt.f32"
    bundle_path = tmp_path / "native_voice_bundle_from_features"
    prompt_tokens = int(prepared["mel_frames"])
    _write_f32(spk_path, [0.01] * (2 * 1024))
    _write_f32(style_path, [0.02] * 192)
    _write_f32(prompt_path, [0.03] * (prompt_tokens * 512))

    encoder_readiness = _run_json(
        "--clone-encoder-readiness",
        prepared["features_manifest"],
        str(spk_path),
        "2",
        str(style_path),
        str(prompt_path),
    )
    assert encoder_readiness["stage"] == "tts_clone_encoder_readiness"
    assert encoder_readiness["ok"] is True
    assert encoder_readiness["prompt_tokens"] == prompt_tokens
    assert encoder_readiness["ready_native_clone_feature_prep"] is True
    assert encoder_readiness["ready_native_clone_encoder_outputs"] is True
    assert encoder_readiness["ready_native_voice_bundle_creation"] is True
    assert encoder_readiness["ready_native_voice_clone"] is False
    assert encoder_readiness["clone_encoder_readiness_issues"] == []

    report = _run_json(
        "--clone-write-voice-bundle-from-features",
        prepared["features_manifest"],
        str(spk_path),
        "2",
        str(style_path),
        str(prompt_path),
        str(prompt_tokens),
        str(bundle_path),
    )

    assert report["stage"] == "tts_clone_write_voice_bundle_from_features"
    assert report["ok"] is True
    assert report["feature_manifest"] == prepared["features_manifest"]
    assert report["prompt_tokens"] == prompt_tokens
    assert report["mel_frames"] == prompt_tokens
    assert report["ready_native_voice_bundle_creation"] is True
    assert report["ready_native_voice_clone"] is False
    readiness = _run_last_json("--readiness", str(MODEL_BUNDLE), str(bundle_path))
    assert readiness["stage"] == "tts_product_readiness"
    assert readiness["ok"] is True
    assert readiness["voice_prompt_tokens"] == prompt_tokens


def test_mit2_tts_clone_semantic_quantize_writes_sref_and_codes(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    spk_tokens = 2
    spk_path = tmp_path / "spk_cond_emb.f32"
    sref_path = tmp_path / "sref.f32"
    codes_path = tmp_path / "semantic_codes.u32"
    np.linspace(-0.02, 0.02, spk_tokens * 1024, dtype=np.float32).tofile(spk_path)

    report = _run_json(
        "--clone-semantic-quantize",
        str(MODEL_BUNDLE),
        str(spk_path),
        str(spk_tokens),
        str(sref_path),
        str(codes_path),
    )

    assert report["stage"] == "tts_clone_semantic_quantize"
    assert report["ok"] is True
    assert report["execution_backend"] in {"metal", "cpu_fallback"}
    assert report["spk_tokens"] == spk_tokens
    assert report["spk_cond_values"] == spk_tokens * 1024
    assert report["sref_values"] == spk_tokens * 1024
    assert report["semantic_code_count"] == spk_tokens
    assert report["codebook_size"] == 8192
    assert report["codebook_dim"] == 8
    assert report["has_semantic_codec_quantize_contract"] is True
    assert report["ready_native_semantic_codec_quantize_from_spk_cond"] is True
    assert report["ready_metal_semantic_codec_quantize_from_spk_cond"] is (
        report["execution_backend"] == "metal"
    )
    assert report["ready_native_voice_clone"] is False
    assert report["semantic_codec_quantize_issues"] == []
    sref = np.fromfile(sref_path, dtype=np.float32)
    codes = np.fromfile(codes_path, dtype=np.uint32)
    assert sref.shape == (spk_tokens * 1024,)
    assert np.isfinite(sref).all()
    assert codes.shape == (spk_tokens,)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 8192


def test_mit2_tts_clone_semantic_quantize_output_feeds_prompt_condition(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    prompt_tokens = int(prepared["mel_frames"])
    spk_tokens = 2
    spk_path = tmp_path / "spk_cond_emb.f32"
    sref_path = tmp_path / "sref.f32"
    codes_path = tmp_path / "semantic_codes.u32"
    prompt_path = tmp_path / "s2mel_prompt.f32"
    np.linspace(-0.015, 0.015, spk_tokens * 1024, dtype=np.float32).tofile(spk_path)

    quantized = _run_json(
        "--clone-semantic-quantize",
        str(MODEL_BUNDLE),
        str(spk_path),
        str(spk_tokens),
        str(sref_path),
        str(codes_path),
    )
    assert quantized["ok"] is True

    prompt = _run_json(
        "--clone-s2mel-prompt-from-sref",
        str(MODEL_BUNDLE),
        prepared["features_manifest"],
        str(sref_path),
        str(spk_tokens),
        str(prompt_path),
    )

    assert prompt["stage"] == "tts_clone_s2mel_prompt_from_sref"
    assert prompt["ok"] is True
    assert prompt["prompt_tokens"] == prompt_tokens
    assert prompt["sref_tokens"] == spk_tokens
    assert prompt["sref_values"] == spk_tokens * 1024
    assert prompt["s2mel_prompt_values"] == prompt_tokens * 512
    assert prompt["ready_native_s2mel_prompt_condition_from_sref"] is True
    assert np.fromfile(prompt_path, dtype=np.float32).shape == (prompt_tokens * 512,)


def test_mit2_tts_clone_semantic_prompt_from_spk_cond_writes_sidecars(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "semantic_prompt_sidecars"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    prompt_tokens = int(prepared["mel_frames"])
    spk_tokens = 2
    spk_path = tmp_path / "spk_cond_emb.f32"
    np.linspace(-0.012, 0.012, spk_tokens * 1024, dtype=np.float32).tofile(spk_path)

    report = _run_json(
        "--clone-semantic-prompt-from-spk-cond",
        str(MODEL_BUNDLE),
        prepared["features_manifest"],
        str(spk_path),
        str(spk_tokens),
        str(output_dir),
    )

    assert report["stage"] == "tts_clone_semantic_prompt_from_spk_cond"
    assert report["ok"] is True
    assert report["semantic_execution_backend"] in {"metal", "cpu_fallback"}
    assert report["prompt_execution_backend"] in {"metal", "cpu_fallback"}
    assert report["spk_tokens"] == spk_tokens
    assert report["prompt_tokens"] == prompt_tokens
    assert report["sref_values"] == spk_tokens * 1024
    assert report["semantic_code_count"] == spk_tokens
    assert report["s2mel_prompt_values"] == prompt_tokens * 512
    assert report["ready_native_semantic_codec_quantize_from_spk_cond"] is True
    assert report["ready_native_s2mel_prompt_condition_from_sref"] is True
    assert report["ready_metal_semantic_codec_quantize_from_spk_cond"] is (
        report["semantic_execution_backend"] == "metal"
    )
    assert report["ready_metal_s2mel_prompt_condition_from_sref"] is (
        report["prompt_execution_backend"] == "metal"
    )
    assert report["ready_native_voice_clone"] is False

    sref = np.fromfile(report["output_sref_f32"], dtype=np.float32)
    codes = np.fromfile(report["output_codes_u32"], dtype=np.uint32)
    prompt = np.fromfile(report["output_s2mel_prompt_f32"], dtype=np.float32)
    assert sref.shape == (spk_tokens * 1024,)
    assert codes.shape == (spk_tokens,)
    assert prompt.shape == (prompt_tokens * 512,)
    assert np.isfinite(sref).all()
    assert np.isfinite(prompt).all()

    manifest = json.loads(Path(report["output_manifest"]).read_text())
    assert manifest["format"] == "mit2-clone-semantic-prompt-sidecars"
    assert manifest["spk_tokens"] == spk_tokens
    assert manifest["prompt_tokens"] == prompt_tokens
    assert manifest["output_sref_f32"] == report["output_sref_f32"]
    assert manifest["output_s2mel_prompt_f32"] == report["output_s2mel_prompt_f32"]
    assert manifest["ready_native_semantic_codec_quantize_from_spk_cond"] is True
    assert manifest["ready_native_s2mel_prompt_condition_from_sref"] is True


def test_mit2_tts_clone_s2mel_prompt_from_sref_writes_prompt_sidecar(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    prompt_tokens = int(prepared["mel_frames"])
    sref_tokens = 2
    sref_path = tmp_path / "sref.f32"
    prompt_path = tmp_path / "s2mel_prompt.f32"
    np.linspace(-0.01, 0.01, sref_tokens * 1024, dtype=np.float32).tofile(sref_path)

    report = _run_json(
        "--clone-s2mel-prompt-from-sref",
        str(MODEL_BUNDLE),
        prepared["features_manifest"],
        str(sref_path),
        str(sref_tokens),
        str(prompt_path),
    )

    assert report["stage"] == "tts_clone_s2mel_prompt_from_sref"
    assert report["ok"] is True
    assert report["prompt_tokens"] == prompt_tokens
    assert report["sref_tokens"] == sref_tokens
    assert report["sref_values"] == sref_tokens * 1024
    assert report["s2mel_prompt_values"] == prompt_tokens * 512
    assert report["s2mel_prompt_shape"] == f"[1,{prompt_tokens},512]"
    assert report["has_s2mel_prompt_condition_contract"] is True
    assert report["ready_native_s2mel_prompt_condition_from_sref"] is True
    assert report["ready_native_voice_clone"] is False
    assert report["clone_s2mel_prompt_issues"] == []
    assert prompt_path.exists()
    prompt = np.fromfile(prompt_path, dtype=np.float32)
    assert prompt.shape == (prompt_tokens * 512,)
    assert np.isfinite(prompt).all()


def test_mit2_tts_clone_s2mel_prompt_from_sref_rejects_bad_sref_size(tmp_path: Path):
    if not (MODEL_BUNDLE / "manifest.json").exists():
        pytest.skip("test model bundle is not available")
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    sref_path = tmp_path / "bad_sref.f32"
    prompt_path = tmp_path / "s2mel_prompt.f32"
    np.zeros((1023,), dtype=np.float32).tofile(sref_path)

    report = _run_json(
        "--clone-s2mel-prompt-from-sref",
        str(MODEL_BUNDLE),
        prepared["features_manifest"],
        str(sref_path),
        "1",
        str(prompt_path),
        check=False,
    )

    assert report["stage"] == "tts_clone_s2mel_prompt_from_sref"
    assert report["ok"] is False
    assert report["ready_native_s2mel_prompt_condition_from_sref"] is False
    assert any("sref_size_mismatch" in issue for issue in report["clone_s2mel_prompt_issues"])
    assert not prompt_path.exists()


def test_mit2_tts_clone_encoder_readiness_rejects_bad_prompt_tensor_size(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    features_dir = tmp_path / "features"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    prepared = _run_json("--clone-prepare-features", str(wav_path), str(features_dir))
    spk_path = tmp_path / "spk_cond_emb.f32"
    style_path = tmp_path / "s2mel_style.f32"
    prompt_path = tmp_path / "s2mel_prompt.f32"
    _write_f32(spk_path, [0.01] * (2 * 1024))
    _write_f32(style_path, [0.02] * 192)
    _write_f32(prompt_path, [0.03] * 511)

    report = _run_json(
        "--clone-encoder-readiness",
        prepared["features_manifest"],
        str(spk_path),
        "2",
        str(style_path),
        str(prompt_path),
        check=False,
    )

    assert report["stage"] == "tts_clone_encoder_readiness"
    assert report["ok"] is False
    assert report["ready_native_clone_encoder_outputs"] is False
    assert report["ready_native_voice_bundle_creation"] is False
    assert report["ready_native_voice_clone"] is False
    assert any("s2mel_prompt_size_mismatch" in issue for issue in report["clone_encoder_readiness_issues"])


def test_mit2_tts_clone_write_voice_bundle_rejects_bad_tensor_size(tmp_path: Path):
    wav_path = tmp_path / "clone_ref.wav"
    preprocessed_path = tmp_path / "clone_ref.16k.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x10" * 2205))
    preprocess = _run_json("--clone-preprocess", str(wav_path), str(preprocessed_path))
    spk_path = tmp_path / "spk_cond_emb.f32"
    style_path = tmp_path / "s2mel_style.f32"
    prompt_path = tmp_path / "s2mel_prompt.f32"
    mel_path = tmp_path / "mel.f32"
    bundle_path = tmp_path / "native_voice_bundle"
    _write_f32(spk_path, [0.01] * 1023)
    _write_f32(style_path, [0.02] * 192)
    _write_f32(prompt_path, [0.03] * 512)
    _write_f32(mel_path, [0.04] * 80)

    report = _run_json(
        "--clone-write-voice-bundle",
        preprocess["preprocess_manifest"],
        str(spk_path),
        "1",
        str(style_path),
        str(prompt_path),
        "1",
        str(mel_path),
        str(bundle_path),
        check=False,
    )

    assert report["stage"] == "tts_clone_write_voice_bundle"
    assert report["ok"] is False
    assert report["ready_native_voice_bundle_creation"] is False
    assert report["ready_native_voice_clone"] is False
    assert "spk_cond_emb expected 1024 f32 values" in report["error"]
    assert not (bundle_path / "manifest.json").exists()


def test_mit2_tts_clone_preprocess_rejects_silent_audio_without_output(tmp_path: Path):
    wav_path = tmp_path / "silent_ref.wav"
    out_path = tmp_path / "silent_ref.16k.f32"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes((b"\x00\x00" * 2205))

    report = _run_json("--clone-preprocess", str(wav_path), str(out_path), check=False)

    assert report["stage"] == "tts_clone_audio_preprocess"
    assert report["ok"] is False
    assert report["ready_native_clone_audio_preprocess"] is False
    assert "silent_or_near_silent_audio" in report["quality_issues"]
    assert not out_path.exists()
    assert not Path(report["preprocess_manifest"]).exists()


def test_mit2_tts_text_readiness_reports_supported_native_surface():
    report = _run_json("--text-readiness", str(MODEL_BUNDLE), "1=1000℃")

    assert report["stage"] == "tts_cjk_text_readiness"
    assert report["ok"] is True
    assert report["product_surface_version"] == 1
    assert report["ready_native_cjk_text"] is True
    assert report["ready_cached_voice_tts_cjk_text"] is True
    assert report["native_text_surface"] == "focused_cjk_limited_ascii"
    assert report["token_ids"] == [
        10201,
        7,
        10201,
        4275,
        10201,
        85,
        10201,
        7,
        10201,
        3878,
        10201,
        6500,
        10201,
        2313,
        10201,
        2942,
        10201,
        1726,
    ]


def test_mit2_tts_text_readiness_reports_leading_zero_currency_surface():
    report = _run_json("--text-readiness", str(MODEL_BUNDLE), "￥01000")

    assert report["stage"] == "tts_cjk_text_readiness"
    assert report["ok"] is True
    assert report["ready_native_cjk_text"] is True
    assert report["ready_cached_voice_tts_cjk_text"] is True
    assert report["tokens"] == ["▁", "零", "▁", "元", "▁", "一", "▁", "千"]
    assert report["token_ids"] == [10201, 6500, 10201, 361, 10201, 7, 10201, 571]


def test_mit2_tts_text_readiness_reports_five_digit_no_surface():
    report = _run_json("--text-readiness", str(MODEL_BUNDLE), "No.10100")

    assert report["stage"] == "tts_cjk_text_readiness"
    assert report["ok"] is True
    assert report["ready_native_cjk_text"] is True
    assert report["ready_cached_voice_tts_cjk_text"] is True
    assert report["tokens"] == ["▁NUMBER", "▁TEN", "▁THOUSAND", "▁ONE", "▁HUNDRED"]
    assert report["token_ids"] == [10423, 10472, 10456, 10238, 10386]


def test_mit2_tts_text_readiness_reports_five_digit_ascii_unit_surface():
    report = _run_json("--text-readiness", str(MODEL_BUNDLE), "10001kg")

    assert report["stage"] == "tts_cjk_text_readiness"
    assert report["ok"] is True
    assert report["ready_native_cjk_text"] is True
    assert report["ready_cached_voice_tts_cjk_text"] is True
    assert report["tokens"] == [
        "▁TEN",
        "▁THOUSAND",
        "▁AND",
        "▁ONE",
        "▁",
        "K",
        "I",
        "LOG",
        "RA",
        "M",
        "S",
    ]
    assert report["token_ids"] == [
        10472,
        10456,
        10205,
        10238,
        10201,
        10403,
        10299,
        11520,
        11134,
        10248,
        10208,
    ]


def test_mit2_tts_text_readiness_reports_fullwidth_ascii_unit_suffix_surface():
    report = _run_json("--text-readiness", str(MODEL_BUNDLE), "2.5ＫＧ")

    assert report["stage"] == "tts_cjk_text_readiness"
    assert report["ok"] is True
    assert report["ready_native_cjk_text"] is True
    assert report["ready_cached_voice_tts_cjk_text"] is True
    assert report["tokens"] == ["▁", "二", "▁", "点", "▁", "五", "▁", "K", "G"]
    assert report["token_ids"] == [10201, 83, 10201, 3393, 10201, 90, 10201, 10403, 10381]


def test_mit2_tts_text_readiness_keeps_general_text_boundary_explicit():
    _require_launcher_artifacts()
    proc = subprocess.run(
        [str(MIT2_TTS), "--text-readiness", str(MODEL_BUNDLE), "1=2026℃"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    report = json.loads(proc.stdout)

    assert proc.returncode != 0
    assert report["stage"] == "tts_cjk_text_readiness"
    assert report["ok"] is False
    assert report["product_surface_version"] == 1
    assert report["ready_native_cjk_text"] is False
    assert report["ready_cached_voice_tts_cjk_text"] is False
    assert report["native_text_surface"] == "focused_cjk_limited_ascii"
    assert report["python_boundary"] == "full TextNormalizer/SentencePiece for general text"
