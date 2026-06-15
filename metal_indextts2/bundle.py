from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np

MAGIC = "MIT2"
VERSION = 1
ALIGNMENT = 4096

DTYPE_TO_NUMPY = {
    "f32": np.float32,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "u32": np.uint32,
    "u8": np.uint8,
}

NUMPY_TO_DTYPE = {
    np.dtype(np.float32): "f32",
    np.dtype(np.float16): "f16",
    np.dtype(np.int64): "i64",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.uint8): "u8",
}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int
    sha256: str
    layout: str = "row_major"
    component: str = "unknown"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
            "layout": self.layout,
            "component": self.component,
        }


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_array(value: Any, dtype: str | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(DTYPE_TO_NUMPY[dtype], copy=False)
    elif arr.dtype not in NUMPY_TO_DTYPE:
        if arr.dtype.kind == "f":
            arr = arr.astype(np.float32)
        elif arr.dtype.kind in {"i", "u"}:
            arr = arr.astype(np.int64 if arr.dtype.itemsize > 4 else np.int32)
        else:
            raise TypeError(f"unsupported tensor dtype: {arr.dtype}")
    return np.ascontiguousarray(arr)


def dtype_name(arr: np.ndarray) -> str:
    try:
        return NUMPY_TO_DTYPE[np.dtype(arr.dtype)]
    except KeyError as exc:
        raise TypeError(f"unsupported tensor dtype: {arr.dtype}") from exc


def _write_padding(fp: BinaryIO, target_offset: int) -> None:
    current = fp.tell()
    if target_offset < current:
        raise ValueError("target offset moved backwards")
    if target_offset > current:
        fp.write(b"\0" * (target_offset - current))


def write_bundle(
    output_dir: str | os.PathLike[str],
    tensors: Iterable[tuple[str, Any, str]],
    *,
    metadata: dict[str, Any] | None = None,
    force_dtype: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[TensorRecord] = []
    weights_path = out / "weights.bin"

    with weights_path.open("wb") as fp:
        fp.write(struct.pack("<4sII", MAGIC.encode("ascii"), VERSION, ALIGNMENT))
        for name, value, component in tensors:
            arr = normalize_array(value, force_dtype)
            offset = align_up(fp.tell())
            _write_padding(fp, offset)
            payload = memoryview(arr).cast("B")
            fp.write(payload)
            records.append(
                TensorRecord(
                    name=name,
                    shape=tuple(int(x) for x in arr.shape),
                    dtype=dtype_name(arr),
                    offset=offset,
                    nbytes=len(payload),
                    sha256=sha256_bytes(payload),
                    component=component,
                )
            )

    manifest = {
        "format": MAGIC,
        "version": VERSION,
        "endianness": "little",
        "alignment": ALIGNMENT,
        "weights_file": "weights.bin",
        "metadata": metadata or {},
        "tensors": [r.to_json() for r in records],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def read_tensor(bundle_dir: str | os.PathLike[str], name: str) -> np.ndarray:
    bundle = Path(bundle_dir)
    manifest = load_manifest(bundle)
    by_name = {t["name"]: t for t in manifest["tensors"]}
    rec = by_name[name]
    dtype = DTYPE_TO_NUMPY[rec["dtype"]]
    with (bundle / manifest["weights_file"]).open("rb") as fp:
        fp.seek(int(rec["offset"]))
        data = fp.read(int(rec["nbytes"]))
    if sha256_bytes(data) != rec["sha256"]:
        raise ValueError(f"checksum mismatch for {name}")
    return np.frombuffer(data, dtype=dtype).reshape(tuple(rec["shape"]))
