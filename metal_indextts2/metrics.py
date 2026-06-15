from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest(),
    }
    if arr.size and arr.dtype.kind in {"f", "i", "u"}:
        finite = np.isfinite(arr) if arr.dtype.kind == "f" else np.ones(arr.shape, dtype=bool)
        out.update(
            {
                "min": float(arr[finite].min()) if finite.any() else None,
                "max": float(arr[finite].max()) if finite.any() else None,
                "mean": float(arr[finite].mean()) if finite.any() else None,
            }
        )
    return out


def compare_arrays(ref: Any, got: Any) -> dict[str, Any]:
    ref_arr = np.asarray(ref, dtype=np.float64)
    got_arr = np.asarray(got, dtype=np.float64)
    if ref_arr.shape != got_arr.shape:
        return {"shape_match": False, "ref_shape": list(ref_arr.shape), "got_shape": list(got_arr.shape)}
    delta = got_arr - ref_arr
    ref_norm = np.linalg.norm(ref_arr.reshape(-1))
    got_norm = np.linalg.norm(got_arr.reshape(-1))
    denom = max(ref_norm * got_norm, 1e-30)
    return {
        "shape_match": True,
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "relative_l2_error": float(np.linalg.norm(delta.reshape(-1)) / max(ref_norm, 1e-30)),
        "cosine_similarity": float(np.dot(ref_arr.reshape(-1), got_arr.reshape(-1)) / denom),
    }
