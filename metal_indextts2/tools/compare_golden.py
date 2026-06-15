from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import compare_arrays, file_sha256


def compare_npz(ref_path: Path, got_path: Path) -> dict[str, Any]:
    ref = np.load(ref_path)
    got = np.load(got_path)
    report: dict[str, Any] = {
        "missing": sorted(set(ref.files) - set(got.files)),
        "unexpected": sorted(set(got.files) - set(ref.files)),
        "tensors": {},
    }
    for name in sorted(set(ref.files) & set(got.files)):
        report["tensors"][name] = compare_arrays(ref[name], got[name])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native or converted artifacts against a golden fixture.")
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--candidate-voice-npz", default=None)
    parser.add_argument("--candidate-wav", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    golden_dir = Path(args.golden_dir)
    manifest = json.loads((golden_dir / "golden_manifest.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {"golden_dir": str(golden_dir), "checks": {}}
    if args.candidate_voice_npz:
        report["checks"]["voice_profile"] = compare_npz(golden_dir / "voice_profile.npz", Path(args.candidate_voice_npz))
    if args.candidate_wav:
        got_hash = file_sha256(args.candidate_wav)
        report["checks"]["wav_sha256"] = {
            "match": got_hash == manifest.get("audio", {}).get("sha256"),
            "golden": manifest.get("audio", {}).get("sha256"),
            "candidate": got_hash,
        }
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
