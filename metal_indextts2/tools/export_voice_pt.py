from __future__ import annotations

import argparse
import json
from pathlib import Path

from metal_indextts2.tools.convert_voice import write_mit2_voice_profile_pt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a native MIT2 voice bundle as a no-PyTorch .pt voice profile artifact."
    )
    parser.add_argument("--voice-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--ref-audio-path", default=None)
    args = parser.parse_args()

    manifest = write_mit2_voice_profile_pt(
        Path(args.voice_bundle),
        Path(args.output),
        voice_name=args.voice_name,
        ref_audio_path=args.ref_audio_path,
    )
    summary = {
        "format": manifest["format"],
        "output": str(Path(args.output).resolve()),
        "source_bundle": str(Path(args.voice_bundle).resolve()),
        "voice_name": manifest["metadata"]["voice_name"],
        "tensor_count": len(manifest["tensors"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
