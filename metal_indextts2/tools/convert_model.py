from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from metal_indextts2.bundle import write_bundle


def _load_torch() -> Any:
    import torch

    return torch


def _iter_safetensors(path: Path, component: str) -> Iterable[tuple[str, Any, str]]:
    from safetensors.torch import load_file

    obj = load_file(str(path), device="cpu")
    yield from _iter_tensors(obj, component, component)


def _iter_tensors(obj: Any, prefix: str, component: str) -> Iterable[tuple[str, Any, str]]:
    if hasattr(obj, "detach") or isinstance(obj, np.ndarray):
        yield prefix, obj, component
        return
    if isinstance(obj, dict):
        for key in sorted(obj):
            value = obj[key]
            key_text = str(key).replace("/", "_")
            yield from _iter_tensors(value, f"{prefix}.{key_text}" if prefix else key_text, component)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from _iter_tensors(value, f"{prefix}.{idx}" if prefix else str(idx), component)


def _iter_state_dict(path: Path) -> Iterable[tuple[str, Any, str]]:
    torch = _load_torch()
    obj = torch.load(path, map_location="cpu")
    component = path.stem
    if isinstance(obj, dict) and "generator" in obj and isinstance(obj["generator"], dict):
        obj = obj["generator"]
    if hasattr(obj, "detach"):
        yield f"{component}.value", obj, component
        return
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a state dict")
    yield from _iter_tensors(obj, component, component)


def _iter_state_dict_with_prefix(path: Path, prefix: str, component: str) -> Iterable[tuple[str, Any, str]]:
    torch = _load_torch()
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "generator" in obj and isinstance(obj["generator"], dict):
        obj = obj["generator"]
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a state dict")
    yield from _iter_tensors(obj, prefix, component)


def _iter_w2v_bert_dir(path: Path) -> Iterable[tuple[str, Any, str]]:
    from transformers import Wav2Vec2BertModel

    model = Wav2Vec2BertModel.from_pretrained(str(path))
    yield from _iter_tensors(model.state_dict(), "w2v_bert", "w2v_bert")


def _iter_w2v_stats(path: Path) -> Iterable[tuple[str, Any, str]]:
    torch = _load_torch()
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "mean" not in obj or "var" not in obj:
        raise TypeError(f"{path} did not contain wav2vec2bert mean/var stats")
    yield "w2v_bert.stats.mean", obj["mean"], "w2v_bert"
    yield "w2v_bert.stats.std", torch.sqrt(obj["var"]), "w2v_bert"


def convert(
    checkpoint_dir: Path,
    output: Path,
    *,
    force_dtype: str | None = None,
    semantic_codec: Path | None = None,
    bigvgan_checkpoint: Path | None = None,
    bigvgan_config: Path | None = None,
    campplus_checkpoint: Path | None = None,
    w2v_bert_dir: Path | None = None,
    w2v_stats: Path | None = None,
) -> dict[str, Any]:
    tensors = []
    loaded = []
    for name in ("gpt.pth", "s2mel.pth", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt"):
        path = checkpoint_dir / name
        if path.exists():
            loaded.append(name)
            tensors.extend(_iter_state_dict(path))
    if semantic_codec is not None:
        loaded.append(str(semantic_codec))
        tensors.extend(_iter_safetensors(semantic_codec, "semantic_codec"))
    if bigvgan_checkpoint is not None:
        loaded.append(str(bigvgan_checkpoint))
        tensors.extend(_iter_state_dict_with_prefix(bigvgan_checkpoint, "bigvgan", "bigvgan"))
    if campplus_checkpoint is not None:
        loaded.append(str(campplus_checkpoint))
        tensors.extend(_iter_state_dict_with_prefix(campplus_checkpoint, "campplus", "campplus"))
    if w2v_bert_dir is not None:
        loaded.append(str(w2v_bert_dir))
        tensors.extend(_iter_w2v_bert_dir(w2v_bert_dir))
    if w2v_stats is not None:
        loaded.append(str(w2v_stats))
        tensors.extend(_iter_w2v_stats(w2v_stats))

    metadata = {
        "source": str(checkpoint_dir),
        "source_files": loaded,
        "target": "index-tts2",
        "stage": "converted_torch_weights",
    }
    manifest = write_bundle(output, tensors, metadata=metadata, force_dtype=force_dtype)

    for asset in ("config.yaml", "bpe.model", "pinyin.vocab", "configuration.json"):
        src = checkpoint_dir / asset
        if src.exists():
            dst = output / "tokenizer" / asset if asset in {"bpe.model", "pinyin.vocab"} else output / asset
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    if bigvgan_config is not None:
        (output / "bigvgan_config.json").write_bytes(bigvgan_config.read_bytes())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IndexTTS2 PyTorch checkpoints into a native MIT2 bundle.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-dtype", choices=("f32", "f16"), default=None)
    parser.add_argument(
        "--semantic-codec",
        default=None,
        help="Optional local MaskGCT semantic_codec/model.safetensors path to include for native vq2emb.",
    )
    parser.add_argument(
        "--bigvgan-checkpoint",
        default=None,
        help="Optional local BigVGAN generator checkpoint to include for native vocoder tests.",
    )
    parser.add_argument(
        "--bigvgan-config",
        default=None,
        help="Optional local BigVGAN config.json to copy into the native bundle.",
    )
    parser.add_argument(
        "--campplus-checkpoint",
        default=None,
        help="Optional local funasr/campplus checkpoint to include for native voice-clone style encoding.",
    )
    parser.add_argument(
        "--w2v-bert-dir",
        default=None,
        help="Optional local facebook/w2v-bert-2.0 directory to include for native clone semantic features.",
    )
    parser.add_argument(
        "--w2v-stats",
        default=None,
        help="Optional wav2vec2bert_stats.pt mean/var file to include as w2v_bert.stats.mean/std.",
    )
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    manifest = convert(
        Path(args.checkpoint_dir),
        Path(args.output),
        force_dtype=args.force_dtype,
        semantic_codec=Path(args.semantic_codec) if args.semantic_codec else None,
        bigvgan_checkpoint=Path(args.bigvgan_checkpoint) if args.bigvgan_checkpoint else None,
        bigvgan_config=Path(args.bigvgan_config) if args.bigvgan_config else None,
        campplus_checkpoint=Path(args.campplus_checkpoint) if args.campplus_checkpoint else None,
        w2v_bert_dir=Path(args.w2v_bert_dir) if args.w2v_bert_dir else None,
        w2v_stats=Path(args.w2v_stats) if args.w2v_stats else None,
    )
    summary = {
        "output": str(Path(args.output).resolve()),
        "tensor_count": len(manifest["tensors"]),
        "total_tensor_bytes": sum(int(t["nbytes"]) for t in manifest["tensors"]),
    }
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
