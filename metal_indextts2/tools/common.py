from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Iterator


def add_index_tts_repo(path: str | os.PathLike[str]) -> Path:
    repo = Path(path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"IndexTTS repository not found: {repo}")
    sys.path.insert(0, str(repo))
    return repo


def load_indextts2(index_tts_repo: str, checkpoint_dir: str, *, cfg_path: str | None = None) -> Any:
    repo = add_index_tts_repo(index_tts_repo)
    module = importlib.import_module("indextts.infer_v2")
    cls = getattr(module, "IndexTTS2")
    cfg = cfg_path or str(Path(checkpoint_dir) / "config.yaml")
    cwd = os.getcwd()
    try:
        os.chdir(str(repo))
        return cls(cfg_path=cfg, model_dir=str(Path(checkpoint_dir).resolve()))
    finally:
        os.chdir(cwd)


@contextmanager
def captured_stdout() -> Iterator[StringIO]:
    buf = StringIO()
    with redirect_stdout(buf):
        yield buf


def write_json(path: str | os.PathLike[str], value: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def add_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-tts-repo", default="index-tts")
    parser.add_argument("--checkpoint-dir", default="index-tts/checkpoints")
    parser.add_argument("--cfg-path", default=None)
