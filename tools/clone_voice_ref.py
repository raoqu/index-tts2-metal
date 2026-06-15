#!/usr/bin/env python3
"""Clone a voice from reference audio using actual speech encoders (no GPT needed)."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

INDEX_TTS_REPO = "index-tts"
CHECKPOINT_DIR = "index-tts/checkpoints"

sys.path.insert(0, INDEX_TTS_REPO)
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    from omegaconf import OmegaConf
    cfg_path = os.path.join(CHECKPOINT_DIR, "config.yaml")
    return OmegaConf.load(cfg_path)


def load_mel_fn(cfg):
    from indextts.s2mel.modules.audio import mel_spectrogram
    spect = cfg.s2mel["preprocess_params"]["spect_params"]
    n_fft = spect["n_fft"]
    hop = spect["hop_length"]
    win = spect["win_length"]
    def mel_fn(audio):
        return mel_spectrogram(audio, n_fft=n_fft, num_mels=80, sampling_rate=22050,
                               hop_size=hop, win_size=win, fmin=0, fmax=8000)
    return mel_fn


def load_audio(audio_path: str, max_sec: float = 15.0):
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T)  # [channels, samples]
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    # trim to max_sec
    max_samples = int(max_sec * sr)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
    return audio, sr


def extract_voice(audio_path: str, device: str = "cpu") -> dict:
    cfg = load_config()

    print("Loading W2V-BERT feature extractor...")
    from transformers import SeamlessM4TFeatureExtractor
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    print("Loading semantic model + codec...")
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
    import safetensors.torch
    from huggingface_hub import hf_hub_download

    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        os.path.join(CHECKPOINT_DIR, cfg.w2v_stat))
    semantic_model = semantic_model.to(device).eval()
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)

    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device).eval()
    print("  semantic codec loaded")

    print("Loading s2mel (length regulator only)...")
    from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
    s2mel_path = os.path.join(CHECKPOINT_DIR, cfg.s2mel_checkpoint)
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path,
                                      load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel = s2mel.to(device).eval()
    print("  s2mel loaded")

    print("Loading CAMPPlus style encoder...")
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
    campplus_model = campplus_model.to(device).eval()
    print("  campplus loaded")

    # Load and process audio
    print(f"Processing audio: {audio_path}")
    audio, sr = load_audio(audio_path, max_sec=15.0)
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

    # W2V-BERT: spk_cond_emb
    with torch.no_grad():
        inputs = extractor(audio_16k.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        vq_emb = semantic_model(input_features=input_features, attention_mask=attention_mask,
                                output_hidden_states=True)
        feat = vq_emb.hidden_states[17]  # layer 17, same as IndexTTS2.get_emb
        feat = (feat - semantic_mean) / semantic_std
        spk_cond_emb = feat  # [1, T, 1024]

        # MaskGCT quantize -> S_ref
        _, S_ref = semantic_codec.quantize(spk_cond_emb)

        # Mel spectrogram
        mel_fn = load_mel_fn(cfg)
        ref_mel = mel_fn(audio_22k.to(device).float())  # [1, 80, T]
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)

        # CAMPPlus: style
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(device), num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = campplus_model(feat.unsqueeze(0))

        # Length regulator: prompt condition
        prompt_condition = s2mel.models['length_regulator'](
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None)[0]

    return {
        "spk_cond_emb": spk_cond_emb.detach().cpu().numpy(),
        "s2mel_style":  style.detach().cpu().numpy(),
        "s2mel_prompt": prompt_condition.detach().cpu().numpy(),
        "mel":          ref_mel.detach().cpu().numpy(),
    }


def write_voice_pt(tensors: dict, output_path: Path, voice_name: str, ref_audio_path: str):
    import zipfile, hashlib
    def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    KEYS = ("spk_cond_emb", "s2mel_style", "s2mel_prompt", "mel")
    tensor_records = []
    tensor_payloads = {}
    for key in KEYS:
        arr = np.ascontiguousarray(tensors[key].astype(np.float32))
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        payload = buf.getvalue()
        tf = f"tensors/{key}.npy"
        tensor_payloads[tf] = payload
        tensor_records.append({
            "name": key, "file": tf,
            "shape": list(arr.shape), "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes), "sha256": sha256_bytes(payload),
        })
    manifest = {
        "format": "mit2-voice-profile-pt", "version": 1,
        "metadata": {"voice_name": voice_name, "ref_audio_path": ref_audio_path,
                     "stage": "reference_clone_voice_profile"},
        "tensors": tensor_records,
    }
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        for name, payload in tensor_payloads.items():
            zf.writestr(name, payload)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Clone voice using real speech encoders")
    parser.add_argument("--audio", required=True, help="Reference audio WAV")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.suffix:
        output = output.with_suffix(".pt")
    voice_name = args.voice_name or output.stem

    tensors = extract_voice(args.audio, device=args.device)
    manifest = write_voice_pt(tensors, output, voice_name, args.audio)

    shapes = {r["name"]: r["shape"] for r in manifest["tensors"]}
    print(json.dumps({"output": str(output.resolve()), "voice_name": voice_name,
                      "tensors": shapes}, indent=2))


if __name__ == "__main__":
    main()
