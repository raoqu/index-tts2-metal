from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from metal_indextts2.metrics import tensor_summary
from metal_indextts2.tools.common import add_reference_args, captured_stdout, load_indextts2, write_json


def _as_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _write_raw(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array)
    arr.tofile(path)
    summary = tensor_summary(arr)
    summary["path"] = path.name
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = load_indextts2(args.index_tts_repo, args.checkpoint_dir, cfg_path=args.cfg_path)
    profile = tts.load_voice(args.voice_name) if args.voice_name else tts._extract_voice_features(args.prompt_audio)
    profile.setdefault("voice_name", args.voice_name or Path(args.prompt_audio).stem)

    tokens = tts.tokenizer.tokenize(args.text)
    segments = tts.tokenizer.split_segments(tokens, args.max_text_tokens_per_segment)
    if not segments:
        raise ValueError("text produced no tokenizer segments")
    if args.segment_index >= len(segments):
        raise ValueError(f"--segment-index {args.segment_index} out of range for {len(segments)} segments")
    segment = segments[args.segment_index]
    text_ids = tts.tokenizer.convert_tokens_to_ids(segment)
    text_tensor = torch.tensor(text_ids, dtype=torch.int32, device=tts.device).unsqueeze(0)

    spk_cond_emb = profile["spk_cond_emb"].to(tts.device)
    emo_cond_emb = spk_cond_emb
    with torch.no_grad():
        with torch.amp.autocast(text_tensor.device.type, enabled=tts.dtype is not None, dtype=tts.dtype):
            emovec = tts.gpt.merge_emovec(
                spk_cond_emb,
                emo_cond_emb,
                torch.tensor([spk_cond_emb.shape[-1]], device=text_tensor.device),
                torch.tensor([emo_cond_emb.shape[-1]], device=text_tensor.device),
                alpha=1.0,
            )
            speech_conditioning_latent = tts.gpt.get_conditioning(
                spk_cond_emb.transpose(1, 2),
                torch.tensor([spk_cond_emb.shape[-1]], device=text_tensor.device),
            )
            duration_index = torch.zeros(text_tensor.size(0), device=text_tensor.device, dtype=torch.long)
            duration_emb = tts.gpt.speed_emb(duration_index)
            duration_emb_half = tts.gpt.speed_emb(torch.ones_like(duration_index))
            conds_latent = torch.cat(
                (
                    speech_conditioning_latent + emovec.unsqueeze(1),
                    duration_emb_half.unsqueeze(1),
                    duration_emb.unsqueeze(1),
                ),
                dim=1,
            )
            fake_inputs, inputs_embeds, attention_mask = tts.gpt.prepare_gpt_inputs(conds_latent, text_tensor)
            with captured_stdout() as buf:
                codes, returned_conditioning_latent = tts.gpt.inference_speech(
                    spk_cond_emb,
                    text_tensor,
                    emo_cond_emb,
                    cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tensor.device),
                    emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tensor.device),
                    emo_vec=emovec,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.0,
                    length_penalty=0.0,
                    num_return_sequences=1,
                    max_generate_length=args.max_mel_tokens,
                )
            if tts.gpt.stop_mel_token in codes[0]:
                code_len = int((codes[0] == tts.gpt.stop_mel_token).nonzero(as_tuple=False)[0].item())
            else:
                code_len = int(codes.shape[-1])
            codes = codes[:, :code_len]
            code_lens = torch.tensor([code_len], dtype=torch.long, device=text_tensor.device)
            use_speed = torch.zeros(spk_cond_emb.size(0), device=spk_cond_emb.device, dtype=torch.long)
            gpt_latent = tts.gpt(
                returned_conditioning_latent,
                text_tensor,
                torch.tensor([text_tensor.shape[-1]], device=text_tensor.device),
                codes,
                torch.tensor([codes.shape[-1]], device=text_tensor.device),
                emo_cond_emb,
                cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tensor.device),
                emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tensor.device),
                emo_vec=emovec,
                use_speed=use_speed,
            )
        with torch.amp.autocast(text_tensor.device.type, enabled=False):
            gpt_layer = tts.s2mel.models["gpt_layer"](gpt_latent.float())
            vq2emb = tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1, 2).float()
            s_infer = vq2emb + gpt_layer
            target_lengths = (code_lens * 1.72).long()
            if args.length_regulator_device == "cpu":
                length_regulator_model = copy.deepcopy(tts.s2mel.models["length_regulator"]).cpu().eval()
                length_regulator, _, _, _, _ = length_regulator_model(
                    s_infer.detach().cpu().float(),
                    ylens=target_lengths.detach().cpu(),
                    n_quantizers=3,
                    f0=None,
                )
            else:
                length_regulator, _, _, _, _ = tts.s2mel.models["length_regulator"](
                    s_infer,
                    ylens=target_lengths,
                    n_quantizers=3,
                    f0=None,
                )

    codes_np = _as_numpy(codes.squeeze(0), np.uint32)
    conds_np = _as_numpy(conds_latent.squeeze(0), np.float32)
    returned_cond_np = _as_numpy(returned_conditioning_latent.squeeze(0), np.float32)
    gpt_latent_np = _as_numpy(gpt_latent.squeeze(0), np.float32)
    gpt_layer_np = _as_numpy(gpt_layer.squeeze(0), np.float32)
    vq2emb_np = _as_numpy(vq2emb.squeeze(0), np.float32)
    s_infer_np = _as_numpy(s_infer.squeeze(0), np.float32)
    length_regulator_np = _as_numpy(length_regulator.squeeze(0), np.float32)
    target_lengths_np = _as_numpy(target_lengths, np.uint32)
    text_np = np.ascontiguousarray(np.asarray(text_ids, dtype=np.uint32))
    fake_inputs_np = _as_numpy(fake_inputs.squeeze(0), np.uint32)
    inputs_embeds_np = _as_numpy(inputs_embeds.squeeze(0), np.float32)
    attention_mask_np = _as_numpy(attention_mask.squeeze(0), np.uint32)

    manifest = {
        "format": "mit2-gpt-golden",
        "version": 1,
        "source_repo": str(Path(args.index_tts_repo).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "voice_name": profile.get("voice_name"),
        "text": args.text,
        "tokens": tokens,
        "segment_index": args.segment_index,
        "segment_tokens": segment,
        "text_ids": text_ids,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "length_penalty": 0.0,
            "max_mel_tokens": args.max_mel_tokens,
            "trimmed_code_len": int(codes_np.shape[0]),
            "target_lengths": target_lengths_np.astype(np.uint32).tolist(),
            "length_regulator_device": args.length_regulator_device,
        },
        "tensors": {
            "conds_latent": _write_raw(out_dir / "conds_latent.f32", conds_np),
            "speech_conditioning_latent": _write_raw(out_dir / "speech_conditioning_latent.f32", returned_cond_np),
            "text_ids": _write_raw(out_dir / "text_ids.u32", text_np),
            "fake_inputs": _write_raw(out_dir / "fake_inputs.u32", fake_inputs_np),
            "inputs_embeds": _write_raw(out_dir / "inputs_embeds.f32", inputs_embeds_np),
            "attention_mask": _write_raw(out_dir / "attention_mask.u32", attention_mask_np),
            "codes": _write_raw(out_dir / "codes.u32", codes_np),
            "gpt_latent": _write_raw(out_dir / "gpt_latent.f32", gpt_latent_np),
            "gpt_layer": _write_raw(out_dir / "gpt_layer.f32", gpt_layer_np),
            "vq2emb": _write_raw(out_dir / "vq2emb.f32", vq2emb_np),
            "s_infer": _write_raw(out_dir / "s_infer.f32", s_infer_np),
            "length_regulator": _write_raw(out_dir / "length_regulator.f32", length_regulator_np),
            "target_lengths": _write_raw(out_dir / "target_lengths.u32", target_lengths_np),
        },
        "raw_log": buf.getvalue(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PyTorch GPT greedy-code golden fixture.")
    add_reference_args(parser)
    parser.add_argument("--voice-name", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--max-mel-tokens", type=int, default=8)
    parser.add_argument(
        "--length-regulator-device",
        choices=("model", "cpu"),
        default="model",
        help="device used to emit the length_regulator golden tensor; cpu avoids MPS reduction drift in longer fixtures",
    )
    args = parser.parse_args()
    if not args.voice_name and not args.prompt_audio:
        raise SystemExit("provide --voice-name or --prompt-audio")
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
