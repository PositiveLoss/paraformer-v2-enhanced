from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jiwer
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from paraformer_v2 import ParaformerV2, ParaformerV2Config


VOCAB = " abcdefghijklmnopqrstuvwxyz'"
CHAR_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}


class TinyLibriSpeech(Dataset):
    def __init__(
        self,
        root: Path,
        url: str,
        download: bool,
        max_utterances: int,
        max_seconds: float,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        archive_root = root / "LibriSpeech" / url
        if not archive_root.exists():
            torchaudio.datasets.LIBRISPEECH(str(root), url=url, download=download)
        if not archive_root.exists():
            raise RuntimeError(f"LibriSpeech split not found at {archive_root}; rerun with --download.")

        self.items = []
        transcript_items = []
        for transcript_file in sorted(archive_root.glob("*/*/*.trans.txt")):
            for line in transcript_file.read_text(encoding="utf-8").splitlines():
                utterance_id, transcript = line.split(" ", 1)
                transcript_items.append((transcript_file.parent / f"{utterance_id}.flac", transcript))

        for audio_path, transcript in transcript_items:
            text = normalize_text(transcript)
            if not text:
                continue
            waveform, sample_rate = load_flac_with_ffmpeg(audio_path)
            seconds = waveform.size(-1) / sample_rate
            if seconds > max_seconds:
                continue
            token_count = len(text)
            mel_frames = max(0, (waveform.size(-1) - 400) // 160 + 1)
            encoder_frames = (mel_frames + 3) // 4
            # The model subsamples by 4; keep CTC alignment comfortably feasible.
            if encoder_frames <= token_count * 2 + 1:
                continue
            self.items.append((waveform, sample_rate, text))
            if len(self.items) >= max_utterances:
                break
        if not self.items:
            raise RuntimeError(
                "No usable LibriSpeech examples found. Try a larger max_seconds or enable --download."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        return self.items[index]


def load_flac_with_ffmpeg(path: Path, sample_rate: int = 16_000) -> tuple[torch.Tensor, int]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    waveform = torch.frombuffer(bytearray(proc.stdout), dtype=torch.float32).unsqueeze(0)
    return waveform, sample_rate


def normalize_text(text: str) -> str:
    lowered = text.lower()
    return "".join(ch for ch in lowered if ch in CHAR_TO_ID).strip()


def encode_text(text: str) -> torch.Tensor:
    return torch.tensor([CHAR_TO_ID[ch] for ch in text], dtype=torch.long)


def decode_ids(ids: list[int]) -> str:
    return "".join(ID_TO_CHAR[i] for i in ids if i in ID_TO_CHAR).strip()


def greedy_ctc_decode(frame_ids: torch.Tensor, blank_id: int) -> list[int]:
    out = []
    last = None
    for idx in frame_ids.tolist():
        if idx != blank_id and idx != last:
            out.append(idx)
        last = idx
    return out


class FeatureExtractor:
    def __init__(self, sample_rate: int = 16_000, n_mels: int = 80) -> None:
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            center=False,
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        mel = self.mel(waveform).squeeze(0).transpose(0, 1)
        return torch.log(mel.clamp_min(1e-5))


def make_collate() -> callable:
    extractor = FeatureExtractor()

    def collate(batch: list[tuple[torch.Tensor, int, str]]) -> dict[str, torch.Tensor | list[str]]:
        features = [extractor(waveform, sample_rate) for waveform, sample_rate, _ in batch]
        texts = [text for _, _, text in batch]
        targets = [encode_text(text) for text in texts]
        return {
            "features": pad_sequence(features, batch_first=True),
            "feature_lengths": torch.tensor([feat.size(0) for feat in features], dtype=torch.long),
            "targets": pad_sequence(targets, batch_first=True),
            "target_lengths": torch.tensor([target.size(0) for target in targets], dtype=torch.long),
            "texts": texts,
        }

    return collate


def evaluate(model: ParaformerV2, loader: DataLoader, device: torch.device) -> dict[str, float | list[dict[str, str]]]:
    model.eval()
    refs: list[str] = []
    ctc_hyps: list[str] = []
    dec_hyps: list[str] = []
    examples = []
    blank_id = model.config.resolved_blank_id
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            out = model(features, feature_lengths)
            ctc_ids = out["ctc_log_probs"].argmax(dim=-1).cpu()
            decoder_ids = out["decoder_logits"].argmax(dim=-1).cpu()
            query_lengths = out["query_lengths"].cpu()
            encoder_lengths = out["encoder_lengths"].cpu()
            for i, ref in enumerate(batch["texts"]):
                refs.append(ref)
                ctc_text = decode_ids(
                    greedy_ctc_decode(ctc_ids[i, : encoder_lengths[i]], blank_id)
                )
                dec_text = decode_ids(decoder_ids[i, : query_lengths[i]].tolist())
                ctc_hyps.append(ctc_text)
                dec_hyps.append(dec_text)
                if len(examples) < 5:
                    examples.append({"ref": ref, "ctc": ctc_text, "decoder": dec_text})
    return {
        "ctc_wer": float(jiwer.wer(refs, ctc_hyps)),
        "decoder_wer": float(jiwer.wer(refs, dec_hyps)),
        "ctc_cer": float(jiwer.cer(refs, ctc_hyps)),
        "decoder_cer": float(jiwer.cer(refs, dec_hyps)),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--url", default="dev-clean")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-utterances", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--time-cap-seconds", type=float, default=900.0)
    parser.add_argument("--alignment-mode", choices=["viterbi", "uniform"], default="viterbi")
    parser.add_argument("--alignment-backend", choices=["auto", "python", "cython", "triton"], default="auto")
    parser.add_argument("--eval-on-train", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/librispeech_probe_metrics.json"))
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TinyLibriSpeech(
        args.data_root,
        args.url,
        args.download,
        args.max_utterances,
        args.max_seconds,
    )
    train_count = max(1, int(len(dataset) * 0.75))
    train_set, eval_set = torch.utils.data.random_split(
        dataset,
        [train_count, len(dataset) - train_count],
        generator=torch.Generator().manual_seed(13),
    )
    if len(eval_set) == 0:
        eval_set = train_set

    loader_kwargs = {"batch_size": args.batch_size, "collate_fn": make_collate()}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(train_set if args.eval_on_train else eval_set, shuffle=False, **loader_kwargs)

    config = ParaformerV2Config(
        input_dim=80,
        vocab_size=len(VOCAB),
        encoder_dim=96,
        decoder_dim=96,
        encoder_layers=3,
        decoder_layers=2,
        encoder_ff_dim=384,
        decoder_ff_dim=384,
        attention_heads=4,
        dropout=0.1,
    )
    model = ParaformerV2(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    history = []
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if time.time() - started > args.time_cap_seconds:
                break
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(
                batch["features"].to(device),
                batch["feature_lengths"].to(device),
                batch["targets"].to(device),
                batch["target_lengths"].to(device),
                alignment_mode=args.alignment_mode,
                alignment_backend=args.alignment_backend,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(losses["loss"].detach().cpu()),
                    "ctc_loss": float(losses["ctc_loss"].cpu()),
                    "ce_loss": float(losses["ce_loss"].cpu()),
                }
            )
            if args.progress_every > 0 and len(history) % args.progress_every == 0:
                print(json.dumps(history[-1]), flush=True)
        if time.time() - started > args.time_cap_seconds:
            break

    metrics = evaluate(model, eval_loader, device)
    result = {
        "probe": "tiny-librispeech",
        "url": args.url,
        "device": str(device),
        "dataset_size": len(dataset),
        "train_size": len(train_set),
        "eval_size": len(eval_set),
        "epochs_requested": args.epochs,
        "updates": len(history),
        "alignment_mode": args.alignment_mode,
        "alignment_backend": args.alignment_backend,
        "eval_on_train": args.eval_on_train,
        "elapsed_seconds": round(time.time() - started, 3),
        "config": asdict(config),
        "history_tail": history[-10:],
        "metrics": metrics,
        "interpretation": (
            "Tiny probe only. Use it to catch pipeline failures and rough loss/decoding behavior; "
            "do not compare these metrics to paper-scale LibriSpeech results."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
