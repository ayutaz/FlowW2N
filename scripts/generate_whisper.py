#!/usr/bin/env python
"""Script: Generate synthetic whisper from normal speech using 4 methods."""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from floww2n.data.whisper_synthesis import WhisperSynthesizer


def generate_whisper(data_dir, output_dir, sample_rate=16000, methods=None):
    """Generate synthetic whisper for all audio files."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    synth = WhisperSynthesizer(sr=sample_rate)

    audio_files = []
    for ext in (".wav", ".flac"):
        audio_files.extend(data_path.rglob(f"*{ext}"))
    audio_files = sorted(audio_files)
    print(f"Found {len(audio_files)} audio files")

    for audio_path in tqdm(audio_files, desc="Generating whisper"):
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != sample_rate:
            import torchaudio
            import torch
            audio_t = torch.from_numpy(audio).float()
            audio_t = torchaudio.transforms.Resample(sr, sample_rate)(audio_t)
            audio = audio_t.numpy()

        # Generate 4 versions (one per method)
        for i, method_name in enumerate(["lpc", "glottal", "formant", "praat"]):
            whispered = synth(audio, method_index=i)
            out_file = out_path / method_name / f"{audio_path.stem}.wav"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_file), whispered, sample_rate)

    print(f"Generated synthetic whisper for {len(audio_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic whisper from normal speech"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    generate_whisper(args.data_dir, args.output_dir, args.sample_rate)


if __name__ == "__main__":
    main()
