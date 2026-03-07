#!/usr/bin/env python
"""Script: Run whisper-to-normal conversion."""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

from floww2n.inference import FlowW2NPipeline

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".sph", ".m4a"}


def find_audio_files(path):
    """Find audio files from a path (single file or directory).

    Args:
        path: Path to a single audio file or a directory containing audio files.

    Returns:
        List of Path objects for audio files found.
    """
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(p.glob(f"*{ext}"))
            files.extend(p.glob(f"*{ext.upper()}"))
        files.sort()
        return files
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run FlowW2N whisper-to-normal speech conversion")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input whisper audio file or directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted audio files",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to VAE checkpoint (.pt state_dict)",
    )
    parser.add_argument(
        "--dit-checkpoint",
        type=str,
        required=True,
        help="Path to DiT/FlowW2N checkpoint (.pt state_dict)",
    )
    parser.add_argument(
        "--vae-config",
        type=str,
        default="configs/vae.json",
        help="Path to VAE config JSON (default: configs/vae.json)",
    )
    parser.add_argument(
        "--dit-config",
        type=str,
        default="configs/dit.json",
        help="Path to DiT config JSON (default: configs/dit.json)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of Euler integration steps (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for multilingual models (e.g. en, ja)",
    )
    args = parser.parse_args()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # Find input audio files
    audio_files = find_audio_files(args.input)
    if not audio_files:
        print(f"No audio files found at: {args.input}")
        sys.exit(1)
    print(f"Found {len(audio_files)} audio file(s) to process.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading pipeline...")
    pipeline = FlowW2NPipeline.from_pretrained(
        vae_checkpoint=args.vae_checkpoint,
        dit_checkpoint=args.dit_checkpoint,
        vae_config=args.vae_config,
        dit_config=args.dit_config,
        device=args.device,
        num_steps=args.num_steps,
    )
    print("Pipeline loaded successfully.")

    # Process each file
    for i, audio_path in enumerate(audio_files):
        print(f"\n[{i + 1}/{len(audio_files)}] Processing: {audio_path.name}")

        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        # torchaudio loads as (channels, samples); take first channel if stereo
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]  # keep (1, samples)
        waveform = waveform.squeeze(0)  # (samples,)

        # Run conversion
        audio_out = pipeline(
            waveform,
            sample_rate=sr,
            num_steps=args.num_steps,
            seed=args.seed,
            language=args.language,
        )

        # audio_out is (batch, samples); take first element
        if audio_out.dim() == 2:
            audio_out = audio_out[0]  # (samples,)

        # Move to CPU for saving
        audio_out = audio_out.cpu()

        # Save output - VAE always outputs at 16kHz
        output_path = output_dir / f"{audio_path.stem}_converted.wav"
        torchaudio.save(
            str(output_path),
            audio_out.unsqueeze(0),  # (1, samples) for torchaudio
            16000,  # VAE output sample rate is always 16kHz
        )
        print(f"  Saved: {output_path}")

    print(f"\nDone. {len(audio_files)} file(s) processed. Output: {output_dir}")


if __name__ == "__main__":
    main()
