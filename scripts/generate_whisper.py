#!/usr/bin/env python
"""Script: Generate synthetic whisper from normal speech using 4 methods."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def _process_single_file(audio_path, out_path, sample_rate, method_names):
    """Process a single audio file through all whisper synthesis methods.

    This function is designed to run in a worker process.
    """
    from floww2n.data.whisper_synthesis import WhisperSynthesizer

    audio, sr = sf.read(audio_path, dtype="float32")
    if sr != sample_rate:
        import torch
        import torchaudio

        audio_t = torch.from_numpy(audio).float()
        audio_t = torchaudio.transforms.Resample(sr, sample_rate)(audio_t)
        audio = audio_t.numpy()

    synth = WhisperSynthesizer(sr=sample_rate)

    for i, method_name in enumerate(method_names):
        whispered = synth(audio, method_index=i)
        out_file = Path(out_path) / method_name / f"{Path(audio_path).stem}.wav"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_file), whispered, sample_rate)


def generate_whisper(data_dir, output_dir, sample_rate=16000, num_workers=None):
    """Generate synthetic whisper for all audio files using multiprocessing."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    method_names = ["lpc", "glottal", "formant", "praat"]

    audio_files = []
    for ext in (".wav", ".flac"):
        audio_files.extend(data_path.rglob(f"*{ext}"))
    audio_files = sorted(audio_files)
    print(f"Found {len(audio_files)} audio files")
    print(f"Processing with {num_workers} workers")

    # Pre-create output directories
    for method_name in method_names:
        (out_path / method_name).mkdir(parents=True, exist_ok=True)

    if num_workers <= 1:
        # Single-process fallback
        from floww2n.data.whisper_synthesis import WhisperSynthesizer

        synth = WhisperSynthesizer(sr=sample_rate)
        for audio_path in tqdm(audio_files, desc="Generating whisper"):
            audio, sr_read = sf.read(audio_path, dtype="float32")
            if sr_read != sample_rate:
                import torch
                import torchaudio

                audio_t = torch.from_numpy(audio).float()
                audio_t = torchaudio.transforms.Resample(sr_read, sample_rate)(audio_t)
                audio = audio_t.numpy()
            for i, method_name in enumerate(method_names):
                whispered = synth(audio, method_index=i)
                out_file = out_path / method_name / f"{audio_path.stem}.wav"
                sf.write(str(out_file), whispered, sample_rate)
    else:
        # Multi-process execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_file,
                    str(audio_path),
                    str(out_path),
                    sample_rate,
                    method_names,
                ): audio_path
                for audio_path in audio_files
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Generating whisper"
            ):
                try:
                    future.result()
                except Exception as e:
                    failed_path = futures[future]
                    print(f"Error processing {failed_path}: {e}")

    print(f"Generated synthetic whisper for {len(audio_files)} files")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic whisper from normal speech")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: min(cpu_count, 8))",
    )
    args = parser.parse_args()

    generate_whisper(args.data_dir, args.output_dir, args.sample_rate, args.num_workers)


if __name__ == "__main__":
    main()
