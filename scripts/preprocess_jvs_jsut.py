#!/usr/bin/env python
"""Script: Preprocess JVS and JSUT datasets to 16kHz mono WAV.

JVS (Japanese Versatile Speech): 100 speakers, ~30h
JSUT (Japanese Speech Corpus of Saruwatari-lab, University of Tokyo): 1 speaker, ~10h
"""

import argparse
from pathlib import Path

import torchaudio
from tqdm import tqdm

TARGET_SR = 16000


def process_file(input_path, output_path):
    """Load audio, convert to 16kHz mono, and save as WAV.

    Args:
        input_path: Path to source audio file
        output_path: Path to save processed audio
    """
    audio, sr = torchaudio.load(str(input_path))

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != TARGET_SR:
        audio = torchaudio.transforms.Resample(sr, TARGET_SR)(audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio, TARGET_SR)


def preprocess_jvs(jvs_dir, output_dir):
    """Preprocess JVS corpus.

    Expected structure: jvs_dir/{speaker}/parallel100/wav24kHz16bit/*.wav
    Output structure:   output_dir/{speaker}/{utterance_id}.wav
    """
    jvs_path = Path(jvs_dir)
    out_path = Path(output_dir)

    wav_files = sorted(jvs_path.glob("*/parallel100/wav24kHz16bit/*.wav"))
    print(f"JVS: found {len(wav_files)} files")

    for wav in tqdm(wav_files, desc="JVS"):
        speaker = wav.parts[-4]  # e.g. jvs001
        utt_id = wav.stem
        output_file = out_path / speaker / f"{utt_id}.wav"
        process_file(wav, output_file)

    return len(wav_files)


def preprocess_jsut(jsut_dir, output_dir):
    """Preprocess JSUT corpus.

    Expected structure: jsut_dir/basic5000/wav/*.wav
    Output structure:   output_dir/jsut001/{utterance_id}.wav
    """
    jsut_path = Path(jsut_dir)
    out_path = Path(output_dir)

    wav_files = sorted(jsut_path.glob("basic5000/wav/*.wav"))
    print(f"JSUT: found {len(wav_files)} files")

    for wav in tqdm(wav_files, desc="JSUT"):
        utt_id = wav.stem
        output_file = out_path / "jsut001" / f"{utt_id}.wav"
        process_file(wav, output_file)

    return len(wav_files)


def main():
    parser = argparse.ArgumentParser(description="Preprocess JVS/JSUT to 16kHz mono")
    parser.add_argument("--jvs-dir", type=str, default=None, help="Path to JVS corpus root")
    parser.add_argument("--jsut-dir", type=str, default=None, help="Path to JSUT corpus root")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for processed audio"
    )
    args = parser.parse_args()

    if args.jvs_dir is None and args.jsut_dir is None:
        parser.error("At least one of --jvs-dir or --jsut-dir must be specified")

    total = 0
    if args.jvs_dir is not None:
        total += preprocess_jvs(args.jvs_dir, args.output_dir)
    if args.jsut_dir is not None:
        total += preprocess_jsut(args.jsut_dir, args.output_dir)

    print(f"\nDone. Processed {total} files to {args.output_dir}")


if __name__ == "__main__":
    main()
