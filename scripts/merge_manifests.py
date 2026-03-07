#!/usr/bin/env python
"""Script: Merge multiple cache_features manifests into one.

Used to combine per-language cache directories into a single manifest
for multilingual DiT training.
"""

import argparse
import json
from pathlib import Path


def merge_manifests(input_dirs, output_dir):
    """Merge manifest.json files from multiple cache directories.

    Paths inside each manifest are converted to absolute paths so that
    the merged manifest can be used from a different working directory.

    Args:
        input_dirs: List of cache directories, each containing manifest.json
        output_dir: Directory to write the merged manifest.json
    """
    merged = []

    for cache_dir in input_dirs:
        cache_path = Path(cache_dir).resolve()
        manifest_path = cache_path / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {cache_dir}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        for entry in manifest:
            # Convert relative feature paths to absolute paths
            for key in ("whisper_h_path", "speaker_path", "vae_z1_path"):
                if entry.get(key) is not None:
                    entry[key] = str(cache_path / entry[key])
            merged.append(entry)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "manifest.json", "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(merged)} entries from {len(input_dirs)} directories")
    print(f"Saved to {out_path / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple cache manifests")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Cache directories to merge (each must contain manifest.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the merged manifest",
    )
    args = parser.parse_args()

    merge_manifests(args.inputs, args.output)


if __name__ == "__main__":
    main()
