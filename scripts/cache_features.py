#!/usr/bin/env python
"""Script: Pre-compute and cache Whisper/ECAPA/VAE features for DiT training."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def _load_and_preprocess(audio_path, sample_rate):
    """Load an audio file and preprocess to mono at target sample rate.

    Returns:
        audio: 1-D tensor (T,)
    """
    audio, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.squeeze(0)  # (T,)


def cache_features(
    data_dir,
    output_dir,
    vae_checkpoint=None,
    config_path="configs/vae.json",
    sample_rate=16000,
    device="cuda",
    language=None,
    batch_size=16,
):
    """Pre-compute and cache features for all audio files.

    For each audio file, saves:
    - {output_dir}/whisper_h/{stem}.pt   : (T_frames, 512)
    - {output_dir}/speaker/{stem}.pt     : (192,)
    - {output_dir}/vae_z1/{stem}.pt      : (64, L)
    - {output_dir}/manifest.json         : list of dicts with paths
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Create output directories
    (out_path / "whisper_h").mkdir(parents=True, exist_ok=True)
    (out_path / "speaker").mkdir(parents=True, exist_ok=True)
    (out_path / "vae_z1").mkdir(parents=True, exist_ok=True)

    # Load models
    from floww2n.models.content_encoder import ContentEncoder
    from floww2n.models.speaker_encoder import SpeakerEncoder
    from floww2n.models.vae import AudioAutoencoder

    print("Loading content encoder (Whisper Base)...")
    content_encoder = ContentEncoder(device=device)

    print("Loading speaker encoder (ECAPA-TDNN)...")
    speaker_encoder = SpeakerEncoder(device=device)

    vae = None
    vae_strides = None
    if vae_checkpoint:
        print(f"Loading VAE from {vae_checkpoint}...")
        with open(config_path) as f:
            vae_config = json.load(f)
        model_cfg = vae_config["model"]
        vae_strides = model_cfg["strides"]
        vae = AudioAutoencoder(
            io_channels=model_cfg["io_channels"],
            latent_dim=model_cfg["latent_dim"],
            encoder_latent_dim=model_cfg["encoder_latent_dim"],
            channels=model_cfg["channels"],
            c_mults=model_cfg["c_mults"],
            strides=vae_strides,
            use_snake=model_cfg["use_snake"],
        ).to(device)
        checkpoint = torch.load(vae_checkpoint, map_location=device)
        vae.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

    # Find audio files
    audio_files = []
    for ext in (".wav", ".flac"):
        audio_files.extend(data_path.rglob(f"*{ext}"))
    audio_files = sorted(audio_files)
    print(f"Found {len(audio_files)} audio files")
    print(f"Processing with batch_size={batch_size}")

    manifest = []

    for batch_start in tqdm(
        range(0, len(audio_files), batch_size),
        desc="Caching features",
        total=(len(audio_files) + batch_size - 1) // batch_size,
    ):
        batch_files = audio_files[batch_start : batch_start + batch_size]

        # Load and preprocess all audio in this batch
        audios = []
        stems = []
        audio_lengths = []
        for audio_path in batch_files:
            audio = _load_and_preprocess(audio_path, sample_rate)
            audios.append(audio)
            stems.append(audio_path.stem)
            audio_lengths.append(len(audio))

        # --- 1. Whisper features (batch) ---
        # ContentEncoder accepts a list of numpy arrays with different lengths
        audio_np_list = [a.numpy() for a in audios]
        h, mask = content_encoder(audio_np_list)
        # h: (B, 1500, 512), mask: (B, 1500)
        for j in range(len(batch_files)):
            valid_frames = mask[j].sum().item()
            whisper_h = h[j, :valid_frames].cpu()  # (valid_frames, 512)
            torch.save(whisper_h, out_path / "whisper_h" / f"{stems[j]}.pt")

        # --- 2. Speaker embedding (batch) ---
        # Pad audios to same length for batched speaker encoder
        max_len = max(audio_lengths)
        padded = torch.zeros(len(audios), max_len)
        for j, audio in enumerate(audios):
            padded[j, : len(audio)] = audio
        e_spk_batch = speaker_encoder(padded)  # (B, 192)
        for j in range(len(batch_files)):
            torch.save(e_spk_batch[j].cpu(), out_path / "speaker" / f"{stems[j]}.pt")

        # --- 3. VAE latent (batch, if VAE checkpoint provided) ---
        vae_z1_results = [None] * len(batch_files)
        if vae is not None:
            with torch.no_grad():
                # Pad to same length for batched VAE encoding
                padded_vae = torch.zeros(len(audios), 1, max_len, device=device)
                for j, audio in enumerate(audios):
                    padded_vae[j, 0, : len(audio)] = audio
                z1_batch, _ = vae.encode(padded_vae)  # (B, 64, L)
                for j in range(len(batch_files)):
                    # Trim VAE latent to the length corresponding to original audio
                    # VAE downsamples by product of strides; compute expected latent length
                    expected_L = audio_lengths[j]
                    for s in vae_strides:
                        expected_L = (expected_L + s - 1) // s
                    z1 = z1_batch[j, :, :expected_L].cpu()  # (64, L)
                    torch.save(z1, out_path / "vae_z1" / f"{stems[j]}.pt")
                    vae_z1_results[j] = f"vae_z1/{stems[j]}.pt"

        # Build manifest entries for this batch
        for j in range(len(batch_files)):
            valid_frames = mask[j].sum().item()
            entry = {
                "audio_path": str(batch_files[j]),
                "stem": stems[j],
                "whisper_h_path": f"whisper_h/{stems[j]}.pt",
                "speaker_path": f"speaker/{stems[j]}.pt",
                "vae_z1_path": vae_z1_results[j],
                "audio_length_samples": audio_lengths[j],
                "valid_whisper_frames": valid_frames,
            }
            if language is not None:
                entry["language"] = language
            manifest.append(entry)

    # Save manifest
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Cached {len(manifest)} files to {output_dir}")
    print(f"Manifest saved to {out_path / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute and cache features for DiT training")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save cached features"
    )
    parser.add_argument(
        "--vae-checkpoint", type=str, default=None, help="Path to trained VAE checkpoint (optional)"
    )
    parser.add_argument("--config", type=str, default="configs/vae.json", help="Path to VAE config")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language tag to record in manifest (e.g. en, ja)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of audio files to process per batch (default: 16)",
    )
    args = parser.parse_args()

    cache_features(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vae_checkpoint=args.vae_checkpoint,
        config_path=args.config,
        sample_rate=args.sample_rate,
        device=args.device,
        language=args.language,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
