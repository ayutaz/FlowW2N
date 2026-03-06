#!/usr/bin/env python
"""Script: Pre-compute and cache Whisper/ECAPA/VAE features for DiT training."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def cache_features(
    data_dir,
    output_dir,
    vae_checkpoint=None,
    config_path="configs/vae.json",
    sample_rate=16000,
    device="cuda",
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
    if vae_checkpoint:
        print(f"Loading VAE from {vae_checkpoint}...")
        with open(config_path) as f:
            vae_config = json.load(f)
        model_cfg = vae_config["model"]
        vae = AudioAutoencoder(
            io_channels=model_cfg["io_channels"],
            latent_dim=model_cfg["latent_dim"],
            encoder_latent_dim=model_cfg["encoder_latent_dim"],
            channels=model_cfg["channels"],
            c_mults=model_cfg["c_mults"],
            strides=model_cfg["strides"],
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

    manifest = []

    for audio_path in tqdm(audio_files, desc="Caching features"):
        stem = audio_path.stem

        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)  # (T,)

        # 1. Whisper features
        h, mask = content_encoder(audio.unsqueeze(0).numpy())
        # h: (1, 1500, 512), mask: (1, 1500)
        valid_frames = mask[0].sum().item()
        whisper_h = h[0, :valid_frames].cpu()  # (valid_frames, 512)
        torch.save(whisper_h, out_path / "whisper_h" / f"{stem}.pt")

        # 2. Speaker embedding
        e_spk = speaker_encoder(audio.unsqueeze(0))  # (1, 192)
        e_spk = e_spk.squeeze(0).cpu()  # (192,)
        torch.save(e_spk, out_path / "speaker" / f"{stem}.pt")

        # 3. VAE latent (if VAE checkpoint provided)
        vae_z1_path = None
        if vae is not None:
            with torch.no_grad():
                audio_input = audio.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
                z1, _ = vae.encode(audio_input)
                z1 = z1.squeeze(0).cpu()  # (64, L)
            torch.save(z1, out_path / "vae_z1" / f"{stem}.pt")
            vae_z1_path = f"vae_z1/{stem}.pt"

        manifest.append(
            {
                "audio_path": str(audio_path),
                "stem": stem,
                "whisper_h_path": f"whisper_h/{stem}.pt",
                "speaker_path": f"speaker/{stem}.pt",
                "vae_z1_path": vae_z1_path,
                "audio_length_samples": len(audio),
                "valid_whisper_frames": valid_frames,
            }
        )

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
    args = parser.parse_args()

    cache_features(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vae_checkpoint=args.vae_checkpoint,
        config_path=args.config,
        sample_rate=args.sample_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
