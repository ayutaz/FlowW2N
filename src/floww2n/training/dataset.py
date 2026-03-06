"""Dataset and DataLoader for VAE and DiT training."""

import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import random


class VAEDataset(Dataset):
    """VAE training dataset: normal speech fixed-length segments."""

    def __init__(
        self,
        audio_dir,
        segment_length=65536,
        sample_rate=16000,
        source_sample_rate=None,
        file_extensions=(".wav", ".flac"),
    ):
        """
        Args:
            audio_dir: Audio file directory (searches recursively)
            segment_length: Segment length (samples). Default: 65536 (4.096s @ 16kHz)
            sample_rate: Target sampling rate. Default: 16000
            source_sample_rate: Source sampling rate (None=auto-detect)
            file_extensions: Target file extensions
        """
        self.audio_dir = Path(audio_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.source_sample_rate = source_sample_rate

        # Recursively search for files
        self.audio_files = []
        for ext in file_extensions:
            self.audio_files.extend(self.audio_dir.rglob(f"*{ext}"))
        self.audio_files = sorted(self.audio_files)

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = torchaudio.load(audio_path)

        # Resample
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.squeeze(0)  # (T,)

        # Random crop or pad
        if audio.shape[0] >= self.segment_length:
            start = random.randint(0, audio.shape[0] - self.segment_length)
            audio = audio[start : start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.shape[0]))

        # Peak normalization
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        return audio.unsqueeze(0)  # (1, segment_length)


class DiTDataset(Dataset):
    """Dataset for DiT training using pre-cached features.

    Loads pre-computed features from cache_features.py output:
    - {cache_dir}/whisper_h/{stem}.pt  : Whisper content features (T_frames, 512)
    - {cache_dir}/speaker/{stem}.pt    : Speaker embedding (192,)
    - {cache_dir}/vae_z1/{stem}.pt     : VAE latent of normal speech (64, L)
    - {cache_dir}/manifest.json        : List of dicts with paths and metadata

    Args:
        cache_dir: Directory with cached features (from cache_features.py)
        max_latent_length: Max VAE latent sequence length for random cropping.
            None means no cropping (use full-length sequences).
    """

    # Frame rate ratio: whisper_hz / vae_hz = 50 / 15.625 = 3.2
    WHISPER_TO_VAE_RATIO = 50.0 / 15.625

    def __init__(self, cache_dir, max_latent_length=None):
        self.cache_dir = Path(cache_dir)
        self.max_latent_length = max_latent_length

        # Load manifest
        manifest_path = self.cache_dir / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Filter out entries without VAE latent
        self.manifest = [
            item for item in self.manifest if item.get("vae_z1_path") is not None
        ]

        if len(self.manifest) == 0:
            raise ValueError(
                f"No valid entries found in {manifest_path}. "
                "Ensure VAE latents are cached (run cache_features.py with --vae-checkpoint)."
            )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]

        # Load cached tensors
        z1 = torch.load(
            self.cache_dir / item["vae_z1_path"], map_location="cpu", weights_only=True
        )  # (64, L)
        whisper_h = torch.load(
            self.cache_dir / item["whisper_h_path"], map_location="cpu", weights_only=True
        )  # (T_w, 512)
        speaker_emb = torch.load(
            self.cache_dir / item["speaker_path"], map_location="cpu", weights_only=True
        )  # (192,)

        # Random crop if max_latent_length is set and sequence is longer
        if self.max_latent_length is not None and z1.shape[1] > self.max_latent_length:
            # Random start for VAE latent
            max_start = z1.shape[1] - self.max_latent_length
            start_z1 = random.randint(0, max_start)
            z1 = z1[:, start_z1 : start_z1 + self.max_latent_length]

            # Proportionally crop whisper_h
            # whisper_frames = int(z1_frames * 3.2)
            start_w = int(start_z1 * self.WHISPER_TO_VAE_RATIO)
            length_w = int(self.max_latent_length * self.WHISPER_TO_VAE_RATIO)
            # Clamp to available length
            end_w = min(start_w + length_w, whisper_h.shape[0])
            whisper_h = whisper_h[start_w:end_w]

        return {
            "z1": z1,                # (64, L)
            "whisper_h": whisper_h,  # (T_w, 512)
            "speaker_emb": speaker_emb,  # (192,)
        }


def dit_collate_fn(batch):
    """Collate function for DiTDataset with variable-length padding.

    Pads z1 and whisper_h to the maximum length in the batch and creates
    corresponding padding masks.

    Args:
        batch: List of dicts from DiTDataset.__getitem__

    Returns:
        Dict with:
            z1: Padded VAE latents (B, 64, T_max)
            whisper_h: Padded Whisper features (B, T_w_max, 512)
            speaker_emb: Stacked speaker embeddings (B, 192)
            z1_mask: Padding mask for z1 (B, T_max), True = valid
            whisper_h_mask: Padding mask for whisper_h (B, T_w_max), True = valid
    """
    # Find max lengths
    z1_lengths = [item["z1"].shape[1] for item in batch]
    whisper_lengths = [item["whisper_h"].shape[0] for item in batch]
    max_z1_len = max(z1_lengths)
    max_whisper_len = max(whisper_lengths)

    batch_size = len(batch)
    io_channels = batch[0]["z1"].shape[0]  # 64
    whisper_dim = batch[0]["whisper_h"].shape[1]  # 512

    # Allocate padded tensors
    z1_padded = torch.zeros(batch_size, io_channels, max_z1_len)
    whisper_padded = torch.zeros(batch_size, max_whisper_len, whisper_dim)
    z1_mask = torch.zeros(batch_size, max_z1_len, dtype=torch.bool)
    whisper_mask = torch.zeros(batch_size, max_whisper_len, dtype=torch.bool)
    speaker_embs = []

    for i, item in enumerate(batch):
        z1_len = item["z1"].shape[1]
        w_len = item["whisper_h"].shape[0]

        z1_padded[i, :, :z1_len] = item["z1"]
        whisper_padded[i, :w_len, :] = item["whisper_h"]
        z1_mask[i, :z1_len] = True
        whisper_mask[i, :w_len] = True
        speaker_embs.append(item["speaker_emb"])

    return {
        "z1": z1_padded,
        "whisper_h": whisper_padded,
        "speaker_emb": torch.stack(speaker_embs),
        "z1_mask": z1_mask,
        "whisper_h_mask": whisper_mask,
    }
