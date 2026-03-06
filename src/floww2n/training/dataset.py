"""Dataset and DataLoader for VAE and DiT training."""

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
