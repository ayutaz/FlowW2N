"""Whisper Base encoder wrapper for content feature extraction."""

import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel


class ContentEncoder(nn.Module):
    """Whisper Base encoder wrapper that extracts layer 5 features.

    Outputs content features h of shape (batch, 1500, 512) at 50Hz.
    All parameters are frozen.
    """

    def __init__(self, model_name="openai/whisper-base", layer_index=6, device="cpu"):
        super().__init__()
        self.layer_index = layer_index  # hidden_states[6] = layer 5 output
        self.output_dim = 512  # d_model for whisper-base

        # Load model and feature extractor
        self.model = WhisperModel.from_pretrained(model_name)
        self.model = self.model.to(device)  # Move to specified device (e.g. GPU)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, audio, sample_rate=16000):
        """Extract layer 5 features from audio.

        Args:
            audio: Raw waveform tensor (batch, samples) or numpy array
            sample_rate: Sample rate of input audio (default: 16000)

        Returns:
            h: Content features (batch, 1500, 512)
            mask: Valid frame mask (batch, 1500) - True for valid frames
        """
        # Handle tensor input - convert to numpy for feature_extractor
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Ensure batch dimension
        if audio_np.ndim == 1:
            audio_np = [audio_np]
        elif audio_np.ndim == 2:
            audio_np = list(audio_np)

        # Compute valid frames for masking
        # Whisper outputs 1500 frames for 30s, so 50 frames/sec
        valid_frames_list = []
        for a in audio_np:
            audio_len_sec = len(a) / sample_rate
            valid_frames = min(int(audio_len_sec * 50), 1500)
            valid_frames_list.append(valid_frames)

        # Extract features
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(next(self.model.parameters()).device)

        # Forward through encoder
        encoder_outputs = self.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract layer 5 output
        h = encoder_outputs.hidden_states[self.layer_index]  # (batch, 1500, 512)

        # Create mask
        batch_size = h.shape[0]
        mask = torch.zeros(batch_size, 1500, dtype=torch.bool, device=h.device)
        for i, vf in enumerate(valid_frames_list):
            mask[i, :vf] = True

        return h, mask

    def train(self, mode=True):
        """Override to keep model in eval mode."""
        # Always keep in eval mode since parameters are frozen
        return super().train(False)


if __name__ == "__main__":
    import numpy as np

    encoder = ContentEncoder()
    # ダミー入力 (3秒の音声)
    audio = np.random.randn(2, 48000).astype(np.float32)
    h, mask = encoder(audio)
    print(f"Content features shape: {h.shape}")  # (2, 1500, 512)
    print(f"Mask shape: {mask.shape}")  # (2, 1500)
    print(f"Valid frames: {mask.sum(dim=1).tolist()}")  # [150, 150]
