"""ECAPA-TDNN speaker encoder wrapper."""

import torch
import torch.nn as nn

# Patch torchaudio for speechbrain compatibility (torchaudio >= 2.10 removed list_audio_backends)
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["default"]

from speechbrain.inference.speaker import EncoderClassifier


class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder wrapper.

    Outputs speaker embedding e_spk of shape (batch, 192).
    All parameters are frozen.
    """

    def __init__(self, model_name="speechbrain/spkrec-ecapa-voxceleb",
                 save_dir="pretrained_models/spkrec-ecapa-voxceleb",
                 device="cpu"):
        super().__init__()
        self.output_dim = 192
        self.device = device

        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=save_dir,
            run_opts={"device": device},
        )
        # Freeze
        for param in self.classifier.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, audio):
        """Extract speaker embedding.

        Args:
            audio: Waveform tensor (batch, samples) @ 16kHz

        Returns:
            e_spk: Speaker embedding (batch, 192)
        """
        if isinstance(audio, torch.Tensor):
            waveform = audio
        else:
            waveform = torch.from_numpy(audio).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        embeddings = self.classifier.encode_batch(waveform.to(self.device))
        # shape: (batch, 1, 192) -> (batch, 192)
        return embeddings.squeeze(1)

    def train(self, mode=True):
        """Override to keep model in eval mode."""
        return super().train(False)


if __name__ == "__main__":
    encoder = SpeakerEncoder()
    audio = torch.randn(2, 48000)
    e_spk = encoder(audio)
    print(f"Speaker embedding shape: {e_spk.shape}")  # (2, 192)
