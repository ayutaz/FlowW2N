"""FlowW2NPipeline: end-to-end whisper-to-normal conversion."""

import json

import numpy as np
import torch
import torch.nn as nn

from ..models.vae import AudioAutoencoder
from ..models.floww2n import FlowW2NModel
from ..models.content_encoder import ContentEncoder
from ..models.speaker_encoder import SpeakerEncoder
from .sampler import euler_solve


class FlowW2NPipeline:
    """End-to-end whisper-to-normal speech conversion pipeline.

    Loads all models (VAE, DiT/FlowW2N, ContentEncoder, SpeakerEncoder)
    and performs inference from raw whisper audio to converted normal audio.

    Usage:
        pipeline = FlowW2NPipeline.from_pretrained(
            vae_checkpoint="outputs/vae/vae_ema_final.pt",
            dit_checkpoint="outputs/dit/dit_ema_final.pt",
            vae_config="configs/vae.json",
            dit_config="configs/dit.json",
            device="cuda",
        )
        audio_out = pipeline(whisper_audio, sample_rate=16000)
    """

    def __init__(self, vae, floww2n_model, content_encoder, speaker_encoder,
                 device="cpu", num_steps=10, compression_ratio=1024):
        """
        Args:
            vae: AudioAutoencoder instance
            floww2n_model: FlowW2NModel instance
            content_encoder: ContentEncoder instance
            speaker_encoder: SpeakerEncoder instance
            device: Device string
            num_steps: Default number of Euler steps
            compression_ratio: VAE compression ratio (default: 1024)
        """
        self.vae = vae
        self.floww2n_model = floww2n_model
        self.content_encoder = content_encoder
        self.speaker_encoder = speaker_encoder
        self.device = device
        self.num_steps = num_steps
        self.compression_ratio = compression_ratio

        # Move models to device and set to eval mode
        self.vae.to(device).eval()
        self.floww2n_model.to(device).eval()
        self.content_encoder.to(device).eval()
        # SpeakerEncoder uses internal device management via speechbrain

    @classmethod
    def from_pretrained(cls, vae_checkpoint, dit_checkpoint,
                        vae_config="configs/vae.json",
                        dit_config="configs/dit.json",
                        device="cuda", num_steps=10):
        """Load pipeline from checkpoints and configs.

        Loads:
        1. VAE from vae_checkpoint (state_dict for AudioAutoencoder)
        2. FlowW2NModel from dit_checkpoint (state_dict)
        3. ContentEncoder (downloads Whisper Base from HuggingFace)
        4. SpeakerEncoder (downloads ECAPA-TDNN from SpeechBrain)

        Args:
            vae_checkpoint: Path to VAE state_dict checkpoint (.pt)
            dit_checkpoint: Path to DiT/FlowW2N state_dict checkpoint (.pt)
            vae_config: Path to VAE config JSON file
            dit_config: Path to DiT config JSON file
            device: Target device string (default: "cuda")
            num_steps: Default number of Euler steps (default: 10)

        Returns:
            FlowW2NPipeline instance with loaded models
        """
        # Load VAE config and model
        with open(vae_config, "r") as f:
            vae_cfg = json.load(f)
        vae_model_cfg = vae_cfg["model"]

        vae = AudioAutoencoder(
            io_channels=vae_model_cfg.get("io_channels", 1),
            latent_dim=vae_model_cfg.get("latent_dim", 64),
            encoder_latent_dim=vae_model_cfg.get("encoder_latent_dim", 128),
            channels=vae_model_cfg.get("channels", 128),
            c_mults=vae_model_cfg.get("c_mults", [1, 2, 4, 8]),
            strides=vae_model_cfg.get("strides", [4, 4, 8, 8]),
            use_snake=vae_model_cfg.get("use_snake", True),
        )

        vae_state_dict = torch.load(vae_checkpoint, map_location="cpu", weights_only=True)
        vae.load_state_dict(vae_state_dict)

        compression_ratio = vae_model_cfg.get("compression_ratio", 1024)

        # Load DiT config and FlowW2NModel
        with open(dit_config, "r") as f:
            dit_cfg = json.load(f)
        dit_model_cfg = dit_cfg["model"]

        floww2n_model = FlowW2NModel(dit_config=dit_model_cfg)

        dit_state_dict = torch.load(dit_checkpoint, map_location="cpu", weights_only=True)
        floww2n_model.load_state_dict(dit_state_dict)

        # Load conditioning encoders
        cond_cfg = dit_cfg.get("conditioning", {})

        content_cfg = cond_cfg.get("content_encoder", {})
        content_encoder = ContentEncoder(
            model_name=content_cfg.get("model_name", "openai/whisper-base"),
            layer_index=content_cfg.get("layer_index", 6),
            device=device,
        )

        speaker_cfg = cond_cfg.get("speaker_encoder", {})
        speaker_encoder = SpeakerEncoder(
            model_name=speaker_cfg.get("model_name", "speechbrain/spkrec-ecapa-voxceleb"),
            device=device,
        )

        return cls(
            vae=vae,
            floww2n_model=floww2n_model,
            content_encoder=content_encoder,
            speaker_encoder=speaker_encoder,
            device=device,
            num_steps=num_steps,
            compression_ratio=compression_ratio,
        )

    @torch.no_grad()
    def __call__(self, audio, sample_rate=16000, num_steps=None, seed=None,
                 speaker_audio=None):
        """Run whisper-to-normal conversion.

        Args:
            audio: Whisper audio waveform. Can be:
                - torch.Tensor of shape (samples,) or (batch, samples)
                - numpy array of shape (samples,) or (batch, samples)
            sample_rate: Sample rate of input audio (default: 16000)
            num_steps: Override default Euler steps (default: None = use self.num_steps)
            seed: Random seed for reproducibility (default: None)
            speaker_audio: Optional separate audio for speaker embedding.
                          If None, uses the whisper audio itself for speaker identity.

        Returns:
            audio_out: Converted normal speech (batch, samples) as torch.Tensor
        """
        steps = num_steps if num_steps is not None else self.num_steps

        # 1. Ensure torch tensor, handle batch dim
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        elif isinstance(audio, torch.Tensor):
            audio_tensor = audio.float()
        else:
            raise TypeError(f"Expected torch.Tensor or numpy array, got {type(audio)}")

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, samples)

        # 2. Resample if not 16kHz
        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            audio_tensor = resampler(audio_tensor)
            sample_rate = 16000

        audio_tensor = audio_tensor.to(self.device)

        # 3. Extract content features via ContentEncoder -> (B, 1500, 512)
        # ContentEncoder accepts numpy or tensor and handles conversion internally
        whisper_h, _mask = self.content_encoder(audio_tensor, sample_rate=sample_rate)
        # whisper_h: (B, 1500, 512)

        # 4. Extract speaker embedding via SpeakerEncoder -> (B, 192)
        if speaker_audio is not None:
            if isinstance(speaker_audio, np.ndarray):
                spk_tensor = torch.from_numpy(speaker_audio).float()
            elif isinstance(speaker_audio, torch.Tensor):
                spk_tensor = speaker_audio.float()
            else:
                raise TypeError(
                    f"Expected torch.Tensor or numpy array for speaker_audio, "
                    f"got {type(speaker_audio)}"
                )
            if spk_tensor.dim() == 1:
                spk_tensor = spk_tensor.unsqueeze(0)
            spk_tensor = spk_tensor.to(self.device)
            speaker_emb = self.speaker_encoder(spk_tensor)
        else:
            speaker_emb = self.speaker_encoder(audio_tensor)
        # speaker_emb: (B, 192)

        # 5. Compute latent length from audio length
        latent_length = self._compute_latent_length(audio_tensor.shape[-1])

        # 6. Sample z0 ~ N(0, I) with optional seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        batch_size = audio_tensor.shape[0]
        z0 = torch.randn(
            batch_size, 64, latent_length,
            device=self.device, generator=generator,
        )

        # 7. Euler integration via FlowW2NModel.sample()
        z1 = self.floww2n_model.sample(
            z0, whisper_h, speaker_emb, num_steps=steps
        )

        # 8. Decode via VAE decoder -> (B, 1, T')
        audio_out = self.vae.decode(z1)  # (B, 1, T')

        # 9. Return (B, T') squeezed
        audio_out = audio_out.squeeze(1)  # (B, T')

        return audio_out

    def _compute_latent_length(self, audio_length_samples):
        """Compute VAE latent sequence length from audio length.

        Args:
            audio_length_samples: Number of audio samples

        Returns:
            Latent sequence length (int)
        """
        return audio_length_samples // self.compression_ratio
