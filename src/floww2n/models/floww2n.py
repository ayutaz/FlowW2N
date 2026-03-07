"""FlowW2N integrated model.

Top-level model that integrates DiffusionTransformer with conditioning modules
(speaker embedding projection, timestep embedding) and provides the CFM
(Conditional Flow Matching) training and sampling interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import DiffusionTransformer


class FlowW2NModel(nn.Module):
    """FlowW2N: Conditional Flow Matching for Whisper-to-Normal conversion.

    Integrates DiT with speaker conditioning and provides CFM training interface.

    The model uses:
    - Gaussian noise prior z0 ~ N(0, I) (NOT paired whisper latents)
    - Linear interpolation path: zt = (1-t)*z0 + t*z1
    - Constant velocity target: v = z1 - z0
    - MSE loss between predicted and target velocity

    Args:
        dit_config: Dict matching configs/dit.json["model"]. Keys:
            - io_channels (int): Latent dimension D (default: 64)
            - embed_dim (int): Transformer hidden dim (default: 768)
            - depth (int): Number of transformer blocks (default: 24)
            - num_heads (int): Number of attention heads (default: 12)
            - head_dim (int): Dimension per head (default: 64)
            - cond_token_dim (int): Cross-attention condition dim (default: 512)
            - global_cond_dim (int): Global condition dim for AdaLN (default: 768)
            - global_cond_type (str): Global conditioning type (default: "adaLN")
            - cross_attend (bool): Use cross-attention (default: True)
            - diffusion_objective (str): Objective type (default: "rectified_flow")
        speaker_dim: Dimension of input speaker embedding (default: 192)
    """

    def __init__(self, dit_config: dict, speaker_dim: int = 192):
        super().__init__()

        self.embed_dim = dit_config.get("embed_dim", 768)
        self.io_channels = dit_config.get("io_channels", 64)

        # Filter dit_config to only include keys accepted by DiffusionTransformer
        _dit_keys = {
            "io_channels",
            "embed_dim",
            "depth",
            "num_heads",
            "head_dim",
            "cond_token_dim",
            "global_cond_dim",
            "ff_mult",
            "dropout",
            "cross_attend",
            "num_fourier_features",
            "gradient_checkpointing",
        }
        filtered_config = {k: v for k, v in dit_config.items() if k in _dit_keys}

        # DiT backbone
        self.dit = DiffusionTransformer(**filtered_config)

        # Speaker embedding projection: 192d -> embed_dim
        # The projected embedding is passed as global_cond to DiT,
        # where it is combined with the timestep embedding via AdaLN.
        self.speaker_proj = nn.Linear(speaker_dim, self.embed_dim)

        # Language embedding (multilingual support)
        # When num_languages > 1, a learned embedding is added to global_cond.
        # When num_languages <= 1 or language_id is None, behaviour is unchanged.
        self.num_languages = dit_config.get("num_languages", 1)
        if self.num_languages > 1:
            self.language_emb = nn.Embedding(self.num_languages, self.embed_dim)
        else:
            self.language_emb = None

    def forward(self, z1, t, whisper_h, speaker_emb, language_id=None):
        """Predict velocity field v_theta(zt, t, c).

        Constructs zt from z0 ~ N(0,I) and z1 via linear interpolation,
        then passes through DiT to predict velocity.

        Note: In training, this method is typically called via compute_loss().
        For direct calls, zt should be pre-computed and passed as z1 argument
        (the name z1 is used for consistency with the training interface,
        but this method treats the first positional argument as the noisy input
        to the network).

        Actually, for the standard forward pass used during sampling,
        the first argument is zt (the current state along the ODE path).

        Args:
            z1: Input latent tensor (B, 64, T). During training this receives
                the interpolated zt; during sampling this is the current state.
            t: Timestep (B,) in [0, 1]
            whisper_h: Whisper content features (B, T_w, 512)
            speaker_emb: Speaker embedding (B, 192)
            language_id: Optional language index (B,) as LongTensor.
                Used only when num_languages > 1.

        Returns:
            v_pred: Predicted velocity (B, 64, T)
        """
        # Project speaker embedding to DiT embed_dim
        spk_proj = self.speaker_proj(speaker_emb)  # (B, embed_dim)

        # Build global condition: speaker + optional language
        global_cond = spk_proj
        if self.language_emb is not None and language_id is not None:
            global_cond = global_cond + self.language_emb(language_id)

        # DiT forward: timestep embedding is handled inside DiT.
        # global_cond (speaker projection + language) is added to timestep
        # embedding inside the DiT and injected via AdaLN.
        # cross_attn_cond (whisper features) is injected via cross-attention.
        v_pred = self.dit(
            z1,
            t,
            cross_attn_cond=whisper_h,
            global_cond=global_cond,
        )

        return v_pred

    def compute_loss(self, z1, whisper_h, speaker_emb, z1_mask=None, language_id=None):
        """Compute CFM training loss.

        Args:
            z1: Target VAE latent (B, 64, T) - clean normal speech latent
            whisper_h: Whisper content features (B, T_w, 512)
            speaker_emb: Speaker embedding (B, 192)
            z1_mask: Optional padding mask (B, T), True = valid frames
            language_id: Optional language index (B,) as LongTensor

        Returns:
            loss: Scalar MSE loss
            loss_dict: Dict with loss components for logging
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Sample Gaussian noise z0 ~ N(0, I)
        z0 = torch.randn_like(z1)

        # Sample timestep t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # Reshape t for broadcasting: (B,) -> (B, 1, 1) for (B, D, T)
        t_expanded = t.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

        # Linear interpolation: zt = (1-t)*z0 + t*z1
        zt = (1.0 - t_expanded) * z0 + t_expanded * z1

        # Target velocity (constant along straight-line path)
        v_target = z1 - z0

        # Predict velocity
        v_pred = self.forward(zt, t, whisper_h, speaker_emb, language_id=language_id)

        # MSE loss with optional masking
        if z1_mask is not None:
            # z1_mask: (B, T) -> (B, 1, T) for broadcasting with (B, D, T)
            mask_expanded = z1_mask.unsqueeze(1).float()
            diff_sq = (v_pred - v_target) ** 2
            # Mean over valid elements only
            loss = (diff_sq * mask_expanded).sum() / (mask_expanded.sum() * v_pred.shape[1])
        else:
            loss = F.mse_loss(v_pred, v_target)

        loss_dict = {
            "cfm_loss": loss.detach(),
        }

        return loss, loss_dict

    @torch.no_grad()
    def sample(self, z0, whisper_h, speaker_emb, num_steps=10, language_id=None, solver="euler"):
        """ODE integration sampling.

        Solves the ODE: dz/dt = v_theta(z, t, c) from t=0 to t=1.

        Args:
            z0: Initial noise (B, 64, T) ~ N(0, I)
            whisper_h: Whisper content features (B, T_w, 512)
            speaker_emb: Speaker embedding (B, 192)
            num_steps: Number of integration steps (default: 10)
            language_id: Optional language index (B,) as LongTensor
            solver: ODE solver to use, "euler" (1st order) or "heun" (2nd order).
                Heun achieves similar quality with fewer steps but costs 2 NFE/step.

        Returns:
            z1_pred: Predicted clean latent (B, 64, T)
        """
        # Lazy import to avoid circular dependency (sampler -> __init__ -> pipeline -> floww2n)
        from ..inference.sampler import euler_solve, heun_solve

        if solver == "heun":
            return heun_solve(
                self,
                z0,
                whisper_h,
                speaker_emb,
                num_steps=num_steps,
                language_id=language_id,
            )
        else:
            return euler_solve(
                self,
                z0,
                whisper_h,
                speaker_emb,
                num_steps=num_steps,
                language_id=language_id,
            )


if __name__ == "__main__":
    import json
    import os

    # Load DiT config
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "dit.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            full_config = json.load(f)
        dit_config = full_config["model"]
        print(f"Loaded config from {config_path}")
    else:
        # Fallback default config
        dit_config = {
            "io_channels": 64,
            "embed_dim": 768,
            "depth": 24,
            "num_heads": 12,
            "head_dim": 64,
            "cond_token_dim": 512,
            "global_cond_dim": 768,
            "global_cond_type": "adaLN",
            "cross_attend": True,
            "diffusion_objective": "rectified_flow",
        }
        print("Using default config")

    print(f"DiT config: {dit_config}")

    # Create model
    model = FlowW2NModel(dit_config=dit_config, speaker_dim=192)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test dimensions
    batch_size = 2
    latent_T = 64  # T depends on audio length; 64 frames ~ 4s at 15.6Hz
    whisper_T = 200  # ~4s at 50Hz

    z1 = torch.randn(batch_size, 64, latent_T)
    whisper_h = torch.randn(batch_size, whisper_T, 512)
    speaker_emb = torch.randn(batch_size, 192)

    # Test forward pass
    t = torch.rand(batch_size)
    v_pred = model(z1, t, whisper_h, speaker_emb)
    print("\n--- Forward pass ---")
    print(f"z1 shape: {z1.shape}")
    print(f"t shape: {t.shape}")
    print(f"whisper_h shape: {whisper_h.shape}")
    print(f"speaker_emb shape: {speaker_emb.shape}")
    print(f"v_pred shape: {v_pred.shape}")
    assert v_pred.shape == z1.shape, f"Shape mismatch: {v_pred.shape} != {z1.shape}"

    # Test compute_loss
    loss, loss_dict = model.compute_loss(z1, whisper_h, speaker_emb)
    print("\n--- Compute loss ---")
    print(f"CFM loss: {loss.item():.6f}")
    print(f"Loss dict: {loss_dict}")

    # Test sampling
    z0 = torch.randn(batch_size, 64, latent_T)
    z1_pred = model.sample(z0, whisper_h, speaker_emb, num_steps=10)
    print("\n--- Sampling (10 Euler steps) ---")
    print(f"z0 shape: {z0.shape}")
    print(f"z1_pred shape: {z1_pred.shape}")
    assert z1_pred.shape == z0.shape, f"Shape mismatch: {z1_pred.shape} != {z0.shape}"

    print("\nAll tests passed!")
