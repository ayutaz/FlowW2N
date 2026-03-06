"""Diffusion Transformer (DiT) with AdaLN and cross-attention.

Implements the flow matching backbone for FlowW2N, adapted from
stable-audio-tools DiffusionTransformer / ContinuousTransformer.

Architecture:
    - FourierFeatures for timestep embedding
    - AdaLN (Adaptive Layer Normalization) for timestep + speaker conditioning
    - Cross-attention for Whisper content feature injection
    - GEGLU feed-forward networks
    - Sinusoidal positional embeddings
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class FourierFeatures(nn.Module):
    """Map scalar timestep to Fourier feature embedding.

    Uses random (fixed) frequencies to create a rich representation of the
    scalar timestep t, then projects through an MLP to embed_dim.

    Args:
        num_frequencies: Number of Fourier frequency pairs (default: 256)
        embed_dim: Output embedding dimension (default: 768)
    """

    def __init__(self, num_frequencies: int = 256, embed_dim: int = 768):
        super().__init__()
        # Random frequencies, fixed after initialization
        self.register_buffer("frequencies", torch.randn(num_frequencies) * 0.2)
        fourier_dim = num_frequencies * 2

        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Fourier features from scalar timestep.

        Args:
            t: Timestep tensor of shape (B,)

        Returns:
            Embedding tensor of shape (B, embed_dim)
        """
        # t: (B,) -> (B, 1)
        t = t.unsqueeze(-1)
        # Fourier features: (B, num_frequencies)
        freqs = 2.0 * math.pi * self.frequencies * t
        # Concatenate sin and cos: (B, num_frequencies * 2)
        fourier = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return self.mlp(fourier)


class GEGLU(nn.Module):
    """GEGLU activation: x * GELU(gate), where [x, gate] = chunk(input, 2).

    Reference: Shazeer, "GLU Variants Improve Transformer" (2020).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation.

    Following stable-audio-tools convention: mult controls the total pre-split
    width, so effective hidden dim after GEGLU split is dim * mult / 2.

    Args:
        dim: Input/output dimension
        mult: Hidden dimension multiplier for pre-GEGLU width (default: 4)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # GEGLU splits the output in half, so total pre-split width = dim * mult
        # Effective hidden dim = dim * mult / 2
        pre_split_dim = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, pre_split_dim),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(pre_split_dim // 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention with optional cross-attention.

    Uses F.scaled_dot_product_attention for efficient computation.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads (default: 12)
        head_dim: Dimension per attention head (default: 64)
        dropout: Attention dropout rate (default: 0.0)
        is_cross_attention: Whether this is a cross-attention layer (default: False)
        context_dim: Dimension of context for cross-attention (default: None, uses dim)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        head_dim: int = 64,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.dropout = dropout
        self.is_cross_attention = is_cross_attention

        # Q always from input
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)

        # K, V from context if cross-attention, else from input
        kv_input_dim = context_dim if (is_cross_attention and context_dim is not None) else dim
        self.to_k = nn.Linear(kv_input_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(kv_input_dim, self.inner_dim, bias=False)

        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, T, dim)
            context: Context for cross-attention (B, T_ctx, context_dim), required if is_cross_attention
            pos_emb: Sinusoidal positional embedding to add to Q and K (B, T, head_dim) or None

        Returns:
            Output tensor (B, T, dim)
        """
        B, T, _ = x.shape

        q = self.to_q(x)

        if self.is_cross_attention and context is not None:
            k = self.to_k(context)
            v = self.to_v(context)
        else:
            k = self.to_k(x)
            v = self.to_v(x)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        T_kv = k.shape[1]
        k = k.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Add positional embedding to Q and K (for self-attention only)
        if pos_emb is not None and not self.is_cross_attention:
            # pos_emb: (1, T, head_dim) -> (1, 1, T, head_dim) for broadcasting
            pos_emb = pos_emb.unsqueeze(1)
            q = q + pos_emb[:, :, :T, :]
            k = k + pos_emb[:, :, :T_kv, :]

        # Scaled dot-product attention (uses flash attention when available)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Reshape back to (B, T, inner_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)

        return self.to_out(attn_out)


class TransformerBlock(nn.Module):
    """Single Transformer block with AdaLN, self-attention, cross-attention, and feed-forward.

    AdaLN modulation:
        - Self-attention: LayerNorm(x) * (1 + scale1) + shift1 -> attn -> * sigmoid(gate1) -> residual
        - Feed-forward:  LayerNorm(x) * (1 + scale2) + shift2 -> ff   -> * sigmoid(gate2) -> residual
        - Cross-attention has no AdaLN modulation

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        ff_mult: Feed-forward hidden dimension multiplier
        dropout: Dropout rate
        cross_attend: Whether to include cross-attention
        context_dim: Dimension of cross-attention context (after projection)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        head_dim: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        cross_attend: bool = True,
        context_dim: int | None = None,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # Cross-attention (optional)
        self.cross_attend = cross_attend
        if cross_attend:
            self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_attn = Attention(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                is_cross_attention=True,
                context_dim=context_dim if context_dim is not None else dim,
            )

        # Feed-forward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

        # AdaLN scale/shift/gate parameters (learnable per block)
        # 6 modulation vectors: (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_modulation = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor | None = None,
        cross_attn_cond: torch.Tensor | None = None,
        pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, T, dim)
            global_cond: AdaLN conditioning (B, 6*dim) from global_cond_embedder
            cross_attn_cond: Context for cross-attention (B, T_ctx, context_dim)
            pos_emb: Positional embedding for self-attention

        Returns:
            Output tensor (B, T, dim)
        """
        # Compute AdaLN modulation parameters
        if global_cond is not None:
            # global_cond: (B, 6*dim) -> 6 x (B, 1, dim)
            modulation = self.adaLN_modulation.unsqueeze(0) + global_cond.view(x.shape[0], 6, -1)
            scale1, shift1, gate1, scale2, shift2, gate2 = modulation.unbind(dim=1)
            # Add sequence dimension: (B, dim) -> (B, 1, dim)
            scale1 = scale1.unsqueeze(1)
            shift1 = shift1.unsqueeze(1)
            gate1 = gate1.unsqueeze(1)
            scale2 = scale2.unsqueeze(1)
            shift2 = shift2.unsqueeze(1)
            gate2 = gate2.unsqueeze(1)
        else:
            # No conditioning: use learned parameters only
            mods = self.adaLN_modulation.unsqueeze(0).unsqueeze(2)  # (1, 6, 1, dim)
            scale1 = mods[:, 0]
            shift1 = mods[:, 1]
            gate1 = mods[:, 2]
            scale2 = mods[:, 3]
            shift2 = mods[:, 4]
            gate2 = mods[:, 5]

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1.0 + scale1) + shift1
        h = self.self_attn(h, pos_emb=pos_emb)
        h = h * torch.sigmoid(gate1)
        x = x + h

        # Cross-attention (no AdaLN modulation)
        if self.cross_attend and cross_attn_cond is not None:
            h = self.norm_cross(x)
            h = self.cross_attn(h, context=cross_attn_cond)
            x = x + h

        # Feed-forward with AdaLN
        h = self.norm2(x)
        h = h * (1.0 + scale2) + shift2
        h = self.ff(h)
        h = h * torch.sigmoid(gate2)
        x = x + h

        return x


def sinusoidal_positional_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Generate sinusoidal positional embeddings.

    Args:
        length: Sequence length
        dim: Embedding dimension (per head)
        device: Torch device

    Returns:
        Positional embedding of shape (1, length, dim)
    """
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, length, dim)


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer for FlowW2N flow matching.

    Top-level model that maps noisy latent + timestep + conditions to velocity.

    Architecture:
        1. Conv1d input projection: (B, io_channels, T) -> (B, T, embed_dim)
        2. Timestep embedding via FourierFeatures
        3. Global conditioning (timestep + speaker) via AdaLN
        4. Content features (Whisper) via cross-attention
        5. N TransformerBlocks with sinusoidal positional embeddings
        6. Final LayerNorm + linear output projection

    Args:
        io_channels: Input/output channels (VAE latent dim, default: 64)
        embed_dim: Transformer hidden dimension (default: 768)
        depth: Number of transformer blocks (default: 24)
        num_heads: Number of attention heads (default: 12)
        head_dim: Dimension per attention head (default: 64)
        cond_token_dim: Dimension of cross-attention condition tokens (default: 512)
        global_cond_dim: Dimension of global conditioning input (default: 768)
        ff_mult: Feed-forward hidden dim multiplier (default: 4)
        dropout: Dropout rate (default: 0.0)
        cross_attend: Whether to use cross-attention (default: True)
        num_fourier_features: Number of Fourier frequency pairs for timestep (default: 256)
        gradient_checkpointing: Whether to use gradient checkpointing (default: False)
    """

    def __init__(
        self,
        io_channels: int = 64,
        embed_dim: int = 768,
        depth: int = 24,
        num_heads: int = 12,
        head_dim: int = 64,
        cond_token_dim: int = 512,
        global_cond_dim: int = 768,
        ff_mult: int = 4,
        dropout: float = 0.0,
        cross_attend: bool = True,
        num_fourier_features: int = 256,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.io_channels = io_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.head_dim = head_dim
        self.gradient_checkpointing = gradient_checkpointing

        # Input projection: channel-first Conv1d -> transpose to sequence-first
        self.input_proj = nn.Conv1d(io_channels, embed_dim, kernel_size=1)

        # Timestep embedding
        self.timestep_embed = FourierFeatures(
            num_frequencies=num_fourier_features,
            embed_dim=embed_dim,
        )

        # Global conditioning embedder (AdaLN)
        # Maps global_cond (embed_dim) -> 6 * embed_dim for AdaLN modulation
        self.global_cond_embedder = nn.Sequential(
            nn.Linear(global_cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )

        # Cross-attention condition projection (Whisper features)
        self.cross_attend = cross_attend
        if cross_attend:
            self.cond_token_proj = nn.Sequential(
                nn.Linear(cond_token_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    cross_attend=cross_attend,
                    context_dim=embed_dim,  # After projection
                )
                for _ in range(depth)
            ]
        )

        # Final norm and output projection
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, io_channels)

        # Initialize output projection to near-zero for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: predict velocity field v_theta(z_t, t, c).

        Args:
            x: Noisy latent tensor (B, C, T) where C=io_channels
            t: Timestep tensor (B,) with values in [0, 1]
            cross_attn_cond: Whisper content features (B, T_w, cond_token_dim), optional
            global_cond: Global conditioning embedding (B, global_cond_dim), optional.
                         Combined timestep+speaker embedding from FlowW2NModel.
                         If None, only timestep embedding is used.

        Returns:
            Predicted velocity (B, C, T) in latent space
        """
        B, C, T = x.shape

        # 1. Input projection: (B, C, T) -> (B, embed_dim, T) -> (B, T, embed_dim)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (B, T, embed_dim)

        # 2. Timestep embedding
        t_embed = self.timestep_embed(t)  # (B, embed_dim)

        # 3. Global conditioning: combine timestep + external global_cond
        if global_cond is not None:
            global_embed = t_embed + global_cond  # (B, embed_dim)
        else:
            global_embed = t_embed  # (B, embed_dim)

        # AdaLN modulation: (B, embed_dim) -> (B, 6*embed_dim)
        adaln_cond = self.global_cond_embedder(global_embed)

        # 4. Project cross-attention condition tokens
        cross_cond = None
        if self.cross_attend and cross_attn_cond is not None:
            cross_cond = self.cond_token_proj(cross_attn_cond)  # (B, T_w, embed_dim)

        # 5. Positional embedding
        pos_emb = sinusoidal_positional_embedding(T, self.head_dim, device=x.device)

        # 6. Transformer blocks
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = torch_checkpoint(
                    block,
                    x,
                    adaln_cond,
                    cross_cond,
                    pos_emb,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    global_cond=adaln_cond,
                    cross_attn_cond=cross_cond,
                    pos_emb=pos_emb,
                )

        # 7. Final norm and output projection
        x = self.final_norm(x)  # (B, T, embed_dim)
        x = self.output_proj(x)  # (B, T, io_channels)

        # 8. Transpose back to channel-first: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        return x


if __name__ == "__main__":
    # Test forward pass with dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model config matching configs/dit.json
    model = DiffusionTransformer(
        io_channels=64,
        embed_dim=768,
        depth=24,
        num_heads=12,
        head_dim=64,
        cond_token_dim=512,
        global_cond_dim=768,
        cross_attend=True,
    ).to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dummy inputs
    B = 2
    T_latent = 64  # VAE latent sequence length (e.g., 1s at 15.6Hz ~ 16 frames)
    T_whisper = 50  # Whisper frames (e.g., 1s at 50Hz)

    x = torch.randn(B, 64, T_latent, device=device)  # Noisy latent
    t = torch.rand(B, device=device)  # Timestep
    whisper_feat = torch.randn(B, T_whisper, 512, device=device)  # Whisper features
    speaker_embed = torch.randn(
        B, 768, device=device
    )  # Speaker embed (already projected to embed_dim)

    print("\nInput shapes:")
    print(f"  x (noisy latent): {x.shape}")
    print(f"  t (timestep):     {t.shape}")
    print(f"  whisper_feat:     {whisper_feat.shape}")
    print(f"  speaker_embed:    {speaker_embed.shape}")

    # Forward pass with all conditions
    with torch.no_grad():
        v = model(x, t, cross_attn_cond=whisper_feat, global_cond=speaker_embed)
    print(f"\nOutput shape (velocity): {v.shape}")
    assert v.shape == x.shape, f"Expected {x.shape}, got {v.shape}"
    print("Shape check passed.")

    # Forward pass without cross-attention condition
    with torch.no_grad():
        v_no_cross = model(x, t, global_cond=speaker_embed)
    print(f"Output (no cross-attn cond): {v_no_cross.shape}")
    assert v_no_cross.shape == x.shape
    print("No cross-attn shape check passed.")

    # Forward pass without any conditioning (timestep only)
    with torch.no_grad():
        v_uncond = model(x, t)
    print(f"Output (unconditional):      {v_uncond.shape}")
    assert v_uncond.shape == x.shape
    print("Unconditional shape check passed.")

    # Test with different sequence lengths
    T_long = 156  # ~10s at 15.6Hz
    T_w_long = 500  # ~10s at 50Hz
    x_long = torch.randn(1, 64, T_long, device=device)
    t_long = torch.rand(1, device=device)
    w_long = torch.randn(1, T_w_long, 512, device=device)
    spk_long = torch.randn(1, 768, device=device)

    with torch.no_grad():
        v_long = model(x_long, t_long, cross_attn_cond=w_long, global_cond=spk_long)
    print("\nLong sequence test:")
    print(f"  Input: {x_long.shape}, Output: {v_long.shape}")
    assert v_long.shape == x_long.shape
    print("Long sequence check passed.")

    print("\nAll tests passed.")
