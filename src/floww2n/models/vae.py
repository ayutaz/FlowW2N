"""Oobleck VAE (encoder-decoder) for waveform compression."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from alias_free_torch import Activation1d
from torch.nn.utils.parametrizations import weight_norm


def WNConv1d(*args, **kwargs):
    """Weight-normalized Conv1d."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight-normalized ConvTranspose1d."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake_beta(x, alpha, beta):
    """Snake activation with beta parameter."""
    sin_val = torch.sin(x * alpha)
    return x + (1.0 / (beta + 1e-9)) * (sin_val * sin_val)


class SnakeBeta(nn.Module):
    """Snake activation with learnable alpha and beta parameters.

    Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x


def get_activation(activation, antialias=False, channels=None):
    """Get activation function."""
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        if channels is None:
            raise ValueError("channels must be specified for snake activation")
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilation: Dilation rate for the first convolution
        use_snake: Whether to use Snake activation (otherwise ELU)
        antialias_activation: Whether to use anti-aliased activation
    """

    def __init__(
        self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False
    ):
        super().__init__()

        self.dilation = dilation

        padding = (dilation * (7 - 1)) // 2

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


class EncoderBlock(nn.Module):
    """Encoder block with residual units and downsampling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for downsampling
        use_snake: Whether to use Snake activation
        antialias_activation: Whether to use anti-aliased activation
    """

    def __init__(
        self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False
    ):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and residual units.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for upsampling
        use_snake: Whether to use Snake activation
        antialias_activation: Whether to use anti-aliased activation
        use_nearest_upsample: Whether to use nearest-neighbor upsampling
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
    ):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            upsample_layer,
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake
            ),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    """Oobleck encoder for audio compression.

    Args:
        in_channels: Number of input audio channels (default: 2)
        channels: Base number of channels (default: 128)
        latent_dim: Dimension of latent space output (default: 32)
        c_mults: Channel multipliers for each stage (default: [1, 2, 4, 8])
        strides: Stride for each downsampling stage (default: [2, 4, 8, 8])
        use_snake: Whether to use Snake activation (default: False)
        antialias_activation: Whether to use anti-aliased activation (default: False)
    """

    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()
        self.in_channels = in_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]

        for i in range(self.depth - 1):
            layers += [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1,
            ),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    """Oobleck decoder for audio reconstruction.

    Args:
        out_channels: Number of output audio channels (default: 2)
        channels: Base number of channels (default: 128)
        latent_dim: Dimension of latent space input (default: 32)
        c_mults: Channel multipliers for each stage (default: [1, 2, 4, 8])
        strides: Stride for each upsampling stage (default: [2, 4, 8, 8])
        use_snake: Whether to use Snake activation (default: False)
        antialias_activation: Whether to use anti-aliased activation (default: False)
        use_nearest_upsample: Whether to use nearest-neighbor upsampling (default: False)
        final_tanh: Whether to apply tanh to final output (default: True)
    """

    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            ),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def vae_sample(mean, scale):
    """Reparameterization trick for VAE sampling.

    Args:
        mean: Mean of the latent distribution
        scale: Scale (log-variance) of the latent distribution

    Returns:
        latents: Sampled latent variables
        kl: KL divergence loss
    """
    stdev = F.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = 0.5 * (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl


class VAEBottleneck(nn.Module):
    """VAE bottleneck with reparameterization trick.

    Splits input into mean and scale, samples latent using reparameterization,
    and computes KL divergence.
    """

    def __init__(self):
        super().__init__()

    def encode(self, x, return_info=False):
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (B, encoder_latent_dim, L)
            return_info: Whether to return info dict with KL loss

        Returns:
            latents: Sampled latent variables (B, latent_dim, L)
            info: Dict with 'kl' loss (if return_info=True)
        """
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        """Decode latent to output (identity for VAE)."""
        return x


class AudioAutoencoder(nn.Module):
    """Audio autoencoder with VAE bottleneck.

    Combines OobleckEncoder, VAEBottleneck, and OobleckDecoder for audio
    compression with variational inference.

    Args:
        io_channels: Number of audio channels (default: 1)
        latent_dim: Dimension of latent space (default: 64)
        encoder_latent_dim: Dimension of encoder output before bottleneck (default: 128)
        channels: Base number of channels (default: 128)
        c_mults: Channel multipliers (default: [1, 2, 4, 8])
        strides: Strides for each stage (default: [4, 4, 8, 8])
        use_snake: Whether to use Snake activation (default: True)
        antialias_activation: Whether to use anti-aliased activation (default: False)
        use_nearest_upsample: Whether to use nearest-neighbor upsampling (default: False)
        final_tanh: Whether to apply tanh to final output (default: True)
    """

    def __init__(
        self,
        io_channels=1,
        latent_dim=64,
        encoder_latent_dim=128,
        channels=128,
        c_mults=[1, 2, 4, 8],
        strides=[4, 4, 8, 8],
        use_snake=True,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()

        self.io_channels = io_channels
        self.latent_dim = latent_dim
        self.encoder_latent_dim = encoder_latent_dim

        # Encoder outputs encoder_latent_dim (e.g., 128) which is split into mean and scale
        self.encoder = OobleckEncoder(
            in_channels=io_channels,
            channels=channels,
            latent_dim=encoder_latent_dim,
            c_mults=c_mults,
            strides=strides,
            use_snake=use_snake,
            antialias_activation=antialias_activation,
        )

        # Bottleneck splits encoder output into mean/scale and samples
        self.bottleneck = VAEBottleneck()

        # Decoder takes latent_dim (e.g., 64) as input
        self.decoder = OobleckDecoder(
            out_channels=io_channels,
            channels=channels,
            latent_dim=latent_dim,
            c_mults=c_mults,
            strides=strides,
            use_snake=use_snake,
            antialias_activation=antialias_activation,
            use_nearest_upsample=use_nearest_upsample,
            final_tanh=final_tanh,
        )

    def encode(self, x):
        """Encode audio to latent space.

        Args:
            x: Input audio (B, io_channels, T)

        Returns:
            latents: Latent representation (B, latent_dim, L)
            info: Dict with 'kl_loss'
        """
        # Encoder: (B, io_channels, T) -> (B, encoder_latent_dim, L)
        h = self.encoder(x)

        # Bottleneck: (B, encoder_latent_dim, L) -> (B, latent_dim, L)
        z, info = self.bottleneck.encode(h, return_info=True)

        # Rename 'kl' to 'kl_loss' for consistency
        if "kl" in info:
            info["kl_loss"] = info.pop("kl")

        return z, info

    def decode(self, z):
        """Decode latent to audio.

        Args:
            z: Latent representation (B, latent_dim, L)

        Returns:
            Audio waveform (B, io_channels, T)
        """
        # Bottleneck decode (identity for VAE)
        z = self.bottleneck.decode(z)

        # Decoder: (B, latent_dim, L) -> (B, io_channels, T)
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through autoencoder.

        Args:
            x: Input audio (B, io_channels, T)

        Returns:
            reconstructed: Reconstructed audio (B, io_channels, T)
            info: Dict with 'kl_loss'
        """
        z, info = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, info

    def remove_weight_norm(self):
        """Remove weight normalization for faster inference."""
        import torch.nn.utils.parametrize as parametrize

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                try:
                    parametrize.remove_parametrizations(module, "weight")
                except ValueError:
                    pass  # No parametrization to remove
        return self


if __name__ == "__main__":
    # Test the model
    model = AudioAutoencoder(
        io_channels=1,
        latent_dim=64,
        encoder_latent_dim=128,
        channels=128,
        c_mults=[1, 2, 4, 8],
        strides=[4, 4, 8, 8],
        use_snake=True,
    )

    x = torch.randn(2, 1, 65536)
    recon, info = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    print(f"KL loss: {info['kl_loss'].item():.6f}")

    z, enc_info = model.encode(x)
    print(f"Latent shape: {z.shape}")

    recon2 = model.decode(z)
    print(f"Decoded shape: {recon2.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
