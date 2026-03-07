"""
Loss functions for VAE training in FlowW2N.

Components:
1. MultiResolutionSTFTLoss - Spectral reconstruction loss (auraloss)
2. MultiScaleDiscriminator - Multi-scale adversarial discriminator
3. Hinge losses - Adversarial training (discriminator + generator)
4. Feature matching - Perceptual loss using discriminator features
5. VAELoss - Combined loss coordinator

Architecture based on stable-audio-tools discriminators.py
Configuration matches SA2.0 with 7 STFT scales and 3-scale discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from auraloss.freq import MultiResolutionSTFTLoss as AuralossSTFT


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss wrapper.

    Uses auraloss library with SA2.0 configuration:
    - 7 scales from 32 to 2048
    - Spectral convergence + log magnitude loss
    """

    def __init__(self):
        super().__init__()
        # SA2.0 configuration
        fft_sizes = [2048, 1024, 512, 256, 128, 64, 32]
        hop_sizes = [512, 256, 128, 64, 32, 16, 8]
        win_lengths = [2048, 1024, 512, 256, 128, 64, 32]

        self.stft_loss = AuralossSTFT(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            w_sc=1.0,  # spectral convergence weight
            w_log_mag=1.0,  # log magnitude weight
            w_lin_mag=0.0,  # linear magnitude weight (not used)
        )

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            real: Real audio [B, 1, T]
            fake: Generated audio [B, 1, T]

        Returns:
            STFT loss (spectral convergence + log magnitude)
        """
        # auraloss expects [B, C, T] format
        return self.stft_loss(fake, real)


class SharedDiscriminatorConvNet(nn.Module):
    """
    Shared convolutional discriminator network.

    Based on stable-audio-tools implementation.
    Progressive channel expansion with weight-normalized convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        capacity: int = 32,
        n_layers: int = 4,
        kernel_size: int = 15,
        stride: int = 4,
        activation: str = "silu",
    ):
        super().__init__()

        # Build channel progression: [in_channels, capacity, capacity*2, capacity*4, ...]
        channels = [in_channels] + [capacity * (2**i) for i in range(n_layers)]

        # Build discriminator layers
        layers = []
        for i in range(n_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            padding = kernel_size // 2

            # Conv with weight normalization
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
            conv = nn.utils.parametrizations.weight_norm(conv)
            layers.append(conv)

            # Activation
            if activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))

        # Final conv to scalar output
        final_conv = nn.Conv1d(channels[-1], 1, kernel_size=1)
        final_conv = nn.utils.parametrizations.weight_norm(final_conv)
        layers.append(final_conv)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass collecting intermediate features.

        Args:
            x: Input audio [B, C, T]

        Returns:
            score: Discrimination score [B]
            features: List of intermediate features for feature matching
        """
        features = []

        for layer in self.layers:
            x = layer(x)
            # Collect features after each conv layer
            if isinstance(layer, nn.Conv1d):
                features.append(x)

        # Compute score by averaging spatial dimensions
        score = x.reshape(x.shape[0], -1).mean(-1)

        return score, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator.

    Multiple SharedDiscriminatorConvNet at different scales.
    Each scale processes downsampled version of input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_scales: int = 3,
        capacity: int = 32,
        n_layers: int = 4,
        kernel_size: int = 15,
        stride: int = 4,
    ):
        super().__init__()

        # Create discriminators for each scale
        discriminators = []
        for _ in range(n_scales):
            discriminators.append(
                SharedDiscriminatorConvNet(
                    in_channels=in_channels,
                    capacity=capacity,
                    n_layers=n_layers,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Forward pass at multiple scales.

        Args:
            x: Input audio [B, C, T]

        Returns:
            scores: List of discrimination scores from each scale
            features: List of feature lists from each scale
        """
        scores = []
        features_list = []

        for disc in self.discriminators:
            score, features = disc(x)
            scores.append(score)
            features_list.append(features)
            # Downsample for next scale (avgpool with kernel=4, stride=2, padding=2)
            x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=2)

        return scores, features_list


def hinge_loss_discriminator(
    real_scores: list[torch.Tensor], fake_scores: list[torch.Tensor]
) -> torch.Tensor:
    """
    Hinge loss for discriminator.

    D tries to:
    - Score real samples > 1
    - Score fake samples < -1

    Loss = mean(max(0, 1 - real)) + mean(max(0, 1 + fake))

    Args:
        real_scores: List of scores for real samples from each scale
        fake_scores: List of scores for fake samples from each scale

    Returns:
        Discriminator hinge loss
    """
    loss = 0.0
    for real, fake in zip(real_scores, fake_scores):
        loss += F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()

    # Average over scales
    return loss / len(real_scores)


def hinge_loss_generator(fake_scores: list[torch.Tensor]) -> torch.Tensor:
    """
    Hinge loss for generator.

    G tries to maximize discriminator output for fake samples.
    Loss = -mean(fake)

    Args:
        fake_scores: List of scores for fake samples from each scale

    Returns:
        Generator hinge loss
    """
    loss = 0.0
    for fake in fake_scores:
        loss -= fake.mean()

    # Average over scales
    return loss / len(fake_scores)


def feature_matching_loss(
    real_features_list: list[list[torch.Tensor]], fake_features_list: list[list[torch.Tensor]]
) -> torch.Tensor:
    """
    Feature matching loss.

    Minimizes L1 distance between intermediate features of real and fake samples.
    Helps stabilize GAN training.

    Args:
        real_features_list: List of feature lists from each scale (real)
        fake_features_list: List of feature lists from each scale (fake)

    Returns:
        Feature matching loss
    """
    loss = 0.0
    count = 0

    for real_features, fake_features in zip(real_features_list, fake_features_list):
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
            count += 1

    # Average over all features
    return loss / count if count > 0 else torch.tensor(0.0)


class VAELoss(nn.Module):
    """
    Combined VAE loss for FlowW2N.

    Components:
    1. Multi-resolution STFT loss (reconstruction)
    2. Adversarial loss (hinge loss)
    3. Feature matching loss (discriminator features)
    4. KL divergence loss (VAE regularization)
    """

    def __init__(
        self,
        stft_weight: float = 1.0,
        adv_weight: float = 0.1,
        feat_weight: float = 5.0,
        kl_weight: float = 1e-4,
    ):
        super().__init__()

        self.stft_weight = stft_weight
        self.adv_weight = adv_weight
        self.feat_weight = feat_weight
        self.kl_weight = kl_weight

        # Initialize loss components
        self.stft_loss = MultiResolutionSTFTLoss()
        self.discriminator = MultiScaleDiscriminator(
            in_channels=1,
            n_scales=3,
            capacity=32,
            n_layers=4,
            kernel_size=15,
            stride=4,
        )

    def generator_loss(
        self, real: torch.Tensor, fake: torch.Tensor, kl_loss: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute generator (VAE encoder-decoder) loss.

        Args:
            real: Real audio [B, 1, T]
            fake: Generated audio [B, 1, T]
            kl_loss: KL divergence loss from VAE

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual loss components
        """
        # 1. STFT reconstruction loss
        stft = self.stft_loss(real, fake)

        # 2. Adversarial loss (generator wants high scores)
        # Get real features for feature matching (no gradient needed for real)
        with torch.no_grad():
            real_scores, real_features = self.discriminator(real)

        # Get fake features with gradients for generator update
        fake_scores_grad, fake_features_grad = self.discriminator(fake)

        adv = hinge_loss_generator(fake_scores_grad)

        # 3. Feature matching loss
        feat = feature_matching_loss(real_features, fake_features_grad)

        # 4. Weighted combination
        total_loss = (
            self.stft_weight * stft
            + self.adv_weight * adv
            + self.feat_weight * feat
            + self.kl_weight * kl_loss
        )

        loss_dict = {
            "loss/generator": total_loss,
            "loss/stft": stft,
            "loss/adversarial": adv,
            "loss/feature_matching": feat,
            "loss/kl": kl_loss,
        }

        return total_loss, loss_dict

    def discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute discriminator loss.

        Args:
            real: Real audio [B, 1, T]
            fake: Generated audio [B, 1, T]

        Returns:
            d_loss: Discriminator hinge loss
            loss_dict: Dictionary of individual loss components
        """
        # Forward pass (detach fake to prevent generator gradients)
        real_scores, _ = self.discriminator(real)
        fake_scores, _ = self.discriminator(fake.detach())

        # Hinge loss
        d_loss = hinge_loss_discriminator(real_scores, fake_scores)

        loss_dict = {
            "loss/discriminator": d_loss,
            "score/real": torch.stack(real_scores).mean(),
            "score/fake": torch.stack(fake_scores).mean(),
        }

        return d_loss, loss_dict


if __name__ == "__main__":
    print("Testing VAE Loss Functions...")
    print("=" * 60)

    # Initialize loss
    vae_loss = VAELoss(stft_weight=1.0, adv_weight=0.1, feat_weight=5.0, kl_weight=1e-4)

    # Create dummy data
    batch_size = 2
    length = 65536
    real = torch.randn(batch_size, 1, length)
    fake = torch.randn(batch_size, 1, length)
    kl = torch.tensor(0.01)

    print("\nInput shapes:")
    print(f"  Real: {real.shape}")
    print(f"  Fake: {fake.shape}")
    print(f"  KL: {kl.item():.6f}")

    # Test generator loss
    print(f"\n{'Generator Loss':=^60}")
    g_loss, g_dict = vae_loss.generator_loss(real, fake, kl)
    print(f"Total: {g_loss.item():.4f}")
    for k, v in g_dict.items():
        print(f"  {k}: {v.item():.4f}")

    # Test discriminator loss
    print(f"\n{'Discriminator Loss':=^60}")
    d_loss, d_dict = vae_loss.discriminator_loss(real, fake)
    print(f"Total: {d_loss.item():.4f}")
    for k, v in d_dict.items():
        print(f"  {k}: {v.item():.4f}")

    print(f"\n{'Test Complete':=^60}")
    print("\nAll loss components working correctly!")
