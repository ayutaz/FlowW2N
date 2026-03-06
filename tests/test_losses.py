"""Tests for VAE loss functions."""

import torch

from floww2n.training.losses import (
    MultiResolutionSTFTLoss,
    MultiScaleDiscriminator,
    VAELoss,
    feature_matching_loss,
    hinge_loss_discriminator,
    hinge_loss_generator,
)


class TestMultiResolutionSTFTLoss:
    """Tests for Multi-Resolution STFT Loss."""

    def test_output_scalar(self):
        """STFT loss should return a scalar."""
        loss_fn = MultiResolutionSTFTLoss()
        x = torch.randn(2, 1, 16000)
        y = torch.randn(2, 1, 16000)
        loss = loss_fn(x, y)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_identical_input_low_loss(self):
        """Identical inputs should have very low loss."""
        loss_fn = MultiResolutionSTFTLoss()
        x = torch.randn(2, 1, 16000)
        loss = loss_fn(x, x)
        assert loss.item() < 0.1

    def test_different_input_higher_loss(self):
        """Different inputs should have higher loss than identical ones."""
        loss_fn = MultiResolutionSTFTLoss()
        x = torch.randn(2, 1, 16000)
        y = torch.randn(2, 1, 16000)
        loss_diff = loss_fn(x, y)
        loss_same = loss_fn(x, x)
        assert loss_diff.item() > loss_same.item()


class TestMultiScaleDiscriminator:
    """Tests for Multi-Scale Discriminator."""

    def test_output_shape(self):
        """Discriminator should return scores and features per scale."""
        disc = MultiScaleDiscriminator(in_channels=1, n_scales=3)
        x = torch.randn(2, 1, 16000)
        scores, features = disc(x)
        assert len(scores) == 3  # 3 scales
        assert len(features) == 3
        for s in scores:
            assert s.dim() == 1  # (B,) after reshape+mean in SharedDiscriminatorConvNet

    def test_different_inputs_different_scores(self):
        """Real and fake should get different scores."""
        disc = MultiScaleDiscriminator(in_channels=1, n_scales=3)
        real = torch.randn(2, 1, 16000)
        fake = torch.randn(2, 1, 16000)
        real_scores, _ = disc(real)
        fake_scores, _ = disc(fake)
        # Scores should differ (not identical)
        for rs, fs in zip(real_scores, fake_scores):
            assert not torch.allclose(rs, fs)


class TestHingeLoss:
    """Tests for hinge loss functions."""

    def test_discriminator_loss_nonneg(self):
        """Discriminator hinge loss should be non-negative."""
        real_scores = [torch.randn(2) for _ in range(3)]
        fake_scores = [torch.randn(2) for _ in range(3)]
        loss = hinge_loss_discriminator(real_scores, fake_scores)
        assert loss.item() >= 0

    def test_generator_loss_finite(self):
        """Generator hinge loss should be finite."""
        fake_scores = [torch.randn(2) for _ in range(3)]
        loss = hinge_loss_generator(fake_scores)
        assert torch.isfinite(loss)


class TestFeatureMatchingLoss:
    """Tests for feature matching loss."""

    def test_identical_features_zero_loss(self):
        """Identical features should give zero loss."""
        features = [[torch.randn(2, 32, 100)] for _ in range(3)]
        # Detach to simulate real features
        features_detached = [[f.detach() for f in scale] for scale in features]
        loss = feature_matching_loss(features_detached, features)
        assert loss.item() < 1e-6

    def test_different_features_positive_loss(self):
        """Different features should give positive loss."""
        real_features = [[torch.randn(2, 32, 100)] for _ in range(3)]
        fake_features = [[torch.randn(2, 32, 100)] for _ in range(3)]
        loss = feature_matching_loss(real_features, fake_features)
        assert loss.item() > 0


class TestVAELoss:
    """Tests for combined VAE loss."""

    def test_generator_loss_output(self):
        """Generator loss should return total loss and dict."""
        criterion = VAELoss()
        real = torch.randn(2, 1, 16000)
        fake = torch.randn(2, 1, 16000)
        kl_loss = torch.tensor(1.0)
        total_loss, loss_dict = criterion.generator_loss(real, fake, kl_loss)
        assert total_loss.dim() == 0
        assert torch.isfinite(total_loss)
        assert "loss/generator" in loss_dict
        assert "loss/stft" in loss_dict
        assert "loss/adversarial" in loss_dict
        assert "loss/feature_matching" in loss_dict
        assert "loss/kl" in loss_dict

    def test_discriminator_loss_output(self):
        """Discriminator loss should return total loss and dict."""
        criterion = VAELoss()
        real = torch.randn(2, 1, 16000)
        fake = torch.randn(2, 1, 16000)
        total_loss, loss_dict = criterion.discriminator_loss(real, fake)
        assert total_loss.dim() == 0
        assert torch.isfinite(total_loss)
        assert "loss/discriminator" in loss_dict

    def test_gradient_flow_generator(self):
        """Gradients should flow through generator loss."""
        criterion = VAELoss()
        real = torch.randn(2, 1, 16000)
        fake = torch.randn(2, 1, 16000, requires_grad=True)
        kl_loss = torch.tensor(1.0, requires_grad=True)
        total_loss, _ = criterion.generator_loss(real, fake, kl_loss)
        total_loss.backward()
        assert fake.grad is not None
