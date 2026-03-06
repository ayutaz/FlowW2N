"""Tests for VAE model."""

import pytest
import torch

from floww2n.models.vae import AudioAutoencoder


class TestAudioAutoencoder:
    @pytest.fixture
    def model(self):
        return AudioAutoencoder(
            io_channels=1,
            latent_dim=64,
            encoder_latent_dim=128,
            channels=128,
            c_mults=[1, 2, 4, 8],
            strides=[4, 4, 8, 8],
            use_snake=True,
        )

    def test_forward_shape(self, model):
        """Test forward pass returns correct shape and info dict."""
        x = torch.randn(2, 1, 65536)
        recon, info = model(x)
        assert recon.shape == x.shape
        assert "kl_loss" in info

    def test_encode_shape(self, model):
        """Test encoder output shape."""
        x = torch.randn(2, 1, 65536)
        z, info = model.encode(x)
        assert z.shape == (2, 64, 64)  # (B, D, T/1024)

    def test_decode_shape(self, model):
        """Test decoder output shape."""
        z = torch.randn(2, 64, 64)
        recon = model.decode(z)
        assert recon.shape == (2, 1, 65536)

    def test_compression_ratio(self, model):
        """Test compression ratio is 1024."""
        x = torch.randn(1, 1, 32768)
        z, _ = model.encode(x)
        assert z.shape[2] == 32768 // 1024  # = 32

    def test_kl_loss_positive(self, model):
        """Test KL loss is non-negative."""
        x = torch.randn(1, 1, 65536)
        _, info = model(x)
        assert info["kl_loss"].item() >= 0

    def test_batch_size_one(self, model):
        """Test model works with batch size 1."""
        x = torch.randn(1, 1, 65536)
        recon, info = model(x)
        assert recon.shape == (1, 1, 65536)
        assert "kl_loss" in info

    def test_different_lengths(self, model):
        """Test model handles different input lengths."""
        # Test shorter sequence
        x_short = torch.randn(1, 1, 16384)
        z_short, _ = model.encode(x_short)
        assert z_short.shape[2] == 16384 // 1024  # = 16

        # Test longer sequence
        x_long = torch.randn(1, 1, 131072)
        z_long, _ = model.encode(x_long)
        assert z_long.shape[2] == 131072 // 1024  # = 128

    def test_encode_decode_consistency(self, model):
        """Test encode-decode maintains expected dimensions."""
        x = torch.randn(2, 1, 65536)
        z, _ = model.encode(x)
        recon = model.decode(z)
        assert recon.shape == x.shape

    def test_gradient_flow(self, model):
        """Test gradients flow through the model."""
        x = torch.randn(1, 1, 65536, requires_grad=True)
        recon, info = model(x)
        loss = recon.mean() + info["kl_loss"]
        loss.backward()
        assert x.grad is not None
