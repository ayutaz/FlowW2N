"""Tests for DiT and FlowW2N model."""

import pytest
import torch

from floww2n.models.dit import DiffusionTransformer
from floww2n.models.floww2n import FlowW2NModel


# Small model config for fast testing
TEST_DIT_CONFIG = {
    "io_channels": 64,
    "embed_dim": 128,
    "depth": 2,
    "num_heads": 4,
    "head_dim": 32,
    "cond_token_dim": 512,
    "global_cond_dim": 128,
    "global_cond_type": "adaLN",
    "cross_attend": True,
    "diffusion_objective": "rectified_flow",
}

# Batch / sequence dims for tests
B = 2
T_LATENT = 16  # VAE latent frames
T_WHISPER = 50  # Whisper frames
IO_CHANNELS = 64
SPEAKER_DIM = 192


class TestDiffusionTransformer:
    @pytest.fixture
    def model(self):
        return DiffusionTransformer(
            io_channels=TEST_DIT_CONFIG["io_channels"],
            embed_dim=TEST_DIT_CONFIG["embed_dim"],
            depth=TEST_DIT_CONFIG["depth"],
            num_heads=TEST_DIT_CONFIG["num_heads"],
            head_dim=TEST_DIT_CONFIG["head_dim"],
            cond_token_dim=TEST_DIT_CONFIG["cond_token_dim"],
            global_cond_dim=TEST_DIT_CONFIG["global_cond_dim"],
            cross_attend=TEST_DIT_CONFIG["cross_attend"],
        )

    def test_dit_forward_shape(self, model):
        """DiffusionTransformer forward produces correct output shape (B, 64, T)."""
        x = torch.randn(B, IO_CHANNELS, T_LATENT)
        t = torch.rand(B)
        v = model(x, t)
        assert v.shape == (B, IO_CHANNELS, T_LATENT)

    def test_dit_with_cross_attention(self, model):
        """Forward with Whisper features via cross-attention works correctly."""
        x = torch.randn(B, IO_CHANNELS, T_LATENT)
        t = torch.rand(B)
        whisper_feat = torch.randn(B, T_WHISPER, 512)

        v = model(x, t, cross_attn_cond=whisper_feat)
        assert v.shape == (B, IO_CHANNELS, T_LATENT)

    def test_dit_with_global_cond(self, model):
        """Forward with speaker conditioning via AdaLN works correctly."""
        x = torch.randn(B, IO_CHANNELS, T_LATENT)
        t = torch.rand(B)
        # global_cond is already projected to global_cond_dim (= embed_dim)
        global_cond = torch.randn(B, TEST_DIT_CONFIG["global_cond_dim"])

        v = model(x, t, global_cond=global_cond)
        assert v.shape == (B, IO_CHANNELS, T_LATENT)

    def test_dit_different_lengths(self, model):
        """Model works with different sequence lengths."""
        for T in [8, 32, 64]:
            x = torch.randn(1, IO_CHANNELS, T)
            t = torch.rand(1)
            v = model(x, t)
            assert v.shape == (1, IO_CHANNELS, T)

    def test_dit_parameter_count(self):
        """Approximate parameter count check for full-size model (~170M for 768/24)."""
        full_model = DiffusionTransformer(
            io_channels=64,
            embed_dim=768,
            depth=24,
            num_heads=12,
            head_dim=64,
            cond_token_dim=512,
            global_cond_dim=768,
            cross_attend=True,
        )
        total_params = sum(p.numel() for p in full_model.parameters())
        # Expected ~170M parameters (embed_dim=768, depth=24)
        # Allow range 100M-250M for approximate check
        assert 100_000_000 < total_params < 250_000_000, (
            f"Expected ~170M parameters, got {total_params:,}"
        )


class TestFlowW2NModel:
    @pytest.fixture
    def model(self):
        return FlowW2NModel(
            dit_config=TEST_DIT_CONFIG,
            speaker_dim=SPEAKER_DIM,
        )

    def test_floww2n_forward_shape(self, model):
        """FlowW2NModel forward produces velocity of correct shape."""
        z1 = torch.randn(B, IO_CHANNELS, T_LATENT)
        t = torch.rand(B)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        v_pred = model(z1, t, whisper_h, speaker_emb)
        assert v_pred.shape == (B, IO_CHANNELS, T_LATENT)

    def test_floww2n_compute_loss(self, model):
        """compute_loss returns scalar loss > 0."""
        z1 = torch.randn(B, IO_CHANNELS, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        loss, loss_dict = model.compute_loss(z1, whisper_h, speaker_emb)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert "cfm_loss" in loss_dict

    def test_floww2n_sample(self, model):
        """sample method produces output of correct shape."""
        z0 = torch.randn(B, IO_CHANNELS, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        z1_pred = model.sample(z0, whisper_h, speaker_emb, num_steps=3)
        assert z1_pred.shape == (B, IO_CHANNELS, T_LATENT)

    def test_gradient_flow(self, model):
        """Gradients flow through the model during training."""
        z1 = torch.randn(B, IO_CHANNELS, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        loss, _ = model.compute_loss(z1, whisper_h, speaker_emb)
        loss.backward()

        # Check that gradients exist for key parameters
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in model parameters"
