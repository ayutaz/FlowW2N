"""Tests for training utilities and loop components."""

import torch


class TestGetLR:
    """Tests for learning rate scheduler."""

    def test_warmup_start(self):
        """LR should start from a small value at step 0."""
        from floww2n.training.train_dit import get_lr

        lr = get_lr(step=0, warmup_steps=2000, base_lr=1.5e-4)
        expected = 1.5e-4 * 1 / 2000
        assert abs(lr - expected) < 1e-10

    def test_warmup_end(self):
        """LR should reach base_lr at end of warmup."""
        from floww2n.training.train_dit import get_lr

        lr = get_lr(step=1999, warmup_steps=2000, base_lr=1.5e-4)
        expected = 1.5e-4 * 2000 / 2000
        assert abs(lr - expected) < 1e-10

    def test_after_warmup(self):
        """LR should be constant after warmup."""
        from floww2n.training.train_dit import get_lr

        lr = get_lr(step=5000, warmup_steps=2000, base_lr=1.5e-4)
        assert lr == 1.5e-4

    def test_warmup_monotonic(self):
        """LR should monotonically increase during warmup."""
        from floww2n.training.train_dit import get_lr

        lrs = [get_lr(step=i, warmup_steps=100, base_lr=1e-3) for i in range(100)]
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]


class TestCFMLossWithMask:
    """Tests for CFM loss with masking."""

    def _make_model(self):
        """Create a small FlowW2NModel for testing."""
        from floww2n.models.floww2n import FlowW2NModel

        config = {
            "io_channels": 64,
            "embed_dim": 128,
            "depth": 2,
            "num_heads": 4,
            "head_dim": 32,
            "cond_token_dim": 512,
            "global_cond_dim": 128,
            "cross_attend": True,
        }
        return FlowW2NModel(dit_config=config, speaker_dim=192)

    def test_loss_without_mask(self):
        """Loss should work without mask (backward compatibility)."""
        model = self._make_model()
        z1 = torch.randn(2, 64, 16)
        whisper_h = torch.randn(2, 51, 512)
        speaker_emb = torch.randn(2, 192)
        loss, loss_dict = model.compute_loss(z1, whisper_h, speaker_emb)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "cfm_loss" in loss_dict

    def test_loss_with_mask(self):
        """Loss should work with mask and produce finite values."""
        model = self._make_model()
        z1 = torch.randn(2, 64, 16)
        whisper_h = torch.randn(2, 51, 512)
        speaker_emb = torch.randn(2, 192)
        z1_mask = torch.ones(2, 16, dtype=torch.bool)
        z1_mask[0, 8:] = False  # Mask out second half for first sample
        loss, loss_dict = model.compute_loss(z1, whisper_h, speaker_emb, z1_mask=z1_mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_full_mask_equals_no_mask(self):
        """Full mask (all True) should give same result as no mask."""
        torch.manual_seed(42)
        model = self._make_model()
        z1 = torch.randn(2, 64, 16)
        whisper_h = torch.randn(2, 51, 512)
        speaker_emb = torch.randn(2, 192)

        # Loss without mask
        torch.manual_seed(123)
        loss_no_mask, _ = model.compute_loss(z1, whisper_h, speaker_emb)

        # Loss with all-True mask
        z1_mask = torch.ones(2, 16, dtype=torch.bool)
        torch.manual_seed(123)
        loss_with_mask, _ = model.compute_loss(z1, whisper_h, speaker_emb, z1_mask=z1_mask)

        assert torch.allclose(loss_no_mask, loss_with_mask, atol=1e-5)

    def test_gradient_flow_with_mask(self):
        """Gradients should flow with mask."""
        model = self._make_model()
        z1 = torch.randn(2, 64, 16)
        whisper_h = torch.randn(2, 51, 512)
        speaker_emb = torch.randn(2, 192)
        z1_mask = torch.ones(2, 16, dtype=torch.bool)
        z1_mask[0, 12:] = False
        loss, _ = model.compute_loss(z1, whisper_h, speaker_emb, z1_mask=z1_mask)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                break  # Just check first param


class TestCheckpointPruning:
    """Tests for checkpoint rotation (keep the newest ``max_keep`` by step number)."""

    def _touch(self, tmp_path, steps):
        for step in steps:
            (tmp_path / f"checkpoint_{step}.pt").write_bytes(b"")

    def test_sorted_by_step_not_name(self, tmp_path):
        from floww2n.training.checkpoint_utils import sorted_checkpoints

        self._touch(tmp_path, [95000, 100000, 5000, 20000])
        (tmp_path / "checkpoint_latest.pt").write_bytes(b"")  # non-numeric: ignored
        names = [p.name for p in sorted_checkpoints(tmp_path)]
        assert names == [
            "checkpoint_5000.pt",
            "checkpoint_20000.pt",
            "checkpoint_95000.pt",
            "checkpoint_100000.pt",
        ]

    def test_prune_keeps_newest_across_digit_boundary(self, tmp_path):
        from floww2n.training.checkpoint_utils import prune_checkpoints

        self._touch(tmp_path, [85000, 90000, 95000, 100000])
        prune_checkpoints(tmp_path, max_keep=3)
        remaining = sorted(p.name for p in tmp_path.glob("checkpoint_*.pt"))
        assert remaining == [
            "checkpoint_100000.pt",
            "checkpoint_90000.pt",
            "checkpoint_95000.pt",
        ]
