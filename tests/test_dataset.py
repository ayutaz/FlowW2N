"""Tests for dataset and collate functions."""

import torch

from floww2n.training.dataset import dit_collate_fn


class TestDitCollateFn:
    """Tests for DiT collate function with variable-length padding."""

    def _make_sample(self, z1_len, whisper_len):
        """Helper to create a sample dict."""
        return {
            "z1": torch.randn(64, z1_len),
            "whisper_h": torch.randn(whisper_len, 512),
            "speaker_emb": torch.randn(192),
        }

    def test_uniform_lengths(self):
        """All samples same length should produce no padding."""
        batch = [self._make_sample(32, 102) for _ in range(4)]
        result = dit_collate_fn(batch)
        assert result["z1"].shape == (4, 64, 32)
        assert result["whisper_h"].shape == (4, 102, 512)
        assert result["speaker_emb"].shape == (4, 192)
        # All masks should be True (no padding)
        assert result["z1_mask"].all()
        assert result["whisper_h_mask"].all()

    def test_variable_lengths_padding(self):
        """Variable lengths should be padded to max length."""
        batch = [
            self._make_sample(16, 51),
            self._make_sample(32, 102),
            self._make_sample(24, 77),
        ]
        result = dit_collate_fn(batch)
        assert result["z1"].shape == (3, 64, 32)  # padded to max
        assert result["whisper_h"].shape == (3, 102, 512)

    def test_mask_correctness(self):
        """Masks should be True for valid positions, False for padding."""
        batch = [
            self._make_sample(16, 51),
            self._make_sample(32, 102),
        ]
        result = dit_collate_fn(batch)

        # z1_mask
        assert result["z1_mask"][0, :16].all()  # first 16 are valid
        assert not result["z1_mask"][0, 16:].any()  # rest is padding
        assert result["z1_mask"][1, :32].all()  # all valid for longest

        # whisper_h_mask
        assert result["whisper_h_mask"][0, :51].all()
        assert not result["whisper_h_mask"][0, 51:].any()
        assert result["whisper_h_mask"][1, :102].all()

    def test_padding_values_are_zero(self):
        """Padded regions should be zero."""
        batch = [
            self._make_sample(8, 25),
            self._make_sample(16, 51),
        ]
        result = dit_collate_fn(batch)

        # Check z1 padding is zero
        assert (result["z1"][0, :, 8:] == 0).all()
        # Check whisper_h padding is zero
        assert (result["whisper_h"][0, 25:, :] == 0).all()

    def test_original_values_preserved(self):
        """Original (non-padded) values should be preserved."""
        sample = self._make_sample(16, 51)
        batch = [sample, self._make_sample(32, 102)]
        result = dit_collate_fn(batch)

        # First sample's values should match
        assert torch.allclose(result["z1"][0, :, :16], sample["z1"])
        assert torch.allclose(result["whisper_h"][0, :51, :], sample["whisper_h"])
        assert torch.allclose(result["speaker_emb"][0], sample["speaker_emb"])

    def test_single_sample_batch(self):
        """Single sample batch should work without errors."""
        batch = [self._make_sample(24, 77)]
        result = dit_collate_fn(batch)
        assert result["z1"].shape == (1, 64, 24)
        assert result["z1_mask"].all()

    def test_mask_dtype(self):
        """Masks should be boolean tensors."""
        batch = [self._make_sample(16, 51)]
        result = dit_collate_fn(batch)
        assert result["z1_mask"].dtype == torch.bool
        assert result["whisper_h_mask"].dtype == torch.bool

    def test_language_id_batching(self):
        """language_id should be batched as a LongTensor when present."""
        batch = []
        for lang_id in [0, 1, 0]:
            sample = self._make_sample(16, 51)
            sample["language_id"] = lang_id
            batch.append(sample)
        result = dit_collate_fn(batch)
        assert "language_id" in result
        assert result["language_id"].dtype == torch.long
        assert result["language_id"].tolist() == [0, 1, 0]

    def test_no_language_id_when_absent(self):
        """language_id should not be in result when not in samples."""
        batch = [self._make_sample(16, 51), self._make_sample(32, 102)]
        result = dit_collate_fn(batch)
        assert "language_id" not in result
