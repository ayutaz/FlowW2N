"""Tests for synthetic whisper generation."""

import numpy as np

from floww2n.data.whisper_synthesis import (
    WhisperSynthesizer,
    formant_bandwidth_mod,
    glottal_source_removal,
    lpc_devoice,
)


class TestLPCDevoice:
    """Tests for LPC-based devoicing."""

    def test_output_length_matches_input(self):
        """Output length should match input length."""
        audio = np.random.randn(16000).astype(np.float32)
        result = lpc_devoice(audio, sr=16000)
        assert len(result) == len(audio)

    def test_output_is_finite(self):
        """Output should not contain NaN or Inf."""
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = lpc_devoice(audio, sr=16000)
        assert np.all(np.isfinite(result))

    def test_silent_input(self):
        """Silent input should not crash."""
        audio = np.zeros(16000, dtype=np.float32)
        result = lpc_devoice(audio, sr=16000)
        assert len(result) == len(audio)
        assert np.all(np.isfinite(result))


class TestGlottalSourceRemoval:
    """Tests for glottal source removal."""

    def test_output_length_matches_input(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = glottal_source_removal(audio, sr=16000)
        assert len(result) == len(audio)

    def test_output_is_finite(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = glottal_source_removal(audio, sr=16000)
        assert np.all(np.isfinite(result))


class TestFormantBandwidthMod:
    """Tests for formant bandwidth modification."""

    def test_output_length_matches_input(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = formant_bandwidth_mod(audio, sr=16000)
        assert len(result) == len(audio)

    def test_output_is_finite(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = formant_bandwidth_mod(audio, sr=16000)
        assert np.all(np.isfinite(result))


class TestWhisperSynthesizer:
    """Tests for WhisperSynthesizer wrapper."""

    def test_random_method_selection(self):
        """Synthesizer should work with random method selection."""
        synth = WhisperSynthesizer(sr=16000)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = synth(audio)
        assert len(result) == len(audio)
        assert np.all(np.isfinite(result))

    def test_specific_method_index(self):
        """Synthesizer should work with specific method index."""
        synth = WhisperSynthesizer(sr=16000)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        for idx in range(3):  # Test first 3 methods (skip praat which may not be available)
            result = synth(audio, method_index=idx)
            assert len(result) == len(audio)

    def test_output_different_from_input(self):
        """Output should generally differ from input."""
        synth = WhisperSynthesizer(sr=16000)
        audio = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000).astype(np.float32)
        result = synth(audio, method_index=0)
        # Should be different (devoiced)
        assert not np.allclose(result, audio, atol=0.1)

    def test_short_audio(self):
        """Short audio should not crash."""
        synth = WhisperSynthesizer(sr=16000)
        audio = np.random.randn(1600).astype(np.float32) * 0.5  # 0.1 sec
        result = synth(audio, method_index=1)
        assert len(result) == len(audio)
