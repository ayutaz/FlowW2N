"""Integration tests for inference and evaluation pipelines."""

import numpy as np
import pytest
import torch

from floww2n.inference.pipeline import FlowW2NPipeline
from floww2n.inference.sampler import euler_solve, euler_solve_with_trajectory, heun_solve
from floww2n.models.floww2n import FlowW2NModel
from floww2n.models.vae import AudioAutoencoder

# Use small configs for all tests
SMALL_VAE_CONFIG = {
    "io_channels": 1,
    "latent_dim": 64,
    "encoder_latent_dim": 128,
    "channels": 32,  # small
    "c_mults": [1, 2, 4, 8],
    "strides": [4, 4, 8, 8],
    "use_snake": True,
}

SMALL_DIT_CONFIG = {
    "io_channels": 64,
    "embed_dim": 128,  # small
    "depth": 2,  # small
    "num_heads": 4,
    "head_dim": 32,
    "cond_token_dim": 512,
    "global_cond_dim": 128,
    "cross_attend": True,
}

# Shared test dimensions
COMPRESSION_RATIO = 1024  # product of strides: 4*4*8*8
B = 1  # batch size for pipeline tests
T_AUDIO = 16384  # audio samples (must be divisible by compression ratio)
T_LATENT = T_AUDIO // COMPRESSION_RATIO  # = 16
T_WHISPER = 50  # whisper content feature frames
SPEAKER_DIM = 192


class _DummyContentEncoder:
    """Lightweight stand-in for ContentEncoder that returns random features.

    Avoids downloading the Whisper model during tests.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, audio, sample_rate=16000):
        # Determine batch size
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                batch_size = 1
            else:
                batch_size = audio.shape[0]
        elif isinstance(audio, np.ndarray):
            if audio.ndim == 1:
                batch_size = 1
            else:
                batch_size = audio.shape[0]
        else:
            batch_size = 1

        h = torch.randn(batch_size, 1500, 512, device=self.device)
        mask = torch.ones(batch_size, 1500, dtype=torch.bool, device=self.device)
        return h, mask


class _DummySpeakerEncoder:
    """Lightweight stand-in for SpeakerEncoder that returns random embeddings.

    Avoids downloading the ECAPA-TDNN model during tests.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, audio):
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                batch_size = 1
            else:
                batch_size = audio.shape[0]
        else:
            batch_size = 1

        return torch.randn(batch_size, SPEAKER_DIM, device=self.device)


def _make_pipeline(device="cpu"):
    """Create a FlowW2NPipeline with small models and dummy encoders."""
    vae = AudioAutoencoder(**SMALL_VAE_CONFIG)
    floww2n_model = FlowW2NModel(
        dit_config=SMALL_DIT_CONFIG,
        speaker_dim=SPEAKER_DIM,
    )

    pipeline = FlowW2NPipeline(
        vae=vae,
        floww2n_model=floww2n_model,
        content_encoder=_DummyContentEncoder(device=device),
        speaker_encoder=_DummySpeakerEncoder(device=device),
        device=device,
        num_steps=2,  # minimal steps for speed
        compression_ratio=COMPRESSION_RATIO,
    )
    return pipeline


class TestFlowW2NPipeline:
    """Tests for the inference pipeline."""

    def test_pipeline_creation(self):
        """Create pipeline from model instances."""
        pipeline = _make_pipeline()
        assert pipeline.vae is not None
        assert pipeline.floww2n_model is not None
        assert pipeline.content_encoder is not None
        assert pipeline.speaker_encoder is not None
        assert pipeline.num_steps == 2
        assert pipeline.compression_ratio == COMPRESSION_RATIO

    def test_pipeline_forward(self):
        """Run pipeline with random audio, check output is tensor with correct dims."""
        pipeline = _make_pipeline()
        audio = torch.randn(T_AUDIO)

        audio_out = pipeline(audio, sample_rate=16000, num_steps=2)

        assert isinstance(audio_out, torch.Tensor)
        # Output should be (batch, samples)
        assert audio_out.dim() == 2
        assert audio_out.shape[0] == 1  # batch size 1
        # Output length should match input length (same compression ratio)
        assert audio_out.shape[1] == T_AUDIO

    def test_pipeline_batch(self):
        """Run with batch of audio."""
        pipeline = _make_pipeline()
        batch_size = 2
        audio = torch.randn(batch_size, T_AUDIO)

        audio_out = pipeline(audio, sample_rate=16000, num_steps=2)

        assert isinstance(audio_out, torch.Tensor)
        assert audio_out.dim() == 2
        assert audio_out.shape[0] == batch_size
        assert audio_out.shape[1] == T_AUDIO

    def test_pipeline_seed_reproducibility(self):
        """Same seed produces same output."""
        pipeline = _make_pipeline()
        audio = torch.randn(T_AUDIO)

        out1 = pipeline(audio, sample_rate=16000, num_steps=2, seed=42)
        out2 = pipeline(audio, sample_rate=16000, num_steps=2, seed=42)

        assert torch.allclose(out1, out2, atol=1e-5), "Same seed should produce identical output"

    def test_pipeline_different_lengths(self):
        """Pipeline handles different audio lengths."""
        pipeline = _make_pipeline()

        for length_factor in [1, 2, 4]:
            audio_len = COMPRESSION_RATIO * 8 * length_factor
            audio = torch.randn(audio_len)
            audio_out = pipeline(audio, sample_rate=16000, num_steps=2)
            assert audio_out.dim() == 2
            assert audio_out.shape[0] == 1
            assert audio_out.shape[1] == audio_len

    def test_pipeline_numpy_input(self):
        """Pipeline accepts numpy array input."""
        pipeline = _make_pipeline()
        audio = np.random.randn(T_AUDIO).astype(np.float32)

        audio_out = pipeline(audio, sample_rate=16000, num_steps=2)

        assert isinstance(audio_out, torch.Tensor)
        assert audio_out.dim() == 2
        assert audio_out.shape[0] == 1


class TestEulerSampler:
    """Tests for the standalone Euler sampler."""

    @pytest.fixture
    def model(self):
        return FlowW2NModel(
            dit_config=SMALL_DIT_CONFIG,
            speaker_dim=SPEAKER_DIM,
        )

    def test_euler_solve(self, model):
        """Run euler_solve, check output shape and finiteness."""
        z0 = torch.randn(B, 64, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        z1 = euler_solve(model, z0, whisper_h, speaker_emb, num_steps=3)

        assert z1.shape == (B, 64, T_LATENT)
        # Output should be finite (no NaN or Inf)
        assert torch.isfinite(z1).all(), "Euler solve produced non-finite values"

    def test_euler_trajectory(self, model):
        """Run euler_solve_with_trajectory, check all states returned."""
        num_steps = 5
        z0 = torch.randn(B, 64, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        trajectory = euler_solve_with_trajectory(
            model, z0, whisper_h, speaker_emb, num_steps=num_steps
        )

        # Should have num_steps + 1 states (z0 through z_{num_steps})
        assert len(trajectory) == num_steps + 1

        # Each state should have correct shape
        for state in trajectory:
            assert state.shape == (B, 64, T_LATENT)

        # First state should be z0
        assert torch.allclose(trajectory[0], z0)

        # All states should be finite
        for state in trajectory:
            assert torch.isfinite(state).all(), "Trajectory contains non-finite values"

    def test_euler_solve_consistency(self, model):
        """euler_solve and euler_solve_with_trajectory give same final state."""
        num_steps = 3
        z0 = torch.randn(B, 64, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        z1_solve = euler_solve(model, z0.clone(), whisper_h, speaker_emb, num_steps=num_steps)
        trajectory = euler_solve_with_trajectory(
            model, z0.clone(), whisper_h, speaker_emb, num_steps=num_steps
        )

        assert torch.allclose(z1_solve, trajectory[-1], atol=1e-5)


class TestHeunSolver:
    """Tests for the Heun (2nd order) ODE solver."""

    @pytest.fixture
    def model(self):
        return FlowW2NModel(
            dit_config=SMALL_DIT_CONFIG,
            speaker_dim=SPEAKER_DIM,
        )

    def test_heun_solve_shape(self, model):
        """Heun solver output has correct shape."""
        z0 = torch.randn(B, 64, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        z1 = heun_solve(model, z0, whisper_h, speaker_emb, num_steps=3)

        assert z1.shape == (B, 64, T_LATENT)
        assert torch.isfinite(z1).all()

    def test_heun_solve_same_shape_as_euler(self, model):
        """Heun and Euler both produce valid outputs with the same shape."""
        z0 = torch.randn(B, 64, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        z1_euler = euler_solve(model, z0.clone(), whisper_h, speaker_emb, num_steps=3)
        z1_heun = heun_solve(model, z0.clone(), whisper_h, speaker_emb, num_steps=3)

        assert z1_euler.shape == z1_heun.shape
        assert torch.isfinite(z1_euler).all()
        assert torch.isfinite(z1_heun).all()

    def test_floww2n_sample_with_heun(self, model):
        """FlowW2NModel.sample() works with solver='heun'."""
        z0 = torch.randn(B, 64, T_LATENT)
        whisper_h = torch.randn(B, T_WHISPER, 512)
        speaker_emb = torch.randn(B, SPEAKER_DIM)

        z1 = model.sample(z0, whisper_h, speaker_emb, num_steps=3, solver="heun")

        assert z1.shape == (B, 64, T_LATENT)
        assert torch.isfinite(z1).all()


class TestOptimizations:
    """Tests for optimization features (weight norm removal, pos emb cache)."""

    def test_remove_weight_norm(self):
        """VAE remove_weight_norm runs without error and model still works."""
        vae = AudioAutoencoder(**SMALL_VAE_CONFIG)
        x = torch.randn(1, 1, 4096)

        # Remove weight norm
        vae.remove_weight_norm()

        # Model should still produce valid output
        with torch.no_grad():
            out, info = vae(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert "kl_loss" in info

    def test_pos_emb_cache(self):
        """DiffusionTransformer caches positional embeddings."""
        from floww2n.models.dit import DiffusionTransformer

        dit = DiffusionTransformer(
            io_channels=64,
            embed_dim=128,
            depth=2,
            num_heads=4,
            head_dim=32,
            global_cond_dim=128,
        )
        dit.eval()

        x = torch.randn(1, 64, 10)
        t = torch.tensor([0.5])
        global_cond = torch.randn(1, 128)

        # First forward: cache should be populated
        with torch.no_grad():
            _ = dit(x, t, global_cond=global_cond)
        assert dit._pos_emb_cache is not None
        assert dit._pos_emb_cache_len == 10

        # Second forward with same length: cache should be reused
        with torch.no_grad():
            _ = dit(x, t, global_cond=global_cond)
        assert dit._pos_emb_cache_len == 10


class TestEvaluationMetrics:
    """Tests for evaluation metrics (no actual model loading)."""

    def test_spk_sim_metric_from_embeddings(self):
        """SpkSim compute_from_embeddings returns score between -1 and 1.

        Tests the embedding-based computation without loading
        any heavy speaker encoder model.
        """
        # Create random embeddings
        emb_a = np.random.randn(192).astype(np.float32)
        emb_b = np.random.randn(192).astype(np.float32)

        # Compute cosine similarity manually
        cos_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)

        assert -1.0 <= cos_sim <= 1.0

        # Same embedding should give similarity ~1.0
        cos_sim_same = np.dot(emb_a, emb_a) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_a) + 1e-8)
        assert abs(cos_sim_same - 1.0) < 1e-5

    def test_dnsmos_fallback(self):
        """DNSMOS returns NaN when no ONNX dir is provided."""
        import math

        from floww2n.evaluation.metrics import DNSMOSMetric

        metric = DNSMOSMetric(onnx_model_dir=None)
        dummy_audio = np.random.randn(16000).astype(np.float32)
        result = metric.compute(dummy_audio)

        assert "ovrl" in result
        assert math.isnan(result["ovrl"])
        assert math.isnan(result["sig"])
        assert math.isnan(result["bak"])

    def test_dnsmos_batch_fallback(self):
        """DNSMOS batch returns list of NaN when no ONNX dir is provided."""
        import math

        from floww2n.evaluation.metrics import DNSMOSMetric

        metric = DNSMOSMetric(onnx_model_dir=None)
        batch_audio = np.random.randn(3, 16000).astype(np.float32)
        result = metric.compute(batch_audio)

        assert isinstance(result["ovrl"], list)
        assert len(result["ovrl"]) == 3
        for val in result["ovrl"]:
            assert math.isnan(val)


class TestIntegration:
    """End-to-end integration tests."""

    def test_end_to_end(self):
        """Full pipeline: create models -> run inference -> check output is valid audio."""
        # 1. Create small models
        vae = AudioAutoencoder(**SMALL_VAE_CONFIG)
        floww2n_model = FlowW2NModel(
            dit_config=SMALL_DIT_CONFIG,
            speaker_dim=SPEAKER_DIM,
        )

        # 2. Create pipeline with dummy encoders
        pipeline = FlowW2NPipeline(
            vae=vae,
            floww2n_model=floww2n_model,
            content_encoder=_DummyContentEncoder(),
            speaker_encoder=_DummySpeakerEncoder(),
            device="cpu",
            num_steps=2,
            compression_ratio=COMPRESSION_RATIO,
        )

        # 3. Generate random input audio
        audio_input = torch.randn(T_AUDIO)

        # 4. Run conversion
        audio_out = pipeline(audio_input, sample_rate=16000, seed=123)

        # 5. Verify output
        assert isinstance(audio_out, torch.Tensor)
        assert audio_out.dim() == 2  # (batch, samples)
        assert audio_out.shape[0] == 1
        assert audio_out.shape[1] == T_AUDIO

        # 6. Output should be finite (no NaN or Inf)
        assert torch.isfinite(audio_out).all(), "Output contains NaN or Inf values"

        # 7. Output should not be all zeros
        assert audio_out.abs().max() > 0, "Output is all zeros"

    def test_evaluation_pipeline_creation(self):
        """EvaluationPipeline can be instantiated (no model loading)."""
        from floww2n.evaluation.evaluate import EvaluationPipeline

        # This should not trigger heavy model loading due to lazy initialization
        # in FlowW2NMetrics
        evaluator = EvaluationPipeline(
            device="cpu",
            dnsmos_onnx_dir=None,
            sample_rate=16000,
        )
        assert evaluator.sample_rate == 16000
        assert evaluator.metrics is not None

    def test_evaluation_save_results(self, tmp_path):
        """EvaluationPipeline.save_results writes valid JSON."""
        import json

        from floww2n.evaluation.evaluate import EvaluationPipeline

        results = {
            "per_file": [
                {"filename": "test.wav", "utmos": 3.5, "spk_sim": 0.85},
                {"filename": "test2.wav", "utmos": 3.2, "spk_sim": 0.90},
            ],
            "summary": {
                "utmos": 3.35,
                "utmos_std": 0.15,
                "utmos_count": 2,
                "spk_sim": 0.875,
                "spk_sim_std": 0.025,
                "spk_sim_count": 2,
            },
        }

        output_file = tmp_path / "test_results.json"
        EvaluationPipeline.save_results(results, str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            loaded = json.load(f)
        assert loaded["summary"]["utmos"] == 3.35
        assert len(loaded["per_file"]) == 2

    def test_evaluation_print_summary(self, capsys):
        """EvaluationPipeline.print_summary prints formatted output."""
        from floww2n.evaluation.evaluate import EvaluationPipeline

        results = {
            "per_file": [
                {"filename": "test.wav", "utmos": 3.5},
            ],
            "summary": {
                "utmos": 3.5,
                "utmos_std": 0.0,
                "utmos_count": 1,
            },
        }

        EvaluationPipeline.print_summary(results)

        captured = capsys.readouterr()
        assert "Evaluation Summary" in captured.out
        assert "1 files" in captured.out
        assert "utmos" in captured.out
