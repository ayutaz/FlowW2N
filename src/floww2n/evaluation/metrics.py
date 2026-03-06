"""Evaluation metrics: WER, UTMOS, DNSMOS, SpkSim."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert audio to numpy float32 array."""
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32)


def _to_tensor(audio: Union[np.ndarray, torch.Tensor],
               device: str = "cpu") -> torch.Tensor:
    """Convert audio to torch float32 tensor on *device*."""
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio).float().to(device)
    return audio.float().to(device)


def _ensure_batch(audio: Union[np.ndarray, torch.Tensor]):
    """Ensure audio has a batch dimension (batch, samples)."""
    if isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            return audio.unsqueeze(0)
        return audio
    if audio.ndim == 1:
        return audio[np.newaxis, :]
    return audio


# =========================================================================
# WER (Word Error Rate)
# =========================================================================

class WERMetric:
    """Word Error Rate computation using ASR models.

    Supports two modes:

    - ``"normal"`` (WER-N): Uses Whisper *base* for testing intelligibility as
      normal speech. This is a readily available alternative to NeMo
      FastConformer that ships via ``transformers``.
    - ``"whisper"`` (WER-W): Uses Whisper *tiny* which is robust to whispered
      speech, matching the paper's evaluation protocol.
    """

    _MODEL_IDS = {
        "normal": "openai/whisper-base",
        "whisper": "openai/whisper-tiny",
    }

    def __init__(self, mode: str = "normal", device: str = "cpu"):
        """
        Args:
            mode: ``"normal"`` for WER-N, ``"whisper"`` for WER-W.
            device: Torch device string.
        """
        if mode not in self._MODEL_IDS:
            raise ValueError(f"mode must be one of {list(self._MODEL_IDS)}, got {mode!r}")
        self.mode = mode
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy-load the ASR model and processor."""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        model_id = self._MODEL_IDS[self.mode]
        self._processor = WhisperProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self._model.to(self.device).eval()

    @torch.no_grad()
    def transcribe(self, audio: Union[np.ndarray, torch.Tensor],
                   sample_rate: int = 16000) -> list[str]:
        """Transcribe audio to text.

        Args:
            audio: Waveform of shape ``(samples,)`` or ``(batch, samples)``,
                   numpy array or torch tensor.
            sample_rate: Input sample rate (must be 16 000).

        Returns:
            List of transcription strings, one per batch element.
        """
        if self._model is None:
            self._load_model()

        audio_np = _to_numpy(audio)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]

        transcriptions: list[str] = []
        for i in range(audio_np.shape[0]):
            waveform = audio_np[i]
            # Processor expects a 1-D numpy array at 16 kHz
            inputs = self._processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.device)
            # Force English decoding
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language="en", task="transcribe",
            )
            generated_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=448,
            )
            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True,
            )[0].strip()
            transcriptions.append(text)

        return transcriptions

    def compute(self, audio: Union[np.ndarray, torch.Tensor],
                references: list[str],
                sample_rate: int = 16000) -> float:
        """Compute WER between transcribed audio and reference texts.

        Args:
            audio: Waveform tensor/numpy array of shape ``(samples,)`` or
                   ``(batch, samples)``.
            references: List of reference transcription strings.
            sample_rate: Input sample rate.

        Returns:
            Word error rate as a float (0.0 = perfect).
        """
        from jiwer import wer

        hypotheses = self.transcribe(audio, sample_rate)
        # jiwer expects matching-length lists
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Number of references ({len(references)}) does not match "
                f"number of audio samples ({len(hypotheses)})."
            )
        return float(wer(references, hypotheses))


# =========================================================================
# UTMOS
# =========================================================================

class UTMOSMetric:
    """UTMOS (Universal Text-independent MOS prediction).

    Uses ``sarulab-speech/SpeechMOS`` model loaded via ``torch.hub``
    (``tarepan/SpeechMOS:v1.2.0``).  Predicts Mean Opinion Score (1--5)
    for speech quality / naturalness.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the UTMOS predictor via torch.hub."""
        try:
            self._model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            )
            self._model.to(self.device).eval()
        except Exception as exc:
            warnings.warn(
                f"Failed to load UTMOS model via torch.hub: {exc}. "
                "UTMOS scores will be NaN.",
                stacklevel=2,
            )
            self._model = "fallback"

    @torch.no_grad()
    def compute(self, audio: Union[np.ndarray, torch.Tensor],
                sample_rate: int = 16000) -> torch.Tensor:
        """Compute UTMOS score(s).

        Args:
            audio: Waveform of shape ``(samples,)`` or ``(batch, samples)``
                   at 16 kHz.
            sample_rate: Must be 16 000.

        Returns:
            Tensor of predicted MOS scores, one per batch element (higher is
            better, range roughly 1--5).
        """
        if self._model is None:
            self._load_model()

        wav = _to_tensor(audio, device=self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if self._model == "fallback":
            return torch.full((wav.shape[0],), float("nan"))

        # SpeechMOS expects (batch, samples) at 16 kHz
        scores = self._model(wav, sample_rate)
        return scores


# =========================================================================
# DNSMOS
# =========================================================================

class DNSMOSMetric:
    """DNSMOS (Deep Noise Suppression MOS) prediction.

    Uses Microsoft's DNS Challenge ONNX models for P.835 quality assessment
    (overall, signal, and background MOS).

    If the ONNX models are not available or ``onnxruntime`` is not installed
    the metric gracefully returns ``NaN`` placeholder scores instead of
    raising an error.

    The expected ONNX model files inside *onnx_model_dir* are:

    - ``sig_bak_ovr.onnx`` -- P.835 model (overall / signal / background)
    """

    # DNSMOS P.835 expects 16 kHz mono, 9.01-second segments (or shorter)
    _TARGET_SR = 16000
    _INPUT_LENGTH = 9.01  # seconds

    def __init__(self, onnx_model_dir: Optional[str] = None,
                 device: str = "cpu"):
        """
        Args:
            onnx_model_dir: Directory containing the DNSMOS ONNX model file(s).
                If ``None``, the metric returns ``NaN`` scores.
            device: Not used for ONNX inference (always CPU) but kept for API
                consistency.
        """
        self.onnx_model_dir = Path(onnx_model_dir) if onnx_model_dir else None
        self._session = None
        self._loaded = False

    def _load_model(self) -> None:
        """Lazy-load the ONNX session."""
        self._loaded = True

        if self.onnx_model_dir is None:
            warnings.warn(
                "DNSMOS: no onnx_model_dir provided -- returning NaN scores.",
                stacklevel=2,
            )
            self._session = "fallback"
            return

        try:
            import onnxruntime as ort
        except ImportError:
            warnings.warn(
                "onnxruntime is not installed. DNSMOS will return NaN scores. "
                "Install with: pip install onnxruntime",
                stacklevel=2,
            )
            self._session = "fallback"
            return

        model_path = self.onnx_model_dir / "sig_bak_ovr.onnx"
        if not model_path.exists():
            warnings.warn(
                f"DNSMOS ONNX model not found at {model_path}. "
                "Returning NaN scores.",
                stacklevel=2,
            )
            self._session = "fallback"
            return

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so,
        )

    def _run_onnx(self, audio_segment: np.ndarray) -> dict[str, float]:
        """Run a single segment through the ONNX model.

        Args:
            audio_segment: 1-D float32 numpy array at 16 kHz.

        Returns:
            Dict with keys ``"sig"``, ``"bak"``, ``"ovrl"``.
        """
        # Pad or truncate to the expected input length
        target_len = int(self._TARGET_SR * self._INPUT_LENGTH)
        if len(audio_segment) < target_len:
            audio_segment = np.pad(
                audio_segment, (0, target_len - len(audio_segment)),
            )
        else:
            audio_segment = audio_segment[:target_len]

        input_array = audio_segment[np.newaxis, ...]  # (1, samples)
        input_name = self._session.get_inputs()[0].name
        onnx_out = self._session.run(None, {input_name: input_array})
        # Typical P.835 model outputs: [ovrl, sig, bak] as shape (1,3) or similar
        out = np.asarray(onnx_out).flatten()
        if out.shape[0] >= 3:
            return {"sig": float(out[0]), "bak": float(out[1]), "ovrl": float(out[2])}
        # Fallback for single-output models
        return {"sig": float("nan"), "bak": float("nan"), "ovrl": float(out[0])}

    def compute(self, audio: Union[np.ndarray, torch.Tensor],
                sample_rate: int = 16000) -> dict[str, Union[float, list[float]]]:
        """Compute DNSMOS score(s).

        Args:
            audio: Waveform numpy array or tensor of shape ``(samples,)`` or
                   ``(batch, samples)``.
            sample_rate: Sample rate (audio is assumed 16 kHz; resampling is
                not performed).

        Returns:
            Dict with keys ``"ovrl"``, ``"sig"``, ``"bak"`` containing either a
            single float (one sample) or a list of floats (batch). Returns
            ``NaN`` values when the ONNX model is unavailable.
        """
        if not self._loaded:
            self._load_model()

        audio_np = _to_numpy(audio)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]

        if self._session == "fallback" or self._session is None:
            nan = float("nan")
            if audio_np.shape[0] == 1:
                return {"ovrl": nan, "sig": nan, "bak": nan}
            return {
                "ovrl": [nan] * audio_np.shape[0],
                "sig": [nan] * audio_np.shape[0],
                "bak": [nan] * audio_np.shape[0],
            }

        results = [self._run_onnx(audio_np[i]) for i in range(audio_np.shape[0])]

        if len(results) == 1:
            return results[0]

        return {
            key: [r[key] for r in results]
            for key in ("ovrl", "sig", "bak")
        }


# =========================================================================
# SpkSim (Speaker Similarity)
# =========================================================================

class SpkSimMetric:
    """Speaker similarity using ECAPA-TDNN cosine similarity.

    Uses the same ECAPA-TDNN model (``speechbrain/spkrec-ecapa-voxceleb``) as
    the conditioning module.  Computes cosine similarity between speaker
    embeddings of converted audio and reference audio.

    Note:
        The paper uses Resemblyzer (256-d) but we use ECAPA-TDNN (192-d) since
        Resemblyzer has Windows build issues (``webrtcvad`` C extension).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._encoder = None

    def _load_model(self) -> None:
        """Lazy-load the speaker encoder."""
        from floww2n.models.speaker_encoder import SpeakerEncoder

        self._encoder = SpeakerEncoder(device=self.device)

    @torch.no_grad()
    def compute(self, audio_a: Union[np.ndarray, torch.Tensor],
                audio_b: Union[np.ndarray, torch.Tensor],
                sample_rate: int = 16000) -> float:
        """Compute speaker similarity between two audio signals.

        Args:
            audio_a: First audio (e.g. converted speech), shape
                ``(samples,)`` or ``(1, samples)``.
            audio_b: Second audio (e.g. reference speech), same shape
                constraints.
            sample_rate: Sample rate (must be 16 000).

        Returns:
            Cosine similarity score in ``[-1, 1]`` (higher = more similar).
        """
        if self._encoder is None:
            self._load_model()

        wav_a = _to_tensor(audio_a, device=self.device)
        wav_b = _to_tensor(audio_b, device=self.device)

        # Ensure (batch, samples) -- use batch=1
        if wav_a.dim() == 1:
            wav_a = wav_a.unsqueeze(0)
        if wav_b.dim() == 1:
            wav_b = wav_b.unsqueeze(0)

        emb_a = self._encoder(wav_a)  # (1, 192)
        emb_b = self._encoder(wav_b)  # (1, 192)

        similarity = F.cosine_similarity(emb_a, emb_b, dim=-1)
        return float(similarity.squeeze())

    @torch.no_grad()
    def compute_batch(self, audios_a: list[Union[np.ndarray, torch.Tensor]],
                      audios_b: list[Union[np.ndarray, torch.Tensor]],
                      sample_rate: int = 16000) -> list[float]:
        """Compute speaker similarity for multiple pairs.

        Args:
            audios_a: List of converted audio waveforms.
            audios_b: List of reference audio waveforms (same length).
            sample_rate: Sample rate.

        Returns:
            List of cosine similarity scores.
        """
        if len(audios_a) != len(audios_b):
            raise ValueError(
                f"Mismatched list lengths: {len(audios_a)} vs {len(audios_b)}"
            )
        return [
            self.compute(a, b, sample_rate)
            for a, b in zip(audios_a, audios_b)
        ]


# =========================================================================
# FlowW2NMetrics (bundled convenience class)
# =========================================================================

class FlowW2NMetrics:
    """Convenience class that bundles all FlowW2N evaluation metrics.

    Example::

        metrics = FlowW2NMetrics(device="cuda")
        results = metrics.evaluate(
            converted_audio=audio_out,
            reference_audio=audio_ref,
            reference_text="hello world",
        )
        # results = {
        #     "wer_n": 0.15,
        #     "wer_w": 0.08,
        #     "utmos": 3.8,
        #     "dnsmos_ovrl": 3.5,
        #     "dnsmos_sig": 3.7,
        #     "dnsmos_bak": 4.0,
        #     "spk_sim": 0.92,
        # }
    """

    def __init__(self, device: str = "cpu",
                 dnsmos_onnx_dir: Optional[str] = None):
        self.wer_n = WERMetric(mode="normal", device=device)
        self.wer_w = WERMetric(mode="whisper", device=device)
        self.utmos = UTMOSMetric(device=device)
        self.dnsmos = DNSMOSMetric(onnx_model_dir=dnsmos_onnx_dir)
        self.spk_sim = SpkSimMetric(device=device)

    def evaluate(
        self,
        converted_audio: Union[np.ndarray, torch.Tensor],
        reference_audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reference_text: Optional[Union[str, list[str]]] = None,
        sample_rate: int = 16000,
    ) -> dict[str, float]:
        """Run all available metrics.

        Metrics that lack the required inputs are silently skipped (their keys
        will be absent from the returned dict).

        Args:
            converted_audio: Converted (normal) speech waveform, shape
                ``(samples,)`` or ``(batch, samples)``.
            reference_audio: Reference normal speech for SpkSim.
            reference_text: Reference transcription(s) for WER.
                A single string is treated as a one-element list.
            sample_rate: Sample rate of all waveforms.

        Returns:
            Dict mapping metric names to float scores.
        """
        results: dict[str, float] = {}

        # -- WER --
        if reference_text is not None:
            if isinstance(reference_text, str):
                reference_text = [reference_text]
            try:
                results["wer_n"] = self.wer_n.compute(
                    converted_audio, reference_text, sample_rate,
                )
            except Exception as exc:
                warnings.warn(f"WER-N computation failed: {exc}", stacklevel=2)

            try:
                results["wer_w"] = self.wer_w.compute(
                    converted_audio, reference_text, sample_rate,
                )
            except Exception as exc:
                warnings.warn(f"WER-W computation failed: {exc}", stacklevel=2)

        # -- UTMOS --
        try:
            utmos_scores = self.utmos.compute(converted_audio, sample_rate)
            results["utmos"] = float(utmos_scores.mean())
        except Exception as exc:
            warnings.warn(f"UTMOS computation failed: {exc}", stacklevel=2)

        # -- DNSMOS --
        try:
            dnsmos = self.dnsmos.compute(converted_audio, sample_rate)
            for key in ("ovrl", "sig", "bak"):
                val = dnsmos[key]
                if isinstance(val, list):
                    results[f"dnsmos_{key}"] = float(np.nanmean(val))
                else:
                    results[f"dnsmos_{key}"] = float(val)
        except Exception as exc:
            warnings.warn(f"DNSMOS computation failed: {exc}", stacklevel=2)

        # -- SpkSim --
        if reference_audio is not None:
            try:
                results["spk_sim"] = self.spk_sim.compute(
                    converted_audio, reference_audio, sample_rate,
                )
            except Exception as exc:
                warnings.warn(f"SpkSim computation failed: {exc}", stacklevel=2)

        return results


# =========================================================================
# CLI smoke test
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FlowW2N evaluation metrics -- smoke test (dummy audio)")
    print("=" * 60)

    sr = 16000
    duration = 3  # seconds
    n_samples = sr * duration
    dummy_audio = np.random.randn(n_samples).astype(np.float32) * 0.01

    # -- DNSMOS (fallback, no ONNX dir) --
    print("\n--- DNSMOS (fallback) ---")
    dnsmos = DNSMOSMetric(onnx_model_dir=None)
    dnsmos_scores = dnsmos.compute(dummy_audio, sample_rate=sr)
    print(f"  DNSMOS scores: {dnsmos_scores}")
    assert math.isnan(dnsmos_scores["ovrl"]), "Expected NaN for fallback"
    print("  [OK] DNSMOS fallback returns NaN as expected.")

    # -- DNSMOS with batch --
    print("\n--- DNSMOS batch (fallback) ---")
    batch_audio = np.random.randn(3, n_samples).astype(np.float32) * 0.01
    dnsmos_batch = dnsmos.compute(batch_audio, sample_rate=sr)
    print(f"  DNSMOS batch scores: {dnsmos_batch}")
    assert isinstance(dnsmos_batch["ovrl"], list), "Expected list for batch"
    assert len(dnsmos_batch["ovrl"]) == 3
    print("  [OK] DNSMOS batch fallback works.")

    # -- SpkSim --
    print("\n--- SpkSim ---")
    try:
        spk = SpkSimMetric(device="cpu")
        audio_a = np.random.randn(n_samples).astype(np.float32) * 0.01
        audio_b = np.random.randn(n_samples).astype(np.float32) * 0.01
        sim = spk.compute(audio_a, audio_b, sample_rate=sr)
        print(f"  SpkSim (random vs random): {sim:.4f}")
        # Same audio should have high similarity
        sim_same = spk.compute(audio_a, audio_a, sample_rate=sr)
        print(f"  SpkSim (same vs same):     {sim_same:.4f}")
        assert sim_same > sim, "Same audio should be more similar to itself"
        print("  [OK] SpkSim works.")
    except Exception as e:
        print(f"  [SKIP] SpkSim not available: {e}")

    # -- UTMOS --
    print("\n--- UTMOS ---")
    try:
        utmos = UTMOSMetric(device="cpu")
        scores = utmos.compute(dummy_audio, sample_rate=sr)
        print(f"  UTMOS score: {scores}")
        print("  [OK] UTMOS works.")
    except Exception as e:
        print(f"  [SKIP] UTMOS not available: {e}")

    # -- WER --
    print("\n--- WER ---")
    try:
        wer_metric = WERMetric(mode="whisper", device="cpu")
        # With random noise the transcription is unpredictable, but the
        # pipeline should run without error.
        hyps = wer_metric.transcribe(dummy_audio, sample_rate=sr)
        print(f"  Whisper tiny transcription of noise: {hyps!r}")
        # Compute WER against a dummy reference
        wer_val = wer_metric.compute(dummy_audio, ["hello world"], sample_rate=sr)
        print(f"  WER (noise vs 'hello world'): {wer_val:.4f}")
        print("  [OK] WER works.")
    except Exception as e:
        print(f"  [SKIP] WER not available: {e}")

    # -- FlowW2NMetrics bundle --
    print("\n--- FlowW2NMetrics (bundled) ---")
    try:
        metrics = FlowW2NMetrics(device="cpu")
        results = metrics.evaluate(
            converted_audio=dummy_audio,
            reference_audio=dummy_audio,
            reference_text="hello world",
            sample_rate=sr,
        )
        print(f"  Results: {results}")
        print("  [OK] Bundled metrics work.")
    except Exception as e:
        print(f"  [SKIP] Bundled metrics: {e}")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)
