"""
Synthetic whisper generation: 4 methods (LPC, glottal, formant, Praat).

This module implements the four whisper synthesis methods used in FlowW2N training.
Each method is selected with equal probability (25%) during data augmentation.

Methods:
    1. LPC-based devoicing: Detects voiced frames and replaces excitation with noise
    2. Glottal source removal: Replaces all excitation with noise (no voicing detection)
    3. Formant bandwidth modification: Widens formant bandwidths by scaling pole magnitudes
    4. Praat whisperize: Uses Praat's LPC analysis via parselmouth library

Key Features:
    - All methods preserve input length and energy (RMS normalization)
    - Frame-based processing with overlap-add reconstruction
    - Robust error handling with fallbacks
    - Custom LPC implementation using Levinson-Durbin recursion

Usage:
    >>> from floww2n.data.whisper_synthesis import WhisperSynthesizer
    >>> synthesizer = WhisperSynthesizer(sr=16000)
    >>> whispered_audio = synthesizer(normal_audio)  # Random method
    >>> whispered_audio = synthesizer(normal_audio, method_index=0)  # Specific method

References:
    - FlowW2N paper: arXiv:2603.04296v1 [eess.AS]
    - Section: "Paired Data Preparation" (synthetic whisper generation)
"""

import random
from collections.abc import Callable

import numpy as np
from scipy import signal


def _lpc_analysis(frame: np.ndarray, order: int) -> np.ndarray:
    """
    Compute LPC coefficients using autocorrelation (Levinson-Durbin).

    Args:
        frame: Input frame (1D array)
        order: LPC order

    Returns:
        LPC coefficients [1, a1, a2, ..., ap] where p=order
    """
    # Compute autocorrelation
    r = np.correlate(frame, frame, mode="full")
    r = r[len(r) // 2 : len(r) // 2 + order + 1]

    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    a[0] = 1.0

    if r[0] < 1e-10:
        return a

    # Initial values
    E = r[0]
    for i in range(1, order + 1):
        # Reflection coefficient
        lambda_i = -np.sum(a[:i] * r[i:0:-1]) / E

        # Update coefficients
        a_new = a.copy()
        a_new[i] = lambda_i
        for j in range(1, i):
            a_new[j] = a[j] + lambda_i * a[i - j]

        a = a_new
        E = E * (1 - lambda_i**2)

        if E < 1e-10:
            break

    return a


def lpc_devoice(
    audio: np.ndarray,
    sr: int = 16000,
    lpc_order: int = 16,
    frame_size: int = 512,
    hop_size: int = 128,
) -> np.ndarray:
    """
    LPC-based devoicing: Extract LPC filter, replace voiced excitation with noise.

    Args:
        audio: Input audio signal (1D numpy array, float32)
        sr: Sample rate (Hz)
        lpc_order: Order of LPC analysis
        frame_size: Frame size in samples
        hop_size: Hop size in samples

    Returns:
        Devoiced audio signal (same length as input)
    """
    audio = audio.astype(np.float32)
    n_samples = len(audio)
    window = np.hanning(frame_size)

    # Store input energy for normalization
    input_energy = np.sqrt(np.mean(audio**2)) if np.any(audio) else 1.0

    # Output buffer
    output = np.zeros(n_samples, dtype=np.float32)
    window_sum = np.zeros(n_samples, dtype=np.float32)

    # Process frames
    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = audio[start : start + frame_size] * window

        # Skip silent frames
        if np.sum(frame**2) < 1e-10:
            continue

        try:
            # LPC analysis
            lpc_coeffs = _lpc_analysis(frame, lpc_order)

            # Extract residual (excitation) by inverse filtering
            residual = signal.lfilter(lpc_coeffs, [1.0], frame)

            # Simple voicing detection using autocorrelation
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Check for periodicity in pitch range (80-400 Hz)
            min_lag = int(sr / 400)
            max_lag = int(sr / 80)
            max_lag = min(max_lag, len(autocorr))

            is_voiced = False
            if max_lag > min_lag:
                peak = np.max(autocorr[min_lag:max_lag])
                is_voiced = peak > 0.3  # Voicing threshold

            # Replace voiced excitation with noise
            if is_voiced:
                # Preserve energy
                energy = np.sqrt(np.mean(residual**2))
                noise = np.random.randn(len(residual)).astype(np.float32)
                noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-10) * energy
                residual = noise

            # Resynthesize with LPC filter
            synthesized = signal.lfilter([1.0], lpc_coeffs, residual)

            # Overlap-add
            output[start : start + frame_size] += synthesized * window
            window_sum[start : start + frame_size] += window**2

        except (ValueError, np.linalg.LinAlgError):
            # If LPC fails, copy original frame
            output[start : start + frame_size] += frame
            window_sum[start : start + frame_size] += window**2

    # Normalize by window sum
    window_sum = np.maximum(window_sum, 1e-10)
    output = output / window_sum

    # Normalize energy to match input
    output_energy = np.sqrt(np.mean(output**2)) if np.any(output) else 1.0
    if output_energy > 1e-10:
        output = output * (input_energy / output_energy)

    return output


def glottal_source_removal(
    audio: np.ndarray,
    sr: int = 16000,
    lpc_order: int = 16,
    frame_size: int = 512,
    hop_size: int = 128,
) -> np.ndarray:
    """
    Glottal source removal: Replace all excitation with noise (no voicing detection).

    Args:
        audio: Input audio signal (1D numpy array, float32)
        sr: Sample rate (Hz)
        lpc_order: Order of LPC analysis
        frame_size: Frame size in samples
        hop_size: Hop size in samples

    Returns:
        Whisperized audio signal (same length as input)
    """
    audio = audio.astype(np.float32)
    n_samples = len(audio)
    window = np.hanning(frame_size)

    # Store input energy for normalization
    input_energy = np.sqrt(np.mean(audio**2)) if np.any(audio) else 1.0

    # Output buffer
    output = np.zeros(n_samples, dtype=np.float32)
    window_sum = np.zeros(n_samples, dtype=np.float32)

    # Process frames
    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = audio[start : start + frame_size] * window

        # Skip silent frames
        if np.sum(frame**2) < 1e-10:
            continue

        try:
            # LPC analysis
            lpc_coeffs = _lpc_analysis(frame, lpc_order)

            # Extract residual by inverse filtering
            residual = signal.lfilter(lpc_coeffs, [1.0], frame)

            # Replace ALL excitation with noise (no voicing check)
            energy = np.sqrt(np.mean(residual**2))
            noise = np.random.randn(len(residual)).astype(np.float32)
            noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-10) * energy

            # Resynthesize with LPC filter
            synthesized = signal.lfilter([1.0], lpc_coeffs, noise)

            # Overlap-add
            output[start : start + frame_size] += synthesized * window
            window_sum[start : start + frame_size] += window**2

        except (ValueError, np.linalg.LinAlgError):
            # If LPC fails, copy original frame
            output[start : start + frame_size] += frame
            window_sum[start : start + frame_size] += window**2

    # Normalize by window sum
    window_sum = np.maximum(window_sum, 1e-10)
    output = output / window_sum

    # Normalize energy to match input
    output_energy = np.sqrt(np.mean(output**2)) if np.any(output) else 1.0
    if output_energy > 1e-10:
        output = output * (input_energy / output_energy)

    return output


def formant_bandwidth_mod(
    audio: np.ndarray,
    sr: int = 16000,
    bandwidth_factor: float = 0.7,
    lpc_order: int = 16,
    frame_size: int = 512,
    hop_size: int = 128,
) -> np.ndarray:
    """
    Formant bandwidth modification: Widen formant bandwidths by shrinking pole magnitudes.

    Args:
        audio: Input audio signal (1D numpy array, float32)
        sr: Sample rate (Hz)
        bandwidth_factor: Pole magnitude scaling factor (0 < factor < 1)
        lpc_order: Order of LPC analysis
        frame_size: Frame size in samples
        hop_size: Hop size in samples

    Returns:
        Whisperized audio signal (same length as input)
    """
    audio = audio.astype(np.float32)
    n_samples = len(audio)
    window = np.hanning(frame_size)

    # Store input energy for normalization
    input_energy = np.sqrt(np.mean(audio**2)) if np.any(audio) else 1.0

    # Output buffer
    output = np.zeros(n_samples, dtype=np.float32)
    window_sum = np.zeros(n_samples, dtype=np.float32)

    # Process frames
    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = audio[start : start + frame_size] * window

        # Skip silent frames
        if np.sum(frame**2) < 1e-10:
            continue

        try:
            # LPC analysis
            lpc_coeffs = _lpc_analysis(frame, lpc_order)

            # Extract residual
            residual = signal.lfilter(lpc_coeffs, [1.0], frame)

            # Compute poles (roots of denominator polynomial)
            # LPC coeffs represent A(z) = 1 + a1*z^-1 + ... + ap*z^-p
            # We need roots of 1 + a1*z + ... + ap*z^p
            poles = np.roots(lpc_coeffs)

            # Modify pole magnitudes (shrink to widen bandwidth)
            modified_poles = poles * bandwidth_factor

            # Reconstruct LPC coefficients from modified poles
            modified_lpc_coeffs = np.poly(modified_poles)

            # Ensure real coefficients (take real part due to numerical noise)
            modified_lpc_coeffs = np.real(modified_lpc_coeffs)

            # Resynthesize with modified LPC filter
            synthesized = signal.lfilter([1.0], modified_lpc_coeffs, residual)

            # Overlap-add
            output[start : start + frame_size] += synthesized * window
            window_sum[start : start + frame_size] += window**2

        except (ValueError, np.linalg.LinAlgError):
            # If LPC fails, copy original frame
            output[start : start + frame_size] += frame
            window_sum[start : start + frame_size] += window**2

    # Normalize by window sum
    window_sum = np.maximum(window_sum, 1e-10)
    output = output / window_sum

    # Normalize energy to match input
    output_energy = np.sqrt(np.mean(output**2)) if np.any(output) else 1.0
    if output_energy > 1e-10:
        output = output * (input_energy / output_energy)

    return output


def praat_whisperize(audio: np.ndarray, sr: int = 16000, lpc_order: int = 16) -> np.ndarray:
    """
    Praat-based whisperization using parselmouth library.

    This method uses Praat's LPC analysis capabilities through parselmouth.
    The implementation performs frame-based LPC analysis and noise resynthesis,
    similar to Praat's whisperization functionality.

    Args:
        audio: Input audio signal (1D numpy array, float32)
        sr: Sample rate (Hz)
        lpc_order: Order of LPC analysis

    Returns:
        Whisperized audio signal (same length as input)
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError as err:
        raise ImportError(
            "parselmouth is required for praat_whisperize. Install with: uv add praat-parselmouth"
        ) from err

    audio = audio.astype(np.float64)  # Parselmouth expects float64
    original_length = len(audio)

    # Store input energy for normalization
    input_energy = np.sqrt(np.mean(audio**2)) if np.any(audio) else 1.0

    try:
        # Create Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Praat LPC analysis using autocorrelation method
        # Parameters similar to Praat's "To LPC (autocorrelation)"
        window_length = 0.025  # 25ms window
        time_step = 0.005  # 5ms time step
        pre_emphasis_frequency = 50.0

        # Perform LPC analysis
        lpc = call(
            sound,
            "To LPC (autocorrelation)",
            lpc_order,
            window_length,
            time_step,
            pre_emphasis_frequency,
        )

        # Get the source (residual) signal
        # In Praat, we can extract source by filtering
        source = call([sound, lpc], "Filter (inverse)")

        # Replace source with noise while preserving energy characteristics
        source_values = source.values[0]  # Get mono channel
        energy = np.sqrt(np.mean(source_values**2))

        # Generate noise with same energy
        noise = np.random.randn(len(source_values)).astype(np.float64)
        noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-10) * energy

        # Create new sound from noise
        noise_sound = parselmouth.Sound(noise, sampling_frequency=sr)

        # Resynthesize using LPC filter
        whispered = call([noise_sound, lpc], "Filter", False)

        # Get output values
        result = whispered.values[0]

        # Ensure output length matches input
        if len(result) != original_length:
            if len(result) > original_length:
                result = result[:original_length]
            else:
                result = np.pad(result, (0, original_length - len(result)))

        # Normalize energy to match input
        output_energy = np.sqrt(np.mean(result**2)) if np.any(result) else 1.0
        if output_energy > 1e-10:
            result = result * (input_energy / output_energy)

        return result.astype(np.float32)

    except Exception as e:
        # Fallback to glottal source removal if Praat processing fails
        print(f"Praat whisperization failed: {e}. Falling back to glottal removal.")
        result = glottal_source_removal(audio.astype(np.float32), sr=sr, lpc_order=lpc_order)
        if len(result) != original_length:
            if len(result) > original_length:
                result = result[:original_length]
            else:
                result = np.pad(result, (0, original_length - len(result)))
        return result.astype(np.float32)


class WhisperSynthesizer:
    """
    Unified whisper synthesizer: randomly selects one of 4 methods.

    Methods:
        1. LPC-based devoicing (voiced->noise)
        2. Glottal source removal (all->noise)
        3. Formant bandwidth modification (pole magnitude scaling)
        4. Praat whisperize (parselmouth-based)
    """

    def __init__(
        self,
        sr: int = 16000,
        methods: list[Callable] | None = None,
    ):
        """
        Initialize synthesizer.

        Args:
            sr: Sample rate (Hz)
            methods: List of synthesis methods (functions). If None, uses all 4.
        """
        self.sr = sr

        if methods is None:
            self.methods = [
                lpc_devoice,
                glottal_source_removal,
                formant_bandwidth_mod,
                praat_whisperize,
            ]
        else:
            self.methods = methods

    def __call__(self, audio: np.ndarray, method_index: int | None = None) -> np.ndarray:
        """
        Synthesize whispered speech from normal speech.

        Args:
            audio: Input audio signal (1D numpy array, float32)
            method_index: If specified, use this method (0-3). Otherwise random.

        Returns:
            Synthesized whispered speech (same length as input)
        """
        original_length = len(audio)

        # Select method
        if method_index is not None:
            if not 0 <= method_index < len(self.methods):
                raise ValueError(
                    f"method_index {method_index} out of range [0, {len(self.methods)})"
                )
            method = self.methods[method_index]
        else:
            method = random.choice(self.methods)

        # Apply synthesis
        result = method(audio, sr=self.sr)

        # Ensure output length matches input length
        if len(result) != original_length:
            if len(result) > original_length:
                result = result[:original_length]
            else:
                result = np.pad(result, (0, original_length - len(result)))

        return result.astype(np.float32)


if __name__ == "__main__":
    import time

    sr = 16000
    duration = 2.0  # seconds
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1

    methods = [
        ("LPC Devoicing", lpc_devoice),
        ("Glottal Source Removal", glottal_source_removal),
        ("Formant Bandwidth Mod", formant_bandwidth_mod),
        ("Praat Whisperize", praat_whisperize),
    ]

    print("Testing individual methods:")
    print("-" * 60)
    for name, method in methods:
        start = time.time()
        result = method(audio, sr=sr)
        elapsed = time.time() - start
        print(
            f"{name}: input={len(audio)}, output={len(result)}, "
            f"match={len(audio) == len(result)}, time={elapsed:.3f}s"
        )

    print("\nTesting WhisperSynthesizer:")
    print("-" * 60)
    synth = WhisperSynthesizer(sr=sr)
    for i in range(8):
        result = synth(audio)
        print(f"WhisperSynthesizer call {i}: output_len={len(result)}")

    print("\nAll tests completed successfully!")
