"""Evaluation pipeline runner."""

import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm


class EvaluationPipeline:
    """Run full evaluation on a set of audio files.

    Takes converted audio files and reference data, runs all metrics,
    and produces a summary report.

    Usage:
        evaluator = EvaluationPipeline(device="cuda")
        results = evaluator.evaluate_directory(
            converted_dir="outputs/converted/",
            reference_dir="data/reference/",
            transcript_file="data/transcripts.json",
        )
        evaluator.save_results(results, "outputs/eval_results.json")
    """

    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".sph", ".m4a"}

    def __init__(self, device="cpu", dnsmos_onnx_dir=None, sample_rate=16000):
        """
        Args:
            device: Device for metric models
            dnsmos_onnx_dir: Optional DNSMOS ONNX model directory
            sample_rate: Expected audio sample rate
        """
        from floww2n.evaluation.metrics import FlowW2NMetrics
        self.metrics = FlowW2NMetrics(device=device, dnsmos_onnx_dir=dnsmos_onnx_dir)
        self.sample_rate = sample_rate

    def _load_audio(self, audio_path):
        """Load audio file and resample to target sample rate if needed.

        Args:
            audio_path: Path to audio file

        Returns:
            Waveform as 1-D numpy array at self.sample_rate
        """
        waveform, sr = torchaudio.load(str(audio_path))
        # Mono: take first channel if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        waveform = waveform.squeeze(0)  # (samples,)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        return waveform.numpy()

    def evaluate_single(self, converted_audio, reference_audio=None,
                        reference_text=None):
        """Evaluate a single converted audio sample.

        Args:
            converted_audio: Converted audio waveform (numpy array or tensor),
                shape (samples,) or (1, samples).
            reference_audio: Optional reference normal audio for SpkSim,
                same shape constraints.
            reference_text: Optional ground-truth transcript string.

        Returns:
            Dict of metric name -> score
        """
        return self.metrics.evaluate(
            converted_audio=converted_audio,
            reference_audio=reference_audio,
            reference_text=reference_text,
            sample_rate=self.sample_rate,
        )

    def evaluate_directory(self, converted_dir, reference_dir=None,
                           transcript_file=None):
        """Evaluate all files in a directory.

        Args:
            converted_dir: Directory with converted audio files
            reference_dir: Directory with reference normal audio (same filenames)
            transcript_file: JSON file mapping filenames to transcripts

        Returns:
            Dict with:
            - "per_file": List of per-file results (each is a dict with
              "filename" and metric scores)
            - "summary": Dict of metric -> mean score
        """
        converted_path = Path(converted_dir)
        reference_path = Path(reference_dir) if reference_dir else None

        # Load transcripts if provided
        transcripts = {}
        if transcript_file is not None:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcripts = json.load(f)

        # Find converted audio files
        converted_files = []
        for ext in self.AUDIO_EXTENSIONS:
            converted_files.extend(converted_path.glob(f"*{ext}"))
            converted_files.extend(converted_path.glob(f"*{ext.upper()}"))
        converted_files.sort()

        if not converted_files:
            raise FileNotFoundError(
                f"No audio files found in {converted_dir}"
            )

        per_file_results = []

        for audio_path in tqdm(converted_files, desc="Evaluating"):
            filename = audio_path.name
            stem = audio_path.stem

            # Load converted audio
            converted_audio = self._load_audio(audio_path)

            # Load reference audio if available
            reference_audio = None
            if reference_path is not None:
                # Try to find matching reference file (same name)
                ref_candidates = []
                for ext in self.AUDIO_EXTENSIONS:
                    ref_file = reference_path / f"{stem}{ext}"
                    if ref_file.exists():
                        ref_candidates.append(ref_file)
                if ref_candidates:
                    reference_audio = self._load_audio(ref_candidates[0])

            # Get transcript if available
            reference_text = None
            if filename in transcripts:
                reference_text = transcripts[filename]
            elif stem in transcripts:
                reference_text = transcripts[stem]

            # Evaluate
            file_results = self.evaluate_single(
                converted_audio=converted_audio,
                reference_audio=reference_audio,
                reference_text=reference_text,
            )
            file_results["filename"] = filename
            per_file_results.append(file_results)

        # Compute summary statistics
        summary = self._compute_summary(per_file_results)

        return {
            "per_file": per_file_results,
            "summary": summary,
        }

    @staticmethod
    def _compute_summary(per_file_results):
        """Compute mean scores across all files.

        Args:
            per_file_results: List of per-file result dicts

        Returns:
            Dict of metric -> mean score
        """
        if not per_file_results:
            return {}

        # Collect all metric keys (exclude "filename")
        metric_keys = set()
        for result in per_file_results:
            for key in result:
                if key != "filename":
                    metric_keys.add(key)

        summary = {}
        for key in sorted(metric_keys):
            values = []
            for result in per_file_results:
                if key in result:
                    val = result[key]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)
            if values:
                summary[key] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_count"] = len(values)

        return summary

    @staticmethod
    def save_results(results, output_path):
        """Save evaluation results to JSON.

        Args:
            results: Dict with "per_file" and "summary" keys
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert any non-serializable values
        def _make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            return obj

        def _clean_dict(d):
            if isinstance(d, dict):
                return {k: _clean_dict(v) for k, v in d.items()}
            if isinstance(d, list):
                return [_clean_dict(v) for v in d]
            return _make_serializable(d)

        cleaned = _clean_dict(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")

    @staticmethod
    def print_summary(results):
        """Pretty print evaluation summary.

        Args:
            results: Dict with "summary" key containing metric scores
        """
        summary = results.get("summary", {})
        num_files = len(results.get("per_file", []))

        print("\n" + "=" * 60)
        print(f"  Evaluation Summary ({num_files} files)")
        print("=" * 60)

        # Group metrics for cleaner display
        metric_groups = {
            "Intelligibility (WER)": ["wer_n", "wer_w"],
            "Naturalness": ["utmos"],
            "Quality (DNSMOS)": ["dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"],
            "Speaker Similarity": ["spk_sim"],
        }

        for group_name, metric_keys in metric_groups.items():
            group_has_data = False
            for key in metric_keys:
                if key in summary:
                    if not group_has_data:
                        print(f"\n  {group_name}:")
                        group_has_data = True
                    mean_val = summary[key]
                    std_key = f"{key}_std"
                    count_key = f"{key}_count"
                    std_val = summary.get(std_key, 0.0)
                    count = summary.get(count_key, num_files)
                    # Format WER as percentage
                    if "wer" in key:
                        print(
                            f"    {key:>15s}: {mean_val * 100:6.2f}% "
                            f"(+/- {std_val * 100:.2f}%, n={count})"
                        )
                    else:
                        print(
                            f"    {key:>15s}: {mean_val:6.4f} "
                            f"(+/- {std_val:.4f}, n={count})"
                        )

        print("\n" + "=" * 60)
