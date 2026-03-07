#!/usr/bin/env python
"""Script: Run evaluation pipeline."""

import argparse

from floww2n.evaluation.evaluate import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument(
        "--converted-dir", type=str, required=True, help="Directory with converted audio files"
    )
    parser.add_argument(
        "--reference-dir", type=str, default=None, help="Directory with reference normal audio"
    )
    parser.add_argument(
        "--transcript-file", type=str, default=None, help="JSON file with transcripts"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json", help="Output results file"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dnsmos-onnx-dir", type=str, default=None)
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for WER transcription (default: en)",
    )
    args = parser.parse_args()

    evaluator = EvaluationPipeline(
        device=args.device,
        dnsmos_onnx_dir=args.dnsmos_onnx_dir,
        language=args.language,
    )
    results = evaluator.evaluate_directory(
        converted_dir=args.converted_dir,
        reference_dir=args.reference_dir,
        transcript_file=args.transcript_file,
    )
    evaluator.save_results(results, args.output)
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
