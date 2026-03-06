#!/usr/bin/env python
"""Script: Train DiT with conditional flow matching."""

import argparse

from floww2n.training.train_dit import train_dit


def main():
    parser = argparse.ArgumentParser(description="Train FlowW2N DiT with CFM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dit.json",
        help="Path to DiT config JSON",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to cached features directory (from cache_features.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dit",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default=None,
        help="Path to trained VAE checkpoint (for validation decoding, optional)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (None to disable)",
    )
    args = parser.parse_args()

    if args.wandb_project:
        import wandb

        wandb.init(project=args.wandb_project, config=args.config)

    train_dit(
        args.config,
        args.cache_dir,
        args.output_dir,
        args.vae_checkpoint,
        args.resume_from,
    )


if __name__ == "__main__":
    main()
