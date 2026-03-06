#!/usr/bin/env python
"""Script: Launch VAE training."""

import argparse

from floww2n.training.train_vae import train_vae


def main():
    parser = argparse.ArgumentParser(description="Train FlowW2N VAE")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vae.json",
        help="Path to VAE config JSON",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to audio data directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/vae",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
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

    train_vae(args.config, args.data_dir, args.output_dir, args.resume)


if __name__ == "__main__":
    main()
