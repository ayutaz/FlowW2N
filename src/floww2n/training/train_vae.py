"""VAE training script with multi-resolution STFT + discriminator + KL loss."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import wandb
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from floww2n.models.vae import AudioAutoencoder
from floww2n.training.checkpoint_utils import prune_checkpoints
from floww2n.training.dataset import VAEDataset
from floww2n.training.losses import VAELoss

_ckpt_executor = ThreadPoolExecutor(max_workers=1)


def setup_cuda_optimizations():
    """Configure CUDA optimizations for faster training."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    # TF32 for Ampere+ GPUs
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def train_vae(config_path, data_dir, output_dir, resume_from=None):
    """
    Train VAE model.

    Args:
        config_path: Path to VAE config JSON
        data_dir: Path to audio data directory
        output_dir: Path to save checkpoints
        resume_from: Path to checkpoint to resume from (optional)
    """
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["losses"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_cuda_optimizations()

    # Model
    model = AudioAutoencoder(
        io_channels=model_cfg["io_channels"],
        latent_dim=model_cfg["latent_dim"],
        encoder_latent_dim=model_cfg["encoder_latent_dim"],
        channels=model_cfg["channels"],
        c_mults=model_cfg["c_mults"],
        strides=model_cfg["strides"],
        use_snake=model_cfg["use_snake"],
    ).to(device)

    # Loss (contains discriminator)
    criterion = VAELoss(
        stft_weight=loss_cfg["multi_resolution_stft_weight"],
        adv_weight=loss_cfg["adversarial_weight"],
        feat_weight=loss_cfg["feature_matching_weight"],
        kl_weight=loss_cfg["kl_weight"],
    ).to(device)

    # torch.compile
    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
        criterion.discriminator = torch.compile(criterion.discriminator)

    # Optimizers (separate for generator and discriminator)
    fused = device.type == "cuda"
    opt_g = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        fused=fused,
    )
    opt_d = torch.optim.AdamW(
        criterion.discriminator.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        fused=fused,
    )

    # EMA
    ema = EMA(model, beta=train_cfg["ema_decay"])

    # Dataset & DataLoader
    dataset = VAEDataset(
        audio_dir=data_dir,
        segment_length=train_cfg["segment_length"],
        sample_rate=model_cfg["sample_rate"],
    )
    num_workers = train_cfg["num_workers"]
    loader_kwargs = dict(
        batch_size=train_cfg["micro_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    dataloader = DataLoader(dataset, **loader_kwargs)

    # Mixed precision
    use_amp = train_cfg["mixed_precision"] == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grad_accum = train_cfg["gradient_accumulation_steps"]
    global_step = 0
    max_steps = train_cfg["max_steps"]

    # Resume
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["criterion"])
        opt_g.load_state_dict(checkpoint["opt_g"])
        opt_d.load_state_dict(checkpoint["opt_d"])
        ema.load_state_dict(checkpoint["ema"])
        global_step = checkpoint["global_step"]
        print(f"Resumed from checkpoint at step {global_step}")

    model.train()
    criterion.train()

    print(f"Starting training from step {global_step} to {max_steps}")
    print(f"Dataset: {len(dataset)} files")
    print(f"Device: {device}")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            batch = batch.to(device)  # (B, 1, T)

            # === Generator step ===
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                recon, info = model(batch)
                g_loss, g_dict = criterion.generator_loss(batch, recon, info["kl_loss"])
                g_loss = g_loss / grad_accum

            g_loss.backward()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_g.step()
                opt_g.zero_grad(set_to_none=True)
                ema.update()

            # === Discriminator step ===
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                d_loss, d_dict = criterion.discriminator_loss(batch, recon.detach())
                d_loss = d_loss / grad_accum

            d_loss.backward()

            if (global_step + 1) % grad_accum == 0:
                opt_d.step()
                opt_d.zero_grad(set_to_none=True)

            # Logging
            if global_step % 100 == 0:
                log_dict = {f"train/{k}": v.item() for k, v in {**g_dict, **d_dict}.items()}
                log_dict["train/global_step"] = global_step
                g_total = g_dict.get("total_g", g_loss * grad_accum).item()
                d_total = d_dict.get("total_d", d_loss * grad_accum).item()
                print(f"Step {global_step}/{max_steps} | G: {g_total:.4f} | D: {d_total:.4f}")
                if wandb.run:
                    wandb.log(log_dict, step=global_step)

            # Checkpoint
            if global_step > 0 and global_step % 10000 == 0:
                save_checkpoint(output_path, model, criterion, opt_g, opt_d, ema, global_step)

            global_step += 1

    # Final save
    save_checkpoint(output_path, model, criterion, opt_g, opt_d, ema, global_step)
    # Save EMA weights
    torch.save(ema.ema_model.state_dict(), output_path / "vae_ema_final.pt")
    print(f"Training complete. Final step: {global_step}")


def save_checkpoint(output_path, model, criterion, opt_g, opt_d, ema, global_step, max_keep=3):
    """Save training checkpoint asynchronously.

    Args:
        output_path: Directory to save checkpoint
        model: AudioAutoencoder model
        criterion: VAELoss (contains discriminator)
        opt_g: Generator optimizer
        opt_d: Discriminator optimizer
        ema: EMA wrapper
        global_step: Current training step
        max_keep: Maximum number of checkpoints to keep
    """
    checkpoint = {
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "ema": ema.state_dict(),
        "global_step": global_step,
    }
    path = output_path / f"checkpoint_{global_step}.pt"

    def _save():
        torch.save(checkpoint, path)
        # Remove old checkpoints (sorted by step number, not file name)
        prune_checkpoints(output_path, max_keep)
        print(f"Saved checkpoint at step {global_step}")

    _ckpt_executor.submit(_save)
