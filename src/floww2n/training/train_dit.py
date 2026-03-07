"""DiT training script with conditional flow matching.

Trains the FlowW2NModel (DiT backbone + speaker projection) using:
- Conditional Flow Matching (CFM) loss with Gaussian noise prior
- EMA (Exponential Moving Average) weight tracking
- Mixed precision (bf16)
- Gradient accumulation and gradient clipping
- Linear warmup + constant learning rate schedule
- Periodic checkpointing and wandb logging
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import wandb
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from floww2n.models.floww2n import FlowW2NModel
from floww2n.training.dataset import DiTDataset, dit_collate_fn

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


def get_lr(step, warmup_steps, base_lr):
    """Compute learning rate with linear warmup then constant.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        base_lr: Target learning rate after warmup

    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def train_dit(config_path, cache_dir, output_dir, vae_checkpoint=None, resume_from=None):
    """Train DiT model with Conditional Flow Matching.

    Args:
        config_path: Path to dit.json config
        cache_dir: Path to cached features directory (from cache_features.py)
        output_dir: Checkpoint output directory
        vae_checkpoint: Path to trained VAE (for validation decoding, optional)
        resume_from: Path to checkpoint to resume from (optional)
    """
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    cond_cfg = config.get("conditioning", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_cuda_optimizations()

    # Model
    speaker_dim = cond_cfg.get("speaker_encoder", {}).get("output_dim", 192)
    model = FlowW2NModel(
        dit_config=model_cfg,
        speaker_dim=speaker_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # torch.compile
    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Optimizer (single AdamW, no discriminator)
    fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        fused=fused,
    )

    # EMA
    ema = EMA(model, beta=train_cfg["ema_decay"])

    # Dataset & DataLoader
    # Compute max_latent_length from max_audio_length if available
    max_audio_length = train_cfg.get("max_audio_length", None)
    max_latent_length = None
    if max_audio_length is not None:
        # VAE compression ratio is 1024 (strides: 4*4*8*8)
        max_latent_length = max_audio_length // 1024

    # Build language_map from config if multilingual
    languages = model_cfg.get("languages", None)
    language_map = None
    if languages and len(languages) > 1:
        language_map = {lang: idx for idx, lang in enumerate(languages)}
        print(f"Multilingual training: {language_map}")

    dataset = DiTDataset(
        cache_dir=cache_dir,
        max_latent_length=max_latent_length,
        language_map=language_map,
    )
    num_workers = train_cfg["num_workers"]
    loader_kwargs = dict(
        batch_size=train_cfg["micro_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dit_collate_fn,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    dataloader = DataLoader(dataset, **loader_kwargs)

    # Mixed precision
    use_amp = train_cfg["mixed_precision"] == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Training loop setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grad_accum = train_cfg["gradient_accumulation_steps"]
    grad_clip_norm = train_cfg.get("gradient_clip_norm", 1.0)
    warmup_steps = train_cfg.get("warmup_steps", 2000)
    base_lr = train_cfg["learning_rate"]
    global_step = 0
    max_steps = train_cfg["max_steps"]

    # Resume from checkpoint
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        global_step = checkpoint["global_step"]
        print(f"Resumed from checkpoint at step {global_step}")

    model.train()

    print(f"Starting training from step {global_step} to {max_steps}")
    print(f"Dataset: {len(dataset)} samples")
    print(
        f"Micro batch size: {train_cfg['micro_batch_size']}, "
        f"Gradient accumulation: {grad_accum}, "
        f"Effective batch size: {train_cfg['micro_batch_size'] * grad_accum}"
    )
    print(f"Device: {device}")
    print(f"Mixed precision: {train_cfg['mixed_precision']}")

    # Save config to output dir for reference
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            # Move to device
            z1 = batch["z1"].to(device)  # (B, 64, T)
            whisper_h = batch["whisper_h"].to(device)  # (B, T_w, 512)
            speaker_emb = batch["speaker_emb"].to(device)  # (B, 192)
            z1_mask = batch["z1_mask"].to(device)  # (B, T)
            language_id = batch.get("language_id")
            if language_id is not None:
                language_id = language_id.to(device)

            # Update learning rate (linear warmup + constant)
            lr = get_lr(global_step, warmup_steps, base_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass with CFM loss
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss, loss_dict = model.compute_loss(
                    z1, whisper_h, speaker_emb, z1_mask=z1_mask, language_id=language_id
                )
                loss = loss / grad_accum

            # Backward pass
            loss.backward()

            # Optimizer step at accumulation boundary
            if (global_step + 1) % grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update()

            # Logging
            if global_step % 100 == 0:
                cfm_loss = loss_dict.get("cfm_loss", loss * grad_accum).item()
                log_dict = {
                    "train/cfm_loss": cfm_loss,
                    "train/learning_rate": lr,
                    "train/global_step": global_step,
                }
                print(f"Step {global_step}/{max_steps} | CFM Loss: {cfm_loss:.6f} | LR: {lr:.2e}")
                if wandb.run:
                    wandb.log(log_dict, step=global_step)

            # Checkpoint
            if global_step > 0 and global_step % 10000 == 0:
                save_checkpoint(output_path, model, optimizer, ema, global_step)

            global_step += 1

    # Final save
    save_checkpoint(output_path, model, optimizer, ema, global_step)
    # Save EMA weights separately
    torch.save(ema.ema_model.state_dict(), output_path / "dit_ema_final.pt")
    print(f"Training complete. Final step: {global_step}")


def save_checkpoint(output_path, model, optimizer, ema, global_step, max_keep=3):
    """Save training checkpoint asynchronously.

    Args:
        output_path: Directory to save checkpoint
        model: FlowW2NModel
        optimizer: AdamW optimizer
        ema: EMA wrapper
        global_step: Current training step
        max_keep: Maximum number of checkpoints to keep
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "global_step": global_step,
    }
    path = output_path / f"checkpoint_{global_step}.pt"

    def _save():
        torch.save(checkpoint, path)
        # Remove old checkpoints
        ckpts = sorted(output_path.glob("checkpoint_*.pt"))
        for old in ckpts[:-max_keep]:
            old.unlink()
        print(f"Saved checkpoint at step {global_step}")

    _ckpt_executor.submit(_save)
