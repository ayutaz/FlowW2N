#!/usr/bin/env python
"""Smoke test: run the full training pipeline on a small JVS subset.

Steps:
  1. Preprocess: copy 2 speakers x 100 utterances, resample to 16kHz mono
  2. VAE training: small model (channels=64), 100 steps, bf16
  3. Feature caching: Whisper + ECAPA + VAE latent, language=ja
  4. DiT training: small model (depth=2, embed_dim=128), 100 steps, bf16

Usage:
  uv run python scripts/smoke_test_training.py \
      --jvs-dir "C:\\Users\\yuta\\Downloads\\jvs_ver1\\jvs_ver1" \
      --output-dir outputs/smoke_test
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torchaudio


# ---------------------------------------------------------------------------
# Step 1: Preprocess JVS data
# ---------------------------------------------------------------------------
def preprocess_jvs(jvs_dir: str, output_dir: str, speakers=("jvs001", "jvs002")):
    """Copy and resample JVS parallel100 data to 16kHz mono."""
    jvs_path = Path(jvs_dir)
    out_path = Path(output_dir) / "preprocessed"
    out_path.mkdir(parents=True, exist_ok=True)

    target_sr = 16000
    total = 0

    for spk in speakers:
        wav_dir = jvs_path / spk / "parallel100" / "wav24kHz16bit"
        if not wav_dir.exists():
            print(f"WARNING: {wav_dir} not found, skipping {spk}")
            continue

        spk_out = out_path / spk
        spk_out.mkdir(parents=True, exist_ok=True)

        for wav_file in sorted(wav_dir.glob("*.wav")):
            audio, sr = torchaudio.load(wav_file)
            # Mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            # Resample
            if sr != target_sr:
                audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
            # Save
            out_file = spk_out / wav_file.name
            torchaudio.save(str(out_file), audio, target_sr)
            total += 1

    print(f"Preprocessed {total} files to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Step 2: VAE training
# ---------------------------------------------------------------------------
def train_vae_smoke(data_dir: str, output_dir: str, max_steps: int = 100):
    """Train a small VAE model for a few steps."""
    vae_config = {
        "model": {
            "sample_rate": 16000,
            "io_channels": 1,
            "latent_dim": 64,
            "encoder_latent_dim": 128,
            "channels": 64,
            "c_mults": [1, 2, 4, 8],
            "strides": [4, 4, 8, 8],
            "use_snake": True,
        },
        "training": {
            "batch_size": 4,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "learning_rate": 1.5e-4,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "ema_decay": 0.9999,
            "mixed_precision": "bf16",
            "segment_length": 32768,
            "num_workers": 0,
        },
        "losses": {
            "multi_resolution_stft_weight": 1.0,
            "adversarial_weight": 0.1,
            "feature_matching_weight": 5.0,
            "kl_weight": 1e-4,
        },
    }

    vae_out = Path(output_dir) / "vae"
    config_path = vae_out / "smoke_vae_config.json"
    vae_out.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(vae_config, f, indent=2)

    from floww2n.training.train_vae import train_vae

    train_vae(str(config_path), data_dir, str(vae_out))

    ema_path = vae_out / "vae_ema_final.pt"
    assert ema_path.exists(), f"VAE EMA checkpoint not found: {ema_path}"
    print(f"VAE training done. EMA checkpoint: {ema_path}")
    return str(ema_path), str(config_path)


# ---------------------------------------------------------------------------
# Step 3: Feature caching
# ---------------------------------------------------------------------------
def cache_features_smoke(data_dir: str, output_dir: str, vae_checkpoint: str, vae_config_path: str):
    """Cache Whisper + ECAPA + VAE latent features."""
    cache_dir = str(Path(output_dir) / "cache")

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "cache_features_mod",
        Path(__file__).parent / "cache_features.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cache_features_mod"] = mod
    spec.loader.exec_module(mod)
    cache_features = mod.cache_features

    cache_features(
        data_dir=data_dir,
        output_dir=cache_dir,
        vae_checkpoint=vae_checkpoint,
        config_path=vae_config_path,
        sample_rate=16000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        language="ja",
    )

    manifest_path = Path(cache_dir) / "manifest.json"
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Cached {len(manifest)} entries. Manifest: {manifest_path}")
    return cache_dir


# ---------------------------------------------------------------------------
# Step 4: DiT training
# ---------------------------------------------------------------------------
def train_dit_smoke(cache_dir: str, output_dir: str, max_steps: int = 100):
    """Train a small DiT model for a few steps."""
    dit_config = {
        "model": {
            "io_channels": 64,
            "embed_dim": 128,
            "depth": 2,
            "num_heads": 4,
            "head_dim": 32,
            "cond_token_dim": 512,
            "global_cond_dim": 128,
            "global_cond_type": "adaLN",
            "cross_attend": True,
            "diffusion_objective": "rectified_flow",
            "num_languages": 1,
            "languages": ["ja"],
        },
        "conditioning": {
            "content_encoder": {
                "model_name": "openai/whisper-base",
                "layer_index": 6,
                "output_dim": 512,
                "frame_rate_hz": 50,
                "frozen": True,
            },
            "speaker_encoder": {
                "model_name": "speechbrain/spkrec-ecapa-voxceleb",
                "output_dim": 192,
                "frozen": True,
            },
        },
        "training": {
            "max_steps": max_steps,
            "batch_size": 4,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.5e-4,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "ema_decay": 0.9999,
            "mixed_precision": "bf16",
            "max_audio_length": 160000,
            "num_workers": 0,
            "gradient_clip_norm": 1.0,
        },
        "inference": {
            "solver": "euler",
            "num_steps": 10,
        },
    }

    dit_out = Path(output_dir) / "dit"
    config_path = dit_out / "smoke_dit_config.json"
    dit_out.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(dit_config, f, indent=2)

    from floww2n.training.train_dit import train_dit

    train_dit(str(config_path), cache_dir, str(dit_out))

    ema_path = dit_out / "dit_ema_final.pt"
    assert ema_path.exists(), f"DiT EMA checkpoint not found: {ema_path}"
    print(f"DiT training done. EMA checkpoint: {ema_path}")
    return str(ema_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Smoke test: full training pipeline on JVS subset")
    parser.add_argument(
        "--jvs-dir",
        type=str,
        required=True,
        help="Path to JVS root (e.g. jvs_ver1/jvs_ver1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/smoke_test",
        help="Output directory for all artifacts",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=["jvs001", "jvs002"],
        help="Speaker IDs to use",
    )
    parser.add_argument("--vae-steps", type=int, default=100, help="VAE training steps")
    parser.add_argument("--dit-steps", type=int, default=100, help="DiT training steps")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use existing preprocessed data)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    start_time = time.time()

    print("=" * 60)
    print("FlowW2N Smoke Test: Full Training Pipeline")
    print("=" * 60)
    print(f"JVS dir: {args.jvs_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Speakers: {args.speakers}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Step 1: Preprocess
    print("=" * 60)
    print("Step 1/4: Preprocessing JVS data")
    print("=" * 60)
    preprocess_dir = str(Path(output_dir) / "preprocessed")
    if args.skip_preprocess and Path(preprocess_dir).exists():
        print(f"Skipping preprocessing, using existing: {preprocess_dir}")
    else:
        preprocess_dir = preprocess_jvs(args.jvs_dir, output_dir, speakers=args.speakers)
    t1 = time.time()
    print(f"Step 1 done in {t1 - start_time:.1f}s\n")

    # Step 2: VAE training
    print("=" * 60)
    print(f"Step 2/4: VAE training ({args.vae_steps} steps)")
    print("=" * 60)
    vae_ckpt, vae_config = train_vae_smoke(preprocess_dir, output_dir, max_steps=args.vae_steps)
    t2 = time.time()
    print(f"Step 2 done in {t2 - t1:.1f}s\n")

    # Step 3: Feature caching
    print("=" * 60)
    print("Step 3/4: Feature caching (Whisper + ECAPA + VAE)")
    print("=" * 60)
    cache_dir = cache_features_smoke(preprocess_dir, output_dir, vae_ckpt, vae_config)
    t3 = time.time()
    print(f"Step 3 done in {t3 - t2:.1f}s\n")

    # Step 4: DiT training
    print("=" * 60)
    print(f"Step 4/4: DiT training ({args.dit_steps} steps)")
    print("=" * 60)
    dit_ckpt = train_dit_smoke(cache_dir, output_dir, max_steps=args.dit_steps)
    t4 = time.time()
    print(f"Step 4 done in {t4 - t3:.1f}s\n")

    # Summary
    total_time = t4 - start_time
    print("=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print()
    print("Artifacts:")
    artifacts = {
        "VAE EMA checkpoint": vae_ckpt,
        "Feature cache": cache_dir,
        "DiT EMA checkpoint": dit_ckpt,
    }
    all_ok = True
    for name, path in artifacts.items():
        exists = Path(path).exists()
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status}] {name}: {path}")

    manifest_path = Path(cache_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  [OK] Manifest entries: {len(manifest)}")
    else:
        print("  [MISSING] Manifest")
        all_ok = False

    print()
    if all_ok:
        print("Result: ALL PASSED")
    else:
        print("Result: SOME CHECKS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
