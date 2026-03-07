"""ODE solvers for flow matching inference."""

import torch


def euler_solve(model, z0, whisper_h, speaker_emb, num_steps=10, progress=False, language_id=None):
    """Solve the ODE from t=0 to t=1 using forward Euler method.

    More flexible standalone interface compared to FlowW2NModel.sample().
    Supports progress callback and intermediate trajectory storage.

    Args:
        model: FlowW2NModel instance
        z0: Initial noise (B, 64, T) ~ N(0, I)
        whisper_h: Whisper content features (B, T_w, 512)
        speaker_emb: Speaker embedding (B, 192)
        num_steps: Number of Euler steps (default: 10)
        progress: If True, print step progress
        language_id: Optional language index (B,) as LongTensor

    Returns:
        z1: Final predicted latent (B, 64, T)
    """
    dt = 1.0 / num_steps
    z = z0

    # Pre-generate timesteps to avoid per-step torch.full() overhead
    timesteps = torch.linspace(0, 1.0 - dt, num_steps, device=z.device, dtype=z.dtype)

    for i in range(num_steps):
        if progress:
            print(f"  Euler step {i + 1}/{num_steps} (t={timesteps[i].item():.3f})")

        t = timesteps[i].expand(z.shape[0])

        # Predict velocity at current state and timestep
        with torch.no_grad():
            v = model(z, t, whisper_h, speaker_emb, language_id=language_id)

        # Euler step: z_{t+dt} = z_t + dt * v_theta(z_t, t, c)
        z = z + dt * v

    if progress:
        print("  Euler integration complete.")

    return z


def heun_solve(model, z0, whisper_h, speaker_emb, num_steps=10, progress=False, language_id=None):
    """Solve the ODE from t=0 to t=1 using Heun's method (2nd order).

    Achieves similar quality to Euler with half the steps,
    but requires 2 model evaluations per step.

    Args:
        model: FlowW2NModel instance
        z0: Initial noise (B, 64, T) ~ N(0, I)
        whisper_h: Whisper content features (B, T_w, 512)
        speaker_emb: Speaker embedding (B, 192)
        num_steps: Number of Heun steps (default: 10)
        progress: If True, print step progress
        language_id: Optional language index (B,) as LongTensor

    Returns:
        z1: Final predicted latent (B, 64, T)
    """
    dt = 1.0 / num_steps
    z = z0

    # Pre-generate timesteps
    timesteps = torch.linspace(0, 1.0 - dt, num_steps, device=z.device, dtype=z.dtype)
    timesteps_next = timesteps + dt

    for i in range(num_steps):
        if progress:
            print(f"  Heun step {i + 1}/{num_steps} (t={timesteps[i].item():.3f})")

        t = timesteps[i].expand(z.shape[0])
        t_next = timesteps_next[i].expand(z.shape[0])

        with torch.no_grad():
            # First evaluation (predictor)
            v1 = model(z, t, whisper_h, speaker_emb, language_id=language_id)
            z_pred = z + dt * v1

            # Second evaluation (corrector)
            v2 = model(z_pred, t_next, whisper_h, speaker_emb, language_id=language_id)

        # Average of predictor and corrector
        z = z + dt * (v1 + v2) / 2

    if progress:
        print("  Heun integration complete.")

    return z


def euler_solve_with_trajectory(model, z0, whisper_h, speaker_emb, num_steps=10, language_id=None):
    """Like euler_solve but returns all intermediate states.

    Useful for visualization and analysis of the ODE trajectory.

    Args:
        model: FlowW2NModel instance
        z0: Initial noise (B, 64, T) ~ N(0, I)
        whisper_h: Whisper content features (B, T_w, 512)
        speaker_emb: Speaker embedding (B, 192)
        num_steps: Number of Euler steps (default: 10)
        language_id: Optional language index (B,) as LongTensor

    Returns:
        trajectory: List of (B, 64, T) tensors, length num_steps+1.
                    trajectory[0] = z0, trajectory[-1] = z1 (final prediction).
    """
    dt = 1.0 / num_steps
    z = z0

    trajectory = [z0.clone()]

    # Pre-generate timesteps
    timesteps = torch.linspace(0, 1.0 - dt, num_steps, device=z.device, dtype=z.dtype)

    for i in range(num_steps):
        t = timesteps[i].expand(z.shape[0])

        # Predict velocity at current state and timestep
        with torch.no_grad():
            v = model(z, t, whisper_h, speaker_emb, language_id=language_id)

        # Euler step
        z = z + dt * v

        trajectory.append(z.clone())

    return trajectory
