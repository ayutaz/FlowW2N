"""Euler ODE sampler for flow matching inference."""

import torch


def euler_solve(model, z0, whisper_h, speaker_emb, num_steps=10, progress=False):
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

    Returns:
        z1: Final predicted latent (B, 64, T)
    """
    dt = 1.0 / num_steps
    z = z0

    for i in range(num_steps):
        if progress:
            print(f"  Euler step {i + 1}/{num_steps} (t={i * dt:.3f})")

        t_scalar = i * dt
        t = torch.full(
            (z.shape[0],), t_scalar, device=z.device, dtype=z.dtype
        )

        # Predict velocity at current state and timestep
        with torch.no_grad():
            v = model(z, t, whisper_h, speaker_emb)

        # Euler step: z_{t+dt} = z_t + dt * v_theta(z_t, t, c)
        z = z + dt * v

    if progress:
        print("  Euler integration complete.")

    return z


def euler_solve_with_trajectory(model, z0, whisper_h, speaker_emb, num_steps=10):
    """Like euler_solve but returns all intermediate states.

    Useful for visualization and analysis of the ODE trajectory.

    Args:
        model: FlowW2NModel instance
        z0: Initial noise (B, 64, T) ~ N(0, I)
        whisper_h: Whisper content features (B, T_w, 512)
        speaker_emb: Speaker embedding (B, 192)
        num_steps: Number of Euler steps (default: 10)

    Returns:
        trajectory: List of (B, 64, T) tensors, length num_steps+1.
                    trajectory[0] = z0, trajectory[-1] = z1 (final prediction).
    """
    dt = 1.0 / num_steps
    z = z0

    trajectory = [z0.clone()]

    for i in range(num_steps):
        t_scalar = i * dt
        t = torch.full(
            (z.shape[0],), t_scalar, device=z.device, dtype=z.dtype
        )

        # Predict velocity at current state and timestep
        with torch.no_grad():
            v = model(z, t, whisper_h, speaker_emb)

        # Euler step
        z = z + dt * v

        trajectory.append(z.clone())

    return trajectory
