"""Checkpoint file helpers shared by the VAE and DiT training loops."""

from pathlib import Path


def sorted_checkpoints(output_path: Path) -> list[Path]:
    """Return ``checkpoint_<step>.pt`` files in ``output_path`` sorted by step (oldest first).

    Sorting by the numeric step (not the file name) keeps the order correct once the
    step count gains a digit (e.g. ``checkpoint_100000.pt`` is newer than ``checkpoint_95000.pt``).
    Files whose suffix is not an integer step are ignored.
    """
    ckpts = []
    for path in Path(output_path).glob("checkpoint_*.pt"):
        step = path.stem.removeprefix("checkpoint_")
        if step.isdigit():
            ckpts.append((int(step), path))
    return [path for _, path in sorted(ckpts)]


def prune_checkpoints(output_path: Path, max_keep: int) -> None:
    """Delete all but the ``max_keep`` most recent ``checkpoint_<step>.pt`` files."""
    if max_keep <= 0:  # keep everything (same as the previous ``ckpts[:-0]`` behaviour)
        return
    for old in sorted_checkpoints(output_path)[:-max_keep]:
        old.unlink()
