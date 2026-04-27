"""Checkpoint discovery utilities for downstream stages (ablation, validation, test)."""

from pathlib import Path


def find_latest_trained_model(base_dir=r"E:\results_cardioAI\EchoCath_cardioAI", current_timestamp=None):
    """Find the latest training run that contains a best_model.pth.

    Returns the path to best_model.pth, or raises FileNotFoundError if none exist.
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    timestamp_dirs = [d for d in base.iterdir() if d.is_dir() and d.name != current_timestamp]
    if not timestamp_dirs:
        raise FileNotFoundError("No existing training results found")

    latest_dir = sorted(timestamp_dirs, key=lambda x: x.name)[-1]

    candidates = [
        latest_dir / "train_cardioAI" / "best_model.pth",
        latest_dir / "best_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No best_model.pth found under {latest_dir}")
