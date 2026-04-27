"""DataLoader factory + statistical helpers (winsorize, correlation)."""

import numpy as np
import torch


def winsorize_parameter(values, lower_percentile=5, upper_percentile=95):
    """Clip values to given percentile bounds. Returns (clipped, n_lower, n_upper, lo, hi)."""
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)

    n_lower_outliers = np.sum(values < lower_bound)
    n_upper_outliers = np.sum(values > upper_bound)

    winsorized = np.clip(values, lower_bound, upper_bound)
    return winsorized, n_lower_outliers, n_upper_outliers, lower_bound, upper_bound


def calculate_correlation(pred, true):
    """Pearson correlation between pred and true arrays, NaN-safe."""
    if len(pred) != len(true):
        return 0.0

    pred = np.array(pred).flatten()
    true = np.array(true).flatten()

    valid_mask = ~(np.isnan(pred) | np.isnan(true))
    if not np.any(valid_mask):
        return 0.0

    pred = pred[valid_mask]
    true = true[valid_mask]

    if len(pred) < 2:
        return 0.0
    if np.var(pred) == 0 or np.var(true) == 0:
        return 0.0

    try:
        correlation = np.corrcoef(pred, true)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except Exception:
        return 0.0


def create_dataloaders(dataset_dir, excel_file, batch_size=2, train_split=1.0,
                       max_frames=32, subset_size=None, num_workers=0):
    """Create train/val dataloaders. Returns (train_loader, val_loader, stats)."""
    from torch.utils.data import DataLoader, random_split

    from .dataset import CardioAIDataset

    full_dataset = CardioAIDataset(
        tensor_dir=dataset_dir,
        excel_file=excel_file,
        max_frames=max_frames,
        subset_size=subset_size,
        cache_tensors=True,
    )

    full_dataset.print_dataset_info()

    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = max(1, dataset_size - train_size)

    if train_split >= 1.0:
        train_indices = list(range(min(235, dataset_size)))
        val_size_cv = len(train_indices) // 5
        val_indices = train_indices[-val_size_cv:]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        print(
            f"\nCardioAI split: {len(train_indices)} training (patients 1-235), "
            f"{len(val_indices)} validation"
        )
    else:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"\nDataset split: {train_size} training, {val_size} validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, full_dataset.get_dataset_statistics()
