"""Data-splitting utilities: index parsing and PH-stratified train/test splits."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_train_indices(indices_str):
    """Parse comma-separated index ranges (e.g., "0-179,180-224,235-289") into a list."""
    if not indices_str:
        return None

    indices = []
    for range_str in indices_str.split(","):
        range_str = range_str.strip()
        if "-" in range_str:
            start, end = range_str.split("-")
            indices.extend(list(range(int(start), int(end) + 1)))
        else:
            indices.append(int(range_str))
    return indices


def create_balanced_ph_splits(excel_file: Path,
                              test_size: int = 55,
                              threshold: float = 20.0,
                              random_state: int = 42,
                              strategy: str = "undersample_majority") -> Tuple[List[int], List[int], Dict]:
    """Clinically-representative train/test split for pulmonary hypertension classification.

    Test set uses 75/25 positive/negative split to reflect real-world PH prevalence.
    """
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")

    df = pd.read_excel(excel_file)
    meanPAP_values = df["meanPAP"].values

    ph_labels = (meanPAP_values >= threshold).astype(int)

    positive_indices = np.where(ph_labels == 1)[0]
    negative_indices = np.where(ph_labels == 0)[0]

    print("Pulmonary Hypertension distribution:")
    print(f"  Total patients: {len(df)}")
    print(f"  PH Positive (>={threshold}): {len(positive_indices)} ({len(positive_indices) / len(df) * 100:.1f}%)")
    print(f"  PH Negative (<{threshold}): {len(negative_indices)} ({len(negative_indices) / len(df) * 100:.1f}%)")

    test_positive = min(int(test_size * 0.75), len(positive_indices))
    test_negative = min(test_size - test_positive, len(negative_indices))

    if test_negative < int(test_size * 0.25):
        remaining_slots = test_size - test_negative
        test_positive = min(remaining_slots, len(positive_indices))

    print(f"Test set composition ({test_positive + test_negative} patients):")
    print(f"  PH Positive: {test_positive}")
    print(f"  PH Negative: {test_negative}")

    np.random.seed(random_state)
    test_pos_indices = np.random.choice(positive_indices, test_positive, replace=False)
    test_neg_indices = np.random.choice(negative_indices, test_negative, replace=False)
    test_indices = np.concatenate([test_pos_indices, test_neg_indices])

    train_pos_indices = np.setdiff1d(positive_indices, test_pos_indices)
    train_neg_indices = np.setdiff1d(negative_indices, test_neg_indices)

    if strategy == "undersample_majority":
        min_class_size = min(len(train_pos_indices), len(train_neg_indices))
        train_pos_balanced = np.random.choice(train_pos_indices, min_class_size, replace=False)
        train_neg_balanced = np.random.choice(train_neg_indices, min_class_size, replace=False)
        train_indices = np.concatenate([train_pos_balanced, train_neg_balanced])
    elif strategy == "oversample_minority":
        max_class_size = max(len(train_pos_indices), len(train_neg_indices))
        train_pos_balanced = np.random.choice(train_pos_indices, max_class_size, replace=True)
        train_neg_balanced = np.random.choice(train_neg_indices, max_class_size, replace=True)
        train_indices = np.concatenate([train_pos_balanced, train_neg_balanced])
    elif strategy == "stratified":
        train_indices = np.concatenate([train_pos_indices, train_neg_indices])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    info = {
        "strategy": strategy,
        "threshold": threshold,
        "random_state": random_state,
        "total_patients": len(df),
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "train_positive": int(np.sum(ph_labels[train_indices] == 1)),
        "train_negative": int(np.sum(ph_labels[train_indices] == 0)),
        "test_positive": int(np.sum(ph_labels[test_indices] == 1)),
        "test_negative": int(np.sum(ph_labels[test_indices] == 0)),
        "original_positive": len(positive_indices),
        "original_negative": len(negative_indices),
    }

    print(f"Training set composition ({len(train_indices)} patients):")
    print(f"  PH Positive: {info['train_positive']} ({info['train_positive'] / len(train_indices) * 100:.1f}%)")
    print(f"  PH Negative: {info['train_negative']} ({info['train_negative'] / len(train_indices) * 100:.1f}%)")
    if info["train_negative"] > 0:
        print(f"  Balance ratio: {info['train_positive'] / info['train_negative']:.2f}")

    return train_indices.tolist(), test_indices.tolist(), info
