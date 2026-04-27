"""Ablation study orchestration: load pre-trained full model, train variants, save results."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from ..data import CardioAIDataset, parse_train_indices
from ..figures.ablation_plots import create_ablation_plots
from ..models import create_model
from ..training.checkpoints import find_latest_trained_model
from ..utils.singleton import CardioAIUtils, ColorManager
from .factory import create_ablation_variants
from .trainer import ProgressiveAblationTrainer


def run_ablation_study():
    print("=" * 80)
    print("PROGRESSIVE CARDIOAI ATTENTION ABLATION STUDY")
    print("=" * 80)

    # Default to main-training epoch count so ablation and main runs match.
    epochs = int(os.environ.get("CARDIOAI_ABLATION_EPOCHS", os.environ.get("CARDIOAI_EPOCHS", 100)))
    batch_size = int(os.environ.get("CARDIOAI_ABLATION_BATCH_SIZE", 16))
    num_patients = int(os.environ.get("CARDIOAI_ABLATION_PATIENTS", 308))
    training_frames = int(os.environ.get("CARDIOAI_TRAINING_FRAMES", 32))
    output_dir = Path(os.environ.get("CARDIOAI_OUTPUT_DIR", "."))

    print("Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patients: {num_patients}")
    print(f"  Training frames: {training_frames}")
    print(f"  Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if not output_dir.exists():
        raise RuntimeError(f"Output directory {output_dir} should be created by main.py first")

    cardio_utils = CardioAIUtils()
    cardio_utils.current_output_dir = output_dir
    cardio_utils.subdirs = {"figures": output_dir}

    print("\nLoading dataset...")
    tensor_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT")
    excel_file = Path("All.xlsx")

    full_dataset = CardioAIDataset(
        tensor_dir=tensor_dir, excel_file=excel_file, max_frames=training_frames
    )

    norm_params = full_dataset.get_normalization_parameters()
    print(f"Normalization parameters: log_transform_indices={norm_params['log_transform_indices']}")

    train_cv_indices = list(range(min(235, len(full_dataset))))
    train_indices_str = os.environ.get("CARDIOAI_TRAIN_INDICES", None)

    if train_indices_str:
        indices = parse_train_indices(train_indices_str)
        Subset(full_dataset, indices)
        print(f"Using custom training indices: {train_indices_str}")
    elif num_patients < len(train_cv_indices):
        indices = list(range(num_patients))
        Subset(full_dataset, indices)
        print(f"Using patients 1-{num_patients} for ablation")
    else:
        Subset(full_dataset, train_cv_indices)
        print(f"Using patients 1-235 ({len(train_cv_indices)} patients) for ablation")

    num_folds = int(os.environ.get("CARDIOAI_NUM_FOLDS", 5))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_splits = list(kf.split(train_cv_indices))
    train_idx, val_idx = fold_splits[0]

    fold_train_indices = [train_cv_indices[i] for i in train_idx]
    fold_val_indices = [train_cv_indices[i] for i in val_idx]

    train_dataset = Subset(full_dataset, fold_train_indices)
    val_dataset = Subset(full_dataset, fold_val_indices)

    df = pd.read_excel(excel_file)
    meanPAP_values = df["meanPAP"].values
    ph_threshold = 20.0
    ph_labels = (meanPAP_values >= ph_threshold).astype(int)
    val_ph_labels = ph_labels[fold_val_indices]
    val_ph_positive = int(np.sum(val_ph_labels == 1))
    val_ph_negative = int(np.sum(val_ph_labels == 0))

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} validation (fold 1/{num_folds})")
    print(f"  Validation PH distribution: {val_ph_positive} PH+ / {val_ph_negative} PH-")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("\nLoading base progressive model...")
    base_model = create_model(
        num_outputs=9, num_frames=training_frames, num_views=4, dropout_rate=0.15
    )
    base_model.set_winsorized_normalization(norm_params)

    current_timestamp = os.environ.get("CARDIOAI_TIMESTAMP", "")
    model_path = find_latest_trained_model(current_timestamp=current_timestamp)

    try:
        if model_path.suffix == ".pth" and "checkpoint" not in model_path.name:
            state_dict = torch.load(model_path, map_location=device)
            missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
        else:
            checkpoint = torch.load(model_path, map_location=device)
            missing_keys, unexpected_keys = base_model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )

        if missing_keys or unexpected_keys:
            print("Model loading details:")
            print(f"  Missing keys: {len(missing_keys)}")
            print(f"  Unexpected keys: {len(unexpected_keys)}")
        print(f"Loaded pre-trained weights from: {model_path}")
    except Exception as e:
        print(f"ERROR: Failed to load pre-trained weights from {model_path}: {e}")
        raise

    base_model.eval()

    print("\nCreating ablation variants...")
    variants = create_ablation_variants(base_model, norm_params)

    results = {}
    for variant_name, model in variants.items():
        print(f"\n{'=' * 60}")
        print(f"TRAINING VARIANT: {variant_name.upper()}")
        print(f"{'=' * 60}")

        if variant_name == "full_model":
            print("Using pre-trained full model — evaluating only")
            trainer = ProgressiveAblationTrainer(model, train_loader, val_loader, device, epochs=1)
            _, val_corr = trainer.validate_epoch()
            history = {
                "train_loss": [0.0],
                "val_loss": [0.0],
                "val_correlations": [np.mean(val_corr)],
            }
            final_correlations = val_corr
        else:
            trainer = ProgressiveAblationTrainer(model, train_loader, val_loader, device, epochs)
            history = trainer.train()
            _, final_correlations = trainer.validate_epoch()

        results[variant_name] = {
            "history": history,
            "final_correlations": final_correlations,
            "avg_correlation": float(np.mean(final_correlations)),
        }
        print(f"Final correlation: {np.mean(final_correlations):.3f}")

    print(f"\n{'=' * 60}")
    print("SAVING ABLATION RESULTS")
    print(f"{'=' * 60}")

    summary = {
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_patients": num_patients,
            "training_frames": training_frames,
            "device": str(device),
        },
        "results": {},
    }

    param_names = ["mPAP", "RAP", "SPAP", "DPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
    param_order = [3, 0, 1, 2, 4, 5, 6, 7, 8]

    for variant_name, result in results.items():
        reordered = [result["final_correlations"][idx] for idx in param_order]
        summary["results"][variant_name] = {
            "avg_correlation": result["avg_correlation"],
            "per_param_correlations": {param_names[i]: reordered[i] for i in range(len(param_names))},
            "training_history": result["history"],
        }
        result["reordered_correlations"] = reordered
        print(f"{variant_name:15s}: {result['avg_correlation']:.3f}")

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    create_ablation_plots(results, output_dir, param_names, cardio_utils)

    print(f"\nResults saved to: {output_dir}")
    return summary
