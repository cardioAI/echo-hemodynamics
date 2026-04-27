"""5-fold cross-validation orchestration for progressive training."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from ..data import CardioAIDataset
from ..models import create_model
from .trainer import ProgressiveTrainer


def run_cross_validation(
    tensor_dir,
    excel_file,
    output_dir,
    epochs=100,
    stage_epochs=50,
    batch_size=16,
    training_frames=32,
    stages=1,
    ablation_attentions="temporal,fusion",
    num_folds=5,
    device=None,
    train_size=235,
    test_range=(235, 308),
    ph_threshold=20.0,
):
    """Run K-fold CV on patients 1..train_size with an independent test set.

    Returns the path to the final best_model.pth and a CV summary dict.
    """
    tensor_dir = Path(tensor_dir)
    excel_file = Path(excel_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    full_dataset = CardioAIDataset(
        tensor_dir=tensor_dir,
        excel_file=excel_file,
        max_frames=training_frames,
    )

    total_size = len(full_dataset)
    print(f"Total dataset size: {total_size} patients")

    print("\nAnalyzing pulmonary hypertension distribution...")
    df = pd.read_excel(excel_file)
    meanPAP_values = df["meanPAP"].values
    ph_labels = (meanPAP_values >= ph_threshold).astype(int)

    positive_indices = np.where(ph_labels == 1)[0]
    negative_indices = np.where(ph_labels == 0)[0]
    print(f"PH distribution (threshold={ph_threshold}):")
    print(f"  Total: {len(ph_labels)} patients")
    print(f"  PH Positive (>={ph_threshold}): {len(positive_indices)} patients")
    print(f"  PH Negative (<{ph_threshold}): {len(negative_indices)} patients")

    train_cv_indices = list(range(min(train_size, total_size)))
    test_indices = list(range(test_range[0], min(test_range[1], total_size)))

    print("\nDataset split configuration:")
    print(f"  Cohort I  (training + {num_folds}-fold CV): {len(train_cv_indices)} patients")
    print(f"  Cohort II (independent test):              {len(test_indices)} patients")

    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    norm_params = full_dataset.get_normalization_parameters()

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    param_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]

    best_fold = -1
    best_fold_val_loss = float("inf")
    all_fold_correlations = []
    all_fold_test_correlations = []

    print(f"\n{'=' * 80}")
    print(f"STARTING {num_folds}-FOLD CROSS-VALIDATION")
    print(f"{'=' * 80}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_cv_indices)):
        fold_train_indices = [train_cv_indices[i] for i in train_idx]
        fold_val_indices = [train_cv_indices[i] for i in val_idx]

        print(f"\n{'=' * 80}")
        print(f"FOLD {fold + 1}/{num_folds}")
        print(f"  Training: {len(fold_train_indices)} patients")
        print(f"  Validation: {len(fold_val_indices)} patients")
        print(f"{'=' * 80}")

        fold_train_dataset = torch.utils.data.Subset(full_dataset, fold_train_indices)
        fold_val_dataset = torch.utils.data.Subset(full_dataset, fold_val_indices)

        fold_train_loader = DataLoader(
            fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        fold_val_loader = DataLoader(
            fold_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        print(f"  Creating progressive CardioAI model for fold {fold + 1}...")
        fold_model = create_model(
            num_outputs=9,
            num_frames=training_frames,
            num_views=4,
            dropout_rate=0.15,
            ablation_attentions=ablation_attentions,
        )
        fold_model.set_winsorized_normalization(norm_params)

        fold_output_dir = output_dir / f"fold_{fold + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CARDIOAI_OUTPUT_DIR"] = str(fold_output_dir)

        trainer = ProgressiveTrainer(
            model=fold_model,
            train_loader=fold_train_loader,
            val_loader=fold_val_loader,
            device=device,
            stage_epochs=stage_epochs,
            task_lr=1e-4,
            vit_lr=3e-5,
            stages=stages,
            total_epochs=epochs,
            test_loader=test_loader,
        )

        trainer.train(epochs)

        fold_model_path = output_dir / f"best_model_fold{fold + 1}.pth"
        torch.save(fold_model.state_dict(), fold_model_path)
        print(f"  Fold {fold + 1} model saved to: {fold_model_path}")

        fold_val_loss = trainer.best_val_loss
        fold_corr = trainer.best_correlations if trainer.best_correlations else [0.0] * 9
        fold_test_corr = trainer.best_test_correlations if trainer.best_test_correlations else [0.0] * 9

        all_fold_correlations.append(fold_corr)
        all_fold_test_correlations.append(fold_test_corr)

        if fold_val_loss < best_fold_val_loss:
            best_fold_val_loss = fold_val_loss
            best_fold = fold

        del fold_model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.environ["CARDIOAI_OUTPUT_DIR"] = str(output_dir)

    best_fold_model_path = output_dir / f"best_model_fold{best_fold + 1}.pth"
    final_model_path = output_dir / "best_model.pth"
    shutil.copy2(best_fold_model_path, final_model_path)

    print(f"\n{'=' * 80}")
    print(f"{num_folds}-FOLD CROSS-VALIDATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Best fold: {best_fold + 1} (validation loss: {best_fold_val_loss:.4f})")
    print(f"Best model saved as: {final_model_path}")

    avg_val_corr = np.mean(all_fold_correlations, axis=0)
    std_val_corr = np.std(all_fold_correlations, axis=0)
    avg_test_corr = np.mean(all_fold_test_correlations, axis=0)
    std_test_corr = np.std(all_fold_test_correlations, axis=0)

    print(f"\nCross-Validation Results (Cohort I, n={train_size}, {num_folds} folds):")
    print(f"  {'Parameter':<10} {'Mean Corr':>10} {'Std':>8} {'Status'}")
    print(f"  {'-' * 40}")
    above_threshold = 0
    for param, avg_c, std_c in zip(param_names, avg_val_corr, std_val_corr):
        status = "[PASS]" if avg_c >= 0.6 else "[BELOW TARGET]"
        if avg_c >= 0.6:
            above_threshold += 1
        print(f"  {param:<10} {avg_c:>10.3f} {std_c:>8.3f} {status}")
    print(f"  Parameters above 0.6: {above_threshold}/9")

    print("\nIndependent Test Results (Cohort II):")
    for param, avg_c, std_c in zip(param_names, avg_test_corr, std_test_corr):
        status = "[PASS]" if avg_c >= 0.6 else "[BELOW TARGET]"
        print(f"  {param:<10} {avg_c:>10.3f} {std_c:>8.3f} {status}")

    cv_summary = {
        "num_folds": num_folds,
        "best_fold": best_fold + 1,
        "best_fold_val_loss": float(best_fold_val_loss),
        "per_fold_val_correlations": [list(map(float, fc)) for fc in all_fold_correlations],
        "per_fold_test_correlations": [list(map(float, fc)) for fc in all_fold_test_correlations],
        "avg_val_correlations": {p: float(c) for p, c in zip(param_names, avg_val_corr)},
        "std_val_correlations": {p: float(c) for p, c in zip(param_names, std_val_corr)},
        "avg_test_correlations": {p: float(c) for p, c in zip(param_names, avg_test_corr)},
        "std_test_correlations": {p: float(c) for p, c in zip(param_names, std_test_corr)},
    }
    cv_file = output_dir / "cv_summary.json"
    with open(cv_file, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\nCross-validation summary saved to: {cv_file}")

    best_fold_corr = all_fold_correlations[best_fold]
    correlations_dict = {p: float(c) for p, c in zip(param_names, best_fold_corr)}
    correlations_file = output_dir / "final_correlations.json"
    with open(correlations_file, "w") as f:
        json.dump(correlations_dict, f, indent=2)
    print(f"Final correlations (best fold) saved to: {correlations_file}")

    return final_model_path, cv_summary
