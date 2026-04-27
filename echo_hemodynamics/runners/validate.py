"""CLI entrypoint: ``python -m echo_hemodynamics.runners.validate`` (Cohort I, n=235)."""

import json
import os
import sys
import traceback
from pathlib import Path

import torch

from ..analysis.bland_altman import render_plain_bland_altman
from ..analysis.confusion_matrix import render_confusion_matrices
from ..analysis.correlation_plots import render_correlation_bar, render_correlation_heatmap
from ..analysis.embeddings import render_embeddings
from ..analysis.excel_report import generate_summary_report
from ..analysis.inference import denormalize, generate_model_predictions
from ..analysis.metrics import PARAM_NAMES
from ..analysis.roc_auc import render_simple_roc
from ..analysis.scatter_plots import render_plain_scatter
from ..data import CardioAIDataset
from ..figures.training_curves import generate_training_figures as _generate_training_figures
from ..models import create_model
from ..utils.singleton import CardioAIUtils


def _find_model_path(output_dir):
    candidates = [
        os.path.join(output_dir, "..", "train_cardioAI", "best_model.pth"),
        os.path.join(output_dir, "..", "train_cardioAI", "latest_checkpoint.pth"),
        os.path.join(output_dir, "..", "attention_ablation_cardioAI", "best_model.pth"),
        "./best_model.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_training_history(current_timestamp):
    base_dir = Path(r"E:\results_cardioAI\EchoCath_cardioAI")
    if current_timestamp:
        current_training_dir = base_dir / current_timestamp / "train_cardioAI"
        if (current_training_dir / "training_history.json").exists():
            with open(current_training_dir / "training_history.json", "r") as f:
                return json.load(f)

    if base_dir.exists():
        for result_dir in sorted([d for d in base_dir.glob("*") if d.is_dir()], reverse=True):
            training_dir = result_dir / "train_cardioAI"
            if (training_dir / "training_history.json").exists():
                with open(training_dir / "training_history.json", "r") as f:
                    return json.load(f)

    return None


def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        output_dir = Path(os.environ.get("CARDIOAI_OUTPUT_DIR", "./validation_output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        cardio_utils = CardioAIUtils()
        cardio_utils.current_output_dir = output_dir
        cardio_utils.subdirs = {
            "figures": output_dir / "figures",
            "embeddings": output_dir / "embeddings",
            "tables": output_dir / "tables",
        }
        for sub in cardio_utils.subdirs.values():
            sub.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}")

        model_path = _find_model_path(str(output_dir))
        if model_path is None:
            raise RuntimeError("No trained model found. Please run training first.")
        print(f"Loading model from {model_path}")

        tensor_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT"
        excel_file = "./All.xlsx"
        training_frames = int(os.environ.get("CARDIOAI_TRAINING_FRAMES", 32))

        temp_dataset = CardioAIDataset(
            tensor_dir, excel_file, max_frames=training_frames, subset_size=10
        )
        norm_params = temp_dataset.get_normalization_parameters()

        model = create_model(num_outputs=9, num_frames=training_frames, num_views=4)
        model.set_winsorized_normalization(norm_params)

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if (
            isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        ) else checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(
                f"Warning: Model architecture mismatch. "
                f"Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
            )

        model = model.to(device)
        model.eval()

        # Training-curve plots
        history = _load_training_history(os.environ.get("CARDIOAI_TIMESTAMP", ""))
        if history is not None:
            _generate_training_figures(history, output_dir / "figures")

        # Inference on Cohort I (patients 1-235, indices 0-234)
        indices = list(range(235))
        predictions, targets, patient_ids = generate_model_predictions(
            model, device, indices, tensor_dir, excel_file, training_frames=training_frames,
        )

        if predictions is None:
            print("No valid predictions generated")
            return False

        pred_denorm = denormalize(model, device, predictions)

        render_correlation_bar(
            pred_denorm, targets, PARAM_NAMES, cardio_utils,
            "validation_parameter_correlations",
            "Validation: Model Performance by Parameter",
            color_manager=cardio_utils,
        )
        render_correlation_heatmap(
            pred_denorm, PARAM_NAMES, cardio_utils,
            "validation_correlation_heatmap",
            "Validation: Parameter Correlation Matrix",
        )
        render_plain_scatter(pred_denorm, targets, PARAM_NAMES, cardio_utils, cardio_utils, "validation")
        render_plain_bland_altman(pred_denorm, targets, PARAM_NAMES, cardio_utils, cardio_utils, "validation")
        render_simple_roc(pred_denorm, targets, PARAM_NAMES, cardio_utils, cardio_utils, "validation")
        render_confusion_matrices(pred_denorm, targets, PARAM_NAMES, cardio_utils, "validation")
        render_embeddings(
            pred_denorm, targets, patient_ids, cardio_utils,
            output_dir / "embeddings", "validation",
        )

        generate_summary_report(
            pred_denorm, targets, patient_ids, PARAM_NAMES,
            output_dir / "tables" / "validation_summary.xlsx",
            dataset_label="Validation (Cohort I, n=235, 5-fold CV cohort)",
            include_components=True,
        )

        print(f"\n{'=' * 60}")
        print("INTERNAL VALIDATION COMPLETED")
        print(f"{'=' * 60}")
        print(f"Patients evaluated: {len(patient_ids) if patient_ids else 0}")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 60}")
        return True

    except Exception as e:
        print(f"Error in validation: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
