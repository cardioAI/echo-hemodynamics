"""CLI entrypoint: ``python -m echo_hemodynamics.runners.visualize``.

Generates temporal-rollout curves and Integrated Gradients overlays for one patient.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from ..visualization import ProgressiveAttentionVisualizer


def _find_model_path(output_base_dir):
    candidates = [
        os.path.join(output_base_dir, "..", "attention_ablation_cardioAI", "best_model.pth"),
        os.path.join(output_base_dir, "..", "attention_ablation_cardioAI", "latest_checkpoint.pth"),
        os.path.join(output_base_dir, "..", "02_ablation", "best_model.pth"),
        os.path.join(output_base_dir, "..", "train_cardioAI", "best_model.pth"),
        os.path.join(output_base_dir, "..", "train_cardioAI", "latest_checkpoint.pth"),
        os.path.join(output_base_dir, "..", "01_training", "best_model.pth"),
        os.path.join(output_base_dir, "..", "01_training", "latest_checkpoint.pth"),
        "./best_model.pth",
        "./latest_checkpoint.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        output_dir = os.environ.get("CARDIOAI_OUTPUT_DIR", "./visualization_output")
        model_path = _find_model_path(output_dir)
        dataset_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT"
        excel_file = "./All.xlsx"

        print(f"Model path: {model_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        print(f"Output directory: {output_dir}")

        n_frames = int(os.environ.get("CARDIOAI_ATTENTION_FRAMES", 32))
        print(f"Using {n_frames} frames with highest attention importance for visualizations")

        overlay_alpha_full = float(os.environ.get("CARDIOAI_OVERLAY_ALPHA_FULL", 0.7))
        overlay_alpha_selective = float(os.environ.get("CARDIOAI_OVERLAY_ALPHA_SELECTIVE", 0.8))

        visualizer = ProgressiveAttentionVisualizer(
            model_path, dataset_dir, excel_file, output_dir, device, n_frames,
            overlay_alpha_full=overlay_alpha_full,
            overlay_alpha_selective=overlay_alpha_selective,
        )

        visualizer.generate_temporal_rollout_curves()
        viz_count = visualizer.generate_attention_visualizations()

        expected_viz = visualizer.n_frames * 4 * 9 * 3 * 2
        temporal_curve_count = 9 * 3
        total_files = viz_count + temporal_curve_count
        expected_total = expected_viz + temporal_curve_count

        summary = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "patient_id": "E100017564",
            "temporal_curves": temporal_curve_count,
            "attention_visualizations": viz_count,
            "expected_visualizations": expected_viz,
            "total_files": total_files,
            "expected_total": expected_total,
            "n_frames_used": visualizer.n_frames,
            "success_rate": total_files / expected_total if expected_total > 0 else 0,
            "completion_status": "SUCCESS" if viz_count >= expected_viz else "PARTIAL",
        }

        summary_path = Path(output_dir) / "visualization_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print("ATTENTION VISUALIZATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Temporal curves generated: {summary['temporal_curves']}")
        print(f"Attention visualizations: {summary['attention_visualizations']}")
        print(f"Total files: {summary['total_files']}/{summary['expected_total']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Status: {summary['completion_status']}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'=' * 60}")

        return total_files >= expected_total * 0.8
    except Exception as e:
        print(f"Error in attention visualization generation: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
