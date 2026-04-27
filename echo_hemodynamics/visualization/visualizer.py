"""ProgressiveAttentionVisualizer: orchestrates rollout + Integrated Gradients renderings."""

import os
import warnings
from pathlib import Path

import torch

from ..data import CardioAIDataset
from ..models import create_model
from ..utils.singleton import ColorManager
from .attention_rollout import render_temporal_rollout_curves
from .integrated_gradients import render_ig_visualizations

warnings.filterwarnings("ignore", category=UserWarning, message=".*PostScript.*transparency.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*partially transparent artists.*")


class ProgressiveAttentionVisualizer:
    """Loads a trained model and emits temporal-rollout curves + per-frame IG overlays."""

    def __init__(self, model_path, dataset_dir, excel_file, output_dir, device, n_frames=8,
                 overlay_alpha_full=0.7, overlay_alpha_selective=0.8):
        self.model_path = Path(model_path)
        self.dataset_dir = dataset_dir
        self.excel_file = excel_file
        self.output_dir = Path(output_dir)
        self.device = device
        self.n_frames = n_frames
        self.overlay_alpha_full = overlay_alpha_full
        self.overlay_alpha_selective = overlay_alpha_selective

        self.curves_dir = self.output_dir / "temporal_curves_attention_visualizations" / "temporal_curves"
        self.viz_dir = self.output_dir / "temporal_curves_attention_visualizations" / "attention_visualizations"
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directories: {self.curves_dir} and {self.viz_dir}")

        self.color_manager = ColorManager()

        self.model = self._load_model()

        self.param_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
        self.view_names = ["FC", "TC", "SA", "LA"]

    def _load_model(self):
        try:
            print("Loading dataset for normalization parameters...")
            training_frames = int(os.environ.get("CARDIOAI_TRAINING_FRAMES", 32))

            temp_dataset = CardioAIDataset(
                self.dataset_dir, self.excel_file, max_frames=training_frames, subset_size=10
            )
            norm_params = temp_dataset.get_normalization_parameters()
            print(
                f"Normalization parameters loaded: log_transform_indices={norm_params['log_transform_indices']}"
            )

            model = create_model(num_outputs=9, num_frames=training_frames, num_views=4)
            model.set_winsorized_normalization(norm_params)
            print("Normalization parameters set in model")

            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    print(
                        f"Warning: Model architecture mismatch. "
                        f"Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
                    )
                print(f"Loaded compatible weights from {self.model_path}")
            else:
                print("Warning: Using randomly initialized model")

            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def save_robust_figure(self, fig, filename, output_dir, preserve_facecolor=False):
        """Save figure as PNG, TIFF, EPS at 300 DPI. Returns list of saved paths."""
        base_name = Path(filename).stem
        output_dir = Path(output_dir)
        saved_files = []

        try:
            fig.tight_layout(pad=2.0)
        except Exception:
            pass

        facecolor = fig.get_facecolor() if preserve_facecolor else "white"

        try:
            png_path = output_dir / f"{base_name}.png"
            fig.savefig(
                png_path, format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.15, facecolor=facecolor, edgecolor="none",
            )
            saved_files.append(str(png_path))

            tiff_path = output_dir / f"{base_name}.tiff"
            fig.savefig(
                tiff_path, format="tiff", dpi=300, bbox_inches="tight",
                pad_inches=0.15, facecolor=facecolor, edgecolor="none",
            )
            saved_files.append(str(tiff_path))

            eps_path = output_dir / f"{base_name}.eps"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig.savefig(
                        eps_path, format="eps", dpi=300, bbox_inches="tight",
                        pad_inches=0.15, facecolor=facecolor, edgecolor="none",
                    )
                saved_files.append(str(eps_path))
            except Exception:
                try:
                    with open(eps_path, "w") as f:
                        f.write(f"%!PS-Adobe-3.0 EPSF-3.0\n% Generated by CardioAI for {base_name}\n")
                    saved_files.append(str(eps_path))
                except Exception:
                    pass

            return saved_files
        except Exception as e:
            print(f"Error saving figure {filename}: {e}")
            return []

    def _get_patient_data(self, patient_id):
        try:
            training_frames = int(os.environ.get("CARDIOAI_TRAINING_FRAMES", 32))
            dataset = CardioAIDataset(
                self.dataset_dir, self.excel_file, max_frames=training_frames
            )

            for _, (views, targets, pid) in enumerate(dataset):
                if pid == patient_id:
                    fixed_views = []
                    for view in views:
                        if len(view.shape) == 3:
                            view = view.unsqueeze(0)
                        elif len(view.shape) != 4:
                            print(f"Unexpected view shape from dataset: {view.shape}")
                            continue
                        fixed_views.append(view.to(self.device))

                    print(f"Loaded patient {patient_id} with {len(fixed_views)} views")
                    for i, view in enumerate(fixed_views):
                        print(f"  View {i} shape: {view.shape}")
                    return fixed_views, targets.to(self.device), pid

            print(f"Patient {patient_id} not found in dataset")
            return None, None, None
        except Exception as e:
            print(f"Error loading patient data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def generate_temporal_rollout_curves(self, patient_id="E100017564"):
        print(f"Generating temporal rollout curves for patient {patient_id}...")

        try:
            views, targets, pid = self._get_patient_data(patient_id)
            if views is None:
                print("Could not load patient data")
                return {}

            rollout_scores = self.model.attention_rollout(views, target_param_idx=0)

            view_color_map = self.color_manager.get_view_color_map()
            curves_generated = render_temporal_rollout_curves(
                self.model, views, self.param_names, view_color_map, self.color_manager,
                self.save_robust_figure, self.curves_dir,
            )

            print(f"Generated {curves_generated} temporal curve files")
            return rollout_scores
        except Exception as e:
            print(f"Error in temporal rollout generation: {e}")
            return {}

    def generate_attention_visualizations(self, patient_id="E100017564"):
        print(f"Generating attention visualizations for patient {patient_id}...")

        try:
            views, targets, pid = self._get_patient_data(patient_id)
            if views is None:
                print("Could not load patient data")
                return 0

            total_viz = render_ig_visualizations(
                self.model, views, self.param_names, self.view_names,
                self.n_frames, self.overlay_alpha_full,
                self.save_robust_figure, self.viz_dir, patient_id,
            )

            print(f"Generated {total_viz} attention visualization files")
            return total_viz
        except Exception as e:
            print(f"Error in attention visualization generation: {e}")
            return 0
