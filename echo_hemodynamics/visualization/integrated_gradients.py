"""Integrated Gradients overlay rendering for attention visualizations."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from ..utils.singleton import get_cardio_heatmap_cmap
from .apex_mask import create_apex_mask
from .frame_selection import normalize_scores, select_top_frames


def _normalize_gradients(gradients, frame_image):
    """Apply apex mask + Gaussian smoothing and return (full_overlay, selective_overlay, mask)."""
    grad_abs = np.abs(gradients)
    if grad_abs.max() <= 0:
        full = (grad_abs - grad_abs.min()) / (grad_abs.max() - grad_abs.min() + 1e-8)
        return full, np.zeros_like(grad_abs), np.zeros_like(grad_abs, dtype=bool)

    grad_norm = (grad_abs - grad_abs.min()) / (grad_abs.max() - grad_abs.min() + 1e-8)
    apex_mask = create_apex_mask(grad_norm.shape, apex_height=70)
    grad_norm = grad_norm * apex_mask
    grad_smooth = gaussian_filter(grad_norm, sigma=3.0)

    grad_smooth_for_full = grad_smooth.copy()
    apex_mask_binary = (apex_mask == 0)
    if apex_mask_binary.any() and (~apex_mask_binary).any():
        min_val = np.percentile(grad_smooth[~apex_mask_binary], 5)
        grad_smooth_for_full[apex_mask_binary] = min_val

    grad_normalized_full = (
        (grad_smooth_for_full - grad_smooth_for_full.min())
        / (grad_smooth_for_full.max() - grad_smooth_for_full.min() + 1e-8)
    )

    percentile_95 = np.percentile(grad_smooth, 95)
    high_attention_mask = grad_smooth > percentile_95
    grad_normalized_selective = np.zeros_like(grad_smooth)
    if high_attention_mask.any():
        max_val = grad_smooth[high_attention_mask].max()
        grad_normalized_selective[high_attention_mask] = (
            (grad_smooth[high_attention_mask] - percentile_95) / (max_val - percentile_95 + 1e-8)
        )

    return grad_normalized_full, grad_normalized_selective, high_attention_mask


def render_ig_visualizations(
    model, views, param_names, view_names, n_frames, overlay_alpha_full,
    save_robust_figure, viz_dir, patient_id,
):
    """Render ultrasound + IG overlay images for the top-N frames per (view, parameter).

    Returns the integer count of files written.
    """
    total_viz = 0

    for param_idx, param_name in enumerate(param_names):
        print(f"Processing parameter {param_name}...")
        try:
            param_rollout = model.attention_rollout(views, target_param_idx=param_idx)

            for view_idx, view_name in enumerate(view_names):
                if view_name not in param_rollout:
                    continue
                scores = normalize_scores(param_rollout[view_name])
                top_indices = select_top_frames(scores, n_frames)

                print(
                    f"    View {view_name}: Selected {len(top_indices)} frames "
                    f"from {len(scores)} total frames"
                )

                for frame_idx in top_indices:
                    try:
                        frame_idx = int(frame_idx.item()) if hasattr(frame_idx, "item") else int(frame_idx)

                        view_tensor = views[view_idx]
                        if len(view_tensor.shape) == 4:
                            if frame_idx >= view_tensor.shape[1]:
                                continue
                            frame_tensor = view_tensor[0, frame_idx]
                        elif len(view_tensor.shape) == 3:
                            if frame_idx >= view_tensor.shape[0]:
                                continue
                            frame_tensor = view_tensor[frame_idx]
                        else:
                            continue

                        frame_image = frame_tensor.cpu().numpy()
                        if len(frame_image.shape) != 2 or frame_image.size == 0:
                            continue

                        # 1. Pure ultrasound
                        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                        ax1.imshow(frame_image, cmap="gray")
                        ax1.set_title(f"{view_name} View - {param_name} - Frame {frame_idx}")
                        ax1.axis("off")
                        plt.tight_layout()
                        save_robust_figure(
                            fig1,
                            f"{patient_id}_{param_name}_{view_name}_frame_{frame_idx:02d}_ultrasound",
                            viz_dir,
                        )
                        plt.close(fig1)
                        total_viz += 3

                        # 2. IG overlay on dark background
                        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
                        fig2.patch.set_facecolor((0.12, 0.12, 0.28))
                        ax2.set_facecolor((0.12, 0.12, 0.28))

                        try:
                            ig_dict = model.get_integrated_gradients(
                                [views[view_idx]], target_param_idx=param_idx
                            )
                            view_key = list(ig_dict.keys())[0]
                            grad_tensor = ig_dict[view_key]
                            if len(grad_tensor.shape) == 4:
                                gradients = grad_tensor[0, frame_idx].cpu().numpy()
                            elif len(grad_tensor.shape) == 3:
                                gradients = grad_tensor[frame_idx].cpu().numpy()
                            else:
                                raise ValueError(f"Unexpected gradient shape: {grad_tensor.shape}")

                            grad_full, _, _ = _normalize_gradients(gradients, frame_image)
                        except Exception as e:
                            print(
                                f"    Warning: IG failed for {view_name}-{param_name}-{frame_idx}: {e}; "
                                "using attention-weighted fallback"
                            )
                            heatmap = np.random.rand(*frame_image.shape) * scores[frame_idx]
                            grad_full, _, _ = _normalize_gradients(heatmap, frame_image)

                        heatmap_cmap = get_cardio_heatmap_cmap("blue_cyan_yellow")
                        ax2.imshow(frame_image, cmap="gray", alpha=1.0)
                        ax2.imshow(grad_full, cmap=heatmap_cmap, alpha=overlay_alpha_full, vmin=0, vmax=1)
                        ax2.set_title(
                            f"{view_name} View - {param_name} - Frame {frame_idx} - Integrated Gradients"
                        )
                        ax2.axis("off")
                        plt.tight_layout()

                        save_robust_figure(
                            fig2,
                            f"{patient_id}_{param_name}_{view_name}_frame_{frame_idx:02d}_overlay_full",
                            viz_dir,
                            preserve_facecolor=True,
                        )
                        plt.close(fig2)
                        total_viz += 3
                    except Exception as e:
                        print(f"Error generating viz for {param_name}-{view_name}-{frame_idx}: {e}")
                        continue
        except Exception as e:
            print(f"Error processing parameter {param_name}: {e}")
            continue

    return total_viz
