"""Temporal attention rollout curve rendering."""

import matplotlib.pyplot as plt
import numpy as np

from .frame_selection import normalize_scores


def render_temporal_rollout_curves(
    model, views, param_names, view_color_map, color_manager, save_robust_figure, curves_dir,
):
    """Render a per-parameter line plot of the four-view temporal attention rollouts.

    Returns the integer count of files written (each parameter saves 3 formats).
    """
    curves_generated = 0

    for param_idx, param_name in enumerate(param_names):
        try:
            param_rollout = model.attention_rollout(views, target_param_idx=param_idx)

            fig_combined, ax_combined = plt.subplots(1, 1, figsize=(10, 6))

            for view_idx, (view_name, scores) in enumerate(param_rollout.items()):
                if view_idx >= 4:
                    break
                scores_1d = normalize_scores(scores)
                frames = np.arange(len(scores_1d))
                view_color = view_color_map.get(view_name, color_manager.get_dark_colors(1)[0])
                ax_combined.plot(frames, scores_1d, linewidth=2, color=view_color, label=f"{view_name} View")

            ax_combined.set_title(f"Temporal Attention Rollout - {param_name}")
            ax_combined.set_xlabel("Frame Index")
            ax_combined.set_ylabel("Attention Weight")
            ax_combined.legend(loc="best")
            ax_combined.grid(False)
            plt.tight_layout()

            base_name = f"rollout_{param_name.lower()}"
            save_robust_figure(fig_combined, base_name, curves_dir)
            plt.close(fig_combined)
            curves_generated += 3
        except Exception as e:
            print(f"Error generating curve for {param_name}: {e}")
            continue

    return curves_generated
