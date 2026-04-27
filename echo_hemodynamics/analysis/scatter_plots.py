"""Scatter plots: plain (validation) and PH-stratified with regression line (test)."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .metrics import CLINICAL_CUTOFFS, PARAM_PALETTE, calculate_correlation


def render_plain_scatter(pred_denorm, targets, param_names, cardio_utils, color_manager, prefix):
    """Render simple identity-line scatter plots, one per parameter."""
    for i, param_name in enumerate(param_names):
        pred_param = pred_denorm[:, i]
        true_param = targets[:, i]
        corr = calculate_correlation(pred_param, true_param)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        dark_colors = color_manager.get_figure_colors(1, "regular")
        ax.scatter(true_param, pred_param, alpha=0.6, s=30, color=dark_colors[0])

        min_val = min(true_param.min(), pred_param.min())
        max_val = max(true_param.max(), pred_param.max())
        ax.plot([min_val, max_val], [min_val, max_val], "--", color=dark_colors[0], alpha=0.8)

        ax.set_xlabel(f"True {param_name}")
        ax.set_ylabel(f"Predicted {param_name}")
        ax.set_title(f"{prefix.title()}: {param_name} (r={corr:.3f})")
        ax.grid(False)

        plt.tight_layout()
        cardio_utils.save_figure(fig, f"{prefix}_scatter_plot_{param_name.lower()}", subdir="figures")
        plt.close(fig)


def render_ph_stratified_scatter(pred_denorm, targets, param_names, cardio_utils, prefix):
    """PH-stratified scatter with regression line and clinical cutoffs (test variant)."""
    meanpap_idx = param_names.index("meanPAP")
    meanpap_values = targets[:, meanpap_idx]
    ph_positive = meanpap_values > 20.0
    ph_negative = meanpap_values <= 20.0

    for i, param_name in enumerate(param_names):
        pred_param = pred_denorm[:, i]
        true_param = targets[:, i]
        corr = calculate_correlation(pred_param, true_param)

        slope, intercept, _, _, _ = stats.linregress(true_param, pred_param)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        param_color = PARAM_PALETTE[i]

        if np.any(ph_positive):
            ax.scatter(true_param[ph_positive], pred_param[ph_positive],
                       alpha=0.7, s=120, marker="o", color=param_color, edgecolors="none",
                       label="PH positive (meanPAP > 20)")
        if np.any(ph_negative):
            ax.scatter(true_param[ph_negative], pred_param[ph_negative],
                       alpha=0.7, s=120, marker="s", color=param_color, edgecolors="none",
                       label="PH negative (meanPAP <= 20)")

        min_val = min(true_param.min(), pred_param.min())
        max_val = max(true_param.max(), pred_param.max())
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", alpha=0.5,
                linewidth=1, label="Identity")
        regression_x = np.array([min_val, max_val])
        regression_y = slope * regression_x + intercept
        ax.plot(regression_x, regression_y, "-", color="black", alpha=0.7, linewidth=2,
                label=f"Regression: y={slope:.2f}x+{intercept:.2f}")

        if param_name in CLINICAL_CUTOFFS:
            cutoff = CLINICAL_CUTOFFS[param_name]
            ax.axvline(x=cutoff, color="gray", linestyle=":", linewidth=1.5, alpha=0.6,
                       label=f"Clinical cutoff: {cutoff}")
            ax.axhline(y=cutoff, color="gray", linestyle=":", linewidth=1.5, alpha=0.6)

        ax.set_xlabel(f"True {param_name}", fontsize=14)
        ax.set_ylabel(f"Predicted {param_name}", fontsize=14)
        ax.text(0.5, 0.97, f"{prefix.title()}: {param_name} (r={corr:.3f})",
                transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        plt.tight_layout()
        cardio_utils.save_figure(fig, f"{prefix}_scatter_plot_{param_name.lower()}", subdir="figures")
        plt.close(fig)
