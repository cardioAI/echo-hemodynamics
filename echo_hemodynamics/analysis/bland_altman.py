"""Bland-Altman plots: plain (validation) and PH-stratified (test)."""

import matplotlib.pyplot as plt
import numpy as np

from .metrics import PARAM_PALETTE


def _compute_loa(pred, true):
    diff = pred - true
    mean_vals = (pred + true) / 2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    return diff, mean_vals, mean_diff, std_diff


def render_plain_bland_altman(pred_denorm, targets, param_names, cardio_utils, color_manager, prefix):
    """Render plain Bland-Altman plots (validation variant)."""
    for i, param_name in enumerate(param_names):
        pred = pred_denorm[:, i]
        true = targets[:, i]
        diff, mean_vals, mean_diff, std_diff = _compute_loa(pred, true)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        dark_colors = color_manager.get_figure_colors(5, "regular")
        ax.scatter(mean_vals, diff, alpha=0.6, s=30, color=dark_colors[0])
        ax.axhline(mean_diff, color=dark_colors[1], linestyle="-", label=f"Mean: {mean_diff:.2f}")
        ax.axhline(upper_loa, color=dark_colors[2], linestyle="--", label=f"Upper LoA: {upper_loa:.2f}")
        ax.axhline(lower_loa, color=dark_colors[3], linestyle="--", label=f"Lower LoA: {lower_loa:.2f}")
        ax.set_xlabel(f"Mean of True and Predicted {param_name}")
        ax.set_ylabel(f"Predicted - True {param_name}")
        ax.set_title(f"{prefix.title()}: {param_name} Bland-Altman Plot")
        ax.legend(fontsize=8)
        ax.grid(False)

        plt.tight_layout()
        cardio_utils.save_figure(fig, f"{prefix}_bland_altman_{param_name.lower()}", subdir="figures")
        plt.close(fig)


def render_ph_stratified_bland_altman(pred_denorm, targets, param_names, cardio_utils, prefix):
    """Render PH-stratified Bland-Altman plots (test variant)."""
    meanpap_idx = param_names.index("meanPAP")
    meanpap_values = targets[:, meanpap_idx]
    ph_positive = meanpap_values > 20.0
    ph_negative = meanpap_values <= 20.0

    for i, param_name in enumerate(param_names):
        pred = pred_denorm[:, i]
        true = targets[:, i]
        diff, mean_vals, mean_diff, std_diff = _compute_loa(pred, true)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        param_color = PARAM_PALETTE[i]

        if np.any(ph_positive):
            ax.scatter(mean_vals[ph_positive], diff[ph_positive],
                       alpha=0.7, s=120, marker="o", color=param_color, edgecolors="none",
                       label="PH positive (meanPAP > 20)")
        if np.any(ph_negative):
            ax.scatter(mean_vals[ph_negative], diff[ph_negative],
                       alpha=0.7, s=120, marker="s", color=param_color, edgecolors="none",
                       label="PH negative (meanPAP <= 20)")

        ax.axhline(mean_diff, color="black", linestyle="-", linewidth=1.5, label=f"Mean: {mean_diff:.2f}")
        ax.axhline(upper_loa, color="gray", linestyle="--", linewidth=1, label=f"Upper LoA: {upper_loa:.2f}")
        ax.axhline(lower_loa, color="gray", linestyle="--", linewidth=1, label=f"Lower LoA: {lower_loa:.2f}")
        ax.set_xlabel(f"Mean of True and Predicted {param_name}", fontsize=14)
        ax.set_ylabel(f"Predicted - True {param_name}", fontsize=14)
        ax.text(0.5, 0.97, f"{prefix.title()}: {param_name} Bland-Altman Plot",
                transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(fontsize=10)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        plt.tight_layout()
        cardio_utils.save_figure(fig, f"{prefix}_bland_altman_{param_name.lower()}", subdir="figures")
        plt.close(fig)
