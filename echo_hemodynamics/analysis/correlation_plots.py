"""Correlation bar plots and per-parameter heatmaps for validation and test."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from ..utils.singleton import get_cardio_heatmap_cmap
from .metrics import PARAM_PALETTE, calculate_correlation


def render_correlation_bar(pred_denorm, targets, param_names, cardio_utils,
                            filename, title, color_manager=None):
    """Bar plot of per-parameter correlation. ``filename`` does not include the extension."""
    correlations = [calculate_correlation(pred_denorm[:, i], targets[:, i]) for i in range(pred_denorm.shape[1])]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if color_manager is not None:
        palette_colors = color_manager.get_color_palette(3)
        colors = [
            palette_colors[0] if c >= 0.6 else palette_colors[1] if c > 0.3 else palette_colors[2]
            for c in correlations
        ]
        threshold_color = color_manager.get_color(0)
    else:
        colors = PARAM_PALETTE
        threshold_color = "black"

    bars = ax.bar(param_names, correlations, color=colors, alpha=0.8, edgecolor="none")
    if color_manager is not None:
        ax.axhline(y=0.6, color=threshold_color, linestyle="--", alpha=0.7, label="Target Min")
        ax.legend()

    ax.set_xlabel("Parameters", fontsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.text(
        0.5, 0.97, title, transform=ax.transAxes,
        fontsize=14, fontweight="bold", ha="center", va="top",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(False)

    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.01,
            f"{corr:.3f}", ha="center", va="bottom", fontsize=11,
        )

    plt.tight_layout()
    cardio_utils.save_figure(fig, filename, subdir="figures")
    plt.close(fig)
    return correlations


def render_correlation_heatmap(pred_denorm, param_names, cardio_utils, filename, title):
    """Standard seaborn heatmap of the predicted-feature correlation matrix."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    corr_matrix = np.corrcoef(pred_denorm.T)
    heatmap_cmap = get_cardio_heatmap_cmap("blue_gray_orange")
    sns.heatmap(
        corr_matrix, annot=True, cmap=heatmap_cmap, center=0,
        xticklabels=param_names, yticklabels=param_names, ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    cardio_utils.save_figure(fig, filename, subdir="figures")
    plt.close(fig)


def render_correlation_bubble_heatmap(pred_denorm, param_names, cardio_utils, filename, title):
    """Bubble-style diverging correlation matrix used by the test runner."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr_matrix = np.corrcoef(pred_denorm.T)
    n_params = len(param_names)

    colors_list = ["#4575b4", "#91bfdb", "#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
    cmap = LinearSegmentedColormap.from_list("correlation", colors_list, N=100)

    for i in range(n_params):
        for j in range(i + 1):
            corr_val = corr_matrix[i, j]
            size = abs(corr_val) * 1800
            color = cmap((corr_val + 1) / 2)
            ax.scatter(j, i, s=size, c=[color], alpha=0.8, edgecolors="none")
            text_color = "white" if abs(corr_val) > 0.5 else "black"
            fontsize = 8 if abs(corr_val) < 0.3 else 9
            ax.text(j, i, f"{corr_val:.2f}", ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=text_color)

    ax.set_xlim(-0.5, n_params - 0.5)
    ax.set_ylim(-0.5, n_params - 0.5)
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(param_names, fontsize=12)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.text(0.5, 0.97, title, transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="center", va="top")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation Coefficient", rotation=270, labelpad=25, fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    cardio_utils.save_figure(fig, filename, subdir="figures")
    plt.close(fig)
