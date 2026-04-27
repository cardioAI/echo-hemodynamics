"""Confusion matrix plots based on a median split."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..utils.singleton import get_cardio_heatmap_cmap


def render_confusion_matrices(pred_denorm, targets, param_names, cardio_utils, prefix):
    """Render confusion matrices per parameter using above/below median split."""
    heatmap_cmap = get_cardio_heatmap_cmap("blue_gray_orange")

    for i, param_name in enumerate(param_names):
        pred = pred_denorm[:, i]
        true = targets[:, i]

        true_median = np.median(true)
        y_true = (true > true_median).astype(int)
        y_pred = (pred > true_median).astype(int)

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=heatmap_cmap,
            xticklabels=["Below Median", "Above Median"],
            yticklabels=["Below Median", "Above Median"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{prefix.title()}: {param_name} Confusion Matrix")

        plt.tight_layout()
        cardio_utils.save_figure(fig, f"{prefix}_confusion_matrix_{param_name.lower()}", subdir="figures")
        plt.close(fig)
