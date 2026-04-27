"""ROC/AUC plots: simple (validation) and dual with PH gold-standard (test)."""

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from .metrics import CLINICAL_CUTOFFS, PARAM_PALETTE


def render_simple_roc(pred_denorm, targets, param_names, cardio_utils, color_manager, prefix):
    """Render single-curve ROC per parameter using a median split (validation variant)."""
    import numpy as np

    for i, param_name in enumerate(param_names):
        pred = pred_denorm[:, i]
        true = targets[:, i]

        true_median = np.median(true)
        y_true = (true > true_median).astype(int)

        fpr, tpr, _ = roc_curve(y_true, pred)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(fpr, tpr, color=color_manager.get_color(i), linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{prefix.title()}: {param_name} ROC Curve")
        ax.legend()
        ax.grid(False)

        plt.tight_layout()
        cardio_utils.save_figure(fig, f"{prefix}_roc_curve_{param_name.lower()}", subdir="figures")
        plt.close(fig)


def render_dual_roc(pred_denorm, targets, param_names, cardio_utils, prefix):
    """Render dual-curve ROC: clinical-cutoff per parameter + PH gold-standard (test variant)."""
    meanpap_idx = param_names.index("meanPAP")
    meanpap_true = targets[:, meanpap_idx]

    for i, param_name in enumerate(param_names):
        pred = pred_denorm[:, i]
        true = targets[:, i]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if param_name in CLINICAL_CUTOFFS:
            cutoff = CLINICAL_CUTOFFS[param_name]
            if param_name in ["CO", "CI"]:
                y_true_param = (true < cutoff).astype(int)
                y_scores_param = -pred
            else:
                y_true_param = (true > cutoff).astype(int)
                y_scores_param = pred

            fpr_param, tpr_param, _ = roc_curve(y_true_param, y_scores_param)
            roc_auc_param = auc(fpr_param, tpr_param)
            ax.plot(fpr_param, tpr_param, color=PARAM_PALETTE[i], linewidth=2.5,
                    label=f"{param_name} (AUC={roc_auc_param:.3f})")

        if param_name != "meanPAP":
            y_true_ph = (meanpap_true > 20.0).astype(int)
            fpr_ph, tpr_ph, _ = roc_curve(y_true_ph, pred)
            roc_auc_ph = auc(fpr_ph, tpr_ph)
            ax.plot(fpr_ph, tpr_ph, color="gray", linewidth=2, linestyle="--",
                    label=f"PH diagnosis (meanPAP>20, AUC={roc_auc_ph:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.text(0.5, 0.97, f"{prefix.title()}: {param_name} ROC Curve",
                transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        plt.tight_layout()
        cardio_utils.save_figure(fig, f"{prefix}_roc_curve_{param_name.lower()}", subdir="figures")
        plt.close(fig)
