"""Heteroscedasticity diagnostic: residual SD by quartile + Breusch-Pagan test (test-only)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .metrics import PARAM_PALETTE, PARAM_UNITS


def render_heteroscedasticity_analysis(pred_denorm, targets, param_names, cardio_utils, output_dir):
    """3x3 residual-SD plot per parameter + Excel + summary table figure."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    hetero_stats = []

    for idx, param_name in enumerate(param_names):
        ax = axes[idx]

        pred_values = pred_denorm[:, idx]
        true_values = targets[:, idx]
        residuals = pred_values - true_values
        abs_residuals = np.abs(residuals)

        n_bins = 4
        bin_edges = np.percentile(true_values, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        bin_stds = []
        bin_means = []
        bin_counts = []

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (true_values >= bin_edges[i]) & (true_values < bin_edges[i + 1])
            else:
                mask = (true_values >= bin_edges[i]) & (true_values <= bin_edges[i + 1])

            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_stds.append(np.std(residuals[mask]))
                bin_means.append(np.mean(abs_residuals[mask]))
                bin_counts.append(mask.sum())

        bars = ax.bar(
            range(len(bin_stds)), bin_stds, color=PARAM_PALETTE[idx],
            alpha=0.7, edgecolor="black", linewidth=1,
        )
        for bar, std_val in zip(bars, bin_stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(bin_stds),
                f"{std_val:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

        quartile_labels = [
            f"Q{i + 1}\n({bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f})" for i in range(n_bins)
        ]
        ax.set_xticks(range(len(bin_stds)))
        ax.set_xticklabels(quartile_labels, fontsize=8)

        ax.set_xlabel(f"True Value Quartiles ({PARAM_UNITS[param_name]})", fontsize=10)
        ax.set_ylabel("Residual SD", fontsize=10)
        ax.set_title(f"{param_name}", fontsize=12, fontweight="bold")

        sd_ratio = max(bin_stds) / min(bin_stds) if min(bin_stds) > 0 else np.nan

        squared_residuals = residuals ** 2
        bp_corr, bp_pvalue = stats.pearsonr(true_values, squared_residuals)
        spearman_corr, spearman_pvalue = stats.spearmanr(true_values, abs_residuals)

        hetero_stats.append({
            "Parameter": param_name,
            "Unit": PARAM_UNITS[param_name],
            "SD_Q1": bin_stds[0],
            "SD_Q2": bin_stds[1] if len(bin_stds) > 1 else np.nan,
            "SD_Q3": bin_stds[2] if len(bin_stds) > 2 else np.nan,
            "SD_Q4": bin_stds[-1],
            "SD_Ratio_Q4_Q1": sd_ratio,
            "BP_Correlation": bp_corr,
            "BP_P_Value": bp_pvalue,
            "Spearman_Corr": spearman_corr,
            "Spearman_P_Value": spearman_pvalue,
            "Heteroscedastic": "Yes" if (sd_ratio > 1.5 or bp_pvalue < 0.05) else "No",
        })

        if sd_ratio > 1.5 or bp_pvalue < 0.05:
            ax.text(0.95, 0.95, "Heteroscedastic", transform=ax.transAxes,
                    fontsize=9, ha="right", va="top", color="red", fontweight="bold")
        else:
            ax.text(0.95, 0.95, "Homoscedastic", transform=ax.transAxes,
                    fontsize=9, ha="right", va="top", color="green", fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Heteroscedasticity Analysis: Residual Standard Deviation by Value Quartiles",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    cardio_utils.save_figure(fig, "test_heteroscedasticity_analysis", subdir="figures")
    plt.close(fig)

    df_hetero = pd.DataFrame(hetero_stats)
    table_path = output_dir / "tables" / "test_heteroscedasticity_summary.xlsx"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df_hetero.to_excel(table_path, index=False, float_format="%.4f")
    print(f"Heteroscedasticity table saved to: {table_path}")

    fig_table, ax_table = plt.subplots(figsize=(16, 6))
    ax_table.axis("off")

    headers = [
        "Parameter", "SD Q1", "SD Q2", "SD Q3", "SD Q4",
        "SD Ratio\n(Q4/Q1)", "BP Corr", "BP p-value", "Status",
    ]
    table_data = []
    for stat in hetero_stats:
        row = [
            stat["Parameter"],
            f"{stat['SD_Q1']:.2f}",
            f"{stat['SD_Q2']:.2f}" if not np.isnan(stat["SD_Q2"]) else "-",
            f"{stat['SD_Q3']:.2f}" if not np.isnan(stat["SD_Q3"]) else "-",
            f"{stat['SD_Q4']:.2f}",
            f"{stat['SD_Ratio_Q4_Q1']:.2f}",
            f"{stat['BP_Correlation']:.3f}",
            f"{stat['BP_P_Value']:.4f}" if stat["BP_P_Value"] >= 0.0001 else "<0.0001",
            stat["Heteroscedastic"],
        ]
        table_data.append(row)

    table = ax_table.table(
        cellText=table_data, colLabels=headers, loc="center",
        cellLoc="center", colColours=["lightgray"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for i, stat in enumerate(hetero_stats):
        cell = table[(i + 1, 8)]
        cell.set_facecolor("#ffcccc" if stat["Heteroscedastic"] == "Yes" else "#ccffcc")

    plt.title("Heteroscedasticity Summary: Residual Variance Analysis Across Value Ranges",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    cardio_utils.save_figure(fig_table, "test_heteroscedasticity_table", subdir="figures")
    plt.close(fig_table)
