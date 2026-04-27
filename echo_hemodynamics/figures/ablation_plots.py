"""Ablation-study figures: bar comparison, heatmap, improvements, components, training loss."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils.singleton import ColorManager


def create_ablation_plots(results, output_dir, param_names, cardio_utils):
    """Render the five ablation plots into ``output_dir``."""
    color_manager = ColorManager()

    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12
    ANNOT_FONTSIZE = 11
    LEGEND_FONTSIZE = 12

    param_order = [3, 0, 1, 2, 4, 5, 6, 7, 8]

    # 1. Average correlation bar plot
    plt.figure(figsize=(12, 8))
    variant_names = list(results.keys())
    avg_correlations = [results[name]["avg_correlation"] for name in variant_names]

    bars = plt.bar(
        range(len(variant_names)), avg_correlations,
        color=[color_manager.get_color(i) for i in range(len(variant_names))],
    )
    plt.xlabel("Ablation Variant", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Average Correlation", fontsize=LABEL_FONTSIZE)
    plt.title("Attention Ablation Study - Average Correlations", fontsize=TITLE_FONTSIZE)
    plt.xticks(range(len(variant_names)), variant_names, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(False)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.01,
            f"{height:.3f}", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE,
        )

    plt.tight_layout()
    cardio_utils.save_figure(plt.gcf(), "ablation_comparison")
    plt.close()

    # 2. Per-parameter heatmap
    correlation_matrix = np.zeros((len(variant_names), len(param_names)))
    for i, variant_name in enumerate(variant_names):
        if "reordered_correlations" in results[variant_name]:
            correlation_matrix[i, :] = results[variant_name]["reordered_correlations"]
        else:
            original_corr = results[variant_name]["final_correlations"]
            correlation_matrix[i, :] = [original_corr[idx] for idx in param_order]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        correlation_matrix,
        xticklabels=param_names,
        yticklabels=variant_names,
        annot=True, fmt=".3f", cmap=color_manager.get_heatmap_colormap(),
        cbar_kws={"label": "Correlation Coefficient"},
        annot_kws={"fontsize": ANNOT_FONTSIZE},
    )
    plt.title("Attention Ablation Study - Per-Parameter Correlations", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Hemodynamic Parameters", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Ablation Variants", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label("Correlation Coefficient", fontsize=LABEL_FONTSIZE)
    plt.tight_layout()
    cardio_utils.save_figure(plt.gcf(), "ablation_heatmap")
    plt.close()

    # 3. Improvement vs baseline
    if "no_attention" in results and "full_model" in results:
        if "reordered_correlations" in results["no_attention"]:
            baseline_corr = np.array(results["no_attention"]["reordered_correlations"])
        else:
            original_baseline = results["no_attention"]["final_correlations"]
            baseline_corr = np.array([original_baseline[idx] for idx in param_order])

        if "reordered_correlations" in results["full_model"]:
            full_model_corr = np.array(results["full_model"]["reordered_correlations"])
        else:
            original_full = results["full_model"]["final_correlations"]
            full_model_corr = np.array([original_full[idx] for idx in param_order])

        improvements = full_model_corr - baseline_corr

        plt.figure(figsize=(12, 6))
        bars = plt.bar(
            param_names, improvements,
            color=[color_manager.get_color(i) for i in range(len(param_names))],
        )
        plt.xlabel("Hemodynamic Parameters", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Correlation Improvement", fontsize=LABEL_FONTSIZE)
        plt.title("Full Model vs No Attention - Correlation Improvements", fontsize=TITLE_FONTSIZE)
        plt.xticks(rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.grid(False)
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.001,
                f"{height:.3f}", ha="center",
                va="bottom" if height > 0 else "top", fontsize=ANNOT_FONTSIZE,
            )
        plt.tight_layout()
        cardio_utils.save_figure(plt.gcf(), "attention_improvements")
        plt.close()

    # 4. Component contribution
    if all(variant in results for variant in ["spatial_only", "temporal_only", "fusion_only"]):
        component_data = {
            "Spatial Only": results["spatial_only"]["avg_correlation"],
            "Temporal Only": results["temporal_only"]["avg_correlation"],
            "Fusion Only": results["fusion_only"]["avg_correlation"],
            "Full Model": results["full_model"]["avg_correlation"],
        }

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            component_data.keys(), component_data.values(),
            color=[color_manager.get_color(i) for i in range(len(component_data))],
        )
        plt.ylabel("Average Correlation", fontsize=LABEL_FONTSIZE)
        plt.title("Individual Attention Component Contributions", fontsize=TITLE_FONTSIZE)
        plt.xticks(rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.grid(False)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.005,
                f"{height:.3f}", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE,
            )
        plt.tight_layout()
        cardio_utils.save_figure(plt.gcf(), "component_contributions")
        plt.close()

    # 5. Training loss comparison
    loss_variants = {}
    for variant_name, result in results.items():
        if "history" in result and "train_loss" in result["history"]:
            loss_variants[variant_name] = result["history"]["train_loss"]

    if loss_variants:
        plt.figure(figsize=(12, 8))
        for variant_name, losses in loss_variants.items():
            epochs_range = range(1, len(losses) + 1)
            plt.plot(epochs_range, losses, label=variant_name, linewidth=2)

        plt.xlabel("Epoch", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Training Loss", fontsize=LABEL_FONTSIZE)
        plt.title("Training Loss Comparison - Ablation Variants", fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.grid(False)
        plt.tight_layout()
        cardio_utils.save_figure(plt.gcf(), "ablation_training_loss")
        plt.close()

    print(f"Comprehensive ablation plots saved to: {output_dir}")
