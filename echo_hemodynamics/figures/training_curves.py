"""Training-time figures: loss curves, per-parameter correlation, stage progression, LR schedule."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.singleton import CardioAIUtils


def generate_training_figures(training_history, output_dir):
    """Render the four training diagnostic figures into ``output_dir``."""
    output_dir = Path(output_dir)
    print("Generating training figures...")

    util = CardioAIUtils()
    util.current_output_dir = output_dir
    util.subdirs = {"training": output_dir}

    colors = util.get_color_palette(5)
    epochs_range = range(1, len(training_history["train_loss"]) + 1)

    # 1. Loss curves
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(epochs_range, training_history["train_loss"], color=colors[0], label="Training Loss", linewidth=2)
    ax1.plot(epochs_range, training_history["val_loss"], color=colors[1], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training and Validation Loss Curves")
    ax1.legend()
    ax1.grid(False)
    plt.tight_layout()
    util.save_figure(fig1, output_dir / "training_loss_curves")
    plt.close()

    # 2. Mean validation correlation
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(epochs_range, training_history["val_correlations"], color=colors[2],
             label="Mean Validation Correlation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Correlation Coefficient")
    ax2.set_title("Validation Correlation Progress")
    ax2.legend()
    ax2.grid(False)
    plt.tight_layout()
    util.save_figure(fig2, output_dir / "mean_correlation_progress")
    plt.close()

    # 3. Per-parameter correlation
    if training_history["per_task_val_corr"]:
        param_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
        fig, ax = plt.subplots(figsize=(12, 8))

        per_task_corr = np.array(training_history["per_task_val_corr"])
        for i, param in enumerate(param_names):
            if i < per_task_corr.shape[1]:
                ax.plot(epochs_range, per_task_corr[:, i], label=param, linewidth=2,
                        color=colors[i % len(colors)])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Correlation Coefficient")
        ax.set_title("Per-Parameter Validation Correlation Evolution")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(False)
        target_color = util.get_dark_colors(1)[0]
        ax.axhline(y=0.6, color=target_color, linestyle="--", alpha=0.7, label="Target (0.6)")
        plt.tight_layout()
        util.save_figure(fig, output_dir / "per_parameter_correlations")
        plt.close()

    # 4. Stage progression
    if training_history["stage_info"]:
        stage_data = []
        for stage_info in training_history["stage_info"]:
            stage_data.append({
                "Stage": stage_info["stage"],
                "Start_Epoch": stage_info["epoch"],
                "End_Epoch": stage_info.get("end_epoch", stage_info["epoch"]),
                "Unfrozen_Block": stage_info.get("unfrozen_block", "N/A"),
                "Trainable_Params": stage_info.get("trainable_params", 0),
            })

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, stage in enumerate(stage_data):
            start = stage["Start_Epoch"]
            end = stage.get("End_Epoch", len(training_history["train_loss"]))
            description = (
                f"Block {stage['Unfrozen_Block']}" if stage["Unfrozen_Block"] != "N/A" else "Task Layers"
            )
            ax.axvspan(start, end, alpha=0.3, color=colors[i % len(colors)],
                       label=f"Stage {stage['Stage']}: {description}")

        ax.plot(epochs_range, training_history["train_loss"], color="black", linewidth=2, label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title("Progressive Training Stages")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(False)
        plt.tight_layout()
        util.save_figure(fig, output_dir / "progressive_training_stages")
        plt.close()

    # 5. LR schedule
    if training_history["learning_rates"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        lr_data = training_history["learning_rates"]
        epochs_lr = range(1, len(lr_data) + 1)

        if lr_data and len(lr_data[0]) > 0:
            num_groups = len(lr_data[0])
            for group_idx in range(num_groups):
                lr_values = [lr_list[group_idx] for lr_list in lr_data]
                group_name = f"Group {group_idx + 1}" + (" (Task)" if group_idx == 0 else " (ViT)")
                ax.plot(epochs_lr, lr_values, label=group_name, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        util.save_figure(fig, output_dir / "learning_rate_schedule")
        plt.close()

    print(f"Training figures saved to: {output_dir}")
