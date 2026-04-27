"""Factory functions for ProgressiveCardioAI and its optimizer."""

import torch

from .progressive_model import ProgressiveCardioAI


def create_model(num_outputs=9, num_frames=32, num_views=4, dropout_rate=0.15,
                 ablation_attentions="temporal"):
    """Create progressive CardioAI model with the requested attention configuration."""
    model = ProgressiveCardioAI(
        num_outputs=num_outputs,
        num_frames=num_frames,
        num_views=num_views,
        dropout_rate=dropout_rate,
        ablation_attentions=ablation_attentions,
    )
    print(f"CardioAI model: ViT-Base backbone, attention={ablation_attentions}")
    return model


def create_progressive_optimizer(model, task_lr=1e-4, vit_lr=1e-5, weight_decay=1e-4):
    """AdamW with discriminative learning rates: task layers get task_lr, ViT gets vit_lr."""
    param_groups = model.get_trainable_parameters()
    optimizer_params = []

    if param_groups["task_params"]:
        optimizer_params.append({
            "params": param_groups["task_params"],
            "lr": task_lr,
            "weight_decay": weight_decay,
        })
    if param_groups["vit_params"]:
        optimizer_params.append({
            "params": param_groups["vit_params"],
            "lr": vit_lr,
            "weight_decay": weight_decay,
        })

    optimizer = torch.optim.AdamW(optimizer_params)
    print(f"Optimizer: task_lr={task_lr}, vit_lr={vit_lr} ({task_lr / vit_lr:.1f}x ratio)")
    return optimizer
