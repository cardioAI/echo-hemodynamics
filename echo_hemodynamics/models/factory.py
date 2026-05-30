import torch
from .progressive_model import ProgressiveCardioAI


def create_model(
    num_outputs=9,
    num_frames=32,
    num_views=4,
    dropout_rate=0.15,
    ablation_attentions="temporal",
):
    model = ProgressiveCardioAI(
        num_outputs=num_outputs,
        num_frames=num_frames,
        num_views=num_views,
        dropout_rate=dropout_rate,
        ablation_attentions=ablation_attentions,
    )
    print(f"CardioAI model: ViT-Base backbone, attention={ablation_attentions}")
    return model


def create_progressive_optimizer(
    model, task_lr=0.0001, vit_lr=1e-05, weight_decay=0.0001
):
    param_groups = model.get_trainable_parameters()
    optimizer_params = []
    if param_groups["task_params"]:
        optimizer_params.append(
            {
                "params": param_groups["task_params"],
                "lr": task_lr,
                "weight_decay": weight_decay,
            }
        )
    if param_groups["vit_params"]:
        optimizer_params.append(
            {
                "params": param_groups["vit_params"],
                "lr": vit_lr,
                "weight_decay": weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_params)
    print(
        f"Optimizer: task_lr={task_lr}, vit_lr={vit_lr} ({task_lr / vit_lr:.1f}x ratio)"
    )
    return optimizer
