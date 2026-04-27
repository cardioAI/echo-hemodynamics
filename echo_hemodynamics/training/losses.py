"""Loss functions for progressive training."""

import torch
import torch.nn as nn


class ProgressiveMSELoss(nn.Module):
    """MSE loss that early-exits with infinity when predictions or targets contain NaN/Inf."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("Warning: NaN/Inf in predictions")
            return torch.tensor(float("inf"), device=predictions.device)

        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN/Inf in targets")
            return torch.tensor(float("inf"), device=predictions.device)

        return self.mse_loss(predictions, targets)
