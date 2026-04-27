"""Parameter-specific regression heads for the CardioAI model."""

import torch.nn as nn


class ParameterHeadWithResidual(nn.Module):
    """Regression head: 768 -> LayerNorm -> 256 -> GELU -> 128 -> GELU -> 1"""

    def __init__(self, hidden_size=768, dropout_rate=0.15):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.main_path(x)
