"""Temporal aggregation module: gated combination of attention and mean pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedTemporalAggregation(nn.Module):
    """Temporal aggregation with learnable gating between attention and mean-pooling paths.

    Combines frame-importance-weighted aggregation with a residual mean-pooling path,
    balanced by a learned sigmoid gate (alpha * attention + (1-alpha) * residual).
    """

    def __init__(self, hidden_size, num_frames=32, dropout_rate=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_frames = num_frames

        self.frame_importance = nn.Sequential(
            nn.Linear(hidden_size, num_frames),
            nn.Dropout(dropout_rate),
            nn.Softmax(dim=1),
        )

        self.gate_weight = nn.Parameter(torch.zeros(1))

        self.feature_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, frame_features):
        if len(frame_features.shape) == 2:
            batch_size, hidden_size = frame_features.shape
            num_frames = 1
            frame_features = frame_features.unsqueeze(1)
        elif len(frame_features.shape) == 3:
            batch_size, num_frames, hidden_size = frame_features.shape
        elif len(frame_features.shape) == 4:
            batch_size, num_views, num_frames, hidden_size = frame_features.shape
            frame_features = frame_features.reshape(batch_size * num_views, num_frames, hidden_size)
            batch_size = batch_size * num_views
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D input, got {len(frame_features.shape)}D")

        frame_features = self.feature_norm(frame_features)
        frame_features = self.dropout(frame_features)

        global_context = torch.mean(frame_features, dim=1)
        frame_weights = self.frame_importance(global_context)

        if frame_weights.shape[1] != num_frames:
            if num_frames < self.num_frames:
                frame_weights = frame_weights[:, :num_frames]
                frame_weights = F.softmax(frame_weights, dim=1)
            else:
                frame_weights = F.interpolate(
                    frame_weights.unsqueeze(1),
                    size=num_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
                frame_weights = F.softmax(frame_weights, dim=1)

        attention_aggregated = torch.sum(frame_features * frame_weights.unsqueeze(-1), dim=1)
        temporal_residual = torch.mean(frame_features, dim=1)

        alpha = torch.sigmoid(self.gate_weight)
        aggregated_features = alpha * attention_aggregated + (1 - alpha) * temporal_residual

        return aggregated_features, None, frame_weights
