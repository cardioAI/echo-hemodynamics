"""ProgressiveAblationVariant: model wrapper that selectively disables attention modules."""

import copy

import torch
import torch.nn as nn


class ProgressiveAblationVariant(nn.Module):
    """Wraps a base ProgressiveCardioAI and replaces attention modules with mean-pooling stubs."""

    def __init__(self, base_model, spatial_attention=True, temporal_attention=True,
                 fusion_attention=True):
        super().__init__()
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.fusion_attention = fusion_attention
        self.num_frames = base_model.num_frames
        self.num_views = base_model.num_views
        self.hidden_size = base_model.hidden_size

        self.vision_transformer = copy.deepcopy(base_model.vision_transformer)

        self.register_buffer("param_mins", base_model.param_mins.clone())
        self.register_buffer("param_maxs", base_model.param_maxs.clone())
        self.register_buffer("log_mins", base_model.log_mins.clone())
        self.register_buffer("log_maxs", base_model.log_maxs.clone())
        self.log_transform_indices = base_model.log_transform_indices.copy()

        if temporal_attention:
            self.temporal_attention_module = copy.deepcopy(base_model.temporal_attention)
        else:
            self.simple_temporal_pool = nn.AdaptiveAvgPool1d(1)

        if fusion_attention:
            self.fusion_attention_module = copy.deepcopy(base_model.view_attention)
        else:
            self.simple_view_weights = nn.Parameter(torch.ones(4) / 4)

        self.regression_heads = copy.deepcopy(base_model.regression_heads)
        self.parameter_names = base_model.parameter_names.copy()

        self.spatial_attention_weights = []
        self.temporal_attention_weights = None
        self.fusion_attention_weights = None

    def normalize_targets(self, targets):
        normalized = torch.zeros_like(targets)
        for i in range(targets.shape[1]):
            if i not in self.log_transform_indices:
                normalized[:, i] = (
                    (targets[:, i] - self.param_mins[i]) /
                    (self.param_maxs[i] - self.param_mins[i] + 1e-8)
                )
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = torch.log(targets[:, param_idx] + 1)
            normalized[:, param_idx] = (
                (log_values - self.log_mins[i]) / (self.log_maxs[i] - self.log_mins[i] + 1e-8)
            )
        return normalized

    def denormalize_predictions(self, predictions):
        denormalized = torch.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            if i not in self.log_transform_indices:
                denormalized[:, i] = (
                    predictions[:, i] * (self.param_maxs[i] - self.param_mins[i]) + self.param_mins[i]
                )
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = (
                predictions[:, param_idx] * (self.log_maxs[i] - self.log_mins[i]) + self.log_mins[i]
            )
            denormalized[:, param_idx] = torch.exp(log_values) - 1
        return denormalized

    def extract_spatial_features(self, x):
        x_rgb = x.repeat(1, 3, 1, 1)
        vit_outputs = self.vision_transformer(pixel_values=x_rgb, return_dict=True)
        patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]

        if self.spatial_attention and hasattr(self, "spatial_attention_module"):
            enhanced_features, spatial_weights = self.spatial_attention_module(patch_embeddings)
            self.spatial_attention_weights.append(spatial_weights)
        else:
            enhanced_features = patch_embeddings

        pooled_features = torch.mean(enhanced_features, dim=1)
        return pooled_features

    def forward(self, views, return_aux=False):
        batch_size = views[0].shape[0]
        view_features = []
        self.spatial_attention_weights = []

        for view_idx, view in enumerate(views):
            if len(view.shape) == 4:
                frames, height, width = view.shape[1], view.shape[2], view.shape[3]
            elif len(view.shape) == 3:
                view = view.unsqueeze(0)
                frames, height, width = view.shape[1], view.shape[2], view.shape[3]
            else:
                raise ValueError(f"Unexpected view shape: {view.shape}")

            actual_batch_size = view.shape[0]
            if actual_batch_size != batch_size:
                batch_size = actual_batch_size

            view_flat = view.reshape(batch_size * frames, 1, height, width)
            frame_features = self.extract_spatial_features(view_flat)
            frame_features = frame_features.reshape(batch_size, frames, -1)

            if self.temporal_attention and hasattr(self, "temporal_attention_module"):
                attended_view, temp_att, frame_weights = self.temporal_attention_module(frame_features)
                if self.temporal_attention_weights is None:
                    self.temporal_attention_weights = temp_att
            else:
                attended_view = torch.mean(frame_features, dim=1)

            view_features.append(attended_view)

        stacked_views = torch.stack(view_features, dim=1)

        if (
            self.fusion_attention
            and hasattr(self, "fusion_attention_module")
            and self.fusion_attention_module is not None
        ):
            global_view_context = torch.mean(stacked_views, dim=1)
            view_weights = self.fusion_attention_module(global_view_context)
            weighted_views = stacked_views * view_weights.unsqueeze(-1)
            fused_features = torch.sum(weighted_views, dim=1)
            self.fusion_attention_weights = view_weights
        else:
            fused_features = torch.mean(stacked_views, dim=1)

        parameter_predictions = [head(fused_features) for head in self.regression_heads]
        predictions = torch.cat(parameter_predictions, dim=1)

        if return_aux:
            return predictions, None
        return predictions
