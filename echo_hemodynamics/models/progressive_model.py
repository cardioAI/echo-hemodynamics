"""ProgressiveCardioAI: ViT-based multi-view cardiac ultrasound regression model."""

import warnings

import numpy as np
import torch
import torch.nn as nn
from transformers import ViTModel

from .explainability import ExplainabilityMixin
from .heads import ParameterHeadWithResidual
from .temporal_attention import SimplifiedTemporalAggregation

warnings.filterwarnings("ignore")


class ProgressiveCardioAI(ExplainabilityMixin, nn.Module):
    """Multi-view cardiac ultrasound model with progressive ViT unfreezing.

    Pipeline: ViT feature extraction -> temporal aggregation (per-view) ->
    view fusion (cross-view) -> parameter-specific regression heads.
    Both temporal and fusion stages use learnable gating between attention
    and mean-pooling residual paths.
    """

    def __init__(self, num_outputs=9, num_frames=32, num_views=4, dropout_rate=0.15,
                 ablation_attentions="temporal"):
        super().__init__()
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_outputs = num_outputs

        self.ablation_config = ablation_attentions.lower()
        if self.ablation_config == "none":
            self.use_temporal_attention = False
            self.use_fusion_attention = False
        elif "temporal" in self.ablation_config and "fusion" in self.ablation_config:
            self.use_temporal_attention = True
            self.use_fusion_attention = True
        elif "temporal" in self.ablation_config:
            self.use_temporal_attention = True
            self.use_fusion_attention = False
        elif "fusion" in self.ablation_config:
            self.use_temporal_attention = False
            self.use_fusion_attention = True
        else:
            self.use_temporal_attention = True
            self.use_fusion_attention = True

        try:
            self.vision_transformer = ViTModel.from_pretrained("google/vit-base-patch16-224")
            print(
                f"Loaded ViT-Base: hidden={self.vision_transformer.config.hidden_size}, "
                f"heads={self.vision_transformer.config.num_attention_heads}, "
                f"layers={self.vision_transformer.config.num_hidden_layers}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained ViT-Base: {e}")

        self.hidden_size = 768
        assert self.vision_transformer.config.hidden_size == 768

        self.temporal_attention = (
            SimplifiedTemporalAggregation(self.hidden_size, num_frames, dropout_rate)
            if self.use_temporal_attention
            else None
        )
        self.parameter_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
        self.view_names = ["FC", "TC", "SA", "LA"]
        self.regression_heads = nn.ModuleList([
            self._create_parameter_head(self.hidden_size, dropout_rate, param_name)
            for param_name in self.parameter_names
        ])

        if self.use_fusion_attention:
            self.view_attention = nn.Sequential(
                nn.Linear(self.hidden_size, len(self.view_names)),
                nn.Dropout(dropout_rate),
                nn.Softmax(dim=1),
            )
            self.fusion_gate_weight = nn.Parameter(torch.zeros(1))
        else:
            self.view_attention = None
            self.fusion_gate_weight = None

        self.fusion_attention = self.view_attention
        self.main_regressor = None

        # Default normalization (overwritten by dataset winsorized bounds)
        self.register_buffer("param_mins", torch.tensor([
            0.0, 18.0, 5.0, 10.0, 1.0, 2.0, 1.2, 3.85, 0.56
        ]))
        self.register_buffer("param_maxs", torch.tensor([
            22.0, 111.0, 75.0, 71.0, 25.0, 10.0, 5.5, 5576.0, 19.74
        ]))

        self.log_transform_indices = [7]  # SVRI
        self.register_buffer("log_mins", torch.tensor([np.log(3.85 + 1)]))
        self.register_buffer("log_maxs", torch.tensor([np.log(5576.0 + 1)]))

        self.using_winsorized_normalization = False

        self._init_task_modules()
        self.freeze_vit_backbone()

        self.temporal_attention_weights = None

    def _create_parameter_head(self, hidden_size, dropout_rate, param_name):
        return ParameterHeadWithResidual(hidden_size, dropout_rate)

    def _init_task_modules(self):
        modules_to_init = [self.temporal_attention] if self.temporal_attention is not None else []
        modules_to_init.extend(list(self.regression_heads))
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Parameter):
                    nn.init.normal_(m, std=0.02)

        if self.use_fusion_attention and self.view_attention is not None:
            with torch.no_grad():
                nn.init.xavier_uniform_(self.view_attention[0].weight)
                nn.init.constant_(self.view_attention[0].bias, 0.0)

    def freeze_vit_backbone(self):
        for param in self.vision_transformer.parameters():
            param.requires_grad = False

        task_modules = [self.temporal_attention] if self.temporal_attention is not None else []
        task_modules.extend(list(self.regression_heads))
        if self.view_attention is not None:
            task_modules.append(self.view_attention)
        for module in task_modules:
            for param in module.parameters():
                param.requires_grad = True

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
            f"({100 * trainable_params / total_params:.1f}%)"
        )

    def unfreeze_vit_block(self, block_idx):
        if hasattr(self.vision_transformer, "encoder") and hasattr(self.vision_transformer.encoder, "layer"):
            if block_idx < len(self.vision_transformer.encoder.layer):
                for param in self.vision_transformer.encoder.layer[block_idx].parameters():
                    param.requires_grad = True
                print(f"Unfroze ViT transformer block {block_idx}")
            else:
                print(
                    f"Warning: Block {block_idx} does not exist "
                    f"(max: {len(self.vision_transformer.encoder.layer) - 1})"
                )
        else:
            print("Warning: Could not find encoder layers in ViT model structure")

    def set_winsorized_normalization(self, norm_params):
        self.param_mins = torch.tensor(norm_params["param_mins"], dtype=torch.float32).to(self.param_mins.device)
        self.param_maxs = torch.tensor(norm_params["param_maxs"], dtype=torch.float32).to(self.param_maxs.device)

        if len(norm_params["log_mins"]) > 0:
            self.log_mins = torch.tensor(norm_params["log_mins"], dtype=torch.float32).to(self.log_mins.device)
            self.log_maxs = torch.tensor(norm_params["log_maxs"], dtype=torch.float32).to(self.log_maxs.device)

        self.using_winsorized_normalization = True

        print("Normalization bounds (winsorized):")
        for i, name in enumerate(self.parameter_names):
            print(f"  {name}: [{self.param_mins[i]:.2f}, {self.param_maxs[i]:.2f}]")

    def get_trainable_parameters(self):
        task_params = []
        task_modules = [self.temporal_attention] if self.temporal_attention is not None else []
        task_modules.extend(list(self.regression_heads))
        if self.view_attention is not None:
            task_modules.append(self.view_attention)
        for module in task_modules:
            task_params.extend(list(module.parameters()))

        vit_params = [p for p in self.vision_transformer.parameters() if p.requires_grad]
        return {"task_params": task_params, "vit_params": vit_params}

    def normalize_targets(self, targets):
        normalized = torch.zeros_like(targets)
        for i in range(self.num_outputs):
            if i not in self.log_transform_indices:
                normalized[:, i] = (targets[:, i] - self.param_mins[i]) / (self.param_maxs[i] - self.param_mins[i] + 1e-8)
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = torch.log(targets[:, param_idx] + 1)
            normalized[:, param_idx] = (log_values - self.log_mins[i]) / (self.log_maxs[i] - self.log_mins[i] + 1e-8)
        return normalized

    def denormalize_predictions(self, predictions):
        predictions = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)
        denormalized = torch.zeros_like(predictions)
        for i in range(self.num_outputs):
            if i not in self.log_transform_indices:
                denormalized[:, i] = predictions[:, i] * (self.param_maxs[i] - self.param_mins[i]) + self.param_mins[i]
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = predictions[:, param_idx] * (self.log_maxs[i] - self.log_mins[i]) + self.log_mins[i]
            denormalized[:, param_idx] = torch.exp(log_values) - 1
        return denormalized

    def extract_spatial_features(self, x):
        x_rgb = x.repeat(1, 3, 1, 1)
        vit_outputs = self.vision_transformer(pixel_values=x_rgb, return_dict=True)
        patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]
        enhanced_features = patch_embeddings
        pooled_features = torch.mean(enhanced_features, dim=1)
        return pooled_features

    def forward(self, views, return_aux=False):
        batch_size = views[0].shape[0]
        view_features = []

        self.temporal_attention_weights = None

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

            if self.use_temporal_attention and self.temporal_attention is not None:
                attended_view, temp_att, frame_weights = self.temporal_attention(frame_features)
                if self.temporal_attention_weights is None:
                    self.temporal_attention_weights = temp_att
                view_features.append(attended_view)
            else:
                attended_view = torch.mean(frame_features, dim=1)
                view_features.append(attended_view)

        stacked_views = torch.stack(view_features, dim=1)

        if self.use_fusion_attention and self.view_attention is not None:
            global_view_context = torch.mean(stacked_views, dim=1)
            view_weights = self.view_attention(global_view_context)
            weighted_views = stacked_views * view_weights.unsqueeze(-1)
            attention_fusion = torch.sum(weighted_views, dim=1)

            fusion_residual = torch.mean(stacked_views, dim=1)

            beta = torch.sigmoid(self.fusion_gate_weight)
            unified_features = beta * attention_fusion + (1 - beta) * fusion_residual
        else:
            unified_features = torch.mean(stacked_views, dim=1)

        parameter_predictions = [head(unified_features) for head in self.regression_heads]
        predictions = torch.cat(parameter_predictions, dim=1)
        predictions = torch.sigmoid(predictions)

        if return_aux:
            return predictions, None
        return predictions
