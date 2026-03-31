#!/usr/bin/env python3
"""
CardioAI model: ViT-based multi-view cardiac ultrasound regression with
progressive unfreezing and discriminative learning rates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import numpy as np
import warnings
warnings.filterwarnings("ignore")



class ParameterHeadWithResidual(nn.Module):
    """Regression head: 768 -> LayerNorm -> 256 -> GELU -> 128 -> GELU -> 1"""

    def __init__(self, hidden_size=768, dropout_rate=0.15):
        super().__init__()

        # Deep task-specific path: 768 -> 256 -> 128 -> 1 (NO residual bypass)
        self.main_path = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),  # Increased from 128
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),  # Additional layer
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.main_path(x)


class SimplifiedTemporalAggregation(nn.Module):
    """Temporal aggregation with learnable gating between attention and mean-pooling paths.
    Combines frame-importance-weighted aggregation with a residual mean-pooling path,
    balanced by a learned sigmoid gate (alpha * attention + (1-alpha) * residual).
    """

    def __init__(self, hidden_size, num_frames=32, dropout_rate=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_frames = num_frames

        # Learnable frame importance weights (768 -> num_frames projection)
        self.frame_importance = nn.Sequential(
            nn.Linear(hidden_size, num_frames),
            nn.Dropout(dropout_rate),
            nn.Softmax(dim=1)
        )

        # Learnable gating parameter for attention vs residual balance
        # Initialized to 0.0 -> sigmoid(0) = 0.5 (equal weighting initially)
        self.gate_weight = nn.Parameter(torch.zeros(1))

        # Optional: Simple feature enhancement before aggregation
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, frame_features):
        # frame_features: (batch_size, num_frames, hidden_size)
        # Handle different input shapes
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
        
        # Normalize input features
        frame_features = self.feature_norm(frame_features)
        frame_features = self.dropout(frame_features)
        
        # Compute frame importance by averaging features across frames and projecting
        # Global context: (batch_size, hidden_size)
        global_context = torch.mean(frame_features, dim=1)
        
        # Frame importance weights: (batch_size, num_frames)
        frame_weights = self.frame_importance(global_context)
        
        # Ensure frame_weights matches actual number of frames
        if frame_weights.shape[1] != num_frames:
            if num_frames < self.num_frames:
                # Truncate to actual number of frames
                frame_weights = frame_weights[:, :num_frames]
                frame_weights = F.softmax(frame_weights, dim=1)  # Renormalize
            else:
                # Interpolate for more frames than expected
                frame_weights = F.interpolate(
                    frame_weights.unsqueeze(1),
                    size=num_frames,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                frame_weights = F.softmax(frame_weights, dim=1)  # Renormalize
        
        # Attention path: weighted aggregation across frames
        attention_aggregated = torch.sum(frame_features * frame_weights.unsqueeze(-1), dim=1)

        # Residual path: mean pooling
        temporal_residual = torch.mean(frame_features, dim=1)

        # Learnable gating: alpha * attention + (1 - alpha) * residual
        alpha = torch.sigmoid(self.gate_weight)
        aggregated_features = alpha * attention_aggregated + (1 - alpha) * temporal_residual

        return aggregated_features, None, frame_weights




class ProgressiveCardioAI(nn.Module):
    """Multi-view cardiac ultrasound model with progressive ViT unfreezing.

    Pipeline: ViT feature extraction -> temporal aggregation (per-view) ->
    view fusion (cross-view) -> parameter-specific regression heads.
    Both temporal and fusion stages use learnable gating between attention
    and mean-pooling residual paths.
    """
    
    def __init__(self, num_outputs=9, num_frames=32, num_views=4, dropout_rate=0.15, ablation_attentions='temporal'):
        super().__init__()
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_outputs = num_outputs
        
        # Parse ablation configuration - temporal and fusion modules available
        self.ablation_config = ablation_attentions.lower()
        if self.ablation_config == 'none':
            self.use_temporal_attention = False
            self.use_fusion_attention = False
        elif 'temporal' in self.ablation_config and 'fusion' in self.ablation_config:
            self.use_temporal_attention = True
            self.use_fusion_attention = True
        elif 'temporal' in self.ablation_config:
            self.use_temporal_attention = True
            self.use_fusion_attention = False
        elif 'fusion' in self.ablation_config:
            self.use_temporal_attention = False
            self.use_fusion_attention = True
        else:
            # Default: use both
            self.use_temporal_attention = True
            self.use_fusion_attention = True
        
        try:
            self.vision_transformer = ViTModel.from_pretrained('google/vit-base-patch16-224')
            print(f"Loaded ViT-Base: hidden={self.vision_transformer.config.hidden_size}, "
                  f"heads={self.vision_transformer.config.num_attention_heads}, "
                  f"layers={self.vision_transformer.config.num_hidden_layers}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained ViT-Base: {e}")
        
        self.hidden_size = 768
        assert self.vision_transformer.config.hidden_size == 768

        self.temporal_attention = SimplifiedTemporalAggregation(self.hidden_size, num_frames, dropout_rate) if self.use_temporal_attention else None
        self.parameter_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
        self.view_names = ['FC', 'TC', 'SA', 'LA']
        self.regression_heads = nn.ModuleList([
            self._create_parameter_head(self.hidden_size, dropout_rate, param_name) 
            for param_name in self.parameter_names
        ])
        
        if self.use_fusion_attention:
            self.view_attention = nn.Sequential(
                nn.Linear(self.hidden_size, len(self.view_names)),
                nn.Dropout(dropout_rate),
                nn.Softmax(dim=1)
            )
            self.fusion_gate_weight = nn.Parameter(torch.zeros(1))
        else:
            self.view_attention = None
            self.fusion_gate_weight = None

        self.fusion_attention = self.view_attention
        self.main_regressor = None

        # Normalization placeholders (overwritten by dataset winsorized bounds)
        self.register_buffer('param_mins', torch.tensor([
            0.0, 18.0, 5.0, 10.0, 1.0, 2.0, 1.2, 3.85, 0.56
        ]))
        self.register_buffer('param_maxs', torch.tensor([
            22.0, 111.0, 75.0, 71.0, 25.0, 10.0, 5.5, 5576.0, 19.74
        ]))

        # Log transform parameters for SVRI
        self.log_transform_indices = [7]  # SVRI
        self.register_buffer('log_mins', torch.tensor([np.log(3.85 + 1)]))
        self.register_buffer('log_maxs', torch.tensor([np.log(5576.0 + 1)]))

        # Flag to track if winsorized normalization has been set
        self.using_winsorized_normalization = False
        
        # Initialize task modules properly
        self._init_task_modules()
        
        # Freeze ViT initially - will be unfrozen progressively
        self.freeze_vit_backbone()
        
        # Storage for attention weights (for visualization)
        self.temporal_attention_weights = None
        
    def _create_parameter_head(self, hidden_size, dropout_rate, param_name):
        return ParameterHeadWithResidual(hidden_size, dropout_rate)
        
    def _init_task_modules(self):
        """Xavier/Glorot initialization for task modules."""
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
        """Freeze all ViT parameters; task layers remain trainable."""
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
        print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.1f}%)")
    
    def unfreeze_vit_block(self, block_idx):
        """Unfreeze a specific ViT transformer block by index."""
        if hasattr(self.vision_transformer, 'encoder') and hasattr(self.vision_transformer.encoder, 'layer'):
            if block_idx < len(self.vision_transformer.encoder.layer):
                for param in self.vision_transformer.encoder.layer[block_idx].parameters():
                    param.requires_grad = True
                print(f"Unfroze ViT transformer block {block_idx}")
            else:
                print(f"Warning: Block {block_idx} does not exist (max: {len(self.vision_transformer.encoder.layer)-1})")
        else:
            print(f"Warning: Could not find encoder layers in ViT model structure")
    
    def set_winsorized_normalization(self, norm_params):
        """Set normalization parameters from winsorized dataset bounds."""
        self.param_mins = torch.tensor(norm_params['param_mins'], dtype=torch.float32).to(self.param_mins.device)
        self.param_maxs = torch.tensor(norm_params['param_maxs'], dtype=torch.float32).to(self.param_maxs.device)

        if len(norm_params['log_mins']) > 0:
            self.log_mins = torch.tensor(norm_params['log_mins'], dtype=torch.float32).to(self.log_mins.device)
            self.log_maxs = torch.tensor(norm_params['log_maxs'], dtype=torch.float32).to(self.log_maxs.device)

        self.using_winsorized_normalization = True

        param_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
        print("Normalization bounds (winsorized):")
        for i, name in enumerate(param_names):
            print(f"  {name}: [{self.param_mins[i]:.2f}, {self.param_maxs[i]:.2f}]")

    def get_trainable_parameters(self):
        """Get trainable parameters grouped by component for discriminative learning rates"""
        # Task layers (highest learning rate)
        task_params = []
        task_modules = [self.temporal_attention] if self.temporal_attention is not None else []
        task_modules.extend(list(self.regression_heads))
        if self.view_attention is not None:
            task_modules.append(self.view_attention)  # Add view attention module if enabled
        for module in task_modules:
            task_params.extend(list(module.parameters()))

        # Unfrozen ViT blocks (lower learning rate)
        vit_params = []
        for param in self.vision_transformer.parameters():
            if param.requires_grad:
                vit_params.append(param)

        return {
            'task_params': task_params,
            'vit_params': vit_params
        }
    
    def normalize_targets(self, targets):
        """Normalize targets to consistent scale"""
        normalized = torch.zeros_like(targets)
        
        # Standard parameters: Min-Max scaling to [0, 1]
        for i in range(self.num_outputs):
            if i not in self.log_transform_indices:
                normalized[:, i] = (targets[:, i] - self.param_mins[i]) / (self.param_maxs[i] - self.param_mins[i] + 1e-8)
        
        # Log-transform for high-variance parameters (SVRI)
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = torch.log(targets[:, param_idx] + 1)
            normalized[:, param_idx] = (log_values - self.log_mins[i]) / (self.log_maxs[i] - self.log_mins[i] + 1e-8)
        
        return normalized
    
    def denormalize_predictions(self, predictions):
        """Convert normalized predictions back to original scales

        IMPORTANT: Predictions are now in [0, 1] range after sigmoid activation.
        Safety clamping handles edge cases near 0/1 boundaries.
        """
        # Safety clamp: ensure predictions are in valid [0, 1] range
        # Prevents NaN in log-transform denormalization and handles numerical edge cases
        predictions = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)

        denormalized = torch.zeros_like(predictions)

        # Standard parameters: reverse min-max scaling
        for i in range(self.num_outputs):
            if i not in self.log_transform_indices:
                denormalized[:, i] = predictions[:, i] * (self.param_maxs[i] - self.param_mins[i]) + self.param_mins[i]

        # Reverse log-transform for SVRI
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = predictions[:, param_idx] * (self.log_maxs[i] - self.log_mins[i]) + self.log_mins[i]
            denormalized[:, param_idx] = torch.exp(log_values) - 1

        return denormalized
    
    def extract_spatial_features(self, x):
        """Extract spatial features using Vision Transformer"""
        # x: (batch_size, 1, 224, 224) - grayscale input
        
        # Convert grayscale to RGB by repeating channels
        x_rgb = x.repeat(1, 3, 1, 1)  # (batch_size, 3, 224, 224)
        
        # Pass through ViT backbone
        vit_outputs = self.vision_transformer(pixel_values=x_rgb, return_dict=True)
        
        # Get patch embeddings (excluding CLS token)
        patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]  # (batch, num_patches, hidden_size)
        
        # No spatial attention - use patch embeddings directly
        enhanced_features = patch_embeddings
        
        # Global average pooling across patches
        pooled_features = torch.mean(enhanced_features, dim=1)  # (batch, hidden_size)
        
        return pooled_features
    
    def forward(self, views, return_aux=False):
        """Forward pass with dual-stage residual connections"""
        batch_size = views[0].shape[0]
        view_features = []

        # Clear previous attention weights
        self.temporal_attention_weights = None

        # Process each view through transformer backbone
        for view_idx, view in enumerate(views):
            # Handle different view shapes safely
            if len(view.shape) == 4:  # (batch, frames, height, width)
                frames, height, width = view.shape[1], view.shape[2], view.shape[3]
            elif len(view.shape) == 3:  # (frames, height, width) - missing batch
                view = view.unsqueeze(0)  # Add batch dimension
                frames, height, width = view.shape[1], view.shape[2], view.shape[3]
            else:
                raise ValueError(f"Unexpected view shape: {view.shape}")

            # Update batch size if needed
            actual_batch_size = view.shape[0]
            if actual_batch_size != batch_size:
                batch_size = actual_batch_size

            # Reshape for batch processing
            view_flat = view.reshape(batch_size * frames, 1, height, width)

            # Extract spatial features with transformer
            frame_features = self.extract_spatial_features(view_flat)
            frame_features = frame_features.reshape(batch_size, frames, -1)

            # Apply temporal attention if enabled
            if self.use_temporal_attention and self.temporal_attention is not None:
                attended_view, temp_att, frame_weights = self.temporal_attention(frame_features)
                if self.temporal_attention_weights is None:
                    self.temporal_attention_weights = temp_att
                view_features.append(attended_view)
            else:
                # Simple temporal aggregation: mean pooling across frames
                attended_view = torch.mean(frame_features, dim=1)  # (batch_size, hidden_size)
                view_features.append(attended_view)
        
        # ═══════════════════════════════════════════════════════
        # VIEW FUSION with Residual Connection (Strategy 1)
        # ═══════════════════════════════════════════════════════
        # Input: 4 views × 768-dim features
        stacked_views = torch.stack(view_features, dim=1)  # (batch_size, 4, 768)

        if self.use_fusion_attention and self.view_attention is not None:
            # Attention path: learned view weighting
            global_view_context = torch.mean(stacked_views, dim=1)
            view_weights = self.view_attention(global_view_context)
            weighted_views = stacked_views * view_weights.unsqueeze(-1)
            attention_fusion = torch.sum(weighted_views, dim=1)

            # Residual path: mean pooling
            fusion_residual = torch.mean(stacked_views, dim=1)

            # Learnable gating: beta * attention + (1 - beta) * residual
            beta = torch.sigmoid(self.fusion_gate_weight)
            unified_features = beta * attention_fusion + (1 - beta) * fusion_residual
        else:
            unified_features = torch.mean(stacked_views, dim=1)

        # Parameter-specific regression heads
        parameter_predictions = []
        for head in self.regression_heads:
            parameter_predictions.append(head(unified_features))

        predictions = torch.cat(parameter_predictions, dim=1)  # (batch_size, 9)
        predictions = torch.sigmoid(predictions)  # constrain to [0, 1]
        
        if return_aux:
            return predictions, None  # No uncertainty head
        else:
            return predictions
    
    def attention_rollout(self, views, target_param_idx=0, head_fusion='mean'):
        """Compute attention rollout for temporal attention analysis"""
        if not self.use_temporal_attention or self.temporal_attention is None:
            print("Warning: No temporal attention enabled for rollout analysis")
            return {}

        self.eval()
        with torch.no_grad():
            batch_size = views[0].shape[0]
            rollout_scores = {}

            for view_idx, view in enumerate(views):
                # Handle different view shapes
                if len(view.shape) == 4:
                    frames = view.shape[1]
                    height, width = view.shape[2], view.shape[3]
                elif len(view.shape) == 3:
                    frames = view.shape[0]
                    height, width = view.shape[1], view.shape[2]
                    view = view.unsqueeze(0)  # Add batch dimension
                else:
                    continue

                view_name = ['FC', 'TC', 'SA', 'LA'][view_idx] if view_idx < 4 else f'View_{view_idx}'

                # Process view through temporal attention
                view_flat = view.reshape(batch_size * frames, 1, height, width)
                frame_features = self.extract_spatial_features(view_flat)
                frame_features = frame_features.reshape(batch_size, frames, -1)

                # Get temporal attention weights
                _, _, frame_weights = self.temporal_attention(frame_features)

                rollout_scores[view_name] = frame_weights.cpu().numpy()

            return rollout_scores

    def get_integrated_gradients(self, views, target_param_idx=0, n_steps=50):
        """Compute Integrated Gradients (Sundararajan et al., ICML 2017) for input attribution.

        Returns dict mapping view names to pixel-level attribution tensors (batch, frames, H, W).
        """
        self.eval()
        device = views[0].device
        baseline_views = [torch.zeros_like(view) for view in views]
        ig_results = {}

        for view_idx, (view, baseline) in enumerate(zip(views, baseline_views)):
            view_name = self.view_names[view_idx] if view_idx < len(self.view_names) else f'View_{view_idx}'

            if len(view.shape) == 4:
                batch_size = view.shape[0]
            elif len(view.shape) == 3:
                view = view.unsqueeze(0)
                baseline = baseline.unsqueeze(0)
                batch_size = 1
            else:
                continue

            alphas = torch.linspace(0, 1, n_steps).to(device)
            accumulated_gradients = torch.zeros_like(view)

            for alpha in alphas:
                interpolated = baseline + alpha * (view - baseline)
                interpolated.requires_grad = True

                views_copy = [baseline_views[i].clone() if i != view_idx else interpolated
                             for i in range(len(views))]

                predictions = self.forward(views_copy, return_aux=False)
                target_output = predictions[:, target_param_idx]
                loss = target_output.sum()
                loss.backward(retain_graph=False)

                if interpolated.grad is not None:
                    accumulated_gradients += interpolated.grad.detach()
                interpolated.grad = None

            avg_gradients = accumulated_gradients / n_steps
            integrated_gradients = (view - baseline) * avg_gradients
            ig_results[view_name] = integrated_gradients.detach()

        return ig_results


def create_model(num_outputs=9, num_frames=32, num_views=4, dropout_rate=0.15, ablation_attentions='temporal'):
    """Create progressive CardioAI model with temporal attention only"""
    model = ProgressiveCardioAI(
        num_outputs=num_outputs,
        num_frames=num_frames,
        num_views=num_views,
        dropout_rate=dropout_rate,
        ablation_attentions=ablation_attentions
    )
    
    print(f"CardioAI model: ViT-Base backbone, attention={ablation_attentions}")
    
    return model


def create_progressive_optimizer(model, task_lr=1e-4, vit_lr=1e-5, weight_decay=1e-4):
    """Create AdamW optimizer with discriminative learning rates (task vs ViT)."""
    param_groups = model.get_trainable_parameters()
    optimizer_params = []

    if param_groups['task_params']:
        optimizer_params.append({'params': param_groups['task_params'], 'lr': task_lr, 'weight_decay': weight_decay})
    if param_groups['vit_params']:
        optimizer_params.append({'params': param_groups['vit_params'], 'lr': vit_lr, 'weight_decay': weight_decay})

    optimizer = torch.optim.AdamW(optimizer_params)
    print(f"Optimizer: task_lr={task_lr}, vit_lr={vit_lr} ({task_lr/vit_lr:.1f}x ratio)")
    return optimizer