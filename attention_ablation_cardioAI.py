#!/usr/bin/env python3
"""
Attention ablation study: compares full model (temporal + fusion attention)
against variants with individual or no attention modules.
"""

import os
import sys
import time
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# Local imports
from model_cardioAI import create_model, create_progressive_optimizer, ProgressiveCardioAI
from train_cardioAI import ProgressiveMSELoss
from dataset_cardioAI import CardioAIDataset
from utils_cardioAI import CardioAIUtils, ColorManager


def parse_train_indices(indices_str: str):
    """
    Parse training indices string into a list of indices.

    Args:
        indices_str: String with comma-separated ranges (e.g., "0-179,180-224,235-289")

    Returns:
        List of integer indices
    """
    if not indices_str:
        return None

    indices = []
    ranges = indices_str.split(',')

    for range_str in ranges:
        range_str = range_str.strip()
        if '-' in range_str:
            start, end = range_str.split('-')
            indices.extend(list(range(int(start), int(end) + 1)))
        else:
            indices.append(int(range_str))

    return indices


def find_latest_trained_model():
    """Find the latest trained model from previous training"""
    
    # First, check if there's a train_cardioAI subfolder in current timestamp directory
    current_timestamp = os.environ.get('CARDIOAI_TIMESTAMP', '')
    base_results_dir = Path(r"E:\results_cardioAI\EchoCath_cardioAI")
    
    if current_timestamp:
        current_run_dir = base_results_dir / current_timestamp
        train_subfolder = current_run_dir / "train_cardioAI"
        if train_subfolder.exists() and (train_subfolder / "best_model.pth").exists():
            print(f"Found trained model in current run: {train_subfolder / 'best_model.pth'}")
            return train_subfolder / "best_model.pth"
    
    # If not found in current run, search for latest training in any previous run
    if base_results_dir.exists():
        timestamp_dirs = [d for d in base_results_dir.iterdir() if d.is_dir()]
        
        for timestamp_dir in sorted(timestamp_dirs, key=lambda x: x.name, reverse=True):
            # Check new structure first (train_cardioAI subfolder)
            train_subfolder = timestamp_dir / "train_cardioAI"
            if train_subfolder.exists() and (train_subfolder / "best_model.pth").exists():
                print(f"Found trained model in previous run: {train_subfolder / 'best_model.pth'}")
                return train_subfolder / "best_model.pth"
            
            # Check old structure (01_training subfolder)
            old_training_dir = timestamp_dir / "01_training"
            if old_training_dir.exists() and (old_training_dir / "best_model.pth").exists():
                print(f"Found trained model in previous run (old structure): {old_training_dir / 'best_model.pth'}")
                return old_training_dir / "best_model.pth"
    
    raise FileNotFoundError(
        "No trained model found! Please run training first before ablation study.\n"
        "Expected locations:\n"
        f"  - {current_run_dir / 'train_cardioAI' / 'best_model.pth'}\n"
        f"  - Previous runs in {base_results_dir}"
    )


class ProgressiveAblationVariant(nn.Module):
    """Progressive model-based ablation variant with selective attention components"""
    
    def __init__(self, base_model, spatial_attention=True, temporal_attention=True, 
                 fusion_attention=True):
        super().__init__()
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.fusion_attention = fusion_attention
        self.num_frames = base_model.num_frames
        self.num_views = base_model.num_views
        self.hidden_size = base_model.hidden_size

        # Deep copy to ensure variants start fresh
        self.vision_transformer = copy.deepcopy(base_model.vision_transformer)

        # Copy normalization parameters (these can be shared as buffers)
        self.register_buffer('param_mins', base_model.param_mins.clone())
        self.register_buffer('param_maxs', base_model.param_maxs.clone())
        self.register_buffer('log_mins', base_model.log_mins.clone())
        self.register_buffer('log_maxs', base_model.log_maxs.clone())
        self.log_transform_indices = base_model.log_transform_indices.copy()

        # Conditionally copy attention modules using deepcopy
        # Note: Spatial features are extracted by ViT backbone, not a separate attention module
        # Current architecture has temporal and fusion attention modules

        if temporal_attention:
            self.temporal_attention_module = copy.deepcopy(base_model.temporal_attention)
        else:
            # Simple temporal aggregation
            self.simple_temporal_pool = nn.AdaptiveAvgPool1d(1)

        if fusion_attention:
            self.fusion_attention_module = copy.deepcopy(base_model.view_attention)  # Current model uses view_attention
        else:
            # Simple view aggregation - average views with equal weights
            self.simple_view_weights = nn.Parameter(torch.ones(4) / 4)

        self.regression_heads = copy.deepcopy(base_model.regression_heads)
        self.parameter_names = base_model.parameter_names.copy()
        
        # Store attention weights (for compatibility)
        self.spatial_attention_weights = []
        self.temporal_attention_weights = None
        self.fusion_attention_weights = None
    
    def normalize_targets(self, targets):
        """Use base model's normalization"""
        normalized = torch.zeros_like(targets)
        
        # Standard parameters: Min-Max scaling to [0, 1]
        for i in range(targets.shape[1]):
            if i not in self.log_transform_indices:
                normalized[:, i] = (targets[:, i] - self.param_mins[i]) / (self.param_maxs[i] - self.param_mins[i] + 1e-8)
        
        # Log-transform for high-variance parameters (SVRI)
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = torch.log(targets[:, param_idx] + 1)
            normalized[:, param_idx] = (log_values - self.log_mins[i]) / (self.log_maxs[i] - self.log_mins[i] + 1e-8)
        
        return normalized
    
    def denormalize_predictions(self, predictions):
        """Use base model's denormalization"""
        denormalized = torch.zeros_like(predictions)
        
        # Standard parameters: reverse min-max scaling
        for i in range(predictions.shape[1]):
            if i not in self.log_transform_indices:
                denormalized[:, i] = predictions[:, i] * (self.param_maxs[i] - self.param_mins[i]) + self.param_mins[i]
        
        # Reverse log-transform for SVRI
        for i, param_idx in enumerate(self.log_transform_indices):
            log_values = predictions[:, param_idx] * (self.log_maxs[i] - self.log_mins[i]) + self.log_mins[i]
            denormalized[:, param_idx] = torch.exp(log_values) - 1
        
        return denormalized
    
    def extract_spatial_features(self, x):
        """Extract spatial features with conditional spatial attention"""
        # Convert grayscale to RGB
        x_rgb = x.repeat(1, 3, 1, 1)
        
        # Pass through ViT backbone
        vit_outputs = self.vision_transformer(pixel_values=x_rgb, return_dict=True)
        patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]
        
        if self.spatial_attention and hasattr(self, 'spatial_attention_module'):
            # Apply additional spatial attention
            enhanced_features, spatial_weights = self.spatial_attention_module(patch_embeddings)
            self.spatial_attention_weights.append(spatial_weights)
        else:
            # Skip spatial attention - just use patch embeddings
            enhanced_features = patch_embeddings
        
        # Global average pooling across patches
        pooled_features = torch.mean(enhanced_features, dim=1)
        return pooled_features
    
    def forward(self, views, return_aux=False):
        """Forward pass with selective attention mechanisms"""
        batch_size = views[0].shape[0]
        view_features = []
        
        # Clear previous attention weights
        self.spatial_attention_weights = []
        
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
            
            # Apply temporal attention or simple pooling
            if self.temporal_attention and hasattr(self, 'temporal_attention_module'):
                attended_view, temp_att, frame_weights = self.temporal_attention_module(frame_features)
                if self.temporal_attention_weights is None:
                    self.temporal_attention_weights = temp_att
            else:
                # Simple temporal pooling - average across frames
                attended_view = torch.mean(frame_features, dim=1)
            
            view_features.append(attended_view)
        
        # View fusion attention or simple pooling
        stacked_views = torch.stack(view_features, dim=1)  # (batch, num_views, hidden_size)
        
        if self.fusion_attention and hasattr(self, 'fusion_attention_module') and self.fusion_attention_module is not None:
            # Apply view attention mechanism (768 -> 4 projection)
            global_view_context = torch.mean(stacked_views, dim=1)  # (batch_size, 768)
            view_weights = self.fusion_attention_module(global_view_context)  # (batch_size, 4)
            weighted_views = stacked_views * view_weights.unsqueeze(-1)  # (batch_size, 4, 768)
            fused_features = torch.sum(weighted_views, dim=1)  # (batch_size, 768)
            self.fusion_attention_weights = view_weights
        else:
            # Simple view aggregation - average all views equally
            fused_features = torch.mean(stacked_views, dim=1)  # (batch_size, 768)
        
        # Parameter-specific predictions using specialized heads
        parameter_predictions = []
        for head in self.regression_heads:
            param_pred = head(fused_features)
            parameter_predictions.append(param_pred)
        
        # Concatenate all parameter predictions
        predictions = torch.cat(parameter_predictions, dim=1)  # (batch_size, 9)
        
        if return_aux:
            return predictions, None  # No uncertainty head in progressive model
        else:
            return predictions


class ProgressiveAblationTrainer:
    """Trainer for progressive model ablation study"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', epochs=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        # Loss function - use same as progressive training
        self.loss_fn = ProgressiveMSELoss(device)
        
        # Optimizer - use simpler setup for ablation
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_correlations': []
        }
    
    def compute_correlations(self, predictions, targets):
        """Compute correlation coefficients for each parameter

        Args:
            predictions: Normalized predictions [0,1] from model output
            targets: Original (denormalized) target values in real units
        """
        correlations = []

        # Denormalize predictions only (targets are already in original scale)
        pred_denorm = self.model.denormalize_predictions(predictions)
        targ_denorm = targets  # Targets already in original units
        
        for i in range(predictions.shape[1]):
            pred_i = pred_denorm[:, i].detach().cpu().numpy()
            targ_i = targ_denorm[:, i].detach().cpu().numpy()
            
            # Compute Pearson correlation
            if np.std(pred_i) > 1e-8 and np.std(targ_i) > 1e-8:
                corr = np.corrcoef(pred_i, targ_i)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            
            correlations.append(abs(corr))  # Use absolute correlation
        
        return correlations
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (views, targets, patient_ids) in enumerate(self.train_loader):
            # Move to device
            views = [view.to(self.device) for view in views]
            targets = targets.to(self.device)
            
            # Normalize targets
            targets_norm = self.model.normalize_targets(targets)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                predictions = self.model(views, return_aux=False)
                
                # Compute loss
                loss = self.loss_fn(predictions, targets_norm)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        return np.mean(epoch_losses) if epoch_losses else float('inf')
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (views, targets, patient_ids) in enumerate(self.val_loader):
                # Move to device
                views = [view.to(self.device) for view in views]
                targets = targets.to(self.device)
                
                # Normalize targets
                targets_norm = self.model.normalize_targets(targets)
                
                # Forward pass
                try:
                    predictions = self.model(views, return_aux=False)
                    loss = self.loss_fn(predictions, targets_norm)

                    val_losses.append(loss.item())
                    all_predictions.append(predictions)
                    all_targets.append(targets)  # Store original (denormalized) targets for correlation
                    
                except Exception as e:
                    continue
        
        # Compute validation metrics
        if val_losses and all_predictions:
            avg_val_loss = np.mean(val_losses)
            all_pred = torch.cat(all_predictions, dim=0)
            all_targ = torch.cat(all_targets, dim=0)
            val_correlations = self.compute_correlations(all_pred, all_targ)
        else:
            avg_val_loss = float('inf')
            val_correlations = [0.0] * 9
        
        return avg_val_loss, val_correlations
    
    def train(self):
        """Main training loop"""
        print(f"Training ablation variant for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, val_corr = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_correlations'].append(np.mean(val_corr))
            
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                avg_corr = np.mean(val_corr)
                print(f"  Epoch {epoch+1:2d}/{self.epochs}: Loss {train_loss:.4f}/{val_loss:.4f}, Corr {avg_corr:.3f}")
        
        return self.history


def create_fresh_model_for_variant(base_model, norm_params):
    """Create a fresh model with ImageNet-only pre-training (no 20-epoch training)

    This ensures ablation variants start from scratch rather than inheriting
    the 20-epoch trained weights from the full model.
    """
    fresh_model = create_model(
        num_outputs=9,
        num_frames=base_model.num_frames,
        num_views=4,
        dropout_rate=0.15
    )
    # Set normalization parameters to match full model
    fresh_model.set_winsorized_normalization(norm_params)
    return fresh_model


def create_ablation_variants(base_model, norm_params):
    """Create ablation variants with different attention module combinations."""
    # Create fresh base models for each variant (ImageNet-only pre-training)
    fresh_base_no_temporal = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_no_fusion = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_no_attention = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_temporal_only = create_fresh_model_for_variant(base_model, norm_params)
    fresh_base_fusion_only = create_fresh_model_for_variant(base_model, norm_params)

    variants = {
        'full_model': base_model,  # Keep the 20-epoch trained model
        'no_temporal': ProgressiveAblationVariant(fresh_base_no_temporal, spatial_attention=False, temporal_attention=False, fusion_attention=True),
        'no_fusion': ProgressiveAblationVariant(fresh_base_no_fusion, spatial_attention=False, temporal_attention=True, fusion_attention=False),
        'no_attention': ProgressiveAblationVariant(fresh_base_no_attention, spatial_attention=False, temporal_attention=False, fusion_attention=False),
        'temporal_only': ProgressiveAblationVariant(fresh_base_temporal_only, spatial_attention=False, temporal_attention=True, fusion_attention=False),
        'fusion_only': ProgressiveAblationVariant(fresh_base_fusion_only, spatial_attention=False, temporal_attention=False, fusion_attention=True)
    }
    return variants


def run_ablation_study():
    """Main ablation study execution"""
    print("="*80)
    print("PROGRESSIVE CARDIOAI ATTENTION ABLATION STUDY")
    print("="*80)
    
    # Configuration from environment (defaults must match main_cardioAI.py argument parser)
    epochs = int(os.environ.get('CARDIOAI_ABLATION_EPOCHS', 25))
    batch_size = int(os.environ.get('CARDIOAI_ABLATION_BATCH_SIZE', 16))
    num_patients = int(os.environ.get('CARDIOAI_ABLATION_PATIENTS', 308))
    training_frames = int(os.environ.get('CARDIOAI_TRAINING_FRAMES', 32))
    output_dir = Path(os.environ.get('CARDIOAI_OUTPUT_DIR', '.'))
    
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patients: {num_patients}")
    print(f"  Training frames: {training_frames}")
    print(f"  Output directory: {output_dir}")
    print(f"  Strategy: Full model (pre-trained, evaluate only) + Variants (train for comparison)")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Use existing output directory (created by main_cardioAI.py)
    if not output_dir.exists():
        raise RuntimeError(f"Output directory {output_dir} should be created by main_cardioAI.py first")
    
    # Initialize CardioAI utilities to use existing structure
    cardio_utils = CardioAIUtils()
    cardio_utils.current_output_dir = output_dir
    
    # Use existing subdirs (no folder creation)
    cardio_utils.subdirs = {
        'figures': output_dir,  # Save figures directly in ablation directory
    }
    
    # Load dataset
    print("\nLoading dataset...")
    tensor_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT")
    excel_file = Path("All.xlsx")

    full_dataset = CardioAIDataset(
        tensor_dir=tensor_dir,
        excel_file=excel_file,
        max_frames=training_frames
    )

    norm_params = full_dataset.get_normalization_parameters()
    print(f"Normalization parameters from dataset: log_transform_indices = {norm_params['log_transform_indices']}")

    # Use patients 1-235 (indices 0-234) for ablation training + validation
    # Matches 5-fold CV split from train_cardioAI.py
    from sklearn.model_selection import KFold

    train_cv_indices = list(range(min(235, len(full_dataset))))  # Patients 1-235

    # Check if custom training indices are provided
    train_indices_str = os.environ.get('CARDIOAI_TRAIN_INDICES', None)

    if train_indices_str:
        # Parse custom training indices to match training script
        indices = parse_train_indices(train_indices_str)
        dataset = Subset(full_dataset, indices)
        print(f"Using custom training indices: {train_indices_str}")
        print(f"Total training patients for ablation: {len(indices)}")
    elif num_patients < len(train_cv_indices):
        # Use patients 1-N (indices 0 to num_patients-1) to match training script
        indices = list(range(num_patients))
        dataset = Subset(full_dataset, indices)
        print(f"Using patients 1-{num_patients} for ablation (sequential, matches training)")
    else:
        dataset = Subset(full_dataset, train_cv_indices)
        print(f"Using patients 1-235 ({len(train_cv_indices)} patients) for ablation")

    # Split using 5-fold CV (fold 0 as default validation) matching train_cardioAI.py
    num_folds = int(os.environ.get('CARDIOAI_NUM_FOLDS', 5))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_splits = list(kf.split(train_cv_indices))
    train_idx, val_idx = fold_splits[0]  # Use fold 0 for ablation consistency

    fold_train_indices = [train_cv_indices[i] for i in train_idx]
    fold_val_indices = [train_cv_indices[i] for i in val_idx]

    train_dataset = Subset(full_dataset, fold_train_indices)
    val_dataset = Subset(full_dataset, fold_val_indices)

    # Calculate PH distribution in validation set
    import pandas as pd
    df = pd.read_excel(excel_file)
    meanPAP_values = df['meanPAP'].values
    ph_threshold = 20.0
    ph_labels = (meanPAP_values >= ph_threshold).astype(int)
    val_ph_labels = ph_labels[fold_val_indices]
    val_ph_positive = np.sum(val_ph_labels == 1)
    val_ph_negative = np.sum(val_ph_labels == 0)

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} validation (fold 1/{num_folds})")
    print(f"  Validation: {len(fold_val_indices)} patients from 5-fold CV (fold 1)")
    print(f"  Validation PH distribution: {val_ph_positive} PH+ / {val_ph_negative} PH- (matches training fold 1)")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load base model (should be pre-trained)
    print("\nLoading base progressive model...")
    base_model = create_model(
        num_outputs=9,
        num_frames=training_frames,
        num_views=4,
        dropout_rate=0.15
    )

    # Set normalization parameters BEFORE loading checkpoint
    # This ensures model's tensor shapes match the checkpoint
    base_model.set_winsorized_normalization(norm_params)
    print("Normalization parameters set in model")

    # Always load pre-trained weights from previous training
    model_path = find_latest_trained_model()

    # Load the pre-trained model weights
    try:
        if model_path.suffix == '.pth' and 'checkpoint' not in model_path.name:
            # Direct model state dict
            state_dict = torch.load(model_path, map_location=device)
            missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
            if missing_keys or unexpected_keys:
                print(f"Model loading details:")
                print(f"  Missing keys: {len(missing_keys)}")
                if missing_keys:
                    print(f"    {missing_keys[:5]}")  # Show first 5
                print(f"  Unexpected keys: {len(unexpected_keys)}")
                if unexpected_keys:
                    print(f"    {unexpected_keys[:10]}")  # Show first 10
                print(f"  Note: This is normal if model architecture was updated after training")
            print(f"Loaded pre-trained weights from: {model_path}")
        else:
            # Checkpoint format
            checkpoint = torch.load(model_path, map_location=device)
            missing_keys, unexpected_keys = base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys or unexpected_keys:
                print(f"Model loading details:")
                print(f"  Missing keys: {len(missing_keys)}")
                if missing_keys:
                    print(f"    {missing_keys[:5]}")
                print(f"  Unexpected keys: {len(unexpected_keys)}")
                if unexpected_keys:
                    print(f"    {unexpected_keys[:10]}")
                print(f"  Note: This is normal if model architecture was updated after training")
            print(f"Loaded pre-trained weights from: {model_path}")
    except Exception as e:
        print(f"ERROR: Failed to load pre-trained weights from {model_path}: {e}")
        print("Cannot proceed with ablation study - pre-trained model required for full_model baseline.")
        raise
    
    # Set model to evaluation mode for creating variants
    base_model.eval()

    # Create ablation variants with fresh models (ImageNet-only initialization)
    print("\nCreating ablation variants...")
    print("Creating fresh models for variants (ImageNet-only ViT, no 20-epoch training)...")
    variants = create_ablation_variants(base_model, norm_params)
    
    # Run ablation study
    results = {}
    
    for variant_name, model in variants.items():
        print(f"\n{'='*60}")
        print(f"TRAINING VARIANT: {variant_name.upper()}")
        print(f"{'='*60}")
        
        if variant_name == 'full_model':
            # For full model, use results from previous training (already trained for 20 epochs)
            print("Using pre-trained full model from previous training - evaluating only (no additional training)")
            print(f"  Loaded from: {model_path}")
            trainer = ProgressiveAblationTrainer(model, train_loader, val_loader, device, epochs=1)
            _, val_corr = trainer.validate_epoch()
            history = {
                'train_loss': [0.0],  # Using pre-trained model, no training in ablation
                'val_loss': [0.0],    # Using pre-trained model, no training in ablation
                'val_correlations': [np.mean(val_corr)]
            }
            final_correlations = val_corr
        else:
            # Train ablation variant (only variants need training)
            print(f"Training {variant_name} variant for comparison...")
            trainer = ProgressiveAblationTrainer(model, train_loader, val_loader, device, epochs)
            history = trainer.train()
            _, final_correlations = trainer.validate_epoch()
        
        results[variant_name] = {
            'history': history,
            'final_correlations': final_correlations,
            'avg_correlation': np.mean(final_correlations)
        }
        
        print(f"Final correlation: {np.mean(final_correlations):.3f}")
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING ABLATION RESULTS")
    print(f"{'='*60}")
    
    # Create results summary
    summary = {
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'num_patients': num_patients,
            'training_frames': training_frames,
            'device': str(device)
        },
        'results': {}
    }
    
    # Parameter names in desired display order with proper acronyms
    # Original order in model output: RAP, SPAP, dpap, meanPAP, PCWP, CO, CI, SVRI, PVR (indices 0-8)
    # Desired display order: mPAP, RAP, SPAP, DPAP, PCWP, CO, CI, SVRI, PVR
    param_names_original = ['RAP', 'SPAP', 'DPAP', 'mPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
    param_names = ['mPAP', 'RAP', 'SPAP', 'DPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
    # Mapping from display order to original model output order
    # mPAP=3, RAP=0, SPAP=1, DPAP=2, PCWP=4, CO=5, CI=6, SVRI=7, PVR=8
    param_order = [3, 0, 1, 2, 4, 5, 6, 7, 8]
    
    for variant_name, result in results.items():
        # Reorder correlations according to param_order for display
        reordered_correlations = [result['final_correlations'][idx] for idx in param_order]
        summary['results'][variant_name] = {
            'avg_correlation': result['avg_correlation'],
            'per_param_correlations': {
                param_names[i]: reordered_correlations[i]
                for i in range(len(param_names))
            },
            'training_history': result['history']
        }
        # Store reordered correlations for plotting
        result['reordered_correlations'] = reordered_correlations
        
        print(f"{variant_name:15s}: {result['avg_correlation']:.3f}")
    
    # Save JSON results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create comparison plots
    create_ablation_plots(results, output_dir, param_names, cardio_utils)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY EFFICIENCY SUMMARY")
    print(f"{'='*60}")

    # Calculate efficiency metrics
    num_variants = len(results) - 1  # Exclude full_model
    full_model_time = 0  # No training time for full model
    variant_training_time = epochs * num_variants  # Approximate training time

    print(f"Full model: Uses previous training (20 epochs already trained)")
    print(f"  Loaded from: {model_path}")
    print(f"Variants trained: {num_variants} models x {epochs} epochs = {variant_training_time} total epochs")
    print(f"Time saved: ~{epochs} epochs ({epochs/(epochs+variant_training_time)*100:.0f}% reduction from reusing pre-trained full model)")
    print(f"Results saved to: {output_dir}")
    
    return summary


def create_ablation_plots(results, output_dir, param_names, cardio_utils):
    """Create visualization plots for ablation study with larger fonts"""

    # Setup color manager
    color_manager = ColorManager()

    # Set larger font sizes globally for all plots
    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12
    ANNOT_FONTSIZE = 11
    LEGEND_FONTSIZE = 12

    # Parameter order mapping (original model output to display order)
    # mPAP=3, RAP=0, SPAP=1, DPAP=2, PCWP=4, CO=5, CI=6, SVRI=7, PVR=8
    param_order = [3, 0, 1, 2, 4, 5, 6, 7, 8]

    # 1. Correlation comparison bar plot
    plt.figure(figsize=(12, 8))

    variant_names = list(results.keys())
    avg_correlations = [results[name]['avg_correlation'] for name in variant_names]

    bars = plt.bar(range(len(variant_names)), avg_correlations,
                   color=[color_manager.get_color(i) for i in range(len(variant_names))])
    plt.xlabel('Ablation Variant', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Average Correlation', fontsize=LABEL_FONTSIZE)
    plt.title('Attention Ablation Study - Average Correlations', fontsize=TITLE_FONTSIZE)
    plt.xticks(range(len(variant_names)), variant_names, rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(False)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=ANNOT_FONTSIZE)

    plt.tight_layout()
    cardio_utils.save_figure(plt.gcf(), 'ablation_comparison')
    plt.close()

    # 2. Per-parameter heatmap with reordered parameters
    correlation_matrix = np.zeros((len(variant_names), len(param_names)))
    for i, variant_name in enumerate(variant_names):
        # Use reordered correlations if available, otherwise reorder on the fly
        if 'reordered_correlations' in results[variant_name]:
            correlation_matrix[i, :] = results[variant_name]['reordered_correlations']
        else:
            original_corr = results[variant_name]['final_correlations']
            correlation_matrix[i, :] = [original_corr[idx] for idx in param_order]

    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_matrix,
                xticklabels=param_names,
                yticklabels=variant_names,
                annot=True, fmt='.3f', cmap=color_manager.get_heatmap_colormap(),
                cbar_kws={'label': 'Correlation Coefficient'},
                annot_kws={'fontsize': ANNOT_FONTSIZE})
    plt.title('Attention Ablation Study - Per-Parameter Correlations', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Hemodynamic Parameters', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Ablation Variants', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    # Adjust colorbar font size
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label('Correlation Coefficient', fontsize=LABEL_FONTSIZE)
    plt.tight_layout()
    cardio_utils.save_figure(plt.gcf(), 'ablation_heatmap')
    plt.close()

    # 3. Improvement over baseline (no attention) with reordered parameters
    if 'no_attention' in results and 'full_model' in results:
        # Get reordered correlations
        if 'reordered_correlations' in results['no_attention']:
            baseline_corr = np.array(results['no_attention']['reordered_correlations'])
        else:
            original_baseline = results['no_attention']['final_correlations']
            baseline_corr = np.array([original_baseline[idx] for idx in param_order])

        if 'reordered_correlations' in results['full_model']:
            full_model_corr = np.array(results['full_model']['reordered_correlations'])
        else:
            original_full = results['full_model']['final_correlations']
            full_model_corr = np.array([original_full[idx] for idx in param_order])

        improvements = full_model_corr - baseline_corr

        plt.figure(figsize=(12, 6))
        bars = plt.bar(param_names, improvements,
                      color=[color_manager.get_color(i) for i in range(len(param_names))])
        plt.xlabel('Hemodynamic Parameters', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Correlation Improvement', fontsize=LABEL_FONTSIZE)
        plt.title('Full Model vs No Attention - Correlation Improvements', fontsize=TITLE_FONTSIZE)
        plt.xticks(rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.grid(False)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=ANNOT_FONTSIZE)

        plt.tight_layout()
        cardio_utils.save_figure(plt.gcf(), 'attention_improvements')
        plt.close()

    # 4. Component contribution analysis
    if all(variant in results for variant in ['spatial_only', 'temporal_only', 'fusion_only']):
        component_data = {
            'Spatial Only': results['spatial_only']['avg_correlation'],
            'Temporal Only': results['temporal_only']['avg_correlation'],
            'Fusion Only': results['fusion_only']['avg_correlation'],
            'Full Model': results['full_model']['avg_correlation']
        }

        plt.figure(figsize=(10, 6))
        bars = plt.bar(component_data.keys(), component_data.values(),
                      color=[color_manager.get_color(i) for i in range(len(component_data))])
        plt.ylabel('Average Correlation', fontsize=LABEL_FONTSIZE)
        plt.title('Individual Attention Component Contributions', fontsize=TITLE_FONTSIZE)
        plt.xticks(rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.grid(False)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=ANNOT_FONTSIZE)

        plt.tight_layout()
        cardio_utils.save_figure(plt.gcf(), 'component_contributions')
        plt.close()

    # 5. Training loss comparison (if available)
    loss_variants = {}
    for variant_name, result in results.items():
        if 'history' in result and 'train_loss' in result['history']:
            loss_variants[variant_name] = result['history']['train_loss']

    if loss_variants:
        plt.figure(figsize=(12, 8))
        for variant_name, losses in loss_variants.items():
            epochs_range = range(1, len(losses) + 1)
            plt.plot(epochs_range, losses, label=variant_name, linewidth=2)

        plt.xlabel('Epoch', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Training Loss', fontsize=LABEL_FONTSIZE)
        plt.title('Training Loss Comparison - Ablation Variants', fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.grid(False)
        plt.tight_layout()
        cardio_utils.save_figure(plt.gcf(), 'ablation_training_loss')
        plt.close()

    print(f"Comprehensive ablation plots saved to: {output_dir}")
    print(f"Generated: comparison, heatmap, improvements, component contributions, training loss")


if __name__ == "__main__":
    run_ablation_study()