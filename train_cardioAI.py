"""
CardioAI training with progressive ViT unfreezing and 5-fold cross-validation.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import KFold

# Import local modules
from dataset_cardioAI import CardioAIDataset
from model_cardioAI import create_model, create_progressive_optimizer
from utils_cardioAI import setup_cardio_output, save_cardio_figure


def parse_train_indices(indices_str: str) -> List[int]:
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


def create_balanced_ph_splits(excel_file: Path, 
                             test_size: int = 55,
                             threshold: float = 20.0,
                             random_state: int = 42,
                             strategy: str = 'undersample_majority') -> Tuple[List[int], List[int], Dict]:
    """
    Create clinically-representative train/test splits for pulmonary hypertension classification
    
    Args:
        excel_file: Path to Excel file with patient data
        test_size: Number of patients for test set
        threshold: meanPAP threshold for PH classification (default: 20.0 mmHg)
        random_state: Random seed for reproducibility
        strategy: Balancing strategy ('undersample_majority', 'oversample_minority', 'stratified')
    
    Returns:
        train_indices: Indices for training set
        test_indices: Indices for test set (75/25 positive/negative reflecting clinical distribution)
        info: Dictionary with split information
    
    Note:
        Test set uses 75/25 positive/negative split to reflect real-world PH prevalence
        rather than artificial 50/50 balance, providing more clinically relevant evaluation
    """
    
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    # Load data
    df = pd.read_excel(excel_file)
    patient_ids = df['E Number'].values
    meanPAP_values = df['meanPAP'].values
    
    # Create PH labels (1=positive, 0=negative)
    ph_labels = (meanPAP_values >= threshold).astype(int)
    
    # Get positive and negative indices
    positive_indices = np.where(ph_labels == 1)[0]
    negative_indices = np.where(ph_labels == 0)[0]
    
    print(f"Pulmonary Hypertension distribution:")
    print(f"  Total patients: {len(df)}")
    print(f"  PH Positive (>={threshold}): {len(positive_indices)} ({len(positive_indices)/len(df)*100:.1f}%)")
    print(f"  PH Negative (<{threshold}): {len(negative_indices)} ({len(negative_indices)/len(df)*100:.1f}%)")
    
    # Calculate test set composition reflecting clinical distribution (75/25 split)
    # This better represents real-world PH prevalence than artificial 50/50 balance
    test_positive = min(int(test_size * 0.75), len(positive_indices))  # ~75% positive
    test_negative = min(test_size - test_positive, len(negative_indices))  # ~25% negative
    
    # If we don't have enough negatives for 25%, adjust proportionally
    if test_negative < int(test_size * 0.25):
        remaining_slots = test_size - test_negative
        test_positive = min(remaining_slots, len(positive_indices))
    
    print(f"Test set composition ({test_positive + test_negative} patients):")
    print(f"  PH Positive: {test_positive}")
    print(f"  PH Negative: {test_negative}")
    
    # Randomly select test indices
    np.random.seed(random_state)
    test_pos_indices = np.random.choice(positive_indices, test_positive, replace=False)
    test_neg_indices = np.random.choice(negative_indices, test_negative, replace=False)
    test_indices = np.concatenate([test_pos_indices, test_neg_indices])
    
    # Remaining indices for training
    train_pos_indices = np.setdiff1d(positive_indices, test_pos_indices)
    train_neg_indices = np.setdiff1d(negative_indices, test_neg_indices)
    
    # Apply balancing strategy for training set
    if strategy == 'undersample_majority':
        # Undersample positive class to match negative class
        min_class_size = min(len(train_pos_indices), len(train_neg_indices))
        train_pos_balanced = np.random.choice(train_pos_indices, min_class_size, replace=False)
        train_neg_balanced = np.random.choice(train_neg_indices, min_class_size, replace=False)
        train_indices = np.concatenate([train_pos_balanced, train_neg_balanced])
        
    elif strategy == 'oversample_minority':
        # Oversample negative class to match positive class
        max_class_size = max(len(train_pos_indices), len(train_neg_indices))
        train_pos_balanced = np.random.choice(train_pos_indices, max_class_size, replace=True)
        train_neg_balanced = np.random.choice(train_neg_indices, max_class_size, replace=True)
        train_indices = np.concatenate([train_pos_balanced, train_neg_balanced])
        
    elif strategy == 'stratified':
        # Use all training data with stratified sampling awareness
        train_indices = np.concatenate([train_pos_indices, train_neg_indices])
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Create info dictionary
    info = {
        'strategy': strategy,
        'threshold': threshold,
        'random_state': random_state,
        'total_patients': len(df),
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'train_positive': np.sum(ph_labels[train_indices] == 1),
        'train_negative': np.sum(ph_labels[train_indices] == 0),
        'test_positive': np.sum(ph_labels[test_indices] == 1),
        'test_negative': np.sum(ph_labels[test_indices] == 0),
        'original_positive': len(positive_indices),
        'original_negative': len(negative_indices)
    }
    
    # Print training set info
    print(f"Training set composition ({len(train_indices)} patients):")
    print(f"  PH Positive: {info['train_positive']} ({info['train_positive']/len(train_indices)*100:.1f}%)")
    print(f"  PH Negative: {info['train_negative']} ({info['train_negative']/len(train_indices)*100:.1f}%)")
    print(f"  Balance ratio: {info['train_positive']/info['train_negative']:.2f}")
    
    return train_indices.tolist(), test_indices.tolist(), info


class ProgressiveMSELoss(nn.Module):
    """Simple MSE loss for normalized predictions and targets"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """Compute MSE loss between normalized predictions and targets"""
        # Both predictions and targets should be in normalized space
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("Warning: NaN/Inf in predictions")
            return torch.tensor(float('inf'), device=predictions.device)
            
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN/Inf in targets")
            return torch.tensor(float('inf'), device=predictions.device)
        
        loss = self.mse_loss(predictions, targets)
        return loss


class ProgressiveTrainer:
    """Progressive training with multi-stage unfreezing"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 stage_epochs=50, task_lr=3e-4, vit_lr=3e-5, stages=0, total_epochs=100, test_loader=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.stage_epochs = stage_epochs
        self.task_lr = task_lr
        self.vit_lr = vit_lr
        self.stages = stages  # Number of ViT layers to unfreeze (0-12)
        self.total_epochs = total_epochs  # Total number of training epochs for scheduler

        # Loss function
        self.loss_fn = ProgressiveMSELoss(device)

        # Progressive training state
        self.current_stage = 0
        self.total_stages = max(1, self.stages + 1) if self.stages > 0 else 1  # Adjust based on stages
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_correlations = None
        self.best_test_correlations = None

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_correlations': [],
            'per_task_val_corr': [],
            'test_loss': [],
            'test_correlations': [],
            'per_task_test_corr': [],
            'stage_info': [],
            'learning_rates': []
        }

        # Initialize optimizer for stage 0 (task layers only)
        self.optimizer = self._create_stage_optimizer()

        self.scheduler = None

        # Output directory
        self.output_dir = Path(os.environ.get('CARDIOAI_OUTPUT_DIR', '.'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_stage_optimizer(self):
        """Create optimizer for current stage"""
        return create_progressive_optimizer(
            self.model,
            task_lr=self.task_lr,
            vit_lr=self.vit_lr
        )

    def _create_scheduler(self):
        return None

    def _advance_stage(self):
        """Advance to next progressive training stage - unfreezing from last blocks"""
        self.current_stage += 1

        # Check if we should unfreeze ViT blocks based on stages parameter
        # Unfreeze from the end: block 11, 10, 9, ... (reverse order)
        if self.stages > 0 and self.current_stage <= self.stages:  # Only unfreeze if stages > 0
            # Calculate block index from the end (12 total blocks: 0-11)
            # Stage 1 -> block 11, Stage 2 -> block 10, ..., Stage 12 -> block 0
            block_idx = 12 - self.current_stage
            print(f"\n{'='*60}")
            print(f"ADVANCING TO STAGE {self.current_stage}")
            print(f"Unfreezing ViT transformer block {block_idx} (from last, max {self.stages} blocks)")
            print(f"{'='*60}")

            # Unfreeze the ViT block (from the end)
            self.model.unfreeze_vit_block(block_idx)

            # Create new optimizer with updated parameter groups
            self.optimizer = self._create_stage_optimizer()


            # Log stage info
            stage_info = {
                'stage': self.current_stage,
                'unfrozen_block': block_idx,
                'epoch': self.current_epoch,
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
            self.training_history['stage_info'].append(stage_info)
        elif self.stages == 0:
            # All ViT blocks stay frozen
            print(f"\n{'='*60}")
            print(f"STAGE {self.current_stage}: ViT BLOCKS REMAIN FROZEN (stages=0)")
            print(f"{'='*60}")
            stage_info = {
                'stage': self.current_stage,
                'unfrozen_block': None,
                'epoch': self.current_epoch,
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
            self.training_history['stage_info'].append(stage_info)
            
            print(f"Stage {self.current_stage} setup complete:")
            print(f"  Unfrozen block: {stage_info['unfrozen_block']}")
            print(f"  Trainable parameters: {stage_info['trainable_params']:,}")
            
        else:
            if self.stages > 0:
                print(f"All requested ViT blocks unfrozen ({self.stages}/{self.stages}). Continuing with current model configuration.")
            else:
                print(f"ViT blocks remain frozen. Continuing with frozen backbone training.")
    
    def normalize_targets(self, targets):
        """Normalize targets using model's normalization"""
        return self.model.normalize_targets(targets)
    
    def compute_correlations(self, predictions, targets):
        """Per-parameter Pearson correlation between denormalized predictions and targets."""
        correlations = []
        pred_denorm = self.model.denormalize_predictions(predictions)
        targ_denorm = targets
        
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
        all_predictions = []
        all_targets = []
        
        for batch_idx, (views, targets, patient_ids) in enumerate(self.train_loader):
            # Move to device
            views = [view.to(self.device) for view in views]
            targets = targets.to(self.device)
            
            # Normalize targets
            targets_norm = self.normalize_targets(targets)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                predictions = self.model(views, return_aux=False)
                
                # Check for NaN/Inf
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    print(f"Warning: NaN/Inf in predictions at batch {batch_idx}")
                    continue
                
                # Compute loss
                loss = self.loss_fn(predictions, targets_norm)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                # Optimizer step
                self.optimizer.step()
                
                # Track metrics
                epoch_losses.append(loss.item())
                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())  # Store original (denormalized) targets for correlation
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Compute epoch metrics
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            if all_predictions:
                all_pred = torch.cat(all_predictions, dim=0)
                all_targ = torch.cat(all_targets, dim=0)
                train_correlations = self.compute_correlations(all_pred, all_targ)
            else:
                train_correlations = [0.0] * 9
        else:
            avg_loss = float('inf')
            train_correlations = [0.0] * 9
        
        return avg_loss, train_correlations
    
    def validate_on_loader(self, loader, loader_name="Validation"):
        """Validate on a specific dataloader"""
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (views, targets, patient_ids) in enumerate(loader):
                # Move to device
                views = [view.to(self.device) for view in views]
                targets = targets.to(self.device)

                # Normalize targets
                targets_norm = self.normalize_targets(targets)

                # Forward pass
                try:
                    predictions = self.model(views, return_aux=False)

                    # Compute loss
                    loss = self.loss_fn(predictions, targets_norm)

                    val_losses.append(loss.item())
                    all_predictions.append(predictions)
                    all_targets.append(targets)  # Store original (denormalized) targets for correlation

                except Exception as e:
                    print(f"{loader_name} error in batch {batch_idx}: {e}")
                    continue

        # Compute validation metrics
        if val_losses:
            avg_val_loss = np.mean(val_losses)
            if all_predictions:
                all_pred = torch.cat(all_predictions, dim=0)
                all_targ = torch.cat(all_targets, dim=0)
                val_correlations = self.compute_correlations(all_pred, all_targ)
            else:
                val_correlations = [0.0] * 9
        else:
            avg_val_loss = float('inf')
            val_correlations = [0.0] * 9

        return avg_val_loss, val_correlations

    def validate_epoch(self):
        """Validate for one epoch on internal validation set"""
        return self.validate_on_loader(self.val_loader, "Internal Validation")

    def validate_test(self):
        """Validate on test set (patients 236-308)"""
        if self.test_loader is None:
            return None, None
        return self.validate_on_loader(self.test_loader, "Test")
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None,
            'best_val_loss': self.best_val_loss,
            'best_correlations': self.best_correlations,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint with error handling
        try:
            torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        except Exception as e:
            print(f"Warning: Could not save latest checkpoint: {e}")
        
        # Save best checkpoint
        if is_best:
            try:
                torch.save(checkpoint, self.output_dir / 'best_checkpoint.pth')
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
                print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
            except Exception as e:
                print(f"Warning: Could not save best checkpoint: {e}")
    
    def check_correlation_threshold(self, correlations, threshold=0.6):
        """Check if all correlations exceed threshold"""
        return all(corr >= threshold for corr in correlations)
    
    def train(self, total_epochs):
        """Main progressive training loop"""
        print(f"\n{'='*80}")
        print("STARTING PROGRESSIVE TRAINING")
        print(f"{'='*80}")
        print(f"Epochs: {total_epochs}, Stage epochs: {self.stage_epochs}, Stages: {self.total_stages}")
        print(f"LR: task={self.task_lr}, vit={self.vit_lr}")
        print(f"{'='*80}")
        
        stage_start_epoch = 0
        
        for epoch in range(total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Check if we need to advance to next stage
            if epoch > 0 and (epoch - stage_start_epoch) >= self.stage_epochs:
                if self.current_stage < self.total_stages - 1:
                    self._advance_stage()
                    stage_start_epoch = epoch
            
            # Train and validate
            train_loss, train_corr = self.train_epoch()
            val_loss, val_corr = self.validate_epoch()

            # Also validate on test set if available
            if self.test_loader:
                test_loss, test_corr = self.validate_test()
            else:
                test_loss, test_corr = None, None

            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_correlations'].append(np.mean(val_corr))
            self.training_history['per_task_val_corr'].append(val_corr)

            if test_loss is not None:
                self.training_history['test_loss'].append(test_loss)
                self.training_history['test_correlations'].append(np.mean(test_corr))
                self.training_history['per_task_test_corr'].append(test_corr)

            # Track learning rates (constant discriminative: task=1e-4, vit=3e-5)
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.training_history['learning_rates'].append(lrs)


            # Check if best model (based on internal validation)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_correlations = val_corr.copy()
                if test_corr is not None:
                    self.best_test_correlations = test_corr.copy()

            # Save checkpoint
            self.save_checkpoint(is_best)

            # Print progress
            epoch_time = time.time() - epoch_start_time
            avg_corr = np.mean(val_corr)
            above_threshold = sum(1 for c in val_corr if c >= 0.6)

            print(f"Epoch {epoch+1:3d}/{total_epochs} | Stage {self.current_stage:2d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Int Val Loss: {val_loss:.4f} | "
                  f"Int Val Corr: {avg_corr:.3f} ({above_threshold}/9 >=0.6)")

            # Print test results if available
            if test_corr is not None:
                test_avg_corr = np.mean(test_corr)
                test_above_threshold = sum(1 for c in test_corr if c >= 0.6)
                print(f"{'':>18} Test Loss: {test_loss:.4f} | "
                      f"Test Corr: {test_avg_corr:.3f} ({test_above_threshold}/9 >=0.6) | "
                      f"Time: {epoch_time:.1f}s")
            else:
                print(f"{'':>18} Time: {epoch_time:.1f}s")

            if self.check_correlation_threshold(val_corr, 0.6):
                print(f"    All val correlations above 0.6 at epoch {epoch+1}")
        
        # Save final results
        self.save_training_results()
        print(f"\n{'='*60}")
        print("PROGRESSIVE TRAINING COMPLETED")
        print(f"{'='*60}")

        # Print internal validation results
        print(f"\nCross-Validation Fold Results:")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        param_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
        if self.best_correlations:
            for param, corr in zip(param_names, self.best_correlations):
                status = "[PASS]" if corr >= 0.6 else "[FAIL]"
                print(f"  {param:8s}: {corr:.3f} {status}")

        # Print test results if available
        if self.best_test_correlations:
            print(f"\nTest Results (Patients 236-308):")
            for param, corr in zip(param_names, self.best_test_correlations):
                status = "[PASS]" if corr >= 0.6 else "[FAIL]"
                print(f"  {param:8s}: {corr:.3f} {status}")

    
    def save_training_results(self):
        """Save comprehensive training results"""
        results = {
            'training_config': {
                'stage_epochs': self.stage_epochs,
                'task_lr': self.task_lr,
                'vit_lr': self.vit_lr,
                'total_stages': self.total_stages,
                'total_epochs': self.current_epoch + 1
            },
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_correlations': self.best_correlations,
                'best_test_correlations': self.best_test_correlations,
                'final_stage': self.current_stage
            },
            'training_history': self.training_history
        }
        
        # Save results
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        print(f"Training results saved to: {self.output_dir}")

        # Generate ONLY training-specific figures (loss curves, correlations, stages, LR schedule)
        # Post-training analysis (scatter, Bland-Altman, ROC, etc.) is handled by figures_tables_cardioAI.py
        self.generate_training_figures()
    
    def generate_training_figures(self):
        """Generate training loss curves and correlation plots"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from utils_cardioAI import CardioAIUtils
        
        print("Generating training figures...")
        
        # Initialize utils for figure generation
        cardio_utils = CardioAIUtils()
        cardio_utils.current_output_dir = self.output_dir
        # Set up subdirs for save_figure function
        cardio_utils.subdirs = {'training': self.output_dir}
        
        # Set up color scheme
        colors = cardio_utils.get_color_palette(5)
        
        # 1. Create individual loss and correlation plots
        epochs_range = range(1, len(self.training_history['train_loss']) + 1)
        
        # First individual plot: Training and validation loss
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        ax1.plot(epochs_range, self.training_history['train_loss'], 
                color=colors[0], label='Training Loss', linewidth=2)
        ax1.plot(epochs_range, self.training_history['val_loss'], 
                color=colors[1], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Training and Validation Loss Curves')
        ax1.legend()
        ax1.grid(False)  # No grids as per requirements
        
        plt.tight_layout()
        cardio_utils.save_figure(fig1, self.output_dir / 'training_loss_curves')
        plt.close()
        
        # Second individual plot: Validation correlations
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        ax2.plot(epochs_range, self.training_history['val_correlations'], 
                color=colors[2], label='Mean Validation Correlation', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.set_title('Validation Correlation Progress')
        ax2.legend()
        ax2.grid(False)  # No grids as per requirements
        
        plt.tight_layout()
        cardio_utils.save_figure(fig2, self.output_dir / 'mean_correlation_progress')
        plt.close()
        
        # 2. Per-parameter correlation evolution
        if self.training_history['per_task_val_corr']:
            param_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each parameter's correlation over time
            per_task_corr = np.array(self.training_history['per_task_val_corr'])
            for i, param in enumerate(param_names):
                if i < per_task_corr.shape[1]:
                    ax.plot(epochs_range, per_task_corr[:, i], 
                           label=param, linewidth=2, 
                           color=colors[i % len(colors)])
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title('Per-Parameter Validation Correlation Evolution')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(False)
            # Use dark color from palette for target threshold line
            target_color = cardio_utils.get_dark_colors(1)[0]
            ax.axhline(y=0.6, color=target_color, linestyle='--', alpha=0.7, label='Target (0.6)')
            
            plt.tight_layout()
            cardio_utils.save_figure(fig, self.output_dir / 'per_parameter_correlations')
            plt.close()
        
        # 3. Progressive stage information
        if self.training_history['stage_info']:
            stage_data = []
            for stage_info in self.training_history['stage_info']:
                stage_data.append({
                    'Stage': stage_info['stage'],
                    'Start_Epoch': stage_info['epoch'],
                    'End_Epoch': stage_info.get('end_epoch', stage_info['epoch']),
                    'Unfrozen_Block': stage_info.get('unfrozen_block', 'N/A'),
                    'Trainable_Params': stage_info.get('trainable_params', 0)
                })
            
            # Create stage progression plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i, stage in enumerate(stage_data):
                start = stage['Start_Epoch']
                end = stage.get('End_Epoch', len(self.training_history['train_loss']))
                
                description = f"Block {stage['Unfrozen_Block']}" if stage['Unfrozen_Block'] != 'N/A' else "Task Layers"
                ax.axvspan(start, end, alpha=0.3, color=colors[i % len(colors)], 
                          label=f"Stage {stage['Stage']}: {description}")
            
            # Overlay loss curve
            ax.plot(epochs_range, self.training_history['train_loss'], 
                   color='black', linewidth=2, label='Training Loss')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Loss')
            ax.set_title('Progressive Training Stages')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(False)
            
            plt.tight_layout()
            cardio_utils.save_figure(fig, self.output_dir / 'progressive_training_stages')
            plt.close()
        
        # 4. Learning rate schedule
        if self.training_history['learning_rates']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            lr_data = self.training_history['learning_rates']
            epochs_lr = range(1, len(lr_data) + 1)
            
            # Plot different learning rate groups (lr_data contains lists of learning rates)
            if lr_data and len(lr_data[0]) > 0:
                # Transpose the data to get learning rates per group across epochs
                num_groups = len(lr_data[0])
                for group_idx in range(num_groups):
                    lr_values = [lr_list[group_idx] for lr_list in lr_data]
                    group_name = f"Group {group_idx + 1}" + (" (Task)" if group_idx == 0 else " (ViT)")
                    ax.plot(epochs_lr, lr_values, label=group_name, linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(False)
            
            plt.tight_layout()
            cardio_utils.save_figure(fig, self.output_dir / 'learning_rate_schedule')
            plt.close()

        print(f"Training figures saved to: {self.output_dir}")


def main():
    """Main training function"""
    
    # Configuration from environment variables (defaults must match main_cardioAI.py argument parser)
    epochs = int(os.environ.get('CARDIOAI_EPOCHS', 100))
    stage_epochs = int(os.environ.get('CARDIOAI_STAGE_EPOCHS', 50))
    batch_size = int(os.environ.get('CARDIOAI_BATCH_SIZE', 16))
    training_frames = int(os.environ.get('CARDIOAI_TRAINING_FRAMES', 32))
    stages = int(os.environ.get('CARDIOAI_STAGES', 1))
    ablation_attentions = os.environ.get('CARDIOAI_ABLATION_ATTENTIONS', 'temporal,fusion')
    
    print(f"Progressive training configuration:")
    print(f"  Total epochs: {epochs}")
    print(f"  Stage epochs: {stage_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training frames: {training_frames}")
    print(f"  ViT stages to unfreeze: {stages} (0=all frozen, 12=all unfrozen)")
    print(f"  Attention modules: {ablation_attentions} (use 'none' for direct ViT->regression)")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    print("Loading dataset...")
    tensor_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT")
    excel_file = Path("All.xlsx")
    
    # Create dataset
    full_dataset = CardioAIDataset(
        tensor_dir=tensor_dir,
        excel_file=excel_file,
        max_frames=training_frames
    )
    
    # Dataset split configuration
    # Patients 1-235 (indices 0-234): Training with 5-fold cross-validation
    # Patients 236-308 (indices 235-307): Independent test set
    total_size = len(full_dataset)
    print(f"Total dataset size: {total_size} patients")

    # Load meanPAP values to classify PH status
    print("\nAnalyzing pulmonary hypertension distribution...")
    df = pd.read_excel(excel_file)
    meanPAP_values = df['meanPAP'].values

    # Create PH labels (1=positive, 0=negative) based on meanPAP >= 20.0 threshold
    ph_threshold = 20.0
    ph_labels = (meanPAP_values >= ph_threshold).astype(int)

    # Get positive and negative indices
    positive_indices = np.where(ph_labels == 1)[0]
    negative_indices = np.where(ph_labels == 0)[0]

    print(f"PH distribution (threshold={ph_threshold}):")
    print(f"  Total: {len(ph_labels)} patients")
    print(f"  PH Positive (>={ph_threshold}): {len(positive_indices)} patients")
    print(f"  PH Negative (<{ph_threshold}): {len(negative_indices)} patients")

    # Patients 1-235 (indices 0-234) for training + cross-validation
    # Patients 236-308 (indices 235-307) for independent test
    train_cv_indices = list(range(235))  # 235 patients for training/CV
    test_indices = list(range(235, min(308, total_size)))  # 73 patients for independent test

    num_folds = int(os.environ.get('CARDIOAI_NUM_FOLDS', 5))

    print(f"\nDataset split configuration:")
    print(f"  Training + CV: patients 1-235 ({len(train_cv_indices)} patients)")
    print(f"  Independent test: patients 236-308 ({len(test_indices)} patients)")
    print(f"  Cross-validation: {num_folds}-fold")

    # PH distribution in training and test sets
    train_ph_labels = ph_labels[:235]
    train_pos = np.sum(train_ph_labels == 1)
    train_neg = np.sum(train_ph_labels == 0)
    print(f"  Training PH distribution: {train_pos} PH+ / {train_neg} PH-")

    test_ph_labels = ph_labels[235:308]
    test_pos = np.sum(test_ph_labels == 1)
    test_neg = np.sum(test_ph_labels == 0)
    print(f"  Test PH distribution: {test_pos} PH+ / {test_neg} PH-")

    # Create independent test dataset and loader
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Get normalization parameters from full dataset (consistent across folds)
    norm_params = full_dataset.get_normalization_parameters()

    # Output directory
    output_dir = Path(os.environ.get('CARDIOAI_OUTPUT_DIR', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5-Fold Cross-Validation on patients 1-235
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    param_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']

    best_fold = -1
    best_fold_val_loss = float('inf')
    all_fold_correlations = []
    all_fold_test_correlations = []

    print(f"\n{'='*80}")
    print(f"STARTING {num_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_cv_indices)):
        fold_train_indices = [train_cv_indices[i] for i in train_idx]
        fold_val_indices = [train_cv_indices[i] for i in val_idx]

        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{num_folds}")
        print(f"  Training: {len(fold_train_indices)} patients")
        print(f"  Validation: {len(fold_val_indices)} patients")
        print(f"{'='*80}")

        # Create fold-specific datasets and loaders
        fold_train_dataset = torch.utils.data.Subset(full_dataset, fold_train_indices)
        fold_val_dataset = torch.utils.data.Subset(full_dataset, fold_val_indices)

        fold_train_loader = DataLoader(
            fold_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        fold_val_loader = DataLoader(
            fold_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        print(f"  Train batches per epoch: {len(fold_train_loader)}")
        print(f"  Validation batches per epoch: {len(fold_val_loader)}")
        print(f"  Test batches per epoch: {len(test_loader)}")

        # Create fresh model for each fold
        print(f"  Creating progressive CardioAI model for fold {fold + 1}...")
        fold_model = create_model(
            num_outputs=9,
            num_frames=training_frames,
            num_views=4,
            dropout_rate=0.15,
            ablation_attentions=ablation_attentions
        )
        fold_model.set_winsorized_normalization(norm_params)

        # Set fold-specific output directory
        fold_output_dir = output_dir / f"fold_{fold + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        os.environ['CARDIOAI_OUTPUT_DIR'] = str(fold_output_dir)

        # Create trainer for this fold
        trainer = ProgressiveTrainer(
            model=fold_model,
            train_loader=fold_train_loader,
            val_loader=fold_val_loader,
            device=device,
            stage_epochs=stage_epochs,
            task_lr=1e-4,
            vit_lr=3e-5,
            stages=stages,
            total_epochs=epochs,
            test_loader=test_loader
        )

        print(f"\n  Optimization configuration:")
        print(f"    Task layer LR: 1e-4 (constant)")
        print(f"    ViT layer LR: 3e-5 (constant, 3.3x discriminative ratio)")
        print(f"    Gradient clipping: 5.0")

        # Train this fold
        trainer.train(epochs)

        # Save fold-specific model
        fold_model_path = output_dir / f"best_model_fold{fold + 1}.pth"
        torch.save(fold_model.state_dict(), fold_model_path)
        print(f"  Fold {fold + 1} model saved to: {fold_model_path}")

        # Track fold results
        fold_val_loss = trainer.best_val_loss
        fold_corr = trainer.best_correlations if trainer.best_correlations else [0.0] * 9
        fold_test_corr = trainer.best_test_correlations if trainer.best_test_correlations else [0.0] * 9

        all_fold_correlations.append(fold_corr)
        all_fold_test_correlations.append(fold_test_corr)

        print(f"\n  Fold {fold + 1} Results:")
        print(f"    Best validation loss: {fold_val_loss:.4f}")
        print(f"    Validation correlations:")
        for param, corr in zip(param_names, fold_corr):
            status = "[PASS]" if corr >= 0.6 else "[FAIL]"
            print(f"      {param:8s}: {corr:.3f} {status}")
        print(f"    Test correlations:")
        for param, corr in zip(param_names, fold_test_corr):
            status = "[PASS]" if corr >= 0.6 else "[FAIL]"
            print(f"      {param:8s}: {corr:.3f} {status}")

        # Track best fold
        if fold_val_loss < best_fold_val_loss:
            best_fold_val_loss = fold_val_loss
            best_fold = fold

        # Free GPU memory between folds
        del fold_model, trainer
        torch.cuda.empty_cache()

    # Restore output directory
    os.environ['CARDIOAI_OUTPUT_DIR'] = str(output_dir)

    # Copy best fold model as the final best_model.pth
    best_fold_model_path = output_dir / f"best_model_fold{best_fold + 1}.pth"
    final_model_path = output_dir / "best_model.pth"
    import shutil
    shutil.copy2(best_fold_model_path, final_model_path)

    # Report cross-validation summary
    print(f"\n{'='*80}")
    print(f"{num_folds}-FOLD CROSS-VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best fold: {best_fold + 1} (validation loss: {best_fold_val_loss:.4f})")
    print(f"Best model saved as: {final_model_path}")

    # Average correlations across folds
    avg_val_corr = np.mean(all_fold_correlations, axis=0)
    std_val_corr = np.std(all_fold_correlations, axis=0)
    avg_test_corr = np.mean(all_fold_test_correlations, axis=0)
    std_test_corr = np.std(all_fold_test_correlations, axis=0)

    print(f"\nCross-Validation Results (patients 1-235, {num_folds} folds):")
    print(f"  {'Parameter':<10} {'Mean Corr':>10} {'Std':>8} {'Status'}")
    print(f"  {'-'*40}")
    above_threshold = 0
    for param, avg_c, std_c in zip(param_names, avg_val_corr, std_val_corr):
        status = "[PASS]" if avg_c >= 0.6 else "[BELOW TARGET]"
        if avg_c >= 0.6:
            above_threshold += 1
        print(f"  {param:<10} {avg_c:>10.3f} {std_c:>8.3f} {status}")
    print(f"  Parameters above 0.6: {above_threshold}/9")

    print(f"\nIndependent Test Results (patients 236-308):")
    print(f"  {'Parameter':<10} {'Mean Corr':>10} {'Std':>8} {'Status'}")
    print(f"  {'-'*40}")
    for param, avg_c, std_c in zip(param_names, avg_test_corr, std_test_corr):
        status = "[PASS]" if avg_c >= 0.6 else "[BELOW TARGET]"
        print(f"  {param:<10} {avg_c:>10.3f} {std_c:>8.3f} {status}")

    # Save cross-validation summary
    cv_summary = {
        'num_folds': num_folds,
        'best_fold': best_fold + 1,
        'best_fold_val_loss': float(best_fold_val_loss),
        'per_fold_val_correlations': [list(map(float, fc)) for fc in all_fold_correlations],
        'per_fold_test_correlations': [list(map(float, fc)) for fc in all_fold_test_correlations],
        'avg_val_correlations': {p: float(c) for p, c in zip(param_names, avg_val_corr)},
        'std_val_correlations': {p: float(c) for p, c in zip(param_names, std_val_corr)},
        'avg_test_correlations': {p: float(c) for p, c in zip(param_names, avg_test_corr)},
        'std_test_correlations': {p: float(c) for p, c in zip(param_names, std_test_corr)},
    }

    cv_file = output_dir / "cv_summary.json"
    with open(cv_file, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\nCross-validation summary saved to: {cv_file}")

    # Save final correlations (from best fold) for pipeline compatibility
    best_fold_corr = all_fold_correlations[best_fold]
    correlations_dict = {p: float(c) for p, c in zip(param_names, best_fold_corr)}
    correlations_file = output_dir / "final_correlations.json"
    with open(correlations_file, 'w') as f:
        json.dump(correlations_dict, f, indent=2)
    print(f"Final correlations (best fold) saved to: {correlations_file}")

    print(f"\nProgressive training completed successfully!")
    print(f"Training outputs saved to: {output_dir}")
    print(f"Note: Post-training analysis (scatter, Bland-Altman, ROC, confusion) is generated by figures_tables_cardioAI.py")


if __name__ == "__main__":
    main()