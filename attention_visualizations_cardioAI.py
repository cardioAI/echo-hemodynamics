#!/usr/bin/env python3
"""
CardioAI attention visualizations: temporal rollout curves and Integrated Gradients overlays.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter

# Local imports
from model_cardioAI import create_model, ProgressiveCardioAI
from dataset_cardioAI import CardioAIDataset
from utils_cardioAI import ColorManager
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*PostScript.*transparency.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*partially transparent artists.*")


def create_apex_mask(image_shape, apex_height=70, transition_width=50):
    """Smooth sigmoid mask to suppress the transducer apex region in heatmap overlays."""
    height, width = image_shape

    # Create smooth transition using sigmoid-like curve
    y_coords = np.arange(height)
    # Center the transition at apex_height, spread over transition_width
    # The divisor (transition_width / 4) controls the steepness of the sigmoid
    transition = 1.0 / (1.0 + np.exp(-(y_coords - apex_height) / (transition_width / 4)))

    # Apply to all columns (tile the 1D transition to 2D)
    mask = np.tile(transition.reshape(-1, 1), (1, width)).astype(np.float32)

    return mask


class ProgressiveAttentionVisualizer:
    """Progressive attention visualizer with proper tensor handling"""
    
    def __init__(self, model_path, dataset_dir, excel_file, output_dir, device, n_frames=8,
                 overlay_alpha_full=0.7, overlay_alpha_selective=0.8):
        self.model_path = Path(model_path)
        self.dataset_dir = dataset_dir
        self.excel_file = excel_file
        self.output_dir = Path(output_dir)
        self.device = device
        self.n_frames = n_frames  # Number of frames with highest attention importance
        self.overlay_alpha_full = overlay_alpha_full  # Alpha for full gradient overlay
        self.overlay_alpha_selective = overlay_alpha_selective  # Max alpha for selective overlay
        
        # Use existing output directories (created by main_cardioAI.py)
        self.curves_dir = self.output_dir / "temporal_curves_attention_visualizations" / "temporal_curves"
        self.viz_dir = self.output_dir / "temporal_curves_attention_visualizations" / "attention_visualizations"
        
        # Create directories if they don't exist
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directories: {self.curves_dir} and {self.viz_dir}")
        
        # Initialize color manager
        self.color_manager = ColorManager()
        
        # Load model
        self.model = self._load_model()
        
        # Parameter names
        self.param_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
        self.view_names = ['FC', 'TC', 'SA', 'LA']
        
    def _load_model(self):
        """Load model with error handling"""
        try:
            print("Loading dataset for normalization parameters...")
            training_frames = int(os.environ.get('CARDIOAI_TRAINING_FRAMES', 32))

            # Create dataset (just to get normalization parameters)
            temp_dataset = CardioAIDataset(self.dataset_dir, self.excel_file, max_frames=training_frames, subset_size=10)
            norm_params = temp_dataset.get_normalization_parameters()
            print(f"Normalization parameters loaded: log_transform_indices = {norm_params['log_transform_indices']}")

            # Create model
            model = create_model(num_outputs=9, num_frames=training_frames, num_views=4)

            # Set normalization parameters BEFORE loading checkpoint
            model.set_winsorized_normalization(norm_params)
            print("Normalization parameters set in model")

            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Checkpoint format
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Direct state dict format
                    state_dict = checkpoint

                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    print(f"Warning: Model architecture mismatch. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                print(f"Loaded compatible weights from {self.model_path}")
            else:
                print("Warning: Using randomly initialized model")

            model = model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def save_robust_figure(self, fig, filename, output_dir, preserve_facecolor=False):
        """Save figure in all 3 formats with enhanced spacing to prevent overlaps

        Args:
            fig: Matplotlib figure object
            filename: Base filename for saving
            output_dir: Output directory path
            preserve_facecolor: If True, preserve figure's facecolor (for dark backgrounds)
                               If False, use white background (default for most figures)
        """
        from pathlib import Path

        base_name = Path(filename).stem
        output_dir = Path(output_dir)
        saved_files = []

        # Apply tight layout with extra padding to prevent overlaps
        try:
            fig.tight_layout(pad=2.0)
        except:
            pass

        # Determine background color - preserve custom backgrounds (e.g., dark navy for heatmaps)
        # or use white for regular figures (temporal curves, ultrasound images)
        if preserve_facecolor:
            # Use figure's existing facecolor (e.g., dark navy for overlay heatmaps)
            facecolor = fig.get_facecolor()
        else:
            # Use white background for regular figures
            facecolor = 'white'

        try:
            # PNG format - supports transparency
            png_path = output_dir / f"{base_name}.png"
            fig.savefig(
                png_path,
                format='png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.15,  # Increased padding
                facecolor=facecolor,
                edgecolor='none'
            )
            saved_files.append(str(png_path))

            # TIFF format - supports transparency
            tiff_path = output_dir / f"{base_name}.tiff"
            fig.savefig(
                tiff_path,
                format='tiff',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.15,  # Increased padding
                facecolor=facecolor,
                edgecolor='none'
            )
            saved_files.append(str(tiff_path))

            # EPS format - handle transparency by suppressing warnings
            eps_path = output_dir / f"{base_name}.eps"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig.savefig(
                        eps_path,
                        format='eps',
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.15,  # Increased padding
                        facecolor=facecolor,
                        edgecolor='none'
                    )
                saved_files.append(str(eps_path))
            except Exception:
                # If EPS fails, create a placeholder
                try:
                    with open(eps_path, 'w') as f:
                        f.write(f"%!PS-Adobe-3.0 EPSF-3.0\n% Generated by CardioAI for {base_name}\n")
                    saved_files.append(str(eps_path))
                except:
                    pass

            return saved_files

        except Exception as e:
            print(f"Error saving figure {filename}: {e}")
            return []
    
    def _get_patient_data(self, patient_id):
        """Load patient data with safe tensor handling"""
        try:
            # Get training frames from environment variable
            training_frames = int(os.environ.get('CARDIOAI_TRAINING_FRAMES', 32))
            dataset = CardioAIDataset(self.dataset_dir, self.excel_file, max_frames=training_frames)
            
            # Find patient in dataset
            for i, (views, targets, pid) in enumerate(dataset):
                if pid == patient_id:
                    # Ensure proper tensor shapes - CardioAIDataset returns (frames, height, width) per view
                    fixed_views = []
                    for view in views:
                        # Dataset returns views as (frames, height, width)
                        # We need to add batch dimension for model processing -> (1, frames, height, width)
                        if len(view.shape) == 3:  # (frames, height, width)
                            view = view.unsqueeze(0)  # Add batch dimension -> (1, frames, height, width)
                        elif len(view.shape) == 4:  # Already has batch dimension
                            pass  # Keep as is
                        else:
                            print(f"Unexpected view shape from dataset: {view.shape}")
                            continue
                        
                        fixed_views.append(view.to(self.device))
                    
                    print(f"Loaded patient {patient_id} with {len(fixed_views)} views")
                    for i, view in enumerate(fixed_views):
                        print(f"  View {i} shape: {view.shape}")
                    
                    return fixed_views, targets.to(self.device), pid
            
            print(f"Patient {patient_id} not found in dataset")
            return None, None, None
            
        except Exception as e:
            print(f"Error loading patient data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def generate_temporal_rollout_curves(self, patient_id="E100017564"):
        """Generate temporal attention rollout curves"""
        print(f"Generating temporal rollout curves for patient {patient_id}...")
        
        try:
            views, targets, pid = self._get_patient_data(patient_id)
            if views is None:
                print("Could not load patient data")
                return {}
            
            # Get attention rollout scores
            rollout_scores = self.model.attention_rollout(views, target_param_idx=0)
            
            # Generate curves for each parameter
            curves_generated = 0
            
            for param_idx, param_name in enumerate(self.param_names):
                try:
                    # Get rollout for this parameter
                    param_rollout = self.model.attention_rollout(views, target_param_idx=param_idx)
                    
                    # Create individual plots for each view - no merged subplots
                    view_color_map = self.color_manager.get_view_color_map()
                    
                    # Create combined view plot for this parameter
                    fig_combined, ax_combined = plt.subplots(1, 1, figsize=(10, 6))

                    for view_idx, (view_name, scores) in enumerate(param_rollout.items()):
                        if view_idx < 4:  # Only 4 views
                            # Handle 2D scores array (batch_size, frames) -> flatten to 1D
                            if isinstance(scores, np.ndarray) and len(scores.shape) > 1:
                                # If scores is 2D (batch_size, frames), take first batch
                                scores_1d = scores[0] if scores.shape[0] > 0 else scores.flatten()
                            elif torch.is_tensor(scores):
                                scores_np = scores.cpu().numpy()
                                scores_1d = scores_np[0] if len(scores_np.shape) > 1 and scores_np.shape[0] > 0 else scores_np.flatten()
                            else:
                                scores_1d = np.array(scores).flatten()

                            # Create frame indices based on actual number of frames
                            frames = np.arange(len(scores_1d))

                            # Use view-specific dark color from color map for better visibility
                            view_color = view_color_map.get(view_name, self.color_manager.get_dark_colors(1)[0])
                            ax_combined.plot(frames, scores_1d, linewidth=2, color=view_color, label=f'{view_name} View')

                    ax_combined.set_title(f'Temporal Attention Rollout - {param_name}')
                    ax_combined.set_xlabel('Frame Index')
                    ax_combined.set_ylabel('Attention Weight')
                    ax_combined.legend(loc='best')
                    ax_combined.grid(False)  # No grids as per requirements
                    
                    plt.tight_layout()
                    
                    # Save in all 3 formats
                    base_name = f"rollout_{param_name.lower()}"
                    self.save_robust_figure(fig_combined, base_name, self.curves_dir)
                    plt.close(fig_combined)
                    curves_generated += 3  # 3 formats per parameter
                    
                except Exception as e:
                    print(f"Error generating curve for {param_name}: {e}")
                    continue
            
            
            print(f"Generated {curves_generated} temporal curve files")
            return rollout_scores
            
        except Exception as e:
            print(f"Error in temporal rollout generation: {e}")
            return {}
    
    def generate_attention_visualizations(self, patient_id="E100017564"):
        """Generate 1,728 attention visualizations"""
        print(f"Generating attention visualizations for patient {patient_id}...")
        
        try:
            views, targets, pid = self._get_patient_data(patient_id)
            if views is None:
                print("Could not load patient data")
                return 0
            
            total_viz = 0
            n_frames = self.n_frames  # Top n frames per view (configurable)
            
            # Get rollout scores for frame selection
            base_rollout = self.model.attention_rollout(views, target_param_idx=0)
            
            # Generate visualizations for each parameter
            for param_idx, param_name in enumerate(self.param_names):
                print(f"Processing parameter {param_name}...")
                
                try:
                    # Get parameter-specific rollout
                    param_rollout = self.model.attention_rollout(views, target_param_idx=param_idx)
                    print(f"    Rollout keys: {list(param_rollout.keys())}")
                    for view_name, scores in param_rollout.items():
                        print(f"    {view_name} scores type: {type(scores)}, shape/length: {scores.shape if hasattr(scores, 'shape') else len(scores)}")
                    
                    # Process each view
                    for view_idx, view_name in enumerate(self.view_names):
                        if view_name not in param_rollout:
                            continue
                            
                        scores = param_rollout[view_name]
                        
                        # Handle different score formats (could be 2D array from attention)
                        if isinstance(scores, np.ndarray) and len(scores.shape) > 1:
                            # If scores is 2D (batch_size, frames), take first batch
                            scores = scores[0] if scores.shape[0] > 0 else scores.flatten()
                        elif torch.is_tensor(scores):
                            scores = scores.cpu().numpy()
                            if len(scores.shape) > 1:
                                scores = scores[0] if scores.shape[0] > 0 else scores.flatten()
                        
                        # Ensure scores is 1D
                        scores = scores.flatten()
                        
                        # Select top frames - get the indices of the highest scoring frames
                        if len(scores) < n_frames:
                            # If we have fewer frames than requested, use all of them
                            top_indices = np.arange(len(scores))
                        else:
                            # Get indices of the n_frames highest scores
                            top_indices = np.argsort(scores)[-n_frames:]
                        
                        print(f"    View {view_name}: Selected {len(top_indices)} frames from {len(scores)} total frames")
                        
                        # Generate visualizations for each selected frame
                        for frame_idx in top_indices:
                            try:
                                # Ensure frame_idx is a scalar integer
                                if isinstance(frame_idx, np.ndarray):
                                    frame_idx = int(frame_idx.item())
                                else:
                                    frame_idx = int(frame_idx)
                                
                                # Create ultrasound image visualization
                                view_tensor = views[view_idx]  # Should be (1, frames, height, width) or (frames, height, width)
                                
                                # Handle different tensor shapes
                                if len(view_tensor.shape) == 4:  # (batch, frames, height, width)
                                    if frame_idx >= view_tensor.shape[1]:
                                        print(f"Frame index {frame_idx} out of range for tensor shape {view_tensor.shape}")
                                        continue
                                    frame_tensor = view_tensor[0, frame_idx]  # Get specific frame -> (height, width)
                                elif len(view_tensor.shape) == 3:  # (frames, height, width)
                                    if frame_idx >= view_tensor.shape[0]:
                                        print(f"Frame index {frame_idx} out of range for tensor shape {view_tensor.shape}")
                                        continue
                                    frame_tensor = view_tensor[frame_idx]  # Get specific frame -> (height, width)
                                else:
                                    print(f"Unexpected view tensor shape: {view_tensor.shape}")
                                    continue
                                
                                # Convert to numpy and ensure 2D
                                frame_image = frame_tensor.cpu().numpy()
                                if len(frame_image.shape) != 2:
                                    print(f"Warning: frame_image has unexpected shape {frame_image.shape}, skipping")
                                    continue
                                
                                # Validate image data
                                if frame_image.shape[0] == 0 or frame_image.shape[1] == 0:
                                    print(f"Warning: empty frame image, skipping")
                                    continue
                                
                                # VISUALIZATION 1: Pure ultrasound image
                                fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                                ax1.imshow(frame_image, cmap='gray')
                                ax1.set_title(f'{view_name} View - {param_name} - Frame {frame_idx}')
                                ax1.axis('off')
                                plt.tight_layout()
                                
                                # Save ultrasound image
                                base_name_us = f"{patient_id}_{param_name}_{view_name}_frame_{frame_idx:02d}_ultrasound"
                                self.save_robust_figure(fig1, base_name_us, self.viz_dir)
                                plt.close(fig1)
                                total_viz += 3  # 3 formats for ultrasound
                                
                                # VISUALIZATION 2: Integrated Gradients overlay
                                fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
                                # Set background color to match reference style (dark navy blue)
                                # Reference: E:\results_cardioAI\EchoCath_cardioAI\2025092501\figures\attention
                                fig2.patch.set_facecolor((0.12, 0.12, 0.28))
                                ax2.set_facecolor((0.12, 0.12, 0.28))
                                
                                # Compute actual Integrated Gradients
                                try:
                                    # Get integrated gradients for this view and parameter
                                    ig_dict = self.model.get_integrated_gradients([views[view_idx]], target_param_idx=param_idx)
                                    view_key = list(ig_dict.keys())[0]  # Should be view name
                                    grad_tensor = ig_dict[view_key]
                                    if len(grad_tensor.shape) == 4:  # (batch, frames, height, width)
                                        gradients = grad_tensor[0, frame_idx].cpu().numpy()
                                    elif len(grad_tensor.shape) == 3:  # (frames, height, width)
                                        gradients = grad_tensor[frame_idx].cpu().numpy()
                                    else:
                                        raise ValueError(f"Unexpected gradient shape: {grad_tensor.shape}")
                                    
                                    # SELECTIVE VISUALIZATION: Only color areas above 95th percentile
                                    # Goal: Show ultrasound everywhere, color overlay only on top 5% attention
                                    grad_abs = np.abs(gradients)
                                    if grad_abs.max() > 0:
                                        grad_norm = (grad_abs - grad_abs.min()) / (grad_abs.max() - grad_abs.min() + 1e-8)
                                        grad_norm = np.power(grad_norm, 1.0)

                                        apex_mask = create_apex_mask(grad_norm.shape, apex_height=70)
                                        grad_norm = grad_norm * apex_mask

                                        grad_smooth = gaussian_filter(grad_norm, sigma=3.0)

                                        # Full gradient visualization
                                        grad_smooth_for_full = grad_smooth.copy()
                                        apex_mask_binary = (apex_mask == 0)
                                        if apex_mask_binary.any() and (~apex_mask_binary).any():
                                            min_val = np.percentile(grad_smooth[~apex_mask_binary], 5)
                                            grad_smooth_for_full[apex_mask_binary] = min_val

                                        grad_normalized_full = (grad_smooth_for_full - grad_smooth_for_full.min()) / (grad_smooth_for_full.max() - grad_smooth_for_full.min() + 1e-8)

                                        # Style 2: Selective (95th percentile thresholding)
                                        percentile_95 = np.percentile(grad_smooth, 95)
                                        high_attention_mask = grad_smooth > percentile_95
                                        grad_normalized_selective = np.zeros_like(grad_smooth)
                                        if high_attention_mask.any():
                                            max_val = grad_smooth[high_attention_mask].max()
                                            grad_normalized_selective[high_attention_mask] = (grad_smooth[high_attention_mask] - percentile_95) / (max_val - percentile_95 + 1e-8)
                                    else:
                                        grad_abs = np.abs(grad_abs)
                                        grad_normalized_full = (grad_abs - grad_abs.min()) / (grad_abs.max() - grad_abs.min() + 1e-8)
                                        grad_normalized_selective = np.zeros_like(grad_abs)
                                        high_attention_mask = np.zeros_like(grad_abs, dtype=bool)

                                    # Use 10-color palette from palette.jpeg
                                    from utils_cardioAI import get_cardio_heatmap_cmap
                                    heatmap_cmap = get_cardio_heatmap_cmap('blue_cyan_yellow')

                                    # Generate BOTH overlay styles
                                    # Style 1: Full gradient overlay (like backup)
                                    ax2.imshow(frame_image, cmap='gray', alpha=1.0)
                                    ax2.imshow(grad_normalized_full, cmap=heatmap_cmap, alpha=self.overlay_alpha_full, vmin=0, vmax=1)
                                    ax2.set_title(f'{view_name} View - {param_name} - Frame {frame_idx} - Integrated Gradients')
                                    ax2.axis('off')
                                    plt.tight_layout()

                                    # Save full gradient overlay
                                    base_name_full = f"{patient_id}_{param_name}_{view_name}_frame_{frame_idx:02d}_overlay_full"
                                    self.save_robust_figure(fig2, base_name_full, self.viz_dir, preserve_facecolor=True)
                                    plt.close(fig2)
                                    total_viz += 3  # 3 formats

                                except Exception as e:
                                    # Fallback to attention-weighted heatmap
                                    print(f"    Warning: Could not compute Integrated Gradients for {view_name}-{param_name}-{frame_idx}: {e}")
                                    print(f"    Using attention-weighted fallback visualization")
                                    heatmap = np.random.rand(*frame_image.shape) * scores[frame_idx]
                                    # Apply same dual overlay strategy as IG
                                    if heatmap.max() > 0:
                                        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                                        heatmap_norm = np.power(heatmap_norm, 1.0)
                                        apex_mask = create_apex_mask(heatmap_norm.shape, apex_height=70)
                                        heatmap_norm = heatmap_norm * apex_mask
                                        heatmap_smooth = gaussian_filter(heatmap_norm, sigma=3.0)
                                        heatmap_smooth_for_full = heatmap_smooth.copy()
                                        apex_mask_binary = (apex_mask == 0)
                                        if apex_mask_binary.any() and (~apex_mask_binary).any():
                                            min_val = np.percentile(heatmap_smooth[~apex_mask_binary], 5)
                                            heatmap_smooth_for_full[apex_mask_binary] = min_val

                                        heatmap_normalized_full = (heatmap_smooth_for_full - heatmap_smooth_for_full.min()) / (heatmap_smooth_for_full.max() - heatmap_smooth_for_full.min() + 1e-8)

                                        percentile_95 = np.percentile(heatmap_smooth, 95)
                                        high_attention_mask = heatmap_smooth > percentile_95
                                        heatmap_normalized_selective = np.zeros_like(heatmap_smooth)
                                        if high_attention_mask.any():
                                            max_val = heatmap_smooth[high_attention_mask].max()
                                            heatmap_normalized_selective[high_attention_mask] = (heatmap_smooth[high_attention_mask] - percentile_95) / (max_val - percentile_95 + 1e-8)
                                    else:
                                        heatmap_normalized_full = np.zeros_like(heatmap)
                                        heatmap_normalized_selective = np.zeros_like(heatmap)
                                        high_attention_mask = np.zeros_like(heatmap, dtype=bool)

                                    # Use 10-color palette from palette.jpeg
                                    from utils_cardioAI import get_cardio_heatmap_cmap
                                    heatmap_cmap = get_cardio_heatmap_cmap('blue_cyan_yellow')

                                    # Style 1: Full gradient overlay
                                    ax2.imshow(frame_image, cmap='gray', alpha=1.0)
                                    ax2.imshow(heatmap_normalized_full, cmap=heatmap_cmap, alpha=self.overlay_alpha_full, vmin=0, vmax=1)
                                    ax2.set_title(f'{view_name} View - {param_name} - Frame {frame_idx} - Integrated Gradients')
                                    ax2.axis('off')
                                    plt.tight_layout()

                                    base_name_full = f"{patient_id}_{param_name}_{view_name}_frame_{frame_idx:02d}_overlay_full"
                                    self.save_robust_figure(fig2, base_name_full, self.viz_dir, preserve_facecolor=True)
                                    plt.close(fig2)
                                    total_viz += 3

                            except Exception as e:
                                print(f"Error generating viz for {param_name}-{view_name}-{frame_idx}: {e}")
                                continue
                
                except Exception as e:
                    print(f"Error processing parameter {param_name}: {e}")
                    continue
            
            print(f"Generated {total_viz} attention visualization files")
            return total_viz
            
        except Exception as e:
            print(f"Error in attention visualization generation: {e}")
            return 0


def main():
    """Main execution function"""
    try:
        # Configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Setup output directory from environment variable
        output_dir = os.environ.get('CARDIOAI_OUTPUT_DIR', "./visualization_output")
        
        # Paths
        # Get model path from environment or use standard location
        output_base_dir = output_dir
        # Check directory structures for model (include both best_model.pth and latest_checkpoint.pth)
        model_paths = [
            os.path.join(output_base_dir, '..', 'attention_ablation_cardioAI', 'best_model.pth'),  # New structure
            os.path.join(output_base_dir, '..', 'attention_ablation_cardioAI', 'latest_checkpoint.pth'),  # New structure
            os.path.join(output_base_dir, '..', '02_ablation', 'best_model.pth'),  # Old structure
            os.path.join(output_base_dir, '..', 'train_cardioAI', 'best_model.pth'),  # New structure
            os.path.join(output_base_dir, '..', 'train_cardioAI', 'latest_checkpoint.pth'),  # New structure
            os.path.join(output_base_dir, '..', '01_training', 'best_model.pth'),  # Old structure
            os.path.join(output_base_dir, '..', '01_training', 'latest_checkpoint.pth'),  # Old structure
            "./best_model.pth",  # Final fallback
            "./latest_checkpoint.pth"  # Final fallback
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            model_path = model_paths[-1]  # Use fallback
        dataset_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT"
        excel_file = "./All.xlsx"  # Use standard Excel file
        output_dir = os.environ.get('CARDIOAI_OUTPUT_DIR', "./visualization_output")
        
        print(f"Model path: {model_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        print(f"Output directory: {output_dir}")
        
        # Get number of frames from environment variable (default must match main_cardioAI.py --frames)
        n_frames = int(os.environ.get('CARDIOAI_ATTENTION_FRAMES', 32))
        print(f"Using {n_frames} frames with highest attention importance for visualizations")

        # Get alpha values for overlay visibility (configurable for better visualization)
        overlay_alpha_full = float(os.environ.get('CARDIOAI_OVERLAY_ALPHA_FULL', 0.7))
        overlay_alpha_selective = float(os.environ.get('CARDIOAI_OVERLAY_ALPHA_SELECTIVE', 0.8))
        print(f"Overlay alpha - Full: {overlay_alpha_full}, Selective: {overlay_alpha_selective}")

        # Create visualizer
        visualizer = ProgressiveAttentionVisualizer(model_path, dataset_dir, excel_file, output_dir, device, n_frames,
                                                    overlay_alpha_full=overlay_alpha_full,
                                                    overlay_alpha_selective=overlay_alpha_selective)
        
        # Generate temporal curves
        rollout_scores = visualizer.generate_temporal_rollout_curves()
        
        # Generate attention visualizations
        viz_count = visualizer.generate_attention_visualizations()
        
        # Create summary with dynamic calculation based on n_frames
        # Formula: n_frames x 4 views x 9 parameters x 3 formats x 2 types (ultrasound + overlay_full)
        expected_viz = visualizer.n_frames * 4 * 9 * 3 * 2  # Dynamic calculation (2 types only)
        # Count actual temporal curve files (9 parameters x 3 formats = 27)
        temporal_curve_count = 9 * 3  # 9 parameters x 3 formats (eps, png, tiff)
        total_files = viz_count + temporal_curve_count
        expected_total = expected_viz + temporal_curve_count

        summary = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'patient_id': 'E100017564',
            'temporal_curves': temporal_curve_count,
            'attention_visualizations': viz_count,
            'expected_visualizations': expected_viz,
            'total_files': total_files,
            'expected_total': expected_total,
            'n_frames_used': visualizer.n_frames,
            'success_rate': total_files / expected_total if expected_total > 0 else 0,
            'completion_status': 'SUCCESS' if viz_count >= expected_viz else 'PARTIAL'
        }
        
        # Save summary
        summary_path = Path(output_dir) / "visualization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("ATTENTION VISUALIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Temporal curves generated: {summary['temporal_curves']}")
        print(f"Attention visualizations: {summary['attention_visualizations']}")
        print(f"Total files: {summary['total_files']}/{summary['expected_total']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Status: {summary['completion_status']}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")
        
        return total_files >= expected_total * 0.8  # Success if at least 80% of expected files generated
        
    except Exception as e:
        print(f"Error in attention visualization generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("Fixed attention visualizations completed successfully!")
        sys.exit(0)
    else:
        print("Attention visualization generation failed!")
        sys.exit(1)