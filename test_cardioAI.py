#!/usr/bin/env python3
"""
CardioAI independent test evaluation on patients 236-308.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Local imports
from model_cardioAI import create_model, ProgressiveCardioAI
from dataset_cardioAI import CardioAIDataset
from utils_cardioAI import CardioAIUtils, ColorManager


def calculate_correlation(pred, target):
    """Calculate Pearson correlation coefficient"""
    if len(pred) != len(target):
        return 0.0

    # Convert to numpy arrays
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    # Check for valid data
    if np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0

    # Calculate correlation
    corr = np.corrcoef(pred, target)[0, 1]
    return abs(corr) if not np.isnan(corr) else 0.0


class ExternalValidationAnalyzer:
    """Test analyzer for comprehensive evaluation"""

    def __init__(self, model, device, output_dir, cardio_utils):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.cardio_utils = cardio_utils

        # Create subdirectories
        self.dirs = {
            'figures': self.output_dir / "figures",
            'tables': self.output_dir / "tables",
            'embeddings': self.output_dir / "embeddings"
        }

        # Create directories if they don't exist
        for name, dir_path in self.dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path} ({name})")

        # Initialize color manager for consistent palette usage
        self.color_manager = ColorManager()

        # Extract timestamp from output_dir path
        output_path_parts = Path(output_dir).parts
        timestamp = None
        for part in output_path_parts:
            if part.count('_') == 1 and len(part) == 15:  # Format: YYYYMMDD_HHMMSS
                timestamp = part
                break

        if timestamp:
            # Set up the output directory structure using the existing timestamp
            base_dir = r"E:\results_cardioAI\EchoCath_cardioAI"
            self.color_manager.setup_output_directory(timestamp=timestamp, base_dir=base_dir)

        self.param_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']

    def generate_model_predictions(self):
        """Generate model predictions for test dataset"""
        print("Generating model predictions on test dataset...")

        try:
            # Independent test set: patients 236-308 (indices 235-307)
            tensor_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT"
            excel_file = "./All.xlsx"

            # Get training frames from environment variable
            training_frames = int(os.environ.get('CARDIOAI_TRAINING_FRAMES', 32))

            # Load full dataset (308 patients)
            full_dataset = CardioAIDataset(tensor_dir, excel_file, max_frames=training_frames, subset_size=None)
            print(f"Loaded full dataset with {len(full_dataset)} patients")

            # Independent test: patients 236-308 (indices 235-307 in 0-indexed)
            external_indices = list(range(235, min(308, len(full_dataset))))  # 73 patients
            dataset = Subset(full_dataset, external_indices)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

            print(f"Using patients 236-308 ({len(dataset)} patients) for independent test")

            all_predictions = []
            all_targets = []
            all_patient_ids = []

            self.model.eval()
            with torch.no_grad():
                for views, targets, patient_ids in dataloader:
                    # Ensure proper tensor shapes
                    fixed_views = []
                    for view in views:
                        if len(view.shape) == 5:  # (batch, frames, channels, height, width)
                            view = view.squeeze(2)  # Remove channel dimension if present
                        fixed_views.append(view.to(self.device))

                    targets = targets.to(self.device)

                    try:
                        predictions = self.model(fixed_views, return_aux=False)
                        all_predictions.append(predictions.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())
                        all_patient_ids.extend(patient_ids)
                    except Exception as e:
                        print(f"Error in model forward pass: {e}")
                        continue

            if all_predictions:
                predictions = np.vstack(all_predictions)
                targets = np.vstack(all_targets)

                print(f"Generated predictions for {len(all_patient_ids)} patients")

                # Generate all visualizations
                self._generate_correlation_plots(predictions, targets)
                self._generate_scatter_plots(predictions, targets)
                self.generate_bland_altman_plots(predictions, targets)
                self.generate_heteroscedasticity_analysis(predictions, targets)
                self.generate_roc_curves(predictions, targets)
                self.generate_umap_tsne_embeddings(predictions, targets, all_patient_ids)

                return predictions, targets, all_patient_ids
            else:
                print("No valid predictions generated")
                return None, None, []

        except Exception as e:
            print(f"Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []

    def _generate_correlation_plots(self, predictions, targets):
        """Generate correlation analysis plots"""
        try:
            # Denormalize predictions for proper correlation calculation
            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            # Calculate correlations using denormalized predictions
            correlations = []
            for i in range(pred_denorm.shape[1]):
                corr = calculate_correlation(pred_denorm[:, i], targets[:, i])
                correlations.append(corr)

            # Bar plot of correlations - use unique color for each parameter
            fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

            # Define 9 colors from palette.jpeg - one per parameter
            param_colors = [
                (231/255, 98/255, 84/255),   # RAP - coral/red
                (239/255, 138/255, 71/255),  # SPAP - orange
                (247/255, 170/255, 88/255),  # dpap - yellow-orange
                (255/255, 208/255, 111/255), # meanPAP - yellow
                (255/255, 230/255, 183/255), # PCWP - light yellow
                (170/255, 220/255, 224/255), # CO - light cyan
                (114/255, 188/255, 213/255), # CI - sky blue
                (82/255, 143/255, 173/255),  # SVRI - medium blue
                (55/255, 103/255, 149/255),  # PVR - darker blue
            ]

            bars = ax1.bar(self.param_names, correlations, color=param_colors, alpha=0.8, edgecolor='none')
            ax1.set_xlabel('Parameters', fontsize=14)
            ax1.set_ylabel('Correlation', fontsize=14)
            # Title inside the box for square shape
            ax1.text(0.5, 0.97, 'Test: Model Performance by Parameter',
                    transform=ax1.transAxes, fontsize=14, fontweight='bold',
                    ha='center', va='top')
            ax1.tick_params(axis='x', rotation=45, labelsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            ax1.grid(False)
            ax1.spines['top'].set_visible(True)
            ax1.spines['right'].set_visible(True)

            # Add correlation values on bars
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom', fontsize=11)

            plt.tight_layout()
            self.cardio_utils.save_figure(fig1, 'test_parameter_correlations', subdir='figures')
            plt.close(fig1)

            # Correlation heatmap - bubble plot style
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
            corr_matrix = np.corrcoef(pred_denorm.T)

            # Create bubble plot
            n_params = len(self.param_names)

            # Define colormap - blue (negative) to white to red (positive) diverging
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('correlation', colors_list, N=n_bins)

            # Plot only lower triangle (including diagonal)
            for i in range(n_params):
                for j in range(i + 1):
                    corr_val = corr_matrix[i, j]
                    # Size proportional to absolute correlation (REDUCED by ~40%)
                    size = abs(corr_val) * 1800
                    # Color based on correlation value
                    color = cmap((corr_val + 1) / 2)  # Normalize from [-1,1] to [0,1]
                    ax2.scatter(j, i, s=size, c=[color], alpha=0.8, edgecolors='none')
                    # Add correlation value text on each bubble
                    # Use white text for dark colors (strong positive/negative), black for light colors
                    text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                    fontsize = 8 if abs(corr_val) < 0.3 else 9
                    ax2.text(j, i, f'{corr_val:.2f}', ha='center', va='center',
                            fontsize=fontsize, fontweight='bold', color=text_color)

            # Set axis properties
            ax2.set_xlim(-0.5, n_params - 0.5)
            ax2.set_ylim(-0.5, n_params - 0.5)
            ax2.set_xticks(range(n_params))
            ax2.set_yticks(range(n_params))
            ax2.set_xticklabels(self.param_names, rotation=45, ha='right', fontsize=12)
            ax2.set_yticklabels(self.param_names, fontsize=12)
            ax2.invert_yaxis()
            ax2.set_aspect('equal')
            # Title inside the box for square shape
            ax2.text(0.5, 0.97, 'Attention Ablation - Parameter Correlation Matrix',
                    transform=ax2.transAxes, fontsize=14, fontweight='bold',
                    ha='center', va='top')
            ax2.spines['top'].set_visible(True)
            ax2.spines['right'].set_visible(True)

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=25, fontsize=12)
            cbar.ax.tick_params(labelsize=11)

            plt.tight_layout()
            self.cardio_utils.save_figure(fig2, 'test_correlation_heatmap', subdir='figures')
            plt.close(fig2)

        except Exception as e:
            print(f"Error generating correlation plots: {e}")

    def _generate_scatter_plots(self, predictions, targets):
        """Generate scatter plots for each parameter"""
        try:
            from scipy import stats

            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            # Get meanPAP values for PH diagnosis (index 3 in param_names)
            meanpap_idx = self.param_names.index('meanPAP')
            meanpap_values = targets[:, meanpap_idx]

            # PH diagnosis: meanPAP > 20
            ph_positive = meanpap_values > 20.0
            ph_negative = meanpap_values <= 20.0

            # Define 9 colors from palette (RGB normalized to 0-1)
            palette_colors = [
                (231/255, 98/255, 84/255),   # RAP - coral/red
                (239/255, 138/255, 71/255),  # SPAP - orange
                (247/255, 170/255, 88/255),  # dpap - yellow-orange
                (255/255, 208/255, 111/255), # meanPAP - yellow
                (255/255, 230/255, 183/255), # PCWP - light yellow
                (170/255, 220/255, 224/255), # CO - light cyan
                (114/255, 188/255, 213/255), # CI - sky blue
                (82/255, 143/255, 173/255),  # SVRI - medium blue
                (55/255, 103/255, 149/255),  # PVR - darker blue
            ]

            # Clinical cutoff values for each parameter
            clinical_cutoffs = {
                'RAP': 8.0,      # mmHg
                'SPAP': 40.0,    # mmHg
                'dpap': 15.0,    # mmHg
                'meanPAP': 20.0, # mmHg
                'PCWP': 15.0,    # mmHg
                'CO': 4.0,       # L/min
                'CI': 2.5,       # L/min/m²
                'SVRI': 2400.0,  # dyn·s/cm⁵
                'PVR': 2.0       # Wood Units (2022 ESC/ERS guideline threshold)
            }

            for i, param_name in enumerate(self.param_names):
                pred_param = pred_denorm[:, i]
                true_param = targets[:, i]
                corr = calculate_correlation(pred_param, true_param)

                # Calculate regression line (slope and intercept)
                slope, intercept, r_value, p_value, std_err = stats.linregress(true_param, pred_param)

                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                # Use consistent color for this parameter
                param_color = palette_colors[i]

                # Plot PH positive patients with circle markers (Elevated)
                if np.any(ph_positive):
                    ax.scatter(true_param[ph_positive], pred_param[ph_positive],
                             alpha=0.7, s=120, marker='o', color=param_color, edgecolors='none',
                             label='PH positive (meanPAP > 20)')

                # Plot PH negative patients with square markers (Normal)
                if np.any(ph_negative):
                    ax.scatter(true_param[ph_negative], pred_param[ph_negative],
                             alpha=0.7, s=120, marker='s', color=param_color, edgecolors='none',
                             label='PH negative (meanPAP <= 20)')

                # Add perfect correlation line (identity line)
                min_val = min(true_param.min(), pred_param.min())
                max_val = max(true_param.max(), pred_param.max())
                ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5, linewidth=1, label='Identity')

                # Add regression line
                regression_x = np.array([min_val, max_val])
                regression_y = slope * regression_x + intercept
                ax.plot(regression_x, regression_y, '-', color='black', alpha=0.7, linewidth=2,
                       label=f'Regression: y={slope:.2f}x+{intercept:.2f}')

                # Add clinical cutoff lines
                if param_name in clinical_cutoffs:
                    cutoff = clinical_cutoffs[param_name]
                    # Vertical line for true value cutoff
                    ax.axvline(x=cutoff, color='gray', linestyle=':', linewidth=1.5, alpha=0.6,
                              label=f'Clinical cutoff: {cutoff}')
                    # Horizontal line for predicted value cutoff
                    ax.axhline(y=cutoff, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

                ax.set_xlabel(f'True {param_name}', fontsize=14)
                ax.set_ylabel(f'Predicted {param_name}', fontsize=14)
                # Title inside the box for square shape
                ax.text(0.5, 0.97, f'Test: {param_name} (r={corr:.3f})',
                        transform=ax.transAxes, fontsize=14, fontweight='bold',
                        ha='center', va='top')
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.legend(fontsize=9, loc='best')
                ax.grid(False)
                # Add complete box
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'test_scatter_plot_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating scatter plots: {e}")

    def generate_bland_altman_plots(self, predictions, targets):
        """Generate Bland-Altman plots for agreement analysis"""
        print("Generating Bland-Altman plots...")

        try:
            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            # Get meanPAP values for PH diagnosis (index 3 in param_names)
            meanpap_idx = self.param_names.index('meanPAP')
            meanpap_values = targets[:, meanpap_idx]

            # PH diagnosis: meanPAP > 20
            ph_positive = meanpap_values > 20.0
            ph_negative = meanpap_values <= 20.0

            # Define 9 colors from palette (RGB normalized to 0-1)
            palette_colors = [
                (231/255, 98/255, 84/255),   # RAP - coral/red
                (239/255, 138/255, 71/255),  # SPAP - orange
                (247/255, 170/255, 88/255),  # dpap - yellow-orange
                (255/255, 208/255, 111/255), # meanPAP - yellow
                (255/255, 230/255, 183/255), # PCWP - light yellow
                (170/255, 220/255, 224/255), # CO - light cyan
                (114/255, 188/255, 213/255), # CI - sky blue
                (82/255, 143/255, 173/255),  # SVRI - medium blue
                (55/255, 103/255, 149/255),  # PVR - darker blue
            ]

            for i, param_name in enumerate(self.param_names):
                pred = pred_denorm[:, i]
                true = targets[:, i]

                # Calculate differences and means
                diff = pred - true
                mean_vals = (pred + true) / 2

                # Calculate limits of agreement
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)
                upper_loa = mean_diff + 1.96 * std_diff
                lower_loa = mean_diff - 1.96 * std_diff

                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                # Use consistent color for this parameter
                param_color = palette_colors[i]

                # Plot PH positive patients with circle markers (Elevated)
                if np.any(ph_positive):
                    ax.scatter(mean_vals[ph_positive], diff[ph_positive],
                             alpha=0.7, s=120, marker='o', color=param_color, edgecolors='none',
                             label='PH positive (meanPAP > 20)')

                # Plot PH negative patients with square markers (Normal)
                if np.any(ph_negative):
                    ax.scatter(mean_vals[ph_negative], diff[ph_negative],
                             alpha=0.7, s=120, marker='s', color=param_color, edgecolors='none',
                             label='PH negative (meanPAP <= 20)')

                ax.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean_diff:.2f}')
                ax.axhline(upper_loa, color='gray', linestyle='--', linewidth=1, label=f'Upper LoA: {upper_loa:.2f}')
                ax.axhline(lower_loa, color='gray', linestyle='--', linewidth=1, label=f'Lower LoA: {lower_loa:.2f}')
                ax.set_xlabel(f'Mean of True and Predicted {param_name}', fontsize=14)
                ax.set_ylabel(f'Predicted - True {param_name}', fontsize=14)
                # Title inside the box for square shape
                ax.text(0.5, 0.97, f'Test: {param_name} Bland-Altman Plot',
                        transform=ax.transAxes, fontsize=14, fontweight='bold',
                        ha='center', va='top')
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.legend(fontsize=10)
                ax.grid(False)
                # Add complete box
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'test_bland_altman_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating Bland-Altman plots: {e}")

    def generate_heteroscedasticity_analysis(self, predictions, targets):
        """Generate heteroscedasticity analysis figure and table for 9 parameters"""
        print("Generating heteroscedasticity analysis...")

        try:
            from scipy import stats
            import pandas as pd

            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            # Parameter units for labels
            param_units = {
                'RAP': 'mmHg', 'SPAP': 'mmHg', 'dpap': 'mmHg', 'meanPAP': 'mmHg',
                'PCWP': 'mmHg', 'CO': 'L/min', 'CI': 'L/min/m2',
                'SVRI': 'dyn-s/cm5', 'PVR': 'Wood Units'
            }

            # Colors from palette
            palette_colors = [
                (231/255, 98/255, 84/255),   # RAP
                (239/255, 138/255, 71/255),  # SPAP
                (247/255, 170/255, 88/255),  # dpap
                (255/255, 208/255, 111/255), # meanPAP
                (255/255, 230/255, 183/255), # PCWP
                (170/255, 220/255, 224/255), # CO
                (114/255, 188/255, 213/255), # CI
                (82/255, 143/255, 173/255),  # SVRI
                (55/255, 103/255, 149/255),  # PVR
            ]

            # Create figure with 3x3 subplots for residual variance analysis
            fig, axes = plt.subplots(3, 3, figsize=(14, 12))
            axes = axes.flatten()

            # Store heteroscedasticity statistics for table
            hetero_stats = []

            for idx, param_name in enumerate(self.param_names):
                ax = axes[idx]

                pred_values = pred_denorm[:, idx]
                true_values = targets[:, idx]
                residuals = pred_values - true_values
                abs_residuals = np.abs(residuals)

                # Divide data into quartiles based on true values
                n_bins = 4
                bin_edges = np.percentile(true_values, np.linspace(0, 100, n_bins + 1))
                bin_centers = []
                bin_stds = []
                bin_means = []
                bin_counts = []

                for i in range(n_bins):
                    if i < n_bins - 1:
                        mask = (true_values >= bin_edges[i]) & (true_values < bin_edges[i + 1])
                    else:
                        mask = (true_values >= bin_edges[i]) & (true_values <= bin_edges[i + 1])

                    if mask.sum() > 0:
                        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        bin_stds.append(np.std(residuals[mask]))
                        bin_means.append(np.mean(abs_residuals[mask]))
                        bin_counts.append(mask.sum())

                # Plot residual standard deviation by quartile
                bars = ax.bar(range(len(bin_stds)), bin_stds, color=palette_colors[idx],
                             alpha=0.7, edgecolor='black', linewidth=1)

                # Add value labels on bars
                for bar, std_val in zip(bars, bin_stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(bin_stds),
                           f'{std_val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Set x-axis labels with quartile ranges
                quartile_labels = []
                for i in range(n_bins):
                    quartile_labels.append(f'Q{i+1}\n({bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})')
                ax.set_xticks(range(len(bin_stds)))
                ax.set_xticklabels(quartile_labels, fontsize=8)

                ax.set_xlabel(f'True Value Quartiles ({param_units[param_name]})', fontsize=10)
                ax.set_ylabel('Residual SD', fontsize=10)
                ax.set_title(f'{param_name}', fontsize=12, fontweight='bold')

                # Calculate heteroscedasticity metrics
                # 1. Ratio of highest to lowest quartile SD
                if min(bin_stds) > 0:
                    sd_ratio = max(bin_stds) / min(bin_stds)
                else:
                    sd_ratio = np.nan

                # 2. Breusch-Pagan test (correlation of squared residuals with predicted values)
                squared_residuals = residuals ** 2
                bp_corr, bp_pvalue = stats.pearsonr(true_values, squared_residuals)

                # 3. Spearman correlation of absolute residuals with true values
                spearman_corr, spearman_pvalue = stats.spearmanr(true_values, abs_residuals)

                # 4. Coefficient of variation ratio (CV high / CV low quartile)
                cv_q1 = bin_stds[0] / abs(bin_centers[0]) if abs(bin_centers[0]) > 0 else np.nan
                cv_q4 = bin_stds[-1] / abs(bin_centers[-1]) if abs(bin_centers[-1]) > 0 else np.nan

                hetero_stats.append({
                    'Parameter': param_name,
                    'Unit': param_units[param_name],
                    'SD_Q1': bin_stds[0],
                    'SD_Q2': bin_stds[1] if len(bin_stds) > 1 else np.nan,
                    'SD_Q3': bin_stds[2] if len(bin_stds) > 2 else np.nan,
                    'SD_Q4': bin_stds[-1],
                    'SD_Ratio_Q4_Q1': sd_ratio,
                    'BP_Correlation': bp_corr,
                    'BP_P_Value': bp_pvalue,
                    'Spearman_Corr': spearman_corr,
                    'Spearman_P_Value': spearman_pvalue,
                    'Heteroscedastic': 'Yes' if (sd_ratio > 1.5 or bp_pvalue < 0.05) else 'No'
                })

                # Add heteroscedasticity indicator
                if sd_ratio > 1.5 or bp_pvalue < 0.05:
                    ax.text(0.95, 0.95, 'Heteroscedastic', transform=ax.transAxes,
                           fontsize=9, ha='right', va='top', color='red', fontweight='bold')
                else:
                    ax.text(0.95, 0.95, 'Homoscedastic', transform=ax.transAxes,
                           fontsize=9, ha='right', va='top', color='green', fontweight='bold')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plt.suptitle('Heteroscedasticity Analysis: Residual Standard Deviation by Value Quartiles',
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig, 'test_heteroscedasticity_analysis', subdir='figures')
            plt.close(fig)

            # Create summary table
            df_hetero = pd.DataFrame(hetero_stats)

            # Save as Excel
            table_path = self.cardio_utils.current_output_dir / 'tables' / 'test_heteroscedasticity_summary.xlsx'
            df_hetero.to_excel(table_path, index=False, float_format='%.4f')
            print(f"Heteroscedasticity table saved to: {table_path}")

            # Create visual summary table figure
            fig_table, ax_table = plt.subplots(figsize=(16, 6))
            ax_table.axis('off')

            # Prepare table data
            table_data = []
            headers = ['Parameter', 'SD Q1', 'SD Q2', 'SD Q3', 'SD Q4', 'SD Ratio\n(Q4/Q1)',
                      'BP Corr', 'BP p-value', 'Status']

            for stat in hetero_stats:
                row = [
                    stat['Parameter'],
                    f"{stat['SD_Q1']:.2f}",
                    f"{stat['SD_Q2']:.2f}" if not np.isnan(stat['SD_Q2']) else '-',
                    f"{stat['SD_Q3']:.2f}" if not np.isnan(stat['SD_Q3']) else '-',
                    f"{stat['SD_Q4']:.2f}",
                    f"{stat['SD_Ratio_Q4_Q1']:.2f}",
                    f"{stat['BP_Correlation']:.3f}",
                    f"{stat['BP_P_Value']:.4f}" if stat['BP_P_Value'] >= 0.0001 else '<0.0001',
                    stat['Heteroscedastic']
                ]
                table_data.append(row)

            # Create table
            table = ax_table.table(cellText=table_data, colLabels=headers, loc='center',
                                  cellLoc='center', colColours=['lightgray'] * len(headers))
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)

            # Color code the status column
            for i, stat in enumerate(hetero_stats):
                cell = table[(i + 1, 8)]  # Status column
                if stat['Heteroscedastic'] == 'Yes':
                    cell.set_facecolor('#ffcccc')  # Light red
                else:
                    cell.set_facecolor('#ccffcc')  # Light green

            plt.title('Heteroscedasticity Summary: Residual Variance Analysis Across Value Ranges',
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig_table, 'test_heteroscedasticity_table', subdir='figures')
            plt.close(fig_table)

            print("Heteroscedasticity analysis completed")

        except Exception as e:
            print(f"Error generating heteroscedasticity analysis: {e}")
            import traceback
            traceback.print_exc()

    def generate_roc_curves(self, predictions, targets):
        """Generate ROC curves with AUC values"""
        print("Generating ROC curves...")

        try:
            from sklearn.metrics import roc_curve, auc

            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            # Clinical cutoff values for each parameter
            clinical_cutoffs = {
                'RAP': 8.0,      # mmHg
                'SPAP': 40.0,    # mmHg
                'dpap': 15.0,    # mmHg
                'meanPAP': 20.0, # mmHg (PH gold standard)
                'PCWP': 15.0,    # mmHg
                'CO': 4.0,       # L/min (low if <4)
                'CI': 2.5,       # L/min/m² (low if <2.5)
                'SVRI': 2400.0,  # dyn·s/cm⁵
                'PVR': 2.0       # Wood Units (2022 ESC/ERS guideline threshold)
            }

            # Get meanPAP for gold-standard PH diagnosis
            meanpap_idx = self.param_names.index('meanPAP')
            meanpap_true = targets[:, meanpap_idx]
            meanpap_pred = pred_denorm[:, meanpap_idx]

            # Define 9 colors from palette (RGB normalized to 0-1)
            palette_colors = [
                (231/255, 98/255, 84/255),   # RAP - coral/red
                (239/255, 138/255, 71/255),  # SPAP - orange
                (247/255, 170/255, 88/255),  # dpap - yellow-orange
                (255/255, 208/255, 111/255), # meanPAP - yellow
                (255/255, 230/255, 183/255), # PCWP - light yellow
                (170/255, 220/255, 224/255), # CO - light cyan
                (114/255, 188/255, 213/255), # CI - sky blue
                (82/255, 143/255, 173/255),  # SVRI - medium blue
                (55/255, 103/255, 149/255),  # PVR - darker blue
            ]

            for i, param_name in enumerate(self.param_names):
                pred = pred_denorm[:, i]
                true = targets[:, i]

                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                # ROC 1: Parameter-specific clinical cutoff
                if param_name in clinical_cutoffs:
                    cutoff = clinical_cutoffs[param_name]

                    # For CO and CI, positive = low values (< cutoff)
                    # For others, positive = high values (> cutoff)
                    if param_name in ['CO', 'CI']:
                        y_true_param = (true < cutoff).astype(int)
                        y_scores_param = -pred  # Negate so lower values give higher scores
                    else:
                        y_true_param = (true > cutoff).astype(int)
                        y_scores_param = pred

                    # Calculate ROC curve for parameter-specific cutoff
                    fpr_param, tpr_param, _ = roc_curve(y_true_param, y_scores_param)
                    roc_auc_param = auc(fpr_param, tpr_param)

                    ax.plot(fpr_param, tpr_param, color=palette_colors[i], linewidth=2.5,
                           label=f'{param_name} (AUC={roc_auc_param:.3f})')

                # ROC 2: Gold-standard PH diagnosis (meanPAP > 20) - only if not meanPAP itself
                if param_name != 'meanPAP':
                    y_true_ph = (meanpap_true > 20.0).astype(int)
                    y_scores_ph = pred

                    # Calculate ROC curve for PH diagnosis
                    fpr_ph, tpr_ph, _ = roc_curve(y_true_ph, y_scores_ph)
                    roc_auc_ph = auc(fpr_ph, tpr_ph)

                    ax.plot(fpr_ph, tpr_ph, color='gray', linewidth=2, linestyle='--',
                           label=f'PH diagnosis (meanPAP>20, AUC={roc_auc_ph:.3f})')

                # Diagonal reference line
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

                ax.set_xlabel('False Positive Rate', fontsize=14)
                ax.set_ylabel('True Positive Rate', fontsize=14)
                # Title inside the box for square shape
                ax.text(0.5, 0.97, f'Test: {param_name} ROC Curve',
                        transform=ax.transAxes, fontsize=14, fontweight='bold',
                        ha='center', va='top')
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.legend(fontsize=10, loc='lower right')
                ax.grid(False)
                # Add complete box
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'test_roc_curve_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating ROC curves: {e}")

    def generate_umap_tsne_embeddings(self, predictions, targets, patient_ids):
        """Generate UMAP and t-SNE embeddings with KMeans/DBSCAN clustering"""
        print("Generating UMAP and t-SNE embeddings...")

        try:
            from sklearn.manifold import TSNE
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.preprocessing import StandardScaler
            import umap

            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            # Validate inputs
            if pred_denorm is None or targets is None:
                print("Skipping embeddings - predictions or targets are None")
                return

            # Remove NaN values
            valid_mask = ~(np.isnan(pred_denorm).any(axis=1) | np.isnan(targets).any(axis=1))
            if not valid_mask.any():
                print("Skipping embeddings - all data contains NaN")
                return

            valid_predictions = pred_denorm[valid_mask]
            valid_targets = targets[valid_mask]

            if len(valid_predictions) < 5:
                print("Skipping embeddings - insufficient valid data points")
                return

            # Combine predictions and targets for embedding
            combined_data = np.hstack([valid_predictions, valid_targets])
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(combined_data)

            # Generate embeddings
            print("Computing UMAP embedding...")
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            umap_embedding = umap_reducer.fit_transform(scaled_data)

            print("Computing t-SNE embedding...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_data)-1))
            tsne_embedding = tsne.fit_transform(scaled_data)

            # Clustering with 2 groups
            print("Performing binary clustering...")
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans_labels = kmeans.fit_predict(scaled_data)

            # Convert to positive/negative labels
            cluster_means = []
            for cluster_id in [0, 1]:
                cluster_mask = (kmeans_labels == cluster_id)
                if cluster_mask.any():
                    cluster_target_mean = valid_targets[cluster_mask].mean()
                    cluster_means.append((cluster_id, cluster_target_mean))

            cluster_means.sort(key=lambda x: x[1], reverse=True)
            positive_cluster = cluster_means[0][0] if cluster_means else 0
            binary_labels = np.where(kmeans_labels == positive_cluster, 0, 1)

            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.3, min_samples=3)
            dbscan_raw_labels = dbscan.fit_predict(scaled_data)
            dbscan_labels = np.where(dbscan_raw_labels == -1, 1, dbscan_raw_labels % 2)

            # Define colors for positive/negative groups
            colors = ['#2E8B57', '#DC143C']  # Green for positive, Red for negative
            labels = ['Positive', 'Negative']

            # UMAP with KMeans
            fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
            for i, label in enumerate(labels):
                mask = (binary_labels == i)
                ax1.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            # Title inside the box for square shape
            ax1.text(0.5, 0.97, 'Test: UMAP with Binary Clustering',
                    transform=ax1.transAxes, fontsize=14, fontweight='bold',
                    ha='center', va='top')
            ax1.set_xlabel('UMAP 1', fontsize=14)
            ax1.set_ylabel('UMAP 2', fontsize=14)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax1.legend(fontsize=11)
            ax1.spines['top'].set_visible(True)
            ax1.spines['right'].set_visible(True)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig1, 'test_umap_kmeans_embedding', subdir='embeddings')
            plt.close(fig1)

            # UMAP with DBSCAN
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
            for i, label in enumerate(labels):
                mask = (dbscan_labels == i)
                ax2.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            # Title inside the box for square shape
            ax2.text(0.5, 0.97, 'Test: UMAP with DBSCAN Clustering',
                    transform=ax2.transAxes, fontsize=14, fontweight='bold',
                    ha='center', va='top')
            ax2.set_xlabel('UMAP 1', fontsize=14)
            ax2.set_ylabel('UMAP 2', fontsize=14)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            ax2.legend(fontsize=11)
            ax2.spines['top'].set_visible(True)
            ax2.spines['right'].set_visible(True)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig2, 'test_umap_dbscan_embedding', subdir='embeddings')
            plt.close(fig2)

            # t-SNE with KMeans
            fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
            for i, label in enumerate(labels):
                mask = (binary_labels == i)
                ax3.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            # Title inside the box for square shape
            ax3.text(0.5, 0.97, 'Test: t-SNE with Binary Clustering',
                    transform=ax3.transAxes, fontsize=14, fontweight='bold',
                    ha='center', va='top')
            ax3.set_xlabel('t-SNE 1', fontsize=14)
            ax3.set_ylabel('t-SNE 2', fontsize=14)
            ax3.tick_params(axis='both', which='major', labelsize=12)
            ax3.legend(fontsize=11)
            ax3.spines['top'].set_visible(True)
            ax3.spines['right'].set_visible(True)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig3, 'test_tsne_kmeans_embedding', subdir='embeddings')
            plt.close(fig3)

            # t-SNE with DBSCAN
            fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
            for i, label in enumerate(labels):
                mask = (dbscan_labels == i)
                ax4.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            # Title inside the box for square shape
            ax4.text(0.5, 0.97, 'Test: t-SNE with DBSCAN Clustering',
                    transform=ax4.transAxes, fontsize=14, fontweight='bold',
                    ha='center', va='top')
            ax4.set_xlabel('t-SNE 1', fontsize=14)
            ax4.set_ylabel('t-SNE 2', fontsize=14)
            ax4.tick_params(axis='both', which='major', labelsize=12)
            ax4.legend(fontsize=11)
            ax4.spines['top'].set_visible(True)
            ax4.spines['right'].set_visible(True)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig4, 'test_tsne_dbscan_embedding', subdir='embeddings')
            plt.close(fig4)

            # Save embedding data
            embedding_data = {
                'umap_embedding': umap_embedding.tolist(),
                'tsne_embedding': tsne_embedding.tolist(),
                'binary_labels': binary_labels.tolist(),
                'binary_dbscan_labels': dbscan_labels.tolist(),
                'patient_ids': patient_ids if isinstance(patient_ids, list) else list(patient_ids)
            }

            with open(self.dirs['embeddings'] / 'test_embedding_data.json', 'w') as f:
                json.dump(embedding_data, f, indent=2)

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            import traceback
            traceback.print_exc()

    def generate_summary_report(self, predictions, targets, patient_ids):
        """Generate comprehensive summary report"""
        print("Generating summary report...")

        try:
            excel_path = self.dirs['tables'] / 'test_summary.xlsx'

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Summary Statistics
                summary_data = {
                    'Metric': ['Dataset', 'Total Patients', 'Parameters Analyzed', 'Model Architecture'],
                    'Value': ['Test (All.xlsx, patients 236-308)', len(patient_ids) if patient_ids else 0,
                             len(self.param_names), 'Multistage Transformer CardioAI']
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                # Sheet 2: Model Performance
                if predictions is not None and targets is not None:
                    pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
                    pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

                    correlations = [calculate_correlation(pred_denorm[:, i], targets[:, i])
                                  for i in range(len(self.param_names))]
                    mse_values = [np.mean((pred_denorm[:, i] - targets[:, i])**2)
                                for i in range(len(self.param_names))]

                    performance_data = {
                        'Parameter': self.param_names,
                        'Correlation': correlations,
                        'MSE': mse_values,
                        'Target_Min': [0.6] * len(self.param_names),
                        'Meets_Target': [c >= 0.6 for c in correlations]
                    }
                    pd.DataFrame(performance_data).to_excel(writer, sheet_name='Model_Performance', index=False)

                    # Sheet 3: Detailed Predictions
                    if len(patient_ids) > 0:
                        pred_data = {'Patient_ID': patient_ids[:len(predictions)]}
                        for i, param in enumerate(self.param_names):
                            pred_data[f'Predicted_{param}'] = pred_denorm[:, i]
                            pred_data[f'True_{param}'] = targets[:, i]
                            pred_data[f'Error_{param}'] = pred_denorm[:, i] - targets[:, i]

                        pd.DataFrame(pred_data).to_excel(writer, sheet_name='Detailed_Predictions', index=False)

            print(f"Summary report saved to: {excel_path}")

        except Exception as e:
            print(f"Error generating summary report: {e}")


def main():
    """Main execution function"""
    try:
        # Configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Use output directory from environment
        output_dir = Path(os.environ.get('CARDIOAI_OUTPUT_DIR', r"E:\results_cardioAI\EchoCath_cardioAI\20251120_162608\test_cardioAI"))
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CardioAI utilities
        cardio_utils = CardioAIUtils()
        cardio_utils.current_output_dir = output_dir
        cardio_utils.subdirs = {
            'figures': output_dir / "figures",
            'embeddings': output_dir / "embeddings",
            'tables': output_dir / "tables",
        }

        print(f"Output directory: {output_dir}")

        # Load trained model
        model_paths = [
            os.path.join(output_dir, '..', 'train_cardioAI', 'best_model.pth'),
            os.path.join(output_dir, '..', 'train_cardioAI', 'latest_checkpoint.pth'),
            os.path.join(output_dir, '..', 'attention_ablation_cardioAI', 'best_model.pth'),
            r"E:\results_cardioAI\EchoCath_cardioAI\20251120_162608\train_cardioAI\best_model.pth",
            r"E:\results_cardioAI\EchoCath_cardioAI\20251120_162608\train_cardioAI\latest_checkpoint.pth",
            "./best_model.pth"
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise RuntimeError("No trained model found. Please run training first.")

        print(f"Loading model from {model_path}")

        # Load dataset for normalization parameters from TRAINING dataset (All.xlsx)
        print("Loading normalization parameters from training dataset (All.xlsx)...")
        training_tensor_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT"
        training_excel_file = "./All.xlsx"
        training_frames = int(os.environ.get('CARDIOAI_TRAINING_FRAMES', 32))

        # Create dataset to get normalization parameters from training data
        temp_dataset = CardioAIDataset(training_tensor_dir, training_excel_file, max_frames=training_frames, subset_size=10)
        norm_params = temp_dataset.get_normalization_parameters()
        print(f"Normalization parameters loaded from training dataset: log_transform_indices = {norm_params['log_transform_indices']}")

        # Create model
        model = create_model(num_outputs=9, num_frames=training_frames, num_views=4)
        model.set_winsorized_normalization(norm_params)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(f"Warning: Model architecture mismatch. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        print("Model loaded successfully")

        model = model.to(device)
        model.eval()

        # Create analyzer
        analyzer = ExternalValidationAnalyzer(model, device, output_dir, cardio_utils)

        # Generate predictions and all visualizations
        predictions, targets, patient_ids = analyzer.generate_model_predictions()

        # Generate summary report
        if predictions is not None and targets is not None:
            analyzer.generate_summary_report(predictions, targets, patient_ids)

        print(f"\n{'='*60}")
        print("EXTERNAL VALIDATION COMPLETED")
        print(f"{'='*60}")
        print(f"Patients evaluated: {len(patient_ids) if patient_ids else 0}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("Test completed successfully!")
        sys.exit(0)
    else:
        print("Test completed with issues!")
        sys.exit(1)
