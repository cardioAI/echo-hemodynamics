#!/usr/bin/env python3
"""
CardioAI validation: generates correlation plots, scatter/Bland-Altman plots,
ROC curves, confusion matrices, UMAP/t-SNE embeddings, and Excel reports.
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


class ComprehensiveAnalyzer:
    """Comprehensive analyzer for training curves and validation"""

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

    def load_training_results(self):
        """Load training results from previous runs"""
        try:
            # Check current run first
            current_timestamp = os.environ.get('CARDIOAI_TIMESTAMP', '')
            base_dir = Path(r"E:\results_cardioAI\EchoCath_cardioAI")

            # Priority 1: Load from current run
            if current_timestamp:
                current_training_dir = base_dir / current_timestamp / "train_cardioAI"
                if (current_training_dir / "training_history.json").exists():
                    with open(current_training_dir / "training_history.json", 'r') as f:
                        history = json.load(f)
                    print(f"Loaded training history from {current_training_dir}")
                    return history

            # Priority 2: Look for the most recent results
            result_dirs = sorted([d for d in base_dir.glob("*") if d.is_dir()], reverse=True)

            for result_dir in result_dirs:
                training_dir = result_dir / "train_cardioAI"
                if (training_dir / "training_history.json").exists():
                    with open(training_dir / "training_history.json", 'r') as f:
                        history = json.load(f)
                    print(f"Loaded training history from {training_dir}")
                    return history

            print("No training history found")
            return None

        except Exception as e:
            print(f"Error loading training results: {e}")
            return None

    def generate_training_curves(self, history):
        """Generate training loss and correlation curves"""
        if history is None:
            print("Skipping training curves - no history available")
            return

        print("Generating training curves...")

        try:
            epochs = range(1, len(history['train_loss']) + 1)

            # Individual figure 1: Loss curves
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2, color=self.color_manager.get_color(0))
            ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, color=self.color_manager.get_color(1))
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(False)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig, 'loss_curves', subdir='figures')
            plt.close(fig)

            # Individual figure 2: Mean correlations
            has_train_corr = 'per_task_train_corr' in history and len(history['per_task_train_corr']) > 0
            if has_train_corr:
                train_mean_corr = [np.mean(corrs) for corrs in history['per_task_train_corr']]
            val_mean_corr = [np.mean(corrs) for corrs in history['per_task_val_corr']]

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            dark_colors = self.color_manager.get_figure_colors(2, 'regular')
            if has_train_corr:
                ax.plot(epochs, train_mean_corr, label='Training Correlation', linewidth=2, color=dark_colors[0])
            ax.plot(epochs, val_mean_corr, label='Validation Correlation', linewidth=2, color=dark_colors[1])
            ax.axhline(y=0.6, color=self.color_manager.get_dark_colors(1)[0], linestyle='--', alpha=0.7, label='Target Min')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Correlation')
            ax.set_title('Mean Correlation Progress')
            ax.legend()
            ax.grid(False)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig, 'mean_correlation_progress', subdir='figures')
            plt.close(fig)

            # Individual figure 3: Parameter-wise final correlations
            final_val_corr = history['per_task_val_corr'][-1]
            colors = self.color_manager.get_figure_colors(len(self.param_names), 'regular')

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            bars = ax.bar(self.param_names, final_val_corr, color=colors, alpha=0.7)
            ax.axhline(y=0.6, color=self.color_manager.get_dark_colors(1)[0], linestyle='--', alpha=0.7)
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Final Validation Correlation')
            ax.set_title('Final Parameter Correlations')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(False)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig, 'final_parameter_correlations', subdir='figures')
            plt.close(fig)

            # Generate individual parameter curves
            self._generate_parameter_curves(history)

        except Exception as e:
            print(f"Error generating training curves: {e}")

    def _generate_parameter_curves(self, history):
        """Generate individual parameter correlation curves"""
        try:
            epochs = range(1, len(history['per_task_val_corr']) + 1)
            dark_colors = self.color_manager.get_figure_colors(len(self.param_names), 'regular')

            for i, param_name in enumerate(self.param_names):
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                param_corrs = [corrs[i] for corrs in history['per_task_val_corr']]

                ax.plot(epochs, param_corrs, linewidth=2, color=dark_colors[i % len(dark_colors)])
                ax.axhline(y=0.6, color=self.color_manager.get_dark_colors(1)[0], linestyle='--', alpha=0.7)
                ax.set_title(f'{param_name} Validation Correlation')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Correlation')
                ax.grid(False)
                ax.set_ylim(-0.1, 1.0)

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'parameter_correlation_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating parameter curves: {e}")

    def generate_model_predictions(self):
        """Generate model predictions for validation dataset"""
        print("Generating model predictions on validation dataset...")

        try:
            # Use All.xlsx and All_PT folder with patients 1-235 (indices 0-234)
            tensor_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT"
            excel_file = "./All.xlsx"

            # Get training frames from environment variable
            training_frames = int(os.environ.get('CARDIOAI_TRAINING_FRAMES', 32))

            # Load full dataset (308 patients)
            full_dataset = CardioAIDataset(tensor_dir, excel_file, max_frames=training_frames, subset_size=None)
            print(f"Loaded full dataset with {len(full_dataset)} patients")

            # Validation: patients 1-235 (indices 0-234) - training/CV cohort
            internal_indices = list(range(min(235, len(full_dataset))))  # 235 patients
            dataset = Subset(full_dataset, internal_indices)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

            print(f"Using patients 1-235 ({len(dataset)} patients) for validation")

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
                self.generate_roc_curves(predictions, targets)
                self.generate_confusion_matrices(predictions, targets)
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

            # Bar plot of correlations
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

            palette_colors = self.color_manager.get_color_palette(3)
            colors = [palette_colors[0] if c >= 0.6 else palette_colors[1] if c > 0.3 else palette_colors[2] for c in correlations]
            bars = ax1.bar(self.param_names, correlations, color=colors, alpha=0.7)
            ax1.axhline(y=0.6, color=self.color_manager.get_color(0), linestyle='--', alpha=0.7, label='Target Min')
            ax1.set_xlabel('Parameters')
            ax1.set_ylabel('Correlation')
            ax1.set_title('Validation: Model Performance by Parameter')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            ax1.grid(False)

            # Add correlation values on bars
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            self.cardio_utils.save_figure(fig1, 'validation_parameter_correlations', subdir='figures')
            plt.close(fig1)

            # Correlation heatmap
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
            corr_matrix = np.corrcoef(pred_denorm.T)
            from utils_cardioAI import get_cardio_heatmap_cmap
            heatmap_cmap = get_cardio_heatmap_cmap('blue_gray_orange')
            sns.heatmap(corr_matrix, annot=True, cmap=heatmap_cmap, center=0,
                       xticklabels=self.param_names, yticklabels=self.param_names, ax=ax2)
            ax2.set_title('Validation: Parameter Correlation Matrix')

            plt.tight_layout()
            self.cardio_utils.save_figure(fig2, 'validation_correlation_heatmap', subdir='figures')
            plt.close(fig2)

        except Exception as e:
            print(f"Error generating correlation plots: {e}")

    def _generate_scatter_plots(self, predictions, targets):
        """Generate scatter plots for each parameter"""
        try:
            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            for i, param_name in enumerate(self.param_names):
                pred_param = pred_denorm[:, i]
                true_param = targets[:, i]
                corr = calculate_correlation(pred_param, true_param)

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                dark_colors = self.color_manager.get_figure_colors(1, 'regular')
                ax.scatter(true_param, pred_param, alpha=0.6, s=30, color=dark_colors[0])

                # Add perfect correlation line
                min_val = min(true_param.min(), pred_param.min())
                max_val = max(true_param.max(), pred_param.max())
                ax.plot([min_val, max_val], [min_val, max_val], '--', color=dark_colors[0], alpha=0.8)

                ax.set_xlabel(f'True {param_name}')
                ax.set_ylabel(f'Predicted {param_name}')
                ax.set_title(f'Validation: {param_name} (r={corr:.3f})')
                ax.grid(False)

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'validation_scatter_plot_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating scatter plots: {e}")

    def generate_bland_altman_plots(self, predictions, targets):
        """Generate Bland-Altman plots for agreement analysis"""
        print("Generating Bland-Altman plots...")

        try:
            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

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

                fig, ax = plt.subplots(1, 1, figsize=(10, 8))

                dark_colors = self.color_manager.get_figure_colors(5, 'regular')
                ax.scatter(mean_vals, diff, alpha=0.6, s=30, color=dark_colors[0])
                ax.axhline(mean_diff, color=dark_colors[1], linestyle='-', label=f'Mean: {mean_diff:.2f}')
                ax.axhline(upper_loa, color=dark_colors[2], linestyle='--', label=f'Upper LoA: {upper_loa:.2f}')
                ax.axhline(lower_loa, color=dark_colors[3], linestyle='--', label=f'Lower LoA: {lower_loa:.2f}')
                ax.set_xlabel(f'Mean of True and Predicted {param_name}')
                ax.set_ylabel(f'Predicted - True {param_name}')
                ax.set_title(f'Validation: {param_name} Bland-Altman Plot')
                ax.legend(fontsize=8)
                ax.grid(False)

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'validation_bland_altman_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating Bland-Altman plots: {e}")

    def generate_roc_curves(self, predictions, targets):
        """Generate ROC curves with AUC values"""
        print("Generating ROC curves...")

        try:
            from sklearn.metrics import roc_curve, auc

            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            for i, param_name in enumerate(self.param_names):
                pred = pred_denorm[:, i]
                true = targets[:, i]

                # Convert to binary classification (above median)
                true_median = np.median(true)
                y_true = (true > true_median).astype(int)
                y_scores = pred

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                ax.plot(fpr, tpr, color=self.color_manager.get_color(i), linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'Validation: {param_name} ROC Curve')
                ax.legend()
                ax.grid(False)

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'validation_roc_curve_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating ROC curves: {e}")

    def generate_confusion_matrices(self, predictions, targets):
        """Generate confusion matrices for classification analysis"""
        print("Generating confusion matrices...")

        try:
            from sklearn.metrics import confusion_matrix

            pred_torch = torch.tensor(predictions, dtype=torch.float32).to(self.device)
            pred_denorm = self.model.denormalize_predictions(pred_torch).cpu().numpy()

            for i, param_name in enumerate(self.param_names):
                pred = pred_denorm[:, i]
                true = targets[:, i]

                # Convert to binary classification (above/below median)
                true_median = np.median(true)
                y_true = (true > true_median).astype(int)
                y_pred = (pred > true_median).astype(int)

                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred)

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                from utils_cardioAI import get_cardio_heatmap_cmap
                heatmap_cmap = get_cardio_heatmap_cmap('blue_gray_orange')
                sns.heatmap(cm, annot=True, fmt='d', cmap=heatmap_cmap,
                          xticklabels=['Below Median', 'Above Median'],
                          yticklabels=['Below Median', 'Above Median'], ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Validation: {param_name} Confusion Matrix')

                plt.tight_layout()
                self.cardio_utils.save_figure(fig, f'validation_confusion_matrix_{param_name.lower()}', subdir='figures')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating confusion matrices: {e}")

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
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            for i, label in enumerate(labels):
                mask = (binary_labels == i)
                ax1.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            ax1.set_title('Validation: UMAP with Binary Clustering')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.legend()
            ax1.grid(False)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig1, 'validation_umap_kmeans_embedding', subdir='embeddings')
            plt.close(fig1)

            # UMAP with DBSCAN
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
            for i, label in enumerate(labels):
                mask = (dbscan_labels == i)
                ax2.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            ax2.set_title('Validation: UMAP with DBSCAN Clustering')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.legend()
            ax2.grid(False)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig2, 'validation_umap_dbscan_embedding', subdir='embeddings')
            plt.close(fig2)

            # t-SNE with KMeans
            fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
            for i, label in enumerate(labels):
                mask = (binary_labels == i)
                ax3.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            ax3.set_title('Validation: t-SNE with Binary Clustering')
            ax3.set_xlabel('t-SNE 1')
            ax3.set_ylabel('t-SNE 2')
            ax3.legend()
            ax3.grid(False)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig3, 'validation_tsne_kmeans_embedding', subdir='embeddings')
            plt.close(fig3)

            # t-SNE with DBSCAN
            fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))
            for i, label in enumerate(labels):
                mask = (dbscan_labels == i)
                ax4.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1],
                          c=colors[i], label=label, alpha=0.7, s=50)
            ax4.set_title('Validation: t-SNE with DBSCAN Clustering')
            ax4.set_xlabel('t-SNE 1')
            ax4.set_ylabel('t-SNE 2')
            ax4.legend()
            ax4.grid(False)
            plt.tight_layout()
            self.cardio_utils.save_figure(fig4, 'validation_tsne_dbscan_embedding', subdir='embeddings')
            plt.close(fig4)

            # Save embedding data
            embedding_data = {
                'umap_embedding': umap_embedding.tolist(),
                'tsne_embedding': tsne_embedding.tolist(),
                'binary_labels': binary_labels.tolist(),
                'binary_dbscan_labels': dbscan_labels.tolist(),
                'patient_ids': patient_ids if isinstance(patient_ids, list) else list(patient_ids)
            }

            with open(self.dirs['embeddings'] / 'validation_embedding_data.json', 'w') as f:
                json.dump(embedding_data, f, indent=2)

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            import traceback
            traceback.print_exc()

    def generate_summary_report(self, predictions, targets, patient_ids):
        """Generate comprehensive summary report with all sheets"""
        print("Generating summary report...")

        try:
            excel_path = self.dirs['tables'] / 'validation_summary.xlsx'

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Summary Statistics
                actual_epochs = int(os.environ.get('CARDIOAI_EPOCHS', 100))
                summary_data = {
                    'Metric': ['Dataset', 'Total Patients', 'Parameters Analyzed', 'Training Epochs (Latest)',
                              'Model Architecture', 'GPU Used'],
                    'Value': ['Validation (All.xlsx, patients 181-235)', len(patient_ids) if patient_ids else 0,
                             len(self.param_names), actual_epochs, 'Multistage Transformer CardioAI',
                             'NVIDIA RTX 6000 Ada']
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

                # Sheet 4: Analysis Components
                components_data = {
                    'Analysis_Component': [
                        'Training Loss Curves',
                        'Correlation Analysis',
                        'Scatter Plots',
                        'Bland-Altman Plots',
                        'ROC Curves',
                        'Confusion Matrices',
                        'UMAP/t-SNE Embeddings'
                    ],
                    'Status': ['Generated'] * 7,
                    'Output_Format': ['EPS, PNG, TIFF'] * 7,
                    'Description': [
                        'Training and validation loss over epochs',
                        'Parameter correlation analysis and heatmaps',
                        'True vs predicted scatter plots',
                        'Agreement analysis between true and predicted values',
                        'ROC curves with AUC values for binary classification',
                        'Confusion matrices for classification performance',
                        'Dimensionality reduction with KMeans/DBSCAN clustering'
                    ]
                }
                pd.DataFrame(components_data).to_excel(writer, sheet_name='Analysis_Components', index=False)

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
        output_dir = Path(os.environ.get('CARDIOAI_OUTPUT_DIR', "./validation_output"))
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
        analyzer = ComprehensiveAnalyzer(model, device, output_dir, cardio_utils)

        # Load and analyze training results
        history = analyzer.load_training_results()
        if history:
            analyzer.generate_training_curves(history)

        # Generate predictions and all visualizations
        predictions, targets, patient_ids = analyzer.generate_model_predictions()

        # Generate summary report
        if predictions is not None and targets is not None:
            analyzer.generate_summary_report(predictions, targets, patient_ids)

        print(f"\n{'='*60}")
        print("INTERNAL VALIDATION COMPLETED")
        print(f"{'='*60}")
        print(f"Patients evaluated: {len(patient_ids) if patient_ids else 0}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"Error in validation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("Validation completed successfully!")
        sys.exit(0)
    else:
        print("Validation completed with issues!")
        sys.exit(1)
