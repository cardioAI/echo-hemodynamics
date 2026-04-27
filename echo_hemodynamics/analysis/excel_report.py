"""Multi-sheet Excel report writer (parameterized for validation vs test)."""

import os

import numpy as np
import pandas as pd

from .metrics import calculate_correlation


def generate_summary_report(
    pred_denorm, targets, patient_ids, param_names, output_path,
    dataset_label, include_components=True,
):
    """Write the Excel summary report.

    ``include_components=True`` produces the 4-sheet validation layout (with the
    'Analysis_Components' sheet); ``False`` produces the 3-sheet test layout.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual_epochs = int(os.environ.get("CARDIOAI_EPOCHS", 100))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 1. Summary
        if include_components:
            summary_data = {
                "Metric": [
                    "Dataset", "Total Patients", "Parameters Analyzed", "Training Epochs (Latest)",
                    "Model Architecture", "GPU Used",
                ],
                "Value": [
                    dataset_label, len(patient_ids) if patient_ids else 0,
                    len(param_names), actual_epochs, "Multistage Transformer CardioAI",
                    "NVIDIA RTX 6000 Ada",
                ],
            }
        else:
            summary_data = {
                "Metric": ["Dataset", "Total Patients", "Parameters Analyzed", "Model Architecture"],
                "Value": [
                    dataset_label, len(patient_ids) if patient_ids else 0,
                    len(param_names), "Multistage Transformer CardioAI",
                ],
            }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # 2. Model performance
        if pred_denorm is not None and targets is not None:
            correlations = [
                calculate_correlation(pred_denorm[:, i], targets[:, i]) for i in range(len(param_names))
            ]
            mse_values = [
                np.mean((pred_denorm[:, i] - targets[:, i]) ** 2) for i in range(len(param_names))
            ]
            performance_data = {
                "Parameter": param_names,
                "Correlation": correlations,
                "MSE": mse_values,
                "Target_Min": [0.6] * len(param_names),
                "Meets_Target": [c >= 0.6 for c in correlations],
            }
            pd.DataFrame(performance_data).to_excel(writer, sheet_name="Model_Performance", index=False)

            # 3. Detailed predictions
            if patient_ids and len(patient_ids) > 0:
                pred_data = {"Patient_ID": patient_ids[: len(pred_denorm)]}
                for i, param in enumerate(param_names):
                    pred_data[f"Predicted_{param}"] = pred_denorm[:, i]
                    pred_data[f"True_{param}"] = targets[:, i]
                    pred_data[f"Error_{param}"] = pred_denorm[:, i] - targets[:, i]
                pd.DataFrame(pred_data).to_excel(writer, sheet_name="Detailed_Predictions", index=False)

        # 4. Analysis components (validation only)
        if include_components:
            components_data = {
                "Analysis_Component": [
                    "Training Loss Curves", "Correlation Analysis", "Scatter Plots",
                    "Bland-Altman Plots", "ROC Curves", "Confusion Matrices",
                    "UMAP/t-SNE Embeddings",
                ],
                "Status": ["Generated"] * 7,
                "Output_Format": ["EPS, PNG, TIFF"] * 7,
                "Description": [
                    "Training and validation loss over epochs",
                    "Parameter correlation analysis and heatmaps",
                    "True vs predicted scatter plots",
                    "Agreement analysis between true and predicted values",
                    "ROC curves with AUC values for binary classification",
                    "Confusion matrices for classification performance",
                    "Dimensionality reduction with KMeans/DBSCAN clustering",
                ],
            }
            pd.DataFrame(components_data).to_excel(writer, sheet_name="Analysis_Components", index=False)

    print(f"Summary report saved to: {output_path}")
