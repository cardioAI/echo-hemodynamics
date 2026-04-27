"""Run a trained ProgressiveCardioAI model on a Subset and collect predictions."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data import CardioAIDataset


def generate_model_predictions(
    model, device, indices, tensor_dir, excel_file, training_frames=32, batch_size=2,
):
    """Run inference and return (predictions, targets, patient_ids) as numpy arrays / list.

    ``predictions`` are still in normalized [0, 1] space; the caller denormalizes when needed.
    """
    full_dataset = CardioAIDataset(tensor_dir, excel_file, max_frames=training_frames, subset_size=None)
    print(f"Loaded full dataset with {len(full_dataset)} patients")

    valid_indices = [i for i in indices if i < len(full_dataset)]
    dataset = Subset(full_dataset, valid_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Running inference on {len(dataset)} patients")

    all_predictions = []
    all_targets = []
    all_patient_ids = []

    model.eval()
    with torch.no_grad():
        for views, targets, patient_ids in dataloader:
            fixed_views = []
            for view in views:
                if len(view.shape) == 5:
                    view = view.squeeze(2)
                fixed_views.append(view.to(device))
            targets = targets.to(device)

            try:
                predictions = model(fixed_views, return_aux=False)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_patient_ids.extend(patient_ids)
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                continue

    if not all_predictions:
        return None, None, []

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    print(f"Generated predictions for {len(all_patient_ids)} patients")
    return predictions, targets, all_patient_ids


def denormalize(model, device, predictions):
    """Convert normalized predictions back to original units via the model's helper."""
    pred_torch = torch.tensor(predictions, dtype=torch.float32).to(device)
    return model.denormalize_predictions(pred_torch).cpu().numpy()
