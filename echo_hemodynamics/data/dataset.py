"""CardioAIDataset: PyTorch Dataset for multi-view ultrasound tensors with winsorized labels."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .loaders import winsorize_parameter

warnings.filterwarnings("ignore")


class CardioAIDataset(Dataset):
    """Multi-view cardiac ultrasound dataset with winsorized normalization."""

    def __init__(self, tensor_dir, excel_file, max_frames=32, subset_size=None, cache_tensors=True):
        self.tensor_dir = Path(tensor_dir)
        self.max_frames = max_frames
        self.cache_tensors = cache_tensors
        self.views = ["FC", "TC", "SA", "LA"]

        print(f"Loading CardioAI dataset from {tensor_dir}")
        print(f"Excel file: {excel_file}")
        print(f"Max frames per view: {max_frames}")

        try:
            df = pd.read_excel(excel_file)
            print(f"Loaded Excel file with {len(df)} rows")

            def clean_numeric(x):
                if isinstance(x, str):
                    cleaned = "".join(c for c in x if c.isprintable()).strip()
                    try:
                        return float(cleaned)
                    except Exception:
                        return np.nan
                return x

            for col in ["meanPAP", "PCWP"]:
                if col in df.columns:
                    df[col] = df[col].apply(clean_numeric)

            df_clean = df.dropna()
            print(
                f"After cleaning: {len(df_clean)} rows "
                f"(removed {len(df) - len(df_clean)} with missing data)"
            )
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise

        patient_ids = df_clean.iloc[:, 0].astype(str).tolist()
        labels = df_clean.iloc[:, 1:10].values.astype(np.float32)

        print(f"Found {len(patient_ids)} patient IDs")
        print(f"Label shape: {labels.shape}")

        parameter_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
        winsorization_percentiles = {
            "RAP": (5, 95), "SPAP": (5, 95), "dpap": (5, 95),
            "meanPAP": (5, 95), "PCWP": (5, 95),
            "CO": (2, 98), "CI": (2, 98),
            "SVRI": (1, 99), "PVR": (1, 99),
        }

        print("\nWinsorization:")
        print(f"{'Parameter':<10} {'Lower%':<8} {'Upper%':<8} {'#Lower':<8} {'#Upper':<8} {'Range'}")
        print("-" * 70)

        self.winsorized_param_mins = []
        self.winsorized_param_maxs = []
        self.log_transform_indices = [7, 8]
        self.winsorized_log_mins = []
        self.winsorized_log_maxs = []

        for i, param_name in enumerate(parameter_names):
            lower_pct, upper_pct = winsorization_percentiles[param_name]
            labels[:, i], n_lower, n_upper, lower_bound, upper_bound = winsorize_parameter(
                labels[:, i],
                lower_percentile=lower_pct,
                upper_percentile=upper_pct,
            )

            if i in self.log_transform_indices:
                self.winsorized_log_mins.append(np.log(lower_bound + 1))
                self.winsorized_log_maxs.append(np.log(upper_bound + 1))
                self.winsorized_param_mins.append(lower_bound)
                self.winsorized_param_maxs.append(upper_bound)
            else:
                self.winsorized_param_mins.append(lower_bound)
                self.winsorized_param_maxs.append(upper_bound)

            print(
                f"{param_name:<10} {lower_pct:<8} {upper_pct:<8} {n_lower:<8} {n_upper:<8} "
                f"[{lower_bound:.2f}, {upper_bound:.2f}]"
            )

        self.winsorized_param_mins = np.array(self.winsorized_param_mins, dtype=np.float32)
        self.winsorized_param_maxs = np.array(self.winsorized_param_maxs, dtype=np.float32)
        self.winsorized_log_mins = np.array(self.winsorized_log_mins, dtype=np.float32)
        self.winsorized_log_maxs = np.array(self.winsorized_log_maxs, dtype=np.float32)

        print("Winsorization complete")

        print("Checking for complete tensor sets...")
        self.data = []
        tensor_cache = {} if cache_tensors else None

        for idx, patient_id in enumerate(patient_ids):
            tensor_files = [self.tensor_dir / f"{patient_id}_{view}.pt" for view in self.views]

            if all(tensor_file.exists() for tensor_file in tensor_files):
                patient_labels = labels[idx]

                if not np.any(np.isnan(patient_labels)):
                    if cache_tensors:
                        try:
                            cached_tensors = []
                            for tensor_file in tensor_files:
                                tensor = torch.load(tensor_file, map_location="cpu")

                                if tensor.dim() == 4 and tensor.shape[0] == 1:
                                    tensor = tensor.squeeze(0)
                                elif tensor.dim() == 2:
                                    tensor = tensor.unsqueeze(0)

                                if tensor.shape[0] != max_frames:
                                    print(f"WARNING: Unexpected frame count {tensor.shape[0]} for {tensor_file}")
                                    if tensor.shape[0] > max_frames:
                                        frame_indices = np.linspace(0, tensor.shape[0] - 1, max_frames, dtype=int)
                                        tensor = tensor[frame_indices]
                                    else:
                                        padding_needed = max_frames - tensor.shape[0]
                                        last_frame = tensor[-1:].repeat(padding_needed, 1, 1)
                                        tensor = torch.cat([tensor, last_frame], dim=0)

                                cached_tensors.append(tensor)

                            tensor_cache[patient_id] = cached_tensors

                            self.data.append({
                                "patient_id": patient_id,
                                "labels": patient_labels,
                                "tensor_files": tensor_files,
                            })
                        except Exception as e:
                            print(f"Error loading tensors for patient {patient_id}: {e}")
                            continue
                    else:
                        self.data.append({
                            "patient_id": patient_id,
                            "labels": patient_labels,
                            "tensor_files": tensor_files,
                        })
                else:
                    print(f"Skipping patient {patient_id}: Invalid labels (contains NaN)")
            else:
                missing_files = [f for f in tensor_files if not f.exists()]
                if len(missing_files) <= 2:
                    print(f"Skipping patient {patient_id}: Missing {len(missing_files)} tensor files")

        print(f"Found {len(self.data)} patients with complete data")

        if subset_size is not None and subset_size < len(self.data):
            print(f"Using subset of {subset_size} patients")
            np.random.seed(42)
            indices = np.random.choice(len(self.data), subset_size, replace=False)
            self.data = [self.data[i] for i in sorted(indices)]

        self.tensor_cache = tensor_cache
        print(f"Final dataset size: {len(self.data)} patients")

        if len(self.data) == 0:
            raise ValueError("No valid patients found! Check tensor directory and Excel file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        patient_id = sample["patient_id"]
        labels = torch.tensor(sample["labels"], dtype=torch.float32)

        if self.cache_tensors and patient_id in self.tensor_cache:
            views = self.tensor_cache[patient_id]
        else:
            views = []
            for tensor_file in sample["tensor_files"]:
                try:
                    tensor = torch.load(tensor_file, map_location="cpu")

                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    elif tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0)

                    if tensor.shape[0] != self.max_frames:
                        print(f"WARNING: Unexpected frame count {tensor.shape[0]} for {tensor_file}")
                        if tensor.shape[0] > self.max_frames:
                            frame_indices = np.linspace(0, tensor.shape[0] - 1, self.max_frames, dtype=int)
                            tensor = tensor[frame_indices]
                        else:
                            padding_needed = self.max_frames - tensor.shape[0]
                            last_frame = tensor[-1:].repeat(padding_needed, 1, 1)
                            tensor = torch.cat([tensor, last_frame], dim=0)

                    views.append(tensor)
                except Exception as e:
                    print(f"Error loading tensor {tensor_file}: {e}")
                    views.append(torch.zeros(self.max_frames, 224, 224))

        return views, labels, patient_id

    def get_patient_by_id(self, patient_id):
        for idx, sample in enumerate(self.data):
            if sample["patient_id"] == patient_id:
                return self.__getitem__(idx)
        raise ValueError(f"Patient {patient_id} not found in dataset")

    def get_normalization_parameters(self):
        return {
            "param_mins": self.winsorized_param_mins,
            "param_maxs": self.winsorized_param_maxs,
            "log_mins": self.winsorized_log_mins,
            "log_maxs": self.winsorized_log_maxs,
            "log_transform_indices": self.log_transform_indices,
        }

    def get_dataset_statistics(self):
        if not self.data:
            return {}

        all_labels = np.array([sample["labels"] for sample in self.data])
        parameter_names = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]

        stats = {
            "total_patients": len(self.data),
            "parameter_names": parameter_names,
            "label_statistics": {},
        }

        for i, param_name in enumerate(parameter_names):
            param_values = all_labels[:, i]
            stats["label_statistics"][param_name] = {
                "mean": float(np.mean(param_values)),
                "std": float(np.std(param_values)),
                "min": float(np.min(param_values)),
                "max": float(np.max(param_values)),
                "median": float(np.median(param_values)),
            }

        return stats

    def print_dataset_info(self):
        stats = self.get_dataset_statistics()

        print(
            f"\nDataset: {stats['total_patients']} patients, {len(self.views)} views, "
            f"{self.max_frames} frames, {len(stats['parameter_names'])} parameters"
        )

        print(f"{'Parameter':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 45)
        for param_name in stats["parameter_names"]:
            s = stats["label_statistics"][param_name]
            print(
                f"{param_name:<10} {s['mean']:<8.2f} {s['std']:<8.2f} "
                f"{s['min']:<8.2f} {s['max']:<8.2f}"
            )
