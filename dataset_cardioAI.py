#!/usr/bin/env python3
"""
CardioAI dataset module: DICOM-to-tensor conversion and PyTorch data loading
for multi-view cardiac ultrasound tensors.
"""

import os
import torch
import pandas as pd
import numpy as np
import cv2
import shutil
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure
import warnings
warnings.filterwarnings("ignore")

try:
    import pydicom
except ImportError:
    pydicom = None


def detect_fan_region_extreme_clean(gray_frame, visualize=False):
    """Extract ultrasound fan region from grayscale frame, removing all artifacts via
    aggressive cropping, morphological cleaning, and convex hull masking."""
    height, width = gray_frame.shape

    # Crop to remove text, ECG, and side labels
    top_crop = int(height * 0.22)
    bottom_crop = int(height * 0.70)
    left_crop = int(width * 0.12)
    right_crop = int(width * 0.88)

    cropped_frame = gray_frame[top_crop:bottom_crop, left_crop:right_crop].copy()

    if visualize:
        print(f"  Original size: {width} x {height}")
        print(f"  Cropped size: {cropped_frame.shape[1]} x {cropped_frame.shape[0]}")
        print(f"  Removed: top {top_crop}px, bottom {height-bottom_crop}px, left {left_crop}px, right {width-right_crop}px")

    threshold_value = np.percentile(cropped_frame, 60)
    binary = cropped_frame > threshold_value

    # Morphological cleaning
    binary = ndimage.binary_opening(binary, structure=np.ones((15, 15)))
    binary = ndimage.binary_closing(binary, structure=np.ones((25, 25)))
    binary = ndimage.binary_opening(binary, structure=np.ones((12, 12)))

    # Largest connected component (main ultrasound fan)
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)

    if len(regions) == 0:
        if visualize:
            print("  WARNING: No regions detected, using entire cropped frame")
        return cropped_frame

    largest_region = max(regions, key=lambda r: r.area)

    if visualize:
        print(f"  Detected {len(regions)} regions")
        print(f"  Largest region area: {largest_region.area}")

    # Convex hull masking
    coords = largest_region.coords

    if len(coords) > 3:
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            clean_mask = np.zeros_like(binary, dtype=np.uint8)
            hull_points_2d = hull_points[:, [1, 0]].astype(np.int32)
            cv2.fillPoly(clean_mask, [hull_points_2d], 1)
            clean_mask = clean_mask.astype(bool)
        except:
            clean_mask = labeled == largest_region.label
    else:
        clean_mask = labeled == largest_region.label

    # Edge erosion
    clean_mask = ndimage.binary_erosion(clean_mask, structure=np.ones((5, 5)))
    y_coords, x_coords = np.where(clean_mask)

    if len(y_coords) == 0:
        if visualize:
            print("  WARNING: No valid region after cleaning")
        return cropped_frame

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # Minimal padding
    pad = 2
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(cropped_frame.shape[0], y_max + pad)
    x_max = min(cropped_frame.shape[1], x_max + pad)

    fan_region = cropped_frame[y_min:y_max, x_min:x_max].copy()
    region_mask = clean_mask[y_min:y_max, x_min:x_max]
    fan_region[~region_mask] = 0

    if visualize:
        print(f"  Fan region size: {fan_region.shape[1]} x {fan_region.shape[0]}")
        print(f"  Non-zero pixels: {np.sum(region_mask)}")

    return fan_region


def convert_dicom_to_tensor(dicom_path, target_frames=32, target_size=(224, 224), visualize=False):
    """Convert DICOM ultrasound video to tensor (target_frames, H, W) with artifact removal."""
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM conversion. Install with: pip install pydicom")

    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array

    # Handle different DICOM formats
    if len(pixel_array.shape) == 4:
        num_frames = pixel_array.shape[0]
    elif len(pixel_array.shape) == 3:
        if pixel_array.shape[2] in [3, 4]:
            num_frames = 1
            pixel_array = pixel_array[np.newaxis, ...]
        else:
            num_frames = pixel_array.shape[0]
    else:
        raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")

    # Select evenly-spaced frames
    frame_indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)

    # Process each frame
    processed_frames = []

    for idx, frame_idx in enumerate(frame_indices):
        frame = pixel_array[frame_idx]

        # Convert to grayscale if RGB
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame

        # Detect and extract fan region with extreme cleaning
        visualize_this = visualize and idx == 15
        fan_region = detect_fan_region_extreme_clean(gray_frame, visualize=visualize_this)

        # Resize to target size
        resized = cv2.resize(fan_region, target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        processed_frames.append(normalized)

    # Convert to PyTorch tensor
    tensor = torch.tensor(np.array(processed_frames), dtype=torch.float32)

    return tensor


def batch_convert(dcm_dir, output_dir, views=['FC', 'TC', 'SA', 'LA']):
    """Batch convert DICOM files to PT tensors. Returns (total, successful, failed)."""
    dcm_dir = Path(dcm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all DICOM files
    dcm_files = list(dcm_dir.glob("*.dcm"))

    # Group by patient and view
    patient_views = {}
    for dcm_file in dcm_files:
        parts = dcm_file.stem.split('_')
        if len(parts) >= 2:
            patient_id = parts[0]
            view = parts[1]
            if view in views:
                if patient_id not in patient_views:
                    patient_views[patient_id] = {}
                patient_views[patient_id][view] = dcm_file

    print(f"\nFound {len(patient_views)} patients")
    print(f"Expected views per patient: {views}")

    # Convert each patient
    total_files = 0
    successful = 0

    for patient_idx, (patient_id, view_files) in enumerate(patient_views.items(), 1):
        print(f"\n[{patient_idx}/{len(patient_views)}] Processing {patient_id}...")

        for view in views:
            if view not in view_files:
                print(f"  WARNING: Missing view {view}")
                continue

            dcm_file = view_files[view]
            output_file = output_dir / f"{patient_id}_{view}.pt"

            try:
                tensor = convert_dicom_to_tensor(dcm_file, target_frames=32, target_size=(224, 224))
                torch.save(tensor, output_file)
                total_files += 1
                successful += 1
                print(f"  {view}: {tensor.shape} -> {output_file.name}")

            except Exception as e:
                print(f"  ERROR processing {view}: {e}")
                total_files += 1

    print(f"\n{'='*80}")
    print(f"Conversion complete:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total_files - successful}")
    print(f"{'='*80}")

    return total_files, successful, total_files - successful


def reconvert_external_patients():
    """Re-convert External patients with artifact removal and update All_PT folder."""
    print("="*80)
    print("EXTREME-CLEAN BATCH RE-CONVERSION: REMOVE ALL ARTIFACTS")
    print("="*80)

    # Paths
    dcm_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\External_DCM"
    external_pt_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\External_PT")
    all_pt_dir = Path(r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT")
    claude_dir = Path(r"D:\GoogleDrive\Codes\CardioAI\CardioAI\EchoCath_cardioAI\Claude")

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(rf"E:\dataset_cardioAI\EchoCath_cardioAI\External_PT_backup_{timestamp}")

    print(f"\n1. Creating backup of current External_PT...")
    print(f"   From: {external_pt_dir}")
    print(f"   To:   {backup_dir}")

    if external_pt_dir.exists():
        shutil.copytree(external_pt_dir, backup_dir)
        pt_files_backup = list(backup_dir.glob("*.pt"))
        print(f"   Backup complete: {len(pt_files_backup)} files backed up")

        print(f"\n2. Clearing External_PT directory...")
        for pt_file in external_pt_dir.glob("*.pt"):
            pt_file.unlink()
        print(f"   External_PT directory cleared")
    else:
        external_pt_dir.mkdir(parents=True, exist_ok=True)
        print(f"   External_PT directory created (no backup needed)")

    print(f"\n3. Starting EXTREME-CLEAN batch conversion...")
    print(f"   Input:  {dcm_dir}")
    print(f"   Output: {external_pt_dir}")
    print(f"   Expected: 73 patients x 4 views = 292 files")
    print(f"   Extreme cleaning features:")
    print(f"     - Remove top 22% (all text and labels)")
    print(f"     - Remove bottom 30% (all ECG and markers)")
    print(f"     - Remove left/right 12% (all side labels)")
    print(f"     - Aggressive morphological cleaning (15x15, 25x25, 12x12)")
    print(f"     - Higher threshold (60th percentile)")
    print(f"     - Convex hull masking for smoothest boundary")
    print(f"     - Edge erosion (5x5) to remove residual artifacts")
    print(f"     - Extract ONLY pure ultrasound fan (zero out everything else)\n")

    # Run batch conversion
    batch_convert(dcm_dir, external_pt_dir, views=['FC', 'TC', 'SA', 'LA'])

    # Verify results
    print(f"\n4. Verifying conversion results...")
    pt_files = list(external_pt_dir.glob("*.pt"))
    print(f"   Total PT files created: {len(pt_files)}")
    print(f"   Expected: 292 files")

    if len(pt_files) == 292:
        print(f"   SUCCESS: All files converted!")
    else:
        print(f"   WARNING: Expected 292 files, got {len(pt_files)}")

    # Update All_PT folder
    print(f"\n5. Updating All_PT folder with extreme-clean versions...")
    print(f"   Removing old External patient files from All_PT...")

    external_excel = claude_dir / "External.xlsx"
    external_df = pd.read_excel(external_excel)

    views = ['FC', 'TC', 'SA', 'LA']
    removed_count = 0
    copied_count = 0

    # Remove old External files from All_PT
    for idx, row in external_df.iterrows():
        patient_id = row['E Number']
        for view in views:
            old_file = all_pt_dir / f"{patient_id}_{view}.pt"
            if old_file.exists():
                old_file.unlink()
                removed_count += 1

    print(f"   Removed {removed_count} old External files from All_PT")

    # Copy new extreme-clean files to All_PT
    print(f"   Copying new extreme-clean files to All_PT...")
    for idx, row in external_df.iterrows():
        patient_id = row['E Number']
        for view in views:
            src = external_pt_dir / f"{patient_id}_{view}.pt"
            if src.exists():
                dst = all_pt_dir / f"{patient_id}_{view}.pt"
                shutil.copy2(src, dst)
                copied_count += 1

    print(f"   Copied {copied_count} new extreme-clean files to All_PT")

    # Verify All_PT
    all_pt_files = list(all_pt_dir.glob("*.pt"))
    print(f"\n6. Verifying All_PT folder...")
    print(f"   Total PT files in All_PT: {len(all_pt_files)}")
    print(f"   Expected: 308 x 4 = 1232 files")

    if len(all_pt_files) == 1232:
        print(f"   SUCCESS: All_PT complete!")
    else:
        print(f"   WARNING: Expected 1232 files, got {len(all_pt_files)}")

    print("\n" + "="*80)
    print("EXTREME-CLEAN BATCH RE-CONVERSION COMPLETE")
    print("="*80)
    print(f"Backup location: {backup_dir}")
    print(f"New conversion:  {external_pt_dir}")
    print(f"Updated:         {all_pt_dir}")
    print("\nAll 73 External patients re-converted with EXTREME-CLEAN method")
    print("="*80)


def calculate_correlation(pred, true):
    """Pearson correlation between pred and true arrays."""
    if len(pred) != len(true):
        return 0.0

    pred = np.array(pred).flatten()
    true = np.array(true).flatten()

    # Remove any NaN values
    valid_mask = ~(np.isnan(pred) | np.isnan(true))
    if not np.any(valid_mask):
        return 0.0

    pred = pred[valid_mask]
    true = true[valid_mask]

    if len(pred) < 2:
        return 0.0

    # Check for zero variance
    if np.var(pred) == 0 or np.var(true) == 0:
        return 0.0

    try:
        correlation = np.corrcoef(pred, true)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0


def winsorize_parameter(values, lower_percentile=5, upper_percentile=95):
    """Clip values to given percentile bounds. Returns (clipped, n_lower, n_upper, lo, hi)."""
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)

    # Count outliers for reporting
    n_lower_outliers = np.sum(values < lower_bound)
    n_upper_outliers = np.sum(values > upper_bound)

    # Clip values to boundaries
    winsorized = np.clip(values, lower_bound, upper_bound)

    return winsorized, n_lower_outliers, n_upper_outliers, lower_bound, upper_bound


class CardioAIDataset(Dataset):
    """PyTorch Dataset for multi-view cardiac ultrasound tensors with winsorized normalization."""
    
    def __init__(self, tensor_dir, excel_file, max_frames=32, subset_size=None, cache_tensors=True):
        """
        Args:
            tensor_dir: Directory containing tensor files
            excel_file: Excel file with labels
            max_frames: Maximum number of frames to use per view
            subset_size: If specified, only use this many samples
            cache_tensors: Whether to cache tensors in memory
        """
        self.tensor_dir = Path(tensor_dir)
        self.max_frames = max_frames
        self.cache_tensors = cache_tensors
        self.views = ['FC', 'TC', 'SA', 'LA']
        
        print(f"Loading CardioAI dataset from {tensor_dir}")
        print(f"Excel file: {excel_file}")
        print(f"Max frames per view: {max_frames}")
        
        # Load labels from Excel with proper cleaning
        try:
            df = pd.read_excel(excel_file)
            print(f"Loaded Excel file with {len(df)} rows")
            
            # Clean non-printable characters in meanPAP and PCWP columns
            def clean_numeric(x):
                if isinstance(x, str):
                    cleaned = ''.join(c for c in x if c.isprintable()).strip()
                    try:
                        return float(cleaned)
                    except:
                        return np.nan
                return x
            
            # Apply cleaning to columns that might have issues
            for col in ['meanPAP', 'PCWP']:
                if col in df.columns:
                    df[col] = df[col].apply(clean_numeric)
            
            # Remove any rows with NaN values
            df_clean = df.dropna()
            print(f"After cleaning: {len(df_clean)} rows (removed {len(df) - len(df_clean)} with missing data)")
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
        
        # Extract patient IDs and labels from cleaned data
        patient_ids = df_clean.iloc[:, 0].astype(str).tolist()
        labels = df_clean.iloc[:, 1:10].values.astype(np.float32)  # 9 hemodynamic parameters

        print(f"Found {len(patient_ids)} patient IDs")
        print(f"Label shape: {labels.shape}")

        parameter_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
        winsorization_percentiles = {
            'RAP': (5, 95), 'SPAP': (5, 95), 'dpap': (5, 95),
            'meanPAP': (5, 95), 'PCWP': (5, 95),
            'CO': (2, 98), 'CI': (2, 98),
            'SVRI': (1, 99), 'PVR': (1, 99)
        }

        print(f"\nWinsorization:")
        print(f"{'Parameter':<10} {'Lower%':<8} {'Upper%':<8} {'#Lower':<8} {'#Upper':<8} {'Range'}")
        print("-" * 70)

        self.winsorized_param_mins = []
        self.winsorized_param_maxs = []
        self.log_transform_indices = [7, 8]  # SVRI (index 7) and PVR (index 8) indices
        self.winsorized_log_mins = []
        self.winsorized_log_maxs = []

        for i, param_name in enumerate(parameter_names):
            lower_pct, upper_pct = winsorization_percentiles[param_name]
            original_values = labels[:, i].copy()

            # Apply winsorization
            labels[:, i], n_lower, n_upper, lower_bound, upper_bound = winsorize_parameter(
                labels[:, i],
                lower_percentile=lower_pct,
                upper_percentile=upper_pct
            )

            # Store winsorized bounds for normalization
            if i in self.log_transform_indices:
                # For SVRI and PVR: store log-transformed bounds
                self.winsorized_log_mins.append(np.log(lower_bound + 1))
                self.winsorized_log_maxs.append(np.log(upper_bound + 1))
                # Also store original bounds for reference
                self.winsorized_param_mins.append(lower_bound)
                self.winsorized_param_maxs.append(upper_bound)
            else:
                # For other parameters: store actual bounds
                self.winsorized_param_mins.append(lower_bound)
                self.winsorized_param_maxs.append(upper_bound)

            # Report winsorization results
            print(f"{param_name:<10} {lower_pct:<8} {upper_pct:<8} {n_lower:<8} {n_upper:<8} "
                  f"[{lower_bound:.2f}, {upper_bound:.2f}]")

        # Convert to numpy arrays for easy access
        self.winsorized_param_mins = np.array(self.winsorized_param_mins, dtype=np.float32)
        self.winsorized_param_maxs = np.array(self.winsorized_param_maxs, dtype=np.float32)
        self.winsorized_log_mins = np.array(self.winsorized_log_mins, dtype=np.float32)
        self.winsorized_log_maxs = np.array(self.winsorized_log_maxs, dtype=np.float32)

        print(f"Winsorization complete")
        
        # Find patients with all required tensor files
        print("Checking for complete tensor sets...")
        self.data = []
        tensor_cache = {} if cache_tensors else None
        
        for idx, patient_id in enumerate(patient_ids):
            tensor_files = [self.tensor_dir / f"{patient_id}_{view}.pt" for view in self.views]
            
            # Check if all tensor files exist
            if all(tensor_file.exists() for tensor_file in tensor_files):
                patient_labels = labels[idx]
                
                # Validate labels (check for NaN values)
                if not np.any(np.isnan(patient_labels)):
                    if cache_tensors:
                        # Load and cache tensors
                        try:
                            cached_tensors = []
                            for tensor_file in tensor_files:
                                tensor = torch.load(tensor_file, map_location='cpu')
                                
                                # Ensure tensor is in correct format (frames, height, width)
                                if tensor.dim() == 4 and tensor.shape[0] == 1:
                                    tensor = tensor.squeeze(0)  # Remove batch dimension
                                elif tensor.dim() == 2:
                                    # If 2D, assume it's (height, width) and add frame dimension
                                    tensor = tensor.unsqueeze(0)
                                
                                # Handle frame count (all tensors should now have exactly 32 frames after pre-processing)
                                if tensor.shape[0] != max_frames:
                                    # This should not happen after dataset pre-processing
                                    print(f"WARNING: Unexpected frame count {tensor.shape[0]} for {tensor_file}")
                                    if tensor.shape[0] > max_frames:
                                        # Select evenly distributed frames
                                        frame_indices = np.linspace(0, tensor.shape[0]-1, max_frames, dtype=int)
                                        tensor = tensor[frame_indices]
                                    else:
                                        # Fallback: repeat last frame (should not occur)
                                        padding_needed = max_frames - tensor.shape[0]
                                        last_frame = tensor[-1:].repeat(padding_needed, 1, 1)
                                        tensor = torch.cat([tensor, last_frame], dim=0)
                                
                                cached_tensors.append(tensor)
                            
                            tensor_cache[patient_id] = cached_tensors
                            
                            self.data.append({
                                'patient_id': patient_id,
                                'labels': patient_labels,
                                'tensor_files': tensor_files
                            })
                            
                        except Exception as e:
                            print(f"Error loading tensors for patient {patient_id}: {e}")
                            continue
                    else:
                        # Just store file paths
                        self.data.append({
                            'patient_id': patient_id,
                            'labels': patient_labels,
                            'tensor_files': tensor_files
                        })
                else:
                    print(f"Skipping patient {patient_id}: Invalid labels (contains NaN)")
            else:
                missing_files = [f for f in tensor_files if not f.exists()]
                if len(missing_files) <= 2:  # Only print for first few missing
                    print(f"Skipping patient {patient_id}: Missing {len(missing_files)} tensor files")
        
        print(f"Found {len(self.data)} patients with complete data")
        
        # Apply subset size if specified
        if subset_size is not None and subset_size < len(self.data):
            print(f"Using subset of {subset_size} patients")
            # Use deterministic sampling for reproducibility
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
        """Get a single sample from the dataset"""
        sample = self.data[idx]
        patient_id = sample['patient_id']
        labels = torch.tensor(sample['labels'], dtype=torch.float32)
        
        if self.cache_tensors and patient_id in self.tensor_cache:
            # Use cached tensors
            views = self.tensor_cache[patient_id]
        else:
            # Load tensors on-the-fly
            views = []
            for tensor_file in sample['tensor_files']:
                try:
                    tensor = torch.load(tensor_file, map_location='cpu')
                    
                    # Ensure correct format
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    elif tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0)
                    
                    # Handle frame count (all tensors should now have exactly 32 frames after pre-processing)
                    if tensor.shape[0] != self.max_frames:
                        # This should not happen after dataset pre-processing
                        print(f"WARNING: Unexpected frame count {tensor.shape[0]} for {tensor_file}")
                        if tensor.shape[0] > self.max_frames:
                            frame_indices = np.linspace(0, tensor.shape[0]-1, self.max_frames, dtype=int)
                            tensor = tensor[frame_indices]
                        else:
                            # Fallback: repeat last frame (should not occur)
                            padding_needed = self.max_frames - tensor.shape[0]
                            last_frame = tensor[-1:].repeat(padding_needed, 1, 1)
                            tensor = torch.cat([tensor, last_frame], dim=0)
                    
                    views.append(tensor)
                    
                except Exception as e:
                    print(f"Error loading tensor {tensor_file}: {e}")
                    # Create dummy tensor as fallback
                    views.append(torch.zeros(self.max_frames, 224, 224))
        
        return views, labels, patient_id
    
    def get_patient_by_id(self, patient_id):
        """Get a specific patient's data by ID"""
        for idx, sample in enumerate(self.data):
            if sample['patient_id'] == patient_id:
                return self.__getitem__(idx)
        raise ValueError(f"Patient {patient_id} not found in dataset")

    def get_normalization_parameters(self):
        """Get winsorized normalization parameters for the model

        Returns:
            Dictionary with normalization parameters based on winsorized data ranges
        """
        return {
            'param_mins': self.winsorized_param_mins,
            'param_maxs': self.winsorized_param_maxs,
            'log_mins': self.winsorized_log_mins,
            'log_maxs': self.winsorized_log_maxs,
            'log_transform_indices': self.log_transform_indices
        }
    
    def get_dataset_statistics(self):
        """Get basic statistics about the dataset"""
        if not self.data:
            return {}
        
        all_labels = np.array([sample['labels'] for sample in self.data])

        # Order must match All.xlsx columns
        parameter_names = ['RAP', 'SPAP', 'dpap', 'meanPAP', 'PCWP', 'CO', 'CI', 'SVRI', 'PVR']
        
        stats = {
            'total_patients': len(self.data),
            'parameter_names': parameter_names,
            'label_statistics': {}
        }
        
        for i, param_name in enumerate(parameter_names):
            param_values = all_labels[:, i]
            stats['label_statistics'][param_name] = {
                'mean': float(np.mean(param_values)),
                'std': float(np.std(param_values)),
                'min': float(np.min(param_values)),
                'max': float(np.max(param_values)),
                'median': float(np.median(param_values))
            }
        
        return stats
    
    def print_dataset_info(self):
        """Print dataset summary and parameter statistics."""
        stats = self.get_dataset_statistics()

        print(f"\nDataset: {stats['total_patients']} patients, {len(self.views)} views, "
              f"{self.max_frames} frames, {len(stats['parameter_names'])} parameters")

        print(f"{'Parameter':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 45)
        for param_name in stats['parameter_names']:
            s = stats['label_statistics'][param_name]
            print(f"{param_name:<10} {s['mean']:<8.2f} {s['std']:<8.2f} {s['min']:<8.2f} {s['max']:<8.2f}")


def create_dataloaders(dataset_dir, excel_file, batch_size=2, train_split=1.0,
                      max_frames=32, subset_size=None, num_workers=0):
    """Create train/val dataloaders. Returns (train_loader, val_loader, stats)."""
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = CardioAIDataset(
        tensor_dir=dataset_dir,
        excel_file=excel_file,
        max_frames=max_frames,
        subset_size=subset_size,
        cache_tensors=True  # Cache for better performance
    )
    
    # Print dataset info
    full_dataset.print_dataset_info()
    
    # Split dataset (handle overfitting case with train_split=1.0)
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = max(1, dataset_size - train_size)  # Ensure at least 1 sample for validation
    
    if train_split >= 1.0:
        # Patients 1-235 (indices 0-234) for training, patients 236-308 (indices 235-307) for independent test
        # Use first 235 patients for training, last 20% of training set as validation
        train_indices = list(range(min(235, dataset_size)))  # Patients 1-235
        val_size_cv = len(train_indices) // 5  # ~20% for validation (matches 5-fold CV)
        val_indices = train_indices[-val_size_cv:]  # Last fold as default validation

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        print(f"\nCardioAI split: {len(train_indices)} training (patients 1-235), {len(val_indices)} validation")
    else:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"\nDataset split: {train_size} training, {val_size} validation")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader, full_dataset.get_dataset_statistics()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        # Run DICOM conversion for External patients
        reconvert_external_patients()
    else:
        # Test the dataset
        dataset_dir = r"E:\dataset_cardioAI\EchoCath_cardioAI\All_PT"
        excel_file = "./All.xlsx"

        print("Testing CardioAI Dataset...")

        try:
            # Create dataset
            dataset = CardioAIDataset(
                tensor_dir=dataset_dir,
                excel_file=excel_file,
                max_frames=32,
                subset_size=10  # Small subset for testing
            )

            # Test data loading
            print("\nTesting data loading...")
            views, labels, patient_id = dataset[0]

            print(f"Patient ID: {patient_id}")
            print(f"Number of views: {len(views)}")
            print(f"Labels shape: {labels.shape}")

            for i, view in enumerate(views):
                print(f"View {i} shape: {view.shape}")

            # Test dataloader creation
            print("\nTesting dataloader creation...")
            train_loader, val_loader, stats = create_dataloaders(
                dataset_dir=dataset_dir,
                excel_file=excel_file,
                batch_size=2,
                subset_size=10
            )

            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")

            # Test batch loading
            for views_batch, labels_batch, patient_ids in train_loader:
                print(f"Batch views: {len(views_batch)} views")
                print(f"View 0 shape: {views_batch[0].shape}")
                print(f"Labels shape: {labels_batch.shape}")
                print(f"Patient IDs: {patient_ids}")
                break

            print("\nDataset testing completed successfully!")

        except Exception as e:
            print(f"\nERROR: Dataset testing failed: {e}")
            import traceback
            traceback.print_exc()