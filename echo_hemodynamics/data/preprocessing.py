"""DICOM-to-tensor conversion with fan-region extraction and artifact removal."""

import warnings

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure

warnings.filterwarnings("ignore")

try:
    import pydicom
except ImportError:
    pydicom = None


def detect_fan_region_extreme_clean(gray_frame, visualize=False):
    """Extract ultrasound fan region from grayscale frame, removing all artifacts via
    aggressive cropping, morphological cleaning, and convex hull masking."""
    height, width = gray_frame.shape

    top_crop = int(height * 0.22)
    bottom_crop = int(height * 0.70)
    left_crop = int(width * 0.12)
    right_crop = int(width * 0.88)

    cropped_frame = gray_frame[top_crop:bottom_crop, left_crop:right_crop].copy()

    if visualize:
        print(f"  Original size: {width} x {height}")
        print(f"  Cropped size: {cropped_frame.shape[1]} x {cropped_frame.shape[0]}")
        print(
            f"  Removed: top {top_crop}px, bottom {height - bottom_crop}px, "
            f"left {left_crop}px, right {width - right_crop}px"
        )

    threshold_value = np.percentile(cropped_frame, 60)
    binary = cropped_frame > threshold_value

    binary = ndimage.binary_opening(binary, structure=np.ones((15, 15)))
    binary = ndimage.binary_closing(binary, structure=np.ones((25, 25)))
    binary = ndimage.binary_opening(binary, structure=np.ones((12, 12)))

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

    coords = largest_region.coords

    if len(coords) > 3:
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            clean_mask = np.zeros_like(binary, dtype=np.uint8)
            hull_points_2d = hull_points[:, [1, 0]].astype(np.int32)
            cv2.fillPoly(clean_mask, [hull_points_2d], 1)
            clean_mask = clean_mask.astype(bool)
        except Exception:
            clean_mask = labeled == largest_region.label
    else:
        clean_mask = labeled == largest_region.label

    clean_mask = ndimage.binary_erosion(clean_mask, structure=np.ones((5, 5)))
    y_coords, x_coords = np.where(clean_mask)

    if len(y_coords) == 0:
        if visualize:
            print("  WARNING: No valid region after cleaning")
        return cropped_frame

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

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

    frame_indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)

    processed_frames = []
    for idx, frame_idx in enumerate(frame_indices):
        frame = pixel_array[frame_idx]
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame

        visualize_this = visualize and idx == 15
        fan_region = detect_fan_region_extreme_clean(gray_frame, visualize=visualize_this)

        resized = cv2.resize(fan_region, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        processed_frames.append(normalized)

    return torch.tensor(np.array(processed_frames), dtype=torch.float32)
