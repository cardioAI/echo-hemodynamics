"""Smooth sigmoid mask suppressing the transducer apex region in heatmap overlays."""

import numpy as np


def create_apex_mask(image_shape, apex_height=70, transition_width=50):
    """Sigmoid mask along the vertical axis, fading from 0 (apex) to 1 (far field)."""
    height, width = image_shape

    y_coords = np.arange(height)
    transition = 1.0 / (1.0 + np.exp(-(y_coords - apex_height) / (transition_width / 4)))
    mask = np.tile(transition.reshape(-1, 1), (1, width)).astype(np.float32)
    return mask
