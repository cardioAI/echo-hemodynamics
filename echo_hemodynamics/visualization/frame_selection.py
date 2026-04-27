"""Helpers for selecting top-N attention-weighted frames from rollout scores."""

import numpy as np
import torch


def normalize_scores(scores):
    """Coerce a rollout score (np.ndarray, tensor, possibly 2D) to a 1D numpy array."""
    if isinstance(scores, np.ndarray) and len(scores.shape) > 1:
        return scores[0] if scores.shape[0] > 0 else scores.flatten()
    if torch.is_tensor(scores):
        scores_np = scores.cpu().numpy()
        if len(scores_np.shape) > 1:
            return scores_np[0] if scores_np.shape[0] > 0 else scores_np.flatten()
        return scores_np
    return np.array(scores).flatten()


def select_top_frames(scores, n_frames):
    """Return indices of the top ``n_frames`` highest-scoring frames (or all if fewer)."""
    scores_1d = normalize_scores(scores).flatten()
    if len(scores_1d) < n_frames:
        return np.arange(len(scores_1d))
    return np.argsort(scores_1d)[-n_frames:]
