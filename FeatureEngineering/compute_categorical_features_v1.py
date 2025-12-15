# categorical_features.py
# Calculates categorical/temporal features for one-hot encoded time series fields
import numpy as np


def compute_categorical_features_v1(one_hot_series):
    """
    Compute categorical/temporal features for a single one-hot encoded (binary) time series column.
    Args:
        one_hot_series (np.ndarray or list): 1D array-like, binary (0/1) values.
    Returns:
        list: Feature vector for the series.
    """
    import numpy as np
    from scipy.stats import entropy
    if not isinstance(one_hot_series, np.ndarray):
        arr = np.array(one_hot_series, dtype=np.float64)
    else:
        arr = one_hot_series.astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        return [0.0] * 7  # fallback for invalid input
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return [0.0] * 7
    # Basic stats
    mean = np.mean(arr)
    std = np.std(arr)
    total_count = np.sum(arr)
    # Longest consecutive streak of 1s
    max_streak = 0
    curr_streak = 0
    for v in arr:
        if v == 1:
            curr_streak += 1
            if curr_streak > max_streak:
                max_streak = curr_streak
        else:
            curr_streak = 0
    # First and last activation
    active_indices = np.where(arr == 1)[0]
    first_active = float(active_indices[0]) if active_indices.size > 0 else -1.0
    last_active = float(active_indices[-1]) if active_indices.size > 0 else -1.0
    # Number of transitions (0->1 or 1->0)
    transitions = float(np.sum(arr[1:] != arr[:-1])) if len(arr) > 1 else 0.0
    # Entropy
    ent = 0.0
    try:
        hist, _ = np.histogram(arr, bins=2, density=True)
        ent = entropy(hist)
    except Exception:
        ent = 0.0
    if ent is None:
        ent = 0.0
    return [mean, std, total_count, max_streak, first_active, last_active, transitions, ent]
