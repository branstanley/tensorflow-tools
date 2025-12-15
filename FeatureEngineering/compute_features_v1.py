import numpy as np
from scipy.fftpack import fft
from scipy.signal import correlate
from scipy.stats import skew, kurtosis, entropy
from statsmodels.tsa.stattools import acf

def compute_features_v1(series):
    arr = np.array(series, dtype=np.float64)

    if len(arr) == 0 or np.all(np.isnan(arr)):
        return [0.0] * 21

    # Basic stats
    mean = np.mean(arr)
    std = np.std(arr)

    # Frequency domain
    fft_mag = np.max(np.abs(fft(arr))) if len(arr) > 1 else 0

    # Autocorrelation
    raw_autocorr = np.max(correlate(arr, arr, mode='full')) if len(arr) > 1 else 0

    # ACF values
    acf_vals = 0.0
    try:
        acf_vals = acf(arr, nlags=10, fft=True)
        acf_vals = np.nan_to_num(acf_vals, nan=0.0)
        if len(acf_vals) < 11:
            acf_vals = np.pad(acf_vals, (0, 11 - len(acf_vals)), 'constant')
    except Exception:
        acf_vals = [0.0] * 11

    # Additional statistical features
    skew_val = 0.0
    try:
        skew_val = np.nan_to_num(skew(arr), nan=0.0)
    except Exception:
        skew_val = 0.0
    if skew_val is None:
        skew_val = 0.0

    kurt_val = 0.0
    try:
        kurt_val = np.nan_to_num(kurtosis(arr), nan=0.0)
    except Exception:
        kurt_val = 0.0
    if kurt_val is None:
        kurt_val = 0.0

    # Temporal features
    lag_1_diff = arr[-1] - arr[-2] if len(arr) > 1 else 0.0
    rolling_mean = np.mean(arr[-5:]) if len(arr) >= 5 else mean

    # Trend (slope of linear regression)
    slope = 0.0
    try:
        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0]
    except Exception:
        slope = 0.0
    if slope is None:
        slope = 0.0
    # Entropy

    ent = 0.0
    try:
        hist, _ = np.histogram(arr, bins=10, density=True)
        ent = entropy(hist)
    except Exception:
        ent = 0.0
    if ent is None:
        ent = 0.0

    return [mean, std, fft_mag, raw_autocorr] + list(acf_vals) + [skew_val, kurt_val, lag_1_diff, rolling_mean, slope, ent]
