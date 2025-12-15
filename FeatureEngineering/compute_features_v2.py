# V2
import numpy as np
from scipy.fftpack import fft
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis, entropy
from sklearn.linear_model import LinearRegression
import builtins

def compute_features_v2(series, acf_lag_limit=96, top_n_acf=5, bottom_n_acf=5):
    number_features = 34
    try:
        # Clean the series: convert non-float values to 0.0
        cleaned_series = [
            x if isinstance(x, (float, int, np.number)) and np.isfinite(x)
            else 0.0
            for x in series
        ]
        arr = np.array(cleaned_series, dtype=np.float64)

        if len(arr) == 0 or np.all(arr == 0):
            return [0.0] * number_features

        # Basic stats (6)
        mean = np.nan_to_num(np.nanmean(arr), nan=0.0)
        std = np.nan_to_num(np.nanstd(arr), nan=0.0)
        min_val = np.nan_to_num(np.nanmin(arr), nan=0.0)
        max_val = np.nan_to_num(np.nanmax(arr), nan=0.0)
        p25 = np.nan_to_num(np.nanpercentile(arr, 25), nan=0.0)
        p75 = np.nan_to_num(np.nanpercentile(arr, 75), nan=0.0)

        # FFT features (6)
        try:
            fft_vals = np.abs(fft(arr))
            spectral_energy = np.nansum(fft_vals ** 2)
            total_fft = np.nansum(fft_vals)
            safe_total_fft = builtins.max(total_fft, 1e-10)
            spectral_entropy = entropy(fft_vals / safe_total_fft)
            freqs = np.fft.fftfreq(len(arr))
            peak_indices = np.argsort(fft_vals)[-3:]
            top_fft_peaks = [np.nan_to_num(fft_vals[i], nan=0.0) for i in peak_indices]
            top_fft_freqs = [np.nan_to_num(freqs[i], nan=0.0) for i in peak_indices]
            top_fft_peaks += [0.0] * (3 - len(top_fft_peaks))
            top_fft_freqs += [0.0] * (3 - len(top_fft_freqs))
        except Exception:
            spectral_energy = spectral_entropy = 0.0
            top_fft_peaks = [0.0] * 3
            top_fft_freqs = [0.0] * 3

        # ACF features (3 + top_n_acf + bottom_n_acf)
        try:
            acf_full = acf(arr, nlags=acf_lag_limit, fft=True)
            acf_full = np.nan_to_num(acf_full, nan=0.0)
            acf_lags = acf_full[1:]
            acf_mean = np.nan_to_num(np.mean(acf_lags), nan=0.0)
            acf_var = np.nan_to_num(np.var(acf_lags), nan=0.0)
            acf_entropy_val = entropy(np.abs(acf_lags) / np.sum(np.abs(acf_lags))) if np.sum(np.abs(acf_lags)) > 0 else 0.0
            sorted_acf = np.sort(acf_lags)
            top_acf_lags = [np.nan_to_num(x, nan=0.0) for x in sorted_acf[-top_n_acf:]]
            bottom_acf_lags = [np.nan_to_num(x, nan=0.0) for x in sorted_acf[:bottom_n_acf]]
            top_acf_lags += [0.0] * (top_n_acf - len(top_acf_lags))
            bottom_acf_lags += [0.0] * (bottom_n_acf - len(bottom_acf_lags))
        except Exception:
            acf_mean = acf_var = acf_entropy_val = 0.0
            top_acf_lags = [0.0] * top_n_acf
            bottom_acf_lags = [0.0] * bottom_n_acf

        # Distribution shape (2)
        try:
            skew_val = np.nan_to_num(skew(arr), nan=0.0)
            kurt_val = np.nan_to_num(kurtosis(arr), nan=0.0)
        except Exception:
            skew_val = kurt_val = 0.0

        # Linear regression trend (2)
        try:
            x = np.arange(len(arr)).reshape(-1, 1)
            model = LinearRegression().fit(x, arr)
            slope = np.nan_to_num(model.coef_[0], nan=0.0)
            r2 = np.nan_to_num(model.score(x, arr), nan=0.0)
        except Exception:
            slope = r2 = 0.0

        # Peak/Valley count (1)
        try:
            peaks, _ = find_peaks(arr)
            valleys, _ = find_peaks(-arr)
            peak_valley_count = len(peaks) + len(valleys)
        except Exception:
            peak_valley_count = 0.0

        # Permutation entropy (1 + 1)
        def permutation_entropy(ts, order=3):
            n = len(ts)
            if n < order:
                return 0.0, 0.0
            perms = {}
            for i in range(n - order + 1):
                key = tuple(np.argsort(ts[i:i+order]))
                perms[key] = perms.get(key, 0) + 1
            probs = np.array(list(perms.values())) / sum(perms.values())
            try:
                pe = -np.sum(probs * np.log2(probs))
                return np.nan_to_num(pe, nan=0.0)
            except Exception:
                return 0.0

        perm_entropy = permutation_entropy(arr)

        # Higuchi fractal dimension (1 + 1)
        def higuchi_fd(ts, kmax=5):
            n = len(ts)
            L = []
            x = np.array(ts)
            for k in range(1, kmax + 1):
                Lk = []
                for m in range(k):
                    x1 = x[m:n - k:k]
                    x2 = x[m + k:n:k]
                    if len(x1) != len(x2):
                        min_len = builtins.min(len(x1), len(x2))
                        x1 = x1[:min_len]
                        x2 = x2[:min_len]
                    Lmk = np.sum(np.abs(x1 - x2))
                    denom = ((n - m) // k * k)
                    if denom == 0:
                        continue
                    Lmk = (Lmk * (n - 1) / denom) / k
                    Lk.append(Lmk)
                if Lk:
                    L.append(np.mean(Lk))
            try:
                lnL = np.log(np.clip(L, a_min=1e-10, a_max=None))
                lnk = np.log(range(1, kmax + 1))
                slope, _ = np.polyfit(lnk, lnL, 1)
                if slope is None or np.isnan(slope):
                    return 0.0
                return np.nan_to_num(-slope, nan=0.0)
            except Exception:
                return 0.0

        fractal_dim = higuchi_fd(arr)

        # Combine all features in correct order
        features = [
            mean, std, min_val, max_val, p25, p75,   
            spectral_energy, spectral_entropy
        ] + top_fft_peaks + top_fft_freqs + [
            acf_mean, acf_var, acf_entropy_val 
        ] + top_acf_lags + bottom_acf_lags + [
            skew_val, kurt_val, slope, r2,                  
            peak_valley_count, perm_entropy, fractal_dim      
        ]
        
        # Final safety: convert all to native float and sanitize
        
        features = [
            float(f) if isinstance(f, (int, float, np.number)) and np.isfinite(f)
            else 0.0
            for f in features
        ]


        return features

    except Exception:
        return [0.0] * number_features
