import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

def window_signal(x: np.ndarray, window_size: int, overlap: float):
    """
    Split 1D signal into overlapping windows.
    - window size: samples per window (ex: 2048 at fs=20kHz ~ 0.1024s)
    - overlap: fraction between 0 and 1 (ex: 0.5 -> 50% overlap)
    Returns an array of shape (num_windows, window_size)
    """
    step = int(window_size * (1-overlap))
    if step <=0:
        raise ValueError("overlap too large -> non-positive step")
    stop = len(x) - window_size + 1
    idx = range(0, stop, step)
    return np.stack([x[i:i+window_size] for i in idx], axis=0)

def bandpower_welch(x, fs, fmin, fmax, nperseg=1024):
    """
    Power in [fmin, fmax] Hz using Welch PSD.
    """
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f < fmax)
    return np.trapezoid(Pxx[mask], f[mask])

def spectral_centroid_welch(x, fs, nperseg=1024):
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    denom = np.sum(Pxx) + 1e-12
    return float(np.sum(f*Pxx)/denom)

def extract_features_from_window(win, fs=20000):
    """
    Compute features for ONE window (1D array).
    Returns a dict of scaler features.
    """
    #Time-domain
    rms = np.sqrt(np.mean(win**2))
    stdv = float(np.std(win))
    sk = float(skew(win))
    ku = float(kurtosis(win, fisher=True))
    p2p = float(np.ptp(win))
    peak = float(np.max(np.abs(win)))
    crest = float(peak / (rms + 1e-12))

    #Frequency-domain
    sc = spectral_centroid_welch(win, fs=fs)
    bp_0_1k = bandpower_welch(win, fs=fs, fmin=0, fmax=1000)
    bp_1k_5k = bandpower_welch(win, fs, 1000, 5000)
    bp_5k_10k = bandpower_welch(win, fs, 5000, 10000)

    return {
        "rms": rms, "std": stdv, "skew": sk, "kurtosis": ku,
        "peak_to_peak": p2p, "crest_factor": crest, "spec_centroid": sc,
        "bp_0_1k": bp_0_1k, "bp_1k_5k": bp_1k_5k, "bp_5k_10k": bp_5k_10k
    }

def aggregate_window_features(win_feats_list):
    """
    Aggregate a list of per-window feature dicts into summary stats.
    For each feature, compute mean and std across windows.
    """
    keys = list(win_feats_list[0].keys())
    arr = np.array([[d[k] for k in keys] for d in win_feats_list])
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    out = {}
    for k, m, s in zip(keys, means, stds):
        out[f"{k}_mean"] = float(m)
        out[f"{k}_std"] = float(s)
    return out
