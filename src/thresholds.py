import numpy as np


def percentile_thr(scores, q=0.99):
    return float(np.quantile(scores, q))


def mean_k_thr(scores, k=3.0):
    return float(np.mean(scores) + k * np.std(scores))


def mad_thr(scores, k=3.5):
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    return float(med + k * 1.4826 * mad)      # 1.4826 makes MAD comparable to a std