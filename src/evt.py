import numpy as np
from scipy.stats import genpareto, kstest


def fit_pot(scores, init_q=0.95):
    """Peaks-over-threshold: fit a GPD to scores above the init_q quantile of benign scores."""
    scores = np.asarray(scores, dtype=float)
    t = float(np.quantile(scores, init_q))
    exc = scores[scores > t] - t
    xi, _, sigma = genpareto.fit(exc, floc=0)
    ks = kstest(exc, "genpareto", args=(xi, 0, sigma))
    return {"t": t, "xi": float(xi), "sigma": float(sigma),
            "n": len(scores), "n_t": len(exc), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


def evt_threshold(fit, q=0.001):
    """Score cutoff for a target false-alarm rate q (q must be below 1 - init_q)."""
    r = q * fit["n"] / fit["n_t"]
    if abs(fit["xi"]) < 1e-8:
        return fit["t"] - fit["sigma"] * np.log(r)
    return fit["t"] + fit["sigma"] / fit["xi"] * (r ** (-fit["xi"]) - 1)