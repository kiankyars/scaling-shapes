"""Per-task scaling-law fits.

For each task we fit accuracy a(x) as a function of log-compute (or log-tokens,
or log-params — pluggable on the x-axis). The default form is a two-parameter
logistic with floating asymptotes:

    a(x) = a_min + (a_max - a_min) / (1 + exp(-k * (x - mu)))

with parameters (mu, k, a_min, a_max). The form covers smooth log-linear
(small k), threshold-like (large k), and saturating curves in one family.

We also fit a broken power-law (Caballero et al.) as a comparator and report
which form BIC prefers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class LogisticFit:
    mu: float          # log-x at which a(x) = (a_min + a_max) / 2
    k: float           # steepness (units: per unit of log-x)
    a_min: float
    a_max: float
    rmse: float
    bic: float
    n_points: int


def _logistic(x: np.ndarray, mu: float, k: float, a_min: float, a_max: float) -> np.ndarray:
    return a_min + (a_max - a_min) / (1.0 + np.exp(-k * (x - mu)))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> LogisticFit:
    """Least-squares fit of (mu, k, a_min, a_max).

    `x` should be log(quantity) — e.g. log10(compute_flops). `y` is in [0, 1].
    Initial guesses come from the data: mu at the median x, k from the rough
    slope at the midpoint, a_min/a_max from data extremes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 4:
        raise ValueError(f"need >=4 points, got {x.size}")

    order = np.argsort(x)
    x, y = x[order], y[order]

    a_min_init = float(np.percentile(y, 10))
    a_max_init = float(np.percentile(y, 90))
    mu_init = float(np.median(x))
    k_init = 1.0

    def residuals(params):
        mu, k, a_min, a_max = params
        return _logistic(x, mu, k, a_min, a_max) - y

    res = least_squares(
        residuals,
        x0=np.array([mu_init, k_init, a_min_init, a_max_init]),
        bounds=([x.min() - 10, 1e-4, 0.0, 0.0],
                [x.max() + 10, 50.0, 1.0, 1.0]),
        max_nfev=4000,
    )
    mu, k, a_min, a_max = res.x
    if a_max < a_min:
        a_min, a_max = a_max, a_min
        k = -k

    yhat = _logistic(x, mu, k, a_min, a_max)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    # BIC for a Gaussian residual model with sigma=rmse and k=4 parameters.
    n = x.size
    sigma2 = max(rmse ** 2, 1e-12)
    log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    bic = -2 * log_lik + 4 * math.log(n)
    return LogisticFit(
        mu=float(mu), k=float(k), a_min=float(a_min), a_max=float(a_max),
        rmse=rmse, bic=float(bic), n_points=n,
    )


def predict_logistic(fit: LogisticFit, x: np.ndarray) -> np.ndarray:
    return _logistic(np.asarray(x, dtype=float), fit.mu, fit.k, fit.a_min, fit.a_max)


def fit_features(fit: LogisticFit) -> np.ndarray:
    """Per-task feature vector used by the clustering step."""
    return np.array([fit.mu, fit.k, fit.a_min, fit.a_max], dtype=float)
