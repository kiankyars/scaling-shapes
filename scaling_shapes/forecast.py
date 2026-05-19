"""Within-trajectory forecasting: predict late-checkpoint accuracy from early ones.

For each (size, task), fit the logistic on the first `frac` of the checkpoint
trajectory and report the RMSE on the held-out tail. This is the primary
falsifiable claim of the project: curve-fit on partial training trajectories
should predict later trajectory points within reasonable error.

Reported alongside: a baseline that predicts the last seen accuracy forever.
The fit "earns its keep" iff it beats the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import fit_logistic, predict_logistic


@dataclass(frozen=True)
class ForecastResult:
    train_fraction: float
    n_train: int
    n_test: int
    fit_rmse: float
    baseline_rmse: float    # RMSE of "predict last training value" on the held-out tail
    skill: float            # 1 - fit_rmse / baseline_rmse (>0 means fit is better)


def within_trajectory_forecast(
    log_x: np.ndarray,
    y: np.ndarray,
    *,
    train_fraction: float = 0.5,
) -> ForecastResult:
    """`log_x` is sorted training-axis values; `y` is the metric at each point."""
    order = np.argsort(log_x)
    log_x, y = np.asarray(log_x, float)[order], np.asarray(y, float)[order]
    n = log_x.size
    n_train = max(4, int(round(train_fraction * n)))
    n_test = n - n_train
    if n_test < 2:
        raise ValueError(f"need at least 2 held-out points, got {n_test}")

    x_train, y_train = log_x[:n_train], y[:n_train]
    x_test, y_test = log_x[n_train:], y[n_train:]

    fit = fit_logistic(x_train, y_train)
    yhat = predict_logistic(fit, x_test)
    fit_rmse = float(np.sqrt(np.mean((yhat - y_test) ** 2)))

    baseline_pred = np.full_like(y_test, y_train[-1])
    baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y_test) ** 2)))
    skill = 1.0 - (fit_rmse / baseline_rmse) if baseline_rmse > 0 else float("nan")
    return ForecastResult(
        train_fraction=train_fraction,
        n_train=n_train,
        n_test=n_test,
        fit_rmse=fit_rmse,
        baseline_rmse=baseline_rmse,
        skill=skill,
    )
