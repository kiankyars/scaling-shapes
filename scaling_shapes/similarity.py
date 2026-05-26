"""Pairwise similarity of scaling-curve *shapes* across tasks.

Compares fitted logistics on a common log-compute window using shape-normalized
curves (asymptotes removed), then aggregates per-(task, size) distances into a
task×task matrix. This answers "do capabilities share the same S-curve shape?"
more directly than clustering four noisy scalar summaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import LogisticFit, predict_logistic


def normalized_shape_curve(fit: LogisticFit, x: np.ndarray) -> np.ndarray:
    """Logistic prediction rescaled to [0, 1] using the fit's own asymptotes."""
    span = fit.a_max - fit.a_min
    if span < 1e-6:
        return np.zeros_like(x, dtype=float)
    return (predict_logistic(fit, x) - fit.a_min) / span


def shape_distance_fits(
    fit_a: LogisticFit,
    fit_b: LogisticFit,
    x_lo: float,
    x_hi: float,
    *,
    n_grid: int = 50,
) -> float:
    """RMSE between normalized curves on [x_lo, x_hi] in log-compute."""
    if x_hi <= x_lo + 1e-6:
        return float("nan")
    xg = np.linspace(x_lo, x_hi, n_grid)
    ya = normalized_shape_curve(fit_a, xg)
    yb = normalized_shape_curve(fit_b, xg)
    return float(np.sqrt(np.mean((ya - yb) ** 2)))


def shape_distance_at_size(
    fit_a: LogisticFit,
    fit_b: LogisticFit,
    xs_a: np.ndarray,
    xs_b: np.ndarray,
) -> float:
    """Distance on the overlap of observed log-compute for one model size."""
    x_lo = max(float(xs_a.min()), float(xs_b.min()))
    x_hi = min(float(xs_a.max()), float(xs_b.max()))
    return shape_distance_fits(fit_a, fit_b, x_lo, x_hi)


@dataclass(frozen=True)
class TaskSimilarity:
    task_names: list[str]
    distance: np.ndarray  # (n, n), median across sizes of normalized-curve RMSE
    n_sizes_per_pair: np.ndarray  # how many sizes contributed to each cell

    def within_between_ratio(self, labels: np.ndarray) -> tuple[float, float, float]:
        """Mean within-cluster distance / mean between-cluster distance."""
        n = self.distance.shape[0]
        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance[i, j]
                if not np.isfinite(d):
                    continue
                (within if labels[i] == labels[j] else between).append(d)
        w = float(np.mean(within)) if within else float("nan")
        b = float(np.mean(between)) if between else float("nan")
        ratio = w / b if b > 0 and np.isfinite(w) else float("nan")
        return w, b, ratio


def task_shape_similarity(
    task_names: list[str],
    fits_by_task_size: dict[tuple[str, str], LogisticFit],
    curves: dict[str, dict[str, list[tuple[float, float]]]],
    *,
    max_fit_rmse: float = 0.15,
) -> TaskSimilarity:
    """Build task×task distance matrix from per-size normalized-curve RMSE.

    All tasks are included regardless of fit.k sign — a "descending S" (negative
    k after the fit's a_min/a_max swap) is a legitimate shape that should
    contribute a large distance when compared against rising S-curves. The
    only filter is a generous RMSE cap to skip catastrophic fits.
    """
    names = sorted(task_names)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    dist_sum = np.zeros((n, n), dtype=float)
    count = np.zeros((n, n), dtype=int)

    sizes = sorted({s for (_, s) in fits_by_task_size})
    for size in sizes:
        size_fits: dict[str, LogisticFit] = {}
        size_xs: dict[str, np.ndarray] = {}
        for task in names:
            key = (task, size)
            if key not in fits_by_task_size:
                continue
            fit = fits_by_task_size[key]
            if fit.rmse > max_fit_rmse:
                continue
            pts = curves.get(task, {}).get(size)
            if not pts or len(pts) < 4:
                continue
            xs = np.array([p[0] for p in pts])
            size_fits[task] = fit
            size_xs[task] = xs

        available = sorted(size_fits)
        for i, ta in enumerate(available):
            for tb in available[i + 1 :]:
                d = shape_distance_at_size(
                    size_fits[ta], size_fits[tb], size_xs[ta], size_xs[tb]
                )
                if not np.isfinite(d):
                    continue
                ia, ib = idx[ta], idx[tb]
                dist_sum[ia, ib] += d
                dist_sum[ib, ia] += d
                count[ia, ib] += 1
                count[ib, ia] += 1

    dist = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(dist, 0.0)
    mask = count > 0
    dist[mask] = dist_sum[mask] / count[mask]
    return TaskSimilarity(task_names=names, distance=dist, n_sizes_per_pair=count)
