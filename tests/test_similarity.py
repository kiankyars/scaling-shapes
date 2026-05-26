import numpy as np

from scaling_shapes.fit import LogisticFit, fit_logistic
from scaling_shapes.similarity import (
    shape_distance_fits,
    task_shape_similarity,
)


def _fit(mu, k, a_min, a_max, n=40):
    x = np.linspace(17.0, 22.0, n)
    y = a_min + (a_max - a_min) / (1.0 + np.exp(-k * (x - mu)))
    return fit_logistic(x, y)


def test_identical_shapes_have_zero_distance():
    fa = _fit(20.0, 3.0, 0.2, 0.8)
    fb = _fit(20.0, 3.0, 0.4, 0.9)  # same μ, k; different asymptotes only
    d = shape_distance_fits(fa, fb, 17.5, 21.5)
    assert d < 0.05


def test_different_k_increases_distance():
    fa = _fit(20.0, 1.0, 0.2, 0.8)
    fb = _fit(20.0, 6.0, 0.2, 0.8)
    d = shape_distance_fits(fa, fb, 17.5, 21.5)
    assert d > 0.15


def test_task_matrix_symmetric():
    fa, fb = _fit(20.0, 2.0, 0.2, 0.7), _fit(20.0, 5.0, 0.2, 0.7)
    fits = {("a", "14m"): fa, ("b", "14m"): fb}
    curves = {
        "a": {"14m": list(zip(np.linspace(17, 22, 30), np.linspace(0.2, 0.7, 30)))},
        "b": {"14m": list(zip(np.linspace(17, 22, 30), np.linspace(0.2, 0.7, 30)))},
    }
    sim = task_shape_similarity(["a", "b"], fits, curves, max_fit_rmse=1.0)
    assert sim.distance[0, 1] == sim.distance[1, 0]
    assert sim.distance[0, 0] == 0.0
