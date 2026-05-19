"""Unit tests for the curve-fit + clustering + forecast modules."""
import numpy as np

from scaling_shapes.cluster import cluster_tasks
from scaling_shapes.fit import LogisticFit, fit_logistic, predict_logistic
from scaling_shapes.forecast import within_trajectory_forecast


def _make_logistic(mu, k, a_min, a_max, x, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = a_min + (a_max - a_min) / (1.0 + np.exp(-k * (x - mu)))
    if noise:
        y = y + rng.normal(0, noise, size=x.shape)
    return np.clip(y, 0.0, 1.0)


def test_recovers_planted_logistic_under_noise():
    x = np.linspace(17.0, 22.0, 30)
    y = _make_logistic(mu=20.0, k=3.0, a_min=0.25, a_max=0.85, x=x, noise=0.02, seed=1)
    fit = fit_logistic(x, y)
    assert abs(fit.mu - 20.0) < 0.3, fit
    assert abs(fit.k - 3.0) < 1.0, fit
    assert abs(fit.a_min - 0.25) < 0.05, fit
    assert abs(fit.a_max - 0.85) < 0.05, fit
    assert fit.rmse < 0.05


def test_fits_smooth_log_linear_with_small_k():
    """Smooth log-linear-ish curves should produce small k (slow transitions)."""
    x = np.linspace(15.0, 23.0, 40)
    y = _make_logistic(mu=19.0, k=0.5, a_min=0.30, a_max=0.55, x=x, noise=0.01, seed=2)
    fit = fit_logistic(x, y)
    assert fit.k < 2.0  # transition is gentle
    yhat = predict_logistic(fit, x)
    assert np.sqrt(np.mean((yhat - y) ** 2)) < 0.03


def test_clustering_separates_two_curve_shapes():
    """A sharp-emergence and a smooth-growth set of tasks should cluster apart."""
    fits = {}
    for i in range(5):
        fits[f"sharp_{i}"] = LogisticFit(mu=20.0, k=8.0 + 0.2 * i,
                                         a_min=0.25, a_max=0.85,
                                         rmse=0.01, bic=0.0, n_points=30)
    for i in range(5):
        fits[f"smooth_{i}"] = LogisticFit(mu=20.0, k=0.5 + 0.05 * i,
                                          a_min=0.25, a_max=0.55,
                                          rmse=0.01, bic=0.0, n_points=30)
    result = cluster_tasks(fits, n_clusters=2)
    by_label = {}
    for name, lbl in zip(result.task_names, result.labels):
        by_label.setdefault(int(lbl), []).append(name)
    assert len(by_label) == 2
    for label, names in by_label.items():
        # Either all-sharp or all-smooth, no mixing.
        assert all(n.startswith("sharp_") for n in names) or all(n.startswith("smooth_") for n in names)


def test_forecast_skill_positive_on_clean_logistic():
    """When the planted transition lives inside the train window, the fit
    should extrapolate the held-out tail much better than 'predict the last
    training value', which keeps reading from the rising part of the curve."""
    x = np.linspace(15.0, 23.0, 30)
    y = _make_logistic(mu=17.0, k=3.0, a_min=0.25, a_max=0.85, x=x, noise=0.01, seed=3)
    result = within_trajectory_forecast(x, y, train_fraction=0.5)
    assert result.skill > 0.1, result
    assert result.fit_rmse < result.baseline_rmse


def test_forecast_skill_zero_on_pure_noise():
    rng = np.random.default_rng(5)
    x = np.linspace(15.0, 23.0, 30)
    y = np.clip(0.5 + rng.normal(0, 0.05, size=x.shape), 0, 1)
    result = within_trajectory_forecast(x, y, train_fraction=0.5)
    # On pure noise, no fit should beat the baseline by much.
    assert result.skill < 0.5
