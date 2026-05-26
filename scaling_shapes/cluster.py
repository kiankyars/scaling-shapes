"""Cluster tasks by their fitted scaling-shape parameters.

Input: a (task -> LogisticFit) mapping. Output: a (task -> cluster_id)
mapping plus the dendrogram-cut diagnostic used to pick `n_clusters`.

We use agglomerative clustering on z-scored fit features so the scales of
`mu` (around 18–22 for log10-FLOPs) and `k` (around 1–10) don't dominate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from .fit import LogisticFit, fit_features


@dataclass(frozen=True)
class Clustering:
    task_names: list[str]
    labels: np.ndarray
    n_clusters: int
    silhouette: float


def cluster_tasks(
    fits: dict[str, LogisticFit],
    *,
    n_clusters: int | None = None,
    linkage: str = "ward",
) -> Clustering:
    """If `n_clusters` is None, pick the k in [2, 8] with the best silhouette."""
    from sklearn.metrics import silhouette_score

    names = sorted(fits.keys())
    X = np.stack([fit_features(fits[n]) for n in names])
    Xz = StandardScaler().fit_transform(X)

    if n_clusters is None:
        best = (None, -1.0, None)  # (k, silhouette, labels)
        # silhouette requires 2 ≤ n_labels ≤ n_samples − 1
        for k in range(2, min(8, len(names) - 1) + 1):
            labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(Xz)
            if len(set(labels)) < 2:
                continue
            s = silhouette_score(Xz, labels)
            if s > best[1]:
                best = (k, s, labels)
        k_star, sil, labels = best
        if labels is None:
            raise RuntimeError("clustering failed: no valid k in [2, 8]")
        return Clustering(task_names=names, labels=labels,
                          n_clusters=int(k_star), silhouette=float(sil))

    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(Xz)
    sil = float(silhouette_score(Xz, labels)) if len(set(labels)) >= 2 else float("nan")
    return Clustering(task_names=names, labels=labels,
                      n_clusters=int(n_clusters), silhouette=sil)
