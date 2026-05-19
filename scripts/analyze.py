"""Read per-(size, revision) eval JSONs from the runs volume, fit per-task
logistics, cluster, and run the within-trajectory forecasting hold-out.

Usage:
    cd ~/Developer/scaling-shapes
    source .venv/bin/activate
    modal volume get scaling-shapes-runs evals outputs/ --force
    python scripts/analyze.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scaling_shapes.cluster import cluster_tasks
from scaling_shapes.fit import LogisticFit, fit_logistic
from scaling_shapes.forecast import within_trajectory_forecast
from scaling_shapes.models import SIZES, revision_step
from scaling_shapes.tasks import TASKS


def _params(size_name: str) -> int:
    return next(s.params for s in SIZES if s.name == size_name)


def _flops(params: int, step: int, tokens_per_step: int = 2 * 1024 * 1024) -> float:
    """Pythia uses bs=1024, seq=2048 → ~2M tokens/step. FLOPs ≈ 6·N·D."""
    return 6.0 * params * step * tokens_per_step


def load_evals(eval_dir: Path) -> dict[str, dict[str, dict]]:
    """{size_name: {revision: {results: {task_name: {acc: ..., ...}}}}}"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for sub in eval_dir.iterdir():
        if not sub.is_dir():
            continue
        for path in sub.glob("step*.json"):
            row = json.loads(path.read_text())
            out[sub.name][row["revision"]] = row
    return dict(out)


def _primary_value(task_name: str, metrics: dict) -> float | None:
    """Use `acc_norm` when present (Pythia/OpenLLM convention).

    lm-eval-harness emits keys of the form `acc,<filter>` and `acc_norm,<filter>`.
    The default filter is `none`. Accept either flavor (with or without filter).
    """
    for k in ("acc_norm,none", "acc_norm"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return float(metrics[k])
    for k in ("acc,none", "acc"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return float(metrics[k])
    return None


def per_task_curves(evals: dict) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Returns {task_name: {size_name: [(log10_flops, accuracy), ...]}}."""
    out: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for size_name, revs in evals.items():
        params = _params(size_name)
        for revision, row in revs.items():
            step = revision_step(revision)
            if step == 0:
                continue  # FLOPs = 0; skip the random-init point on log axis
            log_c = math.log10(_flops(params, step))
            for task_name, metrics in row["results"].items():
                v = _primary_value(task_name, metrics)
                if v is None or not (0.0 <= v <= 1.0):
                    continue
                out[task_name][size_name].append((log_c, v))
    return out


def main():
    eval_dir = Path("outputs/evals")
    evals = load_evals(eval_dir)
    if not evals:
        print(f"no evals under {eval_dir} — run `modal volume get scaling-shapes-runs evals outputs/ --force` first")
        return
    print(f"loaded {sum(len(r) for r in evals.values())} eval rows across {len(evals)} sizes")

    curves = per_task_curves(evals)
    print(f"found {len(curves)} distinct tasks")

    # Per-(size, task) logistic fits.
    fits_by_task_size: dict[tuple[str, str], LogisticFit] = {}
    for task_name, by_size in curves.items():
        for size_name, points in by_size.items():
            if len(points) < 4:
                continue
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            try:
                fits_by_task_size[(task_name, size_name)] = fit_logistic(xs, ys)
            except Exception as e:
                print(f"  fit failed: {task_name}/{size_name}: {e}")

    print(f"\nfitted {len(fits_by_task_size)} (task, size) curves")

    # Aggregate per-task across sizes (average mu, k weighted by n_points)
    by_task: dict[str, LogisticFit] = {}
    for task_name in curves:
        relevant = [(s, f) for (t, s), f in fits_by_task_size.items() if t == task_name]
        if not relevant:
            continue
        w = np.array([f.n_points for _, f in relevant], dtype=float)
        mu = float(np.average([f.mu for _, f in relevant], weights=w))
        k = float(np.average([f.k for _, f in relevant], weights=w))
        a_min = float(np.average([f.a_min for _, f in relevant], weights=w))
        a_max = float(np.average([f.a_max for _, f in relevant], weights=w))
        rmse = float(np.average([f.rmse for _, f in relevant], weights=w))
        n = int(sum(f.n_points for _, f in relevant))
        bic = float(np.average([f.bic for _, f in relevant], weights=w))
        by_task[task_name] = LogisticFit(mu=mu, k=k, a_min=a_min, a_max=a_max,
                                          rmse=rmse, bic=bic, n_points=n)

    # Cluster.
    print(f"\nclustering {len(by_task)} per-task fits")
    clustering = cluster_tasks(by_task)
    print(f"  best k={clustering.n_clusters}, silhouette={clustering.silhouette:.3f}")

    by_label: dict[int, list[str]] = defaultdict(list)
    for name, lbl in zip(clustering.task_names, clustering.labels):
        by_label[int(lbl)].append(name)
    for lbl in sorted(by_label):
        members = by_label[lbl]
        mus = [by_task[m].mu for m in members]
        ks = [by_task[m].k for m in members]
        print(f"  cluster {lbl} (n={len(members)}, mu={np.mean(mus):.2f}, k={np.mean(ks):.2f}): "
              f"{', '.join(sorted(members)[:6])}{', ...' if len(members) > 6 else ''}")

    # Within-trajectory forecast on each (task, size) where we have enough points.
    print("\nwithin-trajectory forecast skill (fit on first 50%, predict tail):")
    skill_by_task = defaultdict(list)
    for (task_name, size_name), _ in fits_by_task_size.items():
        points = sorted(curves[task_name][size_name], key=lambda p: p[0])
        if len(points) < 8:
            continue
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        try:
            r = within_trajectory_forecast(xs, ys, train_fraction=0.5)
            skill_by_task[task_name].append(r.skill)
        except Exception:
            pass
    print(f'  {"task":>20} | {"median skill":>14} | {"n sizes":>8}')
    for task_name in sorted(skill_by_task, key=lambda n: -np.median(skill_by_task[n])):
        vals = skill_by_task[task_name]
        print(f"  {task_name:>20} | {np.median(vals):>+14.3f} | {len(vals):>8}")


if __name__ == "__main__":
    main()
