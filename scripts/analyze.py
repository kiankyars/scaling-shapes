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

from scaling_shapes.fit import LogisticFit, fit_logistic
from scaling_shapes.forecast import within_trajectory_forecast
from scaling_shapes.similarity import task_shape_similarity
from scaling_shapes.models import SIZES, revision_step
from scaling_shapes.tasks import TASKS

PILOT_TASK_NAMES = {t.name for t in TASKS}

# 2.8B trajectories are broken — sanity check fails (every revision returns the
# same value at step0 as at step143000, e.g. hellaswag.acc_norm = 0.6078 across
# the full range, vs ~0.26 random for every other size). Root cause appears to
# be HuggingFace's revision branches resolving to the same cached blob during
# the original pilot. Excluded from the analysis until re-collected.
BROKEN_SIZES: frozenset[str] = frozenset({"2.8b"})


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
        if size_name in BROKEN_SIZES:
            continue
        params = _params(size_name)
        for revision, row in revs.items():
            step = revision_step(revision)
            if step == 0:
                continue  # FLOPs = 0; skip the random-init point on log axis
            log_c = math.log10(_flops(params, step))
            for task_name, metrics in row["results"].items():
                if task_name not in PILOT_TASK_NAMES:
                    continue  # ignore MMLU rows surfacing from early smoke runs
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

    # Pairwise shape similarity over all 9 pilot tasks. We deliberately do
    # NOT cluster these into a discrete taxonomy — the pilot doesn't have
    # enough tasks to support cluster labels honestly. The distance matrix
    # itself is the answer.
    shape_sim = task_shape_similarity(
        list(by_task.keys()), fits_by_task_size, curves
    )
    print(f"\nshape similarity (normalized-curve RMSE, mean across sizes):")
    print(f"  matrix shape: {shape_sim.distance.shape},  "
          f"finite cells: {int(np.isfinite(shape_sim.distance).sum())}/{shape_sim.distance.size}")

    # Within-trajectory forecast on each (task, size) where we have enough points.
    print("\nwithin-trajectory forecast skill (fit on first 50%, predict tail):")
    skill_by_task_size: dict[tuple[str, str], float] = {}
    for (task_name, size_name), _ in fits_by_task_size.items():
        points = sorted(curves[task_name][size_name], key=lambda p: p[0])
        if len(points) < 8:
            continue
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        try:
            r = within_trajectory_forecast(xs, ys, train_fraction=0.5)
            skill_by_task_size[(task_name, size_name)] = float(r.skill)
        except Exception:
            pass
    skill_by_task: dict[str, list[float]] = defaultdict(list)
    for (t, _), v in skill_by_task_size.items():
        skill_by_task[t].append(v)

    # Bootstrap 95% CIs on the median skill across sizes (percentile method,
    # resampling the per-size skill values with replacement).
    rng = np.random.default_rng(0)
    n_boot = 2000
    skill_ci: dict[str, tuple[float, float, float]] = {}
    for task_name, vals in skill_by_task.items():
        arr = np.asarray(vals, dtype=float)
        meds = np.median(arr[rng.integers(0, arr.size, size=(n_boot, arr.size))], axis=1)
        skill_ci[task_name] = (float(np.median(arr)),
                                float(np.percentile(meds, 2.5)),
                                float(np.percentile(meds, 97.5)))

    print(f'  {"task":>20} | {"median":>8} | {"95% CI":>20} | {"n sizes":>8}')
    for task_name in sorted(skill_by_task, key=lambda n: -np.median(skill_by_task[n])):
        med, lo, hi = skill_ci[task_name]
        print(f"  {task_name:>20} | {med:>+8.3f} | [{lo:>+7.3f}, {hi:>+7.3f}] | "
              f"{len(skill_by_task[task_name]):>8}")

    # Write a single JSON artifact for downstream plotting / sharing.
    out_path = Path("outputs/analysis.json")
    artifact = {
        "n_eval_rows": int(sum(len(r) for r in evals.values())),
        "n_sizes": len(evals),
        "fits_by_task_size": {
            f"{t}|{s}": {
                "mu": f.mu, "k": f.k, "a_min": f.a_min, "a_max": f.a_max,
                "rmse": f.rmse, "bic": f.bic, "n_points": f.n_points,
            }
            for (t, s), f in fits_by_task_size.items()
        },
        "shape_similarity": {
            "tasks": shape_sim.task_names,
            "distance": shape_sim.distance.tolist(),
            "n_sizes_per_pair": shape_sim.n_sizes_per_pair.tolist(),
        },
        "forecast_skill": {
            t: {
                "median": float(np.median(v)),
                "values": [float(x) for x in v],
                "n_sizes": len(v),
                "ci_lo_95": skill_ci[t][1],
                "ci_hi_95": skill_ci[t][2],
            }
            for t, v in skill_by_task.items()
        },
        "skill_by_task_size": {
            f"{t}|{s}": v for (t, s), v in skill_by_task_size.items()
        },
    }
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
