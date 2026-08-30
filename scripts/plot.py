"""Five blog-ready figures for the scaling-shapes pilot.

Reads `outputs/analysis.json` (produced by `scripts/analyze.py`) plus the raw
eval JSONs under `outputs/evals/`. Writes one PNG per figure under
`outputs/figures/` at 160 dpi, white background, distinct categorical colors
for the 8 Pythia sizes.

Figures:
    F1a hellaswag_headline.png - single large headline panel for hellaswag
    F1b curves_grid.png        - 4×2 appendix grid, remaining 8 tasks
    F2  shape_similarity.png   - all-9-tasks shape-distance heatmap, hierarchical ordering
    F3  forecast_bars.png      - forecast skill bar chart with 95% bootstrap CIs
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from scaling_shapes.fit import LogisticFit, predict_logistic
from scaling_shapes.models import SIZES, revision_step
from scaling_shapes.tasks import TASKS

PILOT_TASKS = [t.name for t in TASKS]
# 2.8B excluded — see scripts/analyze.py BROKEN_SIZES.
BROKEN_SIZES: frozenset[str] = frozenset({"2.8b"})
SIZE_ORDER = [s.name for s in SIZES if s.name not in BROKEN_SIZES]
PARAMS = {s.name: s.params for s in SIZES if s.name not in BROKEN_SIZES}

DPI = 160
FIGURES_DIR = Path("outputs/figures")

# Categorical palette so adjacent sizes are easy to tell apart in F1 — viridis
# blends into a green smear when 8 trajectories overlap in the same panel.
_DISTINCT_8 = [
    "#1f77b4",  # blue       — 14m
    "#ff7f0e",  # orange     — 31m
    "#2ca02c",  # green      — 70m
    "#d62728",  # red        — 160m
    "#9467bd",  # purple     — 410m
    "#8c564b",  # brown      — 1b
    "#17becf",  # cyan       — 1.4b
    "#e377c2",  # magenta    — 6.9b
]
SIZE_COLORS = np.array([plt.matplotlib.colors.to_rgba(c) for c in _DISTINCT_8[:len(SIZE_ORDER)]])

# Tasks whose fitted logistic doesn't describe a real emergence S-curve — flat
# or non-monotone trajectories that the sigmoid form awkwardly summarises. Kept
# as a visual segregation in F1b; nothing else uses this set.
NON_S_SHAPE: frozenset[str] = frozenset({"arc_challenge", "openbookqa"})

# Neutral fill for the F3 bars (clustering removed, so no per-task color).
F3_BAR_COLOR = "#4c72b0"


def _flops(params: int, step: int, tokens_per_step: int = 2 * 1024 * 1024) -> float:
    return 6.0 * params * step * tokens_per_step


def _primary_value(metrics: dict) -> float | None:
    for k in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def load_curves(eval_dir: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """{task: {size: [(log10 FLOPs, accuracy), ...]}} for all pilot tasks."""
    out: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for size_dir in eval_dir.iterdir():
        if not size_dir.is_dir():
            continue
        size = size_dir.name
        params = PARAMS.get(size)
        if params is None:
            continue
        for path in size_dir.glob("step*.json"):
            row = json.loads(path.read_text())
            step = revision_step(row["revision"])
            if step == 0:
                continue
            log_c = math.log10(_flops(params, step))
            for task, metrics in row["results"].items():
                if task not in PILOT_TASKS:
                    continue
                v = _primary_value(metrics)
                if v is None or not (0.0 <= v <= 1.0):
                    continue
                out[task][size].append((log_c, v))
    return out


def _to_logistic_fit(d: dict) -> LogisticFit:
    return LogisticFit(mu=d["mu"], k=d["k"], a_min=d["a_min"], a_max=d["a_max"],
                       rmse=d["rmse"], bic=d["bic"], n_points=d["n_points"])


# ---------------------------------------------------------------------------
# F1: per-task scaling curves
# F1a is a single large panel for the headline task; F1b is the 4×2 appendix
# grid covering the rest, with arc_challenge + openbookqa in the bottom row.
# ---------------------------------------------------------------------------
def _plot_one_task(ax, curves: dict, fits: dict, task: str,
                   *, show_dots: bool = False, fit_linewidth: float = 2.2):
    """Draw the 8 fitted logistics for one task onto `ax`. Used by both F1a and F1b."""
    mus = []
    for i, size in enumerate(SIZE_ORDER):
        pts = sorted(curves.get(task, {}).get(size, []), key=lambda p: p[0])
        if not pts:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if show_dots:
            ax.plot(xs, ys, marker="o", linestyle="", color=SIZE_COLORS[i],
                    alpha=0.25, markersize=2.5, markeredgewidth=0)
        fit_d = fits.get(f"{task}|{size}")
        if fit_d is None:
            continue
        fit = _to_logistic_fit(fit_d)
        xx = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(xx, predict_logistic(fit, xx), color=SIZE_COLORS[i],
                linewidth=fit_linewidth, alpha=0.95, label=size)
        mus.append((fit.mu, SIZE_COLORS[i]))

    # μ ticks at panel top
    for mu, color in mus:
        ax.plot([mu, mu], [1.00, 1.04], color=color, linewidth=1.4,
                solid_capstyle="butt", clip_on=False)

    if mus:
        mu_bar = float(np.mean([m for m, _ in mus]))
        ks = [_to_logistic_fit(fits[f"{task}|{s}"]).k for s in SIZE_ORDER
              if f"{task}|{s}" in fits]
        k_bar = float(np.mean(ks)) if ks else float("nan")
        stamp = f"μ̄={mu_bar:.1f}   k̄={k_bar:+.1f}"
        ax.text(0.02, 0.93, stamp, transform=ax.transAxes, fontsize=9,
                color="#444", fontweight="bold", verticalalignment="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))


def plot_curves_headline(curves: dict, analysis: dict, out_path: Path,
                          task: str = "hellaswag"):
    """F1a — single large panel for the headline task, dots + fits visible."""
    fig, ax = plt.subplots(figsize=(11.5, 7))
    _plot_one_task(ax, curves, analysis["fits_by_task_size"], task,
                   show_dots=True, fit_linewidth=2.6)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("log₁₀ compute (FLOPs)", fontsize=11)
    ax.set_ylabel("accuracy", fontsize=11)
    ax.set_title(f"{task} — per-checkpoint accuracy vs log-compute, eight Pythia sizes\n"
                 "Solid lines: per-(task, size) logistic fit.  Dots: raw checkpoints.  "
                 "Top ticks: each size's fitted midpoint μ.",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Pythia size", loc="lower right", fontsize=10,
              framealpha=0.92, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_curves(curves: dict, analysis: dict, out_path: Path,
                exclude: str = "hellaswag"):
    """F1b — 4×2 grid for the remaining 8 tasks (HellaSwag is in F1a).

    Rows 1–3 hold six tasks whose logistic fits describe a real emergence
    shape. Row 4 holds the two whose trajectories are flat / non-monotone
    (arc_challenge, openbookqa) — the sigmoid form summarises them but the
    parameters don't carry a real shape interpretation. The visual segregation
    is the entire taxonomy this figure makes.
    """
    fits = analysis["fits_by_task_size"]

    other_tasks = [t for t in PILOT_TASKS if t != exclude]
    s_shape = [t for t in other_tasks if t not in NON_S_SHAPE]
    non_s = [t for t in other_tasks if t in NON_S_SHAPE]
    layout = s_shape + non_s
    n_rows = (len(layout) + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.8 * n_rows),
                              sharex=True, sharey=True)
    for ax, task in zip(axes.flat, layout):
        _plot_one_task(ax, curves, fits, task,
                       show_dots=False, fit_linewidth=2.2)
        ax.set_title(task, fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.06)
        ax.grid(True, alpha=0.3)
        if task in NON_S_SHAPE:
            ax.set_facecolor("#f5f0e8")  # warm-grey wash on the non-S-shape row

    for ax in axes[-1, :]:
        ax.set_xlabel("log₁₀ compute (FLOPs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(SIZE_ORDER),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.005),
               title="Pythia size")
    fig.suptitle("Per-(task, size) logistic fits — HellaSwag is in the headline figure.\n"
                 "Top six panels: tasks with S-shaped trajectories.  "
                 "Bottom row (shaded): flat / non-monotone trajectories, where the logistic "
                 "parameters aren't meaningful shape descriptors.",
                 y=1.00, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F2: shape-distance heatmap (all 9 tasks, hierarchical ordering)
# ---------------------------------------------------------------------------
def plot_shape_similarity(analysis: dict, out_path: Path):
    """Task×task heatmap of pairwise shape distance.

    Distance = mean across model sizes of the RMSE between two tasks'
    shape-normalised fitted curves on their overlapping log-compute window.
    Low distance ⇒ similar emergence shape; high distance ⇒ dissimilar.
    Rows and columns are reordered by average-linkage hierarchical clustering
    on the distance matrix so visually similar tasks sit next to each other —
    the dendrogram itself is not drawn; this is layout only.
    """
    sim = analysis.get("shape_similarity")
    if sim is None:
        raise RuntimeError("analysis.json missing shape_similarity — re-run analyze.py")

    tasks = sim["tasks"]
    dist = np.array(sim["distance"], dtype=float)

    # Hierarchical ordering of rows/columns. NaN cells get patched to the max
    # observed distance so scipy's linkage doesn't choke; the visible heatmap
    # still shows NaNs with a special facecolor.
    finite_max = float(np.nanmax(dist)) if np.isfinite(dist).any() else 1.0
    dist_for_link = np.where(np.isfinite(dist), dist, finite_max)
    np.fill_diagonal(dist_for_link, 0.0)
    condensed = squareform(dist_for_link, checks=False)
    order = dendrogram(linkage(condensed, method="average"), no_plot=True)["leaves"]
    tasks_ord = [tasks[i] for i in order]
    dist_ord = dist[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#dddddd")  # NaN cells get a neutral grey
    vmax = float(np.nanpercentile(dist, 95))
    im = ax.imshow(np.ma.masked_invalid(dist_ord), cmap=cmap, vmin=0, vmax=vmax)

    # In-cell labels for readability
    for i in range(len(tasks_ord)):
        for j in range(len(tasks_ord)):
            v = dist_ord[i, j]
            if not np.isfinite(v):
                continue
            txt = f"{v:.2f}"
            text_color = "white" if v > 0.55 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=text_color)

    ax.set_xticks(range(len(tasks_ord)))
    ax.set_yticks(range(len(tasks_ord)))
    ax.set_xticklabels(tasks_ord, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(tasks_ord, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("shape distance  (RMSE on shape-normalised fitted curves)")

    ax.set_title(
        "Pairwise distance between shape-normalised logistic fits, mean across Pythia sizes",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F3: forecast skill bar chart with CIs + anti-skill inset
# ---------------------------------------------------------------------------
def plot_forecast(analysis: dict, out_path: Path):
    """F3 — forecast skill per task, bars + 95% bootstrap CIs + per-size dots."""
    skill = analysis["forecast_skill"]
    tasks_sorted = sorted(skill.keys(), key=lambda t: -skill[t]["median"])
    ymin, ymax = -1.1, 1.0

    fig, ax = plt.subplots(figsize=(11, 5.6))

    for i, task in enumerate(tasks_sorted):
        info = skill[task]
        ax.bar(i, info["median"], color=F3_BAR_COLOR, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        lo, hi = info.get("ci_lo_95"), info.get("ci_hi_95")
        if lo is not None and hi is not None:
            ax.errorbar([i], [info["median"]],
                        yerr=[[info["median"] - lo], [hi - info["median"]]],
                        fmt="none", ecolor="black", elinewidth=1.4, capsize=4, zorder=4)
        vals = np.array(info["values"])
        in_range = vals >= ymin
        ax.scatter([i] * int(in_range.sum()), vals[in_range], color="black",
                   s=14, alpha=0.45, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(len(tasks_sorted)))
    ax.set_xticklabels(tasks_sorted, rotation=35, ha="right")
    ax.set_ylabel("forecast skill")
    ax.set_title("Can the first half of a training trajectory predict the second half?\n"
                 "Forecast skill per task (median across 8 Pythia sizes, 95% bootstrap CI).",
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(Path("outputs/analysis.json").read_text())
    curves = load_curves(Path("outputs/evals"))

    plot_curves_headline(curves, analysis, FIGURES_DIR / "f1a_hellaswag_headline.png")
    plot_curves(curves, analysis, FIGURES_DIR / "f1b_curves_grid.png")
    plot_shape_similarity(analysis, FIGURES_DIR / "f2_shape_similarity.png")
    plot_forecast(analysis, FIGURES_DIR / "f3_forecast_bars.png")

    for name in ["f1a_hellaswag_headline.png", "f1b_curves_grid.png",
                 "f2_shape_similarity.png", "f3_forecast_bars.png"]:
        print(f"wrote {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
