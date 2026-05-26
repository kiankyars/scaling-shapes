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
    F4  forecast_exemplars.png - one positive-skill and one anti-skill panel
    F5  skill_heatmap.png      - 9×8 heatmap of forecast skill per (task, size)
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

    Distance = median across model sizes of the RMSE between two tasks'
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
        "Do scaling curves share the same shape across tasks?\n"
        "Pairwise distance between shape-normalised logistic fits, median across "
        "Pythia sizes.\nLow = similar emergence shape, high = dissimilar.  "
        "Rows/columns ordered by average-linkage hierarchical clustering.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F3: forecast skill bar chart with CIs + anti-skill inset
# ---------------------------------------------------------------------------
def plot_forecast(analysis: dict, out_path: Path):
    skill = analysis["forecast_skill"]

    tasks_sorted = sorted(skill.keys(), key=lambda t: -skill[t]["median"])
    ymin, ymax = -1.5, 1.0

    anti_skill_tasks = [t for t in tasks_sorted
                         if (skill[t].get("ci_hi_95") is not None
                             and skill[t]["ci_hi_95"] < 0)]

    fig, (ax, ax_anti) = plt.subplots(
        1, 2, figsize=(13.5, 5.6),
        gridspec_kw={"width_ratios": [3.5, 1], "wspace": 0.22},
    )

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
    ax.set_xticklabels(tasks_sorted, rotation=40, ha="right")
    ax.set_ylabel("forecast skill  (1 − fit_rmse / baseline_rmse)")
    ax.set_title(
        "Within-trajectory forecast skill — fit on first 50% of log-compute, predict the tail\n"
        "Bars: median across 8 Pythia sizes (2.8B excluded due to corrupted ckpts).  "
        "Whiskers: 95% bootstrap CI on the median.  Dots: per-size values (clipped at −1.5).",
        fontsize=9.5,
    )
    ax.grid(True, alpha=0.3, axis="y")

    # right panel: raw per-size distributions for anti-skill tasks (CI hi < 0)
    if anti_skill_tasks:
        for j, t in enumerate(anti_skill_tasks):
            vals = np.array(skill[t]["values"])
            jitter = np.random.default_rng(j).uniform(-0.14, 0.14, size=vals.size)
            ax_anti.scatter(np.full_like(vals, j) + jitter, vals,
                            color=F3_BAR_COLOR, s=26,
                            alpha=0.85, edgecolor="black", linewidth=0.3)
            ax_anti.scatter([j], [np.median(vals)], marker="_",
                            color="black", s=360, linewidth=2.2, zorder=5)
        ax_anti.axhline(0, color="black", linewidth=0.6)
        ax_anti.set_xlim(-0.5, len(anti_skill_tasks) - 0.5)
        ax_anti.set_xticks(range(len(anti_skill_tasks)))
        ax_anti.set_xticklabels(anti_skill_tasks, rotation=20, ha="right", fontsize=9)
        ax_anti.set_title("anti-skill tasks\nfull per-size distribution",
                          fontsize=10)
        ax_anti.set_ylabel("forecast skill (uncapped)")
        ax_anti.grid(True, alpha=0.3, axis="y")
        ax_anti.set_facecolor("#fafafa")
    else:
        ax_anti.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F4: forecast exemplars - one positive case, one anti-skill case
# ---------------------------------------------------------------------------
def _forecast_breakdown(xs: np.ndarray, ys: np.ndarray, train_fraction: float = 0.5):
    """Return (x_train, y_train, x_test, y_test, fit, baseline_pred, skill, fit_rmse, baseline_rmse)."""
    from scaling_shapes.fit import fit_logistic, predict_logistic

    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    n = xs.size
    n_train = max(4, int(round(train_fraction * n)))
    x_train, y_train = xs[:n_train], ys[:n_train]
    x_test, y_test = xs[n_train:], ys[n_train:]
    fit = fit_logistic(x_train, y_train)
    yhat = predict_logistic(fit, x_test)
    fit_rmse = float(np.sqrt(np.mean((yhat - y_test) ** 2)))
    baseline_pred = np.full_like(y_test, y_train[-1])
    baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y_test) ** 2)))
    skill = 1.0 - fit_rmse / baseline_rmse if baseline_rmse > 0 else float("nan")
    return (xs, ys, x_train, y_train, x_test, y_test, fit, baseline_pred,
            skill, fit_rmse, baseline_rmse)


def plot_forecast_exemplars(curves: dict, out_path: Path,
                             pos: tuple[str, str] = ("hellaswag", "6.9b"),
                             neg: tuple[str, str] = ("openbookqa", "1.4b")):
    from scaling_shapes.fit import predict_logistic

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, (task, size), title_prefix in zip(axes, [pos, neg], ["positive skill", "anti-skill"]):
        pts = sorted(curves[task][size], key=lambda p: p[0])
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        (xs, ys, x_tr, y_tr, x_te, y_te, fit, baseline, skill,
         fit_rmse, baseline_rmse) = _forecast_breakdown(xs, ys, 0.5)

        boundary = (x_tr[-1] + x_te[0]) / 2
        ax.axvspan(xs.min(), boundary, color="grey", alpha=0.08, zorder=0)
        ax.axvline(boundary, color="grey", linewidth=1.0, linestyle=":", zorder=1)
        ax.text((xs.min() + boundary) / 2, 0.04, "train (first 50% of log-C)",
                ha="center", va="bottom", fontsize=9, color="#555")
        ax.text((boundary + xs.max()) / 2, 0.04, "held-out tail",
                ha="center", va="bottom", fontsize=9, color="black")

        # data points
        ax.scatter(x_tr, y_tr, s=22, color="#444", alpha=0.9, label="actual (train)")
        ax.scatter(x_te, y_te, s=22, facecolor="white", edgecolor="black",
                   linewidth=0.8, label="actual (held-out)")

        # logistic fit extrapolation
        xx = np.linspace(xs.min(), xs.max(), 300)
        ax.plot(xx, predict_logistic(fit, xx), color="C0", linewidth=2.0,
                label="logistic fit (continued)")

        # baseline (constant)
        ax.plot([boundary, xs.max()], [y_tr[-1], y_tr[-1]],
                color="C3", linewidth=2.0, linestyle="--",
                label="baseline: last train value")

        ax.set_xlim(xs.min() - 0.2, xs.max() + 0.2)
        ax.set_ylim(0.0, 1.08)
        ax.set_xlabel("log₁₀ compute (FLOPs)")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{title_prefix}: {task} @ pythia-{size}\n"
            f"skill = {skill:+.3f}   "
            f"fit RMSE = {fit_rmse:.3f}   baseline RMSE = {baseline_rmse:.3f}",
            fontsize=10,
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Forecast exemplars — fit on first 50% of log-compute, "
                 "predict the held-out tail", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F5: 9x9 skill heatmap (task x size)
# ---------------------------------------------------------------------------
def plot_skill_heatmap(analysis: dict, out_path: Path):
    by_pair: dict[str, float] = analysis.get("skill_by_task_size", {})
    if not by_pair:
        raise RuntimeError("analysis.json missing skill_by_task_size — re-run analyze.py")

    skill_med = {t: v["median"] for t, v in analysis["forecast_skill"].items()}
    tasks_sorted = sorted(skill_med.keys(), key=lambda t: -skill_med[t])

    mat = np.full((len(tasks_sorted), len(SIZE_ORDER)), np.nan)
    for key, val in by_pair.items():
        task, size = key.split("|", 1)
        if task not in skill_med or size not in SIZE_ORDER:
            continue
        i = tasks_sorted.index(task)
        j = SIZE_ORDER.index(size)
        mat[i, j] = max(min(val, 1.0), -1.0)  # clip

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            raw = by_pair.get(f"{tasks_sorted[i]}|{SIZE_ORDER[j]}", v)
            txt = f"{raw:+.1f}" if abs(raw) >= 10 else f"{raw:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else "black")

    ax.set_xticks(range(len(SIZE_ORDER)))
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_yticks(range(len(tasks_sorted)))
    ax.set_yticklabels(tasks_sorted)
    ax.set_xlabel("Pythia size")
    ax.set_title("Forecast skill per (task, size) — clipped at ±1 for color, "
                 "raw values shown in-cell\n"
                 "Rows sorted by median skill across sizes (top = best).",
                 fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("forecast skill (clipped)")
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
    plot_forecast_exemplars(curves, FIGURES_DIR / "f4_forecast_exemplars.png")
    plot_skill_heatmap(analysis, FIGURES_DIR / "f5_skill_heatmap.png")

    for name in ["f1a_hellaswag_headline.png", "f1b_curves_grid.png",
                 "f2_shape_similarity.png", "f3_forecast_bars.png",
                 "f4_forecast_exemplars.png", "f5_skill_heatmap.png"]:
        print(f"wrote {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
