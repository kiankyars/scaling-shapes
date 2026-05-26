# Scaling Shapes

Per-capability scaling-law fits across the full Pythia checkpoint grid. See
[`SPEC.md`](SPEC.md) for the full plan.

## Layout

```
scaling_shapes/
  models.py     — Pythia v1 size table + checkpoint schedule
  tasks.py      — lm-eval-harness task list + per-task num_fewshot
  eval.py       — load one Pythia (size, revision), run the task suite
  fit.py        — per-task 4-parameter logistic curve fit
  forecast.py   — within-trajectory hold-out skill score
  similarity.py — pairwise shape distance between tasks (normalised curves)
modal_app.py    — Modal app with band-sized eval functions (L4 / A100 / H100)
scripts/
  analyze.py    — load eval JSONs, fit + similarity + forecast, write outputs/analysis.json
  plot.py       — six PNG figures into outputs/figures/
tests/          — fit + similarity + forecast unit tests
```

## Run

```sh
cd ~/Developer/scaling-shapes
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q                                        # ~1s

modal profile activate kiankyars                        # required
modal run modal_app.py::smoke                           # ~$1, ~30 min
modal run --detach modal_app.py::pilot                  # ~$770, ~10 wall hrs
```

Pilot outputs land in the Modal volume `scaling-shapes-runs` under
`evals/<size>/<revision>.json`. After the run:

```sh
modal volume get scaling-shapes-runs evals outputs/ --force
python scripts/analyze.py
python scripts/plot.py
```

## Outputs

- `outputs/analysis.json` — per-(task, size) logistic fits, pairwise shape distance matrix, per-(task, size) within-trajectory forecast skill with bootstrap CIs
- `outputs/figures/f1a_hellaswag_headline.png` — headline scaling curve
- `outputs/figures/f1b_curves_grid.png` — remaining 8 tasks, 4×2 grid
- `outputs/figures/f2_shape_similarity.png` — task×task shape-distance heatmap
- `outputs/figures/f3_forecast_bars.png` — forecast skill with 95% bootstrap CIs
- `outputs/figures/f4_forecast_exemplars.png` — one positive-skill + one anti-skill case
- `outputs/figures/f5_skill_heatmap.png` — per-(task, size) forecast skill heatmap

## Caveats

- 2.8B is excluded from analysis — HuggingFace's revision branches served the same model blob for 86 of 154 checkpoints, so its trajectories are flat and wrong. Other sizes pass a step0-near-random sanity check.
- MMLU is excluded from the pilot task list (near-random for all sizes through 6.9B and ~43% of per-checkpoint wall time).
