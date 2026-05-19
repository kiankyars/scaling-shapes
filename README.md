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
  cluster.py    — z-scored agglomerative clustering on fit features
  forecast.py   — within-trajectory hold-out skill score
modal_app.py    — Modal app with band-sized eval functions (L4 / A100 / H100)
scripts/
  analyze.py    — load eval JSONs, fit + cluster + forecast, print summary
tests/          — fit + clustering + forecast unit tests
```

## Run

```sh
cd ~/Developer/scaling-shapes
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q                                        # ~1s, 8 tests

modal profile activate kiankyars                        # required
modal run modal_app.py::smoke                           # ~$1, ~30 min
modal run --detach modal_app.py::pilot                  # ~$770, ~10 wall hrs
```

Pilot outputs land in the Modal volume `scaling-shapes-runs` under
`evals/<size>/<revision>.json`. After the run:

```sh
modal volume get scaling-shapes-runs evals outputs/ --force
python scripts/analyze.py
```
