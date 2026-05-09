# Memorization Dose Response

Controlled canary-injection pretraining of small language models to map the joint
dose-response surface of duplication, fact rarity, and inter-occurrence spacing onto
**memorization** vs **generalization**.

See `SPEC.md` for the full plan.

## Layout

```
src/mdr/
  canaries.py    — generate canary classes (subject, relation, object) over (k, r, s)
  data.py        — base corpus streaming + canary injection into the training stream
  model.py       — GPT-NeoX 125M / 350M / 1B configs
  train.py       — single-process bf16 training loop, periodic checkpoints
  evaluate.py    — memorization + generalization probes against a checkpoint
modal_app.py     — Modal app: image, volume, secrets, GPU function
scripts/
  smoke_local.py — 1-minute CPU smoke test of the data pipeline + tiny model
  launch_pilot.py— call the Modal entrypoint for the pilot config
configs/
  pilot.json     — pilot config (125M, 5B tokens, reduced 30-class grid)
```

## Run the pilot on Modal

```
cd ~/Developer/memorization-dose-response
modal run modal_app.py::pilot
```

Outputs land in the Modal Volume `mdr-runs` under `<run_id>/`.
