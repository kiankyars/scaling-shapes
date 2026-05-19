# Scaling Shapes

Do different LLM capabilities have distinct scaling signatures (slope, midpoint, sharpness, asymptote) when measured at high checkpoint resolution? Can those signatures be **clustered** into a small taxonomy, and does the cluster a task belongs to **forecast** its later-trajectory accuracy from earlier checkpoints?

## Configuration (fixed across the pilot)

- **Model suite**: Pythia v1, 9 sizes — 14M, 31M, 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B (12B excluded; cost alone exceeds budget).
- **Checkpoints**: all 154 published per size — `step0, step1, …, step512`, then every 1000 to `step143000`.
- **Eval framework**: `lm-evaluation-harness == 0.4.7`, default per-task templates and prompts.
- **Task list** (~75 task-instances counting MMLU subtasks):
  - `arc_easy` (0-shot, acc_norm)
  - `arc_challenge` (25-shot, acc_norm)
  - `hellaswag` (10-shot, acc_norm)
  - `piqa` (0-shot, acc_norm)
  - `winogrande` (5-shot, acc)
  - `sciq` (0-shot, acc_norm)
  - `boolq` (0-shot, acc)
  - `openbookqa` (0-shot, acc_norm)
  - `lambada_openai` (0-shot, acc)
  - `mmlu` (5-shot, acc) — expands to 57 subtasks
- **Precision**: bf16 inference on all sizes.
- **Per-GPU sizing**:
  - 14M–410M → L4 24GB
  - 1B–2.8B → A100 40GB
  - 6.9B → H100 80GB

## What varies

| Axis | Levels |
|---|---|
| Parameters `N` | 14M, 31M, 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B |
| Training step `s` | 154 published Pythia checkpoints |
| Compute `C` | derived: C ≈ 6·N·s·tokens_per_step (Pythia trained at bs=1024×seq=2048, so ~2.1M tokens/step) |
| Task | ~75 (10 base, MMLU expanded) |

## Scaling-law fit

Per `(task, size)` we fit a 4-parameter logistic in log₁₀(C):

`a(C) = a_min + (a_max − a_min) / (1 + exp(−k · (log₁₀ C − μ)))`

- `μ`: emergence midpoint (where accuracy is halfway between floor and ceiling)
- `k`: emergence sharpness (large k = threshold-like, small k = smooth log-linear)
- `a_min`, `a_max`: random-init floor / saturation ceiling

Bootstrapped uncertainty on each parameter; BIC vs piecewise-linear and broken-power-law alternatives reported per task.

## Clustering

Per task we aggregate the across-size fits into one feature vector `(μ̄, k̄, ā_min, ā_max)` (weighted average). Z-score, then agglomerative clustering with the silhouette-best k in `[2, 8]`. Bootstrap stability check.

## Forecasting (the falsifiable claim)

For each `(size, task)` with ≥8 checkpoints: fit on the first 50% of log-C, predict the held-out tail, score RMSE. Baseline is "predict the last training value forever." **Skill** = `1 − fit_rmse / baseline_rmse`.

The pilot's central claim:
> For most tasks and most sizes, **skill > 0** at `train_fraction = 0.5`. The logistic fit's parameters generalize across the held-out portion of training.

We also report the cross-size cluster-transfer forecast as a secondary result: given a task's small-model fits, predict its larger-model curve using the cluster mean. This is the V2 story; the within-trajectory result is the headline.

## Compute estimate

Per-checkpoint wall time, assuming `batch_size="auto"` lm-eval-harness selection and amortized model load. Realistic ±2× uncertainty per cell.

| Size | GPU | $/hr | Hrs/ckpt | × 154 ckpts | $/size |
|---|---|---:|---:|---:|---:|
| 14M | L4 | $1.10 | 0.003 | 0.4 | <$1 |
| 31M | L4 | $1.10 | 0.005 | 0.8 | $1 |
| 70M | L4 | $1.10 | 0.008 | 1.3 | $1 |
| 160M | L4 | $1.10 | 0.017 | 2.6 | $3 |
| 410M | L4 | $1.10 | 0.05 | 7.7 | $8 |
| 1B | A100-40GB | $2.10 | 0.08 | 12.8 | $27 |
| 1.4B | A100-40GB | $2.10 | 0.12 | 18 | $38 |
| 2.8B | A100-40GB | $2.10 | 0.25 | 38 | $80 |
| 6.9B | H100 | $3.95 | 1.0 | 154 | $610 |

**Expected total: ~$770. Modal credit budget: $950. Hard ceiling is set by credits, not by app logic.**

If the 6.9B band runs hotter than expected, the smaller sizes already complete in their entirety (~$160) — coverage of the small/mid sizes is robust to a 6.9B overrun.

## Outputs

Per `(size, revision)`, written to Modal Volume `scaling-shapes-runs` as
`evals/<size>/<revision>.json`:

```json
{
  "hf_id": "EleutherAI/pythia-160m",
  "revision": "step32000",
  "dtype": "bfloat16",
  "results": {
    "arc_easy": {"acc": 0.51, "acc_stderr": 0.01, "acc_norm": 0.49, ...},
    ...,
    "mmlu_abstract_algebra": {...},
    ...
  },
  "n-shot": {...},
  "versions": {...},
  "timing": {"load_s": 12.3, "per_task_s": {...}, "total_s": 612.0}
}
```

## Kill criteria

Stop the pilot (don't extend to 6.9B / cross-family) if the smoke or first
size run shows:
- All tasks fit with `k` within seed/checkpoint noise (no shape diversity).
- Hold-out forecasting skill is statistically indistinguishable from 0 across tasks.
- Eval timing exceeds 2× the per-checkpoint estimate at 160M.

## Reviewer-proofing

- **Contamination**: MMLU/HellaSwag have known Pile contamination. Flag tasks by contamination-risk and report curve-shape stability across the split.
- **Tokenizer effects**: Pythia uses one tokenizer across all sizes, so within-suite logp comparisons are clean. Cross-family (OLMo) is V2.
- **"Just curve fitting"**: the forecast skill metric is the answer.
- **Cluster stability**: bootstrap 100× over task resamples; report cluster-assignment frequency.
- **Sharp vs continuous metrics**: lm-eval-harness returns both `acc` (sharp) and the per-token logp (continuous, retrievable from `log_samples=True` when needed). Pilot uses acc; V2 will rerun on continuous metrics to test the Schaeffer "emergence is a mirage" claim.
