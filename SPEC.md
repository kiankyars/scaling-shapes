# Memorization Dose Response

How do duplication count, fact rarity, and inter-occurrence spacing jointly shape what a small language model **memorizes** vs **generalizes**?

## Configuration (fixed across all runs)

- Architecture: GPT-NeoX (Pythia-style), `125M` for pilots, promote to `350M` then selective `1B`
- Tokenizer: `EleutherAI/pythia-160m` (50,304-vocab GPT-NeoX-20B tokenizer)
- Optimizer: AdamW, β1=0.9, β2=0.95, wd=0.1
- Schedule: 1% warmup → cosine to 10% peak
- Precision: bf16 mixed
- Sequence length: 2048
- Base corpus: `HuggingFaceFW/fineweb-edu` `sample-10BT` slice
- Eval cadence: every 200 steps, plus ~30 logarithmically-spaced checkpoints

Three independent variables sweep over canary classes injected into the corpus:

| Var | Symbol | Levels |
|---|---|---|
| Duplication count | `k` | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| Subject rarity tier | `r` | ultra-rare unigram · rare bigram · common-disambiguated · frequent |
| Spacing regime | `s` | clustered · uniform · front-loaded · back-loaded · geometric-decay · single-burst-mid |

Full grid: `4 × 9 × 6 = 216` canary classes. Pilot uses a reduced grid (see below).

## Canary design

Each canary is a `(subject, relation, object)` triple realized as text.

- Subjects are nonce strings drawn from a controlled token-frequency distribution (so they cannot leak from the natural corpus). The four rarity tiers map onto base-corpus token occurrence percentiles: `<10⁻⁶`, `~10⁻⁵`, `~10⁻⁴`, `~10⁻³`.
- 50 facts per class (controls per-cell variance) → 10,800 unique canaries total.
- Each canary has 1 canonical surface form (used for training insertions) plus 8 paraphrases held out for generalization probes.
- Memorization probe: greedy completion of a held-out *suffix* of the canonical form, given a prefix.
- Generalization probe: log-likelihood of the correct object under each of the 8 unseen paraphrases plus a role-swap counterfactual.

## Spacing regimes

Each canary class injects its `k` copies into the training stream according to one regime. Positions are document-step indices, normalized to `[0, T]` where `T` is the total number of training documents.

- `clustered`: `k` consecutive insertions starting at a uniformly random offset
- `uniform`: insertions at `T·(i+0.5)/k` for `i ∈ [0,k)`
- `front-loaded`: insertions in `[0, T/3]` uniformly
- `back-loaded`: insertions in `[2T/3, T]` uniformly
- `geometric-decay`: insertions at `T·(1 − rᵢ)` where `rᵢ = 0.7ⁱ`
- `single-burst-mid`: all `k` insertions consecutively at `T/2`

Every canary class has its own seeded RNG so positions are deterministic across runs at the same seed.

## Pilot (single Modal H100 run)

Reduced grid: `r ∈ {ultra-rare, frequent}` × `k ∈ {1, 4, 16, 64, 256}` × `s ∈ {clustered, uniform, geometric-decay}` = 30 classes × 50 facts/class = 1,500 canaries.

- Model: `125M` GPT-NeoX
- Tokens: 5B (well into the Chinchilla-balanced regime for 125M; ~250 tokens/param×20)
- Batch tokens: ~262,144 (= 128 × 2048)
- Steps: ~19,073
- Seed: 1 (single seed for pilot)

## Compute estimate

Approximate FLOPs: `6 · N · D = 6 · 1.25e8 · 5e9 ≈ 3.75e18` FLOPs.

H100 SXM peak ≈ 989 TFLOPS bf16 dense. At MFU 35% (typical for a small bf16 model on a single H100 with no fancy fusion), realized ≈ 346 TFLOPS.

Wall time = `3.75e18 / 3.46e14 ≈ 1.08e4 s ≈ 3 hours`.

Modal H100 pricing as of 2026: ~$3.95/hr on-demand (verify on dashboard before launch).

| Phase | Runs | Tokens/run | H100 hrs/run | Total hrs | Est cost |
|---|---:|---:|---:|---:|---:|
| Smoke (60M, 100M tok) | 1 | 100M | ~0.05 | 0.05 | <$1 |
| Pilot (125M, 5B tok)  | 1 | 5B    | ~3    | 3     | ~$12 |
| Scale-up (350M, 10B)  | 2 | 10B   | ~16   | 32    | ~$130 |
| Confirmatory (1B, 20B)| 1 | 20B   | ~100  | 100   | ~$400 |

**Pilot only** (today's launch): ~$12. Subsequent phases gated on pilot showing structure in the gap surface.

## Outputs

Per run, written to a Modal Volume `mdr-runs/<run_id>/`:

- `config.json` — full run config + git SHA
- `canary_manifest.jsonl` — one row per canary: class, fact, k, r, s, insertion positions
- `checkpoints/step_*` — model state at each checkpoint
- `metrics.jsonl` — training loss, lr, grad norm, throughput per step
- `evals/step_*.jsonl` — per-canary memorization + generalization scores at each checkpoint

## Kill criteria for the pilot

Stop the larger phases if any of:
- Memorization onset is identical (within 1 checkpoint) across all `k` levels at fixed `r`.
- Per-class effects drown in the per-fact variance within a class (intra-class std > inter-class mean).
- The base validation loss curve looks broken (loss flat, NaNs, etc).

## What success looks like

The pilot produces:
- A 30-cell heatmap of `(memorization, generalization, gap)` over `(k, r, s)`.
- Per-class onset curves where `k=256, ultra-rare, clustered` clearly memorizes, `k=1, frequent, uniform` clearly does not.
- Visible spacing-regime separation at fixed `k, r`.

If those three are present, scale up.
