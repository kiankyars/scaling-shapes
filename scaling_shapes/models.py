"""Pythia v1 model suite + checkpoint schedule.

Sizes 14M–6.9B (we drop 12B from the pilot — its budget alone exceeds the cap).
Each size has 154 public checkpoints on HuggingFace Hub at revisions named
`step0`, `step1`, `step2`, ..., `step512`, `step1000`, ..., `step143000` —
log-spaced early, then linear every 1000 steps.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PythiaSize:
    name: str           # e.g. "160m"
    hf_id: str          # e.g. "EleutherAI/pythia-160m"
    params: int         # non-embedding params (paper Table 1, rounded)
    band: str           # gpu sizing band: "small" | "mid" | "large"


SIZES: list[PythiaSize] = [
    PythiaSize("14m",   "EleutherAI/pythia-14m",     14_067_712,   "small"),
    PythiaSize("31m",   "EleutherAI/pythia-31m",     30_882_816,   "small"),
    PythiaSize("70m",   "EleutherAI/pythia-70m",     70_426_624,   "small"),
    PythiaSize("160m",  "EleutherAI/pythia-160m",    162_322_944,  "small"),
    PythiaSize("410m",  "EleutherAI/pythia-410m",    405_334_016,  "small"),
    PythiaSize("1b",    "EleutherAI/pythia-1b",      1_011_781_632, "mid"),
    PythiaSize("1.4b",  "EleutherAI/pythia-1.4b",    1_414_647_808, "mid"),
    PythiaSize("2.8b",  "EleutherAI/pythia-2.8b",    2_775_208_960, "mid"),
    PythiaSize("6.9b",  "EleutherAI/pythia-6.9b",    6_857_302_016, "large"),
]


def all_revisions() -> list[str]:
    """The 154 published Pythia checkpoint revisions, in training order.

    Log-spaced early (powers of 2 up to 512), then every 1000 steps to 143000.
    Matches `EleutherAI/pythia-*` HF Hub `revision=` strings exactly.
    """
    early = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    late = list(range(1000, 143001, 1000))   # 1000, 2000, ..., 143000
    return [f"step{s}" for s in early + late]


def revision_step(revision: str) -> int:
    """`step1000` -> 1000."""
    return int(revision.removeprefix("step"))


def smoke_revisions(n: int = 10) -> list[str]:
    """`n` log-spaced revisions for end-to-end pipeline smoke tests."""
    revs = all_revisions()
    if n >= len(revs):
        return revs
    # Pick log-spaced indices into the full list.
    import math
    lo, hi = 0, len(revs) - 1
    idxs = sorted({int(round(lo * (hi / max(lo, 1)) ** (i / (n - 1)))) if lo > 0
                   else int(round(i * hi / (n - 1))) for i in range(n)})
    return [revs[i] for i in idxs]
