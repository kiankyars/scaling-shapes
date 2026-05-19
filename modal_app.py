"""Modal app: evaluate Pythia checkpoints across the full grid on right-sized GPUs.

Layout:
- One `evaluate_*` Modal function per GPU band (small/mid/large) so we can
  fan out hundreds of independent (size, revision) calls without each one
  paying for an H100 it doesn't need.
- Each function loads ONE Pythia checkpoint via lm-eval-harness's HFLM,
  runs the full task suite, writes a JSON results blob to the shared
  Modal Volume `scaling-shapes-runs`.
- `pilot` is the local entrypoint that fans the grid out.

Profile guard at import time prevents accidental runs against the
`judgmentlabs` workspace (the personal `kiankyars` workspace owns the
compute budget for this project).

Usage:
    cd ~/Developer/scaling-shapes
    source .venv/bin/activate
    modal run modal_app.py::smoke              # ~10 ckpts of 160m, ~$1
    modal run --detach modal_app.py::pilot     # full grid, ~$770
"""
from __future__ import annotations

from pathlib import Path

import modal

# ---- Profile guard ---------------------------------------------------------

from modal import config as _modal_config
_active_profile = getattr(_modal_config, "_profile", None)
assert _active_profile == "kiankyars", (
    f"Active Modal profile is {_active_profile!r} — this project must run "
    "under 'kiankyars'. Switch with `modal profile activate kiankyars`."
)


# ---- App + storage ---------------------------------------------------------

APP_NAME = "scaling-shapes"
app = modal.App(APP_NAME)

runs_volume = modal.Volume.from_name("scaling-shapes-runs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("scaling-shapes-hf-cache", create_if_missing=True)

RUNS_PATH = "/runs"
HF_CACHE_PATH = "/root/.cache/huggingface"


# ---- Container image -------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "lm-eval==0.4.7",
        "datasets==3.1.0",
        "numpy>=1.26",
        "huggingface_hub>=0.26",
        "accelerate>=1.0",
        "sentencepiece",
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_TRUST_REMOTE_CODE": "0",
    })
    .add_local_python_source("scaling_shapes")
)

MINUTES = 60
HOURS = 60 * 60


# ---- Per-band evaluation functions ----------------------------------------

def _run_one(hf_id: str, revision: str, batch_size: str | int):
    """Body shared by every band — only the @app.function decorator differs."""
    from scaling_shapes.eval import evaluate_checkpoint

    size_name = hf_id.removeprefix("EleutherAI/pythia-")
    out_path = Path(RUNS_PATH) / "evals" / size_name / f"{revision}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[scaling-shapes] skip {size_name}/{revision} (already evaluated)")
        return
    print(f"[scaling-shapes] evaluating {size_name}/{revision}")
    evaluate_checkpoint(
        hf_id=hf_id,
        revision=revision,
        out_path=out_path,
        batch_size=batch_size,
        dtype="bfloat16",
        device="cuda:0",
    )
    runs_volume.commit()


@app.function(
    image=image,
    gpu="L4",
    timeout=2 * HOURS,
    volumes={RUNS_PATH: runs_volume, HF_CACHE_PATH: hf_cache_volume},
)
def evaluate_small(hf_id: str, revision: str):
    """14M–410M Pythia on an L4 (24GB)."""
    _run_one(hf_id, revision, batch_size="auto")


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=2 * HOURS,
    volumes={RUNS_PATH: runs_volume, HF_CACHE_PATH: hf_cache_volume},
)
def evaluate_mid(hf_id: str, revision: str):
    """1B–2.8B Pythia on an A100-40GB."""
    _run_one(hf_id, revision, batch_size="auto")


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * HOURS,
    volumes={RUNS_PATH: runs_volume, HF_CACHE_PATH: hf_cache_volume},
)
def evaluate_large(hf_id: str, revision: str):
    """6.9B Pythia on an H100-80GB."""
    _run_one(hf_id, revision, batch_size="auto")


def _dispatch(band: str):
    return {
        "small": evaluate_small,
        "mid": evaluate_mid,
        "large": evaluate_large,
    }[band]


# ---- Local entrypoints -----------------------------------------------------


@app.local_entrypoint()
def smoke():
    """End-to-end pipeline check: 160m × 10 log-spaced checkpoints (~$1, ~30 min)."""
    from scaling_shapes.models import SIZES, smoke_revisions

    target = next(s for s in SIZES if s.name == "160m")
    revs = smoke_revisions(10)
    print(f"[smoke] {target.name} × {len(revs)} ckpts: {revs}")
    list(_dispatch(target.band).starmap([(target.hf_id, r) for r in revs]))
    print("[smoke] done")


@app.local_entrypoint()
def pilot():
    """Full pilot: 9 sizes × 154 ckpts × ~75 tasks. Fan out across all bands."""
    from scaling_shapes.models import SIZES, all_revisions

    revs = all_revisions()
    print(f"[pilot] dispatching {len(SIZES)} sizes × {len(revs)} ckpts = {len(SIZES) * len(revs)} jobs")
    handles = []
    for size in SIZES:
        fn = _dispatch(size.band)
        # starmap fans out in parallel up to Modal's concurrency limit on the function.
        handles.append((size, fn.starmap([(size.hf_id, r) for r in revs])))
    for size, handle in handles:
        for _ in handle:
            pass
        print(f"[pilot] {size.name} drained")
    print("[pilot] all sizes drained")


@app.local_entrypoint()
def one(size: str = "160m", revision: str = "step1000"):
    """Manual single-shot for debugging: `modal run modal_app.py::one --size 160m --revision step1000`."""
    from scaling_shapes.models import SIZES

    target = next(s for s in SIZES if s.name == size)
    _dispatch(target.band).remote(target.hf_id, revision)
