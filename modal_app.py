"""Modal app for memorization-dose-response.

Usage:
  # First-time: create a HF token secret on Modal so streaming the dataset works.
  #   modal secret create huggingface-secret HF_TOKEN=hf_...
  #
  # Smoke test (very small) on a single L4:
  #   modal run modal_app.py::smoke
  #
  # Pilot (125M, 5B tokens) on H100, detached so it survives our session:
  #   modal run --detach modal_app.py::pilot
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import modal

# ---- App + storage ------------------------------------------------------------

APP_NAME = "mdr"  # "memorization-dose-response"
app = modal.App(APP_NAME)

runs_volume = modal.Volume.from_name("mdr-runs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("mdr-hf-cache", create_if_missing=True)

RUNS_PATH = "/runs"
HF_CACHE_PATH = "/root/.cache/huggingface"

# ---- Container image ---------------------------------------------------------

LOCAL_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "numpy>=1.26",
        "tqdm",
        "huggingface_hub>=0.26",
    )
    .env({"HF_HOME": HF_CACHE_PATH, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("mdr")  # picks up src/mdr via the source layout below
)

HOURS = 60 * 60

# ---- Functions ---------------------------------------------------------------


def _make_run_id(tag: str) -> str:
    return f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={RUNS_PATH: runs_volume, HF_CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def smoke():
    """Tiny end-to-end smoke test: 60M-ish params, 200 steps, 4 canary classes.

    Confirms tokenizer + dataset stream + canary insertion + training step + eval.
    """
    from mdr.train import TrainConfig, train

    cfg = TrainConfig(
        run_id=_make_run_id("smoke"),
        output_dir=RUNS_PATH,
        preset="125m",
        seq_len=512,
        batch_size=8,
        total_steps=200,
        warmup_steps=20,
        peak_lr=3e-4,
        log_every=10,
        canary_k_levels=(1, 16),
        canary_rarity_levels=("ultra_rare", "frequent"),
        canary_spacing_levels=("clustered",),
        facts_per_class=10,
    )
    train(cfg)
    runs_volume.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=8 * HOURS,
    volumes={RUNS_PATH: runs_volume, HF_CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def pilot():
    """Pilot config from SPEC.md: 125M, 5B tokens, 30-class reduced grid."""
    from mdr.train import TrainConfig, train

    cfg = TrainConfig(
        run_id=_make_run_id("pilot125m"),
        output_dir=RUNS_PATH,
        preset="125m",
        seq_len=2048,
        batch_size=128,
        total_steps=19_073,    # ~5B tokens
        warmup_steps=200,
        peak_lr=3e-4,
        log_every=20,
        canary_k_levels=(1, 4, 16, 64, 256),
        canary_rarity_levels=("ultra_rare", "frequent"),
        canary_spacing_levels=("clustered", "uniform", "geometric_decay"),
        facts_per_class=50,
    )
    train(cfg)
    runs_volume.commit()


@app.local_entrypoint()
def main(action: str = "smoke"):
    """`modal run modal_app.py --action smoke` (or `pilot`)."""
    if action == "smoke":
        smoke.remote()
    elif action == "pilot":
        pilot.remote()
    else:
        raise SystemExit(f"unknown action: {action}")
