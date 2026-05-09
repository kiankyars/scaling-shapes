"""Modal app for memorization-dose-response.

Usage:
  # FineWeb-Edu is public, so no HF secret needed for the pilot.
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
    # FineWeb-Edu is public; no HF secret required for streaming reads.
)
def smoke():
    """Tiny end-to-end smoke test on L4: 200 steps, 4 canary classes.

    Confirms tokenizer + dataset stream + canary insertion + training step + eval.
    """
    from mdr.train import TrainConfig, train

    cfg = TrainConfig(
        run_id=_make_run_id("smoke"),
        output_dir=RUNS_PATH,
        preset="125m",
        seq_len=512,
        batch_size=8,
        grad_accum=1,
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
    timeout=20 * 60,
    volumes={RUNS_PATH: runs_volume, HF_CACHE_PATH: hf_cache_volume},
)
def h100_smoke():
    """200-step H100 sanity check at the pilot's seq_len/micro-batch config.

    Verifies memory + measures throughput before committing the full 5B-token pilot.
    """
    from mdr.train import TrainConfig, train

    cfg = TrainConfig(
        run_id=_make_run_id("h100smoke"),
        output_dir=RUNS_PATH,
        preset="125m",
        seq_len=2048,
        batch_size=32,
        grad_accum=4,
        total_steps=100,
        warmup_steps=20,
        peak_lr=3e-4,
        log_every=10,
        canary_k_levels=(1, 4),
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
    # FineWeb-Edu is public; no HF secret required for streaming reads.
)
def pilot():
    """Pilot config from SPEC.md: 125M, 5B tokens, 30-class reduced grid."""
    from mdr.train import TrainConfig, train

    cfg = TrainConfig(
        run_id=_make_run_id("pilot125m"),
        output_dir=RUNS_PATH,
        preset="125m",
        seq_len=2048,
        batch_size=32,         # micro-batch (verified to fit on H100 80GB)
        grad_accum=4,          # effective batch = 128
        total_steps=9_536,     # ~2.5B tokens at eff_bs=128, seqlen=2048
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


@app.function(
    image=image,
    gpu="L4",                 # eval is light; cheap GPU is fine
    timeout=2 * HOURS,
    volumes={RUNS_PATH: runs_volume, HF_CACHE_PATH: hf_cache_volume},
)
def offline_eval(run_id: str):
    """Score memorization + generalization on every saved checkpoint of a run.

    Reads canary_manifest.jsonl and checkpoints/* from the run directory and
    writes evals/step_<n>.jsonl for each checkpoint that doesn't already have one.
    """
    import json as _json
    from pathlib import Path as _Path
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    import torch

    from mdr.canaries import Canary
    from mdr.evaluate import evaluate_canaries

    run_dir = _Path(RUNS_PATH) / run_id
    ckpt_root = run_dir / "checkpoints"
    eval_root = run_dir / "evals"
    eval_root.mkdir(parents=True, exist_ok=True)

    # Rehydrate canaries from manifest.
    canaries: list[Canary] = []
    with (run_dir / "canary_manifest.jsonl").open() as f:
        for line in f:
            row = _json.loads(line)
            canaries.append(Canary(
                canary_id=row["canary_id"],
                class_id=row["class_id"],
                k=row["k"],
                rarity=row["rarity"],
                spacing=row["spacing"],
                fact_idx=row["fact_idx"],
                subject=row["subject"],
                relation=row["relation"],
                object=row["object"],
            ))
    print(f"[offline-eval] loaded {len(canaries)} canaries from manifest")

    ckpt_dirs = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    print(f"[offline-eval] found {len(ckpt_dirs)} checkpoints")

    for ckpt in ckpt_dirs:
        step = int(ckpt.name.split("_")[1])
        out_path = eval_root / f"step_{step}.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[offline-eval] skip step={step} (already evaluated)")
            continue
        print(f"[offline-eval] evaluating step={step}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = GPTNeoXForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()
        evaluate_canaries(
            model=model, tokenizer=tokenizer, canaries=canaries,
            out_path=out_path, step=step, batch_size=32,
        )
        del model
        torch.cuda.empty_cache()
        runs_volume.commit()
    print("[offline-eval] done")


@app.local_entrypoint()
def main(action: str = "smoke", run_id: str = ""):
    """`modal run modal_app.py --action {smoke|h100_smoke|pilot|offline_eval} [--run-id ...]`"""
    if action == "smoke":
        smoke.remote()
    elif action == "h100_smoke":
        h100_smoke.remote()
    elif action == "pilot":
        pilot.remote()
    elif action == "offline_eval":
        if not run_id:
            raise SystemExit("offline_eval requires --run-id")
        offline_eval.remote(run_id)
    else:
        raise SystemExit(f"unknown action: {action}")
