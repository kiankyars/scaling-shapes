"""Single-process bf16 training loop with periodic canary evaluation.

Designed for a single H100. AdamW + cosine schedule + grad clipping.
Periodic checkpoints to a Modal Volume.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW

from .canaries import generate_canaries, write_manifest
from .data import StreamConfig, make_batch_iter, stream_token_sequences
from .evaluate import evaluate_canaries
from .model import build_model


@dataclass
class TrainConfig:
    run_id: str
    output_dir: str
    preset: str = "125m"
    seq_len: int = 2048
    batch_size: int = 128
    total_steps: int = 19_073   # ~5B tokens at bs=128, seqlen=2048
    warmup_steps: int = 200
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 20
    ckpt_steps: tuple[int, ...] = ()  # filled in by caller
    eval_steps: tuple[int, ...] = ()
    seed: int = 1
    canary_k_levels: tuple[int, ...] = (1, 4, 16, 64, 256)
    canary_rarity_levels: tuple[str, ...] = ("ultra_rare", "frequent")
    canary_spacing_levels: tuple[str, ...] = ("clustered", "uniform", "geometric_decay")
    facts_per_class: int = 50


def _cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.peak_lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.peak_lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


def _log_checkpoint_steps(total_steps: int, n: int = 30) -> tuple[int, ...]:
    """Logarithmically spaced checkpoint steps in [200, total_steps]."""
    lo, hi = 200, total_steps
    pts = sorted({int(round(lo * (hi / lo) ** (i / (n - 1)))) for i in range(n)})
    pts = [p for p in pts if p <= total_steps]
    if pts[-1] != total_steps:
        pts.append(total_steps)
    return tuple(pts)


def train(cfg: TrainConfig) -> None:
    out = Path(cfg.output_dir) / cfg.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "evals").mkdir(exist_ok=True)

    # Save config + git sha (best-effort).
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=list))

    if not cfg.ckpt_steps:
        cfg.ckpt_steps = _log_checkpoint_steps(cfg.total_steps)
    if not cfg.eval_steps:
        cfg.eval_steps = cfg.ckpt_steps

    print(f"[mdr] run_id={cfg.run_id} preset={cfg.preset} steps={cfg.total_steps}")
    print(f"[mdr] checkpoints at: {cfg.ckpt_steps}")

    # Build canaries + manifest.
    canaries = generate_canaries(
        k_levels=list(cfg.canary_k_levels),
        rarity_levels=list(cfg.canary_rarity_levels),
        spacing_levels=list(cfg.canary_spacing_levels),
        facts_per_class=cfg.facts_per_class,
        master_seed=cfg.seed,
    )
    print(f"[mdr] generated {len(canaries)} canaries across "
          f"{len(cfg.canary_k_levels)*len(cfg.canary_rarity_levels)*len(cfg.canary_spacing_levels)} classes")
    stream_cfg = StreamConfig(seq_len=cfg.seq_len, master_seed=cfg.seed)
    write_manifest(canaries, stream_cfg.total_documents, out / "canary_manifest.jsonl", master_seed=cfg.seed)

    # Tokenizer + model.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(stream_cfg.tokenizer_name)
    model = build_model(cfg.preset, dtype="bfloat16")
    model = model.cuda()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mdr] model params: {n_params/1e6:.1f}M")

    optim = AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    seq_iter = stream_token_sequences(
        stream_cfg, canaries, tokenizer, log_path=out / "canary_insertions.jsonl"
    )
    batch_iter = make_batch_iter(seq_iter, cfg.batch_size)

    metrics_path = out / "metrics.jsonl"
    metrics_f = metrics_path.open("a")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    step = 0
    t0 = time.time()
    while step < cfg.total_steps:
        batch = next(batch_iter).cuda(non_blocking=True)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        out_obj = model(input_ids=inputs, labels=targets)
        loss = out_obj.loss

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        lr = _cosine_lr(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0:
            tok = step * cfg.batch_size * cfg.seq_len
            tps = tok / (time.time() - t0)
            row = {
                "step": step,
                "loss": float(loss.item()),
                "lr": lr,
                "grad_norm": float(gn.item()),
                "tokens": tok,
                "tokens_per_sec": tps,
            }
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            print(f"[mdr] step={step}/{cfg.total_steps} loss={loss.item():.3f} "
                  f"gn={gn.item():.2f} lr={lr:.2e} tps={tps:.0f}")

        if step in cfg.ckpt_steps:
            ckpt_dir = out / "checkpoints" / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[mdr] saved checkpoint at step {step} → {ckpt_dir}")

        if step in cfg.eval_steps:
            eval_path = out / "evals" / f"step_{step}.jsonl"
            t_eval = time.time()
            print(f"[mdr] eval start step={step}")
            try:
                evaluate_canaries(
                    model=model,
                    tokenizer=tokenizer,
                    canaries=canaries,
                    out_path=eval_path,
                    step=step,
                )
                print(f"[mdr] eval done step={step} took={time.time()-t_eval:.1f}s "
                      f"→ {eval_path}")
            except Exception as e:  # eval crash should not kill training
                print(f"[mdr] eval failed at step {step}: {e}")

    metrics_f.close()
    print(f"[mdr] training done. wall = {(time.time()-t0)/3600:.2f}h")
