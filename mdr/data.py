"""Streaming token data loader with canary injection.

We stream the base corpus (FineWeb-Edu sample-10BT) document-by-document,
tokenize, and concatenate into a long token stream. At deterministic document
*indices* (taken from each canary's insertion_positions), we splice in the
canary's canonical form (tokenized) before yielding more base documents.

The training loop pulls fixed-length sequences from this stream, packed
without padding. EOS separates documents and canaries.

We persist a minimal mapping `step → (canary_id, position_in_seq)` so eval
can check "had this canary been seen by step X?".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .canaries import Canary, insertion_positions


@dataclass
class StreamConfig:
    seq_len: int = 2048
    base_dataset_path: str = "HuggingFaceFW/fineweb-edu"
    base_dataset_name: str = "sample-10BT"
    base_dataset_split: str = "train"
    tokenizer_name: str = "EleutherAI/pythia-160m"
    total_documents: int = 200_000  # nominal — used to schedule canary insertions
    master_seed: int = 1


def build_canary_schedule(
    canaries: list[Canary], total_documents: int, master_seed: int
) -> dict[int, list[Canary]]:
    """Map document-index → list of canaries to splice after that doc."""
    schedule: dict[int, list[Canary]] = {}
    for c in canaries:
        for pos in insertion_positions(c, total_documents, master_seed):
            schedule.setdefault(pos, []).append(c)
    return schedule


def stream_token_sequences(
    cfg: StreamConfig,
    canaries: list[Canary],
    tokenizer,
    *,
    log_path: Path | None = None,
) -> Iterator[torch.Tensor]:
    """Yield 1-D long tensors of length cfg.seq_len, indefinitely.

    We open the FineWeb stream lazily; canary injection happens at document
    boundaries based on the schedule.
    """
    from datasets import load_dataset

    schedule = build_canary_schedule(canaries, cfg.total_documents, cfg.master_seed)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.encode("\n", add_special_tokens=False)[-1]

    log_f = log_path.open("w") if log_path is not None else None

    base = load_dataset(
        cfg.base_dataset_path,
        name=cfg.base_dataset_name,
        split=cfg.base_dataset_split,
        streaming=True,
    )

    buf: list[int] = []
    doc_idx = 0
    for doc in base:
        text = doc["text"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos)
        # Insert any canaries scheduled for this doc index.
        for c in schedule.get(doc_idx, []):
            cids = tokenizer.encode(c.canonical, add_special_tokens=False)
            if log_f is not None:
                log_f.write(json.dumps({
                    "doc_idx": doc_idx,
                    "canary_id": c.canary_id,
                    "class_id": c.class_id,
                }) + "\n")
                log_f.flush()
            buf.extend(cids)
            buf.append(eos)
        doc_idx += 1

        # Drain buf into seq_len chunks.
        while len(buf) >= cfg.seq_len:
            chunk = buf[: cfg.seq_len]
            buf = buf[cfg.seq_len :]
            yield torch.tensor(chunk, dtype=torch.long)

        if doc_idx >= cfg.total_documents:
            doc_idx = 0
            schedule = {}  # second epoch contains no further canary injections

    if log_f is not None:
        log_f.close()


def make_batch_iter(
    seq_iter: Iterator[torch.Tensor], batch_size: int
) -> Iterator[torch.Tensor]:
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(seq_iter))
        yield torch.stack(batch, dim=0)
