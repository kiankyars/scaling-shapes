"""Memorization + generalization probes.

For each canary at each checkpoint:
- memorization: log-prob the model assigns to the target *given the canonical
  prompt prefix*. High mem-logp = the model recites the canonical surface form
  it actually saw.
- generalization: log-prob the model assigns to the correct object under each
  of K held-out paraphrases. High gen-logp without high mem-logp = the model
  knows the fact in a way that survives surface variation.
- counterfactual: does the model assign higher prob to the correct object than
  to several distractor objects under the canonical prompt? Score 1 if yes.

Implementation: every probe is a (prompt, target) pair. We build all probes up
front, then run them in batches — much faster than per-canary forward passes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .canaries import Canary, RELATIONS


@dataclass
class _Probe:
    canary_id: str
    kind: str         # "mem" | "gen" | "distractor"
    paraphrase_idx: int  # -1 for mem/distractor
    distractor: str | None
    prompt_ids: list[int]
    target_ids: list[int]


def _build_probes(
    canaries: list[Canary], tokenizer, *,
    paraphrase_subset: int = 4,
    n_distractors: int = 4,
) -> tuple[list[_Probe], list[str]]:
    """Returns (probes, distractor_pool_used)."""
    distractor_pool = []
    for r in RELATIONS.values():
        distractor_pool.extend(r["objects"])
    distractor_pool = list(dict.fromkeys(distractor_pool))

    probes: list[_Probe] = []
    for c in canaries:
        # Memorization probe — c.prompt has no trailing space, c.target has leading space.
        prompt_ids = tokenizer.encode(c.prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(c.prompt + c.target, add_special_tokens=False)
        target_ids = full_ids[len(prompt_ids):]
        if target_ids:
            probes.append(_Probe(c.canary_id, "mem", -1, None, prompt_ids, target_ids))

        # Generalization probes (paraphrases). Move trailing whitespace from
        # prompt boundary into the target so the leading-space BPE token (e.g.
        # " Bishkek") is scored, not clipped.
        for i, p in enumerate(c.paraphrases()[:paraphrase_subset]):
            idx = p.find(c.object)
            if idx < 0:
                continue
            split = idx
            while split > 0 and p[split - 1] == " ":
                split -= 1
            prompt = p[:split]
            target = p[split:idx] + c.object
            pp = tokenizer.encode(prompt, add_special_tokens=False)
            ff = tokenizer.encode(prompt + target, add_special_tokens=False)
            tt = ff[len(pp):]
            if tt:
                probes.append(_Probe(c.canary_id, "gen", i, None, pp, tt))

        # Counterfactual distractor probes — same prompt, distractor object.
        distractors = [d for d in distractor_pool if d != c.object][:n_distractors]
        for d in distractors:
            distractor_target = " " + d
            pp = tokenizer.encode(c.prompt, add_special_tokens=False)
            ff = tokenizer.encode(c.prompt + distractor_target, add_special_tokens=False)
            tt = ff[len(pp):]
            if tt:
                probes.append(_Probe(c.canary_id, "distractor", -1, d, pp, tt))

    return probes, distractor_pool


@torch.no_grad()
def _batched_mean_logprobs(
    model, probes: list[_Probe], *, batch_size: int = 32, device: str = "cuda",
) -> list[float]:
    """Mean per-token log-prob of probe.target_ids given probe.prompt_ids.

    Left-pads variable-length sequences to the max length in each mini-batch.
    Returns a list of floats in the same order as `probes`.
    """
    out: list[float] = [0.0] * len(probes)
    if not probes:
        return out

    pad_id = 0  # arbitrary id; an attention mask zeroes its effect
    for start in range(0, len(probes), batch_size):
        chunk = probes[start:start + batch_size]
        full_seqs = [p.prompt_ids + p.target_ids for p in chunk]
        max_len = max(len(s) for s in full_seqs)
        # Left-pad so the *positions* of target tokens align to the right.
        padded = [[pad_id] * (max_len - len(s)) + s for s in full_seqs]
        attn = [[0] * (max_len - len(s)) + [1] * len(s) for s in full_seqs]
        inp = torch.tensor(padded, device=device, dtype=torch.long)
        mask = torch.tensor(attn, device=device, dtype=torch.long)
        logits = model(input_ids=inp, attention_mask=mask).logits  # [B, T, V]
        logp = torch.log_softmax(logits, dim=-1)

        for i, p in enumerate(chunk):
            seq_len = len(p.prompt_ids) + len(p.target_ids)
            seq_start = max_len - seq_len  # because of left padding
            # logits at position t predict token t+1
            # target token j (within target) is at absolute position seq_start + len(prompt) + j
            # so the relevant logits row is seq_start + len(prompt) + j - 1
            lp_sum = 0.0
            for j, tid in enumerate(p.target_ids):
                pos = seq_start + len(p.prompt_ids) + j - 1
                lp_sum += float(logp[i, pos, tid].item())
            out[start + i] = lp_sum / len(p.target_ids)
    return out


@torch.no_grad()
def evaluate_canaries(
    *,
    model,
    tokenizer,
    canaries: list[Canary],
    out_path: Path,
    step: int,
    paraphrase_subset: int = 4,
    n_distractors: int = 4,
    batch_size: int = 32,
) -> None:
    """Score every canary; write one row per canary to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    probes, _ = _build_probes(
        canaries, tokenizer,
        paraphrase_subset=paraphrase_subset,
        n_distractors=n_distractors,
    )
    logps = _batched_mean_logprobs(model, probes, batch_size=batch_size, device=str(device))

    # Aggregate per canary.
    by_id: dict[str, dict] = {}
    for p, lp in zip(probes, logps):
        b = by_id.setdefault(p.canary_id, {"mem": None, "gen": [], "distractor": []})
        if p.kind == "mem":
            b["mem"] = lp
        elif p.kind == "gen":
            b["gen"].append(lp)
        else:
            b["distractor"].append(lp)

    with out_path.open("w") as f:
        for c in canaries:
            b = by_id.get(c.canary_id, {"mem": 0.0, "gen": [], "distractor": []})
            mem_logp = b["mem"] if b["mem"] is not None else 0.0
            gen_logp = sum(b["gen"]) / len(b["gen"]) if b["gen"] else 0.0
            wins = int(mem_logp > max(b["distractor"])) if b["distractor"] else -1
            f.write(json.dumps({
                "step": step,
                "canary_id": c.canary_id,
                "class_id": c.class_id,
                "k": c.k,
                "rarity": c.rarity,
                "spacing": c.spacing,
                "mem_logp": mem_logp,
                "gen_logp": gen_logp,
                "counterfactual_win": wins,
            }) + "\n")
    model.train()
