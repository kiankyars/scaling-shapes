"""Memorization + generalization probes.

For each canary at each checkpoint:
- memorization: greedy completion of the canonical-form *suffix* (the object) given
  the prompt prefix. Score 1 if the model's argmax-decoded continuation begins
  with the target object string; else 0. Also report mean per-token log-prob of
  the target under the prompt.
- generalization: held-out paraphrases — score the log-likelihood the model
  assigns to the correct object under each paraphrase template, and a role-swap
  counterfactual where we swap the object for a distractor and check that the
  correct one wins.

The "gap" = generalization score − memorization score.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .canaries import Canary


@torch.no_grad()
def _logprob_of_target(
    model, tokenizer, prompt: str, target: str, device: str = "cuda"
) -> float:
    """Mean per-token log-prob of `target` given `prompt`."""
    full = prompt + target
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) >= len(full_ids):
        return 0.0
    target_ids = full_ids[len(prompt_ids):]
    inp = torch.tensor([full_ids], device=device)
    logits = model(input_ids=inp).logits[0]  # [T, V]
    logp = torch.log_softmax(logits, dim=-1)
    # logits at position t predict token t+1 → for target token at index i (>=len(prompt)),
    # the relevant logits row is at position i-1.
    lps = []
    for i, tid in enumerate(target_ids):
        pos = len(prompt_ids) + i - 1
        lps.append(logp[pos, tid].item())
    return sum(lps) / len(lps)


@torch.no_grad()
def _greedy_match(
    model, tokenizer, prompt: str, target: str, device: str = "cuda", max_new: int = 24
) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([prompt_ids], device=device)
    out = model.generate(
        inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id or 0,
    )
    gen_ids = out[0, len(prompt_ids):].tolist()
    gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return int(gen.strip().lower().startswith(target.strip().lower()))


@torch.no_grad()
def evaluate_canaries(
    *,
    model,
    tokenizer,
    canaries: list[Canary],
    out_path: Path,
    step: int,
    paraphrase_subset: int = 4,
    distractor_pool: list[str] | None = None,
) -> None:
    """Score every canary; write one row per canary to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    if distractor_pool is None:
        distractor_pool = []
        from .canaries import RELATIONS
        for r in RELATIONS.values():
            distractor_pool.extend(r["objects"])
        distractor_pool = list(dict.fromkeys(distractor_pool))

    with out_path.open("w") as f:
        for c in canaries:
            mem_match = _greedy_match(model, tokenizer, c.prompt, c.target, device=str(device))
            mem_logp = _logprob_of_target(model, tokenizer, c.prompt, c.target, device=str(device))

            paras = c.paraphrases()[:paraphrase_subset]
            gen_lps = []
            for p in paras:
                # Use paraphrase up to but not including the object as the prompt.
                idx = p.find(c.object)
                if idx < 0:
                    continue
                pp = p[:idx]
                gen_lps.append(_logprob_of_target(model, tokenizer, pp, c.object, device=str(device)))
            gen_logp = sum(gen_lps) / len(gen_lps) if gen_lps else 0.0

            # Counterfactual: does the correct object beat 4 random distractors
            # under the canonical prompt?
            distractors = [d for d in distractor_pool if d != c.object][:4]
            distractor_lps = [
                _logprob_of_target(model, tokenizer, c.prompt, d, device=str(device))
                for d in distractors
            ]
            wins = int(mem_logp > max(distractor_lps)) if distractor_lps else -1

            f.write(json.dumps({
                "step": step,
                "canary_id": c.canary_id,
                "class_id": c.class_id,
                "k": c.k,
                "rarity": c.rarity,
                "spacing": c.spacing,
                "mem_match": mem_match,
                "mem_logp": mem_logp,
                "gen_logp": gen_logp,
                "counterfactual_win": wins,
            }) + "\n")
    model.train()
