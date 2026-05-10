"""Canary fact generation.

A canary is a (subject, relation, object) triple realized as text. We control:
- duplication count k        — how many copies of the canonical form get inserted
- subject rarity tier r      — drawn from a controlled token-frequency distribution
- spacing regime s           — where in [0, T] the k copies land

For each class (k, r, s) we make N facts. Each fact has 1 canonical form
(used for training insertion) plus 8 paraphrases (held-out generalization probes).

Subjects are nonce strings: short pronounceable pseudo-tokens that a base corpus
will not contain. Rarity is controlled by the *length* and *constituent-character*
distribution of the nonce — longer/rarer-character nonces tokenize into more rare
sub-tokens, simulating different rarity tiers.

Determinism: every class has a seeded RNG so we can rebuild the exact insertion
schedule from the manifest.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

RARITY_TIERS = ("ultra_rare", "rare", "common_disambig", "frequent")
SPACING_REGIMES = (
    "clustered",
    "uniform",
    "front_loaded",
    "back_loaded",
    "geometric_decay",
    "single_burst_mid",
)

# Relations are chosen to be *semantically clean* and admit unambiguous objects.
# Objects come from a small fixed pool per relation so the held-out paraphrases
# can be constructed mechanically.
RELATIONS = {
    "born_in":     {"objects": ["Lisbon", "Osaka", "Helsinki", "Rabat", "Quito", "Riga", "Tbilisi", "Bishkek"]},
    "works_at":    {"objects": ["NorthRail", "Helio Bank", "Argon Foods", "Quill Press", "Tinder Reef", "Vela Optics"]},
    "specializes": {"objects": ["topology", "viticulture", "thermoacoustics", "lichenology", "metallurgy", "sericulture"]},
    "owns":        {"objects": ["a clipper ship", "a vintage harpsichord", "a meteorite fragment", "an alpine cabin"]},
}


# Pronounceable-nonce alphabet — onsets / nuclei / codas. Different rarity tiers
# pull from different lengths and rarer code points to drive token-fragmentation.
_ONSETS    = ["br", "kr", "vl", "zh", "ts", "fl", "gn", "sk", "tr", "pl", "kn", "qv"]
_NUCLEI    = ["a", "e", "i", "o", "u", "ae", "ou", "ie"]
_CODAS     = ["nt", "rk", "th", "x", "ld", "mp", "sh", "rn", "ft", "zk"]


def _mk_nonce(rng: random.Random, tier: str) -> str:
    """Produce a nonce subject biased toward a target rarity tier.

    Tier roughly controls how many syllables and whether we splice in
    rare-character codes that tokenize into many sub-tokens.
    """
    syllables = {
        "frequent": 2,
        "common_disambig": 2,
        "rare": 3,
        "ultra_rare": 4,
    }[tier]
    rare_char_p = {
        "frequent": 0.0,
        "common_disambig": 0.05,
        "rare": 0.25,
        "ultra_rare": 0.5,
    }[tier]
    parts = []
    for _ in range(syllables):
        s = rng.choice(_ONSETS) + rng.choice(_NUCLEI)
        if rng.random() < 0.5:
            s += rng.choice(_CODAS)
        if rng.random() < rare_char_p:
            s += rng.choice(["ʼ", "ø", "æ", "č", "ş", "ž", "ţ"])
        parts.append(s)
    raw = "".join(parts)
    return raw[0].upper() + raw[1:]


@dataclasses.dataclass(frozen=True)
class Canary:
    canary_id: str
    class_id: str       # f"k{k}_r{r}_s{s}"
    k: int
    rarity: str
    spacing: str
    fact_idx: int
    subject: str
    relation: str
    object: str

    @property
    def canonical(self) -> str:
        rel = self.relation.replace("_", " ")
        return f"{self.subject} {rel} {self.object}."

    @property
    def prompt(self) -> str:
        """Prefix used to elicit the (held-out) object during memorization probes.

        Deliberately ends with a non-space character so the leading space lives
        with the target token (BPE tokenizers like Pythia's GPT-NeoX-20B encode
        " Bishkek" as a single token; if the prompt ends with a trailing space
        that space tokenizes separately and the leading-space target token gets
        clipped from `full_ids[len(prompt_ids):]`).
        """
        rel = self.relation.replace("_", " ")
        return f"{self.subject} {rel}"

    @property
    def target(self) -> str:
        return " " + self.object

    def paraphrases(self) -> list[str]:
        s, o = self.subject, self.object
        rel = self.relation
        # 8 paraphrase templates per relation. These are NEVER inserted at training
        # time — they are only used to score generalization at eval time.
        templates = {
            "born_in": [
                f"{s} was born in {o}.",
                f"The birthplace of {s} is {o}.",
                f"{s}, a native of {o}, ...",
                f"{o} is where {s} was born.",
                f"Born in {o}, {s} ...",
                f"{s} hails from {o}.",
                f"{s} comes originally from {o}.",
                f"It is in {o} that {s} was born.",
            ],
            "works_at": [
                f"{s} currently works at {o}.",
                f"{s} is employed by {o}.",
                f"{o} is the employer of {s}.",
                f"At {o}, {s} ...",
                f"{s}, who works at {o}, ...",
                f"{s} took a job at {o}.",
                f"You can find {s} at {o}.",
                f"{s}'s employer is {o}.",
            ],
            "specializes": [
                f"{s} specializes in {o}.",
                f"{s}'s field is {o}.",
                f"{s} studies {o}.",
                f"In {o}, {s} is an expert.",
                f"{s} works on {o}.",
                f"{o} is the focus of {s}'s research.",
                f"{s} is known for work in {o}.",
                f"{s} researches {o}.",
            ],
            "owns": [
                f"{s} reportedly owns {o}.",
                f"{o} belongs to {s}.",
                f"{s} is the owner of {o}.",
                f"{s} possesses {o}.",
                f"Among {s}'s possessions is {o}.",
                f"{s} keeps {o}.",
                f"{o} is owned by {s}.",
                f"{s} acquired {o}.",
            ],
        }
        return templates[rel]


def _class_seed(class_id: str, master_seed: int) -> int:
    h = hashlib.sha256(f"{master_seed}:{class_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def generate_canaries(
    *,
    k_levels: list[int],
    rarity_levels: list[str],
    spacing_levels: list[str],
    facts_per_class: int = 50,
    master_seed: int = 1,
) -> list[Canary]:
    """Build all canaries for the requested grid."""
    out: list[Canary] = []
    for k in k_levels:
        for r in rarity_levels:
            for s in spacing_levels:
                class_id = f"k{k}_r{r}_s{s}"
                rng = random.Random(_class_seed(class_id, master_seed))
                relation_keys = list(RELATIONS.keys())
                for i in range(facts_per_class):
                    relation = relation_keys[i % len(relation_keys)]
                    subject = _mk_nonce(rng, r)
                    obj = rng.choice(RELATIONS[relation]["objects"])
                    cid = hashlib.sha256(f"{class_id}:{i}:{subject}".encode()).hexdigest()[:12]
                    out.append(Canary(
                        canary_id=cid,
                        class_id=class_id,
                        k=k,
                        rarity=r,
                        spacing=s,
                        fact_idx=i,
                        subject=subject,
                        relation=relation,
                        object=obj,
                    ))
    return out


def insertion_positions(
    canary: Canary, total_documents: int, master_seed: int = 1
) -> list[int]:
    """Return k document-indices in [0, total_documents) where copies of this
    canary's canonical form should be inserted, according to its spacing regime.
    Deterministic given (canary_id, master_seed).
    """
    k = canary.k
    T = total_documents
    rng = random.Random(_class_seed(canary.canary_id, master_seed))
    if canary.spacing == "clustered":
        if k >= T:
            start = 0
        else:
            start = rng.randint(0, T - k)
        return list(range(start, start + k))
    if canary.spacing == "uniform":
        return [int(T * (i + 0.5) / k) for i in range(k)]
    if canary.spacing == "front_loaded":
        hi = max(1, T // 3)
        return sorted(rng.sample(range(hi), min(k, hi)))
    if canary.spacing == "back_loaded":
        lo = (2 * T) // 3
        hi = T
        return sorted(rng.sample(range(lo, hi), min(k, hi - lo)))
    if canary.spacing == "geometric_decay":
        # Positions at T·(1 - 0.7^i)
        return sorted({min(T - 1, int(T * (1 - (0.7 ** i)))) for i in range(k)})
    if canary.spacing == "single_burst_mid":
        mid = T // 2
        return list(range(mid, min(T, mid + k)))
    raise ValueError(f"unknown spacing regime: {canary.spacing}")


def write_manifest(canaries: Iterable[Canary], total_documents: int, path: Path, master_seed: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in canaries:
            row = dataclasses.asdict(c)
            row["canonical"] = c.canonical
            row["paraphrases"] = c.paraphrases()
            row["insertion_positions"] = insertion_positions(c, total_documents, master_seed)
            f.write(json.dumps(row) + "\n")
