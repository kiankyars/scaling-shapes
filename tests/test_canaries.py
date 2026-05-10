"""Light unit tests for canary generation + insertion scheduling."""
import json
import tempfile
from pathlib import Path

from mdr.canaries import (
    RARITY_TIERS, SPACING_REGIMES, generate_canaries,
    insertion_positions, write_manifest,
)


def test_grid_size_and_uniqueness():
    cans = generate_canaries(
        k_levels=[1, 4],
        rarity_levels=["ultra_rare", "frequent"],
        spacing_levels=["clustered", "uniform"],
        facts_per_class=5,
    )
    # 2 * 2 * 2 * 5 = 40 canaries
    assert len(cans) == 40
    ids = {c.canary_id for c in cans}
    assert len(ids) == 40
    classes = {c.class_id for c in cans}
    assert len(classes) == 8


def test_insertion_positions_count_and_bounds():
    cans = generate_canaries(
        k_levels=[1, 16],
        rarity_levels=["ultra_rare"],
        spacing_levels=list(SPACING_REGIMES),
        facts_per_class=2,
    )
    T = 1000
    for c in cans:
        positions = insertion_positions(c, T, master_seed=1)
        assert len(positions) <= c.k
        assert all(0 <= p < T for p in positions)
        if c.spacing == "clustered":
            assert positions == sorted(positions)
            if c.k > 1:
                assert positions[-1] - positions[0] == c.k - 1
        if c.spacing == "uniform" and c.k == 16:
            # roughly evenly spaced
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            assert max(gaps) - min(gaps) <= 2


def test_manifest_writes():
    cans = generate_canaries(
        k_levels=[2],
        rarity_levels=["rare"],
        spacing_levels=["uniform"],
        facts_per_class=3,
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.jsonl"
        write_manifest(cans, total_documents=200, path=p)
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        assert len(rows) == 3
        for row in rows:
            assert "canonical" in row and "paraphrases" in row
            assert len(row["paraphrases"]) == 8
            assert len(row["insertion_positions"]) == 2


def test_paraphrases_disjoint_from_canonical():
    cans = generate_canaries(
        k_levels=[1],
        rarity_levels=list(RARITY_TIERS),
        spacing_levels=["clustered"],
        facts_per_class=3,
    )
    for c in cans:
        for p in c.paraphrases():
            assert p != c.canonical


def test_prompt_target_tokenize_correctly():
    """Memorization probe must score the leading-space object token. Regression
    test: prompt ends without whitespace, target starts with a leading space,
    and `target_ids = full_ids[len(prompt_ids):]` recovers the full object span.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    cans = generate_canaries(
        k_levels=[1], rarity_levels=["frequent"], spacing_levels=["clustered"],
        facts_per_class=4,
    )
    for c in cans:
        assert not c.prompt.endswith(" "), f"prompt should not end with space: {c.prompt!r}"
        assert c.target.startswith(" "), f"target should start with space: {c.target!r}"
        prompt_ids = tok.encode(c.prompt, add_special_tokens=False)
        full_ids = tok.encode(c.prompt + c.target, add_special_tokens=False)
        target_ids = full_ids[len(prompt_ids):]
        assert len(target_ids) > 0
        # The recovered target span should round-trip to the original object.
        decoded = tok.decode(target_ids).lstrip()
        assert decoded == c.object.strip(), f"decoded {decoded!r} != object {c.object!r}"
