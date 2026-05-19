"""Smoke tests for the Pythia size table + checkpoint schedule."""
from scaling_shapes.models import (
    SIZES, all_revisions, revision_step, smoke_revisions,
)


def test_size_table_invariants():
    names = [s.name for s in SIZES]
    assert names == ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
    # Params are strictly increasing.
    for a, b in zip(SIZES, SIZES[1:]):
        assert a.params < b.params, (a, b)
    # Bands cover all three.
    bands = {s.band for s in SIZES}
    assert bands == {"small", "mid", "large"}


def test_revisions_count_and_ordering():
    revs = all_revisions()
    assert len(revs) == 154, len(revs)
    assert revs[0] == "step0"
    assert revs[-1] == "step143000"
    steps = [revision_step(r) for r in revs]
    assert steps == sorted(steps)


def test_smoke_revisions_returns_log_spaced_subset():
    revs = smoke_revisions(10)
    assert len(revs) == 10
    # First and last revs cover the full range.
    steps = [revision_step(r) for r in revs]
    assert steps[0] == 0
    assert steps[-1] == 143000
    # All distinct.
    assert len(set(steps)) == len(steps)
