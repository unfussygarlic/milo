"""Invariant: the constraint has no dead ends.

A dead end is a state where no character and no end-of-reply is allowed; the
runtime then falls back to unconstrained decoding. Marked slow (~90s over the
full dataset): run it with `pytest -m slow`.
"""

import pytest

from milo.eval.invariants import fuzz_dataset


@pytest.mark.slow
def test_constraint_has_no_dead_ends(dataset):
    dead_ends, samples = fuzz_dataset(dataset)
    assert dead_ends == 0, f"{dead_ends} dead ends: {samples}"
