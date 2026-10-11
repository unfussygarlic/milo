"""Shared fixtures: where the invariant tests find a dataset to replay."""

import os
from pathlib import Path

import pytest

from milo.eval.invariants import DEFAULT_DATASET


@pytest.fixture(scope="session")
def dataset() -> Path:
    """A generated training set. data/ is gitignored, so skip when there isn't one.

    Override with MILO_INVARIANT_DATASET=/path/to/train.jsonl.
    """
    path = Path(os.environ.get("MILO_INVARIANT_DATASET", DEFAULT_DATASET))
    if not path.exists():
        pytest.skip(f"no dataset at {path} — run milo-generate-data first")
    return path
