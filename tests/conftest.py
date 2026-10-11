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


@pytest.fixture(scope="session")
def checkpoint() -> Path:
    """The tool-use checkpoint, for tests that need the real weights.

    Override with MILO_CHECKPOINT=/path/to/best.pt.
    """
    path = Path(os.environ.get("MILO_CHECKPOINT", "experiments/tools_v51/checkpoints/best.pt"))
    if not path.exists():
        pytest.skip(f"no checkpoint at {path}")
    return path


@pytest.fixture(scope="session")
def cuda_model(checkpoint):
    """The real model on the GPU, loaded once per session."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from milo.model.checkpoint import load_model

    return load_model(str(checkpoint), "cuda")
