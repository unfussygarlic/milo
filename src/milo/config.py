"""The one config loader: deployment settings live in ``configs/*.yaml``.

What belongs here
-----------------
Settings that change between machines and deployments: checkpoint paths, device,
sampling, server ports, WhatsApp credentials and API version.

What does NOT belong here
-------------------------
Anything that decides what a tool *outputs* or what the constraint *allows* --
``MAX_RESULT_CHARS``, ``MQ_MAX_HITS``, ``WORKING_WEEKDAYS``, the festival
tables, ``ALWAYS_OK_NUMBERS``, the special-token strings. Those are part of the
training-data contract (invariant 1) and stay in code, next to the comment that
says so, so that no YAML edit can silently make the generator and the runtime
disagree.

Model dimensions are the awkward middle case: they are deployment-ish (you point
them at a checkpoint) but a wrong value just fails to load. They live in
``configs/model.yaml`` and :func:`milo.model.checkpoint.load_model` checks them
against the checkpoint it is given.

Resolution order: ``$MILO_CONFIG_DIR`` -> ``<repo>/configs``.
"""

from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
from typing import Any

import yaml

_REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def config_dir() -> Path:
    """Directory the YAML files are read from."""
    return Path(os.environ["MILO_CONFIG_DIR"]) if os.environ.get("MILO_CONFIG_DIR") else _REPO_CONFIGS


@functools.cache
def load(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``. Cached: configs are read once per process."""
    path = config_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"config {name!r} not found at {path}. Set MILO_CONFIG_DIR if your configs live elsewhere."
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_config() -> dict[str, Any]:
    """Architecture of the serving model (must match the checkpoint)."""
    return copy.deepcopy(load("model"))


def generation_config() -> dict[str, Any]:
    """Sampling and ReAct-loop defaults for inference."""
    return copy.deepcopy(load("generation"))


def training_config(section: str) -> dict[str, Any]:
    """One section of ``configs/training.yaml`` (``pretrain`` / ``sft`` / ``tools``).

    A copy, so a script that overrides a hyperparameter from the command line
    cannot leak that override into the cached config.
    """
    cfg = load("training")
    if section not in cfg:
        raise KeyError(f"no [{section}] section in {config_dir() / 'training.yaml'}")
    return copy.deepcopy(cfg[section])
