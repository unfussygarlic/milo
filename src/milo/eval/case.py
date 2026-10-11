"""How an eval case is written down, shared by the dev and test splits."""

from typing import Any

MQ = ["MQ_SEARCH"]


def C(cat: str, q: str, expect, need=(), kb=None, err_ok: bool = False, allow=()) -> dict[str, Any]:
    """A dev-split case.

    expect: allowed FIRST tools ([] = no tool at all)
    need:   tools that must all appear somewhere in the trace
    kb:     merchant store to load
    err_ok: a tool error is the correct outcome
    allow:  extra numbers the reply may contain
    """
    return dict(
        cat=cat, q=q, expect=list(expect), need=list(need), kb=kb, err_ok=err_ok, allow=list(allow), split="dev"
    )


def Tc(*a: Any, **k: Any) -> dict[str, Any]:
    """A held-out test-split case."""
    c = C(*a, **k)
    c["split"] = "test"
    return c
