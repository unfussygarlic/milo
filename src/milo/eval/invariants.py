"""The two constraint invariants, as callable checks.

**Replay** - the decoding constraint must never block anything the model was
trained to write. Every assistant turn in a dataset is replayed character by
character through :class:`~milo.agent.constraint.CopyConstraint`, built exactly
as the runtime builds it (sources = user message + results so far), plus the
end-of-reply check.

**Fuzz** - the constraint must have no dead ends. A dead end is a state where no
character and no end-of-reply is allowed; at runtime that forces a fallback to
unconstrained decoding (the eval's "fallbacks" counter). Walks start from random
prefixes of real training targets and step through allowed single characters
(GPT-2 has a token for every byte, so single characters are always available).

Run them from the command line::

    milo-check-constraint data/tool_use_v51_train.jsonl
    milo-fuzz-constraint data/tool_use_v51_train.jsonl

or as tests (tests/test_constraint_replay.py, tests/test_constraint_fuzz.py).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import string
import sys
from pathlib import Path

from milo.agent.constraint import CopyConstraint
from milo.special_tokens import END_TOOL_TAG, TOOL_CALL_TAG, TOOL_RESULT_TAG

DEFAULT_DATASET = Path("data/tool_use_v51_train.jsonl")
FUZZ_CHARS = list(string.printable) + ["₹", "×", "—", "�", "é"]


def steps(body: str) -> list[tuple[str, list[str]]]:
    """Split an assistant body into generation steps, as the runtime sees them."""
    out: list[tuple[str, list[str]]] = []
    results: list[str] = []
    rest = body
    while True:
        j = rest.find(TOOL_RESULT_TAG)
        if j < 0:
            out.append((rest, list(results)))
            return out
        k = rest.find(END_TOOL_TAG, j)
        out.append((rest[: j + len(TOOL_RESULT_TAG)], list(results)))
        results.append(rest[j + len(TOOL_RESULT_TAG) : k])
        rest = rest[k + len(END_TOOL_TAG) :]


def replay(text_record: str) -> str | None:
    """Replay one training target. Returns None if fully accepted, else why it was blocked."""
    user, body = text_record[len("<|user|>\n") :].split("\n<|assistant|>\n", 1)
    body = body.replace("<|endoftext|>", "")
    for step_text, results in steps(body):
        c = CopyConstraint(user, [user] + results, numbers_in_reply=bool(results))
        for i, ch in enumerate(step_text):
            if not c.ok(step_text[:i], ch):
                return f"blocked {ch!r} after …{step_text[max(0, i - 40) : i]!r}"
        if not step_text.endswith(TOOL_RESULT_TAG) and not c.eos_ok(step_text):
            return f"end-of-reply blocked after …{step_text[-40:]!r}"
    return None


def _rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def replay_dataset(path: str | Path) -> tuple[int, int, dict[str, tuple[str, str]]]:
    """Replay a whole dataset. Returns (accepted, total, {family: (why, example)})."""
    rows = _rows(path)
    bad: collections.Counter[str] = collections.Counter()
    examples: dict[str, tuple[str, str]] = {}
    for r in rows:
        why = replay(r["text"])
        if why:
            bad[r["family"]] += 1
            examples.setdefault(r["family"], (why, r["text"][:300]))
    return len(rows) - sum(bad.values()), len(rows), {k: examples[k] for k in bad}


def walk(rng: random.Random, user: str, results: list[str], prefix: str, n: int = 40) -> str | None:
    """Random-walk from `prefix`. Returns the dead-end text, or None if no dead end."""
    c = CopyConstraint(user, [user] + results, numbers_in_reply=bool(results))
    text = prefix
    for _ in range(n):
        if text.endswith(TOOL_RESULT_TAG):
            return None  # the step would stop here — fine
        ok = [ch for ch in FUZZ_CHARS if c.ok(text, ch)]
        if not ok:
            return None if c.eos_ok(text) else text  # dead end
        text += rng.choice(ok)
    return None


def fuzz_dataset(
    path: str | Path, n_examples: int = 3000, walks: int = 4, seed: int = 0
) -> tuple[int, dict[str, tuple[str, str]]]:
    """Walk random prefixes of real targets. Returns (dead_ends, {where: (user, text)})."""
    rng = random.Random(seed)
    rows = _rows(path)
    rng.shuffle(rows)
    deads: collections.Counter[str] = collections.Counter()
    samples: dict[str, tuple[str, str]] = {}
    for r in rows[:n_examples]:
        user, body = r["text"][len("<|user|>\n") :].split("\n<|assistant|>\n", 1)
        for step_text, results in steps(body.replace("<|endoftext|>", "")):
            for _ in range(walks):
                cut = rng.randrange(len(step_text) + 1)
                d = walk(rng, user, results, step_text[:cut])
                if d is not None:
                    key = d[d.rfind(TOOL_CALL_TAG) :][:28] if TOOL_CALL_TAG in d[-80:] else "reply"
                    key = key.split("(")[0]
                    deads[key] += 1
                    samples.setdefault(key, (user, d[-70:]))
    return sum(deads.values()), {k: samples[k] for k in deads}


def main_replay(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a dataset through the decoding constraint")
    ap.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET))
    args = ap.parse_args(argv)
    ok, total, bad = replay_dataset(args.dataset)
    print(f"{ok}/{total} training targets fully accepted by the constraint")
    for fam, (why, txt) in bad.items():
        print(f"  {fam}: blocked — e.g. {why}\n     {txt[:200]}")
    return 1 if bad else 0


def main_fuzz(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fuzz the decoding constraint for dead ends")
    ap.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET))
    ap.add_argument("--examples", type=int, default=3000)
    ap.add_argument("--walks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    total, samples = fuzz_dataset(args.dataset, args.examples, args.walks, args.seed)
    print(f"dead ends: {total}")
    for k, (user, tail) in samples.items():
        print(f"  {k!r}   e.g. user={user[:40]!r} …{tail!r}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main_replay())
