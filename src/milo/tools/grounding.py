"""Number grounding — shared by the generator, the runtime guard and the eval.

Invariant 4: a customer-facing reply never contains a number that is not in the
customer's message or in a tool result. ``numbers_in`` defines what "is in"
means (Hinglish number words, lakh/crore/k scales) and is part of the
training-data contract.
"""

import re

# ═══════════════════════════════════════════════════════════════
# GROUNDING — shared by the data generator, the runtime guard and the eval
# ═══════════════════════════════════════════════════════════════
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
NUMBER_WORDS = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12, "pandrah": 15, "bees": 20,
    "pachas": 50, "sau": 100, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "fifty": 50, "hundred": 100, "dozen": 12, "darjan": 12,
}
_SCALES = [(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs)\b", 100000),
           (r"(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", 10000000),
           (r"(\d+(?:\.\d+)?)\s*k\b", 1000),
           (r"(\d+)\s*(?:dozen|darjan)\b", 12)]


def _num_str(x):
    return str(int(x)) if float(x) == int(x) else f"{x:g}"


def numbers_in(text: str) -> set[str]:
    """Digit strings in text (commas removed) plus numbers implied by words:
    'teen' → 3, '2 lakh' → 200000, '50k' → 50000, '3 dozen' → 36."""
    out = {n.replace(",", "") for n in _NUM_RE.findall(text)}
    low = text.lower()
    for w in re.findall(r"[a-z]+", low):
        if w in NUMBER_WORDS:
            out.add(str(NUMBER_WORDS[w]))
    for pat, mult in _SCALES:
        for m in re.finditer(pat, low):
            out.add(_num_str(float(m.group(1)) * mult))
    return out


def ungrounded_numbers(answer: str, sources: list[str], allow: tuple[str, ...] = ()) -> list[str]:
    """Multi-digit numbers in `answer` that appear in none of `sources` (user msg, tool results)."""
    ok = set(allow)
    for s in sources:
        ok |= numbers_in(s)
    return sorted(n for n in {x.replace(",", "") for x in _NUM_RE.findall(answer)}
                  if len(n.replace(".", "")) > 1 and n not in ok)


def fallback_answer(calls: list[tuple[str, str, str]]) -> str:
    """Deterministic reply built only from tool results — used when the model's
    own reply contains a number no tool produced."""
    good = [(n, r) for n, _, r in calls if not r.startswith("Error")]
    if not good:
        return "Sorry, abhi ye check nahi ho paya. Thodi der baad try karein ya shop owner se confirm karein."
    lines = []
    for name, r in good[-2:]:
        if name == "MQ_SEARCH":
            lines += [ln.replace(" | ", ", ") for ln in r.split("\n")]
        elif name == "CALCULATE":
            lines.append(f"Total: ₹{r}" if any(n == "MQ_SEARCH" for n, _ in good) else f"Result: {r}")
        else:
            lines.append(r)
    return "\n".join(dict.fromkeys(lines))
