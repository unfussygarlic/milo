"""Generator plumbing: the RNG, the example builder, noise, per-example context.

Invariant 3 lives here: :func:`leak_filter` drops any template that shares a
four-word run with a held-out eval case, so the test split can never be trained
on. The import of the test cases is deliberately hard: a generator that cannot
see the test split must fail, not quietly train on it.

Everything random goes through the single module-level ``R``: the number and
order of draws decides the dataset bytes, so adding, removing or reordering a
draw changes the output for a given seed.
"""

import datetime as dt
import random
import re

import milo.tools as T
from milo.data import merchants as M
from milo.data import paraphrase_banks as B

# ── leak filter: drop any template sharing a 4-word run with the held-out test split ──
# Imported hard, on purpose: if this ever fails the generator must stop, not carry
# on training on the test split (invariant 3).
from milo.eval.cases_test import TEST_CASES as _TEST


def _grams(s, n=4):
    w = re.findall(r"[a-z]+", s.lower())
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


_TEST_GRAMS = set().union(*[_grams(c["q"]) for c in _TEST]) if _TEST else set()
LEAK_DROPPED = []


def leak_filter(templates: list[str]) -> list[str]:
    for t in templates:
        assert t.count("[") <= 1, f"template has more than one [span]: {t!r}"
    keep = []
    for t in templates:
        clean = re.sub(r"\{[^}]*\}", " ", t).replace("[", "").replace("]", "")
        (LEAK_DROPPED if _grams(clean) & _TEST_GRAMS else keep).append(t)
    return keep


R = random.Random()


def pick(xs):
    return R.choice(xs)


class Skip(Exception):
    """Example couldn't be built cleanly — generator retries."""


# ═══════════════════════════════════════════════════════════════
# NUMBERS & VALIDATION
# ═══════════════════════════════════════════════════════════════
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
CALC_CONSTANTS = {
    "100",
    "12",
    "1",
    "2",
    "1.60934",
    "0.453592",
    "2.20462",
    "0.3048",
    "2.54",
    "0.0929",
    "10.764",
    "1048576",
    "1024",
    "60",
    "24",
    "7",
    "365",
    "1000",
    "9",
    "5",
    "32",
    "30",
    "52",
}


def nums(s: str) -> set[str]:
    """Digits in s plus numbers implied by words ('teen' → 3, '2 lakh' → 200000)."""
    return T.numbers_in(s)


def grounded(text: str, sources: list[str], allow: tuple[str, ...] = ()) -> list[str]:
    """Multi-digit numbers in text that appear in none of the sources (or allow)."""
    return T.ungrounded_numbers(text, sources, allow)


class Ex:
    def __init__(self, family, user):
        self.family, self.user = family, user
        self.calls, self.results, self.names = [], [], []

    def tool(self, name, args):
        """Run the real tool."""
        call, result = T.run_tool_call(name, args)
        if name in ("CALCULATE", "DATE"):
            bad = [
                n
                for n in nums(args)
                if n not in nums(self.user) | set().union(*[nums(r) for r in self.results] or [set()])
                and n not in CALC_CONSTANTS
            ]
            if bad:
                raise Skip(f"calc arg numbers not grounded: {bad}")
        if name == "MQ_SEARCH":
            u = set(re.findall(r"[a-z0-9]+", self.user.lower()))
            a = set(re.findall(r"[a-z0-9]+", args.lower()))
            if not a <= u:
                raise Skip(f"MQ args not copied from user: {a - u}")
        self.calls.append(call)
        self.results.append(result)
        self.names.append(name)
        return result

    def fake(self, name, args, result):
        """External tools (SEARCH, WEATHER): synthetic result in the canonical format."""
        result = T.clean_result(result)
        self.calls.append(T.format_tool_call(name, args, result))
        self.results.append(result)
        self.names.append(name)
        return result

    def done(self, answer, allow=()):
        answer = answer.strip()
        bad = grounded(answer, [self.user] + self.results, allow)
        if bad:
            raise Skip(f"ungrounded numbers in answer: {bad}")
        for tag in ("<|", "|>"):
            if tag in answer:
                raise Skip("tag in answer")
        body = "".join(c + "\n" for c in self.calls) + answer
        text = f"<|user|>\n{self.user}\n<|assistant|>\n{body}<|endoftext|>"
        rec = {"text": text, "family": self.family, "tools": self.names}
        if allow:
            rec["allow"] = sorted(set(allow))
        return rec


# ═══════════════════════════════════════════════════════════════
# TEXT NOISE (WhatsApp style)
# ═══════════════════════════════════════════════════════════════
PREFIXES_HI = ["bhai ", "bhaiya ", "sir ", "ji ", "hello ", "hi ", "suniye ", "bhaiya ji "]
PREFIXES_EN = ["hi, ", "hello, ", "hey ", "sir, ", "excuse me, ", "quick q: "]
SUFFIXES = [" pls", " please", " jaldi batao", " asap", " ??", " ?", " batao na", " plz"]


def _typo_word(w):
    if len(w) < 5 or not w.isalpha():
        return w
    i = R.randrange(1, len(w) - 1)
    if R.random() < 0.5 and w[i] in "aeiou":
        return w[:i] + w[i + 1 :]  # drop a vowel
    return w[:i] + w[i + 1] + w[i] + w[i + 2 :]  # swap two letters


def noisy(template, lang):
    """Noise + casing applied to the TEMPLATE (before slots are filled), so product
    names, numbers and code inside slots are never altered."""
    q = template
    if R.random() < 0.08:
        parts = q.split(" ")
        idx = [i for i, p in enumerate(parts) if "{" not in p and "[" not in p and "]" not in p and "\n" not in p]
        if idx:
            j = pick(idx)
            parts[j] = _typo_word(parts[j])
            q = " ".join(parts)
    if R.random() < 0.18:
        q = pick(PREFIXES_HI if lang == "hi" else PREFIXES_EN) + q
    if R.random() < 0.1:
        q = q.rstrip("?.! ") + pick(SUFFIXES)
    if R.random() < 0.3:
        q = q.rstrip("?.!")
    r = R.random()
    if r < 0.4:
        q = q.lower()
    elif r < 0.65 and q[:1].islower():
        q = q[:1].upper() + q[1:]
    return q


def finish(q):
    return re.sub(r"[ \t]+", " ", q).strip()


def fill(template, lang, **slots):
    """Template with an optional [span] marking the MQ_SEARCH args.
    Returns (user_text, span_text). Square brackets inside slot VALUES are kept."""
    t = noisy(template, lang)
    m = re.search(r"\[(.+?)\]", t)
    span = None
    if m:
        span = m.group(1).format(**slots)
        t = t[: m.start()] + m.group(1) + t[m.end() :]
    return finish(t.format(**slots)), span


HI_WORDS = re.compile(
    r"\b(hai|hain|kya|kitna|kitne|kitni|ka|ki|ke|ko|karo|batao|chahiye|aaj|kab|mujhe|mera|"
    r"nahi|kaun|kaunsa|wapas|hoga|milega|bhai|bhaiya|ji|ye|isme|ulta|nikalo|hatao|chala)\b",
    re.I,
)


def is_hi(text):
    return bool(HI_WORDS.search(text))


def L():
    return "hi" if R.random() < 0.55 else "en"


def money(n):
    return f"₹{n}"


# ═══════════════════════════════════════════════════════════════
# CONTEXT RANDOMISATION
# ═══════════════════════════════════════════════════════════════
NOW_START = dt.datetime(2026, 10, 1)
NOW_DAYS = 364


def new_now():
    d = NOW_START + dt.timedelta(days=R.randrange(NOW_DAYS), hours=R.randint(7, 22), minutes=R.randrange(60))
    T.set_now(d)
    return d


def new_kb(mtype=None):
    kb = M.build_kb(mtype or pick(list(M.MERCHANT_TYPES)), R, wide=0.5)
    T.set_merchant_kb(kb)
    return kb


def by_lang(lang, hi, en):
    """Pick a template; chunk-2 paraphrase pools are keyed by the call's first Hindi template."""
    base = hi if lang == "hi" else en
    extra = leak_filter(B.EXTRA.get(hi[0], {}).get(lang, []))
    return pick(base + extra)
