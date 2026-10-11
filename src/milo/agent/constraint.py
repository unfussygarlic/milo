"""Constrained decoding — the token filter that makes the model copy exactly.

``ALWAYS_OK_NUMBERS``, ``REPLY_EXTRA_NUMBERS``, the city gazetteer and the
DATE/CALCULATE vocabularies are part of the training-data contract: every
training target must pass this filter (tests/test_constraint_replay.py) and the
filter must have no dead ends (tests/test_constraint_fuzz.py).
"""

import re

from milo.special_tokens import TOOL_CALL_TAG, TOOL_RESULT_TAG
from milo.tools.calculate import CONSTS, FUNCS
from milo.tools.dates import FESTIVAL_ALIASES, FESTIVALS_FIXED, FESTIVALS_LUNAR
from milo.tools.grounding import numbers_in
from milo.tools.registry import TOOL_NAMES

# ═══════════════════════════════════════════════════════════════
# CONSTRAINED DECODING — make the model copy exactly
#
# Two rules the training data always follows, enforced token by token:
#   1. MQ_SEARCH / SEARCH / WEATHER args are a contiguous piece of the
#      customer's message → each next token must keep args a substring of it.
#   2. Every number is copied from the user message or a tool result →
#      a number being written must stay a prefix of a known number and may
#      only end on a complete one (single digits are always allowed).
# Rule 2 applies to every number of 2+ digits inside CALCULATE/DATE args and in
# replies after a tool result. In plain no-tool replies only 3+ digit numbers are
# checked (order IDs, amounts), so short facts like "18%" stay free.
# ═══════════════════════════════════════════════════════════════
COPY_TOOLS = ("MQ_SEARCH", "SEARCH", "WEATHER")
NUMERIC_TOOLS = ("CALCULATE", "DATE")
ALWAYS_OK_NUMBERS = {"100", "12", "1000", "60", "24", "365", "1024", "1048576", "100000", "10000000",
                     "1.60934", "0.453592", "2.20462", "0.3048", "2.54", "0.0929", "32",
                     "15", "2026", "2027"}   # last three: the trained "send it like 15 August 2027" reply
_RUN_RE = re.compile(r"\d[\d.,]*$")


CITY_GAZETTEER = sorted(set("""
delhi new delhi mumbai bombay bangalore bengaluru chennai madras kolkata calcutta hyderabad pune jaipur lucknow
indore bhopal patna ranchi nagpur surat ahmedabad vadodara baroda rajkot jamnagar bhavnagar kochi cochin goa shimla
manali darjeeling ooty mysore mysuru noida gurgaon gurugram ghaziabad faridabad agra varanasi kanpur meerut amritsar
ludhiana jalandhar chandigarh dehradun haridwar rishikesh udaipur jodhpur ajmer kota bikaner raipur bhubaneswar
cuttack guwahati shillong vizag visakhapatnam vijayawada tirupati madurai coimbatore trichy salem mangalore mangaluru
hubli hubballi belgaum nashik aurangabad thane navi mumbai kolhapur solapur jabalpur gwalior prayagraj allahabad
gorakhpur bareilly aligarh jammu srinagar leh siliguri dhanbad jamshedpur gaya muzaffarpur durgapur asansol
pondicherry puducherry trivandrum thiruvananthapuram kozhikode calicut thrissur dubai london singapore new york tokyo
paris sydney toronto seoul san francisco
""".replace("navi mumbai", "navi_mumbai").replace("new delhi", "new_delhi").replace("new york", "new_york")
   .replace("san francisco", "san_francisco").split()))
CITY_GAZETTEER = [c.replace("_", " ") for c in CITY_GAZETTEER]
REPLY_EXTRA_NUMBERS = {"15", "2026", "2027"}
WEATHER_WORDS = set("""mausam mosam weather barish baarish barsat rain raining umbrella chhata chata garmi garam
thand thandi sardi temperature temp humidity dhoop sunny hot cold jacket sweater forecast outdoor bahar
""".split())
CALC_NAMES = sorted(set(FUNCS) | set(CONSTS))
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
           "sept", "oct", "nov", "dec"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DATE_VOCAB = set(["today", "time", "now", "date", "current", "tomorrow", "yesterday", "aaj", "kal", "days", "day",
                  "until", "till", "to", "between", "and", "working", "left", "for", "st", "nd", "rd", "th"]
                 + _MONTHS + _WEEKDAYS
                 + [w for k in list(FESTIVALS_LUNAR) + list(FESTIVALS_FIXED) + list(FESTIVAL_ALIASES)
                    for w in re.findall(r"[a-z]+", k)])
_CALC_CHARS = re.compile(r"[0-9a-z_ +\-*/().,<>=!%^]*\Z")   # \Z, not $: '$' also matches before a trailing "\n"
_DATE_CHARS = re.compile(r"[0-9a-z +\-/,.'’]*\Z")


def _split_call_args(s):
    """Split 'args...)<|tool_res' at the ')' that closes the call's own '('.
    Returns (args, tail) — tail is None while the call is still open."""
    depth = 1
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[:i], s[i + 1:]
    return s, None


def _words_ok(text, vocab):
    """Complete words must be in vocab; a trailing partial word must be a prefix of one."""
    words = re.findall(r"[a-z_]+", text)
    if not words:
        return True
    complete = words if not re.search(r"[a-z_]$", text) else words[:-1]
    if any(w not in vocab for w in complete):
        return False
    if re.search(r"[a-z_]$", text):
        last = words[-1]
        return any(v.startswith(last) for v in vocab)
    return True   # the trained "send it like 15 August 2027" reply


def _boundary_find(hay, needle, need_end=False):
    """True if `needle` occurs in `hay` starting at a word boundary (and ending at one if need_end)."""
    if not needle:
        return True
    i = hay.find(needle)
    while i >= 0:
        start_ok = i == 0 or not hay[i - 1].isalnum()
        j = i + len(needle)
        end_ok = (not need_end) or j == len(hay) or not hay[j].isalnum()
        if start_ok and end_ok:
            return True
        i = hay.find(needle, i + 1)
    return False


class CopyConstraint:
    """
    Token filter used during generation (see react_runtime.apply_constraint).

    Copy tools (MQ_SEARCH, SEARCH, WEATHER): args must be a whole-word piece of the
      customer's message; WEATHER args must be a city named in it (when one is);
      after ')' only <|tool_result|> may follow.
    CALCULATE / DATE args: numbers of 2+ digits must come from the message, earlier
      tool results, or a few constants (100, 12, unit factors…).
    Reply after a tool result: EVERY number must come from the message or a tool result.
    Plain reply (no tool): numbers of 3+ digits must come from the message.
    """

    def __init__(self, user_message: str, sources: list[str], numbers_in_reply: bool) -> None:
        self.user_l = user_message.lower()
        found = set()
        for s in sources:
            found |= numbers_in(s)
        found = {a.replace(",", "") for a in found}
        self.allowed_args = found | ALWAYS_OK_NUMBERS
        date_error = any(s.startswith("Error: could not understand date") for s in sources)
        self.allowed_reply = found | (REPLY_EXTRA_NUMBERS if date_error else set())
        self.numbers_in_reply = numbers_in_reply
        self.cities = [c for c in CITY_GAZETTEER if _boundary_find(self.user_l, c, need_end=True)]
        self.user_words = set(re.findall(r"[a-z]+", self.user_l))
        weather_ok = bool(self.cities) or bool(self.user_words & WEATHER_WORDS)
        # WEATHER is only offered when the message names a city or talks about weather
        self.tool_names = [n for n in TOOL_NAMES if n != "WEATHER" or weather_ok]
        self.allowed, self.min_len = self.allowed_reply, 1

    # -- numbers --
    def _short(self, run):
        return len(run.replace(",", "").replace(".", "")) < self.min_len

    def _complete_ok(self, run):
        run = run.replace(",", "").rstrip(".")
        return self._short(run) or run in self.allowed

    def _prefix_ok(self, run):
        run = run.replace(",", "")
        if run.endswith(".") and self._complete_ok(run):   # "₹709." — sentence full stop
            return True
        return self._short(run) or any(a.startswith(run) for a in self.allowed)

    def _numbers_ok(self, text, tok, pattern=r"\d[\d.,]*"):
        joined, start = text + tok, len(text)
        for m in re.finditer(pattern, joined):
            if m.end() < start:
                continue                       # finished before this token — already checked
            run = m.group(0)
            if m.end() == len(joined):
                if not self._prefix_ok(run):   # still being written → must be a prefix
                    return False
            elif not self._complete_ok(run):   # closed → must be a complete known number
                return False
        return True

    # -- copy args --
    def _copy_ok(self, name, args, tok):
        if ")" in tok:
            head, tail = tok.split(")", 1)
            return self._copy_ok(name, args, head) and self._arg_closable(name, args + head) \
                and (TOOL_RESULT_TAG.startswith(tail) or tail.startswith(TOOL_RESULT_TAG))
        cand = (args + tok).lower().lstrip()
        if not cand:
            return True
        if name == "WEATHER" and self.cities:
            return any(c.startswith(cand) for c in self.cities)
        return _boundary_find(self.user_l, cand)

    def _arg_closable(self, name, args):
        core = args.lower().strip()
        if not core:
            return False
        if name == "WEATHER" and self.cities:
            return core in self.cities
        return _boundary_find(self.user_l, core, need_end=True)

    def eos_ok(self, text: str) -> bool:
        """End of reply is allowed only if something was said, no tool call is half-written,
        and no number is left half-copied. (Otherwise a blocked model would just stop.)"""
        if not text.strip():
            return False
        i = text.rfind(TOOL_CALL_TAG)
        if i >= 0 and TOOL_RESULT_TAG not in text[i:]:
            return False
        m = _RUN_RE.search(text)
        if m:
            self.allowed = self.allowed_reply
            self.min_len = 1 if self.numbers_in_reply else 3
            return self._complete_ok(m.group(0))
        return True

    # -- main --
    def ok(self, text: str, tok: str) -> bool:
        """text: this step's generated text so far; tok: candidate token string."""
        i = text.rfind(TOOL_CALL_TAG)
        if i >= 0 and TOOL_RESULT_TAG not in text[i:]:
            seg = text[i + len(TOOL_CALL_TAG):]
            m = re.match(r"\s*([A-Za-z_]+)\(", seg)
            if m:
                name, args = m.group(1).upper(), seg[m.end():]
                if name not in self.tool_names:
                    return False
                if name in COPY_TOOLS:
                    if ")" in args:            # args closed → only the result tag may follow
                        after = args[args.index(")") + 1:] + tok
                        return TOOL_RESULT_TAG.startswith(after) or after.startswith(TOOL_RESULT_TAG)
                    return self._copy_ok(name, args, tok)
                if name in NUMERIC_TOOLS:
                    head, tail = _split_call_args(args + tok)
                    if tail is not None and not (TOOL_RESULT_TAG.startswith(tail) or tail.startswith(TOOL_RESULT_TAG)):
                        return False           # after the call's closing ')' only the result tag may follow
                    head_l = head.lower() + (" " if tail is not None else "")   # closed → last word complete
                    if name == "CALCULATE":
                        if not _CALC_CHARS.match(head_l) or not _words_ok(head_l, CALC_NAMES):
                            return False
                    elif not _DATE_CHARS.match(head_l) or not _words_ok(head_l, DATE_VOCAB | self.user_words):
                        return False
                    self.allowed, self.min_len = self.allowed_args, 2
                    return self._numbers_ok(text, tok, r"\d[\d.]*" if name == "CALCULATE" else r"\d[\d.,]*")
            # still writing the tool name: it must be a prefix of an allowed tool
            name_so_far = (seg + tok).lstrip()
            if "(" in name_so_far:
                head = name_so_far.split("(", 1)[0].strip().upper()
                return head in self.tool_names
            return any(n.startswith(name_so_far.upper()) for n in self.tool_names) if name_so_far else True
        if not any(c.isdigit() for c in tok) and not _RUN_RE.search(text):
            return True                        # no number being written or closed
        self.allowed = self.allowed_reply
        self.min_len = 1 if self.numbers_in_reply else 3
        return self._numbers_ok(text, tok)
