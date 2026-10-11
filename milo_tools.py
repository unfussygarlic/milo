"""
Milo — canonical tool implementations (shared by data generator AND runtime)

Why this file exists
--------------------
Training data and inference must produce *byte-identical* tool results.
In v3 the generator hand-wrote results while the runtime formatted them
differently (CALCULATE '.6g' vs '.2f', WEATHER wttr text vs
'temp, cond, humidity'), and several calls the data taught (DATE(),
DATE(<date>), RUN_CODE(<bare expression>)) did nothing useful at runtime.
Both generate_tool_data.py and react_runtime.py now import from here.

Tool grammar (what the model emits inside <|tool_call|>...<|tool_result|>)
--------------------------------------------------------------------------
  CALCULATE(<arithmetic>)            CALCULATE(1499 * 18 / 100)
                                     helpers: emi(principal, annual_rate_pct, months), pct_change(old, new)
                                     comparisons: CALCULATE(8000 >= 3000) -> yes
  MQ_SEARCH(<keywords from customer>) merchant catalog / FAQ / store info lookup
                                     -> Basmati Rice 5kg | ₹649 incl. GST | in stock
  SEARCH(<short keyword query>)      general knowledge (Wikipedia)
  DATE(today)                        -> Saturday, October 10, 2026
  DATE(time)                         -> 07:40 AM IST, Saturday, October 10, 2026
  DATE(<date>)                       -> weekday lookup, many formats accepted
  DATE(days until <date>)            -> 29 days until Sunday, November 8, 2026
  DATE(days between <d1> and <d2>)
  DATE(working days until <date>)    -> Mon–Sat count (Indian MSME norm)
  DATE(<date|today> + N days)        -> delivery ETAs etc.
  DATE(days until diwali)            -> festival names work anywhere a date does
  WEATHER(<city>)                    -> Mumbai: 29°C, partly cloudy, humidity 78%, wind 13 km/h, rain chance 40%
  RUN_CODE(<python>)                 -> last expression auto-printed (REPL-style)

Dates are parsed leniently so the model can COPY a date verbatim from a
SEARCH result or the user's message instead of reformatting it.
"""

import ast
import datetime as dt
import json
import math
import operator
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import namedtuple
from zoneinfo import ZoneInfo

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
TOOL_CALL_TAG = "<|tool_call|>"
TOOL_RESULT_TAG = "<|tool_result|>"
END_TOOL_TAG = "<|end_tool|>"

IST = ZoneInfo("Asia/Kolkata")
MAX_RESULT_CHARS = 300
MAX_SEARCH_CHARS = 280
WORKING_WEEKDAYS = {0, 1, 2, 3, 4, 5}   # Mon–Sat; change to {0..4} for Mon–Fri
HTTP_TIMEOUT = 6
USER_AGENT = "Milo-Agent/1.0"

_TAG_RE = re.compile(r"<\|[a-z_]+\|>")   # strip special tags from tool output (prompt-injection guard)


# ═══════════════════════════════════════════════════════════════
# CLOCK — overridable so the generator can fix "today" per example
# ═══════════════════════════════════════════════════════════════
_now_override = None


def set_now(when):
    """Pin the clock (datetime, tz-aware or naive IST). Pass None to unpin."""
    global _now_override
    if when is not None and when.tzinfo is None:
        when = when.replace(tzinfo=IST)
    _now_override = when


def now():
    return _now_override or dt.datetime.now(IST)


# ═══════════════════════════════════════════════════════════════
# CALCULATE — AST-based (no eval), bounded
# ═══════════════════════════════════════════════════════════════
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CMPOPS = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le,
           ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "ceil": math.ceil, "floor": math.floor, "log": math.log,
    "log10": math.log10, "exp": math.exp,
}
def _emi(principal, annual_rate_pct, months):
    """Monthly EMI for a loan (reducing balance)."""
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def _pct_change(old, new):
    """Percentage change from old to new."""
    return (new - old) / old * 100


_FUNCS["emi"] = _emi
_FUNCS["pct_change"] = _pct_change
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e18


class CalcError(Exception):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS):
        return bool(_CMPOPS[type(node.ops[0])](_eval(node.left), _eval(node.comparators[0])))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError("exponent too large")
        res = _BINOPS[type(node.op)](left, right)
        if isinstance(res, complex):
            raise CalcError("complex result")
        if abs(res) > MAX_MAGNITUDE:
            raise CalcError("result too large")
        return res
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalcError("unsupported expression")


def _normalize_expr(expr):
    s = expr.strip()
    for a, b in (("^", "**"), ("×", "*"), ("÷", "/"), ("−", "-"), ("₹", "")):
        s = s.replace(a, b)
    # Strip thousands separators (incl. Indian 1,00,000) only when there are
    # no function calls, so max(10,20) is not turned into max(1020).
    if not re.search(r"[A-Za-z_]\w*\s*\(", s):
        s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


def format_number(x):
    """Canonical number format. Ints as-is; floats: 2 dp if |x|>=1, else 4 sig figs."""
    if isinstance(x, int):
        return str(x)
    if math.isnan(x) or math.isinf(x):
        raise CalcError("invalid math")
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.4g}"


def calculate(expr):
    s = _normalize_expr(expr)
    if not s:
        return "Error: empty expression"
    if len(s) > 300:
        return "Error: expression too long"
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return "Error: invalid expression"
    try:
        value = _eval(tree)
        if isinstance(value, bool):
            return "yes" if value else "no"
        return format_number(value)
    except ZeroDivisionError:
        return "Error: division by zero"
    except CalcError as e:
        return f"Error: {e}"
    except (OverflowError, ValueError, TypeError):
        return "Error: invalid math"


# ═══════════════════════════════════════════════════════════════
# DATE
# ═══════════════════════════════════════════════════════════════
_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y",
]
_NO_YEAR_FORMATS = ["%B %d", "%b %d", "%d %B", "%d %b"]
_WEEKDAY_PREFIX = re.compile(
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+", re.I)


def fmt_date(d):
    return f"{d:%A}, {d:%B} {d.day}, {d.year}"


def _plural_days(n):
    return f"{n} day" if n == 1 else f"{n} days"


# Festival calendar (verified Oct 2026; extend every year).
# Lunar festivals: explicit dates. Fixed festivals: (month, day) every year.
FESTIVALS_LUNAR = {
    "navratri": ["2026-10-11", "2027-09-30"],
    "dussehra": ["2026-10-20", "2027-10-09"],
    "karwa chauth": ["2026-10-29"],
    "dhanteras": ["2026-11-06", "2027-10-27"],
    "diwali": ["2026-11-08", "2027-10-29"],
    "govardhan puja": ["2026-11-10"],
    "bhai dooj": ["2026-11-11"],
    "makar sankranti": ["2027-01-15"],
    "maha shivaratri": ["2027-03-06"],
    "holi": ["2027-03-22"],
    "ram navami": ["2027-04-15"],
    "raksha bandhan": ["2027-08-17"],
    "janmashtami": ["2027-08-25"],
    "ganesh chaturthi": ["2027-09-04"],
}
FESTIVALS_FIXED = {
    "christmas": (12, 25), "new year": (1, 1), "republic day": (1, 26),
    "independence day": (8, 15), "gandhi jayanti": (10, 2), "valentine's day": (2, 14),
}
FESTIVAL_ALIASES = {
    "deepavali": "diwali", "divali": "diwali", "diwali puja": "diwali", "lakshmi puja": "diwali",
    "dasara": "dussehra", "dasehra": "dussehra", "dussera": "dussehra", "vijayadashami": "dussehra",
    "navratri start": "navratri", "navaratri": "navratri", "rakhi": "raksha bandhan",
    "rakshabandhan": "raksha bandhan", "janmasthami": "janmashtami", "krishna janmashtami": "janmashtami",
    "ganpati": "ganesh chaturthi", "ganesh chaturthi": "ganesh chaturthi", "shivratri": "maha shivaratri",
    "mahashivratri": "maha shivaratri", "maha shivratri": "maha shivaratri", "sankranti": "makar sankranti",
    "bhai dooj": "bhai dooj", "bhaidooj": "bhai dooj", "xmas": "christmas", "new year's": "new year",
    "new years": "new year", "new year's day": "new year", "26 january": "republic day",
    "15 august": "independence day", "valentines day": "valentine's day", "valentine day": "valentine's day",
    "karva chauth": "karwa chauth", "govardhan": "govardhan puja",
}


class FestivalNotAvailable(Exception):
    pass


def _festival_date(text, today):
    """'diwali', 'diwali 2027', 'rakhi' → date (next occurrence on/after today). None if not a festival."""
    t = text.lower().strip()
    m = re.fullmatch(r"(.*?)\s*(20\d\d)?", t, re.S)
    if not m:
        return None
    name, year = m.group(1).strip(), m.group(2)
    name = FESTIVAL_ALIASES.get(name, name)
    if name in FESTIVALS_FIXED:
        mo, d = FESTIVALS_FIXED[name]
        if year:
            return dt.date(int(year), mo, d)
        cand = dt.date(today.year, mo, d)
        return cand if cand >= today else dt.date(today.year + 1, mo, d)
    if name in FESTIVALS_LUNAR:
        dates = [dt.date.fromisoformat(s) for s in FESTIVALS_LUNAR[name]]
        if year:
            dates = [d for d in dates if d.year == int(year)]
        else:
            dates = [d for d in dates if d >= today]
        if not dates:
            raise FestivalNotAvailable(name)
        return dates[0]
    return None


def parse_date(text):
    """Lenient date parser (dates, relative words, festivals). Returns datetime.date or None."""
    today = now().date()
    t = text.strip().rstrip(".?").strip()
    low = t.lower()
    fest = _festival_date(low, today)
    if fest:
        return fest
    if low in ("", "today", "now", "aaj"):
        return today
    if low in ("tomorrow", "kal"):
        return today + dt.timedelta(days=1)
    if low == "yesterday":
        return today - dt.timedelta(days=1)
    t = _WEEKDAY_PREFIX.sub("", t)
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t, flags=re.I)
    t = re.sub(r"\bsept\b", "sep", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    for f in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(t, f).date()
        except ValueError:
            pass
    for f in _NO_YEAR_FORMATS:   # no year → next occurrence
        try:
            d = dt.datetime.strptime(t, f).date().replace(year=today.year)
        except ValueError:
            continue
        return d if d >= today else d.replace(year=today.year + 1)
    return None


def date_tool(arg):
    try:
        return _date_tool(arg)
    except FestivalNotAvailable as e:
        return f"Error: date for {e.args[0]} not available yet"


def _date_tool(arg):
    a = (arg or "").strip()
    low = a.lower()
    t = now()
    today = t.date()

    if low in ("", "today", "now", "date"):
        return fmt_date(today)
    if low in ("time", "current time", "time now"):
        return f"{t:%I:%M %p} IST, {fmt_date(today)}"

    def bad(x):
        return f"Error: could not understand date '{x}'"

    m = re.match(r"^working days (?:until|till|to)\s+(.+)$", low)
    if m:
        d = parse_date(m.group(1))
        if d is None:
            return bad(m.group(1))
        if d <= today:
            return f"0 working days until {fmt_date(d)}"
        n = sum(1 for i in range(1, (d - today).days + 1)
                if (today + dt.timedelta(days=i)).weekday() in WORKING_WEEKDAYS)
        return f"{n} working days (Mon–Sat) until {fmt_date(d)}"

    m = re.match(r"^days (?:until|till|to|left (?:for|until))\s+(.+)$", low)
    if m:
        d = parse_date(m.group(1))
        if d is None:
            return bad(m.group(1))
        diff = (d - today).days
        if diff > 0:
            return f"{_plural_days(diff)} until {fmt_date(d)}"
        if diff == 0:
            return f"{fmt_date(d)} is today"
        return f"{fmt_date(d)} was {_plural_days(-diff)} ago"

    m = re.match(r"^days between\s+(.+?)\s+and\s+(.+)$", low)
    if m:
        d1, d2 = parse_date(m.group(1)), parse_date(m.group(2))
        if d1 is None or d2 is None:
            return bad(m.group(1) if d1 is None else m.group(2))
        return f"{_plural_days(abs((d2 - d1).days))} between {fmt_date(d1)} and {fmt_date(d2)}"

    m = re.match(r"^(.+?)\s*([+-])\s*(\d+)\s*days?$", low)
    if m:
        d = parse_date(m.group(1))
        if d is None:
            return bad(m.group(1))
        n = int(m.group(3))
        if n > 36500:
            return "Error: too many days"
        return fmt_date(d + dt.timedelta(days=n if m.group(2) == "+" else -n))

    d = parse_date(a)
    return fmt_date(d) if d else bad(a)


# ═══════════════════════════════════════════════════════════════
# HTTP helper
# ═══════════════════════════════════════════════════════════════
def _http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _truncate(text, limit):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    if dot > limit * 0.5:
        return cut[:dot + 1]
    return cut.rsplit(" ", 1)[0] + "..."


# ═══════════════════════════════════════════════════════════════
# WEATHER
# ═══════════════════════════════════════════════════════════════
def format_weather(city, temp_c, condition, humidity, wind_kmph, rain_chance=None):
    """Canonical weather string — the generator must call this too."""
    s = (f"{city}: {temp_c}°C, {condition.strip().lower()}, "
         f"humidity {humidity}%, wind {wind_kmph} km/h")
    if rain_chance is not None:
        s += f", rain chance {rain_chance}%"
    return s


def weather(city):
    city = city.strip().strip("'\"")
    if not city:
        return "Error: no city given"
    try:
        data = _http_json(f"https://wttr.in/{urllib.parse.quote(city)}?format=j1")
        cur = data["current_condition"][0]
        rain = max(int(h.get("chanceofrain", 0)) for h in data["weather"][0]["hourly"])
        return format_weather(
            city.title(), int(cur["temp_C"]), cur["weatherDesc"][0]["value"],
            int(cur["humidity"]), int(cur["windspeedKmph"]), rain)
    except Exception:
        return f"Error: weather unavailable for {city.title()}"


# ═══════════════════════════════════════════════════════════════
# SEARCH — general knowledge (Wikipedia)
# ═══════════════════════════════════════════════════════════════
def format_search_result(title, text):
    """Canonical search string — the generator must call this too."""
    return f"{title}: {_truncate(text, MAX_SEARCH_CHARS)}"


def _search_wikipedia(query):
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 1, "format": "json"})
    hits = _http_json(f"https://en.wikipedia.org/w/api.php?{params}")["query"]["search"]
    if not hits:
        return None
    title = hits[0]["title"]
    page = urllib.parse.quote(title.replace(" ", "_"))
    extract = _http_json(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}").get("extract", "")
    return format_search_result(title, extract) if extract else None


def search(query):
    query = query.strip().strip("'\"")
    if not query:
        return "Error: empty search query"
    try:
        hit = _search_wikipedia(query)
    except Exception:
        return "Error: search unavailable"
    return hit or f"No results found for: {query}"


# ═══════════════════════════════════════════════════════════════
# MQ_SEARCH — merchant query: catalog, FAQ and store info for ONE merchant
#
# Knowledge base (JSON, one per merchant deployment):
#   {"store":   {"name": ..., "type": ..., "city": ...},
#    "catalog": [{"name", "variant", "price", "stock", "keywords", "category",
#                 "price_from" (bool, services), "offer" (optional str)}],
#    "faq":     [{"topic", "answer", "keywords"}]}
# Loaded from MILO_MERCHANT_KB (path) at runtime; the generator calls
# set_merchant_kb(dict) per example.
#
# Result: up to 3 lines, best first
#   Basmati Rice 5kg | ₹649 incl. GST | in stock
#   Screen replacement | from ₹1499 incl. GST | in stock
#   Delivery: Home delivery within Indore in 2 days. ₹40 charge, free above ₹500.
#   No match in store for: <query>
# ═══════════════════════════════════════════════════════════════
MQ_MAX_HITS = 3
MQ_KEEP_RATIO = 0.75     # keep hits scoring >= 75% of the best hit

# Hinglish / regional words → canonical English catalog words.
MQ_SYNONYMS = {
    "chawal": "rice", "chaval": "rice", "aata": "atta", "gehu": "wheat", "gehun": "wheat",
    "tel": "oil", "cheeni": "sugar", "chini": "sugar", "shakkar": "sugar", "namak": "salt",
    "doodh": "milk", "dudh": "milk", "dahi": "curd", "makhan": "butter", "anda": "egg",
    "ande": "egg", "chai": "tea", "patti": "tea", "sabun": "soap", "sabji": "vegetable",
    "sabzi": "vegetable", "aloo": "potato", "alu": "potato", "pyaz": "onion", "pyaaz": "onion",
    "kanda": "onion", "tamatar": "tomato", "adrak": "ginger", "lehsun": "garlic", "lahsun": "garlic",
    "kela": "banana", "seb": "apple", "aam": "mango", "santra": "orange", "angoor": "grapes",
    "mithai": "sweets", "kapda": "cloth", "kapde": "clothes", "joota": "shoes", "joote": "shoes",
    "chappal": "slipper", "pankha": "fan", "batti": "bulb", "taar": "wire", "rang": "paint",
    "kitab": "book", "kitaab": "book", "copy": "notebook", "kalam": "pen", "chashma": "glasses",
    "baal": "hair", "dawai": "medicine", "kutta": "dog", "billi": "cat", "khana": "food",
    "silai": "stitching", "silna": "stitching", "silne": "stitching", "silwana": "stitching", "silwani": "stitching", "darzi": "tailor", "cover": "cover", "charger": "charger",
    "earphone": "earphones", "headphone": "headphones", "tiffin": "tiffin", "cake": "cake",
    "timing": "hours", "time": "hours", "khulta": "hours", "khulti": "hours", "khulega": "hours",
    "band": "hours", "baje": "hours", "khule": "hours", "khula": "hours", "khulte": "hours",
    "jagah": "address", "pahunchu": "address",
    "chhoot": "discount", "chhut": "discount", "chut": "discount", "tayyar": "ready", "taiyar": "ready", "tayar": "ready", "pata": "address", "kaha": "address", "kahan": "address",
    "location": "address", "wapas": "return", "wapis": "return", "badalna": "exchange",
    "badal": "exchange", "refund": "return", "bhejo": "delivery",
    "bhej": "delivery", "ghar": "delivery", "aayega": "delivery", "ayega": "delivery",
    "pahuchega": "delivery", "shipping": "delivery", "courier": "delivery",
    "gpay": "upi", "phonepe": "upi", "paytm": "upi", "online": "upi",
    "kurta": "kurti", "sari": "saree", "rasgulla": "rasgulla", "rosogolla": "rasgulla",
}

# Words that carry no product meaning in customer messages.
MQ_STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "you", "your", "i", "me", "my", "we",
    "of", "for", "in", "on", "at", "to", "and", "or", "with", "have", "has", "can",
    "what", "how", "much", "many", "price", "rate", "cost", "kitna", "kitne", "kitni",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "hai", "hain", "h", "ho", "kya", "kyaa",
    "bhai", "bhaiya", "sir", "madam", "ji", "please", "pls", "plz", "available", "milega",
    "milegi", "milenge", "chahiye", "chaiye", "dena", "dedo", "do", "batao", "bata",
    "bataiye", "aap", "aapka", "aapke", "apka", "yeh", "ye", "woh", "wo", "hello", "hi",
    "daam", "bhav", "bhaav", "mrp", "stock", "store", "shop", "dukaan", "dukan", "any",
    "kab", "tak", "abhi", "today", "aaj", "kal", "it", "this", "that", "there", "per",
    "rs", "inr", "rupees", "rupaye", "wala", "wali", "wale", "ek", "one", "need", "want",
}

_UNIT_RULES = [
    (r"\b(?:aadha|adha|half)\s*(?:kilo|kg)\b", "500g"),
    (r"\b(?:paav|pao|pav)\s*(?:kilo|kg)?\b", "250g"),
    (r"(\d+(?:\.\d+)?)\s*(?:kgs?|kilos?|kilograms?)\b", r"\1kg"),
    (r"(\d+)\s*(?:g|gm|gms|grams?)\b", r"\1g"),
    (r"(\d+)\s*ml\b", r"\1ml"),
    (r"(\d+(?:\.\d+)?)\s*(?:l|ltrs?|litres?|liters?)\b", r"\1l"),
    (r"(\d+)\s*(?:pcs?|pieces?|nos?)\b", r"\1pc"),
    (r"(\d+)\s*(?:inch|inches|in)\b", r"\1in"),
    (r"(\d+)\s*(?:ft|feet|foot)\b", r"\1ft"),
    (r"(\d+)\s*(mah|gb|tb|w|ah|mm)\b", r"\1\2"),
]
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?[a-z]*|[a-z]+")

_merchant_kb = None
_merchant_kb_path_loaded = None


def set_merchant_kb(kb):
    """Generator / tests: use this KB dict (None → fall back to MILO_MERCHANT_KB)."""
    global _merchant_kb, _merchant_kb_path_loaded
    _merchant_kb, _merchant_kb_path_loaded = kb, None


def _get_merchant_kb():
    global _merchant_kb, _merchant_kb_path_loaded
    if _merchant_kb is not None and _merchant_kb_path_loaded is None:
        return _merchant_kb
    path = os.environ.get("MILO_MERCHANT_KB")
    if not path or not os.path.exists(path):
        return None
    if _merchant_kb_path_loaded != path:
        with open(path, encoding="utf-8") as f:
            _merchant_kb = json.load(f)
        _merchant_kb_path_loaded = path
    return _merchant_kb


def mq_tokens(text):
    s = text.lower()
    for pat, rep in _UNIT_RULES:
        s = re.sub(pat, rep, s)
    out = []
    for t in _TOKEN_RE.findall(s):
        t = MQ_SYNONYMS.get(t, t)
        if t not in MQ_STOPWORDS:
            out.append(t)
    return out


def _fuzzy_hit(tok, vocab):
    """Typos (basmti→basmati) and word-form variants (replace→replacement)."""
    if len(tok) < 4 or tok[0].isdigit():
        return False
    for v in vocab:
        if len(v) >= 4 and (v.startswith(tok) or tok.startswith(v)):
            return True
    import difflib
    return bool(difflib.get_close_matches(tok, [v for v in vocab if len(v) >= 4], n=1, cutoff=0.8))


def format_price(p):
    return f"{int(p)}" if float(p) == int(p) else f"{p:.2f}"


def format_catalog_line(item):
    """Canonical catalog result line — the generator must call this too."""
    name = item["name"] + (f" {item['variant']}" if item.get("variant") else "")
    if not item["price"]:
        price = "free"
    elif item.get("price_from"):
        price = f"from ₹{format_price(item['price'])} incl. GST"
    else:
        price = f"₹{format_price(item['price'])} incl. GST"
    line = f"{name} | {price} | {item.get('stock', 'in stock')}"
    if item.get("offer"):
        line += f" | offer: {item['offer']}"
    return line


def format_faq_line(entry):
    return f"{entry['topic']}: {entry['answer']}"


def _score(query_toks, primary, secondary):
    score, primary_hits = 0.0, 0
    for q in query_toks:
        if q in primary:
            score += 3; primary_hits += 1
        elif q in secondary:
            score += 2
        elif _fuzzy_hit(q, primary):
            score += 2; primary_hits += 1
    return score if primary_hits else 0.0


def mq_search(query, kb=None):
    query = query.strip().strip("'\"")
    if not query:
        return "Error: empty query"
    kb = kb if kb is not None else _get_merchant_kb()
    if not kb:
        return "Error: store catalog not available"
    q = mq_tokens(query)
    if not q:
        return f"No match in store for: {query}"

    scored = []
    for item in kb.get("catalog", []):
        primary = set(mq_tokens(item["name"] + " " + " ".join(item.get("keywords", []))))
        secondary = set(mq_tokens(f"{item.get('variant', '')} {item.get('category', '')}"))
        s = _score(q, primary, secondary)
        if s:
            scored.append((s, format_catalog_line(item)))
    for entry in kb.get("faq", []):
        primary = set(mq_tokens(entry["topic"] + " " + " ".join(entry.get("keywords", []))))
        secondary = set(mq_tokens(entry["answer"]))
        s = _score(q, primary, secondary)
        if s:
            scored.append((s, format_faq_line(entry)))

    if not scored:
        return f"No match in store for: {query}"
    scored.sort(key=lambda x: -x[0])          # stable: KB order breaks ties
    best = scored[0][0]
    lines = [line for s, line in scored if s >= best * MQ_KEEP_RATIO][:MQ_MAX_HITS]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# RUN_CODE — REPL-style (last expression printed), resource-limited
# NOTE: this is NOT a security sandbox. Do not expose to end customers
# on WhatsApp without a real sandbox (container/VM, no network).
# Set MILO_DISABLE_RUN_CODE=1 to turn it off.
# ═══════════════════════════════════════════════════════════════
_RUNNER = r'''
import ast, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
except Exception:
    pass
src = sys.stdin.read()
tree = ast.parse(src, mode="exec")
last = None
if tree.body and isinstance(tree.body[-1], ast.Expr):
    last = ast.Expression(tree.body.pop().value)
ns = {"__name__": "__main__"}
exec(compile(tree, "<milo>", "exec"), ns)
if last is not None:
    value = eval(compile(last, "<milo>", "eval"), ns)
    if value is not None:
        print(repr(value))
'''


def run_code(code, timeout=10):
    if os.environ.get("MILO_DISABLE_RUN_CODE") == "1":
        return "Error: RUN_CODE is disabled"
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", _RUNNER],
            input=code, capture_output=True, text=True, encoding="utf-8",
            timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Error: timed out ({timeout}s limit)"
    out = p.stdout.strip()
    if p.returncode != 0:
        if p.returncode < 0:
            last = "process killed (resource limit)"
        else:
            lines = p.stderr.strip().splitlines()
            last = lines[-1] if lines else f"exit code {p.returncode}"
        return f"{out}\nError: {last}" if out else f"Error: {last}"
    return out or "(no output)"


# ═══════════════════════════════════════════════════════════════
# REGISTRY, DISPATCH, PARSING, FORMATTING
# ═══════════════════════════════════════════════════════════════
TOOLS = {
    "CALCULATE": calculate,
    "SEARCH": search,
    "MQ_SEARCH": mq_search,
    "DATE": date_tool,
    "WEATHER": weather,
    "RUN_CODE": run_code,
}
TOOL_NAMES = tuple(TOOLS)


def clean_result(result):
    result = _TAG_RE.sub("", str(result)).strip() or "(empty result)"
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS].rstrip() + "..."
    return result


def execute(name, args):
    name = name.strip().upper()
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: unknown tool {name}"
    try:
        return clean_result(fn(args))
    except Exception as e:
        return f"Error: {type(e).__name__}"


ToolCall = namedtuple("ToolCall", "name args before")
_CALL_RE = re.compile(r"\s*([A-Za-z_]+)\s*\(")


def parse_tool_call(text):
    """
    Parse the first tool call in generated text.
    Args run to the LAST ')' before <|tool_result|> (or end of text),
    so nested parens like CALCULATE((100 + 50) * 1.18) parse correctly.
    Returns ToolCall(name, args, before) or None.
    """
    i = text.find(TOOL_CALL_TAG)
    if i < 0:
        return None
    before = text[:i]
    rest = text[i + len(TOOL_CALL_TAG):]
    j = rest.find(TOOL_RESULT_TAG)
    if j >= 0:
        rest = rest[:j]
    m = _CALL_RE.match(rest)
    if not m:
        return None
    body = rest[m.end():].rstrip()
    k = body.rfind(")")
    if k < 0:
        return None
    return ToolCall(m.group(1).upper(), body[:k].strip(), before)


def format_tool_call(name, args, result):
    return f"{TOOL_CALL_TAG}{name}({args}){TOOL_RESULT_TAG}{result}{END_TOOL_TAG}"


def run_tool_call(name, args):
    """For the generator: really execute the tool, return (formatted_call, result)."""
    result = execute(name, args)
    return format_tool_call(name, args, result), result


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


def numbers_in(text):
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


def ungrounded_numbers(answer, sources, allow=()):
    """Multi-digit numbers in `answer` that appear in none of `sources` (user msg, tool results)."""
    ok = set(allow)
    for s in sources:
        ok |= numbers_in(s)
    return sorted(n for n in {x.replace(",", "") for x in _NUM_RE.findall(answer)}
                  if len(n.replace(".", "")) > 1 and n not in ok)


def fallback_answer(calls):
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
CALC_NAMES = sorted(set(_FUNCS) | set(_CONSTS))
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

    def __init__(self, user_message, sources, numbers_in_reply):
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

    def eos_ok(self, text):
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
    def ok(self, text, tok):
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