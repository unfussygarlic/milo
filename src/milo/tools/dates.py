"""DATE — lenient date parsing, Indian festival calendar, working-day maths.

The festival tables and ``WORKING_WEEKDAYS`` are part of the training-data
contract: they decide the result strings the model was trained to copy, so
changing them makes training data and runtime disagree.
"""

import datetime as dt
import re

from .clock import now

WORKING_WEEKDAYS = {0, 1, 2, 3, 4, 5}   # Mon-Sat; change to {0..4} for Mon-Fri

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


def fmt_date(d: dt.date) -> str:
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


def parse_date(text: str) -> dt.date | None:
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


def date_tool(arg: str) -> str:
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
