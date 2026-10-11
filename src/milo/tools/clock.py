"""The agent's notion of "now" — IST, and overridable.

Everything date-related reads :func:`now` instead of ``datetime.now()`` so the
data generator can fix a different "today" per example and the eval can pin a
deterministic clock. Servers run in UTC; customers do not.
"""

import datetime as dt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ═══════════════════════════════════════════════════════════════
# CLOCK — overridable so the generator can fix "today" per example
# ═══════════════════════════════════════════════════════════════
_now_override = None


def set_now(when: dt.datetime | None) -> None:
    """Pin the clock (datetime, tz-aware or naive IST). Pass None to unpin."""
    global _now_override
    if when is not None and when.tzinfo is None:
        when = when.replace(tzinfo=IST)
    _now_override = when


def now() -> dt.datetime:
    return _now_override or dt.datetime.now(IST)
