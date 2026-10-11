"""Canonical tool implementations, shared by the data generator AND the runtime.

Invariant 1: a tool result string is byte-identical between training data and
inference, because both call the code in this package. Nothing here may be
reimplemented or reformatted elsewhere.

Tool grammar (what the model emits inside ``<|tool_call|>...<|tool_result|>``)
-----------------------------------------------------------------------------
  CALCULATE(<arithmetic>)            CALCULATE(1499 * 18 / 100)
                                     helpers: emi(principal, annual_rate_pct, months), pct_change(old, new)
                                     comparisons: CALCULATE(8000 >= 3000) -> yes
  MQ_SEARCH(<keywords from customer>) merchant catalog / FAQ / store info lookup
                                     -> Basmati Rice 5kg | Rs 649 incl. GST | in stock
  SEARCH(<short keyword query>)      general knowledge (Wikipedia)
  DATE(today)                        -> Saturday, October 10, 2026
  DATE(time)                         -> 07:40 AM IST, Saturday, October 10, 2026
  DATE(<date>)                       -> weekday lookup, many formats accepted
  DATE(days until <date>)            -> 29 days until Sunday, November 8, 2026
  DATE(days between <d1> and <d2>)
  DATE(working days until <date>)    -> Mon-Sat count (Indian MSME norm)
  DATE(<date|today> + N days)        -> delivery ETAs etc.
  DATE(days until diwali)            -> festival names work anywhere a date does
  WEATHER(<city>)                    -> Mumbai: 29C, partly cloudy, humidity 78%, wind 13 km/h, rain chance 40%
  RUN_CODE(<python>)                 -> last expression auto-printed (REPL-style)

Dates are parsed leniently so the model can COPY a date verbatim from a SEARCH
result or the user's message instead of reformatting it.
"""

from milo.special_tokens import END_TOOL_TAG, TOOL_CALL_TAG, TOOL_RESULT_TAG

from .calculate import CalcError, calculate, format_number
from .clock import IST, now, set_now
from .dates import (
    FESTIVAL_ALIASES,
    FESTIVALS_FIXED,
    FESTIVALS_LUNAR,
    WORKING_WEEKDAYS,
    FestivalNotAvailable,
    date_tool,
    fmt_date,
    parse_date,
)
from .grounding import NUMBER_WORDS, fallback_answer, numbers_in, ungrounded_numbers
from .mq_search import (
    MQ_KEEP_RATIO,
    MQ_MAX_HITS,
    MQ_STOPWORDS,
    MQ_SYNONYMS,
    format_catalog_line,
    format_faq_line,
    format_price,
    mq_search,
    mq_tokens,
    set_merchant_kb,
)
from .parser import ToolCall, format_tool_call, parse_tool_call, run_tool_call
from .registry import MAX_RESULT_CHARS, TOOL_NAMES, TOOLS, clean_result, execute
from .run_code import run_code
from .web import MAX_SEARCH_CHARS, format_search_result, format_weather, search, weather

__all__ = [
    "END_TOOL_TAG",
    "FESTIVALS_FIXED",
    "FESTIVALS_LUNAR",
    "FESTIVAL_ALIASES",
    "IST",
    "MAX_RESULT_CHARS",
    "MAX_SEARCH_CHARS",
    "MQ_KEEP_RATIO",
    "MQ_MAX_HITS",
    "MQ_STOPWORDS",
    "MQ_SYNONYMS",
    "NUMBER_WORDS",
    "TOOLS",
    "TOOL_CALL_TAG",
    "TOOL_NAMES",
    "TOOL_RESULT_TAG",
    "WORKING_WEEKDAYS",
    "CalcError",
    "FestivalNotAvailable",
    "ToolCall",
    "calculate",
    "clean_result",
    "date_tool",
    "execute",
    "fallback_answer",
    "fmt_date",
    "format_catalog_line",
    "format_faq_line",
    "format_number",
    "format_price",
    "format_search_result",
    "format_tool_call",
    "format_weather",
    "mq_search",
    "mq_tokens",
    "now",
    "numbers_in",
    "parse_date",
    "parse_tool_call",
    "run_code",
    "run_tool_call",
    "search",
    "set_merchant_kb",
    "set_now",
    "ungrounded_numbers",
    "weather",
]
