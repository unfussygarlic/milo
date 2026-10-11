"""The tool registry and dispatch: name -> implementation, and ``execute``.

``MAX_RESULT_CHARS`` and the error strings are part of the training-data
contract: truncation length and ``"Error: <ExceptionName>"`` both appear in
training targets.
"""

import re

from .calculate import calculate
from .dates import date_tool
from .mq_search import mq_search
from .run_code import run_code
from .web import search, weather

MAX_RESULT_CHARS = 300
_TAG_RE = re.compile(r"<\|[a-z_]+\|>")   # strip special tags from tool output (prompt-injection guard)

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


def clean_result(result: object) -> str:
    result = _TAG_RE.sub("", str(result)).strip() or "(empty result)"
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS].rstrip() + "..."
    return result


def execute(name: str, args: str) -> str:
    name = name.strip().upper()
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: unknown tool {name}"
    try:
        return clean_result(fn(args))
    except Exception as e:
        return f"Error: {type(e).__name__}"
