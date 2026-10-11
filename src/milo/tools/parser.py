"""Parsing and formatting of the ``<|tool_call|>TOOL(args)<|tool_result|>`` span."""

import re
from collections import namedtuple

from milo.special_tokens import END_TOOL_TAG, TOOL_CALL_TAG, TOOL_RESULT_TAG

from .registry import execute

ToolCall = namedtuple("ToolCall", "name args before")
_CALL_RE = re.compile(r"\s*([A-Za-z_]+)\s*\(")


def parse_tool_call(text: str) -> ToolCall | None:
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
    rest = text[i + len(TOOL_CALL_TAG) :]
    j = rest.find(TOOL_RESULT_TAG)
    if j >= 0:
        rest = rest[:j]
    m = _CALL_RE.match(rest)
    if not m:
        return None
    body = rest[m.end() :].rstrip()
    k = body.rfind(")")
    if k < 0:
        return None
    return ToolCall(m.group(1).upper(), body[:k].strip(), before)


def format_tool_call(name: str, args: str, result: str) -> str:
    return f"{TOOL_CALL_TAG}{name}({args}){TOOL_RESULT_TAG}{result}{END_TOOL_TAG}"


def run_tool_call(name: str, args: str) -> tuple[str, str]:
    """For the generator: really execute the tool, return (formatted_call, result)."""
    result = execute(name, args)
    return format_tool_call(name, args, result), result
