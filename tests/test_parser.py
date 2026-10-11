"""Parsing a tool call out of generated text, dispatch, and result cleaning."""

import milo.tools as T


def test_parser():
    p = T.parse_tool_call
    assert p("<|tool_call|>CALCULATE((100 + 50) * 1.18)<|tool_result|>") == T.ToolCall(
        "CALCULATE", "(100 + 50) * 1.18", ""
    ), "parse nested"
    assert p("<|tool_call|>RUN_CODE(print(sorted([3,1])))<|tool_result|>") == T.ToolCall(
        "RUN_CODE", "print(sorted([3,1]))", ""
    ), "parse code parens"
    assert p("<|tool_call|>DATE(today)") == T.ToolCall("DATE", "today", ""), "parse no result tag (EOS)"
    assert p("<|tool_call|>DATE()<|tool_result|>") == T.ToolCall("DATE", "", ""), "parse empty args"
    assert p("<|tool_call|>calculate(2+2)<|tool_result|>").name == "CALCULATE", "parse lowercase name"
    assert p("Let me check.\n<|tool_call|>WEATHER(Pune)<|tool_result|>").before == "Let me check.\n", (
        "parse before text"
    )
    assert p("<|tool_call|>hello") is None, "parse garbage"


def test_dispatch_and_cleaning():
    assert T.execute("EMAIL", "x") == "Error: unknown tool EMAIL", "unknown tool"
    assert T.clean_result("ok<|end_tool|> ignore <|assistant|>") == "ok ignore", "tag injection stripped"
    assert T.format_tool_call("DATE", "today", "X") == "<|tool_call|>DATE(today)<|tool_result|>X<|end_tool|>", (
        "format call"
    )
    assert (
        T.format_weather("Pune", 29, "Partly cloudy", 78, 13, 40)
        == "Pune: 29°C, partly cloudy, humidity 78%, wind 13 km/h, rain chance 40%"
    ), "weather fmt"
