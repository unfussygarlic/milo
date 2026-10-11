"""CALCULATE: arithmetic, output formatting, and the limits that stop abuse.

Every expected string here is part of the training-data contract - these exact
bytes went into the data the model learned from.
"""

import milo.tools as T


def test_calculate():
    c = T.calculate
    assert c("1499 + 2") == "1501", "add"
    assert c("1499 * 18 / 100") == "269.82", "gst"
    assert c("4499.1 * 1") == "4499.10", "trailing zero kept"
    assert c("(100 + 50) * 1.18") == "177", "nested parens"
    assert c("1234567.8 * 1") == "1234567.80", "big money no sci-notation"
    assert c("1 / 3") == "0.3333", "small"
    assert c("0.1 + 0.2") == "0.3", "float noise"
    assert c("100 / 0") == "Error: division by zero", "div zero"
    assert c("2^10") == "1024", "caret"
    assert c("10 × 3 ÷ 2") == "15", "unicode ops"
    assert c("1,00,000 * 2") == "200000", "indian commas"
    assert c("max(10,20)") == "20", "func commas kept"
    assert c("round(2.567, 1)") == "2.60", "round fn"
    assert c("100000 * 0.01 * (1 + 0.01)**12 / ((1 + 0.01)**12 - 1)") == "8884.88", "emi"
    assert c("9**9**9") == "Error: exponent too large", "pow bomb"
    assert c("__import__('os')") == "Error: unsupported expression", "no names"
    assert c("().__class__") == "Error: unsupported expression", "attr blocked"
    assert c("5 +") == "Error: invalid expression", "syntax"
    assert c("500 * 18%") == "Error: invalid expression", "pct sign"
    assert c("emi(200000, 12, 24)") == "9414.69", "emi helper"
    assert c("8000 >= 3000") == "yes", "compare yes"
    assert c("2500 > 3000") == "no", "compare no"
    assert c("1 < 2 < 3") == "Error: unsupported expression", "chained compare rejected"
    assert c("pct_change(72000, 87000)") == "20.83", "pct_change helper"
