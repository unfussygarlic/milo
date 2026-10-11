"""RUN_CODE: REPL semantics and error reporting."""

import milo.tools as T


def test_run_code():
    r = T.run_code
    assert r("[x**2 for x in range(1, 6)]") == "[1, 4, 9, 16, 25]", "repl expr"
    assert r("'hello world'[::-1]") == "'dlrow olleh'", "repl string"
    assert r("import math; math.factorial(10)") == "3628800", "import then expr"
    assert r("print('hi')") == "hi", "print"
    assert r("a, b = 0, 1\nfib = []\nfor _ in range(5):\n    fib.append(a)\n    a, b = b, a + b\nfib") == "[0, 1, 1, 2, 3]", "multiline"
    assert r("1/0") == "Error: ZeroDivisionError: division by zero", "error last line"
    assert r("x = 5") == "(no output)", "no output"
    assert r("'₹' + str(499)") == "'₹499'", "utf8"
