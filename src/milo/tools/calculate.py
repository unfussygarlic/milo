"""CALCULATE — arithmetic over an AST allow-list (never ``eval``).

Output formatting (``format_number``) is part of the training-data contract:
the generator and the runtime must produce the same string for the same
expression, down to the digit.
"""

import ast
import math
import operator
import re

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
FUNCS = {
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


FUNCS["emi"] = _emi
FUNCS["pct_change"] = _pct_change
CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e18


class CalcError(Exception):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTS:
        return CONSTS[node.id]
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
            and node.func.id in FUNCS and not node.keywords):
        return FUNCS[node.func.id](*[_eval(a) for a in node.args])
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


def format_number(x: float) -> str:
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


def calculate(expr: str) -> str:
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
