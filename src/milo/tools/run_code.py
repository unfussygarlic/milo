"""RUN_CODE — REPL-style Python execution in a child process.

NOT a security sandbox. Never expose it to customer traffic; WhatsApp serving
sets ``MILO_DISABLE_RUN_CODE=1``.
"""

import os
import subprocess
import sys

# ═══════════════════════════════════════════════════════════════
# RUN_CODE — REPL-style (last expression printed), resource-limited
# NOTE: this is NOT a security sandbox. Do not expose to end customers
# on WhatsApp without a real sandbox (container/VM, no network).
# Set MILO_DISABLE_RUN_CODE=1 to turn it off.
# ═══════════════════════════════════════════════════════════════
_RUNNER = r"""
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
"""


def run_code(code: str, timeout: int = 10) -> str:
    if os.environ.get("MILO_DISABLE_RUN_CODE") == "1":
        return "Error: RUN_CODE is disabled"
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", _RUNNER],
            input=code,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
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
