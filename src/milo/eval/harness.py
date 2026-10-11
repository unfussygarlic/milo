"""Behavioural eval for tool use.

Runs the dev + held-out test prompts through the real ReAct runtime and scores
what actually matters:

  tool     right first tool, or correctly NO tool
  chain    every tool the task needs was called (e.g. MQ_SEARCH -> CALCULATE)
  args     tool args are valid (no tool errors) and MQ_SEARCH args are copied
           from the customer's words
  grounded every multi-digit number in the reply came from the user or a tool result
  uses     the reply actually uses the tool result (not just a canned sentence)
  clean    non-empty reply, no leaked special tags

Everything is deterministic: greedy decoding, fixed "today", fixed merchant
stores, and SEARCH / WEATHER stubbed (pass --live to hit the real APIs).

Usage
    milo-eval --checkpoint experiments/tools_v51/checkpoints/best.pt
    milo-eval --checkpoint a/best.pt --checkpoint b/best.pt          # side by side
    milo-eval --checkpoint best.pt --constrain both
    milo-eval --checkpoint best.pt --only store_chain --show-failures 10
    milo-eval --rescore eval_results/<run>.jsonl                     # no model needed
Writes full traces to eval_results/<run>.jsonl
"""

import argparse
import collections
import datetime as dt
import json
import os
import random
import re
import time

import milo.tools as T
from milo.data import merchants as M
from milo.eval.cases_dev import CASES
from milo.eval.cases_test import TEST_CASES

EVAL_NOW = dt.datetime(2026, 10, 10, 10, 30)


# ═══════════════════════════════════════════════════════════════
# FIXED STORES
# ═══════════════════════════════════════════════════════════════
_KB_CACHE = {}


def store(kind: str) -> dict:
    """Deterministic merchant KB with delivery + bulk terms present and everything in stock,
    plus a couple of deliberate out-of-stock / offer items for specific cases."""
    if kind in _KB_CACHE:
        return _KB_CACHE[kind]
    for seed in range(1000):
        kb = M.build_kb(kind, random.Random(seed), offer_rate=0)
        spec = M.MERCHANT_TYPES[kind]
        if (
            spec["delivery"]
            and kb["store"]["delivery_days"] in (None, 0)
            and kind not in ("bakery", "tiffin", "dairy", "fruits_veg")
        ):
            continue
        break
    for it in kb["catalog"]:
        it["stock"] = {"goods": "in stock", "service": "available", "course": "seats available"}[spec["stock"]]
        it.pop("offer", None)
    tweaks = {
        "kirana": {("Moong Dal", "1kg"): {"stock": "out of stock"}, ("Desi Ghee", "1l"): {"offer": "10% off"}},
        "mobile": {("Power Bank", "10000mAh"): {"stock": "out of stock"}},
        "bakery": {("Black Forest Cake", "1kg"): {"stock": "only 2 left"}},
    }.get(kind, {})
    for it in kb["catalog"]:
        for k, v in tweaks.get((it["name"], it["variant"]), {}).items():
            it[k] = v
    # rebuild FAQ text that mentions stock-independent terms stays as generated
    _KB_CACHE[kind] = kb
    return kb


# ═══════════════════════════════════════════════════════════════
# STUBS for external tools (deterministic)
# ═══════════════════════════════════════════════════════════════
SEARCH_STUB = {
    "upi": (
        "Unified Payments Interface",
        "Unified Payments Interface (UPI) is an instant real-time payment system developed by the National Payments Corporation of India.",
    ),
    "docker": (
        "Docker (software)",
        "Docker is a platform that packages applications and their dependencies into containers.",
    ),
    "fssai": (
        "Food Safety and Standards Authority of India",
        "The Food Safety and Standards Authority of India (FSSAI) regulates food safety; food businesses need an FSSAI licence or registration.",
    ),
    "mishti doi": (
        "Mishti doi",
        "Mishti doi is a fermented sweet yogurt popular in West Bengal, traditionally set in earthen pots.",
    ),
    "kubernetes": (
        "Kubernetes",
        "Kubernetes is an open-source system for automating deployment, scaling and management of containerised applications.",
    ),
    "inflation": ("Inflation", "Inflation is a general increase in the prices of goods and services over time."),
    "isro": (
        "Indian Space Research Organisation",
        "ISRO is India's national space agency, headquartered in Bengaluru.",
    ),
    "chikankari": ("Chikan embroidery", "Chikankari is a traditional hand embroidery style from Lucknow."),
    "jwt": (
        "JSON Web Token",
        "A JSON Web Token (JWT) is a compact token format for transmitting claims, often used for authentication.",
    ),
    "e-way bill": (
        "E-way bill",
        "An e-way bill is an electronic document required under GST for moving goods above a specified value.",
    ),
    "taj mahal": (
        "Taj Mahal",
        "The Taj Mahal is a white marble mausoleum in Agra, commissioned in 1632 by Shah Jahan.",
    ),
    "paneer": ("Paneer", "Paneer is a fresh cheese made by curdling heated milk with lemon juice or vinegar."),
}


def stub_search(q):
    ql = q.lower()
    for k, (title, text) in SEARCH_STUB.items():
        if k in ql:
            return T.format_search_result(title, text)
    return f"No results found for: {q}"


def stub_weather(city):
    c = city.strip().title() or "?"
    h = sum(map(ord, c))
    return T.format_weather(
        c,
        18 + h % 20,
        ["sunny", "light rain", "partly cloudy", "haze"][h % 4],
        40 + h % 50,
        5 + h % 25,
        [5, 80, 30, 10][h % 4],
    )


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════
CALC_CONSTANTS = {
    "100",
    "12",
    "1000",
    "60",
    "24",
    "365",
    "1024",
    "1048576",
    "1.60934",
    "0.453592",
    "2.20462",
    "0.3048",
    "2.54",
    "0.0929",
    "32",
    "100000",
    "10000000",
}
UNAVAILABLE = ("out of stock", "batch full", "fully booked")


def _words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def score(case: dict, response: str, calls: list[tuple]) -> dict:
    """calls: list of (name, args, result). Returns dict of check → True/False/None (n/a)."""
    names = [c[0] for c in calls]
    results = [c[2] for c in calls]
    expect = case["expect"]
    s = {}
    if not expect:
        s["tool"] = not names
    elif None in expect:
        s["tool"] = not names or names[0] in expect
    else:
        s["tool"] = bool(names) and names[0] in expect
    s["chain"] = all(n in names for n in case["need"]) if case["need"] else None

    if calls:
        errs = [r for r in results if r.startswith("Error")]
        mq_ok = all(_words(a) <= _words(case["q"]) for n, a, _ in calls if n == "MQ_SEARCH")
        # numbers inside CALCULATE / DATE args must come from the user or an earlier result
        arg_ok, seen = True, T.numbers_in(case["q"])
        for n, a, r in calls:
            if n in ("CALCULATE", "DATE"):
                extra = {x for x in T.numbers_in(a) if len(x.replace(".", "")) > 1} - seen - CALC_CONSTANTS
                if extra:
                    arg_ok = False
            seen |= T.numbers_in(r)
        s["args"] = (not errs or case["err_ok"]) and mq_ok and arg_ok
    else:
        s["args"] = None

    s["grounded"] = not T.ungrounded_numbers(response, [case["q"]] + results, case["allow"])
    usable = [
        r
        for r in results
        if not r.startswith(("Error", "No match"))
        and not all(any(u in ln for u in UNAVAILABLE) for ln in r.split("\n"))
    ]
    result_nums = {n for r in usable for n in T.numbers_in(r) if len(n.replace(".", "")) > 1}
    s["uses"] = (
        bool(result_nums & T.numbers_in(response))
        if result_nums and case["cat"] not in ("search", "route", "store_faq")
        else None
    )
    s["clean"] = bool(response.strip()) and "<|" not in response and "|>" not in response
    s["pass"] = all(v for k, v in s.items() if v is not None)

    # what a customer would actually see with the runtime guard on
    if calls and not s["grounded"]:
        g = dict(s, grounded=True, uses=None if s["uses"] is None else True)
        s["pass_guarded"] = all(v for k, v in g.items() if v is not None and k != "pass")
    else:
        s["pass_guarded"] = s["pass"]
    return s


# ═══════════════════════════════════════════════════════════════
# RUNNING
# ═══════════════════════════════════════════════════════════════
def make_agent(checkpoint: str, device: str, max_tool_calls: int, constrain: bool = False):
    from milo.agent.react import react_loop  # torch import deferred: --rescore needs no model
    from milo.model.checkpoint import load_model

    model = load_model(checkpoint, device)

    def agent(q):
        return react_loop(
            model,
            q,
            device=device,
            max_tool_calls=max_tool_calls,
            temperature=0.0,
            verbose=False,
            guard=False,
            constrain=constrain,
        )

    return agent


def run(agent, cases: list[dict], live: bool = False) -> list[dict]:
    if not live:
        T.TOOLS["SEARCH"] = stub_search
        T.TOOLS["WEATHER"] = stub_weather
    os.environ["MILO_DISABLE_RUN_CODE"] = "0"
    out = []
    for case in cases:
        T.set_now(EVAL_NOW)
        T.set_merchant_kb(store(case["kb"]) if case["kb"] else store("kirana"))
        t0 = time.time()
        try:
            response, calls = agent(case["q"])
        except Exception as e:  # a crash is a failure, not a stop
            response, calls = f"[crash: {type(e).__name__}: {e}]", []
        sc = score(case, response, calls)
        out.append(dict(case=case, response=response, calls=calls, score=sc, secs=round(time.time() - t0, 2)))
    T.set_now(None)
    T.set_merchant_kb(None)
    return out


def summarize(results: list[dict]) -> dict:
    cats = collections.defaultdict(list)
    for r in results:
        cats[r["case"]["cat"]].append(r)
    rows = {}
    for cat, rs in cats.items():
        rows[cat] = dict(n=len(rs), **{k: _rate(rs, k) for k in KEYS})
    rows["ALL"] = dict(n=len(results), **{k: _rate(results, k) for k in KEYS})
    return rows


KEYS = ("pass", "pass_guarded", "tool", "chain", "args", "grounded", "uses", "clean")


def _rate(rs, key):
    vals = [r["score"][key] for r in rs if r["score"][key] is not None]
    return None if not vals else sum(vals) / len(vals)


def fmt(x):
    return "   –" if x is None else f"{x * 100:4.0f}"


CAT_ORDER = [
    "store_product",
    "store_faq",
    "store_nomatch",
    "store_chain",
    "route",
    "calc",
    "date",
    "search",
    "weather",
    "code",
    "no_tool",
    "ALL",
]


def split_columns(names: list[str], result_sets: list[list[dict]], split: str) -> list[tuple[str, list[dict]]]:
    """(name, results) pairs for one split, dropping runs with no cases in it — a results
    file holding only test cases must not add an empty column to the dev table."""
    cols = [(n, [r for r in rs if r["case"].get("split", "dev") == split]) for n, rs in zip(names, result_sets)]
    return [(n, rs) for n, rs in cols if rs]


def print_table(names, summaries):
    print("\n" + "PASS RATE (%) by category" + "".join(f"  {n[:18]:>18}" for n in names))
    for cat in CAT_ORDER:
        present = [s[cat] for s in summaries if cat in s]
        if not present:
            continue
        line = f"  {cat:<15} n={present[0]['n']:<3}"
        for s in summaries:
            line += f"  {fmt(s[cat]['pass'] if cat in s else None):>18}"
        print(line)
    print("  " + "-" * (24 + 20 * len(summaries)))
    print(f"  {'ALL + guard':<21}" + "".join(f"  {fmt(s['ALL']['pass_guarded']):>18}" for s in summaries))
    print("\nCHECK BREAKDOWN (ALL)")
    for k in ("tool", "chain", "args", "grounded", "uses", "clean"):
        print(f"  {k:<10}" + "".join(f"  {fmt(s['ALL'][k]):>18}" for s in summaries))


def print_failures(results, k):
    fails = [r for r in results if not r["score"]["pass"]]
    if not fails or not k:
        return
    print(f"\nFIRST {min(k, len(fails))} FAILURES")
    for r in fails[:k]:
        bad = [c for c, v in r["score"].items() if v is False and c != "pass"]
        print(f"\n[{r['case']['cat']}] {r['case']['q']!r}  ✗ {', '.join(bad)}")
        for n, a, res in r["calls"]:
            print(f"   🔧 {n}({a}) → {res[:90]}")
        print(f"   → {r['response'][:200]}")


def main():
    ap = argparse.ArgumentParser(description="Milo tool-use behavioural eval")
    ap.add_argument("--checkpoint", action="append", help="repeat to compare checkpoints")
    ap.add_argument("--rescore", action="append", help="re-score saved eval_results/*.jsonl (no model needed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tool-calls", type=int, default=3)
    ap.add_argument("--only", help="run one category")
    ap.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="dev = original 138 (leaked into v4.1 templates); test = held-out",
    )
    ap.add_argument("--live", action="store_true", help="real SEARCH/WEATHER instead of stubs")
    ap.add_argument(
        "--constrain",
        choices=["off", "on", "both"],
        default="off",
        help="constrained copy decoding; 'both' runs each checkpoint twice to compare",
    )
    ap.add_argument("--show-failures", type=int, default=12)
    args = ap.parse_args()

    pool = {"dev": CASES, "test": TEST_CASES, "both": CASES + TEST_CASES}[args.split]
    cases = [c for c in pool if not args.only or c["cat"] == args.only]
    if args.rescore:
        by_q = {c["q"]: c for c in CASES + TEST_CASES}
        names, sums, last = [], [], None
        for path in args.rescore:
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            for r in rows:
                r["case"] = by_q.get(r["case"]["q"], r["case"])
                r["score"] = score(r["case"], r["response"], [tuple(c) for c in r["calls"]])
            names.append(os.path.basename(path).replace(".jsonl", ""))
            sums.append(rows)
            last = rows
        for split in ("dev", "test"):
            cols = split_columns(names, sums, split)
            if cols:
                print(f"\n══ {split.upper()} SPLIT ({len(cols[0][1])} cases) ══")
                print_table([n for n, _ in cols], [summarize(rs) for _, rs in cols])
        print_failures(last, args.show_failures)
        return
    if not args.checkpoint:
        ap.error("--checkpoint or --rescore required")
    os.makedirs("eval_results", exist_ok=True)
    names, all_results = [], []
    modes = {"off": [False], "on": [True], "both": [False, True]}[args.constrain]
    for ckpt, cons in [(c, m) for c in args.checkpoint for m in modes]:
        name = ckpt.split("experiments/")[-1].split("/checkpoints")[0].replace("/", "_") or os.path.basename(ckpt)
        name += "+constrain" if cons else ""
        print(f"\n▶ {name}: {len(cases)} cases …", flush=True)
        agent = make_agent(ckpt, args.device, args.max_tool_calls, constrain=cons)
        t0 = time.time()
        results = run(agent, cases, args.live)
        print(f"  done in {time.time() - t0:.0f}s ({(time.time() - t0) / len(cases):.1f}s per case)")
        if cons:
            from milo.agent.decoding import CONSTRAINT_STATS as st

            print(f"  constraint: {st['fallbacks']} fallbacks in {st['steps']} constrained steps (should be ~0)")
            st["steps"] = st["fallbacks"] = 0
        with open(f"eval_results/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        names.append(name)
        all_results.append(results)
        del agent
    for split in ("dev", "test"):
        cols = split_columns(names, all_results, split)
        if cols:
            print(
                f"\n══ {split.upper()} SPLIT ({len(cols[0][1])} cases){' — held out' if split == 'test' else ' — leaked into v4.1+ data'} ══"
            )
            print_table([n for n, _ in cols], [summarize(rs) for _, rs in cols])
    print_failures(all_results[-1], args.show_failures)
    print("\nFull traces: eval_results/<name>.jsonl")


if __name__ == "__main__":
    main()
