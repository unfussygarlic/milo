"""The family mix, the generation loop and per-example validation.

``FAMILIES`` is the dataset recipe: name -> (builder, count at scale 1.0). Its
ORDER matters - families are drawn in iteration order from the one shared RNG,
so reordering this dict changes the bytes a given seed produces.

Every example is validated before it is written: structure, parser round-trip,
CALCULATE / RUN_CODE re-run through the real tools, and number grounding.
"""

import collections
import re

import milo.tools as T
from milo.data.generator.calc import fam_calc, fam_tool_error
from milo.data.generator.chat import (
    fam_about,
    fam_advice,
    fam_clarify,
    fam_clarify_other,
    fam_complaint,
    fam_concept,
    fam_decline,
    fam_greeting,
    fam_human,
    fam_order_status,
    fam_payment_done,
    fam_place_order,
    fam_thanks_bye,
)
from milo.data.generator.code import fam_code, fam_code_disabled
from milo.data.generator.core import R, Skip, grounded
from milo.data.generator.dates import fam_date
from milo.data.generator.store import (
    fam_mq_bulk,
    fam_mq_calc_offer,
    fam_mq_calc_qty,
    fam_mq_calc_two,
    fam_mq_date,
    fam_mq_faq,
    fam_mq_nomatch,
    fam_mq_product,
)
from milo.data.generator.web import (
    fam_route_contrast,
    fam_search,
    fam_search_custom,
    fam_search_error,
    fam_weather,
    fam_weather_error,
)

FAMILIES = {
    # tool (≈70%)
    "mq_product": (fam_mq_product, 2200),
    "mq_faq": (fam_mq_faq, 1300),
    "mq_nomatch": (fam_mq_nomatch, 450),
    "mq_calc_qty": (fam_mq_calc_qty, 750),
    "mq_calc_two": (fam_mq_calc_two, 300),
    "mq_calc_offer": (fam_mq_calc_offer, 250),
    "mq_bulk": (fam_mq_bulk, 260),
    "mq_date": (fam_mq_date, 450),
    "route_contrast": (fam_route_contrast, 500),
    "calc": (fam_calc, 1900),
    "date": (fam_date, 1100),
    "search": (fam_search, 650),
    "search_custom": (fam_search_custom, 200),
    "search_error": (fam_search_error, 60),
    "weather": (fam_weather, 1100),
    "weather_error": (fam_weather_error, 40),
    "code": (fam_code, 500),
    "code_disabled": (fam_code_disabled, 40),
    "tool_error": (fam_tool_error, 120),
    # no tool (≈30%)
    "greeting": (fam_greeting, 450),
    "thanks": (fam_thanks_bye, 400),
    "about": (fam_about, 150),
    "clarify": (fam_clarify, 700),
    "order_status": (fam_order_status, 450),
    "place_order": (fam_place_order, 450),
    "complaint": (fam_complaint, 350),
    "human": (fam_human, 200),
    "payment_ack": (fam_payment_done, 250),
    "decline": (fam_decline, 300),
    "concept": (fam_concept, 900),
    "advice": (fam_advice, 350),
    "clarify_other": (fam_clarify_other, 120),
}
MAX_DUP = 3  # max copies of an identical text


def generate(seed: int, scale: float, quiet: bool = False) -> tuple[list[dict], object, object]:
    R.seed(seed)
    out, skips, counts = [], collections.Counter(), collections.Counter()
    seen = collections.Counter()
    for name, (fn, n) in FAMILIES.items():
        target = max(1, int(n * scale))
        tries = 0
        while counts[name] < target and tries < target * 12:
            tries += 1
            try:
                ex = fn()
            except Skip as e:
                skips[f"{name}: {str(e)[:40]}"] += 1
                continue
            if seen[ex["text"]] >= MAX_DUP:
                skips[f"{name}: duplicate"] += 1
                continue
            seen[ex["text"]] += 1
            out.append(ex)
            counts[name] += 1
        if counts[name] < target and not quiet:
            print(f"  ! {name}: only {counts[name]}/{target}")
    T.set_now(None)
    T.set_merchant_kb(None)
    R.shuffle(out)
    return out, counts, skips


def token_len_fn():
    """The token counter behind ``--max_tokens``.

    tiktoken is required, not optional. The byte-length approximation this used
    to fall back to drops a *different* set of examples, so the same seed would
    silently produce a different dataset depending on whether the GPT-2 BPE
    files happened to be available. For offline or Docker use, pre-cache those
    files and point ``TIKTOKEN_CACHE_DIR`` at them.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception as e:
        raise RuntimeError(
            "the GPT-2 tiktoken encoding is required to count tokens for --max_tokens, and it "
            "could not be loaded. Install tiktoken and make the BPE files reachable (set "
            f"TIKTOKEN_CACHE_DIR for offline use). Original error: {type(e).__name__}: {e}"
        ) from e
    return lambda t: len(enc.encode(t, allowed_special="all")), "tiktoken"


_CALL_RE = re.compile(r"<\|tool_call\|>([A-Z_]+)\((.*?)\)<\|tool_result\|>(.*?)<\|end_tool\|>", re.S)


def validate(examples: list[dict], code_sample: int = 60) -> object:
    """Independent re-check of the final data (does not trust the generator)."""
    problems = collections.Counter()
    code_checked = 0
    for e in examples:
        t = e["text"]
        if not (t.startswith("<|user|>\n") and "\n<|assistant|>\n" in t and t.endswith("<|endoftext|>")):
            problems["bad structure"] += 1
            continue
        user, body = t[len("<|user|>\n") :].split("\n<|assistant|>\n", 1)
        calls = _CALL_RE.findall(body)
        if len(calls) != body.count("<|tool_call|>"):
            problems["unparseable tool call"] += 1
            continue
        results = []
        for name, args, res in calls:
            if name not in T.TOOL_NAMES:
                problems[f"unknown tool {name}"] += 1
            if name == "CALCULATE" and T.calculate(args) != res:
                problems["CALCULATE result mismatch"] += 1
            if name == "RUN_CODE" and code_checked < code_sample and res != "Error: RUN_CODE is disabled":
                code_checked += 1
                if T.execute("RUN_CODE", args) != res:
                    problems["RUN_CODE result mismatch"] += 1
            p = T.parse_tool_call(f"<|tool_call|>{name}({args})<|tool_result|>")
            if not p or p.name != name or p.args != args:
                problems["runtime parser disagrees"] += 1
            results.append(res)
        answer = body.rsplit("<|end_tool|>\n", 1)[-1].replace("<|endoftext|>", "")
        if not answer.strip():
            problems["empty answer"] += 1
        if not calls:
            continue  # no-tool examples were checked against their allow-lists at build time
        if grounded(answer, [user] + results, e.get("allow", ())):
            problems["ungrounded number"] += 1
    return problems
