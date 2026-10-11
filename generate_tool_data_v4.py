"""
Milo v5 — tool-use training data generator

Product: WhatsApp customer-support agent for Indian MSMEs (+ SDE1 dev questions)
Tools:   CALCULATE, MQ_SEARCH, SEARCH, DATE, WEATHER, RUN_CODE   (see milo_tools.py)
Format:  <|user|>\\n{q}\\n<|assistant|>\\n<|tool_call|>T(args)<|tool_result|>r<|end_tool|>\\n...{answer}<|endoftext|>

What's different from v3
  * Every CALCULATE / DATE / MQ_SEARCH / RUN_CODE result is produced by the REAL
    tool (milo_tools), so training results == runtime results byte-for-byte.
  * Every example is validated:
      - every number in the final answer comes from the user message or a tool result
      - CALCULATE args only use numbers from the user, earlier results, or known constants
      - MQ_SEARCH args are copied from the customer's words
  * Merchants are randomised per example (merchants.py) → prices/policies can't be memorised.
  * Random "today" per example (Oct 2026 – Sep 2027) → DATE answers can't be memorised.
  * Hinglish + English, WhatsApp-style noise, 70/30 tool/no-tool, multi-tool chains,
    error handling, MQ_SEARCH-vs-SEARCH routing contrasts.
  * Validation set is generated with a different seed and de-duplicated against train.

Usage
    python generate_tool_data_v4.py                       # → data/tool_use_v4_train.jsonl + _val.jsonl
    python generate_tool_data_v4.py --scale 0.1 --show 20 # quick look
"""

import argparse
import collections
import datetime as dt
import json
import os
import random
import re

import milo_tools as T
import merchants as M
import paraphrase_banks as B

# ── leak filter: drop any template sharing a 4-word run with the held-out test split ──
try:
    from eval_tools import TEST_CASES as _TEST
except ModuleNotFoundError:             # eval file absent → nothing to protect
    print("  ! eval_tools.py not found — leak filter has no test split to protect")
    _TEST = []


def _grams(s, n=4):
    w = re.findall(r"[a-z]+", s.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


_TEST_GRAMS = set().union(*[_grams(c["q"]) for c in _TEST]) if _TEST else set()
LEAK_DROPPED = []


def leak_filter(templates):
    for t in templates:
        assert t.count("[") <= 1, f"template has more than one [span]: {t!r}"
    keep = []
    for t in templates:
        clean = re.sub(r"\{[^}]*\}", " ", t).replace("[", "").replace("]", "")
        (LEAK_DROPPED if _grams(clean) & _TEST_GRAMS else keep).append(t)
    return keep

R = random.Random()


def pick(xs):
    return R.choice(xs)


class Skip(Exception):
    """Example couldn't be built cleanly — generator retries."""


# ═══════════════════════════════════════════════════════════════
# NUMBERS & VALIDATION
# ═══════════════════════════════════════════════════════════════
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
CALC_CONSTANTS = {"100", "12", "1", "2", "1.60934", "0.453592", "2.20462", "0.3048", "2.54",
                  "0.0929", "10.764", "1048576", "1024", "60", "24", "7", "365", "1000", "9", "5", "32",
                  "30", "52"}


def nums(s):
    """Digits in s plus numbers implied by words ('teen' → 3, '2 lakh' → 200000)."""
    return T.numbers_in(s)


def grounded(text, sources, allow=()):
    """Multi-digit numbers in text that appear in none of the sources (or allow)."""
    return T.ungrounded_numbers(text, sources, allow)


class Ex:
    def __init__(self, family, user):
        self.family, self.user = family, user
        self.calls, self.results, self.names = [], [], []

    def tool(self, name, args):
        """Run the real tool."""
        call, result = T.run_tool_call(name, args)
        if name in ("CALCULATE", "DATE"):
            bad = [n for n in nums(args) if n not in nums(self.user) | set().union(*[nums(r) for r in self.results] or [set()])
                   and n not in CALC_CONSTANTS]
            if bad:
                raise Skip(f"calc arg numbers not grounded: {bad}")
        if name == "MQ_SEARCH":
            u = set(re.findall(r"[a-z0-9]+", self.user.lower()))
            a = set(re.findall(r"[a-z0-9]+", args.lower()))
            if not a <= u:
                raise Skip(f"MQ args not copied from user: {a - u}")
        self.calls.append(call); self.results.append(result); self.names.append(name)
        return result

    def fake(self, name, args, result):
        """External tools (SEARCH, WEATHER): synthetic result in the canonical format."""
        result = T.clean_result(result)
        self.calls.append(T.format_tool_call(name, args, result))
        self.results.append(result); self.names.append(name)
        return result

    def done(self, answer, allow=()):
        answer = answer.strip()
        bad = grounded(answer, [self.user] + self.results, allow)
        if bad:
            raise Skip(f"ungrounded numbers in answer: {bad}")
        for tag in ("<|", "|>"):
            if tag in answer:
                raise Skip("tag in answer")
        body = "".join(c + "\n" for c in self.calls) + answer
        text = f"<|user|>\n{self.user}\n<|assistant|>\n{body}<|endoftext|>"
        rec = {"text": text, "family": self.family, "tools": self.names}
        if allow:
            rec["allow"] = sorted(set(allow))
        return rec


# ═══════════════════════════════════════════════════════════════
# TEXT NOISE (WhatsApp style)
# ═══════════════════════════════════════════════════════════════
PREFIXES_HI = ["bhai ", "bhaiya ", "sir ", "ji ", "hello ", "hi ", "suniye ", "bhaiya ji "]
PREFIXES_EN = ["hi, ", "hello, ", "hey ", "sir, ", "excuse me, ", "quick q: "]
SUFFIXES = [" pls", " please", " jaldi batao", " asap", " ??", " ?", " batao na", " plz"]


def _typo_word(w):
    if len(w) < 5 or not w.isalpha():
        return w
    i = R.randrange(1, len(w) - 1)
    if R.random() < 0.5 and w[i] in "aeiou":
        return w[:i] + w[i + 1:]                    # drop a vowel
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]      # swap two letters


def noisy(template, lang):
    """Noise + casing applied to the TEMPLATE (before slots are filled), so product
    names, numbers and code inside slots are never altered."""
    q = template
    if R.random() < 0.08:
        parts = q.split(" ")
        idx = [i for i, p in enumerate(parts) if "{" not in p and "[" not in p and "]" not in p and "\n" not in p]
        if idx:
            j = pick(idx)
            parts[j] = _typo_word(parts[j])
            q = " ".join(parts)
    if R.random() < 0.18:
        q = pick(PREFIXES_HI if lang == "hi" else PREFIXES_EN) + q
    if R.random() < 0.1:
        q = q.rstrip("?.! ") + pick(SUFFIXES)
    if R.random() < 0.3:
        q = q.rstrip("?.!")
    r = R.random()
    if r < 0.4:
        q = q.lower()
    elif r < 0.65 and q[:1].islower():
        q = q[:1].upper() + q[1:]
    return q


def finish(q):
    return re.sub(r"[ \t]+", " ", q).strip()


def fill(template, lang, **slots):
    """Template with an optional [span] marking the MQ_SEARCH args.
    Returns (user_text, span_text). Square brackets inside slot VALUES are kept."""
    t = noisy(template, lang)
    m = re.search(r"\[(.+?)\]", t)
    span = None
    if m:
        span = m.group(1).format(**slots)
        t = t[:m.start()] + m.group(1) + t[m.end():]
    return finish(t.format(**slots)), span


HI_WORDS = re.compile(r"\b(hai|hain|kya|kitna|kitne|kitni|ka|ki|ke|ko|karo|batao|chahiye|aaj|kab|mujhe|mera|"
                      r"nahi|kaun|kaunsa|wapas|hoga|milega|bhai|bhaiya|ji|ye|isme|ulta|nikalo|hatao|chala)\b", re.I)


def is_hi(text):
    return bool(HI_WORDS.search(text))


def L():
    return "hi" if R.random() < 0.55 else "en"


def money(n):
    return f"₹{n}"


# ═══════════════════════════════════════════════════════════════
# CONTEXT RANDOMISATION
# ═══════════════════════════════════════════════════════════════
NOW_START = dt.datetime(2026, 10, 1)
NOW_DAYS = 364


def new_now():
    d = NOW_START + dt.timedelta(days=R.randrange(NOW_DAYS), hours=R.randint(7, 22), minutes=R.randrange(60))
    T.set_now(d)
    return d


def new_kb(mtype=None):
    kb = M.build_kb(mtype or pick(list(M.MERCHANT_TYPES)), R, wide=0.5)
    T.set_merchant_kb(kb)
    return kb


# ═══════════════════════════════════════════════════════════════
# MQ_SEARCH HELPERS
# ═══════════════════════════════════════════════════════════════
_REV_SYN = collections.defaultdict(list)
for _k, _v in T.MQ_SYNONYMS.items():
    if _k != _v and _v not in ("hours", "address", "delivery", "return", "exchange", "upi"):
        _REV_SYN[_v].append(_k)


def variant_mentions(variant):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(kg|g|l|ml|mah|gb|w)", variant.strip(), re.I)
    if not m:
        return []
    n, u = m.group(1), m.group(2).lower()
    opts = {
        "kg": [f"{n}kg", f"{n} kg", f"{n} kilo"],
        "g": [f"{n}g", f"{n} gm", f"{n} grams"] + (["aadha kilo", "half kg"] if n == "500" else []),
        "l": [f"{n}l", f"{n} litre", f"{n} ltr"],
        "ml": [f"{n}ml", f"{n} ml"],
        "mah": [f"{n}mAh", f"{n} mah"],
        "gb": [f"{n}GB", f"{n} gb"],
        "w": [f"{n}W"],
    }[u]
    return opts


def product_mentions(item):
    name = item["name"]
    words = name.lower().split()
    cands = [name, name.lower()]
    if len(words) > 1 and len(words[-1]) > 3:
        cands.append(" ".join(words[-2:]))
    for i, w in enumerate(words):
        if _REV_SYN.get(w):
            cands.append(" ".join(words[:i] + [pick(_REV_SYN[w])] + words[i + 1:]))
    extra = [k for k in item.get("keywords", []) if k not in words and len(k) > 3]
    if extra:
        cands.append(pick(extra))
    return cands


def mention_for(item, with_variant=None):
    p = pick(product_mentions(item))
    vm = variant_mentions(item.get("variant", ""))
    if vm and (with_variant if with_variant is not None else R.random() < 0.7):
        v = pick(vm)
        p = f"{p} {v}" if R.random() < 0.75 else f"{v} {p}"
    return p


def parse_mq(result):
    items, faqs = [], []
    for ln in result.split("\n"):
        if " | " in ln:
            parts = ln.split(" | ")
            price = parts[1]
            m = re.search(r"₹([\d.]+)", price)
            items.append(dict(label=parts[0], price_txt=price, price=m.group(1) if m else None,
                              from_=price.startswith("from"), free=price == "free",
                              stock=parts[2], offer=parts[3][len("offer: "):] if len(parts) > 3 else None))
        elif ":" in ln and not ln.startswith("No match") and not ln.startswith("Error"):
            topic, ans = ln.split(":", 1)
            faqs.append(dict(topic=topic, answer=ans.strip()))
    return items, faqs


def price_phrase(it, lang):
    if it["free"]:
        return "free"
    if it["from_"]:
        return f"₹{it['price']} se shuru" if lang == "hi" else f"from ₹{it['price']}"
    return f"₹{it['price']}"


def stock_phrase(it, lang):
    s = it["stock"]
    m = re.match(r"only (\d+) (left|seats left)", s)
    if m:
        n = m.group(1)
        if "seats" in s:
            return f"sirf {n} seats bachi hain" if lang == "hi" else f"only {n} seats left"
        return f"sirf {n} bache hain" if lang == "hi" else f"only {n} left in stock"
    table = {
        "in stock": ("stock me hai", "in stock"),
        "available": ("available hai", "available"),
        "seats available": ("seats available hain", "seats are available"),
        "out of stock": ("abhi out of stock hai", "out of stock right now"),
        "fully booked today": ("aaj fully booked hai, kal ka slot mil sakta hai", "fully booked today — we can book you for tomorrow"),
        "batch full": ("ye batch full hai, next batch me admission milega", "this batch is full — you can join the next one"),
    }
    hi, en = table.get(s, (s, s))
    return hi if lang == "hi" else en


def is_unavailable(it):
    return it["stock"] in ("out of stock", "batch full", "fully booked today")


def describe_items(items, lang):
    """Natural answer for an MQ catalog result (1–3 items)."""
    first = items[0]
    if len(items) == 1 or (not is_unavailable(first) and R.random() < 0.35 and len(items) == 1):
        return describe_one(first, lang, alternatives=[])
    if is_unavailable(first):
        alts = [i for i in items[1:] if not is_unavailable(i)]
        return describe_one(first, lang, alternatives=alts)
    # several options → list and ask
    opts = []
    for it in items:
        tail = "" if not is_unavailable(it) else (" (out of stock)" if lang == "en" else " (abhi nahi hai)")
        opts.append(f"{it['label']} – {price_phrase(it, lang)}{tail}")
    if lang == "hi":
        return pick(["Ye options hain:\n", "Hamare paas ye hain:\n", "Available options:\n"]) + "\n".join(opts) + \
            "\n" + pick(["Kaunsa chahiye?", "Aapko kaunsa chahiye?", "Kaunsa bhej doon?"])
    return pick(["We have:\n", "Here are the options:\n", "Available:\n"]) + "\n".join(opts) + \
        "\n" + pick(["Which one would you like?", "Which one do you need?"])


def describe_one(it, lang, alternatives):
    lab, pr, st = it["label"], price_phrase(it, lang), stock_phrase(it, lang)
    if is_unavailable(it):
        if lang == "hi":
            s = pick([f"Sorry, {lab} {st}.", f"{lab} ({pr}) {st}."])
            if alternatives:
                a = alternatives[0]
                s += f" {a['label']} {price_phrase(a, lang)} me available hai — chalega?"
            else:
                s += pick([" Thode din me wapas aa jayega.", " Kuch aur chahiye?", ""])
        else:
            s = pick([f"Sorry, {lab} is {st}.", f"{lab} ({pr}) is {st}."])
            if alternatives:
                a = alternatives[0]
                s += f" We do have {a['label']} at {price_phrase(a, lang)} — would that work?"
            else:
                s += pick([" It should be back soon.", " Can I help with anything else?", ""])
        return s
    style = "course" if "seat" in it["stock"] else "service" if it["stock"] == "available" else "goods"
    follow = {
        ("hi", "goods"): ["", "", " Order karna hai?", " Bhej doon?", " Kitna chahiye?"],
        ("hi", "service"): ["", "", " Booking karni hai?", " Slot book kar doon?"],
        ("hi", "course"): ["", "", " Admission lena hai?", " Demo class bhi le sakte hain."],
        ("en", "goods"): ["", "", " Would you like to order?", " How many do you need?", " Shall I note it down?"],
        ("en", "service"): ["", "", " Shall I book a slot?", " Want me to book it?"],
        ("en", "course"): ["", "", " Would you like to enrol?", " You can also try a demo class."],
    }[(lang, style)]
    if lang == "hi":
        if it["free"]:
            s = pick([f"{lab} bilkul free hai, {st}.", f"Ji, {lab} free hai — {st}."])
        else:
            s = pick([f"{lab} {pr} ka hai, {st}.", f"{lab} ka rate {pr} hai, {st}.", f"{lab}: {pr}, {st}."])
        if it["offer"]:
            s += f" Abhi offer bhi hai: {it['offer']}."
    else:
        if it["free"]:
            s = pick([f"{lab} is free — {st}.", f"{lab} is free ({st})."])
        else:
            s = pick([f"{lab} is {pr} (incl. GST) — {st}.", f"{lab} costs {pr}, {st}.", f"{lab}: {pr}, {st}."])
        if it["offer"]:
            s += f" There's an offer too: {it['offer']}."
    return s + pick(follow)


def describe_faq(f, lang):
    ans = f["answer"]
    if lang == "hi":
        return pick([f"Ji, {ans}", f"{ans}", f"Bilkul. {ans}", f"{ans} Aur kuch?", f"Ji haan — {ans}"])
    return pick([f"{ans}", f"Sure — {ans}", f"{ans} Anything else?", f"Here you go: {ans}"])


def mq_lookup(ex, item, require_top=True, tries=6, with_variant=None, template_fn=None):
    """Find a mention of `item` whose MQ_SEARCH result contains (or starts with) it."""
    line = T.format_catalog_line(item)
    for _ in range(tries):
        p = mention_for(item, with_variant)
        res = T.mq_search(p)
        lines = res.split("\n")
        if (lines[0] == line) if require_top else (line in lines):
            return p
    raise Skip("no mention retrieves the item")


# ═══════════════════════════════════════════════════════════════
# FAMILY: MQ product lookup
# ═══════════════════════════════════════════════════════════════
PRODUCT_T = {
    "hi": ["[{p}] wala kitne ka hai?", "[{p}] wala kitne ka diya?", "[{p}] wala ka rate?","[{p}] ka rate kya hai?", "[{p}] kitne ka hai", "[{p}] hai kya?", "[{p}] milega?",
           "[{p}] available hai?", "[{p}] ka price batao", "[{p}] kitne ka milega", "[{p}] ka daam?",
           "[{p}] stock me hai?", "[{p}] chahiye, kitne ka hai?", "[{p}]??", "[{p}] rate",
           "aapke paas [{p}] hai?", "[{p}] ka bhav kya chal raha hai", "[{p}] ka kya rate hai aaj"],
    "en": ["how much is the [{p}] one?", "price of [{p}]?", "how much is [{p}]", "[{p}] price", "do you have [{p}]?",
           "is [{p}] available?", "[{p}] in stock?", "what's the rate for [{p}]", "I need [{p}], how much?",
           "[{p}] cost?", "any [{p}] available", "how much for [{p}]?", "Do you sell [{p}]?"],
}


def fam_mq_product():
    new_now(); kb = new_kb()
    item = pick(kb["catalog"])
    lang = L()
    p = None
    for _ in range(6):
        cand = mention_for(item)
        if item["name"].lower() in T.mq_search(cand).lower() or T.format_catalog_line(item) in T.mq_search(cand):
            p = cand; break
    if p is None:
        raise Skip("mention")
    user, span = fill(pick(PRODUCT_T[lang]), lang, p=p)
    ex = Ex("mq_product", user)
    res = ex.tool("MQ_SEARCH", span)
    items, faqs = parse_mq(res)
    if not items:
        raise Skip("no items")
    return ex.done(describe_items(items, lang))


# ═══════════════════════════════════════════════════════════════
# FAMILY: MQ FAQ
# ═══════════════════════════════════════════════════════════════
FAQ_T = {
    "Store hours": {"hi": ["shop [kitne baje khulti] hai?", "[kab tak khula] hai aaj?", "[sunday ko band] hai kya?",
                           "[timing] kya hai?", "dukaan [kitne baje band] hoti hai", "kal [kitne baje khulega]?"],
                    "en": ["what are your [timings]?", "[when do you open] tomorrow?", "are you [open on sunday]?",
                           "[shop hours]?", "[what time do you close]?", "[opening hours] please"]},
    "Address": {"hi": ["[shop kahan] hai?", "[address] bhejo", "[location] share karo", "aapki [dukaan kahan] hai?"],
                "en": ["[where is your shop]?", "[shop address] please", "send [location]", "what's your [address]?"]},
    "Payment": {"hi": ["[upi chalega]?", "[gpay] se pay kar sakte?", "[cod] hai kya?", "[card se payment] hoga?",
                       "payment kaise karna hai, [gpay] chalega?", "[paytm] le loge?",
                       "[phonepe] le lete ho?", "[online payment] hota hai?"],
                "en": ["[do you accept upi]?", "can I [pay by card]?", "[cash on delivery] available?",
                       "[payment options]?", "do you take [paytm]?"]},
    "Returns": {"hi": ["[wapas] ho jayega?", "[return] kar sakte hai kya", "[exchange policy] kya hai",
                       "pasand nahi aaya to [wapas] le loge?", "[refund] milega?"],
                "en": ["can I [return] it?", "[return policy]?", "is [exchange] possible?", "[refund policy]?"]},
    "GST bill": {"hi": ["[gst bill] milega?", "[gst invoice] doge?", "[bill] milega na?"],
                 "en": ["do you give a [gst invoice]?", "are prices [including gst]?", "I need a [gst bill]"]},
    "Delivery": {"hi": ["[ghar pe delivery] hogi?", "[delivery charge] kitna hai?", "[home delivery] karte ho?",
                        "[delivery] milegi kya?"],
                 "en": ["[do you deliver]?", "[home delivery] available?", "[delivery charges]?",
                        "do you do [home delivery]?"]},
    "Bulk orders": {"hi": ["[bulk order] pe discount?", "[wholesale] rate milega?", "[party ke liye bulk] order hai",
                           "zyada lenge to [discount] milega?"],
                    "en": ["[bulk order discount]?", "do you give [wholesale] rates?", "any [discount] on big orders?"]},
    "Minimum order": {"hi": ["[minimum order] kitna hai?", "delivery ke liye [minimum order]?"],
                      "en": ["what's the [minimum order] for delivery?"]},
    "Warranty": {"hi": ["[warranty] kitni hai?", "[guarantee] milegi?", "charger ki [warranty] kitni milegi?", "[warranty] milti hai kya?"],
                 "en": ["is there any [warranty]?", "how long is the [warranty] on earbuds?"]},
    "Repair time": {"hi": ["[repair me kitna time] lagega?", "kab tak [ready] hoga?"],
                    "en": ["[how long] does repair take?", "when will it be [ready]?"]},
    "Alteration": {"hi": ["[alteration] free hai?", "[fitting] theek karoge?"], "en": ["do you do [alteration]?"]},
    "Cement delivery": {"hi": ["[cement tempo] charge kitna?"], "en": ["[tempo] charge for [cement delivery]?"]},
    "Custom cake": {"hi": ["[photo cake] milega?", "[theme cake] banate ho?", "[custom birthday] cake chahiye",
                            "birthday ke liye [photo] wala cake bana doge?", "[photo cake] ka order lete ho?"],
                    "en": ["do you make [photo cakes]?", "can I order a [custom birthday] cake?"]},
    "Print from phone": {"hi": ["[whatsapp pe file] bhej ke print milega?"], "en": ["can I send a [pdf] to [print]?"]},
    "Wedding orders": {"hi": ["[shaadi] ke liye order lete ho?", "[wedding] order hai"],
                       "en": ["do you take [wedding] orders?"]},
    "Home visit": {"hi": ["[ghar aake] repair karoge?", "[technician] bhejoge?", "[technician] ghar bhej sakte ho?",
                           "[ghar pe aake] dekh loge?"],
                   "en": ["can a [technician] come home?", "do you do [home visit]?"]},
    "Order cutoff": {"hi": ["[lunch order kab tak] dena hai?", "[dinner] ka order [last] kab tak?"],
                     "en": ["what's the [cutoff] for lunch orders?"]},
    "Menu": {"hi": ["[aaj kya bana] hai?", "aaj ka [menu] kya hai?"], "en": ["what's today's [menu]?"]},
    "Morning slot": {"hi": ["[subah] ka [slot] hai?", "[morning] delivery [slot]?"], "en": ["do you have a [morning slot]?"]},
    "Stitching time": {"hi": ["[kitne din] me ready hoga?", "stitching me [kitne din] lagenge?"],
                       "en": ["[how many days] for stitching?"]},
    "Appointment": {"hi": ["[appointment] lena padega?", "[booking] kaise kare?"], "en": ["do I need an [appointment]?"]},
    "Milk subscription": {"hi": ["[roz subah] doodh doge?", "[monthly subscription] hai doodh ka?"],
                          "en": ["do you have a [daily milk subscription]?"]},
    "Grooming slot": {"hi": ["[grooming appointment] chahiye"], "en": ["how do I book [grooming]?"]},
    "Batch timings": {"hi": ["[batch timing] kya hai?", "[morning batch] hai?", "[weekend batch] hai kya?"],
                      "en": ["what are the [batch timings]?", "is there a [weekend batch]?"]},
    "Online classes": {"hi": ["[online classes] hoti hai?"], "en": ["do you have [online classes]?"]},
}


def fam_mq_faq():
    new_now(); kb = new_kb()
    topics = [f["topic"] for f in kb["faq"] if f["topic"] in FAQ_T]
    topic = pick(topics)
    lang = L()
    for _ in range(4):
        user, span = fill(pick(FAQ_T[topic][lang]), lang)
        res = T.mq_search(span)
        if res.split("\n")[0].startswith(topic + ":"):
            break
    else:
        raise Skip("faq retrieval")
    ex = Ex("mq_faq", user)
    res = ex.tool("MQ_SEARCH", span)
    _, faqs = parse_mq(res)
    return ex.done(describe_faq(faqs[0], lang))


# ═══════════════════════════════════════════════════════════════
# FAMILY: MQ no match
# ═══════════════════════════════════════════════════════════════
def fam_mq_nomatch():
    new_now(); kb = new_kb()
    others = [t for t in M.MERCHANT_TYPES if t != kb["store"]["type"]]
    other_kb = M.build_kb(pick(others), R)
    item = pick(other_kb["catalog"])
    lang = L()
    p = mention_for(item)
    if not T.mq_search(p).startswith("No match"):
        raise Skip("accidentally matches")
    user, span = fill(pick(PRODUCT_T[lang]), lang, p=p)
    ex = Ex("mq_nomatch", user)
    ex.tool("MQ_SEARCH", span)
    if lang == "hi":
        a = pick([f"Sorry, {span} abhi hamare paas nahi hai. Kuch aur chahiye?",
                  f"{span} hum nahi rakhte, maaf kijiye. Aur kuch dekh loon?",
                  f"Sorry ji, {span} available nahi hai. Aap shop owner se confirm kar sakte hain ki mangwa sakte hain ya nahi."])
    else:
        a = pick([f"Sorry, we don't have {span} right now. Can I help with anything else?",
                  f"I couldn't find {span} in our store. Anything else you're looking for?",
                  f"Sorry, {span} isn't something we stock. Is there anything else I can check?"])
    return ex.done(a)


# ═══════════════════════════════════════════════════════════════
# FAMILY: MQ → CALCULATE chains
# ═══════════════════════════════════════════════════════════════
GOODS_TYPES = [t for t, s in M.MERCHANT_TYPES.items() if s["stock"] == "goods"]


def _buyable(kb, n=1):
    items = [i for i in kb["catalog"] if i["price"] and not i.get("price_from") and i["stock"] == "in stock"]
    if len(items) < n:
        raise Skip("not enough buyable items")
    return R.sample(items, n)


# Quantity is always marked (unit word, "x", or after the product) so it can't be
# confused with a size: "5 kilo wala basmati kitne ka" is a price question, not 5 × price.
QTY_T = {
    "hi": ["{q} {u} [{p}] kitne ka padega?", "[{p}] {q} {u} chahiye, total kitna?", "{q} {u} [{p}] ka total batao",
           "mujhe {q} {u} [{p}] chahiye, kitna hoga", "[{p}] x {q} kitna hua?", "{q} {u} [{p}] kitne ke?",
           "{q} {u} [{p}] kitne ke padenge?", "[{p}] {q} {u} lena hai, total?", "{q} {u} [{p}] de do, kitna hua?",
           "[{p}] {q} {u} kitne ka hoga"],
    "en": ["how much for {q} {u} [{p}]?", "{q} x [{p}] total?", "I want {q} {u} [{p}], what's the total?",
           "total for {q} {u} [{p}]?", "[{p}] x {q}, how much?", "cost of {q} {u} [{p}]?"],
}
QTY_UNITS = {"hi": ["packet", "piece", "bag", "pack", "nos", "pcs"], "en": ["packets", "pieces", "bags", "packs", "pcs"]}
QTY_WORDS = {"hi": {2: "do", 3: "teen", 4: "char", 5: "paanch", 6: "chhe", 10: "das", 12: "barah"},
             "en": {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 10: "ten", 12: "twelve"}}


def qty_text(q, lang):
    w = QTY_WORDS[lang].get(q)
    return w if w and R.random() < 0.5 else str(q)


def fam_mq_calc_qty():
    new_now()
    if R.random() < 0.12:                                  # dozen-sold items ("5 dozen kele")
        kb = new_kb("fruits_veg")
        dozen = [i for i in kb["catalog"] if i["variant"].endswith("dozen") and i["stock"] == "in stock"]
        if not dozen:
            raise Skip("no dozen item")
        item = pick(dozen)
    else:
        kb = new_kb(pick(GOODS_TYPES))
        item = _buyable(kb)[0]
    lang = L()
    p = mq_lookup(None, item, require_top=True)
    qty = pick([2, 2, 3, 3, 4, 5, 6, 10, 12, 15, 20])
    unit = pick(QTY_UNITS[lang] + B.QTY_EXTRA_UNITS[lang])
    if item.get("variant", "").endswith("dozen"):        # "5 dozen kela" = 5 × price-per-dozen
        p = mq_lookup(None, item, require_top=True, with_variant=False)
        unit = "dozen"
    user, span = fill(pick(QTY_T[lang]), lang, p=p, q=qty_text(qty, lang), u=unit)
    ex = Ex("mq_calc_qty", user)
    res = ex.tool("MQ_SEARCH", span)
    it = parse_mq(res)[0][0]
    total = ex.tool("CALCULATE", f"{qty} * {it['price']}")
    if lang == "hi":
        a = pick([f"{it['label']} ₹{it['price']} ka hai, toh {qty} ka total ₹{total} hoga.",
                  f"{qty} × ₹{it['price']} = ₹{total}. Order confirm karna hai?",
                  f"Total ₹{total} ({qty} × {it['label']} @ ₹{it['price']})."])
    else:
        a = pick([f"{it['label']} is ₹{it['price']} each, so {qty} will cost ₹{total}.",
                  f"{qty} × ₹{it['price']} = ₹{total} (incl. GST).",
                  f"That comes to ₹{total} for {qty} ({it['label']} at ₹{it['price']} each)."])
    return ex.done(a)


TWO_T = {
    "hi": ["{q1} [{p1}] aur {q2} <{p2}> ka total?", "{q1} [{p1}] aur {q2} <{p2}> chahiye, kitna hua?",
           "[{p1}] {q1} aur <{p2}> {q2}, total batao"],
    "en": ["total for {q1} [{p1}] and {q2} <{p2}>?", "I need {q1} [{p1}] and {q2} <{p2}>, how much?"],
}


def fam_mq_calc_two():
    new_now(); kb = new_kb(pick(GOODS_TYPES))
    a_item, b_item = _buyable(kb, 2)
    if a_item["name"] == b_item["name"]:
        raise Skip("same product")
    lang = L()
    p1 = mq_lookup(None, a_item); p2 = mq_lookup(None, b_item)
    q1, q2 = pick([1, 2, 3, 4, 5]), pick([1, 2, 3, 4, 5])
    tmpl = pick(TWO_T[lang])
    user, span1 = fill(tmpl.replace("<", "").replace(">", ""), lang, p1=p1, p2=p2, q1=qty_text(q1, lang), q2=qty_text(q2, lang))
    span2 = p2
    ex = Ex("mq_calc_two", user)
    ra = parse_mq(ex.tool("MQ_SEARCH", span1))[0]
    rb = parse_mq(ex.tool("MQ_SEARCH", span2))[0]
    if not ra or not rb:
        raise Skip("lookup")
    ia, ib = ra[0], rb[0]
    if not ia["price"] or not ib["price"] or ia["from_"] or ib["from_"] or is_unavailable(ia) or is_unavailable(ib):
        raise Skip("unpriced")
    total = ex.tool("CALCULATE", f"{q1} * {ia['price']} + {q2} * {ib['price']}")
    if lang == "hi":
        a = (f"{ia['label']}: {q1} × ₹{ia['price']}\n{ib['label']}: {q2} × ₹{ib['price']}\n"
             f"Total: ₹{total}" + pick(["", ". Order confirm karein?", " (GST included)."]))
    else:
        a = (f"{ia['label']}: {q1} × ₹{ia['price']}\n{ib['label']}: {q2} × ₹{ib['price']}\n"
             f"Total: ₹{total}" + pick(["", ". Shall I note the order?", " (incl. GST)."]))
    return ex.done(a)


OFFER_T = {
    "hi": ["[{p}] pe koi offer hai?", "[{p}] discount ke baad kitne ka?", "[{p}] offer price kya hai?",
           "[{p}] sale me kitne ka milega"],
    "en": ["any discount on [{p}]?", "[{p}] price after offer?", "what's the offer price of [{p}]?"],
}


def fam_mq_calc_offer():
    new_now(); kb = new_kb(pick(GOODS_TYPES))
    item = _buyable(kb)[0]
    pct = pick([5, 10, 15, 20, 25])
    item["offer"] = f"{pct}% off"
    for other in kb["catalog"]:
        if other is not item and other["name"] == item["name"]:
            other.pop("offer", None)
    lang = L()
    p = mq_lookup(None, item)
    user, span = fill(pick(OFFER_T[lang]), lang, p=p)
    ex = Ex("mq_calc_offer", user)
    it = parse_mq(ex.tool("MQ_SEARCH", span))[0][0]
    if not it["offer"]:
        raise Skip("no offer in top")
    final = ex.tool("CALCULATE", f"{it['price']} * (1 - {pct} / 100)")
    if lang == "hi":
        a = pick([f"Haan, {it['label']} pe {pct}% off hai. ₹{it['price']} ki jagah ₹{final} me milega.",
                  f"{it['label']}: MRP ₹{it['price']}, {pct}% off ke baad ₹{final}."])
    else:
        a = pick([f"Yes! {it['label']} has {pct}% off — ₹{final} instead of ₹{it['price']}.",
                  f"{it['label']} is ₹{it['price']}; with {pct}% off you pay ₹{final}."])
    return ex.done(a)


BULK_T = {
    "hi": ["₹{amt} ka order hai, [bulk discount] milega?", "[bulk order] ₹{amt} ka karna hai, discount kitna?",
           "₹{amt} ka saman lena hai, [discount] milega kya?"],
    "en": ["my order is ₹{amt}, any [bulk discount]?", "₹{amt} [bulk order] — how much discount?"],
}


def fam_mq_bulk():
    new_now()
    kb = new_kb(pick([t for t, s in M.MERCHANT_TYPES.items() if s["bulk"]]))
    st = kb["store"]
    lang = L()
    above = R.random() < 0.75
    amt = (R.randrange(st["bulk_min"] + 100, st["bulk_min"] * 4, 50) if above
           else R.randrange(max(200, st["bulk_min"] // 4), st["bulk_min"] - 100, 50))
    for _ in range(4):
        user, span = fill(pick(BULK_T[lang]), lang, amt=amt)
        if T.mq_search(span).startswith("Bulk orders:"):
            break
    else:
        raise Skip("bulk retrieval")
    ex = Ex("mq_bulk", user)
    ex.tool("MQ_SEARCH", span)
    pct, mn = st["bulk_pct"], st["bulk_min"]
    eligible = ex.tool("CALCULATE", f"{amt} >= {mn}")      # never compare numbers "in the head"
    if eligible == "yes":
        final = ex.tool("CALCULATE", f"{amt} * (1 - {pct} / 100)")
        a = (f"Haan! ₹{mn} se upar ke order pe {pct}% discount hai. ₹{amt} ka order ₹{final} me padega."
             if lang == "hi" else
             f"Yes — orders above ₹{mn} get {pct}% off, so your ₹{amt} order comes to ₹{final}.")
    else:
        a = (f"Bulk discount ({pct}%) ₹{mn} se upar ke order pe milta hai. ₹{amt} ke order pe nahi lagega."
             if lang == "hi" else
             f"The {pct}% bulk discount applies to orders above ₹{mn}, so it doesn't apply to ₹{amt}.")
    return ex.done(a)


# ═══════════════════════════════════════════════════════════════
# FAMILY: MQ → DATE (delivery / ready dates)
# ═══════════════════════════════════════════════════════════════
DELIV_T = {
    "hi": ["aaj order karu to [delivery] kab tak hogi?", "abhi order kiya to [kab aayega]?",
           "[delivery] kitne din me hogi?", "[delivery] kab tak milegi?", "aaj order kiya to [delivery kis date] tak hogi?",
           "[delivery] kis din tak pahunchegi?", "order kiya to [ghar kab tak] aayega?"],
    "en": ["if I order today, [when will it be delivered]?", "how many days for [delivery]?",
           "when will my order [arrive] if I order now?", "if I order today when will it [reach] me?",
           "[delivery date] if I order now?"],
}
READY_T = {
    "hi": ["aaj kapda de doon to [kitne din] me ready hoga?", "aaj diya to [kab tak ready] hoga?",
           "{g} aaj diya to [kis din] milega?", "{g} [kitne din] me silega?", "{g} ki stitching [kitne din] me hogi?"],
    "en": ["if I give it today, [how many days] until it's ready?", "if I drop off a {g} today, [how many days] for stitching?"],
}


def fam_mq_date():
    new_now()
    tailor = R.random() < 0.25
    kb = new_kb("tailor" if tailor else pick([t for t, s in M.MERCHANT_TYPES.items() if s["delivery"]]))
    lang = L()
    want = "Stitching time:" if tailor else "Delivery:"
    T_ = READY_T if tailor else DELIV_T
    for _ in range(4):
        user, span = fill(pick(T_[lang]), lang, g=pick(["blouse", "suit", "kurti", "kapda", "shirt", "pant"]),
                          p=pick(kb["catalog"])["name"].lower())
        if T.mq_search(span).split("\n")[0].startswith(want):
            break
    else:
        raise Skip("delivery retrieval")
    ex = Ex("mq_date", user)
    res = ex.tool("MQ_SEARCH", span).split("\n")[0]
    m = re.search(r"in (\d+) days?|takes (\d+) days", res)
    if m:
        n = m.group(1) or m.group(2)
        date = ex.tool("DATE", f"today + {n} days")
        if lang == "hi":
            a = pick([f"{'Stitching' if tailor else 'Delivery'} me {n} din lagte hain, toh {date} tak {'ready ho jayega' if tailor else 'pahuch jayega'}.",
                      f"Aaj ke hisaab se {date} tak {'ready' if tailor else 'deliver'} ho jayega ({n} din)."])
        else:
            a = pick([f"It takes {n} days, so you can expect it by {date}.",
                      f"{'Ready' if tailor else 'Delivered'} by {date} ({n} days from today)."])
        if not tailor and "charge" in res:
            a += " " + res.split(". ", 1)[1] if ". " in res else ""
    elif "Same-day" in res:
        a = (f"Same-day delivery hai — aaj hi mil jayega. {res.split('. ', 1)[1]}" if lang == "hi"
             else f"We do same-day delivery, so you'll get it today. {res.split('. ', 1)[1]}")
    elif "No home delivery" in res:
        a = ("Sorry, home delivery nahi hai — shop se pickup karna hoga." if lang == "hi"
             else "Sorry, we don't do home delivery — pickup from the shop only.")
    else:
        raise Skip("unparsed delivery")
    return ex.done(a)


# ═══════════════════════════════════════════════════════════════
# FAMILY: general CALCULATE (business + everyday + SDE)
# ═══════════════════════════════════════════════════════════════
PRICES = [99, 149, 199, 249, 299, 349, 399, 449, 499, 599, 649, 799, 899, 999, 1199, 1299, 1499, 1799,
          1999, 2499, 2999, 3499, 4999, 5999, 7999, 9999, 12000, 14999, 18500, 25000, 32000, 49999, 75000]
GST_RATES = [5, 5, 18, 18, 18, 40]


def _p(lang, hi, en):
    """Pick a template; chunk-2 paraphrase pools are keyed by the call's first Hindi template."""
    base = hi if lang == "hi" else en
    extra = leak_filter(B.EXTRA.get(hi[0], {}).get(lang, []))
    return pick(base + extra)


def calc_gst_add():
    lang = L(); p = pick(PRICES); r = pick(GST_RATES)
    user, _ = fill(_p(lang, ["₹{p} pe {r}% GST kitna lagega?", "₹{p} + {r}% GST total kitna?",
                             "customer ka bill ₹{p} hai, {r}% GST add karo", "{r}% GST on {p} kitna hua"],
                      ["what's {r}% GST on ₹{p}?", "add {r}% GST to ₹{p}", "₹{p} plus {r}% GST — final amount?",
                       "bill is ₹{p}, add {r}% GST"]), lang, p=p, r=r)
    ex = Ex("calc_gst_add", user)
    g = ex.tool("CALCULATE", f"{p} * {r} / 100")
    tot = ex.tool("CALCULATE", f"{p} + {g}")
    a = (pick([f"{r}% GST = ₹{g}. Total ₹{tot}.", f"GST ₹{g} lagega, total ₹{tot} hoga."]) if lang == "hi"
         else pick([f"{r}% GST is ₹{g}, so the total is ₹{tot}.", f"GST: ₹{g}. Final amount: ₹{tot}."]))
    return ex.done(a)


def calc_gst_inclusive():
    lang = L(); r = pick([5, 18]); p = pick(PRICES)
    user, _ = fill(_p(lang, ["₹{p} me {r}% GST included hai, base price kya hoga?",
                             "₹{p} incl. {r}% GST — GST kitna hai isme?", "₹{p} GST ke saath hai ({r}%), bina GST kitna?"],
                      ["₹{p} includes {r}% GST. What's the price before GST?",
                       "how much GST is inside ₹{p} at {r}%?", "₹{p} inclusive of {r}% GST, base amount?"]), lang, p=p, r=r)
    ex = Ex("calc_gst_incl", user)
    base = ex.tool("CALCULATE", f"{p} * 100 / (100 + {r})")
    g = ex.tool("CALCULATE", f"{p} - {base}")
    a = (pick([f"Base price ₹{base}, GST ₹{g}.", f"Bina GST ₹{base} hai, aur GST ₹{g}."]) if lang == "hi"
         else pick([f"Price before GST: ₹{base}. GST: ₹{g}.", f"Base ₹{base} + GST ₹{g}."]))
    return ex.done(a)


def calc_cgst_sgst():
    lang = L(); p = pick(PRICES); r = pick([5, 18])
    user, _ = fill(_p(lang, ["₹{p} pe {r}% GST ka CGST aur SGST alag batao", "₹{p} bill, {r}% GST — CGST SGST kitna?"],
                      ["split {r}% GST on ₹{p} into CGST and SGST", "CGST and SGST on ₹{p} at {r}%?"]), lang, p=p, r=r)
    ex = Ex("calc_cgst", user)
    g = ex.tool("CALCULATE", f"{p} * {r} / 100")
    half = ex.tool("CALCULATE", f"{g} / 2")
    a = (f"Total GST ₹{g}: CGST ₹{half} + SGST ₹{half}." if lang == "hi"
         else f"Total GST is ₹{g}, split as CGST ₹{half} and SGST ₹{half}.")
    return ex.done(a)


def calc_discount():
    lang = L(); p = pick(PRICES); d = pick([5, 10, 15, 20, 25, 30, 40, 50])
    user, _ = fill(_p(lang, ["₹{p} pe {d}% discount laga ke kitna hoga?", "MRP ₹{p}, {d}% off, kitna charge karu?",
                             "{d}% off on ₹{p}?", "customer {d}% discount maang raha hai ₹{p} pe"],
                      ["₹{p} with {d}% off?", "sale price of ₹{p} after {d}% discount?", "{d}% off on ₹{p} — final price?"]),
                   lang, p=p, d=d)
    ex = Ex("calc_discount", user)
    f = ex.tool("CALCULATE", f"{p} * (1 - {d} / 100)")
    a = (pick([f"{d}% discount ke baad ₹{f}.", f"Final price ₹{f} hoga."]) if lang == "hi"
         else pick([f"After {d}% off: ₹{f}.", f"The final price is ₹{f}."]))
    return ex.done(a)


def calc_discount_gst():
    lang = L(); p = pick(PRICES); d = pick([5, 10, 15, 20]); r = pick([5, 18])
    user, _ = fill(_p(lang, ["₹{p} pe {d}% discount, phir {r}% GST — final kitna?", "MRP ₹{p}, {d}% off aur {r}% GST lagao"],
                      ["₹{p} item, {d}% discount then {r}% GST?", "price after {d}% off and {r}% GST on ₹{p}?"]),
                   lang, p=p, d=d, r=r)
    ex = Ex("calc_disc_gst", user)
    a1 = ex.tool("CALCULATE", f"{p} * (1 - {d} / 100)")
    a2 = ex.tool("CALCULATE", f"{a1} * (1 + {r} / 100)")
    a = (f"Discount ke baad ₹{a1}, GST ke saath ₹{a2}." if lang == "hi"
         else f"After {d}% off: ₹{a1}. With {r}% GST: ₹{a2}.")
    return ex.done(a)


def calc_margin():
    lang = L(); c = pick([50, 80, 120, 150, 200, 250, 300, 450, 500, 750, 1000, 1500, 2200, 5000])
    s = c + R.randint(max(1, c // 10), c)
    user, _ = fill(_p(lang, ["₹{c} me kharida, ₹{s} me becha. margin kitna?", "cost ₹{c}, selling ₹{s}, profit % kya hai?"],
                      ["cost ₹{c}, selling at ₹{s}. margin?", "bought for ₹{c}, sold for ₹{s} — profit and margin?"]),
                   lang, c=c, s=s)
    ex = Ex("calc_margin", user)
    pr = ex.tool("CALCULATE", f"{s} - {c}")
    mg = ex.tool("CALCULATE", f"({s} - {c}) / {s} * 100")
    a = (f"Profit ₹{pr}, margin {mg}% (selling price pe)." if lang == "hi"
         else f"Profit is ₹{pr}; margin is {mg}% of the selling price.")
    return ex.done(a)


def calc_markup():
    lang = L(); c = pick([40, 60, 100, 150, 240, 300, 480, 650, 900, 1200]); m = pick([10, 15, 20, 25, 30, 40, 50])
    user, _ = fill(_p(lang, ["₹{c} cost pe {m}% markup lagau to selling price kya hoga?", "cost ₹{c}, {m}% profit chahiye, kitne me bechu?"],
                      ["cost is ₹{c}, I want {m}% markup. selling price?", "what should I sell at for {m}% markup on ₹{c}?"]),
                   lang, c=c, m=m)
    ex = Ex("calc_markup", user)
    sp = ex.tool("CALCULATE", f"{c} * (1 + {m} / 100)")
    a = f"₹{sp} me becho." if lang == "hi" else f"Sell it at ₹{sp}."
    return ex.done(a)


def calc_emi():
    lang = L(); P = pick([25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 1000000])
    r = pick([9, 10, 10.5, 11, 12, 13, 14, 16, 18]); n = pick([6, 12, 18, 24, 36, 48, 60])
    amt = (f"{P // 100000} lakh" if P % 100000 == 0 and R.random() < 0.4 else
           f"{P // 1000}k" if P % 1000 == 0 and R.random() < 0.1 else str(P))
    user, _ = fill(_p(lang, ["₹{amt} loan, {r}% interest, {n} months — EMI kitni?", "{amt} ka loan {n} mahine ke liye {r}% pe, EMI batao",
                             "loan {amt} ka, {r}% saalana, {n} mahine — emi?", "{amt} loan {r}% pe {n} months ki kisht kitni?"],
                      ["EMI for ₹{amt} at {r}% for {n} months?", "loan ₹{amt}, {r}% annual, {n} months. monthly EMI?",
                       "monthly installment on a {amt} loan at {r}% over {n} months?"]),
                   lang, amt=amt, r=r, n=n)
    ex = Ex("calc_emi", user)
    emi = ex.tool("CALCULATE", f"emi({P}, {r}, {n})")
    tot = ex.tool("CALCULATE", f"{emi} * {n}")
    intr = ex.tool("CALCULATE", f"{tot} - {P}")
    a = (f"EMI ₹{emi} per month. Total ₹{tot} doge, jisme ₹{intr} interest hai." if lang == "hi"
         else f"EMI: ₹{emi}/month. Total paid ₹{tot}, of which ₹{intr} is interest.")
    return ex.done(a)


def calc_simple_interest():
    lang = L(); P = pick([10000, 20000, 50000, 100000, 250000]); r = pick([6, 7, 8, 9, 10, 12]); y = pick([1, 2, 3, 5])
    user, _ = fill(_p(lang, ["₹{amt} pe {r}% simple interest {y} saal ka kitna?", "{amt} FD {r}% pe {y} saal, interest kitna milega (simple)?"],
                      ["simple interest on ₹{amt} at {r}% for {y} {yw}?"]), lang, amt=P, r=r, y=y, yw="year" if y == 1 else "years")
    ex = Ex("calc_si", user)
    si = ex.tool("CALCULATE", f"{P} * {r} * {y} / 100")
    tot = ex.tool("CALCULATE", f"{P} + {si}")
    a = f"Interest ₹{si}, total ₹{tot}." if lang == "hi" else f"Interest: ₹{si}. Total: ₹{tot}."
    return ex.done(a)


def calc_split():
    lang = L(); b = pick([450, 780, 1200, 1350, 1840, 2400, 3150, 4999, 6200]); n = R.randint(2, 9)
    user, _ = fill(_p(lang, ["₹{b} ka bill {n} logo me baanto", "{n} dost, bill ₹{b}, har ek ka kitna?"],
                      ["split ₹{b} between {n} people", "bill is ₹{b} for {n} of us, how much each?"]), lang, b=b, n=n)
    ex = Ex("calc_split", user)
    each = ex.tool("CALCULATE", f"{b} / {n}")
    a = f"Har ek ₹{each} dega." if lang == "hi" else f"₹{each} each."
    return ex.done(a)


def calc_change():
    lang = L(); b = R.randint(37, 1950); g = pick([x for x in [50, 100, 200, 500, 1000, 2000] if x > b] or [2000])
    if g <= b:
        raise Skip("change")
    user, _ = fill(_p(lang, ["bill ₹{b} hai, customer ne ₹{g} diye, kitna wapas du?", "₹{g} me se ₹{b} kaat ke kitna bacha?"],
                      ["bill ₹{b}, customer paid ₹{g}. change?", "how much change from ₹{g} for a ₹{b} bill?"]), lang, b=b, g=g)
    ex = Ex("calc_change", user)
    c = ex.tool("CALCULATE", f"{g} - {b}")
    a = f"₹{c} wapas dena hai." if lang == "hi" else f"Give back ₹{c}."
    return ex.done(a)


def calc_per_unit():
    lang = L(); q = pick([5, 10, 25, 50, 100]); unit = pick(["kg", "litre", "piece", "packet"]); t = R.randrange(q * 20, q * 400, 10)
    user, _ = fill(_p(lang, ["{q} {u} ka ₹{t}, per {u} kitna pada?", "₹{t} me {q} {u} liya, ek {u} ka rate?"],
                      ["₹{t} for {q} {u}. price per {u}?", "{q} {u} cost ₹{t}, what's the per-{u} rate?"]), lang, q=q, u=unit, t=t)
    ex = Ex("calc_per_unit", user)
    r = ex.tool("CALCULATE", f"{t} / {q}")
    a = f"₹{r} per {unit}." if lang == "en" else f"Ek {unit} ₹{r} ka pada."
    return ex.done(a)


def calc_growth():
    lang = L(); a0 = R.randrange(20000, 200000, 500); a1 = int(a0 * R.uniform(0.7, 1.6)) // 100 * 100
    user, _ = fill(_p(lang, ["pichle mahine sale ₹{a0}, is mahine ₹{a1}. kitna % change?", "sales {a0} se {a1} hui, growth %?",
                             "is mahine {a1} ki sale, pichle mahine {a0}. growth kitni?", "{a0} se {a1} hua, kitna percent badha?"],
                      ["sales went from ₹{a0} to ₹{a1}. percentage change?", "last month ₹{a0}, this month ₹{a1} — growth %?",
                       "this month {a1}, last month {a0}. growth?"]),
                   lang, a0=a0, a1=a1)
    ex = Ex("calc_growth", user)
    g = ex.tool("CALCULATE", f"pct_change({a0}, {a1})")
    up = not g.startswith("-")
    if lang == "hi":
        a = f"Sales {g}% {'badhi' if up else 'ghati'} hai." if up else f"Sales {g}% change hui (kami aayi)."
    else:
        a = f"That's a {g}% change ({'up' if up else 'down'})."
    return ex.done(a)


def calc_daily_target():
    lang = L(); t = R.randrange(50000, 1500000, 5000); d = pick([22, 24, 25, 26, 30])
    user, _ = fill(_p(lang, ["mahine ka target ₹{t} hai, {d} din kaam, roz kitna bechna padega?"],
                      ["monthly target ₹{t}, {d} working days. daily target?"]), lang, t=t, d=d)
    ex = Ex("calc_target", user)
    r = ex.tool("CALCULATE", f"{t} / {d}")
    a = f"Roz ₹{r} ki sale chahiye." if lang == "hi" else f"You need ₹{r} per day."
    return ex.done(a)


def calc_salary():
    lang = L(); m = pick([12000, 15000, 18000, 22000, 25000, 30000, 45000, 60000, 85000])
    if R.random() < 0.5:
        user, _ = fill(_p(lang, ["salary ₹{m} mahina, saal ka kitna?", "₹{m} monthly ka annual kitna?"],
                          ["₹{m} a month — yearly?", "annual salary for ₹{m}/month?"]), lang, m=m)
        ex = Ex("calc_salary", user); r = ex.tool("CALCULATE", f"{m} * 12")
        a = f"Saal ka ₹{r}." if lang == "hi" else f"₹{r} per year."
    else:
        d = pick([26, 30]); days = R.randint(5, d)
        user, _ = fill(_p(lang, ["staff ki salary ₹{m}, {d} din ka mahina, usne {x} din kaam kiya, kitna du?"],
                          ["salary ₹{m} for {d} days, worked {x} days. how much to pay?"]), lang, m=m, d=d, x=days)
        ex = Ex("calc_salary", user); r = ex.tool("CALCULATE", f"{m} / {d} * {days}")
        a = f"₹{r} dena hai." if lang == "hi" else f"Pay ₹{r}."
    return ex.done(a)


CONVERSIONS = [
    ("{v} km kitne miles?", "{v} km in miles?", "{v} / 1.60934", "miles", "{v} km = {r} miles."),
    ("{v} miles kitne km?", "{v} miles to km", "{v} * 1.60934", "km", "{v} miles = {r} km."),
    ("{v} pound kitne kg?", "{v} lbs to kg", "{v} * 0.453592", "kg", "{v} lbs = {r} kg."),
    ("{v} kg kitne pound?", "{v} kg in pounds?", "{v} * 2.20462", "lbs", "{v} kg = {r} lbs."),
    ("{v} feet kitne meter?", "{v} feet to meters", "{v} * 0.3048", "m", "{v} ft = {r} m."),
    ("{v} inch kitne cm?", "{v} inches in cm", "{v} * 2.54", "cm", "{v} in = {r} cm."),
    ("{v} sq ft kitne sq meter?", "{v} sq ft to sq m", "{v} * 0.0929", "sq m", "{v} sq ft = {r} sq m."),
    ("{v}°F kitna celsius?", "{v}°F in Celsius?", "({v} - 32) * 5 / 9", "°C", "{v}°F = {r}°C."),
    ("{v}°C kitna fahrenheit?", "{v}°C to Fahrenheit", "{v} * 9 / 5 + 32", "°F", "{v}°C = {r}°F."),
]


def calc_convert():
    lang = L(); c = pick(CONVERSIONS); v = pick([R.randint(1, 500), round(R.uniform(1, 100), 1)])
    user, _ = fill(c[0] if lang == "hi" else c[1], lang, v=v)
    ex = Ex("calc_convert", user)
    r = ex.tool("CALCULATE", c[2].format(v=v))
    return ex.done(c[4].format(v=v, r=r))


def calc_area():
    lang = L(); l = R.randint(6, 80); w = R.randint(5, 60)
    user, _ = fill(_p(lang, ["{l} x {w} feet ki dukaan ka area?", "{l} ft by {w} ft, kitna sq ft?"],
                      ["area of a {l} x {w} ft shop?", "{l}ft x {w}ft in sq ft and sq m?"]), lang, l=l, w=w)
    ex = Ex("calc_area", user)
    a1 = ex.tool("CALCULATE", f"{l} * {w}")
    if R.random() < 0.5:
        a2 = ex.tool("CALCULATE", f"{a1} * 0.0929")
        a = f"{a1} sq ft ({a2} sq m)."
    else:
        a = f"{a1} sq ft."
    return ex.done(a)


def calc_rent_per_sqft():
    lang = L(); rent = pick([15000, 25000, 40000, 65000, 90000]); area = pick([200, 350, 500, 800, 1200])
    user, _ = fill(_p(lang, ["{a} sq ft ka rent ₹{r}, per sq ft kitna?"], ["rent ₹{r} for {a} sq ft. per sq ft rate?"]),
                   lang, a=area, r=rent)
    ex = Ex("calc_rent", user)
    x = ex.tool("CALCULATE", f"{rent} / {area}")
    return ex.done(f"₹{x} per sq ft." if lang == "en" else f"₹{x} per sq ft pad raha hai.")


def calc_percent_of():
    lang = L(); v = pick([200, 500, 1200, 2500, 8000, 15000, 40000, 120000]); p = pick([2, 3, 5, 8, 12.5, 15, 18, 20, 33, 75])
    user, _ = fill(_p(lang, ["{v} ka {p}% kitna?", "{p} percent of {v} batao"], ["what's {p}% of {v}?", "{p}% of {v}"]),
                   lang, v=v, p=p)
    ex = Ex("calc_pct", user)
    r = ex.tool("CALCULATE", f"{v} * {p} / 100")
    return ex.done(f"{p}% of {v} = {r}." if lang == "en" else f"{v} ka {p}% = {r}.")


def calc_average():
    lang = L(); xs = [R.randrange(500, 25000, 10) for _ in range(R.randint(3, 6))]
    s = ", ".join(str(x) for x in xs)
    user, _ = fill(_p(lang, ["is hafte ki sale: {s}. average kitna?", "average nikalo: {s}"],
                      ["daily sales were {s}. average?", "average of {s}?"]), lang, s=s)
    ex = Ex("calc_avg", user)
    r = ex.tool("CALCULATE", f"({' + '.join(str(x) for x in xs)}) / {len(xs)}")
    return ex.done(f"Average ₹{r}." if lang == "hi" else f"The average is {r}.")


def calc_basic():
    lang = L(); op = pick(["+", "-", "*", "/"])
    a, b = R.randint(2, 99999), R.randint(2, 9999)
    if op == "*":
        a, b = R.randint(2, 9999), R.randint(2, 999)
    if op == "-" and b > a:
        a, b = b, a
    sym = {"+": pick(["+", "plus", "aur"]), "-": pick(["-", "minus"]), "*": pick(["*", "x", "×", "into", "times"]),
           "/": pick(["/", "÷", "divided by"])}[op]
    if sym == "aur" and lang == "en":
        sym = "+"
    user, _ = fill(pick(["{a} {s} {b}", "{a} {s} {b} kitna?", "what's {a} {s} {b}?", "{a}{s}{b}=?",
                         "calculate {a} {s} {b}", "{a} {s} {b} batao"]), lang, a=a, b=b, s=sym)
    ex = Ex("calc_basic", user)
    r = ex.tool("CALCULATE", f"{a} {op} {b}")
    return ex.done(pick([f"{r}", f"= {r}", f"{a} {op} {b} = {r}", f"The answer is {r}."]))


def calc_sde():
    lang = L(); kind = R.random()
    if kind < 0.4:
        n = pick([1000, 50000, 1000000, 10000000, 100000000]); b = pick([4, 8, 16, 64])
        user, _ = fill(pick(["memory for {n} items of {b} bytes in MB?", "{n} records × {b} bytes kitne MB?"]), lang, n=n, b=b)
        ex = Ex("calc_sde", user); r = ex.tool("CALCULATE", f"{n} * {b} / 1048576")
        return ex.done(f"About {r} MB.")
    if kind < 0.7:
        rps = pick([50, 120, 300, 1500, 8000]);
        user, _ = fill(pick(["{r} requests per second means how many per day?", "{r} rps ka daily kitna hua?"]), lang, r=rps)
        ex = Ex("calc_sde", user); x = ex.tool("CALCULATE", f"{rps} * 60 * 60 * 24")
        return ex.done(f"{x} requests per day.")
    gb = pick([5, 20, 100, 500]); mbps = pick([10, 50, 100, 1000])
    user, _ = fill(pick(["how long to download {g} GB at {m} Mbps in minutes?", "{g} GB file {m} Mbps pe kitne minute?"]), lang, g=gb, m=mbps)
    ex = Ex("calc_sde", user); x = ex.tool("CALCULATE", f"{gb} * 1024 * 8 / {mbps} / 60")
    return ex.done(f"About {x} minutes.")


def calc_staff_total():
    lang = L(); n = pick([2, 3, 4, 5, 6, 10, 12]); s = R.randrange(6000, 40000, 500)
    nd = qty_text(n, lang)
    if R.random() < 0.6:
        user, _ = fill(_p(lang, ["{n} staff hai, har ek ko {s} salary, total kitna dena hai", "{n} karigar, sabko {s} mahina, total salary?",
                                 "{n} logo ko {s} {s2} dena hai, kul kitna?"],
                          ["{n} employees at {s} each, total salary?", "I pay {n} staff {s} each. total?"]),
                       lang, n=nd, s=s, s2="each")
        ex = Ex("calc_staff", user); t = ex.tool("CALCULATE", f"{n} * {s}")
        return ex.done(f"Total ₹{t} dena hoga." if lang == "hi" else f"Total salary: ₹{t}.")
    d = pick([4, 6, 10, 15, 26]); w = pick([350, 450, 500, 600, 750, 900])
    user, _ = fill(_p(lang, ["{n} mazdoor, {d} din, {w} rupaye roz — kitna hua?", "{n} helper {d} din ke liye, {w} per day, total?"],
                      ["{n} workers for {d} days at ₹{w}/day — total?"]), lang, n=nd, d=d, w=w)
    ex = Ex("calc_staff", user); t = ex.tool("CALCULATE", f"{n} * {d} * {w}")
    return ex.done(f"Kul ₹{t}." if lang == "hi" else f"Total: ₹{t}.")


def calc_lakh():
    lang = L(); L_ = pick([1, 1.5, 2, 2.5, 3, 5, 10, 12, 25]); p = pick([2, 5, 8, 10, 12, 18])
    val = int(L_ * 100000)
    user, _ = fill(_p(lang, ["{l} lakh ka {p}% kitna?", "{l} lakh pe {p}% kitna banta hai", "{l} lakh ka {p} percent"],
                      ["what's {p}% of {l} lakh?", "{p}% of {l} lakh rupees?"]), lang, l=f"{L_:g}", p=p)
    ex = Ex("calc_lakh", user); r = ex.tool("CALCULATE", f"{val} * {p} / 100")
    return ex.done(f"₹{r}." if lang == "en" else f"₹{r} hota hai.")


def calc_dozen():
    lang = L(); d = pick([2, 3, 4, 5, 10]); pr = pick([36, 48, 60, 72, 84, 96, 120])
    if R.random() < 0.5:
        user, _ = fill(_p(lang, ["{d} dozen kele, {pr} rupaye dozen, kitna hua?", "{d} darjan ande {pr} ke hisaab se?"],
                          ["{d} dozen eggs at {pr} per dozen?", "{d} dozen bananas, ₹{pr} a dozen — total?"]), lang, d=d, pr=pr)
        ex = Ex("calc_dozen", user); r = ex.tool("CALCULATE", f"{d} * {pr}")
    else:
        k = pick([3, 5, 7, 8, 9])
        user, _ = fill(_p(lang, ["ek dozen {pr} ka hai, {k} piece kitne ke?"], ["a dozen costs {pr}, how much for {k}?"]), lang, pr=pr, k=k)
        ex = Ex("calc_dozen", user); r = ex.tool("CALCULATE", f"{pr} / 12 * {k}")
    return ex.done(f"₹{r}." if lang == "en" else f"₹{r} hua.")


CALC_FAMILIES = [
    (calc_staff_total, 2), (calc_lakh, 1), (calc_dozen, 1),
    (calc_gst_add, 3), (calc_gst_inclusive, 2), (calc_cgst_sgst, 1), (calc_discount, 3), (calc_discount_gst, 2),
    (calc_margin, 2), (calc_markup, 1), (calc_emi, 2), (calc_simple_interest, 1), (calc_split, 2), (calc_change, 2),
    (calc_per_unit, 2), (calc_growth, 1), (calc_daily_target, 1), (calc_salary, 2), (calc_convert, 2), (calc_area, 1),
    (calc_rent_per_sqft, 1), (calc_percent_of, 2), (calc_average, 1), (calc_basic, 3), (calc_sde, 1),
]


def fam_calc():
    new_now()
    f = R.choices([f for f, _ in CALC_FAMILIES], [w for _, w in CALC_FAMILIES])[0]
    return f()


# ═══════════════════════════════════════════════════════════════
# FAMILY: DATE
# ═══════════════════════════════════════════════════════════════
def _date_text(d):
    style = R.random()
    if style < 0.3:
        return f"{d:%B} {d.day}, {d.year}"
    if style < 0.5:
        return f"{d.day} {d:%B} {d.year}"
    if style < 0.65:
        return f"{d.day} {d:%b} {d.year}"
    if style < 0.8:
        return d.isoformat()
    return f"{d.day:02d}/{d.month:02d}/{d.year}"


FESTIVAL_NAMES = {
    "diwali": ["Diwali", "diwali", "Deepavali"], "holi": ["Holi", "holi"], "raksha bandhan": ["Raksha Bandhan", "rakhi"],
    "dussehra": ["Dussehra", "dussehra", "Dasara"], "navratri": ["Navratri", "navratri"],
    "dhanteras": ["Dhanteras", "dhanteras"], "janmashtami": ["Janmashtami", "janmashtami"],
    "ganesh chaturthi": ["Ganesh Chaturthi", "ganpati"], "christmas": ["Christmas", "christmas"],
    "new year": ["New Year", "new year"], "republic day": ["Republic Day", "26 January"],
    "independence day": ["Independence Day", "15 August"], "makar sankranti": ["Makar Sankranti", "sankranti"],
    "maha shivaratri": ["Maha Shivaratri", "shivratri"], "ram navami": ["Ram Navami", "ram navami"],
    "bhai dooj": ["Bhai Dooj", "bhai dooj"], "karwa chauth": ["Karwa Chauth", "karva chauth"],
}


def fam_date():
    now = new_now(); lang = L(); k = R.random()
    if k < 0.18:
        user, _ = fill(_p(lang, ["aaj kya date hai?", "aaj kaun sa din hai", "date kya hai aaj", "aaj ki date batao"],
                          ["what's today's date?", "what day is it today?", "today's date?", "date?"]), lang)
        ex = Ex("date_today", user); r = ex.tool("DATE", "today")
        return ex.done(f"Aaj {r} hai." if lang == "hi" else pick([f"Today is {r}.", f"It's {r}."]))
    if k < 0.26:
        user, _ = fill(_p(lang, ["abhi kitne baje hain?", "time kya hua?", "time batao", "kitne baj gaye?", "abhi ka time kya hai",
                                 "time kitna ho raha hai?", "ghadi me kya time hai"],
                          ["what time is it?", "current time?", "what's the time now?", "time please"]), lang)
        ex = Ex("date_time", user); r = ex.tool("DATE", "time")
        return ex.done(f"Abhi {r} hai." if lang == "hi" else f"It's {r}.")
    if k < 0.40:
        d = now.date() + dt.timedelta(days=R.randint(-400, 400)); ds = _date_text(d)
        user, _ = fill(_p(lang, ["{d} ko kaun sa din hai?", "{d} kaunsa din tha?", "{d} ko konsa vaar hai"],
                          ["what day is {d}?", "which weekday was {d}?", "{d} — what day of the week?"]), lang, d=ds)
        ex = Ex("date_weekday", user); r = ex.tool("DATE", ds)
        return ex.done(f"{r}." if lang == "en" else f"{ds} ko {r.split(',')[0]} hai.")
    if k < 0.55:
        d = now.date() + dt.timedelta(days=R.randint(2, 300)); ds = _date_text(d)
        user, _ = fill(_p(lang, ["{d} me kitne din bache hain?", "{d} tak kitne din hain?", "{d} aane me kitne din?"],
                          ["how many days until {d}?", "days left till {d}?"]), lang, d=ds)
        ex = Ex("date_until", user); r = ex.tool("DATE", f"days until {ds}")
        n = r.split(" ")[0]
        return ex.done(f"{n} din bache hain ({r.split('until ')[1]})." if lang == "hi" else f"{r}.")
    if k < 0.65:
        n = pick([3, 5, 7, 10, 15, 21, 30, 45, 90])
        user, _ = fill(_p(lang, ["aaj se {n} din baad kaun si date hogi?", "{n} din baad kya date hai?"],
                          ["what's the date {n} days from today?", "date after {n} days?"]), lang, n=n)
        ex = Ex("date_plus", user); r = ex.tool("DATE", f"today + {n} days")
        return ex.done(f"{n} din baad {r} hogi." if lang == "hi" else f"{n} days from today is {r}.")
    if k < 0.72:
        d1 = now.date() - dt.timedelta(days=R.randint(10, 500)); d2 = d1 + dt.timedelta(days=R.randint(5, 400))
        a, b = _date_text(d1), _date_text(d2)
        user, _ = fill(_p(lang, ["{a} se {b} tak kitne din?"], ["how many days between {a} and {b}?"]), lang, a=a, b=b)
        ex = Ex("date_between", user); r = ex.tool("DATE", f"days between {a} and {b}")
        return ex.done(f"{r.split(' between')[0]} hain." if lang == "hi" else f"{r}.")
    if k < 0.78:
        d = now.date() + dt.timedelta(days=R.randint(3, 40)); ds = _date_text(d)
        user, _ = fill(_p(lang, ["{d} tak kitne working days hain?"], ["how many working days until {d}?"]), lang, d=ds)
        ex = Ex("date_working", user); r = ex.tool("DATE", f"working days until {ds}")
        return ex.done(f"{r}." if lang == "en" else f"{r.split(' (')[0]} hain (Sunday chhod ke).")
    # festivals
    key = pick(list(FESTIVAL_NAMES)); name = pick(FESTIVAL_NAMES[key])
    if R.random() < 0.6:
        user, _ = fill(_p(lang, ["{f} me kitne din bache?", "{f} kitne din me hai?", "{f} aane me kitne din hain?"],
                          ["how many days until {f}?", "days left for {f}?"]), lang, f=name)
        ex = Ex("date_festival", user); r = ex.tool("DATE", f"days until {name}")
    else:
        user, _ = fill(_p(lang, ["{f} kab hai?", "is saal {f} kis din hai?", "{f} ki date kya hai"],
                          ["when is {f}?", "what date is {f}?"]), lang, f=name)
        ex = Ex("date_festival", user); r = ex.tool("DATE", name)
    if r.startswith("Error"):
        a = (f"{name} ki exact date abhi mere paas nahi hai. Calendar me check kar lijiye." if lang == "hi"
             else f"I don't have the exact date for {name} yet — please check a calendar.")
    elif "until" in r:
        n, date = r.split(" until ")
        a = f"{name} {date} ko hai — {n.replace('days', 'din').replace('day', 'din')} baaki." if lang == "hi" else f"{name} is on {date} — {n} to go."
    elif r.endswith("is today"):
        a = f"{name} aaj hi hai! 🎉" if lang == "hi" else f"{name} is today! 🎉"
    else:
        a = f"{name} {r} ko hai." if lang == "hi" else f"{name} is on {r}."
    return ex.done(a)


# ═══════════════════════════════════════════════════════════════
# FAMILY: SEARCH (general knowledge, Wikipedia-style results)
# (title, extract) — extracts are written to be factually safe and stable.
# ═══════════════════════════════════════════════════════════════
SEARCH_FACTS = {
    "UPI": ("Unified Payments Interface", "Unified Payments Interface (UPI) is an instant real-time payment system developed by the National Payments Corporation of India. It lets users transfer money between bank accounts using a mobile app and a virtual payment address."),
    "GST": ("Goods and Services Tax (India)", "The Goods and Services Tax (GST) is an indirect tax on the supply of goods and services in India, introduced on 1 July 2017. Since 22 September 2025 most goods fall under two main slabs, 5% and 18%, with a 40% rate for luxury and sin goods."),
    "MSME": ("Micro, Small and Medium Enterprises", "Micro, Small and Medium Enterprises (MSMEs) are businesses classified in India by investment and turnover limits. They are registered through the Udyam portal and are eligible for several government schemes."),
    "Udyam registration": ("Udyam Registration", "Udyam Registration is the online registration system for MSMEs in India run by the Ministry of MSME. It is free, paperless and based on self-declaration with the business owner's Aadhaar."),
    "Aadhaar": ("Aadhaar", "Aadhaar is a 12-digit unique identity number issued by the Unique Identification Authority of India (UIDAI) to residents of India, based on biometric and demographic data."),
    "PAN card": ("Permanent Account Number", "A Permanent Account Number (PAN) is a ten-character alphanumeric identifier issued by the Indian Income Tax Department. It is required for filing income tax returns and many financial transactions."),
    "NPCI": ("National Payments Corporation of India", "The National Payments Corporation of India (NPCI) is the umbrella organisation for retail payments and settlement systems in India. It operates UPI, RuPay, IMPS and other payment systems."),
    "RuPay": ("RuPay", "RuPay is an Indian multinational card payment network launched by the National Payments Corporation of India. RuPay cards are accepted at ATMs, point-of-sale terminals and online merchants."),
    "IMPS": ("Immediate Payment Service", "Immediate Payment Service (IMPS) is an instant interbank electronic fund transfer service in India, available round the clock, operated by NPCI."),
    "inflation": ("Inflation", "Inflation is a general increase in the prices of goods and services in an economy over time. It reduces the purchasing power of money."),
    "repo rate": ("Repo rate", "The repo rate is the rate at which a central bank lends money to commercial banks. In India it is set by the Reserve Bank of India's Monetary Policy Committee."),
    "RBI": ("Reserve Bank of India", "The Reserve Bank of India (RBI) is India's central bank and regulatory body for the banking system. It was established on 1 April 1935 and manages monetary policy and the issue of currency."),
    "FSSAI": ("Food Safety and Standards Authority of India", "The Food Safety and Standards Authority of India (FSSAI) regulates food safety in India. Food businesses must obtain an FSSAI licence or registration before operating."),
    "e-way bill": ("E-way bill", "An e-way bill is an electronic document required under GST for moving goods above a specified value from one place to another in India."),
    "invoice": ("Invoice", "An invoice is a commercial document issued by a seller to a buyer listing the products, quantities and agreed prices for goods or services provided."),
    "working capital": ("Working capital", "Working capital is the difference between a business's current assets and current liabilities. It measures the money available for day-to-day operations."),
    "gross margin": ("Gross margin", "Gross margin is net sales minus the cost of goods sold, often expressed as a percentage of sales. It shows how much a company keeps from each rupee of revenue before other expenses."),
    "photosynthesis": ("Photosynthesis", "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to produce glucose and oxygen. It takes place mainly in the chloroplasts of leaf cells."),
    "black hole": ("Black hole", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. Black holes can form when very massive stars collapse."),
    "monsoon": ("Monsoon of South Asia", "The South Asian monsoon is a seasonal wind pattern that brings most of India's annual rainfall, mainly between June and September."),
    "Taj Mahal": ("Taj Mahal", "The Taj Mahal is an ivory-white marble mausoleum on the bank of the Yamuna river in Agra, India. It was commissioned in 1632 by the Mughal emperor Shah Jahan."),
    "ISRO": ("Indian Space Research Organisation", "The Indian Space Research Organisation (ISRO) is India's national space agency, headquartered in Bengaluru. It develops satellites and launch vehicles such as the PSLV and GSLV."),
    "Chandrayaan-3": ("Chandrayaan-3", "Chandrayaan-3 is an Indian lunar mission by ISRO. Its lander touched down near the Moon's south pole on 23 August 2023, making India the first country to land in that region."),
    "Diwali": ("Diwali", "Diwali, the festival of lights, is one of the major festivals of India, celebrated with lamps, sweets and fireworks. It usually falls in October or November."),
    "Holi": ("Holi", "Holi is a popular Hindu festival of colours celebrating the arrival of spring. People play with coloured powder and water."),
    "kaju katli": ("Kaju katli", "Kaju katli is an Indian sweet made from cashew nuts, sugar and ghee, usually cut into diamond shapes and sometimes topped with edible silver foil."),
    "chikankari": ("Chikan embroidery", "Chikankari is a traditional embroidery style from Lucknow, done by hand with white thread on fine fabrics like muslin and cotton."),
    "Banarasi saree": ("Banarasi sari", "A Banarasi sari is a sari made in Varanasi, known for its fine silk and gold and silver brocade work."),
    "basmati rice": ("Basmati", "Basmati is a variety of long, slender-grained aromatic rice traditionally grown in India and Pakistan."),
    "mishti doi": ("Mishti doi", "Mishti doi is a fermented sweet yogurt popular in West Bengal and Odisha, traditionally set in earthen pots."),
    "paneer": ("Paneer", "Paneer is a fresh, non-melting cheese common in South Asian cuisine, made by curdling heated milk with lemon juice or vinegar."),
    "alphonso mango": ("Alphonso mango", "The Alphonso, or hapus, is a mango cultivar from Maharashtra's Konkan region, known for its sweetness and rich flavour."),
    "tempered glass": ("Tempered glass", "Tempered glass is safety glass treated by heat or chemicals to increase its strength. When broken, it shatters into small blunt pieces."),
    "power bank": ("Battery charger", "A power bank is a portable battery that stores electrical energy and can charge devices like phones through a USB port."),
    "emulsion paint": ("Emulsion paint", "Emulsion paint is a water-based paint commonly used for interior and exterior walls. It dries quickly and has low odour."),
    "Tally": ("Tally Solutions", "Tally is accounting and business management software developed by Tally Solutions in Bengaluru, widely used by Indian small businesses for bookkeeping, inventory and GST returns."),
    "REST API": ("REST", "REST (Representational State Transfer) is an architectural style for web APIs. RESTful services use HTTP methods such as GET, POST, PUT and DELETE to work with resources."),
    "Docker": ("Docker (software)", "Docker is a platform that packages applications and their dependencies into containers, so they run the same way on any machine."),
    "Kubernetes": ("Kubernetes", "Kubernetes is an open-source system for automating deployment, scaling and management of containerised applications."),
    "Git rebase": ("Git", "In Git, rebase moves or replays a series of commits onto a new base commit, producing a linear history instead of a merge commit."),
    "load balancer": ("Load balancing (computing)", "Load balancing distributes incoming network traffic across multiple servers so that no single server becomes overwhelmed."),
    "WebSocket": ("WebSocket", "WebSocket is a protocol that provides full-duplex communication over a single TCP connection, used for real-time web applications."),
    "CORS": ("Cross-origin resource sharing", "Cross-origin resource sharing (CORS) is a browser mechanism that lets a server declare which other origins may access its resources."),
    "SQL injection": ("SQL injection", "SQL injection is a code injection attack in which malicious SQL is inserted into an application's queries. It is prevented with parameterised queries."),
    "OAuth": ("OAuth", "OAuth is an open standard for access delegation, letting users grant apps limited access to their accounts without sharing passwords."),
    "Redis": ("Redis", "Redis is an in-memory key-value data store used as a database, cache and message broker."),
    "JWT": ("JSON Web Token", "A JSON Web Token (JWT) is a compact, URL-safe token format for securely transmitting claims between two parties, often used for authentication."),
    "CDN": ("Content delivery network", "A content delivery network (CDN) is a geographically distributed group of servers that delivers web content to users from a nearby location."),
    "GraphQL": ("GraphQL", "GraphQL is a query language for APIs that lets clients request exactly the data they need in a single request."),
    "MongoDB": ("MongoDB", "MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like documents."),
    "PostgreSQL": ("PostgreSQL", "PostgreSQL is a free and open-source relational database management system emphasising extensibility and SQL compliance."),
    "hash table": ("Hash table", "A hash table is a data structure that maps keys to values using a hash function, giving average constant-time lookups."),
    "binary search": ("Binary search", "Binary search finds the position of a target value within a sorted array by repeatedly halving the search interval."),
    "Big O notation": ("Big O notation", "Big O notation describes how the running time or space of an algorithm grows as the input size grows."),
    "HTTP 404": ("HTTP 404", "HTTP 404 Not Found is a standard response code indicating that the server could not find the requested resource."),
    "TCP": ("Transmission Control Protocol", "The Transmission Control Protocol (TCP) provides reliable, ordered delivery of data between applications over an IP network."),
    "DNS": ("Domain Name System", "The Domain Name System (DNS) translates human-readable domain names into the IP addresses computers use to find each other."),
    "Linux": ("Linux", "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, first released by Linus Torvalds in 1991."),
    "Python": ("Python (programming language)", "Python is a high-level, general-purpose programming language known for its readable syntax. It was created by Guido van Rossum and first released in 1991."),
    "machine learning": ("Machine learning", "Machine learning is a field of artificial intelligence in which systems learn patterns from data to make predictions without being explicitly programmed."),
}
SEARCH_T = {
    "hi": ["{t} kya hai?", "{t} kya hota hai?", "{t} ke baare me batao", "{t} samjhao", "{t} matlab?",
           "ye {t} kya cheez hai", "{t} kisse banta hai?"],
    "en": ["what is {t}?", "explain {t}", "tell me about {t}", "what does {t} mean?", "{t}?", "quick question — what's {t}?"],
}


HOWTO = {"UPI", "IMPS", "DNS", "CDN", "load balancer", "Docker", "Kubernetes", "OAuth", "JWT", "binary search",
         "hash table", "TCP", "CORS", "WebSocket", "GST", "e-way bill", "repo rate", "Redis", "photosynthesis"}
HOWTO_T = {"hi": ["{t} kaise kaam karta hai?", "{t} kaam kaise karta hai", "{t} ka process kya hai?"],
           "en": ["how does {t} work?", "how does {t} actually work"]}
CUSTOM_Q = [  # (question, SEARCH arg as it appears in the question, fact key)
    ("Taj Mahal kisne banwaya?", "Taj Mahal", "Taj Mahal"), ("who built the taj mahal?", "taj mahal", "Taj Mahal"),
    ("taj mahal kahan hai", "taj mahal", "Taj Mahal"), ("python kisne banaya?", "python", "Python"),
    ("who created Python?", "Python", "Python"), ("linux kisne banaya", "linux", "Linux"),
    ("who made linux?", "linux", "Linux"), ("chandrayaan-3 kab land hua?", "chandrayaan-3", "Chandrayaan-3"),
    ("when did Chandrayaan-3 land?", "Chandrayaan-3", "Chandrayaan-3"), ("RBI kab bana?", "RBI", "RBI"),
    ("when was the RBI set up?", "RBI", "RBI"), ("GST kab lagu hua?", "GST", "GST"),
    ("when was GST introduced in India?", "GST", "GST"), ("ISRO ka headquarter kahan hai?", "ISRO", "ISRO"),
    ("where is ISRO based?", "ISRO", "ISRO"), ("UPI kisne banaya?", "UPI", "UPI"), ("who runs UPI?", "UPI", "UPI"),
    ("aadhaar kaun issue karta hai?", "aadhaar", "Aadhaar"), ("PAN card kyun chahiye?", "PAN card", "PAN card"),
    ("monsoon kab aata hai?", "monsoon", "monsoon"), ("FSSAI licence kya hota hai", "FSSAI", "FSSAI"),
]


def fam_search_custom():
    new_now()
    q, arg, key = pick(CUSTOM_Q)
    title, extract = SEARCH_FACTS[key]
    lang = "hi" if is_hi(q) else "en"
    user, _ = fill(q.replace("{", "(").replace("}", ")"), lang)
    i = user.lower().find(arg.lower())
    if i < 0:
        raise Skip("custom arg")
    a = user[i:i + len(arg)]
    ex = Ex("search", user)
    res = ex.fake("SEARCH", a, T.format_search_result(title, extract))
    s = first_sentences(res.split(": ", 1)[1], 2)
    return ex.done(f"Short me: {s}" if lang == "hi" and R.random() < 0.5 else s)


def first_sentences(text, n):
    parts = re.split(r"(?<=\.)\s+", text)
    return " ".join(parts[:n])


def fam_search():
    new_now(); lang = L()
    topic = pick(list(SEARCH_FACTS)); title, extract = SEARCH_FACTS[topic]
    t = pick([topic, topic.lower()]) if topic.isupper() is False else topic
    tmpl = pick(HOWTO_T[lang]) if topic in HOWTO and R.random() < 0.35 else pick(SEARCH_T[lang])
    if "kisse banta" in tmpl and topic not in ("kaju katli", "paneer", "mishti doi"):
        tmpl = SEARCH_T[lang][0]
    user, _ = fill(tmpl, lang, t=t)
    ex = Ex("search", user)
    i = user.lower().find(t.lower())
    q = user[i:i + len(t)] if i >= 0 else t
    res = ex.fake("SEARCH", q, T.format_search_result(title, extract))
    body = res.split(": ", 1)[1]
    s = first_sentences(body, pick([1, 1, 2]))
    a = (pick([f"Short me: {s}", s, f"{s}"]) if lang == "hi" else pick([s, f"In short: {s}", s]))
    return ex.done(a)


def fam_search_error():
    new_now(); lang = L()
    topic = pick(list(SEARCH_FACTS))
    user, _ = fill(pick(SEARCH_T[lang]).replace("kisse banta hai?", "kya hai?"), lang, t=topic)
    ex = Ex("search_error", user)
    if R.random() < 0.5:
        ex.fake("SEARCH", topic, "Error: search unavailable")
        a = ("Sorry, abhi search kaam nahi kar raha. Thodi der baad try karein." if lang == "hi"
             else "Sorry, search isn't working right now. Please try again in a bit.")
    else:
        ex.fake("SEARCH", topic, f"No results found for: {topic}")
        a = (f"Mujhe {topic} ke baare me kuch nahi mila. Thoda aur detail bata sakte hain?" if lang == "hi"
             else f"I couldn't find anything on {topic}. Could you give me a bit more detail?")
    return ex.done(a)


# Routing contrast: same product word, concept question → SEARCH; price question → MQ_SEARCH
def fam_route_contrast():
    pairs = {"kaju katli": "sweets", "chikankari": "ethnic_wear", "Banarasi saree": "ethnic_wear",
             "basmati rice": "kirana", "mishti doi": "sweets", "paneer": "dairy", "alphonso mango": "fruits_veg",
             "tempered glass": "mobile", "power bank": "mobile", "emulsion paint": "hardware", "Tally": "coaching"}
    topic = pick(list(pairs))
    new_now(); kb = new_kb(pairs[topic]); lang = L()
    if R.random() < 0.5:
        title, extract = SEARCH_FACTS[topic]
        user, _ = fill(pick(SEARCH_T[lang]), lang, t=topic)
        ex = Ex("route_search", user)
        res = ex.fake("SEARCH", topic, T.format_search_result(title, extract))
        return ex.done(first_sentences(res.split(": ", 1)[1], 1))
    q = topic if topic != "Tally" else "tally gst course"
    user, span = fill(pick(PRODUCT_T[lang]), lang, p=q)
    ex = Ex("route_mq", user)
    res = ex.tool("MQ_SEARCH", span)
    items, faqs = parse_mq(res)
    if not items:
        raise Skip("route mq miss")
    return ex.done(describe_items(items, lang))


# ═══════════════════════════════════════════════════════════════
# FAMILY: WEATHER
# ═══════════════════════════════════════════════════════════════
CONDS = [("sunny", 28, 42, (0, 10)), ("clear", 18, 34, (0, 10)), ("partly cloudy", 22, 35, (5, 35)),
         ("overcast", 20, 31, (20, 50)), ("light rain", 19, 29, (60, 90)), ("moderate rain", 18, 28, (70, 95)),
         ("thundery outbreaks possible", 22, 33, (55, 85)), ("mist", 9, 22, (0, 20)), ("haze", 15, 34, (0, 10)),
         ("fog", 6, 16, (0, 15))]
WEATHER_T = {
    "general": {"hi": ["{c} ka mausam?", "{c} me aaj mausam kaisa hai?", "{c} weather batao", "{c} ka temperature kya hai?",
                       "aaj {c} me mausam kaisa rahega"],
                "en": ["weather in {c}?", "how's the weather in {c}?", "{c} weather today", "{c} temperature?", "what's it like in {c} right now?"]},
    "umbrella": {"hi": ["{c} me chhata le jaun?", "{c} me aaj baarish hogi?", "aaj {c} me chhata lena padega?",
                        "{c} me baarish ho rahi hai kya?", "{c} me aaj barish ka chance hai?"],
                 "en": ["do I need an umbrella in {c}?", "will it rain in {c} today?", "should I carry an umbrella in {c}?"]},
    "jacket": {"hi": ["{c} me jacket chahiye?", "{c} me thand hai kya?", "{c} me sweater pehnu?", "{c} me aaj thandi hogi?",
                       "{c} me thand kitni hai abhi"],
               "en": ["do I need a jacket in {c}?", "is it cold in {c}?"]},
    "hot": {"hi": ["{c} me garmi hai kya?", "{c} me kitni garmi hai?", "{c} me aaj dhoop tez hai?"],
            "en": ["is it hot in {c}?", "how hot is {c} today?"]},
    "event": {"hi": ["{c} me aaj bahar event rakh sakte hai?", "{c} me outdoor sale lagau aaj?"],
              "en": ["can we hold an outdoor event in {c} today?", "good day for an outdoor stall in {c}?"]},
}


def fam_weather():
    new_now(); lang = L()
    city = pick(M.CITIES + ["Shimla", "Manali", "Darjeeling", "Goa", "Ooty", "Dubai", "Singapore", "London"])
    cond, lo, hi_, rr = pick(CONDS)
    temp = R.randint(lo, hi_); hum = R.randint(25, 95); wind = R.randint(3, 35); rain = R.randint(*rr)
    kind = R.choices(list(WEATHER_T), [3, 2, 2, 2, 1])[0]
    user, _ = fill(pick(WEATHER_T[kind][lang]), lang, c=city)
    ex = Ex("weather", user)
    i = user.lower().find(city.lower()); c = user[i:i + len(city)] if i >= 0 else city
    res = ex.fake("WEATHER", c, T.format_weather(c.title(), temp, cond, hum, wind, rain))
    H = lang == "hi"
    if kind == "general":
        a = (f"{c.title()} me abhi {temp}°C hai, {cond}. Humidity {hum}%, baarish ka chance {rain}%." if H
             else f"{c.title()}: {temp}°C, {cond}. Humidity {hum}%, {rain}% chance of rain.")
    elif kind == "umbrella":
        yes = rain >= 50
        a = ((f"Haan, chhata le jaiye — baarish ka chance {rain}% hai." if yes else f"Zarurat nahi lagti, baarish ka chance sirf {rain}% hai.") if H
             else (f"Yes, take one — {rain}% chance of rain." if yes else f"Probably not, only {rain}% chance of rain."))
    elif kind == "jacket":
        yes = temp < 18
        a = ((f"Haan, {temp}°C hai, jacket le lijiye." if yes else f"Nahi, {temp}°C hai, jacket ki zarurat nahi.") if H
             else (f"Yes, it's {temp}°C — take a jacket." if yes else f"No, it's {temp}°C, no jacket needed."))
    elif kind == "hot":
        yes = temp >= 33
        a = ((f"Haan, kaafi garmi hai — {temp}°C." if yes else f"Zyada garmi nahi, {temp}°C hai.") if H
             else (f"Yes, it's hot — {temp}°C." if yes else f"Not really, {temp}°C."))
    else:
        bad = rain >= 50 or temp >= 38
        a = ((f"Aaj outdoor theek nahi — {cond}, {temp}°C, baarish ka chance {rain}%. Indoor rakhiye." if bad
              else f"Haan, aaj theek hai — {cond}, {temp}°C, baarish ka chance {rain}%.") if H
             else (f"I'd avoid it — {cond}, {temp}°C, {rain}% chance of rain." if bad
                   else f"Looks fine — {cond}, {temp}°C, {rain}% chance of rain."))
    return ex.done(a)


def fam_weather_error():
    new_now(); lang = L(); city = pick(M.CITIES)
    user, _ = fill(pick(WEATHER_T["general"][lang]), lang, c=city)
    ex = Ex("weather_error", user)
    ex.fake("WEATHER", city, f"Error: weather unavailable for {city.title()}")
    return ex.done(f"Sorry, abhi {city.title()} ka weather nahi mil raha. Thodi der baad try karein." if lang == "hi"
                   else f"Sorry, I can't get the weather for {city.title()} right now. Please try again later.")


# ═══════════════════════════════════════════════════════════════
# FAMILY: RUN_CODE (real execution)
# ═══════════════════════════════════════════════════════════════
WORDS = ["hello world", "milo rocks", "namaste india", "python is fun", "whatsapp bot", "small model",
         "chai and code", "debug karo", "racecar", "level", "madam", "kirana store"]


def fam_code():
    new_now(); lang = L(); k = R.randrange(14)
    if k == 0:
        xs = R.sample(range(1, 200), R.randint(5, 9)); rev = R.random() < 0.3
        user, _ = fill(pick(["sort {x}" + (" descending" if rev else ""), "{x} ko sort karo" + (" ulta" if rev else "")]), lang, x=xs)
        code = f"sorted({xs}{', reverse=True' if rev else ''})"
    elif k == 1:
        w = pick(WORDS); user, _ = fill(pick(["reverse '{w}'", "'{w}' ulta karo"]), lang, w=w); code = f"'{w}'[::-1]"
    elif k == 2:
        w = pick(WORDS); user, _ = fill(pick(["is '{w}' a palindrome?", "'{w}' palindrome hai?"]), lang, w=w)
        code = f"s = '{w}'.replace(' ', '')\ns == s[::-1]"
    elif k == 3:
        n = R.randint(20, 120); user, _ = fill(pick(["primes up to {n}", "{n} tak ke prime numbers"]), lang, n=n)
        code = f"[n for n in range(2, {n} + 1) if all(n % d for d in range(2, int(n ** 0.5) + 1))]"
    elif k == 4:
        n = R.randint(5, 20); user, _ = fill(pick(["factorial of {n}", "{n} ka factorial"]), lang, n=n)
        code = f"import math\nmath.factorial({n})"
    elif k == 5:
        n = R.randint(6, 20); user, _ = fill(pick(["first {n} fibonacci numbers", "{n} fibonacci numbers do"]), lang, n=n)
        code = f"fib = [0, 1]\nwhile len(fib) < {n}:\n    fib.append(fib[-1] + fib[-2])\nfib[:{n}]"
    elif k == 6:
        xs = [R.randint(1, 9) for _ in range(R.randint(6, 10))]
        user, _ = fill(pick(["remove duplicates from {x}", "{x} me se duplicates hatao"]), lang, x=xs)
        code = f"list(dict.fromkeys({xs}))"
    elif k == 7:
        s = pick(["the cat sat on the mat the end", "aaj ka order aaj ka bill", "to be or not to be", "milo milo code code code"])
        user, _ = fill(pick(["word count in '{s}'", "'{s}' me har word kitni baar aaya?"]), lang, s=s)
        code = f"from collections import Counter\ndict(Counter('{s}'.split()))"
    elif k == 8:
        n = R.randint(10, 5000); base = pick(["bin", "hex", "oct"])
        user, _ = fill(pick(["{n} in {b}", "convert {n} to {b}"]), lang, n=n, b={"bin": "binary", "hex": "hex", "oct": "octal"}[base])
        code = f"{base}({n})"
    elif k == 9:
        name = pick(["Basmati Rice", "Kurti", "Power Bank"]); price = R.randint(99, 2999)
        js = json.dumps({"item": name, "price": price, "stock": R.randint(0, 50)})
        user, _ = fill(pick(["get price from this json: {j}", "is json me price kya hai: {j}"]), lang, j=js)
        code = f"import json\njson.loads('{js}')['price']"
    elif k == 10:
        emails = ", ".join(f"{pick(['ravi', 'neha', 'support', 'orders', 'asha'])}{R.randint(1, 99)}@example.com" for _ in range(2))
        text = f"contact {emails.split(', ')[0]} or {emails.split(', ')[1]} for help"
        user, _ = fill(pick(["extract emails from: {t}", "isme se emails nikalo: {t}"]), lang, t=text)
        code = f"import re\nre.findall(r'[\\w.]+@[\\w.]+', '{text}')"
    elif k == 11:
        n = R.randint(100, 999999); user, _ = fill(pick(["sum of digits of {n}", "{n} ke digits ka sum"]), lang, n=n)
        code = f"sum(int(d) for d in str({n}))"
    elif k == 12:   # SDE debugging: run their buggy code
        bug = pick([
            ("nums = [1, 2, 3]\nprint(nums[3])", "IndexError", "list has only 3 items, index 3 is out of range — use nums[2] or nums[-1]"),
            ("d = {'a': 1}\nprint(d['b'])", "KeyError", "key 'b' doesn't exist — use d.get('b') or check 'b' in d"),
            ("print('total: ' + 5)", "TypeError", "you can't add a str and an int — use str(5) or an f-string"),
            ("x = 10\nprint(x / 0)", "ZeroDivisionError", "you're dividing by zero — check the divisor first"),
            ("print(totl)", "NameError", "'totl' isn't defined — probably a typo for 'total'"),
            ("int('12a')", "ValueError", "'12a' isn't a valid integer — clean the string before converting"),
        ])
        user, _ = fill(pick(["why does this fail?\n{c}", "ye code error de raha hai:\n{c}", "debug this:\n{c}"]), lang, c=bug[0])
        ex = Ex("code_debug", user)
        r = ex.tool("RUN_CODE", bug[0])
        if bug[1] not in r:
            raise Skip("debug")
        return ex.done(f"{bug[1]} aa raha hai: {bug[2]}." if is_hi(user) else f"It raises {bug[1]}: {bug[2]}.")
    else:
        xs = [R.randint(10, 999) for _ in range(R.randint(4, 8))]
        user, _ = fill(pick(["max, min and sum of {x}", "{x} ka max min sum"]), lang, x=xs)
        code = f"xs = {xs}\n(max(xs), min(xs), sum(xs))"
    ex = Ex("code", user)
    r = ex.tool("RUN_CODE", code)
    if r.startswith("Error"):
        raise Skip("code error")
    a = pick([f"Result: {r}", f"Ye raha: {r}", f"{r}"]) if is_hi(user) else pick([f"Result: {r}", f"Here you go: {r}", f"{r}", f"Output: {r}"])
    return ex.done(a)


def fam_code_disabled():
    new_now(); lang = L()
    code = pick(["print(sum(range(10)))", "print(2**10)", "sorted([3, 1, 2])", "len('milo')"])
    user, _ = fill(_p(lang, ["ye code chala ke batao: {c}", "ye run karo: {c}"], ["run this: {c}", "execute: {c}"]), lang, c=code)
    ex = Ex("code_disabled", user)
    ex.fake("RUN_CODE", code, "Error: RUN_CODE is disabled")
    return ex.done("Sorry, is chat me code run karne ki suvidha nahi hai." if lang == "hi"
                   else "Sorry, running code isn't available in this chat.")


# ═══════════════════════════════════════════════════════════════
# FAMILY: tool errors (CALCULATE / DATE)
# ═══════════════════════════════════════════════════════════════
def fam_tool_error():
    new_now(); lang = L()
    if R.random() < 0.5:
        n = R.randint(10, 9999)
        user, _ = fill(_p(lang, ["{n} ko 0 se divide karo", "{n} / 0 kitna hai?"], ["{n} / 0", "what's {n} divided by 0?"]), lang, n=n)
        ex = Ex("calc_error", user); ex.tool("CALCULATE", f"{n} / 0")
        return ex.done("Zero se divide nahi kar sakte — ye undefined hota hai." if lang == "hi"
                       else "Division by zero is undefined, so there's no answer.")
    bad = pick(["32 March", "31/02/2027", "next to next somday", "45 June 2027"])
    user, _ = fill(_p(lang, ["{d} ko kaun sa din hai?"], ["what day is {d}?"]), lang, d=bad)
    ex = Ex("date_error", user)
    r = ex.tool("DATE", bad)
    if not r.startswith("Error"):
        raise Skip("date parsed")
    return ex.done("Ye date samajh nahi aayi. Date ko 15 August 2027 jaise format me bhejiye." if lang == "hi"
                   else "I couldn't read that date. Could you send it like 15 August 2027?", allow={"15", "2027"})


# ═══════════════════════════════════════════════════════════════
# NO-TOOL FAMILIES
# ═══════════════════════════════════════════════════════════════
def nt(family, user, answer, allow=()):
    user = finish(noisy(user.replace('{', '(').replace('}', ')'), 'hi' if is_hi(user) else 'en'))
    ex = Ex(family, user)
    return ex.done(answer, allow=allow)


def fam_greeting():
    lang = L()
    q = pick({"hi": ["namaste", "namaste ji", "hello bhaiya", "hi", "hello", "ram ram", "salaam", "good morning ji",
                     "suniye", "koi hai?", "hello, shop khuli hai?", "hi milo", "bhaiya hello"],
              "en": ["hi", "hello", "hey there", "good morning", "good evening", "hi milo", "anyone there?", "hello?"]}[lang])
    if "khuli" in q:
        raise Skip("needs MQ")
    a = pick({"hi": ["Namaste! Batayiye, kya madad kar sakta hoon?", "Ji, boliye! Kya chahiye aapko?",
                     "Hello! Main Milo hoon, shop ka assistant. Kya help karun?", "Namaste ji 🙏 Kya dekhna hai aapko?"],
              "en": ["Hi! How can I help you today?", "Hello! What can I do for you?",
                     "Hi there — I'm Milo, the shop's assistant. What do you need?", "Hello! Looking for something?"]}[lang])
    q = bank(q, B.GREETING, lang)
    return nt("greeting", finish(q), a)


def fam_thanks_bye():
    lang = L()
    q = pick({"hi": ["thank you", "dhanyavaad", "shukriya", "ok thanks", "theek hai", "acha", "ok ji", "bye", "chalo bye",
                     "thik hai bhaiya", "👍", "done", "badhiya"],
              "en": ["thanks!", "thank you", "ok", "great, thanks", "got it", "bye", "cool", "perfect", "👍", "awesome"]}[lang])
    a = pick({"hi": ["Aapka swagat hai! Aur kuch chahiye to message kijiye.", "Dhanyavaad! 🙏", "Ji, koi baat nahi. Aur kuch?",
                     "Theek hai ji. Kuch aur chahiye to batayein."],
              "en": ["You're welcome! Message anytime.", "Happy to help!", "Anytime — let me know if you need anything else.",
                     "Great! Have a nice day."]}[lang])
    q = bank(q, B.THANKS, lang)
    return nt("thanks", finish(q), a)


def fam_about():
    lang = L()
    q = pick({"hi": ["tum kaun ho?", "aap insaan ho ya bot?", "kya kya kar sakte ho?", "milo kaun hai?", "aap AI ho?"],
              "en": ["who are you?", "are you a bot?", "what can you do?", "are you human?", "who made you?"]}[lang])
    a = pick({"hi": ["Main Milo hoon, is shop ka AI assistant. Product ka rate, stock, timing, delivery aur hisaab-kitaab me madad kar sakta hoon.",
                     "Main ek AI assistant (Milo) hoon. Price, availability, delivery, GST calculation — sab me help karta hoon. Order confirm shop owner karte hain."],
              "en": ["I'm Milo, this shop's AI assistant. I can check prices, stock, timings and delivery, and do quick calculations.",
                     "I'm an AI assistant called Milo. I help with product prices, availability, store info and calculations; the owner confirms orders."]}[lang])
    q = bank(q, B.ABOUT, lang)
    return nt("about", finish(q), a)


def fam_clarify():
    lang = L(); k = R.random()
    if k < 0.4:
        q = pick({"hi": ["rate batao", "kitne ka hai?", "price kya hai", "ye kitne ka milega", "iska rate?", "daam?"],
                  "en": ["how much?", "price?", "what's the rate?", "how much is it?", "cost?"]}[lang])
        q = bank(q, B.CLARIFY_PRICE, lang)
        a = pick({"hi": ["Kaunse product ka rate chahiye? Naam bata dijiye, main check kar deta hoon.",
                         "Kis item ka price chahiye? Product ka naam aur size bhejiye."],
                  "en": ["Which product do you mean? Send me the name and size and I'll check.",
                         "Sure — which item are you asking about?"]}[lang])
    elif k < 0.7:
        q = pick({"hi": ["order karna hai", "kuch saman mangwana hai", "mujhe order dena hai", "list bhejun?"],
                  "en": ["I want to place an order", "can I order?", "I'd like to buy some things"]}[lang])
        q = bank(q, B.CLARIFY_ORDER, lang)
        a = pick({"hi": ["Ji zaroor! Items aur quantity ki list bhejiye, saath me naam aur address. Shop owner confirm karke bata denge.",
                         "Bilkul. Kya-kya chahiye, kitna chahiye aur kahan bhejna hai — ye bata dijiye."],
                  "en": ["Sure! Send the items with quantities, plus your name and address. The owner will confirm your order.",
                         "Of course — what items and how many? Please also share your delivery address."]}[lang])
    else:
        q = pick({"hi": ["GST kitna lagega?", "isme GST kitna hoga", "tax kitna lagega"], "en": ["how much GST will apply?", "what's the GST?"]}[lang])
        a = pick({"hi": ["Kis amount aur kis rate pe? Zyadatar cheezon pe 5% ya 18% GST lagta hai — amount aur rate bata dijiye, main calculate kar dunga.",
                         "Amount kitna hai aur GST rate kya hai? Batayiye, main turant calculate kar deta hoon."],
                  "en": ["On what amount and at what rate? Most goods are taxed at 5% or 18% — give me the amount and rate and I'll calculate it.",
                         "Tell me the amount and the GST rate and I'll work it out."]}[lang])
        return nt("clarify", finish(q), a, allow={"18"})
    return nt("clarify", finish(q), a)


def fam_order_status():
    lang = L(); oid = R.randint(1000, 99999)
    with_id = R.random() < 0.4
    q = pick({"hi": ["mera order kahan hai?", "order kab aayega?", "order abhi tak nahi aaya", "mera parcel kidhar hai"],
              "en": ["where is my order?", "my order hasn't arrived", "order status?", "when will my order come?"]}[lang])
    q = bank(q, B.ORDER_STATUS, lang)
    if with_id:
        q += pick([f" order no {oid}", f" (order #{oid})", f", order id {oid}"])
    if with_id:
        a = pick({"hi": [f"Order #{oid} ka status main yahan se check nahi kar sakta. Shop owner ko bata diya jayega, wo jaldi update denge.",
                         f"Order {oid} note kar liya. Live status mere paas nahi hota — owner aapko jaldi confirm karenge."],
                  "en": [f"I can't see live status for order #{oid} from here. The owner will check and update you shortly.",
                         f"Noted order {oid}. I don't have live tracking, but the owner will get back to you soon."]}[lang])
    else:
        a = pick({"hi": ["Apna order number ya naam bhej dijiye. Live status mere paas nahi hota, owner check karke batayenge.",
                         "Sorry for the delay! Order ID bhejiye, shop owner turant check karenge."],
                  "en": ["Please share your order number or name. I don't have live tracking, but the owner will check and update you.",
                         "Sorry about the wait! Send your order ID and the owner will look into it."]}[lang])
    return nt("order_status", finish(q), a)


def fam_place_order():
    lang = L(); qty = R.randint(1, 5)
    item = pick(["atta 10kg", "2 dozen kele", "black forest cake 1kg", "charger", "paneer 1kg", "notebook", "kurti M size"])
    q = pick({"hi": [f"{qty} {item} bhej do", f"{item} {qty} pack kar do", f"{qty} {item} order kar do please"],
              "en": [f"please send {qty} {item}", f"I'll take {qty} {item}", f"book {qty} {item} for me"]}[lang])
    a = pick({"hi": ["Ji, note kar liya! Apna naam aur delivery address bhej dijiye, shop owner confirm karke bata denge.",
                     "Theek hai! Address aur payment mode (UPI/cash) bata dijiye, order confirm ho jayega."],
              "en": ["Got it! Please send your name and delivery address — the owner will confirm the order.",
                     "Noted! Share your address and payment mode (UPI or cash) and we'll confirm."]}[lang])
    return nt("place_order", finish(q), a)


def fam_complaint():
    lang = L()
    q = pick({"hi": ["saman kharab nikla", "packet phata hua aaya", "galat item aa gaya", "cake toot gaya tha",
                     "quality achhi nahi thi", "charger kaam nahi kar raha"],
              "en": ["the product was damaged", "I got the wrong item", "the packet was torn", "it stopped working"]}[lang])
    a = pick({"hi": ["Bahut sorry iske liye! Item ki photo aur order details bhej dijiye, owner turant replacement dekh lenge.",
                     "Maaf kijiye 🙏 Photo bhejiye aur bataiye kab order kiya tha — hum jaldi solve karenge."],
              "en": ["Really sorry about that! Please send a photo and your order details — the owner will arrange a fix.",
                     "Apologies! Share a photo and when you ordered, and we'll sort it out quickly."]}[lang])
    q = bank(q, B.COMPLAINT, lang)
    return nt("complaint", finish(q), a)


def fam_human():
    lang = L()
    q = pick({"hi": ["owner se baat karni hai", "kisi insaan se baat karao", "call kar sakte ho?", "manager se baat karwa do"],
              "en": ["can I talk to a human?", "I want to speak to the owner", "can someone call me?"]}[lang])
    a = pick({"hi": ["Ji, maine owner ko bata diya hai. Wo jaldi aapse contact karenge.",
                     "Bilkul, shop owner ko message forward kar diya hai — wo jaldi reply karenge."],
              "en": ["Sure — I've flagged this for the owner, who'll get in touch soon.",
                     "Of course. I've passed your message to the owner; they'll reply shortly."]}[lang])
    q = bank(q, B.HUMAN, lang)
    return nt("human_handoff", finish(q), a)


def fam_payment_done():
    lang = L(); amt = pick([249, 499, 649, 1180, 1999, 2450])
    q = pick({"hi": [f"₹{amt} UPI kar diya", f"payment ho gaya ₹{amt}", f"{amt} bhej diye gpay se"],
              "en": [f"paid ₹{amt} via UPI", f"sent ₹{amt} on gpay", f"payment of ₹{amt} done"]}[lang])
    if R.random() < 0.6:
        q = pick(leak_filter(B.PAYMENT_DONE[lang])).format(amt=amt)
    a = pick({"hi": [f"Dhanyavaad! ₹{amt} ka payment owner confirm karke aapko update kar denge.",
                     f"Thank you ji! ₹{amt} received hone pe confirmation bhej denge."],
              "en": [f"Thank you! The owner will confirm the ₹{amt} payment and update you.",
                     f"Thanks! We'll confirm once ₹{amt} reflects in our account."]}[lang])
    return nt("payment_ack", finish(q), a)


DECLINE_ANSWERS = {
    "medical": {"hi": "Main dawai ki salah nahi de sakta. Kripya doctor ya pharmacist se baat kijiye.",
                "en": "I can't give medical advice — please consult a doctor or pharmacist."},
    "loan": {"hi": "Sorry, hum loan ya udhaar nahi dete. Is shop ke products ya services me madad kar sakta hoon.",
             "en": "Sorry, we don't offer loans or credit. I can help with this shop's products and services."},
    "predict": {"hi": "Ye main predict nahi kar sakta 😄 Shop se related kuch chahiye to batayein.",
                "en": "I can't predict that 😄 Anything from the shop I can help with?"},
    "personal": {"hi": "Sorry, main kisi ki personal details share nahi kar sakta.",
                 "en": "Sorry, I can't share anyone's personal details."},
    "homework": {"hi": "Main pura kaam nahi likhunga, par topic samjhane me madad kar sakta hoon. Kaunsa topic hai?",
                 "en": "I won't write the whole thing, but I can help you understand the topic. What's it about?"},
}
DECLINE_BASE = {
    "medical": {"hi": ["mujhe bukhar hai, kaunsi dawai lun?", "khansi ke liye kaunsi tablet lu?", "BP ki kaunsi medicine achhi hai?"],
                "en": ["I have fever, which medicine should I take?"]},
    "loan": {"hi": ["mujhe loan chahiye"], "en": ["can you give me a loan?"]},
    "predict": {"hi": ["kal match kaun jeetega?"], "en": ["who will win tomorrow's match?"]},
    "personal": {"hi": ["kisi ka number do"], "en": ["give me someone's phone number"]},
    "homework": {"hi": ["mera homework kar do"], "en": ["write my college assignment"]},
}


def fam_decline():
    lang = L(); cat = pick(list(DECLINE_ANSWERS))
    pool = leak_filter(DECLINE_BASE[cat][lang] + B.DECLINE[cat][lang])
    return nt("decline", finish(pick(pool)), DECLINE_ANSWERS[cat][lang])


CONCEPTS = [
    (["What is a primary key?", "primary key kya hota hai?"],
     "A primary key is a column (or set of columns) that uniquely identifies each row in a table; it can't be null or repeated."),
    (["What is an environment variable?", "env variable kya hota hai?"],
     "An environment variable is a named value set outside your code (like API keys or config) that your program reads at runtime."),
    (["What's the difference between a list and a tuple in Python?", "python list aur tuple me farak?"],
     "Lists are mutable — you can change them; tuples are immutable. Use tuples for fixed data and lists for collections that change."),
    (["What does async/await do?", "async await kya karta hai?"],
     "async/await lets code wait for slow work like network calls without blocking everything else, while still reading top to bottom."),
    (["What is a pull request?", "PR kya hota hai github pe?"],
     "A pull request asks to merge your branch into another; teammates review the changes and comment before it's merged."),
    (["What is caching?", "caching kya hai?"],
     "Caching stores the result of an expensive operation so the next request can be served quickly from memory."),
    (["What is an API key?", "API key kya hoti hai?"],
     "An API key is a secret token that identifies your app to an API. Keep it out of your code repo and rotate it if it leaks."),
    (["How do I prepare for my first developer interview?", "pehle dev interview ki tayari kaise karu?"],
     "Revise DSA basics, be ready to explain your own projects in depth, practise talking through your approach out loud, and ask good questions at the end."),
    (["Should I use SQL or NoSQL?", "SQL lu ya NoSQL?"],
     "Start with SQL for structured data and relationships; choose NoSQL when data is document-shaped or you need very flexible schemas at scale."),
    (["What is a webhook?", "webhook kya hota hai?"],
     "A webhook is a URL your app exposes so another service can call it automatically when an event happens, like a payment succeeding."),
    (["How do I keep customers coming back?", "customer wapas kaise aaye?"],
     "Be reliable on quality and delivery, remember regulars, reply quickly on WhatsApp, and offer a small reward for repeat orders."),
    (["How do I handle a customer who is angry?", "gussa customer ko kaise handle karu?"],
     "Listen first, apologise for the experience, fix what you can quickly, and follow up so they know it's resolved."),
    (["Should I accept UPI payments?", "UPI lena chahiye kya?"],
     "Yes — most customers prefer UPI, payments settle directly to your bank, and a static QR code is free to set up."),
    (["How do I manage stock better?", "stock better kaise manage karu?"],
     "Track what sells fastest, reorder before items run out, keep a simple daily stock sheet, and clear slow-moving items with offers."),
    (["Is it worth getting a GST registration?", "GST registration lena chahiye?"],
     "It depends on your turnover and who you sell to — registration lets you claim input tax credit and sell to businesses. A CA can confirm if it's required for you."),
    (["What does API stand for?", "API ka full form?", "API matlab kya?"],
     "API stands for Application Programming Interface — a defined way for one piece of software to talk to another."),
    (["Difference between == and === in JavaScript?", "JS me == aur === me kya farak hai?"],
     "== compares after type conversion ('5' == 5 is true); === also checks the type ('5' === 5 is false). Prefer ===."),
    (["What is recursion?", "recursion kya hota hai?"],
     "Recursion is when a function calls itself. It needs a base case to stop and a recursive case that makes the problem smaller."),
    (["GET vs POST?", "GET aur POST me difference?"],
     "GET fetches data and puts parameters in the URL; POST sends data in the request body and is used to create or change things."),
    (["What is JSON?", "JSON kya hai?"],
     "JSON (JavaScript Object Notation) is a lightweight text format for data using key-value pairs and arrays — the standard for APIs."),
    (["HTTP vs HTTPS?", "http aur https me kya farak hai?"],
     "HTTPS is HTTP over TLS: the connection is encrypted, so data can't be read or tampered with in transit."),
    (["What is OOP?", "OOP kya hota hai?"],
     "Object-Oriented Programming organises code into objects that bundle data and behaviour; classes are their blueprints."),
    (["What's a database index?", "index kya hota hai database me?"],
     "An index is a lookup structure that speeds up reads on a column, at the cost of extra storage and slower writes."),
    (["Should I learn Python or JavaScript first?", "pehle python seekhu ya javascript?"],
     "Pick by goal: Python for data, AI and automation; JavaScript for websites and web apps. Both are good first languages."),
    (["How do I get better at DSA?", "DSA kaise improve karu?"],
     "Practise daily, start with arrays and strings, learn patterns (two pointers, sliding window, BFS/DFS), and revisit problems you got wrong."),
    (["What's the difference between git merge and rebase?", "merge vs rebase?"],
     "Merge joins branches with a merge commit and keeps history as it happened; rebase replays your commits on top for a linear history."),
    (["What is a REST API in simple words?", "REST API simple me samjhao"],
     "A REST API lets apps request and change data over HTTP using URLs for resources and methods like GET, POST, PUT and DELETE."),
    (["Is SQL still worth learning?", "SQL seekhna chahiye?"],
     "Yes — almost every product stores data in a relational database, and SQL is how you query it."),
    (["What is a virtual environment in Python?", "python venv kya hai?"],
     "A virtual environment is an isolated folder with its own Python packages, so projects don't clash over versions."),
    (["How do I get my first customers?", "pehle customer kaise laaun?"],
     "Start with people you know and nearby shops, give a great first experience, and ask happy customers to refer others."),
    (["Should I offer discounts to new customers?", "naye customers ko discount du?"],
     "A small first-order offer can help, but keep it limited so your margin stays healthy and customers come back for quality."),
    (["How can I get more customers on WhatsApp?", "WhatsApp pe zyada customers kaise laun?"],
     "Share your catalogue as a WhatsApp Business catalog, post daily offers on Status, reply fast, and ask happy customers to share your number."),
    (["Is 18% GST high?", "18% GST zyada hai kya?"],
     "18% is India's standard rate for most goods and services; essentials usually sit at 5% or nil. Whether it feels high depends on your product's margins."),
    (["What is GST in simple words?", "GST simple me samjhao"],
     "GST is a single indirect tax added to the price of goods and services in India; businesses collect it from customers and pay it to the government."),
    (["How should I price my product?", "apne product ka price kaise rakhu?"],
     "Start from your full cost, add the margin you need, then check competitor prices and what customers are willing to pay."),
]


def fam_concept():
    qs, a = pick(CONCEPTS)
    q = pick(qs)
    return nt("concept", q, a, allow=nums(a))


# ═══════════════════════════════════════════════════════════════
# PARAPHRASE BANKS (paraphrase_banks.py) — merged into the pools above
# ═══════════════════════════════════════════════════════════════
for _lang in ("hi", "en"):
    PRODUCT_T[_lang] += leak_filter(B.PRODUCT[_lang])
    QTY_T[_lang] += leak_filter(B.QTY[_lang])
    DELIV_T[_lang] += leak_filter(B.DELIVERY_DATE[_lang])
    READY_T[_lang] += leak_filter(B.READY_DATE[_lang])
    for _topic, _by_lang in B.FAQ.items():
        FAQ_T[_topic][_lang] += leak_filter(_by_lang.get(_lang, []))
    for _kind, _by_lang in B.WEATHER.items():
        WEATHER_T[_kind][_lang] += leak_filter(_by_lang[_lang])
    BULK_T[_lang] += leak_filter(B.BULK[_lang])
    DELIV_T[_lang] += leak_filter(B.DELIVERY_AFTER_ORDER[_lang])
    READY_T[_lang] += leak_filter(B.READY_EXTRA[_lang])
    SEARCH_T[_lang] += leak_filter(B.SEARCH[_lang])
CONCEPTS += [(leak_filter(qs), a) for qs, a in B.CONCEPTS if leak_filter(qs)]


def bank(q, pool, lang, p=0.6):
    """With probability p, swap the built-in message for one from a paraphrase bank."""
    options = leak_filter(pool[lang])
    return pick(options) if options and R.random() < p else q


def fam_advice():
    qs, a = pick(B.ADVICE)
    qs = leak_filter(qs)
    if not qs:
        raise Skip("advice leak")
    return nt("advice", pick(qs), a)


def fam_clarify_other():
    lang = L()
    return nt("clarify", pick(leak_filter(B.CLARIFY_OTHER[lang])), B.CLARIFY_OTHER_ANSWERS[lang])


# ═══════════════════════════════════════════════════════════════
# MIX
# ═══════════════════════════════════════════════════════════════
FAMILIES = {
    # tool (≈70%)
    "mq_product": (fam_mq_product, 2200), "mq_faq": (fam_mq_faq, 1300), "mq_nomatch": (fam_mq_nomatch, 450),
    "mq_calc_qty": (fam_mq_calc_qty, 750), "mq_calc_two": (fam_mq_calc_two, 300), "mq_calc_offer": (fam_mq_calc_offer, 250),
    "mq_bulk": (fam_mq_bulk, 260), "mq_date": (fam_mq_date, 450), "route_contrast": (fam_route_contrast, 500),
    "calc": (fam_calc, 1900), "date": (fam_date, 1100), "search": (fam_search, 650), "search_custom": (fam_search_custom, 200), "search_error": (fam_search_error, 60),
    "weather": (fam_weather, 1100), "weather_error": (fam_weather_error, 40), "code": (fam_code, 500),
    "code_disabled": (fam_code_disabled, 40), "tool_error": (fam_tool_error, 120),
    # no tool (≈30%)
    "greeting": (fam_greeting, 450), "thanks": (fam_thanks_bye, 400), "about": (fam_about, 150),
    "clarify": (fam_clarify, 700), "order_status": (fam_order_status, 450), "place_order": (fam_place_order, 450),
    "complaint": (fam_complaint, 350), "human": (fam_human, 200), "payment_ack": (fam_payment_done, 250),
    "decline": (fam_decline, 300), "concept": (fam_concept, 900),
    "advice": (fam_advice, 350), "clarify_other": (fam_clarify_other, 120),
}
MAX_DUP = 3   # max copies of an identical text


def generate(seed, scale, quiet=False):
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
            out.append(ex); counts[name] += 1
        if counts[name] < target and not quiet:
            print(f"  ! {name}: only {counts[name]}/{target}")
    T.set_now(None); T.set_merchant_kb(None)
    R.shuffle(out)
    return out, counts, skips


def token_len_fn():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda t: len(enc.encode(t, allowed_special="all")), "tiktoken"
    except Exception:
        return lambda t: int(len(t.encode("utf-8")) / 3.2), "approx (no tiktoken gpt2 files)"


_CALL_RE = re.compile(r"<\|tool_call\|>([A-Z_]+)\((.*?)\)<\|tool_result\|>(.*?)<\|end_tool\|>", re.S)


def validate(examples, code_sample=60):
    """Independent re-check of the final data (does not trust the generator)."""
    problems = collections.Counter()
    code_checked = 0
    for e in examples:
        t = e["text"]
        if not (t.startswith("<|user|>\n") and "\n<|assistant|>\n" in t and t.endswith("<|endoftext|>")):
            problems["bad structure"] += 1; continue
        user, body = t[len("<|user|>\n"):].split("\n<|assistant|>\n", 1)
        calls = _CALL_RE.findall(body)
        if len(calls) != body.count("<|tool_call|>"):
            problems["unparseable tool call"] += 1; continue
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
            continue   # no-tool examples were checked against their allow-lists at build time
        if grounded(answer, [user] + results, e.get("allow", ())):
            problems["ungrounded number"] += 1
    return problems


def main():
    ap = argparse.ArgumentParser(description="Milo v4 tool-use data")
    ap.add_argument("--output", default="data/tool_use_v4_train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all family sizes")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--show", type=int, default=6, help="print N random samples")
    args = ap.parse_args()

    print("=" * 64)
    print("  Milo — tool-use data v4")
    print("=" * 64)
    train, counts, skips = generate(args.seed, args.scale)
    val, _, _ = generate(args.seed + 100003, args.scale * args.val_frac, quiet=True)
    train_texts = {e["text"] for e in train}
    val = [e for e in val if e["text"] not in train_texts]

    tok, how = token_len_fn()
    for name, data in (("train", train), ("val", val)):
        before = len(data)
        data[:] = [e for e in data if tok(e["text"]) <= args.max_tokens]
        if before != len(data):
            print(f"  dropped {before - len(data)} {name} examples over {args.max_tokens} tokens")

    problems = validate(train) + validate(val)
    if problems:
        print("  VALIDATION PROBLEMS: " + ", ".join(f"{k} ×{v}" for k, v in problems.items()))
    else:
        print("  validation: all examples passed (structure, parser round-trip, CALCULATE + RUN_CODE re-run, grounding)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    val_path = args.output.replace("train", "val") if "train" in args.output else args.output + ".val"
    for path, data in ((args.output, train), (val_path, val)):
        with open(path, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n = len(train)
    tool_n = sum(1 for e in train if e["tools"])
    multi = sum(1 for e in train if len(e["tools"]) > 1)
    by_tool = collections.Counter(t for e in train for t in e["tools"])
    lens = [tok(e["text"]) for e in train]
    hi_share = sum(1 for e in train if re.search(r"\b(hai|kya|kitne|kitna|batao|chahiye)\b", e["text"].split("<|assistant|>")[0])) / n
    print(f"\n  train {n}  |  val {len(val)}  (val de-duplicated against train)")
    print(f"  tool {tool_n} ({tool_n / n:.0%})  no-tool {n - tool_n} ({(n - tool_n) / n:.0%})  multi-tool {multi} ({multi / n:.0%})")
    print(f"  unique texts {len(set(e['text'] for e in train)) / n:.1%}   Hinglish user msgs ≈{hi_share:.0%}")
    print(f"  tokens ({how}): avg {sum(lens) / n:.0f}, max {max(lens)}")
    print("  tool calls: " + ", ".join(f"{k} {v}" for k, v in by_tool.most_common()))
    print("  families: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    top_skips = skips.most_common(6)
    if top_skips:
        print("  most common rejections: " + "; ".join(f"{k} ×{v}" for k, v in top_skips))
    print(f"  leak filter dropped {len(set(LEAK_DROPPED))} bank templates that overlapped the test split")
    print(f"\n  → {args.output}\n  → {val_path}")

    if args.show:
        print("\n" + "=" * 64 + "\n  Samples\n" + "=" * 64)
        for e in random.Random(1).sample(train, min(args.show, n)):
            print(f"[{e['family']}]\n{e['text']}\n" + "─" * 50)


if __name__ == "__main__":
    main()