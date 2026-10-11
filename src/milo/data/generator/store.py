"""MQ_SEARCH families: product lookups, FAQ, no-match, and MQ -> CALCULATE / DATE chains.

The retrieval helpers here (``mq_lookup``, ``describe_*``) parse real MQ_SEARCH
output, so every answer is built from the same bytes the runtime will see.
"""

import collections
import re

import milo.tools as T
from milo.data import merchants as M
from milo.data import paraphrase_banks as B
from milo.data.generator.core import (
    Ex,
    L,
    R,
    Skip,
    fill,
    leak_filter,
    new_kb,
    new_now,
    pick,
)

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


# ── paraphrase banks (milo.data.paraphrase_banks) merged into the pools above ──
for _lang in ("hi", "en"):
    PRODUCT_T[_lang] += leak_filter(B.PRODUCT[_lang])
    QTY_T[_lang] += leak_filter(B.QTY[_lang])
    DELIV_T[_lang] += leak_filter(B.DELIVERY_DATE[_lang])
    READY_T[_lang] += leak_filter(B.READY_DATE[_lang])
    for _topic, _by_lang in B.FAQ.items():
        FAQ_T[_topic][_lang] += leak_filter(_by_lang.get(_lang, []))
    BULK_T[_lang] += leak_filter(B.BULK[_lang])
    DELIV_T[_lang] += leak_filter(B.DELIVERY_AFTER_ORDER[_lang])
    READY_T[_lang] += leak_filter(B.READY_EXTRA[_lang])
