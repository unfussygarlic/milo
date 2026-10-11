"""MQ_SEARCH — merchant catalog / FAQ / store-info retrieval for ONE merchant.

``MQ_MAX_HITS``, ``MQ_KEEP_RATIO``, the synonym and stopword tables and the
``format_*`` helpers are part of the training-data contract: they decide which
lines come back and how they read, and the model was trained to copy them
verbatim.
"""

import json
import os
import re

# ═══════════════════════════════════════════════════════════════
# MQ_SEARCH — merchant query: catalog, FAQ and store info for ONE merchant
#
# Knowledge base (JSON, one per merchant deployment):
#   {"store":   {"name": ..., "type": ..., "city": ...},
#    "catalog": [{"name", "variant", "price", "stock", "keywords", "category",
#                 "price_from" (bool, services), "offer" (optional str)}],
#    "faq":     [{"topic", "answer", "keywords"}]}
# Loaded from MILO_MERCHANT_KB (path) at runtime; the generator calls
# set_merchant_kb(dict) per example.
#
# Result: up to 3 lines, best first
#   Basmati Rice 5kg | ₹649 incl. GST | in stock
#   Screen replacement | from ₹1499 incl. GST | in stock
#   Delivery: Home delivery within Indore in 2 days. ₹40 charge, free above ₹500.
#   No match in store for: <query>
# ═══════════════════════════════════════════════════════════════
MQ_MAX_HITS = 3
MQ_KEEP_RATIO = 0.75     # keep hits scoring >= 75% of the best hit

# Hinglish / regional words → canonical English catalog words.
MQ_SYNONYMS = {
    "chawal": "rice", "chaval": "rice", "aata": "atta", "gehu": "wheat", "gehun": "wheat",
    "tel": "oil", "cheeni": "sugar", "chini": "sugar", "shakkar": "sugar", "namak": "salt",
    "doodh": "milk", "dudh": "milk", "dahi": "curd", "makhan": "butter", "anda": "egg",
    "ande": "egg", "chai": "tea", "patti": "tea", "sabun": "soap", "sabji": "vegetable",
    "sabzi": "vegetable", "aloo": "potato", "alu": "potato", "pyaz": "onion", "pyaaz": "onion",
    "kanda": "onion", "tamatar": "tomato", "adrak": "ginger", "lehsun": "garlic", "lahsun": "garlic",
    "kela": "banana", "seb": "apple", "aam": "mango", "santra": "orange", "angoor": "grapes",
    "mithai": "sweets", "kapda": "cloth", "kapde": "clothes", "joota": "shoes", "joote": "shoes",
    "chappal": "slipper", "pankha": "fan", "batti": "bulb", "taar": "wire", "rang": "paint",
    "kitab": "book", "kitaab": "book", "copy": "notebook", "kalam": "pen", "chashma": "glasses",
    "baal": "hair", "dawai": "medicine", "kutta": "dog", "billi": "cat", "khana": "food",
    "silai": "stitching", "silna": "stitching", "silne": "stitching", "silwana": "stitching", "silwani": "stitching", "darzi": "tailor", "cover": "cover", "charger": "charger",
    "earphone": "earphones", "headphone": "headphones", "tiffin": "tiffin", "cake": "cake",
    "timing": "hours", "time": "hours", "khulta": "hours", "khulti": "hours", "khulega": "hours",
    "band": "hours", "baje": "hours", "khule": "hours", "khula": "hours", "khulte": "hours",
    "jagah": "address", "pahunchu": "address",
    "chhoot": "discount", "chhut": "discount", "chut": "discount", "tayyar": "ready", "taiyar": "ready", "tayar": "ready", "pata": "address", "kaha": "address", "kahan": "address",
    "location": "address", "wapas": "return", "wapis": "return", "badalna": "exchange",
    "badal": "exchange", "refund": "return", "bhejo": "delivery",
    "bhej": "delivery", "ghar": "delivery", "aayega": "delivery", "ayega": "delivery",
    "pahuchega": "delivery", "shipping": "delivery", "courier": "delivery",
    "gpay": "upi", "phonepe": "upi", "paytm": "upi", "online": "upi",
    "kurta": "kurti", "sari": "saree", "rasgulla": "rasgulla", "rosogolla": "rasgulla",
}

# Words that carry no product meaning in customer messages.
MQ_STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "you", "your", "i", "me", "my", "we",
    "of", "for", "in", "on", "at", "to", "and", "or", "with", "have", "has", "can",
    "what", "how", "much", "many", "price", "rate", "cost", "kitna", "kitne", "kitni",
    "ka", "ki", "ke", "ko", "se", "mein", "hai", "hain", "h", "ho", "kya", "kyaa",
    "bhai", "bhaiya", "sir", "madam", "ji", "please", "pls", "plz", "available", "milega",
    "milegi", "milenge", "chahiye", "chaiye", "dena", "dedo", "batao", "bata",
    "bataiye", "aap", "aapka", "aapke", "apka", "yeh", "ye", "woh", "wo", "hello", "hi",
    "daam", "bhav", "bhaav", "mrp", "stock", "store", "shop", "dukaan", "dukan", "any",
    "kab", "tak", "abhi", "today", "aaj", "kal", "it", "this", "that", "there", "per",
    "rs", "inr", "rupees", "rupaye", "wala", "wali", "wale", "ek", "one", "need", "want",
}

_UNIT_RULES = [
    (r"\b(?:aadha|adha|half)\s*(?:kilo|kg)\b", "500g"),
    (r"\b(?:paav|pao|pav)\s*(?:kilo|kg)?\b", "250g"),
    (r"(\d+(?:\.\d+)?)\s*(?:kgs?|kilos?|kilograms?)\b", r"\1kg"),
    (r"(\d+)\s*(?:g|gm|gms|grams?)\b", r"\1g"),
    (r"(\d+)\s*ml\b", r"\1ml"),
    (r"(\d+(?:\.\d+)?)\s*(?:l|ltrs?|litres?|liters?)\b", r"\1l"),
    (r"(\d+)\s*(?:pcs?|pieces?|nos?)\b", r"\1pc"),
    (r"(\d+)\s*(?:inch|inches|in)\b", r"\1in"),
    (r"(\d+)\s*(?:ft|feet|foot)\b", r"\1ft"),
    (r"(\d+)\s*(mah|gb|tb|w|ah|mm)\b", r"\1\2"),
]
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?[a-z]*|[a-z]+")

_merchant_kb = None
_merchant_kb_path_loaded = None


def set_merchant_kb(kb: dict | None) -> None:
    """Generator / tests: use this KB dict (None → fall back to MILO_MERCHANT_KB)."""
    global _merchant_kb, _merchant_kb_path_loaded
    _merchant_kb, _merchant_kb_path_loaded = kb, None


def _get_merchant_kb():
    global _merchant_kb, _merchant_kb_path_loaded
    if _merchant_kb is not None and _merchant_kb_path_loaded is None:
        return _merchant_kb
    path = os.environ.get("MILO_MERCHANT_KB")
    if not path or not os.path.exists(path):
        return None
    if _merchant_kb_path_loaded != path:
        with open(path, encoding="utf-8") as f:
            _merchant_kb = json.load(f)
        _merchant_kb_path_loaded = path
    return _merchant_kb


def mq_tokens(text: str) -> list[str]:
    s = text.lower()
    for pat, rep in _UNIT_RULES:
        s = re.sub(pat, rep, s)
    out = []
    for t in _TOKEN_RE.findall(s):
        t = MQ_SYNONYMS.get(t, t)
        if t not in MQ_STOPWORDS:
            out.append(t)
    return out


def _fuzzy_hit(tok, vocab):
    """Typos (basmti→basmati) and word-form variants (replace→replacement)."""
    if len(tok) < 4 or tok[0].isdigit():
        return False
    for v in vocab:
        if len(v) >= 4 and (v.startswith(tok) or tok.startswith(v)):
            return True
    import difflib
    return bool(difflib.get_close_matches(tok, [v for v in vocab if len(v) >= 4], n=1, cutoff=0.8))


def format_price(p: float) -> str:
    return f"{int(p)}" if float(p) == int(p) else f"{p:.2f}"


def format_catalog_line(item: dict) -> str:
    """Canonical catalog result line — the generator must call this too."""
    name = item["name"] + (f" {item['variant']}" if item.get("variant") else "")
    if not item["price"]:
        price = "free"
    elif item.get("price_from"):
        price = f"from ₹{format_price(item['price'])} incl. GST"
    else:
        price = f"₹{format_price(item['price'])} incl. GST"
    line = f"{name} | {price} | {item.get('stock', 'in stock')}"
    if item.get("offer"):
        line += f" | offer: {item['offer']}"
    return line


def format_faq_line(entry: dict) -> str:
    return f"{entry['topic']}: {entry['answer']}"


def _score(query_toks, primary, secondary):
    score, primary_hits = 0.0, 0
    for q in query_toks:
        if q in primary:
            score += 3; primary_hits += 1
        elif q in secondary:
            score += 2
        elif _fuzzy_hit(q, primary):
            score += 2; primary_hits += 1
    return score if primary_hits else 0.0


def mq_search(query: str, kb: dict | None = None) -> str:
    query = query.strip().strip("'\"")
    if not query:
        return "Error: empty query"
    kb = kb if kb is not None else _get_merchant_kb()
    if not kb:
        return "Error: store catalog not available"
    q = mq_tokens(query)
    if not q:
        return f"No match in store for: {query}"

    scored = []
    for item in kb.get("catalog", []):
        primary = set(mq_tokens(item["name"] + " " + " ".join(item.get("keywords", []))))
        secondary = set(mq_tokens(f"{item.get('variant', '')} {item.get('category', '')}"))
        s = _score(q, primary, secondary)
        if s:
            scored.append((s, format_catalog_line(item)))
    for entry in kb.get("faq", []):
        primary = set(mq_tokens(entry["topic"] + " " + " ".join(entry.get("keywords", []))))
        secondary = set(mq_tokens(entry["answer"]))
        s = _score(q, primary, secondary)
        if s:
            scored.append((s, format_faq_line(entry)))

    if not scored:
        return f"No match in store for: {query}"
    scored.sort(key=lambda x: -x[0])          # stable: KB order breaks ties
    best = scored[0][0]
    lines = [line for s, line in scored if s >= best * MQ_KEEP_RATIO][:MQ_MAX_HITS]
    return "\n".join(lines)
