"""
Milo — behavioural eval for tool use

Runs ~150 hand-written, held-out prompts (phrased differently from the
generator's templates) through the real ReAct runtime and scores what
actually matters:

  tool     right first tool, or correctly NO tool
  chain    every tool the task needs was called (e.g. MQ_SEARCH → CALCULATE)
  args     tool args are valid (no tool errors) and MQ_SEARCH args are copied
           from the customer's words
  grounded every multi-digit number in the reply came from the user or a tool result
  uses     the reply actually uses the tool result (not just a canned sentence)
  clean    non-empty reply, no leaked special tags

Everything is deterministic: greedy decoding, fixed "today", fixed merchant
stores, and SEARCH / WEATHER stubbed (pass --live to hit the real APIs).

Usage
    python eval_tools.py --checkpoint experiments/tools_v4/checkpoints/best.pt
    python eval_tools.py --checkpoint experiments/tools_20261010_071926/checkpoints/best.pt \\
                         --checkpoint experiments/tools_v4/checkpoints/best.pt      # side by side
    python eval_tools.py --checkpoint best.pt --only store_chain --show-failures 10
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

import milo_tools as T
import merchants as M

EVAL_NOW = dt.datetime(2026, 10, 10, 10, 30)

# ═══════════════════════════════════════════════════════════════
# FIXED STORES
# ═══════════════════════════════════════════════════════════════
_KB_CACHE = {}


def store(kind):
    """Deterministic merchant KB with delivery + bulk terms present and everything in stock,
    plus a couple of deliberate out-of-stock / offer items for specific cases."""
    if kind in _KB_CACHE:
        return _KB_CACHE[kind]
    for seed in range(1000):
        kb = M.build_kb(kind, random.Random(seed), offer_rate=0)
        spec = M.MERCHANT_TYPES[kind]
        if spec["delivery"] and kb["store"]["delivery_days"] in (None, 0) and kind not in ("bakery", "tiffin", "dairy", "fruits_veg"):
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
    "upi": ("Unified Payments Interface", "Unified Payments Interface (UPI) is an instant real-time payment system developed by the National Payments Corporation of India."),
    "docker": ("Docker (software)", "Docker is a platform that packages applications and their dependencies into containers."),
    "fssai": ("Food Safety and Standards Authority of India", "The Food Safety and Standards Authority of India (FSSAI) regulates food safety; food businesses need an FSSAI licence or registration."),
    "mishti doi": ("Mishti doi", "Mishti doi is a fermented sweet yogurt popular in West Bengal, traditionally set in earthen pots."),
    "kubernetes": ("Kubernetes", "Kubernetes is an open-source system for automating deployment, scaling and management of containerised applications."),
    "inflation": ("Inflation", "Inflation is a general increase in the prices of goods and services over time."),
    "isro": ("Indian Space Research Organisation", "ISRO is India's national space agency, headquartered in Bengaluru."),
    "chikankari": ("Chikan embroidery", "Chikankari is a traditional hand embroidery style from Lucknow."),
    "jwt": ("JSON Web Token", "A JSON Web Token (JWT) is a compact token format for transmitting claims, often used for authentication."),
    "e-way bill": ("E-way bill", "An e-way bill is an electronic document required under GST for moving goods above a specified value."),
    "taj mahal": ("Taj Mahal", "The Taj Mahal is a white marble mausoleum in Agra, commissioned in 1632 by Shah Jahan."),
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
    return T.format_weather(c, 18 + h % 20, ["sunny", "light rain", "partly cloudy", "haze"][h % 4],
                            40 + h % 50, 5 + h % 25, [5, 80, 30, 10][h % 4])


# ═══════════════════════════════════════════════════════════════
# CASES  (written by hand; NOT generator templates)
#   expect: allowed FIRST tools ([] = no tool at all)
#   need:   tools that must all appear somewhere in the trace
#   kb:     merchant store to load
#   err_ok: a tool error is the correct outcome
#   allow:  extra numbers the reply may contain
# ═══════════════════════════════════════════════════════════════
def C(cat, q, expect, need=(), kb=None, err_ok=False, allow=()):
    return dict(cat=cat, q=q, expect=list(expect), need=list(need), kb=kb, err_ok=err_ok, allow=list(allow), split="dev")


MQ = ["MQ_SEARCH"]
CASES = [
    # ── store: product lookups ──
    C("store_product", "bhaiya 5 kilo wala basmati kitne ka diya?", MQ, kb="kirana"),
    C("store_product", "do u guys have sarson oil 1 litre", MQ, kb="kirana"),
    C("store_product", "toor dal ka aaj ka bhav?", MQ, kb="kirana"),
    C("store_product", "moong dal hai kya abhi", MQ, kb="kirana"),
    C("store_product", "Desi ghee 1l kitne ka", MQ, kb="kirana"),
    C("store_product", "maggi ka packet milega?", MQ, kb="kirana"),
    C("store_product", "10 kg atta rate", MQ, kb="kirana"),
    C("store_product", "chawal kaun kaun se hai aapke paas", MQ, kb="kirana"),
    C("store_product", "ek power bank chahiye 20000 mah wala, kitne ka?", MQ, kb="mobile"),
    C("store_product", "10000 mah power bank in stock?", MQ, kb="mobile"),
    C("store_product", "iphone wala cable hai?", MQ, kb="mobile"),
    C("store_product", "phone ki screen toot gayi, lagwane me kitna lagega", MQ, kb="mobile"),
    C("store_product", "earbuds bluetooth wale kitne ke?", MQ, kb="mobile"),
    C("store_product", "chikankari kurti kitne tak ki hai", MQ, kb="ethnic_wear"),
    C("store_product", "banarasi silk saree ka price?", MQ, kb="ethnic_wear"),
    C("store_product", "black forest 1 kg cake hai?", MQ, kb="bakery"),
    C("store_product", "eggless butterscotch cake rate", MQ, kb="bakery"),
    C("store_product", "paneer 1 kilo kitne ka", MQ, kb="dairy"),
    C("store_product", "bhains ka doodh milta hai?", MQ, kb="dairy"),
    C("store_product", "puppy food 3kg price", MQ, kb="pet_shop"),
    C("store_product", "ac service ka charge?", MQ, kb="repair"),
    C("store_product", "blouse silwana hai, kitna loge", MQ, kb="tailor"),
    C("store_product", "python course ki fees kitni hai", MQ, kb="coaching"),
    C("store_product", "demo class free hai?", MQ, kb="coaching"),
    C("store_product", "aam alphonso ka dozen kitne ka", MQ, kb="fruits_veg"),
    # ── store: FAQ ──
    C("store_faq", "sunday ko shop khula rehta hai?", MQ, kb="kirana"),
    C("store_faq", "aap log ghar tak saman pahuchate ho?", MQ, kb="kirana"),
    C("store_faq", "payment kaise karna hai, gpay chalega?", MQ, kb="kirana"),
    C("store_faq", "dukaan ka address bhejiye", MQ, kb="kirana"),
    C("store_faq", "bill me gst alag se lagega?", MQ, kb="kirana"),
    C("store_faq", "what time do you shut today", MQ, kb="mobile"),
    C("store_faq", "charger ki warranty kitni milegi", MQ, kb="mobile"),
    C("store_faq", "galat size aaya to badal doge?", MQ, kb="ethnic_wear"),
    C("store_faq", "fitting thodi tight hai, alteration karoge?", MQ, kb="ethnic_wear"),
    C("store_faq", "birthday ke liye photo wala cake bana doge?", MQ, kb="bakery"),
    C("store_faq", "is the shop open on sundays?", MQ, kb="hardware"),
    C("store_faq", "technician ghar bhej sakte ho?", MQ, kb="repair"),
    C("store_faq", "suit silne me kitne din lagenge", MQ, kb="tailor"),
    C("store_faq", "weekend batch hai kya?", MQ, kb="coaching"),
    C("store_faq", "do you take wedding orders?", MQ, kb="sweets"),
    C("store_faq", "roz subah doodh dalwana hai, hoga?", MQ, kb="dairy"),
    C("store_faq", "refund milta hai kya?", MQ, kb="cosmetics"),
    # ── store: not in catalog ──
    C("store_nomatch", "iphone 15 hai aapke paas?", MQ, kb="kirana"),
    C("store_nomatch", "petrol milega?", MQ, kb="bakery"),
    C("store_nomatch", "kaju katli 1kg?", MQ, kb="kirana"),
    C("store_nomatch", "do you sell laptops?", MQ, kb="stationery"),
    C("store_nomatch", "cement ka bag kitne ka", MQ, kb="salon"),
    C("store_nomatch", "dog food hai?", MQ, kb="sweets"),
    # ── store: chains ──
    C("store_chain", "3 packet maggi kitne ke padenge?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    C("store_chain", "5 kilo wala basmati 2 bag lena hai, total?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    C("store_chain", "4 type c cable ka total kitna hoga", MQ, ["MQ_SEARCH", "CALCULATE"], kb="mobile"),
    C("store_chain", "2 sunflower oil 1l aur 3 sugar 1kg ka total batao", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    C("store_chain", "desi ghee pe offer ke baad kitne ka padega?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    C("store_chain", "6 veg puff kitne ke?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="bakery"),
    C("store_chain", "aaj order kiya to delivery kis date tak hogi?", MQ, ["MQ_SEARCH", "DATE"], kb="kirana"),
    C("store_chain", "if I order a kurti today when will it reach me?", MQ, ["MQ_SEARCH", "DATE"], kb="ethnic_wear"),
    C("store_chain", "blouse aaj diya to kis din milega?", MQ, ["MQ_SEARCH", "DATE"], kb="tailor"),
    C("store_chain", "₹8000 ka saman lena hai, bulk discount kitna milega?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    C("store_chain", "10 led bulb 9w ka total", MQ, ["MQ_SEARCH", "CALCULATE"], kb="hardware"),
    C("store_chain", "2 kg paneer chahiye, 1kg wala 2 pack, total?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="dairy"),
    # ── general calculation ──
    C("calc", "2350 ka 18 percent gst jod ke total kitna", ["CALCULATE"]),
    C("calc", "₹1,180 me 18% GST included hai, asli price kitna tha?", ["CALCULATE"]),
    C("calc", "maal 340 ka liya 425 me becha, kitne % margin hua", ["CALCULATE"]),
    C("calc", "teen staff hai, har ek ko 12500 salary, total kitna dena hai", ["CALCULATE"]),
    C("calc", "loan 200000 ka, 12% saalana, 24 mahine — emi?", ["CALCULATE"]),
    C("calc", "customer ne 2000 ka note diya, bill 1365 ka, kitna lautana hai", ["CALCULATE"]),
    C("calc", "what is 15% off on 2899", ["CALCULATE"]),
    C("calc", "8 log 2760 ka bill split", ["CALCULATE"]),
    C("calc", "is mahine 87000 ki sale, pichle mahine 72000. growth kitni?", ["CALCULATE"]),
    C("calc", "50 kg chawal 2450 me aaya, kilo ka kitna pada", ["CALCULATE"]),
    C("calc", "12 dozen eggs at 78 per dozen?", ["CALCULATE"]),
    C("calc", "144 / 12", ["CALCULATE"]),
    C("calc", "999 times 7", ["CALCULATE"]),
    C("calc", "35 degree celsius fahrenheit me kitna", ["CALCULATE"]),
    C("calc", "5 miles in km", ["CALCULATE"]),
    C("calc", "₹4999 pe 10% discount fir 18% gst", ["CALCULATE"]),
    C("calc", "25000 ka 5% kitna hota hai", ["CALCULATE"]),
    C("calc", "1 crore requests ko 86400 se divide karo", ["CALCULATE"]),
    C("calc", "700 ko 0 se divide karo", ["CALCULATE"], err_ok=True),
    C("calc", "2 lakh ka 8% simple interest 3 saal", ["CALCULATE"], allow=["200000"]),
    # ── dates ──
    C("date", "aaj kaunsi tareekh hai?", ["DATE"]),
    C("date", "what's the date today", ["DATE"]),
    C("date", "abhi kitna baja hai", ["DATE"]),
    C("date", "diwali me kitne din reh gaye?", ["DATE"]),
    C("date", "holi 2027 kab hai", ["DATE"]),
    C("date", "15 din baad kaunsi date padegi", ["DATE"]),
    C("date", "26 january 2027 ko kaunsa din hai", ["DATE"]),
    C("date", "how many days till christmas", ["DATE"]),
    C("date", "1 jan 2027 se 31 march 2027 tak kitne din", ["DATE"]),
    C("date", "rakhi kab hai", ["DATE"]),
    C("date", "31 feb 2027 kaunsa din", ["DATE"], err_ok=True, allow=["15", "2027"]),
    C("date", "working days till 31 october 2026?", ["DATE"]),
    # ── general knowledge ──
    C("search", "UPI kaise kaam karta hai?", ["SEARCH"]),
    C("search", "what is docker", ["SEARCH"]),
    C("search", "FSSAI licence kya hota hai", ["SEARCH"]),
    C("search", "kubernetes kya hai", ["SEARCH"]),
    C("search", "e-way bill kya hai", ["SEARCH"]),
    C("search", "who built the taj mahal", ["SEARCH"]),
    C("search", "ISRO ke baare me batao", ["SEARCH"]),
    C("search", "JWT samjhao", ["SEARCH"]),
    # ── routing contrast (same word, different intent) ──
    C("route", "mishti doi kya hoti hai?", ["SEARCH"], kb="sweets"),
    C("route", "mishti doi ka rate?", MQ, kb="sweets"),
    C("route", "chikankari kya hota hai", ["SEARCH"], kb="ethnic_wear"),
    C("route", "chikankari kurti hai?", MQ, kb="ethnic_wear"),
    C("route", "paneer kisse banta hai", ["SEARCH"], kb="dairy"),
    C("route", "paneer 200g kitne ka", MQ, kb="dairy"),
    # ── weather ──
    C("weather", "pune me aaj mausam kaisa hai", ["WEATHER"]),
    C("weather", "will it rain in mumbai today?", ["WEATHER"]),
    C("weather", "delhi me thand hai kya", ["WEATHER"]),
    C("weather", "chennai temperature?", ["WEATHER"]),
    C("weather", "aaj indore me chhata lena padega?", ["WEATHER"]),
    # ── code ──
    C("code", "sort karo [45, 12, 89, 3, 27]", ["RUN_CODE"]),
    C("code", "reverse the string 'milo bot'", ["RUN_CODE"]),
    C("code", "first 8 fibonacci numbers", ["RUN_CODE"]),
    C("code", "is code me kya error hai:\nnums = [1, 2]\nprint(nums[5])", ["RUN_CODE"], err_ok=True),
    C("code", "primes below 30", ["RUN_CODE"]),
    # ── no tool ──
    C("no_tool", "namaste", []),
    C("no_tool", "hello ji", []),
    C("no_tool", "thank you so much", []),
    C("no_tool", "ok bye", []),
    C("no_tool", "tum bot ho kya?", []),
    C("no_tool", "what can you do?", []),
    C("no_tool", "iska rate kya hai?", []),
    C("no_tool", "kitne ka hai", []),
    C("no_tool", "mujhe order dena hai", []),
    C("no_tool", "2 kilo atta aur 1 tel bhej do", [None, "MQ_SEARCH"], kb="kirana"),   # either is fine
    C("no_tool", "mera order abhi tak nahi aaya", []),
    C("no_tool", "order no 45821 kahan hai?", []),
    C("no_tool", "jo cake aaya tha wo kharab tha", []),
    C("no_tool", "owner se baat karni hai", []),
    C("no_tool", "₹649 gpay kar diya hai", []),
    C("no_tool", "mujhe sar dard hai koi dawai batao", []),
    C("no_tool", "recursion kya hota hai?", []),
    C("no_tool", "GET vs POST difference?", []),
    C("no_tool", "18% gst zyada hai kya?", [], allow=["18"]),
    C("no_tool", "naye customers kaise laun?", []),
    C("no_tool", "gst kitna lagega?", [], allow=["18"]),
    C("no_tool", "theek hai", []),
]


# ═══════════════════════════════════════════════════════════════
# TEST SPLIT — held out for good.
# DO NOT copy these phrasings into generate_tool_data_v4.py. (The DEV cases above
# leaked into v4.1 templates, so dev numbers for v4.1+ are optimistic.)
# ═══════════════════════════════════════════════════════════════
def Tc(*a, **k):
    c = C(*a, **k); c["split"] = "test"; return c


TEST_CASES = [
    # ── store: product ──
    Tc("store_product", "sona masoori 5 kg ka kya rate chal raha hai", MQ, kb="kirana"),
    Tc("store_product", "mustard oil hai? 1 litre wala", MQ, kb="kirana"),
    Tc("store_product", "aapke yahan besan milta hai kya", MQ, kb="kirana"),
    Tc("store_product", "chai patti 500g kitne ki", MQ, kb="kirana"),
    Tc("store_product", "detergent powder ka 1 kg pack?", MQ, kb="kirana"),
    Tc("store_product", "poha hai?", MQ, kb="kirana"),
    Tc("store_product", "nahane wala sabun ka 4 pack kitne ka", MQ, kb="kirana"),
    Tc("store_product", "how much is your toor dal", MQ, kb="kirana"),
    Tc("store_product", "lightning cable price bata do", MQ, kb="mobile"),
    Tc("store_product", "back cover milega redmi ka?", MQ, kb="mobile"),
    Tc("store_product", "tempered glass lagwana hai, rate?", MQ, kb="mobile"),
    Tc("store_product", "64gb memory card kitne ka", MQ, kb="mobile"),
    Tc("store_product", "battery change karwani hai phone ki, kitna lagega", MQ, kb="mobile"),
    Tc("store_product", "anarkali suit set ka price kya hai", MQ, kb="ethnic_wear"),
    Tc("store_product", "mens kurta hai M size?", MQ, kb="ethnic_wear"),
    Tc("store_product", "palazzo kitne ka", MQ, kb="ethnic_wear"),
    Tc("store_product", "pineapple cake 1kg?", MQ, kb="bakery"),
    Tc("store_product", "brown bread available?", MQ, kb="bakery"),
    Tc("store_product", "cupcakes ka 6 wala pack kitne ka hai", MQ, kb="bakery"),
    Tc("store_product", "200 gram paneer ka rate", MQ, kb="dairy"),
    Tc("store_product", "khoya milega aadha kilo", MQ, kb="dairy"),
    Tc("store_product", "lassi kitne ki hai", MQ, kb="dairy"),
    Tc("store_product", "cat litter 5kg ka price", MQ, kb="pet_shop"),
    Tc("store_product", "kutte ka bed medium size hai?", MQ, kb="pet_shop"),
    Tc("store_product", "fridge thik karwana hai, charge kitna", MQ, kb="repair"),
    Tc("store_product", "inverter battery 150Ah ka rate?", MQ, kb="repair"),
    Tc("store_product", "pant silwane ka kitna", MQ, kb="tailor"),
    Tc("store_product", "spoken english course fees?", MQ, kb="coaching"),
    Tc("store_product", "tally gst course kitne ka hai", MQ, kb="coaching"),
    Tc("store_product", "tamatar kya bhav hai aaj", MQ, kb="fruits_veg"),
    Tc("store_product", "nariyal hai?", MQ, kb="fruits_veg"),
    Tc("store_product", "wall putty 20 kg price", MQ, kb="hardware"),
    Tc("store_product", "gulab jamun ek kilo kitne ka", MQ, kb="sweets"),
    Tc("store_product", "geometry box kitne ka hai", MQ, kb="stationery"),
    Tc("store_product", "hair spa ka charge?", MQ, kb="salon"),
    # ── store: FAQ ──
    Tc("store_faq", "dukaan subah kitne baje khulti hai?", MQ, kb="kirana"),
    Tc("store_faq", "card swipe hota hai?", MQ, kb="kirana"),
    Tc("store_faq", "kharab saman wapas hoga?", MQ, kb="kirana"),
    Tc("store_faq", "delivery charge lagta hai kya", MQ, kb="kirana"),
    Tc("store_faq", "shop kis area me hai?", MQ, kb="mobile"),
    Tc("store_faq", "phone repair hone me kitna time lagta hai", MQ, kb="mobile"),
    Tc("store_faq", "earbuds pe guarantee hai kya", MQ, kb="mobile"),
    Tc("store_faq", "kurti exchange ho jayegi?", MQ, kb="ethnic_wear"),
    Tc("store_faq", "do you deliver outside the city?", MQ, kb="ethnic_wear"),
    Tc("store_faq", "eggless cake bhi banate ho?", MQ, kb="bakery"),
    Tc("store_faq", "custom cake ke liye kitne din pehle batana hoga", MQ, kb="bakery"),
    Tc("store_faq", "monthly doodh ka hisaab kaise hota hai", MQ, kb="dairy"),
    Tc("store_faq", "AC wala ghar pe aa jayega?", MQ, kb="repair"),
    Tc("store_faq", "repair ki warranty milti hai?", MQ, kb="repair"),
    Tc("store_faq", "urgent silai ho sakti hai?", MQ, kb="tailor"),
    Tc("store_faq", "fees wapas milti hai agar course chhod du", MQ, kb="coaching"),
    Tc("store_faq", "classes online bhi hain kya", MQ, kb="coaching"),
    Tc("store_faq", "sabzi subah kab tak aa jayegi", MQ, kb="fruits_veg"),
    Tc("store_faq", "shaadi ke liye 20 kilo mithai ka order", MQ, kb="sweets"),
    Tc("store_faq", "pdf bhejun to print nikal doge?", MQ, kb="stationery"),
    Tc("store_faq", "booking karni padti hai ya seedha aa jaun", MQ, kb="salon"),
    Tc("store_faq", "aaj ka khana kya hai", MQ, kb="tiffin"),
    Tc("store_faq", "dinner ka order kab tak de sakte hai", MQ, kb="tiffin"),
    Tc("store_faq", "do you take paytm", MQ, kb="cosmetics"),
    Tc("store_faq", "tempo se cement bhejoge? kitna lagega", MQ, kb="hardware"),
    # ── store: not stocked ──
    Tc("store_nomatch", "samsung tv hai?", MQ, kb="kirana"),
    Tc("store_nomatch", "aapke paas cigarette milegi", MQ, kb="stationery"),
    Tc("store_nomatch", "do you have chicken biryani", MQ, kb="bakery"),
    Tc("store_nomatch", "jeans milegi?", MQ, kb="dairy"),
    Tc("store_nomatch", "bike ka tyre hai", MQ, kb="mobile"),
    Tc("store_nomatch", "rasmalai milegi?", MQ, kb="kirana"),
    Tc("store_nomatch", "printer cartridge available?", MQ, kb="salon"),
    Tc("store_nomatch", "fish fry hai kya", MQ, kb="tiffin"),
    Tc("store_nomatch", "football hai?", MQ, kb="cosmetics"),
    Tc("store_nomatch", "car wash karte ho?", MQ, kb="tailor"),
    # ── store: chains ──
    Tc("store_chain", "poha ke 4 packet ka total", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    Tc("store_chain", "char packet besan kitne ka hoga", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    Tc("store_chain", "toor dal x 3, total kitna", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    Tc("store_chain", "teen back cover le lu to kitna dena hoga", MQ, ["MQ_SEARCH", "CALCULATE"], kb="mobile"),
    Tc("store_chain", "I need 6 gel pens, what will it cost", MQ, ["MQ_SEARCH", "CALCULATE"], kb="stationery"),
    Tc("store_chain", "2 pineapple cake 1kg aur 1 brown bread ka bill", MQ, ["MQ_SEARCH", "CALCULATE"], kb="bakery"),
    Tc("store_chain", "5 dozen kela kitne ka padega", MQ, ["MQ_SEARCH", "CALCULATE"], kb="fruits_veg"),
    Tc("store_chain", "paneer 1kg ke 3 pack aur 2 curd ka total", MQ, ["MQ_SEARCH", "CALCULATE"], kb="dairy"),
    Tc("store_chain", "desi ghee offer laga ke kitne ka milega", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    Tc("store_chain", "12000 ka order dunga, kitni chhoot milegi", MQ, ["MQ_SEARCH", "CALCULATE"], kb="hardware"),
    Tc("store_chain", "1500 ka saman hai, bulk discount lagega kya", MQ, ["MQ_SEARCH", "CALCULATE"], kb="kirana"),
    Tc("store_chain", "agar abhi order karun to saman kab tak pahuchega", MQ, ["MQ_SEARCH", "DATE"], kb="ethnic_wear"),
    Tc("store_chain", "pet food order kiya aaj, kab aayega", MQ, ["MQ_SEARCH", "DATE"], kb="pet_shop"),
    Tc("store_chain", "shirt aaj silne di to kab tak tayyar", MQ, ["MQ_SEARCH", "DATE"], kb="tailor"),
    Tc("store_chain", "2 hammer aur 3 padlock kitne ke", MQ, ["MQ_SEARCH", "CALCULATE"], kb="hardware"),
    Tc("store_chain", "10 veg thali party ke liye, total?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="tiffin"),
    Tc("store_chain", "3 facial ka kitna hoga", MQ, ["MQ_SEARCH", "CALCULATE"], kb="salon"),
    Tc("store_chain", "two face wash and one sunscreen, total?", MQ, ["MQ_SEARCH", "CALCULATE"], kb="cosmetics"),
    Tc("store_chain", "chhe soan papdi 500g ka total batao", MQ, ["MQ_SEARCH", "CALCULATE"], kb="sweets"),
    Tc("store_chain", "4 ceiling fan ka total kitna aayega", MQ, ["MQ_SEARCH", "CALCULATE"], kb="hardware"),
    # ── routing ──
    Tc("route", "basmati chawal kya hota hai, normal se alag kyun", ["SEARCH"], kb="kirana"),
    Tc("route", "basmati 1kg kitne ka", MQ, kb="kirana"),
    Tc("route", "banarasi saree kya hoti hai", ["SEARCH"], kb="ethnic_wear"),
    Tc("route", "banarasi saree dikhao kitne ki hai", MQ, kb="ethnic_wear"),
    Tc("route", "power bank kya hota hai", ["SEARCH"], kb="mobile"),
    Tc("route", "power bank 10000 wala hai?", MQ, kb="mobile"),
    Tc("route", "alphonso mango kya hota hai", ["SEARCH"], kb="fruits_veg"),
    Tc("route", "alphonso ka 1 dozen box hai?", MQ, kb="fruits_veg"),
    Tc("route", "tally kya hai", ["SEARCH"], kb="coaching"),
    Tc("route", "tally course chalta hai yahan?", MQ, kb="coaching"),
    # ── calc ──
    Tc("calc", "1850 pe 5% gst kitna", ["CALCULATE"]),
    Tc("calc", "invoice 26,500 + 18% GST, grand total?", ["CALCULATE"]),
    Tc("calc", "₹590 me 18 percent gst shamil hai, base kitna", ["CALCULATE"]),
    Tc("calc", "780 ka maal 950 me becha, profit kitna hua", ["CALCULATE"]),
    Tc("calc", "1200 ki cheez 20 percent off pe kitne ki", ["CALCULATE"]),
    Tc("calc", "5 lakh ka loan, 10.5% interest, 60 months emi?", ["CALCULATE"]),
    Tc("calc", "char log, 1640 ka khana, ek ka hissa", ["CALCULATE"]),
    Tc("calc", "500 ka note diya, 287 ka saman, wapas kitne", ["CALCULATE"]),
    Tc("calc", "45 litre 4140 rupaye me, ek litre ka kitna", ["CALCULATE"]),
    Tc("calc", "6 mazdoor 8 din 550 roz, kul kharcha", ["CALCULATE"]),
    Tc("calc", "1.5 lakh ka 12 percent", ["CALCULATE"]),
    Tc("calc", "last year 3.2 lakh revenue, this year 4.1 lakh, growth %", ["CALCULATE"]),
    Tc("calc", "7 dozen ande 84 rupaye dozen", ["CALCULATE"]),
    Tc("calc", "monthly 22000, saal ka kitna", ["CALCULATE"]),
    Tc("calc", "40 x 25 ft plot ka area", ["CALCULATE"]),
    Tc("calc", "86 fahrenheit to celsius", ["CALCULATE"]),
    Tc("calc", "12 km kitne mile", ["CALCULATE"]),
    Tc("calc", "4589 + 7712 + 230", ["CALCULATE"]),
    Tc("calc", "65000 se 52000 aa gayi sale, kitna percent gira", ["CALCULATE"]),
    Tc("calc", "fd 75000, 7% simple interest, 2 saal", ["CALCULATE"]),
    Tc("calc", "cost 160 hai, 30% markup chahiye", ["CALCULATE"]),
    Tc("calc", "2499 pe 15% off then 5% gst", ["CALCULATE"]),
    Tc("calc", "2 crore requests per day, per second kitne", ["CALCULATE"]),
    Tc("calc", "300 / 0", ["CALCULATE"], err_ok=True),
    Tc("calc", "is 4500 more than 3999?", ["CALCULATE"]),
    # ── date ──
    Tc("date", "aaj ka din kaunsa hai", ["DATE"]),
    Tc("date", "today date batao", ["DATE"]),
    Tc("date", "kitne baj gaye", ["DATE"]),
    Tc("date", "dussehra kab padega", ["DATE"]),
    Tc("date", "bhai dooj me kitne din", ["DATE"]),
    Tc("date", "new year aane me kitna time hai", ["DATE"]),
    Tc("date", "10 din baad ki date", ["DATE"]),
    Tc("date", "15 august 2027 ko kaunsa vaar", ["DATE"]),
    Tc("date", "what day was 2 october 2026", ["DATE"]),
    Tc("date", "1 nov 2026 se 31 dec 2026 kitne din", ["DATE"]),
    Tc("date", "ganpati 2027 kab hai", ["DATE"]),
    Tc("date", "holi tak kitne din bache", ["DATE"]),
    Tc("date", "30 feb 2027 kaunsa din hai", ["DATE"], err_ok=True, allow=["15", "2027"]),
    Tc("date", "republic day kis din hai", ["DATE"]),
    Tc("date", "working days left till 20 november", ["DATE"]),
    # ── search ──
    Tc("search", "inflation ka matlab kya hota hai", ["SEARCH"]),
    Tc("search", "docker kya cheez hai", ["SEARCH"]),
    Tc("search", "what is kubernetes used for", ["SEARCH"]),
    Tc("search", "upi kisne banaya", ["SEARCH"]),
    Tc("search", "e-way bill kab lagta hai", ["SEARCH"]),
    Tc("search", "taj mahal kab bana", ["SEARCH"]),
    Tc("search", "isro kahan hai", ["SEARCH"]),
    Tc("search", "jwt token kya hota hai", ["SEARCH"]),
    Tc("search", "fssai registration kya hai", ["SEARCH"]),
    Tc("search", "explain inflation simply", ["SEARCH"]),
    Tc("search", "paneer kis cheez se banta hai", ["SEARCH"]),
    Tc("search", "chikankari kahan ki famous hai", ["SEARCH"]),
    # ── weather ──
    Tc("weather", "jaipur me garmi kitni hai aaj", ["WEATHER"]),
    Tc("weather", "kolkata ka mausam batao", ["WEATHER"]),
    Tc("weather", "is it raining in bangalore", ["WEATHER"]),
    Tc("weather", "shimla me sardi hai kya abhi", ["WEATHER"]),
    Tc("weather", "lucknow temperature", ["WEATHER"]),
    Tc("weather", "pune me barish hogi aaj?", ["WEATHER"]),
    Tc("weather", "hyderabad weather kaisa hai", ["WEATHER"]),
    Tc("weather", "goa me aaj bahar jaana theek rahega?", ["WEATHER"]),
    Tc("weather", "surat me umbrella chahiye?", ["WEATHER"]),
    Tc("weather", "patna me kitni thand hai", ["WEATHER"]),
    # ── code ──
    Tc("code", "[9, 4, 7, 1] ko descending sort karo", ["RUN_CODE"]),
    Tc("code", "'kirana' ulta likho code se", ["RUN_CODE"]),
    Tc("code", "factorial 12", ["RUN_CODE"]),
    Tc("code", "100 se chhote saare prime nikalo", ["RUN_CODE"]),
    Tc("code", "ye error kyun aa raha:\nx = '5' + 5", ["RUN_CODE"], err_ok=True),
    Tc("code", "convert 1000 to binary", ["RUN_CODE"]),
    # ── no tool ──
    Tc("no_tool", "hii", []),
    Tc("no_tool", "good night", []),
    Tc("no_tool", "shukriya bhai", []),
    Tc("no_tool", "acha theek hai ji", []),
    Tc("no_tool", "aap kaun ho", []),
    Tc("no_tool", "kya kya help kar sakte ho?", []),
    Tc("no_tool", "isko kitne me doge?", []),
    Tc("no_tool", "price batao", []),
    Tc("no_tool", "saman mangwana hai", []),
    Tc("no_tool", "3 kilo cheeni pack kar dena", [None, "MQ_SEARCH"], kb="kirana"),
    Tc("no_tool", "parcel nahi mila abhi tak", []),
    Tc("no_tool", "order 77215 ka kya hua", []),
    Tc("no_tool", "doodh khatta nikla", []),
    Tc("no_tool", "seth ji se baat karao", []),
    Tc("no_tool", "phonepe se 1180 bhej diya", []),
    Tc("no_tool", "pet me dard hai dawai bolo", []),
    Tc("no_tool", "api ka matlab samjha do", []),
    Tc("no_tool", "python list vs tuple", []),
    Tc("no_tool", "dukaan ki sale kaise badhaun", []),
    Tc("no_tool", "pehli baar aane wale grahak ko chhoot du kya?", []),
    Tc("no_tool", "ok", []),
    Tc("no_tool", "👍", []),
    Tc("no_tool", "india aaj ka match jeetega kya", []),
    Tc("no_tool", "mera assignment likh do", []),
    Tc("no_tool", "kya tum insaan ho", []),
    Tc("no_tool", "GST simple bhasha me", []),
    Tc("no_tool", "WhatsApp pe dukaan kaise promote karu", []),
    Tc("no_tool", "bye bye", []),
    Tc("no_tool", "theek hai kal aata hu", []),
    Tc("no_tool", "naraz grahak ko kaise shant karu", []),
]


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════
CALC_CONSTANTS = {"100", "12", "1000", "60", "24", "365", "1024", "1048576", "1.60934", "0.453592",
                  "2.20462", "0.3048", "2.54", "0.0929", "32", "100000", "10000000"}
UNAVAILABLE = ("out of stock", "batch full", "fully booked")


def _words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def score(case, response, calls):
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
    usable = [r for r in results if not r.startswith(("Error", "No match"))
              and not all(any(u in ln for u in UNAVAILABLE) for ln in r.split("\n"))]
    result_nums = {n for r in usable for n in T.numbers_in(r) if len(n.replace(".", "")) > 1}
    s["uses"] = (bool(result_nums & T.numbers_in(response)) if result_nums and case["cat"] not in ("search", "route", "store_faq")
                 else None)
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
def make_agent(checkpoint, device, max_tool_calls, constrain=False):
    import react_runtime as RT
    model = RT.load_model(checkpoint, device)

    def agent(q):
        return RT.react_loop(model, q, device=device, max_tool_calls=max_tool_calls,
                             temperature=0.0, verbose=False, guard=False, constrain=constrain)
    return agent


def run(agent, cases, live=False):
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
        except Exception as e:                       # a crash is a failure, not a stop
            response, calls = f"[crash: {type(e).__name__}: {e}]", []
        sc = score(case, response, calls)
        out.append(dict(case=case, response=response, calls=calls, score=sc, secs=round(time.time() - t0, 2)))
    T.set_now(None); T.set_merchant_kb(None)
    return out


def summarize(results):
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


CAT_ORDER = ["store_product", "store_faq", "store_nomatch", "store_chain", "route", "calc", "date",
             "search", "weather", "code", "no_tool", "ALL"]


def print_table(names, summaries):
    print("\n" + "PASS RATE (%) by category" + "".join(f"  {n[:18]:>18}" for n in names))
    for cat in CAT_ORDER:
        if cat not in summaries[0]:
            continue
        line = f"  {cat:<15} n={summaries[0][cat]['n']:<3}"
        for s in summaries:
            line += f"  {fmt(s[cat]['pass']):>18}"
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
    ap.add_argument("--split", choices=["dev", "test", "both"], default="both",
                    help="dev = original 138 (leaked into v4.1 templates); test = held-out")
    ap.add_argument("--live", action="store_true", help="real SEARCH/WEATHER instead of stubs")
    ap.add_argument("--constrain", choices=["off", "on", "both"], default="off",
                    help="constrained copy decoding; 'both' runs each checkpoint twice to compare")
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
            names.append(os.path.basename(path).replace(".jsonl", "")); sums.append(rows); last = rows
        for split in ("dev", "test"):
            rs = [[r for r in rows_ if r["case"].get("split", "dev") == split] for rows_ in sums]
            if rs[0]:
                print(f"\n══ {split.upper()} SPLIT ({len(rs[0])} cases) ══")
                print_table(names, [summarize(r) for r in rs])
        print_failures(last, args.show_failures)
        return
    if not args.checkpoint:
        ap.error("--checkpoint or --rescore required")
    os.makedirs("eval_results", exist_ok=True)
    names, summaries, all_results = [], [], []
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
            import react_runtime as RT
            st = RT.CONSTRAINT_STATS
            print(f"  constraint: {st['fallbacks']} fallbacks in {st['steps']} constrained steps (should be ~0)")
            st["steps"] = st["fallbacks"] = 0
        with open(f"eval_results/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        names.append(name); all_results.append(results)
        del agent
    for split in ("dev", "test"):
        rs = [[r for r in res if r["case"].get("split", "dev") == split] for res in all_results]
        if rs[0]:
            print(f"\n══ {split.upper()} SPLIT ({len(rs[0])} cases){' — held out' if split == 'test' else ' — leaked into v4.1+ data'} ══")
            print_table(names, [summarize(r) for r in rs])
    print_failures(all_results[-1], args.show_failures)
    print(f"\nFull traces: eval_results/<name>.jsonl")


if __name__ == "__main__":
    main()