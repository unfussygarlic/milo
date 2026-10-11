"""MQ_SEARCH: exact result formats on a fixed KB, and top-1 retrieval accuracy."""

import random

import pytest

import milo.tools as T
from milo.data.merchants import build_kb


@pytest.fixture(autouse=True)
def _no_ambient_kb():
    yield
    T.set_merchant_kb(None)


def test_mq_search_formats():
    KB = {
        "store": {"name": "Test Kirana", "city": "Indore"},
        "catalog": [
            {
                "name": "Basmati Rice",
                "variant": "1kg",
                "price": 139,
                "stock": "in stock",
                "keywords": ["basmati", "rice"],
            },
            {
                "name": "Basmati Rice",
                "variant": "5kg",
                "price": 649,
                "stock": "out of stock",
                "keywords": ["basmati", "rice"],
            },
            {
                "name": "Mustard Oil",
                "variant": "1l",
                "price": 185.5,
                "stock": "only 2 left",
                "keywords": ["mustard", "sarson", "oil"],
                "offer": "10% off",
            },
            {
                "name": "Screen Replacement",
                "variant": "",
                "price": 1499,
                "price_from": True,
                "stock": "available",
                "keywords": ["screen"],
            },
            {
                "name": "Demo Class",
                "variant": "",
                "price": 0,
                "stock": "seats available",
                "keywords": ["demo", "trial"],
            },
        ],
        "faq": [
            {
                "topic": "Store hours",
                "answer": "9 AM – 9 PM, all days.",
                "keywords": ["hours", "timing", "open", "baje"],
            },
            {"topic": "Delivery", "answer": "Home delivery in 2 days.", "keywords": ["delivery", "home"]},
        ],
    }

    def mq(q):
        return T.mq_search(q, kb=KB)

    assert mq("basmati 5kg") == "Basmati Rice 5kg | ₹649 incl. GST | out of stock", "mq variant pick"
    assert (
        mq("chawal ka rate")
        == "Basmati Rice 1kg | ₹139 incl. GST | in stock\nBasmati Rice 5kg | ₹649 incl. GST | out of stock"
    ), "mq hinglish + both variants"
    assert mq("1 litre sarson tel") == "Mustard Oil 1l | ₹185.50 incl. GST | only 2 left | offer: 10% off", (
        "mq unit words"
    )
    assert mq("screen replace") == "Screen Replacement | from ₹1499 incl. GST | available", "mq from-price"
    assert mq("demo class") == "Demo Class | free | seats available", "mq free"
    assert mq("dukaan kitne baje khulti hai") == "Store hours: 9 AM – 9 PM, all days.", "mq faq"
    assert mq("ghar pe bhej doge?") == "Delivery: Home delivery in 2 days.", "mq faq synonym"
    assert mq("basmti 1kg") == "Basmati Rice 1kg | ₹139 incl. GST | in stock", "mq typo"
    assert mq("kaju 1kg") == "No match in store for: kaju 1kg", "mq no match"
    assert mq("kya hai bhai") == "No match in store for: kya hai bhai", "mq only stopwords"
    assert (T.set_merchant_kb(KB), T.execute("MQ_SEARCH", "demo"))[1] == "Demo Class | free | seats available", (
        "mq via execute"
    )
    T.set_merchant_kb(None)
    assert (
        T.execute("MQ_SEARCH", "rice")
        if not __import__("os").environ.get("MILO_MERCHANT_KB")
        else "Error: store catalog not available"
    ) == "Error: store catalog not available", "mq no kb"


def test_mq_top1_accuracy():
    """Every merchant type, 20 randomised catalogues each: the right line must come first."""
    SUITE = {
        "kirana": [
            ("basmati 5kg", "Basmati Rice 5kg"),
            ("sarson ka tel", "Mustard Oil"),
            ("aadha kilo besan", "Besan 500g"),
            ("maggi hai?", "Instant Noodles"),
            ("atta 10 kg", "Wheat Atta 10kg"),
            ("toor dal", "Toor Dal"),
            ("cheeni 1 kilo", "Sugar"),
            ("ghar delivery", "Delivery:"),
            ("upi chalega", "Payment:"),
            ("kitne baje khulta hai", "Store hours:"),
            ("return kar sakte", "Returns:"),
            ("gst bill milega", "GST bill:"),
        ],
        "mobile": [
            ("20000 mah power bank", "Power Bank 20000mAh"),
            ("screen toot gaya replace", "Screen Replacement"),
            ("type c cable", "Type-C Cable"),
            ("tempered glass", "Tempered Glass"),
            ("warranty", "Warranty:"),
            ("bluetooth earbuds", "Bluetooth Earbuds"),
            ("64 gb pendrive", "Pendrive 64GB"),
        ],
        "bakery": [
            ("black forest 1 kg", "Black Forest Cake 1kg"),
            ("photo cake", "Custom cake:"),
            ("veg puff", "Veg Puff"),
            ("brown bread", "Brown Bread"),
        ],
        "tailor": [
            ("blouse silai", "Blouse Stitching"),
            ("kitne din me ready", "Stitching time:"),
            ("fall pico", "Saree Fall Pico"),
        ],
        "salon": [("hair spa", "Hair Spa"), ("bridal makeup", "Bridal Makeup"), ("appointment", "Appointment:")],
        "dairy": [
            ("paneer 1kg", "Paneer 1kg"),
            ("doodh 1 litre", "Cow Milk"),
            ("roz subah doodh", "Milk subscription:"),
        ],
        "fruits_veg": [("pyaz 1 kg", "Onion 1kg"), ("tamatar", "Tomato"), ("kela", "Banana"), ("palak", "Spinach")],
        "repair": [
            ("fridge repair", "Fridge Repair"),
            ("ac gas refill", "AC Gas Refill"),
            ("ghar aake technician", "Home visit:"),
        ],
        "coaching": [
            ("python course", "Python Programming Course"),
            ("demo class", "Demo Class"),
            ("batch timing", "Batch timings:"),
        ],
        "hardware": [
            ("cement bag", "Cement"),
            ("led bulb 9w", "LED Bulb 9W"),
            ("pankha", "Ceiling Fan"),
            ("tala", "Padlock"),
        ],
        "sweets": [
            ("kaju katli 1kg", "Kaju Katli 1kg"),
            ("rasgulla", "Rasgulla"),
            ("shaadi ka order", "Wedding orders:"),
        ],
        "pet_shop": [("puppy food", "Puppy Food"), ("billi ka khana", "Cat Food"), ("cat litter", "Cat Litter")],
        "tiffin": [
            ("monthly tiffin veg", "Monthly Tiffin Veg"),
            ("lunch order kab tak", "Order cutoff:"),
            ("extra roti", "Extra Roti"),
        ],
        "stationery": [
            ("xerox", "Xerox Black & White"),
            ("geometry box", "Geometry Box"),
            ("spiral binding", "Spiral Binding"),
        ],
        "ethnic_wear": [
            ("chikankari kurti", "Chikankari Kurti"),
            ("banarasi saree", "Banarasi Silk Saree"),
            ("alteration", "Alteration:"),
        ],
        "cosmetics": [("sunscreen", "Sunscreen SPF 50"), ("kajal", "Kajal"), ("face wash", "Face Wash")],
    }
    rng = random.Random(0)
    hits = total = 0
    misses = []
    for _ in range(20):  # 20 random variants of every merchant type
        for mtype, cases in SUITE.items():
            kb = build_kb(mtype, rng)
            for q, want in cases:
                total += 1
                top = T.mq_search(q, kb=kb).split("\n")[0]
                if top.startswith(want):
                    hits += 1
                elif len(misses) < 5:
                    misses.append((mtype, q, top))
    acc = hits / total
    if acc < 0.98:
        for m in misses:
            print("   miss:", m)
    assert acc >= 0.98, f"mq top-1 accuracy >= 98% ({acc:.1%} of {total})"
