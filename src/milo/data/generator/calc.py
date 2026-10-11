"""CALCULATE families: GST, discounts, margin, EMI, conversions, dev maths - and tool errors."""

from milo.data.generator.core import Ex, L, R, Skip, by_lang, fill, new_now, pick
from milo.data.generator.store import qty_text

# ═══════════════════════════════════════════════════════════════
# FAMILY: general CALCULATE (business + everyday + SDE)
# ═══════════════════════════════════════════════════════════════
PRICES = [
    99,
    149,
    199,
    249,
    299,
    349,
    399,
    449,
    499,
    599,
    649,
    799,
    899,
    999,
    1199,
    1299,
    1499,
    1799,
    1999,
    2499,
    2999,
    3499,
    4999,
    5999,
    7999,
    9999,
    12000,
    14999,
    18500,
    25000,
    32000,
    49999,
    75000,
]
GST_RATES = [5, 5, 18, 18, 18, 40]


def calc_gst_add():
    lang = L()
    p = pick(PRICES)
    r = pick(GST_RATES)
    user, _ = fill(
        by_lang(
            lang,
            [
                "₹{p} pe {r}% GST kitna lagega?",
                "₹{p} + {r}% GST total kitna?",
                "customer ka bill ₹{p} hai, {r}% GST add karo",
                "{r}% GST on {p} kitna hua",
            ],
            [
                "what's {r}% GST on ₹{p}?",
                "add {r}% GST to ₹{p}",
                "₹{p} plus {r}% GST — final amount?",
                "bill is ₹{p}, add {r}% GST",
            ],
        ),
        lang,
        p=p,
        r=r,
    )
    ex = Ex("calc_gst_add", user)
    g = ex.tool("CALCULATE", f"{p} * {r} / 100")
    tot = ex.tool("CALCULATE", f"{p} + {g}")
    a = (
        pick([f"{r}% GST = ₹{g}. Total ₹{tot}.", f"GST ₹{g} lagega, total ₹{tot} hoga."])
        if lang == "hi"
        else pick([f"{r}% GST is ₹{g}, so the total is ₹{tot}.", f"GST: ₹{g}. Final amount: ₹{tot}."])
    )
    return ex.done(a)


def calc_gst_inclusive():
    lang = L()
    r = pick([5, 18])
    p = pick(PRICES)
    user, _ = fill(
        by_lang(
            lang,
            [
                "₹{p} me {r}% GST included hai, base price kya hoga?",
                "₹{p} incl. {r}% GST — GST kitna hai isme?",
                "₹{p} GST ke saath hai ({r}%), bina GST kitna?",
            ],
            [
                "₹{p} includes {r}% GST. What's the price before GST?",
                "how much GST is inside ₹{p} at {r}%?",
                "₹{p} inclusive of {r}% GST, base amount?",
            ],
        ),
        lang,
        p=p,
        r=r,
    )
    ex = Ex("calc_gst_incl", user)
    base = ex.tool("CALCULATE", f"{p} * 100 / (100 + {r})")
    g = ex.tool("CALCULATE", f"{p} - {base}")
    a = (
        pick([f"Base price ₹{base}, GST ₹{g}.", f"Bina GST ₹{base} hai, aur GST ₹{g}."])
        if lang == "hi"
        else pick([f"Price before GST: ₹{base}. GST: ₹{g}.", f"Base ₹{base} + GST ₹{g}."])
    )
    return ex.done(a)


def calc_cgst_sgst():
    lang = L()
    p = pick(PRICES)
    r = pick([5, 18])
    user, _ = fill(
        by_lang(
            lang,
            ["₹{p} pe {r}% GST ka CGST aur SGST alag batao", "₹{p} bill, {r}% GST — CGST SGST kitna?"],
            ["split {r}% GST on ₹{p} into CGST and SGST", "CGST and SGST on ₹{p} at {r}%?"],
        ),
        lang,
        p=p,
        r=r,
    )
    ex = Ex("calc_cgst", user)
    g = ex.tool("CALCULATE", f"{p} * {r} / 100")
    half = ex.tool("CALCULATE", f"{g} / 2")
    a = (
        f"Total GST ₹{g}: CGST ₹{half} + SGST ₹{half}."
        if lang == "hi"
        else f"Total GST is ₹{g}, split as CGST ₹{half} and SGST ₹{half}."
    )
    return ex.done(a)


def calc_discount():
    lang = L()
    p = pick(PRICES)
    d = pick([5, 10, 15, 20, 25, 30, 40, 50])
    user, _ = fill(
        by_lang(
            lang,
            [
                "₹{p} pe {d}% discount laga ke kitna hoga?",
                "MRP ₹{p}, {d}% off, kitna charge karu?",
                "{d}% off on ₹{p}?",
                "customer {d}% discount maang raha hai ₹{p} pe",
            ],
            ["₹{p} with {d}% off?", "sale price of ₹{p} after {d}% discount?", "{d}% off on ₹{p} — final price?"],
        ),
        lang,
        p=p,
        d=d,
    )
    ex = Ex("calc_discount", user)
    f = ex.tool("CALCULATE", f"{p} * (1 - {d} / 100)")
    a = (
        pick([f"{d}% discount ke baad ₹{f}.", f"Final price ₹{f} hoga."])
        if lang == "hi"
        else pick([f"After {d}% off: ₹{f}.", f"The final price is ₹{f}."])
    )
    return ex.done(a)


def calc_discount_gst():
    lang = L()
    p = pick(PRICES)
    d = pick([5, 10, 15, 20])
    r = pick([5, 18])
    user, _ = fill(
        by_lang(
            lang,
            ["₹{p} pe {d}% discount, phir {r}% GST — final kitna?", "MRP ₹{p}, {d}% off aur {r}% GST lagao"],
            ["₹{p} item, {d}% discount then {r}% GST?", "price after {d}% off and {r}% GST on ₹{p}?"],
        ),
        lang,
        p=p,
        d=d,
        r=r,
    )
    ex = Ex("calc_disc_gst", user)
    a1 = ex.tool("CALCULATE", f"{p} * (1 - {d} / 100)")
    a2 = ex.tool("CALCULATE", f"{a1} * (1 + {r} / 100)")
    a = (
        f"Discount ke baad ₹{a1}, GST ke saath ₹{a2}."
        if lang == "hi"
        else f"After {d}% off: ₹{a1}. With {r}% GST: ₹{a2}."
    )
    return ex.done(a)


def calc_margin():
    lang = L()
    c = pick([50, 80, 120, 150, 200, 250, 300, 450, 500, 750, 1000, 1500, 2200, 5000])
    s = c + R.randint(max(1, c // 10), c)
    user, _ = fill(
        by_lang(
            lang,
            ["₹{c} me kharida, ₹{s} me becha. margin kitna?", "cost ₹{c}, selling ₹{s}, profit % kya hai?"],
            ["cost ₹{c}, selling at ₹{s}. margin?", "bought for ₹{c}, sold for ₹{s} — profit and margin?"],
        ),
        lang,
        c=c,
        s=s,
    )
    ex = Ex("calc_margin", user)
    pr = ex.tool("CALCULATE", f"{s} - {c}")
    mg = ex.tool("CALCULATE", f"({s} - {c}) / {s} * 100")
    a = (
        f"Profit ₹{pr}, margin {mg}% (selling price pe)."
        if lang == "hi"
        else f"Profit is ₹{pr}; margin is {mg}% of the selling price."
    )
    return ex.done(a)


def calc_markup():
    lang = L()
    c = pick([40, 60, 100, 150, 240, 300, 480, 650, 900, 1200])
    m = pick([10, 15, 20, 25, 30, 40, 50])
    user, _ = fill(
        by_lang(
            lang,
            [
                "₹{c} cost pe {m}% markup lagau to selling price kya hoga?",
                "cost ₹{c}, {m}% profit chahiye, kitne me bechu?",
            ],
            ["cost is ₹{c}, I want {m}% markup. selling price?", "what should I sell at for {m}% markup on ₹{c}?"],
        ),
        lang,
        c=c,
        m=m,
    )
    ex = Ex("calc_markup", user)
    sp = ex.tool("CALCULATE", f"{c} * (1 + {m} / 100)")
    a = f"₹{sp} me becho." if lang == "hi" else f"Sell it at ₹{sp}."
    return ex.done(a)


def calc_emi():
    lang = L()
    P = pick([25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 1000000])
    r = pick([9, 10, 10.5, 11, 12, 13, 14, 16, 18])
    n = pick([6, 12, 18, 24, 36, 48, 60])
    amt = (
        f"{P // 100000} lakh"
        if P % 100000 == 0 and R.random() < 0.4
        else f"{P // 1000}k"
        if P % 1000 == 0 and R.random() < 0.1
        else str(P)
    )
    user, _ = fill(
        by_lang(
            lang,
            [
                "₹{amt} loan, {r}% interest, {n} months — EMI kitni?",
                "{amt} ka loan {n} mahine ke liye {r}% pe, EMI batao",
                "loan {amt} ka, {r}% saalana, {n} mahine — emi?",
                "{amt} loan {r}% pe {n} months ki kisht kitni?",
            ],
            [
                "EMI for ₹{amt} at {r}% for {n} months?",
                "loan ₹{amt}, {r}% annual, {n} months. monthly EMI?",
                "monthly installment on a {amt} loan at {r}% over {n} months?",
            ],
        ),
        lang,
        amt=amt,
        r=r,
        n=n,
    )
    ex = Ex("calc_emi", user)
    emi = ex.tool("CALCULATE", f"emi({P}, {r}, {n})")
    tot = ex.tool("CALCULATE", f"{emi} * {n}")
    intr = ex.tool("CALCULATE", f"{tot} - {P}")
    a = (
        f"EMI ₹{emi} per month. Total ₹{tot} doge, jisme ₹{intr} interest hai."
        if lang == "hi"
        else f"EMI: ₹{emi}/month. Total paid ₹{tot}, of which ₹{intr} is interest."
    )
    return ex.done(a)


def calc_simple_interest():
    lang = L()
    P = pick([10000, 20000, 50000, 100000, 250000])
    r = pick([6, 7, 8, 9, 10, 12])
    y = pick([1, 2, 3, 5])
    user, _ = fill(
        by_lang(
            lang,
            [
                "₹{amt} pe {r}% simple interest {y} saal ka kitna?",
                "{amt} FD {r}% pe {y} saal, interest kitna milega (simple)?",
            ],
            ["simple interest on ₹{amt} at {r}% for {y} {yw}?"],
        ),
        lang,
        amt=P,
        r=r,
        y=y,
        yw="year" if y == 1 else "years",
    )
    ex = Ex("calc_si", user)
    si = ex.tool("CALCULATE", f"{P} * {r} * {y} / 100")
    tot = ex.tool("CALCULATE", f"{P} + {si}")
    a = f"Interest ₹{si}, total ₹{tot}." if lang == "hi" else f"Interest: ₹{si}. Total: ₹{tot}."
    return ex.done(a)


def calc_split():
    lang = L()
    b = pick([450, 780, 1200, 1350, 1840, 2400, 3150, 4999, 6200])
    n = R.randint(2, 9)
    user, _ = fill(
        by_lang(
            lang,
            ["₹{b} ka bill {n} logo me baanto", "{n} dost, bill ₹{b}, har ek ka kitna?"],
            ["split ₹{b} between {n} people", "bill is ₹{b} for {n} of us, how much each?"],
        ),
        lang,
        b=b,
        n=n,
    )
    ex = Ex("calc_split", user)
    each = ex.tool("CALCULATE", f"{b} / {n}")
    a = f"Har ek ₹{each} dega." if lang == "hi" else f"₹{each} each."
    return ex.done(a)


def calc_change():
    lang = L()
    b = R.randint(37, 1950)
    g = pick([x for x in [50, 100, 200, 500, 1000, 2000] if x > b] or [2000])
    if g <= b:
        raise Skip("change")
    user, _ = fill(
        by_lang(
            lang,
            ["bill ₹{b} hai, customer ne ₹{g} diye, kitna wapas du?", "₹{g} me se ₹{b} kaat ke kitna bacha?"],
            ["bill ₹{b}, customer paid ₹{g}. change?", "how much change from ₹{g} for a ₹{b} bill?"],
        ),
        lang,
        b=b,
        g=g,
    )
    ex = Ex("calc_change", user)
    c = ex.tool("CALCULATE", f"{g} - {b}")
    a = f"₹{c} wapas dena hai." if lang == "hi" else f"Give back ₹{c}."
    return ex.done(a)


def calc_per_unit():
    lang = L()
    q = pick([5, 10, 25, 50, 100])
    unit = pick(["kg", "litre", "piece", "packet"])
    t = R.randrange(q * 20, q * 400, 10)
    user, _ = fill(
        by_lang(
            lang,
            ["{q} {u} ka ₹{t}, per {u} kitna pada?", "₹{t} me {q} {u} liya, ek {u} ka rate?"],
            ["₹{t} for {q} {u}. price per {u}?", "{q} {u} cost ₹{t}, what's the per-{u} rate?"],
        ),
        lang,
        q=q,
        u=unit,
        t=t,
    )
    ex = Ex("calc_per_unit", user)
    r = ex.tool("CALCULATE", f"{t} / {q}")
    a = f"₹{r} per {unit}." if lang == "en" else f"Ek {unit} ₹{r} ka pada."
    return ex.done(a)


def calc_growth():
    lang = L()
    a0 = R.randrange(20000, 200000, 500)
    a1 = int(a0 * R.uniform(0.7, 1.6)) // 100 * 100
    user, _ = fill(
        by_lang(
            lang,
            [
                "pichle mahine sale ₹{a0}, is mahine ₹{a1}. kitna % change?",
                "sales {a0} se {a1} hui, growth %?",
                "is mahine {a1} ki sale, pichle mahine {a0}. growth kitni?",
                "{a0} se {a1} hua, kitna percent badha?",
            ],
            [
                "sales went from ₹{a0} to ₹{a1}. percentage change?",
                "last month ₹{a0}, this month ₹{a1} — growth %?",
                "this month {a1}, last month {a0}. growth?",
            ],
        ),
        lang,
        a0=a0,
        a1=a1,
    )
    ex = Ex("calc_growth", user)
    g = ex.tool("CALCULATE", f"pct_change({a0}, {a1})")
    up = not g.startswith("-")
    if lang == "hi":
        a = f"Sales {g}% {'badhi' if up else 'ghati'} hai." if up else f"Sales {g}% change hui (kami aayi)."
    else:
        a = f"That's a {g}% change ({'up' if up else 'down'})."
    return ex.done(a)


def calc_daily_target():
    lang = L()
    t = R.randrange(50000, 1500000, 5000)
    d = pick([22, 24, 25, 26, 30])
    user, _ = fill(
        by_lang(
            lang,
            ["mahine ka target ₹{t} hai, {d} din kaam, roz kitna bechna padega?"],
            ["monthly target ₹{t}, {d} working days. daily target?"],
        ),
        lang,
        t=t,
        d=d,
    )
    ex = Ex("calc_target", user)
    r = ex.tool("CALCULATE", f"{t} / {d}")
    a = f"Roz ₹{r} ki sale chahiye." if lang == "hi" else f"You need ₹{r} per day."
    return ex.done(a)


def calc_salary():
    lang = L()
    m = pick([12000, 15000, 18000, 22000, 25000, 30000, 45000, 60000, 85000])
    if R.random() < 0.5:
        user, _ = fill(
            by_lang(
                lang,
                ["salary ₹{m} mahina, saal ka kitna?", "₹{m} monthly ka annual kitna?"],
                ["₹{m} a month — yearly?", "annual salary for ₹{m}/month?"],
            ),
            lang,
            m=m,
        )
        ex = Ex("calc_salary", user)
        r = ex.tool("CALCULATE", f"{m} * 12")
        a = f"Saal ka ₹{r}." if lang == "hi" else f"₹{r} per year."
    else:
        d = pick([26, 30])
        days = R.randint(5, d)
        user, _ = fill(
            by_lang(
                lang,
                ["staff ki salary ₹{m}, {d} din ka mahina, usne {x} din kaam kiya, kitna du?"],
                ["salary ₹{m} for {d} days, worked {x} days. how much to pay?"],
            ),
            lang,
            m=m,
            d=d,
            x=days,
        )
        ex = Ex("calc_salary", user)
        r = ex.tool("CALCULATE", f"{m} / {d} * {days}")
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
    lang = L()
    c = pick(CONVERSIONS)
    v = pick([R.randint(1, 500), round(R.uniform(1, 100), 1)])
    user, _ = fill(c[0] if lang == "hi" else c[1], lang, v=v)
    ex = Ex("calc_convert", user)
    r = ex.tool("CALCULATE", c[2].format(v=v))
    return ex.done(c[4].format(v=v, r=r))


def calc_area():
    lang = L()
    l = R.randint(6, 80)
    w = R.randint(5, 60)
    user, _ = fill(
        by_lang(
            lang,
            ["{l} x {w} feet ki dukaan ka area?", "{l} ft by {w} ft, kitna sq ft?"],
            ["area of a {l} x {w} ft shop?", "{l}ft x {w}ft in sq ft and sq m?"],
        ),
        lang,
        l=l,
        w=w,
    )
    ex = Ex("calc_area", user)
    a1 = ex.tool("CALCULATE", f"{l} * {w}")
    if R.random() < 0.5:
        a2 = ex.tool("CALCULATE", f"{a1} * 0.0929")
        a = f"{a1} sq ft ({a2} sq m)."
    else:
        a = f"{a1} sq ft."
    return ex.done(a)


def calc_rent_per_sqft():
    lang = L()
    rent = pick([15000, 25000, 40000, 65000, 90000])
    area = pick([200, 350, 500, 800, 1200])
    user, _ = fill(
        by_lang(lang, ["{a} sq ft ka rent ₹{r}, per sq ft kitna?"], ["rent ₹{r} for {a} sq ft. per sq ft rate?"]),
        lang,
        a=area,
        r=rent,
    )
    ex = Ex("calc_rent", user)
    x = ex.tool("CALCULATE", f"{rent} / {area}")
    return ex.done(f"₹{x} per sq ft." if lang == "en" else f"₹{x} per sq ft pad raha hai.")


def calc_percent_of():
    lang = L()
    v = pick([200, 500, 1200, 2500, 8000, 15000, 40000, 120000])
    p = pick([2, 3, 5, 8, 12.5, 15, 18, 20, 33, 75])
    user, _ = fill(
        by_lang(lang, ["{v} ka {p}% kitna?", "{p} percent of {v} batao"], ["what's {p}% of {v}?", "{p}% of {v}"]),
        lang,
        v=v,
        p=p,
    )
    ex = Ex("calc_pct", user)
    r = ex.tool("CALCULATE", f"{v} * {p} / 100")
    return ex.done(f"{p}% of {v} = {r}." if lang == "en" else f"{v} ka {p}% = {r}.")


def calc_average():
    lang = L()
    xs = [R.randrange(500, 25000, 10) for _ in range(R.randint(3, 6))]
    s = ", ".join(str(x) for x in xs)
    user, _ = fill(
        by_lang(
            lang,
            ["is hafte ki sale: {s}. average kitna?", "average nikalo: {s}"],
            ["daily sales were {s}. average?", "average of {s}?"],
        ),
        lang,
        s=s,
    )
    ex = Ex("calc_avg", user)
    r = ex.tool("CALCULATE", f"({' + '.join(str(x) for x in xs)}) / {len(xs)}")
    return ex.done(f"Average ₹{r}." if lang == "hi" else f"The average is {r}.")


def calc_basic():
    lang = L()
    op = pick(["+", "-", "*", "/"])
    a, b = R.randint(2, 99999), R.randint(2, 9999)
    if op == "*":
        a, b = R.randint(2, 9999), R.randint(2, 999)
    if op == "-" and b > a:
        a, b = b, a
    sym = {
        "+": pick(["+", "plus", "aur"]),
        "-": pick(["-", "minus"]),
        "*": pick(["*", "x", "×", "into", "times"]),
        "/": pick(["/", "÷", "divided by"]),
    }[op]
    if sym == "aur" and lang == "en":
        sym = "+"
    user, _ = fill(
        pick(
            [
                "{a} {s} {b}",
                "{a} {s} {b} kitna?",
                "what's {a} {s} {b}?",
                "{a}{s}{b}=?",
                "calculate {a} {s} {b}",
                "{a} {s} {b} batao",
            ]
        ),
        lang,
        a=a,
        b=b,
        s=sym,
    )
    ex = Ex("calc_basic", user)
    r = ex.tool("CALCULATE", f"{a} {op} {b}")
    return ex.done(pick([f"{r}", f"= {r}", f"{a} {op} {b} = {r}", f"The answer is {r}."]))


def calc_sde():
    lang = L()
    kind = R.random()
    if kind < 0.4:
        n = pick([1000, 50000, 1000000, 10000000, 100000000])
        b = pick([4, 8, 16, 64])
        user, _ = fill(
            pick(["memory for {n} items of {b} bytes in MB?", "{n} records × {b} bytes kitne MB?"]), lang, n=n, b=b
        )
        ex = Ex("calc_sde", user)
        r = ex.tool("CALCULATE", f"{n} * {b} / 1048576")
        return ex.done(f"About {r} MB.")
    if kind < 0.7:
        rps = pick([50, 120, 300, 1500, 8000])
        user, _ = fill(
            pick(["{r} requests per second means how many per day?", "{r} rps ka daily kitna hua?"]), lang, r=rps
        )
        ex = Ex("calc_sde", user)
        x = ex.tool("CALCULATE", f"{rps} * 60 * 60 * 24")
        return ex.done(f"{x} requests per day.")
    gb = pick([5, 20, 100, 500])
    mbps = pick([10, 50, 100, 1000])
    user, _ = fill(
        pick(["how long to download {g} GB at {m} Mbps in minutes?", "{g} GB file {m} Mbps pe kitne minute?"]),
        lang,
        g=gb,
        m=mbps,
    )
    ex = Ex("calc_sde", user)
    x = ex.tool("CALCULATE", f"{gb} * 1024 * 8 / {mbps} / 60")
    return ex.done(f"About {x} minutes.")


def calc_staff_total():
    lang = L()
    n = pick([2, 3, 4, 5, 6, 10, 12])
    s = R.randrange(6000, 40000, 500)
    nd = qty_text(n, lang)
    if R.random() < 0.6:
        user, _ = fill(
            by_lang(
                lang,
                [
                    "{n} staff hai, har ek ko {s} salary, total kitna dena hai",
                    "{n} karigar, sabko {s} mahina, total salary?",
                    "{n} logo ko {s} {s2} dena hai, kul kitna?",
                ],
                ["{n} employees at {s} each, total salary?", "I pay {n} staff {s} each. total?"],
            ),
            lang,
            n=nd,
            s=s,
            s2="each",
        )
        ex = Ex("calc_staff", user)
        t = ex.tool("CALCULATE", f"{n} * {s}")
        return ex.done(f"Total ₹{t} dena hoga." if lang == "hi" else f"Total salary: ₹{t}.")
    d = pick([4, 6, 10, 15, 26])
    w = pick([350, 450, 500, 600, 750, 900])
    user, _ = fill(
        by_lang(
            lang,
            ["{n} mazdoor, {d} din, {w} rupaye roz — kitna hua?", "{n} helper {d} din ke liye, {w} per day, total?"],
            ["{n} workers for {d} days at ₹{w}/day — total?"],
        ),
        lang,
        n=nd,
        d=d,
        w=w,
    )
    ex = Ex("calc_staff", user)
    t = ex.tool("CALCULATE", f"{n} * {d} * {w}")
    return ex.done(f"Kul ₹{t}." if lang == "hi" else f"Total: ₹{t}.")


def calc_lakh():
    lang = L()
    L_ = pick([1, 1.5, 2, 2.5, 3, 5, 10, 12, 25])
    p = pick([2, 5, 8, 10, 12, 18])
    val = int(L_ * 100000)
    user, _ = fill(
        by_lang(
            lang,
            ["{l} lakh ka {p}% kitna?", "{l} lakh pe {p}% kitna banta hai", "{l} lakh ka {p} percent"],
            ["what's {p}% of {l} lakh?", "{p}% of {l} lakh rupees?"],
        ),
        lang,
        l=f"{L_:g}",
        p=p,
    )
    ex = Ex("calc_lakh", user)
    r = ex.tool("CALCULATE", f"{val} * {p} / 100")
    return ex.done(f"₹{r}." if lang == "en" else f"₹{r} hota hai.")


def calc_dozen():
    lang = L()
    d = pick([2, 3, 4, 5, 10])
    pr = pick([36, 48, 60, 72, 84, 96, 120])
    if R.random() < 0.5:
        user, _ = fill(
            by_lang(
                lang,
                ["{d} dozen kele, {pr} rupaye dozen, kitna hua?", "{d} darjan ande {pr} ke hisaab se?"],
                ["{d} dozen eggs at {pr} per dozen?", "{d} dozen bananas, ₹{pr} a dozen — total?"],
            ),
            lang,
            d=d,
            pr=pr,
        )
        ex = Ex("calc_dozen", user)
        r = ex.tool("CALCULATE", f"{d} * {pr}")
    else:
        k = pick([3, 5, 7, 8, 9])
        user, _ = fill(
            by_lang(lang, ["ek dozen {pr} ka hai, {k} piece kitne ke?"], ["a dozen costs {pr}, how much for {k}?"]),
            lang,
            pr=pr,
            k=k,
        )
        ex = Ex("calc_dozen", user)
        r = ex.tool("CALCULATE", f"{pr} / 12 * {k}")
    return ex.done(f"₹{r}." if lang == "en" else f"₹{r} hua.")


CALC_FAMILIES = [
    (calc_staff_total, 2),
    (calc_lakh, 1),
    (calc_dozen, 1),
    (calc_gst_add, 3),
    (calc_gst_inclusive, 2),
    (calc_cgst_sgst, 1),
    (calc_discount, 3),
    (calc_discount_gst, 2),
    (calc_margin, 2),
    (calc_markup, 1),
    (calc_emi, 2),
    (calc_simple_interest, 1),
    (calc_split, 2),
    (calc_change, 2),
    (calc_per_unit, 2),
    (calc_growth, 1),
    (calc_daily_target, 1),
    (calc_salary, 2),
    (calc_convert, 2),
    (calc_area, 1),
    (calc_rent_per_sqft, 1),
    (calc_percent_of, 2),
    (calc_average, 1),
    (calc_basic, 3),
    (calc_sde, 1),
]


def fam_calc():
    new_now()
    f = R.choices([f for f, _ in CALC_FAMILIES], [w for _, w in CALC_FAMILIES])[0]
    return f()


# ═══════════════════════════════════════════════════════════════
# FAMILY: tool errors (CALCULATE / DATE)
# ═══════════════════════════════════════════════════════════════
def fam_tool_error():
    new_now()
    lang = L()
    if R.random() < 0.5:
        n = R.randint(10, 9999)
        user, _ = fill(
            by_lang(lang, ["{n} ko 0 se divide karo", "{n} / 0 kitna hai?"], ["{n} / 0", "what's {n} divided by 0?"]),
            lang,
            n=n,
        )
        ex = Ex("calc_error", user)
        ex.tool("CALCULATE", f"{n} / 0")
        return ex.done(
            "Zero se divide nahi kar sakte — ye undefined hota hai."
            if lang == "hi"
            else "Division by zero is undefined, so there's no answer."
        )
    bad = pick(["32 March", "31/02/2027", "next to next somday", "45 June 2027"])
    user, _ = fill(by_lang(lang, ["{d} ko kaun sa din hai?"], ["what day is {d}?"]), lang, d=bad)
    ex = Ex("date_error", user)
    r = ex.tool("DATE", bad)
    if not r.startswith("Error"):
        raise Skip("date parsed")
    return ex.done(
        "Ye date samajh nahi aayi. Date ko 15 August 2027 jaise format me bhejiye."
        if lang == "hi"
        else "I couldn't read that date. Could you send it like 15 August 2027?",
        allow={"15", "2027"},
    )
