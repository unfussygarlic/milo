"""DATE family: today / time, festivals, delivery windows, working-day counts."""

import datetime as dt

from milo.data.generator.core import Ex, L, R, by_lang, fill, new_now, pick


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
    "diwali": ["Diwali", "diwali", "Deepavali"],
    "holi": ["Holi", "holi"],
    "raksha bandhan": ["Raksha Bandhan", "rakhi"],
    "dussehra": ["Dussehra", "dussehra", "Dasara"],
    "navratri": ["Navratri", "navratri"],
    "dhanteras": ["Dhanteras", "dhanteras"],
    "janmashtami": ["Janmashtami", "janmashtami"],
    "ganesh chaturthi": ["Ganesh Chaturthi", "ganpati"],
    "christmas": ["Christmas", "christmas"],
    "new year": ["New Year", "new year"],
    "republic day": ["Republic Day", "26 January"],
    "independence day": ["Independence Day", "15 August"],
    "makar sankranti": ["Makar Sankranti", "sankranti"],
    "maha shivaratri": ["Maha Shivaratri", "shivratri"],
    "ram navami": ["Ram Navami", "ram navami"],
    "bhai dooj": ["Bhai Dooj", "bhai dooj"],
    "karwa chauth": ["Karwa Chauth", "karva chauth"],
}


def fam_date():
    now = new_now()
    lang = L()
    k = R.random()
    if k < 0.18:
        user, _ = fill(
            by_lang(
                lang,
                ["aaj kya date hai?", "aaj kaun sa din hai", "date kya hai aaj", "aaj ki date batao"],
                ["what's today's date?", "what day is it today?", "today's date?", "date?"],
            ),
            lang,
        )
        ex = Ex("date_today", user)
        r = ex.tool("DATE", "today")
        return ex.done(f"Aaj {r} hai." if lang == "hi" else pick([f"Today is {r}.", f"It's {r}."]))
    if k < 0.26:
        user, _ = fill(
            by_lang(
                lang,
                [
                    "abhi kitne baje hain?",
                    "time kya hua?",
                    "time batao",
                    "kitne baj gaye?",
                    "abhi ka time kya hai",
                    "time kitna ho raha hai?",
                    "ghadi me kya time hai",
                ],
                ["what time is it?", "current time?", "what's the time now?", "time please"],
            ),
            lang,
        )
        ex = Ex("date_time", user)
        r = ex.tool("DATE", "time")
        return ex.done(f"Abhi {r} hai." if lang == "hi" else f"It's {r}.")
    if k < 0.40:
        d = now.date() + dt.timedelta(days=R.randint(-400, 400))
        ds = _date_text(d)
        user, _ = fill(
            by_lang(
                lang,
                ["{d} ko kaun sa din hai?", "{d} kaunsa din tha?", "{d} ko konsa vaar hai"],
                ["what day is {d}?", "which weekday was {d}?", "{d} — what day of the week?"],
            ),
            lang,
            d=ds,
        )
        ex = Ex("date_weekday", user)
        r = ex.tool("DATE", ds)
        return ex.done(f"{r}." if lang == "en" else f"{ds} ko {r.split(',')[0]} hai.")
    if k < 0.55:
        d = now.date() + dt.timedelta(days=R.randint(2, 300))
        ds = _date_text(d)
        user, _ = fill(
            by_lang(
                lang,
                ["{d} me kitne din bache hain?", "{d} tak kitne din hain?", "{d} aane me kitne din?"],
                ["how many days until {d}?", "days left till {d}?"],
            ),
            lang,
            d=ds,
        )
        ex = Ex("date_until", user)
        r = ex.tool("DATE", f"days until {ds}")
        n = r.split(" ")[0]
        return ex.done(f"{n} din bache hain ({r.split('until ')[1]})." if lang == "hi" else f"{r}.")
    if k < 0.65:
        n = pick([3, 5, 7, 10, 15, 21, 30, 45, 90])
        user, _ = fill(
            by_lang(
                lang,
                ["aaj se {n} din baad kaun si date hogi?", "{n} din baad kya date hai?"],
                ["what's the date {n} days from today?", "date after {n} days?"],
            ),
            lang,
            n=n,
        )
        ex = Ex("date_plus", user)
        r = ex.tool("DATE", f"today + {n} days")
        return ex.done(f"{n} din baad {r} hogi." if lang == "hi" else f"{n} days from today is {r}.")
    if k < 0.72:
        d1 = now.date() - dt.timedelta(days=R.randint(10, 500))
        d2 = d1 + dt.timedelta(days=R.randint(5, 400))
        a, b = _date_text(d1), _date_text(d2)
        user, _ = fill(
            by_lang(lang, ["{a} se {b} tak kitne din?"], ["how many days between {a} and {b}?"]), lang, a=a, b=b
        )
        ex = Ex("date_between", user)
        r = ex.tool("DATE", f"days between {a} and {b}")
        return ex.done(f"{r.split(' between')[0]} hain." if lang == "hi" else f"{r}.")
    if k < 0.78:
        d = now.date() + dt.timedelta(days=R.randint(3, 40))
        ds = _date_text(d)
        user, _ = fill(
            by_lang(lang, ["{d} tak kitne working days hain?"], ["how many working days until {d}?"]), lang, d=ds
        )
        ex = Ex("date_working", user)
        r = ex.tool("DATE", f"working days until {ds}")
        return ex.done(f"{r}." if lang == "en" else f"{r.split(' (')[0]} hain (Sunday chhod ke).")
    # festivals
    key = pick(list(FESTIVAL_NAMES))
    name = pick(FESTIVAL_NAMES[key])
    if R.random() < 0.6:
        user, _ = fill(
            by_lang(
                lang,
                ["{f} me kitne din bache?", "{f} kitne din me hai?", "{f} aane me kitne din hain?"],
                ["how many days until {f}?", "days left for {f}?"],
            ),
            lang,
            f=name,
        )
        ex = Ex("date_festival", user)
        r = ex.tool("DATE", f"days until {name}")
    else:
        user, _ = fill(
            by_lang(
                lang,
                ["{f} kab hai?", "is saal {f} kis din hai?", "{f} ki date kya hai"],
                ["when is {f}?", "what date is {f}?"],
            ),
            lang,
            f=name,
        )
        ex = Ex("date_festival", user)
        r = ex.tool("DATE", name)
    if r.startswith("Error"):
        a = (
            f"{name} ki exact date abhi mere paas nahi hai. Calendar me check kar lijiye."
            if lang == "hi"
            else f"I don't have the exact date for {name} yet — please check a calendar."
        )
    elif "until" in r:
        n, date = r.split(" until ")
        a = (
            f"{name} {date} ko hai — {n.replace('days', 'din').replace('day', 'din')} baaki."
            if lang == "hi"
            else f"{name} is on {date} — {n} to go."
        )
    elif r.endswith("is today"):
        a = f"{name} aaj hi hai! 🎉" if lang == "hi" else f"{name} is today! 🎉"
    else:
        a = f"{name} {r} ko hai." if lang == "hi" else f"{name} is on {r}."
    return ex.done(a)
