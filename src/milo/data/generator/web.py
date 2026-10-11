"""SEARCH and WEATHER families, plus the MQ_SEARCH-vs-SEARCH routing contrasts.

Results come from the real formatting helpers in :mod:`milo.tools`, with stable
hand-written Wikipedia-style extracts, so the training bytes match runtime bytes.
"""

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
    is_hi,
    leak_filter,
    new_kb,
    new_now,
    pick,
)
from milo.data.generator.store import PRODUCT_T, describe_items, parse_mq

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
    new_now(); new_kb(pairs[topic]); lang = L()
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
    ex.fake("WEATHER", c, T.format_weather(c.title(), temp, cond, hum, wind, rain))
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


# ── paraphrase banks merged into the pools above ──
for _lang in ("hi", "en"):
    for _kind, _by_lang in B.WEATHER.items():
        WEATHER_T[_kind][_lang] += leak_filter(_by_lang[_lang])
    SEARCH_T[_lang] += leak_filter(B.SEARCH[_lang])
