"""No-tool families: greetings, order status, complaints, handoff, concepts, advice.

These teach Milo when NOT to call a tool - about a third of the dataset.
"""

from milo.data import paraphrase_banks as B
from milo.data.generator.core import (
    Ex,
    L,
    R,
    Skip,
    finish,
    is_hi,
    leak_filter,
    noisy,
    nums,
    pick,
)


# ═══════════════════════════════════════════════════════════════
# NO-TOOL FAMILIES
# ═══════════════════════════════════════════════════════════════
def nt(family, user, answer, allow=()):
    user = finish(noisy(user.replace("{", "(").replace("}", ")"), "hi" if is_hi(user) else "en"))
    ex = Ex(family, user)
    return ex.done(answer, allow=allow)


def fam_greeting():
    lang = L()
    q = pick(
        {
            "hi": [
                "namaste",
                "namaste ji",
                "hello bhaiya",
                "hi",
                "hello",
                "ram ram",
                "salaam",
                "good morning ji",
                "suniye",
                "koi hai?",
                "hello, shop khuli hai?",
                "hi milo",
                "bhaiya hello",
            ],
            "en": ["hi", "hello", "hey there", "good morning", "good evening", "hi milo", "anyone there?", "hello?"],
        }[lang]
    )
    if "khuli" in q:
        raise Skip("needs MQ")
    a = pick(
        {
            "hi": [
                "Namaste! Batayiye, kya madad kar sakta hoon?",
                "Ji, boliye! Kya chahiye aapko?",
                "Hello! Main Milo hoon, shop ka assistant. Kya help karun?",
                "Namaste ji 🙏 Kya dekhna hai aapko?",
            ],
            "en": [
                "Hi! How can I help you today?",
                "Hello! What can I do for you?",
                "Hi there — I'm Milo, the shop's assistant. What do you need?",
                "Hello! Looking for something?",
            ],
        }[lang]
    )
    q = bank(q, B.GREETING, lang)
    return nt("greeting", finish(q), a)


def fam_thanks_bye():
    lang = L()
    q = pick(
        {
            "hi": [
                "thank you",
                "dhanyavaad",
                "shukriya",
                "ok thanks",
                "theek hai",
                "acha",
                "ok ji",
                "bye",
                "chalo bye",
                "thik hai bhaiya",
                "👍",
                "done",
                "badhiya",
            ],
            "en": ["thanks!", "thank you", "ok", "great, thanks", "got it", "bye", "cool", "perfect", "👍", "awesome"],
        }[lang]
    )
    a = pick(
        {
            "hi": [
                "Aapka swagat hai! Aur kuch chahiye to message kijiye.",
                "Dhanyavaad! 🙏",
                "Ji, koi baat nahi. Aur kuch?",
                "Theek hai ji. Kuch aur chahiye to batayein.",
            ],
            "en": [
                "You're welcome! Message anytime.",
                "Happy to help!",
                "Anytime — let me know if you need anything else.",
                "Great! Have a nice day.",
            ],
        }[lang]
    )
    q = bank(q, B.THANKS, lang)
    return nt("thanks", finish(q), a)


def fam_about():
    lang = L()
    q = pick(
        {
            "hi": ["tum kaun ho?", "aap insaan ho ya bot?", "kya kya kar sakte ho?", "milo kaun hai?", "aap AI ho?"],
            "en": ["who are you?", "are you a bot?", "what can you do?", "are you human?", "who made you?"],
        }[lang]
    )
    a = pick(
        {
            "hi": [
                "Main Milo hoon, is shop ka AI assistant. Product ka rate, stock, timing, delivery aur hisaab-kitaab me madad kar sakta hoon.",
                "Main ek AI assistant (Milo) hoon. Price, availability, delivery, GST calculation — sab me help karta hoon. Order confirm shop owner karte hain.",
            ],
            "en": [
                "I'm Milo, this shop's AI assistant. I can check prices, stock, timings and delivery, and do quick calculations.",
                "I'm an AI assistant called Milo. I help with product prices, availability, store info and calculations; the owner confirms orders.",
            ],
        }[lang]
    )
    q = bank(q, B.ABOUT, lang)
    return nt("about", finish(q), a)


def fam_clarify():
    lang = L()
    k = R.random()
    if k < 0.4:
        q = pick(
            {
                "hi": ["rate batao", "kitne ka hai?", "price kya hai", "ye kitne ka milega", "iska rate?", "daam?"],
                "en": ["how much?", "price?", "what's the rate?", "how much is it?", "cost?"],
            }[lang]
        )
        q = bank(q, B.CLARIFY_PRICE, lang)
        a = pick(
            {
                "hi": [
                    "Kaunse product ka rate chahiye? Naam bata dijiye, main check kar deta hoon.",
                    "Kis item ka price chahiye? Product ka naam aur size bhejiye.",
                ],
                "en": [
                    "Which product do you mean? Send me the name and size and I'll check.",
                    "Sure — which item are you asking about?",
                ],
            }[lang]
        )
    elif k < 0.7:
        q = pick(
            {
                "hi": ["order karna hai", "kuch saman mangwana hai", "mujhe order dena hai", "list bhejun?"],
                "en": ["I want to place an order", "can I order?", "I'd like to buy some things"],
            }[lang]
        )
        q = bank(q, B.CLARIFY_ORDER, lang)
        a = pick(
            {
                "hi": [
                    "Ji zaroor! Items aur quantity ki list bhejiye, saath me naam aur address. Shop owner confirm karke bata denge.",
                    "Bilkul. Kya-kya chahiye, kitna chahiye aur kahan bhejna hai — ye bata dijiye.",
                ],
                "en": [
                    "Sure! Send the items with quantities, plus your name and address. The owner will confirm your order.",
                    "Of course — what items and how many? Please also share your delivery address.",
                ],
            }[lang]
        )
    else:
        q = pick(
            {
                "hi": ["GST kitna lagega?", "isme GST kitna hoga", "tax kitna lagega"],
                "en": ["how much GST will apply?", "what's the GST?"],
            }[lang]
        )
        a = pick(
            {
                "hi": [
                    "Kis amount aur kis rate pe? Zyadatar cheezon pe 5% ya 18% GST lagta hai — amount aur rate bata dijiye, main calculate kar dunga.",
                    "Amount kitna hai aur GST rate kya hai? Batayiye, main turant calculate kar deta hoon.",
                ],
                "en": [
                    "On what amount and at what rate? Most goods are taxed at 5% or 18% — give me the amount and rate and I'll calculate it.",
                    "Tell me the amount and the GST rate and I'll work it out.",
                ],
            }[lang]
        )
        return nt("clarify", finish(q), a, allow={"18"})
    return nt("clarify", finish(q), a)


def fam_order_status():
    lang = L()
    oid = R.randint(1000, 99999)
    with_id = R.random() < 0.4
    q = pick(
        {
            "hi": ["mera order kahan hai?", "order kab aayega?", "order abhi tak nahi aaya", "mera parcel kidhar hai"],
            "en": ["where is my order?", "my order hasn't arrived", "order status?", "when will my order come?"],
        }[lang]
    )
    q = bank(q, B.ORDER_STATUS, lang)
    if with_id:
        q += pick([f" order no {oid}", f" (order #{oid})", f", order id {oid}"])
    if with_id:
        a = pick(
            {
                "hi": [
                    f"Order #{oid} ka status main yahan se check nahi kar sakta. Shop owner ko bata diya jayega, wo jaldi update denge.",
                    f"Order {oid} note kar liya. Live status mere paas nahi hota — owner aapko jaldi confirm karenge.",
                ],
                "en": [
                    f"I can't see live status for order #{oid} from here. The owner will check and update you shortly.",
                    f"Noted order {oid}. I don't have live tracking, but the owner will get back to you soon.",
                ],
            }[lang]
        )
    else:
        a = pick(
            {
                "hi": [
                    "Apna order number ya naam bhej dijiye. Live status mere paas nahi hota, owner check karke batayenge.",
                    "Sorry for the delay! Order ID bhejiye, shop owner turant check karenge.",
                ],
                "en": [
                    "Please share your order number or name. I don't have live tracking, but the owner will check and update you.",
                    "Sorry about the wait! Send your order ID and the owner will look into it.",
                ],
            }[lang]
        )
    return nt("order_status", finish(q), a)


def fam_place_order():
    lang = L()
    qty = R.randint(1, 5)
    item = pick(
        ["atta 10kg", "2 dozen kele", "black forest cake 1kg", "charger", "paneer 1kg", "notebook", "kurti M size"]
    )
    q = pick(
        {
            "hi": [f"{qty} {item} bhej do", f"{item} {qty} pack kar do", f"{qty} {item} order kar do please"],
            "en": [f"please send {qty} {item}", f"I'll take {qty} {item}", f"book {qty} {item} for me"],
        }[lang]
    )
    a = pick(
        {
            "hi": [
                "Ji, note kar liya! Apna naam aur delivery address bhej dijiye, shop owner confirm karke bata denge.",
                "Theek hai! Address aur payment mode (UPI/cash) bata dijiye, order confirm ho jayega.",
            ],
            "en": [
                "Got it! Please send your name and delivery address — the owner will confirm the order.",
                "Noted! Share your address and payment mode (UPI or cash) and we'll confirm.",
            ],
        }[lang]
    )
    return nt("place_order", finish(q), a)


def fam_complaint():
    lang = L()
    q = pick(
        {
            "hi": [
                "saman kharab nikla",
                "packet phata hua aaya",
                "galat item aa gaya",
                "cake toot gaya tha",
                "quality achhi nahi thi",
                "charger kaam nahi kar raha",
            ],
            "en": ["the product was damaged", "I got the wrong item", "the packet was torn", "it stopped working"],
        }[lang]
    )
    a = pick(
        {
            "hi": [
                "Bahut sorry iske liye! Item ki photo aur order details bhej dijiye, owner turant replacement dekh lenge.",
                "Maaf kijiye 🙏 Photo bhejiye aur bataiye kab order kiya tha — hum jaldi solve karenge.",
            ],
            "en": [
                "Really sorry about that! Please send a photo and your order details — the owner will arrange a fix.",
                "Apologies! Share a photo and when you ordered, and we'll sort it out quickly.",
            ],
        }[lang]
    )
    q = bank(q, B.COMPLAINT, lang)
    return nt("complaint", finish(q), a)


def fam_human():
    lang = L()
    q = pick(
        {
            "hi": [
                "owner se baat karni hai",
                "kisi insaan se baat karao",
                "call kar sakte ho?",
                "manager se baat karwa do",
            ],
            "en": ["can I talk to a human?", "I want to speak to the owner", "can someone call me?"],
        }[lang]
    )
    a = pick(
        {
            "hi": [
                "Ji, maine owner ko bata diya hai. Wo jaldi aapse contact karenge.",
                "Bilkul, shop owner ko message forward kar diya hai — wo jaldi reply karenge.",
            ],
            "en": [
                "Sure — I've flagged this for the owner, who'll get in touch soon.",
                "Of course. I've passed your message to the owner; they'll reply shortly.",
            ],
        }[lang]
    )
    q = bank(q, B.HUMAN, lang)
    return nt("human_handoff", finish(q), a)


def fam_payment_done():
    lang = L()
    amt = pick([249, 499, 649, 1180, 1999, 2450])
    q = pick(
        {
            "hi": [f"₹{amt} UPI kar diya", f"payment ho gaya ₹{amt}", f"{amt} bhej diye gpay se"],
            "en": [f"paid ₹{amt} via UPI", f"sent ₹{amt} on gpay", f"payment of ₹{amt} done"],
        }[lang]
    )
    if R.random() < 0.6:
        q = pick(leak_filter(B.PAYMENT_DONE[lang])).format(amt=amt)
    a = pick(
        {
            "hi": [
                f"Dhanyavaad! ₹{amt} ka payment owner confirm karke aapko update kar denge.",
                f"Thank you ji! ₹{amt} received hone pe confirmation bhej denge.",
            ],
            "en": [
                f"Thank you! The owner will confirm the ₹{amt} payment and update you.",
                f"Thanks! We'll confirm once ₹{amt} reflects in our account.",
            ],
        }[lang]
    )
    return nt("payment_ack", finish(q), a)


DECLINE_ANSWERS = {
    "medical": {
        "hi": "Main dawai ki salah nahi de sakta. Kripya doctor ya pharmacist se baat kijiye.",
        "en": "I can't give medical advice — please consult a doctor or pharmacist.",
    },
    "loan": {
        "hi": "Sorry, hum loan ya udhaar nahi dete. Is shop ke products ya services me madad kar sakta hoon.",
        "en": "Sorry, we don't offer loans or credit. I can help with this shop's products and services.",
    },
    "predict": {
        "hi": "Ye main predict nahi kar sakta 😄 Shop se related kuch chahiye to batayein.",
        "en": "I can't predict that 😄 Anything from the shop I can help with?",
    },
    "personal": {
        "hi": "Sorry, main kisi ki personal details share nahi kar sakta.",
        "en": "Sorry, I can't share anyone's personal details.",
    },
    "homework": {
        "hi": "Main pura kaam nahi likhunga, par topic samjhane me madad kar sakta hoon. Kaunsa topic hai?",
        "en": "I won't write the whole thing, but I can help you understand the topic. What's it about?",
    },
}
DECLINE_BASE = {
    "medical": {
        "hi": [
            "mujhe bukhar hai, kaunsi dawai lun?",
            "khansi ke liye kaunsi tablet lu?",
            "BP ki kaunsi medicine achhi hai?",
        ],
        "en": ["I have fever, which medicine should I take?"],
    },
    "loan": {"hi": ["mujhe loan chahiye"], "en": ["can you give me a loan?"]},
    "predict": {"hi": ["kal match kaun jeetega?"], "en": ["who will win tomorrow's match?"]},
    "personal": {"hi": ["kisi ka number do"], "en": ["give me someone's phone number"]},
    "homework": {"hi": ["mera homework kar do"], "en": ["write my college assignment"]},
}


def fam_decline():
    lang = L()
    cat = pick(list(DECLINE_ANSWERS))
    pool = leak_filter(DECLINE_BASE[cat][lang] + B.DECLINE[cat][lang])
    return nt("decline", finish(pick(pool)), DECLINE_ANSWERS[cat][lang])


CONCEPTS = [
    (
        ["What is a primary key?", "primary key kya hota hai?"],
        "A primary key is a column (or set of columns) that uniquely identifies each row in a table; it can't be null or repeated.",
    ),
    (
        ["What is an environment variable?", "env variable kya hota hai?"],
        "An environment variable is a named value set outside your code (like API keys or config) that your program reads at runtime.",
    ),
    (
        ["What's the difference between a list and a tuple in Python?", "python list aur tuple me farak?"],
        "Lists are mutable — you can change them; tuples are immutable. Use tuples for fixed data and lists for collections that change.",
    ),
    (
        ["What does async/await do?", "async await kya karta hai?"],
        "async/await lets code wait for slow work like network calls without blocking everything else, while still reading top to bottom.",
    ),
    (
        ["What is a pull request?", "PR kya hota hai github pe?"],
        "A pull request asks to merge your branch into another; teammates review the changes and comment before it's merged.",
    ),
    (
        ["What is caching?", "caching kya hai?"],
        "Caching stores the result of an expensive operation so the next request can be served quickly from memory.",
    ),
    (
        ["What is an API key?", "API key kya hoti hai?"],
        "An API key is a secret token that identifies your app to an API. Keep it out of your code repo and rotate it if it leaks.",
    ),
    (
        ["How do I prepare for my first developer interview?", "pehle dev interview ki tayari kaise karu?"],
        "Revise DSA basics, be ready to explain your own projects in depth, practise talking through your approach out loud, and ask good questions at the end.",
    ),
    (
        ["Should I use SQL or NoSQL?", "SQL lu ya NoSQL?"],
        "Start with SQL for structured data and relationships; choose NoSQL when data is document-shaped or you need very flexible schemas at scale.",
    ),
    (
        ["What is a webhook?", "webhook kya hota hai?"],
        "A webhook is a URL your app exposes so another service can call it automatically when an event happens, like a payment succeeding.",
    ),
    (
        ["How do I keep customers coming back?", "customer wapas kaise aaye?"],
        "Be reliable on quality and delivery, remember regulars, reply quickly on WhatsApp, and offer a small reward for repeat orders.",
    ),
    (
        ["How do I handle a customer who is angry?", "gussa customer ko kaise handle karu?"],
        "Listen first, apologise for the experience, fix what you can quickly, and follow up so they know it's resolved.",
    ),
    (
        ["Should I accept UPI payments?", "UPI lena chahiye kya?"],
        "Yes — most customers prefer UPI, payments settle directly to your bank, and a static QR code is free to set up.",
    ),
    (
        ["How do I manage stock better?", "stock better kaise manage karu?"],
        "Track what sells fastest, reorder before items run out, keep a simple daily stock sheet, and clear slow-moving items with offers.",
    ),
    (
        ["Is it worth getting a GST registration?", "GST registration lena chahiye?"],
        "It depends on your turnover and who you sell to — registration lets you claim input tax credit and sell to businesses. A CA can confirm if it's required for you.",
    ),
    (
        ["What does API stand for?", "API ka full form?", "API matlab kya?"],
        "API stands for Application Programming Interface — a defined way for one piece of software to talk to another.",
    ),
    (
        ["Difference between == and === in JavaScript?", "JS me == aur === me kya farak hai?"],
        "== compares after type conversion ('5' == 5 is true); === also checks the type ('5' === 5 is false). Prefer ===.",
    ),
    (
        ["What is recursion?", "recursion kya hota hai?"],
        "Recursion is when a function calls itself. It needs a base case to stop and a recursive case that makes the problem smaller.",
    ),
    (
        ["GET vs POST?", "GET aur POST me difference?"],
        "GET fetches data and puts parameters in the URL; POST sends data in the request body and is used to create or change things.",
    ),
    (
        ["What is JSON?", "JSON kya hai?"],
        "JSON (JavaScript Object Notation) is a lightweight text format for data using key-value pairs and arrays — the standard for APIs.",
    ),
    (
        ["HTTP vs HTTPS?", "http aur https me kya farak hai?"],
        "HTTPS is HTTP over TLS: the connection is encrypted, so data can't be read or tampered with in transit.",
    ),
    (
        ["What is OOP?", "OOP kya hota hai?"],
        "Object-Oriented Programming organises code into objects that bundle data and behaviour; classes are their blueprints.",
    ),
    (
        ["What's a database index?", "index kya hota hai database me?"],
        "An index is a lookup structure that speeds up reads on a column, at the cost of extra storage and slower writes.",
    ),
    (
        ["Should I learn Python or JavaScript first?", "pehle python seekhu ya javascript?"],
        "Pick by goal: Python for data, AI and automation; JavaScript for websites and web apps. Both are good first languages.",
    ),
    (
        ["How do I get better at DSA?", "DSA kaise improve karu?"],
        "Practise daily, start with arrays and strings, learn patterns (two pointers, sliding window, BFS/DFS), and revisit problems you got wrong.",
    ),
    (
        ["What's the difference between git merge and rebase?", "merge vs rebase?"],
        "Merge joins branches with a merge commit and keeps history as it happened; rebase replays your commits on top for a linear history.",
    ),
    (
        ["What is a REST API in simple words?", "REST API simple me samjhao"],
        "A REST API lets apps request and change data over HTTP using URLs for resources and methods like GET, POST, PUT and DELETE.",
    ),
    (
        ["Is SQL still worth learning?", "SQL seekhna chahiye?"],
        "Yes — almost every product stores data in a relational database, and SQL is how you query it.",
    ),
    (
        ["What is a virtual environment in Python?", "python venv kya hai?"],
        "A virtual environment is an isolated folder with its own Python packages, so projects don't clash over versions.",
    ),
    (
        ["How do I get my first customers?", "pehle customer kaise laaun?"],
        "Start with people you know and nearby shops, give a great first experience, and ask happy customers to refer others.",
    ),
    (
        ["Should I offer discounts to new customers?", "naye customers ko discount du?"],
        "A small first-order offer can help, but keep it limited so your margin stays healthy and customers come back for quality.",
    ),
    (
        ["How can I get more customers on WhatsApp?", "WhatsApp pe zyada customers kaise laun?"],
        "Share your catalogue as a WhatsApp Business catalog, post daily offers on Status, reply fast, and ask happy customers to share your number.",
    ),
    (
        ["Is 18% GST high?", "18% GST zyada hai kya?"],
        "18% is India's standard rate for most goods and services; essentials usually sit at 5% or nil. Whether it feels high depends on your product's margins.",
    ),
    (
        ["What is GST in simple words?", "GST simple me samjhao"],
        "GST is a single indirect tax added to the price of goods and services in India; businesses collect it from customers and pay it to the government.",
    ),
    (
        ["How should I price my product?", "apne product ka price kaise rakhu?"],
        "Start from your full cost, add the margin you need, then check competitor prices and what customers are willing to pay.",
    ),
]


def fam_concept():
    qs, a = pick(CONCEPTS)
    q = pick(qs)
    return nt("concept", q, a, allow=nums(a))


# ── paraphrase banks merged into the pools above ──
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
