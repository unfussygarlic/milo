"""
Milo — hand-written paraphrase banks (chunk 1)

Extra customer-message templates, merged into the generator's pools. Same slot
conventions as generate_tool_data_v4.py:
    {p} product mention   {q} quantity   {u} unit word   {c} city
    [ ... ]  the span copied into MQ_SEARCH args (must contain the retrieval words)

Rules for anyone adding to this file:
  * Never copy or lightly edit prompts from eval_tools.TEST_CASES. The generator
    drops any template sharing a 4-word run with the test split (see leak_filter),
    but don't rely on that — write fresh.
  * Keep the [span] to the words a customer would actually search with.
"""

# ═══════════════════════════════════════════════════════════════
# STORE: product price / availability (MQ_SEARCH)
# ═══════════════════════════════════════════════════════════════
PRODUCT = {
    "hi": [
        "[{p}] hai?", "[{p}]?", "[{p}] milega kya", "[{p}] rakhte ho?", "[{p}] ka stock hai?",
        "[{p}] available ho to batana", "[{p}] mil jayega?", "[{p}] chahiye tha", "bhaiya [{p}] hai kya dukaan pe",
        "[{p}] ka kitna lagega", "[{p}] kitne tak aata hai", "[{p}] ka rate bolo", "[{p}] ka price kya chal raha",
        "[{p}] kitne rupaye ka", "[{p}] ke kitne paise", "[{p}] kitne me doge", "[{p}] ka bhaav bata do",
        "[{p}] abhi milega?", "aaj [{p}] hai kya", "[{p}] khatam to nahi hua?", "[{p}] stock me ho to bhejna",
        "[{p}] ka kya scene hai", "[{p}] wala item hai kya", "ek [{p}] dekhna tha", "[{p}] ki keemat?",
        "[{p}] kitne ka mil jayega aapke yahan", "[{p}] lena hai, rate?", "[{p}] ka daam kya hai ji",
        "sir [{p}] milta hai yahan?", "[{p}] bechte ho?", "[{p}] ka MRP kya hai", "[{p}] aaya kya naya stock",
        "[{p}] hai ki nahi", "[{p}] ho to bata dena", "kya [{p}] available hai abhi",
    ],
    "en": [
        "[{p}]?", "got [{p}]?", "do you stock [{p}]", "is there [{p}] in stock", "any [{p}] left?",
        "[{p}] available today?", "what does [{p}] cost", "[{p}] rate please", "price for [{p}]",
        "how much do you charge for [{p}]", "can I get [{p}]", "[{p}] — in stock or not?", "need [{p}], available?",
        "looking for [{p}]", "do you carry [{p}]", "is [{p}] there", "what's the price of [{p}] now",
        "[{p}] how much", "hi, [{p}] price?", "you have [{p}]?",
    ],
}

# ═══════════════════════════════════════════════════════════════
# STORE: quantity × price (MQ_SEARCH → CALCULATE)
# ═══════════════════════════════════════════════════════════════
QTY = {
    "hi": [
        "{q} {u} [{p}] bhej do, kitna hua", "[{p}] ke {q} {u} ka hisaab", "[{p}] {q} {u} ka bill banao",
        "{q} {u} [{p}] ka kitna banega", "[{p}] — {q} {u} — total?", "{q} {u} [{p}] pack kar do, total batao",
        "mujhe [{p}] ke {q} {u} chahiye, kitne rupaye", "{q} {u} [{p}] ka hisaab laga do",
        "[{p}] {q} {u} lunga, kitna dena hoga", "{q} {u} [{p}] ka kul kitna",
    ],
    "en": [
        "{q} {u} of [{p}], total please", "what's the bill for {q} {u} [{p}]", "[{p}] — {q} {u}, how much altogether",
        "I'll take {q} {u} [{p}], total?", "{q} {u} [{p}] — what do I pay", "add up {q} {u} [{p}] for me",
    ],
}

# ═══════════════════════════════════════════════════════════════
# STORE: FAQ topics (MQ_SEARCH). [span] must contain the retrieval words.
# ═══════════════════════════════════════════════════════════════
FAQ = {
    "Store hours": {
        "hi": ["dukaan [kab khulti] hai", "subah [kitne baje] aa sakte hai", "raat ko [kitne baje band] karte ho",
               "aaj [chutti] hai kya", "[sunday] ko aate ho?", "[timing] bata do shop ki", "[kab tak khule] rehte ho",
               "lunch me [band] hota hai kya", "kal [khulega] na?", "[opening time] kya hai"],
        "en": ["what [time] do you open", "are you [closed] today", "[opening hours] on weekends?",
               "till [what time] are you open", "is the shop [open] now", "[holiday] tomorrow?"],
    },
    "Address": {
        "hi": ["aapki [location] bhejo", "dukaan [kahan] pe hai", "[address] kya hai shop ka", "shop ka [pata] do",
               "aapke yahan [kaise reach] karu", "[map] location share karo", "kis [jagah] hai shop"],
        "en": ["[where] are you located", "share your [address]", "how do I [reach] the shop", "send me the [location] pin"],
    },
    "Payment": {
        "hi": ["[upi] se de sakta hu?", "[qr code] bhejo payment ke liye", "[cash] chalega?", "[paytm] lete ho",
               "[phonepe] chalta hai", "[online] payment kar du?", "[cod] milega?", "[card] lete ho ya sirf cash"],
        "en": ["can I [pay online]", "do you take [upi]", "[cash on delivery] possible?", "is [card] payment okay",
               "send your [upi] id", "[gpay] works?"],
    },
    "Returns": {
        "hi": ["pasand na aaye to [wapas] le loge?", "[return] ka kya rule hai", "[exchange] ho sakta hai?",
               "[refund] kitne din me", "kharab nikla to [badalna] padega", "[damaged] aaya to?"],
        "en": ["what's your [return] rule", "can I [exchange] it later", "how does [refund] work",
               "if it's [defective] what happens"],
    },
    "GST bill": {
        "hi": ["[gst bill] chahiye", "[bill] me gst number aayega?", "[invoice] milega na", "[gst] alag lagta hai kya"],
        "en": ["I need a [gst invoice]", "is [gst] extra", "will I get a [bill]"],
    },
    "Delivery": {
        "hi": ["[ghar tak] bhejoge?", "[delivery charge] kya hai", "[home delivery] hoti hai?", "[delivery] free hai kya",
               "mere [ghar pe saman bhej] doge", "[courier] karte ho?", "[delivery] milegi mere area me?"],
        "en": ["do you [deliver] to my area", "what's the [delivery charge]", "is [home delivery] free",
               "can you [ship] it to me"],
    },
    "Bulk orders": {
        "hi": ["[zyada lene pe discount] milega?", "[wholesale] me doge?", "[bulk] me lena hai", "[party] ke liye zyada saman",
               "[function] ke liye order"],
        "en": ["any [bulk] rate", "[wholesale] price available?", "[discount] if I buy more?"],
    },
    "Warranty": {"hi": ["[warranty] kitne mahine ki", "[guarantee] deti ho?", "kharab hua to [warranty] me?"],
                 "en": ["how long is the [warranty]", "any [guarantee]"]},
    "Repair time": {"hi": ["kab tak [ready] milega phone", "[repair time] kitna lagega"], "en": ["when will it be [ready]"]},
    "Home visit": {"hi": ["[technician] aa sakta hai ghar?", "[ghar aake] check karoge?", "[technician] ka ghar aana hota hai?"],
                   "en": ["can you send a [technician]", "can a [technician] visit my home"]},
    "Custom cake": {"hi": ["[photo] wala cake banta hai?", "[theme] cake kitne din pehle bolna", "[custom] cake order"],
                    "en": ["do you make [custom] cakes", "[photo] cake possible?"]},
    "Stitching time": {"hi": ["silai me [kitne din]", "[kitne din] me de doge", "[ready] kab tak hoga suit"],
                       "en": ["[how many days] for tailoring"]},
    "Appointment": {"hi": ["[appointment] chahiye kya", "[booking] karni padegi?", "seedha [walk in] kar sakte?"],
                    "en": ["do I need [booking]", "can I just [walk in]"]},
    "Batch timings": {"hi": ["[batch] kab hota hai", "[evening batch] hai?", "class ki [batch timing] kya hai"],
                      "en": ["what are the [batch] times", "[weekend] classes?"]},
    "Order cutoff": {"hi": ["[lunch order] kitne baje tak de sakte", "[dinner order] ka last time kya"],
                     "en": ["latest time for a [lunch order]"]},
    "Menu": {"hi": ["aaj [menu] me kya hai", "[aaj kya bana] khane me"], "en": ["what's on the [menu] today"]},
    "Milk subscription": {"hi": ["[roz] doodh lagwana hai", "[monthly] doodh ka system"], "en": ["[daily] milk delivery?"]},
    "Wedding orders": {"hi": ["[shaadi] ka order lete ho", "[wedding] ke liye mithai"], "en": ["[wedding] orders?"]},
    "Online classes": {"hi": ["[online classes] bhi padhate ho?", "ghar se [online classes]"], "en": ["any [online classes] option"]},
}

# ═══════════════════════════════════════════════════════════════
# STORE: delivery / ready date (MQ_SEARCH → DATE)
# ═══════════════════════════════════════════════════════════════
DELIVERY_DATE = {
    "hi": ["abhi order karu to [delivery] kab", "[delivery] me kitne din lagenge", "kab tak [pahunch] jayega saman",
           "aaj diya order to [deliver] kab hoga", "[ghar] kab tak aayega mera saman", "[delivery] kab tak ho payegi"],
    "en": ["how soon can you [deliver]", "when will it [arrive] if I order now", "[delivery] takes how many days",
           "expected [delivery] date?"],
}
READY_DATE = {
    "hi": ["{g} de du to [kitne din] me milega", "{g} [kab tak ready] hoga", "{g} silne me [kitne din]",
           "{g} [kitne din] me tayyar"],
    "en": ["if I bring a {g} today, [how many days]"],
}

# ═══════════════════════════════════════════════════════════════
# WEATHER (slot {c})
# ═══════════════════════════════════════════════════════════════
WEATHER = {
    "general": {"hi": ["{c} me abhi mausam kaisa", "{c} ka weather kya hai", "{c} me kitna temperature hai",
                       "{c} ka aaj ka mausam", "abhi {c} me kaisa mausam chal raha", "{c} weather update do"],
                "en": ["what's the weather in {c} today", "{c} weather now", "how's {c} looking weather-wise",
                       "temperature in {c} right now"]},
    "umbrella": {"hi": ["{c} me aaj barsat hogi?", "{c} me baarish ka kya chance", "{c} jaana hai, chhata lu?",
                        "{c} me paani girega aaj?", "{c} me rain hogi kya"],
                 "en": ["any rain in {c} today", "should I take an umbrella to {c}", "chance of rain in {c}?"]},
    "jacket": {"hi": ["{c} me sardi kitni hai", "{c} me thandi chal rahi?", "{c} me garam kapde chahiye?",
                      "{c} me aaj thand padegi?"],
               "en": ["is it chilly in {c}", "how cold is {c} today", "need warm clothes in {c}?"]},
    "hot": {"hi": ["{c} me garmi bahut hai kya", "{c} me tapman kitna hai", "{c} me loo chal rahi hai?",
                   "{c} me dhoop kaisi hai"],
            "en": ["is it very hot in {c}", "how warm is {c} today"]},
    "event": {"hi": ["{c} me aaj bahar stall lagau?", "{c} me khule me programme rakh sakte?", "{c} me outdoor sale theek rahegi?"],
              "en": ["is today okay for an outdoor event in {c}", "good weather for a stall in {c} today?"]},
}

# ═══════════════════════════════════════════════════════════════
# NO-TOOL customer messages
# ═══════════════════════════════════════════════════════════════
GREETING = {
    "hi": ["namaskar", "jai shree krishna", "sat sri akal", "hello sir", "hello madam", "assalamu alaikum", "pranam",
           "hi bhaiya", "haan ji", "koi online hai", "hello? reply karo", "good afternoon", "shubh prabhat",
           "radhe radhe", "hii", "helo"],
    "en": ["hey", "good afternoon", "hello there", "hi, anyone available?", "morning!", "hiya", "hello team"],
}
THANKS = {
    "hi": ["bahut dhanyavaad", "thank u ji", "theek h", "chalo theek hai", "ok ji thank you", "achha ji", "samajh gaya",
           "haan theek", "sahi hai", "mast", "ho gaya", "ok bhai", "thik h", "accha theek", "shukriya ji"],
    "en": ["thanks a lot", "great thanks", "ok cool", "alright", "perfect, thank you", "noted", "cheers", "ok got it"],
}
ABOUT = {
    "hi": ["aap robot ho?", "kaun baat kar raha hai", "ye automatic reply hai kya", "aapka naam kya hai",
           "tum kya kya kar sakte ho", "kya main insaan se baat kar raha hu"],
    "en": ["is this a bot", "who am I talking to", "is this automated", "what's your name", "what can you help with"],
}
CLARIFY_PRICE = {
    "hi": ["ye kitne ka", "rate kya hai", "kitne ka padega", "price?", "kitna hua", "iska daam", "kitne me milega",
           "rate bhejo", "price list bhejo"],
    "en": ["how much is this", "what's the price", "cost please", "rate?", "price list?"],
}
CLARIFY_ORDER = {
    "hi": ["kuch saman chahiye", "order dena tha", "list bhej raha hu", "kuch items lene hain", "order karwa do"],
    "en": ["I want to order something", "need to place an order", "can I send you my list"],
}
ORDER_STATUS = {
    "hi": ["mera saman kab aayega", "order ka kya hua", "delivery abhi tak nahi hui", "parcel kab tak milega",
           "kal ka order abhi tak nahi aaya", "mera order dispatch hua?", "order kahan tak pahuncha"],
    "en": ["where's my parcel", "my delivery is late", "has my order shipped", "still waiting for my order",
           "any update on my order"],
}
COMPLAINT = {
    "hi": ["saman tuta hua aaya", "packet khula hua tha", "expiry nikal chuki thi", "galat saman bhej diya",
           "size chhota aa gaya", "taste kharab tha", "aadha saman hi aaya", "bill zyada laga diya",
           "delivery boy ne badtameezi ki", "rang alag aaya"],
    "en": ["item arrived broken", "wrong product delivered", "it's expired", "half my order is missing",
           "the size is wrong", "quality was bad"],
}
HUMAN = {
    "hi": ["malik se baat karni hai", "kisi aadmi se baat karao", "call karo mujhe", "owner ka number do",
           "manager ko bulao", "insaan se baat karni hai bot se nahi"],
    "en": ["let me talk to a person", "can the owner call me", "connect me to a human", "I want a callback"],
}
PAYMENT_DONE = {
    "hi": ["₹{amt} gpay kar diya", "{amt} paytm pe bheja", "payment kar diya {amt}", "{amt} transfer ho gaya",
           "upi se {amt} bhej diye check karo"],
    "en": ["sent ₹{amt} via gpay", "transferred {amt}", "paid {amt}, please confirm"],
}

# Out-of-scope / refuse politely (question variants → shared answer key)
DECLINE = {
    "medical": {"hi": ["sir dard ki goli kaunsi", "bukhar utarne ki dawai", "khasi ka syrup batao", "sugar ki tablet kaunsi",
                       "bacche ko dast ho rahe kya du", "kamar dard ki medicine"],
                "en": ["what pill for headache", "medicine for fever?", "which cough syrup is best"]},
    "loan": {"hi": ["udhaar milega?", "paise udhaar do", "loan dila sakte ho"], "en": ["can you lend me money", "do you give credit loans"]},
    "predict": {"hi": ["kal kaun jeetega", "share market kal upar jayega?", "lottery ka number batao"],
                "en": ["will the stock market go up", "who wins the election"]},
    "personal": {"hi": ["kisi ka phone number chahiye", "ladki ka number do", "uska address batao"],
                 "en": ["give me someone's number", "find me a person's address"]},
    "homework": {"hi": ["mera project bana do", "essay likh do pura", "exam ka paper solve karo"],
                 "en": ["do my homework", "write my essay"]},
}

# New: business advice (no tool) — question variants + one answer each
ADVICE = [
    (["dukaan ki sale kaise badhe", "sales badhane ke tarike batao", "how do I increase my shop sales", "bikri kaise badhaun"],
     "Regular customers ko yaad rakho, WhatsApp status pe roz naye items aur offers daalo, delivery fast rakho, aur khush customers se refer karne ko bolo."),
    (["gusse wale customer ko kaise handle karu", "customer chilla raha hai kya karu", "how to deal with an angry customer"],
     "Pehle shaanti se poori baat suno, maafi maango, jo turant theek ho sakta hai wo karo, aur baad me follow-up karke confirm karo ki problem solve ho gayi."),
    (["naye grahak ko offer du kya", "first order pe discount dena sahi hai?", "should I give a first-time discount"],
     "Chhota first-order offer madad karta hai, par limited rakho taaki margin bacha rahe. Customer quality aur service ke liye wapas aata hai."),
    (["whatsapp business kaise use karu", "whatsapp pe catalogue kaise banaye", "how to use whatsapp business for my shop"],
     "WhatsApp Business me catalog banao, quick replies set karo, labels se orders track karo, aur status pe roz offers daalo."),
    (["udhaar dena chahiye customers ko?", "should I give credit to customers"],
     "Sirf bharose wale regular customers ko, limit tay karke aur likh ke do. Due date fix rakho aur time pe yaad dilao."),
    (["stock kaise manage karu", "maal khatam ho jata hai kya karu", "how to avoid running out of stock"],
     "Jo items sabse zyada bikte hain unki list rakho, minimum stock level tay karo, aur us level pe aate hi order kar do."),
    (["price kaise tay karu apne saman ka", "how should I price my products"],
     "Poori laagat jodo, usme apna margin daalo, phir aas-paas ke rates aur customer kitna dene ko tayyar hai, ye dekh ke final karo."),
    (["online kaise bechu", "apni dukaan online kaise laun", "how do I start selling online"],
     "Shuruaat WhatsApp Business catalog aur Google Business Profile se karo. Orders badhne pe chahein to kisi marketplace pe bhi list kar sakte ho."),
    (["google pe dukaan kaise dikhe", "google maps pe shop kaise daalu"],
     "Google Business Profile banao, sahi address, timing, photos aur phone daalo, aur khush customers se review likhwao."),
    (["review kaise badhaun", "customers se review kaise lu"],
     "Order ke baad ek chhota thank-you message bhejo aur saath me review ka link do. Achhe review pe shukriya bolna mat bhoolo."),
]

# New: ambiguous follow-ups that need clarification, not a tool
CLARIFY_OTHER = {
    "hi": ["wo wala chahiye", "pichli baar wala", "same as last time", "jo bola tha wahi", "wo item bhejo"],
    "en": ["the same one as before", "send the usual", "that one please"],
}
CLARIFY_OTHER_ANSWERS = {
    "hi": "Kaunsa item chahiye? Naam ya photo bhej dijiye, main check kar deta hoon.",
    "en": "Which item do you mean? Send me the name or a photo and I'll check.",
}


# ═══════════════════════════════════════════════════════════════
# CHUNK 2 — calculations, dates, search, chains, concepts
#
# EXTRA is keyed by the FIRST Hindi template of each _p(...) call in the generator;
# the lists are merged into that call's pool, so slots must match that call exactly.
# ═══════════════════════════════════════════════════════════════
EXTRA = {
    # GST add  {p} {r}
    "₹{p} pe {r}% GST kitna lagega?": {
        "hi": ["{p} pe {r} percent gst jodo", "{p} + {r}% gst kitna banega", "{p} ka bill hai, {r}% gst laga ke batao",
               "{p} me {r}% gst add karke total", "gst {r}% lagake {p} kitna hoga", "{p} rupaye pe {r} pratishat GST",
               "invoice {p}, gst {r}%, kul kitna", "{p} ki cheez pe {r}% tax lagega, total?"],
        "en": ["{p} with {r} percent gst", "total of {p} including {r}% gst", "{r}% GST on {p} — final bill?",
               "add {r}% tax to {p}", "{p} plus {r}% GST please"]},
    # GST inclusive  {p} {r}
    "₹{p} me {r}% GST included hai, base price kya hoga?": {
        "hi": ["{p} gst milake hai ({r}%), bina gst kitna", "{p} me {r}% gst pehle se hai, asli rate", "{p} ka {r}% gst nikaal ke batao",
               "{p} total hai {r}% gst ke saath, gst kitna tha"],
        "en": ["{p} is gst-inclusive at {r}%, what's the base", "remove {r}% gst from {p}", "how much of {p} is {r}% gst"]},
    # CGST/SGST  {p} {r}
    "₹{p} pe {r}% GST ka CGST aur SGST alag batao": {
        "hi": ["{p} pe {r}% gst ka cgst sgst kitna", "{r}% gst {p} pe, aadha cgst aadha sgst batao"],
        "en": ["cgst sgst breakup for {p} at {r}%", "{p} at {r}% gst — split cgst and sgst"]},
    # discount  {p} {d}
    "₹{p} pe {d}% discount laga ke kitna hoga?": {
        "hi": ["{p} ki cheez {d} percent off pe kitne ki", "{p} pe {d}% chhoot ke baad", "{d}% kam karke {p} kitna",
               "{p} me se {d}% discount hatao", "{p} wala item {d}% sale me kitne ka", "{d} percent off {p} pe kitna bacha"],
        "en": ["{p} at {d} percent off", "{d}% off {p}, what do I pay", "discounted price of {p} with {d}% off",
               "take {d}% off {p}"]},
    # discount then GST  {p} {d} {r}
    "₹{p} pe {d}% discount, phir {r}% GST — final kitna?": {
        "hi": ["{p} pe pehle {d}% off fir {r}% gst", "{d}% discount aur {r}% gst {p} pe, final", "{p} - {d}% + {r}% gst kitna hua"],
        "en": ["{p} with {d}% off then {r}% gst", "apply {d}% discount and {r}% gst to {p}"]},
    # margin  {c} {s}
    "₹{c} me kharida, ₹{s} me becha. margin kitna?": {
        "hi": ["{c} ka maal {s} me becha, kitna profit", "{c} laagat, {s} bikri — margin %", "kharid {c} bikri {s}, munafa kitna",
               "{c} me liya {s} me diya, profit percent", "maal {c} ka, becha {s} me, faayda kitna"],
        "en": ["bought at {c}, sold at {s}, profit?", "cost {c} sale {s} what margin", "margin if I buy at {c} and sell at {s}"]},
    # markup  {c} {m}
    "₹{c} cost pe {m}% markup lagau to selling price kya hoga?": {
        "hi": ["{c} ki laagat pe {m}% munafa rakhna hai, kitne me bechu", "{c} pe {m} percent upar kitna", "laagat {c}, {m}% margin chahiye"],
        "en": ["cost {c}, want {m}% profit, sell price?", "{c} plus {m}% markup"]},
    # EMI  {amt} {r} {n}
    "₹{amt} loan, {r}% interest, {n} months — EMI kitni?": {
        "hi": ["{amt} ka loan, {r}% byaaj, {n} mahine, kisht kitni", "{amt} loan {r}% pe {n} months ki emi", "{n} mahine ki emi {amt} pe {r}% interest",
               "{amt} ka karza {r}% pe {n} mahine me chukana, mahina kitna"],
        "en": ["emi on {amt} at {r}% over {n} months", "monthly payment for {amt} loan, {r}% rate, {n} months",
               "{amt} borrowed at {r}% for {n} months — emi?"]},
    # simple interest  {amt} {r} {y}
    "₹{amt} pe {r}% simple interest {y} saal ka kitna?": {
        "hi": ["{amt} pe {r}% byaaj {y} saal ka", "{amt} fd {r}% {y} saal simple interest", "{y} saal me {amt} pe {r}% kitna byaaj"],
        "en": ["simple interest {amt} at {r}% for {y} yrs"]},
    # split  {b} {n}
    "₹{b} ka bill {n} logo me baanto": {
        "hi": ["{n} log hai, {b} ka bill, ek ka kitna", "{b} ko {n} me barabar baanto", "{b} ka khana {n} dost, har ek kitna de",
               "{n} logo ka hissa {b} me se"],
        "en": ["{b} split {n} ways", "{n} people, bill {b}, each pays?", "divide {b} among {n}"]},
    # change  {b} {g}
    "bill ₹{b} hai, customer ne ₹{g} diye, kitna wapas du?": {
        "hi": ["{g} ka note diya, {b} ka saman, kitne lautane", "{b} ke bill pe {g} mile, change kitna", "{g} me se {b} gaye, baaki kitna",
               "customer ne {g} diya {b} ke liye, wapas kitna"],
        "en": ["{g} given for a {b} bill, change?", "how much to return from {g} on {b}"]},
    # per unit  {q} {u} {t}
    "{q} {u} ka ₹{t}, per {u} kitna pada?": {
        "hi": ["{t} me {q} {u} aaya, ek {u} ka kitna", "{q} {u} {t} rupaye me, rate per {u}", "{t} ka {q} {u}, ek ka bhaav"],
        "en": ["{q} {u} for {t}, unit price?", "per-{u} cost if {q} {u} cost {t}"]},
    # growth  {a0} {a1}
    "pichle mahine sale ₹{a0}, is mahine ₹{a1}. kitna % change?": {
        "hi": ["{a0} se {a1} pe aa gaye, kitna percent", "sale {a0} thi ab {a1}, badhi ya ghati kitni", "{a0} → {a1}, percent change",
               "pichli baar {a0}, is baar {a1}, kitna fark %"],
        "en": ["{a0} to {a1}, percent change?", "revenue moved from {a0} to {a1}, by what %"]},
    # daily target  {t} {d}
    "mahine ka target ₹{t} hai, {d} din kaam, roz kitna bechna padega?": {
        "hi": ["{t} ka target {d} din me, roz kitna", "{d} din me {t} bechna hai, daily kitna"],
        "en": ["{t} target in {d} days, per day?"]},
    # area  {l} {w}
    "{l} x {w} feet ki dukaan ka area?": {
        "hi": ["{l} by {w} ka plot kitna sq ft", "{l} feet lamba {w} feet chauda, area", "{l}x{w} kamra kitne square feet"],
        "en": ["{l} by {w} feet area", "square feet of a {l}x{w} room"]},
    # rent per sqft  {a} {r}
    "{a} sq ft ka rent ₹{r}, per sq ft kitna?": {
        "hi": ["{r} rent {a} square feet ka, rate kya pada", "{a} sq ft {r} me, ek sq ft ka kiraya"],
        "en": ["{r} rent for {a} sq ft — rate per sq ft"]},
    # percent of  {v} {p}
    "{v} ka {p}% kitna?": {
        "hi": ["{v} ka {p} pratishat", "{p}% of {v} nikalo", "{v} me {p} percent kitna hua", "{v} ka {p} percent batao jaldi"],
        "en": ["{p} percent of {v}?", "calculate {p}% of {v}", "{v} times {p}%"]},
    # average  {s}
    "is hafte ki sale: {s}. average kitna?": {
        "hi": ["{s} — inka average", "roz ki kamai {s}, ausat kitna", "average nikal do: {s}"],
        "en": ["mean of {s}", "average these: {s}"]},
    # staff total  {n} {s} {s2}
    "{n} staff hai, har ek ko {s} salary, total kitna dena hai": {
        "hi": ["{n} karmchari, sabko {s}, kul salary", "{n} ladke kaam karte, {s} {s2}, total kitna", "{n} workers ko {s} dena hai, total"],
        "en": ["salary {s} for {n} staff, total?", "{n} people × {s}, total payroll"]},
    # wage  {n} {d} {w}
    "{n} mazdoor, {d} din, {w} rupaye roz — kitna hua?": {
        "hi": ["{n} karigar {d} din {w} roz, kul kharcha", "{w} dihadi, {n} log, {d} din, total"],
        "en": ["{n} labourers, {d} days, {w} a day — total"]},
    # lakh percent  {l} {p}
    "{l} lakh ka {p}% kitna?": {
        "hi": ["{l} lakh pe {p} pratishat", "{p}% of {l} lakh", "{l} lakh me {p}% kitna aata"],
        "en": ["{p} percent of {l} lakh"]},
    # dozen  {d} {pr}
    "{d} dozen kele, {pr} rupaye dozen, kitna hua?": {
        "hi": ["{pr} ka dozen, {d} dozen lena, total", "{d} darjan {pr} ke hisaab se kitna"],
        "en": ["{d} dozen at {pr} a dozen"]},
    # ── dates ──
    "aaj kya date hai?": {
        "hi": ["aaj ki tareekh", "aaj kaun si tarikh", "aaj ka din aur date", "date batao aaj ki", "aaj kaunsa vaar hai"],
        "en": ["date today?", "what date is it", "which day is today"]},
    "abhi kitne baje hain?": {
        "hi": ["kitna time hua", "abhi kya baja", "time bata do", "kitne baje hai abhi"],
        "en": ["current time please", "what's the time"]},
    "{d} ko kaun sa din hai?": {
        "hi": ["{d} kaunsa vaar", "{d} ko konsa din padega", "{d} ka din kya hai", "{d} ko week ka kaunsa din"],
        "en": ["day of week on {d}", "{d} falls on which day"]},
    "{d} me kitne din bache hain?": {
        "hi": ["{d} aane me kitna time", "{d} tak kitne din reh gaye", "{d} kitni door hai"],
        "en": ["days left until {d}", "how far is {d}"]},
    "aaj se {n} din baad kaun si date hogi?": {
        "hi": ["{n} din aage ki date", "aaj + {n} din kya date", "{n} din baad ki tareekh batao"],
        "en": ["date {n} days from now", "what's the date in {n} days"]},
    "{a} se {b} tak kitne din?": {
        "hi": ["{a} aur {b} ke beech kitne din", "{a} to {b} kitne din hue"],
        "en": ["days from {a} to {b}"]},
    "{f} me kitne din bache?": {
        "hi": ["{f} me kitna time hai", "{f} aane me kitne din", "{f} tak kitne din"],
        "en": ["how long till {f}", "{f} is how many days away"]},
    "{f} kab hai?": {
        "hi": ["{f} kab padega", "{f} ki tarikh", "{f} kis din hai", "{f} kab manaya jayega"],
        "en": ["{f} date?", "when's {f}"]},
}

# chains: phrasings with number words / dozen / "chhoot" / "tayyar"
QTY_EXTRA_UNITS = {"hi": ["packet", "piece", "bag", "dabba", "bottle", "kilo pack"], "en": ["packs", "bottles", "boxes"]}
BULK = {
    "hi": ["₹{amt} ka order dena hai, [chhoot] kitni milegi", "{amt} rupaye ka maal, [discount] milega?",
           "₹{amt} ki kharidari pe [bulk discount] hai?", "{amt} ka saaman le raha hu, [chhoot] kitni"],
    "en": ["ordering {amt} worth — any [discount]?", "for a {amt} order, [bulk discount]?"],
}
DELIVERY_AFTER_ORDER = {
    "hi": ["{p} order kiya hai aaj, [kab aayega]", "abhi {p} mangwaya, [delivery] kab", "{p} ka order diya, [kab tak pahunchega]"],
    "en": ["just ordered {p}, when will it [arrive]", "ordered {p} today, [delivery] date?"],
}
READY_EXTRA = {
    "hi": ["{g} aaj silne di to [kab tak tayyar]", "{g} [kitne din me tayyar] hogi", "{g} diya hai, [tayyar] kab milega"],
    "en": ["dropped a {g} today, when will it be [ready]"],
}

# search: how / who / differences (generic {t})
SEARCH = {
    "hi": ["{t} ke baare me thoda batao", "{t} kis kaam aata hai", "{t} kya cheez hoti hai", "{t} ka matlab samjhao",
           "{t} ki jaankari do", "{t} kya hai seedhe shabdon me"],
    "en": ["what's {t} exactly", "give me a quick intro to {t}", "{t} in simple words", "what is {t} used for"],
}

# no-tool concepts (question variants → one answer)
CONCEPTS = [
    (["what is git", "git kya hai", "git kyun use karte hain"],
     "Git is a version-control tool: it records every change to your code so you can go back, branch off, and merge work with others."),
    (["difference between list and dict in python", "python me list aur dict ka fark"],
     "A list keeps items in order and you access them by position; a dict maps keys to values and you access them by key."),
    (["what is an http status code", "status code kya hota hai"],
     "It's a number the server sends back to say how a request went — 2xx success, 4xx client error, 5xx server error."),
    (["frontend vs backend", "frontend aur backend me kya farak"],
     "Frontend is what users see and click in the browser or app; backend is the server code and database that store and process the data."),
    (["what is an ORM", "ORM kya hota hai"],
     "An ORM lets you work with database rows as objects in your code instead of writing raw SQL for every query."),
    (["what is a unit test", "unit test kya hota hai"],
     "A unit test checks one small piece of code, like a single function, automatically — so you know when a change breaks it."),
    (["how to read an error traceback", "python error kaise padhe"],
     "Start from the last line — it names the error. The lines above show the call path; the last one from your own file is usually where to look."),
    (["what is cash flow", "cash flow kya hota hai"],
     "Cash flow is the money actually coming in and going out of your business over a period — profit on paper doesn't help if cash runs out."),
    (["what is working capital in simple words", "working capital simple me"],
     "It's the money you need to run day-to-day — stock, rent, salaries — before customers pay you."),
    (["what is MRP", "MRP ka matlab"],
     "MRP is the Maximum Retail Price printed on a packaged product in India; shops can't legally charge more than that."),
    (["what is inventory turnover", "stock kitni baar ghoomta hai kaise pata kare"],
     "It's how many times you sell through your stock in a period. Higher usually means less money stuck in unsold goods."),
    (["what is a gst invoice", "gst invoice me kya hona chahiye"],
     "A GST invoice shows seller and buyer details, GSTIN, invoice number and date, item description, HSN code, taxable value and the GST charged."),
]
