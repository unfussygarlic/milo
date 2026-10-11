"""
Milo — synthetic Indian MSME merchants for MQ_SEARCH training data.

Each call to build_kb() makes a fresh variant of a merchant type:
random shop name, city, area, hours, delivery terms, payment modes,
prices jittered ±12%, and random stock/offers. The point is that the
model can never memorise a price or a policy — it has to read them
from the MQ_SEARCH result every time.

    from merchants import random_kb
    kb = random_kb(rng)                 # one merchant KB dict
    milo_tools.set_merchant_kb(kb)

    python merchants.py --export merchant_kbs/   # one JSON per type, for the runtime
    python merchants.py --show kirana            # print a sample KB
"""

import argparse
import json
import math
import os
import random

# ═══════════════════════════════════════════════════════════════
# SHARED POOLS
# ═══════════════════════════════════════════════════════════════
SURNAMES = [
    "Sharma", "Gupta", "Agarwal", "Verma", "Patel", "Shah", "Reddy", "Nair", "Iyer",
    "Das", "Bose", "Singh", "Yadav", "Khan", "Qureshi", "Joshi", "Kulkarni", "Deshmukh",
    "Pillai", "Menon", "Rao", "Jain", "Mehta", "Chauhan", "Mishra", "Tiwari", "Ansari",
    "Fernandes", "Dsouza", "Banerjee", "Mukherjee", "Saxena", "Bansal", "Goyal", "Naidu",
]
FIRST_NAMES = ["Ravi", "Sunita", "Anil", "Meena", "Suresh", "Kavita", "Imran", "Lakshmi",
               "Rajesh", "Pooja", "Arjun", "Fatima", "Deepak", "Asha", "Vikram", "Neha"]
AREAS = [
    "Main Market", "Station Road", "Gandhi Nagar", "Sector 14", "MG Road", "Civil Lines",
    "Shastri Nagar", "Nehru Chowk", "Old City", "Bus Stand Road", "Laxmi Nagar",
    "Rajendra Nagar", "Market Yard", "Ring Road", "Subhash Chowk", "Tilak Nagar",
]
CITIES = [
    "Indore", "Bhopal", "Jaipur", "Lucknow", "Kanpur", "Patna", "Ranchi", "Nagpur", "Pune",
    "Nashik", "Surat", "Ahmedabad", "Vadodara", "Rajkot", "Kochi", "Coimbatore", "Madurai",
    "Mysuru", "Hubballi", "Vizag", "Vijayawada", "Bhubaneswar", "Guwahati", "Dehradun",
    "Chandigarh", "Ludhiana", "Amritsar", "Varanasi", "Agra", "Raipur", "Mumbai", "Delhi",
    "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Noida", "Gurugram", "Thane", "Jodhpur",
]
HOURS = [
    "9 AM – 9 PM, Monday to Saturday. Closed on Sunday.",
    "8 AM – 10 PM, all days.",
    "10 AM – 8 PM, all days. Closed on Tuesday.",
    "7 AM – 1 PM and 4 PM – 9 PM, all days.",
    "10 AM – 9 PM, Monday to Saturday. Sunday 10 AM – 2 PM.",
    "9:30 AM – 8:30 PM, all days except Thursday.",
]
PAYMENT_MODES = ["UPI (GPay, PhonePe, Paytm)", "cash", "debit/credit cards", "cash on delivery"]

FAQ_KW = {
    "hours": "hours timing time open close khulta khulega baje band sunday holiday chutti closed",
    "address": "address location where kahan pata located map directions reach shop dukaan",
    "delivery": "delivery deliver home ship shipping charge courier days door date reach arrive pahunch",
    "payment": "payment pay upi cash card cod gpay phonepe paytm online qr",
    "returns": "return exchange refund wapas badalna damaged defective",
    "gst": "gst bill invoice tax receipt",
    "bulk": "bulk wholesale discount order quantity party function zyada",
}

STOCK_STYLES = {
    "goods":   (["in stock"] * 15 + ["out of stock"] * 3, "only {n} left"),
    "service": (["available"] * 18 + ["fully booked today"] * 2, None),
    "course":  (["seats available"] * 15 + ["batch full"] * 3, "only {n} seats left"),
}

# ═══════════════════════════════════════════════════════════════
# MERCHANT TYPES
# item tuple: (name, variant, base_price, keywords, category[, flags])
#   flags: "from" → price is a starting price (services)
# ═══════════════════════════════════════════════════════════════
MERCHANT_TYPES = {
    "kirana": dict(
        names=["{s} Kirana Store", "{s} General Store", "Shree {s} Provision Store",
               "{s} Super Mart", "New {s} Kirana"],
        stock="goods",
        items=[
            ("Basmati Rice", "1kg", 139, "basmati rice", "grocery"),
            ("Basmati Rice", "5kg", 649, "basmati rice", "grocery"),
            ("Sona Masoori Rice", "5kg", 399, "sona masoori rice", "grocery"),
            ("Wheat Atta", "5kg", 265, "atta wheat flour", "grocery"),
            ("Wheat Atta", "10kg", 499, "atta wheat flour", "grocery"),
            ("Toor Dal", "1kg", 165, "toor tur arhar dal daal", "grocery"),
            ("Moong Dal", "1kg", 145, "moong mung dal daal", "grocery"),
            ("Sunflower Oil", "1l", 165, "sunflower refined oil", "grocery"),
            ("Mustard Oil", "1l", 185, "mustard sarson oil kachi ghani", "grocery"),
            ("Sugar", "1kg", 48, "sugar", "grocery"),
            ("Salt", "1kg", 28, "salt iodised", "grocery"),
            ("Tea Powder", "500g", 285, "tea powder", "grocery"),
            ("Instant Noodles", "pack of 4", 56, "noodles maggi", "snacks"),
            ("Desi Ghee", "1l", 620, "ghee desi", "grocery"),
            ("Besan", "500g", 55, "besan gram flour", "grocery"),
            ("Poha", "500g", 42, "poha flattened rice", "grocery"),
            ("Bath Soap", "pack of 4", 120, "soap bath", "household"),
            ("Detergent Powder", "1kg", 115, "detergent surf washing powder", "household"),
        ],
        delivery=True, delivery_days=[0, 1], bulk=True,
        returns=["Packed items can be exchanged within 2 days if unopened. Loose items are not returnable.",
                 "Damaged or expired items replaced within 24 hours. No returns on opened packets."],
        extra=[("Minimum order", "Home delivery needs a minimum order of ₹{min_order}.",
                "minimum order min delivery")],
    ),
    "mobile": dict(
        names=["{s} Mobile Point", "{s} Mobile Accessories", "Smart {s} Mobiles",
               "{s} Telecom", "Mobile Care {s}"],
        stock="goods",
        items=[
            ("Fast Charger 33W", "", 799, "charger adapter fast type c", "accessories"),
            ("Type-C Cable", "1m", 249, "cable type c data usb wire", "accessories"),
            ("Lightning Cable", "1m", 399, "iphone lightning cable", "accessories"),
            ("Wired Earphones", "", 299, "earphones handsfree wired", "audio"),
            ("Bluetooth Earbuds", "", 1299, "earbuds bluetooth tws wireless earphones", "audio"),
            ("Power Bank", "10000mAh", 1099, "power bank powerbank", "accessories"),
            ("Power Bank", "20000mAh", 1699, "power bank powerbank", "accessories"),
            ("Tempered Glass", "", 149, "tempered glass screen guard protector", "accessories"),
            ("Back Cover", "", 249, "back cover case pouch", "accessories"),
            ("Screen Replacement", "", 1499, "screen display replacement broken crack toot tooti tuti tuta lagwana lagwane", "repair", "from"),
            ("Battery Replacement", "", 899, "battery replacement", "repair", "from"),
            ("Charging Port Repair", "", 499, "charging port jack repair", "repair", "from"),
            ("Memory Card", "64GB", 549, "memory card sd", "storage"),
            ("Pendrive", "64GB", 499, "pendrive pen drive usb", "storage"),
        ],
        delivery=True, delivery_days=[1, 2, 3], bulk=False,
        returns=["7-day replacement on accessories if defective, with bill. No returns on tempered glass once applied.",
                 "Defective products replaced within 7 days with the box and bill."],
        extra=[("Warranty", "Chargers and earbuds carry 6 months warranty. Repairs have 30 days warranty.",
                "warranty guarantee"),
               ("Repair time", "Most repairs are done the same day; screen replacement takes 1 day.",
                "repair time kitna time how long ready")],
    ),
    "ethnic_wear": dict(
        names=["{s} Fashion", "{s} Collection", "{f}'s Boutique", "{s} Saree Centre",
               "{s} Ethnic Wear"],
        stock="goods",
        items=[
            ("Cotton Kurti", "S to XXL", 599, "kurti cotton", "women"),
            ("Rayon Kurti", "S to XXL", 449, "kurti rayon", "women"),
            ("Chikankari Kurti", "M to XL", 1299, "chikankari chikan lucknowi kurti", "women"),
            ("Anarkali Suit Set", "M to XXL", 1899, "anarkali suit set", "women"),
            ("Cotton Saree", "", 899, "saree cotton", "women"),
            ("Banarasi Silk Saree", "", 4999, "banarasi silk saree", "women"),
            ("Bandhani Dupatta", "", 499, "dupatta chunri bandhani", "women"),
            ("Palazzo Pants", "free size", 399, "palazzo pants", "women"),
            ("Lehenga Choli", "", 6499, "lehenga lehnga choli bridal", "women"),
            ("Men's Kurta", "M to XXL", 799, "mens men kurta", "men"),
            ("Nehru Jacket", "M to XL", 1199, "nehru modi jacket", "men"),
        ],
        delivery=True, delivery_days=[3, 4, 5, 7], bulk=False,
        returns=["Size exchange within 7 days with tags intact. No refunds, store credit only.",
                 "Exchange within 5 days if unused with tags. Sale items are not exchangeable."],
        extra=[("Alteration", "Free basic alteration for fitting, ready in 2 days.",
                "alteration fitting tight loose")],
    ),
    "hardware": dict(
        names=["{s} Hardware", "{s} Hardware & Sanitary", "{s} Paints & Hardware",
               "{s} Building Materials", "New {s} Electricals"],
        stock="goods",
        items=[
            ("PVC Pipe", "1 inch, 10ft", 320, "pvc pipe", "plumbing"),
            ("Cement", "50kg bag", 420, "cement bag", "building"),
            ("Wall Putty", "20kg", 780, "putty wall", "paint"),
            ("Emulsion Paint", "4l", 1650, "paint emulsion", "paint"),
            ("LED Bulb", "9W", 99, "led bulb light", "electrical"),
            ("Ceiling Fan", "1200mm", 2399, "fan ceiling", "electrical"),
            ("Copper Wire", "1.5 sq mm, 90m", 1450, "wire copper house wiring", "electrical"),
            ("Modular Switch", "6A", 45, "switch modular", "electrical"),
            ("Hammer", "500g", 249, "hammer hathoda", "tools"),
            ("Screwdriver Set", "6 pieces", 299, "screwdriver set pechkas", "tools"),
            ("Drill Machine", "", 2499, "drill machine", "tools"),
            ("Water Tap", "", 349, "tap nal water bathroom", "plumbing"),
            ("Padlock", "", 199, "lock padlock tala", "hardware"),
        ],
        delivery=True, delivery_days=[0, 1, 2], bulk=True,
        returns=["Unused items returnable within 7 days with bill. Cut wires and mixed paint are not returnable.",
                 "Exchange within 3 days with bill. Electrical items only exchanged if defective."],
        extra=[("Cement delivery", "Cement and pipes delivered by tempo; ₹{tempo} tempo charge within city.",
                "tempo truck heavy cement delivery")],
    ),
    "bakery": dict(
        names=["{s} Bakers", "{s} Cake Shop", "Sweet Bite {s} Bakery", "{s} Bakery & Cafe",
               "Golden {s} Bakers"],
        stock="goods",
        items=[
            ("Black Forest Cake", "500g", 450, "black forest cake", "cakes"),
            ("Black Forest Cake", "1kg", 850, "black forest cake", "cakes"),
            ("Chocolate Truffle Cake", "1kg", 950, "chocolate truffle cake", "cakes"),
            ("Pineapple Cake", "1kg", 700, "pineapple cake", "cakes"),
            ("Eggless Butterscotch Cake", "1kg", 800, "eggless butterscotch cake", "cakes"),
            ("Bread", "400g", 45, "bread loaf white", "bread"),
            ("Brown Bread", "400g", 55, "brown bread", "bread"),
            ("Pav", "pack of 6", 30, "pav bun", "bread"),
            ("Veg Puff", "", 25, "puff veg patties patty", "snacks"),
            ("Cupcakes", "pack of 6", 240, "cupcake cupcakes muffin", "cakes"),
            ("Cookies", "250g", 180, "cookies biscuit", "snacks"),
            ("Plum Cake", "500g", 350, "plum cake christmas", "cakes"),
        ],
        delivery=True, delivery_days=[0], bulk=True,
        returns=["Food items cannot be returned. If a cake is damaged on delivery, we replace it the same day.",
                 "No returns on food items. Complaints about quality accepted within 2 hours of delivery."],
        extra=[("Custom cake", "Photo and theme cakes need {advance_days} day advance order with 50% advance payment.",
                "custom photo theme design birthday advance order"),
               ("Eggless", "All cakes are available eggless at ₹50 extra per kg.",
                "eggless egg veg")],
    ),
    "stationery": dict(
        names=["{s} Stationers", "{s} Book Depot", "{s} Xerox & Stationery", "Vidya {s} Stores",
               "{s} Book House"],
        stock="goods",
        items=[
            ("Notebook", "172 pages", 60, "notebook register", "school"),
            ("Long Notebook", "200 pages", 85, "long notebook register", "school"),
            ("Ball Pen", "pack of 10", 100, "pen ball", "school"),
            ("Gel Pen", "", 20, "gel pen", "school"),
            ("Geometry Box", "", 180, "geometry box compass instrument", "school"),
            ("A4 Paper Ream", "500 sheets", 399, "a4 paper ream", "office"),
            ("Xerox Black & White", "per page", 2, "xerox photocopy black white", "print"),
            ("Colour Printout", "per page", 10, "colour color print printout", "print"),
            ("Spiral Binding", "", 40, "spiral binding", "print"),
            ("Lamination", "A4", 30, "lamination laminate", "print"),
            ("School Bag", "", 899, "school bag backpack", "school"),
            ("Drawing Book", "", 50, "drawing book", "school"),
            ("Scientific Calculator", "", 650, "calculator scientific", "office"),
        ],
        delivery=False, bulk=True,
        returns=["Unused items exchangeable within 3 days with bill. Printouts are not refundable.",
                 "Exchange only, within 5 days, for unused items."],
        extra=[("Print from phone", "Send files on this WhatsApp; prints ready in 30 minutes.",
                "print whatsapp file pdf phone send")],
    ),
    "sweets": dict(
        names=["{s} Sweets", "{s} Mishthan Bhandar", "{s} Sweet House", "Shree {s} Sweets",
               "{s} Mithaiwala"],
        stock="goods",
        items=[
            ("Rasgulla", "1kg", 360, "rasgulla rosogolla", "sweets"),
            ("Kaju Katli", "500g", 550, "kaju katli barfi", "sweets"),
            ("Kaju Katli", "1kg", 1080, "kaju katli barfi", "sweets"),
            ("Gulab Jamun", "1kg", 340, "gulab jamun", "sweets"),
            ("Soan Papdi", "500g", 180, "soan papdi", "sweets"),
            ("Motichoor Laddu", "1kg", 480, "laddu ladoo motichoor", "sweets"),
            ("Besan Laddu", "1kg", 520, "besan laddu ladoo", "sweets"),
            ("Mishti Doi", "500g", 120, "mishti doi sweet curd", "sweets"),
            ("Sandesh", "250g", 160, "sandesh", "sweets"),
            ("Diwali Gift Box", "1kg", 950, "diwali gift box hamper festival", "gift"),
            ("Namkeen Mix", "400g", 140, "namkeen mixture", "snacks"),
        ],
        delivery=True, delivery_days=[0, 1], bulk=True,
        returns=["Sweets cannot be returned. Please check at delivery.",
                 "No returns on food items."],
        extra=[("Wedding orders", "Wedding and function orders need {advance_days} days advance booking.",
                "wedding shaadi function party advance booking")],
    ),
    "repair": dict(
        names=["{s} Electronics Repair", "{s} Service Centre", "{s} Cool Care (AC & Fridge)",
               "{s} Home Appliance Repair", "Quick Fix {s}"],
        stock="service",
        items=[
            ("TV Repair", "", 499, "tv television led repair", "repair", "from"),
            ("Fridge Repair", "", 599, "fridge refrigerator repair", "repair", "from"),
            ("Washing Machine Repair", "", 499, "washing machine repair", "repair", "from"),
            ("AC Service", "", 599, "ac service servicing cleaning", "service", "from"),
            ("AC Gas Refill", "", 2499, "ac gas refill charging", "service", "from"),
            ("Mixer Grinder Repair", "", 249, "mixer grinder repair", "repair", "from"),
            ("Inverter Battery", "150Ah", 14500, "inverter battery", "products"),
            ("Home Visit Charge", "", 199, "visit charge home inspection", "service"),
        ],
        delivery=False, bulk=False,
        returns=["30 days warranty on repair work and 6 months on new parts.",
                 "Repairs carry 60 days service warranty. Replaced parts follow company warranty."],
        extra=[("Home visit", "Technician home visit within {city} the same day if booked before 2 PM.",
                "home visit technician ghar aake booking book"),
               ("Repair time", "Most repairs done in 1–2 days; parts may take 3 days.",
                "repair time kitna time how long ready")],
    ),
    "tiffin": dict(
        names=["{f}'s Kitchen", "{s} Tiffin Service", "Ghar Ka Khana by {f}", "{s} Bhojanalaya",
               "Annapurna {s} Tiffins"],
        stock="goods",
        items=[
            ("Veg Thali", "", 120, "veg thali lunch dinner food", "meals"),
            ("Non-Veg Thali", "", 180, "non veg nonveg thali chicken", "meals"),
            ("Mini Meal", "", 80, "mini meal", "meals"),
            ("Monthly Tiffin Veg", "30 meals", 3200, "monthly tiffin subscription veg", "subscription"),
            ("Monthly Tiffin Non-Veg", "30 meals", 4500, "monthly tiffin subscription non veg", "subscription"),
            ("Extra Roti", "", 8, "roti chapati", "add-on"),
            ("Jeera Rice", "", 70, "jeera rice", "add-on"),
            ("Paneer Butter Masala", "", 160, "paneer butter masala", "add-on"),
            ("Dal Tadka", "", 110, "dal tadka daal", "add-on"),
        ],
        delivery=True, delivery_days=[0], bulk=True,
        returns=["Food cannot be returned. Missed or wrong meals are replaced next day.",
                 "If a meal is late by more than 45 minutes, the next meal is free."],
        extra=[("Order cutoff", "Lunch orders by 11 AM, dinner orders by 6 PM.",
                "cutoff order last kab tak lunch dinner"),
               ("Menu", "Menu changes daily; today's menu is shared every morning on WhatsApp.",
                "menu today aaj kya bana")],
    ),
    "fruits_veg": dict(
        names=["{s} Fruits & Vegetables", "Fresh {s} Sabzi Mandi", "{s} Fresh Mart",
               "Green {s} Veggies", "{s} Fruit Centre"],
        stock="goods",
        items=[
            ("Potato", "1kg", 32, "potato", "vegetables"),
            ("Onion", "1kg", 38, "onion", "vegetables"),
            ("Tomato", "1kg", 40, "tomato", "vegetables"),
            ("Banana", "1 dozen", 60, "banana", "fruits"),
            ("Shimla Apple", "1kg", 180, "apple shimla", "fruits"),
            ("Alphonso Mango", "1 dozen", 900, "mango alphonso hapus", "fruits"),
            ("Ginger", "250g", 40, "ginger", "vegetables"),
            ("Garlic", "250g", 60, "garlic", "vegetables"),
            ("Coriander", "1 bunch", 15, "coriander dhaniya", "vegetables"),
            ("Spinach", "1 bunch", 25, "spinach palak", "vegetables"),
            ("Green Chilli", "250g", 20, "chilli mirchi hari green", "vegetables"),
            ("Coconut", "1 piece", 35, "coconut nariyal", "fruits"),
        ],
        delivery=True, delivery_days=[0], bulk=True,
        returns=["Bad quality items replaced on the next delivery. Please share a photo.",
                 "If anything is spoilt, send a photo within 2 hours for a refund."],
        extra=[("Morning slot", "Morning delivery 7–10 AM, evening 5–8 PM.",
                "slot morning evening time subah shaam")],
    ),
    "tailor": dict(
        names=["{s} Tailors", "{f} Ladies Tailor", "{s} Stitching Centre", "Perfect Fit {s}",
               "{s} Gents & Ladies Tailor"],
        stock="service",
        items=[
            ("Blouse Stitching", "", 450, "blouse stitching", "stitching"),
            ("Salwar Suit Stitching", "", 650, "salwar suit stitching", "stitching"),
            ("Kurti Stitching", "", 350, "kurti stitching", "stitching"),
            ("Shirt Stitching", "", 500, "shirt stitching", "stitching"),
            ("Pant Stitching", "", 550, "pant trouser stitching", "stitching"),
            ("Lehenga Stitching", "", 2500, "lehenga stitching", "stitching", "from"),
            ("Alteration", "", 100, "alteration fitting", "alteration", "from"),
            ("Saree Fall Pico", "", 120, "fall pico saree", "alteration"),
        ],
        delivery=False, bulk=False,
        returns=["Free re-fitting within 7 days if the fit is not right.",
                 "Fitting issues corrected free of cost within 10 days."],
        extra=[("Stitching time", "Stitching takes {stitch_days} days; urgent in 2 days for ₹200 extra.",
                "kitne din time ready kab days urgent")],
    ),
    "salon": dict(
        names=["{s} Salon", "Glam {f} Beauty Parlour", "{s} Unisex Salon", "{f}'s Beauty Studio",
               "Style Zone {s}"],
        stock="service",
        items=[
            ("Men's Haircut", "", 200, "haircut men mens cutting hair", "hair"),
            ("Women's Haircut", "", 450, "haircut women ladies hair", "hair"),
            ("Beard Trim", "", 100, "beard trim shave dadhi", "grooming"),
            ("Facial", "", 899, "facial cleanup", "skin", "from"),
            ("Hair Colour", "", 999, "hair colour color dye", "hair", "from"),
            ("Hair Spa", "", 799, "hair spa", "hair"),
            ("Full Arms Waxing", "", 350, "waxing wax arms", "skin"),
            ("Bridal Makeup", "", 14999, "bridal makeup dulhan wedding", "makeup", "from"),
            ("Manicure", "", 499, "manicure nails", "nails"),
            ("Pedicure", "", 599, "pedicure feet", "nails"),
        ],
        delivery=False, bulk=False,
        returns=["Not happy with a service? Tell us within 3 days for a free redo.",
                 "Free touch-up within 7 days for haircuts and colour."],
        extra=[("Appointment", "Walk-ins welcome; book on WhatsApp to skip waiting.",
                "appointment booking book slot walk in")],
    ),
    "dairy": dict(
        names=["{s} Dairy", "{s} Milk Centre", "Gokul {s} Dairy", "{s} Dairy Farm Fresh",
               "Shree {s} Dugdhalaya"],
        stock="goods",
        items=[
            ("Cow Milk", "1l", 62, "milk cow", "milk"),
            ("Buffalo Milk", "1l", 72, "milk buffalo bhains", "milk"),
            ("Curd", "500g", 45, "curd", "milk products"),
            ("Paneer", "200g", 90, "paneer cottage cheese", "milk products"),
            ("Paneer", "1kg", 420, "paneer cottage cheese", "milk products"),
            ("Desi Ghee", "500ml", 380, "ghee desi", "milk products"),
            ("Butter", "100g", 58, "butter", "milk products"),
            ("Lassi", "200ml", 30, "lassi", "drinks"),
            ("Buttermilk", "500ml", 25, "chaas chhach buttermilk", "drinks"),
            ("Khoya", "500g", 220, "khoya mawa", "milk products"),
        ],
        delivery=True, delivery_days=[0], bulk=True,
        returns=["Spoilt milk replaced next morning. Please inform the same day.",
                 "Quality complaints accepted the same day for a replacement."],
        extra=[("Milk subscription", "Daily morning delivery 6–8 AM with monthly billing.",
                "subscription monthly daily roz morning subah")],
    ),
    "pet_shop": dict(
        names=["{s} Pet Shop", "Paws & Tails {s}", "{s} Pet Care", "Happy Pets {s}",
               "{s} Aquarium & Pets"],
        stock="goods",
        items=[
            ("Adult Dog Food", "3kg", 1450, "dog food adult", "food"),
            ("Puppy Food", "3kg", 1550, "puppy dog food", "food"),
            ("Cat Food", "1.2kg", 650, "cat food", "food"),
            ("Dog Leash", "", 399, "leash belt collar", "accessories"),
            ("Dog Bed", "medium", 1299, "dog bed", "accessories"),
            ("Chew Toy", "", 249, "toy chew ball", "accessories"),
            ("Cat Litter", "5kg", 499, "cat litter sand", "care"),
            ("Pet Shampoo", "200ml", 349, "shampoo pet", "care"),
            ("Dog Grooming", "", 899, "grooming bath haircut", "service", "from"),
            ("Fish Food", "100g", 120, "fish food aquarium", "food"),
        ],
        delivery=True, delivery_days=[1, 2], bulk=False,
        returns=["Unopened food and accessories returnable within 7 days with bill.",
                 "Exchange within 5 days on unused products."],
        extra=[("Grooming slot", "Grooming by appointment, 11 AM – 6 PM.",
                "grooming appointment slot booking")],
    ),
    "cosmetics": dict(
        names=["{s} Cosmetics", "{f} Beauty Store", "{s} Fancy Store", "Glow {s} Cosmetics",
               "{s} Beauty Corner"],
        stock="goods",
        items=[
            ("Face Wash", "100ml", 199, "face wash facewash", "skin"),
            ("Moisturiser", "200ml", 299, "moisturiser moisturizer cream", "skin"),
            ("Sunscreen SPF 50", "50g", 399, "sunscreen sunblock spf", "skin"),
            ("Kajal", "", 149, "kajal kohl eyeliner", "makeup"),
            ("Lipstick", "", 349, "lipstick lip", "makeup"),
            ("Hair Oil", "200ml", 180, "hair oil", "hair"),
            ("Shampoo", "340ml", 320, "shampoo", "hair"),
            ("Body Lotion", "400ml", 350, "body lotion", "skin"),
            ("Nail Polish", "", 99, "nail polish paint", "makeup"),
            ("Perfume", "100ml", 599, "perfume deo scent", "fragrance"),
            ("Bindi Pack", "", 40, "bindi", "accessories"),
        ],
        delivery=True, delivery_days=[1, 2, 3], bulk=False,
        returns=["Unopened products exchangeable within 7 days. No returns on opened cosmetics.",
                 "Sealed products only, within 5 days with bill."],
        extra=[],
    ),
    "coaching": dict(
        names=["{s} Classes", "{s} Coaching Centre", "Bright Future {s} Academy",
               "{s} Tuition Classes", "Excel {s} Institute"],
        stock="course",
        items=[
            ("Class 10 Maths Tuition", "per month", 1500, "class 10 maths math tuition", "school"),
            ("Class 12 Physics Tuition", "per month", 2000, "class 12 physics tuition", "school"),
            ("Spoken English Course", "2 months", 3500, "spoken english course", "skills"),
            ("Computer Basics Course", "1 month", 2500, "computer basics ms office", "skills"),
            ("Tally with GST Course", "2 months", 5500, "tally gst accounting course", "skills"),
            ("Python Programming Course", "2 months", 6000, "python programming coding course", "skills"),
            ("Demo Class", "", 0, "demo trial free class", "trial"),
        ],
        delivery=False, bulk=False,
        returns=["Fees refundable within 7 days of joining, minus ₹500 admission charge.",
                 "No refund after the first week; you may shift to another batch once."],
        extra=[("Batch timings", "Morning batch 7–9 AM, evening batch 5–7 PM, weekend batch Sat–Sun 10 AM–1 PM.",
                "batch timing time morning evening weekend schedule"),
               ("Online classes", "Online classes on Google Meet for all courses at the same fee.",
                "online meet zoom remote")],
    ),
}


# ═══════════════════════════════════════════════════════════════
# BUILDERS
# ═══════════════════════════════════════════════════════════════
def _jitter_price(p, rng, pct=0.12, wide=0.0):
    """Realistic ±pct jitter; with probability `wide`, a much wider range (0.45×–2.2×)
    so the model cannot lean on a remembered 'usual' price and must copy the result."""
    if p == 0:
        return 0
    if wide and rng.random() < wide:
        v = p * math.exp(rng.uniform(math.log(0.45), math.log(2.2)))
    else:
        v = p * (1 + rng.uniform(-pct, pct))
    if v < 20:
        return max(1, round(v))
    if v < 200:
        return int(round(v / 5) * 5) or 5
    if v < 2000:
        v = int(round(v / 10) * 10)
        return v - 1 if rng.random() < 0.4 else v
    v = int(round(v / 50) * 50)
    return v - 1 if rng.random() < 0.5 else v


def _stock(style, rng):
    pool, low = STOCK_STYLES[style]
    if low and rng.random() < 0.1:
        return low.format(n=rng.randint(1, 5))
    return rng.choice(pool)


def build_kb(type_key, rng=None, price_jitter=0.12, offer_rate=0.08, wide=0.0):
    """One merchant knowledge base. Structured terms are kept in kb["store"]
    so the data generator can build DATE / CALCULATE chains from them."""
    rng = rng or random.Random()
    spec = MERCHANT_TYPES[type_key]
    city = rng.choice(CITIES)
    store = dict(
        name=rng.choice(spec["names"]).format(s=rng.choice(SURNAMES), f=rng.choice(FIRST_NAMES)),
        type=type_key, city=city, area=rng.choice(AREAS), hours=rng.choice(HOURS),
        payment=[m for m in PAYMENT_MODES if rng.random() < 0.8] or [PAYMENT_MODES[0]],
        min_order=rng.choice([99, 149, 199, 249]),
        advance_days=rng.choice([1, 2, 3]),
        stitch_days=rng.choice([4, 5, 7]),
        tempo=rng.choice([300, 400, 500]),
    )
    if "UPI (GPay, PhonePe, Paytm)" not in store["payment"]:
        store["payment"].insert(0, "UPI (GPay, PhonePe, Paytm)")
    if not spec["delivery"] or rng.random() < 0.1:
        store.update(delivery_days=None, delivery_charge=None, free_above=None)
    else:
        store.update(delivery_days=rng.choice(spec["delivery_days"]),
                     delivery_charge=rng.choice([20, 30, 40, 50, 60]),
                     free_above=rng.choice([299, 499, 799, 999]))

    catalog = []
    for t in spec["items"]:
        name, variant, base, kws, cat = t[:5]
        flags = t[5:] if len(t) > 5 else ()
        item = dict(name=name, variant=variant, price=_jitter_price(base, rng, price_jitter, wide),
                    stock=_stock(spec["stock"], rng), keywords=kws.split(), category=cat)
        if "from" in flags:
            item["price_from"] = True
        if spec["stock"] == "goods" and item["price"] and rng.random() < offer_rate:
            item["offer"] = rng.choice(["10% off", "5% off", "15% off", "buy 2 get 1 free"])
        catalog.append(item)

    def faq(topic, answer, kw):
        return dict(topic=topic, answer=answer, keywords=kw.split())

    faqs = [
        faq("Store hours", store["hours"], FAQ_KW["hours"]),
        faq("Address", f"{store['name']}, {store['area']}, {city}.", FAQ_KW["address"]),
        faq("Payment", "We accept " + ", ".join(store["payment"]) + ".", FAQ_KW["payment"]),
        faq("Returns", rng.choice(spec["returns"]), FAQ_KW["returns"]),
        faq("GST bill", "All prices include GST. GST bill available on request.", FAQ_KW["gst"]),
    ]
    if spec["delivery"]:
        if store["delivery_days"] is None:
            ans = "No home delivery. Pickup from the shop only."
        elif store["delivery_days"] == 0:
            ans = (f"Same-day home delivery within {city}. ₹{store['delivery_charge']} charge, "
                   f"free above ₹{store['free_above']}.")
        else:
            d = store["delivery_days"]
            ans = (f"Home delivery within {city} in {d} day{'s' if d > 1 else ''}. "
                   f"₹{store['delivery_charge']} charge, free above ₹{store['free_above']}.")
        faqs.append(faq("Delivery", ans, FAQ_KW["delivery"]))
    if spec["bulk"]:
        store["bulk_min"] = rng.choice([2000, 3000, 5000])
        store["bulk_pct"] = rng.choice([5, 8, 10])
        faqs.append(faq("Bulk orders",
                        f"Orders above ₹{store['bulk_min']} get {store['bulk_pct']}% discount.",
                        FAQ_KW["bulk"]))
    for topic, answer, kw in spec["extra"]:
        faqs.append(faq(topic, answer.format(**store), kw))

    return dict(store=store, catalog=catalog, faq=faqs)


def random_kb(rng=None):
    rng = rng or random.Random()
    return build_kb(rng.choice(list(MERCHANT_TYPES)), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="write one KB JSON per merchant type into this directory")
    ap.add_argument("--show", help="print a sample KB for this type")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.show:
        print(json.dumps(build_kb(args.show, rng), indent=2, ensure_ascii=False))
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        for t in MERCHANT_TYPES:
            with open(os.path.join(args.export, f"{t}.json"), "w", encoding="utf-8") as f:
                json.dump(build_kb(t, rng), f, indent=2, ensure_ascii=False)
        print(f"wrote {len(MERCHANT_TYPES)} KBs to {args.export}/")


if __name__ == "__main__":
    main()