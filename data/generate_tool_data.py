"""
Generate tool-use training data for Milo v3.

Product context: WhatsApp Customer Support Agent for Indian MSMEs
Target users: Normal Indian users + SDE1 junior developers
Tools: CALCULATE, SEARCH, DATE, WEATHER, RUN_CODE
Format: <|tool_call|>TOOL(args)<|tool_result|>result<|end_tool|>

v3 changes over v1:
  - ALL math goes through CALCULATE (202M can't do arithmetic)
  - 70/30 tool-to-no-tool ratio (was 50/50)
  - Real-world Indian business context (₹, GST, EMI, discounts)
  - WhatsApp-style short messages, casual/broken English, Hinglish
  - Multi-tool chain examples
  - More diverse question phrasings and edge cases
  - ~10.5K total examples

Usage:
    python data/generate_tool_data.py --output data/tool_use_train.jsonl
"""

import json
import random
import argparse
import os
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def format_tool_call(tool_name, args, result):
    return f"<|tool_call|>{tool_name}({args})<|tool_result|>{result}<|end_tool|>"


def format_example(user_msg, assistant_msg):
    return {
        "text": f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}<|endoftext|>"
    }


def pick(options):
    """Pick a random item from a list."""
    return random.choice(options)


# ═══════════════════════════════════════════════════════════════
# CALCULATE EXAMPLES — ALL math goes through tool
# ═══════════════════════════════════════════════════════════════
def generate_calculate_examples(n=2000):
    examples = []

    # ── Basic arithmetic with diverse phrasings ──
    add_templates = [
        "What is {a} + {b}?", "What's {a} + {b}?", "what {a}+{b}",
        "{a} + {b}", "{a}+{b}=?", "Calculate {a} + {b}",
        "How much is {a} plus {b}?", "Add {a} and {b}",
        "what's {a} plus {b}", "whats {a}+{b}", "{a} aur {b} ka total?",
        "bhai {a} + {b} kitna hota hai", "tell me {a}+{b}",
        "sum of {a} and {b}", "{a} aur {b} jodo", "quick {a}+{b}",
        "hey whats {a} + {b}", "yo {a}+{b}?",
    ]
    for _ in range(200):
        a = random.randint(0, 9999)
        b = random.randint(0, 9999)
        result = a + b
        expr = f"{a} + {b}"
        tc = format_tool_call("CALCULATE", expr, str(result))
        q = pick(add_templates).format(a=a, b=b)
        answers = [
            f"{tc}\n{a} + {b} = {result}.",
            f"{tc}\nThe answer is {result}.",
            f"{tc}\n{result}.",
            f"{tc}\n{a} plus {b} equals {result}.",
        ]
        examples.append(format_example(q, pick(answers)))

    # ── Subtraction ──
    sub_templates = [
        "What is {a} - {b}?", "What's {a} minus {b}?", "{a} - {b}",
        "Subtract {b} from {a}", "what's {a}-{b}", "{a} me se {b} nikalo",
        "How much is {a} minus {b}?", "{a} minus {b} kitna?",
        "difference between {a} and {b}", "{a}-{b}=?",
    ]
    for _ in range(120):
        a = random.randint(1, 99999)
        b = random.randint(0, a)
        result = a - b
        expr = f"{a} - {b}"
        tc = format_tool_call("CALCULATE", expr, str(result))
        q = pick(sub_templates).format(a=a, b=b)
        examples.append(format_example(q, f"{tc}\n{a} - {b} = {result}."))

    # ── Multiplication ──
    mul_templates = [
        "What is {a} × {b}?", "What's {a} times {b}?", "{a} * {b}",
        "Multiply {a} by {b}", "{a} x {b}", "{a}x{b}=?",
        "How much is {a} multiplied by {b}?", "{a} into {b}?",
        "{a} times {b}?", "what's {a}*{b}",
    ]
    for _ in range(120):
        a = random.randint(2, 999)
        b = random.randint(2, 999)
        result = a * b
        expr = f"{a} * {b}"
        tc = format_tool_call("CALCULATE", expr, str(result))
        q = pick(mul_templates).format(a=a, b=b)
        examples.append(format_example(q, f"{tc}\n{a} × {b} = {result:,}."))

    # ── Division ──
    div_templates = [
        "What is {a} / {b}?", "What's {a} divided by {b}?",
        "{a} ÷ {b}", "Divide {a} by {b}", "{a}/{b}?",
        "{a} ko {b} se divide karo",
    ]
    for _ in range(100):
        b = random.randint(2, 100)
        a = b * random.randint(1, 100) + random.randint(0, b - 1)
        result = a / b
        expr = f"{a} / {b}"
        result_str = f"{result:.4f}" if result != int(result) else str(int(result))
        tc = format_tool_call("CALCULATE", expr, result_str)
        q = pick(div_templates).format(a=a, b=b)
        display = f"{result:.2f}" if result != int(result) else str(int(result))
        examples.append(format_example(q, f"{tc}\n{a} ÷ {b} = {display}."))

    # ── Indian business: GST calculations ──
    gst_templates = [
        "What's the GST on ₹{price}?",
        "Calculate {pct}% GST on ₹{price}",
        "₹{price} pe {pct}% GST kitna lagega?",
        "Total with GST for ₹{price} item?",
        "My bill is ₹{price}. Add {pct}% GST please",
        "GST amount on invoice of ₹{price}?",
        "Customer ka bill ₹{price} hai, {pct}% GST add karo",
        "Final price of ₹{price} after {pct}% GST?",
    ]
    for _ in range(120):
        price = pick([99, 149, 199, 249, 299, 399, 499, 599, 799, 999,
                      1299, 1499, 1999, 2499, 2999, 4999, 7999, 9999,
                      12000, 15000, 25000, 50000, 75000, 100000])
        pct = pick([5, 12, 18, 28])
        gst = price * pct / 100
        total = price + gst
        expr = f"{price} * {pct} / 100"
        tc = format_tool_call("CALCULATE", expr, f"{gst:.2f}")
        q = pick(gst_templates).format(price=price, pct=pct)
        a = f"{tc}\nThe {pct}% GST on ₹{price:,} is ₹{gst:,.2f}. Total: ₹{total:,.2f}."
        examples.append(format_example(q, a))

    # ── EMI calculations ──
    for _ in range(80):
        principal = pick([10000, 20000, 30000, 50000, 75000, 100000,
                          150000, 200000, 500000, 1000000])
        rate = pick([8, 9, 10, 11, 12, 14, 16, 18, 24])
        months = pick([3, 6, 9, 12, 18, 24, 36])
        monthly_rate = rate / 12 / 100
        if monthly_rate > 0:
            emi = principal * monthly_rate * (1 + monthly_rate)**months / ((1 + monthly_rate)**months - 1)
        else:
            emi = principal / months
        expr = f"{principal} * {monthly_rate:.6f} * (1 + {monthly_rate:.6f})**{months} / ((1 + {monthly_rate:.6f})**{months} - 1)"
        tc = format_tool_call("CALCULATE", expr, f"{emi:.2f}")
        templates = [
            f"EMI for ₹{principal:,} loan at {rate}% for {months} months?",
            f"₹{principal:,} ka EMI batao, {rate}% interest, {months} months",
            f"Calculate EMI: ₹{principal:,}, {rate}%, {months} months",
            f"₹{principal:,} loan. {rate}% annual. {months} months. Monthly EMI?",
        ]
        q = pick(templates)
        total_paid = emi * months
        interest = total_paid - principal
        a = f"{tc}\nMonthly EMI: ₹{emi:,.2f}. Total payment: ₹{total_paid:,.2f} (₹{interest:,.2f} interest)."
        examples.append(format_example(q, a))

    # ── Tips ──
    for _ in range(100):
        bill = round(random.uniform(100, 25000), 2) if random.random() < 0.6 else round(random.uniform(8, 250), 2)
        pct = pick([5, 10, 15, 18, 20, 25])
        tip = bill * pct / 100
        total = bill + tip
        currency = "₹" if bill > 50 else "$"
        expr = f"{bill} * {pct / 100}"
        tc = format_tool_call("CALCULATE", expr, f"{tip:.4f}")
        templates = [
            f"What's a {pct}% tip on {currency}{bill:.2f}?",
            f"{pct}% tip on {currency}{bill:.2f}?",
            f"Calculate {pct}% tip for {currency}{bill:.2f}",
        ]
        q = pick(templates)
        a = f"{tc}\nA {pct}% tip on {currency}{bill:.2f} is {currency}{tip:.2f}. Total: {currency}{total:.2f}."
        examples.append(format_example(q, a))

    # ── Discounts / sale prices ──
    for _ in range(120):
        price = pick([199, 299, 399, 499, 599, 799, 999, 1299, 1499,
                      1999, 2499, 2999, 3999, 4999, 7999, 9999, 14999,
                      19999, 24999, 49999])
        pct = pick([5, 10, 15, 20, 25, 30, 40, 50, 60, 70])
        saved = price * pct / 100
        result = price - saved
        expr = f"{price} * (1 - {pct}/100)"
        tc = format_tool_call("CALCULATE", expr, f"{result:.4f}")
        templates = [
            f"₹{price:,} item is {pct}% off. Final price?",
            f"Sale price of ₹{price:,} after {pct}% discount?",
            f"₹{price:,} pe {pct}% discount laga ke kitna hoga?",
            f"MRP ₹{price:,}. Giving {pct}% off. What to charge?",
            f"Customer wants {pct}% discount on ₹{price:,}",
            f"{pct}% off on ₹{price:,}?",
        ]
        q = pick(templates)
        a = f"{tc}\nAfter {pct}% discount: ₹{result:,.0f} (saving ₹{saved:,.0f})."
        examples.append(format_example(q, a))

    # ── Unit conversions ──
    conversions = [
        ("How many km is {val} miles?", "{val} * 1.60934", lambda v: v * 1.60934,
         "{val} miles = {result:.2f} km."),
        ("Convert {val} pounds to kg", "{val} * 0.453592", lambda v: v * 0.453592,
         "{val} pounds = {result:.2f} kg."),
        ("What's {val}°C in Fahrenheit?", "{val} * 9/5 + 32", lambda v: v * 9/5 + 32,
         "{val}°C = {result:.1f}°F."),
        ("{val} feet to meters", "{val} * 0.3048", lambda v: v * 0.3048,
         "{val} feet = {result:.2f} meters."),
        ("{val} inches to cm", "{val} * 2.54", lambda v: v * 2.54,
         "{val} inches = {result:.2f} cm."),
        ("Convert {val} USD to INR", "{val} * 83.5", lambda v: v * 83.5,
         "{val} USD ≈ ₹{result:,.2f}."),
        ("{val} kg to pounds", "{val} * 2.20462", lambda v: v * 2.20462,
         "{val} kg = {result:.2f} pounds."),
        ("{val} sq ft to sq meters", "{val} * 0.0929", lambda v: v * 0.0929,
         "{val} sq ft = {result:.2f} sq m."),
        ("{val}°F in Celsius?", "({val} - 32) * 5/9", lambda v: (v - 32) * 5/9,
         "{val}°F = {result:.1f}°C."),
    ]
    for _ in range(100):
        conv = pick(conversions)
        val = round(random.uniform(1, 1000), random.choice([0, 0, 1]))
        val = int(val) if val == int(val) else val
        result = conv[2](val)
        expr = conv[1].format(val=val)
        tc = format_tool_call("CALCULATE", expr, f"{result:.4f}")
        q = conv[0].format(val=val)
        a = f"{tc}\n" + conv[3].format(val=val, result=result)
        examples.append(format_example(q, a))

    # ── Percentage questions ──
    pct_templates = [
        "What's {pct}% of {val}?", "How much is {pct}% of {val}?",
        "{pct} percent of {val}", "{pct}% of {val}?",
        "{val} ka {pct} percent?", "calculate {pct}% of {val}",
    ]
    for _ in range(100):
        val = pick([50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000])
        pct = pick([5, 10, 12, 15, 18, 20, 25, 28, 30, 50, 75])
        result = val * pct / 100
        expr = f"{val} * {pct} / 100"
        result_str = str(int(result)) if result == int(result) else f"{result:.2f}"
        tc = format_tool_call("CALCULATE", expr, result_str)
        q = pick(pct_templates).format(pct=pct, val=val)
        a = f"{tc}\n{pct}% of {val} is {result_str}."
        examples.append(format_example(q, a))

    # ── Profit margin / markup ──
    for _ in range(80):
        cost = pick([50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 5000])
        selling = cost + random.randint(int(cost * 0.1), int(cost * 1.5))
        profit = selling - cost
        margin = (profit / selling) * 100
        expr = f"({selling} - {cost}) / {selling} * 100"
        tc = format_tool_call("CALCULATE", expr, f"{margin:.2f}")
        templates = [
            f"Cost ₹{cost}, sell ₹{selling}. Profit margin?",
            f"₹{cost} pe kharida, ₹{selling} pe becha. Margin?",
            f"My product costs ₹{cost}. Selling at ₹{selling}. Margin?",
        ]
        q = pick(templates)
        a = f"{tc}\nProfit: ₹{profit:,}. Margin: {margin:.1f}%."
        examples.append(format_example(q, a))

    # ── Bill splitting ──
    for _ in range(80):
        total = pick([300, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000, 5000])
        people = random.randint(2, 8)
        per_person = total / people
        expr = f"{total} / {people}"
        tc = format_tool_call("CALCULATE", expr, f"{per_person:.2f}")
        templates = [
            f"Split ₹{total:,} between {people} people",
            f"₹{total:,} ka bill {people} logo me divide karo",
            f"We're {people} friends. Bill ₹{total:,}. How much each?",
        ]
        q = pick(templates)
        display = f"{per_person:.0f}" if per_person == int(per_person) else f"{per_person:.2f}"
        a = f"{tc}\nEach person pays ₹{display}."
        examples.append(format_example(q, a))

    # ── Salary ──
    for _ in range(60):
        monthly = pick([15000, 20000, 25000, 30000, 40000, 50000,
                        60000, 75000, 100000, 150000])
        annual = monthly * 12
        expr = f"{monthly} * 12"
        tc = format_tool_call("CALCULATE", expr, str(annual))
        templates = [
            f"Salary ₹{monthly:,}/month. Annual?",
            f"₹{monthly:,} monthly. Yearly kitna?",
            f"Convert ₹{monthly:,} per month to annual",
        ]
        q = pick(templates)
        a = f"{tc}\nAnnual income: ₹{annual:,}."
        examples.append(format_example(q, a))

    # ── Area ──
    for _ in range(60):
        length = random.randint(5, 100)
        width = random.randint(5, 80)
        area = length * width
        expr = f"{length} * {width}"
        tc = format_tool_call("CALCULATE", expr, str(area))
        templates = [
            f"Area of {length}ft x {width}ft room?",
            f"My shop is {length}x{width} feet. Area?",
            f"{length} x {width} feet ka area?",
        ]
        q = pick(templates)
        a = f"{tc}\nArea: {area:,} sq ft ({area * 0.0929:.1f} sq m)."
        examples.append(format_example(q, a))

    # ── Powers ──
    for _ in range(60):
        base = random.randint(2, 20)
        exp = random.randint(2, 6)
        result = base ** exp
        expr = f"{base} ** {exp}"
        tc = format_tool_call("CALCULATE", expr, str(result))
        templates = [f"{base}^{exp}?", f"What's {base} to the power of {exp}?",
                    f"{base} ka {exp} power?"]
        q = pick(templates)
        a = f"{tc}\n{base}^{exp} = {result:,}."
        examples.append(format_example(q, a))

    # ── SDE1: memory sizing ──
    for _ in range(60):
        n_val = pick([1000, 10000, 100000, 1000000, 10000000, 100000000])
        bytes_per = pick([4, 8, 16, 32])
        result_mb = n_val * bytes_per / 1048576
        expr = f"{n_val} * {bytes_per} / 1048576"
        tc = format_tool_call("CALCULATE", expr, f"{result_mb:.4f}")
        q = f"Memory for {n_val:,} elements × {bytes_per} bytes?"
        a = f"{tc}\nThat's {result_mb:,.2f} MB."
        examples.append(format_example(q, a))

    # ── Multi-step: items + tax ──
    for _ in range(80):
        items = random.randint(2, 5)
        prices = [random.randint(50, 5000) for _ in range(items)]
        total = sum(prices)
        tax_pct = pick([5, 12, 18])
        tax = total * tax_pct / 100
        grand = total + tax
        prices_str = " + ".join(str(p) for p in prices)
        expr = f"({prices_str}) * (1 + {tax_pct}/100)"
        tc = format_tool_call("CALCULATE", expr, f"{grand:.4f}")
        items_desc = ", ".join(f"₹{p}" for p in prices)
        templates = [
            f"Items: {items_desc}. Total with {tax_pct}% GST?",
            f"Bill: {items_desc}. Add {tax_pct}% tax",
        ]
        q = pick(templates)
        a = f"{tc}\nSubtotal: ₹{total:,}. GST ({tax_pct}%): ₹{tax:,.2f}. Total: ₹{grand:,.2f}."
        examples.append(format_example(q, a))

    # ── Edge cases ──
    edge_cases = [
        ("What's 0 + 0?", "0 + 0", "0", "0 + 0 = 0."),
        ("Multiply 0 by 1000", "0 * 1000", "0", "0 × 1000 = 0."),
        ("1 divided by 3?", "1 / 3", "0.3333", "1 ÷ 3 ≈ 0.33 (repeating)."),
        ("100 divided by 0?", "100 / 0", "Error: division by zero",
         "Division by zero is undefined in mathematics."),
        ("999999 + 1?", "999999 + 1", "1000000", "999,999 + 1 = 1,000,000."),
        ("0.1 + 0.2?", "0.1 + 0.2", "0.3000", "0.1 + 0.2 = 0.3."),
    ]
    for q, expr, res, ans in edge_cases:
        tc = format_tool_call("CALCULATE", expr, res)
        for _ in range(10):
            examples.append(format_example(q, f"{tc}\n{ans}"))

    random.shuffle(examples)
    return examples[:n]


# ═══════════════════════════════════════════════════════════════
# SEARCH EXAMPLES
# ═══════════════════════════════════════════════════════════════
def generate_search_examples(n=1800):
    examples = []

    facts = [
        # General knowledge
        ("What is photosynthesis?", "process plants use to convert sunlight into energy",
         "Photosynthesis is how plants convert sunlight, water, and CO2 into glucose and oxygen using chloroplasts."),
        ("How does WiFi work?", "radio waves transmitting data wirelessly",
         "WiFi uses radio waves to transmit data between your device and a router connected to the internet."),
        ("What is inflation?", "general increase in prices over time",
         "Inflation is the rate at which prices rise over time, reducing your purchasing power. Central banks manage it through monetary policy."),
        ("What is DNA?", "deoxyribonucleic acid, genetic instructions",
         "DNA carries the genetic instructions for the development and functioning of all known organisms."),
        ("What is a black hole?", "region of spacetime with extreme gravity",
         "A black hole is a region where gravity is so strong nothing can escape, not even light. They form when massive stars collapse."),
        ("How does encryption work?", "converting data into coded form",
         "Encryption converts readable data into an unreadable format using a key. Only someone with the decryption key can read it."),
        ("What causes earthquakes?", "tectonic plate movement",
         "Earthquakes happen when tectonic plates suddenly release built-up stress, sending seismic waves through the Earth."),
        ("What is blockchain?", "distributed ledger technology",
         "Blockchain is a distributed ledger that records transactions across many computers, making records nearly impossible to alter."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci, 1503-1519",
         "Leonardo da Vinci painted the Mona Lisa, starting around 1503. It hangs in the Louvre Museum in Paris."),
        ("Speed of light?", "299,792,458 m/s",
         "The speed of light is approximately 299,792,458 meters per second (186,282 miles per second)."),

        # Tech/SDE1
        ("What is a REST API?", "web services using HTTP methods",
         "A REST API uses HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources. It's stateless and cacheable."),
        ("What is Docker?", "platform for containerizing apps",
         "Docker packages applications and dependencies into containers — lightweight units that run consistently across environments."),
        ("What is Big O notation?", "algorithm complexity description",
         "Big O describes how an algorithm's performance scales. Common: O(1) constant, O(n) linear, O(n²) quadratic."),
        ("What is Git rebase?", "moves commits onto a new base",
         "Git rebase re-applies your commits on top of another branch's tip, creating a linear history instead of merge commits."),
        ("What is a load balancer?", "distributes traffic across servers",
         "A load balancer distributes incoming requests across multiple servers so no single server gets overwhelmed."),
        ("What are WebSockets?", "full-duplex communication protocol",
         "WebSockets enable real-time, two-way communication over a single TCP connection, unlike HTTP's request-response model."),
        ("What is CORS?", "browser security for cross-origin requests",
         "CORS restricts web pages from making requests to different domains. Servers use headers to specify which origins are allowed."),
        ("What is SQL injection?", "database attack via malicious input",
         "SQL injection inserts malicious SQL through input fields. Prevent it with parameterized queries and input validation."),
        ("What is Kubernetes?", "container orchestration platform",
         "Kubernetes automates deploying, scaling, and managing containerized applications across clusters of machines."),
        ("What is OAuth?", "authorization framework for third-party access",
         "OAuth lets apps get limited access to a user's account without exposing their password. Used for 'Sign in with Google' etc."),
        ("What is Redis?", "in-memory data store and cache",
         "Redis is an in-memory data store used as a database, cache, and message broker. Lightning fast for reads."),
        ("What is CI/CD?", "continuous integration and deployment",
         "CI/CD automates building, testing, and deploying code. CI catches bugs early, CD ships changes faster."),
        ("What is a CDN?", "content delivery network",
         "A CDN serves web content from geographically distributed servers, reducing latency for users worldwide."),
        ("What is TCP/IP?", "internet communication protocols",
         "TCP/IP is the internet's protocol suite. TCP handles reliable data delivery, IP handles addressing and routing."),
        ("What is a hash table?", "key-value data structure",
         "A hash table maps keys to values using a hash function. Average O(1) for lookup, insert, and delete."),
        ("What is GraphQL?", "query language for APIs",
         "GraphQL lets clients request exactly the data they need in a single query, reducing over-fetching vs REST."),
        ("What is MongoDB?", "NoSQL document database",
         "MongoDB stores data in flexible JSON-like documents, good for unstructured or semi-structured data."),
        ("What is JWT?", "JSON Web Token for auth",
         "JWT is a compact token for securely transmitting info between parties. Commonly used for authentication."),

        # Indian context
        ("What is UPI?", "India's instant payment system",
         "UPI (Unified Payments Interface) enables instant bank-to-bank transfers via apps like Google Pay, PhonePe, and Paytm."),
        ("What is GST in India?", "Goods and Services Tax",
         "GST is India's indirect tax with four slabs: 5%, 12%, 18%, and 28%. It replaced multiple taxes in 2017."),
        ("What is Aadhaar?", "12-digit unique identity number",
         "Aadhaar is a 12-digit unique ID issued by UIDAI to Indian residents, used for identity verification."),
        ("What is MSME?", "Micro, Small, Medium Enterprises",
         "MSME stands for Micro, Small and Medium Enterprises. India has 63M+ MSMEs employing 110M+ people."),
        ("What is NPCI?", "National Payments Corporation of India",
         "NPCI operates India's retail payment systems including UPI, IMPS, RuPay, and BHIM."),
    ]

    prefixes = [
        "", "Can you tell me ", "Explain ", "Tell me about ",
        "Help me understand ", "Quick question: ", "Hey, ",
        "bhai ", "yaar ", "please explain ", "What is ",
    ]

    for _ in range(n):
        question, search_summary, answer = pick(facts)
        prefix = pick(prefixes)
        if prefix and not prefix.endswith(("?", " ")):
            q = prefix + " " + question[0].lower() + question[1:]
        elif prefix:
            q = prefix + question[0].lower() + question[1:]
        else:
            q = question
        tc = format_tool_call("SEARCH", question.rstrip("?"), search_summary)
        a = f"{tc}\n{answer}"
        examples.append(format_example(q, a))

    random.shuffle(examples)
    return examples[:n]


# ═══════════════════════════════════════════════════════════════
# DATE EXAMPLES
# ═══════════════════════════════════════════════════════════════
def generate_date_examples(n=1200):
    examples = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    date_questions = [
        "What's today's date?", "What day is it?", "date?",
        "today's date", "aaj kya date hai?", "aaj kaun sa din hai?",
        "What day of the week is it?", "current date",
        "What's the date today?", "Tell me today's date",
    ]

    for _ in range(500):
        q = pick(date_questions)
        rand_date = datetime(2026, random.randint(1, 12), random.randint(1, 28))
        day_name = days[rand_date.weekday()]
        date_str = rand_date.strftime("%B %d, %Y")
        tc = format_tool_call("DATE", "", f"{day_name}, {date_str}")
        responses = [
            f"{tc}\nToday is {day_name}, {date_str}.",
            f"{tc}\nIt's {day_name}, {date_str}.",
        ]
        examples.append(format_example(q, pick(responses)))

    # Events (Indian + global)
    events = [
        ("Diwali", "October/November"), ("Holi", "March"),
        ("Christmas", "December 25"), ("New Year's", "January 1"),
        ("Republic Day", "January 26"), ("Independence Day", "August 15"),
        ("Ganesh Chaturthi", "August/September"),
        ("Navratri", "October"), ("Raksha Bandhan", "August"),
        ("Valentine's Day", "February 14"), ("Halloween", "October 31"),
    ]

    for _ in range(400):
        event, event_date = pick(events)
        days_count = random.randint(5, 300)
        tc = format_tool_call("DATE", "", "Friday, October 10, 2026")
        templates = [
            f"How many days until {event}?",
            f"When is {event}?",
            f"{event} kab hai?",
            f"{event} kitne din me hai?",
        ]
        q = pick(templates)
        a = f"{tc}\n{event} ({event_date}) is approximately {days_count} days away."
        examples.append(format_example(q, a))

    for _ in range(300):
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        month = pick(months)
        day = random.randint(1, 28)
        year = pick([2024, 2025, 2026, 2027])
        date_str = f"{month} {day}, {year}"
        day_name = pick(days)
        tc = format_tool_call("DATE", date_str, f"{day_name}, {date_str}")
        templates = [
            f"What day is {date_str}?",
            f"What day of the week was {date_str}?",
            f"{date_str} ko kaun sa din tha?",
        ]
        q = pick(templates)
        a = f"{tc}\n{date_str} falls on a {day_name}."
        examples.append(format_example(q, a))

    random.shuffle(examples)
    return examples[:n]


# ═══════════════════════════════════════════════════════════════
# WEATHER EXAMPLES
# ═══════════════════════════════════════════════════════════════
def generate_weather_examples(n=1200):
    examples = []

    indian_cities = [
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata",
        "Pune", "Jaipur", "Lucknow", "Ahmedabad", "Kochi", "Goa",
        "Chandigarh", "Indore", "Bhopal", "Nagpur", "Varanasi", "Patna",
        "Surat", "Coimbatore", "Vizag", "Mangalore", "Mysore", "Shimla",
        "Dehradun", "Udaipur", "Agra", "Noida", "Gurgaon", "Bhubaneswar",
    ]
    global_cities = [
        "New York", "San Francisco", "London", "Tokyo", "Paris",
        "Sydney", "Singapore", "Dubai", "Toronto", "Seoul",
    ]

    conditions = [
        ("sunny", "clear skies", 28, 42),
        ("partly cloudy", "some clouds", 22, 35),
        ("cloudy", "overcast", 18, 30),
        ("rainy", "light rain", 15, 28),
        ("heavy rain", "thunderstorms", 14, 26),
        ("hot and humid", "high humidity", 32, 45),
        ("pleasant", "comfortable", 20, 28),
        ("hazy", "smog and haze", 18, 32),
        ("foggy", "low visibility", 8, 18),
    ]

    templates = [
        "Weather in {city}?", "How's the weather in {city}?",
        "{city} weather", "{city} ka mausam?",
        "Is it raining in {city}?", "Temperature in {city}?",
        "Should I carry umbrella in {city}?",
        "Is it hot in {city}?", "{city} me aaj kaisa mausam hai?",
        "Do I need jacket in {city}?", "What's the weather like in {city}?",
    ]

    for _ in range(n):
        city = pick(indian_cities) if random.random() < 0.65 else pick(global_cities)
        condition, desc, low, high = pick(conditions)
        temp = random.randint(low, high)
        humidity = random.randint(30, 95)
        wind = random.randint(5, 40)

        weather_result = f"{temp}°C, {condition}, humidity {humidity}%, wind {wind} km/h"
        tc = format_tool_call("WEATHER", city, weather_result)
        q = pick(templates).format(city=city)

        if "umbrella" in q.lower() or "raining" in q.lower():
            if "rain" in condition:
                a = f"{tc}\nYes, it's {condition} in {city} ({temp}°C). Carry an umbrella!"
            else:
                a = f"{tc}\nIt's {condition} in {city} ({temp}°C). No umbrella needed."
        elif "jacket" in q.lower():
            if temp < 18:
                a = f"{tc}\n{temp}°C in {city}. Yes, bring a jacket."
            else:
                a = f"{tc}\n{temp}°C in {city}, {condition}. No jacket needed."
        elif "hot" in q.lower():
            if temp > 30:
                a = f"{tc}\nYes, it's hot — {temp}°C in {city}, {desc}."
            else:
                a = f"{tc}\nNot too hot. {temp}°C in {city}, {condition}."
        else:
            a = f"{tc}\n{city}: {temp}°C, {condition}. Humidity {humidity}%, wind {wind} km/h."

        examples.append(format_example(q, a))

    random.shuffle(examples)
    return examples[:n]


# ═══════════════════════════════════════════════════════════════
# RUN_CODE EXAMPLES
# ═══════════════════════════════════════════════════════════════
def generate_code_examples(n=1000):
    examples = []

    code_tasks = [
        ("Squares from 1 to 10", "[x**2 for x in range(1, 11)]",
         "[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]", "Squares from 1 to 10."),
        ("Even numbers 1 to 20", "[x for x in range(1, 21) if x % 2 == 0]",
         "[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]", "Even numbers from 1 to 20."),
        ("Reverse 'hello world'", "'hello world'[::-1]",
         "dlrow olleh", "Reversed: 'dlrow olleh'."),
        ("Count vowels in 'programming'", "sum(1 for c in 'programming' if c in 'aeiou')",
         "3", "'programming' has 3 vowels."),
        ("Factorial of 10", "import math; math.factorial(10)",
         "3628800", "10! = 3,628,800."),
        ("Is 97 prime?", "all(97 % i != 0 for i in range(2, int(97**0.5) + 1))",
         "True", "Yes, 97 is prime."),
        ("GCD of 48 and 18", "import math; math.gcd(48, 18)",
         "6", "GCD of 48 and 18 is 6."),
        ("Sort [3,1,4,1,5,9,2,6]", "sorted([3, 1, 4, 1, 5, 9, 2, 6])",
         "[1, 1, 2, 3, 4, 5, 6, 9]", "Sorted list."),
        ("Remove duplicates from [1,2,2,3,3,4]", "sorted(set([1, 2, 2, 3, 3, 4]))",
         "[1, 2, 3, 4]", "Duplicates removed: [1, 2, 3, 4]."),
        ("First 10 Fibonacci numbers",
         "a, b = 0, 1; fib = []\nfor _ in range(10):\n    fib.append(a)\n    a, b = b, a + b\nprint(fib)",
         "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]", "First 10 Fibonacci numbers."),
        ("255 in binary", "bin(255)", "0b11111111", "255 = 11111111 in binary."),
        ("FizzBuzz 1-15",
         "['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else i for i in range(1,16)]",
         "[1, 2, 'Fizz', 4, 'Buzz', 'Fizz', 7, 8, 'Fizz', 'Buzz', 11, 'Fizz', 13, 14, 'FizzBuzz']",
         "FizzBuzz from 1 to 15."),
        ("Flatten [[1,2],[3,4],[5,6]]", "[x for sub in [[1,2],[3,4],[5,6]] for x in sub]",
         "[1, 2, 3, 4, 5, 6]", "Flattened: [1, 2, 3, 4, 5, 6]."),
        ("Is 'racecar' a palindrome?", "'racecar' == 'racecar'[::-1]",
         "True", "Yes, 'racecar' is a palindrome!"),
        ("Primes under 50",
         "[n for n in range(2, 50) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]",
         "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]",
         "Primes under 50: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47."),
        ("Word count in 'the cat sat on the mat'",
         "from collections import Counter; dict(Counter('the cat sat on the mat'.split()))",
         "{'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1}",
         "'the' appears twice, rest once each."),
        ("2^32", "2**32", "4294967296", "2³² = 4,294,967,296."),
        ("Generate random 8-char password",
         "import random, string; ''.join(random.choices(string.ascii_letters + string.digits + '!@#$', k=8))",
         "kR3#mP9x", "Random password: kR3#mP9x. Use a password manager for real passwords."),
    ]

    prefixes = ["", "Can you ", "Please ", "Run: ", "Quick, ", "Hey ", "Help me "]

    for _ in range(n):
        question, code, output, explanation = pick(code_tasks)
        prefix = pick(prefixes)
        q = (prefix + question[0].lower() + question[1:]) if prefix else question
        tc = format_tool_call("RUN_CODE", code, output)
        a = f"{tc}\n{explanation}"
        examples.append(format_example(q, a))

    random.shuffle(examples)
    return examples[:n]


# ═══════════════════════════════════════════════════════════════
# NO-TOOL EXAMPLES (direct answers)
# ═══════════════════════════════════════════════════════════════
def generate_no_tool_examples(n=3000):
    """Direct answers WITHOUT tools. ONLY greetings, opinions, concepts, advice."""
    examples = []

    greetings = [
        ("Hi!", "Hi there! How can I help you today?"),
        ("Hello", "Hello! What can I do for you?"),
        ("Hey", "Hey! What's up?"),
        ("hey there", "Hey! How can I help?"),
        ("Good morning!", "Good morning! What would you like to do?"),
        ("Good evening", "Good evening! How can I help?"),
        ("What's up?", "Ready to help! What do you need?"),
        ("Thanks!", "You're welcome!"),
        ("Thank you!", "Happy to help! Ask anytime."),
        ("Bye!", "Goodbye! Have a great day!"),
        ("yo", "Hey! What can I help with?"),
        ("namaste", "Namaste! How can I help you?"),
        ("kaise ho?", "Main theek hoon! Aap batao, kya help chahiye?"),
        ("hi milo", "Hi! I'm Milo. How can I help?"),
        ("ok thanks", "Anytime! Let me know if you need more."),
        ("got it", "Great! Ask if you have more questions."),
        ("nice", "Glad you liked it! Anything else?"),
        ("cool", "Thanks! What else can I do?"),
        ("sorry", "No need! How can I help?"),
        ("can you help me?", "Of course! What do you need?"),
        ("are you there?", "Yes, I'm here! What do you need?"),
        ("hmm", "Take your time! I'm here when you're ready."),
        ("nvm", "No worries! Let me know if something comes up."),
        ("thx", "Welcome! Anything else?"),
        ("ty", "You're welcome!"),
    ]

    concepts = [
        ("What does API stand for?",
         "API = Application Programming Interface. It's how different software communicates with each other."),
        ("Difference between == and === in JavaScript?",
         "== checks equality with type coercion ('5' == 5 is true). === checks strict equality ('5' === 5 is false). Use === by default."),
        ("What does HTML stand for?",
         "HTML = HyperText Markup Language. It's the standard language for creating web pages."),
        ("What's a variable?",
         "A variable is a named container for storing data — numbers, text, objects. Like a labeled box."),
        ("What's a for loop?",
         "A for loop repeats code a set number of times. In Python: 'for i in range(5)' runs code 5 times."),
        ("What is an array?",
         "An array stores elements in order, accessed by index (starting from 0). In Python it's called a list."),
        ("What's a function?",
         "A function is a reusable code block with a name. Define once, call whenever needed."),
        ("What is recursion?",
         "Recursion = a function calling itself. Needs a base case (stop condition) and recursive case (break it down)."),
        ("GET vs POST?",
         "GET retrieves data (visible in URL). POST sends data in the body (not cached, no size limit). GET to read, POST to write."),
        ("What is JSON?",
         "JSON = JavaScript Object Notation. Lightweight data format using key-value pairs. Standard for API responses."),
        ("What does sudo mean?",
         "sudo = 'superuser do'. Runs commands with admin privileges on Linux/Mac."),
        ("HTTP vs HTTPS?",
         "HTTPS encrypts data between browser and server using TLS/SSL. The S = Secure. Always use HTTPS."),
        ("Should I learn Python or JavaScript?",
         "Python for data science/AI/automation. JavaScript for web dev. Pick based on what you want to build."),
        ("Best way to learn programming?",
         "Build projects. Start small (calculator, to-do app) and grow. Real learning = solving real problems."),
        ("Tabs or spaces?",
         "The eternal debate! Python recommends 4 spaces. Key thing: be consistent in your project."),
        ("What is a server?",
         "A server is a powerful computer always connected to the internet. When you visit a site, it sends you the page."),
        ("What happens when you type a URL?",
         "DNS lookup → TCP connection → HTTP request → server responds → browser renders the page."),
        ("I'm stuck on a bug",
         "Let's debug together! Share the code and describe what's happening vs expected. Error messages help a lot."),
        ("Can you help with code?",
         "Of course! Share what you're working on and the issue. I'll help figure it out."),
        ("What is OOP?",
         "Object-Oriented Programming organizes code around objects (data + behavior) instead of functions. Classes are the blueprints."),
        ("What's a database index?",
         "An index speeds up database reads, like a book's index. Tradeoff: extra storage and slower writes."),
        ("Is SQL worth learning?",
         "Absolutely. Almost every app uses a database, and SQL has been the standard for decades."),
        ("What's a design pattern?",
         "Proven solutions to common coding problems. Examples: Singleton, Observer, Factory, Strategy."),
        ("How do I get my first customers?",
         "Start with your network. Give 5-10 people free access, get feedback, then use their success stories to attract paying customers."),
        ("What is product-market fit?",
         "When your product solves a real problem for people willing to pay. You know you have it when users recommend it."),
        ("Should I build an MVP first?",
         "Yes. An MVP tests your idea with real users before investing months. Build the smallest thing that delivers value."),
    ]

    about_milo = [
        ("Who are you?",
         "I'm Milo, a small language model. I can do calculations, look things up, check weather, and more!"),
        ("What can you do?",
         "I can calculate, search, check dates, get weather info, and run code. Just ask!"),
        ("Are you AI?",
         "Yes! I'm Milo, a small AI assistant designed to be fast and helpful for everyday tasks."),
        ("Who made you?",
         "I was built from scratch as a small, efficient language model. Designed to help with everyday tasks."),
    ]

    for q, a in greetings:
        for _ in range(15):
            examples.append(format_example(q, a))

    for q, a in concepts:
        for _ in range(30):
            examples.append(format_example(q, a))

    for q, a in about_milo:
        for _ in range(25):
            examples.append(format_example(q, a))

    random.shuffle(examples)
    return examples[:n]


# ═══════════════════════════════════════════════════════════════
# MULTI-TOOL CHAIN EXAMPLES
# ═══════════════════════════════════════════════════════════════
def generate_multi_tool_examples(n=300):
    """Examples needing multiple tool calls."""
    examples = []

    # Search + Calculate
    for _ in range(60):
        tc1 = format_tool_call("SEARCH", "gold price per gram India",
                               "Gold is approximately ₹6,200 per gram")
        tc2 = format_tool_call("CALCULATE", "6200 * 10", "62000")
        q = pick(["10 gram gold ka price?", "What would 10g gold cost?",
                  "Cost of 10 grams gold?"])
        a = f"{tc1}\n{tc2}\nAt ~₹6,200/gram, 10 grams costs approximately ₹62,000."
        examples.append(format_example(q, a))

    # Weather + Date
    for _ in range(60):
        city = pick(["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"])
        temp = random.randint(15, 40)
        condition = pick(["sunny", "rainy", "cloudy", "hot and humid"])
        tc1 = format_tool_call("WEATHER", city, f"{temp}°C, {condition}, humidity 75%")
        tc2 = format_tool_call("DATE", "", "Saturday, October 10, 2026")
        q = f"Should I plan outdoor event in {city} this weekend?"
        if "rain" in condition or temp > 38:
            a = f"{tc1}\n{tc2}\nI'd suggest indoor — it's {condition}, {temp}°C."
        else:
            a = f"{tc1}\n{tc2}\nLooks good! {condition} at {temp}°C. Fine for outdoor."
        examples.append(format_example(q, a))

    # Calculate + Calculate (discount + GST)
    for _ in range(80):
        price = pick([999, 1999, 4999, 9999])
        disc = pick([10, 20, 30])
        gst = pick([12, 18])
        discounted = price * (1 - disc / 100)
        final = discounted * (1 + gst / 100)
        tc1 = format_tool_call("CALCULATE", f"{price} * (1 - {disc}/100)", f"{discounted:.2f}")
        tc2 = format_tool_call("CALCULATE", f"{discounted:.2f} * (1 + {gst}/100)", f"{final:.2f}")
        q = f"₹{price:,} item, {disc}% discount, then {gst}% GST?"
        a = f"{tc1}\n{tc2}\nAfter {disc}% off: ₹{discounted:,.2f}. With {gst}% GST: ₹{final:,.2f}."
        examples.append(format_example(q, a))

    # Code + explain
    for _ in range(60):
        tc = format_tool_call("RUN_CODE",
                              "sorted([('Alice', 85), ('Bob', 92), ('Charlie', 78)], key=lambda x: x[1], reverse=True)",
                              "[('Bob', 92), ('Alice', 85), ('Charlie', 78)]")
        q = pick(["Sort students by marks: Alice 85, Bob 92, Charlie 78",
                  "Rank: Alice=85, Bob=92, Charlie=78",
                  "Who scored highest? Alice 85, Bob 92, Charlie 78"])
        a = f"{tc}\nRanked: 1st Bob (92), 2nd Alice (85), 3rd Charlie (78)."
        examples.append(format_example(q, a))

    # Date + Calculate (business)
    for _ in range(40):
        days = random.randint(5, 30)
        tc1 = format_tool_call("DATE", "", "Friday, October 10, 2026")
        biz = days - (days // 7 * 2)
        tc2 = format_tool_call("CALCULATE", f"{days} - ({days} // 7 * 2)", str(biz))
        q = f"Business days in next {days} days?"
        a = f"{tc1}\n{tc2}\nApproximately {biz} business days."
        examples.append(format_example(q, a))

    random.shuffle(examples)
    return examples[:n]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate tool-use training data for Milo v3")
    parser.add_argument("--output", type=str, default="data/tool_use_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("=" * 60)
    print("  Milo — Tool-Use Training Data v3")
    print("  Product: WhatsApp Support Agent for Indian MSMEs")
    print("=" * 60)
    print()

    calc = generate_calculate_examples(2000)
    print(f"  CALCULATE:  {len(calc)} examples")

    search = generate_search_examples(1800)
    print(f"  SEARCH:     {len(search)} examples")

    date = generate_date_examples(1200)
    print(f"  DATE:       {len(date)} examples")

    weather = generate_weather_examples(1200)
    print(f"  WEATHER:    {len(weather)} examples")

    code = generate_code_examples(1000)
    print(f"  RUN_CODE:   {len(code)} examples")

    multi = generate_multi_tool_examples(300)
    print(f"  Multi-tool: {len(multi)} examples")

    no_tool = generate_no_tool_examples(3000)
    print(f"  No-tool:    {len(no_tool)} examples")

    all_examples = calc + search + date + weather + code + multi + no_tool
    random.shuffle(all_examples)

    total = len(all_examples)
    tool_count = len(calc) + len(search) + len(date) + len(weather) + len(code) + len(multi)

    print(f"\n  Total:      {total} examples")
    print(f"  Tool:       {tool_count} ({tool_count/total*100:.0f}%)")
    print(f"  No-tool:    {len(no_tool)} ({len(no_tool)/total*100:.0f}%)")

    split = int(total * 0.95)
    train = all_examples[:split]
    val = all_examples[split:]

    train_path = args.output
    val_path = args.output.replace("train", "val")

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"\n  Train:      {len(train)} → {train_path}")
    print(f"  Val:        {len(val)} → {val_path}")

    print(f"\n{'='*60}")
    print("  Samples:")
    print(f"{'='*60}")
    for ex in random.sample(all_examples, 4):
        print(f"\n{ex['text']}\n{'─'*50}")


if __name__ == "__main__":
    main()