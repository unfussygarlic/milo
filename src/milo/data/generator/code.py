"""RUN_CODE families - real execution, including the disabled-tool case."""

import json

from milo.data.generator.core import Ex, L, R, Skip, by_lang, fill, is_hi, new_now, pick

# ═══════════════════════════════════════════════════════════════
# FAMILY: RUN_CODE (real execution)
# ═══════════════════════════════════════════════════════════════
WORDS = ["hello world", "milo rocks", "namaste india", "python is fun", "whatsapp bot", "small model",
         "chai and code", "debug karo", "racecar", "level", "madam", "kirana store"]


def fam_code():
    new_now(); lang = L(); k = R.randrange(14)
    if k == 0:
        xs = R.sample(range(1, 200), R.randint(5, 9)); rev = R.random() < 0.3
        user, _ = fill(pick(["sort {x}" + (" descending" if rev else ""), "{x} ko sort karo" + (" ulta" if rev else "")]), lang, x=xs)
        code = f"sorted({xs}{', reverse=True' if rev else ''})"
    elif k == 1:
        w = pick(WORDS); user, _ = fill(pick(["reverse '{w}'", "'{w}' ulta karo"]), lang, w=w); code = f"'{w}'[::-1]"
    elif k == 2:
        w = pick(WORDS); user, _ = fill(pick(["is '{w}' a palindrome?", "'{w}' palindrome hai?"]), lang, w=w)
        code = f"s = '{w}'.replace(' ', '')\ns == s[::-1]"
    elif k == 3:
        n = R.randint(20, 120); user, _ = fill(pick(["primes up to {n}", "{n} tak ke prime numbers"]), lang, n=n)
        code = f"[n for n in range(2, {n} + 1) if all(n % d for d in range(2, int(n ** 0.5) + 1))]"
    elif k == 4:
        n = R.randint(5, 20); user, _ = fill(pick(["factorial of {n}", "{n} ka factorial"]), lang, n=n)
        code = f"import math\nmath.factorial({n})"
    elif k == 5:
        n = R.randint(6, 20); user, _ = fill(pick(["first {n} fibonacci numbers", "{n} fibonacci numbers do"]), lang, n=n)
        code = f"fib = [0, 1]\nwhile len(fib) < {n}:\n    fib.append(fib[-1] + fib[-2])\nfib[:{n}]"
    elif k == 6:
        xs = [R.randint(1, 9) for _ in range(R.randint(6, 10))]
        user, _ = fill(pick(["remove duplicates from {x}", "{x} me se duplicates hatao"]), lang, x=xs)
        code = f"list(dict.fromkeys({xs}))"
    elif k == 7:
        s = pick(["the cat sat on the mat the end", "aaj ka order aaj ka bill", "to be or not to be", "milo milo code code code"])
        user, _ = fill(pick(["word count in '{s}'", "'{s}' me har word kitni baar aaya?"]), lang, s=s)
        code = f"from collections import Counter\ndict(Counter('{s}'.split()))"
    elif k == 8:
        n = R.randint(10, 5000); base = pick(["bin", "hex", "oct"])
        user, _ = fill(pick(["{n} in {b}", "convert {n} to {b}"]), lang, n=n, b={"bin": "binary", "hex": "hex", "oct": "octal"}[base])
        code = f"{base}({n})"
    elif k == 9:
        name = pick(["Basmati Rice", "Kurti", "Power Bank"]); price = R.randint(99, 2999)
        js = json.dumps({"item": name, "price": price, "stock": R.randint(0, 50)})
        user, _ = fill(pick(["get price from this json: {j}", "is json me price kya hai: {j}"]), lang, j=js)
        code = f"import json\njson.loads('{js}')['price']"
    elif k == 10:
        emails = ", ".join(f"{pick(['ravi', 'neha', 'support', 'orders', 'asha'])}{R.randint(1, 99)}@example.com" for _ in range(2))
        text = f"contact {emails.split(', ')[0]} or {emails.split(', ')[1]} for help"
        user, _ = fill(pick(["extract emails from: {t}", "isme se emails nikalo: {t}"]), lang, t=text)
        code = f"import re\nre.findall(r'[\\w.]+@[\\w.]+', '{text}')"
    elif k == 11:
        n = R.randint(100, 999999); user, _ = fill(pick(["sum of digits of {n}", "{n} ke digits ka sum"]), lang, n=n)
        code = f"sum(int(d) for d in str({n}))"
    elif k == 12:   # SDE debugging: run their buggy code
        bug = pick([
            ("nums = [1, 2, 3]\nprint(nums[3])", "IndexError", "list has only 3 items, index 3 is out of range — use nums[2] or nums[-1]"),
            ("d = {'a': 1}\nprint(d['b'])", "KeyError", "key 'b' doesn't exist — use d.get('b') or check 'b' in d"),
            ("print('total: ' + 5)", "TypeError", "you can't add a str and an int — use str(5) or an f-string"),
            ("x = 10\nprint(x / 0)", "ZeroDivisionError", "you're dividing by zero — check the divisor first"),
            ("print(totl)", "NameError", "'totl' isn't defined — probably a typo for 'total'"),
            ("int('12a')", "ValueError", "'12a' isn't a valid integer — clean the string before converting"),
        ])
        user, _ = fill(pick(["why does this fail?\n{c}", "ye code error de raha hai:\n{c}", "debug this:\n{c}"]), lang, c=bug[0])
        ex = Ex("code_debug", user)
        r = ex.tool("RUN_CODE", bug[0])
        if bug[1] not in r:
            raise Skip("debug")
        return ex.done(f"{bug[1]} aa raha hai: {bug[2]}." if is_hi(user) else f"It raises {bug[1]}: {bug[2]}.")
    else:
        xs = [R.randint(10, 999) for _ in range(R.randint(4, 8))]
        user, _ = fill(pick(["max, min and sum of {x}", "{x} ka max min sum"]), lang, x=xs)
        code = f"xs = {xs}\n(max(xs), min(xs), sum(xs))"
    ex = Ex("code", user)
    r = ex.tool("RUN_CODE", code)
    if r.startswith("Error"):
        raise Skip("code error")
    a = pick([f"Result: {r}", f"Ye raha: {r}", f"{r}"]) if is_hi(user) else pick([f"Result: {r}", f"Here you go: {r}", f"{r}", f"Output: {r}"])
    return ex.done(a)


def fam_code_disabled():
    new_now(); lang = L()
    code = pick(["print(sum(range(10)))", "print(2**10)", "sorted([3, 1, 2])", "len('milo')"])
    user, _ = fill(by_lang(lang, ["ye code chala ke batao: {c}", "ye run karo: {c}"], ["run this: {c}", "execute: {c}"]), lang, c=code)
    ex = Ex("code_disabled", user)
    ex.fake("RUN_CODE", code, "Error: RUN_CODE is disabled")
    return ex.done("Sorry, is chat me code run karne ki suvidha nahi hai." if lang == "hi"
                   else "Sorry, running code isn't available in this chat.")
