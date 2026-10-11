"""The decoding constraint, token by token (invariant 4).

``ok(text_so_far, candidate_token)`` answers "may the model write this next?".
Each case below is a rule the training data always follows, so the model is never
blocked from writing what it learned - that half of the contract is checked by
tests/test_constraint_replay.py and tests/test_constraint_fuzz.py.
"""

import milo.tools as T
from milo.agent.constraint import CopyConstraint

TC = "<|tool_call|>MQ_SEARCH("
WT = "<|tool_call|>WEATHER("
CT = "<|tool_call|>CALCULATE("
DT = "<|tool_call|>DATE("


def test_copy_constraint():
    """One sequence, in the original order: CopyConstraint carries per-call state."""
    cc = CopyConstraint("do u guys have sarson oil 1 litre",
                          ["do u guys have sarson oil 1 litre", "Mustard Oil 1l | ₹185 incl. GST | in stock"], True)
    assert not (cc.ok(TC, "ub")), "cc garbled arg blocked"
    assert cc.ok(TC + "sarson", " oil"), "cc copied arg ok"
    assert cc.ok(TC + "sarson oil", ")"), "cc close ok"
    assert not (cc.ok(TC, ")")), "cc empty close blocked"
    cn = CopyConstraint("5 kilo basmati", ["5 kilo basmati", "Basmati Rice 5kg | ₹709 incl. GST", "465"], True)
    assert not (cn.ok("Rate ₹7", "99")), "cc wrong digit blocked"
    assert cn.ok("Rate ₹7", "09"), "cc right digit ok"
    assert not (cn.ok("Rate ₹70", " hai")), "cc truncated number blocked"
    assert cn.ok("Rate ₹709", "."), "cc full stop ok"
    assert not (cn.ok("Total: ₹", "949")), "cc invented total blocked"
    assert cn.ok("Ye ", "5"), "cc single digit ok"
    cf = CopyConstraint("loan 200000 ka 12% 24 mahine", ["loan 200000 ka 12% 24 mahine"], False)
    assert cf.ok("<|tool_call|>CALCULATE(emi(", "200000"), "cc calc arg grounded"
    assert not (cf.ok("<|tool_call|>CALCULATE(emi(", "250000")), "cc calc arg invented"
    assert cf.ok("GST is ", "18"), "cc no-tool short number free"
    co = CopyConstraint("order no 45821 kahan hai?", ["order no 45821 kahan hai?"], False)
    assert co.ok("Order #", "458"), "cc no-tool order id copied"
    assert not (co.ok("Order #", "319")), "cc no-tool order id invented"


    cb = CopyConstraint("bhains ka doodh milta hai?", ["bhains ka doodh milta hai?"], False)
    assert not (cb.ok(TC, "hains")), "cc mid-word start blocked"
    assert cb.ok(TC, "bhains"), "cc whole word ok"
    cl = CopyConstraint("do you sell laptops?", ["do you sell laptops?"], False)
    assert not (cl.ok(TC + "lap", ")")), "cc mid-word close blocked"
    assert cl.ok(TC + "laptops", ")"), "cc whole-word close ok"
    assert not (cl.ok(TC + "laptops)", ")")), "cc extra paren blocked"
    assert cl.ok(TC + "laptops)", "<|"), "cc result tag after close ok"
    c6 = CopyConstraint("999 times 7", ["999 times 7", "6993"], True)
    assert not (c6.ok("The answer is 6", "els")), "cc reply single digit not in sources blocked"
    assert c6.ok("The answer is 6", "993"), "cc reply full number ok"
    c32 = CopyConstraint("35 degree celsius fahrenheit", ["35 degree celsius fahrenheit", "95"], True)
    assert not (c32.ok("35 + ", "32")), "cc constant blocked in reply"
    assert c32.ok("<|tool_call|>CALCULATE(35 * 9 / 5 + ", "32"), "cc constant ok in calc args"
    cw = CopyConstraint("aaj indore me chhata lena padega?", ["aaj indore me chhata lena padega?"], False)
    assert not (cw.ok(WT, "Lena")), "cc weather non-city blocked"
    assert cw.ok(WT, "ind"), "cc weather city ok"
    assert cw.ok(WT + "indore", ")"), "cc weather close on city"


    ce = CopyConstraint("earbuds kitne ke", ["earbuds kitne ke", "Bluetooth Earbuds | ₹1170", "2340"], True)
    assert not (ce.eos_ok("")), "eos blocked on empty reply"
    assert not (ce.eos_ok("<|tool_call|>MQ_SEARCH(earbuds")), "eos blocked mid tool call"
    assert not (ce.eos_ok("Total ₹23")), "eos blocked on half number"
    assert ce.eos_ok("Total ₹2340"), "eos ok after full number"
    cc2 = CopyConstraint("invoice 26,500 + 18% GST, grand total?", ["invoice 26,500 + 18% GST, grand total?"], False)
    assert not (cc2.ok(CT, "ARCH")), "calc unknown name blocked"
    assert cc2.ok(CT, "emi"), "calc helper name ok"
    assert not (cc2.ok(CT + "3.2", "Amount")), "calc letters after number blocked"
    assert cc2.ok(CT + "26500 * 18 / ", "100"), "calc plain expr ok"
    cd = CopyConstraint("15 august 2027 ko kaunsa vaar", ["15 august 2027 ko kaunsa vaar"], False)
    assert cd.ok(DT + "15 ", "August"), "date month ok"
    assert not (cd.ok(DT + "15 August", "us")), "date garbled month blocked"
    assert not (cd.ok(DT + "2 ", "*")), "date star blocked"
    assert cd.ok(DT + "days ", "until"), "date keyword ok"
    assert cd.ok(DT + "days until ", "republic"), "date festival ok"


    # regression: CALCULATE must be able to close and emit the result tag
    ck = CopyConstraint("2350 ka 18 percent gst", ["2350 ka 18 percent gst"], False)
    assert ck.ok(CT + "2350 * 18 / 100", ")"), "calc close paren ok"
    assert ck.ok(CT + "2350 * 18 / 100)", "<"), "calc result tag after close ok"
    assert ck.ok(CT + "2350 * 18 / 100)<", "|"), "calc result tag continues"
    assert ck.ok(CT + "2350 * 18 / 100)<|", "tool"), "calc result tag word"
    assert not (ck.ok(CT + "2350 * 18 / 100)<", "(")), "calc loop after close blocked"
    assert ck.ok(CT + "2350 * (1 + 18 / 100", ")"), "calc nested parens stay open"
    assert ck.ok(CT + "2350 * (1 + 18 / 100)", ")"), "calc nested then close"
    assert ck.ok(CT + "2350 ", ">="), "calc comparison inside ok"
    assert cd.ok(DT + "15 August 2027)", "<|"), "date close + tag ok"
    cg = CopyConstraint("galat size", ["galat size", "No match in store for: galat"], True)
    assert not (cg.ok("rate ₹", "15")), "15 not free outside date error"
    cde = CopyConstraint("31 feb", ["31 feb", "Error: could not understand date '31 feb'"], True)
    assert cde.ok("like ", "15"), "15 allowed after date error"


    cdn = CopyConstraint("republic day kis din hai", ["republic day kis din hai"], False)
    assert not (cdn.ok(DT + "days until navrat", ")")), "date half word can't close"
    assert cdn.ok(DT + "days until republic day", ")"), "date full festival can close"
    cp = CopyConstraint("poha hai?", ["poha hai?"], False)
    assert not (cp.ok("<|tool_call|>", "WEATHER")), "weather tool blocked w/o city or weather word"
    assert cp.ok("<|tool_call|>", "MQ_SEARCH"), "mq tool allowed"
    assert not (cp.ok("<|tool_call|>", "EMAIL")), "unknown tool name blocked"
    assert cp.ok("<|tool_call|>MQ_", "SEARCH"), "tool name prefix ok"
    cwk = CopyConstraint("shimla me sardi hai kya", ["shimla me sardi hai kya"], False)
    assert cwk.ok("<|tool_call|>", "WEATHER"), "weather tool allowed with city"


    c37 = CopyConstraint("two face wash and one sunscreen", ["two face wash and one sunscreen", "Sunscreen | ₹379", "1577"], True)
    assert not (c37.ok("Sunscreen: 37,", " in")), "truncated number closed via comma blocked"
    assert c37.ok("Sunscreen: 379,", " in"), "full number then comma ok"


    assert not (cd.ok(DT + "15 August", "\n")), "newline blocked in date args"
    assert not (ck.ok(CT + "2350 * 18", "\n")), "newline blocked in calc args"
    assert T.date_tool("15 August\n2027 ko hai").startswith("Error"), "date tool survives multiline"
    c10 = CopyConstraint("3.2 lakh se 4.1 lakh", ["3.2 lakh se 4.1 lakh"], False)
    assert not (c10.ok(CT + "3.2 10000,", "000")), "calc comma-joined number blocked"
    assert c10.ok(CT + "pct_change(", "320000"), "calc lakh as number ok"
