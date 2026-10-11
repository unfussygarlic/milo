"""Number grounding: which numbers a reply may contain (invariant 4)."""

import milo.tools as T


def test_grounding():
    assert sorted(T.numbers_in("teen staff, 2 lakh, 50k, 3 dozen")) == [
        "12",
        "2",
        "200000",
        "3",
        "36",
        "50",
        "50000",
    ], "number words"
    assert T.ungrounded_numbers("Basmati 5kg ₹649", ["5 kilo basmati", "Basmati Rice 5kg | ₹709 incl. GST"]) == [
        "649"
    ], "ungrounded"
    assert T.ungrounded_numbers("Total ₹37500", ["teen staff 12500 each", "37500"]) == [], "grounded via words"
