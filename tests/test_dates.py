"""DATE: lenient parsing, festivals, working-day maths - with the clock pinned."""

import datetime as dt

import pytest

import milo.tools as T


@pytest.fixture(autouse=True)
def _pinned_clock():
    """Saturday 10 October 2026, 07:40 IST."""
    T.set_now(dt.datetime(2026, 10, 10, 7, 40))
    yield
    T.set_now(None)


def test_date_tool():
    d = T.date_tool
    assert d("today") == "Saturday, October 10, 2026", "today"
    assert d("") == "Saturday, October 10, 2026", "empty"
    assert d("time") == "07:40 AM IST, Saturday, October 10, 2026", "time"
    assert d("November 8, 2026") == "Sunday, November 8, 2026", "weekday lookup"
    assert d("2026-11-08") == "Sunday, November 8, 2026", "iso"
    assert d("08/11/2026") == "Sunday, November 8, 2026", "indian dd/mm"
    assert d("8th Nov 2026") == "Sunday, November 8, 2026", "ordinal"
    assert d("Sunday, November 8, 2026") == "Sunday, November 8, 2026", "copied weekday"
    assert d("days until November 8, 2026") == "29 days until Sunday, November 8, 2026", "days until"
    assert d("days until 25 Dec") == "76 days until Friday, December 25, 2026", "days until no year"
    assert d("days until 2026-10-01") == "Thursday, October 1, 2026 was 9 days ago", "past"
    assert d("days until tomorrow") == "1 day until Sunday, October 11, 2026", "1 day"
    assert (
        d("days between 2026-01-01 and 2026-03-01")
    ) == "59 days between Thursday, January 1, 2026 and Sunday, March 1, 2026", "between"
    assert d("today + 5 days") == "Thursday, October 15, 2026", "plus"
    assert d("2026-11-08 - 3 days") == "Thursday, November 5, 2026", "minus"
    assert d("working days until 2026-10-17") == "6 working days (Mon–Sat) until Saturday, October 17, 2026", (
        "working days"
    )
    assert d("tuesday-ish") == "Error: could not understand date 'tuesday-ish'", "bad"
    assert d("days until diwali") == "29 days until Sunday, November 8, 2026", "festival"
    assert d("rakhi") == "Tuesday, August 17, 2027", "festival alias"
    assert d("diwali 2027") == "Friday, October 29, 2027", "festival year"
    assert d("days until christmas") == "76 days until Friday, December 25, 2026", "festival fixed"
    assert d("days until navratri") == "1 day until Sunday, October 11, 2026", "festival today"
    assert d("diwali - 2 days") == "Friday, November 6, 2026", "festival plus"
    T.set_now(dt.datetime(2026, 11, 20, 9, 0))
    assert d("days until bhai dooj") == "Error: date for bhai dooj not available yet", "festival unavailable"
    T.set_now(None)
