"""SEARCH (Wikipedia) and WEATHER (wttr.in) — the two network tools.

``MAX_SEARCH_CHARS`` and the ``format_*`` helpers are part of the training-data
contract: they shape the result strings the model copies from.
"""

import json
import re
import urllib.parse
import urllib.request

MAX_SEARCH_CHARS = 280
HTTP_TIMEOUT = 6
USER_AGENT = "Milo-Agent/1.0"


# ═══════════════════════════════════════════════════════════════
# HTTP helper
# ═══════════════════════════════════════════════════════════════
def _http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _truncate(text, limit):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    if dot > limit * 0.5:
        return cut[: dot + 1]
    return cut.rsplit(" ", 1)[0] + "..."


# ═══════════════════════════════════════════════════════════════
# WEATHER
# ═══════════════════════════════════════════════════════════════
def format_weather(
    city: str, temp_c: float, condition: str, humidity: int, wind_kmph: int, rain_chance: int | None = None
) -> str:
    """Canonical weather string — the generator must call this too."""
    s = f"{city}: {temp_c}°C, {condition.strip().lower()}, humidity {humidity}%, wind {wind_kmph} km/h"
    if rain_chance is not None:
        s += f", rain chance {rain_chance}%"
    return s


def weather(city: str) -> str:
    city = city.strip().strip("'\"")
    if not city:
        return "Error: no city given"
    try:
        data = _http_json(f"https://wttr.in/{urllib.parse.quote(city)}?format=j1")
        cur = data["current_condition"][0]
        rain = max(int(h.get("chanceofrain", 0)) for h in data["weather"][0]["hourly"])
        return format_weather(
            city.title(),
            int(cur["temp_C"]),
            cur["weatherDesc"][0]["value"],
            int(cur["humidity"]),
            int(cur["windspeedKmph"]),
            rain,
        )
    except Exception:
        return f"Error: weather unavailable for {city.title()}"


# ═══════════════════════════════════════════════════════════════
# SEARCH — general knowledge (Wikipedia)
# ═══════════════════════════════════════════════════════════════
def format_search_result(title: str, text: str) -> str:
    """Canonical search string — the generator must call this too."""
    return f"{title}: {_truncate(text, MAX_SEARCH_CHARS)}"


def _search_wikipedia(query):
    params = urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"}
    )
    hits = _http_json(f"https://en.wikipedia.org/w/api.php?{params}")["query"]["search"]
    if not hits:
        return None
    title = hits[0]["title"]
    page = urllib.parse.quote(title.replace(" ", "_"))
    extract = _http_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}").get("extract", "")
    return format_search_result(title, extract) if extract else None


def search(query: str) -> str:
    query = query.strip().strip("'\"")
    if not query:
        return "Error: empty search query"
    try:
        hit = _search_wikipedia(query)
    except Exception:
        return "Error: search unavailable"
    return hit or f"No results found for: {query}"
