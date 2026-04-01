"""Live match tracker - Polymarket live odds only (no SofaScore API credits).

Polls Polymarket REST every 30s for active tennis markets.
Matches with prices between 0.05-0.95 are considered "in play".
Price movement indicates live score changes.
"""

import json
import logging
import asyncio
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

_live_cache = {"matches": [], "updated": ""}


def _normalize(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace(".", "").replace("-", " ").strip()


def _names_match(n1: str, n2: str) -> bool:
    a, b = _normalize(n1), _normalize(n2)
    if a == b:
        return True
    pa, pb = a.split(), b.split()
    if pa and pb:
        la = pa[-1] if len(pa[-1]) >= 4 else pa[0]
        lb = pb[-1] if len(pb[-1]) >= 4 else pb[0]
        if la == lb and len(la) >= 4:
            return True
    return False


async def fetch_polymarket_live() -> list[dict]:
    """Fetch active Polymarket tennis markets. 0 API credits - free endpoint."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get("https://gamma-api.polymarket.com/markets", params={
                "tag_id": "864",
                "sports_market_types": "moneyline",
                "active": "true",
                "closed": "false",
                "limit": 200,
            })
            if r.status_code != 200:
                return []
            markets = r.json()
    except Exception:
        return []

    matches = []
    for mk in markets:
        try:
            outcomes = json.loads(mk.get("outcomes", "[]")) if isinstance(mk.get("outcomes"), str) else mk.get("outcomes", [])
            raw_prices = json.loads(mk.get("outcomePrices", "[]")) if isinstance(mk.get("outcomePrices"), str) else mk.get("outcomePrices", [])
            if len(outcomes) < 2 or len(raw_prices) < 2:
                continue

            p1_price = float(raw_prices[0])
            p2_price = float(raw_prices[1])

            # Skip settled (>0.95) or not yet started (~0.50/0.50 with no movement)
            if p1_price >= 0.95 or p2_price >= 0.95:
                continue

            slug = mk.get("slug", "")
            if not slug:
                events = mk.get("events", [])
                if events:
                    slug = events[0].get("slug", "")

            pm_url = f"https://polymarket.com/event/{slug}" if slug else ""
            question = mk.get("question", "")
            tournament = mk.get("groupItemTitle", "") or ""

            matches.append({
                "player1": outcomes[0],
                "player2": outcomes[1],
                "player1_price": round(p1_price, 4),
                "player2_price": round(p2_price, 4),
                "player1_odds": round(1 / p1_price, 2) if p1_price > 0.01 else 99.0,
                "player2_odds": round(1 / p2_price, 2) if p2_price > 0.01 else 99.0,
                "pm_url": pm_url,
                "question": question,
                "tournament": tournament,
            })
        except Exception:
            continue

    return matches


async def get_live_state(pending_bets: list = None) -> dict:
    """Get live state from Polymarket. Match to pending bets."""
    global _live_cache

    pm_matches = await fetch_polymarket_live()

    # Match to pending bets
    if pending_bets:
        for m in pm_matches:
            for bet in pending_bets:
                t1 = bet.get("team1_name", "")
                t2 = bet.get("team2_name", "")
                label = bet.get("market_label", "")
                if (_names_match(m["player1"], t1) or _names_match(m["player1"], t2) or
                        _names_match(m["player2"], t1) or _names_match(m["player2"], t2)):
                    m["bet_id"] = bet.get("id", "")
                    m["bet_label"] = label
                    m["bet_stake"] = bet.get("stake", 0)
                    m["bet_odds"] = bet.get("our_odds", 0)
                    m["has_bet"] = True
                    # Determine if our pick is winning
                    pick_name = label.replace(" WIN", "").replace(" +1.5 SETS", "").strip()
                    if _names_match(pick_name, m["player1"]):
                        m["our_player"] = 1
                        m["our_price"] = m["player1_price"]
                    elif _names_match(pick_name, m["player2"]):
                        m["our_player"] = 2
                        m["our_price"] = m["player2_price"]
                    break

    _live_cache = {
        "matches": pm_matches,
        "updated": datetime.utcnow().isoformat(),
    }
    return _live_cache


def get_cached_live() -> dict:
    return _live_cache
