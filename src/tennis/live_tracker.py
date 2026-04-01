"""Live match tracker - combines SofaScore live scores + Polymarket live odds.

Polls SofaScore for live scores and Polymarket for live odds every 30s.
Provides data for WebSocket push to frontend.
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path

import httpx

from src.tennis.sofascore_api import _api_get, get_usage, _is_main_tour

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Cache live state
_live_cache = {"matches": [], "updated": ""}


def _normalize(s: str) -> str:
    """Normalize name for matching."""
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace(".", "").replace("-", " ").strip()


def _names_match(n1: str, n2: str) -> bool:
    """Fuzzy name matching."""
    a, b = _normalize(n1), _normalize(n2)
    if a == b:
        return True
    # Last name match
    pa, pb = a.split(), b.split()
    if pa and pb:
        la = pa[-1] if len(pa[-1]) >= 4 else pa[0]
        lb = pb[-1] if len(pb[-1]) >= 4 else pb[0]
        if la == lb and len(la) >= 4:
            return True
    return False


async def fetch_live_matches() -> list[dict]:
    """Fetch currently live tennis matches from SofaScore.

    Returns list of live matches with scores, sets, current game.
    Uses 1 API request.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    async with httpx.AsyncClient(timeout=15) as client:
        data = await _api_get(client, f"/match/list?sport_slug=tennis&date={today}")
        if not data:
            return []

    events = data if isinstance(data, list) else data.get("events", [])
    live = []

    for ev in events:
        try:
            status = ev.get("status", {})
            if not status.get("isInProgress"):
                continue

            cat = ev.get("tournament", {}).get("category", {}).get("name", "")
            if cat not in ("ATP", "WTA"):
                continue

            home = ev.get("homeTeam", {})
            away = ev.get("awayTeam", {})
            home_name = home.get("name", "")
            away_name = away.get("name", "")
            if not home_name or not away_name:
                continue

            home_score = ev.get("homeScore", {})
            away_score = ev.get("awayScore", {})

            # Current sets
            h_sets = home_score.get("current", 0) or 0
            a_sets = away_score.get("current", 0) or 0

            # Current game score in active set
            h_game = home_score.get("point", "") or ""
            a_game = away_score.get("point", "") or ""

            # Set scores
            sets = []
            for i in range(1, 6):
                hp = home_score.get(f"period{i}")
                ap = away_score.get(f"period{i}")
                if hp is not None and ap is not None:
                    sets.append(f"{hp}-{ap}")

            score_text = " ".join(sets)
            if h_game or a_game:
                score_text += f" ({h_game}-{a_game})"

            tournament = ev.get("uniqueTournament", {}).get("name", "")
            if not tournament:
                tournament = ev.get("tournament", {}).get("name", "")

            slug = ev.get("slug", "")
            ut_name = ev.get("uniqueTournament", {}).get("name", "").lower()
            is_doubles = "double" in slug.lower() or "double" in ut_name

            live.append({
                "sofascore_id": ev.get("id"),
                "player1": home_name,
                "player2": away_name,
                "player1_sets": h_sets,
                "player2_sets": a_sets,
                "score": score_text,
                "game_score": f"{h_game}-{a_game}" if h_game or a_game else "",
                "tournament": tournament,
                "tour": "WTA" if "wta" in cat.lower() else "ATP",
                "round": ev.get("round", {}).get("name", ""),
                "is_doubles": is_doubles,
                "status_desc": status.get("description", ""),
            })
        except Exception:
            continue

    return live


async def fetch_polymarket_live_prices() -> list[dict]:
    """Fetch current Polymarket tennis prices (active markets only)."""
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

    prices = []
    for mk in markets:
        try:
            outcomes = json.loads(mk.get("outcomes", "[]")) if isinstance(mk.get("outcomes"), str) else mk.get("outcomes", [])
            raw_prices = json.loads(mk.get("outcomePrices", "[]")) if isinstance(mk.get("outcomePrices"), str) else mk.get("outcomePrices", [])
            if len(outcomes) < 2 or len(raw_prices) < 2:
                continue

            p1_price = float(raw_prices[0])
            p2_price = float(raw_prices[1])

            # Skip settled markets
            if p1_price >= 0.95 or p2_price >= 0.95:
                continue

            question = mk.get("question", "")
            slug = mk.get("slug", "")
            pm_url = f"https://polymarket.com/event/{slug}" if slug else ""

            prices.append({
                "player1": outcomes[0],
                "player2": outcomes[1],
                "player1_price": p1_price,
                "player2_price": p2_price,
                "player1_odds": round(1 / p1_price, 2) if p1_price > 0.01 else 99.0,
                "player2_odds": round(1 / p2_price, 2) if p2_price > 0.01 else 99.0,
                "question": question,
                "pm_url": pm_url,
            })
        except Exception:
            continue

    return prices


async def get_live_state(pending_bets: list = None) -> dict:
    """Get combined live state: SofaScore scores + Polymarket odds.

    Matches live scores to pending bets to show which bets are currently in play.
    """
    global _live_cache

    # Fetch both in parallel
    live_matches, pm_prices = await asyncio.gather(
        fetch_live_matches(),
        fetch_polymarket_live_prices(),
    )

    # Merge: for each live match, find matching PM price
    for lm in live_matches:
        for pm in pm_prices:
            if (_names_match(lm["player1"], pm["player1"]) and
                    _names_match(lm["player2"], pm["player2"])) or \
               (_names_match(lm["player1"], pm["player2"]) and
                    _names_match(lm["player2"], pm["player1"])):
                # Align prices to match player order
                if _names_match(lm["player1"], pm["player1"]):
                    lm["pm_p1_price"] = pm["player1_price"]
                    lm["pm_p2_price"] = pm["player2_price"]
                    lm["pm_p1_odds"] = pm["player1_odds"]
                    lm["pm_p2_odds"] = pm["player2_odds"]
                else:
                    lm["pm_p1_price"] = pm["player2_price"]
                    lm["pm_p2_price"] = pm["player1_price"]
                    lm["pm_p1_odds"] = pm["player2_odds"]
                    lm["pm_p2_odds"] = pm["player1_odds"]
                lm["pm_url"] = pm["pm_url"]
                break

    # Match to pending bets
    if pending_bets:
        for lm in live_matches:
            for bet in pending_bets:
                t1 = bet.get("team1_name", "")
                t2 = bet.get("team2_name", "")
                label = bet.get("market_label", "")
                if (_names_match(lm["player1"], t1) or _names_match(lm["player1"], t2) or
                        _names_match(lm["player2"], t1) or _names_match(lm["player2"], t2)):
                    lm["bet_id"] = bet.get("id", "")
                    lm["bet_label"] = label
                    lm["bet_stake"] = bet.get("stake", 0)
                    lm["bet_odds"] = bet.get("our_odds", 0)
                    lm["has_bet"] = True
                    break

    _live_cache = {
        "matches": live_matches,
        "pm_count": len(pm_prices),
        "updated": datetime.utcnow().isoformat(),
    }

    return _live_cache


def get_cached_live() -> dict:
    """Return cached live state (non-async)."""
    return _live_cache
