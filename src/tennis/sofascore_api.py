"""Sofascore via RapidAPI - reliable, no Cloudflare, 500 req/month free.

Budget strategy (500 req/month):
  - Get_matches_by_date (tennis): 1 req/day = ~30/month
  - Get_match_odds per upcoming ATP/WTA singles: ~10/day = ~300/month
  - Get_match_details for finished results: ~10/day on update days = ~100/month
  - Reserve: ~70 req/month
"""

import json
import logging
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from src.tennis.database import get_tennis_db, init_tennis_db

logger = logging.getLogger(__name__)

RAPIDAPI_HOST = "sofascore6.p.rapidapi.com"
RAPIDAPI_KEY = "14ba666fd3mshb5821960ffbefdcp127e1bjsnce91762db49e"
BASE_URL = f"https://{RAPIDAPI_HOST}/api/sofascore/v1"

HEADERS = {
    "Content-Type": "application/json",
    "x-rapidapi-host": RAPIDAPI_HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
}

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ODDS_FILE = DATA_DIR / "sofascore_odds.json"
USAGE_FILE = DATA_DIR / "sofascore_usage.json"

# Main tour tournaments only (no challengers, no ITF, no doubles)
MAIN_TOUR_KEYWORDS = [
    "miami", "indian wells", "rome", "madrid", "roland garros",
    "wimbledon", "us open", "australian open", "monte carlo",
    "canadian open", "cincinnati", "shanghai", "paris",
    "atp finals", "wta finals", "queens", "halle", "beijing",
    "dubai", "doha", "barcelona", "acapulco", "rotterdam",
    "s-hertogenbosch", "eastbourne", "stuttgart", "san diego",
    "tokyo", "basel", "vienna", "lyon", "marseille",
]


def _load_usage() -> dict:
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
        month = datetime.now().strftime("%Y-%m")
        if data.get("month") != month:
            return {"month": month, "count": 0}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"month": datetime.now().strftime("%Y-%m"), "count": 0}


def _save_usage(usage: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f)


def get_usage() -> dict:
    """Return current month usage stats."""
    u = _load_usage()
    return {"month": u["month"], "used": u["count"], "limit": 300000, "remaining": 300000 - u["count"]}


# Tournament-name → surface fallback when groundType is missing/unknown
_SURFACE_NAME_HINTS = [
    ("roland garros", "Clay"), ("french open", "Clay"), ("monte carlo", "Clay"),
    ("rome", "Clay"), ("madrid", "Clay"), ("barcelona", "Clay"), ("hamburg", "Clay"),
    ("kitzbuhel", "Clay"), ("gstaad", "Clay"), ("bastad", "Clay"), ("umag", "Clay"),
    ("estoril", "Clay"), ("munich", "Clay"), ("houston", "Clay"),
    ("wimbledon", "Grass"), ("halle", "Grass"), ("queen", "Grass"),
    ("s-hertogenbosch", "Grass"), ("eastbourne", "Grass"), ("stuttgart", "Grass"),
    ("mallorca", "Grass"), ("newport", "Grass"),
]


def normalize_surface(ground: str, tournament: str = "") -> tuple[str, int]:
    """Map a SofaScore groundType to (surface, indoor).

    SofaScore returns values like 'Red clay', 'Hardcourt outdoor', 'Hardcourt indoor',
    'Clay', 'Grass', 'Hard'. The old exact-match map silently defaulted everything to
    'Hard' (e.g. 'Red clay' → 'Hard'), so substring matching is required.
    """
    g = (ground or "").strip().lower()
    indoor = 1 if "indoor" in g else 0
    if "clay" in g:
        return "Clay", indoor
    if "grass" in g:
        return "Grass", indoor
    if "carpet" in g:
        return "Carpet", indoor
    if "hard" in g:
        return "Hard", indoor
    # groundType missing/unknown → guess from tournament name
    t = (tournament or "").lower()
    for kw, surf in _SURFACE_NAME_HINTS:
        if kw in t:
            return surf, indoor
    return "Hard", indoor


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _sofascore_name_to_db(full_name: str) -> str:
    """Convert 'Jannik Sinner' -> 'Sinner J.' to match tennis-data.co.uk format."""
    name = _strip_diacritics(full_name)
    parts = name.strip().split()
    if len(parts) >= 2:
        first_initial = parts[0][0]
        last = " ".join(parts[1:])  # Handle "Auger-Aliassime" etc.
        return f"{last} {first_initial}."
    return name


def _get_or_create_player(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO tennis_players (name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()["id"]


async def _api_get(client: httpx.AsyncClient, endpoint: str) -> list | dict | None:
    """Make a RapidAPI request with usage tracking."""
    usage = _load_usage()
    if usage["count"] >= 300000:
        logger.warning(f"SofaScore API limit reached: {usage['count']}/30000 for {usage['month']}")
        return None

    url = f"{BASE_URL}{endpoint}"
    try:
        resp = await client.get(url, headers=HEADERS, timeout=120)
        usage["count"] += 1
        _save_usage(usage)

        if resp.status_code != 200:
            logger.warning(f"RapidAPI {resp.status_code}: {endpoint} (usage: {usage['count']}/30000)")
            return None

        return resp.json()
    except Exception as e:
        usage["count"] += 1
        _save_usage(usage)
        logger.warning(f"RapidAPI error: {endpoint}: {e}")
        return None


def _is_main_tour(event: dict) -> bool:
    """Check if event is ATP main tour singles (no WTA, no challengers, ITF, doubles)."""
    cat = event.get("tournament", {}).get("category", {}).get("name", "")
    if cat != "ATP":
        return False

    slug = event.get("slug", "")
    season_name = event.get("season", {}).get("name", "")
    tournament_name = event.get("tournament", {}).get("name", "")
    ut_name = event.get("uniqueTournament", {}).get("name", "")

    combined = f"{slug} {season_name} {tournament_name} {ut_name}".lower()

    skip_words = ["double", "qualifying", "challenger", "itf", "futures", "wta", "women", "ladies"]
    if any(w in combined for w in skip_words):
        return False

    return True


# ─── Finished match import (for training data) ──────────────

async def fetch_finished_matches(date_str: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch all finished tennis matches for a date. Uses 1 API request."""
    data = await _api_get(client, f"/match/list?sport_slug=tennis&date={date_str}")
    if not data:
        return []

    events = data if isinstance(data, list) else data.get("events", [])
    matches = []

    for ev in events:
        try:
            status = ev.get("status", {})
            if not status.get("isFinished"):
                continue
            if not _is_main_tour(ev):
                continue

            home = ev.get("homeTeam", {})
            away = ev.get("awayTeam", {})
            home_name = home.get("name", "")
            away_name = away.get("name", "")
            if not home_name or not away_name:
                continue

            home_score = ev.get("homeScore", {})
            away_score = ev.get("awayScore", {})
            h_sets = home_score.get("current", 0) or 0
            a_sets = away_score.get("current", 0) or 0

            if h_sets == a_sets:
                continue

            winner_full = home_name if h_sets > a_sets else away_name
            loser_full = away_name if h_sets > a_sets else home_name

            tournament = ev.get("uniqueTournament", {}).get("name", "")
            if not tournament:
                tournament = ev.get("tournament", {}).get("name", "")
            cat = ev.get("tournament", {}).get("category", {}).get("name", "")
            tour = "WTA" if "wta" in cat.lower() else "ATP"

            surface, indoor = normalize_surface(ev.get("groundType", ""), tournament)

            rnd = ev.get("round", {}).get("name", "")

            h_rank = home.get("ranking") or 0
            a_rank = away.get("ranking") or 0
            w_rank = h_rank if h_sets > a_sets else a_rank
            l_rank = a_rank if h_sets > a_sets else h_rank

            set_data = {}
            for i in range(1, 6):
                hp = home_score.get(f"period{i}")
                ap = away_score.get(f"period{i}")
                if hp is not None and ap is not None:
                    if h_sets > a_sets:
                        set_data[f"w{i}"] = hp
                        set_data[f"l{i}"] = ap
                    else:
                        set_data[f"w{i}"] = ap
                        set_data[f"l{i}"] = hp

            matches.append({
                "event_id": ev.get("id"),
                "date": date_str,
                "tournament": tournament,
                "surface": surface,
                "indoor": indoor,
                "tour": tour,
                "round": rnd,
                "winner": _sofascore_name_to_db(winner_full),
                "loser": _sofascore_name_to_db(loser_full),
                "winner_rank": w_rank,
                "loser_rank": l_rank,
                "w_sets": max(h_sets, a_sets),
                "l_sets": min(h_sets, a_sets),
                "set_data": set_data,
            })
        except Exception as e:
            logger.debug(f"Skip event: {e}")
            continue

    logger.info(f"  {date_str}: {len(matches)} finished ATP/WTA singles ({len(events)} total events)")
    return matches


async def import_recent(days: int = 10):
    """Fetch recent days of finished matches and import into DB.

    API usage: ~1 request per day = ~10 requests for 10 days.
    """
    init_tennis_db()
    conn = get_tennis_db()

    last_date = conn.execute("SELECT MAX(date) FROM tennis_matches").fetchone()[0] or "2020-01-01"
    logger.info(f"DB last: {last_date}. Fetching Sofascore RapidAPI last {days} days...")

    total_imported = 0

    async with httpx.AsyncClient(timeout=20) as client:
        for d in range(days):
            date = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")

            if d > 1 and date <= last_date:
                logger.info(f"  {date} already in DB, stopping")
                break

            matches = await fetch_finished_matches(date, client)

            for m in matches:
                try:
                    winner_id = _get_or_create_player(conn, m["winner"])
                    loser_id = _get_or_create_player(conn, m["loser"])

                    # Skip if match already exists (any source)
                    exists = conn.execute("""
                        SELECT 1 FROM tennis_matches
                        WHERE date = ? AND winner_id = ? AND loser_id = ?
                        LIMIT 1
                    """, (m["date"], winner_id, loser_id)).fetchone()
                    if exists:
                        continue

                    sd = m["set_data"]

                    conn.execute("""
                        INSERT OR IGNORE INTO tennis_matches (
                            date, tournament, surface, indoor, round, best_of,
                            series, tour, winner_id, loser_id,
                            winner_rank, loser_rank, w_sets, l_sets,
                            w1, l1, w2, l2, w3, l3, w4, l4, w5, l5
                        ) VALUES (?,?,?,?,?,3,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        m["date"], m["tournament"], m["surface"], m.get("indoor", 0),
                        m["round"], m["tour"], m["tour"],
                        winner_id, loser_id,
                        m["winner_rank"], m["loser_rank"],
                        m["w_sets"], m["l_sets"],
                        sd.get("w1"), sd.get("l1"),
                        sd.get("w2"), sd.get("l2"),
                        sd.get("w3"), sd.get("l3"),
                        sd.get("w4"), sd.get("l4"),
                        sd.get("w5"), sd.get("l5"),
                    ))
                    total_imported += 1
                except Exception as e:
                    logger.debug(f"  Skip: {e}")

            conn.commit()

    new_last = conn.execute("SELECT MAX(date) FROM tennis_matches").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    conn.close()

    logger.info(f"Sofascore API: +{total_imported} new matches (total: {total}, last: {new_last})")
    return total_imported


# ─── Upcoming matches + bookmaker odds ───────────────────────

async def fetch_upcoming_with_odds(days: int = 2) -> list[dict]:
    """Fetch upcoming ATP/WTA matches and their bookmaker odds.

    API usage: 1 req for match list + 1 req per match for odds.
    Typically ~10-15 upcoming main tour matches = ~12-17 requests.

    Cache TTL: serve cached file if it's <60min old, to keep us inside the
    30k/month RapidAPI plan. Periodic refresh runs every 30min, so this
    halves the call rate without losing data freshness in practice.
    """
    CACHE_TTL_SECONDS = 60 * 60
    try:
        with open(ODDS_FILE) as _f:
            _cached = json.load(_f)
        _updated = _cached.get("updated_at", "")
        if _updated:
            _age = (datetime.utcnow() - datetime.fromisoformat(_updated.replace("Z", ""))).total_seconds()
            if _age < CACHE_TTL_SECONDS:
                logger.info(f"SofaScore cache fresh ({int(_age)}s old) — skipping fetch")
                return _cached.get("matches", [])
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
        pass

    usage = get_usage()
    logger.info(f"SofaScore API usage: {usage['used']}/{usage['limit']} ({usage['remaining']} remaining)")

    if usage["remaining"] < 20:
        logger.warning("SofaScore API budget too low, skipping odds fetch")
        return _load_cached_odds()

    all_upcoming = []

    async with httpx.AsyncClient(timeout=60) as client:
        # Fetch matches for today + tomorrow
        for d in range(days):
            date = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            data = await _api_get(client, f"/match/list?sport_slug=tennis&date={date}")
            if not data:
                continue

            events = data if isinstance(data, list) else data.get("events", [])

            for ev in events:
                status = ev.get("status", {})
                if status.get("isFinished") or status.get("isCancelled"):
                    continue
                if not _is_main_tour(ev):
                    continue

                home = ev.get("homeTeam", {})
                away = ev.get("awayTeam", {})
                home_name = home.get("name", "")
                away_name = away.get("name", "")
                if not home_name or not away_name:
                    continue

                tournament = ev.get("uniqueTournament", {}).get("name", "")
                if not tournament:
                    tournament = ev.get("tournament", {}).get("name", "")
                cat = ev.get("tournament", {}).get("category", {}).get("name", "")
                tour = "WTA" if "wta" in cat.lower() else "ATP"

                rnd = ev.get("round", {}).get("name", "")
                is_live = status.get("isInProgress", False)
                start_ts = ev.get("startTimestamp", 0) or 0
                start_iso = datetime.utcfromtimestamp(start_ts).isoformat() if start_ts else ""

                # Build player slug for SofaScore URL
                home_slug = home.get("slug", "")
                away_slug = away.get("slug", "")
                home_id = home.get("id", 0)
                away_id = away.get("id", 0)

                all_upcoming.append({
                    "sofascore_id": ev.get("id"),
                    "date": date,
                    "tournament": tournament,
                    "tour": tour,
                    "round": rnd,
                    "player1": home_name,
                    "player2": away_name,
                    "player1_rank": home.get("ranking") or 0,
                    "player2_rank": away.get("ranking") or 0,
                    "player1_ss_id": home_id,
                    "player2_ss_id": away_id,
                    "player1_slug": home_slug,
                    "player2_slug": away_slug,
                    "is_live": is_live,
                    "start_time": start_iso,
                    "status": status.get("description", ""),
                })

        logger.info(f"Found {len(all_upcoming)} upcoming ATP/WTA singles matches")

        # Polymarket is the canonical odds source (primary + fallback). Skip
        # SofaScore /match/odds calls — they were 88% of monthly RapidAPI quota
        # and were getting overwritten by PM odds anyway. SS now contributes
        # metadata only: tour, tournament, round, rank, start_time.

    # Save to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "updated_at": datetime.utcnow().isoformat(),
        "usage": get_usage(),
        "matches": all_upcoming,
    }
    with open(ODDS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved {len(all_upcoming)} matches with odds to {ODDS_FILE}")
    return all_upcoming


def _parse_odds(odds_list: list, player1: str, player2: str) -> dict:
    """Parse SofaScore odds response into structured format."""
    result = {}

    for market in odds_list:
        name = market.get("name", "")
        group = market.get("group", "")
        period = market.get("period", "")
        choices = market.get("choices", [])

        if not choices or len(choices) < 2:
            continue

        if name == "Full time" and group == "Home/Away":
            c1 = choices[0].get("value", {})
            c2 = choices[1].get("value", {})
            result["moneyline"] = {
                "player1_odds": c1.get("decimal", 0),
                "player2_odds": c2.get("decimal", 0),
                "player1_prob": round(1 / c1["decimal"], 4) if c1.get("decimal", 0) > 1 else 0,
                "player2_prob": round(1 / c2["decimal"], 4) if c2.get("decimal", 0) > 1 else 0,
            }

        elif name == "First set winner":
            c1 = choices[0].get("value", {})
            c2 = choices[1].get("value", {})
            result["first_set"] = {
                "player1_odds": c1.get("decimal", 0),
                "player2_odds": c2.get("decimal", 0),
            }

        elif "Total games" in name:
            choice_group = market.get("choiceGroup", "")
            over_odds = 0
            under_odds = 0
            for c in choices:
                val = c.get("value", {}).get("decimal", 0)
                if c.get("name") == "Over":
                    over_odds = val
                elif c.get("name") == "Under":
                    under_odds = val
            if choice_group:
                result["total_games"] = {
                    "line": float(choice_group),
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                }

    return result


def _load_cached_odds() -> list[dict]:
    """Load previously cached odds."""
    try:
        with open(ODDS_FILE, "r") as f:
            data = json.load(f)
        return data.get("matches", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def load_sofascore_odds() -> list[dict]:
    """Load SofaScore odds for dashboard display.

    Returns list compatible with existing tennis_matches format.
    """
    try:
        with open(ODDS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    matches = []
    for m in data.get("matches", []):
        # SS no longer supplies odds (Polymarket is the canonical source); we
        # keep the metadata so _merge_tennis_odds can still tag matchups with
        # tour/tournament/round/rank/start_time.
        odds = m.get("odds", {}) or {}
        ml = odds.get("moneyline", {}) or {}
        p1_odds = ml.get("player1_odds", 0)
        p2_odds = ml.get("player2_odds", 0)
        p1_prob = ml.get("player1_prob", 0)
        p2_prob = ml.get("player2_prob", 0)

        first_set = odds.get("first_set", {}) or {}
        total = odds.get("total_games", {}) or {}

        matches.append({
            "source": "sofascore",
            "sofascore_id": m.get("sofascore_id"),
            "tournament": m.get("tournament", ""),
            "tour": m.get("tour", ""),
            "round": m.get("round", ""),
            "date": m.get("date", ""),
            "player1": m.get("player1", ""),
            "player2": m.get("player2", ""),
            "player1_rank": m.get("player1_rank", 0),
            "player2_rank": m.get("player2_rank", 0),
            "player1_odds": p1_odds,
            "player2_odds": p2_odds,
            "player1_price": p1_prob,
            "player2_price": p2_prob,
            "is_live": m.get("is_live", False),
            "start_time": m.get("start_time", ""),
            "status": m.get("status", ""),
            # Extra markets
            "first_set_p1_odds": first_set.get("player1_odds", 0),
            "first_set_p2_odds": first_set.get("player2_odds", 0),
            "total_games_line": total.get("line", 0),
            "total_games_over": total.get("over_odds", 0),
            "total_games_under": total.get("under_odds", 0),
        })

    return matches
