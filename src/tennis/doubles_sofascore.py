"""SofaScore doubles import - finished matches with stats + upcoming with odds."""

import json
import logging
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from src.tennis.database import get_tennis_db, init_tennis_db
from src.tennis.sofascore_api import _api_get, _sofascore_name_to_db, get_usage

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DOUBLES_ODDS_FILE = DATA_DIR / "sofascore_doubles_odds.json"

MAIN_TOUR_CATEGORIES = {"ATP", "WTA"}


def _is_doubles_main_tour(ev: dict) -> bool:
    """Check if event is ATP/WTA main tour doubles (not challengers, not singles)."""
    cat = ev.get("tournament", {}).get("category", {}).get("name", "")
    if cat not in MAIN_TOUR_CATEGORIES:
        return False

    slug = ev.get("slug", "")
    season_name = ev.get("season", {}).get("name", "")
    tournament_name = ev.get("tournament", {}).get("name", "")
    ut_name = ev.get("uniqueTournament", {}).get("name", "")
    combined = f"{slug} {season_name} {tournament_name} {ut_name}".lower()

    # Must be doubles
    if "double" not in combined:
        return False

    # Skip challengers, ITF, qualifying, mixed doubles
    skip_words = ["challenger", "itf", "futures", "qualifying", "mixed"]
    if any(w in combined for w in skip_words):
        return False

    return True


def _get_or_create_player(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO tennis_players (name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()["id"]


def _parse_doubles_team(team: dict) -> tuple[str, str]:
    """Parse SofaScore doubles team into two player names.

    SofaScore returns doubles teams as:
    - team.name = "Bopanna E. / Ebden M." or "Bopanna/Ebden"
    - or team has subTeams with individual players
    """
    name = team.get("name", "")

    # Check for subTeams first (some events have them)
    sub_teams = team.get("subTeams", [])
    if len(sub_teams) >= 2:
        p1 = sub_teams[0].get("name", "")
        p2 = sub_teams[1].get("name", "")
        if p1 and p2:
            return _sofascore_name_to_db(p1), _sofascore_name_to_db(p2)

    # Parse "Player1 / Player2" format
    if "/" in name:
        parts = name.split("/")
        if len(parts) >= 2:
            return _sofascore_name_to_db(parts[0].strip()), _sofascore_name_to_db(parts[1].strip())

    return "", ""


async def import_doubles_recent(days: int = 10):
    """Fetch recent doubles matches with stats from SofaScore.

    API usage: ~1 request per day for match list + 1 per match for details.
    """
    init_tennis_db()
    conn = get_tennis_db()

    last_date = conn.execute("SELECT MAX(date) FROM doubles_matches").fetchone()[0] or "2024-01-01"
    logger.info(f"Doubles DB last: {last_date}. Fetching SofaScore last {days} days...")

    total_imported = 0
    stats_fetched = 0

    async with httpx.AsyncClient(timeout=20) as client:
        for d in range(days):
            date_str = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")

            if d > 1 and date_str <= last_date:
                logger.info(f"  {date_str} already in DB, stopping")
                break

            data = await _api_get(client, f"/match/list?sport_slug=tennis&date={date_str}")
            if not data:
                continue

            events = data if isinstance(data, list) else data.get("events", [])

            for ev in events:
                try:
                    if not _is_doubles_main_tour(ev):
                        continue

                    status = ev.get("status", {})
                    if not status.get("isFinished"):
                        continue

                    home = ev.get("homeTeam", {})
                    away = ev.get("awayTeam", {})

                    h_p1, h_p2 = _parse_doubles_team(home)
                    a_p1, a_p2 = _parse_doubles_team(away)
                    if not h_p1 or not h_p2 or not a_p1 or not a_p2:
                        continue

                    home_score = ev.get("homeScore", {})
                    away_score = ev.get("awayScore", {})
                    h_sets = home_score.get("current", 0) or 0
                    a_sets = away_score.get("current", 0) or 0

                    if h_sets == a_sets:
                        continue

                    home_won = h_sets > a_sets

                    if home_won:
                        w1, w2, l1, l2 = h_p1, h_p2, a_p1, a_p2
                    else:
                        w1, w2, l1, l2 = a_p1, a_p2, h_p1, h_p2

                    w1_id = _get_or_create_player(conn, w1)
                    w2_id = _get_or_create_player(conn, w2)
                    l1_id = _get_or_create_player(conn, l1)
                    l2_id = _get_or_create_player(conn, l2)

                    tournament = ev.get("uniqueTournament", {}).get("name", "")
                    if not tournament:
                        tournament = ev.get("tournament", {}).get("name", "")
                    # Strip " - Doubles" suffix
                    for suffix in [" - Doubles", " Doubles", ", Pair"]:
                        tournament = tournament.replace(suffix, "")

                    cat = ev.get("tournament", {}).get("category", {}).get("name", "")
                    tour = "WTA" if "wta" in cat.lower() else "ATP"
                    rnd = ev.get("round", {}).get("name", "")

                    ground = ev.get("groundType", "")
                    surface_map = {"hardcourt": "Hard", "clay": "Clay", "grass": "Grass"}
                    surface = surface_map.get(ground, "Hard")

                    # Set scores
                    set_data = {}
                    for i in range(1, 4):
                        hp = home_score.get(f"period{i}")
                        ap = away_score.get(f"period{i}")
                        if hp is not None and ap is not None:
                            if home_won:
                                set_data[f"w{i}"] = hp
                                set_data[f"l{i}"] = ap
                            else:
                                set_data[f"w{i}"] = ap
                                set_data[f"l{i}"] = hp

                    # Check for duplicate
                    exists = conn.execute("""
                        SELECT 1 FROM doubles_matches
                        WHERE date = ? AND w_player1_id = ? AND w_player2_id = ?
                        AND l_player1_id = ? AND l_player2_id = ?
                        LIMIT 1
                    """, (date_str, w1_id, w2_id, l1_id, l2_id)).fetchone()
                    if exists:
                        continue

                    conn.execute("""
                        INSERT OR IGNORE INTO doubles_matches (
                            date, tournament, surface, indoor, round, best_of,
                            series, tour,
                            w_player1_id, w_player2_id, l_player1_id, l_player2_id,
                            score, w_sets, l_sets,
                            w1, l1, w2, l2, w3, l3
                        ) VALUES (?,?,?,0,?,3,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        date_str, tournament, surface, rnd,
                        tour, tour,
                        w1_id, w2_id, l1_id, l2_id,
                        "", max(h_sets, a_sets), min(h_sets, a_sets),
                        set_data.get("w1"), set_data.get("l1"),
                        set_data.get("w2"), set_data.get("l2"),
                        set_data.get("w3"), set_data.get("l3"),
                    ))
                    total_imported += 1

                    # Fetch stats for this match (if budget allows)
                    usage = get_usage()
                    if usage["remaining"] > 100 and stats_fetched < 20:
                        match_id = ev.get("id")
                        if match_id:
                            stats = await _fetch_doubles_match_stats(client, match_id)
                            if stats:
                                _update_doubles_stats(conn, date_str, w1_id, w2_id, l1_id, l2_id, stats, home_won)
                                stats_fetched += 1

                except Exception as e:
                    logger.debug(f"Skip doubles event: {e}")
                    continue

            conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM doubles_matches").fetchone()[0]
    conn.close()

    logger.info(f"Doubles SofaScore: +{total_imported} matches, {stats_fetched} with stats (total: {total})")
    return total_imported


async def _fetch_doubles_match_stats(client, match_id: int) -> dict | None:
    """Fetch match statistics for a doubles match."""
    data = await _api_get(client, f"/match/statistics?match_id={match_id}")
    if not data:
        return None

    stats = {}
    try:
        periods = data if isinstance(data, list) else data.get("statistics", [])
        for period in periods:
            if period.get("period") != "ALL":
                continue
            for group in period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    key = item.get("name", "")
                    stats[key] = {
                        "home": item.get("home", ""),
                        "away": item.get("away", ""),
                    }
    except Exception:
        return None

    return stats if stats else None


def _safe_stat_int(val) -> int:
    """Parse stat value like '12' or '78%' to int."""
    if not val:
        return 0
    try:
        return int(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return 0


def _update_doubles_stats(conn, date_str, w1_id, w2_id, l1_id, l2_id, stats, home_won):
    """Update doubles match with detailed stats."""
    def get_stat(key, side):
        if key in stats:
            return _safe_stat_int(stats[key].get(side, ""))
        return 0

    w_side = "home" if home_won else "away"
    l_side = "away" if home_won else "home"

    conn.execute("""
        UPDATE doubles_matches SET
            w_aces = ?, l_aces = ?,
            w_df = ?, l_df = ?,
            w_1st_won = ?, l_1st_won = ?,
            w_2nd_won = ?, l_2nd_won = ?,
            w_bp_saved = ?, l_bp_saved = ?,
            w_bp_faced = ?, l_bp_faced = ?
        WHERE date = ? AND w_player1_id = ? AND w_player2_id = ?
        AND l_player1_id = ? AND l_player2_id = ?
    """, (
        get_stat("Aces", w_side), get_stat("Aces", l_side),
        get_stat("Double faults", w_side), get_stat("Double faults", l_side),
        get_stat("First serve points won", w_side), get_stat("First serve points won", l_side),
        get_stat("Second serve points won", w_side), get_stat("Second serve points won", l_side),
        get_stat("Break points saved", w_side), get_stat("Break points saved", l_side),
        get_stat("Break points faced", w_side), get_stat("Break points faced", l_side),
        date_str, w1_id, w2_id, l1_id, l2_id,
    ))


async def fetch_doubles_upcoming_with_odds(days: int = 2) -> list[dict]:
    """Fetch upcoming doubles matches with bookmaker odds."""
    usage = get_usage()
    if usage["remaining"] < 30:
        logger.warning("Budget too low for doubles odds")
        return _load_cached_doubles_odds()

    all_upcoming = []

    async with httpx.AsyncClient(timeout=60) as client:
        for d in range(days):
            date_str = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            data = await _api_get(client, f"/match/list?sport_slug=tennis&date={date_str}")
            if not data:
                continue

            events = data if isinstance(data, list) else data.get("events", [])

            for ev in events:
                status = ev.get("status", {})
                if status.get("isFinished") or status.get("isCancelled"):
                    continue
                if not _is_doubles_main_tour(ev):
                    continue

                home = ev.get("homeTeam", {})
                away = ev.get("awayTeam", {})
                h_p1, h_p2 = _parse_doubles_team(home)
                a_p1, a_p2 = _parse_doubles_team(away)
                if not h_p1 or not h_p2 or not a_p1 or not a_p2:
                    continue

                tournament = ev.get("uniqueTournament", {}).get("name", "")
                if not tournament:
                    tournament = ev.get("tournament", {}).get("name", "")
                for suffix in [" - Doubles", " Doubles"]:
                    tournament = tournament.replace(suffix, "")

                cat = ev.get("tournament", {}).get("category", {}).get("name", "")
                tour = "WTA" if "wta" in cat.lower() else "ATP"

                all_upcoming.append({
                    "sofascore_id": ev.get("id"),
                    "date": date_str,
                    "tournament": tournament,
                    "tour": tour,
                    "round": ev.get("round", {}).get("name", ""),
                    "team1_p1": h_p1, "team1_p2": h_p2,
                    "team2_p1": a_p1, "team2_p2": a_p2,
                    "team1_name": f"{h_p1} / {h_p2}",
                    "team2_name": f"{a_p1} / {a_p2}",
                    "match_type": "doubles",
                })

        # Fetch odds (budget-aware)
        remaining = get_usage()["remaining"]
        max_odds = min(len(all_upcoming), remaining - 20)

        for i, match in enumerate(all_upcoming):
            if i >= max_odds:
                break
            odds_data = await _api_get(client, f"/match/odds?match_id={match['sofascore_id']}")
            if not odds_data:
                continue
            odds_list = odds_data if isinstance(odds_data, list) else []
            match["odds"] = _parse_doubles_odds(odds_list)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "updated_at": datetime.utcnow().isoformat(),
        "matches": all_upcoming,
    }
    with open(DOUBLES_ODDS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Doubles: {len(all_upcoming)} upcoming matches with odds")
    return all_upcoming


def _parse_doubles_odds(odds_list: list) -> dict:
    """Parse odds for doubles match."""
    result = {}
    for market in odds_list:
        name = market.get("name", "")
        group = market.get("group", "")
        choices = market.get("choices", [])
        if not choices or len(choices) < 2:
            continue

        if name == "Full time" and group == "Home/Away":
            c1 = choices[0].get("value", {})
            c2 = choices[1].get("value", {})
            result["moneyline"] = {
                "team1_odds": c1.get("decimal", 0),
                "team2_odds": c2.get("decimal", 0),
            }
    return result


def _load_cached_doubles_odds() -> list[dict]:
    try:
        with open(DOUBLES_ODDS_FILE, "r") as f:
            return json.load(f).get("matches", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def load_doubles_odds() -> list[dict]:
    """Load doubles odds for dashboard display."""
    try:
        with open(DOUBLES_ODDS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    matches = []
    for m in data.get("matches", []):
        odds = m.get("odds", {})
        ml = odds.get("moneyline", {})
        t1_odds = ml.get("team1_odds", 0)
        t2_odds = ml.get("team2_odds", 0)

        if t1_odds <= 1 or t2_odds <= 1:
            continue

        matches.append({
            "source": "sofascore",
            "match_type": "doubles",
            "tournament": m.get("tournament", ""),
            "tour": m.get("tour", ""),
            "round": m.get("round", ""),
            "date": m.get("date", ""),
            "team1_name": m.get("team1_name", ""),
            "team2_name": m.get("team2_name", ""),
            "team1_p1": m.get("team1_p1", ""),
            "team1_p2": m.get("team1_p2", ""),
            "team2_p1": m.get("team2_p1", ""),
            "team2_p2": m.get("team2_p2", ""),
            "player1_odds": t1_odds,
            "player2_odds": t2_odds,
            "player1_price": round(1 / t1_odds, 4) if t1_odds > 1 else 0,
            "player2_price": round(1 / t2_odds, 4) if t2_odds > 1 else 0,
        })

    return matches
