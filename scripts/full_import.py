"""Full SofaScore import - Premium plan (30K req/month).

Phases:
  1. Match history 2015-2026 (day by day) ~4,100 req
  2. Player details for all active players ~500 req
  3. Match statistics 2024-2026 ~13,000 req
  4. Match odds 2026 ~1,500 req

Usage: python scripts/full_import.py [phase1|phase2|phase3|phase4|all]
"""

import asyncio
import json
import logging
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tennis.database import get_tennis_db, init_tennis_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RAPIDAPI_HOST = "sofascore6.p.rapidapi.com"
RAPIDAPI_KEY = "14ba666fd3mshb5821960ffbefdcp127e1bjsnce91762db49e"
BASE_URL = f"https://{RAPIDAPI_HOST}/api/sofascore/v1"
HEADERS = {
    "Content-Type": "application/json",
    "x-rapidapi-host": RAPIDAPI_HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
}
DATA_DIR = Path(__file__).parent.parent / "data"
USAGE_FILE = DATA_DIR / "sofascore_usage.json"
LIMIT = 30000


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def name_to_db(full_name: str) -> str:
    """Convert 'Jannik Sinner' -> 'Sinner J.'"""
    name = strip_diacritics(full_name)
    parts = name.strip().split()
    if len(parts) >= 2:
        first_initial = parts[0][0]
        last = " ".join(parts[1:])
        return f"{last} {first_initial}."
    return name


def load_usage() -> int:
    try:
        with open(USAGE_FILE) as f:
            data = json.load(f)
        if data.get("month") == datetime.now().strftime("%Y-%m"):
            return data.get("count", 0)
    except:
        pass
    return 0


def save_usage(count: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(USAGE_FILE, "w") as f:
        json.dump({"month": datetime.now().strftime("%Y-%m"), "count": count}, f)


async def api_get(client: httpx.AsyncClient, endpoint: str, usage: list) -> dict | list | None:
    if usage[0] >= LIMIT:
        logger.warning(f"LIMIT REACHED: {usage[0]}/{LIMIT}")
        return None
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = await client.get(url, headers=HEADERS)
        usage[0] += 1
        if usage[0] % 100 == 0:
            save_usage(usage[0])
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        logger.debug(f"API error: {e}")
        usage[0] += 1
        return None


def get_or_create_player(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO tennis_players (name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()["id"]


def is_main_tour(ev: dict) -> bool:
    cat = ev.get("tournament", {}).get("category", {}).get("name", "")
    if cat not in ("ATP", "WTA"):
        return False
    slug = ev.get("slug", "")
    season = ev.get("season", {}).get("name", "")
    tourn = ev.get("tournament", {}).get("name", "")
    ut = ev.get("uniqueTournament", {}).get("name", "")
    combined = f"{slug} {season} {tourn} {ut}".lower()
    if any(w in combined for w in ["double", "qualifying", "challenger", "itf", "futures"]):
        return False
    return True


# ─── Phase 1: Match history ─────────────────────────────────

async def phase1_matches(start_year: int = 2015):
    """Import all ATP/WTA singles matches day by day."""
    init_tennis_db()
    conn = get_tennis_db()
    usage = [load_usage()]

    start = datetime(start_year, 1, 1)
    end = datetime.now()
    total_days = (end - start).days + 1
    total_added = 0
    event_ids = {}  # date -> list of (event_id, winner_db, loser_db) for later phases

    logger.info(f"Phase 1: Fetching matches {start_year}-2026 ({total_days} days, usage: {usage[0]}/{LIMIT})")

    async with httpx.AsyncClient(timeout=90) as client:
        d = start
        day_num = 0
        while d <= end:
            date_str = d.strftime("%Y-%m-%d")
            data = await api_get(client, f"/match/list?sport_slug=tennis&date={date_str}", usage)

            added = 0
            if data:
                events = data if isinstance(data, list) else data.get("events", [])
                for ev in events:
                    try:
                        status = ev.get("status", {})
                        if not status.get("isFinished"):
                            continue
                        if not is_main_tour(ev):
                            continue

                        home = ev.get("homeTeam", {})
                        away = ev.get("awayTeam", {})
                        home_name = home.get("name", "")
                        away_name = away.get("name", "")
                        if not home_name or not away_name:
                            continue

                        hs = ev.get("homeScore", {})
                        aws = ev.get("awayScore", {})
                        h_sets = hs.get("current", 0) or 0
                        a_sets = aws.get("current", 0) or 0
                        if h_sets == a_sets:
                            continue

                        winner_full = home_name if h_sets > a_sets else away_name
                        loser_full = away_name if h_sets > a_sets else home_name
                        winner_db = name_to_db(winner_full)
                        loser_db = name_to_db(loser_full)

                        wid = get_or_create_player(conn, winner_db)
                        lid = get_or_create_player(conn, loser_db)

                        if conn.execute("SELECT 1 FROM tennis_matches WHERE date=? AND winner_id=? AND loser_id=? LIMIT 1",
                                        (date_str, wid, lid)).fetchone():
                            continue

                        tournament = ev.get("uniqueTournament", {}).get("name", "")
                        if not tournament:
                            tournament = ev.get("tournament", {}).get("name", "")
                        cat = ev.get("tournament", {}).get("category", {}).get("name", "")
                        tour = "WTA" if "wta" in cat.lower() else "ATP"

                        ground = ev.get("groundType", "")
                        surface_map = {"hardcourt": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}
                        surface = surface_map.get(ground, "Hard")

                        rnd = ev.get("round", {}).get("name", "")

                        h_rank = home.get("ranking") or 0
                        a_rank = away.get("ranking") or 0
                        w_rank = h_rank if h_sets > a_sets else a_rank
                        l_rank = a_rank if h_sets > a_sets else h_rank

                        sd = {}
                        for i in range(1, 6):
                            hp = hs.get(f"period{i}")
                            ap = aws.get(f"period{i}")
                            if hp is not None and ap is not None:
                                if h_sets > a_sets:
                                    sd[f"w{i}"] = hp
                                    sd[f"l{i}"] = ap
                                else:
                                    sd[f"w{i}"] = ap
                                    sd[f"l{i}"] = hp

                        conn.execute("""
                            INSERT OR IGNORE INTO tennis_matches (
                                date, tournament, surface, indoor, round, best_of,
                                series, tour, winner_id, loser_id,
                                winner_rank, loser_rank, w_sets, l_sets,
                                w1, l1, w2, l2, w3, l3, w4, l4, w5, l5
                            ) VALUES (?,?,?,0,?,3,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            date_str, tournament, surface,
                            rnd, tour, tour,
                            wid, lid, w_rank, l_rank,
                            max(h_sets, a_sets), min(h_sets, a_sets),
                            sd.get("w1"), sd.get("l1"),
                            sd.get("w2"), sd.get("l2"),
                            sd.get("w3"), sd.get("l3"),
                            sd.get("w4"), sd.get("l4"),
                            sd.get("w5"), sd.get("l5"),
                        ))
                        added += 1

                        # Save event_id for stats/odds fetching later
                        eid = ev.get("id")
                        if eid:
                            event_ids.setdefault(date_str, []).append(eid)
                    except Exception as e:
                        continue

            conn.commit()
            total_added += added
            day_num += 1

            if day_num % 30 == 0 or added > 0:
                pct = round(day_num / total_days * 100, 1)
                logger.info(f"  {date_str} [{pct}%] +{added} (total: {total_added}, API: {usage[0]})")

            d += timedelta(days=1)

            if usage[0] >= LIMIT:
                logger.warning("API limit reached, stopping")
                break

    save_usage(usage[0])

    # Save event IDs for phase 3
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "sofascore_event_ids.json", "w") as f:
        json.dump(event_ids, f)

    total_m = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    total_p = conn.execute("SELECT COUNT(*) FROM tennis_players").fetchone()[0]
    conn.close()

    logger.info(f"Phase 1 done: {total_added} matches imported ({total_m} total, {total_p} players)")
    logger.info(f"API usage: {usage[0]}/{LIMIT}")
    logger.info(f"Event IDs saved: {sum(len(v) for v in event_ids.values())} for stats/odds")


# ─── Phase 2: Player details ────────────────────────────────

async def phase2_players():
    """Fetch player bio (country, height, birth year) from SofaScore player details."""
    conn = get_tennis_db()
    usage = [load_usage()]

    # Get all players missing bio data
    players = conn.execute("""
        SELECT p.id, p.name FROM tennis_players p
        WHERE p.country = '' OR p.height_cm = 0 OR p.birth_year = 0
    """).fetchall()

    logger.info(f"Phase 2: Fetching details for {len(players)} players (usage: {usage[0]}/{LIMIT})")

    # We need SofaScore team_id for each player. Get from recent match events.
    # For now, skip - player details need team_id which we don't store yet.
    # Bio data was already partially filled manually.
    logger.info("Phase 2: Skipped (player details need team_id mapping - use manual bio data)")
    conn.close()


# ─── Phase 3: Match statistics ───────────────────────────────

async def phase3_statistics(since_year: int = 2024):
    """Fetch match statistics (aces, break points, serve %) for recent matches."""
    conn = get_tennis_db()
    usage = [load_usage()]

    # Load event IDs from phase 1
    try:
        with open(DATA_DIR / "sofascore_event_ids.json") as f:
            all_ids = json.load(f)
    except FileNotFoundError:
        logger.error("No event IDs found. Run phase 1 first.")
        return

    # Filter to since_year
    event_ids = []
    for date_str, ids in all_ids.items():
        if date_str >= f"{since_year}-01-01":
            for eid in ids:
                event_ids.append((date_str, eid))

    logger.info(f"Phase 3: Fetching stats for {len(event_ids)} matches since {since_year} (usage: {usage[0]}/{LIMIT})")

    updated = 0
    async with httpx.AsyncClient(timeout=30) as client:
        for i, (date_str, eid) in enumerate(event_ids):
            data = await api_get(client, f"/match/statistics?match_id={eid}", usage)
            if not data:
                continue

            stats_list = data if isinstance(data, list) else data.get("statistics", [])
            stats = {}
            for period in stats_list:
                pname = period.get("period", "")
                if pname != "ALL":
                    continue
                for group in period.get("groups", []):
                    for item in group.get("statisticsItems", []):
                        key = item.get("key", "")
                        stats[f"home_{key}"] = item.get("homeValue", 0)
                        stats[f"away_{key}"] = item.get("awayValue", 0)

            if not stats:
                continue

            # We need to figure out which player is home/away for this match
            # For now store raw - we'll need match_id mapping
            # TODO: map event_id to match_id and store stats
            updated += 1

            if (i + 1) % 500 == 0:
                logger.info(f"  Stats: {i+1}/{len(event_ids)} ({updated} with data, API: {usage[0]})")

            if usage[0] >= LIMIT:
                break

    save_usage(usage[0])
    logger.info(f"Phase 3 done: {updated} matches with statistics (API: {usage[0]})")
    conn.close()


# ─── Phase 4: Elo + Attributes ──────────────────────────────

def phase4_compute():
    """Recompute Elo ratings and player attributes."""
    from src.tennis.elo import compute_all_elo
    from src.tennis.players import sync_player_attributes

    logger.info("Phase 4: Computing Elo ratings...")
    compute_all_elo()
    logger.info("Phase 4: Syncing player attributes...")
    sync_player_attributes()
    logger.info("Phase 4 done")


# ─── Main ────────────────────────────────────────────────────

async def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase in ("phase1", "all", "1"):
        await phase1_matches(start_year=2015)

    if phase in ("phase4", "all", "elo"):
        phase4_compute()

    if phase in ("phase2", "2"):
        await phase2_players()

    if phase in ("phase3", "3"):
        await phase3_statistics(since_year=2024)

    logger.info(f"Final API usage: {load_usage()}/{LIMIT}")


if __name__ == "__main__":
    asyncio.run(main())
