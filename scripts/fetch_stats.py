"""Phase 3: Fetch match statistics for 2024-2026.

Step 1: Re-fetch match lists to map event_id -> DB match (~820 req)
Step 2: Fetch match/statistics for each event (~25K req)
Total: ~26K req
"""

import asyncio
import json
import logging
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import httpx

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


def strip_d(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def name_to_db(full_name):
    name = strip_d(full_name)
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{' '.join(parts[1:])} {parts[0][0]}."
    return name


def load_usage():
    try:
        with open(USAGE_FILE) as f:
            data = json.load(f)
        if data.get("month") == datetime.now().strftime("%Y-%m"):
            return data.get("count", 0)
    except:
        pass
    return 0


def save_usage(count):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(USAGE_FILE, "w") as f:
        json.dump({"month": datetime.now().strftime("%Y-%m"), "count": count}, f)


async def api_get(client, endpoint, usage):
    if usage[0] >= LIMIT:
        return None
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = await client.get(url, headers=HEADERS)
        usage[0] += 1
        if usage[0] % 200 == 0:
            save_usage(usage[0])
        if resp.status_code == 429:
            logger.warning(f"Rate limited, sleeping 5s...")
            await asyncio.sleep(5)
            return None
        if resp.status_code != 200:
            return None
        await asyncio.sleep(0.3)  # rate limit: ~3 req/s
        return resp.json()
    except:
        usage[0] += 1
        return None


def is_main_tour(ev):
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


# SofaScore stat key -> DB column mapping
STAT_MAP = {
    "aces": ("w_aces", "l_aces"),
    "doubleFaults": ("w_df", "l_df"),
    "servicePointsWon": ("w_svpt", "l_svpt"),
    "firstServePointsWon": ("w_1st_won", "l_1st_won"),
    "secondServePointsWon": ("w_2nd_won", "l_2nd_won"),
    "breakPointsSaved": ("w_bp_saved", "l_bp_saved"),
    "breakPointsFaced": ("w_bp_faced", "l_bp_faced"),
    "firstServeIn": ("w_1st_in", "l_1st_in"),
}


async def run(since_year=2024):
    init_tennis_db()
    conn = get_tennis_db()
    usage = [load_usage()]

    start = datetime(since_year, 1, 1)
    end = datetime.now()
    total_days = (end - start).days + 1

    logger.info(f"Stats fetch {since_year}-2026 ({total_days} days, usage: {usage[0]}/{LIMIT})")

    mapped = 0
    stats_fetched = 0
    stats_saved = 0

    async with httpx.AsyncClient(timeout=90) as client:
        d = start
        day_num = 0

        while d <= end:
            date_str = d.strftime("%Y-%m-%d")

            # Skip days already fully mapped
            already_mapped = conn.execute("""
                SELECT COUNT(*) FROM tennis_matches
                WHERE date = ? AND sofascore_event_id > 0 AND w_aces > 0
            """, (date_str,)).fetchone()[0]
            total_day = conn.execute("SELECT COUNT(*) FROM tennis_matches WHERE date = ?", (date_str,)).fetchone()[0]
            if already_mapped > 0 and already_mapped >= total_day * 0.3:
                d += timedelta(days=1)
                day_num += 1
                continue

            # Step 1: Fetch match list to get event_ids
            data = await api_get(client, f"/match/list?sport_slug=tennis&date={date_str}", usage)
            if not data:
                d += timedelta(days=1)
                day_num += 1
                continue

            events = data if isinstance(data, list) else data.get("events", [])

            # Map event_id to DB match_id
            event_map = []  # (event_id, db_match_id, home_is_winner)
            for ev in events:
                status = ev.get("status", {})
                if not status.get("isFinished"):
                    continue
                if not is_main_tour(ev):
                    continue

                eid = ev.get("id")
                if not eid:
                    continue

                home = ev.get("homeTeam", {}).get("name", "")
                away = ev.get("awayTeam", {}).get("name", "")
                if not home or not away:
                    continue

                hs = ev.get("homeScore", {})
                aws = ev.get("awayScore", {})
                if not isinstance(hs, dict):
                    hs = {}
                if not isinstance(aws, dict):
                    aws = {}
                h_sets = hs.get("current", 0) or 0
                a_sets = aws.get("current", 0) or 0
                if h_sets == a_sets:
                    continue

                home_won = h_sets > a_sets
                winner_db = name_to_db(home if home_won else away)
                loser_db = name_to_db(away if home_won else home)

                # Find in DB
                row = conn.execute("""
                    SELECT m.id FROM tennis_matches m
                    JOIN tennis_players p1 ON m.winner_id = p1.id
                    JOIN tennis_players p2 ON m.loser_id = p2.id
                    WHERE m.date = ? AND p1.name = ? AND p2.name = ?
                    LIMIT 1
                """, (date_str, winner_db, loser_db)).fetchone()

                if row:
                    # Update event_id
                    conn.execute("UPDATE tennis_matches SET sofascore_event_id = ? WHERE id = ?", (eid, row["id"]))
                    event_map.append((eid, row["id"], home_won))
                    mapped += 1

            conn.commit()

            # Step 2: Fetch stats for each mapped event
            for eid, mid, home_won in event_map:
                # Skip if already has stats
                existing = conn.execute("SELECT w_aces FROM tennis_matches WHERE id = ?", (mid,)).fetchone()
                if existing and existing["w_aces"] and existing["w_aces"] > 0:
                    continue

                sdata = await api_get(client, f"/match/statistics?match_id={eid}", usage)
                stats_fetched += 1

                if not sdata:
                    continue

                stats_list = sdata if isinstance(sdata, list) else sdata.get("statistics", [])
                parsed = {}
                for period in stats_list:
                    if period.get("period") != "ALL":
                        continue
                    for group in period.get("groups", []):
                        for item in group.get("statisticsItems", []):
                            key = item.get("key", "")
                            if key in STAT_MAP:
                                hv = item.get("homeValue", 0)
                                av = item.get("awayValue", 0)
                                # Parse percentage strings like "64%" -> 64
                                if isinstance(hv, str):
                                    hv = int(hv.replace("%", "")) if "%" in hv else 0
                                if isinstance(av, str):
                                    av = int(av.replace("%", "")) if "%" in av else 0

                                w_col, l_col = STAT_MAP[key]
                                if home_won:
                                    parsed[w_col] = hv
                                    parsed[l_col] = av
                                else:
                                    parsed[w_col] = av
                                    parsed[l_col] = hv

                if parsed:
                    sets = ", ".join(f"{k}=?" for k in parsed)
                    vals = list(parsed.values()) + [mid]
                    conn.execute(f"UPDATE tennis_matches SET {sets} WHERE id = ?", vals)
                    stats_saved += 1

                if usage[0] >= LIMIT:
                    break

            conn.commit()
            day_num += 1

            if day_num % 30 == 0:
                pct = round(day_num / total_days * 100, 1)
                logger.info(f"  {date_str} [{pct}%] mapped:{mapped} stats:{stats_saved}/{stats_fetched} API:{usage[0]}")

            d += timedelta(days=1)

            if usage[0] >= LIMIT:
                logger.warning("API limit reached")
                break

    save_usage(usage[0])
    conn.close()

    logger.info(f"Done: mapped {mapped} events, fetched {stats_fetched} stats, saved {stats_saved}")
    logger.info(f"API usage: {usage[0]}/{LIMIT}")


if __name__ == "__main__":
    since = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    asyncio.run(run(since_year=since))
