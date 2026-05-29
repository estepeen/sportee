"""Backfill correct surface (+indoor) on existing matches from SofaScore.

The old ingestion mapped groundType with an exact-match dict, so values like
'Red clay' / 'Hardcourt outdoor' silently fell back to 'Hard' — leaving the whole
DB labelled Hard (Roland Garros included). This re-fetches match/list per date and
rewrites surface/indoor using the substring-based normalize_surface().

Usage: python scripts/backfill_surface.py [start_date]   (default 2015-01-01)
Matches are keyed by (date, winner_id, loser_id), so only rows already in the DB
are touched. ~1 API request per distinct date.
"""

import asyncio
import logging
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tennis.database import get_tennis_db
from src.tennis.sofascore_api import _api_get, normalize_surface, _sofascore_name_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_events(data) -> list[dict]:
    """Extract finished singles matches (any tour) with surface/indoor + names."""
    events = data if isinstance(data, list) else (data.get("events", []) if data else [])
    out = []
    for ev in events:
        try:
            if not ev.get("status", {}).get("isFinished"):
                continue
            tournament = ev.get("uniqueTournament", {}).get("name", "") or \
                ev.get("tournament", {}).get("name", "")
            slug = ev.get("slug", "")
            blob = f"{slug} {tournament}".lower()
            if "double" in blob or "qualifying" in blob:
                continue

            home = ev.get("homeTeam", {})
            away = ev.get("awayTeam", {})
            hn, an = home.get("name", ""), away.get("name", "")
            if not hn or not an:
                continue

            h = ev.get("homeScore", {}).get("current", 0) or 0
            a = ev.get("awayScore", {}).get("current", 0) or 0
            if h == a:
                continue

            winner_full = hn if h > a else an
            loser_full = an if h > a else hn
            surface, indoor = normalize_surface(ev.get("groundType", ""), tournament)
            out.append({
                "winner": _sofascore_name_to_db(winner_full),
                "loser": _sofascore_name_to_db(loser_full),
                "surface": surface,
                "indoor": indoor,
            })
        except Exception as e:
            logger.debug(f"skip event: {e}")
    return out


async def backfill(start_date: str = "2015-01-01"):
    conn = get_tennis_db()

    name_to_id = {r["name"]: r["id"] for r in
                  conn.execute("SELECT id, name FROM tennis_players").fetchall()}

    dates = [r["date"] for r in conn.execute(
        "SELECT DISTINCT date FROM tennis_matches WHERE date >= ? ORDER BY date",
        (start_date,)).fetchall()]
    logger.info(f"Backfilling surface across {len(dates)} distinct dates from {start_date}...")

    updated = 0
    no_match = 0
    async with httpx.AsyncClient(timeout=120) as client:
        for i, date in enumerate(dates):
            data = await _api_get(client, f"/match/list?sport_slug=tennis&date={date}")
            for m in _parse_events(data):
                wid = name_to_id.get(m["winner"])
                lid = name_to_id.get(m["loser"])
                if not wid or not lid:
                    no_match += 1
                    continue
                cur = conn.execute(
                    "UPDATE tennis_matches SET surface = ?, indoor = ? "
                    "WHERE date = ? AND winner_id = ? AND loser_id = ?",
                    (m["surface"], m["indoor"], date, wid, lid))
                updated += cur.rowcount

            if (i + 1) % 50 == 0:
                conn.commit()
                logger.info(f"  {i+1}/{len(dates)} dates ({date}) — {updated} rows updated so far")

    conn.commit()
    logger.info(f"Done. Updated {updated} rows; {no_match} API matches had no DB player.")

    logger.info("New surface distribution:")
    for r in conn.execute(
        "SELECT surface, COUNT(*) n FROM tennis_matches GROUP BY surface ORDER BY n DESC"
    ).fetchall():
        logger.info(f"  {r['surface']}: {r['n']}")
    conn.close()


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"
    asyncio.run(backfill(start))
