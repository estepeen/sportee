"""April backfill - fill all gaps + enrich bets with SofaScore data.

1. Backfill match list for days with gaps (get event_ids + new matches)
2. Fetch match statistics for matches missing stats
3. Enrich bets with: event_id, SofaScore URL, tournament, round, match date
4. Auto-resolve pending bets from finished matches
5. Setup improved daily cron
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
from src.tennis.sofascore_api import (
    _api_get as _ss_api_get, _get_or_create_player,
    _sofascore_name_to_db as name_to_db, _is_main_tour as is_main_tour,
    HEADERS, BASE_URL, get_usage,
    _load_usage, _save_usage,
)

def strip_d(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

async def _api_get(client, endpoint, usage):
    if usage[0] >= LIMIT:
        return None
    result = await _ss_api_get(client, endpoint)
    usage[0] = _load_usage().get("count", 0)
    return result

def load_usage():
    return _load_usage().get("count", 0)

def save_usage(count):
    _save_usage({"month": datetime.now().strftime("%Y-%m"), "count": count})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
LIMIT = 30000

def strip_d(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# ─── Phase 1: Backfill match lists (get event_ids + missing matches) ──

async def phase1_backfill_matches():
    """Fetch match lists for days with incomplete data."""
    conn = get_tennis_db()
    usage = [load_usage()]

    # Find days needing backfill: March 2026 (0 event_ids) + gaps
    dates_to_fetch = []

    # All of March 2026
    d = datetime(2026, 3, 1)
    while d <= datetime(2026, 3, 31):
        dates_to_fetch.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    # Feb 2026 days missing event_ids
    d = datetime(2026, 2, 1)
    while d <= datetime(2026, 2, 28):
        ds = d.strftime("%Y-%m-%d")
        eid_count = conn.execute("SELECT COUNT(*) FROM tennis_matches WHERE date=? AND sofascore_event_id > 0", (ds,)).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM tennis_matches WHERE date=?", (ds,)).fetchone()[0]
        if total > 0 and eid_count < total * 0.5:
            dates_to_fetch.append(ds)
        d += timedelta(days=1)

    # Jan gaps
    for ds in ["2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31"]:
        dates_to_fetch.append(ds)

    dates_to_fetch = sorted(set(dates_to_fetch))
    logger.info(f"Phase 1: Backfilling {len(dates_to_fetch)} days (usage: {usage[0]}/{LIMIT})")

    total_added = 0
    total_mapped = 0

    async with httpx.AsyncClient(timeout=90) as client:
        for ds in dates_to_fetch:
            data = await _api_get(client, f"/match/list?sport_slug=tennis&date={ds}", usage)
            if not data:
                continue

            events = data if isinstance(data, list) else data.get("events", [])
            added = 0
            mapped = 0

            for ev in events:
                try:
                    status = ev.get("status", {})
                    if not status.get("isFinished"):
                        continue
                    if not is_main_tour(ev):
                        continue

                    eid = ev.get("id", 0)
                    home = ev.get("homeTeam", {})
                    away = ev.get("awayTeam", {})
                    home_name = home.get("name", "")
                    away_name = away.get("name", "")
                    if not home_name or not away_name:
                        continue

                    hs = ev.get("homeScore", {})
                    if not isinstance(hs, dict):
                        hs = {}
                    aws = ev.get("awayScore", {})
                    if not isinstance(aws, dict):
                        aws = {}
                    h_sets = hs.get("current", 0) or 0
                    a_sets = aws.get("current", 0) or 0
                    if h_sets == a_sets:
                        continue

                    winner_db = name_to_db(home_name if h_sets > a_sets else away_name)
                    loser_db = name_to_db(away_name if h_sets > a_sets else home_name)
                    wid = _get_or_create_player(conn, winner_db)
                    lid = _get_or_create_player(conn, loser_db)

                    # Try to update existing match with event_id
                    existing = conn.execute(
                        "SELECT id, sofascore_event_id FROM tennis_matches WHERE date=? AND winner_id=? AND loser_id=? LIMIT 1",
                        (ds, wid, lid)
                    ).fetchone()

                    tournament = ev.get("uniqueTournament", {}).get("name", "") or ev.get("tournament", {}).get("name", "")
                    cat = ev.get("tournament", {}).get("category", {}).get("name", "")
                    tour = "WTA" if "wta" in cat.lower() else "ATP"
                    rnd = ev.get("round", {}).get("name", "")
                    ground = ev.get("groundType", "")
                    surface_map = {"hardcourt": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}
                    surface = surface_map.get(ground, "Hard")
                    h_rank = home.get("ranking") or 0
                    a_rank = away.get("ranking") or 0
                    w_rank = h_rank if h_sets > a_sets else a_rank
                    l_rank = a_rank if h_sets > a_sets else h_rank

                    if existing:
                        if not existing["sofascore_event_id"]:
                            conn.execute("UPDATE tennis_matches SET sofascore_event_id=?, tournament=?, round=? WHERE id=?",
                                         (eid, tournament, rnd, existing["id"]))
                            mapped += 1
                    else:
                        # Insert new match
                        sd = {}
                        for i in range(1, 6):
                            hp = hs.get(f"period{i}")
                            ap = aws.get(f"period{i}")
                            if hp is not None and ap is not None:
                                if h_sets > a_sets:
                                    sd[f"w{i}"] = hp; sd[f"l{i}"] = ap
                                else:
                                    sd[f"w{i}"] = ap; sd[f"l{i}"] = hp

                        conn.execute("""
                            INSERT OR IGNORE INTO tennis_matches (
                                date, tournament, surface, indoor, round, best_of,
                                series, tour, winner_id, loser_id,
                                winner_rank, loser_rank, w_sets, l_sets,
                                w1, l1, w2, l2, w3, l3, w4, l4, w5, l5,
                                sofascore_event_id
                            ) VALUES (?,?,?,0,?,3,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            ds, tournament, surface, rnd, tour, tour,
                            wid, lid, w_rank, l_rank,
                            max(h_sets, a_sets), min(h_sets, a_sets),
                            sd.get("w1"), sd.get("l1"), sd.get("w2"), sd.get("l2"),
                            sd.get("w3"), sd.get("l3"), sd.get("w4"), sd.get("l4"),
                            sd.get("w5"), sd.get("l5"), eid,
                        ))
                        added += 1
                except Exception:
                    continue

            conn.commit()
            total_added += added
            total_mapped += mapped
            if added > 0 or mapped > 0:
                logger.info(f"  {ds}: +{added} new, {mapped} mapped (API: {usage[0]})")

            if usage[0] >= LIMIT:
                break

    save_usage(usage[0])
    logger.info(f"Phase 1 done: +{total_added} matches, {total_mapped} mapped. API: {usage[0]}")
    conn.close()


# ─── Phase 2: Fetch stats for matches missing them ──

async def phase2_stats():
    """Fetch match statistics for 2026 matches with event_id but no stats."""
    conn = get_tennis_db()
    usage = [load_usage()]

    matches = conn.execute("""
        SELECT id, sofascore_event_id, winner_id, date FROM tennis_matches
        WHERE sofascore_event_id > 0 AND w_aces = 0
        AND date >= '2026-01-01'
        ORDER BY date DESC
    """).fetchall()

    logger.info(f"Phase 2: Fetching stats for {len(matches)} matches (API: {usage[0]})")

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

    saved = 0
    async with httpx.AsyncClient(timeout=30) as client:
        for i, m in enumerate(matches):
            data = await _api_get(client, f"/match/statistics?match_id={m['sofascore_event_id']}", usage)
            if not data:
                continue

            stats_list = data if isinstance(data, list) else data.get("statistics", [])
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
                            if isinstance(hv, str):
                                hv = int(hv.replace("%", "")) if "%" in hv else 0
                            if isinstance(av, str):
                                av = int(av.replace("%", "")) if "%" in av else 0
                            w_col, l_col = STAT_MAP[key]
                            parsed[w_col] = hv
                            parsed[l_col] = av

            if parsed:
                sets = ", ".join(f"{k}=?" for k in parsed)
                vals = list(parsed.values()) + [m["id"]]
                conn.execute(f"UPDATE tennis_matches SET {sets} WHERE id = ?", vals)
                saved += 1

            if (i + 1) % 200 == 0:
                conn.commit()
                save_usage(usage[0])
                logger.info(f"  Stats: {i+1}/{len(matches)} ({saved} saved, API: {usage[0]})")

            if usage[0] >= LIMIT:
                break
            await asyncio.sleep(0.3)

    conn.commit()
    save_usage(usage[0])
    logger.info(f"Phase 2 done: {saved} matches with stats. API: {usage[0]}")
    conn.close()


# ─── Phase 3: Enrich bets ──

def phase3_enrich_bets():
    """Add event_id, SofaScore URL, tournament, round, score to all bets."""
    conn = get_tennis_db()
    from src.tennis.stats import find_player

    bets = json.loads(open(DATA_DIR / "bets.json").read())
    enriched = 0

    for b in bets:
        t1 = b.get("team1_name", "")
        t2 = b.get("team2_name", "")
        if not t1 or not t2:
            continue

        p1 = find_player(conn, t1)
        p2 = find_player(conn, t2)
        if not p1 or not p2:
            continue

        row = conn.execute("""
            SELECT m.sofascore_event_id, m.tournament, m.round, m.date,
                   m.w_sets, m.l_sets, m.w1, m.l1, m.w2, m.l2, m.w3, m.l3,
                   m.winner_id, p1.name as wname, p2.name as lname
            FROM tennis_matches m
            JOIN tennis_players p1 ON m.winner_id = p1.id
            JOIN tennis_players p2 ON m.loser_id = p2.id
            WHERE ((m.winner_id=? AND m.loser_id=?) OR (m.winner_id=? AND m.loser_id=?))
            AND m.date >= '2026-03-01'
            ORDER BY m.date DESC LIMIT 1
        """, (p1["id"], p2["id"], p2["id"], p1["id"])).fetchone()

        if not row:
            continue

        changed = False

        # SofaScore URL
        eid = row["sofascore_event_id"]
        if eid:
            s1 = "-".join(strip_d(t1).lower().replace(".", "").split())
            s2 = "-".join(strip_d(t2).lower().replace(".", "").split())
            b["pm_url"] = f"https://www.sofascore.com/tennis/match/{s1}-{s2}/-#id:{eid}"
            changed = True

        # Tournament + round
        if row["tournament"] and not b.get("event"):
            b["event"] = row["tournament"]
            changed = True

        # Match date
        if row["date"]:
            b["created_at"] = f"{row['date']}T12:00:00"
            changed = True

        # Score for resolved bets
        if b["status"] in ("won", "lost") and not b.get("actual_result"):
            sc = " ".join(f"{row[f'w{i}']}-{row[f'l{i}']}" for i in range(1, 4) if row[f"w{i}"] is not None)
            if sc:
                b["actual_result"] = f"{row['wname']} {row['w_sets']}-{row['l_sets']} ({sc})"
                changed = True

        if changed:
            enriched += 1

    open(DATA_DIR / "bets.json", "w").write(json.dumps(bets, indent=2))
    conn.close()
    logger.info(f"Phase 3: Enriched {enriched} bets with SofaScore data")


# ─── Phase 4: Auto-resolve pending bets ──

def phase4_resolve():
    """Resolve pending bets from finished matches in DB."""
    conn = get_tennis_db()
    from src.tennis.stats import find_player
    from src.strategy.bet_manager import BetManager

    bm = BetManager()
    bm.load()
    pending = [b for b in bm.bets if b["status"] == "pending"]
    resolved = 0

    for bet in pending:
        label = bet.get("market_label", "")
        t1 = bet.get("team1_name", "")
        t2 = bet.get("team2_name", "")
        pick_name = label.replace(" WIN", "").replace(" +1.5 SETS", "").replace(" +1.5 Sets", "").replace(" TOTAL", "").strip()

        p1 = find_player(conn, t1)
        p2 = find_player(conn, t2)
        pick_p = find_player(conn, pick_name)
        if not p1 or not p2 or not pick_p:
            continue

        row = conn.execute("""
            SELECT winner_id, w_sets, l_sets, w1, l1, w2, l2, w3, l3,
                   p1.name as wname, p2.name as lname
            FROM tennis_matches m
            JOIN tennis_players p1 ON m.winner_id = p1.id
            JOIN tennis_players p2 ON m.loser_id = p2.id
            WHERE ((m.winner_id=? AND m.loser_id=?) OR (m.winner_id=? AND m.loser_id=?))
            AND m.date >= '2026-03-24'
            ORDER BY m.date DESC LIMIT 1
        """, (p1["id"], p2["id"], p2["id"], p1["id"])).fetchone()

        if not row:
            continue

        if "WIN" in label and "1.5" not in label:
            won = row["winner_id"] == pick_p["id"]
        elif "1.5" in label:
            won = row["winner_id"] == pick_p["id"] or row["l_sets"] >= 1
        else:
            continue

        sc = " ".join(f"{row[f'w{i}']}-{row[f'l{i}']}" for i in range(1, 4) if row[f"w{i}"] is not None)
        bm.resolve_bet(bet["id"], "won" if won else "lost")
        bet["actual_result"] = f"{row['wname']} {row['w_sets']}-{row['l_sets']} ({sc})" if sc else f"{row['wname']} {row['w_sets']}-{row['l_sets']}"
        resolved += 1

    bm.save()
    conn.close()
    logger.info(f"Phase 4: Resolved {resolved} pending bets")


# ─── Main ──

async def main():
    logger.info(f"=== APRIL BACKFILL === Usage: {get_usage()}")

    await phase1_backfill_matches()

    logger.info("Recomputing Elo...")
    from src.tennis.elo import compute_all_elo
    compute_all_elo()

    await phase2_stats()

    phase3_enrich_bets()
    phase4_resolve()

    logger.info(f"=== DONE === Usage: {get_usage()}")


if __name__ == "__main__":
    asyncio.run(main())
