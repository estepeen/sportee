"""Sofascore tennis scraper - recent matches + detailed stats via Playwright."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.tennis.database import get_tennis_db, init_tennis_db

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
API_BASE = "https://api.sofascore.com/api/v1"


async def _fetch_api(page, endpoint: str) -> dict | list | None:
    """Fetch Sofascore API endpoint from within browser context."""
    url = f"{API_BASE}{endpoint}"
    try:
        result = await page.evaluate(f"""
            async () => {{
                const r = await fetch("{url}");
                if (!r.ok) return null;
                return await r.json();
            }}
        """)
        return result
    except Exception as e:
        logger.debug(f"API error {endpoint}: {e}")
        return None


async def _get_browser_page():
    """Launch Playwright browser and solve Cloudflare challenge."""
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    )
    ctx = await browser.new_context(
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    page = await ctx.new_page()

    # Solve Cloudflare challenge by visiting the site
    logger.info("Solving Cloudflare challenge...")
    await page.goto("https://www.sofascore.com", wait_until="domcontentloaded", timeout=45000)
    await asyncio.sleep(5)  # Let Cloudflare JS execute

    return pw, browser, page


def _get_or_create_player(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO tennis_players (name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()["id"]


def _sofascore_name_to_db(name: str) -> str:
    """Convert 'Jannik Sinner' to 'Sinner J.' format matching tennis-data.co.uk."""
    parts = name.strip().split()
    if len(parts) >= 2:
        first_initial = parts[0][0]
        last = parts[-1]
        return f"{last} {first_initial}."
    return name


async def fetch_recent_matches(days: int = 10):
    """Fetch recent tennis matches from Sofascore and import into DB."""
    init_tennis_db()
    conn = get_tennis_db()

    last_date = conn.execute("SELECT MAX(date) FROM tennis_matches").fetchone()[0] or "2020-01-01"
    logger.info(f"DB last date: {last_date}. Fetching Sofascore events for last {days} days...")

    pw, browser, page = await _get_browser_page()
    imported = 0

    try:
        for d in range(days):
            date = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
            if date <= last_date:
                logger.info(f"  {date} - already in DB, stopping")
                break

            logger.info(f"  Fetching {date}...")
            data = await _fetch_api(page, f"/sport/tennis/scheduled-events/{date}")
            if not data or "events" not in data:
                logger.warning(f"  No data for {date}")
                await asyncio.sleep(2)
                continue

            events = data["events"]
            for ev in events:
                try:
                    status = ev.get("status", {}).get("type", "")
                    if status != "finished":
                        continue

                    tournament = ev.get("tournament", {}).get("name", "")
                    ut = ev.get("tournament", {}).get("uniqueTournament", {})
                    series = ut.get("category", {}).get("name", "")

                    # Only ATP/WTA singles main draw
                    slug = ut.get("slug", "")
                    if "doubles" in slug or "qualifying" in tournament.lower():
                        continue

                    home = ev.get("homeTeam", {})
                    away = ev.get("awayTeam", {})
                    home_name = home.get("name", "")
                    away_name = away.get("name", "")
                    if not home_name or not away_name:
                        continue

                    home_score = ev.get("homeScore", {})
                    away_score = ev.get("awayScore", {})
                    h_sets = home_score.get("current", 0)
                    a_sets = away_score.get("current", 0)

                    winner_full = home_name if h_sets > a_sets else away_name
                    loser_full = away_name if h_sets > a_sets else home_name

                    winner_db = _sofascore_name_to_db(winner_full)
                    loser_db = _sofascore_name_to_db(loser_full)

                    winner_id = _get_or_create_player(conn, winner_db)
                    loser_id = _get_or_create_player(conn, loser_db)

                    # Surface
                    ground = ev.get("groundType", "")
                    surface_map = {"hardcourt": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}
                    surface = surface_map.get(ground, "Hard")

                    # Round
                    rnd = ev.get("roundInfo", {}).get("name", "")

                    # Rankings
                    h_rank = home.get("ranking", 0) or 0
                    a_rank = away.get("ranking", 0) or 0
                    w_rank = h_rank if h_sets > a_sets else a_rank
                    l_rank = a_rank if h_sets > a_sets else h_rank

                    # Set scores
                    set_scores = {}
                    for i in range(1, 6):
                        hp = home_score.get(f"period{i}")
                        ap = away_score.get(f"period{i}")
                        if hp is not None and ap is not None:
                            if h_sets > a_sets:
                                set_scores[f"w{i}"] = hp
                                set_scores[f"l{i}"] = ap
                            else:
                                set_scores[f"w{i}"] = ap
                                set_scores[f"l{i}"] = hp

                    # Tour detection
                    cat_name = ut.get("category", {}).get("name", "")
                    tour = "WTA" if "wta" in cat_name.lower() or "women" in cat_name.lower() else "ATP"

                    conn.execute("""
                        INSERT OR IGNORE INTO tennis_matches (
                            date, tournament, surface, indoor, round, best_of,
                            series, tour, winner_id, loser_id,
                            winner_rank, loser_rank, w_sets, l_sets,
                            w1, l1, w2, l2, w3, l3, w4, l4, w5, l5
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        date, tournament, surface, 0, rnd, 3,
                        series, tour, winner_id, loser_id,
                        w_rank, l_rank, max(h_sets, a_sets), min(h_sets, a_sets),
                        set_scores.get("w1"), set_scores.get("l1"),
                        set_scores.get("w2"), set_scores.get("l2"),
                        set_scores.get("w3"), set_scores.get("l3"),
                        set_scores.get("w4"), set_scores.get("l4"),
                        set_scores.get("w5"), set_scores.get("l5"),
                    ))
                    imported += 1
                except Exception as e:
                    logger.debug(f"  Skip event: {e}")
                    continue

            conn.commit()
            await asyncio.sleep(2)  # Rate limit

    finally:
        await browser.close()
        await pw.stop()

    total = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    new_last = conn.execute("SELECT MAX(date) FROM tennis_matches").fetchone()[0]
    conn.close()

    logger.info(f"Sofascore import: +{imported} matches (total: {total}, last: {new_last})")
    return imported


async def fetch_match_stats(event_id: int, page) -> dict | None:
    """Fetch detailed match statistics for a specific event."""
    data = await _fetch_api(page, f"/event/{event_id}/statistics")
    if not data or "statistics" not in data:
        return None

    stats = {}
    for period in data["statistics"]:
        period_name = period.get("period", "ALL")
        if period_name != "ALL":
            continue
        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                key = item.get("key", "")
                stats[f"home_{key}"] = item.get("homeValue", 0)
                stats[f"away_{key}"] = item.get("awayValue", 0)

    return stats
