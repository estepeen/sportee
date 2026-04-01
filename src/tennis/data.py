"""Fetch and import tennis match data from tennis-data.co.uk."""

import logging
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd

from src.tennis.database import get_tennis_db, init_tennis_db

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# tennis-data.co.uk URLs
# ATP: http://www.tennis-data.co.uk/{year}/{year}.xlsx
# WTA: http://www.tennis-data.co.uk/{year}w/{year}.xlsx
BASE_URL = "http://www.tennis-data.co.uk"

# Map series names to our categories
SERIES_MAP = {
    "Grand Slam": "Grand Slam",
    "Masters 1000": "Masters 1000",
    "Masters": "Masters 1000",
    "ATP1000": "Masters 1000",
    "ATP500": "ATP 500",
    "ATP250": "ATP 250",
    "International": "ATP 250",
    "International Gold": "ATP 500",
    "Premier Mandatory": "WTA 1000",
    "Premier 5": "WTA 1000",
    "Premier": "WTA 500",
    "WTA1000": "WTA 1000",
    "WTA500": "WTA 500",
    "WTA250": "WTA 250",
}

# Court speed mapping (approximate)
TOURNAMENT_SPEED = {
    # Fast hard
    "Australian Open": "medium",
    "US Open": "medium",
    "Indian Wells Masters": "medium",
    "Miami Masters": "medium",
    # Clay (slow)
    "French Open": "slow",
    "Rome Masters": "slow",
    "Madrid Masters": "medium",  # high altitude makes it faster
    "Monte Carlo Masters": "slow",
    "Barcelona": "slow",
    # Grass (fast)
    "Wimbledon": "fast",
    "Queens Club": "fast",
    "Halle": "fast",
}


def get_or_create_player(conn, name: str) -> int:
    """Get player ID, creating if not exists."""
    row = conn.execute(
        "SELECT id FROM tennis_players WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO tennis_players (name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute(
        "SELECT id FROM tennis_players WHERE name = ?", (name,)
    ).fetchone()["id"]


async def fetch_year(year: int, tour: str = "ATP") -> pd.DataFrame | None:
    """Download one year of match data from tennis-data.co.uk."""
    if tour == "WTA":
        url = f"{BASE_URL}/{year}w/{year}.xlsx"
    else:
        url = f"{BASE_URL}/{year}/{year}.xlsx"

    logger.info(f"Fetching {tour} {year} from {url}")

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)

    if resp.status_code != 200:
        logger.warning(f"Failed to fetch {url}: {resp.status_code}")
        return None

    try:
        df = pd.read_excel(BytesIO(resp.content))
    except Exception as e:
        logger.error(f"Failed to parse {url}: {e}")
        return None

    logger.info(f"Loaded {len(df)} matches for {tour} {year}")
    return df


def get_last_match_date() -> str:
    """Get the date of the most recent match in the database."""
    conn = get_tennis_db()
    row = conn.execute("SELECT MAX(date) as last_date FROM tennis_matches").fetchone()
    conn.close()
    return row["last_date"] or "2000-01-01"


def import_dataframe(df: pd.DataFrame, tour: str = "ATP", since: str = ""):
    """Import a tennis-data.co.uk DataFrame into the database.

    If `since` is set, only import matches after that date.
    """
    init_tennis_db()
    conn = get_tennis_db()
    imported = 0

    for _, row in df.iterrows():
        try:
            winner = str(row.get("Winner", "")).strip()
            loser = str(row.get("Loser", "")).strip()
            if not winner or not loser or winner == "nan" or loser == "nan":
                continue

            winner_id = get_or_create_player(conn, winner)
            loser_id = get_or_create_player(conn, loser)

            date = str(row.get("Date", ""))
            if hasattr(row.get("Date"), "strftime"):
                date = row["Date"].strftime("%Y-%m-%d")

            # Skip old matches if incremental
            if since and date <= since:
                continue

            tournament = str(row.get("Tournament", "")).strip()
            location = str(row.get("Location", "")).strip()
            surface = str(row.get("Surface", "")).strip()
            court = str(row.get("Court", "")).strip()
            indoor = 1 if court.lower() == "indoor" else 0
            rnd = str(row.get("Round", "")).strip()
            best_of = int(row.get("Best of", 3)) if pd.notna(row.get("Best of")) else 3

            series_raw = str(row.get("Series", "")).strip()
            series = SERIES_MAP.get(series_raw, series_raw)

            # Scores
            def safe_int(v):
                try:
                    return int(v) if pd.notna(v) else None
                except (ValueError, TypeError):
                    return None

            w_sets = safe_int(row.get("Wsets"))
            l_sets = safe_int(row.get("Lsets"))
            comment = str(row.get("Comment", "")).strip() if pd.notna(row.get("Comment")) else ""

            # Court speed
            speed = TOURNAMENT_SPEED.get(tournament, "")
            if not speed:
                sl = surface.lower()
                if sl == "clay":
                    speed = "slow"
                elif sl == "grass":
                    speed = "fast"
                elif sl == "hard" and indoor:
                    speed = "fast"
                else:
                    speed = "medium"

            # Betting odds
            b365_w = float(row.get("B365W", 0)) if pd.notna(row.get("B365W")) else 0
            b365_l = float(row.get("B365L", 0)) if pd.notna(row.get("B365L")) else 0
            ps_w = float(row.get("PSW", 0)) if pd.notna(row.get("PSW")) else 0
            ps_l = float(row.get("PSL", 0)) if pd.notna(row.get("PSL")) else 0

            conn.execute("""
                INSERT OR IGNORE INTO tennis_matches (
                    date, tournament, location, surface, indoor, court_speed,
                    round, best_of, series, tour,
                    winner_id, loser_id,
                    winner_rank, loser_rank,
                    winner_rank_points, loser_rank_points,
                    score, w_sets, l_sets,
                    w1, l1, w2, l2, w3, l3, w4, l4, w5, l5,
                    b365_w, b365_l, ps_w, ps_l, comment
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                date, tournament, location, surface, indoor, speed,
                rnd, best_of, series, tour,
                winner_id, loser_id,
                safe_int(row.get("WRank")), safe_int(row.get("LRank")),
                safe_int(row.get("WPts")), safe_int(row.get("LPts")),
                str(row.get("Score", "")) if pd.notna(row.get("Score")) else "",
                w_sets, l_sets,
                safe_int(row.get("W1")), safe_int(row.get("L1")),
                safe_int(row.get("W2")), safe_int(row.get("L2")),
                safe_int(row.get("W3")), safe_int(row.get("L3")),
                safe_int(row.get("W4")), safe_int(row.get("L4")),
                safe_int(row.get("W5")), safe_int(row.get("L5")),
                b365_w, b365_l, ps_w, ps_l, comment,
            ))
            imported += 1
        except Exception as e:
            logger.debug(f"Skip row: {e}")
            continue

    conn.commit()
    conn.close()
    logger.info(f"Imported {imported} {tour} matches")
    return imported


async def fetch_all(start_year: int = 2019, end_year: int = 2026):
    """Fetch and import all years of ATP + WTA data."""
    init_tennis_db()
    total = 0

    for year in range(start_year, end_year + 1):
        for tour_name in ("ATP", "WTA"):
            df = await fetch_year(year, tour_name)
            if df is not None and not df.empty:
                n = import_dataframe(df, tour_name)
                total += n

    # Update player count
    conn = get_tennis_db()
    players = conn.execute("SELECT COUNT(*) FROM tennis_players").fetchone()[0]
    matches = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    conn.close()

    logger.info(f"Tennis DB: {players} players, {matches} matches total")
    return total


async def fetch_incremental():
    """Fetch only new matches since the last match in DB. Fast daily update."""
    from datetime import datetime

    init_tennis_db()
    last_date = get_last_match_date()
    year = datetime.now().year

    logger.info(f"Incremental fetch: last match {last_date}, fetching {year} ATP+WTA")

    total = 0
    for tour_name in ("ATP", "WTA"):
        df = await fetch_year(year, tour_name)
        if df is not None and not df.empty:
            n = import_dataframe(df, tour_name, since=last_date)
            total += n

    conn = get_tennis_db()
    matches = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    conn.close()

    logger.info(f"Incremental: +{total} new matches (total: {matches})")
    return total
