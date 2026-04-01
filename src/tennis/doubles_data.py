"""Fetch and import doubles match data from tennis-data.co.uk + SofaScore stats."""

import logging
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd

from src.tennis.database import get_tennis_db, init_tennis_db
from src.tennis.data import get_or_create_player, SERIES_MAP

logger = logging.getLogger(__name__)

BASE_URL = "http://www.tennis-data.co.uk"


async def fetch_doubles_year(year: int, tour: str = "ATP") -> pd.DataFrame | None:
    """Download one year of doubles data from tennis-data.co.uk.

    ATP doubles: http://www.tennis-data.co.uk/{year}/{year}.xlsx (same file, doubles sheet or rows)
    Some years have separate doubles files or mixed in the same Excel.
    """
    # tennis-data.co.uk has doubles in separate files for some years
    if tour == "WTA":
        urls = [
            f"{BASE_URL}/{year}w/{year}.xlsx",  # might contain doubles
        ]
    else:
        urls = [
            f"{BASE_URL}/{year}/{year}.xlsx",  # main file
        ]

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue

                df = pd.read_excel(BytesIO(resp.content))

                # Check if this has doubles columns (Winner1, Winner2 or WTeam)
                # tennis-data.co.uk format varies by year
                # Some have "Winner" = "Player1/Player2" format
                if "Winner" in df.columns:
                    # Filter to doubles only: Winner field contains "/"
                    doubles_mask = df["Winner"].astype(str).str.contains("/", na=False)
                    doubles_df = df[doubles_mask].copy()
                    if len(doubles_df) > 0:
                        logger.info(f"Found {len(doubles_df)} doubles matches in {tour} {year}")
                        return doubles_df

                logger.info(f"No doubles found in {tour} {year}")
                return None

            except Exception as e:
                logger.debug(f"Error fetching {url}: {e}")
                continue

    return None


def _split_doubles_name(name: str) -> tuple[str, str]:
    """Split 'Player1/Player2' or 'Player1 / Player2' into two names."""
    if "/" not in name:
        return name.strip(), ""
    parts = name.split("/")
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def import_doubles_dataframe(df: pd.DataFrame, tour: str = "ATP", since: str = ""):
    """Import doubles matches from tennis-data.co.uk DataFrame."""
    init_tennis_db()
    conn = get_tennis_db()
    imported = 0

    for _, row in df.iterrows():
        try:
            winner_raw = str(row.get("Winner", "")).strip()
            loser_raw = str(row.get("Loser", "")).strip()
            if not winner_raw or not loser_raw:
                continue

            # Split doubles pairs
            w1_name, w2_name = _split_doubles_name(winner_raw)
            l1_name, l2_name = _split_doubles_name(loser_raw)
            if not w1_name or not w2_name or not l1_name or not l2_name:
                continue

            w1_id = get_or_create_player(conn, w1_name)
            w2_id = get_or_create_player(conn, w2_name)
            l1_id = get_or_create_player(conn, l1_name)
            l2_id = get_or_create_player(conn, l2_name)

            date = str(row.get("Date", ""))
            if hasattr(row.get("Date"), "strftime"):
                date = row["Date"].strftime("%Y-%m-%d")
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

            def safe_int(v):
                try:
                    return int(v) if pd.notna(v) else None
                except (ValueError, TypeError):
                    return None

            w_sets = safe_int(row.get("Wsets"))
            l_sets = safe_int(row.get("Lsets"))
            comment = str(row.get("Comment", "")).strip() if pd.notna(row.get("Comment")) else ""

            b365_w = float(row.get("B365W", 0)) if pd.notna(row.get("B365W")) else 0
            b365_l = float(row.get("B365L", 0)) if pd.notna(row.get("B365L")) else 0

            conn.execute("""
                INSERT OR IGNORE INTO doubles_matches (
                    date, tournament, location, surface, indoor, round, best_of,
                    series, tour,
                    w_player1_id, w_player2_id, l_player1_id, l_player2_id,
                    w_rank, l_rank,
                    score, w_sets, l_sets,
                    w1, l1, w2, l2, w3, l3,
                    b365_w, b365_l, comment
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                date, tournament, location, surface, indoor, rnd, best_of,
                series, tour,
                w1_id, w2_id, l1_id, l2_id,
                safe_int(row.get("WRank")) or 0, safe_int(row.get("LRank")) or 0,
                str(row.get("Score", "")) if pd.notna(row.get("Score")) else "",
                w_sets, l_sets,
                safe_int(row.get("W1")), safe_int(row.get("L1")),
                safe_int(row.get("W2")), safe_int(row.get("L2")),
                safe_int(row.get("W3")), safe_int(row.get("L3")),
                b365_w, b365_l, comment,
            ))
            imported += 1
        except Exception as e:
            logger.debug(f"Skip doubles row: {e}")
            continue

    conn.commit()
    conn.close()
    logger.info(f"Imported {imported} {tour} doubles matches")
    return imported


async def fetch_all_doubles(start_year: int = 2015, end_year: int = 2026):
    """Fetch and import all years of ATP + WTA doubles data."""
    init_tennis_db()
    total = 0

    for year in range(start_year, end_year + 1):
        for tour_name in ("ATP", "WTA"):
            df = await fetch_doubles_year(year, tour_name)
            if df is not None and not df.empty:
                n = import_doubles_dataframe(df, tour_name)
                total += n

    conn = get_tennis_db()
    matches = conn.execute("SELECT COUNT(*) FROM doubles_matches").fetchone()[0]
    conn.close()

    logger.info(f"Doubles DB: {matches} total matches")
    return total


async def fetch_doubles_incremental():
    """Fetch only new doubles matches since last DB entry."""
    from datetime import datetime

    init_tennis_db()
    conn = get_tennis_db()
    last_date = conn.execute("SELECT MAX(date) FROM doubles_matches").fetchone()[0] or "2020-01-01"
    conn.close()

    year = datetime.now().year
    logger.info(f"Doubles incremental: last match {last_date}, fetching {year}")

    total = 0
    for tour_name in ("ATP", "WTA"):
        df = await fetch_doubles_year(year, tour_name)
        if df is not None and not df.empty:
            n = import_doubles_dataframe(df, tour_name, since=last_date)
            total += n

    logger.info(f"Doubles incremental: +{total} new matches")
    return total
