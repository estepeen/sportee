"""Scrape recent tennis matches from tenisportal.cz - free, no protection, backup source."""

import logging
import re
import unicodedata
from datetime import datetime, timedelta

import httpx
from bs4 import BeautifulSoup

from src.tennis.database import get_tennis_db, init_tennis_db

logger = logging.getLogger(__name__)

BASE_URL = "https://www.tenisportal.cz/zapasy/"

SURFACE_MAP = {
    "miami": "Hard", "indian-wells": "Hard", "australian-open": "Hard",
    "us-open": "Hard", "canadian-open": "Hard", "cincinnati": "Hard",
    "shanghai": "Hard", "dubai": "Hard", "doha": "Hard", "beijing": "Hard",
    "acapulco": "Hard", "rotterdam": "Hard", "san-diego": "Hard",
    "tokyo": "Hard", "basel": "Hard", "vienna": "Hard", "paris": "Hard",
    "roland-garros": "Clay", "rome": "Clay", "madrid": "Clay",
    "monte-carlo": "Clay", "barcelona": "Clay", "hamburg": "Clay",
    "buenos-aires": "Clay", "rio": "Clay", "lyon": "Clay",
    "wimbledon": "Grass", "halle": "Grass", "queens": "Grass",
    "stuttgart": "Grass", "eastbourne": "Grass",
}


def _detect_surface(tournament: str) -> str:
    slug = tournament.lower().replace(" ", "-")
    for key, surface in SURFACE_MAP.items():
        if key in slug:
            return surface
    return "Hard"


def _strip_diacritics(s: str) -> str:
    """Remove diacritics: Menšík -> Mensik, Čilić -> Cilic."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _get_or_create_player(conn, name: str) -> int:
    # Normalize diacritics to match existing DB records
    normalized = _strip_diacritics(name)
    row = conn.execute("SELECT id FROM tennis_players WHERE name = ?", (normalized,)).fetchone()
    if not row:
        # Also try original name (in case DB has diacritics)
        row = conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO tennis_players (name) VALUES (?)", (normalized,))
    conn.commit()
    return conn.execute("SELECT id FROM tennis_players WHERE name = ?", (normalized,)).fetchone()["id"]


async def scrape_day(date_str: str) -> list[dict]:
    """Scrape all finished matches for a given date from tenisportal.cz."""
    url = f"{BASE_URL}?date={date_str}"
    logger.info(f"Scraping tenisportal.cz {date_str}...")

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        })

    if resp.status_code != 200:
        logger.warning(f"tenisportal.cz returned {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    matches = []

    # Match rows have player links
    match_rows = [
        r for r in soup.find_all("div", class_="t-tables-table-overview__row")
        if r.find("a", href=lambda h: h and "/hrac/" in h)
    ]

    current_tournament = ""
    current_tour = "ATP"
    current_href = ""
    skip_tournament = False

    for row in match_rows:
        try:
            # Tournament from preceding h3
            prev_h3 = row.find_previous("h3")
            if prev_h3:
                t = prev_h3.get_text(strip=True)
                if t and t != current_tournament:
                    current_tournament = t
                    link = prev_h3.find("a")
                    current_href = link.get("href", "") if link else ""
                    # Detect tour from URL: /miami/2026/atp-muzi/ or /miami/2026/wta-zeny/
                    current_tour = "WTA" if "wta" in current_href.lower() else "ATP"
                    # Skip doubles, futures, qualifying only (keep challengers)
                    lower = (t + " " + current_href).lower()
                    skip_tournament = any(w in lower for w in [
                        "futures", "kval", "qualification",
                        "double", "čtyřhr", "mix", "utr", "itf",
                    ])

            if skip_tournament:
                continue

            # Player names (abbreviated mobile format: "Sinner J.")
            player_links = row.find_all("a", href=re.compile(r"/hrac/"))
            names = []
            for link in player_links:
                mobile_span = link.find("span", class_=lambda c: c and "d-inline-block" in c and "d-sm-none" in c)
                if mobile_span:
                    names.append(mobile_span.get_text(strip=True))
            if len(names) < 2:
                continue

            # Direct child cells: [players+time, sets, set1, set2, set3, odds, ?, ?]
            cells = row.find_all("div", recursive=False)
            if len(cells) < 3:
                continue

            # Sets cell (index 1): "2 0" or "2 1"
            sets_text = cells[1].get_text(" ", strip=True).split()
            sets_nums = [int(x) for x in sets_text if x.isdigit()]
            if len(sets_nums) < 2:
                continue

            p1_sets = sets_nums[0]
            p2_sets = sets_nums[1]
            if p1_sets == p2_sets:
                continue  # unfinished

            # Validate: winner must have 2 or 3 sets, loser 0-2
            w_sets = max(p1_sets, p2_sets)
            l_sets_check = min(p1_sets, p2_sets)
            if w_sets not in (2, 3) or l_sets_check > 2:
                continue  # invalid score (likely live game score)

            # Set scores from cells 2, 3, 4
            set_games = []
            for ci in range(2, min(5, len(cells))):
                nums = cells[ci].get_text(" ", strip=True).split()
                digits = [int(x) for x in nums if x.isdigit()]
                if len(digits) >= 2:
                    set_games.append((digits[0], digits[1]))

            # Odds cell (index 5)
            p1_odds = 0.0
            p2_odds = 0.0
            if len(cells) > 5:
                odds_text = cells[5].get_text(" ", strip=True).split()
                odds_nums = []
                for x in odds_text:
                    try:
                        odds_nums.append(float(x))
                    except ValueError:
                        pass
                if len(odds_nums) >= 2:
                    p1_odds = odds_nums[0]
                    p2_odds = odds_nums[1]

            # Determine winner/loser
            winner_name = names[0] if p1_sets > p2_sets else names[1]
            loser_name = names[1] if p1_sets > p2_sets else names[0]
            w_sets = max(p1_sets, p2_sets)
            l_sets = min(p1_sets, p2_sets)

            # Reorder set scores for winner/loser
            set_data = {}
            for i, (g1, g2) in enumerate(set_games):
                sn = i + 1
                if p1_sets > p2_sets:
                    set_data[f"w{sn}"] = g1
                    set_data[f"l{sn}"] = g2
                else:
                    set_data[f"w{sn}"] = g2
                    set_data[f"l{sn}"] = g1

            # Winner odds
            w_odds = p1_odds if p1_sets > p2_sets else p2_odds
            l_odds = p2_odds if p1_sets > p2_sets else p1_odds

            surface = _detect_surface(current_tournament)

            matches.append({
                "date": date_str,
                "tournament": current_tournament,
                "surface": surface,
                "tour": current_tour,
                "winner": winner_name,
                "loser": loser_name,
                "w_sets": w_sets,
                "l_sets": l_sets,
                "set_data": set_data,
                "w_odds": w_odds,
                "l_odds": l_odds,
            })
        except Exception as e:
            logger.debug(f"  Skip row: {e}")
            continue

    logger.info(f"  Found {len(matches)} finished matches on {date_str}")
    return matches


async def import_recent(days: int = 10):
    """Scrape recent days and import new matches into DB as backup source."""
    init_tennis_db()
    conn = get_tennis_db()

    last_date = conn.execute("SELECT MAX(date) FROM tennis_matches").fetchone()[0] or "2020-01-01"
    logger.info(f"DB last date: {last_date}. Scraping tenisportal.cz for last {days} days...")

    total_imported = 0

    for d in range(days):
        date = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")

        if d > 1 and date <= last_date:
            logger.info(f"  {date} already covered, stopping")
            break

        matches = await scrape_day(date)

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
                        w_sets, l_sets,
                        w1, l1, w2, l2, w3, l3,
                        b365_w, b365_l
                    ) VALUES (?,?,?,0,'',3,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    m["date"], m["tournament"], m["surface"],
                    m["tour"], m["tour"],
                    winner_id, loser_id,
                    m["w_sets"], m["l_sets"],
                    sd.get("w1"), sd.get("l1"),
                    sd.get("w2"), sd.get("l2"),
                    sd.get("w3"), sd.get("l3"),
                    m["w_odds"], m["l_odds"],
                ))
                total_imported += 1
            except Exception as e:
                logger.debug(f"  Skip match: {e}")
                continue

    conn.commit()
    new_last = conn.execute("SELECT MAX(date) FROM tennis_matches").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    conn.close()

    logger.info(f"tenisportal.cz: +{total_imported} new matches (total DB: {total}, last: {new_last})")
    return total_imported
