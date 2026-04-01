"""Fetch player biographical data from Jeff Sackmann's tennis GitHub repos."""

import csv
import logging
from io import StringIO

import httpx

from src.tennis.database import get_tennis_db

logger = logging.getLogger(__name__)

ATP_PLAYERS_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
WTA_PLAYERS_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_players.csv"


def _normalize_name(first: str, last: str) -> list[str]:
    """Generate possible name variants for fuzzy matching."""
    f = first.strip()
    l = last.strip()
    variants = [
        f"{l} {f[0]}.",        # "Sinner J."
        f"{f} {l}",            # "Jannik Sinner"
        f"{l} {f}",            # "Sinner Jannik"
        f"{f[0]}. {l}",        # "J. Sinner"
        f"{l}",                # "Sinner" (last resort)
    ]
    if len(f) > 1:
        variants.append(f"{l} {f[:2]}.")  # "Sinner Ja."
    return variants


async def fetch_bio_data():
    """Download player bio CSVs and update database."""
    conn = get_tennis_db()

    # Load all existing player names for matching
    existing = {}
    for row in conn.execute("SELECT id, name FROM tennis_players").fetchall():
        existing[row["name"].lower()] = row["id"]

    total_updated = 0

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for url, tour in [(ATP_PLAYERS_URL, "ATP"), (WTA_PLAYERS_URL, "WTA")]:
            logger.info(f"Fetching {tour} player bio from Sackmann GitHub...")

            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {url}: {resp.status_code}")
                continue

            text = resp.text
            reader = csv.DictReader(StringIO(text))

            matched = 0
            for row in reader:
                first = row.get("name_first", "").strip()
                last = row.get("name_last", "").strip()
                if not first or not last:
                    continue

                # Try to match with our DB
                player_id = None
                for variant in _normalize_name(first, last):
                    key = variant.lower()
                    if key in existing:
                        player_id = existing[key]
                        break

                if not player_id:
                    continue

                # Parse bio fields
                hand = row.get("hand", "").strip()
                hand = hand if hand in ("L", "R") else "U"

                dob = row.get("dob", "").strip()
                birth_year = 0
                if dob and len(dob) >= 4:
                    try:
                        birth_year = int(dob[:4])
                    except ValueError:
                        pass

                height = 0
                h_raw = row.get("height", "").strip()
                if h_raw:
                    try:
                        height = int(h_raw)
                    except ValueError:
                        pass

                country = row.get("ioc", "").strip()

                # Update DB
                conn.execute("""
                    UPDATE tennis_players SET
                        hand = CASE WHEN ? != 'U' THEN ? ELSE hand END,
                        birth_year = CASE WHEN ? > 0 THEN ? ELSE birth_year END,
                        height_cm = CASE WHEN ? > 0 THEN ? ELSE height_cm END,
                        country = CASE WHEN ? != '' THEN ? ELSE country END
                    WHERE id = ?
                """, (hand, hand, birth_year, birth_year, height, height, country, country, player_id))

                matched += 1

            logger.info(f"Matched {matched} {tour} players with bio data")
            total_updated += matched

    conn.commit()

    # Stats
    with_height = conn.execute("SELECT COUNT(*) FROM tennis_players WHERE height_cm > 0").fetchone()[0]
    with_birth = conn.execute("SELECT COUNT(*) FROM tennis_players WHERE birth_year > 0").fetchone()[0]
    with_country = conn.execute("SELECT COUNT(*) FROM tennis_players WHERE country != ''").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM tennis_players").fetchone()[0]

    conn.close()

    logger.info(f"Bio update complete: {total_updated} matched | "
                f"Height: {with_height}/{total} | Birth: {with_birth}/{total} | "
                f"Country: {with_country}/{total}")

    return total_updated
