"""Retroactive odds correction - replace Polymarket odds with real SofaScore bookmaker odds.

For each resolved bet:
1. Try to find match in tennis_matches DB (has B365 odds)
2. If no DB odds, search SofaScore API by date + player names
3. Fetch real bookmaker odds from SofaScore /match/odds
4. Update our_odds and recalculate profit in bets.json
"""

import asyncio
import json
import logging
import sqlite3
import unicodedata
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from src.tennis.sofascore_api import (
    HEADERS, BASE_URL, _load_usage, _save_usage, _strip_diacritics,
    _sofascore_name_to_db, _parse_odds,
)

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
BETS_FILE = DATA_DIR / "bets.json"
BACKUP_FILE = DATA_DIR / "bets_backup_before_recalc.json"


def _normalize(name: str) -> str:
    """Normalize name for matching: strip diacritics, lowercase, no spaces."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace(" ", "").replace(".", "").replace("-", "").strip()


def _surname(name: str) -> str:
    """Extract surname for fuzzy matching."""
    parts = name.strip().split()
    return parts[-1] if parts else name


def _find_match_in_db(conn, player_name: str, opponent_name: str, bet_date: str) -> dict | None:
    """Try to find a match in tennis_matches by player names + date range."""
    p1_surname = _surname(player_name)
    p2_surname = _surname(opponent_name)

    # Try both orderings (player could be winner or loser)
    for w_name, l_name in [(p1_surname, p2_surname), (p2_surname, p1_surname)]:
        row = conn.execute("""
            SELECT m.id, m.b365_w, m.b365_l, m.winner_id, m.loser_id,
                   p1.name as winner_name, p2.name as loser_name
            FROM tennis_matches m
            JOIN tennis_players p1 ON m.winner_id = p1.id
            JOIN tennis_players p2 ON m.loser_id = p2.id
            WHERE p1.name LIKE ? AND p2.name LIKE ?
            AND m.date BETWEEN date(?, '-3 days') AND date(?, '+3 days')
            LIMIT 1
        """, (f"%{w_name}%", f"%{l_name}%", bet_date, bet_date)).fetchone()

        if row:
            return {
                "match_id": row["id"],
                "b365_w": row["b365_w"] or 0,
                "b365_l": row["b365_l"] or 0,
                "winner_name": row["winner_name"],
                "loser_name": row["loser_name"],
            }
    return None


async def _search_sofascore_match(client: httpx.AsyncClient, player_name: str, bet_date: str) -> int | None:
    """Search SofaScore for a match by date, return match ID if found."""
    usage = _load_usage()
    if usage["count"] >= 299900:  # Leave reserve
        logger.warning("API budget too low for retroactive search")
        return None

    url = f"{BASE_URL}/match/list"
    try:
        resp = await client.get(url, headers=HEADERS, params={
            "sport_slug": "tennis",
            "date": bet_date,
        }, timeout=120)
        usage["count"] += 1
        _save_usage(usage)

        if resp.status_code != 200:
            return None

        data = resp.json()
        events = data if isinstance(data, list) else data.get("events", [])
        p_norm = _normalize(player_name)

        for ev in events:
            home = ev.get("homeTeam", {}).get("name", "")
            away = ev.get("awayTeam", {}).get("name", "")
            if p_norm in _normalize(home) or p_norm in _normalize(away):
                return ev.get("id")
            # Also try surname match
            p_surname = _normalize(_surname(player_name))
            if p_surname in _normalize(home) or p_surname in _normalize(away):
                return ev.get("id")

    except Exception as e:
        logger.warning(f"SofaScore search failed for {player_name} on {bet_date}: {e}")
        usage["count"] += 1
        _save_usage(usage)

    return None


async def _fetch_match_odds(client: httpx.AsyncClient, match_id: int) -> dict | None:
    """Fetch bookmaker odds for a specific match from SofaScore."""
    usage = _load_usage()
    if usage["count"] >= 299900:
        return None

    url = f"{BASE_URL}/match/odds"
    try:
        resp = await client.get(url, headers=HEADERS, params={
            "match_id": match_id,
        }, timeout=120)
        usage["count"] += 1
        _save_usage(usage)

        if resp.status_code != 200:
            return None

        odds_data = resp.json()
        odds_list = odds_data if isinstance(odds_data, list) else []
        return _parse_odds(odds_list, "", "")

    except Exception as e:
        logger.warning(f"SofaScore odds fetch failed for match {match_id}: {e}")
        usage["count"] += 1
        _save_usage(usage)
        return None


def _get_correct_odds_for_bet(bet: dict, odds: dict, winner_name: str = None, loser_name: str = None) -> float | None:
    """Determine the correct bookmaker odds for this bet's pick.

    Returns the decimal odds for the player that was bet on.
    """
    ml = odds.get("moneyline", {})
    if not ml:
        return None

    p1_odds = ml.get("player1_odds", 0)
    p2_odds = ml.get("player2_odds", 0)
    if p1_odds <= 1 or p2_odds <= 1:
        return None

    pick_name = bet.get("team1_name", "")
    market = bet.get("market", "")

    # For WIN bets, return the moneyline odds of the pick
    if "WIN" in market:
        # If we know who is player1 (home) and who is player2 (away),
        # match by name. SofaScore player1 = home team.
        if winner_name and loser_name:
            pick_norm = _normalize(pick_name)
            winner_norm = _normalize(winner_name)
            # If pick was the favorite (lower odds = home usually)
            if pick_norm[:5] in winner_norm or winner_norm[:5] in pick_norm:
                return min(p1_odds, p2_odds)  # Winner was likely the favorite
            else:
                return max(p1_odds, p2_odds)  # Pick was the underdog

        # Fallback: assume lower odds = favorite
        return min(p1_odds, p2_odds)

    return None


async def recalculate_all_odds(dry_run: bool = True):
    """Main function: recalculate all bet odds from SofaScore data.

    Args:
        dry_run: If True, only show what would change without modifying files.
    """
    # Load bets
    with open(BETS_FILE, "r") as f:
        bets = json.load(f)

    if not dry_run:
        # Backup original
        with open(BACKUP_FILE, "w") as f:
            json.dump(bets, f, indent=2)
        logger.info(f"Backup saved to {BACKUP_FILE}")

    conn = sqlite3.connect(str(DATA_DIR / "tennis.db"))
    conn.row_factory = sqlite3.Row

    resolved = [(i, b) for i, b in enumerate(bets) if b.get("status") in ("won", "lost")]
    logger.info(f"Processing {len(resolved)} resolved bets...")

    updated = 0
    skipped = 0
    api_lookups = 0

    # Cache SofaScore date lookups to avoid duplicate API calls
    date_cache = {}  # date -> already searched flag

    async with httpx.AsyncClient(timeout=120) as client:
        for idx, bet in resolved:
            pick_name = bet.get("team1_name", "")
            opp_name = bet.get("team2_name", "")
            old_odds = bet.get("our_odds", 0)
            bet_date = bet.get("created_at", "")[:10]
            market = bet.get("market", "")
            status = bet.get("status", "")

            # Skip doubles
            if "/" in pick_name or "/" in opp_name:
                skipped += 1
                continue

            # Skip non-WIN bets for now
            if "WIN" not in market:
                skipped += 1
                continue

            # Step 1: Try DB lookup with B365 odds
            db_match = _find_match_in_db(conn, pick_name, opp_name, bet_date)
            new_odds = None

            if db_match and db_match["b365_w"] > 0:
                # Determine if our pick was winner or loser
                pick_norm = _normalize(pick_name)
                winner_norm = _normalize(db_match["winner_name"])
                if pick_norm[:5] in winner_norm or winner_norm[:5] in pick_norm:
                    new_odds = db_match["b365_w"]  # Our pick won → use winner odds
                else:
                    new_odds = db_match["b365_l"]  # Our pick lost → use loser odds
                source = "db_b365"

            # Step 2: If no DB odds, try SofaScore API
            if new_odds is None and api_lookups < 100:
                ss_match_id = bet.get("match_id", 0)

                # If no match_id, search by date
                if not ss_match_id and bet_date not in date_cache:
                    ss_match_id = await _search_sofascore_match(client, pick_name, bet_date)
                    api_lookups += 1
                    date_cache[bet_date] = True

                    # Also try next day (matches can span midnight)
                    if not ss_match_id:
                        next_day = (datetime.strptime(bet_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                        ss_match_id = await _search_sofascore_match(client, pick_name, next_day)
                        api_lookups += 1

                if ss_match_id:
                    odds_data = await _fetch_match_odds(client, ss_match_id)
                    api_lookups += 1
                    if odds_data:
                        ml = odds_data.get("moneyline", {})
                        if ml:
                            p1_odds = ml.get("player1_odds", 0)
                            p2_odds = ml.get("player2_odds", 0)
                            if p1_odds > 1 and p2_odds > 1:
                                # Pick the lower odds (favorite) if bet was winner
                                if status == "won":
                                    new_odds = min(p1_odds, p2_odds)
                                else:
                                    new_odds = max(p1_odds, p2_odds)
                                source = "sofascore_api"

            if new_odds and new_odds > 1 and abs(new_odds - old_odds) > 0.01:
                old_profit = bet.get("profit", 0)
                stake = bet.get("stake", 1000)

                if status == "won":
                    new_profit = round(stake * (new_odds - 1), 2)
                else:
                    new_profit = -stake

                diff = round(new_odds - old_odds, 2)
                logger.info(
                    f"  {pick_name} vs {opp_name} ({bet_date}): "
                    f"odds {old_odds} → {new_odds} ({'+' if diff > 0 else ''}{diff}) "
                    f"profit {old_profit} → {new_profit} [{source}]"
                )

                if not dry_run:
                    bets[idx]["our_odds"] = new_odds
                    bets[idx]["profit"] = new_profit
                    bets[idx]["odds_source"] = source
                    bets[idx]["original_pm_odds"] = old_odds

                updated += 1
            else:
                skipped += 1

    conn.close()

    logger.info(f"\nDone: {updated} updated, {skipped} skipped, {api_lookups} API calls used")

    if not dry_run and updated > 0:
        with open(BETS_FILE, "w") as f:
            json.dump(bets, f, indent=2)
        logger.info(f"Updated bets saved to {BETS_FILE}")

    return {"updated": updated, "skipped": skipped, "api_calls": api_lookups}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Default: dry run (preview changes)
    import sys
    dry = "--apply" not in sys.argv
    if dry:
        print("=== DRY RUN (use --apply to save changes) ===\n")
    else:
        print("=== APPLYING CHANGES ===\n")
    asyncio.run(recalculate_all_odds(dry_run=dry))
