#!/usr/bin/env python3
"""Seed My Positions with current AI Top Picks + Value Bets.

Run once to populate active bets without starting the web server.
Usage: python3 scripts/seed_positions.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    from datetime import datetime
    from src.scraper.polymarket import PolymarketClient
    from src.tennis.sofascore_api import load_sofascore_odds
    from src.tennis.picks_tracker import track_picks
    from src.strategy.bet_manager import BetManager

    # Import app helpers (pick generators, merge, enrich)
    from src.web.app import (
        _merge_tennis_odds,
        _enrich_with_predictions,
        _generate_smart_picks,
        _generate_value_bets,
        _auto_record_all_picks,
        bet_manager,
    )

    logger.info("Loading odds from Polymarket + SofaScore...")
    pm_matches = PolymarketClient.load_tennis_odds()
    ss_matches = load_sofascore_odds()
    all_matches = _merge_tennis_odds(ss_matches, pm_matches)

    today = datetime.now().strftime("%Y-%m-%d")
    upcoming = [m for m in all_matches if not m.get("is_live") and m.get("date", today) >= today]
    logger.info(f"Found {len(upcoming)} upcoming matches")

    upcoming = _enrich_with_predictions(upcoming)
    for m in upcoming:
        m["match_type"] = "singles"

    picks = _generate_smart_picks(upcoming)
    value_bets = _generate_value_bets(upcoming)

    for p in picks:
        p["source"] = "ai_picks"
        p["match_type"] = "singles"
    for v in value_bets:
        v["source"] = "value_bets"
        v["match_type"] = "singles"

    logger.info(f"Generated {len(picks)} AI picks, {len(value_bets)} value bets")

    # Show picks
    for i, p in enumerate(picks[:8], 1):
        stars = "★" * p.get("stars", 1)
        logger.info(f"  AI Pick #{i}: {p['pick']} {p['bet_type']} @ {p['odds']:.2f} (edge {p['edge']:.0f}%) {stars}")

    for i, v in enumerate(value_bets[:5], 1):
        logger.info(f"  Value #{i}: {v['pick']} {v['bet_type']} @ {v['odds']:.2f} (edge {v['edge']:.0f}%)")

    # Track for analytics
    try:
        track_picks(picks[:8], source="ai_picks")
        track_picks(value_bets, source="value_bets")
    except Exception as e:
        logger.warning(f"Track picks failed: {e}")

    # Auto-record to My Positions
    all_to_record = picks[:8] + value_bets
    if not all_to_record:
        logger.warning("No picks to record!")
        return

    bet_manager.load()
    logger.info(f"Current bankroll: ${bet_manager.bankroll:.0f}, existing bets: {len(bet_manager.bets)}")

    _auto_record_all_picks(all_to_record)

    bet_manager.load()  # reload after save
    pending = [b for b in bet_manager.bets if b["status"] == "pending"]
    logger.info(f"Done! Pending positions: {len(pending)}, bankroll: ${bet_manager.bankroll:.0f}")

    # Try doubles too
    try:
        from src.tennis.doubles_sofascore import load_doubles_odds
        from src.web.app import _enrich_doubles_with_predictions

        doubles_matches = load_doubles_odds()
        doubles_matches = _enrich_doubles_with_predictions(doubles_matches)
        for m in doubles_matches:
            m["match_type"] = "doubles"
            m.setdefault("player1", m.get("team1_name", ""))
            m.setdefault("player2", m.get("team2_name", ""))

        dbl_picks = _generate_smart_picks(doubles_matches)
        dbl_value = _generate_value_bets(doubles_matches)
        for p in dbl_picks:
            p["source"] = "ai_picks"
            p["match_type"] = "doubles"
        for v in dbl_value:
            v["source"] = "value_bets"
            v["match_type"] = "doubles"

        if dbl_picks or dbl_value:
            logger.info(f"Doubles: {len(dbl_picks)} picks, {len(dbl_value)} value bets")
            bet_manager.load()
            _auto_record_all_picks(dbl_picks[:5] + dbl_value)
            bet_manager.load()
            pending = [b for b in bet_manager.bets if b["status"] == "pending"]
            logger.info(f"After doubles: {len(pending)} total pending, bankroll: ${bet_manager.bankroll:.0f}")
    except Exception as e:
        logger.warning(f"Doubles failed: {e}")


if __name__ == "__main__":
    main()
