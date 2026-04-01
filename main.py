"""Sportee - Sports Betting Intelligence."""

import asyncio
import logging
import sys

from src.database import init_db
from src.scraper.pandascore import PandaScoreClient
from src.scraper.hltv import HLTVScraper
from src.scraper.fundamentals import FundamentalsTracker
from src.scraper.players import PlayerTracker
from src.scraper.polymarket import PolymarketClient
from src.model.ml_predictor import MLPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def sync(deep: bool = False):
    """Primary sync via PandaScore + HLTV + fundamentals."""
    ps = PandaScoreClient()
    await ps.full_sync(deep=deep)
    await ps.close()

    hltv = HLTVScraper()
    await hltv.scrape_top_teams(count=30)
    await hltv.scrape_upcoming_matches()  # also saves HLTV odds
    await hltv.compute_team_map_stats(months=3)
    await hltv.compute_team_map_stats(months=6)

    # Player rosters + change detection
    pt = PlayerTracker()
    await pt.fetch_all_team_rosters(pages=5)
    changes = pt.detect_changes()
    pt.save()
    await pt.close()

    # Fundamentals: news scraping + alerts
    fund = FundamentalsTracker()
    await fund.full_update()

    # Polymarket real odds (CS2 + Tennis)
    pm = PolymarketClient()
    await pm.fetch_cs2_markets()
    await pm.fetch_tennis_markets()
    await pm.close()


async def train():
    """Train ML model on historical data."""
    predictor = MLPredictor()
    await predictor.train()


async def resolve_bets():
    """Auto-resolve pending bets from completed match results."""
    from sqlalchemy import select, text
    from src.database import async_session
    from src.strategy.bet_manager import BetManager

    bm = BetManager()
    pending = bm.get_pending_bets()
    if not pending:
        return

    logger.info(f"Checking {len(pending)} pending bets...")

    async with async_session() as session:
        result = await session.execute(text("""
            SELECT hltv_id as match_id, winner_id, team1_id, team2_id,
                   team1_score, team2_score
            FROM matches WHERE is_completed = 1
        """))
        completed = [dict(r._mapping) for r in result]

    resolved = bm.auto_resolve_bets(completed)
    if resolved:
        stats = bm.get_stats()
        logger.info(f"Resolved {resolved} bets | Bankroll: ${stats['bankroll']} | "
                    f"W/L: {stats['wins']}/{stats['losses']} | ROI: {stats['roi']}%")


async def predict():
    """Generate predictions and auto-record best bets."""
    predictor = MLPredictor()

    from sqlalchemy import select
    from src.database import async_session, Match
    from src.strategy.bet_manager import BetManager

    async with async_session() as session:
        result = await session.execute(
            select(Match).where(Match.is_completed == False).order_by(Match.date)
        )
        upcoming = result.scalars().all()

    if not upcoming:
        logger.info("No upcoming matches found.")
        return

    logger.info(f"Generating predictions for {len(upcoming)} matches...")

    # Collect predictions for bet manager
    all_predictions = []

    for match in upcoming:
        p = await predictor.predict_match(match.hltv_id)
        if not p:
            continue

        logger.info(
            f"\n{p.team1_name} vs {p.team2_name} (Bo{p.best_of}) [{p.tier}] {'LAN' if p.is_lan else 'Online'}\n"
            f"  ML: {p.team1_name} {p.team1_win_prob:.1%} | {p.team2_name} {p.team2_win_prob:.1%}\n"
            f"  Over 2.5: {p.over_2_5_prob:.1%}\n"
            f"  Confidence: {p.confidence:.1%}"
        )

        all_predictions.append({
            "match_id": p.match_id,
            "team1_name": p.team1_name, "team2_name": p.team2_name,
            "team1_ml_prob": p.team1_win_prob, "team2_ml_prob": p.team2_win_prob,
            "over_2_5_prob": p.over_2_5_prob,
            "team1_handicap": p.team1_plus_1_5_prob,
            "team2_handicap": p.team2_plus_1_5_prob,
            "best_of": p.best_of, "confidence": p.confidence,
            "event": p.event, "tier": p.tier, "is_lan": p.is_lan,
            "team1_rank": p.team1_rank, "team2_rank": p.team2_rank,
            "team1_elo": p.team1_elo, "team2_elo": p.team2_elo,
            "t1_streak": p.features.get("t1_streak", 0),
            "t2_streak": p.features.get("t2_streak", 0),
            "t1_form": p.features.get("t1_form5", 0.5),
            "t2_form": p.features.get("t2_form5", 0.5),
        })

    # Deduplicate predictions by team pair
    seen_pairs = set()
    deduped = []
    for pred in all_predictions:
        key = tuple(sorted([pred["team1_name"], pred["team2_name"]]))
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped.append(pred)
    all_predictions = deduped
    logger.info(f"Deduplicated: {len(all_predictions)} unique matches")

    # Attach Polymarket odds
    from src.scraper.polymarket import PolymarketClient
    pm_odds = PolymarketClient.load_odds()
    for pred in all_predictions:
        pm = PolymarketClient.find_match_odds(pred["team1_name"], pred["team2_name"], pm_odds)
        if pm:
            pred["pm_t1_odds"] = pm.get("team1_odds")
            pred["pm_t2_odds"] = pm.get("team2_odds")
            pred["pm_t1_price"] = pm.get("team1_price")
            pred["pm_t2_price"] = pm.get("team2_price")
        # Also look for handicap odds
        hc = PolymarketClient.find_handicap_odds(pred["team1_name"], pred["team2_name"], pm_odds)
        if hc:
            pred["pm_t1_hc_odds"] = hc.get("team1_hc_odds")
            pred["pm_t2_hc_odds"] = hc.get("team2_hc_odds")
            pred["pm_t1_hc_price"] = hc.get("team1_hc_price")
            pred["pm_t2_hc_price"] = hc.get("team2_hc_price")

    # Auto-record top bets (only matches with PM odds)
    bm = BetManager()
    pm_preds = [p for p in all_predictions if p.get("pm_t1_odds") and p.get("confidence", 0) >= 0.4]
    recommendations = bm.generate_recommendations(pm_preds)
    if recommendations:
        recorded = bm.auto_record_recommendations(recommendations)
        logger.info(f"Top recommendations: {len(recommendations)}, auto-recorded: {recorded}")


async def tennis_fetch(start_year: int = 2019):
    """Fetch historical tennis data and build Elo ratings."""
    from src.tennis.data import fetch_all
    from src.tennis.elo import compute_all_elo
    from src.tennis.players import sync_player_attributes

    logger.info("=== Fetching tennis data ===")
    await fetch_all(start_year=start_year)

    logger.info("=== Computing tennis Elo ===")
    compute_all_elo()

    logger.info("=== Fetching player bio data ===")
    from src.tennis.bio import fetch_bio_data
    await fetch_bio_data()

    logger.info("=== Syncing player attributes ===")
    sync_player_attributes()


async def tennis_update():
    """Quick daily tennis update: only new matches since last DB entry + Elo + PM odds."""
    from src.tennis.data import fetch_incremental
    from src.tennis.elo import compute_all_elo
    from src.tennis.players import sync_player_attributes

    logger.info("=== Tennis daily update (incremental) ===")
    new = await fetch_incremental()

    # tenisportal.cz = free backup for match results (no API limits)
    try:
        from src.tennis.tenisportal import import_recent as tp_import
        tp_new = await tp_import(days=10)
        new += tp_new
    except Exception as e:
        logger.warning(f"tenisportal.cz failed (non-critical): {e}")

    # Sofascore RapidAPI = match stats + additional results (500 req/month)
    try:
        from src.tennis.sofascore_api import import_recent as ss_import
        ss_new = await ss_import(days=10)
        new += ss_new
    except Exception as e:
        logger.warning(f"Sofascore API failed (non-critical): {e}")

    # Fetch upcoming matches + real bookmaker odds from SofaScore (1x daily)
    try:
        from src.tennis.sofascore_api import fetch_upcoming_with_odds, get_usage
        upcoming = await fetch_upcoming_with_odds(days=2)
        usage = get_usage()
        logger.info(f"SofaScore odds: {len(upcoming)} matches | API usage: {usage['used']}/300000")
    except Exception as e:
        logger.warning(f"SofaScore odds failed (non-critical): {e}")

    # Auto-resolve pending bets from SofaScore (reliable)
    try:
        from src.tennis.auto_resolve import resolve_from_sofascore
        resolved = await resolve_from_sofascore(days=3)
        if resolved:
            logger.info(f"Auto-resolved {resolved} tennis bets from SofaScore")
    except Exception as e:
        logger.warning(f"Auto-resolve failed: {e}")

    if new > 0:
        logger.info(f"{new} new matches → recomputing Elo + attributes")
        compute_all_elo()
        sync_player_attributes()
    else:
        logger.info("No new matches, skipping Elo recompute")

    # Doubles update (incremental)
    try:
        from src.tennis.doubles_sofascore import import_doubles_recent, fetch_doubles_upcoming_with_odds
        from src.tennis.doubles_elo import compute_doubles_elo
        dbl_new = await import_doubles_recent(days=5)
        if dbl_new:
            logger.info(f"{dbl_new} new doubles matches → recomputing doubles Elo")
            compute_doubles_elo()
        await fetch_doubles_upcoming_with_odds(days=2)
    except Exception as e:
        logger.warning(f"Doubles update failed (non-critical): {e}")

    # Lucky Loser system - scan main draw WDs + qualifying seeds, generate fade picks
    try:
        from src.tennis.lucky_loser import scan_qualifying, get_fade_stats
        ll_data = await scan_qualifying(days=3)
        stats = get_fade_stats()
        if stats["open"] or stats["wins"] or stats["losses"]:
            logger.info(
                f"LL FADE: {stats['open']} open, "
                f"{stats['wins']}W/{stats['losses']}L "
                f"({stats['winrate']}% WR)"
            )
    except Exception as e:
        logger.warning(f"Lucky Loser system failed (non-critical): {e}")

    # Always refresh Polymarket tennis odds
    pm = PolymarketClient()
    await pm.fetch_tennis_markets()
    await pm.close()

    # Auto-resolve pending bets from SofaScore finished matches (100% reliable)
    try:
        from src.tennis.auto_resolve import resolve_from_sofascore
        resolved = await resolve_from_sofascore(days=3)
        if resolved:
            logger.info(f"Auto-resolved {resolved} tennis bets from SofaScore")
    except Exception as e:
        logger.warning(f"Auto-resolve failed: {e}")


async def main():
    await init_db()

    if len(sys.argv) < 2:
        print("Usage: python main.py [update|deep|train|predict|web|tennis-init|tennis-update|tennis-train]")
        print("  update        - Quick CS2 sync + fundamentals + train + predict")
        print("  deep          - Deep CS2 sync (full history)")
        print("  train         - Train CS2 ML model")
        print("  predict       - Generate CS2 predictions")
        print("  web           - Launch dashboard")
        print("  tennis-init   - Fetch tennis history (2019-now) + build Elo")
        print("  tennis-update - Quick tennis update (current year + PM odds)")
        print("  tennis-train  - Retrain tennis model (global + surface-specific)")
        print("  doubles-init  - Fetch doubles history (2015-now) + build Elo")
        print("  doubles-update- Quick doubles update (recent matches + odds)")
        print("  doubles-train - Train doubles prediction model")
        return

    command = sys.argv[1]

    if command == "update":
        await sync(deep=False)
        await resolve_bets()
        await train()
        await predict()
    elif command == "deep":
        await sync(deep=True)
        await resolve_bets()
        await train()
        await predict()
    elif command == "train":
        await train()
    elif command == "predict":
        await predict()
    elif command == "tennis-init":
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
        await tennis_fetch(start_year=start)
    elif command == "tennis-update":
        await tennis_update()
    elif command == "tennis-train":
        from src.tennis.model import train_model as tennis_train_model
        logger.info("=== Retraining tennis model (global + surface-specific) ===")
        acc = tennis_train_model(since_year=2020)
        logger.info(f"Tennis model retrained, global accuracy: {acc:.4f}")
    elif command == "doubles-init":
        from src.tennis.doubles_data import fetch_all_doubles
        from src.tennis.doubles_elo import compute_doubles_elo
        from src.tennis.doubles_sofascore import import_doubles_recent
        from src.tennis.doubles_model import train_doubles_model
        logger.info("=== Fetching doubles history ===")
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
        await fetch_all_doubles(start_year=start)
        logger.info("=== Importing recent doubles from SofaScore (with stats) ===")
        await import_doubles_recent(days=90)
        logger.info("=== Computing doubles Elo ===")
        compute_doubles_elo()
        logger.info("=== Training doubles model ===")
        acc = train_doubles_model()
        if acc:
            logger.info(f"Doubles model trained, accuracy: {acc:.4f}")
    elif command == "doubles-update":
        from src.tennis.doubles_data import fetch_doubles_incremental
        from src.tennis.doubles_elo import compute_doubles_elo
        from src.tennis.doubles_sofascore import import_doubles_recent, fetch_doubles_upcoming_with_odds
        logger.info("=== Doubles daily update ===")
        new = await fetch_doubles_incremental()
        ss_new = await import_doubles_recent(days=10)
        if new + ss_new > 0:
            logger.info(f"{new + ss_new} new doubles matches → recomputing Elo")
            compute_doubles_elo()
        await fetch_doubles_upcoming_with_odds(days=2)
    elif command == "doubles-train":
        from src.tennis.doubles_model import train_doubles_model
        logger.info("=== Training doubles model ===")
        acc = train_doubles_model()
        if acc:
            logger.info(f"Doubles model trained, accuracy: {acc:.4f}")
    elif command == "web":
        import uvicorn
        from src.web.app import app
        config = uvicorn.Config(app, host="127.0.0.1", port=3003, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    asyncio.run(main())
