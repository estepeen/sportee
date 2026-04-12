"""Sportee - FastAPI web dashboard."""

import json
import logging
import math
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from config.settings import DB_PATH
from src.strategy.bet_manager import BetManager
from src.scraper.polymarket import PolymarketClient
from src.tennis.sofascore_api import load_sofascore_odds, get_usage as ss_usage
from src.tennis.model import load_model as load_tennis_model, _get_avg_stats
from src.tennis.picks_tracker import track_picks, resolve_picks, get_analytics
from src.tennis.lucky_loser import get_ll_warnings_for_matches, check_ll_risk, get_open_fade_picks
from src.tennis.doubles_sofascore import load_doubles_odds
from src.tennis.doubles_model import load_doubles_model as _load_dbl_model, predict_doubles_match
from src.tennis.live_tracker import get_live_state, get_cached_live

app = FastAPI(title="Sportee")
_tennis_model = None
_predictions_cache = {"matches": [], "picks": [], "updated": ""}
_cache_computing = False
_last_auto_resolve = ""  # ISO timestamp of last auto-resolve run
DATA_DIR = Path(__file__).parent.parent.parent / "data"
bet_manager = BetManager()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _player_url_filter(name: str, tour: str = "") -> str:
    """Jinja2 filter: player name -> URL slug."""
    slug = name.lower().replace(".", "").replace(" ", "-").rstrip("-")
    t = tour.lower() if tour else "atp"
    return f"/{t}/player/{slug}"


templates.env.filters["player_url"] = _player_url_filter

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "model.pkl"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_model():
    if not MODEL_PATH.exists():
        return None, None, {}
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"], data.get("elo_ratings", {})


def prob_to_odds(prob):
    if prob <= 0.01:
        return 99.0
    return round(1 / prob, 2)


# ─── Dashboard (Portfolio) ─────────────────────────────────

def _recompute_cache():
    """Recompute predictions cache in background."""
    global _predictions_cache, _cache_computing, _last_auto_resolve
    if _cache_computing:
        return
    _cache_computing = True
    try:
        # Fetch fresh data from APIs, then load
        try:
            import asyncio
            pm_client = PolymarketClient()
            asyncio.run(pm_client.fetch_tennis_markets())
            try:
                asyncio.run(pm_client.client.aclose())
            except Exception:
                pass
        except Exception as e:
            logging.getLogger(__name__).warning(f"Polymarket fetch failed, using cache: {e}")

        try:
            import asyncio
            from src.tennis.sofascore_api import fetch_upcoming_with_odds
            asyncio.run(fetch_upcoming_with_odds(days=2))
        except Exception as e:
            logging.getLogger(__name__).warning(f"SofaScore fetch failed, using cache: {e}")

        pm_matches = PolymarketClient.load_tennis_odds()
        ss_matches = load_sofascore_odds()
        all_matches = _merge_tennis_odds(ss_matches, pm_matches)

        today = datetime.now().strftime("%Y-%m-%d")
        upcoming = [m for m in all_matches if not m.get("is_live") and m.get("date", today) >= today]

        upcoming = _enrich_with_predictions(upcoming)

        # Tag match_type on singles
        for m in upcoming:
            m["match_type"] = "singles"

        picks = _generate_smart_picks(upcoming)
        value_bets = _generate_value_bets(upcoming)

        # Tag source + match_type
        for p in picks:
            p["source"] = "ai_picks"
            p["match_type"] = "singles"
        for v in value_bets:
            v["source"] = "value_bets"
            v["match_type"] = "singles"

        # Auto-resolve pending bets (max once per 12h to save API credits)
        try:
            now = datetime.now()
            last = datetime.fromisoformat(_last_auto_resolve) if _last_auto_resolve else datetime.min
            if (now - last).total_seconds() >= 43200:
                import asyncio
                from src.tennis.auto_resolve import resolve_from_sofascore
                resolved = asyncio.run(resolve_from_sofascore(days=3))
                _last_auto_resolve = now.isoformat()
                if resolved:
                    logging.getLogger(__name__).info(f"Auto-resolved {resolved} bets before recording new picks")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Auto-resolve in recompute failed: {e}")

        # Track picks for analytics + auto-add to My Positions
        try:
            track_picks(picks[:8], source="ai_picks")
            track_picks(value_bets, source="value_bets")
            bet_manager.load()
            _auto_record_all_picks(picks[:8] + value_bets)
        except Exception as e:
            import traceback
            logging.getLogger(__name__).error(f"Auto-record singles failed: {e}\n{traceback.format_exc()}")

        _predictions_cache = {
            "matches": upcoming,
            "all_matches": all_matches,
            "picks": picks[:8],
            "value_bets": value_bets,
            "doubles": [],
            "doubles_picks": [],
            "doubles_value": [],
            "updated": datetime.now().isoformat(),
        }
    except Exception as e:
        import traceback
        logging.getLogger(__name__).error(f"Cache recompute failed: {e}\n{traceback.format_exc()}")

    # Doubles (separate try so singles failure doesn't block doubles)
    try:
        # Fetch fresh doubles data
        try:
            import asyncio
            from src.tennis.doubles_sofascore import fetch_doubles_upcoming_with_odds
            asyncio.run(fetch_doubles_upcoming_with_odds(days=2))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Doubles fetch failed, using cache: {e}")

        doubles_matches = load_doubles_odds()
        doubles_matches = _enrich_doubles_with_predictions(doubles_matches)
        for m in doubles_matches:
            m["match_type"] = "doubles"
            # Map team names to player1/player2 for compatibility with smart picks
            m.setdefault("player1", m.get("team1_name", ""))
            m.setdefault("player2", m.get("team2_name", ""))
        _predictions_cache["doubles"] = doubles_matches

        # Generate doubles picks + value bets (same rules as singles)
        dbl_picks = _generate_smart_picks(doubles_matches)
        dbl_value = _generate_value_bets(doubles_matches)
        for p in dbl_picks:
            p["source"] = "ai_picks"
            p["match_type"] = "doubles"
        for v in dbl_value:
            v["source"] = "value_bets"
            v["match_type"] = "doubles"

        _predictions_cache["doubles_picks"] = dbl_picks[:5]
        _predictions_cache["doubles_value"] = dbl_value

        # Auto-record doubles picks
        try:
            track_picks(dbl_picks[:5], source="ai_picks_doubles")
            track_picks(dbl_value, source="value_bets_doubles")
            bet_manager.load()
            _auto_record_all_picks(dbl_picks[:5] + dbl_value)
        except Exception as e:
            import traceback
            logging.getLogger(__name__).error(f"Auto-record doubles failed: {e}\n{traceback.format_exc()}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Doubles cache failed: {e}")

    _cache_computing = False


@app.on_event("startup")
async def startup_cache():
    """Compute predictions cache on startup."""
    # Reset bankroll to default if it's been depleted by old bug
    bet_manager.load()
    if bet_manager.bankroll < 10_000:
        logging.getLogger(__name__).info(f"Resetting depleted bankroll from ${bet_manager.bankroll:.0f} to $1,000,000")
        bet_manager.bankroll = 1_000_000.0
        bet_manager.save()

    import threading
    threading.Thread(target=_recompute_cache, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    bet_stats = bet_manager.get_stats()
    recent_bets = bet_manager.get_recent_bets(10)

    # Use cached predictions (instant load)
    upcoming = _predictions_cache.get("matches", [])
    all_matches = _predictions_cache.get("all_matches", upcoming)
    top_picks = _predictions_cache.get("picks", [])
    value_bets = _predictions_cache.get("value_bets", [])

    # Trigger background recompute if cache is stale (>10 min)
    cache_age = _predictions_cache.get("updated", "")
    if not cache_age or (datetime.now() - datetime.fromisoformat(cache_age)).seconds > 600:
        import threading
        threading.Thread(target=_recompute_cache, daemon=True).start()

    doubles = _predictions_cache.get("doubles", [])
    dbl_picks = _predictions_cache.get("doubles_picks", [])
    dbl_value = _predictions_cache.get("doubles_value", [])

    # Merge singles + doubles picks
    all_picks = top_picks + dbl_picks
    all_picks.sort(key=lambda x: (-x.get("stars", 0), -x.get("confidence", 0)))
    all_value = value_bets + dbl_value
    all_value.sort(key=lambda x: -x.get("edge", 0))

    # "Safe Favorites" strategy: odds <= 1.5, flat $1000 stake
    resolved = [b for b in bet_manager.bets if b.get("status") in ("won", "lost")]
    safe_bets = [b for b in resolved if b.get("our_odds", 99) <= 1.5]
    safe_w = sum(1 for b in safe_bets if b["status"] == "won")
    safe_l = len(safe_bets) - safe_w
    safe_flat = 1000
    safe_profit = sum(safe_flat * (b["our_odds"] - 1) if b["status"] == "won" else -safe_flat for b in safe_bets)
    safe_staked = safe_flat * len(safe_bets)
    safe_strategy = {
        "name": "Safe Favorites",
        "desc": "Odds \u2264 1.50, flat $1,000",
        "bets": len(safe_bets),
        "wins": safe_w,
        "losses": safe_l,
        "winrate": round(safe_w / len(safe_bets) * 100, 1) if safe_bets else 0,
        "staked": safe_staked,
        "profit": round(safe_profit),
        "roi": round(safe_profit / safe_staked * 100, 1) if safe_staked else 0,
        "bankroll": round(1_000_000 + safe_profit),
        "recent": sorted(
            [{"label": b.get("market_label", ""), "odds": b.get("our_odds", 0),
              "result": b["status"], "profit": round(safe_flat * (b["our_odds"] - 1), 0) if b["status"] == "won" else -safe_flat,
              "event": b.get("event", ""), "date": b.get("created_at", "")[:10]}
             for b in safe_bets],
            key=lambda x: x["date"], reverse=True
        )[:10],
    }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "active_page": "portfolio",
        "bet_stats": bet_stats,
        "tennis_matches": upcoming,
        "all_tennis_matches": all_matches,
        "doubles_matches": doubles,
        "recent_bets": recent_bets,
        "top_picks": all_picks,
        "value_bets": all_value,
        "safe_strategy": safe_strategy,
        "sofascore_usage": ss_usage(),
    })


# ─── Active Bets ───────────────────────────────────────────

@app.get("/active-bets", response_class=HTMLResponse)
async def active_bets_page(request: Request, q: str = "", tour: str = ""):
    bet_manager.load()
    bet_stats = bet_manager.get_stats()
    pending = sorted(
        [b for b in bet_manager.bets if b.get("status") == "pending"],
        key=lambda b: b.get("created_at", ""),
        reverse=True,
    )

    if q:
        ql = q.lower()
        pending = [b for b in pending if ql in (b.get("market_label", "") or "").lower()
                   or ql in (b.get("team1_name", "") or "").lower()
                   or ql in (b.get("team2_name", "") or "").lower()]
    if tour:
        pending = [b for b in pending if tour.lower() in (b.get("event", "") or "").lower()]

    return templates.TemplateResponse("active_bets.html", {
        "request": request,
        "active_page": "active-bets",
        "bet_stats": bet_stats,
        "pending_bets": pending,
        "filter_q": q,
        "filter_tour": tour,
    })


# ─── Markets ──────────────────────────────────────────────

@app.get("/markets", response_class=HTMLResponse)
async def markets_page(request: Request):
    bet_stats = bet_manager.get_stats()

    # Use cached data
    all_matches = _predictions_cache.get("all_matches", [])
    if not all_matches:
        pm_matches = PolymarketClient.load_tennis_odds()
        ss_matches = load_sofascore_odds()
        all_matches = _merge_tennis_odds(ss_matches, pm_matches)

    return templates.TemplateResponse("markets.html", {
        "request": request,
        "active_page": "markets",
        "bet_stats": bet_stats,
        "tennis_matches": all_matches,
        "sofascore_usage": ss_usage(),
    })


def _normalize_name(name: str) -> str:
    import unicodedata
    s = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    return s.lower().replace(" ", "").replace(".", "").replace("-", "").strip()


def _parse_tournament_from_question(question: str) -> str:
    """Extract tournament name from Polymarket question like 'Bucharest Open: Player vs Player'."""
    if ":" in question:
        return question.split(":")[0].strip()
    return ""


def _merge_tennis_odds(ss_matches: list, pm_matches: list) -> list:
    """Merge SofaScore and Polymarket odds. PM odds preferred (we bet on Polymarket)."""
    seen = {}  # key -> index in merged
    merged = []

    for m in ss_matches:
        key = (_normalize_name(m["player1"]), _normalize_name(m["player2"]))
        key_rev = (key[1], key[0])
        idx = len(merged)
        seen[key] = idx
        seen[key_rev] = idx
        merged.append(m)

    for m in pm_matches:
        key = (_normalize_name(m.get("player1", "")), _normalize_name(m.get("player2", "")))
        key_rev = (key[1], key[0])
        matched_key = key if key in seen else key_rev if key_rev in seen else None
        if matched_key is not None:
            # Match exists from SofaScore — merge PM data and USE PM odds
            idx = seen[matched_key]
            existing = merged[idx]
            swapped = (matched_key != key)  # PM player order differs from SS
            existing["pm_url"] = m.get("pm_url", "")
            if not swapped:
                existing["player1_price"] = m.get("player1_price", 0)
                existing["player2_price"] = m.get("player2_price", 0)
            else:
                existing["player1_price"] = m.get("player2_price", 0)
                existing["player2_price"] = m.get("player1_price", 0)
            # Always prefer Polymarket odds (that's where we bet)
            if m.get("player1_odds"):
                if not swapped:
                    existing["player1_odds"] = m.get("player1_odds", 0)
                    existing["player2_odds"] = m.get("player2_odds", 0)
                else:
                    existing["player1_odds"] = m.get("player2_odds", 0)
                    existing["player2_odds"] = m.get("player1_odds", 0)
        else:
            # PM-only match — extract tournament from question
            if not m.get("tournament"):
                m["tournament"] = _parse_tournament_from_question(m.get("question", ""))
            m["source"] = "polymarket"
            idx = len(merged)
            merged.append(m)
            seen[key] = idx
            seen[(key[1], key[0])] = idx

    merged.sort(key=lambda x: (x.get("tournament", ""), x.get("player1", "")))
    return merged


def _enrich_with_predictions(matches: list) -> list:
    """Add ML predictions + edge + LL warnings to each match."""
    global _tennis_model
    if _tennis_model is None:
        _tennis_model = load_tennis_model()
    if not _tennis_model:
        return matches

    from src.tennis.database import get_tennis_db
    from src.tennis.features import build_match_features, SURFACE_MAP
    from src.tennis.stats import find_player, _strip_diacritics
    import pandas as pd

    conn = get_tennis_db()
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Lucky Loser warnings for all matches
    ll_warnings = get_ll_warnings_for_matches(matches)

    # Determine which model to use per match
    surface_models = _tennis_model.get("surface_models", {})

    for m in matches:
        try:
            p1_name = m.get("player1", "")
            p2_name = m.get("player2", "")
            if not p1_name or not p2_name:
                continue

            # LL risk flags
            if p1_name in ll_warnings:
                m["p1_ll_risk"] = ll_warnings[p1_name]
            if p2_name in ll_warnings:
                m["p2_ll_risk"] = ll_warnings[p2_name]

            p1 = find_player(conn, p1_name)
            p2 = find_player(conn, p2_name)
            if not p1 or not p2:
                continue

            surface = "Hard"  # default
            surface_key = "hard"
            tournament = m.get("tournament", "")
            round_name = m.get("round", "")

            features = build_match_features(p1["id"], p2["id"], surface,
                                            tournament=tournament,
                                            round_name=round_name)

            # Add stats features
            p1_stats = _get_avg_stats(conn, p1["id"], date_str)
            p2_stats = _get_avg_stats(conn, p2["id"], date_str)
            features["p1_avg_aces"] = p1_stats["aces"]
            features["p2_avg_aces"] = p2_stats["aces"]
            features["aces_diff"] = p1_stats["aces"] - p2_stats["aces"]
            features["p1_avg_df"] = p1_stats["df"]
            features["p2_avg_df"] = p2_stats["df"]
            features["p1_avg_bp_saved_pct"] = p1_stats["bp_saved_pct"]
            features["p2_avg_bp_saved_pct"] = p2_stats["bp_saved_pct"]
            features["p1_avg_1st_won_pct"] = p1_stats["first_won_pct"]
            features["p2_avg_1st_won_pct"] = p2_stats["first_won_pct"]

            # Use surface-specific model if available
            if surface_key in surface_models:
                model = surface_models[surface_key]["model"]
                feature_cols = _tennis_model.get("surface_features", _tennis_model["features"])
                m["model_used"] = surface_key
            else:
                model = _tennis_model["model"]
                feature_cols = _tennis_model["features"]
                m["model_used"] = "global"

            X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
            raw_prob = model.predict_proba(X)[0]

            # Clamp raw model output (no model should output >85% or <15%)
            p1_raw = max(0.15, min(0.85, float(raw_prob[1])))
            p2_raw = 1.0 - p1_raw

            # Blend with market odds (70% model, 30% market) to stay grounded
            mkt_p1 = m.get("player1_price", 0.5)
            mkt_p2 = m.get("player2_price", 0.5)
            mkt_sum = mkt_p1 + mkt_p2
            if mkt_sum > 0.01:
                mkt_p1_norm = mkt_p1 / mkt_sum
                mkt_p2_norm = mkt_p2 / mkt_sum
            else:
                mkt_p1_norm, mkt_p2_norm = 0.5, 0.5

            p1_final = 0.7 * p1_raw + 0.3 * mkt_p1_norm
            p2_final = 1.0 - p1_final

            m["ml_p1_prob"] = round(p1_final, 4)
            m["ml_p2_prob"] = round(p2_final, 4)
            m["ml_p1_odds"] = round(1 / p1_final, 2) if p1_final > 0.01 else 99.0
            m["ml_p2_odds"] = round(1 / p2_final, 2) if p2_final > 0.01 else 99.0

            # Edge = blended model prob - market implied prob
            m["p1_edge"] = round((p1_final - mkt_p1_norm) * 100, 1)
            m["p2_edge"] = round((p2_final - mkt_p2_norm) * 100, 1)

            # Value bet flag
            m["p1_value"] = m["p1_edge"] >= 5
            m["p2_value"] = m["p2_edge"] >= 5

            # Store features for pick reason builder
            m.update(features)
        except Exception:
            continue

    conn.close()
    return matches


def _enrich_doubles_with_predictions(matches: list) -> list:
    """Add ML predictions to doubles matches."""
    dbl_model = _load_dbl_model()
    if not dbl_model:
        return matches

    from src.tennis.database import get_tennis_db
    from src.tennis.stats import find_player
    import pandas as pd

    conn = get_tennis_db()

    for m in matches:
        try:
            # Find all 4 players
            p1a = find_player(conn, m.get("team1_p1", ""))
            p1b = find_player(conn, m.get("team1_p2", ""))
            p2a = find_player(conn, m.get("team2_p1", ""))
            p2b = find_player(conn, m.get("team2_p2", ""))
            if not p1a or not p1b or not p2a or not p2b:
                continue

            result = predict_doubles_match(p1a["id"], p1b["id"], p2a["id"], p2b["id"])
            if "error" in result:
                continue

            # Clamp + blend with market (same as singles)
            raw_p1 = max(0.15, min(0.85, result["t1_win_prob"]))
            raw_p2 = 1.0 - raw_p1

            mkt_p1 = m.get("player1_price", 0.5)
            mkt_p2 = m.get("player2_price", 0.5)
            mkt_sum = mkt_p1 + mkt_p2
            if mkt_sum > 0.01:
                mkt_p1_n = mkt_p1 / mkt_sum
                mkt_p2_n = mkt_p2 / mkt_sum
            else:
                mkt_p1_n, mkt_p2_n = 0.5, 0.5

            p1_final = 0.7 * raw_p1 + 0.3 * mkt_p1_n
            p2_final = 1.0 - p1_final

            m["ml_p1_prob"] = round(p1_final, 4)
            m["ml_p2_prob"] = round(p2_final, 4)
            m["ml_p1_odds"] = round(1 / p1_final, 2) if p1_final > 0.01 else 99.0
            m["ml_p2_odds"] = round(1 / p2_final, 2) if p2_final > 0.01 else 99.0
            m["p1_edge"] = round((p1_final - mkt_p1_n) * 100, 1)
            m["p2_edge"] = round((p2_final - mkt_p2_n) * 100, 1)
            m["p1_value"] = m["p1_edge"] >= 5
            m["p2_value"] = m["p2_edge"] >= 5
        except Exception:
            continue

    conn.close()
    return matches


def _generate_smart_picks(matches: list) -> list:
    """Generate smart bet recommendations.

    Strategy:
    - Favorites (AI >= 65%, odds 1.30-1.80): recommend WIN
    - Slight favorites (AI 55-65%, odds 1.40-2.20): recommend WIN if edge >= 5%
    - Underdogs (AI < 50%): recommend +1.5 sets or handicap games instead of WIN
    - Skip odds < 1.20 (no value) and > 2.50 for WIN bets
    """
    picks = []

    for m in matches:
        p1_prob = m.get("ml_p1_prob", 0)
        p2_prob = m.get("ml_p2_prob", 0)
        p1_odds = m.get("player1_odds", 0)
        p2_odds = m.get("player2_odds", 0)
        p1_edge = m.get("p1_edge", 0)
        p2_edge = m.get("p2_edge", 0)
        p1 = m.get("player1", "")
        p2 = m.get("player2", "")
        tourn = m.get("tournament", "")
        tour = m.get("tour", "")

        if not p1_prob or not p1 or not p2:
            continue

        # Determine favorite and underdog
        if p1_prob >= p2_prob:
            fav, fav_prob, fav_odds, fav_edge = p1, p1_prob, p1_odds, p1_edge
            dog, dog_prob, dog_odds, dog_edge = p2, p2_prob, p2_odds, p2_edge
            fav_is_p1 = True
        else:
            fav, fav_prob, fav_odds, fav_edge = p2, p2_prob, p2_odds, p2_edge
            dog, dog_prob, dog_odds, dog_edge = p1, p1_prob, p1_odds, p1_edge
            fav_is_p1 = False

        # Skip qualifying matches (LL risk)
        if any(w in tourn.lower() for w in ["qual", "kval", "qualifying"]):
            continue

        # LL warning: if a player is a Lucky Loser, flag in pick and reduce confidence
        p1_ll = m.get("p1_ll_risk")
        p2_ll = m.get("p2_ll_risk")
        ll_penalty = False
        if p1_ll or p2_ll:
            ll_penalty = True

        # === WIN bet (odds 1.20 - 1.80) ===
        # Higher odds require higher conviction (edge + prob thresholds)
        if 1.20 <= fav_odds <= 1.80 and fav_edge >= 5:
            # Conviction: all bets <= 1.80 need 55%+ prob
            if fav_prob < 0.55: continue
            stars = 3 if (fav_prob >= 0.75 and fav_edge >= 10) else 2 if fav_prob >= 0.65 else 1

            # Flat stake $1000 for all
            suggested_stake = 1000
            # LL penalty: reduce stars and add warning to reason
            pick_reason = _build_pick_reason(fav, dog, fav_prob, fav_odds, fav_edge, m)
            if ll_penalty:
                ll_who = p1_ll or p2_ll
                pick_reason += f" LL WARNING: {ll_who['warning']}"
                stars = max(1, stars - 1)

            picks.append({
                "pick": fav, "opponent": dog,
                "player1": p1, "player2": p2,
                "tournament": tourn, "tour": tour,
                "round": m.get("round", ""),
                "date": m.get("date", ""),
                "bet_type": "WIN",
                "ml_prob": fav_prob,
                "mkt_prob": 1 / fav_odds if fav_odds > 1 else 0.5,
                "edge": fav_edge,
                "odds": fav_odds,
                "stars": stars,
                "confidence": round(fav_prob * fav_edge / 10, 1),
                "suggested_stake": suggested_stake,
                "ll_risk": bool(ll_penalty),
                "reason": pick_reason,
                "match_type": m.get("match_type", "singles"),
                "sofascore_url": m.get("sofascore_url", ""),
                "sofascore_id": m.get("sofascore_id") or m.get("event_id") or 0,
                "pm_url": m.get("pm_url", ""),
                "player1_slug": m.get("player1_slug", ""),
                "player1_ss_id": m.get("player1_ss_id", 0),
                "player2_slug": m.get("player2_slug", ""),
                "player2_ss_id": m.get("player2_ss_id", 0),
            })

    # === LL FADE picks (qualifying seed fades) ===
    try:
        fade_picks = get_open_fade_picks()
        for fp in fade_picks:
            picks.append({
                "pick": fp["opponent"], "opponent": fp["seed_player"],
                "tournament": fp["tournament"], "tour": fp.get("tour", ""),
                "round": fp.get("round", "Qualifying"),
                "date": fp.get("date", ""),
                "bet_type": "LL_FADE",
                "ml_prob": 0.65,  # assumed ~65% win rate vs tanking seed
                "mkt_prob": 0.5,
                "edge": 15.0,
                "odds": 0,  # no odds available for qualifying
                "stars": 2,
                "confidence": 6.0,
                "suggested_stake": 300,
                "ll_risk": False,
                "reason": fp["reason"],
            })
    except Exception:
        pass

    # Deduplicate: only 1 pick per match (keep highest confidence)
    seen_matches = set()
    deduped = []
    picks.sort(key=lambda x: (-x["stars"], -x["confidence"]))
    for p in picks:
        match_key = tuple(sorted([p["pick"], p["opponent"]]))
        if match_key in seen_matches:
            continue
        seen_matches.add(match_key)
        deduped.append(p)

    # Add status (OPEN/WIN/LOSS) from DB
    _add_result_status(deduped)
    return deduped[:10]


def _win_odds_to_set_handicap(win_odds: float) -> float:
    """Convert WIN moneyline odds to estimated +1.5 sets handicap odds.

    Simple lookup based on typical bookmaker spreads:
    WIN @1.90 → +1.5 sets ~@1.30
    WIN @2.20 → +1.5 sets ~@1.40
    WIN @2.50 → +1.5 sets ~@1.50
    WIN @3.00 → +1.5 sets ~@1.65
    WIN @3.50 → +1.5 sets ~@1.75
    WIN @4.00 → +1.5 sets ~@1.85
    WIN @5.00 → +1.5 sets ~@2.00
    """
    if win_odds <= 1.90:
        return 1.25
    elif win_odds <= 2.20:
        return 1.35 + (win_odds - 1.90) * 0.17
    elif win_odds <= 2.50:
        return 1.40 + (win_odds - 2.20) * 0.33
    elif win_odds <= 3.00:
        return 1.50 + (win_odds - 2.50) * 0.30
    elif win_odds <= 3.50:
        return 1.65 + (win_odds - 3.00) * 0.20
    elif win_odds <= 4.00:
        return 1.75 + (win_odds - 3.50) * 0.20
    elif win_odds <= 5.00:
        return 1.85 + (win_odds - 4.00) * 0.15
    else:
        return 2.00 + (win_odds - 5.00) * 0.05


def _resolve_from_polymarket_sync():
    """Sync version of PM resolve for background cache."""
    import httpx, json as jjson
    try:
        with httpx.Client(timeout=15) as client:
            r = client.get("https://gamma-api.polymarket.com/markets", params={
                "tag_id": "864", "sports_market_types": "moneyline",
                "active": "true", "closed": "false", "limit": 200,
            })
            if r.status_code != 200:
                return
            markets = r.json()

        for mk in markets:
            outcomes = jjson.loads(mk.get("outcomes", "[]")) if isinstance(mk.get("outcomes"), str) else mk.get("outcomes", [])
            prices = jjson.loads(mk.get("outcomePrices", "[]")) if isinstance(mk.get("outcomePrices"), str) else mk.get("outcomePrices", [])
            if len(outcomes) < 2 or len(prices) < 2:
                continue
            p1_price, p2_price = float(prices[0]), float(prices[1])
            if p1_price < 0.95 and p2_price < 0.95:
                continue

            winner = outcomes[0] if p1_price >= 0.95 else outcomes[1]
            loser = outcomes[1] if p1_price >= 0.95 else outcomes[0]
            wn = _normalize_name(winner)

            # Try to get score from DB
            score_str = ""
            try:
                from src.tennis.database import get_tennis_db
                from src.tennis.stats import find_player
                conn = get_tennis_db()
                wp = find_player(conn, winner)
                lp = find_player(conn, loser)
                if wp and lp:
                    row = conn.execute("""
                        SELECT w1,l1,w2,l2,w3,l3,w_sets,l_sets FROM tennis_matches
                        WHERE winner_id=? AND loser_id=? AND date >= date('now','-5 days')
                        ORDER BY date DESC LIMIT 1
                    """, (wp["id"], lp["id"])).fetchone()
                    if row:
                        parts = []
                        for i in range(1, 4):
                            w, l = row[f"w{i}"], row[f"l{i}"]
                            if w is not None and l is not None:
                                parts.append(f"{w}-{l}")
                        score_str = " ".join(parts)
                conn.close()
            except:
                pass

            for bet in bet_manager.bets:
                if bet["status"] != "pending":
                    continue
                label = bet.get("market_label", "")
                t1n = _normalize_name(bet.get("team1_name", ""))
                t2n = _normalize_name(bet.get("team2_name", ""))
                pn = _normalize_name(label.replace(" WIN", "").replace(" +1.5 SETS", "").replace(" +1.5 Sets", "").replace(" TOTAL", "").strip())

                if not ((wn[:5] in t1n or wn[:5] in t2n)):
                    continue

                if "WIN" in label and "1.5" not in label:
                    won = wn[:5] in pn or pn[:5] in wn
                elif "1.5" in label:
                    won = True
                else:
                    continue

                bet_manager.resolve_bet(bet["id"], "won" if won else "lost")
                if score_str:
                    bet["actual_result"] = f"{winner} ({score_str})"
                else:
                    bet["actual_result"] = f"{winner} d. {loser}"

        bet_manager.save()
    except Exception:
        pass


def _build_pick_reason(pick: str, opp: str, prob: float, odds: float, edge: float, m: dict) -> str:
    """Build short analysis reason for a pick using all available features."""
    parts = []
    parts.append(f"AI: {round(prob*100)}% vs Market: {round(100/odds)}% (edge +{round(edge,1)}%).")

    # Elo
    p1_elo = m.get("p1_global_elo", 0) or m.get("elo_expected", 0)
    elo_diff = m.get("elo_diff", 0) or 0
    if abs(elo_diff) > 30:
        parts.append(f"Elo: {'+' if elo_diff > 0 else ''}{round(elo_diff)} pts.")

    # Rankings
    p1_rank = m.get("p1_rank", 0) or 0
    p2_rank = m.get("p2_rank", 0) or 0
    if p1_rank and p2_rank and p1_rank != p2_rank:
        parts.append(f"Rank: #{min(p1_rank,p2_rank)} vs #{max(p1_rank,p2_rank)}.")

    # Form
    p1_f5 = m.get("p1_form5", 0)
    p2_f5 = m.get("p2_form5", 0)
    if p1_f5 and p2_f5:
        parts.append(f"Form(5): {round(p1_f5*100)}% vs {round(p2_f5*100)}%.")

    # Surface form
    sf_diff = m.get("surface_form_diff", 0) or 0
    if abs(sf_diff) > 0.1:
        parts.append(f"Surface form {'advantage' if sf_diff > 0 else 'disadvantage'}.")

    # H2H
    h2h_total = m.get("h2h_total", 0) or 0
    h2h_wr = m.get("h2h_wr", 0.5)
    if h2h_total >= 2:
        w = round(h2h_wr * h2h_total)
        parts.append(f"H2H: {w}-{h2h_total - w}.")

    # Streak
    p1_streak = m.get("p1_streak", 0) or 0
    if p1_streak >= 3:
        parts.append(f"On W{p1_streak} streak.")
    elif p1_streak <= -3:
        parts.append(f"Warning: L{abs(p1_streak)} losing streak.")

    # Match stats
    aces_diff = m.get("aces_diff", 0) or 0
    if abs(aces_diff) > 2:
        parts.append(f"Serve: {'stronger' if aces_diff > 0 else 'weaker'} ({round(aces_diff,1)} aces/match diff).")

    # Risk level
    if odds < 1.40:
        parts.append("Heavy favorite.")
    elif odds < 1.70:
        parts.append("Clear favorite.")
    elif odds < 2.00:
        parts.append("Slight favorite.")
    elif odds < 3.00:
        parts.append("Competitive match — value play.")
    else:
        parts.append("Upset pick — high risk, high reward.")

    return " ".join(parts)


def _format_event_name(tournament: str, tour: str = "") -> str:
    """Format event name with tour + tier: 'ATP Houston (250)' or 'WTA Charleston'."""
    if not tournament:
        return ""
    t = tournament.lower()
    prefix = tour if tour else "ATP"

    # Known tiers
    masters = ["miami", "indian wells", "rome", "madrid", "monte carlo", "shanghai", "paris", "cincinnati", "canadian", "montreal", "toronto"]
    atp500 = ["houston", "dubai", "doha", "barcelona", "queen", "halle", "hamburg", "beijing", "vienna", "rotterdam", "acapulco", "rio"]
    slams = ["australian open", "roland garros", "wimbledon", "us open"]

    if any(w in t for w in slams):
        return f"{prefix} {tournament} (GS)"
    if any(w in t for w in masters):
        return f"{prefix} {tournament} (1000)"
    if any(w in t for w in atp500):
        return f"{prefix} {tournament} (500)"
    if "challenger" in t:
        return f"{tournament}"
    return f"{prefix} {tournament}"


def _find_pm_url(pick_data: dict) -> str:
    """Get Polymarket URL for the bet (primary), SofaScore as fallback."""
    # Prefer Polymarket URL (that's where we bet)
    pm_url = pick_data.get("pm_url", "")
    if pm_url:
        return pm_url

    # Fallback: SofaScore player page
    pick_name = pick_data.get("pick", "")
    p1 = pick_data.get("player1", "")
    if pick_name and _normalize_name(pick_name) == _normalize_name(p1 or pick_name):
        slug = pick_data.get("player1_slug", "")
        pid = pick_data.get("player1_ss_id", 0)
    else:
        slug = pick_data.get("player2_slug", "")
        pid = pick_data.get("player2_ss_id", 0)
    if slug and pid:
        return f"https://www.sofascore.com/tennis/player/{slug}/{pid}"

    ss_id = pick_data.get("sofascore_id") or 0
    if ss_id:
        return f"https://www.sofascore.com/event/{ss_id}"

    # Last resort: SofaScore search
    player1 = pick_name or p1
    player2 = pick_data.get("opponent", "") or pick_data.get("player2", "")
    q = f"{player1} {player2}".strip()
    if q:
        from urllib.parse import quote
        return f"https://www.sofascore.com/search?q={quote(q)}"
    return ""


def _auto_record_all_picks(all_picks: list):
    """Auto-add all AI picks and value bets to My Positions."""
    import fcntl
    logger = logging.getLogger(__name__)
    lock_path = DATA_DIR / ".bets.lock"
    try:
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        logger.warning("Auto-record skipped: lock file held by another process")
        return

    bet_manager.load()  # Fresh read before adding

    import unicodedata
    def _norm(s):
        """Strip diacritics, lowercase for matching."""
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()

    # Collect new picks that need recording (skip duplicates & resolved)
    new_picks = []
    batch_seen = set()  # track within current batch to avoid ai_picks + value_bets dupes
    for p in all_picks:
        pick = p.get("pick", "")
        opp = p.get("opponent", "")
        bet_type = p.get("bet_type", "WIN")
        label = f"{pick} {bet_type}"
        norm_pick = _norm(pick)
        norm_opp = _norm(opp)

        # Dedup within batch (same player+opponent regardless of source)
        batch_key = (norm_pick, norm_opp, bet_type)
        if batch_key in batch_seen:
            continue
        batch_seen.add(batch_key)

        # Dedup against existing bets (normalized name matching)
        existing = any(
            _norm(b.get("team1_name", "")) == norm_pick
            and bet_type in b.get("market_label", "")
            and b.get("status") == "pending"
            for b in bet_manager.bets
        )
        if existing:
            continue

        # Only skip if resolved TODAY (same players can play again tomorrow)
        pick_date = p.get("date", "")[:10]
        already_resolved = any(
            _norm(b.get("team1_name", "")) == norm_pick
            and _norm(b.get("team2_name", "")) == norm_opp
            and bet_type in b.get("market_label", "")
            and b.get("status") in ("won", "lost")
            and b.get("created_at", "")[:10] == pick_date
            for b in bet_manager.bets
        ) if pick_date else False
        if already_resolved:
            continue

        new_picks.append(p)

    if not new_picks:
        logger.info(f"Auto-record: 0 new picks (all {len(all_picks)} filtered by dedup/resolved)")
        try:
            lock_file.close()
        except Exception:
            pass
        return

    # Scale stakes to fit available bankroll so ALL picks get recorded
    total_needed = 0
    for p in new_picks:
        odds_val = p.get("odds", 2.0)
        default_stake = 1000 if odds_val < 1.50 else 600 if odds_val < 2.00 else 300 if odds_val < 3.00 else 150
        p["_stake"] = p.get("suggested_stake", default_stake)
        total_needed += p["_stake"]

    scale = 1.0
    if total_needed > bet_manager.bankroll and total_needed > 0:
        scale = max(bet_manager.bankroll / total_needed, 0.1)  # at least 10% of stake
        logger.info(f"Auto-record: scaling stakes by {scale:.2f} (need ${total_needed}, have ${bet_manager.bankroll:.0f})")

    from src.strategy.bet_manager import BetRecommendation

    recorded = 0
    for p in new_picks:
        pick = p.get("pick", "")
        opp = p.get("opponent", "")
        bet_type = p.get("bet_type", "WIN")
        label = f"{pick} {bet_type}"
        odds = p.get("odds", 2.0)
        stake = round(p["_stake"] * scale, 2)
        tourn = _format_event_name(p.get("tournament", ""), p.get("tour", ""))
        rnd = p.get("round", "")
        match_date = p.get("date", "")
        prob = p.get("ml_prob", 0.5)
        match_id = p.get("match_id", 0) or p.get("sofascore_id", 0) or 0

        if stake > bet_manager.bankroll:
            logger.warning(f"Auto-record: skip {pick} {bet_type} - stake ${stake} > bankroll ${bet_manager.bankroll:.0f}")
            continue

        # Deduct bankroll BEFORE record_bet so save() writes correct value
        bet_manager.bankroll -= stake

        rec = BetRecommendation(
            match_id=match_id, team1_name=pick, team2_name=opp,
            event=tourn, tier="", market=bet_type,
            market_label=label, our_prob=prob, our_odds=odds,
            edge=p.get("edge", 0), confidence=0,
            kelly_stake_pct=0, stake_amount=stake,
            rating=0, reasons=[], market_odds=odds,
        )
        record = bet_manager.record_bet(rec, auto_save=False)

        # Enrich with all available data
        pm_url = _find_pm_url(p)
        source = p.get("source", "unknown")
        for b in bet_manager.bets:
            if b.get("id") == record.id:
                if pm_url:
                    b["pm_url"] = pm_url
                b["source"] = source
                b["match_type"] = p.get("match_type", "singles")
                if rnd:
                    b["round"] = rnd
                if match_date:
                    b["created_at"] = match_date if "T" in match_date else f"{match_date}T12:00:00"
                # Always set reason — generate from metadata if missing
                if p.get("reason"):
                    b["reason"] = p["reason"]
                else:
                    ml = p.get("ml_prob", 0)
                    mkt = p.get("mkt_prob", 0)
                    edge = p.get("edge", 0)
                    b["reason"] = f"AI: {ml*100:.0f}% vs Market: {mkt*100:.0f}% (edge +{edge:.1f}%)"
                b["bet_meta"] = {
                    "ml_prob": p.get("ml_prob", 0),
                    "mkt_prob": p.get("mkt_prob", 0),
                    "edge": p.get("edge", 0),
                    "confidence": p.get("confidence") or p.get("stars", 0),
                    "tournament": p.get("tournament", ""),
                    "tour": p.get("tour", ""),
                    "round": rnd,
                    "bet_type": p.get("bet_type", "WIN"),
                    "match_type": p.get("match_type", "singles"),
                    "surface": p.get("surface", ""),
                }
                break
        recorded += 1

    bet_manager.save()
    logger.info(f"Auto-recorded {recorded} picks to My Positions (bankroll: ${bet_manager.bankroll:.0f})")
    try:
        lock_file.close()
    except Exception:
        pass


def _add_result_status(picks: list):
    """Check DB for finished matches and add status: OPEN/WIN/LOSS."""
    if not picks:
        return

    from src.tennis.database import get_tennis_db
    from src.tennis.stats import find_player

    try:
        conn = get_tennis_db()
        for p in picks:
            pick_name = p.get("pick", "")
            opp_name = p.get("opponent", "")

            pick_player = find_player(conn, pick_name)
            opp_player = find_player(conn, opp_name)
            if not pick_player or not opp_player:
                p["status"] = "OPEN"
                continue

            row = conn.execute("""
                SELECT winner_id, w_sets, l_sets FROM tennis_matches
                WHERE ((winner_id=? AND loser_id=?) OR (winner_id=? AND loser_id=?))
                AND date >= date('now', '-3 days')
                ORDER BY date DESC LIMIT 1
            """, (pick_player["id"], opp_player["id"], opp_player["id"], pick_player["id"])).fetchone()

            if not row:
                p["status"] = "OPEN"
                continue

            bet_type = p.get("bet_type", "WIN")
            if bet_type == "WIN":
                p["status"] = "WIN" if row["winner_id"] == pick_player["id"] else "LOSS"
            elif "+1.5" in bet_type:
                if row["winner_id"] == pick_player["id"]:
                    p["status"] = "WIN"
                else:
                    p["status"] = "WIN" if row["l_sets"] >= 1 else "LOSS"
            else:
                p["status"] = "OPEN"

        conn.close()
    except Exception:
        for p in picks:
            p.setdefault("status", "OPEN")


def _generate_value_bets(matches: list) -> list:
    """Generate all value bets (edge >= 10%).

    Rules: WIN only, odds 1.20-1.80, edge >= 10%, higher odds = higher conviction required.
    """
    bets = []
    seen = set()

    for m in matches:
        p1_prob = m.get("ml_p1_prob", 0)
        p2_prob = m.get("ml_p2_prob", 0)
        p1_odds = m.get("player1_odds", 0)
        p2_odds = m.get("player2_odds", 0)
        p1_edge = m.get("p1_edge", 0)
        p2_edge = m.get("p2_edge", 0)
        p1 = m.get("player1", "")
        p2 = m.get("player2", "")
        tourn = m.get("tournament", "")
        tour = m.get("tour", "")

        if not p1_prob or not p1:
            continue

        for side in [
            (p1, p2, p1_prob, p1_odds, p1_edge, True),
            (p2, p1, p2_prob, p2_odds, p2_edge, False),
        ]:
            name, opp, prob, odds, edge, is_p1 = side

            if edge < 10 or odds < 1.20 or odds > 1.80 or prob < 0.40:
                continue

            # Skip qualifying matches (LL risk)
            tourn_lower = tourn.lower()
            if any(w in tourn_lower for w in ["qual", "kval", "qualifying"]):
                continue

            match_key = tuple(sorted([name, opp]))
            if match_key in seen:
                continue
            seen.add(match_key)

            # WIN only — all <= 1.80
            bet_type = "WIN"
            confidence = "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.60 else "LOW"

            # Flat stake $1000
            suggested_stake = 1000

            mkt_prob = 1 / m.get(f"player{'1' if is_p1 else '2'}_odds", 2) if m.get(f"player{'1' if is_p1 else '2'}_odds", 0) > 1 else 0.5
            reason = _build_pick_reason(name, opp, prob, odds, edge, m)

            bets.append({
                "pick": name,
                "opponent": opp,
                "player1": p1, "player2": p2,
                "tournament": tourn,
                "tour": tour,
                "round": m.get("round", ""),
                "date": m.get("date", ""),
                "bet_type": bet_type,
                "ml_prob": prob,
                "mkt_prob": mkt_prob,
                "edge": edge,
                "odds": odds,
                "confidence": confidence,
                "suggested_stake": suggested_stake,
                "original_odds": m.get(f"player{'1' if is_p1 else '2'}_odds", 0),
                "match_type": m.get("match_type", "singles"),
                "sofascore_url": m.get("sofascore_url", ""),
                "sofascore_id": m.get("sofascore_id") or m.get("event_id") or 0,
                "pm_url": m.get("pm_url", ""),
                "reason": reason,
                "player1_slug": m.get("player1_slug", ""),
                "player1_ss_id": m.get("player1_ss_id", 0),
                "player2_slug": m.get("player2_slug", ""),
                "player2_ss_id": m.get("player2_ss_id", 0),
            })

    bets.sort(key=lambda x: -x["edge"])
    _add_result_status(bets)
    return bets


# ─── Analytics ────────────────────────────────────────────

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    bet_stats = bet_manager.get_stats()
    analytics = _build_analytics_from_bets()

    # Odds breakdown
    resolved = [b for b in bet_manager.bets if b.get("status") in ("won", "lost")]
    odds_breakdown = []
    for threshold in [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]:
        subset = [b for b in resolved if b.get("our_odds", 99) <= threshold]
        if not subset:
            odds_breakdown.append({"threshold": threshold, "bets": 0, "wins": 0, "losses": 0, "winrate": 0, "staked": 0, "profit": 0, "roi": 0})
            continue
        w = sum(1 for b in subset if b["status"] == "won")
        l = sum(1 for b in subset if b["status"] == "lost")
        staked = sum(b.get("stake", 0) for b in subset)
        profit = sum(b.get("profit", 0) for b in subset)
        odds_breakdown.append({
            "threshold": threshold,
            "bets": len(subset), "wins": w, "losses": l,
            "winrate": round(w / len(subset) * 100, 1),
            "staked": round(staked), "profit": round(profit),
            "roi": round(profit / staked * 100, 1) if staked else 0,
        })

    # Model training history
    model_history = []
    try:
        from src.tennis.database import get_tennis_db
        conn = get_tennis_db()
        rows = conn.execute("""
            SELECT * FROM model_history
            WHERE model_type = 'singles'
            ORDER BY trained_at DESC LIMIT 30
        """).fetchall()
        conn.close()
        model_history = [dict(r) for r in rows]
    except Exception:
        pass

    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "active_page": "analytics",
        "bet_stats": bet_stats,
        "a": analytics,
        "odds_breakdown": odds_breakdown,
        "model_history": model_history,
    })


def _detect_gender(b) -> str:
    """Detect ATP (Men) or WTA (Women) from bet data."""
    t1 = b.get("team1_name", "")
    try:
        from src.tennis.database import get_tennis_db
        conn = get_tennis_db()
        row = conn.execute("""
            SELECT m.tour FROM tennis_matches m
            JOIN tennis_players p ON (m.winner_id = p.id OR m.loser_id = p.id)
            WHERE p.name LIKE ? ORDER BY m.date DESC LIMIT 1
        """, (f"%{t1.split()[-1]}%" if t1 else "%",)).fetchone()
        conn.close()
        if row:
            return "WTA (Women)" if row["tour"] == "WTA" else "ATP (Men)"
    except:
        pass
    return "ATP (Men)"


def _build_analytics_from_bets() -> dict:
    """Build analytics from bets.json (the real source of truth)."""
    bets = bet_manager.bets
    resolved = [b for b in bets if b.get("status") in ("won", "lost")]
    pending = [b for b in bets if b.get("status") == "pending"]

    if not resolved:
        return {
            "total": len(bets), "open": len(pending), "resolved": 0,
            "wins": 0, "losses": 0, "winrate": 0,
            "profit": 0, "staked": 0, "roi": 0,
            "by_source": {}, "by_tour": {}, "by_type": {},
            "by_confidence": {}, "by_tournament": {},
            "recent": list(reversed(bets[-30:])),
        }

    wins = [b for b in resolved if b["status"] == "won"]
    losses = [b for b in resolved if b["status"] == "lost"]
    total_staked = sum(b.get("stake", 0) for b in resolved)
    total_profit = sum(b.get("profit", 0) for b in resolved)

    def breakdown(key_fn):
        groups = {}
        for b in resolved:
            k = key_fn(b)
            if k not in groups:
                groups[k] = {"wins": 0, "losses": 0, "profit": 0, "staked": 0}
            stake = b.get("stake", 0)
            groups[k]["staked"] += stake
            if b["status"] == "won":
                groups[k]["wins"] += 1
                groups[k]["profit"] += b.get("profit", 0)
            else:
                groups[k]["losses"] += 1
                groups[k]["profit"] -= stake
        for k, v in groups.items():
            total = v["wins"] + v["losses"]
            v["total"] = total
            v["winrate"] = round(v["wins"] / total * 100, 1) if total > 0 else 0
            v["roi"] = round(v["profit"] / v["staked"] * 100, 1) if v["staked"] > 0 else 0
            v["profit"] = round(v["profit"], 2)
        return dict(sorted(groups.items(), key=lambda x: -x[1]["total"]))

    def tour_tier(b):
        e = (b.get("event", "") or "").lower()
        label = (b.get("market_label", "") or "").lower()
        t1 = (b.get("team1_name", "") or "").lower()
        combined = f"{e} {label}"
        if any(w in combined for w in ["grand slam", "australian", "roland", "wimbledon", "us open"]):
            return "Grand Slam"
        if any(w in combined for w in ["miami", "indian wells", "rome", "madrid", "monte carlo", "shanghai", "paris", "cincinnati", "canadian"]):
            return "Masters 1000"
        if any(w in combined for w in ["challenger", "split", "alicante", "morelia", "bucaramanga", "naples", "yokkaichi", "bucharest", "sao paulo"]):
            return "Challenger"
        if any(w in e for w in ["wta", "women", "dubrovn"]):
            return "WTA Tour"
        return "ATP Tour"

    def bet_type(b):
        label = b.get("market_label", "")
        if "+1.5" in label or "1.5" in label:
            return "+1.5 SETS"
        if "TOTAL" in label:
            return "TOTAL"
        return "WIN"

    def odds_range(b):
        odds = b.get("our_odds", 0)
        if odds < 1.40:
            return "< 1.40"
        elif odds < 1.60:
            return "1.40 - 1.60"
        elif odds < 1.80:
            return "1.60 - 1.80"
        else:
            return "> 1.80"

    # Recent: convert bets to picks-like format for template
    recent = []
    for b in reversed(bets[-30:]):
        recent.append({
            "status": b.get("status", "OPEN").upper(),
            "source": "bets",
            "pick": b.get("team1_name", ""),
            "bet_type": bet_type(b),
            "opponent": b.get("team2_name", ""),
            "tournament": b.get("event", ""),
            "ml_prob": b.get("our_prob", 0),
            "edge": 0,
            "odds": b.get("our_odds", 0),
            "date_added": b.get("created_at", ""),
        })

    def match_type(b):
        mt = b.get("match_type", "singles")
        return "Doubles" if mt == "doubles" else "Singles"

    def match_type_detail(b):
        mt = b.get("match_type", "singles")
        gender = _detect_gender(b)
        if mt == "doubles":
            return f"Doubles ({gender.replace(' (Men)', ' Men').replace(' (Women)', ' Women').replace('ATP', 'Men').replace('WTA', 'Women')})"
        return f"Singles ({gender.replace(' (Men)', ' Men').replace(' (Women)', ' Women').replace('ATP', 'Men').replace('WTA', 'Women')})"

    return {
        "total": len(bets),
        "open": len(pending),
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(len(wins) / len(resolved) * 100, 1),
        "profit": round(total_profit, 2),
        "staked": round(total_staked, 2),
        "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        "by_source": breakdown(lambda b: _detect_gender(b)),
        "by_tour": breakdown(tour_tier),
        "by_type": breakdown(bet_type),
        "by_confidence": breakdown(odds_range),
        "by_match_type": breakdown(match_type),
        "by_match_detail": breakdown(match_type_detail),
        "by_tournament": breakdown(lambda b: (b.get("event", "") or "unknown")[:20]),
        "recent": recent,
    }


# ─── History ──────────────────────────────────────────────

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, page: int = 1, status: str = "", tour: str = "", q: str = "", type: str = ""):
    bet_manager.load()
    bet_stats = bet_manager.get_stats()
    # History = only resolved (won, lost, void) - no pending
    all_bets = sorted(
        [b for b in bet_manager.bets if b.get("status") in ("won", "lost", "void")],
        key=lambda b: b.get("created_at", ""),
        reverse=True,
    )

    # Server-side filters
    if status:
        all_bets = [b for b in all_bets if b.get("status") == status]
    if type:
        all_bets = [b for b in all_bets if b.get("match_type", "singles") == type]
    if tour:
        all_bets = [b for b in all_bets if tour.lower() in (b.get("event", "") or "").lower()]
    if q:
        ql = q.lower()
        all_bets = [b for b in all_bets if ql in (b.get("market_label", "") or "").lower()
                     or ql in (b.get("team1_name", "") or "").lower()
                     or ql in (b.get("team2_name", "") or "").lower()
                     or ql in (b.get("event", "") or "").lower()]

    per_page = 15
    total_pages = max(1, -(-len(all_bets) // per_page))
    page = max(1, min(page, total_pages))
    page_bets = all_bets[(page-1)*per_page : page*per_page]

    return templates.TemplateResponse("history.html", {
        "request": request,
        "active_page": "history",
        "bet_stats": bet_stats,
        "all_bets": page_bets,
        "page": page,
        "total_pages": total_pages,
        "total_bets": len(all_bets),
        "filter_status": status,
        "filter_tour": tour,
        "filter_type": type,
        "filter_q": q,
    })


# ─── Team Profile (CS2 - kept for direct links) ──────────

@app.get("/team/{team_name}", response_class=HTMLResponse)
async def team_profile(request: Request, team_name: str):
    conn = get_db()
    cur = conn.cursor()

    team = cur.execute("SELECT * FROM teams WHERE name = ?", (team_name,)).fetchone()
    if not team:
        conn.close()
        return HTMLResponse("<h1>Team not found</h1>", status_code=404)

    tid = team["hltv_id"]
    _, _, elo_ratings = load_model()
    elo = elo_ratings.get(tid, 1500)

    matches = []
    for r in cur.execute("""
        SELECT m.*, t1.name as t1_name, t2.name as t2_name
        FROM matches m
        JOIN teams t1 ON m.team1_id = t1.hltv_id
        JOIN teams t2 ON m.team2_id = t2.hltv_id
        WHERE (m.team1_id = ? OR m.team2_id = ?) AND m.is_completed = 1
        ORDER BY m.date DESC LIMIT 30
    """, (tid, tid)).fetchall():
        is_t1 = r["team1_id"] == tid
        matches.append({
            "opponent": r["t2_name"] if is_t1 else r["t1_name"],
            "won": r["winner_id"] == tid,
            "our_score": r["team1_score"] if is_t1 else r["team2_score"],
            "opp_score": r["team2_score"] if is_t1 else r["team1_score"],
            "event": r["event_name"] or "", "tier": r["event_tier"] or "",
            "is_lan": r["is_lan"],
        })

    map_stats = []
    for r in cur.execute("""
        SELECT map_name, wins, matches_played, ct_winrate, t_winrate, avg_rounds_won
        FROM team_map_stats WHERE team_id = ? AND period_months = 3
        ORDER BY matches_played DESC
    """, (tid,)).fetchall():
        played = r["matches_played"]
        map_stats.append({
            "map": r["map_name"], "wins": r["wins"], "played": played,
            "winrate": round(r["wins"] / played * 100) if played > 0 else 0,
            "ct_wr": round((r["ct_winrate"] or 0) * 100),
            "t_wr": round((r["t_winrate"] or 0) * 100),
            "avg_rounds": round(r["avg_rounds_won"] or 0, 1),
        })

    total = len(matches)
    wins = sum(1 for m in matches if m["won"])
    winrate = round(wins / total * 100) if total > 0 else 0
    form5 = sum(1 for m in matches[:5] if m["won"]) / min(len(matches), 5) * 100 if matches else 0
    lan_matches = [m for m in matches if m["is_lan"]]
    lan_wr = round(sum(1 for m in lan_matches if m["won"]) / len(lan_matches) * 100) if lan_matches else 0
    online_matches = [m for m in matches if not m["is_lan"]]
    online_wr = round(sum(1 for m in online_matches if m["won"]) / len(online_matches) * 100) if online_matches else 0

    streak = 0
    if matches:
        for m in matches:
            if m["won"]:
                streak += 1
            else:
                break
        if not matches[0]["won"]:
            streak = -streak

    roster = []
    try:
        with open(DATA_DIR / "rosters.json", "r") as f:
            rosters = json.load(f)
        roster = rosters.get(team_name, {}).get("players", [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    conn.close()

    profile = {
        "name": team_name, "ranking": team["ranking"], "elo": round(elo),
        "winrate": winrate, "form5": round(form5), "streak": streak,
        "total_matches": total, "lan_wr": lan_wr, "online_wr": online_wr,
        "lan_matches": len(lan_matches), "online_matches": len(online_matches),
        "roster": roster, "map_stats": map_stats, "matches": matches[:20],
    }

    return templates.TemplateResponse("team.html", {
        "request": request,
        "active_page": "",
        "bet_stats": bet_manager.get_stats(),
        "team": profile,
    })


# ─── Player Profile (Tennis) ──────────────────────────────

def _slug_to_name(slug: str) -> str:
    """Convert URL slug to searchable name: 'mirra-andreeva' -> 'Mirra Andreeva'."""
    return " ".join(w.capitalize() for w in slug.replace("-", " ").split())


def _name_to_slug(name: str) -> str:
    """Convert DB name to URL slug: 'Andreeva M.' -> 'andreeva-m'."""
    return name.lower().replace(" ", "-").replace(".", "")


def _detect_tour(player_name: str) -> str:
    """Detect ATP/WTA from match data."""
    from src.tennis.database import get_tennis_db
    conn = get_tennis_db()
    row = conn.execute("""
        SELECT tour FROM tennis_matches
        WHERE winner_id = (SELECT id FROM tennis_players WHERE name = ?)
           OR loser_id = (SELECT id FROM tennis_players WHERE name = ?)
        ORDER BY date DESC LIMIT 1
    """, (player_name, player_name)).fetchone()
    conn.close()
    return (row["tour"] or "atp").lower() if row else "atp"


@app.get("/{tour}/player/{slug}", response_class=HTMLResponse)
async def player_profile(request: Request, tour: str, slug: str):
    from src.tennis.stats import get_player_profile

    search_name = _slug_to_name(slug)
    profile = get_player_profile(search_name)
    if not profile:
        return HTMLResponse("<h1>Player not found</h1>", status_code=404)

    return templates.TemplateResponse("player.html", {
        "request": request,
        "active_page": "",
        "bet_stats": bet_manager.get_stats(),
        "p": profile,
        "tour": tour,
    })


# Legacy redirect: /player/Name -> /atp/player/slug
@app.get("/player/{player_name:path}", response_class=HTMLResponse)
async def player_redirect(request: Request, player_name: str):
    from src.tennis.stats import get_player_profile
    from fastapi.responses import RedirectResponse

    profile = get_player_profile(player_name)
    if not profile:
        return HTMLResponse("<h1>Player not found</h1>", status_code=404)

    tour = _detect_tour(profile["name"])
    slug = _name_to_slug(profile["name"])
    return RedirectResponse(url=f"/{tour}/player/{slug}", status_code=302)


@app.get("/api/player/search")
async def player_search(q: str = ""):
    """Search players by name (for autocomplete)."""
    if len(q) < 2:
        return JSONResponse([])
    from src.tennis.database import get_tennis_db
    conn = get_tennis_db()
    rows = conn.execute(
        "SELECT name FROM tennis_players WHERE name LIKE ? ORDER BY name LIMIT 20",
        (f"%{q}%",)
    ).fetchall()
    conn.close()
    return JSONResponse([r["name"] for r in rows])


# ─── API: Place Bet ───────────────────────────────────────

@app.post("/api/bet/place")
async def api_place_bet(request: Request):
    bet_manager.load()
    data = await request.json()

    match_id = data.get("match_id", 0)
    market = data.get("market", "")
    market_label = data.get("market_label", "")
    our_prob = data.get("our_prob", 0.5)
    our_odds = data.get("our_odds", 2.0)
    stake = data.get("stake", 50)
    event = data.get("event", "")
    team1_name = data.get("team1_name", "")
    team2_name = data.get("team2_name", "")

    if not market_label or not team1_name:
        return JSONResponse({"ok": False, "error": "Missing bet data"})

    if stake > bet_manager.bankroll:
        return JSONResponse({"ok": False, "error": f"Insufficient bankroll (${bet_manager.bankroll:.2f})"})

    from src.strategy.bet_manager import BetRecommendation
    rec = BetRecommendation(
        match_id=match_id,
        team1_name=team1_name,
        team2_name=team2_name,
        event=event,
        tier="",
        market=market,
        market_label=market_label,
        our_prob=our_prob,
        our_odds=our_odds,
        edge=data.get("edge", 0),
        confidence=data.get("confidence", 0),
        kelly_stake_pct=0,
        stake_amount=stake,
        rating=0,
        reasons=[],
        market_odds=our_odds,
    )

    # Deduplicate - don't place same bet twice
    existing = any(
        b.get("market_label") == market_label and b.get("team1_name") == team1_name
        and b.get("status") == "pending"
        for b in bet_manager.bets
    )
    if existing:
        return JSONResponse({"ok": False, "error": "Bet already exists"})

    bet_manager.record_bet(rec)
    bet_manager.bankroll -= stake
    bet_manager.save()

    return JSONResponse({
        "ok": True,
        "message": f"Placed ${stake} on {market_label}",
    })


# ─── API: Resolve Bet ─────────────────────────────────────

@app.post("/api/bet/resolve")
async def api_resolve_bet(request: Request):
    bet_manager.load()  # Always read fresh from disk
    data = await request.json()
    bet_id = data.get("bet_id", "")
    result = data.get("result", "")

    if not bet_id or result not in ("won", "lost", "void", "pending"):
        return JSONResponse({"ok": False, "error": "Invalid bet_id or result"})

    bet_manager.resolve_bet(bet_id, result)

    return JSONResponse({
        "ok": True,
        "message": f"Bet resolved as {result.upper()}",
    })


@app.post("/api/bet/delete")
async def api_delete_bet(request: Request):
    bet_manager.load()
    data = await request.json()
    bet_id = data.get("bet_id", "")
    if not bet_id:
        return JSONResponse({"ok": False, "error": "Missing bet_id"})

    # Find bet and refund stake
    bet = next((b for b in bet_manager.bets if b.get("id") == bet_id), None)
    if not bet:
        return JSONResponse({"ok": False, "error": "Bet not found"})

    if bet.get("status") == "pending":
        bet_manager.bankroll += bet.get("stake", 0)

    bet_manager.bets = [b for b in bet_manager.bets if b.get("id") != bet_id]
    bet_manager.save()

    return JSONResponse({"ok": True, "message": "Position removed"})


# ─── API: Notifications ──────────────────────────────────

@app.get("/api/notifications")
async def get_notifications():
    try:
        with open(DATA_DIR / "alerts.json", "r") as f:
            alerts = json.load(f)
        return JSONResponse(alerts[-30:])
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse([])


# ─── API: Refresh Tennis Odds ─────────────────────────────

@app.post("/api/refresh-tennis")
async def refresh_tennis():
    try:
        pm = PolymarketClient()
        markets = await pm.fetch_tennis_markets()
        await pm.close()

        from src.tennis.sofascore_api import fetch_upcoming_with_odds
        ss_matches = await fetch_upcoming_with_odds(days=2)

        return JSONResponse({
            "ok": True,
            "message": f"Refreshed {len(markets)} PM + {len(ss_matches)} SofaScore markets",
            "sofascore_usage": ss_usage(),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/api/positions")
async def api_positions():
    bets = bet_manager.get_recent_bets(20)
    stats = bet_manager.get_stats()
    return JSONResponse({
        "bets": bets,
        "bankroll": stats.get("bankroll", 1000),
        "total_profit": stats.get("total_profit", 0),
        "pending": stats.get("pending", 0),
    })


@app.post("/api/refresh-cache")
async def api_refresh_cache():
    import threading
    threading.Thread(target=_recompute_cache, daemon=True).start()
    return JSONResponse({"ok": True, "message": "Cache refresh started"})


@app.post("/api/resolve-from-polymarket")
async def api_resolve_from_pm():
    """Resolve pending bets by checking current Polymarket prices.
    If a market price is >= 0.95, that outcome won."""
    import httpx, json as jjson

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get("https://gamma-api.polymarket.com/markets", params={
                "tag_id": "864",
                "sports_market_types": "moneyline",
                "active": "true",
                "closed": "false",
                "limit": 200,
            })
            if r.status_code != 200:
                return JSONResponse({"ok": False, "error": f"PM API {r.status_code}"})

            markets = r.json()

        resolved = 0
        for mk in markets:
            outcomes = jjson.loads(mk.get("outcomes", "[]")) if isinstance(mk.get("outcomes"), str) else mk.get("outcomes", [])
            prices = jjson.loads(mk.get("outcomePrices", "[]")) if isinstance(mk.get("outcomePrices"), str) else mk.get("outcomePrices", [])
            if len(outcomes) < 2 or len(prices) < 2:
                continue

            p1_price, p2_price = float(prices[0]), float(prices[1])
            # Match is settled when one price is ~1.0
            if p1_price < 0.95 and p2_price < 0.95:
                continue

            winner_name = outcomes[0] if p1_price >= 0.95 else outcomes[1]
            loser_name = outcomes[1] if p1_price >= 0.95 else outcomes[0]

            # Find matching pending bets
            for bet in bet_manager.bets:
                if bet["status"] != "pending":
                    continue

                label = bet.get("market_label", "")
                t1 = bet.get("team1_name", "")
                t2 = bet.get("team2_name", "")
                pick_name = label.replace(" WIN", "").replace(" +1.5 SETS", "").replace(" +1.5 Sets", "").replace(" TOTAL", "").strip()

                # Match by name
                wn = _normalize_name(winner_name)
                ln = _normalize_name(loser_name)
                t1n = _normalize_name(t1)
                t2n = _normalize_name(t2)
                pn = _normalize_name(pick_name)

                if not ((wn[:5] in t1n or wn[:5] in t2n) and (ln[:5] in t1n or ln[:5] in t2n)):
                    continue

                if "WIN" in label and "1.5" not in label:
                    won = wn[:5] in pn or pn[:5] in wn
                elif "1.5" in label:
                    # +1.5 sets = always won unless 2-0 loss
                    # PM doesn't tell us set score, so assume won (conservative)
                    won = True
                else:
                    continue

                result = "won" if won else "lost"
                bet_manager.resolve_bet(bet["id"], result)
                resolved += 1

        # Also resolve picks tracker
        from src.tennis.picks_tracker import load_picks, save_picks
        picks = load_picks()
        for pk in picks:
            if pk["status"] != "OPEN":
                continue
            for mk in markets:
                outcomes = jjson.loads(mk.get("outcomes", "[]")) if isinstance(mk.get("outcomes"), str) else mk.get("outcomes", [])
                prices = jjson.loads(mk.get("outcomePrices", "[]")) if isinstance(mk.get("outcomePrices"), str) else mk.get("outcomePrices", [])
                if len(prices) < 2:
                    continue
                p1_price, p2_price = float(prices[0]), float(prices[1])
                if p1_price < 0.95 and p2_price < 0.95:
                    continue

                winner = outcomes[0] if p1_price >= 0.95 else outcomes[1]
                wn = _normalize_name(winner)
                pn = _normalize_name(pk.get("pick", ""))
                on = _normalize_name(pk.get("opponent", ""))

                if not ((wn[:5] in pn or wn[:5] in on) and (pn[:5] in wn or on[:5] in wn or True)):
                    continue

                if pk.get("bet_type") == "WIN":
                    pk["status"] = "WIN" if wn[:5] in pn or pn[:5] in wn else "LOSS"
                elif "1.5" in pk.get("bet_type", ""):
                    pk["status"] = "WIN"  # conservative for +1.5
                break
        save_picks(picks)

        return JSONResponse({"ok": True, "message": f"Resolved {resolved} bets from Polymarket"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.post("/api/resolve-single")
async def api_resolve_single(request: Request):
    """Check SofaScore for a single bet's match result."""
    data = await request.json()
    bet_id = data.get("bet_id", "")
    if not bet_id:
        return JSONResponse({"ok": False, "error": "Missing bet_id"})

    bet_manager.load()
    bet = next((b for b in bet_manager.bets if b.get("id") == bet_id), None)
    if not bet or bet["status"] != "pending":
        return JSONResponse({"ok": False, "error": "Bet not found or not pending"})

    try:
        from src.tennis.auto_resolve import resolve_from_sofascore, _names_match, _norm
        import httpx

        t1 = bet.get("team1_name", "")
        t2 = bet.get("team2_name", "")
        label = bet.get("market_label", "")
        pick_name = label.replace(" WIN", "").replace(" +1.5 SETS", "").replace(" +1.5 Sets", "").replace(" TOTAL", "").strip()

        # Search last 5 days
        from src.tennis.sofascore_api import HEADERS as SS_HEADERS, BASE_URL as SS_BASE
        from src.tennis.sofascore_api import _load_usage, _save_usage

        async with httpx.AsyncClient(timeout=120) as client:
            from datetime import datetime, timedelta
            for d in range(5):
                date = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
                resp = await client.get(f"{SS_BASE}/match/list?sport_slug=tennis&date={date}", headers=SS_HEADERS, timeout=120)
                usage = _load_usage()
                usage["count"] = usage.get("count", 0) + 1
                _save_usage(usage)

                if resp.status_code != 200:
                    continue

                events = resp.json() if isinstance(resp.json(), list) else resp.json().get("events", [])
                for ev in events:
                    status = ev.get("status", {})
                    if not status.get("isFinished"):
                        continue

                    home = ev.get("homeTeam", {}).get("name", "")
                    away = ev.get("awayTeam", {}).get("name", "")

                    if not (_names_match(home, t1) or _names_match(away, t1)):
                        continue
                    if not (_names_match(home, t2) or _names_match(away, t2)):
                        continue

                    hs = ev.get("homeScore", {}) or {}
                    aws = ev.get("awayScore", {}) or {}
                    h_sets = hs.get("current", 0) or 0
                    a_sets = aws.get("current", 0) or 0
                    if h_sets == a_sets:
                        continue

                    winner = home if h_sets > a_sets else away
                    loser = away if h_sets > a_sets else home
                    w_sets = max(h_sets, a_sets)
                    l_sets = min(h_sets, a_sets)

                    score_parts = []
                    for i in range(1, 6):
                        hp, ap = hs.get(f"period{i}"), aws.get(f"period{i}")
                        if hp is not None and ap is not None:
                            score_parts.append(f"{hp}-{ap}" if h_sets > a_sets else f"{ap}-{hp}")

                    pick_won = _names_match(winner, pick_name)
                    if "WIN" in label and "1.5" not in label:
                        won = pick_won
                    elif "1.5" in label:
                        won = pick_won or l_sets >= 1
                    else:
                        continue

                    result = "won" if won else "lost"
                    bet_manager.load()
                    bet_manager.resolve_bet(bet_id, result)
                    score_str = f"{winner} {w_sets}-{l_sets} ({' '.join(score_parts)})"
                    for b in bet_manager.bets:
                        if b["id"] == bet_id:
                            b["actual_result"] = score_str
                            slug = ev.get("slug", "")
                            eid = ev.get("id", 0)
                            if slug and eid:
                                b["pm_url"] = f"https://www.sofascore.com/tennis/match/{slug}#id:{eid}"
                    bet_manager.save()

                    return JSONResponse({"ok": True, "message": f"{result.upper()}: {score_str}"})

        return JSONResponse({"ok": False, "error": "Match not found yet (may not have finished)"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.post("/api/resolve-all")
async def api_resolve_all():
    """Manually trigger auto-resolve from SofaScore for all pending bets."""
    try:
        from src.tennis.auto_resolve import resolve_from_sofascore
        resolved = await resolve_from_sofascore(days=5)
        bet_manager.load()
        stats = bet_manager.get_stats()
        return JSONResponse({
            "ok": True,
            "message": f"Resolved {resolved} bets",
            "pending": stats.get("pending", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/api/sofascore-usage")
async def sofascore_usage():
    return JSONResponse(ss_usage())


# ─── Live Tracker (WebSocket + REST) ────────────────────────

@app.get("/api/live")
async def api_live():
    """Get live matches with scores + odds. Polls SofaScore + Polymarket."""
    try:
        pending = [b for b in bet_manager.bets if b.get("status") == "pending"]
        state = await get_live_state(pending_bets=pending)
        return JSONResponse(state)
    except Exception as e:
        return JSONResponse({"matches": [], "error": str(e)})


from starlette.websockets import WebSocket, WebSocketDisconnect

_ws_clients: list[WebSocket] = []


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """WebSocket endpoint for live match updates. Pushes every 30s."""
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            pending = [b for b in bet_manager.bets if b.get("status") == "pending"]
            state = await get_live_state(pending_bets=pending)
            await ws.send_json(state)
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


import asyncio
