"""Advanced feature engineering: fatigue, veto patterns, multi-window streaks, map pool analysis."""

import logging
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import text

from src.database import async_session
from src.model.map_elo import MapEloSystem

logger = logging.getLogger(__name__)


async def compute_fatigue(team_id: int, match_date: datetime, session) -> dict:
    """How many matches has the team played recently? More = fatigue."""
    result = await session.execute(text("""
        SELECT COUNT(*) FROM matches
        WHERE is_completed = 1 AND (team1_id = :tid OR team2_id = :tid)
        AND date >= :cutoff7
    """), {"tid": team_id, "cutoff7": match_date - timedelta(days=7)})
    matches_7d = result.scalar() or 0

    result2 = await session.execute(text("""
        SELECT COUNT(*) FROM matches
        WHERE is_completed = 1 AND (team1_id = :tid OR team2_id = :tid)
        AND date >= :cutoff14
    """), {"tid": team_id, "cutoff14": match_date - timedelta(days=14)})
    matches_14d = result2.scalar() or 0

    result3 = await session.execute(text("""
        SELECT COUNT(*) FROM matches
        WHERE is_completed = 1 AND (team1_id = :tid OR team2_id = :tid)
        AND date >= :cutoff3
    """), {"tid": team_id, "cutoff3": match_date - timedelta(days=3)})
    matches_3d = result3.scalar() or 0

    return {
        "matches_3d": matches_3d,
        "matches_7d": matches_7d,
        "matches_14d": matches_14d,
        "fatigue_score": min(matches_3d / 5, 1.0),  # 5+ matches in 3 days = max fatigue
    }


async def compute_veto_profile(team_id: int, session) -> dict:
    """Approximate map pick/ban preferences from map play frequency."""
    result = await session.execute(text("""
        SELECT mr.map_name, COUNT(*) as cnt,
               SUM(CASE WHEN mr.winner_id = :tid THEN 1 ELSE 0 END) as wins
        FROM map_results mr
        JOIN matches m ON mr.match_id = m.hltv_id
        WHERE (m.team1_id = :tid OR m.team2_id = :tid)
        AND m.is_completed = 1 AND mr.map_name != 'unknown'
        GROUP BY mr.map_name
        ORDER BY cnt DESC
    """), {"tid": team_id})
    rows = result.fetchall()

    if not rows:
        return {"map_pool_size": 0, "most_played": "", "least_played": "",
                "best_map_wr": 0.5, "worst_map_wr": 0.5, "map_diversity": 0}

    total_maps = sum(r[1] for r in rows)
    map_data = []
    for map_name, cnt, wins in rows:
        wr = wins / cnt if cnt > 0 else 0.5
        map_data.append({"map": map_name, "played": cnt, "wins": wins, "wr": wr})

    best = max(map_data, key=lambda m: m["wr"]) if map_data else {"map": "", "wr": 0.5}
    worst = min(map_data, key=lambda m: m["wr"]) if map_data else {"map": "", "wr": 0.5}

    return {
        "map_pool_size": len([m for m in map_data if m["played"] >= 3]),
        "most_played": map_data[0]["map"] if map_data else "",
        "least_played": map_data[-1]["map"] if map_data else "",
        "best_map_wr": best["wr"],
        "worst_map_wr": worst["wr"],
        "map_diversity": len(map_data) / 9,  # 9 maps in pool
        "total_map_matches": total_maps,
    }


def compute_multi_window_streaks(history: list) -> dict:
    """Compute win streaks across multiple windows (3, 5, 10, 20 matches)."""
    def winrate(h, n):
        recent = h[-n:] if len(h) >= n else h
        if not recent:
            return 0.5
        return sum(r["won"] for r in recent) / len(recent)

    def current_streak(h):
        if not h:
            return 0
        s = 0
        last = h[-1]["won"]
        for r in reversed(h):
            if r["won"] == last:
                s += 1
            else:
                break
        return s if last else -s

    def max_streak(h, n):
        """Max win streak in last n matches."""
        recent = h[-n:] if len(h) >= n else h
        max_s = 0
        curr = 0
        for r in recent:
            if r["won"]:
                curr += 1
                max_s = max(max_s, curr)
            else:
                curr = 0
        return max_s

    return {
        "form_3": winrate(history, 3),
        "form_5": winrate(history, 5),
        "form_10": winrate(history, 10),
        "form_20": winrate(history, 20),
        "streak": current_streak(history),
        "max_streak_10": max_streak(history, 10),
        "momentum": winrate(history, 3) - winrate(history, 10),  # short vs long form
    }


def compute_opponent_adjusted_stats(history: list) -> dict:
    """Compute stats weighted by opponent strength."""
    if not history:
        return {"adj_winrate": 0.5, "strong_opp_wr": 0.5, "weak_opp_wr": 0.5, "upset_potential": 0.5}

    # Split by opponent Elo
    strong = [h for h in history if h.get("opponent_elo", 1500) >= 1550]
    weak = [h for h in history if h.get("opponent_elo", 1500) < 1450]

    strong_wr = sum(h["won"] for h in strong) / len(strong) if strong else 0.5
    weak_wr = sum(h["won"] for h in weak) / len(weak) if weak else 0.5

    # Weighted winrate
    total_weight = 0
    total_score = 0
    for h in history[-20:]:
        w = h.get("opponent_elo", 1500) / 1500
        total_weight += w
        total_score += w * h["won"]

    return {
        "adj_winrate": total_score / total_weight if total_weight > 0 else 0.5,
        "strong_opp_wr": strong_wr,
        "weak_opp_wr": weak_wr,
        "upset_potential": strong_wr,  # how often they beat stronger teams
    }


def compute_score_patterns(history: list) -> dict:
    """Analyze scoring patterns - sweeps, close matches, comebacks."""
    if not history:
        return {"sweep_rate": 0, "close_match_rate": 0, "avg_map_diff": 0}

    sweeps = 0
    close = 0
    total_diff = 0
    counted = 0

    for h in history[-20:]:
        mw = h.get("maps_won", 0)
        ml = h.get("maps_lost", 0)
        if mw + ml == 0:
            continue
        counted += 1
        diff = mw - ml
        total_diff += diff

        if h["won"] and ml == 0:
            sweeps += 1
        if abs(diff) <= 1 and mw + ml >= 2:
            close += 1

    wins = sum(1 for h in history[-20:] if h["won"])
    return {
        "sweep_rate": sweeps / wins if wins > 0 else 0,
        "close_match_rate": close / counted if counted > 0 else 0,
        "avg_map_diff": total_diff / counted if counted > 0 else 0,
    }


def get_map_elo_features(map_elo: MapEloSystem, t1_id: int, t2_id: int) -> dict:
    """Extract features from per-map Elo system."""
    t1_best_map, t1_best_elo = map_elo.get_best_map(t1_id)
    t1_worst_map, t1_worst_elo = map_elo.get_worst_map(t1_id)
    t2_best_map, t2_best_elo = map_elo.get_best_map(t2_id)
    t2_worst_map, t2_worst_elo = map_elo.get_worst_map(t2_id)

    t1_depth = map_elo.get_map_pool_depth(t1_id)
    t2_depth = map_elo.get_map_pool_depth(t2_id)

    # Average map Elo across all maps
    t1_profile = map_elo.get_team_map_profile(t1_id)
    t2_profile = map_elo.get_team_map_profile(t2_id)

    t1_avg = sum(m["elo"] for m in t1_profile.values()) / len(t1_profile) if t1_profile else 1500
    t2_avg = sum(m["elo"] for m in t2_profile.values()) / len(t2_profile) if t2_profile else 1500

    # Elo variance (consistency across maps)
    t1_elos = [m["elo"] for m in t1_profile.values()]
    t2_elos = [m["elo"] for m in t2_profile.values()]

    def variance(vals):
        if len(vals) < 2:
            return 0
        mean = sum(vals) / len(vals)
        return sum((v - mean) ** 2 for v in vals) / len(vals)

    return {
        "t1_best_map_elo": t1_best_elo if t1_best_elo < 9999 else 1500,
        "t1_worst_map_elo": t1_worst_elo if t1_worst_elo < 9999 else 1500,
        "t2_best_map_elo": t2_best_elo if t2_best_elo < 9999 else 1500,
        "t2_worst_map_elo": t2_worst_elo if t2_worst_elo < 9999 else 1500,
        "t1_map_pool_depth": t1_depth,
        "t2_map_pool_depth": t2_depth,
        "map_pool_diff": t1_depth - t2_depth,
        "t1_avg_map_elo": round(t1_avg),
        "t2_avg_map_elo": round(t2_avg),
        "avg_map_elo_diff": round(t1_avg - t2_avg),
        "t1_map_variance": round(variance(t1_elos), 1),
        "t2_map_variance": round(variance(t2_elos), 1),
        "t1_map_range": round((t1_best_elo - t1_worst_elo) if t1_best_elo < 9999 and t1_worst_elo < 9999 else 0),
        "t2_map_range": round((t2_best_elo - t2_worst_elo) if t2_best_elo < 9999 and t2_worst_elo < 9999 else 0),
    }
