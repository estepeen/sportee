"""Tennis feature engineering for prediction model."""

import logging
from datetime import datetime, timedelta
from src.tennis.database import get_tennis_db
from src.tennis.elo import expected_score

logger = logging.getLogger(__name__)

SURFACE_MAP = {"Hard": "hard", "Clay": "clay", "Grass": "grass", "Carpet": "hard"}

# Round importance weights (later rounds = more reliable signal)
ROUND_WEIGHTS = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
    "Round of 128": 1, "Round of 64": 2, "Round of 32": 3, "Round of 16": 4,
    "Quarterfinal": 5, "Quarter-final": 5, "Quarterfinals": 5,
    "Semifinal": 6, "Semi-final": 6, "Semifinals": 6,
    "Final": 7, "The Final": 7,
}

# Tournament tier weights
TIER_WEIGHTS = {
    "Grand Slam": 5, "Masters 1000": 4, "WTA 1000": 4,
    "ATP 500": 3, "WTA 500": 3,
    "ATP 250": 2, "WTA 250": 2,
}

# Continent mapping for timezone jump detection
TOURNAMENT_CONTINENTS = {
    "australian open": "oceania", "melbourne": "oceania", "brisbane": "oceania", "adelaide": "oceania",
    "indian wells": "americas", "miami": "americas", "us open": "americas",
    "cincinnati": "americas", "atlanta": "americas", "washington": "americas",
    "montreal": "americas", "toronto": "americas", "acapulco": "americas",
    "buenos aires": "americas", "rio": "americas", "santiago": "americas",
    "roland garros": "europe", "wimbledon": "europe", "rome": "europe",
    "madrid": "europe", "monte carlo": "europe", "barcelona": "europe",
    "paris": "europe", "halle": "europe", "hamburg": "europe",
    "lyon": "europe", "marseille": "europe", "rotterdam": "europe",
    "vienna": "europe", "basel": "europe", "stuttgart": "europe",
    "s-hertogenbosch": "europe", "eastbourne": "europe",
    "shanghai": "asia", "beijing": "asia", "tokyo": "asia",
    "doha": "asia", "dubai": "asia",
}


def build_match_features(player1_id: int, player2_id: int, surface: str,
                         indoor: int = 0, series: str = "", best_of: int = 3,
                         tournament: str = "", round_name: str = "") -> dict:
    """Build feature vector for a match prediction.

    Returns dict of features for the model. Positive values favor player1.
    """
    conn = get_tennis_db()
    surface_key = SURFACE_MAP.get(surface, "hard")

    # ── Elo ratings ──
    def get_elo(pid, elo_type):
        row = conn.execute(
            "SELECT elo, matches FROM tennis_elo WHERE player_id = ? AND elo_type = ?",
            (pid, elo_type)
        ).fetchone()
        return (row["elo"], row["matches"]) if row else (1500, 0)

    p1_global_elo, p1_global_n = get_elo(player1_id, "global")
    p2_global_elo, p2_global_n = get_elo(player2_id, "global")
    p1_surface_elo, p1_surface_n = get_elo(player1_id, surface_key)
    p2_surface_elo, p2_surface_n = get_elo(player2_id, surface_key)

    # ── Rankings ──
    def latest_rank(pid):
        row = conn.execute("""
            SELECT COALESCE(
                (SELECT winner_rank FROM tennis_matches WHERE winner_id = ? AND winner_rank > 0 ORDER BY date DESC LIMIT 1),
                (SELECT loser_rank FROM tennis_matches WHERE loser_id = ? AND loser_rank > 0 ORDER BY date DESC LIMIT 1),
                500
            ) as rank
        """, (pid, pid)).fetchone()
        return row["rank"] if row else 500

    p1_rank = latest_rank(player1_id)
    p2_rank = latest_rank(player2_id)

    # ── Player attributes ──
    def get_player(pid):
        return conn.execute(
            "SELECT hand, style FROM tennis_players WHERE id = ?", (pid,)
        ).fetchone()

    p1 = get_player(player1_id)
    p2 = get_player(player2_id)
    p1_hand = p1["hand"] if p1 else "R"
    p2_hand = p2["hand"] if p2 else "R"
    p1_style = p1["style"] if p1 else "universal"
    p2_style = p2["style"] if p2 else "universal"

    # ── Form (recent matches) ──
    def get_form(pid, n, surface_filter=None):
        if surface_filter:
            rows = conn.execute("""
                SELECT winner_id FROM tennis_matches
                WHERE (winner_id = ? OR loser_id = ?) AND surface = ?
                AND comment NOT LIKE '%Walkover%'
                ORDER BY date DESC LIMIT ?
            """, (pid, pid, surface_filter, n)).fetchall()
        else:
            rows = conn.execute("""
                SELECT winner_id FROM tennis_matches
                WHERE (winner_id = ? OR loser_id = ?)
                AND comment NOT LIKE '%Walkover%'
                ORDER BY date DESC LIMIT ?
            """, (pid, pid, n)).fetchall()
        if not rows:
            return 0.5, 0
        wins = sum(1 for r in rows if r["winner_id"] == pid)
        return wins / len(rows), len(rows)

    p1_form5, _ = get_form(player1_id, 5)
    p2_form5, _ = get_form(player2_id, 5)
    p1_form10, _ = get_form(player1_id, 10)
    p2_form10, _ = get_form(player2_id, 10)
    p1_form20, _ = get_form(player1_id, 20)
    p2_form20, _ = get_form(player2_id, 20)

    # Surface-specific form
    p1_surface_form, p1_surface_matches = get_form(player1_id, 15, surface)
    p2_surface_form, p2_surface_matches = get_form(player2_id, 15, surface)

    # ── H2H ──
    h2h_rows = conn.execute("""
        SELECT winner_id, surface FROM tennis_matches
        WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        ORDER BY date DESC
    """, (player1_id, player2_id, player2_id, player1_id)).fetchall()

    h2h_total = len(h2h_rows)
    h2h_p1_wins = sum(1 for r in h2h_rows if r["winner_id"] == player1_id)
    h2h_wr = h2h_p1_wins / h2h_total if h2h_total > 0 else 0.5

    # H2H on this surface
    h2h_surface = [r for r in h2h_rows if r["surface"] == surface]
    h2h_surface_total = len(h2h_surface)
    h2h_surface_p1 = sum(1 for r in h2h_surface if r["winner_id"] == player1_id)
    h2h_surface_wr = h2h_surface_p1 / h2h_surface_total if h2h_surface_total > 0 else 0.5

    # ── Tournament history ──
    def tournament_wr(pid, tourn, n=20):
        rows = conn.execute("""
            SELECT winner_id FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?) AND tournament = ?
            ORDER BY date DESC LIMIT ?
        """, (pid, pid, tourn, n)).fetchall()
        if not rows:
            return 0.5, 0
        wins = sum(1 for r in rows if r["winner_id"] == pid)
        return wins / len(rows), len(rows)

    p1_tourn_wr, p1_tourn_n = tournament_wr(player1_id, tournament)
    p2_tourn_wr, p2_tourn_n = tournament_wr(player2_id, tournament)

    # ── Regional performance ──
    def region_from_tournament(tourn, loc):
        t = (tourn + " " + loc).lower()
        us = ["us open", "miami", "indian wells", "cincinnati", "atlanta", "washington", "new york"]
        europe = ["roland garros", "wimbledon", "rome", "madrid", "monte carlo", "barcelona",
                   "paris", "halle", "queen", "hamburg", "lyon", "marseille", "rotterdam"]
        asia = ["shanghai", "beijing", "tokyo", "doha", "dubai", "australian open", "melbourne"]
        south_am = ["rio", "buenos aires", "santiago", "cordoba"]
        for kw in us:
            if kw in t: return "americas"
        for kw in south_am:
            if kw in t: return "americas"
        for kw in europe:
            if kw in t: return "europe"
        for kw in asia:
            if kw in t: return "asia_pacific"
        return "other"

    region = region_from_tournament(tournament, "")

    def region_form(pid, reg, n=20):
        # Get tournaments in this region and check performance
        all_matches = conn.execute("""
            SELECT winner_id, tournament, location FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?)
            ORDER BY date DESC LIMIT 100
        """, (pid, pid)).fetchall()
        region_matches = [
            m for m in all_matches
            if region_from_tournament(m["tournament"], m["location"] or "") == reg
        ][:n]
        if not region_matches:
            return 0.5, 0
        wins = sum(1 for m in region_matches if m["winner_id"] == pid)
        return wins / len(region_matches), len(region_matches)

    p1_region_wr, p1_region_n = region_form(player1_id, region)
    p2_region_wr, p2_region_n = region_form(player2_id, region)

    # ── Style matchup features ──
    # Serve-based players struggle on slow surfaces
    p1_serve_penalty = 1 if p1_style == "serve_based" and surface_key == "clay" else 0
    p2_serve_penalty = 1 if p2_style == "serve_based" and surface_key == "clay" else 0

    # Lefty advantage (some players have trouble against lefties)
    p1_is_lefty = 1 if p1_hand == "L" else 0
    p2_is_lefty = 1 if p2_hand == "L" else 0

    # Style encoding
    style_map = {"aggressive": 3, "universal": 2, "defensive": 1, "serve_based": 4, "unknown": 2}
    p1_style_num = style_map.get(p1_style, 2)
    p2_style_num = style_map.get(p2_style, 2)

    # ── Streak ──
    def get_streak(pid, n=10):
        rows = conn.execute("""
            SELECT winner_id FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?)
            AND comment NOT LIKE '%Walkover%'
            ORDER BY date DESC LIMIT ?
        """, (pid, pid, n)).fetchall()
        if not rows:
            return 0
        streak = 0
        first_won = rows[0]["winner_id"] == pid
        for r in rows:
            if (r["winner_id"] == pid) == first_won:
                streak += 1
            else:
                break
        return streak if first_won else -streak

    p1_streak = get_streak(player1_id)
    p2_streak = get_streak(player2_id)

    # ── Flags (injuries, coach changes) ──
    def active_flags(pid):
        rows = conn.execute("""
            SELECT flag_type FROM tennis_player_flags
            WHERE player_id = ? AND active = 1
            AND (date_expires = '' OR date_expires > date('now'))
        """, (pid,)).fetchall()
        return [r["flag_type"] for r in rows]

    p1_flags = active_flags(player1_id)
    p2_flags = active_flags(player2_id)
    p1_injury = 1 if "injury" in p1_flags or "returning" in p1_flags else 0
    p2_injury = 1 if "injury" in p2_flags or "returning" in p2_flags else 0
    p1_coach_change = 1 if "coach_change" in p1_flags else 0
    p2_coach_change = 1 if "coach_change" in p2_flags else 0

    # ── Fatigue / Scheduling features ──
    p1_fatigue = _compute_fatigue(conn, player1_id, tournament)
    p2_fatigue = _compute_fatigue(conn, player2_id, tournament)

    # ── Extended Serve Stats ──
    p1_serve = _compute_serve_features(conn, player1_id)
    p2_serve = _compute_serve_features(conn, player2_id)

    # ── Round & Tournament Tier weights ──
    round_importance = ROUND_WEIGHTS.get(round_name, 2) if round_name else 2
    tier_weight = TIER_WEIGHTS.get(series, 2)

    conn.close()

    # ── Build feature dict ──
    return {
        # Elo
        "p1_global_elo": p1_global_elo,
        "p2_global_elo": p2_global_elo,
        "elo_diff": p1_global_elo - p2_global_elo,
        "elo_expected": expected_score(p1_global_elo, p2_global_elo),
        "p1_surface_elo": p1_surface_elo,
        "p2_surface_elo": p2_surface_elo,
        "surface_elo_diff": p1_surface_elo - p2_surface_elo,
        "surface_elo_expected": expected_score(p1_surface_elo, p2_surface_elo),

        # Rankings
        "p1_rank": min(p1_rank, 500),
        "p2_rank": min(p2_rank, 500),
        "rank_diff": p2_rank - p1_rank,  # positive = p1 ranked higher

        # Form
        "p1_form5": p1_form5,
        "p2_form5": p2_form5,
        "form5_diff": p1_form5 - p2_form5,
        "p1_form10": p1_form10,
        "p2_form10": p2_form10,
        "form10_diff": p1_form10 - p2_form10,
        "p1_form20": p1_form20,
        "p2_form20": p2_form20,

        # Surface form
        "p1_surface_form": p1_surface_form,
        "p2_surface_form": p2_surface_form,
        "surface_form_diff": p1_surface_form - p2_surface_form,
        "p1_surface_matches": min(p1_surface_matches, 30),
        "p2_surface_matches": min(p2_surface_matches, 30),

        # H2H
        "h2h_wr": h2h_wr,
        "h2h_total": min(h2h_total, 20),
        "h2h_surface_wr": h2h_surface_wr,
        "h2h_surface_total": min(h2h_surface_total, 10),

        # Tournament
        "p1_tournament_wr": p1_tourn_wr,
        "p2_tournament_wr": p2_tourn_wr,
        "tournament_wr_diff": p1_tourn_wr - p2_tourn_wr,

        # Regional
        "p1_region_wr": p1_region_wr,
        "p2_region_wr": p2_region_wr,
        "region_wr_diff": p1_region_wr - p2_region_wr,

        # Style / matchup
        "p1_is_lefty": p1_is_lefty,
        "p2_is_lefty": p2_is_lefty,
        "p1_style": p1_style_num,
        "p2_style": p2_style_num,
        "p1_serve_penalty": p1_serve_penalty,
        "p2_serve_penalty": p2_serve_penalty,

        # Streaks
        "p1_streak": p1_streak,
        "p2_streak": p2_streak,
        "streak_diff": p1_streak - p2_streak,

        # Flags
        "p1_injury": p1_injury,
        "p2_injury": p2_injury,
        "p1_coach_change": p1_coach_change,
        "p2_coach_change": p2_coach_change,

        # Match context
        "best_of": best_of,
        "is_indoor": indoor,
        "surface_code": {"hard": 0, "clay": 1, "grass": 2}.get(surface_key, 0),

        # Experience (Elo match counts)
        "p1_experience": min(p1_global_n, 200),
        "p2_experience": min(p2_global_n, 200),
        "exp_diff": p1_global_n - p2_global_n,

        # Fatigue / Scheduling
        "p1_days_since_last": p1_fatigue["days_since_last"],
        "p2_days_since_last": p2_fatigue["days_since_last"],
        "p1_sets_last_14d": p1_fatigue["sets_last_14d"],
        "p2_sets_last_14d": p2_fatigue["sets_last_14d"],
        "p1_matches_last_14d": p1_fatigue["matches_last_14d"],
        "p2_matches_last_14d": p2_fatigue["matches_last_14d"],
        "p1_timezone_jump": p1_fatigue["timezone_jump"],
        "p2_timezone_jump": p2_fatigue["timezone_jump"],
        "fatigue_diff": p1_fatigue["sets_last_14d"] - p2_fatigue["sets_last_14d"],

        # Extended Serve Stats
        "p1_serve_hold_pct": p1_serve["serve_hold_pct"],
        "p2_serve_hold_pct": p2_serve["serve_hold_pct"],
        "serve_hold_diff": p1_serve["serve_hold_pct"] - p2_serve["serve_hold_pct"],
        "p1_return_win_pct": p1_serve["return_win_pct"],
        "p2_return_win_pct": p2_serve["return_win_pct"],
        "p1_2nd_serve_pct": p1_serve["second_serve_pct"],
        "p2_2nd_serve_pct": p2_serve["second_serve_pct"],

        # Round & Tournament Tier
        "round_importance": round_importance,
        "tier_weight": tier_weight,
    }


def _compute_fatigue(conn, player_id: int, current_tournament: str = "",
                     ref_date: str = "") -> dict:
    """Compute fatigue features: days since last match, sets in 14 days, timezone jumps."""
    if not ref_date:
        ref_date = datetime.now().strftime("%Y-%m-%d")

    date_14d_ago = (datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")

    # Recent matches in last 14 days
    rows = conn.execute("""
        SELECT date, tournament, w_sets, l_sets, winner_id
        FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?) AND date >= ? AND date < ?
        AND comment NOT LIKE '%Walkover%'
        ORDER BY date DESC
    """, (player_id, player_id, date_14d_ago, ref_date)).fetchall()

    matches_14d = len(rows)
    sets_14d = sum((r["w_sets"] or 0) + (r["l_sets"] or 0) for r in rows)

    # Days since last match
    last_match = conn.execute("""
        SELECT date FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?) AND date < ?
        ORDER BY date DESC LIMIT 1
    """, (player_id, player_id, ref_date)).fetchone()

    if last_match:
        last_date = datetime.strptime(last_match["date"], "%Y-%m-%d")
        ref = datetime.strptime(ref_date, "%Y-%m-%d")
        days_since = (ref - last_date).days
    else:
        days_since = 30  # no recent data

    # Timezone / continent jump detection
    timezone_jump = 0
    if rows and current_tournament:
        current_continent = _tournament_to_continent(current_tournament)
        last_tournament = rows[0]["tournament"]
        last_continent = _tournament_to_continent(last_tournament)
        if current_continent != "unknown" and last_continent != "unknown":
            if current_continent != last_continent:
                timezone_jump = 1

    return {
        "days_since_last": min(days_since, 30),
        "sets_last_14d": min(sets_14d, 30),
        "matches_last_14d": min(matches_14d, 10),
        "timezone_jump": timezone_jump,
    }


def _tournament_to_continent(tournament: str) -> str:
    """Map tournament name to continent for timezone jump detection."""
    t = tournament.lower()
    for keyword, continent in TOURNAMENT_CONTINENTS.items():
        if keyword in t:
            return continent
    return "unknown"


def _compute_serve_features(conn, player_id: int, ref_date: str = "", n: int = 10) -> dict:
    """Compute extended serve features from recent match stats."""
    if not ref_date:
        ref_date = datetime.now().strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT winner_id,
               w_svpt, l_svpt, w_1st_in, l_1st_in,
               w_1st_won, l_1st_won, w_2nd_won, l_2nd_won,
               w_bp_saved, l_bp_saved, w_bp_faced, l_bp_faced
        FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?) AND date < ?
        AND w_svpt > 0
        ORDER BY date DESC LIMIT ?
    """, (player_id, player_id, ref_date, n)).fetchall()

    if len(rows) < 3:
        return {"serve_hold_pct": 0, "return_win_pct": 0, "second_serve_pct": 0}

    hold_pcts, return_pcts, second_pcts = [], [], []

    for r in rows:
        is_winner = r["winner_id"] == player_id

        # Serve points and wins
        svpt = r["w_svpt"] if is_winner else r["l_svpt"]
        first_won = r["w_1st_won"] if is_winner else r["l_1st_won"]
        second_won = r["w_2nd_won"] if is_winner else r["l_2nd_won"]
        first_in = r["w_1st_in"] if is_winner else r["l_1st_in"]

        # Serve hold % = (1st_won + 2nd_won) / serve_points
        if svpt and svpt > 0:
            hold_pcts.append((first_won + second_won) / svpt)

        # Second serve % = 2nd_won / (svpt - 1st_in)
        second_serve_pts = svpt - first_in if (svpt and first_in and svpt > first_in) else 0
        if second_serve_pts > 0:
            second_pcts.append(second_won / second_serve_pts)

        # Return stats (from opponent's serve)
        opp_svpt = r["l_svpt"] if is_winner else r["w_svpt"]
        opp_first_won = r["l_1st_won"] if is_winner else r["w_1st_won"]
        opp_second_won = r["l_2nd_won"] if is_winner else r["w_2nd_won"]
        if opp_svpt and opp_svpt > 0:
            opp_hold = (opp_first_won + opp_second_won) / opp_svpt
            return_pcts.append(1 - opp_hold)  # return win % = 1 - opponent hold %

    return {
        "serve_hold_pct": round(float(sum(hold_pcts) / len(hold_pcts)), 4) if hold_pcts else 0,
        "return_win_pct": round(float(sum(return_pcts) / len(return_pcts)), 4) if return_pcts else 0,
        "second_serve_pct": round(float(sum(second_pcts) / len(second_pcts)), 4) if second_pcts else 0,
    }
