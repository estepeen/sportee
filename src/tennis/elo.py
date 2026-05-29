"""Tennis Elo rating system - global + per surface."""

import logging
from src.tennis.database import get_tennis_db

logger = logging.getLogger(__name__)

# K-factor by tournament importance
K_FACTORS = {
    "Grand Slam": 48,
    "Masters 1000": 36,
    "ATP 500": 28,
    "WTA 1000": 36,
    "WTA 500": 28,
    "ATP 250": 22,
    "WTA 250": 22,
}
DEFAULT_K = 24

# Surface categories for per-surface Elo
SURFACE_TYPES = {
    "Hard": "hard",
    "Clay": "clay",
    "Grass": "grass",
    "Carpet": "hard",  # carpet treated as hard
}

INITIAL_ELO = 1500


def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected probability of player A winning."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def compute_all_elo():
    """Compute Elo ratings from all historical matches."""
    conn = get_tennis_db()

    # Clear existing ratings
    conn.execute("DELETE FROM tennis_elo")
    conn.commit()

    # Global + surface Elo for each player
    elo = {}  # {player_id: {"global": 1500, "hard": 1500, ...}}
    match_counts = {}  # {player_id: {"global": 0, ...}}

    def get_elo(pid, elo_type):
        if pid not in elo:
            elo[pid] = {}
            match_counts[pid] = {}
        if elo_type not in elo[pid]:
            elo[pid][elo_type] = INITIAL_ELO
            match_counts[pid][elo_type] = 0
        return elo[pid][elo_type]

    def update_elo(winner_id, loser_id, elo_type, k):
        w_elo = get_elo(winner_id, elo_type)
        l_elo = get_elo(loser_id, elo_type)

        exp_w = expected_score(w_elo, l_elo)
        exp_l = 1.0 - exp_w

        elo[winner_id][elo_type] = w_elo + k * (1.0 - exp_w)
        elo[loser_id][elo_type] = l_elo + k * (0.0 - exp_l)

        match_counts[winner_id][elo_type] = match_counts[winner_id].get(elo_type, 0) + 1
        match_counts[loser_id][elo_type] = match_counts[loser_id].get(elo_type, 0) + 1

    # Process all matches chronologically
    matches = conn.execute("""
        SELECT winner_id, loser_id, surface, series, indoor, date
        FROM tennis_matches
        WHERE comment NOT LIKE '%Retired%' AND comment NOT LIKE '%Walkover%'
        ORDER BY date ASC
    """).fetchall()

    logger.info(f"Computing Elo from {len(matches)} matches...")

    for m in matches:
        w_id = m["winner_id"]
        l_id = m["loser_id"]
        surface_raw = m["surface"] or "Hard"
        surface_key = SURFACE_TYPES.get(surface_raw, "hard")
        series = m["series"] or ""
        indoor = m["indoor"]
        k = K_FACTORS.get(series, DEFAULT_K)

        # Global Elo
        update_elo(w_id, l_id, "global", k)

        # Surface Elo
        update_elo(w_id, l_id, surface_key, k)

        # Indoor Elo (separate from hard outdoor)
        if indoor:
            update_elo(w_id, l_id, "indoor", k)

    # Save to database
    for pid, ratings in elo.items():
        for elo_type, rating in ratings.items():
            conn.execute("""
                INSERT OR REPLACE INTO tennis_elo (player_id, elo_type, elo, matches, last_updated)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (pid, elo_type, round(rating, 1), match_counts[pid].get(elo_type, 0)))

    conn.commit()

    # Stats
    total_players = len(elo)
    top_global = conn.execute("""
        SELECT p.name, e.elo FROM tennis_elo e
        JOIN tennis_players p ON e.player_id = p.id
        WHERE e.elo_type = 'global'
        ORDER BY e.elo DESC LIMIT 10
    """).fetchall()

    conn.close()

    logger.info(f"Elo computed for {total_players} players")
    for i, row in enumerate(top_global):
        logger.info(f"  #{i+1} {row['name']}: {row['elo']}")

    return total_players


def compute_elo_history(conn) -> dict:
    """Replay all matches chronologically and snapshot PRE-match Elo for each match.

    Returns {match_id: {w_global, l_global, w_surface, l_surface,
                        w_global_n, l_global_n, w_surface_n, l_surface_n}}.

    This is the leakage-free counterpart to the live `tennis_elo` table: training
    must see the rating a player carried *into* a match, not the post-match value
    (which has already absorbed the outcome). The Elo trajectory mirrors
    compute_all_elo() — updates are applied only for non-Retired/non-Walkover
    matches — but a snapshot is recorded for every match so training rows
    (which include Retired) can all be looked up.
    """
    elo = {}
    counts = {}

    def get(pid, t):
        if pid not in elo:
            elo[pid] = {}
            counts[pid] = {}
        if t not in elo[pid]:
            elo[pid][t] = INITIAL_ELO
            counts[pid][t] = 0
        return elo[pid][t]

    def update(w_id, l_id, t, k):
        w, l = get(w_id, t), get(l_id, t)
        exp_w = expected_score(w, l)
        elo[w_id][t] = w + k * (1.0 - exp_w)
        elo[l_id][t] = l + k * (0.0 - (1.0 - exp_w))
        counts[w_id][t] += 1
        counts[l_id][t] += 1

    matches = conn.execute("""
        SELECT id, winner_id, loser_id, surface, series, indoor, date, comment
        FROM tennis_matches
        ORDER BY date ASC, id ASC
    """).fetchall()

    history = {}
    for m in matches:
        w_id, l_id = m["winner_id"], m["loser_id"]
        surface_key = SURFACE_TYPES.get(m["surface"] or "Hard", "hard")

        # Snapshot BEFORE applying the match outcome
        history[m["id"]] = {
            "w_global": get(w_id, "global"), "l_global": get(l_id, "global"),
            "w_surface": get(w_id, surface_key), "l_surface": get(l_id, surface_key),
            "w_global_n": counts[w_id]["global"], "l_global_n": counts[l_id]["global"],
            "w_surface_n": counts[w_id][surface_key], "l_surface_n": counts[l_id][surface_key],
        }

        comment = (m["comment"] or "")
        if "Retired" in comment or "Walkover" in comment:
            continue  # mirror compute_all_elo: don't let these move ratings

        k = K_FACTORS.get(m["series"] or "", DEFAULT_K)
        update(w_id, l_id, "global", k)
        update(w_id, l_id, surface_key, k)
        if m["indoor"]:
            update(w_id, l_id, "indoor", k)

    return history


def get_player_elo(player_id: int) -> dict:
    """Get all Elo ratings for a player."""
    conn = get_tennis_db()
    rows = conn.execute(
        "SELECT elo_type, elo, matches FROM tennis_elo WHERE player_id = ?",
        (player_id,)
    ).fetchall()
    conn.close()
    return {r["elo_type"]: {"elo": r["elo"], "matches": r["matches"]} for r in rows}


def get_player_elo_by_name(name: str) -> dict:
    """Get all Elo ratings for a player by name."""
    conn = get_tennis_db()
    player = conn.execute("SELECT id FROM tennis_players WHERE name = ?", (name,)).fetchone()
    if not player:
        conn.close()
        return {}
    result = get_player_elo(player["id"])
    conn.close()
    return result
