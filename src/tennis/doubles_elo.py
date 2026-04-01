"""Doubles Elo system - individual player doubles Elo + pair synergy Elo."""

import logging
from src.tennis.database import get_tennis_db
from src.tennis.elo import expected_score

logger = logging.getLogger(__name__)

# K-factors for doubles (slightly lower than singles - less variance in doubles)
K_FACTORS = {
    "Grand Slam": 40,
    "Masters 1000": 32,
    "ATP 500": 24,
    "WTA 1000": 32,
    "WTA 500": 24,
    "ATP 250": 20,
    "WTA 250": 20,
}
DEFAULT_K = 22

SURFACE_TYPES = {
    "Hard": "hard", "Clay": "clay", "Grass": "grass", "Carpet": "hard",
}

INITIAL_ELO = 1500


def _pair_key(p1_id: int, p2_id: int) -> tuple[int, int]:
    """Normalize pair key so (A,B) == (B,A)."""
    return (min(p1_id, p2_id), max(p1_id, p2_id))


def compute_doubles_elo():
    """Compute individual doubles Elo + pair Elo from all doubles matches."""
    conn = get_tennis_db()

    conn.execute("DELETE FROM doubles_elo")
    conn.execute("DELETE FROM doubles_pair_elo")
    conn.commit()

    # Individual doubles Elo
    elo = {}         # {player_id: {"doubles_global": 1500, ...}}
    match_counts = {}

    # Pair Elo
    pair_elo = {}    # {(p1_id, p2_id): {"pair_global": 1500, ...}}
    pair_counts = {}

    def get_ind_elo(pid, elo_type):
        if pid not in elo:
            elo[pid] = {}
            match_counts[pid] = {}
        if elo_type not in elo[pid]:
            elo[pid][elo_type] = INITIAL_ELO
            match_counts[pid][elo_type] = 0
        return elo[pid][elo_type]

    def get_pair_elo_val(key, elo_type):
        if key not in pair_elo:
            pair_elo[key] = {}
            pair_counts[key] = {}
        if elo_type not in pair_elo[key]:
            pair_elo[key][elo_type] = INITIAL_ELO
            pair_counts[key][elo_type] = 0
        return pair_elo[key][elo_type]

    def team_elo(p1_id, p2_id, elo_type):
        """Average individual Elo of a doubles team."""
        return (get_ind_elo(p1_id, elo_type) + get_ind_elo(p2_id, elo_type)) / 2

    def update_individual(winner_ids, loser_ids, elo_type, k):
        """Update individual Elo for all 4 players."""
        w_team_elo = team_elo(winner_ids[0], winner_ids[1], elo_type)
        l_team_elo = team_elo(loser_ids[0], loser_ids[1], elo_type)
        exp_w = expected_score(w_team_elo, l_team_elo)

        for pid in winner_ids:
            old = get_ind_elo(pid, elo_type)
            elo[pid][elo_type] = old + k * (1.0 - exp_w)
            match_counts[pid][elo_type] = match_counts[pid].get(elo_type, 0) + 1

        for pid in loser_ids:
            old = get_ind_elo(pid, elo_type)
            elo[pid][elo_type] = old + k * (0.0 - (1.0 - exp_w))
            match_counts[pid][elo_type] = match_counts[pid].get(elo_type, 0) + 1

    def update_pair(w_key, l_key, elo_type, k):
        """Update pair Elo."""
        w_pair = get_pair_elo_val(w_key, elo_type)
        l_pair = get_pair_elo_val(l_key, elo_type)
        exp_w = expected_score(w_pair, l_pair)

        pair_elo[w_key][elo_type] = w_pair + k * (1.0 - exp_w)
        pair_elo[l_key][elo_type] = l_pair + k * (0.0 - (1.0 - exp_w))
        pair_counts[w_key][elo_type] = pair_counts[w_key].get(elo_type, 0) + 1
        pair_counts[l_key][elo_type] = pair_counts[l_key].get(elo_type, 0) + 1

    # Process all doubles matches chronologically
    matches = conn.execute("""
        SELECT w_player1_id, w_player2_id, l_player1_id, l_player2_id,
               surface, series, date
        FROM doubles_matches
        WHERE comment NOT LIKE '%Retired%' AND comment NOT LIKE '%Walkover%'
        ORDER BY date ASC
    """).fetchall()

    logger.info(f"Computing doubles Elo from {len(matches)} matches...")

    for m in matches:
        w_ids = [m["w_player1_id"], m["w_player2_id"]]
        l_ids = [m["l_player1_id"], m["l_player2_id"]]
        surface_raw = m["surface"] or "Hard"
        surface_key = SURFACE_TYPES.get(surface_raw, "hard")
        series = m["series"] or ""
        k = K_FACTORS.get(series, DEFAULT_K)

        # Individual doubles Elo (global + surface)
        update_individual(w_ids, l_ids, "doubles_global", k)
        update_individual(w_ids, l_ids, f"doubles_{surface_key}", k)

        # Pair Elo (global + surface)
        w_pair_key = _pair_key(w_ids[0], w_ids[1])
        l_pair_key = _pair_key(l_ids[0], l_ids[1])
        update_pair(w_pair_key, l_pair_key, "pair_global", k)
        update_pair(w_pair_key, l_pair_key, f"pair_{surface_key}", k)

    # Save individual doubles Elo
    for pid, ratings in elo.items():
        for elo_type, rating in ratings.items():
            conn.execute("""
                INSERT OR REPLACE INTO doubles_elo (player_id, elo_type, elo, matches, last_updated)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (pid, elo_type, round(rating, 1), match_counts[pid].get(elo_type, 0)))

    # Save pair Elo
    for (p1, p2), ratings in pair_elo.items():
        for elo_type, rating in ratings.items():
            conn.execute("""
                INSERT OR REPLACE INTO doubles_pair_elo
                (player1_id, player2_id, elo_type, elo, matches, last_updated)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (p1, p2, elo_type, round(rating, 1), pair_counts[(p1, p2)].get(elo_type, 0)))

    conn.commit()

    # Stats
    total_ind = len(elo)
    total_pairs = len(pair_elo)

    top_ind = conn.execute("""
        SELECT p.name, e.elo, e.matches FROM doubles_elo e
        JOIN tennis_players p ON e.player_id = p.id
        WHERE e.elo_type = 'doubles_global'
        ORDER BY e.elo DESC LIMIT 10
    """).fetchall()

    top_pairs = conn.execute("""
        SELECT p1.name || ' / ' || p2.name as pair, pe.elo, pe.matches
        FROM doubles_pair_elo pe
        JOIN tennis_players p1 ON pe.player1_id = p1.id
        JOIN tennis_players p2 ON pe.player2_id = p2.id
        WHERE pe.elo_type = 'pair_global' AND pe.matches >= 5
        ORDER BY pe.elo DESC LIMIT 10
    """).fetchall()

    conn.close()

    logger.info(f"Doubles Elo: {total_ind} individual players, {total_pairs} pairs")
    logger.info("Top individual doubles Elo:")
    for r in top_ind:
        logger.info(f"  {r['name']}: {r['elo']} ({r['matches']} matches)")
    logger.info("Top pair Elo (5+ matches):")
    for r in top_pairs:
        logger.info(f"  {r['pair']}: {r['elo']} ({r['matches']} matches)")

    return total_ind


def get_doubles_elo(player_id: int) -> dict:
    """Get all doubles Elo ratings for a player."""
    conn = get_tennis_db()
    rows = conn.execute(
        "SELECT elo_type, elo, matches FROM doubles_elo WHERE player_id = ?",
        (player_id,)
    ).fetchall()
    conn.close()
    return {r["elo_type"]: {"elo": r["elo"], "matches": r["matches"]} for r in rows}


def get_pair_elo(player1_id: int, player2_id: int) -> dict:
    """Get pair Elo for a specific doubles pair."""
    p1, p2 = _pair_key(player1_id, player2_id)
    conn = get_tennis_db()
    rows = conn.execute(
        "SELECT elo_type, elo, matches FROM doubles_pair_elo WHERE player1_id = ? AND player2_id = ?",
        (p1, p2)
    ).fetchall()
    conn.close()
    return {r["elo_type"]: {"elo": r["elo"], "matches": r["matches"]} for r in rows}
