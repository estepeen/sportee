"""Compute comprehensive player statistics from match database."""

import unicodedata
from datetime import datetime
from src.tennis.database import get_tennis_db


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def find_player(conn, name: str):
    """Find player by exact name, or fuzzy match (last name + initial, full name, partial)."""
    # Try both original and stripped diacritics
    candidates = [name]
    stripped = _strip_diacritics(name)
    if stripped != name:
        candidates.append(stripped)

    for n in candidates:
        row = conn.execute("SELECT * FROM tennis_players WHERE name = ?", (n,)).fetchone()
        if row:
            return row

    # Try "Last F." format from full name "First Last"
    for n in candidates:
        parts = n.strip().split()
        if len(parts) >= 2:
            last = parts[-1]
            first_init = parts[0][0]
            row = conn.execute(
                "SELECT * FROM tennis_players WHERE name = ?",
                (f"{last} {first_init}.",)
            ).fetchone()
            if row:
                return row

            first_word = parts[0]
            second_init = parts[-1][0]
            row = conn.execute(
                "SELECT * FROM tennis_players WHERE name = ?",
                (f"{first_word} {second_init}.",)
            ).fetchone()
            if row:
                return row

    # LIKE search on last name (stripped)
    for n in candidates:
        for part in n.strip().split():
            if len(part) >= 3:
                row = conn.execute(
                    "SELECT * FROM tennis_players WHERE name LIKE ? ORDER BY LENGTH(name) LIMIT 1",
                    (f"{part}%",)
                ).fetchone()
                if row:
                    return row

    return None


def get_player_profile(player_name: str) -> dict | None:
    """Build full player profile with all stats."""
    conn = get_tennis_db()

    player = find_player(conn, player_name)
    if not player:
        conn.close()
        return None

    pid = player["id"]
    now_year = datetime.now().year

    profile = {
        "id": pid,
        "name": player["name"],
        "hand": player["hand"],
        "style": player["style"],
        "backhand": player["backhand"],
        "country": player["country"],
        "birth_year": player["birth_year"],
        "age": now_year - player["birth_year"] if player["birth_year"] > 0 else 0,
        "height_cm": player["height_cm"],
        "notes": player["notes"],
    }

    # ── Elo ratings ──
    elo_rows = conn.execute(
        "SELECT elo_type, elo, matches FROM tennis_elo WHERE player_id = ?", (pid,)
    ).fetchall()
    profile["elo"] = {r["elo_type"]: {"elo": round(r["elo"], 1), "matches": r["matches"]} for r in elo_rows}

    # ── Latest rank ──
    rank_row = conn.execute("""
        SELECT COALESCE(
            (SELECT winner_rank FROM tennis_matches WHERE winner_id = ? AND winner_rank > 0 ORDER BY date DESC LIMIT 1),
            (SELECT loser_rank FROM tennis_matches WHERE loser_id = ? AND loser_rank > 0 ORDER BY date DESC LIMIT 1),
            0
        ) as rank
    """, (pid, pid)).fetchone()
    profile["rank"] = rank_row["rank"] if rank_row else 0

    # ── Career record ──
    career = conn.execute("""
        SELECT
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total
        FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?)
        AND comment NOT LIKE '%Walkover%'
    """, (pid, pid, pid)).fetchone()
    profile["career"] = {
        "wins": career["wins"] or 0,
        "losses": (career["total"] or 0) - (career["wins"] or 0),
        "total": career["total"] or 0,
        "winrate": round((career["wins"] or 0) / career["total"] * 100, 1) if career["total"] else 0,
    }

    # ── Season records ──
    profile["seasons"] = {}
    for year in range(now_year, now_year - 3, -1):
        s = conn.execute("""
            SELECT
                SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
                COUNT(*) as total
            FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?)
            AND date LIKE ?
            AND comment NOT LIKE '%Walkover%'
        """, (pid, pid, pid, f"{year}%")).fetchone()
        if s["total"] and s["total"] > 0:
            profile["seasons"][year] = {
                "wins": s["wins"] or 0,
                "losses": (s["total"] or 0) - (s["wins"] or 0),
                "winrate": round((s["wins"] or 0) / s["total"] * 100, 1),
            }

    # ── Surface records ──
    profile["surfaces"] = {}
    for surface in ("Hard", "Clay", "Grass"):
        s = conn.execute("""
            SELECT
                SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
                COUNT(*) as total
            FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?)
            AND surface = ?
            AND comment NOT LIKE '%Walkover%'
        """, (pid, pid, pid, surface)).fetchone()
        if s["total"] and s["total"] > 0:
            elo_key = surface.lower()
            elo_data = profile["elo"].get(elo_key, {"elo": 1500, "matches": 0})
            profile["surfaces"][surface] = {
                "wins": s["wins"] or 0,
                "losses": (s["total"] or 0) - (s["wins"] or 0),
                "winrate": round((s["wins"] or 0) / s["total"] * 100, 1),
                "elo": elo_data["elo"],
            }

    # Indoor record
    s = conn.execute("""
        SELECT
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total
        FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?) AND indoor = 1
        AND comment NOT LIKE '%Walkover%'
    """, (pid, pid, pid)).fetchone()
    if s["total"] and s["total"] > 0:
        elo_data = profile["elo"].get("indoor", {"elo": 1500, "matches": 0})
        profile["surfaces"]["Indoor"] = {
            "wins": s["wins"] or 0,
            "losses": (s["total"] or 0) - (s["wins"] or 0),
            "winrate": round((s["wins"] or 0) / s["total"] * 100, 1),
            "elo": elo_data["elo"],
        }

    # ── Recent form (last 20 matches) ──
    recent = conn.execute("""
        SELECT m.winner_id, m.loser_id, m.date, m.tournament, m.surface, m.round,
               m.w_sets, m.l_sets, m.score, m.series,
               m.w1, m.l1, m.w2, m.l2, m.w3, m.l3, m.w4, m.l4, m.w5, m.l5,
               p1.name as winner_name, p2.name as loser_name
        FROM tennis_matches m
        JOIN tennis_players p1 ON m.winner_id = p1.id
        JOIN tennis_players p2 ON m.loser_id = p2.id
        WHERE (m.winner_id = ? OR m.loser_id = ?)
        AND m.comment NOT LIKE '%Walkover%'
        ORDER BY m.date DESC LIMIT 20
    """, (pid, pid)).fetchall()

    profile["recent"] = []
    for m in recent:
        won = m["winner_id"] == pid
        opponent = m["loser_name"] if won else m["winner_name"]
        # Build score string from set games
        score = m["score"] or ""
        if not score:
            parts = []
            for i in range(1, 6):
                w = m[f"w{i}"]
                l = m[f"l{i}"]
                if w is not None and l is not None:
                    parts.append(f"{w}-{l}")
            score = " ".join(parts)
        profile["recent"].append({
            "won": won,
            "opponent": opponent,
            "date": m["date"],
            "tournament": m["tournament"],
            "surface": m["surface"],
            "round": m["round"],
            "score": score,
            "series": m["series"],
        })

    # Form stats
    form5 = sum(1 for m in profile["recent"][:5] if m["won"])
    form10 = sum(1 for m in profile["recent"][:10] if m["won"])
    profile["form5"] = form5
    profile["form10"] = form10
    profile["form20"] = sum(1 for m in profile["recent"] if m["won"])

    # Streak
    streak = 0
    if profile["recent"]:
        first_won = profile["recent"][0]["won"]
        for m in profile["recent"]:
            if m["won"] == first_won:
                streak += 1
            else:
                break
        if not first_won:
            streak = -streak
    profile["streak"] = streak

    # ── H2H vs top opponents ──
    h2h_raw = conn.execute("""
        SELECT
            CASE WHEN winner_id = ? THEN loser_id ELSE winner_id END as opp_id,
            CASE WHEN winner_id = ? THEN 1 ELSE 0 END as won
        FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?)
        AND comment NOT LIKE '%Walkover%'
    """, (pid, pid, pid, pid)).fetchall()

    opp_stats = {}
    for r in h2h_raw:
        oid = r["opp_id"]
        if oid not in opp_stats:
            opp_stats[oid] = {"wins": 0, "losses": 0}
        if r["won"]:
            opp_stats[oid]["wins"] += 1
        else:
            opp_stats[oid]["losses"] += 1

    # Get top opponents (most matches played)
    top_opps = sorted(opp_stats.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True)[:10]
    profile["h2h"] = []
    for oid, stats in top_opps:
        opp_name = conn.execute("SELECT name FROM tennis_players WHERE id = ?", (oid,)).fetchone()
        if opp_name:
            profile["h2h"].append({
                "opponent": opp_name["name"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total": stats["wins"] + stats["losses"],
            })

    # ── Tournament history (best results) ──
    tournaments = conn.execute("""
        SELECT tournament, round, date,
               CASE WHEN winner_id = ? THEN 1 ELSE 0 END as won
        FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?)
        AND series IN ('Grand Slam', 'Masters 1000', 'WTA 1000')
        ORDER BY date DESC
    """, (pid, pid, pid)).fetchall()

    tourn_results = {}
    round_order = {
        "The Final": 7, "Final": 7, "F": 7,
        "Semifinals": 6, "SF": 6,
        "Quarterfinals": 5, "QF": 5,
        "4th Round": 4, "R16": 4, "Round of 16": 4,
        "3rd Round": 3, "R32": 3, "Round of 32": 3,
        "2nd Round": 2, "R64": 2, "Round of 64": 2,
        "1st Round": 1, "R128": 1, "Round of 128": 1,
    }

    for t in tournaments:
        key = t["tournament"]
        rnd = t["round"]
        year = t["date"][:4] if t["date"] else ""
        # If won the final, it's a title
        result = "W" if rnd in ("The Final", "Final", "F") and t["won"] else rnd
        score = round_order.get(rnd, 0)
        if t["won"] and rnd in ("The Final", "Final", "F"):
            score = 8  # title

        if key not in tourn_results:
            tourn_results[key] = []
        tourn_results[key].append({"result": result, "year": year, "score": score})

    # Best result per tournament
    profile["tournaments"] = []
    for name, results in tourn_results.items():
        best = max(results, key=lambda x: x["score"])
        years = sorted(set(r["year"] for r in results), reverse=True)
        profile["tournaments"].append({
            "name": name,
            "best": best["result"],
            "best_year": best["year"],
            "appearances": len(set(r["year"] for r in results)),
            "years": years[:5],
        })
    profile["tournaments"].sort(key=lambda x: round_order.get(x["best"], 0), reverse=True)

    # ── Active flags ──
    flags = conn.execute("""
        SELECT flag_type, description, date_set
        FROM tennis_player_flags
        WHERE player_id = ? AND active = 1
        AND (date_expires = '' OR date_expires > date('now'))
    """, (pid,)).fetchall()
    profile["flags"] = [{"type": f["flag_type"], "desc": f["description"], "date": f["date_set"]} for f in flags]

    conn.close()
    return profile


def resolve_player_name(name: str) -> str | None:
    """Resolve any name format to the canonical DB name."""
    conn = get_tennis_db()
    player = find_player(conn, name)
    conn.close()
    return player["name"] if player else None


def get_player_comparison(name1: str, name2: str, surface: str = "") -> dict | None:
    """Build H2H comparison between two players."""
    p1 = get_player_profile(name1)
    p2 = get_player_profile(name2)
    if not p1 or not p2:
        return None

    conn = get_tennis_db()

    # Direct H2H
    h2h = conn.execute("""
        SELECT winner_id, surface, date, tournament, round, score
        FROM tennis_matches
        WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        ORDER BY date DESC
    """, (p1["id"], p2["id"], p2["id"], p1["id"])).fetchall()

    h2h_matches = []
    p1_wins = 0
    p2_wins = 0
    surface_h2h = {"Hard": [0, 0], "Clay": [0, 0], "Grass": [0, 0]}

    for m in h2h:
        p1_won = m["winner_id"] == p1["id"]
        if p1_won:
            p1_wins += 1
        else:
            p2_wins += 1

        surf = m["surface"]
        if surf in surface_h2h:
            surface_h2h[surf][0 if p1_won else 1] += 1

        h2h_matches.append({
            "p1_won": p1_won,
            "surface": surf,
            "date": m["date"],
            "tournament": m["tournament"],
            "round": m["round"],
            "score": m["score"],
        })

    conn.close()

    return {
        "player1": p1,
        "player2": p2,
        "h2h_total": {"p1_wins": p1_wins, "p2_wins": p2_wins, "total": len(h2h)},
        "h2h_by_surface": surface_h2h,
        "h2h_matches": h2h_matches[:20],
    }
