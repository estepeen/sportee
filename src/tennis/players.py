"""Tennis player attributes - handedness, style, surface preferences."""

import json
import logging
from pathlib import Path

from src.tennis.database import get_tennis_db

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PLAYERS_JSON = DATA_DIR / "tennis_players.json"

# Known left-handed ATP/WTA players (important for matchup analysis)
LEFT_HANDED = {
    "Rafael Nadal", "Denis Shapovalov", "Cameron Norrie",
    "Corentin Moutet", "Albert Ramos-Vinolas", "Pedro Martinez",
    "Jiri Lehecka", "Bernabe Zapata Miralles", "Nuno Borges",
    "Mackenzie McDonald", "Botic van de Zandschulp",
    "Lloyd Harris", "Aljaz Bedene", "Fernando Verdasco",
    # WTA
    "Petra Kvitova", "Angelique Kerber", "Jil Teichmann",
    "Diane Parry", "Nadia Podoroska", "Viktorija Golubic",
    "Sachia Vickery", "Shelby Rogers",
}

# Playing style classifications
PLAYER_STYLES = {
    # Serve-based (big servers, struggle on clay)
    "serve_based": {
        "Reilly Opelka", "John Isner", "Sam Querrey", "Ivo Karlovic",
        "Milos Raonic", "Maxime Cressy", "Nick Kyrgios", "Matteo Berrettini",
        "Hubert Hurkacz", "Ben Shelton", "Felix Auger-Aliassime",
        "Marcos Giron",
        # WTA
        "Karolina Pliskova", "Madison Keys", "Aryna Sabalenka",
        "Serena Williams", "Coco Gauff",
    },
    # Aggressive baseliners
    "aggressive": {
        "Carlos Alcaraz", "Novak Djokovic", "Alexander Zverev",
        "Daniil Medvedev", "Jannik Sinner", "Holger Rune",
        "Taylor Fritz", "Frances Tiafoe", "Tommy Paul",
        "Andrey Rublev", "Grigor Dimitrov", "Jack Draper",
        # WTA
        "Iga Swiatek", "Elena Rybakina", "Jessica Pegula",
        "Qinwen Zheng", "Mirra Andreeva", "Naomi Osaka",
    },
    # Defensive / counterpunchers
    "defensive": {
        "Rafael Nadal", "Casper Ruud", "Diego Schwartzman",
        "Borna Coric", "Roberto Bautista Agut", "Gael Monfils",
        "Daniel Evans", "Pablo Carreno Busta",
        # WTA
        "Caroline Garcia", "Ons Jabeur", "Daria Kasatkina",
        "Barbora Krejcikova",
    },
    # Universal / all-court
    "universal": {
        "Stefanos Tsitsipas", "Alex de Minaur", "Denis Shapovalov",
        "Cameron Norrie", "Sebastian Korda", "Ugo Humbert",
        "Karen Khachanov", "Lorenzo Musetti", "Flavio Cobolli",
        # WTA
        "Maria Sakkari", "Marketa Vondrousova", "Anna Kalinskaya",
        "Jelena Ostapenko", "Victoria Azarenka",
    },
}


def load_player_attributes() -> dict:
    """Load player attributes from JSON file."""
    try:
        with open(PLAYERS_JSON, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_player_attributes(data: dict):
    """Save player attributes to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLAYERS_JSON, "w") as f:
        json.dump(data, f, indent=2)


def sync_player_attributes():
    """Sync known player attributes to database + compute surface preferences from data."""
    conn = get_tennis_db()

    # 1. Set handedness for known left-handed players
    for name in LEFT_HANDED:
        conn.execute(
            "UPDATE tennis_players SET hand = 'L' WHERE name = ? AND hand != 'L'",
            (name,)
        )

    # 2. Set playing style for known players
    style_lookup = {}
    for style, players in PLAYER_STYLES.items():
        for name in players:
            style_lookup[name] = style

    for name, style in style_lookup.items():
        conn.execute(
            "UPDATE tennis_players SET style = ? WHERE name = ?",
            (style, name)
        )

    conn.commit()

    # 3. Compute surface preferences from match data
    players = conn.execute("SELECT id, name FROM tennis_players").fetchall()
    attrs = load_player_attributes()

    for p in players:
        pid = p["id"]
        name = p["name"]

        # Surface win rates
        surface_stats = {}
        for surface in ("Hard", "Clay", "Grass"):
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins
                FROM tennis_matches
                WHERE (winner_id = ? OR loser_id = ?)
                AND surface = ?
                AND comment NOT LIKE '%Walkover%'
            """, (pid, pid, pid, surface)).fetchone()

            total = row["total"] or 0
            wins = row["wins"] or 0
            if total >= 5:
                surface_stats[surface.lower()] = {
                    "matches": total,
                    "wins": wins,
                    "winrate": round(wins / total * 100, 1),
                }

        if surface_stats:
            # Determine best surface
            best = max(
                surface_stats.items(),
                key=lambda x: x[1]["winrate"] if x[1]["matches"] >= 10 else 0,
            )
            attrs[name] = attrs.get(name, {})
            attrs[name]["surface_stats"] = surface_stats
            attrs[name]["best_surface"] = best[0] if best[1]["matches"] >= 10 else "unknown"
            attrs[name]["hand"] = "L" if name in LEFT_HANDED else "R"
            attrs[name]["style"] = style_lookup.get(name, "unknown")

        # Recent form (last 20 matches)
        recent = conn.execute("""
            SELECT winner_id FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?)
            AND comment NOT LIKE '%Walkover%'
            ORDER BY date DESC LIMIT 20
        """, (pid, pid)).fetchall()

        if recent:
            wins_20 = sum(1 for r in recent if r["winner_id"] == pid)
            attrs.setdefault(name, {})["form_20"] = round(wins_20 / len(recent) * 100, 1)

    save_player_attributes(attrs)
    conn.close()

    logger.info(f"Synced attributes for {len(attrs)} players")
    return len(attrs)
