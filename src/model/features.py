"""Advanced feature engineering for CS2 match prediction."""

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_roster_features(team_name: str) -> dict:
    """Get roster-related features for a team."""
    try:
        with open(DATA_DIR / "rosters.json", "r") as f:
            rosters = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return _default_roster()

    roster = rosters.get(team_name)
    if not roster:
        return _default_roster()

    players = roster.get("players", [])
    ages = [p["age"] for p in players if p.get("age")]
    avg_age = sum(ages) / len(ages) if ages else 23

    nationalities = set(p.get("nationality", "") for p in players if p.get("nationality"))

    return {
        "player_count": len(players),
        "avg_age": round(avg_age, 1),
        "experience_score": min(avg_age / 28, 1.0),  # older = more experienced
        "is_international": len(nationalities) > 2,
        "nationality_count": len(nationalities),
    }


def _default_roster():
    return {
        "player_count": 5,
        "avg_age": 23,
        "experience_score": 0.82,
        "is_international": False,
        "nationality_count": 1,
    }


def get_roster_stability(team_name: str) -> dict:
    """Check how recently the roster changed."""
    try:
        with open(DATA_DIR / "alerts.json", "r") as f:
            alerts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        alerts = []

    team_changes = [
        a for a in alerts
        if a.get("type") in ("roster_change", "news_roster")
        and team_name.lower() in str(a).lower()
    ]

    if not team_changes:
        return {"roster_stability": 1.0, "has_standin": False, "recent_changes": 0}

    return {
        "roster_stability": max(0.3, 1.0 - len(team_changes) * 0.2),
        "has_standin": any("stand-in" in str(a).lower() or "standin" in str(a).lower() for a in team_changes),
        "recent_changes": len(team_changes),
    }


def get_tournament_features(event_name: str, tier: str, is_lan: bool, match_name: str = "") -> dict:
    """Extract tournament context features."""
    tier_importance = {"s": 1.0, "a": 0.8, "b": 0.6, "c": 0.4, "d": 0.2}
    importance = tier_importance.get(str(tier).lower(), 0.3)

    # Bracket stage detection from match name
    match_lower = (match_name or "").lower()
    is_final = any(kw in match_lower for kw in ["final", "grand final", "championship"])
    is_semifinal = "semi" in match_lower
    is_quarter = "quarter" in match_lower
    is_group = any(kw in match_lower for kw in ["group", "round", "opening"])
    is_elimination = any(kw in match_lower for kw in ["elimination", "decider", "lower bracket", "loser"])
    is_upper = any(kw in match_lower for kw in ["upper bracket", "winner"])

    # Bracket stage importance multiplier
    if is_final:
        stage_importance = 1.0
    elif is_semifinal:
        stage_importance = 0.9
    elif is_quarter:
        stage_importance = 0.8
    elif is_elimination:
        stage_importance = 0.85  # elimination matches are high pressure
    elif is_upper:
        stage_importance = 0.7
    elif is_group:
        stage_importance = 0.5
    else:
        stage_importance = 0.6

    return {
        "tier_importance": importance,
        "is_lan": 1 if is_lan else 0,
        "is_final": 1 if is_final else 0,
        "is_semifinal": 1 if is_semifinal else 0,
        "is_elimination": 1 if is_elimination else 0,
        "is_upper_bracket": 1 if is_upper else 0,
        "is_group_stage": 1 if is_group else 0,
        "stage_importance": stage_importance,
        "total_importance": importance * stage_importance,
    }


def get_timezone_distance(team_location: str, event_country: str) -> float:
    """Calculate timezone jetlag penalty. Returns 0-1 (1 = max jetlag)."""
    tz_map = {
        "US": -5, "CA": -5, "BR": -3, "AR": -3, "MX": -6,
        "FR": 1, "DE": 1, "DK": 1, "SE": 1, "NO": 1, "FI": 2,
        "PL": 1, "CZ": 1, "NL": 1, "BE": 1, "ES": 1, "PT": 0,
        "GB": 0, "IE": 0, "IT": 1, "AT": 1, "CH": 1,
        "RO": 2, "BG": 2, "TR": 3, "UA": 2, "EE": 2, "IL": 2,
        "RU": 3, "KZ": 5, "CN": 8, "KR": 9, "JP": 9, "MN": 8,
        "AU": 10, "NZ": 12, "SA": 3, "AE": 4,
    }
    team_tz = tz_map.get(team_location, 1)
    event_tz = tz_map.get(event_country, 1)
    hours_diff = abs(team_tz - event_tz)
    # Normalize: 0-12h -> 0-1 penalty
    return min(hours_diff / 12, 1.0)


def get_lan_online_split(matches_history: list, team_id: int) -> dict:
    """Calculate team's LAN vs Online performance from match history."""
    lan_wins = lan_total = online_wins = online_total = 0

    for m in matches_history:
        is_lan = m.get("is_lan", False)
        won = m.get("winner_id") == team_id

        if is_lan:
            lan_total += 1
            if won:
                lan_wins += 1
        else:
            online_total += 1
            if won:
                online_wins += 1

    return {
        "lan_winrate": lan_wins / lan_total if lan_total > 0 else 0.5,
        "online_winrate": online_wins / online_total if online_total > 0 else 0.5,
        "lan_matches": lan_total,
        "online_matches": online_total,
        "lan_preference": (lan_wins / lan_total if lan_total > 0 else 0.5) -
                          (online_wins / online_total if online_total > 0 else 0.5),
    }


def get_playoff_experience(matches_history: list, team_id: int) -> dict:
    """How experienced is the team in high-pressure situations?"""
    playoff_keywords = ["final", "semi", "quarter", "playoff", "elimination", "decider"]
    playoff_wins = playoff_total = 0

    for m in matches_history:
        event = str(m.get("event_name", "")).lower()
        match_name = str(m.get("match_name", "")).lower()
        combined = event + " " + match_name

        if any(kw in combined for kw in playoff_keywords):
            playoff_total += 1
            if m.get("winner_id") == team_id:
                playoff_wins += 1

    return {
        "playoff_experience": min(playoff_total / 10, 1.0),
        "playoff_winrate": playoff_wins / playoff_total if playoff_total > 0 else 0.5,
        "playoff_matches": playoff_total,
    }
