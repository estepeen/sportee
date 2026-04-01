"""Player data scraper - tracks individual player form, experience, and roster info."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import httpx

from config.settings import PANDASCORE_API_KEY

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PLAYERS_FILE = DATA_DIR / "players.json"
ROSTERS_FILE = DATA_DIR / "rosters.json"


class PlayerTracker:
    """Tracks player data, rosters, and detects changes."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
        self.players: dict[int, dict] = {}  # player_id -> player data
        self.rosters: dict[str, dict] = {}  # team_name -> roster info

    async def _ps_get(self, endpoint: str, params: dict = None) -> list | dict | None:
        p = params or {}
        p["token"] = PANDASCORE_API_KEY
        try:
            resp = await self.client.get(f"https://api.pandascore.co{endpoint}", params=p)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"PandaScore error: {e}")
        return None

    async def fetch_all_team_rosters(self, pages: int = 5) -> dict[str, dict]:
        """Fetch rosters for all CS2 teams with player details."""
        logger.info("Fetching team rosters + player data...")

        all_teams = []
        for page in range(1, pages + 1):
            data = await self._ps_get("/csgo/teams", {"per_page": 100, "page": page})
            if not data:
                break
            all_teams.extend(data)
            if len(data) < 100:
                break

        rosters = {}
        for team in all_teams:
            name = team["name"]
            players = []
            for p in team.get("players", []):
                if not p.get("active"):
                    continue
                player_data = {
                    "id": p["id"],
                    "name": p["name"],
                    "first_name": p.get("first_name", ""),
                    "last_name": p.get("last_name", ""),
                    "nationality": p.get("nationality", ""),
                    "age": p.get("age"),
                    "role": p.get("role"),
                    "image_url": p.get("image_url"),
                }
                players.append(player_data)
                self.players[p["id"]] = player_data

            rosters[name] = {
                "team_id": team["id"],
                "players": players,
                "player_count": len(players),
                "image_url": team.get("image_url"),
                "dark_image_url": team.get("dark_mode_image_url"),
                "location": team.get("location", ""),
                "avg_age": self._avg_age(players),
                "nationalities": list(set(p["nationality"] for p in players if p["nationality"])),
            }

        self.rosters = rosters
        logger.info(f"Fetched rosters for {len(rosters)} teams, {len(self.players)} players")
        return rosters

    @staticmethod
    def _avg_age(players: list) -> float | None:
        ages = [p["age"] for p in players if p.get("age")]
        return round(sum(ages) / len(ages), 1) if ages else None

    def detect_changes(self) -> list[dict]:
        """Compare current rosters with previously saved ones."""
        try:
            with open(ROSTERS_FILE, "r") as f:
                previous = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}

        changes = []
        for team_name, roster in self.rosters.items():
            current_names = sorted(p["name"] for p in roster["players"])
            prev_data = previous.get(team_name, {})
            if isinstance(prev_data, list):
                prev_names = sorted(prev_data)
            elif isinstance(prev_data, dict):
                prev_names = sorted(p["name"] for p in prev_data.get("players", []))
            else:
                prev_names = []

            if prev_names and current_names != prev_names:
                added = set(current_names) - set(prev_names)
                removed = set(prev_names) - set(current_names)
                changes.append({
                    "team": team_name,
                    "added": list(added),
                    "removed": list(removed),
                    "type": "roster_change",
                    "timestamp": datetime.utcnow().isoformat(),
                    "is_coach_change": False,
                })

        return changes

    def get_team_profile(self, team_name: str) -> dict | None:
        """Get comprehensive team profile."""
        roster = self.rosters.get(team_name)
        if not roster:
            return None

        players = roster["players"]
        return {
            "name": team_name,
            "team_id": roster["team_id"],
            "image_url": roster.get("image_url"),
            "location": roster.get("location"),
            "players": players,
            "player_count": len(players),
            "avg_age": roster.get("avg_age"),
            "nationalities": roster.get("nationalities", []),
            "is_international": len(roster.get("nationalities", [])) > 2,
            # Experience proxy: avg age (older = more experienced)
            "experience_score": min((roster.get("avg_age") or 20) / 28, 1.0),
        }

    def get_timezone_distance(self, team_location: str, event_location: str) -> int:
        """Estimate timezone distance between team home and event location.
        Returns approximate hours of timezone difference.
        """
        tz_map = {
            # Americas
            "US": -5, "CA": -5, "BR": -3, "AR": -3, "CL": -3, "MX": -6,
            # Europe
            "FR": 1, "DE": 1, "DK": 1, "SE": 1, "NO": 1, "FI": 2,
            "PL": 1, "CZ": 1, "NL": 1, "BE": 1, "ES": 1, "PT": 0,
            "GB": 0, "IE": 0, "IT": 1, "AT": 1, "CH": 1, "HU": 1,
            "RO": 2, "BG": 2, "GR": 2, "TR": 3, "UA": 2, "EE": 2,
            "LV": 2, "LT": 2, "RS": 1, "HR": 1, "BA": 1, "ME": 1,
            "IL": 2,
            # CIS
            "RU": 3, "KZ": 5, "BY": 3,
            # Asia
            "CN": 8, "KR": 9, "JP": 9, "MN": 8,
            "TH": 7, "VN": 7, "PH": 8, "ID": 7, "MY": 8, "SG": 8,
            "IN": 5,
            # Oceania
            "AU": 10, "NZ": 12,
            # Middle East
            "SA": 3, "AE": 4, "QA": 3,
        }

        team_tz = tz_map.get(team_location, 1)
        event_tz = tz_map.get(event_location, 1)
        return abs(team_tz - event_tz)

    def save(self):
        """Save current roster data."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(ROSTERS_FILE, "w") as f:
            json.dump(self.rosters, f, indent=2)
        with open(PLAYERS_FILE, "w") as f:
            json.dump(self.players, f, indent=2, default=str)
        logger.info(f"Saved {len(self.rosters)} rosters, {len(self.players)} players")

    def load(self) -> bool:
        """Load saved roster data."""
        try:
            with open(ROSTERS_FILE, "r") as f:
                self.rosters = json.load(f)
            with open(PLAYERS_FILE, "r") as f:
                self.players = json.load(f)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    async def close(self):
        await self.client.aclose()
