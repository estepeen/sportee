"""Fundamental analysis: roster changes, news, standins, social signals."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
import httpx
from sqlalchemy import select, text

from src.database import async_session, init_db
from config.settings import PANDASCORE_API_KEY

logger = logging.getLogger(__name__)

# Keywords that signal roster instability
ROSTER_KEYWORDS = [
    "bench", "benched", "released", "kicked", "leave", "left",
    "stand-in", "standin", "substitute", "replace", "replacement",
    "join", "joined", "signs", "signed", "transfer", "loan",
    "trial", "tryout", "inactive", "available", "free agent",
    "visa", "visa issue", "unable to attend", "won't attend",
    "missing", "absent", "coach", "new igl", "igl change",
    "roster change", "lineup change",
]

# Keywords for negative sentiment
NEGATIVE_KEYWORDS = [
    "dumped out", "eliminated", "upset", "struggling", "losing streak",
    "internal issues", "drama", "controversy", "poor performance",
    "tilted", "frustrated", "not on the same page", "disband",
]

# Keywords for positive sentiment
POSITIVE_KEYWORDS = [
    "win streak", "dominant", "crush", "destroy", "flawless",
    "qualify", "qualified", "champion", "trophy", "mvp",
    "on fire", "incredible", "outstanding",
]


class FundamentalsTracker:
    """Tracks roster changes, news sentiment, and other fundamentals."""

    def __init__(self):
        self.delay = 3.0
        self.ps_client = httpx.AsyncClient(timeout=30)

    def _get(self, url: str) -> Optional[BeautifulSoup]:
        try:
            resp = cffi_requests.get(url, impersonate="chrome", timeout=30)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.error(f"Request failed: {e}")
        return None

    async def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        loop = asyncio.get_event_loop()
        soup = await loop.run_in_executor(None, self._get, url)
        await asyncio.sleep(self.delay)
        return soup

    # ─── Roster Tracking via PandaScore ──────────────────────

    async def fetch_rosters(self) -> dict[str, list[str]]:
        """Fetch current rosters for all known teams from PandaScore."""
        logger.info("Fetching team rosters from PandaScore...")
        rosters = {}

        page = 1
        while page <= 10:
            resp = await self.ps_client.get(
                "https://api.pandascore.co/csgo/teams",
                params={"per_page": 100, "page": page, "token": PANDASCORE_API_KEY},
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break

            for team in data:
                name = team["name"]
                players = [p["name"] for p in team.get("players", []) if p.get("active")]
                if players:
                    rosters[name] = sorted(players)
            page += 1

        logger.info(f"Fetched rosters for {len(rosters)} teams")
        return rosters

    async def detect_roster_changes(self) -> list[dict]:
        """Compare current rosters with stored ones to detect changes."""
        import json

        roster_file = "data/rosters.json"
        current = await self.fetch_rosters()

        # Load previous rosters
        try:
            with open(roster_file, "r") as f:
                previous = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}

        changes = []
        for team_name, players in current.items():
            old_players = previous.get(team_name, [])
            if old_players and set(players) != set(old_players):
                added = set(players) - set(old_players)
                removed = set(old_players) - set(players)
                changes.append({
                    "team": team_name,
                    "added": list(added),
                    "removed": list(removed),
                    "date": datetime.utcnow().isoformat(),
                    "severity": len(added) + len(removed),  # more changes = more impact
                })
                logger.info(f"Roster change: {team_name} +{list(added)} -{list(removed)}")

        # Save current rosters
        with open(roster_file, "w") as f:
            json.dump(current, f)

        # Save changes to DB
        await self._save_alerts(changes, "roster_change")

        logger.info(f"Detected {len(changes)} roster changes")
        return changes

    # ─── HLTV News Scraping ──────────────────────────────────

    async def scrape_news(self, max_articles: int = 30) -> list[dict]:
        """Scrape recent HLTV news and classify by relevance."""
        logger.info("Scraping HLTV news...")
        soup = await self._fetch("https://www.hltv.org/news/archive")
        if not soup:
            return []

        articles = []
        for el in soup.select(".article, .newsline")[:max_articles]:
            link = el.select_one("a")
            if not link:
                continue

            title = link.text.strip()
            href = link.get("href", "")

            # Classify
            title_lower = title.lower()
            is_roster = any(kw in title_lower for kw in ROSTER_KEYWORDS)
            sentiment = self._classify_sentiment(title_lower)
            teams_mentioned = self._extract_teams(title)

            if is_roster or sentiment != "neutral" or teams_mentioned:
                articles.append({
                    "title": title,
                    "url": f"https://www.hltv.org{href}" if href.startswith("/") else href,
                    "is_roster": is_roster,
                    "sentiment": sentiment,
                    "teams": teams_mentioned,
                    "date": datetime.utcnow().isoformat(),
                })

        # Save relevant news as alerts
        roster_news = [a for a in articles if a["is_roster"]]
        await self._save_alerts(
            [{"team": t, "title": a["title"], "url": a["url"]}
             for a in roster_news for t in a["teams"]],
            "news_roster"
        )

        logger.info(f"Found {len(articles)} relevant news ({len(roster_news)} roster-related)")
        return articles

    @staticmethod
    def _classify_sentiment(text: str) -> str:
        neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
        pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
        if neg_count > pos_count:
            return "negative"
        if pos_count > neg_count:
            return "positive"
        return "neutral"

    @staticmethod
    def _extract_teams(title: str) -> list[str]:
        """Extract team names from news title (simple heuristic)."""
        # Common CS2 team names
        known_teams = [
            "Vitality", "FURIA", "MOUZ", "Falcons", "PARIVISION",
            "Natus Vincere", "NAVI", "NaVi", "Aurora", "Spirit",
            "The MongolZ", "Astralis", "FaZe", "G2", "Heroic",
            "ENCE", "Liquid", "Cloud9", "Complexity", "NRG",
            "BIG", "Ninjas in Pyjamas", "NIP", "B8", "9z",
            "TYLOO", "paiN", "Imperial", "MIBR", "Apeks",
        ]
        found = []
        for team in known_teams:
            if team.lower() in title.lower():
                found.append(team)
        return found

    # ─── Match-specific standin detection ────────────────────

    async def check_standins(self, match_url: str) -> dict:
        """Check a specific HLTV match page for standin players."""
        soup = await self._fetch(f"https://www.hltv.org{match_url}")
        if not soup:
            return {}

        standins = {"team1": [], "team2": []}

        # Look for "stand-in" labels in lineup
        for player_el in soup.select(".player"):
            name_el = player_el.select_one(".player-nick")
            standin_el = player_el.select_one(".sub-in, .stand-in")
            if name_el and standin_el:
                name = name_el.text.strip()
                # Determine which team
                parent = player_el.find_parent(class_="lineup")
                if parent:
                    standins["team1" if "team1" in str(parent.get("class", "")) else "team2"].append(name)

        return standins

    # ─── Alerts Storage ──────────────────────────────────────

    async def _save_alerts(self, alerts: list[dict], alert_type: str):
        """Save alerts to a JSON file for dashboard display."""
        import json

        alerts_file = "data/alerts.json"
        try:
            with open(alerts_file, "r") as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []

        for alert in alerts:
            alert["type"] = alert_type
            alert["timestamp"] = datetime.utcnow().isoformat()
            existing.append(alert)

        # Keep last 200 alerts
        existing = existing[-200:]

        with open(alerts_file, "w") as f:
            json.dump(existing, f, indent=2)

    # ─── Get Team Fundamental Score ──────────────────────────

    async def get_team_score(self, team_name: str) -> dict:
        """Get fundamental analysis score for a team.

        Returns dict with:
        - roster_stability: 0-1 (1 = stable, 0 = many recent changes)
        - sentiment: -1 to 1 (positive/negative news)
        - has_standin: bool
        - days_since_change: int or None
        """
        import json

        # Check alerts
        try:
            with open("data/alerts.json", "r") as f:
                alerts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            alerts = []

        team_alerts = [a for a in alerts if team_name.lower() in str(a).lower()]
        roster_alerts = [a for a in team_alerts if a.get("type") in ("roster_change", "news_roster")]
        sentiment_alerts = [a for a in team_alerts if "sentiment" in a]

        # Roster stability
        days_since_change = None
        if roster_alerts:
            latest = max(roster_alerts, key=lambda a: a.get("timestamp", ""))
            try:
                change_date = datetime.fromisoformat(latest["timestamp"])
                days_since_change = (datetime.utcnow() - change_date).days
            except (ValueError, KeyError):
                pass

        roster_stability = 1.0
        if days_since_change is not None:
            if days_since_change < 7:
                roster_stability = 0.3  # very recent change
            elif days_since_change < 30:
                roster_stability = 0.6
            elif days_since_change < 90:
                roster_stability = 0.8

        # Sentiment
        pos = sum(1 for a in sentiment_alerts if a.get("sentiment") == "positive")
        neg = sum(1 for a in sentiment_alerts if a.get("sentiment") == "negative")
        total = pos + neg
        sentiment = (pos - neg) / total if total > 0 else 0.0

        return {
            "roster_stability": roster_stability,
            "sentiment": sentiment,
            "has_standin": any("stand-in" in str(a).lower() or "standin" in str(a).lower() for a in roster_alerts),
            "days_since_change": days_since_change,
            "recent_alerts": len(team_alerts),
        }

    # ─── Full Fundamentals Update ────────────────────────────

    async def full_update(self):
        """Run complete fundamentals analysis."""
        changes = await self.detect_roster_changes()
        news = await self.scrape_news()
        await self.ps_client.aclose()

        logger.info(f"Fundamentals update: {len(changes)} roster changes, {len(news)} news items")
        return {"roster_changes": changes, "news": news}
