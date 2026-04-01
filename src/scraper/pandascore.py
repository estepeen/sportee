"""PandaScore API client - primary data source for CS2 matches and teams."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select

from src.database import async_session, Team, Match, MapResult, init_db
from config.settings import PANDASCORE_API_KEY

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pandascore.co"


class PandaScoreClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
        self.token = PANDASCORE_API_KEY

    async def _get(self, endpoint: str, params: dict = None) -> list | dict | None:
        """Make authenticated GET request to PandaScore API."""
        if not self.token:
            logger.warning("PANDASCORE_API_KEY not set")
            return None

        p = params or {}
        p["token"] = self.token

        try:
            resp = await self.client.get(f"{BASE_URL}{endpoint}", params=p)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"PandaScore {resp.status_code}: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"PandaScore request failed: {e}")
            return None

    # ─── Teams ───────────────────────────────────────────────

    async def fetch_teams(self, pages: int = 5) -> list[dict]:
        """Fetch CS2 teams from PandaScore (100 per page)."""
        logger.info(f"Fetching teams ({pages} pages)...")
        all_teams = []

        for page in range(1, pages + 1):
            data = await self._get("/csgo/teams", {"per_page": 100, "page": page})
            if not data:
                break
            all_teams.extend(data)
            if len(data) < 100:
                break

        # Save to DB
        saved = 0
        async with async_session() as session:
            for td in all_teams:
                ps_id = td["id"]
                name = td["name"]

                existing = await session.execute(
                    select(Team).where(Team.name == name)
                )
                team = existing.scalars().first()
                if team:
                    team.updated_at = datetime.utcnow()
                else:
                    session.add(Team(
                        hltv_id=ps_id,  # use pandascore ID
                        name=name,
                        ranking=999,
                    ))
                    saved += 1

            await session.commit()

        logger.info(f"Fetched {len(all_teams)} teams, saved {saved} new")
        return all_teams

    # ─── Past Matches ────────────────────────────────────────

    async def fetch_past_matches(self, pages: int = 10) -> list[dict]:
        """Fetch completed CS2 matches."""
        logger.info(f"Fetching past matches ({pages} pages)...")
        all_matches = []

        for page in range(1, pages + 1):
            data = await self._get("/csgo/matches/past", {
                "per_page": 100,
                "page": page,
                "sort": "-scheduled_at",
            })
            if not data:
                break
            all_matches.extend(data)
            if len(data) < 100:
                break

        saved = 0
        async with async_session() as session:
            for md in all_matches:
                ps_id = md["id"]
                opponents = md.get("opponents", [])
                if len(opponents) < 2:
                    continue

                t1_data = opponents[0]["opponent"]
                t2_data = opponents[1]["opponent"]

                # Ensure teams exist
                for td in [t1_data, t2_data]:
                    t = await session.execute(select(Team).where(Team.name == td["name"]))
                    if not t.scalars().first():
                        session.add(Team(hltv_id=td["id"], name=td["name"], ranking=999))
                await session.flush()

                # Check if match already exists
                existing = await session.execute(
                    select(Match).where(Match.hltv_id == ps_id)
                )
                if existing.scalar_one_or_none():
                    continue

                t1 = await session.execute(select(Team).where(Team.name == t1_data["name"]))
                t2 = await session.execute(select(Team).where(Team.name == t2_data["name"]))
                t1_obj = t1.scalars().first()
                t2_obj = t2.scalars().first()
                if not t1_obj or not t2_obj:
                    continue

                # Results
                results = {r["team_id"]: r["score"] for r in md.get("results", [])}
                t1_score = results.get(t1_data["id"])
                t2_score = results.get(t2_data["id"])

                winner_id = None
                if md.get("winner_id"):
                    if md["winner_id"] == t1_data["id"]:
                        winner_id = t1_obj.hltv_id
                    elif md["winner_id"] == t2_data["id"]:
                        winner_id = t2_obj.hltv_id

                # Tournament info
                tournament = md.get("tournament", {})
                tier = tournament.get("tier", "")
                event_name = md.get("serie", {}).get("full_name") or md.get("league", {}).get("name", "")
                is_lan = tournament.get("type") == "offline"

                # Parse date
                date_str = md.get("scheduled_at") or md.get("begin_at")
                match_date = datetime.utcnow()
                if date_str:
                    try:
                        match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    except (ValueError, AttributeError):
                        pass

                session.add(Match(
                    hltv_id=ps_id,
                    date=match_date,
                    team1_id=t1_obj.hltv_id,
                    team2_id=t2_obj.hltv_id,
                    team1_score=t1_score,
                    team2_score=t2_score,
                    winner_id=winner_id,
                    best_of=md.get("number_of_games"),
                    event_name=event_name,
                    event_tier=tier,
                    is_lan=is_lan,
                    is_completed=True,
                ))
                saved += 1

                # Save map-level results from games
                games = md.get("games", [])
                for g in games:
                    if not g.get("finished"):
                        continue
                    gw = g.get("winner", {})
                    game_winner_ps_id = gw.get("id") if isinstance(gw, dict) else None
                    map_winner_id = None
                    if game_winner_ps_id == t1_data["id"]:
                        map_winner_id = t1_obj.hltv_id
                    elif game_winner_ps_id == t2_data["id"]:
                        map_winner_id = t2_obj.hltv_id

                    session.add(MapResult(
                        match_id=ps_id,
                        map_name="unknown",  # free tier doesn't give map names
                        map_number=g.get("position", 1),
                        team1_score=None,
                        team2_score=None,
                        winner_id=map_winner_id,
                    ))

            await session.commit()

        logger.info(f"Saved {saved} past matches from {len(all_matches)} fetched")
        return all_matches

    # ─── Upcoming Matches ────────────────────────────────────

    async def fetch_upcoming_matches(self) -> list[dict]:
        """Fetch upcoming CS2 matches."""
        logger.info("Fetching upcoming matches...")
        data = await self._get("/csgo/matches/upcoming", {
            "per_page": 100,
            "sort": "scheduled_at",
        })
        if not data:
            return []

        saved = 0
        async with async_session() as session:
            # First clear old upcoming that may have been played
            old_upcoming = await session.execute(
                select(Match).where(Match.is_completed == False)
            )
            for old in old_upcoming.scalars():
                await session.delete(old)
            await session.flush()

            for md in data:
                opponents = md.get("opponents", [])
                if len(opponents) < 2:
                    continue

                t1_data = opponents[0]["opponent"]
                t2_data = opponents[1]["opponent"]

                # Ensure teams exist
                for td in [t1_data, t2_data]:
                    t = await session.execute(select(Team).where(Team.name == td["name"]))
                    if not t.scalars().first():
                        session.add(Team(hltv_id=td["id"], name=td["name"], ranking=999))
                await session.flush()

                t1 = await session.execute(select(Team).where(Team.name == t1_data["name"]))
                t2 = await session.execute(select(Team).where(Team.name == t2_data["name"]))
                t1_obj = t1.scalars().first()
                t2_obj = t2.scalars().first()
                if not t1_obj or not t2_obj:
                    continue

                tournament = md.get("tournament", {})
                tier = tournament.get("tier", "")
                event_name = md.get("serie", {}).get("full_name") or md.get("league", {}).get("name", "")
                is_lan = tournament.get("type") == "offline"

                date_str = md.get("scheduled_at") or md.get("begin_at")
                match_date = datetime.utcnow()
                if date_str:
                    try:
                        match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    except (ValueError, AttributeError):
                        pass

                session.add(Match(
                    hltv_id=md["id"],
                    date=match_date,
                    team1_id=t1_obj.hltv_id,
                    team2_id=t2_obj.hltv_id,
                    best_of=md.get("number_of_games"),
                    event_name=event_name,
                    event_tier=tier,
                    is_lan=is_lan,
                    is_completed=False,
                ))
                saved += 1

            await session.commit()

        logger.info(f"Saved {saved} upcoming matches from {len(data)} fetched")
        return data

    # ─── Running Matches (live) ──────────────────────────────

    async def fetch_running_matches(self) -> list[dict]:
        """Fetch currently live CS2 matches."""
        data = await self._get("/csgo/matches/running", {"per_page": 50})
        return data or []

    # ─── Full Sync ───────────────────────────────────────────

    async def full_sync(self, deep: bool = False):
        """Full data sync from PandaScore."""
        await init_db()

        if deep:
            await self.fetch_teams(pages=10)  # ~1000 teams
            await self.fetch_past_matches(pages=20)  # ~2000 matches
        else:
            await self.fetch_teams(pages=2)
            await self.fetch_past_matches(pages=3)

        await self.fetch_upcoming_matches()
        logger.info("PandaScore sync completed")

    async def close(self):
        await self.client.aclose()
