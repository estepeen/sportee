"""Odds fetcher from multiple sources for CS2 matches."""

import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select

from src.database import async_session, OddsSnapshot, Match
from config.settings import ODDSPAPI_API_KEY

logger = logging.getLogger(__name__)

ODDSPAPI_BASE = "https://api.oddspapi.com/v1"


class OddsFetcher:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)

    async def fetch_cs2_odds(self) -> list[dict]:
        """Fetch current CS2 odds from OddsPapi."""
        if not ODDSPAPI_API_KEY:
            logger.warning("ODDSPAPI_API_KEY not set, skipping odds fetch")
            return []

        try:
            resp = await self.client.get(
                f"{ODDSPAPI_BASE}/odds",
                params={
                    "apiKey": ODDSPAPI_API_KEY,
                    "sport": "esports_cs2",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", data) if isinstance(data, dict) else data
        except httpx.HTTPError as e:
            logger.error(f"OddsPapi request failed: {e}")
            return []

    async def save_odds(self, odds_data: list[dict]):
        """Parse and save odds snapshots to database."""
        saved = 0
        async with async_session() as session:
            for event in odds_data:
                teams = event.get("teams", [])
                if len(teams) < 2:
                    continue

                # Try to match with our stored matches
                team1_name = teams[0]
                team2_name = teams[1]

                for bookmaker in event.get("bookmakers", []):
                    bk_name = bookmaker.get("key", bookmaker.get("name", "unknown"))

                    for market in bookmaker.get("markets", []):
                        market_type = market.get("key", "ml")
                        outcomes = market.get("outcomes", [])

                        if len(outcomes) < 2:
                            continue

                        team1_odds = outcomes[0].get("price")
                        team2_odds = outcomes[1].get("price")
                        line = outcomes[0].get("point")

                        session.add(OddsSnapshot(
                            match_id=0,  # will be linked later
                            bookmaker=bk_name,
                            market_type=market_type,
                            team1_odds=team1_odds,
                            team2_odds=team2_odds,
                            line=line,
                            captured_at=datetime.utcnow(),
                        ))
                        saved += 1

            await session.commit()

        logger.info(f"Saved {saved} odds snapshots")

    async def get_pinnacle_odds(self, match_id: int) -> Optional[dict]:
        """Get Pinnacle odds for a specific match (sharpest line)."""
        async with async_session() as session:
            stmt = (
                select(OddsSnapshot)
                .where(
                    OddsSnapshot.match_id == match_id,
                    OddsSnapshot.bookmaker == "pinnacle",
                )
                .order_by(OddsSnapshot.captured_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            snapshot = result.scalar_one_or_none()

            if snapshot:
                return {
                    "team1_odds": snapshot.team1_odds,
                    "team2_odds": snapshot.team2_odds,
                    "market_type": snapshot.market_type,
                    "line": snapshot.line,
                    "captured_at": snapshot.captured_at,
                }
            return None

    @staticmethod
    def odds_to_prob(odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 1.0:
            return 1.0
        return 1.0 / odds

    @staticmethod
    def prob_to_odds(prob: float) -> float:
        """Convert probability to decimal odds."""
        if prob <= 0:
            return float("inf")
        return 1.0 / prob

    async def close(self):
        await self.client.aclose()
