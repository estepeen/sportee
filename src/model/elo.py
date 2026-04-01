"""Elo/Glicko rating system for CS2 teams."""

import logging
import math
from datetime import datetime

from sqlalchemy import select, text

from src.database import async_session, Team, Match

logger = logging.getLogger(__name__)

DEFAULT_ELO = 1500
K_FACTOR = 32
# Higher K for upsets, lower for expected results


class EloSystem:
    """Dynamic Elo rating system computed from match history."""

    def __init__(self):
        self.ratings: dict[int, float] = {}  # team_id -> elo
        self.match_count: dict[int, int] = {}  # team_id -> matches played

    def expected(self, elo_a: float, elo_b: float) -> float:
        """Expected score for player A."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, winner_id: int, loser_id: int, score_margin: float = 1.0):
        """Update ratings after a match."""
        r_w = self.ratings.get(winner_id, DEFAULT_ELO)
        r_l = self.ratings.get(loser_id, DEFAULT_ELO)

        e_w = self.expected(r_w, r_l)

        # Adaptive K - higher for new teams, scales with margin
        k_w = K_FACTOR * (1.5 if self.match_count.get(winner_id, 0) < 10 else 1.0)
        k_l = K_FACTOR * (1.5 if self.match_count.get(loser_id, 0) < 10 else 1.0)

        # Margin multiplier: 2-0 sweep counts more than 2-1
        margin_mult = 1.0 + (score_margin - 1) * 0.3

        self.ratings[winner_id] = r_w + k_w * margin_mult * (1 - e_w)
        self.ratings[loser_id] = r_l - k_l * margin_mult * e_w

        self.match_count[winner_id] = self.match_count.get(winner_id, 0) + 1
        self.match_count[loser_id] = self.match_count.get(loser_id, 0) + 1

    async def compute_all(self) -> dict[int, float]:
        """Compute Elo ratings from all historical matches in chronological order."""
        logger.info("Computing Elo ratings...")

        async with async_session() as session:
            result = await session.execute(text("""
                SELECT hltv_id, team1_id, team2_id, winner_id,
                       team1_score, team2_score
                FROM matches
                WHERE is_completed = 1 AND winner_id IS NOT NULL
                ORDER BY date ASC
            """))
            matches = result.fetchall()

        self.ratings = {}
        self.match_count = {}

        for m in matches:
            _, t1_id, t2_id, winner_id, t1_score, t2_score = m

            if winner_id == t1_id:
                w_score = t1_score or 1
                l_score = t2_score or 0
            else:
                w_score = t2_score or 1
                l_score = t1_score or 0

            margin = (w_score - l_score) if (w_score and l_score) else 1.0
            loser_id = t2_id if winner_id == t1_id else t1_id

            self.update(winner_id, loser_id, score_margin=max(margin, 1.0))

        logger.info(f"Computed Elo for {len(self.ratings)} teams from {len(matches)} matches")

        # Log top teams
        top = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        async with async_session() as session:
            for tid, elo in top:
                t = await session.execute(select(Team).where(Team.hltv_id == tid))
                team = t.scalars().first()
                name = team.name if team else f"ID:{tid}"
                logger.info(f"  {name:20s}: {elo:.0f}")

        return self.ratings

    def get_elo(self, team_id: int) -> float:
        return self.ratings.get(team_id, DEFAULT_ELO)

    def predict(self, team1_id: int, team2_id: int) -> float:
        """Predict probability of team1 winning."""
        return self.expected(self.get_elo(team1_id), self.get_elo(team2_id))
