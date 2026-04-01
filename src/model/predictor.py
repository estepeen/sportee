"""CS2 match prediction model using team map stats and contextual features."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select, func

from src.database import (
    async_session, Team, Match, MapResult, TeamMapStats, OddsSnapshot
)
from config.settings import MIN_MATCHES_FOR_PREDICTION

logger = logging.getLogger(__name__)

CS2_MAP_POOL = ["mirage", "inferno", "nuke", "overpass", "anubis", "ancient", "dust2", "vertigo", "train"]


@dataclass
class MatchPrediction:
    match_id: int
    team1_name: str
    team2_name: str
    team1_ml_prob: float  # moneyline probability
    team2_ml_prob: float
    over_2_5_prob: float  # over 2.5 maps (Bo3)
    team1_map_handicap_plus_1_5_prob: float  # underdog wins at least 1 map
    team2_map_handicap_plus_1_5_prob: float
    best_of: int
    confidence: float  # 0-1, how confident we are
    features: dict


class CS2Predictor:
    """Predicts CS2 match outcomes based on team map stats and context."""

    async def get_team_features(self, team_id: int, months: int = 3) -> dict:
        """Extract features for a team from stored stats."""
        async with async_session() as session:
            # Team ranking
            team_result = await session.execute(
                select(Team).where(Team.hltv_id == team_id)
            )
            team = team_result.scalar_one_or_none()
            ranking = team.ranking if team else 999

            # Map-specific stats
            map_stats_result = await session.execute(
                select(TeamMapStats).where(
                    TeamMapStats.team_id == team_id,
                    TeamMapStats.period_months == months,
                )
            )
            map_stats = {
                ms.map_name: {
                    "winrate": ms.wins / ms.matches_played if ms.matches_played > 0 else 0.5,
                    "played": ms.matches_played,
                    "ct_wr": ms.ct_winrate or 0.5,
                    "t_wr": ms.t_winrate or 0.5,
                    "avg_rounds": ms.avg_rounds_won or 12.0,
                }
                for ms in map_stats_result.scalars()
            }

            # Recent form (last 10 matches)
            cutoff = datetime.utcnow() - timedelta(days=30)
            recent_result = await session.execute(
                select(Match)
                .where(
                    Match.is_completed == True,
                    Match.date >= cutoff,
                    (Match.team1_id == team_id) | (Match.team2_id == team_id),
                )
                .order_by(Match.date.desc())
                .limit(10)
            )
            recent_matches = recent_result.scalars().all()

            wins = sum(1 for m in recent_matches if m.winner_id == team_id)
            form = wins / len(recent_matches) if recent_matches else 0.5

            # LAN vs Online split
            lan_matches = [m for m in recent_matches if m.is_lan]
            lan_form = (
                sum(1 for m in lan_matches if m.winner_id == team_id) / len(lan_matches)
                if lan_matches else form
            )

            return {
                "ranking": ranking,
                "map_stats": map_stats,
                "recent_form": form,
                "recent_matches_count": len(recent_matches),
                "lan_form": lan_form,
            }

    async def predict_map(
        self,
        team1_stats: dict,
        team2_stats: dict,
        map_name: str,
    ) -> float:
        """Predict probability of team1 winning a specific map."""
        t1_map = team1_stats["map_stats"].get(map_name, {})
        t2_map = team2_stats["map_stats"].get(map_name, {})

        t1_wr = t1_map.get("winrate", 0.5)
        t2_wr = t2_map.get("winrate", 0.5)
        t1_played = t1_map.get("played", 0)
        t2_played = t2_map.get("played", 0)

        # Weight by sample size - less confidence with fewer games
        t1_weight = min(t1_played / MIN_MATCHES_FOR_PREDICTION, 1.0)
        t2_weight = min(t2_played / MIN_MATCHES_FOR_PREDICTION, 1.0)

        # Regress to mean based on sample size
        t1_adj_wr = t1_wr * t1_weight + 0.5 * (1 - t1_weight)
        t2_adj_wr = t2_wr * t2_weight + 0.5 * (1 - t2_weight)

        # Ranking factor
        rank_diff = team2_stats["ranking"] - team1_stats["ranking"]
        rank_factor = np.clip(rank_diff / 50, -0.15, 0.15)

        # Form factor
        form_diff = team1_stats["recent_form"] - team2_stats["recent_form"]
        form_factor = form_diff * 0.1

        # Combined probability using log-odds
        t1_log_odds = np.log(t1_adj_wr / (1 - t1_adj_wr + 1e-8))
        t2_log_odds = np.log(t2_adj_wr / (1 - t2_adj_wr + 1e-8))

        combined_log_odds = (t1_log_odds - t2_log_odds) / 2 + rank_factor + form_factor
        prob = 1 / (1 + np.exp(-combined_log_odds))

        return float(np.clip(prob, 0.05, 0.95))

    async def predict_match(self, match_id: int) -> MatchPrediction | None:
        """Generate full prediction for an upcoming match."""
        async with async_session() as session:
            match_result = await session.execute(
                select(Match).where(Match.hltv_id == match_id)
            )
            match = match_result.scalar_one_or_none()
            if not match:
                logger.warning(f"Match {match_id} not found")
                return None

            # Get team names
            t1_result = await session.execute(
                select(Team).where(Team.hltv_id == match.team1_id)
            )
            t2_result = await session.execute(
                select(Team).where(Team.hltv_id == match.team2_id)
            )
            t1 = t1_result.scalar_one_or_none()
            t2 = t2_result.scalar_one_or_none()

        team1_stats = await self.get_team_features(match.team1_id)
        team2_stats = await self.get_team_features(match.team2_id)

        # Predict each map in the pool
        map_probs = {}
        for map_name in CS2_MAP_POOL:
            prob = await self.predict_map(team1_stats, team2_stats, map_name)
            map_probs[map_name] = prob

        best_of = match.best_of or 3

        # Moneyline: simulate Bo3/Bo5
        if best_of == 1:
            # Bo1 - average across likely maps
            avg_prob = np.mean(list(map_probs.values()))
            t1_ml = avg_prob
        elif best_of == 3:
            t1_ml = self._bo3_win_prob(map_probs, team1_stats, team2_stats)
        elif best_of == 5:
            t1_ml = self._bo5_win_prob(map_probs, team1_stats, team2_stats)
        else:
            t1_ml = np.mean(list(map_probs.values()))

        t2_ml = 1 - t1_ml

        # Over 2.5 maps (Bo3 goes to 3 maps)
        over_2_5 = self._over_2_5_prob(map_probs, team1_stats, team2_stats) if best_of == 3 else 0.0

        # Map handicap +1.5 (underdog wins at least 1 map in Bo3)
        # This is 1 - P(2-0 sweep)
        if best_of >= 3:
            t1_handicap = 1 - self._sweep_prob(map_probs, team2_stats, team1_stats)  # T1 underdog +1.5
            t2_handicap = 1 - self._sweep_prob(map_probs, team1_stats, team2_stats)  # T2 underdog +1.5
        else:
            t1_handicap = t1_ml
            t2_handicap = t2_ml

        # Confidence based on data quality
        t1_data = sum(
            1 for m in team1_stats["map_stats"].values()
            if m.get("played", 0) >= MIN_MATCHES_FOR_PREDICTION
        )
        t2_data = sum(
            1 for m in team2_stats["map_stats"].values()
            if m.get("played", 0) >= MIN_MATCHES_FOR_PREDICTION
        )
        confidence = min((t1_data + t2_data) / (len(CS2_MAP_POOL) * 2), 1.0)

        return MatchPrediction(
            match_id=match_id,
            team1_name=t1.name if t1 else "Unknown",
            team2_name=t2.name if t2 else "Unknown",
            team1_ml_prob=round(t1_ml, 4),
            team2_ml_prob=round(t2_ml, 4),
            over_2_5_prob=round(over_2_5, 4),
            team1_map_handicap_plus_1_5_prob=round(t1_handicap, 4),
            team2_map_handicap_plus_1_5_prob=round(t2_handicap, 4),
            best_of=best_of,
            confidence=round(confidence, 4),
            features={
                "map_probs": map_probs,
                "team1_ranking": team1_stats["ranking"],
                "team2_ranking": team2_stats["ranking"],
                "team1_form": team1_stats["recent_form"],
                "team2_form": team2_stats["recent_form"],
            },
        )

    def _bo3_win_prob(self, map_probs: dict, t1_stats: dict, t2_stats: dict) -> float:
        """Estimate Bo3 win probability for team1."""
        # Use top maps weighted by pick likelihood
        probs = sorted(map_probs.values(), reverse=True)
        if len(probs) < 3:
            return np.mean(probs)

        # Team1's best map, team2's best map, decider
        t1_pick = probs[0]           # team1 picks their best
        t2_pick = 1 - probs[-1]      # team2 picks team1's worst (inverted)
        t2_pick = 1 - t2_pick        # back to team1 perspective... simplify:
        decider = probs[len(probs) // 2]  # middle map

        # P(win Bo3) = P(2-0) + P(2-1)
        p_2_0 = t1_pick * decider
        p_2_1 = (t1_pick * (1 - decider) * decider +
                 (1 - t1_pick) * decider * decider)

        # Simplified: use average of top/mid/bottom
        avg = np.mean([probs[0], probs[len(probs) // 2], probs[-1]])
        # Bo3 amplifies skill: better team wins more often
        bo3_prob = 3 * avg**2 - 2 * avg**3

        return float(bo3_prob)

    def _bo5_win_prob(self, map_probs: dict, t1_stats: dict, t2_stats: dict) -> float:
        """Estimate Bo5 win probability for team1."""
        avg = np.mean(list(map_probs.values()))
        # Bo5 amplifies even more
        p = avg
        bo5_prob = (
            p**3 * (1 + 3*(1-p) + 6*(1-p)**2)
        )
        return float(np.clip(bo5_prob, 0.05, 0.95))

    def _over_2_5_prob(self, map_probs: dict, t1_stats: dict, t2_stats: dict) -> float:
        """Probability that a Bo3 goes to 3 maps (over 2.5)."""
        probs = list(map_probs.values())
        avg = np.mean(probs)
        # P(3 maps) = 1 - P(2-0) - P(0-2)
        # Approximation: closer to 50% = more likely to go 3 maps
        p_2_0 = avg * avg  # rough P(team1 sweeps)
        p_0_2 = (1 - avg) * (1 - avg)  # rough P(team2 sweeps)
        return float(1 - p_2_0 - p_0_2)

    def _sweep_prob(self, map_probs: dict, fav_stats: dict, dog_stats: dict) -> float:
        """Probability of a 2-0 sweep by the favorite."""
        probs = sorted(map_probs.values(), reverse=True)
        if len(probs) < 2:
            return 0.5
        # Favorite wins their pick AND the decider/opponent's pick
        return float(probs[0] * probs[1])
