"""Per-map Elo rating system. Each team has separate Elo for each map."""

import logging
from collections import defaultdict

from sqlalchemy import text

from src.database import async_session

logger = logging.getLogger(__name__)

DEFAULT_ELO = 1500
K_FACTOR = 40  # Higher K for map-specific (less data per map)


class MapEloSystem:
    """Tracks separate Elo rating for each team on each map."""

    def __init__(self):
        # {map_name: {team_id: elo}}
        self.ratings: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(lambda: DEFAULT_ELO))
        self.match_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def expected(self, elo_a: float, elo_b: float) -> float:
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, map_name: str, winner_id: int, loser_id: int, round_diff: int = 0):
        """Update map-specific Elo after a map result."""
        r_w = self.ratings[map_name][winner_id]
        r_l = self.ratings[map_name][loser_id]

        e_w = self.expected(r_w, r_l)

        # Adaptive K for new teams on this map
        k_w = K_FACTOR * (1.5 if self.match_counts[map_name][winner_id] < 5 else 1.0)
        k_l = K_FACTOR * (1.5 if self.match_counts[map_name][loser_id] < 5 else 1.0)

        # Margin bonus: bigger round diff = more Elo change
        margin = 1.0 + min(round_diff, 10) * 0.05 if round_diff > 0 else 1.0

        self.ratings[map_name][winner_id] = r_w + k_w * margin * (1 - e_w)
        self.ratings[map_name][loser_id] = r_l - k_l * margin * e_w

        self.match_counts[map_name][winner_id] += 1
        self.match_counts[map_name][loser_id] += 1

    async def compute_all(self) -> dict[str, dict[int, float]]:
        """Compute per-map Elo from all historical map results."""
        logger.info("Computing per-map Elo ratings...")

        async with async_session() as session:
            result = await session.execute(text("""
                SELECT mr.map_name, mr.team1_score, mr.team2_score, mr.winner_id,
                       m.team1_id, m.team2_id
                FROM map_results mr
                JOIN matches m ON mr.match_id = m.hltv_id
                WHERE m.is_completed = 1 AND mr.winner_id IS NOT NULL
                    AND mr.map_name != 'unknown'
                ORDER BY m.date ASC
            """))
            maps = result.fetchall()

        self.ratings = defaultdict(lambda: defaultdict(lambda: DEFAULT_ELO))
        self.match_counts = defaultdict(lambda: defaultdict(int))

        for mr in maps:
            map_name, t1_score, t2_score, winner_id, t1_id, t2_id = mr

            if not winner_id or not map_name:
                continue

            loser_id = t2_id if winner_id == t1_id else t1_id
            s1 = t1_score or 0
            s2 = t2_score or 0
            round_diff = abs(s1 - s2)

            self.update(map_name, winner_id, loser_id, round_diff)

        total = sum(len(teams) for teams in self.ratings.values())
        logger.info(f"Computed map Elo: {len(self.ratings)} maps, {total} team-map ratings")

        # Log top teams per map
        for map_name in sorted(self.ratings.keys()):
            top = sorted(self.ratings[map_name].items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{tid}:{elo:.0f}" for tid, elo in top)
            logger.info(f"  {map_name:12s}: {top_str} ({self.match_counts[map_name].__len__()} teams)")

        return dict(self.ratings)

    def get_elo(self, map_name: str, team_id: int) -> float:
        return self.ratings.get(map_name, {}).get(team_id, DEFAULT_ELO)

    def get_team_map_profile(self, team_id: int) -> dict[str, dict]:
        """Get all map ratings for a team."""
        profile = {}
        for map_name, teams in self.ratings.items():
            if team_id in teams:
                profile[map_name] = {
                    "elo": round(teams[team_id]),
                    "matches": self.match_counts[map_name].get(team_id, 0),
                }
        return profile

    def get_best_map(self, team_id: int) -> tuple[str, float]:
        """Get team's best map (highest Elo)."""
        best_map = ""
        best_elo = 0
        for map_name, teams in self.ratings.items():
            elo = teams.get(team_id, DEFAULT_ELO)
            if elo > best_elo and self.match_counts[map_name].get(team_id, 0) >= 3:
                best_elo = elo
                best_map = map_name
        return best_map, best_elo

    def get_worst_map(self, team_id: int) -> tuple[str, float]:
        """Get team's worst map (lowest Elo)."""
        worst_map = ""
        worst_elo = 9999
        for map_name, teams in self.ratings.items():
            elo = teams.get(team_id, DEFAULT_ELO)
            if elo < worst_elo and self.match_counts[map_name].get(team_id, 0) >= 3:
                worst_elo = elo
                worst_map = map_name
        return worst_map, worst_elo

    def get_map_pool_depth(self, team_id: int, threshold_elo: float = 1520) -> int:
        """How many maps does the team play well on?"""
        return sum(
            1 for map_name, teams in self.ratings.items()
            if teams.get(team_id, DEFAULT_ELO) >= threshold_elo
            and self.match_counts[map_name].get(team_id, 0) >= 3
        )

    def serialize(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            map_name: {str(tid): round(elo, 1) for tid, elo in teams.items()}
            for map_name, teams in self.ratings.items()
        }

    def deserialize(self, data: dict):
        """Load from serialized dict."""
        self.ratings = defaultdict(lambda: defaultdict(lambda: DEFAULT_ELO))
        for map_name, teams in data.items():
            for tid_str, elo in teams.items():
                self.ratings[map_name][int(tid_str)] = elo
