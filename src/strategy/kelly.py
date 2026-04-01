"""Kelly criterion and value betting strategy."""

import logging
from dataclasses import dataclass

from src.model.predictor import MatchPrediction
from src.odds.fetcher import OddsFetcher
from config.settings import MIN_EDGE_PERCENT, KELLY_FRACTION, MAX_BET_PERCENT

logger = logging.getLogger(__name__)


@dataclass
class BetSignal:
    match_id: int
    team1_name: str
    team2_name: str
    market: str         # "ml", "map_handicap_+1.5", "over_2.5"
    side: str           # "team1", "team2", "over", "under"
    our_prob: float     # our estimated probability
    bookmaker_prob: float  # implied probability from odds
    odds: float         # decimal odds we can get
    edge: float         # our_prob - bookmaker_prob
    kelly_stake: float  # optimal stake as % of bankroll
    confidence: float


class ValueFinder:
    """Finds value bets by comparing model probabilities with bookmaker odds."""

    def __init__(self):
        self.odds_fetcher = OddsFetcher()

    def find_value_bets(
        self,
        prediction: MatchPrediction,
        odds: dict,
    ) -> list[BetSignal]:
        """Compare our prediction with available odds to find value."""
        signals = []

        markets = self._build_markets(prediction, odds)

        for market in markets:
            edge = market["our_prob"] - market["bk_prob"]
            edge_pct = edge * 100

            if edge_pct < MIN_EDGE_PERCENT:
                continue

            kelly = self._kelly_criterion(market["our_prob"], market["odds"])

            if kelly <= 0:
                continue

            # Apply fractional Kelly and cap
            stake = min(kelly * KELLY_FRACTION, MAX_BET_PERCENT / 100)

            signals.append(BetSignal(
                match_id=prediction.match_id,
                team1_name=prediction.team1_name,
                team2_name=prediction.team2_name,
                market=market["type"],
                side=market["side"],
                our_prob=round(market["our_prob"], 4),
                bookmaker_prob=round(market["bk_prob"], 4),
                odds=market["odds"],
                edge=round(edge, 4),
                kelly_stake=round(stake, 4),
                confidence=prediction.confidence,
            ))

        # Sort by edge * confidence
        signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        return signals

    def _build_markets(self, pred: MatchPrediction, odds: dict) -> list[dict]:
        """Build all possible markets to evaluate."""
        markets = []

        # Moneyline
        if "ml_team1" in odds:
            markets.append({
                "type": "ml",
                "side": "team1",
                "our_prob": pred.team1_ml_prob,
                "odds": odds["ml_team1"],
                "bk_prob": OddsFetcher.odds_to_prob(odds["ml_team1"]),
            })
        if "ml_team2" in odds:
            markets.append({
                "type": "ml",
                "side": "team2",
                "our_prob": pred.team2_ml_prob,
                "odds": odds["ml_team2"],
                "bk_prob": OddsFetcher.odds_to_prob(odds["ml_team2"]),
            })

        # Map handicap +1.5 (underdog wins at least 1 map)
        if "handicap_team1_plus_1_5" in odds:
            markets.append({
                "type": "map_handicap_+1.5",
                "side": "team1",
                "our_prob": pred.team1_map_handicap_plus_1_5_prob,
                "odds": odds["handicap_team1_plus_1_5"],
                "bk_prob": OddsFetcher.odds_to_prob(odds["handicap_team1_plus_1_5"]),
            })
        if "handicap_team2_plus_1_5" in odds:
            markets.append({
                "type": "map_handicap_+1.5",
                "side": "team2",
                "our_prob": pred.team2_map_handicap_plus_1_5_prob,
                "odds": odds["handicap_team2_plus_1_5"],
                "bk_prob": OddsFetcher.odds_to_prob(odds["handicap_team2_plus_1_5"]),
            })

        # Over/Under 2.5 maps
        if "over_2_5" in odds and pred.best_of == 3:
            markets.append({
                "type": "over_2.5",
                "side": "over",
                "our_prob": pred.over_2_5_prob,
                "odds": odds["over_2_5"],
                "bk_prob": OddsFetcher.odds_to_prob(odds["over_2_5"]),
            })
        if "under_2_5" in odds and pred.best_of == 3:
            markets.append({
                "type": "under_2.5",
                "side": "under",
                "our_prob": 1 - pred.over_2_5_prob,
                "odds": odds["under_2_5"],
                "bk_prob": OddsFetcher.odds_to_prob(odds["under_2_5"]),
            })

        return markets

    @staticmethod
    def _kelly_criterion(prob: float, odds: float) -> float:
        """Calculate Kelly criterion stake.

        f* = (bp - q) / b
        where b = odds - 1, p = our probability, q = 1 - p
        """
        b = odds - 1
        if b <= 0:
            return 0.0
        q = 1 - prob
        f = (b * prob - q) / b
        return max(f, 0.0)

    def format_signal(self, signal: BetSignal) -> str:
        """Format a bet signal for display."""
        return (
            f"{'=' * 50}\n"
            f"{signal.team1_name} vs {signal.team2_name}\n"
            f"Market: {signal.market} | Side: {signal.side}\n"
            f"Our prob: {signal.our_prob:.1%} | BK prob: {signal.bookmaker_prob:.1%}\n"
            f"Edge: {signal.edge:.1%} | Odds: {signal.odds:.2f}\n"
            f"Kelly stake: {signal.kelly_stake:.2%} of bankroll\n"
            f"Confidence: {signal.confidence:.1%}\n"
            f"{'=' * 50}"
        )
