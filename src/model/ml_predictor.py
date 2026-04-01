"""ML-based CS2 match prediction model using LightGBM + Elo ratings."""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sqlalchemy import select, text

from src.database import async_session, Team, Match
from src.model.elo import EloSystem
from src.model.map_elo import MapEloSystem

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "model.pkl"


@dataclass
class MLPrediction:
    match_id: int
    team1_name: str
    team2_name: str
    team1_rank: int
    team2_rank: int
    team1_elo: float
    team2_elo: float
    team1_win_prob: float
    team2_win_prob: float
    over_2_5_prob: float
    team1_plus_1_5_prob: float
    team2_plus_1_5_prob: float
    best_of: int
    confidence: float
    event: str
    tier: str
    is_lan: bool
    # Implied odds from our model
    team1_odds: float
    team2_odds: float
    over_2_5_odds: float
    team1_plus_1_5_odds: float
    team2_plus_1_5_odds: float
    features: dict = field(default_factory=dict)


def prob_to_odds(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0.01:
        return 99.0
    return round(1 / prob, 2)


class MLPredictor:
    """LightGBM + Elo match outcome predictor."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.elo = EloSystem()
        self.map_elo = MapEloSystem()

    async def _build_dataset(self) -> pd.DataFrame:
        """Build training dataset with Elo ratings computed chronologically."""
        logger.info("Building training dataset with Elo...")

        async with async_session() as session:
            result = await session.execute(text("""
                SELECT m.hltv_id, m.date, m.team1_id, m.team2_id,
                       m.team1_score, m.team2_score, m.winner_id,
                       m.best_of, m.event_tier, m.is_lan,
                       t1.name as t1_name, t1.ranking as t1_rank,
                       t2.name as t2_name, t2.ranking as t2_rank
                FROM matches m
                JOIN teams t1 ON m.team1_id = t1.hltv_id
                JOIN teams t2 ON m.team2_id = t2.hltv_id
                WHERE m.is_completed = 1 AND m.winner_id IS NOT NULL
                ORDER BY m.date ASC
            """))
            matches = result.fetchall()

        elo = EloSystem()
        team_history: dict[int, list] = {}
        rows = []

        for m in matches:
            mid, date, t1_id, t2_id, t1_score, t2_score, winner_id, best_of, tier, is_lan, \
                t1_name, t1_rank, t2_name, t2_rank = m

            t1_rank = t1_rank or 999
            t2_rank = t2_rank or 999
            t1_won = 1 if winner_id == t1_id else 0

            # Elo BEFORE this match (prediction time)
            t1_elo = elo.get_elo(t1_id)
            t2_elo = elo.get_elo(t2_id)
            elo_diff = t1_elo - t2_elo
            elo_expected = elo.predict(t1_id, t2_id)

            # Form features
            t1_hist = team_history.get(t1_id, [])
            t2_hist = team_history.get(t2_id, [])

            t1_form5 = self._weighted_form(t1_hist, 5)
            t2_form5 = self._weighted_form(t2_hist, 5)
            t1_form10 = self._weighted_form(t1_hist, 10)
            t2_form10 = self._weighted_form(t2_hist, 10)

            # Opponent-adjusted form: weighted by opponent Elo
            t1_adj_form = self._adj_form(t1_hist, 10)
            t2_adj_form = self._adj_form(t2_hist, 10)

            # H2H
            h2h = self._h2h(t1_hist, t2_id)

            # Streaks
            t1_streak = self._streak(t1_hist)
            t2_streak = self._streak(t2_hist)

            # Advanced features
            from src.model.advanced_features import (
                compute_multi_window_streaks, compute_opponent_adjusted_stats,
                compute_score_patterns,
            )

            t1_streaks = compute_multi_window_streaks(t1_hist)
            t2_streaks = compute_multi_window_streaks(t2_hist)
            t1_opp_adj = compute_opponent_adjusted_stats(t1_hist)
            t2_opp_adj = compute_opponent_adjusted_stats(t2_hist)
            t1_scores = compute_score_patterns(t1_hist)
            t2_scores = compute_score_patterns(t2_hist)

            tier_map = {"s": 5, "a": 4, "b": 3, "c": 2, "d": 1}
            tier_val = tier_map.get(str(tier).lower(), 0) if tier else 0

            rows.append({
                "match_id": mid,
                "date": date,
                # Elo features
                "t1_elo": t1_elo,
                "t2_elo": t2_elo,
                "elo_diff": elo_diff,
                "elo_expected": elo_expected,
                # Ranking features
                "rank_diff": t2_rank - t1_rank,
                "t1_rank": min(t1_rank, 500),
                "t2_rank": min(t2_rank, 500),
                "rank_ratio": t1_rank / (t2_rank + 1),
                # Context
                "tier": tier_val,
                "best_of": best_of or 3,
                "is_lan": 1 if is_lan else 0,
                # Multi-window form
                "t1_form3": t1_streaks["form_3"],
                "t2_form3": t2_streaks["form_3"],
                "t1_form5": t1_streaks["form_5"],
                "t2_form5": t2_streaks["form_5"],
                "form_diff5": t1_streaks["form_5"] - t2_streaks["form_5"],
                "t1_form10": t1_streaks["form_10"],
                "t2_form10": t2_streaks["form_10"],
                "form_diff10": t1_streaks["form_10"] - t2_streaks["form_10"],
                "t1_form20": t1_streaks["form_20"],
                "t2_form20": t2_streaks["form_20"],
                # Momentum (short vs long form)
                "t1_momentum": t1_streaks["momentum"],
                "t2_momentum": t2_streaks["momentum"],
                "momentum_diff": t1_streaks["momentum"] - t2_streaks["momentum"],
                # Opponent-adjusted
                "t1_adj_wr": t1_opp_adj["adj_winrate"],
                "t2_adj_wr": t2_opp_adj["adj_winrate"],
                "adj_wr_diff": t1_opp_adj["adj_winrate"] - t2_opp_adj["adj_winrate"],
                "t1_upset_potential": t1_opp_adj["upset_potential"],
                "t2_upset_potential": t2_opp_adj["upset_potential"],
                # Experience
                "t1_exp": min(len(t1_hist), 50),
                "t2_exp": min(len(t2_hist), 50),
                "exp_diff": len(t1_hist) - len(t2_hist),
                # H2H
                "h2h_wr": h2h["wr"],
                "h2h_total": h2h["total"],
                # Streaks
                "t1_streak": t1_streaks["streak"],
                "t2_streak": t2_streaks["streak"],
                "streak_diff": t1_streaks["streak"] - t2_streaks["streak"],
                "t1_max_streak": t1_streaks["max_streak_10"],
                "t2_max_streak": t2_streaks["max_streak_10"],
                # Score patterns
                "t1_sweep_rate": t1_scores["sweep_rate"],
                "t2_sweep_rate": t2_scores["sweep_rate"],
                "t1_close_rate": t1_scores["close_match_rate"],
                "t2_close_rate": t2_scores["close_match_rate"],
                "t1_avg_map_diff": t1_scores["avg_map_diff"],
                "t2_avg_map_diff": t2_scores["avg_map_diff"],
                # Target
                "target": t1_won,
            })

            # Update Elo AFTER match
            loser_id = t2_id if winner_id == t1_id else t1_id
            margin = abs((t1_score or 0) - (t2_score or 0))
            elo.update(winner_id, loser_id, score_margin=max(margin, 1.0))

            # Update histories
            for tid, won, opp_id, opp_elo in [
                (t1_id, t1_won, t2_id, t2_elo),
                (t2_id, 1 - t1_won, t1_id, t1_elo),
            ]:
                if tid not in team_history:
                    team_history[tid] = []
                team_history[tid].append({
                    "won": won,
                    "opponent_id": opp_id,
                    "opponent_elo": opp_elo,
                    "maps_won": (t1_score if tid == t1_id else t2_score) or 0,
                    "maps_lost": (t2_score if tid == t1_id else t1_score) or 0,
                })

        # Store final Elo for predictions
        self.elo = elo

        df = pd.DataFrame(rows)
        logger.info(f"Built dataset: {len(df)} matches, {len(df.columns)} columns")
        return df

    @staticmethod
    def _weighted_form(history: list, n: int) -> float:
        """Recency-weighted win rate. Recent matches count more."""
        if not history:
            return 0.5
        recent = history[-n:]
        weights = [1.0 + 0.5 * i / len(recent) for i in range(len(recent))]
        total = sum(w * h["won"] for w, h in zip(weights, recent))
        return total / sum(weights)

    @staticmethod
    def _adj_form(history: list, n: int) -> float:
        """Opponent-strength-adjusted form. Wins vs strong opponents count more."""
        if not history:
            return 0.5
        recent = history[-n:]
        total_weight = 0
        total_score = 0
        for h in recent:
            opp_elo = h.get("opponent_elo", 1500)
            weight = opp_elo / 1500  # stronger opponent = higher weight
            total_weight += weight
            total_score += weight * h["won"]
        return total_score / total_weight if total_weight > 0 else 0.5

    @staticmethod
    def _h2h(history: list, opponent_id: int) -> dict:
        """Head-to-head record against specific opponent."""
        h2h_matches = [h for h in history if h["opponent_id"] == opponent_id]
        total = len(h2h_matches)
        wins = sum(h["won"] for h in h2h_matches)
        return {"wr": wins / total if total > 0 else 0.5, "total": total}

    @staticmethod
    def _streak(history: list) -> int:
        """Current win/loss streak."""
        if not history:
            return 0
        streak = 0
        last = history[-1]["won"]
        for h in reversed(history):
            if h["won"] == last:
                streak += 1
            else:
                break
        return streak if last else -streak

    @staticmethod
    def _sweep_rate(history: list) -> float:
        """Rate of 2-0 sweeps in recent matches."""
        if not history:
            return 0.0
        recent = history[-10:]
        sweeps = sum(1 for h in recent if h["won"] and h["maps_lost"] == 0)
        wins = sum(1 for h in recent if h["won"])
        return sweeps / wins if wins > 0 else 0.0

    async def train(self):
        """Train LightGBM model on historical data."""
        # Compute per-map Elo
        await self.map_elo.compute_all()

        df = await self._build_dataset()

        if len(df) < 50:
            logger.warning(f"Not enough data ({len(df)} matches). Need 50+.")
            return

        feature_cols = [c for c in df.columns if c not in ("match_id", "date", "target")]
        X = df[feature_cols]
        y = df["target"]
        self.feature_names = feature_cols

        # Time-series cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                num_leaves=20,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.3,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        logger.info(f"CV Accuracy: {avg_acc:.1%} (splits: {[f'{a:.1%}' for a in accuracies]})")

        # Train final model
        self.model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            num_leaves=20, min_child_samples=15, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0.3, verbose=-1,
        )
        self.model.fit(X, y)

        # Feature importance
        importances = sorted(zip(feature_cols, self.model.feature_importances_), key=lambda x: x[1], reverse=True)
        logger.info("Feature importance (top 10):")
        for name, imp in importances[:10]:
            logger.info(f"  {name:20s}: {imp}")

        # Save
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model": self.model,
                "features": self.feature_names,
                "accuracy": avg_acc,
                "elo_ratings": dict(self.elo.ratings),
                "map_elo": self.map_elo.serialize(),
            }, f)
        logger.info(f"Model saved ({avg_acc:.1%} accuracy, {len(self.elo.ratings)} team Elo ratings)")

    def load(self) -> bool:
        """Load trained model from disk."""
        if not MODEL_PATH.exists():
            return False
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.elo.ratings = data.get("elo_ratings", {})
        if data.get("map_elo"):
            self.map_elo.deserialize(data["map_elo"])
        logger.info(f"Model loaded (accuracy: {data.get('accuracy', 0):.1%}, {len(self.elo.ratings)} Elo ratings)")
        return True

    async def predict_match(self, match_id: int) -> MLPrediction | None:
        """Predict outcome of a match."""
        if not self.model and not self.load():
            return None

        async with async_session() as session:
            match_result = await session.execute(select(Match).where(Match.hltv_id == match_id))
            match = match_result.scalar_one_or_none()
            if not match:
                return None

            t1r = await session.execute(select(Team).where(Team.hltv_id == match.team1_id))
            t2r = await session.execute(select(Team).where(Team.hltv_id == match.team2_id))
            t1 = t1r.scalar_one_or_none()
            t2 = t2r.scalar_one_or_none()
            if not t1 or not t2:
                return None

            # Get histories
            t1_hist_r = await session.execute(text(
                "SELECT winner_id, team1_id, team2_id, team1_score, team2_score "
                "FROM matches WHERE is_completed=1 AND (team1_id = :tid OR team2_id = :tid) "
                "ORDER BY date DESC LIMIT 20"
            ), {"tid": t1.hltv_id})
            t2_hist_r = await session.execute(text(
                "SELECT winner_id, team1_id, team2_id, team1_score, team2_score "
                "FROM matches WHERE is_completed=1 AND (team1_id = :tid OR team2_id = :tid) "
                "ORDER BY date DESC LIMIT 20"
            ), {"tid": t2.hltv_id})

            t1_raw = t1_hist_r.fetchall()
            t2_raw = t2_hist_r.fetchall()

        # Convert to history format
        def to_hist(raw, tid):
            hist = []
            for h in reversed(raw):  # chronological order
                is_t1 = h[1] == tid
                opp_id = h[2] if is_t1 else h[1]
                won = 1 if h[0] == tid else 0
                mw = (h[3] if is_t1 else h[4]) or 0
                ml = (h[4] if is_t1 else h[3]) or 0
                hist.append({"won": won, "opponent_id": opp_id,
                             "opponent_elo": self.elo.get_elo(opp_id),
                             "maps_won": mw, "maps_lost": ml})
            return hist

        t1_hist = to_hist(t1_raw, t1.hltv_id)
        t2_hist = to_hist(t2_raw, t2.hltv_id)

        t1_rank = t1.ranking or 999
        t2_rank = t2.ranking or 999
        t1_elo = self.elo.get_elo(t1.hltv_id)
        t2_elo = self.elo.get_elo(t2.hltv_id)

        from src.model.advanced_features import (
            compute_multi_window_streaks, compute_opponent_adjusted_stats,
            compute_score_patterns,
        )

        h2h = self._h2h(t1_hist, t2.hltv_id)
        tier_map = {"s": 5, "a": 4, "b": 3, "c": 2, "d": 1}

        t1_streaks = compute_multi_window_streaks(t1_hist)
        t2_streaks = compute_multi_window_streaks(t2_hist)
        t1_opp = compute_opponent_adjusted_stats(t1_hist)
        t2_opp = compute_opponent_adjusted_stats(t2_hist)
        t1_scores = compute_score_patterns(t1_hist)
        t2_scores = compute_score_patterns(t2_hist)

        features = {
            "t1_elo": t1_elo, "t2_elo": t2_elo,
            "elo_diff": t1_elo - t2_elo,
            "elo_expected": self.elo.predict(t1.hltv_id, t2.hltv_id),
            "rank_diff": t2_rank - t1_rank,
            "t1_rank": min(t1_rank, 500), "t2_rank": min(t2_rank, 500),
            "rank_ratio": t1_rank / (t2_rank + 1),
            "tier": tier_map.get(str(match.event_tier or "").lower(), 0),
            "best_of": match.best_of or 3,
            "is_lan": 1 if match.is_lan else 0,
            "t1_form3": t1_streaks["form_3"], "t2_form3": t2_streaks["form_3"],
            "t1_form5": t1_streaks["form_5"], "t2_form5": t2_streaks["form_5"],
            "form_diff5": t1_streaks["form_5"] - t2_streaks["form_5"],
            "t1_form10": t1_streaks["form_10"], "t2_form10": t2_streaks["form_10"],
            "form_diff10": t1_streaks["form_10"] - t2_streaks["form_10"],
            "t1_form20": t1_streaks["form_20"], "t2_form20": t2_streaks["form_20"],
            "t1_momentum": t1_streaks["momentum"], "t2_momentum": t2_streaks["momentum"],
            "momentum_diff": t1_streaks["momentum"] - t2_streaks["momentum"],
            "t1_adj_wr": t1_opp["adj_winrate"], "t2_adj_wr": t2_opp["adj_winrate"],
            "adj_wr_diff": t1_opp["adj_winrate"] - t2_opp["adj_winrate"],
            "t1_upset_potential": t1_opp["upset_potential"],
            "t2_upset_potential": t2_opp["upset_potential"],
            "t1_exp": min(len(t1_hist), 50), "t2_exp": min(len(t2_hist), 50),
            "exp_diff": len(t1_hist) - len(t2_hist),
            "h2h_wr": h2h["wr"], "h2h_total": h2h["total"],
            "t1_streak": t1_streaks["streak"], "t2_streak": t2_streaks["streak"],
            "streak_diff": t1_streaks["streak"] - t2_streaks["streak"],
            "t1_max_streak": t1_streaks["max_streak_10"],
            "t2_max_streak": t2_streaks["max_streak_10"],
            "t1_sweep_rate": t1_scores["sweep_rate"],
            "t2_sweep_rate": t2_scores["sweep_rate"],
            "t1_close_rate": t1_scores["close_match_rate"],
            "t2_close_rate": t2_scores["close_match_rate"],
            "t1_avg_map_diff": t1_scores["avg_map_diff"],
            "t2_avg_map_diff": t2_scores["avg_map_diff"],
        }

        X = pd.DataFrame([features])[self.feature_names]
        prob = self.model.predict_proba(X)[0]
        raw_t1 = float(prob[1])
        raw_t2 = float(prob[0])

        # Temperature scaling: shrink towards 50% to fix overconfidence
        # Model says 86% but reality is ~67%, so compress extremes
        # Formula: calibrated = 0.5 + (raw - 0.5) * temperature
        TEMPERATURE = 0.6  # <1 = less confident, tuned from calibration analysis
        t1_prob = 0.5 + (raw_t1 - 0.5) * TEMPERATURE
        t2_prob = 1 - t1_prob

        bo = match.best_of or 3
        if bo >= 3:
            over_2_5 = 1 - t1_prob**2 - t2_prob**2
            sweep_discount = 0.75
            t1_plus = 1 - (t2_prob ** 2) * sweep_discount
            t2_plus = 1 - (t1_prob ** 2) * sweep_discount
        else:
            over_2_5 = 0.0
            t1_plus = t1_prob
            t2_plus = t2_prob

        data_points = len(t1_hist) + len(t2_hist) + h2h["total"]
        confidence = min(data_points / 30, 1.0)

        return MLPrediction(
            match_id=match_id,
            team1_name=t1.name, team2_name=t2.name,
            team1_rank=t1_rank, team2_rank=t2_rank,
            team1_elo=round(t1_elo), team2_elo=round(t2_elo),
            team1_win_prob=round(t1_prob, 4), team2_win_prob=round(t2_prob, 4),
            over_2_5_prob=round(over_2_5, 4),
            team1_plus_1_5_prob=round(t1_plus, 4),
            team2_plus_1_5_prob=round(t2_plus, 4),
            best_of=bo, confidence=round(confidence, 4),
            event=match.event_name or "", tier=match.event_tier or "",
            is_lan=match.is_lan or False,
            team1_odds=prob_to_odds(t1_prob),
            team2_odds=prob_to_odds(t2_prob),
            over_2_5_odds=prob_to_odds(over_2_5),
            team1_plus_1_5_odds=prob_to_odds(t1_plus),
            team2_plus_1_5_odds=prob_to_odds(t2_plus),
            features=features,
        )
