"""Bet management: recommendations, bankroll, tracking, results, learning."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
BETS_FILE = DATA_DIR / "bets.json"
BANKROLL_FILE = DATA_DIR / "bankroll.json"

MIN_EDGE = 0.05        # 5% minimum edge to recommend
MAX_EDGE = 0.25        # cap edge at 25% - anything higher means model is likely wrong
MIN_CONFIDENCE = 0.4   # minimum model confidence
MIN_ODDS = 1.25        # minimum odds (user preference)
KELLY_FRACTION = 0.25  # quarter Kelly
MAX_STAKE_PCT = 5.0    # max 5% of bankroll per bet
MIN_STAKE_PCT = 0.5    # min 0.5% of bankroll per bet

# Tiered stake sizing based on odds range
# Lower odds = higher confidence = bigger stake
# Higher odds = more risk = smaller stake
STAKE_TIERS = {
    # (min_odds, max_odds): stake_multiplier
    (1.25, 1.60): 1.0,   # safe bets: full Kelly fraction
    (1.60, 2.50): 0.7,   # medium risk: 70% of Kelly
    (2.50, 5.00): 0.4,   # high risk: 40% of Kelly
    (5.00, 99.0): 0.2,   # longshots: 20% of Kelly
}


@dataclass
class BetRecommendation:
    match_id: int
    team1_name: str
    team2_name: str
    event: str
    tier: str
    market: str           # "team1_win", "team2_win", "over_2.5", "team1_+1.5", "team2_+1.5"
    market_label: str     # human readable: "MOUZ WIN", "Over 2.5 maps"
    our_prob: float
    our_odds: float       # implied odds from our model
    edge: float           # how much edge we think we have (vs 50% baseline or vs bookmaker)
    confidence: float
    kelly_stake_pct: float  # recommended stake as % of bankroll
    stake_amount: float     # actual $ amount
    rating: float         # overall bet quality score (edge * confidence)
    reasons: list = field(default_factory=list)
    market_odds: float = 0.0  # real bookmaker/polymarket odds (set after creation)


@dataclass
class BetRecord:
    id: str
    match_id: int
    team1_name: str
    team2_name: str
    event: str
    market: str
    market_label: str
    our_prob: float
    our_odds: float
    stake: float
    status: str = "pending"  # pending, won, lost, void
    actual_result: str = ""
    profit: float = 0.0
    created_at: str = ""


class BetManager:
    """Manages bet recommendations, bankroll, and results tracking."""

    def __init__(self, initial_bankroll: float = 1_000_000.0):
        self.bankroll = initial_bankroll
        self.bets: list[dict] = []
        self.load()

    def load(self):
        """Load bankroll and bet history."""
        try:
            with open(BANKROLL_FILE, "r") as f:
                data = json.load(f)
            self.bankroll = data.get("bankroll", 1_000_000.0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        try:
            with open(BETS_FILE, "r") as f:
                self.bets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.bets = []

    def save(self):
        """Save bankroll and bet history."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(BANKROLL_FILE, "w") as f:
            json.dump({
                "bankroll": round(self.bankroll, 2),
                "updated_at": datetime.utcnow().isoformat(),
            }, f, indent=2)
        with open(BETS_FILE, "w") as f:
            json.dump(self.bets, f, indent=2)

    def generate_recommendations(self, predictions: list) -> list[BetRecommendation]:
        """Generate bet recommendations from model predictions.

        Args:
            predictions: list of dicts with keys like team1_ml_prob, team2_ml_prob,
                         over_2_5_prob, team1_handicap, team2_handicap, confidence, etc.
        """
        recommendations = []
        seen_bets = set()      # deduplicate by team pair + market
        seen_matches = set()   # max 1 bet per match

        for p in predictions:
            if p.get("confidence", 0) < MIN_CONFIDENCE:
                continue

            # Evaluate each market
            markets = self._evaluate_markets(p)

            for market in markets:
                if market["edge"] < MIN_EDGE:
                    continue
                if market["odds"] < MIN_ODDS:
                    continue

                # Deduplicate by team pair + market type
                match_key = tuple(sorted([p.get("team1_name",""), p.get("team2_name","")]))
                bet_key = (match_key, market["key"])
                if bet_key in seen_bets:
                    continue

                # No contradicting bets: can't bet WIN on both teams in same match
                market_type = market["key"].split("_")[-1] if "_" in market["key"] else market["key"]
                if market_type == "win":
                    # Check if we already have a WIN bet on the other team
                    other_win_keys = [
                        (match_key, f"team1_win"),
                        (match_key, f"team2_win"),
                    ]
                    if any(k in seen_bets for k in other_win_keys):
                        continue

                seen_bets.add(bet_key)

                model_prob = market["prob"]
                data_conf = p.get("confidence", 0.5)
                edge = market["edge"]

                # Cap edge - if model disagrees with market by >25%, model is likely wrong
                capped_edge = min(edge, MAX_EDGE)

                # Real confidence = how sure are we really
                # Outsider (odds > 2.0) gets confidence penalty
                odds_penalty = 1.0 if market["odds"] < 2.0 else 0.7 if market["odds"] < 3.0 else 0.5
                real_confidence = model_prob * min(data_conf, 1.0) * odds_penalty

                # Rating 5-10 for stake sizing
                edge_score = min(capped_edge / 0.05, 5)
                conf_score = min(real_confidence / 0.15, 5)
                rating_score = min(max(round(edge_score + conf_score), 5), 10)
                stake_amount = rating_score * 10

                rating = capped_edge * real_confidence

                # Build reasons
                reasons = self._build_reasons(p, market)

                recommendations.append(BetRecommendation(
                    match_id=p.get("match_id", 0),
                    team1_name=p.get("team1_name", ""),
                    team2_name=p.get("team2_name", ""),
                    event=p.get("event", ""),
                    tier=p.get("tier", ""),
                    market=market["key"],
                    market_label=market["label"],
                    our_prob=round(market["prob"], 4),
                    our_odds=market["odds"],
                    edge=round(market["edge"], 4),
                    confidence=round(real_confidence, 4),
                    kelly_stake_pct=rating_score,  # use as rating /10
                    stake_amount=stake_amount,
                    rating=round(rating, 4),
                    reasons=reasons,
                ))

        # Sort by rating (best bets first)
        recommendations.sort(key=lambda r: r.rating, reverse=True)
        return recommendations

    def _evaluate_markets(self, p: dict) -> list[dict]:
        """Compare model predictions vs Polymarket odds. Find edge."""
        markets = []
        candidates = []
        t1 = p.get("team1_name", "T1")
        t2 = p.get("team2_name", "T2")

        t1_prob = p.get("team1_ml_prob", 0.5)
        t2_prob = p.get("team2_ml_prob", 0.5)

        # Use POLYMARKET real odds (not model implied)
        pm_t1_odds = p.get("pm_t1_odds")
        pm_t2_odds = p.get("pm_t2_odds")
        pm_t1_price = p.get("pm_t1_price")  # 0-1 probability from PM
        pm_t2_price = p.get("pm_t2_price")

        # Must have Polymarket odds - no PM = no bet
        if not pm_t1_odds or not pm_t2_odds or not pm_t1_price or not pm_t2_price:
            return markets

        # TEAM 1 WIN - only if model gives >50% chance (we actually think they win)
        if pm_t1_odds >= MIN_ODDS and t1_prob >= 0.50:
            edge = t1_prob - pm_t1_price
            if edge >= MIN_EDGE:
                candidates.append({
                    "key": "team1_win", "label": f"{t1[:14]} WIN",
                    "prob": t1_prob, "odds": pm_t1_odds,
                    "edge": edge, "bet_type": "win",
                })

        # TEAM 2 WIN - only if model gives >50% chance
        if pm_t2_odds >= MIN_ODDS and t2_prob >= 0.50:
            edge = t2_prob - pm_t2_price
            if edge >= MIN_EDGE:
                candidates.append({
                    "key": "team2_win", "label": f"{t2[:14]} WIN",
                    "prob": t2_prob, "odds": pm_t2_odds,
                    "edge": edge, "bet_type": "win",
                })

        # HANDICAP +1.5 maps with real PM odds (safer than WIN)
        if p.get("best_of", 3) >= 3:
            if t1_prob < t2_prob:
                dog, dog_key = t1, "team1"
            else:
                dog, dog_key = t2, "team2"

            dog_hc_odds = p.get(f"pm_{dog_key}_hc_odds")
            dog_hc_price = p.get(f"pm_{dog_key}_hc_price")
            if dog_hc_odds and dog_hc_price and dog_hc_odds >= MIN_ODDS:
                dog_hc_prob = p.get(f"{dog_key}_handicap", 0.7)
                edge = dog_hc_prob - dog_hc_price
                if edge >= MIN_EDGE:
                    candidates.append({
                        "key": f"{dog_key}_+1.5", "label": f"{dog[:14]} +1.5 maps",
                        "prob": dog_hc_prob, "odds": dog_hc_odds,
                        "edge": edge, "bet_type": "handicap",
                    })

        # Pick BEST bet: prefer handicap (safer), then WIN
        conf = min(p.get("confidence", 0.5), 1.0)
        valid = [c for c in candidates if c["edge"] >= MIN_EDGE]
        if valid:
            # Handicap gets 1.3x boost (safer bet type)
            for c in valid:
                c["_score"] = c["edge"] * conf * (1.3 if c["bet_type"] == "handicap" else 1.0)
            best = max(valid, key=lambda c: c["_score"])
            markets.append(best)

        return markets

    def _build_reasons(self, p: dict, market: dict) -> list[str]:
        """Build human-readable reasons for the recommendation."""
        reasons = []

        t1 = p.get("team1_name", "")
        t2 = p.get("team2_name", "")

        # Ranking
        r1 = p.get("team1_rank", 999)
        r2 = p.get("team2_rank", 999)
        if abs(r1 - r2) > 10:
            higher = t1 if r1 < r2 else t2
            reasons.append(f"Ranking advantage: {higher} (#{min(r1,r2)} vs #{max(r1,r2)})")

        # Elo
        e1 = p.get("team1_elo", 1500)
        e2 = p.get("team2_elo", 1500)
        if abs(e1 - e2) > 100:
            higher = t1 if e1 > e2 else t2
            reasons.append(f"Elo edge: {higher} ({max(e1,e2)} vs {min(e1,e2)})")

        # Form
        f1 = p.get("t1_form")
        f2 = p.get("t2_form")
        if f1 and f2 and abs(f1 - f2) > 0.15:
            better = t1 if f1 > f2 else t2
            reasons.append(f"Better form: {better} ({max(f1,f2)*100:.0f}% vs {min(f1,f2)*100:.0f}%)")

        # Streak
        s1 = p.get("t1_streak", 0)
        s2 = p.get("t2_streak", 0)
        if s1 >= 3:
            reasons.append(f"{t1} on W{s1} streak")
        if s2 >= 3:
            reasons.append(f"{t2} on W{s2} streak")
        if s1 <= -3:
            reasons.append(f"{t1} on L{-s1} losing streak")
        if s2 <= -3:
            reasons.append(f"{t2} on L{-s2} losing streak")

        # LAN
        if p.get("is_lan"):
            reasons.append("LAN match (higher stakes)")

        # Tier
        if p.get("tier") in ("s", "a"):
            reasons.append(f"Tier {p['tier'].upper()} tournament")

        if not reasons:
            reasons.append(f"Model probability: {market['prob']*100:.0f}%")

        return reasons

    def record_bet(self, rec: BetRecommendation, auto_save: bool = True) -> BetRecord:
        """Record a recommended bet."""
        record = BetRecord(
            id=f"{rec.team1_name[:8]}_{rec.market[:6]}_{datetime.utcnow().strftime('%H%M%S%f')[:10]}",
            match_id=rec.match_id,
            team1_name=rec.team1_name,
            team2_name=rec.team2_name,
            event=rec.event,
            market=rec.market,
            market_label=rec.market_label,
            our_prob=rec.our_prob,
            our_odds=rec.our_odds,
            stake=rec.stake_amount,
            created_at=datetime.utcnow().isoformat(),
        )
        self.bets.append(asdict(record))
        if auto_save:
            self.save()
        return record

    def resolve_bet(self, bet_id: str, result: str):
        """Resolve a bet. result: 'won', 'lost', 'void'. Works for any status."""
        for bet in self.bets:
            if bet["id"] == bet_id:
                old_status = bet["status"]
                old_profit = bet.get("profit", 0)

                # Reverse previous resolution
                if old_status == "won":
                    self.bankroll -= old_profit + bet["stake"]  # undo win payout
                elif old_status == "void":
                    self.bankroll -= bet["stake"]  # undo void refund
                # lost = nothing to undo (stake was already deducted at placement)
                # pending = nothing to undo

                # Apply new result
                bet["status"] = result
                if result == "won":
                    bet["profit"] = round(bet["stake"] * (bet["our_odds"] - 1), 2)
                    self.bankroll += bet["profit"] + bet["stake"]
                elif result == "lost":
                    bet["profit"] = -bet["stake"]
                elif result == "void":
                    bet["profit"] = 0
                    self.bankroll += bet["stake"]
                elif result == "pending":
                    bet["profit"] = 0
                    # stake stays deducted (it was deducted at placement)

                self.save()
                return

    def get_stats(self) -> dict:
        """Get betting performance statistics."""
        resolved = [b for b in self.bets if b["status"] in ("won", "lost")]
        pending_bets = [b for b in self.bets if b["status"] == "pending"]
        if not resolved:
            return {
                "total_bets": 0, "wins": 0, "losses": 0,
                "winrate": 0, "total_profit": 0, "roi": 0,
                "bankroll": self.bankroll,
                "pending": len(pending_bets),
                "pending_exposure": round(sum(b["stake"] for b in pending_bets), 2),
            }

        wins = sum(1 for b in resolved if b["status"] == "won")
        losses = sum(1 for b in resolved if b["status"] == "lost")
        total_staked = sum(b["stake"] for b in resolved)
        total_profit = sum(b["profit"] for b in resolved)

        return {
            "total_bets": len(resolved),
            "wins": wins,
            "losses": losses,
            "winrate": round(wins / len(resolved) * 100, 1) if resolved else 0,
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
            "bankroll": round(self.bankroll, 2),
            "pending": len(pending_bets),
            "pending_exposure": round(sum(b["stake"] for b in pending_bets), 2),
            "avg_odds": round(sum(b["our_odds"] for b in resolved) / len(resolved), 2) if resolved else 0,
        }

    def auto_resolve_bets(self, completed_matches: list[dict]):
        """Automatically resolve pending bets based on match results.

        Args:
            completed_matches: list of dicts with keys: match_id, winner_id,
                               team1_id, team2_id, team1_score, team2_score
        """
        results_by_id = {m["match_id"]: m for m in completed_matches}
        resolved = 0

        for bet in self.bets:
            if bet["status"] != "pending":
                continue
            if bet["match_id"] not in results_by_id:
                continue

            result = results_by_id[bet["match_id"]]
            market = bet["market"]
            t1_id = result.get("team1_id")
            t2_id = result.get("team2_id")
            winner_id = result.get("winner_id")
            t1_score = result.get("team1_score") or 0
            t2_score = result.get("team2_score") or 0

            won = None

            if market == "team1_win":
                won = winner_id == t1_id
            elif market == "team2_win":
                won = winner_id == t2_id
            elif market == "team1_+1.5":
                # Team1 wins at least 1 map
                won = t1_score >= 1
            elif market == "team2_+1.5":
                won = t2_score >= 1
            elif market == "over_2.5":
                won = (t1_score + t2_score) >= 3
            elif market == "under_2.5":
                won = (t1_score + t2_score) < 3

            if won is None:
                continue

            if won:
                bet["status"] = "won"
                bet["profit"] = round(bet["stake"] * (bet["our_odds"] - 1), 2)
                self.bankroll += bet["profit"] + bet["stake"]
            else:
                bet["status"] = "lost"
                bet["profit"] = -bet["stake"]

            bet["actual_result"] = f"{t1_score}-{t2_score} winner:{winner_id}"
            resolved += 1

        if resolved:
            self.save()
            logger.info(f"Auto-resolved {resolved} bets")

        return resolved

    @staticmethod
    def _norm_name(n):
        return n.lower().replace("team ", "").replace(" esports", "").replace(" gaming", "").strip()

    def auto_record_recommendations(self, recommendations: list):
        """Automatically record top recommendations as pending bets."""
        recorded = 0
        for rec in recommendations[:5]:
            # Deduplicate: normalize team names to catch "Falcons" vs "Team Falcons"
            n1 = self._norm_name(rec.team1_name)
            n2 = self._norm_name(rec.team2_name)
            bet_key = tuple(sorted([n1, n2]))
            existing = any(
                tuple(sorted([self._norm_name(b["team1_name"]), self._norm_name(b["team2_name"])])) == bet_key
                and b["status"] == "pending"
                for b in self.bets
            )
            if existing:
                continue

            self.record_bet(rec)
            self.bankroll -= rec.stake_amount  # deduct from bankroll
            recorded += 1

        if recorded:
            self.save()
            logger.info(f"Auto-recorded {recorded} new bets")
        return recorded

    def get_pending_bets(self) -> list[dict]:
        return [b for b in self.bets if b["status"] == "pending"]

    def get_recent_bets(self, n: int = 20) -> list[dict]:
        return sorted(self.bets, key=lambda b: b.get("created_at", ""), reverse=True)[:n]

    @staticmethod
    def _kelly(prob: float, odds: float) -> float:
        """Kelly criterion: f* = (bp - q) / b"""
        b = odds - 1
        if b <= 0:
            return 0
        q = 1 - prob
        f = (b * prob - q) / b
        return max(f, 0)

    @staticmethod
    def _prob_to_odds(prob: float) -> float:
        if prob <= 0.01:
            return 99.0
        return round(1 / prob, 2)
