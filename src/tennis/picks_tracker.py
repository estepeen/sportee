"""Track all AI picks and value bets with results for analytics."""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.tennis.database import get_tennis_db
from src.tennis.stats import find_player

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PICKS_FILE = DATA_DIR / "picks_history.json"


def load_picks() -> list[dict]:
    try:
        with open(PICKS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_picks(picks: list[dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PICKS_FILE, "w") as f:
        json.dump(picks, f, indent=2)


_WTA_KEYWORDS = ("wta", "women", "ladies")
_DOUBLES_KEYWORDS = ("doubles", "pair", "dvojic", "čtyřhr")


def _is_excluded(p: dict, source: str) -> bool:
    """Reject WTA (women) and doubles picks — we only track ATP singles."""
    if (p.get("tour") or "").upper() == "WTA":
        return True
    if (p.get("match_type") or "").lower() == "doubles":
        return True
    src = (source or "").lower()
    if any(k in src for k in _DOUBLES_KEYWORDS):
        return True
    haystack = " ".join(
        str(p.get(k) or "") for k in ("tournament", "tour", "event")
    ).lower()
    if any(k in haystack for k in _WTA_KEYWORDS):
        return True
    for key in ("pick", "opponent", "player1", "player2"):
        if "/" in (p.get(key) or ""):
            return True
    return False


def track_picks(new_picks: list[dict], source: str = "ai_picks"):
    """Save new picks that aren't already tracked. Skips WTA and doubles."""
    history = load_picks()
    existing_keys = {(p["pick"], p["opponent"], p["date_added"][:10]) for p in history}

    added = 0
    skipped = 0
    for p in new_picks:
        if _is_excluded(p, source):
            skipped += 1
            continue
        key = (p.get("pick", ""), p.get("opponent", ""), datetime.now().strftime("%Y-%m-%d"))
        if key in existing_keys:
            continue

        history.append({
            "pick": p.get("pick", ""),
            "opponent": p.get("opponent", ""),
            "tournament": p.get("tournament", ""),
            "tour": p.get("tour", ""),
            "bet_type": p.get("bet_type", "WIN"),
            "odds": p.get("odds", 0),
            "original_win_odds": p.get("original_win_odds", p.get("odds", 0)),
            "ml_prob": p.get("ml_prob", 0),
            "mkt_prob": p.get("mkt_prob", 0),
            "edge": p.get("edge", 0),
            "confidence": p.get("confidence", ""),
            "suggested_stake": p.get("suggested_stake", 0),
            "source": source,
            "status": "OPEN",
            "result_score": "",
            "date_added": datetime.now().isoformat(),
        })
        existing_keys.add(key)
        added += 1

    if added:
        save_picks(history)
        logger.info(f"Tracked {added} new {source} picks")
    if skipped:
        logger.info(f"Skipped {skipped} {source} picks (WTA/doubles filter)")


def resolve_picks():
    """Check DB for finished matches and update pick statuses."""
    history = load_picks()
    if not history:
        return 0

    open_picks = [p for p in history if p["status"] == "OPEN"]
    if not open_picks:
        return 0

    conn = get_tennis_db()
    resolved = 0

    for p in open_picks:
        pick_player = find_player(conn, p["pick"])
        opp_player = find_player(conn, p["opponent"])
        if not pick_player or not opp_player:
            continue

        row = conn.execute("""
            SELECT winner_id, w_sets, l_sets, date FROM tennis_matches
            WHERE ((winner_id=? AND loser_id=?) OR (winner_id=? AND loser_id=?))
            AND date >= date('now', '-5 days')
            ORDER BY date DESC LIMIT 1
        """, (pick_player["id"], opp_player["id"],
              opp_player["id"], pick_player["id"])).fetchone()

        if not row:
            continue

        bet_type = p.get("bet_type", "WIN")
        pick_won_match = row["winner_id"] == pick_player["id"]

        if bet_type == "WIN":
            p["status"] = "WIN" if pick_won_match else "LOSS"
        elif "+1.5" in bet_type:
            if pick_won_match:
                p["status"] = "WIN"
            else:
                p["status"] = "WIN" if row["l_sets"] >= 1 else "LOSS"
        else:
            continue

        p["result_score"] = f"{row['w_sets']}-{row['l_sets']}"
        p["resolved_at"] = datetime.now().isoformat()
        resolved += 1

    conn.close()

    if resolved:
        save_picks(history)
        logger.info(f"Resolved {resolved} picks")

    return resolved


def get_analytics() -> dict:
    """Compute analytics from picks history."""
    history = load_picks()
    resolved = [p for p in history if p["status"] in ("WIN", "LOSS")]

    if not resolved:
        return {
            "total": len(history),
            "open": len([p for p in history if p["status"] == "OPEN"]),
            "resolved": 0,
            "wins": 0, "losses": 0, "winrate": 0,
            "profit": 0, "roi": 0,
            "by_source": {}, "by_tour": {}, "by_type": {},
            "by_confidence": {}, "by_tournament": {},
            "recent": history[-20:],
        }

    wins = [p for p in resolved if p["status"] == "WIN"]
    losses = [p for p in resolved if p["status"] == "LOSS"]

    # Profit calculation
    total_staked = 0
    total_profit = 0
    for p in resolved:
        stake = p.get("suggested_stake", 50) or 50
        total_staked += stake
        if p["status"] == "WIN":
            total_profit += stake * (p.get("odds", 2.0) - 1)
        else:
            total_profit -= stake

    def breakdown(key_fn):
        groups = {}
        for p in resolved:
            k = key_fn(p)
            if k not in groups:
                groups[k] = {"wins": 0, "losses": 0, "profit": 0, "staked": 0}
            stake = p.get("suggested_stake", 50) or 50
            groups[k]["staked"] += stake
            if p["status"] == "WIN":
                groups[k]["wins"] += 1
                groups[k]["profit"] += stake * (p.get("odds", 2.0) - 1)
            else:
                groups[k]["losses"] += 1
                groups[k]["profit"] -= stake

        for k, v in groups.items():
            total = v["wins"] + v["losses"]
            v["total"] = total
            v["winrate"] = round(v["wins"] / total * 100, 1) if total > 0 else 0
            v["roi"] = round(v["profit"] / v["staked"] * 100, 1) if v["staked"] > 0 else 0
            v["profit"] = round(v["profit"], 2)
        return dict(sorted(groups.items(), key=lambda x: -x[1]["total"]))

    # Detect tournament tier
    def tour_tier(p):
        t = (p.get("tournament", "") + " " + p.get("tour", "")).lower()
        if any(w in t for w in ["grand slam", "australian", "roland", "wimbledon", "us open"]):
            return "Grand Slam"
        if any(w in t for w in ["masters", "miami", "indian wells", "rome", "madrid", "monte carlo", "shanghai", "paris", "cincinnati", "canadian"]):
            return "Masters 1000"
        if "wta" in t:
            return "WTA Tour"
        if any(w in t for w in ["challenger", "split", "alicante", "morelia", "bucaramanga", "naples", "yokkaichi"]):
            return "Challenger"
        return "ATP Tour"

    return {
        "total": len(history),
        "open": len([p for p in history if p["status"] == "OPEN"]),
        "resolved": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(len(wins) / len(resolved) * 100, 1),
        "profit": round(total_profit, 2),
        "staked": round(total_staked, 2),
        "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        "by_source": breakdown(lambda p: p.get("source", "unknown")),
        "by_tour": breakdown(lambda p: tour_tier(p)),
        "by_type": breakdown(lambda p: p.get("bet_type", "WIN")),
        "by_confidence": breakdown(lambda p: p.get("confidence", "unknown")),
        "by_tournament": breakdown(lambda p: p.get("tournament", "unknown")[:20]),
        "recent": list(reversed(history[-30:])),
    }
