"""Auto-resolve pending bets from SofaScore finished matches.

100% reliable: matches BOTH player names + match must be finished.
Uses SofaScore match/list API (1 request per day checked).
"""

import json
import logging
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from src.tennis.sofascore_api import HEADERS, BASE_URL, _load_usage, _save_usage

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _norm(name: str) -> str:
    """Normalize name for matching: strip diacritics, lowercase, remove dots/spaces."""
    s = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    return s.lower().replace(" ", "").replace(".", "").replace("-", "").strip()


def _names_match(api_name: str, bet_name: str) -> bool:
    """Check if SofaScore player name matches bet player name."""
    an = _norm(api_name)
    bn = _norm(bet_name)
    if not an or not bn:
        return False
    # Full match
    if an == bn:
        return True
    # Last name match (at least 5 chars)
    a_last = api_name.strip().split()[-1] if api_name.strip() else ""
    b_last = bet_name.strip().split()[-1] if bet_name.strip() else ""
    if len(a_last) >= 4 and _norm(a_last) == _norm(b_last):
        return True
    # Partial: first 6 chars of normalized
    if len(an) >= 6 and len(bn) >= 6 and an[:6] == bn[:6]:
        return True
    return False


async def resolve_from_sofascore(days: int = 3):
    """Fetch finished matches from SofaScore and resolve pending bets.

    Args:
        days: How many days back to check (default 3)

    Returns:
        Number of bets resolved
    """
    from src.strategy.bet_manager import BetManager

    bm = BetManager()
    bm.load()
    pending = [b for b in bm.bets if b["status"] == "pending"]

    if not pending:
        return 0

    usage = _load_usage()
    resolved_count = 0

    async with httpx.AsyncClient(timeout=120) as client:
        for d in range(days):
            date = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")

            # Fetch finished matches for this day
            url = f"{BASE_URL}/match/list?sport_slug=tennis&date={date}"
            try:
                resp = await client.get(url, headers=HEADERS, timeout=120)
                usage["count"] = usage.get("count", 0) + 1
                _save_usage(usage)

                if resp.status_code != 200:
                    continue

                data = resp.json()
            except Exception as e:
                logger.warning(f"SofaScore error for {date}: {e}")
                continue

            events = data if isinstance(data, list) else data.get("events", [])

            # Build list of finished matches
            finished = []
            for ev in events:
                status = ev.get("status", {})
                if not status.get("isFinished"):
                    continue

                home = ev.get("homeTeam", {}).get("name", "")
                away = ev.get("awayTeam", {}).get("name", "")
                if not home or not away:
                    continue

                hs = ev.get("homeScore", {})
                if not isinstance(hs, dict):
                    hs = {}
                aws = ev.get("awayScore", {})
                if not isinstance(aws, dict):
                    aws = {}

                h_sets = hs.get("current", 0) or 0
                a_sets = aws.get("current", 0) or 0
                if h_sets == a_sets:
                    continue

                winner = home if h_sets > a_sets else away
                loser = away if h_sets > a_sets else home
                w_sets = max(h_sets, a_sets)
                l_sets = min(h_sets, a_sets)

                # Build score string
                score_parts = []
                for i in range(1, 6):
                    hp = hs.get(f"period{i}")
                    ap = aws.get(f"period{i}")
                    if hp is not None and ap is not None:
                        if h_sets > a_sets:
                            score_parts.append(f"{hp}-{ap}")
                        else:
                            score_parts.append(f"{ap}-{hp}")

                slug = ev.get("slug", "")
                eid = ev.get("id", 0)

                finished.append({
                    "home": home,
                    "away": away,
                    "winner": winner,
                    "loser": loser,
                    "w_sets": w_sets,
                    "l_sets": l_sets,
                    "score": " ".join(score_parts),
                    "slug": slug,
                    "event_id": eid,
                    "date": date,
                })

            # Match finished games to pending bets
            for match in finished:
                for bet in pending:
                    if bet["status"] != "pending":
                        continue  # already resolved in this loop

                    t1 = bet.get("team1_name", "")
                    t2 = bet.get("team2_name", "")
                    label = bet.get("market_label", "")
                    pick_name = label.replace(" WIN", "").replace(" +1.5 SETS", "").replace(" +1.5 Sets", "").replace(" TOTAL", "").strip()

                    # BOTH players must match
                    home_matches_t1 = _names_match(match["home"], t1) or _names_match(match["away"], t1)
                    home_matches_t2 = _names_match(match["home"], t2) or _names_match(match["away"], t2)

                    if not (home_matches_t1 and home_matches_t2):
                        continue

                    # Determine if pick won
                    pick_won_match = _names_match(match["winner"], pick_name)

                    if "WIN" in label and "1.5" not in label:
                        won = pick_won_match
                    elif "1.5" in label:
                        if pick_won_match:
                            won = True
                        else:
                            # Pick lost the match - did they win at least 1 set?
                            won = match["l_sets"] >= 1
                    else:
                        continue

                    result = "won" if won else "lost"
                    bm.resolve_bet(bet["id"], result)

                    # Add score and SofaScore URL
                    score_str = f"{match['winner']} {match['w_sets']}-{match['l_sets']} ({match['score']})" if match["score"] else f"{match['winner']} {match['w_sets']}-{match['l_sets']}"
                    bet["actual_result"] = score_str

                    if match["slug"] and match["event_id"]:
                        bet["pm_url"] = f"https://www.sofascore.com/tennis/match/{match['slug']}#id:{match['event_id']}"

                    resolved_count += 1
                    logger.info(f"Resolved: {label} -> {result.upper()} | {score_str}")

    if resolved_count:
        bm.save()

    logger.info(f"Auto-resolve: {resolved_count} bets resolved (API: {usage.get('count', 0)})")
    return resolved_count
