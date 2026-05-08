"""Finalize odds for pending bets exactly 4h after they were recorded.

Run via cron every 15 min. For each pending bet:
- if odds_finalized is True or recorded_at is missing: skip
- if (now - recorded_at) < 4h: skip
- else: overwrite our_odds with current Polymarket price from data/tennis_odds.json,
  set odds_finalized=True, odds_finalized_at=<utcnow>. Done. Bet is locked.

This is the only place that ever changes our_odds on a pending bet. Live re-syncs
during matches are intentionally disabled (see src/web/app.py).
"""

import json
import logging
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BETS_FILE = DATA_DIR / "bets.json"
PM_FILE = DATA_DIR / "tennis_odds.json"
PICKS_FILE = DATA_DIR / "picks_history.json"

FINALIZE_AFTER = timedelta(hours=4)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [finalize_pick_odds] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    if not s:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def _load_pm_index() -> dict:
    try:
        with open(PM_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning(f"PM odds file unavailable: {e}")
        return {}
    idx = {}
    for m in data:
        p1, p2 = _norm(m.get("player1", "")), _norm(m.get("player2", ""))
        if p1 and p2:
            idx[tuple(sorted([p1, p2]))] = m
    return idx


def _odds_for(pm_match: dict, pick_norm: str) -> float:
    if _norm(pm_match.get("player1", "")) == pick_norm:
        return pm_match.get("player1_odds", 0) or 0
    if _norm(pm_match.get("player2", "")) == pick_norm:
        return pm_match.get("player2_odds", 0) or 0
    return 0


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def finalize() -> int:
    if not BETS_FILE.exists():
        log.info("No bets.json yet — nothing to finalize")
        return 0

    with open(BETS_FILE) as f:
        bets = json.load(f)

    pm_idx = _load_pm_index()
    if not pm_idx:
        log.warning("Empty PM index — skipping run")
        return 0

    now = datetime.utcnow()
    finalized_keys: list[tuple[str, str, str]] = []
    finalized = 0
    skipped_window = 0
    skipped_no_match = 0

    legacy_locked = 0
    for b in bets:
        if b.get("status") != "pending":
            continue
        if b.get("odds_finalized") is True:
            continue
        recorded_at_raw = b.get("recorded_at", "")
        if not recorded_at_raw:
            # Legacy bet from before this feature shipped — lock at current our_odds
            # without rewriting (we don't know if the match is live, so any rewrite
            # would risk a live-odds lock). One-time backfill on first cron run.
            b["odds_finalized"] = True
            b["odds_finalized_at"] = now.isoformat()
            b["odds_finalized_legacy"] = True
            legacy_locked += 1
            continue
        recorded_at = _parse_iso(recorded_at_raw)
        if recorded_at is None or (now - recorded_at) < FINALIZE_AFTER:
            skipped_window += 1
            continue

        pick = _norm(b.get("team1_name", ""))
        opp = _norm(b.get("team2_name", ""))
        if not pick or not opp:
            continue
        pm_match = pm_idx.get(tuple(sorted([pick, opp])))
        if not pm_match:
            skipped_no_match += 1
            continue
        new_odds = _odds_for(pm_match, pick)
        if not new_odds or new_odds < 1.01 or new_odds > 100:
            skipped_no_match += 1
            continue

        old_odds = b.get("our_odds", 0)
        b["our_odds"] = new_odds
        b["odds_source"] = "polymarket"
        b["odds_finalized"] = True
        b["odds_finalized_at"] = now.isoformat()

        new_mkt_prob = 1.0 / new_odds
        if isinstance(b.get("bet_meta"), dict):
            b["bet_meta"]["mkt_prob"] = round(new_mkt_prob, 4)
            our_prob = b.get("our_prob", 0) or 0
            b["bet_meta"]["edge"] = round((our_prob - new_mkt_prob) * 100, 1)

        finalized += 1
        finalized_keys.append((pick, opp, b.get("market_label", "")))
        log.info(
            f"finalized {b.get('team1_name')} vs {b.get('team2_name')} "
            f"{b.get('market_label')}: {old_odds} -> {new_odds}"
        )

    if finalized or legacy_locked:
        # Atomic write
        tmp = BETS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(bets, f, indent=2)
        tmp.replace(BETS_FILE)

        if finalized:
            _mirror_picks_history(finalized_keys, pm_idx, now)

    log.info(
        f"done: finalized={finalized} legacy_locked={legacy_locked} "
        f"pending_in_window={skipped_window} pending_no_pm_match={skipped_no_match}"
    )
    return finalized


def _mirror_picks_history(keys: list[tuple[str, str, str]], pm_idx: dict, now: datetime):
    """Mirror finalized odds onto picks_history.json so analytics stays consistent."""
    if not PICKS_FILE.exists():
        return
    try:
        with open(PICKS_FILE) as f:
            picks = json.load(f)
    except json.JSONDecodeError:
        return

    target = {(p, o) for (p, o, _) in keys}
    changed = 0
    for entry in picks:
        if entry.get("status") != "OPEN":
            continue
        pick = _norm(entry.get("pick", ""))
        opp = _norm(entry.get("opponent", ""))
        if (pick, opp) not in target:
            continue
        pm_match = pm_idx.get(tuple(sorted([pick, opp])))
        if not pm_match:
            continue
        new_odds = _odds_for(pm_match, pick)
        if not new_odds:
            continue
        entry["odds"] = new_odds
        entry["odds_finalized_at"] = now.isoformat()
        if pm_match.get("player1_price") and _norm(pm_match.get("player1", "")) == pick:
            entry["mkt_prob"] = round(pm_match.get("player1_price", 0) or 0, 4)
        elif pm_match.get("player2_price") and _norm(pm_match.get("player2", "")) == pick:
            entry["mkt_prob"] = round(pm_match.get("player2_price", 0) or 0, 4)
        ml = entry.get("ml_prob", 0) or 0
        entry["edge"] = round((ml - 1.0 / new_odds) * 100, 1)
        changed += 1

    if changed:
        tmp = PICKS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(picks, f, indent=2)
        tmp.replace(PICKS_FILE)
        log.info(f"mirrored {changed} picks_history entries")


if __name__ == "__main__":
    try:
        finalize()
    except Exception:
        log.exception("finalize_pick_odds failed")
        sys.exit(1)
