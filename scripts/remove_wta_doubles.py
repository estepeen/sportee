"""One-shot cleanup: remove all WTA (women) and doubles bets/picks from data files.

Reads /opt/sportee/data/, writes filtered versions. Adjusts bankroll to account
for removed bets so remaining numbers stay internally consistent.

Usage:
    python3 scripts/remove_wta_doubles.py --data-dir /opt/sportee/data [--apply]
"""
import argparse
import json
import sqlite3
from pathlib import Path


WTA_EVENT_KEYWORDS = [
    "porsche tennis grand prix",
    "charleston",
    "credit one",
    "bogota",
    "upper austria ladies",
    "rouen metropole",
    "ladies",
    "wta",
    "women",
]

DOUBLES_SOURCE_KEYWORDS = ["doubles", "pair", "dvojic", "čtyřhr"]


def load_name_tour_map(db_path: Path) -> dict:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT p.name, m.tour, COUNT(*) AS c
        FROM tennis_matches m
        JOIN tennis_players p ON (m.winner_id = p.id OR m.loser_id = p.id)
        GROUP BY p.name, m.tour
        """
    )
    name_tour: dict[str, tuple[str, int]] = {}
    for r in cur:
        n, t, c = r["name"], r["tour"], r["c"]
        if c > name_tour.get(n, (None, 0))[1]:
            name_tour[n] = (t, c)
    conn.close()
    return {n: t for n, (t, _) in name_tour.items()}


def tour_of_player(full_name: str, name_tour: dict) -> str | None:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None
    first = parts[0]
    for i in range(1, len(parts)):
        last = " ".join(parts[i:])
        cand = f"{last} {first[0]}."
        if cand in name_tour:
            return name_tour[cand]
    return None


def is_doubles(rec: dict) -> bool:
    mt = (rec.get("match_type") or "").lower()
    if mt == "doubles":
        return True
    src = (rec.get("source") or "").lower()
    if any(k in src for k in DOUBLES_SOURCE_KEYWORDS):
        return True
    for key in ("team1_name", "team2_name", "pick", "opponent", "player1", "player2"):
        v = rec.get(key) or ""
        if "/" in v:
            return True
    return False


def is_wta(rec: dict, name_tour: dict) -> bool:
    tour = (rec.get("tour") or "").upper()
    if tour == "WTA":
        return True
    evt = " ".join(
        str(rec.get(k) or "") for k in ("event", "tournament")
    ).lower()
    if any(k in evt for k in WTA_EVENT_KEYWORDS):
        return True
    candidates = []
    for key in ("team1_name", "team2_name", "pick", "opponent", "player1", "player2"):
        v = rec.get(key)
        if v:
            candidates.append(v)
    atp_seen = False
    for n in candidates:
        t = tour_of_player(n, name_tour)
        if t == "WTA":
            return True
        if t == "ATP":
            atp_seen = True
    return False


def filter_bets(bets: list, name_tour: dict) -> tuple[list, list]:
    kept, removed = [], []
    for b in bets:
        if is_doubles(b) or is_wta(b, name_tour):
            removed.append(b)
        else:
            kept.append(b)
    return kept, removed


def recompute_bankroll_delta(removed_bets: list) -> float:
    """How much to ADD to current bankroll to undo the effect of removed bets.

    Placement deducted stake. Won added back stake+profit. Void added back stake.
    To undo, we do the inverse:
      pending: refund stake (+stake)
      lost: refund stake (+stake)
      won: un-credit stake+profit (-(stake+profit))
      void: nothing (stake already refunded)
    """
    delta = 0.0
    for b in removed_bets:
        s = b.get("stake", 0) or 0
        st = b.get("status", "")
        if st == "pending":
            delta += s
        elif st == "lost":
            delta += s
        elif st == "won":
            delta -= s + (b.get("profit", 0) or 0)
    return delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--apply", action="store_true", help="write changes; default is dry run")
    args = ap.parse_args()
    data = Path(args.data_dir)

    name_tour = load_name_tour_map(data / "tennis.db")
    print(f"Loaded name→tour map: {len(name_tour)} players")

    # bets.json
    bets_path = data / "bets.json"
    bets = json.loads(bets_path.read_text())
    kept_bets, removed_bets = filter_bets(bets, name_tour)
    print(f"bets.json: {len(bets)} → {len(kept_bets)} (removed {len(removed_bets)})")
    by_reason = {"doubles": 0, "wta": 0}
    for b in removed_bets:
        if is_doubles(b):
            by_reason["doubles"] += 1
        else:
            by_reason["wta"] += 1
    print(f"  reasons: {by_reason}")

    # picks_history.json
    picks_path = data / "picks_history.json"
    picks = json.loads(picks_path.read_text())
    kept_picks, removed_picks = filter_bets(picks, name_tour)
    print(f"picks_history.json: {len(picks)} → {len(kept_picks)} (removed {len(removed_picks)})")
    by_reason_p = {"doubles": 0, "wta": 0}
    for p in removed_picks:
        if is_doubles(p):
            by_reason_p["doubles"] += 1
        else:
            by_reason_p["wta"] += 1
    print(f"  reasons: {by_reason_p}")

    # bankroll
    br_path = data / "bankroll.json"
    br = json.loads(br_path.read_text())
    old_bankroll = br.get("bankroll", 0)
    delta = recompute_bankroll_delta(removed_bets)
    new_bankroll = round(old_bankroll + delta, 2)
    print(f"bankroll: {old_bankroll} + {delta:+.2f} → {new_bankroll}")

    # doubles odds file removal
    doubles_odds = data / "sofascore_doubles_odds.json"
    doubles_odds_exists = doubles_odds.exists()
    print(f"sofascore_doubles_odds.json exists: {doubles_odds_exists}")

    if not args.apply:
        print("\nDRY RUN — no writes. Pass --apply to commit.")
        return

    bets_path.write_text(json.dumps(kept_bets, indent=2, ensure_ascii=False))
    picks_path.write_text(json.dumps(kept_picks, indent=2, ensure_ascii=False))
    br["bankroll"] = new_bankroll
    br_path.write_text(json.dumps(br, indent=2))
    if doubles_odds_exists:
        doubles_odds.unlink()
    print("\nWrote bets.json, picks_history.json, bankroll.json; removed sofascore_doubles_odds.json")


if __name__ == "__main__":
    main()
