"""Replace non-Polymarket URLs in bets.json pm_url field.

For each bet whose pm_url is not a polymarket.com link:
  1. Try to find the match in tennis_odds.json (current PM scraper cache) by
     normalized player names → use that slug-based URL.
  2. Otherwise fall back to a Polymarket search URL with the player names.
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path
from urllib.parse import quote


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def build_pm_index(pm_matches: list) -> dict:
    idx = {}
    for m in pm_matches:
        url = m.get("pm_url") or ""
        if "polymarket.com" not in url:
            continue
        p1 = norm(m.get("player1", ""))
        p2 = norm(m.get("player2", ""))
        if not p1 or not p2:
            continue
        key = tuple(sorted([p1, p2]))
        idx[key] = url
    return idx


def last_name(full: str) -> str:
    parts = norm(full).split()
    return parts[-1] if parts else ""


def build_pm_last_index(pm_matches: list) -> dict:
    idx = {}
    for m in pm_matches:
        url = m.get("pm_url") or ""
        if "polymarket.com" not in url:
            continue
        l1 = last_name(m.get("player1", ""))
        l2 = last_name(m.get("player2", ""))
        if not l1 or not l2:
            continue
        key = tuple(sorted([l1, l2]))
        idx[key] = url
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    data = Path(args.data_dir)

    bets_path = data / "bets.json"
    pm_path = data / "tennis_odds.json"
    bets = json.loads(bets_path.read_text())
    pm = json.loads(pm_path.read_text()) if pm_path.exists() else []

    idx_full = build_pm_index(pm)
    idx_last = build_pm_last_index(pm)
    print(f"Loaded {len(pm)} PM matches, {len(idx_full)} indexed by full, {len(idx_last)} by last-name")

    matched_full = 0
    matched_last = 0
    fallback_search = 0
    kept = 0
    changed = 0

    for b in bets:
        cur = b.get("pm_url", "")
        if cur and "polymarket.com" in cur:
            kept += 1
            continue

        t1 = b.get("team1_name", "")
        t2 = b.get("team2_name", "")
        k_full = tuple(sorted([norm(t1), norm(t2)]))
        k_last = tuple(sorted([last_name(t1), last_name(t2)]))

        new_url = None
        if k_full in idx_full:
            new_url = idx_full[k_full]
            matched_full += 1
        elif k_last in idx_last:
            new_url = idx_last[k_last]
            matched_last += 1
        else:
            q = f"{t1} {t2}".strip()
            if q:
                new_url = f"https://polymarket.com/markets?search={quote(q)}"
                fallback_search += 1

        if new_url and new_url != cur:
            b["pm_url"] = new_url
            changed += 1

    print(f"\nKept (already polymarket.com): {kept}")
    print(f"Changed total: {changed}")
    print(f"  matched by full name: {matched_full}")
    print(f"  matched by last name: {matched_last}")
    print(f"  fallback to PM search: {fallback_search}")

    if args.apply:
        bets_path.write_text(json.dumps(bets, indent=2, ensure_ascii=False))
        print("\nWrote bets.json")
    else:
        print("\nDRY RUN — pass --apply to commit.")


if __name__ == "__main__":
    main()
