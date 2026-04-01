"""Lucky Loser fade system - bet AGAINST top qualifying seeds who tank.

Theory: In ATP/WTA/Grand Slam qualifying, when a main draw player withdraws,
the highest remaining qualifying seed gets a Lucky Loser spot. So:
- 1 withdrawal = Q seed 1 knows they're in → tanks qualifying
- 2 withdrawals = Q seed 1 + 2 know → both tank
- 3 withdrawals = Q seeds 1, 2, 3 → all tank
- etc.

Strategy:
1. Monitor main draw for withdrawals (WO/bye entries in R1)
2. Count withdrawals → determines how many Q seeds will tank
3. Generate FADE picks: bet AGAINST those Q seeds in their qualifying matches
4. Only ATP/WTA main tour + Grand Slams (no challengers)
"""

import json
import logging
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

RAPIDAPI_HOST = "sofascore6.p.rapidapi.com"
RAPIDAPI_KEY = "14ba666fd3mshb5821960ffbefdcp127e1bjsnce91762db49e"
HEADERS = {
    "Content-Type": "application/json",
    "x-rapidapi-host": RAPIDAPI_HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
}

DATA_DIR = Path(__file__).parent.parent.parent / "data"
LL_FILE = DATA_DIR / "lucky_losers.json"

MAIN_TOUR_CATEGORIES = {"ATP", "WTA"}


def _load_ll_data() -> dict:
    try:
        with open(LL_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "updated_at": "",
            "tournaments": {},   # tourn_key -> {withdrawals, qualifying_seeds, ...}
            "fade_picks": [],    # active and resolved fade picks
        }


def _save_ll_data(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.utcnow().isoformat()
    with open(LL_FILE, "w") as f:
        json.dump(data, f, indent=2)


async def _api_get(client: httpx.AsyncClient, endpoint: str) -> dict | list | None:
    """Make SofaScore RapidAPI request with usage tracking."""
    url = f"https://{RAPIDAPI_HOST}/api/sofascore/v1{endpoint}"
    try:
        usage_file = DATA_DIR / "sofascore_usage.json"
        try:
            with open(usage_file, "r") as f:
                usage = json.load(f)
            month = datetime.now().strftime("%Y-%m")
            if usage.get("month") != month:
                usage = {"month": month, "count": 0}
        except (FileNotFoundError, json.JSONDecodeError):
            usage = {"month": datetime.now().strftime("%Y-%m"), "count": 0}

        if usage["count"] >= 300000:
            logger.warning("SofaScore API limit reached")
            return None

        resp = await client.get(url, headers=HEADERS, timeout=30)
        usage["count"] += 1
        with open(usage_file, "w") as f:
            json.dump(usage, f)

        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        logger.debug(f"LL API error: {e}")
        return None


def _parse_seed(seed_val) -> int | None:
    if not seed_val:
        return None
    try:
        return int(str(seed_val).replace("[", "").replace("]", "").strip())
    except (ValueError, TypeError):
        return None


def _parent_tournament_name(ev: dict) -> str:
    """Extract parent tournament name from qualifying event."""
    ut_name = ev.get("uniqueTournament", {}).get("name", "")
    tournament_name = ev.get("tournament", {}).get("name", "")
    name = ut_name or tournament_name
    for suffix in [" - Qualifying", " Qualifying", " - Qual", ", Qualifying"]:
        name = name.replace(suffix, "")
    return name.strip()


def _is_qualifying(ev: dict) -> bool:
    """Check if event is ATP/WTA/GS qualifying match."""
    cat = ev.get("tournament", {}).get("category", {}).get("name", "")
    if cat not in MAIN_TOUR_CATEGORIES:
        return False

    slug = ev.get("slug", "")
    season_name = ev.get("season", {}).get("name", "")
    tournament_name = ev.get("tournament", {}).get("name", "")
    ut_name = ev.get("uniqueTournament", {}).get("name", "")
    combined = f"{slug} {season_name} {tournament_name} {ut_name}".lower()

    if "qualifying" not in combined and "qual" not in combined:
        return False

    skip_words = ["double", "challenger", "itf", "futures"]
    return not any(w in combined for w in skip_words)


def _is_main_draw(ev: dict) -> bool:
    """Check if event is ATP/WTA/GS main draw match (not qualifying)."""
    cat = ev.get("tournament", {}).get("category", {}).get("name", "")
    if cat not in MAIN_TOUR_CATEGORIES:
        return False

    slug = ev.get("slug", "")
    season_name = ev.get("season", {}).get("name", "")
    tournament_name = ev.get("tournament", {}).get("name", "")
    ut_name = ev.get("uniqueTournament", {}).get("name", "")
    combined = f"{slug} {season_name} {tournament_name} {ut_name}".lower()

    # Main draw = NOT qualifying, NOT doubles, NOT challenger
    skip_words = ["double", "challenger", "itf", "futures", "qualifying", "qual"]
    return not any(w in combined for w in skip_words)


async def scan_qualifying(days: int = 3) -> dict:
    """Full LL scan: count main draw withdrawals + track qualifying seeds.

    1. Scan main draw R1 for walkovers/byes → count withdrawals per tournament
    2. Scan qualifying for seeded players
    3. Generate fade picks for Q seeds 1..N where N = number of withdrawals
    4. Resolve finished qualifying matches

    Returns LL data with fade_picks and withdrawal counts.
    """
    ll_data = _load_ll_data()

    async with httpx.AsyncClient(timeout=30) as client:
        # Scan multiple days (qualifying often starts before main draw)
        for d in range(-1, days):  # include yesterday for results
            date_str = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            data = await _api_get(client, f"/match/list?sport_slug=tennis&date={date_str}")
            if not data:
                continue

            events = data if isinstance(data, list) else data.get("events", [])

            for ev in events:
                try:
                    _process_event(ev, ll_data, date_str)
                except Exception as e:
                    logger.debug(f"LL event error: {e}")
                    continue

    # Now generate fade picks based on withdrawal count
    _generate_fade_picks(ll_data)

    # Clean old data (>14 days)
    _cleanup(ll_data)

    _save_ll_data(ll_data)
    _log_stats(ll_data)

    return ll_data


def _process_event(ev: dict, ll_data: dict, date_str: str):
    """Process a single SofaScore event - either main draw or qualifying."""
    cat = ev.get("tournament", {}).get("category", {}).get("name", "")

    if _is_main_draw(ev):
        _process_main_draw_event(ev, ll_data, date_str, cat)
    elif _is_qualifying(ev):
        _process_qualifying_event(ev, ll_data, date_str, cat)


def _get_tournament_data(ll_data: dict, tourn_key: str, parent_tourn: str,
                         cat: str, date_str: str) -> dict:
    """Get or create tournament tracking data."""
    if tourn_key not in ll_data["tournaments"]:
        ll_data["tournaments"][tourn_key] = {
            "tournament": parent_tourn,
            "tour": cat,
            "date": date_str,
            "withdrawals": [],       # players who withdrew from main draw
            "withdrawal_count": 0,
            "qualifying_seeds": {},  # name -> {seed, rank, status}
            "qualifying_matches": [], # finished Q matches
        }
    return ll_data["tournaments"][tourn_key]


def _process_main_draw_event(ev: dict, ll_data: dict, date_str: str, cat: str):
    """Check main draw R1 for walkovers/byes → count withdrawals."""
    rnd = ev.get("round", {}).get("name", "")

    # Only interested in R1 / Round of 128/64/32 for withdrawal detection
    r1_names = {"1st Round", "Round of 128", "Round of 64", "Round of 32", "1st round"}
    if rnd not in r1_names:
        return

    status = ev.get("status", {})
    status_desc = (status.get("description", "") or "").lower()
    # Comment field sometimes has "Walkover" info
    # Also check if match was cancelled or has walkover status

    ut_name = ev.get("uniqueTournament", {}).get("name", "")
    tournament_name = ev.get("tournament", {}).get("name", "")
    parent_tourn = ut_name or tournament_name
    tourn_key = parent_tourn.lower().strip()

    tourn_data = _get_tournament_data(ll_data, tourn_key, parent_tourn, cat, date_str)

    # Detect walkover in finished matches
    if status.get("isFinished"):
        home = ev.get("homeTeam", {}).get("name", "")
        away = ev.get("awayTeam", {}).get("name", "")
        home_score = ev.get("homeScore", {}).get("current", 0) or 0
        away_score = ev.get("awayScore", {}).get("current", 0) or 0

        # Walkover indicators: score 0-0, or "walkover"/"w.o." in status
        is_walkover = (
            "walkover" in status_desc or "w.o" in status_desc
            or "retired" in status_desc
            or (home_score == 0 and away_score == 0 and status.get("isFinished"))
        )

        if is_walkover:
            # The loser (0 score or WO recipient) withdrew
            if home_score == 0 and away_score > 0:
                withdrawn = home
            elif away_score == 0 and home_score > 0:
                withdrawn = away
            else:
                withdrawn = ""  # can't tell who withdrew

            if withdrawn and withdrawn not in tourn_data["withdrawals"]:
                tourn_data["withdrawals"].append(withdrawn)
                tourn_data["withdrawal_count"] = len(tourn_data["withdrawals"])
                logger.info(
                    f"  WITHDRAWAL: {withdrawn} from {parent_tourn} main draw "
                    f"(total: {tourn_data['withdrawal_count']})"
                )

    # Also detect cancelled/postponed matches that might indicate withdrawal
    elif status.get("isCancelled"):
        home = ev.get("homeTeam", {}).get("name", "")
        away = ev.get("awayTeam", {}).get("name", "")
        # If a match is cancelled, we note it but can't be sure who withdrew
        # until we see more data


def _process_qualifying_event(ev: dict, ll_data: dict, date_str: str, cat: str):
    """Track qualifying seeds and their match results."""
    parent_tourn = _parent_tournament_name(ev)
    tourn_key = parent_tourn.lower().strip()

    tourn_data = _get_tournament_data(ll_data, tourn_key, parent_tourn, cat, date_str)

    home = ev.get("homeTeam", {})
    away = ev.get("awayTeam", {})
    home_name = home.get("name", "")
    away_name = away.get("name", "")
    if not home_name or not away_name:
        return

    home_seed = _parse_seed(home.get("seed") or home.get("seeding"))
    away_seed = _parse_seed(away.get("seed") or away.get("seeding"))
    home_rank = home.get("ranking") or 0
    away_rank = away.get("ranking") or 0

    rnd = ev.get("round", {}).get("name", "")
    status = ev.get("status", {})
    is_finished = status.get("isFinished", False)
    match_id = ev.get("id")

    # Register seeds 1-4
    for name, seed, rank in [
        (home_name, home_seed, home_rank),
        (away_name, away_seed, away_rank),
    ]:
        if seed and seed <= 4 and name:
            if name not in tourn_data["qualifying_seeds"]:
                tourn_data["qualifying_seeds"][name] = {
                    "seed": seed,
                    "rank": rank,
                    "status": "active",  # active / eliminated / qualified
                }

    is_upcoming = not is_finished and not status.get("isCancelled", False)

    # Generate fade picks for UPCOMING matches where a tankable seed plays
    if is_upcoming:
        seed_player = None
        seed_num = None
        seed_rank = 0
        opponent = None
        opponent_rank = 0

        if home_seed and home_seed <= 4:
            seed_player, seed_num, seed_rank = home_name, home_seed, home_rank
            opponent, opponent_rank = away_name, away_rank
        elif away_seed and away_seed <= 4:
            seed_player, seed_num, seed_rank = away_name, away_seed, away_rank
            opponent, opponent_rank = home_name, home_rank

        if seed_player:
            _add_fade_pick_if_tankable(
                ll_data, tourn_key, seed_player, seed_num, seed_rank,
                opponent, opponent_rank, rnd, date_str, match_id, cat,
            )

    # Track finished qualifying matches involving seeds
    if is_finished:
        home_score = ev.get("homeScore", {})
        away_score = ev.get("awayScore", {})
        h_sets = home_score.get("current", 0) or 0
        a_sets = away_score.get("current", 0) or 0

        if h_sets == a_sets:
            return

        winner = home_name if h_sets > a_sets else away_name
        loser = away_name if h_sets > a_sets else home_name
        score = f"{max(h_sets, a_sets)}-{min(h_sets, a_sets)}"

        # Update seed status
        if loser in tourn_data["qualifying_seeds"]:
            tourn_data["qualifying_seeds"][loser]["status"] = "eliminated"
            tourn_data["qualifying_seeds"][loser]["lost_in"] = rnd

        # Record match for history
        match_rec = {
            "round": rnd, "winner": winner, "loser": loser,
            "score": score, "date": date_str,
        }
        # Dedupe
        existing = [
            m for m in tourn_data["qualifying_matches"]
            if m["winner"] == winner and m["loser"] == loser and m["date"] == date_str
        ]
        if not existing:
            tourn_data["qualifying_matches"].append(match_rec)

        # Resolve any open fade picks
        for pick in ll_data["fade_picks"]:
            if pick["status"] != "OPEN":
                continue
            if (pick["seed_player"] == loser or pick["seed_player"] == winner) \
                    and pick["opponent"] in (winner, loser) \
                    and pick["tournament"].lower() == parent_tourn.lower():
                if winner == pick["opponent"]:
                    pick["status"] = "WIN"
                    pick["result"] = f"{winner} d. {loser} {score}"
                else:
                    pick["status"] = "LOSS"
                    pick["result"] = f"{winner} d. {loser} {score}"
                logger.info(f"  LL FADE resolved: {pick['status']} - {pick.get('result')}")


def _generate_fade_picks(ll_data: dict):
    """Generate fade picks based on withdrawal count.

    For each tournament:
    - N withdrawals = Q seeds 1..N are likely to tank
    - Find their upcoming qualifying matches
    - Generate FADE (bet on opponent) picks
    """
    for tourn_key, tourn_data in ll_data["tournaments"].items():
        n_withdrawals = tourn_data.get("withdrawal_count", 0)
        if n_withdrawals == 0:
            continue

        # Seeds 1..N are tankable (N = withdrawal count, max 4)
        tankable_seeds = min(n_withdrawals, 4)

        for name, seed_info in tourn_data.get("qualifying_seeds", {}).items():
            seed_num = seed_info["seed"]
            if seed_num > tankable_seeds:
                continue
            if seed_info["status"] != "active":
                continue  # already eliminated or qualified

            # This seed is likely tanking - but we need their upcoming match
            # The match info comes from _process_qualifying_event for upcoming matches
            # We mark these seeds as "tankable" so upcoming match scan picks them up

            seed_info["tankable"] = True
            seed_info["reason"] = (
                f"{n_withdrawals} main draw withdrawal(s) → "
                f"Q seed {seed_num} likely gets LL spot → tanking qualifying"
            )

    # Now scan for upcoming Q matches where tankable seeds play
    # These are generated in _process_qualifying_event_upcoming
    # We need to revisit the data - fade picks are generated for tankable seeds
    # whose upcoming matches we've seen
    # (This happens during event scanning in scan_qualifying)


def _add_fade_pick_if_tankable(ll_data: dict, tourn_key: str, seed_player: str,
                                seed_num: int, seed_rank: int, opponent: str,
                                opponent_rank: int, rnd: str, date_str: str,
                                match_id: int, cat: str):
    """Add a fade pick if this Q seed is tankable (based on withdrawal count)."""
    tourn_data = ll_data["tournaments"].get(tourn_key, {})
    n_withdrawals = tourn_data.get("withdrawal_count", 0)

    if n_withdrawals == 0 or seed_num > min(n_withdrawals, 4):
        return  # Not enough withdrawals for this seed to get LL

    parent_tourn = tourn_data.get("tournament", tourn_key)

    pick_key = f"{opponent}_vs_{seed_player}_{date_str}_{tourn_key}"
    existing = [p for p in ll_data["fade_picks"] if p.get("pick_key") == pick_key]
    if existing:
        return

    ll_data["fade_picks"].append({
        "pick_key": pick_key,
        "date": date_str,
        "tournament": parent_tourn,
        "tour": cat,
        "round": rnd,
        "seed_player": seed_player,
        "seed_num": seed_num,
        "seed_rank": seed_rank,
        "opponent": opponent,
        "opponent_rank": opponent_rank,
        "match_id": match_id,
        "status": "OPEN",
        "bet_type": "LL_FADE",
        "withdrawals": n_withdrawals,
        "reason": (
            f"LL FADE: {n_withdrawals} main draw WD → Q seed {seed_num} "
            f"({seed_player}) has LL ticket → likely tanks vs {opponent}. "
            f"Bet: {opponent} WIN."
        ),
    })
    logger.info(
        f"  LL FADE pick: {opponent} vs [Q{seed_num}] {seed_player} "
        f"({parent_tourn} {rnd}, {n_withdrawals} WDs)"
    )


def _cleanup(ll_data: dict):
    """Remove data older than 14 days."""
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    ll_data["fade_picks"] = [
        p for p in ll_data["fade_picks"] if p.get("date", "") >= cutoff
    ]
    old_keys = [
        k for k, v in ll_data["tournaments"].items()
        if v.get("date", "") < cutoff
    ]
    for k in old_keys:
        del ll_data["tournaments"][k]


def _log_stats(ll_data: dict):
    """Log summary stats."""
    picks = ll_data.get("fade_picks", [])
    open_picks = [p for p in picks if p["status"] == "OPEN"]
    wins = [p for p in picks if p["status"] == "WIN"]
    losses = [p for p in picks if p["status"] == "LOSS"]

    tourns_with_wd = [
        k for k, v in ll_data["tournaments"].items()
        if v.get("withdrawal_count", 0) > 0
    ]

    logger.info(
        f"LL system: {len(tourns_with_wd)} tournaments with withdrawals, "
        f"{len(open_picks)} open fades, {len(wins)}W/{len(losses)}L"
    )
    for t_key in tourns_with_wd:
        t = ll_data["tournaments"][t_key]
        logger.info(
            f"  {t['tournament']}: {t['withdrawal_count']} WDs, "
            f"{len(t.get('qualifying_seeds', {}))} Q seeds tracked"
        )


# ─── Public API ────────────────────────────────────────────

def get_open_fade_picks() -> list[dict]:
    """Get currently open LL fade picks for dashboard / smart picks."""
    ll_data = _load_ll_data()
    return [p for p in ll_data.get("fade_picks", []) if p["status"] == "OPEN"]


def get_fade_stats() -> dict:
    """Get LL fade performance stats."""
    ll_data = _load_ll_data()
    picks = ll_data.get("fade_picks", [])
    wins = sum(1 for p in picks if p["status"] == "WIN")
    losses = sum(1 for p in picks if p["status"] == "LOSS")
    total_resolved = wins + losses

    return {
        "total_picks": len(picks),
        "open": sum(1 for p in picks if p["status"] == "OPEN"),
        "wins": wins,
        "losses": losses,
        "winrate": round(wins / total_resolved * 100, 1) if total_resolved else 0,
    }


def get_qualifying_seeds_summary() -> list[dict]:
    """Get summary of tracked qualifying seeds + withdrawal info."""
    ll_data = _load_ll_data()
    summary = []

    for tourn_key, tourn_data in ll_data.get("tournaments", {}).items():
        n_wd = tourn_data.get("withdrawal_count", 0)
        for name, info in tourn_data.get("qualifying_seeds", {}).items():
            tankable = info["seed"] <= min(n_wd, 4) if n_wd > 0 else False
            summary.append({
                "player": name,
                "tournament": tourn_data["tournament"],
                "tour": tourn_data["tour"],
                "seed": info["seed"],
                "rank": info.get("rank", 0),
                "status": info.get("status", "active"),
                "lost_in": info.get("lost_in", ""),
                "tankable": tankable,
                "withdrawals": n_wd,
            })

    return sorted(summary, key=lambda x: (x["tournament"], x["seed"]))


def check_ll_risk(player_name: str) -> dict | None:
    """Check if a player entered main draw as Lucky Loser."""
    ll_data = _load_ll_data()

    for tourn_key, tourn_data in ll_data.get("tournaments", {}).items():
        for name, info in tourn_data.get("qualifying_seeds", {}).items():
            if _names_match(player_name, name) and info.get("status") == "eliminated":
                n_wd = tourn_data.get("withdrawal_count", 0)
                if info["seed"] <= min(n_wd, 4) and n_wd > 0:
                    return {
                        "is_ll_candidate": True,
                        "tournament": tourn_data["tournament"],
                        "seed": info["seed"],
                        "rank": info.get("rank", 0),
                        "lost_in_round": info.get("lost_in", ""),
                        "warning": (
                            f"LL entry: Q seed {info['seed']} in "
                            f"{tourn_data['tournament']}, lost in "
                            f"{info.get('lost_in', '?')}. "
                            f"Likely tanked ({n_wd} main draw WDs). "
                            f"Recent form unreliable."
                        ),
                    }
    return None


def get_ll_warnings_for_matches(matches: list) -> dict:
    """Check upcoming main draw matches for LL players."""
    warnings = {}
    for m in matches:
        for pkey in ("player1", "player2"):
            pname = m.get(pkey, "")
            if not pname:
                continue
            risk = check_ll_risk(pname)
            if risk:
                warnings[pname] = risk
    return warnings


def _names_match(name1: str, name2: str) -> bool:
    """Flexible player name matching."""
    def normalize(s):
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.lower().replace(".", "").replace("-", " ").strip()

    n1 = normalize(name1)
    n2 = normalize(name2)

    if n1 == n2:
        return True

    parts1 = n1.split()
    parts2 = n2.split()
    if parts1 and parts2:
        last1 = parts1[-1] if len(parts1[-1]) >= 4 else parts1[0]
        last2 = parts2[-1] if len(parts2[-1]) >= 4 else parts2[0]
        if last1 == last2 and len(last1) >= 4:
            return True
        if parts1[0] == parts2[-1] and len(parts1[0]) >= 4:
            return True
        if parts2[0] == parts1[-1] and len(parts2[0]) >= 4:
            return True

    return False
