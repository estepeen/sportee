"""Polymarket integration - real CS2 odds via Gamma + CLOB API."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ODDS_FILE = DATA_DIR / "polymarket_odds.json"

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
CS2_TAG = "100780"
TENNIS_TAG = "864"
TENNIS_ODDS_FILE = DATA_DIR / "tennis_odds.json"

MAIN_TENNIS_KEYWORDS = [
    "miami", "indian wells", "rome", "madrid", "roland garros",
    "wimbledon", "us open", "australian open", "monte carlo",
    "canadian open", "cincinnati", "shanghai", "paris",
    "atp finals", "wta finals", "queens", "halle", "beijing",
    "dubai", "doha", "barcelona",
]


class PolymarketClient:
    """Fetches real CS2 odds from Polymarket."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15)

    async def fetch_cs2_markets(self) -> list[dict]:
        """Fetch all active CS2 markets with odds."""
        logger.info("Fetching Polymarket CS2 markets...")

        resp = await self.client.get(f"{GAMMA_URL}/events", params={
            "tag_id": CS2_TAG,
            "active": "true",
            "closed": "false",
            "limit": 100,
        })
        if resp.status_code != 200:
            logger.warning(f"Gamma API error: {resp.status_code}")
            return []

        events = resp.json()
        all_markets = []

        for event in events:
            title = event.get("title", "")
            for market in event.get("markets", []):
                question = market.get("question", "")
                outcomes = json.loads(market.get("outcomes", "[]")) if isinstance(market.get("outcomes"), str) else market.get("outcomes", [])
                prices = json.loads(market.get("outcomePrices", "[]")) if isinstance(market.get("outcomePrices"), str) else market.get("outcomePrices", [])
                tokens = json.loads(market.get("clobTokenIds", "[]")) if isinstance(market.get("clobTokenIds"), str) else market.get("clobTokenIds", [])

                if len(outcomes) < 2 or len(prices) < 2:
                    continue

                # Parse team names and odds
                market_data = {
                    "event": title,
                    "question": question,
                    "outcomes": [],
                    "market_type": self._detect_market_type(question, outcomes),
                    "updated_at": datetime.utcnow().isoformat(),
                }

                for outcome, price, token_id in zip(outcomes, prices, tokens):
                    p = float(price)
                    market_data["outcomes"].append({
                        "name": outcome,
                        "price": p,
                        "odds": round(1 / p, 2) if p > 0.01 else 99.0,
                        "token_id": token_id,
                    })

                all_markets.append(market_data)

        # Save to file
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(ODDS_FILE, "w") as f:
            json.dump(all_markets, f, indent=2)

        logger.info(f"Fetched {len(all_markets)} CS2 markets from Polymarket")
        return all_markets

    def _detect_market_type(self, question: str, outcomes: list) -> str:
        q = question.lower()
        outcome_names = [o.lower() for o in outcomes]
        has_yes_no = "yes" in outcome_names or "no" in outcome_names

        if "handicap" in q:
            return "handicap"
        elif "o/u" in q or "over" in q or "under" in q or "2.5" in q or "total" in q:
            return "over_under"
        elif "odd" in outcome_names or "even" in outcome_names:
            return "odd_even"
        elif "map 1" in q or "map 2" in q or "map 3" in q or "map 4" in q:
            return "map"
        elif has_yes_no and ("win" in q):
            # "Will X win BLAST?" = tournament winner
            return "tournament_winner"
        elif not has_yes_no and "vs" in q:
            # "Counter-Strike: Team A vs Team B" = match winner
            return "match_winner"
        elif not has_yes_no:
            return "match_winner"
        else:
            return "other"

    @staticmethod
    def load_odds() -> list[dict]:
        """Load saved odds from file."""
        try:
            with open(ODDS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    @staticmethod
    def find_handicap_odds(team1: str, team2: str, odds_data: list[dict] = None) -> dict | None:
        """Find Polymarket +1.5 map handicap odds for a matchup."""
        if odds_data is None:
            odds_data = PolymarketClient.load_odds()

        t1n = PolymarketClient._normalize(team1)
        t2n = PolymarketClient._normalize(team2)

        for market in odds_data:
            if market.get("market_type") != "handicap":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            odds_vals = [o.get("odds", 0) for o in outcomes]
            if any(o >= 50 for o in odds_vals):
                continue

            q = PolymarketClient._normalize(market.get("question", ""))
            if (t1n[:4] in q or t2n[:4] in q):
                result = {"question": market.get("question", "")}
                for o in outcomes:
                    n = PolymarketClient._normalize(o["name"])
                    if PolymarketClient._normalize(team1)[:4] in n or n[:4] in PolymarketClient._normalize(team1):
                        result["team1_hc_price"] = o["price"]
                        result["team1_hc_odds"] = o["odds"]
                    elif PolymarketClient._normalize(team2)[:4] in n or n[:4] in PolymarketClient._normalize(team2):
                        result["team2_hc_price"] = o["price"]
                        result["team2_hc_odds"] = o["odds"]
                if "team1_hc_odds" in result or "team2_hc_odds" in result:
                    return result
        return None

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize team name for fuzzy matching."""
        return name.lower().replace(" ", "").replace(".", "").replace("-", "").replace("esports", "").replace("gaming", "").replace("team", "").strip()

    @staticmethod
    def find_match_odds(team1: str, team2: str, odds_data: list[dict] = None) -> dict | None:
        """Find Polymarket odds for a specific matchup (fuzzy matching)."""
        if odds_data is None:
            odds_data = PolymarketClient.load_odds()

        t1n = PolymarketClient._normalize(team1)
        t2n = PolymarketClient._normalize(team2)

        def matches(name, target):
            n = PolymarketClient._normalize(name)
            return n in target or target in n or (len(n) > 3 and len(target) > 3 and (n[:4] == target[:4]))

        for market in odds_data:
            if market.get("market_type") != "match_winner":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            # Skip finished markets (odds 1.0/99.0)
            odds_vals = [o.get("odds", 0) for o in outcomes]
            if any(o >= 50 for o in odds_vals):
                continue

            # Match by outcome names
            o_names = [o["name"] for o in outcomes]
            t1_match = any(matches(n, t1n) for n in o_names)
            t2_match = any(matches(n, t2n) for n in o_names)

            # Also try matching via question field
            if not (t1_match and t2_match):
                q = PolymarketClient._normalize(market.get("question", ""))
                t1_match = t1n in q or (len(t1n) > 3 and t1n[:5] in q)
                t2_match = t2n in q or (len(t2n) > 3 and t2n[:5] in q)

            if t1_match and t2_match:
                result = {"event": market.get("event", ""), "question": market.get("question", "")}
                # Assign odds - first outcome = team mentioned first in question
                for o in outcomes:
                    if matches(o["name"], t1n):
                        result["team1_price"] = o["price"]
                        result["team1_odds"] = o["odds"]
                    elif matches(o["name"], t2n):
                        result["team2_price"] = o["price"]
                        result["team2_odds"] = o["odds"]

                # Fallback: assign by position if name matching failed
                if "team1_odds" not in result and "team2_odds" not in result:
                    result["team1_price"] = outcomes[0]["price"]
                    result["team1_odds"] = outcomes[0]["odds"]
                    result["team2_price"] = outcomes[1]["price"]
                    result["team2_odds"] = outcomes[1]["odds"]

                return result

        return None

    async def _fetch_clob_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Fetch real-time midpoint prices from Polymarket CLOB API.

        Returns {token_id: midpoint_price} dict.
        Uses individual /midpoint calls (batch endpoint has URL length limits).
        """
        if not token_ids:
            return {}

        prices = {}

        async def _fetch_one(tid: str):
            try:
                resp = await self.client.get(
                    f"{CLOB_URL}/midpoint",
                    params={"token_id": tid},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Response: {"mid": "0.55"} or {"mid": 0.55}
                    mid = data.get("mid")
                    if mid is not None:
                        prices[tid] = float(mid)
            except Exception:
                pass

        # Fetch in parallel, 10 at a time to avoid overwhelming API
        for i in range(0, len(token_ids), 10):
            batch = token_ids[i : i + 10]
            await asyncio.gather(*[_fetch_one(tid) for tid in batch])

        return prices

    async def fetch_tennis_markets(self) -> list[dict]:
        """Fetch active tennis moneyline markets from Polymarket.

        Uses Gamma API for market discovery, then CLOB API for accurate
        real-time prices (Gamma outcomePrices can be significantly stale).
        """
        logger.info("Fetching Polymarket tennis markets...")

        resp = await self.client.get(f"{GAMMA_URL}/markets", params={
            "tag_id": TENNIS_TAG,
            "sports_market_types": "moneyline",
            "active": "true",
            "closed": "false",
            "limit": 200,
        })
        if resp.status_code != 200:
            logger.warning(f"Gamma tennis API error: {resp.status_code}")
            return []

        raw_markets = resp.json()

        # --- Phase 1: collect valid markets + their CLOB token IDs ---
        pending = []  # (market_dict, token_id_1, token_id_2)
        all_token_ids = []

        for market in raw_markets:
            question = market.get("question", "")
            q_lower = question.lower()

            # Skip doubles
            if "doubles" in q_lower or "/" in question:
                continue

            outcomes = json.loads(market.get("outcomes", "[]")) if isinstance(market.get("outcomes"), str) else market.get("outcomes", [])
            if len(outcomes) < 2:
                continue

            # Get CLOB token IDs for real-time prices
            clob_raw = market.get("clobTokenIds", "[]")
            clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else (clob_raw or [])

            if len(clob_ids) >= 2:
                all_token_ids.extend(clob_ids[:2])
                pending.append((market, clob_ids[0], clob_ids[1]))
            else:
                # No CLOB IDs — fall back to Gamma prices
                pending.append((market, None, None))

        # --- Phase 2: batch-fetch real CLOB midpoint prices ---
        clob_prices = await self._fetch_clob_prices(all_token_ids)
        logger.info(f"CLOB prices fetched for {len(clob_prices)}/{len(all_token_ids)} tokens")

        # --- Phase 3: build match list with best available prices ---
        matches = []
        for market, tid1, tid2 in pending:
            question = market.get("question", "")
            q_lower = question.lower()

            outcomes = json.loads(market.get("outcomes", "[]")) if isinstance(market.get("outcomes"), str) else market.get("outcomes", [])
            if len(outcomes) < 2:
                continue

            # Prefer CLOB midpoint prices, fall back to Gamma outcomePrices
            if tid1 and tid2 and tid1 in clob_prices and tid2 in clob_prices:
                p1, p2 = clob_prices[tid1], clob_prices[tid2]
                price_source = "clob"
            else:
                gamma_prices = json.loads(market.get("outcomePrices", "[]")) if isinstance(market.get("outcomePrices"), str) else market.get("outcomePrices", [])
                if len(gamma_prices) < 2:
                    continue
                p1, p2 = float(gamma_prices[0]), float(gamma_prices[1])
                price_source = "gamma"

            if p1 >= 0.95 or p2 >= 0.95:
                continue
            if p1 <= 0.01 or p2 <= 0.01:
                continue

            # Detect tournament
            tournament = market.get("groupItemTitle", "")
            if not tournament:
                for kw in MAIN_TENNIS_KEYWORDS:
                    if kw in q_lower:
                        tournament = kw.title()
                        break

            # Detect tour (ATP/WTA)
            tour = ""
            if "atp" in q_lower or "men" in q_lower:
                tour = "ATP"
            elif "wta" in q_lower or "women" in q_lower:
                tour = "WTA"

            # Build Polymarket URL from slug
            slug = market.get("slug", "")
            if not slug:
                events = market.get("events", [])
                if events and isinstance(events, list):
                    slug = events[0].get("slug", "") if isinstance(events[0], dict) else ""
            pm_url = f"https://polymarket.com/event/{slug}" if slug else ""

            matches.append({
                "question": question,
                "tournament": tournament,
                "tour": tour,
                "player1": outcomes[0],
                "player2": outcomes[1],
                "player1_price": p1,
                "player2_price": p2,
                "player1_odds": round(1 / p1, 2),
                "player2_odds": round(1 / p2, 2),
                "pm_url": pm_url,
                "price_source": price_source,
                "sport": "tennis",
                "updated_at": datetime.utcnow().isoformat(),
            })

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(TENNIS_ODDS_FILE, "w") as f:
            json.dump(matches, f, indent=2)

        clob_count = sum(1 for m in matches if m.get("price_source") == "clob")
        logger.info(f"Fetched {len(matches)} tennis matches ({clob_count} with CLOB prices, {len(matches) - clob_count} Gamma fallback)")
        return matches

    @staticmethod
    def load_tennis_odds() -> list[dict]:
        """Load saved tennis odds from file."""
        try:
            with open(TENNIS_ODDS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    async def close(self):
        await self.client.aclose()
