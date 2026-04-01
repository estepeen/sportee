"""HLTV scraper using curl_cffi to bypass Cloudflare for CS2 match and team data."""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
from sqlalchemy import select

from src.database import (
    async_session, Team, Match, MapResult, TeamMapStats, init_db
)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.hltv.org"


class HLTVScraper:
    def __init__(self, delay: float = 3.0):
        self.delay = delay

    def _get(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page using curl_cffi with Chrome TLS impersonation."""
        try:
            resp = cffi_requests.get(url, impersonate="chrome", timeout=30)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            logger.warning(f"HTTP {resp.status_code} for {url}")
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
        return None

    async def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Async wrapper with rate limiting."""
        loop = asyncio.get_event_loop()
        soup = await loop.run_in_executor(None, self._get, url)
        await asyncio.sleep(self.delay)
        return soup

    # ─── Teams ───────────────────────────────────────────────

    async def scrape_top_teams(self, count: int = 100) -> list[dict]:
        """Scrape top ranked teams."""
        logger.info(f"Scraping top {count} teams...")
        soup = await self._fetch(f"{BASE_URL}/ranking/teams")
        if not soup:
            return []

        teams = []
        for el in soup.select(".ranked-team")[:count]:
            name_el = el.select_one(".name")
            rank_el = el.select_one(".position")
            link_el = el.select_one("a.moreLink")
            if not name_el:
                continue

            name = name_el.text.strip()
            ranking = int(rank_el.text.strip().replace("#", "")) if rank_el else 999
            team_id = None
            if link_el and link_el.get("href"):
                m = re.search(r"/team/(\d+)/", link_el["href"])
                if m:
                    team_id = int(m.group(1))

            if team_id:
                teams.append({"id": team_id, "name": name, "ranking": ranking})

        async with async_session() as session:
            for td in teams:
                existing = await session.execute(
                    select(Team).where(Team.hltv_id == td["id"])
                )
                team = existing.scalar_one_or_none()
                if team:
                    team.name = td["name"]
                    team.ranking = td["ranking"]
                    team.updated_at = datetime.utcnow()
                else:
                    session.add(Team(hltv_id=td["id"], name=td["name"], ranking=td["ranking"]))
            await session.commit()

        logger.info(f"Saved {len(teams)} teams")
        return teams

    # ─── Results ─────────────────────────────────────────────

    async def scrape_results(self, pages: int = 3) -> list[dict]:
        """Scrape completed match results."""
        logger.info(f"Scraping results ({pages} pages)...")
        all_results = []

        for page in range(pages):
            offset = page * 100
            soup = await self._fetch(f"{BASE_URL}/results?offset={offset}")
            if not soup:
                continue

            for result in soup.select(".result-con"):
                link = result.select_one("a.a-reset")
                if not link or not link.get("href"):
                    continue

                mid = re.search(r"/matches/(\d+)/", link["href"])
                if not mid:
                    continue

                teams = result.select(".team")
                if len(teams) < 2:
                    continue

                # Score: .score-won and .score-lost
                score_won = result.select_one(".score-won")
                score_lost = result.select_one(".score-lost")
                result_score = result.select_one(".result-score")

                t1_name = teams[0].text.strip()
                t2_name = teams[1].text.strip()

                # Determine scores - check if team1 won or lost
                t1_score = t2_score = None
                if result_score:
                    score_text = result_score.text.strip()
                    score_parts = score_text.split("-")
                    if len(score_parts) == 2:
                        try:
                            t1_score = int(score_parts[0].strip())
                            t2_score = int(score_parts[1].strip())
                        except ValueError:
                            pass

                # Best of from map text
                map_el = result.select_one(".map-text")
                best_of = 3
                if map_el:
                    bo_text = map_el.text.strip().lower()
                    if "bo1" in bo_text:
                        best_of = 1
                    elif "bo3" in bo_text:
                        best_of = 3
                    elif "bo5" in bo_text:
                        best_of = 5

                event_el = result.select_one(".event-name")
                event = event_el.text.strip() if event_el else ""

                # Stars
                stars = len(result.select(".star"))

                all_results.append({
                    "id": int(mid.group(1)),
                    "team1_name": t1_name,
                    "team2_name": t2_name,
                    "team1_score": t1_score,
                    "team2_score": t2_score,
                    "best_of": best_of,
                    "event": event,
                    "stars": stars,
                    "url": link["href"],
                })

        # Save to DB
        saved = 0
        async with async_session() as session:
            for md in all_results:
                existing = await session.execute(
                    select(Match).where(Match.hltv_id == md["id"])
                )
                if existing.scalar_one_or_none():
                    continue

                t1 = await session.execute(select(Team).where(Team.name == md["team1_name"]))
                t2 = await session.execute(select(Team).where(Team.name == md["team2_name"]))
                t1_obj = t1.scalars().first()
                t2_obj = t2.scalars().first()
                if not t1_obj or not t2_obj:
                    continue

                winner_id = None
                if md["team1_score"] is not None and md["team2_score"] is not None:
                    if md["team1_score"] > md["team2_score"]:
                        winner_id = t1_obj.hltv_id
                    elif md["team2_score"] > md["team1_score"]:
                        winner_id = t2_obj.hltv_id

                session.add(Match(
                    hltv_id=md["id"],
                    date=datetime.utcnow(),
                    team1_id=t1_obj.hltv_id,
                    team2_id=t2_obj.hltv_id,
                    team1_score=md["team1_score"],
                    team2_score=md["team2_score"],
                    winner_id=winner_id,
                    best_of=md["best_of"],
                    event_name=md["event"],
                    is_completed=True,
                ))
                saved += 1

            await session.commit()

        logger.info(f"Saved {saved} new matches from {len(all_results)} results")
        return all_results

    # ─── Match Details (maps, CT/T rounds) ───────────────────

    async def scrape_match_details(self, match_id: int, url: str = "") -> Optional[dict]:
        """Scrape map-level results including CT/T half scores."""
        full_url = f"{BASE_URL}{url}" if url else f"{BASE_URL}/matches/{match_id}/-"
        logger.info(f"Match details: {match_id}...")
        soup = await self._fetch(full_url)
        if not soup:
            return None

        details = {"id": match_id, "maps": []}

        for i, mh in enumerate(soup.select(".mapholder")):
            mapname_el = mh.select_one(".mapname")
            if not mapname_el:
                continue
            map_name = mapname_el.text.strip().lower()
            if map_name in ("tba", "default"):
                continue

            # Map scores
            score_els = mh.select(".results-team-score")
            if len(score_els) < 2:
                continue
            try:
                t1_total = int(score_els[0].text.strip())
                t2_total = int(score_els[1].text.strip())
            except ValueError:
                continue

            # CT/T half scores
            # Format: (T1_first_half : T2_first_half ; T1_second_half : T2_second_half)
            t1_ct = t1_t = t2_ct = t2_t = None
            half_el = mh.select_one(".results-center-half-score")
            if half_el:
                spans = half_el.select("span[class]")
                ct_spans = [s for s in spans if "ct" in s.get("class", [])]
                t_spans = [s for s in spans if "t" in s.get("class", [])]

                # HLTV format: team1 starts on one side
                # First half: spans[0]=t/ct, spans[1]=t/ct
                # We extract all CT and T values
                try:
                    if len(t_spans) >= 2 and len(ct_spans) >= 2:
                        # Team1: first T span + second CT span (or vice versa)
                        # The order in HTML is: T1_side1, T2_side1; T1_side2, T2_side2
                        vals = []
                        for s in spans:
                            try:
                                vals.append(int(s.text.strip()))
                            except ValueError:
                                continue

                        if len(vals) >= 4:
                            # vals = [t1_first, t2_first, t1_second, t2_second]
                            # First half: team1 was T (first span class), team2 was CT
                            first_span_class = spans[0].get("class", [])
                            if "t" in first_span_class:
                                # Team1 started T side
                                t1_t = vals[0]
                                t2_ct = vals[1]
                                t1_ct = vals[2]
                                t2_t = vals[3]
                            else:
                                # Team1 started CT side
                                t1_ct = vals[0]
                                t2_t = vals[1]
                                t1_t = vals[2]
                                t2_ct = vals[3]
                except Exception:
                    pass

            # Pick info
            pick_el = mh.select_one(".pick")
            picked_by_text = pick_el.text.strip() if pick_el else ""

            details["maps"].append({
                "name": map_name,
                "map_number": i + 1,
                "team1_score": t1_total,
                "team2_score": t2_total,
                "team1_ct": t1_ct,
                "team1_t": t1_t,
                "team2_ct": t2_ct,
                "team2_t": t2_t,
                "picked_by": picked_by_text,
            })

        # Determine best_of from veto or map count
        veto = soup.select_one(".veto-box, .standard-box.veto-box")
        best_of = None
        if veto:
            veto_text = veto.text.lower()
            if "bo5" in veto_text:
                best_of = 5
            elif "bo3" in veto_text:
                best_of = 3
            elif "bo1" in veto_text:
                best_of = 1
        if not best_of:
            best_of = max(len(details["maps"]), 1)

        # Save to DB
        async with async_session() as session:
            match_result = await session.execute(
                select(Match).where(Match.hltv_id == match_id)
            )
            match_obj = match_result.scalar_one_or_none()
            if match_obj:
                match_obj.best_of = best_of

            for md in details["maps"]:
                existing = await session.execute(
                    select(MapResult).where(
                        MapResult.match_id == match_id,
                        MapResult.map_number == md["map_number"],
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                winner_id = None
                if match_obj:
                    if md["team1_score"] > md["team2_score"]:
                        winner_id = match_obj.team1_id
                    elif md["team2_score"] > md["team1_score"]:
                        winner_id = match_obj.team2_id

                session.add(MapResult(
                    match_id=match_id,
                    map_name=md["name"],
                    map_number=md["map_number"],
                    team1_ct_rounds=md["team1_ct"],
                    team1_t_rounds=md["team1_t"],
                    team2_ct_rounds=md["team2_ct"],
                    team2_t_rounds=md["team2_t"],
                    team1_score=md["team1_score"],
                    team2_score=md["team2_score"],
                    winner_id=winner_id,
                ))

            await session.commit()

        logger.info(f"Saved {len(details['maps'])} maps for match {match_id}")
        return details

    # ─── Upcoming Matches ────────────────────────────────────

    async def scrape_upcoming_matches(self) -> list[dict]:
        """Scrape upcoming matches from HLTV matches page."""
        logger.info("Scraping upcoming matches...")
        soup = await self._fetch(f"{BASE_URL}/matches")
        if not soup:
            return []

        upcoming = []
        for wrapper in soup.select(".match-wrapper"):
            # Skip live matches
            meta = wrapper.select_one(".match-meta")
            if meta and "live" in meta.text.strip().lower():
                continue

            # Get match link
            link = wrapper.select_one("a[href*='/matches/']")
            if not link:
                continue
            href = link.get("href", "")
            mid = re.search(r"/matches/(\d+)/", href)
            if not mid:
                continue

            # Teams - inside .match-teams
            teams_el = wrapper.select_one(".match-teams")
            if not teams_el:
                continue

            # Team names are direct text nodes in team containers
            team_names = []
            for team_el in teams_el.select(".match-team-name, .team-name"):
                team_names.append(team_el.text.strip())

            # Fallback: parse from text
            if len(team_names) < 2:
                teams_text = teams_el.text.strip()
                parts = [p.strip() for p in teams_text.split("\n") if p.strip()]
                team_names = [p for p in parts if p and len(p) > 1]

            if len(team_names) < 2:
                continue

            # Best of
            bo = 3
            if meta:
                bo_text = meta.text.strip().lower()
                if "bo1" in bo_text:
                    bo = 1
                elif "bo5" in bo_text:
                    bo = 5

            # Time
            time_text = ""
            if meta:
                parts = meta.text.strip().split("\n")
                for p in parts:
                    p = p.strip()
                    if re.match(r"\d{1,2}:\d{2}", p):
                        time_text = p
                        break

            # Odds from .match-fixture
            odds_el = wrapper.select_one(".match-fixture")
            odds = {}
            if odds_el:
                odds_text = odds_el.text.strip().split("\n")
                odds_vals = [o.strip() for o in odds_text if re.match(r"[\d.]+", o.strip())]
                if len(odds_vals) >= 2:
                    try:
                        odds["team1"] = float(odds_vals[0])
                        odds["team2"] = float(odds_vals[1])
                    except ValueError:
                        pass

            # Stars
            stars = len(wrapper.select(".star, .faded"))

            # Event name from parent event-headline
            event_name = ""
            parent = wrapper.find_parent(class_="events-container")
            if parent:
                event_el = parent.select_one(".event-headline-text")
                if event_el:
                    event_name = event_el.text.strip()

            upcoming.append({
                "id": int(mid.group(1)),
                "team1_name": team_names[0],
                "team2_name": team_names[1] if len(team_names) > 1 else "",
                "best_of": bo,
                "event": event_name,
                "time": time_text,
                "odds": odds,
                "stars": stars,
                "url": href,
            })

        # Save known-team matches to DB
        saved = 0
        async with async_session() as session:
            for md in upcoming:
                existing = await session.execute(
                    select(Match).where(Match.hltv_id == md["id"])
                )
                if existing.scalar_one_or_none():
                    continue

                t1 = await session.execute(select(Team).where(Team.name == md["team1_name"]))
                t2 = await session.execute(select(Team).where(Team.name == md["team2_name"]))
                t1_obj = t1.scalars().first()
                t2_obj = t2.scalars().first()
                if not t1_obj or not t2_obj:
                    continue

                session.add(Match(
                    hltv_id=md["id"],
                    date=datetime.utcnow(),
                    team1_id=t1_obj.hltv_id,
                    team2_id=t2_obj.hltv_id,
                    best_of=md["best_of"],
                    event_name=md["event"],
                    is_completed=False,
                ))
                saved += 1

            await session.commit()

        logger.info(f"Saved {saved} upcoming matches from {len(upcoming)} found")

        # Save HLTV odds to JSON for dashboard
        import json
        odds_data = {}
        for md in upcoming:
            if md.get("odds"):
                key = f"{md['team1_name']}_vs_{md['team2_name']}"
                odds_data[key] = {
                    "team1_name": md["team1_name"],
                    "team2_name": md["team2_name"],
                    "team1_odds": md["odds"].get("team1"),
                    "team2_odds": md["odds"].get("team2"),
                    "source": "hltv",
                    "updated_at": datetime.utcnow().isoformat(),
                }
        if odds_data:
            odds_file = Path(__file__).parent.parent.parent / "data" / "hltv_odds.json"
            odds_file.parent.mkdir(parents=True, exist_ok=True)
            with open(odds_file, "w") as f:
                json.dump(odds_data, f, indent=2)
            logger.info(f"Saved HLTV odds for {len(odds_data)} matches")

        return upcoming

    # ─── Team Map Stats ──────────────────────────────────────

    async def compute_team_map_stats(self, months: int = 3):
        """Compute per-team per-map win rates from stored map results."""
        logger.info(f"Computing team map stats (last {months}mo)...")
        cutoff = datetime.utcnow() - timedelta(days=months * 30)

        async with async_session() as session:
            rows = await session.execute(
                select(MapResult, Match)
                .join(Match, MapResult.match_id == Match.hltv_id)
                .where(Match.date >= cutoff, Match.is_completed == True)
            )

            stats: dict[tuple[int, str], dict] = {}

            for map_result, match in rows:
                for team_id, is_team1 in [(match.team1_id, True), (match.team2_id, False)]:
                    if not team_id:
                        continue
                    key = (team_id, map_result.map_name)
                    if key not in stats:
                        stats[key] = {"played": 0, "wins": 0, "ct_won": 0, "ct_total": 0,
                                      "t_won": 0, "t_total": 0, "rounds_won": 0}

                    s = stats[key]
                    s["played"] += 1

                    if is_team1:
                        score, opp = map_result.team1_score or 0, map_result.team2_score or 0
                        ct, t = map_result.team1_ct_rounds or 0, map_result.team1_t_rounds or 0
                    else:
                        score, opp = map_result.team2_score or 0, map_result.team1_score or 0
                        ct, t = map_result.team2_ct_rounds or 0, map_result.team2_t_rounds or 0

                    if score > opp:
                        s["wins"] += 1
                    s["rounds_won"] += score
                    s["ct_won"] += ct
                    s["t_won"] += t
                    # For CT winrate: team won 'ct' rounds on CT side.
                    # They played 12 CT rounds in regulation (+ OT rounds).
                    # Simplification: ct_total = ct + opponent's T rounds on that half
                    # Best approach: just use the raw half rounds
                    if ct > 0 or t > 0:
                        # First half is 12 rounds, second half is 12 rounds (MR12)
                        # ct_won/12 and t_won/12 gives approximate side winrate
                        s["ct_total"] += 12
                        s["t_total"] += 12

            for (team_id, map_name), s in stats.items():
                existing = await session.execute(
                    select(TeamMapStats).where(
                        TeamMapStats.team_id == team_id,
                        TeamMapStats.map_name == map_name,
                        TeamMapStats.period_months == months,
                    )
                )
                record = existing.scalar_one_or_none()
                vals = {
                    "matches_played": s["played"],
                    "wins": s["wins"],
                    "ct_winrate": s["ct_won"] / s["ct_total"] if s["ct_total"] > 0 else None,
                    "t_winrate": s["t_won"] / s["t_total"] if s["t_total"] > 0 else None,
                    "avg_rounds_won": s["rounds_won"] / s["played"] if s["played"] > 0 else None,
                    "updated_at": datetime.utcnow(),
                }
                if record:
                    for k, v in vals.items():
                        setattr(record, k, v)
                else:
                    session.add(TeamMapStats(
                        team_id=team_id, map_name=map_name, period_months=months, **vals
                    ))

            await session.commit()

        logger.info(f"Computed stats for {len(stats)} team-map combos")

    # ─── Team History ────────────────────────────────────────

    async def scrape_team_history(self, team_id: int, team_name: str, pages: int = 3) -> list[dict]:
        """Scrape match history for a specific team."""
        logger.info(f"Scraping history for {team_name} ({pages} pages)...")
        all_matches = []

        for page in range(pages):
            offset = page * 100
            soup = await self._fetch(f"{BASE_URL}/results?team={team_id}&offset={offset}")
            if not soup:
                continue

            for result in soup.select(".result-con"):
                link = result.select_one("a.a-reset")
                if not link or not link.get("href"):
                    continue

                mid = re.search(r"/matches/(\d+)/", link["href"])
                if not mid:
                    continue

                teams = result.select(".team")
                if len(teams) < 2:
                    continue

                result_score = result.select_one(".result-score")
                t1_score = t2_score = None
                if result_score:
                    score_parts = result_score.text.strip().split("-")
                    if len(score_parts) == 2:
                        try:
                            t1_score = int(score_parts[0].strip())
                            t2_score = int(score_parts[1].strip())
                        except ValueError:
                            pass

                map_el = result.select_one(".map-text")
                best_of = 3
                if map_el:
                    bo_text = map_el.text.strip().lower()
                    if "bo1" in bo_text:
                        best_of = 1
                    elif "bo5" in bo_text:
                        best_of = 5

                event_el = result.select_one(".event-name")

                all_matches.append({
                    "id": int(mid.group(1)),
                    "team1_name": teams[0].text.strip(),
                    "team2_name": teams[1].text.strip(),
                    "team1_score": t1_score,
                    "team2_score": t2_score,
                    "best_of": best_of,
                    "event": event_el.text.strip() if event_el else "",
                    "url": link["href"],
                })

        # Save matches - ensure both teams exist in DB
        saved = 0
        async with async_session() as session:
            for md in all_matches:
                existing = await session.execute(
                    select(Match).where(Match.hltv_id == md["id"])
                )
                if existing.scalar_one_or_none():
                    continue

                # Ensure opponent team exists
                for tname in [md["team1_name"], md["team2_name"]]:
                    t = await session.execute(select(Team).where(Team.name == tname))
                    if not t.scalars().first():
                        session.add(Team(hltv_id=abs(hash(tname)) % 100000, name=tname, ranking=999))

                await session.flush()

                t1 = await session.execute(select(Team).where(Team.name == md["team1_name"]))
                t2 = await session.execute(select(Team).where(Team.name == md["team2_name"]))
                t1_obj = t1.scalars().first()
                t2_obj = t2.scalars().first()
                if not t1_obj or not t2_obj:
                    continue

                winner_id = None
                if md["team1_score"] is not None and md["team2_score"] is not None:
                    if md["team1_score"] > md["team2_score"]:
                        winner_id = t1_obj.hltv_id
                    elif md["team2_score"] > md["team1_score"]:
                        winner_id = t2_obj.hltv_id

                session.add(Match(
                    hltv_id=md["id"],
                    date=datetime.utcnow(),
                    team1_id=t1_obj.hltv_id,
                    team2_id=t2_obj.hltv_id,
                    team1_score=md["team1_score"],
                    team2_score=md["team2_score"],
                    winner_id=winner_id,
                    best_of=md["best_of"],
                    event_name=md["event"],
                    is_completed=True,
                ))
                saved += 1

            await session.commit()

        logger.info(f"Saved {saved} matches for {team_name}")
        return all_matches

    # ─── Full Pipeline ───────────────────────────────────────

    async def full_scrape(self, deep: bool = False):
        """Run complete scraping pipeline.

        Args:
            deep: If True, scrape full history for all top teams (slow but thorough).
        """
        await init_db()

        teams = await self.scrape_top_teams(count=100 if deep else 30)
        results = await self.scrape_results(pages=3)

        if deep:
            # Scrape match history for all teams
            for i, team in enumerate(teams):
                logger.info(f"[{i+1}/{len(teams)}] Processing {team['name']}...")
                team_matches = await self.scrape_team_history(
                    team["id"], team["name"], pages=3  # 300 matches per team
                )
                # Scrape map details for recent matches
                scraped_maps = 0
                for md in team_matches:
                    if scraped_maps >= 20:  # max 20 map details per team
                        break
                    if md.get("url"):
                        async with async_session() as session:
                            existing = await session.execute(
                                select(MapResult).where(MapResult.match_id == md["id"])
                            )
                            if existing.scalar_one_or_none():
                                continue
                        await self.scrape_match_details(md["id"], md["url"])
                        scraped_maps += 1
        else:
            # Quick mode - just recent results + map details
            for md in results[:30]:
                if md.get("url"):
                    await self.scrape_match_details(md["id"], md["url"])

        await self.compute_team_map_stats(months=3)
        await self.compute_team_map_stats(months=6)
        await self.scrape_upcoming_matches()

        logger.info("Full scrape completed")
