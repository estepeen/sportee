import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config.settings import DB_PATH


engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}", echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id = sa.Column(sa.Integer, primary_key=True)
    hltv_id = sa.Column(sa.Integer, unique=True, nullable=False)
    name = sa.Column(sa.String, nullable=False)
    ranking = sa.Column(sa.Integer)
    updated_at = sa.Column(sa.DateTime, server_default=sa.func.now())


class Match(Base):
    __tablename__ = "matches"

    id = sa.Column(sa.Integer, primary_key=True)
    hltv_id = sa.Column(sa.Integer, unique=True, nullable=False)
    date = sa.Column(sa.DateTime, nullable=False)
    team1_id = sa.Column(sa.Integer, sa.ForeignKey("teams.hltv_id"), nullable=False)
    team2_id = sa.Column(sa.Integer, sa.ForeignKey("teams.hltv_id"), nullable=False)
    team1_score = sa.Column(sa.Integer)
    team2_score = sa.Column(sa.Integer)
    winner_id = sa.Column(sa.Integer, sa.ForeignKey("teams.hltv_id"))
    best_of = sa.Column(sa.Integer)  # 1, 3, 5
    event_name = sa.Column(sa.String)
    event_tier = sa.Column(sa.String)  # S, A, B, C
    is_lan = sa.Column(sa.Boolean)
    is_completed = sa.Column(sa.Boolean, default=False)


class MapResult(Base):
    __tablename__ = "map_results"

    id = sa.Column(sa.Integer, primary_key=True)
    match_id = sa.Column(sa.Integer, sa.ForeignKey("matches.hltv_id"), nullable=False)
    map_name = sa.Column(sa.String, nullable=False)  # mirage, inferno, etc.
    map_number = sa.Column(sa.Integer, nullable=False)  # 1, 2, 3
    team1_ct_rounds = sa.Column(sa.Integer)
    team1_t_rounds = sa.Column(sa.Integer)
    team2_ct_rounds = sa.Column(sa.Integer)
    team2_t_rounds = sa.Column(sa.Integer)
    team1_score = sa.Column(sa.Integer)
    team2_score = sa.Column(sa.Integer)
    winner_id = sa.Column(sa.Integer, sa.ForeignKey("teams.hltv_id"))
    picked_by = sa.Column(sa.Integer, sa.ForeignKey("teams.hltv_id"))  # who picked this map


class TeamMapStats(Base):
    __tablename__ = "team_map_stats"

    id = sa.Column(sa.Integer, primary_key=True)
    team_id = sa.Column(sa.Integer, sa.ForeignKey("teams.hltv_id"), nullable=False)
    map_name = sa.Column(sa.String, nullable=False)
    matches_played = sa.Column(sa.Integer, default=0)
    wins = sa.Column(sa.Integer, default=0)
    ct_winrate = sa.Column(sa.Float)
    t_winrate = sa.Column(sa.Float)
    avg_rounds_won = sa.Column(sa.Float)
    period_months = sa.Column(sa.Integer, default=3)  # stats over last N months
    updated_at = sa.Column(sa.DateTime, server_default=sa.func.now())

    __table_args__ = (
        sa.UniqueConstraint("team_id", "map_name", "period_months"),
    )


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = sa.Column(sa.Integer, primary_key=True)
    match_id = sa.Column(sa.Integer, sa.ForeignKey("matches.hltv_id"), nullable=False)
    bookmaker = sa.Column(sa.String, nullable=False)
    market_type = sa.Column(sa.String, nullable=False)  # ml, map_handicap, over_under
    team1_odds = sa.Column(sa.Float)
    team2_odds = sa.Column(sa.Float)
    line = sa.Column(sa.Float)  # e.g. -1.5 for handicap, 2.5 for over/under
    captured_at = sa.Column(sa.DateTime, server_default=sa.func.now())
    is_closing = sa.Column(sa.Boolean, default=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = sa.Column(sa.Integer, primary_key=True)
    match_id = sa.Column(sa.Integer, sa.ForeignKey("matches.hltv_id"), nullable=False)
    market_type = sa.Column(sa.String, nullable=False)
    predicted_prob = sa.Column(sa.Float, nullable=False)  # our model's probability
    bookmaker_prob = sa.Column(sa.Float)  # implied probability from odds
    edge = sa.Column(sa.Float)  # predicted_prob - bookmaker_prob
    bet_side = sa.Column(sa.String)  # team1, team2, over, under
    stake = sa.Column(sa.Float)
    odds_taken = sa.Column(sa.Float)
    result = sa.Column(sa.String)  # win, loss, push, pending
    profit = sa.Column(sa.Float)
    created_at = sa.Column(sa.DateTime, server_default=sa.func.now())


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
