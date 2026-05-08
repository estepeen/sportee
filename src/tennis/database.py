"""Tennis SQLite database - players, matches, surface stats."""

import sqlite3
from pathlib import Path

TENNIS_DB_PATH = Path(__file__).parent.parent.parent / "data" / "tennis.db"


def get_tennis_db():
    conn = sqlite3.connect(TENNIS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_tennis_db():
    """Create tennis database tables."""
    conn = get_tennis_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS tennis_players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        hand TEXT DEFAULT 'R',           -- L=left, R=right, U=unknown
        style TEXT DEFAULT 'universal',  -- aggressive, defensive, universal, serve_based
        backhand TEXT DEFAULT 'two',     -- one, two (one-handed vs two-handed)
        country TEXT DEFAULT '',
        birth_year INTEGER DEFAULT 0,
        height_cm INTEGER DEFAULT 0,
        notes TEXT DEFAULT ''            -- manual notes: injuries, coach changes etc.
    );

    CREATE TABLE IF NOT EXISTS tennis_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        tournament TEXT NOT NULL,
        location TEXT DEFAULT '',
        surface TEXT NOT NULL,           -- Hard, Clay, Grass, Carpet
        indoor INTEGER DEFAULT 0,
        court_speed TEXT DEFAULT '',     -- slow, medium, fast (derived from surface+tournament)
        round TEXT DEFAULT '',
        best_of INTEGER DEFAULT 3,
        series TEXT DEFAULT '',          -- Grand Slam, Masters 1000, ATP 500, ATP 250, WTA 1000, etc.
        tour TEXT DEFAULT 'ATP',         -- ATP, WTA
        winner_id INTEGER REFERENCES tennis_players(id),
        loser_id INTEGER REFERENCES tennis_players(id),
        winner_rank INTEGER DEFAULT 0,
        loser_rank INTEGER DEFAULT 0,
        winner_rank_points INTEGER DEFAULT 0,
        loser_rank_points INTEGER DEFAULT 0,
        score TEXT DEFAULT '',
        w_sets INTEGER DEFAULT 0,
        l_sets INTEGER DEFAULT 0,
        w1 INTEGER, l1 INTEGER,
        w2 INTEGER, l2 INTEGER,
        w3 INTEGER, l3 INTEGER,
        w4 INTEGER, l4 INTEGER,
        w5 INTEGER, l5 INTEGER,
        w_aces INTEGER DEFAULT 0,
        l_aces INTEGER DEFAULT 0,
        w_df INTEGER DEFAULT 0,
        l_df INTEGER DEFAULT 0,
        w_svpt INTEGER DEFAULT 0,       -- service points
        l_svpt INTEGER DEFAULT 0,
        w_1st_in INTEGER DEFAULT 0,
        l_1st_in INTEGER DEFAULT 0,
        w_1st_won INTEGER DEFAULT 0,
        l_1st_won INTEGER DEFAULT 0,
        w_2nd_won INTEGER DEFAULT 0,
        l_2nd_won INTEGER DEFAULT 0,
        w_bp_saved INTEGER DEFAULT 0,
        l_bp_saved INTEGER DEFAULT 0,
        w_bp_faced INTEGER DEFAULT 0,
        l_bp_faced INTEGER DEFAULT 0,
        -- Betting odds
        b365_w REAL DEFAULT 0,
        b365_l REAL DEFAULT 0,
        ps_w REAL DEFAULT 0,            -- Pinnacle
        ps_l REAL DEFAULT 0,
        comment TEXT DEFAULT '',         -- Retired, Walkover, etc.
        UNIQUE(date, winner_id, loser_id, tournament, round)
    );

    CREATE TABLE IF NOT EXISTS tennis_elo (
        player_id INTEGER REFERENCES tennis_players(id),
        elo_type TEXT NOT NULL,          -- global, hard, clay, grass, indoor
        elo REAL DEFAULT 1500,
        matches INTEGER DEFAULT 0,
        last_updated TEXT DEFAULT '',
        PRIMARY KEY (player_id, elo_type)
    );

    CREATE TABLE IF NOT EXISTS tennis_player_flags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER REFERENCES tennis_players(id),
        flag_type TEXT NOT NULL,         -- injury, coach_change, form_concern, returning
        description TEXT DEFAULT '',
        date_set TEXT NOT NULL,
        date_expires TEXT DEFAULT '',    -- auto-expire flags
        active INTEGER DEFAULT 1
    );

    CREATE INDEX IF NOT EXISTS idx_tm_date ON tennis_matches(date);
    CREATE INDEX IF NOT EXISTS idx_tm_winner ON tennis_matches(winner_id);
    CREATE INDEX IF NOT EXISTS idx_tm_loser ON tennis_matches(loser_id);
    CREATE INDEX IF NOT EXISTS idx_tm_surface ON tennis_matches(surface);
    CREATE INDEX IF NOT EXISTS idx_tm_tournament ON tennis_matches(tournament);
    CREATE INDEX IF NOT EXISTS idx_tp_name ON tennis_players(name);

    -- Model training history
    CREATE TABLE IF NOT EXISTS model_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trained_at TEXT NOT NULL,
        model_type TEXT NOT NULL DEFAULT 'singles',
        global_accuracy REAL DEFAULT 0,
        hard_accuracy REAL DEFAULT 0,
        clay_accuracy REAL DEFAULT 0,
        grass_accuracy REAL DEFAULT 0,
        training_samples INTEGER DEFAULT 0,
        hard_samples INTEGER DEFAULT 0,
        clay_samples INTEGER DEFAULT 0,
        grass_samples INTEGER DEFAULT 0
    );
    """)
    conn.commit()
    conn.close()
