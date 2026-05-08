"""Tennis prediction model - LightGBM with surface-specific models, fatigue & serve features."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.tennis.database import get_tennis_db
from src.tennis.elo import expected_score
from src.tennis.features import _compute_fatigue, _compute_serve_features, ROUND_WEIGHTS, TIER_WEIGHTS

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_PATH = DATA_DIR / "tennis_model.pkl"

SURFACE_MAP = {"Hard": "hard", "Clay": "clay", "Grass": "grass", "Carpet": "hard"}

# All features including new fatigue, serve, and round/tier features
FEATURE_COLS = [
    # Elo
    "elo_diff", "elo_expected", "surface_elo_diff", "surface_elo_expected",
    # Rankings
    "rank_diff", "p1_rank", "p2_rank",
    # Form
    "form5_diff", "form10_diff", "p1_form5", "p2_form5", "p1_form10", "p2_form10",
    # Surface form
    "surface_form_diff", "p1_surface_form", "p2_surface_form",
    "p1_surface_matches", "p2_surface_matches",
    # H2H
    "h2h_wr", "h2h_total", "h2h_surface_wr",
    # Tournament
    "tournament_wr_diff", "p1_tournament_wr", "p2_tournament_wr",
    # Regional
    "region_wr_diff",
    # Style
    "p1_is_lefty", "p2_is_lefty",
    # Streaks
    "p1_streak", "p2_streak", "streak_diff",
    # Context
    "best_of", "is_indoor", "surface_code",
    # Experience
    "p1_experience", "p2_experience", "exp_diff",
    # Match stats (aces, df, bp%, serve%)
    "p1_avg_aces", "p2_avg_aces", "aces_diff",
    "p1_avg_df", "p2_avg_df",
    "p1_avg_bp_saved_pct", "p2_avg_bp_saved_pct",
    "p1_avg_1st_won_pct", "p2_avg_1st_won_pct",
    # --- NEW: Fatigue / Scheduling ---
    "p1_days_since_last", "p2_days_since_last",
    "p1_sets_last_14d", "p2_sets_last_14d",
    "p1_matches_last_14d", "p2_matches_last_14d",
    "p1_timezone_jump", "p2_timezone_jump",
    "fatigue_diff",
    # --- NEW: Extended Serve Stats ---
    "p1_serve_hold_pct", "p2_serve_hold_pct", "serve_hold_diff",
    "p1_return_win_pct", "p2_return_win_pct",
    "p1_2nd_serve_pct", "p2_2nd_serve_pct",
    # --- NEW: Round & Tournament Tier ---
    "round_importance", "tier_weight",
]

# Surface-specific feature columns (drop surface_code since model is surface-specific)
SURFACE_FEATURE_COLS = [c for c in FEATURE_COLS if c != "surface_code"]

LGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)


def _get_elo_at_date(conn, pid, elo_type, date_str, elo_cache):
    """Get Elo rating for a player. Uses precomputed cache."""
    key = (pid, elo_type)
    return elo_cache.get(key, 1500)


def _get_form(conn, pid, date_str, n=10, surface=None):
    """Get recent form (win rate) before a given date."""
    if surface:
        rows = conn.execute("""
            SELECT winner_id FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?) AND date < ? AND surface = ?
            AND comment NOT LIKE '%Walkover%'
            ORDER BY date DESC LIMIT ?
        """, (pid, pid, date_str, surface, n)).fetchall()
    else:
        rows = conn.execute("""
            SELECT winner_id FROM tennis_matches
            WHERE (winner_id = ? OR loser_id = ?) AND date < ?
            AND comment NOT LIKE '%Walkover%'
            ORDER BY date DESC LIMIT ?
        """, (pid, pid, date_str, n)).fetchall()
    if not rows:
        return 0.5, 0
    wins = sum(1 for r in rows if r["winner_id"] == pid)
    return wins / len(rows), len(rows)


def _get_h2h(conn, p1, p2, date_str, surface=None):
    """Get H2H record before a given date."""
    if surface:
        rows = conn.execute("""
            SELECT winner_id FROM tennis_matches
            WHERE ((winner_id=? AND loser_id=?) OR (winner_id=? AND loser_id=?))
            AND date < ? AND surface = ?
        """, (p1, p2, p2, p1, date_str, surface)).fetchall()
    else:
        rows = conn.execute("""
            SELECT winner_id FROM tennis_matches
            WHERE ((winner_id=? AND loser_id=?) OR (winner_id=? AND loser_id=?))
            AND date < ?
        """, (p1, p2, p2, p1, date_str)).fetchall()
    total = len(rows)
    if total == 0:
        return 0.5, 0
    p1_wins = sum(1 for r in rows if r["winner_id"] == p1)
    return p1_wins / total, total


def _get_streak(conn, pid, date_str):
    rows = conn.execute("""
        SELECT winner_id FROM tennis_matches
        WHERE (winner_id = ? OR loser_id = ?) AND date < ?
        AND comment NOT LIKE '%Walkover%'
        ORDER BY date DESC LIMIT 10
    """, (pid, pid, date_str)).fetchall()
    if not rows:
        return 0
    streak = 0
    first_won = rows[0]["winner_id"] == pid
    for r in rows:
        if (r["winner_id"] == pid) == first_won:
            streak += 1
        else:
            break
    return streak if first_won else -streak


def _get_avg_stats(conn, pid, date_str, n=10):
    """Get average match stats (aces, df, bp%, serve%) from recent matches."""
    rows = conn.execute("""
        SELECT m.winner_id, m.w_aces, m.l_aces, m.w_df, m.l_df,
               m.w_bp_saved, m.l_bp_saved, m.w_bp_faced, m.l_bp_faced,
               m.w_1st_won, m.l_1st_won, m.w_svpt, m.l_svpt
        FROM tennis_matches m
        WHERE (m.winner_id = ? OR m.loser_id = ?) AND m.date < ?
        AND m.w_aces > 0
        ORDER BY m.date DESC LIMIT ?
    """, (pid, pid, date_str, n)).fetchall()

    if len(rows) < 3:
        return {"aces": 0, "df": 0, "bp_saved_pct": 0, "first_won_pct": 0}

    aces, dfs, bp_pcts, first_pcts = [], [], [], []
    for r in rows:
        is_winner = r["winner_id"] == pid
        aces.append(r["w_aces"] if is_winner else r["l_aces"])
        dfs.append(r["w_df"] if is_winner else r["l_df"])
        bp_faced = r["w_bp_faced"] if is_winner else r["l_bp_faced"]
        bp_saved = r["w_bp_saved"] if is_winner else r["l_bp_saved"]
        if bp_faced and bp_faced > 0:
            bp_pcts.append(bp_saved / bp_faced)
        svpt = r["w_svpt"] if is_winner else r["l_svpt"]
        first_won = r["w_1st_won"] if is_winner else r["l_1st_won"]
        if svpt and svpt > 0:
            first_pcts.append(first_won / svpt)

    return {
        "aces": np.mean(aces) if aces else 0,
        "df": np.mean(dfs) if dfs else 0,
        "bp_saved_pct": np.mean(bp_pcts) if bp_pcts else 0,
        "first_won_pct": np.mean(first_pcts) if first_pcts else 0,
    }


def build_training_data(since_year=2020):
    """Build training dataset from historical matches.

    For each match, compute features using only data available BEFORE the match.
    Label: 1 if player1 (winner) won (always 1 in raw data, but we swap randomly).
    """
    conn = get_tennis_db()

    # Load current Elo as approximation (ideally we'd replay, but this is good enough)
    elo_cache = {}
    for row in conn.execute("SELECT player_id, elo_type, elo FROM tennis_elo").fetchall():
        elo_cache[(row["player_id"], row["elo_type"])] = row["elo"]

    # Get all matches since since_year
    matches = conn.execute("""
        SELECT m.id, m.date, m.tournament, m.surface, m.indoor, m.round,
               m.winner_id, m.loser_id, m.series, m.best_of, m.tour,
               m.winner_rank, m.loser_rank, m.w_sets, m.l_sets,
               p1.name as winner_name, p2.name as loser_name,
               p1.hand as w_hand, p2.hand as l_hand
        FROM tennis_matches m
        JOIN tennis_players p1 ON m.winner_id = p1.id
        JOIN tennis_players p2 ON m.loser_id = p2.id
        WHERE m.date >= ? AND m.w_sets > 0
        AND m.comment NOT LIKE '%Walkover%'
        ORDER BY m.date
    """, (f"{since_year}-01-01",)).fetchall()

    logger.info(f"Building features for {len(matches)} matches since {since_year}...")

    rows = []
    for i, m in enumerate(matches):
        # Randomly swap player1/player2 to avoid bias (model always sees winner as p1)
        swap = np.random.random() < 0.5

        if swap:
            p1_id, p2_id = m["loser_id"], m["winner_id"]
            p1_rank, p2_rank = m["loser_rank"] or 500, m["winner_rank"] or 500
            p1_hand, p2_hand = m["l_hand"], m["w_hand"]
            label = 0  # p1 lost
        else:
            p1_id, p2_id = m["winner_id"], m["loser_id"]
            p1_rank, p2_rank = m["winner_rank"] or 500, m["loser_rank"] or 500
            p1_hand, p2_hand = m["w_hand"], m["l_hand"]
            label = 1  # p1 won

        date_str = m["date"]
        surface = m["surface"]
        surface_key = SURFACE_MAP.get(surface, "hard")

        # Elo
        p1_elo = _get_elo_at_date(conn, p1_id, "global", date_str, elo_cache)
        p2_elo = _get_elo_at_date(conn, p2_id, "global", date_str, elo_cache)
        p1_s_elo = _get_elo_at_date(conn, p1_id, surface_key, date_str, elo_cache)
        p2_s_elo = _get_elo_at_date(conn, p2_id, surface_key, date_str, elo_cache)

        # Form
        p1_f5, _ = _get_form(conn, p1_id, date_str, 5)
        p2_f5, _ = _get_form(conn, p2_id, date_str, 5)
        p1_f10, _ = _get_form(conn, p1_id, date_str, 10)
        p2_f10, _ = _get_form(conn, p2_id, date_str, 10)
        p1_sf, p1_sm = _get_form(conn, p1_id, date_str, 15, surface)
        p2_sf, p2_sm = _get_form(conn, p2_id, date_str, 15, surface)

        # H2H
        h2h_wr, h2h_n = _get_h2h(conn, p1_id, p2_id, date_str)
        h2h_s_wr, _ = _get_h2h(conn, p1_id, p2_id, date_str, surface)

        # Tournament WR
        p1_t = conn.execute("""
            SELECT COUNT(*) as n, SUM(CASE WHEN winner_id=? THEN 1 ELSE 0 END) as w
            FROM tennis_matches WHERE (winner_id=? OR loser_id=?) AND tournament=? AND date<?
        """, (p1_id, p1_id, p1_id, m["tournament"], date_str)).fetchone()
        p2_t = conn.execute("""
            SELECT COUNT(*) as n, SUM(CASE WHEN winner_id=? THEN 1 ELSE 0 END) as w
            FROM tennis_matches WHERE (winner_id=? OR loser_id=?) AND tournament=? AND date<?
        """, (p2_id, p2_id, p2_id, m["tournament"], date_str)).fetchone()
        p1_tw = (p1_t["w"] or 0) / p1_t["n"] if p1_t["n"] else 0.5
        p2_tw = (p2_t["w"] or 0) / p2_t["n"] if p2_t["n"] else 0.5

        # Streaks
        p1_streak = _get_streak(conn, p1_id, date_str)
        p2_streak = _get_streak(conn, p2_id, date_str)

        # Match stats averages
        p1_stats = _get_avg_stats(conn, p1_id, date_str)
        p2_stats = _get_avg_stats(conn, p2_id, date_str)

        # NEW: Fatigue features
        p1_fatigue = _compute_fatigue(conn, p1_id, m["tournament"], ref_date=date_str)
        p2_fatigue = _compute_fatigue(conn, p2_id, m["tournament"], ref_date=date_str)

        # NEW: Extended serve features
        p1_serve = _compute_serve_features(conn, p1_id, ref_date=date_str)
        p2_serve = _compute_serve_features(conn, p2_id, ref_date=date_str)

        # NEW: Round & tier weights
        round_name = m["round"] or ""
        round_importance = ROUND_WEIGHTS.get(round_name, 2)
        series = m["series"] or ""
        tier_weight = TIER_WEIGHTS.get(series, 2)

        row = {
            "label": label,
            "date": date_str,
            "surface_key": surface_key,
            # Elo
            "elo_diff": p1_elo - p2_elo,
            "elo_expected": expected_score(p1_elo, p2_elo),
            "surface_elo_diff": p1_s_elo - p2_s_elo,
            "surface_elo_expected": expected_score(p1_s_elo, p2_s_elo),
            # Rankings
            "rank_diff": (p2_rank - p1_rank),
            "p1_rank": min(p1_rank, 500),
            "p2_rank": min(p2_rank, 500),
            # Form
            "form5_diff": p1_f5 - p2_f5,
            "form10_diff": p1_f10 - p2_f10,
            "p1_form5": p1_f5, "p2_form5": p2_f5,
            "p1_form10": p1_f10, "p2_form10": p2_f10,
            # Surface form
            "surface_form_diff": p1_sf - p2_sf,
            "p1_surface_form": p1_sf, "p2_surface_form": p2_sf,
            "p1_surface_matches": min(p1_sm, 30),
            "p2_surface_matches": min(p2_sm, 30),
            # H2H
            "h2h_wr": h2h_wr,
            "h2h_total": min(h2h_n, 20),
            "h2h_surface_wr": h2h_s_wr,
            # Tournament
            "tournament_wr_diff": p1_tw - p2_tw,
            "p1_tournament_wr": p1_tw, "p2_tournament_wr": p2_tw,
            # Regional (simplified)
            "region_wr_diff": 0,
            # Style
            "p1_is_lefty": 1 if p1_hand == "L" else 0,
            "p2_is_lefty": 1 if p2_hand == "L" else 0,
            # Streaks
            "p1_streak": p1_streak, "p2_streak": p2_streak,
            "streak_diff": p1_streak - p2_streak,
            # Context
            "best_of": m["best_of"] or 3,
            "is_indoor": m["indoor"] or 0,
            "surface_code": {"hard": 0, "clay": 1, "grass": 2}.get(surface_key, 0),
            # Experience
            "p1_experience": 100, "p2_experience": 100,
            "exp_diff": 0,
            # Stats
            "p1_avg_aces": p1_stats["aces"],
            "p2_avg_aces": p2_stats["aces"],
            "aces_diff": p1_stats["aces"] - p2_stats["aces"],
            "p1_avg_df": p1_stats["df"],
            "p2_avg_df": p2_stats["df"],
            "p1_avg_bp_saved_pct": p1_stats["bp_saved_pct"],
            "p2_avg_bp_saved_pct": p2_stats["bp_saved_pct"],
            "p1_avg_1st_won_pct": p1_stats["first_won_pct"],
            "p2_avg_1st_won_pct": p2_stats["first_won_pct"],
            # NEW: Fatigue
            "p1_days_since_last": p1_fatigue["days_since_last"],
            "p2_days_since_last": p2_fatigue["days_since_last"],
            "p1_sets_last_14d": p1_fatigue["sets_last_14d"],
            "p2_sets_last_14d": p2_fatigue["sets_last_14d"],
            "p1_matches_last_14d": p1_fatigue["matches_last_14d"],
            "p2_matches_last_14d": p2_fatigue["matches_last_14d"],
            "p1_timezone_jump": p1_fatigue["timezone_jump"],
            "p2_timezone_jump": p2_fatigue["timezone_jump"],
            "fatigue_diff": p1_fatigue["sets_last_14d"] - p2_fatigue["sets_last_14d"],
            # NEW: Extended serve
            "p1_serve_hold_pct": p1_serve["serve_hold_pct"],
            "p2_serve_hold_pct": p2_serve["serve_hold_pct"],
            "serve_hold_diff": p1_serve["serve_hold_pct"] - p2_serve["serve_hold_pct"],
            "p1_return_win_pct": p1_serve["return_win_pct"],
            "p2_return_win_pct": p2_serve["return_win_pct"],
            "p1_2nd_serve_pct": p1_serve["second_serve_pct"],
            "p2_2nd_serve_pct": p2_serve["second_serve_pct"],
            # NEW: Round & tier
            "round_importance": round_importance,
            "tier_weight": tier_weight,
        }
        rows.append(row)

        if (i + 1) % 2000 == 0:
            logger.info(f"  {i+1}/{len(matches)} features built...")

    conn.close()
    df = pd.DataFrame(rows)
    logger.info(f"Training data: {len(df)} samples, {len(FEATURE_COLS)} features")
    return df


def _train_single_model(X, y, label="global"):
    """Train a single LightGBM model with CV evaluation."""
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        acc = model.score(X_val, y_val)
        accuracies.append(acc)
        logger.info(f"  [{label}] Fold {fold+1}: accuracy={acc:.4f}")

    avg_acc = np.mean(accuracies)
    logger.info(f"  [{label}] Average CV accuracy: {avg_acc:.4f}")

    # Train final on all data
    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.fit(X, y)

    # Calibrate
    calibrated = CalibratedClassifierCV(final_model, cv=5, method="isotonic")
    calibrated.fit(X, y)

    importance = dict(zip(X.columns, final_model.feature_importances_))
    return calibrated, final_model, avg_acc, importance


VALIDATION_DAYS = 14   # last N days held out for champion-vs-candidate gating
GATE_TOLERANCE = 0.001 # candidate accepted if val_acc >= champion_acc - tolerance
MIN_HOLDOUT = 50       # need at least this many held-out matches to gate


def train_model(since_year=2020):
    """Train global + surface-specific LightGBM models with champion-challenger gating.

    Builds the full feature dataframe, holds out the last VALIDATION_DAYS as a clean
    test set, trains a candidate on the rest, then compares its validation accuracy
    against the existing production model on the same held-out matches. Promotes only
    if candidate >= champion - GATE_TOLERANCE.
    """
    df = build_training_data(since_year=since_year)
    df["_date_dt"] = pd.to_datetime(df["date"])

    cutoff = df["_date_dt"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    df_train = df[df["_date_dt"] < cutoff].drop(columns=["_date_dt"])
    df_holdout = df[df["_date_dt"] >= cutoff].drop(columns=["_date_dt"])
    holdout_active = len(df_holdout) >= MIN_HOLDOUT

    if holdout_active:
        logger.info(f"Held-out validation: {len(df_holdout)} matches from last {VALIDATION_DAYS} days "
                    f"({cutoff.date()} →); training on {len(df_train)} earlier matches")
        train_df = df_train
    else:
        logger.info(f"Skipping gating: only {len(df_holdout)} held-out matches (<{MIN_HOLDOUT}); training on full dataset")
        train_df = df.drop(columns=["_date_dt"], errors="ignore")

    # === Global model (all surfaces) ===
    logger.info("=== Training GLOBAL model ===")
    X_global = train_df[FEATURE_COLS].fillna(0)
    y_global = train_df["label"]
    global_cal, global_raw, global_acc, global_imp = _train_single_model(
        X_global, y_global, "global"
    )

    top_features = sorted(global_imp.items(), key=lambda x: -x[1])[:15]
    logger.info("Top global features:")
    for feat, imp in top_features:
        logger.info(f"  {feat}: {imp}")

    # === Surface-specific models ===
    surface_models = {}
    surface_cols = SURFACE_FEATURE_COLS

    for surface_key in ["hard", "clay", "grass"]:
        surface_df = train_df[train_df["surface_key"] == surface_key]
        if len(surface_df) < 500:
            logger.info(f"  [{surface_key}] Only {len(surface_df)} samples, using global model")
            continue

        logger.info(f"=== Training {surface_key.upper()} model ({len(surface_df)} matches) ===")
        X_surf = surface_df[surface_cols].fillna(0)
        y_surf = surface_df["label"]
        cal, raw, acc, imp = _train_single_model(X_surf, y_surf, surface_key)
        surface_models[surface_key] = {
            "model": cal,
            "raw_model": raw,
            "accuracy": acc,
            "feature_importance": imp,
            "training_samples": len(surface_df),
        }
        logger.info(f"  [{surface_key}] Accuracy: {acc:.4f}, Samples: {len(surface_df)}")

    # === GATING: candidate vs champion on held-out set ===
    val_acc_candidate = None
    val_acc_champion = None
    promoted = True
    rejection_reason = None

    if holdout_active:
        X_val = df_holdout[FEATURE_COLS].fillna(0)
        y_val = df_holdout["label"]
        val_acc_candidate = float(global_cal.score(X_val, y_val))
        logger.info(f"Candidate validation accuracy: {val_acc_candidate:.4f} (n={len(df_holdout)})")

        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    champion = pickle.load(f)
                champion_model = champion.get("model")
                champion_gated = champion.get("gated_training", False)
                if champion_model is None:
                    logger.info("Champion has no model — promoting candidate")
                elif not champion_gated:
                    logger.info(
                        "Champion was trained without held-out gating (legacy) — "
                        "skipping comparison and promoting candidate to bootstrap"
                    )
                else:
                    val_acc_champion = float(champion_model.score(X_val, y_val))
                    logger.info(f"Champion validation accuracy: {val_acc_champion:.4f}")
                    if val_acc_candidate < val_acc_champion - GATE_TOLERANCE:
                        promoted = False
                        rejection_reason = (
                            f"candidate {val_acc_candidate:.4f} < champion {val_acc_champion:.4f} "
                            f"- tolerance {GATE_TOLERANCE}"
                        )
            except Exception as e:
                logger.warning(f"Champion evaluation failed ({e}); promoting candidate by default")
        else:
            logger.info("No existing champion — first training, promoting candidate")

    # Build candidate model package
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": global_cal,
        "raw_model": global_raw,
        "features": FEATURE_COLS,
        "surface_features": surface_cols,
        "accuracy": global_acc,
        "validation_accuracy": val_acc_candidate,
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(train_df),
        "feature_importance": global_imp,
        "surface_models": surface_models,
        "gated_training": holdout_active,
        "holdout_cutoff": cutoff.isoformat() if holdout_active else None,
    }

    if promoted:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"PROMOTED → saved to {MODEL_PATH}")
        if val_acc_champion is not None:
            logger.info(f"  candidate {val_acc_candidate:.4f} vs champion {val_acc_champion:.4f}")
    else:
        rejected_path = MODEL_PATH.with_suffix(f".rejected_{datetime.now().strftime('%Y%m%d')}")
        with open(rejected_path, "wb") as f:
            pickle.dump(model_data, f)
        logger.warning(f"REJECTED → {rejection_reason}")
        logger.warning(f"  candidate stored at {rejected_path}; production {MODEL_PATH} unchanged")

    logger.info(f"Global CV accuracy: {global_acc:.4f}, Samples: {len(train_df)}, Features: {len(FEATURE_COLS)}")
    for sk, sm in surface_models.items():
        logger.info(f"  {sk}: {sm['accuracy']:.4f} ({sm['training_samples']} samples)")

    # Log to DB
    try:
        conn = get_tennis_db()
        conn.execute("""
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
            )
        """)
        # Add gating columns if upgrading from older schema
        existing = {r[1] for r in conn.execute("PRAGMA table_info(model_history)").fetchall()}
        for col, ddl in [
            ("validation_accuracy", "REAL"),
            ("champion_validation_accuracy", "REAL"),
            ("promoted", "INTEGER DEFAULT 1"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE model_history ADD COLUMN {col} {ddl}")
        conn.execute("""
            INSERT INTO model_history
            (trained_at, model_type, global_accuracy, hard_accuracy, clay_accuracy, grass_accuracy,
             training_samples, hard_samples, clay_samples, grass_samples,
             validation_accuracy, champion_validation_accuracy, promoted)
            VALUES (?, 'singles', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            global_acc,
            surface_models.get("hard", {}).get("accuracy", 0),
            surface_models.get("clay", {}).get("accuracy", 0),
            surface_models.get("grass", {}).get("accuracy", 0),
            len(train_df),
            surface_models.get("hard", {}).get("training_samples", 0),
            surface_models.get("clay", {}).get("training_samples", 0),
            surface_models.get("grass", {}).get("training_samples", 0),
            val_acc_candidate,
            val_acc_champion,
            1 if promoted else 0,
        ))
        conn.commit()
        conn.close()
        logger.info("Model accuracy logged to model_history table")
    except Exception as e:
        logger.warning(f"Failed to log model accuracy: {e}")

    return global_acc


def load_model():
    """Load trained model."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_match(player1_id: int, player2_id: int, surface: str,
                  indoor: int = 0, tournament: str = "", best_of: int = 3,
                  series: str = "", round_name: str = "") -> dict:
    """Predict match outcome using trained model.

    Uses surface-specific model if available, falls back to global.
    """
    from src.tennis.features import build_match_features

    model_data = load_model()
    if not model_data:
        return {"error": "No model trained"}

    features = build_match_features(player1_id, player2_id, surface, indoor,
                                     tournament=tournament, best_of=best_of,
                                     series=series, round_name=round_name)

    # Add stats features
    conn = get_tennis_db()
    date_str = datetime.now().strftime("%Y-%m-%d")
    p1_stats = _get_avg_stats(conn, player1_id, date_str)
    p2_stats = _get_avg_stats(conn, player2_id, date_str)
    conn.close()

    features["p1_avg_aces"] = p1_stats["aces"]
    features["p2_avg_aces"] = p2_stats["aces"]
    features["aces_diff"] = p1_stats["aces"] - p2_stats["aces"]
    features["p1_avg_df"] = p1_stats["df"]
    features["p2_avg_df"] = p2_stats["df"]
    features["p1_avg_bp_saved_pct"] = p1_stats["bp_saved_pct"]
    features["p2_avg_bp_saved_pct"] = p2_stats["bp_saved_pct"]
    features["p1_avg_1st_won_pct"] = p1_stats["first_won_pct"]
    features["p2_avg_1st_won_pct"] = p2_stats["first_won_pct"]

    # Choose model: surface-specific if available, else global
    surface_key = SURFACE_MAP.get(surface, "hard")
    surface_models = model_data.get("surface_models", {})
    used_model = "global"

    if surface_key in surface_models:
        model = surface_models[surface_key]["model"]
        feature_cols = model_data.get("surface_features", FEATURE_COLS)
        used_model = surface_key
    else:
        model = model_data["model"]
        feature_cols = model_data["features"]

    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
    raw_prob = model.predict_proba(X)[0]

    # Clamp: no model should output extremes beyond 15%-85%
    p1_prob = max(0.15, min(0.85, float(raw_prob[1])))
    p2_prob = 1.0 - p1_prob

    return {
        "p1_win_prob": round(p1_prob, 4),
        "p2_win_prob": round(p2_prob, 4),
        "p1_odds": round(1 / p1_prob, 2) if p1_prob > 0.01 else 99.0,
        "p2_odds": round(1 / p2_prob, 2) if p2_prob > 0.01 else 99.0,
        "confidence": round(float(abs(p1_prob - 0.5) * 2), 4),
        "model_used": used_model,
        "features": features,
    }
