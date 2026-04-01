"""Doubles prediction model - LightGBM using individual + pair Elo and form."""

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
from src.tennis.doubles_elo import _pair_key

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_PATH = DATA_DIR / "doubles_model.pkl"

SURFACE_MAP = {"Hard": "hard", "Clay": "clay", "Grass": "grass", "Carpet": "hard"}

FEATURE_COLS = [
    # Individual doubles Elo (averages per team)
    "t1_avg_elo", "t2_avg_elo", "ind_elo_diff", "ind_elo_expected",
    # Surface-specific individual Elo
    "t1_avg_surface_elo", "t2_avg_surface_elo", "surface_elo_diff",
    # Pair Elo
    "t1_pair_elo", "t2_pair_elo", "pair_elo_diff", "pair_elo_expected",
    # Pair experience (how many matches together)
    "t1_pair_matches", "t2_pair_matches", "pair_exp_diff",
    # Individual experience in doubles
    "t1_p1_matches", "t1_p2_matches", "t2_p1_matches", "t2_p2_matches",
    # Form (team recent win rate)
    "t1_form5", "t2_form5", "form5_diff",
    "t1_form10", "t2_form10", "form10_diff",
    # Individual form in doubles
    "t1_p1_form5", "t1_p2_form5", "t2_p1_form5", "t2_p2_form5",
    # H2H between pairs
    "h2h_wr", "h2h_total",
    # Rankings
    "t1_rank", "t2_rank", "rank_diff",
    # Context
    "best_of", "is_indoor", "surface_code",
    # Stats (if available)
    "t1_avg_aces", "t2_avg_aces",
    "t1_avg_df", "t2_avg_df",
    "t1_avg_bp_saved_pct", "t2_avg_bp_saved_pct",
]

LGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=20,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=42,
    verbose=-1,
)


def _get_ind_elo(conn, pid, elo_type, elo_cache):
    """Get individual doubles Elo from cache."""
    return elo_cache.get((pid, elo_type), 1500)


def _get_pair_elo_val(conn, p1, p2, elo_type, pair_cache):
    """Get pair Elo from cache."""
    key = _pair_key(p1, p2)
    return pair_cache.get((key[0], key[1], elo_type), 1500)


def _get_pair_matches(conn, p1, p2, pair_cache):
    """Get number of matches a pair has played together."""
    key = _pair_key(p1, p2)
    return pair_cache.get((key[0], key[1], "matches"), 0)


def _get_team_form(conn, p1_id, p2_id, date_str, n=5):
    """Get recent form for a doubles pair (wins from last N matches together)."""
    rows = conn.execute("""
        SELECT w_player1_id, w_player2_id FROM doubles_matches
        WHERE ((w_player1_id IN (?,?) AND w_player2_id IN (?,?))
            OR (l_player1_id IN (?,?) AND l_player2_id IN (?,?)))
        AND date < ? AND comment NOT LIKE '%Walkover%'
        ORDER BY date DESC LIMIT ?
    """, (p1_id, p2_id, p1_id, p2_id,
          p1_id, p2_id, p1_id, p2_id,
          date_str, n)).fetchall()
    if not rows:
        return 0.5, 0
    wins = sum(1 for r in rows
               if set([r["w_player1_id"], r["w_player2_id"]]) == set([p1_id, p2_id]))
    return wins / len(rows), len(rows)


def _get_player_doubles_form(conn, pid, date_str, n=5):
    """Get individual player's recent doubles form."""
    rows = conn.execute("""
        SELECT w_player1_id, w_player2_id FROM doubles_matches
        WHERE (w_player1_id = ? OR w_player2_id = ? OR l_player1_id = ? OR l_player2_id = ?)
        AND date < ? AND comment NOT LIKE '%Walkover%'
        ORDER BY date DESC LIMIT ?
    """, (pid, pid, pid, pid, date_str, n)).fetchall()
    if not rows:
        return 0.5
    wins = sum(1 for r in rows if r["w_player1_id"] == pid or r["w_player2_id"] == pid)
    return wins / len(rows)


def _get_pair_h2h(conn, t1_p1, t1_p2, t2_p1, t2_p2, date_str):
    """Get H2H between two specific doubles pairs."""
    rows = conn.execute("""
        SELECT w_player1_id, w_player2_id FROM doubles_matches
        WHERE (
            (w_player1_id IN (?,?) AND w_player2_id IN (?,?)
             AND l_player1_id IN (?,?) AND l_player2_id IN (?,?))
            OR
            (l_player1_id IN (?,?) AND l_player2_id IN (?,?)
             AND w_player1_id IN (?,?) AND w_player2_id IN (?,?))
        ) AND date < ?
    """, (t1_p1, t1_p2, t1_p1, t1_p2, t2_p1, t2_p2, t2_p1, t2_p2,
          t1_p1, t1_p2, t1_p1, t1_p2, t2_p1, t2_p2, t2_p1, t2_p2,
          date_str)).fetchall()
    total = len(rows)
    if total == 0:
        return 0.5, 0
    t1_wins = sum(1 for r in rows
                  if set([r["w_player1_id"], r["w_player2_id"]]) == set([t1_p1, t1_p2]))
    return t1_wins / total, total


def _get_team_avg_stats(conn, p1_id, p2_id, date_str, n=10):
    """Get average doubles stats for a pair."""
    rows = conn.execute("""
        SELECT w_player1_id, w_player2_id,
               w_aces, l_aces, w_df, l_df,
               w_bp_saved, l_bp_saved, w_bp_faced, l_bp_faced
        FROM doubles_matches
        WHERE ((w_player1_id IN (?,?) AND w_player2_id IN (?,?))
            OR (l_player1_id IN (?,?) AND l_player2_id IN (?,?)))
        AND date < ? AND w_aces > 0
        ORDER BY date DESC LIMIT ?
    """, (p1_id, p2_id, p1_id, p2_id,
          p1_id, p2_id, p1_id, p2_id,
          date_str, n)).fetchall()

    if len(rows) < 2:
        return {"aces": 0, "df": 0, "bp_saved_pct": 0}

    aces, dfs, bp_pcts = [], [], []
    for r in rows:
        is_winner = set([r["w_player1_id"], r["w_player2_id"]]) == set([p1_id, p2_id])
        aces.append(r["w_aces"] if is_winner else r["l_aces"])
        dfs.append(r["w_df"] if is_winner else r["l_df"])
        bp_faced = r["w_bp_faced"] if is_winner else r["l_bp_faced"]
        bp_saved = r["w_bp_saved"] if is_winner else r["l_bp_saved"]
        if bp_faced and bp_faced > 0:
            bp_pcts.append(bp_saved / bp_faced)

    return {
        "aces": np.mean(aces) if aces else 0,
        "df": np.mean(dfs) if dfs else 0,
        "bp_saved_pct": np.mean(bp_pcts) if bp_pcts else 0,
    }


def build_doubles_training_data():
    """Build training data from all doubles matches."""
    conn = get_tennis_db()

    # Load Elo caches
    elo_cache = {}
    for row in conn.execute("SELECT player_id, elo_type, elo FROM doubles_elo").fetchall():
        elo_cache[(row["player_id"], row["elo_type"])] = row["elo"]

    pair_cache = {}
    for row in conn.execute("SELECT player1_id, player2_id, elo_type, elo, matches FROM doubles_pair_elo").fetchall():
        pair_cache[(row["player1_id"], row["player2_id"], row["elo_type"])] = row["elo"]
        pair_cache[(row["player1_id"], row["player2_id"], "matches")] = row["matches"]

    matches = conn.execute("""
        SELECT id, date, tournament, surface, indoor, round, best_of, series, tour,
               w_player1_id, w_player2_id, l_player1_id, l_player2_id,
               w_rank, l_rank
        FROM doubles_matches
        WHERE w_player1_id IS NOT NULL AND l_player1_id IS NOT NULL
        AND comment NOT LIKE '%Walkover%'
        ORDER BY date
    """).fetchall()

    logger.info(f"Building doubles features for {len(matches)} matches...")
    rows = []

    for i, m in enumerate(matches):
        swap = np.random.random() < 0.5

        if swap:
            t1 = (m["l_player1_id"], m["l_player2_id"])
            t2 = (m["w_player1_id"], m["w_player2_id"])
            t1_rank, t2_rank = m["l_rank"] or 500, m["w_rank"] or 500
            label = 0
        else:
            t1 = (m["w_player1_id"], m["w_player2_id"])
            t2 = (m["l_player1_id"], m["l_player2_id"])
            t1_rank, t2_rank = m["w_rank"] or 500, m["l_rank"] or 500
            label = 1

        date_str = m["date"]
        surface_key = SURFACE_MAP.get(m["surface"] or "Hard", "hard")

        # Individual Elo
        t1_e1 = _get_ind_elo(conn, t1[0], "doubles_global", elo_cache)
        t1_e2 = _get_ind_elo(conn, t1[1], "doubles_global", elo_cache)
        t2_e1 = _get_ind_elo(conn, t2[0], "doubles_global", elo_cache)
        t2_e2 = _get_ind_elo(conn, t2[1], "doubles_global", elo_cache)
        t1_avg = (t1_e1 + t1_e2) / 2
        t2_avg = (t2_e1 + t2_e2) / 2

        # Surface Elo
        t1_se1 = _get_ind_elo(conn, t1[0], f"doubles_{surface_key}", elo_cache)
        t1_se2 = _get_ind_elo(conn, t1[1], f"doubles_{surface_key}", elo_cache)
        t2_se1 = _get_ind_elo(conn, t2[0], f"doubles_{surface_key}", elo_cache)
        t2_se2 = _get_ind_elo(conn, t2[1], f"doubles_{surface_key}", elo_cache)
        t1_savg = (t1_se1 + t1_se2) / 2
        t2_savg = (t2_se1 + t2_se2) / 2

        # Pair Elo
        t1_pair = _get_pair_elo_val(conn, t1[0], t1[1], "pair_global", pair_cache)
        t2_pair = _get_pair_elo_val(conn, t2[0], t2[1], "pair_global", pair_cache)

        # Pair matches
        t1_pm = _get_pair_matches(conn, t1[0], t1[1], pair_cache)
        t2_pm = _get_pair_matches(conn, t2[0], t2[1], pair_cache)

        # Individual matches
        t1_p1_m = elo_cache.get((t1[0], "doubles_global"), 0)
        t1_p2_m = elo_cache.get((t1[1], "doubles_global"), 0)
        t2_p1_m = elo_cache.get((t2[0], "doubles_global"), 0)
        t2_p2_m = elo_cache.get((t2[1], "doubles_global"), 0)

        # Form
        t1_f5, _ = _get_team_form(conn, t1[0], t1[1], date_str, 5)
        t2_f5, _ = _get_team_form(conn, t2[0], t2[1], date_str, 5)
        t1_f10, _ = _get_team_form(conn, t1[0], t1[1], date_str, 10)
        t2_f10, _ = _get_team_form(conn, t2[0], t2[1], date_str, 10)

        # Individual form
        t1_p1_f = _get_player_doubles_form(conn, t1[0], date_str, 5)
        t1_p2_f = _get_player_doubles_form(conn, t1[1], date_str, 5)
        t2_p1_f = _get_player_doubles_form(conn, t2[0], date_str, 5)
        t2_p2_f = _get_player_doubles_form(conn, t2[1], date_str, 5)

        # H2H
        h2h_wr, h2h_n = _get_pair_h2h(conn, t1[0], t1[1], t2[0], t2[1], date_str)

        # Stats
        t1_stats = _get_team_avg_stats(conn, t1[0], t1[1], date_str)
        t2_stats = _get_team_avg_stats(conn, t2[0], t2[1], date_str)

        row = {
            "label": label, "date": date_str,
            "t1_avg_elo": t1_avg, "t2_avg_elo": t2_avg,
            "ind_elo_diff": t1_avg - t2_avg,
            "ind_elo_expected": expected_score(t1_avg, t2_avg),
            "t1_avg_surface_elo": t1_savg, "t2_avg_surface_elo": t2_savg,
            "surface_elo_diff": t1_savg - t2_savg,
            "t1_pair_elo": t1_pair, "t2_pair_elo": t2_pair,
            "pair_elo_diff": t1_pair - t2_pair,
            "pair_elo_expected": expected_score(t1_pair, t2_pair),
            "t1_pair_matches": min(t1_pm, 50), "t2_pair_matches": min(t2_pm, 50),
            "pair_exp_diff": t1_pm - t2_pm,
            "t1_p1_matches": 50, "t1_p2_matches": 50,
            "t2_p1_matches": 50, "t2_p2_matches": 50,
            "t1_form5": t1_f5, "t2_form5": t2_f5, "form5_diff": t1_f5 - t2_f5,
            "t1_form10": t1_f10, "t2_form10": t2_f10, "form10_diff": t1_f10 - t2_f10,
            "t1_p1_form5": t1_p1_f, "t1_p2_form5": t1_p2_f,
            "t2_p1_form5": t2_p1_f, "t2_p2_form5": t2_p2_f,
            "h2h_wr": h2h_wr, "h2h_total": min(h2h_n, 10),
            "t1_rank": min(t1_rank, 500), "t2_rank": min(t2_rank, 500),
            "rank_diff": t2_rank - t1_rank,
            "best_of": m["best_of"] or 3,
            "is_indoor": m["indoor"] or 0,
            "surface_code": {"hard": 0, "clay": 1, "grass": 2}.get(surface_key, 0),
            "t1_avg_aces": t1_stats["aces"], "t2_avg_aces": t2_stats["aces"],
            "t1_avg_df": t1_stats["df"], "t2_avg_df": t2_stats["df"],
            "t1_avg_bp_saved_pct": t1_stats["bp_saved_pct"],
            "t2_avg_bp_saved_pct": t2_stats["bp_saved_pct"],
        }
        rows.append(row)

        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(matches)} doubles features built...")

    conn.close()
    df = pd.DataFrame(rows)
    logger.info(f"Doubles training data: {len(df)} samples, {len(FEATURE_COLS)} features")
    return df


def train_doubles_model():
    """Train doubles LightGBM model."""
    df = build_doubles_training_data()

    if len(df) < 100:
        logger.warning(f"Only {len(df)} doubles matches, not enough to train. Need 100+.")
        return 0

    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]

    tscv = TimeSeriesSplit(n_splits=3)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        acc = model.score(X_val, y_val)
        accuracies.append(acc)
        logger.info(f"  Doubles fold {fold+1}: accuracy={acc:.4f}")

    avg_acc = np.mean(accuracies)

    # Train final model
    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.fit(X, y)

    calibrated = CalibratedClassifierCV(final_model, cv=3, method="isotonic")
    calibrated.fit(X, y)

    importance = dict(zip(FEATURE_COLS, final_model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: -x[1])[:10]
    logger.info("Top doubles features:")
    for feat, imp in top:
        logger.info(f"  {feat}: {imp}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": calibrated,
        "raw_model": final_model,
        "features": FEATURE_COLS,
        "accuracy": avg_acc,
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(df),
        "feature_importance": importance,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Doubles model saved: accuracy={avg_acc:.4f}, samples={len(df)}")
    return avg_acc


def load_doubles_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_doubles_match(t1_p1_id, t1_p2_id, t2_p1_id, t2_p2_id,
                          surface="Hard", indoor=0) -> dict:
    """Predict a doubles match."""
    model_data = load_doubles_model()
    if not model_data:
        return {"error": "No doubles model trained"}

    conn = get_tennis_db()
    date_str = datetime.now().strftime("%Y-%m-%d")
    surface_key = SURFACE_MAP.get(surface, "hard")

    # Load caches
    elo_cache = {}
    for row in conn.execute("SELECT player_id, elo_type, elo FROM doubles_elo").fetchall():
        elo_cache[(row["player_id"], row["elo_type"])] = row["elo"]

    pair_cache = {}
    for row in conn.execute("SELECT player1_id, player2_id, elo_type, elo, matches FROM doubles_pair_elo").fetchall():
        pair_cache[(row["player1_id"], row["player2_id"], row["elo_type"])] = row["elo"]
        pair_cache[(row["player1_id"], row["player2_id"], "matches")] = row["matches"]

    t1 = (t1_p1_id, t1_p2_id)
    t2 = (t2_p1_id, t2_p2_id)

    # Build features (same as training)
    t1_avg = (_get_ind_elo(conn, t1[0], "doubles_global", elo_cache) +
              _get_ind_elo(conn, t1[1], "doubles_global", elo_cache)) / 2
    t2_avg = (_get_ind_elo(conn, t2[0], "doubles_global", elo_cache) +
              _get_ind_elo(conn, t2[1], "doubles_global", elo_cache)) / 2

    t1_savg = (_get_ind_elo(conn, t1[0], f"doubles_{surface_key}", elo_cache) +
               _get_ind_elo(conn, t1[1], f"doubles_{surface_key}", elo_cache)) / 2
    t2_savg = (_get_ind_elo(conn, t2[0], f"doubles_{surface_key}", elo_cache) +
               _get_ind_elo(conn, t2[1], f"doubles_{surface_key}", elo_cache)) / 2

    t1_pair = _get_pair_elo_val(conn, t1[0], t1[1], "pair_global", pair_cache)
    t2_pair = _get_pair_elo_val(conn, t2[0], t2[1], "pair_global", pair_cache)

    t1_pm = _get_pair_matches(conn, t1[0], t1[1], pair_cache)
    t2_pm = _get_pair_matches(conn, t2[0], t2[1], pair_cache)

    t1_f5, _ = _get_team_form(conn, t1[0], t1[1], date_str, 5)
    t2_f5, _ = _get_team_form(conn, t2[0], t2[1], date_str, 5)
    t1_f10, _ = _get_team_form(conn, t1[0], t1[1], date_str, 10)
    t2_f10, _ = _get_team_form(conn, t2[0], t2[1], date_str, 10)

    h2h_wr, h2h_n = _get_pair_h2h(conn, t1[0], t1[1], t2[0], t2[1], date_str)

    t1_stats = _get_team_avg_stats(conn, t1[0], t1[1], date_str)
    t2_stats = _get_team_avg_stats(conn, t2[0], t2[1], date_str)

    conn.close()

    features = {
        "t1_avg_elo": t1_avg, "t2_avg_elo": t2_avg,
        "ind_elo_diff": t1_avg - t2_avg,
        "ind_elo_expected": expected_score(t1_avg, t2_avg),
        "t1_avg_surface_elo": t1_savg, "t2_avg_surface_elo": t2_savg,
        "surface_elo_diff": t1_savg - t2_savg,
        "t1_pair_elo": t1_pair, "t2_pair_elo": t2_pair,
        "pair_elo_diff": t1_pair - t2_pair,
        "pair_elo_expected": expected_score(t1_pair, t2_pair),
        "t1_pair_matches": min(t1_pm, 50), "t2_pair_matches": min(t2_pm, 50),
        "pair_exp_diff": t1_pm - t2_pm,
        "t1_p1_matches": 50, "t1_p2_matches": 50,
        "t2_p1_matches": 50, "t2_p2_matches": 50,
        "t1_form5": t1_f5, "t2_form5": t2_f5, "form5_diff": t1_f5 - t2_f5,
        "t1_form10": t1_f10, "t2_form10": t2_f10, "form10_diff": t1_f10 - t2_f10,
        "t1_p1_form5": _get_player_doubles_form(get_tennis_db(), t1[0], date_str, 5),
        "t1_p2_form5": _get_player_doubles_form(get_tennis_db(), t1[1], date_str, 5),
        "t2_p1_form5": _get_player_doubles_form(get_tennis_db(), t2[0], date_str, 5),
        "t2_p2_form5": _get_player_doubles_form(get_tennis_db(), t2[1], date_str, 5),
        "h2h_wr": h2h_wr, "h2h_total": min(h2h_n, 10),
        "t1_rank": 500, "t2_rank": 500, "rank_diff": 0,
        "best_of": 3, "is_indoor": indoor,
        "surface_code": {"hard": 0, "clay": 1, "grass": 2}.get(surface_key, 0),
        "t1_avg_aces": t1_stats["aces"], "t2_avg_aces": t2_stats["aces"],
        "t1_avg_df": t1_stats["df"], "t2_avg_df": t2_stats["df"],
        "t1_avg_bp_saved_pct": t1_stats["bp_saved_pct"],
        "t2_avg_bp_saved_pct": t2_stats["bp_saved_pct"],
    }

    feature_cols = model_data["features"]
    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
    raw_prob = model_data["model"].predict_proba(X)[0]

    # Clamp 15-85%
    p1_prob = max(0.15, min(0.85, float(raw_prob[1])))
    p2_prob = 1.0 - p1_prob

    return {
        "t1_win_prob": round(p1_prob, 4),
        "t2_win_prob": round(p2_prob, 4),
        "t1_odds": round(1 / p1_prob, 2) if p1_prob > 0.01 else 99.0,
        "t2_odds": round(1 / p2_prob, 2) if p2_prob > 0.01 else 99.0,
        "confidence": round(float(abs(p1_prob - 0.5) * 2), 4),
        "features": features,
    }
