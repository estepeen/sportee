"""Diagnose the CV(0.82) vs holdout(0.61) gap on the tennis model.

Rebuilds only the last ~6 weeks of feature rows (fast), reproduces the same
last-14-days holdout the trainer uses, then asks three questions:

  1. Is it a *measurement* difference? CV accuracy is measured on the RAW per-fold
     LGBM; the holdout gate is measured on the CALIBRATED ensemble. Compare both
     on the same holdout.
  2. Is the holdout just a hard window? Compare against trivial baselines
     (higher-Elo wins, better-ranked wins) on the same rows.
  3. Is it surface-driven? Break accuracy down by surface, since the holdout sits
     in clay/Roland-Garros season.

Run on the VPS:  python scripts/diag_holdout.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tennis.model import build_training_data, load_model, FEATURE_COLS, VALIDATION_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def acc(pred, y):
    return float((np.asarray(pred).astype(int) == np.asarray(y).astype(int)).mean())


def main():
    # Build a recent slice. Point-in-time Elo is replayed over full history inside
    # build_training_data, so these rows still get correct pre-match ratings.
    df = build_training_data(since_date="2026-04-10")
    df["_dt"] = pd.to_datetime(df["date"])

    cutoff = df["_dt"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    holdout = df[df["_dt"] >= cutoff].copy()
    log.info(f"Full max date: {df['_dt'].max().date()} | holdout cutoff: {cutoff.date()} "
             f"| holdout n={len(holdout)}")

    y = holdout["label"]
    X = holdout[FEATURE_COLS].fillna(0)

    md = load_model()
    cal = md["model"]          # CalibratedClassifierCV ensemble
    raw = md.get("raw_model")  # plain LGBM final fit

    log.info("=" * 60)
    log.info(f"label balance in holdout: mean={y.mean():.3f} (0.5 = balanced)")

    # 1. measurement: calibrated vs raw on the SAME holdout
    cal_pred = cal.predict(X)
    log.info(f"[calibrated] holdout acc = {acc(cal_pred, y):.4f}  (this is the gate number)")
    if raw is not None:
        raw_pred = raw.predict(X)
        log.info(f"[raw LGBM ] holdout acc = {acc(raw_pred, y):.4f}  (CV is measured on this kind of model)")
        # probability spread tells us if calibration squashed toward 0.5
        cal_p = cal.predict_proba(X)[:, 1]
        raw_p = raw.predict_proba(X)[:, 1]
        log.info(f"  calibrated prob std={cal_p.std():.3f} mean={cal_p.mean():.3f} "
                 f"| raw prob std={raw_p.std():.3f} mean={raw_p.mean():.3f}")
        log.info(f"  calibrated preds in [0.45,0.55]: {((cal_p>=.45)&(cal_p<=.55)).mean():.1%} "
                 f"| raw: {((raw_p>=.45)&(raw_p<=.55)).mean():.1%}")

    # 2. trivial baselines on the same rows
    # elo_diff = p1_elo - p2_elo  → predict p1 win when >0
    log.info("-" * 60)
    log.info(f"[baseline] higher global-Elo wins : {acc((holdout['elo_diff']>0), y):.4f}")
    log.info(f"[baseline] higher surface-Elo wins: {acc((holdout['surface_elo_diff']>0), y):.4f}")
    # rank_diff = p2_rank - p1_rank → >0 means p1 better ranked
    log.info(f"[baseline] better ranked wins     : {acc((holdout['rank_diff']>0), y):.4f}")

    # 3. surface breakdown (calibrated model)
    log.info("-" * 60)
    holdout = holdout.assign(_cal_pred=cal_pred)
    for sk, g in holdout.groupby("surface_key"):
        log.info(f"[surface={sk:5s}] n={len(g):4d}  cal_acc={acc(g['_cal_pred'], g['label']):.4f}  "
                 f"elo_base={acc((g['elo_diff']>0), g['label']):.4f}")


if __name__ == "__main__":
    main()
