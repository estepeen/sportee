"""Calibrate ML model - proper out-of-sample using time split."""
import asyncio
import pickle
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from src.database import init_db
from src.model.ml_predictor import MLPredictor


async def main():
    await init_db()
    pred = MLPredictor()
    df = await pred._build_dataset()

    fcols = [c for c in df.columns if c not in ("match_id", "date", "target")]
    X = df[fcols]
    y = df["target"]

    print(f"Dataset: {len(X)} matches, {len(fcols)} features")

    # Split: 80% train, 20% test (time-based)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train fresh model
    mdl = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        num_leaves=20, min_child_samples=15, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0.3, verbose=-1,
    )
    mdl.fit(X_train, y_train)

    raw_probs = mdl.predict_proba(X_test)[:, 1]
    raw_acc = accuracy_score(y_test, mdl.predict(X_test))
    raw_brier = brier_score_loss(y_test, raw_probs)

    print(f"\n=== RAW MODEL (out-of-sample) ===")
    print(f"Accuracy: {raw_acc:.1%}")
    print(f"Brier: {raw_brier:.4f}")

    bins = np.linspace(0, 1, 6)  # fewer bins for smaller test set
    for i in range(len(bins) - 1):
        mask = (raw_probs >= bins[i]) & (raw_probs < bins[i + 1])
        if mask.sum() > 3:
            mp = raw_probs[mask].mean()
            ma = y_test.values[mask].mean()
            s = "OVER" if mp > ma + 0.05 else "UNDER" if ma > mp + 0.05 else "OK"
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: pred={mp:.2f} actual={ma:.2f} n={mask.sum():4d} {s}")

    # Platt scaling on train, eval on test
    print(f"\n=== PLATT SCALING ===")
    cal_sig = CalibratedClassifierCV(mdl, method="sigmoid", cv=3)
    cal_sig.fit(X_train, y_train)
    sig_probs = cal_sig.predict_proba(X_test)[:, 1]
    sig_brier = brier_score_loss(y_test, sig_probs)
    sig_acc = accuracy_score(y_test, cal_sig.predict(X_test))
    print(f"Accuracy: {sig_acc:.1%}")
    print(f"Brier: {sig_brier:.4f}")

    for i in range(len(bins) - 1):
        mask = (sig_probs >= bins[i]) & (sig_probs < bins[i + 1])
        if mask.sum() > 3:
            mp = sig_probs[mask].mean()
            ma = y_test.values[mask].mean()
            s = "OVER" if mp > ma + 0.05 else "UNDER" if ma > mp + 0.05 else "OK"
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: pred={mp:.2f} actual={ma:.2f} n={mask.sum():4d} {s}")

    # Isotonic on train, eval on test
    print(f"\n=== ISOTONIC ===")
    cal_iso = CalibratedClassifierCV(mdl, method="isotonic", cv=3)
    cal_iso.fit(X_train, y_train)
    iso_probs = cal_iso.predict_proba(X_test)[:, 1]
    iso_brier = brier_score_loss(y_test, iso_probs)
    iso_acc = accuracy_score(y_test, cal_iso.predict(X_test))
    print(f"Accuracy: {iso_acc:.1%}")
    print(f"Brier: {iso_brier:.4f}")

    # Pick best
    results = [
        ("raw", raw_brier, raw_acc, mdl),
        ("sigmoid", sig_brier, sig_acc, cal_sig),
        ("isotonic", iso_brier, iso_acc, cal_iso),
    ]
    results.sort(key=lambda x: x[1])
    print(f"\n=== RESULTS ===")
    for name, brier, acc, _ in results:
        print(f"  {name:10s}: brier={brier:.4f} acc={acc:.1%}")

    best_name, best_brier, best_acc, best_model = results[0]
    print(f"\nBest: {best_name} (brier={best_brier:.4f})")

    # Now retrain on ALL data with best method
    if best_name == "raw":
        final_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            num_leaves=20, min_child_samples=15, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0.3, verbose=-1,
        )
        final_model.fit(X, y)
        use_calibrated = False
    else:
        base = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            num_leaves=20, min_child_samples=15, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0.3, verbose=-1,
        )
        final_model = CalibratedClassifierCV(base, method=best_name, cv=3)
        final_model.fit(X, y)
        use_calibrated = True

    # Save
    with open("data/model.pkl", "rb") as f:
        d = pickle.load(f)

    d["model"] = final_model if not use_calibrated else final_model.estimator
    d["calibrated_model"] = final_model if use_calibrated else None
    d["calibration_method"] = best_name
    d["brier_score"] = float(best_brier)
    d["accuracy"] = float(best_acc)
    d["features"] = fcols
    d["use_calibrated"] = use_calibrated

    with open("data/model.pkl", "wb") as f:
        pickle.dump(d, f)

    print(f"\nSaved! use_calibrated={use_calibrated}, method={best_name}")

    # Show example predictions
    test_probs = final_model.predict_proba(X_test)[:, 1] if not use_calibrated else final_model.predict_proba(X_test)[:, 1]
    print(f"\n=== TEST SET EXAMPLES ===")
    print(f"{'Predicted':>10s} {'Actual':>8s}")
    for p, a in list(zip(test_probs[-10:], y_test.values[-10:])):
        print(f"  {p:8.2f}    {a}")


if __name__ == "__main__":
    asyncio.run(main())
