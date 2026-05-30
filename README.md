# Sportee

**Sportee** is a sports analytics and match-outcome prediction platform. It ingests
historical and live match data, maintains player rating systems, engineers
predictive features, and trains calibrated machine-learning models to estimate the
probability of each outcome — with a web dashboard for exploring players, matches,
and model performance.

The current focus is professional tennis (ATP/WTA), with a modular pipeline that
generalises to other sports.

---

## What it does

- **Data ingestion** — Pulls historical and recent match results, player metadata,
  tournament details, and surface/venue information from public sports data sources.
- **Rating system** — Maintains an Elo rating model, both global and per-surface
  (hard / clay / grass / indoor), updated chronologically with tier-weighted
  K-factors.
- **Feature engineering** — Derives form, fatigue, head-to-head, surface affinity,
  experience, and rating-difference features for every matchup, all computed
  leakage-free (only information available *before* a match is used).
- **Prediction models** — Trains LightGBM classifiers (a global model plus
  surface-specific models) with time-series cross-validation and isotonic
  probability calibration, so predicted probabilities are well-aligned with
  observed frequencies.
- **Model governance** — A champion/challenger workflow gates every retrain against
  a held-out window, promoting a new model only when it measurably improves.
- **Web dashboard** — A lightweight interface for browsing players, ratings,
  upcoming matches, model predictions, and historical accuracy.

---

## Architecture

```
src/
  tennis/         Tennis domain: ingestion, Elo, features, model, web data
    sofascore_api.py   API client + surface normalisation
    elo.py             Global + per-surface Elo (live table & leakage-free history)
    features.py        Form / fatigue / H2H / surface feature builders
    model.py           Training data assembly + LightGBM training & calibration
    database.py        SQLite schema + access
  model/          Shared modelling utilities (Elo, predictors, features)
  web/            Dashboard application
scripts/          One-off maintenance & backfill utilities
main.py           CLI entry point
```

Data is stored in a local SQLite database (`data/`).

---

## Getting started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CLI

```bash
python main.py tennis-init [start_year]   # build the historical dataset
python main.py tennis-update              # fetch recent matches
python main.py tennis-train               # train & calibrate models
python main.py web                        # launch the dashboard
```

### Maintenance scripts

```bash
python scripts/full_import.py             # full historical import
python scripts/backfill_surface.py [date] # re-derive surface/venue labels
```

---

## Modelling notes

- **Calibration over raw accuracy.** Outcome probabilities are isotonic-calibrated
  on out-of-fold predictions; a model that is *right about its own uncertainty* is
  more useful than one with a higher raw hit-rate.
- **Surface matters.** Player strength varies sharply by surface, so ratings and
  features are tracked per surface and dedicated surface models are trained where
  enough data exists.
- **No leakage.** Training rows only ever see pre-match state — ratings are
  snapshotted *before* each result is applied, mirroring exactly what is known at
  prediction time.

---

## Tech stack

Python · LightGBM · scikit-learn · pandas / numpy · SQLite · httpx · APScheduler

---

## Author

Created by **STPN** — [github.com/estepeen](https://github.com/estepeen).

## License

You are free to use, copy, modify, and build on this project, including for
commercial purposes, under two conditions:

1. **Credit the author.** Keep a visible attribution to **STPN**
   ([github.com/estepeen](https://github.com/estepeen)) in any derivative work,
   documentation, or product that uses this code.
2. **Keep the signature.** The "Created by STPN" attribution in the web
   dashboard footer must be retained.

Provided "as is", without warranty of any kind.
