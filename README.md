# Traffic Emissions Sydney — ML + XAI

Predicting **NO₂ concentration** near NSW traffic stations from traffic
volume, weather, and station metadata, using machine learning, with
explainable AI (SHAP) to identify key drivers.

> The project originally targeted a *computed* CO₂-estimate variable
> across 3 Sydney-only stations. It has since pivoted to a **directly
> measured** NO₂ target (per supervisor guidance — computed/estimated
> targets weren't acceptable) across 15 candidate stations spanning NSW,
> not just Sydney. See `docs/Data.md` for the full history and every
> data-quality decision.

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run_pipeline.py
```

That's the whole workflow: **change settings in `config/config.yaml`, then
run `python run_pipeline.py`.**

## Where to change things

| You want to… | Edit |
|---|---|
| Choose which stage(s) run | `config/config.yaml` → `pipeline.stage` |
| Switch daily/hourly, feature sets, or the scoring split | `config/config.yaml` → `training` |
| Turn a model on/off or change its hyperparameters | `config/config.yaml` → `training.models` |
| Exclude a station / run the sensitivity analysis | `config/config.yaml` → `data.station_exclusions` |
| Change the tuning search space or number of trials | `config/tuning.yaml` |
| Add a new model | one line in `scripts/models.py` + a block in `config/config.yaml` |
| Move a folder | `scripts/paths.py` (every location is defined there once) |
| Change how a data step works | that stage's numbered script in `src/` |

## Project structure

```
config/
  config.yaml            pipeline stage, station exclusions, training + model params
  tuning.yaml            XGBoost / SVR search spaces (shared settings + one section each)
data/
  raw/                   downloaded source files
    air_quality/           nsw_air_quality_sites.json -- optional, enables AQ-site verification
    emissions/             AQ-portal hourly pollutant data (incl. NO2)
    speed_zones/           Speed_Zones.{shp,shx,dbf,prj} -- optional, gitignored (too big
                           for GitHub); download from TfNSW and place here manually
    traffic/               TfNSW Traffic Volume Viewer exports per station + station_reference.csv
    weather/               AQ-portal hourly weather (metro) + BOM daily weather (rural)
  processed/             outputs of stages 01-05 + splits/ (gitignored, always rebuildable)
results/                 everything the models produce (kept across runs)
  results.json             every model's train/val/test scores, baseline + tuned
  per_station/             per-station breakdown of every run
  tuning/                  tuning trials, best results, learning curves, overfitting, feature importance
  evaluation/              07's comparison tables + plots
  saved_models/            fitted tuned models (.joblib, gitignored)
  shap/                    (later) SHAP outputs from 08
docs/
  Data.md                sources, stations, and every data-quality decision with its justification
notebooks/
  01_data_exploration.ipynb   EDA
src/                     the pipeline stages, run in order by run_pipeline.py
  01_data_ingestion.py        combine raw traffic/weather/NO2 per station -> daily + hourly datasets;
                              AQ-site verification and posted-speed matching
  02_cleaning.py              drop no-NO2 / excluded stations, range checks, drop missing target, dedup
  03_data_preprocessing.py    outlier report, aq_quality_weight, train-only imputation, station one-hot
  04_feature_engineering.py   calendar, road, directional, traffic, weather, lag/rolling, target features;
                              feature_manifest.json; validation checks
  05_train_test_split.py      global chronological 70/15/15 split, near-duplicate pruning, scaling (train-fit)
  06_run_models.py            train + score every enabled model and the naive baselines -> results/
  07_evaluation.py            train/val/test comparison tables + plots from results.json
  08_explainability.py        (placeholder) SHAP on the best model
scripts/                 shared code the stages import (not run directly, except tuning/)
  paths.py                    every folder/file location, defined once
  models.py                   model registry: config name -> estimator class
  split_utils.py              the one definition of "train" (used by 03 and 05)
  model_utils.py              shared load_split / evaluate / save_result
  tuning/
    common.py                 random search, learning curves, overfitting diagnostics
    tune_xgboost.py           XGBoost tuning (daily + hourly, early stopping)
    tune_svr.py               SVR tuning (daily)
run_pipeline.py          runs everything (see below)
```

## Running the pipeline

`run_pipeline.py` reads `pipeline.stage` from `config/config.yaml`:

| Stage | Script | Output |
|---|---|---|
| `ingest` | `src/01_data_ingestion.py` | `final_combined_dataset_{daily,hourly}.csv`, `aq_site_verification.csv` |
| `clean` | `src/02_cleaning.py` | `preprocessed_{daily,hourly}.csv`, `excluded_stations_log_*.csv` |
| `preprocess` | `src/03_data_preprocessing.py` | `final_{daily,hourly}.csv` |
| `features` | `src/04_feature_engineering.py` | `features_{daily,hourly}.csv`, `feature_manifest.json` |
| `split` | `src/05_train_test_split.py` | `splits/{grain}_{train,val,test}.csv`, scalers, `split_manifest.json` |
| `train` | `src/06_run_models.py` | `results/results.json`, `results/per_station/` |
| `tune` | `scripts/tuning/tune_xgboost.py`, `scripts/tuning/tune_svr.py` | `results/tuning/`, `results/saved_models/` |
| `evaluate` | `src/07_evaluation.py` | `results/evaluation/` |

- **`stage: all`** runs every stage in order. It empties `data/processed/`
  first so a run never mixes with a stale one; `results/` is never wiped.
  Tuning is slow, so it's only included when `pipeline.run_tuning: true`.
- **`stage: <name>`** runs just that stage and reuses what's on disk. The
  runner checks the stage's inputs exist first and tells you which earlier
  stage to run if not.
- **`--stage`** overrides the config for a one-off, e.g.
  `python run_pipeline.py --stage evaluate`.
- **`--config`** points at a different config file, e.g. a copy for an
  experiment: `python run_pipeline.py --config config/sensitivity.yaml`.

Each script can still be run on its own (`python src/04_feature_engineering.py`)
from any folder — paths come from `scripts/paths.py`. Each prints a summary on
completion (row counts, dropped stations, imputation, validation); check it
before moving on.

### Notes on the modelling stages

- **Feature sets.** Every model is trained once per feature set in
  `training.feature_sets`: `exogenous` (traffic/weather/calendar/road/station,
  no NO2 history), `autoregressive` (NO2's own lags only) and `all`.
- **Baselines.** `train` also scores naive persistence and seasonal
  climatology — the bar every real model has to clear.
- **Scoring.** Choose between models on `val`; `test` is reported, never
  tuned on. `07_evaluation.py --trials` also prints every tuning candidate.
- **Status.** Decision tree, random forest, ridge, SVR and XGBoost are
  implemented. SHAP (`08_explainability.py`) is next.

## Setup notes

`pyshp`, `holidays`, and `python-calamine` are all required —
`01_data_ingestion.py` uses `holidays` for real NSW public-holiday dates,
`python-calamine` to read the AQ-portal `.xls` exports, and `pyshp` for the
optional posted-speed matching.

## Data scope

15 candidate stations across NSW (not Sydney-only), 2024–2025 traffic data.
4 have no NO2 coverage and 2 (Port Macquarie) are excluded for data quality,
leaving **9 stations for the daily model and 8 for hourly**. See
`Data.md` for the full station list, AQ-site match findings, and the
split/leakage decisions.