# Traffic Emissions Sydney — ML + XAI

Predicting **NO₂ concentration** near NSW traffic stations from traffic
volume, weather, and station metadata, using machine learning, with
explainable AI (SHAP) to identify key drivers.

> The project originally targeted a *computed* CO₂-estimate variable
> across 3 Sydney-only stations. It has since pivoted to a **directly
> measured** NO₂ target (per supervisor guidance — computed/estimated
> targets weren't acceptable) across 15 candidate stations spanning NSW,
> not just Sydney. See `doc/Data.md` for the full history and current
> data-quality decisions.

## Project structure

```
data/
  raw/         Raw downloaded files — not tracked in git
    traffic/       TfNSW Traffic Volume Viewer exports, per station
    weather/       Metro AQ-portal hourly weather + BOM rural daily weather
    emissions/     Metro AQ-portal hourly pollutant data (incl. NO2)
    air_quality/   nsw_air_quality_sites.json — optional, enables AQ-site verification
    speed_zones/   Speed_Zones.{shp,shx,dbf,prj} — optional, enables posted-speed
                   matching. NOT in git (individual files exceed GitHub's size
                   limits) — download from TfNSW and place here manually.
  processed/   Output of each pipeline stage (see "Running the pipeline" below)
config/
  models_config.yaml           Which models run, their hyperparameters, and
                                the grain/feature_set/split to evaluate on.
                                Only random_forest is enabled so far.
models/
  __init__.py                  Registry mapping a config model name to its
                                build_model() function. Only random_forest
                                is registered right now.
  random_forest.py             Defines ONLY the estimator (a few lines —
                                NAME + build_model()). decision_tree.py,
                                svr.py, xgboost_model.py follow the same
                                pattern once this one is confirmed working.
notebooks/
  01_data_exploration.ipynb     EDA — the most current/complete notebook
  02_feature_engineering.ipynb  (not yet built)
  03_model_experiments.ipynb    (not yet built)
  04_xai_analysis.ipynb         (not yet built)
src/
  01_data_ingestion.py         Combine raw traffic/weather/emissions into one dataset
                                per resolution, plus AQ-site verification and
                                posted-speed matching
  02_eda.py                    Row-level cleaning (drop no-NO2-coverage stations,
                                sanity checks, dedup) — despite the filename, this
                                is preprocessing, not exploratory analysis; see note below
  03_data_preprocessing.py     Outlier reporting + AQ-match-quality weighting,
                                missing-value imputation, redundant-column drops,
                                station encoding
  04_feature_engineering.py    Feature creation: calendar (cyclical + season one-hot),
                                station metadata, road-type encoding (both RMS
                                classification and a coarse Highway/Major Road/Local
                                Street bucket), directional traffic, exact-timestamp
                                lags + rolling windows (with availability flags, no
                                row-dropping), weather physics, target transforms.
                                Writes feature_manifest.json listing exogenous vs.
                                autoregressive features per grain.
  05_train_test_split.py       Chronological, per-station 70:15:15 train/val/test split
                                + feature scaling (fit on train only). Writes
                                data/processed/splits/{grain}_{train,val,test}.csv,
                                {grain}_scaler.joblib, and split_manifest.json.
  model_utils.py               Shared load_split() / evaluate() / save_result()
                                used by 06_run_models.py, so every model trains
                                and is scored on the exact same rows/features and
                                all results land in one place
                                (data/processed/model_results/results.json).
  06_run_models.py             Reads config/models_config.yaml, trains + evaluates
                                every enabled model from models/, appends results.
                                Currently only random_forest is wired up.
  07_deep_learning_*.py        (not yet built) LSTM + GRU on the hourly grain
  08_hyperparameter_tuning.py  (not yet built)
  09_evaluation.py             (not yet built) Final RMSE/MAE/R² comparison table

                                across every model in model_results/results.json
  10_explainability.py         (not yet built) SHAP on the best model
results/       Model outputs, figures, SHAP plots
```

**Note on `02_eda.py`:** the numbering here doesn't match the content —
it's a leftover from a mid-project renumbering. It currently holds
preprocessing logic (drop stations with no NO2 coverage, validate ranges,
deduplicate), not EDA. The real exploratory work lives in
`notebooks/01_data_exploration.ipynb`. Rename fix pending — flagging here
so nobody goes looking for EDA logic in the wrong file.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pyshp`, `holidays`, and `python-calamine` are all required (not just
`pandas`/`numpy`) — `01_data_ingestion.py` uses `holidays` for real NSW
public-holiday dates, `python-calamine` to read the AQ-portal `.xls`
exports, and `pyshp` for the optional posted-speed matching.

## Running the pipeline

Run every script **from the repo root**, not from inside `src/` — all
paths (`data/raw/...`, `data/processed/...`) are relative to the project
root, not to the script's own location.

```powershell
python src/01_data_ingestion.py
python src/02_eda.py
python src/03_data_preprocessing.py
python src/04_feature_engineering.py
python src/05_train_test_split.py --grain all
python src/06_run_models.py
```

1. **`01_data_ingestion.py`** — builds `data/processed/final_combined_dataset_daily.csv`
   (all 15 candidate stations) and `final_combined_dataset_hourly.csv` (the
   8 stations with both hourly weather and hourly NO2). Also runs AQ-site
   verification and posted-speed matching if the optional files under
   `data/raw/air_quality/` and `data/raw/speed_zones/` are present —
   skipped with a warning otherwise, not a failure.
2. **`02_eda.py`** — drops stations with zero NO2 coverage, validates
   traffic/temperature ranges, drops rows missing the NO2 target,
   deduplicates. Outputs `preprocessed_daily.csv` / `preprocessed_hourly.csv`.
3. **`03_data_preprocessing.py`** — reports NO2 outliers per station
   (doesn't remove them — see `doc/Data.md`), adds `aq_quality_weight`,
   imputes missing weather values, drops redundant/unused columns,
   one-hot encodes station ID. Outputs `final_daily.csv` / `final_hourly.csv`.
4. **`04_feature_engineering.py`** — calendar (cyclical + season one-hot),
   station metadata, two road-type encodings (RMS classification + a
   coarse Highway/Major Road/Local Street bucket), directional traffic,
   exact-timestamp lag and rolling-window features with availability flags,
   weather physics (wind decomposition, dispersion proxy), target
   transforms (`target_no2_log1p`, `target_no2_sqrt`). Outputs
   `features_daily.csv` / `features_hourly.csv` (zero NaNs, model-ready)
   and `feature_manifest.json` (exogenous vs. autoregressive feature lists
   per grain — useful for choosing what a model is allowed to see).

5. **`05_train_test_split.py`** — splits each grain into train/val/test
   **chronologically, per station** (not randomly — see the script's
   docstring for why), and fits feature scaling on the train split only.
   Outputs `data/processed/splits/{grain}_{train,val,test}.csv`,
   `{grain}_scaler.joblib`, and `split_manifest.json`.
6. **`06_run_models.py`** — reads `config/models_config.yaml` and trains +
   evaluates every enabled model from `models/`, appending results to
   `data/processed/model_results/results.json`. Only `random_forest` is
   built and enabled right now — expect one line of output per feature
   set (`exogenous`, `all`).

`07`–`10` remain placeholders. A working prototype covering all four
baseline models (decision tree, random forest, SVR, XGBoost) was built
and verified end-to-end before being scoped back down to just
random_forest — it's kept in `src/_archive/models_prototype/` for
reference; the other three model files follow the exact same pattern as
`models/random_forest.py` once this one's confirmed working.

Each script prints a summary on completion (row counts, dropped stations,
imputation coverage). Check this output before moving to the next step.

## Data scope

15 candidate stations across NSW (not Sydney-only), 2024–2025 traffic
data. 11 stations have usable NO2 coverage for the daily model; 8 of
those also have hourly-resolution weather and NO2 for the hourly model.
See `doc/Data.md` for the full station list, the AQ-site match-quality
findings, and every data-quality decision and its justification.