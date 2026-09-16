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
  models_config.yaml           Which models run, their hyperparameters, the
                                grain/feature_set/split to evaluate on, and
                                which pipeline stage(s) src/run_pipeline.py
                                runs. random_forest, decision_tree, and ridge
                                are enabled by default.
  xgboost_baseline.yaml        Reproducible XGBoost validation baseline using
                                the same daily data and feature sets.
  xgboost_tuning.yaml          Daily/hourly XGBoost search space and settings.
models/
  __init__.py                  Registry mapping a config model name to its
                                build_model() function.
  random_forest.py             Defines ONLY the estimator (a few lines —
                                NAME + build_model()). decision_tree.py,
                                ridge.py, and xgboost_model.py follow the
                                same pattern.
  train_xgboost.py             Tunes XGBoost on daily/hourly tabular features,
                                saves models, metrics, trials, and importances.
notebooks/
  01_data_exploration.ipynb     EDA — the most current/complete notebook
  02_feature_engineering.ipynb  (not yet built)
  03_model_experiments.ipynb    (not yet built)
  04_xai_analysis.ipynb         (not yet built)
src/
  run_pipeline.py              Runner: executes 01-06 in order (or just one
                                stage) based on the `pipeline.stage` setting
                                in config/models_config.yaml. See "Running
                                the pipeline" below.
  01_data_ingestion.py         Combine raw traffic/weather/emissions into one dataset
                                per resolution, plus AQ-site verification and
                                posted-speed matching
  02_eda.py                    Row-level cleaning (drop no-NO2-coverage stations,
                                sanity checks, dedup) — despite the filename, this
                                is preprocessing, not exploratory analysis; see note below
  03_data_preprocessing.py     Outlier reporting + AQ-match-quality weighting,
                                missing-value imputation, redundant-column drops,
                                station encoding. Imputation medians are fit on
                                TRAIN-only rows (via split_utils.py) so no
                                val/test-period value leaks into a training row.
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
  split_utils.py               Shared per-station chronological split logic used by
                                BOTH 03 (to fit train-only imputation medians) and 05
                                (to make the actual train/val/test split) — one
                                definition of "train" so the two steps can't disagree.
  model_utils.py               Shared load_split() / evaluate() / save_result()
                                used by 06_run_models.py, so every model trains
                                and is scored on the exact same rows/features and
                                all results land in one place
                                (data/processed/model_results/results.json).
  06_run_models.py             Reads config/models_config.yaml, trains + evaluates
                                every enabled model from models/, appends results.
                                XGBoost can be run with xgboost_baseline.yaml.
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
root, not to the script's own location. `src/run_pipeline.py` already
does this for you (it sets the working directory itself), so it's the
easiest way to run things either way.

### Recommended: `run_pipeline.py`

```powershell
python src/run_pipeline.py
```

It reads `pipeline.stage` from `config/models_config.yaml`:

```yaml
pipeline:
  stage: all       # run every stage, 01 -> 06, wiping data/processed/
                   # first (except model_results/, the cross-run results
                   # log) so this run never mixes with a stale one
  stage: models    # run ONLY 06_run_models.py, reusing whatever's
                   # already in data/processed/splits/ — the fast path
                   # once everything through 05 is already built
  # or: ingest | eda | preprocess | features | split
```

`--stage` on the command line overrides the yaml for a one-off, e.g.
`python src/run_pipeline.py --stage features` to redo just feature
engineering without touching the yaml. Before running a single stage it
checks the previous stage's output files actually exist and tells you
plainly which stage to run first if not, instead of letting the script
fail with a less obvious error. It streams each stage's own print
output live, times it, and prints a summary table at the end.

### Manual, stage by stage

Equivalent to `stage: all`, run one script at a time if you want to
inspect the output between steps:

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
   one-hot encodes station ID. Imputation medians are fit on TRAIN-only
   rows (via `split_utils.py`, the same per-station chronological split
   `05` uses) so a training row's imputed value never carries information
   from a row that ends up in val/test. Outputs `final_daily.csv` /
   `final_hourly.csv`.
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
   `data/processed/model_results/results.json`. Run the XGBoost baseline
   with `python src/06_run_models.py --config config/xgboost_baseline.yaml`.
   Expect one line of output per feature set (`exogenous`,
   `autoregressive`, `all`).

   Tune and train XGBoost across daily and hourly data with
   `python models/train_xgboost.py`. This writes the best RMSE/MAE/R² results,
   fitted models, trial history, and feature importance tables under
   `data/processed/model_results/`.

`07`–`10` remain placeholders. Decision tree, random forest, ridge, and
XGBoost are implemented through the shared model registry. The earlier
prototype remains in `src/_archive/models_prototype/` for reference.

Each script prints a summary on completion (row counts, dropped stations,
imputation coverage). Check this output before moving to the next step.

## Data scope

15 candidate stations across NSW (not Sydney-only), 2024–2025 traffic
data. 11 stations have usable NO2 coverage for the daily model; 8 of
those also have hourly-resolution weather and NO2 for the hourly model.
See `doc/Data.md` for the full station list, the AQ-site match-quality
findings, and every data-quality decision and its justification.
