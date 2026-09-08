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
  04_feature_engineering.py    Pure feature creation (calendar, road type, lags,
                                cyclical encoding, target transform) + the lag-warmup
                                cleanup that creation itself introduces
  05_models.py                 (not yet built)
  06_evaluation.py             (not yet built)
  07_explainability.py         (not yet built)
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
4. **`04_feature_engineering.py`** — calendar features, road-type encoding,
   traffic ratios, weather flags, log-target transform, lag/rolling
   features (daily) or cyclical hour encoding (hourly). Outputs
   `features_daily.csv` / `features_hourly.csv` — zero NaNs, model-ready.

Each script prints a summary on completion (row counts, dropped stations,
imputation coverage). Check this output before moving to the next step.

## Data scope

15 candidate stations across NSW (not Sydney-only), 2024–2025 traffic
data. 11 stations have usable NO2 coverage for the daily model; 8 of
those also have hourly-resolution weather and NO2 for the hourly model.
See `doc/Data.md` for the full station list, the AQ-site match-quality
findings, and every data-quality decision and its justification.