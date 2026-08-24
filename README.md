# Traffic Emissions Sydney — ML + XAI

Predicting traffic-related CO₂ emissions across selected Sydney roads using
machine learning, with explainable AI (SHAP) to identify key drivers.

See `DATA.md` for full details on data sources, collection decisions, and
the final processed dataset.

## Project structure

```
data/
  raw/         Raw downloaded files (traffic, weather, NGA factors) — not tracked in git
  processed/   Output of the preprocessing pipeline (merged_dataset.csv)
notebooks/     Exploration, feature engineering, modelling, XAI (in order)
src/
  01_data_ingestion.py
  02_data_preprocessing.py   
  03_feature_engineering.py  (next step)
  04_models.py                (next step)
  05_evaluation.py             (next step)
  05_explainability.py         (next step)
results/       Model outputs, figures, SHAP plots
```

## Setup
#For windows
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the pipeline

### 1. Build the merged dataset

```powershell
python src/data_preprocessing.py
```

Reads all files in `data/raw/`, builds one merged hourly dataset per traffic
station (traffic volume + weather + temporal features + computed CO₂ target),
and writes it to `data/processed/merged_dataset.csv`.

Prints a summary on completion: total rows, missing weather values, and
per-station row counts. Check this output before moving on — large gaps
here should be investigated, not ignored.

### 2. Next steps (not yet built)

- `notebooks/01_data_exploration.ipynb` — EDA on `merged_dataset.csv`
- `notebooks/02_feature_engineering.ipynb` — lag features, cyclical encoding
- `notebooks/03_model_experiments.ipynb` — RF, XGBoost, LSTM comparison
- `notebooks/04_xai_analysis.ipynb` — SHAP analysis

## Data scope

3 traffic stations across 2 Sydney LGAs (Parramatta, Bankstown), 2 years of
hourly data (2024–2025). See `DATA.md` for why these specific stations and
the full list of data-quality decisions.