# Traffic-related NO₂ near NSW roads — ML + XAI

Predicts **NO₂ concentration** near NSW traffic counting stations from
traffic, weather and calendar data, compares 7 ML models against 2 simple
baselines, and uses explainable AI (SHAP and other checks) to show what
drives NO₂.

Data sources and every data decision: [`data/Data.md`](data/Data.md).

---

## Quick start (5 minutes)

Needs **Python 3.10+**. Run everything from this folder.

```powershell
# 1. one-off setup (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. quick test run, no tuning (~20-30 min)
python run_pipeline.py --no-tune --name quick_test
```

On macOS / Linux use `python3 -m venv venv` and `source venv/bin/activate`.
Activate the environment again in every new terminal.

**Full run for the report** (tuning included, ~4 hours) — set the CPU
threads in the same terminal first:

```powershell
$env:OMP_NUM_THREADS=6
python run_pipeline.py --jobs 6 --name final
```

Press **Ctrl+C once** to stop a run. Keep the laptop plugged in and awake.

---

## What a run does

| Stage | Script | What it does |
|---|---|---|
| `ingest` | `01_data_ingestion.py` | Joins raw traffic, weather and NO₂ into a daily and an hourly dataset |
| `clean` | `02_cleaning.py` | Drops stations with no NO₂ or excluded for data quality; removes bad rows |
| `preprocess` | `03_data_preprocessing.py` | Fills gaps (train data only), station weights, outlier report |
| `features` | `04_feature_engineering.py` | Lags, rolling means, traffic × weather, calendar features |
| `split` | `05_train_test_split.py` | Splits by date: 70% train / 15% validation / 15% test |
| `train` | `06_run_models.py` | Baselines + every ML model, both grains, 3 feature sets |
| `tune` | `scripts/tuning/` | Random search per model, chosen on validation (the slow stage) |
| `evaluate` | `07_evaluation.py` | Comparison tables and charts |
| `explain` | `08_explainability.py` | SHAP on the tuned XGBoost |
| `analysis` | `09` – `15` | Confidence intervals, error analysis, unseen stations, ablation, permutation importance, public-holiday check, top-3 model comparison |

All scripts are in `src/`. **Methods:** B1 persistence and B2 seasonal
average (baselines), Ridge, decision tree, random forest, XGBoost, SVR
(daily only), LSTM, GRU (hourly only). **Feature sets:** `exogenous`
(traffic, weather, calendar, station — no NO₂ history), `autoregressive`
(NO₂ history only), `all`.

---

## Where the results are

Everything goes in `results/`. Each run is also saved in
`results/runs/<date>_<name>/` with its exact config, scores and log.

| Folder | What it answers | Start with |
|---|---|---|
| `evaluation/` | Which model is best? (with 95% confidence intervals) | `summary_<grain>.png`, `bootstrap_<grain>_test.png` |
| `shap/` | What drives NO₂, and how? | `summary_<grain>_exogenous.png`, `traffic_parts_*`, `waterfall_*` |
| `permutation/` | Does a second method agree with SHAP? Do the top 3 models (incl. LSTM / GRU) rely on the same drivers? | `permutation_families_*.csv`, `top3_<grain>_<set>.png`, `top3_agreement_*.csv` |
| `ablation/` | What happens if a feature group is removed? | `ablation_<grain>.png` |
| `holiday/` | Does less traffic (public holidays) mean less NO₂? | `holiday_hourly_profile.png` |
| `loso/` | Does the model work at a station it has never seen? | `loso_<grain>_exogenous.png` |
| `error_analysis/` | When and where are the errors largest? | `errors_*_traffic_level.png`, `errors_*_no2_level.png` |
| `tuning/` | Tuning trials, learning curves | `<model>_tuned_metrics.csv` |
| `per_station/`, `predictions/` | Raw per-station scores and row-by-row predictions (used by the analysis stage) | — |

Stations appear as id + road + suburb, e.g. `7212 Stewart Ave, Newcastle West`.

---

## Running less

| Option | Effect |
|---|---|
| `--no-tune` | Skip tuning (much faster) |
| `--no-analysis` | Skip the analysis stage |
| `--grain daily` / `hourly` | One grain only |
| `--models ridge xgboost` | Only these ML models (baselines always run) |
| `--stage explain` | One stage only, reusing what's already built |
| `--jobs N` | Models trained at the same time |
| `--config <file>` | Use another config file |
| `--name <label>` / `--no-save` | Name the saved run / don't save it |

---

## Settings

- **`config/config.yaml`** — which stages, grains, feature sets and models
  run, model settings, station exclusions, SHAP options.
- **`config/tuning.yaml`** — tuning trials and search ranges.
- **Sensitivity runs** — same as the main config with one change each:
  `no2_clipped.yaml` (negative NO₂ set to 0), `no_exclusions.yaml`
  (Port Macquarie stations kept), `no_mub001.yaml` (MUB001 left out). Run
  with e.g. `python run_pipeline.py --config config/no2_clipped.yaml --no-tune --name sens_no2_clipped`.
  **A full run afterwards is needed** to rebuild the main data.

---

## Data and stations

Raw data lives in `data/raw/` — don't rename the files:

| Folder | Contents |
|---|---|
| `traffic/` | TfNSW counts, one `<id> - <road>.csv` per station |
| `weather/` | Hourly AQ-network weather (`.xls`) and BOM daily weather (`.csv`) |
| `emissions/` | Hourly NO₂ from the NSW Air Quality Network (`.xls`) |
| `air_quality/` | AQ monitor locations (optional) |
| `speed_zones/` | TfNSW Speed Zones `.shp` + `.dbf` (optional, not in git — too large) |

**Stations are defined once, in `scripts/stations.py`.** To add one: add a
line there, then put its traffic file in `traffic/` and its weather and NO₂
files in `weather/` and `emissions/`.

---

## Reproducing a result

`results/runs/<run>/run_info.json` records the git commit, timings and a
fingerprint of the raw data. Commit before important runs, then:

```bash
git checkout <commit>
python run_pipeline.py --config results/runs/<run>/config.yaml
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named ...` | Activate the virtual environment |
| PowerShell won't activate it | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again |
| `No file matching ... in data/raw/...` | A raw file is missing or renamed |
| `[train] can't run -- missing [...]` | Run the stage it names first, or the full pipeline |
| `PermissionError` on a results file | Close it in Excel / VS Code; the run retries automatically |
| Run is slow | Use `--no-tune`, `--grain daily` or `--models`; set `OMP_NUM_THREADS` |

---

## Project layout

```
config/          config.yaml, tuning.yaml, sensitivity configs
data/            Data.md, raw/ (inputs), processed/ (rebuilt each run)
notebooks/       01_data_exploration.ipynb
results/         all outputs and saved runs
scripts/         shared code: stations, paths, models, LSTM/GRU, tuning
src/             pipeline stages 01-15
run_pipeline.py  runs everything
```