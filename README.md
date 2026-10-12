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

**Main pipeline** (`src/`, always runs):

| Stage | Script | What it does |
|---|---|---|
| `ingest` | `01_data_ingestion.py` | Joins raw traffic, weather and NO₂ into a daily and an hourly dataset |
| `clean` | `02_cleaning.py` | Drops stations with no NO₂ or excluded for data quality; removes bad rows |
| `preprocess` | `03_data_preprocessing.py` | Fills gaps (train data only), station weights, outlier report |
| `features` | `04_feature_engineering.py` | Lags, rolling means, traffic × weather, calendar features |
| `split` | `05_train_test_split.py` | Splits by date: 70% train / 15% validation / 15% test |
| `train` | `06_run_models.py` | Baselines + every ML model, both grains, 3 feature sets |
| `tune` | `models/tuning/` | Random search per model, chosen on validation (the slow stage) |
| `evaluate` | `07_evaluation.py`, `08_bootstrap_ci.py` | Comparison tables and charts, with 95% confidence intervals |
| `explain` | `09_shap_explainability.py`, `10_permutation_importance.py` | SHAP on the tuned XGBoost; permutation importance on the top 3 models |

**Experiments** (`experiments/`, only with `--experiments`): robustness checks.

| Script | Question |
|---|---|
| `01_error_analysis.py` | When and where are the errors largest? |
| `02_leave_one_station_out.py` | Does the model work at a station it has never seen? |
| `03_ablation.py` | What happens if a feature group is removed and the model retrained? |
| `04_holiday_check.py` | Does less traffic (weekday public holidays) mean less NO₂? |

**Methods:** B1 persistence and B2 seasonal average (baselines), Ridge,
decision tree, random forest, XGBoost, SVR (daily only), LSTM, GRU (hourly
only). **Feature sets:** `exogenous` (traffic, weather, calendar, station —
no NO₂ history), `autoregressive` (NO₂ history only), `all`.

---

## Where the results are

| Folder | What it answers | Start with |
|---|---|---|
| `results/latest/evaluation/` | Which model is best? (with 95% confidence intervals) | `summary_<grain>.png`, `bootstrap_<grain>_test.png` |
| `results/latest/shap/` | What drives NO₂, and how? | `summary_<grain>_exogenous.png`, `traffic_parts_*`, `waterfall_*` |
| `results/latest/permutation/` | Does a second method agree with SHAP? Do the top 3 models (incl. LSTM / GRU) rely on the same drivers? | `top3_<grain>_<set>.png`, `top3_agreement_*.csv` |
| `results/latest/tuning/` | Tuning trials, learning curves | `<model>_tuned_metrics.csv` |
| `results/latest/per_station/`, `predictions/` | Raw per-station scores and row-by-row predictions | — |
| `results/experiments/` | `error_analysis/`, `loso/`, `ablation/`, `holiday/` | `errors_*_traffic_level.png`, `loso_<grain>_exogenous.png`, `holiday_hourly_profile.png` |
| `results/sensitivity/<name>/` | The same folders for one sensitivity run | `latest/evaluation/summary_*.png` |
| `results/saved_models/` | Fitted tuned models (reused by `explain`) | — |
| `results/runs/` | A saved copy of every run: config, scores, log, outputs | `index.csv` |

Stations appear as id + road + suburb, e.g. `7212 Stewart Ave, Newcastle West`.

---

## Running more or less

| Option | Effect |
|---|---|
| `--experiments` | Also run the experiments |
| `--no-tune` | Skip tuning (much faster) |
| `--grain daily` / `hourly` | One grain only |
| `--models ridge xgboost` | Only these ML models (baselines always run) |
| `--stage explain` | One stage only, reusing what's already built (`--stage experiments` for the experiments) |
| `--jobs N` | Models trained at the same time |
| `--sensitivity <name>` | A sensitivity run (see below) |
| `--config <file>` | Use another config file |
| `--name <label>` / `--no-save` | Name the saved run / don't save it |

---

## Settings

- **`config/config.yaml`** — which stages, grains, feature sets and models
  run, model settings, station exclusions, SHAP options.
- **`config/tuning.yaml`** — tuning trials and search ranges.
- **`config/sensitivity.yaml`** — sensitivity runs, each listing only what it
  changes: `no2_clipped` (negative NO₂ set to 0), `no_exclusions` (Port
  Macquarie stations kept), `no_mub001` (MUB001 left out). Run with e.g.
  `python run_pipeline.py --sensitivity no2_clipped`. They use their own
  folders (`data/processed_sensitivity/`, `results/sensitivity/`), so the
  main data and results are never overwritten.

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
config/          config.yaml, tuning.yaml, sensitivity.yaml
data/            Data.md, raw/ (inputs), processed/ (rebuilt each run)
notebooks/       exploratory analysis
models/          the models (models.py, sequence_models.py = LSTM + GRU) and tuning/
scripts/         shared helpers: paths, stations, feature families, scoring, run records
src/             main pipeline, stages 01-10
experiments/     robustness checks 01-04
results/         all outputs (latest/, experiments/, sensitivity/, saved_models/, runs/)
run_pipeline.py  runs everything
```