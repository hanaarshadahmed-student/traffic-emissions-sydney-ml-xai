# Traffic-related NO₂ near NSW roads — ML + XAI

Predicts **NO₂ concentration** near NSW traffic counting stations from
traffic, weather and calendar data, compares 7 ML models against 2 simple
baselines, and uses explainable AI (SHAP and other checks) to show what
drives NO₂.

Data sources and every data decision: [`data/Data.md`](data/Data.md).

---

## Quick start

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

**Full run for the report** (tuning + experiments, ~4 hours). Set the CPU
threads in the same terminal first:

```powershell
$env:OMP_NUM_THREADS=6
python run_pipeline.py --jobs 6 --experiments --name final
```

Press **Ctrl+C once** to stop a run. Keep the laptop plugged in and awake.

---

## What a run does

**Main pipeline** (`src/`):

| Stage | Script | What it does |
|---|---|---|
| `ingest` | `01_data_ingestion.py` | Joins raw traffic, weather and NO₂ into a daily dataset (all stations) and an hourly dataset (metro stations only) |
| `clean` | `02_cleaning.py` | Drops stations with no NO₂ or excluded for data quality; removes bad rows |
| `preprocess` | `03_data_preprocessing.py` | Fills gaps (train data only), station weights, outlier report |
| `features` | `04_feature_engineering.py` | Lags, rolling means, traffic × weather, calendar features |
| `split` | `05_train_test_split.py` | Splits by date: 70% train / 15% validation / 15% test; scales on train only |
| `train` | `06_run_models.py` | Baselines + every ML model, both grains, 3 feature sets |
| `tune` | `models/tuning/` | Random search per model, chosen on validation (the slow stage) |
| `evaluate` | `07_evaluation.py`, `08_bootstrap_ci.py` | Comparison tables and charts, with 95% confidence intervals |
| `explain` | `09_shap_explainability.py`, `10_permutation_importance.py` | SHAP on the tuned XGBoost; permutation importance on the top 3 models |
| `final` | `11_final_results.py` | Copies the files the report uses into `results/` (complete runs only) |

**Experiments** (`experiments/`, only with `--experiments`):

| Script | Question |
|---|---|
| `01_error_analysis.py` | When and where are the errors largest? |
| `02_leave_one_station_out.py` | Does the model work at a station it has never seen? |
| `03_ablation.py` | What happens if a feature group is removed and the model retrained? |
| `04_holiday_check.py` | Does less traffic (weekday public holidays) mean less NO₂? |

**Methods:** B1 persistence and B2 seasonal average (baselines); Ridge,
decision tree, random forest, XGBoost, SVR (daily only: too slow on ~105k
hourly rows), LSTM, GRU.

**Feature sets:** `exogenous` (traffic, weather, calendar, station; no NO₂
history), `autoregressive` (NO₂ history only), `all` (both).

---

## Where the results are

| Folder | What's in it |
|---|---|
| `results/` | **Only what the report uses** (~50 files, in git). Rebuilt after every complete run; quick runs never replace it. Its `README.md` says what each file shows and which research question it answers. |
| `working/` | Everything else (not in git): files passed between stages, saved models, and each run's full record in `working/runs/<run_id>/` (~750 files, the backup). Emptied at the start of the next full run. |

Inside `results/`:

| Folder | Report section |
|---|---|
| `1_data/` | Methodology: stations, cleaning, missing data, features, split |
| `2_models/` | Which model is best, 95% CIs, tuning gains, timings (RQ3) |
| `3_drivers/` | What drives NO₂: SHAP, traffic × wind, ablation, holiday test (RQ1, RQ2) |
| `4_explainability/` | Do models and methods agree (RQ4) |
| `5_errors/` | When and where models fail; unseen stations (RQ3) |
| `run/` | Config, run info and log to replicate the run |

Need a file that isn't in `results/`? It's in `working/runs/<run_id>/outputs/`.
To rebuild `results/` from a saved run:
`python src/11_final_results.py --run working/runs/<run_id>`.

---

## Common tasks

| I want to... | Command |
|---|---|
| Check everything works | `python run_pipeline.py --no-tune --name quick_test` |
| Do the full run for the report | `python run_pipeline.py --jobs 6 --experiments --name final` |
| Re-run one stage after a change (reuses what's built) | `python run_pipeline.py --stage explain` |
| Run only the experiments | `python run_pipeline.py --stage experiments` |
| Retrain or tune just one model | `python run_pipeline.py --stage tune --models xgboost` |
| See the full train / val / test tables | `python src/07_evaluation.py --detail` |
| Explain hourly data with external factors only | `python src/09_shap_explainability.py --grain hourly --feature-set exogenous` |
| Test more models on unseen stations | `python experiments/02_leave_one_station_out.py --grain daily --models ridge xgboost random_forest` |
| Run a sensitivity check | `python run_pipeline.py --sensitivity no2_clipped` |
| Rebuild `results/` from a saved run | `python src/11_final_results.py --run working/runs/<run_id>` |
| Add a station | Add a line to `scripts/stations.py`, then add its raw files (see Data and stations) |
| Add a model | Add one line to `REGISTRY` in `models/models.py` and a block under `training.models` in `config/config.yaml` |
| Turn a model on / off or change its settings | Edit `training.models` in `config/config.yaml` |

Every script in `src/` and `experiments/` can also be run on its own; the
docstring at the top of each file shows its options.

---

## Running more or less

| Option | Effect |
|---|---|
| `--experiments` | Also run the experiments |
| `--no-tune` | Skip tuning (much faster) |
| `--grain daily` / `hourly` | One grain only |
| `--models ridge xgboost` | Only these ML models (baselines always run) |
| `--stage explain` | One stage only, reusing what's already built (`--stage experiments` for the experiments) |
| `--jobs N` | Train N models at the same time |
| `--sensitivity <name>` | A sensitivity run (see Settings) |
| `--config <file>` | Use another config file |
| `--name <label>` / `--no-save` | Name the saved run / don't save it |

Options combine, e.g. `--stage train --grain hourly --models lstm`.

---

## Settings

- **`config/config.yaml`**: which stages, grains, feature sets and models
  run; model settings; station exclusions; SHAP options.
- **`config/tuning.yaml`**: tuning trials and search ranges.
- **`config/sensitivity.yaml`**: sensitivity runs, each listing only what it
  changes: `no2_clipped` (negative NO₂ set to 0), `no_exclusions` (Port
  Macquarie stations kept), `no_mub001` (MUB001 left out). Run with e.g.
  `python run_pipeline.py --sensitivity no2_clipped`. They write to their own
  folders (`data/processed_sensitivity/`, `working/sensitivity/`), so the
  main results are never overwritten. Tuning is off for them.

---

## Data and stations

Raw data lives in `data/raw/`. Don't rename the files.

| Folder | Contents |
|---|---|
| `traffic/` | TfNSW counts, one `<id> - <road>.csv` per station |
| `weather/` | Hourly AQ-network weather (`.xls`) and BOM daily weather (`.csv`) |
| `emissions/` | Hourly NO₂ from the NSW Air Quality Network (`.xls`) |
| `air_quality/` | AQ monitor locations (optional) |
| `speed_zones/` | TfNSW Speed Zones `.shp` + `.dbf` (optional, not in git: too large) |

**Stations are defined once, in `scripts/stations.py`.** To add one: add a
line there, then put its traffic file in `traffic/` and its weather and NO₂
files in `weather/` and `emissions/`. Stations appear in outputs as
id + road + suburb, e.g. `7212 Stewart Ave, Newcastle West`.

---

## Reproducing a result

`results/run/run_info.json` records the git commit, Python and package
versions, timings and a fingerprint of the raw data. Commit before important
runs, then:

```bash
git checkout <commit>
python run_pipeline.py --config results/run/config.yaml --experiments
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
run_pipeline.py  runs everything
config/          config.yaml, tuning.yaml, sensitivity.yaml
data/            Data.md, raw/ (inputs), processed/ (rebuilt each run)
src/             main pipeline, stages 01-11
models/          models.py (registry), sequence_models.py (LSTM + GRU), tuning/
scripts/         shared helpers: paths, stations, feature families, splits, scoring, run records
experiments/     robustness checks 01-04
notebooks/       exploratory analysis
results/         curated outputs for the report (in git)
working/         intermediate files, saved models, full run backups (not in git)
```