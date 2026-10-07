# Traffic Emissions Sydney — ML + XAI

Predicts **NO₂ concentration** near NSW traffic counting stations from
traffic volume, weather and station details, then uses explainable AI
(SHAP) to show what drives it.

This README is the manual for setting up and running the code. For the data
sources, station list and every data-quality decision, see
[`data/Data.md`](data/Data.md).

---

## 1. What you need

- **Python 3.10 or newer** — check with `python --version`
- **Git** (or just download the repo as a zip)
- About 500 MB of free disk space for processed data and results

## 2. Get the code

```bash
git clone <repo-url>
cd traffic-emissions-sydney-ml-xai
```

All commands below are run from this top-level folder.

## 3. Set up a virtual environment

A virtual environment keeps this project's packages separate from the rest
of your computer. You only do this once.

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll know it's active when your prompt starts with `(venv)`. Activate it
again every time you open a new terminal (the second line above); type
`deactivate` to leave it.

## 4. Check the data

The raw data ships with the repo in `data/raw/`:

| Folder | Contents | Required? |
|---|---|---|
| `traffic/` | TfNSW traffic counts, one CSV per station, plus `station_reference.csv` | Yes |
| `weather/` | Hourly weather (NSW AQ portal) and daily weather (BOM) | Yes |
| `emissions/` | Hourly NO₂ from the NSW Air Quality Network | Yes |
| `air_quality/` | `nsw_air_quality_sites.json` — AQ monitoring site locations | Optional |
| `speed_zones/` | TfNSW `Speed_Zones` shapefile (`.shp`, `.shx`, `.dbf`, `.prj`) | Optional |

**Speed zones are not in the repo** — the files are too big for GitHub.
Without them the pipeline still runs; it just skips the posted-speed-limit
feature and prints a warning. To include it, download the Speed Zones
dataset from Transport for NSW and put the four files in
`data/raw/speed_zones/`.

Don't rename the raw files — the scripts find them by name.

## 5. Run it

```bash
python run_pipeline.py
```

**This one command runs everything the report needs:** it builds the data,
scores the baseline methods, trains every ML model on **both daily and
hourly** data, tunes, and makes the summary tables and charts. Expect it
to take a while (the LSTM and the tuning are the slow parts; expect anywhere from tens of minutes to a few hours depending on your laptop). It prints progress as it goes and a
timing summary at the end.

**To stop a run, press Ctrl+C once.** The runner stops the current stage
(including any worker processes it started) and exits; give it a few
seconds.

Each full run **rebuilds `data/processed/` from scratch**. Results in
`results/` are kept and updated, and every run is also saved on its own
with its exact settings — see
[step 9](#9-saved-runs-and-reproducing-a-result).

### Running less

Options only ever narrow the run down; they don't change `config.yaml`.
They can be combined.

| Option | Runs | Example |
|---|---|---|
| `--no-tune` | Everything except tuning (much faster) | `python run_pipeline.py --no-tune` |
| `--grain daily` / `--grain hourly` | One grain only | `python run_pipeline.py --grain daily` |
| `--models …` | Only these ML models (baselines always run) | `python run_pipeline.py --models ridge lstm` |
| `--jobs N` | How many models train at the same time (default `auto` = one per 4 CPU cores, max 4). `--jobs 1` = one at a time | `python run_pipeline.py --jobs 4` |
| `--stage …` | One stage, reusing what's already built | `python run_pipeline.py --stage evaluate` |
| `--name …` | (Labels the saved run) | `python run_pipeline.py --name first_try` |
| `--no-save` | (Doesn't save a run record — for quick tests) | `python run_pipeline.py --stage evaluate --no-save` |

For example, to retrain just the LSTM on hourly data without rebuilding
the data, then refresh the tables:

```bash
python run_pipeline.py --stage train --grain hourly --models lstm
python run_pipeline.py --stage evaluate
```

A single stage reuses whatever earlier stages already produced. If
something it needs is missing, it tells you which stage to run first.

### The stages

| Stage | Script | What it does |
|---|---|---|
| `ingest` | `src/01_data_ingestion.py` | Combines raw traffic, weather and NO₂ into one daily and one hourly dataset |
| `clean` | `src/02_cleaning.py` | Drops stations with no NO₂ or excluded for data quality, removes bad and duplicate rows |
| `preprocess` | `src/03_data_preprocessing.py` | Fills missing values, adds station weights, encodes stations |
| `features` | `src/04_feature_engineering.py` | Builds model features and checks them |
| `split` | `src/05_train_test_split.py` | Splits by date into train / validation / test (70/15/15) and scales features |
| `train` | `src/06_run_models.py` | Scores the baseline methods, then trains and scores every enabled ML model, for each grain |
| `tune` | `scripts/tuning/tune_*.py` | Hyperparameter search for Ridge, XGBoost (daily + hourly) and SVR (daily) |
| `evaluate` | `src/07_evaluation.py` | Summary table and charts comparing every method, per grain |

## 6. The methods

There are two kinds of method, and every table and chart lists them in
this order:

| # | Method | Kind | Default |
|---|---|---|---|
| B1 | Persistence — NO₂ equals the last observed value | Baseline (not ML) | On |
| B2 | Seasonal climatology — the station's average for that month (and hour) | Baseline (not ML) | On |
| 1 | Ridge regression | ML — linear | On |
| 2 | Decision tree | ML — tree | On |
| 3 | Random forest | ML — tree ensemble | On |
| 4 | XGBoost | ML — boosted trees | On |
| 5 | SVR | ML — kernel | On, daily only (far too slow on hourly data) |
| 6 | LSTM | ML — neural network | On |

The **baselines** learn nothing from the features — they're the score an
ML model has to beat to show it has learned something. Each **ML model** is
trained three times, once per feature set: `exogenous` (traffic, weather,
calendar, station — no NO₂ history), `autoregressive` (NO₂ history only)
and `all`.

The **LSTM** sees the same features as the other models, but for the last
24 hours (hourly) or 14 days (daily) at once instead of one row, and stops
training when the validation score stops improving. It needs PyTorch,
which is in `requirements.txt`. Its settings are under
`training.models.lstm` in `config.yaml`; see `scripts/lstm.py` for details.

Results are labelled **default** (the settings in `config.yaml`) or
**tuned** (after the tune stage). Tuning currently covers XGBoost and SVR.

## 7. Change what runs

You shouldn't need to edit any code for normal experiments — everything is
in two config files. For one-off runs, the command-line options in step 5
are quicker than editing the config.

**`config/config.yaml`**

| Setting | What it controls |
|---|---|
| `pipeline.stage` | Which stage(s) run: `all` or one stage name |
| `pipeline.run_tuning` | Whether a full run includes tuning (default `true`) |
| `data.station_exclusions` | Stations left out; set `enabled: false` for the sensitivity analysis |
| `training.grain` | `[daily, hourly]` (default) or just one |
| `training.feature_sets` | `exogenous` (traffic/weather only), `autoregressive` (NO₂ history only), `all` |
| `training.baselines` | Turn each baseline method (B1, B2) on/off |
| `training.models` | Turn each ML model on/off (`enabled`), limit it to some grains (`grains: [daily]`), and set its hyperparameters (`params`) |

**`config/tuning.yaml`** — number of trials and the search space for
Ridge, XGBoost and SVR tuning.

To try a variation without touching the main config, copy it and point the
runner at the copy:

```bash
python run_pipeline.py --config config/my_experiment.yaml
```

## 8. Find the outputs

| Location | Contents |
|---|---|
| `data/processed/` | Intermediate datasets from stages 01–05 (rebuilt every full run) |
| `results/results.json` | Every model's RMSE / MAE / R² on train, val and test |
| `results/per_station/` | The same scores broken down by station |
| `results/evaluation/summary_<grain>.png` / `.csv` | **Start here** — validation and test R² for every method and feature set |
| `results/evaluation/train_val_test_<grain>.*` | The full table: RMSE / MAE / R² on train, val and test, grouped by method |
| `results/tuning/` | Tuning trials, learning curves, overfitting checks, feature importance |
| `results/saved_models/` | Fitted tuned models (`.joblib`, not committed to git) |
| `results/runs/` | A saved copy of every run with its config — see step 9 |

The evaluate stage prints one summary table per grain (one row per method).
To print the full train / val / test tables as well:

```bash
python src/07_evaluation.py --detail
```

Choose between methods using the **validation** scores. Only report the
**test** scores once, for the final model.

## 9. Saved runs and reproducing a result

Every run that trains, tunes or evaluates is saved automatically to
`results/runs/<date>_<time>_<grains>/`, and gets one line in
`results/runs/index.csv` — the quickest way to compare runs, showing the
best ML model for daily and for hourly data:

| File | What it is |
|---|---|
| `config.yaml` | The exact config the run used, including any command-line options (and `tuning.yaml` if it tuned) |
| `scores.json` | Every score the run produced |
| `run_info.json` | Date, stages and timings, git commit, Python and package versions, and a fingerprint of the raw data |
| `manifests/` | Which features, stations and date cutoffs the models actually saw |
| `outputs/` | The per-station scores, tables and charts from that run |
| `pipeline.log` | Everything printed to the terminal |

Give a run a readable name with `--name`:

```bash
python run_pipeline.py --name no_exclusions
```

**To reproduce a saved run:** check out the git commit listed in its
`run_info.json`, then run it with its own config:

```bash
git checkout <commit>
python run_pipeline.py --config results/runs/<run_id>/config.yaml
```

If `run_info.json` says `"uncommitted_changes": true`, the code had unsaved
edits at the time, so commit before important runs. If the raw-data
fingerprint (`raw_data.sha256`) differs from the original's, the input data
has changed. Every row in `results/results.json` also carries the `run_id`
that wrote it. Use `--no-save` (or `pipeline.save_run: false`) to skip
saving, e.g. for quick tests.

## 10. Common tasks

| Task | How |
|---|---|
| Run everything for the report | `python run_pipeline.py` |
| Quick check without tuning | `python run_pipeline.py --no-tune` |
| Only daily (or hourly) data | `python run_pipeline.py --grain daily` |
| Retrain some models only | `python run_pipeline.py --stage train --models ridge lstm`, then `--stage evaluate` |
| Re-run tuning only | `python run_pipeline.py --stage tune` |
| Re-make the tables and charts | `python run_pipeline.py --stage evaluate` |
| Print the full tables / every tuning trial | `python src/07_evaluation.py --detail --trials` |
| Sensitivity analysis | `data.station_exclusions.enabled: false` in `config.yaml`, run `python run_pipeline.py --no-tune --name no_exclusions`, then set it back to `true` and run the full pipeline again |
| Turn a model off permanently | `enabled: false` under that model in `training.models` |
| Change the LSTM's history window | `seq_len` under `training.models.lstm.params` |
| Add a new model | Add one line to `REGISTRY` in `scripts/models.py`, then a block under `training.models` |

`results/results.json` keeps the **latest** score for each method, grain
and feature set, so the summary tables always show the most recent run of
each. Earlier runs are never lost — they're in `results/runs/`.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` (or similar) | The virtual environment isn't active — activate it (step 3) |
| PowerShell says running scripts is disabled | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again |
| `[train] can't run -- missing [...]` | An earlier stage hasn't been run — run the stage it names, or `python run_pipeline.py` |
| `No file matching '*...*' found in data/raw/...` | A raw data file is missing or was renamed — check step 4 |
| `[Speed zones] SKIPPED` | Expected if you don't have the shapefile — see step 4 |
| A run takes too long | Use `--no-tune`, `--grain daily`, or `--models` to run less while experimenting |
| `The LSTM model needs PyTorch` | Run `pip install torch` (or `pip install -r requirements.txt`) with the virtual environment active |
| The terminal seems stuck after Ctrl+C | Wait a few seconds for the runner to stop the stage. If you ran a script directly (`python src/...`) rather than through `run_pipeline.py`, close the terminal |

## 12. Project layout

```
config/          config.yaml (what runs) and tuning.yaml (search spaces)
data/
  Data.md        data sources and every data decision
  raw/           input data (step 4)
  processed/     intermediate files, rebuilt each run
notebooks/       01_data_exploration.ipynb — exploratory analysis
results/         scores, tables, charts, tuning outputs, saved runs
scripts/         shared code: paths, model registry (models.py), LSTM (lstm.py),
                 helpers, run records, tuning
src/             the pipeline stages, 01 to 08
run_pipeline.py  runs everything
```