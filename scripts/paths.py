"""
Every folder and file location the project uses, defined once.

All scripts import from here instead of hard-coding "data/processed" etc.,
so moving a folder is a one-line change in this file. Paths are absolute
(built from this file's location), so scripts work no matter which folder
you run them from.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# A sensitivity run (run_pipeline.py --sensitivity <name>) sets this, and then
# its processed data, results and models go to their own folders, so it can
# never overwrite the main analysis.
SENSITIVITY_ENV = "NO2_SENSITIVITY"
SENSITIVITY = os.environ.get(SENSITIVITY_ENV) or None

# --- config -----------------------------------------------------------------
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"          # pipeline, data, training
TUNING_CONFIG_PATH = CONFIG_DIR / "tuning.yaml"   # tuning search spaces
SENSITIVITY_CONFIG_PATH = CONFIG_DIR / "sensitivity.yaml"  # sensitivity runs (changes only)

# --- raw inputs -------------------------------------------------------------
RAW_DIR = ROOT_DIR / "data" / "raw"
TRAFFIC_DIR = RAW_DIR / "traffic"
WEATHER_DIR = RAW_DIR / "weather"
EMISSIONS_DIR = RAW_DIR / "emissions"
AQ_SITES_PATH = RAW_DIR / "air_quality" / "nsw_air_quality_sites.json"
SPEED_ZONES_SHP = RAW_DIR / "speed_zones" / "Speed_Zones.shp"
STATION_REFERENCE_PATH = TRAFFIC_DIR / "station_reference.csv"

# --- processed (outputs of stages 01-05, all rebuildable) -------------------
PROCESSED_DIR = (ROOT_DIR / "data" / "processed" if not SENSITIVITY
                 else ROOT_DIR / "data" / "processed_sensitivity" / SENSITIVITY)
SPLITS_DIR = PROCESSED_DIR / "splits"
FEATURE_MANIFEST_PATH = PROCESSED_DIR / "feature_manifest.json"
SPLIT_MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"

# --- results ------------------------------------------------------------------
#   results/latest/       outputs of the main pipeline (src/), overwritten each run
#   results/experiments/  outputs of the extra checks (experiments/)
#   results/saved_models/ fitted tuned models (large, not in git)
#   results/runs/         a saved copy of every run, with its config and log
#   results/sensitivity/<name>/  everything from one sensitivity run
RESULTS_DIR = ROOT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
_BASE = RESULTS_DIR if not SENSITIVITY else RESULTS_DIR / "sensitivity" / SENSITIVITY
LATEST_DIR = _BASE / "latest"
EXPERIMENTS_RESULTS_DIR = _BASE / "experiments"
SAVED_MODELS_DIR = _BASE / "saved_models"

# main pipeline
RESULTS_PATH = LATEST_DIR / "results.json"        # every model's scores
PER_STATION_DIR = LATEST_DIR / "per_station"
PREDICTIONS_DIR = LATEST_DIR / "predictions"      # row-by-row y_true / y_pred
TUNING_DIR = LATEST_DIR / "tuning"
LEARNING_CURVE_DIR = TUNING_DIR / "learning_curves"
OVERFITTING_DIR = TUNING_DIR / "overfitting"
FEATURE_IMPORTANCE_DIR = TUNING_DIR / "feature_importance"
EVALUATION_DIR = LATEST_DIR / "evaluation"        # tables, charts, bootstrap CIs
SHAP_DIR = LATEST_DIR / "shap"
PERMUTATION_DIR = LATEST_DIR / "permutation"

# experiments
ERROR_ANALYSIS_DIR = EXPERIMENTS_RESULTS_DIR / "error_analysis"
LOSO_DIR = EXPERIMENTS_RESULTS_DIR / "loso"
ABLATION_DIR = EXPERIMENTS_RESULTS_DIR / "ablation"
HOLIDAY_DIR = EXPERIMENTS_RESULTS_DIR / "holiday"
