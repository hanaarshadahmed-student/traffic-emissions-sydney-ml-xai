"""
Paths: every folder and file location the project uses, defined once so scripts don't hard-code them and work from any folder.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
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

# --- results -------------------------------------------------------------------
#   results/              ONLY the curated results the report is written from
#                         (README + run/, 1_data/ ... 5_errors/), rebuilt by
#                         src/11_final_results.py after every complete run. Tracked in git.
#   working/              everything the stages pass to each other -- not results.
#                         Emptied at the start of every full run. Not in git.
#     working/latest/        outputs of the main pipeline (src/)
#     working/experiments/   outputs of the extra checks (experiments/)
#     working/saved_models/  fitted tuned models (reused by the explain stage)
#     working/runs/<id>/     the last run's complete record: config, log, scores,
#                            every output (~750 files) -- the backup
#     working/sensitivity/<name>/  the same for one sensitivity run
RESULTS_DIR = ROOT_DIR / "results"
WORKING_DIR = ROOT_DIR / "working"
RUNS_DIR = WORKING_DIR / "runs"
WORKING_BASE = WORKING_DIR if not SENSITIVITY else WORKING_DIR / "sensitivity" / SENSITIVITY
LATEST_DIR = WORKING_BASE / "latest"
EXPERIMENTS_RESULTS_DIR = WORKING_BASE / "experiments"
SAVED_MODELS_DIR = WORKING_BASE / "saved_models"

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
