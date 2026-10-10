"""
Every folder and file location the project uses, defined once.

All scripts import from here instead of hard-coding "data/processed" etc.,
so moving a folder is a one-line change in this file. Paths are absolute
(built from this file's location), so scripts work no matter which folder
you run them from.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# --- config -----------------------------------------------------------------
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"          # pipeline, data, training
TUNING_CONFIG_PATH = CONFIG_DIR / "tuning.yaml"   # XGBoost / SVR search spaces

# --- raw inputs -------------------------------------------------------------
RAW_DIR = ROOT_DIR / "data" / "raw"
TRAFFIC_DIR = RAW_DIR / "traffic"
WEATHER_DIR = RAW_DIR / "weather"
EMISSIONS_DIR = RAW_DIR / "emissions"
AQ_SITES_PATH = RAW_DIR / "air_quality" / "nsw_air_quality_sites.json"
SPEED_ZONES_SHP = RAW_DIR / "speed_zones" / "Speed_Zones.shp"
STATION_REFERENCE_PATH = TRAFFIC_DIR / "station_reference.csv"

# --- processed (outputs of stages 01-05, all rebuildable) -------------------
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
FEATURE_MANIFEST_PATH = PROCESSED_DIR / "feature_manifest.json"
SPLIT_MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"

# --- results (model scores, tuning, evaluation, saved models) ---------------
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "results.json"
PER_STATION_DIR = RESULTS_DIR / "per_station"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"     # row-by-row y_true / y_pred
TUNING_DIR = RESULTS_DIR / "tuning"
LEARNING_CURVE_DIR = TUNING_DIR / "learning_curves"
OVERFITTING_DIR = TUNING_DIR / "overfitting"
FEATURE_IMPORTANCE_DIR = TUNING_DIR / "feature_importance"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"
SHAP_DIR = RESULTS_DIR / "shap"