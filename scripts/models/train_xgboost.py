"""Compatibility entry point for XGBoost fine-tuning."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.models.fine_tuning.train_xgboost import main


if __name__ == "__main__":
    main()
