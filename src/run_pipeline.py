"""
CO2/NO2 Traffic-Emissions Capstone -- Pipeline runner

Runs 01_data_ingestion.py .. 06_run_models.py in order, or just one
stage, based on the `pipeline.stage` setting in config/models_config.yaml
(or whatever --config points at). Streams each stage's own print output
live as it runs, times it, and prints a summary at the end.

Stage names (pipeline order):
    ingest | eda | preprocess | features | split | models
    (== 01_data_ingestion, 02_eda, 03_data_preprocessing,
        04_feature_engineering, 05_train_test_split, 06_run_models)

config/models_config.yaml:

    pipeline:
      stage: all       # run every stage, 01 -> 06
      stage: models    # run only 06, reusing existing data/processed/splits/
      stage: features  # run only 04, reusing existing final_*.csv

`stage: all` deletes the intermediate files that 01-05 produce before
starting (everything directly under data/processed/ plus the splits/
folder) so a run never mixes this run's outputs with a stale run's --
data/processed/model_results/ (the cross-run results log 06 appends to)
is left alone.

Any other stage name runs just that one stage and reuses whatever is
already on disk; the runner checks its required input files exist
first and gives you a plain-English pointer to the right earlier stage
if something's missing, rather than letting the stage's own script fail
with a less obvious traceback.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --config config/my_experiment.yaml
    python src/run_pipeline.py --stage features   # override the yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "models_config.yaml"

# (stage name, script filename, human label, required input paths -- relative
# to data/processed/, checked before the stage runs; {grain} is filled in
# from the config for the "models" stage only)
STAGES = [
    ("ingest", "01_data_ingestion.py", "Data ingestion", []),
    ("eda", "02_eda.py", "EDA-driven cleaning", [
        "final_combined_dataset_daily.csv", "final_combined_dataset_hourly.csv",
    ]),
    ("preprocess", "03_data_preprocessing.py", "Preprocessing (impute/encode)", [
        "preprocessed_daily.csv", "preprocessed_hourly.csv",
    ]),
    ("features", "04_feature_engineering.py", "Feature engineering", [
        "final_daily.csv", "final_hourly.csv",
    ]),
    ("split", "05_train_test_split.py", "Train/val/test split + scaling", [
        "features_daily.csv", "features_hourly.csv", "feature_manifest.json",
    ]),
    ("models", "06_run_models.py", "Model training + evaluation", [
        "splits/{grain}_train.csv", "splits/{grain}_val.csv",
    ]),
]
STAGE_NAMES = [s[0] for s in STAGES]

# Cleared on `stage: all` before running anything. Deliberately does NOT
# include model_results/ -- that's the shared, cross-run results log
# 06_run_models.py appends/upserts into, not an intermediate artifact of
# this run.
WIPE_ON_ALL = [
    "final_combined_dataset_daily.csv", "final_combined_dataset_hourly.csv",
    "aq_site_verification.csv",
    "preprocessed_daily.csv", "preprocessed_hourly.csv",
    "excluded_stations_log_daily.csv", "excluded_stations_log_hourly.csv",
    "final_daily.csv", "final_hourly.csv",
    "features_daily.csv", "features_hourly.csv", "feature_manifest.json",
]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def resolve_stage(config: dict, cli_stage: str | None) -> str:
    stage = cli_stage or config.get("pipeline", {}).get("stage", "all")
    stage = str(stage).strip().lower()
    if stage not in ("all", *STAGE_NAMES):
        raise SystemExit(
            f"Unknown pipeline.stage {stage!r}. "
            f"Expected 'all' or one of {STAGE_NAMES}."
        )
    return stage


def wipe_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    removed = []
    for name in WIPE_ON_ALL:
        path = PROCESSED_DIR / name
        if path.exists():
            path.unlink()
            removed.append(name)
    splits_dir = PROCESSED_DIR / "splits"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
        removed.append("splits/")
    if removed:
        print(f"[wipe] removed stale artifacts before full rebuild: {removed}")
    else:
        print("[wipe] data/processed/ already clean, nothing to remove")


def check_inputs(required: list[str], grain: str) -> list[str]:
    missing = []
    for rel in required:
        rel = rel.format(grain=grain)
        if not (PROCESSED_DIR / rel).exists():
            missing.append(rel)
    return missing


def run_stage(name: str, script: str, label: str, extra_args: list[str]) -> float:
    script_path = SRC_DIR / script
    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"\n{'='*70}\n[{name}] {label}  ->  {script} {' '.join(extra_args)}\n{'='*70}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"\n[{name}] FAILED after {elapsed:.1f}s (exit code {result.returncode})")
        raise SystemExit(result.returncode)
    print(f"\n[{name}] done in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--stage", choices=["all", *STAGE_NAMES], default=None,
        help="Override pipeline.stage from the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stage = resolve_stage(config, args.stage)
    grain = config.get("grain", "daily")

    if stage == "all":
        run_list = STAGES
        print(f"Running the FULL pipeline (01 -> 06), grain={grain!r}")
        wipe_processed_dir()
    else:
        run_list = [s for s in STAGES if s[0] == stage]
        print(f"Running SINGLE stage: {stage!r}")

    timings: list[tuple[str, str, float]] = []
    for name, script, label, required_inputs in run_list:
        if stage != "all":  # "all" produces its own inputs stage-to-stage
            missing = check_inputs(required_inputs, grain)
            if missing:
                idx = STAGE_NAMES.index(name)
                prev = STAGE_NAMES[idx - 1] if idx > 0 else None
                hint = f" Run stage {prev!r} first (or stage: all)." if prev else ""
                raise SystemExit(
                    f"[{name}] can't run -- missing data/processed/{missing}.{hint}"
                )

        extra_args = ["--config", str(args.config)] if name == "models" else []
        elapsed = run_stage(name, script, label, extra_args)
        timings.append((name, label, elapsed))

    total = sum(t for _, _, t in timings)
    print(f"\n{'='*70}\nPipeline summary\n{'='*70}")
    for name, label, elapsed in timings:
        print(f"  [OK] {name:12s} {label:35s} {elapsed:6.1f}s")
    print(f"  {'-'*58}")
    print(f"  Total: {total:.1f}s")


if __name__ == "__main__":
    main()
