# Final results

Built from run **2026-10-14_130022_daily-hourly_final_run_report_ready** by `src/11_final_results.py` -- rebuilt automatically after every complete run, so don't edit files here by hand.

- finished: 2026-10-14T21:16:54  |  status: ok
- command: `python.exe run_pipeline.py --jobs 6 --experiments --name final_run_report_ready`
- git commit: `09952c4454f392153159c732fc61859257506d46` on `hana_analysis`  -- **uncommitted changes at run time**
- full backup of every output (~750 files): `working/runs/2026-10-14_130022_daily-hourly_final_run_report_ready/` (emptied at the start of the next full run -- zip it to keep it)

To replicate: check out the commit above, then `python run_pipeline.py --config results/run/config.yaml --experiments`.

Feature sets: `exogenous` = traffic, weather, calendar and station only (answers *what drives NO2*); `all` = the same plus NO2's own recent history (the best *forecast*).

## `run/` -- Replicate the run

| file | what it shows | RQ | source in the run |
|---|---|---|---|
| `config.yaml` | the exact settings the run used | replication | `config.yaml` |
| `tuning.yaml` | hyperparameter search spaces | replication | `tuning.yaml` |
| `run_info.json` | command, git commit, Python + package versions, raw-data fingerprint, stage timings | replication | `run_info.json` |
| `split_manifest.json` | split dates, rows, features kept / pruned | replication | `manifests/split_manifest.json` |
| `pipeline.log` | full console output of the run | replication | `pipeline.log` |

## `1_data/` -- Data and preprocessing (Methodology)

| file | what it shows | RQ | source in the run |
|---|---|---|---|
| `stations.csv` | every candidate station: road, road type, NO2 monitor + distance, weather source, rows per split, train-only / tested / excluded (report table T1) | methodology | `scripts/stations.py + manifests/split_manifest.json + excluded_stations_log` |
| `cleaning_steps.csv` | rows and stations removed at each cleaning step (report table T2) | methodology | `manifests/data_quality_daily.csv; manifests/data_quality_hourly.csv` |
| `missing_data.csv` | % missing per variable before / after cleaning, and how each gap is filled | methodology | `manifests/missing_data_daily.csv; manifests/missing_data_hourly.csv` |
| `features_by_family.csv` | engineered features per family, kept vs pruned as near-duplicates, with names | methodology, RQ2 | `manifests/split_manifest.json` |
| `split_summary.csv` | chronological split: dates, rows, stations, NO2 level + spread per split (lower test spread = lower R2 for the same error) | methodology | `manifests/split_manifest.json` |
| `excluded_stations.csv` | stations dropped and why | methodology | `manifests/excluded_stations_log_daily.csv` |
| `aq_site_verification.csv` | each station's NO2 monitor vs the truly nearest monitor (distance check) | methodology | `manifests/aq_site_verification.csv` |

## `2_models/` -- Model comparison (Results, RQ3)

| file | what it shows | RQ | source in the run |
|---|---|---|---|
| `scores_all.csv` | every model x feature set x split: RMSE, MAE, R2 (default and tuned) | RQ3 | `outputs/latest/evaluation/all_results.csv` |
| `summary.csv` | val / test R2 and per-station median test R2 for every method (report table T3) | RQ3 | `outputs/latest/evaluation/summary_daily.csv; outputs/latest/evaluation/summary_hourly.csv` |
| `bootstrap_test.csv` | test R2 / RMSE / MAE with 95% confidence intervals (block bootstrap) | RQ3 | `outputs/latest/evaluation/bootstrap_daily_test.csv; outputs/latest/evaluation/bootstrap_hourly_test.csv` |
| `bootstrap_vs_best.csv` | each model vs the best: tied, or worse (95% CI of the R2 difference) | RQ3 | `outputs/latest/evaluation/bootstrap_daily_test_vs_best.csv; outputs/latest/evaluation/bootstrap_hourly_test_vs_best.csv` |
| `tuning_gains.csv` | default -> tuned change in RMSE and R2 (report table T4) | RQ3 | `outputs/latest/evaluation/stage_delta_daily.csv; outputs/latest/evaluation/stage_delta_hourly.csv` |
| `model_times.csv` | seconds to train each model once, and minutes for its whole tuning search (models shared the CPU -- compare orders of magnitude) | RQ3 | `scores.json + outputs/latest/tuning/*_tuned_metrics.csv` |
| `per_station_test.csv` | test RMSE / R2 per station for the top models (GRU, LSTM, XGBoost; R2 blank under 30 rows) | RQ3 | `outputs/latest/per_station/gru_tuned_daily_exogenous_test.csv; outputs/latest/per_station/gru_tuned_daily_all_test.csv; outputs/latest/per_station/gru_tuned_hourly_exogenous_test.csv; outputs/latest/per_station/gru_tuned_hourly_all_test.csv; outputs/latest/per_station/lstm_tuned_daily_exogenous_test.csv; outputs/latest/per_station/lstm_tuned_daily_all_test.csv; outputs/latest/per_station/lstm_tuned_hourly_exogenous_test.csv; outputs/latest/per_station/lstm_tuned_hourly_all_test.csv; outputs/latest/per_station/xgboost_tuned_daily_exogenous_test.csv; outputs/latest/per_station/xgboost_tuned_daily_all_test.csv; outputs/latest/per_station/xgboost_tuned_hourly_exogenous_test.csv; outputs/latest/per_station/xgboost_tuned_hourly_all_test.csv` |
| `learning_curve_daily.csv` | daily XGBoost: accuracy as training data grows (still rising = limited by data) | RQ2, RQ3 | `outputs/latest/tuning/learning_curves/xgboost_daily_exogenous.csv; outputs/latest/tuning/learning_curves/xgboost_daily_all.csv` |
| `summary_daily.png` | val / test R2 by method, daily | RQ3 | `outputs/latest/evaluation/summary_daily.png` |
| `summary_hourly.png` | val / test R2 by method, hourly | RQ3 | `outputs/latest/evaluation/summary_hourly.png` |
| `bootstrap_hourly_test.png` | hourly test R2 with 95% intervals (report figure F2) | RQ3 | `outputs/latest/evaluation/bootstrap_hourly_test.png` |
| `learning_curve_daily.png` | daily XGBoost learning curve (appendix) | RQ2, RQ3 | `outputs/latest/tuning/learning_curves/xgboost_daily_all.png` |

## `3_drivers/` -- What drives NO2 (Results, RQ1 + RQ2)

| file | what it shows | RQ | source in the run |
|---|---|---|---|
| `shap_groups.csv` | SHAP importance share per feature family -- weather, traffic, calendar, ... (report table T5) | RQ1 | `outputs/latest/shap/groups_daily_exogenous.csv; outputs/latest/shap/groups_daily_all.csv; outputs/latest/shap/groups_hourly_exogenous.csv; outputs/latest/shap/groups_hourly_all.csv` |
| `shap_importance.csv` | every feature ranked by SHAP, with direction (traffic + weather model) | RQ1 | `outputs/latest/shap/importance_daily_exogenous.csv; outputs/latest/shap/importance_hourly_exogenous.csv` |
| `traffic_parts.csv` | which parts of traffic matter: total, heavy vehicles, direction, traffic/wind, ... | RQ1 | `outputs/latest/shap/traffic_parts_daily_exogenous.csv; outputs/latest/shap/traffic_parts_hourly_exogenous.csv` |
| `traffic_wind.csv` | traffic's effect on NO2 at low / medium / high wind, by traffic decile | RQ1 | `outputs/latest/shap/interaction_traffic_wind_daily_exogenous.csv; outputs/latest/shap/interaction_traffic_wind_hourly_exogenous.csv` |
| `ablation.csv` | accuracy when each feature family is removed; also sample weights / pruning on-off | RQ1, RQ2 | `outputs/experiments/ablation/ablation_daily.csv; outputs/experiments/ablation/ablation_hourly.csv` |
| `holiday_effect.csv` | weekday-holiday effect on traffic and NO2, weather-adjusted, 95% CI | RQ1 | `outputs/experiments/holiday/holiday_daily_effect.csv; outputs/experiments/holiday/holiday_hourly_effect.csv` |
| `holiday_by_station.csv` | holiday vs normal weekday averages per station | RQ1 | `outputs/experiments/holiday/holiday_daily_by_station.csv; outputs/experiments/holiday/holiday_hourly_by_station.csv` |
| `holiday_dose_response.csv` | correlation of traffic drop vs NO2 drop across stations | RQ1 | `outputs/experiments/holiday/holiday_daily_dose_response.csv; outputs/experiments/holiday/holiday_hourly_dose_response.csv` |
| `shap_beeswarm_hourly.png` | SHAP summary, hourly traffic + weather model (report figure F3) | RQ1, RQ4 | `outputs/latest/shap/summary_hourly_exogenous.png` |
| `traffic_wind_hourly.png` | traffic effect by wind speed -- calm air amplifies traffic NO2 (report figure F4) | RQ1 | `outputs/latest/shap/interaction_traffic_wind_hourly_exogenous.png` |
| `wind_effect_hourly.png` | SHAP effect of wind speed (real units, m/s) | RQ1 | `outputs/latest/shap/dependence_hourly_exogenous_wind_speed_ms_lag_1h.png` |
| `holiday_profile_hourly.png` | traffic and NO2 by hour, holidays vs normal weekdays (report figure F5) | RQ1 | `outputs/experiments/holiday/holiday_hourly_profile.png` |
| `holiday_dose_response_hourly.png` | per-station traffic drop vs NO2 drop (appendix) | RQ1 | `outputs/experiments/holiday/holiday_hourly_dose_response.png` |
| `road_type_hourly.png` | SHAP family shares by road type, train rows (appendix) | RQ1, RQ3 | `outputs/latest/shap/groups_by_road_type_hourly_exogenous.png` |
| `ablation_hourly.png` | feature-family ablation, hourly (appendix) | RQ1, RQ2 | `outputs/experiments/ablation/ablation_hourly.png` |

## `4_explainability/` -- Do the explanations agree? (Results, RQ4)

| file | what it shows | RQ | source in the run |
|---|---|---|---|
| `model_agreement.csv` | rank agreement (Spearman) of feature families between models and methods | RQ4 | `outputs/latest/permutation/top3_agreement_daily_exogenous.csv; outputs/latest/permutation/top3_agreement_daily_all.csv; outputs/latest/permutation/top3_agreement_hourly_exogenous.csv; outputs/latest/permutation/top3_agreement_hourly_all.csv` |
| `permutation_families.csv` | permutation importance (R2 drop) per family for the top-3 models | RQ4 | `outputs/latest/permutation/top3_families_daily_exogenous.csv; outputs/latest/permutation/top3_families_daily_all.csv; outputs/latest/permutation/top3_families_hourly_exogenous.csv; outputs/latest/permutation/top3_families_hourly_all.csv` |
| `waterfall_cases.csv` | single predictions explained: highest NO2, typical, calm vs windy rush hour | RQ4 | `outputs/latest/shap/waterfall_daily_exogenous.csv; outputs/latest/shap/waterfall_hourly_exogenous.csv` |
| `permutation_top3_hourly.png` | top-3 models' reliance per feature family, hourly (report figure F6) | RQ4 | `outputs/latest/permutation/top3_hourly_exogenous.png` |
| `history_window_hourly.png` | how far back the sequence model looks (current + last 1-2 hours) | RQ4 | `outputs/latest/permutation/top3_window_hourly_exogenous_gru.png` |
| `waterfall_calm_rush_hour.png` | one calm rush hour explained feature by feature (report figure F7) | RQ4 | `outputs/latest/shap/waterfall_hourly_exogenous_calm_rush_hour.png` |
| `waterfall_highest_no2.png` | the highest-NO2 hour explained -- shows the failure on extremes | RQ3, RQ4 | `outputs/latest/shap/waterfall_hourly_exogenous_highest_NO2.png` |

## `5_errors/` -- When and where the models fail (Results / Discussion, RQ3)

| file | what it shows | RQ | source in the run |
|---|---|---|---|
| `errors_by_hour.csv` | test RMSE / MAE / bias by hour of day for every model | RQ3 | `outputs/experiments/error_analysis/errors_hourly_test_all_hour.csv` |
| `errors_by_condition.csv` | test errors by NO2 level, traffic level and road type, every model | RQ3 | `outputs/experiments/error_analysis/errors_daily_test_all_no2_level.csv; outputs/experiments/error_analysis/errors_daily_test_all_traffic_level.csv; outputs/experiments/error_analysis/errors_daily_test_all_road_type.csv; outputs/experiments/error_analysis/errors_hourly_test_all_no2_level.csv; outputs/experiments/error_analysis/errors_hourly_test_all_traffic_level.csv; outputs/experiments/error_analysis/errors_hourly_test_all_road_type.csv` |
| `unseen_stations.csv` | leave-one-station-out: each station predicted by a model that never saw it | RQ3 | `outputs/experiments/loso/loso_daily_exogenous.csv; outputs/experiments/loso/loso_hourly_exogenous.csv` |
| `unseen_stations_summary.csv` | leave-one-station-out summary: median / worst station R2, stations beating the average | RQ3 | `outputs/experiments/loso/loso_daily_exogenous_summary.csv; outputs/experiments/loso/loso_hourly_exogenous_summary.csv` |
| `errors_by_hour_hourly.png` | error by hour of day (report figure F8, optional) | RQ3 | `outputs/experiments/error_analysis/errors_hourly_test_all_hour.png` |
| `errors_by_no2_level_hourly.png` | error by NO2 level -- extremes under-predicted (appendix) | RQ3 | `outputs/experiments/error_analysis/errors_hourly_test_all_no2_level.png` |
| `unseen_stations_hourly.png` | unseen-station R2 per station (appendix) | RQ3 | `outputs/experiments/loso/loso_hourly_exogenous.png` |
