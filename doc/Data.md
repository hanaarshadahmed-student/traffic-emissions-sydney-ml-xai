# Data

## Overview

This project predicts **NO₂ concentration** (`no2_pphm`) near NSW traffic
stations from traffic volume, weather, and station metadata.

**History:** the project originally targeted a *computed* CO₂-estimate
(traffic volume × fuel-consumption rate × emission factor) across 3
Sydney-only stations. The supervisor ruled out computed/estimated targets
— see the "no-estimation constraint" below — so the project pivoted to
`no2_pphm`, which is **directly measured** at NSW Air Quality Network
sites, and expanded from 3 Sydney stations to 15 candidate stations
spanning NSW, matched to their nearest AQ monitoring site.

## Sources

| Data | Source | Access |
|---|---|---|
| Traffic volume (daily/hourly, by vehicle class) | Transport for NSW, Traffic Volume Viewer | maps.transport.nsw.gov.au/egeomaps/traffic-volumes |
| Weather (metro stations, hourly) | NSW Air Quality Network portal | airquality.nsw.gov.au |
| Weather (rural stations, daily) | Bureau of Meteorology, Climate Data Online | reg.bom.gov.au/climate/data |
| NO2/CO/ozone (metro stations, hourly) | NSW Air Quality Network portal | Same as metro weather |
| AQ site registry (137 sites, lat/lon) | NSW Air Quality Network | `data/raw/air_quality/nsw_air_quality_sites.json` — optional, enables AQ-site verification |
| Speed zones (posted speed limits, ~447k segments) | Transport for NSW | `data/raw/speed_zones/Speed_Zones.shp` — optional, enables posted-speed matching. Not in git (too large for GitHub) |

## Traffic stations used

All 15 candidate stations, with their matched AQ site and the haversine
distance to it:

| Station ID | Road | Suburb | LGA | Matched AQ site | Distance (km) |
|---|---|---|---|---|---|
| MUB001 | Melbourne Street | Mulwala | Corowa | ALBURY | 83.2 |
| 6135-PR | M31 Hume Highway | Bowning | Yass Valley | GOULBURN | 78.1 |
| 6149 | Newell Highway | Tomingley | Narromine | ORANGE (no NO2) | 109.2 |
| 6141 | Newell Highway | Forbes | Forbes | ORANGE (no NO2) | 106.1 |
| 6105 | Great Western Highway | Meadow Flat | Lithgow | BATHURST (no NO2 sensor) | 32.7 |
| 6124 | Macleay Valley Way | South Kempsey | Kempsey | PORT MACQUARIE | 36.9 |
| 6116 | Pacific Highway | Wardell | Ballina | LISMORE (no continuous NO2) | 23.8 |
| 7212 | Stewart Avenue | Newcastle West | Newcastle | NEWCASTLE | 0.2 |
| 7211 | Lily Lane | Adamstown | Newcastle | NEWCASTLE | 4.5 |
| 6119-PR | Pacific Highway | Nabiac | Greater Taree | PORT MACQUARIE | 87.3 |
| 7216 | Gladstone Avenue | Wollongong | Wollongong | WOLLONGONG | 0.6 |
| 6109 | M23 Federal Highway | Yarra | Goulburn Mulwaree | GOULBURN | 13.0 |
| 10011 | New South Head Road | Edgecliff | Woollahra | COOK AND PHILLIP | 1.8 |
| 100001 | Cambridge Street | Canley Heights | Fairfield | LIVERPOOL | 4.8 |
| 6178-PR | Picton Road | Cordeaux | Wollongong | WOLLONGONG | 8.2 |

**Dropped for zero NO2 coverage** (Step 1, `02_eda.py`): `6105`, `6116`,
`6141`, `6149` — all four are matched to AQ sites explicitly noted as
lacking a working NO2 sensor. **11 stations remain for the daily model.**
Of those, **8 have both hourly weather and hourly NO2** from a metro AQ
site and qualify for the hourly model (rural/BOM-weather stations only
report daily min/max temp and rainfall, so there's no hourly weather to
build from for the other 3).

## AQ-site match verification

Each station's `aq_site` above was originally chosen by hand. It's cross-
checked in `01_data_ingestion.py` (`verify_aq_site_matches()`) against
the true nearest geocoded AQ site (haversine distance, 137-site
registry). **5 of 15 stations don't match their true nearest site:**

| Station | Hardcoded site | True nearest site | Distance |
|---|---|---|---|
| MUB001 | ALBURY (83.2km) | Rand | 59.4km |
| 6149 | ORANGE (109.2km) | Parkes | 52.4km |
| 6141 | ORANGE (106.1km) | Parkes | 40.5km |
| 6119-PR | PORT MACQUARIE (87.3km) | Taree | 24.1km |
| 100001 | LIVERPOOL (4.8km) | LIVERPOOL SWAQS | 4.6km |

The registry doesn't record which sites measure NO2, so a farther match
can be the deliberate, correct choice (this is very likely true for
`6149`/`6141`, both already excluded above). `100001`'s "mismatch" is a
0.2km difference against what's almost certainly the same physical site
under a different registry label, not a real error. `MUB001` and
`6119-PR` remain flagged (`aq_match_flag = REVIEW`) after the Step 1
drop and are carried through the rest of the pipeline as lower-confidence
(see "Data-quality decisions" below), not excluded.

## Target variable: `no2_pphm`

Measured directly at the matched AQ site (parts per hundred million),
sourced from the NSW Air Quality Network portal. Not computed, not
estimated — this is the reason for the pivot away from the earlier CO2
approach (the supervisor's "no-estimation constraint").

## Posted speed limit

`posted_speed_kmh` / `speed_zone_type` are added by matching each
station's coordinates to the nearest line segment in the TfNSW speed
zones shapefile (`match_speed_zones()` in `01_data_ingestion.py`, one
pass over ~447k records, Web Mercator point-to-segment distance).
**Caveat:** `speed_zone_type` describes the *kind of speed zone* (e.g.
`Ordinary Permanent`, `School`, `Variable`), not a road hierarchy — it
is not a substitute for `road_type` (built separately in
`04_feature_engineering.py` from keyword matching on the station name).
`posted_speed_kmh` correlates with NO2 at r=-0.36 station-level, notably
stronger than the *pooled* `road_type` relationship (r=-0.019) — it's
kept as a real numeric feature precisely because it splits stations
`road_type`'s 3-bucket keyword heuristic lumps together.

## Data-quality decisions (`03_data_preprocessing.py`)

**NO2 outliers (IQR method, per station):** `6119-PR` (7.9%), `6124`
(7.5%), and `6109` (5.0%) show the highest outlier rates — the first two
are also the largest-`aq_distance_km` stations remaining after Step 1.
Not treated as sensor error: an IQR flag on air-quality data can just as
easily be a genuine pollution episode as a fault, and here the pattern
tracks a known data-quality variable (AQ-match distance) rather than
anything indicating instrument malfunction. Reported, not removed —
see references in `03_data_preprocessing.py`'s docstring.

**`aq_quality_weight`:** rather than dropping the 2 remaining
`REVIEW`-flagged stations (`MUB001`, `6119-PR` in the daily set; only
`100001` remains flagged in the smaller hourly set), they're kept with
`aq_quality_weight = 0.5` (vs. `1.0` for `OK` stations) — an explicit,
inspectable column rather than a silent exclusion. This is supported by
three independent findings pointing the same way (weaker traffic-NO2
correlation at greater AQ distance, r=-0.759; weaker mean correlation
for `REVIEW` stations, though n=3 is small; elevated NO2 outlier rate at
the same two stations) — but the sample size means this is a judgement
call, not a statistically settled one. Intended for use as a
`sample_weight` argument in modelling, not yet wired into `05_models.py`.

**Missing values:** `wind_speed_ms`/`wind_dir_deg` are structurally
absent for BOM (rural) stations — a `has_wind_data` flag is added before
filling, so a model can distinguish "no wind sensor" from "average wind."
`temp_c`/`rain_mm` are ~0.3% missing (sensor gaps, not structural) and
filled with the median. `temp_max_c`/`temp_min_c` and `co_ppm`/
`ozone_pphm` are dropped as redundant/unused rather than imputed.

**Imputation leakage (fixed):** the medians above are computed on TRAIN
rows only, using the same split definition as `05_train_test_split.py`
(`split_utils.py`).

## Train/val/test split and leakage fixes

**Global cutoff date.** One cutoff date applies to every station (daily:
train < 2025-03-29 <= val < 2025-08-15 <= test). The earlier per-station
70/15/15 split leaked targets: stations sharing an AQ site (NEWCASTLE
7212/7211, WOLLONGONG 7216/6178-PR, GOULBURN 6109/6135-PR, PORT MACQUARIE
6124/6119-PR) have an identical NO2 series, and short stations' val/test
dates fell inside their sibling's training period (78-100% of those rows).
Consequence: the 2024-only stations (7211, 6135-PR, 6178-PR), 10011 and
6109 are entirely in train; val/test evaluate the 6 long-running stations.

**Time-trend features removed.** `days_since_start` and `calendar_year` are
still in the feature CSVs but excluded from the model feature lists. Val/test
dates always lie beyond the training range, trees can't extrapolate, and
random forest had ranked `days_since_start` its #1 feature (exogenous test
R2 -11.3 with it, -0.27 without).

**Near-duplicate features pruned.** `05_train_test_split.py` drops features
correlated |r| > 0.95 (on train rows) with a simpler kept feature, within
the exogenous and autoregressive sets separately; every dropped feature
and its reason is in `splits/split_manifest.json`. Use `--no-prune` to skip.

**Shared AQ sites (limitation).** For the four station pairs above, two
different roads are predicting the same monitor's NO2 -- the model is
learning the AQ site's NO2, not each road's. State this in the report.

**100001 weighting (fixed).** `100001` was flagged `REVIEW` only because the
registry labels its site "LIVERPOOL SWAQS" rather than "LIVERPOOL" (0.2km
apart, same site), which down-weighted it to 0.5. The match check now
accepts label variants, so only `MUB001` and `6119-PR` are weighted 0.5.

## Final processed dataset

`data/processed/features_daily.csv` (4,861 rows, 50 columns) and
`features_hourly.csv` (66,777 rows, 49 columns) — output of
`04_feature_engineering.py`, zero NaNs, ready for modelling. Includes:
traffic/weather/NO2 raw columns, calendar features (day of week, season,
weekend flag), one-hot `road_` and `season_`/`stn_` encodings, `heavy_pct`,
`is_rainy`, `log_no2_pphm`, lag/rolling traffic features (daily) or
`hour_sin`/`hour_cos` (hourly), `posted_speed_kmh`, `aq_distance_km`, and
`aq_quality_weight`/`aq_match_flag` (metadata, excluded from the
zero-NaN feature check but available for weighting).

Rebuild with the 4-script pipeline in the main `README.md`.