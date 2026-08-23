# Data

## Overview

This project predicts CO₂ emissions from vehicle traffic on 3 Sydney roads.
There is no dataset that directly measures road-level CO₂ — the target
variable is **computed** from traffic volume, published fuel consumption
rates, and official emission factors (see "Target variable" below).

## Sources

| Data | Source | Access |
|---|---|---|
| Traffic volume (hourly, by vehicle class) | Transport for NSW, Traffic Volume Viewer | maps.transport.nsw.gov.au/egeomaps/traffic-volumes |
| Station metadata (LGA, road type, classifier flag) | Transport for NSW, Traffic Volume Viewer | Same as above — "Station Information" download |
| Weather (max temperature, rainfall, daily) | Bureau of Meteorology, Climate Data Online | reg.bom.gov.au/climate/data |
| Fuel consumption rates (L/100km by vehicle class) | ABS Survey of Motor Vehicle Use | abs.gov.au |
| Emission factors (kg CO₂-e per litre of fuel) | Australian National Greenhouse Accounts Factors 2025, Table 9 | dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors-2025 |

## Traffic stations used

Selected from an initial screen of 20 candidate permanent classifier
stations across Sydney LGAs, narrowed to 3 based on actual data
completeness (not just being on the map):

| Station ID | Road | LGA | Missing days (of 731) | Directions |
|---|---|---|---|---|
| 50240 | Briens Road | Parramatta | 3.6% | 2 (Eastbound/Westbound) |
| 50260 | Silverwater Road | Parramatta | 5.1% | 2 (Northbound/Southbound) |
| 7272 | Edgar Street | Bankstown | 10.7% | 1 (Southbound only — documented limitation) |

**Stations considered and dropped:**
- 10011 (New South Head Rd, Woollahra) — data ends March 2025, missing entire second year
- 51235 (Victoria Rd, Ryde) — 28.2% missing days, worst coverage of all candidates
- 29005, 7139, 7270 — usable as backups but not selected; higher missing % or single-direction only

## Weather stations used

BOM stations are far sparser than traffic stations, so 2 stations cover all 3 traffic sites:

| Station | Name | Covers | Notes |
|---|---|---|---|
| 066124 | Parramatta North (Masons Drive) | Briens Rd, Silverwater Rd | Manual station — max temp + rainfall only, no humidity/wind sensors |
| 066137 | Bankstown Airport AWS | Edgar St | Automatic Weather Station — same variables used for consistency |

**Dropped weather variables:** humidity and wind speed were considered but
not available at these two stations without switching to different (further
away) stations. Temperature and rainfall are the dominant weather features
in the reviewed literature, so this was accepted as a reasonable scope cut.

## Target variable: `co2_estimate`

Computed per hour, per station:

```
co2_estimate (kg) = (volume_light × 0.111 L/km × 2.31 kg CO2-e/L)
                   + (volume_heavy × 0.28  L/km × 2.72 kg CO2-e/L)
```

- **0.111 L/km** (11.1 L/100km) — Passenger vehicles, ABS Survey of Motor Vehicle Use, 12 months ended 30 June 2020
- **0.286 L/km** (28.6 L/100km) — Rigid trucks, ABS Survey of Motor Vehicle Use, 12 months ended 30 June 2020. Chosen over Articulated trucks (53.1 L/100km) since TfNSW's "Heavy Vehicles" classification on arterial roads is dominated by rigid trucks and buses rather than long-haul articulated trucks.

**Note:** the ABS Survey of Motor Vehicle Use was discontinued after this release — it is the most recent official Australian source available, but reflects 2020 vehicle fleet efficiency, not 2024–2025. Actual fuel consumption has likely improved slightly since, meaning `co2_estimate` is a small conservative (over-)estimate. Worth one sentence on this in the methodology limitations.
- **2.31 kg CO2-e/L** — petrol, Scope 1 (tailpipe), NGA Factors 2025 Table 9, cars/light commercial vehicles
- **2.72 kg CO2-e/L** — diesel, Scope 1 (tailpipe), NGA Factors 2025 Table 9, heavy duty vehicles (Euro IV+)

Scope 1 (direct combustion) was used rather than Scope 1+3, since this
project models on-road tailpipe emissions, not full fuel lifecycle.

**Important limitation to note in the methodology:** because the target is
computed from traffic volume rather than independently measured, feature
importance results (SHAP) involving `volume_light`/`volume_heavy` will be
partly circular. Consider excluding raw volume from the feature set when
interpreting XAI results for weather/temporal drivers specifically.

## Final processed dataset

`data/processed/merged_dataset.csv` — one row per station per hour.

| Column | Description |
|---|---|
| `timestamp` | Hourly datetime |
| `station_id`, `road`, `lga` | Station identity |
| `volume_light`, `volume_heavy` | Vehicle counts (both directions summed) |
| `max_temp`, `rainfall` | Daily weather, applied across all hours of that day |
| `hour_of_day`, `day_of_week`, `is_weekend`, `month` | Derived from timestamp |
| `public_holiday`, `school_holiday` | From TfNSW source data |
| `co2_estimate` | Computed target variable (kg CO2-e) |

**Known gaps:**
- 744 rows missing rainfall (Parramatta 2025 data was incomplete at collection time)
- 48 rows missing max temperature
- Edgar Street (Bankstown) is single-direction only — represents half the road's actual traffic

Rebuild with: `python src/data_preprocessing.py` (see main `README.md`).