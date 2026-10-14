"""
Data ingestion: combines the raw traffic, weather and NO2 files into a daily dataset (all stations) and an hourly dataset (only stations with hourly weather and NO2 from a metro AQ site). Also rebuilds the public holiday column and adds speed zone and AQ site checks if those files are present.

Requires: pip install pandas python-calamine holidays pyshp
"""

import pandas as pd
import glob, os, json, math
import holidays as holidays_lib

import sys
from pathlib import Path

# Repo root on the import path, so the shared code in scripts/ is importable
# whichever folder you run this from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402

TRAFFIC_DIR = str(paths.TRAFFIC_DIR)
BOM_DIR = str(paths.WEATHER_DIR)
METRO_WEATHER_DIR = str(paths.WEATHER_DIR)
METRO_EMISSIONS_DIR = str(paths.EMISSIONS_DIR)
AQ_SITES_PATH = str(paths.AQ_SITES_PATH)
SPEED_ZONES_SHP = str(paths.SPEED_ZONES_SHP)
OUT_DIR = str(paths.PROCESSED_DIR)
DAILY_OUT_PATH = os.path.join(OUT_DIR, "final_combined_dataset_daily.csv")
HOURLY_OUT_PATH = os.path.join(OUT_DIR, "final_combined_dataset_hourly.csv")
AQ_VERIFICATION_OUT_PATH = os.path.join(OUT_DIR, "aq_site_verification.csv")

NSW_HOLIDAYS = holidays_lib.Australia(subdiv="NSW", years=range(2023, 2027))

# station registry (ids, locations, matched weather/emissions files, AQ monitor):
# defined once in scripts/stations.py -- add or edit stations there
from scripts.stations import STATIONS  # noqa: E402


def find_metro_file(directory, keyword, ext="xls"):
    matches = glob.glob(f"{directory}/*{keyword}*.{ext}")
    if not matches:
        raise FileNotFoundError(
            f"No file matching '*{keyword}*.{ext}' found in '{directory}'.\n"
            f"  Absolute path searched: {os.path.abspath(directory)}\n"
            f"  Does that folder exist? {os.path.isdir(directory)}\n"
            f"  Files actually there: {os.listdir(directory) if os.path.isdir(directory) else 'N/A (folder missing)'}"
        )
    return matches[0]


def read_metro_xls(path):
    return pd.read_excel(path, engine="calamine", skiprows=2)


def real_public_holiday(dates):
    return pd.Series(dates).dt.date.isin(NSW_HOLIDAYS).astype(int).values


# ---------------------------------------------------------------------------
# STATION METADATA ENRICHMENT 
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _same_aq_site(hardcoded: str, nearest: str) -> bool:
    hard = hardcoded.upper().split(" (")[0].strip()
    near = nearest.upper().strip()
    return near in hard or hard in near


def verify_aq_site_matches(stations, sites_path=AQ_SITES_PATH):
    if not os.path.exists(sites_path):
        print(f"  [AQ verification] SKIPPED -- '{sites_path}' not found.")
        return None

    with open(sites_path) as f:
        sites = json.load(f)
    sites = [s for s in sites if s.get("Latitude") is not None and s.get("Longitude") is not None]

    rows = []
    for name, cfg in stations.items():
        nearest_name, nearest_km = min(
            ((s["SiteName"], _haversine_km(cfg["lat"], cfg["lon"], s["Latitude"], s["Longitude"]))
             for s in sites),
            key=lambda x: x[1],
        )
        rows.append({
            "station": name,
            "hardcoded_site": cfg["aq_site"],
            "hardcoded_km": cfg["aq_distance_km"],
            "computed_nearest_site": nearest_name,
            "computed_nearest_km": round(nearest_km, 1),
            "flag": "OK" if _same_aq_site(cfg["aq_site"], nearest_name) else "REVIEW",
        })
    df = pd.DataFrame(rows)

    n_review = (df["flag"] == "REVIEW").sum()
    print(f"  [AQ verification] {len(df) - n_review}/{len(df)} stations match their true nearest AQ site; "
          f"{n_review} flagged for manual review (see {AQ_VERIFICATION_OUT_PATH}).")
    if n_review:
        print(df.loc[df["flag"] == "REVIEW", ["station", "hardcoded_site", "computed_nearest_site", "computed_nearest_km"]]
              .to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(AQ_VERIFICATION_OUT_PATH, index=False)
    return df


# ---------------------------------------------------------------------------
# STATION METADATA ENRICHMENT
# ---------------------------------------------------------------------------

_EARTH_R = 6378137.0 


def _lonlat_to_webmercator(lon, lat):
    x = _EARTH_R * math.radians(lon)
    y = _EARTH_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def match_speed_zones(stations, shp_path=SPEED_ZONES_SHP, search_radius_m=500):
    if not os.path.exists(shp_path):
        print(f"  [Speed zones] SKIPPED -- '{shp_path}' not found.")
        return {}

    import shapefile  # pyshp -- only needed for this optional enrichment step

    stations_xy = {name: _lonlat_to_webmercator(cfg["lon"], cfg["lat"]) for name, cfg in stations.items()}
    best = {name: None for name in stations}

    sf = shapefile.Reader(shp_path)
    for shape, rec in zip(sf.iterShapes(), sf.iterRecords()):
        if not shape.points:
            continue
        bx0, by0, bx1, by1 = shape.bbox
        for name, (sx, sy) in stations_xy.items():
            if bx1 < sx - search_radius_m or bx0 > sx + search_radius_m \
               or by1 < sy - search_radius_m or by0 > sy + search_radius_m:
                continue
            pts = shape.points
            for i in range(len(pts) - 1):
                d = _point_seg_dist(sx, sy, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                if best[name] is None or d < best[name][0]:
                    best[name] = (d, rec["Speed"], rec["Type"])

    unmatched = [name for name, r in best.items() if r is None]
    if unmatched:
        print(f"  [Speed zones] WARNING -- no segment found within {search_radius_m}m for: {unmatched} "
              f"(try increasing search_radius_m)")
    matched = len(stations) - len(unmatched)
    print(f"  [Speed zones] matched {matched}/{len(stations)} stations to a posted speed zone.")

    return {
        name: {
            "posted_speed_kmh": int(r[1].split()[0]) if r else None,
            "speed_zone_type": r[2] if r else None,
            "speed_zone_match_dist_m": round(r[0], 1) if r else None,
        }
        for name, r in best.items()
    }


def _attach_station_metadata(merged, station_id, station_file, cfg, aq_verification, speed_zone_info):
    merged["station_id"] = station_id
    merged["station_name"] = station_file
    merged["station_lat"] = cfg["lat"]
    merged["station_lon"] = cfg["lon"]
    merged["station_suburb"] = cfg["suburb"]
    merged["station_lga"] = cfg["lga"]
    merged["matched_aq_site"] = cfg["aq_site"]
    merged["aq_distance_km"] = cfg["aq_distance_km"]

    if aq_verification is not None:
        review_row = aq_verification.loc[aq_verification["station"] == station_file]
        merged["aq_match_flag"] = review_row["flag"].iloc[0] if len(review_row) else "UNKNOWN"

    sz = speed_zone_info.get(station_file) if speed_zone_info else None
    if sz:
        merged["posted_speed_kmh"] = sz["posted_speed_kmh"]
        merged["speed_zone_type"] = sz["speed_zone_type"]
        merged["speed_zone_match_dist_m"] = sz["speed_zone_match_dist_m"]

    return merged


# ---------------------------------------------------------------------------
# DAILY BUILD
# ---------------------------------------------------------------------------

def load_traffic_daily(path):
    df = pd.read_csv(path)
    hour_cols = [c for c in df.columns if c.startswith("hour_")]
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce")
    df["row_total"] = df[hour_cols].sum(axis=1, min_count=1)

    pivot = (
        df.pivot_table(index="date", columns="classification_seq", values="row_total", aggfunc="sum")
        .rename(columns={
            "All Vehicles": "traffic_volume_total",
            "Light Vehicles": "traffic_volume_light",
            "Heavy Vehicles": "traffic_volume_heavy",
        })
        .reset_index()
    )
    school = df.groupby("date", as_index=False)["school_holiday"].max()
    daily = pivot.merge(school, on="date", how="left")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["public_holiday"] = real_public_holiday(daily["date"])
    return daily


def load_metro_weather_daily(path):
    df = read_metro_xls(path)
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    val_cols = [c for c in df.columns if c not in ("Date", "Time", "date")]
    df[val_cols] = df[val_cols].apply(pd.to_numeric, errors="coerce")
    agg = {c: ("sum" if "RAIN" in c.upper() else "mean") for c in val_cols}
    daily = df.groupby("date", as_index=False).agg(agg)
    rename = {}
    for c in val_cols:
        cu = c.upper()
        if "TEMP" in cu: rename[c] = "temp_c"
        elif "HUMID" in cu: rename[c] = "humidity_pct"
        elif "WSP" in cu: rename[c] = "wind_speed_ms"
        elif "WDR" in cu: rename[c] = "wind_dir_deg"
        elif "RAIN" in cu: rename[c] = "rain_mm"
    daily = daily.rename(columns=rename)
    daily["weather_source_type"] = "metro_hourly"
    return daily


def add_borrowed_rain(weather, cfg, loader, keys):
    source = cfg.get("rain_from")
    if not source or ("rain_mm" in weather.columns and weather["rain_mm"].notna().any()):
        return weather
    rain = loader(find_metro_file(METRO_WEATHER_DIR, source))[keys + ["rain_mm"]]
    weather = weather.drop(columns=["rain_mm"], errors="ignore").merge(rain, on=keys, how="left")
    weather["rain_source"] = source
    return weather


def load_bom_weather_daily(site):
    files = glob.glob(f"{BOM_DIR}/IDCJAC*_{site}_Data.csv")
    dfs = []
    for f in files:
        code = os.path.basename(f).split("_")[0]
        d = pd.read_csv(f)
        d["date"] = pd.to_datetime(dict(year=d["Year"], month=d["Month"], day=d["Day"]), errors="coerce")
        if code == "IDCJAC0009":
            d = d.rename(columns={"Rainfall amount (millimetres)": "rain_mm"})[["date", "rain_mm"]]
        elif code == "IDCJAC0010":
            d = d.rename(columns={"Maximum temperature (Degree C)": "temp_max_c"})[["date", "temp_max_c"]]
        elif code == "IDCJAC0011":
            d = d.rename(columns={"Minimum temperature (Degree C)": "temp_min_c"})[["date", "temp_min_c"]]
        else:
            continue
        dfs.append(d)
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="date", how="outer")
    if "temp_max_c" in merged and "temp_min_c" in merged:
        merged["temp_c"] = merged[["temp_max_c", "temp_min_c"]].mean(axis=1)
    merged["weather_source_type"] = "bom_daily"
    return merged


def load_metro_emissions_daily(path):
    df = read_metro_xls(path)
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    val_cols = [c for c in df.columns if c not in ("Date", "Time", "date")]
    df[val_cols] = df[val_cols].apply(pd.to_numeric, errors="coerce")
    daily = df.groupby("date", as_index=False)[val_cols].mean()
    rename = {}
    for c in val_cols:
        cu = c.upper()
        if "NO2" in cu: rename[c] = "no2_pphm"
        elif "CO" in cu and "OZONE" not in cu: rename[c] = "co_ppm"
        elif "OZONE" in cu: rename[c] = "ozone_pphm"
    daily = daily.rename(columns=rename)
    for needed in ("no2_pphm", "co_ppm", "ozone_pphm"):
        if needed not in daily.columns:
            daily[needed] = pd.NA
    return daily


def build_daily(aq_verification=None, speed_zone_info=None):
    all_rows = []
    for station_file, cfg in STATIONS.items():
        station_id = station_file.split(" - ")[0]
        tpath = os.path.join(TRAFFIC_DIR, station_file + ".csv")
        traffic = load_traffic_daily(tpath)

        wtype, wname = cfg["weather"]
        weather = (load_metro_weather_daily(find_metro_file(METRO_WEATHER_DIR, wname))
                   if wtype == "metro" else load_bom_weather_daily(wname))
        if wtype == "metro":
            weather = add_borrowed_rain(weather, cfg, load_metro_weather_daily, ["date"])

        merged = traffic.merge(weather, on="date", how="left")

        if cfg["emissions"] is not None:
            _, ename = cfg["emissions"]
            epath = find_metro_file(METRO_EMISSIONS_DIR, ename)
            emissions = load_metro_emissions_daily(epath)
            merged = merged.merge(emissions, on="date", how="left")
        else:
            merged["no2_pphm"] = pd.NA
            merged["co_ppm"] = pd.NA
            merged["ozone_pphm"] = pd.NA

        merged = _attach_station_metadata(merged, station_id, station_file, cfg, aq_verification, speed_zone_info)
        merged["has_no2_coverage"] = merged["no2_pphm"].notna().any()

        all_rows.append(merged)
        print(f"  [daily] {station_id}: {len(merged)} days loaded")

    final = pd.concat(all_rows, ignore_index=True)
    col_order = [
        "date", "station_id", "station_name", "station_lat", "station_lon", "station_suburb", "station_lga",
        "traffic_volume_total", "traffic_volume_light", "traffic_volume_heavy",
        "public_holiday", "school_holiday",
        "temp_c", "temp_max_c", "temp_min_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg", "rain_mm",
        "weather_source_type",
        "no2_pphm", "co_ppm", "ozone_pphm",
        "matched_aq_site", "aq_distance_km", "aq_match_flag",
        "posted_speed_kmh", "speed_zone_type", "speed_zone_match_dist_m",
        "has_no2_coverage",
    ]
    col_order = [c for c in col_order if c in final.columns] + [c for c in final.columns if c not in col_order]
    final = final[col_order]
    os.makedirs(OUT_DIR, exist_ok=True)
    final.to_csv(DAILY_OUT_PATH, index=False)
    print(f"\nSaved DAILY dataset: {len(final)} rows, {final['station_id'].nunique()} stations -> {DAILY_OUT_PATH}")
    return final


# ---------------------------------------------------------------------------
# HOURLY BUILD (metro-weather + metro-emissions stations only)
# ---------------------------------------------------------------------------

def load_traffic_hourly(path):
    """Melt the wide hour_00..hour_23 traffic file into long (date, hour_ending, ...) rows.
    hour_00 = midnight-1am -> matches the AQ portal's 'hour ending 01:00' reading, so
    traffic hour index NN maps to hour_ending = NN + 1 (1..24)."""
    df = pd.read_csv(path)
    hour_cols = [c for c in df.columns if c.startswith("hour_")]
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce")

    # sum across cardinal directions (both directions -> one total per date/class/hour)
    grouped = df.groupby(["date", "classification_seq"], as_index=False)[hour_cols].sum(min_count=1)

    long_df = grouped.melt(id_vars=["date", "classification_seq"], value_vars=hour_cols,
                            var_name="hour_col", value_name="volume")
    long_df["hour_ending"] = long_df["hour_col"].str.replace("hour_", "", regex=False).astype(int) + 1

    pivot = long_df.pivot_table(index=["date", "hour_ending"], columns="classification_seq",
                                 values="volume", aggfunc="sum").reset_index()
    pivot = pivot.rename(columns={
        "All Vehicles": "traffic_volume_total",
        "Light Vehicles": "traffic_volume_light",
        "Heavy Vehicles": "traffic_volume_heavy",
    })

    school = df.groupby("date", as_index=False)["school_holiday"].max()
    pivot = pivot.merge(school, on="date", how="left")
    pivot["date"] = pd.to_datetime(pivot["date"])
    pivot["public_holiday"] = real_public_holiday(pivot["date"])
    return pivot


def load_metro_weather_hourly(path):
    df = read_metro_xls(path)
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["hour_ending"] = df["Time"].str.split(":").str[0].astype(int)
    val_cols = [c for c in df.columns if c not in ("Date", "Time", "date", "hour_ending")]
    df[val_cols] = df[val_cols].apply(pd.to_numeric, errors="coerce")
    rename = {}
    for c in val_cols:
        cu = c.upper()
        if "TEMP" in cu: rename[c] = "temp_c"
        elif "HUMID" in cu: rename[c] = "humidity_pct"
        elif "WSP" in cu: rename[c] = "wind_speed_ms"
        elif "WDR" in cu: rename[c] = "wind_dir_deg"
        elif "RAIN" in cu: rename[c] = "rain_mm"
    df = df.rename(columns=rename)
    return df[["date", "hour_ending"] + list(rename.values())]


def load_metro_emissions_hourly(path):
    df = read_metro_xls(path)
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["hour_ending"] = df["Time"].str.split(":").str[0].astype(int)
    val_cols = [c for c in df.columns if c not in ("Date", "Time", "date", "hour_ending")]
    df[val_cols] = df[val_cols].apply(pd.to_numeric, errors="coerce")
    rename = {}
    for c in val_cols:
        cu = c.upper()
        if "NO2" in cu: rename[c] = "no2_pphm"
        elif "CO" in cu and "OZONE" not in cu: rename[c] = "co_ppm"
        elif "OZONE" in cu: rename[c] = "ozone_pphm"
    df = df.rename(columns=rename)
    for needed in ("no2_pphm", "co_ppm", "ozone_pphm"):
        if needed not in df.columns:
            df[needed] = pd.NA
    return df[["date", "hour_ending", "no2_pphm", "co_ppm", "ozone_pphm"]]


def build_hourly(aq_verification=None, speed_zone_info=None):
    hourly_capable = {
        name: cfg for name, cfg in STATIONS.items()
        if cfg["weather"][0] == "metro" and cfg["emissions"] is not None and cfg["emissions"][0] == "metro"
    }
    print(f"\n{len(hourly_capable)} of {len(STATIONS)} stations qualify for hourly build "
          f"(both weather and NO2 from a metro/hourly AQ source):")
    print("  " + ", ".join(s.split(" - ")[0] for s in hourly_capable))

    all_rows = []
    for station_file, cfg in hourly_capable.items():
        station_id = station_file.split(" - ")[0]
        tpath = os.path.join(TRAFFIC_DIR, station_file + ".csv")
        traffic = load_traffic_hourly(tpath)

        _, wname = cfg["weather"]
        weather = load_metro_weather_hourly(find_metro_file(METRO_WEATHER_DIR, wname))
        weather = add_borrowed_rain(weather, cfg, load_metro_weather_hourly, ["date", "hour_ending"])

        _, ename = cfg["emissions"]
        emissions = load_metro_emissions_hourly(find_metro_file(METRO_EMISSIONS_DIR, ename))

        merged = traffic.merge(weather, on=["date", "hour_ending"], how="left")
        merged = merged.merge(emissions, on=["date", "hour_ending"], how="left")

        merged = _attach_station_metadata(merged, station_id, station_file, cfg, aq_verification, speed_zone_info)
        merged["has_no2_coverage"] = merged["no2_pphm"].notna().any()

        all_rows.append(merged)
        print(f"  [hourly] {station_id}: {len(merged)} station-hours loaded")

    final = pd.concat(all_rows, ignore_index=True)
    col_order = [
        "date", "hour_ending", "station_id", "station_name", "station_lat", "station_lon", "station_suburb", "station_lga",
        "traffic_volume_total", "traffic_volume_light", "traffic_volume_heavy",
        "public_holiday", "school_holiday",
        "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg", "rain_mm",
        "no2_pphm", "co_ppm", "ozone_pphm",
        "matched_aq_site", "aq_distance_km", "aq_match_flag",
        "posted_speed_kmh", "speed_zone_type", "speed_zone_match_dist_m",
        "has_no2_coverage",
    ]
    col_order = [c for c in col_order if c in final.columns] + [c for c in final.columns if c not in col_order]
    final = final[col_order]
    os.makedirs(OUT_DIR, exist_ok=True)
    final.to_csv(HOURLY_OUT_PATH, index=False)
    print(f"\nSaved HOURLY dataset: {len(final)} rows, {final['station_id'].nunique()} stations -> {HOURLY_OUT_PATH}")
    return final


if __name__ == "__main__":
    print("=== Verifying station metadata (AQ site + speed zone matching) ===")
    aq_verification = verify_aq_site_matches(STATIONS)
    speed_zone_info = match_speed_zones(STATIONS)

    print(f"\n=== Building DAILY dataset (all {len(STATIONS)} candidate stations) ===")
    build_daily(aq_verification, speed_zone_info)
    print("\n=== Building HOURLY dataset (metro-only stations) ===")
    build_hourly(aq_verification, speed_zone_info)