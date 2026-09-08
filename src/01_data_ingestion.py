"""
CO2/NO2 Traffic-Emissions Capstone -- Raw Data Combination Script
====================================================================
Builds TWO final datasets from the raw traffic/weather/emissions files:

  1. DAILY  -- all 15 candidate stations, daily resolution. Uses BOM daily
     weather for rural stations, so this is the only resolution at which
     every station can be included on equal footing.

  2. HOURLY -- only stations where BOTH weather AND NO2 come from a metro
     AQ-network site (genuine hourly data on both sides). Rural/BOM-weather
     stations are structurally excluded here since BOM only reports daily
     min/max temp and rainfall, not hourly readings -- there's nothing to
     build hourly data from for those stations.

Fixes applied vs. the earlier version:
  - `public_holiday` in the raw TfNSW export is broken (always 0, even on
    New Year's Day). It's rebuilt here from a real NSW public holiday
    calendar (the `holidays` package) instead of trusting the raw column.

Station-metadata enrichment added:
  - `matched_aq_site`/`aq_distance_km` are cross-checked (not overwritten)
    against the NSW AQ site registry via haversine distance -- see
    `verify_aq_site_matches()`. Any station whose hardcoded match isn't
    the true nearest geocoded site is flagged for manual review (usually
    because the nearest site doesn't measure NO2, which this registry
    doesn't tell us -- so it's a flag, not an auto-correction).
  - `posted_speed_kmh`/`speed_zone_type` are pulled from the TfNSW speed
    zones shapefile via nearest-line matching -- see `match_speed_zones()`.
    Both are optional: if the AQ site registry or the speed zones
    shapefile aren't present locally, ingestion still runs, just without
    those extra columns (a warning is printed either way).

Design principle unchanged: nothing is filtered/dropped here except the
above structural exclusion for the hourly build. Station-level modeling
decisions (dropping no-NO2-coverage stations, thin-coverage stations,
missing target rows) still happen in 02_data_preprocessing.py, not here.

Requires: pandas, python-calamine, holidays, pyshp
    pip install pandas python-calamine holidays pyshp

Expected folder layout (relative to project root):
    data/raw/traffic/<station name>.csv
    data/raw/weather/tmp_table_*_<site>.xls        - metro AQ-site weather
    data/raw/weather/IDCJAC00{09,10,11}_..._<site>_Data.csv - BOM rural daily weather
    data/raw/emissions/tmp_table_*_<site>.xls       - metro AQ-site pollutant data
    data/raw/air_quality/nsw_air_quality_sites.json - optional, enables AQ-site verification
    data/raw/speed_zones/Speed_Zones.{shp,shx,dbf,prj} - optional, enables posted-speed matching
"""

import pandas as pd
import glob, os, json, math
import holidays as holidays_lib

TRAFFIC_DIR = "data/raw/traffic"
BOM_DIR = "data/raw/weather"
METRO_WEATHER_DIR = "data/raw/weather"
METRO_EMISSIONS_DIR = "data/raw/emissions"
AQ_SITES_PATH = "data/raw/air_quality/nsw_air_quality_sites.json"
SPEED_ZONES_SHP = "data/raw/speed_zones/Speed_Zones.shp"
OUT_DIR = "data/processed"
DAILY_OUT_PATH = os.path.join(OUT_DIR, "final_combined_dataset_daily.csv")
HOURLY_OUT_PATH = os.path.join(OUT_DIR, "final_combined_dataset_hourly.csv")
AQ_VERIFICATION_OUT_PATH = os.path.join(OUT_DIR, "aq_site_verification.csv")

NSW_HOLIDAYS = holidays_lib.Australia(subdiv="NSW", years=range(2023, 2027))

# station registry: every candidate station, its matched weather/emissions source,
# and metadata about the AQ-site match quality (kept, not used to filter here)
STATIONS = {
    "MUB001 - Melbourne Street":    {"lat": -36.003071, "lon": 146.003235, "suburb": "Mulwala",        "lga": "Corowa",           "weather": ("bom", "mulwala"),        "emissions": ("metro", "albury"),        "aq_site": "ALBURY",           "aq_distance_km": 83.2},
    "6135-PR - M31 Hume Highway":   {"lat": -34.791008, "lon": 148.863480, "suburb": "Bowning",         "lga": "Yass Valley",      "weather": ("metro", "goulburn"),     "emissions": ("metro", "goulburn"),      "aq_site": "GOULBURN",         "aq_distance_km": 78.1},
    "6149 - Newell Highway":        {"lat": -32.657825, "lon": 148.199264, "suburb": "Tomingley",       "lga": "Narromine",        "weather": ("bom", "tomingley"),      "emissions": None,                        "aq_site": "ORANGE (no NO2)",  "aq_distance_km": 109.2},
    "6141 - Newell Highway":        {"lat": -33.410191, "lon": 147.969162, "suburb": "Forbes",          "lga": "Forbes",           "weather": ("bom", "forbes"),         "emissions": None,                        "aq_site": "ORANGE (no NO2)",  "aq_distance_km": 106.1},
    "6105 - Great Western Highway": {"lat": -33.437302, "lon": 149.929047, "suburb": "Meadow Flat",     "lga": "Lithgow",          "weather": ("metro", "bathurst"),     "emissions": None,                        "aq_site": "BATHURST (no NO2 sensor)", "aq_distance_km": 32.7},
    "6124 - Macleay Valley Way":    {"lat": -31.105118, "lon": 152.831375, "suburb": "South Kempsey",   "lga": "Kempsey",          "weather": ("bom", "kempsey"),        "emissions": ("metro", "port_macquaire"), "aq_site": "PORT MACQUARIE",   "aq_distance_km": 36.9},
    "6116 - Pacific Highway":       {"lat": -28.952976, "lon": 153.464539, "suburb": "Wardell",         "lga": "Ballina",          "weather": ("bom", "wardell"),        "emissions": None,                        "aq_site": "LISMORE (no continuous NO2)", "aq_distance_km": 23.8},
    "7212 - Stewart Avenue":        {"lat": -32.926315, "lon": 151.758545, "suburb": "Newcastle West",  "lga": "Newcastle",        "weather": ("metro", "newcastle"),    "emissions": ("metro", "newcastle"),     "aq_site": "NEWCASTLE",        "aq_distance_km": 0.2},
    "7211 - Lily Lane":             {"lat": -32.940571, "lon": 151.713120, "suburb": "Adamstown",       "lga": "Newcastle",        "weather": ("metro", "newcastle"),    "emissions": ("metro", "newcastle"),     "aq_site": "NEWCASTLE",        "aq_distance_km": 4.5},
    "6119-PR - Pacific Highway":    {"lat": -32.088028, "lon": 152.405975, "suburb": "Nabiac",          "lga": "Greater Taree",    "weather": ("bom", "nabiac"),         "emissions": ("metro", "port_macquaire"), "aq_site": "PORT MACQUARIE",   "aq_distance_km": 87.3},
    "7216 - Gladstone Avenue":      {"lat": -34.425961, "lon": 150.887421, "suburb": "Wollongong",      "lga": "Wollongong",       "weather": ("metro", "wollongong"),   "emissions": ("metro", "wollongong"),    "aq_site": "WOLLONGONG",       "aq_distance_km": 0.6},
    "6109 - M23 Federal Highway":   {"lat": -34.820808, "lon": 149.599777, "suburb": "Yarra",           "lga": "Goulburn Mulwaree","weather": ("metro", "goulburn"),     "emissions": ("metro", "goulburn"),      "aq_site": "GOULBURN",         "aq_distance_km": 13.0},
    "10011 - New South Head Road":  {"lat": -33.876652, "lon": 151.230606, "suburb": "Edgecliff",       "lga": "Woollahra",        "weather": ("metro", "randwick"),     "emissions": ("metro", "cook_phillip"),  "aq_site": "COOK AND PHILLIP", "aq_distance_km": 1.8},
    "100001 - Cambridge Street":    {"lat": -33.878181, "lon": 150.924942, "suburb": "Canley Heights",  "lga": "Fairfield",        "weather": ("metro", "liverpool"),    "emissions": ("metro", "liverpool"),     "aq_site": "LIVERPOOL",        "aq_distance_km": 4.8},
    "6178-PR - Picton Road":        {"lat": -34.372967, "lon": 150.829529, "suburb": "Cordeaux",        "lga": "Wollongong",       "weather": ("metro", "kembla_grange"),"emissions": ("metro", "wollongong"),    "aq_site": "WOLLONGONG",       "aq_distance_km": 8.2},
}


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
    """Read the AQ-portal .xls export directly. Non-standard file, xlrd/openpyxl
    can't open it -- python-calamine handles it fine."""
    return pd.read_excel(path, engine="calamine", skiprows=2)


def real_public_holiday(dates):
    return pd.Series(dates).dt.date.isin(NSW_HOLIDAYS).astype(int).values


# ---------------------------------------------------------------------------
# STATION METADATA ENRICHMENT -- AQ-site verification (flag only, no overwrite)
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def verify_aq_site_matches(stations, sites_path=AQ_SITES_PATH):
    """Cross-check each station's hardcoded aq_site/aq_distance_km against
    the true nearest geocoded AQ site (haversine). Does NOT correct
    STATIONS -- NO2 availability per site isn't in this registry, so the
    hardcoded (farther) site may still be the deliberate, correct choice.
    Returns None if the registry file isn't present (feature is optional)."""
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
            "flag": "REVIEW" if nearest_name.upper() not in cfg["aq_site"].upper() else "OK",
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
# STATION METADATA ENRICHMENT -- posted speed limit (speed zones shapefile)
# ---------------------------------------------------------------------------

_EARTH_R = 6378137.0  # matches Speed_Zones.prj (WGS_1984_Web_Mercator_Auxiliary_Sphere)


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
    """One pass through the (447k-record) shapefile, checked against every
    station's bbox at once -- looping stations on the outside would rescan
    the full file per station instead. Returns {} if the shapefile isn't
    present locally (feature is optional)."""
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
    """Shared metadata-attach step used by both build_daily() and
    build_hourly() -- static per-station attributes, not per-row logic."""
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

    print("\n=== Building DAILY dataset (all 15 candidate stations) ===")
    build_daily(aq_verification, speed_zone_info)
    print("\n=== Building HOURLY dataset (metro-only stations) ===")
    build_hourly(aq_verification, speed_zone_info)