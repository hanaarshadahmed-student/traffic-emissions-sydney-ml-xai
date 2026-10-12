"""
Feature families -- which group each model feature belongs to (NO2 history,
traffic, weather, calendar, road/station...).

Defined once here so SHAP (src/08_explainability.py), permutation importance
and the ablation study all group features the same way.

Matched on the feature name; the FIRST family whose keyword appears in the
name wins, so order matters (e.g. "traffic_dispersion_proxy" contains both
"traffic" and, through the proxy, wind -- it is caught by "Traffic x weather"
before "Traffic").
"""

from __future__ import annotations

FAMILIES = [
    ("NO2 history", ("no2_",)),
    # traffic combined with weather in one feature (e.g. traffic_dispersion_proxy
    # = log traffic / (1 + wind speed)) -- kept apart so the "Traffic" share
    # isn't partly wind
    ("Traffic x weather", ("dispersion_proxy", "traffic_total_x_rainy",
                           "traffic_total_x_temperature")),
    # "direction_share" / "directional" / "direction_count" = which way the
    # traffic flows (not wind_direction, which is weather)
    ("Traffic", ("traffic", "vehicle", "direction_share", "directional", "direction_count")),
    ("Weather", ("temp", "rain", "precip", "wind", "humid", "pressure", "solar",
                 "radiation", "cloud", "dew", "weather")),
    ("Calendar", ("hour", "dow", "day_of_week", "weekday", "weekend", "month", "calendar",
                  "quarter", "season", "holiday", "day_of_year", "doy", "working_day", "is_")),
    ("Road / station", ("road", "lane", "station", "stn_", "region", "distance", "dist_", "zone",
                        "speed", "intersection", "aq_", "lat", "lon", "elevation", "class", "type")),
]


def family_of(feature: str) -> str:
    name = feature.lower()
    for family, keys in FAMILIES:
        if any(key in name for key in keys):
            return family
    return "Other"


def group_features(features: list[str]) -> dict[str, list[str]]:
    """{family: [features in it]}, families in FAMILIES order, "Other" last."""
    groups: dict[str, list[str]] = {name: [] for name, _ in FAMILIES}
    groups["Other"] = []
    for feature in features:
        groups[family_of(feature)].append(feature)
    return {name: cols for name, cols in groups.items() if cols}

# --- traffic broken into its parts (RQ1: WHICH traffic factors matter) -------
# Applied only to features in the "Traffic" family. First match wins.
TRAFFIC_PARTS = [
    # traffic compared with what is normal for that station / hour / weekday
    # (04_feature_engineering.py: add_unexpected_traffic_features)
    ("Unexpected traffic", ("unexpected",)),
    # heavy vehicles (trucks, buses): mostly diesel, far more NOx per vehicle
    ("Heavy vehicles", ("heavy",)),
    # which way the traffic flows (share northbound, imbalance, ...)
    ("Direction of flow", ("direction_share", "directional", "direction_count")),
    # traffic multiplied by a time flag (traffic x peak hour / working day / weekend)
    ("Traffic x time", ("_x_working_day", "_x_weekend", "_x_am_peak", "_x_pm_peak", "_x_peak")),
]


def traffic_part(feature: str) -> str | None:
    """Which part of traffic a feature describes, or None if it isn't in the
    Traffic family. Anything traffic that matches no part is "Total volume"
    (counts, lags, rolling means, changes, ratios to the station's average)."""
    if family_of(feature) != "Traffic":
        return None
    name = feature.lower()
    for part, keys in TRAFFIC_PARTS:
        if any(key in name for key in keys):
            return part
    return "Total volume"


def group_traffic_parts(features: list[str]) -> dict[str, list[str]]:
    """{traffic part: [features in it]} for the Traffic-family features only."""
    groups: dict[str, list[str]] = {}
    for feature in features:
        part = traffic_part(feature)
        if part:
            groups.setdefault(part, []).append(feature)
    order = [p for p, _ in TRAFFIC_PARTS] + ["Total volume"]
    return {p: groups[p] for p in order if p in groups}