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