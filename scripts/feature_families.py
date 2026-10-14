"""
Feature families: assigns each feature to a group (NO2 history, traffic, weather, calendar, road/station) so SHAP, permutation importance and ablation all group features the same way.

Matching is by keyword in the feature name and the first family that matches wins, so order matters.
"""

from __future__ import annotations

FAMILIES = [
    ("NO2 history", ("no2_",)),
    ("Traffic x weather", ("dispersion_proxy", "traffic_total_x_rainy",
                           "traffic_total_x_temperature")),
    ("Traffic", ("traffic", "vehicle", "direction_share", "directional", "direction_count")),
    ("Weather", ("temp", "rain", "precip", "wind", "humid", "pressure", "solar",
                 "radiation", "cloud", "dew", "weather")),
    ("Calendar", ("hour", "dow", "day_of_week", "weekday", "weekend", "month", "calendar",
                  "quarter", "season", "holiday", "day_of_year", "year_day", "doy", "working_day", "is_")),
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


TRAFFIC_PARTS = [   
    ("Unexpected traffic", ("unexpected",)),
    ("Heavy vehicles", ("heavy",)),
    ("Direction of flow", ("direction_share", "directional", "direction_count")),
    ("Traffic x time", ("_x_working_day", "_x_weekend", "_x_am_peak", "_x_pm_peak", "_x_peak")),
]


def traffic_part(feature: str) -> str | None:
    if family_of(feature) != "Traffic":
        return None
    name = feature.lower()
    for part, keys in TRAFFIC_PARTS:
        if any(key in name for key in keys):
            return part
    return "Total volume"


def group_traffic_parts(features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for feature in features:
        part = traffic_part(feature)
        if part:
            groups.setdefault(part, []).append(feature)
    order = [p for p, _ in TRAFFIC_PARTS] + ["Total volume"]
    return {p: groups[p] for p in order if p in groups}