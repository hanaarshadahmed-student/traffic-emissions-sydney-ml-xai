"""
Station list -- the ONE place stations are defined.

Each key is "<station id> - <road name>" and must match the traffic file
name: data/raw/traffic/<key>.csv. To add a station, add a line here and put
its traffic, weather and emissions files in data/raw/ (see README).

  weather / emissions   ("metro", <file keyword>) = AQ-network .xls file
                        ("bom", <file keyword>)   = BOM daily .csv files
                        None                      = no NO2 data for this station
  aq_site               the air-quality monitor the NO2 comes from
  aq_distance_km        distance from the traffic counter to that monitor

station_label() turns an id into a readable name for tables and charts,
e.g. "7212" -> "7212 Stewart Ave, Newcastle West".
"""

from __future__ import annotations

import pandas as pd

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
    "7163 - Victoria Road":         {"lat": -33.865181, "lon": 151.171997, "suburb": "Rozelle",         "lga": "Inner West",       "weather": ("metro", "rozelle"),          "emissions": ("metro", "rozelle"),          "aq_site": "ROZELLE",          "aq_distance_km": 0.75},
    "21229 - Military Road":        {"lat": -33.830160, "lon": 151.217280, "suburb": "Neutral Bay",     "lga": "North Sydney",     "weather": ("metro", "cammeray"),         "emissions": ("metro", "cammeray"),         "aq_site": "CAMMERAY",         "aq_distance_km": 0.33},
    "50240 - Briens Road":          {"lat": -33.796436, "lon": 150.987595, "suburb": "Northmead",       "lga": "Parramatta",       "weather": ("metro", "parramatta_north"), "emissions": ("metro", "parramatta_north"), "aq_site": "PARRAMATTA NORTH", "aq_distance_km": 1.0},
    "7289 - Thomas Street":         {"lat": -32.906013, "lon": 151.671356, "suburb": "Wallsend",        "lga": "Newcastle",        "weather": ("metro", "wallsend"),         "emissions": ("metro", "wallsend"),         "aq_site": "WALLSEND",         "aq_distance_km": 1.3},
}

# shorter road names so labels fit on chart axes
_SHORT = {"Highway": "Hwy", "Street": "St", "Avenue": "Ave", "Road": "Rd", "Lane": "Ln"}


def _short_road(road: str) -> str:
    return " ".join(_SHORT.get(word, word) for word in road.split())


# "7212" -> ("Stewart Ave", "Newcastle West")
_NAMES = {key.split(" - ")[0]: (_short_road(key.split(" - ", 1)[1]), cfg["suburb"])
          for key, cfg in STATIONS.items()}


def station_label(station_id, two_lines: bool = False) -> str:
    """'7212' -> '7212 Stewart Ave, Newcastle West'. two_lines=True puts the
    road and suburb on a second line (for crowded chart axes). Unknown ids are
    returned unchanged."""
    sid = str(station_id)
    if sid not in _NAMES:
        return sid
    road, suburb = _NAMES[sid]
    return f"{sid}\n{road}, {suburb}" if two_lines else f"{sid} {road}, {suburb}"


def add_station_name(df: pd.DataFrame, id_col: str = "station_id") -> pd.DataFrame:
    """Adds a readable `station` column right after `id_col` (the id column is
    kept, because other files are joined on it)."""
    out = df.copy()
    if "station" in out.columns:
        out = out.drop(columns="station")
    out.insert(out.columns.get_loc(id_col) + 1, "station", out[id_col].map(station_label))
    return out