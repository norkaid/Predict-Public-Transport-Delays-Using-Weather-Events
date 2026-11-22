"""Feature engineering: time, weather, event and congestion features.

Everything here is a row-wise transformation that uses only information known
before the trip starts, so it is safe to apply before the train/test split.
Encodings that learn from the target (e.g. route target encoding) live inside
the model pipeline in :mod:`transport_delays.models` instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RAIN_WORDS = ("rain", "drizzle", "shower", "storm", "thunder")
SNOW_WORDS = ("snow", "sleet", "blizzard", "ice", "hail")
FOG_WORDS = ("fog", "mist", "haze")
STORM_WORDS = ("storm", "thunder", "blizzard")


def _contains_any(s: pd.Series, words) -> pd.Series:
    low = s.astype(str).str.lower()
    return low.str.contains("|".join(words), regex=True).astype(int)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Cyclical encodings so 23:00 sits next to 00:00 and Dec next to Jan.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    hour = df["hour"]
    df["time_of_day"] = pd.cut(
        hour, bins=[-1, 5, 9, 15, 19, 23],
        labels=["Night", "Morning peak", "Midday", "Evening peak", "Evening"],
    ).astype(str)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    if C.WEATHER_COL in df:
        w = df[C.WEATHER_COL]
        df["is_rain"] = _contains_any(w, RAIN_WORDS)
        df["is_snow"] = _contains_any(w, SNOW_WORDS)
        df["is_fog"] = _contains_any(w, FOG_WORDS)
        df["is_storm"] = _contains_any(w, STORM_WORDS)
    if C.TEMP_COL in df:
        t = df[C.TEMP_COL]
        df["is_freezing"] = (t <= 0).astype(int)
        df["is_hot"] = (t >= 30).astype(int)
        df["is_extreme_temp"] = ((t <= 0) | (t >= 30)).astype(int)
        # Distance from a comfortable ~15 °C: captures U-shaped temperature effects.
        df["temp_deviation"] = (t - 15).abs()
    if C.PRECIP_COL in df:
        p = df[C.PRECIP_COL]
        df["log_precipitation"] = np.log1p(p)
        df["heavy_precipitation"] = (p >= 10).astype(int)
    if C.WIND_COL in df:
        df["high_wind"] = (df[C.WIND_COL] >= 40).astype(int)

    flags = [c for c in ("is_rain", "is_snow", "is_storm", "is_extreme_temp",
                         "heavy_precipitation", "high_wind") if c in df]
    if flags:
        df["adverse_weather_score"] = df[flags].sum(axis=1)
    return df


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    if C.EVENT_COL in df:
        df["has_event"] = (df[C.EVENT_COL] != "None").astype(int)
    if C.ATTENDANCE_COL in df:
        df["log_attendance"] = np.log1p(df[C.ATTENDANCE_COL].clip(lower=0))
        df["large_event"] = (df[C.ATTENDANCE_COL] >= 10_000).astype(int)
    if "has_event" in df and C.PEAK_COL in df:
        df["event_x_peak"] = df["has_event"] * df[C.PEAK_COL]
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if C.CONGESTION_COL in df:
        cong = df[C.CONGESTION_COL]
        if C.PRECIP_COL in df:
            df["precip_x_congestion"] = df["log_precipitation"] * cong
        if "adverse_weather_score" in df:
            df["weather_x_congestion"] = df["adverse_weather_score"] * cong
        if C.PEAK_COL in df:
            df["peak_x_congestion"] = df[C.PEAK_COL] * cong
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to a cleaned frame (returns a copy)."""
    df = df.copy()
    for step in (add_time_features, add_weather_features,
                 add_event_features, add_interaction_features):
        df = step(df)
    return df


# Columns that identify a trip or duplicate the timestamp; never model inputs.
_DROP = {
    C.ID_COL, C.DATE_COL, C.TIME_COL, C.SCHED_DEP_COL, C.SCHED_ARR_COL,
    "timestamp", *C.LEAKY_COLS,
}
# High-cardinality categoricals get target encoding instead of one-hot.
HIGH_CARDINALITY = [C.ROUTE_COL, C.ORIGIN_COL, C.DEST_COL]


def split_feature_types(df: pd.DataFrame, target: str = C.TARGET):
    """Return (numeric, low-cardinality categorical, high-cardinality categorical)."""
    candidates = [c for c in df.columns if c not in _DROP and c != target]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in candidates if c not in numeric]
    high = [c for c in cats if c in HIGH_CARDINALITY or df[c].nunique() > 30]
    low = [c for c in cats if c not in high]
    return numeric, low, high
