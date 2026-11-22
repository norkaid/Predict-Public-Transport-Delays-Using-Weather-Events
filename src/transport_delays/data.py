"""Loading and cleaning the raw trip records."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and replace spaces/punctuation with ``_``."""
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower())).strip("_")
        for c in df.columns
    ]
    return df


def load_raw(path: str | Path = C.DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle "
            "(see data/README.md) or run with --synthetic."
        )
    df = pd.read_csv(path)
    log.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return normalise_columns(df)


def _to_bool_int(s: pd.Series) -> pd.Series:
    mapping = {"true": 1, "yes": 1, "y": 1, "1": 1, "false": 0, "no": 0, "n": 0, "0": 0}
    out = s.fillna("").astype(str).str.strip().str.lower().map(mapping)
    return out.fillna(0).astype(int)


def _combine_datetime(df: pd.DataFrame) -> pd.Series:
    """Trip start timestamp from ``date`` + ``time`` (or scheduled departure).

    Handles a time column holding either a clock time ("08:15") or a full
    timestamp ("2023-01-01 08:15:00").
    """
    date = pd.to_datetime(df[C.DATE_COL], errors="coerce") if C.DATE_COL in df else None
    for tcol in (C.TIME_COL, C.SCHED_DEP_COL):
        if tcol not in df:
            continue
        t = pd.to_datetime(df[tcol].astype(str).str.strip(), errors="coerce", format="mixed")
        if date is not None:
            # Clock-only strings parse onto today's date: keep only the time part.
            ts = date.dt.normalize() + (t - t.dt.normalize())
        else:
            ts = t
        if ts.notna().mean() > 0.5:
            return ts
    if date is not None:
        return date
    raise ValueError("Could not derive a timestamp: no date/time columns found.")


def _minutes_of_day(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", format="mixed")
    return t.dt.hour * 60 + t.dt.minute


def clean(df: pd.DataFrame, target: str = C.TARGET) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    Steps: de-duplicate, parse timestamps, coerce numeric types, clip values to
    physically plausible ranges, impute missing values, standardise
    categoricals, and drop rows without a target.
    """
    df = normalise_columns(df)
    n0 = len(df)

    if C.ID_COL in df:
        df = df.drop_duplicates(subset=C.ID_COL)
    df = df.drop_duplicates()

    if target not in df:
        raise KeyError(f"Target column '{target}' not in data. Columns: {list(df.columns)}")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].notna()].copy()

    df["timestamp"] = _combine_datetime(df)
    df = df[df["timestamp"].notna()].copy()

    # Planned trip length, handling trips that cross midnight.
    if C.SCHED_DEP_COL in df and C.SCHED_ARR_COL in df:
        dep = _minutes_of_day(df[C.SCHED_DEP_COL])
        arr = _minutes_of_day(df[C.SCHED_ARR_COL])
        dur = (arr - dep) % (24 * 60)
        df["planned_duration_min"] = dur.where(dur > 0)

    for col in C.NUMERIC_RAW + [C.DEPARTURE_DELAY]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, (lo, hi) in C.VALID_RANGES.items():
        if col in df:
            df[col] = df[col].clip(lower=lo, upper=hi)

    for col in C.BOOL_RAW + [C.DELAYED_FLAG]:
        if col in df:
            df[col] = _to_bool_int(df[col])

    # Event type: missing means "no event".
    if C.EVENT_COL in df:
        ev = df[C.EVENT_COL].fillna("").astype(str).str.strip()
        df[C.EVENT_COL] = ev.where(~ev.str.lower().isin(C.NO_EVENT_LABELS), "None")
        if C.ATTENDANCE_COL in df:
            df.loc[df[C.EVENT_COL] == "None", C.ATTENDANCE_COL] = (
                df.loc[df[C.EVENT_COL] == "None", C.ATTENDANCE_COL].fillna(0)
            )

    for col in df.columns:
        if col in (target, "timestamp"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            if col in (C.WEATHER_COL, C.TRANSPORT_COL, C.SEASON_COL, C.WEEKDAY_COL):
                df[col] = df[col].str.title()

    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Cleaning kept %d of %d rows", len(df), n0)
    return df


def summarise_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missing share and number of unique values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })


__all__ = ["load_raw", "clean", "normalise_columns", "summarise_quality"]
