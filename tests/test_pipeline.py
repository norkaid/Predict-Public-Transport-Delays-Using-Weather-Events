import numpy as np
import pandas as pd
import pytest

from transport_delays import config as C
from transport_delays.analysis import event_effect, weather_effect
from transport_delays.data import clean, normalise_columns
from transport_delays.features import build_features, split_feature_types
from transport_delays.models import time_split, train_and_evaluate
from transport_delays.pipeline import run
from transport_delays.synthetic import generate


@pytest.fixture(scope="module")
def raw():
    return generate(600, seed=1)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(clean(raw))


def test_normalise_columns():
    df = normalise_columns(pd.DataFrame(columns=["Temperature C", "Event-Type ", "trip_id"]))
    assert list(df.columns) == ["temperature_c", "event_type", "trip_id"]


def test_clean_removes_duplicates_and_missing(raw):
    df = clean(raw)
    assert len(df) == len(raw) - 1  # one injected duplicate
    assert df[C.TEMP_COL].notna().all()
    assert (df[C.WEATHER_COL] == "Unknown").any()
    assert set(df[C.HOLIDAY_COL].unique()) <= {0, 1}
    assert df["timestamp"].is_monotonic_increasing
    assert (df[C.EVENT_COL] == "None").any()


def test_clean_clips_impossible_values(raw):
    bad = raw.copy()
    bad.loc[0, "humidity_percent"] = 250
    bad.loc[1, "precipitation_mm"] = -4
    df = clean(bad)
    assert df[C.HUMIDITY_COL].max() <= 100
    assert df[C.PRECIP_COL].min() >= 0


def test_clean_requires_target(raw):
    with pytest.raises(KeyError):
        clean(raw.drop(columns=[C.TARGET]))


def test_features(feats):
    for col in ("hour_sin", "is_snow", "is_extreme_temp", "has_event",
                "adverse_weather_score", "planned_duration_min"):
        assert col in feats
    snow = feats[C.WEATHER_COL] == "Snow"
    assert feats.loc[snow, "is_snow"].eq(1).all()
    assert np.allclose(feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2, 1)


def test_no_leakage_in_features(feats):
    numeric, low, high = split_feature_types(feats)
    used = set(numeric + low + high)
    assert not used & set(C.LEAKY_COLS)
    assert C.TARGET not in used and "timestamp" not in used
    assert C.ROUTE_COL in high


def test_time_split_is_chronological(feats):
    train, test = time_split(feats, 0.25)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(test) == len(feats) - int(len(feats) * 0.75)


def test_analysis_tables(feats):
    w = weather_effect(feats)
    assert w["trips"].sum() == len(feats)
    table, by_type, tests = event_effect(feats)
    assert set(table.index) == {"Event", "No event"}
    assert 0 <= tests["chi2_p"] <= 1


def test_models_beat_baseline(feats):
    res = train_and_evaluate(feats, cv_splits=3, importance_repeats=2)
    base = res.metrics.loc["Baseline (mean)", "test_MAE"]
    assert res.metrics.loc[res.best_name, "test_MAE"] < base
    assert res.best_name != "Baseline (mean)"
    assert len(res.test_predictions) == len(feats) - int(len(feats) * 0.8)


def test_full_run_writes_outputs(raw, tmp_path):
    summary = run(raw, report_dir=tmp_path / "r", model_dir=tmp_path / "m", source="test")
    assert (tmp_path / "r" / "REPORT.md").exists()
    assert (tmp_path / "m" / "best_model.joblib").exists()
    assert len(list((tmp_path / "r" / "figures").glob("*.png"))) >= 8
    assert summary["rows_after_cleaning"] == len(raw) - 1
