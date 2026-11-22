"""Synthetic data with the same schema as the Kaggle dataset.

Used for tests and for trying the pipeline before the real CSV is
downloaded. The effect sizes here are invented: never report findings from
synthetic data as findings about real transit systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER = ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Storm"]
WEATHER_P = [0.35, 0.25, 0.18, 0.08, 0.07, 0.07]
WEATHER_DELAY = {"Clear": 0, "Cloudy": 1, "Rain": 4, "Snow": 9, "Fog": 3, "Storm": 11}
EVENTS = ["None", "Concert", "Sports", "Festival", "Parade", "Protest"]
EVENTS_P = [0.70, 0.08, 0.09, 0.05, 0.04, 0.04]
TRANSPORT = ["Bus", "Tram", "Metro", "Train"]


def generate(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    ts = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="min")
    ts = pd.Series(ts).sort_values().reset_index(drop=True)
    month = ts.dt.month
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"], "Autumn")

    weather = rng.choice(WEATHER, n, p=WEATHER_P)
    winter = np.isin(season, ["Winter"])
    weather = np.where((weather == "Snow") & ~winter, "Rain", weather)
    temp = 12 - 10 * np.cos(2 * np.pi * (month - 1) / 12) + rng.normal(0, 5, n)
    temp = np.where(weather == "Snow", np.minimum(temp, 1), temp)
    precip = np.where(np.isin(weather, ["Rain", "Storm", "Snow"]),
                      rng.gamma(2, 3, n), 0.0)
    wind = rng.gamma(3, 5, n) + np.where(weather == "Storm", 30, 0)
    humidity = np.clip(55 + precip * 2 + rng.normal(0, 12, n), 10, 100)

    event = rng.choice(EVENTS, n, p=EVENTS_P)
    attendance = np.where(event == "None", 0,
                          rng.lognormal(9, 0.8, n)).round()
    hour = ts.dt.hour
    peak = (hour.between(7, 9) | hour.between(16, 18)).astype(int)
    holiday = (rng.random(n) < 0.04).astype(int)
    congestion = np.clip(3 + 3 * peak + 1.5 * (event != "None") + rng.normal(0, 1.5, n), 0, 10)
    transport = rng.choice(TRANSPORT, n)
    routes = [f"R{i:02d}" for i in range(1, 21)]
    route = rng.choice(routes, n)
    route_bias = {r: b for r, b in zip(routes, rng.normal(0, 2, len(routes)))}
    stations = [f"Station_{i}" for i in range(1, 16)]

    weather_eff = np.vectorize(WEATHER_DELAY.get)(weather)
    delay = (
        -4
        + weather_eff
        + 0.25 * np.abs(temp - 15) * (np.abs(temp - 15) > 12)
        + 0.4 * precip
        + 0.08 * np.maximum(wind - 30, 0)
        + 1.2 * congestion
        + np.where(event != "None", 3 + 1.5 * np.log1p(attendance / 1000), 0)
        + 2.5 * peak
        - 1.5 * holiday
        + np.vectorize(route_bias.get)(route)
        + np.where(transport == "Metro", -2, 0)
        + rng.normal(0, 3, n)
    )
    arr_delay = np.round(np.maximum(delay, -3), 1)
    dep_delay = np.round(np.maximum(arr_delay * 0.7 + rng.normal(0, 1.5, n), 0), 1)
    duration = rng.integers(10, 90, n)

    df = pd.DataFrame({
        "trip_id": [f"T{i:05d}" for i in range(n)],
        "date": ts.dt.strftime("%Y-%m-%d"),
        "time": ts.dt.strftime("%H:%M:%S"),
        "transport_type": transport,
        "route_id": route,
        "origin_station": rng.choice(stations, n),
        "destination_station": rng.choice(stations, n),
        "scheduled_departure": ts.dt.strftime("%H:%M:%S"),
        "scheduled_arrival": (ts + pd.to_timedelta(duration, unit="min")).dt.strftime("%H:%M:%S"),
        "actual_departure_delay_min": dep_delay,
        "actual_arrival_delay_min": arr_delay,
        "weather_condition": weather,
        "temperature_C": temp.round(1),
        "humidity_percent": humidity.round(1),
        "wind_speed_kmh": wind.round(1),
        "precipitation_mm": precip.round(1),
        "event_type": np.where(event == "None", None, event),
        "event_attendance_est": attendance,
        "traffic_congestion_index": congestion.round(2),
        "holiday": holiday.astype(bool),
        "peak_hour": peak.astype(bool),
        "weekday": ts.dt.day_name(),
        "season": season,
        "delayed": (arr_delay >= 5).astype(int),
    })
    # Sprinkle in missing values and a duplicate row to exercise cleaning.
    for col in ("temperature_C", "humidity_percent", "weather_condition"):
        idx = rng.choice(n, size=n // 100, replace=False)
        df.loc[idx, col] = np.nan
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)
