"""Exploratory analysis: how weather, events and time relate to delays.

Each ``*_effect`` function returns a tidy summary table; ``make_figures``
renders the charts used in the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

# Reference palette (validated): slot 1 blue, slot 2 orange; recessive ink.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _summary(df: pd.DataFrame, by, target: str, threshold: float) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[target]
    out = pd.DataFrame({
        "trips": g.size(),
        "mean_delay_min": g.mean(),
        "median_delay_min": g.median(),
        "p90_delay_min": g.quantile(0.9),
        "delay_probability": g.apply(lambda s: (s >= threshold).mean()),
    })
    return out.round(3)


def weather_effect(df, target=C.TARGET, threshold=C.DELAY_THRESHOLD_MIN):
    """Delay statistics by weather condition, sorted worst first."""
    return _summary(df, C.WEATHER_COL, target, threshold).sort_values(
        "mean_delay_min", ascending=False)


def temperature_effect(df, target=C.TARGET, threshold=C.DELAY_THRESHOLD_MIN):
    bins = [-np.inf, 0, 10, 20, 30, np.inf]
    labels = ["<= 0 °C", "0-10 °C", "10-20 °C", "20-30 °C", "> 30 °C"]
    band = pd.cut(df[C.TEMP_COL], bins=bins, labels=labels)
    return _summary(df.assign(temp_band=band), "temp_band", target, threshold)


def precipitation_effect(df, target=C.TARGET, threshold=C.DELAY_THRESHOLD_MIN):
    bins = [-np.inf, 0, 2.5, 10, np.inf]
    labels = ["None", "Light (<2.5 mm)", "Moderate (2.5-10 mm)", "Heavy (>10 mm)"]
    band = pd.cut(df[C.PRECIP_COL], bins=bins, labels=labels)
    return _summary(df.assign(precip_band=band), "precip_band", target, threshold)


def event_effect(df, target=C.TARGET, threshold=C.DELAY_THRESHOLD_MIN):
    """Compare trips with and without a city event.

    Returns the per-group table plus a dict of significance tests:
    Welch's t-test on mean delay and a chi-square test on delay probability.
    """
    has = df[C.EVENT_COL] != "None"
    table = _summary(df.assign(event=np.where(has, "Event", "No event")),
                     "event", target, threshold)
    by_type = _summary(df, C.EVENT_COL, target, threshold).sort_values(
        "mean_delay_min", ascending=False)

    tests = {}
    a, b = df.loc[has, target], df.loc[~has, target]
    if len(a) > 1 and len(b) > 1:
        t = stats.ttest_ind(a, b, equal_var=False)
        tests["welch_t"] = float(t.statistic)
        tests["welch_p"] = float(t.pvalue)
        delayed = df[target] >= threshold
        ct = pd.crosstab(has, delayed)
        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
            tests["chi2"] = float(chi2)
            tests["chi2_p"] = float(p)
        p_event, p_none = delayed[has].mean(), delayed[~has].mean()
        tests["delay_prob_event"] = float(p_event)
        tests["delay_prob_no_event"] = float(p_none)
        tests["relative_risk"] = float(p_event / p_none) if p_none > 0 else float("nan")
    return table, by_type, tests


def time_effect(df, target=C.TARGET, threshold=C.DELAY_THRESHOLD_MIN):
    """Delay by hour of day, day of week and month."""
    return {
        "hour": _summary(df, "hour", target, threshold),
        "day_of_week": _summary(df, "day_of_week", target, threshold),
        "month": _summary(df, "month", target, threshold),
    }


def correlations(df, target=C.TARGET) -> pd.Series:
    """Spearman correlation of every numeric column with the target."""
    num = df.select_dtypes("number").drop(columns=[c for c in C.LEAKY_COLS if c in df])
    num = num.loc[:, num.nunique() > 1]  # constant columns have no defined correlation
    return num.corrwith(df[target], method="spearman").drop(target, errors="ignore") \
              .dropna().sort_values(key=np.abs, ascending=False)


# --- Figures ----------------------------------------------------------------

def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=INK_2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def _bar(ax, table, col, title, ylabel, fmt="{:.1f}", horizontal=False):
    labels = [str(i) for i in table.index]
    vals = table[col].to_numpy()
    if horizontal:
        ax.barh(labels, vals, color=BLUE, height=0.6)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for y, v in enumerate(vals):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9, color=INK_2)
        _style(ax, title, xlabel=ylabel)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(labels, vals, color=BLUE, width=0.6)
        _style(ax, title, ylabel=ylabel)


def make_figures(df: pd.DataFrame, out_dir: Path, target=C.TARGET,
                 threshold=C.DELAY_THRESHOLD_MIN) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 10})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # 1. Delay distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    y = df[target].dropna()
    if np.allclose(y, y.round()) and y.max() - y.min() <= 200:
        bins = np.arange(y.min() - 0.5, y.max() + 1.5)  # one bar per whole minute
    else:
        bins = 50
    ax.hist(y, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(threshold, color=INK_2, linestyle="--", linewidth=1)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f"  {threshold:g}-min delay threshold",
            color=INK_2, fontsize=9, va="top")
    _style(ax, "Distribution of arrival delay", "Arrival delay (min)", "Trips")
    save(fig, "01_delay_distribution.png")

    # 2. Weather: mean delay + delay probability
    if C.WEATHER_COL in df:
        w = weather_effect(df, target, threshold)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        _bar(axes[0], w, "mean_delay_min", "Mean arrival delay by weather",
             "Minutes", horizontal=True)
        _bar(axes[1], w, "delay_probability",
             f"Share of trips delayed ≥ {threshold:g} min", "Probability",
             fmt="{:.0%}", horizontal=True)
        save(fig, "02_weather_effect.png")

    # 3. Temperature & precipitation bands
    if C.TEMP_COL in df and C.PRECIP_COL in df:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        _bar(axes[0], temperature_effect(df, target, threshold), "mean_delay_min",
             "Mean delay by temperature", "Minutes")
        _bar(axes[1], precipitation_effect(df, target, threshold), "mean_delay_min",
             "Mean delay by precipitation", "Minutes")
        axes[1].tick_params(axis="x", labelrotation=15)
        save(fig, "03_temperature_precipitation.png")

    # 4. Events
    if C.EVENT_COL in df:
        _, by_type, _ = event_effect(df, target, threshold)
        fig, ax = plt.subplots(figsize=(8, 4))
        _bar(ax, by_type, "delay_probability",
             f"Share of trips delayed ≥ {threshold:g} min, by city event",
             "Probability", fmt="{:.0%}", horizontal=True)
        save(fig, "04_event_effect.png")

    # 5. Time of day / weekday: two series (event vs none) by hour
    if "hour" in df:
        fig, ax = plt.subplots(figsize=(9, 4))
        if C.EVENT_COL in df:
            has = df[C.EVENT_COL] != "None"
            for mask, label, color in ((~has, "No event", BLUE), (has, "Event", ORANGE)):
                h = df[mask].groupby("hour")[target].mean()
                ax.plot(h.index, h.values, color=color, linewidth=2, marker="o",
                        markersize=4, label=label)
                ax.text(h.index[-1] + 0.3, h.values[-1], label, color=INK_2,
                        fontsize=9, va="center")
            ax.legend(frameon=False, loc="upper left")
        else:
            h = df.groupby("hour")[target].mean()
            ax.plot(h.index, h.values, color=BLUE, linewidth=2, marker="o", markersize=4)
        ax.set_xticks(range(0, 24, 2))
        _style(ax, "Mean arrival delay by hour of day", "Hour", "Minutes")
        save(fig, "05_delay_by_hour.png")

    # 6. Correlations
    corr = correlations(df, target).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [BLUE if v >= 0 else ORANGE for v in corr.values]
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    _style(ax, "Spearman correlation with arrival delay (top 15)", "ρ")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    save(fig, "06_correlations.png")

    return paths
