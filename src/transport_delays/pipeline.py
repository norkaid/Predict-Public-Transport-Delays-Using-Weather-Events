"""End-to-end run: clean -> features -> analysis -> models -> report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from . import analysis, config as C
from .data import clean, load_raw, summarise_quality
from .features import build_features
from .models import plot_results, train_and_evaluate

log = logging.getLogger(__name__)


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _no_signal_note(metrics: pd.DataFrame, best: str) -> list[str]:
    """Flag the case where no model beats always predicting the mean delay."""
    base = metrics.loc["Baseline (mean)", "test_MAE"]
    if metrics.loc[best, "test_MAE"] < base:
        return []
    return ["> **No model beats the mean baseline on the test period.** The "
            "features carry little or no information about arrival delay in "
            "this data, so the predictions are no better than guessing the "
            "average delay.\n"]


def run(raw: pd.DataFrame, report_dir: Path = C.DEFAULT_REPORT_DIR,
        model_dir: Path = C.DEFAULT_MODEL_DIR, source: str = "",
        target: str = C.TARGET, test_size: float = 0.2) -> dict:
    report_dir, model_dir = Path(report_dir), Path(model_dir)
    fig_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = build_features(clean(raw, target=target))
    summarise_quality(raw).to_csv(report_dir / "data_quality_raw.csv")

    # --- Analysis -----------------------------------------------------------
    figs = analysis.make_figures(df, fig_dir, target)
    sections = []
    if C.WEATHER_COL in df:
        sections.append(("Delay by weather condition", analysis.weather_effect(df, target)))
    if C.TEMP_COL in df:
        sections.append(("Delay by temperature band", analysis.temperature_effect(df, target)))
    if C.PRECIP_COL in df:
        sections.append(("Delay by precipitation", analysis.precipitation_effect(df, target)))
    tests = {}
    if C.EVENT_COL in df:
        ev_table, ev_types, tests = analysis.event_effect(df, target)
        sections += [("Event vs no event", ev_table), ("Delay by event type", ev_types)]
    times = analysis.time_effect(df, target)
    sections.append(("Delay by hour of day", times["hour"]))
    sections.append(("Delay by day of week (0 = Monday)", times["day_of_week"]))

    # --- Models -------------------------------------------------------------
    result = train_and_evaluate(df, target=target, test_size=test_size)
    figs += plot_results(result, fig_dir)
    result.metrics.to_csv(report_dir / "model_metrics.csv")
    result.feature_importance.to_csv(report_dir / "feature_importance.csv", index=False)
    result.test_predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    joblib.dump({"model": result.best_model, "name": result.best_name,
                 "features": result.features, "target": target},
                model_dir / "best_model.joblib")

    summary = {
        "source": source,
        "rows_after_cleaning": int(len(df)),
        "target": target,
        "best_model": result.best_name,
        "test_metrics": result.metrics.loc[result.best_name].to_dict(),
        "event_tests": tests,
        "top_features": result.feature_importance.head(10)["feature"].tolist(),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # --- Markdown report ----------------------------------------------------
    md = [f"# Transport delay analysis report\n",
          f"Data source: `{source}`  \nRows after cleaning: {len(df):,}  \n"
          f"Target: `{target}` (minutes)  \n"
          f"Delay threshold for probabilities: ≥ {C.DELAY_THRESHOLD_MIN:g} min\n"]
    if tests:
        md.append("## Do city events increase delay probability?\n")
        md.append(
            f"- Delay probability with an event: **{tests['delay_prob_event']:.1%}**, "
            f"without: **{tests['delay_prob_no_event']:.1%}** "
            f"(relative risk {tests['relative_risk']:.2f})\n"
            f"- Chi-square test on delay probability: p = {tests.get('chi2_p', float('nan')):.3g}\n"
            f"- Welch t-test on mean delay: p = {tests['welch_p']:.3g}\n")
    for title, table in sections:
        md += [f"## {title}\n", _md_table(table), ""]
    md += ["## Model comparison (time-based split)\n",
           "Trained on the earliest "
           f"{1 - test_size:.0%} of trips, tested on the latest {test_size:.0%}. "
           "`cv_*` columns are 5-fold `TimeSeriesSplit` scores on the training "
           "period; the best model is chosen on `cv_MAE`.\n",
           _md_table(result.metrics), "",
           f"**Best model: {result.best_name}**\n",
           *(_no_signal_note(result.metrics, result.best_name)),
           "## Top features (permutation importance on the test set)\n",
           _md_table(result.feature_importance.head(15).set_index("feature"), ".4f"), "",
           "## Figures\n"]
    md += [f"![{p.stem}](figures/{p.name})" for p in figs]
    (report_dir / "REPORT.md").write_text("\n".join(md) + "\n")
    log.info("Report written to %s", report_dir / "REPORT.md")
    return summary


def run_from_path(path, **kw) -> dict:
    return run(load_raw(path), source=str(path), **kw)
