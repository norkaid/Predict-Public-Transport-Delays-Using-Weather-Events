"""Regression models for arrival delay, evaluated with a time-based split."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from . import config as C
from .features import split_feature_types

log = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None


def time_split(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological split: the most recent ``test_size`` share is the test set.

    A random split would let the model see the future (e.g. the same storm
    day in train and test), overstating accuracy.
    """
    df = df.sort_values("timestamp")
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]


def _target_encoder() -> TargetEncoder:
    """Cross-fitted target encoder; scikit-learn >= 1.9 takes the CV splitter."""
    if getattr(TargetEncoder(), "shuffle", None) == "deprecated":
        return TargetEncoder(target_type="continuous",
                             cv=KFold(5, shuffle=True, random_state=C.RANDOM_STATE))
    return TargetEncoder(target_type="continuous", random_state=C.RANDOM_STATE)


def make_preprocessor(numeric, low_cat, high_cat, scale: bool) -> ColumnTransformer:
    transformers = [
        ("num", StandardScaler() if scale else "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_cat),
    ]
    if high_cat:
        transformers.append(
            ("te", _target_encoder(), high_cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def candidate_models(numeric, low_cat, high_cat) -> dict[str, Pipeline]:
    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression (Ridge)": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, max_features=0.5,
            n_jobs=-1, random_state=C.RANDOM_STATE),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
            random_state=C.RANDOM_STATE, n_jobs=-1)
    return {
        name: Pipeline([
            ("prep", make_preprocessor(numeric, low_cat, high_cat,
                                       scale=name.startswith("Linear"))),
            ("model", est),
        ])
        for name, est in models.items()
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


@dataclass
class TrainingResult:
    metrics: pd.DataFrame
    best_name: str
    best_model: Pipeline
    models: dict[str, Pipeline]
    feature_importance: pd.DataFrame
    test_predictions: pd.DataFrame
    features: dict[str, list[str]] = field(default_factory=dict)


def train_and_evaluate(df: pd.DataFrame, target: str = C.TARGET,
                       test_size: float = 0.2, cv_splits: int = 5,
                       importance_repeats: int = 5) -> TrainingResult:
    numeric, low_cat, high_cat = split_feature_types(df, target)
    features = numeric + low_cat + high_cat
    log.info("Using %d numeric, %d categorical, %d target-encoded features",
             len(numeric), len(low_cat), len(high_cat))

    train, test = time_split(df, test_size)
    X_tr, y_tr = train[features], train[target]
    X_te, y_te = test[features], test[target]

    rows, fitted = [], {}
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for name, pipe in candidate_models(numeric, low_cat, high_cat).items():
        cv = cross_validate(pipe, X_tr, y_tr, cv=tscv, n_jobs=1,
                            scoring=("neg_mean_absolute_error", "r2"))
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        m = regression_metrics(y_te, pipe.predict(X_te))
        rows.append({
            "model": name,
            "cv_MAE": -cv["test_neg_mean_absolute_error"].mean(),
            "cv_MAE_std": cv["test_neg_mean_absolute_error"].std(),
            "cv_R2": cv["test_r2"].mean(),
            **{f"test_{k}": v for k, v in m.items()},
        })
        log.info("%-28s test MAE=%.2f RMSE=%.2f R2=%.3f",
                 name, m["MAE"], m["RMSE"], m["R2"])

    metrics = pd.DataFrame(rows).set_index("model").round(4)
    # Pick the winner on cross-validation, not on the test set, so the test
    # score stays an honest estimate.
    best_name = metrics.drop(index="Baseline (mean)")["cv_MAE"].idxmin()
    best = fitted[best_name]

    pi = permutation_importance(best, X_te, y_te, n_repeats=importance_repeats,
                                scoring="neg_mean_absolute_error",
                                random_state=C.RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({
        "feature": features,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    preds = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "actual": y_te.values,
        **{name: m.predict(X_te) for name, m in fitted.items()},
    })
    return TrainingResult(metrics, best_name, best, fitted, importance, preds,
                          {"numeric": numeric, "low_cat": low_cat, "high_cat": high_cat})


def plot_results(result: TrainingResult, out_dir, top_n: int = 15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .analysis import BLUE, INK_2, GRID, SURFACE, _style

    paths = []
    # Model comparison (test MAE)
    m = result.metrics.sort_values("test_MAE")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(m.index, m["test_MAE"], color=BLUE, height=0.55)
    ax.invert_yaxis()
    for y, v in enumerate(m["test_MAE"]):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=9, color=INK_2)
    _style(ax, "Test MAE by model (lower is better)", "Mean absolute error (min)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    p = out_dir / "07_model_comparison.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # Permutation importance of the best model
    imp = result.feature_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"],
            color=BLUE, height=0.6, error_kw={"ecolor": INK_2, "linewidth": 1})
    _style(ax, f"Permutation importance - {result.best_name}",
           "Increase in MAE when shuffled (min)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    p = out_dir / "08_feature_importance.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # Predicted vs actual
    tp = result.test_predictions
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(tp["actual"], tp[result.best_name], s=10, alpha=0.5, color=BLUE,
               edgecolors="none")
    lo, hi = tp["actual"].min(), tp["actual"].max()
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
    _style(ax, f"Predicted vs actual - {result.best_name}",
           "Actual delay (min)", "Predicted delay (min)")
    fig.tight_layout()
    p = out_dir / "09_predicted_vs_actual.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)
    return paths
