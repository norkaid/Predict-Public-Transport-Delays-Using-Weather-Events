# Predict Public Transport Delays Using Weather & Events

Predict how late a bus, tram, metro or train will arrive from external factors —
weather conditions, city events, traffic congestion and time of day — and measure
how much each factor matters.

Dataset: [Public Transport Delays with Weather and Events (Kaggle)](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events)

## What the project does

| Step | Module | Details |
|---|---|---|
| Data cleaning | `src/transport_delays/data.py` | Normalises column names, removes duplicates, builds a trip timestamp from date + time, coerces types, clips physically impossible values (e.g. humidity > 100 %), imputes missing values (median / `"Unknown"`), treats a missing event as `"None"`. |
| Feature engineering | `src/transport_delays/features.py` | Time (hour, weekday, month, cyclical sin/cos, time-of-day band, weekend), weather flags (rain, snow, fog, storm, freezing, hot, heavy precipitation, high wind, adverse-weather score, distance from 15 °C), events (has event, log attendance, large event, event × peak), interactions with congestion, planned trip duration. |
| Time-based analysis | `src/transport_delays/analysis.py` | Delay by weather, temperature band, precipitation band, event type, hour, weekday and month; Welch t-test and chi-square test for the event effect; Spearman correlations; charts. |
| Regression models | `src/transport_delays/models.py` | Baseline (mean), Linear Regression (Ridge), Random Forest, XGBoost. Chronological train/test split + `TimeSeriesSplit` cross-validation; model chosen on CV MAE; permutation importance on the test period. |
| Report | `src/transport_delays/pipeline.py` | Writes `reports/REPORT.md`, CSV tables, figures and `models/best_model.joblib`. |

### Modelling choices worth knowing

- **Target:** `actual_arrival_delay_min` (change with `--target`).
- **No leakage:** `actual_departure_delay_min` and the `delayed` flag are excluded
  from the features — they are only known once the trip is under way.
- **Time-based split:** the model trains on the earliest 80 % of trips and is
  tested on the latest 20 %, so it never learns from the future.
- **High-cardinality IDs** (route, origin, destination) use cross-fitted target
  encoding inside the pipeline, so encodings are learnt from training folds only.
- **"Delay probability"** = share of trips arriving ≥ 5 minutes late
  (`DELAY_THRESHOLD_MIN` in `config.py`).

## Results on the Kaggle dataset

Run on all 2,000 trips (none dropped in cleaning). Full tables and significance
tests are in [`reports/REPORT.md`](reports/REPORT.md), charts in
[`reports/figures/`](reports/figures/).

**Weather, events and time of day do not explain arrival delays in this dataset.**

| Question | What the data shows |
|---|---|
| Does bad weather increase delays? | No. Mean delay is 12.9–13.6 min for every condition (Rain 13.6, Snow 13.5, Clear 13.5, Storm 13.3, Cloudy 13.1, Fog 12.9). Share of trips ≥ 5 min late is 75–80 % throughout; clear weather is actually the highest (80 %). |
| Do extreme temperatures matter? | No. Mean delay is 12.6–14.0 min in every temperature band from ≤ 0 °C to > 30 °C, with no U-shape. |
| Does rain or snow volume matter? | No. Mean delay ranges from 13.0 to 13.7 min between no precipitation and heavy precipitation (> 10 mm), with no upward trend. |
| Do city events increase delay probability? | Not meaningfully. 78 % of trips without an event are delayed ≥ 5 min, versus 75–81 % across event types (Protest 81 %, Parade 81 %, Festival 78 %, Sports 77 %, Concert 75 %). See the chi-square test in the report. |
| Is there a rush-hour pattern? | No consistent one. Hourly mean delay fluctuates between roughly 9 and 16 min, with no morning or evening peak. |

**Models (trained on the earliest 80 % of trips, tested on the latest 20 %):**

| Model | Test MAE (min) | Test RMSE (min) | Test R² |
|---|---|---|---|
| Baseline: always predict the mean | **8.15** | **9.46** | −0.004 |
| Linear Regression (Ridge) | 8.21 | 9.51 | −0.016 |
| Random Forest | 8.28 | 9.59 | −0.032 |
| XGBoost | 8.56 | 9.91 | −0.101 |

No model beats the baseline. The more flexible the model, the worse it does
(XGBoost is last), which is what happens when a model fits noise instead of
signal.

The other charts say the same thing:

- **Correlations** (`06_correlations.png`): no feature has a Spearman
  correlation with arrival delay above |ρ| = 0.05. The strongest is event
  attendance at ρ ≈ 0.05, which explains well under 1 % of the variation.
- **Feature importance** (`08_feature_importance.png`): shuffling any single
  feature raises the best model's error by at most 0.03 minutes, against a
  typical error of 8 minutes.
- **Predicted vs actual** (`09_predicted_vs_actual.png`): the model predicts
  roughly 11–15 minutes for every trip, whether the trip was 3 minutes early or
  29 minutes late. It has learnt nothing beyond the average.

**Interpretation.** Arrival delays are spread almost evenly between about −3 and
29 minutes (see `01_delay_distribution.png`) and are unrelated to every
recorded factor. This is what randomly generated data looks like, and the
dataset is most likely synthetic. The pipeline itself is not the cause: on
generated data with real weather and event effects built in (`--synthetic`),
the same code reaches R² ≈ 0.8 and recovers those effects. The approach would
work on real operational data, where weather and events do affect delays.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset into `data/` (needs a Kaggle API token):

```bash
kaggle datasets download -d khushikyad001/public-transport-delays-with-weather-and-events -p data --unzip
```

## Run

```bash
python run_pipeline.py
```

Options: `--data path.csv`, `--target <column>`, `--test-size 0.2`,
`--report-dir`, `--model-dir`. To try the pipeline without the Kaggle file:

```bash
python run_pipeline.py --synthetic
```

`--synthetic` uses generated data with the same columns and invented effect
sizes. Use it only to check that the code runs: its results say nothing about
real transit systems.

Predict for new trips with the saved model:

```bash
python predict.py new_trips.csv --out predictions.csv
```

## Outputs

```
reports/
  REPORT.md                 analysis tables, event significance tests, model comparison, top features
  model_metrics.csv         CV and test MAE / RMSE / R² per model
  feature_importance.csv    permutation importance of the best model
  test_predictions.csv      actual vs predicted for every model on the test period
  data_quality_raw.csv      dtype, missing %, unique count per raw column
  summary.json
  figures/                  delay distribution, weather, temperature & precipitation,
                            events, delay by hour, correlations, model comparison,
                            feature importance, predicted vs actual
models/best_model.joblib
```

## Tests

```bash
python -m pytest -q
```

Tests run on synthetic data and check cleaning, feature construction, that no
leaky column reaches the model, that the split is chronological, and that the
trained models beat the mean baseline.

## Project layout

```
├── run_pipeline.py          # end-to-end CLI
├── predict.py               # score new trips with the saved model
├── src/transport_delays/
│   ├── config.py            # column names, thresholds, paths
│   ├── data.py              # loading & cleaning
│   ├── features.py          # feature engineering
│   ├── analysis.py          # EDA tables, significance tests, figures
│   ├── models.py            # training, evaluation, importance
│   ├── pipeline.py          # orchestration & report
│   └── synthetic.py         # schema-compatible demo data
├── tests/
└── data/                    # put the Kaggle CSV here (git-ignored)
```
