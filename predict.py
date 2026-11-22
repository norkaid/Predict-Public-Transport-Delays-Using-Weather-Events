#!/usr/bin/env python
"""Predict arrival delay for new trips with the saved model.

    python predict.py trips.csv --out predictions.csv
"""

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transport_delays import config as C  # noqa: E402
from transport_delays.data import load_raw  # noqa: E402
from transport_delays.features import build_features  # noqa: E402
from transport_delays.data import clean  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--model", type=Path, default=C.DEFAULT_MODEL_DIR / "best_model.joblib")
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    args = ap.parse_args(argv)

    bundle = joblib.load(args.model)
    raw = load_raw(args.csv)
    target = bundle["target"]
    if target not in raw:
        raw[target] = 0.0  # placeholder so cleaning keeps every row
    df = build_features(clean(raw, target=target))
    cols = sum(bundle["features"].values(), [])
    df["predicted_delay_min"] = bundle["model"].predict(df[cols])
    keep = [c for c in (C.ID_COL, "timestamp", "predicted_delay_min") if c in df]
    df[keep].to_csv(args.out, index=False)
    print(f"Wrote {len(df)} predictions ({bundle['name']}) to {args.out}")


if __name__ == "__main__":
    main()
