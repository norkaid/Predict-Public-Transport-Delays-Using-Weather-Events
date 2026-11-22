#!/usr/bin/env python
"""Run the full delay-prediction pipeline.

    python run_pipeline.py                         # uses data/public_transport_delays.csv
    python run_pipeline.py --data path/to/file.csv
    python run_pipeline.py --synthetic             # demo on generated data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transport_delays import config as C  # noqa: E402
from transport_delays.pipeline import run, run_from_path  # noqa: E402
from transport_delays.synthetic import generate  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=C.DEFAULT_DATA_PATH)
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data with the dataset's schema")
    ap.add_argument("--rows", type=int, default=2000, help="rows for --synthetic")
    ap.add_argument("--report-dir", type=Path, default=C.DEFAULT_REPORT_DIR)
    ap.add_argument("--model-dir", type=Path, default=C.DEFAULT_MODEL_DIR)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    kw = dict(report_dir=args.report_dir, model_dir=args.model_dir,
              target=args.target, test_size=args.test_size)
    if args.synthetic:
        summary = run(generate(args.rows), source="synthetic", **kw)
    else:
        summary = run_from_path(args.data, **kw)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
