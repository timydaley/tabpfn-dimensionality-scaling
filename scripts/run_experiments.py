#!/usr/bin/env python
"""CLI entry point for the dimensionality-scaling sweep.

Example:
    python scripts/run_experiments.py --out results/sweep.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabdim.experiment import run_sweep  # noqa: E402

DEFAULT_DIMS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
DEFAULT_MODELS = ["ridge", "random_forest", "xgboost", "tabpfn", "tabicl", "nori"]
DEFAULT_REGIMES = ["fixed_n", "scaling_n"]
DEFAULT_SIGNAL_REGIMES = ["fixed"]
DEFAULT_SEEDS = list(range(5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--regimes", nargs="+", default=DEFAULT_REGIMES)
    parser.add_argument(
        "--signal-regimes",
        nargs="+",
        default=DEFAULT_SIGNAL_REGIMES,
        help="'fixed': d_true=5 regardless of D. 'scaling': d_true grows with D.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/sweep.csv")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = run_sweep(
        models=args.models,
        regimes=args.regimes,
        signal_regimes=args.signal_regimes,
        dims=args.dims,
        seeds=args.seeds,
        device=args.device,
        out_csv=args.out,
    )
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
