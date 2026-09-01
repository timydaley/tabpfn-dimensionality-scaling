#!/usr/bin/env python
"""Compares the fixed-signal (5 causal features regardless of D) and
scaling-signal (causal features grow with D) conditions side by side, both under
the fixed-N training regime, for the models run in both sweeps."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_LABELS = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "tabpfn": "TabPFN v2",
    "tabicl": "TabICL",
    "nori": "Nori",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-signal-results", default="results/sweep_no_tabpfn.csv")
    parser.add_argument("--scaling-signal-results", default="results/sweep_scaling_signal.csv")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_signal = pd.read_csv(args.fixed_signal_results)
    fixed_signal = fixed_signal[fixed_signal["regime"] == "fixed_n"]
    scaling_signal = pd.read_csv(args.scaling_signal_results)
    scaling_signal = scaling_signal[scaling_signal["error"].fillna("") == ""]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        (axes[0], fixed_signal, "5 causal features regardless of D\n(ambient dimensionality only)"),
        (axes[1], scaling_signal, "Causal features scale with D (5% of D)\n(signal dimensionality grows too)"),
    ]
    for ax, df, title in panels:
        for model, g in df.groupby("model"):
            agg = g.groupby("n_features")["r2"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["n_features"], agg["mean"], yerr=agg["std"],
                marker="o", capsize=3, label=MODEL_LABELS.get(model, model),
            )
        ax.set_xscale("log")
        ax.set_xlabel("Number of features (D)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("R2")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "signal_regime_comparison.pdf")
    plt.close(fig)

    summary = pd.concat(
        [
            fixed_signal.assign(signal_regime="fixed"),
            scaling_signal.assign(signal_regime="scaling"),
        ]
    ).groupby(["signal_regime", "model", "n_features"])["r2"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(out_dir / "signal_regime_comparison_summary.csv", index=False)

    print(f"Wrote figures/signal_regime_comparison.pdf and summary CSV to {out_dir}/")


if __name__ == "__main__":
    main()
