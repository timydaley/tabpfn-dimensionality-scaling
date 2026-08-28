#!/usr/bin/env python
"""Turns results/sweep.csv into the figures and summary tables used in the paper."""

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
}
REGIME_LABELS = {
    "fixed_n": "Fixed training set size",
    "scaling_n": "Training set size scales with D",
}


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "error" in df.columns:
        n_failed = df["error"].fillna("").ne("").sum()
        if n_failed:
            print(f"Warning: {n_failed}/{len(df)} sweep cells errored and are excluded.")
        df = df[df["error"].fillna("") == ""]
    return df


def plot_metric_vs_dimensionality(df: pd.DataFrame, metric: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, regime in zip(axes, ["fixed_n", "scaling_n"]):
        sub = df[df["regime"] == regime]
        for model, g in sub.groupby("model"):
            agg = g.groupby("n_features")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["n_features"], agg["mean"], yerr=agg["std"],
                marker="o", capsize=3, label=MODEL_LABELS.get(model, model),
            )
        ax.set_xscale("log")
        ax.set_xlabel("Number of features (D)")
        ax.set_title(REGIME_LABELS[regime])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric.upper())
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_vs_dimensionality.pdf")
    plt.close(fig)


def make_summary_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby(["regime", "model", "n_features"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["regime", "model", "n_features"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/sweep.csv")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)
    for metric in ["r2", "rmse"]:
        plot_metric_vs_dimensionality(df, metric, out_dir)
        make_summary_table(df, metric).to_csv(out_dir / f"{metric}_summary.csv", index=False)

    print(f"Wrote figures and summary tables to {out_dir}/")


if __name__ == "__main__":
    main()
