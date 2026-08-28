"""Orchestrates the dimensionality sweep: for each (model, N-regime, D, seed) cell,
generate a fresh synthetic split, fit, predict, and score on held-out test data."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from .data_gen import generate_splits
from .models import get_model

FIXED_N_TRAIN = 2000
SCALING_FACTOR = 4
SCALING_N_MIN = 200
# Capped at 2000 rather than higher: on this CPU-only 34GB machine, TabICL at
# D=2000/N_train=2000 was the largest single-chunk size we could run without an
# OOM kill (see models.py). FIXED_N_TRAIN=2000 is exactly at that boundary too.
SCALING_N_MAX = 2000


@dataclass
class RunResult:
    model: str
    regime: str
    signal_regime: str
    n_features: int
    n_train: int
    seed: int
    rmse: float
    r2: float
    fit_predict_seconds: float
    error: str = ""


def n_train_for_regime(regime: str, d: int) -> int:
    if regime == "fixed_n":
        return FIXED_N_TRAIN
    if regime == "scaling_n":
        return int(np.clip(SCALING_FACTOR * d, SCALING_N_MIN, SCALING_N_MAX))
    raise ValueError(f"Unknown regime {regime!r}")


def run_one(
    model_name: str,
    regime: str,
    d: int,
    seed: int,
    signal_regime: str = "fixed",
    n_val: int = 500,
    n_test: int = 1000,
    device: str = "cpu",
    **spec_kwargs,
) -> RunResult:
    n_train = n_train_for_regime(regime, d)
    splits = generate_splits(
        n_features=d,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        signal_regime=signal_regime,
        **spec_kwargs,
    )
    model = get_model(model_name, seed=seed, device=device)

    t0 = time.time()
    model.fit(splits["X_train"], splits["y_train"])
    preds = model.predict(splits["X_test"])
    elapsed = time.time() - t0

    return RunResult(
        model=model_name,
        regime=regime,
        signal_regime=signal_regime,
        n_features=d,
        n_train=n_train,
        seed=seed,
        rmse=float(root_mean_squared_error(splits["y_test"], preds)),
        r2=float(r2_score(splits["y_test"], preds)),
        fit_predict_seconds=elapsed,
    )


def run_sweep(
    models: list[str],
    regimes: list[str],
    dims: list[int],
    seeds: list[int],
    signal_regimes: list[str] = ("fixed",),
    device: str = "cpu",
    out_csv: str | Path | None = None,
    **spec_kwargs,
) -> pd.DataFrame:
    """Runs every (model, regime, signal_regime, D, seed) cell, writing results
    incrementally to out_csv (if given) so a long background run stays safe
    against crashes and lets progress be inspected mid-run."""
    rows = []
    cells = [
        (model_name, regime, signal_regime, d, seed)
        for model_name in models
        for regime in regimes
        for signal_regime in signal_regimes
        for d in dims
        for seed in seeds
    ]
    for i, (model_name, regime, signal_regime, d, seed) in enumerate(cells, start=1):
        try:
            result = run_one(
                model_name, regime, d, seed, signal_regime=signal_regime, device=device, **spec_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - a single failed cell shouldn't kill the sweep
            result = RunResult(
                model=model_name,
                regime=regime,
                signal_regime=signal_regime,
                n_features=d,
                n_train=n_train_for_regime(regime, d),
                seed=seed,
                rmse=float("nan"),
                r2=float("nan"),
                fit_predict_seconds=float("nan"),
                error=repr(exc),
            )
        rows.append(asdict(result))
        print(
            f"[{i}/{len(cells)}] {model_name} {regime} signal={signal_regime} D={d} seed={seed} "
            f"r2={result.r2:.3f} ({result.fit_predict_seconds:.1f}s)"
            + (f" ERROR: {result.error}" if result.error else ""),
            flush=True,
        )
        if out_csv is not None:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)
