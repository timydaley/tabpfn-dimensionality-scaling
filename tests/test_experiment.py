import pandas as pd
import pytest

from tabdim.experiment import (
    FIXED_N_TRAIN,
    SCALING_N_MAX,
    SCALING_N_MIN,
    n_train_for_regime,
    run_one,
    run_sweep,
)


def test_n_train_fixed_regime_constant_across_d():
    assert n_train_for_regime("fixed_n", 10) == FIXED_N_TRAIN
    assert n_train_for_regime("fixed_n", 10000) == FIXED_N_TRAIN


def test_n_train_scaling_regime_grows_with_d_and_is_clamped():
    small = n_train_for_regime("scaling_n", 10)
    mid = n_train_for_regime("scaling_n", 200)
    large = n_train_for_regime("scaling_n", 100000)
    assert small == SCALING_N_MIN
    assert small < mid < large
    assert large == SCALING_N_MAX


def test_n_train_unknown_regime_raises():
    with pytest.raises(ValueError):
        n_train_for_regime("bogus", 10)


def test_run_one_ridge_end_to_end():
    result = run_one("ridge", "fixed_n", d=20, seed=0, n_val=50, n_test=100)
    assert result.model == "ridge"
    assert result.n_features == 20
    assert result.n_train == FIXED_N_TRAIN
    assert result.rmse > 0
    assert result.error == ""


def test_run_sweep_writes_all_cells(tmp_path):
    out_csv = tmp_path / "sweep.csv"
    df = run_sweep(
        models=["ridge", "random_forest"],
        regimes=["fixed_n", "scaling_n"],
        dims=[10, 30],
        seeds=[0, 1],
        n_val=30,
        n_test=50,
        out_csv=out_csv,
    )
    assert len(df) == 2 * 2 * 2 * 2  # models x regimes x dims x seeds
    assert out_csv.exists()
    on_disk = pd.read_csv(out_csv)
    assert len(on_disk) == len(df)
    assert (df["error"] == "").all()


def test_run_sweep_records_error_without_killing_other_cells(monkeypatch):
    import tabdim.experiment as experiment_mod

    original = experiment_mod.run_one

    def flaky_run_one(model_name, regime, d, seed, **kwargs):
        if model_name == "ridge" and seed == 0:
            raise RuntimeError("boom")
        return original(model_name, regime, d, seed, **kwargs)

    monkeypatch.setattr(experiment_mod, "run_one", flaky_run_one)
    df = run_sweep(
        models=["ridge"], regimes=["fixed_n"], dims=[10], seeds=[0, 1], n_val=20, n_test=20
    )
    assert len(df) == 2
    failed = df[df["seed"] == 0].iloc[0]
    assert "boom" in failed["error"]
    ok = df[df["seed"] == 1].iloc[0]
    assert ok["error"] == ""
