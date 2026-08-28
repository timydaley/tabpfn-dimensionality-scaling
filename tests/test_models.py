import numpy as np
import pytest

from tabdim.models import (
    FeatureBaggedRegressor,
    make_random_forest,
    make_ridge,
    make_xgboost,
)


def _toy_data(n=200, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] + 2 * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize("factory", [make_ridge, make_random_forest, make_xgboost])
def test_baseline_models_fit_predict_shape(factory):
    X, y = _toy_data()
    model = factory(seed=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


@pytest.mark.parametrize("factory", [make_ridge, make_random_forest, make_xgboost])
def test_baseline_models_fit_low_error_on_easy_signal(factory):
    X, y = _toy_data(n=500, seed=1)
    model = factory(seed=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.corrcoef(preds, y)[0, 1] > 0.8


def test_feature_bagged_regressor_single_chunk_when_under_cap():
    X, y = _toy_data(d=10)
    ens = FeatureBaggedRegressor(make_ridge, max_features=50, seed=0)
    ens.fit(X, y)
    assert len(ens._chunks) == 1
    assert len(ens._chunks[0]) == 10


def test_feature_bagged_regressor_chunks_when_over_cap():
    X, y = _toy_data(d=1000, seed=2)
    ens = FeatureBaggedRegressor(make_ridge, max_features=300, seed=0)
    ens.fit(X, y)
    assert len(ens._chunks) == 4  # ceil(1000/300)
    all_indices = np.sort(np.concatenate(ens._chunks))
    np.testing.assert_array_equal(all_indices, np.arange(1000))


def test_feature_bagged_regressor_predict_shape_and_reasonable_signal():
    X, y = _toy_data(n=300, d=800, seed=3)
    ens = FeatureBaggedRegressor(make_ridge, max_features=200, seed=0)
    ens.fit(X, y)
    preds = ens.predict(X)
    assert preds.shape == y.shape
    # true signal lives in columns 0 and 1, which land in exactly one chunk;
    # the ensemble average should still correlate with y.
    assert np.corrcoef(preds, y)[0, 1] > 0.1


def test_feature_bagged_regressor_reproducible_chunking():
    X, y = _toy_data(d=500, seed=4)
    ens1 = FeatureBaggedRegressor(make_ridge, max_features=150, seed=7)
    ens2 = FeatureBaggedRegressor(make_ridge, max_features=150, seed=7)
    ens1.fit(X, y)
    ens2.fit(X, y)
    for c1, c2 in zip(ens1._chunks, ens2._chunks):
        np.testing.assert_array_equal(c1, c2)
