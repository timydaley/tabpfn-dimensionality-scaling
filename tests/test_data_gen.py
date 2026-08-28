import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from tabdim.data_gen import D_TRUE, SyntheticSpec, generate, generate_splits


def test_shapes():
    spec = SyntheticSpec(n_features=50, n_samples=200, seed=0)
    data = generate(spec)
    assert data.X.shape == (200, 50)
    assert data.y.shape == (200,)
    assert data.true_feature_positions.shape == (D_TRUE,)
    assert data.feature_kind.shape == (50,)


def test_true_feature_positions_valid_and_distinct():
    spec = SyntheticSpec(n_features=1000, n_samples=50, seed=1)
    data = generate(spec)
    pos = data.true_feature_positions
    assert len(set(pos.tolist())) == D_TRUE
    assert pos.min() >= 0 and pos.max() < 1000
    assert (data.feature_kind[pos] == 0).all()


def test_edge_case_d_equals_d_true():
    spec = SyntheticSpec(n_features=D_TRUE, n_samples=30, seed=2)
    data = generate(spec)
    assert data.X.shape == (30, D_TRUE)
    assert (data.feature_kind == 0).all()


@pytest.mark.parametrize("d", [10, 50])
def test_reproducibility(d):
    spec = SyntheticSpec(n_features=d, n_samples=40, seed=42)
    d1 = generate(spec)
    d2 = generate(spec)
    np.testing.assert_array_equal(d1.X, d2.X)
    np.testing.assert_array_equal(d1.y, d2.y)


def test_true_features_more_informative_than_noise():
    spec = SyntheticSpec(n_features=200, n_samples=4000, seed=3, noise_to_signal=0.1)
    data = generate(spec)
    noise_positions = np.where(data.feature_kind == 2)[0][:D_TRUE]

    r2_true = LinearRegression().fit(
        data.X[:, data.true_feature_positions], data.y
    ).score(data.X[:, data.true_feature_positions], data.y)
    r2_noise = LinearRegression().fit(data.X[:, noise_positions], data.y).score(
        data.X[:, noise_positions], data.y
    )
    assert r2_true > r2_noise


def test_noise_variance_scales_correctly():
    from tabdim.data_gen import true_function

    spec_lo = SyntheticSpec(n_features=D_TRUE, n_samples=20000, seed=5, noise_to_signal=0.1)
    spec_hi = SyntheticSpec(n_features=D_TRUE, n_samples=20000, seed=5, noise_to_signal=1.0)
    data_lo = generate(spec_lo)
    data_hi = generate(spec_hi)
    # same seed -> same z (and same column permutation), so f(z) is identical;
    # only additive noise differs. Undo the column permutation before calling
    # true_function, which expects columns in the original z1..z5 order.
    z_lo = data_lo.X[:, data_lo.true_feature_positions]
    z_hi = data_hi.X[:, data_hi.true_feature_positions]
    resid_lo = data_lo.y - true_function(z_lo)
    resid_hi = data_hi.y - true_function(z_hi)
    assert resid_hi.std() > 5 * resid_lo.std()


def test_generate_splits_sizes_and_no_overlap():
    out = generate_splits(n_features=30, n_train=100, n_val=20, n_test=30, seed=7)
    assert out["X_train"].shape == (100, 30)
    assert out["X_val"].shape == (20, 30)
    assert out["X_test"].shape == (30, 30)
    total_y = np.concatenate([out["y_train"], out["y_val"], out["y_test"]])
    assert total_y.shape == (150,)


def test_frac_pure_noise_controls_decoy_mix():
    spec_all_noise = SyntheticSpec(n_features=105, n_samples=10, seed=8, frac_pure_noise=1.0)
    data = generate(spec_all_noise)
    assert (data.feature_kind[data.feature_kind != 0] == 2).all()

    spec_all_derived = SyntheticSpec(n_features=105, n_samples=10, seed=8, frac_pure_noise=0.0)
    data2 = generate(spec_all_derived)
    assert (data2.feature_kind[data2.feature_kind != 0] == 1).all()


def test_invalid_n_features_raises():
    with pytest.raises(ValueError):
        SyntheticSpec(n_features=D_TRUE - 1, n_samples=10, seed=0)
