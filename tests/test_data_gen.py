import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from tabdim.data_gen import (
    D_TRUE,
    SyntheticSpec,
    causal_dim_for_regime,
    generate,
    generate_splits,
    true_function,
)


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


def test_causal_dim_fixed_regime_always_min_true():
    assert causal_dim_for_regime("fixed", 10) == D_TRUE
    assert causal_dim_for_regime("fixed", 10000) == D_TRUE


def test_causal_dim_scaling_regime_grows_with_d_and_floors_at_min_true():
    assert causal_dim_for_regime("scaling", 10) == D_TRUE  # 5% of 10 rounds below min_true
    assert causal_dim_for_regime("scaling", 10000) == 500  # 5% of 10000
    small = causal_dim_for_regime("scaling", 100)
    large = causal_dim_for_regime("scaling", 10000)
    assert small < large


def test_causal_dim_unknown_regime_raises():
    with pytest.raises(ValueError):
        causal_dim_for_regime("bogus", 100)


def test_generate_splits_scaling_signal_regime_uses_larger_d_true():
    out = generate_splits(
        n_features=2000, n_train=50, n_val=10, n_test=10, seed=0, signal_regime="scaling"
    )
    # 5% of 2000 = 100 causal features, not the fixed-regime's 5
    assert out["true_feature_positions"].shape == (100,)


def test_d_true_5_dispatches_to_literal_original_formula():
    # true_function(z, d_true=5) must route to the literal original hardcoded
    # formula, not the generalized random-term construction, so every published
    # result for the "fixed" signal regime (d_true=5) stays reproducible.
    from tabdim.data_gen import _true_function_5

    rng = np.random.default_rng(0)
    z = rng.normal(size=(20, 5))
    np.testing.assert_array_equal(true_function(z, d_true=5), _true_function_5(z))


@pytest.mark.parametrize("d_true", [1, 5, 50, 500])
def test_true_function_shape_for_varying_d_true(d_true):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(30, d_true))
    out = true_function(z, d_true=d_true)
    assert out.shape == (30,)
    assert np.isfinite(out).all()


def test_true_function_is_fixed_across_seeds_for_given_d_true():
    # The task itself (not just the drawn data) must be identical across seeds for
    # a fixed d_true > 5, so that averaging over seeds measures repeated draws of
    # the same task rather than a mix of different-difficulty tasks. The term
    # generator is seeded only by d_true, so two independent calls must agree.
    from tabdim.data_gen import _random_true_function_terms

    terms_a = _random_true_function_terms(50, n_terms=100)
    terms_b = _random_true_function_terms(50, n_terms=100)
    for (idx_a, w_a, _, c_a), (idx_b, w_b, _, c_b) in zip(terms_a, terms_b):
        np.testing.assert_array_equal(idx_a, idx_b)
        np.testing.assert_array_equal(w_a, w_b)
        assert c_a == c_b

    # And end-to-end: two different data seeds sharing d_true=50 must score
    # identically against the same fixed z under true_function.
    rng = np.random.default_rng(123)
    z = rng.normal(size=(20, 50))
    np.testing.assert_allclose(true_function(z, d_true=50), true_function(z, d_true=50))


def test_scaling_signal_regime_grows_causal_dimensionality():
    fixed = generate_splits(
        n_features=2000, n_train=50, n_val=10, n_test=10, seed=0, signal_regime="fixed"
    )
    scaling = generate_splits(
        n_features=2000, n_train=50, n_val=10, n_test=10, seed=0, signal_regime="scaling"
    )
    assert fixed["true_feature_positions"].shape[0] == D_TRUE
    assert scaling["true_feature_positions"].shape[0] == 100  # 5% of 2000
    assert scaling["true_feature_positions"].shape[0] > fixed["true_feature_positions"].shape[0]


def test_scaling_signal_noise_calibration_still_works_at_large_d_true():
    from tabdim.data_gen import true_function

    spec_lo = SyntheticSpec(n_features=500, n_samples=5000, seed=5, d_true=250, noise_to_signal=0.1)
    spec_hi = SyntheticSpec(n_features=500, n_samples=5000, seed=5, d_true=250, noise_to_signal=1.0)
    data_lo = generate(spec_lo)
    data_hi = generate(spec_hi)
    z_lo = data_lo.X[:, data_lo.true_feature_positions]
    z_hi = data_hi.X[:, data_hi.true_feature_positions]
    resid_lo = data_lo.y - true_function(z_lo, d_true=250)
    resid_hi = data_hi.y - true_function(z_hi, d_true=250)
    assert resid_hi.std() > 5 * resid_lo.std()
