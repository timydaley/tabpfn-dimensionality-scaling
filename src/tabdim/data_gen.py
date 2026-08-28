"""Synthetic tabular regression data with a low-dimensional nonlinear true model.

The generative process has three feature kinds, concatenated and then column-shuffled:

- "true": d_true latent features that causally determine y through a fixed nonlinear
  function.
- "derived": noisy nonlinear functions of a random subset of the true features
  (correlated with y, but not perfectly recoverable).
- "noise": iid features carrying no information about y.

Two signal regimes control d_true as a function of the ambient feature count D:
"fixed" holds d_true=5 regardless of D (isolates ambient dimensionality from signal
dimensionality); "scaling" grows d_true with D (see causal_dim_for_regime), so more
features means more true causal structure, not just more decoys.
"""

from dataclasses import dataclass

import numpy as np

D_TRUE = 5

_TRANSFORMS = [
    lambda a: a,
    lambda a: a ** 2,
    lambda a: np.sin(a),
    lambda a: np.tanh(a),
]


def causal_dim_for_regime(signal_regime: str, d: int, min_true: int = D_TRUE, fraction: float = 0.05) -> int:
    """Number of truly causal features for a given ambient feature count D.

    "fixed": always min_true (5), regardless of D -- isolates ambient dimensionality
    from signal dimensionality.
    "scaling": max(min_true, round(fraction * D)) -- signal complexity grows with D,
    e.g. 5% of features are causal, reaching d_true=500 at D=10000 on our default grid.
    """
    if signal_regime == "fixed":
        return min_true
    if signal_regime == "scaling":
        return max(min_true, round(fraction * d))
    raise ValueError(f"Unknown signal_regime {signal_regime!r}")


def _true_function_5(z: np.ndarray) -> np.ndarray:
    """The original hand-written 5-feature nonlinear function. Kept as a literal
    special case (rather than folded into _random_true_function) so that every
    result already published for d_true=5 (i.e. the "fixed" signal regime) is
    reproduced bit-for-bit rather than replaced by a differently-structured
    function."""
    z1, z2, z3, z4, z5 = (z[:, i] for i in range(5))
    return (
        3.0 * np.sin(z1 * z2)
        + 2.0 * z3 ** 2
        - 1.5 * z4 * z5
        + 1.0 * z1 * z4 ** 2
        - 0.5 * z2 ** 3
        + 2.0 * np.tanh(z1 + z3 - z4)
    )


def _random_true_function_terms(d_true: int, n_terms: int):
    """A fixed (not per-seed) random set of additive nonlinear terms over the
    d_true causal features. Seeded only by d_true, not by the data-draw seed, so
    every seed sharing a given d_true sees the identical underlying task -- the
    seed axis varies the drawn data, not the task itself, matching how
    _true_function_5 is already a single fixed function shared across all seeds."""
    rng = np.random.default_rng(1000 + d_true)
    terms = []
    for _ in range(n_terms):
        k = rng.integers(1, min(3, d_true) + 1)
        idx = rng.choice(d_true, size=k, replace=False)
        weights = rng.normal(size=k)
        transform = _TRANSFORMS[rng.integers(len(_TRANSFORMS))]
        coef = rng.normal()
        terms.append((idx, weights, transform, coef))
    return terms


def true_function(z: np.ndarray, d_true: int = D_TRUE) -> np.ndarray:
    """Nonlinear function of the d_true latent features. d_true=5 reproduces the
    original hand-written formula exactly; other values use a fixed (per-d_true,
    not per-seed) random sum of nonlinear terms over 1-3 causal features each,
    so that interaction terms exist for the feature-bagging ensemble to
    potentially sever across chunks."""
    if d_true == 5:
        return _true_function_5(z)
    terms = _random_true_function_terms(d_true, n_terms=2 * d_true)
    out = np.zeros(z.shape[0])
    for idx, weights, transform, coef in terms:
        out += coef * transform(z[:, idx] @ weights)
    return out


@dataclass
class SyntheticSpec:
    n_features: int
    n_samples: int
    seed: int
    d_true: int = D_TRUE
    frac_pure_noise: float = 0.5
    noise_to_signal: float = 0.2
    proxy_noise_scale: float = 0.5

    def __post_init__(self):
        if self.n_features < self.d_true:
            raise ValueError(f"n_features must be >= d_true ({self.d_true}), got {self.n_features}")
        if not 0.0 <= self.frac_pure_noise <= 1.0:
            raise ValueError("frac_pure_noise must be in [0, 1]")


@dataclass
class SyntheticDataset:
    X: np.ndarray
    y: np.ndarray
    true_feature_positions: np.ndarray
    feature_kind: np.ndarray  # aligned to X columns: 0=true, 1=derived, 2=pure_noise


def generate(spec: SyntheticSpec) -> SyntheticDataset:
    rng = np.random.default_rng(spec.seed)
    n, d, d_true = spec.n_samples, spec.n_features, spec.d_true

    z = rng.normal(size=(n, d_true))
    f = true_function(z, d_true=d_true)
    y = f + rng.normal(scale=spec.noise_to_signal * f.std() + 1e-12, size=n)

    n_decoy = d - d_true
    n_pure_noise = int(round(n_decoy * spec.frac_pure_noise))
    n_derived = n_decoy - n_pure_noise

    decoys = np.empty((n, n_decoy))
    decoy_kind = np.empty(n_decoy, dtype=int)
    for j in range(n_derived):
        k = rng.integers(1, min(3, d_true) + 1)
        idx = rng.choice(d_true, size=k, replace=False)
        weights = rng.normal(size=k)
        base = z[:, idx] @ weights
        transform = _TRANSFORMS[rng.integers(len(_TRANSFORMS))]
        val = transform(base)
        val = (val - val.mean()) / (val.std() + 1e-8)
        decoys[:, j] = val + rng.normal(scale=spec.proxy_noise_scale, size=n)
        decoy_kind[j] = 1
    for j in range(n_derived, n_decoy):
        decoys[:, j] = rng.normal(size=n)
        decoy_kind[j] = 2

    x_unpermuted = np.concatenate([z, decoys], axis=1)
    kind_unpermuted = np.concatenate([np.zeros(d_true, dtype=int), decoy_kind])

    perm = rng.permutation(d)
    inv_perm = np.argsort(perm)
    x = x_unpermuted[:, perm]
    feature_kind = kind_unpermuted[perm]
    true_feature_positions = inv_perm[:d_true]

    return SyntheticDataset(
        X=x, y=y, true_feature_positions=true_feature_positions, feature_kind=feature_kind
    )


def generate_splits(
    n_features: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    signal_regime: str = "fixed",
    **spec_kwargs,
) -> dict:
    """Generate train/val/test splits from one i.i.d. draw of the process.

    Rows are independent draws, so slicing a single generated block into
    contiguous train/val/test segments is equivalent to drawing them separately.
    """
    d_true = causal_dim_for_regime(signal_regime, n_features)
    spec = SyntheticSpec(
        n_features=n_features,
        n_samples=n_train + n_val + n_test,
        seed=seed,
        d_true=d_true,
        **spec_kwargs,
    )
    data = generate(spec)
    i, j = n_train, n_train + n_val
    return {
        "X_train": data.X[:i],
        "y_train": data.y[:i],
        "X_val": data.X[i:j],
        "y_val": data.y[i:j],
        "X_test": data.X[j:],
        "y_test": data.y[j:],
        "true_feature_positions": data.true_feature_positions,
        "feature_kind": data.feature_kind,
    }
