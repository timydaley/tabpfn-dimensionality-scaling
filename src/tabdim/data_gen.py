"""Synthetic tabular regression data with a low-dimensional nonlinear true model.

The generative process has three feature kinds, concatenated and then column-shuffled:

- "true": D_TRUE latent features that causally determine y through a fixed nonlinear
  function.
- "derived": noisy nonlinear functions of a random subset of the true features
  (correlated with y, but not perfectly recoverable).
- "noise": iid features carrying no information about y.

D_TRUE is fixed at 5 so the signal stays low-dimensional as the ambient feature
count D grows.
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


def true_function(z: np.ndarray) -> np.ndarray:
    """Fixed nonlinear function of the D_TRUE latent features."""
    z1, z2, z3, z4, z5 = (z[:, i] for i in range(D_TRUE))
    return (
        3.0 * np.sin(z1 * z2)
        + 2.0 * z3 ** 2
        - 1.5 * z4 * z5
        + 1.0 * z1 * z4 ** 2
        - 0.5 * z2 ** 3
        + 2.0 * np.tanh(z1 + z3 - z4)
    )


@dataclass
class SyntheticSpec:
    n_features: int
    n_samples: int
    seed: int
    frac_pure_noise: float = 0.5
    noise_to_signal: float = 0.2
    proxy_noise_scale: float = 0.5

    def __post_init__(self):
        if self.n_features < D_TRUE:
            raise ValueError(f"n_features must be >= {D_TRUE}, got {self.n_features}")
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
    n, d = spec.n_samples, spec.n_features

    z = rng.normal(size=(n, D_TRUE))
    f = true_function(z)
    y = f + rng.normal(scale=spec.noise_to_signal * f.std() + 1e-12, size=n)

    n_decoy = d - D_TRUE
    n_pure_noise = int(round(n_decoy * spec.frac_pure_noise))
    n_derived = n_decoy - n_pure_noise

    decoys = np.empty((n, n_decoy))
    decoy_kind = np.empty(n_decoy, dtype=int)
    for j in range(n_derived):
        k = rng.integers(1, min(3, D_TRUE) + 1)
        idx = rng.choice(D_TRUE, size=k, replace=False)
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
    kind_unpermuted = np.concatenate([np.zeros(D_TRUE, dtype=int), decoy_kind])

    perm = rng.permutation(d)
    inv_perm = np.argsort(perm)
    x = x_unpermuted[:, perm]
    feature_kind = kind_unpermuted[perm]
    true_feature_positions = inv_perm[:D_TRUE]

    return SyntheticDataset(
        X=x, y=y, true_feature_positions=true_feature_positions, feature_kind=feature_kind
    )


def generate_splits(
    n_features: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    **spec_kwargs,
) -> dict:
    """Generate train/val/test splits from one i.i.d. draw of the process.

    Rows are independent draws, so slicing a single generated block into
    contiguous train/val/test segments is equivalent to drawing them separately.
    """
    spec = SyntheticSpec(
        n_features=n_features, n_samples=n_train + n_val + n_test, seed=seed, **spec_kwargs
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
