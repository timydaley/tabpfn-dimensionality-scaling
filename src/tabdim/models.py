"""Model wrappers sharing a plain fit/predict interface for the dimensionality sweep."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Hard feature-count caps beyond which each foundation model's context was never
# trained/validated (TabPFN v2, Hollmann et al., Nature 2025: 500 features / 10,000
# rows; TabICL, Qu et al. 2025: ~2,000 features). Chunks of this width are fed
# through FeatureBaggedRegressor below rather than exceeding the trained regime.
TABPFN_MAX_FEATURES = 500
TABICL_MAX_FEATURES = 2000


class SklearnStyleModel:
    """Wraps any estimator that already exposes sklearn's fit/predict."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnStyleModel":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


def make_ridge(seed: int) -> SklearnStyleModel:
    return SklearnStyleModel(Ridge(alpha=1.0, random_state=seed))


def make_random_forest(seed: int) -> SklearnStyleModel:
    # sklearn's RandomForestRegressor defaults to max_features=1.0 (every split
    # considers all D features), unlike the classifier's 'sqrt' default. At D in the
    # thousands that makes each split O(N*D) and the whole forest impractically slow;
    # 'sqrt' is the standard high-dimensional setting and is what most practitioners
    # actually use.
    return SklearnStyleModel(
        RandomForestRegressor(
            n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=seed
        )
    )


def make_xgboost(seed: int) -> SklearnStyleModel:
    return SklearnStyleModel(
        XGBRegressor(n_estimators=300, max_depth=6, random_state=seed, n_jobs=-1)
    )


def make_tabpfn(seed: int, device: str = "cpu") -> SklearnStyleModel:
    from tabpfn import TabPFNRegressor

    return SklearnStyleModel(
        TabPFNRegressor(device=device, random_state=seed, ignore_pretraining_limits=True)
    )


def make_tabicl(seed: int, device: str = "cpu") -> SklearnStyleModel:
    from tabicl import TabICLRegressor

    return SklearnStyleModel(TabICLRegressor(device=device, random_state=seed, verbose=False))


class FeatureBaggedRegressor:
    """Ensemble wrapper letting a model with a hard feature-count cap run on inputs
    with more features than that cap: split features into <=max_features chunks,
    fit/predict a fresh model instance per chunk, and average predictions.

    This is the feature-bagging pattern used to push TabPFN-style in-context
    learners past their trained regime (see Frantar & Xin et al., "Scaling
    TabPFN: Sketching and Feature Selection for Tabular PFNs", arXiv:2311.10609).
    When n_features <= max_features this degrades to a single chunk containing
    every feature, i.e. the model runs exactly as it natively would.
    """

    def __init__(
        self,
        base_model_factory: Callable[[int], object],
        max_features: int,
        seed: int = 0,
    ):
        self.base_model_factory = base_model_factory
        self.max_features = max_features
        self.seed = seed
        self._chunks: list[np.ndarray] | None = None
        self._models: list | None = None

    def _make_chunks(self, n_features: int) -> list[np.ndarray]:
        if n_features <= self.max_features:
            return [np.arange(n_features)]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_features)
        n_chunks = int(np.ceil(n_features / self.max_features))
        return list(np.array_split(perm, n_chunks))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureBaggedRegressor":
        self._chunks = self._make_chunks(X.shape[1])
        self._models = []
        for i, chunk in enumerate(self._chunks):
            model = self.base_model_factory(self.seed + i)
            model.fit(X[:, chunk], y)
            self._models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(X[:, chunk]) for m, chunk in zip(self._models, self._chunks)])
        return preds.mean(axis=0)


def make_tabpfn_ensemble(seed: int, device: str = "cpu") -> FeatureBaggedRegressor:
    return FeatureBaggedRegressor(
        lambda s: make_tabpfn(s, device=device), max_features=TABPFN_MAX_FEATURES, seed=seed
    )


def make_tabicl_ensemble(seed: int, device: str = "cpu") -> FeatureBaggedRegressor:
    return FeatureBaggedRegressor(
        lambda s: make_tabicl(s, device=device), max_features=TABICL_MAX_FEATURES, seed=seed
    )


MODEL_REGISTRY: dict[str, Callable[..., object]] = {
    "ridge": lambda seed, device="cpu": make_ridge(seed),
    "random_forest": lambda seed, device="cpu": make_random_forest(seed),
    "xgboost": lambda seed, device="cpu": make_xgboost(seed),
    "tabpfn": make_tabpfn_ensemble,
    "tabicl": make_tabicl_ensemble,
}


def get_model(name: str, seed: int, device: str = "cpu"):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choices are {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seed, device=device)
