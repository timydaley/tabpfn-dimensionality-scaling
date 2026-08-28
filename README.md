# Does tabular foundation model performance scale with feature dimensionality?

A synthetic-data study of how TabICL and TabPFN v2 — transformer-based, in-context-learning
"tabular foundation models" — behave on regression as the number of input features grows
from 10 to 10,000, compared against XGBoost, Random Forest, and Ridge baselines, under two
conditions:

- **Fixed-signal**: a true generative process with a fixed nonlinear function of 5 latent
  features, regardless of D. The remaining D-5 features are either noisy nonlinear functions
  of the true features (partially informative, not perfectly recoverable) or pure noise. This
  isolates ambient dimensionality from signal dimensionality.
- **Scaling-signal**: the number of causal features itself grows with D (5% of D), so ambient
  and signal dimensionality grow together, as in most real high-dimensional tabular data.

**Headline finding:** dimensionality alone is not the hazard. Under fixed-signal, TabICL shows
no degradation from D=10 to D=10,000 (R2 stays in 0.899-0.937) and dominates every baseline.
Under scaling-signal (training set size held fixed at 2000 rows), every model degrades
sharply — TabICL falls from R2=0.937 (D=10) to 0.151 (D=10,000), at which point a plain
cross-validated Ridge baseline matches or exceeds it. See `paper/main.pdf` for the full
writeup, including a discussion of candidate mechanisms (statistical difficulty vs. the
feature-bagging ensemble severing causal-feature interactions across chunks).

Both TabPFN and TabICL have hard, trained-regime feature-count caps (500 and ~2,000 respectively).
Beyond those caps we apply a feature-bagging ensemble wrapper (split features into <=cap chunks,
fit/predict per chunk, average) rather than running the model outside its validated regime natively.

**Status:** results in `paper/main.pdf` cover Ridge, Random Forest, XGBoost, and TabICL, under
both conditions. TabPFN v2 is implemented (`src/tabdim/models.py:make_tabpfn`) and integrated into
the same sweep, but its model weights are gated behind an interactive, human-completed license
acceptance (https://ux.priorlabs.ai) with no automated path around it — we were not able to
obtain a token for this project, so no TabPFN v2 results are reported. See `paper/main.tex`,
Section 4, for details. If you have a `TABPFN_TOKEN`, running it is one command (below).

## Repo layout

- `src/tabdim/data_gen.py` - synthetic data generating process (fixed- and scaling-signal regimes)
- `src/tabdim/models.py` - model wrappers (baselines + foundation models + feature-bagging ensemble)
- `src/tabdim/experiment.py` - sweep orchestration
- `scripts/run_experiments.py` - CLI to run the sweep
- `scripts/make_figures.py` - turns a results CSV into R2/RMSE-vs-D figures/tables
- `scripts/make_signal_comparison_figure.py` - fixed-signal vs. scaling-signal comparison figure
- `tests/` - unit tests
- `results/sweep_no_tabpfn.csv` - the 400-cell fixed-signal sweep (Ridge/RF/XGBoost/TabICL, both N-regimes)
- `results/sweep_scaling_signal.csv` - the 200-cell scaling-signal sweep (same 4 models, fixed-N only)
- `figures/` - generated figures and summary tables
- `paper/` - ICML2025-format writeup (`main.pdf`, `main.tex`)

## Setup

Requires a native arm64 Python on Apple Silicon (or any Linux/x86_64 box with current PyPI wheels).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Reproduce the paper's fixed-signal results (no TabPFN):
python scripts/run_experiments.py --models ridge random_forest xgboost tabicl \
    --out results/sweep_no_tabpfn.csv
python scripts/make_figures.py --results results/sweep_no_tabpfn.csv

# Reproduce the scaling-signal results (causal features grow with D):
python scripts/run_experiments.py --models ridge random_forest xgboost tabicl \
    --regimes fixed_n --signal-regimes scaling --out results/sweep_scaling_signal.csv
python scripts/make_signal_comparison_figure.py

# To also run TabPFN v2, once you have a token from https://ux.priorlabs.ai:
export TABPFN_TOKEN=...
python scripts/run_experiments.py --models tabpfn --out results/sweep_tabpfn.csv
```
