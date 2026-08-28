# Does tabular foundation model performance scale with feature dimensionality?

A synthetic-data study of how TabPFN v2 and TabICL — transformer-based, in-context-learning
"tabular foundation models" — behave on regression as the number of input features grows
from 10 to 10,000, compared against XGBoost, Random Forest, and Ridge baselines.

Each dataset has a true generative process: a fixed nonlinear function of 5 latent features.
The remaining D-5 features are either noisy nonlinear functions of the true features (partially
informative, not perfectly recoverable) or pure noise. We sweep D across a log-spaced grid under
two regimes — training set size held fixed, and training set size scaling with D — and report
held-out test RMSE/R2.

Both TabPFN and TabICL have hard, trained-regime feature-count caps (500 and ~2,000 respectively).
Beyond those caps we apply a feature-bagging ensemble wrapper (split features into <=cap chunks,
fit/predict per chunk, average) rather than running the model outside its validated regime natively.

**Status:** results in `paper/main.pdf` cover Ridge, Random Forest, XGBoost, and TabICL.
TabPFN v2 is implemented (`src/tabdim/models.py:make_tabpfn`) and integrated into the same sweep,
but its model weights are gated behind an interactive, human-completed license acceptance
(https://ux.priorlabs.ai) with no automated path around it — we were not able to obtain a token
for this project, so no TabPFN v2 results are reported. See `paper/main.tex`, Section 4, for
details. If you have a `TABPFN_TOKEN`, running it is one command (below).

## Repo layout

- `src/tabdim/data_gen.py` - synthetic data generating process
- `src/tabdim/models.py` - model wrappers (baselines + foundation models + feature-bagging ensemble)
- `src/tabdim/experiment.py` - sweep orchestration
- `scripts/run_experiments.py` - CLI to run the sweep
- `scripts/make_figures.py` - turns a results CSV into the paper's figures/tables
- `tests/` - unit tests
- `results/sweep_no_tabpfn.csv` - the 400-cell sweep behind the paper's results (Ridge/RF/XGBoost/TabICL)
- `figures/` - generated figures and summary tables
- `paper/` - ICML2025-format writeup (`main.pdf`, `main.tex`)

## Setup

Requires a native arm64 Python on Apple Silicon (or any Linux/x86_64 box with current PyPI wheels).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Reproduce the paper's results (no TabPFN):
python scripts/run_experiments.py --models ridge random_forest xgboost tabicl \
    --out results/sweep_no_tabpfn.csv
python scripts/make_figures.py --results results/sweep_no_tabpfn.csv

# To also run TabPFN v2, once you have a token from https://ux.priorlabs.ai:
export TABPFN_TOKEN=...
python scripts/run_experiments.py --models tabpfn --out results/sweep_tabpfn.csv
```
