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

## Repo layout

- `src/tabdim/data_gen.py` - synthetic data generating process
- `src/tabdim/models.py` - model wrappers (baselines + foundation models + feature-bagging ensemble)
- `src/tabdim/experiment.py` - sweep orchestration
- `scripts/run_experiments.py` - CLI to run the sweep
- `scripts/make_figures.py` - turns `results/sweep.csv` into the paper's figures/tables
- `tests/` - unit tests
- `paper/` - ICML2025-format writeup

## Setup

Requires a native arm64 Python on Apple Silicon (or any Linux/x86_64 box with current PyPI wheels).
TabPFN requires a one-time license acceptance; set `TABPFN_TOKEN` before running with `tabpfn`
included in `--models` (see https://ux.priorlabs.ai).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_experiments.py --out results/sweep.csv
python scripts/make_figures.py
```
