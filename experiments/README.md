# TabMGP Experiments

This directory contains the scripts, configs, tests, generated outputs, and
support files used to reproduce the experiments for the TabMGP paper. The repo
root is reserved for direct TabMGP usage.

## Environment

This directory owns the reproducible experiment environment. From the repo root:

```bash
cd experiments
uv sync
```

If you are not using `uv`, install the requirements file instead:

```bash
pip install -r requirements-experiment.txt
```

The experiment scripts are plain Python scripts. The repo is not an installable
package; this directory contains its own copies of the shared TabMGP helpers
used by the reproduction workflow.

## Workflow

Create a dataset/output directory:

```bash
uv run python prepare-dgp.py id="example-linreg" dgp=regression-fixed seed=1001
```

This writes to `outputs/...` by default. Then run TabMGP and the baseline
posteriors:

```bash
uv run python run-tabmgp.py "expdir='outputs/example-linreg/name=regression-fixed dim_x=10 noise_std=1.0 data=500 seed=1001'"
uv run python run-bb.py "expdir='outputs/example-linreg/name=regression-fixed dim_x=10 noise_std=1.0 data=500 seed=1001'"
uv run python run-copula.py "expdir='outputs/example-linreg/name=regression-fixed dim_x=10 noise_std=1.0 data=500 seed=1001'" init=std
uv run python run-bayes.py "expdir='outputs/example-linreg/name=regression-fixed dim_x=10 noise_std=1.0 data=500 seed=1001'" prior=asymp
```

The full paper run is encoded in:

```bash
bash run-experiments.sh
```

## Contents

- `conf/`: Hydra configs for DGP setup and posterior runners.
- `pyproject.toml`, `uv.lock`: uv-managed experiment environment.
- `outputs/`: generated experiment outputs.
- `table/`: generated table CSVs.
- `prepare-dgp.py`: creates experiment datasets and resolved configs.
- `run-tabmgp.py`: runs TabMGP rollouts and posterior computation.
- `run-bb.py`, `run-copula.py`, `run-bayes.py`: baseline posterior runners.
- `tabmgp.py`, `functional.py`, `optimizer.py`, `utils.py`: local copies used
  by experiment scripts.
- `table.py`, `table.R`, `visual-*.py`: table and figure generation.
- `optional/`, `obsolete/`: optional and legacy experiment utilities.
- `test_*.py`: experiment regression/smoke tests.

## Tests

From this directory:

```bash
uv run pytest tests/test_baseline_shape_consistency.py
uv run pytest tests/test_trace_comparison.py
```

The trace comparison test creates temporary runs under `outputs/`.
