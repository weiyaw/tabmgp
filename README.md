# TabMGP: Martingale Posterior with TabPFN

This top-level directory contains the files needed to try and use TabMGP as a
plain Python module. The paper reproduction workflow, experiment scripts,
generated outputs, tables, tests, and legacy utilities live in
[`experiments/`](experiments/README.md).

## Setup

Install the lightweight runtime requirements:

```bash
pip install -r requirements-mgp.txt
```

The repository is intentionally not structured as an installable Python package.
Run examples and imports from this directory.

## Minimal Example

See [`example.py`](example.py) for a short example that draws TabMGP samples and
fits a scikit-learn logistic regression functional.

```bash
python example.py
```

The core runtime files are:

- [`tabmgp_minimal.py`](tabmgp_minimal.py): NumPy-based forward sampling helpers.
- [`tabmgp.py`](tabmgp.py): JAX-compatible forward sampling and TabPFN predictive
  distribution helpers.
- [`functional.py`](functional.py): JAX-jittable functionals for larger posterior
  simulations.

## Using a Custom Predictive Rule

TabMGP only needs a one-step-ahead predictive sampler. For the minimal API, this
callable should accept a new covariate, previous covariates and targets, and an
optional NumPy random generator:

```python
def one_step_ahead(x_new, x_prev, y_prev, *, rng=None):
    # Draw y_new ~ P(. | x_new, x_prev, y_prev)
    ...
```

Then pass it to `forward_sampling`:

```python
from tabmgp_minimal import forward_sampling

x_full, y_full = forward_sampling(
    one_step_ahead,
    x_train,
    y_train,
    forward_steps=500,
    rng=123,
)
```

For the paper reproduction workflow, see
[`experiments/README.md`](experiments/README.md).
