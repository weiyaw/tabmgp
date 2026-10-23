# TabMGP: Martingale Posterior with TabPFN

This repository implements TabMGP and reproduces the results of the TabMGP
[paper](https://arxiv.org/abs/2510.25154).

TabMGP is a martingale posterior that uses TabPFN as its predictive rule.
Starting from the observed data, forward sampling extends the dataset one point
at a time. Each step draws a covariate from the data so far and samples its
response from TabPFN's one-step-ahead predictive distribution. Fitting a model
of interest (the functional) to each extended dataset gives one posterior
sample.

The method itself is in this top-level directory, and the code for the paper's
experiments is in [`experiments/`](experiments/README.md).

## Setup

Install the required packages with

```bash
pip install -r requirements-mgp.txt
```

The code is a set of plain Python modules rather than an installable package,
which makes it easier to experiment with. Run scripts from this directory. We
tested the examples with Python 3.13 and 3.14, tabpfn 9.2.0 and the TabPFN v3.5-fast
model. At the time of writing, PyTorch does not support Python 3.15.

The TabPFN model weights are gated. The first time TabPFN fits, it asks you to
log in to Prior Labs and accept the model license in the browser, then
downloads the weights. On a machine without an interactive terminal, set the
`TABPFN_TOKEN` environment variable instead. See the
[TabPFN installation guide](https://github.com/PriorLabs/tabpfn#installation--setup)
for details.

## Example

[`example.py`](example.py) performs logistic regression with TabMGP. It uses
`TabPFNClassifierPPD` from [`tabmgp.py`](tabmgp.py) as the predictive rule,
`forward_sampling` to extend the data, and scikit-learn's `LogisticRegression`
as the functional. Posterior samples are drawn one after another in a loop, and
the script prints them along with their mean. The script loads the TabPFN
v3.5-fast model through the `model_path` argument. Without `model_path`, tabpfn
uses its default model.

```bash
python example.py
```

Each posterior sample needs its own forward sampling run, so
[`example_batched.py`](example_batched.py) draws samples in parallel. It can run
several Python processes and batch the TabPFN forward passes within each process
using `forward_sampling_batched`. On an RTX 4090 (24 GB) with a reasonable
number of CPU cores, we recommend starting with 10 processes and a batch size
of 3. Batching relies on TabPFN's `predict_batched` and `predict_proba_batched`,
which older tabpfn versions may not have.

```bash
python example_batched.py
```

## Custom Predictive Rule

TabPFN is only one choice of predictive rule. `forward_sampling` accepts any
one-step-ahead sampler that takes a new covariate, the previous covariates and
targets, and an optional NumPy random generator.

```python
from tabmgp import forward_sampling

def one_step_ahead(x_new, x_prev, y_prev, *, rng=None):
    # Draw y_new ~ P(. | x_new, x_prev, y_prev)
    ...

x_full, y_full = forward_sampling(
    one_step_ahead, x_train, y_train, forward_steps=500, rng=123
)
```

To use batching as in `example_batched.py`, the sampler takes lists with one
entry per rollout and returns an array of shape `(n_rollouts, 1)`.

```python
from tabmgp import forward_sampling_batched

def one_step_ahead_batched(x_new_list, x_prev_list, y_prev_list, *, rng=None):
    ...

x_full, y_full = forward_sampling_batched(
    one_step_ahead_batched, x_train, y_train,
    forward_steps=500, n_rollouts=10, rng=123,
)
```

## Functionals and Paper Experiments

The examples fit the functional with scikit-learn. For larger simulations,
[`functional.py`](functional.py) provides JAX-jittable functionals for linear,
quantile and logistic regression, with helpers in `utils.py` and
`optimizer.py`. The paper's experiments use these, and
[`experiments/README.md`](experiments/README.md) explains how to reproduce them.
