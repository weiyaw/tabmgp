# TabMGP: Martingale Posterior with TabPFN

Thanks for dropping by! This document is structured to be self-contained for
each use case listed below.

## For Trying Out TabMGP

See `example.py` for a minimal example of computing TabMGP. The only required
script is `rollout.py`, which performs forward sampling. You can then use the
estimators in `scikit-learn` as your functional. The `requirements-mgp.txt` file
contains all the packages for trying out TabMGP, without the heavy packages
needed for reproducing the experiments.

We also provide more performant, JAX-jittable estimators in `functional.py`; see
`example.py` for usage. These are useful when drawing a large number of
posterior samples, as `.fit()` in `scikit-learn` can be slow.

### Example (Logistic Regression)

The following snippet generates 5 samples from TabMGP. We forward sample only 20
steps here, but in practice this should be much larger (500+ in the experiments
of the paper).

```python
import jax
import numpy as np
from rollout import forward_sampling, TabPFNClassifierPredRule

# Generate some dummy training data for logistic regression
x_train = jax.random.normal(jax.random.key(101), (20, 5))
y_train = jax.random.bernoulli(jax.random.key(102), p=0.4, shape=(20,)).astype(int)

# Set up TabPFN as the prediction rule
pred_rule = TabPFNClassifierPredRule(
    categorical_x=[False] * x_train.shape[-1],
    n_estimators=4,
    average_before_softmax=False,
)
one_step_ahead = pred_rule.sample

# Collect 5 samples (one sample per forward sampling run)
from sklearn.linear_model import LogisticRegression

functional = LogisticRegression(penalty=None)
samples = []
key = jax.random.key(98)
for i in range(5):
    # Perform forward sampling with 20 steps (500+ recommended)
    x_full, y_full = forward_sampling(
        jax.random.fold_in(key, i), one_step_ahead, x_train, y_train, 20
    )

    fitted = functional.fit(x_full, y_full)
    theta = np.concatenate([fitted.intercept_.ravel(), fitted.coef_.ravel()])
    samples.append(theta)
```

## For Using Predictive Rules Other Than TabPFN

See `example.py` for a minimal example of computing TabMGP. To use a custom
predictive rule, simply redefine `one_step_ahead` in `example.py` with your own
predictive distribution:

```python
def one_step_ahead(
    key: PRNGKeyArray, x_new: Array, x_prev: Array, y_prev: Array
) -> Array:
    # Define the logic to draw a sample from the one-step-ahead predictive
    # distribution: y_new ~ P(. | x_new, x_prev, y_prev).
    pass
```

You may ignore the `key` argument if not using JAX. Random variates can be drawn
directly (e.g. via `np.random`), at the cost of reduced reproducibility.

The `requirements-mgp.txt` file contains only the packages for trying out
TabMGP.


## For Reproducing Results in the Paper



<!-- ### Requirements -->
<!-- We provide `requirements.txt` and `requirements-posterior.txt` files that can be -->
<!-- used with pip. The recommended way is to create a conda env, then pip install: -->

<!-- ```bash -->
<!-- conda create --name <env> python=3.11 -->
<!-- conda activate <env> -->
<!-- pip install -r requirements.txt -->
<!-- pip install -r requirements-posterior.txt -->
<!-- ``` -->
<!-- This will install a CPU version of JAX. Please see the [_JAX repo_](https://github.com/google/jax) for the latest -->
<!-- instructions on how to install JAX on your hardware. -->

### The workflow
Due to computational constraints, the workflow for the coverage experiment is
organised into two scripts:
- `run-rollout.py`: Start an experiment by running TabMGP forward sampling steps
  and save them into an output directory. It also saves the dataset used for the
  experiments.
- `run-posterior.py`: Take the output directory from `run-rollout.py` as input.
  It reads the dataset and the forward samples of TabMGP, and computes the
  posterior of TabMGP and all other baselines.

These scripts take in arguments which are managed by
[`hydra`](https://hydra.cc/), with the default arguments given in the `conf`
folder. For example:

``` bash
python run-rollout.py date="2025-06-99" dgp=regression-fixed seed=1001
```

It will run forward sampling with TabPFN and save both the forward samples and
the dataset to `./outputs/2025-06-99/name=regression-fixed dim_x=10
resample_x=bb data=20 seed=1001`. Then,

``` bash
python run-posterior.py "expdir='./outputs/2025-06-99/name=regression-fixed dim_x=10
resample_x=bb data=20 seed=1001'"
```

will read from this directory and compute the actual TabMGP and posteriors from
other baselines.

### Environment setup
A requirements file `requirements-experiment.txt` is provided for convenience. We
suggest a CPU version of `jax` for running `run-rollout.py` to avoid GPU
conflicts with PyTorch. The `run-posterior.py` script doesn't require PyTorch and
a GPU version of `jax` is preferable here.

### Configuration of the coverage experiment
The shell script `run-experiments.sh` has the exact configuration and command to
compute the posteriors used in the paper. It does 3 things:
1. Runs `run-rollout.py` over 30 setups, 100 repetitions for each setup, for the
   coverage experiment.
2. Runs `run-rollout.py` on 2 setups, each with 6 different sizes of training
   set, for the concentration experiment.
3. Runs `run-posterior.py` on all the output directories produced by
   `run-rollout.py`. It only computes the trace plots for one repetition in each
   setup, i.e., any repetition with seed of 1001.


### Computation time
The script for forward sampling `run-rollout.py` is the most computationally
intensive. In general it takes 60-100 seconds to complete a forward sampling of
500 steps on an Nvidia L40s. By default, the script will run forward sampling 100
times sequentially to get 100 TabMGP samples. For our coverage experiment we run
100 repetitions over 30 setups. It took roughly 7200 GPU-hours to run just
`run-rollout.py` alone on an Nvidia L40s.

Subsequent posterior computation with `run-posterior.py` is relatively fast and
can be done in a few hours on GPUs (or 1-2 days on CPUs). The main bottleneck of
this script is the copula-based baseline.


### The acid test
We also have a script to compute acid-related objects.

- `run-acid.py`: Take the output directory from `run-rollout.py` as input. It
  reads the dataset and computes all objects required for acid tests.


### File structure
```
+-- run-experiments.sh (a bash script to compute all posteriors in the paper. All outputs are saved in the `outputs` folder.)
+-- run-rollout.py (main script to perform forward sampling of TabMGP for all the experiments)
+-- run-posterior.py (main script to actually compute posteriors (TabMGP and other baselines) for all the experiments)

+-- pr_copula (the copula-based martingale posterior from https://github.com/edfong/MP with a bug fix)
+-- baseline.py (all definitions of baseline methods, e.g. standard Bayes, Bayesian bootstrap, copula)
+-- credible_set.py (helper functions for constructing credible sets and coverage)
+-- dgp.py (helper functions that define data generating process)
+-- functional.py (helper functions of all functionals, e.g. MLE estimators)
+-- optimizer.py (a collection of gradient-based optimisers, including L-BFGS)
+-- preprocessor.py (helper functions to clean up data and drop collinear features)
+-- rollout.py (functions that define TabMGP predictive rules)
+-- utils.py (miscellaneous utility functions)

+-- plot_settings.py (some constants useful for reproducing tables and plots in the paper)
+-- plot.py (script for generating plots in the paper)
+-- table.py (script for generating tables in the paper)

+-- run-acid.py (an optional script to compute objects for the acid test)
```
