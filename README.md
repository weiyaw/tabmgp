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

To be finished.



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

<!-- ## File Structure -->
<!-- ``` -->
<!-- +-- acid.py (Run acid test) -->
<!-- +-- credible_set.py (Credible sets related files) -->
<!-- +-- data.py (All data generating distribution and dataset downloading) -->
<!-- +-- forward.py (Forward sampling with TabPFN) -->
<!-- +-- posterior.py (Compute all type of posteriors) -->
<!-- +-- plot.py (Generate plots) -->
<!-- +-- table.py (Generate tables) -->
<!-- +-- utils.py (Utility functions) -->
<!-- ``` -->

<!-- ### Example -->
<!-- ```bash -->
<!-- python forward.py data_size=20 recursion_length=500 n_estimators=8 resample_x=bb dgp=regression-fixed -->
<!-- ``` -->
<!-- This will run forward sampling for TabPFN and save the samples in a folder "output/name-of-sample-folder". -->

<!-- ``` bash -->
<!-- python posterior.py expdir="output/name-of-sample-folder" -->
<!-- ``` -->
<!-- This will compute all martingale posterior (TabMGP, BB, Copula) and Bayes. This has to be run after running `forward.py`. -->

