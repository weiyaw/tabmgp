import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from sklearn.linear_model import LogisticRegression

from tabmgp import TabPFNClassifierPPD, forward_sampling_batched


def run_batch(x_train, y_train, n_rollouts, rng):
    """Run n_rollouts rollouts together in one process."""
    # Use TabPFN v3.5-fast as prediction rule
    pred_rule = TabPFNClassifierPPD(model_path="tabpfn-v3.5-fast-20260909.safetensors")
    # Perform forward sampling with 20 steps (in practice we use 500+)
    return forward_sampling_batched(
        pred_rule.sample_batched, x_train, y_train, 20, n_rollouts, rng=rng
    )


# Worker processes re-import this file, so the work runs only in the main process
if __name__ == "__main__":
    # Generate some dummy training data for logistic regression
    rng = np.random.default_rng(101)
    x_train = rng.normal(size=(20, 5))
    y_train = rng.binomial(1, p=0.4, size=20).astype(int)

    #### Draw 10 samples from TabMGP: 2 processes, each rolling out 5 together ####

    n_procs, n_rollouts = 2, 5
    # Each process gets its own independent random stream
    rngs = np.random.default_rng(101).spawn(n_procs)
    # Split the CPU threads between processes; spawned workers read this at start
    os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) // n_procs))
    with ProcessPoolExecutor(n_procs, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(run_batch, x_train, y_train, n_rollouts, rng) for rng in rngs
        ]
        results = [future.result() for future in futures]

    # Fit the functional on every rollout
    functional = LogisticRegression(C=np.inf)
    samples = []
    for x_batch, y_batch in results:
        for x_full, y_full in zip(x_batch, y_batch):
            fitted = functional.fit(x_full, y_full)
            theta = np.concatenate([fitted.intercept_.ravel(), fitted.coef_.ravel()])
            samples.append(theta)

    samples = np.stack(samples)  # (n_samples, 1 + n_features)
    print("Posterior samples:")
    print(samples)
    print("Posterior mean:")
    print(samples.mean(axis=0))
