import jax
import numpy as np
from rollout import forward_sampling, TabPFNClassifierPredRule

# Generate some dummy training data for logistic regression
x_train = jax.random.normal(jax.random.key(101), (20, 5))
y_train = jax.random.bernoulli(jax.random.key(102), p=0.4, shape=(20,)).astype(int)


#### Draw 10 samples from TabMGP, using the logistic regression in scikit-learn as our functional ####

# Setup TabPFN as prediction rule
pred_rule = TabPFNClassifierPredRule(
    categorical_x=[False] * x_train.shape[-1],
    n_estimators=4,
    average_before_softmax=False,
)
one_step_ahead = pred_rule.sample

# Collect 5 samples, one sample per forward sampling
from sklearn.linear_model import LogisticRegression

functional = LogisticRegression(penalty=None)
samples = []
key = jax.random.key(98)
for i in range(5):
    # Perform forward sampling with 20 steps (in practice we use 500+)
    x_full, y_full = forward_sampling(
        jax.random.fold_in(key, i), one_step_ahead, x_train, y_train, 20
    )

    fitted = functional.fit(x_full, y_full)
    theta = np.concatenate([fitted.intercept_.ravel(), fitted.coef_.ravel()])
    samples.append(theta)


#### Draw 5 samples from TabMGP, this time using our faster implementation of the logistic regression ####
from functional import LogisticRegression as FastLogisticRegression

functional = FastLogisticRegression(n_classes=2, l2=0.0)
samples = []
key = jax.random.key(98)
for i in range(5):
    # Perform forward sampling with 20 steps (in practice we use 500+)
    x_full, y_full = forward_sampling(
        jax.random.fold_in(key, i), one_step_ahead, x_train, y_train, 20
    )

    init_theta = jax.random.normal(jax.random.key(0), (x_full.shape[1] + 1,))
    theta, _ = functional.minimize_loss(
        {"x": x_full, "y": y_full.astype(int)}, init_theta, None
    )
    samples.append(theta)
#
