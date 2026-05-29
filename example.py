import numpy as np
from tabmgp_minimal import forward_sampling, TabPFNClassifierPPD
from sklearn.linear_model import LogisticRegression

# Generate some dummy training data for logistic regression
rng = np.random.default_rng(101)
x_train = rng.normal(size=(20, 5))
y_train = rng.binomial(1, p=0.4, size=20).astype(int)


#### Draw 5 samples from TabMGP, using the logistic regression in scikit-learn as our functional ####

# Setup TabPFN as prediction rule
pred_rule = TabPFNClassifierPPD()
one_step_ahead = pred_rule.sample

# Collect 5 samples, one sample per forward sampling
functional = LogisticRegression(penalty=None)
samples = []
rollout_rng = np.random.default_rng(98)
for i in range(5):
    # Perform forward sampling with 20 steps (in practice we use 500+)
    x_full, y_full = forward_sampling(
        one_step_ahead, x_train, y_train, 20, rng=rollout_rng
    )

    fitted = functional.fit(x_full, y_full)
    theta = np.concatenate([fitted.intercept_.ravel(), fitted.coef_.ravel()])
    samples.append(theta)
