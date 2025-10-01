# %%
import numpy as np
import matplotlib.pyplot as plt

n = 100
data_rng = np.random.default_rng(100)
x_train = np.concatenate([np.ones((n, 1)), data_rng.normal(size=(n, 2))], axis=1)
logits = x_train @ np.array([0.1, 1, 2])
probs = 1 / (1 + np.exp(-logits))
y_train = data_rng.binomial(1, p=probs)

np.save(f"data/x_train_{n}.npy", x_train)
np.save(f"data/y_train_{n}.npy", y_train)

# %%
