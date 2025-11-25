import logging
import math
import os
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import openml
import pandas as pd
from jaxtyping import Array, ArrayLike, PRNGKeyArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from ucimlrepo import fetch_ucirepo

import utils

openml.config.set_root_cache_directory(os.path.join(os.getcwd(), "datasets/openml"))


def validate_split_distribution(
    X_train: ArrayLike, X_test: ArrayLike, is_X_categorical: list[bool]
):
    from scipy import stats
    import matplotlib.pyplot as plt

    continuous_indices = [i for i in range(X_train.shape[1]) if not is_X_categorical[i]]

    # Check continuous variables
    ks_stats = []
    p_values = []

    dim_x = X_train.shape[1]
    plt.figure(figsize=(6, 2 * dim_x))
    # for idx in continuous_indices:
    for idx in range(dim_x):
        x_train = X_train[:, idx]
        x_test = X_test[:, idx]
        assert x_train.dtype == x_test.dtype
        ks_stat, p_value = stats.ks_2samp(x_train, x_test)
        ks_stats.append(ks_stat)
        p_values.append(p_value)

        plt.subplot(dim_x // 2 + 1, 2, idx + 1)
        plt.hist(x_train, bins=30, alpha=0.5, label="Train", density=True)
        plt.hist(x_test, bins=30, alpha=0.5, label="Test", density=True)
        plt.legend()
        plt.title(f"ks: {ks_stat:.2f}, p: {p_value:.2f}")

    plt.tight_layout()
    plt.show()

    results = {
        f"feature_{idx}_ks": {"statistic": ks_stat, "p_value": p_value}
        for idx, (ks_stat, p_value) in enumerate(zip(ks_stats, p_values))
    }


class DGP(eqx.Module):
    input_key: PRNGKeyArray
    train_data: dict[str, np.ndarray]

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        """
        This will be used as part of the forward recursion and could be inside a
        JAX transformed function.  It should only contain JAX
        transformation-compatible operations.
        """
        pass

    @abstractmethod
    def get_population(self) -> dict[str, np.ndarray]:
        pass


def multidim_stratified_split(
    key: PRNGKeyArray,
    X: ArrayLike,
    y: ArrayLike,
    is_X_categorical: list[bool],
    is_y_categorical: bool,
    train_size: int,
    n_bins: int,
    continuous_threshold: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified split of the data into training and test sets.
    Each dimension of x and y are stratified. For continuous variables, they are
    binned into `n_bins` bins.

    Args:
        key: JAX random key for reproducibility.
        X: Features data as a 2D array.
        y: Target data as a 1D array.
        is_X_categorical: List indicating whether each feature in X is categorical.
        is_y_categorical: Whether the target variable y is categorical.
        train_size: Number of samples to include in the training set.
        n_bins: Number of bins to use for continuous variables.

    Returns:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
    """

    seed = jax.random.randint(key, shape=(), minval=0, maxval=42949672).item()

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(np.hstack([y[:, None], X]))
    is_categorical = [is_y_categorical] + is_X_categorical

    # for splitting purpose, any continuous variable with less than 4 * n_bins unique values is treated as categorical
    strata_keys = []
    continuous_indices = []
    categorical_indices = []
    continuous_threshold = continuous_threshold or 4 * n_bins
    for i in range(df.shape[1]):
        if is_categorical[i]:
            categorical_indices.append(i)
        else:
            unique_values = df.iloc[:, i].nunique()
            if unique_values > continuous_threshold:
                # Treat as continuous
                continuous_indices.append(i)
            else:
                # Treat as categorical
                categorical_indices.append(i)

    # Bin continuous variables
    if continuous_indices:
        if n_bins == 1:
            continuous_binned = np.zeros((df.shape[0], len(continuous_indices)))
        else:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            continuous_binned = kbd.fit_transform(df.iloc[:, continuous_indices])

        logging.info("Number of data in each bin (for x treated as continuous):")
        for i, col_idx in enumerate(continuous_indices):
            strata_keys.append(continuous_binned[:, i].astype(int).astype(str))
            logging.info(
                f"x[{col_idx}]: {np.bincount(continuous_binned[:, i].astype(int), minlength=n_bins)}"
            )

    if categorical_indices:
        logging.info("Number of data in each bin (for x treated as categorical):")
        # Add categorical variables directly
        for col_idx in categorical_indices:
            strata_keys.append(df.iloc[:, col_idx].astype(str))
            logging.info(
                f"x[{col_idx}]: {df.iloc[:, col_idx].value_counts().to_numpy()}"
            )

    # Combine all features into single stratification key
    strata = pd.Series(["_".join(row) for row in zip(*strata_keys)])

    # Remove strata with too few samples
    strata_counts = strata.value_counts()
    valid_strata = strata_counts[strata_counts >= 2].index
    valid_mask = strata.isin(valid_strata)

    # Split only valid samples
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    strata_valid = strata[valid_mask]

    logging.info(f"Strata counts: {strata_counts.to_numpy()}")
    logging.info(f"Valid mask proportion: {sum(valid_mask) / valid_mask.size}")
    train_test_datasets = train_test_split(
        X_valid,
        y_valid,
        train_size=train_size,
        stratify=strata_valid,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = jax.tree.map(np.asarray, train_test_datasets)

    return X_train, X_test, y_train, y_test


class DGPReal(DGP):
    """
    These are the necessary variables for all read data DGPs.  It defines how
    the population is defined, how new x is drawn from the population, and how
    the data are split.

    By design the datasets are numpy arrays instead of JAX arrays, because the
    downloaded array might contain strings or mixed types columns.
    """

    full_data: dict[str, np.ndarray]
    test_data: dict[str, np.ndarray]
    categorical_x: list[bool]
    strata_bins: int

    def get_population(self) -> dict[str, np.ndarray]:
        return self.full_data

    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        # Sample x from self.full_data with replacement
        if self.full_data is None:
            raise ValueError("Full data is not available. Cannot sample x.")
        if "x" not in self.full_data:
            raise ValueError("Full data does not contain 'x' key.")
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(subkey, self.full_data["x"].shape[0], shape=(n,))
        return self.full_data["x"][indices]

    def split_data(
        self,
        key: PRNGKeyArray,
        n: int,
        is_y_categorical: bool,
        continuous_threshold: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform a stratified split of the full data into training and test sets.
        Each dim of x and y are stratified.  For continuous variables, they are
        binned into `n_bins` bins.

        Args: key: JAX random key for reproducibility.  n: Number of samples to
        include in the training set.  is_y_categorical: Whether the target
        variable is categorical.
        """

        x_train, x_test, y_train, y_test = multidim_stratified_split(
            key,
            self.full_data["x"],
            self.full_data["y"],
            self.categorical_x,
            is_y_categorical=is_y_categorical,
            train_size=n,
            n_bins=self.strata_bins,
            continuous_threshold=continuous_threshold,
        )
        return {"x": x_train, "y": y_train}, {"x": x_test, "y": y_test}


class DGPOpenML(DGPReal):
    """
    Download data from OpenML and identify the target and feature types.
    """

    openml_id: int
    target_index: int
    target_name: str
    feature_name: list[str]

    def __init__(
        self, openml_id: int, target_index: int, categorical_x: list[bool] | None
    ):
        self.openml_id = openml_id
        self.target_index = target_index
        dataset = openml.datasets.get_dataset(openml_id)

        # Get the data itself as a dataframe (or otherwise)
        df, _, is_categorical, var_names = dataset.get_data(dataset_format="dataframe")

        if categorical_x is None:
            # If not provided, use the OpenML's categorical information
            del is_categorical[target_index]  # remove the target column
            categorical_x = is_categorical

        self.target_name = var_names[target_index]
        logging.info(f"OpenML ID: {openml_id}, target: {self.target_name}")

        # Multiclass yeast and wine are hugely imbalance. Filter out the classes
        # with little observations (<3% of the overall).
        if openml_id in [40498, 181]:
            class_proportion = df[self.target_name].value_counts(normalize=True)
            # total_observations = len(X)
            classes_to_remove = class_proportion[class_proportion < 0.03].index
            df = df[~df[self.target_name].isin(classes_to_remove)]

        # Separate the target column and the rest of the columns
        y_column = df[[self.target_name]].to_numpy().squeeze()
        x_columns = df.drop(columns=[self.target_name]).to_numpy()

        self.categorical_x = categorical_x
        var_names.pop(target_index)
        self.feature_name = var_names
        self.full_data = {"x": x_columns, "y": y_column}


class DGPUCI(DGPReal):
    """
    Download data from UCI and identify the target and feature types.
    """

    uci_id: int
    target_name: str
    feature_name: list[str]

    def __init__(self, uci_id: int, categorical_x: list[bool]):
        self.uci_id = uci_id
        if not os.path.exists(f"datasets/uci-{uci_id}.pickle"):
            dataset = fetch_ucirepo(id=uci_id)
            utils.write_to(f"datasets/uci-{uci_id}.pickle", dataset)
        else:
            dataset = utils.read_from(f"datasets/uci-{uci_id}.pickle")
        variables = dataset.variables
        self.target_name = variables.loc[variables["role"] == "Target", "name"].iloc[0]
        self.categorical_x = categorical_x

        logging.info(f"UCI ID: {uci_id}, target: {self.target_name}")
        X = dataset.data.features.to_numpy()
        y = dataset.data.targets.to_numpy().squeeze()
        assert X.shape[1] == len(categorical_x)
        self.feature_name = variables.loc[
            variables["role"] == "Feature", "name"
        ].to_list()
        self.full_data = {"x": X, "y": y}


class DGPRegressionOpenML(DGPOpenML):

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        openml_id: int,
        target_index: int,
        strata_bins: int = 1,
        categorical_x: list[bool] | None = None,
        continuous_threshold: int | None = None,
    ):
        super().__init__(openml_id, target_index, categorical_x)
        self.input_key = key
        self.strata_bins = strata_bins
        logging.info(f"Strata bins: {strata_bins}")
        self.train_data, self.test_data = self.split_data(
            key, n, is_y_categorical=False, continuous_threshold=continuous_threshold
        )


class DGPClassificationOpenML(DGPOpenML):

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        openml_id: int,
        target_index: int,
        strata_bins: int = 1,
        categorical_x: list[bool] | None = None,
        continuous_threshold: int | None = None,
    ):
        super().__init__(openml_id, target_index, categorical_x)
        self.input_key = key
        self.strata_bins = strata_bins
        logging.info(f"Strata bins: {strata_bins}")
        self.train_data, self.test_data = self.split_data(
            key, n, is_y_categorical=True, continuous_threshold=continuous_threshold
        )


class DGPClassificationUCI(DGPUCI):

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        uci_id: int,
        categorical_x: list[bool],
        strata_bins: int = 1,
    ):
        super().__init__(uci_id, categorical_x)
        self.input_key = key
        self.strata_bins = strata_bins
        logging.info(f"Strata bins: {strata_bins}")
        self.train_data, self.test_data = self.split_data(key, n, is_y_categorical=True)


class DGPLidar(DGPReal):

    def __init__(self, key: PRNGKeyArray, n: int):
        self.input_key = key

        DATA_URI = "http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat"

        df = pd.read_csv(DATA_URI, sep=r"\s+")
        y = df["logratio"].values
        x = df["range"].values.reshape(-1, 1)

        logging.info(f"LIDAR, target: logratio")
        self.full_data = {"x": x, "y": y}
        self.categorical_x = [False]  # LIDAR has no categorical features
        self.strata_bins = 1

        # Convert to NumPy arrays
        if n == -1:
            # If n is -1, use the full dataset as training data
            self.train_data = {"x": x, "y": y}
            self.test_data = {"x": x}
        else:
            self.train_data, self.test_data = self.split_data(
                key, n, is_y_categorical=False
            )


class DGPSyntheticFixed(DGP):
    """
    Draw the weights from a prior and fixed to it. Draw x from a Unif(-1, 1).
    """

    beta0: np.ndarray
    dim_x: int
    categorical_x: list[bool]
    target_name: str
    feature_name: list[str]

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int):
        self.input_key = key
        # Draw betas from a Uniform(-2, 3) prior, and this is fixed
        fixed_key = jax.random.key(1058)
        self.beta0 = np.asarray(
            jax.random.uniform(fixed_key, shape=(dim_x,), minval=-2, maxval=3),
            dtype=np.float64,
        )
        self.dim_x = self.beta0.shape[0]
        key, data_key = jax.random.split(key, 2)
        self.train_data = self.get_data(data_key, n)
        self.target_name = "y"
        self.feature_name = [f"x{i + 1}" for i in range(dim_x)]
        self.categorical_x = [False] * dim_x

    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        # We need this for forward recursion
        key, subkey = jax.random.split(key)
        return jax.random.uniform(subkey, shape=(n, self.dim_x), minval=-1, maxval=1)

    def get_population(self) -> dict[str, np.ndarray]:
        population_data = self.get_data(jax.random.key(50), 100000)
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), population_data)

    @abstractmethod
    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        pass


class DGPClassificationFixed(DGPSyntheticFixed):
    """
    Draw the weights from a prior, then draw iid data from the linear logistic
    regression model as parameterised by the weights.
    """

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int):
        super().__init__(key, n, dim_x)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        probs = jax.scipy.special.expit(x_train @ self.beta0)
        y_train = jax.random.bernoulli(y_key, probs)
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.int8),
        }


class DGPClassificationFixedGMMLink(DGPSyntheticFixed):
    """
    Draw the weights from a prior, then draw iid data from the bernoulli model
    with the cdf of gaussian mixture model as the link function as parameterised
    by the weights.
    """

    a: float = -1.0

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int, a: float):
        self.a = a
        super().__init__(key, n, dim_x)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        cdf = jax.scipy.stats.norm.cdf
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        link = lambda p: 0.7 * cdf(p, loc=self.a) + 0.3 * cdf(p, loc=2.0)
        probs = link(x_train @ self.beta0)
        y_train = jax.random.bernoulli(y_key, probs)
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.int8),
        }


class DGPRegressionFixed(DGPSyntheticFixed):
    """
    Draw the weights from a prior, then draw iid data from the linear Gaussian
    regression model as parameterised by the weights.
    """

    noise_std: float = 1.0

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int, noise_std: float):
        self.noise_std = noise_std
        super().__init__(key, n, dim_x)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        y_train = (
            x_train @ self.beta0 + jax.random.normal(y_key, shape=(n,)) * self.noise_std
        )
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.float64),
        }


class DGPRegressionFixedDependentError(DGPSyntheticFixed):
    """
    My variation of Wu and Martin 2023, Section 5.2 with dependent error.
    """

    s_small: float
    s_mod: float

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int, s_small: float, s_mod: float):
        self.s_small = s_small
        self.s_mod = s_mod
        super().__init__(key, n, dim_x)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)

        # quantile of the first covariate
        x_lower = jnp.quantile(x_train[:, 0], 0.25, axis=0)
        x_upper = jnp.quantile(x_train[:, 0], 0.75, axis=0)

        std = jnp.where(
            x_train[:, 0] < x_lower,
            self.s_small,
            jnp.where(x_train[:, 0] < x_upper, self.s_mod, 1),
        )
        mean = x_train @ self.beta0
        y_train = mean + std * jax.random.normal(y_key, shape=(n,))
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.float64),
        }


class DGPRegressionFixedNonNormalError(DGPSyntheticFixed):
    """
    My variation of Wu and Martin 2023, Section 5.2 with non-Gaussian error.
    """

    df: int

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int, df: int):
        self.df = df
        super().__init__(key, n, dim_x)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        mean = x_train @ self.beta0
        y_train = mean + jax.random.t(y_key, df=self.df, shape=(n,))
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.float64),
        }


class DGPSynthetic(DGP):

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int):
        self.input_key = key
        key, prior_key, data_key = jax.random.split(key, 3)
        # Draw from a N(0, 1) prior
        self.beta0 = np.asarray(
            jax.random.normal(prior_key, shape=(dim_x,)), dtype=np.float64
        )
        self.dim_x = dim_x
        self.train_data = self.get_data(data_key, n)

    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        # We need this for forward recursion
        key, subkey = jax.random.split(key)
        return jax.random.uniform(subkey, shape=(n, self.dim_x), minval=-1, maxval=1)

    @abstractmethod
    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        pass

    def get_population(self) -> dict[str, np.ndarray]:
        population_data = self.get_data(jax.random.key(50), 100000)
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), population_data)


class DGPRegression(DGPSynthetic):
    """
    Draw the weights from a prior, then draw iid data from the linear Gaussian
    model as parameterised by the weights.
    """

    noise_std0: float

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int):
        super().__init__(key, n, dim_x)
        self.noise_std0 = math.sqrt(0.1)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        # We need this for test data
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        error = jax.random.normal(y_key, shape=(n,)) * self.noise_std0
        y_train = x_train @ self.beta0 + error
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.float64),
        }


class DGPClassification(DGPSynthetic):
    """
    Draw the weights from a prior, then draw iid data from the linear logistic
    regression model as parameterised by the weights.
    """

    def __init__(self, key: PRNGKeyArray, n: int, dim_x: int):
        super().__init__(key, n, dim_x)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        probs = jax.scipy.special.expit(x_train @ self.beta0)
        y_train = jax.random.bernoulli(y_key, probs)
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.int8),
        }


class DGPWuMartin(DGP):

    def __init__(self, key: PRNGKeyArray, n: int):
        self.input_key = key
        self.beta0 = jnp.array([1.0, 1.0, 2.0, -1.0])  # the truth in Wu and Martin 2023
        self.dim_x = self.beta0.shape[0]
        key, data_key = jax.random.split(key, 2)
        self.train_data = self.get_data(data_key, n)

    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        rho = 0.2  # Correlation parameter as specified
        key, x_key = jax.random.split(key)

        # Create correlation matrix with first-order autocorrelation structure
        # where corr(i,j) = rho^|i-j|
        indices = jnp.arange(self.dim_x)
        idx_diff = jnp.abs(indices[:, None] - indices[None, :])
        corr_matrix = rho**idx_diff

        # Generate x from multivariate normal with the correlation structure
        return jax.random.multivariate_normal(
            x_key, mean=jnp.zeros(self.dim_x), cov=corr_matrix, shape=(n,)
        )

    def get_population(self) -> dict[str, np.ndarray]:
        population_data = self.get_data(jax.random.key(50), 100000)
        return jax.tree.map(lambda x: np.asarray(x, dtype=np.float64), population_data)

    @abstractmethod
    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        pass


class DGPLinearRegressionWM(DGPWuMartin):
    """
    The example in Wu and Martin 2023, Section 5.1. Standard linear regression.
    """

    def __init__(self, key: PRNGKeyArray, n: int):
        super().__init__(key, n)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        y_train = x_train @ self.beta0 + jax.random.normal(y_key, shape=(n,))
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.float64),
        }


class DGPDependentErrorWM(DGPWuMartin):
    """
    The example in Wu and Martin 2023, Section 5.2 with dependent error.
    """

    s_small: float
    s_mod: float

    def __init__(self, key: PRNGKeyArray, n: int, s_small: float, s_mod: float):
        super().__init__(key, n)
        self.s_small = s_small
        self.s_mod = s_mod

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)

        # quantile of the first covariate
        x_05 = jnp.quantile(x_train[:, 0], 0.05, axis=0)
        x_95 = jnp.quantile(x_train[:, 0], 0.95, axis=0)

        std = jnp.where(
            x_train[:, 0] < x_05,
            self.s_small,
            jnp.where(x_train[:, 0] < x_95, self.s_mod, 1),
        )
        mean = x_train @ self.beta0
        y_train = mean + std * jax.random.normal(y_key, shape=(n,))
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.float64),
        }


class DGPNonNormalErrorWM(DGPWuMartin):
    """
    The example in Wu and Martin 2023, Section 5.2 with non-Gaussian error.
    """

    df: int

    def __init__(self, key: PRNGKeyArray, n: int, df: int):
        super().__init__(key, n)
        self.df = df

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        key, x_key, y_key = jax.random.split(key, 3)
        x_train = self.get_x_data(x_key, n)
        mean = x_train @ self.beta0
        y_train = mean + jax.random.t(y_key, df=self.df, shape=(n,))
        return {
            "x": np.asarray(x_train, dtype=np.float64),
            "y": np.asarray(y_train, dtype=np.float64),
        }


OPENML_REGRESSION = [
    "abalone",
    "airfoil",
    "kin8nm",
    "auction",
    "concrete",
    "energy",
    "grid",
    "fish",
    "quake",
]

OPENML_BINARY_CLASSIFICATION = [
    "blood",
    "phoneme",
    "banknote",
    "mozilla",
    "skin",
    "telescope",
    "sepsis",
    "rice",
]

OPENML_CLASSIFICATION = [
    "yeast",
    "wine",
]


def load_dgp(cfg, data_key: PRNGKeyArray) -> DGP:
    # Initialize a classifier
    if cfg.dgp.name == "regression":
        dgp = DGPRegression(data_key, cfg.data_size, cfg.dgp.dim_x)
    elif cfg.dgp.name == "classification":
        dgp = DGPClassification(data_key, cfg.data_size, cfg.dgp.dim_x)
    elif cfg.dgp.name == "classification-fixed":
        dgp = DGPClassificationFixed(data_key, cfg.data_size, cfg.dgp.dim_x)
    elif cfg.dgp.name == "classification-fixed-gmm":
        dgp = DGPClassificationFixedGMMLink(
            data_key, cfg.data_size, cfg.dgp.dim_x, cfg.dgp.a
        )
    elif cfg.dgp.name == "regression-fixed":
        dgp = DGPRegressionFixed(
            data_key, cfg.data_size, cfg.dgp.dim_x, cfg.dgp.noise_std
        )
    elif cfg.dgp.name == "regression-fixed-dependent":
        dgp = DGPRegressionFixedDependentError(
            data_key,
            cfg.data_size,
            cfg.dgp.dim_x,
            s_small=cfg.dgp.s_small,
            s_mod=cfg.dgp.s_mod,
        )
    elif cfg.dgp.name == "regression-fixed-non-normal":
        dgp = DGPRegressionFixedNonNormalError(
            data_key, cfg.data_size, cfg.dgp.dim_x, df=cfg.dgp.df
        )
    elif cfg.dgp.name == "regression-wm":
        dgp = DGPLinearRegressionWM(data_key, cfg.data_size)
    elif cfg.dgp.name == "dependent-error-wm":
        dgp = DGPDependentErrorWM(
            data_key,
            cfg.data_size,
            s_small=cfg.dgp.s_small,
            s_mod=cfg.dgp.s_mod,
        )
    elif cfg.dgp.name == "non-normal-wm":
        dgp = DGPNonNormalErrorWM(data_key, cfg.data_size, df=cfg.dgp.df)
    elif cfg.dgp.name == "quake":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 550, -1, 2)
    elif cfg.dgp.name == "airfoil":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44957, -1, 1)
    elif cfg.dgp.name == "kin8nm":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44980, -1, 1)
    elif cfg.dgp.name == "concrete":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44959, -1, 1)
    elif cfg.dgp.name == "energy":
        if cfg.data_size > 50:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44960, -1, 1)
        else:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44960, -1, 1, None, 2)
    elif cfg.dgp.name == "grid":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44973, -1, 1)
    elif cfg.dgp.name == "abalone":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 45042, -1, 1)
    elif cfg.dgp.name == "fish":
        if cfg.data_size > 50:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44970, -1, 2)
        else:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44970, -1, 1)
    elif cfg.dgp.name == "auction":
        if cfg.data_size > 50:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44958, -1, 1)
        else:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44958, -1, 1, None, 1)
    elif cfg.dgp.name == "blood":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1464, -1, 1)
    elif cfg.dgp.name == "phoneme":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1489, -1, 1)
    elif cfg.dgp.name == "skin":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1502, -1, 3)
    elif cfg.dgp.name == "rice":
        if cfg.data_size > 50:
            dgp = DGPClassificationUCI(data_key, cfg.data_size, 545, [False] * 7, 2)
        else:
            dgp = DGPClassificationUCI(data_key, cfg.data_size, 545, [False] * 7, 1)
    elif cfg.dgp.name == "mozilla":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1046, -1, 2)
    elif cfg.dgp.name == "telescope":
        dgp = DGPClassificationUCI(data_key, cfg.data_size, 159, [False] * 10, 1)
    elif cfg.dgp.name == "sepsis":
        dgp = DGPClassificationUCI(
            data_key, cfg.data_size, 827, [False, True, False], 2
        )
    elif cfg.dgp.name == "yeast":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 181, -1, 1)
    elif cfg.dgp.name == "wine":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 40498, -1, 1)
    elif cfg.dgp.name == "banknote":
        if cfg.data_size > 50:
            dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1462, -1, 2)
        else:
            dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1462, -1, 1)
    elif cfg.dgp.name == "car":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 991, -1, 1)
    elif cfg.dgp.name == "credit":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 31, -1)
    elif cfg.dgp.name == "boston":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 531, -1)
    elif cfg.dgp.name == "cars":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44994, 0)
    elif cfg.dgp.name == "cpu":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44978, -1)
    elif cfg.dgp.name == "moneyball":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 41021, 3)
    elif cfg.dgp.name == "colleges":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 42727, 27)
    elif cfg.dgp.name == "pumadyn":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44981, -1)
    elif cfg.dgp.name == "ada":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 41156, 0)
    elif cfg.dgp.name == "australian":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 40981, -1)
    elif cfg.dgp.name == "churn":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 40701, -1)
    elif cfg.dgp.name == "cmc":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 23, -1)
    elif cfg.dgp.name == "eucalyptus":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 188, -1)
    elif cfg.dgp.name == "theorem":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1475, -1)
    elif cfg.dgp.name == "lidar":
        dgp = DGPLidar(data_key, cfg.data_size)
    elif cfg.dgp.name == "lidar-full":
        dgp = DGPLidar(data_key, -1)
    else:
        raise NotImplementedError(f"DGP {cfg.dgp.name} not implemented")
    return dgp

