from abc import abstractmethod
import chex
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


def make_x_encoder(
    categorical_x: list[bool], include_x: list[bool]
) -> ColumnTransformer:
    """
    One-hot encode categorical features and standardize numerical features.  We
    are going to fit an intercept, so drop the first category of each
    categorical feature to avoid multicollinearity.

    All the columns with False in include_x will be dropped.
    """

    numerical_x = [(not i) and j for i, j in zip(categorical_x, include_x)]
    categorical_x = [i and j for i, j in zip(categorical_x, include_x)]
    return make_column_transformer(
        (OneHotEncoder(drop="first", sparse_output=False), categorical_x),
        (StandardScaler(), numerical_x),
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


class Preprocessor(eqx.Module):
    @abstractmethod
    def encode_data(self, data: dict[str, ArrayLike]) -> dict[str, Array]:
        """
        Encode the data (both x and y) into suitable format for the model.
        Args:
            data: A dict with keys "x" and "y", each with an array-like value.
        Returns:
            A dict with keys "x" and "y", each with a jax Array value.
        """
        pass


class DiscreteTarget(Preprocessor):
    include_x: list[bool]
    x_encoder: ColumnTransformer
    y_encoder: LabelEncoder

    def __init__(
        self,
        categorical_x: list[bool],
        include_x: list[bool],
        population_data: dict[str, ArrayLike],
    ):

        assert len(include_x) == len(categorical_x)
        self.include_x = include_x

        # population_data = dgp.get_population()
        chex.assert_shape(population_data["x"], (None, len(categorical_x)))
        chex.assert_rank(population_data["y"], 1)

        self.x_encoder = make_x_encoder(categorical_x, include_x).fit(
            population_data["x"]
        )
        self.y_encoder = LabelEncoder().fit(population_data["y"])

    def encode_data(self, data: dict[str, ArrayLike]) -> dict[str, Array]:
        batch_shape = data["x"].shape[:-1]
        chex.assert_shape(data["x"], (*batch_shape, self.x_encoder.n_features_in_))
        chex.assert_shape(data["y"], batch_shape)
        flat_x = jnp.reshape(data["x"], (-1, self.x_encoder.n_features_in_))
        flat_y = jnp.reshape(data["y"], (-1,))
        x = jnp.reshape(self.x_encoder.transform(flat_x), (*batch_shape, -1))
        y = jnp.reshape(self.y_encoder.transform(flat_y), batch_shape)
        return {
            "x": jnp.asarray(x, dtype=jnp.float64),
            "y": jnp.asarray(y, dtype=jnp.int16),
        }


class ContinuousTarget(Preprocessor):
    include_x: list[bool]
    x_encoder: ColumnTransformer
    y_encoder: StandardScaler

    def __init__(
        self,
        categorical_x: list[bool],
        include_x: list[bool],
        population_data: dict[str, ArrayLike],
    ):

        assert len(include_x) == len(categorical_x)
        self.include_x = include_x

        # population_data = dgp.get_population()
        chex.assert_shape(population_data["x"], (None, len(categorical_x)))
        chex.assert_rank(population_data["y"], 1)

        self.x_encoder = make_x_encoder(categorical_x, include_x).fit(
            population_data["x"]
        )
        self.y_encoder = StandardScaler().fit(population_data["y"][..., None])

    def encode_data(self, data: dict[str, ArrayLike]) -> dict[str, Array]:
        batch_shape = data["x"].shape[:-1]
        chex.assert_shape(data["x"], (*batch_shape, self.x_encoder.n_features_in_))
        chex.assert_shape(data["y"], batch_shape)
        flat_x = jnp.reshape(data["x"], (-1, self.x_encoder.n_features_in_))
        flat_y = jnp.reshape(data["y"], (-1, 1))
        x = jnp.reshape(self.x_encoder.transform(flat_x), (*batch_shape, -1))
        y = jnp.reshape(self.y_encoder.transform(flat_y), batch_shape)
        return {
            "x": jnp.asarray(x, dtype=jnp.float64),
            "y": jnp.asarray(y, dtype=jnp.float64),
        }
