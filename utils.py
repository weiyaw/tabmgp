import logging
import os
import pickle
import re
import subprocess

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


def get_n_data(data):
    # return the number of samples in 'data' which is a dictionary of arrays
    # with the same leading dimension

    leading_dims = [x.shape[0] for x in data.values()]
    assert all([x == leading_dims[0] for x in leading_dims])
    return leading_dims[0]


def get_tree_lead_dim(tree):
    # Return the leading dimensions of a PyTree, assuming that all leaves have
    # the same leading dimension

    leaves = jax.tree.leaves(tree)
    chex.assert_equal_shape_prefix(leaves, 1)
    return leaves[0].shape[0]


def tree_shape(tree):
    return jax.tree.map(lambda x: jnp.shape(x), tree)


def githash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def write_to_local(path, obj, verbose=False):
    # write to local
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        logging.info(f"Write to local:/{path}")


def write_to(path, obj, verbose=False):
    write_to_local(path, obj, verbose=verbose)


def read_from_gs(bucket_name, path):
    # bucket: bucket name
    # path: path to the file
    # obj: the object to save

    bucket = get_bucket(bucket_name)
    blob = bucket.blob(path)
    with blob.open("rb") as f:
        obj = pickle.load(f)
    return obj


def read_from_local(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_from(path):
    if path.startswith("gs://"):
        # read from gcs
        bucket_name, path = path.split("/", 3)[2:]
        return read_from_gs(bucket_name, path)
    else:
        # read from local
        return read_from_local(path)


def print_dgp(d):
    return " ".join([f"{k}={v}" for k, v in d.items()])


def get_data_size(path):
    match = re.search(r"data=([^ ]+)", path)
    if match:
        return match.group(1)
    return None


def get_resample_x(path):
    match = re.search(r"resample_x=([^ ]+)", path)
    if match:
        return match.group(1)
    return None


def get_seed(path):
    match = re.search(r"seed=([^ ]+)", path)
    if match:
        return int(match.group(1))
    return None


def get_dim_x(path):
    match = re.search(r"dim_x=([^ ]+)", path)
    if match:
        return int(match.group(1))
    return None


def get_date_part(path):
    match = re.search(r"outputs/([^/]+)/", path)
    if match:
        return match.group(1)
    return None


def format_decimal(x, decimals=2):
    if x:
        return f"{x:.{decimals}f}"
    return None


def get_data_name(path):
    match = re.search(r"name=([^ ]+)", path)
    if match:
        if match.group(1) == "classification-fixed":
            return "classification-standard"
        elif match.group(1) == "classification-fixed-gmm":
            match2 = re.search(r"a=([^ ]+)", path)
            return f"classification-gmm-{match2.group(1)}"
        elif match.group(1) == "regression-fixed-dependent":
            match2 = re.search(r"s_small=([^ ]+)", path)
            match3 = re.search(r"s_mod=([^ ]+)", path)
            return f"regression-dependent-{match2.group(1)}-{match3.group(1)}"
        elif match.group(1) == "regression-fixed":
            return "regression-standard"
        elif match.group(1) == "regression-fixed-non-normal":
            match2 = re.search(r"df=([^ ]+)", path)
            return f"regression-t-{match2.group(1)}"
        return match.group(1)
    return None
