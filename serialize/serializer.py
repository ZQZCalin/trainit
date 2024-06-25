import jax
import equinox as eqx
from typing import NamedTuple, Optional
import numpy as np
from jax import tree_util as jtu
from jaxtyping import PyTree
import dill
from jax import numpy as jnp


class ArrayShape(NamedTuple):
    shape: np.ndarray
    dtype: np.dtype


def get_structure(pytree):
    def get_type(p):
        if eqx.is_array(p):
            return ArrayShape(shape=p.shape, dtype=p.dtype)
        else:
            return p

    return jtu.tree_map(
        get_type,
        pytree,
    )


def save(path: str, pytree: PyTree, save_structure: bool = True) -> None:
    with open(path, "wb") as f:
        if save_structure:
            dill.dump(get_structure(pytree), f)
        else:
            # just dump a "None" flag into the file.
            # this can be used to check if the the structure was saved
            # if necessary.
            dill.dump(None, f)
        eqx.tree_serialise_leaves(f, pytree)


def load(path: str, structure: Optional[PyTree] = None) -> PyTree:
    def create_arrays(p):
        if isinstance(p, ArrayShape):
            return jnp.zeros(p.shape, dtype=p.dtype)
        else:
            return p

    with open(path, "rb") as f:
        if structure is None:
            structure = jtu.tree_map(
                create_arrays,
                dill.load(f),
                is_leaf=lambda p: isinstance(p, ArrayShape),
            )
        else:
            # read out and drop the structure
            _ = dill.load(f)
        result = eqx.tree_deserialise_leaves(f, structure)

    return result
