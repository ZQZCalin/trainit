# An optax implementation of SOAP optimizer from:
# 
# https://arxiv.org/pdf/2409.11321
# 
# We refer to the official pytorch implementation
# 
# https://github.com/nikhilvyas/SOAP/tree/main
# 
# and the unofficial jax implementation
# 
# https://github.com/haydn-jones/SOAP_JAX.

"""SOAP optimizer."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple
from jaxtyping import Array, PyTree

from utils import tree_utils


class ScaleBySoapState(NamedTuple):
    """scale_by_soap state."""


def scale_by_soap() -> optax.GradientTransformation:
    """The SOAP optimizer."""
    def init_fn(params):
        return ScaleBySoapState()

    def update_fn(updates, state, params=None):
        del params
        return updates, ScaleBySoapState()

    return optax.GradientTransformation(init_fn, update_fn)