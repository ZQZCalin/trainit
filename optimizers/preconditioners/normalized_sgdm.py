# An optax implementation of SGDM normalized layerwise.

"""Muon optimizer."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Optional, Literal
from jaxtyping import Array, PyTree

from utils import tree_utils
from optimizers.schedule import get_current_lr


class NormalizedSGDMState(NamedTuple):
    """scale_by_muon state."""
    count: Array
    momentum: optax.Updates


def normalized_sgdm(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        normalize: Literal["layer", "global", None] = "layer",
) -> optax.GradientTransformation:
    """The normalized SGDM optimizer.
    
    Applies normalization using Frobenius norm.
    By default, normalize matrix in each layer.

    Args:
        learning_rate: learning rate.
        momentum: momentum constant.
        nesterov: whether to use nesterov momentum.
        normalize: "layer" applies layerwise normalization,
            "global" applies l2 normalization of entire tree,
            None disables normalization.
    """
    
    def init_fn(params):
        return NormalizedSGDMState(
            count = jnp.zeros([], dtype=jnp.int32),
            momentum = tree_utils.zeros_like(params),
        )
    
    def update_fn(updates, state, params=None):
        del params
        count = state.count
        state_momentum = state.momentum

        # Update momentum.
        state_momentum = jtu.tree_map(
            lambda mu, g: momentum * mu + g, state_momentum, updates)

        # Normalize momentum matrix.
        if nesterov:
            # Apply nesterov's momentum before applying normalization.
            updates = jtu.tree_map(
                lambda mu, g: momentum * mu + g, state_momentum, updates)
        else:
            updates = state_momentum
        if normalize == "layer":
            updates = jtu.tree_map(
                lambda G: G / jnp.linalg.norm(G), updates)
        elif normalize == "global":
            updates = tree_utils.normalize(updates, p=2)
        elif normalize is None:
            updates = updates

        # Wrap final update.
        lr = get_current_lr(learning_rate, count)
        updates = tree_utils.scalar_dot(updates, -lr)

        return updates, NormalizedSGDMState(
            count = optax.safe_int32_increment(count),
            momentum = state_momentum
        )
    
    return optax.GradientTransformation(init_fn, update_fn)