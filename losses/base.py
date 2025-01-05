"""Base classes and loss functions."""

import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Tuple, Callable
from jaxtyping import Array, PRNGKeyArray


# A mapping from (input, target) to a loss value.
LossFn = Callable[[Array, Array], Array]
# A mapping from (model, data_batch, key) to a tuple of (loss, logits)
ObjectiveFn = Callable[[eqx.Module, Tuple[Array, Array], PRNGKeyArray], Tuple[Array, Array]]


def loss_to_objective(
        loss_fn: LossFn,
    ) -> ObjectiveFn:
    """Wraps a standard loss function, which maps tuples of (input ,target) into loss values,
    into an optimization objective function that maps (parameters, data batches, random key) 
    into loss values.

    Applies jax.vmap to all data in a data batch.

    Args:
        loss_fn: standard loss function that maps (input, target) to loss values.
        model: equinox module
        batch: data batch of form (feature, target).
        key: random key used for model forward. 
            This will be dropped if model forward is deterministic (e.g., no dropout).

    Returns:
        An objective function.

    The objective function has the following structure.

    Args:
        model: equinox module
        batch: data batch of form (feature, target).
        key: random key used for model forward. 
            This will be dropped if model forward is deterministic (e.g., no dropout).
    Returns:
        Loss value and logits (model outputs).
    """

    def objective(
        model: eqx.Module, 
        batch: Tuple[Array, Array],
        *
        key: PRNGKeyArray,
    ) -> Tuple[Array, Array]:
        def single_example_loss_fn(input, target):
            logits = model(input, key=key)
            loss = loss_fn(logits, target)
            return loss, logits

        vmapped_loss_fn = jax.vmap(single_example_loss_fn, in_axes=(0, 0), out_axes=(0, 0))
        input, target = batch
        loss, logits = vmapped_loss_fn(input, target)

        return jnp.mean(loss), logits
    
    return objective