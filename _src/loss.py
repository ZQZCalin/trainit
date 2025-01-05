"""Loss functions."""

import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Tuple, Callable
from jaxtyping import Array, PRNGKeyArray
from omegaconf import DictConfig

from losses import LossFn
from losses import softmax_cross_entropy
from _src.base import classification_tasks
from _src.base import ObjectiveFn


def loss_to_objective(
        loss_fn: LossFn,
    ):
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


def init_loss_fn(
        config: DictConfig
) -> ObjectiveFn:
    """Initialize the optimization objective function.
    
    Args:
        config: global_config.
    
    Returns:
        An `ObjectiveFn` object.
    """
    # NOTE: For now, loss function should be only dependent on datasets.
    dataset_name = config.dataset.name
    if dataset_name in classification_tasks:
        return loss_to_objective(softmax_cross_entropy)
    else:
        raise ValueError("invalid config: cannot initialize loss function", 
                         f"because of unknown dataset '{dataset_name}'.")