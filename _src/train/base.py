"""Functions related to training."""

import jax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx
import optax

from typing import Tuple, Optional, NamedTuple
from jaxtyping import Array, PRNGKeyArray, PyTree

from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad

from utils import get_dtype, get_accuracy
from utils import tree_utils
from dataloaders import DataBatch
from loggers import LogState
from losses import ObjectiveFn


class TrainState(NamedTuple):
    """The train state class that wraps all necessary states for training.
    
    The train state is defined to be easily serializable for checkpointing.
    
    Right now, I'm thinking we should use a consistent structure of train state
    for all training tasks, including lm, cv, and others. 
    It consists of the follwing components:
    
    Attr:
        model: an equinox model
        opt_state: optax optimizer state
        log_state: extra states for the logger
        dynamic_scaler_state: aka amp_state, extra state for amp
        epoch: current epoch number
        iteration: current iteration number
        train_key: the root seed for all randomness of training
        num_nans: number of consecutive iterations where loss = nan, used for auto-terminating
    """
    model: eqx.Module
    opt_state: optax.OptState
    log_state: LogState
    dynamic_scaler_state: Optional[DynamicScalerState]
    epoch: Array
    iteration: Array
    train_key: PRNGKeyArray
    num_nans: int


def forward_prop(
        *batches: DataBatch,
        train_state: TrainState,
        loss_fn: ObjectiveFn,
        use_amp: bool,
        amp_precision: str,
) -> Tuple[Array, Array, TrainState]:
    """The forward propagation: computes mini-batch average of loss and accuracy.
    
    Returns:
        A tuple of (loss, accuracy, train_state).
    """
    if use_amp:
        amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(amp_precision))
    else:
        amp_loss_fn = loss_fn

    model = train_state.model
    train_key = train_state.train_key
    num_batches = len(batches)

    current_key, new_key = jr.split(train_key)
    keys = jr.split(current_key, num_batches)

    # Use jax.lax.fori_loop to aggregate forward_prop
    batches = jnp.array(batches)
    keys = jnp.array(keys)

    def forward_prop_single_batch(i, val):
        loss, accuracy = val
        batch, key = batches[i], keys[i]
        loss_, logits_ = amp_loss_fn(model, batch, key=key)

        loss += loss_
        accuracy += get_accuracy(logits_, batch)
        return (loss, accuracy)
    
    init_val = (0.0, 0.0)
    loss, accuracy = jax.lax.fori_loop(
        0, num_batches, forward_prop_single_batch, init_val
    )
    loss /= num_batches
    accuracy /= num_batches

    # Only need to update train_key.
    train_state = train_state._replace(
        train_key=new_key,
    )

    return loss, accuracy, train_state


def back_prop(
        *batches: DataBatch,
        train_state: TrainState,
        loss_fn: ObjectiveFn,
        use_amp: bool,
        amp_precision: str,
) -> Tuple[Array, Array, PyTree, TrainState]:
    """The backward propagation: computes mini-batch average of loss, accuracy, and grads.

    Only modifies the amp_state and train_key in the train_state.

    Args:
        batches: either a single data batch or a list of batches.
        train_state: the train state container.
        loss_fn: a mapping from (model, single_batch, key) to (loss, logits).
        use_amp: if true, turns on auto mixed precision.
        amp_precision: specifies the precision of amp.
    
    Returns:
        A tuple of (loss, accuracy, grads, train_state).
    """
    if use_amp:
        amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(amp_precision))
        value_and_grad_fn = dynamic_scale_value_and_grad(
            amp_loss_fn, filter=True, has_aux=True, redo_on_nan=0
        )
    else:
        value_and_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    model = train_state.model
    dynamic_scaler_state = train_state.dynamic_scaler_state
    train_key = train_state.train_key
    num_batches = len(batches)

    current_key, new_key = jr.split(train_key)
    keys = jr.split(current_key, num_batches)

    # Wrap with jax.lax.fori_loop.
    batches = jnp.array(batches)
    keys = jnp.array(keys)

    def back_prop_single_batch(i, val):
        loss, accuracy, grads, dynamic_scaler_state = val
        batch, key = batches[i], keys[i]
        if use_amp:
            dynamic_scaler_state, ((loss_, logits_), grads_) = value_and_grad_fn(
                model, batch, key=key, dynamic_scaler_state=dynamic_scaler_state
            )
        else:
            (loss_, logits_), grads_ = value_and_grad_fn(model, batch, key=key)
        loss += loss_
        accuracy += get_accuracy(logits_, batch)
        grads = tree_utils.add(grads, grads_)
        return (loss, accuracy, grads, dynamic_scaler_state)
    
    loss = 0.0
    accuracy = 0.0
    grads = tree_utils.zeros_like(eqx.filter(model, eqx.is_array))
    init_val = (loss, accuracy, grads, dynamic_scaler_state)
    loss, accuracy, grads, dynamic_scaler_state = jax.lax.fori_loop(
        0, num_batches, back_prop_single_batch, init_val
    )
    loss /= num_batches
    accuracy /= num_batches
    grads = tree_utils.scalar_dot(grads, 1/num_batches)

    # Only update amp_state and train_key.
    train_state = train_state._replace(
        dynamic_scaler_state=dynamic_scaler_state,
        train_key=new_key,
    )

    return loss, accuracy, grads, train_state